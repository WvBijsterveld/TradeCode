import argparse
import atexit
import csv
import glob
import os
import time
import math
import errno
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch as th
import torch.nn as nn
import zmq
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv


# --- CONFIG ---
DEFAULT_PORT = 5555

DEFAULT_CHECKPOINT_DIR = "./ai_checkpoints_v31"
DEFAULT_STUDY_NAME = "prop_crusher_v31_asia_inverse_fvg"
DEFAULT_STORAGE_DB = "sqlite:///v31_optuna.db"

WINDOW_SIZE = 30
BASE_FEATURES = 15
EXTRA_TREND_SR_FEATURES = 4
EXTRA_IFVG_ZONE_FEATURES = 3

# Defaults (overridden by --pair)
PIP_VALUE = 0.01
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.2

ASIA_START_HOUR = 0
ASIA_END_HOUR = 8


def _quantize_pips_0p1(v: float) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if not np.isfinite(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    x = max(x, 0.1)
    return float(round(x * 10.0) / 10.0)


def _rotate_log_if_schema_changed(log_path: str, expected_fieldnames: list[str]) -> None:
    if not os.path.exists(log_path):
        return

    try:
        with open(log_path, "r", newline="") as f:
            first_line = f.readline().strip("\r\n")
        expected = ",".join(expected_fieldnames)
        if first_line == expected:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        rotated = f"{log_path}.bak_{ts}"
        os.rename(log_path, rotated)
        print(f"🗂️  Rotated log due to schema change: {rotated}")
    except Exception:
        # If anything goes wrong, keep appending to avoid breaking the bridge.
        return


def _rotate_log_if_too_big(log_path: str, *, max_mb: float) -> None:
    if max_mb is None:
        return
    try:
        max_mb_f = float(max_mb)
    except Exception:
        return
    if max_mb_f <= 0.0:
        return
    if not os.path.exists(log_path):
        return
    try:
        size_b = int(os.path.getsize(log_path))
        max_b = int(max_mb_f * 1024 * 1024)
        if size_b <= max_b:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        rotated = f"{log_path}.bak_size_{ts}"
        os.rename(log_path, rotated)
        print(f"🗂️  Rotated log due to size ({size_b / (1024 * 1024):.1f}MB > {max_mb_f:.1f}MB): {rotated}")
    except Exception:
        return


@dataclass
class _CsvSink:
    enabled: bool
    file: Optional[object] = None
    writer: Optional[csv.DictWriter] = None
    label: str = "log"
    _warned: bool = False

    # Buffering to avoid per-bar disk I/O bottlenecks in backtests.
    buffer_size: int = 1000
    flush_interval_s: float = 5.0
    _buffer: list = field(default_factory=list)
    _last_flush_ts: float = 0.0

    def __post_init__(self) -> None:
        self._last_flush_ts = time.time()

    def _disable_due_to_disk_full(self) -> None:
        self.enabled = False
        if not self._warned:
            self._warned = True
            print(f"⚠️  {self.label} disabled: no space left on device. Continuing without logging.")
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass
        self.file = None
        self.writer = None
        try:
            self._buffer.clear()
        except Exception:
            pass

    def _flush_buffer(self, *, force: bool) -> None:
        if not self.enabled or self.writer is None or self.file is None:
            return
        if not self._buffer:
            return
        now = time.time()
        if not force:
            try:
                # Flush periodically for live safety, even if buffer isn't full.
                if float(self.flush_interval_s) > 0.0 and (now - float(self._last_flush_ts)) < float(self.flush_interval_s):
                    return
            except Exception:
                pass
        try:
            self.writer.writerows(self._buffer)
            self.file.flush()
            self._buffer.clear()
            self._last_flush_ts = now
        except OSError as e:
            if getattr(e, "errno", None) in (errno.ENOSPC, 28):
                self._disable_due_to_disk_full()
                return
            raise

    def close(self) -> None:
        # Best-effort final flush.
        try:
            self._flush_buffer(force=True)
        except Exception:
            pass
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass
        self.file = None
        self.writer = None

    def write_row(self, row: dict) -> None:
        if not self.enabled or self.writer is None or self.file is None:
            return
        try:
            self._buffer.append(row)
            # Flush when buffer is full.
            if int(self.buffer_size) > 0 and len(self._buffer) >= int(self.buffer_size):
                self._flush_buffer(force=True)
            else:
                self._flush_buffer(force=False)
        except OSError as e:
            # If disk fills up mid-run, keep the bridge trading (fail-closed on logging).
            if getattr(e, "errno", None) in (errno.ENOSPC, 28):
                self._disable_due_to_disk_full()
                return
            raise
        except Exception:
            raise


class SupervisedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        mods = []
        d = int(input_dim)
        for _ in range(int(layers)):
            mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        mods += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*mods)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)
    
# --- V3 ARCHITECTURE BLOCK ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SupervisedTransformer(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        if seq_len > 0:
            self.feat_dim = input_dim // seq_len
        else:
            self.feat_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(self.feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        if self.seq_len > 0:
            x = x.view(x.size(0), self.seq_len, self.feat_dim)
        x = x.permute(1, 0, 2)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        last_step = output[-1, :, :]
        return self.decoder(last_step)


class SupervisedLSTM(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        if seq_len > 0:
            self.feat_dim = input_dim // seq_len
        else:
            self.feat_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if self.seq_len > 0:
            x = x.view(x.size(0), self.seq_len, self.feat_dim)
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        return self.head(last_hidden)


def _supervised_input_dim(model: Optional[SupervisedMLP]) -> Optional[int]:
    try:
        if model is None:
            return None
        first = model.net[0]
        return int(getattr(first, "in_features"))
    except Exception:
        return None


@dataclass
class _AdaptiveQualityGate:
    enabled: bool
    window_trades: int
    loss_streak_trigger: int
    bump_per_loss: float
    bump_max: float
    cooldown_bars: int
    reset_after_bars: int = 0

    recent_pnls: deque = None
    loss_streak: int = 0
    cooldown_until_bar: int = -1
    last_trade_close_bar: int = -1

    def __post_init__(self) -> None:
        if self.recent_pnls is None:
            self.recent_pnls = deque(maxlen=max(int(self.window_trades), 1))

    def _maybe_reset_due_to_inactivity(self, *, bar_index: int) -> None:
        if not self.enabled:
            return
        try:
            reset_bars = int(self.reset_after_bars)
        except Exception:
            reset_bars = 0
        if reset_bars <= 0:
            return
        if int(self.last_trade_close_bar) < 0:
            return
        try:
            inactive = int(bar_index) - int(self.last_trade_close_bar)
        except Exception:
            return
        if inactive >= reset_bars:
            self.loss_streak = 0
            self.cooldown_until_bar = -1

    def on_trade_closed(self, pnl: float, *, bar_index: int) -> None:
        if not self.enabled:
            return
        try:
            pnl_f = float(pnl)
        except Exception:
            return
        if not np.isfinite(pnl_f):
            return
        try:
            self.last_trade_close_bar = int(bar_index)
        except Exception:
            self.last_trade_close_bar = -1
        self.recent_pnls.append(pnl_f)
        if pnl_f < 0.0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        if int(self.cooldown_bars) > 0 and int(self.loss_streak_trigger) > 0:
            if int(self.loss_streak) >= int(self.loss_streak_trigger):
                self.cooldown_until_bar = max(int(self.cooldown_until_bar), int(bar_index) + int(self.cooldown_bars))

    def in_cooldown(self, *, bar_index: int) -> bool:
        if not self.enabled:
            return False
        self._maybe_reset_due_to_inactivity(bar_index=int(bar_index))
        return int(bar_index) <= int(self.cooldown_until_bar)

    def bump(self, *, bar_index: int) -> float:
        if not self.enabled:
            return 0.0
        self._maybe_reset_due_to_inactivity(bar_index=int(bar_index))
        trig = int(self.loss_streak_trigger)
        if trig <= 0:
            return 0.0
        extra_losses = max(0, int(self.loss_streak) - trig + 1)
        bump = float(extra_losses) * float(self.bump_per_loss)
        return float(min(float(self.bump_max), max(0.0, bump)))

    def recent_pf(self) -> float:
        if not self.enabled or not self.recent_pnls:
            return float("nan")
        wins = 0.0
        losses = 0.0
        for p in self.recent_pnls:
            if p > 0.0:
                wins += float(p)
            elif p < 0.0:
                losses += -float(p)
        if losses <= 0.0:
            return float("inf") if wins > 0.0 else float("nan")
        return float(wins / losses)


def _load_supervised_model(path: str, model_type: str = "mlp", window_size: int = WINDOW_SIZE) -> tuple[SupervisedMLP, dict]:
    # Load checkpoint and construct either an MLP or the supervised transformer.
    ckpt = th.load(path, map_location="cpu")
    input_dim = int(ckpt.get("input_dim"))
    # Prefer explicit model_type passed by caller; fallback to checkpoint meta when available.
    if model_type is None:
        model_type = ckpt.get("meta", {}).get("model_type", "mlp") if isinstance(ckpt.get("meta"), dict) else "mlp"

    sd = ckpt.get("state_dict") if "state_dict" in ckpt else ckpt
    keys = list(sd.keys()) if isinstance(sd, dict) else []

    def _looks_like_mlp(keys_list: list[str]) -> bool:
        return any(k.startswith("net.") or k.startswith("net") for k in keys_list)

    def _looks_like_transformer(keys_list: list[str]) -> bool:
        return any(("embedding" in k) or ("transformer_encoder" in k) or ("pos_encoder" in k) for k in keys_list)

    def _looks_like_lstm(keys_list: list[str]) -> bool:
        return any(("lstm" in k) or ("head." in k) for k in keys_list)

    # Decide which architecture to instantiate. Prefer explicit model_type, but auto-detect on mismatch.
    requested_is_transformer = "transformer" in str(model_type).lower()
    requested_is_lstm = "lstm" in str(model_type).lower()

    # Try to pick a sensible model based on checkpoint contents.
    chosen_transformer = False
    chosen_lstm = False
    if _looks_like_mlp(keys):
        chosen_transformer = False
        chosen_lstm = False
    elif _looks_like_transformer(keys):
        chosen_transformer = True
    elif _looks_like_lstm(keys):
        chosen_lstm = True
    else:
        # Fallback to requested type (transformer/lstm/mlp)
        chosen_transformer = requested_is_transformer
        chosen_lstm = requested_is_lstm

    # Warn when user requested transformer but checkpoint doesn't look like one
    if requested_is_transformer and not chosen_transformer:
        print(f"⚠️  Requested transformer but checkpoint doesn't look like one; loading best guess: {path}")
    if requested_is_lstm and not chosen_lstm:
        print(f"⚠️  Requested lstm but checkpoint doesn't look like one; loading best guess: {path}")

    if chosen_transformer:
        print(f"Loading V3 Transformer: {path}")
        model = SupervisedTransformer(
            input_dim=input_dim,
            seq_len=int(window_size),
            d_model=128,
            nhead=4,
            num_layers=2,
        )
    elif chosen_lstm:
        print(f"Loading V4 LSTM: {path}")
        model = SupervisedLSTM(
            input_dim=input_dim,
            seq_len=int(window_size),
            hidden_dim=128,
            num_layers=2,
        )
    else:
        model = SupervisedMLP(input_dim=input_dim)

    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        # Provide a helpful error that prompts the user to check model_type/path pairing.
        raise RuntimeError(f"Failed loading checkpoint {path} into {'Transformer' if chosen_transformer else 'MLP'}: {e!r}")

    model.eval()
    meta = ckpt.get("meta", {}) or {}
    return model, meta


def _safe_mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return -1.0


def _extract_tuned_threshold(meta: dict) -> Optional[float]:
    try:
        if not isinstance(meta, dict):
            return None
        t = meta.get("tuned_threshold", None)
        if t is None:
            return None
        t = float(t)
        if 0.0 < t < 1.0:
            return t
        return None
    except Exception:
        return None


def _extract_tuned_tp_margin_pips(meta: dict) -> Optional[float]:
    try:
        if not isinstance(meta, dict):
            return None
        m = meta.get("tuned_tp_margin_pips", None)
        if m is None:
            return None
        m = _quantize_pips_0p1(float(m))
        return m if m > 0.0 else None
    except Exception:
        return None


def _resolve_sup_thresholds(*, args, sup_meta: dict) -> tuple[float, float]:
    long_overridden = args.sup_thresh_long is not None
    short_overridden = args.sup_thresh_short is not None

    base_long = float(args.sup_thresh_long) if long_overridden else float(args.sup_threshold)
    base_short = float(args.sup_thresh_short) if short_overridden else float(args.sup_threshold)
    if not bool(getattr(args, "sup_auto_threshold", False)):
        return float(base_long), float(base_short)

    # If --sup-auto-threshold is enabled, use tuned thresholds from the model meta,
    # but only when the user did NOT explicitly override the threshold via CLI.
    t_long = _extract_tuned_threshold((sup_meta or {}).get("long", {}) if isinstance(sup_meta, dict) else {})
    t_short = _extract_tuned_threshold((sup_meta or {}).get("short", {}) if isinstance(sup_meta, dict) else {})
    if (not long_overridden) and (t_long is not None):
        base_long = float(t_long)
    if (not short_overridden) and (t_short is not None):
        base_short = float(t_short)
    return float(base_long), float(base_short)


def _resolve_tp_margins(*, args, sup_meta: dict) -> tuple[float, float]:
    base = _quantize_pips_0p1(getattr(args, "tp_margin_pips", 0.0))
    long_overridden = getattr(args, "tp_margin_long_pips", None) is not None
    short_overridden = getattr(args, "tp_margin_short_pips", None) is not None
    base_long = (
        _quantize_pips_0p1(float(args.tp_margin_long_pips))
        if long_overridden
        else float(base)
    )
    base_short = (
        _quantize_pips_0p1(float(args.tp_margin_short_pips))
        if short_overridden
        else float(base)
    )

    if not bool(getattr(args, "sup_auto_threshold", False)):
        return float(base_long), float(base_short)

    m_long = _extract_tuned_tp_margin_pips((sup_meta or {}).get("long", {}) if isinstance(sup_meta, dict) else {})
    m_short = _extract_tuned_tp_margin_pips((sup_meta or {}).get("short", {}) if isinstance(sup_meta, dict) else {})
    if (not long_overridden) and (m_long is not None):
        base_long = float(m_long)
    if (not short_overridden) and (m_short is not None):
        base_short = float(m_short)
    return float(base_long), float(base_short)


def _maybe_hot_reload_supervised(
    *,
    sup_long_path: str,
    sup_short_path: str,
    sup_long: Optional[SupervisedMLP],
    sup_short: Optional[SupervisedMLP],
    last_mtime_long: float,
    last_mtime_short: float,
    expected_input_dim: int,
    model_type: str = "mlp",
    window_size: int = WINDOW_SIZE,
) -> tuple[Optional[SupervisedMLP], Optional[SupervisedMLP], dict, float, float]:
    """Reload supervised models if files changed.

    Returns (sup_long, sup_short, meta, last_mtime_long, last_mtime_short).
    If reload fails or dims mismatch, keeps the previous models.
    """
    m_l = _safe_mtime(sup_long_path)
    m_s = _safe_mtime(sup_short_path)

    changed = (m_l > last_mtime_long) or (m_s > last_mtime_short)
    if not changed:
        return sup_long, sup_short, {}, last_mtime_long, last_mtime_short

    try:
        new_long, meta_l = _load_supervised_model(str(sup_long_path), model_type=model_type, window_size=window_size)
        new_short, meta_s = _load_supervised_model(str(sup_short_path), model_type=model_type, window_size=window_size)

        # Basic input dim sanity check
        dim_l = int(meta_l.get("input_dim", meta_l.get("n_features", 0)))
        dim_s = int(meta_s.get("input_dim", meta_s.get("n_features", 0)))
        # Prefer the stored checkpoint input_dim if present
        # (We also stored it as ckpt['input_dim'] in the trainer.)
        # If meta doesn't include it, we just don't enforce.
        ckpt_ok = True
        try:
            # model net starts with Linear(expected_input_dim, ...)
            # but that's not directly accessible without digging; use expected_input_dim as the enforced value.
            pass
        except Exception:
            ckpt_ok = True

        # Enforce expected input dim by doing a lightweight forward shape check
        with th.no_grad():
            x = th.zeros((1, int(expected_input_dim)), dtype=th.float32)
            _ = new_long(x)
            _ = new_short(x)

        print(f"\n🔁 Hot-reloaded supervised models: long={sup_long_path} short={sup_short_path}")
        return new_long, new_short, {"long": meta_l, "short": meta_s}, m_l, m_s
    except Exception as e:
        print(f"\n⚠️  Hot-reload failed (keeping current models): {e!r}")
        return sup_long, sup_short, {}, last_mtime_long, last_mtime_short


def _device_for_m2_air() -> str:
    # Prefer CUDA on Windows/NVIDIA, then Apple MPS, else CPU.
    try:
        if th.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "mps" if th.backends.mps.is_available() else "cpu"


# --- TRANSFORMER (match training architecture) ---
class MarketTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, nhead: int = 2, layers: int = 1):
        super().__init__(observation_space, features_dim)
        self.d_model = 32
        n_features = observation_space.shape[0]
        window_size = observation_space.shape[1]
        self.embedding = nn.Linear(n_features, self.d_model)
        self.pos_encoder = nn.Parameter(th.zeros(1, window_size, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(window_size * self.d_model, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.permute(0, 2, 1)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.linear(x)


# --- DUMMY ENV (for SB3 load/predict) ---
class BridgeEnvV31(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = int(n_features)
        self.observation_space = spaces.Box(low=-20, high=20, shape=(self.n_features, WINDOW_SIZE), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        return np.zeros((self.n_features, WINDOW_SIZE), dtype=np.float32), {}

    def step(self, action):
        # Dummy step: this env is only used so SB3 can load the policy.
        obs = np.zeros((self.n_features, WINDOW_SIZE), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def valid_action_mask(self):
        return np.array([True, True, True], dtype=bool)


@dataclass
class LiveParams:
    min_gap_pips: float = 1.5
    min_body_atr: float = 0.8
    max_fvg_age_bars: int = 60
    sl_buffer_atr: float = 0.25
    min_sl_pips: float = 0.0
    max_sl_pips: float = 0.0
    min_tp_dist_pips: float = 0.0
    min_rr: float = 0.0
    min_rr_long: float = 0.0
    min_rr_short: float = 0.0
    tp_margin_long_pips: float = 0.0
    tp_margin_short_pips: float = 0.0


def load_best_params() -> LiveParams:
    try:
        raise RuntimeError("Use load_best_params_from(study_name, storage_db)")
    except Exception:
        return LiveParams()


def load_best_params_from(*, study_name: str, storage_db: str) -> LiveParams:
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_db)
        bp = study.best_params
        return LiveParams(
            min_gap_pips=float(bp.get("min_gap_pips", 1.5)),
            min_body_atr=float(bp.get("min_body_atr", 0.8)),
            max_fvg_age_bars=int(bp.get("max_fvg_age_bars", 60)),
            sl_buffer_atr=float(bp.get("sl_buffer_atr", 0.25)),
            min_sl_pips=float(bp.get("min_sl_pips", 0.0)),
            max_sl_pips=float(bp.get("max_sl_pips", 0.0)),
            min_tp_dist_pips=float(bp.get("min_tp_dist_pips", 0.0)),
            min_rr=float(bp.get("min_rr", 0.0)),
            min_rr_long=float(bp.get("min_rr_long", 0.0)),
            min_rr_short=float(bp.get("min_rr_short", 0.0)),
        )
    except Exception:
        return LiveParams()


class ModelManager:
    def __init__(self, env, *, checkpoint_dir: str):
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.model: Optional[MaskablePPO] = None
        self.current_model_path: str = ""
        self.last_mtime: float = 0.0
        self.load_latest_model(force=True)

    def _find_latest_model(self) -> Optional[str]:
        # Prefer the stable exported best model if present.
        best_path = os.path.join(self.checkpoint_dir, "brain_best.zip")
        if os.path.exists(best_path):
            return best_path

        files = glob.glob(os.path.join(self.checkpoint_dir, "model_trial_*.zip"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def load_latest_model(self, force: bool = False):
        path = self._find_latest_model()
        if not path:
            return
        mtime = os.path.getmtime(path)
        if (not force) and path == self.current_model_path and mtime <= self.last_mtime:
            return

        print(f"\nReloading model: {path}...")
        self.model = MaskablePPO.load(path, env=self.env, device=_device_for_m2_air())
        self.current_model_path = path
        self.last_mtime = mtime
        print("Model loaded.")

    def predict(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        if self.model is None:
            return 0
        if time.time() - self.last_mtime > 60:
            self.load_latest_model(force=False)
        action, _ = self.model.predict(obs, deterministic=True, action_masks=action_mask)
        if isinstance(action, np.ndarray):
            return int(action.item())
        return int(action)


class LiveFeatureBuilder:
    def __init__(
        self,
        window_size: int,
        params: LiveParams,
        *,
        trade_start_hour: int,
        trade_end_hour: int,
        asia_start_hour: int,
        asia_end_hour: int,
        include_trend_sr: bool,
        include_ifvg_zone_features: bool,
        use_ifvg_inversion_logic: bool,
        ifvg_retest_near_atr: Optional[float] = None,
        atr_min_pips: float = 0.0,
        atr_max_pips: float = 0.0,
        trend_lookback: int = 50,
        sr_lookback: int = 288,
        structure_window: int = 24,
        trend_slope_short_max: Optional[float] = None,
        trend_slope_long_min: Optional[float] = None,
        trend_slope_min_abs: Optional[float] = None,
        day_trend_filter: bool = False,
        day_trend_lookback: int = 240,
        day_trend_threshold: float = 0.0,
        day_trend_min_hour: int = 0,
        require_day_trend_for_longs: bool = False,
        require_day_trend_for_shorts: bool = False,

        levels_mode: str = "asia",
        atr_sl_mult: float = 1.0,
        atr_rr: float = 1.4,
    ):
        self.window_size = window_size
        self.params = params
        self.trade_start_hour = int(trade_start_hour)
        self.trade_end_hour = int(trade_end_hour)
        self.asia_start_hour = int(asia_start_hour)
        self.asia_end_hour = int(asia_end_hour)
        self.include_trend_sr = bool(include_trend_sr)
        self.include_ifvg_zone_features = bool(include_ifvg_zone_features)
        self.use_ifvg_inversion_logic = bool(use_ifvg_inversion_logic)
        # How near (in ATR units) a bar must be to an iFVG edge to count as a "near retest".
        # Smaller values -> tighter retest gating (fewer allowed retests).
        self.ifvg_retest_near_atr = float(ifvg_retest_near_atr) if (ifvg_retest_near_atr is not None) else 0.25
        self.atr_min_pips = float(atr_min_pips or 0.0)
        self.atr_max_pips = float(atr_max_pips or 0.0)
        self.trend_lookback = int(trend_lookback)
        self.sr_lookback = int(sr_lookback)
        self.structure_window = int(structure_window)

        self.trend_slope_short_max = float(trend_slope_short_max) if trend_slope_short_max is not None else None
        self.trend_slope_long_min = float(trend_slope_long_min) if trend_slope_long_min is not None else None
        self.trend_slope_min_abs = float(trend_slope_min_abs) if trend_slope_min_abs is not None else None

        self.day_trend_filter = bool(day_trend_filter)
        self.day_trend_lookback = int(day_trend_lookback)
        self.day_trend_threshold = float(day_trend_threshold or 0.0)
        self.day_trend_min_hour = int(day_trend_min_hour)
        self.require_day_trend_for_longs = bool(require_day_trend_for_longs)
        self.require_day_trend_for_shorts = bool(require_day_trend_for_shorts)

        self.levels_mode = str(levels_mode or "asia").strip().lower()
        if self.levels_mode not in ("asia", "atr", "asia_then_atr"):
            self.levels_mode = "asia"
        self.atr_sl_mult = float(atr_sl_mult or 1.0)
        self.atr_rr = float(atr_rr or 1.4)

        self.last_atr_pips: float = float("nan")
        self.atr_ok: bool = True
        self.last_trend_slope_atr: float = 0.0

        # Daily trend regime (computed once/day; used to block counter-trend trades).
        self.day_trend_bias: int = 0  # -1 down, 0 unset/neutral, +1 up
        self.day_trend_is_set: bool = False
        self.day_trend_slope_atr: float = float("nan")
        self.last_day_trend_block_long: bool = False
        self.last_day_trend_block_short: bool = False
        self.last_day_trend_req_block_long: bool = False
        self.last_day_trend_req_block_short: bool = False

        # Keep only as much history as we actually need for features.
        # This matters a lot during replay training (hundreds of thousands of bars).
        self._max_history = int(
            max(
                60,
                self.window_size + 2,
                self.trend_lookback,
                self.sr_lookback,
                self.day_trend_lookback,
                2 * self.structure_window,
            )
            + 5
        )

        self.history = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "Corr_Close", "Hour"]
        )

        # day/session state (we infer new day when hour wraps down)
        self._last_hour: Optional[int] = None
        self._day_id: int = 0
        self._bar_in_day: int = 0
        self._global_bar_index: int = 0

        self.asia_high: float = np.nan
        self.asia_low: float = np.nan
        self.day_bias: int = 0

        # active FVG zones
        self._bull_bottom = np.nan
        self._bull_top = np.nan
        self._bull_age = 0
        self._bear_bottom = np.nan
        self._bear_top = np.nan
        self._bear_age = 0

        # inverted zones (iFVG)
        self._inv_bull_bottom = np.nan
        self._inv_bull_top = np.nan
        self._inv_bull_age = 0
        self._inv_bear_bottom = np.nan
        self._inv_bear_top = np.nan
        self._inv_bear_age = 0

    def _new_day_if_needed(self, hour: int):
        if self._last_hour is None:
            self._last_hour = hour
            return
        if hour < self._last_hour:
            self._day_id += 1
            self._bar_in_day = 0
            self.asia_high = np.nan
            self.asia_low = np.nan
            self.day_bias = 0

            self.day_trend_bias = 0
            self.day_trend_is_set = False
            self.day_trend_slope_atr = float("nan")
            self.last_day_trend_block_long = False
            self.last_day_trend_block_short = False
            self.last_day_trend_req_block_long = False
            self.last_day_trend_req_block_short = False

            self._bull_bottom = np.nan
            self._bull_top = np.nan
            self._bull_age = 0
            self._bear_bottom = np.nan
            self._bear_top = np.nan
            self._bear_age = 0

            self._inv_bull_bottom = np.nan
            self._inv_bull_top = np.nan
            self._inv_bull_age = 0
            self._inv_bear_bottom = np.nan
            self._inv_bear_top = np.nan
            self._inv_bear_age = 0
        self._last_hour = hour

    def add_bar(self, row: dict):
        hour = int(row.get("Hour", 0))
        self._new_day_if_needed(hour)
        self._bar_in_day += 1
        self._global_bar_index += 1

        # Avoid pd.concat per bar (very slow during replay training). loc assignment is much faster.
        self.history.loc[len(self.history)] = row
        if len(self.history) > int(self._max_history):
            self.history = self.history.iloc[-int(self._max_history) :].reset_index(drop=True)

        # Update Asia levels during session
        if self.asia_start_hour <= hour < self.asia_end_hour:
            hi = float(row["High"])
            lo = float(row["Low"])
            self.asia_high = hi if not np.isfinite(self.asia_high) else max(self.asia_high, hi)
            self.asia_low = lo if not np.isfinite(self.asia_low) else min(self.asia_low, lo)

        # Determine bias once per day (friend-version)
        if hour >= self.asia_end_hour and self.day_bias == 0 and np.isfinite(self.asia_high) and np.isfinite(self.asia_low):
            hi = float(row["High"])
            lo = float(row["Low"])
            cl = float(row["Close"])
            bear = (hi >= self.asia_high) and (cl < self.asia_high)
            bull = (lo <= self.asia_low) and (cl > self.asia_low)
            if bear:
                self.day_bias = -1
            elif bull:
                self.day_bias = 1

        # Update FVG state / allow flags
        self._update_fvg_state()

    def _compute_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean().bfill().fillna(tr.mean())

    def _update_fvg_state(self):
        df = self.history
        n = len(df)
        if n < 3:
            return

        # age out
        if np.isfinite(self._bull_bottom):
            self._bull_age += 1
            if self._bull_age > int(self.params.max_fvg_age_bars):
                self._bull_bottom = np.nan
                self._bull_top = np.nan
                self._bull_age = 0

        if np.isfinite(self._bear_bottom):
            self._bear_age += 1
            if self._bear_age > int(self.params.max_fvg_age_bars):
                self._bear_bottom = np.nan
                self._bear_top = np.nan
                self._bear_age = 0

        if np.isfinite(self._inv_bull_bottom):
            self._inv_bull_age += 1
            if self._inv_bull_age > int(self.params.max_fvg_age_bars):
                self._inv_bull_bottom = np.nan
                self._inv_bull_top = np.nan
                self._inv_bull_age = 0

        if np.isfinite(self._inv_bear_bottom):
            self._inv_bear_age += 1
            if self._inv_bear_age > int(self.params.max_fvg_age_bars):
                self._inv_bear_bottom = np.nan
                self._inv_bear_top = np.nan
                self._inv_bear_age = 0

        # compute ATR at i-1 for displacement filter
        atr = self._compute_atr_series(df).values
        i = n - 1
        body = abs(float(df.loc[i - 1, "Close"]) - float(df.loc[i - 1, "Open"]))
        atr_i = float(max(float(atr[i - 1]), 1e-6))
        min_gap = float(self.params.min_gap_pips) * PIP_VALUE

        # FVG detection on current bar (i)
        hi_i2 = float(df.loc[i - 2, "High"])
        lo_i2 = float(df.loc[i - 2, "Low"])
        hi_i = float(df.loc[i, "High"])
        lo_i = float(df.loc[i, "Low"])

        # bullish FVG: High[i-2] < Low[i]
        if (hi_i2 < lo_i) and ((lo_i - hi_i2) >= min_gap) and (body >= float(self.params.min_body_atr) * atr_i):
            self._bull_bottom = hi_i2
            self._bull_top = lo_i
            self._bull_age = 0

        # bearish FVG: Low[i-2] > High[i]
        if (lo_i2 > hi_i) and ((lo_i2 - hi_i) >= min_gap) and (body >= float(self.params.min_body_atr) * atr_i):
            self._bear_bottom = hi_i
            self._bear_top = lo_i2
            self._bear_age = 0

        # inversion (wick-through) — must run BEFORE wick-based clearing.
        if np.isfinite(self._bull_bottom) and (lo_i <= float(self._bull_bottom)):
            self._inv_bull_bottom = float(self._bull_bottom)
            self._inv_bull_top = float(self._bull_top)
            self._inv_bull_age = 0
            self._bull_bottom = np.nan
            self._bull_top = np.nan
            self._bull_age = 0

        if np.isfinite(self._bear_top) and (hi_i >= float(self._bear_top)):
            self._inv_bear_bottom = float(self._bear_bottom)
            self._inv_bear_top = float(self._bear_top)
            self._inv_bear_age = 0
            self._bear_bottom = np.nan
            self._bear_top = np.nan
            self._bear_age = 0

        # clear zones if fully traded through (only for zones that did NOT invert)
        if np.isfinite(self._bull_bottom) and lo_i <= self._bull_bottom:
            self._bull_bottom = np.nan
            self._bull_top = np.nan
            self._bull_age = 0

        if np.isfinite(self._bear_top) and hi_i >= self._bear_top:
            self._bear_bottom = np.nan
            self._bear_top = np.nan
            self._bear_age = 0

    def _corr_series(self, close: pd.Series, corr_close: pd.Series) -> pd.Series:
        try:
            return close.rolling(50).corr(corr_close).fillna(0.0)
        except Exception:
            return pd.Series(np.zeros(len(close), dtype=np.float32))

    def get_obs_and_mask(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        df = self.history.copy()
        if len(df) < max(60, self.window_size + 2):
            self.last_atr_pips = float("nan")
            self.atr_ok = True
            return None, np.array([[True, False, False]], dtype=bool)

        df["Open"] = df["Open"].replace(0, 1e-5).astype(float)
        df["Close"] = df["Close"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Volume"] = df["Volume"].astype(float)

        if "Corr_Close" not in df.columns:
            df["Corr_Close"] = 0.0
        df["Corr_Close"] = df["Corr_Close"].astype(float).fillna(0.0)

        # time features (approximate 5m bars without explicit minutes)
        minutes = float((self._bar_in_day - 1) * 5 % 1440)
        angle = 2 * np.pi * (minutes / 1440.0)
        tod_sin = float(np.sin(angle))
        tod_cos = float(np.cos(angle))
        dow = 0.0

        # indicators
        df["Corr"] = self._corr_series(df["Close"], df["Corr_Close"])
        atr = self._compute_atr_series(df)

        prev_close = df["Close"].shift(1).replace(0, np.nan)
        df["Rel_Open"] = (((df["Open"] - prev_close) / prev_close) * 1000).fillna(0.0)
        df["Rel_High"] = ((df["High"] - df["Open"]) / df["Open"]) * 1000
        df["Rel_Low"] = ((df["Low"] - df["Open"]) / df["Open"]) * 1000
        df["Rel_Close"] = ((df["Close"] - df["Open"]) / df["Open"]) * 1000

        vol_ma = df["Volume"].rolling(50).mean().replace(0, 1.0)
        df["Rel_Vol"] = (df["Volume"] / vol_ma).fillna(0.0)
        df["Rel_Corr"] = df["Corr"].fillna(0.0)

        clip_cols = ["Rel_Open", "Rel_High", "Rel_Low", "Rel_Close", "Rel_Vol", "Rel_Corr"]
        df[clip_cols] = df[clip_cols].clip(-20.0, 20.0)

        # asia distance features (use current stored asia levels)
        last_close = float(df["Close"].iloc[-1])
        last_atr = float(max(float(atr.iloc[-1]), 1e-6))
        try:
            self.last_atr_pips = float(last_atr / float(max(PIP_VALUE, 1e-9)))
        except Exception:
            self.last_atr_pips = float("nan")

        # --- V5 context features ---
        # macro_trend: 4-hour slope (48 * 5min bars) normalized by last_atr
        macro_trend = 0.0
        macro_period = 48
        if len(df) >= macro_period:
            y = df["Close"].iloc[-macro_period :].astype(float).values
            x = np.arange(macro_period, dtype=np.float32)
            x = x - x.mean()
            y = y - float(np.mean(y))
            denom = float(np.dot(x, x))
            slope = float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0
            try:
                macro_trend = float(np.clip(slope / float(max(last_atr, 1e-6)), -10.0, 10.0))
            except Exception:
                macro_trend = 0.0

        # vol_regime: ratio of recent ATR to long-term ATR (long-term ATR from ATR series)
        try:
            long_term_atr = float(max(float(atr.rolling(macro_period).mean().iloc[-1]), 1e-6)) if len(atr) >= macro_period else float(max(float(atr.iloc[-1]), 1e-6))
            vol_regime = float(np.clip(float(last_atr) / float(max(long_term_atr, 1e-6)), -10.0, 10.0))
        except Exception:
            vol_regime = 0.0

        # rsi (14) normalized to -1..1
        rsi_norm = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
        try:
            period_rsi = 14
            delta = df["Close"].astype(float).diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.rolling(period_rsi).mean()
            avg_loss = loss.rolling(period_rsi).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi = rsi.fillna(50.0)
            rsi_norm = (rsi - 50.0) / 50.0
            rsi_norm = rsi_norm.clip(-1.0, 1.0)
        except Exception:
            rsi_norm = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

        # Attach V5 context cols to df so window picks them up (macro_trend/vol_regime are constant across window)
        df["macro_trend"] = float(macro_trend)
        df["vol_regime"] = float(vol_regime)
        df["rsi"] = rsi_norm

        atr_ok = True
        if float(self.atr_min_pips) > 0.0 and np.isfinite(self.last_atr_pips):
            atr_ok = atr_ok and (self.last_atr_pips >= float(self.atr_min_pips))
        if float(self.atr_max_pips) > 0.0 and np.isfinite(self.last_atr_pips):
            atr_ok = atr_ok and (self.last_atr_pips <= float(self.atr_max_pips))
        self.atr_ok = bool(atr_ok)
        ah = float(self.asia_high) if np.isfinite(self.asia_high) else 0.0
        al = float(self.asia_low) if np.isfinite(self.asia_low) else 0.0

        dist_to_asia_high_atr = float(((ah - last_close) / last_atr)) if np.isfinite(self.asia_high) else 0.0
        dist_to_asia_low_atr = float(((last_close - al) / last_atr)) if np.isfinite(self.asia_low) else 0.0
        dist_to_asia_high_atr = float(np.clip(dist_to_asia_high_atr, -10.0, 10.0))
        dist_to_asia_low_atr = float(np.clip(dist_to_asia_low_atr, -10.0, 10.0))

        # Current bar hour (used for session filters and optional daily trend-bias calculation)
        hour = int(df["Hour"].iloc[-1]) if "Hour" in df.columns else 0

        # build feature matrix for last window
        window = df.iloc[-self.window_size :].copy()
        window["tod_sin"] = tod_sin
        window["tod_cos"] = tod_cos
        window["dow"] = dow
        window["dist_to_asia_high_atr"] = dist_to_asia_high_atr
        window["dist_to_asia_low_atr"] = dist_to_asia_low_atr
        window["asia_bias"] = float(self.day_bias)

        # optional: iFVG zone state features (constant across the window, matching supervised training)
        if self.include_ifvg_zone_features:
            z_bot = np.nan
            z_top = np.nan
            if self.day_bias == 1 and np.isfinite(self._inv_bear_bottom):
                z_bot = float(self._inv_bear_bottom)
                z_top = float(self._inv_bear_top)
            elif self.day_bias == -1 and np.isfinite(self._inv_bull_bottom):
                z_bot = float(self._inv_bull_bottom)
                z_top = float(self._inv_bull_top)

            if np.isfinite(z_bot) and np.isfinite(z_top):
                hi = float(df["High"].iloc[-1])
                lo = float(df["Low"].iloc[-1])
                cl = float(df["Close"].iloc[-1])
                in_zone = (lo <= z_top) and (hi >= z_bot)
                window["ifvg_dist_to_top_atr"] = float(np.clip((z_top - cl) / last_atr, -10.0, 10.0))
                window["ifvg_dist_to_bottom_atr"] = float(np.clip((cl - z_bot) / last_atr, -10.0, 10.0))
                window["ifvg_in_zone"] = 1.0 if in_zone else 0.0
            else:
                window["ifvg_dist_to_top_atr"] = 0.0
                window["ifvg_dist_to_bottom_atr"] = 0.0
                window["ifvg_in_zone"] = 0.0

        # optional: trend + structure + S/R
        if self.include_trend_sr:
            lb = int(max(10, self.trend_lookback))
            if len(df) >= lb:
                y = df["Close"].iloc[-lb:].astype(float).values
                x = np.arange(lb, dtype=np.float32)
                x = x - x.mean()
                y = y - float(np.mean(y))
                denom = float(np.dot(x, x))
                slope = float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0
                trend_slope_atr = float(np.clip(slope / last_atr, -10.0, 10.0))
            else:
                trend_slope_atr = 0.0

            self.last_trend_slope_atr = float(trend_slope_atr)

            # Support/Resistance + structure features (independent from daily trend regime).
            sr_lb = int(max(20, self.sr_lookback))
            if len(df) >= sr_lb:
                sr_hi = float(df["High"].iloc[-sr_lb:].max())
                sr_lo = float(df["Low"].iloc[-sr_lb:].min())
                dist_to_sr_high_atr = float(np.clip(((sr_hi - last_close) / last_atr), -10.0, 10.0))
                dist_to_sr_low_atr = float(np.clip(((last_close - sr_lo) / last_atr), -10.0, 10.0))
            else:
                dist_to_sr_high_atr = 0.0
                dist_to_sr_low_atr = 0.0

            w = int(max(10, self.structure_window))
            if len(df) >= 2 * w:
                recent_hi = float(df["High"].iloc[-w:].max())
                recent_lo = float(df["Low"].iloc[-w:].min())
                prior_hi = float(df["High"].iloc[-2 * w : -w].max())
                prior_lo = float(df["Low"].iloc[-2 * w : -w].min())
                bull = (recent_hi > prior_hi) and (recent_lo > prior_lo)
                bear = (recent_hi < prior_hi) and (recent_lo < prior_lo)
                structure = float((1 if bull else 0) - (1 if bear else 0))
            else:
                structure = 0.0

            window["trend_slope_atr"] = trend_slope_atr
            window["structure"] = structure
            window["dist_to_sr_high_atr"] = dist_to_sr_high_atr
            window["dist_to_sr_low_atr"] = dist_to_sr_low_atr

        # Daily trend regime: compute once/day when enough bars exist and (optionally) after a minimum hour.
        # Uses ATR-normalized linear-regression slope over a longer lookback than the per-bar trend_slope_atr.
        if self.day_trend_filter and (not bool(self.day_trend_is_set)):
            lb_day = int(max(20, self.day_trend_lookback))
            if (hour >= int(self.day_trend_min_hour)) and (len(df) >= lb_day) and np.isfinite(last_atr):
                y = df["Close"].iloc[-lb_day:].astype(float).values
                x = np.arange(lb_day, dtype=np.float32)
                x = x - x.mean()
                y = y - float(np.mean(y))
                denom = float(np.dot(x, x))
                slope = float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0
                slope_atr = float(np.clip(slope / float(max(last_atr, 1e-6)), -10.0, 10.0))
                self.day_trend_slope_atr = float(slope_atr)

                th = float(self.day_trend_threshold)
                if th > 0.0:
                    if float(slope_atr) >= float(th):
                        self.day_trend_bias = 1
                        self.day_trend_is_set = True
                    elif float(slope_atr) <= -float(th):
                        self.day_trend_bias = -1
                        self.day_trend_is_set = True

        feature_cols = [
            "Rel_Open",
            "Rel_High",
            "Rel_Low",
            "Rel_Close",
            "Rel_Vol",
            "Rel_Corr",
            "tod_sin",
            "tod_cos",
            "dow",
            "dist_to_asia_high_atr",
            "dist_to_asia_low_atr",
            "asia_bias",
            "macro_trend",
            "vol_regime",
            "rsi",
        ]

        if self.include_trend_sr:
            feature_cols += [
                "trend_slope_atr",
                "structure",
                "dist_to_sr_high_atr",
                "dist_to_sr_low_atr",
            ]

        if self.include_ifvg_zone_features:
            feature_cols += [
                "ifvg_dist_to_top_atr",
                "ifvg_dist_to_bottom_atr",
                "ifvg_in_zone",
            ]

        x = window[feature_cols].values.T.astype(np.float32)
        obs = np.expand_dims(x, axis=0)  # (1, features, window)

        # action mask: require bias and an active retest condition
        allow_long = False
        allow_short = False
        hi = float(df["High"].iloc[-1])
        lo = float(df["Low"].iloc[-1])
        cl = float(df["Close"].iloc[-1])

        if self.use_ifvg_inversion_logic:
            # Long: inverted-bear zone retest (touch/overlap or close near an edge)
            if self.day_bias == 1 and np.isfinite(self._inv_bear_bottom):
                z_bot = float(self._inv_bear_bottom)
                z_top = float(self._inv_bear_top)
                in_zone = (lo <= z_top) and (hi >= z_bot)
                # Allow near-touches to avoid missing entries due to bar resolution.
                near_atr = float(getattr(self, "ifvg_retest_near_atr", 0.25))
                dist_edge = min(abs(cl - z_bot), abs(cl - z_top)) / float(max(last_atr, 1e-6))
                allow_long = bool(in_zone or (dist_edge <= near_atr))

            # Short: inverted-bull zone retest (touch/overlap or close near an edge)
            if self.day_bias == -1 and np.isfinite(self._inv_bull_bottom):
                z_bot = float(self._inv_bull_bottom)
                z_top = float(self._inv_bull_top)
                in_zone = (lo <= z_top) and (hi >= z_bot)
                near_atr = float(getattr(self, "ifvg_retest_near_atr", 0.25))
                dist_edge = min(abs(cl - z_bot), abs(cl - z_top)) / float(max(last_atr, 1e-6))
                allow_short = bool(in_zone or (dist_edge <= near_atr))
        else:
            # Legacy overlap-with-FVG-zone gating
            if self.day_bias == 1 and np.isfinite(self._bull_bottom):
                allow_long = (lo <= float(self._bull_top)) and (hi >= float(self._bull_bottom))
            if self.day_bias == -1 and np.isfinite(self._bear_bottom):
                allow_short = (hi >= float(self._bear_bottom)) and (lo <= float(self._bear_top))

        # London/NY session filter
        if self.trade_start_hour == self.trade_end_hour:
            in_time = True
        elif self.trade_start_hour < self.trade_end_hour:
            in_time = (hour >= self.trade_start_hour) and (hour < self.trade_end_hour)
        else:
            # wrap-around window (e.g. 21 -> 7)
            in_time = (hour >= self.trade_start_hour) or (hour < self.trade_end_hour)

        allow_long = bool(allow_long and self.atr_ok)
        allow_short = bool(allow_short and self.atr_ok)

        # Optional trend-alignment gate: prevent counter-trend trades.
        # For shorts, require trend slope (normalized by ATR) <= trend_slope_short_max.
        # For longs, require trend slope >= trend_slope_long_min.
        if self.include_trend_sr and np.isfinite(self.last_trend_slope_atr):
            if self.trend_slope_short_max is not None:
                allow_short = bool(allow_short and (float(self.last_trend_slope_atr) <= float(self.trend_slope_short_max)))
            if self.trend_slope_long_min is not None:
                allow_long = bool(allow_long and (float(self.last_trend_slope_atr) >= float(self.trend_slope_long_min)))

            # Optional sideways filter: require |trend_slope_atr| >= trend_slope_min_abs.
            # This avoids starving trades by forcing slope sign alignment at the entry bar.
            if self.trend_slope_min_abs is not None and np.isfinite(self.trend_slope_min_abs):
                min_abs = float(self.trend_slope_min_abs)
                if min_abs > 0.0:
                    ok = bool(abs(float(self.last_trend_slope_atr)) >= min_abs)
                    allow_long = bool(allow_long and ok)
                    allow_short = bool(allow_short and ok)

        # Optional daily trend requirement: only allow entries on days where a strong daily bias is set.
        # This is useful when one side (often shorts) performs poorly unless the day is clearly trending.
        self.last_day_trend_req_block_long = False
        self.last_day_trend_req_block_short = False
        if self.day_trend_filter:
            if self.require_day_trend_for_longs and int(self.day_trend_bias) != 1:
                if bool(allow_long):
                    self.last_day_trend_req_block_long = True
                allow_long = False
            if self.require_day_trend_for_shorts and int(self.day_trend_bias) != -1:
                if bool(allow_short):
                    self.last_day_trend_req_block_short = True
                allow_short = False

        # Optional daily trend filter: once/day bias blocks counter-trend trades for the rest of the day.
        self.last_day_trend_block_long = False
        self.last_day_trend_block_short = False
        if self.day_trend_filter and int(self.day_trend_bias) != 0:
            if int(self.day_trend_bias) > 0:
                # Up day -> block shorts
                if bool(allow_short):
                    self.last_day_trend_block_short = True
                allow_short = False
            elif int(self.day_trend_bias) < 0:
                # Down day -> block longs
                if bool(allow_long):
                    self.last_day_trend_block_long = True
                allow_long = False
        mask = np.array([[True, bool(in_time and allow_long), bool(in_time and allow_short)]], dtype=bool)
        return obs, mask

    def get_trade_levels(self, action: int) -> Optional[Tuple[float, float, float]]:
        if len(self.history) < 20:
            return None
        close = float(self.history["Close"].iloc[-1])
        high = float(self.history["High"].iloc[-1])
        low = float(self.history["Low"].iloc[-1])
        hour = int(self.history["Hour"].iloc[-1])
        if not (np.isfinite(close) and np.isfinite(high) and np.isfinite(low)):
            return None

        mode = str(getattr(self, "levels_mode", "asia")).strip().lower()
        if mode not in ("asia", "atr", "asia_then_atr"):
            mode = "asia"

        df = self.history.copy()
        atr = self._compute_atr_series(df)
        last_atr = float(max(float(atr.iloc[-1]), 1e-6))

        spread = SPREAD_PIPS * PIP_VALUE
        slippage = SLIPPAGE_PIPS * PIP_VALUE
        sl_buf = float(self.params.sl_buffer_atr) * last_atr

        margin_pips = 0.0
        if action == 1:
            margin_pips = _quantize_pips_0p1(getattr(self.params, "tp_margin_long_pips", 0.0))
        elif action == 2:
            margin_pips = _quantize_pips_0p1(getattr(self.params, "tp_margin_short_pips", 0.0))
        tp_margin = float(margin_pips) * PIP_VALUE

        min_sl_pips = float(getattr(self.params, "min_sl_pips", 0.0) or 0.0)
        max_sl_pips = float(getattr(self.params, "max_sl_pips", 0.0) or 0.0)
        min_tp_dist_pips = float(getattr(self.params, "min_tp_dist_pips", 0.0) or 0.0)

        min_rr = float(getattr(self.params, "min_rr", 0.0) or 0.0)
        min_rr_long = float(getattr(self.params, "min_rr_long", 0.0) or 0.0)
        min_rr_short = float(getattr(self.params, "min_rr_short", 0.0) or 0.0)

        def _atr_levels(act: int) -> Optional[Tuple[float, float, float]]:
            atr_sl_mult = float(getattr(self, "atr_sl_mult", 1.0) or 1.0)
            atr_rr = float(getattr(self, "atr_rr", 1.4) or 1.4)
            if not np.isfinite(atr_sl_mult) or atr_sl_mult <= 0.0:
                atr_sl_mult = 1.0
            if not np.isfinite(atr_rr) or atr_rr <= 0.0:
                atr_rr = 1.4

            risk_dist = float(atr_sl_mult) * float(last_atr)
            if risk_dist <= 1e-12:
                return None

            if act == 1:
                entry = close + spread / 2.0 + slippage
                sl = entry - risk_dist
                sl = _clamp_sl(entry, sl, is_long=True)

                rr_req_eff = float(min_rr)
                if min_rr_long > 0.0:
                    rr_req_eff = max(rr_req_eff, float(min_rr_long))
                rr_eff = float(rr_req_eff) if rr_req_eff > 0.0 else float(atr_rr)
                tp = entry + rr_eff * abs(entry - sl)

                if not (tp > entry and sl < entry):
                    return None
                if min_tp_dist_pips > 0.0:
                    tp_dist_pips = (tp - entry) / float(PIP_VALUE)
                    if tp_dist_pips < float(min_tp_dist_pips):
                        return None
                return entry, sl, tp

            if act == 2:
                entry = close - spread / 2.0 - slippage
                sl = entry + risk_dist
                sl = _clamp_sl(entry, sl, is_long=False)

                rr_req_eff = float(min_rr)
                if min_rr_short > 0.0:
                    rr_req_eff = max(rr_req_eff, float(min_rr_short))
                rr_eff = float(rr_req_eff) if rr_req_eff > 0.0 else float(atr_rr)
                tp = entry - rr_eff * abs(entry - sl)

                if not (tp < entry and sl > entry):
                    return None
                if min_tp_dist_pips > 0.0:
                    tp_dist_pips = (entry - tp) / float(PIP_VALUE)
                    if tp_dist_pips < float(min_tp_dist_pips):
                        return None
                return entry, sl, tp

            return None

        if mode == "atr":
            return _atr_levels(action)

        # Asia-based modes require Asia range and post-Asia hours.
        if not (np.isfinite(self.asia_high) and np.isfinite(self.asia_low)):
            return _atr_levels(action) if mode == "asia_then_atr" else None
        if hour < self.asia_end_hour:
            return _atr_levels(action) if mode == "asia_then_atr" else None

        def _rr_ok(entry_: float, sl_: float, tp_: float, *, is_long: bool) -> bool:
            rr_req = float(min_rr)
            if is_long and min_rr_long > 0.0:
                rr_req = max(rr_req, float(min_rr_long))
            if (not is_long) and min_rr_short > 0.0:
                rr_req = max(rr_req, float(min_rr_short))
            if rr_req <= 0.0:
                return True
            risk = abs(entry_ - sl_) / float(PIP_VALUE)
            reward = abs(tp_ - entry_) / float(PIP_VALUE)
            if risk <= 1e-9:
                return False
            rr = reward / risk
            return rr >= rr_req

        def _clamp_sl(entry_: float, sl_: float, *, is_long: bool) -> float:
            if not (min_sl_pips > 0.0 or max_sl_pips > 0.0):
                return sl_
            dist_pips = abs(entry_ - sl_) / float(PIP_VALUE)
            new_dist = float(dist_pips)
            if min_sl_pips > 0.0:
                new_dist = max(new_dist, float(min_sl_pips))
            if max_sl_pips > 0.0:
                new_dist = min(new_dist, float(max_sl_pips))
            if abs(new_dist - dist_pips) <= 1e-9:
                return sl_
            return (entry_ - new_dist * float(PIP_VALUE)) if is_long else (entry_ + new_dist * float(PIP_VALUE))

        if action == 1:
            entry = close + spread / 2.0 + slippage
            tp = float(self.asia_high) - spread / 2.0 - tp_margin
            sl = float(self.asia_low) - sl_buf - spread / 2.0
            sl = _clamp_sl(entry, sl, is_long=True)
            if not (tp > entry and sl < entry):
                return _atr_levels(action) if mode == "asia_then_atr" else None
            if min_tp_dist_pips > 0.0:
                tp_dist_pips = (tp - entry) / float(PIP_VALUE)
                if tp_dist_pips < float(min_tp_dist_pips):
                    return _atr_levels(action) if mode == "asia_then_atr" else None
            if not _rr_ok(entry, sl, tp, is_long=True):
                return _atr_levels(action) if mode == "asia_then_atr" else None
            return entry, sl, tp

        if action == 2:
            entry = close - spread / 2.0 - slippage
            tp = float(self.asia_low) + spread / 2.0 + tp_margin
            sl = float(self.asia_high) + sl_buf + spread / 2.0
            sl = _clamp_sl(entry, sl, is_long=False)
            if not (tp < entry and sl > entry):
                return _atr_levels(action) if mode == "asia_then_atr" else None
            if min_tp_dist_pips > 0.0:
                tp_dist_pips = (entry - tp) / float(PIP_VALUE)
                if tp_dist_pips < float(min_tp_dist_pips):
                    return _atr_levels(action) if mode == "asia_then_atr" else None
            if not _rr_ok(entry, sl, tp, is_long=False):
                return _atr_levels(action) if mode == "asia_then_atr" else None
            return entry, sl, tp

        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["eurjpy", "eurusd"], default="eurjpy")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage-db", default=DEFAULT_STORAGE_DB)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hour-offset", type=int, default=0, help="Shift incoming Hour by this many hours (e.g. broker->UTC alignment)")
    parser.add_argument("--asia-start-hour", type=int, default=ASIA_START_HOUR, help="Asia range start hour (0-23) in the adjusted time")
    parser.add_argument("--asia-end-hour", type=int, default=ASIA_END_HOUR, help="Asia range end hour (0-23) in the adjusted time")
    parser.add_argument("--trade-start-hour", type=int, default=8, help="Only allow entries at/after this hour (0-23)")
    parser.add_argument("--trade-end-hour", type=int, default=16, help="Only allow entries before this hour (0-23)")
    parser.add_argument("--pip-value", type=float, default=0.0, help="Override pip value (e.g. 0.0001 for EURUSD, 0.01 for JPY pairs). 0 keeps default.")
    parser.add_argument("--min-gap-pips", type=float, default=None, help="Override live FVG min gap (pips). If not set, uses Optuna best/default.")
    parser.add_argument("--min-body-atr", type=float, default=None, help="Override live FVG displacement filter (body >= min_body_atr * ATR).")
    parser.add_argument("--max-fvg-age-bars", type=int, default=None, help="Override live FVG/iFVG max age in bars.")
    parser.add_argument("--sl-buffer-atr", type=float, default=None, help="Override SL buffer in ATR used for SL/TP levels.")
    parser.add_argument(
        "--min-sl-pips",
        type=float,
        default=None,
        help="If set >0, enforce a minimum SL distance (pips) from entry (helps avoid too-tight stops).",
    )
    parser.add_argument(
        "--max-sl-pips",
        type=float,
        default=None,
        help="If set >0, enforce a maximum SL distance (pips) from entry (caps tail losses).",
    )
    parser.add_argument(
        "--min-tp-dist-pips",
        type=float,
        default=None,
        help="If set >0, suppress trades where TP distance from entry is smaller than this many pips (avoid low-upside setups).",
    )
    parser.add_argument(
        "--min-rr",
        type=float,
        default=None,
        help="If set >0, suppress trades where (TP distance / SL distance) is below this (reward:risk filter).",
    )
    parser.add_argument(
        "--min-rr-long",
        type=float,
        default=None,
        help="Optional override for long minimum reward:risk. Effective min is max(--min-rr, --min-rr-long).",
    )
    parser.add_argument(
        "--min-rr-short",
        type=float,
        default=None,
        help="Optional override for short minimum reward:risk. Effective min is max(--min-rr, --min-rr-short).",
    )
    parser.add_argument(
        "--tp-margin-pips",
        type=float,
        default=0.0,
        help="If >0, set TP inside Asia high/low by this many pips (min 0.1, step 0.1). Long: AH-margin. Short: AL+margin.",
    )
    parser.add_argument(
        "--tp-margin-long-pips",
        type=float,
        default=None,
        help="Optional override for long TP margin (pips). Defaults to --tp-margin-pips.",
    )
    parser.add_argument(
        "--tp-margin-short-pips",
        type=float,
        default=None,
        help="Optional override for short TP margin (pips). Defaults to --tp-margin-pips.",
    )
    parser.add_argument(
        "--levels-mode",
        choices=["asia", "atr", "asia_then_atr"],
        default="asia",
        help=(
            "How to compute SL/TP levels for entries. "
            "'asia' uses Asia range targets (default). "
            "'atr' uses ATR-based SL/TP (more trades). "
            "'asia_then_atr' falls back to ATR levels when Asia levels are invalid."
        ),
    )
    parser.add_argument(
        "--model-type", 
        default="mlp", 
        choices=["mlp", "transformer", "lstm"], 
        help="Model architecture",
    )
    parser.add_argument(
        "--atr-sl-mult",
        type=float,
        default=1.0,
        help="ATR levels: SL distance = atr_sl_mult * ATR (price units). Used by --levels-mode atr and as fallback in asia_then_atr.",
    )
    parser.add_argument(
        "--atr-rr",
        type=float,
        default=1.4,
        help="ATR levels: if min_rr (or min_rr_long/short) is set, TP uses that RR; otherwise uses atr_rr.",
    )
    parser.add_argument(
        "--min-atr-pips",
        type=float,
        default=0.0,
        help="If >0, suppress entries when ATR (pips) is below this (regime filter).",
    )
    parser.add_argument(
        "--max-atr-pips",
        type=float,
        default=0.0,
        help="If >0, suppress entries when ATR (pips) is above this (regime filter).",
    )
    parser.add_argument("--trend-sr", action="store_true", help="Include trend + structure + support/resistance features (requires a matching trained model)")
    parser.add_argument(
        "--trend-slope-short-max",
        type=float,
        default=None,
        help="If set, only allow SHORT entries when trend_slope_atr <= this value (e.g. 0.0 blocks shorts in uptrends). Requires --trend-sr.",
    )
    parser.add_argument(
        "--trend-slope-long-min",
        type=float,
        default=None,
        help="If set, only allow LONG entries when trend_slope_atr >= this value. Requires --trend-sr.",
    )
    parser.add_argument(
        "--trend-slope-min-abs",
        type=float,
        default=None,
        help="If set, only allow entries when abs(trend_slope_atr) >= this value (sideways filter). Requires --trend-sr.",
    )
    parser.add_argument(
        "--day-trend-filter",
        action="store_true",
        help=(
            "If set, compute a once-per-day trend bias from ATR-normalized slope over --day-trend-lookback bars, "
            "and block counter-trend trades for the rest of the day."
        ),
    )
    parser.add_argument(
        "--day-trend-lookback",
        type=int,
        default=240,
        help="Lookback bars for daily trend slope (used only when --day-trend-filter is set).",
    )
    parser.add_argument(
        "--day-trend-threshold",
        type=float,
        default=0.12,
        help=(
            "ATR-normalized slope threshold to lock a daily bias (e.g. 0.12). "
            "Bias sets to +1 when slope_atr>=th, -1 when slope_atr<=-th. If never crossed, bias stays unset (0)."
        ),
    )
    parser.add_argument(
        "--day-trend-min-hour",
        type=int,
        default=0,
        help="Only allow daily trend bias to lock after this hour (0-23). Helps avoid basing the day on thin overnight bars.",
    )
    parser.add_argument(
        "--require-day-trend-for-longs",
        action="store_true",
        help="If set (and --day-trend-filter), only allow LONG entries when day_trend_bias == +1 (strong up-day).",
    )
    parser.add_argument(
        "--require-day-trend-for-shorts",
        action="store_true",
        help="If set (and --day-trend-filter), only allow SHORT entries when day_trend_bias == -1 (strong down-day).",
    )
    parser.add_argument("--mode", choices=["rl", "supervised"], default="rl", help="Inference mode: SB3 RL policy or supervised classifiers")
    parser.add_argument("--ifvg-zone-features", action="store_true", help="Add iFVG zone state features (requires a matching supervised model)")
    parser.add_argument("--sup-long", default="./ai_supervised_v1/sup_long_eurusd_trendsr.pt", help="Path to supervised long .pt")
    parser.add_argument("--sup-short", default="./ai_supervised_v1/sup_short_eurusd_trendsr.pt", help="Path to supervised short .pt")
    parser.add_argument("--sup-threshold", type=float, default=0.55, help="Min probability to take a trade in supervised mode")
    parser.add_argument(
        "--sup-thresh-long",
        type=float,
        default=None,
        help="Optional override for long threshold (defaults to --sup-threshold)",
    )
    parser.add_argument(
        "--sup-thresh-short",
        type=float,
        default=None,
        help="Optional override for short threshold (defaults to --sup-threshold)",
    )
    parser.add_argument(
        "--sup-auto-threshold",
        action="store_true",
        help="If set, use tuned_threshold stored in the supervised .pt meta (per side). Falls back to CLI thresholds if missing.",
    )
    parser.add_argument(
        "--disable-longs",
        action="store_true",
        help="If set, never send long entry actions (still logs signals).",
    )
    parser.add_argument(
        "--disable-shorts",
        action="store_true",
        help="If set, never send short entry actions (still logs signals).",
    )
    parser.add_argument("--sup-hot-reload", action="store_true", help="Hot-reload supervised .pt files on change")
    parser.add_argument(
        "--adaptive-threshold",
        action="store_true",
        help="Adaptive quality gate (supervised mode): raise thresholds and optionally cooldown after loss streaks to trade less but higher quality.",
    )
    parser.add_argument("--adaptive-window-trades", type=int, default=30, help="Rolling window size (trades) for adaptive stats (logging only for now).")
    parser.add_argument("--adaptive-loss-streak-trigger", type=int, default=3, help="When consecutive losing trades >= this, start increasing thresholds / cooldown.")
    parser.add_argument("--adaptive-bump-per-loss", type=float, default=0.03, help="Threshold bump added per loss beyond trigger.")
    parser.add_argument("--adaptive-bump-max", type=float, default=0.12, help="Maximum threshold bump.")
    parser.add_argument("--adaptive-cooldown-bars", type=int, default=36, help="After loss-streak trigger, suppress entries for this many bars (0 disables cooldown).")
    parser.add_argument(
        "--adaptive-reset-after-bars",
        type=int,
        default=0,
        help=(
            "If >0, reset adaptive loss_streak/cooldown after this many bars without any trade close. "
            "Prevents threshold bumps from getting stuck when the bot stops trading."
        ),
    )
    parser.add_argument(
        "--send-confidence",
        action="store_true",
        help="Append a 4th field to the response: confidence (max(p_long,p_short)) in supervised mode. cTrader ignores extra fields safely.",
    )
    parser.add_argument(
        "--sup-min-prob-gap",
        type=float,
        default=0.0,
        help="If >0 (supervised mode), require |p_long - p_short| >= this to take an entry. Helps avoid low-contrast/coin-flip signals.",
    )
    parser.add_argument(
        "--sup-min-prob-gap-long",
        type=float,
        default=None,
        help=(
            "Optional override for LONG minimum prob-gap in supervised mode. Effective min is max(--sup-min-prob-gap, --sup-min-prob-gap-long). "
            "Lets you be stricter on one side without starving the other."
        ),
    )
    parser.add_argument(
        "--sup-min-prob-gap-short",
        type=float,
        default=None,
        help=(
            "Optional override for SHORT minimum prob-gap in supervised mode. Effective min is max(--sup-min-prob-gap, --sup-min-prob-gap-short). "
            "Lets you be stricter on one side without starving the other."
        ),
    )
    parser.add_argument(
        "--sup-min-confidence",
        type=float,
        default=0.0,
        help="If >0 (supervised mode), require max(p_long,p_short) >= this to take an entry (confidence filter).",
    )

    parser.add_argument(
        "--ifvg-retest-near-atr",
        type=float,
        default=None,
        help="Override near-ATR distance for iFVG retest gating (default 0.25). Smaller -> tighter retests.",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=0,
        help="If >0, limit the number of entry actions sent per day_id to this count.",
    )
    parser.add_argument("--paper", action="store_true", help="Paper mode: never send trade actions (always returns 0;0;0) but logs signals")
    parser.add_argument("--log", default="./bridge_v31_signals.csv", help="CSV log file for signals")
    parser.add_argument(
        "--disable-signal-log",
        action="store_true",
        help="Disable signal CSV logging (useful for long backtests to avoid huge logs).",
    )
    parser.add_argument(
        "--log-max-mb",
        type=float,
        default=0.0,
        help="If >0, rotate the signal log when it exceeds this size (MB).",
    )
    parser.add_argument(
        "--trades-log",
        default="./bridge_v31_trades.csv",
        help="CSV log file for closed trades (requires old cTrader format with InTrade;Pnl)",
    )
    parser.add_argument(
        "--disable-trades-log",
        action="store_true",
        help="Disable closed-trade CSV logging (reward-learning will not work).",
    )
    parser.add_argument(
        "--trades-log-max-mb",
        type=float,
        default=0.0,
        help="If >0, rotate the trades log when it exceeds this size (MB).",
    )
    parser.add_argument(
        "--mining", 
        action="store_true", 
        help="Skip model loading to generate training data.",
        )
    args = parser.parse_args()

    global PIP_VALUE
    if args.pair == "eurusd":
        PIP_VALUE = 0.0001
    if float(args.pip_value) > 0:
        PIP_VALUE = float(args.pip_value)

    params = load_best_params_from(study_name=args.study_name, storage_db=args.storage_db)
    # Allow explicit overrides (keeps live behavior aligned with training/backtest settings).
    if args.min_gap_pips is not None:
        params.min_gap_pips = float(args.min_gap_pips)
    if args.min_body_atr is not None:
        params.min_body_atr = float(args.min_body_atr)
    if args.max_fvg_age_bars is not None:
        params.max_fvg_age_bars = int(args.max_fvg_age_bars)
    if args.sl_buffer_atr is not None:
        params.sl_buffer_atr = float(args.sl_buffer_atr)
    if getattr(args, "min_sl_pips", None) is not None:
        params.min_sl_pips = _quantize_pips_0p1(float(args.min_sl_pips))
    if getattr(args, "max_sl_pips", None) is not None:
        params.max_sl_pips = _quantize_pips_0p1(float(args.max_sl_pips))
    if getattr(args, "min_tp_dist_pips", None) is not None:
        params.min_tp_dist_pips = _quantize_pips_0p1(float(args.min_tp_dist_pips))

    if getattr(args, "min_rr", None) is not None:
        params.min_rr = float(args.min_rr)
    if getattr(args, "min_rr_long", None) is not None:
        params.min_rr_long = float(args.min_rr_long)
    if getattr(args, "min_rr_short", None) is not None:
        params.min_rr_short = float(args.min_rr_short)

    base_margin = _quantize_pips_0p1(float(getattr(args, "tp_margin_pips", 0.0)))
    params.tp_margin_long_pips = (
        _quantize_pips_0p1(float(args.tp_margin_long_pips)) if args.tp_margin_long_pips is not None else float(base_margin)
    )
    params.tp_margin_short_pips = (
        _quantize_pips_0p1(float(args.tp_margin_short_pips)) if args.tp_margin_short_pips is not None else float(base_margin)
    )
    print(
        f"Live params | gap_pips={params.min_gap_pips:.2f} body_atr={params.min_body_atr:.2f} "
        f"age={params.max_fvg_age_bars} sl_buf_atr={params.sl_buffer_atr:.2f} sl_pips(min/max)={params.min_sl_pips:.1f}/{params.max_sl_pips:.1f} "
        f"tp_dist_min_pips={params.min_tp_dist_pips:.1f} "
        f"rr(min/L/S)={params.min_rr:.2f}/{params.min_rr_long:.2f}/{params.min_rr_short:.2f} "
        f"tp_margin_pips(L/S)={params.tp_margin_long_pips:.1f}/{params.tp_margin_short_pips:.1f}"
    )

    if bool(args.ifvg_zone_features) and str(args.mode) != "supervised":
        raise SystemExit("--ifvg-zone-features is supported for --mode supervised (to match the trained .pt input dim).")

    n_features = int(
        BASE_FEATURES
        + (EXTRA_TREND_SR_FEATURES if args.trend_sr else 0)
        + (EXTRA_IFVG_ZONE_FEATURES if args.ifvg_zone_features else 0)
    )
    manager: Optional[ModelManager] = None
    sup_long: Optional[SupervisedMLP] = None
    sup_short: Optional[SupervisedMLP] = None
    sup_meta: dict = {}
    sup_mtime_long = -1.0
    sup_mtime_short = -1.0

   if args.mode == "rl":
        env = DummyVecEnv([lambda: BridgeEnvV31(n_features)])
        manager = ModelManager(env, checkpoint_dir=args.checkpoint_dir)
    else:
        # --- START FIX ---
        if args.mining:
            print("⛏️ MINING MODE: Skipping model loading.")
        else:
            # Only load models if NOT mining
            try:
                sup_long, meta_l = _load_supervised_model(str(args.sup_long), model_type=getattr(args, "model_type", "mlp"), window_size=WINDOW_SIZE)
                sup_short, meta_s = _load_supervised_model(str(args.sup_short), model_type=getattr(args, "model_type", "mlp"), window_size=WINDOW_SIZE)
                sup_meta = {"long": meta_l, "short": meta_s}
                sup_mtime_long = _safe_mtime(str(args.sup_long))
                sup_mtime_short = _safe_mtime(str(args.sup_short))
            except Exception as e:
                raise SystemExit(f"Failed to load supervised models: {e!r}")

            model_in_dim = _supervised_input_dim(sup_long)
            expected_in_dim = int(n_features) * int(WINDOW_SIZE)
            
            if model_in_dim is not None and int(model_in_dim) != int(expected_in_dim):
                raise SystemExit(f"Input dim mismatch: model={model_in_dim} vs config={expected_in_dim}")
        # --- END FIX ---
    builder = LiveFeatureBuilder(
        window_size=WINDOW_SIZE,
        params=params,
        trade_start_hour=int(args.trade_start_hour),
        trade_end_hour=int(args.trade_end_hour),
        asia_start_hour=int(args.asia_start_hour),
        asia_end_hour=int(args.asia_end_hour),
        include_trend_sr=bool(args.trend_sr),
        include_ifvg_zone_features=bool(args.ifvg_zone_features),
        use_ifvg_inversion_logic=(str(args.mode) == "supervised"),
        atr_min_pips=float(args.min_atr_pips or 0.0),
        atr_max_pips=float(args.max_atr_pips or 0.0),
        trend_slope_short_max=(float(args.trend_slope_short_max) if args.trend_slope_short_max is not None else None),
        trend_slope_long_min=(float(args.trend_slope_long_min) if args.trend_slope_long_min is not None else None),
        trend_slope_min_abs=(float(args.trend_slope_min_abs) if getattr(args, "trend_slope_min_abs", None) is not None else None),
        day_trend_filter=bool(getattr(args, "day_trend_filter", False)),
        day_trend_lookback=int(getattr(args, "day_trend_lookback", 240)),
        day_trend_threshold=float(getattr(args, "day_trend_threshold", 0.0) or 0.0),
        day_trend_min_hour=int(getattr(args, "day_trend_min_hour", 0)),
        require_day_trend_for_longs=bool(getattr(args, "require_day_trend_for_longs", False)),
        require_day_trend_for_shorts=bool(getattr(args, "require_day_trend_for_shorts", False)),
        ifvg_retest_near_atr=(float(args.ifvg_retest_near_atr) if getattr(args, "ifvg_retest_near_atr", None) is not None else None),

        levels_mode=str(getattr(args, "levels_mode", "asia")),
        atr_sl_mult=float(getattr(args, "atr_sl_mult", 1.0) or 1.0),
        atr_rr=float(getattr(args, "atr_rr", 1.4) or 1.4),
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{int(args.port)}")

    print(
        f"V31 Live Bridge listening on tcp://*:{int(args.port)} | device={_device_for_m2_air()} | pair={args.pair} | "
        f"trade_hours={int(args.trade_start_hour)}-{int(args.trade_end_hour)} | "
        f"asia_hours={int(args.asia_start_hour)}-{int(args.asia_end_hour)} | hour_offset={int(args.hour_offset)} | pip={PIP_VALUE}"
    )
    if args.mode == "rl":
        print(f"Mode: rl | Checkpoint dir: {args.checkpoint_dir} (prefers brain_best.zip if present)")
    else:
        thresh_long, thresh_short = _resolve_sup_thresholds(args=args, sup_meta=sup_meta)
        m_long, m_short = _resolve_tp_margins(args=args, sup_meta=sup_meta)
        params.tp_margin_long_pips = float(m_long)
        params.tp_margin_short_pips = float(m_short)
        print(
            f"Mode: supervised | long={args.sup_long} | short={args.sup_short} | "
            f"threshold={float(args.sup_threshold):.2f} | long_thresh={thresh_long:.2f} | short_thresh={thresh_short:.2f}"
        )
        if bool(args.trend_sr) and (
            args.trend_slope_short_max is not None
            or args.trend_slope_long_min is not None
            or getattr(args, "trend_slope_min_abs", None) is not None
        ):
            print(
                "Trend gate: "
                f"short_max={args.trend_slope_short_max if args.trend_slope_short_max is not None else 'off'} "
                f"long_min={args.trend_slope_long_min if args.trend_slope_long_min is not None else 'off'}"
            )
            if getattr(args, "trend_slope_min_abs", None) is not None:
                print(f"Trend gate: min_abs={float(args.trend_slope_min_abs):.4f}")
        if bool(getattr(args, "day_trend_filter", False)):
            print(
                "Day trend filter: "
                f"lookback={int(getattr(args, 'day_trend_lookback', 0))} "
                f"th={float(getattr(args, 'day_trend_threshold', 0.0) or 0.0):.4f} "
                f"min_hour={int(getattr(args, 'day_trend_min_hour', 0))}"
            )
        if str(getattr(args, "levels_mode", "asia")) != "asia":
            print(
                "Levels: "
                f"mode={str(getattr(args, 'levels_mode', 'asia'))} "
                f"atr_sl_mult={float(getattr(args, 'atr_sl_mult', 1.0) or 1.0):.2f} "
                f"atr_rr={float(getattr(args, 'atr_rr', 1.4) or 1.4):.2f}"
            )
        if bool(getattr(args, "sup_auto_threshold", False)):
            print(f"Exec: TP margin pips (L/S)={params.tp_margin_long_pips:.1f}/{params.tp_margin_short_pips:.1f} (autoload if present)")
    print("Expected msg formats:")
    print("  new:    Open;High;Low;Close;Volume;Corr_Close;Hour")
    print("  old:    Open;High;Low;Close;Hour;InTrade;Pnl")

    # CSV logger (best-effort; should never break trading)
    log_path = args.log
    log_fieldnames = [
        "ts_unix",
        "pair",
        "model_path",
        "day_id",
        "bar_in_day",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "corr_close",
        "hour",
        "in_time",
        "in_session",
        "asia_high",
        "asia_low",
        "bias",
        "day_trend_bias",
        "day_trend_is_set",
        "day_trend_slope_atr",
        "atr_pips",
        "trend_slope_atr",
        "atr_ok",
        "allow_long",
        "allow_short",
        "action_model",
        "action_sent",
        "suppressed_by_disable",
        "levels_ok",
        "sl",
        "tp",
        "rr_req",
        "rr",
        "tp_margin_long_pips",
        "tp_margin_short_pips",
        "tp_margin_used_pips",
        "intrade",
        "pnl",
    ]

    if args.mode == "supervised":
        log_fieldnames += [
            "p_long",
            "p_short",
            "prob_gap",
            "sup_thresh",
            "sup_thresh_long",
            "sup_thresh_short",
            "sup_thresh_long_eff",
            "sup_thresh_short_eff",
            "confidence",
            "reject_reasons",
            "adaptive_on",
            "adaptive_bump",
            "adaptive_pf",
            "adaptive_loss_streak",
            "adaptive_cooldown",
            "choose_long",
            "choose_short",
        ]

    log_sink = _CsvSink(enabled=(not bool(getattr(args, "disable_signal_log", False))), label="signal log")
    if log_sink.enabled:
        _rotate_log_if_too_big(log_path, max_mb=float(getattr(args, "log_max_mb", 0.0) or 0.0))
        _rotate_log_if_schema_changed(log_path, log_fieldnames)
        log_exists = os.path.exists(log_path)
        try:
            log_f = open(log_path, "a", newline="")
            log_w = csv.DictWriter(log_f, fieldnames=log_fieldnames)
            log_sink.file = log_f
            log_sink.writer = log_w
            if not log_exists:
                log_w.writeheader()
                log_f.flush()
        except OSError as e:
            if getattr(e, "errno", None) in (errno.ENOSPC, 28):
                log_sink.enabled = False
                print("⚠️  signal log disabled: no space left on device. Continuing without logging.")
            else:
                log_sink.enabled = False
                print(f"⚠️  signal log disabled: failed to open log file: {e!r}")
        except Exception as e:
            log_sink.enabled = False
            print(f"⚠️  signal log disabled: failed to open log file: {e!r}")

    # Closed-trade logger (best-effort; only meaningful if cTrader sends old format fields)
    trades_log_path = args.trades_log
    trades_fieldnames = [
        "ts_open_unix",
        "ts_close_unix",
        "pair",
        "direction",
        "pnl",
        "win",
        "sl",
        "tp",
        "day_id_open",
        "bar_in_day_open",
        "day_id_close",
        "bar_in_day_close",
    ]
    trades_sink = _CsvSink(enabled=(not bool(getattr(args, "disable_trades_log", False))), label="trades log")
    if trades_sink.enabled:
        _rotate_log_if_too_big(trades_log_path, max_mb=float(getattr(args, "trades_log_max_mb", 0.0) or 0.0))
        _rotate_log_if_schema_changed(trades_log_path, trades_fieldnames)
        trades_exists = os.path.exists(trades_log_path)
        try:
            trades_f = open(trades_log_path, "a", newline="")
            trades_w = csv.DictWriter(trades_f, fieldnames=trades_fieldnames)
            trades_sink.file = trades_f
            trades_sink.writer = trades_w
            if not trades_exists:
                trades_w.writeheader()
                trades_f.flush()
        except OSError as e:
            if getattr(e, "errno", None) in (errno.ENOSPC, 28):
                trades_sink.enabled = False
                print("⚠️  trades log disabled: no space left on device. Continuing without trade logging.")
            else:
                trades_sink.enabled = False
                print(f"⚠️  trades log disabled: failed to open trades log: {e!r}")
        except Exception as e:
            trades_sink.enabled = False
            print(f"⚠️  trades log disabled: failed to open trades log: {e!r}")

    # Track cTrader-reported trade state (old format only)
    last_intrade: Optional[int] = None
    open_trade: Optional[dict] = None
    last_trade_pnl: Optional[float] = None
    pending_close_trade: Optional[dict] = None
    # When flat, cTrader sends last CLOSED trade PnL on every bar.
    # If a trade opens+closes within the same bar, InTrade may remain 0 forever.
    # Detect lastClosedPnl changes while flat to avoid missing those trades.
    last_flat_closed_pnl: Optional[float] = None
    last_sent_action: int = 0
    last_sent_sl: float = 0.0
    last_sent_tp: float = 0.0
    last_sent_bar_index: Optional[int] = None
    # Daily entry limit (by builder day_id)
    trades_taken_by_day: dict[int, int] = {}

    adaptive = _AdaptiveQualityGate(
        enabled=bool(getattr(args, "adaptive_threshold", False)) and args.mode == "supervised",
        window_trades=int(getattr(args, "adaptive_window_trades", 30) or 30),
        loss_streak_trigger=int(getattr(args, "adaptive_loss_streak_trigger", 3) or 3),
        bump_per_loss=float(getattr(args, "adaptive_bump_per_loss", 0.03) or 0.03),
        bump_max=float(getattr(args, "adaptive_bump_max", 0.12) or 0.12),
        cooldown_bars=int(getattr(args, "adaptive_cooldown_bars", 0) or 0),
        reset_after_bars=int(getattr(args, "adaptive_reset_after_bars", 0) or 0),
    )

    # Ensure buffered logs flush on normal process exit.
    try:
        atexit.register(log_sink.close)
        atexit.register(trades_sink.close)
    except Exception:
        pass

    while True:
        try:
            msg = socket.recv_string()
        except zmq.error.ZMQError as e:
            # Don't crash the bridge on transient ZMQ errors (e.g., bad client state).
            print(f"⚠️  ZMQ recv error (continuing): {e!r}")
            time.sleep(0.1)
            continue
        try:
            parts = [float(x.replace(",", ".")) for x in msg.split(";")]
            if len(parts) < 5:
                socket.send_string("0;0;0")
                continue

            # Heuristic support for the old cTrader format.
            # If parts[4] looks like an hour and parts[5] is 0/1, treat as old format.
            open_, high, low, close = parts[0], parts[1], parts[2], parts[3]
            volume = 0.0
            corr_close = 0.0
            hour = int(parts[4])
            intrade: Optional[int] = None
            pnl: Optional[float] = None

            if len(parts) >= 7:
                maybe_hour = int(parts[4])
                maybe_intrade = int(parts[5])
                if 0 <= maybe_hour <= 23 and (maybe_intrade == 0 or maybe_intrade == 1):
                    hour = maybe_hour
                    intrade = int(maybe_intrade)
                    try:
                        pnl = float(parts[6])
                    except Exception:
                        pnl = None
                else:
                    volume = float(parts[4])
                    corr_close = float(parts[5])
                    hour = int(parts[6])

            hour = int((hour + int(args.hour_offset)) % 24)

            row = {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "Corr_Close": corr_close,
                "Hour": hour,
            }

            builder.add_bar(row)
            obs, mask = builder.get_obs_and_mask()
            bar_index = int(getattr(builder, "_global_bar_index", 0))

            if obs is None:
                # Log a "waiting" row if needed, or just send 0
                socket.send_string("0;0;0")
                continue

            # Hot-reload supervised weights if enabled (Skipped during Mining).
            if (not args.mining) and args.mode == "supervised" and bool(args.sup_hot_reload):
                sup_long, sup_short, meta_new, sup_mtime_long, sup_mtime_short = _maybe_hot_reload_supervised(
                    sup_long_path=str(args.sup_long),
                    sup_short_path=str(args.sup_short),
                    sup_long=sup_long,
                    sup_short=sup_short,
                    last_mtime_long=float(sup_mtime_long),
                    last_mtime_short=float(sup_mtime_short),
                    expected_input_dim=int(obs[0].reshape(1, -1).shape[1]),
                    model_type=str(getattr(args, "model_type", "mlp")),
                    window_size=int(WINDOW_SIZE),
                )
                if meta_new:
                    sup_meta = meta_new

            # --- INFERENCE LOGIC (With Mining Fix) ---
            p_long = 0.0
            p_short = 0.0
            confidence = 0.0
            prob_gap = 0.0
            adaptive_bump = 0.0
            adaptive_pf = float("nan")
            adaptive_cooldown = False
            action = 0

            # 1. MINING MODE: Skip all AI, just fall through to logging
            if args.mining:
                action = 0
            
            # 2. RL MODE
            elif args.mode == "rl":
                assert manager is not None
                action = manager.predict(obs, mask)
            
            # 3. SUPERVISED MODE
            else:
                # Ensure models are loaded (they might be None if mining check failed upstream)
                if sup_long is not None and sup_short is not None:
                    x = th.tensor(obs[0].reshape(1, -1), dtype=th.float32)
                    with th.no_grad():
                        p_long = float(th.sigmoid(sup_long(x)).item())
                        p_short = float(th.sigmoid(sup_short(x)).item())

                    confidence = float(max(p_long, p_short))
                    prob_gap = float(abs(float(p_long) - float(p_short)))

                    allow_l = bool(mask[0, 1])
                    allow_s = bool(mask[0, 2])
                    thresh_long, thresh_short = _resolve_sup_thresholds(args=args, sup_meta=sup_meta)

                    thresh_long_eff = float(thresh_long)
                    thresh_short_eff = float(thresh_short)

                    # Adaptive quality gate
                    if bool(getattr(args, "adaptive_threshold", False)):
                        adaptive_bump = float(adaptive.bump(bar_index=bar_index))
                        adaptive_pf = float(adaptive.recent_pf())
                        adaptive_cooldown = bool(adaptive.in_cooldown(bar_index=bar_index))
                        thresh_long_eff = float(min(0.99, max(0.0, float(thresh_long) + float(adaptive_bump))))
                        thresh_short_eff = float(min(0.99, max(0.0, float(thresh_short) + float(adaptive_bump))))

                    choose_long = allow_l and (p_long >= thresh_long_eff)
                    choose_short = allow_s and (p_short >= thresh_short_eff)

                    if choose_long and choose_short:
                        action = 1 if p_long >= p_short else 2
                    elif choose_long:
                        action = 1
                    elif choose_short:
                        action = 2
                    else:
                        action = 0

                    # Filters
                    try:
                        min_conf = float(getattr(args, "sup_min_confidence", 0.0) or 0.0)
                    except Exception:
                        min_conf = 0.0
                    if float(confidence) < float(min_conf):
                        action = 0

                    # Prob Gap Filter
                    try:
                        min_gap_base = float(getattr(args, "sup_min_prob_gap", 0.0) or 0.0)
                    except Exception:
                        min_gap_base = 0.0
                    min_gap_long = min_gap_base
                    min_gap_short = min_gap_base
                    
                    if getattr(args, "sup_min_prob_gap_long", None) is not None:
                        min_gap_long = max(min_gap_long, float(args.sup_min_prob_gap_long))
                    if getattr(args, "sup_min_prob_gap_short", None) is not None:
                        min_gap_short = max(min_gap_short, float(args.sup_min_prob_gap_short))

                    if int(action) == 1 and min_gap_long > 0.0:
                        if abs(float(p_long) - float(p_short)) < float(min_gap_long):
                            action = 0
                    if int(action) == 2 and min_gap_short > 0.0:
                        if abs(float(p_long) - float(p_short)) < float(min_gap_short):
                            action = 0

                    if adaptive_cooldown:
                        action = 0

            # Safety: if model returns an entry action but mask disallows it, hold.
            if action in (1, 2) and not bool(mask[0, action]):
                action = 0

            sl = 0.0
            # ... (rest of loop continues)
            tp = 0.0
            entry = 0.0
            levels = None
            if action in (1, 2):
                levels = builder.get_trade_levels(action)
                if levels is not None:
                    entry, sl, tp = levels

            rr_req = ""
            rr = ""
            if int(action) in (1, 2):
                min_rr = float(getattr(params, "min_rr", 0.0) or 0.0)
                min_rr_long = float(getattr(params, "min_rr_long", 0.0) or 0.0)
                min_rr_short = float(getattr(params, "min_rr_short", 0.0) or 0.0)
                eff = float(min_rr)
                if int(action) == 1 and min_rr_long > 0.0:
                    eff = max(eff, float(min_rr_long))
                if int(action) == 2 and min_rr_short > 0.0:
                    eff = max(eff, float(min_rr_short))
                rr_req = f"{eff:.3f}"
                if levels is not None:
                    risk = abs(float(entry) - float(sl)) / float(PIP_VALUE)
                    reward = abs(float(tp) - float(entry)) / float(PIP_VALUE)
                    rr = ("" if risk <= 1e-9 else f"{(reward / risk):.3f}")

            # What we actually send to cTrader.
            action_model = int(action)
            action_sent = int(action_model)
            suppressed_by_disable = False
            if args.paper:
                action_sent = 0
            if action_sent != 0 and levels is None:
                action_sent = 0

            # Optional: suppress one direction (useful if one side is clearly unprofitable).
            if int(action_sent) == 1 and bool(getattr(args, "disable_longs", False)):
                action_sent = 0
                suppressed_by_disable = True
            if int(action_sent) == 2 and bool(getattr(args, "disable_shorts", False)):
                action_sent = 0
                suppressed_by_disable = True

            # Safety: never allow an entry with missing/invalid protection levels.
            if action_sent in (1, 2):
                try:
                    sl_f = float(sl)
                    tp_f = float(tp)
                except Exception:
                    sl_f = float("nan")
                    tp_f = float("nan")

                if (not np.isfinite(sl_f)) or (not np.isfinite(tp_f)):
                    action_sent = 0
                elif abs(sl_f) < 1e-12 or abs(tp_f) < 1e-12:
                    action_sent = 0
                else:
                    # Basic direction sanity: SL/TP should straddle entry price.
                    close_f = float(row["Close"])
                    if int(action_sent) == 1:  # long
                        if not (sl_f < close_f < tp_f):
                            action_sent = 0
                    elif int(action_sent) == 2:  # short
                        if not (tp_f < close_f < sl_f):
                            action_sent = 0

            # Optional: enforce max trades per day (entries only).
            if action_sent in (1, 2) and int(args.max_trades_per_day) > 0:
                cur_day_id = int(getattr(builder, "_day_id", 0))
                taken = int(trades_taken_by_day.get(cur_day_id, 0))
                if taken >= int(args.max_trades_per_day):
                    action_sent = 0
                else:
                    trades_taken_by_day[cur_day_id] = taken + 1

            # Build a compact explanation for why an entry was suppressed.
            reject_reasons: list[str] = []
            if args.mode == "supervised":
                if int(action_model) in (1, 2) and int(action_sent) == 0:
                    # Direction disabled
                    if bool(suppressed_by_disable):
                        reject_reasons.append("disabled_dir")

                    # Missing/invalid levels
                    if levels is None:
                        reject_reasons.append("no_levels")
                    else:
                        try:
                            sl_f = float(sl)
                            tp_f = float(tp)
                            if (not np.isfinite(sl_f)) or (not np.isfinite(tp_f)) or abs(sl_f) < 1e-12 or abs(tp_f) < 1e-12:
                                reject_reasons.append("bad_levels")
                        except Exception:
                            reject_reasons.append("bad_levels")

                    # RR gate
                    try:
                        if rr_req != "" and rr != "":
                            if float(rr) + 1e-9 < float(rr_req):
                                reject_reasons.append("rr")
                    except Exception:
                        pass

                    # Daily cap
                    if int(args.max_trades_per_day) > 0:
                        try:
                            cur_day_id = int(getattr(builder, "_day_id", 0))
                            taken = int(trades_taken_by_day.get(cur_day_id, 0))
                            if taken >= int(args.max_trades_per_day):
                                reject_reasons.append("daily_cap")
                        except Exception:
                            pass

                    # Prob-gap gate (if enabled)
                    try:
                        min_gap_base = float(getattr(args, "sup_min_prob_gap", 0.0) or 0.0)
                    except Exception:
                        min_gap_base = 0.0
                    eff_gap = float(min_gap_base)
                    try:
                        if int(action_model) == 1 and getattr(args, "sup_min_prob_gap_long", None) is not None:
                            eff_gap = max(eff_gap, float(args.sup_min_prob_gap_long))
                    except Exception:
                        pass
                    try:
                        if int(action_model) == 2 and getattr(args, "sup_min_prob_gap_short", None) is not None:
                            eff_gap = max(eff_gap, float(args.sup_min_prob_gap_short))
                    except Exception:
                        pass
                    if eff_gap > 0.0 and float(prob_gap) < float(eff_gap):
                        reject_reasons.append("prob_gap")

                    # Daily trend block (if enabled)
                    try:
                        if int(action_model) == 1 and bool(getattr(builder, "last_day_trend_block_long", False)):
                            reject_reasons.append("day_trend")
                        if int(action_model) == 2 and bool(getattr(builder, "last_day_trend_block_short", False)):
                            reject_reasons.append("day_trend")
                    except Exception:
                        pass

                    # Daily trend requirement (if enabled)
                    try:
                        if int(action_model) == 1 and bool(getattr(builder, "last_day_trend_req_block_long", False)):
                            reject_reasons.append("day_trend_req")
                        if int(action_model) == 2 and bool(getattr(builder, "last_day_trend_req_block_short", False)):
                            reject_reasons.append("day_trend_req")
                    except Exception:
                        pass

                    # Adaptive cooldown
                    if bool(adaptive_cooldown):
                        reject_reasons.append("adaptive_cooldown")

                # If the model is below thresholds, that's still useful to know.
                if int(action_model) == 0:
                    if not bool(mask[0, 1]) and not bool(mask[0, 2]):
                        reject_reasons.append("mask_block")
                    else:
                        reject_reasons.append("below_thresh")

            # Track last sent intent so we can attribute subsequent InTrade/PnL to an entry direction.
            if action_sent in (1, 2) and levels is not None:
                last_sent_action = int(action_sent)
                last_sent_sl = float(sl)
                last_sent_tp = float(tp)
                last_sent_bar_index = int(bar_index)

            # Raw time window (independent of retest gating).
            bar_hour = int(row["Hour"])
            if int(args.trade_start_hour) == int(args.trade_end_hour):
                in_time = True
            elif int(args.trade_start_hour) < int(args.trade_end_hour):
                in_time = (bar_hour >= int(args.trade_start_hour)) and (bar_hour < int(args.trade_end_hour))
            else:
                in_time = (bar_hour >= int(args.trade_start_hour)) or (bar_hour < int(args.trade_end_hour))

            # Log every processed bar with model suggestion.
            log_sink.write_row(
                {
                    "ts_unix": f"{time.time():.3f}",
                    "pair": args.pair,
                    "model_path": (manager.current_model_path if manager is not None else "supervised"),
                    "day_id": int(getattr(builder, "_day_id", 0)),
                    "bar_in_day": int(getattr(builder, "_bar_in_day", 0)),
                    "open": f"{float(row['Open']):.8f}",
                    "high": f"{float(row['High']):.8f}",
                    "low": f"{float(row['Low']):.8f}",
                    "close": f"{float(row['Close']):.8f}",
                    "volume": f"{float(row.get('Volume', 0.0)):.2f}",
                    "corr_close": f"{float(row.get('Corr_Close', 0.0)):.8f}",
                    "hour": int(row["Hour"]),
                    "in_time": bool(in_time),
                    "in_session": bool(mask[0, 1] or mask[0, 2]),
                    "asia_high": f"{float(builder.asia_high) if np.isfinite(builder.asia_high) else 0.0:.8f}",
                    "asia_low": f"{float(builder.asia_low) if np.isfinite(builder.asia_low) else 0.0:.8f}",
                    "bias": int(builder.day_bias),
                    "day_trend_bias": int(getattr(builder, "day_trend_bias", 0)),
                    "day_trend_is_set": bool(getattr(builder, "day_trend_is_set", False)),
                    "day_trend_slope_atr": (
                        "" if not np.isfinite(getattr(builder, "day_trend_slope_atr", float("nan"))) else f"{float(builder.day_trend_slope_atr):.4f}"
                    ),
                    "atr_pips": ("" if not np.isfinite(getattr(builder, "last_atr_pips", float("nan"))) else f"{float(builder.last_atr_pips):.4f}"),
                    "trend_slope_atr": (
                        "" if not np.isfinite(getattr(builder, "last_trend_slope_atr", float("nan"))) else f"{float(builder.last_trend_slope_atr):.4f}"
                    ),
                    "atr_ok": bool(getattr(builder, "atr_ok", True)),
                    "allow_long": bool(mask[0, 1]),
                    "allow_short": bool(mask[0, 2]),
                    "action_model": int(action_model),
                    "action_sent": int(action_sent),
                    "suppressed_by_disable": bool(suppressed_by_disable),
                    "levels_ok": bool(levels is not None),
                    "sl": f"{float(sl):.8f}",
                    "tp": f"{float(tp):.8f}",
                    "rr_req": rr_req,
                    "rr": rr,
                    "tp_margin_long_pips": f"{float(_quantize_pips_0p1(getattr(params, 'tp_margin_long_pips', 0.0))):.1f}",
                    "tp_margin_short_pips": f"{float(_quantize_pips_0p1(getattr(params, 'tp_margin_short_pips', 0.0))):.1f}",
                    "tp_margin_used_pips": (
                        f"{float(_quantize_pips_0p1(getattr(params, 'tp_margin_long_pips', 0.0))):.1f}"
                        if int(action_model) == 1
                        else (
                            f"{float(_quantize_pips_0p1(getattr(params, 'tp_margin_short_pips', 0.0))):.1f}"
                            if int(action_model) == 2
                            else "0.0"
                        )
                    ),
                    "intrade": ("" if intrade is None else int(intrade)),
                    "pnl": ("" if pnl is None else f"{float(pnl):.4f}"),
                    **(
                        {
                            "p_long": f"{p_long:.4f}",
                            "p_short": f"{p_short:.4f}",
                            "prob_gap": f"{float(prob_gap):.4f}",
                            "sup_thresh": f"{float(args.sup_threshold):.4f}",
                            "sup_thresh_long": f"{float(thresh_long):.4f}",
                            "sup_thresh_short": f"{float(thresh_short):.4f}",
                            "sup_thresh_long_eff": f"{float(thresh_long_eff):.4f}",
                            "sup_thresh_short_eff": f"{float(thresh_short_eff):.4f}",
                            "confidence": f"{float(confidence):.4f}",
                            "reject_reasons": "|".join(reject_reasons),
                            "adaptive_on": bool(getattr(args, "adaptive_threshold", False)),
                            "adaptive_bump": f"{float(adaptive_bump):.4f}",
                            "adaptive_pf": ("" if not np.isfinite(adaptive_pf) else f"{float(adaptive_pf):.4f}"),
                            "adaptive_loss_streak": int(getattr(adaptive, "loss_streak", 0)),
                            "adaptive_cooldown": bool(adaptive_cooldown),
                            "choose_long": bool(choose_long) if args.mode == "supervised" else False,
                            "choose_short": bool(choose_short) if args.mode == "supervised" else False,
                        }
                        if args.mode == "supervised"
                        else {}
                    ),
                }
            )

            # If cTrader provides old-format InTrade/PnL, log closed trades for reward-based learning.
            if intrade is not None:
                just_logged_close = False

                # Keep the last finite in-trade PnL so we can recover if the close bar reports 0.
                if int(intrade) == 1 and pnl is not None:
                    try:
                        pnl_f = float(pnl)
                        if np.isfinite(pnl_f):
                            last_trade_pnl = pnl_f
                    except Exception:
                        pass

                if last_intrade is None:
                    last_intrade = int(intrade)

                # If we start receiving bars while already in-trade, create an open_trade so the
                # eventual close can still be logged.
                if int(last_intrade) == 1 and int(intrade) == 1 and open_trade is None:
                    open_trade = {
                        "ts_open_unix": float(time.time()),
                        "pair": args.pair,
                        "direction": int(last_sent_action) if int(last_sent_action) in (1, 2) else 0,
                        "sl": float(last_sent_sl),
                        "tp": float(last_sent_tp),
                        "day_id_open": int(getattr(builder, "_day_id", 0)),
                        "bar_in_day_open": int(getattr(builder, "_bar_in_day", 0)),
                    }
                    last_trade_pnl = None

                # 0 -> 1 means a trade opened
                if int(last_intrade) == 0 and int(intrade) == 1:
                    open_trade = {
                        "ts_open_unix": float(time.time()),
                        "pair": args.pair,
                        "direction": int(last_sent_action) if int(last_sent_action) in (1, 2) else 0,
                        "sl": float(last_sent_sl),
                        "tp": float(last_sent_tp),
                        "day_id_open": int(getattr(builder, "_day_id", 0)),
                        "bar_in_day_open": int(getattr(builder, "_bar_in_day", 0)),
                    }
                    last_trade_pnl = None

                # 1 -> 0 means a trade closed
                if int(last_intrade) == 1 and int(intrade) == 0 and open_trade is not None:
                    # Prefer logging the close using cTrader's flat lastClosedPnl (pnl when intrade==0).
                    # The close bar often reports 0.0 while the next flat bar reports the real lastClosedPnl.
                    pnl_f: Optional[float] = None
                    if pnl is not None:
                        try:
                            pnl_try = float(pnl)
                            if np.isfinite(pnl_try) and abs(pnl_try) > 1e-12:
                                pnl_f = pnl_try
                        except Exception:
                            pnl_f = None

                    # If the close bar reports 0.0 (common), fall back to last in-trade PnL.
                    if pnl_f is None and last_trade_pnl is not None:
                        try:
                            pnl_f = float(last_trade_pnl)
                        except Exception:
                            pnl_f = None

                    if pnl_f is not None:
                        adaptive.on_trade_closed(float(pnl_f), bar_index=bar_index)
                        trades_sink.write_row(
                            {
                                "ts_open_unix": f"{float(open_trade.get('ts_open_unix', 0.0)):.3f}",
                                "ts_close_unix": f"{time.time():.3f}",
                                "pair": str(open_trade.get("pair", args.pair)),
                                "direction": int(open_trade.get("direction", 0)),
                                "pnl": f"{pnl_f:.4f}",
                                "win": bool(pnl_f > 0.0),
                                "sl": f"{float(open_trade.get('sl', 0.0)):.8f}",
                                "tp": f"{float(open_trade.get('tp', 0.0)):.8f}",
                                "day_id_open": int(open_trade.get("day_id_open", 0)),
                                "bar_in_day_open": int(open_trade.get("bar_in_day_open", 0)),
                                "day_id_close": int(getattr(builder, "_day_id", 0)),
                                "bar_in_day_close": int(getattr(builder, "_bar_in_day", 0)),
                            }
                        )
                        just_logged_close = True
                        last_flat_closed_pnl = float(pnl_f)
                    else:
                        # Defer close logging until we observe a new lastClosedPnl while flat.
                        pending_close_trade = dict(open_trade)

                    open_trade = None
                    last_trade_pnl = None

                # If flat and the reported PnL changes, that's a new closed trade (cBot uses
                # lastClosedPnl when flat). This catches trades that open+close within a bar.
                if int(intrade) == 0 and pnl is not None and not just_logged_close:
                    try:
                        pnl_closed = float(pnl)
                    except Exception:
                        pnl_closed = float("nan")
                    if np.isfinite(pnl_closed):
                        # Only attribute if we recently sent an action; avoids accidental attribution
                        # if other positions are present.
                        recent = False
                        try:
                            if last_sent_bar_index is not None:
                                recent = (int(bar_index) - int(last_sent_bar_index)) <= 2
                        except Exception:
                            recent = False

                        # If we deferred a 1->0 close, always prefer using that open_trade metadata.
                        if pending_close_trade is not None and recent:
                            adaptive.on_trade_closed(float(pnl_closed), bar_index=bar_index)
                            trades_sink.write_row(
                                {
                                    "ts_open_unix": f"{float(pending_close_trade.get('ts_open_unix', 0.0)):.3f}",
                                    "ts_close_unix": f"{time.time():.3f}",
                                    "pair": str(pending_close_trade.get("pair", args.pair)),
                                    "direction": int(pending_close_trade.get("direction", 0)),
                                    "pnl": f"{float(pnl_closed):.4f}",
                                    "win": bool(float(pnl_closed) > 0.0),
                                    "sl": f"{float(pending_close_trade.get('sl', 0.0)):.8f}",
                                    "tp": f"{float(pending_close_trade.get('tp', 0.0)):.8f}",
                                    "day_id_open": int(pending_close_trade.get("day_id_open", 0)),
                                    "bar_in_day_open": int(pending_close_trade.get("bar_in_day_open", 0)),
                                    "day_id_close": int(getattr(builder, "_day_id", 0)),
                                    "bar_in_day_close": int(getattr(builder, "_bar_in_day", 0)),
                                }
                            )
                            pending_close_trade = None
                            last_flat_closed_pnl = float(pnl_closed)
                        else:
                            if last_flat_closed_pnl is None:
                                last_flat_closed_pnl = float(pnl_closed)
                            elif abs(float(pnl_closed) - float(last_flat_closed_pnl)) > 1e-9:
                                if recent:
                                    adaptive.on_trade_closed(float(pnl_closed), bar_index=bar_index)
                                    trades_sink.write_row(
                                        {
                                            "ts_open_unix": f"{time.time():.3f}",
                                            "ts_close_unix": f"{time.time():.3f}",
                                            "pair": args.pair,
                                            "direction": int(last_sent_action) if int(last_sent_action) in (1, 2) else 0,
                                            "pnl": f"{float(pnl_closed):.4f}",
                                            "win": bool(float(pnl_closed) > 0.0),
                                            "sl": f"{float(last_sent_sl):.8f}",
                                            "tp": f"{float(last_sent_tp):.8f}",
                                            "day_id_open": int(getattr(builder, "_day_id", 0)),
                                            "bar_in_day_open": int(getattr(builder, "_bar_in_day", 0)),
                                            "day_id_close": int(getattr(builder, "_day_id", 0)),
                                            "bar_in_day_close": int(getattr(builder, "_bar_in_day", 0)),
                                        }
                                    )

                                last_flat_closed_pnl = float(pnl_closed)

                last_intrade = int(intrade)

                if action_sent == 0:
                    if bool(getattr(args, "send_confidence", False)):
                        socket.send_string(f"0;0;0;{float(confidence):.4f}")
                    else:
                        socket.send_string("0;0;0")
                    continue

                if bool(getattr(args, "send_confidence", False)):
                    socket.send_string(f"{action_sent};{float(sl):.5f};{float(tp):.5f};{float(confidence):.4f}")
                else:
                    socket.send_string(f"{action_sent};{float(sl):.5f};{float(tp):.5f}")

        except Exception as e:
            print(f"❌ {e!r} | msg={msg}")
            try:
                socket.send_string("0;0;0")
            except Exception:
                pass


if __name__ == "__main__":
    main()
