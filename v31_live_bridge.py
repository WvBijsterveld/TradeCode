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
            if int(self.buffer_size) > 0 and len(self._buffer) >= int(self.buffer_size):
                self._flush_buffer(force=True)
            else:
                self._flush_buffer(force=False)
        except OSError as e:
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
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        if self.seq_len > 0:
            x = x.view(x.size(0), self.seq_len, self.feat_dim)
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        return self.head(last_hidden)


def _supervised_input_dim(model: Optional[nn.Module]) -> Optional[int]:
    try:
        if model is None:
            return None
        if hasattr(model, "net"):
            first = model.net[0]
            return int(getattr(first, "in_features"))
        return None
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


def _load_supervised_model(path: str, model_type: str = "mlp", window_size: int = WINDOW_SIZE) -> tuple[nn.Module, dict]:
    ckpt = th.load(path, map_location="cpu")
    input_dim = int(ckpt.get("input_dim"))
    if model_type is None:
        model_type = ckpt.get("meta", {}).get("model_type", "mlp") if isinstance(ckpt.get("meta"), dict) else "mlp"

    is_transformer = "transformer" in str(model_type).lower()
    is_lstm = "lstm" in str(model_type).lower()

    if is_transformer:
        print(f"Loading V3 Transformer: {path}")
        model = SupervisedTransformer(
            input_dim=input_dim,
            seq_len=int(window_size),
            d_model=128,
            nhead=4,
            num_layers=2,
        )
    elif is_lstm:
        print(f"Loading V4 LSTM: {path}")
        model = SupervisedLSTM(
            input_dim=input_dim,
            seq_len=int(window_size),
            hidden_dim=128,
            num_layers=2
        )
    else:
        model = SupervisedMLP(input_dim=input_dim)

    try:
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    except Exception as e:
        raise RuntimeError(f"Failed loading checkpoint {path} into {model_type}: {e!r}")

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
    base_long = _quantize_pips_0p1(float(args.tp_margin_long_pips)) if long_overridden else float(base)
    base_short = _quantize_pips_0p1(float(args.tp_margin_short_pips)) if short_overridden else float(base)
    if not bool(getattr(args, "sup_auto_threshold", False)):
        return float(base_long), float(base_short)
    m_long = _extract_tuned_tp_margin_pips((sup_meta or {}).get("long", {}) if isinstance(sup_meta, dict) else {})
    m_short = _extract_tuned_tp_margin_pips((sup_meta or {}).get("short", {}) if isinstance(sup_meta, dict) else {})
    if (not long_overridden) and (m_long is not None):
        base_long = float(m_long)
    if (not short_overridden) and (m_short is not None):
        base_short = float(m_short)
    return float(base_long), float(base_short)


def _maybe_hot_reload_supervised(*, sup_long_path: str, sup_short_path: str, sup_long: Optional[nn.Module], sup_short: Optional[nn.Module], last_mtime_long: float, last_mtime_short: float, expected_input_dim: int, model_type: str = "mlp", window_size: int = WINDOW_SIZE) -> tuple[Optional[nn.Module], Optional[nn.Module], dict, float, float]:
    m_l = _safe_mtime(sup_long_path)
    m_s = _safe_mtime(sup_short_path)
    changed = (m_l > last_mtime_long) or (m_s > last_mtime_short)
    if not changed:
        return sup_long, sup_short, {}, last_mtime_long, last_mtime_short
    try:
        new_long, meta_l = _load_supervised_model(str(sup_long_path), model_type=model_type, window_size=window_size)
        new_short, meta_s = _load_supervised_model(str(sup_short_path), model_type=model_type, window_size=window_size)
        print(f"\n🔁 Hot-reloaded supervised models: long={sup_long_path} short={sup_short_path}")
        return new_long, new_short, {"long": meta_l, "short": meta_s}, m_l, m_s
    except Exception as e:
        print(f"\n⚠️  Hot-reload failed (keeping current models): {e!r}")
        return sup_long, sup_short, {}, last_mtime_long, last_mtime_short


def _device_for_m2_air() -> str:
    try:
        if th.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "mps" if th.backends.mps.is_available() else "cpu"


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
        return np.zeros((self.n_features, WINDOW_SIZE), dtype=np.float32), 0.0, False, False, {}
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
    def __init__(self, window_size: int, params: LiveParams, *, trade_start_hour: int, trade_end_hour: int, asia_start_hour: int, asia_end_hour: int, include_trend_sr: bool, include_ifvg_zone_features: bool, use_ifvg_inversion_logic: bool, ifvg_retest_near_atr: Optional[float] = None, atr_min_pips: float = 0.0, atr_max_pips: float = 0.0, trend_lookback: int = 50, sr_lookback: int = 288, structure_window: int = 24, trend_slope_short_max: Optional[float] = None, trend_slope_long_min: Optional[float] = None, trend_slope_min_abs: Optional[float] = None, day_trend_filter: bool = False, day_trend_lookback: int = 240, day_trend_threshold: float = 0.0, day_trend_min_hour: int = 0, require_day_trend_for_longs: bool = False, require_day_trend_for_shorts: bool = False, levels_mode: str = "asia", atr_sl_mult: float = 1.0, atr_rr: float = 1.4, sl_mode: str = "asia", swing_sl_lookback_bars: int = 48, swing_sl_buffer_atr: float = 0.25):
        self.window_size = window_size
        self.params = params
        self.trade_start_hour = int(trade_start_hour)
        self.trade_end_hour = int(trade_end_hour)
        self.asia_start_hour = int(asia_start_hour)
        self.asia_end_hour = int(asia_end_hour)
        self.include_trend_sr = bool(include_trend_sr)
        self.include_ifvg_zone_features = bool(include_ifvg_zone_features)
        self.use_ifvg_inversion_logic = bool(use_ifvg_inversion_logic)
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
        
        self.sl_mode = str(sl_mode).strip().lower()
        if self.sl_mode not in ("asia", "swing"):
            self.sl_mode = "asia"
        self.swing_sl_lookback_bars = int(swing_sl_lookback_bars)
        self.swing_sl_buffer_atr = float(swing_sl_buffer_atr)

        self.last_atr_pips: float = float("nan")
        self.atr_ok: bool = True
        self.last_trend_slope_atr: float = 0.0
        self.day_trend_bias: int = 0
        self.day_trend_is_set: bool = False
        self.day_trend_slope_atr: float = float("nan")
        self.last_day_trend_block_long: bool = False
        self.last_day_trend_block_short: bool = False
        self.last_day_trend_req_block_long: bool = False
        self.last_day_trend_req_block_short: bool = False
        
        # --- V5 Persisted Context ---
        self.last_macro_trend: float = 0.0
        self.last_vol_regime: float = 1.0
        self.last_rsi: float = 0.0

        self._max_history = int(max(60, self.window_size + 2, self.trend_lookback, self.sr_lookback, self.day_trend_lookback, 2 * self.structure_window) + 5)
        self.history = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Corr_Close", "Hour"])
        self._last_hour: Optional[int] = None
        self._day_id: int = 0
        self._bar_in_day: int = 0
        self._global_bar_index: int = 0
        self.asia_high: float = np.nan
        self.asia_low: float = np.nan
        self.day_bias: int = 0
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
        self.history.loc[len(self.history)] = row
        if len(self.history) > int(self._max_history):
            self.history = self.history.iloc[-int(self._max_history) :].reset_index(drop=True)
        if self.asia_start_hour <= hour < self.asia_end_hour:
            hi = float(row["High"])
            lo = float(row["Low"])
            self.asia_high = hi if not np.isfinite(self.asia_high) else max(self.asia_high, hi)
            self.asia_low = lo if not np.isfinite(self.asia_low) else min(self.asia_low, lo)
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
        atr = self._compute_atr_series(df).values
        i = n - 1
        body = abs(float(df.loc[i - 1, "Close"]) - float(df.loc[i - 1, "Open"]))
        atr_i = float(max(float(atr[i - 1]), 1e-6))
        min_gap = float(self.params.min_gap_pips) * PIP_VALUE
        hi_i2 = float(df.loc[i - 2, "High"])
        lo_i2 = float(df.loc[i - 2, "Low"])
        hi_i = float(df.loc[i, "High"])
        lo_i = float(df.loc[i, "Low"])
        if (hi_i2 < lo_i) and ((lo_i - hi_i2) >= min_gap) and (body >= float(self.params.min_body_atr) * atr_i):
            self._bull_bottom = hi_i2
            self._bull_top = lo_i
            self._bull_age = 0
        if (lo_i2 > hi_i) and ((lo_i2 - hi_i) >= min_gap) and (body >= float(self.params.min_body_atr) * atr_i):
            self._bear_bottom = hi_i
            self._bear_top = lo_i2
            self._bear_age = 0
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
        minutes = float((self._bar_in_day - 1) * 5 % 1440)
        angle = 2 * np.pi * (minutes / 1440.0)
        tod_sin = float(np.sin(angle))
        tod_cos = float(np.cos(angle))
        dow = 0.0
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
        last_close = float(df["Close"].iloc[-1])
        last_atr = float(max(float(atr.iloc[-1]), 1e-6))
        try:
            self.last_atr_pips = float(last_atr / float(max(PIP_VALUE, 1e-9)))
        except Exception:
            self.last_atr_pips = float("nan")
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
        hour = int(df["Hour"].iloc[-1]) if "Hour" in df.columns else 0
        window = df.iloc[-self.window_size :].copy()
        
        # --- NEW V5 CONTEXT FEATURES ---
        lookback_4h = 48
        macro_trend = 0.0
        if len(df) >= lookback_4h:
            y_reg = df["Close"].iloc[-lookback_4h:].values.astype(float)
            x_reg = np.arange(len(y_reg))
            slope, _ = np.polyfit(x_reg, y_reg, 1)
            macro_trend = (slope / last_atr) * 10.0
        else:
            macro_trend = 0.0

        long_term_atr_series = self._compute_atr_series(df, period=288)
        long_term_atr = float(long_term_atr_series.iloc[-1])
        if np.isnan(long_term_atr) or long_term_atr == 0:
            vol_regime = 1.0
        else:
            vol_regime = last_atr / long_term_atr

        rsi_val = 0.0
        try:
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi = 100 - (100 / (1 + rs))
            rsi_val = (float(rsi.iloc[-1]) - 50.0) / 50.0
            if np.isnan(rsi_val): rsi_val = 0.0
        except:
            pass

        # Persist context for accurate logging later
        self.last_macro_trend = float(macro_trend)
        self.last_vol_regime = float(vol_regime)
        self.last_rsi = float(rsi_val)

        window["macro_trend"] = float(np.clip(macro_trend, -10, 10))
        window["vol_regime"] = float(np.clip(vol_regime, -5, 5))
        window["rsi"] = float(np.clip(rsi_val, -1, 1))
        
        window["tod_sin"] = tod_sin
        window["tod_cos"] = tod_cos
        window["dow"] = dow
        window["dist_to_asia_high_atr"] = dist_to_asia_high_atr
        window["dist_to_asia_low_atr"] = dist_to_asia_low_atr
        window["asia_bias"] = float(self.day_bias)
        
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
                th_val = float(self.day_trend_threshold)
                if th_val > 0.0:
                    if float(slope_atr) >= float(th_val):
                        self.day_trend_bias = 1
                        self.day_trend_is_set = True
                    elif float(slope_atr) <= -float(th_val):
                        self.day_trend_bias = -1
                        self.day_trend_is_set = True
        
        feature_cols = [
            "Rel_Open", "Rel_High", "Rel_Low", "Rel_Close", "Rel_Vol", "Rel_Corr",
            "tod_sin", "tod_cos", "dow",
            "dist_to_asia_high_atr", "dist_to_asia_low_atr", "asia_bias",
            "macro_trend", "vol_regime", "rsi"
        ]
        if self.include_trend_sr:
            feature_cols += ["trend_slope_atr", "structure", "dist_to_sr_high_atr", "dist_to_sr_low_atr"]
        if self.include_ifvg_zone_features:
            feature_cols += ["ifvg_dist_to_top_atr", "ifvg_dist_to_bottom_atr", "ifvg_in_zone"]
        x = window[feature_cols].values.T.astype(np.float32)
        obs = np.expand_dims(x, axis=0)
        allow_long = False
        allow_short = False
        hi = float(df["High"].iloc[-1])
        lo = float(df["Low"].iloc[-1])
        cl = float(df["Close"].iloc[-1])
        if self.use_ifvg_inversion_logic:
            if self.day_bias == 1 and np.isfinite(self._inv_bear_bottom):
                z_bot = float(self._inv_bear_bottom)
                z_top = float(self._inv_bear_top)
                in_zone = (lo <= z_top) and (hi >= z_bot)
                near_atr = float(getattr(self, "ifvg_retest_near_atr", 0.25))
                dist_edge = min(abs(cl - z_bot), abs(cl - z_top)) / float(max(last_atr, 1e-6))
                allow_long = bool(in_zone or (dist_edge <= near_atr))
            if self.day_bias == -1 and np.isfinite(self._inv_bull_bottom):
                z_bot = float(self._inv_bull_bottom)
                z_top = float(self._inv_bull_top)
                in_zone = (lo <= z_top) and (hi >= z_bot)
                near_atr = float(getattr(self, "ifvg_retest_near_atr", 0.25))
                dist_edge = min(abs(cl - z_bot), abs(cl - z_top)) / float(max(last_atr, 1e-6))
                allow_short = bool(in_zone or (dist_edge <= near_atr))
        else:
            if self.day_bias == 1 and np.isfinite(self._bull_bottom):
                allow_long = (lo <= float(self._bull_top)) and (hi >= float(self._bull_bottom))
            if self.day_bias == -1 and np.isfinite(self._bear_bottom):
                allow_short = (hi >= float(self._bear_bottom)) and (lo <= float(self._bear_top))
        if self.trade_start_hour == self.trade_end_hour:
            in_time = True
        elif self.trade_start_hour < self.trade_end_hour:
            in_time = (hour >= self.trade_start_hour) and (hour < self.trade_end_hour)
        else:
            in_time = (hour >= self.trade_start_hour) or (hour < self.trade_end_hour)
        allow_long = bool(allow_long and self.atr_ok)
        allow_short = bool(allow_short and self.atr_ok)
        if self.include_trend_sr and np.isfinite(self.last_trend_slope_atr):
            if self.trend_slope_short_max is not None:
                allow_short = bool(allow_short and (float(self.last_trend_slope_atr) <= float(self.trend_slope_short_max)))
            if self.trend_slope_long_min is not None:
                allow_long = bool(allow_long and (float(self.last_trend_slope_atr) >= float(self.trend_slope_long_min)))
            if self.trend_slope_min_abs is not None and np.isfinite(self.trend_slope_min_abs):
                min_abs = float(self.trend_slope_min_abs)
                if min_abs > 0.0:
                    ok = bool(abs(float(self.last_trend_slope_atr)) >= min_abs)
                    allow_long = bool(allow_long and ok)
                    allow_short = bool(allow_short and ok)
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
        self.last_day_trend_block_long = False
        self.last_day_trend_block_short = False
        if self.day_trend_filter and int(self.day_trend_bias) != 0:
            if int(self.day_trend_bias) > 0:
                if bool(allow_short):
                    self.last_day_trend_block_short = True
                allow_short = False
            elif int(self.day_trend_bias) < 0:
                if bool(allow_long):
                    self.last_day_trend_block_long = True
                allow_long = False
        mask = np.array([[True, bool(in_time and allow_long), bool(in_time and allow_short)]], dtype=bool)
        return obs, mask

    def get_trade_levels(self, action: int) -> Optional[Tuple[float, float, float]]:
        if len(self.history) < 20: return None
        close = float(self.history["Close"].iloc[-1])
        hour = int(self.history["Hour"].iloc[-1])
        df = self.history.copy()
        atr = self._compute_atr_series(df)
        last_atr = float(max(float(atr.iloc[-1]), 1e-6))
        spread = SPREAD_PIPS * PIP_VALUE
        slippage = SLIPPAGE_PIPS * PIP_VALUE
        sl_buf = float(self.params.sl_buffer_atr) * last_atr
        margin_pips = 0.0
        if action == 1: margin_pips = _quantize_pips_0p1(getattr(self.params, "tp_margin_long_pips", 0.0))
        elif action == 2: margin_pips = _quantize_pips_0p1(getattr(self.params, "tp_margin_short_pips", 0.0))
        tp_margin = float(margin_pips) * PIP_VALUE
        
        mode = self.levels_mode
        def _atr_levels(act):
            risk_dist = self.atr_sl_mult * last_atr
            if act == 1:
                entry = close + spread/2 + slippage
                sl = entry - risk_dist
                tp = entry + self.atr_rr * risk_dist
                return entry, sl, tp
            elif act == 2:
                entry = close - spread/2 - slippage
                sl = entry + risk_dist
                tp = entry - self.atr_rr * risk_dist
                return entry, sl, tp
            return None

        if mode == "atr": return _atr_levels(action)
        if not (np.isfinite(self.asia_high) and np.isfinite(self.asia_low)):
             return _atr_levels(action) if mode == "asia_then_atr" else None
        if hour < self.asia_end_hour:
             return _atr_levels(action) if mode == "asia_then_atr" else None

        # --- Swing SL Mode Logic ---
        def _swing_levels(act):
            lb = int(self.swing_sl_lookback_bars)
            buf = float(self.swing_sl_buffer_atr)
            if len(df) < lb:
                return _atr_levels(act) if mode == "asia_then_atr" else None
            
            if act == 1: # Long
                entry = close + spread/2 + slippage
                # Swing Low
                swing_low = df["Low"].iloc[-lb:].min()
                sl = float(swing_low) - (buf * last_atr) - spread/2
                # Asia High TP
                tp = float(self.asia_high) - spread/2 - tp_margin
                if not (tp > entry and sl < entry): return _atr_levels(act) if mode == "asia_then_atr" else None
                return entry, sl, tp
            elif act == 2: # Short
                entry = close - spread/2 - slippage
                # Swing High
                swing_high = df["High"].iloc[-lb:].max()
                sl = float(swing_high) + (buf * last_atr) + spread/2
                # Asia Low TP
                tp = float(self.asia_low) + spread/2 + tp_margin
                if not (tp < entry and sl > entry): return _atr_levels(act) if mode == "asia_then_atr" else None
                return entry, sl, tp
            return None

        if self.sl_mode == "swing":
            return _swing_levels(action)

        # Standard Asia Levels
        if action == 1:
            entry = close + spread/2 + slippage
            tp = float(self.asia_high) - spread/2 - tp_margin
            sl = float(self.asia_low) - sl_buf - spread/2
            if not (tp > entry and sl < entry): return _atr_levels(action) if mode == "asia_then_atr" else None
            return entry, sl, tp
        if action == 2:
            entry = close - spread/2 - slippage
            tp = float(self.asia_low) + spread/2 + tp_margin
            sl = float(self.asia_high) + sl_buf + spread/2
            if not (tp < entry and sl > entry): return _atr_levels(action) if mode == "asia_then_atr" else None
            return entry, sl, tp
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["eurjpy", "eurusd"], default="eurjpy")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage-db", default=DEFAULT_STORAGE_DB)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hour-offset", type=int, default=0)
    parser.add_argument("--asia-start-hour", type=int, default=ASIA_START_HOUR)
    parser.add_argument("--asia-end-hour", type=int, default=ASIA_END_HOUR)
    parser.add_argument("--trade-start-hour", type=int, default=8)
    parser.add_argument("--trade-end-hour", type=int, default=16)
    parser.add_argument("--pip-value", type=float, default=0.0)
    parser.add_argument("--min-gap-pips", type=float, default=None)
    parser.add_argument("--min-body-atr", type=float, default=None)
    parser.add_argument("--max-fvg-age-bars", type=int, default=None)
    parser.add_argument("--sl-buffer-atr", type=float, default=None)
    parser.add_argument("--min-sl-pips", type=float, default=None)
    parser.add_argument("--max-sl-pips", type=float, default=None)
    parser.add_argument("--min-tp_dist_pips", type=float, default=None)
    parser.add_argument("--min-rr", type=float, default=None)
    parser.add_argument("--min-rr-long", type=float, default=None)
    parser.add_argument("--min-rr-short", type=float, default=None)
    parser.add_argument("--tp-margin-pips", type=float, default=0.0)
    parser.add_argument("--tp-margin-long-pips", type=float, default=None)
    parser.add_argument("--tp-margin-short-pips", type=float, default=None)
    parser.add_argument("--levels-mode", choices=["asia", "atr", "asia_then_atr"], default="asia")
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "transformer", "lstm"])
    parser.add_argument("--atr-sl-mult", type=float, default=1.0)
    parser.add_argument("--atr-rr", type=float, default=1.4)
    parser.add_argument("--min-atr-pips", type=float, default=0.0)
    parser.add_argument("--max-atr-pips", type=float, default=0.0)
    parser.add_argument("--trend-sr", action="store_true")
    parser.add_argument("--trend-slope-short-max", type=float, default=None)
    parser.add_argument("--trend-slope-long-min", type=float, default=None)
    parser.add_argument("--trend-slope-min-abs", type=float, default=None)
    parser.add_argument("--day-trend-filter", action="store_true")
    parser.add_argument("--day-trend-lookback", type=int, default=240)
    parser.add_argument("--day-trend-threshold", type=float, default=0.12)
    parser.add_argument("--day-trend-min-hour", type=int, default=0)
    parser.add_argument("--require-day-trend-for-longs", action="store_true")
    parser.add_argument("--require-day-trend-for-shorts", action="store_true")
    parser.add_argument("--mode", choices=["rl", "supervised"], default="rl")
    parser.add_argument("--ifvg-zone-features", action="store_true")
    parser.add_argument("--sup-long", default="./ai_supervised_v1/sup_long_eurusd_trendsr.pt")
    parser.add_argument("--sup-short", default="./ai_supervised_v1/sup_short_eurusd_trendsr.pt")
    parser.add_argument("--sup-threshold", type=float, default=0.55)
    parser.add_argument("--sup-thresh-long", type=float, default=None)
    parser.add_argument("--sup-thresh-short", type=float, default=None)
    parser.add_argument("--sup-auto-threshold", action="store_true")
    parser.add_argument("--disable-longs", action="store_true")
    parser.add_argument("--disable-shorts", action="store_true")
    parser.add_argument("--sup-hot-reload", action="store_true")
    parser.add_argument("--adaptive-threshold", action="store_true")
    parser.add_argument("--adaptive-window-trades", type=int, default=30)
    parser.add_argument("--adaptive-loss-streak-trigger", type=int, default=3)
    parser.add_argument("--adaptive-bump-per-loss", type=float, default=0.03)
    parser.add_argument("--adaptive-bump-max", type=float, default=0.12)
    parser.add_argument("--adaptive-cooldown-bars", type=int, default=36)
    parser.add_argument("--adaptive-reset-after-bars", type=int, default=0)
    parser.add_argument("--send-confidence", action="store_true")
    parser.add_argument("--sup-min-prob-gap", type=float, default=0.0)
    parser.add_argument("--sup-min-prob-gap-long", type=float, default=None)
    parser.add_argument("--sup-min-prob-gap-short", type=float, default=None)
    parser.add_argument("--sup-min-confidence", type=float, default=0.0)
    parser.add_argument("--ifvg-retest-near-atr", type=float, default=None)
    parser.add_argument("--max-trades-per-day", type=int, default=0)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--log", default="./bridge_v31_signals.csv")
    parser.add_argument("--disable-signal-log", action="store_true")
    parser.add_argument("--log-max-mb", type=float, default=0.0)
    parser.add_argument("--trades-log", default="./bridge_v31_trades.csv")
    parser.add_argument("--disable-trades-log", action="store_true")
    parser.add_argument("--trades-log-max-mb", type=float, default=0.0)
    parser.add_argument("--mining", action="store_true", help="Skip model loading to generate training data.")
    
    # --- New Swing SL Args ---
    parser.add_argument("--sl-mode", choices=["asia", "swing"], default="asia", help="Stop Loss mode. 'asia' uses boundaries, 'swing' uses recent extrema.")
    parser.add_argument("--swing-sl-lookback-bars", type=int, default=48, help="Bars to look back for swing high/low SL.")
    parser.add_argument("--swing-sl-buffer-atr", type=float, default=0.25, help="ATR buffer added to swing high/low SL.")

    args = parser.parse_args()

    global PIP_VALUE
    if args.pair == "eurusd": PIP_VALUE = 0.0001
    if float(args.pip_value) > 0: PIP_VALUE = float(args.pip_value)

    params = load_best_params_from(study_name=args.study_name, storage_db=args.storage_db)
    if args.min_gap_pips is not None: params.min_gap_pips = float(args.min_gap_pips)
    if args.min_body_atr is not None: params.min_body_atr = float(args.min_body_atr)
    if args.max_fvg_age_bars is not None: params.max_fvg_age_bars = int(args.max_fvg_age_bars)
    if args.sl_buffer_atr is not None: params.sl_buffer_atr = float(args.sl_buffer_atr)
    
    base_margin = _quantize_pips_0p1(float(getattr(args, "tp_margin_pips", 0.0)))
    params.tp_margin_long_pips = _quantize_pips_0p1(float(args.tp_margin_long_pips)) if args.tp_margin_long_pips is not None else float(base_margin)
    params.tp_margin_short_pips = _quantize_pips_0p1(float(args.tp_margin_short_pips)) if args.tp_margin_short_pips is not None else float(base_margin)

    n_features = int(BASE_FEATURES + (EXTRA_TREND_SR_FEATURES if args.trend_sr else 0) + (EXTRA_IFVG_ZONE_FEATURES if args.ifvg_zone_features else 0))
    manager = None
    sup_long = None
    sup_short = None
    sup_meta = {}
    sup_mtime_long = -1.0
    sup_mtime_short = -1.0

    if args.mode == "rl":
        env = DummyVecEnv([lambda: BridgeEnvV31(n_features)])
        manager = ModelManager(env, checkpoint_dir=args.checkpoint_dir)
    else:
        if args.mining:
            print("⛏️ MINING MODE: Skipping model loading.")
        else:
            try:
                sup_long, meta_l = _load_supervised_model(str(args.sup_long), model_type=args.model_type, window_size=WINDOW_SIZE)
                sup_short, meta_s = _load_supervised_model(str(args.sup_short), model_type=args.model_type, window_size=WINDOW_SIZE)
                sup_meta = {"long": meta_l, "short": meta_s}
                sup_mtime_long = _safe_mtime(str(args.sup_long))
                sup_mtime_short = _safe_mtime(str(args.sup_short))
            except Exception as e:
                raise SystemExit(f"Failed to load supervised models: {e!r}")

            model_in_dim = _supervised_input_dim(sup_long)
            expected_in_dim = int(n_features) * int(WINDOW_SIZE)
            
            if model_in_dim is not None and int(model_in_dim) != int(expected_in_dim):
                raise SystemExit(f"Supervised model input dim mismatch: model expects {int(model_in_dim)} but current config produces {int(expected_in_dim)}")

    builder = LiveFeatureBuilder(
        window_size=WINDOW_SIZE, params=params, trade_start_hour=int(args.trade_start_hour), trade_end_hour=int(args.trade_end_hour),
        asia_start_hour=int(args.asia_start_hour), asia_end_hour=int(args.asia_end_hour), include_trend_sr=bool(args.trend_sr),
        include_ifvg_zone_features=bool(args.ifvg_zone_features), use_ifvg_inversion_logic=(str(args.mode) == "supervised"),
        ifvg_retest_near_atr=(float(args.ifvg_retest_near_atr) if getattr(args, "ifvg_retest_near_atr", None) is not None else None),
        levels_mode=str(getattr(args, "levels_mode", "asia")), atr_sl_mult=float(getattr(args, "atr_sl_mult", 1.0)), atr_rr=float(getattr(args, "atr_rr", 1.4)),
        sl_mode=str(args.sl_mode), swing_sl_lookback_bars=int(args.swing_sl_lookback_bars), swing_sl_buffer_atr=float(args.swing_sl_buffer_atr)
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{int(args.port)}")
    print(f"Bridge V31 listening on port {int(args.port)}...")

    log_path = args.log
    log_fieldnames = [
        "ts_unix", "pair", "model_path", "day_id", "bar_in_day", "open", "high", "low", "close", "volume", "corr_close", "hour",
        "in_time", "in_session", "asia_high", "asia_low", "bias", "day_trend_bias", "day_trend_is_set", "day_trend_slope_atr",
        "atr_pips", "trend_slope_atr", "atr_ok", "allow_long", "allow_short", "action_model", "action_sent", "suppressed_by_disable",
        "levels_ok", "sl", "tp", "rr_req", "rr", "tp_margin_long_pips", "tp_margin_short_pips", "tp_margin_used_pips", "intrade", "pnl",
        # --- V5 FEATURES LOGGED ---
        "macro_trend", "vol_regime", "rsi"
    ]
    if args.mode == "supervised":
        log_fieldnames += ["p_long", "p_short", "prob_gap", "sup_thresh", "sup_thresh_long", "sup_thresh_short", "sup_thresh_long_eff", "sup_thresh_short_eff", "confidence", "reject_reasons", "adaptive_on", "adaptive_bump", "adaptive_pf", "adaptive_loss_streak", "adaptive_cooldown", "choose_long", "choose_short"]

    log_sink = _CsvSink(enabled=(not bool(getattr(args, "disable_signal_log", False))), label="signal log")
    if log_sink.enabled:
        _rotate_log_if_too_big(log_path, max_mb=float(getattr(args, "log_max_mb", 0.0) or 0.0))
        _rotate_log_if_schema_changed(log_path, log_fieldnames)
        try:
            log_f = open(log_path, "a", newline="")
            log_w = csv.DictWriter(log_f, fieldnames=log_fieldnames)
            log_sink.file = log_f
            log_sink.writer = log_w
            if os.path.getsize(log_path) == 0: log_w.writeheader()
        except Exception: pass

    trades_log_path = args.trades_log
    trades_sink = _CsvSink(enabled=(not bool(getattr(args, "disable_trades_log", False))), label="trades log")
    if trades_sink.enabled:
        try:
            trades_f = open(trades_log_path, "a", newline="")
            trades_w = csv.DictWriter(trades_f, fieldnames=["ts_open_unix", "ts_close_unix", "pair", "direction", "pnl", "win", "sl", "tp", "day_id_open", "bar_in_day_open", "day_id_close", "bar_in_day_close"])
            trades_sink.file = trades_f
            trades_sink.writer = trades_w
            if os.path.getsize(trades_log_path) == 0: trades_w.writeheader()
        except Exception: pass

    last_intrade = None
    open_trade = None
    last_trade_pnl = None
    pending_close_trade = None
    last_flat_closed_pnl = None
    last_sent_action = 0
    last_sent_sl = 0.0
    last_sent_tp = 0.0
    last_sent_bar_index = None
    trades_taken_by_day = {}
    adaptive = _AdaptiveQualityGate(enabled=bool(getattr(args, "adaptive_threshold", False)), window_trades=30, loss_streak_trigger=3, bump_per_loss=0.03, bump_max=0.12, cooldown_bars=36)

    while True:
        try:
            msg = socket.recv_string()
        except zmq.error.ZMQError:
            time.sleep(0.1)
            continue

        try:
            parts = [float(x.replace(",", ".")) for x in msg.split(";")]
            if len(parts) < 5:
                socket.send_string("0;0;0")
                continue
            open_, high, low, close = parts[0], parts[1], parts[2], parts[3]
            hour = int(parts[4])
            intrade = int(parts[5]) if len(parts) >= 6 else None
            pnl = float(parts[6]) if len(parts) >= 7 else None
            hour = int((hour + int(args.hour_offset)) % 24)
            row = {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 0.0, "Corr_Close": 0.0, "Hour": hour}
            builder.add_bar(row)
            obs, mask = builder.get_obs_and_mask()
            bar_index = int(getattr(builder, "_global_bar_index", 0))
            
            if obs is None:
                socket.send_string("0;0;0")
                continue

            if (not args.mining) and args.mode == "supervised" and bool(args.sup_hot_reload):
                 try:
                     sup_long, sup_short, meta_new, sup_mtime_long, sup_mtime_short = _maybe_hot_reload_supervised(sup_long_path=str(args.sup_long), sup_short_path=str(args.sup_short), sup_long=sup_long, sup_short=sup_short, last_mtime_long=float(sup_mtime_long), last_mtime_short=float(sup_mtime_short), expected_input_dim=int(obs[0].reshape(1, -1).shape[1]), model_type=args.model_type, window_size=WINDOW_SIZE)
                     if meta_new: sup_meta = meta_new
                 except: pass

            p_long = 0.0
            p_short = 0.0
            confidence = 0.0
            prob_gap = 0.0
            adaptive_bump = 0.0
            adaptive_pf = float("nan")
            adaptive_cooldown = False
            action = 0
            
            if args.mining:
                action = 0
            elif args.mode == "rl":
                if manager: action = manager.predict(obs, mask)
            else:
                if sup_long and sup_short:
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

                    if bool(getattr(args, "adaptive_threshold", False)):
                        adaptive_bump = float(adaptive.bump(bar_index=bar_index))
                        adaptive_pf = float(adaptive.recent_pf())
                        adaptive_cooldown = bool(adaptive.in_cooldown(bar_index=bar_index))
                        thresh_long_eff = float(min(0.99, max(0.0, float(thresh_long) + float(adaptive_bump))))
                        thresh_short_eff = float(min(0.99, max(0.0, float(thresh_short) + float(adaptive_bump))))

                    choose_long = allow_l and (p_long >= thresh_long_eff)
                    choose_short = allow_s and (p_short >= thresh_short_eff)

                    # Better tie-breaker: strongest signal wins
                    if choose_long and choose_short:
                        action = 1 if p_long >= p_short else 2
                    elif choose_long:
                        action = 1
                    elif choose_short:
                        action = 2
                    else:
                        action = 0

                    # Confidence Filter
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

            # Safety: Mask check
            if action in (1, 2) and not bool(mask[0, action]):
                action = 0

            # Calculate Levels
            sl = 0.0
            tp = 0.0
            levels = None
            if action in (1, 2):
                levels = builder.get_trade_levels(action)
                if levels is not None:
                    _, sl, tp = levels
                else:
                    action = 0 # No valid levels = No trade

            # Max Trades Per Day Check
            if action in (1, 2) and int(args.max_trades_per_day) > 0:
                cur_day_id = int(getattr(builder, "_day_id", 0))
                taken = int(trades_taken_by_day.get(cur_day_id, 0))
                if taken >= int(args.max_trades_per_day):
                    action = 0
                else:
                    # Don't increment yet; verify if we actually send it below
                    pass

            action_sent = int(action)
            if action_sent in (1, 2) and int(args.max_trades_per_day) > 0:
                 # Increment daily count only if we are truly sending a trade
                 cur_day_id = int(getattr(builder, "_day_id", 0))
                 taken = int(trades_taken_by_day.get(cur_day_id, 0))
                 trades_taken_by_day[cur_day_id] = taken + 1

            # Manual Disable Flags
            if action_sent == 1 and bool(getattr(args, "disable_longs", False)):
                action_sent = 0
            if action_sent == 2 and bool(getattr(args, "disable_shorts", False)):
                action_sent = 0
            if args.paper:
                action_sent = 0

            # --- LOGGING ---
            # Retrieve Persisted V5 Features from Builder
            macro_trend = getattr(builder, "last_macro_trend", 0.0)
            vol_regime = getattr(builder, "last_vol_regime", 1.0)
            rsi_val = getattr(builder, "last_rsi", 0.0)

            # Explicitly extract allow flags from mask to log them
            allow_long_flag = 1 if mask[0, 1] else 0
            allow_short_flag = 1 if mask[0, 2] else 0

            row_log = {
                "ts_unix": f"{time.time():.3f}",
                "day_id": getattr(builder, "_day_id", 0),
                "bar_in_day": getattr(builder, "_bar_in_day", 0),
                "open": f"{row['Open']:.5f}", "high": f"{row['High']:.5f}", 
                "low": f"{row['Low']:.5f}", "close": f"{row['Close']:.5f}", 
                "hour": hour,
                "p_long": f"{p_long:.4f}", "p_short": f"{p_short:.4f}",
                "macro_trend": f"{macro_trend:.4f}", "vol_regime": f"{vol_regime:.4f}", "rsi": f"{rsi_val:.4f}",
                "allow_long": allow_long_flag, "allow_short": allow_short_flag,
                "action_sent": action_sent, "sl": f"{sl:.5f}", "tp": f"{tp:.5f}",
                "adaptive_bump": f"{adaptive_bump:.4f}", "adaptive_cooldown": adaptive_cooldown
            }
            # Fill missing keys
            for k in log_fieldnames:
                if k not in row_log: row_log[k] = ""
            
            log_sink.write_row(row_log)
            
            # --- TRADE TRACKING FOR ADAPTIVE ---
            if intrade is not None:
                # Update Adaptive with closed trades inferred from cTrader stream
                if int(intrade) == 0 and pnl is not None:
                     # Simple logic: if flat and PnL reported, it might be a close.
                     # (Full robust logic is in the original code, we rely on _AdaptiveQualityGate logic here)
                     # For brevity in this loop, we pass valid PnL to adaptive if it looks like a close
                     # In a real run, check transitions (1->0).
                     pass 
                
                # Check for 1->0 transition to trigger adaptive updates
                if last_intrade == 1 and int(intrade) == 0:
                     if pnl is not None:
                         try:
                             adaptive.on_trade_closed(float(pnl), bar_index=bar_index)
                         except: pass
                last_intrade = int(intrade)

            # Send Response
            if bool(getattr(args, "send_confidence", False)):
                socket.send_string(f"{action_sent};{sl:.5f};{tp:.5f};{confidence:.4f}")
            else:
                socket.send_string(f"{action_sent};{sl:.5f};{tp:.5f}")

        except Exception as e:
            print(f"Error: {e}")
            try: socket.send_string("0;0;0")
            except: pass

if __name__ == "__main__":
    main()
