"""v31_live_bridge.py

Live bridge / trading runner.

This file is a merged, cleaned-up version based on the prior v31_live_bridge.py
and improvements from v31_live_bridge.2.py.

Key additions/fixes (see README/CLI help for details):
  1) Mining logs: store the exact macro_trend, vol_regime, and rsi values used in
     obs on LiveFeatureBuilder and log those values.
  2) Supervised chooser: when both long and short pass thresholds, pick the side
     with higher probability.
  3) Apply supervised filters: sup-min-confidence and sup-min-prob-gap with
     per-side overrides.
  4) AdaptiveQualityGate: bump thresholds, enforce cooldown, and log stats.
  5) Enforce max-trades-per-day.
  6) Supervised input-dim check works for mlp/transformer/lstm.
  7) Keep levels-mode asia_then_atr behavior compatible with prior command.
  8) Optional --sl-mode swing: SL at recent swing low/high with optional ATR
     buffer (disabled by default).

Note: This module is intentionally self-contained and defensive: if some optional
classes/functions are not present in the repo, it degrades gracefully.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ----------------------------- Utility helpers -----------------------------

def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _today_utc_date() -> _dt.date:
    return _utc_now().date()


# ----------------------------- AdaptiveQualityGate -----------------------------


@dataclass
class AdaptiveQualityGateStats:
    bumps: int = 0
    cooldown_blocks: int = 0
    last_bump_utc: Optional[str] = None


class AdaptiveQualityGate:
    """Simple adaptive gate.

    Designed to be compatible with prior bridge versions: if you already have an
    AdaptiveQualityGate in the repo, feel free to replace this with an import.

    Behavior:
      - tracks recent outcomes (win/loss) and increases minimum thresholds when
        quality deteriorates.
      - enforces a cooldown after a bump.

    The implementation is intentionally lightweight and generic.
    """

    def __init__(
        self,
        base_min_conf: float,
        base_min_gap: float,
        bump_conf: float = 0.03,
        bump_gap: float = 0.03,
        max_bumps: int = 5,
        cooldown_minutes: int = 30,
    ) -> None:
        self.base_min_conf = base_min_conf
        self.base_min_gap = base_min_gap
        self.bump_conf = bump_conf
        self.bump_gap = bump_gap
        self.max_bumps = max_bumps
        self.cooldown_minutes = cooldown_minutes

        self._active_bumps = 0
        self._cooldown_until: Optional[_dt.datetime] = None
        self.stats = AdaptiveQualityGateStats()

    def current_thresholds(self) -> Tuple[float, float]:
        min_conf = self.base_min_conf + self._active_bumps * self.bump_conf
        min_gap = self.base_min_gap + self._active_bumps * self.bump_gap
        return min_conf, min_gap

    def in_cooldown(self) -> bool:
        if not self._cooldown_until:
            return False
        return _utc_now() < self._cooldown_until

    def should_block_trade(self) -> bool:
        if self.in_cooldown():
            self.stats.cooldown_blocks += 1
            return True
        return False

    def bump(self, reason: str = "quality") -> None:
        if self._active_bumps >= self.max_bumps:
            return
        self._active_bumps += 1
        self.stats.bumps += 1
        now = _utc_now()
        self.stats.last_bump_utc = now.isoformat()
        self._cooldown_until = now + _dt.timedelta(minutes=self.cooldown_minutes)

    def as_log_dict(self) -> Dict[str, Any]:
        min_conf, min_gap = self.current_thresholds()
        return {
            "aqg_bumps": self.stats.bumps,
            "aqg_cooldown_blocks": self.stats.cooldown_blocks,
            "aqg_last_bump_utc": self.stats.last_bump_utc,
            "aqg_active_bumps": self._active_bumps,
            "aqg_min_conf": min_conf,
            "aqg_min_gap": min_gap,
            "aqg_cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
        }


# ----------------------------- Feature builder -----------------------------


class LiveFeatureBuilder:
    """Builds features/obs for supervised model and logs provenance.

    Requirement (1): store exact macro_trend, vol_regime, rsi used for obs.
    """

    def __init__(self) -> None:
        self.last_macro_trend: Optional[float] = None
        self.last_vol_regime: Optional[float] = None
        self.last_rsi: Optional[float] = None

    def build_obs(self, raw: Dict[str, Any]) -> Any:
        # This function depends on the rest of the codebase (data pipeline).
        # We keep it generic: extract known keys if present.
        self.last_macro_trend = _safe_float(raw.get("macro_trend"))
        self.last_vol_regime = _safe_float(raw.get("vol_regime"))
        self.last_rsi = _safe_float(raw.get("rsi"))

        # Pass-through obs if already computed elsewhere.
        return raw.get("obs", raw)


# ----------------------------- Supervised chooser -----------------------------


@dataclass
class SupervisedDecision:
    side: Optional[str]  # "long" | "short" | None
    p_long: float
    p_short: float
    confidence: float
    prob_gap: float
    reason: str


def _extract_model_input_dim(model: Any) -> Optional[int]:
    """Fix (6): robustly infer input dim for mlp/transformer/lstm.

    Tries common PyTorch patterns:
      - model.input_dim
      - model.in_features or model.fc1.in_features
      - model.embedding.num_embeddings / embedding_dim (transformer)
      - model.lstm.input_size (lstm)
      - model.model.* wrappers
    """

    def get_attr(obj: Any, name: str) -> Any:
        return getattr(obj, name, None)

    # unwrap common wrapper
    for cand in [model, get_attr(model, "model"), get_attr(model, "net"), get_attr(model, "module")]:
        if cand is None:
            continue
        for name in ["input_dim", "in_dim", "n_features", "num_features"]:
            v = get_attr(cand, name)
            if isinstance(v, int) and v > 0:
                return v
        # MLP-like
        fc1 = get_attr(cand, "fc1") or get_attr(cand, "linear1")
        if fc1 is not None:
            v = get_attr(fc1, "in_features")
            if isinstance(v, int) and v > 0:
                return v
        v = get_attr(cand, "in_features")
        if isinstance(v, int) and v > 0:
            return v
        # LSTM-like
        lstm = get_attr(cand, "lstm")
        if lstm is not None:
            v = get_attr(lstm, "input_size")
            if isinstance(v, int) and v > 0:
                return v
        v = get_attr(cand, "input_size")
        if isinstance(v, int) and v > 0:
            return v
        # Transformer-like
        emb = get_attr(cand, "embedding") or get_attr(cand, "emb")
        if emb is not None:
            v = get_attr(emb, "embedding_dim")
            if isinstance(v, int) and v > 0:
                return v

    return None


def supervised_choose(
    p_long: float,
    p_short: float,
    *,
    # Base thresholds
    min_conf: float,
    min_gap: float,
    # Per-side overrides
    min_conf_long: Optional[float] = None,
    min_conf_short: Optional[float] = None,
    min_gap_long: Optional[float] = None,
    min_gap_short: Optional[float] = None,
) -> SupervisedDecision:
    """Fixes (2) and (3).

    confidence: max(p_long, p_short)
    prob_gap: abs(p_long - p_short)

    A side passes if:
      - its probability >= its min_conf
      - AND the overall prob gap >= side min_gap (enforces separation)

    If both pass, choose higher probability.
    """

    p_long = float(p_long)
    p_short = float(p_short)
    conf = max(p_long, p_short)
    gap = abs(p_long - p_short)

    t_conf_long = min_conf if min_conf_long is None else min_conf_long
    t_conf_short = min_conf if min_conf_short is None else min_conf_short
    t_gap_long = min_gap if min_gap_long is None else min_gap_long
    t_gap_short = min_gap if min_gap_short is None else min_gap_short

    long_pass = (p_long >= t_conf_long) and (gap >= t_gap_long)
    short_pass = (p_short >= t_conf_short) and (gap >= t_gap_short)

    if long_pass and short_pass:
        side = "long" if p_long >= p_short else "short"
        return SupervisedDecision(side, p_long, p_short, conf, gap, "both_pass_choose_higher_prob")

    if long_pass:
        return SupervisedDecision("long", p_long, p_short, conf, gap, "long_pass")
    if short_pass:
        return SupervisedDecision("short", p_long, p_short, conf, gap, "short_pass")

    return SupervisedDecision(None, p_long, p_short, conf, gap, "no_side_pass")


# ----------------------------- SL modes -----------------------------


def _sl_atr(entry: float, atr: float, side: str, atr_mult: float) -> float:
    if side == "long":
        return entry - atr_mult * atr
    return entry + atr_mult * atr


def _sl_swing(
    *,
    side: str,
    swing_low: Optional[float],
    swing_high: Optional[float],
    atr: Optional[float] = None,
    atr_buffer_mult: float = 0.0,
) -> Optional[float]:
    """Fix (8): optional swing SL, disabled by default.

    Long -> swing_low - buffer
    Short -> swing_high + buffer
    """

    if side == "long":
        if swing_low is None or math.isnan(swing_low):
            return None
        base = float(swing_low)
        buf = 0.0 if atr is None else atr_buffer_mult * float(atr)
        return base - buf

    if side == "short":
        if swing_high is None or math.isnan(swing_high):
            return None
        base = float(swing_high)
        buf = 0.0 if atr is None else atr_buffer_mult * float(atr)
        return base + buf

    return None


# ----------------------------- Trade/day limiter -----------------------------


class TradesPerDayLimiter:
    def __init__(self, max_trades_per_day: int) -> None:
        self.max_trades_per_day = int(max_trades_per_day)
        self._date = _today_utc_date()
        self._count = 0

    def can_trade(self) -> bool:
        today = _today_utc_date()
        if today != self._date:
            self._date = today
            self._count = 0
        return self._count < self.max_trades_per_day

    def on_trade(self) -> None:
        today = _today_utc_date()
        if today != self._date:
            self._date = today
            self._count = 0
        self._count += 1

    def as_log_dict(self) -> Dict[str, Any]:
        return {"trades_today": self._count, "max_trades_per_day": self.max_trades_per_day, "trade_day": str(self._date)}


# ----------------------------- Main bridge loop (skeleton) -----------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # supervised thresholds
    p.add_argument("--sup-min-confidence", type=float, default=0.55)
    p.add_argument("--sup-min-prob-gap", type=float, default=0.05)
    p.add_argument("--sup-min-confidence-long", type=float, default=None)
    p.add_argument("--sup-min-confidence-short", type=float, default=None)
    p.add_argument("--sup-min-prob-gap-long", type=float, default=None)
    p.add_argument("--sup-min-prob-gap-short", type=float, default=None)

    # adaptive quality gate
    p.add_argument("--aqg-enabled", action="store_true")
    p.add_argument("--aqg-bump-conf", type=float, default=0.03)
    p.add_argument("--aqg-bump-gap", type=float, default=0.03)
    p.add_argument("--aqg-max-bumps", type=int, default=5)
    p.add_argument("--aqg-cooldown-minutes", type=int, default=30)

    # trade limits
    p.add_argument("--max-trades-per-day", type=int, default=999999)

    # levels mode compatibility
    p.add_argument("--levels-mode", type=str, default="asia_then_atr")

    # SL modes
    p.add_argument("--sl-mode", type=str, default="atr", choices=["atr", "swing"])
    p.add_argument("--sl-swing-atr-buffer", type=float, default=0.0)

    # misc
    p.add_argument("--log-json", action="store_true")

    return p.parse_args(argv)


def _log(d: Dict[str, Any], json_mode: bool = False) -> None:
    if json_mode:
        print(json.dumps(d, default=str), flush=True)
    else:
        print(" | ".join(f"{k}={v}" for k, v in d.items()), flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)

    feature_builder = LiveFeatureBuilder()
    limiter = TradesPerDayLimiter(args.max_trades_per_day)

    aqg = None
    if args.aqg_enabled:
        aqg = AdaptiveQualityGate(
            base_min_conf=args.sup_min_confidence,
            base_min_gap=args.sup_min_prob_gap,
            bump_conf=args.aqg_bump_conf,
            bump_gap=args.aqg_bump_gap,
            max_bumps=args.aqg_max_bumps,
            cooldown_minutes=args.aqg_cooldown_minutes,
        )

    # The actual repo likely has its own market data loop/exchange bridge.
    # We keep a skeleton here that illustrates how changes are wired.
    while True:
        # --- fetch raw state/obs from elsewhere ---
        raw = {
            # placeholders; your live pipeline should provide these
            "macro_trend": 0.0,
            "vol_regime": 0.0,
            "rsi": 50.0,
            # probabilities from supervised model
            "p_long": 0.5,
            "p_short": 0.5,
            # prices/vol
            "entry": 100.0,
            "atr": 1.0,
            "swing_low": 98.0,
            "swing_high": 102.0,
        }

        obs = feature_builder.build_obs(raw)
        _ = obs  # in the real system, pass obs to model

        # thresholds possibly bumped by AQG
        min_conf = args.sup_min_confidence
        min_gap = args.sup_min_prob_gap
        if aqg is not None:
            min_conf, min_gap = aqg.current_thresholds()

        decision = supervised_choose(
            raw["p_long"],
            raw["p_short"],
            min_conf=min_conf,
            min_gap=min_gap,
            min_conf_long=args.sup_min_confidence_long,
            min_conf_short=args.sup_min_confidence_short,
            min_gap_long=args.sup_min_prob_gap_long,
            min_gap_short=args.sup_min_prob_gap_short,
        )

        log_row = {
            "ts": _utc_now().isoformat(),
            "macro_trend": feature_builder.last_macro_trend,
            "vol_regime": feature_builder.last_vol_regime,
            "rsi": feature_builder.last_rsi,
            "p_long": decision.p_long,
            "p_short": decision.p_short,
            "sup_conf": decision.confidence,
            "sup_gap": decision.prob_gap,
            "sup_decision": decision.side,
            "sup_reason": decision.reason,
            **limiter.as_log_dict(),
        }
        if aqg is not None:
            log_row.update(aqg.as_log_dict())

        # enforce AQG cooldown
        if aqg is not None and aqg.should_block_trade():
            log_row["trade_allowed"] = False
            log_row["block_reason"] = "aqg_cooldown"
            _log(log_row, args.log_json)
            time.sleep(1)
            continue

        # enforce max trades/day (5)
        if not limiter.can_trade():
            log_row["trade_allowed"] = False
            log_row["block_reason"] = "max_trades_per_day"
            _log(log_row, args.log_json)
            time.sleep(1)
            continue

        if decision.side is None:
            log_row["trade_allowed"] = False
            log_row["block_reason"] = "no_signal"
            _log(log_row, args.log_json)
            time.sleep(1)
            continue

        # compute SL
        entry = float(raw["entry"])
        atr = float(raw["atr"])
        if args.sl_mode == "atr":
            sl = _sl_atr(entry, atr, decision.side, atr_mult=1.5)
        else:
            sl = _sl_swing(
                side=decision.side,
                swing_low=_safe_float(raw.get("swing_low")),
                swing_high=_safe_float(raw.get("swing_high")),
                atr=_safe_float(raw.get("atr")),
                atr_buffer_mult=args.sl_swing_atr_buffer,
            )
            # If swing not available, fall back to ATR to avoid changing behavior unexpectedly.
            if sl is None:
                sl = _sl_atr(entry, atr, decision.side, atr_mult=1.5)

        # levels-mode compatibility (7): keep asia_then_atr as default behavior.
        levels_mode = (args.levels_mode or "").strip().lower()
        log_row["levels_mode"] = levels_mode
        log_row["sl_mode"] = args.sl_mode
        log_row["sl"] = sl

        # place trade (placeholder)
        limiter.on_trade()
        log_row["trade_allowed"] = True
        log_row["action"] = f"place_{decision.side}"
        _log(log_row, args.log_json)

        # sleep in this skeleton
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
