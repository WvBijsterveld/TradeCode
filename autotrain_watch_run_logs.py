#!/usr/bin/env python3
"""Always-on auto-training loop driven by backtest run_logs.

Watches a directory (default: ./run_logs) for new pairs of files:
  *_signals.csv and the matching *_trades.csv

When a new run appears and both files are stable (no size/mtime changes for
--quiet-seconds), it snapshots them and trains via supervised_train_from_bridge_v1.py.

This lets you:
- Keep running cTrader backtests normally (each run writes its own timestamped logs)
- Continuously retrain/deploy supervised models from the latest completed run

Notes
- This script does NOT run backtests; it only reacts to logs.
- By default it deploys to ./ai_supervised_v1 (same place the bridge loads from).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class FileState:
    path: Path
    size: int
    mtime: float


def _stat(path: Path) -> Optional[FileState]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return FileState(path=path, size=int(st.st_size), mtime=float(st.st_mtime))


def _is_stable_pair(signals: Path, trades: Optional[Path], *, quiet_seconds: int) -> bool:
    """Return True if file(s) have been unchanged for quiet_seconds."""
    now = time.time()
    s = _stat(signals)
    if s is None:
        return False

    # We store the last-seen state on the FileState itself by reading cached info from a sidecar.
    # But to keep this simple/robust, use mtime age as our stability proxy.
    if (now - s.mtime) < float(quiet_seconds):
        return False

    if trades is not None:
        t = _stat(trades)
        if t is None:
            return False
        if (now - t.mtime) < float(quiet_seconds):
            return False

    return True


def _find_latest_run(run_logs: Path) -> Optional[Tuple[Path, Optional[Path]]]:
    signals_files = sorted(run_logs.glob("*_signals.csv"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for s in signals_files:
        # Prefer matching trades file with same stem prefix.
        # Convention in our runner scripts: ..._<ts>_signals.csv / ..._<ts>_trades.csv
        trades = s.with_name(s.name.replace("_signals.csv", "_trades.csv"))
        if trades.exists():
            return s, trades
        # If no trades file exists, still allow training (sim labels only).
        return s, None
    return None


def _load_state(state_path: Path) -> dict:
    try:
        if not state_path.exists():
            return {}
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, state_path)


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--run-logs-dir", default="./run_logs", help="Directory containing *_signals.csv and *_trades.csv")
    p.add_argument("--state", default="./run_logs/.autotrain_state.json", help="State file to avoid retraining the same run")
    p.add_argument("--archive-dir", default="./run_logs/autotrain_snapshots", help="Where to snapshot inputs + trainer logs")

    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--quiet-seconds", type=int, default=120)

    # Trainer config (forwarded to supervised_train_from_bridge_v1.py)
    p.add_argument("--pair", choices=["eurusd", "eurjpy"], default="eurusd")
    p.add_argument("--out-dir", default="./ai_supervised_v1", help="Deploy directory for .pt outputs")

    p.add_argument("--trend-sr", action="store_true")
    p.add_argument("--ifvg-zone-features", action="store_true")
    p.add_argument("--window-size", type=int, default=30)

    p.add_argument("--asia-start-hour", type=int, default=0)
    p.add_argument("--asia-end-hour", type=int, default=8)
    p.add_argument("--trade-start-hour", type=int, default=8)
    p.add_argument("--trade-end-hour", type=int, default=16)

    p.add_argument("--min-gap-pips", type=float, default=2.8753)
    p.add_argument("--min-body-atr", type=float, default=0.8793)
    p.add_argument("--max-fvg-age-bars", type=int, default=118)
    p.add_argument("--sl-buffer-atr", type=float, default=0.5378)
    p.add_argument("--tp-margin-pips", type=float, default=0.0)
    p.add_argument("--tp-margin-max-pips", type=float, default=3.0)
    p.add_argument("--max-hold-bars", type=int, default=288)

    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", default="auto", help="auto|cpu|mps")
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--interop-threads", type=int, default=1)

    p.add_argument("--val-days", type=int, default=3)
    p.add_argument("--deploy-min-improve", type=float, default=0.0)
    p.add_argument("--tail-days", type=int, default=0)
    p.add_argument("--tail-rows", type=int, default=0)

    p.add_argument("--executed-trade-weight", type=float, default=5.0)
    p.add_argument("--reward-scale-win", type=float, default=0.0)
    p.add_argument("--reward-scale-loss", type=float, default=0.0)

    p.add_argument("--no-tune-thresholds", action="store_true")
    p.add_argument("--no-tune-tp-margin", action="store_true")
    p.add_argument("--fp-cost", type=float, default=3.0)
    p.add_argument("--min-val-trades", type=int, default=5)

    args = p.parse_args()

    root = Path(__file__).resolve().parent
    run_logs = (root / str(args.run_logs_dir)).resolve()
    state_path = (root / str(args.state)).resolve()
    archive_root = (root / str(args.archive_dir)).resolve()

    run_logs.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    last_processed = str(state.get("last_processed_signals", "") or "")

    print("autotrain_watch_run_logs starting")
    print(f"  run_logs_dir={run_logs}")
    print(f"  state={state_path}")
    print(f"  archive_dir={archive_root}")
    print(f"  last_processed={last_processed or '<none>'}")

    while True:
        try:
            latest = _find_latest_run(run_logs)
            if latest is None:
                time.sleep(float(args.poll_seconds))
                continue

            signals, trades = latest

            if str(signals) == last_processed:
                time.sleep(float(args.poll_seconds))
                continue

            if not _is_stable_pair(signals, trades, quiet_seconds=int(args.quiet_seconds)):
                time.sleep(float(args.poll_seconds))
                continue

            snap_dir = archive_root / f"snap_{args.pair}_{_ts()}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_signals = snap_dir / signals.name
            snap_signals.write_bytes(signals.read_bytes())

            snap_trades = None
            if trades is not None and trades.exists():
                snap_trades = snap_dir / trades.name
                snap_trades.write_bytes(trades.read_bytes())

            trainer_log = snap_dir / "trainer.out"

            cmd = [
                sys.executable,
                str(root / "supervised_train_from_bridge_v1.py"),
                "--bridge-log",
                str(snap_signals),
                "--pair",
                str(args.pair),
                "--out-dir",
                str((root / str(args.out_dir)).resolve()),
                "--asia-start-hour",
                str(int(args.asia_start_hour)),
                "--asia-end-hour",
                str(int(args.asia_end_hour)),
                "--trade-start-hour",
                str(int(args.trade_start_hour)),
                "--trade-end-hour",
                str(int(args.trade_end_hour)),
                "--min-gap-pips",
                str(float(args.min_gap_pips)),
                "--min-body-atr",
                str(float(args.min_body_atr)),
                "--max-fvg-age-bars",
                str(int(args.max_fvg_age_bars)),
                "--sl-buffer-atr",
                str(float(args.sl_buffer_atr)),
                "--tp-margin-pips",
                str(float(args.tp_margin_pips)),
                "--tp-margin-max-pips",
                str(float(args.tp_margin_max_pips)),
                "--max-hold-bars",
                str(int(args.max_hold_bars)),
                "--window-size",
                str(int(args.window_size)),
                "--epochs",
                str(int(args.epochs)),
                "--lr",
                str(float(args.lr)),
                "--batch",
                str(int(args.batch)),
                "--seed",
                str(int(args.seed)),
                "--device",
                str(args.device),
                "--num-threads",
                str(int(args.num_threads)),
                "--interop-threads",
                str(int(args.interop_threads)),
                "--val-days",
                str(int(args.val_days)),
                "--deploy-min-improve",
                str(float(args.deploy_min_improve)),
                "--tail-days",
                str(int(args.tail_days)),
                "--tail-rows",
                str(int(args.tail_rows)),
                "--executed-trade-weight",
                str(float(args.executed_trade_weight)),
                "--reward-scale-win",
                str(float(args.reward_scale_win)),
                "--reward-scale-loss",
                str(float(args.reward_scale_loss)),
                "--fp-cost",
                str(float(args.fp_cost)),
                "--min-val-trades",
                str(int(args.min_val_trades)),
            ]

            if bool(args.trend_sr):
                cmd.append("--trend-sr")
            if bool(args.ifvg_zone_features):
                cmd.append("--ifvg-zone-features")
            if bool(args.no_tune_thresholds):
                cmd.append("--no-tune-thresholds")
            if bool(args.no_tune_tp_margin):
                cmd.append("--no-tune-tp-margin")
            if snap_trades is not None:
                cmd += ["--trades-log", str(snap_trades)]

            print(f"\nNew stable run detected:\n  signals={signals.name}\n  trades={(trades.name if trades else '<none>')}\n  snapshot={snap_dir}")
            print("Launching trainer:")
            print("  " + " ".join(cmd))

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"

            with trainer_log.open("w") as lf:
                proc = subprocess.Popen(cmd, cwd=str(root), stdout=lf, stderr=subprocess.STDOUT, env=env)
                rc = proc.wait()

            print(f"Trainer finished rc={rc} log={trainer_log}")

            # Mark as processed regardless of rc to avoid infinite retries on a bad run.
            last_processed = str(signals)
            state["last_processed_signals"] = last_processed
            state["last_snapshot_dir"] = str(snap_dir)
            state["last_trainer_log"] = str(trainer_log)
            state["last_trainer_rc"] = int(rc)
            _save_state(state_path, state)

        except Exception as e:
            print(f"loop error: {e!r}")

        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
