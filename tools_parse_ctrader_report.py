#!/usr/bin/env python3
"""Extract headline stats from a cTrader HTML backtest report.

cTrader exports a self-contained HTML that embeds a JSON blob in:
  <script type="application/json" id="backtesting-report"> ... </script>

This tool prints the key metrics and tries to locate the trade statistics node.

Usage:
  python3 tools_parse_ctrader_report.py report2.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_report_json(html: str) -> Dict[str, Any]:
    m = re.search(
        r'<script\s+type="application/json"\s+id="backtesting-report">(\{.*?\})\s*</script>',
        html,
        re.S,
    )
    if not m:
        raise SystemExit("Could not find embedded report JSON")
    return json.loads(m.group(1))


def _walk_candidates(obj: Any) -> List[Tuple[str, Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any]]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            keys = set(node.keys())
            if {
                "profitFactor",
                "winRate",
                "grossProfit",
                "grossLoss",
                "totalTrades",
                "trades",
            }.intersection(keys):
                candidates.append((path, node))
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)

    walk(obj, "")
    return candidates


def _print_kv(title: str, d: Dict[str, Any], keys: List[str]) -> None:
    print(title)
    for k in keys:
        if k in d:
            print(f"  {k}: {d[k]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse cTrader backtesting HTML report")
    ap.add_argument("report", nargs="+", help="Path(s) to report*.html")
    ap.add_argument("--json", action="store_true", help="Output a single JSON object per report")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON (implies --json)")
    args = ap.parse_args()

    out_json = bool(args.json or args.pretty)
    pretty = bool(args.pretty)

    def _as_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def _ms_to_dt(ms: Any) -> str:
        try:
            return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat()
        except Exception:
            return ""

    def _ms_to_ym(ms: Any) -> str:
        try:
            dt = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
            return dt.strftime("%Y-%m")
        except Exception:
            return "unknown"

    for idx, report_path in enumerate(args.report):
        path = Path(report_path)
        if idx > 0 and not out_json:
            print("\n" + ("=" * 80) + "\n")

        html = path.read_text(encoding="utf-8", errors="ignore")
        data = _extract_report_json(html)

        main_node = data.get("main", {}) if isinstance(data, dict) else {}

        if out_json:
            stats = data.get("tradeStatistics", {}) if isinstance(data, dict) else {}
            history = data.get("history", {}) if isinstance(data, dict) else {}
            history_items = history.get("items") if isinstance(history, dict) else None
            if not isinstance(history_items, list):
                history_items = []

            first_trade = ""
            last_trade_close = ""
            by_month: dict[str, list[float]] = defaultdict(list)

            if history_items:
                try:
                    first_trade = _ms_to_dt(
                        min(
                            it.get("entryTime")
                            for it in history_items
                            if isinstance(it, dict) and it.get("entryTime") is not None
                        )
                    )
                except Exception:
                    first_trade = ""

                try:
                    last_trade_close = _ms_to_dt(
                        max(
                            (it.get("closeTime") if it.get("closeTime") is not None else it.get("entryTime"))
                            for it in history_items
                            if isinstance(it, dict) and it.get("entryTime") is not None
                        )
                    )
                except Exception:
                    last_trade_close = ""

                for it in history_items:
                    if not isinstance(it, dict):
                        continue
                    ts = it.get("closeTime") if it.get("closeTime") is not None else it.get("entryTime")
                    by_month[_ms_to_ym(ts)].append(_as_float(it.get("net")))

            out = {
                "file": str(path),
                "symbol": main_node.get("symbol"),
                "period": main_node.get("period"),
                "testingPeriod": (main_node.get("testingPeriod") or {}).get("formatted")
                if isinstance(main_node.get("testingPeriod"), dict)
                else None,
                "dataType": (main_node.get("data") or {}).get("type") if isinstance(main_node.get("data"), dict) else None,
                "startingCapital": main_node.get("startingCapital"),
                "roi": main_node.get("roi"),
                "netProfit": (stats.get("netProfit") or {}).get("all") if isinstance(stats, dict) else None,
                "profitFactor": (stats.get("profitFactor") or {}).get("all") if isinstance(stats, dict) else None,
                "totalTrades": (stats.get("totalTrades") or {}).get("all") if isinstance(stats, dict) else None,
                "profitFactor_long": (stats.get("profitFactor") or {}).get("long") if isinstance(stats, dict) else None,
                "profitFactor_short": (stats.get("profitFactor") or {}).get("short") if isinstance(stats, dict) else None,
                "netProfit_long": (stats.get("netProfit") or {}).get("long") if isinstance(stats, dict) else None,
                "netProfit_short": (stats.get("netProfit") or {}).get("short") if isinstance(stats, dict) else None,
                "maxConsecutiveLosingTrades": (stats.get("maxConsecutiveLosingTrades") or {}).get("all") if isinstance(stats, dict) else None,
                "maxBalanceDrawdownPercent": (
                    (data.get("equity") or {}).get("maxBalanceDrawdownPercent")
                    if isinstance(data.get("equity"), dict)
                    else None
                ),
                "maxEquityDrawdownPercent": (
                    (data.get("equity") or {}).get("maxEquityDrawdownPercent")
                    if isinstance(data.get("equity"), dict)
                    else None
                ),
                "firstTradeUtc": first_trade or None,
                "lastTradeCloseUtc": last_trade_close or None,
                "monthlyNetUtc": {k: float(sum(v)) for k, v in sorted(by_month.items())},
            }
            print(json.dumps(out, indent=2 if pretty else None, sort_keys=False))
            continue

        headline_keys = [
            "symbol",
            "period",
            "roi",
            "netProfit",
            "startingCapital",
            "endingEquity",
            "endingBalance",
        ]
        _print_kv("HEADLINE", main_node, headline_keys)

        testing_period = main_node.get("testingPeriod") if isinstance(main_node, dict) else None
        if isinstance(testing_period, dict) and "formatted" in testing_period:
            print(f"  testingPeriod: {testing_period['formatted']}")

        spread = main_node.get("spread") if isinstance(main_node, dict) else None
        if isinstance(spread, dict):
            print(f"  spread: {spread.get('type')} value={spread.get('value')}")

        comm = main_node.get("commissions") if isinstance(main_node, dict) else None
        if isinstance(comm, dict):
            print(f"  commissions: {comm.get('type')} value={comm.get('value')}")

        # Drawdown is exported under the 'equity' node.
        equity = data.get("equity", {}) if isinstance(data, dict) else {}
        if isinstance(equity, dict):
            _print_kv(
                "\nDRAWDOWN",
                equity,
                [
                    "maxBalanceDrawdownPercent",
                    "maxEquityDrawdownPercent",
                    "maxBalanceDrawdownAbsolute",
                    "maxEquityDrawdownAbsolute",
                ],
            )

        # History (per-trade list)
        history = data.get("history", {}) if isinstance(data, dict) else {}
        history_items = history.get("items") if isinstance(history, dict) else None
        if isinstance(history_items, list) and history_items:
            nets = [_as_float(it.get("net")) for it in history_items if isinstance(it, dict)]
            wins = [x for x in nets if x > 0]
            losses = [-x for x in nets if x < 0]
            gross_profit = sum(wins)
            gross_loss = sum(losses)
            pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

            print("\nHISTORY (DERIVED)")
            print(f"  trades: {len(nets)}")
            print(f"  net: {sum(nets):.2f}")
            print(f"  winRate: {100.0 * (len(wins) / len(nets)):.2f}%")
            print(f"  profitFactor(net): {pf:.2f}")

            first_n = 10
            if len(nets) > 1:
                first_slice = nets[: min(first_n, len(nets))]
                rest_slice = nets[min(first_n, len(nets)) :]
                print(f"  first{len(first_slice)}_net: {sum(first_slice):.2f}")
                if rest_slice:
                    print(f"  last{len(rest_slice)}_net: {sum(rest_slice):.2f}")

            by_month: Dict[str, List[float]] = {}
            for it in history_items:
                if not isinstance(it, dict):
                    continue
                ts = it.get("closeTime") if it.get("closeTime") is not None else it.get("entryTime")
                by_month.setdefault(_ms_to_ym(ts), []).append(_as_float(it.get("net")))

            if by_month:
                print("\nMONTHLY NET (UTC)")
                for ym in sorted(by_month.keys()):
                    vals = by_month[ym]
                    if not vals:
                        continue
                    w = sum(1 for v in vals if v > 0)
                    l = sum(1 for v in vals if v < 0)
                    print(f"  {ym}: net={sum(vals):.2f} trades={len(vals)} (W{w}/L{l})")

        # Try direct known nodes first
        trade_stats = None
        found_key = None
        for k in ["tradeStatistics", "tradingStatistics", "tradesStatistics"]:
            node = data.get(k)
            if isinstance(node, dict):
                trade_stats = node
                found_key = k
                break

        print("\nTRADE STATISTICS")
        if isinstance(trade_stats, dict):
            print(f"  node: <root>.{found_key}")
            _print_kv("  common:", trade_stats, ["netProfit", "profitFactor", "commissions", "swaps"])
            _print_kv(
                "  counts:",
                trade_stats,
                ["totalTrades", "winningTrades", "losingTrades", "maxConsecutiveLosingTrades"],
            )
            _print_kv("  extremes:", trade_stats, ["largestWinningTrade", "largestLosingTrade"])

            # Derived win rate (safer than relying on differing field names)
            try:
                total = trade_stats.get("totalTrades", {}).get("all")
                wins = trade_stats.get("winningTrades", {}).get("all")
                if isinstance(total, (int, float)) and isinstance(wins, (int, float)) and total:
                    print(f"  derivedWinRate(all): {100.0 * wins / total:.2f}%")
            except Exception:
                pass
        else:
            # Heuristic search
            candidates = _walk_candidates(data)
            print(f"  direct node not found; candidates={len(candidates)}")
            # Print up to 8 candidate summaries
            shown = 0
            for path_str, node in candidates:
                show = {
                    kk: node.get(kk)
                    for kk in ["totalTrades", "trades", "winRate", "profitFactor", "grossProfit", "grossLoss", "netProfit"]
                    if kk in node
                }
                if not show:
                    continue
                print(f"  - {path_str}: {show}")
                shown += 1
                if shown >= 8:
                    break


if __name__ == "__main__":
    main()
