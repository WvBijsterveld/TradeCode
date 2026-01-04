# Strategy Notes (Living Document)

Purpose: capture what we learn (good vs bad trades), why changes were made, and what to try next.
This prevents us from repeating the same mistakes and makes iteration faster.

## Current Situation (as of 2025-12-27)

- We finally have a true 1-year backtest with tick data (Report21).
- Report21 summary:
  - Net: -407 on 50k, PF 0.92, trades 70.
  - Longs slightly positive (PF ~1.06), shorts losing (PF ~0.79).
  - High win rate (~60%) but PF<1 ⇒ average loser > average winner (plus commissions/swaps).
- Recent over-filtering risk: disabling one side + high RR + high prob-gap can starve trades ("1 trade in 2 months").
- History reminder: in shorter windows (e.g. 4 months) shorts sometimes outperform longs ⇒ regime-dependence is real.

## What Makes Trades Good (Hypotheses)

These are conditions that tend to produce profitable + consistent outcomes.

### 1) Clear directional edge (not a coin-flip)
- In supervised mode, when one side probability clearly dominates the other (large `prob_gap`), we avoid random/indecisive entries.
- Practical implication: keep a minimum `prob_gap`, but tune it so we don’t starve trades.

### 2) Trend alignment and structure context
- Trend slope normalized by ATR (`trend_slope_atr`) helps avoid shorting strong uptrends or buying strong downtrends.
- Structure/SR context can reduce entries into “no-man’s land” and favor retests.
- Practical implication: trend gate should block obvious counter-trend trades, but not be so strict it removes most opportunities.

### 3) Reasonable reward:risk (R:R)
- PF<1 with high win rate suggests losers outweigh winners.
- Enforcing a minimum R:R helps push the distribution toward positive expectancy.
- Practical implication: raise `min_rr` gradually; consider asymmetric `min_rr_long` vs `min_rr_short`.

### 4) Avoid bad liquidity times / chop
- Limiting to a consistent trading window (e.g. 8–16 local) helps skip dead/erratic periods.
- Practical implication: keep hours consistent when comparing runs.

### 5) Avoid “revenge trading” / loss streak spirals
- Adaptive threshold and cooldown reduce churn during bad regimes.
- Practical implication: keep adaptive logic, but ensure reset works (global bar index + inactivity reset).

## What Makes Trades Bad (Hypotheses)

### 1) Low-contrast signals
- p_long ≈ p_short means the model is unsure; these behave like coin flips.
- Symptom: many entries with small `prob_gap`.

### 2) Counter-trend entries
- Shorts taken while slope is positive, or longs taken while slope is negative.
- Symptom: poor direction PF (often persistent across long backtests).

### 3) Poor R:R or tiny TP / oversized SL
- R:R < 1.0 makes profitability difficult even at high win rate.
- Symptom: PF<1 despite win rate > 50%.

### 4) Over-filtering (starvation)
- Too many gates at once can reduce trades to near-zero.
- Symptom: “1 trade in 2 months”, inconsistent exposure, results dominated by randomness.

### 5) Regime mismatch
- A side that works in a 4-month window can lose in the full year.
- Symptom: direction PF flips when expanding test horizon.

## How the Code Behaves (Bridge Mental Model)

In `v31_live_bridge.py` (supervised mode):
- Builds features (windowed), predicts p_long and p_short.
- Applies time/session mask.
- Applies thresholds (per-side + optional adaptive bump).
- Optionally applies `prob_gap` filter.
- Computes trade levels (SL/TP) and enforces RR.
- Optionally limits trades/day.
- Writes a bar-level log row with:
  - probabilities, thresholds, rr, prob_gap
  - `reject_reasons` explaining why an entry was suppressed.

This is critical: we can debug “why didn’t we trade?” objectively, not by guessing.

## What Worked / Didn’t (Memory)

Worked:
- Robust logging / buffered CSV (prevents stalls).
- Trend alignment gate and RR gating reduced churn.
- Inactivity reset for adaptive threshold prevented “stuck high threshold”.

Didn’t / risky:
- Silent input-dimension mismatch (fixed with fail-fast check).
- Over-tightening multiple gates at once (starves trades).
- Trusting short-horizon backtests as globally true (regime mismatch).

## Improvements to Try (Order Matters)

### Phase A: Find a profitable core with enough trades
1) Keep both sides enabled.
2) Use moderate RR + moderate prob-gap.
3) Use trend gate as a soft blocker (0.0 boundary), not extreme.
4) Confirm trades/year is reasonable (not < ~20 unless PF is excellent).

### Phase B: Fix the weak side without killing the system
- Use asymmetric controls:
  - `min_rr_short` > `min_rr_long` if shorts have worse expectancy.
  - `sup_min_prob_gap_short` > `sup_min_prob_gap_long` if shorts are noisier.
  - `trend_slope_short_max` tighter if shorting uptrends is the failure mode.

### Phase C: Instrumentation and decision memory
- Always keep `reject_reasons` and `prob_gap` in logs.
- Add a small tool to summarize:
  - which reject reasons dominate
  - distribution of prob_gap and rr at executed entries
  - rough association between prob-gap buckets and trade outcomes

## Run Checklist (Consistency)

When comparing runs, keep constant unless intentionally changing:
- Data type (tick vs bars)
- Trading hours
- Thresholds
- RR
- Trend gates
- TP margins

Record each run:
- Command line / params
- Report net/PF/trades and long/short split
- Any unusual behavior (starvation, stuck adaptive, missing data)
