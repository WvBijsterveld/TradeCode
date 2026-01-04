using System;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Internals;
using NetMQ;
using NetMQ.Sockets;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class AI_Bridge_V10 : Robot
    {
        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0)]
        public double RiskPercent { get; set; }

        [Parameter("BE Trigger (pips)", DefaultValue = 4.0)]
        public double BreakEvenTriggerPips { get; set; }

        [Parameter("BE Plus (pips)", DefaultValue = 0.3)]
        public double BreakEvenPlusPips { get; set; }

        [Parameter("Lock Trigger (pips)", DefaultValue = 9.0)]
        public double ProfitLockTriggerPips { get; set; }

        [Parameter("Lock Plus (pips)", DefaultValue = 1.5)]
        public double ProfitLockPlusPips { get; set; }

        [Parameter("Max DD From Daily Peak (%)", DefaultValue = 3.0)]
        public double MaxDrawdownFromDailyPeakPct { get; set; }

        [Parameter("Max DD From All-Time Peak (%)", DefaultValue = 4.5)]
        public double MaxDrawdownFromAllTimePeakPct { get; set; }

        [Parameter("All-Time DD Mode (off|cooldown|halt|riskoff)", DefaultValue = "cooldown")]
        public string AllTimeDdMode { get; set; }

        [Parameter("All-Time DD Cooldown (hours)", DefaultValue = 24)]
        public int AllTimeDdCooldownHours { get; set; }

        [Parameter("Cooldown Volume Mult", DefaultValue = 0.5)]
        public double CooldownVolumeMultiplier { get; set; }

        [Parameter("Max Consecutive Losses", DefaultValue = 3)]
        public int MaxConsecutiveLosses { get; set; }

        [Parameter("Max Trades Per Day", DefaultValue = 0)]
        public int MaxTradesPerDay { get; set; }

        [Parameter("Resume Next Day", DefaultValue = true)]
        public bool ResumeNextDay { get; set; }

        private const string Label = "AI_V10";

        private RequestSocket _socket;
        private bool _isConnected = false;
        private double _lastClosedPnl = 0.0;

        private DateTime _currentDay;
        private double _dailyEquityStart;
        private double _dailyEquityPeak;
        private double _allTimeEquityPeak;
        private int _consecutiveLosses;
        private int _tradesTakenToday;
        private bool _tradingHalted;
        private bool _permanentHalt;
        private string _haltReason;
        private DateTime? _riskOffUntil;

        private void SetHalt(string reason, bool permanent)
        {
            _tradingHalted = true;
            if (permanent)
                _permanentHalt = true;

            if (_haltReason != reason)
            {
                _haltReason = reason;
                Print(reason);
            }
        }

        protected override void OnStart()
        {
            try
            {
                _socket = new RequestSocket();
                _socket.Connect("tcp://localhost:5555");
                _isConnected = true;
                Print("CONNECTED to Python AI Brain.");

                // Some cTrader/cAlgo versions don't support overriding OnPositionClosed.
                // Use the Positions.Closed event instead.
                Positions.Closed += OnPositionsClosed;

                _currentDay = Server.Time.Date;
                _dailyEquityStart = Account.Equity;
                _dailyEquityPeak = Account.Equity;
                _tradesTakenToday = 0;
                _allTimeEquityPeak = Account.Equity;
                _consecutiveLosses = 0;
                _tradingHalted = false;
                _permanentHalt = false;
                _haltReason = null;
                _riskOffUntil = null;
            }
            catch (Exception ex)
            {
                Print("Connection Failed: " + ex.Message);
                Stop();
            }
        }

        private void OnPositionsClosed(PositionClosedEventArgs args)
        {
            if (args.Position == null)
                return;

            if (args.Position.SymbolName != SymbolName)
                return;

            if (args.Position.Label != Label)
                return;

            // Realized PnL of the trade that just closed (reward signal)
            _lastClosedPnl = args.Position.NetProfit;

            if (_lastClosedPnl < 0)
                _consecutiveLosses += 1;
            else
                _consecutiveLosses = 0;

            if (MaxConsecutiveLosses > 0 && _consecutiveLosses >= MaxConsecutiveLosses)
            {
                SetHalt($"HALT: consecutive losses={_consecutiveLosses} >= {MaxConsecutiveLosses} (until next day={ResumeNextDay}).", permanent: false);
            }
        }

        protected override void OnBar()
        {
            if (!_isConnected) return;

            // Expire cooldown (if any).
            if (_riskOffUntil.HasValue && Server.Time >= _riskOffUntil.Value && !_permanentHalt)
            {
                _riskOffUntil = null;
                _tradingHalted = false;
            }

            // Daily reset / resume.
            var barDay = Bars.Last(1).OpenTime.Date;
            if (barDay != _currentDay)
            {
                _currentDay = barDay;
                _dailyEquityStart = Account.Equity;
                _dailyEquityPeak = Account.Equity;
                _consecutiveLosses = 0;
                _tradesTakenToday = 0;
                if (ResumeNextDay && !_permanentHalt)
                    _tradingHalted = false;
            }

            // Update daily equity peak and enforce drawdown guard.
            if (Account.Equity > _dailyEquityPeak)
                _dailyEquityPeak = Account.Equity;

            // Track all-time equity peak and enforce a trailing circuit-breaker.
            if (Account.Equity > _allTimeEquityPeak)
                _allTimeEquityPeak = Account.Equity;

            if (!_permanentHalt && MaxDrawdownFromAllTimePeakPct > 0 && _allTimeEquityPeak > 0)
            {
                var ddPctAllTime = 100.0 * (_allTimeEquityPeak - Account.Equity) / _allTimeEquityPeak;
                if (ddPctAllTime >= MaxDrawdownFromAllTimePeakPct)
                {
                    var mode = (AllTimeDdMode ?? "").Trim().ToLowerInvariant();
                    if (mode == "off")
                    {
                        // no-op
                    }
                    else if (mode == "halt")
                    {
                        SetHalt(
                            $"HALT: ALL-TIME drawdown {ddPctAllTime:F2}% >= {MaxDrawdownFromAllTimePeakPct:F2}% (peak={_allTimeEquityPeak:F2} equity={Account.Equity:F2})",
                            permanent: true
                        );
                    }
                    else if (mode == "riskoff")
                    {
                        // Risk-off: reduce size for a while but do not halt entries.
                        // This is useful for backtests where you still want a full-year sample.
                        if (!_riskOffUntil.HasValue || Server.Time >= _riskOffUntil.Value)
                        {
                            var hours = AllTimeDdCooldownHours;
                            if (hours < 1) hours = 1;
                            _riskOffUntil = Server.Time.AddHours(hours);
                            Print(
                                $"RISK-OFF: ALL-TIME drawdown {ddPctAllTime:F2}% >= {MaxDrawdownFromAllTimePeakPct:F2}% " +
                                $"(peak={_allTimeEquityPeak:F2} equity={Account.Equity:F2}) until={_riskOffUntil:yyyy-MM-dd HH:mm} " +
                                $"(volume_mult={CooldownVolumeMultiplier:F2})"
                            );
                        }
                    }
                    else
                    {
                        // Default: cooldown. Pause new entries for a bit, then resume.
                        if (!_riskOffUntil.HasValue || Server.Time >= _riskOffUntil.Value)
                        {
                            var hours = AllTimeDdCooldownHours;
                            if (hours < 1) hours = 1;
                            _riskOffUntil = Server.Time.AddHours(hours);
                            SetHalt(
                                $"COOLDOWN: ALL-TIME drawdown {ddPctAllTime:F2}% >= {MaxDrawdownFromAllTimePeakPct:F2}% (peak={_allTimeEquityPeak:F2} equity={Account.Equity:F2}) until={_riskOffUntil:yyyy-MM-dd HH:mm}",
                                permanent: false
                            );
                        }
                    }
                }
            }

            if (MaxDrawdownFromDailyPeakPct > 0 && _dailyEquityPeak > 0)
            {
                var ddPct = 100.0 * (_dailyEquityPeak - Account.Equity) / _dailyEquityPeak;
                if (ddPct >= MaxDrawdownFromDailyPeakPct)
                {
                    SetHalt($"HALT: daily drawdown {ddPct:F2}% >= {MaxDrawdownFromDailyPeakPct:F2}% (peak={_dailyEquityPeak:F2} equity={Account.Equity:F2})", permanent: false);
                }
            }

            ManageOpenPositions();

            // If halted, keep managing open trades but do not open new ones.
            if (_tradingHalted)
                return;

            var lastBar = Bars.Last(1);
            double open = lastBar.Open;
            double high = lastBar.High;
            double low = lastBar.Low;
            double close = lastBar.Close;

            // Use bar time for consistency in backtests
            int hour = lastBar.OpenTime.Hour;

            // Only consider positions opened by this bot/label on this symbol
            var myPositions = Positions.FindAll(Label, SymbolName);
            int inTrade = (myPositions != null && myPositions.Length > 0) ? 1 : 0;

            // If in-trade, send open PnL; if flat, send last CLOSED trade PnL
            double pnl = (inTrade == 1) ? myPositions.Sum(p => p.NetProfit) : _lastClosedPnl;

            string msg = string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0};{1};{2};{3};{4};{5};{6}",
                open, high, low, close, hour, inTrade, pnl);

            try
            {
                _socket.SendFrame(msg);

                string response = "";
                if (_socket.TryReceiveFrameString(TimeSpan.FromSeconds(5), out response))
                {
                    string[] parts = response.Split(';');
                    if (parts == null || parts.Length < 3)
                    {
                        Print($"AI response invalid (expected 3 fields): '{response}'");
                        return;
                    }

                    int action;
                    if (!int.TryParse(parts[0], out action))
                    {
                        Print($"AI response invalid action: '{response}'");
                        return;
                    }

                    double slPrice;
                    double tpPrice;
                    if (!double.TryParse(parts[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out slPrice) ||
                        !double.TryParse(parts[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out tpPrice))
                    {
                        Print($"AI response invalid SL/TP: '{response}'");
                        return;
                    }

                    // Quiet mode: don't log every candle; only log when taking action.

                    // Hard guard: never open a position without valid protective levels.
                    if (action > 0 && (double.IsNaN(slPrice) || double.IsInfinity(slPrice) || slPrice <= 0 || double.IsNaN(tpPrice) || double.IsInfinity(tpPrice) || tpPrice <= 0))
                    {
                        Print($"Refusing trade: invalid SL/TP from AI. Raw='{response}'");
                        return;
                    }

                    // Prefer checking your label+symbol, not global Positions.Count
                    if (action > 0 && Positions.Find(Label, SymbolName) == null)
                        ExecuteTrade(action, slPrice, tpPrice);
                }
                else
                {
                    Print("Timeout waiting for AI response.");
                    _socket.Dispose();
                    _socket = new RequestSocket();
                    _socket.Connect("tcp://localhost:5555");
                }
            }
            catch (Exception ex)
            {
                Print("Communication Error: " + ex.Message);
            }
        }

        protected override void OnTick()
        {
            ManageOpenPositions();
        }

        private void ManageOpenPositions()
        {
            var myPositions = Positions.FindAll(Label, SymbolName);
            if (myPositions == null || myPositions.Length == 0)
                return;

            // Small epsilon to avoid spammy ModifyPosition calls
            double eps = 0.05 * Symbol.PipSize;

            foreach (var pos in myPositions)
            {
                if (pos == null)
                    continue;

                if (pos.TradeType == TradeType.Buy)
                {
                    double profitPips = (Symbol.Bid - pos.EntryPrice) / Symbol.PipSize;
                    double targetSl = double.NaN;

                    if (profitPips >= ProfitLockTriggerPips)
                        targetSl = pos.EntryPrice + ProfitLockPlusPips * Symbol.PipSize;
                    else if (profitPips >= BreakEvenTriggerPips)
                        targetSl = pos.EntryPrice + BreakEvenPlusPips * Symbol.PipSize;

                    if (!double.IsNaN(targetSl))
                    {
                        targetSl = NormalizeToTick(targetSl);
                        if (pos.StopLoss == null || targetSl > pos.StopLoss.Value + eps)
                        {
                            Print($"Move SL (BUY) -> {targetSl} | entry={pos.EntryPrice} | pnl(pips)={profitPips:F1}");
                            var r = ModifyPosition(pos, targetSl, pos.TakeProfit);
                            if (r == null || !r.IsSuccessful)
                                Print($"ModifyPosition failed (BUY): {(r == null ? "null" : r.Error.ToString())}");
                        }
                    }
                }
                else if (pos.TradeType == TradeType.Sell)
                {
                    double profitPips = (pos.EntryPrice - Symbol.Ask) / Symbol.PipSize;
                    double targetSl = double.NaN;

                    if (profitPips >= ProfitLockTriggerPips)
                        targetSl = pos.EntryPrice - ProfitLockPlusPips * Symbol.PipSize;
                    else if (profitPips >= BreakEvenTriggerPips)
                        targetSl = pos.EntryPrice - BreakEvenPlusPips * Symbol.PipSize;

                    if (!double.IsNaN(targetSl))
                    {
                        targetSl = NormalizeToTick(targetSl);
                        if (pos.StopLoss == null || targetSl < pos.StopLoss.Value - eps)
                        {
                            Print($"Move SL (SELL) -> {targetSl} | entry={pos.EntryPrice} | pnl(pips)={profitPips:F1}");
                            var r = ModifyPosition(pos, targetSl, pos.TakeProfit);
                            if (r == null || !r.IsSuccessful)
                                Print($"ModifyPosition failed (SELL): {(r == null ? "null" : r.Error.ToString())}");
                        }
                    }
                }
            }
        }

        private double NormalizeToTick(double price)
        {
            // Some Automate API builds don't expose Symbol.NormalizePrice.
            // Normalize using TickSize/Digits to ensure prices are valid.
            var ts = Symbol.TickSize;
            if (ts > 0)
            {
                price = Math.Round(price / ts) * ts;
            }
            return Math.Round(price, Symbol.Digits);
        }

        private void ExecuteTrade(int action, double slPrice, double tpPrice)
        {
            var close = Bars.Last(1).Close;

            // Enforce per-day trade cap (cBot-side safeguard).
            if (MaxTradesPerDay > 0 && _tradesTakenToday >= MaxTradesPerDay)
            {
                Print($"Daily trade cap reached ({_tradesTakenToday}/{MaxTradesPerDay}). Skipping entry.");
                return;
            }

            if (double.IsNaN(slPrice) || double.IsInfinity(slPrice) || slPrice <= 0 || double.IsNaN(tpPrice) || double.IsInfinity(tpPrice) || tpPrice <= 0)
                return;

            TradeType dir = action == 2 ? TradeType.Sell : TradeType.Buy;

            double slPips = Math.Abs(close - slPrice) / Symbol.PipSize;
            double tpPips = Math.Abs(close - tpPrice) / Symbol.PipSize;

            if (double.IsNaN(slPips) || double.IsInfinity(slPips) || double.IsNaN(tpPips) || double.IsInfinity(tpPips)) return;
            if (slPips < 2 || tpPips < 2) return;
            if (slPips > 500 || tpPips > 500) return;

            // --- Dynamic risk-based volume calculation with rounding and clamping ---
            double accountEquity = Account.Equity;
            double riskDollars = accountEquity * (RiskPercent / 100.0); // e.g. 1% of equity
            double pipValuePerLot = Symbol.PipValue * 100000; // $ per pip per lot (standard lot)
            double lots = 0.0;
            if (slPips > 0.0 && pipValuePerLot > 0.0)
                lots = riskDollars / (slPips * pipValuePerLot);
            // Debug: print intermediate risk-sizing values for post-mortem
            Print($"RISK DEBUG: AccountEquity={accountEquity:F2} risk=${riskDollars:F2} slPips={slPips:F2} Symbol.PipValue={Symbol.PipValue:F8} pipValuePerLot={pipValuePerLot:F6}");
            Print($"RISK DEBUG: raw_lots={lots:F6}");
            // Round lots to nearest allowed step (e.g., 0.01 lot)
            double lotStep = Symbol.VolumeStep / 100000.0;
            // Use floor rounding to ensure we do not exceed the desired risk due to rounding up.
            if (lotStep > 0.0)
                lots = Math.Floor(lots / lotStep) * lotStep;
            Print($"RISK DEBUG: rounded_lots={lots:F6} lotStep={lotStep:F6}");
            // Clamp to min/max lot size
            double minLots = Symbol.VolumeMin / 100000.0;
            double maxLots = Symbol.VolumeMax / 100000.0;
            if (lots < minLots) lots = minLots;
            if (lots > maxLots) lots = maxLots;
            // Convert to units and round to nearest allowed step
            // Convert to units and floor to nearest allowed step to avoid oversizing.
            double volumeUnits = Math.Floor(lots * 100000.0 / Symbol.VolumeStep) * Symbol.VolumeStep;
            Print($"Placing {dir} | close={close} | SL(pips)={slPips:F1} TP(pips)={tpPips:F1} | SL={slPrice} TP={tpPrice} | lots={lots:F3} | units={volumeUnits} | risk=${riskDollars:F2}");
            var res = ExecuteMarketOrder(dir, SymbolName, volumeUnits, Label, slPips, tpPips);
            if (res == null || !res.IsSuccessful)
            {
                Print($"ExecuteMarketOrder failed: {(res == null ? "null" : res.Error.ToString())}");
            }
            else
            {
                // Count only successful executed opens for the daily cap.
                _tradesTakenToday += 1;
                Print($"Trade opened. Today's trades: {_tradesTakenToday}/{MaxTradesPerDay}");
            }
        }

        protected override void OnStop()
        {
            Positions.Closed -= OnPositionsClosed;
            if (_socket != null) _socket.Dispose();
        }

        // VolumeInUnits property removed: now calculated per-trade in ExecuteTrade
    }
}
