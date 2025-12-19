# Performance Improvement Roadmap

> Analysis Date: 2025-12-13 | Bot Version: 1.25.4
> Last Updated: 2025-12-14 (v1.28.0)

This roadmap documents gaps identified compared to professional trading systems, prioritized by **P&L impact** rather than institutional features.

---

## Completed Items

### 4.2 Minimum Profit Threshold (v1.25.0)

**Status:** DONE

Rejects trades where stop distance < 2× round-trip fees (1.2%) to ensure positive expected value.

- Added `estimated_fee_percent` to `ValidatorConfig`
- Added `_check_profit_margin()` validation in `validator.py`
- Warns when margin is tight (between 1.2% and 1.8%)

### 4.1 Maker Fee Optimization (v1.25.1 → v1.25.4)

**Status:** REVERTED

Added post-only (maker) orders in v1.25.1 to save ~0.2% per trade. **Reverted in v1.25.4**
due to fundamental race condition issues with GTC orders - risk of double-fills outweighs
fee savings.

Execution now uses simple IOC → market fallback which is reliable and predictable.

### 4.3 AI as Regime Setter

**Status:** DEFERRED

Deferred pending data on AI review impact. Need to measure:
- How often AI vetoes trades
- How often AI changes position size
- Whether per-trade review adds value vs hourly regime setting

Consider adding metrics tracking before implementing this change.

### 1.2 Regime-Adaptive Indicator Weights (v1.27.x)

**Status:** DONE

AI-driven weight profile selection that adapts indicator weights to market conditions.

- Weight profiles: TREND, RANGE, VOLATILE, DEFAULT in `weight_profile_selector.py`
- AI selects optimal profile via OpenRouter with 15-minute caching
- Circuit breaker fallback to rule-based selection after 3 failures
- Database persistence of profile changes
- Full test coverage in `tests/test_weight_profile_selector.py`

**Implementation exceeds roadmap scope** - uses AI selection instead of simple regime mapping.

---

## Partial Implementations

### Break-Even Stop Logic (part of 3.1)
- `runner.py:1444-1467` moves stop to break-even when profit threshold reached
- Missing: Multi-level scale-out and partial sells

### Volume Analysis with Whale Detection (2.3) - DONE (v1.27.40)
- `signal_scorer.py:481-544` checks volume_ratio thresholds
- 1.5-3x: High volume (20% signal boost, configurable)
- 3x+: Whale activity (30% signal boost, `_whale_activity` flag, configurable)
- Whale direction detection (bullish/bearish/neutral based on price movement)
- Whale alerts integrated into AI reviewer/judge prompts
- Database persistence in `whale_events` table with paper/live separation
- Configurable thresholds: `WHALE_VOLUME_THRESHOLD`, `WHALE_DIRECTION_THRESHOLD`, `WHALE_BOOST_PERCENT`, `HIGH_VOLUME_BOOST_PERCENT`

### Fear & Greed Sentiment (part of 2.2)
- `src/ai/sentiment.py` fetches Fear & Greed Index
- `regime.py` applies sentiment adjustments
- Missing: Funding rates, open interest, long/short ratios

---

## Current Strengths

The bot already has excellent defensive architecture:

- Multi-indicator confluence scoring (-100 to +100)
- Multi-agent AI trade review (3 reviewers + judge)
- Market regime adaptation (sentiment, volatility, trend)
- Comprehensive safety systems (kill switch, circuit breaker, loss limiter)
- ATR-based dynamic position sizing
- Trailing stops with break-even protection
- Paper trading mode
- Multi-exchange support (Coinbase, Kraken)
- Hot-reload configuration
- Telegram notifications + dashboard

## Priority 1: Critical P&L Improvements

### 1.1 Multi-Timeframe Confirmation (v1.28.0)

**Status:** DONE

Multi-timeframe confirmation using Daily + 4-Hour trends to reduce false signals.

- Fetches ONE_DAY and FOUR_HOUR candles with caching (60min and 30min respectively)
- Both timeframes must agree for strong bias (bullish/bearish), otherwise neutral
- Score modifiers: +20 for aligned trades, -20 for counter-trend trades
- HTF context shown to AI reviewers for better decision making
- Signal history table (`signal_history`) stores every signal for post-mortem analysis

**Why 4H instead of 6H?**
- FOUR_HOUR chosen for broader exchange compatibility (Kraken doesn't support 6-hour candles)
- 4H divides evenly into 24 hours (6 candles/day) for consistent daily alignment
- Provides good intermediate timeframe between daily and hourly trading

**Expected Impact:** 30-50% reduction in false signals

**Original Problem:** Trading signals on a single timeframe (e.g., 15-min) without checking higher timeframe trend. Trading against the Daily/4H trend is the #1 cause of stop-outs.

**Current Behavior:**
```
15m RSI oversold → BUY
But Daily trend is bearish → price keeps falling → stopped out
```

**Solution:**
```python
# In runner.py, fetch multiple timeframes
daily_candles = self.client.get_candles(pair, "ONE_DAY", limit=50)
four_hour_candles = self.client.get_candles(pair, "FOUR_HOUR", limit=50)

# Determine macro bias
daily_trend = self._get_trend(daily_candles)  # bullish/bearish/neutral

# In SignalScorer, apply bias modifier
if daily_trend == "bullish" and signal_action == "buy":
    score += 20  # Boost aligned trades
elif daily_trend == "bearish" and signal_action == "buy":
    score -= 20  # Penalize counter-trend trades
```

**Files to Modify:**
- `src/daemon/runner.py` - Fetch additional timeframes
- `src/strategy/signal_scorer.py` - Accept and apply bias modifier

**Expected Impact:** 30-50% reduction in false signals

**Effort:** Medium

---

### 1.2 Regime-Adaptive Indicator Weights

**Status:** DONE - See Completed Items section above.

---

### 1.3 Support/Resistance Awareness

**Problem:** Bot trades purely on indicator values, ignoring obvious price levels where reversals typically occur.

**Current Behavior:**
```
RSI oversold + MACD cross → BUY at $99,800
But $100,000 is massive resistance → immediate rejection
```

**Solution:**
```python
# New file: src/strategy/price_structure.py
class PriceStructure:
    def find_swing_levels(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        """Find recent swing highs and lows."""
        highs = df['high'].rolling(window=5, center=True).max()
        lows = df['low'].rolling(window=5, center=True).min()

        resistance_levels = []  # Local maxima
        support_levels = []     # Local minima

        # Also add round numbers
        current_price = df['close'].iloc[-1]
        round_levels = self._get_round_numbers(current_price)

        return {
            'resistance': resistance_levels + round_levels['above'],
            'support': support_levels + round_levels['below']
        }

    def is_near_resistance(self, price: float, levels: list, threshold: float = 0.005) -> bool:
        """Check if price is within 0.5% of any resistance level."""
        for level in levels:
            if abs(price - level) / level < threshold:
                return True
        return False
```

**Integration in SignalScorer:**
```python
# Veto buys near resistance
if action == "buy" and price_structure.is_near_resistance(price, levels):
    score -= 30  # Heavy penalty

# Use S/R for take-profit targets instead of arbitrary ATR multiples
```

**Files to Create/Modify:**
- `src/strategy/price_structure.py` - New file
- `src/strategy/signal_scorer.py` - Integrate S/R filter
- `src/strategy/position_sizer.py` - Use S/R for targets

**Expected Impact:** Avoid obvious "wall" trades, better profit targets

**Effort:** Medium

---

## Priority 2: High-Value Signal Enhancements

### 2.1 RSI/MACD Divergence Detection

**Problem:** Divergences are powerful reversal signals not currently detected.

**Example:**
```
Price: New high at $102,000
RSI: Lower high than previous peak
= Bearish divergence → reversal likely
```

**Solution:**
```python
# In src/indicators/divergence.py
def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 14) -> str:
    """Detect bullish or bearish divergence."""
    # Find price peaks/troughs
    price_highs = find_peaks(price, lookback)
    indicator_highs = find_peaks(indicator, lookback)

    # Bearish: Price higher high, indicator lower high
    if price_highs[-1] > price_highs[-2] and indicator_highs[-1] < indicator_highs[-2]:
        return "bearish_divergence"

    # Bullish: Price lower low, indicator higher low
    # ... similar logic

    return "none"
```

**Files to Create:**
- `src/indicators/divergence.py`

**Expected Impact:** Catch reversals before lagging indicators confirm

**Effort:** Medium

---

### 2.2 Crypto-Native Sentiment Data

**Problem:** Fear & Greed Index updates once daily. Crypto-specific data is more actionable.

**Data Sources:**

| Signal | API Source | Interpretation |
|--------|------------|----------------|
| Funding Rate | Exchange API | >0.1% = overleveraged longs, correction likely |
| Open Interest | Exchange API | Rising OI + rising price = strong trend |
| Long/Short Ratio | Exchange API | >70% long = contrarian sell signal |
| Liquidations | Coinglass | Large cascade = potential reversal point |

**Solution:**
```python
# In src/ai/sentiment.py or new src/data/derivatives.py
async def fetch_funding_rate(exchange: str, symbol: str) -> float:
    """Fetch current funding rate from exchange."""
    # Coinbase doesn't have perps, would need Binance/Bybit for this
    pass

async def fetch_open_interest(symbol: str) -> dict:
    """Fetch open interest data."""
    # Use Coinglass API or exchange API
    pass

# Integration in regime.py
if funding_rate > 0.001:  # 0.1%
    threshold_adj += 10  # Harder to buy (correction likely)
```

**Files to Create/Modify:**
- `src/data/derivatives.py` - New file for derivatives data
- `src/strategy/regime.py` - Integrate new signals

**Expected Impact:** Earlier warning of overleveraged conditions

**Effort:** Medium (requires additional API integration)

---

### 2.3 Volume Spike Detection

**Status:** DONE (v1.27.40) - See Partial Implementations section above for details.

---

## Priority 3: Exit Strategy Improvements

### 3.1 Partial Profit Taking (Scale Out)

**Problem:** Current exit is trailing stop only. Winners can turn to losers when trailing gives back gains.

**Solution:**
```python
# In position_sizer.py or new exit_manager.py
class ExitStrategy:
    def calculate_scale_out_levels(self, entry_price: Decimal, atr: Decimal) -> list:
        return [
            {'level': entry_price + (atr * 1.5), 'sell_percent': 50, 'action': 'move_stop_to_breakeven'},
            {'level': entry_price + (atr * 3.0), 'sell_percent': 25, 'action': 'trail_remainder'},
        ]
```

**Integration:**
```python
# In runner.py, check scale-out levels before trailing stop logic
for level in exit_strategy.levels:
    if current_price >= level['level'] and not level['triggered']:
        self._execute_partial_sell(level['sell_percent'])
        level['triggered'] = True
```

**Files to Create/Modify:**
- `src/strategy/exit_manager.py` - New file
- `src/daemon/runner.py` - Integrate scale-out checks

**Expected Impact:** Lock in profits, let winners run

**Effort:** Medium

---

### 3.2 Time-Based Exit

**Problem:** Positions can be stuck in sideways markets for days, tying up capital.

**Solution:**
```python
# Track position age
position_age_hours = (now - position.entry_time).total_seconds() / 3600

# If position is flat (< 1% move) after 48 hours, consider closing
if position_age_hours > 48 and abs(unrealized_pnl_percent) < 1.0:
    logger.info("time_based_exit_triggered", age_hours=position_age_hours)
    # Close position or reduce size
```

**Expected Impact:** Free up capital from stale positions

**Effort:** Low

---

## Priority 4: Quick Wins

### 4.1 Maker Fee Optimization

**Problem:** IOC orders act as taker (pay ~0.05% fee). Maker orders earn rebates (+0.02%).

**Current Code (`runner.py:1141`):**
```python
result = self.client.limit_buy_ioc(...)  # Always IOC = taker
```

**Solution:**
```python
# Try Post-Only first, fall back to IOC after timeout
result = self.client.limit_buy_post_only(pair, size, limit_price)
await asyncio.sleep(3)  # Wait for fill

if not result.filled:
    result = self.client.limit_buy_ioc(pair, size, limit_price)  # Fall back
```

**Expected Impact:** 0.07% savings per trade (0.7% over 1000 trades)

**Effort:** Low

---

### 4.2 Minimum Profit Threshold

**Problem:** If stop distance < 2× round-trip fees, trade has negative expected value.

**Solution:**
```python
# In validator.py
def _check_expected_value(self, order: OrderRequest) -> ValidationResult:
    round_trip_fees = 0.006  # 0.6% for Coinbase (0.3% each way)
    stop_distance_percent = self.config.stop_loss_atr_multiplier * atr / price

    if stop_distance_percent < round_trip_fees * 2:
        return ValidationResult(
            valid=False,
            reason=f"Stop too tight for fees. Stop: {stop_distance_percent:.2%}, Min: {round_trip_fees * 2:.2%}"
        )
```

**Expected Impact:** Avoid negative EV trades

**Effort:** Low

---

### 4.3 AI as Regime Setter (Not Trade Gatekeeper)

**Problem:** AI review blocks each trade for 5-10 seconds. In fast markets, this causes slippage.

**Current Flow:**
```
Signal → Wait for AI (5-10s) → Execute (on stale price)
```

**Proposed Flow:**
```
AI runs every 15-60 mins → Sets strategy mode ("aggressive", "conservative", "long-only")
Signal → Execute immediately based on mode
```

**Implementation:**
```python
# Run AI asynchronously to set regime
async def _hourly_ai_regime_update(self):
    result = await self.trade_reviewer.analyze_market(...)
    self._ai_strategy_mode = result.judge_recommendation  # "accumulate", "reduce", "wait"

# In trading iteration, use mode but don't wait for AI
if self._ai_strategy_mode == "reduce":
    safety_multiplier *= 0.5
elif self._ai_strategy_mode == "wait":
    return  # Skip trading entirely
# Otherwise execute immediately
```

**Expected Impact:** Faster execution, less slippage

**Effort:** Low-Medium

---

## Implementation Order

| Phase | Items | Status |
|-------|-------|--------|
| **Phase 1** | 4.2 Min profit check | DONE (v1.25.0) |
| **Phase 1a** | 4.1 Maker fees | REVERTED (v1.25.4) - race conditions |
| **Phase 1b** | 4.3 AI regime setter | DEFERRED (needs metrics) |
| **Phase 1c** | 1.2 Adaptive weights | DONE (v1.27.x) |
| **Phase 1d** | 2.3 Volume spikes / Whale detection | DONE (v1.27.40) |
| **Phase 2** | 1.1 Multi-timeframe | DONE (v1.28.0) |
| **Phase 3** | 1.3 S/R awareness | Pending |
| **Phase 4** | 3.1 Scale-out exits, 2.1 Divergence | Pending |
| **Phase 5** | 2.2 Crypto sentiment APIs, 3.2 Time exits | Pending |

---

## Backtesting Requirement

Before implementing Phase 2+, consider adding a backtesting harness. Without it, you're live-testing strategy changes with real capital.

**Minimal Backtesting Approach:**
```python
# Create BacktestRunner that:
# 1. Loads historical candles from database (rate_history table)
# 2. Mocks ExchangeClient to return historical data
# 3. Runs _trading_iteration() against past data
# 4. Tracks simulated P&L and metrics
```

This allows validating weight changes, new indicators, etc. before deployment.

---

## Metrics to Track

After implementing changes, track:

| Metric | Current | Target |
|--------|---------|--------|
| Win Rate | ? | +10% improvement |
| Average Win / Average Loss | ? | > 1.5 |
| Max Drawdown | ? | < 15% |
| Sharpe Ratio | ? | > 1.0 |
| Trades per Week | ? | Quality over quantity |

---

## Notes

- All changes should be tested in paper trading first
- Bump version for each significant change
- Consider A/B testing by running paper vs live with different parameters
