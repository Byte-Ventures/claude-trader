# Future Test Improvements

Recommendations from PR #6 bot review (2025-12-01)

## High Priority

### 1. Database Isolation Testing
**Priority**: Medium
**Requirement**: Per CLAUDE.md - all database operations MUST support PAPER and ACTUAL modes

```python
def test_paper_live_data_isolation():
    """Verify paper trades don't leak into live queries and vice versa."""
    # Add paper trade
    db.add_trade(symbol="BTC-USD", is_paper=True, ...)

    # Add live trade
    db.add_trade(symbol="BTC-USD", is_paper=False, ...)

    # Verify isolation
    paper_trades = db.get_trades(is_paper=True)
    live_trades = db.get_trades(is_paper=False)

    assert len(paper_trades) == 1
    assert len(live_trades) == 1
    assert paper_trades[0].id != live_trades[0].id
```

**Current Status**: Database module at 0% test coverage

### 2. Regression Test for Crash Protection Bug
**Priority**: High
**Purpose**: Document the v1.14.17 bug fix and prevent future regressions

```python
def test_regression_crash_protection_equals_minimum():
    """Regression test for bug where current == min_in_window returned True."""
    # Scenario: Price making new lows (current equals minimum)
    prices = pd.Series([100, 95, 90, 90, 90])
    scorer = SignalScorer()

    # Old behavior (v1.14.16): would return True (incorrect)
    # New behavior (v1.14.17+): returns False (correct)
    assert scorer.is_price_stabilized(prices, window_candles=5) is False
```

## Medium Priority

### 3. Exchange Client Integration Tests
**Priority**: Medium
**Current Status**: Exchange clients at 0% test coverage

Recommendations:
- Add tests with mocked API responses
- Test against exchange sandbox APIs
- Verify order placement, cancellation, position tracking
- Test API error handling and retries

### 4. Property-Based Testing for Indicators
**Priority**: Low
**Tool**: Use `hypothesis` for mathematical invariants

```python
from hypothesis import given, strategies as st

@given(prices=st.lists(st.floats(min_value=1, max_value=1000), min_size=20))
def test_rsi_always_between_0_and_100(prices):
    """RSI should always be in valid range regardless of input."""
    rsi = calculate_rsi(pd.Series(prices), period=14)
    assert 0 <= rsi <= 100
```

## Low Priority

### 5. Performance Benchmarks
**Priority**: Low
**Purpose**: Catch performance regressions

```python
def test_signal_scorer_performance_benchmark(benchmark, sample_ohlcv_data):
    """Ensure signal scoring completes in <100ms for 1000 candles."""
    scorer = SignalScorer()
    df = sample_ohlcv_data(length=1000)

    result = benchmark(scorer.calculate_score, df)

    assert benchmark.stats.mean < 0.1  # <100ms average
```

### 6. Mutation Testing
**Tool**: `mutmut`
**Purpose**: Verify test quality by introducing mutations

## Current Test Coverage (v1.14.18)

| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| circuit_breaker.py | 40 | 100% | ✅ |
| loss_limiter.py | 52 | 99% | ✅ |
| kill_switch.py | 35 | 100% | ✅ |
| validator.py | 47 | 99% | ✅ |
| signal_scorer.py | 60 | 87% | ✅ |
| position_sizer.py | 40 | 99% | ✅ |
| regime.py | 35 | 99% | ✅ |
| **Safety & Strategy** | **309** | **~96%** | ✅ |
| database.py | 0 | 0% | ⚠️ |
| Exchange clients | 0 | 0% | ⚠️ |
| **Overall** | **309** | **30%** | - |

## Post-Merge Monitoring

1. Monitor production for 24-48 hours after v1.14.18 deployment
2. Watch for:
   - Crash protection behavior during market volatility
   - Loss limiter callback errors in logs
   - Validator boundary conditions (90% position, 5% price deviation)
3. Tag release with v1.14.18 after successful deployment
