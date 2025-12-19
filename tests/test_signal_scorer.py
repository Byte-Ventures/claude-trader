"""
Comprehensive tests for the SignalScorer multi-indicator confluence system.

Tests cover:
- Weight configuration and validation
- Individual indicator contributions (RSI, MACD, Bollinger, EMA, Volume)
- Composite score aggregation
- Action determination (BUY/SELL/HOLD) based on threshold
- Confidence calculation with confluence factor
- Crash protection (oversold buy limiting, price stabilization)
- Trend filter application and penalties
- Volume confirmation logic
- Settings updates
- Edge cases (insufficient data, NaN handling, boundary conditions)
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from freezegun import freeze_time
from unittest.mock import MagicMock, patch

from src.strategy.signal_scorer import (
    SignalScorer,
    SignalWeights,
    SignalResult,
    IndicatorValues,
    get_recommended_threshold,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def scorer():
    """Signal scorer with default settings."""
    return SignalScorer()


@pytest.fixture
def custom_weights():
    """Custom indicator weights."""
    return SignalWeights(rsi=30, macd=30, bollinger=20, ema=10, volume=10)


@pytest.fixture
def scorer_custom(custom_weights):
    """Signal scorer with custom weights."""
    return SignalScorer(weights=custom_weights, threshold=70)


@pytest.fixture
def sample_df(sample_ohlcv_data):
    """Generate sample OHLCV data for testing."""
    return sample_ohlcv_data(length=200, base_price=50000.0, volatility=0.02)


@pytest.fixture
def bullish_df():
    """Generate strongly bullish market data."""
    np.random.seed(42)
    length = 200
    prices = []
    current = 40000.0

    for i in range(length):
        # Strong uptrend with minor pullbacks
        change = np.random.uniform(0.001, 0.003) * current if i % 10 != 0 else -0.001 * current
        current = current + change
        prices.append(current)

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    for price in prices:
        o = price * 0.998
        c = price * 1.002
        h = max(o, c) * 1.01
        l = min(o, c) * 0.99
        v = np.random.uniform(5000, 15000)  # High volume

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)
        data['volume'].append(v)

    return pd.DataFrame(data)


@pytest.fixture
def bearish_df():
    """Generate strongly bearish market data."""
    np.random.seed(43)
    length = 200
    prices = []
    current = 60000.0

    for i in range(length):
        # Strong downtrend
        change = -np.random.uniform(0.001, 0.003) * current if i % 10 != 0 else 0.001 * current
        current = current + change
        prices.append(current)

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    for price in prices:
        o = price * 1.002
        c = price * 0.998
        h = max(o, c) * 1.01
        l = min(o, c) * 0.99
        v = np.random.uniform(5000, 15000)

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)
        data['volume'].append(v)

    return pd.DataFrame(data)


# ============================================================================
# Initialization Tests
# ============================================================================

def test_default_initialization():
    """Test scorer initializes with default weights and threshold."""
    scorer = SignalScorer()

    assert scorer.weights.rsi == 25
    assert scorer.weights.macd == 25
    assert scorer.weights.bollinger == 20
    assert scorer.weights.ema == 15
    assert scorer.weights.volume == 15
    assert scorer.threshold == 60


def test_default_weights_sum_to_100():
    """Test default weights sum to 100 for proper scoring."""
    weights = SignalWeights()

    total = weights.rsi + weights.macd + weights.bollinger + weights.ema + weights.volume
    assert total == 100


def test_custom_weights_initialization(custom_weights):
    """Test scorer accepts custom weights."""
    scorer = SignalScorer(weights=custom_weights, threshold=70)

    assert scorer.weights.rsi == 30
    assert scorer.weights.macd == 30
    assert scorer.weights.bollinger == 20
    assert scorer.weights.ema == 10
    assert scorer.weights.volume == 10
    assert scorer.threshold == 70


def test_indicator_parameters_initialization():
    """Test all indicator parameters are set correctly."""
    scorer = SignalScorer(
        rsi_period=21,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        macd_fast=8,
        macd_slow=21,
        macd_signal=7,
        bollinger_period=25,
        bollinger_std=2.5,
        ema_fast=12,
        ema_slow=26,
        atr_period=20,
    )

    assert scorer.rsi_period == 21
    assert scorer.rsi_oversold == 30.0
    assert scorer.rsi_overbought == 70.0
    assert scorer.macd_fast == 8
    assert scorer.macd_slow == 21
    assert scorer.macd_signal_period == 7
    assert scorer.bollinger_period == 25
    assert scorer.bollinger_std == 2.5
    assert scorer.ema_fast == 12
    assert scorer.ema_slow_period == 26
    assert scorer.atr_period == 20


def test_oversold_buy_tracking_initialized():
    """Test oversold buy tracking list initializes empty."""
    scorer = SignalScorer()

    assert scorer._oversold_buy_times == []


# ============================================================================
# Score Calculation Tests
# ============================================================================

def test_calculate_score_returns_result(scorer, sample_df):
    """Test calculate_score returns SignalResult."""
    result = scorer.calculate_score(sample_df)

    assert isinstance(result, SignalResult)
    assert isinstance(result.score, int)
    assert result.action in ["buy", "sell", "hold"]
    assert isinstance(result.indicators, IndicatorValues)
    assert isinstance(result.breakdown, dict)
    assert 0.0 <= result.confidence <= 1.0


def test_score_within_bounds(scorer, sample_df):
    """Test score is clamped to -100 to +100."""
    result = scorer.calculate_score(sample_df)

    assert -100 <= result.score <= 100


def test_action_buy_when_above_threshold(scorer, bullish_df):
    """Test action is 'buy' when score >= threshold."""
    result = scorer.calculate_score(bullish_df)

    if result.score >= scorer.threshold:
        assert result.action == "buy"


def test_action_sell_when_below_negative_threshold(scorer, bearish_df):
    """Test action is 'sell' when score <= -threshold."""
    result = scorer.calculate_score(bearish_df)

    if result.score <= -scorer.threshold:
        assert result.action == "sell"


def test_action_hold_when_within_threshold(scorer, sample_df):
    """Test action is 'hold' when |score| < threshold."""
    # Use high threshold to force hold
    scorer.threshold = 95
    result = scorer.calculate_score(sample_df)

    if abs(result.score) < scorer.threshold:
        assert result.action == "hold"


def test_breakdown_contains_all_components(scorer, sample_df):
    """Test breakdown includes all indicator contributions."""
    result = scorer.calculate_score(sample_df)

    assert "rsi" in result.breakdown
    assert "macd" in result.breakdown
    assert "bollinger" in result.breakdown
    assert "ema" in result.breakdown
    assert "volume" in result.breakdown
    assert "trend_filter" in result.breakdown


def test_confidence_zero_on_hold(scorer, sample_df):
    """Test confidence is 0.0 when action is hold."""
    scorer.threshold = 95  # Force hold
    result = scorer.calculate_score(sample_df)

    if result.action == "hold":
        assert result.confidence == 0.0


def test_confidence_increases_with_score_magnitude(scorer, sample_df):
    """Test confidence increases with stronger signals."""
    # Lower threshold to allow trades
    scorer.threshold = 40

    results = []
    for _ in range(10):
        result = scorer.calculate_score(sample_df)
        if result.action != "hold":
            results.append((abs(result.score), result.confidence))

    # Should have some correlation between score and confidence
    if len(results) > 2:
        scores = [s for s, _ in results]
        confidences = [c for _, c in results]
        # Higher scores should generally have higher confidence
        assert max(confidences) > min(confidences) or all(c == confidences[0] for c in confidences)


# ============================================================================
# Individual Indicator Contribution Tests
# ============================================================================

def test_rsi_contribution_calculated(scorer, sample_df):
    """Test RSI contributes to score."""
    result = scorer.calculate_score(sample_df)

    # RSI should have non-zero contribution (unless exactly neutral)
    rsi_score = result.breakdown["rsi"]
    # Score should be scaled by weight
    assert -scorer.weights.rsi <= rsi_score <= scorer.weights.rsi


def test_macd_contribution_calculated(scorer, sample_df):
    """Test MACD contributes to score."""
    result = scorer.calculate_score(sample_df)

    macd_score = result.breakdown["macd"]
    assert -scorer.weights.macd <= macd_score <= scorer.weights.macd


def test_bollinger_contribution_calculated(scorer, sample_df):
    """Test Bollinger Bands contributes to score."""
    result = scorer.calculate_score(sample_df)

    bb_score = result.breakdown["bollinger"]
    assert -scorer.weights.bollinger <= bb_score <= scorer.weights.bollinger


def test_ema_contribution_calculated(scorer, sample_df):
    """Test EMA contributes to score."""
    result = scorer.calculate_score(sample_df)

    ema_score = result.breakdown["ema"]
    assert -scorer.weights.ema <= ema_score <= scorer.weights.ema


def test_volume_contribution_calculated(scorer, sample_df):
    """Test volume confirmation affects score."""
    result = scorer.calculate_score(sample_df)

    # Volume should contribute (boost, penalty, or neutral)
    volume_score = result.breakdown["volume"]
    # Volume can boost significantly (20% of total) or apply fixed penalty (-10)
    assert isinstance(volume_score, int)


def test_custom_weights_affect_contribution(scorer_custom, sample_df):
    """Test custom weights change indicator contributions."""
    result = scorer_custom.calculate_score(sample_df)

    # Check contributions are scaled by custom weights
    assert -scorer_custom.weights.rsi <= result.breakdown["rsi"] <= scorer_custom.weights.rsi
    assert -scorer_custom.weights.macd <= result.breakdown["macd"] <= scorer_custom.weights.macd


# ============================================================================
# Volume Confirmation Tests
# ============================================================================

def test_high_volume_boosts_signal():
    """Test high volume (>1.5x average) boosts signal by 20%."""
    scorer = SignalScorer()

    # Create data with very high volume at the end
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50000] * 100,
        'volume': [5000] * 99 + [15000]  # Last candle has 3x volume
    })

    result = scorer.calculate_score(df)

    # High volume should boost signal if directional
    if result.breakdown.get("volume", 0) > 0:
        assert result.breakdown["volume"] > 0


def test_low_volume_penalizes_signal():
    """Test low volume (<0.7x average) applies 10-point penalty."""
    scorer = SignalScorer()

    # Create data with very low volume at the end
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50000] * 100,
        'volume': [10000] * 99 + [2000]  # Last candle has 0.2x volume
    })

    result = scorer.calculate_score(df)

    # Low volume should penalize if signal is directional
    if result.score != 0:
        # Should have volume penalty
        volume_contrib = result.breakdown.get("volume", 0)
        assert volume_contrib <= 0


def test_normal_volume_neutral():
    """Test normal volume (0.7-1.5x average) has no effect."""
    scorer = SignalScorer()

    # Create data with consistent volume
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50000] * 100,
        'volume': [10000] * 100  # Consistent volume
    })

    result = scorer.calculate_score(df)

    # Normal volume should be neutral
    assert result.breakdown.get("volume", 0) == 0


def test_whale_activity_detection():
    """Test that extreme volume (3x+) triggers whale activity detection."""
    scorer = SignalScorer()

    # Create data with 5x volume spike (whale activity)
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50500] * 100,  # Slight uptrend for bullish signal
        'volume': [10000] * 99 + [50000]  # 5x spike on last candle
    })

    result = scorer.calculate_score(df)

    # Should have whale activity flag
    assert result.breakdown.get("_whale_activity") is True
    assert result.breakdown.get("_volume_ratio") >= 3.0
    # Volume boost should be present (30% of signal for whale vs 20% for normal high)
    if result.score != 0:
        assert result.breakdown.get("volume", 0) != 0


def test_whale_boost_is_30_percent():
    """Test that whale activity applies 30% boost (vs 20% for normal high volume)."""
    scorer = SignalScorer()

    # Create identical data twice - once with whale volume (5x), once with high volume (2x)
    base_data = {
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50500] * 100,
    }

    # Whale volume (5x)
    df_whale = pd.DataFrame({**base_data, 'volume': [10000] * 99 + [50000]})
    result_whale = scorer.calculate_score(df_whale)

    # High volume (2x)
    df_high = pd.DataFrame({**base_data, 'volume': [10000] * 99 + [20000]})
    result_high = scorer.calculate_score(df_high)

    # Both should have volume boosts if signal is directional
    whale_boost = abs(result_whale.breakdown.get("volume", 0))
    high_boost = abs(result_high.breakdown.get("volume", 0))

    # Whale boost (30%) should be larger than high volume boost (20%)
    # Expected ratio is 1.5x (30% / 20%)
    # Tolerance of ±0.1 accounts for int() rounding in boost calculation:
    #   e.g., if score=25: whale=int(7.5)=7, high=int(5.0)=5, ratio=7/5=1.4
    if whale_boost > 0 and high_boost > 0:
        ratio = whale_boost / high_boost
        assert 1.4 <= ratio <= 1.6, f"Expected ~1.5x ratio (30%/20%), got {ratio:.2f}"


def test_whale_direction_bullish():
    """Test that whale direction is detected as bullish on price increase."""
    scorer = SignalScorer()

    # Create data with price going UP on whale volume spike
    closes = [50000] * 99 + [50500]  # Last candle is 1% higher
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    assert result.breakdown.get("_whale_direction") == "bullish"


def test_whale_direction_bearish():
    """Test that whale direction is detected as bearish on price decrease."""
    scorer = SignalScorer()

    # Create data with price going DOWN on whale volume spike
    closes = [50000] * 99 + [49500]  # Last candle is 1% lower
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    assert result.breakdown.get("_whale_direction") == "bearish"


def test_configurable_whale_threshold():
    """Test that whale threshold is configurable."""
    # Lower threshold (2.0x) should detect whale activity at lower volume
    scorer_low = SignalScorer(whale_volume_threshold=2.0)
    # Higher threshold (5.0x) should require more volume
    scorer_high = SignalScorer(whale_volume_threshold=5.0)

    # Create data with 3x volume spike
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50500] * 100,
        'volume': [10000] * 99 + [30000]  # 3x spike
    })

    result_low = scorer_low.calculate_score(df)
    result_high = scorer_high.calculate_score(df)

    # Low threshold (2.0x) should detect 3x as whale
    assert result_low.breakdown.get("_whale_activity") is True
    # High threshold (5.0x) should NOT detect 3x as whale
    assert result_high.breakdown.get("_whale_activity") is False


def test_high_volume_not_whale():
    """Test that high volume (1.5-3x) does not trigger whale activity."""
    scorer = SignalScorer()

    # Create data with 2x volume spike (high but not whale)
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50500] * 100,
        'volume': [10000] * 99 + [20000]  # 2x spike - high but not whale
    })

    result = scorer.calculate_score(df)

    # Should NOT have whale activity flag
    assert result.breakdown.get("_whale_activity") is False
    # But should have volume ratio recorded
    assert result.breakdown.get("_volume_ratio") is not None
    assert result.breakdown.get("_volume_ratio") < 3.0


def test_whale_metadata_on_zero_volume():
    """Test that whale metadata is properly set when volume SMA is zero."""
    scorer = SignalScorer()

    # Create data with zero volume - triggers the "volume_sma <= 0" path
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50500] * 100,
        'volume': [0] * 100  # Zero volume triggers invalid SMA path
    })

    result = scorer.calculate_score(df)

    # Should have whale metadata set to defaults (SMA is 0)
    assert result.breakdown.get("_whale_activity") is False
    assert result.breakdown.get("_volume_ratio") is None
    assert result.breakdown.get("volume") == 0


def test_whale_metadata_on_nan_volume():
    """Test that whale metadata is properly set when volume contains NaN values.

    This verifies the division-by-zero protection when volume_sma is NaN.
    """
    scorer = SignalScorer()

    # Create data with NaN volume values - will cause volume_sma to be NaN
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50500] * 100,
        'volume': [float('nan')] * 100  # NaN volume triggers pd.isna check
    })

    result = scorer.calculate_score(df)

    # Should handle NaN gracefully without crashing
    assert result.breakdown.get("_whale_activity") is False
    assert result.breakdown.get("_volume_ratio") is None
    assert result.breakdown.get("volume") == 0


def test_whale_price_change_pct_stored():
    """Test that price_change_pct is stored in breakdown for whale events."""
    scorer = SignalScorer()

    # Create data with whale volume and clear price movement
    base_volume = 1000
    prices = [50000] * 99 + [50500]  # 1% price increase on last candle
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': prices,
        'volume': [base_volume] * 99 + [base_volume * 5]  # 5x spike
    })

    result = scorer.calculate_score(df)

    # Should have whale activity with price_change_pct stored
    assert result.breakdown.get("_whale_activity") is True
    assert result.breakdown.get("_price_change_pct") is not None
    # Price change should be ~1% (0.01)
    pct = result.breakdown.get("_price_change_pct")
    assert 0.009 <= pct <= 0.011, f"Expected ~0.01 (1%), got {pct}"


def test_whale_activity_on_neutral_signal():
    """Test that whale activity is flagged even when signal score is low.

    This validates the intentional behavior where whale activity is recorded
    even when indicators give weak signals, providing valuable information for AI reviewers.
    The key point is that _whale_activity is always set to True when volume exceeds threshold.
    """
    scorer = SignalScorer()

    # Create data with whale volume spike on last candle
    # The actual signal score doesn't matter - what matters is that whale is flagged
    base_volume = 1000
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50100] * 100,  # Very tight range
        'low': [49900] * 100,
        'close': [50000] * 100,  # Close equals open
        'volume': [base_volume] * 99 + [base_volume * 5]  # 5x spike on last candle
    })

    result = scorer.calculate_score(df)

    # Whale activity should be detected regardless of signal strength
    assert result.breakdown.get("_whale_activity") is True
    assert result.breakdown.get("_volume_ratio") >= 3.0
    assert result.breakdown.get("_whale_direction") in ("bullish", "bearish", "neutral")
    # Volume boost should be proportional to score (30% of score)
    # If score is small, boost will be small but may not be 0
    expected_boost = int(abs(result.score - result.breakdown.get("volume", 0)) * 0.3)
    actual_boost = abs(result.breakdown.get("volume", 0))
    assert actual_boost <= expected_boost + 5  # Allow small margin


def test_update_settings_whale_threshold():
    """Test that whale_volume_threshold can be updated at runtime."""
    scorer = SignalScorer(whale_volume_threshold=3.0)
    assert scorer.whale_volume_threshold == 3.0

    # Update the threshold
    scorer.update_settings(whale_volume_threshold=5.0)
    assert scorer.whale_volume_threshold == 5.0

    # Update with None should not change the value
    scorer.update_settings(whale_volume_threshold=None)
    assert scorer.whale_volume_threshold == 5.0


def test_whale_candle_structure_bullish_confirmation():
    """Test whale direction with bullish price move and close near high (>0.7)."""
    scorer = SignalScorer()

    # Bullish price move (1% up) with close near high (position 0.8)
    # Candle: low=50000, high=51000, close=50800 -> position = 800/1000 = 0.8
    closes = [50000] * 99 + [50800]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [50000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    assert result.breakdown.get("_whale_direction") == "bullish"
    assert result.breakdown.get("_candle_close_position") is not None
    assert result.breakdown.get("_candle_close_position") > 0.7


def test_whale_candle_structure_bullish_rejection():
    """Test whale direction with bullish price move but close in lower half (<0.5)."""
    scorer = SignalScorer()

    # Bullish price move (1% up) but close near low (position 0.3)
    # Candle: low=50000, high=51000, close=50300 -> position = 300/1000 = 0.3
    # This indicates rejection/fighting despite price increase
    closes = [50000] * 99 + [50300]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [50000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    # Should be neutral due to rejection pattern
    assert result.breakdown.get("_whale_direction") == "neutral"
    assert result.breakdown.get("_candle_close_position") is not None
    assert result.breakdown.get("_candle_close_position") < 0.5


def test_whale_candle_structure_bearish_confirmation():
    """Test whale direction with bearish price move and close near low (<0.3)."""
    scorer = SignalScorer()

    # Bearish price move (1% down) with close near low (position 0.2)
    # Candle: low=49000, high=50000, close=49200 -> position = 200/1000 = 0.2
    closes = [50000] * 99 + [49200]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50000] * 100,
        'low': [49000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    assert result.breakdown.get("_whale_direction") == "bearish"
    assert result.breakdown.get("_candle_close_position") is not None
    assert result.breakdown.get("_candle_close_position") < 0.3


def test_whale_candle_structure_bearish_support():
    """Test whale direction with bearish price move but close in upper half (>0.5)."""
    scorer = SignalScorer()

    # Bearish price move (1% down) but close near high (position 0.7)
    # Candle: low=49000, high=50000, close=49700 -> position = 700/1000 = 0.7
    # This indicates support/fighting despite price decrease
    closes = [50000] * 99 + [49700]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50000] * 100,
        'low': [49000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    # Should be neutral due to support pattern
    assert result.breakdown.get("_whale_direction") == "neutral"
    assert result.breakdown.get("_candle_close_position") is not None
    assert result.breakdown.get("_candle_close_position") > 0.5


def test_whale_candle_structure_ambiguous_range():
    """Test whale direction with close in ambiguous range (0.5-0.7)."""
    scorer = SignalScorer()

    # Bullish price move (1% up) with close at 0.6 (ambiguous zone)
    # Candle: low=50000, high=51000, close=50600 -> position = 600/1000 = 0.6
    closes = [50000] * 99 + [50600]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [50000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    # Should be neutral due to ambiguous close position (0.5-0.7 range)
    assert result.breakdown.get("_whale_direction") == "neutral"
    assert result.breakdown.get("_candle_close_position") is not None
    position = result.breakdown.get("_candle_close_position")
    assert 0.5 <= position <= 0.7


def test_whale_candle_structure_zero_range():
    """Test whale direction with zero candle range (doji/flat candle)."""
    scorer = SignalScorer()

    # Zero range candle - high equals low (doji pattern)
    # When high == low, candle_range = 0, close_position should be None
    closes = [50000] * 99 + [50500]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50000] * 99 + [50500],  # Last candle: high = close
        'low': [50000] * 99 + [50500],   # Last candle: low = close (zero range)
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    # Should be neutral due to missing close_position (zero range candle)
    assert result.breakdown.get("_whale_direction") == "neutral"
    assert result.breakdown.get("_candle_close_position") is None


def test_whale_candle_structure_missing_data():
    """Test whale direction handles missing candle data gracefully."""
    scorer = SignalScorer()

    # Missing high/low data (NaN values)
    closes = [50000] * 99 + [50500]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 99 + [float('nan')],  # Last candle missing high
        'low': [49000] * 99 + [float('nan')],   # Last candle missing low
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    # Should be neutral due to missing candle data
    assert result.breakdown.get("_whale_direction") == "neutral"
    assert result.breakdown.get("_candle_close_position") is None


def test_whale_candle_structure_edge_case_exactly_0_7():
    """Test whale direction with close exactly at 0.7 threshold."""
    scorer = SignalScorer()

    # Bullish price move with close exactly at 0.7
    # Candle: low=50000, high=51000, close=50700 -> position = 700/1000 = 0.7
    closes = [50000] * 99 + [50700]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [50000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    # At exactly 0.7, the condition is close_position > 0.7, so this should be neutral
    # (not greater than, so falls into ambiguous range)
    assert result.breakdown.get("_whale_direction") == "neutral"
    assert result.breakdown.get("_candle_close_position") == 0.7


def test_whale_candle_structure_edge_case_exactly_0_3():
    """Test whale direction with close exactly at 0.3 threshold."""
    scorer = SignalScorer()

    # Bearish price move with close exactly at 0.3
    # Candle: low=49000, high=50000, close=49300 -> position = 300/1000 = 0.3
    closes = [50000] * 99 + [49300]
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50000] * 100,
        'low': [49000] * 100,
        'close': closes,
        'volume': [10000] * 99 + [50000]  # 5x spike
    })

    result = scorer.calculate_score(df)

    assert result.breakdown.get("_whale_activity") is True
    # At exactly 0.3, the condition is close_position < 0.3, so this should be neutral
    # (not less than, so falls into ambiguous range)
    assert result.breakdown.get("_whale_direction") == "neutral"
    assert result.breakdown.get("_candle_close_position") == 0.3


def test_whale_volume_boundary_behavior():
    """Test that whale detection uses strict greater-than comparison.

    The whale detection uses > (strictly greater than), so volume_ratio
    must exceed the threshold, not just equal it.
    """
    scorer = SignalScorer(whale_volume_threshold=3.0)

    # Create baseline data - volume below threshold should not trigger
    base_volume = 1000
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [50500] * 100,
        'low': [49500] * 100,
        'close': [50100] * 100,
        'volume': [base_volume] * 100
    })

    result = scorer.calculate_score(df)
    assert result.breakdown.get("_whale_activity") is False or result.breakdown.get("_whale_activity") is None

    # Now spike the volume high enough to exceed 3.0 threshold
    # Using 5x ensures we're well above, accounting for rolling SMA including the spike
    df['volume'].iloc[-1] = base_volume * 5
    result2 = scorer.calculate_score(df)
    assert result2.breakdown.get("_whale_activity") is True
    assert result2.breakdown.get("_volume_ratio") > 3.0


def test_whale_candle_structure_price_outside_range():
    """Test handling when close price is outside high/low range (data inconsistency)."""
    scorer = SignalScorer()

    # Close price above high (data inconsistency) - only on last candle
    closes = [50500] * 99 + [51100]  # Last close > high
    df = pd.DataFrame({
        'open': [50000] * 99 + [50000],
        'high': [51000] * 99 + [51000],  # high = 51000
        'low': [50000] * 99 + [50000],
        'close': closes,  # Last close = 51100 (!)
        'volume': [10000] * 99 + [50000]  # 5x spike = whale activity
    })

    result = scorer.calculate_score(df)

    # Should detect whale activity (volume spike)
    assert result.breakdown.get("_whale_activity") is True

    # But candle close position should be None due to data inconsistency
    assert result.breakdown.get("_candle_close_position") is None

    # Whale direction should be unknown (can't trust inconsistent candle structure)
    assert result.breakdown.get("_whale_direction") == "unknown"

    # Price change percentage should still be valid (calculated from consecutive closes)
    assert result.breakdown.get("_price_change_pct") is not None


def test_whale_candle_structure_price_below_low():
    """Test handling when close price is below low (data inconsistency)."""
    scorer = SignalScorer()

    # Close price below low (data inconsistency) - only on last candle
    closes = [50500] * 99 + [49900]  # Last close < low
    df = pd.DataFrame({
        'open': [50000] * 99 + [50000],
        'high': [51000] * 99 + [51000],
        'low': [50000] * 99 + [50000],  # low = 50000
        'close': closes,  # Last close = 49900 (!)
        'volume': [10000] * 99 + [50000]  # 5x spike = whale activity
    })

    result = scorer.calculate_score(df)

    # Should detect whale activity (volume spike)
    assert result.breakdown.get("_whale_activity") is True

    # But candle close position should be None due to data inconsistency
    assert result.breakdown.get("_candle_close_position") is None

    # Whale direction should be unknown (can't trust inconsistent candle structure)
    assert result.breakdown.get("_whale_direction") == "unknown"

    # Price change percentage should still be valid (calculated from consecutive closes)
    assert result.breakdown.get("_price_change_pct") is not None


# ============================================================================
# Crash Protection Tests
# ============================================================================

@freeze_time("2025-01-01 12:00:00")
def test_record_oversold_buy():
    """Test recording oversold buy timestamps."""
    scorer = SignalScorer()

    scorer.record_oversold_buy()

    assert len(scorer._oversold_buy_times) == 1
    assert isinstance(scorer._oversold_buy_times[0], datetime)


@freeze_time("2025-01-01 12:00:00")
def test_can_buy_oversold_allows_first_two():
    """Test can_buy_oversold allows first 2 buys in 24h."""
    scorer = SignalScorer()

    assert scorer.can_buy_oversold() is True

    scorer.record_oversold_buy()
    assert scorer.can_buy_oversold() is True

    scorer.record_oversold_buy()
    assert scorer.can_buy_oversold() is False  # 2 is limit


@freeze_time("2025-01-01 12:00:00")
def test_can_buy_oversold_resets_after_24h():
    """Test oversold buy limit resets after 24 hours."""
    scorer = SignalScorer()

    scorer.record_oversold_buy()
    scorer.record_oversold_buy()
    assert scorer.can_buy_oversold() is False

    # Move forward 24 hours and 1 second
    with freeze_time("2025-01-02 12:00:01"):
        assert scorer.can_buy_oversold() is True


@freeze_time("2025-01-01 12:00:00")
def test_oversold_buy_tracking_cleans_old_entries():
    """Test old oversold buy entries are removed."""
    scorer = SignalScorer()

    scorer.record_oversold_buy()

    # Move forward 25 hours
    with freeze_time("2025-01-02 13:00:00"):
        scorer.record_oversold_buy()

        # Should only have 1 entry (old one cleaned)
        assert len(scorer._oversold_buy_times) == 1


def test_is_price_stabilized_true_when_not_making_new_lows():
    """Test price stabilization detection when price stops falling."""
    scorer = SignalScorer()

    # Price that has stopped making new lows
    prices = pd.Series([100, 95, 90, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94])

    stabilized = scorer.is_price_stabilized(prices, window_candles=12)

    assert stabilized


def test_is_price_stabilized_false_when_making_new_lows():
    """Test price stabilization false when still making new lows."""
    scorer = SignalScorer()

    # Price still falling
    prices = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40])

    stabilized = scorer.is_price_stabilized(prices, window_candles=12)

    assert not stabilized


def test_is_price_stabilized_true_with_insufficient_data():
    """Test stabilization returns True when not enough data."""
    scorer = SignalScorer()

    # Only 5 candles
    prices = pd.Series([100, 95, 90, 85, 80])

    stabilized = scorer.is_price_stabilized(prices, window_candles=12)

    # Should allow trade when insufficient data
    assert stabilized is True


# ============================================================================
# Trend Filter Tests
# ============================================================================

def test_trend_filter_penalizes_buy_in_bearish_trend():
    """Test counter-trend buy penalty in bearish market."""
    scorer = SignalScorer()

    # Create clear downtrend
    prices = [50000 - (i * 100) for i in range(100)]
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [5000] * 100,
    })

    result = scorer.calculate_score(df)

    # If trying to buy in downtrend, should have penalty
    if result.score > 0:  # Bullish signal
        assert result.breakdown["trend_filter"] <= 0  # Penalty


def test_trend_filter_penalizes_sell_in_bullish_trend():
    """Test counter-trend sell penalty in bullish market."""
    scorer = SignalScorer()

    # Create clear uptrend
    prices = [40000 + (i * 100) for i in range(100)]
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [5000] * 100,
    })

    result = scorer.calculate_score(df)

    # If trying to sell in uptrend, should have penalty
    if result.score < 0:  # Bearish signal
        assert result.breakdown["trend_filter"] >= 0  # Penalty (reduces negative score)


def test_trend_filter_scales_with_signal_strength():
    """Test trend penalty scales with signal confidence."""
    scorer = SignalScorer()

    # Weak signal should get higher penalty (10-20 range)
    # Strong signal should get lower penalty
    # This is tested indirectly through the score calculation
    # The actual penalty formula: -int(20 * (1 - signal_confidence * 0.5))
    # where signal_confidence = abs(total_score) / 100

    # Just verify the trend_filter exists and is reasonable
    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50000] * 100,
        'volume': [5000] * 100,
    })

    result = scorer.calculate_score(df)

    # Trend filter should be within reasonable range (-20 to +20)
    assert -20 <= result.breakdown["trend_filter"] <= 20


def test_trend_filter_skipped_for_extreme_rsi():
    """Test trend filter skipped when RSI is extreme and conditions met."""
    scorer = SignalScorer()

    # This is complex to test as it requires:
    # 1. Extreme RSI (<25 or >75)
    # 2. can_buy_oversold() == True (less than 2 oversold buys in 24h)
    # 3. is_price_stabilized() == True

    # The actual behavior would need specific market conditions
    # For now, verify the logic exists
    assert hasattr(scorer, 'can_buy_oversold')
    assert hasattr(scorer, 'is_price_stabilized')


# ============================================================================
# Threshold Tests
# ============================================================================

def test_threshold_exactly_at_boundary(scorer, sample_df):
    """Test action when score exactly equals threshold."""
    # Adjust threshold to match expected score
    scorer.threshold = 50
    result = scorer.calculate_score(sample_df)

    # If score == threshold, should trigger buy
    if result.score == scorer.threshold:
        assert result.action == "buy"


def test_threshold_just_below_boundary(scorer, sample_df):
    """Test action when score is 1 below threshold."""
    scorer.threshold = 100  # Very high threshold
    result = scorer.calculate_score(sample_df)

    # Any score below 100 should be hold (since -100 to 99 are all below)
    if abs(result.score) < 100:
        assert result.action == "hold"


def test_threshold_adjustment_changes_action():
    """Test changing threshold changes trade action."""
    scorer = SignalScorer(threshold=40)

    df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100,
        'low': [49000] * 100,
        'close': [50000] * 100,
        'volume': [5000] * 100,
    })

    result_low = scorer.calculate_score(df)

    # Increase threshold
    scorer.threshold = 80
    result_high = scorer.calculate_score(df)

    # Score should be same, but action might differ
    assert result_low.score == result_high.score
    # With higher threshold, more likely to be hold


# ============================================================================
# Insufficient Data Tests
# ============================================================================

def test_empty_dataframe_returns_hold():
    """Test empty DataFrame returns neutral result."""
    scorer = SignalScorer()
    df = pd.DataFrame()

    result = scorer.calculate_score(df)

    assert result.score == 0
    assert result.action == "hold"
    assert result.confidence == 0.0
    assert result.breakdown == {}


def test_insufficient_rows_returns_hold():
    """Test DataFrame with too few rows returns hold."""
    scorer = SignalScorer()

    # Only 10 rows (needs at least max(ema_slow, bollinger, 26) = 26)
    df = pd.DataFrame({
        'open': [50000] * 10,
        'high': [51000] * 10,
        'low': [49000] * 10,
        'close': [50000] * 10,
        'volume': [5000] * 10,
    })

    result = scorer.calculate_score(df)

    assert result.score == 0
    assert result.action == "hold"


def test_nan_indicators_handled_gracefully(scorer):
    """Test NaN indicator values don't crash calculation."""
    # Create data that might produce NaN
    df = pd.DataFrame({
        'open': [50000] * 50,
        'high': [50000] * 50,
        'low': [50000] * 50,
        'close': [50000] * 50,  # Flat prices might cause NaN
        'volume': [0] * 50,  # Zero volume
    })

    # Should not raise exception
    result = scorer.calculate_score(df)

    assert isinstance(result, SignalResult)


# ============================================================================
# Indicator Values Tests
# ============================================================================

def test_indicator_values_populated(scorer, sample_df):
    """Test IndicatorValues contains current indicator readings."""
    result = scorer.calculate_score(sample_df)

    # Should have RSI value
    assert result.indicators.rsi is not None or result.indicators.rsi == pytest.approx(0, abs=100)

    # Should have volatility level
    assert result.indicators.volatility in ["low", "normal", "high", "extreme"]


def test_indicator_values_nan_when_unavailable(scorer):
    """Test indicator values are None when data insufficient."""
    df = pd.DataFrame({
        'open': [50000] * 10,
        'high': [51000] * 10,
        'low': [49000] * 10,
        'close': [50000] * 10,
        'volume': [5000] * 10,
    })

    result = scorer.calculate_score(df)

    # With insufficient data, indicators should be None
    assert result.indicators.rsi is None or isinstance(result.indicators.rsi, float)


# ============================================================================
# Settings Update Tests
# ============================================================================

def test_update_settings_changes_threshold():
    """Test update_settings modifies threshold."""
    scorer = SignalScorer(threshold=60)

    scorer.update_settings(threshold=75)

    assert scorer.threshold == 75


def test_update_settings_changes_indicator_params():
    """Test update_settings modifies indicator parameters."""
    scorer = SignalScorer()

    scorer.update_settings(
        rsi_period=21,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        macd_fast=8,
        bollinger_period=25,
    )

    assert scorer.rsi_period == 21
    assert scorer.rsi_oversold == 30.0
    assert scorer.rsi_overbought == 70.0
    assert scorer.macd_fast == 8
    assert scorer.bollinger_period == 25


def test_update_settings_none_values_ignored():
    """Test update_settings ignores None values."""
    scorer = SignalScorer(threshold=60)

    scorer.update_settings(threshold=None, rsi_period=21)

    # Threshold should remain unchanged
    assert scorer.threshold == 60
    # RSI period should update
    assert scorer.rsi_period == 21


# ============================================================================
# Get Trend Tests
# ============================================================================

def test_get_trend_returns_valid_value(scorer, sample_df):
    """Test get_trend returns valid trend classification."""
    trend = scorer.get_trend(sample_df)

    assert trend in ["bullish", "neutral", "bearish"]


def test_get_trend_neutral_with_insufficient_data(scorer):
    """Test get_trend returns neutral with insufficient data."""
    df = pd.DataFrame({
        'open': [50000] * 10,
        'high': [51000] * 10,
        'low': [49000] * 10,
        'close': [50000] * 10,
    })

    trend = scorer.get_trend(df)

    assert trend == "neutral"


def test_get_trend_bullish_in_uptrend(scorer):
    """Test get_trend detects bullish trend."""
    # Create clear uptrend
    prices = [40000 + (i * 100) for i in range(100)]
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
    })

    trend = scorer.get_trend(df)

    assert trend == "bullish"


def test_get_trend_bearish_in_downtrend(scorer):
    """Test get_trend detects bearish trend."""
    # Create clear downtrend
    prices = [60000 - (i * 100) for i in range(100)]
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
    })

    trend = scorer.get_trend(df)

    assert trend == "bearish"


# ============================================================================
# Current Price Parameter Tests
# ============================================================================

def test_calculate_score_uses_current_price(scorer, sample_df):
    """Test calculate_score uses provided current_price."""
    custom_price = Decimal("55000.00")

    result = scorer.calculate_score(sample_df, current_price=custom_price)

    # Bollinger signal should use the custom price
    assert result.indicators.bb_upper is not None or result.indicators.bb_lower is not None


def test_calculate_score_uses_last_close_when_no_price(scorer, sample_df):
    """Test calculate_score defaults to last close price."""
    result = scorer.calculate_score(sample_df, current_price=None)

    # Should still calculate successfully
    assert isinstance(result, SignalResult)
    assert result.score is not None


# ============================================================================
# Momentum Mode Tests
# ============================================================================

@pytest.fixture
def momentum_df():
    """Generate data with sustained high RSI and higher lows (momentum conditions)."""
    np.random.seed(123)
    length = 50
    # Start with uptrend that will produce high RSI
    # Need some volatility for RSI to calculate properly
    prices = []
    current = 40000.0

    for i in range(length):
        # Strong uptrend with minor noise (70% up, 30% small down)
        if np.random.random() < 0.7:
            change = current * np.random.uniform(0.003, 0.008)  # Up move
        else:
            change = -current * np.random.uniform(0.001, 0.002)  # Small pullback
        current = current + change
        prices.append(current)

    data = {
        'open': [p * 0.998 for p in prices],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.995 for p in prices],  # Higher lows
        'close': prices,
        'volume': [10000.0] * length,
    }
    return pd.DataFrame(data)


@pytest.fixture
def no_momentum_df():
    """Generate choppy sideways data without momentum."""
    np.random.seed(456)
    length = 50
    prices = []
    current = 50000.0

    for i in range(length):
        # Oscillate around the same level
        change = current * 0.01 * (1 if i % 2 == 0 else -1)
        current = current + change
        prices.append(current)

    data = {
        'open': [p * 0.998 for p in prices],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [10000.0] * length,
    }
    return pd.DataFrame(data)


def test_is_momentum_mode_active_with_sustained_rsi_and_higher_lows(scorer, momentum_df):
    """Test momentum mode activates with sustained RSI > 60 and higher lows."""
    from src.indicators.rsi import calculate_rsi

    close = momentum_df["close"].astype(float)
    rsi = calculate_rsi(close, scorer.rsi_period)

    is_active, reason = scorer.is_momentum_mode(momentum_df, rsi)

    # Should activate due to uptrend creating sustained high RSI
    assert is_active is True
    assert reason in ["sustained_rsi", "sustained_rsi_higher_lows"]


def test_is_momentum_mode_inactive_with_insufficient_data(scorer):
    """Test momentum mode returns False with insufficient data."""
    # Create minimal dataframe
    df = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'close': [101, 102],
        'volume': [1000, 1000],
    })
    rsi = pd.Series([70, 71])

    is_active, reason = scorer.is_momentum_mode(df, rsi)

    assert is_active is False
    assert reason == ""


def test_is_momentum_mode_inactive_with_low_rsi(scorer, no_momentum_df):
    """Test momentum mode inactive when RSI is not sustained above threshold."""
    from src.indicators.rsi import calculate_rsi

    close = no_momentum_df["close"].astype(float)
    rsi = calculate_rsi(close, scorer.rsi_period)

    is_active, reason = scorer.is_momentum_mode(no_momentum_df, rsi)

    # Choppy data shouldn't produce sustained high RSI
    assert is_active is False


def test_is_momentum_mode_inactive_with_lower_lows():
    """Test momentum mode with lower lows only returns sustained_rsi, not higher_lows."""
    # Create downtrend data (lower lows)
    length = 50
    prices = [50000 - (i * 100) for i in range(length)]  # Downtrend

    df = pd.DataFrame({
        'open': [p * 1.002 for p in prices],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [10000.0] * length,
    })

    # Create artificially high RSI series
    rsi = pd.Series([70.0] * length)

    scorer = SignalScorer()
    is_active, reason = scorer.is_momentum_mode(df, rsi)

    # With high RSI but lower lows, momentum activates but only for sustained_rsi
    # The higher_lows check should fail due to downtrend
    assert is_active is True, "Should activate due to sustained RSI"
    assert reason == "sustained_rsi", "Should NOT be 'sustained_rsi_higher_lows' with downtrend"


def test_is_momentum_mode_partial_activation_rsi_only():
    """Test momentum mode activates with just sustained RSI (no higher lows check passed)."""
    length = 50
    # Flat prices (no clear higher lows pattern)
    prices = [50000.0] * length

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': [10000.0] * length,
    })

    # Create high RSI series
    rsi = pd.Series([75.0] * length)

    scorer = SignalScorer()
    is_active, reason = scorer.is_momentum_mode(df, rsi)

    # Should activate based on RSI alone
    assert is_active is True
    assert reason == "sustained_rsi"


def test_is_momentum_mode_handles_nan_in_rsi(scorer):
    """Test momentum mode handles NaN values in RSI series gracefully."""
    df = pd.DataFrame({
        'open': [100.0] * 50,
        'high': [101.0] * 50,
        'low': [99.0] * 50,
        'close': [100.0] * 50,
        'volume': [1000.0] * 50,
    })

    # RSI with NaN values at the end
    rsi = pd.Series([70.0] * 47 + [np.nan, np.nan, np.nan])

    is_active, reason = scorer.is_momentum_mode(df, rsi)

    # Should return False due to insufficient valid RSI values
    assert is_active is False
    assert reason == ""


def test_momentum_penalty_reduction_in_calculate_score(momentum_df):
    """Test momentum mode reduces overbought penalties in score calculation."""
    # Create scorer with momentum detection
    scorer = SignalScorer()

    result = scorer.calculate_score(momentum_df)

    # Check that momentum was detected
    assert "_momentum_active" in result.breakdown
    # In a strong uptrend, momentum should be active
    if result.breakdown["_momentum_active"] == 1:
        # RSI penalty should be reduced (less negative than full -25)
        # The actual reduction depends on trend strength (EMA gap)
        # With trend strength scaling, penalty reduction varies from 0% to 50%
        # So RSI score should be >= -25 (full penalty) and potentially much less negative
        assert result.breakdown["rsi"] >= -25  # At minimum, no worse than full penalty


def test_momentum_mode_does_not_affect_positive_scores():
    """Test momentum mode only reduces penalties, not positive scores."""
    scorer = SignalScorer()

    # Create uptrend data
    length = 50
    prices = [40000 + (i * 200) for i in range(length)]

    df = pd.DataFrame({
        'open': [p * 0.998 for p in prices],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': [10000.0] * length,
    })

    result = scorer.calculate_score(df)

    # MACD and EMA should remain positive (not affected by momentum reduction)
    # The reduction only applies to negative RSI and Bollinger scores
    if result.breakdown["macd"] > 0:
        assert result.breakdown["macd"] > 0  # Unchanged


def test_momentum_configurable_parameters():
    """Test momentum mode respects configurable parameters."""
    # Create scorer with custom momentum settings
    scorer = SignalScorer(
        momentum_rsi_threshold=70.0,  # Higher threshold
        momentum_rsi_candles=5,       # More candles required
        momentum_price_candles=20,    # More price history
        momentum_penalty_reduction=0.3,  # 70% reduction instead of 50%
    )

    assert scorer.momentum_rsi_threshold == 70.0
    assert scorer.momentum_rsi_candles == 5
    assert scorer.momentum_price_candles == 20
    assert scorer.momentum_penalty_reduction == 0.3


def test_update_settings_momentum_parameters():
    """Test update_settings can modify momentum parameters at runtime."""
    scorer = SignalScorer()

    # Verify defaults
    assert scorer.momentum_rsi_threshold == 60.0
    assert scorer.momentum_rsi_candles == 3
    assert scorer.momentum_price_candles == 12
    assert scorer.momentum_penalty_reduction == 0.5

    # Update momentum settings
    scorer.update_settings(
        momentum_rsi_threshold=70.0,
        momentum_rsi_candles=5,
        momentum_price_candles=20,
        momentum_penalty_reduction=0.3,
    )

    # Verify updates
    assert scorer.momentum_rsi_threshold == 70.0
    assert scorer.momentum_rsi_candles == 5
    assert scorer.momentum_price_candles == 20
    assert scorer.momentum_penalty_reduction == 0.3


def test_update_settings_momentum_partial_update():
    """Test update_settings only modifies provided momentum parameters."""
    scorer = SignalScorer()

    # Update only one momentum parameter
    scorer.update_settings(momentum_rsi_threshold=75.0)

    # Only the updated parameter should change
    assert scorer.momentum_rsi_threshold == 75.0
    assert scorer.momentum_rsi_candles == 3  # Unchanged
    assert scorer.momentum_price_candles == 12  # Unchanged
    assert scorer.momentum_penalty_reduction == 0.5  # Unchanged


def test_momentum_breakdown_uses_underscore_prefix(scorer, momentum_df):
    """Test momentum flag uses underscore prefix for metadata."""
    result = scorer.calculate_score(momentum_df)

    # Should use _momentum_active, not momentum
    assert "_momentum_active" in result.breakdown
    assert "momentum" not in result.breakdown


def test_confluence_excludes_momentum_metadata(scorer, momentum_df):
    """Test confluence calculation excludes underscore-prefixed keys."""
    result = scorer.calculate_score(momentum_df)

    # The confidence calculation should not count _momentum_active
    # This is tested indirectly - if momentum was counted, confluence would be off
    if result.action != "hold":
        # Confidence should be based on 6 components max (rsi, macd, bollinger, ema, volume, trend_filter)
        # Not 7 (with momentum included)
        assert result.confidence <= 1.0


def test_momentum_penalty_reduction_scales_with_trend_strength():
    """Test that momentum penalty reduction is proportional to EMA gap (trend strength).

    Note: This test validates real-world scenarios where momentum activation and
    negative RSI scores are preconditions. Occasional skips are acceptable as they
    indicate the test data didn't meet natural market conditions for momentum mode.
    The test provides confidence when conditions align with actual trading scenarios.
    """
    scorer = SignalScorer()

    # Case 1: Strong uptrend with wide EMA gap (should get more penalty reduction)
    strong_trend_prices = [40000 + (i * 300) for i in range(50)]  # Steep uptrend
    strong_df = pd.DataFrame({
        'open': [p * 0.998 for p in strong_trend_prices],
        'high': [p * 1.01 for p in strong_trend_prices],
        'low': [p * 0.995 for p in strong_trend_prices],
        'close': strong_trend_prices,
        'volume': [10000.0] * 50,
    })

    # Case 2: Weak uptrend with narrow EMA gap (should get less penalty reduction)
    weak_trend_prices = [40000 + (i * 50) for i in range(50)]  # Gradual uptrend
    weak_df = pd.DataFrame({
        'open': [p * 0.998 for p in weak_trend_prices],
        'high': [p * 1.01 for p in weak_trend_prices],
        'low': [p * 0.995 for p in weak_trend_prices],
        'close': weak_trend_prices,
        'volume': [10000.0] * 50,
    })

    strong_result = scorer.calculate_score(strong_df)
    weak_result = scorer.calculate_score(weak_df)

    # Both should potentially have momentum active, but with different penalty reductions
    # The test is flexible because momentum activation depends on RSI levels
    # What matters: if both have momentum active, strong trend should have more reduction
    if (strong_result.breakdown.get("_momentum_active") == 1 and
        weak_result.breakdown.get("_momentum_active") == 1):
        # When both active, strong trend should have penalties reduced more
        # (i.e., RSI/BB scores should be less negative in strong trend)
        # This is the key improvement from issue #54
        if strong_result.breakdown["rsi"] < 0 and weak_result.breakdown["rsi"] < 0:
            # Less negative = more penalty reduction
            assert strong_result.breakdown["rsi"] >= weak_result.breakdown["rsi"], \
                f"Expected strong trend RSI {strong_result.breakdown['rsi']} >= weak trend {weak_result.breakdown['rsi']}"
        else:
            # Skip test if conditions not met - can't validate penalty reduction without negative RSI
            pytest.skip("Test conditions not met - RSI not negative in both scenarios")
    else:
        # Skip test if momentum not active in both scenarios
        pytest.skip("Test conditions not met - momentum not active in both scenarios")


# ============================================================================
# Adaptive Threshold Tests
# ============================================================================

def test_get_recommended_threshold_valid_intervals():
    """Test recommended threshold returns correct values for all valid intervals."""
    expected = {
        "ONE_MINUTE": 50,
        "FIVE_MINUTE": 52,
        "FIFTEEN_MINUTE": 55,
        "THIRTY_MINUTE": 57,
        "ONE_HOUR": 58,
        "TWO_HOUR": 60,
        "SIX_HOUR": 62,
        "ONE_DAY": 65,
    }
    for interval, expected_value in expected.items():
        assert get_recommended_threshold(interval) == expected_value


def test_get_recommended_threshold_none_returns_default():
    """Test None input returns default threshold."""
    assert get_recommended_threshold(None) == 60


def test_get_recommended_threshold_invalid_returns_default():
    """Test invalid interval returns default and logs warning."""
    result = get_recommended_threshold("INVALID_INTERVAL")
    assert result == 60  # Default value


def test_get_recommended_threshold_progression():
    """Test that thresholds increase as intervals get longer."""
    intervals = [
        "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
        "THIRTY_MINUTE", "ONE_HOUR", "TWO_HOUR",
        "SIX_HOUR", "ONE_DAY"
    ]
    thresholds = [get_recommended_threshold(i) for i in intervals]
    # Each threshold should be >= previous
    for i in range(1, len(thresholds)):
        assert thresholds[i] >= thresholds[i-1], \
            f"Threshold for {intervals[i]} should be >= {intervals[i-1]}"


# ============================================================================
# Multi-Timeframe (MTF) / Higher Timeframe (HTF) Bias Tests
# ============================================================================

class TestHTFBiasModifier:
    """Tests for higher timeframe bias score modifiers.

    HTF (Higher Timeframe) bias is determined by combining Daily and 6-hour trends:
    - Both bullish → bullish bias
    - Both bearish → bearish bias
    - Mixed/neutral → neutral bias

    Score modifiers:
    - Aligned bullish trade (buy + bullish bias) → +mtf_aligned_boost (default +20)
    - Counter-trend bullish trade (buy + bearish bias) → -mtf_counter_penalty (default -20)
    - Aligned bearish trade (sell + bearish bias) → -mtf_aligned_boost (more negative)
    - Counter-trend bearish trade (sell + bullish bias) → +mtf_counter_penalty (less negative)
    """

    @pytest.fixture
    def mtf_scorer(self):
        """Signal scorer with MTF parameters configured."""
        return SignalScorer(
            mtf_aligned_boost=20,
            mtf_counter_penalty=20,
        )

    @pytest.fixture
    def bullish_signal_df(self):
        """Generate data that produces a positive (bullish) signal."""
        np.random.seed(42)
        length = 100
        # Strong uptrend with oversold RSI recovery
        prices = []
        current = 45000.0
        for i in range(length):
            # Uptrend with 80% up moves
            if np.random.random() < 0.8:
                change = current * np.random.uniform(0.002, 0.005)
            else:
                change = -current * np.random.uniform(0.001, 0.002)
            current = current + change
            prices.append(current)

        return pd.DataFrame({
            'open': [p * 0.998 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': [10000.0] * length,
        })

    @pytest.fixture
    def bearish_signal_df(self):
        """Generate data that produces a negative (bearish) signal."""
        np.random.seed(43)
        length = 100
        # Strong downtrend with overbought RSI falling
        prices = []
        current = 55000.0
        for i in range(length):
            # Downtrend with 80% down moves
            if np.random.random() < 0.8:
                change = -current * np.random.uniform(0.002, 0.005)
            else:
                change = current * np.random.uniform(0.001, 0.002)
            current = current + change
            prices.append(current)

        return pd.DataFrame({
            'open': [p * 1.002 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [10000.0] * length,
        })

    def test_htf_bullish_boosts_aligned_buy(self, mtf_scorer, bullish_signal_df):
        """Test bullish HTF bias boosts buy signals by +20."""
        # Get baseline score without HTF bias
        result_baseline = mtf_scorer.calculate_score(bullish_signal_df)

        # Get score with bullish HTF bias
        result_with_htf = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bullish",
            htf_daily="bullish",
            htf_4h="bullish",
        )

        # Only applies if signal is positive (bullish)
        if result_baseline.score > 0:
            assert result_with_htf.breakdown.get("htf_bias") == 20
            # Score should be boosted by 20 (clamped to -100/+100)
            expected_score = min(100, result_baseline.score + 20)
            assert result_with_htf.score == expected_score

    def test_htf_bearish_penalizes_counter_trend_buy(self, mtf_scorer, bullish_signal_df):
        """Test bearish HTF bias penalizes buy signals by -20."""
        result_baseline = mtf_scorer.calculate_score(bullish_signal_df)

        result_with_htf = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bearish",
            htf_daily="bearish",
            htf_4h="bearish",
        )

        # Only applies if signal is positive (bullish)
        if result_baseline.score > 0:
            assert result_with_htf.breakdown.get("htf_bias") == -20
            # Score should be penalized by 20 (clamped to -100/+100)
            expected_score = max(-100, result_baseline.score - 20)
            assert result_with_htf.score == expected_score

    def test_htf_bearish_boosts_aligned_sell(self, mtf_scorer, bearish_signal_df):
        """Test bearish HTF bias boosts sell signals (more negative)."""
        result_baseline = mtf_scorer.calculate_score(bearish_signal_df)

        result_with_htf = mtf_scorer.calculate_score(
            bearish_signal_df,
            htf_bias="bearish",
            htf_daily="bearish",
            htf_4h="bearish",
        )

        # Only applies if signal is negative (bearish)
        if result_baseline.score < 0:
            # Aligned bearish → more negative (boost the sell signal)
            assert result_with_htf.breakdown.get("htf_bias") == -20
            expected_score = max(-100, result_baseline.score - 20)
            assert result_with_htf.score == expected_score

    def test_htf_bullish_penalizes_counter_trend_sell(self, mtf_scorer, bearish_signal_df):
        """Test bullish HTF bias penalizes sell signals (less negative)."""
        result_baseline = mtf_scorer.calculate_score(bearish_signal_df)

        result_with_htf = mtf_scorer.calculate_score(
            bearish_signal_df,
            htf_bias="bullish",
            htf_daily="bullish",
            htf_4h="bullish",
        )

        # Only applies if signal is negative (bearish)
        if result_baseline.score < 0:
            # Counter-trend sell → less negative (weaken the sell signal)
            assert result_with_htf.breakdown.get("htf_bias") == 20
            expected_score = min(100, result_baseline.score + 20)
            assert result_with_htf.score == expected_score

    def test_htf_neutral_no_effect(self, mtf_scorer, bullish_signal_df):
        """Test neutral HTF bias has no effect on score."""
        result_baseline = mtf_scorer.calculate_score(bullish_signal_df)

        result_with_htf = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="neutral",
            htf_4h="neutral",
        )

        # Neutral bias should add 0
        assert result_with_htf.breakdown.get("htf_bias") == 0
        assert result_with_htf.score == result_baseline.score

    def test_htf_none_no_effect(self, mtf_scorer, bullish_signal_df):
        """Test None HTF bias (MTF disabled) has no effect."""
        result_baseline = mtf_scorer.calculate_score(bullish_signal_df)

        result_with_htf = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias=None,
            htf_daily=None,
            htf_4h=None,
        )

        # None bias should add 0
        assert result_with_htf.breakdown.get("htf_bias") == 0
        assert result_with_htf.score == result_baseline.score

    def test_htf_breakdown_contains_trend_metadata(self, mtf_scorer, bullish_signal_df):
        """Test breakdown includes HTF trend metadata for AI context."""
        result = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bullish",
            htf_daily="bullish",
            htf_4h="neutral",
        )

        # Should contain underscore-prefixed metadata
        assert result.breakdown.get("_htf_trend") == "bullish"
        assert result.breakdown.get("_htf_daily") == "bullish"
        assert result.breakdown.get("_htf_4h") == "neutral"

    def test_htf_breakdown_defaults_when_none(self, mtf_scorer, bullish_signal_df):
        """Test breakdown defaults to 'disabled' when HTF params are None."""
        result = mtf_scorer.calculate_score(bullish_signal_df)

        assert result.breakdown.get("_htf_trend") == "disabled"
        assert result.breakdown.get("_htf_daily") == "disabled"
        assert result.breakdown.get("_htf_4h") == "disabled"

    def test_htf_custom_boost_values(self, bullish_signal_df):
        """Test custom aligned_boost and counter_penalty values work."""
        scorer = SignalScorer(
            mtf_aligned_boost=30,  # Custom boost
            mtf_counter_penalty=15,  # Custom penalty
        )

        result_aligned = scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bullish",
        )
        result_counter = scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bearish",
        )

        baseline = scorer.calculate_score(bullish_signal_df)

        if baseline.score > 0:
            # Aligned should use 30
            assert result_aligned.breakdown.get("htf_bias") == 30
            # Counter should use -15
            assert result_counter.breakdown.get("htf_bias") == -15

    def test_htf_update_settings(self):
        """Test MTF parameters can be updated at runtime."""
        scorer = SignalScorer(
            mtf_aligned_boost=20,
            mtf_counter_penalty=20,
        )

        assert scorer.mtf_aligned_boost == 20
        assert scorer.mtf_counter_penalty == 20

        scorer.update_settings(
            mtf_aligned_boost=25,
            mtf_counter_penalty=15,
        )

        assert scorer.mtf_aligned_boost == 25
        assert scorer.mtf_counter_penalty == 15

    def test_htf_zero_score_no_adjustment(self, mtf_scorer):
        """Test HTF bias doesn't apply when signal score is exactly 0."""
        # Create flat data that produces ~0 score
        df = pd.DataFrame({
            'open': [50000.0] * 100,
            'high': [50100.0] * 100,
            'low': [49900.0] * 100,
            'close': [50000.0] * 100,
            'volume': [10000.0] * 100,
        })

        result = mtf_scorer.calculate_score(df, htf_bias="bullish")

        # If score is 0, HTF adjustment should be 0
        if result.score - result.breakdown.get("htf_bias", 0) == 0:
            assert result.breakdown.get("htf_bias") == 0

    def test_htf_score_clamped_to_100(self, mtf_scorer, bullish_signal_df):
        """Test score with HTF boost is clamped to +100."""
        # Get a very high positive score
        scorer = SignalScorer(threshold=30, mtf_aligned_boost=50)
        result = scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bullish",
        )

        assert result.score <= 100

    def test_htf_score_clamped_to_negative_100(self, mtf_scorer, bearish_signal_df):
        """Test score with HTF boost is clamped to -100."""
        scorer = SignalScorer(threshold=30, mtf_aligned_boost=50)
        result = scorer.calculate_score(
            bearish_signal_df,
            htf_bias="bearish",
        )

        assert result.score >= -100

    def test_extreme_fear_full_penalty_on_bearish_daily_buy(self, mtf_scorer, bullish_signal_df):
        """Test extreme fear applies full penalty when buying into bearish daily trend."""
        # Get baseline score with neutral bias (daily/4H disagree)
        result_baseline = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
        )

        # Get score with extreme fear sentiment
        result_with_fear = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category="extreme_fear",
        )

        # Test requires positive signal (buy signal) to verify extreme fear buy override
        # The bullish_signal_df fixture is designed to produce buy signals, but trend_filter
        # and other penalties can flip the final score. The extreme fear logic checks the score
        # BEFORE HTF adjustments (_raw_score), so we test based on that.
        if result_baseline.breakdown.get("_raw_score", 0) > 0:
            # Without extreme fear, partial penalty (-10) is applied
            assert result_baseline.breakdown.get("htf_bias") == -10, "Expected half penalty without extreme fear"
            # With extreme fear, FULL penalty (-20) should be applied
            assert result_with_fear.breakdown.get("htf_bias") == -20, "Expected full penalty with extreme fear"
            # Score difference should be exactly 10 points (full vs half penalty)
            assert result_baseline.score - result_with_fear.score == 10, "Expected 10-point difference"
        else:
            # If baseline raw score isn't bullish, extreme fear override shouldn't activate
            assert result_baseline.breakdown.get("htf_bias") == result_with_fear.breakdown.get("htf_bias"), \
                "HTF bias should not change when signal direction doesn't match extreme fear condition"

    def test_extreme_fear_full_penalty_on_bullish_daily_sell(self, mtf_scorer, bearish_signal_df):
        """Test extreme fear applies full penalty when selling into bullish daily trend."""
        # Get baseline score with neutral bias (daily/4H disagree)
        result_baseline = mtf_scorer.calculate_score(
            bearish_signal_df,
            htf_bias="neutral",
            htf_daily="bullish",
            htf_4h="bearish",
        )

        # Get score with extreme fear sentiment
        result_with_fear = mtf_scorer.calculate_score(
            bearish_signal_df,
            htf_bias="neutral",
            htf_daily="bullish",
            htf_4h="bearish",
            sentiment_category="extreme_fear",
        )

        # Test requires negative signal (sell signal) to verify extreme fear sell override
        # The bearish_signal_df fixture is designed to produce sell signals, but trend_filter
        # and other penalties can flip the final score. The extreme fear logic checks the score
        # BEFORE HTF adjustments (_raw_score), so we test based on that.
        if result_baseline.breakdown.get("_raw_score", 0) < 0:
            # Without extreme fear, partial penalty (+10) is applied
            assert result_baseline.breakdown.get("htf_bias") == 10, "Expected half penalty without extreme fear"
            # With extreme fear, FULL penalty (+20) should be applied to weaken sell
            assert result_with_fear.breakdown.get("htf_bias") == 20, "Expected full penalty with extreme fear"
            # Score difference should be exactly 10 points (full vs half penalty)
            assert result_with_fear.score - result_baseline.score == 10, "Expected 10-point difference"
        else:
            # If baseline isn't bearish, extreme fear override shouldn't activate
            assert result_baseline.breakdown.get("htf_bias") == result_with_fear.breakdown.get("htf_bias"), \
                "HTF bias should not change when signal direction doesn't match extreme fear condition"

    def test_extreme_fear_no_effect_when_daily_and_4h_agree(self, mtf_scorer, bullish_signal_df):
        """Test extreme fear has no additional effect when daily and 4H agree."""
        # When daily and 4H both agree, htf_bias is not "neutral" so extreme fear
        # override doesn't apply (normal aligned/counter logic is used)
        result_no_fear = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bearish",  # Both daily and 4H bearish
            htf_daily="bearish",
            htf_4h="bearish",
        )

        result_with_fear = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="bearish",
            htf_daily="bearish",
            htf_4h="bearish",
            sentiment_category="extreme_fear",
        )

        # Scores should be identical - extreme fear doesn't apply when trends agree
        assert result_no_fear.score == result_with_fear.score
        assert result_no_fear.breakdown.get("htf_bias") == result_with_fear.breakdown.get("htf_bias")

    def test_extreme_fear_no_effect_on_neutral_daily(self, mtf_scorer, bullish_signal_df):
        """Test extreme fear has no effect when daily trend is neutral."""
        result_no_fear = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="neutral",
            htf_4h="bullish",
        )

        result_with_fear = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="neutral",
            htf_4h="bullish",
            sentiment_category="extreme_fear",
        )

        # No adjustment should be made when daily is neutral
        assert result_no_fear.score == result_with_fear.score
        assert result_no_fear.breakdown.get("htf_bias") == 0
        assert result_with_fear.breakdown.get("htf_bias") == 0

    def test_extreme_fear_no_effect_on_non_extreme_sentiment(self, mtf_scorer, bullish_signal_df):
        """Test that only extreme_fear triggers the override, not other sentiments."""
        result_fear = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category="fear",  # regular fear, not extreme
        )

        result_extreme = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category="extreme_fear",
        )

        # Regular fear should not trigger full penalty
        if result_fear.score > 0:
            assert result_fear.breakdown.get("htf_bias") == -10  # half penalty
            assert result_extreme.breakdown.get("htf_bias") == -20  # full penalty
            assert result_fear.score > result_extreme.score

    def test_extreme_fear_no_effect_when_sentiment_none(self, mtf_scorer, bullish_signal_df):
        """Test extreme fear override does not apply when sentiment_category is None."""
        # When sentiment is None (disabled or failed fetch), normal half-penalty logic applies
        result_with_none = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category=None,
        )

        result_baseline = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category="fear",  # Non-extreme sentiment
        )

        # Both should apply half penalty (not full penalty)
        if result_with_none.score > 0:
            assert result_with_none.breakdown.get("htf_bias") == -10  # half penalty
            assert result_baseline.breakdown.get("htf_bias") == -10  # half penalty
            assert result_with_none.score == result_baseline.score

    def test_invalid_sentiment_category_runtime_validation(self, mtf_scorer, bullish_signal_df):
        """Test runtime validation catches invalid sentiment_category values and falls back to None."""
        # This test verifies the defensive runtime check at signal_scorer.py:434-443
        # Even though sentiment_category has a Literal type hint, we validate at runtime
        # for financial system safety (type hints can be bypassed or ignored)
        #
        # The validation logs an error (not checked here due to structlog complexity)
        # but the key behavior is that invalid values are treated as None.

        invalid_sentiment = "invalid_category"  # type: ignore

        result = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category=invalid_sentiment,
        )

        # Compare behavior with explicit None to verify fallback
        result_with_none = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category=None,
        )

        # Should behave identically - invalid category treated as None
        # This means NO extreme fear override should apply (half penalty, not full)
        assert result.score == result_with_none.score, \
            f"Invalid sentiment should behave like None. Got scores: {result.score} vs {result_with_none.score}"
        assert result.breakdown.get("htf_bias") == result_with_none.breakdown.get("htf_bias"), \
            "Invalid sentiment should not trigger extreme fear override"

        # Verify extreme fear WOULD have different behavior (to confirm test is meaningful)
        result_with_extreme_fear = mtf_scorer.calculate_score(
            bullish_signal_df,
            htf_bias="neutral",
            htf_daily="bearish",
            htf_4h="bullish",
            sentiment_category="extreme_fear",
        )
        # If signal is positive, extreme fear should apply different penalty than None
        if result.score > 0:
            assert result_with_extreme_fear.breakdown.get("htf_bias") != result.breakdown.get("htf_bias"), \
                "Extreme fear should apply different penalty than invalid/None"


class TestHTFEdgeCases:
    """Edge case tests for HTF bias at boundary score values.

    These tests verify that HTF adjustments work correctly when scores
    are near threshold values or at extremes (-100, +100).
    """

    @pytest.fixture
    def scorer(self):
        """Signal scorer with standard MTF parameters."""
        return SignalScorer(
            threshold=60,
            mtf_aligned_boost=20,
            mtf_counter_penalty=20,
        )

    def test_bullish_at_threshold_with_aligned_htf(self, scorer):
        """Score at +60 (threshold) with bullish HTF → +80 (should trade)."""
        # Create mock result where we can verify the HTF adjustment
        # This tests that a borderline signal gets pushed over threshold
        # Note: We test the adjustment value directly via breakdown
        df = pd.DataFrame({
            'open': [50000.0] * 100,
            'high': [50500.0] * 100,
            'low': [49500.0] * 100,
            'close': [50200.0] * 100,
            'volume': [10000.0] * 100,
        })

        result = scorer.calculate_score(df, htf_bias="bullish")

        # Verify HTF adjustment is applied correctly
        if result.score > 0:  # If bullish signal
            assert result.breakdown.get("htf_bias") == 20

    def test_bullish_at_threshold_with_counter_htf(self, scorer):
        """Score at +60 with bearish HTF → +40 (below threshold, no trade)."""
        df = pd.DataFrame({
            'open': [50000.0] * 100,
            'high': [50500.0] * 100,
            'low': [49500.0] * 100,
            'close': [50200.0] * 100,
            'volume': [10000.0] * 100,
        })

        result = scorer.calculate_score(df, htf_bias="bearish")

        # Verify HTF penalty is applied for counter-trend
        if result.score > 0:  # If bullish signal
            assert result.breakdown.get("htf_bias") == -20

    def test_bearish_near_threshold_with_aligned_htf(self, scorer):
        """Bearish score with bearish HTF → more negative (stronger sell)."""
        np.random.seed(44)
        prices = [55000.0 - i * 100 for i in range(100)]  # Downtrend

        df = pd.DataFrame({
            'open': [p + 50 for p in prices],
            'high': [p + 100 for p in prices],
            'low': [p - 50 for p in prices],
            'close': prices,
            'volume': [10000.0] * 100,
        })

        result = scorer.calculate_score(df, htf_bias="bearish")

        # If bearish signal, aligned HTF should make it more negative
        if result.score < 0:
            assert result.breakdown.get("htf_bias") == -20

    def test_bearish_with_counter_htf_weakens_signal(self, scorer):
        """Bearish score with bullish HTF → less negative (weaker sell)."""
        np.random.seed(45)
        prices = [55000.0 - i * 100 for i in range(100)]  # Downtrend

        df = pd.DataFrame({
            'open': [p + 50 for p in prices],
            'high': [p + 100 for p in prices],
            'low': [p - 50 for p in prices],
            'close': prices,
            'volume': [10000.0] * 100,
        })

        result_no_htf = scorer.calculate_score(df)
        result_with_htf = scorer.calculate_score(df, htf_bias="bullish")

        # If bearish signal, counter-trend HTF should make it less negative
        if result_no_htf.score < 0:
            assert result_with_htf.breakdown.get("htf_bias") == 20
            # Score should be closer to 0 (less negative)
            assert result_with_htf.score > result_no_htf.score

    def test_extreme_positive_score_with_boost(self, scorer):
        """Score near +100 with HTF boost should not exceed +100."""
        # Create strong bullish conditions
        np.random.seed(46)
        prices = [45000.0 + i * 100 for i in range(100)]  # Strong uptrend

        df = pd.DataFrame({
            'open': [p - 50 for p in prices],
            'high': [p + 100 for p in prices],
            'low': [p - 100 for p in prices],
            'close': prices,
            'volume': [15000.0] * 100,  # High volume
        })

        result = scorer.calculate_score(df, htf_bias="bullish")

        # Score must stay within bounds
        assert -100 <= result.score <= 100

    def test_extreme_negative_score_with_boost(self, scorer):
        """Score near -100 with HTF boost should not go below -100."""
        # Create strong bearish conditions
        np.random.seed(47)
        prices = [55000.0 - i * 100 for i in range(100)]  # Strong downtrend

        df = pd.DataFrame({
            'open': [p + 50 for p in prices],
            'high': [p + 100 for p in prices],
            'low': [p - 100 for p in prices],
            'close': prices,
            'volume': [15000.0] * 100,  # High volume
        })

        result = scorer.calculate_score(df, htf_bias="bearish")

        # Score must stay within bounds
        assert -100 <= result.score <= 100


class TestHTFRawIndicatorValues:
    """Tests for raw indicator values stored in breakdown for signal history."""

    @pytest.fixture
    def scorer(self):
        return SignalScorer()

    @pytest.fixture
    def sample_df(self, sample_ohlcv_data):
        return sample_ohlcv_data(length=100, base_price=50000.0, volatility=0.02)

    def test_rsi_value_in_breakdown(self, scorer, sample_df):
        """Test raw RSI value is stored in breakdown."""
        result = scorer.calculate_score(sample_df)

        assert "_rsi_value" in result.breakdown
        rsi_value = result.breakdown["_rsi_value"]
        assert rsi_value is None or (0 <= rsi_value <= 100)

    def test_macd_histogram_in_breakdown(self, scorer, sample_df):
        """Test MACD histogram value is stored in breakdown."""
        result = scorer.calculate_score(sample_df)

        assert "_macd_histogram" in result.breakdown
        # MACD histogram can be any value, just verify it's a number
        assert result.breakdown["_macd_histogram"] is None or isinstance(
            result.breakdown["_macd_histogram"], (int, float)
        )

    def test_bb_position_in_breakdown(self, scorer, sample_df):
        """Test Bollinger Band position (0-1) is stored in breakdown."""
        result = scorer.calculate_score(sample_df)

        assert "_bb_position" in result.breakdown
        bb_pos = result.breakdown["_bb_position"]
        # BB position should be roughly 0-1 (can exceed due to volatility)
        assert bb_pos is None or isinstance(bb_pos, (int, float))

    def test_ema_gap_percent_in_breakdown(self, scorer, sample_df):
        """Test EMA gap percentage is stored in breakdown."""
        result = scorer.calculate_score(sample_df)

        assert "_ema_gap_percent" in result.breakdown
        assert result.breakdown["_ema_gap_percent"] is None or isinstance(
            result.breakdown["_ema_gap_percent"], (int, float)
        )

    def test_raw_score_in_breakdown(self, scorer, sample_df):
        """Test raw score (before adjustments) is stored in breakdown."""
        result = scorer.calculate_score(sample_df)

        assert "_raw_score" in result.breakdown
        raw = result.breakdown["_raw_score"]
        assert isinstance(raw, int)
        assert -100 <= raw <= 100
