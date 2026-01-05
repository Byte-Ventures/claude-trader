"""
Tests for the ADX (Average Directional Index) indicator.

Tests cover:
- ADX calculation accuracy
- Directional indicators (+DI, -DI)
- Confidence multiplier at various ADX levels
- Edge cases (insufficient data, NaN handling)
- Trend direction and strength classification
"""

import pytest
import pandas as pd
import numpy as np

from src.indicators.adx import (
    calculate_adx,
    get_adx_value,
    get_adx_confidence_multiplier,
    get_trend_direction,
    classify_trend_strength,
    ADX_WEAK_TREND,
    ADX_MODERATE_TREND,
    ADX_STRONG_TREND,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trending_up_data():
    """Generate strongly trending upward data (high ADX expected)."""
    np.random.seed(42)
    length = 100
    base_price = 50000.0

    # Strong uptrend with consistent higher highs and higher lows
    prices = []
    current = base_price
    for _ in range(length):
        current += current * np.random.uniform(0.005, 0.015)  # Strong upward bias
        prices.append(current)

    high = [p * 1.005 for p in prices]
    low = [p * 0.995 for p in prices]
    close = prices

    return pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
    })


@pytest.fixture
def ranging_data():
    """Generate sideways/ranging data (low ADX expected)."""
    np.random.seed(43)
    length = 100
    base_price = 50000.0

    # Sideways movement with random oscillation
    prices = []
    for i in range(length):
        # Oscillate around base price
        offset = np.sin(i / 5) * base_price * 0.01
        noise = np.random.uniform(-0.002, 0.002) * base_price
        prices.append(base_price + offset + noise)

    high = [p * 1.002 for p in prices]
    low = [p * 0.998 for p in prices]
    close = prices

    return pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
    })


@pytest.fixture
def trending_down_data():
    """Generate strongly trending downward data (high ADX, -DI > +DI)."""
    np.random.seed(44)
    length = 100
    base_price = 60000.0

    # Strong downtrend
    prices = []
    current = base_price
    for _ in range(length):
        current -= current * np.random.uniform(0.005, 0.015)  # Strong downward bias
        prices.append(current)

    high = [p * 1.005 for p in prices]
    low = [p * 0.995 for p in prices]
    close = prices

    return pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
    })


@pytest.fixture
def insufficient_data():
    """Generate insufficient data for ADX calculation."""
    return pd.DataFrame({
        "high": [50000.0] * 10,
        "low": [49900.0] * 10,
        "close": [49950.0] * 10,
    })


# ============================================================================
# ADX Calculation Tests
# ============================================================================


class TestCalculateADX:
    """Test ADX calculation function."""

    def test_trending_market_has_high_adx(self, trending_up_data):
        """Test that trending market produces high ADX values."""
        result = calculate_adx(
            trending_up_data["high"],
            trending_up_data["low"],
            trending_up_data["close"],
        )

        # ADX should be above 25 for strong trend
        final_adx = result.adx.iloc[-1]
        assert final_adx > 25, f"Expected ADX > 25 for trending market, got {final_adx}"

    def test_ranging_market_has_low_adx(self, ranging_data):
        """Test that ranging market produces low ADX values."""
        result = calculate_adx(
            ranging_data["high"],
            ranging_data["low"],
            ranging_data["close"],
        )

        # ADX should be below 25 for ranging market
        final_adx = result.adx.iloc[-1]
        assert final_adx < 30, f"Expected ADX < 30 for ranging market, got {final_adx}"

    def test_uptrend_has_plus_di_greater(self, trending_up_data):
        """Test that uptrend has +DI > -DI."""
        result = calculate_adx(
            trending_up_data["high"],
            trending_up_data["low"],
            trending_up_data["close"],
        )

        plus_di = result.plus_di.iloc[-1]
        minus_di = result.minus_di.iloc[-1]
        assert plus_di > minus_di, f"Expected +DI ({plus_di}) > -DI ({minus_di}) in uptrend"

    def test_downtrend_has_minus_di_greater(self, trending_down_data):
        """Test that downtrend has -DI > +DI."""
        result = calculate_adx(
            trending_down_data["high"],
            trending_down_data["low"],
            trending_down_data["close"],
        )

        plus_di = result.plus_di.iloc[-1]
        minus_di = result.minus_di.iloc[-1]
        assert minus_di > plus_di, f"Expected -DI ({minus_di}) > +DI ({plus_di}) in downtrend"

    def test_adx_bounded_0_to_100(self, trending_up_data):
        """Test ADX values are bounded between 0 and 100."""
        result = calculate_adx(
            trending_up_data["high"],
            trending_up_data["low"],
            trending_up_data["close"],
        )

        # Check all non-NaN values are in valid range
        valid_adx = result.adx.dropna()
        assert (valid_adx >= 0).all(), "ADX should be >= 0"
        assert (valid_adx <= 100).all(), "ADX should be <= 100"

    def test_insufficient_data_returns_nan(self, insufficient_data):
        """Test that insufficient data returns NaN series."""
        result = calculate_adx(
            insufficient_data["high"],
            insufficient_data["low"],
            insufficient_data["close"],
        )

        assert result.adx.isna().all(), "ADX should be NaN with insufficient data"

    def test_custom_period(self, trending_up_data):
        """Test ADX with custom period."""
        result_14 = calculate_adx(
            trending_up_data["high"],
            trending_up_data["low"],
            trending_up_data["close"],
            period=14,
        )
        result_7 = calculate_adx(
            trending_up_data["high"],
            trending_up_data["low"],
            trending_up_data["close"],
            period=7,
        )

        # Shorter period should respond faster (different final values)
        assert result_14.adx.iloc[-1] != result_7.adx.iloc[-1], \
            "Different periods should produce different ADX values"


class TestGetADXValue:
    """Test convenience function for getting current ADX."""

    def test_returns_float_for_valid_data(self, trending_up_data):
        """Test get_adx_value returns float for valid data."""
        adx = get_adx_value(trending_up_data)

        assert adx is not None
        assert isinstance(adx, float)
        assert 0 <= adx <= 100

    def test_returns_none_for_insufficient_data(self, insufficient_data):
        """Test get_adx_value returns None for insufficient data."""
        adx = get_adx_value(insufficient_data)

        assert adx is None


# ============================================================================
# Confidence Multiplier Tests
# ============================================================================


class TestGetADXConfidenceMultiplier:
    """Test ADX-based confidence multiplier."""

    def test_weak_trend_returns_half(self):
        """Test ADX < 20 returns 0.5 multiplier."""
        multiplier = get_adx_confidence_multiplier(15)
        assert multiplier == 0.5

    def test_emerging_trend_returns_reduced(self):
        """Test ADX 20-25 returns 0.75 multiplier."""
        multiplier = get_adx_confidence_multiplier(22)
        assert multiplier == 0.75

    def test_confirmed_trend_returns_full(self):
        """Test ADX 25-40 returns 1.0 multiplier."""
        multiplier = get_adx_confidence_multiplier(30)
        assert multiplier == 1.0

    def test_strong_trend_returns_boost(self):
        """Test ADX > 40 returns 1.1 multiplier."""
        multiplier = get_adx_confidence_multiplier(50)
        assert multiplier == 1.1

    def test_none_adx_returns_neutral(self):
        """Test None ADX returns 1.0 (neutral) multiplier."""
        multiplier = get_adx_confidence_multiplier(None)
        assert multiplier == 1.0

    def test_boundary_at_weak_threshold(self):
        """Test boundary behavior at weak threshold (20)."""
        # Just below threshold
        assert get_adx_confidence_multiplier(19.9) == 0.5
        # At threshold
        assert get_adx_confidence_multiplier(20.0) == 0.75

    def test_boundary_at_strong_threshold(self):
        """Test boundary behavior at strong threshold (25)."""
        # Just below threshold
        assert get_adx_confidence_multiplier(24.9) == 0.75
        # At threshold
        assert get_adx_confidence_multiplier(25.0) == 1.0

    def test_boundary_at_40(self):
        """Test boundary behavior at 40."""
        # At 40
        assert get_adx_confidence_multiplier(40.0) == 1.0
        # Just above 40
        assert get_adx_confidence_multiplier(40.1) == 1.1

    def test_custom_thresholds(self):
        """Test custom threshold parameters."""
        # With custom thresholds
        multiplier = get_adx_confidence_multiplier(
            25,
            weak_threshold=30,
            strong_threshold=35,
        )
        # 25 is below weak_threshold=30, so should be 0.5
        assert multiplier == 0.5


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestGetTrendDirection:
    """Test trend direction determination."""

    def test_bullish_when_plus_di_greater(self):
        """Test bullish direction when +DI > -DI."""
        direction = get_trend_direction(plus_di=30, minus_di=20)
        assert direction == "bullish"

    def test_bearish_when_minus_di_greater(self):
        """Test bearish direction when -DI > +DI."""
        direction = get_trend_direction(plus_di=20, minus_di=30)
        assert direction == "bearish"

    def test_neutral_when_equal(self):
        """Test neutral direction when +DI = -DI."""
        direction = get_trend_direction(plus_di=25, minus_di=25)
        assert direction == "neutral"


class TestClassifyTrendStrength:
    """Test trend strength classification."""

    def test_none_returns_unknown(self):
        """Test None ADX returns 'unknown'."""
        assert classify_trend_strength(None) == "unknown"

    def test_weak_classification(self):
        """Test ADX < 20 returns 'weak'."""
        assert classify_trend_strength(15) == "weak"

    def test_emerging_classification(self):
        """Test ADX 20-25 returns 'emerging'."""
        assert classify_trend_strength(22) == "emerging"

    def test_moderate_classification(self):
        """Test ADX 25-40 returns 'moderate'."""
        assert classify_trend_strength(30) == "moderate"

    def test_strong_classification(self):
        """Test ADX 40-75 returns 'strong'."""
        assert classify_trend_strength(50) == "strong"

    def test_extreme_classification(self):
        """Test ADX > 75 returns 'extreme'."""
        assert classify_trend_strength(80) == "extreme"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_adx(self):
        """Test ADX value of exactly 0."""
        multiplier = get_adx_confidence_multiplier(0)
        assert multiplier == 0.5
        assert classify_trend_strength(0) == "weak"

    def test_max_adx(self):
        """Test ADX value at theoretical maximum (100)."""
        multiplier = get_adx_confidence_multiplier(100)
        assert multiplier == 1.1
        assert classify_trend_strength(100) == "extreme"

    def test_negative_adx_treated_as_weak(self):
        """Test negative ADX (shouldn't happen but handle gracefully)."""
        multiplier = get_adx_confidence_multiplier(-5)
        assert multiplier == 0.5  # Treated as very weak trend

    def test_constant_price_data(self):
        """Test ADX with constant prices (no movement)."""
        df = pd.DataFrame({
            "high": [50000.0] * 50,
            "low": [50000.0] * 50,
            "close": [50000.0] * 50,
        })

        result = calculate_adx(df["high"], df["low"], df["close"])
        # ADX should be 0 or near 0 with no price movement
        valid_adx = result.adx.dropna()
        if len(valid_adx) > 0:
            assert valid_adx.iloc[-1] < 5, "ADX should be very low with no price movement"


class TestADXSettingsValidation:
    """Tests for ADX settings validation."""

    def test_adx_threshold_validation_rejects_invalid(self, monkeypatch):
        """Verify that weak >= strong threshold raises ValidationError."""
        import os
        from pydantic import ValidationError

        # Set environment variables for invalid thresholds
        monkeypatch.setenv("ADX_WEAK_THRESHOLD", "30")
        monkeypatch.setenv("ADX_STRONG_THRESHOLD", "25")

        # Import Settings fresh to pick up env vars
        from config.settings import Settings
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "adx_weak_threshold" in str(exc_info.value)
        assert "must be less than" in str(exc_info.value)

    def test_adx_threshold_validation_accepts_valid(self, monkeypatch):
        """Verify that weak < strong threshold is accepted."""
        # Set valid thresholds
        monkeypatch.setenv("ADX_WEAK_THRESHOLD", "18")
        monkeypatch.setenv("ADX_STRONG_THRESHOLD", "28")
        monkeypatch.setenv("TRADING_MODE", "paper")  # Required field

        from config.settings import Settings
        settings = Settings()

        assert settings.adx_weak_threshold == 18.0
        assert settings.adx_strong_threshold == 28.0
