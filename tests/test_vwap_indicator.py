"""Tests for VWAP indicator signal generation."""

import pytest
from decimal import Decimal

from src.indicators.vwap import (
    get_vwap_signal_graduated,
    calculate_price_vs_vwap_percent,
)


class TestGetVWAPSignalGraduated:
    """Tests for get_vwap_signal_graduated function."""

    def test_price_above_vwap_full_signal(self):
        """Price 1% above VWAP should return -1.0 (full bearish)."""
        # Price is 101, VWAP is 100 -> 1% above
        signal = get_vwap_signal_graduated(101.0, 100.0, threshold_percent=0.5)
        assert signal == -1.0

    def test_price_below_vwap_full_signal(self):
        """Price 1% below VWAP should return +1.0 (full bullish)."""
        # Price is 99, VWAP is 100 -> 1% below
        signal = get_vwap_signal_graduated(99.0, 100.0, threshold_percent=0.5)
        assert signal == 1.0

    def test_price_slightly_above_vwap(self):
        """Price 0.5% above VWAP should return moderate bearish signal."""
        # Price is 100.5, VWAP is 100 -> 0.5% above
        signal = get_vwap_signal_graduated(100.5, 100.0, threshold_percent=0.5)
        assert -1.0 < signal < 0.0  # Should be negative (bearish)
        assert signal == pytest.approx(-0.5, abs=0.1)

    def test_price_slightly_below_vwap(self):
        """Price 0.5% below VWAP should return moderate bullish signal."""
        # Price is 99.5, VWAP is 100 -> 0.5% below
        signal = get_vwap_signal_graduated(99.5, 100.0, threshold_percent=0.5)
        assert 0.0 < signal < 1.0  # Should be positive (bullish)
        assert signal == pytest.approx(0.5, abs=0.1)

    def test_price_within_dead_zone(self):
        """Price within threshold should return 0.0 (dead zone)."""
        # Price is 100.2, VWAP is 100 -> 0.2% above (below 0.5% threshold)
        signal = get_vwap_signal_graduated(100.2, 100.0, threshold_percent=0.5)
        assert signal == 0.0

        # Price is 99.8, VWAP is 100 -> 0.2% below (below 0.5% threshold)
        signal = get_vwap_signal_graduated(99.8, 100.0, threshold_percent=0.5)
        assert signal == 0.0

    def test_price_exactly_at_vwap(self):
        """Price exactly at VWAP should return 0.0."""
        signal = get_vwap_signal_graduated(100.0, 100.0, threshold_percent=0.5)
        assert signal == 0.0

    def test_vwap_none_returns_zero(self):
        """Missing VWAP should return 0.0."""
        signal = get_vwap_signal_graduated(100.0, None, threshold_percent=0.5)
        assert signal == 0.0

    def test_price_zero_returns_zero(self):
        """Zero price should return 0.0."""
        signal = get_vwap_signal_graduated(0.0, 100.0, threshold_percent=0.5)
        assert signal == 0.0

    def test_vwap_zero_returns_zero(self):
        """Zero VWAP should return 0.0."""
        signal = get_vwap_signal_graduated(100.0, 0.0, threshold_percent=0.5)
        assert signal == 0.0

    def test_negative_price_returns_zero(self):
        """Negative price should return 0.0."""
        signal = get_vwap_signal_graduated(-100.0, 100.0, threshold_percent=0.5)
        assert signal == 0.0

    def test_decimal_inputs(self):
        """Should handle Decimal inputs correctly."""
        signal = get_vwap_signal_graduated(
            Decimal("101.0"), Decimal("100.0"), threshold_percent=0.5
        )
        assert signal == -1.0

    def test_custom_threshold(self):
        """Custom threshold should affect dead zone size."""
        # 0.3% deviation with 0.5% threshold -> dead zone (return 0)
        signal = get_vwap_signal_graduated(100.3, 100.0, threshold_percent=0.5)
        assert signal == 0.0

        # 0.3% deviation with 0.2% threshold -> outside dead zone
        signal = get_vwap_signal_graduated(100.3, 100.0, threshold_percent=0.2)
        assert signal != 0.0

    def test_invalid_threshold_uses_default(self):
        """Invalid threshold should fall back to default."""
        # Zero threshold should use default
        signal = get_vwap_signal_graduated(101.0, 100.0, threshold_percent=0.0)
        assert signal == -1.0  # Should still work with default

    def test_signal_clamped_at_extremes(self):
        """Signal should be clamped between -1.0 and +1.0."""
        # Very large deviation (5% above)
        signal = get_vwap_signal_graduated(105.0, 100.0, threshold_percent=0.5)
        assert signal == -1.0

        # Very large deviation (5% below)
        signal = get_vwap_signal_graduated(95.0, 100.0, threshold_percent=0.5)
        assert signal == 1.0


class TestCalculatePriceVsVWAPPercent:
    """Tests for calculate_price_vs_vwap_percent function."""

    def test_price_above_vwap(self):
        """Price above VWAP should return positive percentage."""
        result = calculate_price_vs_vwap_percent(101.0, 100.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_price_below_vwap(self):
        """Price below VWAP should return negative percentage."""
        result = calculate_price_vs_vwap_percent(99.0, 100.0)
        assert result == pytest.approx(-1.0, abs=0.01)

    def test_price_at_vwap(self):
        """Price at VWAP should return 0.0."""
        result = calculate_price_vs_vwap_percent(100.0, 100.0)
        assert result == 0.0

    def test_vwap_none_returns_none(self):
        """Missing VWAP should return None."""
        result = calculate_price_vs_vwap_percent(100.0, None)
        assert result is None

    def test_vwap_zero_returns_none(self):
        """Zero VWAP should return None (avoid division by zero)."""
        result = calculate_price_vs_vwap_percent(100.0, 0.0)
        assert result is None

    def test_decimal_inputs(self):
        """Should handle Decimal inputs correctly."""
        result = calculate_price_vs_vwap_percent(Decimal("101.0"), Decimal("100.0"))
        assert result == pytest.approx(1.0, abs=0.01)
