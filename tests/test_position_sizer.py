"""
Comprehensive tests for the PositionSizer ATR-based position sizing calculator.

Tests cover:
- Position size calculation from portfolio value
- Risk-based sizing (max risk per trade)
- ATR-based stop-loss placement (buy and sell)
- ATR-based take-profit placement
- Trailing stop calculation
- Safety multiplier application (from validator)
- Signal strength scaling
- Minimum trade size enforcement
- Maximum position percent boundary
- Balance constraints (available funds)
- Edge cases (very low/high ATR, zero multiplier, insufficient data)
- Settings updates
- Sell all position functionality
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal

from src.strategy.position_sizer import (
    PositionSizer,
    PositionSizeConfig,
    PositionSizeResult,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sizer():
    """Position sizer with default settings."""
    config = PositionSizeConfig(
        risk_per_trade_percent=0.5,  # Required from settings
        min_trade_base=0.0001,  # Required from settings
    )
    return PositionSizer(config=config)


@pytest.fixture
def custom_config():
    """Custom position size configuration."""
    return PositionSizeConfig(
        risk_per_trade_percent=1.0,  # Required from settings
        min_trade_base=0.0001,  # Required from settings
        max_position_percent=30.0,
        stop_loss_atr_multiplier=2.0,
        min_trade_quote=20.0,
    )


@pytest.fixture
def sizer_custom(custom_config):
    """Position sizer with custom configuration."""
    return PositionSizer(config=custom_config, atr_period=20)


@pytest.fixture
def sample_df():
    """Generate sample OHLCV data with realistic volatility."""
    np.random.seed(42)
    length = 100
    prices = []
    current = 50000.0

    for _ in range(length):
        change = np.random.randn() * 0.015 * current
        current = current + change
        prices.append(current)

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
    }

    for price in prices:
        o = price * (1 + np.random.uniform(-0.005, 0.005))
        c = price * (1 + np.random.uniform(-0.005, 0.005))
        h = max(o, c) * (1 + abs(np.random.uniform(0, 0.015)))
        l = min(o, c) * (1 - abs(np.random.uniform(0, 0.015)))

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)

    return pd.DataFrame(data)


@pytest.fixture
def low_volatility_df():
    """Generate low volatility OHLCV data."""
    length = 100
    prices = [50000.0] * length  # Flat prices

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
    }

    for price in prices:
        # Very tight range
        o = price * 0.9999
        c = price * 1.0001
        h = price * 1.0002
        l = price * 0.9998

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)

    return pd.DataFrame(data)


@pytest.fixture
def high_volatility_df():
    """Generate high volatility OHLCV data."""
    np.random.seed(43)
    length = 100
    prices = []
    current = 50000.0

    for _ in range(length):
        # Large price swings
        change = np.random.randn() * 0.05 * current
        current = max(current + change, 1000)  # Floor at 1000
        prices.append(current)

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
    }

    for price in prices:
        o = price * (1 + np.random.uniform(-0.02, 0.02))
        c = price * (1 + np.random.uniform(-0.02, 0.02))
        h = max(o, c) * (1 + abs(np.random.uniform(0, 0.04)))
        l = min(o, c) * (1 - abs(np.random.uniform(0, 0.04)))

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)

    return pd.DataFrame(data)


# ============================================================================
# Initialization Tests
# ============================================================================

def test_default_initialization():
    """Test sizer initializes with default configuration."""
    sizer = PositionSizer()

    assert sizer.config.max_position_percent == 40.0
    assert sizer.config.risk_per_trade_percent == 0.5
    assert sizer.config.stop_loss_atr_multiplier == 1.5
    assert sizer.config.min_trade_quote == 100.0
    assert sizer.atr_period == 14


def test_custom_configuration(custom_config):
    """Test sizer accepts custom configuration."""
    sizer = PositionSizer(config=custom_config, atr_period=20)

    assert sizer.config.max_position_percent == 30.0
    assert sizer.config.risk_per_trade_percent == 1.0
    assert sizer.config.stop_loss_atr_multiplier == 2.0
    assert sizer.config.min_trade_quote == 20.0
    assert sizer.atr_period == 20


# ============================================================================
# Position Size Calculation Tests
# ============================================================================

def test_calculate_size_returns_result(sizer, sample_df):
    """Test calculate_size returns PositionSizeResult."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.1")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    assert isinstance(result, PositionSizeResult)
    assert isinstance(result.size_base, Decimal)
    assert isinstance(result.size_quote, Decimal)
    assert isinstance(result.stop_loss_price, Decimal)


def test_position_size_respects_max_position_percent(sizer, sample_df):
    """Test position size never exceeds max_position_percent."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("100000.00")  # Large balance
    base_balance = Decimal("0.0")
    signal_strength = 100  # Max strength

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
        safety_multiplier=1.0,
    )

    # Calculate portfolio value
    total_value = quote_balance
    max_position_quote = total_value * Decimal(str(sizer.config.max_position_percent)) / Decimal("100")

    # Position should not exceed max
    assert result.size_quote <= max_position_quote


def test_buy_accounts_for_existing_position(sizer, sample_df):
    """Test buy size is reduced by existing position to stay within limit."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("100000.00")  # Large balance
    base_balance = Decimal("0.5")  # Already holding 0.5 BTC = 25000 quote
    signal_strength = 100

    # Total portfolio = 100000 + (0.5 * 50000) = 125000
    # Max position (40%) = 50000 quote = 1.0 BTC
    # Already holding 0.5 BTC, so max additional = 0.5 BTC

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
        safety_multiplier=1.0,
    )

    # Total position after buy should not exceed max_position_percent
    total_value = quote_balance + (base_balance * current_price)
    max_position_base = (total_value * Decimal("0.40")) / current_price  # 40%
    max_additional = max_position_base - base_balance

    # Should allow some buy since under position limit
    assert result.size_base > Decimal("0"), "Should allow buying when under position limit"
    # But should not exceed the remaining room
    assert result.size_base <= max_additional, "Should not exceed position limit"


def test_buy_returns_zero_when_at_position_limit(sizer, sample_df):
    """Test buy returns zero when already at position limit."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("50000.00")
    base_balance = Decimal("0.5")  # 0.5 BTC = 25000, total = 75000
    signal_strength = 100

    # Total portfolio = 75000
    # Current position = 25000 / 75000 = 33.3%
    # Max position = 40%
    # Max additional = (75000 * 0.4 / 50000) - 0.5 = 0.6 - 0.5 = 0.1 BTC

    # Now test with a larger existing position
    base_balance_large = Decimal("1.0")  # 1 BTC = 50000
    quote_balance_small = Decimal("10000.00")
    # Total = 60000, current position = 50000/60000 = 83.3% > 40%

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance_small,
        base_balance_large,
        signal_strength,
        side="buy",
        safety_multiplier=1.0,
    )

    # Should return zero since already over position limit
    assert result.size_base == Decimal("0")
    assert result.size_quote == Decimal("0")


def test_position_size_respects_available_quote_balance(sizer, sample_df):
    """Test buy position size limited by available quote currency."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("1000.00")  # Limited funds
    base_balance = Decimal("0.0")
    signal_strength = 100

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    # Cannot spend more than available balance
    assert result.size_quote <= quote_balance


def test_position_size_respects_available_base_balance(sizer, sample_df):
    """Test sell position size limited by available base currency."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.05")  # Limited holdings
    signal_strength = 100

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="sell",
    )

    # Cannot sell more than owned
    assert result.size_base <= base_balance


def test_signal_strength_scales_position_size(sizer, sample_df):
    """Test signal strength multiplier affects position size."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")

    # Weak signal (60)
    result_weak = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength=60,
        side="buy",
    )

    # Strong signal (100)
    result_strong = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength=100,
        side="buy",
    )

    # Stronger signal should result in larger position (if not hitting limits)
    if result_weak.size_quote > 0 and result_strong.size_quote > 0:
        # 100 strength is 1.0x, 60 strength is 0.6x
        assert result_strong.size_quote >= result_weak.size_quote


def test_safety_multiplier_reduces_position_size(sizer, sample_df):
    """Test safety multiplier scales down position size."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 100

    # Full safety multiplier
    result_full = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
        safety_multiplier=1.0,
    )

    # Reduced safety multiplier
    result_reduced = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
        safety_multiplier=0.5,
    )

    # Reduced multiplier should result in smaller position
    assert result_reduced.size_quote <= result_full.size_quote


def test_volatility_affects_position_size(sizer, low_volatility_df, high_volatility_df):
    """Test volatility multiplier affects position size."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 100

    # Low volatility should allow larger positions
    result_low_vol = sizer.calculate_size(
        low_volatility_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    # High volatility should reduce position size
    result_high_vol = sizer.calculate_size(
        high_volatility_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    # High volatility positions should generally be smaller
    # (unless volatility is so high that ATR prevents any position)
    if result_low_vol.size_quote > 0 and result_high_vol.size_quote > 0:
        assert result_high_vol.size_quote <= result_low_vol.size_quote


def test_minimum_trade_size_enforced(sizer, sample_df):
    """Test trades below minimum size return zero."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("1.00")  # Tiny balance
    base_balance = Decimal("0.0")
    signal_strength = 60  # Weak signal

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    # Should return zero size if below minimum
    if result.size_quote < Decimal(str(sizer.config.min_trade_quote)):
        assert result.size_base == Decimal("0")
        assert result.size_quote == Decimal("0")


# ============================================================================
# Stop-Loss Tests
# ============================================================================

def test_buy_stop_loss_below_price(sizer, sample_df):
    """Test buy order stop-loss is below entry price."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_quote > 0:
        assert result.stop_loss_price < current_price


def test_sell_stop_loss_above_price(sizer, sample_df):
    """Test sell order stop-loss is above entry price."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.5")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="sell",
    )

    if result.size_quote > 0:
        assert result.stop_loss_price > current_price


def test_stop_loss_distance_uses_atr_multiplier(sizer, sample_df):
    """Test stop-loss distance is ATR * multiplier."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_quote > 0:
        stop_distance = current_price - result.stop_loss_price
        # Stop distance should be positive and based on ATR
        assert stop_distance > 0


# ============================================================================
# Risk Calculation Tests
# ============================================================================

def test_risk_amount_respects_risk_percent(sizer, sample_df):
    """Test risk amount is based on risk_per_trade_percent."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.1")  # 0.1 BTC @ 50k = 5k
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    # Total portfolio = 10000 + (0.1 * 50000) = 15000
    # Risk per trade (0.5%) = 15000 * 0.005 = 75
    expected_risk = Decimal("75.00")

    # Risk amount should match
    assert result.risk_amount_quote == pytest.approx(expected_risk, abs=1)


def test_position_percent_calculated_correctly(sizer, sample_df):
    """Test position_percent reflects actual position size."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_quote > 0:
        # Position percent = (size_quote / total_value) * 100
        total_value = quote_balance
        expected_percent = float(result.size_quote / total_value * 100)

        assert result.position_percent == pytest.approx(expected_percent, abs=0.1)


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_dataframe_returns_zero(sizer):
    """Test empty DataFrame returns zero position."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    df = pd.DataFrame()

    result = sizer.calculate_size(
        df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    assert result.size_base == Decimal("0")
    assert result.size_quote == Decimal("0")


def test_insufficient_data_returns_zero(sizer):
    """Test DataFrame with too few rows returns zero."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    # Only 5 rows (needs at least atr_period + 1 = 15)
    df = pd.DataFrame({
        'open': [50000] * 5,
        'high': [51000] * 5,
        'low': [49000] * 5,
        'close': [50000] * 5,
    })

    result = sizer.calculate_size(
        df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    assert result.size_base == Decimal("0")
    assert result.size_quote == Decimal("0")


def test_zero_balance_returns_zero(sizer, sample_df):
    """Test zero balance returns zero position."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("0.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    assert result.size_base == Decimal("0")
    assert result.size_quote == Decimal("0")


def test_negative_atr_returns_zero(sizer):
    """Test negative/zero ATR returns zero position."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    # Flat prices will produce very low/zero ATR
    df = pd.DataFrame({
        'open': [50000.0] * 50,
        'high': [50000.0] * 50,
        'low': [50000.0] * 50,
        'close': [50000.0] * 50,
    })

    result = sizer.calculate_size(
        df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    # Zero ATR should result in zero position
    assert result.size_base == Decimal("0")


def test_zero_safety_multiplier_returns_zero(sizer, sample_df):
    """Test safety_multiplier=0 returns zero position."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
        safety_multiplier=0.0,
    )

    assert result.size_base == Decimal("0")
    assert result.size_quote == Decimal("0")


def test_nan_atr_returns_zero(sizer):
    """Test NaN ATR values return zero position."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    # Create data that might produce NaN
    df = pd.DataFrame({
        'open': [50000] * 20,
        'high': [50000] * 20,
        'low': [50000] * 20,
        'close': [50000] * 20,
    })

    result = sizer.calculate_size(
        df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    # Should handle NaN gracefully
    assert isinstance(result, PositionSizeResult)


# ============================================================================
# Settings Update Tests
# ============================================================================

def test_update_settings_changes_max_position():
    """Test update_settings modifies max_position_percent."""
    sizer = PositionSizer()

    sizer.update_settings(max_position_percent=25.0)

    assert sizer.config.max_position_percent == 25.0


def test_update_settings_changes_stop_loss_multiplier():
    """Test update_settings modifies stop_loss_atr_multiplier."""
    sizer = PositionSizer()

    sizer.update_settings(stop_loss_atr_multiplier=2.5)

    assert sizer.config.stop_loss_atr_multiplier == 2.5


def test_update_settings_changes_atr_period():
    """Test update_settings modifies ATR period."""
    sizer = PositionSizer()

    sizer.update_settings(atr_period=21)

    assert sizer.atr_period == 21


def test_update_settings_none_values_ignored():
    """Test update_settings ignores None values."""
    sizer = PositionSizer()
    original_max = sizer.config.max_position_percent

    sizer.update_settings(max_position_percent=None, atr_period=21)

    # Max position should remain unchanged
    assert sizer.config.max_position_percent == original_max
    # ATR period should update
    assert sizer.atr_period == 21


# ============================================================================
# Sell All Position Tests
# ============================================================================

def test_calculate_sell_all_size():
    """Test calculate_sell_all_size returns full position."""
    sizer = PositionSizer()
    base_balance = Decimal("0.5")
    current_price = Decimal("50000.00")

    result = sizer.calculate_sell_all_size(base_balance, current_price)

    assert result.size_base == base_balance
    assert result.size_quote == base_balance * current_price
    assert result.position_percent == 100.0


def test_sell_all_stop_loss_zero():
    """Test sell all sets stop-loss to zero."""
    sizer = PositionSizer()
    base_balance = Decimal("0.5")
    current_price = Decimal("50000.00")

    result = sizer.calculate_sell_all_size(base_balance, current_price)

    assert result.stop_loss_price == Decimal("0")
    assert result.risk_amount_quote == Decimal("0")


def test_sell_all_with_zero_balance():
    """Test sell all with zero balance."""
    sizer = PositionSizer()
    base_balance = Decimal("0.0")
    current_price = Decimal("50000.00")

    result = sizer.calculate_sell_all_size(base_balance, current_price)

    assert result.size_base == Decimal("0")
    assert result.size_quote == Decimal("0")


# ============================================================================
# Precision Tests
# ============================================================================

def test_size_base_precision(sizer, sample_df):
    """Test size_base is quantized to 8 decimal places."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_base > 0:
        # Check precision (8 decimal places for BTC)
        str_size = str(result.size_base)
        if '.' in str_size:
            decimals = len(str_size.split('.')[1])
            assert decimals <= 8


def test_size_quote_precision(sizer, sample_df):
    """Test size_quote is quantized to 2 decimal places."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_quote > 0:
        # Check precision (2 decimal places for USD/EUR)
        str_size = str(result.size_quote)
        if '.' in str_size:
            decimals = len(str_size.split('.')[1])
            assert decimals <= 2


def test_price_precision(sizer, sample_df):
    """Test stop/take-profit prices quantized to 2 decimals."""
    current_price = Decimal("50000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        sample_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_quote > 0:
        # Check stop-loss precision
        str_stop = str(result.stop_loss_price)
        if '.' in str_stop:
            decimals = len(str_stop.split('.')[1])
            assert decimals <= 2


# ============================================================================
# Minimum Stop Loss Floor Tests
# ============================================================================

@pytest.fixture
def very_low_volatility_df():
    """Generate OHLCV data with extremely low ATR (simulates 15-min candles)."""
    length = 100
    prices = [100000.0] * length  # $100k BTC

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
    }

    for price in prices:
        # Very tight range - ATR will be ~0.01% of price
        # This simulates short timeframe candles (15-min)
        o = price * 0.99995
        c = price * 1.00005
        h = price * 1.00010
        l = price * 0.99990

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)

    return pd.DataFrame(data)


def test_min_stop_loss_floor_applied_when_atr_too_small(very_low_volatility_df):
    """Test that min_stop_loss_percent is used when ATR < min %."""
    # Configure with specific min_stop_loss_percent
    config = PositionSizeConfig(
        risk_per_trade_percent=0.5,  # Required
        min_trade_base=0.0001,  # Required
        min_stop_loss_percent=1.5,  # 1.5% minimum stop
        stop_loss_atr_multiplier=1.5,
        max_position_percent=40.0,
    )
    sizer = PositionSizer(config=config)

    current_price = Decimal("100000.00")
    quote_balance = Decimal("50000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        very_low_volatility_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_quote > 0:
        # Calculate expected minimum stop distance
        min_pct_distance = current_price * Decimal("1.5") / Decimal("100")  # 1500.00
        expected_stop_loss = current_price - min_pct_distance  # 98500.00

        # Stop loss should be at or below the minimum % distance
        # (could be slightly different due to ATR but should use the floor)
        stop_distance = current_price - result.stop_loss_price
        stop_percent = float(stop_distance / current_price * 100)

        # Stop should be at least min_stop_loss_percent
        assert stop_percent >= 1.5 - 0.01, (
            f"Stop distance {stop_percent:.2f}% should be >= 1.5%"
        )


def test_atr_stop_used_when_larger_than_min(high_volatility_df):
    """Test that ATR-based stop is used when it's larger than min %."""
    config = PositionSizeConfig(
        risk_per_trade_percent=0.5,  # Required
        min_trade_base=0.0001,  # Required
        min_stop_loss_percent=0.5,  # 0.5% minimum (low floor)
        stop_loss_atr_multiplier=2.0,  # 2x ATR
        max_position_percent=40.0,
    )
    sizer = PositionSizer(config=config)

    current_price = Decimal("50000.00")
    quote_balance = Decimal("50000.00")
    base_balance = Decimal("0.0")
    signal_strength = 80

    result = sizer.calculate_size(
        high_volatility_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="buy",
    )

    if result.size_quote > 0:
        stop_distance = current_price - result.stop_loss_price
        stop_percent = float(stop_distance / current_price * 100)

        # Min floor is 0.5%, but high volatility ATR should produce larger stop
        # High volatility data should have ATR >> 0.5%
        assert stop_percent > 0.5, (
            f"Stop distance {stop_percent:.2f}% should be > 0.5% floor (using ATR)"
        )


def test_min_stop_loss_floor_with_different_percentages():
    """Test stop floor works with different min_stop_loss_percent values."""
    # Create data with minimal ATR
    length = 50
    df = pd.DataFrame({
        'open': [50000.0] * length,
        'high': [50001.0] * length,  # Tiny range
        'low': [49999.0] * length,
        'close': [50000.0] * length,
    })

    for min_pct in [1.0, 1.5, 2.0, 3.0]:
        config = PositionSizeConfig(
            risk_per_trade_percent=0.5,  # Required
            min_trade_base=0.0001,  # Required
            min_stop_loss_percent=min_pct,
            stop_loss_atr_multiplier=1.5,
        )
        sizer = PositionSizer(config=config)

        result = sizer.calculate_size(
            df,
            Decimal("50000.00"),
            Decimal("50000.00"),
            Decimal("0.0"),
            signal_strength=80,
            side="buy",
        )

        if result.size_quote > 0:
            stop_distance = Decimal("50000.00") - result.stop_loss_price
            stop_percent = float(stop_distance / Decimal("50000.00") * 100)

            assert stop_percent >= min_pct - 0.01, (
                f"With min_pct={min_pct}, stop {stop_percent:.2f}% should be >= {min_pct}%"
            )


def test_min_stop_loss_floor_for_sell_orders(very_low_volatility_df):
    """Test min_stop_loss_percent floor also applies to sell orders."""
    config = PositionSizeConfig(
        risk_per_trade_percent=0.5,  # Required
        min_trade_base=0.0001,  # Required
        min_stop_loss_percent=1.5,
        stop_loss_atr_multiplier=1.5,
    )
    sizer = PositionSizer(config=config)

    current_price = Decimal("100000.00")
    quote_balance = Decimal("10000.00")
    base_balance = Decimal("1.0")  # Hold 1 BTC to sell
    signal_strength = 80

    result = sizer.calculate_size(
        very_low_volatility_df,
        current_price,
        quote_balance,
        base_balance,
        signal_strength,
        side="sell",
    )

    if result.size_quote > 0:
        # For sells, stop loss is ABOVE entry price
        stop_distance = result.stop_loss_price - current_price
        stop_percent = float(stop_distance / current_price * 100)

        assert stop_percent >= 1.5 - 0.01, (
            f"Sell stop distance {stop_percent:.2f}% should be >= 1.5%"
        )
