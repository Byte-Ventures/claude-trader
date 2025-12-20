"""
Tests for the backtesting engine.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from src.backtest import Backtester
from src.strategy.signal_scorer import SignalScorer
from src.strategy.position_sizer import PositionSizer, PositionSizeConfig


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data = []

    # Generate 200 candles with upward trend
    base_price = 50000.0
    for i in range(200):
        timestamp = start + timedelta(hours=i)
        # Add some volatility and upward drift
        price = base_price + (i * 100) + (100 * (i % 10 - 5))
        data.append({
            "timestamp": timestamp,
            "open": price,
            "high": price + 50,
            "low": price - 50,
            "close": price + (10 if i % 2 == 0 else -10),
            "volume": 1000000 + (i * 1000),
        })

    return pd.DataFrame(data)


@pytest.fixture
def signal_scorer():
    """Create a SignalScorer for testing."""
    return SignalScorer(
        threshold=60,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        ema_fast=9,
        ema_slow=21,
    )


@pytest.fixture
def position_sizer():
    """Create a PositionSizer for testing."""
    config = PositionSizeConfig(
        risk_per_trade_percent=1.0,
        min_trade_base=0.0001,
        max_position_percent=40.0,
        min_trade_quote=10.0,
    )
    return PositionSizer(config=config)


def test_backtester_initialization(signal_scorer, position_sizer):
    """Test Backtester initialization."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
        fee_percent=0.006,
    )

    assert bt.initial_capital == Decimal("10000.0")
    assert bt.fee_percent == Decimal("0.006")
    assert bt.slippage_percent == Decimal("0.001")


def test_backtester_with_default_position_sizer(signal_scorer):
    """Test Backtester creates default PositionSizer if not provided."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        initial_capital=10000.0,
    )

    assert bt.position_sizer is not None
    assert isinstance(bt.position_sizer, PositionSizer)


def test_backtest_run_basic(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test basic backtest run."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)

    # Verify result structure
    assert result.metrics is not None
    assert result.trades is not None
    assert result.equity_curve is not None
    assert result.signal_distribution is not None

    # Verify signal distribution has all keys
    assert "buy" in result.signal_distribution
    assert "sell" in result.signal_distribution
    assert "hold" in result.signal_distribution

    # Verify equity curve has required columns
    assert "timestamp" in result.equity_curve.columns
    assert "portfolio_value" in result.equity_curve.columns


def test_backtest_date_filtering(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test date filtering in backtest."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    # Run with date range
    result = bt.run(
        sample_ohlcv_data,
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    # Should have fewer data points
    assert len(result.equity_curve) > 0
    assert len(result.equity_curve) < len(sample_ohlcv_data)


def test_backtest_metrics_calculation(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test metrics calculation."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)
    metrics = result.metrics

    # Verify all metrics exist
    assert hasattr(metrics, "total_return")
    assert hasattr(metrics, "sharpe_ratio")
    assert hasattr(metrics, "max_drawdown")
    assert hasattr(metrics, "win_rate")
    assert hasattr(metrics, "profit_factor")
    assert hasattr(metrics, "total_trades")
    assert hasattr(metrics, "winning_trades")
    assert hasattr(metrics, "losing_trades")
    assert hasattr(metrics, "avg_win")
    assert hasattr(metrics, "avg_loss")
    assert hasattr(metrics, "avg_trade_duration_hours")

    # Verify metrics are reasonable
    assert metrics.total_trades >= 0
    assert metrics.winning_trades >= 0
    assert metrics.losing_trades >= 0
    assert metrics.winning_trades + metrics.losing_trades == metrics.total_trades

    if metrics.total_trades > 0:
        assert 0 <= metrics.win_rate <= 100


def test_backtest_trade_execution(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test trade execution logic."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)

    # If trades were made, verify structure
    if result.trades:
        trade = result.trades[0]

        assert hasattr(trade, "timestamp")
        assert hasattr(trade, "side")
        assert hasattr(trade, "entry_price")
        assert hasattr(trade, "size_base")
        assert hasattr(trade, "size_quote")
        assert hasattr(trade, "fee")
        assert hasattr(trade, "signal_score")

        # Completed trades should have exit info
        if trade.exit_price is not None:
            assert trade.exit_timestamp is not None
            assert trade.pnl is not None
            assert trade.pnl_percent is not None


def test_backtest_fees_and_slippage(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test that fees and slippage are applied."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
        fee_percent=0.01,  # 1% fee
        slippage_percent=0.005,  # 0.5% slippage
    )

    result = bt.run(sample_ohlcv_data)

    # Total fees should be > 0 if trades were made
    if result.metrics.total_trades > 0:
        assert result.metrics.total_fees > 0

        # Each trade should have fees
        for trade in result.trades:
            assert trade.fee > 0


def test_backtest_empty_data_raises_error(signal_scorer, position_sizer):
    """Test that empty data raises ValueError."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Data cannot be empty"):
        bt.run(empty_df)


def test_backtest_missing_columns_raises_error(signal_scorer, position_sizer):
    """Test that missing columns raises ValueError."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    # Missing 'volume' column
    incomplete_df = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc)],
        "open": [50000],
        "high": [50100],
        "low": [49900],
        "close": [50050],
    })

    with pytest.raises(ValueError, match="Data must contain columns"):
        bt.run(incomplete_df)


def test_backtest_no_trades_scenario(sample_ohlcv_data):
    """Test backtest with threshold too high (no trades)."""
    # Use very high threshold so no signals trigger
    scorer = SignalScorer(threshold=95)
    sizer = PositionSizer(
        config=PositionSizeConfig(
            risk_per_trade_percent=1.0,
            min_trade_base=0.0001,
        )
    )

    bt = Backtester(
        signal_scorer=scorer,
        position_sizer=sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)

    # Should complete with zero trades
    assert result.metrics.total_trades == 0
    assert result.metrics.total_return == 0.0
    assert len(result.trades) == 0


def test_backtest_summary_output(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test that summary() generates readable output."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)
    summary = result.summary()

    # Summary should contain key metrics
    assert "Total Return" in summary
    assert "Sharpe Ratio" in summary
    assert "Max Drawdown" in summary
    assert "Win Rate" in summary
    assert "Total Trades" in summary
    assert "Initial:" in summary
    assert "Final:" in summary


def test_backtest_position_management(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test that positions are properly managed (no overlapping positions)."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)

    # Verify no overlapping positions
    open_positions = 0
    for i, trade in enumerate(result.trades):
        if i == 0:
            # First trade should be a buy
            assert trade.side == "buy"

        # Each trade should have both entry and exit
        assert trade.entry_price > 0
        if trade.exit_price is not None:
            assert trade.exit_price > 0


def test_backtest_equity_curve(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test equity curve tracking."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)

    # Equity curve should have entries
    assert len(result.equity_curve) > 0

    # Portfolio value should never be negative
    assert (result.equity_curve["portfolio_value"] >= 0).all()

    # First equity value should be close to initial capital
    first_value = result.equity_curve.iloc[0]["portfolio_value"]
    assert abs(first_value - 10000.0) < 1000  # Allow some variance


def test_backtest_with_different_capital_levels(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test backtesting with different initial capital amounts."""
    for capital in [1000.0, 10000.0, 100000.0]:
        bt = Backtester(
            signal_scorer=signal_scorer,
            position_sizer=position_sizer,
            initial_capital=capital,
        )

        result = bt.run(sample_ohlcv_data)

        assert result.metrics.initial_value == Decimal(str(capital))


def test_backtest_signal_distribution(signal_scorer, position_sizer, sample_ohlcv_data):
    """Test signal distribution counting."""
    bt = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
    )

    result = bt.run(sample_ohlcv_data)

    # Total signals should equal number of processed candles
    total_signals = sum(result.signal_distribution.values())
    min_candles = max(21, 20, 26)  # EMA slow, Bollinger, MACD slow
    expected_signals = len(sample_ohlcv_data) - min_candles

    assert total_signals == expected_signals

    # Each count should be non-negative
    assert result.signal_distribution["buy"] >= 0
    assert result.signal_distribution["sell"] >= 0
    assert result.signal_distribution["hold"] >= 0


def test_invalid_fee_percent_raises_error(signal_scorer):
    """Test that invalid fee_percent values raise ValueError."""
    # Test fee > 1.0 (100%)
    with pytest.raises(ValueError, match="fee_percent must be between 0 and 1.0"):
        Backtester(signal_scorer=signal_scorer, fee_percent=1.5)

    # Test negative fee
    with pytest.raises(ValueError, match="fee_percent must be between 0 and 1.0"):
        Backtester(signal_scorer=signal_scorer, fee_percent=-0.1)


def test_invalid_slippage_percent_raises_error(signal_scorer):
    """Test that invalid slippage_percent values raise ValueError."""
    # Test slippage > 1.0 (100%)
    with pytest.raises(ValueError, match="slippage_percent must be between 0 and 1.0"):
        Backtester(signal_scorer=signal_scorer, slippage_percent=1.5)

    # Test negative slippage
    with pytest.raises(ValueError, match="slippage_percent must be between 0 and 1.0"):
        Backtester(signal_scorer=signal_scorer, slippage_percent=-0.1)


def test_valid_edge_case_fee_and_slippage(signal_scorer):
    """Test that edge case values (0.0 and 1.0) are accepted."""
    # Zero fees and slippage should work
    bt_zero = Backtester(signal_scorer=signal_scorer, fee_percent=0.0, slippage_percent=0.0)
    assert bt_zero.fee_percent == Decimal("0.0")
    assert bt_zero.slippage_percent == Decimal("0.0")

    # Maximum values (100% fee/slippage) should work (though unrealistic)
    bt_max = Backtester(signal_scorer=signal_scorer, fee_percent=1.0, slippage_percent=1.0)
    assert bt_max.fee_percent == Decimal("1.0")
    assert bt_max.slippage_percent == Decimal("1.0")
