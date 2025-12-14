"""
Comprehensive tests for the Database state management module.

Tests cover:
- Paper/live data separation (CRITICAL - prevents mixing test and real money data)
- Position tracking (create, update, get current, history)
- Order management (create, update, recent orders)
- Trade recording with balance snapshots
- Daily statistics (update, get, date ranges)
- Trailing stops (create, update, deactivate, DCA updates)
- Rate history (record, bulk insert, get, timestamps, duplicates)
- System state (set, get, delete, JSON encoding)
- Regime history (record, get last, get history)
- Session management (commit/rollback, context manager)
- Decimal precision preservation
- Timestamp handling (UTC, naive warnings)
- Unique constraints and duplicate handling
- Error handling and transaction rollback
"""

import pytest
from decimal import Decimal
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.state.database import (
    Database,
    Position,
    Order,
    Trade,
    DailyStats,
    SystemState,
    TrailingStop,
    RegimeHistory,
    RateHistory,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def db_path(tmp_path):
    """Create a valid database path within the temp directory."""
    # Mock Path.cwd to make the tmp_path appear as project root
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        yield data_dir / "test_trader.db"


@pytest.fixture
def db(db_path):
    """Initialize a fresh database instance for each test."""
    with patch("pathlib.Path.cwd", return_value=db_path.parent.parent):
        db_instance = Database(db_path)
        yield db_instance


# ============================================================================
# Position Tests - Paper/Live Separation
# ============================================================================

def test_position_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live positions remain strictly separated.
    Risk: Mixing paper/live data causes financial loss.
    """
    symbol = "BTC-USD"

    # Create LIVE position
    live_pos = db.update_position(
        symbol=symbol,
        quantity=Decimal("1.5"),
        average_cost=Decimal("50000.00"),
        is_paper=False
    )

    # Create PAPER position for same symbol
    paper_pos = db.update_position(
        symbol=symbol,
        quantity=Decimal("10.0"),
        average_cost=Decimal("45000.00"),
        is_paper=True
    )

    # Verify retrieval separation
    fetched_live = db.get_current_position(symbol, is_paper=False)
    fetched_paper = db.get_current_position(symbol, is_paper=True)

    assert fetched_live is not None
    assert fetched_paper is not None
    assert fetched_live.id != fetched_paper.id

    # Verify data integrity
    assert fetched_live.get_quantity() == Decimal("1.5")
    assert fetched_paper.get_quantity() == Decimal("10.0")
    assert fetched_live.is_paper is False
    assert fetched_paper.is_paper is True


def test_position_decimal_precision(db):
    """
    Verify high-precision decimals are stored and retrieved exactly.
    Risk: Float rounding errors accumulating over time.
    """
    symbol = "ETH-USD"
    # Value that cannot be represented exactly in binary floating point
    specific_qty = Decimal("0.123456789012345678")
    specific_cost = Decimal("1234.56789123456789")

    db.update_position(
        symbol=symbol,
        quantity=specific_qty,
        average_cost=specific_cost,
        is_paper=False
    )

    pos = db.get_current_position(symbol, is_paper=False)

    # Must match exactly - string storage preserves precision
    assert pos.get_quantity() == specific_qty
    assert pos.get_average_cost() == specific_cost
    assert isinstance(pos.get_quantity(), Decimal)


def test_position_history_management(db):
    """
    Verify updating position marks old one as not current.
    Risk: Multiple 'current' positions causing calculation errors.
    """
    symbol = "SOL-USD"

    # Initial position
    pos1 = db.update_position(
        symbol=symbol,
        quantity=Decimal("10"),
        average_cost=Decimal("100"),
        is_paper=False
    )

    # Updated position
    pos2 = db.update_position(
        symbol=symbol,
        quantity=Decimal("15"),
        average_cost=Decimal("105"),
        is_paper=False
    )

    # Verify only one is current
    with db.session() as session:
        all_positions = session.query(Position).filter(
            Position.symbol == symbol,
            Position.is_paper == False
        ).all()

        assert len(all_positions) == 2

        p1 = next(p for p in all_positions if p.id == pos1.id)
        p2 = next(p for p in all_positions if p.id == pos2.id)

        assert p1.is_current is False
        assert p2.is_current is True


def test_get_current_position_empty(db):
    """Verify behavior when no position exists."""
    pos = db.get_current_position("NON-EXISTENT", is_paper=False)
    assert pos is None


# ============================================================================
# Order Tests - Paper/Live Separation
# ============================================================================

def test_order_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live orders remain separated.
    """
    symbol = "BTC-USD"

    # Create live order
    live_order = db.create_order(
        side="buy",
        size=Decimal("0.1"),
        symbol=symbol,
        is_paper=False
    )

    # Create paper order
    paper_order = db.create_order(
        side="sell",
        size=Decimal("0.2"),
        symbol=symbol,
        is_paper=True
    )

    # Verify they have different IDs
    assert live_order.id != paper_order.id
    assert live_order.is_paper is False
    assert paper_order.is_paper is True


def test_create_order_default_values(db):
    """Test order creation with default values."""
    order = db.create_order(
        side="buy",
        size=Decimal("0.05"),
        is_paper=False
    )

    assert order.symbol == "BTC-USD"
    assert order.order_type == "market"
    assert order.status == "pending"
    assert order.filled_size == "0"
    assert order.fee == "0"
    assert order.is_paper is False


def test_update_order_fields(db):
    """Test updating various order fields."""
    order = db.create_order(
        side="buy",
        size=Decimal("0.1"),
        is_paper=False
    )

    updated = db.update_order(
        order_id=order.id,
        exchange_order_id="exchange-123",
        status="filled",
        filled_size=Decimal("0.1"),
        filled_price=Decimal("50000"),
        fee=Decimal("5.00")
    )

    assert updated.exchange_order_id == "exchange-123"
    assert updated.status == "filled"
    assert updated.filled_size == "0.1"
    assert updated.filled_price == "50000"
    assert updated.fee == "5.00"


def test_update_nonexistent_order(db):
    """Test updating an order that doesn't exist."""
    result = db.update_order(order_id=99999, status="filled")
    assert result is None


def test_get_recent_orders(db):
    """Test retrieving recent orders with time filter."""
    # Create some orders
    for i in range(5):
        db.create_order(
            side="buy" if i % 2 == 0 else "sell",
            size=Decimal("0.01"),
            is_paper=False
        )

    recent = db.get_recent_orders(hours=24)
    assert len(recent) == 5


# ============================================================================
# Trade Tests - Paper/Live Separation & Balance Snapshots
# ============================================================================

def test_trade_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live trades remain separated.
    """
    symbol = "BTC-USD"

    # Record live trade
    live_trade = db.record_trade(
        side="buy",
        size=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("5.00"),
        realized_pnl=Decimal("-5.00"),
        symbol=symbol,
        is_paper=False,
        quote_balance_after=Decimal("4000"),
        base_balance_after=Decimal("0.1"),
        spot_rate=Decimal("50000")
    )

    # Record paper trade
    paper_trade = db.record_trade(
        side="sell",
        size=Decimal("0.2"),
        price=Decimal("51000"),
        symbol=symbol,
        is_paper=True,
        quote_balance_after=Decimal("15000"),
        base_balance_after=Decimal("0.8")
    )

    assert live_trade.id != paper_trade.id
    assert live_trade.is_paper is False
    assert paper_trade.is_paper is True


def test_record_trade_with_balance_snapshot(db):
    """Test trade recording includes balance snapshot."""
    trade = db.record_trade(
        side="buy",
        size=Decimal("0.05"),
        price=Decimal("50000"),
        fee=Decimal("2.50"),
        realized_pnl=Decimal("-2.50"),
        is_paper=False,
        quote_balance_after=Decimal("5000"),
        base_balance_after=Decimal("0.15"),
        spot_rate=Decimal("50000")
    )

    assert trade.quote_balance_after == "5000"
    assert trade.base_balance_after == "0.15"
    assert trade.spot_rate == "50000"


def test_get_last_paper_balance(db):
    """Test retrieving last paper trading balance."""
    # Record several paper trades
    db.record_trade(
        side="buy",
        size=Decimal("1.0"),
        price=Decimal("40000"),
        is_paper=True,
        quote_balance_after=Decimal("20000"),
        base_balance_after=Decimal("1.0"),
        spot_rate=Decimal("40000")
    )

    db.record_trade(
        side="buy",
        size=Decimal("0.5"),
        price=Decimal("41000"),
        is_paper=True,
        quote_balance_after=Decimal("15000"),
        base_balance_after=Decimal("1.5"),
        spot_rate=Decimal("41000")
    )

    # Get last balance
    result = db.get_last_paper_balance()

    assert result is not None
    quote, base, rate = result
    assert quote == Decimal("15000")
    assert base == Decimal("1.5")
    assert rate == Decimal("41000")


def test_get_last_paper_balance_no_trades(db):
    """Test get_last_paper_balance with no trades."""
    result = db.get_last_paper_balance()
    assert result is None


def test_get_last_paper_balance_ignores_live(db):
    """Test that get_last_paper_balance ignores live trades."""
    # Record live trade
    db.record_trade(
        side="buy",
        size=Decimal("1.0"),
        price=Decimal("50000"),
        is_paper=False,
        quote_balance_after=Decimal("10000"),
        base_balance_after=Decimal("1.0"),
        spot_rate=Decimal("50000")
    )

    result = db.get_last_paper_balance()
    assert result is None


# ============================================================================
# Daily Stats Tests - Paper/Live Separation
# ============================================================================

def test_daily_stats_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live daily stats remain separated.
    """
    today = datetime.now(timezone.utc).date()

    # Update live stats
    db.update_daily_stats(
        starting_balance=Decimal("10000"),
        ending_balance=Decimal("10500"),
        realized_pnl=Decimal("500"),
        total_trades=5,
        is_paper=False
    )

    # Update paper stats
    db.update_daily_stats(
        starting_balance=Decimal("50000"),
        ending_balance=Decimal("52000"),
        realized_pnl=Decimal("2000"),
        total_trades=10,
        is_paper=True
    )

    # Retrieve and verify separation
    live_stats = db.get_daily_stats(today, is_paper=False)
    paper_stats = db.get_daily_stats(today, is_paper=True)

    assert live_stats is not None
    assert paper_stats is not None
    assert live_stats.id != paper_stats.id
    assert Decimal(live_stats.realized_pnl) == Decimal("500")
    assert Decimal(paper_stats.realized_pnl) == Decimal("2000")


def test_daily_stats_create_if_not_exists(db):
    """Test that update_daily_stats creates record if it doesn't exist."""
    today = datetime.now(timezone.utc).date()

    db.update_daily_stats(
        starting_balance=Decimal("10000"),
        is_paper=False
    )

    stats = db.get_daily_stats(today, is_paper=False)
    assert stats is not None
    assert Decimal(stats.starting_balance) == Decimal("10000")


def test_daily_stats_update_existing(db):
    """Test updating existing daily stats record."""
    today = datetime.now(timezone.utc).date()

    # Create initial stats
    db.update_daily_stats(
        starting_balance=Decimal("10000"),
        total_trades=0,
        is_paper=False
    )

    # Update with new values
    db.update_daily_stats(
        ending_balance=Decimal("10500"),
        total_trades=5,
        is_paper=False
    )

    stats = db.get_daily_stats(today, is_paper=False)
    assert Decimal(stats.ending_balance) == Decimal("10500")
    assert stats.total_trades == 5


def test_get_daily_stats_range(db):
    """Test retrieving daily stats for a date range."""
    start = date(2024, 1, 1)
    end = date(2024, 1, 5)

    # Create stats for each day
    for i in range(5):
        target_date = start + timedelta(days=i)
        mock_now = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        with patch('src.state.database.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_now
            db.update_daily_stats(
                starting_balance=Decimal("10000"),
                total_trades=i,
                is_paper=False
            )

    stats_list = db.get_daily_stats_range(start, end, is_paper=False)
    assert len(stats_list) == 5
    assert stats_list[0].total_trades == 0
    assert stats_list[4].total_trades == 4


# ============================================================================
# System State Tests - JSON Encoding/Decoding
# ============================================================================

def test_system_state_set_and_get(db):
    """Test setting and getting system state."""
    db.set_state("test_key", {"value": 123, "name": "test"})

    result = db.get_state("test_key")
    assert result == {"value": 123, "name": "test"}


def test_system_state_update_existing(db):
    """Test updating existing system state."""
    db.set_state("key", "initial")
    db.set_state("key", "updated")

    result = db.get_state("key")
    assert result == "updated"


def test_system_state_get_default(db):
    """Test getting non-existent state returns default."""
    result = db.get_state("nonexistent", default="default_value")
    assert result == "default_value"


def test_system_state_delete(db):
    """Test deleting system state."""
    db.set_state("to_delete", "value")

    deleted = db.delete_state("to_delete")
    assert deleted is True

    result = db.get_state("to_delete")
    assert result is None


def test_system_state_delete_nonexistent(db):
    """Test deleting non-existent state."""
    deleted = db.delete_state("nonexistent")
    assert deleted is False


# ============================================================================
# Trailing Stop Tests - DCA Safety
# ============================================================================

def test_trailing_stop_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live trailing stops remain separated.
    """
    symbol = "BTC-USD"

    # Create live trailing stop
    live_ts = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        is_paper=False
    )

    # Create paper trailing stop
    paper_ts = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("45000"),
        trailing_activation=Decimal("47000"),
        trailing_distance=Decimal("900"),
        is_paper=True
    )

    # Verify separation
    active_live = db.get_active_trailing_stop(symbol, is_paper=False)
    active_paper = db.get_active_trailing_stop(symbol, is_paper=True)

    assert active_live is not None
    assert active_paper is not None
    assert active_live.id != active_paper.id
    assert active_live.get_entry_price() == Decimal("50000")
    assert active_paper.get_entry_price() == Decimal("45000")


def test_trailing_stop_deactivates_old(db):
    """Test that creating new trailing stop deactivates old one."""
    symbol = "BTC-USD"

    # Create first stop
    ts1 = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        is_paper=False
    )

    # Create second stop (should deactivate first)
    ts2 = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("51000"),
        trailing_activation=Decimal("53000"),
        trailing_distance=Decimal("1000"),
        is_paper=False
    )

    # Verify only second is active
    with db.session() as session:
        old = session.query(TrailingStop).filter(TrailingStop.id == ts1.id).first()
        new = session.query(TrailingStop).filter(TrailingStop.id == ts2.id).first()

        assert old.is_active is False
        assert new.is_active is True


def test_update_trailing_stop_for_dca(db):
    """
    CRITICAL: Test DCA update doesn't deactivate stop (no protection gap).
    """
    symbol = "BTC-USD"

    # Create initial stop
    ts = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        hard_stop=Decimal("48000"),
        is_paper=False
    )

    # Update for DCA (position average changed)
    updated = db.update_trailing_stop_for_dca(
        symbol=symbol,
        entry_price=Decimal("49000"),  # New average cost
        trailing_activation=Decimal("51000"),
        trailing_distance=Decimal("1000"),
        hard_stop=Decimal("47000"),
        is_paper=False
    )

    assert updated is not None
    assert updated.id == ts.id  # Same record
    assert updated.is_active is True  # Still active
    assert updated.get_entry_price() == Decimal("49000")
    assert updated.get_hard_stop() == Decimal("47000")


def test_update_trailing_stop_level(db):
    """Test updating the trailing stop level."""
    ts = db.create_trailing_stop(
        symbol="BTC-USD",
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        is_paper=False
    )

    updated = db.update_trailing_stop(
        trailing_stop_id=ts.id,
        new_stop_level=Decimal("51500")
    )

    assert updated.get_trailing_stop() == Decimal("51500")


def test_deactivate_trailing_stop(db):
    """Test deactivating trailing stop."""
    db.create_trailing_stop(
        symbol="BTC-USD",
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        is_paper=False
    )

    result = db.deactivate_trailing_stop("BTC-USD", is_paper=False)
    assert result is True

    active = db.get_active_trailing_stop("BTC-USD", is_paper=False)
    assert active is None


def test_update_trailing_stop_breakeven(db):
    """
    CRITICAL: Test break-even stop activation moves hard stop to entry price.
    """
    ts = db.create_trailing_stop(
        symbol="BTC-USD",
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        hard_stop=Decimal("48000"),  # Initial hard stop below entry
        is_paper=False
    )

    # Verify initial state
    assert ts.is_breakeven_active() is False
    assert ts.get_hard_stop() == Decimal("48000")

    # Activate break-even protection
    updated = db.update_trailing_stop_breakeven(
        trailing_stop_id=ts.id,
        new_hard_stop=Decimal("50000")  # Move to entry price
    )

    assert updated is not None
    assert updated.is_breakeven_active() is True
    assert updated.get_hard_stop() == Decimal("50000")  # Now at entry

    # Verify persistence (fetch fresh from DB)
    fresh_ts = db.get_active_trailing_stop("BTC-USD", is_paper=False)
    assert fresh_ts is not None
    assert fresh_ts.is_breakeven_active() is True
    assert fresh_ts.get_hard_stop() == Decimal("50000")


def test_breakeven_flag_resets_on_dca(db):
    """
    CRITICAL: Test that break-even flag resets when DCA occurs.

    When position is averaged, the break-even level changes,
    so the flag must reset to allow re-triggering at new level.
    """
    symbol = "BTC-USD"

    # Create stop with break-even already triggered
    ts = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        hard_stop=Decimal("50000"),  # Already at break-even
        is_paper=False
    )
    # Manually trigger break-even
    db.update_trailing_stop_breakeven(ts.id, new_hard_stop=Decimal("50000"))

    # Verify break-even is active
    active = db.get_active_trailing_stop(symbol, is_paper=False)
    assert active.is_breakeven_active() is True

    # DCA update should reset break-even flag
    updated = db.update_trailing_stop_for_dca(
        symbol=symbol,
        entry_price=Decimal("49000"),  # New average cost
        trailing_activation=Decimal("51000"),
        trailing_distance=Decimal("1000"),
        hard_stop=Decimal("47000"),  # New hard stop based on new avg
        is_paper=False
    )

    assert updated.is_breakeven_active() is False  # Reset for re-triggering
    assert updated.get_hard_stop() == Decimal("47000")  # New calculated hard stop


def test_breakeven_paper_live_separation(db):
    """
    CRITICAL: Verify break-even state is separate for paper and live trading.
    """
    symbol = "BTC-USD"

    # Create paper stop
    paper_ts = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        hard_stop=Decimal("48000"),
        is_paper=True
    )

    # Create live stop
    live_ts = db.create_trailing_stop(
        symbol=symbol,
        side="buy",
        entry_price=Decimal("50000"),
        trailing_activation=Decimal("52000"),
        trailing_distance=Decimal("1000"),
        hard_stop=Decimal("48000"),
        is_paper=False
    )

    # Trigger break-even on paper only
    db.update_trailing_stop_breakeven(paper_ts.id, new_hard_stop=Decimal("50000"))

    # Verify separation
    paper_active = db.get_active_trailing_stop(symbol, is_paper=True)
    live_active = db.get_active_trailing_stop(symbol, is_paper=False)

    assert paper_active.is_breakeven_active() is True
    assert live_active.is_breakeven_active() is False


# ============================================================================
# Rate History Tests - Duplicate Handling & Atomicity
# ============================================================================

def test_rate_history_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live rate history remain separated.
    """
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    # Record live rate
    live_rate = db.record_rate(
        timestamp=timestamp,
        open_price=Decimal("50000"),
        high_price=Decimal("50500"),
        low_price=Decimal("49500"),
        close_price=Decimal("50200"),
        volume=Decimal("100"),
        is_paper=False
    )

    # Record paper rate for same timestamp
    paper_rate = db.record_rate(
        timestamp=timestamp,
        open_price=Decimal("51000"),
        high_price=Decimal("51500"),
        low_price=Decimal("50500"),
        close_price=Decimal("51200"),
        volume=Decimal("200"),
        is_paper=True
    )

    assert live_rate is not None
    assert paper_rate is not None
    assert live_rate.id != paper_rate.id


def test_rate_history_duplicate_skipped(db):
    """Test that duplicate rates are skipped."""
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    # Record first time
    rate1 = db.record_rate(
        timestamp=timestamp,
        open_price=Decimal("50000"),
        high_price=Decimal("50500"),
        low_price=Decimal("49500"),
        close_price=Decimal("50200"),
        volume=Decimal("100"),
        is_paper=False
    )

    # Try to record duplicate
    rate2 = db.record_rate(
        timestamp=timestamp,
        open_price=Decimal("50000"),
        high_price=Decimal("50500"),
        low_price=Decimal("49500"),
        close_price=Decimal("50200"),
        volume=Decimal("100"),
        is_paper=False
    )

    assert rate1 is not None
    assert rate2 is None  # Duplicate skipped


def test_record_rates_bulk(db):
    """Test bulk insert of rate history."""
    candles = [
        {
            "timestamp": datetime(2024, 1, 1, 12, i, 0),
            "open": Decimal("50000") + i,
            "high": Decimal("50500") + i,
            "low": Decimal("49500") + i,
            "close": Decimal("50200") + i,
            "volume": Decimal("100")
        }
        for i in range(5)
    ]

    inserted = db.record_rates_bulk(candles, is_paper=False)

    assert inserted == 5

    # Verify data
    count = db.get_rate_count(is_paper=False)
    assert count == 5


def test_record_rates_bulk_skip_duplicates(db):
    """Test bulk insert skips duplicates."""
    candles = [
        {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "open": Decimal("50000"),
            "high": Decimal("50500"),
            "low": Decimal("49500"),
            "close": Decimal("50200"),
            "volume": Decimal("100")
        }
    ]

    # Insert first time
    inserted1 = db.record_rates_bulk(candles, is_paper=False)
    assert inserted1 == 1

    # Try to insert duplicate
    inserted2 = db.record_rates_bulk(candles, is_paper=False)
    assert inserted2 == 0  # Duplicate skipped


def test_record_rates_bulk_partial_duplicates(db):
    """Test bulk insert with mix of new and duplicate candles."""
    # Insert initial candles
    initial = [
        {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "open": Decimal("50000"),
            "high": Decimal("50500"),
            "low": Decimal("49500"),
            "close": Decimal("50200"),
            "volume": Decimal("100")
        }
    ]
    db.record_rates_bulk(initial, is_paper=False)

    # Try to insert mix of duplicate and new
    mixed = [
        initial[0],  # Duplicate
        {
            "timestamp": datetime(2024, 1, 1, 12, 1, 0),
            "open": Decimal("50200"),
            "high": Decimal("50700"),
            "low": Decimal("50000"),
            "close": Decimal("50400"),
            "volume": Decimal("110")
        }
    ]

    inserted = db.record_rates_bulk(mixed, is_paper=False)
    assert inserted == 1  # Only new one inserted


def test_get_rates_with_time_filter(db):
    """Test retrieving rates with time filter."""
    candles = [
        {
            "timestamp": datetime(2024, 1, 1, 12, i, 0),
            "open": Decimal("50000"),
            "high": Decimal("50500"),
            "low": Decimal("49500"),
            "close": Decimal("50200"),
            "volume": Decimal("100")
        }
        for i in range(10)
    ]

    db.record_rates_bulk(candles, is_paper=False)

    # Get rates in range
    rates = db.get_rates(
        start=datetime(2024, 1, 1, 12, 3, 0),
        end=datetime(2024, 1, 1, 12, 7, 0),
        is_paper=False
    )

    assert len(rates) == 5  # Minutes 3, 4, 5, 6, 7


def test_get_latest_rate_timestamp(db):
    """Test getting latest rate timestamp."""
    candles = [
        {
            "timestamp": datetime(2024, 1, 1, 12, i, 0),
            "open": Decimal("50000"),
            "high": Decimal("50500"),
            "low": Decimal("49500"),
            "close": Decimal("50200"),
            "volume": Decimal("100")
        }
        for i in range(5)
    ]

    db.record_rates_bulk(candles, is_paper=False)

    latest = db.get_latest_rate_timestamp(is_paper=False)

    assert latest == datetime(2024, 1, 1, 12, 4, 0)


def test_get_latest_rate_timestamp_empty(db):
    """Test getting latest timestamp when no rates exist."""
    latest = db.get_latest_rate_timestamp(is_paper=False)
    assert latest is None


def test_rate_history_decimal_conversion(db):
    """Test RateHistory decimal conversion methods."""
    rate = db.record_rate(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        open_price=Decimal("50000.12"),
        high_price=Decimal("50500.34"),
        low_price=Decimal("49500.56"),
        close_price=Decimal("50200.78"),
        volume=Decimal("100.99"),
        is_paper=False
    )

    assert rate.get_open() == Decimal("50000.12")
    assert rate.get_high() == Decimal("50500.34")
    assert rate.get_low() == Decimal("49500.56")
    assert rate.get_close() == Decimal("50200.78")
    assert rate.get_volume() == Decimal("100.99")


def test_rate_history_negative_price_validation(db):
    """Test RateHistory rejects negative prices."""
    rate = db.record_rate(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        open_price=Decimal("-50000"),
        high_price=Decimal("50500"),
        low_price=Decimal("49500"),
        close_price=Decimal("50200"),
        volume=Decimal("100"),
        is_paper=False
    )

    with pytest.raises(ValueError, match="Invalid open_price"):
        rate.get_open()


def test_rate_history_timestamp_conversion(db):
    """Test timezone-aware timestamp conversion to naive UTC."""
    # Timezone-aware timestamp
    aware_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    candles = [{
        "timestamp": aware_ts,
        "open": Decimal("50000"),
        "high": Decimal("50500"),
        "low": Decimal("49500"),
        "close": Decimal("50200"),
        "volume": Decimal("100")
    }]

    db.record_rates_bulk(candles, is_paper=False)

    # Retrieve and verify stored as naive UTC
    rates = db.get_rates(is_paper=False)
    assert len(rates) == 1
    assert rates[0].timestamp.tzinfo is None  # Naive datetime


def test_rate_history_pandas_timestamp_conversion(db):
    """Test Pandas Timestamp conversion to datetime."""
    # Pandas Timestamp
    pd_ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")

    candles = [{
        "timestamp": pd_ts,
        "open": Decimal("50000"),
        "high": Decimal("50500"),
        "low": Decimal("49500"),
        "close": Decimal("50200"),
        "volume": Decimal("100")
    }]

    db.record_rates_bulk(candles, is_paper=False)

    rates = db.get_rates(is_paper=False)
    assert len(rates) == 1
    assert isinstance(rates[0].timestamp, datetime)


# ============================================================================
# Regime History Tests - Paper/Live Separation
# ============================================================================

def test_regime_history_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live regime history remain separated.
    """
    # Record live regime
    live_regime = db.record_regime_change(
        regime_name="risk_on",
        threshold_adjustment=-10,
        position_multiplier=1.2,
        sentiment_value=25,
        sentiment_category="fear",
        is_paper=False
    )

    # Record paper regime
    paper_regime = db.record_regime_change(
        regime_name="risk_off",
        threshold_adjustment=10,
        position_multiplier=0.8,
        sentiment_value=75,
        sentiment_category="greed",
        is_paper=True
    )

    # Verify separation
    last_live = db.get_last_regime(is_paper=False)
    last_paper = db.get_last_regime(is_paper=True)

    assert last_live == "risk_on"
    assert last_paper == "risk_off"


def test_get_regime_history(db):
    """Test retrieving regime change history."""
    # Record several regime changes
    for i in range(5):
        db.record_regime_change(
            regime_name=f"regime_{i}",
            threshold_adjustment=i,
            position_multiplier=1.0,
            is_paper=False
        )

    history = db.get_regime_history(hours=24, is_paper=False)
    assert len(history) == 5


def test_get_last_regime_empty(db):
    """Test get_last_regime with no history."""
    last = db.get_last_regime(is_paper=False)
    assert last is None


# ============================================================================
# Session Management Tests - Commit/Rollback
# ============================================================================

def test_session_commit_on_success(db):
    """Test session commits on successful operation."""
    db.set_state("test_commit", "value")

    # Verify committed (can read in new session)
    result = db.get_state("test_commit")
    assert result == "value"


def test_session_rollback_on_exception(db):
    """Test session rolls back on exception."""
    try:
        with db.session() as session:
            # Insert a state
            state = SystemState(key="rollback_test", value='"initial"')
            session.add(state)
            session.flush()

            # Force an exception
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Verify rolled back (key doesn't exist)
    result = db.get_state("rollback_test")
    assert result is None


# ============================================================================
# Database Path Validation Tests
# ============================================================================

def test_database_path_validation():
    """Test database path must be within allowed directory."""
    with patch("pathlib.Path.cwd", return_value=Path("/tmp/project")):
        # Try to create database outside allowed directory
        with pytest.raises(ValueError, match="must be within"):
            Database(Path("/etc/passwd"))


def test_database_creates_parent_directory(tmp_path):
    """Test database creates parent directory if it doesn't exist."""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        nested_path = tmp_path / "data" / "nested" / "deep" / "test.db"

        db_instance = Database(nested_path)

        assert nested_path.parent.exists()


# ============================================================================
# Edge Cases & Error Handling
# ============================================================================

def test_very_large_decimal_values(db):
    """Test handling of very large decimal values."""
    large_value = Decimal("999999999999.99999999")

    db.update_position(
        symbol="TEST-USD",
        quantity=large_value,
        average_cost=large_value,
        is_paper=False
    )

    pos = db.get_current_position("TEST-USD", is_paper=False)
    assert pos.get_quantity() == large_value


def test_very_small_decimal_values(db):
    """Test handling of very small decimal values."""
    small_value = Decimal("0.00000001")

    db.update_position(
        symbol="TEST-USD",
        quantity=small_value,
        average_cost=small_value,
        is_paper=False
    )

    pos = db.get_current_position("TEST-USD", is_paper=False)
    assert pos.get_quantity() == small_value


def test_empty_query_results(db):
    """Test empty query results return appropriate types."""
    # Empty list
    orders = db.get_recent_orders(hours=24)
    assert orders == []

    # None
    position = db.get_current_position("NONEXISTENT", is_paper=False)
    assert position is None
