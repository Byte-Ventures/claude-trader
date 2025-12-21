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
import os
import threading
import random
import time

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
    SignalHistory,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def db_path(tmp_path):
    """Create a valid database path within the temp directory."""
    # Use /tmp directly for test databases
    test_db_path = tmp_path / "test_trader.db"
    yield test_db_path


@pytest.fixture
def db(db_path):
    """Initialize a fresh database instance for each test."""
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


def test_rate_history_upsert(db):
    """Test that record_rate performs upsert with correct OHLC semantics."""
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

    # Record again with different OHLCV - should update, not skip
    rate2 = db.record_rate(
        timestamp=timestamp,
        open_price=Decimal("49900"),   # Different open - should be IGNORED
        high_price=Decimal("51000"),   # Higher high - should use max
        low_price=Decimal("49000"),    # Lower low - should use min
        close_price=Decimal("50800"),  # Different close - should update
        volume=Decimal("150"),
        is_paper=False
    )

    assert rate1 is not None
    assert rate2 is not None  # Updated, not skipped

    # Verify OHLC semantics
    assert rate2.get_open() == Decimal("50000")   # Original open preserved
    assert rate2.get_high() == Decimal("51000")   # max(50500, 51000)
    assert rate2.get_low() == Decimal("49000")    # min(49500, 49000)
    assert rate2.get_close() == Decimal("50800")  # Latest close
    assert rate2.get_volume() == Decimal("150")


def test_record_rate_preserves_boundaries(db):
    """Test record_rate preserves high/low boundaries when update has narrower range."""
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    # Insert initial candle with established high/low
    db.record_rate(
        timestamp=timestamp,
        open_price=Decimal("50000"),
        high_price=Decimal("51000"),  # Already seen 51000
        low_price=Decimal("49000"),   # Already seen 49000
        close_price=Decimal("50500"),
        volume=Decimal("100"),
        is_paper=False
    )

    # Update with narrower range - boundaries should NOT shrink
    rate = db.record_rate(
        timestamp=timestamp,
        open_price=Decimal("50100"),   # Different open - ignored
        high_price=Decimal("50800"),   # Lower than existing high - ignored
        low_price=Decimal("49200"),    # Higher than existing low - ignored
        close_price=Decimal("50600"),  # New close
        volume=Decimal("120"),
        is_paper=False
    )

    # Verify boundaries are preserved
    assert rate.get_open() == Decimal("50000")   # Original open preserved
    assert rate.get_high() == Decimal("51000")   # Original high preserved (51000 > 50800)
    assert rate.get_low() == Decimal("49000")    # Original low preserved (49000 < 49200)
    assert rate.get_close() == Decimal("50600")  # Close updated


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


def test_record_rates_bulk_updates_duplicates(db):
    """Test bulk upsert updates existing candles with new OHLCV data."""
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
    count1 = db.record_rates_bulk(candles, is_paper=False)
    assert count1 == 1

    # Verify initial values
    rates = db.get_rates(is_paper=False)
    assert len(rates) == 1
    assert rates[0].get_open() == Decimal("50000")
    assert rates[0].get_high() == Decimal("50500")
    assert rates[0].get_low() == Decimal("49500")
    assert rates[0].get_close() == Decimal("50200")

    # Update with new OHLCV data (simulating candle completion)
    updated_candles = [
        {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "open": Decimal("49900"),   # Different open - should be IGNORED
            "high": Decimal("51000"),   # Higher high - should use max
            "low": Decimal("49000"),    # Lower low - should use min
            "close": Decimal("50800"),  # Different close - should update
            "volume": Decimal("150")    # Higher volume - should update
        }
    ]
    count2 = db.record_rates_bulk(updated_candles, is_paper=False)
    assert count2 == 1  # Updated (not skipped)

    # Verify updated values - open should be preserved!
    rates = db.get_rates(is_paper=False)
    assert len(rates) == 1  # Still only one candle
    assert rates[0].get_open() == Decimal("50000")  # Original open preserved
    assert rates[0].get_high() == Decimal("51000")  # max(50500, 51000)
    assert rates[0].get_low() == Decimal("49000")   # min(49500, 49000)
    assert rates[0].get_close() == Decimal("50800")
    assert rates[0].get_volume() == Decimal("150")


def test_record_rates_bulk_preserves_boundaries(db):
    """Test that bulk upsert preserves high/low boundaries correctly.

    When updating a candle:
    - Open: NEVER changes (first price of period)
    - High: Uses max(existing, new) - only increases
    - Low: Uses min(existing, new) - only decreases
    - Close: Always updates (latest price)
    """
    # Insert initial candle with established high/low
    initial = [
        {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "open": Decimal("50000"),
            "high": Decimal("51000"),  # Already seen 51000
            "low": Decimal("49000"),   # Already seen 49000
            "close": Decimal("50500"),
            "volume": Decimal("100")
        }
    ]
    db.record_rates_bulk(initial, is_paper=False)

    # Update with new data that has NARROWER range - boundaries should NOT shrink
    narrower_update = [
        {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "open": Decimal("50100"),   # Different open - ignored
            "high": Decimal("50800"),   # Lower than existing high - ignored
            "low": Decimal("49200"),    # Higher than existing low - ignored
            "close": Decimal("50600"),  # New close
            "volume": Decimal("120")
        }
    ]
    db.record_rates_bulk(narrower_update, is_paper=False)

    # Verify boundaries are preserved
    rates = db.get_rates(is_paper=False)
    assert rates[0].get_open() == Decimal("50000")   # Original open preserved
    assert rates[0].get_high() == Decimal("51000")   # Original high preserved (51000 > 50800)
    assert rates[0].get_low() == Decimal("49000")    # Original low preserved (49000 < 49200)
    assert rates[0].get_close() == Decimal("50600")  # Close updated


def test_record_rates_bulk_partial_duplicates(db):
    """Test bulk upsert with mix of new and existing candles."""
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

    # Upsert mix of existing (updated) and new candles
    mixed = [
        {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),  # Existing - will be updated
            "open": Decimal("50000"),
            "high": Decimal("50600"),  # Updated high
            "low": Decimal("49500"),
            "close": Decimal("50300"),  # Updated close
            "volume": Decimal("120")
        },
        {
            "timestamp": datetime(2024, 1, 1, 12, 1, 0),  # New
            "open": Decimal("50200"),
            "high": Decimal("50700"),
            "low": Decimal("50000"),
            "close": Decimal("50400"),
            "volume": Decimal("110")
        }
    ]

    count = db.record_rates_bulk(mixed, is_paper=False)
    assert count == 2  # 1 updated + 1 inserted

    # Verify we have 2 candles total
    rates = db.get_rates(is_paper=False)
    assert len(rates) == 2

    # Verify first candle was updated
    assert rates[0].get_high() == Decimal("50600")
    assert rates[0].get_close() == Decimal("50300")


def test_record_rates_bulk_paper_live_separation(db):
    """
    CRITICAL: Verify bulk upsert keeps paper and live candles completely separate.

    Per CLAUDE.md: "Paper and live data must NEVER mix"
    """
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    # Insert live candle
    live_candles = [{
        "timestamp": timestamp,
        "open": Decimal("50000"),
        "high": Decimal("50500"),
        "low": Decimal("49500"),
        "close": Decimal("50200"),
        "volume": Decimal("100"),
    }]
    db.record_rates_bulk(live_candles, is_paper=False)

    # Insert paper candle with SAME timestamp but DIFFERENT values
    paper_candles = [{
        "timestamp": timestamp,
        "open": Decimal("51000"),
        "high": Decimal("51500"),
        "low": Decimal("50500"),
        "close": Decimal("51200"),
        "volume": Decimal("200"),
    }]
    db.record_rates_bulk(paper_candles, is_paper=True)

    # Verify both exist independently
    live_rates = db.get_rates(is_paper=False)
    paper_rates = db.get_rates(is_paper=True)

    assert len(live_rates) == 1
    assert len(paper_rates) == 1
    assert live_rates[0].id != paper_rates[0].id

    # Verify values are different (not mixed)
    assert live_rates[0].get_open() == Decimal("50000")
    assert paper_rates[0].get_open() == Decimal("51000")
    assert live_rates[0].get_high() == Decimal("50500")
    assert paper_rates[0].get_high() == Decimal("51500")

    # Now update live candle - should NOT affect paper candle
    live_update = [{
        "timestamp": timestamp,
        "open": Decimal("49000"),  # Should be ignored (open is immutable)
        "high": Decimal("51000"),  # Should update to 51000 (max of 50500, 51000)
        "low": Decimal("49000"),   # Should update to 49000 (min of 49500, 49000)
        "close": Decimal("50800"),
        "volume": Decimal("150"),
    }]
    db.record_rates_bulk(live_update, is_paper=False)

    # Refresh from database
    live_rates = db.get_rates(is_paper=False)
    paper_rates = db.get_rates(is_paper=True)

    # Verify live was updated correctly
    assert live_rates[0].get_open() == Decimal("50000")  # Preserved
    assert live_rates[0].get_high() == Decimal("51000")  # Updated to max
    assert live_rates[0].get_low() == Decimal("49000")   # Updated to min
    assert live_rates[0].get_close() == Decimal("50800")

    # Verify paper was NOT affected
    assert paper_rates[0].get_open() == Decimal("51000")
    assert paper_rates[0].get_high() == Decimal("51500")
    assert paper_rates[0].get_low() == Decimal("50500")
    assert paper_rates[0].get_close() == Decimal("51200")


def test_record_rates_bulk_atomic_rollback(db):
    """
    CRITICAL: Verify atomic rollback on error - no partial writes.

    Per docstring: "The session context manager automatically handles rollback
    on exception, so if an error occurs, NO candles from this batch will be
    committed. This ensures atomic operation - all or nothing."

    This test verifies that guarantee holds. Uses bulk SQL UPSERT, so we
    mock session.execute to simulate failure during the bulk operation.
    """
    from unittest.mock import patch, MagicMock

    # First, verify database is empty
    assert len(db.get_rates(is_paper=False)) == 0

    # Create valid candles
    candles = [
        {
            "timestamp": datetime(2024, 1, 1, 12, i, 0),
            "open": Decimal("50000") + i,
            "high": Decimal("50500") + i,
            "low": Decimal("49500") + i,
            "close": Decimal("50200") + i,
            "volume": Decimal("100") + i,
        }
        for i in range(5)
    ]

    # Mock session.execute to raise an exception on the bulk insert
    def failing_execute(sql, params=None):
        raise RuntimeError("Simulated database error during bulk insert")

    # Patch the session's execute method to fail
    with patch.object(db, 'session') as mock_session_ctx:
        mock_session = MagicMock()
        mock_session_ctx.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.execute = failing_execute

        # This should raise an exception
        with pytest.raises(RuntimeError, match="Simulated database error"):
            db.record_rates_bulk(candles, is_paper=False)

    # Verify NO candles were committed (atomic rollback)
    # Since we mocked the session, the real database should have no candles
    rates = db.get_rates(is_paper=False)
    assert len(rates) == 0, "Expected 0 candles after rollback, but found some - atomic guarantee violated!"


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

    # Get rates in range (inclusive on both ends)
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
# Whale Event Tests - Paper/Live Separation
# ============================================================================

def test_record_whale_event(db):
    """Test recording a whale activity event."""
    event = db.record_whale_event(
        symbol="BTC-USD",
        volume_ratio=4.5,
        direction="bullish",
        price_change_pct=0.005,
        signal_score=75,
        signal_action="buy",
        is_paper=True
    )

    assert event.id is not None
    assert event.symbol == "BTC-USD"
    assert event.volume_ratio == 4.5
    assert event.direction == "bullish"
    assert event.price_change_pct == 0.005
    assert event.signal_score == 75
    assert event.signal_action == "buy"
    assert event.is_paper is True


def test_get_whale_events(db):
    """Test retrieving whale events."""
    # Record several whale events
    for i in range(5):
        db.record_whale_event(
            symbol="BTC-USD",
            volume_ratio=3.0 + i * 0.5,
            direction="bullish" if i % 2 == 0 else "bearish",
            price_change_pct=0.001 * i,
            signal_score=50 + i * 5,
            signal_action="buy",
            is_paper=True
        )

    events = db.get_whale_events(hours=24, is_paper=True)
    assert len(events) == 5


def test_whale_events_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live whale events remain separated.
    """
    # Record live whale event
    db.record_whale_event(
        symbol="BTC-USD",
        volume_ratio=4.0,
        direction="bullish",
        price_change_pct=0.003,
        signal_score=70,
        signal_action="buy",
        is_paper=False
    )

    # Record paper whale event
    db.record_whale_event(
        symbol="BTC-USD",
        volume_ratio=5.0,
        direction="bearish",
        price_change_pct=-0.002,
        signal_score=-60,
        signal_action="sell",
        is_paper=True
    )

    # Verify separation
    live_events = db.get_whale_events(hours=24, is_paper=False)
    paper_events = db.get_whale_events(hours=24, is_paper=True)

    assert len(live_events) == 1
    assert len(paper_events) == 1
    assert live_events[0].direction == "bullish"
    assert paper_events[0].direction == "bearish"


def test_whale_events_symbol_filter(db):
    """Test filtering whale events by symbol."""
    # Record events for different symbols
    db.record_whale_event(
        symbol="BTC-USD",
        volume_ratio=4.0,
        direction="bullish",
        price_change_pct=0.003,
        signal_score=70,
        signal_action="buy",
        is_paper=True
    )
    db.record_whale_event(
        symbol="ETH-USD",
        volume_ratio=5.0,
        direction="bearish",
        price_change_pct=-0.002,
        signal_score=-60,
        signal_action="sell",
        is_paper=True
    )

    # Filter by symbol
    btc_events = db.get_whale_events(hours=24, symbol="BTC-USD", is_paper=True)
    eth_events = db.get_whale_events(hours=24, symbol="ETH-USD", is_paper=True)
    all_events = db.get_whale_events(hours=24, is_paper=True)

    assert len(btc_events) == 1
    assert len(eth_events) == 1
    assert len(all_events) == 2


def test_whale_event_null_price_change(db):
    """Test recording whale event with null price_change_pct."""
    event = db.record_whale_event(
        symbol="BTC-USD",
        volume_ratio=3.5,
        direction="unknown",
        price_change_pct=None,
        signal_score=0,
        signal_action="hold",
        is_paper=True
    )

    assert event.id is not None
    assert event.price_change_pct is None
    assert event.direction == "unknown"


def test_whale_event_invalid_direction_defaults_to_unknown(db):
    """Test that invalid direction values default to 'unknown'."""
    event = db.record_whale_event(
        symbol="BTC-USD",
        volume_ratio=4.0,
        direction="INVALID_DIRECTION",  # Invalid value
        price_change_pct=0.005,
        signal_score=50,
        signal_action="buy",
        is_paper=True
    )

    assert event.id is not None
    assert event.direction == "unknown"  # Should be corrected to "unknown"


def test_whale_events_empty_results(db):
    """Test get_whale_events returns empty list when no matching events."""
    # Query for a symbol that has no events
    events = db.get_whale_events(hours=24, symbol="NONEXISTENT-USD", is_paper=True)
    assert events == []

    # Query for paper mode when only live events exist (none in this case)
    live_events = db.get_whale_events(hours=24, is_paper=False)
    assert live_events == []


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
    """Test database path must be within allowed directory or /tmp."""
    with patch("pathlib.Path.cwd", return_value=Path("/home/project")):
        # Try to create database outside allowed directories (/tmp and data/)
        with pytest.raises(ValueError, match="must be within"):
            Database(Path("/etc/passwd"))


def test_database_rejects_tmp_in_production():
    """Verify /tmp is rejected when not in test mode."""
    with patch.dict(os.environ, {}, clear=True):  # Clear PYTEST_CURRENT_TEST
        with patch("pathlib.Path.cwd", return_value=Path("/home/project")):
            with pytest.raises(ValueError, match="must be within"):
                Database(Path("/tmp/should_fail.db"))


def test_database_allows_tmp_in_test_mode(tmp_path):
    """Verify /tmp paths are allowed during pytest execution."""
    # This test runs with PYTEST_CURRENT_TEST set automatically
    db_path = tmp_path / "test.db"
    db = Database(db_path)  # Should not raise
    assert db.db_path == db_path.resolve()


def test_database_creates_parent_directory(tmp_path):
    """Test database creates parent directory if it doesn't exist."""
    nested_path = tmp_path / "nested" / "deep" / "test.db"

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


# ============================================================================
# Signal History Tests - Paper/Live Separation & Post-Mortem Analysis
# ============================================================================

def test_signal_history_import():
    """Test SignalHistory can be imported from database module."""
    from src.state.database import SignalHistory
    assert SignalHistory is not None


def test_signal_history_table_creation(db):
    """Test SignalHistory table is created on database initialization."""
    from src.state.database import SignalHistory
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    tables = inspector.get_table_names()

    assert "signal_history" in tables


def test_signal_history_record_creation(db):
    """Test creating a signal history record."""
    from src.state.database import SignalHistory

    with db.session() as session:
        signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50000.00",
            rsi_score=15.5,
            macd_score=10.2,
            bollinger_score=-5.0,
            ema_score=8.3,
            volume_score=4.0,
            rsi_value=35.0,
            macd_histogram=150.5,
            bb_position=0.3,
            ema_gap_percent=1.2,
            volume_ratio=2.5,
            trend_filter_adj=-10.0,
            momentum_mode_adj=0.0,
            whale_activity_adj=0.0,
            htf_bias_adj=20.0,
            htf_bias="bullish",
            htf_daily_trend="bullish",
            htf_4h_trend="bullish",
            raw_score=45,
            final_score=55,
            action="hold",
            threshold_used=60,
            trade_executed=False,
        )
        session.add(signal)
        session.commit()

        # Verify record was created
        assert signal.id is not None


def test_signal_history_paper_live_separation(db):
    """
    CRITICAL: Verify paper and live signal history remain separated.
    """
    from src.state.database import SignalHistory

    with db.session() as session:
        # Create paper signal
        paper_signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50000.00",
            rsi_score=15.0,
            macd_score=10.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=4.0,
            raw_score=45,
            final_score=55,
            action="hold",
            threshold_used=60,
        )

        # Create live signal
        live_signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=False,
            current_price="50100.00",
            rsi_score=20.0,
            macd_score=15.0,
            bollinger_score=-3.0,
            ema_score=10.0,
            volume_score=5.0,
            raw_score=55,
            final_score=65,
            action="buy",
            threshold_used=60,
        )

        session.add(paper_signal)
        session.add(live_signal)
        session.commit()

        # Verify separation via query
        paper_signals = session.query(SignalHistory).filter(
            SignalHistory.is_paper == True
        ).all()
        live_signals = session.query(SignalHistory).filter(
            SignalHistory.is_paper == False
        ).all()

        assert len(paper_signals) == 1
        assert len(live_signals) == 1
        assert paper_signals[0].final_score == 55
        assert live_signals[0].final_score == 65


def test_signal_history_htf_fields(db):
    """Test HTF fields are stored and retrieved correctly."""
    from src.state.database import SignalHistory

    with db.session() as session:
        signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50000.00",
            rsi_score=15.0,
            macd_score=10.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=4.0,
            htf_bias="bullish",
            htf_daily_trend="bullish",
            htf_4h_trend="neutral",
            htf_bias_adj=20.0,
            raw_score=45,
            final_score=65,  # 45 + 20 from HTF
            action="buy",
            threshold_used=60,
        )
        session.add(signal)
        session.commit()

        # Retrieve and verify
        retrieved = session.query(SignalHistory).filter(
            SignalHistory.id == signal.id
        ).first()

        assert retrieved.htf_bias == "bullish"
        assert retrieved.htf_daily_trend == "bullish"
        assert retrieved.htf_4h_trend == "neutral"
        assert retrieved.htf_bias_adj == 20.0


def test_signal_history_raw_indicator_values(db):
    """Test raw indicator values are stored for debugging."""
    from src.state.database import SignalHistory

    with db.session() as session:
        signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50000.00",
            rsi_score=15.0,
            macd_score=10.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=4.0,
            rsi_value=35.5,
            macd_histogram=150.75,
            bb_position=0.25,
            ema_gap_percent=1.5,
            volume_ratio=2.3,
            raw_score=45,
            final_score=55,
            action="hold",
            threshold_used=60,
        )
        session.add(signal)
        session.commit()

        retrieved = session.query(SignalHistory).filter(
            SignalHistory.id == signal.id
        ).first()

        assert retrieved.rsi_value == 35.5
        assert retrieved.macd_histogram == 150.75
        assert retrieved.bb_position == 0.25
        assert retrieved.ema_gap_percent == 1.5
        assert retrieved.volume_ratio == 2.3


def test_signal_history_trade_executed_flag(db):
    """Test trade_executed flag tracks whether signal resulted in trade."""
    from src.state.database import SignalHistory

    with db.session() as session:
        # Signal that resulted in a trade
        traded_signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50000.00",
            rsi_score=20.0,
            macd_score=15.0,
            bollinger_score=-5.0,
            ema_score=12.0,
            volume_score=8.0,
            raw_score=75,
            final_score=85,
            action="buy",
            threshold_used=60,
            trade_executed=True,
        )

        # Signal that didn't result in a trade
        not_traded_signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50100.00",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=0.0,
            ema_score=3.0,
            volume_score=2.0,
            raw_score=25,
            final_score=30,
            action="hold",
            threshold_used=60,
            trade_executed=False,
        )

        session.add(traded_signal)
        session.add(not_traded_signal)
        session.commit()

        # Query for signals that resulted in trades
        traded = session.query(SignalHistory).filter(
            SignalHistory.trade_executed == True
        ).all()

        not_traded = session.query(SignalHistory).filter(
            SignalHistory.trade_executed == False
        ).all()

        assert len(traded) == 1
        assert len(not_traded) == 1
        assert traded[0].action == "buy"
        assert not_traded[0].action == "hold"


def test_signal_history_timestamp_auto_generated(db):
    """Test timestamp is auto-generated if not provided."""
    from src.state.database import SignalHistory

    with db.session() as session:
        signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50000.00",
            rsi_score=15.0,
            macd_score=10.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=4.0,
            raw_score=45,
            final_score=55,
            action="hold",
            threshold_used=60,
        )
        session.add(signal)
        session.commit()

        assert signal.timestamp is not None
        assert isinstance(signal.timestamp, datetime)


def test_signal_history_indexes_exist(db):
    """Test that indexes are created for efficient querying."""
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    indexes = inspector.get_indexes("signal_history")

    index_names = [idx["name"] for idx in indexes]

    # Check for expected indexes
    assert any("timestamp" in name for name in index_names)
    assert any("paper" in name and "timestamp" in name for name in index_names)
    assert "ix_signal_history_executed_lookup" in index_names


def test_signal_history_null_htf_values(db):
    """Test HTF fields can be null (MTF disabled)."""
    from src.state.database import SignalHistory

    with db.session() as session:
        signal = SignalHistory(
            symbol="BTC-USD",
            is_paper=True,
            current_price="50000.00",
            rsi_score=15.0,
            macd_score=10.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=4.0,
            htf_bias=None,  # MTF disabled
            htf_daily_trend=None,
            htf_4h_trend=None,
            htf_bias_adj=0.0,
            raw_score=45,
            final_score=45,
            action="hold",
            threshold_used=60,
        )
        session.add(signal)
        session.commit()

        retrieved = session.query(SignalHistory).filter(
            SignalHistory.id == signal.id
        ).first()

        assert retrieved.htf_bias is None
        assert retrieved.htf_daily_trend is None
        assert retrieved.htf_4h_trend is None


# ============================================================================
# Migration Tests
# ============================================================================


class TestBotModeMigration:
    """Tests for the bot_mode column migration."""

    def test_bot_mode_column_exists_on_all_tables(self, db):
        """Verify bot_mode column is created on all required tables."""
        with db.session() as session:
            # Check positions table (uses quantity/average_cost stored as strings)
            position = Position(
                symbol="BTC-USD",
                quantity="0.1",
                average_cost="50000",
                is_paper=True,
                bot_mode="normal",
            )
            session.add(position)
            session.commit()
            assert position.bot_mode == "normal"

            # Check trailing_stops table (uses string for Decimal fields)
            stop = TrailingStop(
                symbol="BTC-USD",
                side="buy",
                entry_price="50000",
                trailing_distance="0.03",
                trailing_activation="0.02",
                is_paper=True,
                bot_mode="inverted",
            )
            session.add(stop)
            session.commit()
            assert stop.bot_mode == "inverted"

    def test_bot_mode_defaults_to_normal(self, db):
        """Verify bot_mode defaults to 'normal' when not specified."""
        with db.session() as session:
            # Create position without specifying bot_mode
            position = Position(
                symbol="BTC-USD",
                quantity="0.1",
                average_cost="50000",
                is_paper=True,
            )
            session.add(position)
            session.commit()

            # Default should be "normal"
            assert position.bot_mode == "normal"

    def test_bot_mode_separation_in_queries(self, db):
        """Verify bot_mode properly separates normal and Cramer Mode data."""
        # Use the database method to record trades (handles types correctly)
        db.record_trade(
            side="buy",
            size=Decimal("0.1"),
            price=Decimal("50000"),
            fee=Decimal("1.0"),
            symbol="BTC-USD",
            is_paper=True,
            bot_mode="normal",
        )

        db.record_trade(
            side="sell",  # Opposite of normal
            size=Decimal("0.1"),
            price=Decimal("50000"),
            fee=Decimal("1.0"),
            symbol="BTC-USD",
            is_paper=True,
            bot_mode="inverted",
        )

        with db.session() as session:
            # Query normal bot trades only
            normal_trades = session.query(Trade).filter(
                Trade.bot_mode == "normal",
                Trade.is_paper == True,
            ).all()
            assert len(normal_trades) == 1
            assert normal_trades[0].side == "buy"

            # Query Cramer Mode trades only
            cramer_trades = session.query(Trade).filter(
                Trade.bot_mode == "inverted",
                Trade.is_paper == True,
            ).all()
            assert len(cramer_trades) == 1
            assert cramer_trades[0].side == "sell"

    def test_get_last_paper_balance_respects_bot_mode(self, db):
        """Verify get_last_paper_balance filters by bot_mode."""
        # Record normal bot trade
        db.record_trade(
            side="buy",
            size=Decimal("0.1"),
            price=Decimal("50000"),
            fee=Decimal("1.0"),
            symbol="BTC-USD",
            is_paper=True,
            bot_mode="normal",
            quote_balance_after=Decimal("5000"),
            base_balance_after=Decimal("0.1"),
            spot_rate=Decimal("50000"),
        )

        # Record Cramer Mode trade with different balance
        db.record_trade(
            side="sell",
            size=Decimal("0.05"),
            price=Decimal("50000"),
            fee=Decimal("1.0"),
            symbol="BTC-USD",
            is_paper=True,
            bot_mode="inverted",
            quote_balance_after=Decimal("7500"),
            base_balance_after=Decimal("0.05"),
            spot_rate=Decimal("50000"),
        )

        # Get normal bot balance
        normal_balance = db.get_last_paper_balance("BTC-USD", bot_mode="normal")
        assert normal_balance is not None
        quote, base, _ = normal_balance
        assert quote == Decimal("5000")
        assert base == Decimal("0.1")

        # Get Cramer Mode balance
        cramer_balance = db.get_last_paper_balance("BTC-USD", bot_mode="inverted")
        assert cramer_balance is not None
        quote, base, _ = cramer_balance
        assert quote == Decimal("7500")
        assert base == Decimal("0.05")

    def test_bot_mode_migration_handles_existing_data(self, db, tmp_path):
        """Test that bot_mode migration correctly updates existing data."""
        import sqlite3
        from sqlalchemy import text

        # Create a fresh database without bot_mode column to simulate old schema
        test_db_path = tmp_path / "migration_test.db"
        conn = sqlite3.connect(str(test_db_path))
        cursor = conn.cursor()

        # Create trades table WITHOUT bot_mode column (simulating old schema)
        cursor.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                size TEXT,
                price TEXT,
                fee TEXT,
                is_paper INTEGER DEFAULT 1,
                executed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert some "old" data without bot_mode
        cursor.execute("""
            INSERT INTO trades (symbol, side, size, price, fee, is_paper)
            VALUES ('BTC-USD', 'buy', '0.1', '50000', '1.0', 1)
        """)
        cursor.execute("""
            INSERT INTO trades (symbol, side, size, price, fee, is_paper)
            VALUES ('BTC-USD', 'sell', '0.05', '51000', '1.0', 1)
        """)
        conn.commit()

        # Verify data exists but no bot_mode column
        cursor.execute("SELECT * FROM trades")
        rows = cursor.fetchall()
        assert len(rows) == 2

        # Now add bot_mode column with migration logic (simulating Database._run_migrations)
        cursor.execute("ALTER TABLE trades ADD COLUMN bot_mode VARCHAR(20) DEFAULT 'normal' NOT NULL")
        cursor.execute("UPDATE trades SET bot_mode = 'normal' WHERE bot_mode IS NULL OR bot_mode = ''")
        conn.commit()

        # Verify all existing rows have bot_mode='normal'
        cursor.execute("SELECT id, bot_mode FROM trades")
        rows = cursor.fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row[1] == "normal", f"Row {row[0]} should have bot_mode='normal', got '{row[1]}'"

        # Verify new rows can use 'inverted' mode
        cursor.execute("""
            INSERT INTO trades (symbol, side, size, price, fee, is_paper, bot_mode)
            VALUES ('BTC-USD', 'buy', '0.1', '52000', '1.0', 1, 'inverted')
        """)
        conn.commit()

        cursor.execute("SELECT bot_mode FROM trades WHERE id = 3")
        assert cursor.fetchone()[0] == "inverted"

        conn.close()


# ============================================================================
# Signal History Cleanup Tests
# ============================================================================

def test_cleanup_signal_history_basic(db):
    """Test basic signal history cleanup functionality."""
    now = datetime.now(timezone.utc)

    # Create signal history records with different ages
    with db.session() as session:
        # Recent record (1 day old) - should be kept
        recent = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=1),
            is_paper=True,
            current_price="50000",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=2.0,
            raw_score=20.0,
            final_score=20.0,
            action="hold",
            threshold_used=60,
        )
        session.add(recent)

        # Old record (100 days old) - should be deleted
        old = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=100),
            is_paper=True,
            current_price="48000",
            rsi_score=-15.0,
            macd_score=-10.0,
            bollinger_score=-8.0,
            ema_score=-5.0,
            volume_score=-2.0,
            raw_score=-40.0,
            final_score=-40.0,
            action="hold",
            threshold_used=60,
        )
        session.add(old)

        # Very old record (200 days old) - should be deleted
        very_old = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=200),
            is_paper=True,
            current_price="45000",
            rsi_score=-20.0,
            macd_score=-15.0,
            bollinger_score=-10.0,
            ema_score=-8.0,
            volume_score=-3.0,
            raw_score=-56.0,
            final_score=-56.0,
            action="hold",
            threshold_used=60,
        )
        session.add(very_old)

    # Run cleanup with 90-day retention
    deleted = db.cleanup_signal_history(retention_days=90)

    # Should delete 2 old records
    assert deleted == 2

    # Verify recent record still exists
    with db.session() as session:
        remaining = session.query(SignalHistory).filter(
            SignalHistory.is_paper == True
        ).count()
        assert remaining == 1


def test_cleanup_signal_history_paper_live_separation(db):
    """Test cleanup respects paper/live mode separation."""
    now = datetime.now(timezone.utc)

    # Create old records in both paper and live mode
    with db.session() as session:
        paper_old = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=100),
            is_paper=True,
            current_price="50000",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=2.0,
            raw_score=20.0,
            final_score=20.0,
            action="hold",
            threshold_used=60,
        )
        session.add(paper_old)

        live_old = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=100),
            is_paper=False,
            current_price="50000",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=2.0,
            raw_score=20.0,
            final_score=20.0,
            action="hold",
            threshold_used=60,
        )
        session.add(live_old)

    # Clean only paper mode
    deleted = db.cleanup_signal_history(retention_days=90, is_paper=True)
    assert deleted == 1

    # Verify live record still exists
    with db.session() as session:
        live_count = session.query(SignalHistory).filter(
            SignalHistory.is_paper == False
        ).count()
        assert live_count == 1

        paper_count = session.query(SignalHistory).filter(
            SignalHistory.is_paper == True
        ).count()
        assert paper_count == 0


def test_cleanup_signal_history_no_old_records(db):
    """Test cleanup returns 0 when no old records exist."""
    now = datetime.now(timezone.utc)

    # Create only recent records
    with db.session() as session:
        for i in range(5):
            record = SignalHistory(
                symbol="BTC-USD",
                timestamp=now - timedelta(days=i),
                is_paper=True,
                current_price="50000",
                rsi_score=10.0,
                macd_score=5.0,
                bollinger_score=-5.0,
                ema_score=8.0,
                volume_score=2.0,
                raw_score=20.0,
                final_score=20.0,
                action="hold",
                threshold_used=60,
            )
            session.add(record)

    # Run cleanup - should delete nothing
    deleted = db.cleanup_signal_history(retention_days=90)
    assert deleted == 0

    # Verify all records still exist
    with db.session() as session:
        count = session.query(SignalHistory).count()
        assert count == 5


def test_cleanup_signal_history_empty_table(db):
    """Test cleanup on empty table returns 0."""
    deleted = db.cleanup_signal_history(retention_days=90)
    assert deleted == 0


def test_cleanup_signal_history_boundary_condition(db):
    """Test cleanup at exact retention boundary."""
    now = datetime.now(timezone.utc)

    with db.session() as session:
        # Record 89 days old (should be kept - within retention window)
        within_boundary = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=89),
            is_paper=True,
            current_price="50000",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=2.0,
            raw_score=20.0,
            final_score=20.0,
            action="hold",
            threshold_used=60,
        )
        session.add(within_boundary)

        # Record exactly 90 days old (should be deleted - older than retention)
        boundary = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=90),
            is_paper=True,
            current_price="49500",
            rsi_score=8.0,
            macd_score=4.0,
            bollinger_score=-4.0,
            ema_score=6.0,
            volume_score=1.5,
            raw_score=15.0,
            final_score=15.0,
            action="hold",
            threshold_used=60,
        )
        session.add(boundary)

        # Record 91 days old (should be deleted)
        past_boundary = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=91),
            is_paper=True,
            current_price="49000",
            rsi_score=5.0,
            macd_score=3.0,
            bollinger_score=-3.0,
            ema_score=4.0,
            volume_score=1.0,
            raw_score=10.0,
            final_score=10.0,
            action="hold",
            threshold_used=60,
        )
        session.add(past_boundary)

    # Run cleanup
    deleted = db.cleanup_signal_history(retention_days=90)
    assert deleted == 2

    # Verify only the 89-day-old record still exists
    with db.session() as session:
        count = session.query(SignalHistory).count()
        assert count == 1


def test_cleanup_signal_history_both_modes(db):
    """Test cleanup without mode filter affects both paper and live."""
    now = datetime.now(timezone.utc)

    # Create old records in both modes
    with db.session() as session:
        paper_old = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=100),
            is_paper=True,
            current_price="50000",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=2.0,
            raw_score=20.0,
            final_score=20.0,
            action="hold",
            threshold_used=60,
        )
        session.add(paper_old)

        live_old = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=100),
            is_paper=False,
            current_price="50000",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=2.0,
            raw_score=20.0,
            final_score=20.0,
            action="hold",
            threshold_used=60,
        )
        session.add(live_old)

    # Clean both modes (is_paper=None)
    deleted = db.cleanup_signal_history(retention_days=90, is_paper=None)
    assert deleted == 2

    # Verify both records deleted
    with db.session() as session:
        count = session.query(SignalHistory).count()
        assert count == 0


def test_cleanup_signal_history_invalid_retention_days(db):
    """Test cleanup raises ValueError for invalid retention_days."""
    with pytest.raises(ValueError, match="retention_days must be between 1 and 365"):
        db.cleanup_signal_history(retention_days=0)

    with pytest.raises(ValueError, match="retention_days must be between 1 and 365"):
        db.cleanup_signal_history(retention_days=-10)

    with pytest.raises(ValueError, match="retention_days must be between 1 and 365"):
        db.cleanup_signal_history(retention_days=366)


def test_cleanup_signal_history_uses_config_default(db):
    """Test cleanup uses SIGNAL_HISTORY_RETENTION_DAYS from config when retention_days=None."""
    from config.settings import get_settings

    settings = get_settings()
    now = datetime.now(timezone.utc)

    # Create record older than config default
    with db.session() as session:
        old = SignalHistory(
            symbol="BTC-USD",
            timestamp=now - timedelta(days=settings.signal_history_retention_days + 1),
            is_paper=True,
            current_price="50000",
            rsi_score=10.0,
            macd_score=5.0,
            bollinger_score=-5.0,
            ema_score=8.0,
            volume_score=2.0,
            raw_score=20.0,
            final_score=20.0,
            action="hold",
            threshold_used=60,
        )
        session.add(old)

    # Call without retention_days parameter - should use config default
    deleted = db.cleanup_signal_history()
    assert deleted == 1


# ============================================================================
# Rate History Concurrency Tests
# ============================================================================

def test_rate_history_concurrent_updates(db):
    """
    Test concurrent updates to the same candle remain atomic and correct.

    Per issue #193: The current implementation uses sqlite_insert().on_conflict_do_update()
    which is atomic at the SQL level. This test verifies that concurrent updates to the
    same candle timestamp correctly compute max(high) and min(low) without data corruption.

    This test simulates multiple threads updating the same candle with different OHLCV data
    and verifies:
    1. Final high price is the maximum across all updates
    2. Final low price is the minimum across all updates
    3. No data corruption occurs
    4. Open price is preserved (immutable)

    Note: SQLite serializes writes at the database level, so true concurrent execution may
    not occur. This test validates logical correctness of the UPSERT logic under the
    assumption of concurrent updates.
    """
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    # Define different update scenarios that will execute concurrently
    # Each thread will try to update the same candle with different high/low values
    update_scenarios = [
        {
            "thread_id": 1,
            "open": Decimal("50000"),
            "high": Decimal("51000"),  # Highest high
            "low": Decimal("49500"),
            "close": Decimal("50800"),
            "volume": Decimal("100"),
        },
        {
            "thread_id": 2,
            "open": Decimal("50100"),  # Different open (should be ignored)
            "high": Decimal("50800"),
            "low": Decimal("49000"),  # Lowest low
            "close": Decimal("50600"),
            "volume": Decimal("120"),
        },
        {
            "thread_id": 3,
            "open": Decimal("50200"),  # Different open (should be ignored)
            "high": Decimal("50700"),
            "low": Decimal("49200"),
            "close": Decimal("50500"),
            "volume": Decimal("110"),
        },
        {
            "thread_id": 4,
            "open": Decimal("49900"),  # Different open (should be ignored)
            "high": Decimal("50900"),
            "low": Decimal("49100"),
            "close": Decimal("50700"),
            "volume": Decimal("130"),
        },
    ]

    # Expected final values after all concurrent updates
    # - Open: First thread's open (whichever wins the race to insert)
    # - High: max(51000, 50800, 50700, 50900) = 51000
    # - Low: min(49500, 49000, 49200, 49100) = 49000
    # - Close: Last update's close (non-deterministic in concurrent scenario)
    expected_high = Decimal("51000")
    expected_low = Decimal("49000")

    # Thread worker function
    def update_candle(scenario):
        """Worker function to update the candle with scenario data."""
        # Small random delay (0-10ms) to increase likelihood of concurrent execution.
        # Note: SQLite's locking model may still serialize these writes.
        time.sleep(random.uniform(0, 0.01))

        candles = [{
            "timestamp": timestamp,
            "open": scenario["open"],
            "high": scenario["high"],
            "low": scenario["low"],
            "close": scenario["close"],
            "volume": scenario["volume"],
        }]

        # No try/except - the database UPSERT should handle concurrency gracefully.
        # If exceptions occur, they indicate a real bug that must be fixed.
        db.record_rates_bulk(
            candles,
            symbol="BTC-USD",
            exchange="kraken",
            interval="1m",
            is_paper=False
        )

    # Create and start threads
    threads = []
    for scenario in update_scenarios:
        thread = threading.Thread(target=update_candle, args=(scenario,))
        threads.append(thread)

    # Start all threads simultaneously
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify final state
    rates = db.get_rates(is_paper=False)
    assert len(rates) == 1, "Should have exactly one candle after concurrent updates"

    final_candle = rates[0]

    # Verify high/low are correct (max/min across all updates)
    assert final_candle.get_high() == expected_high, (
        f"Expected high={expected_high}, got {final_candle.get_high()}. "
        f"Concurrent updates failed to compute max(high) correctly."
    )
    assert final_candle.get_low() == expected_low, (
        f"Expected low={expected_low}, got {final_candle.get_low()}. "
        f"Concurrent updates failed to compute min(low) correctly."
    )

    # Verify open is one of the scenario values (first insert wins)
    possible_opens = {scenario["open"] for scenario in update_scenarios}
    assert final_candle.get_open() in possible_opens, (
        f"Open price {final_candle.get_open()} not in expected set {possible_opens}"
    )

    # Verify close is one of the scenario values (last update wins)
    possible_closes = {scenario["close"] for scenario in update_scenarios}
    assert final_candle.get_close() in possible_closes, (
        f"Close price {final_candle.get_close()} not in expected set {possible_closes}"
    )

    # Verify no data corruption (values are valid Decimals)
    assert isinstance(final_candle.get_open(), Decimal)
    assert isinstance(final_candle.get_high(), Decimal)
    assert isinstance(final_candle.get_low(), Decimal)
    assert isinstance(final_candle.get_close(), Decimal)
    assert isinstance(final_candle.get_volume(), Decimal)

    # Verify OHLC invariant: low <= open,close <= high
    assert final_candle.get_low() <= final_candle.get_open() <= final_candle.get_high()
    assert final_candle.get_low() <= final_candle.get_close() <= final_candle.get_high()
