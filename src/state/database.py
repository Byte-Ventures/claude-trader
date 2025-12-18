"""
SQLite database for state persistence and trade history.

Tables:
- positions: Current and historical positions
- orders: All orders placed
- trades: Executed trades (fills)
- daily_stats: Daily trading statistics
- system_state: Key-value store for recovery state
- rate_history: Historical OHLCV price data for analysis and replay
- whale_events: Historical record of whale activity detections
- signal_history: Historical signal calculations for post-mortem analysis
"""

import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    inspect,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

import structlog

logger = structlog.get_logger(__name__)

Base = declarative_base()


class BotMode(str, Enum):
    """Bot trading mode for Cramer Mode feature.

    NORMAL: Standard trading bot
    INVERTED: Cramer Mode - executes opposite trades for comparison
    """

    NORMAL = "normal"
    INVERTED = "inverted"


class Position(Base):
    """Current and historical positions."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    quantity = Column(String(50), nullable=False)  # Stored as string for Decimal precision
    average_cost = Column(String(50), nullable=False)
    unrealized_pnl = Column(String(50), default="0")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    is_current = Column(Boolean, default=True)
    is_paper = Column(Boolean, default=False)
    bot_mode = Column(String(20), default="normal", nullable=False)  # "normal" or "inverted"

    def get_quantity(self) -> Decimal:
        return Decimal(self.quantity)

    def get_average_cost(self) -> Decimal:
        return Decimal(self.average_cost)

    def get_unrealized_pnl(self) -> Decimal:
        return Decimal(self.unrealized_pnl)


class Order(Base):
    """All orders placed."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange_order_id = Column(String(100), unique=True, nullable=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    side = Column(String(10), nullable=False)  # "buy" or "sell"
    order_type = Column(String(20), nullable=False, default="market")
    size = Column(String(50), nullable=False)
    price = Column(String(50), nullable=True)  # Null for market orders
    status = Column(String(20), nullable=False, default="pending")
    filled_size = Column(String(50), default="0")
    filled_price = Column(String(50), nullable=True)
    fee = Column(String(50), default="0")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    error_message = Column(Text, nullable=True)
    is_paper = Column(Boolean, default=False)


class Trade(Base):
    """Executed trades (fills)."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, nullable=True)  # FK to orders
    exchange_trade_id = Column(String(100), nullable=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    side = Column(String(10), nullable=False)
    size = Column(String(50), nullable=False)
    price = Column(String(50), nullable=False)
    fee = Column(String(50), default="0")
    realized_pnl = Column(String(50), default="0")
    executed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_paper = Column(Boolean, default=False)
    bot_mode = Column(String(20), default="normal", nullable=False)  # "normal" or "inverted"
    # Balance snapshot after trade
    quote_balance_after = Column(String(50), nullable=True)
    base_balance_after = Column(String(50), nullable=True)
    spot_rate = Column(String(50), nullable=True)  # BTC rate at time of trade


class DailyStats(Base):
    """Daily trading statistics."""

    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    is_paper = Column(Boolean, default=False, nullable=False)
    bot_mode = Column(String(20), default="normal", nullable=False)  # "normal" or "inverted"
    starting_balance = Column(String(50), nullable=False)  # In quote currency
    ending_balance = Column(String(50), nullable=True)  # In quote currency
    starting_price = Column(String(50), nullable=True)  # BTC price at start of day
    ending_price = Column(String(50), nullable=True)  # BTC price at end of day
    realized_pnl = Column(String(50), default="0")
    unrealized_pnl = Column(String(50), default="0")
    total_trades = Column(Integer, default=0)
    total_volume = Column(String(50), default="0")  # In quote currency
    max_drawdown_percent = Column(String(50), default="0")


class SystemState(Base):
    """Key-value store for system state and recovery."""

    __tablename__ = "system_state"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)  # JSON-encoded
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Notification(Base):
    """Notification history for dashboard display."""

    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(50), nullable=False)  # trade, error, circuit_breaker, etc.
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_paper = Column(Boolean, default=False)


class RegimeHistory(Base):
    """Historical record of market regime changes."""

    __tablename__ = "regime_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    regime_name = Column(String(30), nullable=False)  # risk_on, neutral, risk_off, etc.
    threshold_adjustment = Column(Integer, nullable=False)
    position_multiplier = Column(String(10), nullable=False)
    sentiment_value = Column(Integer, nullable=True)  # Fear & Greed value
    sentiment_category = Column(String(30), nullable=True)
    volatility_level = Column(String(20), nullable=True)
    trend_direction = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_paper = Column(Boolean, default=False)


class WeightProfileHistory(Base):
    """Historical record of AI weight profile selections."""

    __tablename__ = "weight_profile_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_name = Column(String(20), nullable=False)  # trending, ranging, volatile, default
    confidence = Column(String(10), nullable=False)  # AI confidence 0.0-1.0
    reasoning = Column(Text, nullable=True)  # AI reasoning
    market_context = Column(Text, nullable=True)  # JSON with market data
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_paper = Column(Boolean, default=False)


class TrailingStop(Base):
    """Trailing stop tracking for open positions."""

    __tablename__ = "trailing_stops"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    side = Column(String(10), nullable=False)  # "buy" or "sell"
    entry_price = Column(String(50), nullable=False)
    trailing_stop = Column(String(50), nullable=True)  # Current stop level
    trailing_activation = Column(String(50), nullable=True)  # Price where trailing activates
    trailing_distance = Column(String(50), nullable=True)  # ATR-based distance
    hard_stop = Column(String(50), nullable=True)  # Hard stop: emergency exit, moves to entry at break-even
    take_profit_price = Column(String(50), nullable=True)  # Target profit price for automatic exit
    breakeven_triggered = Column(Boolean, default=False)  # True when stop moved to break-even
    is_active = Column(Boolean, default=False)
    is_paper = Column(Boolean, default=False)
    bot_mode = Column(String(20), default="normal", nullable=False)  # "normal" or "inverted"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def get_entry_price(self) -> Decimal:
        return Decimal(self.entry_price)

    def get_trailing_stop(self) -> Optional[Decimal]:
        return Decimal(self.trailing_stop) if self.trailing_stop else None

    def get_trailing_activation(self) -> Optional[Decimal]:
        return Decimal(self.trailing_activation) if self.trailing_activation else None

    def get_trailing_distance(self) -> Optional[Decimal]:
        return Decimal(self.trailing_distance) if self.trailing_distance else None

    def get_hard_stop(self) -> Optional[Decimal]:
        return Decimal(self.hard_stop) if self.hard_stop else None

    def get_take_profit_price(self) -> Optional[Decimal]:
        """Return take profit target price, or None if not set."""
        return Decimal(self.take_profit_price) if self.take_profit_price else None

    def is_breakeven_active(self) -> bool:
        """Return True if break-even protection has been triggered."""
        return self.breakeven_triggered or False


class RateHistory(Base):
    """Historical OHLCV price data for analysis and replay testing.

    Stores candlestick data from exchanges for:
    - Backtesting and strategy replay
    - Historical analysis and reporting
    - Training data for ML models

    Note: All timestamps are stored in UTC.
    """

    __tablename__ = "rate_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    exchange = Column(String(20), nullable=False, default="kraken")
    interval = Column(String(10), nullable=False, default="1m")  # 1m, 5m, 15m, 1h, 1d
    timestamp = Column(DateTime, nullable=False)  # Candle start time (UTC)
    open_price = Column(String(50), nullable=False)
    high_price = Column(String(50), nullable=False)
    low_price = Column(String(50), nullable=False)
    close_price = Column(String(50), nullable=False)
    volume = Column(String(50), nullable=False)
    is_paper = Column(Boolean, nullable=False, default=False)  # Paper vs live data
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Unique constraint: one candle per symbol/exchange/interval/timestamp/is_paper
    __table_args__ = (
        UniqueConstraint('symbol', 'exchange', 'interval', 'timestamp', 'is_paper', name='uq_rate_candle_v2'),
        Index('ix_rate_history_lookup_v2', 'symbol', 'exchange', 'interval', 'is_paper', 'timestamp'),
    )

    def get_open(self) -> Decimal:
        try:
            value = Decimal(self.open_price) if self.open_price else Decimal("0")
            if value < Decimal("0"):
                raise ValueError(f"Negative open price: {value}")
            return value
        except Exception as e:
            logger.error("decimal_conversion_failed", field="open_price", value=self.open_price, error=str(e), rate_id=self.id)
            raise ValueError(f"Invalid open_price: {self.open_price}") from e

    def get_high(self) -> Decimal:
        try:
            value = Decimal(self.high_price) if self.high_price else Decimal("0")
            if value < Decimal("0"):
                raise ValueError(f"Negative high price: {value}")
            return value
        except Exception as e:
            logger.error("decimal_conversion_failed", field="high_price", value=self.high_price, error=str(e), rate_id=self.id)
            raise ValueError(f"Invalid high_price: {self.high_price}") from e

    def get_low(self) -> Decimal:
        try:
            value = Decimal(self.low_price) if self.low_price else Decimal("0")
            if value < Decimal("0"):
                raise ValueError(f"Negative low price: {value}")
            return value
        except Exception as e:
            logger.error("decimal_conversion_failed", field="low_price", value=self.low_price, error=str(e), rate_id=self.id)
            raise ValueError(f"Invalid low_price: {self.low_price}") from e

    def get_close(self) -> Decimal:
        try:
            value = Decimal(self.close_price) if self.close_price else Decimal("0")
            if value < Decimal("0"):
                raise ValueError(f"Negative close price: {value}")
            return value
        except Exception as e:
            logger.error("decimal_conversion_failed", field="close_price", value=self.close_price, error=str(e), rate_id=self.id)
            raise ValueError(f"Invalid close_price: {self.close_price}") from e

    def get_volume(self) -> Decimal:
        try:
            value = Decimal(self.volume) if self.volume else Decimal("0")
            if value < Decimal("0"):
                raise ValueError(f"Negative volume: {value}")
            return value
        except Exception as e:
            logger.error("decimal_conversion_failed", field="volume", value=self.volume, error=str(e), rate_id=self.id)
            raise ValueError(f"Invalid volume: {self.volume}") from e


class WhaleEvent(Base):
    """Historical record of whale activity detections.

    Tracks volume spikes exceeding the whale threshold, including direction
    and signal context for post-trade analysis and pattern recognition.
    """

    __tablename__ = "whale_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))  # When whale was detected
    volume_ratio = Column(Float, nullable=False)  # e.g., 3.45
    direction = Column(String(10), nullable=False)  # bullish/bearish/neutral
    price_change_pct = Column(Float, nullable=True)  # e.g., 0.0035
    signal_score = Column(Integer, nullable=False)  # -100 to +100
    signal_action = Column(String(10), nullable=False)  # buy/sell/hold
    is_paper = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('ix_whale_events_lookup', 'symbol', 'is_paper', 'timestamp'),
    )


class SignalHistory(Base):
    """Historical signal calculations for post-mortem analysis.

    Stores every signal calculation with full indicator breakdown,
    enabling analysis of why trades were taken or missed.
    """

    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_paper = Column(Boolean, default=False)

    # Price context
    current_price = Column(String(50), nullable=False)

    # Individual indicator scores
    rsi_score = Column(Float, nullable=False)
    macd_score = Column(Float, nullable=False)
    bollinger_score = Column(Float, nullable=False)
    ema_score = Column(Float, nullable=False)
    volume_score = Column(Float, nullable=False)

    # Raw indicator values (for debugging)
    rsi_value = Column(Float, nullable=True)
    macd_histogram = Column(Float, nullable=True)
    bb_position = Column(Float, nullable=True)  # % position in bands
    ema_gap_percent = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)

    # Adjustments applied
    trend_filter_adj = Column(Float, default=0)
    momentum_mode_adj = Column(Float, default=0)
    whale_activity_adj = Column(Float, default=0)
    htf_bias_adj = Column(Float, default=0)

    # HTF context
    htf_bias = Column(String(10), nullable=True)  # combined: bullish/bearish/neutral
    htf_daily_trend = Column(String(10), nullable=True)
    htf_4h_trend = Column(String(10), nullable=True)

    # Final result
    raw_score = Column(Float, nullable=False)  # Before adjustments
    final_score = Column(Float, nullable=False)  # After all adjustments
    action = Column(String(10), nullable=False)  # buy/sell/hold
    threshold_used = Column(Integer, nullable=False)

    # Whether this signal resulted in a trade
    trade_executed = Column(Boolean, default=False)

    __table_args__ = (
        Index('ix_signal_history_timestamp', 'timestamp'),
        Index('ix_signal_history_paper_timestamp', 'is_paper', 'timestamp'),
        Index('ix_signal_history_symbol_paper_time', 'symbol', 'is_paper', 'timestamp'),
    )


class Database:
    """
    Database manager for trading state persistence.

    Handles:
    - Position tracking
    - Order history
    - Trade logging
    - Daily statistics
    - Recovery state
    """

    def __init__(self, db_path: Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file

        Raises:
            ValueError: If db_path is outside the allowed data directory
        """
        # Resolve to absolute path and validate
        db_path = db_path.resolve()
        allowed_root = (Path.cwd() / "data").resolve()

        try:
            db_path.relative_to(allowed_root)
        except ValueError:
            raise ValueError(f"Database path must be within {allowed_root}")

        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        # Create tables - we use create_all() instead of Alembic for simplicity.
        # SQLite handles new tables automatically; schema changes use _run_migrations().
        # create_all() is idempotent - creates missing tables, skips existing ones.
        Base.metadata.create_all(self.engine)

        # Run migrations for existing databases (column renames, etc.)
        self._run_migrations()

        # Verify critical tables exist (especially for upgrades from older versions)
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        logger.info(
            "database_initialized",
            path=str(db_path),
            tables=tables,
            signal_history_exists="signal_history" in tables,
        )

    def _run_migrations(self) -> None:
        """Run database migrations for schema changes."""
        with self.engine.connect() as conn:
            # Check if old column names exist and rename them
            try:
                # Check orders table for old column name
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'")
                )
                orders_schema = result.scalar()
                if orders_schema and "coinbase_order_id" in orders_schema:
                    conn.execute(
                        text("ALTER TABLE orders RENAME COLUMN coinbase_order_id TO exchange_order_id")
                    )
                    logger.info("migrated_orders_column", old="coinbase_order_id", new="exchange_order_id")
                    conn.commit()
            except Exception as e:
                logger.debug("orders_migration_skipped", reason=str(e))

            try:
                # Check trades table for old column name
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'")
                )
                trades_schema = result.scalar()
                if trades_schema and "coinbase_trade_id" in trades_schema:
                    conn.execute(
                        text("ALTER TABLE trades RENAME COLUMN coinbase_trade_id TO exchange_trade_id")
                    )
                    logger.info("migrated_trades_column", old="coinbase_trade_id", new="exchange_trade_id")
                    conn.commit()
            except Exception as e:
                logger.debug("trades_migration_skipped", reason=str(e))

            # Migrate daily_stats columns from USD-specific to generic names
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='daily_stats'")
                )
                stats_schema = result.scalar()
                if stats_schema:
                    if "starting_balance_usd" in stats_schema:
                        conn.execute(text("ALTER TABLE daily_stats RENAME COLUMN starting_balance_usd TO starting_balance"))
                        logger.info("migrated_daily_stats_column", old="starting_balance_usd", new="starting_balance")
                    if "ending_balance_usd" in stats_schema:
                        conn.execute(text("ALTER TABLE daily_stats RENAME COLUMN ending_balance_usd TO ending_balance"))
                        logger.info("migrated_daily_stats_column", old="ending_balance_usd", new="ending_balance")
                    if "total_volume_usd" in stats_schema:
                        conn.execute(text("ALTER TABLE daily_stats RENAME COLUMN total_volume_usd TO total_volume"))
                        logger.info("migrated_daily_stats_column", old="total_volume_usd", new="total_volume")
                    conn.commit()
            except Exception as e:
                logger.debug("daily_stats_migration_skipped", reason=str(e))

            # Add is_paper column to positions table
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='positions'")
                )
                positions_schema = result.scalar()
                if positions_schema and "is_paper" not in positions_schema:
                    conn.execute(text("ALTER TABLE positions ADD COLUMN is_paper BOOLEAN DEFAULT 0"))
                    logger.info("migrated_positions_added_is_paper")
                    conn.commit()
            except Exception as e:
                logger.debug("positions_is_paper_migration_skipped", reason=str(e))

            # Add balance snapshot columns to trades table
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'")
                )
                trades_schema = result.scalar()
                if trades_schema:
                    if "quote_balance_after" not in trades_schema:
                        conn.execute(text("ALTER TABLE trades ADD COLUMN quote_balance_after VARCHAR(50)"))
                        logger.info("migrated_trades_added_quote_balance_after")
                    if "base_balance_after" not in trades_schema:
                        conn.execute(text("ALTER TABLE trades ADD COLUMN base_balance_after VARCHAR(50)"))
                        logger.info("migrated_trades_added_base_balance_after")
                    if "spot_rate" not in trades_schema:
                        conn.execute(text("ALTER TABLE trades ADD COLUMN spot_rate VARCHAR(50)"))
                        logger.info("migrated_trades_added_spot_rate")
                    conn.commit()
            except Exception as e:
                logger.debug("trades_balance_migration_skipped", reason=str(e))

            # Migrate daily_stats to new schema with is_paper
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='daily_stats'")
                )
                stats_schema = result.scalar()
                if stats_schema and "is_paper" not in stats_schema:
                    # Rename old table, create new one, copy data
                    conn.execute(text("ALTER TABLE daily_stats RENAME TO daily_stats_old"))
                    conn.commit()
                    # Create new table with proper schema
                    conn.execute(text("""
                        CREATE TABLE daily_stats (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            date DATE NOT NULL,
                            is_paper BOOLEAN NOT NULL DEFAULT 0,
                            starting_balance VARCHAR(50) NOT NULL,
                            ending_balance VARCHAR(50),
                            realized_pnl VARCHAR(50) DEFAULT '0',
                            unrealized_pnl VARCHAR(50) DEFAULT '0',
                            total_trades INTEGER DEFAULT 0,
                            total_volume VARCHAR(50) DEFAULT '0',
                            max_drawdown_percent VARCHAR(50) DEFAULT '0'
                        )
                    """))
                    # Copy old data (assume existing data is live trading)
                    conn.execute(text("""
                        INSERT INTO daily_stats (date, is_paper, starting_balance, ending_balance,
                            realized_pnl, unrealized_pnl, total_trades, total_volume, max_drawdown_percent)
                        SELECT date, 0, starting_balance, ending_balance,
                            realized_pnl, unrealized_pnl, total_trades, total_volume, max_drawdown_percent
                        FROM daily_stats_old
                    """))
                    conn.execute(text("DROP TABLE daily_stats_old"))
                    conn.commit()
                    logger.info("migrated_daily_stats_added_is_paper")
            except Exception as e:
                logger.debug("daily_stats_is_paper_migration_skipped", reason=str(e))

            # Add starting_price and ending_price columns to daily_stats
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='daily_stats'")
                )
                stats_schema = result.scalar()
                if stats_schema:
                    if "starting_price" not in stats_schema:
                        conn.execute(text("ALTER TABLE daily_stats ADD COLUMN starting_price VARCHAR(50)"))
                        logger.info("migrated_daily_stats_added_starting_price")
                    if "ending_price" not in stats_schema:
                        conn.execute(text("ALTER TABLE daily_stats ADD COLUMN ending_price VARCHAR(50)"))
                        logger.info("migrated_daily_stats_added_ending_price")
                    conn.commit()
            except Exception as e:
                logger.debug("daily_stats_price_migration_skipped", reason=str(e))

            # Add hard_stop column to trailing_stops table
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='trailing_stops'")
                )
                ts_schema = result.scalar()
                if ts_schema and "hard_stop" not in ts_schema:
                    conn.execute(text("ALTER TABLE trailing_stops ADD COLUMN hard_stop VARCHAR(50)"))
                    logger.info("migrated_trailing_stops_added_hard_stop")
                    conn.commit()
            except Exception as e:
                logger.debug("trailing_stops_hard_stop_migration_skipped", reason=str(e))

            # Add breakeven_triggered column to trailing_stops table
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='trailing_stops'")
                )
                ts_schema = result.scalar()
                if ts_schema and "breakeven_triggered" not in ts_schema:
                    conn.execute(text("ALTER TABLE trailing_stops ADD COLUMN breakeven_triggered BOOLEAN DEFAULT 0"))
                    logger.info("migrated_trailing_stops_added_breakeven_triggered")
                    conn.commit()
            except Exception as e:
                logger.debug("trailing_stops_breakeven_migration_skipped", reason=str(e))

            # Add is_paper column to rate_history table
            # NOTE: SQLite doesn't support ALTER CONSTRAINT, so the unique constraint
            # (uq_rate_candle_v2) that includes is_paper won't be created on existing DBs.
            # For existing databases: either recreate the table or accept the limitation.
            # New databases created after this version will have correct constraints.
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='rate_history'")
                )
                rh_schema = result.scalar()
                if rh_schema and "is_paper" not in rh_schema:
                    conn.execute(text("ALTER TABLE rate_history ADD COLUMN is_paper BOOLEAN DEFAULT 0"))
                    logger.info("migrated_rate_history_added_is_paper")
                    logger.warning(
                        "rate_history_constraint_limitation",
                        message="Unique constraint uq_rate_candle_v2 not updated on existing DB. "
                                "Paper/live data separation relies on application logic. "
                                "For full constraint support, recreate rate_history table."
                    )
                    conn.commit()
            except Exception as e:
                logger.debug("rate_history_is_paper_migration_skipped", reason=str(e))

            # Migrate whale_events from string to float columns (v1.27.38)
            # SQLite doesn't support ALTER COLUMN, so we recreate the table
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='whale_events'")
                )
                whale_schema = result.scalar()
                # Check if table exists with old string-based schema (volume_ratio as VARCHAR)
                if whale_schema and "volume_ratio VARCHAR" in whale_schema:
                    # Backup existing data
                    conn.execute(text("ALTER TABLE whale_events RENAME TO whale_events_old"))
                    # Create new table with proper types (Float)
                    conn.execute(text("""
                        CREATE TABLE whale_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol VARCHAR(20) NOT NULL DEFAULT 'BTC-USD',
                            timestamp DATETIME NOT NULL DEFAULT (datetime('now')),
                            volume_ratio FLOAT NOT NULL,
                            direction VARCHAR(10) NOT NULL,
                            price_change_pct FLOAT,
                            signal_score INTEGER NOT NULL,
                            signal_action VARCHAR(10) NOT NULL,
                            is_paper BOOLEAN DEFAULT 0,
                            created_at DATETIME
                        )
                    """))
                    # Copy data with type conversion (SQLite handles string->float)
                    conn.execute(text("""
                        INSERT INTO whale_events (id, symbol, timestamp, volume_ratio, direction,
                            price_change_pct, signal_score, signal_action, is_paper, created_at)
                        SELECT id, symbol, timestamp, CAST(volume_ratio AS REAL), direction,
                            CAST(price_change_pct AS REAL), signal_score, signal_action, is_paper, created_at
                        FROM whale_events_old
                    """))
                    conn.execute(text("DROP TABLE whale_events_old"))
                    # Recreate the index
                    conn.execute(text(
                        "CREATE INDEX ix_whale_events_lookup ON whale_events (symbol, is_paper, timestamp)"
                    ))
                    conn.commit()
                    logger.info("migrated_whale_events_to_float_columns")
            except Exception as e:
                logger.debug("whale_events_float_migration_skipped", reason=str(e))

            # Rename htf_6h_trend to htf_4h_trend in signal_history (v1.28.13)
            # Code now uses 4-hour candles, variable name should match
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='signal_history'")
                )
                sh_schema = result.scalar()
                if sh_schema and "htf_6h_trend" in sh_schema:
                    conn.execute(
                        text("ALTER TABLE signal_history RENAME COLUMN htf_6h_trend TO htf_4h_trend")
                    )
                    logger.info("migrated_signal_history_column", old="htf_6h_trend", new="htf_4h_trend")
                    conn.commit()
            except Exception as e:
                logger.debug("signal_history_htf_migration_skipped", reason=str(e))

            # Add take_profit_price column to trailing_stops table
            try:
                result = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type='table' AND name='trailing_stops'")
                )
                ts_schema = result.scalar()
                if ts_schema and "take_profit_price" not in ts_schema:
                    conn.execute(text("ALTER TABLE trailing_stops ADD COLUMN take_profit_price VARCHAR(50)"))
                    logger.info("migrated_trailing_stops_added_take_profit_price")
                    conn.commit()
            except Exception as e:
                logger.debug("trailing_stops_take_profit_migration_skipped", reason=str(e))

            # Add bot_mode column to tables for Cramer Mode feature
            for table_name in ["positions", "trades", "daily_stats", "trailing_stops"]:
                try:
                    result = conn.execute(
                        text(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    )
                    table_schema = result.scalar()
                    if table_schema and "bot_mode" not in table_schema:
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN bot_mode VARCHAR(20) DEFAULT 'normal' NOT NULL"))
                        # Explicitly update existing rows to ensure they have bot_mode='normal'
                        # (SQLite versions may not auto-apply default to existing rows)
                        conn.execute(text(f"UPDATE {table_name} SET bot_mode = 'normal' WHERE bot_mode IS NULL OR bot_mode = ''"))
                        logger.info(f"migrated_{table_name}_added_bot_mode")
                        conn.commit()
                except Exception as e:
                    logger.debug(f"{table_name}_bot_mode_migration_skipped", reason=str(e))

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic commit/rollback."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Position methods
    def get_current_position(
        self, symbol: str = "BTC-USD", is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL
    ) -> Optional[Position]:
        """Get current position for a symbol."""
        with self.session() as session:
            return (
                session.query(Position)
                .filter(
                    Position.symbol == symbol,
                    Position.is_current == True,
                    Position.is_paper == is_paper,
                    Position.bot_mode == bot_mode,
                )
                .first()
            )

    def update_position(
        self,
        symbol: str,
        quantity: Decimal,
        average_cost: Decimal,
        unrealized_pnl: Decimal = Decimal("0"),
        is_paper: bool = False,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> Position:
        """Update or create current position."""
        with self.session() as session:
            # Mark old position as not current (only for same is_paper mode and bot_mode)
            session.query(Position).filter(
                Position.symbol == symbol,
                Position.is_current == True,
                Position.is_paper == is_paper,
                Position.bot_mode == bot_mode,
            ).update({"is_current": False})

            # Create new position
            position = Position(
                symbol=symbol,
                quantity=str(quantity),
                average_cost=str(average_cost),
                unrealized_pnl=str(unrealized_pnl),
                is_current=True,
                is_paper=is_paper,
                bot_mode=bot_mode,
            )
            session.add(position)
            session.flush()
            return position

    # Order methods
    def create_order(
        self,
        side: str,
        size: Decimal,
        order_type: str = "market",
        price: Optional[Decimal] = None,
        symbol: str = "BTC-USD",
        is_paper: bool = False,
    ) -> Order:
        """Create a new order record."""
        with self.session() as session:
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                size=str(size),
                price=str(price) if price else None,
                status="pending",
                is_paper=is_paper,
            )
            session.add(order)
            session.flush()
            return order

    def update_order(
        self,
        order_id: int,
        exchange_order_id: Optional[str] = None,
        status: Optional[str] = None,
        filled_size: Optional[Decimal] = None,
        filled_price: Optional[Decimal] = None,
        fee: Optional[Decimal] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Order]:
        """Update an existing order."""
        with self.session() as session:
            order = session.query(Order).filter(Order.id == order_id).first()
            if not order:
                return None

            if exchange_order_id:
                order.exchange_order_id = exchange_order_id
            if status:
                order.status = status
            if filled_size is not None:
                order.filled_size = str(filled_size)
            if filled_price is not None:
                order.filled_price = str(filled_price)
            if fee is not None:
                order.fee = str(fee)
            if error_message:
                order.error_message = error_message

            session.flush()
            return order

    def get_recent_orders(
        self, hours: int = 24, symbol: str = "BTC-USD"
    ) -> list[Order]:
        """Get recent orders."""
        with self.session() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return (
                session.query(Order)
                .filter(Order.symbol == symbol, Order.created_at > cutoff)
                .order_by(Order.created_at.desc())
                .all()
            )

    # Trade methods
    def record_trade(
        self,
        side: str,
        size: Decimal,
        price: Decimal,
        fee: Decimal = Decimal("0"),
        realized_pnl: Decimal = Decimal("0"),
        order_id: Optional[int] = None,
        exchange_trade_id: Optional[str] = None,
        symbol: str = "BTC-USD",
        is_paper: bool = False,
        bot_mode: BotMode = BotMode.NORMAL,
        quote_balance_after: Optional[Decimal] = None,
        base_balance_after: Optional[Decimal] = None,
        spot_rate: Optional[Decimal] = None,
    ) -> Trade:
        """Record an executed trade with balance snapshot."""
        with self.session() as session:
            trade = Trade(
                order_id=order_id,
                exchange_trade_id=exchange_trade_id,
                symbol=symbol,
                side=side,
                size=str(size),
                price=str(price),
                fee=str(fee),
                realized_pnl=str(realized_pnl),
                is_paper=is_paper,
                bot_mode=bot_mode,
                quote_balance_after=str(quote_balance_after) if quote_balance_after is not None else None,
                base_balance_after=str(base_balance_after) if base_balance_after is not None else None,
                spot_rate=str(spot_rate) if spot_rate is not None else None,
            )
            session.add(trade)
            session.flush()

            logger.info(
                "trade_recorded",
                trade_id=trade.id,
                side=side,
                size=str(size),
                price=str(price),
                is_paper=is_paper,
            )
            return trade

    def get_trades_today(self, symbol: str = "BTC-USD") -> list[Trade]:
        """Get all trades from today."""
        with self.session() as session:
            today = datetime.now(timezone.utc).date()
            return (
                session.query(Trade)
                .filter(
                    Trade.symbol == symbol,
                    func.date(Trade.executed_at) == today,
                )
                .order_by(Trade.executed_at.desc())
                .all()
            )

    def get_recent_trades(
        self,
        limit: int = 20,
        is_paper: bool = False,
        symbol: Optional[str] = None,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> list[Trade]:
        """
        Get recent trades filtered by paper/live mode and bot_mode.

        Args:
            limit: Maximum number of trades to return
            is_paper: Whether to get paper trades or live trades
            symbol: Optional symbol filter (e.g., "BTC-USD")
            bot_mode: Bot mode filter ("normal" or "inverted")

        Returns:
            List of Trade objects ordered by execution time (most recent first)
        """
        with self.session() as session:
            query = session.query(Trade).filter(
                Trade.is_paper == is_paper,
                Trade.bot_mode == bot_mode,
            )
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            return query.order_by(Trade.executed_at.desc()).limit(limit).all()

    def get_last_trade_by_side(
        self,
        side: str,
        symbol: str = "BTC-USD",
        is_paper: bool = False,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> Optional[Trade]:
        """
        Get the most recent trade for a specific side (buy/sell).

        Used for trade cooldown calculations.

        Args:
            side: Trade side ("buy" or "sell")
            symbol: Trading pair symbol
            is_paper: Whether to query paper trades
            bot_mode: Bot mode filter ("normal" or "inverted")

        Returns:
            Most recent Trade object for the given side, or None
        """
        with self.session() as session:
            return (
                session.query(Trade)
                .filter(
                    Trade.side == side,
                    Trade.symbol == symbol,
                    Trade.is_paper == is_paper,
                    Trade.bot_mode == bot_mode,
                )
                .order_by(Trade.executed_at.desc())
                .first()
            )

    def get_last_paper_balance(
        self, symbol: str = "BTC-USD", bot_mode: BotMode = BotMode.NORMAL
    ) -> Optional[tuple[Decimal, Decimal, Optional[Decimal]]]:
        """
        Get the last recorded paper trading balance.

        Returns:
            Tuple of (quote_balance, base_balance, spot_rate) or None if no trades
        """
        with self.session() as session:
            trade = (
                session.query(Trade)
                .filter(
                    Trade.symbol == symbol,
                    Trade.is_paper == True,
                    Trade.bot_mode == bot_mode,
                    Trade.quote_balance_after.isnot(None),
                )
                .order_by(Trade.executed_at.desc())
                .first()
            )

            if trade and trade.quote_balance_after and trade.base_balance_after:
                return (
                    Decimal(trade.quote_balance_after),
                    Decimal(trade.base_balance_after),
                    Decimal(trade.spot_rate) if trade.spot_rate else None,
                )
            return None

    # Daily stats methods
    def update_daily_stats(
        self,
        starting_balance: Optional[Decimal] = None,
        ending_balance: Optional[Decimal] = None,
        starting_price: Optional[Decimal] = None,
        ending_price: Optional[Decimal] = None,
        realized_pnl: Optional[Decimal] = None,
        total_trades: Optional[int] = None,
        volume: Optional[Decimal] = None,
        max_drawdown: Optional[Decimal] = None,
        is_paper: bool = False,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> None:
        """Update today's statistics (UTC)."""
        today = datetime.now(timezone.utc).date()

        with self.session() as session:
            stats = (
                session.query(DailyStats)
                .filter(
                    DailyStats.date == today,
                    DailyStats.is_paper == is_paper,
                    DailyStats.bot_mode == bot_mode,
                )
                .first()
            )

            if not stats:
                stats = DailyStats(
                    date=today,
                    starting_balance=str(starting_balance or Decimal("0")),
                    is_paper=is_paper,
                    bot_mode=bot_mode,
                )
                session.add(stats)

            # Update starting_balance if provided and not already set (or is "0")
            if starting_balance is not None and (not stats.starting_balance or stats.starting_balance == "0"):
                stats.starting_balance = str(starting_balance)
            if starting_price is not None:
                stats.starting_price = str(starting_price)
            if ending_balance is not None:
                stats.ending_balance = str(ending_balance)
            if ending_price is not None:
                stats.ending_price = str(ending_price)
            if realized_pnl is not None:
                stats.realized_pnl = str(realized_pnl)
            if total_trades is not None:
                stats.total_trades = total_trades
            if volume is not None:
                stats.total_volume = str(volume)
            if max_drawdown is not None:
                stats.max_drawdown_percent = str(max_drawdown)

            session.commit()

    def increment_daily_trade_count(self, is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL) -> None:
        """Increment today's trade count by 1 (UTC)."""
        today = datetime.now(timezone.utc).date()

        with self.session() as session:
            stats = (
                session.query(DailyStats)
                .filter(
                    DailyStats.date == today,
                    DailyStats.is_paper == is_paper,
                    DailyStats.bot_mode == bot_mode,
                )
                .first()
            )

            if not stats:
                # Create the record if it doesn't exist
                # starting_balance will be updated when daemon records it
                stats = DailyStats(
                    date=today,
                    is_paper=is_paper,
                    bot_mode=bot_mode,
                    total_trades=0,
                    starting_balance="0",  # Placeholder until daemon sets it
                )
                session.add(stats)

            stats.total_trades = (stats.total_trades or 0) + 1
            session.commit()

    def count_todays_trades(self, is_paper: bool = False, symbol: Optional[str] = None, bot_mode: BotMode = BotMode.NORMAL) -> int:
        """Count trades executed today (UTC)."""
        today = datetime.now(timezone.utc).date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())

        with self.session() as session:
            query = session.query(Trade).filter(
                Trade.is_paper == is_paper,
                Trade.bot_mode == bot_mode,
                Trade.executed_at >= today_start,
                Trade.executed_at <= today_end,
            )
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            return query.count()

    def get_todays_realized_pnl(self, is_paper: bool = False, symbol: Optional[str] = None, bot_mode: BotMode = BotMode.NORMAL) -> Decimal:
        """Sum realized P&L from today's trades (UTC) using SQL aggregation."""
        today = datetime.now(timezone.utc).date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())

        with self.session() as session:
            # Use SQL SUM with COALESCE for better performance with large trade histories
            # COALESCE handles NULL values explicitly (buy orders have NULL realized_pnl)
            query = session.query(
                func.coalesce(func.sum(Trade.realized_pnl), 0)
            ).filter(
                Trade.is_paper == is_paper,
                Trade.bot_mode == bot_mode,
                Trade.executed_at >= today_start,
                Trade.executed_at <= today_end,
            )
            if symbol:
                query = query.filter(Trade.symbol == symbol)

            result = query.scalar()
            if result is None or result == 0:
                return Decimal("0")
            try:
                return Decimal(str(result))
            except (InvalidOperation, ValueError) as e:
                logger.error("invalid_pnl_sum", result=result, error=str(e))
                return Decimal("0")

    def get_daily_stats(
        self, target_date: Optional[date] = None, is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL
    ) -> Optional[DailyStats]:
        """Get statistics for a specific date (defaults to today UTC)."""
        target_date = target_date or datetime.now(timezone.utc).date()

        with self.session() as session:
            return (
                session.query(DailyStats)
                .filter(
                    DailyStats.date == target_date,
                    DailyStats.is_paper == is_paper,
                    DailyStats.bot_mode == bot_mode,
                )
                .first()
            )

    def get_daily_stats_range(
        self, start_date: date, end_date: date, is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL
    ) -> list[DailyStats]:
        """Get statistics for a date range (inclusive)."""
        with self.session() as session:
            return (
                session.query(DailyStats)
                .filter(
                    DailyStats.date >= start_date,
                    DailyStats.date <= end_date,
                    DailyStats.is_paper == is_paper,
                    DailyStats.bot_mode == bot_mode,
                )
                .order_by(DailyStats.date.asc())
                .all()
            )

    # System state methods
    def set_state(self, key: str, value: Any) -> None:
        """Set a system state value."""
        with self.session() as session:
            state = session.query(SystemState).filter(SystemState.key == key).first()

            if state:
                state.value = json.dumps(value)
            else:
                state = SystemState(key=key, value=json.dumps(value))
                session.add(state)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a system state value."""
        with self.session() as session:
            state = session.query(SystemState).filter(SystemState.key == key).first()

            if state:
                return json.loads(state.value)
            return default

    def delete_state(self, key: str) -> bool:
        """Delete a system state value."""
        with self.session() as session:
            result = session.query(SystemState).filter(SystemState.key == key).delete()
            return result > 0

    # Notification methods
    def save_notification(
        self,
        type: str,
        title: str,
        message: str,
        is_paper: bool = False,
    ) -> Notification:
        """Save a notification to the database."""
        with self.session() as session:
            notification = Notification(
                type=type,
                title=title,
                message=message,
                is_paper=is_paper,
            )
            session.add(notification)
            session.flush()
            return notification

    def get_recent_notifications(
        self,
        limit: int = 50,
        is_paper: Optional[bool] = None,
    ) -> list[Notification]:
        """Get recent notifications."""
        with self.session() as session:
            query = session.query(Notification)
            if is_paper is not None:
                query = query.filter(Notification.is_paper == is_paper)
            return query.order_by(Notification.created_at.desc()).limit(limit).all()

    # Trailing stop methods
    def create_trailing_stop(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        trailing_activation: Decimal,
        trailing_distance: Decimal,
        is_paper: bool = False,
        bot_mode: BotMode = BotMode.NORMAL,
        hard_stop: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
    ) -> TrailingStop:
        """Create a new trailing stop record with optional hard stop-loss and take profit."""
        with self.session() as session:
            # Deactivate any existing active trailing stops for this symbol/mode/bot_mode
            session.query(TrailingStop).filter(
                TrailingStop.symbol == symbol,
                TrailingStop.is_paper == is_paper,
                TrailingStop.bot_mode == bot_mode,
                TrailingStop.is_active == True,
            ).update({"is_active": False})

            trailing_stop = TrailingStop(
                symbol=symbol,
                side=side,
                entry_price=str(entry_price),
                trailing_activation=str(trailing_activation),
                trailing_distance=str(trailing_distance),
                hard_stop=str(hard_stop) if hard_stop else None,
                take_profit_price=str(take_profit_price) if take_profit_price else None,
                is_active=True,
                is_paper=is_paper,
                bot_mode=bot_mode,
            )
            session.add(trailing_stop)
            session.flush()

            logger.info(
                "trailing_stop_created",
                id=trailing_stop.id,
                side=side,
                entry_price=str(entry_price),
                activation=str(trailing_activation),
                distance=str(trailing_distance),
                hard_stop=str(hard_stop) if hard_stop else None,
                take_profit_price=str(take_profit_price) if take_profit_price else None,
                bot_mode=bot_mode,
            )
            return trailing_stop

    def get_active_trailing_stop(
        self, symbol: str = "BTC-USD", is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL
    ) -> Optional[TrailingStop]:
        """Get the active trailing stop for a symbol."""
        with self.session() as session:
            return (
                session.query(TrailingStop)
                .filter(
                    TrailingStop.symbol == symbol,
                    TrailingStop.is_paper == is_paper,
                    TrailingStop.bot_mode == bot_mode,
                    TrailingStop.is_active == True,
                )
                .first()
            )

    def update_trailing_stop(
        self,
        trailing_stop_id: int,
        new_stop_level: Optional[Decimal] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[TrailingStop]:
        """Update a trailing stop record."""
        with self.session() as session:
            ts = session.query(TrailingStop).filter(TrailingStop.id == trailing_stop_id).first()
            if not ts:
                return None

            if new_stop_level is not None:
                ts.trailing_stop = str(new_stop_level)
            if is_active is not None:
                ts.is_active = is_active

            session.flush()
            return ts

    def update_trailing_stop_breakeven(
        self,
        trailing_stop_id: int,
        new_hard_stop: Decimal,
    ) -> Optional[TrailingStop]:
        """
        Activate break-even protection by moving hard stop to entry price.

        This is called when price reaches the break-even threshold (e.g., +0.5 ATR).
        Once triggered, the position is protected at entry - no loss possible.
        """
        with self.session() as session:
            ts = session.query(TrailingStop).filter(TrailingStop.id == trailing_stop_id).first()
            if not ts:
                return None

            ts.hard_stop = str(new_hard_stop)
            ts.breakeven_triggered = True

            session.flush()
            return ts

    def update_trailing_stop_for_dca(
        self,
        symbol: str,
        entry_price: Decimal,
        trailing_activation: Decimal,
        trailing_distance: Decimal,
        hard_stop: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        is_paper: bool = False,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> Optional[TrailingStop]:
        """
        Update existing trailing stop for DCA (position averaging).

        Unlike create_trailing_stop, this method updates the existing stop
        in-place without deactivating it, ensuring there's NO window where
        the position is unprotected.

        Returns the updated TrailingStop, or None if no active stop exists.
        """
        with self.session() as session:
            ts = (
                session.query(TrailingStop)
                .filter(
                    TrailingStop.symbol == symbol,
                    TrailingStop.is_paper == is_paper,
                    TrailingStop.bot_mode == bot_mode,
                    TrailingStop.is_active == True,
                )
                .first()
            )

            if not ts:
                return None

            # Update stop parameters based on new avg_cost
            ts.entry_price = str(entry_price)
            ts.trailing_activation = str(trailing_activation)
            ts.trailing_distance = str(trailing_distance)
            ts.hard_stop = str(hard_stop) if hard_stop else None
            ts.take_profit_price = str(take_profit_price) if take_profit_price else None
            # Reset break-even flag so it can re-trigger at new avg_cost level
            ts.breakeven_triggered = False
            # Note: Don't reset trailing_stop level - let it continue trailing

            session.flush()

            logger.info(
                "trailing_stop_updated_for_dca",
                id=ts.id,
                entry_price=str(entry_price),
                activation=str(trailing_activation),
                hard_stop=str(hard_stop) if hard_stop else None,
                take_profit_price=str(take_profit_price) if take_profit_price else None,
            )
            return ts

    def deactivate_trailing_stop(
        self, symbol: str = "BTC-USD", is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL
    ) -> bool:
        """Deactivate all trailing stops for a symbol."""
        with self.session() as session:
            result = session.query(TrailingStop).filter(
                TrailingStop.symbol == symbol,
                TrailingStop.is_paper == is_paper,
                TrailingStop.bot_mode == bot_mode,
                TrailingStop.is_active == True,
            ).update({"is_active": False})
            return result > 0

    # Regime history methods
    def record_regime_change(
        self,
        regime_name: str,
        threshold_adjustment: int,
        position_multiplier: float,
        sentiment_value: Optional[int] = None,
        sentiment_category: Optional[str] = None,
        volatility_level: Optional[str] = None,
        trend_direction: Optional[str] = None,
        is_paper: bool = False,
    ) -> RegimeHistory:
        """Record a market regime change."""
        with self.session() as session:
            record = RegimeHistory(
                regime_name=regime_name,
                threshold_adjustment=threshold_adjustment,
                position_multiplier=str(position_multiplier),
                sentiment_value=sentiment_value,
                sentiment_category=sentiment_category,
                volatility_level=volatility_level,
                trend_direction=trend_direction,
                is_paper=is_paper,
            )
            session.add(record)
            session.flush()

            logger.info(
                "regime_change_recorded",
                regime=regime_name,
                threshold_adj=threshold_adjustment,
                position_mult=position_multiplier,
            )
            return record

    def get_last_regime(self, is_paper: bool = False) -> Optional[str]:
        """Get the last recorded regime name for session recovery."""
        with self.session() as session:
            record = (
                session.query(RegimeHistory)
                .filter(RegimeHistory.is_paper == is_paper)
                .order_by(RegimeHistory.created_at.desc())
                .first()
            )
            return record.regime_name if record else None

    def get_regime_history(
        self, hours: int = 24, is_paper: bool = False
    ) -> list[RegimeHistory]:
        """Get regime change history for the past N hours."""
        with self.session() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return (
                session.query(RegimeHistory)
                .filter(
                    RegimeHistory.is_paper == is_paper,
                    RegimeHistory.created_at > cutoff,
                )
                .order_by(RegimeHistory.created_at.desc())
                .all()
            )

    # Weight profile history methods
    def record_weight_profile_change(
        self,
        profile_name: str,
        confidence: float,
        reasoning: str,
        market_context: dict,
        is_paper: bool = False,
    ) -> WeightProfileHistory:
        """Record an AI weight profile selection."""
        import json

        with self.session() as session:
            record = WeightProfileHistory(
                profile_name=profile_name,
                confidence=str(confidence),
                reasoning=reasoning,
                market_context=json.dumps(market_context),
                is_paper=is_paper,
            )
            session.add(record)
            session.flush()

            logger.info(
                "weight_profile_recorded",
                profile=profile_name,
                confidence=confidence,
            )
            return record

    def get_last_weight_profile(self, is_paper: bool = False) -> Optional[str]:
        """Get the last recorded weight profile name for session recovery."""
        with self.session() as session:
            record = (
                session.query(WeightProfileHistory)
                .filter(WeightProfileHistory.is_paper == is_paper)
                .order_by(WeightProfileHistory.created_at.desc())
                .first()
            )
            return record.profile_name if record else None

    # Whale activity methods
    def record_whale_event(
        self,
        symbol: str,
        volume_ratio: float,
        direction: str,
        price_change_pct: Optional[float],
        signal_score: int,
        signal_action: str,
        is_paper: bool = False,
    ) -> WhaleEvent:
        """Record a whale activity detection event.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            volume_ratio: Volume ratio vs average (e.g., 3.45)
            direction: Whale direction ("bullish", "bearish", "neutral")
            price_change_pct: Price change percentage during spike
            signal_score: Signal score at time of detection (-100 to +100)
            signal_action: Signal action ("buy", "sell", "hold")
            is_paper: Whether this is paper trading

        Returns:
            WhaleEvent: The recorded event
        """
        # Validate direction
        valid_directions = {"bullish", "bearish", "neutral", "unknown"}
        if direction not in valid_directions:
            logger.warning("invalid_whale_direction", direction=direction, defaulting_to="unknown")
            direction = "unknown"

        with self.session() as session:
            record = WhaleEvent(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                volume_ratio=volume_ratio,
                direction=direction,
                price_change_pct=price_change_pct,
                signal_score=signal_score,
                signal_action=signal_action,
                is_paper=is_paper,
            )
            session.add(record)
            session.flush()

            logger.info(
                "whale_event_recorded",
                symbol=symbol,
                volume_ratio=volume_ratio,
                direction=direction,
                signal_score=signal_score,
            )
            return record

    def get_whale_events(
        self, hours: int = 24, symbol: Optional[str] = None, is_paper: bool = False
    ) -> list[WhaleEvent]:
        """Get whale events for the past N hours.

        Args:
            hours: Number of hours to look back
            symbol: Filter by symbol (optional)
            is_paper: Whether to query paper or live data

        Returns:
            List of WhaleEvent records
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.session() as session:
            query = (
                session.query(WhaleEvent)
                .filter(WhaleEvent.is_paper == is_paper)
                .filter(WhaleEvent.timestamp >= cutoff)
            )
            if symbol:
                query = query.filter(WhaleEvent.symbol == symbol)
            return query.order_by(WhaleEvent.timestamp.desc()).all()

    # Rate history methods
    def record_rate(
        self,
        timestamp: datetime,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal,
        symbol: str = "BTC-USD",
        exchange: str = "kraken",
        interval: str = "1m",
        is_paper: bool = False,
    ) -> Optional[RateHistory]:
        """
        Record a single OHLCV candle.

        Uses INSERT OR IGNORE to skip duplicates (same symbol/exchange/interval/timestamp/is_paper).
        """
        with self.session() as session:
            # Check if candle already exists
            existing = (
                session.query(RateHistory)
                .filter(
                    RateHistory.symbol == symbol,
                    RateHistory.exchange == exchange,
                    RateHistory.interval == interval,
                    RateHistory.timestamp == timestamp,
                    RateHistory.is_paper == is_paper,
                )
                .first()
            )

            if existing:
                return None  # Skip duplicate

            rate = RateHistory(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                timestamp=timestamp,
                open_price=str(open_price),
                high_price=str(high_price),
                low_price=str(low_price),
                close_price=str(close_price),
                volume=str(volume),
                is_paper=is_paper,
            )
            session.add(rate)
            return rate

    def record_rates_bulk(
        self,
        candles: list[dict],
        symbol: str = "BTC-USD",
        exchange: str = "kraken",
        interval: str = "1m",
        is_paper: bool = False,
    ) -> int:
        """
        Record multiple OHLCV candles efficiently.

        Args:
            candles: List of dicts with keys: timestamp, open, high, low, close, volume
            is_paper: Whether this is paper trading data

        Returns:
            Number of new candles inserted (skips duplicates)

        Raises:
            Exception: Re-raised if batch insert fails. The session context manager
                      automatically handles rollback on exception, so if an error
                      occurs, NO candles from this batch will be committed.
                      This ensures atomic operation - all or nothing.
        """
        if not candles:
            return 0

        inserted = 0
        try:
            with self.session() as session:
                # Normalize timestamps to naive UTC datetime (pandas Timestamp -> datetime)
                def to_datetime(ts):
                    if hasattr(ts, 'to_pydatetime'):
                        dt = ts.to_pydatetime()
                    else:
                        dt = ts
                    # Convert to UTC before removing tzinfo (SQLite stores naive datetimes)
                    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    else:
                        # Naive datetime - assumed to be UTC, warn since this could hide bugs
                        logger.warning("rate_history_naive_timestamp", timestamp=str(dt))
                    return dt

                candle_timestamps = [to_datetime(c["timestamp"]) for c in candles]

                # Batch fetch existing timestamps in one query for performance
                existing_timestamps = set(
                    row[0] for row in session.query(RateHistory.timestamp)
                    .filter(
                        RateHistory.symbol == symbol,
                        RateHistory.exchange == exchange,
                        RateHistory.interval == interval,
                        RateHistory.is_paper == is_paper,
                        RateHistory.timestamp.in_(candle_timestamps),
                    )
                    .all()
                )

                # Insert only new candles (batch - flush happens at context exit)
                for candle, normalized_ts in zip(candles, candle_timestamps):
                    if normalized_ts not in existing_timestamps:
                        rate = RateHistory(
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            timestamp=normalized_ts,
                            open_price=str(candle["open"]),
                            high_price=str(candle["high"]),
                            low_price=str(candle["low"]),
                            close_price=str(candle["close"]),
                            volume=str(candle["volume"]),
                            is_paper=is_paper,
                        )
                        session.add(rate)
                        inserted += 1

            if inserted > 0:
                logger.info(
                    "rates_recorded",
                    count=inserted,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                )
            return inserted
        except Exception as e:
            logger.error(
                "rate_bulk_insert_failed",
                error=str(e),
                count=len(candles),
                inserted_before_error=inserted,
            )
            raise  # Re-raise to signal failure to caller

    def get_rates(
        self,
        symbol: str = "BTC-USD",
        exchange: str = "kraken",
        interval: str = "1m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
        is_paper: bool = False,
    ) -> list[RateHistory]:
        """
        Get historical rate data for analysis or replay.

        Args:
            symbol: Trading pair
            exchange: Exchange name
            interval: Candle interval
            start: Start timestamp (inclusive, UTC)
            end: End timestamp (inclusive, UTC)
            limit: Maximum records to return
            is_paper: Whether to fetch paper trading data

        Returns:
            List of RateHistory records ordered by timestamp ASC
        """
        with self.session() as session:
            query = session.query(RateHistory).filter(
                RateHistory.symbol == symbol,
                RateHistory.exchange == exchange,
                RateHistory.interval == interval,
                RateHistory.is_paper == is_paper,
            )

            if start:
                query = query.filter(RateHistory.timestamp >= start)
            if end:
                query = query.filter(RateHistory.timestamp <= end)

            return (
                query
                .order_by(RateHistory.timestamp.asc())
                .limit(limit)
                .all()
            )

    def get_latest_rate_timestamp(
        self,
        symbol: str = "BTC-USD",
        exchange: str = "kraken",
        interval: str = "1m",
        is_paper: bool = False,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent stored candle (UTC)."""
        with self.session() as session:
            result = (
                session.query(RateHistory.timestamp)
                .filter(
                    RateHistory.symbol == symbol,
                    RateHistory.exchange == exchange,
                    RateHistory.interval == interval,
                    RateHistory.is_paper == is_paper,
                )
                .order_by(RateHistory.timestamp.desc())
                .first()
            )
            return result[0] if result else None

    def get_rate_count(
        self,
        symbol: str = "BTC-USD",
        exchange: str = "kraken",
        interval: str = "1m",
        is_paper: bool = False,
    ) -> int:
        """Get total number of stored candles for a symbol/exchange/interval."""
        with self.session() as session:
            return (
                session.query(func.count(RateHistory.id))
                .filter(
                    RateHistory.symbol == symbol,
                    RateHistory.exchange == exchange,
                    RateHistory.interval == interval,
                    RateHistory.is_paper == is_paper,
                )
                .scalar()
            )
