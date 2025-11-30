"""
SQLite database for state persistence and trade history.

Tables:
- positions: Current and historical positions
- orders: All orders placed
- trades: Executed trades (fills)
- daily_stats: Daily trading statistics
- system_state: Key-value store for recovery state
"""

import json
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Generator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

import structlog

logger = structlog.get_logger(__name__)

Base = declarative_base()


class Position(Base):
    """Current and historical positions."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, default="BTC-USD")
    quantity = Column(String(50), nullable=False)  # Stored as string for Decimal precision
    average_cost = Column(String(50), nullable=False)
    unrealized_pnl = Column(String(50), default="0")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_current = Column(Boolean, default=True)

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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
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
    executed_at = Column(DateTime, default=datetime.utcnow)
    is_paper = Column(Boolean, default=False)


class DailyStats(Base):
    """Daily trading statistics."""

    __tablename__ = "daily_stats"

    date = Column(Date, primary_key=True)
    starting_balance = Column(String(50), nullable=False)  # In quote currency
    ending_balance = Column(String(50), nullable=True)  # In quote currency
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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Run migrations for existing databases
        self._run_migrations()

        logger.info("database_initialized", path=str(db_path))

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
    def get_current_position(self, symbol: str = "BTC-USD") -> Optional[Position]:
        """Get current position for a symbol."""
        with self.session() as session:
            return (
                session.query(Position)
                .filter(Position.symbol == symbol, Position.is_current == True)
                .first()
            )

    def update_position(
        self,
        symbol: str,
        quantity: Decimal,
        average_cost: Decimal,
        unrealized_pnl: Decimal = Decimal("0"),
    ) -> Position:
        """Update or create current position."""
        with self.session() as session:
            # Mark old position as not current
            session.query(Position).filter(
                Position.symbol == symbol, Position.is_current == True
            ).update({"is_current": False})

            # Create new position
            position = Position(
                symbol=symbol,
                quantity=str(quantity),
                average_cost=str(average_cost),
                unrealized_pnl=str(unrealized_pnl),
                is_current=True,
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
            cutoff = datetime.utcnow() - timedelta(hours=hours)
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
    ) -> Trade:
        """Record an executed trade."""
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
            today = datetime.utcnow().date()
            return (
                session.query(Trade)
                .filter(
                    Trade.symbol == symbol,
                    func.date(Trade.executed_at) == today,
                )
                .order_by(Trade.executed_at.desc())
                .all()
            )

    # Daily stats methods
    def update_daily_stats(
        self,
        starting_balance: Optional[Decimal] = None,
        ending_balance: Optional[Decimal] = None,
        realized_pnl: Optional[Decimal] = None,
        total_trades: Optional[int] = None,
        volume: Optional[Decimal] = None,
        max_drawdown: Optional[Decimal] = None,
    ) -> DailyStats:
        """Update today's statistics."""
        today = date.today()

        with self.session() as session:
            stats = session.query(DailyStats).filter(DailyStats.date == today).first()

            if not stats:
                stats = DailyStats(
                    date=today,
                    starting_balance=str(starting_balance or Decimal("0")),
                )
                session.add(stats)

            if ending_balance is not None:
                stats.ending_balance = str(ending_balance)
            if realized_pnl is not None:
                stats.realized_pnl = str(realized_pnl)
            if total_trades is not None:
                stats.total_trades = total_trades
            if volume is not None:
                stats.total_volume = str(volume)
            if max_drawdown is not None:
                stats.max_drawdown_percent = str(max_drawdown)

            session.flush()
            return stats

    def get_daily_stats(self, target_date: Optional[date] = None) -> Optional[DailyStats]:
        """Get statistics for a specific date."""
        target_date = target_date or date.today()

        with self.session() as session:
            return (
                session.query(DailyStats)
                .filter(DailyStats.date == target_date)
                .first()
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


# Import timedelta at the top where other imports are
from datetime import timedelta
