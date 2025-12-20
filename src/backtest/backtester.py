"""
Historical backtesting engine for strategy validation.

Runs SignalScorer against historical OHLCV data with simulated
trade execution, fees, and slippage to validate strategy performance.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import pandas as pd
import structlog

from src.strategy.signal_scorer import SignalScorer
from src.strategy.position_sizer import PositionSizer, PositionSizeConfig

logger = structlog.get_logger(__name__)


@dataclass
class BacktestTrade:
    """Record of a backtest trade execution."""

    timestamp: datetime
    side: str  # "buy" or "sell"
    entry_price: Decimal
    size_base: Decimal
    size_quote: Decimal
    fee: Decimal
    signal_score: int
    stop_loss_price: Decimal
    take_profit_price: Decimal
    exit_price: Optional[Decimal] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "signal", "stop_loss", "take_profit", "end_of_backtest"
    pnl: Optional[Decimal] = None
    pnl_percent: Optional[Decimal] = None


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest run."""

    # Returns
    total_return: float  # Total portfolio return (%)
    sharpe_ratio: float  # Risk-adjusted return

    # Risk
    max_drawdown: float  # Maximum peak-to-trough decline (%)
    volatility: float  # Annualized volatility (%)

    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # Winning trades / total trades (%)

    # Profitability
    profit_factor: float  # Gross profit / gross loss
    avg_win: float  # Average winning trade (%)
    avg_loss: float  # Average losing trade (%)
    avg_trade_duration_hours: float

    # Portfolio
    final_value: Decimal
    initial_value: Decimal
    total_fees: Decimal


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: pd.DataFrame  # timestamp, portfolio_value
    signal_distribution: dict[str, int]  # Count of each signal action

    def summary(self) -> str:
        """Generate human-readable summary."""
        m = self.metrics
        return f"""
Backtest Results
================
Total Return: {m.total_return:.2f}%
Sharpe Ratio: {m.sharpe_ratio:.2f}
Max Drawdown: {m.max_drawdown:.2f}%
Win Rate: {m.win_rate:.1f}%
Profit Factor: {m.profit_factor:.2f}
Total Trades: {m.total_trades}

Portfolio:
  Initial: ${m.initial_value:,.2f}
  Final: ${m.final_value:,.2f}
  Fees: ${m.total_fees:,.2f}

Trade Stats:
  Avg Win: {m.avg_win:.2f}%
  Avg Loss: {m.avg_loss:.2f}%
  Avg Duration: {m.avg_trade_duration_hours:.1f}h

Signal Distribution:
  Buy: {self.signal_distribution.get('buy', 0)}
  Sell: {self.signal_distribution.get('sell', 0)}
  Hold: {self.signal_distribution.get('hold', 0)}
""".strip()


class Backtester:
    """
    Historical backtesting engine.

    Features:
    - Sequential candle-by-candle execution
    - Realistic fill simulation with slippage and fees
    - Position sizing with ATR-based risk management
    - Comprehensive performance metrics
    - Signal distribution analysis

    Limitations:
    - Long-only positions: This backtester only supports long positions (buy to open,
      sell to close). Short selling is not currently implemented.

    Example:
        >>> bt = Backtester(
        ...     signal_scorer=SignalScorer(threshold=60),
        ...     position_sizer=PositionSizer(),
        ...     initial_capital=10000,
        ...     fee_percent=0.006,
        ... )
        >>> results = bt.run(historical_df, start_date="2024-01-01")
        >>> print(results.summary())
    """

    def __init__(
        self,
        signal_scorer: SignalScorer,
        position_sizer: Optional[PositionSizer] = None,
        initial_capital: float = 10000.0,
        fee_percent: float = 0.006,  # 0.6% (Coinbase Advanced taker)
        slippage_percent: float = 0.001,  # 0.1%
        min_trade_size: float = 10.0,  # Minimum trade size in quote currency
    ):
        """
        Initialize backtester.

        Args:
            signal_scorer: SignalScorer instance with strategy parameters
            position_sizer: PositionSizer for trade sizing (creates default if None)
            initial_capital: Starting capital in quote currency
            fee_percent: Trading fee as decimal (0.006 = 0.6%)
            slippage_percent: Slippage as decimal (0.001 = 0.1%)
            min_trade_size: Minimum trade size in quote currency (default: 10.0)
        """
        self.signal_scorer = signal_scorer
        self.position_sizer = position_sizer or PositionSizer(
            config=PositionSizeConfig(
                risk_per_trade_percent=0.5,
                min_trade_base=0.0001,
            )
        )
        # Validate fee and slippage ranges to catch configuration errors
        if not (0 <= fee_percent <= 1.0):
            raise ValueError(f"fee_percent must be between 0 and 1.0, got {fee_percent}")
        if not (0 <= slippage_percent <= 1.0):
            raise ValueError(f"slippage_percent must be between 0 and 1.0, got {slippage_percent}")

        self.initial_capital = Decimal(str(initial_capital))
        self.fee_percent = Decimal(str(fee_percent))
        self.slippage_percent = Decimal(str(slippage_percent))
        self.min_trade_size = Decimal(str(min_trade_size))

    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical OHLCV data.

        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
                 Must be sorted by timestamp ascending
            start_date: Optional start date (YYYY-MM-DD) to filter data
            end_date: Optional end date (YYYY-MM-DD) to filter data

        Returns:
            BacktestResult with metrics, trades, and equity curve

        Raises:
            ValueError: If data is missing required columns or is unsorted
        """
        # Validate data
        if data.empty:
            raise ValueError("Data cannot be empty")

        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Filter by date range
        df = data.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            if start_date:
                # Handle timezone-aware comparisons
                start_ts = pd.to_datetime(start_date)
                if df["timestamp"].dt.tz is not None and start_ts.tz is None:
                    start_ts = start_ts.tz_localize("UTC")
                df = df[df["timestamp"] >= start_ts]
            if end_date:
                # Handle timezone-aware comparisons
                end_ts = pd.to_datetime(end_date)
                if df["timestamp"].dt.tz is not None and end_ts.tz is None:
                    end_ts = end_ts.tz_localize("UTC")
                df = df[df["timestamp"] <= end_ts]

        if df.empty:
            raise ValueError("No data after applying date filters")

        logger.info(
            "backtest_starting",
            rows=len(df),
            start=df["timestamp"].iloc[0],
            end=df["timestamp"].iloc[-1],
            initial_capital=str(self.initial_capital),
        )

        # Initialize state
        quote_balance = self.initial_capital
        base_balance = Decimal("0")
        trades: list[BacktestTrade] = []
        equity_curve = []
        signal_counts = {"buy": 0, "sell": 0, "hold": 0}

        # For tracking metrics
        total_fees = Decimal("0")
        open_position: Optional[BacktestTrade] = None

        # Need minimum candles for indicators
        # Use defensive attribute access in case SignalScorer internals change
        min_candles = max(
            getattr(self.signal_scorer, "ema_slow_period", 21),
            getattr(self.signal_scorer, "bollinger_period", 20),
            26,  # MACD slow
            self.position_sizer.atr_period,  # ATR period for position sizing
        )

        # Run through each candle
        for idx in range(len(df)):
            # Need enough history for indicators
            if idx < min_candles:
                continue

            # Get current candle and historical window
            current_idx = idx
            hist_df = df.iloc[:current_idx + 1].copy()
            current_candle = df.iloc[current_idx]
            current_price = Decimal(str(current_candle["close"]))
            timestamp = current_candle["timestamp"]

            # Calculate signal
            # NOTE: No look-ahead bias here. We pass hist_df which includes the current candle,
            # but this is correct because:
            # 1. We assume the current candle has CLOSED (we're using its close price)
            # 2. Indicators (RSI, MACD, etc.) are calculated using all data up to and including
            #    this completed candle, which matches real-world trading behavior
            # 3. In live trading, you wait for candle close, then calculate indicators using
            #    that completed candle's data - this simulation replicates that exactly
            signal_result = self.signal_scorer.calculate_score(hist_df)

            # Validate signal result (defensive check for financial system)
            if not hasattr(signal_result, 'action'):
                raise ValueError(f"SignalScorer.calculate_score() returned invalid result: missing 'action' attribute")
            if signal_result.action not in {'buy', 'sell', 'hold'}:
                raise ValueError(f"Invalid signal action from SignalScorer: {signal_result.action}. Must be 'buy', 'sell', or 'hold'")

            signal_counts[signal_result.action] += 1

            # Check for stop-loss or take-profit hits on open position
            # This must happen BEFORE checking for new signals to ensure realistic exit behavior
            if open_position is not None:
                candle_high = Decimal(str(current_candle["high"]))
                candle_low = Decimal(str(current_candle["low"]))

                # For long positions: check if low hit stop-loss or high hit take-profit
                stop_hit = candle_low <= open_position.stop_loss_price
                tp_hit = candle_high >= open_position.take_profit_price

                if stop_hit or tp_hit:
                    # IMPORTANT: When both stop-loss and take-profit are hit within the same candle,
                    # we assume stop-loss was triggered first. This is a conservative approach that:
                    # 1. Prevents overstating backtest performance by avoiding the assumption that
                    #    the more favorable exit (take-profit) occurred
                    # 2. Better represents worst-case execution scenarios
                    # 3. Accounts for the fact that without tick data, we cannot determine the
                    #    actual intra-candle price sequence
                    # Alternative: Could use candle open→close direction as a heuristic, but this
                    # adds complexity and may still not reflect reality in volatile markets.
                    exit_reason = "stop_loss" if stop_hit else "take_profit"
                    fill_price = open_position.stop_loss_price if stop_hit else open_position.take_profit_price

                    # Calculate proceeds and fee
                    gross_proceeds = base_balance * fill_price
                    fee = gross_proceeds * self.fee_percent
                    net_proceeds = gross_proceeds - fee

                    # Execute exit
                    quote_balance += net_proceeds
                    total_fees += fee

                    # Calculate P&L
                    pnl = gross_proceeds - open_position.size_quote - fee - open_position.fee
                    pnl_percent = (pnl / open_position.size_quote) * Decimal("100")

                    # Update and record trade
                    open_position.exit_price = fill_price
                    open_position.exit_timestamp = timestamp
                    open_position.exit_reason = exit_reason
                    open_position.pnl = pnl
                    open_position.pnl_percent = pnl_percent
                    trades.append(open_position)

                    logger.debug(
                        "backtest_stop_tp_exit",
                        timestamp=timestamp,
                        reason=exit_reason,
                        price=str(fill_price),
                        size_base=str(base_balance),
                        pnl=str(pnl),
                        pnl_percent=str(pnl_percent),
                    )

                    base_balance = Decimal("0")
                    open_position = None

            # Execute trade logic
            if signal_result.action == "buy" and open_position is None and quote_balance > 0:
                # Calculate position size
                size_result = self.position_sizer.calculate_size(
                    df=hist_df,
                    current_price=current_price,
                    quote_balance=quote_balance,
                    base_balance=base_balance,
                    signal_strength=abs(signal_result.score),
                    side="buy",
                )

                if size_result.size_quote >= self.min_trade_size:
                    # Apply slippage (price goes up for buys)
                    # NOTE: Fill price is based on current candle's close price + slippage.
                    # This assumes instant execution at candle close, which is optimistic.
                    # In live trading, orders execute at market price (often next candle's open).
                    # The slippage parameter partially compensates for this execution delay.
                    fill_price = current_price * (Decimal("1") + self.slippage_percent)

                    # Calculate fee and actual cost
                    fee = size_result.size_quote * self.fee_percent
                    total_cost = size_result.size_quote + fee

                    # Check sufficient balance
                    if total_cost <= quote_balance:
                        # Execute buy
                        base_received = (size_result.size_quote / fill_price).quantize(
                            Decimal("0.00000001"), rounding=ROUND_DOWN
                        )
                        quote_balance -= total_cost
                        base_balance += base_received
                        total_fees += fee

                        open_position = BacktestTrade(
                            timestamp=timestamp,
                            side="buy",
                            entry_price=fill_price,
                            size_base=base_received,
                            size_quote=size_result.size_quote,
                            fee=fee,
                            signal_score=signal_result.score,
                            stop_loss_price=size_result.stop_loss_price,
                            take_profit_price=size_result.take_profit_price,
                        )

                        logger.debug(
                            "backtest_buy",
                            timestamp=timestamp,
                            price=str(fill_price),
                            size_base=str(base_received),
                            fee=str(fee),
                        )
                    else:
                        logger.debug(
                            "trade_rejected",
                            reason="insufficient_balance",
                            timestamp=timestamp,
                            required=str(total_cost),
                            available=str(quote_balance),
                        )
                else:
                    logger.debug(
                        "trade_rejected",
                        reason="below_minimum_size",
                        timestamp=timestamp,
                        size_quote=str(size_result.size_quote),
                        min_size=str(self.min_trade_size),
                    )

            elif signal_result.action == "sell" and open_position is not None:
                # Close position on sell signal
                # NOTE: Fill price uses current candle's close price - slippage (see buy logic for details)
                fill_price = current_price * (Decimal("1") - self.slippage_percent)

                # Calculate proceeds and fee
                gross_proceeds = base_balance * fill_price
                fee = gross_proceeds * self.fee_percent
                net_proceeds = gross_proceeds - fee

                # Execute sell
                quote_balance += net_proceeds
                total_fees += fee

                # Calculate P&L
                pnl = gross_proceeds - open_position.size_quote - fee - open_position.fee
                pnl_percent = (pnl / open_position.size_quote) * Decimal("100")

                # Update and record trade
                open_position.exit_price = fill_price
                open_position.exit_timestamp = timestamp
                open_position.exit_reason = "signal"
                open_position.pnl = pnl
                open_position.pnl_percent = pnl_percent
                trades.append(open_position)

                logger.debug(
                    "backtest_sell",
                    timestamp=timestamp,
                    price=str(fill_price),
                    size_base=str(base_balance),
                    pnl=str(pnl),
                    pnl_percent=str(pnl_percent),
                )

                base_balance = Decimal("0")
                open_position = None

            # Record equity
            portfolio_value = quote_balance + (base_balance * current_price)
            equity_curve.append({
                "timestamp": timestamp,
                "portfolio_value": float(portfolio_value),
            })

        # Close any remaining position at final price
        if open_position is not None:
            final_price = Decimal(str(df.iloc[-1]["close"]))
            final_timestamp = df.iloc[-1]["timestamp"]

            fill_price = final_price * (Decimal("1") - self.slippage_percent)
            gross_proceeds = base_balance * fill_price
            fee = gross_proceeds * self.fee_percent
            net_proceeds = gross_proceeds - fee

            quote_balance += net_proceeds
            total_fees += fee

            pnl = gross_proceeds - open_position.size_quote - fee - open_position.fee
            pnl_percent = (pnl / open_position.size_quote) * Decimal("100")

            open_position.exit_price = fill_price
            open_position.exit_timestamp = final_timestamp
            open_position.exit_reason = "end_of_backtest"
            open_position.pnl = pnl
            open_position.pnl_percent = pnl_percent
            trades.append(open_position)

            base_balance = Decimal("0")

        # Calculate metrics
        final_value = quote_balance + (base_balance * Decimal(str(df.iloc[-1]["close"])))
        metrics = self._calculate_metrics(
            trades=trades,
            equity_curve_data=equity_curve,
            initial_value=self.initial_capital,
            final_value=final_value,
            total_fees=total_fees,
        )

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)

        result = BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=equity_df,
            signal_distribution=signal_counts,
        )

        logger.info(
            "backtest_complete",
            total_trades=len(trades),
            total_return=f"{metrics.total_return:.2f}%",
            sharpe_ratio=f"{metrics.sharpe_ratio:.2f}",
            max_drawdown=f"{metrics.max_drawdown:.2f}%",
            win_rate=f"{metrics.win_rate:.1f}%",
        )

        return result

    def _calculate_metrics(
        self,
        trades: list[BacktestTrade],
        equity_curve_data: list[dict],
        initial_value: Decimal,
        final_value: Decimal,
        total_fees: Decimal,
    ) -> BacktestMetrics:
        """
        Calculate performance metrics from backtest results.

        Note on Decimal to float conversion:
        This method converts Decimal values to float for metrics calculation. This is
        acceptable because:
        1. Trade execution uses Decimal for precision (critical for money calculations)
        2. Statistical metrics (Sharpe ratio, volatility, etc.) don't require the same
           precision and are typically displayed as rounded values anyway
        3. The conversion happens AFTER all trade calculations are complete, so it
           doesn't affect the integrity of the backtest simulation
        4. Float arithmetic is faster for statistical calculations on large datasets
        """

        if not trades:
            return BacktestMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                avg_trade_duration_hours=0.0,
                final_value=final_value,
                initial_value=initial_value,
                total_fees=total_fees,
            )

        # Total return
        total_return = float((final_value - initial_value) / initial_value * 100)

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0.0

        # Profit factor (uses absolute P&L, not percentages)
        # A $100 profit on a $1000 trade (10%) is more significant than
        # a $10 profit on a $50 trade (20%) for portfolio performance
        gross_profit = sum(float(t.pnl or 0) for t in winning_trades)
        gross_loss = abs(sum(float(t.pnl or 0) for t in losing_trades))
        # If no losses but profits exist, use large finite value; if both zero, use 0.0
        # Using 999.0 instead of infinity to avoid JSON serialization issues and
        # ensure compatibility with downstream calculations
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0

        # Average win/loss (in percentage terms for readability)
        gross_profit_pct = sum(float(t.pnl_percent or 0) for t in winning_trades)
        gross_loss_pct = abs(sum(float(t.pnl_percent or 0) for t in losing_trades))
        avg_win = (gross_profit_pct / len(winning_trades)) if winning_trades else 0.0
        avg_loss = (gross_loss_pct / len(losing_trades)) if losing_trades else 0.0

        # Average trade duration
        durations = []
        for t in trades:
            if t.exit_timestamp:
                duration = (t.exit_timestamp - t.timestamp).total_seconds() / 3600
                durations.append(duration)
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # Equity curve analysis
        equity_df = pd.DataFrame(equity_curve_data)
        if not equity_df.empty:
            # Calculate returns
            equity_df["returns"] = equity_df["portfolio_value"].pct_change()

            # Calculate annualization factor from actual data frequency
            time_diffs = equity_df["timestamp"].diff()
            avg_period_seconds = time_diffs.dt.total_seconds().median()
            if avg_period_seconds <= 0:
                raise ValueError(
                    f"Cannot determine data frequency for annualization: avg_period_seconds={avg_period_seconds}. "
                    "This indicates invalid or constant timestamps in the equity curve."
                )
            periods_per_year = (365.25 * 24 * 3600) / avg_period_seconds

            # Sharpe ratio (annualized using actual data frequency)
            # NOTE: This is a simplified Sharpe ratio assuming 0% risk-free rate.
            # For crypto trading, this is reasonable as there's no true "risk-free"
            # alternative (even stablecoins carry risk). For traditional assets,
            # consider subtracting the risk-free rate (e.g., T-bill yield) from mean_return.
            # Formula: (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)
            mean_return = equity_df["returns"].mean()
            std_return = equity_df["returns"].std()
            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * (periods_per_year ** 0.5)
            else:
                sharpe_ratio = 0.0

            # Volatility (annualized using actual data frequency)
            volatility = std_return * (periods_per_year ** 0.5) * 100

            # Max drawdown
            # NOTE: This calculates drawdown using close prices only. Actual intra-candle
            # drawdown could be worse if we considered high/low prices. For example, a flash
            # crash to the candle low followed by recovery to the close would not show the
            # full drawdown magnitude. This is acceptable for candle-close trading strategies
            # but may underestimate risk for stop-loss placement.
            equity_df["peak"] = equity_df["portfolio_value"].cummax()
            equity_df["drawdown"] = (
                (equity_df["portfolio_value"] - equity_df["peak"]) / equity_df["peak"] * 100
            )
            max_drawdown = abs(equity_df["drawdown"].min())
        else:
            sharpe_ratio = 0.0
            volatility = 0.0
            max_drawdown = 0.0

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration_hours=avg_duration,
            final_value=final_value,
            initial_value=initial_value,
            total_fees=total_fees,
        )
