"""
Example demonstrating historical backtesting with SignalScorer.

This script shows how to:
1. Load historical OHLCV data
2. Configure a trading strategy
3. Run a backtest
4. Analyze performance metrics
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.backtest import Backtester
from src.strategy.signal_scorer import SignalScorer
from src.strategy.position_sizer import PositionSizer, PositionSizeConfig


def generate_sample_data(days: int = 30) -> pd.DataFrame:
    """
    Generate sample OHLCV data for demonstration.

    In production, load actual historical data from CSV or exchange API.
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data = []

    base_price = 50000.0
    for i in range(days * 24):  # Hourly candles
        timestamp = start + timedelta(hours=i)
        # Simulate price movement with trend and volatility
        trend = i * 50  # Upward trend
        volatility = 200 * ((i % 24) - 12)  # Intraday volatility
        price = base_price + trend + volatility

        data.append({
            "timestamp": timestamp,
            "open": price,
            "high": price + 100,
            "low": price - 100,
            "close": price + (50 if i % 2 == 0 else -50),
            "volume": 1000000 + (i * 10000),
        })

    return pd.DataFrame(data)


def main():
    """Run backtest example."""

    # 1. Configure strategy
    signal_scorer = SignalScorer(
        threshold=60,  # Signal threshold
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        ema_fast=9,
        ema_slow=21,
        candle_interval="ONE_HOUR",
    )

    position_sizer = PositionSizer(
        config=PositionSizeConfig(
            risk_per_trade_percent=1.0,  # Risk 1% per trade
            min_trade_base=0.0001,
            max_position_percent=40.0,
            min_trade_quote=10.0,
        )
    )

    # 2. Create backtester
    backtester = Backtester(
        signal_scorer=signal_scorer,
        position_sizer=position_sizer,
        initial_capital=10000.0,
        fee_percent=0.006,  # 0.6% (Coinbase Advanced)
        slippage_percent=0.001,  # 0.1%
    )

    # 3. Load data
    print("Loading historical data...")
    historical_data = generate_sample_data(days=30)
    print(f"Loaded {len(historical_data)} candles")

    # 4. Run backtest
    print("\nRunning backtest...")
    result = backtester.run(
        data=historical_data,
        start_date="2024-01-03",  # Skip first few candles for indicators
    )

    # 5. Display results
    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # 6. Show trade details
    if result.trades:
        print("\nTrade Details:")
        print("-" * 60)
        for i, trade in enumerate(result.trades[:5], 1):  # Show first 5 trades
            print(f"\nTrade #{i}:")
            print(f"  Entry: {trade.timestamp.strftime('%Y-%m-%d %H:%M')} @ ${trade.entry_price:,.2f}")
            if trade.exit_timestamp:
                print(f"  Exit:  {trade.exit_timestamp.strftime('%Y-%m-%d %H:%M')} @ ${trade.exit_price:,.2f}")
                print(f"  P&L:   ${trade.pnl:,.2f} ({trade.pnl_percent:+.2f}%)")
                duration = (trade.exit_timestamp - trade.timestamp).total_seconds() / 3600
                print(f"  Duration: {duration:.1f} hours")

        if len(result.trades) > 5:
            print(f"\n... and {len(result.trades) - 5} more trades")

    # 7. Signal distribution
    print("\nSignal Analysis:")
    print("-" * 60)
    total_signals = sum(result.signal_distribution.values())
    for action, count in result.signal_distribution.items():
        pct = (count / total_signals * 100) if total_signals > 0 else 0
        print(f"  {action.capitalize():6s}: {count:4d} ({pct:5.1f}%)")

    # 8. Export equity curve (optional)
    # result.equity_curve.to_csv("equity_curve.csv", index=False)
    # print("\nEquity curve exported to equity_curve.csv")


if __name__ == "__main__":
    main()
