"""
Historical backtesting module for strategy validation.

Allows testing SignalScorer and trading strategies against historical
market data to validate performance before live/paper deployment.
"""

from src.backtest.backtester import Backtester, BacktestResult, BacktestMetrics

__all__ = ["Backtester", "BacktestResult", "BacktestMetrics"]
