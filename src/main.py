"""
Bitcoin Trading Bot - Main Entry Point

A Python-based automated trading bot for Coinbase with:
- Multi-indicator confluence strategy (RSI, MACD, Bollinger, EMA, ATR)
- Comprehensive safety systems (kill switch, circuit breaker, loss limits)
- Paper trading mode for testing
- Telegram notifications

Usage:
    python -m src.main

Configuration:
    Copy .env.example to .env and configure your settings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog

from config.logging_config import setup_logging, get_logger
from config.settings import get_settings, TradingMode
from src.daemon.runner import TradingDaemon


def main() -> int:
    """
    Main entry point for the trading bot.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load settings
        settings = get_settings()

        # Setup logging
        setup_logging(
            log_level=settings.log_level,
            log_file=settings.log_file,
        )

        logger = get_logger(__name__)

        # Print startup banner
        mode = "PAPER" if settings.is_paper_trading else "LIVE"
        logger.info(
            "starting_trading_bot",
            mode=mode,
            trading_pair=settings.trading_pair,
            check_interval=settings.check_interval_seconds,
        )

        print("\n" + "=" * 50)
        print("  Bitcoin Trading Bot")
        print("=" * 50)
        print(f"  Mode: {mode}")
        print(f"  Pair: {settings.trading_pair}")
        print(f"  Interval: {settings.check_interval_seconds}s")
        print(f"  Position Size: {settings.position_size_percent}%")
        print(f"  Signal Threshold: {settings.signal_threshold}")
        print("=" * 50)

        if settings.is_live_trading:
            print("\n⚠️  WARNING: LIVE TRADING MODE")
            print("  Real money will be used for trades!")
            print("  Press Ctrl+C to cancel within 5 seconds...")
            import time
            time.sleep(5)

        print("\nStarting trading daemon...")
        print("Press Ctrl+C to stop\n")

        # Create and run daemon
        daemon = TradingDaemon(settings)
        daemon.run()

        return 0

    except KeyboardInterrupt:
        print("\n\nShutdown requested. Exiting...")
        return 0

    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)

        # Try to get logger for error logging
        try:
            logger = get_logger(__name__)
            logger.critical("fatal_startup_error", error=str(e))
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
