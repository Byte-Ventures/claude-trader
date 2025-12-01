"""
Pytest configuration and shared fixtures for claude-trader tests.
"""

import sys
from pathlib import Path
from decimal import Decimal

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Silence structlog during tests
import structlog


def _mock_logger_factory(*args):
    """Factory that creates mock loggers for testing."""
    mock = MagicMock()
    # Configure mock methods to return the mock itself (for chaining)
    mock.bind.return_value = mock
    return mock


structlog.configure(
    processors=[],
    logger_factory=_mock_logger_factory,
)


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def decimal_balance():
    """Decimal balance helper."""
    def _decimal(value: float) -> Decimal:
        return Decimal(str(value))
    return _decimal


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV market data for testing indicators."""
    def _generate(length=100, base_price=100.0, volatility=0.02):
        """
        Generate realistic OHLCV data.

        Args:
            length: Number of candles
            base_price: Starting price
            volatility: Price volatility (0.01 = 1%)
        """
        np.random.seed(42)  # Deterministic for tests

        prices = []
        current = base_price

        for _ in range(length):
            # Random walk with drift
            change = np.random.randn() * volatility * current
            current = current + change
            prices.append(current)

        # Generate OHLCV from prices
        data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }

        for price in prices:
            o = price * (1 + np.random.uniform(-0.005, 0.005))
            c = price * (1 + np.random.uniform(-0.005, 0.005))
            h = max(o, c) * (1 + abs(np.random.uniform(0, 0.01)))
            l = min(o, c) * (1 - abs(np.random.uniform(0, 0.01)))
            v = np.random.uniform(1000, 10000)

            data['open'].append(o)
            data['high'].append(h)
            data['low'].append(l)
            data['close'].append(c)
            data['volume'].append(v)

        return pd.DataFrame(data)

    return _generate
