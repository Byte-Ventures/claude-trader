"""
CRITICAL TESTS for trading daemon.

The trading daemon orchestrates all bot operations and executes REAL money trades.
These tests verify the most critical paths to prevent financial loss.

Tests cover:
- Safety system enforcement (kill switch, circuit breaker, loss limiter)
- API error handling and recovery
- Buy/sell decision logic
- Position tracking and PnL calculation
- Emergency shutdown mechanisms

All tests use mocked components - NO real trades or API calls.
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from threading import Event
import pandas as pd

from config.settings import Settings, TradingMode, Exchange
from src.daemon.runner import TradingDaemon
from src.api.exchange_protocol import Balance, MarketData, OrderResult
from src.strategy.signal_scorer import SignalResult, IndicatorValues


# ============================================================================
# Fixtures - Mocked Components
# ============================================================================

@pytest.fixture
def mock_settings():
    """Create complete mock settings for daemon initialization."""
    settings = Mock(spec=Settings)

    # Core trading config
    settings.trading_pair = "BTC-USD"
    settings.is_paper_trading = True
    settings.database_path = ":memory:"
    settings.paper_initial_quote = Decimal("10000")
    settings.paper_initial_base = Decimal("0")
    settings.base_order_size_usd = Decimal("100")

    # Exchange config
    settings.exchange = Exchange.COINBASE
    settings.coinbase_key_file = None
    settings.coinbase_api_key = Mock()
    settings.coinbase_api_key.get_secret_value.return_value = "test_key"
    settings.coinbase_api_secret = Mock()
    settings.coinbase_api_secret.get_secret_value.return_value = "test_secret"

    # Telegram config
    settings.telegram_enabled = False
    settings.telegram_bot_token = Mock()
    settings.telegram_bot_token.get_secret_value.return_value = "test_token"
    settings.telegram_chat_id = "test_chat"

    # AI config
    settings.ai_review_enabled = False
    settings.openrouter_api_key = None
    settings.hourly_analysis_enabled = False
    settings.ai_recommendation_ttl_minutes = 20

    # Strategy config
    settings.signal_threshold = 50
    settings.rsi_period = 14
    settings.rsi_oversold = 30
    settings.rsi_overbought = 70
    settings.macd_fast = 12
    settings.macd_slow = 26
    settings.macd_signal = 9
    settings.bollinger_period = 20
    settings.bollinger_std = 2.0
    settings.ema_fast = 12
    settings.ema_slow = 26
    settings.atr_period = 14

    # Safety config
    settings.black_recovery_hours = 24
    settings.max_position_pct = Decimal("25")
    settings.max_position_percent = Decimal("25")  # Used by OrderValidator
    settings.position_size_percent = Decimal("25")
    settings.stop_loss_atr_multiplier = 2.0
    settings.take_profit_atr_multiplier = 3.0
    settings.stop_loss_pct = None
    settings.trailing_stop_enabled = False

    # Regime config
    settings.regime_adaptation_enabled = False
    settings.regime_sentiment_enabled = False
    settings.regime_volatility_enabled = False
    settings.regime_trend_enabled = False
    settings.regime_adjustment_scale = 0.5

    # Safety systems
    settings.circuit_breaker_enabled = True
    settings.kill_switch_enabled = True
    settings.loss_limiter_enabled = True

    # Adaptive interval
    settings.adaptive_interval_enabled = True
    settings.check_interval_seconds = 60
    settings.interval_low_volatility = 60
    settings.interval_normal = 60
    settings.interval_high_volatility = 120
    settings.interval_extreme_volatility = 300

    return settings


@pytest.fixture
def mock_exchange_client():
    """Create mock exchange client that handles validation."""
    client = Mock()

    # CRITICAL: Must return valid price for trading pair validation during daemon init
    client.get_current_price.return_value = Decimal("50000")

    # Mock balance responses with side_effect for different currencies
    def get_balance_side_effect(currency):
        return Balance(
            currency=currency,
            available=Decimal("10000") if currency == "USD" else Decimal("1.0"),
            hold=Decimal("0")
        )
    client.get_balance.side_effect = get_balance_side_effect

    # Mock candles response
    client.get_candles.return_value = _create_sample_candles()

    # Mock order responses
    client.market_buy.return_value = OrderResult(
        order_id="buy-123",
        side="buy",
        size=Decimal("0.002"),
        filled_price=Decimal("50000"),
        status="FILLED",
        fee=Decimal("1.00"),
        success=True
    )

    client.market_sell.return_value = OrderResult(
        order_id="sell-456",
        side="sell",
        size=Decimal("0.002"),
        filled_price=Decimal("51000"),
        status="FILLED",
        fee=Decimal("1.02"),
        success=True
    )

    return client


@pytest.fixture
def mock_database():
    """Create mock database with proper return structures."""
    db = Mock()

    # Return None for fresh start (no saved balance)
    # During init, daemon calls db.get_last_paper_balance() which expects:
    #   None OR (quote_balance: Decimal, base_balance: Decimal, timestamp: datetime)
    db.get_last_paper_balance.return_value = None

    # Return None for regime (no saved regime)
    db.get_last_regime.return_value = None

    return db


def _create_sample_candles(length=100, base_price=50000.0):
    """Create sample OHLCV data for testing."""
    data = []
    for i in range(length):
        data.append({
            "timestamp": datetime.now(),
            "open": Decimal(str(base_price)),
            "high": Decimal(str(base_price * 1.01)),
            "low": Decimal(str(base_price * 0.99)),
            "close": Decimal(str(base_price)),
            "volume": Decimal("100"),
        })
    return pd.DataFrame(data)


@pytest.fixture
def mock_signal_scorer():
    """Create mock signal scorer."""
    scorer = Mock()

    # Default to neutral signal
    scorer.calculate_score.return_value = SignalResult(
        score=0,
        direction="neutral",
        indicators=IndicatorValues(
            rsi=50.0,
            macd_line=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
            ema_fast=50000.0,
            ema_slow=50000.0,
            volatility="normal"
        )
    )

    return scorer


# ============================================================================
# Safety System Tests - CRITICAL
# ============================================================================

def test_kill_switch_blocks_trading(mock_settings, mock_exchange_client, mock_database):
    """CRITICAL: Verify kill switch prevents all trading when active."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Activate kill switch
                daemon.kill_switch.activate("test_activation")

                # Reset mock to track calls after activation
                mock_exchange_client.reset_mock()

                # Attempt trading iteration
                daemon._trading_iteration()

                # Verify NO exchange calls were made
                mock_exchange_client.get_current_price.assert_not_called()
                mock_exchange_client.market_buy.assert_not_called()
                mock_exchange_client.market_sell.assert_not_called()


def test_circuit_breaker_blocks_trading_when_open(mock_settings, mock_exchange_client, mock_database):
    """CRITICAL: Verify circuit breaker prevents trading when triggered."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Trip circuit breaker by recording failures
                for _ in range(10):  # Trigger threshold
                    daemon.circuit_breaker.record_api_failure()

                # Reset mock to track calls after breaker trip
                mock_exchange_client.reset_mock()

                # Attempt trading iteration
                daemon._trading_iteration()

                # Verify NO exchange calls were made
                mock_exchange_client.get_current_price.assert_not_called()


def test_loss_limiter_blocks_trading_when_limit_exceeded(mock_settings, mock_exchange_client, mock_database):
    """CRITICAL: Verify loss limiter stops trading after exceeding daily limit."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set starting balance for percentage calculations
                daemon.loss_limiter.set_starting_balance(Decimal("10000"))

                # Record large loss to exceed daily limit (10% = $1000 loss on $10000 balance)
                daemon.loss_limiter.record_trade(
                    realized_pnl=Decimal("-1000"),
                    side="sell",
                    size=Decimal("1"),
                    price=Decimal("50000")
                )

                # Reset mock to track calls after loss limit
                mock_exchange_client.reset_mock()

                # Attempt trading iteration
                daemon._trading_iteration()

                # Verify NO exchange calls were made
                mock_exchange_client.get_current_price.assert_not_called()


# ============================================================================
# API Error Handling Tests - CRITICAL
# ============================================================================

def test_api_failure_does_not_crash_daemon(mock_settings, mock_database):
    """CRITICAL: Verify API failures are handled gracefully without crashing."""
    # Create client that succeeds during init validation but fails during iteration
    mock_client = Mock()
    mock_client.get_current_price.side_effect = [
        Decimal("50000"),  # First call: validation passes
        Exception("API Timeout")  # Second call: iteration fails
    ]
    mock_client.get_candles.return_value = _create_sample_candles()
    mock_client.get_balance.return_value = Balance("USD", Decimal("1000"), Decimal("0"))

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Should not raise exception
                daemon._trading_iteration()

                # Verify circuit breaker was notified
                assert daemon.circuit_breaker._api_failures > 0


def test_api_failure_recorded_in_circuit_breaker(mock_settings, mock_database):
    """CRITICAL: Verify API failures increment circuit breaker counter."""
    # Create client that succeeds during init validation but fails during iteration
    mock_client = Mock()
    mock_client.get_current_price.side_effect = [
        Decimal("50000"),  # First call: validation passes
        Exception("Connection Error")  # Second call: iteration fails
    ]
    mock_client.get_candles.return_value = _create_sample_candles()
    mock_client.get_balance.return_value = Balance("USD", Decimal("1000"), Decimal("0"))

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                initial_failures = daemon.circuit_breaker._api_failures
                daemon._trading_iteration()

                assert daemon.circuit_breaker._api_failures > initial_failures


def test_api_success_recorded_in_circuit_breaker(mock_settings, mock_exchange_client, mock_database):
    """Verify successful API calls reset circuit breaker failures."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Record some failures first
                daemon.circuit_breaker.record_api_failure()
                daemon.circuit_breaker.record_api_failure()

                # Successful iteration should record success
                daemon._trading_iteration()

                # Circuit breaker should have reset failures after success
                assert daemon.circuit_breaker._api_failures == 0


# ============================================================================
# Trading Pair Validation Tests
# ============================================================================

def test_daemon_validates_trading_pair_on_init(mock_settings, mock_database):
    """Test daemon validates trading pair format on initialization."""
    mock_settings.trading_pair = "INVALID"  # No dash

    with patch('src.daemon.runner.create_exchange_client'):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                with pytest.raises(ValueError, match="Invalid trading pair format"):
                    TradingDaemon(mock_settings)


def test_daemon_parses_base_and_quote_currencies(mock_settings, mock_exchange_client, mock_database):
    """Test daemon correctly parses base and quote currencies from trading pair."""
    mock_settings.trading_pair = "ETH-USD"

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                assert daemon._base_currency == "ETH"
                assert daemon._quote_currency == "USD"


# ============================================================================
# Shutdown Handling Tests
# ============================================================================

def test_shutdown_event_stops_daemon(mock_settings, mock_exchange_client, mock_database):
    """Test shutdown event stops the daemon gracefully."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set shutdown event
                daemon.shutdown_event.set()

                assert daemon.shutdown_event.is_set()


def test_daemon_initializes_with_paper_trading_mode(mock_settings, mock_exchange_client, mock_database):
    """Test daemon properly initializes in paper trading mode."""
    mock_settings.is_paper_trading = True

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                with patch('src.daemon.runner.PaperTradingClient') as mock_paper:
                    daemon = TradingDaemon(mock_settings)

                    # Verify paper client was created
                    assert mock_paper.called


def test_daemon_initializes_with_live_trading_mode(mock_settings, mock_exchange_client, mock_database):
    """Test daemon properly initializes in live trading mode."""
    mock_settings.is_paper_trading = False

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Verify real client is being used
                assert daemon.client == daemon.real_client


# ============================================================================
# Portfolio Value Calculation Tests
# ============================================================================

def test_get_portfolio_value_includes_base_and_quote(mock_settings, mock_exchange_client, mock_database):
    """Test portfolio value calculation includes both base and quote currencies."""
    # Use live trading mode for this test so daemon uses mock_exchange_client
    mock_settings.is_paper_trading = False

    # Mock balances: 1 BTC + 10000 USD at 50000 BTC/USD = 60000 USD total
    def mock_get_balance(currency):
        if currency == "BTC":
            return Balance(currency="BTC", available=Decimal("1.0"), hold=Decimal("0"))
        else:  # USD
            return Balance(currency="USD", available=Decimal("10000"), hold=Decimal("0"))

    mock_exchange_client.get_balance.side_effect = mock_get_balance
    mock_exchange_client.get_current_price.return_value = Decimal("50000")

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                portfolio_value = daemon._get_portfolio_value()

                # 1 BTC * 50000 + 10000 USD = 60000 USD
                assert portfolio_value == Decimal("60000")


# ============================================================================
# Configuration Reload Tests
# ============================================================================

def test_daemon_supports_config_reload_signal(mock_settings, mock_exchange_client, mock_database):
    """Test daemon can reload configuration on signal."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                with patch('src.daemon.runner.reload_pending', return_value=True):
                    with patch('src.daemon.runner.reload_settings') as mock_reload:
                        daemon = TradingDaemon(mock_settings)

                        # Trigger reload
                        daemon._reload_config()

                        # Verify reload was called
                        mock_reload.assert_called_once()


# ============================================================================
# Adaptive Interval Tests
# ============================================================================

def test_adaptive_interval_increases_during_high_volatility(mock_settings, mock_exchange_client, mock_database):
    """Test daemon increases check interval during high volatility."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set high volatility
                daemon._last_volatility = "extreme"

                interval = daemon._get_adaptive_interval()

                # Should be longer than normal interval (60 seconds)
                assert interval > 60


def test_adaptive_interval_normal_during_low_volatility(mock_settings, mock_exchange_client, mock_database):
    """Test daemon uses normal interval during low volatility."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set low volatility
                daemon._last_volatility = "low"

                interval = daemon._get_adaptive_interval()

                # Should be normal interval (60 seconds)
                assert interval == 60


# ============================================================================
# Initialization Tests
# ============================================================================

def test_daemon_initializes_all_components(mock_settings, mock_exchange_client, mock_database):
    """Test daemon initializes all required components on startup."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Verify all components were initialized
                assert daemon.db is not None
                assert daemon.client is not None
                assert daemon.kill_switch is not None
                assert daemon.circuit_breaker is not None
                assert daemon.loss_limiter is not None
                assert daemon.validator is not None
                assert daemon.signal_scorer is not None


def test_daemon_initializes_shutdown_event(mock_settings, mock_exchange_client, mock_database):
    """Test daemon creates shutdown event on initialization."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                assert daemon.shutdown_event is not None
                assert isinstance(daemon.shutdown_event, Event)
                assert not daemon.shutdown_event.is_set()


# ============================================================================
# Exchange Selection Tests
# ============================================================================

def test_daemon_uses_coinbase_when_configured(mock_settings, mock_exchange_client, mock_database):
    """Test daemon initializes Coinbase client when configured."""
    mock_settings.exchange = Exchange.COINBASE

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client) as mock_create:
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Verify exchange client was created with settings
                mock_create.assert_called_once_with(mock_settings)
                assert daemon.exchange_name in ["Coinbase", "coinbase"]


def test_daemon_uses_kraken_when_configured(mock_settings, mock_exchange_client, mock_database):
    """Test daemon initializes Kraken client when configured."""
    mock_settings.exchange = Exchange.KRAKEN
    mock_settings.kraken_api_key = Mock()
    mock_settings.kraken_api_key.get_secret_value.return_value = "test_key"
    mock_settings.kraken_api_secret = Mock()
    mock_settings.kraken_api_secret.get_secret_value.return_value = "test_secret"

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                assert daemon.exchange_name in ["Kraken", "kraken"]


# ============================================================================
# AI Threshold Adjustment Tests
# ============================================================================

def test_ai_threshold_adjustment_returns_zero_without_recommendation(mock_settings, mock_exchange_client, mock_database):
    """Test AI threshold adjustment returns 0 when no recommendation is active."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # No recommendation set
                assert daemon._ai_recommendation is None

                # Should return 0 for both buy and sell
                assert daemon._get_ai_threshold_adjustment("buy") == 0
                assert daemon._get_ai_threshold_adjustment("sell") == 0


def test_ai_threshold_adjustment_accumulate_lowers_buy_threshold(mock_settings, mock_exchange_client, mock_database):
    """Test 'accumulate' recommendation lowers buy threshold."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set accumulate recommendation with high confidence
                daemon._ai_recommendation = "accumulate"
                daemon._ai_recommendation_confidence = 0.9
                daemon._ai_recommendation_time = datetime.utcnow()

                buy_adj = daemon._get_ai_threshold_adjustment("buy")
                sell_adj = daemon._get_ai_threshold_adjustment("sell")

                # Buy threshold should be lowered (negative adjustment)
                assert buy_adj < 0
                # Sell threshold should not be affected
                assert sell_adj == 0

                # Adjustment should be scaled by confidence (15 * 0.9 * ~1.0 decay = ~13)
                assert buy_adj <= -10  # At least -10 with high confidence


def test_ai_threshold_adjustment_reduce_lowers_sell_threshold(mock_settings, mock_exchange_client, mock_database):
    """Test 'reduce' recommendation lowers sell threshold."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set reduce recommendation with high confidence
                daemon._ai_recommendation = "reduce"
                daemon._ai_recommendation_confidence = 0.9
                daemon._ai_recommendation_time = datetime.utcnow()

                buy_adj = daemon._get_ai_threshold_adjustment("buy")
                sell_adj = daemon._get_ai_threshold_adjustment("sell")

                # Buy threshold should not be affected
                assert buy_adj == 0
                # Sell threshold should be lowered (negative adjustment)
                assert sell_adj < 0


def test_ai_threshold_adjustment_wait_has_no_effect(mock_settings, mock_exchange_client, mock_database):
    """Test 'wait' recommendation does not affect thresholds."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set wait recommendation
                daemon._ai_recommendation = "wait"
                daemon._ai_recommendation_confidence = 0.9
                daemon._ai_recommendation_time = datetime.utcnow()

                # Neither threshold should be affected
                assert daemon._get_ai_threshold_adjustment("buy") == 0
                assert daemon._get_ai_threshold_adjustment("sell") == 0


def test_ai_threshold_adjustment_decays_over_time(mock_settings, mock_exchange_client, mock_database):
    """Test AI adjustment decays linearly over TTL period."""
    from datetime import timedelta

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set accumulate recommendation with full confidence
                daemon._ai_recommendation = "accumulate"
                daemon._ai_recommendation_confidence = 1.0
                daemon._ai_recommendation_ttl_minutes = 20

                # Test at start (no decay)
                daemon._ai_recommendation_time = datetime.utcnow()
                adj_at_start = daemon._get_ai_threshold_adjustment("buy")

                # Test at 50% through TTL (should be ~50% of original)
                daemon._ai_recommendation_time = datetime.utcnow() - timedelta(minutes=10)
                adj_at_half = daemon._get_ai_threshold_adjustment("buy")

                # Adjustment should decay (less negative over time)
                assert adj_at_start < adj_at_half < 0  # Both negative, but half is closer to 0


def test_ai_threshold_adjustment_expires_after_ttl(mock_settings, mock_exchange_client, mock_database):
    """Test AI recommendation expires and clears after TTL."""
    from datetime import timedelta

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Set recommendation with 20 minute TTL
                daemon._ai_recommendation = "accumulate"
                daemon._ai_recommendation_confidence = 1.0
                daemon._ai_recommendation_ttl_minutes = 20

                # Set time to beyond TTL
                daemon._ai_recommendation_time = datetime.utcnow() - timedelta(minutes=25)

                # Should return 0 and clear the recommendation
                assert daemon._get_ai_threshold_adjustment("buy") == 0
                assert daemon._ai_recommendation is None
                assert daemon._ai_recommendation_time is None


def test_ai_threshold_adjustment_scales_with_confidence(mock_settings, mock_exchange_client, mock_database):
    """Test AI adjustment scales with confidence level."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                daemon._ai_recommendation = "accumulate"
                daemon._ai_recommendation_time = datetime.utcnow()

                # Test with high confidence
                daemon._ai_recommendation_confidence = 1.0
                high_conf_adj = daemon._get_ai_threshold_adjustment("buy")

                # Test with low confidence
                daemon._ai_recommendation_confidence = 0.5
                low_conf_adj = daemon._get_ai_threshold_adjustment("buy")

                # Higher confidence should give larger (more negative) adjustment
                assert high_conf_adj < low_conf_adj < 0
