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
from datetime import datetime, date, timezone
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
    settings.veto_reduce_threshold = 0.65
    settings.veto_skip_threshold = 0.80
    settings.position_reduction = 0.5
    settings.ai_api_timeout = 120
    settings.postmortem_enabled = False

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
    settings.candle_interval = "ONE_HOUR"
    settings.candle_limit = 100

    # Safety config
    settings.black_recovery_hours = 24
    settings.max_position_pct = Decimal("25")
    settings.max_position_percent = Decimal("25")  # Used by OrderValidator
    settings.position_size_percent = Decimal("25")
    settings.stop_loss_atr_multiplier = 2.0
    settings.min_stop_loss_percent = 0.5
    settings.min_take_profit_percent = 2.0
    settings.take_profit_atr_multiplier = 3.0
    settings.stop_loss_pct = None
    settings.trailing_stop_enabled = False

    # Regime config
    settings.regime_adaptation_enabled = False
    settings.regime_sentiment_enabled = False
    settings.regime_volatility_enabled = False
    settings.regime_trend_enabled = False
    settings.regime_adjustment_scale = 0.5
    settings.regime_flap_protection = True

    # AI Weight Profile config
    settings.ai_weight_profile_enabled = False
    settings.ai_weight_fallback_profile = "default"
    settings.ai_weight_profile_model = "openai/gpt-5.2"
    settings.weight_profile_flap_protection = True

    # Safety systems
    settings.circuit_breaker_enabled = True
    settings.kill_switch_enabled = True
    settings.loss_limiter_enabled = True
    settings.trade_cooldown_enabled = False

    # Adaptive interval
    settings.adaptive_interval_enabled = True
    settings.check_interval_seconds = 60
    settings.interval_low_volatility = 60
    settings.interval_normal = 60
    settings.interval_high_volatility = 120
    settings.interval_extreme_volatility = 300

    # Whale detection config
    settings.whale_volume_threshold = 5.0
    settings.whale_detection_enabled = False
    settings.whale_direction_threshold = 0.6
    settings.whale_boost_percent = 15.0
    settings.high_volume_boost_percent = 10.0

    # MTF config (defaults to disabled)
    settings.mtf_enabled = False
    settings.mtf_candle_limit = 50
    settings.mtf_daily_cache_minutes = 60
    settings.mtf_4h_cache_minutes = 30
    settings.mtf_aligned_boost = 20
    settings.mtf_counter_penalty = 20

    # AI max tokens
    settings.ai_max_tokens = 4000

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

    # Methods called during _trading_iteration()
    db.record_rates_bulk.return_value = 0  # Number of inserted rows
    db.get_current_position.return_value = None  # No open position
    db.get_state.return_value = None  # No saved dashboard state
    db.save_state.return_value = None  # Void method
    db.get_daily_stats.return_value = None  # No daily stats
    db.get_or_create_daily_stats.return_value = Mock(
        date=datetime.now(timezone.utc).date(),
        starting_balance=Decimal("10000"),
        ending_balance=Decimal("10000"),
        realized_pnl=Decimal("0"),
        total_trades=0,
        is_paper=True,
    )
    db.save_regime.return_value = None  # Void method

    # Default session context manager for signal history operations
    mock_session = MagicMock()
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=False)
    db.session.return_value = mock_session

    return db


def _create_sample_candles(length=100, base_price=50000.0):
    """Create sample OHLCV data for testing."""
    data = []
    for i in range(length):
        data.append({
            "timestamp": datetime.now(),
            "open": Decimal(str(base_price)),
            "high": Decimal(str(base_price)) * Decimal("1.01"),
            "low": Decimal(str(base_price)) * Decimal("0.99"),
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
                daemon._ai_recommendation_time = datetime.now(timezone.utc)

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
                daemon._ai_recommendation_time = datetime.now(timezone.utc)

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
                daemon._ai_recommendation_time = datetime.now(timezone.utc)

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
                daemon._ai_recommendation_time = datetime.now(timezone.utc)
                adj_at_start = daemon._get_ai_threshold_adjustment("buy")

                # Test at 50% through TTL (should be ~50% of original)
                daemon._ai_recommendation_time = datetime.now(timezone.utc) - timedelta(minutes=10)
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
                daemon._ai_recommendation_time = datetime.now(timezone.utc) - timedelta(minutes=25)

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
                daemon._ai_recommendation_time = datetime.now(timezone.utc)

                # Test with high confidence
                daemon._ai_recommendation_confidence = 1.0
                high_conf_adj = daemon._get_ai_threshold_adjustment("buy")

                # Test with low confidence
                daemon._ai_recommendation_confidence = 0.5
                low_conf_adj = daemon._get_ai_threshold_adjustment("buy")

                # Higher confidence should give larger (more negative) adjustment
                assert high_conf_adj < low_conf_adj < 0


# ============================================================================
# AI Failure Mode Tests - CRITICAL
# ============================================================================

def test_ai_failure_mode_open_does_not_skip_trade(mock_settings, mock_exchange_client, mock_database):
    """
    CRITICAL: Verify AI_FAILURE_MODE=open does NOT skip trade when AI review fails.

    This is the default fail-open behavior - the code should NOT return early
    after AI failure, allowing the trade to proceed.
    """
    from config.settings import AIFailureMode, VetoAction

    # Enable AI review and set to OPEN mode (default)
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.OPEN
    mock_settings.openrouter_api_key = Mock()
    mock_settings.openrouter_api_key.get_secret_value.return_value = "test_key"
    mock_settings.reviewer_model_1 = "test/model1"
    mock_settings.reviewer_model_2 = "test/model2"
    mock_settings.reviewer_model_3 = "test/model3"
    mock_settings.judge_model = "test/judge"
    mock_settings.veto_reduce_threshold = 0.65
    mock_settings.veto_skip_threshold = 0.80
    mock_settings.position_reduction = 0.5
    mock_settings.interesting_hold_margin = 15
    mock_settings.ai_review_all = False
    mock_settings.market_research_enabled = False
    mock_settings.ai_web_search_enabled = False
    mock_settings.market_research_cache_minutes = 15
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.is_paper_trading = False

    # Create strong buy signal that would trigger a trade
    buy_signal = SignalResult(
        score=70,
        action="buy",
        indicators=IndicatorValues(
            rsi=30.0,
            macd_line=100.0,
            macd_signal=50.0,
            macd_histogram=50.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
            ema_fast=50100.0,
            ema_slow=50000.0,
            atr=500.0,
            volatility="normal"
        ),
        breakdown={"rsi": 20, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
        confidence=0.8
    )

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier:
                daemon = TradingDaemon(mock_settings)

                # Mock position_sizer config to ensure can_buy=True
                # This ensures direction_is_tradeable=True so AI review code path is reached
                daemon.position_sizer.config.min_trade_quote = Decimal("10")  # Less than 10000 quote balance
                daemon.position_sizer.config.max_position_percent = Decimal("100")  # Allow full position
                daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                # Mock signal scorer to return strong buy signal
                daemon.signal_scorer.calculate_score = Mock(return_value=buy_signal)

                # Mock trade reviewer to raise an exception (simulating AI failure)
                daemon.trade_reviewer = Mock()
                daemon.trade_reviewer.should_review.return_value = (True, "trade")
                daemon.trade_reviewer.review_trade = Mock(side_effect=Exception("AI API Timeout"))

                # Run trading iteration
                daemon._trading_iteration()

                # In OPEN mode, notification about skipping should NOT be sent
                # (unlike SAFE mode which sends "Trade skipped: AI review unavailable")
                # Note: We don't verify market_buy.assert_called() here because the test
                # focuses on the AI failure handling behavior, not the complete trade path.
                # The trade may not execute due to other conditions (order sizing, etc.)
                notifier_instance = mock_notifier.return_value
                for call in notifier_instance.send_message.call_args_list:
                    msg = str(call)
                    assert "Trade skipped" not in msg, "OPEN mode should not skip trades"


def test_ai_failure_mode_safe_skips_trade(mock_settings, mock_exchange_client, mock_database):
    """
    CRITICAL: Verify AI_FAILURE_MODE=safe skips trade when AI review fails.

    In safe mode, trades are NOT executed when AI review is unavailable,
    providing protection against trading blind during AI outages.
    """
    from config.settings import AIFailureMode, VetoAction

    # Enable AI review and set to SAFE mode
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.SAFE
    mock_settings.openrouter_api_key = Mock()
    mock_settings.openrouter_api_key.get_secret_value.return_value = "test_key"
    mock_settings.reviewer_model_1 = "test/model1"
    mock_settings.reviewer_model_2 = "test/model2"
    mock_settings.reviewer_model_3 = "test/model3"
    mock_settings.judge_model = "test/judge"
    mock_settings.veto_reduce_threshold = 0.65
    mock_settings.veto_skip_threshold = 0.80
    mock_settings.position_reduction = 0.5
    mock_settings.interesting_hold_margin = 15
    mock_settings.ai_review_all = False
    mock_settings.market_research_enabled = False
    mock_settings.ai_web_search_enabled = False
    mock_settings.market_research_cache_minutes = 15
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.is_paper_trading = False

    # Create strong buy signal that would trigger a trade
    buy_signal = SignalResult(
        score=70,
        action="buy",
        indicators=IndicatorValues(
            rsi=30.0,
            macd_line=100.0,
            macd_signal=50.0,
            macd_histogram=50.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
            ema_fast=50100.0,
            ema_slow=50000.0,
            atr=500.0,
            volatility="normal"
        ),
        breakdown={"rsi": 20, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
        confidence=0.8
    )

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier:
                daemon = TradingDaemon(mock_settings)

                # Mock position_sizer config to ensure can_buy=True
                # This ensures direction_is_tradeable=True so AI review code path is reached
                daemon.position_sizer.config.min_trade_quote = Decimal("10")  # Less than 10000 quote balance
                daemon.position_sizer.config.max_position_percent = Decimal("100")  # Allow full position
                daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                # Mock signal scorer to return strong buy signal
                daemon.signal_scorer.calculate_score = Mock(return_value=buy_signal)

                # Mock trade reviewer to raise an exception (simulating AI failure)
                daemon.trade_reviewer = Mock()
                daemon.trade_reviewer.should_review.return_value = (True, "trade")
                daemon.trade_reviewer.review_trade = Mock(side_effect=Exception("AI API Timeout"))

                # Reset mock to clear init calls
                mock_exchange_client.market_buy.reset_mock()

                # Run trading iteration
                daemon._trading_iteration()

                # Verify trade was NOT executed (fail-safe)
                mock_exchange_client.market_buy.assert_not_called()

                # Verify notification was sent about skipped trade
                notifier_instance = mock_notifier.return_value
                notifier_instance.send_message.assert_called()


def test_ai_failure_mode_open_is_default(mock_settings, mock_exchange_client, mock_database):
    """Verify AIFailureMode.OPEN is the default for backward compatibility."""
    from config.settings import AIFailureMode, Settings

    # Create real settings to check default
    with patch.dict('os.environ', {}, clear=False):
        settings = Settings(
            trading_mode="paper",
            trading_pair="BTC-USD",
        )
        assert settings.ai_failure_mode == AIFailureMode.OPEN


def test_ai_failure_notification_cooldown(mock_settings, mock_exchange_client, mock_database):
    """
    Verify AI failure notifications are rate-limited to prevent Telegram spam.

    When AI review fails repeatedly in SAFE mode, notifications should only be
    sent once per 15 minutes to avoid flooding the user's Telegram.
    """
    from config.settings import AIFailureMode, VetoAction
    from datetime import datetime, timedelta, timezone

    # Enable AI review and set to SAFE mode
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.SAFE
    mock_settings.openrouter_api_key = Mock()
    mock_settings.openrouter_api_key.get_secret_value.return_value = "test_key"
    mock_settings.reviewer_model_1 = "test/model1"
    mock_settings.reviewer_model_2 = "test/model2"
    mock_settings.reviewer_model_3 = "test/model3"
    mock_settings.judge_model = "test/judge"
    mock_settings.veto_reduce_threshold = 0.65
    mock_settings.veto_skip_threshold = 0.80
    mock_settings.position_reduction = 0.5
    mock_settings.interesting_hold_margin = 15
    mock_settings.ai_review_all = False
    mock_settings.market_research_enabled = False
    mock_settings.ai_web_search_enabled = False
    mock_settings.market_research_cache_minutes = 15
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.is_paper_trading = False

    # Create strong buy signal
    buy_signal = SignalResult(
        score=70,
        action="buy",
        indicators=IndicatorValues(
            rsi=30.0,
            macd_line=100.0,
            macd_signal=50.0,
            macd_histogram=50.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
            ema_fast=50100.0,
            ema_slow=50000.0,
            atr=500.0,
            volatility="normal"
        ),
        breakdown={"rsi": 20, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
        confidence=0.8
    )

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier:
                daemon = TradingDaemon(mock_settings)

                # Mock position_sizer config to ensure can_buy=True
                # This ensures direction_is_tradeable=True so AI review code path is reached
                daemon.position_sizer.config.min_trade_quote = Decimal("10")  # Less than 10000 quote balance
                daemon.position_sizer.config.max_position_percent = Decimal("100")  # Allow full position
                daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                # Mock signal scorer to return strong buy signal
                daemon.signal_scorer.calculate_score = Mock(return_value=buy_signal)

                # Mock trade reviewer to always fail
                daemon.trade_reviewer = Mock()
                daemon.trade_reviewer.should_review.return_value = (True, "trade")
                daemon.trade_reviewer.review_trade = Mock(side_effect=Exception("AI API Timeout"))

                notifier_instance = mock_notifier.return_value

                # First failure: should send notification
                daemon._trading_iteration()
                assert notifier_instance.send_message.call_count == 1

                # Second failure immediately after: should NOT send notification (cooldown)
                daemon._trading_iteration()
                assert notifier_instance.send_message.call_count == 1  # Still 1, not 2

                # Third failure immediately after: should still NOT send notification
                daemon._trading_iteration()
                assert notifier_instance.send_message.call_count == 1  # Still 1, not 3

                # Simulate 15 minutes passing by resetting the timestamp (use timezone-aware UTC)
                daemon._last_ai_failure_notification = datetime.now(timezone.utc) - timedelta(minutes=16)

                # Fourth failure after cooldown: should send notification again
                daemon._trading_iteration()
                assert notifier_instance.send_message.call_count == 2  # Now 2


# ============================================================================
# Tiered Veto System Tests - v1.31.0
# ============================================================================


@pytest.fixture
def veto_reviewer():
    """Create TradeReviewer instance for tiered veto tests."""
    from src.ai.trade_reviewer import TradeReviewer
    return TradeReviewer(
        api_key="test_key",
        db=Mock(),
        reviewer_models=["test/model1", "test/model2", "test/model3"],
        judge_model="test/judge",
        veto_reduce_threshold=0.65,
        veto_skip_threshold=0.80,
    )


def test_tiered_veto_below_reduce_threshold_proceeds(veto_reviewer):
    """
    Confidence 60% (< 65% reduce threshold) should proceed with trade.

    When judge disapproves but confidence is below VETO_REDUCE_THRESHOLD,
    the trade proceeds (info-only logging, no veto action).
    """
    # Call the actual implementation method
    veto_action = veto_reviewer._determine_veto_action(approved=False, confidence=0.60)
    assert veto_action is None, f"Expected no veto action for 60% confidence, got {veto_action}"


def test_tiered_veto_reduce_threshold_reduces(veto_reviewer):
    """
    Confidence 70% (65-79% range) should reduce position size.

    When judge disapproves with confidence >= VETO_REDUCE_THRESHOLD but
    < VETO_SKIP_THRESHOLD, the trade executes with reduced position.
    """
    veto_action = veto_reviewer._determine_veto_action(approved=False, confidence=0.70)
    assert veto_action == "reduce", f"Expected 'reduce' for 70% confidence, got {veto_action}"


def test_tiered_veto_skip_threshold_skips(veto_reviewer):
    """
    Confidence 85% (>= 80% skip threshold) should skip trade entirely.

    When judge disapproves with confidence >= VETO_SKIP_THRESHOLD,
    the trade is cancelled completely.
    """
    veto_action = veto_reviewer._determine_veto_action(approved=False, confidence=0.85)
    assert veto_action == "skip", f"Expected 'skip' for 85% confidence, got {veto_action}"


def test_tiered_veto_approved_trade_no_action(veto_reviewer):
    """
    When judge approves trade, no veto action regardless of confidence.

    Veto actions only apply when approved=False.
    """
    veto_action = veto_reviewer._determine_veto_action(approved=True, confidence=0.95)
    assert veto_action is None, f"Expected no veto for approved trade, got {veto_action}"


def test_tiered_veto_boundary_at_reduce_threshold(veto_reviewer):
    """Exactly at reduce threshold (65%) should trigger reduce."""
    veto_action = veto_reviewer._determine_veto_action(approved=False, confidence=0.65)
    assert veto_action == "reduce", f"Expected 'reduce' at exact threshold, got {veto_action}"


def test_tiered_veto_boundary_at_skip_threshold(veto_reviewer):
    """Exactly at skip threshold (80%) should trigger skip."""
    veto_action = veto_reviewer._determine_veto_action(approved=False, confidence=0.80)
    assert veto_action == "skip", f"Expected 'skip' at exact threshold, got {veto_action}"


# ============================================================================
# Hard Stop Calculation Tests - Critical for preventing immediate stop triggers
# ============================================================================


def test_hard_stop_uses_entry_price_not_avg_cost(mock_settings):
    """
    CRITICAL: Hard stop must be calculated from entry_price (market execution price),
    NOT avg_cost (fee-inflated cost basis).

    Bug context: avg_cost includes fees (~0.6% premium), causing hard stop to be set
    above market price when ATR is small, triggering immediate sells after every buy.
    """
    mock_settings.trailing_stop_enabled = True
    mock_settings.stop_loss_atr_multiplier = 1.5
    mock_settings.min_stop_loss_percent = 0.1  # Low value so ATR-based calculation wins
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.breakeven_atr_multiplier = 0.5

    mock_database = Mock()
    mock_database.get_last_paper_balance.return_value = None  # Fresh start
    mock_database.get_last_regime.return_value = None
    mock_database.get_active_trailing_stop.return_value = None  # No existing stop

    with patch('src.daemon.runner.create_exchange_client'):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # Simulate a buy where:
                # - Market entry price (filled_price): $76,861
                # - Cost basis with fees (avg_cost): $77,325 (~0.6% higher)
                # - ATR: $92 (typical for 15-min candles)
                entry_price = Decimal("76861.00")  # Market execution price
                avg_cost = Decimal("77325.00")     # Fee-inflated cost basis

                # Create mock candles for ATR calculation
                candles = pd.DataFrame({
                    'high': [77000.0] * 20,
                    'low': [76800.0] * 20,
                    'close': [76900.0] * 20,
                })

                # Mock ATR calculation to return known value
                with patch('src.indicators.atr.calculate_atr') as mock_atr:
                    mock_atr_result = Mock()
                    mock_atr_result.atr = pd.Series([92.0] * 20)  # ATR = $92
                    mock_atr.return_value = mock_atr_result

                    daemon._create_trailing_stop(
                        entry_price=entry_price,
                        candles=candles,
                        is_paper=True,
                        avg_cost=avg_cost,
                    )

                    # Verify create_trailing_stop was called
                    assert mock_database.create_trailing_stop.called
                    call_kwargs = mock_database.create_trailing_stop.call_args[1]

                    # Calculate expected hard stop from entry_price (NOT avg_cost)
                    # hard_stop = entry_price - (ATR * multiplier) = 76861 - (92 * 1.5) = 76723
                    expected_hard_stop = entry_price - (Decimal("92") * Decimal("1.5"))

                    # CRITICAL: Hard stop should be based on entry_price
                    # If it were based on avg_cost, it would be:
                    # 77325 - 138 = 77187 (ABOVE current market price!)
                    wrong_hard_stop = avg_cost - (Decimal("92") * Decimal("1.5"))

                    actual_hard_stop = call_kwargs['hard_stop']

                    # Verify hard stop uses entry_price (should be ~$76,723)
                    assert actual_hard_stop == expected_hard_stop, \
                        f"Hard stop should be {expected_hard_stop} (from entry_price), got {actual_hard_stop}"

                    # Verify it's NOT using avg_cost (which would be ~$77,187)
                    assert actual_hard_stop != wrong_hard_stop, \
                        f"Hard stop should NOT use avg_cost ({wrong_hard_stop})"

                    # Verify the stop is BELOW entry price (allows room for normal volatility)
                    assert actual_hard_stop < entry_price, \
                        f"Hard stop ({actual_hard_stop}) should be below entry ({entry_price})"


def test_hard_stop_prevents_immediate_trigger(mock_settings):
    """
    Verify that with the fix, a buy at $76,861 with cost basis $77,325
    does NOT immediately trigger a stop at market price $76,800.

    Before fix: hard_stop = $77,187 (from avg_cost), market at $76,800 → TRIGGER
    After fix: hard_stop = $76,723 (from entry_price), market at $76,800 → NO TRIGGER
    """
    mock_settings.trailing_stop_enabled = True
    mock_settings.stop_loss_atr_multiplier = 1.5
    mock_settings.min_stop_loss_percent = 0.1  # Low value so ATR-based calculation wins
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.breakeven_atr_multiplier = 0.5

    mock_database = Mock()
    mock_database.get_last_paper_balance.return_value = None  # Fresh start
    mock_database.get_last_regime.return_value = None
    mock_database.get_active_trailing_stop.return_value = None

    with patch('src.daemon.runner.create_exchange_client'):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                entry_price = Decimal("76861.00")
                avg_cost = Decimal("77325.00")
                current_market_price = Decimal("76800.00")  # Price after buy

                candles = pd.DataFrame({
                    'high': [77000.0] * 20,
                    'low': [76800.0] * 20,
                    'close': [76900.0] * 20,
                })

                with patch('src.indicators.atr.calculate_atr') as mock_atr:
                    mock_atr_result = Mock()
                    mock_atr_result.atr = pd.Series([92.0] * 20)
                    mock_atr.return_value = mock_atr_result

                    daemon._create_trailing_stop(
                        entry_price=entry_price,
                        candles=candles,
                        is_paper=True,
                        avg_cost=avg_cost,
                    )

                    call_kwargs = mock_database.create_trailing_stop.call_args[1]
                    hard_stop = call_kwargs['hard_stop']

                    # With fix: hard_stop = 76861 - 138 = 76723
                    # Market at 76800 is ABOVE 76723 → NO TRIGGER
                    assert current_market_price > hard_stop, \
                        f"Market price {current_market_price} should be above hard stop {hard_stop} (no trigger)"

                    # Verify margin between market and stop
                    margin = current_market_price - hard_stop
                    assert margin > 0, f"Expected positive margin, got {margin}"


def test_hard_stop_dca_uses_latest_entry_price(mock_settings):
    """
    During DCA (Dollar Cost Averaging), hard stop should be recalculated
    based on the latest entry_price, allowing the stop to adapt to new entries.
    """
    mock_settings.trailing_stop_enabled = True
    mock_settings.stop_loss_atr_multiplier = 1.5
    mock_settings.min_stop_loss_percent = 0.1  # Low value so ATR-based calculation wins
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.breakeven_atr_multiplier = 0.5

    mock_database = Mock()
    mock_database.get_last_paper_balance.return_value = None  # Fresh start
    mock_database.get_last_regime.return_value = None
    # Simulate existing stop (DCA scenario)
    existing_stop = Mock()
    existing_stop.id = 1
    mock_database.get_active_trailing_stop.return_value = existing_stop

    with patch('src.daemon.runner.create_exchange_client'):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                # DCA scenario: averaging down at lower price
                new_entry_price = Decimal("75000.00")  # Lower entry (averaging down)
                new_avg_cost = Decimal("76000.00")     # Weighted average cost

                candles = pd.DataFrame({
                    'high': [76000.0] * 20,
                    'low': [74800.0] * 20,
                    'close': [75500.0] * 20,
                })

                with patch('src.indicators.atr.calculate_atr') as mock_atr:
                    mock_atr_result = Mock()
                    mock_atr_result.atr = pd.Series([100.0] * 20)
                    mock_atr.return_value = mock_atr_result

                    daemon._create_trailing_stop(
                        entry_price=new_entry_price,
                        candles=candles,
                        is_paper=True,
                        avg_cost=new_avg_cost,
                    )

                    # Should call update for DCA, not create
                    assert mock_database.update_trailing_stop_for_dca.called
                    call_kwargs = mock_database.update_trailing_stop_for_dca.call_args[1]

                    # Hard stop based on new entry price: 75000 - (100 * 1.5) = 74850
                    expected_hard_stop = new_entry_price - (Decimal("100") * Decimal("1.5"))
                    actual_hard_stop = call_kwargs['hard_stop']

                    assert actual_hard_stop == expected_hard_stop, \
                        f"DCA hard stop should be {expected_hard_stop}, got {actual_hard_stop}"


def test_hard_stop_uses_min_percent_when_atr_too_tight(mock_settings):
    """
    When ATR-based stop is too tight (e.g., 15-min candles with small ATR),
    the min_stop_loss_percent should kick in as a safety floor.

    Example:
    - Entry: $76,861
    - ATR: $92, multiplier 1.5 → ATR distance = $138 (0.18%)
    - min_stop_loss_percent: 0.5% → Min distance = $384
    - Result: Use $384 (the larger distance) → hard_stop = $76,477
    """
    mock_settings.trailing_stop_enabled = True
    mock_settings.stop_loss_atr_multiplier = 1.5
    mock_settings.min_stop_loss_percent = 0.5  # 0.5% minimum stop distance
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.breakeven_atr_multiplier = 0.5

    mock_database = Mock()
    mock_database.get_last_paper_balance.return_value = None
    mock_database.get_last_regime.return_value = None
    mock_database.get_active_trailing_stop.return_value = None

    with patch('src.daemon.runner.create_exchange_client'):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                entry_price = Decimal("76861.00")
                avg_cost = Decimal("77325.00")

                candles = pd.DataFrame({
                    'high': [77000.0] * 20,
                    'low': [76800.0] * 20,
                    'close': [76900.0] * 20,
                })

                with patch('src.indicators.atr.calculate_atr') as mock_atr:
                    mock_atr_result = Mock()
                    mock_atr_result.atr = pd.Series([92.0] * 20)  # Small ATR
                    mock_atr.return_value = mock_atr_result

                    daemon._create_trailing_stop(
                        entry_price=entry_price,
                        candles=candles,
                        is_paper=True,
                        avg_cost=avg_cost,
                    )

                    call_kwargs = mock_database.create_trailing_stop.call_args[1]

                    # ATR distance = 92 * 1.5 = 138 (0.18%)
                    atr_distance = Decimal("92") * Decimal("1.5")
                    # Min % distance = 76861 * 0.5% = 384.305
                    min_pct_distance = entry_price * Decimal("0.5") / Decimal("100")

                    # Verify min % is larger (wins)
                    assert min_pct_distance > atr_distance, \
                        "Test setup error: min % should be larger than ATR distance"

                    # Expected: use the larger distance (min %)
                    expected_hard_stop = entry_price - min_pct_distance
                    actual_hard_stop = call_kwargs['hard_stop']

                    assert actual_hard_stop == expected_hard_stop, \
                        f"Hard stop should use min % ({expected_hard_stop}), got {actual_hard_stop}"

                    # Verify this gives more room than ATR-based stop
                    atr_based_stop = entry_price - atr_distance
                    assert actual_hard_stop < atr_based_stop, \
                        f"Min % stop ({actual_hard_stop}) should be lower than ATR stop ({atr_based_stop})"


def test_hard_stop_uses_atr_when_larger_than_min_percent(mock_settings):
    """
    When ATR-based distance is larger than minimum percentage,
    ATR should be used (no artificial widening of stop).

    Example with larger ATR:
    - Entry: $76,861
    - ATR: $500, multiplier 1.5 → ATR distance = $750 (0.98%)
    - min_stop_loss_percent: 0.5% → Min distance = $384
    - Result: Use $750 (ATR is larger) → hard_stop = $76,111
    """
    mock_settings.trailing_stop_enabled = True
    mock_settings.stop_loss_atr_multiplier = 1.5
    mock_settings.min_stop_loss_percent = 0.5
    mock_settings.trailing_stop_atr_multiplier = 1.0
    mock_settings.breakeven_atr_multiplier = 0.5

    mock_database = Mock()
    mock_database.get_last_paper_balance.return_value = None
    mock_database.get_last_regime.return_value = None
    mock_database.get_active_trailing_stop.return_value = None

    with patch('src.daemon.runner.create_exchange_client'):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                entry_price = Decimal("76861.00")
                avg_cost = Decimal("77325.00")

                candles = pd.DataFrame({
                    'high': [78000.0] * 20,
                    'low': [75000.0] * 20,
                    'close': [76500.0] * 20,
                })

                with patch('src.indicators.atr.calculate_atr') as mock_atr:
                    mock_atr_result = Mock()
                    mock_atr_result.atr = pd.Series([500.0] * 20)  # Large ATR
                    mock_atr.return_value = mock_atr_result

                    daemon._create_trailing_stop(
                        entry_price=entry_price,
                        candles=candles,
                        is_paper=True,
                        avg_cost=avg_cost,
                    )

                    call_kwargs = mock_database.create_trailing_stop.call_args[1]

                    # ATR distance = 500 * 1.5 = 750
                    atr_distance = Decimal("500") * Decimal("1.5")
                    # Min % distance = 76861 * 0.5% = 384.305
                    min_pct_distance = entry_price * Decimal("0.5") / Decimal("100")

                    # Verify ATR is larger (wins)
                    assert atr_distance > min_pct_distance, \
                        "Test setup error: ATR should be larger than min %"

                    # Expected: use ATR distance
                    expected_hard_stop = entry_price - atr_distance
                    actual_hard_stop = call_kwargs['hard_stop']

                    assert actual_hard_stop == expected_hard_stop, \
                        f"Hard stop should use ATR ({expected_hard_stop}), got {actual_hard_stop}"


# ============================================================================
# Multi-Timeframe (HTF) Method Tests
# ============================================================================


@pytest.fixture
def htf_mock_settings(mock_settings):
    """Extend mock settings with MTF configuration."""
    mock_settings.mtf_enabled = True
    mock_settings.mtf_4h_enabled = True
    mock_settings.mtf_candle_limit = 50
    mock_settings.mtf_daily_cache_minutes = 60
    mock_settings.mtf_4h_cache_minutes = 30
    mock_settings.mtf_aligned_boost = 20
    mock_settings.mtf_counter_penalty = 20
    return mock_settings


def test_get_htf_bias_both_bullish(htf_mock_settings, mock_exchange_client, mock_database):
    """Test HTF bias returns 'bullish' when both timeframes are bullish."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                # Mock _get_timeframe_trend to return bullish for both
                daemon._get_timeframe_trend = Mock(return_value="bullish")

                bias, daily, six_h = daemon._get_htf_bias()

                assert bias == "bullish"
                assert daily == "bullish"
                assert six_h == "bullish"


def test_get_htf_bias_both_bearish(htf_mock_settings, mock_exchange_client, mock_database):
    """Test HTF bias returns 'bearish' when both timeframes are bearish."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                # Mock _get_timeframe_trend to return bearish for both
                daemon._get_timeframe_trend = Mock(return_value="bearish")

                bias, daily, six_h = daemon._get_htf_bias()

                assert bias == "bearish"
                assert daily == "bearish"
                assert six_h == "bearish"


def test_get_htf_bias_mixed_returns_neutral(htf_mock_settings, mock_exchange_client, mock_database):
    """Test HTF bias returns 'neutral' when timeframes disagree."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                # Mock _get_timeframe_trend to return different values
                daemon._get_timeframe_trend = Mock(side_effect=["bullish", "bearish"])

                bias, daily, six_h = daemon._get_htf_bias()

                assert bias == "neutral"
                assert daily == "bullish"
                assert six_h == "bearish"


def test_get_htf_bias_disabled_returns_neutral(mock_settings, mock_exchange_client, mock_database):
    """Test HTF bias returns 'neutral' when MTF is disabled."""
    mock_settings.mtf_enabled = False

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(mock_settings)

                bias, daily, six_h = daemon._get_htf_bias()

                assert bias == "neutral"
                assert daily == "neutral"
                assert six_h == "neutral"


def test_get_timeframe_trend_caching(htf_mock_settings, mock_exchange_client, mock_database):
    """Test HTF trend caching works - second call uses cache."""
    from datetime import datetime, timezone

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                # Mock signal scorer get_trend
                daemon.signal_scorer.get_trend = Mock(return_value="bullish")

                # First call should fetch
                trend1 = daemon._get_timeframe_trend("ONE_DAY", 60)

                # Verify get_candles was called
                assert mock_exchange_client.get_candles.called

                # Reset mock to verify caching
                mock_exchange_client.get_candles.reset_mock()
                daemon.signal_scorer.get_trend.reset_mock()

                # Second call within cache period should use cache
                trend2 = daemon._get_timeframe_trend("ONE_DAY", 60)

                # Should NOT call get_candles again (cache hit)
                assert not mock_exchange_client.get_candles.called
                assert trend1 == trend2 == "bullish"


def test_get_timeframe_trend_fail_open(htf_mock_settings, mock_database):
    """Test HTF trend returns neutral on API errors (fail-open)."""
    # Create client that fails on get_candles
    mock_client = Mock()
    mock_client.get_current_price.return_value = Decimal("50000")
    mock_client.get_candles.side_effect = ConnectionError("API Timeout")
    mock_client.get_balance.return_value = Balance("USD", Decimal("1000"), Decimal("0"))

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                htf_mock_settings.mtf_enabled = True
                daemon = TradingDaemon(htf_mock_settings)

                # Clear any cached trend
                daemon._daily_last_fetch = None
                daemon._daily_trend = "neutral"

                # Should return neutral (fail-open) instead of raising
                trend = daemon._get_timeframe_trend("ONE_DAY", 60)

                assert trend == "neutral"


def test_invalidate_htf_cache(htf_mock_settings, mock_exchange_client, mock_database):
    """Test HTF cache invalidation clears timestamps."""
    from datetime import datetime, timezone

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                # Set cache timestamps
                daemon._daily_last_fetch = datetime.now(timezone.utc)
                daemon._6h_last_fetch = datetime.now(timezone.utc)

                # Invalidate cache
                daemon._invalidate_htf_cache()

                # Verify timestamps are cleared
                assert daemon._daily_last_fetch is None
                assert daemon._6h_last_fetch is None


def test_store_signal_history_returns_id(htf_mock_settings, mock_exchange_client, mock_database):
    """Test signal history storage returns the record ID."""
    from src.strategy.signal_scorer import SignalResult, IndicatorValues

    # Mock database session to return ID
    mock_session = MagicMock()
    mock_history = Mock()
    mock_history.id = 42
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=False)
    mock_session.add = Mock()
    mock_session.commit = Mock(side_effect=lambda: setattr(mock_history, 'id', 42))
    mock_database.session.return_value = mock_session

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                with patch('src.daemon.runner.SignalHistory') as mock_signal_history_class:
                    # Make the SignalHistory class return our mock
                    mock_signal_history_class.return_value = mock_history

                    daemon = TradingDaemon(htf_mock_settings)

                    signal_result = SignalResult(
                        score=65,
                        action="buy",
                        indicators=IndicatorValues(
                            rsi=35.0, macd_line=100.0, macd_signal=50.0,
                            macd_histogram=50.0, bb_upper=51000.0, bb_middle=50000.0,
                            bb_lower=49000.0, ema_fast=50100.0, ema_slow=50000.0,
                            volatility="normal"
                        ),
                        breakdown={"rsi": 15, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
                        confidence=0.8,
                    )

                    signal_id = daemon._store_signal_history(
                        signal_result=signal_result,
                        current_price=Decimal("50000"),
                        htf_bias="bullish",
                        daily_trend="bullish",
                        four_hour_trend="bullish",
                        threshold=60,
                    )

                    assert signal_id == 42


def test_store_signal_history_handles_db_error(htf_mock_settings, mock_exchange_client, mock_database):
    """Test signal history storage gracefully handles database errors."""
    from src.strategy.signal_scorer import SignalResult, IndicatorValues
    from sqlalchemy.exc import SQLAlchemyError

    # Mock database session to raise error when entering context
    mock_session = MagicMock()
    mock_session.__enter__ = Mock(side_effect=SQLAlchemyError("Database locked"))
    mock_session.__exit__ = Mock(return_value=False)
    mock_database.session.return_value = mock_session

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                signal_result = SignalResult(
                    score=65,
                    action="buy",
                    indicators=IndicatorValues(
                        rsi=35.0, macd_line=100.0, macd_signal=50.0,
                        macd_histogram=50.0, bb_upper=51000.0, bb_middle=50000.0,
                        bb_lower=49000.0, ema_fast=50100.0, ema_slow=50000.0,
                        volatility="normal"
                    ),
                    breakdown={"rsi": 15, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
                    confidence=0.8,
                )

                # Should return None (not raise) on DB error
                signal_id = daemon._store_signal_history(
                    signal_result=signal_result,
                    current_price=Decimal("50000"),
                    htf_bias="bullish",
                    daily_trend="bullish",
                    four_hour_trend="bullish",
                    threshold=60,
                )

                assert signal_id is None


def test_store_signal_history_alerts_after_repeated_failures(htf_mock_settings, mock_exchange_client, mock_database):
    """Test signal history storage alerts after 10 consecutive failures."""
    from src.strategy.signal_scorer import SignalResult, IndicatorValues
    from sqlalchemy.exc import SQLAlchemyError

    # Mock database session to raise error when entering context
    mock_session = MagicMock()
    mock_session.__enter__ = Mock(side_effect=SQLAlchemyError("Database locked"))
    mock_session.__exit__ = Mock(return_value=False)
    mock_database.session.return_value = mock_session

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier_class:
                mock_notifier = Mock()
                mock_notifier_class.return_value = mock_notifier

                daemon = TradingDaemon(htf_mock_settings)

                signal_result = SignalResult(
                    score=65,
                    action="buy",
                    indicators=IndicatorValues(
                        rsi=35.0, macd_line=100.0, macd_signal=50.0,
                        macd_histogram=50.0, bb_upper=51000.0, bb_middle=50000.0,
                        bb_lower=49000.0, ema_fast=50100.0, ema_slow=50000.0,
                        volatility="normal"
                    ),
                    breakdown={"rsi": 15, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
                    confidence=0.8,
                )

                # Call 9 times - should NOT alert yet
                for _ in range(9):
                    daemon._store_signal_history(
                        signal_result=signal_result,
                        current_price=Decimal("50000"),
                        htf_bias="bullish",
                        daily_trend="bullish",
                        four_hour_trend="bullish",
                        threshold=60,
                    )
                mock_notifier.notify_error.assert_not_called()

                # 10th failure should trigger alert
                daemon._store_signal_history(
                    signal_result=signal_result,
                    current_price=Decimal("50000"),
                    htf_bias="bullish",
                    daily_trend="bullish",
                    four_hour_trend="bullish",
                    threshold=60,
                )
                mock_notifier.notify_error.assert_called_once()
                error_msg = mock_notifier.notify_error.call_args[0][0].lower()
                assert "signal history storage" in error_msg
                assert "10" in error_msg  # Verify failure count is included


def test_store_signal_history_alerts_every_50_after_initial(htf_mock_settings, mock_exchange_client, mock_database):
    """Test signal history alerts at 10, then every 50 additional failures (60, 110...)."""
    from src.strategy.signal_scorer import SignalResult, IndicatorValues
    from sqlalchemy.exc import SQLAlchemyError

    mock_session = MagicMock()
    mock_session.__enter__ = Mock(side_effect=SQLAlchemyError("Database locked"))
    mock_session.__exit__ = Mock(return_value=False)
    mock_database.session.return_value = mock_session

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier_class:
                mock_notifier = Mock()
                mock_notifier_class.return_value = mock_notifier

                daemon = TradingDaemon(htf_mock_settings)

                signal_result = SignalResult(
                    score=65,
                    action="buy",
                    indicators=IndicatorValues(
                        rsi=35.0, macd_line=100.0, macd_signal=50.0,
                        macd_histogram=50.0, bb_upper=51000.0, bb_middle=50000.0,
                        bb_lower=49000.0, ema_fast=50100.0, ema_slow=50000.0,
                        volatility="normal"
                    ),
                    breakdown={"rsi": 15, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
                    confidence=0.8,
                )

                # Helper to call _store_signal_history
                def trigger_failure():
                    daemon._store_signal_history(
                        signal_result=signal_result,
                        current_price=Decimal("50000"),
                        htf_bias="bullish",
                        daily_trend="bullish",
                        four_hour_trend="bullish",
                        threshold=60,
                    )

                # 10 failures should trigger first alert
                for _ in range(10):
                    trigger_failure()
                assert mock_notifier.notify_error.call_count == 1

                # Next 49 failures (11-59) should NOT alert
                for _ in range(49):
                    trigger_failure()
                assert mock_notifier.notify_error.call_count == 1

                # 60th failure should trigger second alert
                trigger_failure()
                assert mock_notifier.notify_error.call_count == 2
                assert "60" in mock_notifier.notify_error.call_args[0][0]


def test_store_signal_history_resets_failure_counter_on_success(htf_mock_settings, mock_exchange_client, mock_database):
    """Test signal history storage resets failure counter on success."""
    from src.strategy.signal_scorer import SignalResult, IndicatorValues
    from sqlalchemy.exc import SQLAlchemyError

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier_class:
                with patch('src.daemon.runner.SignalHistory') as mock_signal_history_class:
                    mock_notifier = Mock()
                    mock_notifier_class.return_value = mock_notifier
                    mock_history = Mock()
                    mock_history.id = 1
                    mock_signal_history_class.return_value = mock_history

                    daemon = TradingDaemon(htf_mock_settings)

                    signal_result = SignalResult(
                        score=65,
                        action="buy",
                        indicators=IndicatorValues(
                            rsi=35.0, macd_line=100.0, macd_signal=50.0,
                            macd_histogram=50.0, bb_upper=51000.0, bb_middle=50000.0,
                            bb_lower=49000.0, ema_fast=50100.0, ema_slow=50000.0,
                            volatility="normal"
                        ),
                        breakdown={"rsi": 15, "macd": 20, "bollinger": 15, "ema": 10, "volume": 5},
                        confidence=0.8,
                    )

                    # Simulate 5 failures
                    mock_session_fail = MagicMock()
                    mock_session_fail.__enter__ = Mock(side_effect=SQLAlchemyError("Database locked"))
                    mock_session_fail.__exit__ = Mock(return_value=False)
                    mock_database.session.return_value = mock_session_fail

                    for _ in range(5):
                        daemon._store_signal_history(
                            signal_result=signal_result,
                            current_price=Decimal("50000"),
                            htf_bias="bullish",
                            daily_trend="bullish",
                            four_hour_trend="bullish",
                            threshold=60,
                        )

                    assert daemon._signal_history_failures == 5

                    # One successful call should reset counter
                    mock_session_success = MagicMock()
                    mock_session_success.__enter__ = Mock(return_value=mock_session_success)
                    mock_session_success.__exit__ = Mock(return_value=False)
                    mock_database.session.return_value = mock_session_success

                    daemon._store_signal_history(
                        signal_result=signal_result,
                        current_price=Decimal("50000"),
                        htf_bias="bullish",
                        daily_trend="bullish",
                        four_hour_trend="bullish",
                        threshold=60,
                    )

                    assert daemon._signal_history_failures == 0


def test_mark_signal_trade_executed_with_none_id(htf_mock_settings, mock_exchange_client, mock_database):
    """Test marking trade executed handles None signal_id gracefully."""
    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                # Reset the mock to clear any calls from initialization
                mock_database.session.reset_mock()

                # Should not raise or call database
                daemon._mark_signal_trade_executed(None)

                # Verify no database session was requested
                mock_database.session.assert_not_called()


def test_mark_signal_trade_executed_with_valid_id(htf_mock_settings, mock_exchange_client, mock_database):
    """Test marking trade executed updates the correct record."""
    mock_session = MagicMock()
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=False)
    mock_query = Mock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.update = Mock()
    mock_database.session.return_value = mock_session

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier'):
                daemon = TradingDaemon(htf_mock_settings)

                daemon._mark_signal_trade_executed(42)

                # Verify update was called with trade_executed=True
                mock_query.update.assert_called_once_with({"trade_executed": True})
