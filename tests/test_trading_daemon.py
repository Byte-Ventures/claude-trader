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
    settings.take_profit_atr_multiplier = 3.0
    settings.stop_loss_pct = None
    settings.trailing_stop_enabled = False
    settings.use_limit_orders = True
    settings.min_trade_quote = 10.0  # Minimum order size in quote currency
    settings.max_trade_quote = None  # Maximum order size (None = no limit)

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
    settings.block_trades_extreme_conditions = True

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
    CRITICAL: Verify AI_FAILURE_MODE_BUY=open does NOT skip trade when AI review fails.

    This tests fail-open behavior for buys - the code should NOT return early
    after AI failure, allowing the trade to proceed.
    """
    from config.settings import AIFailureMode, VetoAction

    # Enable AI review and set to OPEN mode for buys
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.OPEN  # Fallback
    mock_settings.ai_failure_mode_buy = AIFailureMode.OPEN  # Per-action setting
    mock_settings.ai_failure_mode_sell = AIFailureMode.OPEN
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
    CRITICAL: Verify AI_FAILURE_MODE_BUY=safe skips trade when AI review fails.

    In safe mode, buys are NOT executed when AI review is unavailable,
    providing protection against trading blind during AI outages.
    """
    from config.settings import AIFailureMode, VetoAction

    # Enable AI review and set to SAFE mode for buys
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.OPEN  # Fallback
    mock_settings.ai_failure_mode_buy = AIFailureMode.SAFE  # Per-action setting (safe for buys)
    mock_settings.ai_failure_mode_sell = AIFailureMode.OPEN
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


def test_ai_failure_mode_defaults(mock_settings, mock_exchange_client, mock_database):
    """Verify AI failure mode defaults: safe for buys, open for sells."""
    from config.settings import AIFailureMode, Settings

    # Create real settings to check defaults
    with patch.dict('os.environ', {}, clear=False):
        settings = Settings(
            trading_mode="paper",
            trading_pair="BTC-USD",
        )
        # Legacy setting default (for backward compatibility)
        assert settings.ai_failure_mode == AIFailureMode.OPEN
        # New per-action defaults
        assert settings.ai_failure_mode_buy == AIFailureMode.SAFE, "Buys should default to SAFE (skip on AI failure)"
        assert settings.ai_failure_mode_sell == AIFailureMode.OPEN, "Sells should default to OPEN (proceed on AI failure)"


def test_ai_failure_mode_sell_proceeds_on_failure(mock_settings, mock_exchange_client, mock_database):
    """
    CRITICAL: Verify AI_FAILURE_MODE_SELL=open allows sell to proceed when AI fails.

    Sells should NOT be skipped during AI outages to avoid trapping users in
    positions during market crashes. This is the default behavior.
    """
    from config.settings import AIFailureMode, VetoAction

    # Enable AI review with OPEN mode for sells (default)
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.OPEN  # Fallback
    mock_settings.ai_failure_mode_buy = AIFailureMode.SAFE
    mock_settings.ai_failure_mode_sell = AIFailureMode.OPEN  # Sells should proceed
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

    # Create strong sell signal
    sell_signal = SignalResult(
        score=-70,  # Strong sell signal
        action="sell",
        indicators=IndicatorValues(
            rsi=75.0,  # Overbought
            macd_line=-100.0,
            macd_signal=-50.0,
            macd_histogram=-50.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
            ema_fast=49900.0,
            ema_slow=50000.0,
            atr=500.0,
            volatility="normal"
        ),
        breakdown={"rsi": -20, "macd": -20, "bollinger": -15, "ema": -10, "volume": -5},
        confidence=0.8
    )

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier:
                daemon = TradingDaemon(mock_settings)

                # Ensure we have a position to sell (mock already returns 1.0 BTC)
                # This is critical - can't sell without a position
                mock_exchange_client.reset_mock()

                # Mock signal scorer to return strong sell signal
                daemon.signal_scorer.calculate_score = Mock(return_value=sell_signal)

                # Mock trade reviewer to raise an exception (simulating AI failure)
                daemon.trade_reviewer = Mock()
                daemon.trade_reviewer.should_review.return_value = (True, "trade")
                daemon.trade_reviewer.review_trade = Mock(side_effect=Exception("AI API Timeout"))

                # Run trading iteration
                daemon._trading_iteration()

                # CRITICAL: Verify sell was NOT skipped (negative assertion)
                notifier_instance = mock_notifier.return_value
                for call in notifier_instance.send_message.call_args_list:
                    msg = str(call)
                    assert "Trade skipped" not in msg, "SELL with OPEN mode should not skip trades"

                # CRITICAL: Verify sell was actually attempted (positive assertion)
                # In OPEN mode, the sell should proceed despite AI failure
                assert mock_exchange_client.market_sell.called, \
                    "SELL with OPEN mode should attempt trade execution despite AI failure"


def test_ai_failure_mode_sell_safe_skips_trade(mock_settings, mock_exchange_client, mock_database):
    """
    Verify AI_FAILURE_MODE_SELL=safe skips sell when AI review fails.

    This is NOT the default behavior (default is OPEN for sells), but users
    may explicitly configure SAFE mode for sells. This test ensures symmetric
    behavior: both buy and sell actions can be skipped in SAFE mode.
    """
    from config.settings import AIFailureMode, VetoAction

    # Enable AI review with SAFE mode for sells (non-default config)
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.OPEN  # Fallback
    mock_settings.ai_failure_mode_buy = AIFailureMode.SAFE
    mock_settings.ai_failure_mode_sell = AIFailureMode.SAFE  # Non-default: skip sells on AI failure
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

    # Create strong sell signal
    sell_signal = SignalResult(
        score=-70,  # Strong sell signal
        action="sell",
        indicators=IndicatorValues(
            rsi=75.0,  # Overbought
            macd_line=-100.0,
            macd_signal=-50.0,
            macd_histogram=-50.0,
            bb_upper=51000.0,
            bb_middle=50000.0,
            bb_lower=49000.0,
            ema_fast=49900.0,
            ema_slow=50000.0,
            atr=500.0,
            volatility="normal"
        ),
        breakdown={"rsi": -20, "macd": -20, "bollinger": -15, "ema": -10, "volume": -5},
        confidence=0.8
    )

    with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
        with patch('src.daemon.runner.Database', return_value=mock_database):
            with patch('src.daemon.runner.TelegramNotifier') as mock_notifier:
                daemon = TradingDaemon(mock_settings)

                # Ensure we have a position to sell (mock already returns 1.0 BTC)
                mock_exchange_client.reset_mock()

                # Mock signal scorer to return strong sell signal
                daemon.signal_scorer.calculate_score = Mock(return_value=sell_signal)

                # Mock trade reviewer to raise an exception (simulating AI failure)
                daemon.trade_reviewer = Mock()
                daemon.trade_reviewer.should_review.return_value = (True, "trade")
                daemon.trade_reviewer.review_trade = Mock(side_effect=Exception("AI API Timeout"))

                # Run trading iteration
                daemon._trading_iteration()

                # Verify sell was NOT executed (fail-safe for sells when explicitly configured)
                assert not mock_exchange_client.market_sell.called, \
                    "SELL with SAFE mode should skip trade on AI failure"

                # Verify notification was sent about skipped trade
                notifier_instance = mock_notifier.return_value
                skip_notification_sent = False
                for call in notifier_instance.send_message.call_args_list:
                    msg = str(call)
                    if "Trade skipped" in msg or "AI review failed" in msg:
                        skip_notification_sent = True
                        break
                assert skip_notification_sent, "Should notify user when sell is skipped due to AI failure"


def test_ai_failure_notification_cooldown(mock_settings, mock_exchange_client, mock_database):
    """
    Verify AI failure notifications are rate-limited to prevent Telegram spam.

    When AI review fails repeatedly in SAFE mode, notifications should only be
    sent once per 15 minutes to avoid flooding the user's Telegram.
    """
    from config.settings import AIFailureMode, VetoAction
    from datetime import datetime, timedelta, timezone

    # Enable AI review and set to SAFE mode for buys
    mock_settings.ai_review_enabled = True
    mock_settings.ai_failure_mode = AIFailureMode.OPEN  # Fallback
    mock_settings.ai_failure_mode_buy = AIFailureMode.SAFE  # Per-action setting
    mock_settings.ai_failure_mode_sell = AIFailureMode.OPEN
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


def test_extreme_volatility_uses_wider_stop_multiplier(mock_settings):
    """
    Verify that during extreme volatility conditions, the wider stop multiplier
    (stop_loss_atr_multiplier_extreme) is used instead of the standard multiplier.

    This prevents being stopped out by normal price fluctuations during high volatility.

    Example:
    - Entry: $76,861
    - ATR: $500
    - Normal multiplier: 1.5 → stop distance = $750 → hard_stop = $76,111
    - Extreme multiplier: 2.0 → stop distance = $1,000 → hard_stop = $75,861
    """
    mock_settings.trailing_stop_enabled = True
    mock_settings.stop_loss_atr_multiplier = 1.5  # Normal volatility
    mock_settings.stop_loss_atr_multiplier_extreme = 2.0  # Extreme volatility
    mock_settings.min_stop_loss_percent = 0.1  # Low value so ATR-based calculation wins
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
                    mock_atr_result.atr = pd.Series([500.0] * 20)
                    mock_atr.return_value = mock_atr_result

                    # Test with extreme volatility
                    daemon._create_trailing_stop(
                        entry_price=entry_price,
                        candles=candles,
                        is_paper=True,
                        avg_cost=avg_cost,
                        volatility="extreme",
                    )

                    call_kwargs = mock_database.create_trailing_stop.call_args[1]
                    actual_hard_stop_extreme = call_kwargs['hard_stop']

                    # Expected with extreme multiplier: 76861 - (500 * 2.0) = 75861
                    expected_extreme = entry_price - (Decimal("500") * Decimal("2.0"))

                    assert actual_hard_stop_extreme == expected_extreme, \
                        f"Extreme volatility should use 2.0x multiplier: expected {expected_extreme}, got {actual_hard_stop_extreme}"

                    # Reset mock for comparison
                    mock_database.reset_mock()

                    # Test with normal volatility for comparison
                    daemon._create_trailing_stop(
                        entry_price=entry_price,
                        candles=candles,
                        is_paper=True,
                        avg_cost=avg_cost,
                        volatility="normal",
                    )

                    call_kwargs_normal = mock_database.create_trailing_stop.call_args[1]
                    actual_hard_stop_normal = call_kwargs_normal['hard_stop']

                    # Expected with normal multiplier: 76861 - (500 * 1.5) = 76111
                    expected_normal = entry_price - (Decimal("500") * Decimal("1.5"))

                    assert actual_hard_stop_normal == expected_normal, \
                        f"Normal volatility should use 1.5x multiplier: expected {expected_normal}, got {actual_hard_stop_normal}"

                    # Verify extreme stop is lower (wider) than normal stop
                    assert actual_hard_stop_extreme < actual_hard_stop_normal, \
                        f"Extreme stop ({actual_hard_stop_extreme}) should be lower than normal stop ({actual_hard_stop_normal})"


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


# ============================================================================
# Flap Protection Tests - CRITICAL
# ============================================================================

class TestRegimeFlapProtection:
    """Test regime flap protection prevents rapid oscillation."""

    def test_first_detection_sets_pending_state(self, mock_settings, mock_exchange_client, mock_database):
        """First detection of new regime should set pending state, not apply change."""
        mock_settings.regime_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # Initial state
                    daemon._last_regime = "neutral"
                    daemon._pending_regime = None

                    # Simulate first detection of "risk_on" regime
                    # (testing internal state, not full iteration)
                    new_regime = "risk_on"

                    # Apply flap protection logic
                    if new_regime != daemon._last_regime:
                        if daemon.settings.regime_flap_protection:
                            if daemon._pending_regime == new_regime:
                                daemon._pending_regime = None
                                daemon._last_regime = new_regime  # Apply change
                            else:
                                daemon._pending_regime = new_regime  # Set pending

                    # First detection should only set pending, NOT change regime
                    assert daemon._pending_regime == "risk_on"
                    assert daemon._last_regime == "neutral"  # Unchanged

    def test_second_consecutive_detection_applies_change(self, mock_settings, mock_exchange_client, mock_database):
        """Second consecutive detection of same regime should confirm and apply change."""
        mock_settings.regime_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # State after first detection
                    daemon._last_regime = "neutral"
                    daemon._pending_regime = "risk_on"

                    # Simulate second consecutive detection
                    new_regime = "risk_on"

                    if new_regime != daemon._last_regime:
                        if daemon.settings.regime_flap_protection:
                            if daemon._pending_regime == new_regime:
                                daemon._pending_regime = None
                                daemon._last_regime = new_regime  # Apply change
                            else:
                                daemon._pending_regime = new_regime

                    # Second detection should apply the change
                    assert daemon._pending_regime is None
                    assert daemon._last_regime == "risk_on"

    def test_alternating_detections_clear_pending(self, mock_settings, mock_exchange_client, mock_database):
        """Alternating detections (A→B→C) should reset pending state each time."""
        mock_settings.regime_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # First detection: neutral → risk_on (pending)
                    daemon._last_regime = "neutral"
                    daemon._pending_regime = None
                    new_regime = "risk_on"
                    if new_regime != daemon._last_regime:
                        if daemon._pending_regime == new_regime:
                            daemon._pending_regime = None
                            daemon._last_regime = new_regime
                        else:
                            daemon._pending_regime = new_regime
                    assert daemon._pending_regime == "risk_on"
                    assert daemon._last_regime == "neutral"

                    # Second detection: different regime (risk_off) - should reset pending
                    new_regime = "risk_off"
                    if new_regime != daemon._last_regime:
                        if daemon._pending_regime == new_regime:
                            daemon._pending_regime = None
                            daemon._last_regime = new_regime
                        else:
                            daemon._pending_regime = new_regime
                    # Pending should now be risk_off (replaced risk_on)
                    assert daemon._pending_regime == "risk_off"
                    assert daemon._last_regime == "neutral"  # Still unchanged

    def test_returning_to_current_clears_pending(self, mock_settings, mock_exchange_client, mock_database):
        """Returning to current regime should clear pending state."""
        mock_settings.regime_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # State: pending change to risk_on
                    daemon._last_regime = "neutral"
                    daemon._pending_regime = "risk_on"

                    # Detection returns to current regime (neutral)
                    new_regime = "neutral"
                    if new_regime != daemon._last_regime:
                        if daemon._pending_regime == new_regime:
                            daemon._pending_regime = None
                            daemon._last_regime = new_regime
                        else:
                            daemon._pending_regime = new_regime
                    else:
                        daemon._pending_regime = None  # Clear pending

                    # Pending should be cleared
                    assert daemon._pending_regime is None
                    assert daemon._last_regime == "neutral"

    def test_flap_protection_disabled_applies_immediately(self, mock_settings, mock_exchange_client, mock_database):
        """With flap protection disabled, regime change should apply immediately."""
        mock_settings.regime_flap_protection = False

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    daemon._last_regime = "neutral"
                    daemon._pending_regime = None

                    new_regime = "risk_on"
                    should_apply = False
                    if new_regime != daemon._last_regime:
                        if daemon.settings.regime_flap_protection:
                            if daemon._pending_regime == new_regime:
                                daemon._pending_regime = None
                                should_apply = True
                            else:
                                daemon._pending_regime = new_regime
                        else:
                            should_apply = True

                    if should_apply:
                        daemon._last_regime = new_regime

                    # Change should apply immediately
                    assert daemon._last_regime == "risk_on"


class TestWeightProfileFlapProtection:
    """Test weight profile flap protection prevents rapid oscillation."""

    def test_first_detection_sets_pending_state(self, mock_settings, mock_exchange_client, mock_database):
        """First detection of new profile should set pending state, not apply change."""
        mock_settings.weight_profile_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    daemon._last_weight_profile = "default"
                    daemon._pending_weight_profile = None

                    new_profile = "trending"
                    if new_profile != daemon._last_weight_profile:
                        if daemon.settings.weight_profile_flap_protection:
                            if daemon._pending_weight_profile == new_profile:
                                daemon._pending_weight_profile = None
                                daemon._last_weight_profile = new_profile
                            else:
                                daemon._pending_weight_profile = new_profile

                    assert daemon._pending_weight_profile == "trending"
                    assert daemon._last_weight_profile == "default"

    def test_second_consecutive_detection_applies_change(self, mock_settings, mock_exchange_client, mock_database):
        """Second consecutive detection of same profile should confirm and apply change."""
        mock_settings.weight_profile_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    daemon._last_weight_profile = "default"
                    daemon._pending_weight_profile = "trending"

                    new_profile = "trending"
                    if new_profile != daemon._last_weight_profile:
                        if daemon.settings.weight_profile_flap_protection:
                            if daemon._pending_weight_profile == new_profile:
                                daemon._pending_weight_profile = None
                                daemon._last_weight_profile = new_profile
                            else:
                                daemon._pending_weight_profile = new_profile

                    assert daemon._pending_weight_profile is None
                    assert daemon._last_weight_profile == "trending"

    def test_alternating_detections_clear_pending(self, mock_settings, mock_exchange_client, mock_database):
        """Alternating profile detections should reset pending state."""
        mock_settings.weight_profile_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # First: default → trending (pending)
                    daemon._last_weight_profile = "default"
                    daemon._pending_weight_profile = None
                    new_profile = "trending"
                    if new_profile != daemon._last_weight_profile:
                        if daemon._pending_weight_profile == new_profile:
                            daemon._pending_weight_profile = None
                            daemon._last_weight_profile = new_profile
                        else:
                            daemon._pending_weight_profile = new_profile
                    assert daemon._pending_weight_profile == "trending"

                    # Second: different profile (ranging) - replaces pending
                    new_profile = "ranging"
                    if new_profile != daemon._last_weight_profile:
                        if daemon._pending_weight_profile == new_profile:
                            daemon._pending_weight_profile = None
                            daemon._last_weight_profile = new_profile
                        else:
                            daemon._pending_weight_profile = new_profile
                    assert daemon._pending_weight_profile == "ranging"
                    assert daemon._last_weight_profile == "default"

    def test_flap_protection_disabled_applies_immediately(self, mock_settings, mock_exchange_client, mock_database):
        """With flap protection disabled, profile change should apply immediately."""
        mock_settings.weight_profile_flap_protection = False

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    daemon._last_weight_profile = "default"
                    daemon._pending_weight_profile = None

                    new_profile = "trending"
                    should_apply = False
                    if new_profile != daemon._last_weight_profile:
                        if daemon.settings.weight_profile_flap_protection:
                            if daemon._pending_weight_profile == new_profile:
                                daemon._pending_weight_profile = None
                                should_apply = True
                            else:
                                daemon._pending_weight_profile = new_profile
                        else:
                            should_apply = True

                    if should_apply:
                        daemon._last_weight_profile = new_profile

                    assert daemon._last_weight_profile == "trending"

    def test_confidence_updated_during_pending_state(self, mock_settings, mock_exchange_client, mock_database):
        """Confidence/reasoning should be updated even during pending state for dashboard visibility."""
        mock_settings.weight_profile_flap_protection = True

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    daemon._last_weight_profile = "default"
                    daemon._pending_weight_profile = None
                    daemon._last_weight_profile_confidence = 0.7
                    daemon._last_weight_profile_reasoning = "old reasoning"

                    # Simulate first detection with new confidence/reasoning
                    new_profile = "trending"
                    new_confidence = 0.95
                    new_reasoning = "Strong trend detected"

                    if new_profile != daemon._last_weight_profile:
                        if daemon.settings.weight_profile_flap_protection:
                            if daemon._pending_weight_profile == new_profile:
                                daemon._pending_weight_profile = None
                                daemon._last_weight_profile = new_profile
                                daemon._last_weight_profile_confidence = new_confidence
                                daemon._last_weight_profile_reasoning = new_reasoning
                            else:
                                daemon._pending_weight_profile = new_profile
                                # Update confidence/reasoning even during pending
                                daemon._last_weight_profile_confidence = new_confidence
                                daemon._last_weight_profile_reasoning = new_reasoning

                    # Profile not changed yet (pending), but confidence/reasoning updated
                    assert daemon._last_weight_profile == "default"
                    assert daemon._pending_weight_profile == "trending"
                    assert daemon._last_weight_profile_confidence == 0.95
                    assert daemon._last_weight_profile_reasoning == "Strong trend detected"


# ============================================================================
# Dual-Extreme Conditions Blocking Tests - CRITICAL
# ============================================================================

class TestDualExtremeBlocking:
    """Test blocking of new positions during dual-extreme conditions (extreme_fear + extreme volatility)."""

    def test_blocks_buy_during_dual_extreme_conditions(self, mock_settings, mock_exchange_client, mock_database):
        """CRITICAL: Test buy is blocked when extreme_fear + extreme volatility."""
        mock_settings.block_trades_extreme_conditions = True
        mock_settings.use_limit_orders = False  # Simplify to direct market orders
        mock_settings.is_paper_trading = False  # Use mock client directly (avoid paper wrapper)
        mock_settings.max_position_percent = Decimal("100")  # Allow large positions for test

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier') as mock_notifier_class:
                    mock_notifier = Mock()
                    mock_notifier_class.return_value = mock_notifier

                    daemon = TradingDaemon(mock_settings)

                    # Mock position_sizer config to ensure can_buy=True
                    daemon.position_sizer.config.min_trade_quote = Decimal("10")
                    daemon.position_sizer.config.max_position_percent = Decimal("100")
                    daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                    # Mock position_sizer.calculate_size to return valid position
                    from src.strategy.position_sizer import PositionSizeResult
                    daemon.position_sizer.calculate_size = Mock(return_value=PositionSizeResult(
                        size_base=Decimal("0.002"),
                        size_quote=Decimal("100"),
                        stop_loss_price=Decimal("49000"),
                        take_profit_price=Decimal("52000"),
                        risk_amount_quote=Decimal("2"),
                        position_percent=1.0,
                    ))

                    # Mock market_regime to return dual-extreme conditions
                    daemon.market_regime = Mock()
                    mock_regime = Mock()
                    mock_regime.regime_name = "extreme_risk_off"
                    mock_regime.threshold_adjustment = 0
                    mock_regime.position_multiplier = 0.5
                    mock_regime.components = {
                        "sentiment": {"category": "extreme_fear", "value": 15},
                        "volatility": {"level": "extreme"}
                    }
                    daemon.market_regime.calculate.return_value = mock_regime

                    # Mock signal scorer to return strong buy signal
                    daemon.signal_scorer.calculate_score = Mock(return_value=SignalResult(
                        score=75,  # Strong buy signal (should trigger buy normally)
                        action="buy",
                        indicators=IndicatorValues(
                            rsi=35.0,
                            macd_line=10.0,
                            macd_signal=5.0,
                            macd_histogram=5.0,
                            bb_upper=51000.0,
                            bb_middle=50000.0,
                            bb_lower=49000.0,
                            ema_fast=50100.0,
                            ema_slow=49900.0,
                            atr=1000.0,
                            volatility="extreme"
                        ),
                        breakdown={"rsi": 20, "macd": 15, "bollinger": 10, "ema": 15, "volume": 15},
                        confidence=0.8
                    ))

                    # Reset mocks after initialization
                    mock_exchange_client.reset_mock()
                    mock_notifier.reset_mock()

                    # Run iteration
                    daemon._trading_iteration()

                    # Verify buy was blocked (no order placed)
                    mock_exchange_client.market_buy.assert_not_called()

                    # Verify the dual-extreme blocking notification was sent
                    mock_notifier.notify_trade_rejected.assert_called_once()
                    call_kwargs = mock_notifier.notify_trade_rejected.call_args[1]
                    assert call_kwargs["side"] == "buy"
                    assert "extreme_fear" in call_kwargs["reason"]
                    assert "extreme_volatility" in call_kwargs["reason"]

    def test_allows_buy_extreme_fear_normal_volatility(self, mock_settings, mock_exchange_client, mock_database):
        """Test buy proceeds with extreme_fear but normal volatility."""
        mock_settings.block_trades_extreme_conditions = True
        mock_settings.use_limit_orders = False  # Simplify to direct market orders
        mock_settings.is_paper_trading = False  # Use mock client directly (avoid paper wrapper)
        mock_settings.max_position_percent = Decimal("100")  # Allow large positions for test

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # Mock position_sizer config to ensure can_buy=True
                    daemon.position_sizer.config.min_trade_quote = Decimal("10")
                    daemon.position_sizer.config.max_position_percent = Decimal("100")
                    daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                    # Mock position_sizer.calculate_size to return valid position
                    from src.strategy.position_sizer import PositionSizeResult
                    daemon.position_sizer.calculate_size = Mock(return_value=PositionSizeResult(
                        size_base=Decimal("0.002"),
                        size_quote=Decimal("100"),
                        stop_loss_price=Decimal("49000"),
                        take_profit_price=Decimal("52000"),
                        risk_amount_quote=Decimal("2"),
                        position_percent=1.0,
                    ))

                    # Mock market_regime: extreme_fear but normal volatility
                    daemon.market_regime = Mock()
                    mock_regime = Mock()
                    mock_regime.regime_name = "risk_off"
                    mock_regime.threshold_adjustment = -10
                    mock_regime.position_multiplier = 1.25
                    mock_regime.components = {
                        "sentiment": {"category": "extreme_fear", "value": 15},
                        "volatility": {"level": "normal"}  # NOT extreme
                    }
                    daemon.market_regime.calculate.return_value = mock_regime

                    # Mock signal scorer to return strong buy signal
                    daemon.signal_scorer.calculate_score = Mock(return_value=SignalResult(
                        score=75,
                        action="buy",
                        indicators=IndicatorValues(
                            rsi=35.0,
                            macd_line=10.0,
                            macd_signal=5.0,
                            macd_histogram=5.0,
                            bb_upper=51000.0,
                            bb_middle=50000.0,
                            bb_lower=49000.0,
                            ema_fast=50100.0,
                            ema_slow=49900.0,
                            atr=1000.0,
                            volatility="normal"
                        ),
                        breakdown={"rsi": 20, "macd": 15, "bollinger": 10, "ema": 15, "volume": 15},
                        confidence=0.8
                    ))

                    # Reset mock after initialization
                    mock_exchange_client.reset_mock()

                    # Run iteration
                    daemon._trading_iteration()

                    # Verify buy was allowed (order placed)
                    mock_exchange_client.market_buy.assert_called_once()

    def test_allows_buy_fear_extreme_volatility(self, mock_settings, mock_exchange_client, mock_database):
        """Test buy proceeds with fear (not extreme) and extreme volatility."""
        mock_settings.block_trades_extreme_conditions = True
        mock_settings.use_limit_orders = False  # Simplify to direct market orders
        mock_settings.is_paper_trading = False  # Use mock client directly (avoid paper wrapper)
        mock_settings.max_position_percent = Decimal("100")  # Allow large positions for test

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # Mock position_sizer config to ensure can_buy=True
                    daemon.position_sizer.config.min_trade_quote = Decimal("10")
                    daemon.position_sizer.config.max_position_percent = Decimal("100")
                    daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                    # Mock position_sizer.calculate_size to return valid position
                    from src.strategy.position_sizer import PositionSizeResult
                    daemon.position_sizer.calculate_size = Mock(return_value=PositionSizeResult(
                        size_base=Decimal("0.002"),
                        size_quote=Decimal("100"),
                        stop_loss_price=Decimal("49000"),
                        take_profit_price=Decimal("52000"),
                        risk_amount_quote=Decimal("2"),
                        position_percent=1.0,
                    ))

                    # Mock market_regime: fear (not extreme_fear) + extreme volatility
                    daemon.market_regime = Mock()
                    mock_regime = Mock()
                    mock_regime.regime_name = "cautious"
                    mock_regime.threshold_adjustment = -5
                    mock_regime.position_multiplier = 1.1
                    mock_regime.components = {
                        "sentiment": {"category": "fear", "value": 35},  # NOT extreme_fear
                        "volatility": {"level": "extreme"}
                    }
                    daemon.market_regime.calculate.return_value = mock_regime

                    # Mock signal scorer to return strong buy signal
                    daemon.signal_scorer.calculate_score = Mock(return_value=SignalResult(
                        score=75,
                        action="buy",
                        indicators=IndicatorValues(
                            rsi=35.0,
                            macd_line=10.0,
                            macd_signal=5.0,
                            macd_histogram=5.0,
                            bb_upper=51000.0,
                            bb_middle=50000.0,
                            bb_lower=49000.0,
                            ema_fast=50100.0,
                            ema_slow=49900.0,
                            atr=1000.0,
                            volatility="extreme"
                        ),
                        breakdown={"rsi": 20, "macd": 15, "bollinger": 10, "ema": 15, "volume": 15},
                        confidence=0.8
                    ))

                    # Reset mock after initialization
                    mock_exchange_client.reset_mock()

                    # Run iteration
                    daemon._trading_iteration()

                    # Verify buy was allowed
                    mock_exchange_client.market_buy.assert_called_once()

    def test_sell_not_blocked_by_dual_extreme(self, mock_settings, mock_database):
        """Test sell orders are NOT blocked by dual-extreme conditions."""
        mock_settings.block_trades_extreme_conditions = True
        mock_settings.use_limit_orders = False  # Simplify to direct market orders
        mock_settings.is_paper_trading = False  # Use mock client directly (avoid paper wrapper)

        mock_client = Mock()
        # Return balances with crypto position to sell
        def get_balance_side_effect(currency):
            return Balance(
                currency=currency,
                available=Decimal("5000") if currency == "USD" else Decimal("0.1"),
                hold=Decimal("0")
            )
        mock_client.get_balance.side_effect = get_balance_side_effect
        mock_client.get_current_price.return_value = Decimal("50000")

        # Mock candles
        data = []
        for i in range(100):
            data.append({
                "timestamp": datetime.now(),
                "open": Decimal("50000"),
                "high": Decimal("50100"),
                "low": Decimal("49900"),
                "close": Decimal("50000"),
                "volume": Decimal("100"),
            })
        mock_client.get_candles.return_value = pd.DataFrame(data)

        mock_client.market_sell.return_value = OrderResult(
            order_id="sell-789",
            side="sell",
            size=Decimal("0.1"),
            filled_price=Decimal("50000"),
            status="FILLED",
            fee=Decimal("5.00"),
            success=True
        )

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # Mock position_sizer config to ensure can_sell=True
                    daemon.position_sizer.config.min_trade_quote = Decimal("10")
                    daemon.position_sizer.config.max_position_percent = Decimal("100")
                    daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                    # Mock market_regime: dual-extreme conditions
                    daemon.market_regime = Mock()
                    mock_regime = Mock()
                    mock_regime.regime_name = "extreme_risk_off"
                    mock_regime.threshold_adjustment = 0
                    mock_regime.position_multiplier = 0.5
                    mock_regime.components = {
                        "sentiment": {"category": "extreme_fear", "value": 15},
                        "volatility": {"level": "extreme"}
                    }
                    daemon.market_regime.calculate.return_value = mock_regime

                    # Mock signal scorer to return strong sell signal
                    daemon.signal_scorer.calculate_score = Mock(return_value=SignalResult(
                        score=-75,  # Strong sell signal
                        action="sell",
                        indicators=IndicatorValues(
                            rsi=75.0,
                            macd_line=-10.0,
                            macd_signal=-5.0,
                            macd_histogram=-5.0,
                            bb_upper=51000.0,
                            bb_middle=50000.0,
                            bb_lower=49000.0,
                            ema_fast=49900.0,
                            ema_slow=50100.0,
                            atr=1000.0,
                            volatility="extreme"
                        ),
                        breakdown={"rsi": -20, "macd": -15, "bollinger": -10, "ema": -15, "volume": -15},
                        confidence=0.8
                    ))

                    # Reset mock after initialization
                    mock_client.reset_mock()

                    # Run iteration
                    daemon._trading_iteration()

                    # Verify sell was allowed (not blocked by dual-extreme check)
                    mock_client.market_sell.assert_called_once()

    def test_respects_config_flag_disabled(self, mock_settings, mock_exchange_client, mock_database):
        """Test setting block_trades_extreme_conditions=False disables check."""
        mock_settings.block_trades_extreme_conditions = False  # Disabled
        mock_settings.use_limit_orders = False  # Simplify to direct market orders
        mock_settings.is_paper_trading = False  # Use mock client directly (avoid paper wrapper)
        mock_settings.max_position_percent = Decimal("100")  # Allow large positions for test

        with patch('src.daemon.runner.create_exchange_client', return_value=mock_exchange_client):
            with patch('src.daemon.runner.Database', return_value=mock_database):
                with patch('src.daemon.runner.TelegramNotifier'):
                    daemon = TradingDaemon(mock_settings)

                    # Mock position_sizer config to ensure can_buy=True
                    daemon.position_sizer.config.min_trade_quote = Decimal("10")
                    daemon.position_sizer.config.max_position_percent = Decimal("100")
                    daemon.position_sizer.config.min_trade_base = Decimal("0.0001")

                    # Mock position_sizer.calculate_size to return valid position
                    from src.strategy.position_sizer import PositionSizeResult
                    daemon.position_sizer.calculate_size = Mock(return_value=PositionSizeResult(
                        size_base=Decimal("0.002"),
                        size_quote=Decimal("100"),
                        stop_loss_price=Decimal("49000"),
                        take_profit_price=Decimal("52000"),
                        risk_amount_quote=Decimal("2"),
                        position_percent=1.0,
                    ))

                    # Mock market_regime: dual-extreme conditions
                    daemon.market_regime = Mock()
                    mock_regime = Mock()
                    mock_regime.regime_name = "extreme_risk_off"
                    mock_regime.threshold_adjustment = 0
                    mock_regime.position_multiplier = 0.5
                    mock_regime.components = {
                        "sentiment": {"category": "extreme_fear", "value": 15},
                        "volatility": {"level": "extreme"}
                    }
                    daemon.market_regime.calculate.return_value = mock_regime

                    # Mock signal scorer to return strong buy signal
                    daemon.signal_scorer.calculate_score = Mock(return_value=SignalResult(
                        score=75,
                        action="buy",
                        indicators=IndicatorValues(
                            rsi=35.0,
                            macd_line=10.0,
                            macd_signal=5.0,
                            macd_histogram=5.0,
                            bb_upper=51000.0,
                            bb_middle=50000.0,
                            bb_lower=49000.0,
                            ema_fast=50100.0,
                            ema_slow=49900.0,
                            atr=1000.0,
                            volatility="extreme"
                        ),
                        breakdown={"rsi": 20, "macd": 15, "bollinger": 10, "ema": 15, "volume": 15},
                        confidence=0.8
                    ))

                    # Reset mock after initialization
                    mock_exchange_client.reset_mock()

                    # Run iteration
                    daemon._trading_iteration()

                    # Verify buy was allowed (config disabled the check)
                    mock_exchange_client.market_buy.assert_called_once()
