"""
Comprehensive tests for the MarketRegime detection and strategy adaptation system.

Tests cover:
- Sentiment scoring (Fear & Greed Index: 0-100)
- Sentiment classification (extreme_fear, fear, neutral, greed, extreme_greed)
- Volatility level detection (low/normal/high/extreme)
- Trend direction classification (bullish/neutral/bearish)
- Regime adjustment calculation (threshold and position multipliers)
- Adjustment scale application (0.0 to 2.0)
- Combined regime effects on trading parameters
- Component toggles (sentiment/volatility/trend enabled/disabled)
- Regime name classification (risk_on, opportunistic, neutral, cautious, risk_off)
- API fallback when Fear & Greed unavailable
- Cache behavior (15-minute TTL)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from freezegun import freeze_time

from src.strategy.regime import (
    MarketRegime,
    RegimeConfig,
    RegimeAdjustments,
    get_cached_sentiment,
)
from src.ai.sentiment import FearGreedResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def regime():
    """Market regime with default configuration."""
    return MarketRegime()


@pytest.fixture
def custom_config():
    """Custom regime configuration."""
    return RegimeConfig(
        enabled=True,
        sentiment_enabled=True,
        volatility_enabled=True,
        trend_enabled=True,
        adjustment_scale=1.5,
    )


@pytest.fixture
def regime_custom(custom_config):
    """Market regime with custom configuration."""
    return MarketRegime(config=custom_config)


@pytest.fixture
def regime_disabled():
    """Market regime with regime system disabled."""
    config = RegimeConfig(enabled=False)
    return MarketRegime(config=config)


@pytest.fixture
def extreme_fear_sentiment():
    """Fear & Greed result indicating extreme fear."""
    return FearGreedResult(
        value=15,
        classification="Extreme Fear",
        timestamp=None,
    )


@pytest.fixture
def fear_sentiment():
    """Fear & Greed result indicating fear."""
    return FearGreedResult(
        value=35,
        classification="Fear",
        timestamp=None,
    )


@pytest.fixture
def neutral_sentiment():
    """Fear & Greed result indicating neutral."""
    return FearGreedResult(
        value=50,
        classification="Neutral",
        timestamp=None,
    )


@pytest.fixture
def greed_sentiment():
    """Fear & Greed result indicating greed."""
    return FearGreedResult(
        value=65,
        classification="Greed",
        timestamp=None,
    )


@pytest.fixture
def extreme_greed_sentiment():
    """Fear & Greed result indicating extreme greed."""
    return FearGreedResult(
        value=85,
        classification="Extreme Greed",
        timestamp=None,
    )


# ============================================================================
# Initialization Tests
# ============================================================================

def test_default_initialization():
    """Test regime initializes with default configuration."""
    regime = MarketRegime()

    assert regime.config.enabled is True
    assert regime.config.sentiment_enabled is True
    assert regime.config.volatility_enabled is True
    assert regime.config.trend_enabled is True
    assert regime.config.adjustment_scale == 1.0


def test_custom_configuration(custom_config):
    """Test regime accepts custom configuration."""
    regime = MarketRegime(config=custom_config)

    assert regime.config.enabled is True
    assert regime.config.sentiment_enabled is True
    assert regime.config.volatility_enabled is True
    assert regime.config.trend_enabled is True
    assert regime.config.adjustment_scale == 1.5


def test_adjustment_constants_defined():
    """Test regime adjustment constants are properly defined."""
    assert "extreme_fear" in MarketRegime.SENTIMENT_ADJUSTMENTS
    assert "fear" in MarketRegime.SENTIMENT_ADJUSTMENTS
    assert "neutral" in MarketRegime.SENTIMENT_ADJUSTMENTS
    assert "greed" in MarketRegime.SENTIMENT_ADJUSTMENTS
    assert "extreme_greed" in MarketRegime.SENTIMENT_ADJUSTMENTS

    assert "low" in MarketRegime.VOLATILITY_ADJUSTMENTS
    assert "normal" in MarketRegime.VOLATILITY_ADJUSTMENTS
    assert "high" in MarketRegime.VOLATILITY_ADJUSTMENTS
    assert "extreme" in MarketRegime.VOLATILITY_ADJUSTMENTS

    assert "bullish" in MarketRegime.TREND_ADJUSTMENTS
    assert "neutral" in MarketRegime.TREND_ADJUSTMENTS
    assert "bearish" in MarketRegime.TREND_ADJUSTMENTS


# ============================================================================
# Sentiment Classification Tests
# ============================================================================

def test_classify_sentiment_extreme_fear(regime):
    """Test sentiment classification for extreme fear (0-24)."""
    assert regime._classify_sentiment(0) == "extreme_fear"
    assert regime._classify_sentiment(15) == "extreme_fear"
    assert regime._classify_sentiment(24) == "extreme_fear"


def test_classify_sentiment_fear(regime):
    """Test sentiment classification for fear (25-44)."""
    assert regime._classify_sentiment(25) == "fear"
    assert regime._classify_sentiment(35) == "fear"
    assert regime._classify_sentiment(44) == "fear"


def test_classify_sentiment_neutral(regime):
    """Test sentiment classification for neutral (45-55)."""
    assert regime._classify_sentiment(45) == "neutral"
    assert regime._classify_sentiment(50) == "neutral"
    assert regime._classify_sentiment(55) == "neutral"


def test_classify_sentiment_greed(regime):
    """Test sentiment classification for greed (56-75)."""
    assert regime._classify_sentiment(56) == "greed"
    assert regime._classify_sentiment(65) == "greed"
    assert regime._classify_sentiment(75) == "greed"


def test_classify_sentiment_extreme_greed(regime):
    """Test sentiment classification for extreme greed (76-100)."""
    assert regime._classify_sentiment(76) == "extreme_greed"
    assert regime._classify_sentiment(85) == "extreme_greed"
    assert regime._classify_sentiment(100) == "extreme_greed"


def test_classify_sentiment_boundary_values(regime):
    """Test sentiment classification at exact boundary values."""
    assert regime._classify_sentiment(24) == "extreme_fear"
    assert regime._classify_sentiment(25) == "fear"

    assert regime._classify_sentiment(44) == "fear"
    assert regime._classify_sentiment(45) == "neutral"

    assert regime._classify_sentiment(55) == "neutral"
    assert regime._classify_sentiment(56) == "greed"

    assert regime._classify_sentiment(75) == "greed"
    assert regime._classify_sentiment(76) == "extreme_greed"


# ============================================================================
# Regime Calculation Tests
# ============================================================================

def test_calculate_returns_adjustments(regime, neutral_sentiment):
    """Test calculate returns RegimeAdjustments."""
    result = regime.calculate(
        sentiment=neutral_sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    assert isinstance(result, RegimeAdjustments)
    assert isinstance(result.threshold_adjustment, int)
    assert isinstance(result.position_multiplier, float)
    assert isinstance(result.regime_name, str)
    assert isinstance(result.components, dict)


def test_disabled_regime_returns_neutral(regime_disabled):
    """Test disabled regime returns neutral adjustments."""
    result = regime_disabled.calculate(
        sentiment=None,
        volatility="high",
        trend="bearish",
        signal_action="buy",
    )

    assert result.threshold_adjustment == 0
    assert result.position_multiplier == 1.0
    assert result.regime_name == "disabled"
    assert result.components == {}


def test_extreme_fear_lowers_threshold(regime, extreme_fear_sentiment):
    """Test extreme fear reduces threshold (easier to trade)."""
    result = regime.calculate(
        sentiment=extreme_fear_sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Extreme fear should lower threshold (negative adjustment)
    assert result.threshold_adjustment <= 0
    assert "sentiment" in result.components
    assert result.components["sentiment"]["category"] == "extreme_fear"


def test_extreme_fear_increases_position(regime, extreme_fear_sentiment):
    """Test extreme fear increases position multiplier."""
    result = regime.calculate(
        sentiment=extreme_fear_sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Extreme fear should increase position (>1.0)
    assert result.position_multiplier >= 1.0


def test_extreme_greed_raises_threshold(regime, extreme_greed_sentiment):
    """Test extreme greed raises threshold (harder to trade)."""
    result = regime.calculate(
        sentiment=extreme_greed_sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Extreme greed should raise threshold (positive adjustment)
    assert result.threshold_adjustment >= 0
    assert result.components["sentiment"]["category"] == "extreme_greed"


def test_extreme_greed_decreases_position(regime, extreme_greed_sentiment):
    """Test extreme greed decreases position multiplier."""
    result = regime.calculate(
        sentiment=extreme_greed_sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Extreme greed should decrease position (<1.0)
    assert result.position_multiplier <= 1.0


# ============================================================================
# Volatility Adjustment Tests
# ============================================================================

def test_low_volatility_lowers_threshold(regime):
    """Test low volatility makes it easier to trade."""
    result = regime.calculate(
        sentiment=None,
        volatility="low",
        trend="neutral",
        signal_action="buy",
    )

    # Low volatility should lower threshold
    assert result.threshold_adjustment <= 0
    assert "volatility" in result.components
    assert result.components["volatility"]["level"] == "low"


def test_high_volatility_raises_threshold(regime):
    """Test high volatility makes it harder to trade."""
    result = regime.calculate(
        sentiment=None,
        volatility="high",
        trend="neutral",
        signal_action="buy",
    )

    # High volatility should raise threshold
    assert result.threshold_adjustment >= 0
    assert result.components["volatility"]["level"] == "high"


def test_extreme_volatility_maximum_caution(regime):
    """Test extreme volatility applies maximum caution."""
    result = regime.calculate(
        sentiment=None,
        volatility="extreme",
        trend="neutral",
        signal_action="buy",
    )

    # Extreme volatility should have highest threshold adjustment
    assert result.threshold_adjustment >= 5
    # And lowest position multiplier
    assert result.position_multiplier <= 1.0


# ============================================================================
# Trend Adjustment Tests
# ============================================================================

def test_bullish_trend_favors_buys(regime):
    """Test bullish trend lowers buy threshold."""
    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="bullish",
        signal_action="buy",
    )

    # Bullish trend should make buying easier
    assert result.threshold_adjustment <= 0


def test_bullish_trend_penalizes_sells(regime):
    """Test bullish trend raises sell threshold."""
    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="bullish",
        signal_action="sell",
    )

    # Bullish trend should make selling harder
    assert result.threshold_adjustment >= 0


def test_bearish_trend_penalizes_buys(regime):
    """Test bearish trend raises buy threshold."""
    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="bearish",
        signal_action="buy",
    )

    # Bearish trend should make buying harder
    assert result.threshold_adjustment >= 0


def test_bearish_trend_favors_sells(regime):
    """Test bearish trend lowers sell threshold."""
    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="bearish",
        signal_action="sell",
    )

    # Bearish trend should make selling easier
    assert result.threshold_adjustment <= 0


def test_neutral_trend_no_adjustment(regime):
    """Test neutral trend has no effect."""
    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Neutral trend component should be zero
    if "trend" in result.components:
        assert result.components["trend"]["threshold_adj"] == 0


def test_hold_signal_no_trend_adjustment(regime):
    """Test hold signal gets no trend adjustment."""
    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="bullish",
        signal_action="hold",
    )

    # Hold should not apply trend adjustments
    if "trend" in result.components:
        assert result.components["trend"]["threshold_adj"] == 0


# ============================================================================
# Combined Adjustment Tests
# ============================================================================

def test_combined_extreme_fear_and_low_volatility(regime, extreme_fear_sentiment):
    """Test combined extreme fear + low volatility = very aggressive."""
    result = regime.calculate(
        sentiment=extreme_fear_sentiment,
        volatility="low",
        trend="neutral",
        signal_action="buy",
    )

    # Both should lower threshold
    assert result.threshold_adjustment <= -10
    # Should be risk_on or opportunistic
    assert result.regime_name in ["risk_on", "opportunistic"]


def test_combined_extreme_greed_and_high_volatility(regime, extreme_greed_sentiment):
    """Test combined extreme greed + high volatility = very cautious."""
    result = regime.calculate(
        sentiment=extreme_greed_sentiment,
        volatility="high",
        trend="neutral",
        signal_action="buy",
    )

    # Both should raise threshold
    assert result.threshold_adjustment >= 10
    # Should be risk_off or cautious
    assert result.regime_name in ["risk_off", "cautious"]


def test_threshold_adjustment_clamped_to_range(regime, extreme_fear_sentiment):
    """Test threshold adjustment is clamped to -20 to +20."""
    # Try to create extreme adjustments
    result = regime.calculate(
        sentiment=extreme_fear_sentiment,
        volatility="low",
        trend="bullish",
        signal_action="buy",
    )

    assert -20 <= result.threshold_adjustment <= 20


def test_position_multiplier_clamped_to_range(regime, extreme_greed_sentiment):
    """Test position multiplier is clamped to 0.5 to 1.5."""
    # Try to create extreme multipliers
    result = regime.calculate(
        sentiment=extreme_greed_sentiment,
        volatility="extreme",
        trend="neutral",
        signal_action="buy",
    )

    assert 0.5 <= result.position_multiplier <= 1.5


# ============================================================================
# Adjustment Scale Tests
# ============================================================================

def test_adjustment_scale_amplifies_effects(regime_custom, extreme_fear_sentiment):
    """Test adjustment_scale=1.5 amplifies regime effects."""
    # Custom regime has scale=1.5
    result = regime_custom.calculate(
        sentiment=extreme_fear_sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # With scale 1.5, adjustments should be larger
    assert result.threshold_adjustment != 0


def test_adjustment_scale_zero_neutralizes():
    """Test adjustment_scale=0 neutralizes all adjustments."""
    config = RegimeConfig(adjustment_scale=0.0)
    regime = MarketRegime(config=config)

    result = regime.calculate(
        sentiment=FearGreedResult(value=10, classification="Extreme Fear", timestamp=None),
        volatility="extreme",
        trend="bearish",
        signal_action="buy",
    )

    # With scale 0, all adjustments should be zero
    assert result.threshold_adjustment == 0
    assert result.position_multiplier == 1.0


# ============================================================================
# Component Toggle Tests
# ============================================================================

def test_sentiment_disabled_ignores_fear_greed(extreme_fear_sentiment):
    """Test sentiment_enabled=False ignores Fear & Greed."""
    config = RegimeConfig(sentiment_enabled=False)
    regime = MarketRegime(config=config)

    result = regime.calculate(
        sentiment=extreme_fear_sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Sentiment component should not exist
    assert "sentiment" not in result.components


def test_volatility_disabled_ignores_volatility():
    """Test volatility_enabled=False ignores volatility level."""
    config = RegimeConfig(volatility_enabled=False)
    regime = MarketRegime(config=config)

    result = regime.calculate(
        sentiment=None,
        volatility="extreme",
        trend="neutral",
        signal_action="buy",
    )

    # Volatility component should not exist
    assert "volatility" not in result.components


def test_trend_disabled_ignores_trend():
    """Test trend_enabled=False ignores trend direction."""
    config = RegimeConfig(trend_enabled=False)
    regime = MarketRegime(config=config)

    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="bearish",
        signal_action="buy",
    )

    # Trend component should not exist
    assert "trend" not in result.components


# ============================================================================
# Regime Name Classification Tests
# ============================================================================

def test_regime_name_risk_on():
    """Test regime name is 'risk_on' for threshold <= -10."""
    regime = MarketRegime()

    result = regime.calculate(
        sentiment=FearGreedResult(value=10, classification="Extreme Fear", timestamp=None),
        volatility="low",
        trend="neutral",
        signal_action="buy",
    )

    if result.threshold_adjustment <= -10:
        assert result.regime_name == "risk_on"


def test_regime_name_opportunistic():
    """Test regime name is 'opportunistic' for -10 < threshold <= -5."""
    regime = MarketRegime()

    result = regime.calculate(
        sentiment=FearGreedResult(value=30, classification="Fear", timestamp=None),
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    if -10 < result.threshold_adjustment <= -5:
        assert result.regime_name == "opportunistic"


def test_regime_name_neutral():
    """Test regime name is 'neutral' for -5 < threshold < 5."""
    regime = MarketRegime()

    result = regime.calculate(
        sentiment=FearGreedResult(value=50, classification="Neutral", timestamp=None),
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    if -5 < result.threshold_adjustment < 5:
        assert result.regime_name == "neutral"


def test_regime_name_cautious():
    """Test regime name is 'cautious' for 5 <= threshold < 10."""
    regime = MarketRegime()

    result = regime.calculate(
        sentiment=FearGreedResult(value=70, classification="Greed", timestamp=None),
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    if 5 <= result.threshold_adjustment < 10:
        assert result.regime_name == "cautious"


def test_regime_name_risk_off():
    """Test regime name is 'risk_off' for threshold >= 10."""
    regime = MarketRegime()

    result = regime.calculate(
        sentiment=FearGreedResult(value=90, classification="Extreme Greed", timestamp=None),
        volatility="high",
        trend="neutral",
        signal_action="buy",
    )

    if result.threshold_adjustment >= 10:
        assert result.regime_name == "risk_off"


# ============================================================================
# None Sentiment Tests
# ============================================================================

def test_none_sentiment_skips_sentiment_component(regime):
    """Test None sentiment is handled gracefully."""
    result = regime.calculate(
        sentiment=None,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Sentiment component should not exist
    assert "sentiment" not in result.components


def test_sentiment_value_none_skips_component(regime):
    """Test sentiment with value=None is skipped."""
    sentiment = FearGreedResult(value=None, classification="Unknown", timestamp=None)

    result = regime.calculate(
        sentiment=sentiment,
        volatility="normal",
        trend="neutral",
        signal_action="buy",
    )

    # Sentiment component should not exist when value is None
    assert "sentiment" not in result.components


# ============================================================================
# Cached Sentiment Tests
# ============================================================================

@pytest.mark.asyncio
@freeze_time("2025-01-01 12:00:00")
async def test_get_cached_sentiment_fetches_first_time():
    """Test get_cached_sentiment fetches on first call."""
    with patch("src.strategy.regime.fetch_fear_greed_index", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FearGreedResult(value=50, classification="Neutral", timestamp=None)

        # Clear cache
        import src.strategy.regime as regime_module
        regime_module._sentiment_cache = None
        regime_module._sentiment_last_fetch = None

        result = await get_cached_sentiment()

        assert result is not None
        assert result.value == 50
        mock_fetch.assert_called_once()


@pytest.mark.asyncio
@freeze_time("2025-01-01 12:00:00")
async def test_get_cached_sentiment_uses_cache_within_ttl():
    """Test get_cached_sentiment uses cache within 15-minute TTL."""
    with patch("src.strategy.regime.fetch_fear_greed_index", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FearGreedResult(value=50, classification="Neutral", timestamp=None)

        # Clear cache and fetch first time
        import src.strategy.regime as regime_module
        regime_module._sentiment_cache = None
        regime_module._sentiment_last_fetch = None

        result1 = await get_cached_sentiment()

        # Move forward 10 minutes (within TTL)
        with freeze_time("2025-01-01 12:10:00"):
            result2 = await get_cached_sentiment()

            # Should use cache, only 1 fetch
            assert mock_fetch.call_count == 1
            assert result2 is not None


@pytest.mark.asyncio
@freeze_time("2025-01-01 12:00:00")
async def test_get_cached_sentiment_refetches_after_ttl():
    """Test get_cached_sentiment refetches after 15-minute TTL."""
    with patch("src.strategy.regime.fetch_fear_greed_index", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = FearGreedResult(value=50, classification="Neutral", timestamp=None)

        # Clear cache and fetch first time
        import src.strategy.regime as regime_module
        regime_module._sentiment_cache = None
        regime_module._sentiment_last_fetch = None

        result1 = await get_cached_sentiment()

        # Move forward 16 minutes (past TTL)
        with freeze_time("2025-01-01 12:16:00"):
            result2 = await get_cached_sentiment()

            # Should refetch after TTL, 2 fetches total
            assert mock_fetch.call_count == 2


@pytest.mark.asyncio
@freeze_time("2025-01-01 12:00:00")
async def test_get_cached_sentiment_handles_api_failure():
    """Test get_cached_sentiment handles API failures gracefully."""
    with patch("src.strategy.regime.fetch_fear_greed_index", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = Exception("API Error")

        # Clear cache
        import src.strategy.regime as regime_module
        regime_module._sentiment_cache = None
        regime_module._sentiment_last_fetch = None

        result = await get_cached_sentiment()

        # Should return None on error with no cache
        assert result is None


@pytest.mark.asyncio
@freeze_time("2025-01-01 12:00:00")
async def test_get_cached_sentiment_keeps_stale_cache_on_error():
    """Test get_cached_sentiment keeps stale cache when API fails."""
    with patch("src.strategy.regime.fetch_fear_greed_index", new_callable=AsyncMock) as mock_fetch:
        # First call succeeds
        mock_fetch.return_value = FearGreedResult(value=50, classification="Neutral", timestamp=None)

        # Clear cache
        import src.strategy.regime as regime_module
        regime_module._sentiment_cache = None
        regime_module._sentiment_last_fetch = None

        result1 = await get_cached_sentiment()
        assert result1 is not None

        # Move forward past TTL and make API fail
        with freeze_time("2025-01-01 12:20:00"):
            mock_fetch.side_effect = Exception("API Error")

            result2 = await get_cached_sentiment()

            # Should keep stale cache instead of returning None
            assert result2 is not None
            assert result2.value == 50


# ============================================================================
# Trend-Aware Sentiment Modifier Tests
# ============================================================================

class TestTrendAwareSentimentModifiers:
    """Test the trend-aware sentiment modification system."""

    # ------------------------------------------------------------------------
    # EXTREME FEAR + BUY Tests
    # ------------------------------------------------------------------------

    def test_extreme_fear_bearish_buy_nullifies_discount(self, regime, extreme_fear_sentiment):
        """Test extreme_fear + bearish + buy nullifies the threshold discount.

        Fear in a downtrend is justified, not an opportunity.
        """
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )

        # threshold_mult=0.0 should eliminate the -10 discount
        assert result.components["sentiment"]["threshold_adj"] == 0
        assert result.components["sentiment"]["trend_modified"] is True
        assert result.components["sentiment"]["original_threshold_adj"] == -10

    def test_extreme_fear_bullish_buy_amplifies_discount(self, regime, extreme_fear_sentiment):
        """Test extreme_fear + bullish + buy amplifies the opportunity.

        Fear in an uptrend is a contrarian buying opportunity.
        """
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="buy",
        )

        # threshold_mult=1.2 should amplify -10 to -12
        assert result.components["sentiment"]["threshold_adj"] == -12
        assert result.components["sentiment"]["trend_modified"] is True

    def test_extreme_fear_neutral_buy_reduces_discount(self, regime, extreme_fear_sentiment):
        """Test extreme_fear + neutral + buy slightly reduces discount."""
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="neutral",
            signal_action="buy",
        )

        # threshold_mult=0.8 should reduce -10 to -8
        assert result.components["sentiment"]["threshold_adj"] == -8
        assert result.components["sentiment"]["trend_modified"] is True

    # ------------------------------------------------------------------------
    # EXTREME FEAR + SELL Tests
    # ------------------------------------------------------------------------

    def test_extreme_fear_bullish_sell_reduces_ease(self, regime, extreme_fear_sentiment):
        """Test extreme_fear + bullish + sell makes selling harder.

        Wall of worry - don't panic sell during uptrends.
        """
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="sell",
        )

        # threshold_mult=0.3 should reduce -10 to -3
        assert result.components["sentiment"]["threshold_adj"] == -3
        assert result.components["sentiment"]["trend_modified"] is True

    def test_extreme_fear_bearish_sell_moderate_ease(self, regime, extreme_fear_sentiment):
        """Test extreme_fear + bearish + sell allows cautious selling.

        Fear is justified but extreme fear can mark bottoms.
        """
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="sell",
        )

        # threshold_mult=0.7 should reduce -10 to -7
        assert result.components["sentiment"]["threshold_adj"] == -7
        assert result.components["sentiment"]["trend_modified"] is True

    # ------------------------------------------------------------------------
    # EXTREME GREED + BUY Tests
    # ------------------------------------------------------------------------

    def test_extreme_greed_bullish_buy_reduces_penalty(self, regime, extreme_greed_sentiment):
        """Test extreme_greed + bullish + buy reduces the penalty.

        Momentum is real - don't penalize buying too harshly.
        """
        result = regime.calculate(
            sentiment=extreme_greed_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="buy",
        )

        # threshold_mult=0.5 should reduce +15 to +8 (round(15 * 0.5) = 8)
        assert result.components["sentiment"]["threshold_adj"] == 8
        assert result.components["sentiment"]["trend_modified"] is True
        assert result.components["sentiment"]["original_threshold_adj"] == 15

    def test_extreme_greed_bearish_buy_amplifies_penalty(self, regime, extreme_greed_sentiment):
        """Test extreme_greed + bearish + buy amplifies the penalty.

        Denial phase - don't buy when trend is down and people are euphoric.
        """
        result = regime.calculate(
            sentiment=extreme_greed_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )

        # threshold_mult=1.5 should amplify +15 to +22, clamped to +20
        assert result.components["sentiment"]["threshold_adj"] >= 20
        assert result.components["sentiment"]["trend_modified"] is True

    # ------------------------------------------------------------------------
    # EXTREME GREED + SELL Tests (PRIME OPPORTUNITY)
    # ------------------------------------------------------------------------

    def test_extreme_greed_bullish_sell_prime_opportunity(self, regime, extreme_greed_sentiment):
        """Test extreme_greed + bullish + sell is the PRIME selling opportunity.

        'Be fearful when others are greedy' - take profits at euphoria.
        """
        result = regime.calculate(
            sentiment=extreme_greed_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="sell",
        )

        # threshold_mult=1.5 should amplify +15 to +22 (clamped to +20)
        # This makes it EASIER to sell (threshold penalty reduced relative to signal)
        # Note: For sells, the sentiment threshold still adds, but the signal is negative
        # So a higher positive sentiment_threshold means less penalty on the negative score
        assert result.components["sentiment"]["threshold_adj"] >= 15
        assert result.components["sentiment"]["trend_modified"] is True
        # Position modifier of 1.2 means we take MORE of the base position change
        # Base extreme_greed = 0.75, so deviation = -0.25
        # modified = 1.0 + (-0.25 * 1.2) = 0.7 (smaller position overall, but larger sell)
        # This is correct - we're selling MORE aggressively

    def test_extreme_greed_bearish_sell_denial_phase(self, regime, extreme_greed_sentiment):
        """Test extreme_greed + bearish + sell catches denial phase."""
        result = regime.calculate(
            sentiment=extreme_greed_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="sell",
        )

        # threshold_mult=1.3 should amplify +15 to +19
        assert result.components["sentiment"]["threshold_adj"] >= 15
        assert result.components["sentiment"]["trend_modified"] is True

    # ------------------------------------------------------------------------
    # FEAR (non-extreme) Tests
    # ------------------------------------------------------------------------

    def test_fear_bearish_buy_reduces_discount(self, regime, fear_sentiment):
        """Test fear + bearish + buy reduces the discount significantly."""
        result = regime.calculate(
            sentiment=fear_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )

        # threshold_mult=0.3 should reduce -5 to -1 (int(-5 * 0.3) = -1)
        assert result.components["sentiment"]["threshold_adj"] >= -2
        assert result.components["sentiment"]["trend_modified"] is True

    def test_fear_bullish_buy_slight_amplification(self, regime, fear_sentiment):
        """Test fear + bullish + buy slightly amplifies the opportunity."""
        result = regime.calculate(
            sentiment=fear_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="buy",
        )

        # threshold_mult=1.1 should amplify -5 to -6 (round(-5 * 1.1) = round(-5.5) = -6)
        assert result.components["sentiment"]["threshold_adj"] == -6
        assert result.components["sentiment"]["trend_modified"] is True

    # ------------------------------------------------------------------------
    # GREED (non-extreme) Tests
    # ------------------------------------------------------------------------

    def test_greed_bullish_buy_reduces_penalty(self, regime, greed_sentiment):
        """Test greed + bullish + buy reduces the penalty."""
        result = regime.calculate(
            sentiment=greed_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="buy",
        )

        # threshold_mult=0.7 should reduce +5 to +4 (round(5 * 0.7) = round(3.5) = 4)
        assert result.components["sentiment"]["threshold_adj"] == 4
        assert result.components["sentiment"]["trend_modified"] is True

    def test_greed_bullish_sell_good_for_profits(self, regime, greed_sentiment):
        """Test greed + bullish + sell is a good time for profits."""
        result = regime.calculate(
            sentiment=greed_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="sell",
        )

        # threshold_mult=1.2 should amplify +5 to +6
        assert result.components["sentiment"]["threshold_adj"] == 6
        assert result.components["sentiment"]["trend_modified"] is True

    # ------------------------------------------------------------------------
    # NEUTRAL Sentiment Tests (should use defaults)
    # ------------------------------------------------------------------------

    def test_neutral_sentiment_not_modified(self, regime, neutral_sentiment):
        """Test neutral sentiment is not affected by trend modifiers."""
        result = regime.calculate(
            sentiment=neutral_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )

        # Neutral sentiment has threshold=0, so no modification
        assert result.components["sentiment"]["threshold_adj"] == 0
        # Should not be marked as trend_modified since base is 0
        assert result.components["sentiment"].get("trend_modified", False) is False

    # ------------------------------------------------------------------------
    # Position Multiplier Tests
    # ------------------------------------------------------------------------

    def test_extreme_fear_bearish_buy_reduces_position(self, regime, extreme_fear_sentiment):
        """Test extreme_fear + bearish + buy reduces position size."""
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )

        # position_mult=0.7 should reduce the position
        # Base extreme_fear position is 1.25, modified by 0.7
        assert result.components["sentiment"]["position_mult"] < 1.25

    def test_extreme_greed_bullish_sell_amplifies_position_effect(self, regime, extreme_greed_sentiment):
        """Test extreme_greed + bullish + sell amplifies the position reduction.

        During prime selling opportunity, we sell MORE aggressively.
        Base extreme_greed position = 0.75 (sell 25% less).
        With position_mult=1.2, we amplify this: 1.0 + (-0.25 * 1.2) = 0.7.
        This means we sell 30% more than normal (more aggressive selling).
        """
        result = regime.calculate(
            sentiment=extreme_greed_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="sell",
        )

        # With modifier 1.2 on base 0.75: 1.0 + (0.75 - 1.0) * 1.2 = 0.7
        assert result.components["sentiment"]["position_mult"] == 0.7

    # ------------------------------------------------------------------------
    # Modifier Table Completeness Tests
    # ------------------------------------------------------------------------

    def test_all_extreme_fear_combinations_defined(self):
        """Test all extreme_fear trend/signal combinations are in modifier table."""
        from src.strategy.regime import MarketRegime

        for trend in ["bullish", "bearish", "neutral"]:
            for signal in ["buy", "sell"]:
                key = ("extreme_fear", trend, signal)
                assert key in MarketRegime.SENTIMENT_TREND_MODIFIERS, f"Missing: {key}"

    def test_all_extreme_greed_combinations_defined(self):
        """Test all extreme_greed trend/signal combinations are in modifier table."""
        from src.strategy.regime import MarketRegime

        for trend in ["bullish", "bearish", "neutral"]:
            for signal in ["buy", "sell"]:
                key = ("extreme_greed", trend, signal)
                assert key in MarketRegime.SENTIMENT_TREND_MODIFIERS, f"Missing: {key}"

    def test_all_fear_combinations_defined(self):
        """Test all fear trend/signal combinations are in modifier table."""
        from src.strategy.regime import MarketRegime

        for trend in ["bullish", "bearish", "neutral"]:
            for signal in ["buy", "sell"]:
                key = ("fear", trend, signal)
                assert key in MarketRegime.SENTIMENT_TREND_MODIFIERS, f"Missing: {key}"

    def test_all_greed_combinations_defined(self):
        """Test all greed trend/signal combinations are in modifier table."""
        from src.strategy.regime import MarketRegime

        for trend in ["bullish", "bearish", "neutral"]:
            for signal in ["buy", "sell"]:
                key = ("greed", trend, signal)
                assert key in MarketRegime.SENTIMENT_TREND_MODIFIERS, f"Missing: {key}"

    def test_hold_signal_uses_default_modifier(self, regime, extreme_fear_sentiment):
        """Test hold signal uses default 1.0 modifier (not in table)."""
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="hold",
        )

        # Hold is not in the modifier table, should use default 1.0
        # So extreme_fear -10 should be applied in full
        assert result.components["sentiment"]["threshold_adj"] == -10
        assert result.components["sentiment"].get("trend_modified", False) is False

    # ------------------------------------------------------------------------
    # Integration Tests
    # ------------------------------------------------------------------------

    def test_real_scenario_fear_bearish_buy_prevented(self, regime, extreme_fear_sentiment):
        """Test the actual scenario from postmortem: fear + bearish should not help buys.

        This was the bug: extreme fear lowered threshold during bearish trend,
        inviting counter-trend trades.
        """
        # Old behavior: threshold -10 (easier to buy)
        # New behavior: threshold 0 (no discount for buying in downtrend)
        result = regime.calculate(
            sentiment=extreme_fear_sentiment,
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )

        # The sentiment component should NOT lower the threshold
        assert result.components["sentiment"]["threshold_adj"] >= 0, \
            "Fear should not lower threshold for buys in bearish trend"

    def test_real_scenario_greed_bullish_sell_encouraged(self, regime, extreme_greed_sentiment):
        """Test that greed + bullish makes selling EASIER (higher threshold adjustment).

        This is the prime opportunity to take profits.
        """
        result_neutral_trend = regime.calculate(
            sentiment=extreme_greed_sentiment,
            volatility="normal",
            trend="neutral",
            signal_action="sell",
        )

        result_bullish_trend = regime.calculate(
            sentiment=extreme_greed_sentiment,
            volatility="normal",
            trend="bullish",
            signal_action="sell",
        )

        # Bullish trend should amplify the greed effect for sells
        assert result_bullish_trend.components["sentiment"]["threshold_adj"] >= \
               result_neutral_trend.components["sentiment"]["threshold_adj"]


# ============================================================================
# Custom Modifiers Tests
# ============================================================================

class TestCustomModifiers:
    """Test custom sentiment-trend modifiers configuration."""

    def test_custom_modifiers_override_defaults(self):
        """Test custom modifiers override hardcoded defaults."""
        # Create custom modifiers with all 30 required entries (5 sentiments × 3 trends × 2 signals)
        custom = {
            "extreme_fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
        }

        regime = MarketRegime(custom_modifiers=custom)

        # Verify the custom value is used (all should be 1.0, 1.0)
        assert regime.sentiment_trend_modifiers[("extreme_fear", "bearish", "buy")] == {
            "threshold_mult": 1.0, "position_mult": 1.0
        }
        # Verify defaults would have been different
        assert MarketRegime.SENTIMENT_TREND_MODIFIERS[("extreme_fear", "bearish", "buy")]["threshold_mult"] == 0.0

    def test_custom_modifiers_actually_applied(self):
        """Test custom modifiers are actually used in calculations."""
        # Create custom modifiers that neutralize extreme_fear in bearish trends
        custom = {
            "extreme_fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},  # Full discount applied
            "extreme_fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
        }

        regime = MarketRegime(custom_modifiers=custom)

        # Calculate with extreme fear in bearish trend
        result = regime.calculate(
            sentiment=FearGreedResult(value=15, classification="Extreme Fear", timestamp=None),
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )

        # With threshold_mult=1.0, the full -10 discount should apply
        assert result.components["sentiment"]["threshold_adj"] == -10

        # Compare to default regime which should nullify the discount
        default_regime = MarketRegime()
        default_result = default_regime.calculate(
            sentiment=FearGreedResult(value=15, classification="Extreme Fear", timestamp=None),
            volatility="normal",
            trend="bearish",
            signal_action="buy",
        )
        # Default should nullify (threshold_mult=0.0)
        assert default_result.components["sentiment"]["threshold_adj"] == 0

    def test_incomplete_custom_modifiers_falls_back(self):
        """Test incomplete custom modifiers fall back to defaults."""
        # Only provide 5 entries instead of 30
        incomplete = {
            "extreme_fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
        }

        regime = MarketRegime(custom_modifiers=incomplete)

        # Should fall back to defaults
        assert regime.sentiment_trend_modifiers == MarketRegime.SENTIMENT_TREND_MODIFIERS

    def test_invalid_key_format_skipped(self):
        """Test invalid key formats are skipped during parsing."""
        # Mix valid and invalid keys
        mixed = {
            # Valid
            "extreme_fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            # Invalid format
            "invalid_key": {"threshold_mult": 1.0, "position_mult": 1.0},
            "another_bad_one": {"threshold_mult": 1.0, "position_mult": 1.0},
            # More valid
            "fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
        }

        regime = MarketRegime(custom_modifiers=mixed)

        # Invalid keys should be skipped, but we only have 30 valid ones so it should work
        assert len(regime.sentiment_trend_modifiers) == 30

    def test_string_parsing_handles_underscores_in_sentiment(self):
        """Test robust parsing handles sentiment names with underscores."""
        # Use a sentiment name with underscores
        custom = {
            "extreme_fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "fear_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "neutral_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bearish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_bullish_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_buy": {"threshold_mult": 1.0, "position_mult": 1.0},
            "extreme_greed_neutral_sell": {"threshold_mult": 1.0, "position_mult": 1.0},
        }

        regime = MarketRegime(custom_modifiers=custom)

        # Should correctly parse extreme_fear as sentiment
        assert ("extreme_fear", "bearish", "buy") in regime.sentiment_trend_modifiers
        assert regime.sentiment_trend_modifiers[("extreme_fear", "bearish", "buy")] == {
            "threshold_mult": 1.0, "position_mult": 1.0
        }

    def test_no_custom_modifiers_uses_defaults(self):
        """Test MarketRegime without custom_modifiers uses defaults."""
        regime = MarketRegime()

        # Should use hardcoded defaults
        assert regime.sentiment_trend_modifiers == MarketRegime.SENTIMENT_TREND_MODIFIERS

    def test_none_custom_modifiers_uses_defaults(self):
        """Test MarketRegime with custom_modifiers=None uses defaults."""
        regime = MarketRegime(custom_modifiers=None)

        # Should use hardcoded defaults
        assert regime.sentiment_trend_modifiers == MarketRegime.SENTIMENT_TREND_MODIFIERS
