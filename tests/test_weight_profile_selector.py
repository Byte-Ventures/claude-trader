"""
Tests for the AI-driven weight profile selector.

Tests cover:
- Profile selection for each market condition
- Fallback behavior when AI unavailable
- Caching behavior (should_update)
- Circuit breaker after consecutive failures
- Invalid response handling
- Weight profile updates
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.strategy.weight_profile_selector import (
    WeightProfileSelector,
    ProfileSelectorConfig,
    ProfileSelection,
    WEIGHT_PROFILES,
)
from src.strategy.signal_scorer import SignalWeights


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default configuration for testing."""
    return ProfileSelectorConfig(
        enabled=True,
        cache_minutes=15,
        fallback_profile="default",
        model="test-model",
    )


@pytest.fixture
def selector(config):
    """Create a weight profile selector with default config."""
    return WeightProfileSelector(
        api_key="test-api-key",
        config=config,
    )


# ============================================================================
# Profile Configuration Tests
# ============================================================================

def test_weight_profiles_exist():
    """Test all expected profiles are defined."""
    expected_profiles = ["trending", "ranging", "volatile", "default"]
    for profile in expected_profiles:
        assert profile in WEIGHT_PROFILES
        assert isinstance(WEIGHT_PROFILES[profile], SignalWeights)


def test_trending_profile_emphasizes_macd_ema():
    """Trending profile should emphasize MACD and EMA."""
    profile = WEIGHT_PROFILES["trending"]
    assert profile.macd >= profile.rsi
    assert profile.ema >= profile.rsi
    assert profile.macd + profile.ema > profile.rsi + profile.bollinger


def test_ranging_profile_emphasizes_rsi_bb():
    """Ranging profile should emphasize RSI and Bollinger."""
    profile = WEIGHT_PROFILES["ranging"]
    assert profile.rsi >= profile.macd
    assert profile.bollinger >= profile.macd
    assert profile.rsi + profile.bollinger > profile.macd + profile.ema


def test_volatile_profile_emphasizes_bollinger():
    """Volatile profile should have higher Bollinger weight."""
    profile = WEIGHT_PROFILES["volatile"]
    assert profile.bollinger >= profile.rsi
    assert profile.bollinger >= profile.macd


def test_default_profile_is_balanced():
    """Default profile should have balanced weights."""
    profile = WEIGHT_PROFILES["default"]
    # RSI and MACD should be equal
    assert profile.rsi == profile.macd


def test_all_profiles_sum_to_100():
    """All profile weights should sum to 100."""
    for name, profile in WEIGHT_PROFILES.items():
        total = profile.rsi + profile.macd + profile.bollinger + profile.ema + profile.volume
        assert total == 100, f"Profile '{name}' weights sum to {total}, expected 100"


# ============================================================================
# Initialization Tests
# ============================================================================

def test_selector_initialization(selector):
    """Test selector initializes with default values."""
    assert selector.config.enabled is True
    assert selector.config.cache_minutes == 15
    assert selector.config.fallback_profile == "default"
    assert selector._cached_selection is None
    assert selector._consecutive_failures == 0


def test_selector_get_current_weights_returns_default(selector):
    """Without cache, should return fallback profile weights."""
    weights = selector.get_current_weights()
    assert weights == WEIGHT_PROFILES["default"]


def test_selector_get_current_profile_returns_fallback(selector):
    """Without cache, should return fallback profile name."""
    profile = selector.get_current_profile()
    assert profile == "default"


# ============================================================================
# Caching Tests
# ============================================================================

def test_should_update_true_when_no_cache(selector):
    """Should update when no previous selection exists."""
    assert selector.should_update() is True


def test_should_update_false_when_cache_fresh(selector):
    """Should not update when cache is fresh."""
    selector._last_selection_time = datetime.utcnow()
    assert selector.should_update() is False


def test_should_update_true_when_cache_expired(selector):
    """Should update when cache has expired."""
    selector._last_selection_time = datetime.utcnow() - timedelta(minutes=20)
    selector.config.cache_minutes = 15
    assert selector.should_update() is True


def test_should_update_false_when_disabled(selector):
    """Should not update when selector is disabled."""
    selector.config.enabled = False
    selector._last_selection_time = None  # No previous selection
    assert selector.should_update() is False


def test_cache_invalidation(selector):
    """Test cache invalidation clears cached data."""
    selector._cached_selection = ProfileSelection(
        profile_name="trending",
        weights=WEIGHT_PROFILES["trending"],
        confidence=0.8,
        reasoning="Test",
        selected_at=datetime.utcnow(),
        market_context={},
    )
    selector._last_selection_time = datetime.utcnow()

    selector.invalidate_cache()

    assert selector._cached_selection is None
    assert selector._last_selection_time is None


# ============================================================================
# Fallback Selection Tests
# ============================================================================

def test_fallback_volatile_condition(selector):
    """High volatility should use volatile profile."""
    result = selector._fallback_selection(
        indicators={"rsi": 50},
        volatility="extreme",
        trend="neutral",
    )
    assert result.profile_name == "volatile"
    assert result.confidence == 0.5  # Lower confidence for fallback
    assert result.market_context.get("fallback") is True


def test_fallback_high_volatility(selector):
    """High volatility should use volatile profile."""
    result = selector._fallback_selection(
        indicators={"rsi": 50},
        volatility="high",
        trend="neutral",
    )
    assert result.profile_name == "volatile"


def test_fallback_trending_bullish(selector):
    """Bullish trend should use trending profile."""
    result = selector._fallback_selection(
        indicators={"rsi": 50},
        volatility="normal",
        trend="bullish",
    )
    assert result.profile_name == "trending"


def test_fallback_trending_bearish(selector):
    """Bearish trend should use trending profile."""
    result = selector._fallback_selection(
        indicators={"rsi": 50},
        volatility="normal",
        trend="bearish",
    )
    assert result.profile_name == "trending"


def test_fallback_ranging_low_rsi(selector):
    """Low RSI should use ranging profile (mean reversion)."""
    result = selector._fallback_selection(
        indicators={"rsi": 28},
        volatility="normal",
        trend="neutral",
    )
    assert result.profile_name == "ranging"


def test_fallback_ranging_high_rsi(selector):
    """High RSI should use ranging profile (mean reversion)."""
    result = selector._fallback_selection(
        indicators={"rsi": 72},
        volatility="normal",
        trend="neutral",
    )
    assert result.profile_name == "ranging"


def test_fallback_default_condition(selector):
    """No clear condition should use default profile."""
    result = selector._fallback_selection(
        indicators={"rsi": 50},
        volatility="normal",
        trend="neutral",
    )
    assert result.profile_name == "default"


def test_fallback_caches_selection(selector):
    """Fallback selection should be cached."""
    result = selector._fallback_selection(
        indicators={"rsi": 50},
        volatility="normal",
        trend="neutral",
    )

    assert selector._cached_selection is not None
    assert selector._cached_selection.profile_name == result.profile_name
    assert selector._last_selection_time is not None


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

def test_circuit_breaker_opens_after_failures(selector):
    """Circuit breaker should open after max failures."""
    selector._consecutive_failures = 3

    # Should use fallback when circuit is open
    result = selector._fallback_selection(
        indicators={"rsi": 50},
        volatility="normal",
        trend="neutral",
    )

    assert result.market_context.get("fallback") is True


def test_circuit_breaker_reset(selector):
    """Circuit breaker reset should clear failure count."""
    selector._consecutive_failures = 3
    selector.reset_circuit_breaker()
    assert selector._consecutive_failures == 0


# ============================================================================
# Response Parsing Tests
# ============================================================================

def test_parse_valid_json_response(selector):
    """Test parsing valid JSON response."""
    content = '{"profile": "trending", "confidence": 0.85, "reasoning": "Strong uptrend"}'
    result = selector._parse_response(content)

    assert result["profile"] == "trending"
    assert result["confidence"] == 0.85
    assert result["reasoning"] == "Strong uptrend"


def test_parse_json_in_text(selector):
    """Test parsing JSON embedded in text."""
    content = 'Here is my analysis: {"profile": "ranging", "confidence": 0.7, "reasoning": "Sideways"}'
    result = selector._parse_response(content)

    assert result["profile"] == "ranging"


def test_parse_unknown_profile_defaults(selector):
    """Unknown profile should default to 'default'."""
    content = '{"profile": "unknown_profile", "confidence": 0.5, "reasoning": "Test"}'
    result = selector._parse_response(content)

    assert result["profile"] == "default"


def test_parse_confidence_clamping_high(selector):
    """Confidence above 1.0 should be clamped."""
    content = '{"profile": "trending", "confidence": 1.5, "reasoning": "Test"}'
    result = selector._parse_response(content)

    assert result["confidence"] == 1.0


def test_parse_confidence_clamping_low(selector):
    """Negative confidence should be clamped to 0."""
    content = '{"profile": "trending", "confidence": -0.5, "reasoning": "Test"}'
    result = selector._parse_response(content)

    assert result["confidence"] == 0.0


def test_parse_missing_confidence_defaults(selector):
    """Missing confidence should default to 0.7."""
    content = '{"profile": "trending", "reasoning": "Test"}'
    result = selector._parse_response(content)

    assert result["confidence"] == 0.7


def test_parse_missing_reasoning_defaults(selector):
    """Missing reasoning should have default text."""
    content = '{"profile": "trending", "confidence": 0.8}'
    result = selector._parse_response(content)

    assert result["reasoning"] == "No reasoning provided"


def test_parse_invalid_json_raises(selector):
    """Invalid JSON should raise ValueError."""
    content = "This is not JSON at all"

    with pytest.raises(ValueError, match="No valid JSON"):
        selector._parse_response(content)


# ============================================================================
# Prompt Building Tests
# ============================================================================

def test_prompt_includes_all_data(selector):
    """Prompt should include all market data."""
    prompt = selector._build_prompt(
        indicators={"rsi": 45.5, "macd_histogram": 0.25, "bb_percent_b": 0.65},
        volatility="normal",
        trend="bullish",
        current_price=Decimal("100000.00"),
        fear_greed=55,
    )

    assert "100,000" in prompt  # Price formatted
    assert "NORMAL" in prompt  # Volatility
    assert "bullish" in prompt  # Trend
    assert "45.5" in prompt  # RSI
    assert "0.25" in prompt  # MACD histogram
    assert "0.65" in prompt  # BB %B
    assert "55" in prompt  # Fear & Greed


def test_prompt_handles_missing_fear_greed(selector):
    """Prompt should handle missing Fear & Greed."""
    prompt = selector._build_prompt(
        indicators={"rsi": 50},
        volatility="normal",
        trend="neutral",
        current_price=Decimal("50000.00"),
        fear_greed=None,
    )

    assert "N/A" in prompt


def test_prompt_handles_missing_indicators(selector):
    """Prompt should handle missing indicator values."""
    prompt = selector._build_prompt(
        indicators={},
        volatility="normal",
        trend="neutral",
        current_price=Decimal("50000.00"),
        fear_greed=None,
    )

    assert "N/A" in prompt


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_select_profile_returns_cached(selector):
    """Should return cached selection when still valid."""
    cached = ProfileSelection(
        profile_name="trending",
        weights=WEIGHT_PROFILES["trending"],
        confidence=0.8,
        reasoning="Cached",
        selected_at=datetime.utcnow(),
        market_context={},
    )
    selector._cached_selection = cached
    selector._last_selection_time = datetime.utcnow()

    result = await selector.select_profile(
        indicators={"rsi": 50},
        volatility="normal",
        trend="neutral",
        current_price=Decimal("100000"),
    )

    assert result == cached


@pytest.mark.asyncio
async def test_select_profile_uses_fallback_on_circuit_open(selector):
    """Should use fallback when circuit breaker is open."""
    selector._consecutive_failures = 3

    result = await selector.select_profile(
        indicators={"rsi": 50},
        volatility="high",
        trend="neutral",
        current_price=Decimal("100000"),
    )

    assert result.market_context.get("fallback") is True
    assert result.profile_name == "volatile"  # High volatility fallback
