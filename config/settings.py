"""
Configuration settings with Pydantic validation.
All settings are loaded from environment variables.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

import os
import warnings

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Trading mode selection."""
    PAPER = "paper"
    LIVE = "live"


class Exchange(str, Enum):
    """Supported exchanges."""
    COINBASE = "coinbase"
    KRAKEN = "kraken"


class VetoAction(str, Enum):
    """Claude AI veto actions.

    Note: Since v1.31.0, the tiered veto system automatically selects
    SKIP or REDUCE based on judge confidence level. DELAY and INFO
    are deprecated and no longer used by the tiered system.
    """
    SKIP = "skip"      # Skip trade entirely (used when confidence >= veto_skip_threshold)
    REDUCE = "reduce"  # Reduce position size (used when confidence >= veto_reduce_threshold)
    DELAY = "delay"    # DEPRECATED in v1.31.0 - not used by tiered veto system
    INFO = "info"      # DEPRECATED in v1.31.0 - confidence below reduce_threshold proceeds automatically


class AIFailureMode(str, Enum):
    """Behavior when AI trade review fails or times out."""
    OPEN = "open"    # Proceed with trade (current behavior, fail-open)
    SAFE = "safe"    # Skip trade if AI unreachable (fail-safe)


class Settings(BaseSettings):
    """Main application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Exchange Selection
    exchange: Exchange = Field(
        default=Exchange.COINBASE,
        description="Exchange to use: coinbase or kraken"
    )

    # Coinbase API - can use either key file OR key+secret
    coinbase_key_file: Optional[Path] = Field(
        default=None,
        description="Path to CDP API key JSON file"
    )
    coinbase_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Coinbase API key (alternative to key_file)"
    )
    coinbase_api_secret: Optional[SecretStr] = Field(
        default=None,
        description="Coinbase API secret (alternative to key_file)"
    )

    # Kraken API
    kraken_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Kraken API key"
    )
    kraken_api_secret: Optional[SecretStr] = Field(
        default=None,
        description="Kraken API secret (base64-encoded)"
    )

    # Trading Mode
    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Trading mode: paper (simulated) or live (real money)"
    )

    # Live Trading Confirmation (must be True to run in live mode)
    i_understand_that_i_will_lose_all_my_money: bool = Field(
        default=False,
        description="Must be set to True to acknowledge the risks of live trading"
    )

    # Paper Trading Initial Balances (default: ~50/50 split)
    paper_initial_quote: float = Field(
        default=5000.0,
        ge=0.0,
        description="Initial quote currency balance for paper trading (e.g., USD, EUR)"
    )
    paper_initial_base: float = Field(
        default=0.05,
        ge=0.0,
        description="Initial base currency balance for paper trading (e.g., BTC)"
    )

    # Trading Parameters
    trading_pair: str = Field(
        default="BTC-USD",
        description="Trading pair symbol"
    )
    position_size_percent: float = Field(
        default=40.0,
        ge=1.0,
        le=100.0,
        description="Percentage of portfolio to use for positions"
    )
    check_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Seconds between trading checks (fallback when adaptive disabled)"
    )

    # Candle Settings for Technical Analysis
    candle_interval: str = Field(
        default="ONE_HOUR",
        pattern="^(ONE_MINUTE|FIVE_MINUTE|FIFTEEN_MINUTE|THIRTY_MINUTE|ONE_HOUR|TWO_HOUR|SIX_HOUR|ONE_DAY)$",
        description="Candlestick granularity for technical analysis"
    )
    candle_limit: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Number of candles to fetch for indicator calculation"
    )

    # Adaptive Interval Settings
    adaptive_interval_enabled: bool = Field(
        default=True,
        description="Enable adaptive check intervals based on market volatility"
    )
    interval_low_volatility: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Check interval during low volatility (seconds)"
    )
    interval_normal: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Check interval during normal volatility (seconds)"
    )
    interval_high_volatility: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Check interval during high volatility (seconds)"
    )
    interval_extreme_volatility: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Check interval during extreme volatility (seconds)"
    )

    # Strategy Parameters - RSI
    rsi_period: int = Field(default=14, ge=2, le=50)
    rsi_oversold: float = Field(default=35.0, ge=10.0, le=50.0)
    rsi_overbought: float = Field(default=65.0, ge=50.0, le=90.0)

    # Strategy Parameters - EMA
    ema_fast: int = Field(default=9, ge=2, le=50)
    ema_slow: int = Field(default=21, ge=5, le=200)

    # Strategy Parameters - MACD
    macd_fast: int = Field(default=12, ge=2, le=50)
    macd_slow: int = Field(default=26, ge=5, le=100)
    macd_signal: int = Field(default=9, ge=2, le=50)

    # Strategy Parameters - Bollinger Bands
    bollinger_period: int = Field(default=20, ge=5, le=100)
    bollinger_std: float = Field(default=2.0, ge=1.0, le=4.0)

    # Strategy Parameters - ATR
    atr_period: int = Field(default=14, ge=2, le=50)

    # Strategy Parameters - Signal
    signal_threshold: int = Field(
        default=60,
        ge=40,
        le=100,
        description="Minimum score to trigger a trade (out of 100)"
    )

    # Crash Protection Parameters
    max_oversold_buys_24h: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum allowed buy trades while RSI is oversold (<30) within 24 hours (prevents averaging into crashes)"
    )
    price_stabilization_window: int = Field(
        default=12,
        ge=5,
        le=50,
        description="Number of candles to wait for price stabilization after extreme RSI (25/75) before allowing trades"
    )
    extreme_rsi_lower: int = Field(
        default=25,
        ge=10,
        le=30,
        description="Extreme lower RSI threshold for crash protection"
    )
    extreme_rsi_upper: int = Field(
        default=75,
        ge=70,
        le=90,
        description="Extreme upper RSI threshold for crash protection"
    )

    # Strategy Parameters - Volume/Whale Detection
    whale_volume_threshold: float = Field(
        default=3.0,
        ge=1.5,
        le=10.0,
        description="Volume ratio threshold for whale activity detection (e.g., 3.0 = 3x average volume)"
    )
    whale_direction_threshold: float = Field(
        default=0.003,
        ge=0.0005,
        le=0.01,
        description="Price change threshold for whale direction classification (0.003 = 0.3%)"
    )
    whale_boost_percent: float = Field(
        default=0.30,
        ge=0.1,
        le=0.5,
        description="Signal boost multiplier for whale activity (0.30 = 30%)"
    )
    high_volume_boost_percent: float = Field(
        default=0.20,
        ge=0.1,
        le=0.4,
        description="Signal boost multiplier for high volume (0.20 = 20%)"
    )

    # Volume Analysis Parameters
    volume_sma_window: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Window size for volume moving average calculation (candles)"
    )
    high_volume_threshold: float = Field(
        default=1.5,
        ge=1.1,
        le=3.0,
        description="Volume ratio threshold for high volume detection (1.5 = 150% of average)"
    )
    low_volume_threshold: float = Field(
        default=0.7,
        ge=0.3,
        le=0.9,
        description="Volume ratio threshold for low volume detection (0.7 = 70% of average)"
    )
    low_volume_penalty: int = Field(
        default=10,
        ge=0,
        le=30,
        description="Score penalty points for low volume trades"
    )
    trend_filter_penalty: int = Field(
        default=20,
        ge=10,
        le=40,
        description="Score penalty points for counter-trend trades"
    )

    # Risk Management
    risk_per_trade_percent: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Risk per trade as percentage of portfolio (used in position sizing calculations)"
    )
    stop_loss_atr_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Stop loss distance as ATR multiple"
    )
    stop_loss_atr_multiplier_extreme: float = Field(
        default=2.0,
        ge=1.5,
        le=3.0,
        description="Stop loss ATR multiplier during extreme volatility conditions"
    )
    min_stop_loss_percent: float = Field(
        default=2.5,
        ge=0.1,
        le=10.0,
        description="Minimum stop loss distance as percentage below entry (prevents whipsaw exits during normal market volatility)"
    )
    take_profit_atr_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Take profit distance as ATR multiple"
    )
    enable_take_profit: bool = Field(
        default=True,
        description="Enable automatic take profit exits at target price"
    )
    trailing_stop_atr_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Trailing stop distance as ATR multiple"
    )
    breakeven_atr_multiplier: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Move stop to break-even when profit reaches this ATR multiple"
    )
    use_limit_orders: bool = Field(
        default=True,
        description="Use limit IOC orders instead of market orders to reduce slippage"
    )
    limit_order_offset_percent: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Offset from bid/ask for limit orders (0.1 = 0.1%)"
    )
    max_daily_loss_percent: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Maximum daily loss before halting (percentage)"
    )
    max_hourly_loss_percent: float = Field(
        default=3.0,
        ge=0.5,
        le=20.0,
        description="Maximum hourly loss before pausing (percentage)"
    )
    loss_throttle_start_percent: float = Field(
        default=50.0,
        ge=10.0,
        le=90.0,
        description="Loss percentage at which position size throttling begins (% of max loss limit)"
    )
    loss_throttle_min_multiplier: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Minimum position size multiplier when at max loss (0.3 = reduce to 30%)"
    )
    max_position_percent: float = Field(
        default=80.0,
        ge=10.0,
        le=100.0,
        description="Maximum position size as percentage of portfolio"
    )
    estimated_fee_percent: float = Field(
        default=0.006,
        ge=0.001,
        le=0.02,
        description="Estimated round-trip trading fee as decimal (0.006 = 0.6%, typical for Coinbase)"
    )
    profit_margin_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Minimum profit margin as multiple of fees for expected value (EV) validation"
    )

    # Order Size Limits (absolute limits in quote currency)
    min_trade_quote: float = Field(
        default=10.0,
        ge=1.0,
        le=1000.0,
        description="Minimum order size in quote currency (e.g., EUR/USD). Orders below this are skipped."
    )
    min_trade_base: float = Field(
        default=0.0001,
        ge=0.00001,
        le=0.01,
        description="Minimum order size in base currency (e.g., BTC). Exchange-specific minimum."
    )
    max_trade_quote: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=100000.0,
        description="Maximum order size in quote currency. None = no limit (use position_size_percent only)."
    )

    # Trade Cooldown - prevents rapid consecutive trades
    trade_cooldown_enabled: bool = Field(
        default=True,
        description="Enable cooldown between same-direction trades"
    )
    buy_cooldown_minutes: int = Field(
        default=15,
        ge=0,
        le=60,
        description="Minimum minutes between buy trades (0 = disabled)"
    )
    sell_cooldown_minutes: int = Field(
        default=0,
        ge=0,
        le=60,
        description="Minimum minutes between sell trades (0 = disabled for safety)"
    )
    buy_price_change_percent: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Price must drop this % from last buy to buy again (0 = disabled)"
    )
    sell_price_change_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Price must rise this % from last sell to sell again (0 = disabled)"
    )

    # Circuit Breaker
    black_recovery_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=168,
        description="Hours before BLACK state auto-downgrades to RED (None=manual only)"
    )

    # Telegram
    telegram_bot_token: Optional[SecretStr] = Field(
        default=None,
        description="Telegram bot token from @BotFather"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None,
        description="Telegram chat ID for notifications"
    )
    telegram_enabled: bool = Field(
        default=True,
        description="Enable Telegram notifications"
    )

    # Multi-Agent Trade Review (via OpenRouter)
    openrouter_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenRouter API key for AI trade review"
    )
    ai_review_enabled: bool = Field(
        default=False,
        description="Enable multi-agent AI trade review via OpenRouter"
    )
    reviewer_model_1: str = Field(
        default="x-ai/grok-4-fast",
        description="First reviewer model (stance randomly assigned)"
    )
    reviewer_model_2: str = Field(
        default="qwen/qwen3-next-80b-a3b-instruct",
        description="Second reviewer model (stance randomly assigned)"
    )
    reviewer_model_3: str = Field(
        default="google/gemini-2.5-flash",
        description="Third reviewer model (stance randomly assigned)"
    )
    judge_model: str = Field(
        default="deepseek/deepseek-chat-v3.1",
        description="Judge model for final decision synthesis"
    )
    # Tiered veto system: action depends on judge's confidence level
    # When judge disapproves: <65% proceed, 65-79% reduce, >=80% skip
    veto_reduce_threshold: float = Field(
        default=0.65,
        ge=0.5,
        le=1.0,
        description="Judge confidence threshold to reduce position (lower tier)"
    )
    veto_skip_threshold: float = Field(
        default=0.80,
        ge=0.5,
        le=1.0,
        description="Judge confidence threshold to skip trade entirely (higher tier)"
    )
    position_reduction: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Position size multiplier when veto_reduce_threshold is met"
    )
    interesting_hold_margin: int = Field(
        default=15,
        ge=5,
        le=30,
        description="Score margin from threshold for 'interesting hold' analysis"
    )
    ai_recommendation_ttl_minutes: int = Field(
        default=20,
        ge=5,
        le=60,
        description="How long AI 'accumulate'/'reduce' recommendations influence thresholds (minutes)"
    )
    ai_review_all: bool = Field(
        default=False,
        description="Review ALL decisions with AI (for debugging/testing)"
    )
    ai_failure_mode: AIFailureMode = Field(
        default=AIFailureMode.OPEN,
        description="DEPRECATED: Use ai_failure_mode_buy/sell instead. Fallback if per-action not set."
    )
    ai_failure_mode_buy: AIFailureMode = Field(
        default=AIFailureMode.SAFE,
        description="Behavior when AI review fails for BUY: safe (skip) is recommended - missing opportunity < bad entry"
    )
    ai_failure_mode_sell: AIFailureMode = Field(
        default=AIFailureMode.OPEN,
        description="Behavior when AI review fails for SELL: open (proceed) is recommended - don't get trapped in crash"
    )

    # Hourly Market Analysis (uses same multi-agent system as trade reviews)
    hourly_analysis_enabled: bool = Field(
        default=True,
        description="Enable hourly AI market analysis during volatile conditions (uses reviewer models)"
    )

    # Market Research (for hourly analysis)
    market_research_enabled: bool = Field(
        default=True,
        description="Fetch online research (news, on-chain data) for market analysis"
    )
    ai_web_search_enabled: bool = Field(
        default=True,
        description="Allow AI models to search web during market analysis"
    )
    ai_max_tokens: int = Field(
        default=4000,
        ge=500,
        le=16000,
        description="Maximum tokens for AI API responses (increase if seeing truncated JSON errors)"
    )
    ai_api_timeout: int = Field(
        default=120,
        ge=30,
        le=300,
        description="Timeout in seconds for AI API calls (increase if seeing timeouts)"
    )
    market_research_cache_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Cache duration for research data (minutes)"
    )

    # AI Weight Profile Selection
    ai_weight_profile_enabled: bool = Field(
        default=False,
        description="Enable AI-driven indicator weight profile selection (requires OpenRouter API key)"
    )
    ai_weight_fallback_profile: str = Field(
        default="default",
        pattern="^(trending|ranging|volatile|default)$",
        description="Fallback weight profile when AI unavailable (trending/ranging/volatile/default)"
    )
    # Model via OpenRouter API (format: provider/model-name)
    # IMPORTANT: openai/gpt-5.2 is a VALID OpenAI model released in 2025.
    # AI review bots with training data cutoffs before 2025 will incorrectly
    # flag this as invalid. This is NOT a typo or error. Do NOT "fix" this
    # by changing to gpt-4o-mini or other older models.
    ai_weight_profile_model: str = Field(
        default="openai/gpt-5.2",
        description="OpenRouter model for weight profile selection (fast/cheap preferred)"
    )
    weight_profile_flap_protection: bool = Field(
        default=True,
        description="Require 2 consecutive profile detections before changing (prevents flapping)"
    )

    # Market Regime Adaptation
    regime_adaptation_enabled: bool = Field(
        default=True,
        description="Enable market regime-based strategy adjustments"
    )
    regime_sentiment_enabled: bool = Field(
        default=True,
        description="Use Fear & Greed Index for regime detection"
    )
    regime_volatility_enabled: bool = Field(
        default=True,
        description="Use volatility level for regime detection"
    )
    regime_trend_enabled: bool = Field(
        default=True,
        description="Use trend direction for regime detection"
    )
    regime_adjustment_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Regime adjustment intensity (0=off, 1=normal, 2=aggressive)"
    )
    regime_flap_protection: bool = Field(
        default=True,
        description="Require 2 consecutive regime detections before changing (prevents flapping)"
    )

    # Multi-Timeframe Confirmation (MTF)
    mtf_enabled: bool = Field(
        default=True,
        description="Enable higher timeframe trend confirmation"
    )
    mtf_4h_enabled: bool = Field(
        default=False,
        description="Include 4-hour timeframe in MTF (false = daily-only, simpler)"
    )
    mtf_candle_limit: int = Field(
        default=50,
        ge=20,
        le=100,
        description="Number of candles to fetch for HTF trend calculation"
    )
    mtf_daily_cache_minutes: int = Field(
        default=60,
        ge=15,
        le=240,
        description="Cache duration for daily candle data (minutes)"
    )
    mtf_4h_cache_minutes: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Cache duration for 4-hour candle data (minutes)"
    )
    mtf_aligned_boost: int = Field(
        default=20,
        ge=5,
        le=40,
        description="Score boost for trades aligned with HTF trend"
    )
    mtf_counter_penalty: int = Field(
        default=20,
        ge=5,
        le=40,
        description="Score penalty for trades against HTF trend"
    )

    # Database
    database_path: Path = Field(
        default=Path("data/trading.db"),
        description="Path to SQLite database"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Path = Field(
        default=Path("logs/trading.log"),
        description="Path to log file"
    )

    # Dashboard
    dashboard_host: str = Field(
        default="127.0.0.1",
        description="Dashboard web server bind address (use 0.0.0.0 for network access)"
    )
    dashboard_port: int = Field(
        default=8081,
        ge=1024,
        le=65535,
        description="Dashboard web server port"
    )

    # Post-Mortem Analysis (automated trade analysis using Claude CLI)
    postmortem_enabled: bool = Field(
        default=False,
        description="Enable automatic post-mortem analysis after trades (requires Claude CLI)"
    )
    postmortem_create_discussion: bool = Field(
        default=False,
        description="Create GitHub Discussion with analysis (requires gh CLI and Post-Mortems category)"
    )

    # Dual-Extreme Conditions Protection
    block_trades_extreme_conditions: bool = Field(
        default=True,
        description="Block new positions when both sentiment and volatility are extreme"
    )

    # Sentiment-Trend Modifiers for Market Regime Adaptation
    sentiment_trend_modifiers: Optional[dict] = Field(
        default=None,
        description="Custom sentiment-trend interaction modifiers. If None, uses hardcoded defaults in regime.py. Format: JSON object with keys like 'extreme_fear_bearish_buy' containing threshold_mult and position_mult values."
    )

    # Cramer Mode Mode (paper trading only)
    enable_cramer_mode: bool = Field(
        default=False,
        description="Enable Cramer Mode: execute opposite trade alongside each normal trade for comparison (paper mode only)"
    )

    @field_validator("ema_slow")
    @classmethod
    def validate_ema_slow(cls, v: int, info) -> int:
        """Ensure slow EMA is greater than fast EMA."""
        if "ema_fast" in info.data and v <= info.data["ema_fast"]:
            raise ValueError("ema_slow must be greater than ema_fast")
        return v

    @field_validator("macd_slow")
    @classmethod
    def validate_macd_slow(cls, v: int, info) -> int:
        """Ensure slow MACD is greater than fast MACD."""
        if "macd_fast" in info.data and v <= info.data["macd_fast"]:
            raise ValueError("macd_slow must be greater than macd_fast")
        return v

    @field_validator("rsi_overbought")
    @classmethod
    def validate_rsi_overbought(cls, v: float, info) -> float:
        """Ensure overbought is greater than oversold."""
        if "rsi_oversold" in info.data and v <= info.data["rsi_oversold"]:
            raise ValueError("rsi_overbought must be greater than rsi_oversold")
        return v

    @model_validator(mode="after")
    def validate_telegram_config(self) -> "Settings":
        """Validate Telegram configuration if enabled."""
        if self.telegram_enabled:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                # Disable Telegram if not configured
                self.telegram_enabled = False
        return self

    @model_validator(mode="after")
    def validate_ai_review_config(self) -> "Settings":
        """Validate AI review configuration if enabled."""
        if self.ai_review_enabled:
            if not self.openrouter_api_key:
                raise ValueError(
                    "AI_REVIEW_ENABLED is true but OPENROUTER_API_KEY is not set. "
                    "Either set OPENROUTER_API_KEY or disable AI_REVIEW_ENABLED."
                )

            # Test API key with minimal request
            import httpx
            try:
                api_key = self.openrouter_api_key.get_secret_value()
                response = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/Byte-Ventures/claude-trader",
                        "X-Title": "Claude Trader",
                    },
                    json={
                        "model": "openai/gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                    },
                    timeout=10.0,
                )

                if response.status_code == 401:
                    raise ValueError(
                        "OPENROUTER_API_KEY is invalid. Check your API key at https://openrouter.ai/keys"
                    )
                elif response.status_code == 403:
                    raise ValueError(
                        "OPENROUTER_API_KEY is valid but lacks required permissions. "
                        "Check your API key settings at https://openrouter.ai/keys"
                    )
                elif response.status_code >= 400:
                    raise ValueError(
                        f"OpenRouter API test failed with status {response.status_code}. "
                        f"Response: {response.text[:200]}"
                    )

            except httpx.TimeoutException:
                raise ValueError(
                    "OpenRouter API test timed out. Check your internet connection."
                )
            except httpx.RequestError as e:
                raise ValueError(
                    f"OpenRouter API test failed: {str(e)}. Check your internet connection."
                )

        return self

    @model_validator(mode="after")
    def validate_stop_loss_config(self) -> "Settings":
        """
        Validate stop-loss and trailing stop configuration.

        Trailing stop activates at 1 ATR profit from avg_cost.
        Hard stop is at stop_loss_atr_multiplier below avg_cost.

        If stop_loss_atr_multiplier < 1, the hard stop distance is smaller
        than the trailing activation distance, meaning the position is more
        likely to hit hard stop before trailing can activate (less profit potential).
        """
        import warnings

        if self.stop_loss_atr_multiplier < 1.0:
            warnings.warn(
                f"stop_loss_atr_multiplier ({self.stop_loss_atr_multiplier}) is less than "
                f"trailing activation (1.0 ATR). Hard stop may trigger before trailing "
                f"stop can activate, reducing profit potential. Consider increasing to >= 1.0.",
                UserWarning,
            )

        # Warn if stop loss is very tight relative to trailing distance
        if self.stop_loss_atr_multiplier < self.trailing_stop_atr_multiplier:
            warnings.warn(
                f"stop_loss_atr_multiplier ({self.stop_loss_atr_multiplier}) is less than "
                f"trailing_stop_atr_multiplier ({self.trailing_stop_atr_multiplier}). "
                f"This creates tight stops with loose trails - ensure this is intentional.",
                UserWarning,
            )

        return self

    @model_validator(mode="after")
    def validate_veto_thresholds(self) -> "Settings":
        """Validate veto threshold ordering and minimum gap."""
        if self.veto_reduce_threshold >= self.veto_skip_threshold:
            raise ValueError(
                f"veto_reduce_threshold ({self.veto_reduce_threshold}) must be less than "
                f"veto_skip_threshold ({self.veto_skip_threshold})"
            )

        # Require at least 5% gap between thresholds to ensure meaningful "reduce" tier
        gap = self.veto_skip_threshold - self.veto_reduce_threshold
        if gap < 0.05:
            raise ValueError(
                f"veto thresholds must have at least 0.05 gap between them "
                f"(current gap: {gap:.3f})"
            )
        return self

    @model_validator(mode="after")
    def validate_trade_size_limits(self) -> "Settings":
        """Validate order size limits are properly ordered."""
        if self.max_trade_quote is not None:
            if self.max_trade_quote < self.min_trade_quote:
                raise ValueError(
                    f"max_trade_quote ({self.max_trade_quote}) must be >= "
                    f"min_trade_quote ({self.min_trade_quote})"
                )
        return self

    @model_validator(mode="after")
    def validate_extreme_rsi_thresholds(self) -> "Settings":
        """Validate extreme RSI thresholds are properly ordered with minimum gap."""
        if self.extreme_rsi_lower >= self.extreme_rsi_upper:
            raise ValueError(
                f"extreme_rsi_lower ({self.extreme_rsi_lower}) must be < "
                f"extreme_rsi_upper ({self.extreme_rsi_upper})"
            )

        # Require at least 40-point gap for meaningful crash/pump detection
        gap = self.extreme_rsi_upper - self.extreme_rsi_lower
        if gap < 40:
            raise ValueError(
                f"extreme_rsi thresholds must have at least 40-point gap "
                f"(current gap: {gap}). Narrow gaps create false crash/pump signals."
            )
        return self

    @field_validator("sentiment_trend_modifiers")
    @classmethod
    def validate_sentiment_trend_modifiers(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate sentiment-trend modifiers configuration."""
        if v is None:
            return None

        # Expected keys (24 combinations)
        sentiments = ["extreme_fear", "fear", "greed", "extreme_greed"]
        trends = ["bullish", "bearish", "neutral"]
        signals = ["buy", "sell"]

        expected_keys = [
            f"{sentiment}_{trend}_{signal}"
            for sentiment in sentiments
            for trend in trends
            for signal in signals
        ]

        # Validate all required keys present
        missing_keys = set(expected_keys) - set(v.keys())
        if missing_keys:
            raise ValueError(
                f"sentiment_trend_modifiers missing required keys: {sorted(missing_keys)}. "
                f"All 24 combinations must be present."
            )

        # Validate each entry has correct structure and valid ranges
        for key, modifiers in v.items():
            if key not in expected_keys:
                raise ValueError(
                    f"sentiment_trend_modifiers has invalid key: {key}. "
                    f"Valid keys are combinations like 'extreme_fear_bearish_buy'."
                )

            if not isinstance(modifiers, dict):
                raise ValueError(
                    f"sentiment_trend_modifiers[{key}] must be a dict, got {type(modifiers)}"
                )

            if "threshold_mult" not in modifiers or "position_mult" not in modifiers:
                raise ValueError(
                    f"sentiment_trend_modifiers[{key}] must have 'threshold_mult' and 'position_mult' keys"
                )

            threshold_mult = modifiers["threshold_mult"]
            position_mult = modifiers["position_mult"]

            # Validate ranges
            if not (0.0 <= threshold_mult <= 2.0):
                raise ValueError(
                    f"sentiment_trend_modifiers[{key}].threshold_mult must be 0.0-2.0, got {threshold_mult}"
                )

            if not (0.5 <= position_mult <= 1.5):
                raise ValueError(
                    f"sentiment_trend_modifiers[{key}].position_mult must be 0.5-1.5, got {position_mult}"
                )

        return v

    @model_validator(mode="before")
    @classmethod
    def migrate_deprecated_claude_vars(cls, data: dict) -> dict:
        """
        Backward compatibility: support old CLAUDE_* environment variable names.

        Maps deprecated names to new names with a deprecation warning.
        Also migrates old VETO_ACTION/VETO_THRESHOLD to tiered thresholds (v1.31.0).
        """
        # Mapping of old CLAUDE_* vars to new names
        deprecated_mapping = {
            "CLAUDE_POSITION_REDUCTION": "POSITION_REDUCTION",
            "CLAUDE_INTERESTING_HOLD_MARGIN": "INTERESTING_HOLD_MARGIN",
        }

        for old_name, new_name in deprecated_mapping.items():
            old_value = os.environ.get(old_name)
            new_value = os.environ.get(new_name)

            # If old var is set but new var is not, use old value and warn
            if old_value is not None and new_value is None:
                warnings.warn(
                    f"Environment variable {old_name} is deprecated. Use {new_name} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Set the new key in the data dict
                key = new_name.lower()
                if key not in data or data.get(key) is None:
                    data[key] = old_value

        # v1.31.0: Migrate old VETO_ACTION/VETO_THRESHOLD to tiered thresholds
        old_action = os.environ.get("VETO_ACTION", "").lower()
        old_threshold = os.environ.get("VETO_THRESHOLD")

        if old_action or old_threshold:
            # Check if new tiered thresholds are already set
            has_new_reduce = os.environ.get("VETO_REDUCE_THRESHOLD") or data.get("veto_reduce_threshold")
            has_new_skip = os.environ.get("VETO_SKIP_THRESHOLD") or data.get("veto_skip_threshold")

            if not has_new_reduce and not has_new_skip:
                # Migrate based on old action type
                if old_action == "skip" and old_threshold:
                    data["veto_skip_threshold"] = float(old_threshold)
                    warnings.warn(
                        f"VETO_ACTION=skip with VETO_THRESHOLD={old_threshold} is deprecated. "
                        f"Migrated to VETO_SKIP_THRESHOLD={old_threshold}. "
                        "Update your .env to use VETO_REDUCE_THRESHOLD and VETO_SKIP_THRESHOLD.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                elif old_action == "reduce" and old_threshold:
                    threshold_val = float(old_threshold)
                    data["veto_reduce_threshold"] = threshold_val
                    # Ensure skip threshold maintains 5% gap (required by validation)
                    default_skip = 0.80
                    if threshold_val > default_skip - 0.05:
                        data["veto_skip_threshold"] = min(1.0, threshold_val + 0.10)
                    warnings.warn(
                        f"VETO_ACTION=reduce with VETO_THRESHOLD={old_threshold} is deprecated. "
                        f"Migrated to VETO_REDUCE_THRESHOLD={old_threshold}. "
                        "Update your .env to use VETO_REDUCE_THRESHOLD and VETO_SKIP_THRESHOLD.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                elif old_action in ["delay", "info"]:
                    warnings.warn(
                        f"VETO_ACTION={old_action} is no longer supported in v1.31.0. "
                        "The tiered veto system now uses VETO_REDUCE_THRESHOLD (default 0.65) "
                        "and VETO_SKIP_THRESHOLD (default 0.80). Update your .env file.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

        return data

    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.trading_mode == TradingMode.PAPER

    @property
    def is_live_trading(self) -> bool:
        """Check if running in live trading mode."""
        return self.trading_mode == TradingMode.LIVE

    @property
    def is_coinbase(self) -> bool:
        """Check if using Coinbase exchange."""
        return self.exchange == Exchange.COINBASE

    @property
    def is_kraken(self) -> bool:
        """Check if using Kraken exchange."""
        return self.exchange == Exchange.KRAKEN


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None
_reload_requested: bool = False


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def request_reload() -> None:
    """Request a settings reload (called from signal handler)."""
    global _reload_requested
    _reload_requested = True


def reload_pending() -> bool:
    """Check if a reload has been requested."""
    return _reload_requested


def reload_settings() -> tuple[Settings, dict[str, tuple]]:
    """
    Reload settings from .env file.

    Returns:
        Tuple of (new_settings, changes_dict)
        changes_dict maps field_name -> (old_value, new_value)
    """
    global _settings, _reload_requested
    _reload_requested = False

    old_settings = _settings

    # Force re-read from .env by creating new instance
    new_settings = Settings()

    # Calculate what changed (exclude secrets for logging)
    changes: dict[str, tuple] = {}
    if old_settings:
        for field_name in Settings.model_fields:
            old_val = getattr(old_settings, field_name)
            new_val = getattr(new_settings, field_name)
            # Skip SecretStr fields for security
            if hasattr(old_val, "get_secret_value"):
                continue
            if old_val != new_val:
                changes[field_name] = (old_val, new_val)

    _settings = new_settings
    return new_settings, changes
