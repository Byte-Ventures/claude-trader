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
    """Claude AI veto actions."""
    SKIP = "skip"      # Skip trade entirely
    REDUCE = "reduce"  # Reduce position size
    DELAY = "delay"    # Delay trade (user checks Telegram)
    INFO = "info"      # Log but proceed with trade


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

    # Risk Management
    stop_loss_atr_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Stop loss distance as ATR multiple"
    )
    min_stop_loss_percent: float = Field(
        default=1.5,
        ge=0.1,
        le=10.0,
        description="Minimum stop loss distance as percentage below entry (safety floor for short timeframes)"
    )
    take_profit_atr_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Take profit distance as ATR multiple"
    )
    min_take_profit_percent: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Minimum take profit distance as percentage above entry (safety floor)"
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
    max_position_percent: float = Field(
        default=80.0,
        ge=10.0,
        le=100.0,
        description="Maximum position size as percentage of portfolio"
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
    veto_action: VetoAction = Field(
        default=VetoAction.INFO,
        description="Action on veto: skip, reduce, delay, info"
    )
    veto_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Confidence threshold to trigger veto"
    )
    position_reduction: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Position size multiplier for 'reduce' veto action"
    )
    delay_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Minutes to delay for 'delay' veto action"
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
        description="Behavior when AI review fails: open (proceed with trade) or safe (skip trade)"
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

    @model_validator(mode="before")
    @classmethod
    def migrate_deprecated_claude_vars(cls, data: dict) -> dict:
        """
        Backward compatibility: support old CLAUDE_* environment variable names.

        Maps deprecated names to new names with a deprecation warning.
        """
        # Mapping of old CLAUDE_* vars to new names
        deprecated_mapping = {
            "CLAUDE_VETO_ACTION": "VETO_ACTION",
            "CLAUDE_VETO_THRESHOLD": "VETO_THRESHOLD",
            "CLAUDE_POSITION_REDUCTION": "POSITION_REDUCTION",
            "CLAUDE_DELAY_MINUTES": "DELAY_MINUTES",
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
