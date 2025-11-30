"""
Configuration settings with Pydantic validation.
All settings are loaded from environment variables.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

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
        description="Seconds between trading checks"
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

    # Risk Management
    stop_loss_atr_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Stop loss distance as ATR multiple"
    )
    take_profit_atr_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Take profit distance as ATR multiple"
    )
    trailing_stop_atr_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Trailing stop distance as ATR multiple"
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
