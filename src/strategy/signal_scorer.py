"""
Multi-indicator confluence signal scorer.

Combines signals from multiple technical indicators to generate
a composite trading signal score from -100 (strong sell) to +100 (strong buy).

Trade execution requires score magnitude >= threshold (default: 60).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
import structlog

from src.indicators.rsi import calculate_rsi, get_rsi_signal_graduated
from src.indicators.macd import calculate_macd, get_macd_signal_graduated
from src.indicators.bollinger import calculate_bollinger_bands, get_bollinger_signal_graduated
from src.indicators.ema import calculate_ema_crossover, get_ema_signal_graduated, get_ema_trend
from src.indicators.atr import calculate_atr, get_volatility_level

logger = structlog.get_logger(__name__)


@dataclass
class SignalWeights:
    """Weights for each indicator in the composite score."""

    rsi: int = 25
    macd: int = 25
    bollinger: int = 20
    ema: int = 15
    volume: int = 15


@dataclass
class IndicatorValues:
    """Current values of all indicators."""

    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    atr: Optional[float] = None
    volatility: str = "normal"


@dataclass
class SignalResult:
    """Result of signal calculation."""

    score: int  # -100 to +100
    action: str  # "buy", "sell", or "hold"
    indicators: IndicatorValues
    breakdown: dict[str, int]  # Score contribution by indicator
    confidence: float  # 0.0 to 1.0


class SignalScorer:
    """
    Confluence-based signal scoring system.

    Combines multiple indicators to generate a trading signal:
    - RSI: Momentum and overbought/oversold
    - MACD: Trend direction and momentum
    - Bollinger Bands: Volatility and mean reversion
    - EMA Crossover: Short-term trend
    - Volume: Confirmation of moves

    Each indicator contributes a weighted score. Trades execute when
    the total score exceeds the threshold.
    """

    def __init__(
        self,
        weights: Optional[SignalWeights] = None,
        threshold: int = 60,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        ema_fast: int = 9,
        ema_slow: int = 21,
        atr_period: int = 14,
    ):
        """
        Initialize signal scorer.

        Args:
            weights: Weights for each indicator
            threshold: Minimum score to trigger trade
            *: Indicator parameters
        """
        self.weights = weights or SignalWeights()
        self.threshold = threshold

        # Crash protection: track oversold buys
        self._oversold_buy_times: list[datetime] = []

        # Indicator parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.ema_fast = ema_fast
        self.ema_slow_period = ema_slow
        self.atr_period = atr_period

    def update_settings(
        self,
        threshold: Optional[int] = None,
        rsi_period: Optional[int] = None,
        rsi_oversold: Optional[float] = None,
        rsi_overbought: Optional[float] = None,
        macd_fast: Optional[int] = None,
        macd_slow: Optional[int] = None,
        macd_signal: Optional[int] = None,
        bollinger_period: Optional[int] = None,
        bollinger_std: Optional[float] = None,
        ema_fast: Optional[int] = None,
        ema_slow: Optional[int] = None,
        atr_period: Optional[int] = None,
    ) -> None:
        """
        Update scorer settings at runtime.

        Only updates parameters that are explicitly provided (not None).
        """
        if threshold is not None:
            self.threshold = threshold
        if rsi_period is not None:
            self.rsi_period = rsi_period
        if rsi_oversold is not None:
            self.rsi_oversold = rsi_oversold
        if rsi_overbought is not None:
            self.rsi_overbought = rsi_overbought
        if macd_fast is not None:
            self.macd_fast = macd_fast
        if macd_slow is not None:
            self.macd_slow = macd_slow
        if macd_signal is not None:
            self.macd_signal_period = macd_signal
        if bollinger_period is not None:
            self.bollinger_period = bollinger_period
        if bollinger_std is not None:
            self.bollinger_std = bollinger_std
        if ema_fast is not None:
            self.ema_fast = ema_fast
        if ema_slow is not None:
            self.ema_slow_period = ema_slow
        if atr_period is not None:
            self.atr_period = atr_period

        logger.info("signal_scorer_settings_updated")

    def is_momentum_mode(self, df: pd.DataFrame, rsi: pd.Series) -> tuple[bool, str]:
        """
        Detect if market is in sustained momentum mode.

        Momentum mode is active when RSI has been elevated for multiple candles
        and price structure shows higher lows (bullish continuation pattern).
        When active, overbought penalties are reduced to allow riding trends.

        Args:
            df: DataFrame with OHLCV data
            rsi: RSI series

        Returns:
            Tuple of (is_momentum_active, reason_string)
        """
        if len(rsi) < 3 or len(df) < 12:
            return False, ""

        # Condition 1: RSI sustained above 60 for 3+ candles
        recent_rsi = rsi.tail(3)
        rsi_sustained = bool((recent_rsi > 60).all())

        # Condition 2: Price making higher lows (bullish structure)
        close = df["close"].astype(float)
        recent_close = close.tail(12)
        higher_lows = True

        for i in range(3, len(recent_close)):
            window_min = recent_close.iloc[max(0, i - 3):i].min()
            if recent_close.iloc[i] <= window_min:
                higher_lows = False
                break

        if rsi_sustained and higher_lows:
            return True, "sustained_rsi_higher_lows"
        elif rsi_sustained:
            return True, "sustained_rsi"
        return False, ""

    def record_oversold_buy(self) -> None:
        """Record an oversold buy for rate limiting during crashes."""
        now = datetime.now()
        self._oversold_buy_times.append(now)
        # Clean old entries
        cutoff = now - timedelta(hours=24)
        self._oversold_buy_times = [t for t in self._oversold_buy_times if t > cutoff]
        logger.debug("oversold_buy_recorded", count=len(self._oversold_buy_times))

    def can_buy_oversold(self) -> bool:
        """Check if we can make another oversold buy (max 2 per 24h)."""
        now = datetime.now()
        cutoff = now - timedelta(hours=24)
        recent_buys = [t for t in self._oversold_buy_times if t > cutoff]
        return len(recent_buys) < 2

    def is_price_stabilized(self, close_prices: pd.Series, window_candles: int = 12) -> bool:
        """
        Check if price has stopped falling (stabilized).

        Args:
            close_prices: Series of closing prices
            window_candles: Number of candles to check (default 12 = ~2 hours with 10-min candles)

        Returns:
            True if price is stable/recovering (not making new lows)
        """
        if len(close_prices) < window_candles:
            return True  # Not enough data, allow trade

        recent = close_prices.tail(window_candles)
        current = recent.iloc[-1]
        min_in_window = recent.min()

        # Stabilized if current price > minimum in window (not making new lows)
        # Must be strictly greater - if current == min, we just made a new low
        return current > min_in_window

    def calculate_score(
        self,
        df: pd.DataFrame,
        current_price: Optional[Decimal] = None,
    ) -> SignalResult:
        """
        Calculate composite signal score from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            current_price: Current price (uses latest close if not provided)

        Returns:
            SignalResult with score, action, and breakdown
        """
        if df.empty or len(df) < max(self.ema_slow_period, self.bollinger_period, 26):
            return SignalResult(
                score=0,
                action="hold",
                indicators=IndicatorValues(),
                breakdown={},
                confidence=0.0,
            )

        # Get price series
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else None

        price = float(current_price) if current_price else close.iloc[-1]

        # Calculate all indicators
        rsi = calculate_rsi(close, self.rsi_period)
        macd_result = calculate_macd(close, self.macd_fast, self.macd_slow, self.macd_signal_period)
        bollinger = calculate_bollinger_bands(close, self.bollinger_period, self.bollinger_std)
        ema_result = calculate_ema_crossover(close, self.ema_fast, self.ema_slow_period)
        atr_result = calculate_atr(high, low, close, self.atr_period)

        # Store current indicator values
        indicators = IndicatorValues(
            rsi=rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
            macd_line=macd_result.macd_line.iloc[-1] if not pd.isna(macd_result.macd_line.iloc[-1]) else None,
            macd_signal=macd_result.signal_line.iloc[-1] if not pd.isna(macd_result.signal_line.iloc[-1]) else None,
            macd_histogram=macd_result.histogram.iloc[-1] if not pd.isna(macd_result.histogram.iloc[-1]) else None,
            bb_upper=bollinger.upper_band.iloc[-1] if not pd.isna(bollinger.upper_band.iloc[-1]) else None,
            bb_middle=bollinger.middle_band.iloc[-1] if not pd.isna(bollinger.middle_band.iloc[-1]) else None,
            bb_lower=bollinger.lower_band.iloc[-1] if not pd.isna(bollinger.lower_band.iloc[-1]) else None,
            ema_fast=ema_result.ema_fast.iloc[-1] if not pd.isna(ema_result.ema_fast.iloc[-1]) else None,
            ema_slow=ema_result.ema_slow.iloc[-1] if not pd.isna(ema_result.ema_slow.iloc[-1]) else None,
            atr=atr_result.atr.iloc[-1] if not pd.isna(atr_result.atr.iloc[-1]) else None,
            volatility=get_volatility_level(atr_result),
        )

        # Calculate individual scores
        breakdown = {}
        total_score = 0

        # RSI component (graduated: returns -1.0 to +1.0)
        rsi_signal = get_rsi_signal_graduated(indicators.rsi, self.rsi_oversold, self.rsi_overbought)
        rsi_score = int(rsi_signal * self.weights.rsi)

        # MACD component (graduated: returns -1.0 to +1.0)
        macd_signal = get_macd_signal_graduated(macd_result, price)
        macd_score = int(macd_signal * self.weights.macd)

        # Bollinger Bands component (graduated: returns -1.0 to +1.0)
        bb_signal = get_bollinger_signal_graduated(price, bollinger)
        bb_score = int(bb_signal * self.weights.bollinger)

        # EMA component (graduated: returns -1.0 to +1.0)
        ema_signal = get_ema_signal_graduated(ema_result)
        ema_score = int(ema_signal * self.weights.ema)

        # Momentum mode: reduce overbought penalties during sustained uptrends
        momentum_active, momentum_reason = self.is_momentum_mode(df, rsi)
        if momentum_active:
            original_rsi = rsi_score
            original_bb = bb_score
            # Reduce overbought penalties by 50% (only negative scores)
            if rsi_score < 0:
                rsi_score = rsi_score // 2
            if bb_score < 0:
                bb_score = bb_score // 2
            logger.info(
                "momentum_mode_active",
                reason=momentum_reason,
                rsi_original=original_rsi,
                rsi_adjusted=rsi_score,
                bb_original=original_bb,
                bb_adjusted=bb_score,
            )
        breakdown["momentum"] = 1 if momentum_active else 0

        breakdown["rsi"] = rsi_score
        total_score += rsi_score
        breakdown["macd"] = macd_score
        total_score += macd_score
        breakdown["bollinger"] = bb_score
        total_score += bb_score
        breakdown["ema"] = ema_score
        total_score += ema_score

        # Volume confirmation (boost on high volume, penalty on low volume)
        if volume is not None and len(volume) >= 20:
            volume_sma = volume.rolling(window=20).mean().iloc[-1]
            current_volume = volume.iloc[-1]

            if not pd.isna(volume_sma) and volume_sma > 0:
                volume_ratio = current_volume / volume_sma

                if volume_ratio > 1.5:
                    # High volume: boost signal by 20%
                    volume_boost = int(abs(total_score) * 0.2)
                    if total_score > 0:
                        breakdown["volume"] = volume_boost
                        total_score += volume_boost
                    elif total_score < 0:
                        breakdown["volume"] = -volume_boost
                        total_score -= volume_boost
                    else:
                        breakdown["volume"] = 0
                elif volume_ratio < 0.7:
                    # Low volume: fixed 10-point penalty (consistent behavior)
                    if total_score > 0:
                        breakdown["volume"] = -10
                        total_score -= 10
                    elif total_score < 0:
                        breakdown["volume"] = 10
                        total_score += 10
                    else:
                        breakdown["volume"] = 0
                else:
                    breakdown["volume"] = 0
            else:
                breakdown["volume"] = 0
        else:
            breakdown["volume"] = 0

        # Trend filter: penalize counter-trend trades (scaled by signal strength)
        # Skip penalty for extreme RSI (mean-reversion zones) with crash protection
        trend = get_ema_trend(ema_result)
        trend_adjustment = 0
        rsi_extreme = indicators.rsi is not None and (indicators.rsi < 25 or indicators.rsi > 75)

        # Crash protection checks for mean-reversion trades
        can_mean_revert = self.can_buy_oversold()
        price_stable = self.is_price_stabilized(close)

        if rsi_extreme and can_mean_revert and price_stable:
            # Allow mean-reversion trade - all conditions met
            logger.debug("trend_filter_skipped", reason="extreme_rsi_stable", rsi=indicators.rsi)
        elif rsi_extreme and not can_mean_revert:
            # Hit buy limit - apply trend filter anyway
            logger.debug("trend_filter_applied", reason="oversold_buy_limit_reached", rsi=indicators.rsi)
            if total_score > 0 and trend == "bearish":
                signal_confidence = abs(total_score) / 100
                trend_adjustment = -int(20 * (1 - signal_confidence * 0.5))
                total_score += trend_adjustment
        elif rsi_extreme and not price_stable:
            # Price still falling - apply trend filter
            logger.debug("trend_filter_applied", reason="price_still_falling", rsi=indicators.rsi)
            if total_score > 0 and trend == "bearish":
                signal_confidence = abs(total_score) / 100
                trend_adjustment = -int(20 * (1 - signal_confidence * 0.5))
                total_score += trend_adjustment
        elif total_score > 0 and trend == "bearish":
            # Scale penalty: stronger signals get less penalty (10-20 points)
            signal_confidence = abs(total_score) / 100
            trend_adjustment = -int(20 * (1 - signal_confidence * 0.5))
            total_score += trend_adjustment
        elif total_score < 0 and trend == "bullish":
            # Scale penalty: stronger signals get less penalty
            signal_confidence = abs(total_score) / 100
            trend_adjustment = int(20 * (1 - signal_confidence * 0.5))
            total_score += trend_adjustment
        breakdown["trend_filter"] = trend_adjustment

        # Clamp score to -100 to +100
        total_score = max(-100, min(100, total_score))

        # Determine action
        if total_score >= self.threshold:
            action = "buy"
        elif total_score <= -self.threshold:
            action = "sell"
        else:
            action = "hold"

        # Calculate confidence with confluence factor
        # Combines magnitude with how many indicators agree
        if action != "hold":
            # Count agreeing indicators (non-zero contributions, excluding momentum flag)
            confluence_count = sum(
                1 for key, score in breakdown.items()
                if score != 0 and key != "momentum"
            )
            confluence_factor = confluence_count / 6  # 6 components including trend_filter

            # Combine magnitude and confluence (equally weighted)
            magnitude_confidence = abs(total_score) / 100
            confidence = (magnitude_confidence + confluence_factor) / 2
            confidence = min(1.0, confidence)
        else:
            confidence = 0.0

        result = SignalResult(
            score=total_score,
            action=action,
            indicators=indicators,
            breakdown=breakdown,
            confidence=confidence,
        )

        logger.debug(
            "signal_calculated",
            score=total_score,
            action=action,
            breakdown=breakdown,
            confidence=confidence,
        )

        return result

    def get_trend(self, df: pd.DataFrame) -> str:
        """
        Get current market trend.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            "bullish", "bearish", or "neutral"
        """
        if df.empty or len(df) < self.ema_slow_period:
            return "neutral"

        close = df["close"].astype(float)
        ema_result = calculate_ema_crossover(close, self.ema_fast, self.ema_slow_period)

        return get_ema_trend(ema_result)
