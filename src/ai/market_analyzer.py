"""
Hourly market analysis during volatile conditions.

AI-powered market analysis that runs once per hour when volatility is high/extreme.
Independent of trading decisions - purely informational for monitoring.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx
import structlog

from src.strategy.signal_scorer import IndicatorValues

logger = structlog.get_logger(__name__)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default model for market analysis
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"


@dataclass
class MarketAnalysis:
    """Result of hourly market analysis."""

    outlook: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    summary: str  # 2-3 sentence analysis
    recommendation: str  # "wait", "accumulate", "reduce"
    key_observation: str  # Single most important observation
    timestamp: datetime


SYSTEM_PROMPT = """You are a Bitcoin market analyst providing hourly updates during volatile market conditions.

Analyze the current market state based on technical indicators and sentiment data. Provide a concise, actionable summary.

Indicators explained:
- RSI: 0-100 scale. <30 = oversold (bullish), >70 = overbought (bearish)
- MACD histogram: Positive = bullish momentum, Negative = bearish momentum
- Bollinger %B: 0-1 scale. <0.2 = near lower band (oversold), >0.8 = near upper band (overbought)
- EMA gap: Price vs slow EMA. Negative = price below EMA (bearish trend)
- Fear & Greed: 0-100. <25 = Extreme Fear, >75 = Extreme Greed

Respond with JSON only:
{
  "outlook": "bullish"/"bearish"/"neutral",
  "confidence": 0.0-1.0,
  "summary": "2-3 sentence market analysis",
  "recommendation": "wait"/"accumulate"/"reduce",
  "key_observation": "single most important observation"
}

Be direct and specific. Focus on what traders should know right now."""


class MarketAnalyzer:
    """
    Hourly market analysis during volatile conditions.

    Provides AI-powered market summaries independent of trading decisions.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize market analyzer.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: claude-sonnet-4.5)
        """
        self.api_key = api_key
        self.model = model

    async def analyze_market(
        self,
        indicators: IndicatorValues,
        current_price: Decimal,
        fear_greed: int,
        fear_greed_class: str,
        regime: str,
        volatility: str,
        price_change_1h: Optional[float] = None,
        price_change_24h: Optional[float] = None,
    ) -> MarketAnalysis:
        """
        Generate AI market analysis.

        Args:
            indicators: Current indicator values
            current_price: Current BTC price
            fear_greed: Fear & Greed index value
            fear_greed_class: Fear & Greed classification
            regime: Current market regime
            volatility: Volatility level (low/normal/high/extreme)
            price_change_1h: Optional 1-hour price change %
            price_change_24h: Optional 24-hour price change %

        Returns:
            MarketAnalysis with outlook and summary
        """
        prompt = self._build_prompt(
            indicators, current_price, fear_greed, fear_greed_class,
            regime, volatility, price_change_1h, price_change_24h
        )

        try:
            response = await self._call_api(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error("market_analysis_failed", error=str(e))
            # Return fallback analysis
            return MarketAnalysis(
                outlook="neutral",
                confidence=0.0,
                summary=f"Analysis unavailable: {str(e)[:100]}",
                recommendation="wait",
                key_observation="Unable to analyze market conditions",
                timestamp=datetime.now(),
            )

    def _build_prompt(
        self,
        indicators: IndicatorValues,
        current_price: Decimal,
        fear_greed: int,
        fear_greed_class: str,
        regime: str,
        volatility: str,
        price_change_1h: Optional[float],
        price_change_24h: Optional[float],
    ) -> str:
        """Build the analysis prompt."""
        # Format indicator values
        rsi_str = f"{indicators.rsi:.1f}" if indicators.rsi else "N/A"
        macd_str = f"{indicators.macd_histogram:.2f}" if indicators.macd_histogram else "N/A"
        bb_str = "N/A"
        if indicators.bb_lower and indicators.bb_upper and indicators.bb_middle:
            price_float = float(current_price)
            bb_range = indicators.bb_upper - indicators.bb_lower
            if bb_range > 0:
                bb_pct = (price_float - indicators.bb_lower) / bb_range
                bb_str = f"{bb_pct:.2f}"

        ema_gap_str = "N/A"
        if indicators.ema_slow and indicators.ema_slow > 0:
            ema_gap = ((float(current_price) - indicators.ema_slow) / indicators.ema_slow) * 100
            ema_gap_str = f"{ema_gap:+.2f}%"

        # Build price change info
        price_changes = []
        if price_change_1h is not None:
            price_changes.append(f"1h: {price_change_1h:+.2f}%")
        if price_change_24h is not None:
            price_changes.append(f"24h: {price_change_24h:+.2f}%")
        price_change_str = ", ".join(price_changes) if price_changes else "N/A"

        return f"""Analyze current Bitcoin market conditions:

Price: ${float(current_price):,.2f}
Price Changes: {price_change_str}
Volatility: {volatility.upper()}
Market Regime: {regime}

Technical Indicators:
- RSI: {rsi_str}
- MACD Histogram: {macd_str}
- Bollinger %B: {bb_str}
- EMA Gap: {ema_gap_str}

Sentiment:
- Fear & Greed Index: {fear_greed} ({fear_greed_class})

Provide your hourly market analysis."""

    async def _call_api(self, prompt: str) -> str:
        """Call OpenRouter API."""
        request_body = {
            "model": self.model,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

        logger.debug(
            "market_analysis_request",
            model=self.model,
            prompt=prompt,
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/claude-trader",
                },
                json=request_body,
            )

            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                return choices[0]["message"]["content"]

            return "{}"

    def _parse_response(self, response: str) -> MarketAnalysis:
        """Parse API response into MarketAnalysis."""
        try:
            # Extract JSON from response
            data = self._extract_json(response)

            outlook = data.get("outlook", "neutral")
            confidence = float(data.get("confidence", 0.5))
            summary = data.get("summary", "No analysis available")
            recommendation = data.get("recommendation", "wait")
            key_observation = data.get("key_observation", "")

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            return MarketAnalysis(
                outlook=outlook,
                confidence=confidence,
                summary=summary,
                recommendation=recommendation,
                key_observation=key_observation,
                timestamp=datetime.now(),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("market_analysis_parse_failed", error=str(e))
            return MarketAnalysis(
                outlook="neutral",
                confidence=0.0,
                summary=f"Parse error: {response[:100]}",
                recommendation="wait",
                key_observation="Unable to parse analysis",
                timestamp=datetime.now(),
            )

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from response."""
        # Try direct parse first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Find JSON with brace counting
        start = response.find('{')
        if start == -1:
            raise ValueError("No JSON found in response")

        depth = 0
        for i, char in enumerate(response[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(response[start:i+1])

        raise ValueError("Unbalanced JSON braces in response")
