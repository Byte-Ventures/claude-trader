"""
Claude AI integration for trade analysis.

Provides:
- TradeReviewer: Pre-trade validation with sentiment analysis
- Fear & Greed Index fetching
- Historical trade analysis
"""

from src.ai.trade_reviewer import TradeReviewer, ReviewResult

__all__ = ["TradeReviewer", "ReviewResult"]
