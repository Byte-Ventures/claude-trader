"""Tests for multi-agent trade reviewer."""

import pytest
from unittest.mock import MagicMock

from src.ai.trade_reviewer import TradeReviewer


class TestTradingStyle:
    """Tests for _get_trading_style() method."""

    @pytest.fixture
    def mock_db(self):
        """Create minimal mock database."""
        return MagicMock()

    @pytest.fixture
    def reviewer_models(self):
        """Reviewer model list."""
        return ["model1", "model2", "model3"]

    @pytest.fixture
    def judge_model(self):
        """Judge model name."""
        return "judge-model"

    def _create_reviewer(self, mock_db, reviewer_models, judge_model, candle_interval):
        """Helper to create a TradeReviewer with given candle interval."""
        return TradeReviewer(
            api_key="test-key",
            db=mock_db,
            reviewer_models=reviewer_models,
            judge_model=judge_model,
            candle_interval=candle_interval,
        )

    def test_daytrading_one_minute(self, mock_db, reviewer_models, judge_model):
        """ONE_MINUTE candles should be classified as daytrading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "ONE_MINUTE")
        style, description = reviewer._get_trading_style()

        assert style == "daytrading"
        assert "short-term" in description
        assert "minutes" in description

    def test_daytrading_five_minute(self, mock_db, reviewer_models, judge_model):
        """FIVE_MINUTE candles should be classified as daytrading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "FIVE_MINUTE")
        style, description = reviewer._get_trading_style()

        assert style == "daytrading"

    def test_daytrading_fifteen_minute(self, mock_db, reviewer_models, judge_model):
        """FIFTEEN_MINUTE candles should be classified as daytrading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "FIFTEEN_MINUTE")
        style, description = reviewer._get_trading_style()

        assert style == "daytrading"

    def test_swing_thirty_minute(self, mock_db, reviewer_models, judge_model):
        """THIRTY_MINUTE candles should be classified as swing trading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "THIRTY_MINUTE")
        style, description = reviewer._get_trading_style()

        assert style == "swing"
        assert "hours" in description

    def test_swing_one_hour(self, mock_db, reviewer_models, judge_model):
        """ONE_HOUR candles should be classified as swing trading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "ONE_HOUR")
        style, description = reviewer._get_trading_style()

        assert style == "swing"

    def test_swing_two_hour(self, mock_db, reviewer_models, judge_model):
        """TWO_HOUR candles should be classified as swing trading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "TWO_HOUR")
        style, description = reviewer._get_trading_style()

        assert style == "swing"

    def test_position_four_hour(self, mock_db, reviewer_models, judge_model):
        """FOUR_HOUR candles should be classified as position trading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "FOUR_HOUR")
        style, description = reviewer._get_trading_style()

        assert style == "position"
        assert "days" in description

    def test_position_six_hour(self, mock_db, reviewer_models, judge_model):
        """SIX_HOUR candles should be classified as position trading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "SIX_HOUR")
        style, description = reviewer._get_trading_style()

        assert style == "position"

    def test_position_one_day(self, mock_db, reviewer_models, judge_model):
        """ONE_DAY candles should be classified as position trading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "ONE_DAY")
        style, description = reviewer._get_trading_style()

        assert style == "position"

    def test_default_candle_interval(self, mock_db, reviewer_models, judge_model):
        """Default candle interval (ONE_HOUR) should be swing trading."""
        # Create reviewer without specifying candle_interval (uses default)
        reviewer = TradeReviewer(
            api_key="test-key",
            db=mock_db,
            reviewer_models=reviewer_models,
            judge_model=judge_model,
        )
        style, description = reviewer._get_trading_style()

        # Default is ONE_HOUR = swing trading
        assert style == "swing"

    def test_unknown_interval_falls_to_position(self, mock_db, reviewer_models, judge_model):
        """Unknown candle intervals should default to position trading."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, "UNKNOWN_INTERVAL")
        style, description = reviewer._get_trading_style()

        # Unknown intervals fall through to position trading (conservative default)
        assert style == "position"


class TestHTFDisplayFormatting:
    """Tests for HTF (Higher Timeframe) data display formatting."""

    @pytest.fixture
    def mock_db(self):
        """Create minimal mock database."""
        return MagicMock()

    @pytest.fixture
    def reviewer(self, mock_db):
        """Create a TradeReviewer instance."""
        return TradeReviewer(
            api_key="test-key",
            db=mock_db,
            reviewer_models=["model1"],
            judge_model="judge-model",
        )

    def test_htf_unknown_values_display_correctly(self, reviewer):
        """Test that 'unknown' HTF values are formatted correctly with .upper()."""
        # Create context with None HTF values (which become "unknown")
        context = {
            'price': 50000.0,
            'score': 75,
            'threshold': 50,
            'breakdown': {
                '_htf_trend': None,  # Will become "unknown"
                '_htf_daily': None,  # Will become "unknown"
                '_htf_4h': None,     # Will become "unknown"
            },
            'trading_style_desc': 'swing trading',
            'candle_interval': 'ONE_HOUR',
            'fear_greed': 50,
            'fear_greed_class': 'Neutral',
            'win_rate': 60.0,
            'net_pnl': 1000.0,
            'total_trades': 10,
            'action': 'buy',
        }

        # Build the prompt (which includes HTF formatting)
        prompt = reviewer._build_reviewer_prompt(context)

        # Verify that "unknown" values are properly uppercased in the prompt
        assert "HIGHER TIMEFRAME BIAS: UNKNOWN (Daily: UNKNOWN, 4H: UNKNOWN)" in prompt

    def test_htf_mixed_values_display_correctly(self, reviewer):
        """Test HTF display with mix of unknown and actual trend values."""
        context = {
            'price': 50000.0,
            'score': 75,
            'threshold': 50,
            'breakdown': {
                '_htf_trend': 'bullish',
                '_htf_daily': None,      # Will become "unknown"
                '_htf_4h': 'bearish',
            },
            'trading_style_desc': 'swing trading',
            'candle_interval': 'ONE_HOUR',
            'fear_greed': 50,
            'fear_greed_class': 'Neutral',
            'win_rate': 60.0,
            'net_pnl': 1000.0,
            'total_trades': 10,
            'action': 'buy',
        }

        prompt = reviewer._build_reviewer_prompt(context)

        # Verify mixed values are properly formatted
        assert "HIGHER TIMEFRAME BIAS: BULLISH (Daily: UNKNOWN, 4H: BEARISH)" in prompt
