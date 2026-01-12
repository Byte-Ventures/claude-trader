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


class TestHTFNullSafety:
    """Tests for HTF (Higher Timeframe) null safety."""

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

    def test_build_reviewer_prompt_hides_htf_for_empty_string_values(self, mock_db, reviewer_models, judge_model):
        """Test that empty string HTF values result in HTF line being hidden."""
        reviewer = TradeReviewer(
            api_key="test-key",
            db=mock_db,
            reviewer_models=reviewer_models,
            judge_model=judge_model,
        )

        context = {
            'breakdown': {
                '_htf_trend': '',      # Empty string - not actionable
                '_htf_daily': '',
                '_htf_4h': '',
            },
            'score': 75,
            'threshold': 70,
            'price': 50000,
            'candle_interval': '1h',
            'trading_style_desc': 'swing trading (hours to days)',
            'position_percent': 50.0,
            'action': 'buy',
            'fear_greed': 50,
            'fear_greed_class': 'Neutral',
            'win_rate': 60.0,
            'net_pnl': 1000.0,
            'total_trades': 10,
        }

        prompt = reviewer._build_reviewer_prompt(context)

        # Empty string HTF trend is not actionable, so HTF line should be hidden
        assert 'HIGHER TIMEFRAME BIAS' not in prompt

    def test_build_reviewer_prompt_hides_htf_for_none_values(self, mock_db, reviewer_models, judge_model):
        """Test that None HTF values result in HTF line being hidden."""
        reviewer = TradeReviewer(
            api_key="test-key",
            db=mock_db,
            reviewer_models=reviewer_models,
            judge_model=judge_model,
        )

        context = {
            'breakdown': {
                '_htf_trend': None,    # None -> "unknown" -> hidden
                '_htf_daily': None,
                '_htf_4h': None,
            },
            'score': 75,
            'threshold': 70,
            'price': 50000,
            'candle_interval': '1h',
            'trading_style_desc': 'swing trading (hours to days)',
            'position_percent': 50.0,
            'action': 'buy',
            'fear_greed': 50,
            'fear_greed_class': 'Neutral',
            'win_rate': 60.0,
            'net_pnl': 1000.0,
            'total_trades': 10,
        }

        prompt = reviewer._build_reviewer_prompt(context)

        # None values become "unknown" which is not actionable, so HTF line should be hidden
        assert 'HIGHER TIMEFRAME BIAS' not in prompt

    def test_build_reviewer_prompt_shows_htf_for_bullish_trend(self, mock_db, reviewer_models, judge_model):
        """Test that actionable HTF trends (bullish/bearish) are shown."""
        reviewer = TradeReviewer(
            api_key="test-key",
            db=mock_db,
            reviewer_models=reviewer_models,
            judge_model=judge_model,
        )

        context = {
            'breakdown': {
                '_htf_trend': 'bullish',
                '_htf_daily': 'bullish',
                '_htf_4h': 'neutral',
            },
            'score': 75,
            'threshold': 70,
            'price': 50000,
            'candle_interval': '1h',
            'trading_style_desc': 'swing trading (hours to days)',
            'position_percent': 50.0,
            'action': 'buy',
            'fear_greed': 50,
            'fear_greed_class': 'Neutral',
            'win_rate': 60.0,
            'net_pnl': 1000.0,
            'total_trades': 10,
        }

        prompt = reviewer._build_reviewer_prompt(context)

        # Bullish trend is actionable, so HTF line should be shown
        assert 'HIGHER TIMEFRAME BIAS: BULLISH (Daily: BULLISH, 4H: NEUTRAL)' in prompt


class TestMomentumConcernVeto:
    """Tests for momentum concern veto threshold feature."""

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

    def _create_reviewer(self, mock_db, reviewer_models, judge_model, momentum_threshold=0.70):
        """Helper to create a TradeReviewer with given momentum threshold."""
        return TradeReviewer(
            api_key="test-key",
            db=mock_db,
            reviewer_models=reviewer_models,
            judge_model=judge_model,
            veto_reduce_threshold=0.65,
            veto_skip_threshold=0.80,
            veto_skip_threshold_momentum=momentum_threshold,
        )

    def test_has_momentum_concern_detects_momentum_confirmation(self, mock_db, reviewer_models, judge_model):
        """Test _has_momentum_concern detects 'momentum confirmation' phrase."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        assert reviewer._has_momentum_concern("momentum confirmation is critical and currently lacking")
        assert reviewer._has_momentum_concern("The momentum confirmation appears to be missing")
        assert reviewer._has_momentum_concern("MOMENTUM CONFIRMATION is needed")  # case insensitive

    def test_has_momentum_concern_detects_lacking_confirmation_with_momentum(self, mock_db, reviewer_models, judge_model):
        """Test _has_momentum_concern detects generic confirmation phrases when momentum is mentioned."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        # Generic phrases WITH momentum context should trigger
        assert reviewer._has_momentum_concern("Signal is lacking confirmation from volume, momentum is weak")
        assert reviewer._has_momentum_concern("The momentum-based trade lacks confirmation")
        assert reviewer._has_momentum_concern("Trend is without confirmation, momentum indicators unclear")
        assert reviewer._has_momentum_concern("Missing confirmation from momentum signals")

    def test_has_momentum_concern_ignores_generic_phrases_without_momentum(self, mock_db, reviewer_models, judge_model):
        """Test _has_momentum_concern ignores generic confirmation phrases when momentum is NOT mentioned."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        # Generic phrases WITHOUT momentum context should NOT trigger
        assert not reviewer._has_momentum_concern("RSI confirmation is lacking")
        assert not reviewer._has_momentum_concern("Volume confirmation is lacking")
        assert not reviewer._has_momentum_concern("Signal lacks confirmation from trend indicators")
        assert not reviewer._has_momentum_concern("Missing confirmation from MACD")

    def test_has_momentum_concern_detects_momentum_specific_phrases(self, mock_db, reviewer_models, judge_model):
        """Test _has_momentum_concern detects momentum-specific phrases."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        # Momentum-specific phrases that should always match
        assert reviewer._has_momentum_concern("weak momentum signals detected")
        assert reviewer._has_momentum_concern("momentum divergence is concerning")
        assert reviewer._has_momentum_concern("momentum not confirmed by volume")
        assert reviewer._has_momentum_concern("unconfirmed momentum makes this risky")
        assert reviewer._has_momentum_concern("there is no momentum in this move")
        assert reviewer._has_momentum_concern("the trade lacks momentum")
        assert reviewer._has_momentum_concern("lacking momentum indicators support")
        assert reviewer._has_momentum_concern("proceeding without momentum is risky")
        assert reviewer._has_momentum_concern("momentum weakness is evident")

    def test_has_momentum_concern_detects_no_momentum_confirmation(self, mock_db, reviewer_models, judge_model):
        """Test _has_momentum_concern detects 'no momentum confirmation' phrase."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        # "no momentum confirmation" should match (PR #356 review feedback)
        assert reviewer._has_momentum_concern("there is no momentum confirmation for this signal")
        assert reviewer._has_momentum_concern("No momentum confirmation detected")

    def test_has_momentum_concern_detects_momentum_not_aligned(self, mock_db, reviewer_models, judge_model):
        """Test _has_momentum_concern detects 'momentum not aligned' phrase."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        # "momentum not aligned" should match (PR #356 review feedback)
        assert reviewer._has_momentum_concern("momentum not aligned with trend")
        assert reviewer._has_momentum_concern("the momentum is not aligned")

    def test_has_momentum_concern_returns_false_for_normal_reasoning(self, mock_db, reviewer_models, judge_model):
        """Test _has_momentum_concern returns False for normal reasoning."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        assert not reviewer._has_momentum_concern("RSI is overbought")
        assert not reviewer._has_momentum_concern("Strong bullish signal confirmed by volume")
        assert not reviewer._has_momentum_concern("Momentum looks good")

    def test_veto_skip_at_momentum_threshold_with_momentum_concern(self, mock_db, reviewer_models, judge_model):
        """Test SKIP veto when confidence >= momentum threshold AND momentum concern present."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, momentum_threshold=0.70)

        # 72% confidence with momentum concern should SKIP (>= 70% threshold)
        veto = reviewer._determine_veto_action(
            approved=False,
            confidence=0.72,
            reasoning="momentum confirmation is critical and currently lacking"
        )
        assert veto == "skip"

    def test_veto_reduce_below_momentum_threshold_with_momentum_concern(self, mock_db, reviewer_models, judge_model):
        """Test REDUCE veto when confidence < momentum threshold but >= reduce threshold."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, momentum_threshold=0.70)

        # 68% confidence with momentum concern should REDUCE (>= 65% reduce, < 70% momentum)
        veto = reviewer._determine_veto_action(
            approved=False,
            confidence=0.68,
            reasoning="momentum confirmation is critical and currently lacking"
        )
        assert veto == "reduce"

    def test_no_veto_for_normal_reasoning_below_skip_threshold(self, mock_db, reviewer_models, judge_model):
        """Test normal behavior when no momentum concern present."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, momentum_threshold=0.70)

        # 72% confidence without momentum concern should REDUCE (standard behavior)
        veto = reviewer._determine_veto_action(
            approved=False,
            confidence=0.72,
            reasoning="RSI is overbought and trend is weak"
        )
        assert veto == "reduce"  # Standard tier: 65-80% = reduce

    def test_veto_skip_at_standard_threshold_without_momentum_concern(self, mock_db, reviewer_models, judge_model):
        """Test standard SKIP threshold still works for non-momentum concerns."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, momentum_threshold=0.70)

        # 82% confidence without momentum concern should SKIP at standard threshold
        veto = reviewer._determine_veto_action(
            approved=False,
            confidence=0.82,
            reasoning="RSI is overbought and volume is declining"
        )
        assert veto == "skip"

    def test_no_veto_when_approved(self, mock_db, reviewer_models, judge_model):
        """Test no veto when judge approves, even with momentum concerns."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model)

        veto = reviewer._determine_veto_action(
            approved=True,
            confidence=0.75,
            reasoning="momentum confirmation is good despite earlier concerns"
        )
        assert veto is None

    def test_exact_momentum_threshold_triggers_skip(self, mock_db, reviewer_models, judge_model):
        """Test that exactly at momentum threshold triggers skip."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, momentum_threshold=0.70)

        veto = reviewer._determine_veto_action(
            approved=False,
            confidence=0.70,  # Exactly at threshold
            reasoning="lacking confirmation from momentum indicators"
        )
        assert veto == "skip"

    def test_empty_reasoning_uses_standard_thresholds(self, mock_db, reviewer_models, judge_model):
        """Test that empty reasoning falls back to standard thresholds."""
        reviewer = self._create_reviewer(mock_db, reviewer_models, judge_model, momentum_threshold=0.70)

        # 72% with empty reasoning should REDUCE (standard behavior)
        veto = reviewer._determine_veto_action(
            approved=False,
            confidence=0.72,
            reasoning=""
        )
        assert veto == "reduce"
