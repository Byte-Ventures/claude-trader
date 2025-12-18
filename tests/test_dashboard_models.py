"""Tests for dashboard Pydantic models."""

from src.dashboard.models import SignalInfo


def test_signal_info_breakdown_mixed_types():
    """Test SignalInfo accepts breakdown with mixed types.

    The breakdown dict contains:
    - Integers: score contributions (rsi, macd, bollinger, etc.)
    - Floats: raw indicator values (_rsi_value, _bb_position, etc.)
    - Strings: HTF trends (_htf_trend, _htf_daily, _htf_4h)
    - Booleans: flags (_whale_activity, _momentum_active)
    """
    breakdown = {
        "rsi": 25,  # int - score contribution
        "macd": -10,  # int - score contribution
        "_rsi_value": 43.61,  # float - raw value
        "_bb_position": 0.247,  # float - raw value
        "_htf_trend": "bearish",  # str - trend direction
        "_htf_daily": "bullish",  # str - trend direction
        "_whale_activity": True,  # bool - flag
    }
    signal_info = SignalInfo(
        score=60,
        action="buy",
        threshold=60,
        breakdown=breakdown,
        confidence=0.75,
    )
    data = signal_info.model_dump()
    assert data["breakdown"] == breakdown
    assert data["score"] == 60
    assert data["action"] == "buy"
