"""
Tests for Telegram notification system.

Tests cover:
- Trade rejection notifications
- Input validation for notify_trade_rejected
- Paper/live mode indicators
- Message formatting
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock

from src.notifications.telegram import TelegramNotifier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_bot():
    """Create a mock Telegram Bot."""
    with patch("src.notifications.telegram.Bot") as mock_bot_class:
        mock_bot_instance = MagicMock()
        mock_bot_instance.send_message = AsyncMock(return_value=MagicMock())
        mock_bot_class.return_value = mock_bot_instance
        yield mock_bot_instance


@pytest.fixture
def notifier(mock_bot):
    """Create a notifier with mocked bot."""
    return TelegramNotifier(
        bot_token="test_token",
        chat_id="test_chat_id",
        enabled=True,
        db=None,
        is_paper=False,
    )


@pytest.fixture
def paper_notifier(mock_bot):
    """Create a paper mode notifier with mocked bot."""
    return TelegramNotifier(
        bot_token="test_token",
        chat_id="test_chat_id",
        enabled=True,
        db=None,
        is_paper=True,
    )


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock()
    db.save_notification = MagicMock()
    return db


@pytest.fixture
def notifier_with_db(mock_bot, mock_db):
    """Create a notifier with mocked bot and database."""
    return TelegramNotifier(
        bot_token="test_token",
        chat_id="test_chat_id",
        enabled=True,
        db=mock_db,
        is_paper=False,
    )


# ============================================================================
# notify_trade_rejected Tests
# ============================================================================

def test_notify_trade_rejected_basic(notifier, mock_bot):
    """Test basic trade rejection notification."""
    notifier.notify_trade_rejected(
        side="buy",
        reason="Stop too tight",
        price=Decimal("100000.00"),
        signal_score=75,
        size_quote=Decimal("5000.00"),
        is_paper=False,
    )

    # Verify message was sent
    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    # Verify message contents
    assert "Trade Rejected" in message
    assert "BUY" in message
    assert "Stop too tight" in message


def test_notify_trade_rejected_paper_mode(notifier, mock_bot):
    """Test trade rejection shows PAPER indicator."""
    notifier.notify_trade_rejected(
        side="sell",
        reason="Profit margin too low",
        is_paper=True,
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    assert "[PAPER]" in message


def test_notify_trade_rejected_live_mode(notifier, mock_bot):
    """Test trade rejection in live mode does not show PAPER indicator."""
    notifier.notify_trade_rejected(
        side="buy",
        reason="Position limit exceeded",
        is_paper=False,
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    assert "[PAPER]" not in message


def test_notify_trade_rejected_invalid_side_returns_early(notifier, mock_bot):
    """Test invalid side value causes early return without sending."""
    notifier.notify_trade_rejected(
        side="invalid",  # Not "buy" or "sell"
        reason="Test reason",
    )

    # Message should NOT be sent for invalid side
    mock_bot.send_message.assert_not_called()


def test_notify_trade_rejected_invalid_price_returns_early(notifier, mock_bot):
    """Test negative price causes early return without sending."""
    notifier.notify_trade_rejected(
        side="buy",
        reason="Test reason",
        price=Decimal("-100.00"),  # Invalid negative price
    )

    # Message should NOT be sent for invalid price
    mock_bot.send_message.assert_not_called()


def test_notify_trade_rejected_zero_price_returns_early(notifier, mock_bot):
    """Test zero price causes early return without sending."""
    notifier.notify_trade_rejected(
        side="buy",
        reason="Test reason",
        price=Decimal("0.00"),  # Invalid zero price
    )

    # Message should NOT be sent for zero price
    mock_bot.send_message.assert_not_called()


def test_notify_trade_rejected_out_of_range_signal_score_still_sends(notifier, mock_bot):
    """Test out of range signal score logs warning but still sends."""
    notifier.notify_trade_rejected(
        side="buy",
        reason="Test reason",
        signal_score=150,  # Out of -100 to 100 range
    )

    # Message should still be sent (just warns)
    mock_bot.send_message.assert_called()


def test_notify_trade_rejected_optional_fields(notifier, mock_bot):
    """Test notification works with only required fields."""
    notifier.notify_trade_rejected(
        side="sell",
        reason="Validation failed",
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    assert "Trade Rejected" in message
    assert "SELL" in message
    assert "Validation failed" in message


def test_notify_trade_rejected_saves_to_dashboard(mock_bot, mock_db):
    """Test trade rejection is saved to dashboard."""
    notifier = TelegramNotifier(
        bot_token="test_token",
        chat_id="test_chat_id",
        enabled=True,
        db=mock_db,
        is_paper=False,
    )

    notifier.notify_trade_rejected(
        side="buy",
        reason="Stop too tight",
    )

    # Verify notification was saved to dashboard
    mock_db.save_notification.assert_called_once()
    call_args = mock_db.save_notification.call_args
    assert call_args.kwargs.get("type") == "trade_rejected"


def test_notify_trade_rejected_paper_vs_live_separation(mock_bot, mock_db):
    """Test paper and live rejections save with correct is_paper flag."""
    # Create two notifiers - one paper, one live
    paper_notifier = TelegramNotifier(
        bot_token="test_token",
        chat_id="test_chat_id",
        enabled=True,
        db=mock_db,
        is_paper=True,
    )

    live_notifier = TelegramNotifier(
        bot_token="test_token",
        chat_id="test_chat_id",
        enabled=True,
        db=mock_db,
        is_paper=False,
    )

    # Send paper rejection
    paper_notifier.notify_trade_rejected(
        side="buy",
        reason="Paper rejection",
        is_paper=True,
    )

    # Verify paper flag in dashboard save
    paper_call = mock_db.save_notification.call_args_list[0]
    assert paper_call.kwargs.get("is_paper") is True

    mock_db.save_notification.reset_mock()

    # Send live rejection
    live_notifier.notify_trade_rejected(
        side="buy",
        reason="Live rejection",
        is_paper=False,
    )

    # Verify live flag in dashboard save
    live_call = mock_db.save_notification.call_args
    assert live_call.kwargs.get("is_paper") is False


# ============================================================================
# notify_trade Tests
# ============================================================================

def test_notify_trade_basic(notifier, mock_bot):
    """Test basic trade notification."""
    notifier.notify_trade(
        side="buy",
        size=Decimal("0.001"),
        price=Decimal("100000.00"),
        fee=Decimal("0.60"),
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    assert "Trade Executed" in message
    assert "BUY" in message


def test_notify_trade_with_stop_loss(notifier, mock_bot):
    """Test trade notification includes stop loss for buys."""
    notifier.notify_trade(
        side="buy",
        size=Decimal("0.001"),
        price=Decimal("100000.00"),
        fee=Decimal("0.60"),
        stop_loss=Decimal("98500.00"),
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    assert "Stop Loss" in message


def test_notify_trade_with_realized_pnl(notifier, mock_bot):
    """Test sell trade notification includes realized P&L."""
    notifier.notify_trade(
        side="sell",
        size=Decimal("0.001"),
        price=Decimal("102000.00"),
        fee=Decimal("0.60"),
        realized_pnl=Decimal("150.00"),
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    assert "P&L" in message


def test_notify_trade_paper_mode(notifier, mock_bot):
    """Test trade notification shows PAPER indicator."""
    notifier.notify_trade(
        side="buy",
        size=Decimal("0.001"),
        price=Decimal("100000.00"),
        fee=Decimal("0.60"),
        is_paper=True,
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    assert "[PAPER]" in message


# ============================================================================
# Disabled Notifier Tests
# ============================================================================

def test_disabled_notifier_does_not_send():
    """Test disabled notifier does not send messages."""
    notifier = TelegramNotifier(
        bot_token="",
        chat_id="",
        enabled=False,
    )

    # This should not raise, just do nothing
    notifier.notify_trade_rejected(
        side="buy",
        reason="Test",
    )

    # No bot should be initialized
    assert notifier._bot is None


# ============================================================================
# notify_error Truncation Tests
# ============================================================================

def test_notify_error_truncates_long_error_with_smart_truncation(notifier, mock_bot):
    """Test that long non-stack-trace error messages use smart truncation (first 400 + last 100)."""
    # Create error with 600 chars (exceeds 500 char limit)
    long_error = "A" * 400 + "B" * 200

    notifier.notify_error(
        error=long_error,
        context="Test context",
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    # Verify smart truncation occurred:
    # - Should have first 400 chars (all A's)
    # - Should have "..." in the middle
    # - Should have last 100 chars (all B's)
    assert "A" * 400 in message
    assert "..." in message
    assert "B" * 100 in message
    # Should NOT contain the full original error
    assert long_error not in message


def test_notify_error_truncates_long_context_with_smart_truncation(notifier, mock_bot):
    """Test that long context messages use smart truncation (first 400 + last 100)."""
    # Create context with 600 chars (exceeds 500 char limit)
    long_context = "X" * 400 + "Y" * 200

    notifier.notify_error(
        error="Short error",
        context=long_context,
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    # Verify smart truncation occurred for context
    assert "X" * 400 in message
    assert "..." in message
    assert "Y" * 100 in message
    # Should NOT contain the full original context
    assert long_context not in message


def test_notify_error_does_not_truncate_short_messages(notifier, mock_bot):
    """Test that short error and context messages are not truncated."""
    short_error = "Database connection failed"
    short_context = "Main trading loop"

    notifier.notify_error(
        error=short_error,
        context=short_context,
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    # Short messages should appear exactly as provided
    assert short_error in message
    assert short_context in message
    # No truncation indicator
    assert message.count("...") == 0


def test_notify_error_truncates_both_error_and_context(notifier, mock_bot):
    """Test that both error and context are truncated when both are long."""
    long_error = "E" * 600
    long_context = "C" * 600

    notifier.notify_error(
        error=long_error,
        context=long_context,
    )

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    # Both should be truncated
    # Count the ellipsis - should have 2 (one for error, one for context)
    assert message.count("...") == 2


def test_notify_error_exactly_at_limit(notifier, mock_bot):
    """Test messages exactly at 500 chars are not truncated."""
    exactly_500 = "A" * 500
    notifier.notify_error(error=exactly_500)
    message = mock_bot.send_message.call_args.kwargs["text"]
    assert exactly_500 in message
    assert message.count("...") == 0


def test_notify_error_handles_empty_strings(notifier, mock_bot):
    """Test empty error and context strings don't cause issues."""
    notifier.notify_error(error="", context="")
    # Should not raise exceptions
    mock_bot.send_message.assert_called()
    message = mock_bot.send_message.call_args.kwargs["text"]
    # Should contain the error structure but with empty values
    assert "System Error" in message


def test_notify_error_handles_unicode(notifier, mock_bot):
    """Test truncation works with unicode characters."""
    long_unicode = "🔥" * 600  # Emoji characters
    notifier.notify_error(error=long_unicode)
    # Should not raise exceptions
    mock_bot.send_message.assert_called()


def test_notify_error_handles_unicode_at_boundary(notifier, mock_bot):
    """Test unicode characters at exact truncation boundaries don't break."""
    # Test emoji at the 250-char boundary (stack trace truncation)
    # Create string with emoji at position 249-250 to test slicing doesn't break multi-byte chars
    stack_trace = "Traceback:\n" + "a" * 238 + "🔥" + "b" * 300
    notifier.notify_error(error=stack_trace, context="boundary_test")

    # Should not raise exceptions and should successfully send
    mock_bot.send_message.assert_called()

    # Verify message is valid (no unicode errors)
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")
    assert isinstance(message, str)
    assert len(message) > 0
    assert "..." in message  # Truncation occurred


def test_notify_error_truncates_stack_traces_with_balanced_split(notifier, mock_bot):
    """Test stack traces use balanced 250/250 truncation to preserve error type at both ends."""
    # Create a realistic stack trace with 600 chars
    stack_trace = (
        "Traceback (most recent call last):\n"
        + "  File /app/src/daemon/runner.py, line 1025\n" * 10  # Repeating lines
        + "sqlalchemy.exc.OperationalError: database is locked"
    )
    # Make it longer than 500 chars
    stack_trace = stack_trace + "X" * (600 - len(stack_trace))

    notifier.notify_error(error=stack_trace, context="Test")

    mock_bot.send_message.assert_called()
    call_args = mock_bot.send_message.call_args
    message = call_args.kwargs.get("text", call_args.args[1] if len(call_args.args) > 1 else "")

    # Verify balanced truncation occurred (250/250 for stack traces)
    assert "Traceback" in message  # Should preserve start
    assert "..." in message
    # Should preserve end (last 250 chars)
    assert message.count("...") == 1
