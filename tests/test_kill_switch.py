"""
Comprehensive tests for the KillSwitch emergency stop system.

Tests cover:
- File-based activation (data/KILL_SWITCH file)
- Manual programmatic activation
- Signal-based activation (SIGUSR1)
- Reset with confirmation code
- Callback execution on activation
- State persistence
- Edge cases and error handling
"""

import pytest
import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.safety.kill_switch import (
    KillSwitch,
    KillSwitchActiveError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_kill_switch_file(tmp_path):
    """Temporary kill switch file path."""
    return tmp_path / "KILL_SWITCH"


@pytest.fixture
def kill_switch(temp_kill_switch_file, monkeypatch):
    """Kill switch with temporary file location."""
    # Override the KILL_SWITCH_FILE path
    monkeypatch.setattr(KillSwitch, "KILL_SWITCH_FILE", temp_kill_switch_file)
    return KillSwitch()


@pytest.fixture
def callback_mock():
    """Mock callback for activation notifications."""
    return MagicMock()


@pytest.fixture
def kill_switch_with_callback(temp_kill_switch_file, monkeypatch, callback_mock):
    """Kill switch with callback attached."""
    monkeypatch.setattr(KillSwitch, "KILL_SWITCH_FILE", temp_kill_switch_file)
    return KillSwitch(on_activate=callback_mock)


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization_default():
    """Test kill switch initializes in inactive state."""
    ks = KillSwitch()
    assert ks.is_active is False
    assert ks._manual_active is False
    assert ks._activation_reason is None
    assert ks._signal_registered is False


def test_initialization_with_callback(callback_mock):
    """Test kill switch accepts callback."""
    ks = KillSwitch(on_activate=callback_mock)
    assert ks._on_activate is callback_mock


# ============================================================================
# Manual Activation Tests
# ============================================================================

def test_manual_activation(kill_switch):
    """Test manual activation via activate() method."""
    kill_switch.activate("Manual test activation")

    assert kill_switch.is_active is True
    assert kill_switch.reason == "Manual test activation"


def test_activation_sets_manual_flag(kill_switch):
    """Test activation sets internal manual active flag."""
    kill_switch.activate("Test")

    assert kill_switch._manual_active is True
    assert kill_switch._activation_reason == "Test"


def test_activation_callback_called(kill_switch_with_callback, callback_mock):
    """Test callback is called when kill switch activated."""
    kill_switch_with_callback.activate("Emergency stop")

    callback_mock.assert_called_once_with("Emergency stop")


def test_activation_callback_exception_caught(kill_switch_with_callback, callback_mock):
    """Test callback exceptions are caught and don't crash."""
    callback_mock.side_effect = Exception("Callback failed!")

    # Should not raise
    kill_switch_with_callback.activate("Test")

    assert kill_switch_with_callback.is_active is True


def test_activate_when_already_active(kill_switch):
    """Test activating when already active doesn't change state."""
    kill_switch.activate("First reason")
    kill_switch.activate("Second reason")

    # Should keep first reason
    assert kill_switch.reason == "First reason"


# ============================================================================
# File-Based Activation Tests
# ============================================================================

def test_file_activation_creates_file(kill_switch):
    """Test creating kill switch file activates the switch."""
    kill_switch.create_file_switch("Test file activation")

    assert kill_switch.is_active is True
    assert kill_switch.KILL_SWITCH_FILE.exists()


def test_file_activation_writes_reason(kill_switch):
    """Test kill switch file contains activation reason."""
    kill_switch.create_file_switch("Emergency: Market crash")

    content = kill_switch.KILL_SWITCH_FILE.read_text()
    assert "Emergency: Market crash" in content


def test_file_exists_activates_switch(kill_switch):
    """Test existing file activates kill switch."""
    # Create file directly
    kill_switch.KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    kill_switch.KILL_SWITCH_FILE.write_text("Manual activation\n")

    assert kill_switch.is_active is True
    assert kill_switch.reason == "Kill switch file exists"


def test_file_removal_deactivates_file_switch(kill_switch):
    """Test removing file deactivates file-based switch."""
    kill_switch.create_file_switch("Test")

    result = kill_switch.remove_file_switch()

    assert result is True
    assert not kill_switch.KILL_SWITCH_FILE.exists()
    assert kill_switch.is_active is False  # Manual flag not set


def test_remove_file_when_no_file_exists(kill_switch):
    """Test removing non-existent file returns False."""
    result = kill_switch.remove_file_switch()

    assert result is False


def test_file_and_manual_both_active(kill_switch):
    """Test kill switch active if either file or manual is active."""
    kill_switch.activate("Manual")
    kill_switch.create_file_switch("File")

    assert kill_switch.is_active is True

    # Remove file, should still be active due to manual
    kill_switch.remove_file_switch()
    assert kill_switch.is_active is True


def test_is_active_checks_file_every_time(kill_switch):
    """Test is_active property checks file status each time."""
    assert kill_switch.is_active is False

    # Create file externally
    kill_switch.KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    kill_switch.KILL_SWITCH_FILE.write_text("External activation\n")

    # Should detect file now
    assert kill_switch.is_active is True


# ============================================================================
# Signal Handler Tests
# ============================================================================

def test_signal_handler_registration(kill_switch):
    """Test SIGUSR1 signal handler can be registered."""
    kill_switch.register_signal_handler()

    assert kill_switch._signal_registered is True


def test_signal_handler_idempotent(kill_switch):
    """Test registering signal handler multiple times is safe."""
    kill_switch.register_signal_handler()
    kill_switch.register_signal_handler()

    assert kill_switch._signal_registered is True


@patch('signal.signal')
def test_signal_handler_activates_kill_switch(mock_signal, kill_switch):
    """Test SIGUSR1 signal activates kill switch."""
    kill_switch.register_signal_handler()

    # Get the registered handler
    assert mock_signal.called
    handler = mock_signal.call_args[0][1]

    # Simulate signal
    handler(signal.SIGUSR1, None)

    assert kill_switch.is_active is True
    assert "SIGUSR1" in kill_switch.reason


@patch('signal.signal')
def test_signal_registration_failure_handled(mock_signal, kill_switch):
    """Test signal registration failures are handled gracefully."""
    mock_signal.side_effect = OSError("Signal not supported")

    # Should not raise
    kill_switch.register_signal_handler()

    assert kill_switch._signal_registered is False


# ============================================================================
# Reset Tests
# ============================================================================

def test_reset_with_valid_code(kill_switch):
    """Test reset succeeds with valid confirmation code."""
    kill_switch.activate("Test")

    result = kill_switch.reset("RESET_CONFIRMED")

    assert result is True
    assert kill_switch.is_active is False
    assert kill_switch._manual_active is False
    assert kill_switch._activation_reason is None


def test_reset_with_invalid_code(kill_switch):
    """Test reset fails with invalid confirmation code."""
    kill_switch.activate("Test")

    result = kill_switch.reset("WRONG_CODE")

    assert result is False
    assert kill_switch.is_active is True


def test_reset_blocked_when_file_exists(kill_switch):
    """Test reset is blocked if kill switch file still exists."""
    kill_switch.activate("Manual")
    kill_switch.create_file_switch("File")

    result = kill_switch.reset("RESET_CONFIRMED")

    assert result is False
    assert kill_switch.is_active is True


def test_reset_clears_reason(kill_switch):
    """Test reset clears activation reason."""
    kill_switch.activate("Emergency stop")
    kill_switch.reset("RESET_CONFIRMED")

    assert kill_switch.reason is None


def test_reset_only_clears_manual_not_file(kill_switch):
    """Test reset only clears manual activation, not file."""
    kill_switch.activate("Manual")
    kill_switch.create_file_switch("File")
    kill_switch.remove_file_switch()

    # Reset manual
    result = kill_switch.reset("RESET_CONFIRMED")

    assert result is True
    assert kill_switch.is_active is False


# ============================================================================
# State Property Tests
# ============================================================================

def test_reason_property_when_manual(kill_switch):
    """Test reason property returns manual activation reason."""
    kill_switch.activate("Manual emergency stop")

    assert kill_switch.reason == "Manual emergency stop"


def test_reason_property_when_file(kill_switch):
    """Test reason property when file-based activation."""
    kill_switch.create_file_switch("File activation")

    assert kill_switch.reason == "Kill switch file exists"


def test_reason_property_file_takes_precedence(kill_switch):
    """Test file activation reason takes precedence over manual."""
    kill_switch.activate("Manual")
    kill_switch.create_file_switch("File")

    # File reason should be shown
    assert kill_switch.reason == "Kill switch file exists"


def test_reason_when_not_active(kill_switch):
    """Test reason is None when kill switch not active."""
    assert kill_switch.reason is None


# ============================================================================
# Check and Raise Tests
# ============================================================================

def test_check_and_raise_throws_when_active(kill_switch):
    """Test check_and_raise throws exception when active."""
    kill_switch.activate("Emergency")

    with pytest.raises(KillSwitchActiveError) as exc_info:
        kill_switch.check_and_raise()

    assert "Emergency" in str(exc_info.value)


def test_check_and_raise_passes_when_inactive(kill_switch):
    """Test check_and_raise doesn't throw when inactive."""
    # Should not raise
    kill_switch.check_and_raise()


def test_check_and_raise_with_file_activation(kill_switch):
    """Test check_and_raise with file-based activation."""
    kill_switch.create_file_switch("Test")

    with pytest.raises(KillSwitchActiveError) as exc_info:
        kill_switch.check_and_raise()

    assert "Kill switch file exists" in str(exc_info.value)


# ============================================================================
# Edge Cases & Error Handling Tests
# ============================================================================

def test_file_parent_directory_created(temp_kill_switch_file, monkeypatch):
    """Test parent directory is created if it doesn't exist."""
    # Use a nested path that doesn't exist
    nested_path = temp_kill_switch_file.parent / "nested" / "path" / "KILL_SWITCH"
    monkeypatch.setattr(KillSwitch, "KILL_SWITCH_FILE", nested_path)

    ks = KillSwitch()
    ks.create_file_switch("Test")

    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_concurrent_activation_safe(kill_switch):
    """Test multiple activations don't corrupt state."""
    kill_switch.activate("First")
    kill_switch.activate("Second")
    kill_switch.activate("Third")

    assert kill_switch.is_active is True
    assert kill_switch.reason == "First"  # Keeps first reason


def test_activation_after_reset(kill_switch):
    """Test kill switch can be reactivated after reset."""
    kill_switch.activate("First")
    kill_switch.reset("RESET_CONFIRMED")
    kill_switch.activate("Second")

    assert kill_switch.is_active is True
    assert kill_switch.reason == "Second"


def test_file_content_format(kill_switch):
    """Test kill switch file has expected format."""
    reason = "Market crash detected"
    kill_switch.create_file_switch(reason)

    content = kill_switch.KILL_SWITCH_FILE.read_text()
    assert "Activated:" in content
    assert reason in content


def test_multiple_callbacks_not_called_on_reactivation(kill_switch_with_callback, callback_mock):
    """Test callback only called on first activation."""
    kill_switch_with_callback.activate("First")
    callback_mock.reset_mock()

    kill_switch_with_callback.activate("Second")

    callback_mock.assert_not_called()


def test_is_active_without_file_or_manual(kill_switch):
    """Test is_active is False when neither file nor manual active."""
    assert kill_switch.is_active is False
    assert kill_switch._manual_active is False
    assert not kill_switch.KILL_SWITCH_FILE.exists()
