"""
Emergency kill switch for immediate trading halt.

Activation methods:
1. File-based: Create data/KILL_SWITCH file
2. Signal-based: Send SIGUSR1 to process
3. Programmatic: Call activate() method

When active, all trading operations are blocked until manual reset.
"""

import signal
from pathlib import Path
from typing import Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class KillSwitch:
    """
    Emergency stop mechanism for the trading bot.

    The kill switch can be activated via:
    - File: Touch data/KILL_SWITCH
    - Signal: kill -SIGUSR1 <pid>
    - Code: kill_switch.activate("reason")

    Once active, trading halts until manually reset.
    """

    KILL_SWITCH_FILE = Path("data/KILL_SWITCH")

    def __init__(
        self,
        on_activate: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize kill switch.

        Args:
            on_activate: Callback function when kill switch is activated
        """
        self._manual_active = False
        self._activation_reason: Optional[str] = None
        self._on_activate = on_activate
        self._signal_registered = False

    def register_signal_handler(self) -> None:
        """Register SIGUSR1 signal handler for kill switch activation."""
        if self._signal_registered:
            return

        def handle_sigusr1(signum: int, frame) -> None:
            self.activate("SIGUSR1 signal received")

        try:
            signal.signal(signal.SIGUSR1, handle_sigusr1)
            self._signal_registered = True
            logger.info("kill_switch_signal_registered", signal="SIGUSR1")
        except (ValueError, OSError) as e:
            # Signal registration may fail in some environments
            logger.warning("kill_switch_signal_failed", error=str(e))

    @property
    def is_active(self) -> bool:
        """Check if kill switch is currently active."""
        return self._check_file_switch() or self._manual_active

    @property
    def reason(self) -> Optional[str]:
        """Get the reason for activation."""
        if self._check_file_switch():
            return "Kill switch file exists"
        return self._activation_reason

    def _check_file_switch(self) -> bool:
        """Check if kill switch file exists."""
        return self.KILL_SWITCH_FILE.exists()

    def activate(self, reason: str) -> None:
        """
        Activate the kill switch.

        Args:
            reason: Reason for activation (for logging/alerts)
        """
        if self._manual_active:
            logger.warning("kill_switch_already_active", reason=self._activation_reason)
            return

        self._manual_active = True
        self._activation_reason = reason

        logger.critical(
            "kill_switch_activated",
            reason=reason,
            file_exists=self._check_file_switch(),
        )

        # Call the activation callback (e.g., send Telegram alert)
        if self._on_activate:
            try:
                self._on_activate(reason)
            except Exception as e:
                logger.error("kill_switch_callback_failed", error=str(e))

    def reset(self, confirm_code: str = "RESET_CONFIRMED") -> bool:
        """
        Reset the kill switch (requires confirmation).

        Args:
            confirm_code: Confirmation code to prevent accidental resets

        Returns:
            True if reset successful, False otherwise
        """
        if confirm_code != "RESET_CONFIRMED":
            logger.warning("kill_switch_reset_failed", reason="Invalid confirmation code")
            return False

        if self._check_file_switch():
            logger.warning(
                "kill_switch_reset_blocked",
                reason="Kill switch file still exists. Remove data/KILL_SWITCH first.",
            )
            return False

        self._manual_active = False
        self._activation_reason = None

        logger.info("kill_switch_reset")
        return True

    def create_file_switch(self, reason: str = "Manual activation") -> None:
        """Create the kill switch file."""
        self.KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.KILL_SWITCH_FILE.write_text(f"Activated: {reason}\n")
        logger.info("kill_switch_file_created", path=str(self.KILL_SWITCH_FILE))

    def remove_file_switch(self) -> bool:
        """Remove the kill switch file."""
        if self.KILL_SWITCH_FILE.exists():
            self.KILL_SWITCH_FILE.unlink()
            logger.info("kill_switch_file_removed")
            return True
        return False

    def check_and_raise(self) -> None:
        """Check kill switch and raise exception if active."""
        if self.is_active:
            raise KillSwitchActiveError(self.reason or "Kill switch is active")


class KillSwitchActiveError(Exception):
    """Exception raised when kill switch is active."""

    pass
