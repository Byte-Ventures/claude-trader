"""
Telegram notification system.

Sends alerts for:
- Trade executions
- Order failures
- Circuit breaker events
- Kill switch activation
- Daily summaries
- System health issues
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from time import time
from typing import Optional

import structlog
from telegram import Bot
from telegram.error import TelegramError

logger = structlog.get_logger(__name__)


class TelegramNotifier:
    """
    Telegram notification system for trading alerts.

    Setup:
    1. Message @BotFather on Telegram
    2. Create new bot with /newbot
    3. Copy the bot token
    4. Message @userinfobot to get your chat_id
    5. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._bot: Optional[Bot] = None
        self._last_message_time: float = 0.0
        self._min_interval: float = 1.0  # Minimum seconds between messages

        if enabled and bot_token and chat_id:
            self._bot = Bot(token=bot_token)
            logger.info("telegram_notifier_initialized")
        else:
            logger.warning("telegram_notifier_disabled")

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.

        Args:
            message: Message text
            parse_mode: HTML or Markdown

        Returns:
            True if sent successfully
        """
        if not self.enabled or not self._bot:
            logger.debug("telegram_message_skipped", reason="disabled")
            return False

        # Rate limiting - skip message if too soon (non-blocking)
        now = time()
        elapsed = now - self._last_message_time
        if elapsed < self._min_interval:
            logger.debug("telegram_rate_limited", skipped=True, wait_time=self._min_interval - elapsed)
            return False
        self._last_message_time = now

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            return True
        except TelegramError as e:
            logger.error("telegram_send_failed", error=str(e))
            return False

    def send_message_sync(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message synchronously (for non-async contexts).

        Args:
            message: Message text
            parse_mode: HTML or Markdown

        Returns:
            True if sent successfully
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task if already in async context
                asyncio.create_task(self.send_message(message, parse_mode))
                return True
            else:
                return loop.run_until_complete(self.send_message(message, parse_mode))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.send_message(message, parse_mode))

    # Convenience methods for common notifications

    def notify_trade(
        self,
        side: str,
        size: Decimal,
        price: Decimal,
        fee: Decimal,
        is_paper: bool = False,
    ) -> None:
        """Send notification for trade execution."""
        mode = "[PAPER] " if is_paper else ""
        emoji = "🟢" if side == "buy" else "🔴"

        message = (
            f"{emoji} <b>{mode}Trade Executed</b>\n\n"
            f"Side: {side.upper()}\n"
            f"Size: {size:.8f} BTC\n"
            f"Price: ${price:,.2f}\n"
            f"Fee: ${fee:.2f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)

    def notify_order_failed(
        self,
        side: str,
        size: Decimal,
        error: str,
    ) -> None:
        """Send notification for failed order."""
        message = (
            f"⚠️ <b>Order Failed</b>\n\n"
            f"Side: {side.upper()}\n"
            f"Size: {size:.8f} BTC\n"
            f"Error: {error}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)

    def notify_circuit_breaker(
        self,
        level: str,
        reason: str,
    ) -> None:
        """Send notification for circuit breaker event."""
        emoji = {
            "yellow": "🟡",
            "red": "🔴",
            "black": "⚫",
        }.get(level, "⚪")

        message = (
            f"{emoji} <b>Circuit Breaker: {level.upper()}</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Trading has been {'paused' if level != 'black' else 'HALTED (manual reset required)'}."
        )

        self.send_message_sync(message)

    def notify_kill_switch(self, reason: str) -> None:
        """Send notification for kill switch activation."""
        message = (
            f"🚨 <b>KILL SWITCH ACTIVATED</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"⚠️ ALL TRADING HALTED\n"
            f"Manual reset required to resume."
        )

        self.send_message_sync(message)

    def notify_loss_limit(self, limit_type: str, loss_percent: float) -> None:
        """Send notification for loss limit hit."""
        message = (
            f"🛑 <b>{limit_type.title()} Loss Limit Hit</b>\n\n"
            f"Loss: {loss_percent:.1f}%\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Trading paused until limit resets."
        )

        self.send_message_sync(message)

    def notify_daily_summary(
        self,
        starting_balance: Decimal,
        ending_balance: Decimal,
        realized_pnl: Decimal,
        total_trades: int,
        is_paper: bool = False,
    ) -> None:
        """Send daily trading summary."""
        mode = "[PAPER] " if is_paper else ""
        pnl_percent = (realized_pnl / starting_balance * 100) if starting_balance > 0 else Decimal("0")
        pnl_emoji = "📈" if realized_pnl >= 0 else "📉"

        message = (
            f"📊 <b>{mode}Daily Summary</b>\n\n"
            f"Starting Balance: ${starting_balance:,.2f}\n"
            f"Ending Balance: ${ending_balance:,.2f}\n"
            f"{pnl_emoji} P&L: ${realized_pnl:+,.2f} ({pnl_percent:+.1f}%)\n"
            f"Total Trades: {total_trades}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}"
        )

        self.send_message_sync(message)

    def notify_startup(self, mode: str, balance: Decimal, exchange: str = "Coinbase") -> None:
        """Send notification on bot startup."""
        message = (
            f"🤖 <b>Trading Bot Started</b>\n\n"
            f"Exchange: {exchange}\n"
            f"Mode: {mode.upper()}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)

    def notify_shutdown(self, reason: str = "Manual shutdown") -> None:
        """Send notification on bot shutdown."""
        message = (
            f"🔌 <b>Trading Bot Stopped</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)

    def notify_error(self, error: str, context: str = "") -> None:
        """Send notification for system error."""
        message = (
            f"❌ <b>System Error</b>\n\n"
            f"Error: {error}\n"
            f"Context: {context}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)
