"""
Telegram notification system.

Sends alerts for:
- Trade executions
- Order failures
- Circuit breaker events
- Kill switch activation
- Daily summaries
- System health issues

Features:
- Rate limiting to prevent spam
- Message deduplication (no repeated messages within cooldown)
- Async message sending with sync wrapper
"""

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from decimal import Decimal
from time import time
from typing import Optional, TYPE_CHECKING

import structlog
from telegram import Bot
from telegram.error import TelegramError

if TYPE_CHECKING:
    from src.safety.circuit_breaker import CircuitBreaker
    from src.safety.kill_switch import KillSwitch

logger = structlog.get_logger(__name__)


# Default cooldown periods for different message types (seconds)
# These prevent spam - same message won't repeat within cooldown period
DEFAULT_COOLDOWNS = {
    "circuit_breaker": 3600,  # 1 hour - don't repeat same breaker status
    "loss_limit": 3600,       # 1 hour
    "error": 1800,            # 30 minutes - same error won't repeat
    "order_failed": 1800,     # 30 minutes
    "trade_rejected": 0,      # Always send - must notify user of rejected trades
    "regime_change": 0,       # Always send (only fires on actual change)
    "weight_profile": 900,    # 15 minutes - prevent oscillation spam
    "trade": 0,               # Always send trade notifications
    "trade_review": 0,        # Always send actual trade reviews
    "interesting_hold": 1800, # 30 minutes - throttle repetitive hold alerts
    "daily_summary": 0,       # Always send summaries
    "startup": 0,             # Always send
    "shutdown": 0,            # Always send
    "kill_switch": 0,         # Always send (critical)
    "market_analysis": 0,     # Always send (fires once per hour)
}


class TelegramNotifier:
    """
    Telegram notification system for trading alerts.

    Setup:
    1. Message @BotFather on Telegram
    2. Create new bot with /newbot
    3. Copy the bot token
    4. Message @userinfobot to get your chat_id
    5. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env

    Features:
    - Rate limiting between messages
    - Deduplication of repeated messages
    - Configurable cooldown per message type
    """

    # Maximum length for error/context messages in notifications
    # Balance completeness (preserves most error details) vs actionability
    # (fits on one screen, easier to parse in mobile app)
    # Telegram API limit is 4096, but shorter messages are more actionable
    MAX_ERROR_MSG_LENGTH = 500

    # Ellipsis string and length for truncation operations
    ELLIPSIS = "..."
    ELLIPSIS_LEN = len(ELLIPSIS)  # 3 characters

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        db=None,
        is_paper: bool = False,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
            enabled: Whether notifications are enabled
            db: Optional database instance for saving notifications to dashboard
            is_paper: Whether in paper trading mode
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._db = db
        self._is_paper = is_paper
        self._bot: Optional[Bot] = None
        self._last_message_time: float = 0.0
        self._min_interval: float = 1.0  # Minimum seconds between messages

        # Deduplication: track last message time by type
        self._last_message_by_type: dict[str, float] = {}
        # Track last message content hash to detect duplicates
        self._last_message_hash: dict[str, str] = {}

        # Safety system references for command handling
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        self._kill_switch: Optional["KillSwitch"] = None

        # Command handling state
        self._last_update_id: int = 0
        self._command_check_interval: float = 10.0  # Check every 10 seconds
        self._last_command_check: float = 0.0

        if enabled and bot_token and chat_id:
            self._bot = Bot(token=bot_token)
            logger.info("telegram_notifier_initialized")
        else:
            logger.warning("telegram_notifier_disabled")

    def _save_to_dashboard(self, msg_type: str, title: str, message: str) -> None:
        """Save notification to database for dashboard display."""
        if self._db:
            try:
                self._db.save_notification(
                    type=msg_type,
                    title=title,
                    message=message,
                    is_paper=self._is_paper,
                )
            except Exception as e:
                logger.error("notification_save_failed", error=str(e))

    def _should_send(self, msg_type: str, message: str) -> bool:
        """
        Check if we should send this message based on deduplication rules.

        Args:
            msg_type: Type of message (circuit_breaker, error, trade, etc.)
            message: The message content

        Returns:
            True if message should be sent
        """
        now = time()
        cooldown = DEFAULT_COOLDOWNS.get(msg_type, 60)

        # Always send if no cooldown
        if cooldown == 0:
            return True

        # Check if we're within cooldown for this message type
        last_time = self._last_message_by_type.get(msg_type, 0)
        if now - last_time < cooldown:
            # Within cooldown - check if message is different
            msg_hash = hashlib.md5(message.encode()).hexdigest()[:8]
            last_hash = self._last_message_hash.get(msg_type, "")

            if msg_hash == last_hash:
                # Same message within cooldown - skip
                logger.debug(
                    "telegram_message_deduplicated",
                    msg_type=msg_type,
                    cooldown_remaining=int(cooldown - (now - last_time)),
                )
                return False

        return True

    def _record_sent(self, msg_type: str, message: str) -> None:
        """Record that a message was sent for deduplication."""
        self._last_message_by_type[msg_type] = time()
        self._last_message_hash[msg_type] = hashlib.md5(message.encode()).hexdigest()[:8]

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
        Send a message synchronously (blocking).

        Args:
            message: Message text
            parse_mode: HTML or Markdown

        Returns:
            True if sent successfully
        """
        import concurrent.futures

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In async context - run in thread pool to avoid blocking
                # and actually wait for result (previous code used create_task
                # which never awaited, so messages were never sent!)
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.send_message(message, parse_mode))
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.send_message(message, parse_mode))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.send_message(message, parse_mode))
        except concurrent.futures.TimeoutError:
            logger.warning("telegram_send_timeout", timeout=30)
            return False
        except Exception as e:
            logger.error("telegram_send_error", error=str(e))
            return False

    # Command handling methods

    def set_safety_systems(
        self,
        circuit_breaker: "CircuitBreaker",
        kill_switch: "KillSwitch",
    ) -> None:
        """
        Set references to safety systems for command handling.

        Args:
            circuit_breaker: Circuit breaker instance
            kill_switch: Kill switch instance
        """
        self._circuit_breaker = circuit_breaker
        self._kill_switch = kill_switch
        logger.info("telegram_command_handling_enabled")

    def check_commands(self, loop: asyncio.AbstractEventLoop = None) -> None:
        """
        Check for and process incoming Telegram commands.

        Call this periodically from the main loop. It's rate-limited internally.

        Args:
            loop: Optional event loop to use. If provided, uses run_until_complete()
                  instead of creating a new loop with asyncio.run(). This is more
                  efficient and avoids potential conflicts with existing event loops.
        """
        if not self.enabled or not self._bot:
            return

        # Rate limit command checks
        now = time()
        if now - self._last_command_check < self._command_check_interval:
            return
        self._last_command_check = now

        try:
            if loop:
                # Use provided event loop (more efficient, avoids conflicts)
                loop.run_until_complete(self._async_check_commands())
            else:
                # Fallback: create new event loop (less efficient)
                asyncio.run(self._async_check_commands())
        except Exception as e:
            logger.debug("telegram_command_check_error", error=str(e))

    async def _async_check_commands(self) -> None:
        """Async implementation of command checking."""
        try:
            updates = await self._bot.get_updates(
                offset=self._last_update_id + 1,
                timeout=1,
                allowed_updates=["message"],
            )

            for update in updates:
                self._last_update_id = update.update_id

                if update.message and update.message.text:
                    # Only process commands from authorized chat
                    if str(update.message.chat_id) != self.chat_id:
                        logger.warning(
                            "telegram_unauthorized_command",
                            chat_id=update.message.chat_id,
                        )
                        continue

                    text = update.message.text.strip()
                    await self._handle_command(text)

        except TelegramError as e:
            logger.debug("telegram_get_updates_error", error=str(e))

    async def _handle_command(self, text: str) -> None:
        """Handle a single command."""
        if text == "/reset":
            await self._cmd_reset()
        elif text == "/status":
            await self._cmd_status()
        elif text == "/help":
            await self._cmd_help()
        # Ignore non-commands

    async def _cmd_reset(self) -> None:
        """Handle /reset command - reset circuit breaker from BLACK state."""
        if not self._circuit_breaker:
            await self.send_message("❌ Circuit breaker not available")
            return

        from src.safety.circuit_breaker import BreakerLevel

        status = self._circuit_breaker.status
        if status.level == BreakerLevel.BLACK:
            success = self._circuit_breaker.manual_reset("RESET_CONFIRMED")
            if success:
                await self.send_message(
                    "✅ <b>Circuit Breaker Reset</b>\n\n"
                    "Status: BLACK → GREEN\n"
                    "Trading has resumed."
                )
                logger.info("circuit_breaker_reset_via_telegram")
            else:
                await self.send_message("❌ Reset failed - check logs")
        elif status.level == BreakerLevel.RED:
            await self.send_message(
                f"⚠️ Circuit breaker is RED (not BLACK)\n\n"
                f"Reason: {status.reason}\n"
                f"Auto-recovery: {status.cooldown_until.strftime('%H:%M:%S') if status.cooldown_until else 'N/A'}\n\n"
                "RED state auto-recovers. Use /reset only for BLACK state."
            )
        else:
            await self.send_message(
                f"ℹ️ Circuit breaker is {status.level.name}\n\n"
                "No reset needed - trading is active."
            )

    async def _cmd_status(self) -> None:
        """Handle /status command - show current system status."""
        lines = ["📊 <b>System Status</b>\n"]

        # Circuit breaker status
        if self._circuit_breaker:
            from src.safety.circuit_breaker import BreakerLevel
            status = self._circuit_breaker.status
            level_emoji = {
                BreakerLevel.GREEN: "🟢",
                BreakerLevel.YELLOW: "🟡",
                BreakerLevel.RED: "🔴",
                BreakerLevel.BLACK: "⚫",
            }
            lines.append(
                f"Circuit Breaker: {level_emoji.get(status.level, '⚪')} {status.level.name}"
            )
            if status.reason:
                lines.append(f"  └ {status.reason}")
            if status.cooldown_until:
                lines.append(f"  └ Recovers: {status.cooldown_until.strftime('%H:%M:%S')}")

        # Kill switch status
        if self._kill_switch:
            if self._kill_switch.is_active:
                lines.append(f"Kill Switch: 🛑 ACTIVE - {self._kill_switch.reason}")
            else:
                lines.append("Kill Switch: ✅ Inactive")

        # Trading mode
        mode = "📝 PAPER" if self._is_paper else "💰 LIVE"
        lines.append(f"Mode: {mode}")

        await self.send_message("\n".join(lines))

    async def _cmd_help(self) -> None:
        """Handle /help command - show available commands."""
        await self.send_message(
            "🤖 <b>Trading Bot Commands</b>\n\n"
            "/status - Show system status\n"
            "/reset - Reset circuit breaker (BLACK → GREEN)\n"
            "/help - Show this message"
        )

    # Convenience methods for common notifications

    def notify_trade(
        self,
        side: str,
        size: Decimal,
        price: Decimal,
        fee: Decimal,
        is_paper: bool = False,
        signal_score: Optional[int] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        position_percent: Optional[float] = None,
        realized_pnl: Optional[Decimal] = None,
        entry_price: Optional[Decimal] = None,
    ) -> None:
        """Send notification for trade execution."""
        mode = "[PAPER] " if is_paper else ""
        emoji = "🟢" if side == "buy" else "🔴"

        # Build message with required fields
        lines = [
            f"{emoji} <b>{mode}Trade Executed</b>\n",
            f"Side: {side.upper()}",
            f"Size: {size:.8f}",
            f"Price: ¤{price:,.2f}",
            f"Fee: ¤{fee:.2f}",
        ]

        # Add optional context
        if signal_score is not None:
            lines.append(f"Signal Score: {signal_score}")
        if stop_loss is not None and side == "buy":
            stop_pct = ((price - stop_loss) / price * 100) if price > 0 else 0
            lines.append(f"Stop Loss: ¤{stop_loss:,.2f} ({stop_pct:.1f}% below)")
        if take_profit is not None and side == "buy":
            tp_pct = ((take_profit - price) / price * 100) if price > 0 else 0
            lines.append(f"Take Profit: ¤{take_profit:,.2f} ({tp_pct:.1f}% above)")
        if position_percent is not None:
            lines.append(f"Position Size: {position_percent:.1f}% of portfolio")
        if realized_pnl is not None and side == "sell":
            pnl_emoji = "📈" if realized_pnl >= 0 else "📉"
            # Show entry price, exit price, and P&L percentage for sells
            if entry_price is not None and entry_price > 0:
                pnl_pct = ((price - entry_price) / entry_price * 100)
                lines.append(f"Entry Price: ¤{entry_price:,.2f}")
                lines.append(f"Exit Price: ¤{price:,.2f}")
                lines.append(f"{pnl_emoji} P&L: ¤{realized_pnl:+,.2f} ({pnl_pct:+.2f}%)")
            else:
                # Fallback if entry price not available
                lines.append(f"{pnl_emoji} Realized P&L: ¤{realized_pnl:+,.2f}")

        lines.append(f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

        message = "\n".join(lines)
        if self.send_message_sync(message):
            logger.info("trade_notification_sent", side=side, size=str(size), price=str(price))
        self._save_to_dashboard("trade", f"{side.upper()} {size:.6f}", message)

    def notify_trade_rejected(
        self,
        side: str,
        reason: str,
        price: Optional[Decimal] = None,
        signal_score: Optional[int] = None,
        size_quote: Optional[Decimal] = None,
        is_paper: bool = False,
    ) -> None:
        """Send notification when a trade is rejected by validation."""
        # Input validation
        if side not in ("buy", "sell"):
            logger.error("invalid_trade_side", side=side)
            return
        if signal_score is not None and not (-100 <= signal_score <= 100):
            logger.warning("signal_score_out_of_range", score=signal_score)
        if price is not None and price <= 0:
            logger.error("invalid_price", price=str(price))
            return

        mode = "[PAPER] " if is_paper else ""
        emoji = "⛔"

        lines = [
            f"{emoji} <b>{mode}Trade Rejected</b>\n",
            f"Side: {side.upper()}",
            f"Reason: {reason}",
        ]

        if price is not None:
            lines.append(f"Price: ¤{price:,.2f}")
        if signal_score is not None:
            lines.append(f"Signal Score: {signal_score}")
        if size_quote is not None:
            lines.append(f"Intended Size: ¤{size_quote:,.2f}")

        lines.append(f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

        message = "\n".join(lines)

        if self.send_message_sync(message):
            logger.info("trade_rejected_notification_sent", side=side, reason=reason)
        self._save_to_dashboard("trade_rejected", f"{side.upper()} Rejected", message)

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
            f"Size: {size:.8f}\n"
            f"Error: {error}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Deduplicate order failed messages (same error)
        if not self._should_send("order_failed", error):
            return

        if self.send_message_sync(message):
            self._record_sent("order_failed", error)
        self._save_to_dashboard("order_failed", f"Order Failed: {side.upper()}", message)

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
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Trading has been {'paused' if level != 'black' else 'HALTED (manual reset required)'}."
        )

        # Deduplicate circuit breaker messages
        if not self._should_send("circuit_breaker", f"{level}:{reason}"):
            return

        if self.send_message_sync(message):
            self._record_sent("circuit_breaker", f"{level}:{reason}")
        self._save_to_dashboard("circuit_breaker", f"Circuit Breaker: {level.upper()}", message)

    def notify_kill_switch(self, reason: str) -> None:
        """Send notification for kill switch activation."""
        message = (
            f"🚨 <b>KILL SWITCH ACTIVATED</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"⚠️ ALL TRADING HALTED\n"
            f"Manual reset required to resume."
        )

        self.send_message_sync(message)
        self._save_to_dashboard("kill_switch", "Kill Switch Activated", message)

    def notify_loss_limit(self, limit_type: str, loss_percent: float) -> None:
        """Send notification for loss limit hit."""
        message = (
            f"🛑 <b>{limit_type.title()} Loss Limit Hit</b>\n\n"
            f"Loss: {loss_percent:.1f}%\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Trading paused until limit resets."
        )

        # Deduplicate loss limit messages
        dedup_key = f"{limit_type}:{int(loss_percent)}"
        if not self._should_send("loss_limit", dedup_key):
            return

        if self.send_message_sync(message):
            self._record_sent("loss_limit", dedup_key)
        self._save_to_dashboard(
            "loss_limit",
            f"{limit_type.title()} Loss Limit",
            message,
        )

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
            f"Starting Balance: ¤{starting_balance:,.2f}\n"
            f"Ending Balance: ¤{ending_balance:,.2f}\n"
            f"{pnl_emoji} P&L: ¤{realized_pnl:+,.2f} ({pnl_percent:+.1f}%)\n"
            f"Total Trades: {total_trades}\n"
            f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        )

        self.send_message_sync(message)
        self._save_to_dashboard("daily_summary", f"{mode}Daily Summary", message)

    def notify_startup(self, mode: str, balance: Decimal, exchange: str = "Coinbase") -> None:
        """Send notification on bot startup."""
        display_mode = "PAPER" if mode.lower() == "paper" else mode.upper()
        message = (
            f"🤖 <b>Trading Bot Started</b>\n\n"
            f"Exchange: {exchange}\n"
            f"Mode: {display_mode}\n"
            f"Balance: ¤{balance:,.2f}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)
        self._save_to_dashboard("startup", f"Bot Started ({display_mode})", message)

    def notify_shutdown(self, reason: str = "Manual shutdown") -> None:
        """Send notification on bot shutdown."""
        message = (
            f"🔌 <b>Trading Bot Stopped</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)
        self._save_to_dashboard("shutdown", "Bot Stopped", message)

    def _safe_truncate(self, text: str, max_len: int, from_end: bool = False) -> str:
        """
        Safely truncate text without breaking unicode characters.

        Args:
            text: Text to truncate
            max_len: Maximum length after truncation
            from_end: If True, take from end; if False, take from start

        Returns:
            Truncated text with unicode characters intact
        """
        if from_end:
            # Truncate from end
            candidate = text[-max_len:]
            # Encode and decode to clean up any broken unicode characters at boundaries
            # errors='ignore' removes broken characters, then decode back to string
            return candidate.encode('utf-8', errors='ignore').decode('utf-8')
        else:
            # Truncate from start
            candidate = text[:max_len]
            # Encode and decode to clean up any broken unicode characters at boundaries
            return candidate.encode('utf-8', errors='ignore').decode('utf-8')

    def notify_error(self, error: str, context: str = "") -> None:
        """Send notification for system error."""
        MAX_LEN = self.MAX_ERROR_MSG_LENGTH

        # Calculate dedup key BEFORE truncation using full error text.
        # This ensures deduplication is based on the complete error, preventing
        # false collisions when different errors have identical truncated forms.
        # Example: Two 600-char errors with different endings would both truncate
        # to the same first 500 chars, but have different dedup keys based on full text.
        dedup_key = f"{error}:{context}"

        if len(error) > MAX_LEN:
            # Log full error for debugging before truncation
            logger.debug(f"Truncating long error message ({len(error)} chars)",
                        error_preview=error[:100])

            # Early exit check - if neither keyword present, skip expensive regex
            is_stack_trace = False
            if 'Traceback' in error or 'File "' in error or ' at ' in error:
                # For stack traces, prioritize start + end (error type at both locations)
                # For other errors, keep first 400 + last 100 to preserve error message
                # Improved detection to handle edge cases:
                # - Multiple consecutive frames indicate stack trace
                # - Partial stack traces (middle section only)
                # - Python tracebacks and JavaScript stack traces
                # Use regex for flexible indentation matching (handles varying spaces/tabs)
                is_stack_trace = (
                    len(re.findall(r'\n\s+File "', error)) >= 2 or  # Multiple Python frames (flexible indentation)
                    len(re.findall(r'\n\s+at ', error)) >= 2 or      # Multiple JavaScript frames (flexible indentation)
                    ('Traceback' in error and 'File "' in error)  # Single frame Python
                )
            if is_stack_trace:
                error = self._safe_truncate(error, 250) + self.ELLIPSIS + self._safe_truncate(error, 250, from_end=True)
            else:
                error = self._safe_truncate(error, 400) + self.ELLIPSIS + self._safe_truncate(error, 100, from_end=True)

        if len(context) > MAX_LEN:
            # Log full context for debugging before truncation
            logger.debug(f"Truncating long context message ({len(context)} chars)",
                        context_preview=context[:100])

            # Early exit check - if neither keyword present, skip expensive regex
            is_context_stack_trace = False
            if 'Traceback' in context or 'File "' in context or ' at ' in context:
                # Apply same smart truncation to context - balanced split for stack traces,
                # preserve beginning for regular text (usually more relevant)
                is_context_stack_trace = (
                    len(re.findall(r'\n\s+File "', context)) >= 2 or  # Multiple Python frames (flexible indentation)
                    len(re.findall(r'\n\s+at ', context)) >= 2 or      # Multiple JavaScript frames (flexible indentation)
                    ('Traceback' in context and 'File "' in context)  # Single frame Python
                )
            if is_context_stack_trace:
                context = self._safe_truncate(context, 250) + self.ELLIPSIS + self._safe_truncate(context, 250, from_end=True)
            else:
                context = self._safe_truncate(context, 400) + self.ELLIPSIS + self._safe_truncate(context, 100, from_end=True)

        message = (
            f"❌ <b>System Error</b>\n\n"
            f"Error: {error}\n"
            f"Context: {context}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Validate total message length against Telegram's 4096 char limit
        # This is a defense-in-depth check to ensure message formatting changes
        # don't accidentally exceed the limit
        TELEGRAM_MAX_LENGTH = 4096
        if len(message) > TELEGRAM_MAX_LENGTH:
            logger.error(
                f"Message exceeds Telegram limit ({len(message)} chars), "
                f"truncating more aggressively"
            )
            # Calculate overhead (header + footer + formatting)
            overhead = len(message) - len(error) - len(context)
            # Split remaining budget equally between error and context
            # Ensure minimum 10 chars per field for defensive programming
            per_field_budget = max(10, (TELEGRAM_MAX_LENGTH - overhead) // 2)

            # Re-truncate with aggressive limits using safe truncation
            if len(error) > per_field_budget:
                # Remove existing ellipsis if present to avoid "..."..."
                error_clean = error[:-self.ELLIPSIS_LEN] if error.endswith(self.ELLIPSIS) else error
                error = self._safe_truncate(error_clean, per_field_budget - self.ELLIPSIS_LEN) + self.ELLIPSIS
            if len(context) > per_field_budget:
                # Remove existing ellipsis if present to avoid "..."..."
                context_clean = context[:-self.ELLIPSIS_LEN] if context.endswith(self.ELLIPSIS) else context
                context = self._safe_truncate(context_clean, per_field_budget - self.ELLIPSIS_LEN) + self.ELLIPSIS

            # Rebuild message
            message = (
                f"❌ <b>System Error</b>\n\n"
                f"Error: {error}\n"
                f"Context: {context}\n"
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # Deduplicate error messages
        if not self._should_send("error", dedup_key):
            return

        if self.send_message_sync(message):
            self._record_sent("error", dedup_key)
        self._save_to_dashboard("error", "System Error", message)

    def notify_trade_review(self, review, review_type: str) -> None:
        """
        Send multi-agent AI trade analysis to Telegram.

        Args:
            review: MultiAgentReviewResult from TradeReviewer
            review_type: "trade", "interesting_hold", or "hold"
        """
        ctx = review.trade_context

        # Determine message type for deduplication
        # Actual trades always go through, interesting holds get throttled
        msg_type = "trade_review" if review_type == "trade" else "interesting_hold"

        # Format signal breakdown
        breakdown = ctx.get('breakdown', {})
        breakdown_text = self._format_signal_breakdown(breakdown)

        if review_type in ("interesting_hold", "hold"):
            # Hold notification with multi-agent summary
            title = "🔍 <b>Interesting Hold</b>" if review_type == "interesting_hold" else "📋 <b>Hold Analysis</b>"

            # Build agent summary for holds
            # For holds: approved=True means "hold is correct", approved=False means "should act"
            stance_emoji = {"pro": "🟢", "neutral": "⚪", "opposing": "🔴"}
            agent_lines = []
            for agent in review.reviews:
                model_short = agent.model.split("/")[-1]
                # Descriptive verdict for holds
                if agent.approved:
                    verdict = "✅ Hold"
                else:
                    verdict = "❌ Act"
                conf = f"({agent.confidence*100:.0f}%)"
                stance_label = agent.stance.capitalize()
                # Use summary field (short) for notification display
                summary = getattr(agent, 'summary', None) or agent.reasoning[:80]
                agent_lines.append(
                    f"{stance_emoji.get(agent.stance, '⚪')} <b>{model_short}</b> ({stance_label}): "
                    f"{verdict} {conf}\n  <i>{summary}</i>"
                )
            agents_text = "\n\n".join(agent_lines) if agent_lines else "No reviews"

            # Format recommendation
            rec_emoji = {"wait": "⏳", "accumulate": "📈", "reduce": "📉"}
            rec_text = {
                "wait": "Wait for clearer signals",
                "accumulate": "Good opportunity to accumulate",
                "reduce": "Consider reducing exposure",
            }
            recommendation = review.judge_recommendation

            # For holds: APPROVED means "consider action", REJECTED means "hold confirmed"
            # Flip terminology to be intuitive for holds
            if review.judge_decision:
                judge_decision_text = "⚠️ ACTION SUGGESTED"
            else:
                judge_decision_text = "✅ HOLD CONFIRMED"

            message = (
                f"{title}\n\n"
                f"Signal Score: {ctx.get('score', 0)}/100 (threshold: {ctx.get('threshold', 60)})\n"
                f"Price: ¤{ctx.get('price', 0):,.2f}\n"
                f"📊 Fear & Greed: {ctx.get('fear_greed', 'N/A')} ({ctx.get('fear_greed_class', '')})\n\n"
                f"<b>Signal Breakdown</b>:\n{breakdown_text}\n\n"
                f"<b>Agent Reviews</b>:\n{agents_text}\n\n"
                f"<b>━━━ Judge Decision ━━━</b>\n"
                f"{judge_decision_text} ({review.judge_confidence*100:.0f}% confidence)\n"
                f"{rec_emoji.get(recommendation, '📌')} Recommendation: <b>{rec_text.get(recommendation, recommendation.upper())}</b>\n\n"
                f"<i>{review.judge_reasoning}</i>\n\n"
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            # Multi-agent trade review notification
            action = ctx.get('action', 'unknown').upper()

            # Build agent reviews section
            # For trades: approved=True means "approve trade", approved=False means "reject trade"
            stance_emoji = {"pro": "🟢", "neutral": "⚪", "opposing": "🔴"}
            agent_lines = []

            for agent in review.reviews:
                model_short = agent.model.split("/")[-1]
                # Descriptive verdict for trades
                if agent.approved:
                    verdict = "✅ Trade"
                else:
                    verdict = "❌ Skip"
                conf = f"({agent.confidence*100:.0f}%)"
                stance_label = agent.stance.capitalize()
                # Use summary field (short) for notification display
                summary = getattr(agent, 'summary', None) or agent.reasoning[:80]
                agent_lines.append(
                    f"{stance_emoji.get(agent.stance, '⚪')} <b>{model_short}</b> ({stance_label}): "
                    f"{verdict} {conf}\n  <i>{summary}</i>"
                )

            agents_text = "\n\n".join(agent_lines) if agent_lines else "  No agent reviews"

            # Format recommendation
            rec_emoji = {"wait": "⏳", "accumulate": "📈", "reduce": "📉"}
            rec_text = {
                "wait": "Wait for clearer signals",
                "accumulate": "Good opportunity to accumulate",
                "reduce": "Consider reducing exposure",
            }
            recommendation = review.judge_recommendation

            # Judge decision
            judge_decision_text = "✅ APPROVED" if review.judge_decision else "⛔ REJECTED"

            # Veto action explanation
            veto_action = review.final_veto_action
            # Build trade size section FIRST - THIS IS THE MOST IMPORTANT INFO
            estimated_size = ctx.get('estimated_size')
            trading_pair = ctx.get('trading_pair', 'BTC-USD')
            parts = trading_pair.split('-')
            base_symbol = parts[0] if len(parts) >= 1 else 'BTC'
            quote_symbol = parts[1] if len(parts) >= 2 else 'USD'

            trade_size_section = ""
            size_base = 0
            size_quote = 0
            if estimated_size:
                size_base = estimated_size.get('size_base', 0)
                size_quote = estimated_size.get('size_quote', 0)
                trade_size_section = (
                    f"\n<b>📦 Trade Size</b>:\n"
                    f"  {size_base:.6f} {base_symbol} (¤{size_quote:,.2f} {quote_symbol})\n"
                )

            # Build veto explanation with actual amounts
            if veto_action:
                if veto_action == "reduce" and size_base > 0:
                    reduced_base = size_base * 0.5
                    reduced_quote = size_quote * 0.5
                    veto_text = (
                        f"\n\n<b>Veto Action</b>: ⚠️ POSITION REDUCED TO 50%\n"
                        f"  Original: {size_base:.6f} {base_symbol} (¤{size_quote:,.2f})\n"
                        f"  Reduced: {reduced_base:.6f} {base_symbol} (¤{reduced_quote:,.2f})"
                    )
                else:
                    veto_explanations = {
                        "skip": "🚫 TRADE CANCELLED",
                        "reduce": "⚠️ POSITION REDUCED TO 50%",
                        "delay": "⏸️ TRADE DELAYED 15 MIN",
                        "info": "ℹ️ WARNING LOGGED, TRADE PROCEEDS",
                    }
                    veto_text = f"\n\n<b>Veto Action</b>: {veto_explanations.get(veto_action, veto_action.upper())}"
            else:
                veto_text = ""

            # Final outcome
            if review.judge_decision:
                outcome = "✅ <b>TRADE WILL EXECUTE</b>"
            elif veto_action == "skip":
                outcome = "🚫 <b>TRADE BLOCKED</b>"
            elif veto_action == "reduce":
                outcome = "⚠️ <b>TRADE EXECUTES (REDUCED)</b>"
            elif veto_action == "delay":
                outcome = "⏸️ <b>TRADE DELAYED</b>"
            else:
                outcome = "ℹ️ <b>TRADE PROCEEDS (INFO ONLY)</b>"

            # Build portfolio section if balances available
            portfolio_section = ""
            quote_balance = ctx.get('quote_balance')
            base_balance = ctx.get('base_balance')
            portfolio_value = ctx.get('portfolio_value')
            position_pct = ctx.get('position_percent', 0)

            if quote_balance is not None and base_balance is not None:
                portfolio_section = (
                    f"\n<b>Portfolio</b>:\n"
                    f"  💰 Available: ¤{quote_balance:,.2f} {quote_symbol}\n"
                    f"  ₿ Holdings: {base_balance:.6f} {base_symbol}\n"
                    f"  📊 Position: {position_pct:.1f}% of portfolio\n"
                )
                if portfolio_value is not None:
                    portfolio_section += f"  💼 Total Value: ¤{portfolio_value:,.2f}\n"

            message = (
                f"🤖 <b>Multi-Agent Trade Review</b>\n\n"
                f"📊 <b>Trade</b>: {action} @ ¤{ctx.get('price', 0):,.2f}\n"
                f"Signal Score: {ctx.get('score', 0)}/100\n"
                f"Fear & Greed: {ctx.get('fear_greed', 'N/A')} ({ctx.get('fear_greed_class', '')})"
                f"{trade_size_section}"
                f"{portfolio_section}\n"
                f"<b>Agent Reviews</b>:\n{agents_text}\n\n"
                f"<b>━━━ Judge Decision ━━━</b>\n"
                f"{judge_decision_text} ({review.judge_confidence*100:.0f}% confidence)\n"
                f"{rec_emoji.get(recommendation, '📌')} Recommendation: <b>{rec_text.get(recommendation, recommendation.upper())}</b>\n\n"
                f"<i>{review.judge_reasoning}</i>"
                f"{veto_text}\n\n"
                f"{outcome}\n\n"
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # For interesting holds, use deduplication to avoid spam
        # Key on score + recommendation to detect meaningful changes
        dedup_key = f"{ctx.get('score', 0)}_{review.judge_recommendation}_{review.judge_decision}"

        if not self._should_send(msg_type, dedup_key):
            logger.debug(
                "interesting_hold_throttled",
                score=ctx.get('score'),
                recommendation=review.judge_recommendation,
            )
            return

        if self.send_message_sync(message):
            self._record_sent(msg_type, dedup_key)
        title = f"AI Review: {review_type.replace('_', ' ').title()}"
        self._save_to_dashboard(msg_type, title, message)

    def notify_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        threshold_adj: int,
        position_mult: float,
        components: dict,
    ) -> None:
        """
        Send notification when market regime changes.

        Args:
            old_regime: Previous regime name
            new_regime: New regime name
            threshold_adj: Threshold adjustment being applied
            position_mult: Position size multiplier being applied
            components: Breakdown of regime components (sentiment, volatility, trend)
        """
        # Regime emojis
        regime_emoji = {
            "risk_on": "🟢",
            "opportunistic": "🟡",
            "neutral": "⚪",
            "cautious": "🟠",
            "risk_off": "🔴",
            "disabled": "⚫",
        }

        old_emoji = regime_emoji.get(old_regime, "⚪")
        new_emoji = regime_emoji.get(new_regime, "⚪")

        # Format components breakdown
        component_lines = []
        if "sentiment" in components:
            s = components["sentiment"]
            component_lines.append(
                f"  Fear & Greed: {s.get('value', 'N/A')} ({s.get('category', '')})"
            )
        if "volatility" in components:
            v = components["volatility"]
            component_lines.append(f"  Volatility: {v.get('level', 'N/A')}")
        if "trend" in components:
            t = components["trend"]
            component_lines.append(f"  Trend: {t.get('direction', 'N/A')}")

        components_text = "\n".join(component_lines) if component_lines else "  No data"

        # Threshold direction
        if threshold_adj < 0:
            threshold_text = f"{threshold_adj} (easier to trade)"
        elif threshold_adj > 0:
            threshold_text = f"+{threshold_adj} (harder to trade)"
        else:
            threshold_text = "0 (no change)"

        message = (
            f"📊 <b>Market Regime Changed</b>\n\n"
            f"{old_emoji} {old_regime.replace('_', ' ').title()} → "
            f"{new_emoji} <b>{new_regime.replace('_', ' ').title()}</b>\n\n"
            f"<b>Adjustments</b>:\n"
            f"  Threshold: {threshold_text}\n"
            f"  Position size: {position_mult:.2f}×\n\n"
            f"<b>Market Conditions</b>:\n{components_text}\n\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)
        self._save_to_dashboard(
            "regime_change",
            f"Regime: {old_regime} → {new_regime}",
            message,
        )

    def notify_weight_profile(
        self,
        old_profile: str,
        new_profile: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """
        Send notification when weight profile changes.

        Args:
            old_profile: Previous profile name
            new_profile: New profile name
            confidence: AI confidence (0.0 to 1.0)
            reasoning: AI reasoning for the selection
        """
        # Deduplicate to prevent oscillation spam (e.g., repeated A→B→A→B)
        dedup_key = f"{old_profile}:{new_profile}"
        if not self._should_send("weight_profile", dedup_key):
            logger.debug(
                "weight_profile_notification_throttled",
                old=old_profile,
                new=new_profile,
            )
            return

        profile_emoji = {
            "trending": "📈",
            "ranging": "↔️",
            "volatile": "⚡",
            "default": "⚖️",
        }.get(new_profile, "🔄")

        confidence_pct = int(confidence * 100)
        message = (
            f"{profile_emoji} <b>Weight Profile Changed</b>\n\n"
            f"<b>Profile:</b> {old_profile} → {new_profile}\n"
            f"<b>Confidence:</b> {confidence_pct}%\n\n"
            f"<b>AI Reasoning:</b>\n{reasoning}"
        )

        if self.send_message_sync(message):
            self._record_sent("weight_profile", dedup_key)
        self._save_to_dashboard(
            "weight_profile",
            f"Weight Profile: {new_profile}",
            message,
        )

    def _format_signal_breakdown(self, breakdown: dict) -> str:
        """Format signal breakdown for Telegram display."""
        if not breakdown:
            return "  No breakdown available"

        # Order indicators for consistent display
        indicator_order = ['rsi', 'macd', 'bollinger', 'ema', 'volume', 'trend_filter']
        indicator_names = {
            'rsi': 'RSI',
            'macd': 'MACD',
            'bollinger': 'Bollinger',
            'ema': 'EMA',
            'volume': 'Volume',
            'trend_filter': 'Trend Filter',
        }

        lines = []
        for key in indicator_order:
            if key in breakdown:
                value = breakdown[key]
                name = indicator_names.get(key, key.upper())
                # Show sign and value with visual indicator
                if value > 0:
                    lines.append(f"  📈 {name}: +{value}")
                elif value < 0:
                    lines.append(f"  📉 {name}: {value}")
                else:
                    lines.append(f"  ➖ {name}: 0")

        return "\n".join(lines) if lines else "  No breakdown available"

    def notify_market_analysis(
        self,
        review,
        indicators,
        volatility: str,
        fear_greed: int,
        fear_greed_class: str,
        current_price: Decimal,
        analysis_reason: str = "hourly_volatile",
    ) -> None:
        """
        Send multi-agent hourly market analysis notification.

        Args:
            review: MultiAgentReviewResult from TradeReviewer.analyze_market()
            indicators: Current indicator values
            volatility: Volatility level
            fear_greed: Fear & Greed index value
            fear_greed_class: Fear & Greed classification
            current_price: Current BTC price
            analysis_reason: Why analysis was triggered (hourly_volatile or post_volatility)
        """
        # Stance emoji (for market analysis: bullish/neutral/bearish)
        stance_emoji = {"bullish": "🟢", "neutral": "⚪", "bearish": "🔴"}

        # Volatility emoji
        vol_emoji = {
            "low": "🌙",
            "normal": "☀️",
            "high": "⚡",
            "extreme": "🌪️",
        }

        # Recommendation emoji
        rec_emoji = {
            "wait": "⏳",
            "accumulate": "📈",
            "reduce": "📉",
        }
        rec_text = {
            "wait": "Wait for clearer signals",
            "accumulate": "Good opportunity to accumulate",
            "reduce": "Consider reducing exposure",
        }

        # Format RSI status
        rsi_status = "N/A"
        if indicators.rsi:
            if indicators.rsi < 30:
                rsi_status = f"{indicators.rsi:.1f} (Oversold)"
            elif indicators.rsi > 70:
                rsi_status = f"{indicators.rsi:.1f} (Overbought)"
            else:
                rsi_status = f"{indicators.rsi:.1f}"

        # Format MACD
        macd_status = "N/A"
        if indicators.macd_histogram:
            if indicators.macd_histogram > 0:
                macd_status = f"{indicators.macd_histogram:.0f} (Bullish)"
            else:
                macd_status = f"{indicators.macd_histogram:.0f} (Bearish)"

        # Build agent reviews section
        agent_lines = []
        for agent in review.reviews:
            model_short = agent.model.split("/")[-1]
            stance_label = agent.stance.capitalize()
            outlook = agent.sentiment.capitalize()  # outlook stored in sentiment field
            conf = f"({agent.confidence*100:.0f}%)"
            summary = getattr(agent, 'summary', None) or agent.reasoning[:80]
            agent_lines.append(
                f"{stance_emoji.get(agent.stance, '⚪')} <b>{model_short}</b> ({stance_label}): "
                f"{outlook} {conf}\n  <i>{summary}</i>"
            )
        agents_text = "\n\n".join(agent_lines) if agent_lines else "No reviews"

        recommendation = review.judge_recommendation

        # Title based on analysis reason
        if analysis_reason == "post_volatility":
            title = "📊 <b>Post-Volatility Analysis</b> (Market Calmed)"
        else:
            title = "📊 <b>Hourly Market Analysis</b>"

        message = (
            f"{title}\n\n"
            f"<b>Volatility</b>: {vol_emoji.get(volatility, '☀️')} {volatility.title()}\n\n"
            f"<b>Current Indicators</b>:\n"
            f"  💰 Price: ¤{float(current_price):,.2f}\n"
            f"  📊 RSI: {rsi_status}\n"
            f"  📈 MACD: {macd_status}\n"
            f"  😨 Fear & Greed: {fear_greed} ({fear_greed_class})\n\n"
            f"<b>Analyst Reviews</b>:\n{agents_text}\n\n"
            f"<b>━━━ Judge Synthesis ━━━</b>\n"
            f"Confidence: {review.judge_confidence*100:.0f}%\n"
            f"{rec_emoji.get(recommendation, '📌')} Recommendation: <b>{rec_text.get(recommendation, recommendation.upper())}</b>\n\n"
            f"<i>{review.judge_reasoning}</i>\n\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)
        self._save_to_dashboard("market_analysis", "Hourly Market Analysis", message)

    def notify_periodic_report(
        self,
        period: str,
        report_date: str,
        portfolio_return: float,
        btc_return: float,
        alpha: float,
        pnl: Decimal,
        ending_balance: Decimal,
        trades: int,
        is_paper: bool = False,
    ) -> None:
        """
        Send periodic performance report (daily, weekly, monthly).

        Args:
            period: Report period ("Daily", "Weekly", "Monthly")
            report_date: Date or date range string
            portfolio_return: Portfolio return percentage
            btc_return: BTC HODL return percentage
            alpha: Alpha (outperformance vs BTC)
            pnl: Profit/loss amount
            ending_balance: Ending balance
            trades: Number of trades
            is_paper: Whether in paper trading mode
        """
        # Determine performance emoji
        threshold = 1 if period == "Daily" else 2
        if alpha > threshold:
            perf_emoji = "🚀"
        elif alpha > 0:
            perf_emoji = "✅"
        elif alpha > -threshold:
            perf_emoji = "➖"
        else:
            perf_emoji = "📉"

        mode = "PAPER" if is_paper else "LIVE"

        # Period-specific emoji
        period_emoji = {"Daily": "📊", "Weekly": "📅", "Monthly": "📆"}.get(period, "📊")

        message = (
            f"{period_emoji} <b>{period} Report</b> ({mode})\n"
            f"{report_date}\n\n"
            f"<b>Portfolio</b>: {portfolio_return:+.2f}%\n"
            f"<b>BTC (HODL)</b>: {btc_return:+.2f}%\n"
            f"{perf_emoji} <b>Alpha</b>: {alpha:+.2f}%\n\n"
            f"P&L: €{pnl:+,.2f}\n"
            f"Balance: €{ending_balance:,.2f}\n"
            f"Trades: {trades}"
        )

        self.send_message_sync(message)
        self._save_to_dashboard(
            "periodic_report",
            f"{period} Report ({mode})",
            message,
        )
