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
from datetime import datetime
from decimal import Decimal
from time import time
from typing import Optional

import structlog
from telegram import Bot
from telegram.error import TelegramError

logger = structlog.get_logger(__name__)


# Default cooldown periods for different message types (seconds)
# These prevent spam - same message won't repeat within cooldown period
DEFAULT_COOLDOWNS = {
    "circuit_breaker": 3600,  # 1 hour - don't repeat same breaker status
    "loss_limit": 3600,       # 1 hour
    "error": 1800,            # 30 minutes - same error won't repeat
    "order_failed": 1800,     # 30 minutes
    "regime_change": 0,       # Always send (only fires on actual change)
    "trade": 0,               # Always send trade notifications
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

        # Deduplication: track last message time by type
        self._last_message_by_type: dict[str, float] = {}
        # Track last message content hash to detect duplicates
        self._last_message_hash: dict[str, str] = {}

        if enabled and bot_token and chat_id:
            self._bot = Bot(token=bot_token)
            logger.info("telegram_notifier_initialized")
        else:
            logger.warning("telegram_notifier_disabled")

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
            f"Size: {size:.8f}\n"
            f"Price: ¤{price:,.2f}\n"
            f"Fee: ¤{fee:.2f}\n"
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
            f"Size: {size:.8f}\n"
            f"Error: {error}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Deduplicate order failed messages (same error)
        if not self._should_send("order_failed", error):
            return

        if self.send_message_sync(message):
            self._record_sent("order_failed", error)

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

        # Deduplicate circuit breaker messages
        if not self._should_send("circuit_breaker", f"{level}:{reason}"):
            return

        if self.send_message_sync(message):
            self._record_sent("circuit_breaker", f"{level}:{reason}")

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

        # Deduplicate loss limit messages
        dedup_key = f"{limit_type}:{int(loss_percent)}"
        if not self._should_send("loss_limit", dedup_key):
            return

        if self.send_message_sync(message):
            self._record_sent("loss_limit", dedup_key)

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
            f"Date: {datetime.now().strftime('%Y-%m-%d')}"
        )

        self.send_message_sync(message)

    def notify_startup(self, mode: str, balance: Decimal, exchange: str = "Coinbase") -> None:
        """Send notification on bot startup."""
        message = (
            f"🤖 <b>Trading Bot Started</b>\n\n"
            f"Exchange: {exchange}\n"
            f"Mode: {mode.upper()}\n"
            f"Balance: ¤{balance:,.2f}\n"
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

        # Deduplicate error messages
        dedup_key = f"{error}:{context}"
        if not self._should_send("error", dedup_key):
            return

        if self.send_message_sync(message):
            self._record_sent("error", dedup_key)

    def notify_trade_review(self, review, review_type: str) -> None:
        """
        Send multi-agent AI trade analysis to Telegram.

        Args:
            review: MultiAgentReviewResult from TradeReviewer
            review_type: "trade", "interesting_hold", or "hold"
        """
        ctx = review.trade_context

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
                f"Signal Score: {ctx.get('score', 0)}/100 (threshold: 60)\n"
                f"Price: ${ctx.get('price', 0):,.2f}\n"
                f"📊 Fear & Greed: {ctx.get('fear_greed', 'N/A')} ({ctx.get('fear_greed_class', '')})\n\n"
                f"<b>Signal Breakdown</b>:\n{breakdown_text}\n\n"
                f"<b>Agent Reviews</b>:\n{agents_text}\n\n"
                f"<b>━━━ Judge Decision ━━━</b>\n"
                f"{judge_decision_text} ({review.judge_confidence*100:.0f}% confidence)\n"
                f"{rec_emoji.get(recommendation, '📌')} Recommendation: <b>{rec_text.get(recommendation, recommendation.upper())}</b>\n\n"
                f"<i>{review.judge_reasoning}</i>\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
            if veto_action:
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

            message = (
                f"🤖 <b>Multi-Agent Trade Review</b>\n\n"
                f"📊 <b>Trade</b>: {action} @ ${ctx.get('price', 0):,.2f}\n"
                f"Signal Score: {ctx.get('score', 0)}/100\n"
                f"Fear & Greed: {ctx.get('fear_greed', 'N/A')} ({ctx.get('fear_greed_class', '')})\n\n"
                f"<b>Agent Reviews</b>:\n{agents_text}\n\n"
                f"<b>━━━ Judge Decision ━━━</b>\n"
                f"{judge_decision_text} ({review.judge_confidence*100:.0f}% confidence)\n"
                f"{rec_emoji.get(recommendation, '📌')} Recommendation: <b>{rec_text.get(recommendation, recommendation.upper())}</b>\n\n"
                f"<i>{review.judge_reasoning}</i>"
                f"{veto_text}\n\n"
                f"{outcome}\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

        self.send_message_sync(message)

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
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)

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
        analysis,
        indicators,
        volatility: str,
        fear_greed: int,
        fear_greed_class: str,
        current_price: Decimal,
    ) -> None:
        """
        Send hourly market analysis notification.

        Args:
            analysis: MarketAnalysis result from AI
            indicators: Current indicator values
            volatility: Volatility level
            fear_greed: Fear & Greed index value
            fear_greed_class: Fear & Greed classification
            current_price: Current BTC price
        """
        # Outlook emoji
        outlook_emoji = {
            "bullish": "🟢",
            "bearish": "🔴",
            "neutral": "⚪",
        }

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

        message = (
            f"📊 <b>Hourly Market Analysis</b>\n\n"
            f"<b>Outlook</b>: {outlook_emoji.get(analysis.outlook, '⚪')} "
            f"{analysis.outlook.title()} ({analysis.confidence*100:.0f}% confidence)\n"
            f"<b>Volatility</b>: {vol_emoji.get(volatility, '☀️')} {volatility.title()}\n\n"
            f"<b>Current Indicators</b>:\n"
            f"  💰 Price: ${float(current_price):,.2f}\n"
            f"  📊 RSI: {rsi_status}\n"
            f"  📈 MACD: {macd_status}\n"
            f"  😨 Fear & Greed: {fear_greed} ({fear_greed_class})\n\n"
            f"💡 <b>Summary</b>:\n{analysis.summary}\n\n"
            f"{rec_emoji.get(analysis.recommendation, '⏳')} <b>Recommendation</b>: "
            f"{analysis.recommendation.title()}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.send_message_sync(message)
