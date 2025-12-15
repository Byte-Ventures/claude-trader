#!/usr/bin/env python
"""
Post-mortem analysis script for claude-trader.

Analyzes trades using Claude Code CLI to identify algorithmic weaknesses
and creates GitHub Discussions with improvement recommendations.

Defaults: --paper --last 1 --include-source

Usage:
    python tools/postmortem.py                       # Analyze last paper trade
    python tools/postmortem.py --last 5              # Analyze last 5 paper trades
    python tools/postmortem.py --live --last 3       # Analyze last 3 live trades
    python tools/postmortem.py --create-discussion   # Create GitHub Discussion
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.state.database import (
    Notification,
    RegimeHistory,
    SignalHistory,
    Trade,
    WhaleEvent,
)


@dataclass
class TradeContext:
    """A trade with its associated signal, regime, and whale context."""

    # Trade data
    trade_id: int
    symbol: str
    side: str
    size: Decimal
    price: Decimal
    fee: Decimal
    realized_pnl: Optional[Decimal]
    executed_at: datetime

    # Signal data
    signal_id: Optional[int] = None
    signal_timestamp: Optional[datetime] = None
    current_price: Optional[Decimal] = None
    final_score: Optional[float] = None
    raw_score: Optional[float] = None
    threshold_used: Optional[int] = None
    action: Optional[str] = None

    # Indicator scores
    rsi_score: Optional[float] = None
    macd_score: Optional[float] = None
    bollinger_score: Optional[float] = None
    ema_score: Optional[float] = None
    volume_score: Optional[float] = None

    # Raw indicator values
    rsi_value: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_position: Optional[float] = None
    ema_gap_percent: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Adjustments
    trend_filter_adj: Optional[float] = None
    momentum_mode_adj: Optional[float] = None
    whale_activity_adj: Optional[float] = None
    htf_bias_adj: Optional[float] = None

    # HTF context
    htf_bias: Optional[str] = None
    htf_daily_trend: Optional[str] = None
    htf_4h_trend: Optional[str] = None

    # Regime context
    regime_name: Optional[str] = None
    sentiment_value: Optional[int] = None
    sentiment_category: Optional[str] = None
    volatility_level: Optional[str] = None
    trend_direction: Optional[str] = None

    # Whale events
    whale_events: list = field(default_factory=list)

    # Notifications
    notifications: list = field(default_factory=list)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Post-mortem analysis of trading bot trades using Claude AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/postmortem.py                       # Analyze last paper trade (default)
  python tools/postmortem.py --last 5              # Analyze last 5 paper trades
  python tools/postmortem.py --live --last 3       # Analyze last 3 live trades
  python tools/postmortem.py --trade-id 42         # Analyze specific trade
  python tools/postmortem.py --create-discussion   # Create GitHub Discussion with analysis
  python tools/postmortem.py --no-source           # Don't include source code in prompt
        """,
    )

    # Mode selection (paper is default)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--paper", action="store_true", default=True, help="Analyze paper trades (default)")
    mode_group.add_argument("--live", action="store_true", help="Analyze live trades")

    # Trade selection (mutually exclusive)
    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        "--last", type=int, metavar="N", help="Analyze last N trades"
    )
    select_group.add_argument(
        "--trade-id", type=int, help="Analyze specific trade by ID"
    )
    select_group.add_argument(
        "--start", type=str, help="Start date (YYYY-MM-DD) for date range"
    )

    # Additional options
    parser.add_argument(
        "--end", type=str, help="End date (YYYY-MM-DD) for date range"
    )
    parser.add_argument(
        "--create-discussion", action="store_true", help="Create GitHub Discussion with analysis"
    )
    parser.add_argument(
        "--print-context",
        action="store_true",
        help="Print context only without Claude analysis (debug)",
    )
    # Auto-detect paths: try local first, fallback to production
    local_db = project_root / "data" / "trading.db"
    prod_db = Path("/opt/claude-trader/data/trading.db")
    default_db = local_db if local_db.exists() else prod_db
    parser.add_argument(
        "--db",
        type=str,
        default=str(default_db),
        help=f"Database path (default: {default_db})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    # Source root: try local first, fallback to production
    local_source = project_root
    prod_source = Path("/opt/claude-trader")
    default_source = local_source if (local_source / "src").exists() else prod_source
    parser.add_argument(
        "--source-root",
        type=str,
        default=str(default_source),
        help=f"Path to trading bot source code (default: {default_source})",
    )
    parser.add_argument(
        "--no-source",
        action="store_true",
        help="Don't include source files in prompt (default: include them)",
    )

    return parser


def get_session(db_path: str) -> Session:
    """Create a read-only database session."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()


def fetch_trades_with_context(
    session: Session,
    is_paper: bool,
    trade_ids: Optional[list[int]] = None,
    limit: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> list[TradeContext]:
    """
    Fetch trades and correlate with signals, regime, and whale events.

    Signal correlation: Find signal_history where trade_executed=True
    within 120 seconds before trade.executed_at (accounts for order placement latency).
    """
    # Build trade query
    query = session.query(Trade).filter(Trade.is_paper == is_paper)

    if trade_ids:
        query = query.filter(Trade.id.in_(trade_ids))
    if start_date:
        query = query.filter(Trade.executed_at >= start_date)
    if end_date:
        query = query.filter(Trade.executed_at <= end_date)

    query = query.order_by(Trade.executed_at.desc())

    if limit:
        query = query.limit(limit)

    trades = query.all()
    results = []

    for trade in trades:
        # Find matching signal (within 120s window before trade)
        window_start = trade.executed_at - timedelta(seconds=120)
        signal = (
            session.query(SignalHistory)
            .filter(
                SignalHistory.is_paper == is_paper,
                SignalHistory.trade_executed == True,  # noqa: E712
                SignalHistory.timestamp >= window_start,
                SignalHistory.timestamp <= trade.executed_at,
            )
            .order_by(SignalHistory.timestamp.desc())
            .first()
        )

        # Find regime at trade time
        regime = (
            session.query(RegimeHistory)
            .filter(
                RegimeHistory.is_paper == is_paper,
                RegimeHistory.created_at <= trade.executed_at,
            )
            .order_by(RegimeHistory.created_at.desc())
            .first()
        )

        # Find whale events near trade (within 10 minutes)
        whale_window = timedelta(minutes=10)
        whales = (
            session.query(WhaleEvent)
            .filter(
                WhaleEvent.is_paper == is_paper,
                WhaleEvent.timestamp >= trade.executed_at - whale_window,
                WhaleEvent.timestamp <= trade.executed_at + whale_window,
            )
            .all()
        )

        # Find notifications near trade (within 5 minutes)
        notif_window = timedelta(minutes=5)
        notifications = (
            session.query(Notification)
            .filter(
                Notification.is_paper == is_paper,
                Notification.created_at >= trade.executed_at - notif_window,
                Notification.created_at <= trade.executed_at + notif_window,
            )
            .order_by(Notification.created_at.asc())
            .all()
        )

        # Build TradeContext
        ctx = TradeContext(
            trade_id=trade.id,
            symbol=trade.symbol,
            side=trade.side,
            size=Decimal(trade.size),
            price=Decimal(trade.price),
            fee=Decimal(trade.fee) if trade.fee else Decimal("0"),
            realized_pnl=Decimal(trade.realized_pnl) if trade.realized_pnl else None,
            executed_at=trade.executed_at,
        )

        # Add signal data if found
        if signal:
            ctx.signal_id = signal.id
            ctx.signal_timestamp = signal.timestamp
            ctx.current_price = (
                Decimal(signal.current_price) if signal.current_price else None
            )
            ctx.final_score = signal.final_score
            ctx.raw_score = signal.raw_score
            ctx.threshold_used = signal.threshold_used
            ctx.action = signal.action

            ctx.rsi_score = signal.rsi_score
            ctx.macd_score = signal.macd_score
            ctx.bollinger_score = signal.bollinger_score
            ctx.ema_score = signal.ema_score
            ctx.volume_score = signal.volume_score

            ctx.rsi_value = signal.rsi_value
            ctx.macd_histogram = signal.macd_histogram
            ctx.bb_position = signal.bb_position
            ctx.ema_gap_percent = signal.ema_gap_percent
            ctx.volume_ratio = signal.volume_ratio

            ctx.trend_filter_adj = signal.trend_filter_adj
            ctx.momentum_mode_adj = signal.momentum_mode_adj
            ctx.whale_activity_adj = signal.whale_activity_adj
            ctx.htf_bias_adj = signal.htf_bias_adj

            ctx.htf_bias = signal.htf_bias
            ctx.htf_daily_trend = signal.htf_daily_trend
            ctx.htf_4h_trend = signal.htf_4h_trend

        # Add regime data if found
        if regime:
            ctx.regime_name = regime.regime_name
            ctx.sentiment_value = regime.sentiment_value
            ctx.sentiment_category = regime.sentiment_category
            ctx.volatility_level = regime.volatility_level
            ctx.trend_direction = regime.trend_direction

        # Add whale events
        ctx.whale_events = [
            {
                "timestamp": w.timestamp.isoformat(),
                "volume_ratio": w.volume_ratio,
                "direction": w.direction,
                "price_change_pct": w.price_change_pct,
            }
            for w in whales
        ]

        # Add notifications
        ctx.notifications = [
            {
                "timestamp": n.created_at.isoformat(),
                "type": n.type,
                "title": n.title,
                "message": n.message,
            }
            for n in notifications
        ]

        results.append(ctx)

    return results


def format_trade_context(ctx: TradeContext) -> str:
    """Format a single trade with its context for Claude analysis."""
    lines = [
        f"## Trade #{ctx.trade_id}",
        f"- **Time:** {ctx.executed_at.isoformat()}",
        f"- **Side:** {ctx.side.upper()}",
        f"- **Symbol:** {ctx.symbol}",
        f"- **Size:** {ctx.size}",
        f"- **Price:** ${ctx.price:,.2f}",
        f"- **Fee:** ${ctx.fee:,.4f}",
    ]

    if ctx.realized_pnl is not None:
        pnl_sign = "+" if ctx.realized_pnl >= 0 else ""
        lines.append(f"- **Realized P&L:** {pnl_sign}${ctx.realized_pnl:,.2f}")

    if ctx.signal_id:
        lines.extend(
            [
                "",
                "### Signal Analysis",
                f"- **Final Score:** {fmt(ctx.final_score)} (threshold: {ctx.threshold_used})",
                f"- **Raw Score:** {fmt(ctx.raw_score)}",
                f"- **Action:** {ctx.action}",
                "",
                "#### Indicator Scores",
                "| Indicator | Score | Raw Value |",
                "|-----------|-------|-----------|",
                f"| RSI | {fmt(ctx.rsi_score)} | {fmt(ctx.rsi_value)} |",
                f"| MACD | {fmt(ctx.macd_score)} | {fmt(ctx.macd_histogram, '.4f')} |",
                f"| Bollinger | {fmt(ctx.bollinger_score)} | {fmt(ctx.bb_position, '.2f')} |",
                f"| EMA | {fmt(ctx.ema_score)} | {fmt(ctx.ema_gap_percent, '.2f', '%')} |",
                f"| Volume | {fmt(ctx.volume_score)} | {fmt(ctx.volume_ratio, '.2f', 'x')} |",
                "",
                "#### Score Adjustments",
                f"- Trend Filter: {ctx.trend_filter_adj:+.1f}" if ctx.trend_filter_adj else "- Trend Filter: 0",
                f"- Momentum Mode: {ctx.momentum_mode_adj:+.1f}" if ctx.momentum_mode_adj else "- Momentum Mode: 0",
                f"- Whale Activity: {ctx.whale_activity_adj:+.1f}" if ctx.whale_activity_adj else "- Whale Activity: 0",
                f"- HTF Bias: {ctx.htf_bias_adj:+.1f}" if ctx.htf_bias_adj else "- HTF Bias: 0",
                "",
                "#### Higher Timeframe Context",
                f"- HTF Bias: {ctx.htf_bias or 'N/A'}",
                f"- Daily Trend: {ctx.htf_daily_trend or 'N/A'}",
                f"- 4H Trend: {ctx.htf_4h_trend or 'N/A'}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Signal Analysis",
                "**WARNING:** No matching signal found for this trade",
            ]
        )

    # Market regime context
    if ctx.regime_name:
        lines.extend(
            [
                "",
                "### Market Regime",
                f"- Regime: {ctx.regime_name}",
            ]
        )
        if ctx.sentiment_value is not None:
            lines.append(
                f"- Sentiment: {ctx.sentiment_category} ({ctx.sentiment_value}/100)"
            )
        if ctx.volatility_level:
            lines.append(f"- Volatility: {ctx.volatility_level}")
        if ctx.trend_direction:
            lines.append(f"- Trend: {ctx.trend_direction}")

    # Whale events
    if ctx.whale_events:
        lines.extend(
            [
                "",
                "### Whale Activity Near Trade",
            ]
        )
        for whale in ctx.whale_events:
            lines.append(
                f"- {whale['timestamp']}: {whale['direction']} whale, "
                f"{whale['volume_ratio']:.1f}x volume"
                + (
                    f", {whale['price_change_pct']:.2f}% price change"
                    if whale["price_change_pct"]
                    else ""
                )
            )

    # Notifications
    if ctx.notifications:
        lines.extend(
            [
                "",
                "### Notifications",
            ]
        )
        for notif in ctx.notifications:
            lines.append(f"- [{notif['type']}] **{notif['title']}**: {notif['message']}")

    return "\n".join(lines)


def read_source_file(path: Path) -> Optional[str]:
    """Read a source file if it exists."""
    try:
        if path.exists():
            return path.read_text()
    except Exception:
        pass
    return None


def fmt(value: Optional[float], spec: str = ".1f", suffix: str = "") -> str:
    """Format an optional float value, returning 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:{spec}}{suffix}"


def format_analysis_prompt(
    trades: list[TradeContext],
    is_paper: bool,
    source_root: Path,
    db_path: str,
    include_source: bool = False,
) -> str:
    """Generate the full prompt for Claude analysis."""
    mode = "PAPER" if is_paper else "LIVE"

    # Calculate summary stats
    total_pnl = sum(t.realized_pnl for t in trades if t.realized_pnl is not None)
    sell_trades = [t for t in trades if t.side == "sell" and t.realized_pnl is not None]
    winning = sum(1 for t in sell_trades if t.realized_pnl > 0)
    losing = sum(1 for t in sell_trades if t.realized_pnl < 0)

    # Get threshold used (from first trade with signal)
    threshold = None
    for t in trades:
        if t.threshold_used:
            threshold = t.threshold_used
            break

    prompt = f"""# Trading Bot Post-Mortem Analysis

**Mode:** {mode} trading
**Trades analyzed:** {len(trades)}
**Date range:** {trades[-1].executed_at.strftime('%Y-%m-%d %H:%M')} to {trades[0].executed_at.strftime('%Y-%m-%d %H:%M')}
**Total Realized P&L:** ${total_pnl:,.2f}
**Sell trades:** {len(sell_trades)} (Won: {winning}, Lost: {losing})

## Source Code Location

The trading bot source code is located at: `{source_root}`

Key files for analysis:
- `{source_root}/src/strategy/signal_scorer.py` - Signal calculation logic (RSI, MACD, Bollinger, EMA, Volume scoring)
- `{source_root}/src/daemon/runner.py` - Main trading loop and trade execution
- `{source_root}/config/settings.py` - Configuration schema and defaults (parameter names and types)
- `{source_root}/src/safety/` - Safety systems (circuit breaker, loss limiter)

IMPORTANT: When suggesting config changes, reference parameter names from config/settings.py.
**NEVER read or include .env file contents** - it contains secrets (API keys, tokens).
**NEVER include any credentials, API keys, or secrets in your analysis.**
"""

    # Optionally include source code
    if include_source:
        signal_scorer = read_source_file(source_root / "src/strategy/signal_scorer.py")
        settings = read_source_file(source_root / "config/settings.py")

        if signal_scorer:
            prompt += f"""
## Source Code: signal_scorer.py

```python
{signal_scorer}
```
"""
        if settings:
            prompt += f"""
## Source Code: settings.py

```python
{settings}
```
"""

    prompt += """
## Purpose

Identify algorithmic weaknesses in the trading bot and suggest specific improvements.
Focus on patterns that lead to losing trades and missed opportunities.

## Analysis Required

1. **Indicator Performance Analysis**
   - Which indicators contributed to winning vs losing trades?
   - Are any indicators generating consistent false signals?
   - What indicator combinations worked best/worst?

2. **Threshold Analysis**
   - Is the signal threshold (currently {threshold or 'unknown'}) appropriate?
   - Are trades being taken on weak signals?
   - Should threshold vary by market regime?

3. **Adjustment Effectiveness**
   - Is trend_filter_adj preventing bad trades effectively?
   - Is htf_bias_adj improving trade quality?
   - Are whale_activity adjustments helping or hurting?

4. **Missing Safeguards**
   - What patterns precede losing trades that could be detected?
   - What additional filters would improve win rate?

5. **Specific Recommendations**
   - Config changes with specific values (e.g., "increase threshold from 65 to 70")
   - Code changes to add/modify logic
   - New indicators or data sources to consider

## Important Instructions

- You have access to tools. If you need more data, you CAN read the database using sqlite3:
  ```bash
  sqlite3 {db_path} "SELECT * FROM signal_history WHERE is_paper={1 if is_paper else 0} ORDER BY timestamp DESC LIMIT 10"
  ```
- Database location: `{db_path}`
- Key tables: `trades`, `signal_history`, `regime_history`, `whale_events`, `notifications`, `daily_stats`
- All tables have `is_paper` column (1=paper, 0=live).
- Log files: `{source_root}/logs/trading.log` contains detailed bot activity including regime calculations, indicator values, and trade decisions. Use grep/read to search for specific trade timestamps.
- Source code is at `{source_root}/src/` - key files: `strategy/signal_scorer.py`, `strategy/regime.py`, `daemon/runner.py`.
- Provide concrete, actionable recommendations.
- End with a summary of your top 3 recommendations.

---

# Trade Data

"""

    for ctx in trades:
        prompt += format_trade_context(ctx)
        prompt += "\n\n---\n\n"

    return prompt


def invoke_claude(prompt: str, verbose: bool = False) -> str:
    """
    Invoke Claude CLI for trade analysis.

    Uses --allowedTools to permit database queries and file reads without prompts.
    """
    if verbose:
        print(f"[INFO] Sending {len(prompt)} chars to Claude...")
        print("[INFO] Claude will have tool access - may take longer...")

    result = subprocess.run(
        [
            "claude", "-p", prompt,
            "--allowedTools", "Bash(sqlite3:*)", "Read", "Grep", "Glob",
        ],
        capture_output=True,
        text=True,
        timeout=900,  # 15 minute timeout (tools take longer)
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr}")

    return result.stdout


def get_discussion_ids(verbose: bool = False) -> tuple[str, str]:
    """Fetch repository ID and Post-Mortems category ID from current repo."""
    import json

    if verbose:
        print("[INFO] Fetching repository and category IDs...")

    # Get repo owner/name from git remote
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "owner,name"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get repo info: {result.stderr}")

    repo_info = json.loads(result.stdout)
    owner = repo_info["owner"]["login"]
    name = repo_info["name"]

    # Fetch repo ID and category ID via GraphQL
    query = f'''
    query {{
      repository(owner: "{owner}", name: "{name}") {{
        id
        discussionCategories(first: 20) {{
          nodes {{
            id
            name
            slug
          }}
        }}
      }}
    }}
    '''

    result = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={query}"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch discussion categories: {result.stderr}")

    response = json.loads(result.stdout)
    if "errors" in response:
        raise RuntimeError(f"GraphQL error: {response['errors']}")

    repo_id = response["data"]["repository"]["id"]
    categories = response["data"]["repository"]["discussionCategories"]["nodes"]

    # Find Post-Mortems category
    category_id = None
    for cat in categories:
        if cat["slug"] == "post-mortems":
            category_id = cat["id"]
            break

    if not category_id:
        raise RuntimeError(
            "Post-Mortems category not found. "
            "Create it in Settings → Discussions → New Category"
        )

    return repo_id, category_id


def create_github_discussion(
    title: str,
    body: str,
    verbose: bool = False,
) -> str:
    """Create GitHub Discussion using GraphQL API. Returns discussion URL."""
    import json

    if verbose:
        print(f"[INFO] Creating GitHub Discussion: {title}")

    repo_id, category_id = get_discussion_ids(verbose)

    # Use GraphQL variables for proper escaping (prevents injection)
    query = """
    mutation CreateDiscussion($repoId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
      createDiscussion(input: {
        repositoryId: $repoId,
        categoryId: $categoryId,
        title: $title,
        body: $body
      }) {
        discussion {
          url
        }
      }
    }
    """

    result = subprocess.run(
        [
            "gh", "api", "graphql",
            "-f", f"query={query}",
            "-F", f"repoId={repo_id}",
            "-F", f"categoryId={category_id}",
            "-F", f"title={title}",
            "-F", f"body={body}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"GitHub Discussion creation failed: {result.stderr}")

    response = json.loads(result.stdout)
    if "errors" in response:
        raise RuntimeError(f"GraphQL error: {response['errors']}")

    return response["data"]["createDiscussion"]["discussion"]["url"]


def trigger_postmortem_review(
    discussion_url: str,
    title: str,
    body: str,
    verbose: bool = False,
) -> bool:
    """Trigger repository_dispatch event to start post-mortem review workflow.

    The claude-code-action doesn't support discussion events, so we use
    repository_dispatch to trigger the review workflow with discussion data.

    Returns True if successful, False otherwise.
    """
    import json as json_module

    if verbose:
        print("[INFO] Triggering post-mortem review workflow...")

    # repository_dispatch payload (max 10 keys, max 65535 chars total)
    # Truncate body to leave room for other fields
    max_body_len = 60000
    truncated_body = body[:max_body_len] + "..." if len(body) > max_body_len else body

    payload = json_module.dumps({
        "event_type": "postmortem-review",
        "client_payload": {
            "discussion_url": discussion_url,
            "title": title,
            "body": truncated_body,
        }
    })

    result = subprocess.run(
        [
            "gh", "api",
            "-X", "POST",
            "/repos/{owner}/{repo}/dispatches",
            "--input", "-",
        ],
        input=payload,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if verbose:
            print(f"[WARNING] Failed to trigger review workflow: {result.stderr}")
        return False

    if verbose:
        print("[INFO] Post-mortem review workflow triggered")
    return True


def format_discussion_body(
    trades: list[TradeContext],
    analysis: str,
    is_paper: bool,
) -> str:
    """Format the GitHub Discussion body."""
    mode = "PAPER" if is_paper else "LIVE"

    # Calculate summary stats
    total_pnl = sum(t.realized_pnl for t in trades if t.realized_pnl is not None)
    sell_trades = [t for t in trades if t.side == "sell" and t.realized_pnl is not None]
    winning = sum(1 for t in sell_trades if t.realized_pnl > 0)
    losing = sum(1 for t in sell_trades if t.realized_pnl < 0)
    win_rate = (winning / len(sell_trades) * 100) if sell_trades else 0

    body = f"""## Trade Post-Mortem Analysis

**Mode:** {mode}
**Trades Analyzed:** {len(trades)}
**Date Range:** {trades[-1].executed_at.strftime('%Y-%m-%d')} to {trades[0].executed_at.strftime('%Y-%m-%d')}

### Summary Statistics
- **Total Realized P&L:** ${total_pnl:,.2f}
- **Winning Trades:** {winning}
- **Losing Trades:** {losing}
- **Win Rate:** {win_rate:.1f}%

---

## Claude Analysis

{analysis}

---

## Trade Details

<details>
<summary>Click to expand full trade data</summary>

"""

    for ctx in trades:
        body += format_trade_context(ctx)
        body += "\n\n"

    body += "</details>\n"

    return body


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Determine mode (paper is default, --live overrides)
    is_paper = not args.live

    # Validate database path BEFORE resolving (prevents symlink bypass)
    if ".." in args.db:
        print(f"[ERROR] Invalid database path: {args.db}")
        print("[HINT] Path traversal (..) not allowed")
        return 1

    db_path = Path(args.db).resolve()  # Now safe to resolve
    if not db_path.is_absolute():
        print(f"[ERROR] Invalid database path: {args.db}")
        print("[HINT] Use absolute paths")
        return 1
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return 1

    # Connect to database
    if args.verbose:
        print(f"[INFO] Connecting to {db_path}...")
    session = get_session(str(db_path))

    try:
        # Parse and validate date range
        start_date = None
        end_date = None
        try:
            if args.start:
                start_date = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
            if args.end:
                end_date = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

            if start_date and end_date and end_date < start_date:
                print("[ERROR] End date must be after start date")
                return 1
        except ValueError as e:
            print(f"[ERROR] Invalid date format: {e}")
            print("[HINT] Use YYYY-MM-DD format (e.g., 2024-12-01)")
            return 1

        # Determine limit (default: 1 trade)
        limit = args.last if args.last else (None if args.start else 1)

        # Fetch trades with context
        if args.verbose:
            print("[INFO] Fetching trades...")

        trade_ids = [args.trade_id] if args.trade_id else None
        trades = fetch_trades_with_context(
            session=session,
            is_paper=is_paper,
            trade_ids=trade_ids,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )

        if not trades:
            print("[ERROR] No trades found matching criteria")
            return 1

        mode = "paper" if is_paper else "live"
        print(f"[INFO] Found {len(trades)} {mode} trades to analyze")

        # Format prompt - include source root for code references
        source_root = Path(args.source_root)
        prompt = format_analysis_prompt(
            trades,
            is_paper,
            source_root,
            db_path=args.db,
            include_source=not args.no_source,
        )

        if args.print_context:
            print("\n" + "=" * 60)
            print("CONTEXT (would be sent to Claude):")
            print("=" * 60)
            print(prompt)
            return 0

        # Invoke Claude CLI
        print("[INFO] Analyzing trades with Claude...")
        analysis = invoke_claude(prompt, verbose=args.verbose)

        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS:")
        print("=" * 60)
        print(analysis)

        # Create GitHub Discussion
        if args.create_discussion:
            print("\n[INFO] Creating GitHub Discussion...")

            date_str = trades[0].executed_at.strftime("%Y-%m-%d")
            title = f"[Post-Mortem] {mode.title()} Trade Analysis - {date_str}"

            body = format_discussion_body(trades, analysis, is_paper)

            try:
                discussion_url = create_github_discussion(
                    title=title,
                    body=body,
                    verbose=args.verbose,
                )
                print(f"[SUCCESS] GitHub Discussion created: {discussion_url}")

                # Trigger the post-mortem review workflow
                trigger_postmortem_review(
                    discussion_url=discussion_url,
                    title=title,
                    body=body,
                    verbose=args.verbose,
                )
            except Exception as e:
                print(f"[WARNING] Failed to create GitHub Discussion: {e}")
                print("[INFO] Analysis completed but discussion not created")

        return 0

    except subprocess.TimeoutExpired:
        print("[ERROR] Claude CLI timed out after 15 minutes")
        return 1
    except FileNotFoundError as e:
        if "claude" in str(e):
            print(
                "[ERROR] Claude CLI not found. Ensure 'claude' is installed and in PATH"
            )
        elif "gh" in str(e):
            print(
                "[ERROR] GitHub CLI not found. Ensure 'gh' is installed and authenticated"
            )
        else:
            print(f"[ERROR] File not found: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Analysis cancelled")
        return 130
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
