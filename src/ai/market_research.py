"""
Market research module for fetching online data.

Fetches crypto news, on-chain data, and other market research
from free APIs with caching to avoid rate limits.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import structlog
from cachetools import TTLCache
from defusedxml.ElementTree import fromstring
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = structlog.get_logger(__name__)

# RSS feed configuration
COINTELEGRAPH_RSS_URL = "https://cointelegraph.com/rss"
COINTELEGRAPH_SOURCE_NAME = "CoinTelegraph"
MAX_TITLE_LENGTH = 100  # Prevent excessively long titles in AI prompts

# Cache storage with automatic TTL expiration and bounded size
# Default: 15 min TTL, max 100 entries to prevent memory leaks
_cache: TTLCache = TTLCache(maxsize=100, ttl=900)


def set_cache_ttl(minutes: int) -> None:
    """
    Set cache TTL in minutes.

    Note: This recreates the cache, clearing existing entries.
    """
    global _cache
    _cache = TTLCache(maxsize=100, ttl=minutes * 60)


def _get_cached(key: str) -> Optional[Any]:
    """Get cached value if not expired."""
    return _cache.get(key)


def _set_cached(key: str, value: Any) -> None:
    """Cache a value with automatic TTL expiration."""
    _cache[key] = value


@dataclass
class NewsItem:
    """
    Single news article.

    Note: The sentiment field is always set to "neutral" because we do NOT
    perform hardcoded sentiment analysis. The AI model analyzing the market
    should determine sentiment from the news titles and context. This field
    exists for display formatting only (shows "~" in prompts).
    """
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment: str = "neutral"  # Always neutral - AI determines actual sentiment


@dataclass
class OnChainData:
    """Bitcoin on-chain metrics."""
    hashrate_eh: float  # Exahashes per second
    difficulty: float
    mempool_size_mb: float
    mempool_tx_count: int
    avg_fee_sat_vb: float  # Satoshis per virtual byte
    blocks_24h: int


@dataclass
class MarketResearch:
    """Combined market research data."""
    news: list[NewsItem] = field(default_factory=list)
    onchain: Optional[OnChainData] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: list[str] = field(default_factory=list)


async def _fetch_crypto_news_request() -> str:
    """Execute the actual HTTP request to CoinTelegraph RSS feed.

    CoinTelegraph provides quality crypto news via RSS feed.
    Returns raw XML string for parsing.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            COINTELEGRAPH_RSS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; claude-trader/1.0)"},
        )
        response.raise_for_status()
        return response.text


async def fetch_crypto_news(limit: int = 5) -> list[NewsItem]:
    """
    Fetch latest crypto news from CoinTelegraph RSS feed.

    Source: https://cointelegraph.com/rss
    Format: RSS 2.0 XML
    Rate limit: None (public RSS feed)

    Note: This function expects RSS 2.0 format. If CoinTelegraph changes their
    RSS format or URL, monitor logs for 'crypto_news_no_items' warnings which
    could indicate a feed format change.
    """
    cached = _get_cached("crypto_news")
    if cached is not None:
        logger.debug("crypto_news_cache_hit")
        return cached

    try:
        logger.info("fetching_crypto_news", url=COINTELEGRAPH_RSS_URL)

        # Retry with exponential backoff: 1s, 2s, 4s
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            reraise=True,
        )
        async def fetch_with_retry():
            return await _fetch_crypto_news_request()

        xml_content = await fetch_with_retry()

        # Parse RSS XML using defusedxml to prevent XXE attacks
        try:
            root = fromstring(xml_content)
        except Exception as e:
            # defusedxml raises various exceptions for malicious XML
            logger.warning("crypto_news_xml_parse_error", error=str(e))
            # Cache empty result to prevent retry storms during outages
            _set_cached("crypto_news", [])
            return []

        # Find all <item> elements in RSS feed
        items = root.findall(".//item")
        if not items:
            logger.warning("crypto_news_no_items", message="RSS feed contained no items")
            return []

        # Note: We intentionally extract only title, link, and pubDate from RSS items.
        # The <description> field is not extracted because:
        # 1. Keeps prompts concise (avoids overwhelming AI with long article summaries)
        # 2. Titles alone provide sufficient signal for sentiment/trend analysis
        # 3. AI can determine sentiment from headlines without full descriptions
        news_items = []
        for item in items[:limit]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_date_el = item.find("pubDate")

            # Validate title content
            title_text = (title_el.text or "").strip() if title_el is not None else ""
            if not title_text:
                logger.warning("crypto_news_empty_title", url=link_el.text if link_el is not None else "unknown")
                continue  # Skip items with empty titles

            # Validate URL - use urlparse for robust domain validation
            # This prevents edge cases like "https://cointelegraph.com.evil.com/"
            url_text = (link_el.text or "").strip() if link_el is not None else ""
            try:
                parsed = urlparse(url_text)
                # Verify HTTPS scheme and cointelegraph.com domain
                if parsed.scheme != "https" or parsed.netloc != "cointelegraph.com":
                    logger.warning("crypto_news_invalid_url", url=url_text, title=title_text[:40])
                    continue  # Skip items with invalid URLs
            except ValueError:
                logger.warning("crypto_news_malformed_url", url=url_text, title=title_text[:40])
                continue

            # Parse publication date (RFC 2822 format)
            # Default to current time if parsing fails to ensure news items remain usable
            # rather than being discarded entirely (timestamps used for display only)
            published_at = datetime.now(timezone.utc)
            if pub_date_el is not None and pub_date_el.text:
                try:
                    published_at = parsedate_to_datetime(pub_date_el.text)
                except (ValueError, TypeError):
                    # Date parsing failed - use current time as fallback
                    logger.warning(
                        "crypto_news_date_parse_failed",
                        raw_date=pub_date_el.text,
                        title=title_text[:40]
                    )

            news_items.append(NewsItem(
                title=title_text[:MAX_TITLE_LENGTH],
                source=COINTELEGRAPH_SOURCE_NAME,
                url=url_text,
                published_at=published_at,
                # sentiment defaults to "neutral" - AI analyzes actual sentiment
            ))

        _set_cached("crypto_news", news_items)
        logger.info(
            "crypto_news_fetched",
            count=len(news_items),
            titles=[n.title[:40] for n in news_items[:3]],
        )
        return news_items

    except httpx.TimeoutException:
        logger.warning("crypto_news_timeout", attempts=3, timeout_sec=15)
        # Cache empty result to prevent retry storms during outages
        # Uses same TTL as successful results (15 min default)
        _set_cached("crypto_news", [])
        return []
    except httpx.HTTPStatusError as e:
        logger.warning("crypto_news_http_error", status=e.response.status_code, url=str(e.request.url))
        # Cache empty result to prevent retry storms during outages
        _set_cached("crypto_news", [])
        return []
    except httpx.HTTPError as e:
        logger.warning("crypto_news_network_error", error=str(e))
        # Cache empty result to prevent retry storms during outages
        _set_cached("crypto_news", [])
        return []
    except Exception as e:
        # Unexpected errors (XML parsing bugs, programming errors, etc.)
        logger.error("crypto_news_unexpected_error", error=str(e), type=type(e).__name__)
        # Cache empty result to prevent retry storms during outages
        _set_cached("crypto_news", [])
        return []


async def fetch_onchain_data() -> Optional[OnChainData]:
    """
    Fetch Bitcoin on-chain data from Blockchain.info and Mempool.space.

    APIs:
    - https://api.blockchain.info/stats (general stats)
    - https://mempool.space/api/v1/fees/recommended (fee estimates)
    - https://mempool.space/api/mempool (mempool stats)

    Rate limits: ~200 requests/5min for Blockchain.info, unlimited for Mempool.space
    """
    cached = _get_cached("onchain_data")
    if cached is not None:
        logger.debug("onchain_data_cache_hit")
        return cached

    try:
        logger.info("fetching_onchain_data", sources=["blockchain.info", "mempool.space"])
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Fetch in parallel
            blockchain_task = client.get("https://api.blockchain.info/stats")
            mempool_task = client.get("https://mempool.space/api/mempool")
            fees_task = client.get("https://mempool.space/api/v1/fees/recommended")

            blockchain_resp, mempool_resp, fees_resp = await asyncio.gather(
                blockchain_task, mempool_task, fees_task,
                return_exceptions=True
            )

        # Parse Blockchain.info stats
        hashrate_eh = 0.0
        difficulty = 0.0
        blocks_24h = 0

        if not isinstance(blockchain_resp, Exception):
            blockchain_resp.raise_for_status()
            bc_data = blockchain_resp.json()
            # Hash rate is in GH/s, convert to EH/s (1 EH = 1e9 GH)
            hashrate_eh = bc_data.get("hash_rate", 0) / 1_000_000_000
            difficulty = bc_data.get("difficulty", 0)
            blocks_24h = bc_data.get("n_blocks_mined", 0)

        # Parse Mempool.space data
        mempool_size_mb = 0.0
        mempool_tx_count = 0
        avg_fee = 0.0

        if not isinstance(mempool_resp, Exception):
            mempool_resp.raise_for_status()
            mp_data = mempool_resp.json()
            mempool_size_mb = mp_data.get("vsize", 0) / 1_000_000
            mempool_tx_count = mp_data.get("count", 0)

        if not isinstance(fees_resp, Exception):
            fees_resp.raise_for_status()
            fees_data = fees_resp.json()
            # Use "halfHourFee" as a reasonable average
            avg_fee = fees_data.get("halfHourFee", 0)

        onchain = OnChainData(
            hashrate_eh=round(hashrate_eh, 1),
            difficulty=difficulty,
            mempool_size_mb=round(mempool_size_mb, 1),
            mempool_tx_count=mempool_tx_count,
            avg_fee_sat_vb=avg_fee,
            blocks_24h=blocks_24h,
        )

        _set_cached("onchain_data", onchain)
        logger.info(
            "onchain_data_fetched",
            hashrate_eh=onchain.hashrate_eh,
            mempool_mb=onchain.mempool_size_mb,
            avg_fee=onchain.avg_fee_sat_vb,
        )
        return onchain

    except Exception as e:
        logger.error("onchain_data_fetch_failed", error=str(e) or type(e).__name__)
        return None


async def fetch_market_research() -> MarketResearch:
    """
    Fetch all market research data in parallel.

    Returns MarketResearch with available data and any errors.
    """
    errors = []

    # Fetch all data in parallel
    news_task = fetch_crypto_news()
    onchain_task = fetch_onchain_data()

    news, onchain = await asyncio.gather(
        news_task, onchain_task,
        return_exceptions=True
    )

    # Handle exceptions
    if isinstance(news, Exception):
        errors.append(f"News fetch failed: {str(news)[:50]}")
        news = []
    if isinstance(onchain, Exception):
        errors.append(f"On-chain fetch failed: {str(onchain)[:50]}")
        onchain = None

    return MarketResearch(
        news=news,
        onchain=onchain,
        fetched_at=datetime.now(timezone.utc),
        errors=errors,
    )


def format_research_for_prompt(research: MarketResearch) -> str:
    """
    Format market research for inclusion in AI prompts.

    Returns a formatted string with news and on-chain data.
    """
    sections = []

    # News section
    if research.news:
        news_lines = ["Recent Bitcoin News:"]
        for item in research.news[:5]:
            time_ago = _time_ago(item.published_at)
            sentiment_emoji = {"positive": "+", "negative": "-", "neutral": "~"}[item.sentiment]
            news_lines.append(f"  [{sentiment_emoji}] \"{item.title}\" ({item.source}) - {time_ago}")
        sections.append("\n".join(news_lines))

    # On-chain section
    if research.onchain:
        oc = research.onchain
        onchain_lines = [
            "On-Chain Data:",
            f"  - Hashrate: {oc.hashrate_eh} EH/s",
            f"  - Mempool: {oc.mempool_size_mb} MB ({oc.mempool_tx_count:,} txs)",
            f"  - Avg Fee: {oc.avg_fee_sat_vb} sat/vB",
            f"  - Blocks (24h): {oc.blocks_24h}",
        ]
        sections.append("\n".join(onchain_lines))

    if not sections:
        return "(No research data available)"

    return "\n\n".join(sections)


def _time_ago(dt: datetime) -> str:
    """Format datetime as relative time string."""
    delta = datetime.now(timezone.utc) - dt
    hours = delta.total_seconds() / 3600

    if hours < 1:
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes}m ago"
    elif hours < 24:
        return f"{int(hours)}h ago"
    else:
        days = int(hours / 24)
        return f"{days}d ago"
