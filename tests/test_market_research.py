"""Tests for market research module."""

import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timezone

from src.ai.market_research import (
    fetch_crypto_news,
    NewsItem,
    _cache,
    COINTELEGRAPH_SOURCE_NAME,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


# Sample RSS XML for testing
# Note: Using Dec 21, 2023 as test fixture date - sufficiently in the past to be
# clearly historical test data, avoiding confusion with current dates
SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Cointelegraph</title>
    <item>
      <title>Bitcoin breaks $100K as institutional demand surges</title>
      <link>https://cointelegraph.com/news/bitcoin-100k</link>
      <pubDate>Thu, 21 Dec 2023 08:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Ethereum ETF sees record inflows</title>
      <link>https://cointelegraph.com/news/eth-etf</link>
      <pubDate>Thu, 21 Dec 2023 07:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Fed signals crypto-friendly policy shift</title>
      <link>https://cointelegraph.com/news/fed-crypto</link>
      <pubDate>Thu, 21 Dec 2023 06:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""


class TestFetchCryptoNews:
    """Tests for fetch_crypto_news function."""

    @pytest.mark.asyncio
    async def test_successful_fetch_returns_news_items(self):
        """Test successful RSS fetch returns list of NewsItem."""
        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=SAMPLE_RSS_XML,
        ):
            result = await fetch_crypto_news(limit=5)

        assert len(result) == 3
        assert all(isinstance(item, NewsItem) for item in result)
        assert result[0].title == "Bitcoin breaks $100K as institutional demand surges"
        assert result[0].source == COINTELEGRAPH_SOURCE_NAME
        assert "cointelegraph.com" in result[0].url

    @pytest.mark.asyncio
    async def test_invalid_xml_returns_empty_list(self):
        """Test invalid XML returns empty list without crashing."""
        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value="<not valid xml",
        ):
            result = await fetch_crypto_news(limit=5)

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_rss_feed_returns_empty_list(self):
        """Test RSS feed with no items returns empty list."""
        empty_rss = """<?xml version="1.0"?>
        <rss version="2.0"><channel><title>Empty</title></channel></rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=empty_rss,
        ):
            result = await fetch_crypto_news(limit=5)

        assert result == []

    @pytest.mark.asyncio
    async def test_limit_parameter_respected(self):
        """Test limit parameter limits number of items returned."""
        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=SAMPLE_RSS_XML,
        ):
            result = await fetch_crypto_news(limit=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_value(self):
        """Test cached response is returned without API call."""
        cached_items = [
            NewsItem(
                title="Cached news",
                source="Cache",
                url="",
                published_at=datetime.now(timezone.utc),
            )
        ]
        _cache["crypto_news"] = cached_items

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
        ) as mock_fetch:
            result = await fetch_crypto_news(limit=5)

        mock_fetch.assert_not_called()
        assert result == cached_items

    @pytest.mark.asyncio
    async def test_exception_returns_empty_list(self):
        """Test exception during fetch returns empty list."""
        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            result = await fetch_crypto_news(limit=5)

        assert result == []

    @pytest.mark.asyncio
    async def test_missing_elements_handled_gracefully(self):
        """Test items with missing date element are parsed with current time."""
        incomplete_rss = """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>News with no date</title>
              <link>https://cointelegraph.com/news/test</link>
            </item>
          </channel>
        </rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=incomplete_rss,
        ):
            result = await fetch_crypto_news(limit=5)

        assert len(result) == 1
        assert result[0].title == "News with no date"
        assert result[0].url == "https://cointelegraph.com/news/test"
        assert result[0].source == COINTELEGRAPH_SOURCE_NAME
        # Should use current time as fallback
        assert result[0].published_at is not None

    @pytest.mark.asyncio
    async def test_pubdate_parsing(self):
        """Test RFC 2822 date parsing works correctly."""
        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=SAMPLE_RSS_XML,
        ):
            result = await fetch_crypto_news(limit=1)

        assert len(result) == 1
        assert result[0].published_at.year == 2023
        assert result[0].published_at.month == 12
        assert result[0].published_at.day == 21

    @pytest.mark.asyncio
    async def test_xxe_attack_prevented(self):
        """Test XXE attack is prevented by defusedxml."""
        # XXE payload attempting to read /etc/passwd
        xxe_payload = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
        <rss version="2.0">
          <channel>
            <item>
              <title>&xxe;</title>
              <link>https://cointelegraph.com/news/test</link>
              <pubDate>Thu, 21 Dec 2023 08:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=xxe_payload,
        ):
            result = await fetch_crypto_news(limit=5)

        # defusedxml should prevent parsing and return empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_title_skipped(self):
        """Test items with empty titles are skipped."""
        rss_with_empty_title = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title></title>
              <link>https://cointelegraph.com/news/test1</link>
              <pubDate>Thu, 21 Dec 2023 08:00:00 +0000</pubDate>
            </item>
            <item>
              <title>Valid Title</title>
              <link>https://cointelegraph.com/news/test2</link>
              <pubDate>Thu, 21 Dec 2023 07:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=rss_with_empty_title,
        ):
            result = await fetch_crypto_news(limit=5)

        assert len(result) == 1
        assert result[0].title == "Valid Title"

    @pytest.mark.asyncio
    async def test_invalid_url_skipped(self):
        """Test items with URLs not from cointelegraph.com are skipped."""
        rss_with_invalid_url = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Malicious Link</title>
              <link>https://evil.com/phishing</link>
              <pubDate>Thu, 21 Dec 2023 08:00:00 +0000</pubDate>
            </item>
            <item>
              <title>Valid Link</title>
              <link>https://cointelegraph.com/news/valid</link>
              <pubDate>Thu, 21 Dec 2023 07:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=rss_with_invalid_url,
        ):
            result = await fetch_crypto_news(limit=5)

        assert len(result) == 1
        assert result[0].title == "Valid Link"
        assert result[0].url == "https://cointelegraph.com/news/valid"

    @pytest.mark.asyncio
    async def test_whitespace_only_title_skipped(self):
        """Test items with whitespace-only titles are skipped."""
        rss_with_whitespace_title = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>   </title>
              <link>https://cointelegraph.com/news/test1</link>
              <pubDate>Thu, 21 Dec 2023 08:00:00 +0000</pubDate>
            </item>
            <item>
              <title>Valid Title</title>
              <link>https://cointelegraph.com/news/test2</link>
              <pubDate>Thu, 21 Dec 2023 07:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=rss_with_whitespace_title,
        ):
            result = await fetch_crypto_news(limit=5)

        assert len(result) == 1
        assert result[0].title == "Valid Title"

    @pytest.mark.asyncio
    async def test_long_title_truncated(self):
        """Test titles longer than MAX_TITLE_LENGTH are truncated."""
        from src.ai.market_research import MAX_TITLE_LENGTH

        long_title = "A" * 150  # Title longer than MAX_TITLE_LENGTH (100)
        rss_with_long_title = f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>{long_title}</title>
              <link>https://cointelegraph.com/news/test</link>
              <pubDate>Thu, 21 Dec 2023 08:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=rss_with_long_title,
        ):
            result = await fetch_crypto_news(limit=1)

        assert len(result) == 1
        assert len(result[0].title) == MAX_TITLE_LENGTH
        assert result[0].title == "A" * MAX_TITLE_LENGTH

    @pytest.mark.asyncio
    async def test_http_url_rejected(self):
        """Test items with HTTP URLs (not HTTPS) are rejected."""
        rss_with_http_url = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>HTTP Link (Insecure)</title>
              <link>http://cointelegraph.com/news/insecure</link>
              <pubDate>Thu, 21 Dec 2023 08:00:00 +0000</pubDate>
            </item>
            <item>
              <title>HTTPS Link (Secure)</title>
              <link>https://cointelegraph.com/news/secure</link>
              <pubDate>Thu, 21 Dec 2023 07:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        with patch(
            "src.ai.market_research._fetch_crypto_news_request",
            new_callable=AsyncMock,
            return_value=rss_with_http_url,
        ):
            result = await fetch_crypto_news(limit=5)

        # Only HTTPS link should be included
        assert len(result) == 1
        assert result[0].title == "HTTPS Link (Secure)"
        assert result[0].url == "https://cointelegraph.com/news/secure"


class TestCoinTelegraphRSSIntegration:
    """Live RSS feed integration tests (skipped in CI)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Set to False to run live test locally
        reason="Live RSS test - run manually to verify connectivity"
    )
    async def test_live_rss_returns_news(self):
        """Test actual CoinTelegraph RSS returns valid news items.

        Run this test manually to verify RSS connectivity:
        pytest tests/test_market_research.py::TestCoinTelegraphRSSIntegration -v -s
        """
        _cache.clear()

        result = await fetch_crypto_news(limit=5)

        assert len(result) > 0, "RSS should return at least one news item"
        assert all(isinstance(item, NewsItem) for item in result)
        assert all(item.title for item in result), "All items should have titles"
        assert all(item.url for item in result), "All items should have URLs"

        print("\nLive RSS results:")
        for item in result:
            print(f"\n  Title: {item.title}")
            print(f"  Source: {item.source}")
            print(f"  Date: {item.published_at}")
