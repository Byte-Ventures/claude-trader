"""
Web search tool for AI models.

Provides web search capability via DuckDuckGo for use as an
OpenRouter tool during market analysis.
"""

import asyncio
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Tool definition for OpenRouter API
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current cryptocurrency news, market analysis, or sentiment. Use this to find recent information not included in the pre-fetched data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'Bitcoin ETF news today', 'BTC whale activity', 'crypto market sentiment')"
                }
            },
            "required": ["query"]
        }
    }
}


async def execute_web_search(query: str, max_results: int = 3) -> str:
    """
    Execute web search using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Formatted string with search results for AI consumption
    """
    try:
        # Import here to handle missing dependency gracefully
        from ddgs import DDGS
    except ImportError:
        logger.error("ddgs_not_installed")
        return "Web search unavailable: ddgs package not installed"

    try:
        # Run sync search in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _sync_search(query, max_results)
        )

        if not results:
            return f"No results found for: {query}"

        # Format results for AI
        formatted = [f"Search results for '{query}':\n"]
        titles = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")[:200]
            href = result.get("href", "")
            formatted.append(f"{i}. {title}\n   {body}\n   Source: {href}\n")
            titles.append(title[:60])

        logger.info(
            "web_search_executed",
            query=query,
            results=len(results),
            titles=titles,
        )
        return "\n".join(formatted)

    except Exception as e:
        logger.error("web_search_failed", query=query, error=str(e))
        return f"Search failed: {str(e)[:100]}"


def _sync_search(query: str, max_results: int) -> list[dict]:
    """Synchronous search wrapper."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(
            query,
            max_results=max_results,
            safesearch="moderate",
        ))
    return results


async def handle_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """
    Handle tool calls from OpenRouter response.

    Args:
        tool_calls: List of tool call objects from API response

    Returns:
        List of tool results to send back to API
    """
    results = []

    for call in tool_calls:
        tool_id = call.get("id", "")
        function = call.get("function", {})
        name = function.get("name", "")
        arguments = function.get("arguments", {})

        if name == "web_search":
            # Parse arguments (may be JSON string)
            if isinstance(arguments, str):
                import json
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"query": arguments}

            query = arguments.get("query", "")
            if query:
                content = await execute_web_search(query)
            else:
                content = "Error: No search query provided"

            results.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": content,
            })
        else:
            # Unknown tool
            results.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"Unknown tool: {name}",
            })

    return results


def get_tools_for_model(model: str, enabled: bool = True) -> Optional[list[dict]]:
    """
    Get appropriate tools for a model.

    Some models don't support tool calling well, so we may need
    to skip tools for certain models.

    Args:
        model: Model identifier (e.g., "google/gemini-2.5-flash")
        enabled: Global enable/disable flag

    Returns:
        List of tool definitions or None if disabled/unsupported
    """
    if not enabled:
        return None

    # Models known to not support tools well (add as discovered)
    unsupported_models = []

    model_lower = model.lower()
    for unsupported in unsupported_models:
        if unsupported in model_lower:
            logger.debug("tools_disabled_for_model", model=model)
            return None

    return [WEB_SEARCH_TOOL]
