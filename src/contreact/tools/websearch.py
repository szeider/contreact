"""Web search tools using Tavily API."""

import os
from urllib.parse import urlparse
from langchain_core.tools import tool

# Lazy-loaded Tavily client
_tavily_client = None


def _get_tavily_client():
    """Get or create Tavily client (lazy initialization)."""
    global _tavily_client
    if _tavily_client is None:
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError("Tavily requires 'tavily-python'. Install with: uv add tavily-python")
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# Tool descriptions - injected into agent's system prompt
WEB_SEARCH_DESCRIPTION = """
**web_search**: Search the web for current information.

Use this tool when you need to:
- Find up-to-date information beyond your training data
- Research topics, news, or recent developments
- Verify facts or find sources

Returns a list of relevant results with titles, snippets, and URLs.
""".strip()


EXTRACT_CONTENT_DESCRIPTION = """
**extract_content**: Extract full content from a specific URL.

Use this tool when you need to:
- Read the full content of a webpage found via web_search
- Get detailed information from a specific source
- Access article text, documentation, or other web content

**Warning**: This tool often fails due to JavaScript rendering, paywalls, or
anti-bot measures. Prefer using web_search results directly when possible.

Returns the extracted text content from the URL.
""".strip()


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily API.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with titles, snippets, and URLs
    """
    query = (query or "").strip()
    if not query:
        return "Error: Search query cannot be empty."

    max_results = max(1, min(int(max_results), 10))

    try:
        client = _get_tavily_client()
        response = client.search(query=query, max_results=max_results)

        raw_results = response.get('results', []) if isinstance(response, dict) else []
        results = []
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            title = r.get('title') or 'No title'
            content = r.get('content') or 'No content'
            url = r.get('url', '')
            results.append(f"**{title}**\n{content}\nURL: {url}")

        if not results:
            return f"No results found for: {query}"

        return f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def extract_content(url: str) -> str:
    """Extract full content from a specific URL using Tavily.

    Args:
        url: The URL to extract content from

    Returns:
        The extracted text content from the URL
    """
    url = (url or "").strip()
    if not url:
        return "Error: URL cannot be empty."

    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        return "Error: URL must start with http:// or https://"

    try:
        client = _get_tavily_client()
        response = client.extract(urls=[url])  # API expects list

        content = ""
        if isinstance(response, dict):
            # Response structure: {"results": [{"url": "...", "raw_content": "..."}]}
            results = response.get('results', [])
            if results and isinstance(results, list) and isinstance(results[0], dict):
                first = results[0]
                content = first.get('raw_content') or first.get('content') or first.get('text') or ""
            else:
                content = response.get('raw_content') or response.get('content') or ""
        else:
            content = str(response) if response else ""

        content = content.strip()
        if not content:
            return f"No content extracted from: {url}"

        return f"Content from {url}:\n\n{content}"

    except Exception as e:
        return f"Content extraction failed: {str(e)}"
