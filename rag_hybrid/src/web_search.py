"""Small server-side adapter for Tavily web search."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchConfigurationError(RuntimeError):
    """Raised when the optional search provider has not been configured."""


class WebSearchError(RuntimeError):
    """Raised when the search provider cannot return usable results."""


def search_web(query: str) -> dict:
    """Search the web and return a short answer with linked source snippets."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise WebSearchConfigurationError(
            "Web search needs TAVILY_API_KEY in .env. Add the key and restart the app."
        )

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
                "include_raw_content": False,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        raise WebSearchError("Could not connect to the web search provider.") from exc

    if response.status_code == 401:
        raise WebSearchConfigurationError(
            "Tavily rejected TAVILY_API_KEY. Check the key in .env and restart the app."
        )
    if response.status_code == 429:
        raise WebSearchError("Tavily search rate limit reached. Try again shortly.")
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise WebSearchError("The web search provider returned an invalid response.") from exc

    sources = []
    for result in payload.get("results", []):
        url = str(result.get("url") or "").strip()
        parsed_url = urlsplit(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            continue
        sources.append(
            {
                "title": str(result.get("title") or url),
                "url": url,
                "content": str(result.get("content") or "").strip(),
                "score": result.get("score"),
            }
        )

    answer = str(payload.get("answer") or "").strip()
    if not answer:
        answer = (
            f"I found {len(sources)} relevant web page(s). See the source links below."
            if sources
            else "I could not find relevant web pages for that query."
        )
    return {"answer": answer, "sources": sources}
