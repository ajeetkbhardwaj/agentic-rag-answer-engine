"""
Web Search Agent (LangChain-style wrapper).

Provides a simple `WebAgent.search(query, top_k)` API that attempts to
use configured search providers (SerpAPI, Tavily) and falls back to a
lightweight HTTP fetch when those aren't configured.

Returned results are normalized into a list of citation-like dicts:
{ 'title', 'url', 'snippet', 'source_type', 'confidence' }
"""
from typing import List, Dict, Any, Optional
import logging
import requests
from config import config

logger = logging.getLogger(__name__)


class WebAgent:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or config.web_search.get("provider", "tavily") if hasattr(config, 'web_search') else "tavily"
        self.serpapi_key = config.SERPAPI_API_KEY
        self.tavily_key = config.TAVILY_API_KEY

    def _search_serpapi(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using SerpAPI (if API key present)."""
        if not self.serpapi_key:
            return []
        try:
            params = {
                'engine': 'google',
                'q': query,
                'num': top_k,
                'api_key': self.serpapi_key,
            }
            resp = requests.get('https://serpapi.com/search', params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get('organic_results', [])[:top_k]:
                results.append({
                    'title': item.get('title') or item.get('link'),
                    'url': item.get('link'),
                    'snippet': item.get('snippet') or item.get('snippet_text') or '',
                    'source_type': 'internet',
                    'confidence': 0.8,
                })
            return results
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")
            return []

    def _search_tavily(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using Tavily SDK if available and configured."""
        if not self.tavily_key:
            return []
        try:
            import tavily
            client = tavily.Client(api_key=self.tavily_key)
            hits = client.search(query, limit=top_k)
            results = []
            for h in hits:
                results.append({
                    'title': h.get('title', '') or h.get('url', ''),
                    'url': h.get('url'),
                    'snippet': h.get('snippet', ''),
                    'source_type': 'internet',
                    'confidence': float(h.get('score', 0.7)),
                })
            return results
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return []

    def _simple_http_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback: query SerpAPI-free results using DuckDuckGo's HTML (simple scrape).

        Note: This is a minimal fallback and shouldn't be used for heavy production.
        """
        try:
            # Use DuckDuckGo HTML for a simple result set
            resp = requests.get('https://duckduckgo.com/html/', params={'q': query}, timeout=10, headers={'User-Agent': 'aisys-bot/1.0'})
            resp.raise_for_status()
            text = resp.text
            # Very simple parsing to find links/snippets
            results = []
            # Use naive split heuristics to extract top_k results
            parts = text.split('<a rel="nofollow" class="result__a"')
            for p in parts[1:top_k+1]:
                # extract href
                try:
                    href_part = p.split('href="', 1)[1]
                    url = href_part.split('"', 1)[0]
                except Exception:
                    url = ''
                # extract title
                try:
                    title = p.split('>')[1].split('<', 1)[0]
                except Exception:
                    title = url
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': '',
                    'source_type': 'internet',
                    'confidence': 0.5,
                })
            return results
        except Exception as e:
            logger.warning(f"Simple HTTP search failed: {e}")
            return []

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Public search method returning normalized results."""
        # Try configured providers in order
        results = []
        if self.serpapi_key:
            results = self._search_serpapi(query, top_k)
            if results:
                return results
        if self.tavily_key:
            results = self._search_tavily(query, top_k)
            if results:
                return results

        # Final fallback
        return self._simple_http_search(query, top_k)


# Convenience instance
_default_web_agent = None

def get_web_agent() -> WebAgent:
    global _default_web_agent
    if _default_web_agent is None:
        _default_web_agent = WebAgent()
    return _default_web_agent
