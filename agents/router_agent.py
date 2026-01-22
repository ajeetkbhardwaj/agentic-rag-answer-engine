"""
Query Router Agent (LangGraph-style) - lightweight decision logic.

This module exposes a `RouterAgent` class with a `decide` method that
chooses which retrieval paths to use: 'documents', 'web', or 'both'.
It is written to be easily embedded into a LangGraph node or called
directly from orchestration code.
"""
from typing import Dict, Any
import re


class RouterAgent:
    """Simple router to decide retrieval sources for a query.

    The decision is based on:
    - Presence of uploaded documents (metadata['has_uploaded_docs'])
    - Query keywords indicating recent or trending info
    - Query scope words that imply internal-only
    """

    WEB_KEYWORDS = [
        r"latest",
        r"recent",
        r"202\d",
        r"research",
        r"report",
        r"news",
        r"trends",
        r"stat(e|istics)",
    ]

    INTERNAL_KEYWORDS = [
        r"our",
        r"internal",
        r"company",
        r"client",
        r"confidential",
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _looks_like_web_query(self, query: str) -> bool:
        q = query.lower()
        for kw in self.WEB_KEYWORDS:
            if re.search(kw, q):
                return True
        return False

    def _looks_like_internal_only(self, query: str) -> bool:
        q = query.lower()
        for kw in self.INTERNAL_KEYWORDS:
            if re.search(kw, q):
                return True
        return False

    def decide(self, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide which agents to call for the given query.

        Args:
            query: The user query string
            metadata: Runtime metadata (e.g., {'has_uploaded_docs': True})

        Returns:
            A decision dict: { 'use_documents': bool, 'use_web': bool, 'reason': str }
        """
        has_docs = bool(metadata.get("has_uploaded_docs", False))

        # If explicitly internal-sounding, prefer documents only
        if has_docs and self._looks_like_internal_only(query):
            return {"use_documents": True, "use_web": False, "reason": "Internal scope detected"}

        # If query looks like it needs the web (latest/recent), prefer web or both
        web_needed = self._looks_like_web_query(query)

        if has_docs and web_needed:
            return {"use_documents": True, "use_web": True, "reason": "Combined: internal + recent/external"}
        if has_docs and not web_needed:
            return {"use_documents": True, "use_web": False, "reason": "Documents sufficient"}
        if not has_docs and web_needed:
            return {"use_documents": False, "use_web": True, "reason": "No docs, web search required"}

        # Default: use both if in doubt
        return {"use_documents": has_docs, "use_web": True, "reason": "Default to both when uncertain"}


# Small helper for LangGraph-style node if needed

def langgraph_node(query: str, metadata: Dict[str, Any]):
    agent = RouterAgent()
    return agent.decide(query, metadata)
