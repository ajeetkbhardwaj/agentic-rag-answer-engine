"""
Evidence Fusion Agent (CrewAI-style lightweight orchestration).

Responsibilities:
- Merge evidence from document RAG and web results
- Deduplicate and rank evidence
- Produce citation objects
- Optionally synthesize a short combined summary using the LLM
"""
from typing import List, Dict, Any
import logging
from collections import OrderedDict

from llm.factory import get_llm
from config import config

logger = logging.getLogger(__name__)


class FusionAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # priority weights (documents > web)
        self.source_priority = {
            'document': 1.2,
            'internet': 1.0,
        }

    def _normalize_snippet(self, text: str) -> str:
        return ' '.join(text.strip().split())[:500]

    def _make_citation(self, claim: str, source_type: str, source: str, url: str = None, confidence: float = 0.5) -> Dict[str, Any]:
        return {
            'claim': claim,
            'source_type': source_type,
            'source': source,
            'url': url,
            'confidence': float(confidence),
        }

    def fuse(self, doc_results: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse document and web evidence into ranked citation list.

        Args:
            doc_results: List of dicts from DocRAGAgent.retrieve or query
            web_results: List of dicts from WebAgent.search

        Returns:
            { 'evidence': [citation objects], 'summary_prompt': str }
        """
        merged = []
        seen = set()

        # Process document results first (higher priority)
        for d in doc_results or []:
            snippet = self._normalize_snippet(d.get('content', d.get('text', '')))
            key = snippet[:200]
            if key in seen:
                continue
            seen.add(key)
            score = d.get('score') or 0.6
            source = d.get('source', d.get('metadata', {}).get('source', 'internal'))
            citation = self._make_citation(
                claim=snippet,
                source_type='document',
                source=source,
                url=d.get('metadata', {}).get('original_path') or None,
                confidence=min(1.0, score * self.source_priority['document'])
            )
            merged.append(citation)

        # Then web results
        for w in web_results or []:
            snippet = self._normalize_snippet(w.get('snippet', w.get('title', '')))
            key = snippet[:200]
            if key in seen:
                continue
            seen.add(key)
            score = w.get('confidence', 0.5)
            citation = self._make_citation(
                claim=snippet,
                source_type='internet',
                source=w.get('title') or w.get('url'),
                url=w.get('url'),
                confidence=min(1.0, score * self.source_priority['internet'])
            )
            merged.append(citation)

        # Rank evidence by confidence descending
        merged = sorted(merged, key=lambda x: x['confidence'], reverse=True)

        # Build a small prompt for the Answer Generator if needed
        summary_prompt = self._build_summary_prompt(merged)

        return {
            'evidence': merged,
            'summary_prompt': summary_prompt,
        }

    def _build_summary_prompt(self, citations: List[Dict[str, Any]]) -> str:
        lines = ["Synthesize the following evidence into a concise, grounded answer with inline citations:"]
        for i, c in enumerate(citations[:10], start=1):
            src = c['source']
            lines.append(f"[{i}] ({c['source_type']}) {c['claim'][:300]} -- source: {src}")
        lines.append('\nProvide a short answer (3-5 sentences) and list the sources by number.')
        return '\n'.join(lines)

    def synthesize_answer(self, fused: Dict[str, Any], max_tokens: int = 512) -> Dict[str, Any]:
        """Use the configured LLM to synthesize a final answer from fused evidence.

        Returns a dict with 'answer' and 'citations'.
        """
        llm = get_llm()
        prompt = fused.get('summary_prompt')
        if not prompt:
            return {'answer': '', 'citations': fused.get('evidence', [])}

        try:
            # Use a chat-style call if available
            messages = [
                {'role': 'system', 'content': 'You are an expert assistant that must not hallucinate. Use only the provided evidence.'},
                {'role': 'user', 'content': prompt},
            ]
            resp = llm.chat(messages)
            return {'answer': resp, 'citations': fused.get('evidence', [])}
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            # Fallback: simple concatenation
            summary = ' '.join([c['claim'] for c in fused.get('evidence', [])[:3]])
            return {'answer': summary, 'citations': fused.get('evidence', [])}


# Convenience instance
_default_fusion_agent = None

def get_fusion_agent() -> FusionAgent:
    global _default_fusion_agent
    if _default_fusion_agent is None:
        _default_fusion_agent = FusionAgent()
    return _default_fusion_agent
