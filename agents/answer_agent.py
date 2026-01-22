"""
Answer Generator Agent: formats citations and produces final grounded answer.

Exposes `AnswerAgent.generate_answer(fused_evidence, user_query)` which
returns a dict with `answer` (string), `citations` (list), and `sources`.
"""
from typing import List, Dict, Any
import logging

from llm.factory import get_llm

logger = logging.getLogger(__name__)


class AnswerAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def _format_citations(self, citations: List[Dict[str, Any]]) -> str:
        lines = []
        for i, c in enumerate(citations, start=1):
            src = c.get('source') or c.get('url') or 'unknown'
            stype = c.get('source_type', 'unknown')
            lines.append(f"[{i}] {src} ({stype})")
        return "\n".join(lines)

    def generate_answer(self, fused: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Generate a final answer from fused evidence.

        Args:
            fused: Output from FusionAgent.fuse()
            user_query: Original user query (for context)

        Returns:
            { 'answer': str, 'citations': list, 'sources_text': str }
        """
        citations = fused.get('evidence', [])
        prompt_lines = [
            "You are an assistant that must not hallucinate. Answer concisely using only the provided evidence.",
            f"User question: {user_query}",
            "Evidence:"
        ]
        for i, c in enumerate(citations[:15], start=1):
            claim = c.get('claim', '')
            src = c.get('source', '')
            prompt_lines.append(f"[{i}] {claim} (source: {src})")

        prompt_lines.append("\nProduce a short answer (3-6 sentences) and attach inline numeric citations like [1], [2]. Then list the referenced sources and their metadata.")
        prompt = "\n".join(prompt_lines)

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that must cite sources and avoid adding new facts."},
                {"role": "user", "content": prompt},
            ]
            resp = self.llm.chat(messages)
            sources_text = self._format_citations(citations)
            return {"answer": resp, "citations": citations, "sources_text": sources_text}
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Fallback: simple aggregation
            short = ' '.join([c.get('claim', '') for c in citations[:3]])
            sources_text = self._format_citations(citations)
            return {"answer": short, "citations": citations, "sources_text": sources_text}


# Convenience
_default_answer_agent = None

def get_answer_agent() -> AnswerAgent:
    global _default_answer_agent
    if _default_answer_agent is None:
        _default_answer_agent = AnswerAgent()
    return _default_answer_agent
