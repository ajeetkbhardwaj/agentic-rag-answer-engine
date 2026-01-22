"""
Simple orchestrator that wires RouterAgent, DocRAGAgent, WebAgent, FusionAgent, and AnswerAgent.

Provides `run_query(query, metadata)` which returns a structured response with
decision, evidence, and final answer+citations.
"""
from typing import Dict, Any
import logging

from agents.router_agent import RouterAgent
from agents.doc_rag_agent import get_doc_rag_agent
from agents.web_agent import get_web_agent
from agents.fusion_agent import get_fusion_agent
from agents.answer_agent import get_answer_agent

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.router = RouterAgent()
        self.doc_agent = get_doc_rag_agent()
        self.web_agent = get_web_agent()
        self.fusion = get_fusion_agent()
        self.answer_agent = get_answer_agent()

    def run_query(self, query: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        decision = self.router.decide(query, metadata)

        doc_results = []
        web_results = []

        # Document retrieval if requested
        if decision.get('use_documents'):
            try:
                # Use retrieve to get passages; if no index, handle gracefully
                doc_results = self.doc_agent.retrieve(query, top_k=metadata.get('top_k'))
            except Exception as e:
                logger.warning(f"Document retrieval failed: {e}")
                doc_results = []

        # Web retrieval if requested
        if decision.get('use_web'):
            try:
                web_results = self.web_agent.search(query, top_k=metadata.get('top_k', 5))
            except Exception as e:
                logger.warning(f"Web retrieval failed: {e}")
                web_results = []

        # Fuse evidence
        fused = self.fusion.fuse(doc_results=doc_results, web_results=web_results)

        # Generate final answer
        final = self.answer_agent.generate_answer(fused, user_query=query)

        return {
            'decision': decision,
            'doc_count': len(doc_results),
            'web_count': len(web_results),
            'evidence': fused.get('evidence', []),
            'answer': final.get('answer'),
            'sources_text': final.get('sources_text'),
            'citations': final.get('citations'),
        }


# Simple CLI demo
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, required=True)
    args = parser.parse_args()

    orch = Orchestrator()
    out = orch.run_query(args.query, metadata={'has_uploaded_docs': False})
    print('Decision:', out['decision'])
    print('\nAnswer:\n', out['answer'])
    print('\nSources:\n', out['sources_text'])
