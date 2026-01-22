"""
Document RAG Agent wrapping the ingestion IndexBuilder (LlamaIndex).

Provides:
- ingest_file
- ingest_directory
- retrieve
- query
- get_index_stats

This is a lightweight agent API intended for orchestration layers.
"""
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from ingestion.index_builder import IndexBuilder
from ingestion.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class DocRAGAgent:
    """Document-centric RAG agent using LlamaIndex via IndexBuilder."""

    def __init__(self, index_builder: Optional[IndexBuilder] = None):
        self.index_builder = index_builder or IndexBuilder()

    def ingest_file(self, file_path: str, document_id: Optional[str] = None) -> str:
        """Load a single file and add to the index.

        Returns the document_id used.
        """
        doc = DocumentLoader.load(file_path)
        doc_id = document_id or Path(file_path).stem
        self.index_builder.add_document(
            document_content=doc["content"],
            document_id=doc_id,
            source=doc["filename"],
            metadata={"original_path": doc["file_path"]},
        )
        logger.info(f"Ingested file {file_path} as {doc_id}")
        return doc_id

    def ingest_directory(self, directory: str) -> List[str]:
        """Load all supported files in a directory and add to the index."""
        docs = DocumentLoader.load_multiple(directory)
        prepared = []
        for d in docs:
            prepared.append({
                "content": d["content"],
                "id": Path(d["file_path"]).stem,
                "source": d["file_path"],
                "metadata": {"filename": d["filename"]},
            })
        ids = self.index_builder.add_documents_batch(prepared)
        logger.info(f"Ingested directory {directory} with {len(ids)} documents")
        return ids

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top_k document passages relevant to the query."""
        return self.index_builder.retrieve(query=query, top_k=top_k)

    def query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Run a query through the index's query engine and return synthesized answer + sources."""
        return self.index_builder.query(query=query, top_k=top_k)

    def get_index_stats(self) -> Dict[str, Any]:
        return self.index_builder.get_index_stats()


# Convenience singleton for simple imports
_default_agent = None

def get_doc_rag_agent() -> DocRAGAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = DocRAGAgent()
    return _default_agent
