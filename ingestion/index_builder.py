"""
Vector index building and management using LlamaIndex.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import TextNode, BaseNode
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import config
from ingestion.chunking import DocumentChunker, TextChunk

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build and manage vector indices using LlamaIndex."""
    
    def __init__(self):
        """Initialize index builder with configured embedding model."""
        self.embedding_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=config.GEMINI_API_KEY,
        )
        self.vector_store = self._init_vector_store()
        self.index = None
    
    def _init_vector_store(self):
        """Initialize vector store (Qdrant)."""
        if config.VECTOR_DB_TYPE == "qdrant":
            # Create Qdrant client
            client = QdrantClient(path=config.QDRANT_PATH)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="documents"
            )
            logger.info(f"Initialized Qdrant vector store at {config.QDRANT_PATH}")
            return vector_store
        else:
            raise ValueError(f"Unsupported vector DB: {config.VECTOR_DB_TYPE}")
    
    def add_document(
        self,
        document_content: str,
        document_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a single document to the index.
        
        Args:
            document_content: The document text
            document_id: Unique identifier for the document
            source: Source file name/path
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Chunk the document
        chunks = DocumentChunker.chunk_text(
            text=document_content,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            document_id=document_id,
            source=source,
        )
        
        # Convert chunks to LlamaIndex nodes
        nodes = self._chunks_to_nodes(chunks, metadata or {})
        
        # Add to index
        if self.index is None:
            from llama_index.core import StorageContext, VectorStoreIndex
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embedding_model,
            )
        else:
            self.index.insert_nodes(nodes)
        
        logger.info(f"Added document {document_id} with {len(nodes)} nodes")
        return document_id
    
    def add_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple documents to the index.
        
        Args:
            documents: List of document dicts with 'content', 'id', 'source'
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        for doc in documents:
            try:
                doc_id = self.add_document(
                    document_content=doc['content'],
                    document_id=doc.get('id', Path(doc['source']).stem),
                    source=doc['source'],
                    metadata=doc.get('metadata', {})
                )
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to add document {doc['source']}: {e}")
        
        return doc_ids
    
    def _chunks_to_nodes(
        self,
        chunks: List[TextChunk],
        base_metadata: Dict[str, Any]
    ) -> List[BaseNode]:
        """Convert TextChunk objects to LlamaIndex nodes."""
        nodes = []
        
        for chunk in chunks:
            metadata = {**base_metadata, **chunk.metadata}
            
            node = TextNode(
                text=chunk.content,
                metadata=metadata,
                id_=chunk.chunk_id,
            )
            nodes.append(node)
        
        return nodes
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (uses config default if None)
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved documents with metadata
        """
        if self.index is None:
            logger.warning("Index is empty, no results to retrieve")
            return []
        
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        # Create retriever
        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            embed_model=self.embedding_model,
        )
        
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        
        # Format results
        results = []
        for node in nodes:
            results.append({
                "content": node.get_content(),
                "metadata": node.metadata,
                "score": node.score if hasattr(node, 'score') else None,
                "source": node.metadata.get('source', 'unknown'),
            })
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:100]}")
        return results
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query the index and get synthesized response.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Query response with sources
        """
        if self.index is None:
            return {"response": "", "sources": []}
        
        query_engine = self.index.as_query_engine(
            top_k=top_k or config.RETRIEVAL_TOP_K,
            embed_model=self.embedding_model,
        )
        
        response = query_engine.query(query)
        
        return {
            "response": str(response),
            "sources": self._extract_sources(response),
        }
    
    def _extract_sources(self, response: Any) -> List[Dict[str, str]]:
        """Extract source information from response."""
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                sources.append({
                    "source": node.metadata.get('source', 'unknown'),
                    "content_preview": node.get_content()[:200],
                })
        return sources
    
    def save_index(self, path: str) -> None:
        """Save index to disk."""
        if self.index:
            self.index.storage_context.persist(persist_dir=path)
            logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load index from disk."""
        from llama_index.core import load_index_from_storage, StorageContext
        
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=path
        )
        self.index = load_index_from_storage(storage_context)
        logger.info(f"Loaded index from {path}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {"status": "empty"}
        
        return {
            "status": "active",
            "embedding_model": "gemini-embedding",
            "vector_store": config.VECTOR_DB_TYPE,
        }
