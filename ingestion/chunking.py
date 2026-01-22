"""
Document chunking and text processing.
"""
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str


class DocumentChunker:
    """Handle document chunking strategies."""
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        document_id: str = "",
        source: str = ""
    ) -> List[TextChunk]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            document_id: ID of source document
            source: Source document path/name
            
        Returns:
            List of TextChunk objects
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        chunks = []
        start = 0
        chunk_counter = 0
        
        while start < len(text):
            # Get chunk end position
            end = min(start + chunk_size, len(text))
            
            # Try to end at sentence boundary if not at end of text
            if end < len(text):
                # Look for last sentence boundary
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                last_boundary = max(last_period, last_newline)
                
                if last_boundary > start:
                    end = last_boundary + 1
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunk = TextChunk(
                    content=chunk_content,
                    metadata={
                        "document_id": document_id,
                        "source": source,
                        "chunk_index": chunk_counter,
                        "start_position": start,
                        "end_position": end,
                    },
                    chunk_id=f"{document_id}_chunk_{chunk_counter}",
                    source=source,
                )
                chunks.append(chunk)
                chunk_counter += 1
            
            # Move to next chunk with overlap
            start = end - chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks
    
    @staticmethod
    def chunk_by_sections(
        text: str,
        document_id: str = "",
        source: str = ""
    ) -> List[TextChunk]:
        """
        Chunk text by natural sections (headers, paragraphs).
        
        Args:
            text: Text to chunk
            document_id: ID of source document
            source: Source document path/name
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                chunk = TextChunk(
                    content=para,
                    metadata={
                        "document_id": document_id,
                        "source": source,
                        "chunk_index": idx,
                    },
                    chunk_id=f"{document_id}_section_{idx}",
                    source=source,
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} section chunks from {source}")
        return chunks
    
    @staticmethod
    def chunk_by_tokens(
        text: str,
        max_tokens: int = 500,
        document_id: str = "",
        source: str = "",
    ) -> List[TextChunk]:
        """
        Chunk text by approximate token count (simple word-based estimation).
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (estimated as words)
            document_id: ID of source document
            source: Source document path/name
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        words = text.split()
        chunk_words = []
        chunk_counter = 0
        
        for word in words:
            chunk_words.append(word)
            
            # Estimate tokens (rough: words * 1.3)
            if len(chunk_words) >= max_tokens / 1.3:
                content = ' '.join(chunk_words).strip()
                if content:
                    chunk = TextChunk(
                        content=content,
                        metadata={
                            "document_id": document_id,
                            "source": source,
                            "chunk_index": chunk_counter,
                            "word_count": len(chunk_words),
                        },
                        chunk_id=f"{document_id}_token_chunk_{chunk_counter}",
                        source=source,
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                chunk_words = []
        
        # Add remaining words
        if chunk_words:
            content = ' '.join(chunk_words).strip()
            chunk = TextChunk(
                content=content,
                metadata={
                    "document_id": document_id,
                    "source": source,
                    "chunk_index": chunk_counter,
                    "word_count": len(chunk_words),
                },
                chunk_id=f"{document_id}_token_chunk_{chunk_counter}",
                source=source,
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} token-based chunks from {source}")
        return chunks
