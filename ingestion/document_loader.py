"""
Document loading and processing utilities.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handle multi-format document loading."""
    
    SUPPORTED_FORMATS = {
        '.pdf': 'load_pdf',
        '.docx': 'load_docx',
        '.txt': 'load_txt',
        '.csv': 'load_csv',
        '.md': 'load_markdown',
    }
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load PDF file."""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """Load DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            raise
    
    @staticmethod
    def load_csv(file_path: str) -> str:
        """Load CSV file."""
        try:
            import csv
            text = ""
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text += " | ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
            return text
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise
    
    @staticmethod
    def load_markdown(file_path: str) -> str:
        """Load Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading Markdown {file_path}: {e}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> Dict[str, Any]:
        """
        Load document from file path.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with document metadata and content
        """
        path = Path(file_path)
        file_ext = path.suffix.lower()
        
        if file_ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        loader_method = getattr(cls, cls.SUPPORTED_FORMATS[file_ext])
        content = loader_method(file_path)
        
        return {
            "filename": path.name,
            "file_path": str(file_path),
            "file_type": file_ext[1:],
            "content": content,
            "file_size": path.stat().st_size,
        }
    
    @classmethod
    def load_multiple(cls, directory: str) -> List[Dict[str, Any]]:
        """Load multiple documents from directory."""
        documents = []
        path = Path(directory)
        
        for file_path in path.glob("*"):
            if file_path.suffix.lower() in cls.SUPPORTED_FORMATS:
                try:
                    doc = cls.load(str(file_path))
                    documents.append(doc)
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
        
        return documents
