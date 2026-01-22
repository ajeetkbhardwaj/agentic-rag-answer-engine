"""
Configuration management for the AI Answering System.
"""
import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class."""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
    STORAGE_DIR = BASE_DIR / "storage"
    UPLOAD_DIR = STORAGE_DIR / "uploads"
    
    # Create necessary directories
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Vector DB Configuration
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "qdrant")
    QDRANT_PATH = os.getenv("QDRANT_PATH", str(STORAGE_DIR / "qdrant_storage"))
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Web Search Configuration
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    
    # Application Settings
    PORT = int(os.getenv("PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # RAG Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))
    
    # CrewAI Settings
    CREW_VERBOSE = os.getenv("CREW_VERBOSE", "true").lower() == "true"
    CREW_MEMORY = os.getenv("CREW_MEMORY", "true").lower() == "true"
    
    @classmethod
    def load_yaml(cls) -> dict:
        """Load configuration from YAML file."""
        if cls.CONFIG_PATH.exists():
            with open(cls.CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical configuration."""
        if cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        if cls.LLM_PROVIDER == "openrouter" and not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set")


# Global config instance
config = Config()
yaml_config = config.load_yaml()
