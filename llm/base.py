"""
Base LLM abstraction for provider-agnostic access.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2048):
        """
        Initialize LLM.
        
        Args:
            model: Model name/identifier
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Generate completion for a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def stream_complete(self, prompt: str):
        """
        Stream completion for a prompt.
        
        Args:
            prompt: Input prompt
            
        Yields:
            Text chunks
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
