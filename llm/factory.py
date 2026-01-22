"""
LLM factory for provider-agnostic LLM instantiation.
"""
import logging
from typing import Optional

from config import config
from llm.base import BaseLLM
from llm.gemini import GeminiLLM
from llm.openrouter import OpenRouterLLM
try:
    from llm.mock_llm import MockLLM
except Exception:
    MockLLM = None

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances based on configuration."""
    
    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> BaseLLM:
        """
        Create an LLM instance based on provider and model.
        
        Args:
            provider: LLM provider (gemini, openrouter) - uses config if None
            model: Model name - uses config if None
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            
        Returns:
            BaseLLM instance
        """
        provider = provider or config.LLM_PROVIDER
        model = model or config.LLM_MODEL
        
        logger.info(f"Creating {provider} LLM with model: {model}")
        
        if provider.lower() == "gemini":
            return GeminiLLM(
                api_key=config.GEMINI_API_KEY,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider.lower() == "openrouter":
            return OpenRouterLLM(
                api_key=config.OPENROUTER_API_KEY,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider.lower() == "mock":
            if MockLLM is None:
                raise ValueError("MockLLM not available")
            return MockLLM(model=model, temperature=temperature, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> list:
        """Get list of available LLM providers."""
        return ["gemini", "openrouter"]
    
    @staticmethod
    def get_available_models(provider: str) -> list:
        """Get available models for a provider."""
        if provider.lower() == "gemini":
            return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        elif provider.lower() == "openrouter":
            return list(OpenRouterLLM.SUPPORTED_MODELS.keys())
        return []


# Global LLM instance
_llm_instance = None


def get_llm() -> BaseLLM:
    """Get or create global LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMFactory.create_llm()
    return _llm_instance


def set_llm(llm: BaseLLM) -> None:
    """Set global LLM instance."""
    global _llm_instance
    _llm_instance = llm
