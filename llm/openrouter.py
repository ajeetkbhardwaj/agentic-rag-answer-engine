"""
OpenRouter LLM provider implementation for multi-model support.
"""
import logging
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI
from llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM implementation supporting multiple models."""
    
    SUPPORTED_MODELS = {
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4o": "openai/gpt-4o",
        "claude-3-opus": "anthropic/claude-3-opus",
        "claude-3-sonnet": "anthropic/claude-3-sonnet",
        "mistral-large": "mistralai/mistral-large",
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize OpenRouter LLM.
        
        Args:
            api_key: OpenRouter API key
            model: Model name or OpenRouter model ID
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        """
        super().__init__(model, temperature, max_tokens)
        
        # Map model name if using shorthand
        actual_model = self.SUPPORTED_MODELS.get(model, model)
        
        self.client = ChatOpenAI(
            model=actual_model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"Initialized OpenRouter LLM with model: {actual_model}")
    
    def complete(self, prompt: str) -> str:
        """
        Generate completion for a prompt using OpenRouter.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error in OpenRouter completion: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Generated response
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            # Convert to LangChain message format
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            response = self.client.invoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error in OpenRouter chat: {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings (using OpenRouter embedding endpoint or fallback).
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.client.openai_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model="openai/text-embedding-3-small",
            )
            result = embeddings.embed_query(text)
            return result
        except Exception as e:
            logger.warning(f"Embedding via OpenRouter failed, using fallback: {e}")
            # Fallback to a mock embedding for demo purposes
            return [0.0] * 1536
    
    def stream_complete(self, prompt: str):
        """
        Stream completion for a prompt.
        
        Args:
            prompt: Input prompt
            
        Yields:
            Text chunks
        """
        try:
            from langchain_core.messages import HumanMessage
            
            response = self.client.stream([HumanMessage(content=prompt)])
            for chunk in response:
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error in OpenRouter streaming: {e}")
            raise
