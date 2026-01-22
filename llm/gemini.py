"""
Google Gemini LLM provider implementation.
"""
import logging
from typing import Optional, List, Dict, Any

import google.generativeai as genai
from llm.base import BaseLLM

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Google API key
            model: Model name (gemini-1.5-pro, gemini-1.5-flash, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        """
        super().__init__(model, temperature, max_tokens)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
            ),
        )
        logger.info(f"Initialized Gemini LLM with model: {model}")
    
    def complete(self, prompt: str) -> str:
        """
        Generate completion for a prompt using Gemini.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini completion: {e}")
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
            # Convert messages to Gemini format
            chat_history = []
            for msg in messages[:-1]:  # All but last are history
                chat_history.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [msg["content"]],
                })
            
            # Start chat session
            chat = self.client.start_chat(history=chat_history)
            
            # Send the last message
            response = chat.send_message(messages[-1]["content"])
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini chat: {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using Gemini Embedding API.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error in Gemini embedding: {e}")
            raise
    
    def stream_complete(self, prompt: str):
        """
        Stream completion for a prompt.
        
        Args:
            prompt: Input prompt
            
        Yields:
            Text chunks
        """
        try:
            response = self.client.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error in Gemini streaming: {e}")
            raise
