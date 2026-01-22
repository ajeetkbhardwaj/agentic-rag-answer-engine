"""
Mock LLM implementation for offline testing.
"""
from typing import List, Dict, Any
from llm.base import BaseLLM


class MockLLM(BaseLLM):
    def __init__(self, model: str = "mock-model", temperature: float = 0.0, max_tokens: int = 256):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

    def complete(self, prompt: str) -> str:
        return f"MOCK_REPLY: {prompt[:200]}"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Echo last user message as a safe mock
        for m in reversed(messages):
            if m.get('role') == 'user':
                return f"MOCK_CHAT_REPLY: {m.get('content')[:400]}"
        return "MOCK_CHAT_REPLY: (no user message)"

    def embed(self, text: str) -> List[float]:
        # deterministic mock embedding
        vec = [float((ord(c) % 10) / 10.0) for c in text[:128]]
        # pad / trim to 32 dims
        if len(vec) < 32:
            vec += [0.0] * (32 - len(vec))
        return vec[:32]

    def stream_complete(self, prompt: str):
        # yield in two chunks
        mid = len(prompt) // 2
        yield f"MOCK_STREAM: {prompt[:mid]}"
        yield f"MOCK_STREAM: {prompt[mid:mid+200]}"
