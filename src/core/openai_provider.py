import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI
from src.core.llm_provider import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Extraction from OpenAI response
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "openai"
        }

    def generate_response(self, system_prompt: str, history: list, prompt: str) -> dict:
        """
        Generate a response with full conversation history.
        Overrides base class to properly handle multi-turn conversations.
        Returns dict with content, usage, latency_ms, and provider.
        """
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "openai"
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
