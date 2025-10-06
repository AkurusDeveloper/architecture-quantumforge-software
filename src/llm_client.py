"""LLM client for generating responses"""
from __future__ import annotations
import os
from typing import List, Dict, Optional
from openai import OpenAI


class LLMClient:
    """Client for interacting with LLM providers"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        max_tokens: int = 800,
        temperature: float = 0.2
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider (openai, anthropic, local)
            model: Model name
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
        """
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Please set it in .env file"
                )
            self.client = OpenAI(api_key=api_key)
            print(f"✅ OpenAI client initialized: {model}")
        else:
            raise NotImplementedError(
                f"Provider '{provider}' not implemented. "
                "Currently supported: openai"
            )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            
        Returns:
            Generated text response
        """
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"❌ Error calling OpenAI API: {e}")
                return f"[Error: {str(e)}]"
        
        raise NotImplementedError(f"Provider '{self.provider}' not implemented")
    
    def get_info(self) -> Dict[str, any]:
        """Get client information."""
        return {
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
