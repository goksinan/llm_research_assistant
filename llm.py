"""
Handles LLM provider integration and switching.
Provides a unified interface for working with different LLM providers
while maintaining consistent response formatting and error handling.
"""
from typing import Dict, Optional, List, Union
import os
import logging
from dataclasses import dataclass
from enum import Enum
import json

import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

@dataclass
class LLMResponse:
    """Standardized response format across providers"""
    content: str
    usage: Dict[str, int]
    model: str
    provider: LLMProvider

class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass

class ProviderError(LLMError):
    """Exception for provider-specific errors"""
    pass

class LLMManager:
    """Manages interactions with different LLM providers"""
    
    def __init__(
        self,
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
        model_config: Optional[Dict] = None
    ):
        """
        Initialize the LLM manager.
        
        Args:
            provider: LLM provider to use ('openai' or 'anthropic')
            model_config: Optional configuration for the LLM model
        
        Raises:
            ValueError: If provider is not supported
            ProviderError: If provider initialization fails
        """
        if isinstance(provider, str):
            try:
                self.provider = LLMProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.provider = provider
            
        self.model_config = model_config or {}
        self._setup_provider()
        
    def _setup_provider(self):
        """Initialize the selected LLM provider"""
        try:
            if self.provider == LLMProvider.OPENAI:
                # Default to GPT-4 if not specified
                self.model = self.model_config.get('model', 'gpt-4o')
                self.client = openai.OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY')
                )
                
            elif self.provider == LLMProvider.ANTHROPIC:
                # Default to Claude 3 Opus if not specified
                self.model = self.model_config.get('model', 'claude-3-opus-20240229')
                self.client = Anthropic(
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                )
                
        except Exception as e:
            raise ProviderError(f"Failed to initialize {self.provider}: {str(e)}")
            
    def generate_response(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a response from the LLM provider.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
            
        Returns:
            LLMResponse object containing the generated response
            
        Raises:
            ProviderError: If the API call fails
        """
        try:
            if self.provider == LLMProvider.OPENAI:
                return self._generate_openai_response(
                    prompt, context, max_tokens, temperature
                )
            elif self.provider == LLMProvider.ANTHROPIC:
                return self._generate_anthropic_response(
                    prompt, context, max_tokens, temperature
                )
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise ProviderError(f"Failed to generate response: {str(e)}")
            
    def _generate_openai_response(
        self,
        prompt: str,
        context: Optional[Dict],
        max_tokens: Optional[int],
        temperature: float,
    ) -> LLMResponse:
        """Generate response using OpenAI's API"""
        messages = []
        
        # Add context if provided
        if context and context.get('history'):
            messages.extend(context['history'])
            
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=response.usage.model_dump(),
            model=self.model,
            provider=self.provider
        )
        
    def _generate_anthropic_response(
        self,
        prompt: str,
        context: Optional[Dict],
        max_tokens: Optional[int],
        temperature: float,
    ) -> LLMResponse:
        """Generate response using Anthropic's API"""
        # Construct the system prompt if context is provided
        system_prompt = ""
        if context and context.get('system_prompt'):
            system_prompt = context['system_prompt']
            
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            usage=response.usage.model_dump(),
            model=self.model,
            provider=self.provider
        )
        
    def get_token_estimate(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        This is a rough estimate based on words/4 for English text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text.split()) // 4
        
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is set and working.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Try a minimal API call
            self.generate_response(
                "test",
                max_tokens=5,
                temperature=0.0
            )
            return True
        except Exception:
            return False
        

if __name__ == "__main__":
    llm = LLMManager()
    llm.validate_api_key()