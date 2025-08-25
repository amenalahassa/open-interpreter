"""
Anthropic provider for OS mode using the native Anthropic API.
"""

import os
from typing import Any, Dict, List, Optional
from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex
from anthropic.types.beta import BetaMessageParam
from .base_provider import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's native API."""
    
    BETA_FLAG = "computer-use-2024-10-22"
    
    def __init__(self, api_key: Optional[str] = None, provider_type: str = "anthropic"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: API key for Anthropic
            provider_type: Type of Anthropic provider ("anthropic", "bedrock", or "vertex")
        """
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.provider_type = provider_type
        
        if provider_type == "anthropic":
            self.client = Anthropic(api_key=self.api_key)
            self.default_model = "claude-3-5-sonnet-20241022"
        elif provider_type == "bedrock":
            self.client = AnthropicBedrock()
            self.default_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        elif provider_type == "vertex":
            self.client = AnthropicVertex()
            self.default_model = "claude-3-5-sonnet-v2@20241022"
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    async def create_completion(
        self,
        messages: List[BetaMessageParam],
        model: str,
        system: str,
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
        **kwargs
    ):
        """Create a completion using Anthropic's API."""
        
        # Use native Anthropic API with beta flag
        raw_response = self.client.beta.messages.with_raw_response.create(
            model=model or self.default_model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            betas=[self.BETA_FLAG],
            **kwargs
        )
        
        response = raw_response.parse()
        response_headers = raw_response.headers
        
        # Return the response in a format compatible with the sampling loop
        yield {
            "response": response,
            "headers": response_headers
        }
    
    def supports_vision(self) -> bool:
        """Anthropic models support vision."""
        return True
    
    def supports_tools(self) -> bool:
        """Anthropic models support tool calling."""
        return True
    
    def get_model_name(self) -> str:
        """Get the default model name."""
        return self.default_model