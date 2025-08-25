"""
Base provider interface for OS mode providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from anthropic.types.beta import BetaMessageParam


class BaseProvider(ABC):
    """Abstract base class for OS mode providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    @abstractmethod
    async def create_completion(
        self,
        messages: List[BetaMessageParam],
        model: str,
        system: str,
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Create a completion with the provider's API.
        
        Args:
            messages: List of messages in conversation
            model: Model identifier
            system: System prompt
            tools: List of available tools
            max_tokens: Maximum tokens for response
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Response chunks from the model
        """
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if the provider supports vision capabilities."""
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the default model name for this provider."""
        pass