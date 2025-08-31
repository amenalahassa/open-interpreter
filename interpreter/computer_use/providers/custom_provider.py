"""
Custom provider for OS mode that uses Open Interpreter's configured LLM.
"""

import base64
import json
import traceback
from typing import Any, Dict, List, Optional, Callable
from anthropic.types.beta import BetaMessageParam, BetaMessage, BetaTextBlockParam
from .base_provider import BaseProvider
from .model_adapter import (
    convert_anthropic_to_openai_messages,
    convert_openai_to_anthropic_response,
    convert_computer_tool_to_function,
    parse_tool_calls_from_response
)


class CustomProvider(BaseProvider):
    """Provider that uses Open Interpreter's configured LLM."""
    
    def __init__(self, interpreter):
        """
        Initialize custom provider with an interpreter instance.
        
        Args:
            interpreter: Open Interpreter instance with configured LLM
        """
        super().__init__()
        self.interpreter = interpreter
        self.llm = interpreter.llm
        
        # Check capabilities
        self._check_capabilities()
    
    def _check_capabilities(self):
        """Check if the configured model has required capabilities."""
        # Load the model configuration if not already loaded
        if not self.llm._is_loaded:
            self.llm.load()
        
        # if not self.supports_vision():
        #     raise ValueError(
        #         f"Model {self.llm.model} does not support vision. "
        #         "OS mode requires a vision-capable model."
        #     )
        #
        # if not self.supports_tools():
        #     print(
        #         f"Warning: Model {self.llm.model} may not support tool calling. "
        #         "OS mode will attempt to work with text-based function calling."
        #     )
    
    async def create_completion(
        self,
        messages: List[BetaMessageParam],
        model: str,
        system: str,
        tools: List[Dict[str, Any]],
        max_tokens: int = 4096,
        **kwargs
    ):
        """Create a completion using Open Interpreter's LLM."""
        
        # Convert Anthropic format to Open Interpreter's internal format
        oi_messages = []
        
        # Add system message
        if system:
            oi_messages.append({
                "role": "system",
                "type": "message",
                "content": system
            })
        
        # Convert messages
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", [])
            
            if isinstance(content, str):
                oi_messages.append({
                    "role": role,
                    "type": "message", 
                    "content": content
                })
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        
                        if block_type == "text":
                            oi_messages.append({
                                "role": role,
                                "type": "message",
                                "content": block.get("text", "")
                            })
                        elif block_type == "image":
                            # Handle image messages
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                oi_messages.append({
                                    "role": role,
                                    "type": "image",
                                    "format": "base64",
                                    "content": source.get("data", "")
                                })
                        elif block_type == "tool_result":
                            # Handle tool results
                            result_content = block.get("content", [])
                            for result_block in result_content:
                                if isinstance(result_block, dict):
                                    if result_block.get("type") == "text":
                                        oi_messages.append({
                                            "role": "computer",
                                            "type": "console",
                                            "format": "output",
                                            "content": result_block.get("text", "")
                                        })
        
        try:
            # Use Open Interpreter's internal run method
            response_messages = []
            for response in self.llm.run(oi_messages):
                if response.get("role") == "assistant":
                    response_messages.append(response)
            
            # Convert response to Anthropic format
            if response_messages:
                last_response = response_messages[-1]
                content_text = last_response.get("content", "")
                
                # Check for tool calls in the response
                tool_calls = parse_tool_calls_from_response(content_text)
                
                # Create Anthropic response
                anthropic_response = convert_openai_to_anthropic_response(
                    content=content_text,
                    tool_calls=tool_calls,
                    model=model or self.llm.model
                )
                
                yield {
                    "response": anthropic_response,
                    "headers": {}
                }
            else:
                # No response, create empty response
                anthropic_response = convert_openai_to_anthropic_response(
                    content="I apologize, but I didn't generate a response.",
                    tool_calls=None,
                    model=model or self.llm.model
                )
                
                yield {
                    "response": anthropic_response,
                    "headers": {}
                }
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error calling LLM: {e}")
            traceback.print_exc()
            
            # Return an error response in Anthropic format
            error_response = convert_openai_to_anthropic_response(
                content=f"Error: {str(e)}",
                tool_calls=None,
                model=model or self.llm.model
            )
            yield {
                "response": error_response,
                "headers": {}
            }
    
    def supports_vision(self) -> bool:
        """Check if the configured model supports vision."""
        return bool(self.llm.supports_vision)
    
    def supports_tools(self) -> bool:
        """Check if the configured model supports tool calling."""
        return bool(self.llm.supports_functions)
    
    def get_model_name(self) -> str:
        """Get the configured model name."""
        return self.llm.model