"""
Model adapter for converting between Anthropic and OpenAI message formats.
"""

import base64
import json
from typing import Any, Dict, List, Optional, Union
from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaImageBlockParam,
    BetaToolUseBlockParam,
    BetaToolResultBlockParam,
    BetaContentBlockParam
)


def convert_anthropic_to_openai_messages(
    messages: List[BetaMessageParam],
    system: str
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic message format to OpenAI format.
    
    Args:
        messages: List of Anthropic format messages
        system: System prompt to prepend
        
    Returns:
        List of OpenAI format messages
    """
    openai_messages = []
    
    # Add system message
    if system:
        openai_messages.append({
            "role": "system",
            "content": system
        })
    
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", [])
        
        # Handle different content types
        if isinstance(content, str):
            # Simple text content
            openai_messages.append({
                "role": role,
                "content": content
            })
        elif isinstance(content, list):
            # Complex content with multiple blocks
            openai_content = []
            tool_calls = []
            
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    
                    if block_type == "text":
                        openai_content.append({
                            "type": "text",
                            "text": block.get("text", "")
                        })
                    
                    elif block_type == "image":
                        # Convert Anthropic image format to OpenAI format
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            image_data = source.get("data", "")
                            media_type = source.get("media_type", "image/png")
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            })
                    
                    elif block_type == "tool_use":
                        # Convert tool use to OpenAI function call format
                        tool_calls.append({
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {}))
                            }
                        })
                    
                    elif block_type == "tool_result":
                        # Convert tool result to text content
                        result_content = block.get("content", [])
                        for result_block in result_content:
                            if isinstance(result_block, dict) and result_block.get("type") == "text":
                                openai_content.append({
                                    "type": "text",
                                    "text": f"Tool result: {result_block.get('text', '')}"
                                })
            
            # Build the OpenAI message
            if openai_content or tool_calls:
                msg_dict = {"role": role}
                
                if openai_content:
                    # If there's only text content, flatten it
                    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
                        msg_dict["content"] = openai_content[0]["text"]
                    else:
                        msg_dict["content"] = openai_content
                
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                
                openai_messages.append(msg_dict)
    
    return openai_messages


def convert_openai_to_anthropic_response(
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    model: str = "unknown"
) -> BetaMessage:
    """
    Convert OpenAI response format to Anthropic BetaMessage format.
    
    Args:
        content: Text content from the response
        tool_calls: Optional list of tool calls from the response
        model: Model name
        
    Returns:
        Anthropic BetaMessage object
    """
    anthropic_content = []
    
    # Add text content if present
    if content:
        anthropic_content.append(
            BetaTextBlockParam(type="text", text=content)
        )
    
    # Add tool calls if present
    if tool_calls:
        for call in tool_calls:
            if call.get("type") == "function" or "function" in call:
                func = call.get("function", call)
                
                # Parse arguments if they're a string
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                anthropic_content.append(
                    BetaToolUseBlockParam(
                        type="tool_use",
                        id=call.get("id", f"tool_{len(anthropic_content)}"),
                        name=func.get("name", "computer"),
                        input=args
                    )
                )
    
    # Create the BetaMessage
    return BetaMessage(
        id=f"msg_{model[:8]}",
        content=anthropic_content,
        role="assistant",
        model=model,
        stop_reason="stop_sequence" if not tool_calls else "tool_use",
        stop_sequence=None,
        type="message",
        usage={
            "input_tokens": 0,
            "output_tokens": 0
        }
    )


def convert_computer_tool_to_function(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Anthropic computer tool definition to OpenAI function format.
    
    Args:
        tool: Anthropic tool definition
        
    Returns:
        OpenAI function definition
    """
    return {
        "name": "computer",
        "description": "Control the computer screen, mouse, and keyboard",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "screenshot",
                        "left_click",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "type",
                        "key",
                        "mouse_move",
                        "left_click_drag",
                        "cursor_position"
                    ],
                    "description": "The action to perform"
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "The [x, y] coordinate for mouse actions"
                },
                "text": {
                    "type": "string",
                    "description": "Text to type or key combination to press"
                }
            },
            "required": ["action"]
        }
    }


def parse_tool_calls_from_response(response_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse tool calls from a text response (for models without native tool calling).
    
    This is a fallback for models that don't support native tool calling.
    It looks for function calls in a specific format in the text.
    
    Args:
        response_text: The text response from the model
        
    Returns:
        List of parsed tool calls or None if no calls found
    """
    tool_calls = []
    
    # Look for patterns like: computer.action(params)
    # This is a simple implementation that can be enhanced
    import re
    
    # Pattern to match function calls
    pattern = r'computer\.(\w+)\((.*?)\)'
    matches = re.findall(pattern, response_text)
    
    for action, params in matches:
        tool_call = {
            "type": "function",
            "function": {
                "name": "computer",
                "arguments": json.dumps({"action": action})
            }
        }
        
        # Try to parse parameters
        if params:
            try:
                # Simple parameter parsing
                if "," in params:
                    parts = params.split(",")
                    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
                        # Coordinates
                        tool_call["function"]["arguments"] = json.dumps({
                            "action": action,
                            "coordinate": [int(parts[0].strip()), int(parts[1].strip())]
                        })
                elif params.startswith('"') and params.endswith('"'):
                    # Text parameter
                    tool_call["function"]["arguments"] = json.dumps({
                        "action": action,
                        "text": params.strip('"')
                    })
            except:
                pass
        
        tool_calls.append(tool_call)
    
    return tool_calls if tool_calls else None