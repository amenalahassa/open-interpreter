"""
Modified loop.py with provider system support.
Based on Anthropic's computer use example.
"""

import asyncio
import json
import os
import platform
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime

try:
    from enum import StrEnum
except ImportError:  # 3.10 compatibility
    from enum import Enum as StrEnum

from typing import Any, List, cast

import requests
from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult
from .providers import BaseProvider, AnthropicProvider, CustomProvider

BETA_FLAG = "computer-use-2024-10-22"

from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rich import print as rich_print
from rich.markdown import Markdown
from rich.rule import Rule

# Add this near the top of the file, with other imports and global variables
messages: List[BetaMessageParam] = []
# Global interpreter instance for custom provider
global_interpreter = None


def print_markdown(message):
    """
    Display markdown message. Works with multiline strings with lots of indentation.
    Will automatically make single line > tags beautiful.
    """

    for line in message.split("\n"):
        line = line.strip()
        if line == "":
            print("")
        elif line == "---":
            rich_print(Rule(style="white"))
        else:
            try:
                rich_print(Markdown(line))
            except UnicodeEncodeError as e:
                # Replace the problematic character or handle the error as needed
                print("Error displaying line:", line)

    if "\n" not in message and message.startswith(">"):
        # Aesthetic choice. For these tags, they need a space below them
        print("")


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    CUSTOM = "custom"  # Add custom provider type


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.CUSTOM: "configured",  # Will use interpreter's configured model
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are an AI assistant with access to a computer running on {"Mac OS" if platform.system() == "Darwin" else platform.system()} with internet access.
* When using your computer function calls, they take a while to run and send back to you. Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>"""

# Update the SYSTEM_PROMPT for Mac OS
if platform.system() == "Darwin":
    SYSTEM_PROMPT += """
<IMPORTANT>
* Open applications using Spotlight by using the computer tool to simulate pressing Command+Space, typing the application name, and pressing Enter.
</IMPORTANT>"""


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_key: str = None,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    interpreter=None,  # Add interpreter parameter for custom provider
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        # BashTool(),
        # EditTool(),
    )
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    while True:
        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, only_n_most_recent_images)

        # Initialize the appropriate provider
        provider_instance = None
        use_provider_system = False
        
        if provider == APIProvider.CUSTOM:
            if not interpreter:
                raise ValueError("Interpreter instance required for custom provider")
            provider_instance = CustomProvider(interpreter)
            use_provider_system = True
        elif provider in [APIProvider.ANTHROPIC, APIProvider.VERTEX, APIProvider.BEDROCK]:
            # We can use the provider system for these too
            if provider == APIProvider.ANTHROPIC:
                provider_instance = AnthropicProvider(api_key=api_key, provider_type="anthropic")
            elif provider == APIProvider.VERTEX:
                provider_instance = AnthropicProvider(provider_type="vertex")
            elif provider == APIProvider.BEDROCK:
                provider_instance = AnthropicProvider(provider_type="bedrock")
            use_provider_system = True

        if use_provider_system and provider_instance:
            # Use the new provider system
            response_data = None
            async for result in provider_instance.create_completion(
                messages=messages,
                model=model,
                system=system,
                tools=tool_collection.to_params(),
                max_tokens=max_tokens
            ):
                response_data = result
                break  # For now, just get the first response
            
            if response_data:
                response = response_data.get("response")
                
                # Process the response
                messages.append(
                    {
                        "role": "assistant",
                        "content": cast(list[BetaContentBlockParam], response.content),
                    }
                )

                tool_result_content: list[BetaToolResultBlockParam] = []
                for content_block in cast(list[BetaContentBlock], response.content):
                    output_callback(content_block)
                    if content_block.type == "tool_use":
                        result = await tool_collection.run(
                            name=content_block.name,
                            tool_input=cast(dict[str, Any], content_block.input),
                        )
                        tool_result_content.append(
                            _make_api_tool_result(result, content_block.id)
                        )
                        tool_output_callback(result, content_block.id)

                if not tool_result_content:
                    # Done!
                    yield {"type": "messages", "messages": messages}
                    break

                messages.append({"content": tool_result_content, "role": "user"})
        else:
            # Use the original implementation for backward compatibility
            if provider == APIProvider.ANTHROPIC:
                client = Anthropic(api_key=api_key)
            elif provider == APIProvider.VERTEX:
                client = AnthropicVertex()
            elif provider == APIProvider.BEDROCK:
                client = AnthropicBedrock()
            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Call the API
            raw_response = client.beta.messages.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=system,
                tools=tool_collection.to_params(),
                betas=["computer-use-2024-10-22"],
                stream=True,
            )

            response_content = []
            current_block = None

            for chunk in raw_response:
                if isinstance(chunk, BetaRawContentBlockStartEvent):
                    current_block = chunk.content_block
                elif isinstance(chunk, BetaRawContentBlockDeltaEvent):
                    if chunk.delta.type == "text_delta":
                        print(f"{chunk.delta.text}", end="", flush=True)
                        yield {"type": "chunk", "chunk": chunk.delta.text}
                        await asyncio.sleep(0)
                        if current_block and current_block.type == "text":
                            current_block.text += chunk.delta.text
                    elif chunk.delta.type == "input_json_delta":
                        print(f"{chunk.delta.partial_json}", end="", flush=True)
                        if current_block and current_block.type == "tool_use":
                            if not hasattr(current_block, "partial_json"):
                                current_block.partial_json = ""
                            current_block.partial_json += chunk.delta.partial_json
                elif isinstance(chunk, BetaRawContentBlockStopEvent):
                    if current_block:
                        if hasattr(current_block, "partial_json"):
                            # Finished a tool call
                            current_block.input = json.loads(current_block.partial_json)
                            delattr(current_block, "partial_json")
                        else:
                            # Finished a message
                            print("\n")
                            yield {"type": "chunk", "chunk": "\n"}
                            await asyncio.sleep(0)
                        response_content.append(current_block)
                        current_block = None

            response = BetaMessage(
                id=str(uuid.uuid4()),
                content=response_content,
                role="assistant",
                model=model,
                stop_reason=None,
                stop_sequence=None,
                type="message",
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": cast(list[BetaContentBlockParam], response.content),
                }
            )

            tool_result_content: list[BetaToolResultBlockParam] = []
            for content_block in cast(list[BetaContentBlock], response.content):
                output_callback(content_block)
                if content_block.type == "tool_use":
                    result = await tool_collection.run(
                        name=content_block.name,
                        tool_input=cast(dict[str, Any], content_block.input),
                    )
                    tool_result_content.append(
                        _make_api_tool_result(result, content_block.id)
                    )
                    tool_output_callback(result, content_block.id)

            if not tool_result_content:
                # Done!
                yield {"type": "messages", "messages": messages}
                break

            messages.append({"content": tool_result_content, "role": "user"})


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 5,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text


async def main(interpreter=None):
    global exit_flag, global_interpreter
    
    # Store interpreter globally if provided
    if interpreter:
        global_interpreter = interpreter
    
    messages: List[BetaMessageParam] = []
    
    # Determine provider and model based on configuration
    provider = APIProvider.ANTHROPIC  # Default
    model = PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Check if we should use custom provider
    if global_interpreter:
        # Configure interpreter from command line arguments
        if "--model" in sys.argv:
            model_idx = sys.argv.index("--model")
            if model_idx + 1 < len(sys.argv):
                global_interpreter.llm.model = sys.argv[model_idx + 1]
        
        if "--api_base" in sys.argv:
            api_base_idx = sys.argv.index("--api_base")
            if api_base_idx + 1 < len(sys.argv):
                global_interpreter.llm.api_base = sys.argv[api_base_idx + 1]
        
        if "--api_key" in sys.argv:
            api_key_idx = sys.argv.index("--api_key")
            if api_key_idx + 1 < len(sys.argv):
                global_interpreter.llm.api_key = sys.argv[api_key_idx + 1]
        
        if "--max_tokens" in sys.argv:
            max_tokens_idx = sys.argv.index("--max_tokens")
            if max_tokens_idx + 1 < len(sys.argv):
                global_interpreter.llm.max_tokens = int(sys.argv[max_tokens_idx + 1])
        
        if "--context_window" in sys.argv:
            context_window_idx = sys.argv.index("--context_window")
            if context_window_idx + 1 < len(sys.argv):
                global_interpreter.llm.context_window = int(sys.argv[context_window_idx + 1])
        
        # Try to use custom provider
        try:
            provider = APIProvider.CUSTOM
            model = global_interpreter.llm.model
            print_markdown(f"> Using custom provider with model: **{model}**")
            
            # Check if model supports required capabilities
            if not global_interpreter.llm._is_loaded:
                global_interpreter.llm.load()
            
            if not global_interpreter.llm.supports_vision:
                print_markdown("> ⚠️ Warning: Model may not support vision. OS mode requires vision capabilities.")
            
            if not global_interpreter.llm.supports_functions:
                print_markdown("> ⚠️ Warning: Model may not support function calling. Tool calls may not work properly.")
            
        except Exception as e:
            print_markdown(f"> Error initializing custom provider: {e}")
            print_markdown("> Falling back to Anthropic provider...")
            provider = APIProvider.ANTHROPIC
            model = PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC]
    
    system_prompt_suffix = ""
    
    # Rest of the main function remains the same...
    print()
    print_markdown("Welcome to **Open Interpreter OS Mode**.\n")
    print_markdown("---")
    time.sleep(0.5)
    print()
    
    # Main interaction loop
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        })
        
        def output_callback(content_block: BetaContentBlock):
            if content_block.type == "text" and content_block.text:
                print(content_block)
                pass  # Text is already printed in the loop
        
        def tool_output_callback(result: ToolResult, tool_id: str):
            if result.output:
                print(f"\nTool output: {result.output}")
            if result.error:
                print(f"\nTool error: {result.error}")
        
        try:
            async for result in sampling_loop(
                model=model,
                provider=provider,
                system_prompt_suffix=system_prompt_suffix,
                messages=messages,
                output_callback=output_callback,
                tool_output_callback=tool_output_callback,
                api_key=api_key,
                interpreter=global_interpreter,
            ):
                if result["type"] == "messages":
                    messages = result["messages"]
                    break
        except Exception as e:
            print(f"Error: {e}")
            break


def run_async_main(interpreter=None):
    """Entry point that can accept an interpreter instance."""
    global global_interpreter
    global_interpreter = interpreter
    
    if "--server" in sys.argv:
        # Server mode not yet supported with custom provider
        print("Server mode not yet supported with custom provider")
        return
    else:
        asyncio.run(main(interpreter))


if __name__ == "__main__":
    run_async_main()