# Custom OS Mode Provider Implementation Plan

## Overview
This document outlines the implementation plan for adding a custom provider to OS mode that uses Open Interpreter's configured model instead of requiring Anthropic's API.

## Goals
- Enable OS mode to work with any vision-capable LLM configured in Open Interpreter
- Maintain compatibility with existing Anthropic-based OS mode
- Support local models (LM Studio, Ollama, etc.) and other API providers (OpenAI, etc.)
- Provide seamless fallback mechanism if model lacks required capabilities

## Architecture Design

### 1. Provider System Extension
- Add `CUSTOM` provider type to existing provider enum
- Create provider factory pattern for better extensibility
- Implement capability detection system

### 2. Model Adapter Layer
Create abstraction layer between Anthropic's computer use format and Open Interpreter's LLM interface:
- Message format translation (Anthropic ↔ OpenAI/Generic)
- Tool definition conversion
- Response format normalization

### 3. Component Structure
```
interpreter/
├── computer_use/
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base_provider.py       # Abstract base provider
│   │   ├── anthropic_provider.py  # Existing Anthropic logic
│   │   ├── custom_provider.py     # New custom provider
│   │   └── model_adapter.py       # Format translation utilities
│   ├── loop.py                    # Modified to support providers
│   └── tools/                     # Existing tools
```

## Implementation Steps

### Phase 1: Foundation (Current)
1. ✅ Create feature branch
2. ✅ Document implementation plan
3. Analyze existing model configuration system
4. Create provider module structure

### Phase 2: Core Implementation
5. Implement base provider interface
6. Extract Anthropic-specific logic to anthropic_provider.py
7. Create custom_provider.py with Open Interpreter integration
8. Implement message/tool format translators

### Phase 3: Integration
9. Modify loop.py to use provider system
10. Update __init__.py entry point
11. Add capability detection for vision and tool support
12. Implement fallback mechanism

### Phase 4: Testing & Documentation
13. Test with LM Studio (Gemma model)
14. Test with OpenAI GPT-4V
15. Test with local vision models
16. Update README with usage instructions
17. Create comprehensive test suite

## Technical Details

### Message Format Translation

#### Anthropic Format:
```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Click on the button"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    ]
}
```

#### OpenAI Format:
```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Click on the button"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
}
```

### Tool Definition Translation

#### Anthropic Computer Use:
```python
{
    "name": "computer",
    "type": "computer_20241022",
    "display_width_px": 1920,
    "display_height_px": 1080,
    "display_number": null
}
```

#### OpenAI Function Calling:
```python
{
    "type": "function",
    "function": {
        "name": "computer_control",
        "description": "Control computer with mouse and keyboard",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["screenshot", "click", "type", ...]},
                "coordinate": {"type": "array", "items": {"type": "integer"}},
                "text": {"type": "string"}
            }
        }
    }
}
```

### Model Capability Detection

```python
def check_model_capabilities(interpreter):
    """Check if current model supports required OS mode features"""
    capabilities = {
        'vision': False,
        'tool_calling': False,
        'streaming': False
    }
    
    # Check vision support
    if hasattr(interpreter.llm, 'supports_vision'):
        capabilities['vision'] = interpreter.llm.supports_vision
    
    # Check tool calling support
    if hasattr(interpreter.llm, 'supports_functions'):
        capabilities['tool_calling'] = interpreter.llm.supports_functions
    
    return capabilities
```

### Provider Selection Logic

```python
def select_provider(interpreter, force_anthropic=False):
    """Select appropriate provider based on configuration and capabilities"""
    
    if force_anthropic or os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    
    capabilities = check_model_capabilities(interpreter)
    
    if capabilities['vision'] and capabilities['tool_calling']:
        return CustomProvider(interpreter)
    else:
        raise ValueError("Current model doesn't support required OS mode features")
```

## Configuration Options

### New CLI Flags:
- `--os-provider [auto|anthropic|custom]` - Provider selection
- `--os-fallback` - Enable automatic fallback to Anthropic
- `--os-debug` - Enable debug logging for OS mode

### Usage Examples:

```bash
# Use configured model for OS mode
interpreter --os --os-provider custom

# Use with LM Studio
interpreter --os --model gemma-2-9b --api_base "http://localhost:1234/v1" --api_key "fake_key"

# Force Anthropic provider
interpreter --os --os-provider anthropic

# Auto-detect best provider
interpreter --os --os-provider auto
```

## Error Handling

1. **Model Capability Errors**: Clear messages when model lacks vision/tool support
2. **Format Translation Errors**: Graceful degradation with warnings
3. **Performance Issues**: Timeout handling for slow local models
4. **Fallback Mechanism**: Automatic switch to Anthropic if available

## Testing Strategy

### Unit Tests:
- Message format translation
- Tool definition conversion
- Capability detection

### Integration Tests:
- Full OS mode flow with different providers
- Fallback mechanism
- Error scenarios

### Manual Testing:
- LM Studio with Gemma model
- OpenAI GPT-4V
- Anthropic Claude 3.5
- Local vision models (LLaVA, etc.)

## Success Criteria

1. OS mode works with any vision-capable, tool-calling LLM
2. No regression in existing Anthropic-based functionality
3. Clear error messages and fallback options
4. Performance acceptable for local models
5. Documentation clear and comprehensive

## Timeline

- Phase 1: 1 hour (Documentation & Analysis)
- Phase 2: 3 hours (Core Implementation)
- Phase 3: 2 hours (Integration)
- Phase 4: 2 hours (Testing & Documentation)

Total estimated time: 8 hours

## Notes

- Priority is maintaining backward compatibility
- Focus on clean abstraction between providers
- Ensure extensibility for future providers
- Consider performance implications for local models