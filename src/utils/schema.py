import inspect
from typing import Callable, Any, get_origin, get_args

def _type_mapping(python_type: Any) -> str:
    """Maps basic Python types to JSON Schema types."""
    if python_type == str:
        return "string"
    elif python_type == int:
        return "integer"
    elif python_type == float:
        return "number"
    elif python_type == bool:
        return "boolean"
    elif python_type == list or get_origin(python_type) == list:
        return "array"
    elif python_type == dict or get_origin(python_type) == dict:
        return "object"
    return "string"  # fallback

def function_to_schema(func: Callable) -> dict:
    """
    Reflects on a Python function's type hints and docstring to generate
    an OpenAI-compatible JSON schema for tool calling.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or f"Execute {func.__name__}"
    
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        
        properties[name] = {
            "type": _type_mapping(param_type),
            "description": f"The {name} parameter."
        }
        
        if param.default == inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
