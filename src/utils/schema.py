import inspect
import types
from typing import Callable, Any, Union, get_origin, get_args


def _unwrap_optional(python_type: Any) -> Any:
    origin = get_origin(python_type)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(python_type) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return python_type


def _type_mapping(python_type: Any) -> str:
    """Maps basic Python types to JSON Schema types."""
    python_type = _unwrap_optional(python_type)
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


def _schema_for_type(python_type: Any) -> dict:
    unwrapped = _unwrap_optional(python_type)
    base_type = _type_mapping(unwrapped)
    if base_type == "array":
        args = get_args(unwrapped)
        item_type = args[0] if args else str
        return {
            "type": "array",
            "items": _schema_for_type(item_type),
        }
    return {"type": base_type}


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
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
            
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        
        prop_schema = _schema_for_type(param_type)
        prop_schema["description"] = f"The {name} parameter."
        properties[name] = prop_schema
        
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
