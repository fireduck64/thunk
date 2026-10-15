import os
import tomllib
from typing import Any, Dict, Optional

_CONFIG_PATH: Optional[str] = None

def set_config_path(config_path: str) -> None:
    """Sets the active configuration file path for this process and child processes."""
    global _CONFIG_PATH
    _CONFIG_PATH = config_path
    os.environ["THUNK_CONFIG"] = config_path

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads the TOML configuration file. Requires an explicit path via argument, set_config_path(), or THUNK_CONFIG."""
    resolved_path = config_path or _CONFIG_PATH or os.environ.get("THUNK_CONFIG")
    if not resolved_path:
        raise ValueError("No configuration file specified. Pass -c / --config <path> or set THUNK_CONFIG.")

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    
    with open(resolved_path, "rb") as f:
        return tomllib.load(f)
