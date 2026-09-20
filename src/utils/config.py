import os
import tomllib
from typing import Any, Dict

def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Loads the TOML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "rb") as f:
        return tomllib.load(f)
