#!/var/nvme/thunk_venv/bin/python
import sys
import os

# Ensure the script can import src modules when run from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config

try:
    from google import genai
except ImportError:
    print("Error: google-genai not found. Ensure you are running this script within the correct virtual environment.", file=sys.stderr)
    sys.exit(1)

def main():
    # Load config
    try:
        config = load_config()
        oracle_config = config.get("oracle", {})
        api_key = oracle_config.get("gemini_api_key")
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    if not api_key:
        print("Error: No 'gemini_api_key' found in config.toml.", file=sys.stderr)
        sys.exit(1)

    # List Models
    try:
        client = genai.Client(api_key=api_key)
        print("Available Oracle Models:")
        print("-" * 30)
        for model in client.models.list():
            # Google often prefixes model names with 'models/', we can strip it for cleaner output
            name = model.name.replace("models/", "") if model.name else "Unknown"
            print(f"- {name}")
    except Exception as e:
        print(f"Error fetching models: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
