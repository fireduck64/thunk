#!./venv/bin/python3
import sys
import os
import argparse

# Ensure the script can import src modules when run from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config, set_config_path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai not found. Ensure you are running this script within the correct virtual environment.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Send stdin to the Cloud AI Oracle (Google Gemini)")
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    parser.add_argument("-m", "--model", type=str, help="The model to use (overrides config)")
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)

    # Read stdin
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: No input provided on stdin.", file=sys.stderr)
        sys.exit(1)

    # Load config
    try:
        config = load_config()
        oracle_config = config.get("oracle", {})
        api_key = oracle_config.get("gemini_api_key")
        default_model = oracle_config.get("gemini_model", "gemini-3.8-flash")
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    if not api_key:
        print("Error: No 'gemini_api_key' found in config.toml.", file=sys.stderr)
        sys.exit(1)

    model_name = args.model if args.model else default_model

    # Query Oracle
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        print(response.text)
    except Exception as e:
        print(f"Error querying oracle: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
