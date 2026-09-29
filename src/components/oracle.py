import asyncio
from typing import List, Callable
from src.components.base import BaseComponent
from src.utils.config import load_config
from google import genai
from google.genai import types

class OracleComponent(BaseComponent):
    """
    Provides the agent with access to a cloud-based AI model (Oracle) 
    to look up current events, facts, or perform advanced reasoning.
    """
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.oracle_config = self.config.get("oracle", {})
        self.api_key = self.oracle_config.get("gemini_api_key")
        self.model_name = self.oracle_config.get("gemini_model", "gemini-2.5-flash")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            print("[Oracle] Warning: No gemini_api_key found in config.toml. Oracle tool will be disabled.")

    def get_system_prompt_addition(self) -> str:
        return (
            "--- ORACLE (CLOUD AI) ---\n"
            "You have access to a cloud-based AI Oracle (Google Gemini). You can use the `ask_oracle` tool "
            "to look up current events, general knowledge, or ask questions that fall outside your training data or context."
        )

    def get_tools(self) -> List[Callable]:
        if not self.client:
            return []
        return [self.ask_oracle]

    async def ask_oracle(self, query: str) -> str:
        """
        Queries the Cloud AI Oracle (Google Gemini) with the provided text.
        Use this tool to look up current events, facts, or seek advice on complex topics.
        """
        print(f"[Oracle] Consulting the oracle for: '{query}'")
        
        try:
            # We use the synchronous generate_content method inside asyncio.to_thread 
            # to avoid blocking the agent's main event loop.
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=query,
            )
            return response.text
        except Exception as e:
            error_msg = f"Failed to consult the oracle: {str(e)}"
            print(f"[Oracle] {error_msg}")
            return error_msg
