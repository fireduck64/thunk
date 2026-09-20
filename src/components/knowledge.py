from typing import List, Callable, Dict, Any
from qdrant_client import QdrantClient
from src.components.base import BaseComponent
from src.utils.config import load_config
import requests

class KnowledgeBaseComponent(BaseComponent):
    """
    Provides the agent with access to a Vector Database (Qdrant) 
    containing ingested documents (like EPUBs).
    """
    def __init__(self):
        self.config = load_config()
        self.host = self.config["vectordb"]["host"]
        self.port = self.config["vectordb"]["port"]
        self.collection = self.config["vectordb"]["collection_name"]
        
        self.ollama_url = self.config["embeddings"]["api_url"]
        self.embed_model = self.config["embeddings"]["model_name"]
        self.api_key = self.config["embeddings"]["api_key"]
        
        # We don't connect to Qdrant in __init__ because the DB might not be up yet,
        # and we want to lazy-load it when the tool is actually called.
        self._client = None

    def get_system_prompt_addition(self) -> str:
        return (
            "You have access to a vast Knowledge Base containing books, manuals, and documents. "
            "If you need to look up facts, lore, or specific information that you do not "
            "currently have in memory, use the `search_knowledge_base` tool. It performs "
            "semantic searches, so you can search using natural language questions."
        )

    def get_tools(self) -> List[Callable]:
        return [self.search_knowledge_base]

    def _get_embedding(self, text: str) -> list[float]:
        """Helper to call the embeddings API."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        response = requests.post(
            self.ollama_url, 
            json={
                "model": self.embed_model,
                "prompt": text
            },
            headers=headers
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def search_knowledge_base(self, query: str, num_results: int = 3) -> str:
        """
        Searches the vector database for text chunks semantically similar to the query.
        Useful for retrieving facts or context from ingested books or documents.
        """
        print(f"[KnowledgeBase] Searching for: '{query}'")
        
        try:
            if not self._client:
                self._client = QdrantClient(host=self.host, port=self.port)
                
            query_vector = self._get_embedding(query)
            
            search_result = self._client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=num_results
            )
            
            if not search_result:
                return f"No relevant information found in the knowledge base for '{query}'."
                
            results = []
            for i, hit in enumerate(search_result):
                source = hit.payload.get('source', 'Unknown Document')
                text = hit.payload.get('text', '')
                score = round(hit.score, 3)
                results.append(f"--- Result {i+1} (Source: {source}, Relevance: {score}) ---\n{text}")
                
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Error accessing knowledge base: {str(e)}"
