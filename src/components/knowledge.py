from typing import List, Callable, Dict, Any
from qdrant_client import QdrantClient
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.metrics import MetricSpan
import requests

class KnowledgeBaseComponent(BaseComponent):
    """
    Provides the agent with access to a Vector Database (Qdrant) 
    containing ingested documents (like EPUBs).
    """
    def __init__(self, thread_id: str = "main"):
        self.config = load_config()
        self.agent_name = self.config.get("agent", {}).get("name", "Thunk")
        self.thread_id = (thread_id or "main").strip() or "main"
        self.host = self.config["vectordb"]["host"]
        self.port = self.config["vectordb"]["port"]
        
        vectordb_cfg = self.config.get("vectordb", {})
        if "knowledge_collection" not in vectordb_cfg or not vectordb_cfg["knowledge_collection"]:
            raise ValueError("KnowledgeBaseComponent requires 'knowledge_collection' to be explicitly defined in config under [vectordb].knowledge_collection")
        self.collection = vectordb_cfg["knowledge_collection"]
        
        self.ollama_url = self.config["embeddings"]["api_url"]
        self.embed_model = self.config["embeddings"]["model_name"]
        self.api_key = self.config["embeddings"]["api_key"]
        
        # We don't connect to Qdrant in __init__ because the DB might not be up yet,
        # and we want to lazy-load it when the tool is actually called.
        self._client = None

    def get_system_prompt_addition(self) -> str:
        # We remove the tool instruction here, as GlobalSearch handles it now
        return ""

    def get_tools(self) -> List[Callable]:
        # We no longer expose this directly to the LLM
        return []

    def execute_search(self, query: str) -> str:
        """Standardized interface for GlobalSearchComponent."""
        # We just reuse the existing search logic, but limit it to 2 results 
        # so GlobalSearch doesn't explode the context window with too much text
        return self.search_knowledge_base(query, num_results=2)

    def _get_embedding(self, text: str) -> list[float]:
        """Helper to call the embeddings API."""
        agent_name = getattr(getattr(self, "agent", None), "name", None) or getattr(self, "agent_name", "Thunk")
        thread_id = getattr(getattr(self, "agent", None), "thread_id", None) or getattr(self, "thread_id", "main") or "main"
        headers = {"X-Agent-ID": f"{agent_name}-{thread_id}-knowledge"}
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
        agent_name = getattr(getattr(self, "agent", None), "name", None) or getattr(self, "agent_name", "Thunk")
        thread_id = getattr(getattr(self, "agent", None), "thread_id", None) or getattr(self, "thread_id", "main") or "main"

        with MetricSpan(
            operation="vector_search",
            agent=agent_name,
            thread=thread_id,
            component="knowledge",
            model=self.embed_model,
        ) as span:
            span.inc("request_bytes", len((query or "").encode("utf-8")))
            try:
                if not self._client:
                    self._client = QdrantClient(host=self.host, port=self.port)
                    
                with span.subtimer("embedding_ms"):
                    query_vector = self._get_embedding(query)
                
                with span.subtimer("db_lookup_ms"):
                    search_result = self._client.search(
                        collection_name=self.collection,
                        query_vector=query_vector,
                        limit=num_results
                    )
                span.inc("db_lookup", 1)
                span.inc("db_lookup_records", len(search_result))
                
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
                span.fail()
                return f"Error accessing knowledge base: {str(e)}"
