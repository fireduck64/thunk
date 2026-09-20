import uuid
import requests
from datetime import datetime
from typing import List, Callable, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.components.base import BaseComponent
from src.utils.config import load_config

class VectorMemoryComponent(BaseComponent):
    """
    Provides the agent with a personal, writable vector database.
    Allows the agent to save unstructured thoughts, lore, and facts, 
    and retrieve them later using semantic search.
    """
    def __init__(self, collection_name: str = "agent_memory"):
        self.config = load_config()
        self.host = self.config["vectordb"]["host"]
        self.port = self.config["vectordb"]["port"]
        self.collection = collection_name
        
        self.ollama_url = self.config["embeddings"]["api_url"]
        self.embed_model = self.config["embeddings"]["model_name"]
        self.api_key = self.config["embeddings"]["api_key"]
        
        self._client = None
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection in Qdrant if it doesn't exist."""
        try:
            client = QdrantClient(host=self.host, port=self.port)
            try:
                client.get_collection(self.collection)
            except Exception:
                print(f"[VectorMemory] Creating collection '{self.collection}'...")
                client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"[VectorMemory Error] Failed to initialize Qdrant: {e}")

    def get_system_prompt_addition(self) -> str:
        return (
            "You have access to a Semantic Vector Memory (a personal, writable vector database). "
            "Unlike Structured Notes (which require specific keys), you can use `save_semantic_memory` "
            "to store unstructured facts, thoughts, or lore. You can then use `search_semantic_memory` "
            "to recall them later based on meaning or concepts rather than exact keywords."
        )

    def get_tools(self) -> List[Callable]:
        return [self.save_semantic_memory, self.search_semantic_memory]

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

    def save_semantic_memory(self, content: str) -> str:
        """
        Saves a block of text into your personal semantic memory. 
        You can search for this later using meaning/concepts.
        """
        print(f"[VectorMemory] Saving memory: {content[:50]}...")
        try:
            if not self._client:
                self._client = QdrantClient(host=self.host, port=self.port)
                
            vector = self._get_embedding(content)
            
            payload = {
                "text": content,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Using UUID4 since agent memories don't need deterministic overwrites like RAG ingestion
            point_id = str(uuid.uuid4())
            
            self._client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)]
            )
            return "Memory saved successfully."
            
        except Exception as e:
            return f"Error saving memory: {str(e)}"

    def search_semantic_memory(self, query: str, num_results: int = 3) -> str:
        """
        Searches your personal semantic memory for saved facts related to the query.
        Returns the closest matches.
        """
        print(f"[VectorMemory] Searching for: '{query}'")
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
                return f"No relevant memories found for '{query}'."
                
            results = []
            for i, hit in enumerate(search_result):
                text = hit.payload.get('text', '')
                timestamp = hit.payload.get('timestamp', 'Unknown Time')
                score = round(hit.score, 3)
                results.append(f"--- Memory {i+1} (Relevance: {score}, Saved: {timestamp}) ---\n{text}")
                
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Error accessing semantic memory: {str(e)}"
