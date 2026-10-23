import sys
import time
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
    def __init__(self, collection_name: str = None):
        self.config = load_config()
        self.host = self.config["vectordb"]["host"]
        self.port = self.config["vectordb"]["port"]
        
        if collection_name is None:
            vectordb_cfg = self.config.get("vectordb", {})
            if "memory_collection" not in vectordb_cfg or not vectordb_cfg["memory_collection"]:
                raise ValueError("VectorMemoryComponent requires 'memory_collection' to be explicitly defined in config under [vectordb].memory_collection")
            self.collection = vectordb_cfg["memory_collection"]
        else:
            self.collection = collection_name
        
        self.ollama_url = self.config["embeddings"]["api_url"]
        self.embed_model = self.config["embeddings"]["model_name"]
        self.api_key = self.config["embeddings"]["api_key"]

        self.chunk_size = int(self.config.get("vectordb", {}).get("chunk_size", 4000))
        self.chunk_overlap = int(self.config.get("vectordb", {}).get("chunk_overlap", 400))
        self.max_retries = 5
        self.retry_base_delay = 2.0
        
        self._client = None
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection in Qdrant if it doesn't exist. Retries loudly and exits on failure."""
        for attempt in range(self.max_retries):
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
                self._client = client
                return
            except Exception as e:
                print(
                    f"[VectorMemory ERROR] Collection initialization attempt {attempt + 1}/{self.max_retries} failed: {e}",
                    flush=True,
                )
                if attempt + 1 < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    if delay > 0:
                        time.sleep(delay)

        print(
            f"[VectorMemory FATAL] Failed to initialize Qdrant collection '{self.collection}' after {self.max_retries} attempts. Exiting process.",
            flush=True,
        )
        sys.exit(1)

    def _chunk_text(self, text: str) -> List[str]:
        """Splits text exceeding chunk_size into overlapping chunks that fit the embedding context window."""
        if len(text) <= self.chunk_size:
            return [text]

        raw_chunks = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            end = start + self.chunk_size
            raw_chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step

        if text.startswith("Consolidated Memory Block:") and len(raw_chunks) > 1:
            total = len(raw_chunks)
            formatted = [raw_chunks[0]]
            for idx in range(1, total):
                formatted.append(f"Consolidated Memory Block (Part {idx + 1}/{total}):\n{raw_chunks[idx]}")
            return formatted

        return raw_chunks

    def get_system_prompt_addition(self) -> str:
        return (
            "You have a personal Semantic Vector Memory. Use `save_semantic_memory` "
            "to store unstructured facts, thoughts, or lore. "
            "(To retrieve them, use the `global_search` tool). "
            "Use `dump_semantic_memory` to view all saved items."
        )

    def get_tools(self) -> List[Callable]:
        # Expose the save tool and the dump tool directly
        return [self.save_semantic_memory, self.dump_semantic_memory]

    def execute_search(self, query: str) -> str:
        """Standardized interface for GlobalSearchComponent."""
        return self.search_semantic_memory(query, num_results=2)

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
        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise ValueError(f"Embeddings API returned empty embedding: {data}")
        return embedding

    def _embed_with_auto_split(self, text: str) -> List[tuple[str, list[float]]]:
        """
        Embeds a text chunk. If the embedding endpoint rejects the chunk for exceeding
        the model's token context window (e.g. dense punctuation or code), recursively
        splits the chunk in half so no content is lost or rejected.
        """
        try:
            return [(text, self._get_embedding(text))]
        except requests.exceptions.HTTPError as e:
            resp_text = e.response.text if e.response is not None else ""
            if "exceeds the context length" in resp_text and len(text) > 200:
                mid = len(text) // 2
                print(
                    f"[VectorMemory] Chunk of {len(text)} chars exceeded embedding token context; splitting in half...",
                    flush=True,
                )
                return self._embed_with_auto_split(text[:mid]) + self._embed_with_auto_split(text[mid:])
            raise

    def save_semantic_memory(self, content: str) -> str:
        """
        Saves a block of text into your personal semantic memory. 
        You can search for this later using meaning/concepts.
        """
        print(f"[VectorMemory] Saving memory: {content[:50]}...")
        chunks = self._chunk_text(content)
        timestamp = datetime.utcnow().isoformat() + "Z"

        for attempt in range(self.max_retries):
            try:
                if not self._client:
                    self._client = QdrantClient(host=self.host, port=self.port)

                points = []
                for chunk in chunks:
                    for sub_chunk, vector in self._embed_with_auto_split(chunk):
                        payload = {
                            "text": sub_chunk,
                            "timestamp": timestamp,
                        }
                        # Using UUID4 since agent memories don't need deterministic overwrites like RAG ingestion
                        point_id = str(uuid.uuid4())
                        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

                self._client.upsert(
                    collection_name=self.collection,
                    points=points,
                )
                return "Memory saved successfully."

            except Exception as e:
                print(
                    f"[VectorMemory ERROR] Save memory attempt {attempt + 1}/{self.max_retries} failed: {e}",
                    flush=True,
                )
                if attempt + 1 < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    if delay > 0:
                        time.sleep(delay)

        print(
            f"[VectorMemory FATAL] Failed to save semantic memory after {self.max_retries} attempts. Exiting process.",
            flush=True,
        )
        sys.exit(1)

    def search_semantic_memory_raw(self, query: str, num_results: int = 8) -> List[Dict[str, Any]]:
        """
        Searches personal semantic memory and returns raw structured hits:
        [{"id": str, "content": str, "metadata": dict, "score": float}, ...]
        """
        print(f"[VectorMemory] Searching raw hits (limit={num_results}) for: '{query[:80]}'")
        if not self._client:
            self._client = QdrantClient(host=self.host, port=self.port)

        query_vector = self._get_embedding(query)

        search_result = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=num_results,
        )

        hits = []
        for hit in search_result:
            payload = dict(hit.payload or {})
            hits.append(
                {
                    "id": str(hit.id),
                    "content": payload.get("text", ""),
                    "metadata": payload,
                    "score": float(hit.score),
                }
            )
        return hits

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

    def dump_semantic_memory(self, limit: int = 10, offset: int = 0) -> str:
        """
        Dumps the raw contents of your personal semantic memory. 
        Use this tool to see exactly what you have stored in the vector database.
        Use limit and offset to paginate through large memory collections.
        """
        print(f"[VectorMemory] Dumping memory (limit={limit}, offset={offset})...")
        try:
            if not self._client:
                self._client = QdrantClient(host=self.host, port=self.port)
            
            # Use scroll to retrieve raw points without vector search
            result, next_page_offset = self._client.scroll(
                collection_name=self.collection,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            if not result:
                return f"No memories found (offset={offset})."
                
            dump_lines = []
            for i, point in enumerate(result):
                text = point.payload.get('text', '')
                timestamp = point.payload.get('timestamp', 'Unknown Time')
                dump_lines.append(f"--- Record {offset + i + 1} (Saved: {timestamp}) ID: {point.id} ---\n{text}")
                
            response = "\n\n".join(dump_lines)
            if next_page_offset:
                response += f"\n\n(More records exist. To view them, call this tool again with offset={next_page_offset})"
            else:
                response += f"\n\n(End of memory records)"
                
            return response
            
        except Exception as e:
            return f"Error dumping semantic memory: {str(e)}"
