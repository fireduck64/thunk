import sys
import os
import requests
from qdrant_client import QdrantClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config

config = load_config()

QDRANT_HOST = config["vectordb"]["host"]
QDRANT_PORT = config["vectordb"]["port"]
COLLECTION_NAME = config["vectordb"]["collection_name"]
OLLAMA_URL = config["embeddings"]["api_url"]
EMBEDDING_MODEL = config["embeddings"]["model_name"]
API_KEY = config["embeddings"]["api_key"]

def get_embedding(text: str) -> list[float]:
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
        
    response = requests.post(
        OLLAMA_URL, 
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        headers=headers
    )
    response.raise_for_status()
    return response.json()["embedding"]

def search(query: str, limit: int = 3):
    print(f"Searching for: '{query}'")
    
    # 1. Convert the search query into a vector
    query_vector = get_embedding(query)
    
    # 2. Search Qdrant for the closest matching vectors
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    
    # 3. Display the results
    print(f"\n--- Found {len(search_result)} results ---\n")
    for i, hit in enumerate(search_result):
        # hit.score is how closely it matched (1.0 is exact)
        print(f"Result {i+1} (Score: {hit.score:.3f}):")
        print(f"Source: {hit.payload.get('source')} | Chunk: {hit.payload.get('chunk_index')}")
        print("-" * 40)
        print(hit.payload.get('text'))
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tools/query.py <search_term>")
        sys.exit(1)
        
    search(sys.argv[1])
