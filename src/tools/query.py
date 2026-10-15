import argparse
import sys
import os
import requests
from qdrant_client import QdrantClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, set_config_path

def get_embedding(text: str, ollama_url: str, embedding_model: str, api_key: str) -> list[float]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    response = requests.post(
        ollama_url, 
        json={
            "model": embedding_model,
            "prompt": text
        },
        headers=headers
    )
    response.raise_for_status()
    return response.json()["embedding"]

def search(query: str, limit: int = 3):
    config = load_config()
    qdrant_host = config["vectordb"]["host"]
    qdrant_port = config["vectordb"]["port"]
    collection_name = config["vectordb"].get("knowledge_collection", config["vectordb"].get("collection_name", "knowledge_base"))
    ollama_url = config["embeddings"]["api_url"]
    embedding_model = config["embeddings"]["model_name"]
    api_key = config["embeddings"]["api_key"]

    print(f"Searching for: '{query}'")
    
    # 1. Convert the search query into a vector
    query_vector = get_embedding(query, ollama_url, embedding_model, api_key)
    
    # 2. Search Qdrant for the closest matching vectors
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    search_result = client.search(
        collection_name=collection_name,
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
    parser = argparse.ArgumentParser(description="Search the Qdrant knowledge base")
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)
        
    search(args.query)
