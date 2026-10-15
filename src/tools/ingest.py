import argparse
import os
import sys
import json
import uuid
import requests
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, set_config_path

def get_embedding(text: str, ollama_url: str, embedding_model: str, api_key: str) -> list[float]:
    """Calls Ollama to convert text into a vector (list of floats)."""
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

def extract_text_from_epub(epub_path: str) -> str:
    """Extracts raw text from an EPUB file."""
    print(f"Reading {epub_path}...")
    book = epub.read_epub(epub_path)
    full_text = []
    
    # Iterate through all the "documents" (usually chapters) in the epub
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        if text:
            full_text.append(text)
            
    return "\n\n".join(full_text)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Splits a large text into smaller overlapping chunks."""
    print("Chunking text...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap # Move forward, leaving some overlap for context
    return chunks

def ingest_file(epub_path: str):
    config = load_config()
    qdrant_host = config["vectordb"]["host"]
    qdrant_port = config["vectordb"]["port"]
    collection_name = config["vectordb"].get("knowledge_collection", config["vectordb"].get("collection_name", "knowledge_base"))
    ollama_url = config["embeddings"]["api_url"]
    embedding_model = config["embeddings"]["model_name"]
    api_key = config["embeddings"]["api_key"]

    # 1. Connect to Qdrant Vector DB
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    # Check if collection exists, if not create it.
    # nomic-embed-text produces vectors of size 768.
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists.")
    except Exception:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    # 2. Extract and chunk text
    text = extract_text_from_epub(epub_path)
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generate embeddings and upload to Qdrant
    points = []
    title = os.path.basename(epub_path)

    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Embedding chunk {i}/{len(chunks)}...")
        
        vector = get_embedding(chunk, ollama_url, embedding_model, api_key)
        
        # We store the original text in the payload so we can read it later
        payload = {
            "source": title,
            "chunk_index": i,
            "text": chunk
        }
        
        # Generate a deterministic UUID based on the source file and chunk index.
        # This makes the ingestion script idempotent. If we run it again, 
        # it will just overwrite the exact same IDs in Qdrant.
        hash_input = f"{title}_{i}".encode('utf-8')
        deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hash_input.decode('utf-8')))
        
        points.append(
            PointStruct(
                id=deterministic_id, 
                vector=vector, 
                payload=payload
            )
        )

        # Batch upload every 50 points to be efficient
        if len(points) >= 50:
            client.upsert(collection_name=collection_name, points=points)
            points = []

    # Upload any remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest an EPUB file into Qdrant")
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    parser.add_argument("epub_path", type=str, help="Path to the EPUB file")
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)
    
    ingest_file(args.epub_path)
