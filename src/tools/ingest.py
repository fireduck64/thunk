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
from src.utils.config import load_config

config = load_config()

# Configuration
QDRANT_HOST = config["vectordb"]["host"]
QDRANT_PORT = config["vectordb"]["port"]
COLLECTION_NAME = config["vectordb"]["collection_name"]
OLLAMA_URL = config["embeddings"]["api_url"]
EMBEDDING_MODEL = config["embeddings"]["model_name"]
API_KEY = config["embeddings"]["api_key"]

def get_embedding(text: str) -> list[float]:
    """Calls Ollama to convert text into a vector (list of floats)."""
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
    # 1. Connect to Qdrant Vector DB
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Check if collection exists, if not create it.
    # nomic-embed-text produces vectors of size 768.
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' exists.")
    except Exception:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
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
        
        vector = get_embedding(chunk)
        
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
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

    # Upload any remaining points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/tools/ingest.py <path_to_epub>")
        sys.exit(1)
    
    ingest_file(sys.argv[1])
