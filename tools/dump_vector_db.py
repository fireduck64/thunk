#!/usr/bin/env python3
import sys
import os

# Add the project root to the path so we can import src.utils.config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qdrant_client import QdrantClient
from src.utils.config import load_config

def main():
    config = load_config()
    host = config.get("vectordb", {}).get("host", "localhost")
    port = config.get("vectordb", {}).get("port", 6333)
    
    # We use "agent_memory" as defined in main.py, but let the user override it via argv
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "agent_memory"
    
    print(f"Connecting to Qdrant at {host}:{port}...")
    try:
        client = QdrantClient(host=host, port=port)
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return

    print(f"Fetching records from collection '{collection_name}'...")
    
    try:
        offset = None
        limit = 50
        total_records = 0
        
        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            if not records:
                break
                
            for point in records:
                total_records += 1
                timestamp = point.payload.get('timestamp', 'Unknown Time')
                text = point.payload.get('text', '')
                
                print(f"\n{'='*80}")
                print(f"ID: {point.id}")
                print(f"Saved: {timestamp}")
                print(f"{'-'*80}")
                print(text)
                
            if next_offset is None:
                break
            offset = next_offset
            
        print(f"\n{'='*80}")
        print(f"Total records found: {total_records}")
        
    except Exception as e:
        print(f"Error accessing collection: {e}")

if __name__ == "__main__":
    main()
