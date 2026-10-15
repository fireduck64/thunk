import argparse
import os
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, set_config_path

def process_book(book_path: str):
    """Runs the ingest.py script for a single book."""
    script_path = os.path.join(os.path.dirname(__file__), "ingest.py")
    print(f"\n[IngestDir] Starting ingestion for: {book_path}")
    try:
        # We run it as a subprocess to keep memory perfectly isolated per book
        result = subprocess.run(
            [sys.executable, script_path, book_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[IngestDir] Successfully finished: {os.path.basename(book_path)}")
    except subprocess.CalledProcessError as e:
        print(f"[IngestDir] ERROR processing {book_path}:\n{e.stderr}")

def ingest_directory(directory: str, max_workers: int = 3):
    """Finds all .epub files recursively and ingests them."""
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
        
    # Find all epubs recursively
    search_pattern = os.path.join(directory, "**", "*.epub")
    epub_files = glob.glob(search_pattern, recursive=True)
    
    if not epub_files:
        print(f"No .epub files found in '{directory}'.")
        return
        
    print(f"Found {len(epub_files)} EPUB files in '{directory}'.")
    print(f"Starting ingestion with {max_workers} concurrent workers...\n")
    
    # Process them in parallel using a ThreadPool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_book, epub_files)
        
    print("\n[IngestDir] All files processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a directory of EPUB files into Qdrant")
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    parser.add_argument("directory", type=str, nargs="?", default="/vault/ebook", help="Path to directory of EPUBs")
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)
    # Validate config upfront before spawning worker processes
    load_config()
        
    ingest_directory(args.directory)
