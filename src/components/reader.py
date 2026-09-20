import os
import glob
from typing import List, Callable, Dict, Any
from ebooklib import epub
from bs4 import BeautifulSoup
from src.components.base import BaseComponent

class SequentialReaderComponent(BaseComponent):
    """
    Allows the agent to open EPUB books and read them sequentially, chunk by chunk.
    This is useful for deep comprehension tasks where semantic search is insufficient.
    """
    def __init__(self, library_dir: str = "books", chunk_size: int = 2500):
        self.library_dir = library_dir
        self.chunk_size = chunk_size
        
        # Agent's reading state
        self.current_book: str = None
        self.current_chunks: List[str] = []
        self.current_index: int = 0
        
        # Ensure library directory exists
        os.makedirs(self.library_dir, exist_ok=True)

    def get_system_prompt_addition(self) -> str:
        prompt = (
            "You have access to a sequential document reader. While your knowledge base search "
            "is good for finding facts, the sequential reader allows you to sit down and read "
            "an entire book from start to finish.\n"
            "Use `list_library` to see available books. Use `open_book` to load a book into "
            "your active reading state. Use `read_next_chunk` and `read_previous_chunk` to navigate "
            "through the open book."
        )
        if self.current_book:
            prompt += f"\n[Status: You currently have '{os.path.basename(self.current_book)}' open at chunk {self.current_index}/{len(self.current_chunks)-1}]."
        else:
            prompt += "\n[Status: You do not have a book open.]"
            
        return prompt

    def get_tools(self) -> List[Callable]:
        return [
            self.list_library,
            self.open_book,
            self.read_next_chunk,
            self.read_previous_chunk
        ]

    # --- Tool Implementations ---

    def list_library(self) -> List[str]:
        """Returns a list of all EPUB filenames available to read."""
        files = glob.glob(os.path.join(self.library_dir, "*.epub"))
        return [os.path.basename(f) for f in files]

    def open_book(self, filename: str) -> str:
        """
        Opens an EPUB file from the library, extracts its text, and prepares it for reading.
        Resets your reading position to the beginning.
        """
        file_path = os.path.join(self.library_dir, filename)
        if not os.path.exists(file_path):
            return f"Error: Book '{filename}' not found in the library."

        try:
            print(f"[Reader] Extracting text from {filename}...")
            book = epub.read_epub(file_path)
            full_text = []
            
            # Simple linear extraction
            for item in book.get_items_of_type(9): # ITEM_DOCUMENT
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if text:
                    full_text.append(text)
                    
            combined_text = "\n\n".join(full_text)
            
            # Split into hard chunks (no overlap needed for sequential reading)
            self.current_chunks = [
                combined_text[i:i+self.chunk_size] 
                for i in range(0, len(combined_text), self.chunk_size)
            ]
            self.current_book = file_path
            self.current_index = 0
            
            return f"Successfully opened '{filename}'. The book has been split into {len(self.current_chunks)} reading chunks. Use read_next_chunk to begin."
            
        except Exception as e:
            return f"Error opening book: {str(e)}"

    def read_next_chunk(self) -> str:
        """Returns the next chunk of text from the currently open book and advances your bookmark."""
        if not self.current_book:
            return "Error: No book is currently open. Use open_book first."
            
        if self.current_index >= len(self.current_chunks):
            return "You have reached the end of the book."
            
        chunk = self.current_chunks[self.current_index]
        self.current_index += 1
        
        return f"--- Chunk {self.current_index}/{len(self.current_chunks)} ---\n{chunk}"

    def read_previous_chunk(self) -> str:
        """Moves your bookmark back one chunk and returns that text."""
        if not self.current_book:
            return "Error: No book is currently open."
            
        if self.current_index <= 1:
            self.current_index = 0
            return "You are already at the beginning of the book."
            
        # If we just read chunk 1 (index 1), going back means we want index 0
        self.current_index -= 2 
        if self.current_index < 0:
            self.current_index = 0
            
        chunk = self.current_chunks[self.current_index]
        self.current_index += 1 # Advance pointer again so next read gets the right one
        
        return f"--- Chunk {self.current_index}/{len(self.current_chunks)} ---\n{chunk}"
