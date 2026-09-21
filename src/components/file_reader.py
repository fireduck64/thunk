import os
from typing import List, Callable
from src.components.base import BaseComponent

class FileReaderComponent(BaseComponent):
    """
    Allows the agent to explicitly read arbitrary local text/code files.
    Includes path traversal protections and line-offset reading.
    """
    def __init__(self, allowed_dir: str = "/"):
        # Default to root if not specified, but typically this should be restricted
        self.allowed_dir = os.path.abspath(allowed_dir)

    def get_system_prompt_addition(self) -> str:
        return (
            "You have access to a local file reader. Use `read_file` to read the contents "
            "of arbitrary text or code files on the system. You can specify start and end lines "
            "to read specific sections without bloating your context window."
        )

    def get_tools(self) -> List[Callable]:
        return [self.read_file]

    def read_file(self, filepath: str, start_line: int = 1, end_line: int = 200) -> str:
        """
        Reads the contents of a local file.
        Specify start_line and end_line (1-indexed) to read specific chunks of large files.
        Max 500 lines per read.
        """
        try:
            # Security check: resolve absolute path and prevent traversal
            abs_path = os.path.abspath(filepath)
            
            # If allowed_dir is not root, ensure we are inside it
            if self.allowed_dir != "/" and not abs_path.startswith(self.allowed_dir):
                return f"Error: Access denied. Cannot read files outside of {self.allowed_dir}"
                
            if not os.path.exists(abs_path):
                return f"Error: File '{filepath}' does not exist."
                
            if not os.path.isfile(abs_path):
                return f"Error: '{filepath}' is not a file."
                
            # Limit the number of lines read to prevent context explosion
            if end_line - start_line > 500:
                end_line = start_line + 500
                
            if start_line < 1:
                start_line = 1

            lines_read = []
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f, 1):
                    if i >= start_line and i <= end_line:
                        lines_read.append(f"{i}: {line.rstrip()}")
                    elif i > end_line:
                        break
                        
            if not lines_read:
                return f"No content found between lines {start_line} and {end_line}."
                
            content = "\n".join(lines_read)
            return f"--- File: {filepath} (Lines {start_line}-{end_line}) ---\n{content}"
            
        except Exception as e:
            return f"Error reading file '{filepath}': {str(e)}"
