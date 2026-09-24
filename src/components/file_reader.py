import os
from typing import List, Callable, Dict, Optional
from src.components.base import BaseComponent

class FileReaderComponent(BaseComponent):
    """
    Allows the agent to explicitly read arbitrary local text/code files
    from configured libraries. Includes path traversal protections.
    """
    def __init__(self, libraries: Dict[str, str] = None):
        self.libraries = {}
        if libraries:
            for name, path in libraries.items():
                self.libraries[name] = os.path.abspath(path)

    def get_system_prompt_addition(self) -> str:
        return (
            "You have access to a local file reader. "
            "Use `list_libraries` to see available directories. "
            "Use `list_files` to explore the contents of a library. "
            "Use `read_file` to read the contents of text or code files. "
            "You can specify start and end lines to read specific sections."
        )

    def get_tools(self) -> List[Callable]:
        return [self.list_libraries, self.list_files, self.read_file]

    def _resolve_and_check_path(self, library: str, rel_path: str) -> str:
        """Helper to resolve paths safely within a library's bounds."""
        if library not in self.libraries:
            raise ValueError(f"Library '{library}' is not configured.")
            
        base_dir = self.libraries[library]
        safe_rel_path = rel_path.lstrip(os.sep) if rel_path else ""
        abs_path = os.path.abspath(os.path.join(base_dir, safe_rel_path))
        
        if os.path.commonpath([base_dir, abs_path]) != base_dir:
            raise ValueError(f"Access denied. Path is outside of library '{library}'.")
            
        return abs_path

    def list_libraries(self) -> str:
        """Lists all configured library names available for reading."""
        if not self.libraries:
            return "No libraries are currently configured."
            
        result = "Available libraries:\n"
        for name, path in self.libraries.items():
            result += f"- {name}: mounted at {path}\n"
        return result

    def list_files(self, library: str, path: str = "") -> str:
        """
        Lists files and directories inside a specified library.
        Leave 'path' empty to list the root of the library.
        """
        try:
            target_path = self._resolve_and_check_path(library, path)
            
            if not os.path.exists(target_path):
                return f"Error: Path '{path}' does not exist in library '{library}'."
                
            if not os.path.isdir(target_path):
                return f"Error: '{path}' is not a directory."
                
            entries = os.listdir(target_path)
            if not entries:
                return f"Directory '{path or '/'}' in library '{library}' is empty."
                
            # Sort for readability: dirs first, then files
            dirs = sorted([e for e in entries if os.path.isdir(os.path.join(target_path, e))])
            files = sorted([e for e in entries if os.path.isfile(os.path.join(target_path, e))])
            
            result = [f"Contents of {library} : {path or '/'}\n"]
            for d in dirs:
                result.append(f"[DIR]  {d}")
            for f in files:
                result.append(f"[FILE] {f}")
                
            return "\n".join(result)
            
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def read_file(self, library: str, filepath: str, start_line: int = 1, end_line: int = 200) -> str:
        """
        Reads the contents of a file from a specified library.
        Specify start_line and end_line (1-indexed) to read specific chunks of large files.
        Max 500 lines per read.
        """
        try:
            abs_path = self._resolve_and_check_path(library, filepath)
                
            if not os.path.exists(abs_path):
                return f"Error: File '{filepath}' does not exist in library '{library}'."
                
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
