import os
from pathlib import Path

# --- CORE TOOLS ---

def read_file(path: str) -> str:
    """Reads the contents of a file."""
    try:
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{path}': {e}"

def write_file(path: str, content: str) -> str:
    """Writes content to a file, creating directories if necessary."""
    try:
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success: Wrote to '{path}'"
    except Exception as e:
        return f"Error writing file '{path}': {e}"

def list_files(path: str = ".") -> str:
    """Lists all files in the given directory and subdirectories."""
    try:
        if not os.path.exists(path):
            return f"Error: Directory '{path}' does not exist."
            
        files = []
        for root, _, filenames in os.walk(path):
            # Ignore hidden directories like .git and virtual environments
            if any(part.startswith('.') or part in ('.venv', 'venv', '__pycache__', 'adapters') for part in Path(root).parts):
                continue
            for filename in filenames:
                if not filename.startswith('.'):
                    files.append(os.path.join(root, filename))
        
        if not files:
            return "No files found (or directory is empty)."
            
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files in '{path}': {e}"

# --- TOOL REGISTRY ---
# Used for passing available tools to the models
AVAILABLE_TOOLS_SCHEMA = """
Available Tools:
1. `read_file(path)`: Returns the content of the file.
2. `write_file(path, content)`: Creates or overwrites a file with the given content.
3. `list_files(path=".")`: Lists all non-hidden files in the specified directory.

You must reply with ONLY a JSON block using the following format, nothing else:
{
    "tool": "tool_name",
    "path": "path_to_file_or_dir",
    "content": "content_to_write_if_any"
}
"""
