import os
from src.components.base import BaseComponent
from src.utils.config import load_config

class DirectivesComponent(BaseComponent):
    """
    Injects high-level objectives and behavioral directives into the agent's core prompt.
    This gives the agent 'purpose' beyond just answering questions.
    """
    def __init__(self, file_path: str = None):
        super().__init__()
        if file_path is None:
            config = load_config()
            directives_cfg = config.get("directives", {})
            if "file_path" not in directives_cfg or not directives_cfg["file_path"]:
                raise ValueError("DirectivesComponent requires 'file_path' to be explicitly defined in config under [directives].file_path")
            self.file_path = directives_cfg["file_path"]
        else:
            self.file_path = file_path

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Directives file not found: {self.file_path}")

    def get_system_prompt_addition(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
