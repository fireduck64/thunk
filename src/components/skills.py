import os
from typing import List, Callable, Dict
from src.components.base import BaseComponent

class SkillComponent(BaseComponent):
    """
    Allows the agent to create, save, and load its own reusable skills.
    Skills are stored as Markdown files with YAML-style frontmatter for summaries.
    """
    def __init__(self, skills_dir: str = "skills"):
        super().__init__()
        self.skills_dir = skills_dir
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)

    def _parse_skill_file(self, filepath: str) -> Dict[str, str]:
        """Parses a skill markdown file, extracting the summary and text."""
        summary = "No summary provided."
        text = ""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if content.startswith("---"):
                # Split on --- to find frontmatter
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    text = parts[2].strip()
                    
                    # Extract summary from frontmatter
                    for line in frontmatter.split('\n'):
                        if line.startswith("summary:"):
                            summary = line.replace("summary:", "", 1).strip()
                            break
                else:
                    text = content
            else:
                text = content
        except Exception as e:
            print(f"[SkillComponent] Error reading skill file {filepath}: {e}")
            
        return {"summary": summary, "text": text}

    def get_system_prompt_addition(self) -> str:
        prompt = (
            "--- SKILLS ---\n"
            "You have the ability to create, load, and manage your own reusable skills (procedural memory). "
            "Use these to document complex, multi-step workflows you discover so you don't have to figure them out again.\n\n"
            "**Rules:**\n"
            "* You must keep the total number of skills capped at 10. If you reach 10, use `delete_skill` to remove obsolete skills.\n"
            "* When using `save_skill`, you must start the summary with 'Use when...' and keep it extremely short (1 sentence max).\n\n"
            "**Current Available Skills:**\n"
        )
        
        skills_found = False
        try:
            for filename in os.listdir(self.skills_dir):
                if filename.endswith(".md"):
                    name = filename[:-3]
                    filepath = os.path.join(self.skills_dir, filename)
                    parsed = self._parse_skill_file(filepath)
                    prompt += f"* `{name}`: {parsed['summary']}\n"
                    skills_found = True
        except Exception as e:
            print(f"[SkillComponent] Error listing skills for prompt: {e}")
            
        if not skills_found:
            prompt += "(No skills currently saved.)\n"
            
        return prompt

    def get_tools(self) -> List[Callable]:
        return [
            self.list_skills,
            self.load_skill,
            self.save_skill,
            self.delete_skill
        ]

    def list_skills(self) -> List[Dict[str, str]]:
        """
        Lists all available skills and their summaries.
        """
        skills = []
        try:
            for filename in os.listdir(self.skills_dir):
                if filename.endswith(".md"):
                    name = filename[:-3]
                    filepath = os.path.join(self.skills_dir, filename)
                    parsed = self._parse_skill_file(filepath)
                    skills.append({"name": name, "summary": parsed['summary']})
        except Exception as e:
            return [{"error": str(e)}]
        return skills

    def load_skill(self, name: str) -> str:
        """
        Loads the full instruction text of a specific skill.
        """
        filepath = os.path.join(self.skills_dir, f"{name}.md")
        if not os.path.exists(filepath):
            return f"Error: Skill '{name}' does not exist."
            
        parsed = self._parse_skill_file(filepath)
        return parsed["text"]

    def save_skill(self, name: str, summary: str, text: str) -> str:
        """
        Saves or overwrites a skill. 
        The summary MUST start with 'Use when...' and be 1 sentence.
        """
        filepath = os.path.join(self.skills_dir, f"{name}.md")
        
        content = (
            "---\n"
            f"summary: {summary}\n"
            "---\n\n"
            f"{text}"
        )
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Check count for the warning message
            skill_count = len([f for f in os.listdir(self.skills_dir) if f.endswith(".md")])
            msg = f"Success: Skill '{name}' saved."
            if skill_count > 10:
                msg += f" WARNING: You now have {skill_count} skills. You MUST delete an old skill to respect your 10-skill cap."
            return msg
        except Exception as e:
            return f"Error saving skill: {e}"

    def delete_skill(self, name: str) -> str:
        """
        Deletes a skill permanently.
        """
        filepath = os.path.join(self.skills_dir, f"{name}.md")
        if not os.path.exists(filepath):
            return f"Error: Skill '{name}' does not exist."
            
        try:
            os.remove(filepath)
            return f"Success: Skill '{name}' deleted."
        except Exception as e:
            return f"Error deleting skill: {e}"
