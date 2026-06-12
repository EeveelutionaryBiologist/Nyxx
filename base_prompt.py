
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_PROMPT_DIR = THIS_DIR / "Personality"

def load_base_prompt() -> str:
    try:
        with open(BASE_PROMPT_DIR / "Personality.md") as f:
            prompt = f.read().strip()
    except Exception as e:
        print("[SYSTEM] Critical: Personality.md not found. Was it moved?")
        sys.exit()
    
    return prompt


BASE_PROMPT = load_base_prompt()
