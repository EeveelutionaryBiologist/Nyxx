
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BASE_PROMPT_DIR = THIS_DIR / "Personality"

fallback_prompt = "You are an agentic AI assistant. Execute tools when local resources are requested."


def load_base_prompt() -> str:
    try:
        with open(BASE_PROMPT_DIR / "PERSONALITY.md") as f:
            prompt = f.read().strip()
    except Exception:
        print("[SYSTEM] Personality.md not found. Was it moved? \nUsing Fallback...")
        prompt = fallback_prompt
    
    return prompt


BASE_PROMPT = load_base_prompt()
