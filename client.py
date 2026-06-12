
import os
import sys
import json
from pathlib import Path
from openai import OpenAI


THIS_DIR = Path(__file__).resolve().parent

def load_environment_dict() -> str:
    # TODO: Decoding from encrypt later - right now irrelevant as on local machine
    with open(THIS_DIR / "env.json") as json_file:
        data = json.load(json_file)

    return data


def initialize_client():
    env = load_environment_dict()

    for provider in ["GEMINI", "OPENAI", "ANTHROPIC"]:
        api_key = env.get(f"{provider}_API_KEY", "")

        if len(api_key) == 0:
            continue

        base_url = env.get(provider).get("BASE_URL")
        model = env.get(provider).get("MODEL_NAME")

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            return client, model
        except Exception as e:
            print(f"[ERROR] Could not open client: \n{e}")
            continue
    
    # Alternatively, use local model as default
    # TODO: Make switching of local models more intuitive
    base_url = env.get("OLLAMA").get("BASE_URL")
    model = env.get("OLLAMA").get("MODEL_NAME")

    try:
        client = OpenAI(
            base_url= base_url, 
            api_key="ollama"  # Required string parameter by the SDK, though ignored by Ollama
        )
    except Exception as e:
        print(f"[ERROR] Could not open client: \n{e}")
        sys.exit()
    return client, model


CLIENT, MODEL_NAME = initialize_client()
