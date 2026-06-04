
import os
import sys
import json
from openai import OpenAI

from base_prompt import BASE_PROMPT

MODEL_NAME = "gemma4:26b"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CURRENT_DIR = os.getcwd()

URL_DICT = {
    "GEMINI": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "OPENAI": "",
    "ANTHROPIC": "",
    "OLLAMA": "http://localhost:11434/v1",
}


# from pydantic_ai import Agent, RunContext



def load_environment_dict() -> str:
    # TODO: Decoding from encrypt later - right now irrelevant as on local machine
    with open('env.json') as json_file:
        data = json.load(json_file)

    return data


def initialize_client():
    env = load_environment_dict()

    for key in ["GEMINI_API_KEY", "OPENAI_KEY", "ANTHROPIC_KEY"]:
        api_key = env.get("GEMINI_API_KEY", "")
        base_url = URL_DICT.get(key.split("_")[0], "")

        if len(api_key) == 0:
            continue
        else:
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                return client
            except Exception as e:
                print(f"[ERROR] Could not open client: \n{e}")
                sys.exit()
    
    # Alternatively, use local model as default
    # TODO: Make switching of local models more intuitive
    base_url = env.get("OLLAMA_URL", "")

    if len(base_url) == 0:
        base_url = URL_DICT.get("OLLAMA")

    try:
        client = OpenAI(
            base_url= base_url, 
            api_key="ollama"  # Required string parameter by the SDK, though ignored by Ollama
        )
    except Exception as e:
        print(f"[ERROR] Could not open client: \n{e}")
        sys.exit()
    return client


CLIENT = initialize_client()
