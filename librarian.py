
import json
from pathlib import Path
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from knowledge_graph import KnowledgeRelationshipGraph

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Embedding"
# Switched to Qwen2.5 (3B Instruct) 
LIBRARIAN_MODEL_PATH = MODEL_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"

# Global variable for the model
librarian_llm = None


class KnowledgeTriple(BaseModel):
    subject: str = Field(description="The main entity (e.g., 'User', 'Project X')")
    predicate: str = Field(description="The relationship (e.g., 'LIKES', 'WORKS_ON')")
    object: str = Field(description="The target entity (e.g., 'Python', 'Nyxx')")

class TripleExtraction(BaseModel):
    triples: list[KnowledgeTriple]

def load_librarian_model():
    """Downloads (if necessary) and loads the local Librarian model into RAM."""
    global librarian_llm
    if not LIBRARIAN_MODEL_PATH.exists():
        print("[SYSTEM] Downloading librarian background model...")
        hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            local_dir=MODEL_DIR,
        )
    print("[SYSTEM] Initializing Llama.cpp Librarian Model in RAM...")
    librarian_llm = Llama(
        model_path=str(LIBRARIAN_MODEL_PATH), 
        n_ctx=4096,
        n_gpu_layers=0,       # CRITICAL: Force CPU/RAM
        use_mlock=True,       # Prevents OS from swapping it out of your 96GB RAM
        verbose=False,
        chat_format="chatml"  # Crucial format setting for Qwen models
    )

def extract_triples_from_text(text: str) -> list[KnowledgeTriple]:
    """Tasks the Librarian model with extracting structured knowledge triples."""
    if librarian_llm is None:
        raise ValueError("Librarian model not loaded. Call load_librarian_model() first.")
        
    system_prompt = (
        "You are a strict data extraction AI. Your task is to extract "
        "Subject-Predicate-Object triples from the user's text. "
        "The 'subject' and 'object' should be concise nouns/entities (e.g., 'User', 'Max', 'Shoes'). "
        "The 'predicate' should be a capitalized relationship (e.g., 'OWNS', 'CHEWS_ON', 'WORKS_AT'). "
        "If no relations exist, return an empty list."
    )
    
    print(f"[LIBRARIAN] Extracting entities from: '{text[:50]}...'")
    
    # We use create_chat_completion and pass the Pydantic schema to force JSON output
    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract triples from this text: '{text}'"}
        ],
        response_format={
            "type": "json_object",
            "schema": TripleExtraction.model_json_schema()
        },
        temperature=0.1 # Very low temp for deterministic extraction
    )
    
    try:
        # Parse the guaranteed JSON output
        output_str = response['choices'][0]['message']['content']
        extracted_data = json.loads(output_str)
        
        # Validate through Pydantic
        validated_data = TripleExtraction(**extracted_data)
        return validated_data.triples
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Failed to parse triples: {e}")
        return []

def consolidate_memory_to_graph(memory_text: str, kg: KnowledgeRelationshipGraph):
    """
    The main pipeline function. Pass a raw memory string and your graph object here.
    """
    triples = extract_triples_from_text(memory_text)
    
    if not triples:
        print("[LIBRARIAN] No actionable relations found to map.")
        return 
        
    for triple in triples:
        kg.add_relationship(triple.subject, triple.predicate, triple.object)
        print(f"  -> Mapped Knowledge: {triple.subject} [{triple.predicate}] {triple.object}")
        
    kg.write_graph()
