import os
from pathlib import Path
import chromadb
import uuid
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ==========================================
# DIRECTORY/EMBEDDING MODEL INITIALIZATION 
# ==========================================

RAG_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = RAG_DIR / "Embedding"
GGUF_MODEL_PATH = EMBEDDING_DIR / "bge-base-en-v1.5-f16.gguf"
CHROMA_PATH = RAG_DIR / "Memory" / "chromadb"  # Recreates "Memory/chromadb" relative to RAG's parent folder
FACTS_PATH = RAG_DIR / "Facts"

EMBEDDING_MODEL_FILENAME = "bge-base-en-v1.5-f16.gguf"

# Ensure directory structures are fully created on initialization
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)

# Handle the download sequence if local embedding model is not found
if not GGUF_MODEL_PATH.exists():
    print(f"[SYSTEM] Model target not found. Downloading straight to: {GGUF_MODEL_PATH}")
    hf_hub_download(
        repo_id="CompendiumLabs/bge-base-en-v1.5-gguf",
        filename=EMBEDDING_MODEL_FILENAME,
        local_dir=EMBEDDING_DIR,
        local_dir_use_symlinks=False  # Pulls raw binaries, resolving partition linkage errors
    )
    print("[SYSTEM] Model download finished!")


# ==========================================
# CPU EMBEDDING HANDLER
# ==========================================

class LocalCPUEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_path: str):
        print("[SYSTEM] Initializing standalone CPU embedding worker...")
        self.model = Llama(
            model_path=model_path,
            embedding=True,
            n_gpu_layers=0,  
            verbose=False
        )

    def __call__(self, input: Documents) -> Embeddings:
        results = []
        for text in input:
            output = self.model.create_embedding(text)
            results.append(output["data"][0]["embedding"])
        return results


# ==========================================
# CHROMADB HANDLER
# ==========================================

class ChromaDBInterface:
    def __init__(self):
        self.db_path = CHROMA_PATH
        self.cpu_ef = LocalCPUEmbedding(model_path=str(GGUF_MODEL_PATH))
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name="agent_memories",
            embedding_function=self.cpu_ef,
            metadata={"hnsw:space": "cosine"}
        )
    
    # ==========================================
    # UTILITY AND RETRIEVAL FUNCTIONS
    # ==========================================
    def add_chunk_to_db(self, chunk: str, chunk_id: str = "", source: str = "user_declaration"):
        """Adds a single memory chunk to ChromaDB."""
        """It is not advised to set ID manually"""
        if len(chunk_id) > 0:
            local_uuid = chunk_id
        else:
            local_uuid = uuid.uuid1()

        self.memory_collection.upsert(
            documents=[chunk],
            ids=[str(local_uuid)],
            metadatas=[{"source": source}]
        )

    def db_retrieve(self, query: str, top_n: int = 5) -> list[str]:
        results = self.memory_collection.query(
            query_texts=[query],
            n_results=top_n
        )
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        return []

    def delete_chunk_from_db(self, chunk_id: str):
        """Removes a specific entry from ChromaDB using its ID."""
        try:
            self.memory_collection.delete(ids=[chunk_id])
            print(f"[SYSTEM] Successfully deleted memory with ID: {chunk_id}")
        except Exception as e:
            print(f"[ERROR] Failed to delete entry {chunk_id}: {e}")

    def initialize_db(self):
        """Seeds the DB if it is currently empty."""
        facts_to_learn = parse_facts_to_learn()

        if self.memory_collection.count() == 0:
            print("[SYSTEM] Memory database empty. Seeding initial facts...")
            for i, chunk in enumerate(facts_to_learn):
                self.add_chunk_to_db(chunk)
                print(f"Committed fact #{i}: [{chunk}] to memory.")
        else:
            print(f"[SYSTEM] Loaded {self.memory_collection.count()} existing memories from disk.")

    def db_length(self):
        return self.memory_collection.count()

    def get_all_memories(self):
        """Retrieves all documents and their metadata from the collection."""
        # include=['documents', 'metadatas'] ensures you get the text and the source info
        results = self.memory_collection.get(include=['documents', 'metadatas'])
        if results['documents']:
            # Zip them together into a more readable format (e.g., list of dicts)
            return [
                {"id": doc_id, "text": doc, "metadata": meta}
                for doc_id, doc, meta in zip(results['ids'], results['documents'], results['metadatas'])
            ]
        return []
    

def parse_facts_to_learn():
    with open(FACTS_PATH / "facts.txt", 'r') as f:
        parsed_facts = [line for line in f.readlines() if len(line) > 0]
    return parsed_facts


# initialize_db()