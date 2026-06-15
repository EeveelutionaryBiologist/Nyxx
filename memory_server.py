
import os
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
app = FastAPI(title="Nyxx Memory Microservice")

# Directories based on your existing architecture
BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "Memory"
CHROMA_PATH = MEMORY_DIR / "chromadb"
SQLITE_PATH = MEMORY_DIR / "metadata.db"
EMBEDDING_DIR = BASE_DIR / "Embedding"
GGUF_MODEL_PATH = EMBEDDING_DIR / "bge-base-en-v1.5-f16.gguf"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. DATABASE INITIALIZATION
# ==========================================
def init_sqlite():
    """Creates the SQLite database for metadata and hit-counters if it doesn't exist."""
    # Add check_same_thread=False here!
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            hit_count INTEGER DEFAULT 0,
            created_at DATETIME,
            last_accessed DATETIME
        )
    ''')
    conn.commit()
    return conn

# Global variables for DB connections and models
sqlite_conn = init_sqlite()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name="nyxx_memory")
embedder = None

# ==========================================
# 3. EMBEDDING MODEL SETUP
# ==========================================
@app.on_event("startup")
def load_embedding_model():
    """Downloads (if necessary) and loads the local embedding model on server startup."""
    global embedder
    if not GGUF_MODEL_PATH.exists():
        print("[SYSTEM] Downloading embedding model...")
        hf_hub_download(
            repo_id="CompendiumLabs/bge-base-en-v1.5-gguf",
            filename="bge-base-en-v1.5-f16.gguf",
            local_dir=EMBEDDING_DIR,
        )
    print("[SYSTEM] Initializing Llama.cpp Embedder...")
    embedder = Llama(
        model_path=str(GGUF_MODEL_PATH), 
        n_gpu_layers=0,
        embedding=True, 
        verbose=False,
        use_mlock=True,
    )

def get_embedding(text: str) -> list[float]:
    """Generates a vector embedding for a given text."""
    response = embedder.create_embedding(text)
    return response["data"][0]["embedding"]

# ==========================================
# 4. API ENDPOINTS
# ==========================================
class MemoryInput(BaseModel):
    text: str

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

@app.post("/memory/add")
def add_memory(memory: MemoryInput):
    """Saves a new memory into both ChromaDB and SQLite."""
    mem_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    # 1. Generate Embedding
    vector = get_embedding(memory.text)
    
    # 2. Save to ChromaDB (for vector search)
    collection.add(
        embeddings=[vector],
        documents=[memory.text],
        ids=[mem_id]
    )
    
    # 3. Save to SQLite (for metadata & hit counter)
    cursor = sqlite_conn.cursor()
    cursor.execute(
        "INSERT INTO memories (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
        (mem_id, memory.text, now, now)
    )
    sqlite_conn.commit()
    
    return {"status": "success", "id": mem_id, "message": "Memory added."}

@app.post("/memory/search")
def search_memory(search: SearchQuery):
    """Searches for memories and increments their hit-counter."""
    # 1. Generate query embedding
    query_vector = get_embedding(search.query)
    
    # 2. Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=search.top_k
    )
    
    if not results['ids'] or not results['ids'][0]:
        return {"results": []}
    
    found_ids = results['ids'][0]
    now = datetime.now().isoformat()
    final_results = []
    
    # 3. Retrieve metadata from SQLite and update hit-counter
    cursor = sqlite_conn.cursor()
    for mem_id in found_ids:
        # Increment the counter and update the timestamp
        cursor.execute("""
            UPDATE memories 
            SET hit_count = hit_count + 1, last_accessed = ? 
            WHERE id = ?
        """, (now, mem_id))
        sqlite_conn.commit()
        
        # Fetch the updated row
        cursor.execute("SELECT content, hit_count, created_at FROM memories WHERE id = ?", (mem_id,))
        row = cursor.fetchone()
        
        if row:
            final_results.append({
                "id": mem_id,
                "text": row[0],
                "hit_count": row[1],
                "created_at": row[2]
            })
            
    return {"results": final_results}

@app.post("/memory/consolidate")
def trigger_consolidation():
    """
    Placeholder endpoint for Memory Consolidation.
    This will eventually iterate over SQLite, find high-hit memories, 
    and synthesize them using an LLM.
    """
    cursor = sqlite_conn.cursor()
    # Example: Find all memories accessed more than 5 times
    cursor.execute("SELECT id, content, hit_count FROM memories WHERE hit_count > 5")
    frequent_memories = cursor.fetchall()
    
    # TODO: Implement LLM summarization and clustering logic here
    
    return {
        "status": "pending", 
        "message": "Consolidation logic under construction.",
        "frequent_memories_found": len(frequent_memories)
    }

@app.get("/memory/all")
def get_all_memories():
    """Retrieves all memories currently stored in the SQLite database."""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id, content, hit_count, created_at FROM memories ORDER BY created_at DESC")
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "text": row[1],
            "hit_count": row[2],
            "created_at": row[3]
        })
    return {"results": results}

@app.delete("/memory/clear")
def clear_all_memories():
    """Wipes both ChromaDB and SQLite databases completely."""
    # 1. Clear SQLite
    cursor = sqlite_conn.cursor()
    cursor.execute("DELETE FROM memories")
    sqlite_conn.commit()
    
    # 2. Clear ChromaDB
    # ChromaDB doesn't have a simple 'delete all' without IDs, so we delete and recreate the collection
    global collection
    chroma_client.delete_collection("nyxx_memory")
    collection = chroma_client.create_collection("nyxx_memory")
    
    return {"status": "success", "message": "All memories deleted."}
