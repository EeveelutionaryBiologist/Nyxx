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

from librarian import (
    load_librarian_model, 
    process_memory_chunk, 
    extract_entities_from_text, 
    librarian_summarize
)
from knowledge_graph import KnowledgeRelationshipGraph


# ==========================================
# 1. DIRECTORY SETUP & CONFIGURATION
# ==========================================
app = FastAPI(title="Nyxx Memory Microservice")

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "Memory"
CHROMA_PATH = MEMORY_DIR / "chromadb"
SQLITE_PATH = MEMORY_DIR / "metadata.db"
EMBEDDING_DIR = BASE_DIR / "Embedding"
GGUF_MODEL_PATH = EMBEDDING_DIR / "bge-base-en-v1.5-f16.gguf"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. INITIALIZE MODELS & GRAPH
# ==========================================
embedder = None

@app.on_event("startup")
def startup_event():
    """Runs on Uvicorn startup."""
    global embedder
    
    # Load Embedding Model
    if not GGUF_MODEL_PATH.exists():
        print("[SYSTEM] Downloading embedding model...")
        hf_hub_download(
            repo_id="CompendiumLabs/bge-base-en-v1.5-gguf",
            filename="bge-base-en-v1.5-f16.gguf",
            local_dir=EMBEDDING_DIR
        )
    print("[SYSTEM] Initializing Llama.cpp Embedder...")
    embedder = Llama(model_path=str(GGUF_MODEL_PATH), embedding=True, verbose=False)
    
    # Load background Librarian
    load_librarian_model()

def get_embedding(text: str) -> list[float]:
    response = embedder.create_embedding(text)
    return response["data"][0]["embedding"]

# Initialize Knowledge Graph
knowledge_graph = KnowledgeRelationshipGraph(str(MEMORY_DIR / "knowledge_graph.json"))

# ==========================================
# 3. DATABASE INITIALIZATION
# ==========================================
def init_sqlite():
    # check_same_thread=False is critical to prevent FastAPI threading crashes
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

sqlite_conn = init_sqlite()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name="nyxx_memory")

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
    """Processes raw text via Librarian, saving Atomic facts to ChromaDB and Triples to Graph."""
    now = datetime.now().isoformat()
    cursor = sqlite_conn.cursor()
    
    # 1. Ask Librarian to process the chunk
    processed_data = process_memory_chunk(memory.text)
    if not processed_data:
        raise HTTPException(status_code=500, detail="Librarian failed to process memory.")

    # 2. Store original raw chunk in SQLite just for posterity
    raw_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO memories (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
        (raw_id, memory.text, now, now)
    )
    sqlite_conn.commit()

    # 3. Save Atomic Facts to ChromaDB & SQLite
    for fact in processed_data.atomic_facts:
        fact_id = str(uuid.uuid4())
        vector = get_embedding(fact)
        
        collection.add(
            embeddings=[vector],
            documents=[fact],
            ids=[fact_id]
        )
        cursor.execute(
            "INSERT INTO memories (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
            (fact_id, fact, now, now)
        )
    sqlite_conn.commit()

    # 4. Save Triples to Knowledge Graph
    for triple in processed_data.triples:
        knowledge_graph.add_relationship(triple.subject, triple.predicate, triple.object)
        print(f"  -> Graph Mapped: {triple.subject} [{triple.predicate}] {triple.object}")
    
    return {
        "status": "success", 
        "message": f"Added {len(processed_data.atomic_facts)} standalone facts and {len(processed_data.triples)} graph relations."
    }

@app.post("/memory/search")
def search_memory(search: SearchQuery):
    """Searches ChromaDB (vectors) and Knowledge Graph (relations)."""
    now = datetime.now().isoformat()
    cursor = sqlite_conn.cursor()
    
    # --- 1. VECTOR SEARCH ---
    query_vector = get_embedding(search.query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=search.top_k
    )
    
    final_results = []
    if results['ids'] and results['ids'][0]:
        for mem_id in results['ids'][0]:
            cursor.execute("""
                UPDATE memories 
                SET hit_count = hit_count + 1, last_accessed = ? 
                WHERE id = ?
            """, (now, mem_id))
            sqlite_conn.commit()
            
            cursor.execute("SELECT content, hit_count FROM memories WHERE id = ?", (mem_id,))
            row = cursor.fetchone()
            if row:
                final_results.append({
                    "text": row[0],
                    "hit_count": row[1]
                })

    # --- 2. GRAPH RETRIEVAL ---
    relation_facts = [] # Initialize safely
    extracted = extract_entities_from_text(search.query)
    
    if extracted and hasattr(extracted, 'entities'):
        for entity in extracted.entities:
            facts = knowledge_graph.retrieve_relationships(entity.name, depth=1)
            if facts:
                relation_facts.extend(facts)
    
    summarized_context = ""
    if relation_facts:
        unique_facts = list(set(relation_facts)) # Deduplicate facts before summary
        # summarized_context = librarian_summarize(unique_facts) # <--- This eats quite a lot of power...
        summarized_context = "\n".join(unique_facts)
        
    return {
        "results": final_results,
        "relational_context": summarized_context
    }

@app.get("/memory/all")
def get_all_memories():
    """Retrieves all memories currently stored in the SQLite database."""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id, content, hit_count, created_at FROM memories ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return {"results": [{"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3]} for r in rows]}

@app.delete("/memory/clear")
def clear_all_memories():
    """Wipes ChromaDB, SQLite, and the Knowledge Graph completely."""
    # 1. Clear SQLite
    cursor = sqlite_conn.cursor()
    cursor.execute("DELETE FROM memories")
    sqlite_conn.commit()
    
    # 2. Clear ChromaDB
    global collection
    chroma_client.delete_collection("nyxx_memory")
    collection = chroma_client.create_collection("nyxx_memory")
    
    # 3. Clear Graph
    knowledge_graph.G.clear()
    knowledge_graph.write_graph()
    
    return {"status": "success", "message": "All databases and graphs wiped clean."}
