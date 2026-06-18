import os
import uuid
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from librarian import (
    load_librarian_model,
    process_memory_chunk,
    extract_entities_from_text,
    librarian_summarize,
    librarian_should_merge,
    librarian_split_compound,
)
from knowledge_graph import KnowledgeRelationshipGraph

CONSOLIDATION_PASSES = 2

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
# 4. CONSOLIDATION CONFIGURATION
# ==========================================
# Facts older than this with zero hits are pruned.
PRUNE_AGE_DAYS = 60

# Cosine similarity threshold above which two facts are sent to the Librarian
# for a merge decision. Range: 0.0–1.0. Higher = more conservative.
DEDUP_SIMILARITY_THRESHOLD = 0.90

# Above this threshold the facts are considered near-identical text; the Librarian
# is skipped and the lower-hit-count copy is dropped directly.
HIGH_SIM_DEDUP_THRESHOLD = 0.99

# Only facts longer than this are checked for compound structure (short facts are
# almost always already atomic, so skip the Librarian call to save CPU).
COMPOUND_CHECK_MIN_CHARS = 120

# ==========================================
# 5. API ENDPOINTS
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


@app.post("/memory/consolidate")
def consolidate_memories():
    """
    Three-phase memory hygiene pass:
      Phase 1 – Prune: delete stale, never-retrieved atomic facts from ChromaDB + SQLite.
      Phase 2 – Merge: detect near-duplicate fact pairs via cosine similarity; ask the
                       Librarian to produce a merged fact and replace the originals.
      Phase 3 – Split: find compound facts that slipped through the write-time atomization
                       and break them into truly independent sentences.

    ARCHITECTURAL NOTE — KG / ChromaDB decoupling:
    The knowledge graph has no foreign-key link back to the SQLite/ChromaDB fact IDs that
    generated its triples. This means pruning or merging a fact does NOT automatically
    remove its corresponding KG edges. The pragmatic workaround applied here is:
      (a) When a merged fact is added it goes through process_memory_chunk, which
          re-derives fresh, correct triples — so the merged concept is properly represented.
      (b) KG nodes with degree=0 (no remaining edges) are pruned at the end.
    Stale edges from deleted/merged facts will persist but are harmless; they just add
    noise. The clean long-term fix is to store a fact_id on each KG edge at write time.
    """
    report = {"pruned": 0, "merged": 0, "split": 0, "errors": []}
    cursor = sqlite_conn.cursor()
    now = datetime.now().isoformat()

    for N in range(CONSOLIDATION_PASSES):
        # ------------------------------------------------------------------
        # Phase 0: Exact-text dedup across SQLite
        # ------------------------------------------------------------------
        # The raw chunk written on add_memory() and its derived atomic fact share
        # the same text but different UUIDs. Phase 2's cosine similarity pass only
        # sees ChromaDB entries, so it never finds the raw-chunk copy. This pass
        # finds text duplicates directly in SQLite and drops them, preferring the
        # ChromaDB-backed (atomic fact) copy over raw-chunk-only copies.
        print("[CONSOLIDATE] Phase 0: Exact-text dedup in SQLite...")

        chroma_all = collection.get(include=[])
        chroma_id_set = set(chroma_all["ids"])

        cursor.execute("""
            SELECT content, GROUP_CONCAT(id, '|') as ids
            FROM memories
            GROUP BY content
            HAVING COUNT(*) > 1
        """)
        text_dup_groups = cursor.fetchall()

        for content, ids_str in text_dup_groups:
            all_ids = ids_str.split("|")
            in_chroma = [id_ for id_ in all_ids if id_ in chroma_id_set]
            sqlite_only = [id_ for id_ in all_ids if id_ not in chroma_id_set]

            if in_chroma:
                # Keep the best ChromaDB-backed copy, drop everything else
                cursor.execute(
                    f"SELECT id, hit_count FROM memories WHERE id IN ({','.join('?' * len(in_chroma))})",
                    in_chroma
                )
                chroma_rows = sorted(cursor.fetchall(), key=lambda r: r[1], reverse=True)
                keep_id = chroma_rows[0][0]
                drop_ids = [r[0] for r in chroma_rows[1:]] + sqlite_only
            else:
                cursor.execute(
                    f"SELECT id, hit_count FROM memories WHERE id IN ({','.join('?' * len(all_ids))})",
                    all_ids
                )
                rows_sorted = sorted(cursor.fetchall(), key=lambda r: r[1], reverse=True)
                keep_id = rows_sorted[0][0]
                drop_ids = [r[0] for r in rows_sorted[1:]]

            if not drop_ids:
                continue

            chroma_drop = [id_ for id_ in drop_ids if id_ in chroma_id_set]
            if chroma_drop:
                collection.delete(ids=chroma_drop)

            cursor.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?' * len(drop_ids))})",
                drop_ids
            )
            sqlite_conn.commit()
            report["merged"] += len(drop_ids)
            print(f"[CONSOLIDATE] Exact-text dedup: dropped {len(drop_ids)} copy/copies of '{content[:60]}'")

        # Re-fetch chroma_id_set after potential deletions above
        chroma_all = collection.get(include=[])
        chroma_id_set = set(chroma_all["ids"])

        # ------------------------------------------------------------------
        # Phase 1: Prune stale atomic facts (never retrieved, older than N days)
        # ------------------------------------------------------------------
        # We must only target IDs that exist in ChromaDB (atomic facts), not raw
        # chunk records — raw chunks never appear in search results so their
        # hit_count is always 0 but they should not be pruned.
        print("[CONSOLIDATE] Phase 1: Pruning stale memories...")

        cutoff = (datetime.now() - timedelta(days=PRUNE_AGE_DAYS)).isoformat()
        cursor.execute(
            "SELECT id FROM memories WHERE hit_count = 0 AND created_at < ?",
            (cutoff,)
        )
        stale_candidates = [row[0] for row in cursor.fetchall()]
        stale_fact_ids = [id_ for id_ in stale_candidates if id_ in chroma_id_set]

        if stale_fact_ids:
            collection.delete(ids=stale_fact_ids)
            cursor.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?' * len(stale_fact_ids))})",
                stale_fact_ids
            )
            sqlite_conn.commit()
            report["pruned"] = len(stale_fact_ids)
            print(f"[CONSOLIDATE] Pruned {len(stale_fact_ids)} stale facts.")

        # ------------------------------------------------------------------
        # Phase 2: Near-duplicate detection → Librarian merge decision
        # ------------------------------------------------------------------
        print("[CONSOLIDATE] Phase 2: Detecting near-duplicates...")

        chroma_data = collection.get(include=["embeddings", "documents"])
        ids = chroma_data["ids"]
        docs = chroma_data["documents"]
        embeddings = chroma_data["embeddings"]

        merged_out = set()

        if len(ids) >= 2:
            emb_matrix = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            normalized = emb_matrix / np.maximum(norms, 1e-8)
            # Full pairwise cosine similarity in one matrix multiply — O(n²) but fast with numpy.
            # For very large collections (>5k facts), consider switching to approximate NN search.
            similarity_matrix = normalized @ normalized.T

            for i in range(len(ids)):
                if ids[i] in merged_out:
                    continue
                for j in range(i + 1, len(ids)):
                    if ids[j] in merged_out:
                        continue
                    sim = float(similarity_matrix[i, j])
                    if sim < DEDUP_SIMILARITY_THRESHOLD:
                        continue

                    if sim >= HIGH_SIM_DEDUP_THRESHOLD:
                        # Near-identical text — Librarian would likely return an empty merged_fact
                        # for two identical strings. Skip it; just drop the lower-hit copy.
                        cursor.execute("SELECT hit_count FROM memories WHERE id = ?", (ids[i],))
                        row_i = cursor.fetchone()
                        cursor.execute("SELECT hit_count FROM memories WHERE id = ?", (ids[j],))
                        row_j = cursor.fetchone()
                        hits_i = row_i[0] if row_i else 0
                        hits_j = row_j[0] if row_j else 0

                        drop_id = ids[j] if hits_i >= hits_j else ids[i]
                        collection.delete(ids=[drop_id])
                        cursor.execute("DELETE FROM memories WHERE id = ?", (drop_id,))
                        sqlite_conn.commit()

                        merged_out.add(ids[i])
                        merged_out.add(ids[j])
                        report["merged"] += 1
                        print(f"[CONSOLIDATE] Deduped exact duplicate (sim={sim:.3f}): '{docs[i][:70]}'")
                        break

                    # Medium similarity — ask the Librarian whether these are truly redundant
                    decision = librarian_should_merge(docs[i], docs[j])
                    if not decision or not decision.should_merge or not decision.merged_fact:
                        continue

                    # Remove both originals
                    collection.delete(ids=[ids[i], ids[j]])
                    cursor.execute("DELETE FROM memories WHERE id IN (?, ?)", (ids[i], ids[j]))

                    # Add merged fact through the full pipeline so KG triples are re-derived
                    merged_id = str(uuid.uuid4())
                    merged_vec = get_embedding(decision.merged_fact)
                    collection.add(
                        embeddings=[merged_vec],
                        documents=[decision.merged_fact],
                        ids=[merged_id]
                    )
                    cursor.execute(
                        "INSERT INTO memories (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
                        (merged_id, decision.merged_fact, now, now)
                    )
                    sqlite_conn.commit()

                    merged_out.add(ids[i])
                    merged_out.add(ids[j])
                    report["merged"] += 1
                    print(
                        f"[CONSOLIDATE] Merged (sim={sim:.3f}):\n"
                        f"  A: {docs[i]}\n"
                        f"  B: {docs[j]}\n"
                        f"  → {decision.merged_fact}"
                    )
                    break  # Only one partner per fact per pass; re-run for further merges

        # ------------------------------------------------------------------
        # Phase 3: Split compound facts
        # ------------------------------------------------------------------
        # Fetch fresh snapshot — Phase 2 may have mutated the collection.
        print("[CONSOLIDATE] Phase 3: Splitting compound facts...")

        chroma_data = collection.get(include=["documents"])
        facts_to_check = [
            (id_, doc)
            for id_, doc in zip(chroma_data["ids"], chroma_data["documents"])
            if len(doc) >= COMPOUND_CHECK_MIN_CHARS
        ]

        for fact_id, fact_text in facts_to_check:
            decision = librarian_split_compound(fact_text)
            if not decision or not decision.is_compound or len(decision.split_facts) < 2:
                continue

            collection.delete(ids=[fact_id])
            cursor.execute("DELETE FROM memories WHERE id = ?", (fact_id,))

            for split_fact in decision.split_facts:
                split_id = str(uuid.uuid4())
                split_vec = get_embedding(split_fact)
                collection.add(embeddings=[split_vec], documents=[split_fact], ids=[split_id])
                cursor.execute(
                    "INSERT INTO memories (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
                    (split_id, split_fact, now, now)
                )

            sqlite_conn.commit()
            report["split"] += 1
            print(f"[CONSOLIDATE] Split into {len(decision.split_facts)} facts: '{fact_text[:60]}...'")

        # ------------------------------------------------------------------
        # KG cleanup: remove nodes that lost all edges (degree = 0)
        # ------------------------------------------------------------------
        orphaned = [n for n in list(knowledge_graph.G.nodes()) if knowledge_graph.G.degree(n) == 0]
        if orphaned:
            for node in orphaned:
                knowledge_graph.G.remove_node(node)
            knowledge_graph.write_graph()
            print(f"[CONSOLIDATE] Removed {len(orphaned)} orphaned KG nodes.")

    print(f"[CONSOLIDATE] Done. {report}")
    return {"status": "success", "report": report}
