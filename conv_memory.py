
import calendar
import re
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet
from RAG import ChromaDBInterface

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "i", "you", "it",
    "to", "of", "in", "on", "and", "or", "but", "my", "your", "can",
    "do", "how", "what", "why", "when", "that", "this", "with", "for",
    "be", "have", "has", "had", "just", "so", "if", "at", "up", "me",
    "we", "they", "he", "she", "not", "no", "yes", "its", "as", "by"
}


class ConversationalMemory:
    def __init__(self, chroma_interface: ChromaDBInterface, metadata_db_path: Path):
        self._fernet = self._load_or_create_key(metadata_db_path.parent / ".conv_key")

        # Embeddings are stored separately from documents, so we can encrypt the text
        # while keeping the vector index computed from plaintext.
        self._cpu_ef = chroma_interface.cpu_ef
        self.collection = chroma_interface.chroma_client.get_or_create_collection(
            name="conversation_history",
            metadata={"hnsw:space": "cosine"}
            # No embedding_function here — we pass pre-computed embeddings on every upsert
        )
        self.db_path = metadata_db_path
        self._retrieval_count = 0
        self._init_tracking_db()

    def _load_or_create_key(self, key_path: Path) -> Fernet:
        if key_path.exists():
            key = key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            key_path.write_bytes(key)
            key_path.chmod(0o600)
            print(f"[SYSTEM] Conv memory key generated at {key_path}")
        return Fernet(key)

    def _init_tracking_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conv_chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    session_id TEXT,
                    topic TEXT,
                    hit_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    last_accessed TEXT
                )
            """)

    def store_chunk(self, text: str, source: str, session_id: str):
        if not text or not text.strip():
            return

        chunk_id = str(uuid.uuid1())
        topic = self._extract_topic(text)
        now = datetime.now().isoformat()

        # Compute embedding from plaintext, then encrypt the document
        embedding = self._cpu_ef([text])[0]
        ciphertext = self._fernet.encrypt(text.encode()).decode()

        self.collection.upsert(
            documents=[ciphertext],
            embeddings=[embedding],
            ids=[chunk_id],
            metadatas=[{"source": source, "session_id": session_id, "topic": topic, "created_at": now}]
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conv_chunks (id, source, session_id, topic, hit_count, created_at, last_accessed) VALUES (?, ?, ?, ?, 0, ?, ?)",
                (chunk_id, source, session_id, topic, now, now)
            )

    def _prune_stale(self):
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            stale_ids = [
                row[0] for row in conn.execute(
                    "SELECT id FROM conv_chunks WHERE last_accessed < ?", (cutoff,)
                ).fetchall()
            ]
        if not stale_ids:
            return
        self.collection.delete(ids=stale_ids)
        placeholders = ",".join("?" * len(stale_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DELETE FROM conv_chunks WHERE id IN ({placeholders})", stale_ids)
        print(f"[SYSTEM] Conv memory pruned {len(stale_ids)} stale entr{'y' if len(stale_ids) == 1 else 'ies'} (>30 days unretrieved).")

    def retrieve_similar(self, query: str, top_k: int, current_session_id: str) -> list[dict]:
        self._retrieval_count += 1
        if self._retrieval_count % 100 == 0:
            self._prune_stale()

        total_count = self.collection.count()
        if total_count == 0:
            return []

        fetch_n = min(top_k * 3, total_count)

        # Query using a pre-computed embedding so ChromaDB doesn't try to embed the query text itself
        query_embedding = self._cpu_ef([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            include=["documents", "metadatas"]
        )

        ids = results["ids"][0]
        ciphertexts = results["documents"][0]
        metas = results["metadatas"][0]

        # Filter out chunks from the current session
        candidates = [
            (chunk_id, ciphertext, meta)
            for chunk_id, ciphertext, meta in zip(ids, ciphertexts, metas)
            if meta.get("session_id") != current_session_id
        ]

        if not candidates:
            return []

        # Join with sqlite for hit_count
        candidate_ids = [c[0] for c in candidates]
        placeholders = ",".join("?" * len(candidate_ids))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, hit_count FROM conv_chunks WHERE id IN ({placeholders})",
                candidate_ids
            ).fetchall()
        hit_map = {row[0]: row[1] for row in rows}

        # Sort by hit_count descending, take top_k
        candidates.sort(key=lambda c: hit_map.get(c[0], 0), reverse=True)
        top = candidates[:top_k]

        # Update hit_count and last_accessed for returned entries
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for chunk_id, _, _ in top:
                conn.execute(
                    "UPDATE conv_chunks SET hit_count = hit_count + 1, last_accessed = ? WHERE id = ?",
                    (now, chunk_id)
                )

        # Decrypt and return
        output = []
        for chunk_id, ciphertext, meta in top:
            output.append({
                "text": self._decrypt(ciphertext),
                "source": meta.get("source", "?"),
                "topic": meta.get("topic", ""),
                "created_at": meta.get("created_at", ""),
                "hit_count": hit_map.get(chunk_id, 0)
            })
        return output

    def _parse_date_hint(self, hint: str) -> str | None:
        """Returns an ISO date string 'YYYY-MM-DD' or None if unparseable."""
        hint = hint.strip().lower()
        today = date.today()

        if hint == "today":
            return today.isoformat()
        if hint == "yesterday":
            return (today - timedelta(days=1)).isoformat()

        try:
            return date.fromisoformat(hint).isoformat()
        except ValueError:
            pass

        # "June 17", "Jun 17", "june 17", etc.
        month_map = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
        month_map.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

        parts = hint.split()
        if len(parts) == 2:
            month_str, day_str = parts[0], parts[1].rstrip(".,")
            if month_str in month_map and day_str.isdigit():
                month = month_map[month_str]
                day = int(day_str)
                year = today.year
                try:
                    d = date(year, month, day)
                    if d > today:
                        d = date(year - 1, month, day)
                    return d.isoformat()
                except ValueError:
                    pass

        return None

    def retrieve_by_time(self, date_hint: str, keyword: str = "", top_k: int = 20) -> str:
        """Returns a formatted conversation history string for the model to read."""
        hint = date_hint.strip().lower()

        with sqlite3.connect(self.db_path) as conn:
            if hint in ("", "recent", "last session", "last time"):
                row = conn.execute(
                    "SELECT session_id FROM conv_chunks ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                if not row:
                    return "No conversation history found."
                session_id = row[0]
                db_rows = conn.execute(
                    "SELECT id, source, topic, created_at FROM conv_chunks WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,)
                ).fetchall()
            else:
                date_str = self._parse_date_hint(hint)
                if not date_str:
                    return f"Could not interpret date '{date_hint}'. Try: 'last time', 'yesterday', 'June 17', or 'YYYY-MM-DD'."
                db_rows = conn.execute(
                    "SELECT id, source, topic, created_at FROM conv_chunks WHERE created_at LIKE ? ORDER BY created_at ASC",
                    (f"{date_str}%",)
                ).fetchall()

        if not db_rows:
            return "No conversation history found for that time period."

        # Keyword filter on topic (fast, no embedding needed)
        if keyword:
            kw = keyword.lower()
            db_rows = [r for r in db_rows if kw in r[2].lower()]
            if not db_rows:
                return f"No conversation history found for that period matching '{keyword}'."

        db_rows = db_rows[:top_k]
        ids = [r[0] for r in db_rows]
        id_to_meta = {r[0]: r for r in db_rows}

        chroma_result = self.collection.get(ids=ids, include=["documents"])

        header_date = db_rows[0][3][:10]
        lines = [f"[CONVERSATION HISTORY — {header_date}]\n"]

        for chunk_id, ciphertext in zip(chroma_result["ids"], chroma_result["documents"]):
            meta = id_to_meta.get(chunk_id)
            if not meta:
                continue
            _, source, _, created_at = meta
            text = self._decrypt(ciphertext)
            time_str = created_at[11:16] if len(created_at) > 16 else ""
            lines.append(f"[{time_str} | {source}] {text}")

        lines.append(f"\n({len(ids)} chunk(s) found)")
        return "\n".join(lines)

    def _decrypt(self, document: str) -> str:
        """Decrypt a stored document. Returns raw text for pre-encryption legacy entries."""
        if not document.startswith("gAAAAA"):
            return document  # Legacy plaintext stored before encryption was added
        try:
            return self._fernet.decrypt(document.encode()).decode()
        except Exception:
            return "[unreadable: encryption key mismatch]"

    def _extract_topic(self, text: str) -> str:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        content_words = [w for w in words if w not in STOPWORDS]
        chosen = content_words[:5] if content_words else words[:5]
        return " ".join(chosen) if chosen else "general"
