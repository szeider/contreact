"""Similarity detection for memory deduplication."""

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import create_embedding_provider


class SimilarityAdvisor:
    """Embedding-based similarity detection for memories."""

    def __init__(self, db_path: Path, embedding_config: dict):
        """Initialize similarity advisor.

        Args:
            db_path: Path to SQLite database
            embedding_config: Configuration dict for embedding provider
                - provider: "openrouter" or "local"
                - model: Model identifier
                - thresholds: Similarity thresholds (optional)

        Raises:
            ValueError: If embedding model mismatch detected
        """
        self.db_path = Path(db_path)
        self.provider = create_embedding_provider(embedding_config)
        self.thresholds = embedding_config.get("thresholds", {})
        self._cache = None  # Cached embeddings

        # Initialize embeddings table (must be before validation)
        self._init_schema()

        # Validate model matches existing embeddings
        self.validate_model()

    def _get_connection(self) -> sqlite3.Connection:
        """Create SQLite connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")  # Enable foreign key enforcement
        return conn

    def _init_schema(self) -> None:
        """Initialize embeddings table if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (key) REFERENCES memories(key) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def validate_model(self) -> None:
        """Validate embeddings in database match configured model.

        Raises:
            ValueError: If model or dimension mismatch detected
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT model, embedding_dim
                FROM memory_embeddings
            """).fetchall()

            if not rows:
                return  # No embeddings yet

            # Check for consistency
            db_model = rows[0]["model"]
            db_dim = rows[0]["embedding_dim"]
            config_model = self.provider.model_name()
            config_dim = self.provider.dimension()

            if len(rows) > 1:
                raise ValueError(
                    "Multiple embedding models found in database! "
                    "Please delete embeddings: DELETE FROM memory_embeddings"
                )

            if db_model != config_model or db_dim != config_dim:
                raise ValueError(
                    f"\n{'=' * 60}\n"
                    f"Embedding Model Mismatch!\n"
                    f"{'=' * 60}\n"
                    f"Database embeddings: {db_model} ({db_dim} dimensions)\n"
                    f"Config specifies:    {config_model} ({config_dim} dimensions)\n\n"
                    f"Options:\n"
                    f"1. Revert config to match database:\n"
                    f"   \"model\": \"{db_model}\"\n\n"
                    f"2. Delete existing embeddings (keeps memories):\n"
                    f"   DELETE FROM memory_embeddings;\n\n"
                    f"3. Delete all memories and embeddings:\n"
                    f"   DELETE FROM memories;\n"
                    f"{'=' * 60}"
                )

    def add_embedding(self, key: str, value: str) -> None:
        """Generate and store embedding for memory.

        Args:
            key: Memory key
            value: Memory value to embed
        """
        # Generate embedding
        embedding = self.provider.encode(value)

        # Store in database
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_embeddings
                (key, embedding, model, embedding_dim)
                VALUES (?, ?, ?, ?)
            """, (
                key,
                embedding.tobytes(),
                self.provider.model_name(),
                self.provider.dimension()
            ))
            conn.commit()

        # Update cache
        if self._cache is not None:
            self._cache[key] = embedding

    def _load_all_embeddings(self) -> dict[str, np.ndarray]:
        """Load all embeddings from database into memory cache.

        Returns:
            Dict mapping keys to embedding vectors
        """
        if self._cache is None:
            self._cache = {}

            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT key, embedding, embedding_dim
                    FROM memory_embeddings
                """).fetchall()

                for row in rows:
                    embedding_bytes = row["embedding"]
                    dim = row["embedding_dim"]
                    # Reconstruct numpy array from bytes
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    if embedding.shape[0] != dim:
                        # Reshape if needed
                        embedding = embedding.reshape(dim)
                    self._cache[row["key"]] = embedding

        return self._cache

    def find_similar(
        self,
        text: str,
        exclude_keys: Optional[list[str]] = None,
        min_similarity: Optional[float] = None
    ) -> list[dict]:
        """Find similar memories using cosine similarity.

        Args:
            text: Text to find similar memories for
            exclude_keys: Keys to exclude from results (e.g., current key)
            min_similarity: Minimum similarity threshold (default from config)

        Returns:
            List of dicts with keys: key, similarity, preview
            Sorted by similarity (highest first)
        """
        # Load all embeddings
        all_embeddings = self._load_all_embeddings()

        if not all_embeddings:
            return []

        # Get minimum similarity threshold
        if min_similarity is None:
            min_similarity = self.thresholds.get("min_similarity", 0.5)

        # Generate embedding for query text
        query_embedding = self.provider.encode(text)

        # Calculate similarities
        results = []
        for key, embedding in all_embeddings.items():
            if exclude_keys and key in exclude_keys:
                continue

            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]

            if similarity >= min_similarity:
                # Get preview from database
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT value FROM memories WHERE key = ?",
                        (key,)
                    ).fetchone()
                    preview = row["value"][:120] if row else ""

                results.append({
                    "key": key,
                    "similarity": float(similarity),
                    "preview": preview
                })

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results

    def post_write_feedback(self, key: str, value: str) -> Optional[str]:
        """Generate similarity feedback after writing memory.

        Args:
            key: Memory key that was written
            value: Memory value

        Returns:
            Formatted feedback string or None if no similar memories
        """
        # Add embedding for this memory
        self.add_embedding(key, value)

        # Find similar memories (excluding this key)
        similar = self.find_similar(value, exclude_keys=[key])

        if not similar:
            return None

        # Get thresholds
        duplicate_threshold = self.thresholds.get("duplicate", 0.95)
        similar_threshold = self.thresholds.get("similar", 0.80)

        # Filter to only show relevant similarities
        top_match = similar[0]

        if top_match["similarity"] < similar_threshold:
            return None  # Not similar enough to mention

        # Build feedback
        feedback_lines = ["\nSimilarity Check:"]

        for match in similar[:3]:  # Show top 3
            similarity_pct = int(match["similarity"] * 100)
            preview = match["preview"]
            if len(preview) > 80:
                preview = preview[:77] + "..."

            feedback_lines.append(
                f"• {match['key']} ({similarity_pct}% similar)"
            )
            feedback_lines.append(f"  Preview: {preview}")

            # Add suggestion for high similarity
            if match["similarity"] >= duplicate_threshold:
                feedback_lines.append(
                    f"  → Very similar! Consider consolidating with memory_write('{match['key']}', combined_content)"
                )
            elif match["similarity"] >= similar_threshold:
                feedback_lines.append(
                    f"  → Consider reading '{match['key']}' and merging insights"
                )

        return "\n".join(feedback_lines)
