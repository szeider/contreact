"""SQLite-based memory storage for contreact agents."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class MemoryStore:
    """SQLite-based key-value storage with FTS5 full-text search."""

    def __init__(self, db_path: Path, max_value_length: int = 50000):
        """Initialize memory store.

        Args:
            db_path: Path to SQLite database (shared with checkpoint.sqlite)
            max_value_length: Maximum characters per memory value (default 50K)
        """
        self.db_path = Path(db_path)
        self.max_value_length = max_value_length
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create SQLite connection with 30s timeout."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,  # 30 second timeout for concurrent access
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row  # Access columns by name
        conn.execute("PRAGMA foreign_keys=ON")  # Enable foreign key enforcement
        return conn

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        with self._get_connection() as conn:
            # Main memories table
            # Note: CHECK constraint uses hardcoded 50000 (SQLite doesn't allow parameters)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL CHECK(length(value) <= 50000),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    tool_call_number INTEGER
                )
            """)

            # Index for chronological ordering
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_updated
                ON memories(updated_at DESC)
            """)

            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    key,
                    value,
                    content='memories',
                    content_rowid='rowid'
                )
            """)

            # Trigger: sync FTS on insert
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_insert
                AFTER INSERT ON memories
                BEGIN
                    INSERT INTO memories_fts(rowid, key, value)
                    VALUES (NEW.rowid, NEW.key, NEW.value);
                END
            """)

            # Trigger: sync FTS on delete
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_delete
                AFTER DELETE ON memories
                BEGIN
                    DELETE FROM memories_fts WHERE rowid = OLD.rowid;
                END
            """)

            # Trigger: sync FTS on update
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_update
                AFTER UPDATE ON memories
                BEGIN
                    UPDATE memories_fts
                    SET key = NEW.key, value = NEW.value
                    WHERE rowid = NEW.rowid;
                END
            """)

            # Trigger: update timestamp on update
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_memory_timestamp
                AFTER UPDATE ON memories
                FOR EACH ROW
                BEGIN
                    UPDATE memories
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE key = NEW.key;
                END
            """)

            conn.commit()

    def list_entries(self) -> list[tuple[str, datetime]]:
        """List all memory entries with timestamps, newest first.

        Returns:
            List of (key, updated_at) tuples, ordered by updated_at DESC
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT key, updated_at
                FROM memories
                ORDER BY updated_at DESC
            """).fetchall()

            return [
                (row["key"], datetime.fromisoformat(row["updated_at"]))
                for row in rows
            ]

    def read(self, key: str) -> Optional[str]:
        """Read value for specific key.

        Args:
            key: Memory key to read

        Returns:
            Memory value or None if key not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM memories WHERE key = ?",
                (key,)
            ).fetchone()

            return row["value"] if row else None

    def write(self, key: str, value: str, tool_call_number: Optional[int] = None) -> bool:
        """Create or update memory.

        Args:
            key: Memory key
            value: Memory value (max 50K chars by default)
            tool_call_number: Optional tool call number for tracking

        Returns:
            True if this was an update (key existed), False if new entry

        Raises:
            sqlite3.IntegrityError: If value exceeds max_value_length
        """
        # Enforce max_value_length in Python (matches DB CHECK constraint)
        if len(value) > self.max_value_length:
            raise sqlite3.IntegrityError(
                f"Value exceeds max_value_length ({len(value)} > {self.max_value_length})"
            )

        # Check if key exists (for return value)
        is_update = self.read(key) is not None

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO memories (key, value, tool_call_number)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    tool_call_number = excluded.tool_call_number,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, value, tool_call_number))
            conn.commit()

        return is_update

    def delete(self, key: str) -> bool:
        """Delete memory key.

        Args:
            key: Memory key to delete

        Returns:
            True if key existed and was deleted, False if key not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE key = ?",
                (key,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Full-text search using SQLite FTS5 with BM25 ranking.

        Args:
            query: Search query (FTS5 syntax supported)
            limit: Maximum results to return (default 20)

        Returns:
            List of (key, bm25_score) tuples, ordered by relevance (best first)
            Note: BM25 scores are negative (higher = better match)
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT m.key, bm25(memories_fts) as score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.rowid
                WHERE memories_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, limit)).fetchall()

            return [(row["key"], row["score"]) for row in rows]

    def count(self) -> int:
        """Get total number of memories.

        Returns:
            Total count of memory entries
        """
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM memories").fetchone()
            return row["count"]
