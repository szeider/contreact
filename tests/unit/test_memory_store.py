"""Unit tests for contreact.memory.store module."""

import pytest
import sqlite3
from datetime import datetime

from contreact.memory.store import MemoryStore


class TestMemoryStoreInit:
    """Tests for MemoryStore initialization."""

    def test_creates_database(self, temp_db):
        """Should create SQLite database file."""
        MemoryStore(temp_db)
        assert temp_db.exists()

    def test_creates_memories_table(self, temp_db):
        """Should create memories table."""
        MemoryStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_fts_table(self, temp_db):
        """Should create FTS5 virtual table."""
        MemoryStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestMemoryStoreWrite:
    """Tests for MemoryStore.write method."""

    def test_write_new_memory(self, memory_store):
        """Should write new memory and return False."""
        is_update = memory_store.write("test_key", "test value")
        assert is_update is False

    def test_write_update_memory(self, memory_store):
        """Should update existing memory and return True."""
        memory_store.write("test_key", "original value")
        is_update = memory_store.write("test_key", "updated value")
        assert is_update is True

    def test_write_stores_value(self, memory_store):
        """Should store value correctly."""
        memory_store.write("my_key", "my value")
        value = memory_store.read("my_key")
        assert value == "my value"

    def test_write_with_tool_call_number(self, memory_store):
        """Should accept tool_call_number parameter."""
        memory_store.write("key", "value", tool_call_number=42)
        # Should not raise

    def test_write_long_value(self, memory_store):
        """Should accept values up to 50K chars."""
        long_value = "x" * 50000
        memory_store.write("long_key", long_value)
        retrieved = memory_store.read("long_key")
        assert len(retrieved) == 50000

    def test_write_rejects_too_long_value(self, memory_store):
        """Should reject values over 50K chars."""
        too_long = "x" * 50001
        with pytest.raises(sqlite3.IntegrityError):
            memory_store.write("key", too_long)


class TestMemoryStoreRead:
    """Tests for MemoryStore.read method."""

    def test_read_existing_key(self, memory_store):
        """Should return value for existing key."""
        memory_store.write("key1", "value1")
        assert memory_store.read("key1") == "value1"

    def test_read_nonexistent_key(self, memory_store):
        """Should return None for nonexistent key."""
        assert memory_store.read("nonexistent") is None

    def test_read_preserves_unicode(self, memory_store):
        """Should preserve Unicode characters."""
        memory_store.write("unicode", "Hello \U0001F600 World!")
        assert memory_store.read("unicode") == "Hello \U0001F600 World!"

    def test_read_preserves_newlines(self, memory_store):
        """Should preserve newlines and formatting."""
        multiline = "Line 1\nLine 2\n\tIndented"
        memory_store.write("multiline", multiline)
        assert memory_store.read("multiline") == multiline


class TestMemoryStoreDelete:
    """Tests for MemoryStore.delete method."""

    def test_delete_existing_key(self, memory_store):
        """Should delete existing key and return True."""
        memory_store.write("to_delete", "value")
        deleted = memory_store.delete("to_delete")
        assert deleted is True
        assert memory_store.read("to_delete") is None

    def test_delete_nonexistent_key(self, memory_store):
        """Should return False for nonexistent key."""
        deleted = memory_store.delete("nonexistent")
        assert deleted is False


class TestMemoryStoreListEntries:
    """Tests for MemoryStore.list_entries method."""

    def test_list_empty(self, memory_store):
        """Should return empty list when no memories."""
        entries = memory_store.list_entries()
        assert entries == []

    def test_list_returns_keys_with_timestamps(self, memory_store):
        """Should return list of (key, timestamp) tuples."""
        memory_store.write("key1", "value1")
        memory_store.write("key2", "value2")

        entries = memory_store.list_entries()
        assert len(entries) == 2

        # Check format
        for key, timestamp in entries:
            assert isinstance(key, str)
            assert isinstance(timestamp, datetime)

    def test_list_updates_order_on_update(self, memory_store):
        """Updating memory should move it to top of list."""
        memory_store.write("first", "value")
        memory_store.write("second", "value")
        memory_store.write("first", "updated")  # Update first

        entries = memory_store.list_entries()
        key_names = [k for k, _ in entries]

        # First should now be at top
        assert key_names[0] == "first"


class TestMemoryStoreSearch:
    """Tests for MemoryStore.search method."""

    def test_search_empty_store(self, memory_store):
        """Should return empty list when no memories."""
        results = memory_store.search("anything")
        assert results == []

    def test_search_finds_matching_key(self, memory_store):
        """Should find memories with matching key."""
        memory_store.write("python_guide", "Python is a programming language")
        memory_store.write("java_guide", "Java is also a programming language")

        results = memory_store.search("python")
        assert len(results) >= 1
        keys = [k for k, _ in results]
        assert "python_guide" in keys

    def test_search_finds_matching_value(self, memory_store):
        """Should find memories with matching value content."""
        memory_store.write("animals", "The quick brown fox jumps")
        memory_store.write("colors", "The sky is blue")

        results = memory_store.search("fox")
        assert len(results) >= 1
        keys = [k for k, _ in results]
        assert "animals" in keys

    def test_search_returns_scores(self, memory_store):
        """Should return BM25 scores with results."""
        memory_store.write("key1", "test content")

        results = memory_store.search("test")
        assert len(results) >= 1

        key, score = results[0]
        assert isinstance(score, float)

    def test_search_respects_limit(self, memory_store):
        """Should respect limit parameter."""
        for i in range(25):
            memory_store.write(f"test_key_{i}", f"test value {i}")

        results = memory_store.search("test", limit=5)
        assert len(results) <= 5


class TestMemoryStoreCount:
    """Tests for MemoryStore.count method."""

    def test_count_empty(self, memory_store):
        """Should return 0 when no memories."""
        assert memory_store.count() == 0

    def test_count_after_writes(self, memory_store):
        """Should return correct count."""
        memory_store.write("key1", "value1")
        memory_store.write("key2", "value2")
        memory_store.write("key3", "value3")

        assert memory_store.count() == 3

    def test_count_after_delete(self, memory_store):
        """Should update count after deletion."""
        memory_store.write("key1", "value1")
        memory_store.write("key2", "value2")
        memory_store.delete("key1")

        assert memory_store.count() == 1
