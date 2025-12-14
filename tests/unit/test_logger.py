"""Unit tests for contreact.logger module."""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from contreact.logger import log_message, load_history


class TestLogMessage:
    """Tests for log_message function."""

    def test_creates_history_file(self, temp_dir):
        """Should create history.jsonl file if not exists."""
        run_name = str(temp_dir / "test_run")
        Path(run_name).mkdir()

        msg = HumanMessage(content="Test message")
        log_message(run_name, msg)

        log_path = Path(run_name) / "history.jsonl"
        assert log_path.exists()

    def test_appends_message_as_jsonl(self, temp_dir):
        """Should append messages as JSONL lines."""
        run_name = str(temp_dir / "test_run")
        Path(run_name).mkdir()

        msg1 = HumanMessage(content="First message")
        msg2 = AIMessage(content="Second message")

        log_message(run_name, msg1)
        log_message(run_name, msg2)

        log_path = Path(run_name) / "history.jsonl"
        lines = log_path.read_text().strip().split('\n')

        assert len(lines) == 2

        # Verify JSON format
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])
        assert data1["type"] == "human"
        assert data2["type"] == "ai"

    def test_logs_different_message_types(self, temp_dir):
        """Should handle different message types."""
        run_name = str(temp_dir / "test_run")
        Path(run_name).mkdir()

        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User input"),
            AIMessage(content="AI response"),
            ToolMessage(content="Tool result", tool_call_id="tc123"),
        ]

        for msg in messages:
            log_message(run_name, msg)

        log_path = Path(run_name) / "history.jsonl"
        lines = log_path.read_text().strip().split('\n')
        assert len(lines) == 4

    def test_works_with_existing_directory(self, temp_dir):
        """Should work when directory already exists."""
        run_name = str(temp_dir / "existing_run")
        Path(run_name).mkdir()

        msg = HumanMessage(content="Test")
        log_message(run_name, msg)

        log_path = Path(run_name) / "history.jsonl"
        assert log_path.exists()


class TestLoadHistory:
    """Tests for load_history function."""

    def test_returns_empty_for_nonexistent(self, temp_dir):
        """Should return empty list if no history file."""
        run_name = str(temp_dir / "nonexistent_run")
        history = load_history(run_name)
        assert history == []

    def test_loads_messages_correctly(self, temp_dir):
        """Should load messages back as Message objects."""
        run_name = str(temp_dir / "test_run")
        Path(run_name).mkdir()

        # Write messages using log_message
        log_message(run_name, HumanMessage(content="Hello"))
        log_message(run_name, AIMessage(content="Hi there"))

        # Load them back
        loaded = load_history(run_name)

        assert len(loaded) == 2
        assert loaded[0].content == "Hello"
        assert loaded[1].content == "Hi there"

    def test_handles_empty_lines(self, temp_dir):
        """Should ignore empty lines in history file."""
        run_name = str(temp_dir / "test_run")
        Path(run_name).mkdir()

        # First log a message properly
        log_message(run_name, HumanMessage(content="Test"))

        # Append empty lines manually
        log_path = Path(run_name) / "history.jsonl"
        with open(log_path, 'a') as f:
            f.write("\n\n")

        loaded = load_history(run_name)
        assert len(loaded) == 1
        assert loaded[0].content == "Test"

    def test_round_trip_preserves_content(self, temp_dir):
        """Log and load should preserve message content."""
        run_name = str(temp_dir / "test_run")
        Path(run_name).mkdir()

        original_content = "You are a helpful assistant."
        log_message(run_name, SystemMessage(content=original_content))

        loaded = load_history(run_name)
        assert len(loaded) == 1
        assert loaded[0].content == original_content
