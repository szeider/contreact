"""Pytest fixtures and configuration for ContReAct tests."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary SQLite database."""
    db_path = temp_dir / "test.sqlite"
    yield db_path
    # Cleanup handled by temp_dir


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "run_name": "test_run",
        "description": "Test configuration",
        "model": {
            "name": "anthropic/claude-sonnet-4.5",
            "temperature": 0.0,
            "streaming": True,
            "max_tokens": 16384
        },
        "tools": ["think", "stop"],
        "thread_id": "test_thread",
        "max_tool_calls": 5,
        "prompts": {
            "react_prompt": "react_prompt.md",
            "instance_prompt": "instance_prompt.md"
        }
    }


@pytest.fixture
def sample_run_dir(temp_dir, sample_config):
    """Create a sample run directory with config and prompts."""
    run_dir = temp_dir / "test_run"
    run_dir.mkdir()

    # Write config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f)

    # Write react prompt
    react_prompt = run_dir / "react_prompt.md"
    react_prompt.write_text(
        "# ReAct Agent\n\n"
        "You are a continuous ReAct agent. Use the tools available to you to complete tasks.\n\n"
        "Think step by step. Use the think tool to reason about the problem before taking action.\n"
    )

    # Write instance prompt
    instance_prompt = run_dir / "instance_prompt.md"
    instance_prompt.write_text("# Test Instance\n\nYou are a test agent.")

    return run_dir


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Mock the OPENROUTER_API_KEY environment variable."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key-12345")


@pytest.fixture
def memory_store(temp_db):
    """Create a MemoryStore instance for testing."""
    from contreact.memory import MemoryStore
    return MemoryStore(temp_db)


@pytest.fixture
def canvas_store(temp_db):
    """Create a CanvasStore instance for testing."""
    from contreact.tools.canvas import CanvasStore
    return CanvasStore(temp_db)
