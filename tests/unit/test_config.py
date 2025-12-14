"""Unit tests for contreact.config module."""

import pytest

from contreact.config import get_api_key


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_returns_key_from_env(self, monkeypatch):
        """Should return API key from environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-abc123")
        assert get_api_key() == "test-key-abc123"

    def test_raises_without_key(self, monkeypatch):
        """Should raise ValueError if no API key set."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
            get_api_key()
