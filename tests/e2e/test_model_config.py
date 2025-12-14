"""End-to-end tests for model configuration loading."""

import json
import pytest
from unittest.mock import patch, MagicMock

from contreact.main import load_config, create_llm


class TestLoadConfig:
    """Tests for loading config.json with new model format."""

    def test_loads_model_config_dict(self, sample_run_dir):
        """Config with model dict loads correctly."""
        config = load_config(str(sample_run_dir))
        assert isinstance(config["model"], dict)
        assert config["model"]["name"] == "anthropic/claude-sonnet-4.5"

    def test_loads_model_parameters(self, sample_run_dir):
        """Model parameters are preserved."""
        config = load_config(str(sample_run_dir))
        assert config["model"]["temperature"] == 0.0
        assert config["model"]["streaming"] is True
        assert config["model"]["max_tokens"] == 16384

    def test_loads_minimal_model_config(self, temp_dir):
        """Config with only model.name works."""
        run_dir = temp_dir / "minimal_run"
        run_dir.mkdir()
        config = {
            "model": {"name": "openai/gpt-5.2"},
            "tools": ["think"]
        }
        (run_dir / "config.json").write_text(json.dumps(config))

        loaded = load_config(str(run_dir))
        assert loaded["model"]["name"] == "openai/gpt-5.2"


class TestCreateLlm:
    """Tests for create_llm with model config dict."""

    def test_creates_llm_with_full_config(self, mock_env_api_key):
        """LLM created with all parameters."""
        model_config = {
            "name": "anthropic/claude-sonnet-4.5",
            "temperature": 0.5,
            "max_tokens": 8000,
            "streaming": True
        }

        with patch("contreact.main.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm(model_config)

            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "anthropic/claude-sonnet-4.5"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 8000
            assert call_kwargs["streaming"] is True

    def test_creates_llm_with_minimal_config(self, mock_env_api_key):
        """LLM created with just model name."""
        model_config = {"name": "x-ai/grok-4"}

        with patch("contreact.main.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm(model_config)

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "x-ai/grok-4"
            assert call_kwargs["streaming"] is True  # default

    def test_passes_model_kwargs(self, mock_env_api_key):
        """model_kwargs passed to LLM."""
        model_config = {
            "name": "openai/gpt-5.1",
            "model_kwargs": {
                "stream_options": {"include_usage": True},
                "parallel_tool_calls": False
            }
        }

        with patch("contreact.main.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm(model_config)

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model_kwargs"]["parallel_tool_calls"] is False

    def test_passes_extra_body(self, mock_env_api_key):
        """extra_body passed to LLM."""
        model_config = {
            "name": "openai/gpt-5.2",
            "extra_body": {"reasoning": {"effort": "medium"}}
        }

        with patch("contreact.main.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm(model_config)

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["extra_body"]["reasoning"]["effort"] == "medium"

    def test_passes_top_p(self, mock_env_api_key):
        """top_p parameter passed to LLM."""
        model_config = {
            "name": "openai/gpt-5",
            "top_p": 0.7
        }

        with patch("contreact.main.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm(model_config)

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["top_p"] == 0.7

    def test_passes_request_timeout(self, mock_env_api_key):
        """request_timeout parameter passed to LLM."""
        model_config = {
            "name": "google/gemini-2.5-pro",
            "request_timeout": 120
        }

        with patch("contreact.main.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm(model_config)

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["request_timeout"] == 120

    def test_uses_openrouter_base_url(self, mock_env_api_key):
        """LLM uses OpenRouter base URL."""
        model_config = {"name": "anthropic/claude-opus-4.5"}

        with patch("contreact.main.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_llm(model_config)

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"


class TestModelConfigValidation:
    """Tests for model config validation."""

    def test_missing_model_name_raises(self, temp_dir, mock_env_api_key):
        """Missing model.name raises ValueError."""
        from contreact.main import main

        run_dir = temp_dir / "bad_run"
        run_dir.mkdir()
        config = {
            "model": {"temperature": 0.5},  # missing name
            "tools": ["think"]
        }
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "react_prompt.md").write_text("# ReAct Agent\n")
        (run_dir / "instance_prompt.md").write_text("Test prompt")

        # Patch sys.argv and catch the ValueError
        with patch("sys.argv", ["contreact", str(run_dir)]):
            with pytest.raises(ValueError, match="model.name"):
                main()

    def test_empty_model_dict_raises(self, temp_dir, mock_env_api_key):
        """Empty model dict raises ValueError."""
        from contreact.main import main

        run_dir = temp_dir / "empty_model_run"
        run_dir.mkdir()
        config = {
            "model": {},
            "tools": ["think"]
        }
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "react_prompt.md").write_text("# ReAct Agent\n")
        (run_dir / "instance_prompt.md").write_text("Test prompt")

        with patch("sys.argv", ["contreact", str(run_dir)]):
            with pytest.raises(ValueError, match="model.name"):
                main()
