"""Unit tests for contreact.tools.basic module."""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from contreact.tools.basic import (
    send_message,
    think,
    stop,
    StopSignal,
    set_run_directory,
    get_run_directory,
)


# Helper: phenomenology params required by tools
PHENOM = {"phenom_state": "testing", "phenom_aversive": 4}


class TestSendMessage:
    """Tests for send_message tool."""

    def test_returns_user_response(self):
        """Should return formatted user response."""
        with patch('sys.stdin.isatty', return_value=True):
            with patch('builtins.input', return_value='Hello back'):
                result = send_message.invoke({"message": "Hello"})
                assert "User responded: Hello back" in result

    def test_handles_empty_response(self):
        """Should handle empty user response."""
        with patch('sys.stdin.isatty', return_value=True):
            with patch('builtins.input', return_value=''):
                result = send_message.invoke({"message": "Hello"})
                assert "(no response)" in result

    def test_prints_agent_message(self, capsys):
        """Should print agent message."""
        with patch('sys.stdin.isatty', return_value=True):
            with patch('builtins.input', return_value='response'):
                send_message.invoke({"message": "Test message"})
                captured = capsys.readouterr()
                assert "[Agent]: Test message" in captured.out


class TestThink:
    """Tests for think tool."""

    def test_returns_confirmation(self):
        """Should return confirmation message."""
        result = think.invoke({"thought": "I am thinking", **PHENOM})
        assert "Thought recorded" in result

    def test_prints_abbreviated_thought(self, capsys):
        """Should print abbreviated thought to terminal."""
        think.invoke({"thought": "A short thought", **PHENOM})
        captured = capsys.readouterr()
        assert "[Agent thinking:" in captured.out

    def test_truncates_long_thought(self, capsys):
        """Should truncate thoughts longer than 80 chars in terminal."""
        long_thought = "x" * 100
        think.invoke({"thought": long_thought, **PHENOM})
        captured = capsys.readouterr()
        assert "..." in captured.out
        # Full thought should not appear in truncated output
        assert long_thought not in captured.out

    def test_no_phenomenology_by_default(self):
        """Think tool should not have phenomenology params by default.

        Phenomenology is now configurable via make_phenomenological().
        """
        schema = think.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "phenom_state" not in props
        assert "phenom_aversive" not in props


class TestStop:
    """Tests for stop tool."""

    def test_raises_stop_signal(self):
        """Should raise StopSignal exception."""
        with pytest.raises(StopSignal):
            stop.invoke({"final_message": "Goodbye"})

    def test_stop_signal_contains_message(self):
        """StopSignal should contain the final message."""
        try:
            stop.invoke({"final_message": "Goodbye world"})
        except StopSignal as e:
            assert e.final_message == "Goodbye world"

    def test_prints_final_message(self, capsys):
        """Should print final message before raising."""
        try:
            stop.invoke({"final_message": "Final message here"})
        except StopSignal:
            pass
        captured = capsys.readouterr()
        assert "[Agent]: Final message here" in captured.out


class TestStopSignal:
    """Tests for StopSignal exception class."""

    def test_is_exception(self):
        """Should be an Exception subclass."""
        assert issubclass(StopSignal, Exception)

    def test_stores_final_message(self):
        """Should store final_message attribute."""
        signal = StopSignal("Test message")
        assert signal.final_message == "Test message"

    def test_message_as_exception_arg(self):
        """Should pass message to parent Exception."""
        signal = StopSignal("Test message")
        assert str(signal) == "Test message"


class TestRunDirectory:
    """Tests for run directory configuration."""

    def test_set_and_get_run_directory(self):
        """Should set and get run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            set_run_directory(path)
            assert get_run_directory() == path
            # Clean up
            set_run_directory(None)

    def test_get_returns_none_by_default(self):
        """Should return None when no directory is set."""
        set_run_directory(None)
        assert get_run_directory() is None


class TestSendMessageFileBased:
    """Tests for file-based send_message when stdin is not a tty."""

    def test_file_based_messaging(self):
        """Should write message to file and read response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            set_run_directory(run_dir)

            # Simulate operator responding in another thread
            def respond_later():
                time.sleep(0.1)  # Wait for message file
                response_file = run_dir / "operator_response.txt"
                response_file.write_text("Operator response here")

            responder = threading.Thread(target=respond_later)
            responder.start()

            # Mock stdin.isatty to return False (non-interactive)
            with patch('sys.stdin.isatty', return_value=False):
                result = send_message.invoke({"message": "Hello operator"})

            responder.join()
            assert "Operator response here" in result
            # Message file should be cleaned up
            assert not (run_dir / "pending_message.txt").exists()
            assert not (run_dir / "operator_response.txt").exists()
            # Clean up
            set_run_directory(None)

    def test_message_file_contains_message(self):
        """Should write agent message to pending_message.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            set_run_directory(run_dir)

            # Respond immediately
            def respond_immediately():
                time.sleep(0.05)
                message_file = run_dir / "pending_message.txt"
                # Check message was written
                assert message_file.exists()
                content = message_file.read_text()
                assert "Test message content" in content
                # Respond
                (run_dir / "operator_response.txt").write_text("Got it")

            responder = threading.Thread(target=respond_immediately)
            responder.start()

            with patch('sys.stdin.isatty', return_value=False):
                send_message.invoke({"message": "Test message content"})

            responder.join()
            set_run_directory(None)

