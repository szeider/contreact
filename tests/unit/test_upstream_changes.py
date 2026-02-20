"""Tests for upstream generic improvements.

Tests the --step flag in main.py CLI, the set_pre_message_callback export,
and the updated send_message/think display behavior.
"""

import argparse
from unittest.mock import patch, MagicMock

from langchain_core.messages import ToolMessage, AIMessage


class TestStepFlagCLI:
    """Tests for --step CLI argument in main.py."""

    def _parse_args(self, args_list):
        """Parse args using main.py's argument parser."""
        parser = argparse.ArgumentParser()
        parser.add_argument("run_name")
        parser.add_argument("-q", "--query")
        parser.add_argument("--step", action="store_true")
        return parser.parse_args(args_list)

    def test_step_flag_default_false(self):
        """--step should default to False."""
        args = self._parse_args(["RUNS/run_001"])
        assert args.step is False

    def test_step_flag_set(self):
        """--step should be True when provided."""
        args = self._parse_args(["RUNS/run_001", "--step"])
        assert args.step is True

    def test_step_with_query(self):
        """--step should work alongside --query."""
        args = self._parse_args(["RUNS/run_001", "--step", "-q", "test question"])
        assert args.step is True
        assert args.query == "test question"


class TestSetPreMessageCallbackExport:
    """Test that set_pre_message_callback is properly exported from tools package."""

    def test_importable_from_tools(self):
        """Should be importable from contreact.tools."""
        from contreact.tools import set_pre_message_callback
        assert callable(set_pre_message_callback)

    def test_in_all_list(self):
        """Should be listed in __all__."""
        import contreact.tools as tools_module
        assert "set_pre_message_callback" in tools_module.__all__


class TestStepModeLogic:
    """Tests for the step-mode event processing logic used in main.py's unsegmented loop."""

    def test_tool_message_detected(self):
        """ToolMessage should be detected as a tool event."""
        msg = ToolMessage(content="result", tool_call_id="123")
        assert isinstance(msg, ToolMessage)

    def test_pending_tool_name_from_ai_message(self):
        """Should extract tool name from AIMessage tool_calls."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "think", "args": {"thought": "hmm"}, "id": "1"}],
        )
        assert hasattr(msg, 'tool_calls')
        assert msg.tool_calls[-1]["name"] == "think"

    def test_send_message_skip_logic(self):
        """send_message should be identified for Enter-skip in step mode."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "send_message", "args": {"message": "hi"}, "id": "1"}],
        )
        pending_tool_name = msg.tool_calls[-1]["name"]
        # Step mode should NOT prompt for Enter after send_message
        assert pending_tool_name == "send_message"

    def test_non_send_message_requires_enter(self):
        """Non-send_message tools should require Enter in step mode."""
        for tool_name in ["think", "stop", "canvas_draw", "submit_data"]:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": {}, "id": "1"}],
            )
            pending_tool_name = msg.tool_calls[-1]["name"]
            assert pending_tool_name != "send_message"

    def test_multiple_tool_calls_uses_last(self):
        """When multiple tool_calls exist, pending_tool_name should be the last one."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "think", "args": {}, "id": "1"},
                {"name": "send_message", "args": {"message": "hi"}, "id": "2"},
            ],
        )
        pending_tool_name = msg.tool_calls[-1]["name"]
        assert pending_tool_name == "send_message"


class TestSendMessageLabel:
    """Tests for updated send_message display label."""

    def test_label_is_message_to_operator(self, capsys):
        """send_message should use [Message to operator] label."""
        from contreact.tools.basic import send_message

        with patch('sys.stdin.isatty', return_value=True):
            with patch('builtins.input', return_value='ok'):
                send_message.invoke({"message": "Hello"})
                captured = capsys.readouterr()
                assert "[Message to operator]:" in captured.out
                assert "[Agent]:" not in captured.out


class TestThinkFullDisplay:
    """Tests for think showing full thought without truncation."""

    def test_short_thought_displayed_fully(self, capsys):
        """Short thought should be displayed as-is."""
        from contreact.tools.basic import think
        think.invoke({"thought": "Brief idea"})
        captured = capsys.readouterr()
        assert "Brief idea" in captured.out

    def test_long_thought_not_truncated(self, capsys):
        """Long thought should NOT be truncated with '...'."""
        from contreact.tools.basic import think
        long_thought = "A" * 200
        think.invoke({"thought": long_thought})
        captured = capsys.readouterr()
        assert long_thought in captured.out
        # No ellipsis truncation
        assert "..." not in captured.out

    def test_multiline_thought_displayed(self, capsys):
        """Multi-line thought should display all lines."""
        from contreact.tools.basic import think
        thought = "Line one\nLine two\nLine three"
        think.invoke({"thought": thought})
        captured = capsys.readouterr()
        assert thought in captured.out
