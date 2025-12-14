"""End-to-end tests for segmented execution mode."""

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from contreact.graph import create_segmented_graph, create_graph


class TestSegmentedModeExecution:
    """Tests for segmented mode cycle execution."""

    def test_segmented_graph_ends_on_no_tool_calls(self):
        """Segmented graph should end when agent produces no tool calls."""
        mock_tool = MagicMock()
        mock_tool.name = "think"
        mock_tool.invoke = MagicMock(return_value="Thought recorded")

        graph = create_segmented_graph([mock_tool])

        # Create a mock LLM that returns a response without tool calls
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=AIMessage(content="I have finished thinking."))

        config = {"configurable": {"thread_id": "test", "llm": mock_llm}}
        inputs = {"messages": [SystemMessage(content="You are a test agent.")]}

        # Run graph - should end after one agent call (no tool calls)
        events = list(graph.stream(inputs, config, stream_mode="updates"))

        # Should have exactly one event (agent node)
        assert len(events) == 1
        assert "agent" in events[0]

    def test_segmented_graph_continues_with_tool_calls(self):
        """Segmented graph should continue when agent makes tool calls."""
        mock_tool = MagicMock()
        mock_tool.name = "think"
        mock_tool.invoke = MagicMock(return_value="Thought recorded")

        graph = create_segmented_graph([mock_tool])

        # Track call count to vary responses
        call_count = [0]

        def mock_invoke(messages, config=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: make a tool call
                return AIMessage(
                    content="Let me think...",
                    tool_calls=[{"id": "tc1", "name": "think", "args": {"thought": "test"}}]
                )
            else:
                # Second call: no tool calls, end cycle
                return AIMessage(content="Done thinking.")

        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke

        config = {"configurable": {"thread_id": "test", "llm": mock_llm}}
        inputs = {"messages": [SystemMessage(content="You are a test agent.")]}

        events = list(graph.stream(inputs, config, stream_mode="updates"))

        # Should have: agent (tool call) -> tools -> agent (no tool call, end)
        assert len(events) == 3
        assert "agent" in events[0]
        assert "tools" in events[1]
        assert "agent" in events[2]

    def test_segmented_mode_multiple_cycles(self):
        """Test running multiple cycles with wake messages."""
        mock_tool = MagicMock()
        mock_tool.name = "think"
        mock_tool.invoke = MagicMock(return_value="Thought recorded")

        graph = create_segmented_graph([mock_tool])

        # Track cycles
        cycle_count = [0]
        wake_messages_seen = []

        def mock_invoke(messages, config=None):
            # Check for wake messages in input
            for msg in messages:
                if isinstance(msg, HumanMessage) and "continue (cycle:" in msg.content:
                    wake_messages_seen.append(msg.content)

            cycle_count[0] += 1
            # Always end immediately for this test (no tool calls)
            return AIMessage(content=f"Cycle {cycle_count[0]} complete.")

        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke

        config = {"configurable": {"thread_id": "test", "llm": mock_llm}}

        # Run 3 cycles manually (simulating main.py loop)
        num_cycles = 3
        inputs = {"messages": [SystemMessage(content="You are a test agent.")]}

        for cycle_num in range(1, num_cycles + 1):
            events = list(graph.stream(inputs, config, stream_mode="updates"))

            # Each cycle should have at least one event
            assert len(events) >= 1

            # Prepare wake message for next cycle
            if cycle_num < num_cycles:
                wake_message = f"continue (cycle: {cycle_num + 1})"
                inputs = {"messages": [HumanMessage(content=wake_message)]}

        # Verify we ran all cycles
        assert cycle_count[0] == num_cycles

        # Verify wake messages were seen (cycles 2 and 3)
        assert len(wake_messages_seen) == 2
        assert "continue (cycle: 2)" in wake_messages_seen[0]
        assert "continue (cycle: 3)" in wake_messages_seen[1]

    def test_unsegmented_graph_never_ends(self):
        """Unsegmented graph should continue even without tool calls."""
        mock_tool = MagicMock()
        mock_tool.name = "think"
        mock_tool.invoke = MagicMock(return_value="Thought recorded")

        graph = create_graph([mock_tool])

        call_count = [0]

        def mock_invoke(messages, config=None):
            call_count[0] += 1
            # Return without tool calls - unsegmented should still continue
            return AIMessage(content="No tools needed.")

        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke

        # Use recursion_limit to stop the infinite loop
        config = {"configurable": {"thread_id": "test", "llm": mock_llm}, "recursion_limit": 10}
        inputs = {"messages": [SystemMessage(content="You are a test agent.")]}

        # Unsegmented graph will loop until recursion limit
        from langgraph.errors import GraphRecursionError
        with pytest.raises(GraphRecursionError):
            list(graph.stream(inputs, config, stream_mode="updates"))

        # Key assertion: unsegmented mode called the LLM multiple times
        # (more than once, proving it didn't end on first no-tool-call response)
        assert call_count[0] > 1, "Unsegmented should loop multiple times"

    def test_segmented_mode_with_tool_execution(self):
        """Test segmented mode executes tools correctly within a cycle."""
        tool_calls_executed = []

        mock_tool = MagicMock()
        mock_tool.name = "think"

        def track_tool_call(args):
            tool_calls_executed.append(args)
            return "Thought recorded"

        mock_tool.invoke = track_tool_call

        graph = create_segmented_graph([mock_tool])

        call_count = [0]

        def mock_invoke(messages, config=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "tc1", "name": "think", "args": {"thought": "first thought"}}]
                )
            elif call_count[0] == 2:
                return AIMessage(
                    content="",
                    tool_calls=[{"id": "tc2", "name": "think", "args": {"thought": "second thought"}}]
                )
            else:
                return AIMessage(content="Finished thinking.")

        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke

        config = {"configurable": {"thread_id": "test", "llm": mock_llm}}
        inputs = {"messages": [SystemMessage(content="Test")]}

        events = list(graph.stream(inputs, config, stream_mode="updates"))

        # Should have: agent -> tools -> agent -> tools -> agent (end)
        assert len(events) == 5

        # Two tool calls executed
        assert len(tool_calls_executed) == 2
        assert tool_calls_executed[0] == {"thought": "first thought"}
        assert tool_calls_executed[1] == {"thought": "second thought"}


class TestSegmentedVsUnsegmentedBehavior:
    """Compare behavior between segmented and unsegmented modes."""

    def test_same_tool_calls_different_endings(self):
        """Both modes should execute same tools, but end differently."""
        mock_tool = MagicMock()
        mock_tool.name = "think"
        mock_tool.invoke = MagicMock(return_value="OK")

        # Test with response that has no tool calls
        def mock_invoke_no_tools(messages, config=None):
            return AIMessage(content="Done.")

        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke_no_tools

        config = {"configurable": {"thread_id": "test", "llm": mock_llm}, "recursion_limit": 5}
        inputs = {"messages": [SystemMessage(content="Test")]}

        # Segmented: should end after one call
        segmented_graph = create_segmented_graph([mock_tool])
        segmented_events = list(segmented_graph.stream(inputs, config, stream_mode="updates"))
        assert len(segmented_events) == 1  # Just agent, then END

        # Unsegmented: will keep looping (until recursion limit)
        unsegmented_graph = create_graph([mock_tool])
        unsegmented_events = []
        try:
            for event in unsegmented_graph.stream(inputs, config, stream_mode="updates"):
                unsegmented_events.append(event)
        except Exception:
            pass  # Will hit recursion limit or similar

        # Unsegmented should have more events (kept trying to loop)
        assert len(unsegmented_events) > len(segmented_events)
