"""Unit tests for contreact.graph module."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from contreact.graph import AgentState, create_graph, create_segmented_graph, strip_images_from_history


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_state_accepts_messages(self):
        """Should accept messages list."""
        state: AgentState = {
            "messages": [
                SystemMessage(content="You are a test"),
                HumanMessage(content="Hello"),
            ]
        }
        assert len(state["messages"]) == 2


class TestCreateGraph:
    """Tests for create_graph function."""

    def test_creates_graph_with_tools(self):
        """Should create graph with provided tools."""
        # Create mock tools
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="result")

        graph = create_graph([mock_tool])
        assert graph is not None

    def test_creates_graph_without_checkpointer(self):
        """Should work without checkpointer."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        graph = create_graph([mock_tool], checkpointer=None)
        assert graph is not None

    def test_creates_graph_with_checkpointer(self):
        """Should accept checkpointer parameter."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_checkpointer = MagicMock()

        graph = create_graph([mock_tool], checkpointer=mock_checkpointer)
        assert graph is not None


class TestCreateSegmentedGraph:
    """Tests for create_segmented_graph function (Frank-style cycles)."""

    def test_creates_graph_with_tools(self):
        """Should create segmented graph with provided tools."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="result")

        graph = create_segmented_graph([mock_tool])
        assert graph is not None

    def test_creates_graph_without_checkpointer(self):
        """Should work without checkpointer."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        graph = create_segmented_graph([mock_tool], checkpointer=None)
        assert graph is not None

    def test_creates_graph_with_checkpointer(self):
        """Should accept checkpointer parameter."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_checkpointer = MagicMock()

        graph = create_segmented_graph([mock_tool], checkpointer=mock_checkpointer)
        assert graph is not None


class TestStripImagesFromHistory:
    """Tests for strip_images_from_history function.

    Note: LangChain's ToolMessage converts dict content to string repr,
    so tests use the actual string format that occurs at runtime.
    """

    def test_empty_messages(self):
        """Should handle empty message list."""
        result = strip_images_from_history([])
        assert result == []

    def test_no_tool_messages(self):
        """Should pass through messages without ToolMessages unchanged."""
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        result = strip_images_from_history(messages)
        assert len(result) == 3
        assert result[0].content == "System prompt"
        assert result[1].content == "Hello"
        assert result[2].content == "Hi there"

    def test_single_tool_message_preserved(self):
        """Single ToolMessage should keep its image (it's the most recent)."""
        # LangChain converts dict to string, so we test with actual string format
        content = "{'message': 'Drew circle', 'image': b'PNG_BYTES'}"
        messages = [
            SystemMessage(content="System"),
            ToolMessage(content=content, name="canvas_draw", tool_call_id="tc1"),
        ]
        result = strip_images_from_history(messages)
        assert len(result) == 2
        # Image should be preserved (not stripped) since it's the only/latest ToolMessage
        assert "b'PNG_BYTES'" in result[1].content

    def test_strips_old_images_keeps_latest(self):
        """Should strip images from older ToolMessages, keep latest."""
        old_content = "{'message': 'First draw', 'image': b'OLD_PNG_DATA'}"
        new_content = "{'message': 'Second draw', 'image': b'NEW_PNG_DATA'}"
        messages = [
            SystemMessage(content="System"),
            ToolMessage(content=old_content, name="canvas_draw", tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[{"id": "tc2", "name": "canvas_draw", "args": {}}]),
            ToolMessage(content=new_content, name="canvas_draw", tool_call_id="tc2"),
        ]
        result = strip_images_from_history(messages)
        assert len(result) == 4
        # First ToolMessage should have image stripped
        assert "[image stripped from history]" in result[1].content
        assert "First draw" in result[1].content
        assert "b'OLD_PNG" not in result[1].content
        # Second ToolMessage should keep image
        assert "b'NEW_PNG_DATA'" in result[3].content
        assert "Second draw" in result[3].content

    def test_keep_last_n_images(self):
        """Should keep images from last N ToolMessages."""
        messages = [
            ToolMessage(content="{'image': b'IMG1'}", name="t", tool_call_id="1"),
            ToolMessage(content="{'image': b'IMG2'}", name="t", tool_call_id="2"),
            ToolMessage(content="{'image': b'IMG3'}", name="t", tool_call_id="3"),
        ]
        result = strip_images_from_history(messages, keep_last_n=2)
        assert "[image stripped from history]" in result[0].content
        assert "b'IMG2'" in result[1].content
        assert "b'IMG3'" in result[2].content

    def test_preserves_non_image_content(self):
        """Should preserve other fields in ToolMessage content."""
        old_content = "{'message': 'Result', 'svg_saved': '/path/to/file.svg', 'image': b'PNG'}"
        new_content = "{'message': 'Latest', 'image': b'LATEST'}"
        messages = [
            ToolMessage(content=old_content, name="canvas_draw", tool_call_id="tc1"),
            ToolMessage(content=new_content, name="canvas_draw", tool_call_id="tc2"),
        ]
        result = strip_images_from_history(messages)
        # First message stripped but other fields preserved
        assert "Result" in result[0].content
        assert "/path/to/file.svg" in result[0].content
        assert "[image stripped from history]" in result[0].content
        assert "b'PNG'" not in result[0].content

    def test_handles_plain_string_content(self):
        """Should pass through ToolMessages with plain string content unchanged."""
        messages = [
            ToolMessage(content="Just a string result", name="think", tool_call_id="tc1"),
            ToolMessage(content="{'image': b'PNG'}", name="canvas_draw", tool_call_id="tc2"),
        ]
        result = strip_images_from_history(messages)
        assert result[0].content == "Just a string result"
        # Latest message preserved
        assert "b'PNG'" in result[1].content

    def test_handles_list_content(self):
        """Should handle list content with nested dicts containing images."""
        # List content is preserved by LangChain, so test with actual list
        messages = [
            ToolMessage(
                content=[{"type": "text", "text": "Result"}, {"type": "image", "image": b"OLD"}],
                name="multi_tool",
                tool_call_id="tc1",
            ),
            ToolMessage(
                content="{'image': b'LATEST'}",
                name="canvas_draw",
                tool_call_id="tc2",
            ),
        ]
        result = strip_images_from_history(messages)
        # First message (list) should have nested image stripped
        assert result[0].content[0] == {"type": "text", "text": "Result"}
        assert result[0].content[1]["image"] == "[image stripped from history]"
        # Latest message preserved
        assert "b'LATEST'" in result[1].content

    def test_preserves_tool_message_metadata(self):
        """Should preserve ToolMessage fields like name, tool_call_id."""
        messages = [
            ToolMessage(content="{'image': b'OLD'}", name="canvas_draw", tool_call_id="original_id_123"),
            ToolMessage(content="{'image': b'NEW'}", name="canvas_view", tool_call_id="latest_id_456"),
        ]
        result = strip_images_from_history(messages)
        assert result[0].name == "canvas_draw"
        assert result[0].tool_call_id == "original_id_123"
        assert result[1].name == "canvas_view"
        assert result[1].tool_call_id == "latest_id_456"

    def test_no_images_to_strip(self):
        """Should handle ToolMessages without image field gracefully."""
        messages = [
            ToolMessage(content="{'status': 'ok', 'count': 5}", name="memory_list", tool_call_id="tc1"),
            ToolMessage(content="{'status': 'done'}", name="submit_data", tool_call_id="tc2"),
        ]
        result = strip_images_from_history(messages)
        # Content should be unchanged since no 'image': b'...' pattern
        assert result[0].content == "{'status': 'ok', 'count': 5}"
        assert result[1].content == "{'status': 'done'}"

    def test_real_png_data(self):
        """Test with realistic PNG byte content like in actual history."""
        old_content = "{'message': 'Created', 'image': b'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR', 'svg': '/path.svg'}"
        new_content = "{'message': 'Latest', 'image': b'\\x89PNG\\r\\n\\x00\\x00'}"
        messages = [
            ToolMessage(content=old_content, name="canvas_draw", tool_call_id="tc1"),
            ToolMessage(content=new_content, name="canvas_draw", tool_call_id="tc2"),
        ]
        result = strip_images_from_history(messages)
        # Old image stripped
        assert "\\x89PNG" not in result[0].content
        assert "[image stripped from history]" in result[0].content
        assert "Created" in result[0].content
        assert "/path.svg" in result[0].content
        # Latest preserved
        assert "\\x89PNG" in result[1].content
