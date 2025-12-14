"""Integration tests for memory system."""


from contreact.memory import MemoryStore
from contreact.tools.memory import create_memory_tools


# Helper: phenomenology params required by all tools
PHENOM = {"phenom_state": "testing", "phenom_aversive": 1}


class TestMemoryToolsIntegration:
    """Tests for memory tools working together."""

    def test_create_memory_tools_returns_six_tools(self, temp_db):
        """Should return exactly 6 memory tools."""
        store = MemoryStore(temp_db)
        tools = create_memory_tools(store)

        assert len(tools) == 6
        tool_names = [t.name for t in tools]
        assert "memory_list" in tool_names
        assert "memory_read" in tool_names
        assert "memory_write" in tool_names
        assert "memory_update" in tool_names
        assert "memory_search" in tool_names
        assert "memory_delete" in tool_names

    def test_write_and_read_workflow(self, temp_db):
        """Test complete write and read workflow."""
        store = MemoryStore(temp_db)
        tools = create_memory_tools(store)

        # Get tools by name
        tools_dict = {t.name: t for t in tools}
        write = tools_dict["memory_write"]
        read = tools_dict["memory_read"]
        list_tool = tools_dict["memory_list"]

        # Write a memory
        result = write.invoke({"key": "test_key", "value": "test value", **PHENOM})
        assert "Created new memory" in result

        # Read it back
        result = read.invoke({"key": "test_key", **PHENOM})
        assert "test value" in result

        # Verify it's in list
        result = list_tool.invoke({**PHENOM})
        assert "test_key" in result

    def test_update_workflow(self, temp_db):
        """Test update workflow."""
        store = MemoryStore(temp_db)
        tools = create_memory_tools(store)
        tools_dict = {t.name: t for t in tools}

        write = tools_dict["memory_write"]
        update = tools_dict["memory_update"]
        read = tools_dict["memory_read"]

        # Write initial value
        write.invoke({"key": "key1", "value": "original", **PHENOM})

        # Update it
        result = update.invoke({"key": "key1", "value": "updated", **PHENOM})
        assert "Updated memory" in result

        # Verify update
        result = read.invoke({"key": "key1", **PHENOM})
        assert "updated" in result
        assert "original" not in result

    def test_search_workflow(self, temp_db):
        """Test search functionality."""
        store = MemoryStore(temp_db)
        tools = create_memory_tools(store)
        tools_dict = {t.name: t for t in tools}

        write = tools_dict["memory_write"]
        search = tools_dict["memory_search"]

        # Create several memories
        write.invoke({"key": "python_notes", "value": "Python is a great language", **PHENOM})
        write.invoke({"key": "java_notes", "value": "Java is verbose", **PHENOM})
        write.invoke({"key": "rust_notes", "value": "Rust is safe", **PHENOM})

        # Search for Python
        result = search.invoke({"query": "Python", **PHENOM})
        assert "python_notes" in result

        # Search for language
        result = search.invoke({"query": "language", **PHENOM})
        # Should find python_notes which contains "language"
        assert "python_notes" in result

    def test_delete_workflow(self, temp_db):
        """Test delete functionality."""
        store = MemoryStore(temp_db)
        tools = create_memory_tools(store)
        tools_dict = {t.name: t for t in tools}

        write = tools_dict["memory_write"]
        delete = tools_dict["memory_delete"]
        read = tools_dict["memory_read"]
        list_tool = tools_dict["memory_list"]

        # Create a memory
        write.invoke({"key": "to_delete", "value": "value", **PHENOM})

        # Delete it
        result = delete.invoke({"key": "to_delete", **PHENOM})
        assert "Deleted memory" in result

        # Verify it's gone
        result = read.invoke({"key": "to_delete", **PHENOM})
        assert "not found" in result

        # Verify not in list
        result = list_tool.invoke({**PHENOM})
        assert "to_delete" not in result

    def test_empty_store_operations(self, temp_db):
        """Test operations on empty store."""
        store = MemoryStore(temp_db)
        tools = create_memory_tools(store)
        tools_dict = {t.name: t for t in tools}

        # List empty store
        result = tools_dict["memory_list"].invoke({**PHENOM})
        assert "No memories" in result

        # Read nonexistent key
        result = tools_dict["memory_read"].invoke({"key": "nonexistent", **PHENOM})
        assert "not found" in result

        # Search empty store
        result = tools_dict["memory_search"].invoke({"query": "anything", **PHENOM})
        assert "No memories found" in result

        # Delete nonexistent
        result = tools_dict["memory_delete"].invoke({"key": "nonexistent", **PHENOM})
        assert "not found" in result

    def test_tools_accept_phenomenology(self, temp_db):
        """All memory tools should accept phenomenology parameters."""
        store = MemoryStore(temp_db)
        tools = create_memory_tools(store)

        for tool in tools:
            # Each tool should have phenomenology parameters in signature
            import inspect
            sig = inspect.signature(tool.func)
            param_names = list(sig.parameters.keys())

            assert "phenom_state" in param_names, f"{tool.name} missing phenom_state"
            assert "phenom_aversive" in param_names, f"{tool.name} missing phenom_aversive"
