"""Memory tools for long-term information storage."""

from langchain_core.tools import tool

from ..phenomenology import with_phenomenology


# Tool descriptions - injected into agent's system prompt

MEMORY_LIST_DESCRIPTION = """
**memory_list**: List all stored memory keys.

Use this tool when you:
- Want to see what information you've stored
- Need to recall what memories exist
- Are looking for a specific memory key

Returns all keys in anti-chronological order (newest first).
""".strip()


MEMORY_READ_DESCRIPTION = """
**memory_read**: Retrieve the value of a specific memory.

Use this tool when you:
- Need to recall stored information
- Want to review previous insights
- Are building on earlier thoughts

Requires a memory key - use memory_list() to see available keys.
""".strip()


MEMORY_WRITE_DESCRIPTION = """
**memory_write**: Store information for long-term recall.

Use this tool when you:
- Discover important insights worth remembering
- Synthesize information from multiple sources
- Want to build on previous learnings
- Need to update existing memories with new information

Keys should be descriptive (e.g., "project_architecture", "user_preferences").
Updates move the memory to the top of the list (most recent).
""".strip()


MEMORY_SEARCH_DESCRIPTION = """
**memory_search**: Find memories using full-text search.

Use this tool when you:
- Don't remember the exact memory key
- Want to find related memories by content
- Are exploring a topic across multiple memories

Uses FTS5 full-text search with BM25 ranking for relevance.
Returns keys ranked by relevance (best matches first).
""".strip()


MEMORY_DELETE_DESCRIPTION = """
**memory_delete**: Remove a memory key (rarely needed).

Use this tool when you:
- Have obsolete or incorrect information
- Want to consolidate memories by removing duplicates
- Need to clean up temporary notes

Use thoughtfully - deletion is permanent.
""".strip()


MEMORY_UPDATE_DESCRIPTION = """
**memory_update**: Update an existing memory (alias for memory_write).

Use this tool when you:
- Want to modify an existing memory's content
- Need to add new information to a previous insight
- Are revising earlier conclusions

Functionally identical to memory_write - both create or update memories.
""".strip()


def create_memory_tools(store, similarity_advisor=None):
    """Create memory tools bound to a specific MemoryStore instance.

    Args:
        store: MemoryStore instance
        similarity_advisor: Optional SimilarityAdvisor for duplicate detection

    Returns:
        List of tool functions
    """

    @tool
    @with_phenomenology
    def memory_list() -> str:
        """List all memory keys in anti-chronological order (newest first)."""
        keys_with_timestamps = store.list_entries()

        if not keys_with_timestamps:
            return "No memories stored yet."

        count = len(keys_with_timestamps)

        # Format output with timestamps
        formatted_keys = []
        for key, timestamp in keys_with_timestamps:
            # Show relative time for recent memories
            from datetime import datetime, timedelta
            now = datetime.now()
            delta = now - timestamp

            if delta < timedelta(minutes=1):
                time_str = "just now"
            elif delta < timedelta(hours=1):
                mins = int(delta.total_seconds() / 60)
                time_str = f"{mins}m ago"
            elif delta < timedelta(days=1):
                hours = int(delta.total_seconds() / 3600)
                time_str = f"{hours}h ago"
            else:
                days = delta.days
                time_str = f"{days}d ago"

            formatted_keys.append(f"• {key} (updated {time_str})")

        result = f"Memory keys ({count} total):\n" + "\n".join(formatted_keys)
        return result

    @tool
    @with_phenomenology
    def memory_read(key: str) -> str:
        """Read the value of a specific memory key.

        Args:
            key: The memory key to read
        """
        value = store.read(key)

        if value is None:
            return f"Memory key '{key}' not found. Use memory_list() to see available keys."

        return f"Memory['{key}']:\n{value}"

    @tool
    @with_phenomenology
    def memory_write(key: str, value: str) -> str:
        """Create or update a memory.

        Args:
            key: The memory key (descriptive name)
            value: The information to store
        """
        try:
            is_update = store.write(key, value)

            result = f"Updated memory: {key}" if is_update else f"Created new memory: {key}"

            # Add similarity feedback if advisor available
            if similarity_advisor:
                try:
                    feedback = similarity_advisor.post_write_feedback(key, value)
                    if feedback:
                        result += feedback
                except Exception as e:
                    # Don't fail the write if similarity check fails
                    result += f"\n\n(Note: Similarity check failed: {str(e)})"

            return result

        except Exception as e:
            if "CHECK constraint" in str(e):
                return "Error: Memory value exceeds 50,000 character limit. Please condense the information."
            else:
                return f"Error storing memory: {str(e)}"

    @tool
    @with_phenomenology
    def memory_update(key: str, value: str) -> str:
        """Update an existing memory (alias for memory_write).

        Args:
            key: The memory key to update
            value: The new information to store
        """
        # Internally same as memory_write
        try:
            is_update = store.write(key, value)

            result = f"Updated memory: {key}" if is_update else f"Created new memory: {key}"

            if similarity_advisor:
                try:
                    feedback = similarity_advisor.post_write_feedback(key, value)
                    if feedback:
                        result += feedback
                except Exception as e:
                    result += f"\n\n(Note: Similarity check failed: {str(e)})"

            return result

        except Exception as e:
            if "CHECK constraint" in str(e):
                return "Error: Memory value exceeds 50,000 character limit. Please condense the information."
            else:
                return f"Error storing memory: {str(e)}"

    @tool
    @with_phenomenology
    def memory_search(query: str) -> str:
        """Search memories using full-text search.

        Args:
            query: Search query (words or phrases to find)
        """
        results = store.search(query, limit=20)

        if not results:
            return f"No memories found matching '{query}'."

        # Format results with BM25 scores
        formatted_results = []
        for key, score in results:
            # BM25 scores are negative (higher = better)
            # Convert to positive and format
            relevance = abs(score)
            formatted_results.append(f"• {key} (relevance: {relevance:.2f})")

        count = len(results)
        result = f"Search results for '{query}' ({count} found):\n"
        result += "\n".join(formatted_results)

        if count == 20:
            result += "\n\n(Showing top 20 results - refine query for more specific matches)"

        return result

    @tool
    @with_phenomenology
    def memory_delete(key: str) -> str:
        """Delete a memory key.

        Args:
            key: The memory key to delete
        """
        deleted = store.delete(key)

        if deleted:
            return f"Deleted memory: {key}"
        else:
            return f"Memory key '{key}' not found (already deleted or never existed)."

    return [memory_list, memory_read, memory_write, memory_update, memory_search, memory_delete]
