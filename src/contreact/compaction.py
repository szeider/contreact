"""History compaction for ContReAct agent.

Compacts conversation history by summarizing older messages while
preserving recent context. Triggered by message count threshold.
"""

from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import RemoveMessage


# Marker for compacted history detection
COMPACTED_MARKER = "[COMPACTED HISTORY"

# Wrapper template for compacted history
COMPACTED_HISTORY_WRAPPER = """[COMPACTED HISTORY - REFERENCE ONLY]
This is a summary of your previous activity for context. Do NOT re-analyze this content - it's your own past work already processed.

{summary_text}

[END OF COMPACTED HISTORY - Continue with NEW activity]"""


def should_compact(messages: list[BaseMessage], config: dict) -> bool:
    """Check if compaction is needed based on message count.

    Args:
        messages: Current message history
        config: Run configuration with compaction settings

    Returns:
        True if compaction should be triggered
    """
    compaction_cfg = config.get("compaction", {})
    if not compaction_cfg.get("enabled", False):
        return False

    trigger = compaction_cfg.get("trigger_messages", 60)
    keep_recent = compaction_cfg.get("keep_recent", 20)

    # Count non-system messages (excluding any existing compacted history)
    msg_count = sum(
        1 for m in messages
        if not isinstance(m, SystemMessage)
    )

    # Must exceed trigger AND have more than keep_recent (to avoid infinite loop)
    return msg_count >= trigger and msg_count > keep_recent


def load_compaction_prompt(run_name: str) -> str:
    """Load compaction prompt from run folder.

    Args:
        run_name: Path to run folder

    Returns:
        Compaction prompt text

    Raises:
        FileNotFoundError: If compaction_prompt.md is missing
    """
    prompt_path = Path(run_name) / "compaction_prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Compaction is enabled but compaction_prompt.md not found: {prompt_path}. "
            "Copy compaction_prompt.md from project root or disable compaction."
        )
    return prompt_path.read_text(encoding="utf-8")


def _messages_to_text(messages: list[BaseMessage]) -> str:
    """Convert messages to text format for summarization.

    Args:
        messages: Messages to convert

    Returns:
        Text representation of messages
    """
    lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Skip system messages in summary input
            continue
        elif isinstance(msg, HumanMessage):
            lines.append(f"[Human]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                lines.append(f"[Assistant]: {str(msg.content)}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    # Handle both dict and object shapes
                    name = getattr(tc, "name", None) or tc.get("name", "unknown")
                    args = getattr(tc, "args", None) if hasattr(tc, "args") else tc.get("args", {})
                    lines.append(f"[Tool Call]: {name}({args})")
        elif isinstance(msg, ToolMessage):
            # Truncate long tool results
            content = str(msg.content)
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            lines.append(f"[Tool Result ({msg.name})]: {content}")

    return "\n\n".join(lines)


def _extract_system_prompt(messages: list[BaseMessage]) -> SystemMessage | None:
    """Extract the original system prompt (not compacted history).

    Args:
        messages: Message history

    Returns:
        Original system prompt or None
    """
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Skip compacted history markers
            if COMPACTED_MARKER not in str(msg.content):
                return msg
    return None


def _extract_previous_summary(messages: list[BaseMessage]) -> str | None:
    """Extract previous compacted summary if it exists.

    Args:
        messages: Message history

    Returns:
        Previous summary text or None
    """
    for msg in messages:
        if isinstance(msg, SystemMessage) and COMPACTED_MARKER in str(msg.content):
            return str(msg.content)
    return None


def compact_history(
    messages: list[BaseMessage],
    llm,
    compaction_prompt: str,
    config: dict,
    run_name: str
) -> list[BaseMessage]:
    """Compact history by summarizing older messages.

    Uses the same LLM to generate a summary of older messages,
    preserving recent messages for continuity.

    Args:
        messages: Current message history
        llm: LLM instance (same as agent uses)
        compaction_prompt: Prompt template for summarization
        config: Run configuration
        run_name: Path to run folder (for logging)

    Returns:
        Compacted message list

    Raises:
        RuntimeError: If summarization fails
    """
    compaction_cfg = config.get("compaction", {})
    keep_recent = compaction_cfg.get("keep_recent", 20)

    # Extract components (system_prompt stays in state, we just need previous_summary for context)
    _system_prompt = _extract_system_prompt(messages)  # noqa: F841 - kept for clarity
    previous_summary = _extract_previous_summary(messages)

    # Get non-system messages
    content_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(content_messages) <= keep_recent:
        # Nothing to compact
        return messages

    # Split into messages to summarize and messages to keep
    split_idx = len(content_messages) - keep_recent

    # Avoid orphaned ToolMessages: if split lands on a ToolMessage,
    # move split back to include the preceding AIMessage (tool call)
    while split_idx > 0 and isinstance(content_messages[split_idx], ToolMessage):
        split_idx -= 1

    messages_to_summarize = content_messages[:split_idx]
    # messages_to_keep stay in state (not re-added), we just remove the summarized ones
    _messages_to_keep = content_messages[split_idx:]  # noqa: F841 - documents the split

    # Build context for summarization
    context_parts = []
    if previous_summary:
        context_parts.append("Previous summary exists - build upon it, don't repeat.")
        context_parts.append(f"Previous summary:\n{previous_summary}")
    else:
        context_parts.append("This is the first summarization of this conversation.")

    context_description = "\n\n".join(context_parts)
    messages_text = _messages_to_text(messages_to_summarize)

    # Build summarization prompt
    try:
        full_prompt = compaction_prompt.format(
            context_description=context_description,
            messages_text=messages_text,
            message_count=len(messages_to_summarize)
        )
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            f"Compaction prompt formatting failed: {e}. "
            "If compaction_prompt.md contains literal '{' or '}', escape them as '{{' and '}}'."
        ) from e

    print(f"[Compacting history: {len(content_messages)} -> ~{keep_recent + 2} messages]")

    # Call LLM for summarization (without tools)
    try:
        # Use base LLM without tools bound
        base_llm = llm.bind_tools([])  # Empty tools list
        response = base_llm.invoke([
            SystemMessage(content="You are a summarization assistant. Create concise, accurate summaries."),
            HumanMessage(content=full_prompt)
        ])
        summary_text = response.content
    except Exception as e:
        raise RuntimeError(f"Compaction failed: {type(e).__name__}: {e}")

    # Build compacted history with RemoveMessage for deleted messages
    compacted: list[BaseMessage] = []

    # 1. Remove all messages that were summarized (by ID)
    for msg in messages_to_summarize:
        if hasattr(msg, 'id') and msg.id:
            compacted.append(RemoveMessage(id=msg.id))

    # 2. Also remove any previous compacted history system message
    for msg in messages:
        if isinstance(msg, SystemMessage) and COMPACTED_MARKER in str(msg.content):
            if hasattr(msg, 'id') and msg.id:
                compacted.append(RemoveMessage(id=msg.id))

    # 3. Add new compacted history as system message
    wrapped_summary = COMPACTED_HISTORY_WRAPPER.format(summary_text=summary_text)
    compacted.append(SystemMessage(content=wrapped_summary))

    # Note: system_prompt and messages_to_keep are already in state, don't re-add them

    print(f"[Compaction complete: {len(messages)} -> ~{keep_recent + 2} messages, summary: {len(summary_text)} chars]")

    return compacted


def should_truncate(messages: list[BaseMessage], config: dict) -> bool:
    """Check if truncation is needed based on message count.

    Truncation is a lightweight alternative to compaction: it simply drops
    old messages (no summary, no LLM call). Suitable for exhibition mode
    where the agent can re-derive state from canvas/memory.

    Args:
        messages: Current message history
        config: Run configuration with truncation settings

    Returns:
        True if truncation should be triggered
    """
    truncation_cfg = config.get("truncation", {})
    if not truncation_cfg.get("enabled", False):
        return False

    keep_recent = truncation_cfg.get("keep_recent", 40)
    # Trigger with a buffer to avoid truncating on every single tool call
    trigger_margin = max(4, keep_recent // 5)  # at least 4, or 20% of keep_recent

    # Count non-system messages
    msg_count = sum(
        1 for m in messages
        if not isinstance(m, SystemMessage)
    )

    return msg_count >= keep_recent + trigger_margin


def truncate_history(messages: list[BaseMessage], config: dict) -> list[BaseMessage]:
    """Truncate history by dropping old messages (no summary, no LLM call).

    Returns RemoveMessage directives for old non-system messages, keeping
    the last keep_recent. Avoids splitting AIMessage from its ToolMessages.

    Args:
        messages: Current message history
        config: Run configuration with truncation settings

    Returns:
        List of RemoveMessage directives to apply
    """
    truncation_cfg = config.get("truncation", {})
    keep_recent = truncation_cfg.get("keep_recent", 40)

    # Get non-system messages
    content_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(content_messages) <= keep_recent:
        return []

    # Split point: keep the last keep_recent messages
    split_idx = len(content_messages) - keep_recent

    # Avoid orphaned ToolMessages: if split lands on a ToolMessage,
    # move split back to include the preceding AIMessage (tool call)
    while split_idx > 0 and isinstance(content_messages[split_idx], ToolMessage):
        split_idx -= 1

    messages_to_remove = content_messages[:split_idx]

    if not messages_to_remove:
        return []

    # Build RemoveMessage directives (skip messages without id)
    removals: list[BaseMessage] = []
    for msg in messages_to_remove:
        if hasattr(msg, 'id') and msg.id:
            removals.append(RemoveMessage(id=msg.id))

    if not removals:
        return []

    print(f"[Truncating history: {len(content_messages)} -> ~{keep_recent} messages, removing {len(removals)}]")

    return removals
