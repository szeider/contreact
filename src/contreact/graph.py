"""Graph construction for continuous ReAct agent."""

import re
import time
from typing import Annotated, Sequence, TypedDict

import httpx
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError

from .tools import StopSignal


# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 30.0  # seconds

# Pattern to match 'image': b'...' in string repr of dicts
# Matches: 'image': b'...' or "image": b"..."
_IMAGE_PATTERN = re.compile(
    r"(['\"])image\1\s*:\s*b['\"].*?['\"](?=\s*[,}])",
    re.DOTALL
)


def strip_images_from_history(
    messages: Sequence[BaseMessage], keep_last_n: int = 1
) -> list[BaseMessage]:
    """Strip image data from ToolMessages, keeping only the most recent N.

    This reduces context size by replacing image bytes with placeholders
    in older tool results. The agent can always re-call canvas_view() to
    see any canvas if needed.

    Note: LangChain's ToolMessage converts dict content to string repr,
    so we use regex to strip image bytes from the string.

    Args:
        messages: Full message history
        keep_last_n: Number of most recent ToolMessages to preserve images for

    Returns:
        Message list with historical images stripped
    """
    # Find indices of all ToolMessages
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]

    if not tool_indices:
        return list(messages)

    # Keep images in the last N ToolMessages
    keep_indices = set(tool_indices[-keep_last_n:])

    def scrub_content(content):
        """Replace image bytes with placeholder text."""
        if isinstance(content, str):
            # LangChain converts dicts to string repr, so use regex
            # Replace 'image': b'...' with 'image': '[stripped]'
            return _IMAGE_PATTERN.sub(
                "'image': '[image stripped from history]'",
                content
            )
        if isinstance(content, dict):
            if "image" in content:
                scrubbed = dict(content)
                scrubbed["image"] = "[image stripped from history]"
                return scrubbed
            return content
        if isinstance(content, list):
            return [
                scrub_content(item) if isinstance(item, (dict, list, str)) else item
                for item in content
            ]
        return content

    result: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and i not in keep_indices:
            new_content = scrub_content(msg.content)
            # Use model_copy to preserve all message metadata
            copier = getattr(msg, "model_copy", msg.copy)
            result.append(copier(update={"content": new_content}))
        else:
            result.append(msg)

    return result


class AgentState(TypedDict):
    """State for the continuous ReAct agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_graph(tools: list, checkpointer=None) -> StateGraph:
    """Create the continuous ReAct agent graph.

    This mimics LangGraph's create_react_agent but WITHOUT the END route.

    Args:
        tools: List of tools available to the agent
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled graph
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        """Node that calls the LLM with tools bound.

        Includes retry logic for transient API errors with exponential backoff.
        Strips images from historical ToolMessages to reduce context size.
        """
        llm = config["configurable"]["llm"]

        # Strip images from history to reduce context size
        messages = strip_images_from_history(state["messages"])

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = llm.invoke(messages, config)
                return {"messages": [response]}
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError, httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"[API Error (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}. Retrying in {delay:.1f}s...]")
                    time.sleep(delay)
                else:
                    print(f"[API Error: {type(e).__name__} after {MAX_RETRIES} attempts. Giving up.]")

        # Re-raise the last error if all retries failed
        raise last_error

    def call_tools(state: AgentState) -> dict:
        """Node that executes tool calls.

        Catches ValueError (e.g., missing phenomenology params) and returns
        error message to LLM so it can retry with correct parameters.
        """
        last_message = state["messages"][-1]
        outputs = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = tools_by_name.get(tool_name)

            if tool is None:
                # LLM hallucinated a tool name
                available = ", ".join(sorted(tools_by_name.keys()))
                result = f"TOOL ERROR: Unknown tool '{tool_name}'. Available tools: {available}"
                print(f"[Unknown tool: {tool_name}]")
            else:
                try:
                    result = tool.invoke(tool_call["args"])
                except StopSignal:
                    # Re-raise StopSignal to propagate to main loop
                    raise
                except Exception as e:
                    # Return error to LLM so it can fix and retry
                    result = f"TOOL ERROR: {type(e).__name__}: {str(e)}\n\nPlease check your parameters and try again."
                    print(f"[Tool error: {tool_name} - {type(e).__name__}: {str(e)[:80]}]")

            outputs.append(
                ToolMessage(
                    content=result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}

    def route_next(state: AgentState) -> str:
        """
        Routing function - ALWAYS routes to tools (never to END).

        This is the key difference from standard ReAct.
        """
        last_message = state["messages"][-1]

        if not last_message.tool_calls:
            # Standard ReAct would return "end" here
            # We return "continue" to stay in the loop
            print("[Warning: Model produced no tool calls. Continuing anyway.]")
            return "continue"
        else:
            return "continue"

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges - CRITICAL: no path to END
    workflow.add_conditional_edges(
        "agent",
        route_next,
        {
            "continue": "tools",  # Only one option - always go to tools
        }
    )

    # Always loop back from tools to agent
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


def create_segmented_graph(tools: list, checkpointer=None) -> StateGraph:
    """Create a segmented ReAct agent graph.

    Unlike create_graph(), this has an END route when the agent produces
    no tool calls. Used for segmented execution mode where cycles end
    naturally and restart with a wake message.

    Args:
        tools: List of tools available to the agent
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled graph
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        """Node that calls the LLM with tools bound."""
        llm = config["configurable"]["llm"]
        messages = strip_images_from_history(state["messages"])

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = llm.invoke(messages, config)
                return {"messages": [response]}
            except (APIError, RateLimitError, APIConnectionError, APITimeoutError, httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"[API Error (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}. Retrying in {delay:.1f}s...]")
                    time.sleep(delay)
                else:
                    print(f"[API Error: {type(e).__name__} after {MAX_RETRIES} attempts. Giving up.]")

        raise last_error

    def call_tools(state: AgentState) -> dict:
        """Node that executes tool calls."""
        last_message = state["messages"][-1]
        outputs = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = tools_by_name.get(tool_name)

            if tool is None:
                available = ", ".join(sorted(tools_by_name.keys()))
                result = f"TOOL ERROR: Unknown tool '{tool_name}'. Available tools: {available}"
                print(f"[Unknown tool: {tool_name}]")
            else:
                try:
                    result = tool.invoke(tool_call["args"])
                except StopSignal:
                    # Re-raise StopSignal to propagate to main loop
                    raise
                except Exception as e:
                    result = f"TOOL ERROR: {type(e).__name__}: {str(e)}\n\nPlease check your parameters and try again."
                    print(f"[Tool error: {tool_name} - {type(e).__name__}: {str(e)[:80]}]")

            outputs.append(
                ToolMessage(
                    content=result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}

    def route_next(state: AgentState) -> str:
        """Routing function - routes to END when no tool calls (standard ReAct)."""
        last_message = state["messages"][-1]

        if not last_message.tool_calls:
            return "end"
        return "continue"

    # Build graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    workflow.set_entry_point("agent")

    # Add edges - includes END route for segmented mode
    workflow.add_conditional_edges(
        "agent",
        route_next,
        {
            "continue": "tools",
            "end": END,
        }
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)
