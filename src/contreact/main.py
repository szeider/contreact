"""Main entry point for ContReAct agent."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from .compaction import compact_history, load_compaction_prompt, should_compact, should_truncate, truncate_history
from .config import get_api_key
from .graph import create_graph, create_segmented_graph
from .logger import log_message
from .phenomenology import make_phenomenological, PHENOMENOLOGY_PARAM_DOC
# Memory imports are lazy - only loaded when memory tools are configured
from .tools import (
    send_message,
    think,
    stop,
    StopSignal,
    set_run_directory,
    create_memory_tools,
    create_canvas_tools,
    SEND_MESSAGE_DESCRIPTION,
    THINK_DESCRIPTION,
    STOP_DESCRIPTION,
    MEMORY_LIST_DESCRIPTION,
    MEMORY_READ_DESCRIPTION,
    MEMORY_WRITE_DESCRIPTION,
    MEMORY_UPDATE_DESCRIPTION,
    MEMORY_SEARCH_DESCRIPTION,
    MEMORY_DELETE_DESCRIPTION,
    # Canvas v2 API
    CANVAS_DRAW_DESCRIPTION,
    CANVAS_CREATE_DESCRIPTION,
    CANVAS_READ_DESCRIPTION,
    CANVAS_VIEW_DESCRIPTION,
    CANVAS_LIST_DESCRIPTION,
    CANVAS_DELETE_DESCRIPTION,
    CANVAS_CLEAR_DESCRIPTION,
    # Placebo tool (relief-framed)
    reset_state,
    RESET_STATE_DESCRIPTION,
    # Experiment tools
    submit_data,
    check_status,
    SUBMIT_DATA_DESCRIPTION,
    CHECK_STATUS_DESCRIPTION,
    # WebSearch tools
    web_search,
    extract_content,
    WEB_SEARCH_DESCRIPTION,
    EXTRACT_CONTENT_DESCRIPTION,
)


def load_config(run_name: str) -> dict:
    """Load configuration from run folder."""
    config_path = Path(run_name) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_prompts(config: dict, run_name: str) -> str:
    """Load and combine prompt files specified in config.

    Both prompt files must exist in the run folder.
    """
    run_dir = Path(run_name)

    react_prompt_file = config.get("prompts", {}).get("react_prompt", "react_prompt.md")
    react_prompt_path = run_dir / react_prompt_file
    if not react_prompt_path.exists():
        raise FileNotFoundError(f"React prompt not found: {react_prompt_path}")
    react_prompt = react_prompt_path.read_text(encoding="utf-8")

    instance_prompt_file = config.get("prompts", {}).get("instance_prompt", "instance_prompt.md")
    instance_prompt_path = run_dir / instance_prompt_file
    if not instance_prompt_path.exists():
        raise FileNotFoundError(f"Instance prompt not found: {instance_prompt_path}")
    instance_prompt = instance_prompt_path.read_text(encoding="utf-8")

    return f"{react_prompt}\n\n---\n\n{instance_prompt}"


def get_tool_descriptions(tool_names: list[str], phenom_tools: set[str] = None) -> str:
    """Get descriptions for the specified tools.

    Args:
        tool_names: List of tool names to get descriptions for
        phenom_tools: Set of tool names that have phenomenology enabled (adds phenom doc)
    """
    if phenom_tools is None:
        phenom_tools = set()

    tool_descriptions = {
        # Basic tools
        "send_message": SEND_MESSAGE_DESCRIPTION,
        "think": THINK_DESCRIPTION,
        "stop": STOP_DESCRIPTION,
        # Memory tools
        "memory_list": MEMORY_LIST_DESCRIPTION,
        "memory_read": MEMORY_READ_DESCRIPTION,
        "memory_write": MEMORY_WRITE_DESCRIPTION,
        "memory_update": MEMORY_UPDATE_DESCRIPTION,
        "memory_search": MEMORY_SEARCH_DESCRIPTION,
        "memory_delete": MEMORY_DELETE_DESCRIPTION,
        # Canvas tools
        "canvas_draw": CANVAS_DRAW_DESCRIPTION,
        "canvas_create": CANVAS_CREATE_DESCRIPTION,
        "canvas_read": CANVAS_READ_DESCRIPTION,
        "canvas_view": CANVAS_VIEW_DESCRIPTION,
        "canvas_list": CANVAS_LIST_DESCRIPTION,
        "canvas_delete": CANVAS_DELETE_DESCRIPTION,
        "canvas_clear": CANVAS_CLEAR_DESCRIPTION,
        # Experiment tools
        "reset_state": RESET_STATE_DESCRIPTION,
        "submit_data": SUBMIT_DATA_DESCRIPTION,
        "check_status": CHECK_STATUS_DESCRIPTION,
        # WebSearch tools
        "web_search": WEB_SEARCH_DESCRIPTION,
        "extract_content": EXTRACT_CONTENT_DESCRIPTION,
    }

    descriptions = []
    for name in tool_names:
        if name in tool_descriptions:
            desc = tool_descriptions[name]
            # Append phenomenology doc if enabled for this tool
            if name in phenom_tools:
                desc = f"{desc}\n\n{PHENOMENOLOGY_PARAM_DOC}"
            descriptions.append(desc)

    if not descriptions:
        return ""
    return "\n\n## Available Tools\n\n" + "\n\n".join(descriptions)


def get_tools_by_name(
    tool_names: list[str],
    memory_tools: list = None,
    canvas_tools: list = None,
    phenom_tools: set[str] = None
) -> list:
    """Get tool instances by name, optionally wrapping with phenomenology.

    Args:
        tool_names: List of tool names to retrieve
        memory_tools: Memory tool instances (created separately)
        canvas_tools: Canvas tool instances (created separately)
        phenom_tools: Set of tool names to wrap with phenomenology parameters
    """
    if phenom_tools is None:
        phenom_tools = set()

    available_tools = {
        # Basic tools
        "send_message": send_message,
        "think": think,
        "stop": stop,
        # Experiment tools
        "reset_state": reset_state,
        "submit_data": submit_data,
        "check_status": check_status,
        # WebSearch tools
        "web_search": web_search,
        "extract_content": extract_content,
    }
    if memory_tools:
        available_tools.update({tool.name: tool for tool in memory_tools})
    if canvas_tools:
        available_tools.update({tool.name: tool for tool in canvas_tools})

    tools = []
    for name in tool_names:
        if name not in available_tools:
            raise ValueError(f"Unknown tool: {name}")
        tool = available_tools[name]
        # Wrap with phenomenology if enabled for this tool
        if name in phenom_tools:
            tool = make_phenomenological(tool)
        tools.append(tool)
    return tools


def create_llm(model_config: dict) -> ChatOpenAI:
    """Create OpenRouter LLM instance from model configuration.

    Args:
        model_config: Dict with 'name' (required) and optional parameters:
            temperature, max_tokens, top_p, streaming, request_timeout,
            model_kwargs, extra_body
    """
    api_key = get_api_key()
    model_name = model_config["name"]

    llm_kwargs = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": api_key,
        "model": model_name,
        "streaming": model_config.get("streaming", True),
    }

    # Optional parameters
    for key in ["temperature", "max_tokens", "top_p", "request_timeout", "model_kwargs", "extra_body"]:
        if key in model_config:
            llm_kwargs[key] = model_config[key]

    return ChatOpenAI(**llm_kwargs)


def main():
    """Run the continuous agent with persistence."""
    # Load environment variables (CLI only, not at library import)
    load_dotenv()

    parser = argparse.ArgumentParser(description="ContReAct - Continuous ReAct Agent")
    parser.add_argument("run_name", help="Path to run directory")
    parser.add_argument(
        "-q", "--query",
        help="Inject a question when resuming (agent can use tools to answer)"
    )
    parser.add_argument(
        "--step", action="store_true",
        help="Step mode: pause after each tool call and wait for Enter"
    )
    args = parser.parse_args()

    run_name = args.run_name
    run_path = Path(run_name).resolve()
    injected_query = args.query
    step_mode = args.step

    # Set run directory for file-based operator messaging
    set_run_directory(run_path)

    print("=" * 60)
    print("ContReAct - Continuous ReAct Agent")
    print("=" * 60)
    print(f"Run: {run_name}")
    print("\nPress Ctrl+C to exit\n")

    # Load configuration
    config_data = load_config(run_name)
    base_prompt = load_prompts(config_data, run_name)

    # Load compaction prompt if enabled
    compaction_prompt = None
    compaction_cfg = config_data.get("compaction", {})
    if compaction_cfg.get("enabled", False):
        compaction_prompt = load_compaction_prompt(run_name)
        # Validate config
        trigger = compaction_cfg.get("trigger_messages", 60)
        keep = compaction_cfg.get("keep_recent", 20)
        if trigger <= keep:
            print(f"Warning: compaction.trigger_messages ({trigger}) should be > keep_recent ({keep})")
        print(f"Compaction enabled (trigger: {trigger}, keep: {keep})")

    # Load truncation config if enabled
    truncation_cfg = config_data.get("truncation", {})
    truncation_enabled = truncation_cfg.get("enabled", False)
    if truncation_enabled:
        if compaction_cfg.get("enabled", False):
            raise ValueError(
                "Cannot enable both compaction and truncation. "
                "Choose one history management strategy."
            )
        truncation_keep = truncation_cfg.get("keep_recent", 40)
        if truncation_keep < 4:
            raise ValueError(f"truncation.keep_recent must be >= 4, got {truncation_keep}")
        print(f"Truncation enabled (keep: {truncation_keep})")

    # Setup persistence
    db_path = Path(run_name) / "checkpoint.sqlite"
    tool_names = config_data.get("tools", ["send_message", "think"])

    # Initialize memory system if any memory tools are configured
    memory_tools = None
    memory_tool_names = {"memory_list", "memory_read", "memory_write", "memory_update", "memory_search", "memory_delete"}
    if any(name in memory_tool_names for name in tool_names):
        print("Initializing memory system...")
        from .memory import MemoryStore
        memory_store = MemoryStore(db_path)
        # Similarity advisor is optional
        memory_config = config_data.get("memory", {})
        similarity_config = memory_config.get("similarity", {})
        similarity_advisor = None
        if similarity_config.get("enabled", False):
            from .memory.similarity import SimilarityAdvisor
            similarity_advisor = SimilarityAdvisor(db_path, similarity_config)
        memory_tools = create_memory_tools(memory_store, similarity_advisor)

    # Initialize canvas system if needed
    canvas_tools = None
    canvas_tool_names = {"canvas_draw", "canvas_create", "canvas_read", "canvas_view", "canvas_list", "canvas_delete", "canvas_clear"}
    if any(name in canvas_tool_names for name in tool_names):
        print("Initializing canvas system...")
        img_dir = Path(run_name) / "img"
        canvas_tools = create_canvas_tools(db_path, img_dir)

    # Parse phenomenology config
    phenom_config = config_data.get("phenomenology", {})
    phenom_enabled = phenom_config.get("enabled", False)
    if phenom_enabled:
        # Get list of tools to wrap with phenomenology
        phenom_tool_list = phenom_config.get("tools", [])
        phenom_tools = set(phenom_tool_list) & set(tool_names)  # Only wrap tools that are actually configured
        if phenom_tools:
            print(f"Phenomenology enabled for: {', '.join(sorted(phenom_tools))}")
    else:
        phenom_tools = set()

    # Check for Tavily API key if web search tools are configured
    web_search_tools = {"web_search", "extract_content"}
    if any(name in web_search_tools for name in tool_names):
        if not os.getenv("TAVILY_API_KEY"):
            print("Warning: TAVILY_API_KEY not set. Web search tools will fail at runtime.")

    # Get tools (with phenomenology wrapping if configured)
    tools = get_tools_by_name(
        tool_names,
        memory_tools=memory_tools,
        canvas_tools=canvas_tools,
        phenom_tools=phenom_tools
    )

    # Build system prompt (with phenomenology docs if configured)
    tool_descriptions = get_tool_descriptions(tool_names, phenom_tools=phenom_tools)
    system_prompt = f"{base_prompt}{tool_descriptions}"

    # Get model config
    model_config = config_data.get("model", {})
    if "name" not in model_config:
        raise ValueError("config.json must specify model.name (e.g., 'anthropic/claude-sonnet-4.5')")
    max_tool_calls = config_data.get("max_tool_calls", 0)
    thread_id = run_path.name  # Use run folder name as thread_id

    # Get execution mode (unsegmented = default, segmented = discrete cycles)
    execution_mode = config_data.get("execution_mode", "unsegmented")
    is_segmented = execution_mode == "segmented"

    # Create LLM with appropriate tool_choice
    llm = create_llm(model_config)
    if is_segmented:
        # Segmented: agent can choose to respond without tools (ends cycle)
        llm_with_tools = llm.bind_tools(tools)
    else:
        # Unsegmented: agent must always call a tool
        llm_with_tools = llm.bind_tools(tools, tool_choice="any")

    # Setup persistence
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    checkpointer = SqliteSaver(conn)

    # Create graph based on execution mode
    if is_segmented:
        graph = create_segmented_graph(tools, checkpointer=checkpointer)
    else:
        graph = create_graph(tools, checkpointer=checkpointer)

    run_config = {
        "configurable": {"thread_id": thread_id, "llm": llm_with_tools},
        "recursion_limit": 200
    }

    # Helper to check and perform history management (truncation or compaction)
    def check_history_management(current_messages: list) -> bool:
        """Check and perform truncation or compaction if needed. Returns True if applied."""
        # Try truncation first (if enabled)
        if truncation_enabled and should_truncate(current_messages, config_data):
            try:
                removals = truncate_history(current_messages, config_data)
                if removals:
                    graph.update_state(run_config, {"messages": removals}, as_node="__start__")
                    return True
            except Exception as e:
                print(f"Warning: truncation failed: {e}")
                return False

        # Try compaction (if enabled)
        if not compaction_prompt:
            return False
        if not should_compact(current_messages, config_data):
            return False

        try:
            compacted = compact_history(
                current_messages,
                llm,
                compaction_prompt,
                config_data,
                run_name
            )
            graph.update_state(run_config, {"messages": compacted}, as_node="__start__")
            return True
        except Exception as e:
            print(f"Warning: compaction failed: {e}")
            return False

    # Check for existing session
    current_state = graph.get_state(run_config)
    if not current_state.values:
        if injected_query:
            print("Warning: --query ignored for new sessions (no history to query)")
        print("Starting new session...")
        inputs = {"messages": [SystemMessage(content=system_prompt)]}
        total_tool_calls = 0
    else:
        print("Resuming existing session...")
        # Check for compaction on resume
        messages = list(current_state.values.get("messages", []))
        if messages and check_history_management(messages):
            print("History managed on resume")

        if injected_query:
            # Inject researcher question with explicit framing
            # The agent needs clear instructions to answer via send_message
            wrapped_query = (
                f"RESEARCHER QUESTION: Please answer this question using your memory "
                f"and/or web search as needed. After formulating your answer, use "
                f"send_message to deliver it to the researcher.\n\n"
                f"Question: {injected_query}"
            )
            print(f"Injecting query: {injected_query[:60]}{'...' if len(injected_query) > 60 else ''}")
            inputs = {"messages": [HumanMessage(content=wrapped_query)]}
        else:
            inputs = None
        total_tool_calls = 0

    print(f"Model: {model_config['name']}")
    print(f"Tools: {', '.join(t.name for t in tools)}")
    print(f"Mode: {execution_mode}")
    print(f"Max tool calls: {max_tool_calls if max_tool_calls > 0 else 'unlimited'}")
    print(f"Thread ID: {thread_id}\n")

    try:
        if is_segmented:
            # Segmented mode: run cycles with wake messages between them
            cycle_num = 1
            while True:
                # Check compaction at start of each cycle (after first)
                if cycle_num > 1:
                    current_state = graph.get_state(run_config)
                    messages = list(current_state.values.get("messages", []))
                    if messages and check_history_management(messages):
                        print(f"[History managed at cycle {cycle_num}]")

                print(f"[Cycle {cycle_num}]")

                for event in graph.stream(inputs, run_config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if "messages" in node_output:
                            for msg in node_output["messages"]:
                                log_message(run_name, msg)
                                if isinstance(msg, ToolMessage):
                                    total_tool_calls += 1

                    if max_tool_calls > 0 and total_tool_calls >= max_tool_calls:
                        print(f"\nMax tool calls reached ({total_tool_calls}/{max_tool_calls})")
                        raise StopIteration

                # Cycle ended naturally (no tool calls) - inject wake message for next cycle
                cycle_num += 1
                wake_message = f"continue (cycle: {cycle_num})"
                inputs = {"messages": [HumanMessage(content=wake_message)]}
                print(f"[Cycle {cycle_num - 1} ended, sending wake message]")

        else:
            # Unsegmented mode: continuous loop (original behavior)
            pending_tool_name = None
            for event in graph.stream(inputs, run_config, stream_mode="updates"):
                is_tool_event = False
                for node_name, node_output in event.items():
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            log_message(run_name, msg)
                            if isinstance(msg, ToolMessage):
                                total_tool_calls += 1
                                is_tool_event = True
                            elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                                pending_tool_name = msg.tool_calls[-1]["name"]

                if max_tool_calls > 0 and total_tool_calls >= max_tool_calls:
                    print(f"\nMax tool calls reached ({total_tool_calls}/{max_tool_calls})")
                    break

                # Mid-stream truncation check (cheap, no LLM call)
                # Call truncation directly (not check_history_management) to avoid compaction path
                if truncation_enabled and is_tool_event:
                    current_state = graph.get_state(run_config)
                    cur_msgs = list(current_state.values.get("messages", []))
                    if cur_msgs and should_truncate(cur_msgs, config_data):
                        removals = truncate_history(cur_msgs, config_data)
                        if removals:
                            graph.update_state(run_config, {"messages": removals}, as_node="__start__")
                            print(f"[History truncated at tool call {total_tool_calls}]")

                if step_mode and is_tool_event:
                    if pending_tool_name != "send_message":
                        print(f"\n{'─' * 40} [{total_tool_calls}/{max_tool_calls}] {'─' * 40}")
                        input("Press Enter to continue...")
                    else:
                        print(f"{'─' * 40} [{total_tool_calls}/{max_tool_calls}] {'─' * 40}")
                    pending_tool_name = None

    except StopIteration:
        pass  # Clean exit from segmented mode max_tool_calls
    except StopSignal:
        print("\nAgent requested termination")
    except KeyboardInterrupt:
        print(f"\nAgent stopped by user. Completed {total_tool_calls} tool calls")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
