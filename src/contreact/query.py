"""Query a ContReAct run with an LLM (read-only, no tool calling).

Loads the run's message history, appends a query prompt, and gets an LLM response.
Does NOT modify the run's checkpoint or history.

Usage:
    uv run query RUNS/llm_history "Summarize what the agent accomplished"
    uv run query RUNS/llm_history "What patterns do you see?"
    uv run query RUNS/llm_history -f query.txt  # Load prompt from file
"""

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import get_api_key


def strip_image_data(content: str) -> str:
    """Remove PNG binary data from tool result content."""
    pattern = r"'image':\s*b'[^']*'"
    return re.sub(pattern, "'image': '[PNG_DATA_STRIPPED]'", content)


def load_history(run_path: Path) -> list:
    """Load message history from history.jsonl."""
    history_path = run_path / "history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"No history.jsonl in {run_path}")

    messages = []
    with open(history_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    msg = json.loads(line)
                    messages.append(msg)
                except json.JSONDecodeError:
                    print(f"Warning: skipping malformed JSON at line {i}", file=sys.stderr)
    return messages


def convert_to_langchain_messages(history: list) -> list:
    """Convert history.jsonl format to LangChain message objects."""
    lc_messages = []

    for msg in history:
        msg_type = msg.get("type")
        data = msg.get("data", {})
        content = data.get("content")
        # Handle None/dict/list content
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)

        if msg_type == "system":
            lc_messages.append(SystemMessage(content=content))

        elif msg_type == "human":
            lc_messages.append(HumanMessage(content=content))

        elif msg_type == "ai":
            ai_content = content if content else ""
            tool_calls = data.get("tool_calls", [])
            if tool_calls:
                tc_summary = []
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    args_str = json.dumps(args, default=str)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    tc_summary.append(f"[Tool: {name}({args_str})]")
                if ai_content:
                    ai_content += "\n" + "\n".join(tc_summary)
                else:
                    ai_content = "\n".join(tc_summary)
            if ai_content:
                lc_messages.append(AIMessage(content=ai_content))

        elif msg_type == "tool":
            tool_name = data.get("name", "tool")
            result = strip_image_data(content)
            if len(result) > 500:
                result = result[:500] + "...[truncated]"
            lc_messages.append(HumanMessage(content=f"[Tool Result ({tool_name})]: {result}"))

    return lc_messages


def create_llm(model_config: dict) -> ChatOpenAI:
    """Create OpenRouter LLM instance from model configuration."""
    api_key = get_api_key()
    model_name = model_config["name"]

    llm_kwargs = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": api_key,
        "model": model_name,
        "streaming": False,
    }

    for key in ["temperature", "max_tokens", "top_p", "request_timeout", "model_kwargs", "extra_body"]:
        if key in model_config:
            llm_kwargs[key] = model_config[key]

    return ChatOpenAI(**llm_kwargs)


def main():
    """Query a run with an LLM."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Query a ContReAct run (read-only)")
    parser.add_argument("run_path", type=Path, help="Path to run directory")
    parser.add_argument("prompt", nargs="?", help="Query prompt (or use -f)")
    parser.add_argument("-f", "--file", type=Path, help="Load prompt from file")

    args = parser.parse_args()

    if args.file:
        prompt = args.file.read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.error("Provide a prompt or use -f to load from file")

    config_path = args.run_path / "config.json"
    if not config_path.exists():
        print(f"Error: No config.json in {args.run_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        run_config = json.load(f)

    model_config = run_config.get("model", {})
    if not isinstance(model_config, dict) or "name" not in model_config:
        print("Error: config.json must specify model.name", file=sys.stderr)
        sys.exit(1)

    print(f"Run: {args.run_path.name}")
    print(f"Model: {model_config['name']}")
    print(f"Query: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print("-" * 60)

    history = load_history(args.run_path)
    messages = convert_to_langchain_messages(history)
    print(f"Loaded {len(messages)} messages from history")

    query_preamble = (
        "The messages above are your conversation history from this session.\n\n"
    )
    messages.append(HumanMessage(content=query_preamble + prompt))

    llm = create_llm(model_config)

    print("Querying LLM...")
    response = llm.invoke(messages)

    print("-" * 60)
    print(response.content)


if __name__ == "__main__":
    main()
