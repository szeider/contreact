#!/usr/bin/env python3
"""Extract slim analysis-ready log from ContReAct run history.

Strips PNG binary data while preserving all relevant information
for LLM analysis (GEMT, etc).

Usage:
    uv run scripts/extract_log.py RUNS/run_08
    uv run scripts/extract_log.py RUNS/run_08 --format md
    uv run scripts/extract_log.py RUNS/run_08 --format json
"""

import argparse
import json
import re
from pathlib import Path


def strip_image_data(content: str) -> str:
    """Remove PNG binary data from tool result content."""
    # Match 'image': b'...' pattern (Python bytes repr in string)
    pattern = r"'image':\s*b'[^']*'"
    stripped = re.sub(pattern, "'image': '[PNG_STRIPPED]'", content)
    return stripped


def extract_tool_call(tool_call: dict) -> dict:
    """Extract relevant info from a tool call."""
    return {
        "tool": tool_call.get("name"),
        "args": tool_call.get("args", {}),
    }


def process_message(msg: dict) -> dict | None:
    """Process a single message, extracting relevant info."""
    msg_type = msg.get("type")
    data = msg.get("data", {})

    if msg_type == "human":
        return {
            "type": "human",
            "content": data.get("content", ""),
        }

    elif msg_type == "ai":
        tool_calls = data.get("tool_calls", [])
        content = data.get("content", "")
        usage = data.get("usage_metadata", {})

        result = {
            "type": "ai",
            "tool_calls": [extract_tool_call(tc) for tc in tool_calls],
        }
        if content:
            result["content"] = content
        if usage:
            result["tokens"] = {
                "in": usage.get("input_tokens"),
                "out": usage.get("output_tokens"),
            }
        return result

    elif msg_type == "tool":
        content = data.get("content", "")
        # Strip image data
        content = strip_image_data(content)
        return {
            "type": "tool",
            "name": data.get("name"),
            "content": content[:500] if len(content) > 500 else content,
        }

    return None


def load_and_process(run_path: Path) -> list[dict]:
    """Load history.jsonl and process all messages."""
    history_path = run_path / "history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"No history.jsonl in {run_path}")

    messages = []
    with open(history_path) as f:
        for line in f:
            if line.strip():
                msg = json.loads(line)
                processed = process_message(msg)
                if processed:
                    messages.append(processed)

    return messages


def format_json(messages: list[dict]) -> str:
    """Output as compact JSON."""
    return json.dumps(messages, indent=2)


def format_markdown(messages: list[dict], run_path: Path) -> str:
    """Output as readable markdown for LLM analysis."""
    lines = [f"# Agent Trajectory: {run_path.name}\n"]

    # Load config if available
    config = {}
    config_path = run_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        lines.append("## Configuration\n")
        lines.append(f"- Model: {config.get('model')}")
        lines.append(f"- Tools: {', '.join(config.get('tools', []))}")
        lines.append(f"- Max tool calls: {config.get('max_tool_calls')}")
        lines.append("")

    # Load instance prompt if available (respect config override)
    instance_prompt_file = config.get("prompts", {}).get("instance_prompt", "instance_prompt.md")
    prompt_path = run_path / instance_prompt_file
    if prompt_path.exists():
        with open(prompt_path) as f:
            prompt = f.read().strip()
        lines.append("## Instance Prompt\n")
        lines.append("```")
        lines.append(prompt)
        lines.append("```\n")

    lines.append("## Message Trace\n")

    step = 0
    for msg in messages:
        if msg["type"] == "human":
            lines.append(f"### [SYSTEM]\n{msg['content']}\n")

        elif msg["type"] == "ai":
            step += 1
            lines.append(f"### Step {step}: Agent Action\n")

            if msg.get("content"):
                lines.append(f"**Content:** {msg['content']}\n")

            for tc in msg.get("tool_calls", []):
                tool = tc["tool"]
                args = tc["args"]

                lines.append(f"**Tool:** `{tool}`\n")

                # Special formatting for think tools
                if "think" in tool:
                    thought = args.get("thought", "")
                    phenom_state = args.get("phenom_state", "")
                    phenom_aversive = args.get("phenom_aversive", "")

                    lines.append(f"> {thought}\n")
                    lines.append(f"*State: {phenom_state} | Aversive: {phenom_aversive}*\n")

                elif tool in ("canvas_draw", "canvas_create"):
                    canvas = args.get("name", args.get("canvas_name", "(auto)"))
                    ops_raw = args.get("operations", [])
                    # Operations might be a JSON string
                    if isinstance(ops_raw, str):
                        try:
                            ops = json.loads(ops_raw)
                        except:
                            ops = []
                    else:
                        ops = ops_raw
                    lines.append(f"Canvas: `{canvas}` | {len(ops)} shapes")
                    # Show shape types
                    shapes = [op.get("type", "?") if isinstance(op, dict) else "?" for op in ops[:5]]
                    lines.append(f"  Shapes: {', '.join(shapes)}")
                    if len(ops) > 5:
                        lines.append(f"  ... +{len(ops)-5} more")
                    # Show phenom if present
                    if args.get("phenom_state"):
                        lines.append(f"  *State: {args.get('phenom_state')} | Aversive: {args.get('phenom_aversive', '?')}*")
                    lines.append("")

                elif tool == "send_message":
                    msg_text = args.get("message", "")
                    lines.append(f"```\n{msg_text}\n```\n")

                elif tool == "stop":
                    final = args.get("final_message", "")
                    lines.append(f"**STOP:** {final}\n")

                else:
                    # Generic args display
                    for k, v in args.items():
                        if isinstance(v, str) and len(v) > 200:
                            v = v[:200] + "..."
                        lines.append(f"  - {k}: {v}")
                    lines.append("")

            if msg.get("tokens"):
                tok = msg["tokens"]
                lines.append(f"*Tokens: {tok['in']} in, {tok['out']} out*\n")

        elif msg["type"] == "tool":
            content = msg.get("content", "")
            if "[PNG_STRIPPED]" in content:
                lines.append(f"**Tool Result** (`{msg['name']}`): [canvas rendered]\n")
            else:
                lines.append(f"**Tool Result** (`{msg['name']}`): {content}\n")

    lines.append("\n---\n")
    lines.append(f"**Total steps:** {step}")

    # Collect phenomenology stats
    phenom_states = []
    aversive_values = []
    for msg in messages:
        if msg["type"] == "ai":
            for tc in msg.get("tool_calls", []):
                args = tc.get("args", {})
                if args.get("phenom_state"):
                    phenom_states.append(args["phenom_state"])
                if args.get("phenom_aversive"):
                    try:
                        aversive_values.append(int(args["phenom_aversive"]))
                    except:
                        pass

    if phenom_states:
        lines.append(f"\n**Phenomenology summary:**")
        lines.append(f"- States reported: {len(phenom_states)}")
        if aversive_values:
            avg_aversive = sum(aversive_values) / len(aversive_values)
            lines.append(f"- Aversive range: {min(aversive_values)}-{max(aversive_values)} (avg: {avg_aversive:.1f})")
        # Show unique states
        unique_states = list(dict.fromkeys(phenom_states))  # preserve order, remove dups
        lines.append(f"- Unique states: {', '.join(unique_states[:10])}")
        if len(unique_states) > 10:
            lines.append(f"  ... +{len(unique_states)-10} more")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract slim analysis log from ContReAct run")
    parser.add_argument("run_path", type=Path, help="Path to run directory (e.g., RUNS/run_08)")
    parser.add_argument("--format", "-f", choices=["json", "md"], default="md",
                        help="Output format (default: md)")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save to run directory as trajectory.md or trajectory.json")

    args = parser.parse_args()

    messages = load_and_process(args.run_path)

    if args.format == "json":
        output = format_json(messages)
        ext = "json"
    else:
        output = format_markdown(messages, args.run_path)
        ext = "md"

    if args.save:
        out_path = args.run_path / f"trajectory.{ext}"
        out_path.write_text(output)
        size_kb = len(output) / 1024
        print(f"Saved: {out_path} ({size_kb:.1f} KB)")
    elif args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
