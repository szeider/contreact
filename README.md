# ContReAct

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![UV](https://img.shields.io/badge/Packaged%20with-UV-purple)](https://github.com/astral-sh/uv)
[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-green)](https://github.com/langchain-ai/langgraph)

A Python implementation of a continuous ReAct (Reasoning and Acting) agent for large language models. This project showcases and tests the ContReAct framework.

Unlike traditional ReAct agents that complete a task and stop, this agent keeps going---accumulating knowledge, exploring topics, and building persistent memory over time.

Two execution modes: **segmented** runs in discrete cycles with reflection and planning between each; **unsegmented** runs continuously with forced tool calls. Both modes persist state, so you can stop anytime and resume later.

## Installation

```bash
# Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/szeider/contreact.git
cd contreact
uv sync

# Set up API keys
cp .env.example .env
# Edit .env with your keys
```

You need:
- [OpenRouter](https://openrouter.ai) API key (for LLM access)
- [Tavily](https://tavily.com) API key (optional, for web search tools)

## Quick start

```bash
# Copy the quickstart example
cp -r examples/quickstart my_run

# Start the agent
uv run contreact my_run

# The agent will think through the task and send you messages
# Press Ctrl+C to stop anytime (state is saved to checkpoint)

# Resume later - continues exactly where it left off
uv run contreact my_run
```

## How it works

**Segmented mode** runs in cycles:

1. **Reflect** --- Review previous cycle, check what's in memory
2. **Plan** --- Decide what to explore next
3. **Execute** --- Search web, write to memory
4. **Repeat**

Between cycles, the agent sends itself a message containing its plan for the next iteration. This self-feedback keeps it on track without external direction.

**Unsegmented mode** forces tool calls continuously without natural pauses. The agent must call a tool on every turn.

Both modes persist state. Stop anytime; it resumes where it left off.

## Configuration

Each run folder needs two files:

```
my_run/
├── config.json          # Model, tools, execution settings
└── instance_prompt.md   # Task instructions
```

Optional: Add `compaction_prompt.md` if you enable history compaction for long-running sessions.

See `examples/quickstart/` for a minimal setup, or `examples/carbon_capture/` for a full-featured example with memory and web search.

### config.json

```json
{
  "model": {
    "name": "anthropic/claude-sonnet-4.5",
    "temperature": 0.0
  },
  "tools": ["think", "send_message", "memory_write", "memory_read",
            "memory_list", "memory_search", "web_search"],
  "execution_mode": "segmented",
  "max_tool_calls": 100
}
```

### Execution modes

**Unsegmented** (default): Agent runs continuously with forced tool calls. Simple and direct.

**Segmented**: Agent completes a cycle, pauses, receives a wake message, starts next cycle. Provides natural checkpoints for monitoring long-running research tasks.

## Tools

### Basic

| Tool | Description |
|------|-------------|
| `think` | Internal reasoning, logged but not sent anywhere |
| `send_message` | Send message to operator, wait for response |
| `stop` | Terminate the agent |

### Memory

| Tool | Description |
|------|-------------|
| `memory_write` | Store or update information with a key (upsert) |
| `memory_read` | Retrieve stored information |
| `memory_list` | List all memory keys |
| `memory_search` | Search memories by content |
| `memory_update` | Alias for memory_write (both create or update) |
| `memory_delete` | Remove a memory |

Both `memory_write` and `memory_update` behave identically---they create a new memory or update an existing one.

### Web search

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via Tavily |
| `extract_content` | Extract text from a URL (often fails due to site restrictions) |

**Note**: `extract_content` may fail on many sites due to JavaScript rendering, paywalls, or anti-bot measures. The agent should prefer using `web_search` results directly when possible.

### Canvas

| Tool | Description |
|------|-------------|
| `canvas_draw` | Draw shapes on a named canvas (JSON operations) |
| `canvas_view` | Render canvas to PNG and view it |
| `canvas_read` | Get raw operations from a canvas |
| `canvas_list` | List all canvases |
| `canvas_clear` | Clear a canvas |
| `canvas_delete` | Delete a canvas |

## Asking questions

After the agent has collected data, ask questions:

```bash
# Agent answers from memory or does additional research
uv run contreact my_run -q "What is the cost per ton for DAC?"

# Read-only query (no tool access, doesn't modify state)
uv run query my_run "Summarize what you've learned"
```

**How `-q` works**: The query is wrapped with instructions telling the agent to use its memory and tools, then deliver the answer via `send_message`. After answering, the agent continues running (use Ctrl+C to stop). The query becomes part of the session history.

**How `query` works**: Sends your question directly to the LLM with the full conversation history but no tool access. Read-only---doesn't modify state. Good for summaries and analysis.

## Project structure

```
src/contreact/
├── main.py          # Entry point
├── graph.py         # LangGraph state machine
├── compaction.py    # History summarization
├── memory/          # Memory storage + similarity
└── tools/           # Tool implementations
```

## Caveats

- **Thread ID is the folder name**: The checkpoint database uses the run folder name as thread ID. If you rename or move the folder, resumption will start a new session instead of continuing the old one.
- **Minimal terminal output**: The agent prints abbreviated status messages. For detailed history, see `history.jsonl` in the run folder.
- **Token usage**: Usage metadata (input/output tokens) is logged to `history.jsonl` but not displayed in the terminal. Parse the log file to calculate costs.

## License

Apache 2.0
