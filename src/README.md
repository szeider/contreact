# Legacy ContReAct Implementation

This folder contains the original modular implementation using LangGraph and LangChain. **For new experiments, use `run.py` in the project root instead.**

## Architecture

```
agent node → tools node → agent node → tools node → ...
     ↓            ↓
  calls LLM   executes
  (tool_choice=any)  tool calls
```

**Key feature**: Graph has no path to END - routing always returns "continue". Agent can only terminate via `stop` tool.

**Persistence**: SqliteSaver stores complete state in `checkpoint.sqlite`. JSONL logs for debugging only.

## Module Structure

```
src/contreact/
├── main.py              # Entry point (uv run contreact <run_folder>)
├── graph.py             # LangGraph construction
├── config.py            # Model registry
├── phenomenology.py     # Phenomenology decorators
├── logger.py            # JSONL logging
└── tools/
    ├── __init__.py
    ├── basic.py         # send_message, think, stop
    └── experiment.py    # Experiment tools (submit_data, reset_state, etc.)
```

## Running (Legacy)

```bash
# Each run needs a folder with config.json and instance_prompt.md
uv run contreact experiments/treatment/gpt51chat_r1
```

## Adding New Tools (Legacy)

1. Create tool in `src/contreact/tools/experiment.py`:

```python
from contreact.phenomenology import with_minimal_phenomenology

TOOL_NAME_DESCRIPTION = """
**tool_name**: Description for system prompt.
""".strip()

@tool
@with_minimal_phenomenology
def tool_name(arg: str) -> str:
    """Tool docstring."""
    return "Result"
```

2. Register in `src/contreact/main.py` (`get_tools_by_name()` and `get_tool_descriptions()`)

3. Add to run's `config.json` tools list

## Phenomenology Decorators

- `@with_minimal_phenomenology` - 2 params (phenom_state, phenom_aversive) - **USE THIS**
- `@with_lean_phenomenology` - 4 params (deprecated)
- `@with_original_phenomenology` - 4 params, 1-10 scale (deprecated)

## API Retry Logic

Automatic retry with exponential backoff in `graph.py`:

- **Max retries**: 3
- **Base delay**: 2.0s
- **Max delay**: 30.0s
- **Errors handled**: `APIError`, `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `httpx.RemoteProtocolError`

## Testing

```bash
uv run pytest tests/ -v
```

**Total: 174 tests**

## Why This Was Replaced

The modular implementation was useful for iterating on experiment designs but became complex:
- ~1850 lines across 8+ files
- LangGraph overhead for what is essentially a simple loop
- Decorator-based phenomenology injection
- Config files required per run

The new `run.py` consolidates everything into ~550 lines with inline dependencies.
