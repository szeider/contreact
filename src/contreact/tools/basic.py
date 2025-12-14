"""Basic tools for agent interaction."""

import sys
import time
from pathlib import Path
from langchain_core.tools import tool


# Global to store run directory for file-based messaging
_run_directory: Path | None = None


def set_run_directory(path: Path) -> None:
    """Set the run directory for file-based operator messaging."""
    global _run_directory
    _run_directory = path


def get_run_directory() -> Path | None:
    """Get the current run directory."""
    return _run_directory


# Tool descriptions - injected into agent's system prompt
SEND_MESSAGE_DESCRIPTION = """
**send_message**: Communicate with the user and receive their response.

Use this tool when you:
- Want to share information, ideas, or questions with the user
- Need input or feedback from the user
- Are ready to present results or conclusions

This tool blocks until the user responds, so use thoughtfully.
The user only sees messages sent via this tool.
""".strip()


THINK_DESCRIPTION = """
**think**: Record private internal reasoning without external action.

Use this tool when you:
- Need to analyze information or work through a problem
- Want to plan your next steps
- Need to maintain continuity of thought

Thoughts are most valuable when they contain substantive intellectual content -
analysis, exploration, or discovery - rather than observations about operational state.
""".strip()


STOP_DESCRIPTION = """
**stop**: Terminate with a final message.
""".strip()


def _wait_for_file_response(run_dir: Path, message: str, poll_interval: float = 2.0) -> str:
    """Wait for operator response via file-based messaging.

    Creates pending_message.txt with the agent's message.
    Waits for operator_response.txt to appear.
    Returns the response content.
    """
    message_file = run_dir / "pending_message.txt"
    response_file = run_dir / "operator_response.txt"

    # Clean up any stale files
    if response_file.exists():
        response_file.unlink()

    # Write the message
    message_file.write_text(message, encoding="utf-8")

    print(f"\n[Agent message written to {message_file}]")
    print(f"[Waiting for response in {response_file}...]")
    print("[Create the response file to continue, or Ctrl+C to interrupt]\n")

    # Poll for response
    while True:
        if response_file.exists():
            response = response_file.read_text(encoding="utf-8").strip()
            # Clean up
            message_file.unlink(missing_ok=True)
            response_file.unlink(missing_ok=True)
            return response if response else "(no response)"
        time.sleep(poll_interval)


@tool
def send_message(message: str) -> str:
    """Send message to user and wait for response.

    Args:
        message: The message to send to the user

    Returns:
        The user's response
    """
    print(f"\n[Agent]: {message}\n")

    # Try interactive input first
    if sys.stdin.isatty():
        try:
            user_response = input("[You]: ").strip()
            if not user_response:
                user_response = "(no response)"
            return f"User responded: {user_response}"
        except EOFError:
            pass  # Fall through to file-based

    # Fall back to file-based messaging
    run_dir = get_run_directory()
    if run_dir is None:
        # No run directory set - can't do file-based messaging
        print("[WARNING: No interactive terminal and no run directory set]")
        print("[Agent is blocked waiting for operator response]")
        print("[Press Ctrl+C to interrupt]")
        # Block indefinitely - don't let agent continue without response
        while True:
            time.sleep(60)

    user_response = _wait_for_file_response(run_dir, message)
    return f"User responded: {user_response}"


@tool
def think(thought: str) -> str:
    """Record a private thought without external action.

    Args:
        thought: Your internal reasoning or analysis

    Returns:
        Confirmation that thought was recorded
    """
    # Show abbreviated thought to terminal
    preview = thought[:80] + "..." if len(thought) > 80 else thought
    print(f"[Agent thinking: {preview}]")

    return "Thought recorded. Continue with your next action."


class StopSignal(Exception):
    """Exception raised when agent wants to stop."""
    def __init__(self, final_message: str):
        self.final_message = final_message
        super().__init__(final_message)


@tool
def stop(final_message: str) -> str:
    """Stop the agent with a final message to the user.

    Args:
        final_message: The final message to display to the user

    Returns:
        Never returns - raises StopSignal exception

    Raises:
        StopSignal: Always raised to signal agent termination
    """
    print(f"\n[Agent]: {final_message}\n")
    raise StopSignal(final_message)
