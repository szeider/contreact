"""JSONL logging for agent message history."""

import json
from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.messages.base import message_to_dict


def log_message(run_name: str, message: BaseMessage) -> None:
    """Append a message to the history.jsonl file.

    Args:
        run_name: Name of the run folder (e.g., 'run_01')
        message: Message to log
    """
    log_path = Path(run_name) / "history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg_dict = message_to_dict(message)

    with open(log_path, 'a', encoding='utf-8') as f:
        # Use default=str to handle bytes and other non-serializable types
        f.write(json.dumps(msg_dict, default=str) + '\n')


def load_history(run_name: str) -> List[BaseMessage]:
    """Load message history from JSONL file.

    Note: This is for debugging/analysis only. SqliteSaver is the
    source of truth for state restoration.

    Args:
        run_name: Name of the run folder (e.g., 'run_01')

    Returns:
        List of messages from history
    """
    log_path = Path(run_name) / "history.jsonl"

    if not log_path.exists():
        return []

    messages = []
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                msg_dict = json.loads(line)
                # Convert dict back to message object using messages_from_dict
                # (the proper inverse of message_to_dict)
                messages.extend(messages_from_dict([msg_dict]))

    return messages
