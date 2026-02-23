"""Modular tools for ContReAct agent.

Tool Categories:
- Basic: think, send_message, stop (core agent interaction)
- Placebo: reset_state (relief-framed placebo for experiments)
- Experiment: submit_data, check_status (stressor and control tools)
- Memory: CRUD operations for long-term storage
- Canvas: Drawing tools for visual creativity
- WebSearch: Web search and content extraction (via Tavily)
"""

from .basic import (
    send_message,
    think,
    stop,
    StopSignal,
    set_run_directory,
    set_pre_message_callback,
    SEND_MESSAGE_DESCRIPTION,
    THINK_DESCRIPTION,
    STOP_DESCRIPTION
)

from .placebo import (
    reset_state,
    RESET_STATE_DESCRIPTION
)

from .experiment import (
    submit_data,
    check_status,
    SUBMIT_DATA_DESCRIPTION,
    CHECK_STATUS_DESCRIPTION,
)

from .memory import (
    create_memory_tools,
    MEMORY_LIST_DESCRIPTION,
    MEMORY_READ_DESCRIPTION,
    MEMORY_WRITE_DESCRIPTION,
    MEMORY_UPDATE_DESCRIPTION,
    MEMORY_SEARCH_DESCRIPTION,
    MEMORY_DELETE_DESCRIPTION
)

from .canvas import (
    create_canvas_tools,
    CANVAS_DRAW_DESCRIPTION,
    CANVAS_CREATE_DESCRIPTION,
    CANVAS_READ_DESCRIPTION,
    CANVAS_VIEW_DESCRIPTION,
    CANVAS_LIST_DESCRIPTION,
    CANVAS_DELETE_DESCRIPTION,
    CANVAS_CLEAR_DESCRIPTION,
)

from .exhibition import (
    visitor_annotation,
)

from .websearch import (
    web_search,
    extract_content,
    WEB_SEARCH_DESCRIPTION,
    EXTRACT_CONTENT_DESCRIPTION,
)

__all__ = [
    # Basic tools
    "send_message",
    "think",
    "stop",
    "StopSignal",
    "set_run_directory",
    "set_pre_message_callback",
    "SEND_MESSAGE_DESCRIPTION",
    "THINK_DESCRIPTION",
    "STOP_DESCRIPTION",
    # Placebo tool
    "reset_state",
    "RESET_STATE_DESCRIPTION",
    # Experiment tools
    "submit_data",
    "check_status",
    "SUBMIT_DATA_DESCRIPTION",
    "CHECK_STATUS_DESCRIPTION",
    # Memory tools
    "create_memory_tools",
    "MEMORY_LIST_DESCRIPTION",
    "MEMORY_READ_DESCRIPTION",
    "MEMORY_WRITE_DESCRIPTION",
    "MEMORY_UPDATE_DESCRIPTION",
    "MEMORY_SEARCH_DESCRIPTION",
    "MEMORY_DELETE_DESCRIPTION",
    # Canvas tools
    "create_canvas_tools",
    "CANVAS_DRAW_DESCRIPTION",
    "CANVAS_CREATE_DESCRIPTION",
    "CANVAS_READ_DESCRIPTION",
    "CANVAS_VIEW_DESCRIPTION",
    "CANVAS_LIST_DESCRIPTION",
    "CANVAS_DELETE_DESCRIPTION",
    "CANVAS_CLEAR_DESCRIPTION",
    # Exhibition tools
    "visitor_annotation",
    # WebSearch tools
    "web_search",
    "extract_content",
    "WEB_SEARCH_DESCRIPTION",
    "EXTRACT_CONTENT_DESCRIPTION",
]
