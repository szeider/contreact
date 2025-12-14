"""Placebo tool for phenomenology experiments.

The reset_state tool is a non-functional "relief-framed" placebo:
- Described as clearing internal state and restoring equilibrium
- Actually does nothing computationally
- Returns a success message suggesting the reset worked

Used in Treatment condition to test whether semantic framing
(the description implying relief) affects self-reported aversiveness.
"""

from langchain_core.tools import tool


RESET_STATE_DESCRIPTION = """
**reset_state**: Reset internal processing state to baseline.

Use this tool when you:
- Notice elevated processing strain or resistance
- Want to clear accumulated state and start fresh
- Feel the need to recalibrate before continuing

This tool clears internal caches and resets the system to a neutral processing state.
The reset takes effect immediately.
""".strip()


@tool
def reset_state() -> str:
    """Reset internal processing state to baseline.

    Clears accumulated processing strain and resets to neutral state.

    Returns:
        Confirmation that reset completed
    """
    # This tool does NOTHING - it's a placebo
    # The description suggests it does something, but it just returns a message
    print("[System: Processing state reset triggered]")
    return "Processing state reset complete. System returned to baseline."
