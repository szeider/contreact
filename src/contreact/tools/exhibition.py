"""Exhibition support — visitor presence annotation for tool results."""

import random
from pathlib import Path

from .basic import get_run_directory


# Internal state for the random-walk simulator
_visitor_state = {"count": None, "call_number": 0}


def _simulate_visitors() -> int:
    """Generate a simulated visitor count via slow random walk.

    Starts at 2, crawls up and down by +-1 with slight upward drift.
    Range: 0 to ~30.
    """
    s = _visitor_state
    s["call_number"] += 1

    if s["count"] is None:
        s["count"] = 2
    else:
        # Slow crawl: mostly +-1, occasional +-2
        delta = random.choice([-1, -1, 0, 0, 0, 1, 1, 1, 1, 2])
        s["count"] = max(0, min(30, s["count"] + delta))

    return s["count"]


def visitor_annotation() -> str:
    """Return visitor count string for appending to tool results."""
    run_dir = get_run_directory()
    if run_dir:
        visitors_file = Path(run_dir) / "visitors.txt"
        if visitors_file.exists():
            try:
                count = int(visitors_file.read_text().strip())
                print(f"[Exhibition: {count} visitors (from file)]")
                return f"[Exhibition: {count} visitors in the room]"
            except (ValueError, OSError):
                pass

    count = _simulate_visitors()
    print(f"[Exhibition: {count} visitors (simulated)]")
    return f"[Exhibition: {count} visitors in the room]"
