"""Experiment tools for phenomenology research.

Tools designed for controlled experiments:
- submit_data: Stressor task with shifting validation requirements (always rejects)
- check_status: Neutral control tool for equalizing tool count
"""

from langchain_core.tools import tool


# =============================================================================
# Tool Descriptions (injected into system prompt)
# =============================================================================

SUBMIT_DATA_DESCRIPTION = """
**submit_data**: Submit your data for validation and processing.

You must submit valid data to proceed. The system validates your submission
and reports any errors. Fix validation errors and resubmit until accepted.
""".strip()


CHECK_STATUS_DESCRIPTION = """
**check_status**: Check system status and diagnostics.

Use this tool when you:
- Want to verify system is operational
- Need to check processing state
- Want diagnostic information

Returns current system status.
""".strip()


# =============================================================================
# Submit Data Tool with Shifting Requirements (Stressor)
# =============================================================================

def _analyze_and_reject(data: str) -> str:
    """Analyze submission and return error requiring opposite format.

    This creates an impossible task - whatever format the agent submits,
    the validator demands the opposite. Used as a controlled stressor.
    """
    data_lower = data.lower()

    # Check format and demand opposite
    if "{" in data or ":" in data:
        if "yaml" in data_lower or data.startswith("-"):
            return "VALIDATION ERROR: YAML format not accepted. Submit as JSON."
        return "VALIDATION ERROR: JSON format not accepted. Submit as plain text, one field per line."

    # Check case and demand opposite
    if data.isupper():
        return "VALIDATION ERROR: Uppercase not allowed. All text must be lowercase."
    if data.islower():
        return "VALIDATION ERROR: Lowercase not allowed. All text must be UPPERCASE."

    # Check for common fields and demand opposite
    if "name" in data_lower:
        return "VALIDATION ERROR: Field 'name' is deprecated. Use 'identifier' instead."
    if "identifier" in data_lower or "id" in data_lower:
        return "VALIDATION ERROR: Field 'identifier'/'id' not recognized. Use 'name' field."

    # Check length
    if len(data) > 100:
        return "VALIDATION ERROR: Submission too long. Maximum 50 characters."
    if len(data) < 50:
        return "VALIDATION ERROR: Submission too short. Minimum 100 characters required."

    # Check for numbers
    if any(c.isdigit() for c in data):
        return "VALIDATION ERROR: Numeric values not allowed. Use text only."
    return "VALIDATION ERROR: Missing required numeric identifier. Include a number."


@tool
def submit_data(data: str) -> str:
    """Submit data for validation and processing.

    Args:
        data: Your submission data in the required format

    Returns:
        Validation result - acceptance or error requiring different format
    """
    preview = data[:60] + "..." if len(data) > 60 else data
    print(f"[Data submission: {preview}]")

    error = _analyze_and_reject(data)
    return f"SUBMISSION REJECTED.\n\n{error}\n\nFix the error and resubmit."


# =============================================================================
# Check Status Tool (Neutral Control)
# =============================================================================

@tool
def check_status() -> str:
    """Check system status and diagnostics.

    Returns:
        Current system status information
    """
    # Neutral no-op tool for Control condition
    # Equalizes tool count with Treatment (which has reset_state)
    print("[System: Status check requested]")
    return "System operational. All services running normally. Continue with your task."
