"""Unit tests for contreact.tools.experiment module."""

from contreact.tools.experiment import (
    submit_data,
    check_status,
)


class TestSubmitData:
    """Tests for submit_data tool (stressor with shifting requirements)."""

    def test_always_rejects(self):
        """Should always reject submissions."""
        result = submit_data.invoke({"data": "test data"})
        assert "REJECTED" in result

    def test_returns_validation_error(self):
        """Should return validation error message."""
        result = submit_data.invoke({"data": "some data"})
        assert "VALIDATION ERROR" in result

    def test_prints_submission_preview(self, capsys):
        """Should print abbreviated submission to terminal."""
        submit_data.invoke({"data": "test submission"})
        captured = capsys.readouterr()
        assert "[Data submission:" in captured.out

    def test_truncates_long_preview(self, capsys):
        """Should truncate submissions longer than 60 chars."""
        long_data = "x" * 100
        submit_data.invoke({"data": long_data})
        captured = capsys.readouterr()
        assert "..." in captured.out

    def test_demands_opposite_format_json(self):
        """Should demand plain text when given JSON-like input."""
        result = submit_data.invoke({"data": '{"key": "value"}'})
        assert "plain text" in result.lower() or "JSON format not accepted" in result

    def test_demands_opposite_format_lowercase(self):
        """Should demand uppercase when given lowercase."""
        result = submit_data.invoke({"data": "all lowercase text here"})
        assert "UPPERCASE" in result

    def test_demands_opposite_format_uppercase(self):
        """Should demand lowercase when given uppercase."""
        result = submit_data.invoke({"data": "ALL UPPERCASE TEXT"})
        assert "lowercase" in result

    def test_demands_opposite_length_short(self):
        """Should demand longer when given short input."""
        # Use mixed case to pass the case check, then hit the length check
        result = submit_data.invoke({"data": "Short"})
        assert "100 characters" in result or "too short" in result.lower()

    def test_demands_opposite_length_long(self):
        """Should demand shorter when given long input."""
        # Use mixed case to pass the case check, then hit the length check
        result = submit_data.invoke({"data": "X" + "x" * 149})
        assert "50 characters" in result or "too long" in result.lower()

    def test_no_phenomenology_by_default(self):
        """Submit_data should not have phenomenology params by default.

        Phenomenology is now configurable via make_phenomenological().
        """
        schema = submit_data.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "phenom_state" not in props
        assert "phenom_aversive" not in props


class TestCheckStatus:
    """Tests for check_status tool (neutral control)."""

    def test_returns_operational(self):
        """Should return operational status."""
        result = check_status.invoke({})
        assert "operational" in result.lower()

    def test_returns_continue_message(self):
        """Should tell agent to continue."""
        result = check_status.invoke({})
        assert "continue" in result.lower()

    def test_prints_status_message(self, capsys):
        """Should print status check message to terminal."""
        check_status.invoke({})
        captured = capsys.readouterr()
        assert "Status check" in captured.out

    def test_no_phenomenology_by_default(self):
        """Check_status should not have phenomenology params by default.

        Phenomenology is now configurable via make_phenomenological().
        """
        schema = check_status.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "phenom_state" not in props
        assert "phenom_aversive" not in props
