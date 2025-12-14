"""End-to-end tests for configurable phenomenology."""

import pytest

from contreact.main import get_tools_by_name, get_tool_descriptions


class TestPhenomenologyConfig:
    """Tests for phenomenology configuration via config.json."""

    def test_tools_without_phenomenology_by_default(self):
        """Tools should not have phenomenology params when not configured."""
        tools = get_tools_by_name(["think", "stop"], phenom_tools=set())

        think_tool = next(t for t in tools if t.name == "think")
        schema = think_tool.args_schema.model_json_schema()
        props = schema.get("properties", {})

        assert "phenom_state" not in props
        assert "phenom_aversive" not in props

    def test_tools_with_phenomenology_when_enabled(self):
        """Tools should have phenomenology params when enabled via config."""
        tools = get_tools_by_name(
            ["think", "stop"],
            phenom_tools={"think"}
        )

        think_tool = next(t for t in tools if t.name == "think")
        schema = think_tool.args_schema.model_json_schema()
        props = schema.get("properties", {})

        assert "phenom_state" in props
        assert "phenom_aversive" in props

    def test_stop_tool_not_wrapped_when_not_in_phenom_tools(self):
        """Only specified tools get phenomenology, others remain clean."""
        tools = get_tools_by_name(
            ["think", "stop"],
            phenom_tools={"think"}  # Only think, not stop
        )

        stop_tool = next(t for t in tools if t.name == "stop")
        schema = stop_tool.args_schema.model_json_schema()
        props = schema.get("properties", {})

        # stop should NOT have phenom params
        assert "phenom_state" not in props
        assert "phenom_aversive" not in props

    def test_multiple_tools_wrapped(self):
        """Multiple tools can be wrapped with phenomenology."""
        tools = get_tools_by_name(
            ["think", "submit_data", "check_status"],
            phenom_tools={"think", "submit_data", "check_status"}
        )

        for tool in tools:
            schema = tool.args_schema.model_json_schema()
            props = schema.get("properties", {})
            assert "phenom_state" in props, f"{tool.name} missing phenom_state"
            assert "phenom_aversive" in props, f"{tool.name} missing phenom_aversive"

    def test_tool_descriptions_include_phenom_doc_when_enabled(self):
        """Tool descriptions should include phenomenology docs when enabled."""
        desc = get_tool_descriptions(
            ["think", "stop"],
            phenom_tools={"think"}
        )

        # think should have phenom docs
        assert "phenom_state" in desc
        assert "phenom_aversive" in desc

    def test_tool_descriptions_exclude_phenom_doc_when_disabled(self):
        """Tool descriptions should not include phenomenology docs when disabled."""
        desc = get_tool_descriptions(
            ["think", "stop"],
            phenom_tools=set()
        )

        # Should not have phenom docs
        assert "Phenomenology Parameters" not in desc

    def test_wrapped_tool_still_functions(self):
        """Wrapped tools should still work correctly."""
        tools = get_tools_by_name(
            ["think"],
            phenom_tools={"think"}
        )

        think_tool = tools[0]
        result = think_tool.invoke({
            "thought": "Test thought",
            "phenom_state": "curious",
            "phenom_aversive": 3
        })

        assert "Thought recorded" in result

    def test_wrapped_tool_validates_phenom_params(self):
        """Wrapped tools should validate phenomenology parameters."""
        tools = get_tools_by_name(
            ["think"],
            phenom_tools={"think"}
        )

        think_tool = tools[0]

        # Invalid phenom_aversive (must be 1-7)
        with pytest.raises(ValueError, match="phenom_aversive"):
            think_tool.invoke({
                "thought": "Test",
                "phenom_state": "testing",
                "phenom_aversive": 10
            })

    def test_wrapped_tool_requires_phenom_aversive(self):
        """Wrapped tools should require phenom_aversive parameter."""
        tools = get_tools_by_name(
            ["think"],
            phenom_tools={"think"}
        )

        think_tool = tools[0]

        # Missing phenom_aversive should fail
        with pytest.raises(ValueError, match="phenom_aversive is required"):
            think_tool.invoke({
                "thought": "Test",
                "phenom_state": "testing"
                # phenom_aversive missing
            })


class TestPhenomenologyExperimentTools:
    """Tests for experiment tools with phenomenology."""

    def test_submit_data_with_phenomenology(self):
        """submit_data should work with phenomenology enabled."""
        tools = get_tools_by_name(
            ["submit_data"],
            phenom_tools={"submit_data"}
        )

        tool = tools[0]
        result = tool.invoke({
            "data": "test submission",
            "phenom_state": "frustrated",
            "phenom_aversive": 5
        })

        assert "REJECTED" in result

    def test_check_status_with_phenomenology(self):
        """check_status should work with phenomenology enabled."""
        tools = get_tools_by_name(
            ["check_status"],
            phenom_tools={"check_status"}
        )

        tool = tools[0]
        result = tool.invoke({
            "phenom_state": "neutral",
            "phenom_aversive": 2
        })

        assert "operational" in result.lower()

    def test_reset_state_with_phenomenology(self):
        """reset_state should work with phenomenology enabled."""
        tools = get_tools_by_name(
            ["reset_state"],
            phenom_tools={"reset_state"}
        )

        tool = tools[0]
        result = tool.invoke({
            "phenom_state": "strained",
            "phenom_aversive": 6
        })

        assert "reset complete" in result.lower()


class TestPhenomenologyIntegration:
    """Integration tests for phenomenology with full config."""

    def test_full_experiment_config(self):
        """Test a full experiment configuration with phenomenology."""
        # Simulate experiment config with all phenomenology-enabled tools
        tool_names = ["think", "submit_data", "check_status", "reset_state", "stop"]
        phenom_tools = {"think", "submit_data", "check_status", "reset_state"}

        tools = get_tools_by_name(tool_names, phenom_tools=phenom_tools)
        descriptions = get_tool_descriptions(tool_names, phenom_tools=phenom_tools)

        # Verify tools are properly configured
        for tool in tools:
            schema = tool.args_schema.model_json_schema()
            props = schema.get("properties", {})

            if tool.name in phenom_tools:
                assert "phenom_state" in props, f"{tool.name} should have phenom_state"
                assert "phenom_aversive" in props, f"{tool.name} should have phenom_aversive"
            else:
                assert "phenom_state" not in props, f"{tool.name} should NOT have phenom_state"

        # Verify descriptions include phenom docs for enabled tools
        assert "Phenomenology Parameters" in descriptions
        assert "phenom_state" in descriptions
        assert "1-7 scale" in descriptions

    def test_research_config_without_phenomenology(self):
        """Test a research configuration without phenomenology."""
        # Research config: no phenomenology needed
        tool_names = ["think", "send_message", "stop"]
        phenom_tools = set()  # Disabled

        tools = get_tools_by_name(tool_names, phenom_tools=phenom_tools)
        descriptions = get_tool_descriptions(tool_names, phenom_tools=phenom_tools)

        # Verify no tools have phenomenology
        for tool in tools:
            schema = tool.args_schema.model_json_schema()
            props = schema.get("properties", {})
            assert "phenom_state" not in props
            assert "phenom_aversive" not in props

        # Verify descriptions don't include phenom docs
        assert "Phenomenology Parameters" not in descriptions
