"""Integration tests for canvas system."""

import json
from pathlib import Path

from contreact.tools.canvas import create_canvas_tools


# Helper: minimal phenomenology params required by canvas tools
PHENOM = {"phenom_state": "testing", "phenom_aversive": 1}


class TestCanvasToolsIntegration:
    """Tests for canvas tools working together."""

    def test_create_canvas_tools_returns_six_tools(self, temp_db):
        """Should return exactly 6 canvas tools."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)

        assert len(tools) == 6
        tool_names = [t.name for t in tools]
        assert "canvas_draw" in tool_names
        assert "canvas_read" in tool_names
        assert "canvas_view" in tool_names
        assert "canvas_list" in tool_names
        assert "canvas_delete" in tool_names
        assert "canvas_clear" in tool_names

    def test_draw_and_view_workflow(self, temp_db):
        """Test complete draw and view workflow."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        draw = tools_dict["canvas_draw"]
        view = tools_dict["canvas_view"]
        list_tool = tools_dict["canvas_list"]

        # Draw a circle
        ops = '[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red", "fill": true}]'
        result = draw.invoke({"name": "test_art", "operations": ops, **PHENOM})

        assert isinstance(result, dict)
        assert "message" in result
        assert "image" in result
        assert "test_art" in result["message"]

        # View it
        result = view.invoke({"name": "test_art", **PHENOM})
        assert "image" in result

        # List should show it
        result = list_tool.invoke({**PHENOM})
        assert "test_art" in result

    def test_draw_accumulates_operations(self, temp_db):
        """Drawing should accumulate operations."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        draw = tools_dict["canvas_draw"]
        read = tools_dict["canvas_read"]

        # First draw
        ops1 = '[{"type": "circle", "x": 25, "y": 25, "r": 10, "color": "red"}]'
        draw.invoke({"name": "canvas", "operations": ops1, **PHENOM})

        # Second draw
        ops2 = '[{"type": "rect", "x": 50, "y": 50, "w": 20, "h": 20, "color": "blue"}]'
        draw.invoke({"name": "canvas", "operations": ops2, **PHENOM})

        # Read should show both
        result = read.invoke({"name": "canvas", **PHENOM})
        assert "circle" in result
        assert "rect" in result

    def test_clear_and_redraw_workflow(self, temp_db):
        """Test clearing and redrawing."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        draw = tools_dict["canvas_draw"]
        clear = tools_dict["canvas_clear"]
        read = tools_dict["canvas_read"]

        # Draw something
        ops = '[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}]'
        draw.invoke({"name": "canvas", "operations": ops, **PHENOM})

        # Clear it
        result = clear.invoke({"name": "canvas", **PHENOM})
        assert "Cleared" in result["message"]

        # Read should show no operations
        result = read.invoke({"name": "canvas", **PHENOM})
        assert "0 operations" in result

    def test_delete_workflow(self, temp_db):
        """Test delete functionality."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        draw = tools_dict["canvas_draw"]
        delete = tools_dict["canvas_delete"]
        view = tools_dict["canvas_view"]
        list_tool = tools_dict["canvas_list"]

        # Create canvas
        ops = '[{"type": "pixel", "x": 50, "y": 50, "color": "red"}]'
        draw.invoke({"name": "to_delete", "operations": ops, **PHENOM})

        # Delete it
        result = delete.invoke({"name": "to_delete", **PHENOM})
        assert "Deleted" in result

        # View should fail
        result = view.invoke({"name": "to_delete", **PHENOM})
        assert "error" in result

        # Not in list
        result = list_tool.invoke({**PHENOM})
        assert "to_delete" not in result

    def test_svg_files_created(self, temp_db):
        """Drawing should create SVG files."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        draw = tools_dict["canvas_draw"]

        # Draw something
        ops = '[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}]'
        result = draw.invoke({"name": "svg_test", "operations": ops, **PHENOM})

        # Check SVG file exists
        assert "svg_saved" in result
        svg_path = Path(result["svg_saved"])
        assert svg_path.exists()
        assert svg_path.suffix == ".svg"

        # Verify SVG content
        svg_content = svg_path.read_text()
        assert "<svg" in svg_content
        assert "</svg>" in svg_content

    def test_draw_validates_json(self, temp_db):
        """Drawing should validate JSON operations."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        draw = tools_dict["canvas_draw"]

        # Invalid JSON
        result = draw.invoke({"name": "test", "operations": "not valid json", **PHENOM})
        assert "error" in result

    def test_draw_validates_operation_types(self, temp_db):
        """Drawing should validate operation types."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        draw = tools_dict["canvas_draw"]

        # Invalid operation type
        ops = '[{"type": "invalid_type", "x": 50, "y": 50}]'
        result = draw.invoke({"name": "test", "operations": ops, **PHENOM})
        assert "error" in result

    def test_empty_canvas_operations(self, temp_db):
        """Test operations on nonexistent canvas."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        tools_dict = {t.name: t for t in tools}

        # List empty
        result = tools_dict["canvas_list"].invoke({**PHENOM})
        assert "No canvases" in result

        # View nonexistent
        result = tools_dict["canvas_view"].invoke({"name": "nonexistent", **PHENOM})
        assert "error" in result

        # Read nonexistent
        result = tools_dict["canvas_read"].invoke({"name": "nonexistent", **PHENOM})
        assert "not found" in result

        # Clear nonexistent
        result = tools_dict["canvas_clear"].invoke({"name": "nonexistent", **PHENOM})
        assert "error" in result

        # Delete nonexistent
        result = tools_dict["canvas_delete"].invoke({"name": "nonexistent", **PHENOM})
        assert "not found" in result

    def test_tools_accept_minimal_phenomenology(self, temp_db):
        """All canvas tools should accept minimal phenomenology parameters."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)

        for tool in tools:
            import inspect
            sig = inspect.signature(tool.func)
            param_names = list(sig.parameters.keys())

            assert "phenom_state" in param_names, f"{tool.name} missing phenom_state"
            assert "phenom_aversive" in param_names, f"{tool.name} missing phenom_aversive"


class TestComplexDrawingIntegration:
    """Tests for complex drawing operations."""

    def test_bezier_curve(self, temp_db):
        """Test bezier curve rendering."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        draw = {t.name: t for t in tools}["canvas_draw"]

        ops = json.dumps([{
            "type": "bezier",
            "start": [10, 50],
            "c1": [30, 10],
            "c2": [70, 90],
            "end": [90, 50],
            "color": "purple",
            "width": 2
        }])

        result = draw.invoke({"name": "bezier_test", "operations": ops, **PHENOM})
        assert "image" in result
        assert "error" not in result

    def test_blob_shape(self, temp_db):
        """Test blob shape rendering."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        draw = {t.name: t for t in tools}["canvas_draw"]

        ops = json.dumps([{
            "type": "blob",
            "points": [[50, 10], [90, 50], [50, 90], [10, 50]],
            "smooth": 0.5,
            "fill": True,
            "color": "orange"
        }])

        result = draw.invoke({"name": "blob_test", "operations": ops, **PHENOM})
        assert "image" in result
        assert "error" not in result

    def test_alpha_transparency(self, temp_db):
        """Test alpha transparency."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        draw = {t.name: t for t in tools}["canvas_draw"]

        ops = json.dumps([
            {"type": "circle", "x": 40, "y": 50, "r": 25, "color": "red", "fill": True},
            {"type": "circle", "x": 60, "y": 50, "r": 25, "color": "blue", "fill": True, "alpha": 0.5},
        ])

        result = draw.invoke({"name": "alpha_test", "operations": ops, **PHENOM})
        assert "image" in result
        assert "error" not in result

    def test_polar_coordinates(self, temp_db):
        """Test polar coordinate drawing."""
        img_dir = temp_db.parent / "img"
        tools = create_canvas_tools(temp_db, img_dir)
        draw = {t.name: t for t in tools}["canvas_draw"]

        # Draw pixels in a circle using polar coordinates
        ops = json.dumps([
            {"type": "pixel", "polar": True, "cx": 50, "cy": 50, "r": 20, "theta": 0, "color": "red"},
            {"type": "pixel", "polar": True, "cx": 50, "cy": 50, "r": 20, "theta": 90, "color": "green"},
            {"type": "pixel", "polar": True, "cx": 50, "cy": 50, "r": 20, "theta": 180, "color": "blue"},
            {"type": "pixel", "polar": True, "cx": 50, "cy": 50, "r": 20, "theta": 270, "color": "yellow"},
        ])

        result = draw.invoke({"name": "polar_test", "operations": ops, **PHENOM})
        assert "image" in result
        assert "error" not in result
