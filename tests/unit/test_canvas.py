"""Unit tests for contreact.tools.canvas module."""

from datetime import datetime

from contreact.tools.canvas import CanvasStore, CanvasRenderer


class TestCanvasStoreInit:
    """Tests for CanvasStore initialization."""

    def test_creates_database(self, temp_db):
        """Should create SQLite database file."""
        CanvasStore(temp_db)
        assert temp_db.exists()

    def test_creates_img_directory(self, temp_db):
        """Should create img directory for SVG files."""
        CanvasStore(temp_db)
        img_dir = temp_db.parent / "img"
        assert img_dir.exists()


class TestCanvasStoreCreate:
    """Tests for CanvasStore.save method."""

    def test_create_new_canvas(self, canvas_store):
        """Should create new canvas and return False."""
        ops = [{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}]
        is_update = canvas_store.save("my_canvas", ops)
        assert is_update is False

    def test_update_existing_canvas(self, canvas_store):
        """Should update existing canvas and return True."""
        ops1 = [{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}]
        ops2 = [{"type": "rect", "x": 10, "y": 10, "w": 20, "h": 20, "color": "blue"}]

        canvas_store.save("canvas", ops1)
        is_update = canvas_store.save("canvas", ops2)

        assert is_update is True

    def test_operations_are_appended(self, canvas_store):
        """Operations should be appended on update."""
        ops1 = [{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}]
        ops2 = [{"type": "rect", "x": 10, "y": 10, "w": 20, "h": 20, "color": "blue"}]

        canvas_store.save("canvas", ops1)
        canvas_store.save("canvas", ops2)

        canvas = canvas_store.get_canvas("canvas")
        assert len(canvas["operations"]) == 2

    def test_default_dimensions(self, canvas_store):
        """New canvas should have 160x90 dimensions by default (16:9)."""
        canvas_store.save("canvas", [])
        canvas = canvas_store.get_canvas("canvas")
        assert canvas["width"] == 160
        assert canvas["height"] == 90


class TestCanvasStoreGet:
    """Tests for CanvasStore.get_canvas method."""

    def test_get_existing_canvas(self, canvas_store):
        """Should return canvas data for existing canvas."""
        ops = [{"type": "pixel", "x": 10, "y": 10, "color": "blue"}]
        canvas_store.save("test", ops)

        canvas = canvas_store.get_canvas("test")
        assert canvas is not None
        assert canvas["name"] == "test"
        assert canvas["operations"] == ops

    def test_get_nonexistent_canvas(self, canvas_store):
        """Should return None for nonexistent canvas."""
        canvas = canvas_store.get_canvas("nonexistent")
        assert canvas is None

    def test_get_returns_timestamps(self, canvas_store):
        """Should return created_at and updated_at timestamps."""
        canvas_store.save("test", [])

        canvas = canvas_store.get_canvas("test")
        assert isinstance(canvas["created_at"], datetime)
        assert isinstance(canvas["updated_at"], datetime)


class TestCanvasStoreList:
    """Tests for CanvasStore.list_entries method."""

    def test_list_empty(self, canvas_store):
        """Should return empty list when no canvases."""
        canvases = canvas_store.list_entries()
        assert canvases == []

    def test_list_returns_all_canvases(self, canvas_store):
        """Should return all canvases."""
        canvas_store.save("canvas1", [])
        canvas_store.save("canvas2", [])
        canvas_store.save("canvas3", [])

        canvases = canvas_store.list_entries()
        assert len(canvases) == 3

    def test_list_returns_dimensions(self, canvas_store):
        """Should return name, width, height, updated_at."""
        canvas_store.save("canvas", [])

        canvases = canvas_store.list_entries()
        name, width, height, updated_at = canvases[0]

        assert name == "canvas"
        assert width == 160
        assert height == 90
        assert isinstance(updated_at, datetime)

class TestCanvasStoreClear:
    """Tests for CanvasStore.clear_canvas method."""

    def test_clear_existing_canvas(self, canvas_store):
        """Should clear operations and return True."""
        ops = [{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}]
        canvas_store.save("test", ops)

        cleared = canvas_store.clear_canvas("test")
        assert cleared is True

        canvas = canvas_store.get_canvas("test")
        assert canvas["operations"] == []

    def test_clear_nonexistent_canvas(self, canvas_store):
        """Should return False for nonexistent canvas."""
        cleared = canvas_store.clear_canvas("nonexistent")
        assert cleared is False


class TestCanvasStoreDelete:
    """Tests for CanvasStore.delete_canvas method."""

    def test_delete_existing_canvas(self, canvas_store):
        """Should delete canvas and return True."""
        canvas_store.save("to_delete", [])

        deleted = canvas_store.delete_canvas("to_delete")
        assert deleted is True
        assert canvas_store.get_canvas("to_delete") is None

    def test_delete_nonexistent_canvas(self, canvas_store):
        """Should return False for nonexistent canvas."""
        deleted = canvas_store.delete_canvas("nonexistent")
        assert deleted is False


class TestCanvasRendererColors:
    """Tests for CanvasRenderer.parse_color method."""

    def test_parse_named_color(self):
        """Should parse named colors."""
        renderer = CanvasRenderer(100, 100)
        assert renderer.parse_color("red") == (255, 0, 0, 255)
        assert renderer.parse_color("blue") == (0, 0, 255, 255)
        assert renderer.parse_color("white") == (255, 255, 255, 255)

    def test_parse_hex_color(self):
        """Should parse hex colors."""
        renderer = CanvasRenderer(100, 100)
        assert renderer.parse_color("#FF0000") == (255, 0, 0, 255)
        assert renderer.parse_color("#00FF00") == (0, 255, 0, 255)

    def test_parse_rgb_list(self):
        """Should parse RGB list."""
        renderer = CanvasRenderer(100, 100)
        assert renderer.parse_color([255, 128, 0]) == (255, 128, 0, 255)

    def test_parse_with_alpha(self):
        """Should apply alpha parameter."""
        renderer = CanvasRenderer(100, 100)
        color = renderer.parse_color("red", alpha=0.5)
        assert color == (255, 0, 0, 127)  # 0.5 * 255 = 127

    def test_parse_case_insensitive(self):
        """Should handle case-insensitive named colors."""
        renderer = CanvasRenderer(100, 100)
        assert renderer.parse_color("RED") == (255, 0, 0, 255)
        assert renderer.parse_color("Blue") == (0, 0, 255, 255)


class TestCanvasRendererRender:
    """Tests for CanvasRenderer.render_to_image method.

    Note: CanvasRenderer uses RENDER_SCALE=4, so 100x100 logical becomes 400x400 rendered.
    Pixel coordinates in tests must be scaled by 4.
    """
    SCALE = 4  # Must match CanvasRenderer.RENDER_SCALE

    def test_render_empty_canvas(self):
        """Should render white canvas with no operations."""
        renderer = CanvasRenderer(100, 100)
        img = renderer.render_to_image([])

        # 100x100 logical * 4 scale = 400x400 rendered
        assert img.size == (400, 400)
        # Check center pixel is white (50*4=200)
        pixel = img.getpixel((200, 200))
        assert pixel == (255, 255, 255, 255)

    def test_render_pixel(self):
        """Should render pixel operation."""
        renderer = CanvasRenderer(100, 100)
        ops = [{"type": "pixel", "x": 50, "y": 50, "color": "red"}]
        img = renderer.render_to_image(ops)

        # Pixel at logical (50,50) -> rendered (200,200)
        pixel = img.getpixel((200, 200))
        assert pixel == (255, 0, 0, 255)

    def test_render_circle(self):
        """Should render filled circle."""
        renderer = CanvasRenderer(100, 100)
        ops = [{"type": "circle", "x": 50, "y": 50, "r": 10, "color": "blue", "fill": True}]
        img = renderer.render_to_image(ops)

        # Center at logical (50,50) -> rendered (200,200)
        pixel = img.getpixel((200, 200))
        assert pixel == (0, 0, 255, 255)

    def test_render_rect(self):
        """Should render filled rectangle."""
        renderer = CanvasRenderer(100, 100)
        ops = [{"type": "rect", "x": 20, "y": 20, "w": 30, "h": 30, "color": "green", "fill": True}]
        img = renderer.render_to_image(ops)

        # Inside rect at logical (35,35) -> rendered (140,140)
        pixel = img.getpixel((140, 140))
        assert pixel == (0, 255, 0, 255)

    def test_render_with_alpha(self):
        """Should render with alpha transparency."""
        renderer = CanvasRenderer(100, 100)
        ops = [{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red", "alpha": 0.5, "fill": True}]
        img = renderer.render_to_image(ops)

        # Center at logical (50,50) -> rendered (200,200)
        pixel = img.getpixel((200, 200))
        # Result should be approximately (255, 127, 127, 255) due to alpha compositing
        assert pixel[0] > 200  # Red component high
        assert pixel[1] > 100  # Some green from white background

    def test_render_to_png_returns_bytes(self):
        """Should return PNG bytes."""
        renderer = CanvasRenderer(100, 100)
        png_bytes = renderer.render_to_png([])

        assert isinstance(png_bytes, bytes)
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic number

    def test_render_to_svg_returns_string(self):
        """Should return SVG string."""
        renderer = CanvasRenderer(100, 100)
        svg = renderer.render_to_svg([])

        assert isinstance(svg, str)
        assert svg.startswith('<svg')
        assert '</svg>' in svg


class TestCanvasRendererPolar:
    """Tests for polar coordinate support.

    Note: polar_to_cartesian returns scaled coords by default. Use scale=False for logical.
    """

    def test_polar_to_cartesian(self):
        """Should convert polar to cartesian correctly (logical coords)."""
        renderer = CanvasRenderer(100, 100)

        # 0 degrees should be to the right (using scale=False for logical coords)
        x, y = renderer.polar_to_cartesian(50, 50, 10, 0, scale=False)
        assert x == 60
        assert y == 50

        # 90 degrees should be down
        x, y = renderer.polar_to_cartesian(50, 50, 10, 90, scale=False)
        assert x == 50
        assert y == 60

    def test_render_polar_pixel(self):
        """Should render pixel with polar coordinates."""
        renderer = CanvasRenderer(100, 100)
        ops = [{"type": "pixel", "polar": True, "cx": 50, "cy": 50, "r": 10, "theta": 0, "color": "red"}]
        img = renderer.render_to_image(ops)

        # Logical (60, 50) -> rendered (240, 200)
        pixel = img.getpixel((240, 200))
        assert pixel == (255, 0, 0, 255)
