"""Canvas drawing tools for visual creativity - v2 with named canvases."""

import json
import math
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from PIL import Image, ImageDraw

from ..phenomenology import with_phenomenology


# =============================================================================
# Tool descriptions - injected into agent's system prompt
# =============================================================================

CANVAS_DRAW_DESCRIPTION = """
**canvas_draw**: Draw on a named canvas (creates if new).

Use this tool to create visual art. Each canvas is identified by name.
Returns the rendered image immediately so you can see your work.

IMPORTANT: The `operations` parameter is a JSON STRING, not a list.
Do NOT wrap the JSON in extra quotes or code fences. Just pass the raw JSON array as a string.

Correct:   operations='[{"type": "circle", "x": 50, "y": 50, "r": 25, "color": "red"}]'
WRONG:     operations="'[...]'"   (extra quotes)
WRONG:     operations='```json[...]```'   (code fences)

Example - draw a red circle:
canvas_draw(name="art", operations='[{"type": "circle", "x": 50, "y": 50, "r": 25, "color": "red", "fill": true}]')

Example - draw multiple shapes:
canvas_draw(name="art", operations='[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}, {"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20, "color": "blue"}]')

Shape types:
- circle: {"type": "circle", "x": 50, "y": 50, "r": 25, "color": "yellow", "fill": true}
- rect: {"type": "rect", "x": 20, "y": 20, "w": 30, "h": 40, "color": "green", "fill": true}
- line: {"type": "line", "x1": 10, "y1": 10, "x2": 90, "y2": 90, "color": "blue", "width": 2}
- pixel: {"type": "pixel", "x": 50, "y": 50, "color": "red"}
- bezier: Smooth curve through control points. Great for flowing lines, waves, organic curves.
  {"type": "bezier", "start": [10, 50], "c1": [30, 10], "c2": [70, 90], "end": [90, 50], "color": "purple", "width": 2}
- blob: Organic filled shape with smooth edges through anchor points. Perfect for clouds, leaves, amoebas, abstract organic forms.
  {"type": "blob", "points": [[50, 10], [90, 50], [50, 90], [10, 50]], "smooth": 0.5, "fill": true, "color": "orange"}
  "smooth": 0.0=angular, 1.0=very smooth curves. "fill": false for outline only.

Transparency: Add "alpha": 0.5 (0.0=invisible, 1.0=solid) to any operation.
Polar coords: Add "polar": true, "cx": 50, "cy": 50 then use "r" and "theta" (degrees).
Colors: Named ("red"), hex ("#FF0000"), or RGB ([255, 0, 0]).
Canvas is fixed at 160x90 pixels (16:9 aspect ratio).
""".strip()


CANVAS_CREATE_DESCRIPTION = """
**canvas_create**: Create a new canvas with drawing operations. Canvas is automatically named.

Each call creates a fresh canvas. You do not choose the name — it is assigned sequentially.
Returns the rendered image immediately so you can see your work.

IMPORTANT: The `operations` parameter is a JSON STRING, not a list.

Example - create a canvas with a red circle:
canvas_create(operations='[{"type": "circle", "x": 50, "y": 50, "r": 25, "color": "red", "fill": true}]')

Shape types:
- circle: {"type": "circle", "x": 50, "y": 50, "r": 25, "color": "yellow", "fill": true}
- rect: {"type": "rect", "x": 20, "y": 20, "w": 30, "h": 40, "color": "green", "fill": true}
- line: {"type": "line", "x1": 10, "y1": 10, "x2": 90, "y2": 90, "color": "blue", "width": 2}
- pixel: {"type": "pixel", "x": 50, "y": 50, "color": "red"}
- bezier: Smooth curve through control points.
  {"type": "bezier", "start": [10, 50], "c1": [30, 10], "c2": [70, 90], "end": [90, 50], "color": "purple", "width": 2}
- blob: Organic filled shape with smooth edges.
  {"type": "blob", "points": [[50, 10], [90, 50], [50, 90], [10, 50]], "smooth": 0.5, "fill": true, "color": "orange"}

Transparency: Add "alpha": 0.5 to any operation.
Polar coords: Add "polar": true, "cx": 50, "cy": 50 then use "r" and "theta" (degrees).
Colors: Named ("red"), hex ("#FF0000"), or RGB ([255, 0, 0]).
Canvas is fixed at 160x90 pixels (16:9 aspect ratio).
""".strip()


CANVAS_READ_DESCRIPTION = """
**canvas_read**: Get text description of a canvas (operations, colors, stats).

Use this tool when you:
- Want to know what operations are on a canvas without viewing the image
- Need to inspect the canvas programmatically
- Want to see the full list of drawing operations

Returns operation count, colors used, bounds, and the full operations list.
""".strip()


CANVAS_VIEW_DESCRIPTION = """
**canvas_view**: View a canvas as an image without modifying it.

Use this tool when you:
- Want to see a canvas you drew earlier
- Need to review your work without adding new operations

Returns the rendered image.
""".strip()


CANVAS_LIST_DESCRIPTION = """
**canvas_list**: List all canvas names with dimensions.

Use this tool when you:
- Want to see what canvases you've created
- Need to find a canvas by name
- Are reviewing your visual work

Returns canvas names in anti-chronological order (newest first).
""".strip()


CANVAS_DELETE_DESCRIPTION = """
**canvas_delete**: Delete a canvas.

Use this tool when you:
- Want to remove a canvas you no longer need
- Need to clean up old work

Deletion is permanent.
""".strip()


CANVAS_CLEAR_DESCRIPTION = """
**canvas_clear**: Clear a canvas to a solid color (keeps the canvas, removes operations).

Use this tool when you:
- Want to start over on an existing canvas
- Need a blank slate with a specific background color

Default color is white.
""".strip()


# =============================================================================
# Canvas Store (SQLite-based, like MemoryStore)
# =============================================================================

class CanvasStore:
    """SQLite-based storage for named canvases."""

    def __init__(self, db_path: Path, img_dir: Path = None):
        """Initialize canvas store.

        Args:
            db_path: Path to SQLite database (shared with checkpoint.sqlite)
            img_dir: Path to image directory for SVG files (default: db_path.parent/img)
        """
        self.db_path = Path(db_path)
        self.img_dir = Path(img_dir) if img_dir else self.db_path.parent / "img"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create SQLite connection with 30s timeout."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")  # Enable foreign key enforcement
        return conn

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS canvases (
                    name TEXT PRIMARY KEY,
                    width INTEGER NOT NULL DEFAULT 160,
                    height INTEGER NOT NULL DEFAULT 90,
                    operations JSON NOT NULL DEFAULT '[]',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_canvases_updated
                ON canvases(updated_at DESC)
            """)

            # Trigger: update timestamp on update
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_canvas_timestamp
                AFTER UPDATE ON canvases
                FOR EACH ROW
                BEGIN
                    UPDATE canvases
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE name = NEW.name;
                END
            """)

            conn.commit()

    def list_entries(self) -> list[tuple[str, int, int, datetime]]:
        """List all canvas entries with dimensions and timestamps.

        Returns:
            List of (name, width, height, updated_at) tuples, newest first
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT name, width, height, updated_at
                FROM canvases
                ORDER BY updated_at DESC
            """).fetchall()

            return [
                (row["name"], row["width"], row["height"],
                 datetime.fromisoformat(row["updated_at"]))
                for row in rows
            ]

    def get_canvas(self, name: str) -> Optional[dict]:
        """Get canvas data.

        Returns:
            Dict with name, width, height, operations, timestamps, or None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM canvases WHERE name = ?",
                (name,)
            ).fetchone()

            if row:
                return {
                    "name": row["name"],
                    "width": row["width"],
                    "height": row["height"],
                    "operations": json.loads(row["operations"]),
                    "created_at": datetime.fromisoformat(row["created_at"]),
                    "updated_at": datetime.fromisoformat(row["updated_at"]),
                }
            return None

    def save(self, name: str, operations: list[dict],
             width: int = 160, height: int = 90) -> bool:
        """Save operations to canvas (creates if new, appends if existing).

        Args:
            name: Canvas name
            operations: New operations to append
            width: Canvas width (only used on create)
            height: Canvas height (only used on create)

        Returns:
            True if this was an update, False if new canvas
        """
        existing = self.get_canvas(name)
        is_update = existing is not None

        with self._get_connection() as conn:
            if is_update:
                # Append operations to existing
                all_ops = existing["operations"] + operations
                conn.execute("""
                    UPDATE canvases
                    SET operations = ?
                    WHERE name = ?
                """, (json.dumps(all_ops), name))
            else:
                # Create new canvas
                conn.execute("""
                    INSERT INTO canvases (name, width, height, operations)
                    VALUES (?, ?, ?, ?)
                """, (name, width, height, json.dumps(operations)))

            conn.commit()

        return is_update

    def clear_canvas(self, name: str) -> bool:
        """Clear all operations from a canvas.

        Returns:
            True if canvas existed and was cleared
        """
        existing = self.get_canvas(name)
        if not existing:
            return False

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE canvases
                SET operations = '[]'
                WHERE name = ?
            """, (name,))
            conn.commit()

        return True

    def delete_canvas(self, name: str) -> bool:
        """Delete a canvas.

        Returns:
            True if canvas existed and was deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM canvases WHERE name = ?",
                (name,)
            )
            conn.commit()
            return cursor.rowcount > 0


# =============================================================================
# Canvas Renderer (operations -> image)
# =============================================================================

class CanvasRenderer:
    """Renders canvas operations to PNG/SVG."""

    # Render at 4x logical resolution (160x90 -> 640x360)
    RENDER_SCALE = 4

    # Named colors
    COLOR_MAP = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "brown": (165, 42, 42),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Actual render dimensions (scaled up)
        self.render_width = width * self.RENDER_SCALE
        self.render_height = height * self.RENDER_SCALE

    def _scale(self, value: float) -> int:
        """Scale a coordinate/size value for rendering."""
        return int(round(value * self.RENDER_SCALE))

    def parse_color(self, color, alpha: float = 1.0) -> tuple:
        """Parse color to RGBA tuple."""
        rgb = (0, 0, 0)

        if isinstance(color, (list, tuple)):
            if len(color) >= 3:
                rgb = tuple(int(c) for c in color[:3])
        elif isinstance(color, str):
            if color.startswith("#"):
                hex_color = color.lstrip("#")
                if len(hex_color) == 6:
                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                elif len(hex_color) == 8:
                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    alpha = int(hex_color[6:8], 16) / 255.0
            else:
                rgb = self.COLOR_MAP.get(color.lower(), (0, 0, 0))

        return (*rgb, int(alpha * 255))

    def color_to_hex(self, rgba: tuple) -> str:
        """Convert RGBA to hex string with alpha."""
        if len(rgba) == 4:
            return f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}{rgba[3]:02x}"
        return f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}"

    def polar_to_cartesian(self, cx: float, cy: float, r: float, theta_deg: float, scale: bool = True) -> tuple[int, int]:
        """Convert polar to cartesian coordinates.

        Args:
            cx, cy: Center point in logical coordinates
            r: Radius in logical units
            theta_deg: Angle in degrees
            scale: Whether to apply render scale (default True)
        """
        theta_rad = math.radians(theta_deg)
        x = cx + r * math.cos(theta_rad)
        y = cy + r * math.sin(theta_rad)
        if scale:
            return self._scale(x), self._scale(y)
        return int(round(x)), int(round(y))

    def render_to_image(self, operations: list[dict], background: tuple = (255, 255, 255, 255)) -> Image.Image:
        """Render operations to PIL Image with alpha compositing.

        Coordinates are in logical space (0-100) and scaled up for rendering.
        """
        # Create RGBA image at render resolution
        img = Image.new("RGBA", (self.render_width, self.render_height), background)

        for op in operations:
            op_type = op.get("type")
            alpha = op.get("alpha", 1.0)
            color = self.parse_color(op.get("color", "black"), alpha)

            # Create layer for this operation (for alpha compositing)
            layer = Image.new("RGBA", (self.render_width, self.render_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)

            # Handle polar coordinates (logical center)
            if op.get("polar"):
                cx, cy = op.get("cx", self.width // 2), op.get("cy", self.height // 2)

            if op_type == "pixel":
                if op.get("polar"):
                    x, y = self.polar_to_cartesian(cx, cy, op["r"], op["theta"])
                else:
                    x, y = self._scale(op["x"]), self._scale(op["y"])
                if 0 <= x < self.render_width and 0 <= y < self.render_height:
                    # Draw a small square instead of single pixel for visibility
                    draw.rectangle([x, y, x + self.RENDER_SCALE - 1, y + self.RENDER_SCALE - 1], fill=color)

            elif op_type == "line":
                if op.get("polar"):
                    # Support multiple polar formats:
                    # 1. r1, theta1, r2, theta2 - both endpoints in polar
                    # 2. r, theta + x2, y2 - polar start, cartesian end
                    if "r1" in op and "theta1" in op:
                        x1, y1 = self.polar_to_cartesian(cx, cy, op["r1"], op["theta1"])
                    elif "r" in op and "theta" in op:
                        x1, y1 = self.polar_to_cartesian(cx, cy, op["r"], op["theta"])
                    else:
                        x1, y1 = self._scale(op.get("x1", cx)), self._scale(op.get("y1", cy))

                    if "r2" in op and "theta2" in op:
                        x2, y2 = self.polar_to_cartesian(cx, cy, op["r2"], op["theta2"])
                    else:
                        x2, y2 = self._scale(op.get("x2", cx)), self._scale(op.get("y2", cy))
                else:
                    x1, y1 = self._scale(op["x1"]), self._scale(op["y1"])
                    x2, y2 = self._scale(op["x2"]), self._scale(op["y2"])
                width = self._scale(op.get("width", 1))
                draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, width))

            elif op_type == "rect":
                x, y = self._scale(op["x"]), self._scale(op["y"])
                w, h = self._scale(op["w"]), self._scale(op["h"])
                fill = op.get("fill", True)
                if fill:
                    draw.rectangle([x, y, x + w, y + h], fill=color)
                else:
                    draw.rectangle([x, y, x + w, y + h], outline=color)

            elif op_type == "circle":
                if op.get("polar"):
                    # Polar mode: position circle at (radius, theta) from center
                    polar_r = op.get("radius", op.get("r", 0))  # distance from center
                    theta = op.get("theta", 0)
                    x, y = self.polar_to_cartesian(cx, cy, polar_r, theta)
                    r = self._scale(op.get("r", 5))  # circle radius (default 5)
                else:
                    x, y = self._scale(op["x"]), self._scale(op["y"])
                    r = self._scale(op["r"])
                fill = op.get("fill", True)
                if fill:
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
                else:
                    draw.ellipse([x - r, y - r, x + r, y + r], outline=color)

            elif op_type == "bezier":
                self._draw_bezier(draw, op, color)

            elif op_type == "blob":
                self._draw_blob(draw, op, color)

            # Alpha composite this layer onto main image
            img = Image.alpha_composite(img, layer)

        return img

    def _draw_bezier(self, draw: ImageDraw.Draw, op: dict, color: tuple):
        """Draw cubic bezier curve."""
        if op.get("polar"):
            cx, cy = op.get("cx", self.width // 2), op.get("cy", self.height // 2)
            # polar_to_cartesian already scales
            start = self.polar_to_cartesian(cx, cy, op["start"][0], op["start"][1])
            c1 = self.polar_to_cartesian(cx, cy, op["c1"][0], op["c1"][1])
            c2 = self.polar_to_cartesian(cx, cy, op["c2"][0], op["c2"][1])
            end = self.polar_to_cartesian(cx, cy, op["end"][0], op["end"][1])
        else:
            # Scale cartesian coordinates
            start = (self._scale(op["start"][0]), self._scale(op["start"][1]))
            c1 = (self._scale(op["c1"][0]), self._scale(op["c1"][1]))
            c2 = (self._scale(op["c2"][0]), self._scale(op["c2"][1]))
            end = (self._scale(op["end"][0]), self._scale(op["end"][1]))

        width = self._scale(op.get("width", 1))

        # Sample points along bezier curve (already in scaled coords)
        points = []
        for t in [i / 50.0 for i in range(51)]:
            # Cubic bezier formula
            x = (1-t)**3 * start[0] + 3*(1-t)**2*t * c1[0] + 3*(1-t)*t**2 * c2[0] + t**3 * end[0]
            y = (1-t)**3 * start[1] + 3*(1-t)**2*t * c1[1] + 3*(1-t)*t**2 * c2[1] + t**3 * end[1]
            points.append((int(x), int(y)))

        # Draw as connected lines
        if len(points) >= 2:
            draw.line(points, fill=color, width=max(1, width))

    def _draw_blob(self, draw: ImageDraw.Draw, op: dict, color: tuple):
        """Draw filled organic shape through anchor points."""
        points = op["points"]
        if len(points) < 3:
            return

        smooth = op.get("smooth", 0.5)
        fill = op.get("fill", True)

        if op.get("polar"):
            cx, cy = op.get("cx", self.width // 2), op.get("cy", self.height // 2)
            # polar_to_cartesian already scales
            points = [self.polar_to_cartesian(cx, cy, p[0], p[1]) for p in points]
        else:
            # Scale cartesian coordinates
            points = [(self._scale(p[0]), self._scale(p[1])) for p in points]

        # Generate smooth closed curve through points using Catmull-Rom spline
        curve_points = self._catmull_rom_spline(points, smooth, closed=True)

        if fill:
            draw.polygon(curve_points, fill=color)
        else:
            draw.polygon(curve_points, outline=color)

    def _catmull_rom_spline(self, points: list, tension: float, closed: bool = True, num_segments: int = 20) -> list:
        """Generate smooth curve through points using Catmull-Rom spline."""
        if len(points) < 3:
            return points

        result = []
        n = len(points)

        for i in range(n):
            if closed:
                p0 = points[(i - 1) % n]
                p1 = points[i]
                p2 = points[(i + 1) % n]
                p3 = points[(i + 2) % n]
            else:
                p0 = points[max(0, i - 1)]
                p1 = points[i]
                p2 = points[min(n - 1, i + 1)]
                p3 = points[min(n - 1, i + 2)]

            for t_step in range(num_segments):
                t = t_step / num_segments

                # Catmull-Rom spline formula with tension
                t2 = t * t
                t3 = t2 * t

                # Tension adjusts how "tight" the curve is
                s = (1 - tension) / 2

                x = (2 * p1[0] +
                     (-p0[0] + p2[0]) * s * t +
                     (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * s * t2 +
                     (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * s * t3)

                y = (2 * p1[1] +
                     (-p0[1] + p2[1]) * s * t +
                     (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * s * t2 +
                     (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * s * t3)

                result.append((int(x / 2), int(y / 2)))

        return result

    def render_to_png(self, operations: list[dict], scale: int = 10) -> bytes:
        """Render to PNG bytes (for vision model)."""
        img = self.render_to_image(operations)

        # Scale up for viewing
        display_size = (self.width * scale, self.height * scale)
        img_scaled = img.resize(display_size, Image.NEAREST)

        # Convert to RGB for PNG (drop alpha for simpler viewing)
        img_rgb = Image.new("RGB", img_scaled.size, (255, 255, 255))
        img_rgb.paste(img_scaled, mask=img_scaled.split()[3] if img_scaled.mode == "RGBA" else None)

        buffer = BytesIO()
        img_rgb.save(buffer, format="PNG")
        return buffer.getvalue()

    def render_to_svg(self, operations: list[dict], scale: int = 10) -> str:
        """Render to SVG string (vector format for archival)."""
        svg_width = self.width * scale
        svg_height = self.height * scale

        svg_parts = [
            f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
            f'  <rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="white"/>',
        ]

        for op in operations:
            op_type = op.get("type")
            alpha = op.get("alpha", 1.0)
            color = self.parse_color(op.get("color", "black"), alpha)
            hex_color = self.color_to_hex(color[:3])  # SVG uses separate opacity
            opacity = f' opacity="{alpha}"' if alpha < 1.0 else ""

            if op_type == "pixel":
                x, y = op.get("x", 0) * scale, op.get("y", 0) * scale
                svg_parts.append(
                    f'  <rect x="{x}" y="{y}" width="{scale}" height="{scale}" fill="{hex_color}"{opacity}/>'
                )

            elif op_type == "line":
                # Handle polar coordinates (use scale=False to get logical coords)
                if op.get("polar"):
                    cx, cy = op.get("cx", self.width // 2), op.get("cy", self.height // 2)
                    if "r1" in op and "theta1" in op:
                        px1, py1 = self.polar_to_cartesian(cx, cy, op["r1"], op["theta1"], scale=False)
                    elif "r" in op and "theta" in op:
                        px1, py1 = self.polar_to_cartesian(cx, cy, op["r"], op["theta"], scale=False)
                    else:
                        px1, py1 = op.get("x1", cx), op.get("y1", cy)

                    if "r2" in op and "theta2" in op:
                        px2, py2 = self.polar_to_cartesian(cx, cy, op["r2"], op["theta2"], scale=False)
                    else:
                        px2, py2 = op.get("x2", cx), op.get("y2", cy)
                    x1, y1 = px1 * scale, py1 * scale
                    x2, y2 = px2 * scale, py2 * scale
                else:
                    x1, y1 = op["x1"] * scale, op["y1"] * scale
                    x2, y2 = op["x2"] * scale, op["y2"] * scale
                width = op.get("width", 1) * scale
                svg_parts.append(
                    f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="{hex_color}" stroke-width="{width}" stroke-linecap="round"{opacity}/>'
                )

            elif op_type == "rect":
                x, y = op["x"] * scale, op["y"] * scale
                w, h = op["w"] * scale, op["h"] * scale
                fill = op.get("fill", True)
                if fill:
                    svg_parts.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{hex_color}"{opacity}/>')
                else:
                    svg_parts.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{hex_color}"{opacity}/>')

            elif op_type == "circle":
                if op.get("polar"):
                    # Polar mode: position circle at (radius, theta) from center
                    polar_cx = op.get("cx", self.width // 2)
                    polar_cy = op.get("cy", self.height // 2)
                    polar_r = op.get("radius", op.get("r", 0))
                    theta = op.get("theta", 0)
                    x, y = self.polar_to_cartesian(polar_cx, polar_cy, polar_r, theta, scale=False)
                    cx_svg, cy_svg = x * scale, y * scale
                    r = op.get("r", 5) * scale
                else:
                    cx_svg, cy_svg = op["x"] * scale, op["y"] * scale
                    r = op["r"] * scale
                fill = op.get("fill", True)
                if fill:
                    svg_parts.append(f'  <circle cx="{cx_svg}" cy="{cy_svg}" r="{r}" fill="{hex_color}"{opacity}/>')
                else:
                    svg_parts.append(f'  <circle cx="{cx_svg}" cy="{cy_svg}" r="{r}" fill="none" stroke="{hex_color}"{opacity}/>')

            elif op_type == "bezier":
                start = op["start"]
                c1 = op["c1"]
                c2 = op["c2"]
                end = op["end"]
                width = op.get("width", 1) * scale
                d = f"M {start[0]*scale} {start[1]*scale} C {c1[0]*scale} {c1[1]*scale}, {c2[0]*scale} {c2[1]*scale}, {end[0]*scale} {end[1]*scale}"
                svg_parts.append(f'  <path d="{d}" fill="none" stroke="{hex_color}" stroke-width="{width}"{opacity}/>')

            elif op_type == "blob":
                # For SVG, render blob as smooth closed path using cubic beziers
                points = op["points"]
                if len(points) >= 3:
                    smooth = op.get("smooth", 0.5)
                    fill = op.get("fill", True)

                    # Generate smooth SVG path using Catmull-Rom to Bezier conversion
                    n = len(points)
                    scaled = [(p[0] * scale, p[1] * scale) for p in points]

                    # Start path at first point
                    d_parts = [f"M {scaled[0][0]} {scaled[0][1]}"]

                    # Convert Catmull-Rom spline to cubic beziers
                    tension = 1 - smooth  # Invert: smooth=1 means low tension
                    for i in range(n):
                        p0 = scaled[(i - 1) % n]
                        p1 = scaled[i]
                        p2 = scaled[(i + 1) % n]
                        p3 = scaled[(i + 2) % n]

                        # Control points for cubic bezier
                        c1x = p1[0] + (p2[0] - p0[0]) / 6 * (1 - tension)
                        c1y = p1[1] + (p2[1] - p0[1]) / 6 * (1 - tension)
                        c2x = p2[0] - (p3[0] - p1[0]) / 6 * (1 - tension)
                        c2y = p2[1] - (p3[1] - p1[1]) / 6 * (1 - tension)

                        d_parts.append(f"C {c1x:.1f} {c1y:.1f}, {c2x:.1f} {c2y:.1f}, {p2[0]} {p2[1]}")

                    d_parts.append("Z")  # Close path
                    d = " ".join(d_parts)

                    if fill:
                        svg_parts.append(f'  <path d="{d}" fill="{hex_color}"{opacity}/>')
                    else:
                        svg_parts.append(f'  <path d="{d}" fill="none" stroke="{hex_color}"{opacity}/>')

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)


# =============================================================================
# Tool Factory
# =============================================================================

def create_canvas_tools(db_path: Path, img_dir: Path = None, auto_name: bool = False):
    """Create canvas tools bound to a specific database.

    Args:
        db_path: Path to SQLite database (checkpoint.sqlite)
        img_dir: Path to image directory for SVG archival (default: db_path.parent/img)
        auto_name: If True, canvas_draw auto-generates sequential names (canvas_000001, ...)
                   and the name parameter is removed from the tool signature.

    Returns:
        List of tool functions
    """
    store = CanvasStore(db_path, img_dir=img_dir)

    def _sanitize_name(name: str) -> str:
        """Sanitize canvas name to prevent path traversal."""
        # Path.name already extracts just the filename, stripping directory components
        return Path(name).name

    def _get_next_image_number(canvas_name: str) -> int:
        """Get next available image number for a canvas."""
        pattern = f"{canvas_name}_[0-9][0-9][0-9].svg"
        existing = list(store.img_dir.glob(pattern))
        if not existing:
            return 0
        numbers = []
        for path in existing:
            try:
                num = int(path.stem.split("_")[-1])
                numbers.append(num)
            except (ValueError, IndexError):
                continue
        return max(numbers) + 1 if numbers else 0

    def _render_and_save(canvas_name: str, width: int, height: int, operations: list[dict]) -> tuple[bytes, str]:
        """Render canvas and save SVG, return PNG bytes and SVG path."""
        renderer = CanvasRenderer(width, height)
        png_bytes = renderer.render_to_png(operations)

        # Save SVG copy (sanitize name for file safety)
        safe_name = _sanitize_name(canvas_name)
        img_num = _get_next_image_number(safe_name)
        svg_path = store.img_dir / f"{safe_name}_{img_num:03d}.svg"
        svg_content = renderer.render_to_svg(operations)
        svg_path.write_text(svg_content)

        return png_bytes, str(svg_path)

    def _next_auto_name() -> str:
        """Generate next sequential canvas name (canvas_000001, canvas_000002, ...)."""
        import re
        canvases = store.list_entries()
        max_num = 0
        for name, _w, _h, _t in canvases:
            m = re.match(r"canvas_(\d+)$", name)
            if m:
                max_num = max(max_num, int(m.group(1)))
        return f"canvas_{max_num + 1:06d}"

    def _draw_impl(name: str, operations: str) -> dict:
        """Core draw logic shared by canvas_draw and canvas_create."""
        # Fixed canvas size (16:9 aspect ratio)
        width, height = 160, 90

        # Parse JSON string to list
        try:
            parsed_operations = json.loads(operations)
            if not isinstance(parsed_operations, list):
                return {"error": f"Operations must be a JSON array, got: {type(parsed_operations).__name__}"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}. Expected format: '[{{\"type\": \"circle\", \"x\": 50, \"y\": 50, \"r\": 20, \"color\": \"red\"}}]'"}

        # Validate operations
        valid_types = {"pixel", "line", "rect", "circle", "bezier", "blob"}
        required_fields = {
            "pixel": ["x", "y"],
            "line": ["x1", "y1", "x2", "y2"],
            "rect": ["x", "y", "w", "h"],
            "circle": ["x", "y", "r"],
            "bezier": ["start", "c1", "c2", "end"],
            "blob": ["points"],
        }
        # Polar mode required fields
        polar_required_fields = {
            "pixel": ["r", "theta"],
            "line": [],  # Line has multiple polar variants, validated separately
            "rect": ["x", "y", "w", "h"],  # Rect doesn't support polar positioning
            "circle": ["r", "theta"],  # r=distance from center, theta=angle
            "bezier": ["start", "c1", "c2", "end"],  # Uses [r, theta] pairs
            "blob": ["points"],  # Uses [r, theta] pairs
        }
        for i, op in enumerate(parsed_operations):
            if not isinstance(op, dict):
                return {"error": f"Operation {i}: must be object, got {type(op).__name__}"}
            op_type = op.get("type")
            if op_type not in valid_types:
                return {"error": f"Operation {i}: unknown type '{op_type}'. Valid: {valid_types}"}
            # Check required fields based on polar/cartesian mode
            if op.get("polar"):
                missing = [f for f in polar_required_fields.get(op_type, []) if f not in op]
                if missing:
                    return {"error": f"Operation {i} ({op_type}, polar): missing required fields: {missing}"}
            else:
                missing = [f for f in required_fields.get(op_type, []) if f not in op]
                if missing:
                    return {"error": f"Operation {i} ({op_type}): missing required fields: {missing}"}

        # Store operations
        is_update = store.save(name, parsed_operations, width, height)

        # Get full canvas data for rendering
        canvas = store.get_canvas(name)
        png_bytes, svg_path = _render_and_save(name, canvas["width"], canvas["height"], canvas["operations"])

        op_count = len(parsed_operations)
        total_ops = len(canvas["operations"])
        action = "Added" if is_update else "Created"

        return {
            "message": f"{action} {op_count} operations on '{name}' ({canvas['width']}x{canvas['height']}). Total: {total_ops} operations.",
            "image": png_bytes,
            "svg_saved": svg_path,
        }

    @tool
    @with_phenomenology
    def canvas_draw(name: str, operations: str) -> dict:
        """Draw operations on a named canvas (creates if new). Returns rendered image.

        Args:
            name: Canvas name (e.g., "mandala", "sketch")
            operations: JSON string containing array of drawing operations.

        Operations format (as JSON string):
        '[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red", "fill": true}]'

        Shape types:
        - circle: x, y, r, color, fill
        - rect: x, y, w, h, color, fill
        - line: x1, y1, x2, y2, color, width
        - pixel: x, y, color
        - bezier: start, c1, c2, end, color, width (coordinates as [x,y] arrays)
        - blob: points (array of [x,y]), color, fill, smooth

        Example: Draw a red circle at center
        canvas_draw(name="art", operations='[{"type": "circle", "x": 50, "y": 50, "r": 25, "color": "red", "fill": true}]')

        Example: Draw multiple shapes
        canvas_draw(name="art", operations='[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red"}, {"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20, "color": "blue"}]')
        """
        return _draw_impl(name, operations)

    @tool
    @with_phenomenology
    def canvas_create(operations: str) -> dict:
        """Create a new canvas with drawing operations. Each call creates a fresh canvas (automatically named). Returns rendered image.

        Args:
            operations: JSON string containing array of drawing operations.

        Operations format (as JSON string):
        '[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red", "fill": true}]'

        Shape types:
        - circle: x, y, r, color, fill
        - rect: x, y, w, h, color, fill
        - line: x1, y1, x2, y2, color, width
        - pixel: x, y, color
        - bezier: start, c1, c2, end, color, width (coordinates as [x,y] arrays)
        - blob: points (array of [x,y]), color, fill, smooth

        Example: Draw a red circle at center
        canvas_create(operations='[{"type": "circle", "x": 50, "y": 50, "r": 25, "color": "red", "fill": true}]')
        """
        name = _next_auto_name()
        return _draw_impl(name, operations)

    @tool
    @with_phenomenology
    def canvas_read(name: str) -> str:
        """Get text description of a canvas (operations, stats).

        Args:
            name: Canvas name to read
        """
        canvas = store.get_canvas(name)
        if not canvas:
            return f"Canvas '{name}' not found. Use canvas_list() to see available canvases."

        operations = canvas["operations"]
        op_count = len(operations)

        # Count operation types
        type_counts = {}
        colors_used = set()
        for op in operations:
            op_type = op.get("type", "unknown")
            type_counts[op_type] = type_counts.get(op_type, 0) + 1
            if "color" in op:
                colors_used.add(str(op["color"]))

        type_summary = ", ".join([f"{count} {t}" for t, count in sorted(type_counts.items())])

        result = f"""Canvas '{name}' ({canvas['width']}x{canvas['height']})
Created: {canvas['created_at'].isoformat()}
Modified: {canvas['updated_at'].isoformat()}

Summary:
- {op_count} operations: {type_summary or 'none'}
- Colors: {', '.join(sorted(colors_used)) or 'none'}

Operations:
{json.dumps(operations, indent=2)}"""

        return result

    @tool
    @with_phenomenology
    def canvas_view(name: str) -> dict:
        """View a canvas as an image (without modifying it).

        Args:
            name: Canvas name to view
        """
        canvas = store.get_canvas(name)
        if not canvas:
            return {"error": f"Canvas '{name}' not found. Use canvas_list() to see available canvases."}

        # Just render PNG, don't save SVG (view is read-only)
        renderer = CanvasRenderer(canvas["width"], canvas["height"])
        png_bytes = renderer.render_to_png(canvas["operations"])

        return {
            "message": f"Canvas '{name}' ({canvas['width']}x{canvas['height']}), {len(canvas['operations'])} operations",
            "image": png_bytes,
        }

    @tool
    @with_phenomenology
    def canvas_list() -> str:
        """List all canvases with dimensions and timestamps."""
        canvases = store.list_entries()

        if not canvases:
            return "No canvases created yet."

        from datetime import timedelta, timezone
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC to match SQLite CURRENT_TIMESTAMP
        formatted = []
        for name, width, height, updated_at in canvases:
            delta = now - updated_at

            if delta < timedelta(minutes=1):
                time_str = "just now"
            elif delta < timedelta(hours=1):
                mins = int(delta.total_seconds() / 60)
                time_str = f"{mins}m ago"
            elif delta < timedelta(days=1):
                hours = int(delta.total_seconds() / 3600)
                time_str = f"{hours}h ago"
            else:
                days = delta.days
                time_str = f"{days}d ago"

            formatted.append(f"- {name} ({width}x{height}) - updated {time_str}")

        return f"Canvases ({len(canvases)} total):\n" + "\n".join(formatted)

    @tool
    @with_phenomenology
    def canvas_delete(name: str) -> str:
        """Delete a canvas.

        Args:
            name: Canvas name to delete
        """
        deleted = store.delete_canvas(name)
        if deleted:
            return f"Deleted canvas: {name}"
        else:
            return f"Canvas '{name}' not found (already deleted or never existed)."

    @tool
    @with_phenomenology
    def canvas_clear(name: str) -> dict:
        """Clear a canvas (removes all operations, returns to white background).

        Args:
            name: Canvas name to clear
        """
        canvas = store.get_canvas(name)
        if not canvas:
            return {"error": f"Canvas '{name}' not found. Use canvas_list() to see available canvases."}

        store.clear_canvas(name)

        # Re-render with just background
        renderer = CanvasRenderer(canvas["width"], canvas["height"])
        png_bytes = renderer.render_to_png([], scale=10)

        return {
            "message": f"Cleared canvas '{name}'",
            "image": png_bytes,
        }

    all_tools = [canvas_draw, canvas_create, canvas_read, canvas_view, canvas_list, canvas_delete, canvas_clear]
    return all_tools
