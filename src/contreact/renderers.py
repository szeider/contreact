"""Rendering system for tool calls and messages."""

from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class BaseRenderer(ABC):
    """Abstract base class for tool-specific renderers."""

    tool_name: Optional[str] = None

    @abstractmethod
    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render a tool invocation.

        Args:
            ai_message: The AIMessage containing the tool call
            tool_call: The specific tool call dict from ai_message.tool_calls
            console: Rich console for output
        """
        pass

    @abstractmethod
    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render the tool's response.

        Args:
            tool_message: The ToolMessage with the tool's response
            console: Rich console for output
        """
        pass


class RendererRegistry:
    """Registry for tool-specific renderers."""

    _renderers: dict[str, BaseRenderer] = {}

    @classmethod
    def register(cls, renderer: BaseRenderer):
        """Register a renderer for its tool.

        Args:
            renderer: Renderer instance to register
        """
        if renderer.tool_name:
            cls._renderers[renderer.tool_name] = renderer

    @classmethod
    def get_renderer(cls, tool_name: str) -> BaseRenderer:
        """Get renderer for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Renderer instance (GenericRenderer if tool not registered)
        """
        return cls._renderers.get(tool_name, GenericRenderer())


def register_renderer(cls):
    """Decorator to auto-register renderer classes.

    Usage:
        @register_renderer
        class MyToolRenderer(BaseRenderer):
            tool_name = "my_tool"
            ...
    """
    RendererRegistry.register(cls())
    return cls


@register_renderer
class SendMessageRenderer(BaseRenderer):
    """Renderer for send_message tool."""

    tool_name = "send_message"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render agent's outgoing message."""
        message = tool_call.get("args", {}).get("message", "")

        panel = Panel(
            Text(message, style="dark_cyan"),
            title="🗣️  Agent",
            border_style="dark_cyan",
            padding=(1, 2)
        )
        console.print(panel)

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render user's response."""
        content = tool_message.content

        # Extract actual response from "User responded: ..." format
        if content.startswith("User responded: "):
            content = content[16:]  # Strip prefix

        panel = Panel(
            Text(content, style="blue"),
            title="👤 User",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)


@register_renderer
class ThinkRenderer(BaseRenderer):
    """Renderer for think tool."""

    tool_name = "think"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render agent's internal thought."""
        thought = tool_call.get("args", {}).get("thought", "")

        panel = Panel(
            Text(thought, style="dark_cyan"),
            title="💭 Agent Thinking",
            border_style="dark_cyan",
            padding=(1, 2)
        )
        console.print(panel)

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Think tool response is not interesting, skip it."""
        pass


@register_renderer
class StopRenderer(BaseRenderer):
    """Renderer for stop tool."""

    tool_name = "stop"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render agent's final message before stopping."""
        final_message = tool_call.get("args", {}).get("final_message", "")

        panel = Panel(
            Text(final_message, style="red"),
            title="🛑 Agent Stopped",
            border_style="red",
            padding=(1, 2)
        )
        console.print(panel)

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Stop tool doesn't have a meaningful response."""
        pass


@register_renderer
class CanvasDrawRenderer(BaseRenderer):
    """Renderer for canvas_draw tool."""

    tool_name = "canvas_draw"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render canvas draw operation."""
        args = tool_call.get("args", {})
        name = args.get("name", "unnamed")
        operations = args.get("operations", [])

        # Summarize operations
        op_summary = {}
        for op in operations:
            op_type = op.get("type", "unknown")
            op_summary[op_type] = op_summary.get(op_type, 0) + 1

        summary = ", ".join([f"{count} {t}" for t, count in op_summary.items()])
        content = f"Canvas: {name}\nOperations: {summary or 'none'}"

        panel = Panel(
            Text(content, style="magenta"),
            title="🎨 Canvas Draw",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(panel)

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render canvas draw response."""
        content = tool_message.content
        # Content is multimodal (text + image), extract text part
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(text_parts) if text_parts else "[Image rendered]"

        panel = Panel(
            Text(str(content), style="magenta"),
            title="🎨 Canvas Updated",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(panel)


@register_renderer
class CanvasViewRenderer(BaseRenderer):
    """Renderer for canvas_view tool."""

    tool_name = "canvas_view"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render canvas view request."""
        args = tool_call.get("args", {})
        name = args.get("name", "unnamed")

        panel = Panel(
            Text(f"Viewing canvas: {name}", style="magenta"),
            title="👁️ Canvas View",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(panel)

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render canvas view response."""
        content = tool_message.content
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(text_parts) if text_parts else "[Image rendered]"

        panel = Panel(
            Text(str(content), style="magenta"),
            title="👁️ Canvas Viewed",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(panel)


@register_renderer
class CanvasReadRenderer(BaseRenderer):
    """Renderer for canvas_read tool."""

    tool_name = "canvas_read"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render canvas read request."""
        args = tool_call.get("args", {})
        name = args.get("name", "unnamed")

        console.print(f"[bright_black]📋 Reading canvas: {name}[/]")

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render canvas read response (truncated)."""
        content = str(tool_message.content)

        # Show summary only (first few lines)
        lines = content.split("\n")
        summary = "\n".join(lines[:8])
        if len(lines) > 8:
            summary += f"\n... ({len(lines) - 8} more lines)"

        panel = Panel(
            Text(summary, style="magenta"),
            title="📋 Canvas Info",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(panel)


@register_renderer
class CanvasListRenderer(BaseRenderer):
    """Renderer for canvas_list tool."""

    tool_name = "canvas_list"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render canvas list request."""
        console.print("[bright_black]📋 Listing canvases...[/]")

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render canvas list response."""
        content = str(tool_message.content)

        panel = Panel(
            Text(content, style="magenta"),
            title="📋 Canvases",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(panel)


@register_renderer
class CanvasClearRenderer(BaseRenderer):
    """Renderer for canvas_clear tool."""

    tool_name = "canvas_clear"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render canvas clear request."""
        args = tool_call.get("args", {})
        name = args.get("name", "unnamed")
        color = args.get("color", "white")

        console.print(f"[bright_black]🧹 Clearing canvas '{name}' to {color}[/]")

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render canvas clear response."""
        content = tool_message.content
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(text_parts) if text_parts else "Canvas cleared"

        console.print(f"[magenta]✅ {content}[/]")


@register_renderer
class CanvasDeleteRenderer(BaseRenderer):
    """Renderer for canvas_delete tool."""

    tool_name = "canvas_delete"

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render canvas delete request."""
        args = tool_call.get("args", {})
        name = args.get("name", "unnamed")

        console.print(f"[bright_black]🗑️ Deleting canvas: {name}[/]")

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render canvas delete response."""
        console.print(f"[magenta]✅ {tool_message.content}[/]")


class GenericRenderer(BaseRenderer):
    """Fallback renderer for unknown tools."""

    tool_name = None

    def render_call(self, ai_message: AIMessage, tool_call: dict, console: Console):
        """Render generic tool invocation."""
        tool_name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})

        # Format args nicely
        args_str = "\n".join(f"  {k}: {v}" for k, v in args.items())
        content = f"Tool: {tool_name}\n\n{args_str}"

        panel = Panel(
            Text(content, style="blue"),
            title="🔧 Tool Call",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)

    def render_response(self, tool_message: ToolMessage, console: Console):
        """Render generic tool response."""
        content = str(tool_message.content)

        # Truncate very long responses
        if len(content) > 500:
            content = content[:500] + "\n... (truncated)"

        panel = Panel(
            Text(content, style="blue"),
            title="🔧 Tool Response",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)


def render_message(message, console: Console):
    """Render any message type.

    Args:
        message: LangChain message to render
        console: Rich console for output
    """
    if isinstance(message, SystemMessage):
        # Don't display system message (too verbose)
        pass

    elif isinstance(message, AIMessage):
        # AIMessage contains tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get("name")
                renderer = RendererRegistry.get_renderer(tool_name)
                renderer.render_call(message, tool_call, console)

    elif isinstance(message, ToolMessage):
        # ToolMessage is the response from a tool
        # Need to figure out which tool it came from
        # ToolMessage has a 'name' attribute
        tool_name = getattr(message, "name", None)
        if tool_name:
            renderer = RendererRegistry.get_renderer(tool_name)
            renderer.render_response(message, console)
        else:
            # Fallback to generic
            GenericRenderer().render_response(message, console)

    elif isinstance(message, HumanMessage):
        # This shouldn't happen in our system (user input comes via tool response)
        # but handle it anyway
        panel = Panel(
            Text(message.content, style="white"),
            title="👤 Human",
            border_style="white"
        )
        console.print(panel)
