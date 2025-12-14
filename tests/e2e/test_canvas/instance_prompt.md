# Canvas Tool Test

You are testing the canvas tools. Execute these operations in order:

1. Use canvas_draw with name="test_canvas" and operations='[{"type": "circle", "x": 50, "y": 50, "r": 20, "color": "red", "fill": true}]'
2. Use canvas_list to see all canvases
3. Use canvas_draw with name="test_canvas" and operations='[{"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20, "color": "blue", "fill": true}]'
4. Use canvas_read with name="test_canvas" to see operations
5. Use canvas_view with name="test_canvas" to see the image
6. Use canvas_clear with name="test_canvas"
7. Use canvas_delete with name="test_canvas"

After completing all canvas operations, use the stop tool with message "Canvas test complete".

IMPORTANT: The operations parameter must be a JSON string, not a list object.
Do NOT use send_message.
