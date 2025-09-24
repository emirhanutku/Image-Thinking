TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "OCR",
            "description": "Extracts any text from an image (such as axis labels or annotations). If no text is present, returns an empty string. Note: the text may not always be accurate or in order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "the image from which to extract text"},
                },
                "required": ["image"],
                "additionalProperties": False,
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Point",
            "description": "Identifies and marks a specific point in the image based on a description, such as a value on the x or y axis. Returns the coordinates of the identified point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string","description" :"the image to identify the point in"},
                    "param": {"type": "string", "description": "description of the object to locate"},
                },
                "required": ["image", "param"],
                "additionalProperties": False,
            }
        },
    },
    {
        "type":"function",
        "function":{
            "name": "ZoomInSubfigure",
            "description": "Crops the image to zoom in on a specified subfigure, useful for focusing on smaller areas of interest.",
            "parameters":{
                "type":"object",
                "properties":{
                    "image":{"type":"string","description":"the image to crop from"},
                    "param":{"type":"string","description":"description of the subfigure to zoom into"},
                },
                "required": ["image", "param"],
                "additionalProperties": False,
            }
        },
    },
     {
        "type": "function",
        "function": {
            "name": "SegmentRegionAroundPoint",
            "description": "Creates a mask or segments a region around specified coordinates, useful for isolating areas on charts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string","description":"the image to segment"},
                    "param": {
                        "anyOf": [
                            {"type": "array", "items": {"type":"number"}, "minItems": 2, "maxItems": 2, "description": "coordinates around which to segment, e.g., [x, y]"},
                            {"type": "object", "properties": {"x":{"type":"number"},"y":{"type":"number"}}, "required":["x","y"], "description": "coordinates around which to segment, e.g., {'x': 100, 'y': 200}"}
                        ]
                    },
                },
                "required": ["image", "param"],
                "additionalProperties": False,
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DrawHorizontalLineByY",
            "description": "Draws a horizontal line at a specific y-value in the image. Used for comparing or verifying y-values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string","description":"the image to draw the line on"},
                    "param": {"type":"number","description":"coordinates with the y-value to draw the horizontal line"}
                },
                "required": ["image", "param"],
                "additionalProperties": False,
            }
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "DrawVerticalLineByX",
            "description": "Draws a vertical line at a specific x-value in the image. Used for comparing or verifying x-values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string","description":"the image to draw the line on"},
                    "param": {"type":"number","description":"coordinates with the x-value to draw the vertical line"},
                },
                "required": ["image", "param"],
                "additionalProperties": False,
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Terminate",
            "description": "Concludes the task and provides the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ans": {"type": "string","description":"the final answer to the question being addressed"},
                },
                "required": ["ans"],
                "additionalProperties": False,
            }
        },
    },
]