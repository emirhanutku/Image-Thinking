
ACTIONS_ENUM = [
    "OCR",
    "Point",
    "ZoomInSubfigure",
    "SegmentRegionAroundPoint",
    "DrawHorizontalLineByY",
    "DrawVerticalLineByX",
    "Terminate",
]

PLANNING_SCHEMA = {
    "name": "VisionPlan",
    "schema": {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": ACTIONS_ENUM},
                        "arguments": {
                            "type": "object",

                            "properties": {
                                "image": {"type": "string"},
                                "param":  {"type": "string"},
                                "ans":    {"type": ["string","number"]},
                            },
                            "additionalProperties": True
                        }
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False
                },
                "minItems": 1
            }
        },
        "required": ["thought", "actions"],
        "additionalProperties": False
    },
}
  

RATIONALE_SCHEMA = {
    "name": "tool_rationale",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "problem_signature": {"type": "string"},
            "tools_sequence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool": {"type": "string"},
                        "purpose": {"type": "string"},
                        "inputs_needed": {"type": "array", "items": {"type": "string"}},
                        "outputs_produced": {"type": "array", "items": {"type": "string"}},
                        "preconditions": {"type": "array", "items": {"type": "string"}},
                        "why_order": {"type": "string"},
                        "success_criteria": {"type": "array", "items": {"type": "string"}},
                        "fallbacks": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["tool", "purpose", "why_order"]
                }
            },
            "global_rules": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["problem_signature", "tools_sequence"]
    },
    "strict": False
}
