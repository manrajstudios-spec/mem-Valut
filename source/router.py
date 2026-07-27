import json
from call_model import ask_model

with open("Data/router_prompt.text",'r') as file:
    sys_prompt = file.read()

router_schema = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": (
                "A short natural-language topic describing what the user "
                "is currently discussing, usually 1 to 5 words."
            ),
            "minLength": 1
        },

        "web_search": {
            "type": "object",
            "properties": {
                "required": {
                    "type": "boolean"
                },
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1
                    },
                    "maxItems": 3
                }
            },
            "required": [
                "required",
                "queries"
            ],
            "additionalProperties": False
        },

        "memory": {
            "type": "object",
            "properties": {
                "required": {
                    "type": "boolean"
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "events",
                            "facts",
                            "preferences",
                            "goals",
                            "decisions",
                            "pending_tasks"
                        ]
                    },
                    "uniqueItems": True
                },
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1
                    },
                    "maxItems": 3
                }
            },
            "required": [
                "required",
                "categories",
                "queries"
            ],
            "additionalProperties": False
        },

        "project_memory": {
            "type": "object",
            "properties": {
                "required": {
                    "type": "boolean"
                },
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1
                    },
                    "uniqueItems": True
                },
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1
                    },
                    "maxItems": 3
                }
            },
            "required": [
                "required",
                "projects",
                "queries"
            ],
            "additionalProperties": False
        },

        "tool_use": {
            "type": "object",
            "properties": {
                "required": {
                    "type": "boolean",
                    "const": False
                },
                "tools": {
                    "type": "array",
                    "maxItems": 0
                }
            },
            "required": [
                "required",
                "tools"
            ],
            "additionalProperties": False
        },

        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        }
    },

    "required": [
        "topic",
        "web_search",
        "memory",
        "project_memory",
        "tool_use",
        "confidence"
    ],

    "additionalProperties": False
}

def route_msg(hist):
    json_out = ask_model([{"role":sys_prompt}] + hist,router_schema)
    parsed = json.loads(json_out)
    