"""
schemas for request json
"""

class Schemas:
    addimage_schema = {
        "type":"object",
            "required": ["image_path"],
            "properties": {
                "image_path": {
                    "type": "array",
                    "items": {
                        "type":"string"
                    },
                    "minItems": 1
                },
                "image_tag": {
                    "type": "string"
                },
                "group": {
                    "type": "string",
                    "enum": ["good", "bad", "mask"]
                }
            }
        }
    
    train_schema = {
        "type": "object",
        "oneOf": [
            {
                "required": ["image_id"]
            },
            {
                "required": ["image_tag"]
            }
        ],
        "properties": {
            "image_id": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "minItems": 1
            },
            "image_tag": {
                "type": "string"
            },
            "model_tag": {
                "type": "string"
            },
            "parameters": {
                "type": "object",
                "additionalProperties": {
                    "type": ["string", "object"]
                }
            }
        }
    }

    predict_schema = {
        "type": "object",
        "oneOf": [
            {
                "required": ["model_id", "image_paths"]
            },
            {
                "required": ["tag", "image_paths"]
            }
        ],
        "properties": {
            "model_id": {
                "type": "number"
            },
            "tag": {
                "type": "string"
            },
            "image_paths": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 1
            },
            "parameters": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "save": {
                "type": "string",
                "enum": ["none", "all", "ok_only", "ng_only"]
            }
        }
    }

    predict_as_file_schema = {
        "type": "object",
        "oneOf": [
            {
                "required": ["model_id"]
            },
            {
                "required": ["tag"]
            }
        ],
        "properties": {
            "model_id": {
                "type": "number"
            },
            "tag": {
                "type": "string"
            },
            "parameters": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "save": {
                "type": "string",
                "enum": ["none", "all", "ok_only", "ng_only"]
            }
        }
    }