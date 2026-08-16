"""Dedicated one-Artifact evidence analyst used by standard RAG."""

from typing import Any


CHUNK_ANALYST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevance": {
            "type": "string",
            "enum": ["relevant", "partial", "irrelevant"],
        },
        "summary": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
        "unresolved_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["relevance", "summary", "claims", "unresolved_questions"],
    "additionalProperties": False,
}


__all__ = ["CHUNK_ANALYST_SCHEMA"]
