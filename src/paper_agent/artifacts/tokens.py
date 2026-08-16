"""Shared, deterministic token estimation used by policies and views."""

import re

TOKEN_PATTERN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Estimate token count with a stable offline approximation."""
    return len(TOKEN_PATTERN.findall(text))
