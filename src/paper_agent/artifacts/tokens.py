"""Shared, deterministic token estimation used by policies and views."""

import re

TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|"
    r"[A-Za-z0-9_]+|"
    r"[^\w\s]|"
    r"[^\W\d_]+",
    re.UNICODE,
)


def count_tokens(text: str) -> int:
    """Estimate token count with a stable offline approximation."""
    return len(TOKEN_PATTERN.findall(text))
