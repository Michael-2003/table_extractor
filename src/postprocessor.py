from typing import List
import re

def _cleanup_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u200f", "").replace("\u200e", "")  # strip bidi marks
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_matrix(cells: List[List[dict]]):
    """Return a rectangular list[list[str]] from assigned cells (no typing)."""
    return [[ _cleanup_text(c.get("value") or "") for c in row ] for row in cells]
