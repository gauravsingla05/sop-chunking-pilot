"""Recursive character chunker — LangChain-style hierarchical splitting.

Tries to split on the largest separator first ("\\n\\n"), falls back to
"\\n", then ". ", then " ", then character. Within each split, recursively
collapses pieces that are still over budget. This is the default chunker
in most production RAG pipelines and a competitive baseline.

We implement it from scratch (rather than importing LangChain) so the
project has a minimal dependency footprint and the behavior is pinned.
"""

from __future__ import annotations

from ._base import Chunk, Document, estimate_tokens

NAME = "recursive"
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64
CHARS_PER_TOKEN = 4

# Order matters — try the largest semantic break first.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_with_separator(text: str, sep: str) -> list[str]:
    if sep == "":
        # Final fallback: char-level split.
        return list(text)
    parts: list[str] = []
    cur = 0
    while True:
        i = text.find(sep, cur)
        if i < 0:
            parts.append(text[cur:])
            break
        # Keep the separator attached to the left part so reassembly is lossless.
        parts.append(text[cur : i + len(sep)])
        cur = i + len(sep)
    return [p for p in parts if p]


def _split_recursive(text: str, target_chars: int, sep_idx: int = 0) -> list[str]:
    """Greedy join of pieces produced by SEPARATORS[sep_idx] up to target_chars.
    Pieces themselves longer than target_chars are recursively split with the
    next separator."""
    if len(text) <= target_chars:
        return [text]
    if sep_idx >= len(SEPARATORS):
        # Hard fallback: take target_chars chunks.
        return [text[i : i + target_chars] for i in range(0, len(text), target_chars)]

    parts = _split_with_separator(text, SEPARATORS[sep_idx])
    out: list[str] = []
    buf = ""
    for p in parts:
        if len(p) > target_chars:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_recursive(p, target_chars, sep_idx + 1))
            continue
        if len(buf) + len(p) <= target_chars:
            buf += p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def chunk(doc: Document) -> list[Chunk]:
    text = doc.text
    if not text:
        return []
    target_chars = TARGET_TOKENS * CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN

    pieces = _split_recursive(text, target_chars)

    # Apply overlap: glue the last N chars of each chunk onto the next.
    chunks: list[Chunk] = []
    prev_tail = ""
    for idx, piece in enumerate(pieces):
        body = (prev_tail + piece) if prev_tail else piece
        chunks.append(Chunk(
            idx=idx, text=body,
            meta={"chunker": NAME, "approx_tokens": estimate_tokens(body)},
        ))
        prev_tail = body[-overlap_chars:] if overlap_chars > 0 else ""
    return chunks
