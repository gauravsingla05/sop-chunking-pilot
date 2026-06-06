"""Fixed-size chunker — the simplest baseline.

Splits the text stream into windows of approximately TARGET_TOKENS each
with OVERLAP tokens shared between adjacent chunks. Token counts are
estimated via the ~4-chars-per-token rule (see `_base.estimate_tokens`).
No respect for sentence, paragraph, or step boundaries — that's the
point of this baseline.

Defaults match the values reported in the paper (512 / 64).
"""

from __future__ import annotations

from ._base import Chunk, Document, estimate_tokens

NAME = "fixed"
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64
CHARS_PER_TOKEN = 4   # has to match estimate_tokens for self-consistency


def chunk(doc: Document) -> list[Chunk]:
    text = doc.text
    if not text:
        return []
    window_chars = TARGET_TOKENS * CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN
    step = max(1, window_chars - overlap_chars)

    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(text):
        body = text[i : i + window_chars]
        if body.strip():
            chunks.append(Chunk(
                idx=idx, text=body,
                meta={"chunker": NAME, "char_start": i, "char_end": i + len(body),
                      "approx_tokens": estimate_tokens(body)},
            ))
            idx += 1
        i += step
    return chunks
