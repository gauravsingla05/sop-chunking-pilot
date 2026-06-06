"""Shared types for chunkers.

Every chunker implements:

    chunk(doc: Document) -> list[Chunk]

where Document carries the extracted text and (optionally) layout blocks.
Chunks have a small canonical shape so the extractor + merger don't have
to special-case different chunker outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.extract_text import Block  # re-export so chunkers don't import private path


@dataclass
class Document:
    """One SOP in the pipeline."""
    sha256: str
    source: str                  # e.g. "osha", "nrc"
    title: str
    text: str                    # full extracted stream (with <<<PAGE_BREAK n>>> markers)
    blocks: list[Block] = field(default_factory=list)   # layout-aware; may be []
    text_dir: Optional[Path] = None   # data/text/<sha256>/


@dataclass
class Chunk:
    """One produced chunk."""
    idx: int                     # 0-based position in the chunk sequence
    text: str
    # Optional provenance — useful for the merger when chunks span pages.
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    # Optional provenance — block id range when the chunker is layout-aware.
    start_block: Optional[int] = None
    end_block: Optional[int] = None
    # Any chunker-specific metadata for debugging / failure analysis.
    meta: dict = field(default_factory=dict)


def estimate_tokens(s: str) -> int:
    """Coarse but deterministic token estimate without tiktoken.

    Good enough for chunker sizing; we don't need tokenizer-exact counts
    because chunk-size targets here are advisory."""
    if not s:
        return 0
    # ~4 chars per token is the long-running rule-of-thumb for English.
    return max(1, len(s) // 4)
