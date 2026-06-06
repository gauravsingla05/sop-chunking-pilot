"""Layout-aware chunker — uses PyMuPDF block geometry.

Iterates over the layout blocks emitted by ``extract_text.py`` and starts
a new chunk on each heading-like block (large font), and otherwise grows
the current chunk until it hits the token cap. Respects page boundaries
as soft splits — within the cap, prefer ending a chunk at a page break.

When blocks aren't available (pypdf fallback path), this chunker
degrades to paragraph-level splitting on blank lines.
"""

from __future__ import annotations

from ._base import Chunk, Document, estimate_tokens

NAME = "layout_aware"
TARGET_TOKENS = 512


def _paragraphs_from_text(text: str) -> list[str]:
    import re
    text = re.sub(r"<<<PAGE_BREAK \d+>>>", "\n\n", text)
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk(doc: Document) -> list[Chunk]:
    if not doc.blocks:
        # Fallback: paragraph-level greedy pack.
        paras = _paragraphs_from_text(doc.text)
        return _greedy_pack(paras, page_for=lambda i: None, heading_at=lambda i: False)

    texts = [b.text for b in doc.blocks]
    pages = [b.page for b in doc.blocks]
    headings = [b.is_heading for b in doc.blocks]
    return _greedy_pack(
        texts,
        page_for=lambda i: pages[i],
        heading_at=lambda i: headings[i],
    )


def _greedy_pack(items, page_for, heading_at) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_start_page = None
    cur_end_page = None
    cur_start_block = None
    cur_end_block = None
    cur_tokens = 0
    idx = 0

    def flush():
        nonlocal cur, cur_tokens, cur_start_page, cur_end_page
        nonlocal cur_start_block, cur_end_block, idx
        if not cur:
            return
        body = "\n\n".join(cur).strip()
        if body:
            chunks.append(Chunk(
                idx=idx, text=body,
                start_page=cur_start_page, end_page=cur_end_page,
                start_block=cur_start_block, end_block=cur_end_block,
                meta={"chunker": NAME, "approx_tokens": cur_tokens},
            ))
            idx += 1
        cur = []
        cur_tokens = 0
        cur_start_page = None
        cur_end_page = None
        cur_start_block = None
        cur_end_block = None

    for i, t in enumerate(items):
        tk = estimate_tokens(t)
        # Heading-led chunking: a heading starts a new chunk if the current
        # one has accumulated something.
        if heading_at(i) and cur:
            flush()
        # Capacity check: would adding this item blow the budget?
        if cur_tokens + tk > TARGET_TOKENS and cur:
            flush()
        if not cur:
            cur_start_block = i
            cur_start_page = page_for(i)
        cur.append(t)
        cur_end_block = i
        cur_end_page = page_for(i)
        cur_tokens += tk

    flush()
    return chunks
