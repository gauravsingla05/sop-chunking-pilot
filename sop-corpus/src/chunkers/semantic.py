"""Semantic chunker — embedding-breakpoint clustering.

Walks the document one sentence at a time, embeds each, and inserts a
chunk boundary whenever the cosine distance to the previous sentence
sits above a percentile threshold of all neighbouring-pair distances in
the document. Greg Kamradt popularised this strategy; we mirror it.

Embedding backend: sentence-transformers (small, all-MiniLM-L6-v2) so we
don't need a network round-trip per sentence. Falls back to a TF-IDF
similarity if sentence-transformers isn't installed — strictly worse, but
keeps the baseline runnable on a fresh checkout.
"""

from __future__ import annotations

import re
import logging

import numpy as np

from ._base import Chunk, Document, estimate_tokens

log = logging.getLogger("chunkers.semantic")

NAME = "semantic"
PERCENTILE = 95            # cut when distance is in the top (100-p)% of pairs
MIN_CHARS = 200            # don't emit chunks smaller than this
MAX_TOKENS = 700           # soft cap; if a "semantic" chunk gets huge, split it

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = None

    def _get_st_model():
        global _ST_MODEL
        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _ST_MODEL

    def _embed(sentences: list[str]) -> np.ndarray:
        return _get_st_model().encode(sentences, normalize_embeddings=True,
                                      show_progress_bar=False)
    _BACKEND = "sentence-transformers"
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    def _embed(sentences: list[str]) -> np.ndarray:
        # Char-ngram tfidf — degrades gracefully on tiny inputs.
        v = TfidfVectorizer(ngram_range=(1, 2), max_features=4096, lowercase=True)
        m = v.fit_transform(sentences).astype(float).toarray()
        return normalize(m, axis=1)
    _BACKEND = "tfidf-fallback"


_SENT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\(])")


def _split_sentences(text: str) -> list[str]:
    # Page-break markers are sentinels we don't want to embed.
    text = re.sub(r"<<<PAGE_BREAK \d+>>>", "\n", text)
    # Cheap sentence splitter.
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_RE.split(para):
            s = s.strip()
            if s:
                out.append(s)
    return out


def chunk(doc: Document) -> list[Chunk]:
    sentences = _split_sentences(doc.text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(idx=0, text=sentences[0],
                      meta={"chunker": NAME, "backend": _BACKEND, "approx_tokens": estimate_tokens(body)})]

    try:
        embs = _embed(sentences)
    except Exception as e:
        log.warning("embed failed (%s) — falling back to single chunk", e)
        return [Chunk(idx=0, text=doc.text, meta={"chunker": NAME, "error": str(e)})]

    # Cosine distances between consecutive sentences (normalised embeddings ->
    # dot product = cosine similarity).
    sims = (embs[:-1] * embs[1:]).sum(axis=1)
    dists = 1.0 - sims
    threshold = np.percentile(dists, PERCENTILE)

    chunks: list[Chunk] = []
    cur: list[str] = []
    idx = 0
    for i, s in enumerate(sentences):
        cur.append(s)
        # If next-distance exceeds threshold, cut here.
        if i < len(dists) and dists[i] >= threshold:
            body = " ".join(cur).strip()
            if len(body) >= MIN_CHARS:
                chunks.append(Chunk(idx=idx, text=body,
                                    meta={"chunker": NAME, "backend": _BACKEND, "approx_tokens": estimate_tokens(body)}))
                idx += 1
                cur = []
    if cur:
        body = " ".join(cur).strip()
        if body:
            chunks.append(Chunk(idx=idx, text=body,
                                meta={"chunker": NAME, "backend": _BACKEND, "approx_tokens": estimate_tokens(body)}))

    # Enforce a soft max-tokens — if a "semantic" chunk ballooned, slice it
    # so the extractor's context isn't blown.
    final: list[Chunk] = []
    next_idx = 0
    for c in chunks:
        toks = estimate_tokens(c.text)
        if toks <= MAX_TOKENS:
            c.idx = next_idx
            final.append(c)
            next_idx += 1
            continue
        slice_chars = MAX_TOKENS * 4
        for j in range(0, len(c.text), slice_chars):
            piece = c.text[j : j + slice_chars]
            final.append(Chunk(idx=next_idx, text=piece,
                               meta={**c.meta, "split_oversize": True}))
            next_idx += 1
    return final
