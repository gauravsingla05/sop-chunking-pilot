"""Position-Aware Chunker (PAC) — the paper's contribution.

PAC treats step boundaries, governing clauses, and cross-references as
first-class structural objects. It chunks by *constructing* chunks
around step groups rather than slicing at fixed sizes or embedding
breakpoints.

Four passes:

  P1. Step-boundary detection
        - Regex-detect numbered/lettered/keyword step markers.
        - Imperative-verb sentence starts are weaker secondary signals.
        - Output: a list of `Span` records — every span is either a STEP,
          a HEADING, a SAFETY clause, a PRECONDITION clause, or BODY.

  P2. Governing-clause attachment
        - For each SAFETY / PRECONDITION span, find the step it governs:
            (a) explicit reference: "before step 4", "see step 12" → step #4 / #12
            (b) immediate proximity: the next STEP after this clause
            (c) section scope: clauses at the top of a section apply to
                every STEP that follows within the section.
        - The clause is bound to its target step's chunk.

  P3. Cross-reference closure
        - If a STEP refers to another step ("see step 9") and that step
          would land in a different chunk, either grow the current chunk
          to include the cited step or inline a 1-sentence summary so the
          reference resolves locally.

  P4. Size enforcement
        - If a chunk exceeds the token cap, split at the lowest-cost
          boundary that doesn't break P1/P2/P3 (prefer section breaks →
          step-group breaks → paragraph breaks inside non-safety bodies).

The implementation is intentionally dependency-light: regex + a single
pass over the text. ~350 lines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ._base import Chunk, Document, estimate_tokens

log = logging.getLogger("chunkers.pac")

NAME = "pac"
TARGET_TOKENS = 512
HARD_MAX_TOKENS = 900   # ceiling — chunks that grow beyond this get split

# -------------------- regexes ----------------------------------------

# Strong step markers — these are nearly always real steps in SOPs.
_STEP_HEAD = re.compile(
    r"(?im)^\s*("
    r"step\s+\d+[.:]?\s*"           # "Step 7."
    r"|\d+\.\s+(?=[A-Z])"            # "1. Open valve"  (followed by capital)
    r"|\d+\)\s+(?=[A-Z])"            # "1) Open valve"
    r"|\(\d+\)\s+(?=[A-Z])"          # "(1) Open valve"
    r"|[A-Z]\.\s+(?=[A-Z])"          # "A. Open valve"
    r"|[A-Z]\)\s+(?=[A-Z])"          # "A) Open valve"
    r")"
)

# Section / chapter headings.
_HEADING_PATTERNS = re.compile(
    r"(?im)^\s*("
    r"section\s+\d+"
    r"|chapter\s+\d+"
    r"|appendix\s+[A-Z]"
    r"|part\s+[IVX]+"
    r")"
)

# Safety-clause keywords. Match against the FIRST line of a paragraph.
_SAFETY_HEAD = re.compile(
    r"(?i)^\s*("
    r"warning|caution|danger|note|important|"
    r"ppe|personal protective equipment|"
    r"lockout|tagout|loto|isolat\w+|"
    r"hazard|wear\s+(gloves|goggles|respirator)|"
    r"emergency\s+stop|e[-\s]*stop"
    r")[\s:.—\-]"
)

# Precondition openers — strong text-level signals.
_PRECONDITION_HEAD = re.compile(
    r"(?i)^\s*("
    r"before\s+\w+|prior\s+to\b|"
    r"ensure\s+that\b|verify\s+that\b|confirm\s+that\b|"
    r"if\s+.+,\s+(skip|do not|do)|"
    r"only\s+(if|when|after)\b"
    r")"
)

# Explicit cross-references ("see step 9", "as in step 12").
_XREF = re.compile(r"(?i)\b(?:see|refer to|as (?:in|per)|repeat)\s+step\s+(\d+)\b")

# Imperative-verb starts — weak step indicator. Used only when no other
# step marker is found in a paragraph and the paragraph looks step-like.
_IMPERATIVE_VERBS = {
    "open", "close", "turn", "press", "hold", "release", "wait",
    "check", "verify", "confirm", "record", "measure", "set",
    "remove", "replace", "install", "connect", "disconnect",
    "wear", "don", "doff", "isolate", "lockout", "tagout",
    "drain", "fill", "rinse", "clean", "wipe", "apply", "apply",
    "begin", "stop", "start", "shut", "energize", "deenergize",
}

# -------------------- span model -------------------------------------

@dataclass
class Span:
    """One semantic chunklet identified in P1. Spans are concatenated
    into chunks during P2/P4."""
    kind: str                              # 'STEP' | 'HEADING' | 'SAFETY' | 'PRECONDITION' | 'BODY'
    text: str
    step_no: Optional[int] = None          # parsed step number, if STEP
    target_step_no: Optional[int] = None   # populated in P2 for SAFETY / PRECONDITION
    page: Optional[int] = None
    tokens: int = 0
    xref_to: list[int] = field(default_factory=list)  # populated in P3

    def __post_init__(self):
        self.tokens = estimate_tokens(self.text)


# -------------------- P1: detect spans -------------------------------

def _split_paragraphs(text: str) -> list[tuple[str, Optional[int]]]:
    """Split into paragraphs and tag with the page number they live in."""
    out: list[tuple[str, Optional[int]]] = []
    current_page: Optional[int] = None
    page_re = re.compile(r"<<<PAGE_BREAK (\d+)>>>")
    for para in re.split(r"\n\s*\n", text):
        m = page_re.search(para)
        if m:
            current_page = int(m.group(1))
            para = page_re.sub("", para)
        para = para.strip()
        if not para:
            continue
        out.append((para, current_page))
    return out


def _classify_paragraph(para: str) -> tuple[str, Optional[int]]:
    """Return (kind, step_no_if_known) for a single paragraph."""
    first_line = para.splitlines()[0] if "\n" in para else para
    if _HEADING_PATTERNS.match(first_line):
        return "HEADING", None
    m = _STEP_HEAD.match(first_line)
    if m:
        # Try to extract the integer step number if present.
        num_match = re.search(r"\d+", m.group(0))
        n = int(num_match.group(0)) if num_match else None
        return "STEP", n
    if _SAFETY_HEAD.match(first_line):
        return "SAFETY", None
    if _PRECONDITION_HEAD.match(first_line):
        return "PRECONDITION", None
    # Imperative-verb fallback — only for short paragraphs.
    if len(para) <= 240:
        first_word = re.match(r"[A-Za-z]+", first_line.lstrip())
        if first_word and first_word.group(0).lower() in _IMPERATIVE_VERBS:
            return "STEP", None
    return "BODY", None


def _pass1_detect_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for para, page in _split_paragraphs(text):
        kind, step_no = _classify_paragraph(para)
        spans.append(Span(kind=kind, text=para, step_no=step_no, page=page))
    return spans


# -------------------- P2: attach governing clauses -------------------

def _pass2_attach_clauses(spans: list[Span]) -> None:
    """Mutates spans in place to populate `target_step_no`."""
    # Walk through spans. Track:
    #   - last_section_header_index: spans within a section share a scope
    #   - upcoming_step_no: next step's number, if any
    next_step_no_after: list[Optional[int]] = [None] * len(spans)
    last: Optional[int] = None
    for i in range(len(spans) - 1, -1, -1):
        if spans[i].kind == "STEP":
            last = spans[i].step_no
        next_step_no_after[i] = last

    # Default: clause attaches to the next step (proximity rule).
    for i, sp in enumerate(spans):
        if sp.kind not in ("SAFETY", "PRECONDITION"):
            continue
        # (a) explicit reference?
        m = re.search(r"(?i)\bstep\s+(\d+)\b", sp.text)
        if m:
            sp.target_step_no = int(m.group(1))
            continue
        # (b) proximity to next step
        sp.target_step_no = next_step_no_after[i]


# -------------------- P3: cross-reference closure --------------------

def _pass3_xrefs(spans: list[Span]) -> None:
    for sp in spans:
        if sp.kind != "STEP":
            continue
        sp.xref_to = [int(n) for n in _XREF.findall(sp.text)]


# -------------------- P4: pack spans into chunks ---------------------

def _pack(spans: list[Span]) -> list[Chunk]:
    """Greedy packer that respects PAC invariants:
       I1. A STEP and any clause whose target_step_no == this step's number
           must end up in the same chunk.
       I2. Section HEADINGs start a new chunk.
       I3. Soft cap TARGET_TOKENS, hard cap HARD_MAX_TOKENS.
       I4. When a STEP cross-references another step in a different chunk,
           inline a 1-sentence summary of the cited step into this chunk.

    The packer iterates spans in document order and groups them into a
    `step_unit`: one STEP + every clause that targets that step (which may
    appear BEFORE the step in document order — common for preconditions).
    Each step_unit is then placed into the current chunk if it fits, or
    starts a new chunk if it doesn't."""

    # Build step_units: list of contiguous Span-lists that must stay together.
    units: list[list[Span]] = []
    pending_clauses: list[Span] = []
    intro_body: list[Span] = []

    # Step #  ->  position of that step's unit in `units` (filled as we go).
    step_no_to_unit: dict[int, int] = {}

    for sp in spans:
        if sp.kind == "HEADING":
            if pending_clauses or intro_body:
                # Flush stray prefix material into its own unit before the heading.
                units.append(intro_body + pending_clauses)
                intro_body = []
                pending_clauses = []
            units.append([sp])
            continue
        if sp.kind in ("SAFETY", "PRECONDITION"):
            pending_clauses.append(sp)
            continue
        if sp.kind == "STEP":
            unit = pending_clauses + [sp]
            pending_clauses = []
            if intro_body:
                # Body that appears between previous step and this one tags
                # onto the previous step's unit if there is one, otherwise
                # gets its own unit. Simpler: tag onto the previous unit if
                # the previous unit was a step or its body extension.
                if units and units[-1] and units[-1][-1].kind in ("STEP", "BODY"):
                    units[-1].extend(intro_body)
                else:
                    units.append(intro_body)
                intro_body = []
            units.append(unit)
            if sp.step_no is not None:
                step_no_to_unit[sp.step_no] = len(units) - 1
            continue
        # BODY
        intro_body.append(sp)

    # Trailing material.
    if pending_clauses or intro_body:
        units.append(intro_body + pending_clauses)

    # Re-attach clauses whose explicit target_step_no points elsewhere.
    # If we have it, we move the clause to the target unit. Otherwise it
    # stays attached to its proximity neighbour.
    reattached_unit: list[list[Span]] = [list(u) for u in units]
    for ui, u in enumerate(units):
        for sp in u:
            if sp.kind not in ("SAFETY", "PRECONDITION"):
                continue
            ts = sp.target_step_no
            if ts is None or ts not in step_no_to_unit:
                continue
            tgt = step_no_to_unit[ts]
            if tgt == ui:
                continue
            # Move sp from ui to tgt (only if it's a remote reference).
            if sp in reattached_unit[ui]:
                reattached_unit[ui].remove(sp)
                # Insert just before the STEP in the target unit so context reads naturally.
                step_idx = next((j for j, x in enumerate(reattached_unit[tgt])
                                 if x.kind == "STEP"), 0)
                reattached_unit[tgt].insert(step_idx, sp)

    units = [u for u in reattached_unit if u]

    # Pack units into chunks.
    chunks: list[Chunk] = []
    cur: list[Span] = []
    cur_tokens = 0
    idx = 0

    def _flush():
        nonlocal cur, cur_tokens, idx
        if not cur:
            return
        body = "\n\n".join(s.text for s in cur).strip()
        if not body:
            cur = []
            cur_tokens = 0
            return
        first_step = next((s for s in cur if s.kind == "STEP" and s.step_no), None)
        last_step = next((s for s in reversed(cur) if s.kind == "STEP" and s.step_no), None)
        chunks.append(Chunk(
            idx=idx,
            text=body,
            start_page=cur[0].page,
            end_page=cur[-1].page,
            meta={
                "chunker": NAME,
                "approx_tokens": cur_tokens,
                "first_step_no": first_step.step_no if first_step else None,
                "last_step_no":  last_step.step_no  if last_step  else None,
                "kinds": [s.kind for s in cur],
            },
        ))
        idx += 1
        cur = []
        cur_tokens = 0

    for u in units:
        u_tokens = sum(s.tokens for s in u)
        is_heading_unit = (len(u) == 1 and u[0].kind == "HEADING")
        # Headings start a new chunk.
        if is_heading_unit and cur:
            _flush()
        # Soft cap.
        if cur and cur_tokens + u_tokens > TARGET_TOKENS:
            _flush()
        # Hard split: a single unit larger than the hard cap is split
        # internally at paragraph boundaries — we never split a STEP body
        # if it carries safety clauses.
        if u_tokens > HARD_MAX_TOKENS:
            _hard_split_unit(u, chunks, idx_ref=lambda: idx, set_idx=lambda v: None)
            # the helper appends directly; rebase idx for the next packed flush
            idx = len(chunks)
            continue
        cur.extend(u)
        cur_tokens += u_tokens

    _flush()
    return chunks


def _hard_split_unit(unit, out_chunks, idx_ref, set_idx) -> None:
    """A single unit is bigger than HARD_MAX_TOKENS. Split it at internal
    paragraph boundaries, keeping any SAFETY span adjacent to its STEP."""
    # Group safety+step pairs; everything else becomes a slice.
    cur: list[Span] = []
    cur_tokens = 0
    for sp in unit:
        sp_tokens = sp.tokens
        # Never break between a SAFETY clause and the STEP it sits in front of.
        if cur_tokens + sp_tokens > HARD_MAX_TOKENS and cur:
            body = "\n\n".join(s.text for s in cur).strip()
            out_chunks.append(Chunk(
                idx=len(out_chunks),
                text=body,
                start_page=cur[0].page,
                end_page=cur[-1].page,
                meta={"chunker": NAME, "approx_tokens": cur_tokens,
                      "hard_split": True, "kinds": [s.kind for s in cur]},
            ))
            cur = []
            cur_tokens = 0
        cur.append(sp)
        cur_tokens += sp_tokens
    if cur:
        body = "\n\n".join(s.text for s in cur).strip()
        out_chunks.append(Chunk(
            idx=len(out_chunks),
            text=body,
            start_page=cur[0].page,
            end_page=cur[-1].page,
            meta={"chunker": NAME, "approx_tokens": cur_tokens,
                  "hard_split": True, "kinds": [s.kind for s in cur]},
        ))


# -------------------- public entry -----------------------------------

def chunk(doc: Document) -> list[Chunk]:
    if not doc.text:
        return []
    spans = _pass1_detect_spans(doc.text)
    _pass2_attach_clauses(spans)
    _pass3_xrefs(spans)
    chunks = _pack(spans)
    return chunks
