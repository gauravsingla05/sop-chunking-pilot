"""Merge per-chunk extractor outputs into one document-level workflow graph.

Each chunk's extractor returns a self-contained graph with chunk-local IDs
(s1, s2, p1, c1, ...). The merger:

  1. Renames every node to a globally-unique id (cN_sM).
  2. Concatenates steps in chunk order and assigns a global ordinal.
  3. Detects overlapping (duplicate) steps across adjacent chunks and
     merges them — this matters because every baseline chunker uses some
     amount of overlap, and most preconditions/constraints near a chunk
     boundary will appear in BOTH chunks.
  4. Unions preconditions and constraints with the same dedup approach.
  5. Remaps edge endpoints to global ids; drops edges whose endpoints
     don't resolve.

Duplicate detection: Jaccard similarity on lower-cased word sets, with a
threshold (0.78) tuned for short imperative sentences. Cheap, dependency-
free, and good enough for the chunk-overlap case where the same sentence
is verbatim in both chunks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("pipeline.merge")

_WORD = re.compile(r"\w+")
DUP_THRESHOLD = 0.78


def _bag(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


@dataclass
class MergedGraph:
    """The document-level graph produced by `merge()`. Same shape as one
    chunk's graph but with globally-unique ids and ordered steps."""
    steps: list[dict]
    preconditions: list[dict]
    constraints: list[dict]
    edges: list[dict]
    # Bookkeeping for the eval pipeline.
    duplicate_count: int = 0
    dropped_edge_count: int = 0


def merge(chunk_graphs: list[dict]) -> MergedGraph:
    """chunk_graphs: list of per-chunk graphs in chunk order."""
    # ---- 1) namespace every id ---------------------------------------
    namespaced: list[dict] = []
    for ci, g in enumerate(chunk_graphs):
        ns = {"steps": [], "preconditions": [], "constraints": [], "edges": []}
        renames: dict[str, str] = {}

        def ren(old: str, kind: str) -> str:
            new = f"c{ci}_{old}"
            renames[old] = new
            return new

        for s in g.get("steps") or []:
            new_id = ren(s.get("id", "s?"), "s")
            ns["steps"].append({
                "id": new_id,
                "ordinal": s.get("ordinal", 0),
                "text": s.get("text", "") or "",
                "page": s.get("page"),
                "_chunk": ci,
            })
        for p in g.get("preconditions") or []:
            new_id = ren(p.get("id", "p?"), "p")
            ns["preconditions"].append({
                "id": new_id,
                "text": p.get("text", "") or "",
                "_chunk": ci,
            })
        for c in g.get("constraints") or []:
            new_id = ren(c.get("id", "c?"), "c")
            ns["constraints"].append({
                "id": new_id,
                "text": c.get("text", "") or "",
                "kind": c.get("kind", "other") or "other",
                "_chunk": ci,
            })
        for e in g.get("edges") or []:
            f, t = e.get("from"), e.get("to")
            if f in renames:
                f = renames[f]
            if t in renames:
                t = renames[t]
            ns["edges"].append({
                "type": e.get("type"),
                "from": f,
                "to": t,
                "_chunk": ci,
            })
        namespaced.append(ns)

    # ---- 2) merge steps in chunk order, dedup against the tail of the
    #         previous chunk (overlap zone) ----------------------------
    merged_steps: list[dict] = []
    id_alias: dict[str, str] = {}   # old_id -> canonical_id
    dup_count = 0

    for ci, ns in enumerate(namespaced):
        # Dedup window = last K steps already in merged_steps.
        WINDOW = 12
        tail = merged_steps[-WINDOW:] if merged_steps else []
        tail_bags = [(s["id"], _bag(s["text"])) for s in tail]

        for s in ns["steps"]:
            sb = _bag(s["text"])
            best_id = None
            best_sim = 0.0
            for tid, tb in tail_bags:
                j = _jaccard(sb, tb)
                if j > best_sim:
                    best_sim = j
                    best_id = tid
            if best_id and best_sim >= DUP_THRESHOLD:
                id_alias[s["id"]] = best_id
                dup_count += 1
                continue
            merged_steps.append(s)

    # Assign final ordinals.
    for i, s in enumerate(merged_steps, start=1):
        s["ordinal"] = i

    # ---- 3) merge preconditions and constraints with the same logic --
    def _merge_clauses(key: str) -> list[dict]:
        out: list[dict] = []
        for ns in namespaced:
            for c in ns[key]:
                cb = _bag(c["text"])
                matched = None
                for prev in reversed(out[-30:]):       # short tail
                    j = _jaccard(cb, _bag(prev["text"]))
                    if j >= DUP_THRESHOLD:
                        matched = prev["id"]
                        break
                if matched:
                    id_alias[c["id"]] = matched
                else:
                    out.append(c)
        return out

    merged_preconditions = _merge_clauses("preconditions")
    merged_constraints = _merge_clauses("constraints")

    # ---- 4) merge edges; remap aliased endpoints; drop danglers ------
    known_ids = {x["id"] for x in merged_steps}
    known_ids |= {x["id"] for x in merged_preconditions}
    known_ids |= {x["id"] for x in merged_constraints}

    seen: set[tuple[str, str, str]] = set()
    merged_edges: list[dict] = []
    dropped = 0

    # Also synthesise `precedes` edges from the merged step ordering.
    for a, b in zip(merged_steps, merged_steps[1:]):
        merged_edges.append({"type": "precedes", "from": a["id"], "to": b["id"]})
        seen.add(("precedes", a["id"], b["id"]))

    for ns in namespaced:
        for e in ns["edges"]:
            f = id_alias.get(e["from"], e["from"])
            t = id_alias.get(e["to"], e["to"])
            if f not in known_ids or t not in known_ids:
                dropped += 1
                continue
            tup = (e["type"], f, t)
            if tup in seen:
                continue
            seen.add(tup)
            merged_edges.append({"type": e["type"], "from": f, "to": t})

    log.debug("merge: chunks=%d steps=%d preconds=%d constraints=%d edges=%d dup=%d dropped=%d",
              len(chunk_graphs), len(merged_steps),
              len(merged_preconditions), len(merged_constraints),
              len(merged_edges), dup_count, dropped)

    return MergedGraph(
        steps=merged_steps,
        preconditions=merged_preconditions,
        constraints=merged_constraints,
        edges=merged_edges,
        duplicate_count=dup_count,
        dropped_edge_count=dropped,
    )
