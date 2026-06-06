"""Structural-fidelity metrics for SOP-to-workflow extraction.

Each metric takes a ground-truth graph (G*) and an extracted graph (Ĝ)
in the same schema as src.extractor.schema.

Ids are NOT shared between G* and Ĝ — gold annotations have their own
ids — so step matching is done by text similarity. We support two
similarity backends (selected by the `use_embeddings` flag, default
on if sentence-transformers is available):

  embeddings (default)  — cosine similarity on sentence-transformers
                          all-MiniLM-L6-v2 normalised embeddings.
                          Tolerates paraphrase (Gemini reflows step text).
                          Threshold default 0.65.

  jaccard (fallback)    — set-Jaccard on word bags. Strict; misses
                          paraphrased pairs. Threshold default 0.55.

Metrics implemented:
  step_order_tau(...)            — Kendall's τ between gold and extracted
                                   step ordering, missing steps penalised.
  precondition_recall(...)       — fraction of gold precondition_of edges
                                   present in extraction.
  constraint_f1(...)             — F1 on constrains edges (presence + correct
                                   target attachment).
  orphan_constraint_rate(...)    — fraction of extracted safety constraints
                                   that have no constrains edge.
  summarize(...)                 — bundle the above into one dict.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("eval.metrics")

_WORD = re.compile(r"\w+")
DEFAULT_JACCARD_THRESHOLD = 0.55
DEFAULT_EMBED_THRESHOLD   = 0.65

# Embedding backend — loaded once, on first use.
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

_ST_MODEL = None
def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None and _HAS_ST:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def _bag(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


# -------------------- alignment --------------------------------------

def _greedy_one_to_one(pairs):
    """pairs: iterable of (sim, gid, pid). Returns dict gid->pid."""
    pairs = sorted(pairs, reverse=True)
    align: dict[str, str] = {}
    used_pred: set[str] = set()
    for _, gid, pid in pairs:
        if gid in align or pid in used_pred:
            continue
        align[gid] = pid
        used_pred.add(pid)
    return align


def _align_by_jaccard(gold_items, pred_items, threshold):
    g = [(s["id"], _bag(s["text"])) for s in gold_items]
    p = [(s["id"], _bag(s["text"])) for s in pred_items]
    pairs = []
    for gid, gb in g:
        for pid, pb in p:
            j = _jaccard(gb, pb)
            if j >= threshold:
                pairs.append((j, gid, pid))
    return _greedy_one_to_one(pairs)


def _align_by_embeddings(gold_items, pred_items, threshold):
    model = _get_st_model()
    if model is None:
        return _align_by_jaccard(gold_items, pred_items, DEFAULT_JACCARD_THRESHOLD)
    gold_texts = [s.get("text") or "" for s in gold_items]
    pred_texts = [s.get("text") or "" for s in pred_items]
    if not gold_texts or not pred_texts:
        return {}
    g_emb = model.encode(gold_texts, normalize_embeddings=True, show_progress_bar=False)
    p_emb = model.encode(pred_texts, normalize_embeddings=True, show_progress_bar=False)
    # Cosine sim matrix; since both are L2-normed, dot product = cosine.
    sims = np.asarray(g_emb) @ np.asarray(p_emb).T
    pairs = []
    for i, gs in enumerate(gold_items):
        for j, ps in enumerate(pred_items):
            s = float(sims[i, j])
            if s >= threshold:
                pairs.append((s, gs["id"], ps["id"]))
    return _greedy_one_to_one(pairs)


def _align_steps(gold_steps, pred_steps, *,
                 use_embeddings=True,
                 threshold=None):
    if use_embeddings and _HAS_ST:
        return _align_by_embeddings(
            gold_steps, pred_steps,
            threshold if threshold is not None else DEFAULT_EMBED_THRESHOLD,
        )
    return _align_by_jaccard(
        gold_steps, pred_steps,
        threshold if threshold is not None else DEFAULT_JACCARD_THRESHOLD,
    )


def _align_clauses(gold_items, pred_items, *,
                   use_embeddings=True,
                   threshold=None):
    return _align_steps(gold_items, pred_items,
                        use_embeddings=use_embeddings,
                        threshold=threshold)


# Back-compat default for any external callers:
DEFAULT_MATCH_THRESHOLD = DEFAULT_EMBED_THRESHOLD


# -------------------- metrics ----------------------------------------

def step_order_tau(gold_graph, pred_graph, *,
                   use_embeddings=True,
                   threshold=None) -> float:
    """Kendall's τ between gold and predicted step orderings.

    Missing gold steps count as max-distance inversions: they sit at the
    END of the predicted order. This penalises chunkers that drop steps
    just as harshly as chunkers that misorder them — which is what we
    want for SOP safety."""
    gold_steps = sorted(gold_graph.get("steps", []),
                        key=lambda s: s.get("ordinal", 0))
    pred_steps = pred_graph.get("steps", [])
    if not gold_steps:
        return 1.0
    align = _align_steps(gold_steps, pred_steps,
                         use_embeddings=use_embeddings, threshold=threshold)

    # Position of each gold step in the predicted order, by aligned id.
    pred_pos = {s["id"]: i for i, s in enumerate(pred_steps)}
    n_pred = len(pred_steps)

    # Build the predicted-rank sequence in gold order. Missing aligns get
    # placed beyond the predicted sequence (worst-case position).
    seq: list[int] = []
    for gs in gold_steps:
        pid = align.get(gs["id"])
        seq.append(pred_pos.get(pid, n_pred) if pid else n_pred)

    n = len(seq)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                concordant += 1
            elif seq[i] > seq[j]:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    return (concordant - discordant) / total_pairs


def precondition_recall(gold_graph, pred_graph, *,
                        use_embeddings=True,
                        threshold=None) -> float:
    """Recall on (precondition_of, step) edges from gold."""
    gold_steps = gold_graph.get("steps", [])
    gold_pre = gold_graph.get("preconditions", [])
    pred_steps = pred_graph.get("steps", [])
    pred_pre = pred_graph.get("preconditions", [])

    gold_edges = [e for e in gold_graph.get("edges", [])
                  if e.get("type") == "precondition_of"]
    if not gold_edges:
        return 1.0   # no preconditions to recall — vacuously perfect

    step_align = _align_steps(gold_steps, pred_steps,
                              use_embeddings=use_embeddings, threshold=threshold)
    pre_align = _align_clauses(gold_pre, pred_pre,
                               use_embeddings=use_embeddings, threshold=threshold)

    pred_pre_edges = {
        (e["from"], e["to"])
        for e in pred_graph.get("edges", [])
        if e.get("type") == "precondition_of"
    }

    recovered = 0
    for e in gold_edges:
        ap = pre_align.get(e["from"])
        as_ = step_align.get(e["to"])
        if ap and as_ and (ap, as_) in pred_pre_edges:
            recovered += 1
    return recovered / len(gold_edges)


def constraint_f1(gold_graph, pred_graph, *,
                  use_embeddings=True,
                  threshold=None) -> dict:
    """F1 on (constrains, step) edges. Returns precision/recall/f1 dict."""
    gold_steps = gold_graph.get("steps", [])
    gold_con = gold_graph.get("constraints", [])
    pred_steps = pred_graph.get("steps", [])
    pred_con = pred_graph.get("constraints", [])

    gold_edges = {(e["from"], e["to"])
                  for e in gold_graph.get("edges", [])
                  if e.get("type") == "constrains"}
    pred_edges = {(e["from"], e["to"])
                  for e in pred_graph.get("edges", [])
                  if e.get("type") == "constrains"}

    step_align = _align_steps(gold_steps, pred_steps,
                              use_embeddings=use_embeddings, threshold=threshold)
    con_align = _align_clauses(gold_con, pred_con,
                               use_embeddings=use_embeddings, threshold=threshold)

    # Project gold edges into predicted-id space using alignments.
    projected_gold = set()
    for gf, gt in gold_edges:
        pf = con_align.get(gf)
        pt = step_align.get(gt)
        if pf and pt:
            projected_gold.add((pf, pt))

    tp = len(projected_gold & pred_edges)
    fp = len(pred_edges - projected_gold)
    fn = len(projected_gold - pred_edges) + len(gold_edges) - len(projected_gold)
    # FN includes gold edges that couldn't even be projected — that's a
    # constraint we failed to recover at all.

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def orphan_constraint_rate(pred_graph) -> float:
    """Fraction of extracted constraints that lack any `constrains` edge.

    Higher = more safety clauses stranded with no attached step. This is
    the metric where a corrupted chunk most obviously shows up."""
    constraints = pred_graph.get("constraints", []) or []
    if not constraints:
        return 0.0
    attached = {e["from"] for e in pred_graph.get("edges", [])
                if e.get("type") == "constrains"}
    orphans = sum(1 for c in constraints if c.get("id") not in attached)
    return orphans / len(constraints)


def step_count_fidelity(gold_graph, pred_graph) -> float:
    """1 − |pred − gold| / max(pred, gold).  Range [0, 1], higher is better.

    Exposes the overgeneration pathology: chunkers that emit one chunk per
    paragraph cause the extractor to declare every paragraph a "step",
    yielding 5-10× more predicted steps than gold. The chunker that
    respects step boundaries (PAC) produces a count close to gold and
    scores near 1.0; over-segmenting baselines score low."""
    n_g = len(gold_graph.get("steps", []) or [])
    n_p = len(pred_graph.get("steps", []) or [])
    if max(n_g, n_p) == 0:
        return 1.0
    return 1.0 - abs(n_p - n_g) / max(n_g, n_p)


def step_precision(gold_graph, pred_graph, *,
                   use_embeddings=True,
                   threshold=None) -> float:
    """Fraction of predicted steps that align to at least one gold step.

    The natural counterpart to step-order τ, which only measures alignment
    of GOLD steps. Together, the two metrics catch both undergeneration
    (low τ) and overgeneration (low precision)."""
    gold_steps = gold_graph.get("steps", []) or []
    pred_steps = pred_graph.get("steps", []) or []
    if not pred_steps:
        return 1.0
    align = _align_steps(gold_steps, pred_steps,
                         use_embeddings=use_embeddings, threshold=threshold)
    aligned_preds = set(align.values())
    return len(aligned_preds) / len(pred_steps)


@dataclass
class MetricResult:
    step_tau: float
    step_precision: float
    step_count_fidelity: float
    precondition_recall: float
    constraint_f1: float
    constraint_precision: float
    constraint_recall: float
    orphan_constraint_rate: float
    n_gold_steps: int
    n_pred_steps: int
    n_gold_constraints: int
    n_pred_constraints: int

    def to_dict(self) -> dict:
        return {
            "step_tau": self.step_tau,
            "step_precision": self.step_precision,
            "step_count_fidelity": self.step_count_fidelity,
            "precondition_recall": self.precondition_recall,
            "constraint_f1": self.constraint_f1,
            "constraint_precision": self.constraint_precision,
            "constraint_recall": self.constraint_recall,
            "orphan_constraint_rate": self.orphan_constraint_rate,
            "n_gold_steps": self.n_gold_steps,
            "n_pred_steps": self.n_pred_steps,
            "n_gold_constraints": self.n_gold_constraints,
            "n_pred_constraints": self.n_pred_constraints,
        }


def summarize(gold_graph, pred_graph, *,
              use_embeddings=True,
              threshold=None) -> MetricResult:
    tau = step_order_tau(gold_graph, pred_graph,
                         use_embeddings=use_embeddings, threshold=threshold)
    sprec = step_precision(gold_graph, pred_graph,
                           use_embeddings=use_embeddings, threshold=threshold)
    scf = step_count_fidelity(gold_graph, pred_graph)
    pre_rec = precondition_recall(gold_graph, pred_graph,
                                  use_embeddings=use_embeddings, threshold=threshold)
    cf1 = constraint_f1(gold_graph, pred_graph,
                        use_embeddings=use_embeddings, threshold=threshold)
    orph = orphan_constraint_rate(pred_graph)
    return MetricResult(
        step_tau=tau,
        step_precision=sprec,
        step_count_fidelity=scf,
        precondition_recall=pre_rec,
        constraint_f1=cf1["f1"],
        constraint_precision=cf1["precision"],
        constraint_recall=cf1["recall"],
        orphan_constraint_rate=orph,
        n_gold_steps=len(gold_graph.get("steps", []) or []),
        n_pred_steps=len(pred_graph.get("steps", []) or []),
        n_gold_constraints=len(gold_graph.get("constraints", []) or []),
        n_pred_constraints=len(pred_graph.get("constraints", []) or []),
    )
