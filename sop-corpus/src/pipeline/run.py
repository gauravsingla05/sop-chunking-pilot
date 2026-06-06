"""End-to-end pipeline runner.

For a (chunker × document) cell:
   chunk -> per-chunk extraction -> merge -> evaluate -> write JSON.

Idempotent at the cell level: re-running skips cells whose output JSON
already exists. Deletes need to be explicit. This lets a long run resume
cleanly after a network blip or a Ctrl-C.

Outputs (under sop-corpus/data/runs/<run_id>/):
   chunks/<sha>_<chunker>.json        ← chunk texts + provenance
   chunk_graphs/<sha>_<chunker>.json  ← per-chunk extractor outputs
   merged/<sha>_<chunker>.json        ← document-level extracted graph
   metrics.jsonl                       ← one line per (doc, chunker) with all metrics
   oracle/<sha>.json                   ← oracle "ground-truth" graph

Usage:
    # subscription-backed run, 20-chunk sample per cell
    python -m src.pipeline.run --pilot data/pilot.csv --sample-chunks 20

    # dry-run (no LLM calls — for shape debugging only)
    python -m src.pipeline.run --pilot data/pilot.csv --dry-run

    # one chunker, one doc — fastest debug loop
    python -m src.pipeline.run --pilot data/pilot.csv --only-chunker pac \
            --only-sha f7d91ded58525990386db506e69461a1c68a670b80eff3dfc7715815d006610c
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.chunkers import ALL as ALL_CHUNKERS
from src.chunkers._base import Document, Block
from src.extractor import extract_from_chunk
from src.extractor.llm import BACKEND as EXTRACTOR_BACKEND
from src.pipeline.merge import merge
from src.eval.metrics import summarize

log = logging.getLogger("pipeline.run")

HERE = Path(__file__).resolve().parents[2]   # sop-corpus/
DATA = HERE / "data"
RUNS = DATA / "runs"

ORACLE_PROMPT_NOTE = """\
ORACLE PASS — produce the ground-truth workflow graph for the document
below. Treat the input as the source SOP (or, when truncated, its main
content section).

Be thorough, not conservative:

- A "step" is any imperative or expected action: numbered steps, lettered
  steps, bullet actions, or directive sentences like "Verify that...",
  "Determine whether...", "Inspect...", "Confirm...". Include all of them.
- Preconditions include gates like "Before X, ensure Y", "If A, then B",
  "Only when C", or any clause that conditions a step's execution.
- Constraints include any safety-relevant clause: PPE, lockout/tagout,
  isolation, warning/caution/danger statements, hazard advisories.
- If a document is a checklist, inspection procedure, or technical
  manual chapter, extract its action items as steps.
- Steps should be in document order with ordinals 1..N.

Return ONLY the JSON object that matches the workflow-graph schema. No
prose, no markdown fences. Empty arrays are allowed only if the document
genuinely contains no procedural content; otherwise extract everything
present.
"""


# -------------------- doc load -----------------------------------------

def load_doc(row: dict) -> Optional[Document]:
    sha = row["sha256"]
    tdir = DATA / "text" / sha
    if not tdir.exists():
        log.warning("no extracted text for sha=%s — did you run extract_text.py?", sha[:10])
        return None
    text = (tdir / "text.txt").read_text(encoding="utf-8")
    blocks: list[Block] = []
    blocks_file = tdir / "blocks.jsonl"
    if blocks_file.exists():
        for line in blocks_file.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            blocks.append(Block(
                page=d["page"], bbox=tuple(d["bbox"]),
                text=d["text"], font_size=d["font_size"],
                is_heading=d["is_heading"],
            ))
    return Document(
        sha256=sha, source=row["source"], title=row["title"],
        text=text, blocks=blocks, text_dir=tdir,
    )


# -------------------- per-cell run -------------------------------------

def _run_one_cell(
    *,
    doc: Document,
    chunker_name: str,
    sample_chunks: Optional[int],
    seed: int,
    dry_run: bool,
    out_dir: Path,
) -> dict:
    """Run one (doc, chunker) cell. Returns the metrics row written to disk."""
    chunker = ALL_CHUNKERS[chunker_name]
    short = doc.sha256[:10]

    chunks_file = out_dir / "chunks"      / f"{doc.sha256}_{chunker_name}.json"
    cg_file     = out_dir / "chunk_graphs" / f"{doc.sha256}_{chunker_name}.json"
    merged_file = out_dir / "merged"      / f"{doc.sha256}_{chunker_name}.json"

    for d in (chunks_file.parent, cg_file.parent, merged_file.parent):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: chunk (always re-run, cheap).
    t0 = time.time()
    chunks = chunker.chunk(doc)
    chunk_time = time.time() - t0

    # Sample for cost control.
    rng = random.Random(seed)
    if sample_chunks and len(chunks) > sample_chunks:
        sampled_idx = sorted(rng.sample(range(len(chunks)), sample_chunks))
        sampled_chunks = [chunks[i] for i in sampled_idx]
    else:
        sampled_idx = list(range(len(chunks)))
        sampled_chunks = chunks

    chunks_file.write_text(json.dumps({
        "doc_sha": doc.sha256,
        "doc_title": doc.title,
        "chunker": chunker_name,
        "n_chunks_total": len(chunks),
        "n_chunks_sampled": len(sampled_chunks),
        "sampled_idx": sampled_idx,
        "chunks": [{"idx": c.idx, "tokens": c.meta.get("approx_tokens"),
                    "text": c.text, "start_page": c.start_page,
                    "end_page": c.end_page, "meta": c.meta}
                   for c in sampled_chunks],
    }, ensure_ascii=False), encoding="utf-8")

    # Step 2: extract per chunk (resumable).
    if cg_file.exists():
        cg = json.loads(cg_file.read_text(encoding="utf-8"))
    else:
        cg = {"doc_sha": doc.sha256, "chunker": chunker_name, "graphs": []}

    done_idx = {g["chunk_idx"] for g in cg["graphs"]}
    extractor_secs = 0.0
    in_tokens = out_tokens = cache_tokens = 0
    errors = 0
    for c in sampled_chunks:
        if c.idx in done_idx:
            continue
        t1 = time.time()
        r = extract_from_chunk(
            c.text,
            chunk_idx=c.idx,
            doc_title=doc.title,
            dry_run=dry_run,
        )
        extractor_secs += time.time() - t1
        cg["graphs"].append({
            "chunk_idx": c.idx,
            "graph": r.graph,
            "error": r.error,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cached_tokens": r.cached_tokens,
        })
        in_tokens += (r.input_tokens or 0)
        out_tokens += (r.output_tokens or 0)
        cache_tokens += (r.cached_tokens or 0)
        if r.error:
            errors += 1
            log.warning("[%s/%s/chunk=%d] extractor error: %s",
                        short, chunker_name, c.idx, r.error)
        # Persist after every call so a crash doesn't lose work.
        cg_file.write_text(json.dumps(cg, ensure_ascii=False), encoding="utf-8")

    # Step 3: merge per-chunk graphs into one doc-level graph (sample-aware).
    cg["graphs"].sort(key=lambda g: g["chunk_idx"])
    merged = merge([g["graph"] for g in cg["graphs"]])
    merged_dict = {
        "steps": merged.steps,
        "preconditions": merged.preconditions,
        "constraints": merged.constraints,
        "edges": merged.edges,
        "_duplicate_count": merged.duplicate_count,
        "_dropped_edge_count": merged.dropped_edge_count,
    }
    merged_file.write_text(json.dumps(merged_dict, ensure_ascii=False), encoding="utf-8")

    return {
        "doc_sha": doc.sha256,
        "doc_title": doc.title,
        "doc_source": doc.source,
        "chunker": chunker_name,
        "n_chunks_total": len(chunks),
        "n_chunks_sampled": len(sampled_chunks),
        "extractor_errors": errors,
        "chunk_time_s": round(chunk_time, 2),
        "extractor_time_s": round(extractor_secs, 2),
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "cached_tokens": cache_tokens,
        "n_steps_extracted": len(merged.steps),
        "n_preconds_extracted": len(merged.preconditions),
        "n_constraints_extracted": len(merged.constraints),
        "n_edges_extracted": len(merged.edges),
    }


# -------------------- oracle -------------------------------------------

def _oracle_pass(*, doc: Document, dry_run: bool, out_dir: Path) -> dict:
    """Run a single extractor call on the WHOLE doc to produce a silver-
    standard ground-truth graph. Truncates very long docs to keep one call
    feasible — documented as a methodology limitation."""
    out_file = out_dir / "oracle" / f"{doc.sha256}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))

    # Cap input at ~400K chars (~100k tokens). Gemini Flash handles 1M+
    # tokens, so this is comfortable. The original 30K cap landed entirely
    # on table-of-contents / front matter for longer SOPs and produced
    # empty oracle graphs.
    MAX_CHARS = 400_000
    text = doc.text if len(doc.text) <= MAX_CHARS else doc.text[:MAX_CHARS]
    truncated = len(doc.text) > MAX_CHARS

    # Oracle outputs the whole document's graph at once, so they need a
    # bigger output budget than a per-chunk call. The default 4000 was
    # silently truncating long-doc oracle responses into broken JSON.
    r = extract_from_chunk(
        ORACLE_PROMPT_NOTE + "\n\n" + text,
        chunk_idx=-1,
        doc_title=doc.title,
        max_tokens=16000,
        dry_run=dry_run,
    )
    payload = {
        "doc_sha": doc.sha256,
        "doc_title": doc.title,
        "truncated_input": truncated,
        "graph": r.graph,
        "error": r.error,
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


# -------------------- main ---------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", default="data/pilot.csv",
                   help="CSV of SOPs to run on (default: data/pilot.csv)")
    p.add_argument("--run-id", default="pilot-001",
                   help="subdir under data/runs/ for this experiment")
    p.add_argument("--sample-chunks", type=int, default=20,
                   help="cap chunks per doc-chunker cell (None = all). default 20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-chunker", default=None, help="run just this chunker")
    p.add_argument("--only-sha", default=None, help="run just this doc")
    p.add_argument("--skip-oracle", action="store_true",
                   help="don't run the oracle pass (useful when iterating chunkers only)")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the pipeline without LLM calls")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info("extractor backend: %s | dry_run=%s | sample_chunks=%s",
             EXTRACTOR_BACKEND, args.dry_run, args.sample_chunks)

    pilot_csv = Path(args.pilot)
    if not pilot_csv.is_absolute():
        pilot_csv = HERE / pilot_csv
    rows = list(csv.DictReader(open(pilot_csv)))
    if args.only_sha:
        rows = [r for r in rows if r["sha256"] == args.only_sha]
    log.info("pilot rows: %d", len(rows))

    chunker_names = (
        [args.only_chunker] if args.only_chunker else list(ALL_CHUNKERS.keys())
    )

    run_dir = RUNS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = run_dir / "metrics.jsonl"

    all_metrics: list[dict] = []

    for row in rows:
        doc = load_doc(row)
        if doc is None:
            continue
        log.info("doc: %s  (%s, %d chars)", doc.title[:60], doc.source, len(doc.text))

        # Oracle pass (once per doc).
        oracle = None
        if not args.skip_oracle:
            oracle = _oracle_pass(doc=doc, dry_run=args.dry_run, out_dir=run_dir)
            if oracle.get("error"):
                log.warning("  oracle error: %s", oracle["error"])

        for cn in chunker_names:
            cell_t0 = time.time()
            row_metric = _run_one_cell(
                doc=doc, chunker_name=cn,
                sample_chunks=args.sample_chunks,
                seed=args.seed, dry_run=args.dry_run,
                out_dir=run_dir,
            )
            row_metric["wall_clock_s"] = round(time.time() - cell_t0, 2)

            # Compute structural metrics against the oracle graph.
            if oracle and oracle.get("graph"):
                merged_path = run_dir / "merged" / f"{doc.sha256}_{cn}.json"
                pred = json.loads(merged_path.read_text(encoding="utf-8"))
                m = summarize(oracle["graph"], pred)
                row_metric.update({"metric_" + k: v for k, v in m.to_dict().items()})

            log.info(
                "  [%s] sampled=%d/%d errors=%d steps=%d cons=%d wall=%.1fs",
                cn, row_metric["n_chunks_sampled"], row_metric["n_chunks_total"],
                row_metric["extractor_errors"], row_metric["n_steps_extracted"],
                row_metric["n_constraints_extracted"], row_metric["wall_clock_s"],
            )

            all_metrics.append(row_metric)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row_metric, ensure_ascii=False) + "\n")

    log.info("done: wrote %d metric rows to %s", len(all_metrics), metrics_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
