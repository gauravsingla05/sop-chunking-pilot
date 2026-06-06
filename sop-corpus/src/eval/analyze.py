"""Headline analysis of one pilot run.

Reads:
  - data/pilot.csv                              (doc list)
  - data/runs/<run-id>/oracle/<sha>.json        (per-doc ground truth)
  - data/runs/<run-id>/merged/<sha>_<chunk>.json (per-cell predictions)

Computes per-doc-per-chunker metrics (using src.eval.metrics) and prints:
  1. Per-chunker mean/median across docs for every metric.
  2. Paired Wilcoxon signed-rank: PAC vs each baseline, every metric.
  3. Per-doc breakdown so the reader can see where PAC helps most.

Writes:
  data/runs/<run-id>/analysis/per_cell.csv
  data/runs/<run-id>/analysis/per_chunker_summary.csv
  data/runs/<run-id>/analysis/pac_vs_baselines.csv
  data/runs/<run-id>/analysis/headline.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from src.eval.metrics import summarize

HERE = Path(__file__).resolve().parents[2]   # sop-corpus/

CHUNKERS = ["fixed", "recursive", "semantic", "layout_aware", "pac"]
METRICS = [
    "step_tau",
    "step_precision",
    "step_count_fidelity",
    "precondition_recall",
    "constraint_f1",
    "orphan_constraint_rate",
]


def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float]:
    """Tiny paired-test implementation so we don't need scipy.
    Returns (W, two-sided-p-approximation).
    Uses normal approximation valid for n >= 6."""
    paired = [d for d in deltas if d != 0]
    n = len(paired)
    if n == 0:
        return 0.0, 1.0
    abs_sorted = sorted([(abs(d), d) for d in paired])
    # Rank with average for ties.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_sorted[j + 1][0] == abs_sorted[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    w_plus = sum(r for r, (_, d) in zip(ranks, abs_sorted) if d > 0)
    w_minus = sum(r for r, (_, d) in zip(ranks, abs_sorted) if d < 0)
    w = min(w_plus, w_minus)
    # Normal approximation
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return w, 1.0
    z = (w - mean) / (var ** 0.5)
    # Two-sided p from |z| via the Phi approximation:
    # erf approximation
    import math
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return w, p


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="pilot-001-gemini")
    p.add_argument("--pilot", default="data/pilot.csv")
    args = p.parse_args(argv)

    run_dir = HERE / "data" / "runs" / args.run_id
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 2

    pilot_csv = HERE / args.pilot
    pilot_rows = list(csv.DictReader(open(pilot_csv)))

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell: list[dict] = []
    skipped = 0

    for row in pilot_rows:
        sha = row["sha256"]
        oracle_path = run_dir / "oracle" / f"{sha}.json"
        if not oracle_path.exists():
            print(f"  no oracle for {row['title'][:40]} — skipping", file=sys.stderr)
            skipped += 1
            continue
        oracle = json.loads(oracle_path.read_text())
        gold = oracle.get("graph") or {}
        if not gold.get("steps"):
            print(f"  oracle empty for {row['title'][:40]} — skipping", file=sys.stderr)
            skipped += 1
            continue

        for cn in CHUNKERS:
            merged_path = run_dir / "merged" / f"{sha}_{cn}.json"
            if not merged_path.exists():
                continue
            pred = json.loads(merged_path.read_text())
            m = summarize(gold, pred).to_dict()
            per_cell.append({
                "doc_sha": sha[:10],
                "doc_title": row["title"][:60],
                "doc_source": row["source"],
                "doc_pages": row["page_count"],
                "chunker": cn,
                **m,
            })

    if not per_cell:
        print("no cells found in run", file=sys.stderr)
        return 2

    # ---------------- Write per-cell CSV ----------------
    cols = ["doc_sha", "doc_title", "doc_source", "doc_pages", "chunker"] + METRICS + [
        "constraint_precision", "constraint_recall",
        "n_gold_steps", "n_pred_steps",
        "n_gold_constraints", "n_pred_constraints",
    ]
    with (out_dir / "per_cell.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_cell:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---------------- Per-chunker summary ----------------
    by_chunker: dict[str, list[dict]] = defaultdict(list)
    for r in per_cell:
        by_chunker[r["chunker"]].append(r)

    summary_rows = []
    for cn in CHUNKERS:
        cells = by_chunker.get(cn, [])
        if not cells:
            continue
        row = {"chunker": cn, "n_cells": len(cells)}
        for m in METRICS:
            vals = [c[m] for c in cells]
            row[f"{m}_mean"] = round(statistics.fmean(vals), 4)
            row[f"{m}_median"] = round(statistics.median(vals), 4)
        summary_rows.append(row)

    sum_cols = ["chunker", "n_cells"] + [
        c for m in METRICS for c in (f"{m}_mean", f"{m}_median")
    ]
    with (out_dir / "per_chunker_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_cols)
        w.writeheader()
        w.writerows(summary_rows)

    # ---------------- PAC vs baselines, paired ----------------
    pac_by_doc = {r["doc_sha"]: r for r in by_chunker.get("pac", [])}
    paired_rows = []
    for baseline in ["fixed", "recursive", "semantic", "layout_aware"]:
        for m in METRICS:
            deltas = []
            for r in by_chunker.get(baseline, []):
                pacr = pac_by_doc.get(r["doc_sha"])
                if not pacr:
                    continue
                # delta = PAC - baseline. Positive means PAC wins for higher-is-better;
                # for orphan_constraint_rate we flip sign so positive still means "PAC wins".
                delta = pacr[m] - r[m]
                if m == "orphan_constraint_rate":
                    delta = -delta
                deltas.append(delta)
            if not deltas:
                continue
            w, p = wilcoxon_signed_rank(deltas)
            paired_rows.append({
                "baseline": baseline,
                "metric": m,
                "n_paired": len(deltas),
                "pac_mean_delta": round(statistics.fmean(deltas), 4),
                "pac_wins": sum(1 for d in deltas if d > 0),
                "pac_ties": sum(1 for d in deltas if d == 0),
                "pac_losses": sum(1 for d in deltas if d < 0),
                "wilcoxon_W": round(w, 2),
                "wilcoxon_p": round(p, 4),
            })
    paired_cols = ["baseline", "metric", "n_paired", "pac_mean_delta",
                   "pac_wins", "pac_ties", "pac_losses", "wilcoxon_W", "wilcoxon_p"]
    with (out_dir / "pac_vs_baselines.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=paired_cols)
        w.writeheader()
        w.writerows(paired_rows)

    # ---------------- Headline text ----------------
    lines = []
    lines.append(f"Run: {args.run_id}    docs: {len(pilot_rows)}    cells: {len(per_cell)}    skipped: {skipped}")
    lines.append("")
    lines.append("Per-chunker mean of each structural-fidelity metric (higher τ/recall/F1 is better; lower orphan is better):")
    lines.append("")
    lines.append(f"  {'chunker':<14} {'n':>3}  {'τ':>6} {'sprec':>6} {'count':>6} {'pre_rec':>8} {'cons_F1':>8} {'orphan':>7}")
    for r in summary_rows:
        lines.append(
            f"  {r['chunker']:<14} {r['n_cells']:>3}  "
            f"{r['step_tau_mean']:>6.3f} "
            f"{r['step_precision_mean']:>6.3f} "
            f"{r['step_count_fidelity_mean']:>6.3f} "
            f"{r['precondition_recall_mean']:>8.3f} "
            f"{r['constraint_f1_mean']:>8.3f} "
            f"{r['orphan_constraint_rate_mean']:>7.3f}"
        )
    lines.append("")
    lines.append("PAC vs each baseline (paired Wilcoxon signed-rank; Δ = PAC − baseline;")
    lines.append("for orphan_constraint_rate we negate so '+Δ' always means PAC wins):")
    lines.append("")
    lines.append(f"  {'baseline':<14} {'metric':<22} {'n':>3} {'Δ':>8} {'W':>6} {'pac:tie:base':>13} {'p':>7}")
    for r in paired_rows:
        sig = "*" if r['wilcoxon_p'] < 0.05 else " "
        lines.append(
            f"  {r['baseline']:<14} {r['metric']:<22} {r['n_paired']:>3} "
            f"{r['pac_mean_delta']:>+8.4f} {r['wilcoxon_W']:>+6.1f} "
            f"  {r['pac_wins']:>2}:{r['pac_ties']:>2}:{r['pac_losses']:<2}  "
            f"{r['wilcoxon_p']:>6.4f}{sig}"
        )
    lines.append("")
    headline = "\n".join(lines)
    (out_dir / "headline.txt").write_text(headline)
    print(headline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
