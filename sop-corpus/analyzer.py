"""PDF analysis: page count, text extraction, structural signals.

Kept dependency-light: pypdf for text + page count, langdetect for language.
"""

from __future__ import annotations

import hashlib
import io
import re
from typing import Optional

try:
    from pypdf import PdfReader
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency `pypdf`. Run `pip install -r requirements.txt`."
    ) from e

try:
    from langdetect import detect as _detect_lang, DetectorFactory
    DetectorFactory.seed = 0
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency `langdetect`. Run `pip install -r requirements.txt`."
    ) from e


# Compiled regexes — keep them at module scope so they're built once.
_STEP_MARKERS = re.compile(
    r"(?im)^\s*("
    r"step\s+\d+"                 # "Step 1", "Step 12."
    r"|\d+\.\s+\w"                # "1. Wash"
    r"|\(\d+\)\s*\w"              # "(1) Wash"
    r"|[a-z]\)\s*\w"              # "a) Wash"
    r")"
)
_FLOW_MARKERS = re.compile(
    r"(?im)\b(first|then|next|finally|afterwards|prior to|before|after)\b"
)
_SAFETY_TERMS = re.compile(
    r"(?i)\b("
    r"ppe|personal protective equipment|lockout|tagout|lock\s*out|tag\s*out|"
    r"loto|hazard|warning|caution|danger|spill|isolation|"
    r"emergency stop|e[-\s]*stop|"
    r"ventilation|respirator|goggles|gloves|fume hood|"
    r"hazmat|msds|sds|safety data sheet"
    r")\b"
)


def _safe_extract_text(reader: PdfReader, max_pages: int = 40) -> str:
    """Pull text from up to `max_pages` pages. Many SOPs cluster their step
    list near the front; for analysis we don't need the whole doc."""
    out = []
    for i, page in enumerate(reader.pages[:max_pages]):
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out)


def analyze_pdf(pdf_bytes: bytes) -> dict:
    """Run all structural checks on raw PDF bytes.

    Returns a dict — never raises on malformed PDFs; instead it sets
    `page_count=None` and `text=""` so callers can decide whether to drop
    the row or keep it as a failed candidate."""
    out: dict = {
        "page_count": None,
        "estimated_steps": None,
        "length_bucket": "unknown",
        "language": None,
        "has_steps": False,
        "has_safety_clauses": False,
        "bytes": len(pdf_bytes),
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "title": "",
    }

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return out

    try:
        out["page_count"] = len(reader.pages)
    except Exception:
        pass

    # Title from PDF metadata if available.
    try:
        meta = reader.metadata or {}
        t = (meta.get("/Title") or "").strip() if hasattr(meta, "get") else ""
        out["title"] = t
    except Exception:
        pass

    text = _safe_extract_text(reader)
    if not text:
        return out

    # Language detection on a sample.
    try:
        sample = text[:4000]
        out["language"] = _detect_lang(sample) if sample.strip() else None
    except Exception:
        out["language"] = None

    step_hits = _STEP_MARKERS.findall(text)
    flow_hits = _FLOW_MARKERS.findall(text)
    safety_hits = _SAFETY_TERMS.findall(text)

    estimated_steps = len(step_hits) or (len(flow_hits) // 2)
    out["estimated_steps"] = estimated_steps
    out["has_steps"] = estimated_steps >= 3
    out["has_safety_clauses"] = len(safety_hits) >= 1

    if estimated_steps < 20:
        out["length_bucket"] = "short"
    elif estimated_steps <= 40:
        out["length_bucket"] = "medium"
    else:
        out["length_bucket"] = "long"

    return out


def classify_domain(url: str, title: str, text_sample: str = "") -> str:
    """Pick a domain bucket from URL + title + extracted text. Per-source
    `domain_hint` overrides this — see scrape.py."""
    blob = " ".join([url or "", title or "", text_sample or ""]).lower()
    if any(k in blob for k in (
        "chem", "hazcom", "hazard", "process safety", "reactor",
        "spill", "lab safety", "biosaf", "ppe", "lockout", "tagout",
        "nuclear", "radio", "radio", "loto",
    )):
        return "chemical"
    if any(k in blob for k in ("food", "haccp", "kitchen", "meat", "poultry", "subsistence")):
        return "food"
    if any(k in blob for k in ("pharm", "drug", "gmp", "validation", "biolog", "vaccine")):
        return "pharma"
    if any(k in blob for k in ("assembly", "solder", "esd", "electron", "circuit", "manufactur")):
        return "electronics"
    return "general"
