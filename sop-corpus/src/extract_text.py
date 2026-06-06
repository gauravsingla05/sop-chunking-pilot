"""PDF -> text + layout blocks.

Two outputs per PDF, written under ``data/text/<sha256>/``:

  text.txt        : a single UTF-8 stream (paragraphs separated by blank lines).
                    Page boundaries are marked inline with `<<<PAGE_BREAK n>>>`
                    so chunkers that care about page topology can find them.

  blocks.jsonl    : one JSON object per layout block, line-delimited:
                    {"page": int, "bbox": [x0,y0,x1,y1], "text": str,
                     "font_size": float, "is_heading": bool}

We use pypdf for text/page extraction (already a dependency). Layout blocks
are derived from PyMuPDF (fitz) if available — it's a more reliable source
of bounding boxes; otherwise we fall back to a paragraph-level approximation
using blank-line splits.

Usage as a CLI:
    python -m src.extract_text data/pilot.csv
    python -m src.extract_text path/to/file.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Iterator

from pypdf import PdfReader  # already required for the corpus scraper

# PyMuPDF (`fitz`) gives us proper layout blocks with bboxes. Optional —
# without it we degrade to a paragraph-level approximation.
try:
    import fitz  # type: ignore
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False


log = logging.getLogger("extract_text")

HERE = Path(__file__).resolve().parents[1]   # sop-corpus/
DATA = HERE / "data"
TEXT_DIR = DATA / "text"
PDF_DIR = DATA / "pdfs"

PAGE_BREAK = "<<<PAGE_BREAK {n}>>>"


@dataclass
class Block:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    is_heading: bool

    def to_json(self) -> str:
        return json.dumps({
            "page": self.page,
            "bbox": list(self.bbox),
            "text": self.text,
            "font_size": self.font_size,
            "is_heading": self.is_heading,
        }, ensure_ascii=False)


# -------------------- extraction --------------------------------------

def _extract_with_fitz(pdf_path: Path) -> tuple[str, list[Block]]:
    """Preferred path. Returns (text_stream, blocks)."""
    doc = fitz.open(pdf_path)
    text_parts: list[str] = []
    blocks: list[Block] = []

    # Sample median font size to decide what counts as a heading.
    sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes.append(span.get("size", 0))
    body_size = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
    heading_threshold = body_size * 1.2

    for pno, page in enumerate(doc, start=1):
        page_text: list[str] = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            text_lines: list[str] = []
            max_font = 0.0
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if line_text:
                    text_lines.append(line_text)
                for span in line["spans"]:
                    max_font = max(max_font, span.get("size", 0))
            block_text = "\n".join(text_lines).strip()
            if not block_text:
                continue
            bbox = tuple(round(c, 2) for c in block.get("bbox", (0, 0, 0, 0)))
            blocks.append(Block(
                page=pno,
                bbox=bbox,            # type: ignore[arg-type]
                text=block_text,
                font_size=round(max_font, 2),
                is_heading=max_font >= heading_threshold,
            ))
            page_text.append(block_text)
        text_parts.append("\n\n".join(page_text))
        text_parts.append(PAGE_BREAK.format(n=pno))

    return "\n\n".join(text_parts), blocks


def _extract_with_pypdf(pdf_path: Path) -> tuple[str, list[Block]]:
    """Fallback when fitz isn't available. No bboxes, paragraph-level only."""
    reader = PdfReader(str(pdf_path))
    text_parts: list[str] = []
    blocks: list[Block] = []

    for pno, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            log.debug("pypdf page %s failed: %s", pno, e)
            raw = ""
        # Split into paragraph-like blocks on blank lines.
        for para in re.split(r"\n\s*\n", raw):
            para = para.strip()
            if not para:
                continue
            blocks.append(Block(
                page=pno,
                bbox=(0.0, 0.0, 0.0, 0.0),
                text=para,
                font_size=0.0,
                is_heading=False,
            ))
        text_parts.append(raw.strip())
        text_parts.append(PAGE_BREAK.format(n=pno))

    return "\n\n".join(text_parts), blocks


def extract(pdf_path: Path) -> tuple[str, list[Block]]:
    """Top-level extractor — picks fitz if available, pypdf otherwise."""
    if _HAS_FITZ:
        return _extract_with_fitz(pdf_path)
    return _extract_with_pypdf(pdf_path)


# -------------------- I/O helpers -------------------------------------

def pdf_sha(pdf_path: Path) -> str:
    """Stable id for a PDF — matches what scrape.py writes into manifest."""
    h = sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_extracted(sha: str, text: str, blocks: list[Block]) -> Path:
    out_dir = TEXT_DIR / sha
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "text.txt").write_text(text, encoding="utf-8")
    with (out_dir / "blocks.jsonl").open("w", encoding="utf-8") as f:
        for b in blocks:
            f.write(b.to_json() + "\n")
    return out_dir


def iter_pilot_rows(pilot_csv: Path) -> Iterator[dict]:
    with pilot_csv.open() as f:
        yield from csv.DictReader(f)


def find_pdf_for_row(row: dict) -> Path | None:
    """Locate the PDF on disk that matches a pilot.csv row.

    scrape.py wrote each PDF to ``data/pdfs/<source>/<filename>`` using a
    URL-derived filename. We match on sha256 because the URL→filename map
    isn't injective."""
    sha_want = row["sha256"]
    source = row["source"]
    candidates = list((PDF_DIR / source).glob("*.pdf"))
    for c in candidates:
        if pdf_sha(c) == sha_want:
            return c
    # Fallback: search across all sources.
    for c in PDF_DIR.glob("**/*.pdf"):
        if pdf_sha(c) == sha_want:
            return c
    return None


# -------------------- CLI ---------------------------------------------

def _run_csv(pilot_csv: Path) -> None:
    ok = miss = err = 0
    for row in iter_pilot_rows(pilot_csv):
        pdf = find_pdf_for_row(row)
        if pdf is None:
            log.warning("no local PDF for %s (sha=%s) — skipping",
                        row["title"][:60], row["sha256"][:10])
            miss += 1
            continue
        try:
            text, blocks = extract(pdf)
            out = write_extracted(row["sha256"], text, blocks)
            log.info("%s -> %s  (%d chars, %d blocks)",
                     pdf.name, out, len(text), len(blocks))
            ok += 1
        except Exception as e:
            log.error("extract failed for %s: %s", pdf, e)
            err += 1
    log.info("done: ok=%d miss=%d err=%d", ok, miss, err)


def _run_file(pdf: Path) -> None:
    text, blocks = extract(pdf)
    sha = pdf_sha(pdf)
    out = write_extracted(sha, text, blocks)
    log.info("%s -> %s  (%d chars, %d blocks)",
             pdf.name, out, len(text), len(blocks))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="data/pilot.csv (or any csv) or a PDF path")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    backend = "PyMuPDF (fitz)" if _HAS_FITZ else "pypdf (fallback)"
    log.info("text-extraction backend: %s", backend)

    inp = Path(args.input)
    if inp.suffix.lower() == ".csv":
        _run_csv(inp)
    elif inp.suffix.lower() == ".pdf":
        _run_file(inp)
    else:
        log.error("unsupported input: %s", inp)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
