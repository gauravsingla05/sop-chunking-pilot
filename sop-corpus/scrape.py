"""SOP corpus scraper — drives source modules, downloads PDFs, writes
the manifest CSV.

Usage:
    python scrape.py
    python scrape.py --target 300 --per-source 40
    python scrape.py --only osha
    python scrape.py --discover-only        # list PDF URLs, no download
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import random
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from sources import ALL as ALL_SOURCES, SOURCES as ALL_SOURCE_MODULES
from sources._base import PdfRow
from analyzer import analyze_pdf, classify_domain


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PDF_DIR = DATA_DIR / "pdfs"
MANIFEST = DATA_DIR / "manifest.csv"

USER_AGENT = (
    "SOP-Corpus-Builder/0.1 "
    "(academic research; contact: gouravsingla05@gmail.com)"
)
REQUEST_TIMEOUT = 30
POLITE_DELAY_SECONDS = 0.8   # between requests to the same host
MAX_PDF_BYTES = 30 * 1024 * 1024   # skip PDFs bigger than 30 MB

# Two-hop crawl budget: when an index page yields no PDFs, follow this many
# same-host HTML links from it and look for PDFs on each linked page.
TWO_HOP_PAGES_PER_INDEX = 30

log = logging.getLogger("sop")


# -------------------- HTTP helpers ----------------------------------

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _polite_sleep():
    # Tiny jitter so we don't hammer a host in lockstep.
    time.sleep(POLITE_DELAY_SECONDS + random.uniform(0, 0.4))


def fetch_html(url: str) -> str | None:
    try:
        r = _session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code != 200 or "html" not in (r.headers.get("Content-Type") or "").lower():
            return None
        return r.text
    except requests.RequestException as e:
        log.debug("fetch_html failed: %s — %s", url, e)
        return None


def fetch_pdf(url: str) -> bytes | None:
    try:
        # HEAD first so we can short-circuit on size.
        h = _session.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        clen = int(h.headers.get("Content-Length", "0") or "0")
        if clen and clen > MAX_PDF_BYTES:
            log.debug("skip too large: %s (%d bytes)", url, clen)
            return None
        r = _session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True)
        if r.status_code != 200:
            return None
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and not url.lower().endswith(".pdf"):
            return None
        body = io.BytesIO()
        for chunk in r.iter_content(chunk_size=64 * 1024):
            body.write(chunk)
            if body.tell() > MAX_PDF_BYTES:
                return None
        return body.getvalue()
    except requests.RequestException as e:
        log.debug("fetch_pdf failed: %s — %s", url, e)
        return None


# -------------------- Discovery -------------------------------------

def _allow_hosts(source_mod) -> set[str]:
    allow = set(getattr(source_mod, "ALLOW_HOSTS", []))
    if not allow:
        for u in source_mod.DISCOVERY_URLS:
            allow.add(urlparse(u).netloc)
    return allow


def _extract_links(html: str, base_url: str, allow_hosts: set[str]):
    """Yield (kind, absolute_url) for every same-host link in `html`.
    kind ∈ {'pdf', 'html'}. Fragments and mailto/tel are skipped."""
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absu = urljoin(base_url, href).split("#", 1)[0]
        host = urlparse(absu).netloc
        if host not in allow_hosts:
            continue
        path = urlparse(absu).path.lower()
        if path.endswith(".pdf"):
            yield "pdf", absu
        elif path.endswith((".html", ".htm", "/", "")) or "." not in path.rsplit("/", 1)[-1]:
            # Heuristic: treat extension-less or .html paths as HTML pages
            # worth following on the second hop.
            yield "html", absu


def discover_pdf_urls(source_mod, per_source_target: int) -> list[str]:
    """Discover PDF URLs from a source.

    Phase 1 (one-hop): collect direct `.pdf` links from every DISCOVERY_URL.
    Phase 2 (two-hop): if we're short of the target, follow up to
    TWO_HOP_PAGES_PER_INDEX same-host HTML links from each DISCOVERY_URL
    and look for `.pdf` links on those linked pages. This unblocks sites
    where the browse page is a list of landing pages (one click → PDF).

    Restricts to ALLOW_HOSTS to avoid off-site spider crawls.
    """
    allow = _allow_hosts(source_mod)
    found: list[str] = []
    seen: set[str] = set()

    # --- Phase 1: one-hop ---------------------------------------------
    index_to_html: list[tuple[str, str]] = []   # cache for phase 2 reuse
    for index_url in source_mod.DISCOVERY_URLS:
        if len(found) >= per_source_target:
            break
        log.info("[%s] discovering (1-hop): %s", source_mod.SOURCE_NAME, index_url)
        html = fetch_html(index_url)
        _polite_sleep()
        if not html:
            continue
        index_to_html.append((index_url, html))
        for kind, absu in _extract_links(html, index_url, allow):
            if kind != "pdf" or absu in seen:
                continue
            seen.add(absu)
            found.append(absu)
            if len(found) >= per_source_target:
                break

    if len(found) >= per_source_target:
        log.info("[%s] discovered %d PDFs (1-hop sufficed)",
                 source_mod.SOURCE_NAME, len(found))
        return found

    log.info("[%s] 1-hop yielded %d; trying 2-hop crawl…",
             source_mod.SOURCE_NAME, len(found))

    # --- Phase 2: two-hop ---------------------------------------------
    visited_html: set[str] = set(u for u, _ in index_to_html)

    for index_url, html in index_to_html:
        if len(found) >= per_source_target:
            break
        # Collect candidate HTML links from this index page.
        html_links: list[str] = []
        for kind, absu in _extract_links(html, index_url, allow):
            if kind != "html" or absu in visited_html:
                continue
            visited_html.add(absu)
            html_links.append(absu)
            if len(html_links) >= TWO_HOP_PAGES_PER_INDEX:
                break

        # Visit each linked HTML page and harvest PDFs.
        for page_url in html_links:
            if len(found) >= per_source_target:
                break
            sub = fetch_html(page_url)
            _polite_sleep()
            if not sub:
                continue
            for kind, absu in _extract_links(sub, page_url, allow):
                if kind != "pdf" or absu in seen:
                    continue
                seen.add(absu)
                found.append(absu)
                if len(found) >= per_source_target:
                    break

    log.info("[%s] discovered %d PDFs (after 2-hop)",
             source_mod.SOURCE_NAME, len(found))
    return found


# -------------------- Per-PDF processing -----------------------------

def process_pdf(url: str, source_mod) -> PdfRow | None:
    """Download + analyze one PDF. Returns None on hard failure."""
    pdf = fetch_pdf(url)
    if not pdf:
        return None

    info = analyze_pdf(pdf)
    # Drop PDFs we can't read at all (no page count) — they're likely
    # scanned-only or malformed and not useful for chunking experiments.
    if not info.get("page_count"):
        return None

    title = info.get("title") or _title_from_url(url)
    # Domain: source hint first, fall back to keyword classifier.
    hint_fn = getattr(source_mod, "domain_hint", None)
    domain = (hint_fn(url, title) if hint_fn else None) or \
             classify_domain(url, title, "")

    return PdfRow(
        url=url,
        source=source_mod.SOURCE_NAME,
        domain=domain,
        title=title,
        page_count=info["page_count"],
        estimated_steps=info["estimated_steps"],
        length_bucket=info["length_bucket"],
        language=info["language"],
        has_steps=info["has_steps"],
        has_safety_clauses=info["has_safety_clauses"],
        license=source_mod.LICENSE,
        redistribute_ok=source_mod.REDISTRIBUTE_OK,
        bytes=info["bytes"],
        sha256=info["sha256"],
    )


def save_pdf(pdf_bytes: bytes, source: str, url: str) -> Path:
    out_dir = PDF_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_name_from_url(url)
    p = out_dir / name
    p.write_bytes(pdf_bytes)
    return p


def _title_from_url(url: str) -> str:
    stem = Path(urlparse(url).path).stem.replace("_", " ").replace("-", " ")
    return stem.strip().title()[:200]


def _safe_name_from_url(url: str) -> str:
    stem = Path(urlparse(url).path).name or "document.pdf"
    if not stem.lower().endswith(".pdf"):
        stem += ".pdf"
    return stem[:200]


# -------------------- Main driver -----------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=200, help="overall target PDFs")
    ap.add_argument("--per-source", type=int, default=30, help="cap per source")
    ap.add_argument("--only", type=str, default=None,
                    help="run only one source (e.g. osha, army, fda)")
    ap.add_argument("--discover-only", action="store_true",
                    help="list PDF URLs without downloading")
    ap.add_argument("--save-pdfs", action="store_true",
                    help="also write the downloaded PDFs to data/pdfs/<source>/")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        if args.only not in ALL_SOURCES:
            log.error("Unknown source %r. Choices: %s",
                      args.only, ", ".join(ALL_SOURCES))
            sys.exit(2)
        sources = [ALL_SOURCES[args.only]]
    else:
        sources = ALL_SOURCE_MODULES

    # ---- Phase 1: discovery
    candidates: list[tuple[object, str]] = []
    for src in sources:
        if len(candidates) >= args.target:
            break
        urls = discover_pdf_urls(src, args.per_source)
        for u in urls:
            candidates.append((src, u))
            if len(candidates) >= args.target:
                break

    log.info("Phase 1 complete: %d total candidate PDFs", len(candidates))

    if args.discover_only:
        out = DATA_DIR / "candidates.txt"
        with out.open("w") as f:
            for src, u in candidates:
                f.write(f"{src.SOURCE_NAME}\t{u}\n")
        log.info("Wrote %s", out)
        return

    # ---- Phase 2: download + analyze
    rows: list[PdfRow] = []
    seen_hashes: set[str] = set()

    for src, url in tqdm(candidates, desc="processing", unit="pdf"):
        try:
            pdf_bytes = fetch_pdf(url)
            _polite_sleep()
            if not pdf_bytes:
                continue
            info = analyze_pdf(pdf_bytes)
            if not info.get("page_count"):
                continue
            sha = info["sha256"]
            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)

            title = info.get("title") or _title_from_url(url)
            hint_fn = getattr(src, "domain_hint", None)
            domain = (hint_fn(url, title) if hint_fn else None) or \
                     classify_domain(url, title, "")

            row = PdfRow(
                url=url,
                source=src.SOURCE_NAME,
                domain=domain,
                title=title,
                page_count=info["page_count"],
                estimated_steps=info["estimated_steps"],
                length_bucket=info["length_bucket"],
                language=info["language"],
                has_steps=info["has_steps"],
                has_safety_clauses=info["has_safety_clauses"],
                license=src.LICENSE,
                redistribute_ok=src.REDISTRIBUTE_OK,
                bytes=info["bytes"],
                sha256=info["sha256"],
            )
            rows.append(row)

            if args.save_pdfs:
                save_pdf(pdf_bytes, src.SOURCE_NAME, url)

        except KeyboardInterrupt:
            log.warning("Interrupted — writing partial manifest.")
            break
        except Exception as e:
            log.debug("error on %s: %s", url, e)
            continue

    # ---- Phase 3: write manifest
    log.info("Phase 3: writing manifest with %d rows", len(rows))
    write_manifest(rows)


def write_manifest(rows: list[PdfRow]):
    if not rows:
        log.warning("No rows — manifest will be empty. Did discovery return any PDFs?")
    cols = [
        "url", "source", "domain", "title",
        "page_count", "estimated_steps", "length_bucket",
        "language", "has_steps", "has_safety_clauses",
        "license", "redistribute_ok",
        "bytes", "sha256",
    ]
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())
    log.info("Wrote %s", MANIFEST)


if __name__ == "__main__":
    main()
