# SOP Corpus Builder

Scrapes real, public, US-Government SOPs and procedural documents into a
manifest CSV. Drives corpus assembly for the
*Position-Aware Chunking for Industrial SOP-to-Workflow Automation* paper
([`../paper/peerj-sop-submission/`](../paper/peerj-sop-submission/)).

## What it does

1. Visits a curated list of **discovery URLs** (per-source index/listing pages
   on osha.gov, apd.army.mil, fda.gov, nrc.gov, fsis.usda.gov, energy.gov,
   cdc.gov, epa.gov, plus a few university EHS pages).
2. Extracts every `.pdf` link from each discovery page.
3. Downloads each PDF (with rate-limiting + retry).
4. Analyses the PDF: page count, estimated step count, whether it has step
   markers, whether it has safety clauses, language.
5. Classifies the document into a domain (chemical / food / pharma /
   electronics / general).
6. Writes one row per surviving PDF to `data/manifest.csv`.

All URLs in the output are **verified to exist** at scrape time — no
fabricated links.

## CSV schema

| Column | Source | Notes |
|---|---|---|
| `url` | scraped | Direct PDF URL |
| `source` | per-source module | e.g. `osha`, `army`, `fda` |
| `domain` | classifier | `chemical` / `food` / `pharma` / `electronics` / `general` |
| `title` | PDF metadata or filename | Falls back to filename if PDF has no title metadata |
| `page_count` | PyPDF | |
| `estimated_steps` | regex on extracted text | Counts `Step N`, numbered list items, `First/Then/Next…` markers |
| `length_bucket` | derived | `short` (<20 steps), `medium` (20–40), `long` (>40) |
| `language` | langdetect | Filter to `en` for first-pass annotation |
| `has_steps` | regex | Boolean; FALSE rows usually aren't SOPs |
| `has_safety_clauses` | regex | Boolean; `PPE`, `Warning`, `Caution`, `Hazard`, `Lockout` etc. |
| `license` | per-source module | e.g. `US Gov public domain` |
| `redistribute_ok` | per-source module | `yes` / `no` / `needs-check` |
| `bytes` | HEAD or file size | |
| `sha256` | computed | For deduplication |

## Install

```sh
cd sop-corpus
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```sh
# default: 200 PDFs, ~25 per source, ~120s per source
python scrape.py

# tune
python scrape.py --target 300 --per-source 40 --concurrency 4

# only one source (debugging)
python scrape.py --only osha

# skip download; just discover URLs (fast)
python scrape.py --discover-only
```

Output: `data/manifest.csv` and downloaded PDFs in `data/pdfs/<source>/`.

## Adding a new source

1. Create `sources/<name>.py` defining:
   - `DISCOVERY_URLS: list[str]`
   - `LICENSE: str`
   - `REDISTRIBUTE_OK: str`  # `yes` | `no` | `needs-check`
   - `domain_hint(url, title) -> str | None`  # optional
2. Register it in `sources/__init__.py`.
3. Re-run `scrape.py --only <name>` to verify.

The base scraper handles fetching, retries, PDF analysis, and CSV writing.
A new source is ~20 lines of code.

## Honest disclaimer

Discovery URLs target listing pages that index PDFs. Listing pages
occasionally change layout. If a source returns zero PDFs after a layout
change, the per-source `discover()` may need updating. Code is intentionally
small and modular so this is a 5-minute fix.

Source rules of thumb (assertions to verify on first run):
- OSHA, FDA, NRC, USDA-FSIS, DOE, CDC, EPA, Army APD → US Federal works,
  generally treated as public domain in the United States.
- University EHS PDFs → typically posted publicly but **redistribution
  varies**; the scraper marks them `needs-check` so you don't ship them in
  the released dataset without confirmation.
