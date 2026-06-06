# SOP Chunking Pilot

Research artefact for the paper *Step-Count Fidelity as a Companion
Metric for Chunking Procedural Text, and the Instability of LLM-Judged
Chunker Evaluation: A Pilot on Industrial Standard-Operating-Procedure
Automation.*

This repository contains the corpus manifest, the full evaluation pipeline,
both LLM extractor runs (Gemini Flash and OpenAI gpt-4o-mini), the
matched-prompt multiplicity-isolation outputs, the analysis scripts that
produced the tables in the paper, and the LaTeX source for the MDPI MAKE
submission.

## Repository layout

```
sop-chunking-pilot/
├── README.md
├── LICENSE                                       (MIT)
├── sop-corpus/                                   pipeline source
│   ├── src/
│   │   ├── chunkers/        fixed, recursive, semantic, layout, PAC
│   │   ├── extractor/       multi-backend LLM client (Gemini, OpenAI, Anthropic, Claude CLI)
│   │   ├── pipeline/        run.py (cell-level resumable runner), merge.py
│   │   └── eval/            metrics.py, analyze.py
│   ├── scrape.py            (re-build the corpus from public URLs)
│   ├── requirements.txt
│   └── README.md            corpus-side details
├── data/
│   ├── manifest.csv         all 88 candidate documents (URL, SHA-256, metadata)
│   ├── pilot.csv            tiny smoke-test slice
│   ├── pilot-n26.csv        the 26-document headline pilot
│   ├── pilot-procedural.csv 20-document procedural-only subset
│   └── runs/
│       ├── pilot-001-gemini/    Gemini Flash run (chunk_graphs, merged, oracle, metrics, analysis CSVs)
│       └── pilot-002-openai/    OpenAI gpt-4o-mini cross-extractor run
├── paper/
│   └── mdpi-make-submission/            LaTeX source, MDPI mdpi template, MAKE journal
├── supp_table_s1_chunk_counts.csv       chunk counts per (document, chunker)
├── supp_table_s2_per_stratum.csv        per-stratum metric means
├── supp_table_s3_cross_extractor.csv    Gemini vs OpenAI on the 15-doc paired slice
└── supp_table_s4_idempotency_control.csv matched-prompt multiplicity isolation on 8 small docs
```

## What is in here vs. what is not

- **Included:** all source code, the corpus manifest, all per-cell JSON
  outputs from both extractor runs, the merged document-level graphs, the
  oracle graphs, the metrics JSONL, the analysis CSVs, the supplementary
  tables, and the LaTeX sources for both papers.
- **Not included:** the source PDF binaries. All 26 documents are
  US-government public-domain works retrievable directly from the URLs in
  `data/manifest.csv`. The `scrape.py` script re-downloads them.

## Reproducing the pilot

The pipeline writes per-cell checkpoints so re-runs skip any cell whose
output already exists. A full re-run from clean state costs roughly:
Gemini Flash on the free tier, no money, ~3-4 hours; OpenAI gpt-4o-mini,
$2-5, ~2 hours.

### 1. Install

```sh
git clone https://github.com/gauravsingla05/sop-chunking-pilot.git
cd sop-chunking-pilot/sop-corpus
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Re-download the corpus PDFs

```sh
python scrape.py --manifest ../data/manifest.csv --out-dir ../data/pdfs
python -m src.pipeline.extract_text --pilot ../data/pilot-n26.csv
```

### 3. Run the pipeline

The backend is selected automatically from the available API key; or force
one with `EXTRACTOR_BACKEND`.

```sh
# Gemini Flash (free tier)
export GEMINI_API_KEY=your-key-here
EXTRACTOR_BACKEND=gemini-api python -m src.pipeline.run \
    --pilot ../data/pilot-n26.csv \
    --run-id pilot-001-gemini \
    --sample-chunks 20 --seed 0

# OpenAI gpt-4o-mini (cross-extractor robustness check)
export OPENAI_API_KEY=your-key-here
EXTRACTOR_BACKEND=openai-api OPENAI_MODEL=gpt-4o-mini \
  python -m src.pipeline.run \
    --pilot ../data/pilot-n26.csv \
    --run-id pilot-002-openai \
    --sample-chunks 20 --seed 0
```

A `.env` file at the repository root with `GEMINI_API_KEY=...` /
`OPENAI_API_KEY=...` is picked up automatically. Set `SOP_DOTENV=/path/to/.env`
to override.

### 4. Re-derive the analysis tables

```sh
python -m src.eval.analyze --run-id pilot-001-gemini --pilot ../data/pilot-n26.csv
python -m src.eval.analyze --run-id pilot-002-openai --pilot ../data/pilot-n26.csv
```

Outputs are written under `data/runs/<run-id>/analysis*`.

### 5. Build the papers

Each paper folder is a standalone LaTeX project.

```sh
cd paper/mdpi-make-submission
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

The MDPI submission compiles cleanly on Overleaf with the project zip
created from `paper/mdpi-make-submission/`.

## How chunkers are added

A chunker is a class with a `chunk(doc) -> list[Chunk]` method registered
in `sop-corpus/src/chunkers/__init__.py`. The five chunkers in the paper
are in `src/chunkers/{fixed.py, recursive.py, semantic.py, layout_aware.py, pac.py}`.

## Reference PAC implementation

PAC is implemented in `sop-corpus/src/chunkers/pac.py` (approximately 400
lines of Python). The four-pass structure, the regex patterns, the
imperative-verb lexicon, and the token-budget constants are all documented
in the supplementary information of the paper (sections S1 and S2).

## Citing this work

If you use this corpus or pipeline, please cite the paper. A DOI for this
repository will be added here once the GitHub --> Zenodo integration is
configured and a release is tagged.

## License

[MIT](LICENSE) for the code; the corpus documents are US-government
public-domain works and carry no additional restrictions beyond their
source agencies' terms.
