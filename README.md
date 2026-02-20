# PHAT — Patent Prosecution History Analysis Tool

**Automated AI-powered analysis of USPTO patent file histories for litigation support, portfolio review, and prosecution strategy.**

PHAT ingests a patent's Image File Wrapper PDF, segments it into individual documents, and runs multi-layered AI analysis to extract claims evolution, prosecution statements, estoppel risks, prior art distinctions, and more — then outputs a structured report ready for attorney review.

---

## Features

### Core Analysis (Layer 1)

- **PDF Segmentation** — Bookmark-based primary strategy with pattern-matching fallback. Automatically splits a monolithic USPTO file history PDF into discrete documents (Office Actions, Amendments, Notices of Allowance, etc.).
- **Document Triage** — AI-driven classification that skips low-value documents (fee transmittals, drawings, filing receipts) and prioritizes high-value ones (appeal briefs, allowance reasons).
- **Claims Extraction & Tracking** — Extracts every claim version across the prosecution timeline and tracks additions, amendments, cancellations, and dependency changes.
- **Rejection Analysis** — Parses 35 U.S.C. §§ 101, 102, 103, 112, and double-patenting rejections with cited prior art, rejection basis, and affected claims.
- **Amendment Analysis** — Identifies claim amendments, captures before/after text, and flags scope-narrowing language that may create prosecution history estoppel.
- **Argument Extraction** — Extracts applicant and examiner statements, classifies them by relevance category (Claim Construction, Estoppel, Prior Art Distinction, Definition/Interpretation, etc.), and assigns litigation-risk scores.
- **Interview Summary Analysis** — Extracts concessions and agreements from examiner interview summaries.
- **Restriction/Election Analysis** — Captures restriction requirements, elected/non-elected groups, and traversal arguments.
- **Terminal Disclaimer Detection** — Identifies terminal disclaimers with linked patent/application numbers.
- **Means-Plus-Function Detection** — Flags § 112(f) claim limitations with corresponding structure identification.
- **Final Claims Integration** — Auto-fetches issued claims from Google Patents (or accepts manual input) and uses them as ground truth for amendment tracking.

### Post-Processing Synthesis (Layers 2 & 3)

- **Shadow Examiner** — AI critique that identifies potential validity risks an opposing party might raise, with severity ratings and supporting evidence.
- **Definition Synthesis** — Consolidates all prosecution statements about key claim terms into per-term definition summaries with consistency analysis.
- **Claim Narratives** — Generates plain-language prosecution histories for each independent claim.
- **Strategic Tensions** — Detects contradictions or inconsistencies in positions taken across different prosecution documents.
- **Themes of Patentability** — Groups arguments into thematic clusters to reveal the prosecution's overarching narrative.
- **Term Boundary Extraction** — Identifies actionable scope boundaries for key claim terms based on prosecution statements.
- **Prior Art Consolidation** — Structures scattered prior art citations into a unified reference table.
- **Claim Type Classification** — Categorizes final claims as Method, System, Computer-Readable Medium, etc.
- **Vulnerability Cards** — Per-claim litigation vulnerability assessments combining estoppel risks, prior art exposure, and amendment history.

### Integrity & Quality

- **Quote Verification** — Every AI-extracted quote is verified against the source text using fuzzy matching to detect hallucinations. Unverified quotes are flagged or downgraded automatically.
- **OCR Cleaning** — Conservative regex-based correction of common OCR errors in patent documents, with optional Tesseract re-OCR fallback for garbled text layers.
- **Intelligent Chunking** — Documents exceeding the AI context window are split with overlap and results are merged with deduplication.
- **Resume Support** — If a run fails mid-analysis, PHAT detects the incomplete state and resumes from where it left off.

---

## Supported AI Providers

| Provider | Full Model | Fast Model |
|----------|-----------|------------|
| **Anthropic Claude** | claude-sonnet-4-5 | claude-haiku-4-5 |
| **Google Gemini** | gemini-2.5-pro | gemini-2.5-flash |
| **DeepSeek** | deepseek-reasoner | deepseek-chat |

PHAT supports three model modes: **full** (best quality), **fast** (cheaper/faster), and **custom** (per-task model assignment so you can use the full model for critical tasks like amendment analysis while using the fast model for triage).

---

## Installation

### Prerequisites

- Python 3.10+
- An API key for at least one supported provider (Claude, Gemini, or DeepSeek)
- *(Optional)* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for re-OCR fallback on garbled pages

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/phat.git
cd phat

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The following packages are required:

```
pdfplumber
pypdf
PyMuPDF          # Optional — enables re-OCR fallback for garbled pages
sqlalchemy
httpx
pyyaml
```

If you don't have a `requirements.txt` yet, you can install directly:

```bash
pip install pdfplumber pypdf PyMuPDF sqlalchemy httpx pyyaml
```

### Configuration

Copy and edit the configuration file:

```bash
cp config.yaml config.yaml  # already provided — edit in place
```

At minimum, add your API key for your chosen provider:

```yaml
ai_provider: "claude"       # or "gemini" or "deepseek"
model_mode: "full"           # "full", "fast", or "custom"

api_keys:
  claude: "sk-ant-..."
  gemini: "AIza..."
  deepseek: "sk-..."
```

All other settings (post-processing toggles, chunking limits, verification thresholds, triage skip-lists, etc.) have sensible defaults and are documented with inline comments in `config.yaml`.

---

## Usage

### Basic Workflow

```bash
# 1. Place USPTO file history PDF(s) in the input/ folder

# 2. Run analysis
python main.py

# 3. Reports appear in output/
```

PHAT will automatically attempt to fetch the patent's final issued claims from Google Patents. If auto-fetch fails, you'll be prompted to paste them manually (or skip with `--no-claims`).

### Command-Line Options

```
python main.py                              # Process all PDFs in input/
python main.py --file patent.pdf            # Process a specific file
python main.py --fast                       # Use fast/cheap model
python main.py --provider gemini --fast     # Use Gemini Flash
python main.py --report <patent_id>         # Regenerate report for an analyzed patent
python main.py --list                       # List all analyzed patents
python main.py --no-claims                  # Skip the final claims prompt
python main.py --claims-file claims.txt     # Load final claims from a text file
python main.py --add-claims <patent_id>     # Add final claims to an existing analysis
python main.py --no-report                  # Run analysis only, skip report generation
python main.py --verbose                    # Enable verbose AI input/output logging
python main.py --config custom.yaml         # Use an alternate config file
```

### Output

PHAT generates three output files per patent:

| File | Description |
|------|-------------|
| `report_<patent>.md` | Markdown report for human review |
| `report_<patent>.html` | Styled HTML report (same content, browser-viewable) |
| `data_<patent>.json` | Structured JSON data export for programmatic consumption |

---

## Project Structure

```
phat/
├── main.py                 # CLI entry point and orchestration
├── config.yaml             # All configuration options
├── src/
│   ├── ai_providers.py     # Claude / Gemini / DeepSeek abstraction layer
│   ├── analysis_engine.py  # Core analysis orchestration (Layers 1–3)
│   ├── chunking.py         # Large-document splitting and result merging
│   ├── database.py         # SQLAlchemy ORM models (SQLite)
│   ├── final_claims.py     # Google Patents fetcher and claims parser
│   ├── ocr_cleaner.py      # OCR error correction and re-OCR fallback
│   ├── pdf_processor.py    # PDF extraction, bookmark segmentation, triage
│   ├── prompts.py          # All AI prompt templates
│   ├── report_generator.py # Markdown, HTML, and JSON report generation
│   └── verification.py     # Quote verification / hallucination detection
├── input/                  # Place USPTO file history PDFs here
└── output/                 # Generated reports and database
    ├── phat_database.db
    └── phat.log
```

---

## How It Works

1. **PDF Ingestion** — The file history PDF is opened and segmented into individual documents using PDF bookmarks (with pattern-based fallback). OCR cleaning is applied to each segment.

2. **Document Triage** — Each segment is classified by type. Low-value documents are skipped; high-priority documents are flagged for deeper analysis.

3. **Layer 1 Analysis** — Each document is sent to the AI with a task-specific prompt (rejection analysis, amendment analysis, argument extraction, etc.). Extracted quotes are verified against the source text. Results are stored in a local SQLite database.

4. **Layer 2–3 Synthesis** — After all documents are analyzed, cross-document post-processing runs: shadow examiner critique, definition synthesis, thematic grouping, strategic tension detection, vulnerability cards, and more.

5. **Report Generation** — All structured data is assembled into Markdown, HTML, and JSON reports.

