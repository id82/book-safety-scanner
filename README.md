# book-safety-scanner

Scans EPUB books for content not suitable for young children and optionally produces a redacted child-safe version with AI-generated plot bridges replacing the removed passages.

## How it works

### Detection (`scan`)

A three-pass pipeline analyses the book at paragraph granularity:

| Pass | Method | Purpose |
|---|---|---|
| 1 | Regex keyword filter | Triage: flags candidate paragraphs across 6 categories |
| 2 | LLM (Haiku) contextual analysis | Scores each candidate 0–5 with rationale and exact quote |
| 3 | LLM (Haiku) chapter coherence | Checks unflagged chapters for sustained implicit themes |

Each flagged paragraph is scored across six categories: **Violence**, **Sexual Content**, **Drugs & Alcohol**, **Strong Language**, **Psychological Horror**, and **Distressing Themes**.

Adjacent flagged paragraphs are merged into contiguous **skip regions** using a gap-fill rule (≤3 clean paragraphs between blocks are absorbed). The score threshold is configurable by age band:

| Age band | Max acceptable score |
|---|---|
| `under_7` | 1.0 |
| `7_10` | 2.0 |
| `10_12` | 2.5 |
| `12_plus` | 5.0 |

All LLM results are cached in a local SQLite database so interrupted scans resume from where they stopped.

### Redaction (`redact`)

For each skip region, Claude Sonnet writes a 1–2 sentence plot bridge that:
- Preserves narrative continuity and plot-essential information
- Matches the tone and style of the surrounding text
- Contains no reference to the removed content

The original EPUB is then rewritten with the flagged paragraphs replaced by the bridges, producing a standalone child-safe EPUB. Bridge text is cached in the same SQLite database.

## Installation

Requires [uv](https://github.com/astral-sh/uv) and the [Claude CLI](https://claude.ai/download) installed and authenticated.

```bash
git clone https://github.com/id82/book-safety-scanner
cd book-safety-scanner
uv sync
```

## Usage

### Scan a book

```bash
uv run book-safety scan book.epub --age-band 10_12
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--age-band` | `10_12` | Age band: `under_7`, `7_10`, `10_12`, `12_plus` |
| `--output` / `-o` | `scan_results.json` | JSON manifest output path |
| `--html` / `--no-html` | `--html` | Also generate an HTML report |
| `--db` | `scan_cache.db` | SQLite cache path |
| `--pass1-only` | off | Keyword filter only, no LLM calls |
| `--skip-pass3` | off | Skip chapter coherence check |
| `--delay` | `0.3` | Seconds between LLM requests |

The scan produces:
- A **JSON manifest** listing all skip regions with severity scores, triggering quotes, and summaries
- An **HTML report** with colour-coded severity cards

### Generate a child-safe EPUB

```bash
uv run book-safety redact book.epub \
  --manifest scan_results.json \
  --db scan_cache.db \
  --output book_childsafe.epub
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--manifest` / `-m` | `scan_results.json` | Manifest from the scan command |
| `--db` | `scan_cache.db` | SQLite cache (used for bridge caching) |
| `--output` / `-o` | `<original>_childsafe.epub` | Output EPUB path |

## Output

### JSON manifest

```json
{
  "book": { "title": "...", "author": "...", "total_words": 143126 },
  "scan": { "age_band": "10_12", "total_skip_regions": 20 },
  "book_profile": {
    "Sexual Content": 2.10,
    "Violence": 1.07
  },
  "skip_regions": [
    {
      "chapter": 9,
      "chapter_title": "Chapter 9",
      "paragraph_range": [1468, 1470],
      "categories": ["Sexual Content", "Psychological Horror"],
      "max_severity": 3.0,
      "summary": "Explicit discussion of marital sexual dysfunction...",
      "triggering_quotes": ["periodic contact was for the precise purpose of sexual intercourse"]
    }
  ]
}
```

### Redacted EPUB

Replaced passages are marked with `class="content-bridge"` on the replacement `<p>` element, so a parent reviewing the file can identify substitutions. The text itself reads naturally with no gaps or meta-language.

## Models

| Task | Model |
|---|---|
| Pass 1 keyword filter | Local regex (no LLM) |
| Pass 2 & 3 detection | Claude Haiku (fast, cheap, accurate for classification) |
| Bridge writing | Claude Sonnet (higher quality prose generation) |

The Claude CLI is used via `claude -p` subprocess calls with `--exclude-dynamic-system-prompt-sections` to maximise prompt cache hit rates across the repeated detection calls.

## Example: The Robots of Dawn (Isaac Asimov)

143,126 words. Age band `10_12` (threshold 2.5).

- Pass 1: 368 / 4,608 paragraphs flagged (8%)
- Pass 2: 328 paragraphs with non-zero LLM scores
- Final: **20 skip regions** across chapters 9, 10, 13, 18, 19, 20, 22

The primary concern is sexual content (mean score 2.10/5), concentrated in interview scenes where characters describe their personal histories. The core mystery, robot ethics themes, and political intrigue are entirely age-appropriate. The redacted version preserves all of these while replacing the problematic scenes with plot bridges.
