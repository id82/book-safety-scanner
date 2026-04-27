"""Three-pass content detection pipeline."""

import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .database import ScanDatabase
from .ingestion import Paragraph, build_context_window
from .taxonomy import CATEGORIES, KeywordHit, scan_keywords

# ── JSON schemas for structured output ──────────────────────────────────────

_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": "Severity 0=none 1=very mild 2=mild 3=moderate 4=strong 5=extreme",
        },
        "rationale": {"type": "string"},
        "exact_quote": {"type": ["string", "null"]},
    },
    "required": ["score", "rationale", "exact_quote"],
}

CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {cat: _CATEGORY_SCHEMA for cat in CATEGORIES},
    "required": CATEGORIES,
}

CHAPTER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "has_implicit_concern": {"type": "boolean"},
        "categories_of_concern": {
            "type": "array",
            "items": {"type": "string", "enum": CATEGORIES},
        },
        "summary": {"type": "string"},
    },
    "required": ["has_implicit_concern", "categories_of_concern", "summary"],
}

SYSTEM_PROMPT = """\
You are a children's content safety classifier. Analyse passages from books and \
identify content that may be inappropriate for young readers.

Scoring guide (0-5):
  violence      : 1=mild conflict/threat, 2=cartoon-style violence, 3=injury described, \
4=graphic/repeated harm, 5=gore/torture/extreme
  sexual_content: 1=hand-holding/innocent romance, 2=suggestive language, \
3=non-graphic intimacy implied, 4=explicit description, 5=explicit acts
  drugs_alcohol : 1=passive mention, 2=casual use depicted, 3=intoxication shown, \
4=use glorified, 5=trafficking/glorified heavy use
  language      : 1=mild expletive (hell/damn), 2=moderate profanity, \
3=strong profanity, 4=slurs/hate language, 5=extreme/repeated slurs
  psychological : 1=mild fear/sadness, 2=sustained dread, \
3=abuse depicted/self-harm referenced, 4=graphic trauma, 5=detailed self-harm/suicide
  themes        : 1=death mentioned briefly, 2=loss/abandonment subplot, \
3=traumatic theme central to scene, 4=war crimes/genocide referenced, \
5=sustained traumatic theme throughout

Context lines beginning with ">>> " are the primary paragraph being assessed. \
Adjacent lines are context only — do not score them.

Be precise. Many mentions of death, fear, or conflict are age-appropriate; \
score only what would genuinely concern a thoughtful parent.\
"""

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CategoryScore:
    score: int
    rationale: str
    exact_quote: str | None


@dataclass
class ParagraphAnalysis:
    para_id: int
    keyword_hits: dict[str, list[KeywordHit]] = field(default_factory=dict)
    llm_scores: dict[str, CategoryScore] = field(default_factory=dict)

    def max_score(self) -> float:
        if not self.llm_scores:
            return 0.0
        return float(max(s.score for s in self.llm_scores.values()))

    def flagged_categories(self) -> list[str]:
        return [cat for cat, s in self.llm_scores.items() if s.score > 0]


# ── Claude CLI subprocess helper ─────────────────────────────────────────────

_RATE_LIMIT_SIGNALS = ("rate limit", "429", "too many requests", "overloaded")
_MAX_RETRIES = 4


def _run_claude(
    prompt: str,
    json_schema: dict,
    model: str = "haiku",
    timeout: int = 60,
) -> dict | None:
    """Call `claude -p -` and return structured_output dict, or None on failure.

    Retries up to _MAX_RETRIES times with exponential backoff on rate-limit errors.
    """
    cmd = [
        "claude", "-p", "-",
        "--model", model,
        "--output-format", "stream-json",
        "--json-schema", json.dumps(json_schema),
        "--append-system-prompt", SYSTEM_PROMPT,
        "--exclude-dynamic-system-prompt-sections",
        "--no-session-persistence",
        "--permission-mode", "bypassPermissions",
    ]

    for attempt in range(_MAX_RETRIES):
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None

        if any(sig in result.stderr.lower() for sig in _RATE_LIMIT_SIGNALS):
            backoff = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(backoff)
            continue

        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("type") == "result":
                    return msg.get("structured_output")
            except json.JSONDecodeError:
                continue

        return None

    return None


# ── Pass 1: Keyword filter ────────────────────────────────────────────────────

def run_pass1(
    paragraphs: list[Paragraph],
    db: ScanDatabase,
    progress_cb: Callable[[int, int], None] | None = None,
) -> set[int]:
    """Scan all paragraphs with keyword patterns. Returns set of candidate para IDs."""
    flagged = db.get_keyword_flagged_ids()

    if not flagged:
        for i, para in enumerate(paragraphs):
            hits = scan_keywords(para.text)
            if hits:
                db.insert_keyword_flags(para.id, hits)
                flagged.add(para.id)
            if progress_cb:
                progress_cb(i + 1, len(paragraphs))
        flagged = db.get_keyword_flagged_ids()
    else:
        if progress_cb:
            progress_cb(len(paragraphs), len(paragraphs))

    return flagged


# ── Pass 2: LLM contextual analysis ──────────────────────────────────────────

def run_pass2(
    paragraphs: list[Paragraph],
    candidate_ids: set[int],
    db: ScanDatabase,
    progress_cb: Callable[[int, int], None] | None = None,
    workers: int = 8,
) -> dict[int, dict]:
    """Run LLM analysis on candidate paragraphs. Returns {para_id: llm_result}."""
    already_done = db.get_analyzed_ids()
    todo = [p for p in paragraphs if p.id in candidate_ids and p.id not in already_done]

    if not todo:
        return db.load_llm_results()

    def _analyse(para: Paragraph) -> tuple[int, dict | None]:
        context_text = build_context_window(paragraphs, para.id, window=2)
        return para.id, _run_claude(
            prompt=f"Analyse this passage:\n\n{context_text}",
            json_schema=CLASSIFY_SCHEMA,
        )

    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_analyse, para): para for para in todo}
        for future in as_completed(futures):
            para_id, result = future.result()
            if result:
                db.insert_llm_result(para_id, result)
            done_count += 1
            if progress_cb:
                progress_cb(done_count, len(todo))

    return db.load_llm_results()


# ── Pass 3: Chapter coherence check ──────────────────────────────────────────

def run_pass3(
    paragraphs: list[Paragraph],
    llm_results: dict[int, dict],
    db: ScanDatabase,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[int, dict]:
    """Run chapter-level coherence check on chapters with zero paragraph flags."""
    flagged_chapters = {
        para.chapter_num
        for para in paragraphs
        if para.id in llm_results
        and any(llm_results[para.id].get(cat, {}).get("score", 0) > 0 for cat in CATEGORIES)
    }
    all_chapters = sorted({p.chapter_num for p in paragraphs})
    clean_chapters = [c for c in all_chapters if c not in flagged_chapters]
    already_done = db.get_analyzed_chapters()
    todo = [c for c in clean_chapters if c not in already_done]

    for done_count, chapter_num in enumerate(todo):
        chapter_paras = [p for p in paragraphs if p.chapter_num == chapter_num]
        if not chapter_paras:
            continue

        chapter_title = chapter_paras[0].chapter_title
        # Sample up to 30 paragraphs evenly to stay within token budget
        step = max(1, len(chapter_paras) // 30)
        sampled = "\n\n".join(p.text for p in chapter_paras[::step][:30])

        result = _run_claude(
            prompt=(
                f"Chapter {chapter_num} ('{chapter_title}'): no paragraph-level flags raised. "
                f"Are there sustained implicit themes of concern?\n\n{sampled}"
            ),
            json_schema=CHAPTER_SCHEMA,
        )
        if result:
            db.insert_chapter_result(chapter_num, result)

        if progress_cb:
            progress_cb(done_count + 1, len(todo))

    return db.load_chapter_results()


# ── Result assembly ───────────────────────────────────────────────────────────

def assemble_analyses(
    paragraphs: list[Paragraph],
    candidate_ids: set[int],
    llm_results: dict[int, dict],
    db: ScanDatabase,
) -> dict[int, ParagraphAnalysis]:
    """Merge Pass 1 and Pass 2 results into ParagraphAnalysis objects."""
    analyses: dict[int, ParagraphAnalysis] = {}

    for para in paragraphs:
        if para.id not in candidate_ids and para.id not in llm_results:
            continue

        kw_cats = db.get_keyword_categories(para.id) if para.id in candidate_ids else []
        kw_hits: dict[str, list[KeywordHit]] = {c: [] for c in kw_cats}

        llm = llm_results.get(para.id, {})
        llm_scores: dict[str, CategoryScore] = {}
        for cat in CATEGORIES:
            cat_data = llm.get(cat, {})
            if isinstance(cat_data, dict) and cat_data.get("score", 0) > 0:
                llm_scores[cat] = CategoryScore(
                    score=int(cat_data.get("score", 0)),
                    rationale=cat_data.get("rationale", ""),
                    exact_quote=cat_data.get("exact_quote"),
                )

        analyses[para.id] = ParagraphAnalysis(
            para_id=para.id,
            keyword_hits=kw_hits,
            llm_scores=llm_scores,
        )

    return analyses
