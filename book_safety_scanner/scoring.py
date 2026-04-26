"""Score aggregation at paragraph, chapter, and book level."""

from dataclasses import dataclass, field

from .ingestion import BookMeta, Paragraph
from .pipeline import ParagraphAnalysis
from .taxonomy import AGE_BANDS, CATEGORIES, CATEGORY_LABELS


@dataclass
class SkipRegion:
    chapter_num: int
    chapter_title: str
    para_range: tuple[int, int]   # (start_global_id, end_global_id) inclusive
    para_indices: tuple[int, int] # (first_para_index_in_chapter, last_para_index_in_chapter)
    categories: list[str]
    max_severity: float
    summary: str
    quotes: list[str] = field(default_factory=list)


@dataclass
class ChapterScore:
    chapter_num: int
    chapter_title: str
    max_score: float
    mean_top_quartile: float
    categories: list[str]


@dataclass
class BookProfile:
    category_scores: dict[str, float]
    overall_max: float
    total_flagged_paragraphs: int
    total_skip_regions: int


def compute_paragraph_score(analysis: ParagraphAnalysis) -> float:
    if not analysis.llm_scores:
        return 0.0
    return float(max(s.score for s in analysis.llm_scores.values()))


def compute_chapter_scores(
    paragraphs: list[Paragraph],
    analyses: dict[int, ParagraphAnalysis],
) -> dict[int, ChapterScore]:
    """Aggregate scores per chapter."""
    chapters: dict[int, list[float]] = {}
    chapter_titles: dict[int, str] = {}
    chapter_cats: dict[int, set[str]] = {}

    for para in paragraphs:
        cnum = para.chapter_num
        chapter_titles.setdefault(cnum, para.chapter_title)
        chapter_cats.setdefault(cnum, set())
        chapters.setdefault(cnum, [])

        if para.id in analyses:
            score = compute_paragraph_score(analyses[para.id])
            chapters[cnum].append(score)
            for cat in analyses[para.id].flagged_categories():
                chapter_cats[cnum].add(cat)
        else:
            chapters[cnum].append(0.0)

    result: dict[int, ChapterScore] = {}
    for cnum, scores in chapters.items():
        non_zero = sorted([s for s in scores if s > 0], reverse=True)
        top_q = non_zero[: max(1, len(non_zero) // 4)] if non_zero else [0.0]
        result[cnum] = ChapterScore(
            chapter_num=cnum,
            chapter_title=chapter_titles[cnum],
            max_score=max(scores) if scores else 0.0,
            mean_top_quartile=sum(top_q) / len(top_q),
            categories=sorted(chapter_cats[cnum]),
        )

    return result


def compute_book_profile(analyses: dict[int, ParagraphAnalysis]) -> BookProfile:
    cat_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}

    for analysis in analyses.values():
        for cat in CATEGORIES:
            if cat in analysis.llm_scores:
                cat_scores[cat].append(float(analysis.llm_scores[cat].score))

    category_means = {
        cat: (sum(scores) / len(scores) if scores else 0.0)
        for cat, scores in cat_scores.items()
    }
    overall_max = max((max(s) for s in cat_scores.values() if s), default=0.0)

    return BookProfile(
        category_scores=category_means,
        overall_max=overall_max,
        total_flagged_paragraphs=sum(
            1 for a in analyses.values() if a.max_score() > 0
        ),
        total_skip_regions=0,  # filled in after merge_skip_regions
    )


def merge_skip_regions(
    paragraphs: list[Paragraph],
    analyses: dict[int, ParagraphAnalysis],
    age_band: str,
    gap_fill: int = 3,
) -> list[SkipRegion]:
    """Merge adjacent flagged paragraphs into contiguous skip blocks."""
    threshold = AGE_BANDS[age_band]

    # Collect (global_id, score) for paragraphs above threshold
    flagged: set[int] = {
        para.id
        for para in paragraphs
        if para.id in analyses and compute_paragraph_score(analyses[para.id]) > threshold
    }

    if not flagged:
        return []

    # Group consecutive flagged paragraph IDs (by global order), applying gap-fill
    sorted_ids = sorted(flagged)
    groups: list[list[int]] = [[sorted_ids[0]]]

    for pid in sorted_ids[1:]:
        last = groups[-1][-1]
        # Gap between last flagged and this one
        gap = pid - last - 1
        if gap <= gap_fill:
            groups[-1].append(pid)
        else:
            groups.append([pid])

    para_by_id: dict[int, Paragraph] = {p.id: p for p in paragraphs}

    skip_regions: list[SkipRegion] = []
    for group in groups:
        first_id, last_id = group[0], group[-1]
        first_para = para_by_id[first_id]
        last_para = para_by_id[last_id]

        # Collect all categories and quotes across the group
        cats: set[str] = set()
        quotes: list[str] = []
        max_sev = 0.0
        for pid in group:
            if pid in analyses:
                an = analyses[pid]
                max_sev = max(max_sev, compute_paragraph_score(an))
                cats.update(an.flagged_categories())
                for cat_score in an.llm_scores.values():
                    if cat_score.exact_quote:
                        quotes.append(cat_score.exact_quote)

        # Build summary from rationales of highest-scoring paragraphs
        top_pid = max(
            (pid for pid in group if pid in analyses),
            key=lambda pid: compute_paragraph_score(analyses[pid]),
            default=group[0],
        )
        summary_parts = []
        if top_pid in analyses:
            for cat, cs in analyses[top_pid].llm_scores.items():
                if cs.score > 0:
                    summary_parts.append(cs.rationale)
        summary = " ".join(summary_parts[:2]) or "Flagged content"

        skip_regions.append(
            SkipRegion(
                chapter_num=first_para.chapter_num,
                chapter_title=first_para.chapter_title,
                para_range=(first_id, last_id),
                para_indices=(first_para.paragraph_index, last_para.paragraph_index),
                categories=sorted(cats),
                max_severity=round(max_sev, 1),
                summary=summary,
                quotes=list(dict.fromkeys(quotes))[:5],
            )
        )

    return skip_regions
