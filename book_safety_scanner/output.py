"""JSON manifest and HTML report generation."""

import json
from pathlib import Path

from .ingestion import BookMeta
from .scoring import BookProfile, ChapterScore, SkipRegion
from .taxonomy import AGE_BANDS, CATEGORY_LABELS

# ── JSON manifest ─────────────────────────────────────────────────────────────

def build_manifest(
    meta: BookMeta,
    age_band: str,
    skip_regions: list[SkipRegion],
    chapter_scores: dict[int, ChapterScore],
    book_profile: BookProfile,
) -> dict:
    return {
        "book": {
            "title": meta.title,
            "author": meta.author,
            "path": meta.path,
            "total_paragraphs": meta.total_paragraphs,
            "total_words": meta.total_words,
        },
        "scan": {
            "age_band": age_band,
            "age_band_threshold": AGE_BANDS[age_band],
            "total_flagged_paragraphs": book_profile.total_flagged_paragraphs,
            "total_skip_regions": len(skip_regions),
        },
        "book_profile": {
            CATEGORY_LABELS[cat]: round(score, 2)
            for cat, score in book_profile.category_scores.items()
        },
        "chapter_summary": [
            {
                "chapter": cs.chapter_num,
                "title": cs.chapter_title,
                "max_score": round(cs.max_score, 1),
                "mean_top_quartile": round(cs.mean_top_quartile, 1),
                "categories": cs.categories,
            }
            for cs in sorted(chapter_scores.values(), key=lambda c: c.chapter_num)
            if cs.max_score > 0
        ],
        "skip_regions": [
            {
                "chapter": sr.chapter_num,
                "chapter_title": sr.chapter_title,
                "paragraph_range": list(sr.para_range),
                "paragraph_indices_in_chapter": list(sr.para_indices),
                "categories": [CATEGORY_LABELS[c] for c in sr.categories],
                "max_severity": sr.max_severity,
                "summary": sr.summary,
                "triggering_quotes": sr.quotes,
            }
            for sr in skip_regions
        ],
    }


def save_manifest(manifest: dict, path: str | Path):
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


# ── HTML report ───────────────────────────────────────────────────────────────

_SEVERITY_COLOUR = {
    0: "#e8f5e9",
    1: "#fff9c4",
    2: "#ffe0b2",
    3: "#ffccbc",
    4: "#ef9a9a",
    5: "#e53935",
}


def _severity_colour(score: float) -> str:
    return _SEVERITY_COLOUR.get(min(5, int(score)), "#fff")


def _category_pills(categories: list[str]) -> str:
    colours = {
        "Violence": "#ef5350",
        "Sexual Content": "#ec407a",
        "Drugs & Alcohol": "#ab47bc",
        "Strong Language": "#5c6bc0",
        "Psychological Horror": "#26a69a",
        "Distressing Themes": "#ff7043",
    }
    pills = []
    for cat in categories:
        colour = colours.get(cat, "#90a4ae")
        pills.append(
            f'<span style="background:{colour};color:#fff;border-radius:4px;'
            f'padding:2px 7px;font-size:0.8em;margin-right:4px">{cat}</span>'
        )
    return "".join(pills)


def build_html_report(manifest: dict) -> str:
    book = manifest["book"]
    scan = manifest["scan"]
    profile = manifest["book_profile"]
    skip_regions = manifest["skip_regions"]

    profile_rows = "\n".join(
        f"<tr><td>{cat}</td>"
        f'<td><div style="background:{_severity_colour(score)};'
        f'border-radius:3px;padding:2px 8px;display:inline-block">'
        f"{score:.2f} / 5</div></td></tr>"
        for cat, score in profile.items()
    )

    region_cards = ""
    for i, sr in enumerate(skip_regions, 1):
        quotes_html = ""
        if sr["triggering_quotes"]:
            quotes_html = "<ul>" + "".join(
                f'<li><em>"{q}"</em></li>' for q in sr["triggering_quotes"]
            ) + "</ul>"

        bg = _severity_colour(sr["max_severity"])
        pills = _category_pills(sr["categories"])
        region_cards += f"""
        <div style="border:1px solid #ddd;border-radius:6px;margin-bottom:16px;
                    border-left:6px solid {bg};padding:14px 18px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong>Region {i}: {sr["chapter_title"]}</strong>
            <span style="font-size:0.9em;color:#666">
              Para {sr["paragraph_range"][0]}–{sr["paragraph_range"][1]} &nbsp;|&nbsp;
              Severity {sr["max_severity"]:.1f}
            </span>
          </div>
          <div style="margin:6px 0">{pills}</div>
          <p style="margin:6px 0;color:#333">{sr["summary"]}</p>
          {quotes_html}
        </div>"""

    no_flags_msg = (
        '<p style="color:#388e3c">No passages exceeded the age-band threshold.</p>'
        if not skip_regions
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Book Safety Report: {book["title"]}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto;
          padding: 0 20px; color: #212121; }}
  h1 {{ border-bottom: 2px solid #1565c0; padding-bottom: 8px; }}
  h2 {{ color: #1565c0; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ padding: 8px 12px; border: 1px solid #e0e0e0; text-align: left; }}
  th {{ background: #e3f2fd; }}
</style>
</head>
<body>

<h1>Book Safety Report</h1>
<table>
  <tr><th>Title</th><td>{book["title"]}</td></tr>
  <tr><th>Author</th><td>{book["author"]}</td></tr>
  <tr><th>Age Band</th><td>{scan["age_band"]} (threshold: {scan["age_band_threshold"]})</td></tr>
  <tr><th>Total Words</th><td>{book["total_words"]:,}</td></tr>
  <tr><th>Flagged Paragraphs</th><td>{scan["total_flagged_paragraphs"]}</td></tr>
  <tr><th>Skip Regions</th><td>{scan["total_skip_regions"]}</td></tr>
</table>

<h2>Book Profile</h2>
<table>
  <tr><th>Category</th><th>Mean Score</th></tr>
  {profile_rows}
</table>

<h2>Skip Regions</h2>
{no_flags_msg}
{region_cards}

</body>
</html>"""


def save_html_report(manifest: dict, path: str | Path):
    Path(path).write_text(build_html_report(manifest), encoding="utf-8")
