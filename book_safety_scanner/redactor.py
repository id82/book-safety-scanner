"""EPUB redaction: replace skip regions with Sonnet-generated plot bridges."""

import sqlite3
import warnings
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

from .ingestion import _clean_text
from .pipeline import _run_claude

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")

# ── Bridge generation ─────────────────────────────────────────────────────────

BRIDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "bridge": {
            "type": "string",
            "description": "1-2 sentence age-appropriate plot bridge replacing the removed passage",
        }
    },
    "required": ["bridge"],
}

BRIDGE_SYSTEM_PROMPT = """\
You are editing a novel to make it suitable for readers aged 10-12. A passage \
has been removed because it contains adult content. Write a short, seamless \
plot bridge to replace it.

Rules:
- 1 to 2 sentences maximum
- Maintain narrative continuity so the reader feels no gap
- Preserve any plot-essential information: character revelations, decisions, \
emotional shifts, outcomes
- Match the style and tone of the surrounding text
- Do NOT reference, hint at, or allude to the removed content in any way
- Do NOT reproduce, paraphrase, or echo any wording from the passage description you are given
- Do NOT use meta-language like "[scene omitted]" or "later that evening"
- Output only the bridge sentence(s), nothing else\
"""


def _strip_quotes(text: str) -> str:
    """Remove any quoted substrings so exact wording cannot leak into the bridge."""
    import re
    # Remove content inside single or double typographic/straight quotes
    text = re.sub(r"['‘’][^'‘’]{0,300}['‘’]", "", text)
    text = re.sub(r'"[^"]{0,300}"', "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def generate_bridge(
    summary: str,
    before_texts: list[str],
    after_texts: list[str],
) -> str:
    """Call Sonnet to write a plot bridge for a removed skip region."""
    before_block = "\n".join(f"  {t}" for t in before_texts[-2:])
    after_block = "\n".join(f"  {t}" for t in after_texts[:2])

    prompt = (
        f"Removed passage contained: {_strip_quotes(summary)}\n\n"
        f"Text immediately before the removed passage:\n{before_block}\n\n"
        f"Text immediately after the removed passage:\n{after_block}\n\n"
        "Write a seamless 1-2 sentence bridge."
    )

    result = _run_claude(prompt, BRIDGE_SCHEMA, model="sonnet")
    if result and result.get("bridge"):
        return result["bridge"].strip()
    return ""


# ── EPUB element mapping ──────────────────────────────────────────────────────

def _is_in_excluded_tag(element) -> bool:
    for parent in element.parents:
        if getattr(parent, "name", None) in ("script", "style", "nav"):
            return True
    return False


def _iter_para_elements(soup: BeautifulSoup):
    """Yield paragraph BeautifulSoup elements using the same logic as ingestion."""
    for element in soup.find_all(["h1", "h2", "h3", "p", "div"]):
        if _is_in_excluded_tag(element):
            continue
        if element.name == "div" and element.find(["p", "h1", "h2", "h3", "div"]):
            continue
        text = _clean_text(element.get_text(separator=" "))
        if not text or len(text) < 15:
            continue
        if element.name in ("h1", "h2", "h3"):
            continue
        yield element


def build_element_map(
    epub_path: str | Path,
) -> tuple[epub.EpubBook, dict[int, tuple[str, object]], dict[str, tuple[object, object]]]:
    """
    Parse the EPUB and return:
      - the EpubBook object
      - element_map: {global_para_id: (item_name, bs4_element)}
      - item_soups: {item_name: (EpubHtml_item, BeautifulSoup)}
    """
    book = epub.read_epub(str(epub_path))

    element_map: dict[int, tuple[str, object]] = {}
    item_soups: dict[str, tuple[object, object]] = {}
    global_id = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        para_elements = list(_iter_para_elements(soup))

        if not para_elements:
            continue

        item_name = item.get_name()
        item_soups[item_name] = (item, soup)

        for element in para_elements:
            element_map[global_id] = (item_name, element)
            global_id += 1

    return book, element_map, item_soups


# ── Context fetching from DB ──────────────────────────────────────────────────

def get_context_texts(
    db_path: str | Path,
    start_id: int,
    end_id: int,
    window: int = 3,
) -> tuple[list[str], list[str]]:
    """Return (before_texts, after_texts) around a skip region."""
    conn = sqlite3.connect(str(db_path))
    before = conn.execute(
        "SELECT text FROM paragraphs WHERE id < ? ORDER BY id DESC LIMIT ?",
        (start_id, window),
    ).fetchall()
    after = conn.execute(
        "SELECT text FROM paragraphs WHERE id > ? ORDER BY id ASC LIMIT ?",
        (end_id, window),
    ).fetchall()
    conn.close()
    return (
        [r[0] for r in reversed(before)],
        [r[0] for r in after],
    )


# ── EPUB rewriting ────────────────────────────────────────────────────────────

def rewrite_epub(
    epub_path: str | Path,
    skip_regions: list[dict],
    bridges: list[str],
    output_path: str | Path,
) -> None:
    """
    Rewrite the EPUB, replacing each skip region with its bridge text.
    Paragraphs in the skip region are removed; the bridge is inserted at the
    position of the first paragraph in the region.
    """
    book, element_map, item_soups = build_element_map(epub_path)
    modified_items: set[str] = set()

    for skip_region, bridge_text in zip(skip_regions, bridges):
        start_id, end_id = skip_region["paragraph_range"]
        para_ids = range(start_id, end_id + 1)

        # Find the first para_id that exists in the element map
        first_mapped = next((pid for pid in para_ids if pid in element_map), None)
        if first_mapped is None:
            continue

        for pid in para_ids:
            if pid not in element_map:
                continue
            item_name, element = element_map[pid]
            modified_items.add(item_name)

            if pid == first_mapped and bridge_text:
                _, soup = item_soups[item_name]
                new_p = soup.new_tag("p")
                new_p["class"] = "content-bridge"
                new_p.string = bridge_text
                element.replace_with(new_p)
            else:
                element.decompose()

    # Flush modified soups back to their EPUB items
    for item_name in modified_items:
        item, soup = item_soups[item_name]
        item.set_content(str(soup).encode("utf-8"))

    # ebooklib NCX writer fails if any TOC entry has uid=None
    _fix_toc_uids(book.toc)
    epub.write_epub(str(output_path), book)


def _fix_toc_uids(toc, _counter: list | None = None) -> None:
    """Assign placeholder UIDs to any TOC entries missing one (in-place)."""
    if _counter is None:
        _counter = [0]
    for item in toc:
        if isinstance(item, tuple):
            section, children = item
            if getattr(section, "uid", None) is None:
                section.uid = f"navpoint-{_counter[0]}"
                _counter[0] += 1
            _fix_toc_uids(children, _counter)
        else:
            if getattr(item, "uid", None) is None:
                item.uid = f"navpoint-{_counter[0]}"
                _counter[0] += 1
