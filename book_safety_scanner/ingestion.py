"""EPUB ingestion and structural parsing."""

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")


@dataclass
class Paragraph:
    id: int
    chapter_num: int
    chapter_title: str
    paragraph_index: int  # within chapter
    text: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())

    def context_label(self) -> str:
        return f"Ch.{self.chapter_num} '{self.chapter_title}' §{self.paragraph_index}"


@dataclass
class BookMeta:
    title: str
    author: str
    path: str
    total_paragraphs: int
    total_words: int


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_html_item(html_bytes: bytes) -> tuple[str | None, list[str]]:
    """Extract (chapter_title, [paragraph_texts]) from an HTML spine item."""
    soup = BeautifulSoup(html_bytes, "lxml")

    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    chapter_title: str | None = None
    paragraphs: list[str] = []

    for element in soup.find_all(["h1", "h2", "h3", "p", "div"]):
        # Only direct text-bearing elements, skip nested divs with block children
        if element.name == "div":
            # Include divs only if they have no block-level children
            if element.find(["p", "h1", "h2", "h3", "div"]):
                continue

        text = _clean_text(element.get_text(separator=" "))
        if not text or len(text) < 15:
            continue

        if element.name in ("h1", "h2", "h3"):
            if chapter_title is None:
                chapter_title = text
        else:
            paragraphs.append(text)

    return chapter_title, paragraphs


def parse_epub(path: str | Path) -> tuple[BookMeta, list[Paragraph]]:
    """Parse EPUB into structured paragraphs with chapter metadata."""
    path = Path(path)
    book = epub.read_epub(str(path))

    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else path.stem
    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else "Unknown"

    paragraphs: list[Paragraph] = []
    global_id = 0
    chapter_num = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        chapter_title, para_texts = _parse_html_item(item.get_content())

        if not para_texts:
            continue

        chapter_num += 1
        resolved_title = chapter_title or f"Chapter {chapter_num}"

        for para_index, text in enumerate(para_texts):
            paragraphs.append(
                Paragraph(
                    id=global_id,
                    chapter_num=chapter_num,
                    chapter_title=resolved_title,
                    paragraph_index=para_index,
                    text=text,
                )
            )
            global_id += 1

    meta = BookMeta(
        title=title,
        author=author,
        path=str(path),
        total_paragraphs=len(paragraphs),
        total_words=sum(p.word_count for p in paragraphs),
    )

    return meta, paragraphs


def build_context_window(
    paragraphs: list[Paragraph], index: int, window: int = 2
) -> str:
    """Return text of the paragraph at index plus ±window neighbours."""
    start = max(0, index - window)
    end = min(len(paragraphs), index + window + 1)
    parts = []
    for i in range(start, end):
        marker = ">>> " if i == index else "    "
        parts.append(f"{marker}{paragraphs[i].text}")
    return "\n\n".join(parts)
