"""SQLite caching layer for chunks and LLM analysis results."""

import json
import sqlite3
from pathlib import Path

from .ingestion import Paragraph
from .taxonomy import CATEGORIES

SCHEMA = """
CREATE TABLE IF NOT EXISTS paragraphs (
    id          INTEGER PRIMARY KEY,
    chapter_num INTEGER NOT NULL,
    chapter_title TEXT NOT NULL,
    para_index  INTEGER NOT NULL,
    text        TEXT NOT NULL,
    word_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS keyword_flags (
    para_id     INTEGER NOT NULL,
    category    TEXT NOT NULL,
    matched_text TEXT NOT NULL,
    PRIMARY KEY (para_id, category, matched_text),
    FOREIGN KEY (para_id) REFERENCES paragraphs(id)
);

CREATE TABLE IF NOT EXISTS llm_results (
    para_id     INTEGER PRIMARY KEY,
    raw_json    TEXT NOT NULL,
    FOREIGN KEY (para_id) REFERENCES paragraphs(id)
);

CREATE TABLE IF NOT EXISTS chapter_results (
    chapter_num INTEGER PRIMARY KEY,
    raw_json    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bridges (
    start_id    INTEGER NOT NULL,
    end_id      INTEGER NOT NULL,
    bridge_text TEXT NOT NULL,
    PRIMARY KEY (start_id, end_id)
);
"""


class ScanDatabase:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # --- paragraphs ---

    def has_paragraphs(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()
        return row[0] > 0

    def insert_paragraphs(self, paragraphs: list[Paragraph]):
        self._conn.executemany(
            "INSERT OR IGNORE INTO paragraphs VALUES (?,?,?,?,?,?)",
            [
                (p.id, p.chapter_num, p.chapter_title, p.paragraph_index, p.text, p.word_count)
                for p in paragraphs
            ],
        )
        self._conn.commit()

    def load_paragraphs(self) -> list[Paragraph]:
        rows = self._conn.execute(
            "SELECT id, chapter_num, chapter_title, para_index, text FROM paragraphs ORDER BY id"
        ).fetchall()
        return [
            Paragraph(
                id=r["id"],
                chapter_num=r["chapter_num"],
                chapter_title=r["chapter_title"],
                paragraph_index=r["para_index"],
                text=r["text"],
            )
            for r in rows
        ]

    # --- keyword flags ---

    def insert_keyword_flags(self, para_id: int, hits: list):
        self._conn.executemany(
            "INSERT OR IGNORE INTO keyword_flags VALUES (?,?,?)",
            [(para_id, h.category, h.matched_text) for h in hits],
        )
        self._conn.commit()

    def get_keyword_flagged_ids(self) -> set[int]:
        rows = self._conn.execute("SELECT DISTINCT para_id FROM keyword_flags").fetchall()
        return {r["para_id"] for r in rows}

    def get_keyword_categories(self, para_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT category FROM keyword_flags WHERE para_id=?", (para_id,)
        ).fetchall()
        return [r["category"] for r in rows]

    # --- LLM results ---

    def get_analyzed_ids(self) -> set[int]:
        rows = self._conn.execute("SELECT para_id FROM llm_results").fetchall()
        return {r["para_id"] for r in rows}

    def insert_llm_result(self, para_id: int, result: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_results VALUES (?,?)",
            (para_id, json.dumps(result)),
        )
        self._conn.commit()

    def load_llm_results(self) -> dict[int, dict]:
        rows = self._conn.execute("SELECT para_id, raw_json FROM llm_results").fetchall()
        return {r["para_id"]: json.loads(r["raw_json"]) for r in rows}

    # --- chapter results ---

    def get_analyzed_chapters(self) -> set[int]:
        rows = self._conn.execute("SELECT chapter_num FROM chapter_results").fetchall()
        return {r["chapter_num"] for r in rows}

    def insert_chapter_result(self, chapter_num: int, result: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO chapter_results VALUES (?,?)",
            (chapter_num, json.dumps(result)),
        )
        self._conn.commit()

    def load_chapter_results(self) -> dict[int, dict]:
        rows = self._conn.execute(
            "SELECT chapter_num, raw_json FROM chapter_results"
        ).fetchall()
        return {r["chapter_num"]: json.loads(r["raw_json"]) for r in rows}

    # --- bridges ---

    def get_bridge(self, start_id: int, end_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT bridge_text FROM bridges WHERE start_id=? AND end_id=?",
            (start_id, end_id),
        ).fetchone()
        return row["bridge_text"] if row else None

    def insert_bridge(self, start_id: int, end_id: int, bridge_text: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO bridges VALUES (?,?,?)",
            (start_id, end_id, bridge_text),
        )
        self._conn.commit()

    def load_bridges(self) -> dict[tuple[int, int], str]:
        rows = self._conn.execute(
            "SELECT start_id, end_id, bridge_text FROM bridges"
        ).fetchall()
        return {(r["start_id"], r["end_id"]): r["bridge_text"] for r in rows}
