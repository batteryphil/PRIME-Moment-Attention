"""
PRIME-Novelist: Autonomous Long-Form Fiction & Creative Writing Engine
Pivots PRIME's multi-scale memory and autonomous daemon architecture
toward long-form novel generation, chapter planning, character state tracking,
and manuscript assembly targeting a 50,000-word full novel.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

NOVEL_DIR = Path(__file__).resolve().parents[1] / "novels" / "tethered_in_smoke_and_sin"
CHAPTERS_DIR = NOVEL_DIR / "chapters"
BOOK_BIBLE_PATH = NOVEL_DIR / "BOOK_BIBLE.json"
MANUSCRIPT_PATH = NOVEL_DIR / "MANUSCRIPT.md"
VAULT_DB_PATH = Path(__file__).resolve().parents[1] / "scout" / "vault" / "scout_vault.db"

class PrimeNovelist:
    def __init__(self):
        CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        self.bible = self._load_bible()
        self._init_db()

    def _load_bible(self) -> Dict[str, Any]:
        if BOOK_BIBLE_PATH.exists():
            return json.loads(BOOK_BIBLE_PATH.read_text(encoding="utf-8"))
        return {}

    def _init_db(self):
        conn = sqlite3.connect(VAULT_DB_PATH)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS novel_chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_title TEXT,
            chapter_number INTEGER UNIQUE,
            title TEXT,
            word_count INTEGER,
            content TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS novel_progress (
            id INTEGER PRIMARY KEY,
            novel_title TEXT,
            total_words INTEGER,
            target_words INTEGER,
            completed_chapters INTEGER,
            total_chapters INTEGER,
            current_act INTEGER,
            status TEXT,
            last_updated TEXT
        );
        """)
        conn.commit()
        conn.close()

    def get_progress(self) -> Dict[str, Any]:
        """Calculates current novel progress from chapter files."""
        chapters = self.bible.get("chapters", [])
        total_chapters = len(chapters)
        
        written_chapters = sorted(CHAPTERS_DIR.glob("chapter_*.md"))
        total_words = 0
        chapter_details = []

        for ch_file in written_chapters:
            text = ch_file.read_text(encoding="utf-8")
            words = len(text.split())
            total_words += words
            num = int(ch_file.stem.split("_")[1])
            title = ""
            for ch in chapters:
                if ch["chapter"] == num:
                    title = ch["title"]
                    break
            chapter_details.append({
                "chapter": num,
                "title": title or ch_file.stem,
                "words": words,
                "path": str(ch_file)
            })

        target_words = self.bible.get("target_word_count", 50000)
        pct = round((total_words / target_words) * 100.0, 1) if target_words > 0 else 0.0

        return {
            "title": self.bible.get("title", "Tethered in Smoke and Sin"),
            "genre": self.bible.get("genre", "Spicy Romantasy"),
            "total_words": total_words,
            "target_words": target_words,
            "percent_complete": pct,
            "completed_chapters": len(written_chapters),
            "total_chapters": total_chapters,
            "chapter_details": chapter_details
        }

    def assemble_manuscript(self) -> str:
        """Assembles all written chapters into a single master MANUSCRIPT.md."""
        written_chapters = sorted(CHAPTERS_DIR.glob("chapter_*.md"))
        lines = [
            f"# {self.bible.get('title', 'Tethered in Smoke and Sin')}",
            f"### A {self.bible.get('genre', 'Romantasy Novel')}",
            "",
            "---",
            ""
        ]

        for ch_file in written_chapters:
            content = ch_file.read_text(encoding="utf-8")
            lines.append(content)
            lines.append("\n\n---\n\n")

        full_manuscript = "\n".join(lines)
        MANUSCRIPT_PATH.write_text(full_manuscript, encoding="utf-8")
        return full_manuscript

    def sync_to_database(self):
        """Syncs all written chapters and progress into SQLite vault."""
        progress = self.get_progress()
        conn = sqlite3.connect(VAULT_DB_PATH)
        cur = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()

        for ch in progress["chapter_details"]:
            file_path = Path(ch["path"])
            content = file_path.read_text(encoding="utf-8")
            cur.execute("""
            INSERT INTO novel_chapters (novel_title, chapter_number, title, word_count, content, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'COMPLETED', ?, ?)
            ON CONFLICT(chapter_number) DO UPDATE SET
                word_count=excluded.word_count,
                content=excluded.content,
                updated_at=excluded.updated_at;
            """, (progress["title"], ch["chapter"], ch["title"], ch["words"], content, now_str, now_str))

        cur.execute("""
        INSERT INTO novel_progress (id, novel_title, total_words, target_words, completed_chapters, total_chapters, current_act, status, last_updated)
        VALUES (1, ?, ?, ?, ?, ?, 1, 'WRITING', ?)
        ON CONFLICT(id) DO UPDATE SET
            total_words=excluded.total_words,
            completed_chapters=excluded.completed_chapters,
            last_updated=excluded.last_updated;
        """, (progress["title"], progress["total_words"], progress["target_words"], progress["completed_chapters"], progress["total_chapters"], now_str))

        conn.commit()
        conn.close()

if __name__ == "__main__":
    novelist = PrimeNovelist()
    novelist.assemble_manuscript()
    novelist.sync_to_database()
    prog = novelist.get_progress()
    print(f"[*] Novel: {prog['title']}")
    print(f"[*] Progress: {prog['total_words']:,} / {prog['target_words']:,} words ({prog['percent_complete']}%)")
    print(f"[*] Completed Chapters: {prog['completed_chapters']} / {prog['total_chapters']}")
    for ch in prog['chapter_details']:
        print(f"    - Chapter {ch['chapter']}: {ch['title']} ({ch['words']:,} words)")
