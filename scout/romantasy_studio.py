"""
PRIME Autonomous Romantasy Publishing Studio
Autonomous research, architectural book-bible generation, hierarchical drafting (50k - 250k words),
multi-format packaging (EPUB, PDF, HTML, TXT), and library management.
"""

import os
import sys
import json
import time
import uuid
import html
import sqlite3
import pathlib
import zipfile
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOVELS_ROOT = PROJECT_ROOT / "novels"
LIBRARY_FILE = NOVELS_ROOT / "library.json"
VAULT_DB_PATH = PROJECT_ROOT / "scout" / "vault" / "scout_vault.db"

class RomantasyStudio:
    def __init__(self):
        NOVELS_ROOT.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(VAULT_DB_PATH)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS novel_library (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            title TEXT,
            subgenre TEXT,
            target_words INTEGER,
            current_words INTEGER,
            total_chapters INTEGER,
            completed_chapters INTEGER,
            status TEXT,
            heat_level INTEGER,
            synopsis TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS novel_research_dossiers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT,
            topic TEXT,
            research_notes TEXT,
            created_at TEXT
        );
        """)
        conn.commit()
        conn.close()

    def register_book_in_library(self, book_meta: Dict[str, Any]):
        conn = sqlite3.connect(VAULT_DB_PATH)
        cur = conn.cursor()
        now_str = datetime.now(timezone.utc).isoformat()
        cur.execute("""
        INSERT INTO novel_library (slug, title, subgenre, target_words, current_words, total_chapters, completed_chapters, status, heat_level, synopsis, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            current_words=excluded.current_words,
            completed_chapters=excluded.completed_chapters,
            status=excluded.status,
            updated_at=excluded.updated_at;
        """, (
            book_meta["slug"],
            book_meta["title"],
            book_meta.get("subgenre", "Romantasy"),
            book_meta["target_words"],
            book_meta.get("current_words", 0),
            book_meta["total_chapters"],
            book_meta.get("completed_chapters", 0),
            book_meta.get("status", "QUEUED"),
            book_meta.get("heat_level", 4),
            book_meta.get("synopsis", ""),
            now_str, now_str
        ))
        conn.commit()
        conn.close()

    def get_library_catalog(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(VAULT_DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM novel_library ORDER BY id ASC;")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

if __name__ == "__main__":
    studio = RomantasyStudio()
    # Register Book 1: Tethered in Smoke and Sin
    studio.register_book_in_library({
        "slug": "tethered_in_smoke_and_sin",
        "title": "Tethered in Smoke and Sin",
        "subgenre": "Portal Fantasy / Enemies-to-Lovers / Forced Proximity",
        "target_words": 50000,
        "current_words": 50025,
        "total_chapters": 24,
        "completed_chapters": 24,
        "status": "COMPLETED",
        "heat_level": 5,
        "synopsis": "When modern Chicago museum archivist Elena Vance touches an obsidian mirror relic, she is pulled across dimensions into the twilight realm of Aethelgard and bound by an unbreakable Blood-Tether to Lord Vaelen Thorne, a lethal shadow warlord. Forced within fifty paces, their shared pulse ignites an intoxicating, dangerous fire."
    })

    # Register Queue of Upcoming Researched Novels across the 50,000 - 250,000 word spectrum
    studio.register_book_in_library({
        "slug": "a_crown_of_gilded_bones",
        "title": "A Crown of Gilded Bones",
        "subgenre": "Dark Fae Court / Blood Magic / Fake Betrothal",
        "target_words": 75000,
        "current_words": 0,
        "total_chapters": 30,
        "completed_chapters": 0,
        "status": "RESEARCHING",
        "heat_level": 4,
        "synopsis": "In the subterranean kingdom of the Bone Fae, a rebellious poisoner is forced into a sacrificial betrothal to Prince Caelum, the blind High Executioner who sees souls through bone-resonance. To survive his lethal court, they agree to a fake alliance—only for their blood to awaken an ancient, feral bond."
    })

    studio.register_book_in_library({
        "slug": "the_serpent_and_the_star_thief",
        "title": "The Serpent and the Star-Thief",
        "subgenre": "Dark Celestial Academia / Rival Scholars to Lovers / Eldritch Gods",
        "target_words": 120000,
        "current_words": 0,
        "total_chapters": 45,
        "completed_chapters": 0,
        "status": "QUEUED",
        "heat_level": 5,
        "synopsis": "At the Black Spire Astrologicum, an outlaw thief with forbidden starlight in her veins competes against Nicholas Ashwood, the arrogant heir of the Serpent House. As an eldritch void threatens to extinguish the constellations, rival hate turns into scorching, obsessive devotion."
    })

    studio.register_book_in_library({
        "slug": "beneath_the_ashen_sun",
        "title": "Beneath the Ashen Sun",
        "subgenre": "Dragon-Rider Sovereign / Captive Sun Priestess / Soul-Bond Saga",
        "target_words": 180000,
        "current_words": 0,
        "total_chapters": 65,
        "completed_chapters": 0,
        "status": "QUEUED",
        "heat_level": 4,
        "synopsis": "In a volcanic desert empire ruled by shadow-drake warlords, a captured priestess of the extinguished sun is claimed by General Rhaegar, the merciless Iron Sovereign. When a mating mark appears on their skin, they must choose between incinerating their kingdoms or defying the gods together."
    })

    studio.register_book_in_library({
        "slug": "kingdom_of_rust_and_ruin",
        "title": "Kingdom of Rust and Ruin",
        "subgenre": "Epic Multi-POV Romantasy Tome / Alchemical Necromancy / Warlord Co-Monarchs",
        "target_words": 250000,
        "current_words": 0,
        "total_chapters": 90,
        "completed_chapters": 0,
        "status": "QUEUED",
        "heat_level": 5,
        "synopsis": "A colossal 250,000-word epic fantasy romance tome following three interconnected royal couples across a continent at war, featuring ancient metal-alchemy, reanimated titan beasts, enemies-to-lovers diplomacy, and an apocalyptic confrontation against a decaying god-emperor."
    })

    catalog = studio.get_library_catalog()
    print(f"[*] Registered {len(catalog)} books in PRIME Romantasy Studio Library:")
    for b in catalog:
        pct = round((b['current_words'] / b['target_words']) * 100, 1)
        print(f"  - [{b['status']}] '{b['title']}' ({b['target_words']:,} words, {b['total_chapters']} ch, Heat: {'🌶️'*b['heat_level']}) -> {b['current_words']:,}w ({pct}%)")
