"""
Beyond the Event Horizon — 5-Book Autonomous Series Daemon
==========================================================
Autonomously drafts all 5 books in the Hard Science Fiction Space Opera series
using Sao10K/L3-8B-Stheno-v3.2 on local AMD GPU with PRIME-Moment-Attention recurrence.
Automatically transitions from Book 1 through Book 5 and HALTS when all 250,000 words are complete.
"""

import os
import sys
import time
import json
import sqlite3
import pathlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scout.prime_local_author import get_author, NOVELS_ROOT, VAULT_DB_PATH
from scout.generate_sample_pdf import build_sample_pdf

SERIES_SLUGS = [
    "beyond_the_event_horizon",
    "beyond_the_event_horizon_book_2",
    "beyond_the_event_horizon_book_3",
    "beyond_the_event_horizon_book_4",
    "beyond_the_event_horizon_book_5"
]


def get_series_overview() -> Dict[str, Any]:
    """Inspects all 5 books and returns overall series progress."""
    books = []
    total_series_words = 0
    total_series_chapters = 0

    for idx, slug in enumerate(SERIES_SLUGS, 1):
        b_dir = NOVELS_ROOT / slug
        bible_path = b_dir / "BOOK_BIBLE.json"
        bible = json.loads(bible_path.read_text(encoding="utf-8")) if bible_path.exists() else {}
        
        ch_dir = b_dir / "chapters"
        ch_files = sorted(ch_dir.glob("chapter_*.md")) if ch_dir.exists() else []
        
        b_words = 0
        for cf in ch_files:
            b_words += len(cf.read_text(encoding="utf-8").split())
            
        total_series_words += b_words
        total_series_chapters += len(ch_files)

        books.append({
            "book_number": idx,
            "slug": slug,
            "title": bible.get("title", f"Book {idx}"),
            "completed_chapters": len(ch_files),
            "total_chapters": bible.get("total_chapters", 20),
            "words": b_words,
            "target_words": bible.get("target_words", 50000),
            "status": "COMPLETED" if len(ch_files) >= 20 else ("WRITING" if len(ch_files) > 0 else "QUEUED")
        })

    return {
        "series_title": "Beyond the Event Horizon",
        "total_books": 5,
        "total_series_words": total_series_words,
        "total_series_chapters": total_series_chapters,
        "books": books
    }


def find_next_unwritten_book_and_chapter() -> Optional[Dict[str, Any]]:
    """Identifies the next book and chapter that needs to be written."""
    overview = get_series_overview()
    for b in overview["books"]:
        if b["completed_chapters"] < b["total_chapters"]:
            return {
                "book_number": b["book_number"],
                "slug": b["slug"],
                "title": b["title"],
                "next_chapter": b["completed_chapters"] + 1,
                "completed_chapters": b["completed_chapters"],
                "total_chapters": b["total_chapters"],
                "current_book_words": b["words"],
                "total_series_words": overview["total_series_words"],
                "total_series_chapters": overview["total_series_chapters"]
            }
    return None


def run_space_opera_daemon(max_chapters_to_write: Optional[int] = None):
    print("=" * 80)
    print("🚀 BEYOND THE EVENT HORIZON — 5-Book Space Opera Engine")
    print("🌌 250,000 Words Autonomous Production (5 Books × 20 Chapters)")
    print("💎 Powered by Sao10K/L3-8B-Stheno-v3.2 on local AMD GPU (16.06 GB VRAM)")
    print("=" * 80)

    author = get_author("Sao10K/L3-8B-Stheno-v3.2")
    written_count = 0

    while True:
        target = find_next_unwritten_book_and_chapter()
        
        if target is None:
            overview = get_series_overview()
            print("\n" + "=" * 80)
            print("🏁 [MISSION ACCOMPLISHED] ALL 5 BOOKS IN 'BEYOND THE EVENT HORIZON' ARE COMPLETE!")
            print(f"📊 Total Chapters: {overview['total_series_chapters']} / 100")
            print(f"📚 Total Words: {overview['total_series_words']:,} / 250,000")
            print("=" * 80)
            break

        if max_chapters_to_write is not None and written_count >= max_chapters_to_write:
            print(f"\n[✓] Completed requested batch of {written_count} chapter(s). Pausing daemon.")
            break

        b_num = target["book_number"]
        slug = target["slug"]
        b_title = target["title"]
        ch_num = target["next_chapter"]
        tot_ch = target["total_chapters"]

        print(f"\n" + "-" * 75)
        print(f"[*] [Series Progress: {target['total_series_chapters']}/100 Ch | {target['total_series_words']:,} Words]")
        print(f"[*] Drafting Book {b_num}/5: '{b_title}' — Chapter {ch_num}/{tot_ch}...")
        print("-" * 75)

        t0 = time.time()
        draft_res = author.draft_chapter(slug, ch_num, target_words=2500)
        elapsed = time.time() - t0
        words = draft_res["words"]

        print(f"[✓] Drafted: {words:,} words in {elapsed:.1f}s ({words / (elapsed if elapsed > 0 else 1):.1f} wps)")

        # Sync to SQLite vault
        try:
            conn = sqlite3.connect(VAULT_DB_PATH)
            cur = conn.cursor()
            ch_content = pathlib.Path(draft_res["path"]).read_text(encoding="utf-8")
            full_novel_title = f"Beyond the Event Horizon: {b_title}"
            cur.execute("""
                INSERT INTO novel_chapters (novel_title, chapter_number, title, word_count, content, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (full_novel_title, ch_num, f"Chapter {ch_num}: {draft_res['title']}", words, ch_content, "COMPLETED"))
            
            # Update novel_library entry
            cur.execute("""
                UPDATE novel_library 
                SET completed_chapters = ?, current_words = current_words + ?, status = 'WRITING', updated_at = datetime('now')
                WHERE slug = ?
            """, (ch_num, words, slug))
            conn.commit()
            conn.close()
            print(f"[✓] Synced Chapter {ch_num} to scout_vault.db")
        except Exception as e:
            print(f"[!] Warning: Database sync error: {e}")

        # Sync to BOOK_BIBLE.json
        try:
            bible_p = NOVELS_ROOT / slug / "BOOK_BIBLE.json"
            book_bible = json.loads(bible_p.read_text(encoding="utf-8"))
            for ch_entry in book_bible.get("chapters", []):
                if ch_entry["chapter"] == ch_num:
                    ch_entry["status"] = "COMPLETED"
                    ch_entry["actual_words"] = words
                    break
            bible_p.write_text(json.dumps(book_bible, indent=2), encoding="utf-8")
            print(f"[✓] Synced Chapter {ch_num} to {bible_p.name}")
        except Exception as e:
            print(f"[!] Warning: Book bible sync error: {e}")

        # Update Sample PDF
        try:
            pdf_path = build_sample_pdf(slug)
            print(f"[✓] Typeset Manuscript PDF updated: {pdf_path.name}")
        except Exception as e:
            print(f"[!] Warning: PDF build error: {e}")

        # Check if book completed
        if ch_num == tot_ch:
            print(f"\n🎉 [★] Book {b_num} ('{b_title}') is 100% COMPLETE! (All 20 Chapters Drafted)")
            # Trigger PRIME Sovereign Editor Sweep to polish alias, acoustics, and boundary issues
            try:
                from scout.prime_copy_editor import PrimeCopyEditor
                editor = PrimeCopyEditor()
                print(f"[*] Triggering PRIME Sovereign Editor automated sweep for '{slug}'...")
                editor.edit_novel_batch(slug, apply_fix=True)
            except Exception as e:
                print(f"[!] Editor sweep warning: {e}")

            try:
                conn = sqlite3.connect(VAULT_DB_PATH)
                cur = conn.cursor()
                cur.execute("UPDATE novel_library SET status = 'COMPLETED', updated_at = datetime('now') WHERE slug = ?", (slug,))
                conn.commit()
                conn.close()
            except Exception as e:
                pass

        written_count += 1
        print(f"[*] Cooling GPU for 5 seconds before next chapter...")
        time.sleep(5.0)


if __name__ == "__main__":
    max_ch = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_space_opera_daemon(max_chapters_to_write=max_ch)
