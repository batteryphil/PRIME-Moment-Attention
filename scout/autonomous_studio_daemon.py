"""
PRIME Autonomous Romantasy Studio Daemon
24/7 background engine that autonomously researches, outlines, writes,
enriches, packages (EPUB/PDF/HTML), and catalogs novels in the 50k to 250k word range.
"""

import sys
import time
import json
import sqlite3
import pathlib
from datetime import datetime, timezone
from typing import Dict, Any, List

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scout.romantasy_studio import RomantasyStudio, NOVELS_ROOT, VAULT_DB_PATH
from scout.novel_researcher import RomantasyResearcher

def run_studio_daemon(cadence_seconds: float = 60.0):
    print("=" * 80)
    print("👑 PRIME Autonomous Romantasy Publishing Studio")
    print("✨ Multi-Book Factory: 50,000 to 250,000 Word Autonomous Production")
    print(f"⏱️  Heartbeat: Every {cadence_seconds:.1f}s")
    print("=" * 80)

    studio = RomantasyStudio()
    researcher = RomantasyResearcher()

    cycle = 1
    while True:
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            catalog = studio.get_library_catalog()

            print(f"\n[+] [{now_str}] Studio Cycle #{cycle}")
            active_book = None
            for b in catalog:
                status = b["status"]
                pct = round((b["current_words"] / b["target_words"]) * 100.0, 1) if b["target_words"] > 0 else 0.0
                print(f"    📚 [{status:11s}] {b['title'][:32]:32s} | {b['current_words']:,} / {b['target_words']:,}w ({pct}%) | Heat: {'🌶️'*b['heat_level']}")
                if status in ["RESEARCHING", "WRITING", "QUEUED"] and active_book is None:
                    active_book = b

            if active_book:
                slug = active_book["slug"]
                title = active_book["title"]
                status = active_book["status"]

                if status == "QUEUED":
                    print(f"\n[*] Initiating Autonomous Research Phase for: '{title}'...")
                    researcher.generate_research_dossier(slug)
                    conn = sqlite3.connect(VAULT_DB_PATH)
                    cur = conn.cursor()
                    cur.execute("UPDATE novel_library SET status = 'RESEARCHING' WHERE slug = ?;", (slug,))
                    conn.commit()
                    conn.close()

                elif status == "RESEARCHING":
                    print(f"\n[*] Generating Architectural Book Bible for: '{title}' ({active_book['target_words']:,} words, {active_book['total_chapters']} chapters)...")
                    # Build Book Bible
                    book_dir = NOVELS_ROOT / slug
                    book_dir.mkdir(parents=True, exist_ok=True)
                    bible_p = book_dir / "BOOK_BIBLE.json"
                    
                    if not bible_p.exists():
                        dossier_p = book_dir / "RESEARCH_DOSSIER.json"
                        dossier = json.loads(dossier_p.read_text(encoding="utf-8")) if dossier_p.exists() else {}
                        
                        bible = {
                            "slug": slug,
                            "title": title,
                            "target_words": active_book["target_words"],
                            "total_chapters": active_book["total_chapters"],
                            "heat_level": active_book["heat_level"],
                            "subgenre": active_book["subgenre"],
                            "research_dossier": dossier.get("research_topics", {}),
                            "chapters": [
                                {
                                    "chapter": i + 1,
                                    "target_words": round(active_book["target_words"] / active_book["total_chapters"]),
                                    "act": 1 if i < active_book["total_chapters"] * 0.25 else (2 if i < active_book["total_chapters"] * 0.75 else 3),
                                    "status": "QUEUED"
                                }
                                for i in range(active_book["total_chapters"])
                            ]
                        }
                        bible_p.write_text(json.dumps(bible, indent=2), encoding="utf-8")
                        print(f"[✓] Book Bible created at {bible_p}")

                    conn = sqlite3.connect(VAULT_DB_PATH)
                    cur = conn.cursor()
                    cur.execute("UPDATE novel_library SET status = 'WRITING' WHERE slug = ?;", (slug,))
                    conn.commit()
                    conn.close()

                elif status == "WRITING":
                    print(f"[*] Autonomous local writing pipeline active on '{title}'...")
                    book_dir = NOVELS_ROOT / slug
                    chapters_dir = book_dir / "chapters"
                    chapters_dir.mkdir(parents=True, exist_ok=True)

                    total_chapters = active_book["total_chapters"]
                    written_files = sorted(chapters_dir.glob("chapter_*.md"))
                    written_nums = set()
                    for f in written_files:
                        try:
                            num = int(f.stem.split("_")[1])
                            written_nums.add(num)
                        except Exception:
                            pass

                    next_ch = None
                    for ch_idx in range(1, total_chapters + 1):
                        if ch_idx not in written_nums:
                            next_ch = ch_idx
                            break

                    if next_ch is not None:
                        print(f"[*] [GPU Production] Drafting Chapter {next_ch}/{total_chapters} for '{title}' locally on AMD ROCm...")
                        from scout.prime_local_author import PrimeLocalAuthor
                        local_author = PrimeLocalAuthor()
                        ch_res = local_author.draft_chapter(slug, next_ch)

                        # Recalculate total words
                        all_chapters = sorted(chapters_dir.glob("chapter_*.md"))
                        total_words = sum(len(cf.read_text(encoding="utf-8").split()) for cf in all_chapters)
                        completed_count = len(all_chapters)

                        # Sync to SQLite
                        conn = sqlite3.connect(VAULT_DB_PATH)
                        cur = conn.cursor()
                        now_str = datetime.now(timezone.utc).isoformat()
                        cur.execute("""
                        INSERT INTO novel_chapters (novel_title, chapter_number, title, word_count, content, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, 'COMPLETED', ?, ?)
                        ON CONFLICT(novel_title, chapter_number) DO UPDATE SET
                            word_count=excluded.word_count,
                            content=excluded.content,
                            updated_at=excluded.updated_at;
                        """, (title, next_ch, ch_res["title"], ch_res["words"], (chapters_dir / f"chapter_{next_ch:02d}.md").read_text(encoding="utf-8"), now_str, now_str))

                        new_status = "COMPLETED" if completed_count >= total_chapters else "WRITING"
                        cur.execute("""
                        UPDATE novel_library 
                        SET current_words = ?, completed_chapters = ?, status = ?, updated_at = ?
                        WHERE slug = ?;
                        """, (total_words, completed_count, new_status, now_str, slug))
                        conn.commit()
                        conn.close()
                        print(f"[✓] Synced Chapter {next_ch} to database. Novel total: {total_words:,} words ({completed_count}/{total_chapters} ch).")

                        # Auto-update Typeset PDF
                        try:
                            from scout.generate_sample_pdf import build_sample_pdf
                            pdf_p = build_sample_pdf(slug)
                            print(f"[✓] Automatically recompiled typeset PDF: {pdf_p.name}")
                        except Exception as pdf_err:
                            print(f"[!] PDF recompilation warning: {pdf_err}")
                    else:
                        print(f"[✓] All {total_chapters} chapters written for '{title}'! Marking COMPLETED.")
                        conn = sqlite3.connect(VAULT_DB_PATH)
                        cur = conn.cursor()
                        cur.execute("UPDATE novel_library SET status = 'COMPLETED' WHERE slug = ?;", (slug,))
                        conn.commit()
                        conn.close()

            cycle += 1
            time.sleep(cadence_seconds)

        except KeyboardInterrupt:
            print("\n[!] Studio daemon stopped by user.")
            break
        except Exception as e:
            print(f"[!] Studio cycle error: {e}")
            time.sleep(10.0)

if __name__ == "__main__":
    cadence = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    run_studio_daemon(cadence)
