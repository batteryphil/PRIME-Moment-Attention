"""
PRIME Sovereign Copy Editor (Tier 1 + Tier 2)
=============================================
Orchestrates automated editorial passes, diagnostics, and repairs
across entire novels and individual chapters.
"""

import sys
import json
import sqlite3
import pathlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scout.editor_linter import lint_chapter_file, lint_and_repair_text
from scout.generate_sample_pdf import build_sample_pdf
from scout.prime_local_author import NOVELS_ROOT, VAULT_DB_PATH

COPY_EDITOR_SYSTEM_PROMPT = """You are a senior commercial science fiction copy editor specializing in space opera, military sci-fi, and high-tension character dynamics.

YOUR EDITORIAL DIRECTIVES:
- Preserve the author's original voice, style, and tone completely.
- Fix character alias collisions (Vex is Captain Astrid Ross; Leo Mercer is Apex-One). Never allow Astrid/Vex to refer to herself as a separate person.
- Enforce vacuum physics: zero sound in space. Sound must be conducted through hull plating, bones, or cockpit air.
- Eliminate repetitive clichés (e.g., 'a bitter taste in her mouth', 'time seemed to stretch', 'balls of void').
- Ensure dialogue attribution is crisp and natural.
- Output ONLY the polished chapter markdown without commentary, notes, or explanations."""


class PrimeCopyEditor:
    """
    Two-tiered copy-editing engine for automated manuscript refinement.
    """
    def __init__(self):
        self.novels_root = NOVELS_ROOT
        self.db_path = VAULT_DB_PATH

    def edit_chapter(self, slug: str, chapter_num: int, apply_fix: bool = True) -> Dict[str, Any]:
        """Lints and repairs a single chapter file."""
        ch_file = self.novels_root / slug / "chapters" / f"chapter_{chapter_num:02d}.md"
        if not ch_file.exists():
            raise FileNotFoundError(f"Chapter file not found: {ch_file}")

        lint_res = lint_chapter_file(ch_file, apply_fix=apply_fix)

        # Sync back to DB if changed
        if lint_res["changed"] and apply_fix:
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                new_text = ch_file.read_text(encoding="utf-8")
                words = len(new_text.split())
                cur.execute("""
                    UPDATE novel_chapters
                    SET content = ?, word_count = ?, updated_at = datetime('now')
                    WHERE chapter_number = ? AND (novel_title LIKE ? OR novel_title LIKE ?)
                """, (new_text, words, chapter_num, f"%{slug}%", "%Beyond the Event Horizon%"))
                conn.commit()
                conn.close()
            except Exception as e:
                lint_res["db_error"] = str(e)

        return lint_res

    def edit_novel_batch(self, slug: str, apply_fix: bool = True) -> Dict[str, Any]:
        """Performs a comprehensive editorial sweep over all chapters in a novel."""
        ch_dir = self.novels_root / slug / "chapters"
        if not ch_dir.exists():
            raise FileNotFoundError(f"Chapters directory not found: {ch_dir}")

        ch_files = sorted(ch_dir.glob("chapter_*.md"))
        total_issues = 0
        modified_chapters = 0
        issues_by_category = {}
        chapter_reports = []

        print(f"\n" + "=" * 80)
        print(f"📖 [PRIME Sovereign Editor] Sweeping Novel: '{slug}' ({len(ch_files)} chapters)")
        print("=" * 80)

        for ch_file in ch_files:
            ch_num = int(ch_file.stem.split("_")[-1])
            res = self.edit_chapter(slug, ch_num, apply_fix=apply_fix)
            chapter_reports.append(res)
            
            if res["changed"]:
                modified_chapters += 1
                total_issues += res["issues_count"]
                for iss in res["issues"]:
                    cat = iss["category"]
                    issues_by_category[cat] = issues_by_category.get(cat, 0) + 1
                print(f"  [✓] Chapter {ch_num:02d}: Fixed {res['issues_count']} issue(s)")
            else:
                print(f"  [-] Chapter {ch_num:02d}: Clean (0 issues)")

        # Recompile full novel PDF
        print(f"\n[*] Recompiling typeset Trade Paperback PDF for '{slug}'...")
        pdf_path = build_sample_pdf(slug)

        summary = {
            "slug": slug,
            "chapters_scanned": len(ch_files),
            "chapters_modified": modified_chapters,
            "total_issues_fixed": total_issues,
            "breakdown_by_category": issues_by_category,
            "pdf_generated": str(pdf_path)
        }

        print("\n" + "-" * 80)
        print(f"[★] Editorial Sweep Complete!")
        print(f"    Total Issues Fixed: {total_issues}")
        print(f"    Modified Chapters: {modified_chapters} / {len(ch_files)}")
        for cat, cnt in issues_by_category.items():
            print(f"      • {cat}: {cnt}")
        print(f"    Updated PDF: {pdf_path.name}")
        print("-" * 80 + "\n")

        return summary


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "beyond_the_event_horizon"
    editor = PrimeCopyEditor()
    res = editor.edit_novel_batch(slug, apply_fix=True)
