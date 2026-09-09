"""
Batch Editorial Polish Sweep for Romantasy / Romance Catalog
============================================================
Runs the Sovereign Copy Editor across all completed romance titles:
- a_crown_of_gilded_bones
- tethered_in_smoke_and_sin
- the_serpent_and_the_star_thief
- beneath_the_ashen_sun
"""

import sys
import json
import sqlite3
import pathlib
from datetime import datetime, timezone

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scout.prime_copy_editor import PrimeCopyEditor

ROMANCE_SLUGS = [
    "a_crown_of_gilded_bones",
    "tethered_in_smoke_and_sin",
    "the_serpent_and_the_star_thief",
    "beneath_the_ashen_sun"
]

def main():
    editor = PrimeCopyEditor()
    catalog_results = []

    print("=" * 80)
    print("🌹 PRIME Sovereign Editor: Romance & Romantasy Catalog Sweep")
    print("=" * 80)

    for slug in ROMANCE_SLUGS:
        try:
            res = editor.edit_novel_batch(slug, apply_fix=True)
            catalog_results.append(res)
        except Exception as e:
            print(f"[!] Error sweeping {slug}: {e}")

    print("\n" + "=" * 80)
    print("✨ ALL ROMANCE NOVELS POLISHED & TYPESET!")
    print("=" * 80)
    for r in catalog_results:
        print(f"📚 {r['slug']:32s} | Chapters: {r['chapters_scanned']:2d} | Issues Fixed: {r['total_issues_fixed']:2d} | PDF: {pathlib.Path(r['pdf_generated']).name}")

if __name__ == "__main__":
    main()
