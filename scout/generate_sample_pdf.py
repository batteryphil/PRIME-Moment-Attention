"""
PRIME Novelist Sample PDF Generator
===================================
Produces trade-paperback typeset PDFs with drop-caps, book indentation,
proper italicized internal monologue, and ornamental scene breaks.
"""

import os
import re
import sys
import json
import html
import shutil
import pathlib
import subprocess

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
NOVELS_ROOT = PROJECT_ROOT / "novels"
ARTIFACT_DIR = pathlib.Path("/home/phil/.gemini/antigravity/brain/7c18e5eb-42ed-4718-9b39-2cb50a9df868")
LO_PROFILE = pathlib.Path("/home/phil/.gemini/antigravity/scratch/lo_profile")
LO_PROFILE.mkdir(parents=True, exist_ok=True)

def markdown_to_typeset_html(text: str) -> str:
    """Parses markdown manuscript text into typeset book HTML."""
    # Strip prompt meta headers
    text = re.sub(r'^(Instruction|Target Heat Level|Novel|Objective|Premise|Chapter \d+:?):.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Story text:?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s+.*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Normalize scene breaks
    text = re.sub(r'^(\s*(\*|\-)\s*){3,}$', '___SCENE_BREAK___', text, flags=re.MULTILINE)

    paragraphs = re.split(r'\n\s*\n+', text)
    html_parts = []
    is_first_after_break = True
    is_first_in_chapter = True

    for p in paragraphs:
        trimmed = p.strip()
        if not trimmed:
            continue

        if trimmed == '___SCENE_BREAK___':
            html_parts.append('<div class="scene-break">✦ ✦ ✦</div>')
            is_first_after_break = True
            continue

        # HTML Escape base text first
        escaped = html.escape(trimmed)

        # Bold: **bold** or __bold__
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
        escaped = re.sub(r'__(.+?)__', r'<strong>\1</strong>', escaped)

        # Italics / internal monologue / whispers: *text* or _text_
        escaped = re.sub(r'\*([^\*\n]+?)\*', r'<em>\1</em>', escaped)
        escaped = re.sub(r'_([^_\n]+?)_', r'<em>\1</em>', escaped)

        # Dialogue breaks / soft breaks
        escaped = re.sub(r'\n\s*([“"«—\-])', r'<br>\1', escaped)
        escaped = re.sub(r'\n\s*', ' ', escaped)

        if is_first_after_break:
            if is_first_in_chapter and len(escaped) > 1 and escaped[0].isalpha():
                first_char = escaped[0]
                rest = escaped[1:]
                html_parts.append(f'<p class="no-indent"><span class="dropcap">{first_char}</span>{rest}</p>')
                is_first_in_chapter = False
            else:
                html_parts.append(f'<p class="no-indent">{escaped}</p>')
            is_first_after_break = False
        else:
            html_parts.append(f'<p>{escaped}</p>')

    return '\n'.join(html_parts)


def build_sample_pdf(slug: str = "a_crown_of_gilded_bones", max_chapters: int = None) -> pathlib.Path:
    """Builds a formatted trade paperback PDF containing all written chapters up to max_chapters."""
    book_dir = NOVELS_ROOT / slug
    chapters_dir = book_dir / "chapters"
    if not chapters_dir.exists():
        raise FileNotFoundError(f"Chapters directory not found: {chapters_dir}")

    ch_files = sorted(chapters_dir.glob("chapter_*.md"))
    if not ch_files:
        raise FileNotFoundError(f"No chapter files found in {chapters_dir}")

    if max_chapters is not None:
        ch_files = ch_files[:max_chapters]

    export_dir = book_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    bible_p = book_dir / "BOOK_BIBLE.json"
    bible = json.loads(bible_p.read_text(encoding="utf-8")) if bible_p.exists() else {}
    title = bible.get("title", slug.replace("_", " ").title())
    genre = bible.get("subgenre", "Space Opera" if "horizon" in slug else "Dark Gothic Stalker Romantasy")

    chapters_html = []
    total_words = 0

    for ch_idx, ch_file in enumerate(ch_files, 1):
        raw_text = ch_file.read_text(encoding="utf-8")
        words = len(raw_text.split())
        total_words += words

        first_line = raw_text.strip().split("\n")[0]
        ch_title = first_line.replace("# ", "").strip() if first_line.startswith("#") else f"Chapter {ch_idx}"
        body_html = markdown_to_typeset_html(raw_text)

        chapters_html.append(f"""
        <h2 class="chapter-title">{html.escape(ch_title)}</h2>
        <div class="chapter-words-meta">Manuscript Excerpt • {words:,} Words</div>
        {body_html}
        """)

    all_chapters_content = "\n".join(chapters_html)

    model_name = "Sao10K/L3-8B-Stheno-v3.2" if "horizon" in slug else "NousResearch/Meta-Llama-3-8B"
    genre_feature = "Hard Sci-Fi • Relativistic Physics • Tactical Combat" if "horizon" in slug else "Heat: 5/5 🌶️ • Gothic Romance"
    badge_label = "Complete Manuscript Edition" if len(ch_files) >= 20 else f"Official Advance Excerpt (Chapters 1–{len(ch_files)})"

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)} — Sample Excerpt</title>
    <style>
    @page {{
        size: 5.5in 8.5in;
        margin: 0.85in 0.75in 0.85in 0.75in;
        @bottom-center {{
            content: counter(page);
            font-family: "Liberation Serif", "Georgia", serif;
            font-size: 9pt;
            color: #666;
        }}
    }}
    body {{
        font-family: "Liberation Serif", "Georgia", serif;
        font-size: 11pt;
        line-height: 1.62;
        color: #1a1a1a;
        background: #ffffff;
    }}
    .title-page {{
        text-align: center;
        page-break-after: always;
        padding-top: 1.8in;
    }}
    h1.book-title {{
        font-size: 24pt;
        font-weight: bold;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #2e1065;
        margin-bottom: 0.2in;
        line-height: 1.2;
    }}
    .book-sub {{
        font-size: 11.5pt;
        font-style: italic;
        color: #581c87;
        margin-bottom: 0.5in;
        letter-spacing: 0.05em;
    }}
    .badge {{
        display: inline-block;
        padding: 6px 16px;
        border: 1px solid #c084fc;
        background: #faf5ff;
        font-size: 8.5pt;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #7e22ce;
        font-weight: bold;
        border-radius: 4px;
        margin-bottom: 0.8in;
    }}
    .meta-box {{
        font-size: 9pt;
        color: #6b7280;
        line-height: 1.6;
        border-top: 1px solid #e5e7eb;
        padding-top: 0.4in;
        max-width: 3.5in;
        margin: 0 auto;
    }}
    h2.chapter-title {{
        text-align: center;
        font-size: 15.5pt;
        font-weight: bold;
        letter-spacing: 0.08em;
        color: #3b0764;
        margin-top: 0.8in;
        margin-bottom: 0.3in;
        page-break-before: always;
    }}
    .chapter-words-meta {{
        text-align: center;
        font-size: 8.5pt;
        color: #8b5cf6;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.45in;
    }}
    p {{
        margin: 0 0 0.45em 0;
        text-indent: 20pt;
        text-align: justify;
    }}
    p.no-indent {{
        text-indent: 0 !important;
    }}
    .dropcap {{
        text-indent: 0;
    }}
    span.drop-cap {{
        float: left;
        font-size: 3.2em;
        line-height: 0.8;
        padding-top: 4px;
        padding-right: 6px;
        padding-bottom: 0;
        font-weight: 700;
        font-family: 'Crimson Text', 'Georgia', serif;
        color: #111827;
    }}
    .scene-break {{
        text-align: center;
        margin-top: 1.1em;
        margin-bottom: 1.1em;
        color: #6b7280;
        font-size: 11pt;
        letter-spacing: 0.3em;
    }}
    em {{
        font-style: italic;
    }}
    strong {{
        font-weight: bold;
        color: #111827;
    }}
    </style>
</head>
<body>
    <div class="title-page">
        <h1 class="book-title">{html.escape(title)}</h1>
        <div class="book-sub">A {html.escape(genre)}</div>
        <div class="badge">{badge_label}</div>
        <div class="meta-box">
            Drafted 100% Locally via PRIME Attention Engine<br>
            {model_name} on AMD GPU<br>
            Chapters Included: 1–{len(ch_files)} • Total Words: {total_words:,} • {genre_feature}
        </div>
    </div>

    {all_chapters_content}
</body>
</html>
"""

    temp_html = export_dir / "temp_chapter_sample.html"
    temp_html.write_text(full_html, encoding="utf-8")

    out_pdf_name = f"{slug}_Chapter_{len(ch_files)}_Sample.pdf" if len(ch_files) < 20 else f"{slug}_Full_Novel.pdf"
    pdf_dest = export_dir / out_pdf_name

    cmd = [
        "libreoffice",
        f"-env:UserInstallation=file://{LO_PROFILE}",
        "--headless",
        "--convert-to", "pdf:writer_pdf_Export",
        str(temp_html),
        "--outdir", str(export_dir)
    ]

    print(f"[*] Running LibreOffice headless PDF compilation for Chapters 1–{len(ch_files)} ({total_words:,} words)...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    generated_tmp = export_dir / "temp_chapter_sample.pdf"

    if generated_tmp.exists():
        if pdf_dest.exists():
            pdf_dest.unlink()
        generated_tmp.rename(pdf_dest)
        print(f"[✓] Generated PDF: {pdf_dest} ({pdf_dest.stat().st_size:,} bytes)")
    else:
        raise RuntimeError(f"LibreOffice failed: {res.stdout} | {res.stderr}")

    if temp_html.exists():
        temp_html.unlink()

    # Copy to artifacts directory
    artifact_target = ARTIFACT_DIR / out_pdf_name
    shutil.copy2(pdf_dest, artifact_target)
    print(f"[✓] Copied to artifact directory: {artifact_target}")

    return pdf_dest

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "beyond_the_event_horizon"
    build_sample_pdf(slug)
