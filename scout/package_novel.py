import html
import json
import os
import re
import pathlib
import subprocess
import uuid
import zipfile

BASE_DIR = pathlib.Path('/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention')
NOVEL_DIR = BASE_DIR / 'novels' / 'tethered_in_smoke_and_sin'
CHAPTERS_DIR = NOVEL_DIR / 'chapters'
EXPORT_DIR = NOVEL_DIR / 'export'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
LO_PROFILE = pathlib.Path('/home/phil/.gemini/antigravity/scratch/lo_profile')
LO_PROFILE.mkdir(parents=True, exist_ok=True)

BIBLE_PATH = NOVEL_DIR / 'BOOK_BIBLE.json'
bible = json.loads(BIBLE_PATH.read_text(encoding='utf-8')) if BIBLE_PATH.exists() else {}

def format_manuscript_html(body: str, is_epub: bool = False) -> str:
    """Formats markdown body into typographic book HTML without raw asterisks."""
    text = re.sub(r'^(\s*(\*|\-)\s*){3,}$', '___SCENE_BREAK___', body, flags=re.MULTILINE)
    paras = re.split(r'\n\s*\n+', text)
    parts = []
    is_first = True
    for p in paras:
        p_strip = p.strip()
        if not p_strip:
            continue
        if p_strip == '___SCENE_BREAK___':
            parts.append('<div class="scene-break">✦ ✦ ✦</div>')
            is_first = True
            continue
        esc = html.escape(p_strip)
        esc = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', esc)
        esc = re.sub(r'__(.+?)__', r'<strong>\1</strong>', esc)
        esc = re.sub(r'\*([^\*\n]+?)\*', r'<em>\1</em>', esc)
        esc = re.sub(r'_([^_\n]+?)_', r'<em>\1</em>', esc)
        esc = re.sub(r'\n\s*([“"«—\-])', r'<br/>\1' if is_epub else r'<br>\1', esc)
        esc = re.sub(r'\n\s*', ' ', esc)
        cls = ' class="no-indent"' if is_first else ''
        parts.append(f'<p{cls}>{esc}</p>')
        is_first = False
    return '\n'.join(parts)


TITLE = bible.get('title', 'Tethered in Smoke and Sin')
GENRE = bible.get('genre', 'Spicy Portal Fantasy Romance')
AUTHOR = "PRIME Autonomous Novelist"
DESCRIPTION = "A full-length 50,000-word adult spicy fantasy romance novel. When modern Chicago antiquities archivist Elena Vance touches a cracked obsidian relic, she is ripped across dimensions into the twilight realm of Aethelgard and bound by an unbreakable Blood-Tether to Lord Vaelen Thorne, a ruthless shadow warlord. Forced within fifty paces of each other, their shared heartbeats ignite an intoxicating, dangerous passion that will either destroy the realm or burn down an empire."

chapter_files = sorted(CHAPTERS_DIR.glob('chapter_*.md'))
chapters_data = []

for cf in chapter_files:
    raw_text = cf.read_text(encoding='utf-8')
    lines = raw_text.strip().split('\n')
    ch_title = lines[0].replace('# ', '').strip()
    body_text = '\n'.join(lines[1:]).strip()
    words = len(raw_text.split())
    num = int(cf.stem.split('_')[1])
    chapters_data.append({
        'num': num,
        'file_stem': cf.stem,
        'title': ch_title,
        'body': body_text,
        'words': words
    })

total_words = sum(c['words'] for c in chapters_data)
print(f"Loaded {len(chapters_data)} chapters. Total words: {total_words:,}")

# -------------------------------------------------------------
# 1. BUILD CLEAN FORMATTED PLAIN TEXT (.txt)
# -------------------------------------------------------------
txt_path = EXPORT_DIR / f"{TITLE.replace(' ', '_')}.txt"
txt_lines = [
    "=" * 75,
    f"  {TITLE.upper()}",
    f"  A {GENRE}",
    f"  Words: {total_words:,} | Complete 24 Chapters",
    "=" * 75,
    "",
    DESCRIPTION,
    "",
    "=" * 75,
    ""
]

for ch in chapters_data:
    txt_lines.append("")
    txt_lines.append("*" * 60)
    txt_lines.append(f"  {ch['title'].upper()} ({ch['words']:,} words)")
    txt_lines.append("*" * 60)
    txt_lines.append("")
    txt_lines.append(ch['body'])
    txt_lines.append("")

txt_path.write_text('\n'.join(txt_lines), encoding='utf-8')
print(f"[✓] Created TXT: {txt_path} ({txt_path.stat().st_size:,} bytes)")

# -------------------------------------------------------------
# 2. BUILD STANDALONE SELF-CONTAINED OFFLINE HTML E-READER (.html)
# -------------------------------------------------------------
html_reader_path = EXPORT_DIR / f"{TITLE.replace(' ', '_')}_Offline_Reader.html"

chapters_json_str = json.dumps([
    {
        'num': ch['num'],
        'title': ch['title'],
        'words': ch['words'],
        'html': format_manuscript_html(ch['body'])
    }
    for ch in chapters_data
])

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{TITLE} — Complete Novel ({total_words:,} words)</title>
    <style>
        :root {{
            --bg: #0f141c;
            --surface: #18202c;
            --border: #2d3748;
            --accent: #f472b6;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
        }}
        body.sepia {{
            --bg: #2d2620;
            --surface: #3a322b;
            --border: #52473d;
            --accent: #f6ad55;
            --text: #f7fafc;
            --text-muted: #cbd5e0;
        }}
        body.light {{
            --bg: #fdfbf7;
            --surface: #f3eee3;
            --border: #e2d9cc;
            --accent: #d53f8c;
            --text: #2d3748;
            --text-muted: #718096;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: "Georgia", "Cambria", "Times New Roman", serif;
            line-height: 1.85;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            transition: background 0.2s ease, color 0.2s ease;
        }}
        header {{
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 0.85rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .title-group {{ display: flex; align-items: center; gap: 0.75rem; }}
        .book-title {{ font-size: 1.15rem; font-weight: bold; color: var(--accent); }}
        .badge {{
            font-size: 0.75rem;
            background: rgba(244, 114, 182, 0.15);
            color: var(--accent);
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-family: sans-serif;
            font-weight: 600;
        }}
        .controls {{ display: flex; align-items: center; gap: 0.5rem; }}
        select, button {{
            background: var(--bg);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
        }}
        .container {{
            max-width: 780px;
            margin: 2rem auto;
            padding: 0 1.5rem 6rem;
            flex: 1;
        }}
        .chapter-header {{
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        .chapter-title {{
            font-size: 2.2rem;
            color: var(--accent);
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }}
        .chapter-meta {{
            font-size: 0.9rem;
            color: var(--text-muted);
            font-family: sans-serif;
        }}
        p {{
            margin-bottom: 1.4rem;
            text-indent: 1.75rem;
            font-size: 1.12rem;
        }}
        p:first-of-type {{ text-indent: 0; }}
        .break {{
            text-align: center;
            margin: 2.5rem 0;
            color: var(--accent);
            font-size: 1.4rem;
            letter-spacing: 0.5rem;
        }}
        .nav-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 3.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }}
        .btn-nav {{
            background: linear-gradient(135deg, #ec4899, #8b5cf6);
            color: white;
            border: none;
            padding: 0.65rem 1.4rem;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            font-family: sans-serif;
        }}
        .btn-nav:disabled {{ opacity: 0.3; cursor: not-allowed; }}
    </style>
</head>
<body>
    <header>
        <div class="title-group">
            <span class="book-title">📖 {TITLE}</span>
            <span class="badge">{total_words:,} words • 24 Chapters</span>
        </div>
        <div class="controls">
            <select id="chapterSelect" onchange="goToChapter(parseInt(this.value))"></select>
            <button onclick="cycleTheme()">🎨 Theme</button>
            <button onclick="toggleFont()">Aa Font</button>
        </div>
    </header>

    <div class="container">
        <div class="chapter-header">
            <h1 class="chapter-title" id="dispTitle">Loading...</h1>
            <div class="chapter-meta" id="dispMeta"></div>
        </div>
        <div id="contentArea"></div>
        <div class="nav-footer">
            <button class="btn-nav" id="btnPrev" onclick="prevChapter()">← Previous Chapter</button>
            <span id="footerPage" style="font-size: 0.85rem; color: var(--text-muted); font-family: sans-serif;"></span>
            <button class="btn-nav" id="btnNext" onclick="nextChapter()">Next Chapter →</button>
        </div>
    </div>

    <script>
        const chapters = {chapters_json_str};
        let currentIdx = 0;
        let isSerif = true;
        const themes = ['', 'sepia', 'light'];
        let themeIdx = 0;

        function init() {{
            const sel = document.getElementById('chapterSelect');
            chapters.forEach((ch, i) => {{
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = `Ch ${{ch.num}}: ${{ch.title.split(': ')[1] || ch.title}} (${{ch.words.toLocaleString()}}w)`;
                sel.appendChild(opt);
            }});
            renderChapter(0);
        }}

        function renderChapter(idx) {{
            currentIdx = idx;
            const ch = chapters[idx];
            document.getElementById('chapterSelect').value = idx;
            document.getElementById('dispTitle').textContent = ch.title;
            document.getElementById('dispMeta').textContent = `Chapter ${{ch.num}} of ${{chapters.length}} • ${{ch.words.toLocaleString()}} words`;
            document.getElementById('footerPage').textContent = `Chapter ${{ch.num}} of ${{chapters.length}}`;
            document.getElementById('contentArea').innerHTML = ch.html;
            
            document.getElementById('btnPrev').disabled = (idx === 0);
            document.getElementById('btnNext').disabled = (idx === chapters.length - 1);
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        function goToChapter(idx) {{ renderChapter(idx); }}
        function prevChapter() {{ if (currentIdx > 0) renderChapter(currentIdx - 1); }}
        function nextChapter() {{ if (currentIdx < chapters.length - 1) renderChapter(currentIdx + 1); }}

        function cycleTheme() {{
            themeIdx = (themeIdx + 1) % themes.length;
            document.body.className = themes[themeIdx];
        }}

        function toggleFont() {{
            isSerif = !isSerif;
            document.body.style.fontFamily = isSerif ? '"Georgia", serif' : '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        }}

        window.onload = init;
    </script>
</body>
</html>
"""
html_reader_path.write_text(html_content, encoding='utf-8')
print(f"[✓] Created HTML Reader: {html_reader_path} ({html_reader_path.stat().st_size:,} bytes)")

# -------------------------------------------------------------
# 3. BUILD OFFICIAL IDPF EPUB (.epub)
# -------------------------------------------------------------
epub_path = EXPORT_DIR / f"{TITLE.replace(' ', '_')}.epub"
book_uuid = str(uuid.uuid4())

with zipfile.ZipFile(epub_path, 'w', zipfile.ZIP_DEFLATED) as ep:
    # 1. mimetype (MUST be first, uncompressed)
    ep.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)

    # 2. META-INF/container.xml
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>"""
    ep.writestr('META-INF/container.xml', container_xml)

    # 3. OEBPS/style.css
    css_content = """
body {
    font-family: "Georgia", "Times New Roman", serif;
    line-height: 1.6;
    margin: 5%;
    padding: 0;
    color: #1a1a1a;
}
h1, h2 {
    color: #7c2d12;
    text-align: center;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
}
h1 { font-size: 1.8em; }
h2 { font-size: 1.4em; }
p {
    margin-bottom: 0.8em;
    text-indent: 1.5em;
}
p.first {
    text-indent: 0;
}
.title-page {
    text-align: center;
    margin-top: 20%;
}
.book-title {
    font-size: 2.2em;
    color: #581c87;
    margin-bottom: 0.2em;
}
.subtitle {
    font-size: 1.2em;
    font-style: italic;
    color: #4b5563;
    margin-bottom: 2em;
}
.author {
    font-size: 1.1em;
    margin-bottom: 3em;
}
.meta-badge {
    font-size: 0.9em;
    color: #6b7280;
}
.scene-break {
    text-align: center;
    margin: 1.5em 0;
    color: #7e22ce;
    letter-spacing: 0.5em;
}
"""
    ep.writestr('OEBPS/style.css', css_content)

    # 4. OEBPS/title.xhtml
    title_xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>{html.escape(TITLE)}</title>
    <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
    <div class="title-page">
        <h1 class="book-title">{html.escape(TITLE)}</h1>
        <div class="subtitle">A {html.escape(GENRE)}</div>
        <div class="author">By {html.escape(AUTHOR)}</div>
        <div class="meta-badge">{total_words:,} Words • Complete 24-Chapter Novel</div>
        <div style="margin-top: 3em; font-size: 0.95em; line-height: 1.6; text-align: justify; padding: 0 10%;">
            {html.escape(DESCRIPTION)}
        </div>
    </div>
</body>
</html>"""
    ep.writestr('OEBPS/title.xhtml', title_xhtml)

    # 5. Chapter XHTML files
    for ch in chapters_data:
        body_xhtml = format_manuscript_html(ch['body'], is_epub=True)

        ch_xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>{html.escape(ch['title'])}</title>
    <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
    <h1>{html.escape(ch['title'])}</h1>
    <div style="text-align: center; font-size: 0.85em; color: #6b7280; margin-bottom: 1.5em;">{ch['words']:,} words</div>
    {body_xhtml}
</body>
</html>"""
        ep.writestr(f"OEBPS/{ch['file_stem']}.xhtml", ch_xhtml)

    # 6. OEBPS/toc.ncx (for older EPUB readers / Kindle)
    ncx_points = []
    ncx_points.append(f"""
        <navPoint id="navpoint-title" playOrder="1">
            <navLabel><text>Title Page</text></navLabel>
            <content src="title.xhtml"/>
        </navPoint>""")
    for i, ch in enumerate(chapters_data, start=2):
        ncx_points.append(f"""
        <navPoint id="navpoint-{ch['file_stem']}" playOrder="{i}">
            <navLabel><text>{html.escape(ch['title'])}</text></navLabel>
            <content src="{ch['file_stem']}.xhtml"/>
        </navPoint>""")

    ncx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="urn:uuid:{book_uuid}"/>
        <meta name="dtb:depth" content="1"/>
        <meta name="dtb:totalPageCount" content="0"/>
        <meta name="dtb:maxPageNumber" content="0"/>
    </head>
    <docTitle><text>{html.escape(TITLE)}</text></docTitle>
    <navMap>
        {''.join(ncx_points)}
    </navMap>
</ncx>"""
    ep.writestr('OEBPS/toc.ncx', ncx_content)

    # 7. OEBPS/nav.xhtml (EPUB 3 navigation document)
    nav_items = [f'<li><a href="{ch["file_stem"]}.xhtml">{html.escape(ch["title"])}</a></li>' for ch in chapters_data]
    nav_xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>Table of Contents</title>
    <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
    <nav epub:type="toc" id="toc">
        <h1>Table of Contents</h1>
        <ol>
            <li><a href="title.xhtml">Title Page</a></li>
            {''.join(nav_items)}
        </ol>
    </nav>
</body>
</html>"""
    ep.writestr('OEBPS/nav.xhtml', nav_xhtml)

    # 8. OEBPS/content.opf
    manifest_items = [
        '<item id="style" href="style.css" media-type="text/css"/>',
        '<item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>',
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
    ]
    for ch in chapters_data:
        manifest_items.append(f'<item id="{ch["file_stem"]}" href="{ch["file_stem"]}.xhtml" media-type="application/xhtml+xml"/>')

    spine_items = ['<itemref idref="title"/>']
    for ch in chapters_data:
        spine_items.append(f'<itemref idref="{ch["file_stem"]}"/>')

    opf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookId" version="3.0">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="BookId">urn:uuid:{book_uuid}</dc:identifier>
        <dc:title>{html.escape(TITLE)}</dc:title>
        <dc:language>en</dc:language>
        <dc:creator>{html.escape(AUTHOR)}</dc:creator>
        <dc:description>{html.escape(DESCRIPTION)}</dc:description>
        <meta property="dcterms:modified">2026-09-07T23:00:00Z</meta>
    </metadata>
    <manifest>
        {''.join(manifest_items)}
    </manifest>
    <spine toc="ncx">
        {''.join(spine_items)}
    </spine>
</package>"""
    ep.writestr('OEBPS/content.opf', opf_content)

print(f"[✓] Created EPUB: {epub_path} ({epub_path.stat().st_size:,} bytes)")

# -------------------------------------------------------------
# 4. BUILD PDF VIA LIBREOFFICE HEADLESS
# -------------------------------------------------------------
# We create a clean HTML file for LibreOffice to convert
book_doc_html = EXPORT_DIR / "temp_book_for_pdf.html"
doc_html_body = [
    f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{TITLE}</title><style>
    @page {{ size: 6in 9in; margin: 0.75in; }}
    body {{ font-family: "Liberation Serif", "Times New Roman", serif; font-size: 11pt; line-height: 1.5; color: #111; }}
    h1 {{ font-size: 20pt; text-align: center; margin-top: 3in; color: #3b0764; }}
    h2 {{ font-size: 15pt; text-align: center; margin-top: 1in; margin-bottom: 0.5in; color: #4c1d95; page-break-before: always; }}
    p {{ margin-bottom: 0.45em; text-indent: 20pt; text-align: justify; }}
    p.no-indent {{ text-indent: 0 !important; }}
    .scene-break {{ text-align: center; margin: 2em 0; letter-spacing: 0.5em; color: #9333ea; }}
    em {{ font-style: italic; color: #4a044e; }}
    strong {{ font-weight: bold; color: #111827; }}
    .title-sub {{ text-align: center; font-size: 12pt; font-style: italic; margin-bottom: 2in; color: #6b7280; }}
    </style></head><body>
    <h1>{TITLE}</h1>
    <div class="title-sub">A {GENRE}<br><br>{total_words:,} Words • Complete Novel</div>
    """
]
for ch in chapters_data:
    doc_html_body.append(f"<h2>{ch['title']}</h2>")
    doc_html_body.append(format_manuscript_html(ch['body']))
doc_html_body.append("</body></html>")
book_doc_html.write_text('\n'.join(doc_html_body), encoding='utf-8')

pdf_path = EXPORT_DIR / f"{TITLE.replace(' ', '_')}.pdf"
print("[*] Generating PDF via LibreOffice headless...")
try:
    cmd = [
        "libreoffice",
        f"-env:UserInstallation=file://{LO_PROFILE}",
        "--headless",
        "--convert-to", "pdf:writer_pdf_Export",
        str(book_doc_html), "--outdir", str(EXPORT_DIR)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    generated_pdf = EXPORT_DIR / "temp_book_for_pdf.pdf"
    if generated_pdf.exists():
        if pdf_path.exists():
            pdf_path.unlink()
        generated_pdf.rename(pdf_path)
        print(f"[✓] Created PDF: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    else:
        print(f"[!] LibreOffice PDF conversion output: {res.stdout} | {res.stderr}")
except Exception as e:
    print(f"[!] PDF generation error: {e}")
finally:
    if book_doc_html.exists():
        book_doc_html.unlink()

# -------------------------------------------------------------
# 5. BUILD EMAILABLE ALL-IN-ONE ZIP ARCHIVE
# -------------------------------------------------------------
zip_bundle_path = EXPORT_DIR / f"{TITLE.replace(' ', '_')}_Complete_Package.zip"

with zipfile.ZipFile(zip_bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add EPUB
    if epub_path.exists():
        zf.write(epub_path, arcname=f"{TITLE.replace(' ', '_')}.epub")
    # Add Offline HTML Reader
    if html_reader_path.exists():
        zf.write(html_reader_path, arcname=f"{TITLE.replace(' ', '_')}_Offline_Reader.html")
    # Add PDF
    if pdf_path.exists():
        zf.write(pdf_path, arcname=f"{TITLE.replace(' ', '_')}.pdf")
    # Add TXT
    if txt_path.exists():
        zf.write(txt_path, arcname=f"{TITLE.replace(' ', '_')}.txt")
    # Add Raw Markdown Manuscript
    manuscript_md = NOVEL_DIR / 'MANUSCRIPT.md'
    if manuscript_md.exists():
        zf.write(manuscript_md, arcname=f"{TITLE.replace(' ', '_')}_Manuscript.md")

print(f"[✓] Created Complete Emailable ZIP Package: {zip_bundle_path} ({zip_bundle_path.stat().st_size:,} bytes)")
print("=" * 70)
print("ALL PACKAGED FILES READY IN:")
print(f"  {EXPORT_DIR}")
print("=" * 70)
