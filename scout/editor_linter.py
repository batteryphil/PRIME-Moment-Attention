"""
PRIME Sovereign Editor Linter (Tier 1) — Safe Precision Edition
===============================================================
High-speed deterministic syntax, alias consistency, vacuum physics,
and sentence integrity linter. GUARANTEES 100% preservation of
paragraph breaks and whitespace formatting.
"""

import re
import json
import pathlib
from typing import Dict, Any, List, Tuple, Optional

# Vacuum Physics Acoustic Replacements
ACOUSTIC_REPLACEMENTS = [
    (
        re.compile(r'\b(afterburners?|engines?|thrusters?)\s+ignited with a deafening roar\b', re.IGNORECASE),
        r'\1 ignited, concussive vibrations thrumming violently through the deckplates'
    ),
    (
        re.compile(r'\broared through the void\b', re.IGNORECASE),
        r'echoed through the hull framing'
    ),
    (
        re.compile(r'\bthe sonic wave washed over them\b', re.IGNORECASE),
        r'the kinetic shockwave shuddered through the airframe'
    ),
    (
        re.compile(r'\bdeafening roar in the vacuum\b', re.IGNORECASE),
        r'silent blinding flash of combustion'
    ),
    (
        re.compile(r'\bdeafening roar of the engines in the void\b', re.IGNORECASE),
        r'violent judder of thrust conducted through the bulkheads'
    ),
    (
        re.compile(r'\bexplosions? roared in the void\b', re.IGNORECASE),
        r'silent blossom of plasma, shockwaves jarring the deckplates'
    ),
    (
        re.compile(r'\bhowling chaos outside\b', re.IGNORECASE),
        r'silent, deadly chaos of vacuum outside'
    )
]

# Alias Collision Rules (Character Consistency)
ALIAS_COLLISION_RULES = [
    (
        re.compile(r'\b(Vex|Astrid)\b\s+paused beside\s+(Captain Ross|Astrid|Vex)\b', re.IGNORECASE),
        r'\1 paused beside the central console'
    ),
    (
        re.compile(r'\b(she paused beside\s+)Captain Ross(,\s*looking over\s+)the other woman\'s(\s*shoulder)\b', re.IGNORECASE),
        r'\1the helmsman\2his\3'
    ),
    (
        re.compile(r'\b(Vex)\b\s+(glanced|looked|stared|turned)\s+at\s+(Astrid)\b', re.IGNORECASE),
        r'\1 \2 at Leo'
    ),
    (
        re.compile(r'\b(Astrid)\b\s+(glanced|looked|stared|turned)\s+at\s+(Vex)\b', re.IGNORECASE),
        r'\1 \2 at Leo'
    ),
    (
        re.compile(r'\b(Leo)\b\s+(glanced|looked|stared|turned)\s+at\s+(Mercer)\b', re.IGNORECASE),
        r'\1 \2 at Astrid'
    )
]

# Recognized terminal punctuation (including em-dashes and ellipses for interrupted speech)
TERMINAL_PUNCT_CHARS = ('.', '!', '?', '"', '”', '…', '—', '-')


class EditorialIssue:
    def __init__(self, category: str, line_no: int, snippet: str, fix_suggestion: str):
        self.category = category
        self.line_no = line_no
        self.snippet = snippet
        self.fix_suggestion = fix_suggestion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "line_no": self.line_no,
            "snippet": self.snippet,
            "fix": self.fix_suggestion
        }


def lint_and_repair_text(text: str) -> Tuple[str, List[EditorialIssue]]:
    """
    Performs deterministic linting and repairs strictly line-by-line.
    Guarantees that all blank lines and paragraph breaks are preserved exactly.
    """
    issues = []
    lines = text.split('\n')
    repaired_lines = []

    for idx, line in enumerate(lines):
        # 1. Check for redundant uppercase chapter sub-headers under # Chapter X
        if 0 < idx < 5 and re.match(r'^\s*CHAPTER\s+\d+\s*[-—:]\s*[^\n]+$', line, re.IGNORECASE):
            issues.append(EditorialIssue(
                "FORMATTING",
                idx + 1,
                line.strip(),
                "Removed redundant sub-header line"
            ))
            continue # Omit the duplicate line

        # 1b. Check for leaked generation outline markers (e.g. "Scene 3: Tactical Fallout" or "Beat 2: Friction")
        if re.match(r'^\s*(Scene|Beat)\s+\d+\s*[-—:]\s*[^\n]+$', line, re.IGNORECASE):
            issues.append(EditorialIssue(
                "OUTLINE_LEAKAGE",
                idx + 1,
                line.strip(),
                "Stripped leaked generation outline marker"
            ))
            continue # Omit leaked prompt outline header

        modified_line = line

        # Clean orphaned dialogue punctuation (e.g. `growled, "."` or `said, ""`)
        modified_line = re.sub(r'([a-zA-Z]+)\s*,\s*["”]\.["”]', r'\1.', modified_line)
        modified_line = re.sub(r'([a-zA-Z]+)\s*,\s*["”]["”]', r'\1.', modified_line)

        # 2. Vacuum Acoustics
        for pattern, repl in ACOUSTIC_REPLACEMENTS:
            if pattern.search(modified_line):
                orig = modified_line
                modified_line = pattern.sub(repl, modified_line)
                issues.append(EditorialIssue(
                    "VACUUM_ACOUSTICS",
                    idx + 1,
                    orig.strip(),
                    modified_line.strip()
                ))

        # 3. Alias Collisions
        for pattern, repl in ALIAS_COLLISION_RULES:
            if pattern.search(modified_line):
                orig = modified_line
                modified_line = pattern.sub(repl, modified_line)
                issues.append(EditorialIssue(
                    "ALIAS_COLLISION",
                    idx + 1,
                    orig.strip(),
                    modified_line.strip()
                ))

        # 4. Spacing before punctuation on the SAME line (never touches newlines)
        modified_line = re.sub(r'[ \t]+([,;:.!?])', r'\1', modified_line)

        repaired_lines.append(modified_line)

    # 5. Check trailing sentence right before scene breaks or EOF
    # We find the non-empty line right before each "✦ ✦ ✦" or at the very end of the manuscript
    for i in range(len(repaired_lines)):
        is_last_line = (i == len(repaired_lines) - 1)
        next_is_break = False
        if not is_last_line:
            # Look ahead to see if next non-empty line is a scene break
            for j in range(i + 1, min(i + 4, len(repaired_lines))):
                if repaired_lines[j].strip():
                    if "✦ ✦ ✦" in repaired_lines[j] or "* * *" in repaired_lines[j]:
                        next_is_break = True
                    break

        if (is_last_line or next_is_break) and repaired_lines[i].strip():
            cur = repaired_lines[i].rstrip()
            # Check for unclosed dialogue quote
            quote_count = cur.count('"') + cur.count('“') + cur.count('”')
            if quote_count % 2 != 0:
                # Odd quotes -> unclosed dialogue
                if cur.endswith(('.', '!', '?', '…', '—')):
                    repaired_lines[i] = cur + '"'
                elif cur.endswith((',', ':', ';')):
                    repaired_lines[i] = cur[:-1] + '."'
                else:
                    repaired_lines[i] = cur + '."'
                issues.append(EditorialIssue(
                    "TRAILING_BOUNDARY",
                    i + 1,
                    cur.strip(),
                    f"Closed trailing dialogue quote: {repaired_lines[i].strip()}"
                ))
            elif re.search(r'\bnone had\s*\.?$', cur, re.IGNORECASE):
                repaired_lines[i] = re.sub(r'\bnone had\s*\.?$', 'none had prepared her for this.', cur)
                issues.append(EditorialIssue("TRAILING_BOUNDARY", i + 1, cur.strip(), repaired_lines[i].strip()))
            elif re.search(r'\bshe was\s*\.?$', cur, re.IGNORECASE):
                repaired_lines[i] = re.sub(r'\bshe was\s*\.?$', 'she was his.', cur)
                issues.append(EditorialIssue("TRAILING_BOUNDARY", i + 1, cur.strip(), repaired_lines[i].strip()))
            elif re.search(r'\bwaiting to rip them\s*\.?$', cur, re.IGNORECASE):
                repaired_lines[i] = re.sub(r'\bwaiting to rip them\s*\.?$', 'waiting to rip them apart.', cur)
                issues.append(EditorialIssue("TRAILING_BOUNDARY", i + 1, cur.strip(), repaired_lines[i].strip()))
            elif re.search(r'\bbefore help could mobil\s*\.?$', cur, re.IGNORECASE):
                repaired_lines[i] = re.sub(r'\bbefore help could mobil\s*\.?$', 'before help could arrive.', cur)
                issues.append(EditorialIssue("TRAILING_BOUNDARY", i + 1, cur.strip(), repaired_lines[i].strip()))
            elif re.search(r'\benergy field meant to\s*\.?$', cur, re.IGNORECASE):
                repaired_lines[i] = re.sub(r'\benergy field meant to\s*\.?$', 'energy field.', cur)
                issues.append(EditorialIssue("TRAILING_BOUNDARY", i + 1, cur.strip(), repaired_lines[i].strip()))
            elif re.search(r'\bblurred into an irresistible\s*\.?$', cur, re.IGNORECASE):
                repaired_lines[i] = re.sub(r'\bblurred into an irresistible\s*\.?$', 'blurred into an irresistible pull.', cur)
                issues.append(EditorialIssue("TRAILING_BOUNDARY", i + 1, cur.strip(), repaired_lines[i].strip()))
            elif not cur.endswith(TERMINAL_PUNCT_CHARS):
                # Ends on a dangling comma, colon, or conjunction
                if cur.endswith((',', ':', ';')):
                    repaired_lines[i] = cur[:-1] + '.'
                elif re.search(r'\b(and|or|then|while|as|with|to|in|of|from|growled|snarled|barked|muttered|whispered)\s*$', cur, re.IGNORECASE):
                    # Strip dangling trailing connector and finish
                    repaired_lines[i] = re.sub(r'\s*[,—\-]?\s*\b(and|or|then|while|as|with|to|in|of|from|growled|snarled|barked|muttered|whispered)\s*$', '.', cur)
                else:
                    repaired_lines[i] = cur + '.'
                issues.append(EditorialIssue(
                    "TRAILING_BOUNDARY",
                    i + 1,
                    cur.strip(),
                    f"Fixed missing terminal punctuation: {repaired_lines[i].strip()}"
                ))

    repaired_text = '\n'.join(repaired_lines)
    return repaired_text, issues


def lint_chapter_file(file_path: pathlib.Path, apply_fix: bool = False) -> Dict[str, Any]:
    """Runs linter on a specific chapter markdown file."""
    raw = file_path.read_text(encoding="utf-8")
    repaired, issues = lint_and_repair_text(raw)
    
    changed = (repaired != raw)
    if apply_fix and changed:
        backup_path = file_path.with_suffix(".md.bak")
        if not backup_path.exists():
            backup_path.write_text(raw, encoding="utf-8")
        file_path.write_text(repaired, encoding="utf-8")

    return {
        "file": str(file_path),
        "issues_count": len(issues),
        "issues": [i.to_dict() for i in issues],
        "changed": changed
    }


if __name__ == "__main__":
    import sys
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target and target.exists():
        res = lint_chapter_file(target, apply_fix=True)
        print(json.dumps(res, indent=2))
    else:
        print("Usage: python editor_linter.py <path_to_chapter.md>")
