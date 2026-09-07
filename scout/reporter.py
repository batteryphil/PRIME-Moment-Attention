"""
PRIME-Scout: Daily Briefing Reporter
Generates high-signal Markdown daily briefings with score badges,
telemetry tables, sandbox results, and actionable research synergies.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from scout.config import REPORTS_DIR
from scout.database import record_briefing

class BriefingReporter:
    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_briefing(self, evaluations: List[Dict[str, Any]], date_str: Optional[str] = None) -> Path:
        date_str = date_str or datetime.utcnow().strftime("%Y-%m-%d")
        file_path = self.reports_dir / f"briefing_{date_str}.md"
        latest_path = self.reports_dir / "briefing_latest.md"

        # Sort evaluations by Alignment Score descending, then Viability Score
        sorted_evals = sorted(evaluations, key=lambda x: (x.get("alignment_score", 0) * 1.2 + x.get("viability_score", 0)), reverse=True)
        
        must_read = [e for e in sorted_evals if e.get("verdict") == "MUST_READ" or e.get("alignment_score", 0) >= 80]
        worth_exploring = [e for e in sorted_evals if e not in must_read and (e.get("verdict") == "WORTH_EXPLORING" or e.get("alignment_score", 0) >= 60)]
        other_repos = [e for e in sorted_evals if e not in must_read and e not in worth_exploring]

        lines = [
            f"# 🔭 PRIME-Scout Daily Intelligence Briefing: {date_str}",
            "",
            "> **Autonomous Local Research Briefing for batteryphil / PRIME-Moment-Attention**",
            f"> Generated at {datetime.utcnow().strftime('%H:%M:%S UTC')} using Stage 7 Hybrid local evaluation & sandbox testing.",
            "",
            "## 📊 Daily Summary Telemetry",
            f"- **Repositories Scanned Today**: {len(sorted_evals)}",
            f"- **High-Synergy Discoveries**: {len(must_read)} Priority Must-Reads | {len(worth_exploring)} Worth Exploring",
            f"- **Sandbox Status**: {sum(1 for e in sorted_evals if e.get('sandbox_status') == 'PASS')} Verified Passing",
            "",
            "## 🏆 Executive Leaderboard",
            "",
            "| Repository | Stars | Viability | Alignment | Verdict | Sandbox |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ]

        for e in sorted_evals:
            v_score = e.get("viability_score", 0)
            a_score = e.get("alignment_score", 0)
            verdict = e.get("verdict", "MONITOR")
            s_status = e.get("sandbox_status", "SKIPPED")
            
            s_badge = f"`{s_status}`"
            v_badge = f"**{v_score}/100**"
            a_badge = f"**{a_score}/100**"
            
            repo_link = f"[{e.get('full_name', e.get('name', 'Repo'))}]({e.get('repo_url', '#')})"
            lines.append(f"| {repo_link} | {e.get('stars', 0):,} | {v_badge} | {a_badge} | `{verdict}` | {s_badge} |")

        lines.append("")
        lines.append("---")
        lines.append("")

        if must_read:
            lines.append("## 🌟 Priority Must-Read Repositories")
            lines.append("")
            for e in must_read:
                lines.extend(self._format_deep_dive(e))

        if worth_exploring:
            lines.append("## 💡 Notable & Worth Exploring")
            lines.append("")
            for e in worth_exploring:
                lines.extend(self._format_deep_dive(e))

        if other_repos:
            lines.append("## 📋 Monitored & Background Repositories")
            lines.append("")
            for e in other_repos:
                lines.extend(self._format_deep_dive(e, compact=True))

        lines.append("---")
        lines.append("*Report generated autonomously by PRIME-Scout. To launch the interactive dashboard, run `python -m scout.scout_cli ui` or visit `http://localhost:7860`.*")

        content = "\n".join(lines)
        file_path.write_text(content)
        
        # Also update briefing_latest.md
        shutil.copyfile(file_path, latest_path)

        # Record in database
        summary_text = f"Scanned {len(sorted_evals)} repos. Top matches: {len(must_read)} must-read, {len(worth_exploring)} worth exploring."
        record_briefing(date_str, str(file_path), summary_text, len(sorted_evals))

        print(f"[+] Daily briefing written to: {file_path}")
        print(f"[+] Latest briefing updated at: {latest_path}")
        return file_path

    def _format_deep_dive(self, e: Dict[str, Any], compact: bool = False) -> List[str]:
        full_name = e.get("full_name", e.get("name", "Unknown"))
        repo_url = e.get("repo_url", "#")
        stars = e.get("stars", 0)
        v_score = e.get("viability_score", 0)
        a_score = e.get("alignment_score", 0)
        verdict = e.get("verdict", "MONITOR")
        s_status = e.get("sandbox_status", "SKIPPED")
        
        blocks = [
            f"### [{full_name}]({repo_url}) ({stars:,} ★)",
            f"> **Verdict**: `{verdict}` | **Viability**: `{v_score}/100` | **Alignment with PRIME**: `{a_score}/100` | **Sandbox**: `{s_status}`",
            "",
            f"**Executive Pitch**:\n{e.get('executive_pitch', 'No pitch available.')}",
            "",
            f"**Synergy & Integration Ideas with Your Work**:\n{e.get('synergy_notes', 'None noted.')}",
            "",
        ]

        if not compact:
            blocks.extend([
                f"**Technical Critique & Code Health**:\n{e.get('technical_critique', 'No critique available.')}",
                "",
                f"<details><summary>🔍 Sandbox Execution Log ({s_status})</summary>",
                "",
                "```text",
                e.get("sandbox_log", "No sandbox log available.").strip(),
                "```",
                "</details>",
                "",
            ])

        blocks.append("")
        return blocks

if __name__ == "__main__":
    reporter = BriefingReporter()
    sample_eval = [{
        "full_name": "sustcsonglin/flash-linear-attention",
        "name": "flash-linear-attention",
        "repo_url": "https://github.com/sustcsonglin/flash-linear-attention",
        "stars": 2450,
        "viability_score": 92,
        "alignment_score": 95,
        "verdict": "MUST_READ",
        "sandbox_status": "PASS",
        "executive_pitch": "Hardware-accelerated linear attention implementations in Triton across RetNet, GLA, and Mamba.",
        "technical_critique": "Extremely clean Triton kernels with optimized SRAM tiling and causal mask handling.",
        "synergy_notes": "We can port their chunked recurrent Triton kernel template directly to PRIME 2nd-order moment attention on ROCm.",
        "sandbox_log": "All 12 AST checks passed. Triton detected. Import clean."
    }]
    path = reporter.generate_daily_briefing(sample_eval)
    print("Report sample generated at:", path)
