"""
PRIME-Scout: Modern Local Web Dashboard with Two-Way Chat & Experimentation
FastAPI + Standalone Dark-Mode UI:
- 100% Self-Contained, ZERO external CDN dependencies
- Two-way conversation interface (Phil <-> PRIME-Scout)
- Agent Inquiries / Questions Queue with one-click reply
- Micro-benchmark experiment executor
- STRICT COMMIT BAN enforced.
"""

import os
import re
import html
import shutil
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from scout.config import UI_CONFIG, REPORTS_DIR, VAULT_DIR, SANDBOX_DIR, DB_PATH, MODEL_CONFIG, SAFETY_POLICY
from scout.database import (
    get_connection, get_recent_evaluations, get_evaluation_by_url,
    get_briefings, upsert_repository, record_evaluation,
    save_chat_message, get_chat_history,
    save_agent_question, get_pending_questions, answer_agent_question,
    get_repo_experiments, get_recent_theories, get_theory_by_id,
    get_autonomous_state, update_autonomous_state
)
from scout.sandbox import SandboxRunner
from scout.evaluator import PrimeScoutEvaluator
from scout.experimenter import RepoExperimenter
from scout.github_client import GitHubClient
from scout.reporter import BriefingReporter
from scout.scientist import AutonomousScientist

app = FastAPI(title=UI_CONFIG["title"])

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sandbox_runner = SandboxRunner()
evaluator = PrimeScoutEvaluator(mode="auto")
experimenter = RepoExperimenter()
github_client = GitHubClient()
reporter = BriefingReporter()

class RepoTestRequest(BaseModel):
    repo_url: str
    run_llm: bool = True

class ChatRequest(BaseModel):
    message: str
    context_repo_url: Optional[str] = None

class AnswerQuestionRequest(BaseModel):
    question_id: int
    answer: str

class ExperimentRequest(BaseModel):
    repo_url: str

def render_markdown_to_html(md_text: str) -> str:
    if not md_text:
        return "<p style='color: #9ca3af;'>No content available.</p>"

    code_blocks = []
    def save_code_block(match):
        idx = len(code_blocks)
        code = html.escape(match.group(2))
        lang = match.group(1) or ""
        code_blocks.append(f'<pre><code class="lang-{lang}">{code}</code></pre>')
        return f"__CODE_BLOCK_{idx}__"
    
    text = re.sub(r'```([a-zA-Z0-9_-]*)\n(.*?)```', save_code_block, md_text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', lambda m: f'<code>{html.escape(m.group(1))}</code>', text)
    text = text.replace("<details>", "___DETAILS_START___").replace("</details>", "___DETAILS_END___")
    text = re.sub(r'<summary>(.*?)</summary>', r'___SUMMARY_START___\1___SUMMARY_END___', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" style="color: #38bdf8; text-decoration: underline;">\1</a>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)

    lines = text.split("\n")
    out_lines = []
    in_table = False
    table_rows = []

    for line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            parts = [p.strip() for p in s[1:-1].split("|")]
            if all(set(p).issubset({'-', ':', ' '}) for p in parts if p):
                continue
            table_rows.append(parts)
            in_table = True
            continue
        elif in_table:
            out_lines.append("<div style='overflow-x: auto;'><table style='width: 100%; border-collapse: collapse; margin: 1rem 0;'>")
            for i, row in enumerate(table_rows):
                out_lines.append("<tr>")
                for cell in row:
                    tag = "th" if i == 0 else "td"
                    style = "border: 1px solid #24324a; padding: 0.6rem 0.8rem; text-align: left;"
                    if i == 0:
                        style += " background: rgba(36, 50, 74, 0.4); font-weight: 600;"
                    out_lines.append(f"<{tag} style='{style}'>{cell}</{tag}>")
                out_lines.append("</tr>")
            out_lines.append("</table></div>")
            table_rows = []
            in_table = False

        if not s:
            out_lines.append("<br>")
            continue

        if s.startswith("# "):
            out_lines.append(f"<h1 style='font-size: 1.75rem; font-weight: 700; margin: 1.5rem 0 0.75rem; color: #f3f4f6;'>{s[2:]}</h1>")
        elif s.startswith("## "):
            out_lines.append(f"<h2 style='font-size: 1.35rem; font-weight: 600; margin: 1.5rem 0 0.75rem; color: #f3f4f6;'>{s[3:]}</h2>")
        elif s.startswith("### "):
            out_lines.append(f"<h3 style='font-size: 1.15rem; font-weight: 600; margin: 1.25rem 0 0.5rem; color: #e5e7eb;'>{s[4:]}</h3>")
        elif s.startswith("> "):
            out_lines.append(f"<blockquote style='border-left: 3px solid #38bdf8; padding-left: 1rem; color: #9ca3af; margin: 0.75rem 0;'>{s[2:]}</blockquote>")
        elif s.startswith("- "):
            out_lines.append(f"<li style='margin-left: 1.5rem; margin-bottom: 0.25rem;'>{s[2:]}</li>")
        elif s == "---":
            out_lines.append("<hr style='border: none; border-top: 1px solid #24324a; margin: 1.5rem 0;'>")
        else:
            out_lines.append(f"<p style='margin-bottom: 0.5rem;'>{s}</p>")

    if in_table and table_rows:
        out_lines.append("<div style='overflow-x: auto;'><table style='width: 100%; border-collapse: collapse; margin: 1rem 0;'>")
        for i, row in enumerate(table_rows):
            out_lines.append("<tr>")
            for cell in row:
                tag = "th" if i == 0 else "td"
                style = "border: 1px solid #24324a; padding: 0.6rem 0.8rem; text-align: left;"
                if i == 0:
                    style += " background: rgba(36, 50, 74, 0.4); font-weight: 600;"
                out_lines.append(f"<{tag} style='{style}'>{cell}</{tag}>")
            out_lines.append("</tr>")
        out_lines.append("</table></div>")

    rendered = "\n".join(out_lines)
    rendered = rendered.replace("___DETAILS_START___", "<details style='margin: 0.75rem 0; background: rgba(0,0,0,0.3); border: 1px solid #24324a; border-radius: 6px; padding: 0.75rem;'>")
    rendered = rendered.replace("___DETAILS_END___", "</details>")
    rendered = re.sub(r'___SUMMARY_START___(.*?)___SUMMARY_END___', r'<summary style="cursor: pointer; font-weight: 600; color: #38bdf8;">\1</summary>', rendered)

    for idx, cb in enumerate(code_blocks):
        rendered = rendered.replace(f"__CODE_BLOCK_{idx}__", cb)

    return rendered

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRIME-Scout | Local AI Repository Intelligence</title>
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #111827;
            --surface-hover: #1f2937;
            --border: #24324a;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --text: #f3f4f6;
            --text-muted: #9ca3af;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --card-bg: rgba(17, 24, 39, 0.85);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            line-height: 1.5;
        }
        header {
            background: rgba(17, 24, 39, 0.95);
            border-bottom: 1px solid var(--border);
            padding: 0.85rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .brand { display: flex; align-items: center; gap: 0.75rem; }
        .brand-icon { font-size: 1.6rem; }
        .brand-title { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
        .brand-subtitle { font-size: 0.75rem; color: var(--text-muted); font-family: monospace; }
        .header-pills { display: flex; align-items: center; gap: 0.75rem; }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            background: rgba(36, 50, 74, 0.5);
            border: 1px solid var(--border);
            border-radius: 9999px;
            font-size: 0.78rem;
            font-family: monospace;
        }
        .dot-pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--success); box-shadow: 0 0 8px var(--success); }
        .btn {
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .btn:hover { opacity: 0.95; box-shadow: 0 0 15px var(--accent-glow); transform: translateY(-1px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-secondary { background: var(--surface); border: 1px solid var(--border); color: var(--text); }
        .btn-secondary:hover { background: var(--surface-hover); }

        .tabs-nav {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            padding: 0 2rem;
            gap: 1.5rem;
        }
        .tab-btn {
            padding: 0.9rem 0.2rem;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .tab-btn:hover { color: var(--text); }
        .tab-btn.active { color: var(--accent); border-bottom: 2px solid var(--accent); }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        .card-title { font-size: 1.15rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; }

        #toast {
            display: none;
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid var(--accent);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            font-size: 0.9rem;
            z-index: 100;
            max-width: 400px;
        }

        .badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: monospace;
        }
        .badge-must-read { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-worth-exploring { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
        .badge-monitor { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .score-pill {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.2rem 0.5rem;
            font-size: 0.75rem;
            font-family: monospace;
        }

        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

        .input-group { display: flex; gap: 0.75rem; margin-bottom: 1.25rem; }
        .text-input {
            flex: 1;
            background: #060911;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text);
            font-size: 0.95rem;
            font-family: monospace;
        }
        .text-input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 8px var(--accent-glow); }

        .terminal {
            background: #050811;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.82rem;
            color: #a7f3d0;
            max-height: 280px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        .repo-item { background: rgba(24, 33, 50, 0.6); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem; }
        .repo-name { font-size: 1.1rem; font-weight: 600; color: var(--accent); text-decoration: none; }
        
        /* Chat UI */
        .chat-box {
            display: flex;
            flex-direction: column;
            height: 480px;
            background: #050811;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .chat-messages {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .msg {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .msg-user {
            align-self: flex-end;
            background: #1d4ed8;
            color: #eff6ff;
            border-bottom-right-radius: 2px;
        }
        .msg-scout {
            align-self: flex-start;
            background: #1e293b;
            color: #f1f5f9;
            border: 1px solid var(--border);
            border-bottom-left-radius: 2px;
        }
        .chat-input-row {
            display: flex;
            padding: 0.75rem;
            background: var(--surface);
            border-top: 1px solid var(--border);
            gap: 0.5rem;
        }

        /* Questions Box */
        .question-item {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #38bdf8;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        pre { background: #060911; border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem; overflow-x: auto; font-family: monospace; font-size: 0.82rem; }
        code { font-family: monospace; background: rgba(36, 50, 74, 0.6); padding: 0.15rem 0.35rem; border-radius: 4px; font-size: 0.85em; }
    </style>
</head>
<body>
    <div id="toast"></div>

    <header>
        <div class="brand">
            <div class="brand-icon">🔭</div>
            <div>
                <div class="brand-title">PRIME-Scout</div>
                <div class="brand-subtitle">Autonomous Local Intelligence Agent</div>
            </div>
        </div>
        <div class="header-pills">
            <div class="pill" style="color: #34d399;"><span class="dot-pulse"></span> AMD ROCm 7.2 Active</div>
            <div class="pill">🧠 Qwen2.5-Coder-1.5B</div>
            <div class="pill" style="color: #f87171;">🔒 Commit Ban: Active</div>
            <button class="btn" id="btn-scan" onclick="triggerScan()">⚡ Run Discovery Now</button>
        </div>
    </header>

    <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab(this, 'tab-briefing')">📰 Daily Briefing</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-theories')">🔬 Autonomous Lab & Theories (<span id="theory-count">0</span>)</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-chat')">💬 Talk to PRIME-Scout</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-sandbox')">🧪 Sandbox & Experiments</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-vault')">🏛️ Memory Vault (<span id="vault-count">0</span>)</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-fs')">📁 Easy File System</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-novel')">📖 Novel Reader (50k Words)</button>
    </div>

    <main>
        
        <!-- Tab 6: Novel Reader -->
        <div id="tab-novel" class="tab-content">
            <div class="card" style="border-color: #ec4899;">
                <div class="card-header">
                    <div class="card-title" style="color: #f472b6;">
                        📖 Tethered in Smoke and Sin <span class="badge badge-completed" style="margin-left: 0.75rem;">50,025 Words • 24 Chapters Complete</span>
                    </div>
                    <div style="display: flex; gap: 0.5rem;">
                        <a href="/reader" target="_blank" class="btn" style="background: linear-gradient(135deg, #ec4899, #8b5cf6); text-decoration: none;">✨ Open Fullscreen E-Reader</a>
                        <a href="/api/novel/download" class="btn btn-secondary" style="text-decoration: none;">📥 Download Manuscript (.md)</a>
                    </div>
                </div>
                <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; background: rgba(236, 72, 153, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(236, 72, 153, 0.2);">
                    <div>
                        <div style="font-size: 0.85rem; color: #f472b6; font-weight: 600;">GENRE: Spicy Romantasy (Portal Fantasy, Enemies-to-Lovers, Forced Proximity)</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Protagonist: Elena Vance (Antiquities Archivist) • Male Lead: Lord Vaelen Thorne (Shadow Warlord)</div>
                    </div>
                    <div style="margin-left: auto; text-align: right;">
                        <span style="font-size: 1.25rem; font-weight: 700; color: #38bdf8;">50,025</span> / 50,000 words <span style="color: #10b981;">(100%)</span>
                    </div>
                </div>

                <div style="display: flex; gap: 1.5rem; min-height: 500px;">
                    <!-- Chapter List Sidebar -->
                    <div style="width: 280px; flex-shrink: 0; background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; max-height: 650px; overflow-y: auto;">
                        <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">Table of Contents</div>
                        <div id="novel-chapter-list"><p style="color: var(--text-muted); font-size: 0.85rem;">Loading chapters...</p></div>
                    </div>
                    <!-- Chapter Content Viewer -->
                    <div style="flex: 1; background: rgba(17, 24, 39, 0.6); border: 1px solid var(--border); border-radius: 8px; padding: 2rem; max-height: 650px; overflow-y: auto;">
                        <div id="novel-reader-content" style="line-height: 1.8; font-size: 1.05rem; font-family: Georgia, serif; color: #e2e8f0;">
                            <h2 style="color: #f472b6; margin-bottom: 1rem;">Select a Chapter to Begin Reading</h2>
                            <p style="color: var(--text-muted);">All 24 chapters and 50,025 words are generated and stored in SQLite. Click any chapter on the left, or open the Fullscreen E-Reader.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 1: Daily Briefing -->
        <div id="tab-briefing" class="tab-content active">
            <!-- Questions Queue for Phil -->
            <div id="questions-container" style="display: none; margin-bottom: 1.5rem;">
                <div class="card" style="border-color: #38bdf8;">
                    <div class="card-header">
                        <div class="card-title" style="color: #38bdf8;">❓ Strategic Inquiries from PRIME-Scout for You</div>
                    </div>
                    <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">
                        PRIME-Scout identified decisions or trade-offs during candidate repository analysis that require your technical direction:
                    </p>
                    <div id="questions-list"></div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">📅 Today's Intelligence Digest</div>
                    <button class="btn btn-secondary" onclick="loadLatestBriefing()">🔄 Refresh Briefing</button>
                </div>
                <div id="briefing-container">
                    <p style="color: var(--text-muted);">Loading latest briefing from <code>scout/reports/</code>...</p>
                </div>
            </div>
        </div>

        <!-- Tab 2: Talk to PRIME-Scout (Chat) -->
        <div id="tab-chat" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">💬 Direct Dialogue with PRIME-Scout</div>
                    <span style="font-size: 0.8rem; color: var(--text-muted); font-family: monospace;">Local LLM on ROCm • Zero-Commit Sandbox Partner</span>
                </div>
                <p style="color: var(--text-muted); font-size: 0.88rem; margin-bottom: 1rem;">
                    Ask questions about any discovered repository, ask the agent to explain architectural details, or direct it to run specific experiments.
                </p>

                <div class="chat-box">
                    <div class="chat-messages" id="chat-messages">
                        <div class="msg msg-scout">
                            Hello Phil! I am PRIME-Scout, running locally on your workstation GPU. I'm actively studying candidate repositories for linear attention, Triton/ROCm kernels, and sub-quadratic sequence models. How can I help you today?
                        </div>
                    </div>
                    <div class="chat-input-row">
                        <input type="text" id="chat-input" class="text-input" placeholder="Ask PRIME-Scout a question or give research direction..." onkeydown="if(event.key==='Enter') sendChatMessage()">
                        <button class="btn" id="btn-chat-send" onclick="sendChatMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 3: Sandbox & Experiments -->
        <div id="tab-sandbox" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🧪 Isolated Sandbox & Benchmark Chambers</div>
                    <span class="badge" style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444;">🔒 Git Commits Blocked</span>
                </div>
                <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
                    Paste any GitHub URL to clone, AST-audit, smoke-test, and run a micro-benchmark experiment. All candidate code executes in strictly isolated chambers.
                </p>
                <div class="input-group">
                    <input type="text" id="repo-url-input" class="text-input" placeholder="https://github.com/owner/repository" value="https://github.com/sustcsonglin/flash-linear-attention">
                    <button class="btn" id="btn-test" onclick="runSandboxTest()">🚀 Execute Sandbox & Benchmark</button>
                </div>

                <div class="grid-2">
                    <div>
                        <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-muted);">Execution Terminal:</h4>
                        <div id="sandbox-term" class="terminal">[Ready] Awaiting repository URL...</div>
                    </div>
                    <div>
                        <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-muted);">Telemetry & Evaluation Result:</h4>
                        <div id="eval-result-card" style="background: #050811; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; min-height: 280px;">
                            <p style="color: var(--text-muted); font-size: 0.85rem;">Evaluation metrics and benchmark numbers will appear here.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 4: Memory Vault -->
        <div id="tab-vault" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🏛️ Evaluated Repositories Vault</div>
                    <button class="btn btn-secondary" onclick="loadVault()">🔄 Refresh Vault</button>
                </div>
                <div id="vault-list">
                    <p style="color: var(--text-muted);">Loading persistent memory vault...</p>
                </div>
            </div>
        </div>

        <!-- Tab 5: File System Explorer -->
        <div id="tab-fs" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📁 PRIME-Scout File System Layout</div>
                    <button class="btn btn-secondary" onclick="cleanSandboxActive()">🧹 Clean Active Sandboxes</button>
                </div>
                <div style="font-family: monospace; font-size: 0.88rem; background: #060911; padding: 1.25rem; border-radius: 8px; border: 1px solid var(--border); line-height: 1.8;">
                    <div>📂 <b>scout/</b></div>
                    <div style="padding-left: 1.5rem;">📁 <b>reports/</b> &nbsp; <span style="color: var(--text-muted);">(Daily markdown digests for reading)</span></div>
                    <div style="padding-left: 3rem;">📄 <a href="/api/briefing/latest" target="_blank" style="color: var(--accent);">briefing_latest.md</a></div>
                    <div style="padding-left: 1.5rem;">📁 <b>vault/</b> &nbsp; <span style="color: var(--text-muted);">(SQLite persistent memory & telemetry)</span></div>
                    <div style="padding-left: 3rem;">🗄️ scout_vault.db</div>
                    <div style="padding-left: 1.5rem;">📁 <b>sandbox/active/</b> &nbsp; <span style="color: var(--text-muted);">(Isolated git checkouts & smoke test chambers)</span></div>
                    <div id="sandbox-active-list" style="padding-left: 3rem; color: var(--text-muted);">Scanning active checkouts...</div>
                </div>
            </div>
        </div>

        <!-- Tab 6: Autonomous Lab & Theories -->
        <div id="tab-theories" class="tab-content">
            <div class="card" style="border-color: #818cf8; margin-bottom: 1.5rem;">
                <div class="card-header">
                    <div>
                        <div class="card-title" style="color: #818cf8;">🔬 Autonomous Hypothesis & Empirical Lab</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.2rem;">
                            PRIME-Scout continuously mines architectures, formulates novel mathematical conjectures, writes PyTorch tests, executes benchmarks on GPU, and records empirical laws.
                        </div>
                    </div>
                    <div style="display: flex; gap: 0.5rem;">
                        <button class="btn btn-secondary" id="btn-toggle-engine" onclick="toggleAutonomousEngine()">⏸️ Pause Daemon</button>
                        <button class="btn" id="btn-trigger-theory" onclick="triggerTheoryCycle()">⚡ Formulate Theory Now</button>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1rem; padding: 1rem; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid var(--border);">
                    <div>
                        <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Engine State</div>
                        <div id="stat-auto-state" style="font-weight: 700; color: #34d399; font-size: 1.1rem; margin-top: 0.25rem;">ACTIVE</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Current Action</div>
                        <div id="stat-auto-action" style="font-weight: 600; color: #e5e7eb; font-size: 0.85rem; margin-top: 0.25rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">IDLE</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Cycles Completed</div>
                        <div id="stat-auto-cycles" style="font-weight: 700; color: #38bdf8; font-size: 1.1rem; margin-top: 0.25rem;">0</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Theories Formulated</div>
                        <div id="stat-auto-theories" style="font-weight: 700; color: #a78bfa; font-size: 1.1rem; margin-top: 0.25rem;">0</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Tests Executed</div>
                        <div id="stat-auto-tests" style="font-weight: 700; color: #fbbf24; font-size: 1.1rem; margin-top: 0.25rem;">0</div>
                    </div>
                </div>
            </div>

            <!-- List of Theories -->
            <div id="theories-container">
                <p style="color: var(--text-muted);">Loading formulated theories & test telemetry...</p>
            </div>
        </div>
    </main>

    <script>
        function showToast(msg, duration = 4000) {
            const t = document.getElementById('toast');
            t.innerHTML = msg;
            t.style.display = 'block';
            setTimeout(() => { t.style.display = 'none'; }, duration);
        }

        function switchTab(btn, tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            if (btn) btn.classList.add('active');
            if (tabId === 'tab-theories') { loadTheories(); loadAutonomousStatus(); }
            if (tabId === 'tab-vault') loadVault();
            if (tabId === 'tab-fs') loadFs();
            if (tabId === 'tab-chat') loadChatHistory();
        }

        async function loadLatestBriefing() {
            try {
                const res = await fetch('/api/briefing/latest');
                const data = await res.json();
                if (data.html) {
                    document.getElementById('briefing-container').innerHTML = data.html;
                } else {
                    document.getElementById('briefing-container').innerHTML = '<p style="color: var(--text-muted);">No briefing found. Click <b>⚡ Run Discovery Now</b> to run the agent.</p>';
                }
                loadQuestions();
            } catch (err) {
                console.error(err);
                document.getElementById('briefing-container').innerHTML = '<p style="color: var(--danger);">Failed to load briefing.</p>';
            }
        }

        async function loadQuestions() {
            try {
                const res = await fetch('/api/questions');
                const data = await res.json();
                const container = document.getElementById('questions-container');
                const list = document.getElementById('questions-list');
                if (data.questions && data.questions.length > 0) {
                    container.style.display = 'block';
                    list.innerHTML = data.questions.map(q => `
                        <div class="question-item">
                            <div style="font-weight: 600; color: #38bdf8; margin-bottom: 0.25rem;">${q.repo_name || 'Candidate Repository'}</div>
                            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">${q.question}</div>
                            <div style="display: flex; gap: 0.5rem;">
                                <input type="text" id="ans-${q.id}" class="text-input" placeholder="Type your instruction or preference..." style="padding: 0.4rem 0.75rem; font-size: 0.85rem;">
                                <button class="btn" style="padding: 0.4rem 0.75rem; font-size: 0.8rem;" onclick="submitAnswer(${q.id})">Reply</button>
                            </div>
                        </div>
                    `).join('');
                } else {
                    container.style.display = 'none';
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function submitAnswer(qid) {
            const input = document.getElementById(`ans-${qid}`);
            const text = input.value.trim();
            if (!text) return;
            try {
                await fetch('/api/questions/answer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question_id: qid, answer: text })
                });
                showToast('✅ Direction recorded in agent memory vault!');
                loadQuestions();
            } catch (err) {
                showToast('❌ Error: ' + err.message);
            }
        }

        async function loadChatHistory() {
            try {
                const res = await fetch('/api/chat/history');
                const data = await res.json();
                const container = document.getElementById('chat-messages');
                if (data.history && data.history.length > 0) {
                    container.innerHTML = data.history.map(m => `
                        <div class="msg ${m.sender === 'user' ? 'msg-user' : 'msg-scout'}">
                            ${m.content}
                        </div>
                    `).join('');
                    container.scrollTop = container.scrollHeight;
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById('chat-input');
            const msg = input.value.trim();
            if (!msg) return;

            const container = document.getElementById('chat-messages');
            container.innerHTML += `<div class="msg msg-user">${msg}</div>`;
            container.scrollTop = container.scrollHeight;
            input.value = '';

            const btn = document.getElementById('btn-chat-send');
            btn.disabled = true;
            btn.innerText = 'Thinking...';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                container.innerHTML += `<div class="msg msg-scout">${data.response}</div>`;
                container.scrollTop = container.scrollHeight;
            } catch (err) {
                container.innerHTML += `<div class="msg msg-scout" style="color: var(--danger);">Error communicating with agent: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Send';
            }
        }

        async function loadVault() {
            try {
                const res = await fetch('/api/repos');
                const data = await res.json();
                document.getElementById('vault-count').innerText = data.repos.length;
                const container = document.getElementById('vault-list');
                if (!data.repos || data.repos.length === 0) {
                    container.innerHTML = '<p style="color: var(--text-muted);">Memory vault is empty. Run discovery to populate.</p>';
                    return;
                }
                let html = '';
                for (const r of data.repos) {
                    const vClass = r.verdict === 'MUST_READ' ? 'badge-must-read' : (r.verdict === 'WORTH_EXPLORING' ? 'badge-worth-exploring' : 'badge-monitor');
                    html += `
                    <div class="repo-item">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                            <div>
                                <a href="${r.repo_url}" target="_blank" class="repo-name">${r.full_name || r.name}</a>
                                <span style="color: var(--text-muted); font-size: 0.8rem; margin-left: 0.5rem;">${(r.stars || 0).toLocaleString()} ★</span>
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <span class="badge ${vClass}">${r.verdict}</span>
                                <span class="score-pill">Viability: <b>${r.viability_score}/100</b></span>
                                <span class="score-pill">PRIME Alignment: <b>${r.alignment_score}/100</b></span>
                                <span class="score-pill">Sandbox: <b>${r.sandbox_status}</b></span>
                            </div>
                        </div>
                        <p style="font-size: 0.88rem; color: #d1d5db; margin-bottom: 0.5rem;"><b>Pitch:</b> ${r.executive_pitch || r.description}</p>
                        <p style="font-size: 0.85rem; color: #93c5fd; margin-bottom: 0.5rem;"><b>Synergy:</b> ${r.synergy_notes || 'N/A'}</p>
                        <p style="font-size: 0.8rem; color: var(--text-muted);"><b>Critique:</b> ${r.technical_critique || 'N/A'}</p>
                    </div>
                    `;
                }
                container.innerHTML = html;
            } catch (err) {
                console.error(err);
            }
        }

        async function runSandboxTest() {
            const url = document.getElementById('repo-url-input').value.trim();
            if (!url) {
                showToast('⚠️ Please enter a GitHub repository URL.');
                return;
            }

            const btn = document.getElementById('btn-test');
            const term = document.getElementById('sandbox-term');
            const resultCard = document.getElementById('eval-result-card');

            btn.disabled = true;
            btn.innerText = '⏳ Testing & Benchmarking...';
            term.innerText = `[*] Initiating sandbox chamber for ${url}...\n[*] Git Commit Policy: STRICT BAN (Read-Only Clone)\n[*] Auditing hardware kernels and Python AST...\n`;
            resultCard.innerHTML = '<p style="color: var(--accent);">Processing in isolated sandbox chamber (git clone -> dependency scan -> AST check -> import test -> LLM reasoning)...</p>';

            try {
                const res = await fetch('/api/test-repo', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ repo_url: url, run_llm: true })
                });
                const data = await res.json();
                term.innerText = data.sandbox_log || data.message || 'Complete.';
                
                if (data.evaluation) {
                    const e = data.evaluation;
                    const vClass = e.verdict === 'MUST_READ' ? 'badge-must-read' : (e.verdict === 'WORTH_EXPLORING' ? 'badge-worth-exploring' : 'badge-monitor');
                    resultCard.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <span class="badge ${vClass}" style="font-size: 0.85rem;">${e.verdict}</span>
                            <div>
                                <span class="score-pill" style="color: #38bdf8;">Viability: <b>${e.viability_score}/100</b></span>
                                <span class="score-pill" style="color: #a78bfa;">Alignment: <b>${e.alignment_score}/100</b></span>
                            </div>
                        </div>
                        <div style="font-size: 0.88rem; margin-bottom: 0.5rem;"><b>Executive Pitch:</b> ${e.executive_pitch}</div>
                        <div style="font-size: 0.85rem; color: #93c5fd; margin-bottom: 0.5rem;"><b>PRIME Synergy:</b> ${e.synergy_notes}</div>
                        <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.5rem;"><b>Technical Critique:</b> ${e.technical_critique}</div>
                        ${e.question_for_phil ? `<div style="font-size: 0.85rem; color: #fde047; padding: 0.5rem; background: rgba(253, 224, 71, 0.1); border-radius: 4px;"><b>Agent Question:</b> ${e.question_for_phil}</div>` : ''}
                    `;
                }
                showToast('✅ Sandbox test & evaluation completed!');
                loadVault();
            } catch (err) {
                term.innerText += `\n[!] Error: ${err.message}`;
                resultCard.innerHTML = '<p style="color: var(--danger);">Failed to complete test.</p>';
                showToast('❌ Test failed: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = '🚀 Execute Sandbox & Benchmark';
            }
        }

        async function triggerScan() {
            const btn = document.getElementById('btn-scan');
            btn.disabled = true;
            btn.innerText = '⏳ Scanning...';
            showToast('⚡ Discovery scan initiated in background! Updates will appear in briefing shortly.');

            try {
                await fetch('/api/scan', { method: 'POST' });
                setTimeout(() => {
                    loadLatestBriefing();
                    loadVault();
                    btn.disabled = false;
                    btn.innerText = '⚡ Run Discovery Now';
                }, 8000);
            } catch (err) {
                btn.disabled = false;
                btn.innerText = '⚡ Run Discovery Now';
                showToast('❌ Failed to trigger scan: ' + err.message);
            }
        }

        async function loadFs() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const container = document.getElementById('sandbox-active-list');
                if (data.active_sandboxes && data.active_sandboxes.length > 0) {
                    container.innerHTML = data.active_sandboxes.map(s => `<div>📦 ${s}</div>`).join('');
                } else {
                    container.innerHTML = '<span style="color: var(--text-muted);">(No active sandboxes)</span>';
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function loadAutonomousStatus() {
            try {
                const res = await fetch('/api/autonomous/status');
                const data = await res.json();
                const stateEl = document.getElementById('stat-auto-state');
                const actionEl = document.getElementById('stat-auto-action');
                const cyclesEl = document.getElementById('stat-auto-cycles');
                const theoriesEl = document.getElementById('stat-auto-theories');
                const testsEl = document.getElementById('stat-auto-tests');
                const toggleBtn = document.getElementById('btn-toggle-engine');

                if (stateEl) {
                    stateEl.innerText = data.is_active ? 'ACTIVE' : 'PAUSED';
                    stateEl.style.color = data.is_active ? '#34d399' : '#f87171';
                }
                if (actionEl) actionEl.innerText = `${data.current_action}: ${data.current_target}`;
                if (cyclesEl) cyclesEl.innerText = data.cycles_completed || 0;
                if (theoriesEl) theoriesEl.innerText = data.theories_generated || 0;
                if (testsEl) testsEl.innerText = data.tests_executed || 0;
                if (toggleBtn) {
                    toggleBtn.innerText = data.is_active ? '⏸️ Pause Daemon' : '▶️ Resume Daemon';
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function loadTheories() {
            try {
                const res = await fetch('/api/theories');
                const data = await res.json();
                const list = document.getElementById('theories-container');
                const countBadge = document.getElementById('theory-count');
                if (countBadge) countBadge.innerText = data.theories ? data.theories.length : 0;

                if (!data.theories || data.theories.length === 0) {
                    list.innerHTML = '<p style="color: var(--text-muted);">No theories formulated yet. Click <b>⚡ Formulate Theory Now</b> to launch a scientific inquiry cycle.</p>';
                    return;
                }

                list.innerHTML = data.theories.map(th => {
                    let badgeColor = '#9ca3af';
                    let badgeBg = 'rgba(156, 163, 175, 0.1)';
                    if (th.status === 'VALIDATED') {
                        badgeColor = '#34d399';
                        badgeBg = 'rgba(52, 211, 153, 0.15)';
                    } else if (th.status === 'FALSIFIED') {
                        badgeColor = '#f59e0b';
                        badgeBg = 'rgba(245, 158, 11, 0.15)';
                    } else if (th.status === 'RUNTIME_ERROR' || th.status === 'ERROR') {
                        badgeColor = '#f87171';
                        badgeBg = 'rgba(248, 113, 113, 0.15)';
                    }

                    const telemetryStr = th.telemetry ? JSON.stringify(th.telemetry, null, 2) : 'No telemetry';

                    let domainBadge = '';
                    if (th.domain === 'BATTERY_PHYSICS') {
                        domainBadge = `<span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); font-weight: 700; margin-right: 0.5rem;">🔋 NASA Battery Physics</span>`;
                    } else if (th.domain === 'THEORETICAL_MATHEMATICS') {
                        domainBadge = `<span class="badge" style="background: rgba(147, 51, 234, 0.2); color: #c084fc; border: 1px solid rgba(147, 51, 234, 0.4); font-weight: 700; margin-right: 0.5rem;">📐 Pure Mathematics</span>`;
                    } else {
                        domainBadge = `<span class="badge" style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); font-weight: 700; margin-right: 0.5rem;">⚡ Architecture</span>`;
                    }

                    let branchBadge = '';
                    if (th.branch_type === 'ADAPTIVE_REFINEMENT') {
                        branchBadge = `<span class="badge" style="background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); font-weight: 700; margin-right: 0.5rem;">🧬 Refinement of #${th.parent_theory_id}</span>`;
                    } else if (th.branch_type === 'DEEPENING_ADVANCE') {
                        branchBadge = `<span class="badge" style="background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); font-weight: 700; margin-right: 0.5rem;">🚀 Advance from #${th.parent_theory_id}</span>`;
                    } else if (th.branch_type === 'FRONTIER_SEED') {
                        branchBadge = `<span class="badge" style="background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); font-weight: 700; margin-right: 0.5rem;">🌱 Frontier Seed</span>`;
                    }

                    const isMultiVar = th.variables && th.variables.length > 1;
                    const varDesc = isMultiVar ? `&nbsp;<span style="color: #94a3b8; font-size: 0.78rem;">(2D Surface: ${th.variables.join(' &times; ')})</span>` : '';

                    return `
                        <div class="card" style="margin-bottom: 1.25rem; border-left: 4px solid ${badgeColor};">
                            <div class="card-header" style="align-items: flex-start;">
                                <div>
                                    <div style="font-weight: 700; font-size: 1.05rem; color: #f3f4f6; margin-bottom: 0.25rem;">${th.title}</div>
                                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                                        Target: <span style="color: #38bdf8;">${th.target_repo_name || 'PRIME Ecosystem'}</span>
                                        ${th.target_repo_url ? ` &middot; <a href="${th.target_repo_url}" target="_blank" style="color: #38bdf8; text-decoration: underline;">GitHub</a>` : ''}
                                        &middot; Formulated: ${th.created_at ? th.created_at.substring(0, 19).replace('T', ' ') : ''}
                                    </div>
                                </div>
                                <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 0.3rem;">
                                    ${domainBadge}
                                    ${branchBadge}
                                    <span class="badge" style="background: ${badgeBg}; color: ${badgeColor}; font-weight: 700;">${th.status}</span>
                                </div>
                            </div>

                            <div style="margin: 0.75rem 0; font-size: 0.88rem; line-height: 1.6;">
                                <div style="margin-bottom: 0.5rem;"><b style="color: #a78bfa;">Hypothesis:</b> ${th.hypothesis}</div>
                                <div style="color: var(--text-muted); margin-bottom: 0.5rem;"><b style="color: #9ca3af;">Motivation:</b> ${th.motivation}</div>
                                ${th.empirical_conclusion ? `<div style="background: rgba(0,0,0,0.3); padding: 0.6rem 0.8rem; border-radius: 6px; border-left: 3px solid ${badgeColor}; margin-bottom: 0.5rem;"><b>Empirical Conclusion:</b> ${th.empirical_conclusion}</div>` : ''}
                                ${th.discovered_equation ? `
                                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.4); border-left: 4px solid #10b981; padding: 0.6rem 0.8rem; border-radius: 6px; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                                    <div>
                                        <span style="color: #34d399; font-weight: 700; font-size: 0.85rem; margin-right: 0.5rem;">📐 PRIME-Net Invariant:</span>
                                        <code style="background: #022c22; color: #6ee7b7; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.95rem; font-weight: bold; border: 1px solid #059669;">y = ${th.discovered_equation}</code>
                                        ${varDesc}
                                    </div>
                                    <span style="background: rgba(52, 211, 153, 0.15); color: #34d399; font-size: 0.78rem; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px;">Fit R² = ${th.equation_r2 !== null && th.equation_r2 !== undefined ? th.equation_r2 : '1.000'}</span>
                                </div>` : ''}
                                ${th.synergy_notes ? `<div style="font-size: 0.82rem; color: #38bdf8;"><b>PRIME Synergy:</b> ${th.synergy_notes}</div>` : ''}
                            </div>

                            <details style="margin-top: 0.75rem; background: #060911; border: 1px solid var(--border); border-radius: 6px; padding: 0.6rem;">
                                <summary style="cursor: pointer; font-size: 0.82rem; color: var(--text-muted); font-weight: 600;">View Synthesized PyTorch Test Code & Telemetry</summary>
                                <div style="margin-top: 0.5rem;">
                                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.25rem;">SYNTHESIZED PYTORCH TEST:</div>
                                    <pre style="background: rgba(0,0,0,0.5); padding: 0.75rem; border-radius: 4px; overflow-x: auto; font-size: 0.78rem; color: #93c5fd; max-height: 250px;"><code>${th.synthesized_code || 'No code recorded.'}</code></pre>
                                    <div style="font-size: 0.75rem; color: var(--text-muted); margin: 0.5rem 0 0.25rem;">EMPIRICAL TELEMETRY:</div>
                                    <pre style="background: rgba(0,0,0,0.5); padding: 0.5rem; border-radius: 4px; overflow-x: auto; font-size: 0.75rem; color: #a7f3d0;"><code>${telemetryStr}</code></pre>
                                </div>
                            </details>
                        </div>
                    `;
                }).join('');
            } catch (err) {
                console.error(err);
            }
        }

        async function triggerTheoryCycle() {
            const btn = document.getElementById('btn-trigger-theory');
            btn.disabled = true;
            btn.innerText = '⚡ Synthesizing Theory...';
            showToast('🔬 Scientific cycle launched! Formulating hypothesis & synthesizing test...');

            try {
                await fetch('/api/autonomous/trigger_cycle', { method: 'POST' });
                setTimeout(() => {
                    loadTheories();
                    loadAutonomousStatus();
                    btn.disabled = false;
                    btn.innerText = '⚡ Formulate Theory Now';
                    showToast('✅ New theory & experiment logged in vault!');
                }, 7000);
            } catch (err) {
                btn.disabled = false;
                btn.innerText = '⚡ Formulate Theory Now';
                showToast('❌ Error: ' + err.message);
            }
        }

        async function toggleAutonomousEngine() {
            try {
                const res = await fetch('/api/autonomous/toggle', { method: 'POST' });
                const data = await res.json();
                showToast(data.is_active ? '▶️ Autonomous Engine Resumed' : '⏸️ Autonomous Engine Paused');
                loadAutonomousStatus();
            } catch (err) {
                showToast('❌ Failed to toggle engine: ' + err.message);
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            loadLatestBriefing();
            loadVault();
            loadTheories();
            loadAutonomousStatus();

            // Periodic live pulse
            setInterval(() => {
                loadAutonomousStatus();
            }, 6000);
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/api/status")
def get_status():
    import torch
    active_sandboxes = [p.name for p in SANDBOX_DIR.iterdir() if p.is_dir()] if SANDBOX_DIR.exists() else []
    vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    return {
        "status": "ONLINE",
        "device": MODEL_CONFIG["device"],
        "vram_allocated_mb": round(vram_mb, 1),
        "model_id": MODEL_CONFIG["model_id"],
        "safety_commit_ban": not SAFETY_POLICY["ALLOW_GIT_COMMIT"],
        "db_path": str(DB_PATH),
        "active_sandboxes": active_sandboxes
    }

@app.get("/api/briefing/latest")
def get_latest_briefing():
    latest_file = REPORTS_DIR / "briefing_latest.md"
    if latest_file.exists():
        raw_md = latest_file.read_text(errors="replace")
        html_content = render_markdown_to_html(raw_md)
        return {
            "exists": True,
            "markdown": raw_md,
            "html": html_content,
            "file_path": str(latest_file)
        }
    return {"exists": False, "markdown": "", "html": ""}

@app.get("/api/repos")
def list_repositories(limit: int = 50):
    repos = get_recent_evaluations(limit=limit)
    return {"repos": repos}

@app.post("/api/test-repo")
def test_repo_endpoint(req: RepoTestRequest):
    url = req.repo_url.strip()
    parts = [p for p in url.rstrip("/").split("/") if p]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")
        
    owner = parts[-2]
    name = parts[-1]
    full_name = f"{owner}/{name}"

    repo_meta = {
        "repo_url": url,
        "name": name,
        "full_name": full_name,
        "owner": owner,
        "description": f"Target repository {full_name}",
        "stars": 0,
        "forks": 0,
        "language": "Python",
        "topics": []
    }
    repo_id = upsert_repository(repo_meta)

    sandbox_res = sandbox_runner.evaluate_repo_sandbox(url, owner, name, repo_id=repo_id)
    readme_text = github_client.fetch_readme(full_name)

    eval_res = evaluator.evaluate(repo_meta, readme_text, sandbox_res)
    eval_id = record_evaluation(repo_id, eval_res)
    eval_res["id"] = eval_id

    return {
        "success": True,
        "full_name": full_name,
        "sandbox_status": sandbox_res["status"],
        "sandbox_log": sandbox_res["log"],
        "evaluation": eval_res
    }

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    reply = evaluator.chat(req.message, context_repo_url=req.context_repo_url)
    return {"response": reply}

@app.get("/api/chat/history")
def chat_history_endpoint():
    history = get_chat_history(limit=50)
    return {"history": history}

@app.get("/api/questions")
def get_questions_endpoint():
    questions = get_pending_questions()
    return {"questions": questions}

@app.post("/api/questions/answer")
def answer_question_endpoint(req: AnswerQuestionRequest):
    success = answer_agent_question(req.question_id, req.answer)
    return {"success": success}

# ==============================================================================
# AUTONOMOUS THEORIES & LAB API ROUTES
# ==============================================================================

@app.get("/api/theories")
def get_theories_endpoint(limit: int = 50):
    theories = get_recent_theories(limit=limit)
    return {"theories": theories}

@app.get("/api/autonomous/status")
def get_autonomous_status_endpoint():
    state = get_autonomous_state()
    return state

@app.post("/api/autonomous/toggle")
def toggle_autonomous_engine():
    state = get_autonomous_state()
    new_active = 0 if state.get("is_active", 1) else 1
    update_autonomous_state(is_active=new_active, current_action="PAUSED" if new_active == 0 else "RESUMED")
    return {"is_active": new_active}

@app.post("/api/autonomous/trigger_cycle")
def trigger_theory_cycle_endpoint(background_tasks: BackgroundTasks):
    def _run_theory_bg():
        scientist = AutonomousScientist()
        scientist.run_full_discovery_and_theorize_cycle()
        
    background_tasks.add_task(_run_theory_bg)
    return {"status": "CYCLE_INITIATED", "message": "Autonomous scientific inquiry cycle launched."}

@app.post("/api/scan")
def trigger_scan_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(_execute_full_discovery_pipeline)
    return {"status": "SCAN_INITIATED", "message": "Background discovery pipeline running."}

@app.delete("/api/sandbox/clean")
def clean_sandboxes():
    if SANDBOX_DIR.exists():
        for item in SANDBOX_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
    return {"status": "CLEANED"}

def _execute_full_discovery_pipeline(limit: int = 4):
    print("[*] Starting automated daily discovery pipeline...")
    candidates = github_client.discover_candidates(max_per_query=2)[:limit]
    evaluations = []

    for c in candidates:
        url = c["repo_url"]
        owner = c["owner"]
        name = c["name"]
        full_name = c["full_name"]
        
        repo_id = upsert_repository(c)
        sandbox_res = sandbox_runner.evaluate_repo_sandbox(url, owner, name, repo_id=repo_id)
        readme = github_client.fetch_readme(full_name)
        eval_res = evaluator.evaluate(c, readme, sandbox_res)
        record_evaluation(repo_id, eval_res)
        
        merged = {**c, **eval_res}
        evaluations.append(merged)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reporter.generate_daily_briefing(evaluations, date_str=today_str)
    print(f"[+] Automated discovery pipeline complete. Briefing written for {today_str}.")


from fastapi.responses import FileResponse

@app.get("/api/novel/library")
def get_novel_library():
    from scout.romantasy_studio import RomantasyStudio
    studio = RomantasyStudio()
    return studio.get_library_catalog()

@app.get("/api/novel/progress")
def get_novel_progress(slug: str = "a_crown_of_gilded_bones"):
    from scout.romantasy_studio import NOVELS_ROOT, VAULT_DB_PATH
    import sqlite3
    
    conn = sqlite3.connect(VAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM novel_library WHERE slug = ?;", (slug,))
    book = cur.fetchone()
    if not book:
        cur.execute("SELECT * FROM novel_library WHERE slug = 'a_crown_of_gilded_bones';")
        book = cur.fetchone()
    if not book:
        cur.execute("SELECT * FROM novel_library ORDER BY id ASC LIMIT 1;")
        book = cur.fetchone()
    conn.close()
    
    if not book:
        return {"error": "No books found", "chapter_details": []}
        
    b_dict = dict(book)
    actual_slug = b_dict["slug"]
    book_dir = NOVELS_ROOT / actual_slug
    chapters_dir = book_dir / "chapters"
    
    chapter_list = []
    if chapters_dir.exists():
        for ch_f in sorted(chapters_dir.glob("chapter_*.md")):
            text = ch_f.read_text(encoding="utf-8")
            words = len(text.split())
            try:
                num = int(ch_f.stem.split("_")[1])
            except:
                num = len(chapter_list) + 1
            first_line = text.split("\n")[0].replace("# ", "").strip()
            chapter_list.append({
                "chapter": num,
                "title": first_line,
                "words": words,
                "path": str(ch_f)
            })
            
    total_words = sum(c["words"] for c in chapter_list)
    target_words = b_dict.get("target_words", 50000)
    pct = round((total_words / target_words) * 100, 1) if target_words > 0 else 0
    
    return {
        "title": b_dict.get("title", actual_slug),
        "slug": actual_slug,
        "subgenre": b_dict.get("subgenre", "Dark Gothic Romantasy"),
        "total_words": total_words,
        "target_words": target_words,
        "percent_complete": pct,
        "completed_chapters": len(chapter_list),
        "total_chapters": b_dict.get("total_chapters", 30),
        "chapter_details": chapter_list
    }


@app.get("/api/novel/chapter/{chapter_number}")
def get_novel_chapter(chapter_number: int, slug: str = "tethered_in_smoke_and_sin"):
    from scout.romantasy_studio import NOVELS_ROOT
    from scout.novelist import CHAPTERS_DIR
    if slug and slug != "tethered_in_smoke_and_sin":
        ch_file = NOVELS_ROOT / slug / "chapters" / f"chapter_{chapter_number:02d}.md"
    else:
        ch_file = CHAPTERS_DIR / f"chapter_{chapter_number:02d}.md"

    if not ch_file.exists():
        raise HTTPException(status_code=404, detail=f"Chapter {chapter_number} not found for {slug}")
    content = ch_file.read_text(encoding="utf-8")
    title_line = content.split("\n")[0].replace("# ", "").strip()
    return {
        "chapter_number": chapter_number,
        "title": title_line,
        "content": content,
        "word_count": len(content.split()),
        "slug": slug
    }

@app.get("/api/novel/manuscript")
def get_novel_manuscript():
    from scout.novelist import PrimeNovelist, MANUSCRIPT_PATH
    if not MANUSCRIPT_PATH.exists():
        novelist = PrimeNovelist()
        novelist.assemble_manuscript()
    content = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    return {"content": content, "total_words": len(content.split())}

@app.get("/api/novel/download")
def download_novel_manuscript(format: str = "zip"):
    from scout.novelist import NOVEL_DIR, MANUSCRIPT_PATH
    export_dir = NOVEL_DIR / "export"
    if format == "epub":
        return FileResponse(path=str(export_dir / "Tethered_in_Smoke_and_Sin.epub"), filename="Tethered_in_Smoke_and_Sin.epub", media_type="application/epub+zip")
    elif format == "pdf":
        return FileResponse(path=str(export_dir / "Tethered_in_Smoke_and_Sin.pdf"), filename="Tethered_in_Smoke_and_Sin.pdf", media_type="application/pdf")
    elif format == "html":
        return FileResponse(path=str(export_dir / "Tethered_in_Smoke_and_Sin_Offline_Reader.html"), filename="Tethered_in_Smoke_and_Sin_Offline_Reader.html", media_type="text/html")
    elif format == "txt":
        return FileResponse(path=str(export_dir / "Tethered_in_Smoke_and_Sin.txt"), filename="Tethered_in_Smoke_and_Sin.txt", media_type="text/plain")
    elif format == "md":
        return FileResponse(path=str(MANUSCRIPT_PATH), filename="Tethered_in_Smoke_and_Sin_Manuscript.md", media_type="text/markdown")
    else:
        return FileResponse(path=str(export_dir / "Tethered_in_Smoke_and_Sin_Complete_Package.zip"), filename="Tethered_in_Smoke_and_Sin_Complete_Package.zip", media_type="application/zip")

@app.get("/api/novel/sample_pdf")
def get_sample_pdf():
    sample_path = Path(__file__).resolve().parents[1] / "novels" / "a_crown_of_gilded_bones" / "export" / "A_Crown_of_Gilded_Bones_Chapter_1_Sample.pdf"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample PDF not found")
    return FileResponse(
        path=str(sample_path),
        filename="A_Crown_of_Gilded_Bones_Chapter_1_Sample.pdf",
        media_type="application/pdf"
    )

_shared_author = None

def get_author():
    global _shared_author
    if _shared_author is None:
        from scout.prime_local_author import PrimeLocalAuthor
        _shared_author = PrimeLocalAuthor()
    return _shared_author

class AuthorPromptRequest(BaseModel):
    prompt: str
    heat_level: int = 5
    max_tokens: int = 800

@app.post("/api/author/prompt")
def direct_author_prompt_endpoint(req: AuthorPromptRequest):
    import torch
    author = get_author()
    t0 = time.time()
    generated = author.generate_scene(req.prompt, max_new_tokens=req.max_tokens, temperature=0.8, top_p=0.92)
    gen_time = time.time() - t0
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    
    return {
        "text": generated,
        "words": len(generated.split()),
        "time_seconds": round(gen_time, 2),
        "vram_gb": round(vram_gb, 2)
    }

class Launch50kRequest(BaseModel):
    title: str
    prompt: str
    target_words: int = 50000
    total_chapters: int = 20
    heat_level: int = 5
    subgenre: str = "Dark Gothic Stalker Romantasy"

@app.post("/api/novel/launch_50k")
def launch_50k_novel_endpoint(req: Launch50kRequest):
    import re
    from scout.romantasy_studio import RomantasyStudio, NOVELS_ROOT, VAULT_DB_PATH
    
    slug = re.sub(r'[^a-z0-9]+', '_', req.title.lower()).strip('_')
    if not slug:
        slug = f"novel_{int(time.time())}"
        
    studio = RomantasyStudio()
    book_meta = {
        "slug": slug,
        "title": req.title,
        "subgenre": req.subgenre,
        "target_words": req.target_words,
        "current_words": 0,
        "total_chapters": req.total_chapters,
        "completed_chapters": 0,
        "status": "WRITING",
        "heat_level": req.heat_level,
        "synopsis": req.prompt
    }
    studio.register_book_in_library(book_meta)
    
    book_dir = NOVELS_ROOT / slug
    book_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir = book_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    
    bible = {
        "slug": slug,
        "title": req.title,
        "subgenre": req.subgenre,
        "target_words": req.target_words,
        "total_chapters": req.total_chapters,
        "heat_level": req.heat_level,
        "synopsis": req.prompt,
        "chapters": [
            {
                "chapter": i + 1,
                "title": f"Chapter {i + 1}",
                "target_words": round(req.target_words / req.total_chapters),
                "act": 1 if i < req.total_chapters * 0.25 else (2 if i < req.total_chapters * 0.75 else 3),
                "status": "QUEUED"
            }
            for i in range(req.total_chapters)
        ]
    }
    (book_dir / "BOOK_BIBLE.json").write_text(json.dumps(bible, indent=2), encoding="utf-8")
    
    return {
        "status": "LAUNCHED",
        "slug": slug,
        "title": req.title,
        "target_words": req.target_words,
        "total_chapters": req.total_chapters
    }

@app.get("/api/novel/status_detail")
def get_novel_status_detail(slug: str = "a_crown_of_gilded_bones"):
    from scout.romantasy_studio import NOVELS_ROOT, VAULT_DB_PATH
    import sqlite3
    
    conn = sqlite3.connect(VAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM novel_library WHERE slug = ?;", (slug,))
    book = cur.fetchone()
    if not book:
        cur.execute("SELECT * FROM novel_library WHERE status = 'WRITING' ORDER BY id ASC LIMIT 1;")
        book = cur.fetchone()
    if not book:
        cur.execute("SELECT * FROM novel_library ORDER BY id ASC LIMIT 1;")
        book = cur.fetchone()
    conn.close()
    
    if not book:
        return {"error": "No books found"}
        
    b_dict = dict(book)
    book_dir = NOVELS_ROOT / b_dict["slug"]
    chapters_dir = book_dir / "chapters"
    
    chapter_list = []
    if chapters_dir.exists():
        for ch_f in sorted(chapters_dir.glob("chapter_*.md")):
            text = ch_f.read_text(encoding="utf-8")
            words = len(text.split())
            try:
                num = int(ch_f.stem.split("_")[1])
            except:
                num = len(chapter_list) + 1
            title_line = text.split("\n")[0].replace("# ", "")
            chapter_list.append({
                "chapter": num,
                "title": title_line,
                "words": words,
                "path": str(ch_f)
            })
            
    total_words = sum(c["words"] for c in chapter_list)
    b_dict["current_words"] = total_words
    b_dict["completed_chapters"] = len(chapter_list)
    b_dict["chapter_list"] = chapter_list
    pct = round((total_words / b_dict["target_words"]) * 100, 1) if b_dict["target_words"] > 0 else 0
    b_dict["percent_complete"] = pct
    
    return b_dict

@app.post("/api/novel/draft_chapter_step")
def draft_chapter_step_endpoint(slug: str = "a_crown_of_gilded_bones"):
    from scout.romantasy_studio import NOVELS_ROOT, VAULT_DB_PATH
    from scout.prime_local_author import PrimeLocalAuthor
    import sqlite3
    
    book_dir = NOVELS_ROOT / slug
    chapters_dir = book_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(VAULT_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT total_chapters, title, target_words FROM novel_library WHERE slug = ?;", (slug,))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Novel not found")
        
    total_chapters, title, target_words = row
    written_files = sorted(chapters_dir.glob("chapter_*.md"))
    next_ch = len(written_files) + 1
    
    if next_ch > total_chapters:
        return {"status": "ALREADY_COMPLETE", "slug": slug, "completed": total_chapters}
        
    author = get_author()
    res = author.draft_chapter(slug, next_ch)
    
    all_chapters = sorted(chapters_dir.glob("chapter_*.md"))
    total_words = sum(len(cf.read_text(encoding="utf-8").split()) for cf in all_chapters)
    
    conn = sqlite3.connect(VAULT_DB_PATH)
    cur = conn.cursor()
    now_str = datetime.now(timezone.utc).isoformat()
    cur.execute("""
    UPDATE novel_library
    SET current_words = ?, completed_chapters = ?, status = ?, updated_at = ?
    WHERE slug = ?;
    """, (total_words, len(all_chapters), "COMPLETED" if len(all_chapters) >= total_chapters else "WRITING", now_str, slug))
    conn.commit()
    conn.close()
    
    ch_file = chapters_dir / f"chapter_{next_ch:02d}.md"
    ch_text = ch_file.read_text(encoding="utf-8") if ch_file.exists() else ""

    return {
        "status": "SUCCESS",
        "chapter": next_ch,
        "title": res["title"],
        "words": res["words"],
        "total_words": total_words,
        "percent": round((total_words / target_words) * 100, 1),
        "content": ch_text
    }

@app.get("/api/novel/latest_chapter")
def get_latest_chapter(slug: str = "a_crown_of_gilded_bones"):
    from scout.romantasy_studio import NOVELS_ROOT
    chapters_dir = NOVELS_ROOT / slug / "chapters"
    if not chapters_dir.exists():
        raise HTTPException(status_code=404, detail="Novel not found")
    ch_files = sorted(chapters_dir.glob("chapter_*.md"))
    if not ch_files:
        raise HTTPException(status_code=404, detail="No chapters found")
    latest_file = ch_files[-1]
    content = latest_file.read_text(encoding="utf-8")
    title_line = content.split("\n")[0].replace("# ", "").strip()
    return {
        "slug": slug,
        "chapter": len(ch_files),
        "title": title_line,
        "content": content,
        "words": len(content.split())
    }

STANDALONE_READER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRIME Reader — Dark Romantasy E-Reader</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --accent: #f472b6;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --font-serif: "Georgia", "Palatino", "Liberation Serif", serif;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: var(--font-serif);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            line-height: 1.85;
        }
        header {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 0.85rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .header-left { display: flex; align-items: center; gap: 1rem; }
        .reader-controls { display: flex; align-items: center; gap: 0.8rem; }
        select, button {
            background: #21262d;
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0.45rem 0.85rem;
            border-radius: 6px;
            font-size: 0.88rem;
            cursor: pointer;
        }
        select:focus, button:focus { outline: none; border-color: var(--accent); }
        .container {
            max-width: 820px;
            margin: 2.5rem auto;
            padding: 0 1.5rem 6rem;
            flex: 1;
        }
        .chapter-header {
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        .chapter-title { font-size: 2.2rem; color: var(--accent); margin-bottom: 0.5rem; font-weight: 700; }
        .chapter-meta { font-size: 0.92rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
        
        #chapter-body {
            font-size: 1.15rem;
            line-height: 1.85;
            color: var(--text);
            text-rendering: optimizeLegibility;
        }
        #chapter-body p {
            margin-bottom: 1.35rem;
            text-indent: 2rem;
            text-align: justify;
            text-justify: inter-word;
            hyphens: auto;
        }
        #chapter-body p.no-indent,
        #chapter-body p:first-of-type {
            text-indent: 0 !important;
        }
        #chapter-body p.lead-dropcap:first-of-type::first-letter {
            float: left;
            font-size: 3.6em;
            line-height: 0.8;
            padding-top: 4px;
            padding-right: 10px;
            padding-bottom: 2px;
            font-family: var(--font-serif);
            color: var(--accent);
            font-weight: bold;
        }
        #chapter-body em {
            font-style: italic;
            color: #fbcfe8;
        }
        #chapter-body strong {
            color: #ffffff;
            font-weight: 600;
        }
        .scene-break {
            text-align: center;
            margin: 2.5rem 0;
            color: var(--accent);
            font-size: 1.3rem;
            letter-spacing: 0.5em;
            opacity: 0.85;
            user-select: none;
        }
        .nav-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 3.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }
        .btn-nav {
            background: linear-gradient(135deg, #ec4899, #8b5cf6);
            color: white;
            border: none;
            padding: 0.6rem 1.4rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-nav:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-left">
            <a href="/" style="color: var(--text-muted); text-decoration: none; font-size: 0.85rem;">← Dashboard</a>
            <select id="novel-select" onchange="changeNovel(this.value)" style="font-weight: 700; color: var(--accent); max-width: 280px;"></select>
            <span id="novel-badge" style="font-size: 0.78rem; background: rgba(236,72,153,0.2); color: #f472b6; padding: 0.2rem 0.6rem; border-radius: 9999px;">Loading...</span>
        </div>
        <div class="reader-controls">
            <select id="chapter-select" onchange="changeChapter(this.value)"></select>
            <button onclick="toggleFont()">Font: Aa</button>
            <button onclick="toggleTheme()">Theme</button>
            <a id="download-pdf-btn" href="/api/novel/sample_pdf" target="_blank" style="text-decoration: none;">
                <button style="background: linear-gradient(135deg, #ec4899, #8b5cf6); color: white; border: none; font-weight: 600;">📥 Sample PDF</button>
            </a>
        </div>
    </header>

    <div class="container">
        <div class="chapter-header">
            <h1 class="chapter-title" id="disp-title">Loading...</h1>
            <div class="chapter-meta" id="disp-meta">Chapter 1</div>
        </div>
        <div id="chapter-body"></div>
        <div class="nav-footer">
            <button class="btn-nav" id="btn-prev" onclick="prevChapter()">← Previous Chapter</button>
            <span id="footer-progress" style="font-size: 0.85rem; color: var(--text-muted);"></span>
            <button class="btn-nav" id="btn-next" onclick="nextChapter()">Next Chapter →</button>
        </div>
    </div>

    <script>
        let curSlug = 'a_crown_of_gilded_bones';
        let curChapter = 1;
        let totalChapters = 1;
        let chaptersData = [];
        let isSerif = true;
        let themeIdx = 0;
        const themes = [
            { bg: '#0d1117', text: '#e6edf3', surface: '#161b22', border: '#30363d' },
            { bg: '#2b2622', text: '#f5e8d8', surface: '#36302b', border: '#4d443c' },
            { bg: '#fbf0d9', text: '#2d251e', surface: '#ede0c7', border: '#d9caa8' }
        ];

        function formatManuscriptText(rawText) {
            if (!rawText) return '';
            let text = rawText
                .replace(/<\\|[a-z0-9_]+\\|>/gi, '')
                .replace(/^(Instruction|Target Heat Level|Novel|Objective|Premise|Chapter \\d+:?):.*$/gmi, '')
                .replace(/^Story text:?\\s*/gmi, '')
                .replace(/^#{1,6}\\s+.*$/gm, '')
                .trim();

            text = text.replace(/^(\\s*(\\*|\\-)\\s*){3,}$/gm, '___SCENE_BREAK___');

            let paragraphs = text.split(/\\n\\s*\\n+/);
            let htmlParts = [];
            let isFirstAfterBreak = true;
            let isFirstInChapter = true;

            for (let p of paragraphs) {
                let trimmed = p.trim();
                if (!trimmed) continue;

                if (trimmed === '___SCENE_BREAK___') {
                    htmlParts.push('<div class="scene-break">✦ ✦ ✦</div>');
                    isFirstAfterBreak = true;
                    continue;
                }

                let escaped = trimmed
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');

                escaped = escaped.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
                escaped = escaped.replace(/__(.+?)__/g, '<strong>$1</strong>');
                escaped = escaped.replace(/\\*([^\\*\\n]+?)\\*/g, '<em>$1</em>');
                escaped = escaped.replace(/_([^\\_\\n]+?)_/g, '<em>$1</em>');
                escaped = escaped.replace(/\\n\\s*([“"«—\\-])/g, '<br>$1');
                escaped = escaped.replace(/\\n\\s*/g, ' ');

                let classes = [];
                if (isFirstAfterBreak) {
                    classes.push('no-indent');
                    if (isFirstInChapter) {
                        classes.push('lead-dropcap');
                        isFirstInChapter = false;
                    }
                    isFirstAfterBreak = false;
                }

                let classAttr = classes.length > 0 ? ` class="${classes.join(' ')}"` : '';
                htmlParts.push(`<p${classAttr}>${escaped}</p>`);
            }

            return htmlParts.join('\\n');
        }

        function init() {
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.has('slug')) curSlug = urlParams.get('slug');

            fetch('/api/novel/library')
                .then(r => r.json())
                .then(books => {
                    const novelSel = document.getElementById('novel-select');
                    novelSel.innerHTML = '';
                    books.forEach(b => {
                        const opt = document.createElement('option');
                        opt.value = b.slug;
                        opt.textContent = '📖 ' + b.title;
                        if (b.slug === curSlug) opt.selected = true;
                        novelSel.appendChild(opt);
                    });
                    loadNovel(curSlug);
                });
        }

        function changeNovel(slug) {
            curSlug = slug;
            loadNovel(slug);
        }

        function loadNovel(slug) {
            fetch('/api/novel/progress?slug=' + encodeURIComponent(slug))
                .then(r => r.json())
                .then(data => {
                    totalChapters = data.total_chapters || 1;
                    chaptersData = data.chapter_details || [];
                    
                    const badge = document.getElementById('novel-badge');
                    badge.textContent = `${(data.total_words || 0).toLocaleString()} Words • ${data.completed_chapters}/${totalChapters} Chs`;
                    
                    const chSel = document.getElementById('chapter-select');
                    chSel.innerHTML = '';
                    if (chaptersData.length === 0) {
                        const opt = document.createElement('option');
                        opt.value = 1;
                        opt.textContent = 'No chapters yet';
                        chSel.appendChild(opt);
                        document.getElementById('disp-title').textContent = data.title;
                        document.getElementById('disp-meta').textContent = 'Drafting in progress...';
                        document.getElementById('chapter-body').innerHTML = '<p class="no-indent" style="text-align:center; color: var(--text-muted); margin-top: 3rem;">This novel is currently queued to be drafted on the local GPU.</p>';
                        return;
                    }
                    
                    chaptersData.forEach(ch => {
                        const opt = document.createElement('option');
                        opt.value = ch.chapter;
                        opt.textContent = 'Ch ' + ch.chapter + ': ' + ch.title + ' (' + ch.words.toLocaleString() + 'w)';
                        chSel.appendChild(opt);
                    });
                    
                    loadChapter(chaptersData[0].chapter);
                });
        }

        function loadChapter(num) {
            curChapter = num;
            document.getElementById('chapter-select').value = num;
            fetch(`/api/novel/chapter/${num}?slug=${encodeURIComponent(curSlug)}`)
                .then(r => r.json())
                .then(data => {
                    document.getElementById('disp-title').textContent = data.title;
                    document.getElementById('disp-meta').textContent = `Chapter ${num} of ${totalChapters} • ${data.word_count.toLocaleString()} words`;
                    document.getElementById('footer-progress').textContent = `Reading Chapter ${num} of ${totalChapters}`;
                    
                    document.getElementById('chapter-body').innerHTML = formatManuscriptText(data.content);
                    window.scrollTo({ top: 0, behavior: 'smooth' });

                    const chIdx = chaptersData.findIndex(c => c.chapter === num);
                    document.getElementById('btn-prev').disabled = (chIdx <= 0);
                    document.getElementById('btn-next').disabled = (chIdx >= chaptersData.length - 1 || chIdx < 0);
                });
        }

        function changeChapter(val) { loadChapter(parseInt(val)); }
        function prevChapter() {
            const chIdx = chaptersData.findIndex(c => c.chapter === curChapter);
            if (chIdx > 0) loadChapter(chaptersData[chIdx - 1].chapter);
        }
        function nextChapter() {
            const chIdx = chaptersData.findIndex(c => c.chapter === curChapter);
            if (chIdx < chaptersData.length - 1 && chIdx >= 0) loadChapter(chaptersData[chIdx + 1].chapter);
        }

        function toggleFont() {
            isSerif = !isSerif;
            document.body.style.fontFamily = isSerif ? 'Georgia, serif' : '-apple-system, BlinkMacSystemFont, sans-serif';
        }

        function toggleTheme() {
            themeIdx = (themeIdx + 1) % themes.length;
            const t = themes[themeIdx];
            document.documentElement.style.setProperty('--bg', t.bg);
            document.documentElement.style.setProperty('--text', t.text);
            document.documentElement.style.setProperty('--surface', t.surface);
            document.documentElement.style.setProperty('--border', t.border);
        }

        window.onload = init;
    </script>
</body>
</html>
"""

STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>👑 PRIME Local Author Studio — Easy Desktop Control</title>
    <style>
        :root {
            --bg: #090d16;
            --surface: #121824;
            --surface-hover: #1b2333;
            --border: #232d42;
            --accent: #f43f5e;
            --accent-hover: #fb7185;
            --accent-glow: rgba(244, 63, 94, 0.25);
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --font-serif: "Georgia", "Palatino", serif;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: var(--font-ui);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 0.9rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: 0.05em;
        }
        .badge-gpu {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #34d399;
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 9999px;
            font-family: monospace;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .nav-links {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .nav-btn {
            background: #1e293b;
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0.45rem 0.9rem;
            border-radius: 6px;
            font-size: 0.85rem;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .nav-btn:hover {
            background: var(--surface-hover);
            border-color: var(--accent);
        }
        .layout {
            max-width: 1200px;
            margin: 1.5rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 1.5rem;
            flex: 1;
            width: 100%;
        }
        @media (max-width: 900px) {
            .layout { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .card-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        label {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
            display: block;
        }
        textarea {
            width: 100%;
            height: 140px;
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 0.75rem;
            font-family: var(--font-ui);
            font-size: 0.95rem;
            line-height: 1.5;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }
        .tropes-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .trope-pill {
            background: #1e293b;
            border: 1px solid var(--border);
            color: #cbd5e1;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.15s;
        }
        .trope-pill:hover {
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }
        .heat-selector {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.35rem;
        }
        .heat-btn {
            background: #1e293b;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.5rem 0.2rem;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            text-align: center;
            transition: all 0.15s;
        }
        .heat-btn.active {
            background: var(--accent);
            border-color: var(--accent-hover);
            color: white;
            font-weight: bold;
            box-shadow: 0 0 10px var(--accent-glow);
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }
        select {
            width: 100%;
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            padding: 0.5rem 0.65rem;
            font-size: 0.88rem;
        }
        .btn-generate {
            background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
            color: white;
            border: none;
            padding: 0.85rem 1.25rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            box-shadow: 0 4px 14px var(--accent-glow);
            transition: all 0.15s ease;
        }
        .btn-generate:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }
        .btn-generate:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        /* Right Output Panel */
        .output-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .output-header {
            background: #161e2e;
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .output-meta {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-family: monospace;
        }
        .output-actions {
            display: flex;
            gap: 0.5rem;
        }
        .output-canvas {
            flex: 1;
            padding: 2.5rem 3rem;
            overflow-y: auto;
            max-height: 750px;
            font-family: var(--font-serif);
            font-size: 1.15rem;
            line-height: 1.85;
            color: #f8fafc;
            background: #0b0f19;
            text-rendering: optimizeLegibility;
        }
        .output-canvas p {
            margin-bottom: 1.35rem;
            text-indent: 2rem;
            text-align: justify;
            text-justify: inter-word;
            hyphens: auto;
        }
        .output-canvas p.no-indent,
        .output-canvas p:first-of-type {
            text-indent: 0 !important;
        }
        .output-canvas p.lead-dropcap:first-of-type::first-letter {
            float: left;
            font-size: 3.6em;
            line-height: 0.8;
            padding-top: 4px;
            padding-right: 10px;
            padding-bottom: 2px;
            font-family: var(--font-serif);
            color: var(--accent);
            font-weight: bold;
        }
        .output-canvas em {
            font-style: italic;
            color: #fbcfe8;
        }
        .output-canvas strong {
            color: #ffffff;
            font-weight: 600;
        }
        .output-canvas .scene-break {
            text-align: center;
            margin: 2.5rem 0;
            color: var(--accent);
            font-size: 1.3rem;
            letter-spacing: 0.5em;
            opacity: 0.85;
            user-select: none;
        }

        .loading-state {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 300px;
            gap: 1rem;
            color: var(--accent);
        }
        .spinner {
            width: 44px;
            height: 44px;
            border: 3px solid rgba(244, 63, 94, 0.2);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .mode-tabs {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .mode-tab {
            background: #161e2e;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.6rem 0.5rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            text-align: center;
            transition: all 0.15s;
        }
        .mode-tab.active {
            background: #1e293b;
            color: var(--accent);
            border-color: var(--accent);
            box-shadow: 0 0 10px var(--accent-glow);
        }
        .progress-box {
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.9rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }
        .progress-bar-bg {
            background: #1e293b;
            height: 10px;
            border-radius: 9999px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #f43f5e 0%, #ec4899 100%);
            height: 100%;
            width: 0%;
            transition: width 0.4s ease;
        }
        .chapter-pill-list {
            max-height: 160px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            padding-right: 4px;
        }
        .chapter-pill {
            background: #161e2e;
            border: 1px solid var(--border);
            padding: 0.35rem 0.6rem;
            border-radius: 4px;
            font-size: 0.78rem;
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span>👑 PRIME Local Author Studio</span>
            <span class="badge-gpu">● AMD ROCm • Meta-Llama-3-8B</span>
        </div>
        <div class="nav-links">
            <a href="/reader" class="nav-btn">📖 Open Reader</a>
            <a href="/api/novel/sample_pdf" class="nav-btn">📄 Sample PDF</a>
            <a href="/" class="nav-btn">🔬 Lab Dashboard</a>
        </div>
    </header>

    <div class="layout">
        <!-- Control Panel -->
        <div class="card">
            <div class="mode-tabs">
                <button class="mode-tab active" id="tab-scene" onclick="switchMode('scene')">📝 Scene / Chapter Mode</button>
                <button class="mode-tab" id="tab-50k" onclick="switchMode('50k')">📚 Full 50k Novel Mode</button>
            </div>

            <!-- Single Scene Mode -->
            <div id="section-scene" style="display: flex; flex-direction: column; gap: 1rem;">
                <div class="card-title">✍️ Tell Local AI What To Write</div>
                
                <div>
                    <label>Direct Story / Scene Prompt</label>
                    <textarea id="prompt-input" placeholder="e.g. Write a dark, intense scene where Caelum pins Aurelia against the obsidian altar in the bone crypts. Knife to throat, explicit dirty talk, he makes her beg for it..."></textarea>
                </div>

                <div>
                    <label>⚡ Quick Trope Ideas (Click to Fill)</label>
                    <div class="tropes-container">
                        <span class="trope-pill" onclick="setPrompt('Caelum catches Aurelia harvesting venom in the dead of night. He disarms her, pins her against the bone wyrm-rib, and forces her to feel his hardness while kissing her neck.')">🔪 Knife & Wall Pin</span>
                        <span class="trope-pill" onclick="setPrompt('Aurelia is bathing in the executioner estate. Caelum walks in blind, feeling her heartbeat and wet skin through bone-resonance, refusing to leave.')">🛁 Bathhouse Intrusion</span>
                        <span class="trope-pill" onclick="setPrompt('Primal chase through the gothic catacombs. Caelum whispers \'Run, little viper\' into the dark as she desperately flees.')">🩸 Primal Chase</span>
                        <span class="trope-pill" onclick="setPrompt('A rival noble tries to lay hands on Aurelia at the royal banquet. Caelum unleashes unhinged vigilante fury in her defense.')">👑 \'Who Did This To You?\'</span>
                        <span class="trope-pill" onclick="setPrompt('Forced betrothal bedroom scene. Aurelia has a blade under her pillow; Caelum slides into her bed and takes it from her.')">🖤 Bedchamber Claim</span>
                    </div>
                </div>

                <div>
                    <label>🔥 Spice / Smut Level</label>
                    <div class="heat-selector">
                        <button class="heat-btn" onclick="setHeat(1, this)">🌶️ 1<br><small>Tension</small></button>
                        <button class="heat-btn" onclick="setHeat(2, this)">🌶️ 2<br><small>Banter</small></button>
                        <button class="heat-btn" onclick="setHeat(3, this)">🌶️ 3<br><small>Steamy</small></button>
                        <button class="heat-btn" onclick="setHeat(4, this)">🌶️ 4<br><small>Explicit</small></button>
                        <button class="heat-btn active" onclick="setHeat(5, this)">🌶️ 5<br><small>Smut</small></button>
                    </div>
                </div>

                <div class="form-row">
                    <div>
                        <label>Target Length</label>
                        <select id="token-select">
                            <option value="600">Short Scene (~500 words)</option>
                            <option value="900" selected>Full Scene (~850 words)</option>
                            <option value="1200">Extended Scene (~1,100 words)</option>
                        </select>
                    </div>
                    <div>
                        <label>Character Context</label>
                        <select id="char-select">
                            <option value="caelum_aurelia" selected>Caelum & Aurelia (Gilded Bones)</option>
                            <option value="vaelen_elena">Vaelen & Elena (Smoke & Sin)</option>
                            <option value="custom">Custom / Freestyle</option>
                        </select>
                    </div>
                </div>

                <button class="btn-generate" id="btn-generate" onclick="generateProse()">
                    <span>⚡ Generate Scene on Local GPU</span>
                </button>
            </div>

            <!-- Full 50,000-Word Novel Mode -->
            <div id="section-50k" style="display: none; flex-direction: column; gap: 1rem;">
                <div class="card-title">🚀 Full 50,000-Word Novel Generator</div>
                
                <div>
                    <label>Book Title</label>
                    <input type="text" id="novel-title-input" value="A Crown of Gilded Bones" style="width: 100%; background: #090d16; border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 0.6rem; font-size: 0.95rem;">
                </div>

                <div>
                    <label>Premise & Dark Tropes</label>
                    <textarea id="novel-premise-input" style="height: 90px;">In a subterranean gothic bone kingdom, a blind royal executioner obsessed with a lethal poisoner stalks her through the catacombs. Forced betrothal, knife play, 5/5 heat.</textarea>
                </div>

                <div class="form-row">
                    <div>
                        <label>Target Word Count</label>
                        <select id="novel-words-select">
                            <option value="50000" selected>50,000 Words (20 Chapters • Full Book)</option>
                            <option value="75000">75,000 Words (30 Chapters)</option>
                            <option value="100000">100,000 Words (40 Chapters)</option>
                        </select>
                    </div>
                    <div>
                        <label>Spice Setting</label>
                        <select id="novel-heat-select">
                            <option value="5" selected>🌶️ 5/5 Extreme Dark Smut</option>
                            <option value="4">🌶️ 4/5 High Heat</option>
                            <option value="3">🌶️ 3/5 Steamy</option>
                        </select>
                    </div>
                </div>

                <button class="btn-generate" onclick="launch50kNovel()">
                    <span>🚀 Launch 50,000-Word Autonomous Production</span>
                </button>

                <!-- Live 50k Progress Card -->
                <div class="progress-box" id="novel-live-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong id="live-novel-title" style="color: var(--accent);">A Crown of Gilded Bones</strong>
                        <span id="live-novel-status" style="font-size: 0.75rem; background: #1e293b; padding: 2px 6px; border-radius: 4px;">WRITING</span>
                    </div>

                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" id="live-progress-fill" style="width: 3%;"></div>
                    </div>

                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted);">
                        <span id="live-words-count">1,567 / 75,000 words (2.1%)</span>
                        <span id="live-chapters-count">1 / 30 chapters</span>
                    </div>

                    <div class="chapter-pill-list" id="live-chapter-list">
                        <div class="chapter-pill"><span>Ch 1: The Ossuary Citadel</span><span>1,567 words [✓]</span></div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.3rem;">
                        <button class="nav-btn" onclick="draftNextChapter()" id="btn-next-ch" style="background: var(--accent); color: white; border: none; font-weight: 600;">⚡ Draft Next Chapter</button>
                        <a href="/reader" class="nav-btn" style="text-align: center;">📖 Open Reader</a>
                    </div>
                </div>
            </div>
        </div>

        <!-- Output Display -->
        <div class="output-card">
            <div class="output-header">
                <div class="output-meta" id="output-meta">Waiting for prompt...</div>
                <div class="output-actions">
                    <button class="nav-btn" onclick="copyText()">📋 Copy</button>
                    <a href="/api/novel/sample_pdf" class="nav-btn" target="_blank">📄 View PDF</a>
                </div>
            </div>

            <div class="loading-state" id="loading-state">
                <div class="spinner"></div>
                <div style="font-weight: 600; font-size: 1.05rem;" id="loading-title">Writing on AMD Radeon GPU (bfloat16)...</div>
                <div style="font-size: 0.85rem; color: var(--text-muted);" id="loading-sub">Executing local token generation with PRIME constant memory...</div>
            </div>

            <div class="output-canvas" id="output-canvas">
                <p style="color: var(--text-muted); font-style: italic; text-indent: 0;">
                    Your generated scene or 50,000-word book status will appear here formatted and ready to read.<br><br>
                    <strong>Choose a mode:</strong><br>
                    • <strong>📝 Scene Mode</strong>: Quick, high-heat scenes generated in seconds.<br>
                    • <strong>📚 Full 50k Novel Mode</strong>: Generates an entire 50,000-word novel chapter-by-chapter directly on your GPU.
                </p>
            </div>
        </div>
    </div>

    <script>
        let curHeat = 5;

        function setHeat(level, btn) {
            curHeat = level;
            document.querySelectorAll('.heat-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }

        function setPrompt(text) {
            document.getElementById('prompt-input').value = text;
        }

        function formatManuscriptText(rawText) {
            if (!rawText) return '';
            let text = rawText
                .replace(/<\\|[a-z0-9_]+\\|>/gi, '')
                .replace(/^(Instruction|Target Heat Level|Novel|Objective|Premise|Chapter \\d+:?):.*$/gmi, '')
                .replace(/^Story text:?\\s*/gmi, '')
                .replace(/^#{1,6}\\s+.*$/gm, '')
                .trim();

            text = text.replace(/^(\\\\s*(\\\\*|\\\\-)\\\\s*){3,}$/gm, '___SCENE_BREAK___');

            let paragraphs = text.split(/\\n\\s*\\n+/);
            let htmlParts = [];
            let isFirstAfterBreak = true;
            let isFirstInChapter = true;

            for (let p of paragraphs) {
                let trimmed = p.trim();
                if (!trimmed) continue;

                if (trimmed === '___SCENE_BREAK___') {
                    htmlParts.push('<div class="scene-break">✦ ✦ ✦</div>');
                    isFirstAfterBreak = true;
                    continue;
                }

                let escaped = trimmed
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');

                escaped = escaped.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
                escaped = escaped.replace(/__(.+?)__/g, '<strong>$1</strong>');
                escaped = escaped.replace(/\\*([^\\*\\n]+?)\\*/g, '<em>$1</em>');
                escaped = escaped.replace(/_([^\\_\\n]+?)_/g, '<em>$1</em>');
                escaped = escaped.replace(/\\n\\s*([“"«—\\-])/g, '<br>$1');
                escaped = escaped.replace(/\\n\\s*/g, ' ');

                let classes = [];
                if (isFirstAfterBreak) {
                    classes.push('no-indent');
                    if (isFirstInChapter) {
                        classes.push('lead-dropcap');
                        isFirstInChapter = false;
                    }
                    isFirstAfterBreak = false;
                }

                let classAttr = classes.length > 0 ? ` class="${classes.join(' ')}"` : '';
                htmlParts.push(`<p${classAttr}>${escaped}</p>`);
            }

            return htmlParts.join('\\n');
        }

        async function generateProse() {
            const prompt = document.getElementById('prompt-input').value.trim();
            if (!prompt) {
                alert('Please enter a prompt or click a quick trope!');
                return;
            }

            const btn = document.getElementById('btn-generate');
            const loading = document.getElementById('loading-state');
            const canvas = document.getElementById('output-canvas');
            const meta = document.getElementById('output-meta');
            const maxTokens = parseInt(document.getElementById('token-select').value);

            btn.disabled = true;
            btn.innerHTML = '<span>⏳ Generating on GPU...</span>';
            loading.style.display = 'flex';
            canvas.style.display = 'none';
            meta.textContent = 'Generating on AMD GPU...';

            try {
                const res = await fetch('/api/author/prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        heat_level: curHeat,
                        max_tokens: maxTokens
                    })
                });

                if (!res.ok) throw new Error('Generation failed: ' + res.statusText);
                const data = await res.json();

                canvas.innerHTML = formatManuscriptText(data.text);
                meta.textContent = `${data.words} words • ${data.time_seconds}s (${(data.words / data.time_seconds).toFixed(1)} w/s) • VRAM: ${data.vram_gb} GB`;
            } catch (err) {
                canvas.innerHTML = '<p style="color: #f87171;">Error: ' + err.message + '</p>';
                meta.textContent = 'Error occurred';
            } finally {
                loading.style.display = 'none';
                canvas.style.display = 'block';
                btn.disabled = false;
                btn.innerHTML = '<span>⚡ Generate on Local GPU</span>';
            }
        }

        let curMode = 'scene';
        let currentSlug = 'a_crown_of_gilded_bones';

        function switchMode(mode) {
            curMode = mode;
            if (mode === 'scene') {
                document.getElementById('tab-scene').classList.add('active');
                document.getElementById('tab-50k').classList.remove('active');
                document.getElementById('section-scene').style.display = 'flex';
                document.getElementById('section-50k').style.display = 'none';
            } else {
                document.getElementById('tab-50k').classList.add('active');
                document.getElementById('tab-scene').classList.remove('active');
                document.getElementById('section-scene').style.display = 'none';
                document.getElementById('section-50k').style.display = 'flex';
                refreshNovelStatus();
            }
        }

        async function refreshNovelStatus() {
            try {
                const res = await fetch('/api/novel/status_detail?slug=' + encodeURIComponent(currentSlug));
                if (!res.ok) return;
                const data = await res.json();
                if (data.title) {
                    document.getElementById('live-novel-title').textContent = data.title;
                    document.getElementById('live-novel-status').textContent = data.status || 'WRITING';
                    const pct = data.percent_complete || 0;
                    document.getElementById('live-progress-fill').style.width = pct + '%';
                    document.getElementById('live-words-count').textContent = `${(data.current_words || 0).toLocaleString()} / ${(data.target_words || 50000).toLocaleString()} words (${pct}%)`;
                    document.getElementById('live-chapters-count').textContent = `${data.completed_chapters || 0} / ${data.total_chapters || 20} chapters`;
                    
                    const listEl = document.getElementById('live-chapter-list');
                    if (data.chapter_list && data.chapter_list.length > 0) {
                        listEl.innerHTML = data.chapter_list.map(ch => 
                            `<div class="chapter-pill"><span>Ch ${ch.chapter}: ${ch.title}</span><span>${ch.words.toLocaleString()} words [✓]</span></div>`
                        ).join('');
                    } else {
                        listEl.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; padding: 0.25rem;">No chapters written yet. Click Draft Next Chapter or run the autonomous daemon!</div>';
                    }
                }
            } catch (e) {
                console.error('Error refreshing novel status:', e);
            }
        }

        async function launch50kNovel() {
            const title = document.getElementById('novel-title-input').value.trim();
            const prompt = document.getElementById('novel-premise-input').value.trim();
            const targetWords = parseInt(document.getElementById('novel-words-select').value);
            const heat = parseInt(document.getElementById('novel-heat-select').value);

            if (!title) {
                alert('Please enter a book title');
                return;
            }

            const canvas = document.getElementById('output-canvas');
            const meta = document.getElementById('output-meta');
            meta.textContent = 'Launching 50,000-Word Autonomous Novel Production...';
            canvas.innerHTML = `<h3>🚀 Registering '${title}' for Full Novel Production...</h3><p>Generating Book Bible, chapter progression outline, and SQLite tracking entry...</p>`;

            try {
                const totalCh = targetWords === 50000 ? 20 : (targetWords === 75000 ? 30 : 40);
                const res = await fetch('/api/novel/launch_50k', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: title,
                        prompt: prompt,
                        target_words: targetWords,
                        total_chapters: totalCh,
                        heat_level: heat
                    })
                });
                const data = await res.json();
                currentSlug = data.slug;
                canvas.innerHTML = `
                    <h2 style="color: var(--accent);">👑 Autonomous Novel Project Launched!</h2>
                    <p><strong>Title:</strong> ${data.title}</p>
                    <p><strong>Slug:</strong> <code>${data.slug}</code></p>
                    <p><strong>Target Length:</strong> ${data.target_words.toLocaleString()} words (${data.total_chapters} Chapters)</p>
                    <p><strong>Heat Level:</strong> 🌶️ ${heat}/5</p>
                    <div class="scene-break">✦ ✦ ✦</div>
                    <p>The book bible has been registered. The autonomous studio daemon and the "Draft Next Chapter" button below can now write chapters sequentially on your local GPU.</p>
                `;
                refreshNovelStatus();
            } catch (err) {
                canvas.innerHTML = `<p style="color: #f87171;">Error launching novel: ${err.message}</p>`;
            }
        }

        async function draftNextChapter() {
            const btn = document.getElementById('btn-next-ch');
            const loading = document.getElementById('loading-state');
            const canvas = document.getElementById('output-canvas');
            const meta = document.getElementById('output-meta');
            const loadingTitle = document.getElementById('loading-title');
            const loadingSub = document.getElementById('loading-sub');

            btn.disabled = true;
            btn.textContent = '⏳ Drafting on GPU...';
            loading.style.display = 'flex';
            canvas.style.display = 'none';
            loadingTitle.textContent = 'Drafting Next Chapter on AMD GPU...';
            loadingSub.textContent = 'Local Meta-Llama-3-8B writing multi-scene narrative with constant PRIME memory...';
            meta.textContent = 'Drafting chapter on GPU...';

            try {
                const res = await fetch('/api/novel/draft_chapter_step?slug=' + encodeURIComponent(currentSlug), {
                    method: 'POST'
                });
                const data = await res.json();
                if (data.status === 'ALREADY_COMPLETE') {
                    canvas.innerHTML = `<h2>🎉 Book Complete!</h2><p>All ${data.completed} chapters have been written.</p>`;
                } else {
                    canvas.innerHTML = `<h2 style="color: var(--accent); margin-bottom: 1.5rem; text-align: center;">${data.title}</h2>` + formatManuscriptText(data.content || '');
                    meta.textContent = `Drafted ${data.words || 0} words • Total Novel: ${(data.total_words || 0).toLocaleString()} words (${data.percent || 0}%)`;
                }
                refreshNovelStatus();
            } catch (err) {
                canvas.innerHTML = `<p style="color: #f87171;">Error drafting chapter: ${err.message}</p>`;
                meta.textContent = 'Error occurred';
            } finally {
                loading.style.display = 'none';
                canvas.style.display = 'block';
                btn.disabled = false;
                btn.textContent = '⚡ Draft Next Chapter';
            }
        }

        function copyText() {
            const text = document.getElementById('output-canvas').innerText;
            navigator.clipboard.writeText(text);
            alert('Copied to clipboard!');
        }

        async function initStudio() {
            refreshNovelStatus();
            try {
                const res = await fetch('/api/novel/latest_chapter?slug=' + encodeURIComponent(currentSlug));
                if (res.ok) {
                    const data = await res.json();
                    const canvas = document.getElementById('output-canvas');
                    const meta = document.getElementById('output-meta');
                    canvas.innerHTML = `<h2 style="color: var(--accent); margin-bottom: 1.5rem; text-align: center;">${data.title}</h2>` + formatManuscriptText(data.content || '');
                    meta.textContent = `Displaying ${data.title} (${data.words.toLocaleString()} words) • Ready for next scene or chapter.`;
                }
            } catch (e) {
                console.log('No existing chapter preview:', e);
            }
        }

        // Auto-refresh status if in 50k mode
        setInterval(() => {
            if (curMode === '50k') refreshNovelStatus();
        }, 12000);

        // Initial check on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initStudio);
        } else {
            initStudio();
        }
    </script>
</body>
</html>
"""

@app.get("/studio", response_class=HTMLResponse)
@app.get("/author", response_class=HTMLResponse)
@app.get("/easy", response_class=HTMLResponse)
def serve_author_studio():
    return HTMLResponse(content=STUDIO_HTML)

@app.get("/reader", response_class=HTMLResponse)
def serve_novel_reader():
    return HTMLResponse(content=STANDALONE_READER_HTML)

def run_server(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
