"""
PRIME-Scout: Modern Local Web Dashboard
FastAPI + Standalone Dark-Mode UI for reviewing daily briefings,
inspecting the memory vault, and running on-demand repository sandbox tests.
"""

import os
import shutil
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scout.config import UI_CONFIG, REPORTS_DIR, VAULT_DIR, SANDBOX_DIR, DB_PATH, MODEL_CONFIG
from scout.database import (
    get_connection, get_recent_evaluations, get_evaluation_by_url,
    get_briefings, upsert_repository, record_evaluation
)
from scout.sandbox import SandboxRunner
from scout.evaluator import PrimeScoutEvaluator
from scout.github_client import GitHubClient
from scout.reporter import BriefingReporter

app = FastAPI(title=UI_CONFIG["title"])

# Global singletons
sandbox_runner = SandboxRunner()
evaluator = PrimeScoutEvaluator(mode="auto")
github_client = GitHubClient()
reporter = BriefingReporter()

class RepoTestRequest(BaseModel):
    repo_url: str
    run_llm: bool = True

class ScanRequest(BaseModel):
    max_repos: int = 4

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRIME-Scout | Local AI Repository Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            line-height: 1.5;
        }
        header {
            background: rgba(17, 24, 39, 0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 0.85rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .brand-icon {
            font-size: 1.6rem;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .brand-subtitle {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
        .header-pills {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            background: rgba(36, 50, 74, 0.5);
            border: 1px solid var(--border);
            border-radius: 9999px;
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .dot-pulse {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }
        .btn {
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .btn:hover {
            opacity: 0.95;
            box-shadow: 0 0 15px var(--accent-glow);
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
        }
        .btn-secondary:hover {
            background: var(--surface-hover);
        }

        /* Nav Tabs */
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
            transition: all 0.2s ease;
        }
        .tab-btn:hover { color: var(--text); }
        .tab-btn.active {
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.2s ease; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

        /* Cards */
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
        .card-title {
            font-size: 1.15rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Briefing View */
        .markdown-body {
            color: #d1d5db;
            line-height: 1.7;
        }
        .markdown-body h1, .markdown-body h2, .markdown-body h3 {
            color: #f3f4f6;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            font-weight: 600;
        }
        .markdown-body table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        .markdown-body th, .markdown-body td {
            border: 1px solid var(--border);
            padding: 0.6rem 0.8rem;
            text-align: left;
        }
        .markdown-body th { background: rgba(36, 50, 74, 0.4); }
        .markdown-body pre {
            background: #060911;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .markdown-body code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(36, 50, 74, 0.6);
            padding: 0.15rem 0.35rem;
            border-radius: 4px;
            font-size: 0.85em;
        }
        .markdown-body blockquote {
            border-left: 3px solid var(--accent);
            padding-left: 1rem;
            color: var(--text-muted);
            margin: 1rem 0;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge-must-read { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-worth-exploring { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
        .badge-monitor { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-pass { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-sandbox-pass { background: rgba(16, 185, 129, 0.2); color: #10b981; }
        .badge-sandbox-missing { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
        .badge-sandbox-fail { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

        /* Grid */
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

        /* Inputs */
        .input-group {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .text-input {
            flex: 1;
            background: #060911;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text);
            font-size: 0.95rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .text-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 8px var(--accent-glow);
        }

        /* Live Terminal */
        .terminal {
            background: #050811;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: #a7f3d0;
            max-height: 280px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        /* Repo Card in Explorer */
        .repo-item {
            background: rgba(24, 33, 50, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        .repo-item:hover {
            border-color: var(--accent);
            transform: translateY(-1px);
        }
        .repo-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }
        .repo-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--accent);
            text-decoration: none;
        }
        .repo-scores {
            display: flex;
            gap: 0.5rem;
        }
        .score-pill {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.2rem 0.5rem;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Filesystem Explorer */
        .fs-tree {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            background: #060911;
            padding: 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .fs-item {
            padding: 0.25rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-icon">🔭</div>
            <div>
                <div class="brand-title">PRIME-Scout</div>
                <div class="brand-subtitle">Autonomous Local Intelligence Agent</div>
            </div>
        </div>
        <div class="header-pills">
            <div class="pill"><span class="dot-pulse"></span> AMD ROCm 7.2 Active</div>
            <div class="pill">🧠 Qwen2.5-Coder-1.5B (2.87 GB VRAM)</div>
            <button class="btn" onclick="triggerScan()">⚡ Run Discovery Now</button>
        </div>
    </header>

    <div class="tabs-nav">
        <button class="tab-btn active" onclick="showTab('tab-briefing')">📰 Daily Briefing</button>
        <button class="tab-btn" onclick="showTab('tab-sandbox')">🧪 Sandbox & Deep-Dive</button>
        <button class="tab-btn" onclick="showTab('tab-vault')">🏛️ Memory Vault (<span id="vault-count">0</span>)</button>
        <button class="tab-btn" onclick="showTab('tab-fs')">📁 Easy File System</button>
    </div>

    <main>
        <!-- Tab 1: Daily Briefing -->
        <div id="tab-briefing" class="tab-content active">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📅 Today's Intelligence Digest</div>
                    <button class="btn btn-secondary" onclick="loadLatestBriefing()">🔄 Refresh Briefing</button>
                </div>
                <div id="briefing-container" class="markdown-body">
                    <p style="color: var(--text-muted);">Loading latest briefing from <code>scout/reports/</code>...</p>
                </div>
            </div>
        </div>

        <!-- Tab 2: Interactive Sandbox & Deep-Dive -->
        <div id="tab-sandbox" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🧪 On-Demand Sandbox Clone & Evaluation</div>
                </div>
                <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
                    Paste any GitHub repository URL below. PRIME-Scout will shallow-clone it into <code>scout/sandbox/active/</code>,
                    audit dependencies for Triton/ROCm/CUDA, run an AST syntax check and isolated import smoke test, and stream a comprehensive evaluation via the local model.
                </p>
                <div class="input-group">
                    <input type="text" id="repo-url-input" class="text-input" placeholder="https://github.com/owner/repository" value="https://github.com/sustcsonglin/flash-linear-attention">
                    <button class="btn" onclick="runSandboxTest()">🚀 Execute Sandbox & Evaluate</button>
                </div>

                <div class="grid-2">
                    <div>
                        <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-muted);">Sandbox Terminal Output:</h4>
                        <div id="sandbox-term" class="terminal">[Ready] Awaiting repository URL...</div>
                    </div>
                    <div>
                        <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-muted);">Evaluation Result:</h4>
                        <div id="eval-result-card" style="background: #050811; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; min-height: 280px;">
                            <p style="color: var(--text-muted); font-size: 0.85rem;">Evaluation metrics and critique will appear here once the sandbox test finishes.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 3: Memory Vault -->
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

        <!-- Tab 4: File System Explorer -->
        <div id="tab-fs" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📁 PRIME-Scout File System Layout</div>
                    <button class="btn btn-secondary" onclick="cleanSandboxActive()">🧹 Clean Active Sandboxes</button>
                </div>
                <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
                    PRIME-Scout maintains a predictable, zero-clutter file system hierarchy designed for human inspection:
                </p>
                <div class="fs-tree" id="fs-details">
                    <div class="fs-item">📂 <b>scout/</b></div>
                    <div class="fs-item" style="padding-left: 1.5rem;">📁 <b>reports/</b> &nbsp; <span style="color: var(--text-muted);">(Daily markdown digests for reading)</span></div>
                    <div class="fs-item" style="padding-left: 3rem;">📄 <a href="/api/briefing/latest" target="_blank" style="color: var(--accent);">briefing_latest.md</a></div>
                    <div class="fs-item" style="padding-left: 1.5rem;">📁 <b>vault/</b> &nbsp; <span style="color: var(--text-muted);">(SQLite persistent memory & telemetry)</span></div>
                    <div class="fs-item" style="padding-left: 3rem;">🗄️ scout_vault.db</div>
                    <div class="fs-item" style="padding-left: 1.5rem;">📁 <b>sandbox/active/</b> &nbsp; <span style="color: var(--text-muted);">(Isolated git checkouts & smoke test chambers)</span></div>
                    <div id="sandbox-active-list" style="padding-left: 3rem; color: var(--text-muted);">Scanning active checkouts...</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            if (tabId === 'tab-vault') loadVault();
            if (tabId === 'tab-fs') loadFs();
        }

        async function loadLatestBriefing() {
            try {
                const res = await fetch('/api/briefing/latest');
                const data = await res.json();
                if (data.markdown) {
                    document.getElementById('briefing-container').innerHTML = marked.parse(data.markdown);
                } else {
                    document.getElementById('briefing-container').innerHTML = '<p>No briefing generated yet. Click <b>⚡ Run Discovery Now</b> to trigger your first briefing!</p>';
                }
            } catch (err) {
                console.error(err);
                document.getElementById('briefing-container').innerHTML = '<p style="color: var(--danger);">Failed to load briefing.</p>';
            }
        }

        async function loadVault() {
            try {
                const res = await fetch('/api/repos');
                const data = await res.json();
                document.getElementById('vault-count').innerText = data.repos.length;
                const container = document.getElementById('vault-list');
                if (data.repos.length === 0) {
                    container.innerHTML = '<p style="color: var(--text-muted);">Memory vault is empty. Run discovery to populate.</p>';
                    return;
                }
                let html = '';
                for (const r of data.repos) {
                    const vClass = r.verdict === 'MUST_READ' ? 'badge-must-read' : (r.verdict === 'WORTH_EXPLORING' ? 'badge-worth-exploring' : 'badge-monitor');
                    html += `
                    <div class="repo-item">
                        <div class="repo-header">
                            <div>
                                <a href="${r.repo_url}" target="_blank" class="repo-name">${r.full_name || r.name}</a>
                                <span style="color: var(--text-muted); font-size: 0.8rem; margin-left: 0.5rem;">${r.stars.toLocaleString()} ★</span>
                            </div>
                            <div class="repo-scores">
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
            if (!url) return alert('Please enter a GitHub URL.');

            const term = document.getElementById('sandbox-term');
            const resultCard = document.getElementById('eval-result-card');
            term.innerText = `[*] Initiating sandbox test for ${url}...\n[*] Submitting job to local worker...\n`;
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
                        <div style="font-size: 0.82rem; color: var(--text-muted);"><b>Technical Critique:</b> ${e.technical_critique}</div>
                    `;
                }
                loadVault();
            } catch (err) {
                term.innerText += `\n[!] Error: ${err.message}`;
                resultCard.innerHTML = '<p style="color: var(--danger);">Failed to complete test.</p>';
            }
        }

        async function triggerScan() {
            if (!confirm('Run full discovery scan now? PRIME-Scout will search GitHub, test candidates in sandbox, and generate today\'s briefing.')) return;
            alert('Discovery scan initiated in background! Check terminal output or refresh briefing in ~30 seconds.');
            await fetch('/api/scan', { method: 'POST' });
            setTimeout(loadLatestBriefing, 5000);
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

        async function cleanSandboxActive() {
            if (!confirm('Clean all cloned sandboxes in scout/sandbox/active/?')) return;
            await fetch('/api/sandbox/clean', { method: 'DELETE' });
            alert('Sandboxes cleaned.');
            loadFs();
        }

        // Init
        loadLatestBriefing();
        loadVault();
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
        "db_path": str(DB_PATH),
        "active_sandboxes": active_sandboxes
    }

@app.get("/api/briefing/latest")
def get_latest_briefing():
    latest_file = REPORTS_DIR / "briefing_latest.md"
    if latest_file.exists():
        return {
            "exists": True,
            "markdown": latest_file.read_text(errors="replace"),
            "file_path": str(latest_file)
        }
    return {"exists": False, "markdown": ""}

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

    # 1. Sandbox execution
    sandbox_res = sandbox_runner.evaluate_repo_sandbox(url, owner, name)
    
    # 2. Fetch README
    readme_text = github_client.fetch_readme(full_name)
    
    # 3. Create repo record
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

    # 4. Evaluate
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
        
        print(f"[*] Processing candidate: {full_name}...")
        repo_id = upsert_repository(c)
        
        sandbox_res = sandbox_runner.evaluate_repo_sandbox(url, owner, name)
        readme = github_client.fetch_readme(full_name)
        
        eval_res = evaluator.evaluate(c, readme, sandbox_res)
        record_evaluation(repo_id, eval_res)
        
        # Merge for reporter
        merged = {**c, **eval_res}
        evaluations.append(merged)

    # Generate daily briefing
    today_str = datetime.now().strftime("%Y-%m-%d")
    reporter.generate_daily_briefing(evaluations, date_str=today_str)
    print(f"[+] Automated discovery pipeline complete. Briefing written for {today_str}.")

def run_server(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
