# 🔭 PRIME-Scout: Autonomous Local AI Repository Intelligence Agent

**PRIME-Scout** is a specialized, autonomous local research agent designed to run on your workstation. It actively scans GitHub for new and trending repositories aligning with your research interests (linear attention, ROCm/CUDA kernels, sub-quadratic transformers, Triton compilers, continuous memory, and decision transformers).

PRIME-Scout clones candidate repositories into an **isolated local execution sandbox**, audits hardware kernels and Python syntax, smoke-tests imports, evaluates technical viability using your **local GPU LLM (Qwen2.5-Coder-1.5B)**, and synthesizes a high-signal **Daily Intelligence Briefing** alongside an **interactive web dashboard**.

---

## 📁 File System Layout

The directory structure is organized for transparency and ease of inspection:

```
PRIME-Moment-Attention/scout/
├── config.py                 # Target research interests, keywords, thresholds
├── github_client.py          # GitHub API search & code fetcher
├── database.py               # SQLite persistence vault (repositories & evaluations)
├── sandbox.py                # Isolated execution sandbox (git clone, AST audit, smoke tests)
├── evaluator.py              # Local GPU LLM evaluation engine (Qwen2.5-Coder-1.5B)
├── reporter.py               # Markdown daily briefing generator
├── web_ui.py                 # Modern dark-mode FastAPI web dashboard
├── scout_cli.py              # CLI entry point
│
├── reports/                  # [READING VAULT] Daily markdown briefings
│   ├── briefing_latest.md    # Always points to today's newest briefing
│   └── briefing_YYYY-MM-DD.md
├── vault/                    # [PERSISTENCE]
│   └── scout_vault.db        # SQLite database of all evaluated repos & scores
└── sandbox/
    └── active/               # [EXECUTION SANDBOX] Isolated repo checkouts for testing
```

---

## 🚀 Quickstart

### 1. Launch the Interactive Web Dashboard
```bash
python -m scout.scout_cli ui --port 7860
```
Open your browser at `http://localhost:7860`:
* **📰 Daily Briefing**: Read today's curated digest with score badges and deep dives.
* **🧪 Sandbox & Deep-Dive**: Paste *any* GitHub repository URL to trigger an on-demand clone, sandbox test, and LLM evaluation.
* **🏛️ Memory Vault**: Search and filter through all past evaluations.
* **📁 File System**: View and clean active sandbox checkouts.

### 2. Run Today's Discovery Pipeline Manually
```bash
python -m scout.scout_cli run --limit 5
```
Scans GitHub, clones candidates, executes sandbox tests, runs local model evaluations, and writes `scout/reports/briefing_YYYY-MM-DD.md`.

### 3. Test a Single Repository in the Sandbox
```bash
python -m scout.scout_cli test https://github.com/sustcsonglin/flash-linear-attention
```

### 4. Read Today's Briefing in Terminal
```bash
python -m scout.scout_cli report
```

---

## ⏰ Automated Daily Scheduling (Cron)

To have PRIME-Scout automatically run once a day (e.g. at 6:00 AM) and prepare your briefing before you start work:

```bash
crontab -e
```
Add the following entry:
```cron
0 6 * * * /home/phil/.gemini/antigravity/scratch/venv/bin/python /home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/scout/scout_cli.py run --limit 5 >> /home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/scout/vault/scout_cron.log 2>&1
```

---

## 🛡️ Sandbox Safety Features
* **Isolated subprocesses**: All smoke tests execute in isolated subprocesses with timeouts (30–45s).
* **AST Validation**: Python code is parsed into Abstract Syntax Trees before execution to catch syntax corruptions.
* **Disk Quota Control**: Clones use `--depth 1` to minimize bandwidth and storage. Active sandboxes can be purged with a single button in the Web UI or via `rm -rf scout/sandbox/active/*`.
