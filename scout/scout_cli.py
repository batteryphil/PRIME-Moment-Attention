"""
PRIME-Scout: Command Line Interface
Main entry point for autonomous scanning, sandbox testing, report generation, and Web UI.
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from scout.config import UI_CONFIG, REPORTS_DIR
from scout.database import init_db, upsert_repository, record_evaluation, get_recent_evaluations
from scout.github_client import GitHubClient
from scout.sandbox import SandboxRunner
from scout.evaluator import PrimeScoutEvaluator
from scout.reporter import BriefingReporter

def cmd_run(args):
    print("="*80)
    print("🔭 PRIME-Scout: Autonomous Discovery & Evaluation Pipeline")
    print("="*80)
    
    init_db()
    client = GitHubClient()
    sandbox = SandboxRunner()
    evaluator = PrimeScoutEvaluator(mode="heuristic" if args.heuristic else "auto")
    reporter = BriefingReporter()

    print(f"[*] Discovering candidates (limit: {args.limit})...")
    candidates = client.discover_candidates(max_per_query=args.limit_per_query)[:args.limit]
    print(f"[+] Found {len(candidates)} candidates for evaluation.")

    evaluations = []
    for i, c in enumerate(candidates, 1):
        url = c["repo_url"]
        owner = c["owner"]
        name = c["name"]
        full_name = c["full_name"]
        
        print(f"\n[{i}/{len(candidates)}] ----------------------------------------")
        print(f"[*] Target: {full_name} ({c.get('stars', 0):,} ★)")
        print(f"    URL: {url}")
        
        repo_id = upsert_repository(c)
        
        # Sandbox test
        print(f"[*] Running sandbox clone & tests...")
        sandbox_res = sandbox.evaluate_repo_sandbox(url, owner, name)
        print(f"    Sandbox Status: {sandbox_res['status']} ({sandbox_res['duration_sec']}s)")
        
        # Fetch README
        print(f"[*] Fetching README...")
        readme = client.fetch_readme(full_name)
        
        # Local LLM Evaluation
        print(f"[*] Evaluating via PRIME-Scout ({evaluator.mode} mode)...")
        eval_res = evaluator.evaluate(c, readme, sandbox_res)
        record_evaluation(repo_id, eval_res)
        
        print(f"    Verdict: {eval_res['verdict']} | Viability: {eval_res['viability_score']}/100 | Alignment: {eval_res['alignment_score']}/100")
        print(f"    Pitch: {eval_res['executive_pitch'][:100]}...")
        
        merged = {**c, **eval_res}
        evaluations.append(merged)

    # Generate daily briefing
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_path = reporter.generate_daily_briefing(evaluations, date_str=today_str)
    print("\n" + "="*80)
    print(f"🎉 Daily intelligence run complete! Briefing saved to:\n    {report_path}")
    print("="*80)

def cmd_test(args):
    url = args.repo_url.strip()
    parts = [p for p in url.rstrip("/").split("/") if p]
    if len(parts) < 2:
        print(f"[!] Invalid GitHub repository URL: {url}")
        sys.exit(1)
        
    owner = parts[-2]
    name = parts[-1]
    full_name = f"{owner}/{name}"

    print(f"[*] On-Demand Testing: {full_name}")
    init_db()
    sandbox = SandboxRunner()
    client = GitHubClient()
    evaluator = PrimeScoutEvaluator(mode="heuristic" if args.heuristic else "auto")

    # 1. Sandbox test
    print(f"[*] Executing sandbox verification...")
    sandbox_res = sandbox.evaluate_repo_sandbox(url, owner, name)
    print(f"[+] Sandbox Status: {sandbox_res['status']} ({sandbox_res['duration_sec']}s)")
    print(sandbox_res['log'])

    # 2. Fetch README
    readme = client.fetch_readme(full_name)

    # 3. Evaluate
    repo_meta = {
        "repo_url": url,
        "name": name,
        "full_name": full_name,
        "owner": owner,
        "stars": 0,
        "language": "Python",
        "topics": []
    }
    repo_id = upsert_repository(repo_meta)
    
    print(f"[*] Running PRIME-Scout evaluation...")
    eval_res = evaluator.evaluate(repo_meta, readme, sandbox_res)
    record_evaluation(repo_id, eval_res)

    print("\n" + "="*80)
    print(f"EVALUATION CARD: {full_name}")
    print("="*80)
    print(f"Verdict:           {eval_res['verdict']}")
    print(f"Viability Score:   {eval_res['viability_score']}/100")
    print(f"Alignment Score:   {eval_res['alignment_score']}/100")
    print(f"Sandbox Status:    {eval_res['sandbox_status']}")
    print(f"\nExecutive Pitch:\n{eval_res['executive_pitch']}")
    print(f"\nTechnical Critique:\n{eval_res['technical_critique']}")
    print(f"\nPRIME Synergy Notes:\n{eval_res['synergy_notes']}")
    print("="*80)

def cmd_ui(args):
    print(f"[*] Launching PRIME-Scout Web Dashboard on http://{args.host}:{args.port}...")
    from scout.web_ui import run_server
    run_server(host=args.host, port=args.port)

def cmd_report(args):
    latest = REPORTS_DIR / "briefing_latest.md"
    if not latest.exists():
        print("[!] No briefing found. Run `python -m scout.scout_cli run` first.")
        return
    print(latest.read_text(errors="replace"))

def main():
    parser = argparse.ArgumentParser(description="PRIME-Scout: Autonomous Local Repository Intelligence Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run
    p_run = subparsers.add_parser("run", help="Run full discovery, sandbox testing, and daily briefing")
    p_run.add_argument("--limit", type=int, default=4, help="Maximum candidates to evaluate")
    p_run.add_argument("--limit-per-query", type=int, default=2, help="Candidates per search query")
    p_run.add_argument("--heuristic", action="store_true", help="Use fast heuristic evaluation without loading LLM weights")
    p_run.set_defaults(func=cmd_run)

    # Test
    p_test = subparsers.add_parser("test", help="Test a single repository in the sandbox")
    p_test.add_argument("repo_url", type=str, help="GitHub repository URL")
    p_test.add_argument("--heuristic", action="store_true", help="Use fast heuristic evaluation")
    p_test.set_defaults(func=cmd_test)

    # UI
    p_ui = subparsers.add_parser("ui", help="Launch the interactive web dashboard")
    p_ui.add_argument("--host", type=str, default=UI_CONFIG["host"], help="Host address")
    p_ui.add_argument("--port", type=int, default=UI_CONFIG["port"], help="Port number")
    p_ui.set_defaults(func=cmd_ui)

    # Report
    p_rep = subparsers.add_parser("report", help="View the latest briefing in terminal")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
