"""
PRIME-Scout: Command Line Interface
Main entry point for autonomous scanning, sandbox experimentation, report generation,
interactive web UI, and two-way terminal chat.
"""

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from scout.config import UI_CONFIG, REPORTS_DIR, SAFETY_POLICY
from scout.database import (
    init_db, upsert_repository, record_evaluation, get_recent_evaluations,
    get_pending_questions, answer_agent_question, get_evaluated_repo_urls
)
from scout.github_client import GitHubClient
from scout.sandbox import SandboxRunner
from scout.evaluator import PrimeScoutEvaluator
from scout.reporter import BriefingReporter

def cmd_run(args):
    print("="*80)
    print("🔭 PRIME-Scout: Autonomous Discovery & Experimentation Pipeline")
    print(f"🔒 Safety Policy: ALLOW_GIT_COMMIT={SAFETY_POLICY['ALLOW_GIT_COMMIT']} (Commits Blocked)")
    print("="*80)
    
    init_db()
    client = GitHubClient()
    sandbox = SandboxRunner()
    evaluator = PrimeScoutEvaluator(mode="heuristic" if args.heuristic else "auto")
    reporter = BriefingReporter()

    evaluated_urls = set() if getattr(args, "force", False) else get_evaluated_repo_urls()
    print(f"[*] Discovering fresh candidates (excluding {len(evaluated_urls)} already-evaluated repos, limit: {args.limit})...")
    candidates = client.discover_candidates(max_per_query=args.limit_per_query, exclude_urls=evaluated_urls)[:args.limit]
    if not candidates:
        print("[!] All candidate repositories in current query pool have already been evaluated.")
        print("    Use `--force` to re-evaluate or add new search topics in scout/config.py.")
        return

    print(f"[+] Found {len(candidates)} new candidate(s) for evaluation.")

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
        
        print(f"[*] Running sandbox clone & tests (Never-Commit active)...")
        sandbox_res = sandbox.evaluate_repo_sandbox(url, owner, name, repo_id=repo_id)
        print(f"    Sandbox Status: {sandbox_res['status']} ({sandbox_res['duration_sec']}s)")
        
        print(f"[*] Fetching README...")
        readme = client.fetch_readme(full_name)
        
        print(f"[*] Evaluating via PRIME-Scout ({evaluator.mode} mode)...")
        eval_res = evaluator.evaluate(c, readme, sandbox_res)
        record_evaluation(repo_id, eval_res)
        
        print(f"    Verdict: {eval_res['verdict']} | Viability: {eval_res['viability_score']}/100 | Alignment: {eval_res['alignment_score']}/100")
        if eval_res.get("question_for_phil"):
            print(f"    ❓ Agent Question: {eval_res['question_for_phil']}")
        
        merged = {**c, **eval_res}
        evaluations.append(merged)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

    print(f"[*] On-Demand Sandbox Experiment: {full_name}")
    print(f"🔒 Safety Policy: ALLOW_GIT_COMMIT={SAFETY_POLICY['ALLOW_GIT_COMMIT']}")
    init_db()
    sandbox = SandboxRunner()
    client = GitHubClient()
    evaluator = PrimeScoutEvaluator(mode="heuristic" if args.heuristic else "auto")

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

    print(f"[*] Executing sandbox verification & micro-benchmarks...")
    sandbox_res = sandbox.evaluate_repo_sandbox(url, owner, name, repo_id=repo_id)
    print(f"[+] Sandbox Status: {sandbox_res['status']} ({sandbox_res['duration_sec']}s)")
    print(sandbox_res['log'])

    readme = client.fetch_readme(full_name)
    eval_res = evaluator.evaluate(repo_meta, readme, sandbox_res)
    record_evaluation(repo_id, eval_res)

    print("\n" + "="*80)
    print(f"EVALUATION & EXPERIMENT CARD: {full_name}")
    print("="*80)
    print(f"Verdict:           {eval_res['verdict']}")
    print(f"Viability Score:   {eval_res['viability_score']}/100")
    print(f"Alignment Score:   {eval_res['alignment_score']}/100")
    print(f"Sandbox Status:    {eval_res['sandbox_status']}")
    print(f"\nExecutive Pitch:\n{eval_res['executive_pitch']}")
    print(f"\nTechnical Critique:\n{eval_res['technical_critique']}")
    print(f"\nPRIME Synergy Notes:\n{eval_res['synergy_notes']}")
    if eval_res.get("question_for_phil"):
        print(f"\n❓ Question for Phil:\n{eval_res['question_for_phil']}")
    print("="*80)

def cmd_chat(args):
    print("="*80)
    print("💬 PRIME-Scout: Two-Way Research Terminal Dialogue")
    print("Local Qwen2.5-Coder model on ROCm. Type 'exit' or 'quit' to end.")
    print("="*80)
    
    init_db()
    evaluator = PrimeScoutEvaluator(mode="heuristic" if args.heuristic else "auto")
    
    # Check pending questions
    pending = get_pending_questions()
    if pending:
        print(f"\n[!] PRIME-Scout has {len(pending)} pending question(s) for you:")
        for q in pending[:3]:
            print(f"    - [ID: {q['id']} | {q.get('repo_name', 'Repo')}]: {q['question']}")
        print("    (You can answer them anytime in this chat)\n")

    while True:
        try:
            user_msg = input("\nPhil > ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ["exit", "quit", "q"]:
                print("[*] Ending dialogue session. Goodbye!")
                break
                
            reply = evaluator.chat(user_msg)
            print(f"\nPRIME-Scout > {reply}")
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting.")
            break

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

def cmd_daemon(args):
    import time
    print("="*80)
    print("🤖 PRIME-Scout: Continuous 24/7 Autonomous Research Daemon")
    print(f"🔒 Safety Policy: ALLOW_GIT_COMMIT={SAFETY_POLICY['ALLOW_GIT_COMMIT']} (Commits Blocked)")
    print(f"⏱️  Scan Frequency: Every {args.interval_hours} hour(s)")
    print("="*80)
    
    cycle_num = 1
    while True:
        try:
            print(f"\n[+] [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Initiating Cycle #{cycle_num}...")
            cmd_run(args)
        except Exception as e:
            print(f"[!] Cycle #{cycle_num} encountered error: {e}")
            
        cycle_num += 1
        sleep_sec = max(60, int(args.interval_hours * 3600))
        print(f"\n[*] Cycle complete. Next discovery pass in {args.interval_hours} hour(s) ({sleep_sec}s). Sleeping...")
        try:
            time.sleep(sleep_sec)
        except (KeyboardInterrupt, SystemExit):
            print("\n[*] PRIME-Scout daemon gracefully stopped.")
            break

def cmd_lab(args):
    from scout.scientist import AutonomousScientist
    scientist = AutonomousScientist()
    if args.once:
        print("[*] Running single autonomous scientific inquiry cycle...")
        res = scientist.run_full_discovery_and_theorize_cycle()
        print(f"\n[+] Completed: {res['title']}")
        print(f"    Verdict: {res['status']}")
        print(f"    Conclusion: {res['conclusion']}")
    else:
        from scout.autonomous_daemon import run_autonomous_daemon
        run_autonomous_daemon(interval_sec=args.interval)

def cmd_narrate(args):
    from scout.audio_narrator import narrate_chapter_file
    print("=" * 80)
    print(f"🎙️ PRIME Sovereign Audio Narrator: '{args.slug}' (Chapter {args.chapter})")
    print("=" * 80)
    res = narrate_chapter_file(novel_slug=args.slug, chapter_num=args.chapter, device=args.device)
    print("\n" + "=" * 80)
    print(f"🎉 Narration Complete!")
    print(f"    Duration: {res['duration_min']:.2f} min ({res['duration_sec']:.1f}s)")
    print(f"    Master MP3: {res['mp3_path']}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="PRIME-Scout: Autonomous Local Repository Intelligence Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Narrate
    p_narrate = subparsers.add_parser("narrate", help="Narrate a novel chapter into an ACX-compliant multi-voice audiobook")
    p_narrate.add_argument("--slug", type=str, default="beyond_the_event_horizon", help="Novel directory slug")
    p_narrate.add_argument("--chapter", type=int, default=1, help="Chapter number (default 1)")
    p_narrate.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    p_narrate.set_defaults(func=cmd_narrate)

    # Lab / Scientist
    p_lab = subparsers.add_parser("lab", help="Autonomous theory formulation and test synthesis engine")
    p_lab.add_argument("--once", action="store_true", help="Run a single scientific cycle and exit")
    p_lab.add_argument("--interval", type=float, default=60.0, help="Seconds between cycles in daemon mode (default 60s)")
    p_lab.set_defaults(func=cmd_lab)

    # Run
    p_run = subparsers.add_parser("run", help="Run full discovery, sandbox experimentation, and daily briefing")
    p_run.add_argument("--limit", type=int, default=4, help="Maximum candidates to evaluate")
    p_run.add_argument("--limit-per-query", type=int, default=2, help="Candidates per search query")
    p_run.add_argument("--heuristic", action="store_true", help="Use fast heuristic evaluation without loading LLM weights")
    p_run.add_argument("--force", action="store_true", help="Force re-evaluation of previously audited repositories")
    p_run.set_defaults(func=cmd_run)

    # Daemon
    p_daemon = subparsers.add_parser("daemon", help="Run continuously in background, discovering new repos every N hours")
    p_daemon.add_argument("--interval-hours", type=float, default=6.0, help="Hours between discovery passes (e.g. 6.0)")
    p_daemon.add_argument("--limit", type=int, default=4, help="Maximum candidates to evaluate per cycle")
    p_daemon.add_argument("--limit-per-query", type=int, default=2, help="Candidates per search query")
    p_daemon.add_argument("--heuristic", action="store_true", help="Use fast heuristic evaluation")
    p_daemon.add_argument("--force", action="store_true", help="Force re-evaluation")
    p_daemon.set_defaults(func=cmd_daemon)

    # Test
    p_test = subparsers.add_parser("test", help="Test and experiment on a single repository in the sandbox")
    p_test.add_argument("repo_url", type=str, help="GitHub repository URL")
    p_test.add_argument("--heuristic", action="store_true", help="Use fast heuristic evaluation")
    p_test.set_defaults(func=cmd_test)

    # Chat
    p_chat = subparsers.add_parser("chat", help="Start an interactive terminal conversation with PRIME-Scout")
    p_chat.add_argument("--heuristic", action="store_true", help="Use offline heuristic mode")
    p_chat.set_defaults(func=cmd_chat)

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
