#!/usr/bin/env python3
"""
Falcon Autonomous Researcher Real-Time Monitor
==============================================
Live dashboard tracking:
  - GPU VRAM & System RAM utilization
  - CPU load
  - Active research phase & domain topic
  - Live thought stream snippet (what Falcon-10B is thinking inside <think>)
  - Total theories generated & symbolic formulas verified
"""

import os
import sys
import time
import json
import psutil

STATUS_FILE = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/live_status.json"

def format_bar(used, total, length=24):
    if total <= 0:
        return "[" + " " * length + "]"
    frac = min(1.0, max(0.0, used / total))
    filled = int(round(frac * length))
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {used:.2f}/{total:.2f} GB ({frac * 100:.1f}%)"

def render_dashboard():
    if not os.path.exists(STATUS_FILE):
        print("\n" + "=" * 70)
        print(" [!] Waiting for Falcon Autonomous Researcher to initialize...")
        print(f"     Status file not found yet at {STATUS_FILE}")
        print("=" * 70 + "\n")
        return

    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Error reading status file: {e}")
        return

    status = data.get("status", "UNKNOWN")
    cycle = data.get("cycle", 0)
    domain = data.get("domain", "N/A")
    topic = data.get("topic", "N/A")
    uptime_sec = data.get("uptime_seconds", 0)
    uptime_str = time.strftime("%Hh %Mm %Ss", time.gmtime(uptime_sec))
    pid = data.get("pid", 0)
    
    # Check if process is alive (robust against container PID namespace isolation)
    file_fresh = (time.time() - os.path.getmtime(STATUS_FILE) < 60) if os.path.exists(STATUS_FILE) else False
    is_alive = (psutil.pid_exists(pid) if pid else False) or file_fresh
    alive_badge = "\033[92m● ACTIVE\033[0m" if is_alive else "\033[91m● STOPPED\033[0m"
    
    vram_alloc = data.get("vram_allocated_gb", 0.0)
    vram_total = data.get("vram_total_gb", 16.0)
    ram_used = data.get("ram_used_gb", 0.0)
    ram_total = data.get("ram_total_gb", 128.0)
    cpu_pct = data.get("cpu_percent", 0.0)
    
    theories = data.get("total_theories_generated", 0)
    repos = data.get("total_repos_studied", 0)
    papers = data.get("total_papers_ingested", 0)
    formulas = data.get("total_formulas_verified", 0)
    
    last_theory = data.get("last_theory_title", "None")
    last_formula = data.get("last_verified_formula", "None")
    thought = data.get("live_thought_snippet", "").strip()
    
    # Status color
    status_color = "\033[96m" # Cyan
    if status == "THINKING":
        status_color = "\033[93m" # Yellow
    elif status == "VERIFYING":
        status_color = "\033[92m" # Green
    elif status == "SEARCHING_WEB":
        status_color = "\033[95m" # Magenta
        
    print("\033[2J\033[H", end="") # Clear terminal
    print("\033[1m================================================================================\033[0m")
    print(f"\033[1;34m   🧠 FALCON3-10B (1.58-BIT) AUTONOMOUS RESEARCH ENGINE MONITOR\033[0m")
    print("\033[1m================================================================================\033[0m")
    print(f" Status: {status_color}{status}\033[0m  |  Process: {alive_badge} (PID {pid})  |  Uptime: {uptime_str}")
    print(f" Active Cycle: \033[1m#{cycle}\033[0m  |  Domain: \033[1;37m{domain}\033[0m")
    print(f" Current Topic: {topic}")
    print("-" * 80)
    print("\033[1;36m[HARDWARE & MEMORY TELEMETRY]\033[0m")
    print(f"  GPU VRAM : {format_bar(vram_alloc, vram_total)}")
    print(f"  Host RAM : {format_bar(ram_used, ram_total)}")
    print(f"  CPU Load : {cpu_pct:.1f}% utilization")
    print("-" * 80)
    print("\033[1;36m[RESEARCH PRODUCTIVITY COUNTERS]\033[0m")
    print(f"  • Theories Formulated : \033[1;92m{theories}\033[0m")
    print(f"  • Repos Analyzed      : \033[1m{repos}\033[0m (Phil's local workspace)")
    print(f"  • Papers & Repos Web  : \033[1m{papers}\033[0m (GitHub & CrossRef APIs)")
    print(f"  • PRIME-Net Formulas  : \033[1;92m{formulas}\033[0m verified symbolically")
    print(f"  • Last Formula Passed : {last_formula}")
    print("-" * 80)
    print("\033[1;33m[INSIDE THE MIND OF FALCON-10B (LIVE COGNITIVE STREAM)]\033[0m")
    if thought:
        # Wrap thought lines cleanly
        lines = thought.split("\n")
        recent_lines = lines[-8:] # Show last 8 lines
        for l in recent_lines:
            print(f"  \033[90m>\033[0m {l[:85]}")
    else:
        print("  \033[90m(No live thought stream active at this micro-second...)\033[0m")
    print("\033[1m================================================================================\033[0m")
    print(f" Updated: {data.get('timestamp', time.strftime('%X'))}  |  (Press Ctrl+C to exit monitor)")

def main():
    watch_mode = "--watch" in sys.argv or "-w" in sys.argv
    if watch_mode:
        try:
            while True:
                render_dashboard()
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n[*] Exited monitor.")
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
