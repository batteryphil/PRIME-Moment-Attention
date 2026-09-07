"""
PRIME-Scout: Autonomous Background Research Daemon
Continuously operates in the background:
- Mines repository architectures and frontiers
- Formulates novel mathematical theories & conjectures
- Synthesizes isolated PyTorch micro-benchmarks
- Executes empirical experiments on ROCm/AMD GPU
- Records telemetry, benchmarks, and discoveries into scout_vault.db
- STRICT COMMIT BAN: NEVER COMMITS OR PUSHES TO GIT.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

from scout.config import SAFETY_POLICY
from scout.database import get_autonomous_state, update_autonomous_state
from scout.scientist import AutonomousScientist

def run_autonomous_daemon(interval_sec: float = 60.0):
    print("="*80)
    print("🔬 PRIME-Scout: Continuous Autonomous Science & Research Daemon")
    print(f"🔒 Safety Policy: ALLOW_GIT_COMMIT={SAFETY_POLICY['ALLOW_GIT_COMMIT']} (Commits strictly blocked)")
    print(f"⏱️  Pace: 1 cycle every {interval_sec} seconds (preserves GPU responsiveness)")
    print("="*80)

    scientist = AutonomousScientist()
    update_autonomous_state(is_active=1, current_action="RUNNING", current_target="Starting research loop")

    while True:
        try:
            state = get_autonomous_state()
            if not state.get("is_active", 1):
                print("[*] Autonomous engine is currently paused in database. Sleeping 10s...")
                time.sleep(10)
                continue

            cycle_start = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            print(f"\n[+] [{cycle_start}] Initiating autonomous scientific inquiry...")
            
            result = scientist.run_full_discovery_and_theorize_cycle()
            print(f"[+] Result: {result['title']} -> {result['status']}")
            print(f"    Verdict: {result['conclusion']}")

        except (KeyboardInterrupt, SystemExit):
            print("\n[*] Autonomous research daemon shutting down cleanly.")
            update_autonomous_state(is_active=0, current_action="STOPPED", current_target="User interrupted")
            break
        except Exception as e:
            print(f"[!] Error in autonomous research cycle: {e}")
            update_autonomous_state(current_action="ERROR", current_target=str(e))

        print(f"[*] Cooldown: Sleeping {interval_sec}s before next hypothesis...")
        try:
            time.sleep(interval_sec)
        except (KeyboardInterrupt, SystemExit):
            print("\n[*] Exiting during cooldown.")
            break

if __name__ == "__main__":
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    run_autonomous_daemon(interval_sec=interval)
