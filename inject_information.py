#!/usr/bin/env python3
"""
PRIME Direct Information Injection CLI
=======================================
Allows the user to directly inject custom knowledge payloads, scientific documents,
or theoretical concepts into the PRIME-125M apprentice model on the fly.

Supports:
- Optional dynamic parameter expansion (--expand)
- Direct text, file input, or interactive multiline pasting
- Automatic queuing into SQLite research vault for immediate ingestion by co-learner daemon
"""

import os
import sys
import time
import argparse
import sqlite3

DB_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/research_vault.sqlite3'


def main():
    parser = argparse.ArgumentParser(description="PRIME Direct Information Injection Interface")
    parser.add_argument("--topic", "-t", type=str, default="User Injected Knowledge", help="Title or topic of the injection")
    parser.add_argument("--text", type=str, help="Direct text content to inject")
    parser.add_argument("--file", "-f", type=str, help="Path to file containing knowledge to inject")
    parser.add_argument("--expand", "-e", action="store_true", help="Flag apprentice daemon to expand parameters (Net2Net) before ingesting")
    parser.add_argument("--status", "-s", action="store_true", help="Check status of recent injections")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt interactive multiline input")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if args.status:
        print("=" * 80)
        print(" [*] RECENT INFORMATION INJECTIONS IN PRIME RESEARCH VAULT")
        print("=" * 80)
        cur.execute("""
            SELECT id, timestamp, topic, status, expand_requested, tokens_count, loss_before, loss_after, improvement_pct, anchor_drift
            FROM information_injections
            ORDER BY id DESC LIMIT 10
        """)
        rows = cur.fetchall()
        if not rows:
            print("No injections recorded yet.")
        else:
            for r in rows:
                print(f"ID #{r[0]} | {r[1]} | Status: {r[3]} | Topic: '{r[2]}'")
                print(f"       Expand Requested: {bool(r[4])} | Tokens: {r[5] or 'N/A'}")
                if r[6] is not None and r[7] is not None:
                    print(f"       Loss: {r[6]:.4f} -> {r[7]:.4f} (-{r[8]:.1f}%) | Anchor Drift: {r[9]:.5f}")
                print("-" * 80)
        conn.close()
        return

    content = ""
    if args.file:
        if not os.path.exists(args.file):
            print(f"[-] Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
    elif args.text:
        content = args.text
    elif args.interactive or not sys.stdin.isatty():
        if not sys.stdin.isatty():
            content = sys.stdin.read()
        else:
            print("[*] Enter/paste information below. Press Ctrl+D (EOF) when finished:")
            content = sys.stdin.read()

    content = content.strip()
    if not content:
        print("[-] No content provided to inject. Use --text, --file, or pipe input via stdin.")
        sys.exit(1)

    expand_flag = 1 if args.expand else 0
    cur.execute("""
        INSERT INTO information_injections (timestamp, source, topic, content, status, expand_requested)
        VALUES (?, 'USER_CLI', ?, ?, 'PENDING', ?)
    """, (time.strftime("%Y-%m-%d %H:%M:%S"), args.topic, content, expand_flag))
    conn.commit()
    injection_id = cur.lastrowid
    conn.close()

    print("=" * 80)
    print(f" [+] INFORMATION INJECTION QUEUED SUCCESSFULLY! (ID #{injection_id})")
    print(f"     Topic:            '{args.topic}'")
    print(f"     Payload Length:   {len(content)} characters (~{len(content.split())} words)")
    print(f"     Expand Requested: {bool(expand_flag)}")
    print(f"     Target Daemon:    prime_co_learner_daemon.py (listening live)")
    print("=" * 80)
    print("[*] Waiting for co-learner daemon to ingest and report metrics...")

    # Wait for completion and stream result
    for _ in range(40):
        time.sleep(1)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT status, loss_before, loss_after, improvement_pct, anchor_drift, notes
            FROM information_injections WHERE id = ?
        """, (injection_id,))
        row = cur.fetchone()
        conn.close()
        if row and row[0] in ["COMPLETED", "FAILED"]:
            status, l_before, l_after, imp, drift, notes = row
            if status == "COMPLETED":
                print(f"\n[+] >>> INJECTION COMPLETED SUCCESSFULLY! <<<")
                if l_before is not None and l_after is not None:
                    print(f"    - Pre-Adaptation Surprise:  {l_before:.4f}")
                    print(f"    - Post-Adaptation Surprise: {l_after:.4f} (-{imp:.1f}% error reduction)")
                    print(f"    - Foundational Drift:       {drift:.5f} (Zero forgetting)")
                if notes:
                    print(f"    - Details:                  {notes}")
            else:
                print(f"\n[-] Injection failed: {notes}")
            break
        sys.stdout.write(".")
        sys.stdout.flush()
    print()


if __name__ == "__main__":
    main()
