#!/usr/bin/env python3
"""
PRIME Moment Attention: Unified Empirical Reproduction Runner
=============================================================

Reproduces all empirical experiments documented in the scientific dossier:
  - Experiment A: Mathematical Taylor convergence and tensor formulation (D in {2, 4, 8})
  - Experiment B: 1-Million token context scaling benchmark (latency & memory)
  - Experiment C: Needle information survival curve vs unweighted ELU+1 linear attention
  - Experiment D: Multi-needle independent query retrieval across 2,048 tokens
  - Experiment E: State overwrite and belief revision dynamics
  - Experiment F: Differentiable multiscale timescale learning via backpropagation
  - Experiment G: Long-rollout numerical stability across 2,048 autoregressive steps

Usage:
  python reproduce_all.py --all
  python reproduce_all.py --exp a
  python reproduce_all.py --exp b
  python reproduce_all.py --list
"""

import sys
import os
import argparse
import subprocess

EXPERIMENTS = {
    "a": ("exp_a_taylor_convergence.py", "Experiment A: Mathematical Taylor Convergence (D in {2, 4, 8})"),
    "b": ("exp_b_1m_context_scaling.py", "Experiment B: 1-Million Token Scaling Benchmark"),
    "c": ("exp_c_needle_retention.py", "Experiment C: Needle Information Retention vs ELU+1"),
    "d": ("exp_d_multi_needle_recall.py", "Experiment D: Multi-Needle Associative Recall"),
    "e": ("exp_e_state_overwrite.py", "Experiment E: State Overwrite & Contradiction Dynamics"),
    "f": ("exp_f_learnable_timescales.py", "Experiment F: Learnable Timescales via Backpropagation"),
    "g": ("exp_g_long_rollout_stability.py", "Experiment G: 2048-Step Rollout Numerical Stability"),
}

def run_experiment(exp_key: str):
    if exp_key.lower() not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_key}. Available: {list(EXPERIMENTS.keys())}")
        return False
    
    script_name, title = EXPERIMENTS[exp_key.lower()]
    script_path = os.path.join(os.path.dirname(__file__), "experiments", script_name)
    
    print("\n" + "=" * 80)
    print(f"RUNNING: {title}")
    print(f"SCRIPT:  {script_path}")
    print("=" * 80 + "\n")
    
    res = subprocess.run([sys.executable, script_path])
    return res.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Reproduce PRIME Moment Attention Empirical Benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all empirical experiments sequentially")
    parser.add_argument("--exp", type=str, help="Run a specific experiment (a, b, c, d, e, f, g)")
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Empirical Reproduction Experiments:")
        for k, (s, desc) in EXPERIMENTS.items():
            print(f"  --exp {k}: {desc}")
        print()
        return
        
    if args.all:
        for k in sorted(EXPERIMENTS.keys()):
            success = run_experiment(k)
            if not success:
                print(f"[!] Experiment {k} encountered an error.")
        print("\n[+] All requested benchmarks completed.")
    elif args.exp:
        run_experiment(args.exp)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
