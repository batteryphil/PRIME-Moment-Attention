#!/usr/bin/env python3
"""
PRIME Moment Attention: Unified Empirical Reproduction Runner
=============================================================

Reproduces all empirical experiments documented in the scientific dossier:
  - Experiment A: Mathematical Taylor convergence and diagonal tensor error (D in {2, 4, 8})
  - Experiment B: 1-Million token context scaling benchmark (latency & memory)
  - Experiment C: Needle information survival curve vs unweighted ELU+1 linear attention
  - Experiment D: Multi-needle independent query retrieval across 2,048 tokens
  - Experiment E: State overwrite & belief revision dynamics
  - Experiment E2: Sequential multi-stage contradiction dynamics (BLUE -> RED -> GREEN -> YELLOW)
  - Experiment F: Differentiable multiscale timescale learning via backpropagation
  - Experiment F2: Three-condition timescale convergence (Fixed vs Learnable vs Random Init)
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
    "e2": ("exp_e2_sequential_contradiction.py", "Experiment E2: Sequential Multi-Stage Contradiction Dynamics"),
    "f": ("exp_f_learnable_timescales.py", "Experiment F: Learnable Timescales via Backpropagation"),
    "f2": ("exp_f2_timescale_convergence.py", "Experiment F2: Three-Condition Timescale Convergence"),
    "g": ("exp_g_long_rollout_stability.py", "Experiment G: 2048-Step Rollout Numerical Stability"),
    "h": ("exp_h_prime_selective.py", "Experiment H: 100% PRIME-Selective Distillation Engine (1.5B)"),
    "i": ("exp_i_deep_hook_telemetry.py", "Experiment I: Deep Hook Telemetry & Multi-Battery Diagnostic Harness"),
    "j": ("exp_j_domain1_vision_clip_dinov2.py", "Experiment J: Domain 1 - Multimodal Vision (CLIP ViT-B/32 & DINOv2)"),
    "k": ("exp_k_domain2_audio_whisper.py", "Experiment K: Domain 2 - Speech Recognition (OpenAI Whisper-tiny)"),
    "l": ("exp_l_domain3_chronos_dynamics.py", "Experiment L: Domain 3 - Physical Dynamics (Amazon Chronos-T5-mini)"),
    "m": ("exp_m_domain4_chemberta_molecules.py", "Experiment M: Domain 4 - Cheminformatics (ChemBERTa-77M-MTR)"),
    "n": ("exp_n_domain5_decision_transformer.py", "Experiment N: Domain 5 - Offline RL & Continuous Control (Decision Transformer)"),
    "o": ("exp_o_protein_folding_esm2.py", "Experiment O: Structural Biology - Protein Folding (ESM-2 150M)"),
    "p": ("exp_p_diffusion_llm_mdlm.py", "Experiment P: Discrete Diffusion LLMs (MDLM NeurIPS 2024)"),
    "q": ("exp_q_image_diffusion_sd15.py", "Experiment Q: 2D Spatial Image Diffusion (Stable Diffusion 1.5)"),
    "r": ("exp_r_video_diffusion_animatediff.py", "Experiment R: Temporal Video Diffusion (AnimateDiff v1.5-2)"),
    "s": ("exp_s_trunk_matrix_decision_transformer.py", "Experiment S: Decision Transformer Trunk Substitution Matrix (6x3 Grid)"),
    "t": ("exp_t_trunk_matrix_chronos.py", "Experiment T: Amazon Chronos Trunk Substitution Matrix (7x3 Grid)"),
    "u": ("exp_u_dt_brutal_controls.py", "Experiment U: Decision Transformer Brutal Controls (Random, Zero, Shuffled)"),
    "v": ("exp_v_chronos_regularization_sweep.py", "Experiment V: Chronos Layer 1 Regularization Sweep across 10 Physical Regimes"),
}

def run_experiment(exp_key: str):
    key = exp_key.lower()
    if key not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_key}. Available: {list(EXPERIMENTS.keys())}")
        return False
    
    script_name, title = EXPERIMENTS[key]
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
    parser.add_argument("--exp", type=str, help="Run a specific experiment (a, b, c, d, e, e2, f, f2, g)")
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Empirical Reproduction Experiments:")
        for k, (s, desc) in EXPERIMENTS.items():
            print(f"  --exp {k:<3}: {desc}")
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
