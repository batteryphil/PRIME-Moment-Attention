"""
PRIME-Scout: Configuration and Research Profile
Defines research interests, search queries, sandbox limits, and directory paths.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCOUT_DIR = BASE_DIR / "scout"
REPORTS_DIR = SCOUT_DIR / "reports"
VAULT_DIR = SCOUT_DIR / "vault"
SANDBOX_DIR = SCOUT_DIR / "sandbox" / "active"
STATIC_DIR = SCOUT_DIR / "static"
DB_PATH = VAULT_DIR / "scout_vault.db"

# Ensure directories exist
for p in [REPORTS_DIR, VAULT_DIR, SANDBOX_DIR, STATIC_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# User Research Profile (batteryphil / PRIME-Moment-Attention)
RESEARCH_PROFILE = {
    "user_repos": [
        "https://github.com/batteryphil/PRIME-Moment-Attention",
    ],
    "primary_topics": [
        "linear-attention",
        "moment-attention",
        "sub-quadratic-transformers",
        "rocm",
        "hip",
        "triton-kernel",
        "kv-cache-compression",
        "continuous-memory",
        "recurrent-transformer",
        "decision-transformer",
        "generative-video",
    ],
    "search_queries": [
        "linear attention language model",
        "triton kernel rocm attention",
        "constant memory transformer long context",
        "state space model recurrent attention",
        "recurrent memory decision transformer",
        "kv cache compression efficient attention",
        "fast causal attention amd rocm",
    ],
    "relevance_keywords": [
        "linear attention", "prime", "moment", "recurrent", "triton",
        "rocm", "hip", "subquadratic", "mamba", "rwkv", "retnet",
        "kv cache", "chunked prefill", "kernel", "decision transformer",
        "continuous control", "long context", "needle in a haystack",
    ]
}

# Sandbox Execution Limits
SANDBOX_CONFIG = {
    "clone_depth": 1,
    "max_clone_time_sec": 30,
    "max_test_time_sec": 45,
    "max_repo_size_mb": 150,
    "python_executable": str(Path(os.environ.get("VIRTUAL_ENV", "/home/phil/.gemini/antigravity/scratch/venv")) / "bin" / "python"),
}

# LLM Inference Configuration
MODEL_CONFIG = {
    "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "use_stage7_hybrid": True,
    "device": "cuda" if os.environ.get("ROCM_PATH") or os.path.exists("/dev/kfd") else "cpu",
    "max_context_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.9,
}

# Web UI Configuration
UI_CONFIG = {
    "host": "0.0.0.0",
    "port": 7860,
    "title": "PRIME-Scout: Local Repository Intelligence Agent",
}
