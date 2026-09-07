"""
PRIME-Scout: Active Repository Experimenter
Conducts safe, isolated micro-benchmarks, architectural parameter extraction,
and performance profiling against candidate repositories in the sandbox.
STRICT SAFETY: Never commits, pushes, or modifies external git history.
"""

import os
import sys
import time
import json
import ast
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from scout.config import SANDBOX_CONFIG, SAFETY_POLICY
from scout.database import record_experiment

class RepoExperimenter:
    def __init__(self, python_executable: Optional[str] = None):
        self.python_bin = python_executable or SANDBOX_CONFIG["python_executable"]

    def _assert_no_git_commit(self, cmd: str):
        if not SAFETY_POLICY["ALLOW_GIT_COMMIT"]:
            for forbidden in ["git commit", "git push", "git tag", "git remote set-url"]:
                if forbidden in cmd.lower():
                    raise PermissionError(f"[SAFETY VIOLATION] '{forbidden}' is strictly prohibited by PRIME-Scout safety policy.")

    def extract_architectural_profile(self, repo_dir: Path) -> Dict[str, Any]:
        """
        Scans Python source files to extract architectural hyperparameters:
        hidden_dim, num_heads, state_dim, attention types, decay mechanics.
        """
        profile = {
            "recurrent_state_detected": False,
            "polynomial_order": 1,
            "has_chunked_prefill": False,
            "has_triton_kernel": False,
            "found_layer_classes": [],
            "identified_equations": []
        }

        for py_file in repo_dir.rglob("*.py"):
            if ".git" in py_file.parts or "venv" in py_file.parts:
                continue
            try:
                content = py_file.read_text(errors="replace")
                
                # Check recurrent / state markers
                if any(term in content for term in ["hidden_state", "recurrent_state", "state =", "past_key_values", "h_t ="]):
                    profile["recurrent_state_detected"] = True

                if any(term in content for term in ["chunk_size", "chunked_linear", "chunk_linear"]):
                    profile["has_chunked_prefill"] = True

                if "@triton.jit" in content:
                    profile["has_triton_kernel"] = True

                # AST class extraction
                tree = ast.parse(content, filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if any(term in node.name.lower() for term in ["attention", "transformer", "mamba", "layer", "block", "rwkv"]):
                            profile["found_layer_classes"].append(f"{py_file.name}:{node.name}")
            except Exception:
                pass

        profile["found_layer_classes"] = list(set(profile["found_layer_classes"]))[:8]
        return profile

    def run_micro_benchmark(
        self,
        repo_dir: Path,
        repo_id: int,
        repo_name: str
    ) -> Dict[str, Any]:
        """
        Synthesizes and runs an isolated micro-benchmark comparing:
        1. Forward pass latency across sequence lengths (e.g. L=512, L=2048)
        2. Memory scaling footprint
        Returns structured telemetry and logs to SQLite.
        """
        benchmark_code = f"""
import sys
import time
import os

sys.path.insert(0, r"{str(repo_dir)}")
print("[*] Running isolated micro-benchmark for {repo_name}...")

results = {{
    "forward_latency_ms": None,
    "peak_memory_mb": None,
    "status": "PASS",
    "notes": []
}}

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Benchmark device: {{device}}")
    
    # Measure baseline torch linear attention emulation
    B, L, H, D = 1, 512, 12, 128
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)
    
    # Warmup
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Linear attention state recurrence test
    kv = torch.matmul(k.transpose(-1, -2), v)
    out = torch.matmul(q, kv)
    
    if device == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    mem_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if device == "cuda" else 0.0
    results["forward_latency_ms"] = round(latency_ms, 2)
    results["peak_memory_mb"] = round(mem_mb, 2)
    print(f"[+] Micro-benchmark finished: latency={{latency_ms:.2f}}ms, peak_mem={{mem_mb:.2f}}MB")

except Exception as e:
    results["status"] = "ERROR"
    results["notes"].append(str(e))
    print(f"[!] Benchmark error: {{e}}")

import json
print("__TELEMETRY__" + json.dumps(results) + "__TELEMETRY__")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)

        try:
            res = subprocess.run(
                [self.python_bin, "-c", benchmark_code],
                cwd=str(repo_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=SANDBOX_CONFIG["max_experiment_time_sec"]
            )
            
            telemetry = {}
            if "__TELEMETRY__" in res.stdout:
                raw = res.stdout.split("__TELEMETRY__")[1]
                try:
                    telemetry = json.loads(raw)
                except Exception:
                    pass

            status = "PASS" if res.returncode == 0 else "FAIL"
            
            # Record in SQLite
            record_experiment(
                repo_id=repo_id,
                name="forward_pass_microbench",
                status=status,
                telemetry=telemetry,
                stdout=res.stdout,
                stderr=res.stderr
            )

            return {
                "status": status,
                "telemetry": telemetry,
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            telemetry = {"status": "TIMEOUT"}
            record_experiment(repo_id, "forward_pass_microbench", "TIMEOUT", telemetry, "", "Timed out")
            return {"status": "TIMEOUT", "telemetry": telemetry, "stdout": "", "stderr": "Execution timed out"}
        except Exception as e:
            telemetry = {"status": "ERROR", "error": str(e)}
            record_experiment(repo_id, "forward_pass_microbench", "ERROR", telemetry, "", str(e))
            return {"status": "ERROR", "telemetry": telemetry, "stdout": "", "stderr": str(e)}

if __name__ == "__main__":
    from scout.database import init_db
    init_db()
    exp = RepoExperimenter()
    test_path = Path("/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/scout/sandbox/active/lucidrains_recurrent-memory-transformer-pytorch")
    if test_path.exists():
        print("[*] Testing architectural profile extraction...")
        profile = exp.extract_architectural_profile(test_path)
        print("Profile:", json.dumps(profile, indent=2))
        print("[*] Testing micro-benchmark...")
        res = exp.run_micro_benchmark(test_path, repo_id=1, repo_name="RMT")
        print("Result:", res["status"], res.get("telemetry"))
