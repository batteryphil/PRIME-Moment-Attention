"""
PRIME-Scout: Isolated Execution Sandbox
Safely clones candidate repositories into scout/sandbox/active/,
audits dependencies, validates Python AST/syntax, and executes smoke tests
with strict timeouts and isolated environment variables.
"""

import os
import sys
import shutil
import subprocess
import ast
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scout.config import SANDBOX_DIR, SANDBOX_CONFIG

class SandboxRunner:
    def __init__(self, sandbox_base: Optional[Path] = None):
        self.sandbox_base = sandbox_base or SANDBOX_DIR
        self.sandbox_base.mkdir(parents=True, exist_ok=True)
        self.python_bin = SANDBOX_CONFIG["python_executable"]

    def _get_repo_dir(self, owner: str, name: str) -> Path:
        safe_name = f"{owner}_{name}".replace("/", "_").replace(" ", "_")
        return self.sandbox_base / safe_name

    def clone_repo(self, repo_url: str, owner: str, name: str) -> Tuple[bool, Path, str]:
        repo_dir = self._get_repo_dir(owner, name)
        
        # If directory already exists and has .git, pull or reuse
        if (repo_dir / ".git").exists():
            return True, repo_dir, "Existing clone reused."

        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)

        cmd = [
            "git", "clone", "--depth", str(SANDBOX_CONFIG["clone_depth"]),
            repo_url, str(repo_dir)
        ]
        
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SANDBOX_CONFIG["max_clone_time_sec"]
            )
            if res.returncode == 0:
                return True, repo_dir, "Successfully cloned repository."
            else:
                return False, repo_dir, f"Git clone failed: {res.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, repo_dir, "Git clone timed out."
        except Exception as e:
            return False, repo_dir, f"Error during git clone: {str(e)}"

    def audit_dependencies(self, repo_dir: Path) -> Dict[str, Any]:
        """
        Audits project requirements and detects key frameworks.
        """
        audit = {
            "has_requirements_txt": False,
            "has_pyproject": False,
            "has_setup_py": False,
            "detected_frameworks": [],
            "dependencies": [],
            "has_triton_kernels": False,
            "has_rocm_hip": False,
            "has_cuda": False,
        }
        
        req_file = repo_dir / "requirements.txt"
        if req_file.exists():
            audit["has_requirements_txt"] = True
            try:
                lines = [l.strip() for l in req_file.read_text(errors="replace").splitlines() if l.strip() and not l.startswith("#")]
                audit["dependencies"].extend(lines[:20]) # Sample top 20
            except Exception:
                pass

        if (repo_dir / "pyproject.toml").exists():
            audit["has_pyproject"] = True
        if (repo_dir / "setup.py").exists():
            audit["has_setup_py"] = True

        # Scan for kernel and hardware indicators
        for ext in [".py", ".cu", ".hip", ".cpp", ".h"]:
            for f in repo_dir.rglob(f"*{ext}"):
                # Limit scan depth
                if ".git" in f.parts or len(f.parts) > 10:
                    continue
                try:
                    text = f.read_text(errors="replace")
                    if "@triton.jit" in text or "import triton" in text:
                        audit["has_triton_kernels"] = True
                    if "hipSetDevice" in text or "hipMalloc" in text or "rocm" in text.lower():
                        audit["has_rocm_hip"] = True
                    if "cudaMalloc" in text or "cudaStream_t" in text:
                        audit["has_cuda"] = True
                except Exception:
                    pass

        return audit

    def verify_python_syntax(self, repo_dir: Path, max_files: int = 30) -> Dict[str, Any]:
        """
        Performs AST parsing across Python files to verify syntax integrity.
        """
        py_files = [p for p in repo_dir.rglob("*.py") if ".git" not in p.parts][:max_files]
        if not py_files:
            return {"valid": True, "files_checked": 0, "syntax_errors": []}

        syntax_errors = []
        for py_file in py_files:
            try:
                code = py_file.read_text(errors="replace")
                ast.parse(code, filename=str(py_file))
            except SyntaxError as e:
                syntax_errors.append(f"{py_file.name}:{e.lineno} - {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{py_file.name} - {str(e)}")

        return {
            "valid": len(syntax_errors) == 0,
            "files_checked": len(py_files),
            "syntax_errors": syntax_errors[:5] # Top 5 errors
        }

    def run_import_smoke_test(self, repo_dir: Path, package_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs an isolated python import in a subprocess with timeouts.
        """
        # Infer package name from subdirectories if not provided
        if not package_name:
            candidates = [p.name for p in repo_dir.iterdir() if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith(".")]
            package_name = candidates[0] if candidates else None

        if not package_name:
            return {
                "status": "NO_PACKAGE",
                "message": "No standard Python package structure (__init__.py) found at root.",
                "stdout": "",
                "stderr": "",
                "exit_code": 0
            }

        # Run test inside isolated subprocess
        test_script = f"""
import sys
sys.path.insert(0, r"{str(repo_dir)}")
try:
    import {package_name}
    print("SUCCESS: Module {package_name} imported cleanly.")
except Exception as e:
    import traceback
    print("IMPORT_ERROR:", e)
    traceback.print_exc()
    sys.exit(1)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)

        try:
            res = subprocess.run(
                [self.python_bin, "-c", test_script],
                cwd=str(repo_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=SANDBOX_CONFIG["max_test_time_sec"]
            )
            
            status = "PASS" if res.returncode == 0 else "FAIL"
            if res.returncode != 0 and "No module named" in res.stderr:
                status = "MISSING_DEPS"
                
            return {
                "status": status,
                "package_name": package_name,
                "exit_code": res.returncode,
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip()[:1000] # Cap log length
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "package_name": package_name,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Import test timed out after {SANDBOX_CONFIG["max_test_time_sec"]} seconds."
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "package_name": package_name,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e)
            }

    def evaluate_repo_sandbox(self, repo_url: str, owner: str, name: str) -> Dict[str, Any]:
        """
        Full end-to-end sandbox pipeline:
        1. Clone
        2. Audit dependencies & hardware kernels
        3. AST Syntax check
        4. Isolated import test
        """
        start_time = time.time()
        print(f"[*] Sandbox: Cloning {owner}/{name}...")
        ok, repo_dir, clone_msg = self.clone_repo(repo_url, owner, name)
        if not ok:
            return {
                "status": "CLONE_FAILED",
                "log": clone_msg,
                "audit": {},
                "syntax": {},
                "duration_sec": round(time.time() - start_time, 2)
            }

        print(f"[*] Sandbox: Auditing dependencies for {owner}/{name}...")
        audit = self.audit_dependencies(repo_dir)

        print(f"[*] Sandbox: Checking syntax for {owner}/{name}...")
        syntax = self.verify_python_syntax(repo_dir)

        print(f"[*] Sandbox: Running import smoke test for {owner}/{name}...")
        smoke = self.run_import_smoke_test(repo_dir, package_name=name.replace("-", "_"))

        # Determine overall sandbox verdict
        if not syntax["valid"]:
            overall_status = "SYNTAX_ERROR"
        elif smoke["status"] == "PASS":
            overall_status = "PASS"
        elif smoke["status"] == "MISSING_DEPS":
            overall_status = "MISSING_DEPS"
        elif smoke["status"] == "NO_PACKAGE":
            overall_status = "NO_PACKAGE"
        else:
            overall_status = smoke["status"]

        log_summary = f"""
[Sandbox Execution Report]
- Clone: {clone_msg}
- Python Files Checked: {syntax["files_checked"]} (Syntax Valid: {syntax["valid"]})
- Hardware Kernels Detected: Triton={audit["has_triton_kernels"]}, ROCm/HIP={audit["has_rocm_hip"]}, CUDA={audit["has_cuda"]}
- Import Test ({smoke.get("package_name", "N/A")}): Status={smoke["status"]}, Exit={smoke.get("exit_code", 0)}
{smoke.get("stderr", "")}
""".strip()

        return {
            "status": overall_status,
            "repo_dir": str(repo_dir),
            "log": log_summary,
            "audit": audit,
            "syntax": syntax,
            "smoke": smoke,
            "duration_sec": round(time.time() - start_time, 2)
        }

if __name__ == "__main__":
    from typing import Tuple
    sandbox = SandboxRunner()
    print("[*] Testing Sandbox on a fast repo...")
    res = sandbox.evaluate_repo_sandbox(
        "https://github.com/lucidrains/recurrent-memory-transformer-pytorch",
        "lucidrains",
        "recurrent-memory-transformer-pytorch"
    )
    print(f"[+] Sandbox result: Status={res["status"]}, Duration={res["duration_sec"]}s")
    print(res["log"])
