"""
PRIME-Scout <-> PRIME-Net Integration Bridge
Connects the Autonomous Scientist to batteryphil's PRIME-Net engine:
Pareto-Refined Invariant Mining Engine (AFPO / Symbolic Regression).

When PRIME-Scout benchmarks an architectural hypothesis, this bridge
takes the empirical measurements and discovers the exact closed-form
symbolic invariant equation governing the behavior.
STRICT SAFETY POLICY: NEVER COMMITS OR PUSHES TO GIT.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from scout.config import BASE_DIR

PRIME_NET_DIR = BASE_DIR / "scout" / "sandbox" / "active" / "batteryphil_PRIME-Net"

class PrimeNetBridge:
    def __init__(self):
        self._available = False
        self._prime_core = None
        self._init_engine()

    def _init_engine(self):
        if not PRIME_NET_DIR.exists():
            return
        
        prime_path = str(PRIME_NET_DIR)
        if prime_path not in sys.path:
            sys.path.insert(0, prime_path)
            
        try:
            import prime_core
            self._prime_core = prime_core
            self._available = True
        except Exception as e:
            print(f"[!] PRIME-Net import warning: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def discover_empirical_invariant(
        self,
        X: Any,
        y: Any,
        timeout_sec: float = 6.0,
        pop_size: int = 64,
        max_generations: int = 60,
        var_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs PRIME-Net symbolic regression on empirical telemetry data.
        Supports 1D and multi-dimensional (2D, 3D) feature matrices X.
        Returns:
        - equation_str: string representation of best SymPy equation (X1, X2, ...)
        - named_equation_str: optional equation string with user-provided variable names
        - r2_score: float R^2 fit quality
        - mse: mean squared error
        - discovery_gen: generation discovered
        - num_vars: number of input variables
        - variables: list of variable names
        """
        if not self._available:
            self._init_engine()
            if not self._available:
                return {
                    "equation_str": None,
                    "named_equation_str": None,
                    "r2_score": 0.0,
                    "mse": float("inf"),
                    "num_vars": 1,
                    "variables": var_names or ["X1"],
                    "status": "ENGINE_UNAVAILABLE"
                }

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        num_vars = int(X_arr.shape[1])
        default_names = [f"X{i+1}" for i in range(num_vars)]
        resolved_names = var_names if (var_names and len(var_names) == num_vars) else default_names

        try:
            res = self._prime_core.run_prime_engine(
                X_arr, y_arr,
                timeout_sec=timeout_sec,
                pop_size=pop_size,
                max_generations=max_generations
            )

            sympy_expr = res.get("best_sympy")
            eq_str = str(sympy_expr) if sympy_expr is not None else None
            named_eq_str = eq_str

            # Substitute user-provided variable names into SymPy expression if available
            if sympy_expr is not None and resolved_names != default_names:
                try:
                    import sympy as sp
                    subs_dict = {sp.Symbol(f"X{i+1}"): sp.Symbol(resolved_names[i]) for i in range(num_vars)}
                    named_expr = sympy_expr.subs(subs_dict)
                    named_eq_str = str(named_expr)
                except Exception:
                    named_eq_str = eq_str

            r2 = float(res.get("train_r2", 0.0))
            mse = float(res.get("train_mse", 0.0))

            return {
                "equation_str": eq_str,
                "named_equation_str": named_eq_str,
                "r2_score": round(r2, 4),
                "mse": round(mse, 6),
                "discovery_gen": res.get("discovery_gen", 0),
                "num_vars": num_vars,
                "variables": resolved_names,
                "status": "SUCCESS" if eq_str else "NO_CONVERGENCE"
            }
        except Exception as e:
            return {
                "equation_str": None,
                "named_equation_str": None,
                "r2_score": 0.0,
                "mse": float("inf"),
                "num_vars": num_vars,
                "variables": resolved_names,
                "status": f"ERROR: {e}"
            }

if __name__ == "__main__":
    bridge = PrimeNetBridge()
    print("[*] Testing PrimeNetBridge...")
    print(f"    Available: {bridge.is_available}")
    if bridge.is_available:
        # Synthesize exponential decay curve: y = 100 * (0.95 ^ x)
        x = np.arange(1, 20, dtype=np.float64)
        y = 100.0 * np.exp(-0.05 * x)
        inv = bridge.discover_empirical_invariant(x, y, timeout_sec=5.0)
        print("Discovered Invariant:", inv)
