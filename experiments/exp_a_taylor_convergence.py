"""
Experiment A: Mathematical Correctness and Taylor Convergence at Small Head Dimensions (D in {2, 4, 8})

Validates:
1. Exact Softmax Attention
2. Full 2nd-order Taylor Expansion (using full outer product rank-3 tensor S2)
3. Diagonal 2nd-order Moment Approximation (PRIME Moment Attention, O(D^2) state)

Tests across scaling magnitudes s = |q^T k| / sqrt(D) in [0.01, 2.0] to demonstrate
the O(s^3) residual regime and quantify the diagonal approximation.
"""

import torch
import numpy as np

def run_exp_a():
    torch.manual_seed(42)
    device = "cpu"
    
    print("=" * 80)
    print("EXPERIMENT A: MATHEMATICAL TAYLOR CONVERGENCE & TENSOR ERROR ANALYSIS")
    print("=" * 80)
    
    dims = [2, 4, 8]
    scales = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    N = 100 # sequence length
    
    results = {}
    
    for D in dims:
        print(f"\n--- Head Dimension D = {D} (Seq Len L = {N}) ---")
        results[D] = []
        
        # Generate random baseline queries, keys, values
        V = torch.randn(N, D, dtype=torch.float64, device=device)
        V = V / np.sqrt(D)
        
        for scale in scales:
            # Generate Q and K scaled such that average |q^T k| / sqrt(D) ~ scale
            Q = torch.randn(1, D, dtype=torch.float64, device=device)
            Q = Q / torch.norm(Q) * np.sqrt(scale * np.sqrt(D))
            
            K = torch.randn(N, D, dtype=torch.float64, device=device)
            K = K / torch.norm(K, dim=-1, keepdim=True) * np.sqrt(scale * np.sqrt(D))
            
            # Compute raw logits s_j = (q . k_j) / sqrt(D)
            logits = (Q @ K.T) / np.sqrt(D) # [1, N]
            actual_max_s = torch.max(torch.abs(logits)).item()
            
            # 1. Exact Softmax Attention
            attn_weights = torch.softmax(logits, dim=-1) # [1, N]
            y_softmax = attn_weights @ V # [1, D]
            
            # 2. Full 2nd-Order Taylor Attention
            # exp(s) ~ 1 + s + 0.5 * s^2
            s = logits.squeeze(0) # [N]
            poly_weights_full = 1.0 + s + 0.5 * (s ** 2)
            den_full = torch.sum(poly_weights_full)
            y_full_direct = torch.sum(poly_weights_full.unsqueeze(1) * V, dim=0, keepdim=True) / den_full
            
            # Recurrent tensor formulation:
            # S0 = sum v_j               [D]
            # S1 = sum k_j * v_j^T       [D, D]
            # S2 = sum (k_j x k_j) x v_j [D, D, D]
            S0 = torch.sum(V, dim=0) # [D]
            S1 = K.T @ V             # [D, D]
            S2_full = torch.einsum('na,nb,nc->abc', K, K, V) # [D, D, D]
            
            K0 = float(N)
            K1 = torch.sum(K, dim=0) # [D]
            K2_full = K.T @ K        # [D, D]
            
            # Numerator & Denominator via tensor contraction:
            term1_num = (Q @ S1).squeeze(0) / np.sqrt(D) # [D]
            term2_num = torch.einsum('a,b,abc->c', Q.squeeze(0), Q.squeeze(0), S2_full) / (2.0 * D) # [D]
            num_recurrent = S0 + term1_num + term2_num # [D]
            
            term1_den = (Q @ K1).item() / np.sqrt(D)
            term2_den = (Q @ K2_full @ Q.T).item() / (2.0 * D)
            den_recurrent = K0 + term1_den + term2_den
            
            y_full_recurrent = (num_recurrent / den_recurrent).unsqueeze(0)
            
            # Check equivalence
            tensor_equiv_error = torch.norm(y_full_direct - y_full_recurrent).item()
            assert tensor_equiv_error < 1e-12, f"Tensor equivalence failed: {tensor_equiv_error}"
            
            # 3. Diagonal 2nd-Order PRIME Moment Approximation
            # Approximates k k^T by diag(k^2), keeping S2 in O(D^2) parameter space:
            S2_diag = (K ** 2).T @ V # [D, D]
            K2_diag = torch.sum(K ** 2, dim=0) # [D]
            
            term2_num_diag = ((Q ** 2) @ S2_diag).squeeze(0) / (2.0 * D) # [D]
            term2_den_diag = ((Q ** 2) @ K2_diag).item() / (2.0 * D)
            
            num_diag = S0 + term1_num + term2_num_diag
            den_diag = K0 + term1_den + term2_den_diag
            y_diag_prime = (num_diag / den_diag).unsqueeze(0)
            
            # Compute Errors
            err_full = torch.norm(y_full_recurrent - y_softmax).item()
            err_diag = torch.norm(y_diag_prime - y_softmax).item()
            err_diag_vs_full = torch.norm(y_diag_prime - y_full_recurrent).item()
            
            results[D].append({
                "scale": scale,
                "max_s": actual_max_s,
                "err_full": err_full,
                "err_diag": err_diag,
                "err_diag_vs_full": err_diag_vs_full
            })
            
            print(f"  Scale={scale:4.2f} (max|s|={actual_max_s:4.2f}) | "
                  f"Full Taylor Err: {err_full:8.5f} | "
                  f"PRIME Diag Err: {err_diag:8.5f} | "
                  f"Diag vs Full: {err_diag_vs_full:8.5f}")

    print("\n" + "=" * 80)
    print("SCALING REGIME VERIFICATION (Checking O(s^3) Taylor Residual):")
    print("=" * 80)
    for D in dims:
        s1, s2 = results[D][0], results[D][1]
        slope_full = (np.log(s2["err_full"]) - np.log(s1["err_full"])) / (np.log(s2["scale"]) - np.log(s1["scale"]))
        print(f"Dimension D={D}: Empirical Taylor convergence slope = {slope_full:.2f} (Theoretical O(s^3) = 3.00)")
        
    return results

if __name__ == "__main__":
    run_exp_a()
