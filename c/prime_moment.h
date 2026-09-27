/*
 * PRIME Moment Attention: Zero-Dependency C99 Native Header
 * =========================================================
 * Constant-state O(1) second-order recurrent attention operator.
 *
 * Implements:
 *   1. Second-order Taylor polynomial recurrence:
 *      exp( (q_t^T k_j) / sqrt(D) ) ~ 1 + s_tj + 0.5 * s_tj^2
 *   2. Gated Delta-PRIME decoupled erase/write error delta updates:
 *      e_t = v_t - (S_1 k_t + 0.5 S_2 k_t^2)
 *      S_1 <- (1 - b_t) S_1 + w_t e_t k_t^T
 *   3. Dual-state Lyapunov contractive norm projection (||S_1||_F, ||S_2||_F <= C_max)
 *   4. Second-order Harmonic Phasor RoPE (fundamental theta and 2nd harmonic 2*theta)
 *
 * Memory Complexity:
 *   - Strictly O(1) in sequence length L.
 *   - Exactly (2*D^2 + 3*D + 1) * sizeof(float) per attention head.
 */

#ifndef PRIME_MOMENT_H
#define PRIME_MOMENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_heads;
    int head_dim;
    float decay;            /* e.g. 0.9995f */
    float eps;              /* e.g. 1e-4f */
    int use_qk_norm;        /* 1 to apply LayerNorm to Q and K, 0 otherwise */
    int use_delta_rule;     /* 1 for Gated Delta-PRIME error correction, 0 for additive */
    float lyapunov_bound;   /* Maximum Frobenius norm ceiling (e.g. 25.0f, 0 to disable) */
    int use_harmonic_rope;  /* 1 for 2nd-order harmonic phasor rotation, 0 otherwise */

    /* Wave 3 Frontier Advancements */
    int use_gdn2_decoupled; /* 1 for NVIDIA Gated DeltaNet-2 decoupled channel erase/write */
    int use_differential;   /* 1 for Microsoft Differential Taylor Attention (noise cancellation) */
    float diff_lambda;      /* Differential noise subtraction factor lambda (default: 0.5f) */
    int latent_dim;         /* DeepSeek MLA latent key dimension d_c (e.g. 16; 0 for full D) */

    /* Wave 4 Frontier Advancements */
    int use_rwkv7_curvature_delta; /* 1 for RWKV-7 2nd-order curvature error delta update */
    int use_symplectic_integrator; /* 1 for Symplectic Hamiltonian phase-space volume preservation */
    float symplectic_theta;        /* Symplectic rotation angle theta (default: 0.01f) */
} prime_config_t;

typedef struct {
    int num_heads;
    int head_dim;
    int latent_dim;         /* 0 for full D x D, >0 for DeepSeek MLA compressed D x d_c */
    size_t state_bytes;     /* Total allocated bytes for recurrent state */
    
    /* Recurrent state tensors (flattened contiguous arrays):
     * S0: [H, D]          (0th value moment: sum v_j)
     * S1: [H, D, K_DIM]   (1st covariance tensor, where K_DIM = latent_dim ? latent_dim : head_dim)
     * S2: [H, D, K_DIM]   (2nd curvature moment)
     * K0: [H]             (0th key count: sum 1)
     * K1: [H, K_DIM]      (1st key sum)
     * K2: [H, K_DIM]      (2nd key square sum)
     */
    float *s0;
    float *s1;
    float *s2;
    float *k0;
    float *k1;
    float *k2;

    /* DeepSeek MLA Projection Weight: W_c [head_dim, latent_dim] */
    float *w_mla_c;
} prime_state_t;

/* Create default configuration */
prime_config_t prime_default_config(int num_heads, int head_dim);

/* Create Wave 3 frontier configuration */
prime_config_t prime_wave3_config(int num_heads, int head_dim, int latent_dim);

/* Create Wave 4 frontier configuration (RWKV-7 curvature delta + Symplectic flow + MLA) */
prime_config_t prime_wave4_config(int num_heads, int head_dim, int latent_dim);

/* Compute total state Frobenius energy: sum(||S_1||_F^2 + ||S_2||_F^2) */
float prime_state_frobenius_energy(const prime_state_t *state);

/* Allocate recurrent state memory (latent_dim == 0 for standard full D x D) */
prime_state_t* prime_state_create_ext(int num_heads, int head_dim, int latent_dim);

/* Backward-compatible state allocator (calls prime_state_create_ext with latent_dim=0) */
prime_state_t* prime_state_create(int num_heads, int head_dim);

/* Reset all state tensors to zero */
void prime_state_reset(prime_state_t *state);

/* Free allocated state memory */
void prime_state_free(prime_state_t *state);

/* Apply dual-state Lyapunov contractive projection */
void prime_lyapunov_project(prime_state_t *state, float bound);

/* Apply Second-Order Harmonic Phasor RoPE */
void prime_apply_harmonic_rope(
    float *q, float *k, float *q2, float *k2,
    int head_dim, int pos, float base_freq
);

/*
 * Single-step autoregressive decode (Additive or Gated Delta-PRIME):
 * Inputs:
 *   cfg    : Attention configuration
 *   state  : Pointer to recurrent state (updated in-place)
 *   q_in   : Query tensor  [num_heads, head_dim]
 *   k_in   : Key tensor    [num_heads, head_dim]
 *   v_in   : Value tensor  [num_heads, head_dim]
 *   b_gate : Optional erase gate [num_heads, head_dim] (NULL defaults to 0.05)
 *   w_gate : Optional write gate [num_heads, head_dim] (NULL defaults to 0.25)
 * Outputs:
 *   out    : Output tensor [num_heads, head_dim]
 */
void prime_step_delta(
    const prime_config_t *cfg,
    prime_state_t *state,
    const float *q_in,
    const float *k_in,
    const float *v_in,
    const float *b_gate,
    const float *w_gate,
    float *out
);

/* Backward-compatible standard step (calls prime_step_delta with NULL gates) */
void prime_step(
    const prime_config_t *cfg,
    prime_state_t *state,
    const float *q_in,
    const float *k_in,
    const float *v_in,
    float *out
);

/* Multi-token prefill sequence */
void prime_prefill(
    const prime_config_t *cfg,
    prime_state_t *state,
    int seq_len,
    const float *q_seq,
    const float *k_seq,
    const float *v_seq,
    float *out_seq
);

#ifdef __cplusplus
}
#endif

#endif /* PRIME_MOMENT_H */
