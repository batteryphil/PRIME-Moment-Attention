#ifndef PRIME_LIE_SU2_H
#define PRIME_LIE_SU2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Non-Abelian SU(2) / Unitary Lie Group State Transition Engine
 *
 * Implements isometric Lie algebra rotations across state channels.
 * Parameterized by SU(2) generator angles theta = (theta_1, theta_2, theta_3)
 * mapped to unit quaternions u = (w, x, y, z) where ||u||_2 = 1.0000000.
 *
 * Guaranteed Properties:
 *   1. det(U) = 1, U^dagger U = I (Strict Isometry)
 *   2. ||U * v||_2 == ||v||_2 (Zero energy dissipation, zero explosion)
 *   3. Non-abelian commutation: U(a) U(b) != U(b) U(a)
 */
typedef struct {
    int dim;               /* state dimension (multiple of 4) */
    int num_blocks;        /* dim / 4 */
    float *thetas;         /* 3 * num_blocks rotation generator angles */
    float *rotations_4x4;  /* 16 * num_blocks precomputed orthogonal matrices */
    size_t state_bytes;    /* total memory */
} prime_su2_engine_t;

/**
 * Create an SU(2) Lie group transition engine for dimension dim.
 */
prime_su2_engine_t* prime_su2_create(int dim);

/**
 * Free SU(2) engine.
 */
void prime_su2_free(prime_su2_engine_t *engine);

/**
 * Set generator angles for block b (0 <= b < num_blocks).
 * Maps theta = (t1, t2, t3) via exponential map exp(i * theta . sigma) to unit quaternion.
 */
void prime_su2_set_block_generators(
    prime_su2_engine_t *engine,
    int block_idx,
    float t1,
    float t2,
    float t3
);

/**
 * Initialize all blocks with smooth harmonic angles.
 */
void prime_su2_init_harmonic(prime_su2_engine_t *engine, float base_frequency);

/**
 * Apply isometric unitary transition:
 *   state_out = U(theta) * state_in
 * Guarantees ||state_out||_2 == ||state_in||_2 identically.
 */
void prime_su2_apply(
    const prime_su2_engine_t *engine,
    const float *state_in,
    float *state_out
);

/**
 * Recurrent state step with input injection:
 *   state_{t} = U(theta_t) * state_{t-1} + delta_in
 */
void prime_su2_step(
    const prime_su2_engine_t *engine,
    const float *state_prev,
    const float *delta_in,
    float *state_next
);

/**
 * Measure Euclidean norm of state vector.
 */
float prime_su2_norm(const float *vec, int dim);

#ifdef __cplusplus
}
#endif

#endif /* PRIME_LIE_SU2_H */
