#ifndef PRIME_FRACTAL_MEMORY_H
#define PRIME_FRACTAL_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PRIME_FRACTAL_OCTAVES 8

/**
 * Fractal Multi-Scale Memory Bank
 *
 * Implements 8 geometric decay octaves:
 *   gamma_k = 1.0 - 2^{-(k+1)}, for k in [0, 7]
 *   gamma = [0.5000, 0.7500, 0.8750, 0.9375, 0.96875, 0.984375, 0.9921875, 0.99609375]
 *
 * Each octave maintains an associative memory matrix M_k in R^{d x d},
 * updated via surprise-gated error gradients:
 *   M_{k, t} = gamma_k * M_{k, t-1} + eta * (error_k \otimes k_t)
 */
typedef struct {
    int d_model;                                  /* feature dimension d */
    int num_octaves;                              /* H = 8 */
    float gammas[PRIME_FRACTAL_OCTAVES];          /* decay factor per octave */
    float head_weights[PRIME_FRACTAL_OCTAVES];    /* power-law query weights */
    float *matrices;                              /* num_octaves * d * d floats */
    float learning_rate;                          /* eta */
    size_t state_bytes;                           /* total allocated memory */
} prime_fractal_memory_t;

/**
 * Allocate and initialize an 8-octave fractal memory bank.
 */
prime_fractal_memory_t* prime_fractal_memory_create(int d_model, float learning_rate);

/**
 * Free fractal memory bank.
 */
void prime_fractal_memory_free(prime_fractal_memory_t *bank);

/**
 * Reset memory states to zero.
 */
void prime_fractal_memory_reset(prime_fractal_memory_t *bank);

/**
 * Process a single time step:
 *   k_t: input key vector [d]
 *   v_t: input value vector [d]
 *   out_y: retrieved multi-scale blended output vector [d]
 * Returns total mean surprise across all octaves.
 */
float prime_fractal_memory_step(
    prime_fractal_memory_t *bank,
    const float *k_t,
    const float *v_t,
    float *out_y
);

/**
 * Query associative recall of a key vector q across all octaves.
 */
void prime_fractal_memory_query(
    const prime_fractal_memory_t *bank,
    const float *q,
    float *out_y
);

/**
 * Query associative recall specifically from octave index k.
 */
void prime_fractal_memory_query_octave(
    const prime_fractal_memory_t *bank,
    int octave_idx,
    const float *q,
    float *out_y
);

#ifdef __cplusplus
}
#endif

#endif /* PRIME_FRACTAL_MEMORY_H */
