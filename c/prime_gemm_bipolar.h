#ifndef PRIME_GEMM_BIPOLAR_H
#define PRIME_GEMM_BIPOLAR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 1-Bit Bipolar Weight Matrix (-1, +1)
 *
 * Each weight is represented by 1 bit:
 *   1 -> +1
 *   0 -> -1
 *
 * Packed into 64-bit uint64_t words.
 * Storage for M x K matrix: M * ceil(K / 64) * 8 bytes.
 * For 1024x1024: 131,072 bytes (128 KB) -> 32x smaller than FP32, 16x smaller than FP16.
 */
typedef struct {
    int rows;              /* M */
    int cols;              /* K */
    int words_per_row;     /* ceil(K / 64) */
    uint64_t *bits;        /* packed bitmask: rows * words_per_row */
    float scale;           /* mean absolute weight scale alpha */
    size_t total_bytes;    /* total heap footprint in bytes */
} prime_bipolar_matrix_t;

/**
 * Allocate a 1-bit bipolar matrix of dimension rows x cols.
 */
prime_bipolar_matrix_t* prime_bipolar_create(int rows, int cols);

/**
 * Free 1-bit bipolar matrix.
 */
void prime_bipolar_free(prime_bipolar_matrix_t *mat);

/**
 * Quantize and pack float weights into 1-bit bipolar representation.
 * Computes mean scale alpha = (1/N) * sum(|W|) and sign bits.
 */
void prime_bipolar_pack(prime_bipolar_matrix_t *mat, const float *float_weights);

/**
 * Mixed-precision GEMV: Y = scale * (W_bipolar * X_float)
 * Computes matrix-vector multiplication with 1-bit bipolar weights and float inputs.
 */
void prime_bipolar_gemv(
    const prime_bipolar_matrix_t *mat,
    const float *x,
    float *y
);

/**
 * Pure 1-bit XNOR-popcount GEMV:
 * Y = scale * (2 * popcount(~(W ^ X)) - K)
 *
 * Both weights and inputs are packed bipolar bitmasks.
 * Completely multiplier-free: executes purely with bitwise XNOR and hardware popcount!
 */
void prime_bipolar_gemv_pure1bit(
    const prime_bipolar_matrix_t *mat,
    const uint64_t *x_bits,
    float *y,
    float activation_scale
);

/**
 * Pack a float vector into bipolar bitmask: sign(x) >= 0 -> 1 (+1), else 0 (-1).
 */
void prime_bipolar_pack_vector(const float *x, int len, uint64_t *out_bits);

#ifdef __cplusplus
}
#endif

#endif /* PRIME_GEMM_BIPOLAR_H */
