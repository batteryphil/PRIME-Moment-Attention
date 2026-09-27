/*
 * PRIME Native 1.58-Bit Ternary Bitmask GEMM Header
 * ==================================================
 * Packed 2-bit per weight {-1, 0, +1} multiplication-free GEMM.
 * Eliminates all floating-point multiplications from linear projections,
 * replacing them with bitwise mask testing and sign accumulation.
 *
 * Memory Footprint:
 *   Exactly 2 bits per parameter = 8x compression over FP16.
 *   10B parameter model = 2.38 GB packed in memory.
 */

#ifndef PRIME_GEMM_TERNARY_H
#define PRIME_GEMM_TERNARY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int rows;
    int cols;
    int words_per_row;
    size_t total_bytes;
    uint64_t *pos_words; /* [rows * words_per_row] */
    uint64_t *neg_words; /* [rows * words_per_row] */
} prime_ternary_matrix_t;

/* Allocate packed ternary matrix */
prime_ternary_matrix_t* prime_ternary_create(int rows, int cols);

/* Pack from continuous float weights using ternary quantization (e.g. threshold = 0.33f) */
void prime_ternary_pack(
    prime_ternary_matrix_t *mat,
    const float *float_weights,
    float threshold
);

/*
 * Compute Y = W * X without any floating-point multiplications:
 * For each row i: Y_i = sum_{j in pos} X_j - sum_{j in neg} X_j
 */
void prime_ternary_gemv(
    const prime_ternary_matrix_t *mat,
    const float *x,
    float *y
);

/* Compute batch GEMM: Y = W * X [cols, batch_size] -> Y [rows, batch_size] */
void prime_ternary_gemm(
    const prime_ternary_matrix_t *mat,
    const float *x,
    float *y,
    int batch_size
);

/* Free allocated ternary matrix */
void prime_ternary_free(prime_ternary_matrix_t *mat);

#ifdef __cplusplus
}
#endif

#endif /* PRIME_GEMM_TERNARY_H */
