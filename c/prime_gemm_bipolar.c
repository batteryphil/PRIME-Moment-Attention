#include "prime_gemm_bipolar.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

prime_bipolar_matrix_t* prime_bipolar_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;

    prime_bipolar_matrix_t *mat = (prime_bipolar_matrix_t*)malloc(sizeof(prime_bipolar_matrix_t));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->words_per_row = (cols + 63) / 64;
    mat->scale = 1.0f;

    size_t num_words = (size_t)mat->rows * (size_t)mat->words_per_row;
    size_t bitmask_bytes = num_words * sizeof(uint64_t);

    mat->bits = (uint64_t*)calloc(num_words, sizeof(uint64_t));
    if (!mat->bits) {
        free(mat);
        return NULL;
    }

    mat->total_bytes = sizeof(prime_bipolar_matrix_t) + bitmask_bytes;
    return mat;
}

void prime_bipolar_free(prime_bipolar_matrix_t *mat) {
    if (!mat) return;
    if (mat->bits) free(mat->bits);
    free(mat);
}

void prime_bipolar_pack(prime_bipolar_matrix_t *mat, const float *float_weights) {
    if (!mat || !float_weights) return;

    int M = mat->rows;
    int K = mat->cols;
    int wpr = mat->words_per_row;

    /* Compute mean absolute scale alpha = (1 / (M * K)) * sum(|W|) */
    double sum_abs = 0.0;
    size_t total_elements = (size_t)M * (size_t)K;
    for (size_t i = 0; i < total_elements; i++) {
        sum_abs += fabsf(float_weights[i]);
    }
    mat->scale = (float)(sum_abs / (double)total_elements);
    if (mat->scale < 1e-8f) mat->scale = 1.0f;

    /* Zero bitmask */
    memset(mat->bits, 0, (size_t)M * (size_t)wpr * sizeof(uint64_t));

    /* Pack signs: W >= 0.0f -> bit 1 (+1), W < 0.0f -> bit 0 (-1) */
    for (int r = 0; r < M; r++) {
        const float *row_in = float_weights + (size_t)r * (size_t)K;
        uint64_t *row_bits = mat->bits + (size_t)r * (size_t)wpr;

        for (int c = 0; c < K; c++) {
            if (row_in[c] >= 0.0f) {
                int word_idx = c / 64;
                int bit_idx = c % 64;
                row_bits[word_idx] |= (1ULL << bit_idx);
            }
        }
    }
}

void prime_bipolar_pack_vector(const float *x, int len, uint64_t *out_bits) {
    if (!x || !out_bits || len <= 0) return;
    int wpr = (len + 63) / 64;
    memset(out_bits, 0, (size_t)wpr * sizeof(uint64_t));

    for (int c = 0; c < len; c++) {
        if (x[c] >= 0.0f) {
            int word_idx = c / 64;
            int bit_idx = c % 64;
            out_bits[word_idx] |= (1ULL << bit_idx);
        }
    }
}

void prime_bipolar_gemv_pure1bit(
    const prime_bipolar_matrix_t *mat,
    const uint64_t *x_bits,
    float *y,
    float activation_scale
) {
    if (!mat || !x_bits || !y) return;

    int M = mat->rows;
    int K = mat->cols;
    int wpr = mat->words_per_row;
    int rem = K % 64;
    uint64_t last_word_mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);

    float total_scalar = mat->scale * activation_scale;

    for (int r = 0; r < M; r++) {
        const uint64_t *row_bits = mat->bits + (size_t)r * (size_t)wpr;
        int match_count = 0;

        for (int w = 0; w < wpr; w++) {
            uint64_t w_word = row_bits[w];
            uint64_t x_word = x_bits[w];
            uint64_t xnor_val = ~(w_word ^ x_word);

            if (w == wpr - 1) {
                xnor_val &= last_word_mask;
            }
            match_count += __builtin_popcountll(xnor_val);
        }

        /* Dot product identity: matches - mismatches = 2*matches - K */
        int dot_int = 2 * match_count - K;
        y[r] = (float)dot_int * total_scalar;
    }
}

void prime_bipolar_gemv(
    const prime_bipolar_matrix_t *mat,
    const float *x,
    float *y
) {
    if (!mat || !x || !y) return;

    int M = mat->rows;
    int K = mat->cols;
    int wpr = mat->words_per_row;
    float scale = mat->scale;

    for (int r = 0; r < M; r++) {
        const uint64_t *row_bits = mat->bits + (size_t)r * (size_t)wpr;
        float acc = 0.0f;
        int c = 0;

        for (int w = 0; w < wpr && c < K; w++) {
            uint64_t bits = row_bits[w];
            int limit = (c + 64 <= K) ? 64 : (K - c);

            for (int b = 0; b < limit; b++) {
                float sign = (bits & (1ULL << b)) ? 1.0f : -1.0f;
                acc += sign * x[c + b];
            }
            c += limit;
        }

        y[r] = acc * scale;
    }
}
