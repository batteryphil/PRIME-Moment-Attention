/*
 * PRIME Native 1.58-Bit Ternary Bitmask GEMM Implementation
 * ==========================================================
 * Multiplication-free sign-accumulating matrix operations.
 */

#include "prime_gemm_ternary.h"
#include <stdlib.h>
#include <string.h>

prime_ternary_matrix_t* prime_ternary_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;

    prime_ternary_matrix_t *mat = (prime_ternary_matrix_t*)malloc(sizeof(prime_ternary_matrix_t));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->words_per_row = (cols + 63) / 64;

    size_t total_words = (size_t)rows * mat->words_per_row;
    mat->total_bytes = sizeof(prime_ternary_matrix_t) + 2 * total_words * sizeof(uint64_t);

    mat->pos_words = (uint64_t*)calloc(total_words, sizeof(uint64_t));
    mat->neg_words = (uint64_t*)calloc(total_words, sizeof(uint64_t));

    if (!mat->pos_words || !mat->neg_words) {
        if (mat->pos_words) free(mat->pos_words);
        if (mat->neg_words) free(mat->neg_words);
        free(mat);
        return NULL;
    }

    return mat;
}

void prime_ternary_pack(
    prime_ternary_matrix_t *mat,
    const float *float_weights,
    float threshold
) {
    if (!mat || !float_weights) return;

    size_t total_words = (size_t)mat->rows * mat->words_per_row;
    memset(mat->pos_words, 0, total_words * sizeof(uint64_t));
    memset(mat->neg_words, 0, total_words * sizeof(uint64_t));

    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            float w = float_weights[r * mat->cols + c];
            int word_idx = r * mat->words_per_row + (c / 64);
            int bit_idx = c % 64;

            if (w > threshold) {
                mat->pos_words[word_idx] |= (1ULL << bit_idx);
            } else if (w < -threshold) {
                mat->neg_words[word_idx] |= (1ULL << bit_idx);
            }
        }
    }
}

void prime_ternary_gemv(
    const prime_ternary_matrix_t *mat,
    const float *x,
    float *y
) {
    if (!mat || !x || !y) return;

    int cols = mat->cols;
    int words_per_row = mat->words_per_row;

    for (int r = 0; r < mat->rows; r++) {
        float sum = 0.0f;
        const uint64_t *r_pos = mat->pos_words + r * words_per_row;
        const uint64_t *r_neg = mat->neg_words + r * words_per_row;

        for (int w = 0; w < words_per_row; w++) {
            uint64_t pos = r_pos[w];
            uint64_t neg = r_neg[w];
            int col_base = w * 64;

            while (pos) {
                int bit = __builtin_ctzll(pos);
                int c = col_base + bit;
                if (c < cols) sum += x[c];
                pos &= (pos - 1);
            }

            while (neg) {
                int bit = __builtin_ctzll(neg);
                int c = col_base + bit;
                if (c < cols) sum -= x[c];
                neg &= (neg - 1);
            }
        }

        y[r] = sum;
    }
}

void prime_ternary_gemm(
    const prime_ternary_matrix_t *mat,
    const float *x,
    float *y,
    int batch_size
) {
    if (!mat || !x || !y || batch_size <= 0) return;

    for (int b = 0; b < batch_size; b++) {
        const float *x_col = x + b * mat->cols;
        float *y_col = y + b * mat->rows;
        prime_ternary_gemv(mat, x_col, y_col);
    }
}

void prime_ternary_free(prime_ternary_matrix_t *mat) {
    if (!mat) return;
    if (mat->pos_words) free(mat->pos_words);
    if (mat->neg_words) free(mat->neg_words);
    free(mat);
}
