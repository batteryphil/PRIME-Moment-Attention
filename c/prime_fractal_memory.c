#include "prime_fractal_memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

prime_fractal_memory_t* prime_fractal_memory_create(int d_model, float learning_rate) {
    if (d_model <= 0) return NULL;

    prime_fractal_memory_t *bank = (prime_fractal_memory_t*)malloc(sizeof(prime_fractal_memory_t));
    if (!bank) return NULL;

    bank->d_model = d_model;
    bank->num_octaves = PRIME_FRACTAL_OCTAVES;
    bank->learning_rate = (learning_rate > 0.0f) ? learning_rate : 0.01f;

    /* Compute geometric power-law decay octaves */
    float weight_sum = 0.0f;
    for (int k = 0; k < PRIME_FRACTAL_OCTAVES; k++) {
        bank->gammas[k] = 1.0f - powf(0.5f, (float)(k + 1));
        bank->head_weights[k] = 1.0f / sqrtf((float)(k + 1));
        weight_sum += bank->head_weights[k];
    }

    /* Normalize power-law query weights */
    for (int k = 0; k < PRIME_FRACTAL_OCTAVES; k++) {
        bank->head_weights[k] /= weight_sum;
    }

    size_t total_elements = (size_t)PRIME_FRACTAL_OCTAVES * (size_t)d_model * (size_t)d_model;
    bank->matrices = (float*)calloc(total_elements, sizeof(float));
    if (!bank->matrices) {
        free(bank);
        return NULL;
    }

    bank->state_bytes = sizeof(prime_fractal_memory_t) + total_elements * sizeof(float);
    return bank;
}

void prime_fractal_memory_free(prime_fractal_memory_t *bank) {
    if (!bank) return;
    if (bank->matrices) free(bank->matrices);
    free(bank);
}

void prime_fractal_memory_reset(prime_fractal_memory_t *bank) {
    if (!bank || !bank->matrices) return;
    size_t total_elements = (size_t)bank->num_octaves * (size_t)bank->d_model * (size_t)bank->d_model;
    memset(bank->matrices, 0, total_elements * sizeof(float));
}

float prime_fractal_memory_step(
    prime_fractal_memory_t *bank,
    const float *k_t,
    const float *v_t,
    float *out_y
) {
    if (!bank || !k_t || !v_t) return 0.0f;

    int D = bank->d_model;
    int H = bank->num_octaves;
    float eta = bank->learning_rate;

    float *pred = (float*)malloc((size_t)D * sizeof(float));
    float *err  = (float*)malloc((size_t)D * sizeof(float));
    if (!pred || !err) {
        if (pred) free(pred);
        if (err) free(err);
        return 0.0f;
    }

    if (out_y) {
        memset(out_y, 0, (size_t)D * sizeof(float));
    }

    float total_surprise = 0.0f;

    for (int k = 0; k < H; k++) {
        float *M_k = bank->matrices + (size_t)k * (size_t)D * (size_t)D;
        float gamma = bank->gammas[k];
        float w_k = bank->head_weights[k];

        /* 1. Predict value: pred = M_k * k_t */
        float octave_err_sq = 0.0f;
        for (int i = 0; i < D; i++) {
            float dot = 0.0f;
            const float *row = M_k + (size_t)i * (size_t)D;
            for (int j = 0; j < D; j++) {
                dot += row[j] * k_t[j];
            }
            pred[i] = dot;
            float diff = v_t[i] - dot;
            err[i] = diff;
            octave_err_sq += diff * diff;

            if (out_y) {
                out_y[i] += w_k * dot;
            }
        }

        total_surprise += sqrtf(octave_err_sq / (float)D + 1e-8f);

        /* 2. Surprise-gated associative update with octave decay */
        for (int i = 0; i < D; i++) {
            float *row = M_k + (size_t)i * (size_t)D;
            float grad_i = eta * err[i];
            for (int j = 0; j < D; j++) {
                row[j] = gamma * row[j] + grad_i * k_t[j];
            }
        }
    }

    free(pred);
    free(err);

    return total_surprise / (float)H;
}

void prime_fractal_memory_query(
    const prime_fractal_memory_t *bank,
    const float *q,
    float *out_y
) {
    if (!bank || !q || !out_y) return;

    int D = bank->d_model;
    int H = bank->num_octaves;
    memset(out_y, 0, (size_t)D * sizeof(float));

    for (int k = 0; k < H; k++) {
        const float *M_k = bank->matrices + (size_t)k * (size_t)D * (size_t)D;
        float w_k = bank->head_weights[k];

        for (int i = 0; i < D; i++) {
            float dot = 0.0f;
            const float *row = M_k + (size_t)i * (size_t)D;
            for (int j = 0; j < D; j++) {
                dot += row[j] * q[j];
            }
            out_y[i] += w_k * dot;
        }
    }
}

void prime_fractal_memory_query_octave(
    const prime_fractal_memory_t *bank,
    int octave_idx,
    const float *q,
    float *out_y
) {
    if (!bank || !q || !out_y) return;
    if (octave_idx < 0 || octave_idx >= bank->num_octaves) return;

    int D = bank->d_model;
    const float *M_k = bank->matrices + (size_t)octave_idx * (size_t)D * (size_t)D;

    for (int i = 0; i < D; i++) {
        float dot = 0.0f;
        const float *row = M_k + (size_t)i * (size_t)D;
        for (int j = 0; j < D; j++) {
            dot += row[j] * q[j];
        }
        out_y[i] = dot;
    }
}
