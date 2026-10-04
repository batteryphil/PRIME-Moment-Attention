/*
 * GTRM: Generative Thought Reconstruction Map - Native C99 Implementation
 * =======================================================================
 * Biomimetic 64 KB Constant-Memory Cognitive Layer.
 */

#include "prime_gtrm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void init_random_weights(float *w, size_t count, float scale) {
    for (size_t i = 0; i < count; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        w[i] = r * scale;
    }
}

prime_gtrm_t* prime_gtrm_create(int d_model, int d_map, float decay) {
    if (d_model <= 0 || d_map <= 0) return NULL;

    prime_gtrm_t *gtrm = (prime_gtrm_t*)malloc(sizeof(prime_gtrm_t));
    if (!gtrm) return NULL;

    gtrm->d_model = d_model;
    gtrm->d_map = d_map;
    gtrm->decay = (decay > 0.0f) ? decay : 0.9995f;
    gtrm->eps = 1e-4f;
    gtrm->state_bytes = (size_t)d_map * d_model * sizeof(float);

    /* Allocate state buffer */
    gtrm->m2_state = (float*)calloc((size_t)d_map * d_model, sizeof(float));

    /* Allocate Titans Surprise-Momentum buffer S */
    gtrm->s_momentum = (float*)calloc((size_t)d_map * d_model, sizeof(float));
    gtrm->momentum_decay = 0.90f;
    gtrm->use_titans_surprise = 1;
    gtrm->last_surprise = 0.0f;

    /* Allocate weight projections */
    gtrm->w_map = (float*)malloc((size_t)d_map * d_model * sizeof(float));
    gtrm->w_salience = (float*)malloc((size_t)d_model * sizeof(float));
    gtrm->w_v = (float*)malloc((size_t)d_model * d_model * sizeof(float));
    gtrm->w_q = (float*)malloc((size_t)d_map * d_model * sizeof(float));
    gtrm->w_recon = (float*)malloc((size_t)d_model * d_model * sizeof(float));

    if (!gtrm->m2_state || !gtrm->s_momentum || !gtrm->w_map || !gtrm->w_salience || !gtrm->w_v || !gtrm->w_q || !gtrm->w_recon) {
        prime_gtrm_free(gtrm);
        return NULL;
    }

    /* Initialize projections with Xavier/He scaling */
    float s_map = 1.0f / sqrtf((float)d_model);
    init_random_weights(gtrm->w_map, (size_t)d_map * d_model, s_map);
    init_random_weights(gtrm->w_salience, (size_t)d_model, s_map);
    init_random_weights(gtrm->w_v, (size_t)d_model * d_model, s_map);
    init_random_weights(gtrm->w_q, (size_t)d_map * d_model, s_map);
    init_random_weights(gtrm->w_recon, (size_t)d_model * d_model, s_map);

    return gtrm;
}

void prime_gtrm_set_titans_mode(prime_gtrm_t *gtrm, int enabled, float momentum_decay) {
    if (!gtrm) return;
    gtrm->use_titans_surprise = enabled;
    if (momentum_decay >= 0.0f && momentum_decay < 1.0f) {
        gtrm->momentum_decay = momentum_decay;
    }
}

float prime_gtrm_get_last_surprise(const prime_gtrm_t *gtrm) {
    return gtrm ? gtrm->last_surprise : 0.0f;
}

void prime_gtrm_reset(prime_gtrm_t *gtrm) {
    if (!gtrm || !gtrm->m2_state) return;
    memset(gtrm->m2_state, 0, gtrm->state_bytes);
    if (gtrm->s_momentum) memset(gtrm->s_momentum, 0, gtrm->state_bytes);
    gtrm->last_surprise = 0.0f;
}

void prime_gtrm_free(prime_gtrm_t *gtrm) {
    if (!gtrm) return;
    if (gtrm->m2_state) free(gtrm->m2_state);
    if (gtrm->s_momentum) free(gtrm->s_momentum);
    if (gtrm->w_map) free(gtrm->w_map);
    if (gtrm->w_salience) free(gtrm->w_salience);
    if (gtrm->w_v) free(gtrm->w_v);
    if (gtrm->w_q) free(gtrm->w_q);
    if (gtrm->w_recon) free(gtrm->w_recon);
    free(gtrm);
}

void prime_gtrm_step(
    prime_gtrm_t *gtrm,
    const float *x_in,
    float *out
) {
    int D = gtrm->d_model;
    int M = gtrm->d_map;
    float decay = gtrm->decay;
    float eps = gtrm->eps;

    float m_vec[64];    /* d_map <= 64 */
    float q_vec[64];
    float m_sq[64];
    float q_sq[64];
    float v_vec[1024];  /* d_model <= 1024 */
    float recall[1024];

    /* 1. Topological Blueprint Encoder: m_t = W_map @ x_in (with RMSNorm) */
    float m_sq_sum = 0.0f;
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const float *w_row = gtrm->w_map + i * D;
        for (int j = 0; j < D; j++) sum += w_row[j] * x_in[j];
        m_vec[i] = sum;
        m_sq_sum += sum * sum;
    }
    float m_rms = 1.0f / sqrtf(m_sq_sum / (float)M + 1e-6f);
    for (int i = 0; i < M; i++) {
        m_vec[i] *= m_rms;
        m_sq[i] = m_vec[i] * m_vec[i];
    }

    /* 2. Novelty / Salience Gate: gamma_t = sigmoid(W_sal @ x_in) */
    float sal_dot = 0.0f;
    for (int j = 0; j < D; j++) sal_dot += gtrm->w_salience[j] * x_in[j];
    float gamma_t = sigmoidf_fast(sal_dot);

    /* 3. Memory Value Payload: v_t = W_v @ x_in */
    for (int i = 0; i < D; i++) {
        float sum = 0.0f;
        const float *w_row = gtrm->w_v + i * D;
        for (int j = 0; j < D; j++) sum += w_row[j] * x_in[j];
        v_vec[i] = sum;
    }

    /* 4. Query Projection: q_t = W_q @ x_in (with RMSNorm) */
    float q_sq_sum = 0.0f;
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const float *w_row = gtrm->w_q + i * D;
        for (int j = 0; j < D; j++) sum += w_row[j] * x_in[j];
        q_vec[i] = sum;
        q_sq_sum += sum * sum;
    }
    float q_rms = 1.0f / sqrtf(q_sq_sum / (float)M + 1e-6f);
    float q_norm_sum = 0.0f;
    for (int i = 0; i < M; i++) {
        q_vec[i] *= q_rms;
        q_sq[i] = q_vec[i] * q_vec[i];
        q_norm_sum += q_sq[i];
    }
    float inv_q_norm = 1.0f / (q_norm_sum + eps);

    /* 5. Update 64 KB Quadratic Manifold M^(2) */
    if (gtrm->use_titans_surprise) {
        /* 5a. Google Titans Neural Memory: Test-time surprise error
         * Forward Associative Recall: v_hat = m_sq^T @ M^(2) / (sum m_sq + eps)
         */
        float m_norm_sum = 0.0f;
        for (int i = 0; i < M; i++) m_norm_sum += m_sq[i];
        float inv_m_norm = 1.0f / (m_norm_sum + eps);

        float v_hat[1024];
        float err_sq_sum = 0.0f;
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += m_sq[i] * gtrm->m2_state[i * D + j];
            }
            v_hat[j] = sum * inv_m_norm;
            float diff = v_vec[j] - v_hat[j];
            err_sq_sum += diff * diff;
        }
        float surprise_norm = sqrtf(err_sq_sum / (float)D);
        gtrm->last_surprise = surprise_norm;

        /* Surprise-driven dynamic learning rate:
         * Familiar token -> error ~ 0 -> theta_t ~ 0 (skip consolidation).
         * Surprising token -> error > 0 -> theta_t ~ gamma_t (consolidate).
         */
        float surprise_factor = tanhf(surprise_norm * 2.0f);
        float eta = gtrm->momentum_decay;
        float theta_t = gamma_t * surprise_factor;

        /* Historical Momentum Matrix S_t & Manifold Consolidation M_t:
         * S_t = eta * S_{t-1} + theta_t * (m_sq @ error^T)
         * M_t = decay * M_{t-1} + S_t
         */
        for (int i = 0; i < M; i++) {
            float *m2_row = gtrm->m2_state + i * D;
            float *s_row = gtrm->s_momentum + i * D;
            float m_i = m_sq[i];
            for (int j = 0; j < D; j++) {
                float err_j = v_vec[j] - v_hat[j];
                s_row[j] = eta * s_row[j] + theta_t * m_i * err_j;
                m2_row[j] = decay * m2_row[j] + s_row[j];
            }
        }
    } else {
        gtrm->last_surprise = 1.0f;
        for (int i = 0; i < M; i++) {
            float gated_m_i = m_sq[i] * gamma_t;
            float *m2_row = gtrm->m2_state + i * D;
            for (int j = 0; j < D; j++) {
                m2_row[j] = decay * m2_row[j] + gated_m_i * v_vec[j];
            }
        }
    }

    /* 6. Dynamic Thought Reconstruction Readout: recall = q_sq^T @ M^(2) / norm */
    for (int j = 0; j < D; j++) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += q_sq[i] * gtrm->m2_state[i * D + j];
        }
        recall[j] = sum * inv_q_norm;
    }

    /* 7. Reconstruct thought into hidden representation: out = x_in + W_recon @ recall */
    for (int i = 0; i < D; i++) {
        float sum = 0.0f;
        const float *w_row = gtrm->w_recon + i * D;
        for (int j = 0; j < D; j++) sum += w_row[j] * recall[j];
        out[i] = x_in[i] + sum * 0.2f; /* injected thought */
    }
}

void prime_gtrm_prefill(
    prime_gtrm_t *gtrm,
    int seq_len,
    const float *x_seq,
    float *out_seq
) {
    int D = gtrm->d_model;
    for (int t = 0; t < seq_len; t++) {
        prime_gtrm_step(
            gtrm,
            x_seq + t * D,
            out_seq + t * D
        );
    }
}
