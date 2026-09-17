/*
 * PRIME Moment Attention: Zero-Dependency C99 Native Implementation
 * =================================================================
 * Constant-state O(1) second-order recurrent attention operator.
 */

#include "prime_moment.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STACK_HEAD_DIM 512

prime_config_t prime_default_config(int num_heads, int head_dim) {
    prime_config_t cfg;
    cfg.num_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.decay = 0.9995f;
    cfg.eps = 1e-4f;
    cfg.use_qk_norm = 0;
    return cfg;
}

prime_state_t* prime_state_create(int num_heads, int head_dim) {
    if (num_heads <= 0 || head_dim <= 0) {
        return NULL;
    }

    prime_state_t *state = (prime_state_t*)malloc(sizeof(prime_state_t));
    if (!state) return NULL;

    state->num_heads = num_heads;
    state->head_dim = head_dim;

    size_t s0_size = (size_t)num_heads * head_dim * sizeof(float);
    size_t s1_size = (size_t)num_heads * head_dim * head_dim * sizeof(float);
    size_t s2_size = (size_t)num_heads * head_dim * head_dim * sizeof(float);
    size_t k0_size = (size_t)num_heads * sizeof(float);
    size_t k1_size = (size_t)num_heads * head_dim * sizeof(float);
    size_t k2_size = (size_t)num_heads * head_dim * sizeof(float);

    state->state_bytes = s0_size + s1_size + s2_size + k0_size + k1_size + k2_size;

    state->s0 = (float*)malloc(s0_size);
    state->s1 = (float*)malloc(s1_size);
    state->s2 = (float*)malloc(s2_size);
    state->k0 = (float*)malloc(k0_size);
    state->k1 = (float*)malloc(k1_size);
    state->k2 = (float*)malloc(k2_size);

    if (!state->s0 || !state->s1 || !state->s2 || !state->k0 || !state->k1 || !state->k2) {
        prime_state_free(state);
        return NULL;
    }

    prime_state_reset(state);
    return state;
}

void prime_state_reset(prime_state_t *state) {
    if (!state) return;
    int H = state->num_heads;
    int D = state->head_dim;

    memset(state->s0, 0, (size_t)H * D * sizeof(float));
    memset(state->s1, 0, (size_t)H * D * D * sizeof(float));
    memset(state->s2, 0, (size_t)H * D * D * sizeof(float));
    memset(state->k0, 0, (size_t)H * sizeof(float));
    memset(state->k1, 0, (size_t)H * D * sizeof(float));
    memset(state->k2, 0, (size_t)H * D * sizeof(float));
}

void prime_state_free(prime_state_t *state) {
    if (!state) return;
    if (state->s0) free(state->s0);
    if (state->s1) free(state->s1);
    if (state->s2) free(state->s2);
    if (state->k0) free(state->k0);
    if (state->k1) free(state->k1);
    if (state->k2) free(state->k2);
    free(state);
}

static void apply_layer_norm(const float *src, float *dst, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += src[i];
    }
    float mean = sum / (float)dim;

    float sq_diff_sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = src[i] - mean;
        sq_diff_sum += diff * diff;
    }
    float variance = sq_diff_sum / (float)dim;
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);

    for (int i = 0; i < dim; i++) {
        dst[i] = (src[i] - mean) * inv_std;
    }
}

void prime_step(
    const prime_config_t *cfg,
    prime_state_t *state,
    const float *q_in,
    const float *k_in,
    const float *v_in,
    float *out
) {
    int H = cfg->num_heads;
    int D = cfg->head_dim;
    float decay = cfg->decay;
    float eps = cfg->eps;
    float q_scale = 1.0f / sqrtf((float)D);

    /* Allocate small stack scratch buffers if D <= MAX_STACK_HEAD_DIM */
    float q_norm_buf[MAX_STACK_HEAD_DIM];
    float k_norm_buf[MAX_STACK_HEAD_DIM];
    float num_buf[MAX_STACK_HEAD_DIM];

    float *q_cur = (D <= MAX_STACK_HEAD_DIM) ? q_norm_buf : (float*)malloc((size_t)D * sizeof(float));
    float *k_cur = (D <= MAX_STACK_HEAD_DIM) ? k_norm_buf : (float*)malloc((size_t)D * sizeof(float));
    float *num   = (D <= MAX_STACK_HEAD_DIM) ? num_buf   : (float*)malloc((size_t)D * sizeof(float));

    for (int h = 0; h < H; h++) {
        const float *qh = q_in + (size_t)h * D;
        const float *kh = k_in + (size_t)h * D;
        const float *vh = v_in + (size_t)h * D;
        float *outh     = out  + (size_t)h * D;

        float *s0 = state->s0 + (size_t)h * D;
        float *s1 = state->s1 + (size_t)h * D * D;
        float *s2 = state->s2 + (size_t)h * D * D;
        float *k0 = &state->k0[h];
        float *k1 = state->k1 + (size_t)h * D;
        float *k2 = state->k2 + (size_t)h * D;

        /* QK normalization or direct copy with scaling */
        if (cfg->use_qk_norm) {
            apply_layer_norm(qh, q_cur, D);
            apply_layer_norm(kh, k_cur, D);
        } else {
            memcpy(q_cur, qh, (size_t)D * sizeof(float));
            memcpy(k_cur, kh, (size_t)D * sizeof(float));
        }

        /* Scale query vector by 1 / sqrt(D) */
        for (int d = 0; d < D; d++) {
            q_cur[d] *= q_scale;
        }

        /* 1. Recurrent State Update (2-way Register Tiling) */
        *k0 = decay * (*k0) + 1.0f;

        int d = 0;
        for (; d <= D - 2; d += 2) {
            float kd0 = k_cur[d + 0], kd0_sq = kd0 * kd0;
            float kd1 = k_cur[d + 1], kd1_sq = kd1 * kd1;

            s0[d + 0] = decay * s0[d + 0] + vh[d + 0];
            s0[d + 1] = decay * s0[d + 1] + vh[d + 1];

            k1[d + 0] = decay * k1[d + 0] + kd0;
            k1[d + 1] = decay * k1[d + 1] + kd1;

            k2[d + 0] = decay * k2[d + 0] + kd0_sq;
            k2[d + 1] = decay * k2[d + 1] + kd1_sq;

            float *s1_r0 = s1 + (size_t)(d + 0) * D;
            float *s1_r1 = s1 + (size_t)(d + 1) * D;
            float *s2_r0 = s2 + (size_t)(d + 0) * D;
            float *s2_r1 = s2 + (size_t)(d + 1) * D;

            for (int e = 0; e < D; e++) {
                float ve = vh[e];
                s1_r0[e] = decay * s1_r0[e] + kd0 * ve;
                s2_r0[e] = decay * s2_r0[e] + kd0_sq * ve;
                s1_r1[e] = decay * s1_r1[e] + kd1 * ve;
                s2_r1[e] = decay * s2_r1[e] + kd1_sq * ve;
            }
        }
        for (; d < D; d++) {
            float kd = k_cur[d];
            float kd2 = kd * kd;

            s0[d] = decay * s0[d] + vh[d];
            k1[d] = decay * k1[d] + kd;
            k2[d] = decay * k2[d] + kd2;

            float *s1_row = s1 + (size_t)d * D;
            float *s2_row = s2 + (size_t)d * D;

            for (int e = 0; e < D; e++) {
                s1_row[e] = decay * s1_row[e] + kd * vh[e];
                s2_row[e] = decay * s2_row[e] + kd2 * vh[e];
            }
        }

        /* 2. Output Contraction (Numerator - Tiled Accumulation) */
        for (int e = 0; e < D; e++) {
            num[e] = s0[e];
        }

        d = 0;
        for (; d <= D - 2; d += 2) {
            float q0 = q_cur[d + 0], q0_sq = 0.5f * q0 * q0;
            float q1 = q_cur[d + 1], q1_sq = 0.5f * q1 * q1;

            const float *s1_r0 = s1 + (size_t)(d + 0) * D;
            const float *s1_r1 = s1 + (size_t)(d + 1) * D;
            const float *s2_r0 = s2 + (size_t)(d + 0) * D;
            const float *s2_r1 = s2 + (size_t)(d + 1) * D;

            for (int e = 0; e < D; e++) {
                num[e] += (q0 * s1_r0[e] + q0_sq * s2_r0[e])
                        + (q1 * s1_r1[e] + q1_sq * s2_r1[e]);
            }
        }
        for (; d < D; d++) {
            float qd = q_cur[d];
            float qd2_half = 0.5f * qd * qd;
            const float *s1_row = s1 + (size_t)d * D;
            const float *s2_row = s2 + (size_t)d * D;

            for (int e = 0; e < D; e++) {
                num[e] += qd * s1_row[e] + qd2_half * s2_row[e];
            }
        }

        /* 3. Normalization Contraction (Denominator) */
        float den = *k0;
        int d_den = 0;
        for (; d_den <= D - 4; d_den += 4) {
            float q0 = q_cur[d_den + 0], q1 = q_cur[d_den + 1];
            float q2 = q_cur[d_den + 2], q3 = q_cur[d_den + 3];
            den += (q0 * k1[d_den + 0] + 0.5f * (q0 * q0) * k2[d_den + 0])
                 + (q1 * k1[d_den + 1] + 0.5f * (q1 * q1) * k2[d_den + 1])
                 + (q2 * k1[d_den + 2] + 0.5f * (q2 * q2) * k2[d_den + 2])
                 + (q3 * k1[d_den + 3] + 0.5f * (q3 * q3) * k2[d_den + 3]);
        }
        for (; d_den < D; d_den++) {
            float qd = q_cur[d_den];
            den += qd * k1[d_den] + 0.5f * (qd * qd) * k2[d_den];
        }

        if (den < eps) {
            den = eps;
        }
        float inv_den = 1.0f / den;

        for (int e = 0; e < D; e++) {
            outh[e] = num[e] * inv_den;
        }
    }

    if (D > MAX_STACK_HEAD_DIM) {
        free(q_cur);
        free(k_cur);
        free(num);
    }
}

void prime_prefill(
    const prime_config_t *cfg,
    prime_state_t *state,
    int seq_len,
    const float *q_seq,
    const float *k_seq,
    const float *v_seq,
    float *out_seq
) {
    size_t step_stride = (size_t)cfg->num_heads * cfg->head_dim;
    for (int t = 0; t < seq_len; t++) {
        const float *q_t = q_seq + (size_t)t * step_stride;
        const float *k_t = k_seq + (size_t)t * step_stride;
        const float *v_t = v_seq + (size_t)t * step_stride;
        float *out_t     = out_seq + (size_t)t * step_stride;

        prime_step(cfg, state, q_t, k_t, v_t, out_t);
    }
}
