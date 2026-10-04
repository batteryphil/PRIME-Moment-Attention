/*
 * PRIME Moment Attention: Zero-Dependency C99 Native Implementation
 * =================================================================
 * Constant-state O(1) second-order recurrent attention operator.
 * Supports:
 *   - Baseline second-order Taylor polynomial attention
 *   - Gated Delta-PRIME decoupled erase/write error delta updates
 *   - Dual-state Lyapunov contractive norm projection
 *   - Second-order Harmonic Phasor RoPE
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
    cfg.use_delta_rule = 1;
    cfg.lyapunov_bound = 25.0f;
    cfg.use_harmonic_rope = 0;
    cfg.use_gdn2_decoupled = 0;
    cfg.use_differential = 0;
    cfg.diff_lambda = 0.5f;
    cfg.latent_dim = 0;
    cfg.use_rwkv7_curvature_delta = 0;
    cfg.use_symplectic_integrator = 0;
    cfg.symplectic_theta = 0.01f;
    return cfg;
}

prime_config_t prime_wave3_config(int num_heads, int head_dim, int latent_dim) {
    prime_config_t cfg = prime_default_config(num_heads, head_dim);
    cfg.use_delta_rule = 1;
    cfg.use_gdn2_decoupled = 1;
    cfg.use_differential = 1;
    cfg.diff_lambda = 0.5f;
    cfg.latent_dim = latent_dim;
    cfg.lyapunov_bound = 15.0f;
    return cfg;
}

prime_config_t prime_wave4_config(int num_heads, int head_dim, int latent_dim) {
    prime_config_t cfg = prime_wave3_config(num_heads, head_dim, latent_dim);
    cfg.use_rwkv7_curvature_delta = 1;
    cfg.use_symplectic_integrator = 1;
    cfg.symplectic_theta = 0.01f;
    return cfg;
}

float prime_state_frobenius_energy(const prime_state_t *state) {
    if (!state) return 0.0f;
    int H = state->num_heads;
    int D = state->head_dim;
    int K = (state->latent_dim > 0) ? state->latent_dim : D;
    size_t total_elements = (size_t)H * D * K;

    float energy = 0.0f;
    for (size_t i = 0; i < total_elements; i++) {
        float val1 = state->s1[i];
        float val2 = state->s2[i];
        energy += val1 * val1 + val2 * val2;
    }
    return energy;
}

prime_state_t* prime_state_create_ext(int num_heads, int head_dim, int latent_dim) {
    if (num_heads <= 0 || head_dim <= 0) {
        return NULL;
    }

    prime_state_t *state = (prime_state_t*)malloc(sizeof(prime_state_t));
    if (!state) return NULL;

    state->num_heads = num_heads;
    state->head_dim = head_dim;
    state->latent_dim = (latent_dim > 0) ? latent_dim : 0;
    int k_dim = (state->latent_dim > 0) ? state->latent_dim : head_dim;

    size_t s0_size = (size_t)num_heads * head_dim * sizeof(float);
    size_t s1_size = (size_t)num_heads * head_dim * k_dim * sizeof(float);
    size_t s2_size = (size_t)num_heads * head_dim * k_dim * sizeof(float);
    size_t k0_size = (size_t)num_heads * sizeof(float);
    size_t k1_size = (size_t)num_heads * k_dim * sizeof(float);
    size_t k2_size = (size_t)num_heads * k_dim * sizeof(float);

    state->state_bytes = s0_size + s1_size + s2_size + k0_size + k1_size + k2_size;

    state->s0 = (float*)malloc(s0_size);
    state->s1 = (float*)malloc(s1_size);
    state->s2 = (float*)malloc(s2_size);
    state->k0 = (float*)malloc(k0_size);
    state->k1 = (float*)malloc(k1_size);
    state->k2 = (float*)malloc(k2_size);

    if (state->latent_dim > 0) {
        state->w_mla_c = (float*)malloc((size_t)head_dim * k_dim * sizeof(float));
        if (state->w_mla_c) {
            float s_mla = 1.0f / sqrtf((float)head_dim);
            for (size_t i = 0; i < (size_t)head_dim * k_dim; i++) {
                float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                state->w_mla_c[i] = r * s_mla;
            }
        }
    } else {
        state->w_mla_c = NULL;
    }

    if (!state->s0 || !state->s1 || !state->s2 || !state->k0 || !state->k1 || !state->k2 ||
        (state->latent_dim > 0 && !state->w_mla_c)) {
        prime_state_free(state);
        return NULL;
    }

    prime_state_reset(state);
    return state;
}

prime_state_t* prime_state_create(int num_heads, int head_dim) {
    return prime_state_create_ext(num_heads, head_dim, 0);
}

void prime_state_reset(prime_state_t *state) {
    if (!state) return;
    int H = state->num_heads;
    int D = state->head_dim;
    int K = (state->latent_dim > 0) ? state->latent_dim : D;

    memset(state->s0, 0, (size_t)H * D * sizeof(float));
    memset(state->s1, 0, (size_t)H * D * K * sizeof(float));
    memset(state->s2, 0, (size_t)H * D * K * sizeof(float));
    memset(state->k0, 0, (size_t)H * sizeof(float));
    memset(state->k1, 0, (size_t)H * K * sizeof(float));
    memset(state->k2, 0, (size_t)H * K * sizeof(float));
}

void prime_state_free(prime_state_t *state) {
    if (!state) return;
    if (state->s0) free(state->s0);
    if (state->s1) free(state->s1);
    if (state->s2) free(state->s2);
    if (state->k0) free(state->k0);
    if (state->k1) free(state->k1);
    if (state->k2) free(state->k2);
    if (state->w_mla_c) free(state->w_mla_c);
    free(state);
}

void prime_lyapunov_project(prime_state_t *state, float bound) {
    if (!state || bound <= 0.0f) return;
    int H = state->num_heads;
    int D = state->head_dim;
    int K = (state->latent_dim > 0) ? state->latent_dim : D;
    int DK = D * K;

    for (int h = 0; h < H; h++) {
        float *s1_head = state->s1 + h * DK;
        float *s2_head = state->s2 + h * DK;

        /* S1 Frobenius norm */
        float s1_sum = 0.0f;
        for (int i = 0; i < DK; i++) s1_sum += s1_head[i] * s1_head[i];
        float s1_norm = sqrtf(s1_sum);
        if (s1_norm > bound) {
            float scale1 = bound / (s1_norm + 1e-6f);
            for (int i = 0; i < DK; i++) s1_head[i] *= scale1;
        }

        /* S2 Frobenius norm */
        float s2_sum = 0.0f;
        for (int i = 0; i < DK; i++) s2_sum += s2_head[i] * s2_head[i];
        float s2_norm = sqrtf(s2_sum);
        if (s2_norm > bound) {
            float scale2 = bound / (s2_norm + 1e-6f);
            for (int i = 0; i < DK; i++) s2_head[i] *= scale2;
        }
    }
}

void prime_apply_harmonic_rope(
    float *q, float *k, float *q2, float *k2,
    int head_dim, int pos, float base_freq
) {
    for (int j = 0; j < head_dim; j += 2) {
        float freq = 1.0f / powf(base_freq, (float)j / (float)head_dim);
        float angle1 = (float)pos * freq;
        float cos1 = cosf(angle1);
        float sin1 = sinf(angle1);

        /* 1st harmonic on (q, k) */
        float q_re = q[j], q_im = q[j + 1];
        q[j]     = q_re * cos1 - q_im * sin1;
        q[j + 1] = q_re * sin1 + q_im * cos1;

        float k_re = k[j], k_im = k[j + 1];
        k[j]     = k_re * cos1 - k_im * sin1;
        k[j + 1] = k_re * sin1 + k_im * cos1;

        /* 2nd harmonic on (q^2, k^2) */
        float angle2 = 2.0f * angle1;
        float cos2 = cosf(angle2);
        float sin2 = sinf(angle2);

        float q2_re = q2[j], q2_im = q2[j + 1];
        q2[j]     = q2_re * cos2 - q2_im * sin2;
        q2[j + 1] = q2_re * sin2 + q2_im * cos2;

        float k2_re = k2[j], k2_im = k2[j + 1];
        k2[j]     = k2_re * cos2 - k2_im * sin2;
        k2[j + 1] = k2_re * sin2 + k2_im * cos2;
    }
}

static void apply_layer_norm(const float *src, float *dst, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += src[i];
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

void prime_step_delta(
    const prime_config_t *cfg,
    prime_state_t *state,
    const float *q_in,
    const float *k_in,
    const float *v_in,
    const float *b_gate,
    const float *w_gate,
    float *out
) {
    int H = cfg->num_heads;
    int D = cfg->head_dim;
    int K = (state->latent_dim > 0) ? state->latent_dim : D;
    float decay = cfg->decay;
    float eps = cfg->eps;
    float inv_sqrt_k = 1.0f / sqrtf((float)K);

    float q_buf[MAX_STACK_HEAD_DIM];
    float k_buf[MAX_STACK_HEAD_DIM];
    float q_k[MAX_STACK_HEAD_DIM];
    float k_k[MAX_STACK_HEAD_DIM];
    float k_sq[MAX_STACK_HEAD_DIM];
    float q_sq[MAX_STACK_HEAD_DIM];
    float k_hat[MAX_STACK_HEAD_DIM];
    float k_hat2[MAX_STACK_HEAD_DIM];
    float num[MAX_STACK_HEAD_DIM];
    float error_t[MAX_STACK_HEAD_DIM];

    for (int h = 0; h < H; h++) {
        const float *qh_in = q_in + h * D;
        const float *kh_in = k_in + h * D;
        const float *vh = v_in + h * D;
        float *out_h = out + h * D;

        float *s0 = state->s0 + h * D;
        float *s1 = state->s1 + h * D * K;
        float *s2 = state->s2 + h * D * K;
        float *k0 = state->k0 + h;
        float *k1 = state->k1 + h * K;
        float *k2 = state->k2 + h * K;

        if (cfg->use_qk_norm) {
            apply_layer_norm(qh_in, q_buf, D);
            apply_layer_norm(kh_in, k_buf, D);
        } else {
            memcpy(q_buf, qh_in, D * sizeof(float));
            memcpy(k_buf, kh_in, D * sizeof(float));
        }

        /* DeepSeek MLA Latent Projection (if latent_dim > 0) */
        if (state->latent_dim > 0 && state->w_mla_c) {
            for (int m = 0; m < K; m++) {
                float q_sum = 0.0f;
                float k_sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    float w = state->w_mla_c[d * K + m];
                    q_sum += w * q_buf[d];
                    k_sum += w * k_buf[d];
                }
                q_k[m] = q_sum;
                k_k[m] = k_sum;
            }
        } else {
            memcpy(q_k, q_buf, D * sizeof(float));
            memcpy(k_k, k_buf, D * sizeof(float));
        }

        /* Scale queries and compute moments */
        float k_norm_sq = 0.0f;
        float k2_norm_sq = 0.0f;
        for (int m = 0; m < K; m++) {
            q_k[m] *= inv_sqrt_k;
            q_sq[m] = q_k[m] * q_k[m];
            k_sq[m] = k_k[m] * k_k[m];
            k_norm_sq += k_k[m] * k_k[m];
            k2_norm_sq += k_sq[m] * k_sq[m];
        }

        /* L2 Normalization on Keys (strictly bounds operator spectral radius <= 1.0) */
        float inv_k_norm = 1.0f / sqrtf(k_norm_sq + 1e-6f);
        float inv_k2_norm = 1.0f / sqrtf(k2_norm_sq + 1e-6f);
        for (int m = 0; m < K; m++) {
            k_hat[m] = k_k[m] * inv_k_norm;
            k_hat2[m] = k_sq[m] * inv_k2_norm;
        }

        /* -------------------------------------------------------------
         * Mode 1: Gated Delta Recurrence (GDN-2 or Delta-PRIME)
         * ------------------------------------------------------------- */
        if (cfg->use_delta_rule) {
            /* 1. Value prediction: v_hat = S0 + S1 @ k_hat + 0.5 * S2 @ k_hat2 */
            for (int d = 0; d < D; d++) {
                float v_hat_d = s0[d];
                const float *s1_row = s1 + d * K;
                const float *s2_row = s2 + d * K;
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                for (int m = 0; m < K; m++) {
                    sum1 += s1_row[m] * k_hat[m];
                    sum2 += s2_row[m] * k_hat2[m];
                }
                v_hat_d += sum1 + 0.5f * sum2;
                error_t[d] = vh[d] - v_hat_d;
            }

            /* 2. NVIDIA Gated DeltaNet-2 Decoupled Erase / Write */
            if (cfg->use_gdn2_decoupled) {
                float alpha_vec[MAX_STACK_HEAD_DIM];
                float b_k_hat[MAX_STACK_HEAD_DIM];
                float b_k_hat2[MAX_STACK_HEAD_DIM];
                float b_mean = 0.0f;

                for (int m = 0; m < K; m++) {
                    float b_m = b_gate ? b_gate[h * K + m] : 0.05f;
                    if (b_m < 0.0f) b_m = 0.0f;
                    if (b_m > 0.99f) b_m = 0.99f;
                    alpha_vec[m] = decay * (1.0f - b_m);
                    b_k_hat[m] = b_m * k_hat[m];
                    b_k_hat2[m] = b_m * k_hat2[m];
                    b_mean += b_m;
                }
                b_mean /= (float)K;

                for (int d = 0; d < D; d++) {
                    float w_d = w_gate ? w_gate[h * D + d] : 0.25f;
                    if (w_d < 0.0f) w_d = 0.0f;
                    float w_e_d = w_d * error_t[d];

                    s0[d] = decay * (1.0f - b_mean) * s0[d] + w_e_d * 0.1f;

                    float *s1_row = s1 + d * K;
                    float *s2_row = s2 + d * K;

                    float proj1_d = 0.0f;
                    float proj2_d = 0.0f;
                    for (int m = 0; m < K; m++) {
                        proj1_d += b_k_hat[m] * (alpha_vec[m] * s1_row[m]);
                        proj2_d += b_k_hat2[m] * (alpha_vec[m] * s2_row[m]);
                    }

                    float delta_res1 = (w_e_d - proj1_d);
                    float delta_res2 = (w_e_d - proj2_d);

                    if (cfg->use_rwkv7_curvature_delta) {
                        /* RWKV-7 Error-Correcting Curvature Update:
                         * Curvature target is (v_d * k_hat_d), predicted curvature is S2 @ k_hat2 */
                        float curv_target_d = vh[d] * k_hat[d % K];
                        float curv_pred_d = 0.0f;
                        for (int m = 0; m < K; m++) {
                            curv_pred_d += s2_row[m] * k_hat2[m];
                        }
                        float curv_error_d = curv_target_d - curv_pred_d;
                        float w_e2_d = w_d * curv_error_d;
                        delta_res2 = (w_e2_d - proj2_d);
                    }

                    for (int m = 0; m < K; m++) {
                        s1_row[m] = alpha_vec[m] * s1_row[m] + k_hat[m] * delta_res1;
                        s2_row[m] = alpha_vec[m] * s2_row[m] + k_hat2[m] * delta_res2;
                    }
                }
            } else {
                /* Standard Gated Delta-PRIME */
                for (int d = 0; d < D; d++) {
                    float b_d = b_gate ? b_gate[h * D + d] : 0.05f;
                    float w_d = w_gate ? w_gate[h * D + d] : 0.25f;
                    float erase_factor = (1.0f - b_d);
                    float gamma_b = decay * erase_factor;
                    float delta_payload = w_d * error_t[d] * 0.1f;

                    s0[d] = gamma_b * s0[d] + delta_payload;

                    float *s1_row = s1 + d * K;
                    float *s2_row = s2 + d * K;
                    for (int m = 0; m < K; m++) {
                        s1_row[m] = gamma_b * s1_row[m] + delta_payload * k_hat[m];
                        s2_row[m] = gamma_b * s2_row[m] + delta_payload * k_hat2[m];
                    }
                }
            }

            /* 2b. Symplectic Hamiltonian Phase-Space Flow (Wave 4) */
            if (cfg->use_symplectic_integrator) {
                float cos_th = cosf(cfg->symplectic_theta);
                float sin_th = sinf(cfg->symplectic_theta);
                for (int d = 0; d < D; d++) {
                    float *s1_row = s1 + d * K;
                    float *s2_row = s2 + d * K;
                    for (int m = 0; m < K; m++) {
                        float q_pos = s1_row[m];
                        float p_mom = s2_row[m];
                        s1_row[m] = cos_th * q_pos + sin_th * p_mom;
                        s2_row[m] = -sin_th * q_pos + cos_th * p_mom;
                    }
                }
            }

            /* 3. Output Readout: Microsoft Differential Taylor Attention or Standard */
            if (cfg->use_differential) {
                float lambda_val = cfg->diff_lambda;
                float diff_buf[MAX_STACK_HEAD_DIM];
                float sum_sq = 0.0f;

                for (int d = 0; d < D; d++) {
                    float term1 = 0.0f;
                    float term2 = 0.0f;
                    float noise_term1 = 0.0f;
                    float noise_term2 = 0.0f;
                    for (int m = 0; m < K; m++) {
                        float s1_val = s1[d * K + m];
                        float s2_val = s2[d * K + m];
                        float q_m = q_k[m];
                        float q_m2 = q_sq[m];
                        float q_noise = 0.5f * q_m;
                        float q_noise2 = 0.25f * q_m2;

                        term1 += q_m * s1_val;
                        term2 += q_m2 * s2_val;
                        noise_term1 += q_noise * s1_val;
                        noise_term2 += q_noise2 * s2_val;
                    }
                    float read1 = s0[d] + term1 + 0.5f * term2;
                    float read2 = s0[d] + noise_term1 + 0.5f * noise_term2;
                    float d_val = read1 - lambda_val * read2;
                    diff_buf[d] = d_val;
                    sum_sq += d_val * d_val;
                }

                /* Head-wise RMSNorm */
                float rms = 1.0f / sqrtf(sum_sq / (float)D + 1e-5f);
                for (int d = 0; d < D; d++) {
                    out_h[d] = diff_buf[d] * rms;
                }
            } else {
                for (int d = 0; d < D; d++) {
                    float term1 = 0.0f;
                    float term2 = 0.0f;
                    for (int m = 0; m < K; m++) {
                        term1 += q_k[m] * s1[d * K + m];
                        term2 += q_sq[m] * s2[d * K + m];
                    }
                    out_h[d] = s0[d] + term1 + 0.5f * term2;
                }
            }

        /* -------------------------------------------------------------
         * Mode 2: Standard Additive Moment Recurrence
         * ------------------------------------------------------------- */
        } else {
            *k0 = decay * (*k0) + 1.0f;
            for (int d = 0; d < D; d++) {
                s0[d] = decay * s0[d] + vh[d];
            }
            for (int m = 0; m < K; m++) {
                k1[m] = decay * k1[m] + k_k[m];
                k2[m] = decay * k2[m] + k_sq[m];
            }
            for (int d = 0; d < D; d++) {
                float *s1_row = s1 + d * K;
                float *s2_row = s2 + d * K;
                float v_val = vh[d];
                for (int m = 0; m < K; m++) {
                    s1_row[m] = decay * s1_row[m] + k_k[m] * v_val;
                    s2_row[m] = decay * s2_row[m] + k_sq[m] * v_val;
                }
            }

            for (int d = 0; d < D; d++) num[d] = s0[d];
            for (int d = 0; d < D; d++) {
                float term1 = 0.0f;
                float term2 = 0.0f;
                const float *s1_row = s1 + d * K;
                const float *s2_row = s2 + d * K;
                for (int m = 0; m < K; m++) {
                    term1 += q_k[m] * s1_row[m];
                    term2 += 0.5f * q_sq[m] * s2_row[m];
                }
                num[d] += term1 + term2;
            }

            float den = *k0;
            for (int m = 0; m < K; m++) {
                den += q_k[m] * k1[m] + 0.5f * q_sq[m] * k2[m];
            }
            if (den < eps) den = eps;
            float inv_den = 1.0f / den;
            for (int d = 0; d < D; d++) {
                out_h[d] = num[d] * inv_den;
            }
        }
    }

    /* 4. Dual-state Lyapunov contractive projection */
    if (cfg->lyapunov_bound > 0.0f) {
        prime_lyapunov_project(state, cfg->lyapunov_bound);
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
    prime_step_delta(cfg, state, q_in, k_in, v_in, NULL, NULL, out);
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
    int stride = cfg->num_heads * cfg->head_dim;
    for (int t = 0; t < seq_len; t++) {
        prime_step(
            cfg,
            state,
            q_seq + t * stride,
            k_seq + t * stride,
            v_seq + t * stride,
            out_seq + t * stride
        );
    }
}
