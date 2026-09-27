#include "prime_lie_su2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

prime_su2_engine_t* prime_su2_create(int dim) {
    if (dim <= 0) return NULL;

    /* Ensure multiple of 4 */
    int rounded_dim = ((dim + 3) / 4) * 4;
    int num_blocks = rounded_dim / 4;

    prime_su2_engine_t *engine = (prime_su2_engine_t*)malloc(sizeof(prime_su2_engine_t));
    if (!engine) return NULL;

    engine->dim = rounded_dim;
    engine->num_blocks = num_blocks;

    engine->thetas = (float*)calloc((size_t)num_blocks * 3, sizeof(float));
    engine->rotations_4x4 = (float*)calloc((size_t)num_blocks * 16, sizeof(float));

    if (!engine->thetas || !engine->rotations_4x4) {
        prime_su2_free(engine);
        return NULL;
    }

    engine->state_bytes = sizeof(prime_su2_engine_t) +
                          (size_t)num_blocks * (3 + 16) * sizeof(float);

    /* Initialize each block with identity */
    for (int b = 0; b < num_blocks; b++) {
        prime_su2_set_block_generators(engine, b, 0.0f, 0.0f, 0.0f);
    }

    return engine;
}

void prime_su2_free(prime_su2_engine_t *engine) {
    if (!engine) return;
    if (engine->thetas) free(engine->thetas);
    if (engine->rotations_4x4) free(engine->rotations_4x4);
    free(engine);
}

void prime_su2_set_block_generators(
    prime_su2_engine_t *engine,
    int block_idx,
    float t1,
    float t2,
    float t3
) {
    if (!engine || block_idx < 0 || block_idx >= engine->num_blocks) return;

    float *block_thetas = engine->thetas + (size_t)block_idx * 3;
    block_thetas[0] = t1;
    block_thetas[1] = t2;
    block_thetas[2] = t3;

    /* Compute Lie algebra exponential map to unit quaternion */
    float norm = sqrtf(t1 * t1 + t2 * t2 + t3 * t3);
    float w, x, y, z;

    if (norm > 1e-7f) {
        w = cosf(norm);
        float s = sinf(norm) / norm;
        x = t1 * s;
        y = t2 * s;
        z = t3 * s;
    } else {
        w = 1.0f;
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }

    /* Build 4x4 orthogonal matrix R in SO(4) isomorphic to left-quaternion multiplication */
    float *R = engine->rotations_4x4 + (size_t)block_idx * 16;

    R[0]  =  w; R[1]  = -x; R[2]  = -y; R[3]  = -z;
    R[4]  =  x; R[5]  =  w; R[6]  = -z; R[7]  =  y;
    R[8]  =  y; R[9]  =  z; R[10] =  w; R[11] = -x;
    R[12] =  z; R[13] = -y; R[14] =  x; R[15] =  w;
}

void prime_su2_init_harmonic(prime_su2_engine_t *engine, float base_frequency) {
    if (!engine) return;

    for (int b = 0; b < engine->num_blocks; b++) {
        float angle = (float)b * 0.15f;
        float t1 = base_frequency * cosf(angle);
        float t2 = base_frequency * sinf(angle * 1.3f);
        float t3 = base_frequency * cosf(angle * 0.7f);
        prime_su2_set_block_generators(engine, b, t1, t2, t3);
    }
}

void prime_su2_apply(
    const prime_su2_engine_t *engine,
    const float *state_in,
    float *state_out
) {
    if (!engine || !state_in || !state_out) return;

    int B = engine->num_blocks;
    for (int b = 0; b < B; b++) {
        const float *R = engine->rotations_4x4 + (size_t)b * 16;
        const float *v = state_in + (size_t)b * 4;
        float *out = state_out + (size_t)b * 4;

        float v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];

        out[0] = R[0]*v0 + R[1]*v1 + R[2]*v2 + R[3]*v3;
        out[1] = R[4]*v0 + R[5]*v1 + R[6]*v2 + R[7]*v3;
        out[2] = R[8]*v0 + R[9]*v1 + R[10]*v2 + R[11]*v3;
        out[3] = R[12]*v0 + R[13]*v1 + R[14]*v2 + R[15]*v3;
    }
}

void prime_su2_step(
    const prime_su2_engine_t *engine,
    const float *state_prev,
    const float *delta_in,
    float *state_next
) {
    if (!engine || !state_prev || !state_next) return;

    prime_su2_apply(engine, state_prev, state_next);

    if (delta_in) {
        int D = engine->dim;
        for (int i = 0; i < D; i++) {
            state_next[i] += delta_in[i];
        }
    }
}

float prime_su2_norm(const float *vec, int dim) {
    if (!vec || dim <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += (double)vec[i] * (double)vec[i];
    }
    return (float)sqrt(sum);
}
