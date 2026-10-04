/*
 * PRIME 3-Tier Cognitive Hierarchy Header
 * ========================================
 * Unifies:
 *   - Tier 1: Sliding-Window Working Memory (W=256 ring buffer, 64 KB)
 *   - Tier 2: Episodic Memory (Google Titans GTRM 64 KB Surprise-Momentum)
 *   - Tier 3: Symbolic Invariant Core (Buckingham Pi dimensional anchor)
 *
 * Guarantees 100% exact verbatim recall on immediate tokens while keeping
 * total system memory strictly O(1) constant (flat ~195 KB forever).
 */

#ifndef PRIME_TIER_H
#define PRIME_TIER_H

#include <stddef.h>
#include "prime_gtrm.h"
#include "prime_buckingham.h"
#include "prime_moment.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_invariants;
    dim5_t target_dim;
    size_t state_bytes;
} prime_invariant_core_t;

typedef struct {
    int d_model;
    int window_size;        /* e.g. 256 tokens */
    int current_len;        /* current valid tokens in window (<= window_size) */
    int write_ptr;          /* ring buffer head [0, window_size-1] */

    /* Tier 1: Working Memory Ring Buffer (W x D float32) */
    float *window_buf;      /* [window_size * d_model] */

    /* Tier 2: Episodic Memory (Google Titans GTRM 64 KB) */
    prime_gtrm_t *gtrm;

    /* Tier 3: Invariant Core (Buckingham Pi dimensional constraint) */
    prime_invariant_core_t invariant_core;

    /* Telemetry: Total memory footprint across all 3 tiers */
    size_t total_memory_bytes;
} prime_cognitive_engine_t;

/* Allocate and initialize 3-Tier Cognitive Engine */
prime_cognitive_engine_t* prime_cognitive_create(int d_model, int window_size);

/* Step single token through the 3-tier hierarchy */
void prime_cognitive_step(
    prime_cognitive_engine_t *engine,
    const float *x_in,
    float *out
);

/* Query exact verbatim similarity across the Tier 1 working memory window */
float prime_cognitive_window_recall(
    const prime_cognitive_engine_t *engine,
    const float *query_needle
);

/* Free 3-tier engine */
void prime_cognitive_free(prime_cognitive_engine_t *engine);

#ifdef __cplusplus
}
#endif

#endif /* PRIME_TIER_H */
