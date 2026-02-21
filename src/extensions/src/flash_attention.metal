#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Flash Attention v2 — GPU kernel
//
// Grid  : (N, Tr) threadgroups   — one threadgroup per (head, query-block) pair
// Threads: Br threads per threadgroup (one per query row in the block)
//          mapped as: p = simd_gid * 32 + simd_lid
//
// Compile-time constants (kernel specialised for E ≤ 128, Bc ≤ 32):
//   E_MAX = 128  (head dimension upper bound)
//   BC_MAX = 32  (KV block size upper bound)
//
// Threadgroup memory (16 KB, reused for K then V):
//   kv_smem[BC_MAX * E_MAX]
//
// Per-thread registers:
//   q_reg[E_MAX]   — one Q row
//   o_reg[E_MAX]   — output accumulator
//   s_row[BC_MAX]  — attention scores for one query row
//   p_row[BC_MAX]  — softmax numerators
//   l_i, m_i       — running sum and max

[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int* mask_shape [[buffer(5)]],
    constant const int64_t* mask_strides [[buffer(6)]],
    device const int &N [[buffer(7)]],
    device const int &L [[buffer(8)]],
    device const int &S [[buffer(9)]],
    device const int &E [[buffer(10)]],
    device const int &num_kv_heads [[buffer(11)]],
    device const int &num_heads [[buffer(12)]],
    device const float &scale [[buffer(13)]],
    device const int &Br [[buffer(14)]],
    device const int &Bc [[buffer(15)]],
    [[maybe_unused]] device const int &Tr [[buffer(16)]],
    device const int &Tc [[buffer(17)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    // ── Thread / block identity ──────────────────────────────────────────────
    const int n = group_id.x;   // query head index  ∈ [0, N)
    const int i = group_id.y;   // query block index ∈ [0, Tr)
    const int p = simd_gid * 32 + simd_lid;  // query row within block ∈ [0, Br)

    // GQA: map query head → shared KV head
    const int G    = num_heads / num_kv_heads;
    const int n_kv = (n / num_heads) * num_kv_heads + (n % num_heads) / G;

    // Absolute query position handled by this thread
    const int qs    = i * Br;
    const int q_row = qs + p;   // may exceed L for out-of-bounds threads

    // ── Threadgroup SRAM — reused for K then V ───────────────────────────────
    threadgroup float kv_smem[32 * 128];  // BC_MAX * E_MAX

    // ── Per-thread registers ─────────────────────────────────────────────────
    float q_reg[128];   // Q[n][q_row][:]
    float o_reg[128];   // output accumulator
    float l_i = 0.0f;
    float m_i = -INFINITY;

    for (int h = 0; h < E; h++) o_reg[h] = 0.0f;

    // Load Q row into registers (skip for out-of-bounds threads)
    if (q_row < L) {
        const int base = (n * L + q_row) * E;
        for (int h = 0; h < E; h++) q_reg[h] = q[base + h];
    }

    // ── Main loop over KV blocks ─────────────────────────────────────────────
    for (int j = 0; j < Tc; j++) {
        const int ks = j * Bc;
        const int ke = min(ks + Bc, S);
        const int bc = ke - ks;  // actual KV columns in this block

        // ── Phase 1: load K_j into threadgroup memory ────────────────────────
        // Thread p loads row p of K_j  (safe even when q_row >= L)
        if (p < bc) {
            const int base = (n_kv * S + ks + p) * E;
            for (int h = 0; h < E; h++) kv_smem[p * E + h] = k[base + h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // K ready

        // ── Phase 2: S[p][c] = scale * dot(Q[p], K[c]) + mask[p][c] ─────────
        float s_row[32];
        if (q_row < L) {
            for (int c = 0; c < bc; c++) {
                float dot = 0.0f;
                for (int h = 0; h < E; h++) dot += q_reg[h] * kv_smem[c * E + h];
                int64_t mask_off = (int64_t)n * mask_strides[0]
                                 + (int64_t)q_row * mask_strides[1]
                                 + (int64_t)(ks + c) * mask_strides[2];
                s_row[c] = scale * dot + mask[mask_off];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // K reads done → safe to overwrite with V

        // ── Phase 2b: online softmax update ──────────────────────────────────
        float p_row[32];
        if (q_row < L) {
            float row_max = -INFINITY;
            for (int c = 0; c < bc; c++) row_max = max(row_max, s_row[c]);

            const float m_new = max(m_i, row_max);
            const float alpha  = exp(m_i - m_new);  // ∈ [0,1]; 0 when m_i = -inf

            float rowsum = 0.0f;
            for (int c = 0; c < bc; c++) {
                p_row[c] = exp(s_row[c] - m_new);
                rowsum  += p_row[c];
            }

            l_i = alpha * l_i + rowsum;
            m_i = m_new;

            // Rescale existing O accumulator
            for (int h = 0; h < E; h++) o_reg[h] *= alpha;
        }

        // ── Phase 3: load V_j into threadgroup memory (overwrite K slot) ─────
        if (p < bc) {
            const int base = (n_kv * S + ks + p) * E;
            for (int h = 0; h < E; h++) kv_smem[p * E + h] = v[base + h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // V ready

        // ── Phase 4: O[p] += P̃[p] @ V_j ─────────────────────────────────────
        if (q_row < L) {
            for (int c = 0; c < bc; c++) {
                const float pv = p_row[c];
                for (int h = 0; h < E; h++) o_reg[h] += pv * kv_smem[c * E + h];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // V reads done → safe to overwrite with K next iter
    }

    // ── Normalize and write output ───────────────────────────────────────────
    if (q_row < L) {
        const float inv_l = 1.0f / l_i;
        const int   base  = (n * L + q_row) * E;
        for (int h = 0; h < E; h++) out[base + h] = o_reg[h] * inv_l;
    }
}
