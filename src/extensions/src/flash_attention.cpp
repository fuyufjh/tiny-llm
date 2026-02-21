#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "flash_attention.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

/** Flash Attention v2
 *
 * q: [N, L, H] (float32)
 * k: [N_KV, S, H] (float32)
 * v: [N_KV, S, H] (float32)
 * mask: [N, L, S] (float32)
 * output: [N, L, H] (float32)
 * 
 * where
 * N: batch_size * num_heads (number of query heads across all batches)
 * L: sequence length of query
 * S: sequence length of key/value
 * H: head dimension
 * N_KV: batch_size * num_kv_heads, typically N_KV = N / num_heads * num_kv_heads
 * num_heads: number of attention heads
 * num_kv_heads: number of key/value heads
 */
mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          const float scale, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s) {
    if (q.dtype() != mx::float32 || k.dtype() != mx::float32 || v.dtype() != mx::float32 || mask.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: all input arrays must be float32");
    }
    if (q.shape().size() != 3 || k.shape().size() != 3 || v.shape().size() != 3) {
        throw std::runtime_error("flash_attention: all input arrays must be 3D");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }
    if (mask.shape().size() != 3) {
        throw std::runtime_error("flash_attention: mask must be 3D");
    }
    if (q.shape()[0] % num_heads != 0) {
        throw std::runtime_error("flash_attention: q.shape[0] must be divisible by num_heads");
    }
    if (k.shape()[0] % num_kv_heads != 0 || v.shape()[0] % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: k.shape[0] and v.shape[0] must be divisible by num_kv_heads");
    }
    if (q.shape()[2] != k.shape()[2] || q.shape()[2] != v.shape()[2]) {
        throw std::runtime_error("flash_attention: q.shape[2] must be equal to k.shape[2] and v.shape[2]");
    }
    if (q.shape()[0] / num_heads != k.shape()[0] / num_kv_heads) {
        throw std::runtime_error("flash_attention: number of heads mismatch");
    }
    if (k.shape()[1] != v.shape()[1]) {
        throw std::runtime_error("flash_attention: k.shape[1] must be equal to v.shape[1]");
    }
    if (mask.shape()[0] != q.shape()[0] || mask.shape()[1] != q.shape()[1] || mask.shape()[2] != k.shape()[1]) {
        throw std::runtime_error("flash_attention: mask must be broadcastable to q, k, v");
    }

    return mx::array(q.shape(), mx::float32,
                     std::make_shared<FlashAttention>(to_stream(s), scale, num_kv_heads, num_heads), {q, k, v, mask});
}

void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
auto &q = inputs[0];
    auto &k = inputs[1];
    auto &v = inputs[2];
    auto &mask = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);

    if (!q.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q must be contiguous");
    }
    if (!k.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: k must be contiguous");
    }
    if (!v.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: v must be contiguous");
    }

    // Launch the CPU kernel
    encoder.dispatch([out_ptr = out.data<float>(), out_shape = out.shape(), q = mx::array::unsafe_weak_copy(q),
                      k = mx::array::unsafe_weak_copy(k), v = mx::array::unsafe_weak_copy(v),
                      mask = mx::array::unsafe_weak_copy(mask), num_heads = num_heads_, num_kv_heads = num_kv_heads_,
                      scale = scale_]() {
        const float* q_ptr    = q.data<float>();
        const float* k_ptr    = k.data<float>();
        const float* v_ptr    = v.data<float>();
        const float* mask_ptr = mask.data<float>();

        const int N = q.shape()[0];
        const int L = q.shape()[1];
        const int H = q.shape()[2];
        const int S = k.shape()[1];
        const int G = num_heads / num_kv_heads;  // GQA group size

        constexpr int B_r = 32;  // query block size
        constexpr int B_c = 32;  // key/value block size

        const int T_r = (L + B_r - 1) / B_r;
        const int T_c = (S + B_c - 1) / B_c;

        // Temporary buffers (SRAM analogue)
        std::vector<float> S_buf(B_r * B_c);
        std::vector<float> P_buf(B_r * B_c);
        std::vector<float> O_buf(B_r * H);
        std::vector<float> l_buf(B_r);
        std::vector<float> m_buf(B_r);

        for (int n = 0; n < N; n++) {
            // GQA: map query head n to its shared KV head
            int n_kv = (n / num_heads) * num_kv_heads + (n % num_heads) / G;

            const float* Qn    = q_ptr    + n    * L * H;
            const float* Kn_kv = k_ptr    + n_kv * S * H;
            const float* Vn_kv = v_ptr    + n_kv * S * H;
            const float* Mn    = mask_ptr + n    * L * S;
            float*       On    = out_ptr  + n    * L * H;

            for (int i = 0; i < T_r; i++) {
                const int qs = i * B_r;
                const int qe = std::min(qs + B_r, L);
                const int br = qe - qs;  // actual rows in this block

                std::fill(O_buf.begin(), O_buf.begin() + br * H, 0.0f);
                std::fill(l_buf.begin(), l_buf.begin() + br, 0.0f);
                std::fill(m_buf.begin(), m_buf.begin() + br, -std::numeric_limits<float>::infinity());

                for (int j = 0; j < T_c; j++) {
                    const int ks = j * B_c;
                    const int ke = std::min(ks + B_c, S);
                    const int bc = ke - ks;  // actual cols in this block

                    // S_buf[p][c] = scale * dot(Q[qs+p], K[ks+c]) + Mask[qs+p][ks+c]
                    for (int p = 0; p < br; p++) {
                        const float* qrow = Qn + (qs + p) * H;
                        for (int c = 0; c < bc; c++) {
                            const float* krow = Kn_kv + (ks + c) * H;
                            float dot = 0.0f;
                            for (int h = 0; h < H; h++) {
                                dot += qrow[h] * krow[h];
                            }
                            S_buf[p * B_c + c] = scale * dot + Mn[(qs + p) * S + (ks + c)];
                        }
                    }

                    // Online softmax update and output accumulation
                    for (int p = 0; p < br; p++) {
                        // Row max of current block
                        float row_max = -std::numeric_limits<float>::infinity();
                        for (int c = 0; c < bc; c++) {
                            row_max = std::max(row_max, S_buf[p * B_c + c]);
                        }

                        const float m_new = std::max(m_buf[p], row_max);
                        // alpha = exp(m_old - m_new) in [0, 1]; naturally 0 when m_old = -inf
                        const float alpha = std::exp(m_buf[p] - m_new);

                        // P̃[p][c] = exp(S[p][c] - m_new)
                        float rowsum = 0.0f;
                        for (int c = 0; c < bc; c++) {
                            P_buf[p * B_c + c] = std::exp(S_buf[p * B_c + c] - m_new);
                            rowsum += P_buf[p * B_c + c];
                        }

                        // Update running statistics
                        l_buf[p] = alpha * l_buf[p] + rowsum;
                        m_buf[p] = m_new;

                        // O[p] = alpha * O[p] + P̃[p] @ V_j
                        float* op = O_buf.data() + p * H;
                        for (int h = 0; h < H; h++) {
                            float pv = 0.0f;
                            for (int c = 0; c < bc; c++) {
                                pv += P_buf[p * B_c + c] * Vn_kv[(ks + c) * H + h];
                            }
                            op[h] = alpha * op[h] + pv;
                        }
                    }
                }

                // Normalize by l and write to output HBM
                for (int p = 0; p < br; p++) {
                    const float inv_l = 1.0f / l_buf[p];
                    for (int h = 0; h < H; h++) {
                        On[(qs + p) * H + h] = O_buf[p * H + h] * inv_l;
                    }
                }
            }
        }

    });
}

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &q    = inputs[0];
    auto &k    = inputs[1];
    auto &v    = inputs[2];
    auto &mask = inputs[3];
    auto &out  = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    const int N  = q.shape()[0];
    const int L  = q.shape()[1];
    const int E  = q.shape()[2];
    const int S  = k.shape()[1];
    const int Br = 32;
    const int Bc = 32;
    const int Tr = (L + Br - 1) / Br;
    const int Tc = (S + Bc - 1) / Bc;

    auto &s      = stream();
    auto &d      = mx::metal::device(s.device);
    auto  kernel = d.get_kernel("flash_attention_f32_e128", d.get_library("tiny_llm_ext"));
    auto &enc    = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(kernel);

    // Buffers 0–4: input/output arrays
    enc.set_input_array(q,    0);
    enc.set_input_array(k,    1);
    enc.set_input_array(v,    2);
    enc.set_input_array(mask, 3);
    enc.set_output_array(out, 4);

    // Buffers 5–6: mask shape and strides (for broadcasting support)
    enc.set_vector_bytes(mask.shape(),   5);
    enc.set_vector_bytes(mask.strides(), 6);

    // Buffers 7–17: scalar parameters
    enc.set_bytes(N,            7);
    enc.set_bytes(L,            8);
    enc.set_bytes(S,            9);
    enc.set_bytes(E,           10);
    enc.set_bytes(num_kv_heads_, 11);
    enc.set_bytes(num_heads_,    12);
    enc.set_bytes(scale_,        13);
    enc.set_bytes(Br,           14);
    enc.set_bytes(Bc,           15);
    enc.set_bytes(Tr,           16);
    enc.set_bytes(Tc,           17);

    // Grid: N * Br total threads in x (= N threadgroups × Br threads each)
    //       Tr total threads in y  (= Tr threadgroups × 1 thread each)
    // Threadgroup: Br threads (= 1 SIMD group of 32 when Br = 32)
    MTL::Size grid_dims  = MTL::Size(N * Br, Tr, 1);
    MTL::Size group_dims = MTL::Size(Br, 1, 1);
    enc.dispatch_threads(grid_dims, group_dims);
}

}  // namespace tiny_llm_ext
