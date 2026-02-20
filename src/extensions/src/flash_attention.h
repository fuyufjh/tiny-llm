#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

// Q: [N, L, E]
// K: [N_KV, S, E]
// V: [N_KV, S, E]
// mask: [N, L, S]
// output: [N, L, E]
mx::array flash_attention(const mx::array &q,
                          const mx::array &k,
                          const mx::array &v,
                          const mx::array &mask,
                          const float scale,
                          const int num_kv_heads,
                          const int num_heads,
                          mx::StreamOrDevice s = {});

class FlashAttention : public mx::Primitive {
public:
    explicit FlashAttention(mx::Stream stream, const float scale, const int num_kv_heads, const int num_heads)
        : mx::Primitive(stream), scale_(scale), num_kv_heads_(num_kv_heads), num_heads_(num_heads) {}

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override {
        throw std::runtime_error("FlashAttention has no vmap implementation.");
    }

    const char *name() const override { return "FlashAttention"; }

private:
    float scale_;
    int num_kv_heads_;
    int num_heads_;
};

}  // namespace tiny_llm_ext
