#include <cstdint>
#include <stdexcept>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "flash_attention.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

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
    throw std::runtime_error("FlashAttention::eval_cpu not implemented");
}

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("FlashAttention::eval_gpu not implemented");
}

}  // namespace tiny_llm_ext
