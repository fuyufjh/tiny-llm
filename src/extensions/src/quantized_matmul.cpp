#include <cstdint>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "quantized_matmul.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

mx::array quantized_matmul(const mx::array &scales,
                           const mx::array &biases,
                           const int group_size,
                           const int bits,
                           const mx::array &a,
                           const mx::array &b_quantized,
                           const bool transpose_b,
                           mx::StreamOrDevice s /* = {} */
) {
    if (scales.dtype() != mx::float16 && scales.dtype() != mx::float32) {
        throw std::runtime_error("quantized_matmul: scales must be float16 or float32");
    }
    if (scales.dtype() != biases.dtype()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same dtype");
    }
    if (b_quantized.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b_quantized must be uint32");
    }
    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: a must have the same dtype as scales");
    }
    if (a.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    if (b_quantized.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: b_quantized must be a 2D array");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    if (group_size != 64) {
        throw std::runtime_error("quantized_matmul: group_size must be 64");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed (transpose_b must be true)");
    }
    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }
    if (b_quantized.shape()[0] != scales.shape()[0]) {
        throw std::runtime_error("quantized_matmul: b_quantized and scales must have the same number of rows");
    }
    if (b_quantized.shape()[1] != scales.shape()[1] * group_size / 8) {
        throw std::runtime_error("quantized_matmul: b_quantized shape is inconsistent with scales and group_size");
    }
    if (a.shape()[1] != b_quantized.shape()[1] * 8) {
        throw std::runtime_error("quantized_matmul: a columns must equal b_quantized columns * 8");
    }

    // Output shape: [M, K] where A is [M, N] and B is [K, N/8]
    auto out_shape = a.shape();
    out_shape[1] = b_quantized.shape()[0];

    return mx::array(
        /* shape = */ out_shape,
        /* dtype = */ a.dtype(),
        /* primitive = */ std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
        /* inputs = */ {scales, biases, a, b_quantized});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive CPU Backend Implementation
///////////////////////////////////////////////////////////////////////////////

template<typename T> // T can be float16_t or float
void quantized_matmul_impl(
    const mx::array &scales, const mx::array &biases,
    const mx::array &a, const mx::array &b,
    mx::array &out, int group_size, int bits, mx::Stream stream) {

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    encoder.dispatch([out_ptr = out.data<T>(), out_shape = out.shape(), out_strides = out.strides(),
                      group_size = group_size, bits = bits,
                      a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b),
                      scales = mx::array::unsafe_weak_copy(scales),
                      biases = mx::array::unsafe_weak_copy(biases)]() {

        // Dimensions:
        //   A: [M, N]  (activations, not quantized)
        //   B: [K, N/8] (weights, packed 4-bit)
        //   out: [M, K]
        int m = a.shape()[0], n = a.shape()[1], k = b.shape()[0];

        // Each group of `group_size` contiguous values in a row of B shares one scale/bias.
        // Each group occupies `group_size * bits / 32` uint32_t elements.
        const int group_per_row = n / group_size;       // number of groups per output row
        const int packs_per_item = 32 / bits;           // 4-bit values packed per uint32 (= 8)
        const int items_per_group = group_size / packs_per_item; // uint32 elements per group (= 8)

        const T *a_ptr = a.data<T>();
        const T *scales_ptr = scales.data<T>();
        const T *biases_ptr = biases.data<T>();
        const uint32_t *b_ptr = b.data<uint32_t>();
        const uint32_t pack_mask = (1u << bits) - 1u; // 0xF for 4 bits

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0.0f; // always accumulate in float for better precision

                for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                    // Scale and bias are indexed as [j, group_idx] in a flattened [K, group_per_row] layout
                    int64_t scales_idx = mx::elem_to_loc(j * group_per_row + group_idx, scales.shape(), scales.strides());
                    int64_t biases_idx = mx::elem_to_loc(j * group_per_row + group_idx, biases.shape(), biases.strides());
                    T scale = static_cast<T>(scales_ptr[scales_idx]);
                    T bias  = static_cast<T>(biases_ptr[biases_idx]);

                    // Starting index in A for this group: row i, column group_idx * group_size
                    int64_t a_idx = mx::elem_to_loc(i * n + group_idx * group_size, a.shape(), a.strides());
                    // Starting index in B for this group: row j, pack (group_idx * group_size) / packs_per_item
                    int64_t b_idx = mx::elem_to_loc((j * n + group_idx * group_size) / packs_per_item, b.shape(), b.strides());

                    for (int item_idx = 0; item_idx < items_per_group; item_idx++) {
                        uint32_t b_val = b_ptr[b_idx];
                        // Reinterpret the uint32 as 4 bytes for easier nibble extraction
                        const uint8_t *b_bytes = reinterpret_cast<const uint8_t *>(&b_val);

                        for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                            // Extract one 4-bit quantized value:
                            //   byte index: pack_idx / 2
                            //   nibble:     low bits when pack_idx is even, high bits when odd
                            uint8_t item_val = (b_bytes[pack_idx / 2] >> ((pack_idx % 2) * bits)) & pack_mask;
                            T a_val  = static_cast<T>(a_ptr[a_idx]);
                            T b_real = static_cast<T>(item_val) * scale + bias;
                            sum += static_cast<float>(a_val) * static_cast<float>(b_real);
                            a_idx += 1;
                        }
                        b_idx += 1;
                    }
                }

                int64_t out_idx = mx::elem_to_loc(i * k + j, out_shape, out_strides);
                out_ptr[out_idx] = static_cast<T>(sum);
            }
        }
    });
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a      = inputs[2];
    auto &b      = inputs[3];
    auto &out    = outputs[0];

    switch (a.dtype()) {
        case mx::float16:
            quantized_matmul_impl<float16_t>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        case mx::float32:
            quantized_matmul_impl<float>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        default:
            throw std::runtime_error("QuantizedMatmul: unsupported dtype");
    }
}

#ifdef _METAL_

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a      = inputs[2];
    auto &b      = inputs[3];
    auto &out    = outputs[0];

    auto &s = stream();
    auto &device = mx::metal::device(s.device);

    out.set_data(mx::allocator::malloc(out.nbytes()));

    int M = a.shape()[0];
    int N = a.shape()[1];
    int K = b.shape()[0];

    // Kernel name matches instantiate_kernel() calls in quantized_matmul.metal
    const char *kname = nullptr;
    if (out.dtype() == mx::float32) {
        kname = "quantized_matmul_float";
    } else if (out.dtype() == mx::float16) {
        kname = "quantized_matmul_half";
    } else {
        throw std::runtime_error("QuantizedMatmul: unsupported dtype for GPU");
    }
    auto kernel = device.get_kernel(kname, device.get_library("tiny_llm_ext"));

    auto &compute_encoder = device.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Buffers match [[buffer(N)]] annotations in the Metal kernel
    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    // One thread per output element
    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    size_t nelem = size_t(M) * K;
    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
    MTL::Size grid_dims = MTL::Size(nelem, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else  // Metal is not available

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("QuantizedMatmul has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

void QuantizedMatmul::print(std::ostream &os) {
    os << name() << "(group_size=" << group_size_ << ", bits=" << bits_ << ")";
}

bool QuantizedMatmul::is_equivalent(const mx::Primitive &other) const {
    const QuantizedMatmul &r_other = static_cast<const QuantizedMatmul &>(other);
    return group_size_ == r_other.group_size_ && bits_ == r_other.bits_;
}

}  // namespace tiny_llm_ext
