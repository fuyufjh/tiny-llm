#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 * Perform a quantized matrix multiplication.
 * 
 * The input matrix b is quantized with the given scales and biases, 
 * where each group of group_size values in b share the same scale and bias. 
 *
 * Input:
 *   A: M × N (float16, activations)
 *   B_quantized: K × (N/8) (uint32, packed weights)
 *   scales: K × (N/64) (float16)
 *   biases: K × (N/64) (float16)
 * 
 * Output:
 *   C: M × K (float16)
 * 
 * For each output element C[i, k]:
 *   sum = 0
 *   for each group g in 0..(N/64 - 1):
 *     scale = scales[k, g]
 *     bias = biases[k, g]
 *     
 *     # Process 64 values in the group (8 uint32 packs)
 *     for each pack p in 0..7:
 *       packed_value = B_quantized[k, g*8 + p]
 *       
 *       # Unpack 8 × 4-bit values
 *       for bit_offset in [0, 4, 8, 12, 16, 20, 24, 28]:
 *         quantized = (packed_value >> bit_offset) & 0xF
 *         b_value = quantized * scale + bias
 *         a_value = A[i, g*64 + p*8 + bit_offset/4]
 *         sum += a_value * b_value
 *   
 *   C[i, k] = sum
 * ```
 **/
mx::array quantized_matmul(const mx::array &scales,         // Input array scales
                           const mx::array &biases,         // Input array biases
                           const int group_size,            // Group size
                           const int bits,                  // Number of bits
                           const mx::array &a,              // Input array a (not quantized)
                           const mx::array &b_quantized,    // Input array b (quantized)
                           const bool transpose_b,          // Whether to transpose b
                           mx::StreamOrDevice s /* = {} */  // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class QuantizedMatmul : public mx::Primitive {
public:
    explicit QuantizedMatmul(mx::Stream stream, int group_size, int bits) : mx::Primitive(stream), group_size_(group_size), bits_(bits) {};

    /**
     * A primitive must know how to evaluate itself on the CPU/GPU
     * for the given inputs and populate the output array.
     *
     * To avoid unnecessary allocations, the evaluation function
     * is responsible for allocating space for the array.
     */
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    /** Print the primitive. */
    void print(std::ostream &os);

    /** Name of the primitive (not virtual in some MLX versions). */
    const char *name() const override { return "QuantizedMatmul"; }

    /** Equivalence check **/
    bool is_equivalent(const mx::Primitive &other) const override;

private:
    int group_size_;
    int bits_;
};

}  // namespace tiny_llm_ext
