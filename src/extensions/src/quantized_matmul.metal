#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

// Quantized matrix multiplication kernel.
//
// Each thread computes one output element out[i, k].
//
// Dimensions (row-contiguous):
//   a:      [M, N]      float (activations)
//   b:      [K, N/8]    uint32 (4-bit packed weights, 8 values per uint32)
//   scales: [K, N/G]    float (one scale per group)
//   biases: [K, N/G]    float (one bias per group)
//   out:    [M, K]      float
//
// where G = group_size (typically 64).
template <typename T>
[[kernel]] void quantized_matmul(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {

    if (index >= uint(M * K)) return;

    const int i = int(index) / K; // output row
    const int k = int(index) % K; // output col

    // Constants
    const int group_size = 64; // values per group sharing scale/bias
    const int packs_per_item = 8; // 4-bit values per uint32
    const int items_per_group = group_size / packs_per_item; // uint32s per group
    const int group_per_row = N / group_size; // groups along N dimension

    float sum = 0.0f;
    for (int g = 0; g < group_per_row; g++) {
        T scale = T(scales[k * group_per_row + g]);
        T bias = T(biases[k * group_per_row + g]);

        int a_base = i * N + g * group_size;
        int b_base = k * N / packs_per_item + g * items_per_group;

        for (int item = 0; item < items_per_group; item++) {
            uint b_val = b[b_base + item];

            #pragma clang loop unroll(full)
            for (int p = 0; p < packs_per_item; p++) {
                uint q = (b_val >> (p * 4)) & 0xFu; // dequantize 4-bit value
                T b_real = T(q) * scale + bias;
                T a_val = T(a[a_base + item * packs_per_item + p]);
                sum += a_val * b_real;
            }
        }
    }

    out[i * K + k] = T(sum);
}

instantiate_kernel("quantized_matmul_float", quantized_matmul, float)
instantiate_kernel("quantized_matmul_half", quantized_matmul, half)
