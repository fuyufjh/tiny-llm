// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "tiny_llm_ext.h"
#include "axpby.h"
#include "quantized_matmul.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext::load_library, "device"_a, "path"_a);

    m.def("axpby", &tiny_llm_ext::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a, nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");
    
    m.def("quantized_matmul", &tiny_llm_ext::quantized_matmul,
          "scales"_a, "biases"_a, "group_size"_a, "bits"_a, "a"_a, "b_quantized"_a, "transpose_b"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Perform a quantized matrix multiplication.

        The input matrix b is quantized with the given scales and biases,
        where each group of group_size values in b share the same scale and bias.

        For each output element C[i, k]:
            sum = 0
            for each group g in 0..(N/64 - 1):
                scale = scales[k, g]
                bias = biases[k, g]

                # Process 64 values in the group (8 uint32 packs)
                for each pack p in 0..7:
                    packed_value = B_quantized[k, g*8 + p]

                    # Unpack 8 × 4-bit values
                    for bit_offset in [0, 4, 8, 12, 16, 20, 24, 28]:
                        quantized = (packed_value >> bit_offset) & 0xF
                        b_value = quantized * scale + bias
                        a_value = A[i, g*64 + p*8 + bit_offset/4]
                        sum += a_value * b_value

        C[i, k] = sum

        Args:
            scales (array): K × (N/64) array of float16 scales.
            biases (array): K × (N/64) array of float16 biases.
            group_size (int): Number of values sharing the same scale and bias (must be 64).
            bits (int): Number of bits per quantized value (must be 4).
            a (array): M × N array of float16 activations.
            b_quantized (array): K × (N/8) array of uint32 packed quantized weights.
            transpose_b (bool): Whether to transpose b before multiplication.

        Returns:
            array: M × K array of float16 results.
      )");
}
