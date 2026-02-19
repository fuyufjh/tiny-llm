import mlx.core as mx
from typing import Any

from extensions import tiny_llm_ext


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array, # K × (N/group_size) (float16)
    biases: mx.array, # K × (N/group_size) (float16)
    group_size: int,
    bits: int,
    a: mx.array, # M × N (float16, activations)
    b_quantized: mx.array, # K × (N/bits) (uint32, packed weights)
    transpose_b: bool = False,
) -> mx.array: # M × K (float16)
    *N, D = a.shape
    a = a.reshape(-1, D)
    a = mx.contiguous(a)
    b_quantized = mx.contiguous(b_quantized)
    return tiny_llm_ext.quantized_matmul(
        scales, biases, group_size, bits, a, b_quantized, transpose_b
    ).reshape(*N, -1)



def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    return quantized_matmul(
        w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
    ) + (bias if bias is not None else 0)
