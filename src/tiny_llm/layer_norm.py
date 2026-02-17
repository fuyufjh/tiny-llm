import mlx.core as mx

# Root Mean Square Layer Normalization
class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (x
                * mx.rsqrt(x.astype(mx.float32).square().mean(-1, keepdims=True) + self.eps) 
                * self.weight).astype(x.dtype)
