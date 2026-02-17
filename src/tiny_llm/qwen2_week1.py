import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads # grouped
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.rope = RoPE(self.head_dim, max_seq_len, theta, False)

    def __call__(
        self,
        x: mx.array, # batch_size x seq_length x hidden_size
        mask: mx.array | str | None = None,
    ) -> mx.array:
        batch_size, seq_length, _ = x.shape
        
        # Q,K,V: (batch_size, seq_length, num_(kv_)heads, head_dim)
        q = linear(x, self.wq, self.bq).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = linear(x, self.wk, self.bk).reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv, self.bv).reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        q = self.rope(q, offset=slice(0, seq_length)) # batch_size x seq_length x num_heads x head_dim
        k = self.rope(k, offset=slice(0, seq_length))

        q = q.transpose(0, 2, 1, 3) # batch_size x num_heads x seq_length x head_dim
        k = k.transpose(0, 2, 1, 3) # batch_size x num_kv_heads x seq_length x head_dim
        v = v.transpose(0, 2, 1, 3) # batch_size x num_kv_heads x seq_length x head_dim

        x = scaled_dot_product_attention_grouped(q, k, v, mask=mask) # batch_size x num_heads x seq_length x head_dim

        x = x.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)
        x = linear(x, self.wo)
        return x


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        activation = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        return linear(activation * up, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
