import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # default scale = 1/sqrt(D)
    if scale is None:
        D = query.shape[-1]
        scale = 1.0 / (D ** 0.5)

    # Q @ K^T
    scores = query @ mx.swapaxes(key, -1, -2)
    scores = scores * scale
    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ value
    return output


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array, # (num_heads * head_dim) x hidden_size
        wk: mx.array, # (num_heads * head_dim) x hidden_size
        wv: mx.array, # (num_heads * head_dim) x hidden_size
        wo: mx.array, # hidden_size x (num_heads * head_dim)
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array, # batch_size x head_dim x seq_length_q x hidden_size
        key: mx.array, # batch_size x head_dim x seq_length_k x hidden_size
        value: mx.array, # batch_size x head_dim x seq_length_k x hidden_size
        mask: mx.array | None = None, # batch_size x num_heads x seq_length_q x seq_length_k
    ) -> mx.array:
        batch_size, seq_length, _ = query.shape

        # Linear projections
        # (batch_size, seq_length, hidden_size)  # the raw result of linear projection
        # reshape -> (batch_size, seq_length, num_heads, head_dim)  # divide hidden_size into num_heads heads
        # transpose -> (batch_size, num_heads, seq_length, head_dim)  # Move batch_size and num_heads to the front for attention computation
        q = linear(query, self.wq).reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = linear(key, self.wk).reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = linear(value, self.wv).reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attention_output = scaled_dot_product_attention_simple(q, k, v, mask=mask)

        # Concatenate heads and final linear projection
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)
        output = linear(attention_output, self.wo)
        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
