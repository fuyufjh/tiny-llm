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
    # k means which diagonal to start the mask, 
    # e.g. if L=S, k=1 to mask out the upper diagonal (exclude the main diagonal)
    #      if L=5, S=3, k=3 to mask out the upper half diagonal (3 elements in the top-right corner)
    k = S - L + 1 
    mask = mx.triu(mx.full((L, S), -mx.inf, dtype=dtype), k)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array, # batch_size x num_query_heads x seq_length_q x head_dim
    key: mx.array, # batch_size x num_heads x seq_length_kv x head_dim
    value: mx.array, # batch_size x num_heads x seq_length_kv x head_dim
    scale: float | None = None,
    mask: mx.array | str | None = None, # batch_size x num_heads x seq_length_q x seq_length_kv, or "causal"
) -> mx.array:
    num_query_heads, seq_length_q, head_dim = query.shape[-3:]
    num_heads, seq_length_kv, _ = key.shape[-3:]
    batch_dims = query.shape[:-3]

    # In the grouped attention setting, we have num_query_heads = num_heads * num_repeats, 
    # where num_repeats is how many query heads attend to the same key/value heads.
    n_repeats = num_query_heads // num_heads

    # default scale = 1/sqrt(D)
    if scale is None:
        D = query.shape[-1]
        scale = 1.0 / (D ** 0.5)
    
    # Resharp query to extract the `num_repeats`
    query = query.reshape(*batch_dims, num_heads, n_repeats, seq_length_q, head_dim)
    # Then, insert an extra dimension to key and value for broadcasting
    key = key.reshape(*batch_dims, num_heads, 1, seq_length_kv, head_dim)
    value = value.reshape(*batch_dims, num_heads, 1, seq_length_kv, head_dim)

    # Q @ K^T — compute in float32 to avoid float16 overflow (QK dot products
    # can exceed 65504 for large head_dim / large activations, e.g. Qwen2-7B)
    orig_dtype = query.dtype
    scores = query.astype(mx.float32) @ mx.swapaxes(key.astype(mx.float32), -1, -2)
    scores = scores * scale
    if isinstance(mask, mx.array):
        mask = mask.reshape(scores.shape)
        scores = scores + mask.astype(mx.float32)
    elif isinstance(mask, str) and mask == "causal":
        mask = causal_mask(seq_length_q, seq_length_kv, mx.float32)
        scores = scores + mask

    attention_weights = softmax(scores, axis=-1)
    output = (attention_weights @ value.astype(mx.float32)).astype(orig_dtype)

    # Reshape output back
    output = output.reshape(*batch_dims, num_query_heads, seq_length_q, head_dim)
    return output


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
