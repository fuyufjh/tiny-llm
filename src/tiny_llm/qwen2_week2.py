import mlx.core as mx

from tiny_llm.quantize import quantized_linear
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache
from extensions import tiny_llm_ext


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
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
        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        x: mx.array, # batch_size x seq_length x hidden_size
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        batch_size, seq_length, _ = x.shape
        
        # Q,K,V: (batch_size, seq_length, num_(kv_)heads, head_dim)
        q = quantized_linear(x, self.wq, self.bq).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = quantized_linear(x, self.wk, self.bk).reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = quantized_linear(x, self.wv, self.bv).reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        offset_slices = [slice(offset, offset + seq_length) for offset in offsets]
        q = self.rope(q, offset=offset_slices).astype(mx.float16) # batch_size x seq_length x num_heads x head_dim
        k = self.rope(k, offset=offset_slices).astype(mx.float16)

        # Update the cache and fetch the updated full key and value
        k, v, _, mask = cache.update_and_fetch(k, v, mask_length=seq_length, mask=mask)

        if self.use_flash_attention:
            S_total = k.shape[1]
            N    = batch_size * self.num_heads
            N_kv = batch_size * self.num_kv_heads
            scale = 1.0 / (self.head_dim ** 0.5)

            # Reshape: [batch, seq, heads, dim] → [batch*heads, seq, dim]
            # q/k/v are already float16 from RoPE — no cast needed
            q_fa = q.transpose(0, 2, 1, 3).reshape(N, seq_length, self.head_dim)
            k_fa = k.transpose(0, 2, 1, 3).reshape(N_kv, S_total, self.head_dim)
            v_fa = v.transpose(0, 2, 1, 3).reshape(N_kv, S_total, self.head_dim)

            # Build additive mask [1, seq_length, S_total]
            if isinstance(mask, str) and mask == "causal":
                # q absolute positions: [S_total - seq_length, S_total)
                q_idx = mx.arange(S_total - seq_length, S_total)  # [L]
                k_idx = mx.arange(S_total)                         # [S_total]
                # causal: key must not be in the future relative to the query
                mask_fa = mx.where(
                    k_idx[None, :] <= q_idx[:, None],
                    mx.zeros((seq_length, S_total), dtype=mx.float32),
                    mx.full((seq_length, S_total), float("-inf"), dtype=mx.float32),
                )[None]  # [1, seq_length, S_total]
            elif mask is None:
                mask_fa = mx.zeros((1, seq_length, S_total), dtype=mx.float32)
            else:
                mask_fa = mask
                if mask_fa.ndim == 4:
                    mask_fa = mask_fa.squeeze(1)  # [B, 1, L, S] → [B, L, S]
                mask_fa = mask_fa.astype(mx.float32)

            # Broadcast mask to [N, seq_length, S_total] (zero-copy via strides)
            mask_fa = mx.broadcast_to(mask_fa, (N, seq_length, S_total))

            x = tiny_llm_ext.flash_attention(
                q_fa, k_fa, v_fa, mask_fa,
                scale, self.num_kv_heads, self.num_heads,
            )
            # x: [N, seq_length, head_dim] → [batch_size, seq_length, hidden_size]
            # output is already float16 — no cast needed
            x = x.reshape(batch_size, self.num_heads, seq_length, self.head_dim)
            x = x.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)
        else:
            q = q.transpose(0, 2, 1, 3) # batch_size x num_heads x seq_length x head_dim
            k = k.transpose(0, 2, 1, 3) # batch_size x num_kv_heads x seq_length x head_dim
            v = v.transpose(0, 2, 1, 3) # batch_size x num_kv_heads x seq_length x head_dim

            x = scaled_dot_product_attention_grouped(q, k, v, mask=mask) # batch_size x num_heads x seq_length x head_dim

            x = x.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)
        x = quantized_linear(x, self.wo)
        return x


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        activation = silu(quantized_linear(x, self.w_gate))
        up = quantized_linear(x, self.w_up)
        return quantized_linear(activation * up, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.mha = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, weight=w_input_layernorm)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, weight=w_post_attention_layernorm)


    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.mha(self.input_layernorm(x), [offset,], cache, mask)
        x = x + r
        r = self.mlp(self.post_attention_layernorm(x))
        return x + r


class Qwen2ModelWeek2:
    # input
    # | (tokens: N..)
    # Embedding
    # | (N.. x hidden_size); note that hidden_size==embedding_dim
    # Qwen2TransformerBlock
    # | (N.. x hidden_size)
    # Qwen2TransformerBlock
    # | (N.. x hidden_size)
    # ...
    # |
    # RMSNorm 
    # | (N.. x hidden_size)
    # Embedding::as_linear  OR  Linear (lm_head)
    # | (N.. x vocab_size)
    # output
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        import mlx_lm.models.qwen2 as qwen2
        args : qwen2.ModelArgs = mlx_model.args
        model : qwen2.Qwen2Model = mlx_model.model

        self.num_hidden_layers = args.num_hidden_layers
        self.embedding = Embedding(
            vocab_size=args.vocab_size,
            embedding_dim=args.hidden_size,
            weight=dequantize_linear(model.embed_tokens).astype(mx.float16), # TODO: use quantized?
        )

        self.blocks = []
        for i in range(args.num_hidden_layers):
            block = Qwen2TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=QuantizedWeights.from_mlx_layer(model.layers[i].self_attn.q_proj),
                wk=QuantizedWeights.from_mlx_layer(model.layers[i].self_attn.k_proj),
                wv=QuantizedWeights.from_mlx_layer(model.layers[i].self_attn.v_proj),
                wo=QuantizedWeights.from_mlx_layer(model.layers[i].self_attn.o_proj),
                bq=model.layers[i].self_attn.q_proj.bias,
                bk=model.layers[i].self_attn.k_proj.bias,
                bv=model.layers[i].self_attn.v_proj.bias,
                w_gate=QuantizedWeights.from_mlx_layer(model.layers[i].mlp.gate_proj),
                w_up=QuantizedWeights.from_mlx_layer(model.layers[i].mlp.up_proj),
                w_down=QuantizedWeights.from_mlx_layer(model.layers[i].mlp.down_proj),
                w_input_layernorm=model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=model.layers[i].post_attention_layernorm.weight,
                max_seq_len=args.max_position_embeddings,
                theta=int(args.rope_theta),
                use_flash_attention=enable_flash_attn,
            )
            self.blocks.append(block)
        
        self.rms_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps, weight=model.norm.weight)

        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = QuantizedWeights.from_mlx_layer(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        x = self.embedding(inputs)
        for i in range(self.num_hidden_layers):
            x = self.blocks[i](x, offset, cache[i], mask="causal")
        x = self.rms_norm(x)
        if self.lm_head is not None:
            x = quantized_linear(x, self.lm_head)
        else:
            x = self.embedding.as_linear(x)
        return x
