import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int, # max sequence length?
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        # Compute frequencies: shape (dims // 2,)
        # freq_i = 1 / (base ^ (2i / dims)) for i = 0, 1, ..., dims//2 - 1
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2) / dims))
        
        # Compute positions: shape (seq_len,)
        positions = mx.arange(seq_len)
        
        # Outer product: positions[:, None] @ inv_freq[None, :]
        # Shape: (seq_len, dims // 2)
        freqs = positions[:, None] * inv_freq[None, :]
        
        # Compute cos and sin: shape (seq_len, dims // 2)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

    def __call__(
        self, 
        x: mx.array,  # batch_size x seq_length x num_heads x head_dim
        offset: list[slice] | slice | None = None
    ) -> mx.array:
        batch_size, seq_length, num_heads, head_dim = x.shape
        
        # Extract the slice of frequencies based on offset
        if offset is None:
            cos_freqs = self.cos_freqs[:seq_length]
            sin_freqs = self.sin_freqs[:seq_length]
        elif isinstance(offset, slice):
            cos_freqs = self.cos_freqs[offset]
            sin_freqs = self.sin_freqs[offset]
        elif isinstance(offset, list):
            # TODO
            raise NotImplementedError("List of slices not implemented yet")
        else:
            raise ValueError("Invalid offset type")
        
        if self.traditional:
            # Traditional RoPE: pairs (x[0], x[1]), (x[2], x[3]), ...
            # Reshape x to (batch, seq_len, num_heads, dims // 2, 2)
            x_reshaped = x.reshape(batch_size, seq_length, num_heads, head_dim // 2, 2)
            
            # x_reshaped[:, :, :, :, 0] is the even indices
            # x_reshaped[:, :, :, :, 1] is the odd indices
            # cos_freqs and sin_freqs have shape (seq_length, dims // 2)
            
            # Reshape for broadcasting: (1, seq_length, 1, dims // 2)
            cos_freqs = cos_freqs[None, :, None, :]  # (1, seq_length, 1, dims // 2)
            sin_freqs = sin_freqs[None, :, None, :]  # (1, seq_length, 1, dims // 2)
            
            # Apply rotation to each pair
            # output[..., 0] = x[..., 0] * cos - x[..., 1] * sin
            # output[..., 1] = x[..., 0] * sin + x[..., 1] * cos
            output = mx.zeros_like(x_reshaped)
            output[..., 0] = x_reshaped[..., 0] * cos_freqs - x_reshaped[..., 1] * sin_freqs
            output[..., 1] = x_reshaped[..., 0] * sin_freqs + x_reshaped[..., 1] * cos_freqs
            
            # Reshape back to original shape
            return output.reshape(batch_size, seq_length, num_heads, head_dim)
        else:
            # Non-traditional RoPE for Qwen2: split dimensions in half
            # First half gets frequencies applied one way, second half another way
            half_dim = head_dim // 2
            x1 = x[..., :half_dim]  # (batch, seq_length, num_heads, half_dim)
            x2 = x[..., half_dim:]  # (batch, seq_length, num_heads, half_dim)
            
            # Reshape for broadcasting
            cos_freqs = cos_freqs[None, :, None, :]  # (1, seq_length, 1, dims // 2)
            sin_freqs = sin_freqs[None, :, None, :]  # (1, seq_length, 1, dims // 2)
            
            # Apply rotation to both halves
            # output1 = x1 * cos - x2 * sin
            # output2 = x1 * sin + x2 * cos
            output1 = x1 * cos_freqs - x2 * sin_freqs
            output2 = x1 * sin_freqs + x2 * cos_freqs
            
            # Concatenate back
            return mx.concatenate([output1, output2], axis=-1)
            
