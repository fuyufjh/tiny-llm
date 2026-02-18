import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    
    def sample(logprobs: mx.array):
        if temp == 0: # Greedy decoding
            return mx.argmax(logprobs, axis=-1)
        
        if top_p is not None: # Top-p (Nucleus) Sampling
            # Sort logprobs in descending order
            sorted_indices = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

            # Convert to probabilities and compute cumulative sum
            sorted_probs = mx.softmax(sorted_logprobs, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

            # Keep tokens until cumulative probability exceeds top_p, while keep the last one that exceeds top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            shifted = mx.concatenate([mx.zeros_like(sorted_indices_to_remove[..., :1]), sorted_indices_to_remove[..., :-1]], axis=-1)
            sorted_indices_to_remove = shifted

            # Map the mask back to original indices
            indices_to_remove = mx.take_along_axis(sorted_indices_to_remove, mx.argsort(sorted_indices, axis=-1), axis=-1)

            # Set removed indices to -inf
            logprobs = mx.where(indices_to_remove, float("-inf"), logprobs)

        if top_k is not None: # Top-k Sampling
            k = min(top_k, logprobs.shape[-1])  # just in case top_k is larger than vocab size

            # Set logprobs outside top-k to -inf
            top_k_logprobs = mx.topk(logprobs, k=k, axis=-1)
            mask = logprobs < mx.min(top_k_logprobs, axis=-1, keepdims=True)
            logprobs = mx.where(mask, float("-inf"), logprobs)

        # Sample from the filtered distribution
        result = mx.random.categorical(logprobs / temp, axis=-1)
        return result

    return sample
