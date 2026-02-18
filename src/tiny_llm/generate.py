import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from tiny_llm.kv_cache import TinyKvFullCache
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        y = y[None, :]  # add batch dimension
        h = model(y)
        logits = h[:, -1, :]
        if sampler is None:
            return mx.argmax(logits, axis=-1)
        else:
            logprobs = logits - mx.logsumexp(logits, keepdims=True)  # for numerical stability
            return sampler(logprobs)

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    while True:
        token = _step(model, tokens)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)



def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    # kv cache for each layer
    kv_caches = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    def _step(model, y, offset):
        y = y[None, :]  # add batch dimension
        h = model(y, offset=offset, cache=kv_caches)
        logits = h[:, -1, :]
        return mx.argmax(logits, axis=-1)

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    last_offset = 0
    while True:
        token = _step(model, tokens[last_offset:], offset=last_offset)
        last_offset = tokens.shape[0]
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
