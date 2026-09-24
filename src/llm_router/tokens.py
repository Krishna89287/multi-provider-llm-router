"""
Token counting.

For OpenAI models we use tiktoken, which is the real tokenizer OpenAI uses, so
the counts match what you get billed for. Anthropic and Google use different
tokenizers that are not open source, so for those we fall back to tiktoken's
cl100k_base as an approximation. It is close enough for cost estimation and
routing decisions, and the code says clearly when it is approximating rather
than pretending the number is exact.
"""
from functools import lru_cache

try:
    import tiktoken

    _HAVE_TIKTOKEN = True
except ImportError:  # pragma: no cover - exercised only when tiktoken is missing
    _HAVE_TIKTOKEN = False


# Which encoding to use per model. Anything not listed uses cl100k_base.
_ENCODING_FOR_MODEL = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
}


@lru_cache(maxsize=8)
def _encoding(name: str):
    return tiktoken.get_encoding(name)


def count_tokens(text: str, model: str) -> int:
    """
    Number of tokens in `text` for `model`.

    Exact for OpenAI models, approximate for the rest. If tiktoken is not
    installed at all, falls back to a rough words * 1.3 heuristic so the router
    still runs.
    """
    if not _HAVE_TIKTOKEN:
        return max(1, int(len(text.split()) * 1.3))

    encoding_name = _ENCODING_FOR_MODEL.get(model, "cl100k_base")
    return len(_encoding(encoding_name).encode(text))


def is_approximate(model: str) -> bool:
    """True when the count for this model is an approximation, not the real tokenizer."""
    return model not in _ENCODING_FOR_MODEL
