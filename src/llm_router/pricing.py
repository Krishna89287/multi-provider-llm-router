"""
Published per-token prices for the models the router knows about.

Prices are USD per 1 million tokens, split into input (prompt) and output
(completion), taken from each provider's public pricing page. They change over
time, so treat this as a config file you keep up to date rather than a source of
truth. Last checked: 2026-01.

Local models (Ollama) are priced at zero here because you run them on your own
hardware. That is not really "free" once you count electricity and GPU time, but
there is no per-token bill, which is what this table is about.
"""
from dataclasses import dataclass

PRICING_LAST_CHECKED = "2026-01"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""
    input_per_m: float
    output_per_m: float


# Keyed by the model id the router uses internally.
PRICES = {
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4o-mini": Price(0.15, 0.60),
    "claude-3-5-sonnet": Price(3.00, 15.00),
    "claude-3-5-haiku": Price(0.80, 4.00),
    "gemini-1.5-flash": Price(0.075, 0.30),
    "mistral-7b-local": Price(0.0, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of a single call in USD, from real token counts and the table above."""
    if model not in PRICES:
        raise KeyError(f"No price on file for model '{model}'. Add it to PRICES.")
    p = PRICES[model]
    return (input_tokens / 1_000_000) * p.input_per_m + (
        output_tokens / 1_000_000
    ) * p.output_per_m
