# multi-provider-llm-router

Route each prompt to the cheapest model that can handle it, estimate the cost
from real token counts, and fail over to another provider when one is down.

[![CI](https://github.com/Krishna89287/multi-provider-llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/Krishna89287/multi-provider-llm-router/actions/workflows/ci.yml)

Most apps send every request to one strong, expensive model. But a lot of real
traffic is light: classification, short summaries, simple lookups. Those do not
need the top model. This router looks at what a prompt is asking for, picks a
model by the strategy you choose (cheapest, best, or a balance), works out the
cost from real token counts and published prices, and calls the provider. If a
provider errors or its circuit breaker is open, it moves on to the next option.

It is deliberately simple and deterministic, so you can always explain why a
given prompt went where it did.

## What is real and what is not

- **Token counts are real.** OpenAI models use tiktoken (the actual tokenizer);
  other providers use `cl100k_base` as an approximation, and the output says so
  with `(~tokens)`.
- **Prices are real** published per-million-token rates, in `pricing.py`, dated
  so you know when to refresh them.
- **Provider calls are real** through the official `openai` and `anthropic` SDKs
  when the matching API key is set. With no key it uses a clearly marked mock
  provider so the whole thing runs and tests offline. Mock responses print
  `[mock]`.
- The cost benchmark below assumes a fixed number of output tokens per response
  (you cannot know it before the call). The same assumption is applied to both
  sides, so the comparison is fair, and you can change it in `benchmark.py`.

## Getting started

```bash
pip install -e ".[dev]"
python -m llm_router.cli --strategy cost
```

That runs entirely offline in mock mode. To make real calls:

```bash
pip install openai anthropic
export OPENAI_API_KEY=sk-...        # and/or ANTHROPIC_API_KEY
python -m llm_router.cli --strategy cost
```

## What it looks like running

```
strategy: cost

Write a Python function that reverses a linked list. -> ollama/mistral-7b-local [mock]
    task=coding in=11 out=38 cost=$0.000000 (~tokens)
Summarize the attached quarterly report in three bul -> google/gemini-1.5-flash [mock]
    task=summarization in=12 out=38 cost=$0.000012 (~tokens)
Classify this review as positive, negative, or neutr -> openai/gpt-4o-mini [mock]
    task=classification in=12 out=37 cost=$0.000024
Explain why adding an index can slow down writes.    -> openai/gpt-4o [mock]
    task=reasoning in=10 out=36 cost=$0.000385
What time is it in Tokyo right now?                  -> ollama/mistral-7b-local [mock]
    task=general in=9 out=38 cost=$0.000000 (~tokens)
```

## Cost benchmark

`python benchmark/benchmark.py` compares routing against sending everything to
gpt-4o, on a workload that is mostly light tasks:

```
task            routed to               routed $    baseline $
classification  gpt-4o-mini             0.000156      0.002592
summarization   gemini-1.5-flash        0.000078      0.002585
general         mistral-7b-local        0.000000      0.002577
general         mistral-7b-local        0.000000      0.002580
summarization   gemini-1.5-flash        0.000078      0.002590
classification  gpt-4o-mini             0.000155      0.002577
coding          mistral-7b-local        0.000000      0.002590
reasoning       gpt-4o                  0.002585      0.002585
--------------------------------------------------------------
totals                                  0.003050      0.020678

routing was 85.2% cheaper on this workload ($0.017627 saved over 8 calls)
```

The size of the saving depends entirely on your traffic mix. A workload that is
all hard reasoning would show little difference, because the router would send
most of it to the strong model anyway. That is the honest point: routing helps
in proportion to how much of your traffic is light.

## How it fits together

- `tokens.py` counts tokens (tiktoken).
- `pricing.py` holds the price table and the cost calculation.
- `providers.py` wraps each provider's SDK, with a mock fallback.
- `circuit_breaker.py` skips a provider that has failed repeatedly, for a cooldown.
- `router.py` classifies the task, ranks models by the chosen strategy, calls the
  provider, and fails over on error.

## Strategies

- `cost`: cheapest model that fits the task.
- `quality`: strongest model (by a documented, hand-assigned tier) that fits.
- `balanced`: cheapest model above the weakest tier.

## Tests

```bash
pytest -q
```

**Stack:** Python, tiktoken. Optional: `openai`, `anthropic` SDKs for real calls.

---

Built by [Krishna Gove](https://github.com/Krishna89287), working on AI and cloud infrastructure in Munich.
