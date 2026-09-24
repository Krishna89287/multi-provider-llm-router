"""
Cost benchmark: routing by task vs sending everything to one strong model.

For a fixed set of prompts this measures the total estimated cost two ways:

  1. Routed: each prompt goes to the cheapest model the router picks for its task.
  2. Baseline: every prompt goes to gpt-4o (a strong, expensive default).

Input token counts are real (tiktoken). Output tokens are not known ahead of a
real call, so we assume a fixed output length per response and state that
assumption openly; the same assumption is applied to both sides, so the
comparison is fair. Change ASSUMED_OUTPUT_TOKENS to see how the gap moves.

The point is not a marketing number. It is: for a workload that is mostly light
tasks, routing the light ones to cheaper models saves real money, and here is
how much given these prices.
"""
from __future__ import annotations

from llm_router import pricing
from llm_router.router import Router
from llm_router.tokens import count_tokens

ASSUMED_OUTPUT_TOKENS = 256
BASELINE_MODEL = "gpt-4o"

# A workload that is mostly light tasks with a few that genuinely need a top model.
WORKLOAD = [
    "Classify this support ticket as billing, technical, or account.",
    "Summarize this email thread in two sentences.",
    "What is the capital of Australia?",
    "Translate 'good morning' into German.",
    "Summarize the release notes into a short changelog.",
    "Label the sentiment of this tweet.",
    "Write and debug a Python function to merge two sorted lists.",
    "Analyze why this distributed cache keeps returning stale reads.",
]


def main() -> None:
    router = Router(strategy="cost")
    routed_total = 0.0
    baseline_total = 0.0
    rows = []

    for prompt in WORKLOAD:
        task = router.classify(prompt)
        chosen = router.candidates(task)[0]
        in_tokens = count_tokens(prompt, chosen.id)

        routed_cost = pricing.cost_usd(chosen.id, in_tokens, ASSUMED_OUTPUT_TOKENS)
        base_in = count_tokens(prompt, BASELINE_MODEL)
        baseline_cost = pricing.cost_usd(BASELINE_MODEL, base_in, ASSUMED_OUTPUT_TOKENS)

        routed_total += routed_cost
        baseline_total += baseline_cost
        rows.append((task.value, chosen.id, routed_cost, baseline_cost))

    print(f"assumed output tokens per response: {ASSUMED_OUTPUT_TOKENS}")
    print(f"baseline model: {BASELINE_MODEL}")
    print(f"prices last checked: {pricing.PRICING_LAST_CHECKED}\n")
    print(f"{'task':<16}{'routed to':<20}{'routed $':>12}{'baseline $':>14}")
    for task, model, routed_cost, baseline_cost in rows:
        print(f"{task:<16}{model:<20}{routed_cost:>12.6f}{baseline_cost:>14.6f}")

    saved = baseline_total - routed_total
    pct = (saved / baseline_total) * 100 if baseline_total else 0.0
    print("-" * 62)
    print(f"{'totals':<36}{routed_total:>12.6f}{baseline_total:>14.6f}")
    print(f"\nrouting was {pct:.1f}% cheaper on this workload "
          f"(${saved:.6f} saved over {len(WORKLOAD)} calls)")


if __name__ == "__main__":
    main()
