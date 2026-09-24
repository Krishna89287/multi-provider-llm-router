"""
Command-line demo: route a few prompts and print where each one went.

Run without any API keys and every call is served by the mock provider (marked
[mock]). Set OPENAI_API_KEY or ANTHROPIC_API_KEY and those calls go to the real
API. Either way the token counts and cost estimates are computed the same way.
"""
import argparse

from .router import Router

DEMO_PROMPTS = [
    "Write a Python function that reverses a linked list.",
    "Summarize the attached quarterly report in three bullet points.",
    "Classify this review as positive, negative, or neutral.",
    "Explain why adding an index can slow down writes.",
    "What time is it in Tokyo right now?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-provider LLM router demo")
    parser.add_argument(
        "--strategy", default="cost", choices=["cost", "quality", "balanced"]
    )
    parser.add_argument("--prompt", help="route a single prompt instead of the demo set")
    args = parser.parse_args()

    router = Router(strategy=args.strategy)
    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS

    print(f"strategy: {args.strategy}\n")
    for prompt in prompts:
        r = router.route(prompt)
        tag = " [mock]" if r.is_mock else ""
        approx = " (~tokens)" if r.approx_tokens else ""
        fb = " [fell back]" if r.fell_back else ""
        print(f"{prompt[:52]:<52} -> {r.provider}/{r.model}{tag}{fb}")
        print(
            f"    task={r.task.value} in={r.input_tokens} out={r.output_tokens}"
            f" cost=${r.cost_usd:.6f}{approx}"
        )

    print("\nstats:", router.stats())


if __name__ == "__main__":
    main()
