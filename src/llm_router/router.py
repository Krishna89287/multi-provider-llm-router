"""
The router itself.

It picks a model for each prompt based on the task it looks like and the
strategy you chose, estimates the cost from real token counts and real prices,
calls the provider, and falls over to the next best model if a provider errors
or its circuit is open. Selection is deterministic and easy to explain, which
matters more than being clever.
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum

from . import pricing
from .tokens import count_tokens, is_approximate
from .circuit_breaker import CircuitBreaker
from .providers import get_client, ProviderError

logger = logging.getLogger(__name__)


class Task(str, Enum):
    CODING = "coding"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    REASONING = "reasoning"
    GENERAL = "general"


# Quality tier is a documented, hand-assigned rank (1 = strongest) used only by
# the "quality" and "balanced" strategies. It is a judgement call, not a
# measured benchmark, and is labelled that way on purpose.
@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    good_at: tuple[Task, ...]
    quality_tier: int


CATALOG = [
    Model("gpt-4o", "openai", (Task.CODING, Task.REASONING), 1),
    Model("claude-3-5-sonnet", "anthropic", (Task.REASONING, Task.SUMMARIZATION), 1),
    Model("gpt-4o-mini", "openai", (Task.GENERAL, Task.CLASSIFICATION), 2),
    Model("claude-3-5-haiku", "anthropic", (Task.GENERAL, Task.CLASSIFICATION), 2),
    Model("gemini-1.5-flash", "google", (Task.SUMMARIZATION, Task.GENERAL), 2),
    Model("mistral-7b-local", "ollama", (Task.GENERAL, Task.CODING), 3),
]

# Order matters: the first task whose keywords match wins. Classification is
# checked before coding so that "classify" is not mistaken for "class".
_KEYWORDS = {
    Task.CLASSIFICATION: ("classify", "classification", "category", "label", "sentiment"),
    Task.SUMMARIZATION: ("summarize", "summary", "tldr", "shorten", "changelog"),
    Task.CODING: ("code", "function", "class", "debug", "python", "javascript", "sql"),
    Task.REASONING: ("why", "analyze", "reason", "explain", "compare"),
}


@dataclass
class Response:
    text: str
    model: str
    provider: str
    task: Task
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    is_mock: bool
    approx_tokens: bool
    fell_back: bool = False


@dataclass
class Router:
    strategy: str = "cost"  # "cost", "quality", or "balanced"
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    requests_by_model: dict[str, int] = field(default_factory=dict)
    total_cost_usd: float = 0.0

    def classify(self, prompt: str) -> Task:
        low = prompt.lower()
        for task, words in _KEYWORDS.items():
            # Whole-word match so "classify" does not trigger on "class", etc.
            if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words):
                return task
        return Task.GENERAL

    def _estimated_input_price(self, model: str) -> float:
        # Price of the input side only, per 1M tokens, used to rank by cost.
        return pricing.PRICES[model].input_per_m

    def candidates(self, task: Task) -> list[Model]:
        """Models that suit the task (or all models if none are tagged), ordered by strategy."""
        usable = [
            m
            for m in CATALOG
            if not self.circuit_breaker.is_open(m.provider)
        ]
        fit = [m for m in usable if task in m.good_at] or usable
        if not fit:
            return []

        if self.strategy == "cost":
            return sorted(fit, key=lambda m: self._estimated_input_price(m.id))
        if self.strategy == "quality":
            return sorted(fit, key=lambda m: (m.quality_tier, self._estimated_input_price(m.id)))
        if self.strategy == "balanced":
            # Cheapest model that is at least tier 2 (skip the very weakest).
            good_enough = [m for m in fit if m.quality_tier <= 2] or fit
            return sorted(good_enough, key=lambda m: self._estimated_input_price(m.id))
        raise ValueError(f"unknown strategy: {self.strategy}")

    def route(self, prompt: str, max_tokens: int = 256) -> Response:
        task = self.classify(prompt)
        ordered = self.candidates(task)
        if not ordered:
            raise RuntimeError("no providers available (all circuits open)")

        last_error: Exception | None = None
        for attempt, model in enumerate(ordered):
            client = get_client(model.provider)
            started = time.monotonic()
            try:
                completion = client.complete(prompt, model.id, max_tokens=max_tokens)
            except ProviderError as exc:
                self.circuit_breaker.record_failure(model.provider)
                last_error = exc
                logger.warning("model %s failed, trying next: %s", model.id, exc)
                continue

            latency_ms = (time.monotonic() - started) * 1000
            self.circuit_breaker.record_success(model.provider)
            cost = pricing.cost_usd(
                model.id, completion.input_tokens, completion.output_tokens
            )
            self.total_cost_usd += cost
            self.requests_by_model[model.id] = self.requests_by_model.get(model.id, 0) + 1

            return Response(
                text=completion.text,
                model=model.id,
                provider=model.provider,
                task=task,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                is_mock=completion.is_mock,
                approx_tokens=is_approximate(model.id),
                fell_back=attempt > 0,
            )

        raise RuntimeError(f"all providers failed; last error: {last_error}")

    def stats(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_requests": sum(self.requests_by_model.values()),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "requests_by_model": dict(self.requests_by_model),
            "open_circuits": self.circuit_breaker.open_circuits(),
        }
