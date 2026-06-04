"""
Multi-Provider LLM Router
Intelligent routing across OpenAI, Anthropic, Google, and local models
"""
import time
import random
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class TaskType(Enum):
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    GENERAL = "general"


@dataclass
class ProviderConfig:
    provider: Provider
    model: str
    cost_per_1k_tokens: float
    avg_latency_ms: float
    max_tokens: int
    strengths: List[TaskType]
    is_available: bool = True
    error_count: int = 0


@dataclass
class RouterResponse:
    content: str
    provider: str
    model: str
    latency_ms: float
    cost_usd: float
    fallback_used: bool = False


class CircuitBreaker:
    """
    Circuit breaker for LLM provider calls.
    Opens after too many failures, preventing cascade failures.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures: Dict[str, int] = {}
        self.last_failure: Dict[str, float] = {}
        self.open_circuits: set = set()

    def is_open(self, provider: str) -> bool:
        if provider not in self.open_circuits:
            return False
        if time.time() - self.last_failure.get(provider, 0) > self.reset_timeout:
            self.open_circuits.discard(provider)
            self.failures[provider] = 0
            logger.info(f"Circuit breaker reset for: {provider}")
            return False
        return True

    def record_failure(self, provider: str) -> None:
        self.failures[provider] = self.failures.get(provider, 0) + 1
        self.last_failure[provider] = time.time()
        if self.failures[provider] >= self.failure_threshold:
            self.open_circuits.add(provider)
            logger.warning(f"Circuit breaker opened for: {provider}")

    def record_success(self, provider: str) -> None:
        self.failures[provider] = 0
        self.open_circuits.discard(provider)


class LLMRouter:
    """
    Intelligent multi-provider LLM router.
    Routes based on task type, cost, latency, and provider health.
    Falls back automatically on failure.
    """

    PROVIDER_CONFIGS = [
        ProviderConfig(Provider.OPENAI, "gpt-4", 0.03, 2000, 8192,
                      [TaskType.CODING, TaskType.REASONING]),
        ProviderConfig(Provider.OPENAI, "gpt-3.5-turbo", 0.001, 800, 4096,
                      [TaskType.GENERAL, TaskType.CLASSIFICATION]),
        ProviderConfig(Provider.ANTHROPIC, "claude-3-sonnet", 0.003, 1500, 100000,
                      [TaskType.REASONING, TaskType.CREATIVE, TaskType.SUMMARIZATION]),
        ProviderConfig(Provider.ANTHROPIC, "claude-3-haiku", 0.00025, 600, 100000,
                      [TaskType.GENERAL, TaskType.CLASSIFICATION]),
        ProviderConfig(Provider.GOOGLE, "gemini-1.5-flash", 0.00035, 500, 32768,
                      [TaskType.GENERAL, TaskType.SUMMARIZATION]),
        ProviderConfig(Provider.OLLAMA, "mistral-7b", 0.0, 3000, 4096,
                      [TaskType.GENERAL, TaskType.CODING]),
    ]

    def __init__(self, strategy: str = "cost_optimized"):
        self.strategy = strategy
        self.circuit_breaker = CircuitBreaker()
        self.request_count: Dict[str, int] = {}
        self.total_cost: float = 0.0
        logger.info(f"LLMRouter initialized with strategy: {strategy}")

    def classify_task(self, prompt: str) -> TaskType:
        """Classify the task type from the prompt."""
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["code", "function", "class", "debug", "python", "javascript"]):
            return TaskType.CODING
        if any(w in prompt_lower for w in ["summarize", "summary", "tldr"]):
            return TaskType.SUMMARIZATION
        if any(w in prompt_lower for w in ["creative", "story", "poem", "write"]):
            return TaskType.CREATIVE
        if any(w in prompt_lower for w in ["classify", "category", "label", "sentiment"]):
            return TaskType.CLASSIFICATION
        if any(w in prompt_lower for w in ["reason", "analyze", "think", "explain why"]):
            return TaskType.REASONING
        return TaskType.GENERAL

    def select_provider(self, task_type: TaskType, max_cost: Optional[float] = None) -> Optional[ProviderConfig]:
        """Select best provider based on strategy."""
        available = [
            p for p in self.PROVIDER_CONFIGS
            if p.is_available and not self.circuit_breaker.is_open(p.provider.value)
        ]

        if max_cost:
            available = [p for p in available if p.cost_per_1k_tokens <= max_cost]

        if not available:
            logger.error("No available providers!")
            return None

        if self.strategy == "cost_optimized":
            best_for_task = [p for p in available if task_type in p.strengths]
            pool = best_for_task if best_for_task else available
            return min(pool, key=lambda x: x.cost_per_1k_tokens)

        elif self.strategy == "performance":
            best_for_task = [p for p in available if task_type in p.strengths]
            pool = best_for_task if best_for_task else available
            return min(pool, key=lambda x: x.avg_latency_ms)

        elif self.strategy == "load_balanced":
            return random.choice(available)

        return available[0]

    def route(self, prompt: str, max_cost: Optional[float] = None) -> RouterResponse:
        """Route request to best available provider with fallback."""
        task_type = self.classify_task(prompt)
        selected = self.select_provider(task_type, max_cost)

        if not selected:
            raise RuntimeError("No providers available")

        start_time = time.time()
        fallback_used = False

        try:
            # In production: call actual provider SDK
            response_text = f"[{selected.provider.value}/{selected.model}] Response to: {prompt[:50]}..."
            latency = (time.time() - start_time) * 1000
            cost = (len(prompt.split()) / 1000) * selected.cost_per_1k_tokens

            self.circuit_breaker.record_success(selected.provider.value)
            self.total_cost += cost
            self.request_count[selected.provider.value] = self.request_count.get(selected.provider.value, 0) + 1

            return RouterResponse(
                content=response_text,
                provider=selected.provider.value,
                model=selected.model,
                latency_ms=latency,
                cost_usd=cost,
                fallback_used=fallback_used
            )

        except Exception as e:
            self.circuit_breaker.record_failure(selected.provider.value)
            logger.error(f"Provider {selected.provider.value} failed: {e}")

            fallback = self.select_provider(task_type, max_cost)
            if fallback and fallback.provider != selected.provider:
                logger.info(f"Falling back to: {fallback.provider.value}")
                return RouterResponse(
                    content=f"[FALLBACK {fallback.provider.value}/{fallback.model}] Response to: {prompt[:50]}...",
                    provider=fallback.provider.value,
                    model=fallback.model,
                    latency_ms=(time.time() - start_time) * 1000,
                    cost_usd=0.0,
                    fallback_used=True
                )
            raise

    def get_stats(self) -> Dict:
        return {
            "strategy": self.strategy,
            "total_requests": sum(self.request_count.values()),
            "total_cost_usd": round(self.total_cost, 4),
            "requests_by_provider": self.request_count,
            "open_circuits": list(self.circuit_breaker.open_circuits)
        }


if __name__ == "__main__":
    router = LLMRouter(strategy="cost_optimized")

    test_prompts = [
        "Write a Python function to sort a list",
        "Summarize this document in 3 bullet points",
        "Classify this text as positive or negative sentiment",
        "Explain why recursion can cause stack overflow",
    ]

    for prompt in test_prompts:
        response = router.route(prompt, max_cost=0.01)
        print(f"Prompt: {prompt[:40]}...")
        print(f"Routed to: {response.provider}/{response.model}")
        print(f"Cost: ${response.cost_usd:.6f}, Latency: {response.latency_ms:.1f}ms")
        print()

    print("Stats:", router.get_stats())
