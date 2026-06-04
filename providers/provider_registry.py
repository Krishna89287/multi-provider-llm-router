"""
Provider Registry
Manages registration and health checking of LLM providers
"""
import time
import logging
from typing import Dict, List, Optional
from router.llm_router import Provider, ProviderConfig, TaskType

logger = logging.getLogger(__name__)


class ProviderHealth:
    def __init__(self):
        self.last_check: float = 0
        self.is_healthy: bool = True
        self.latency_ms: float = 0
        self.error_rate: float = 0.0
        self.check_count: int = 0


class ProviderRegistry:
    """
    Central registry for all LLM providers.
    Tracks health, performance, and availability.
    """

    def __init__(self):
        self.providers: Dict[str, ProviderConfig] = {}
        self.health: Dict[str, ProviderHealth] = {}
        self._register_defaults()
        logger.info("ProviderRegistry initialized")

    def _register_defaults(self):
        defaults = [
            ProviderConfig(Provider.OPENAI, "gpt-4", 0.03, 2000, 8192,
                          [TaskType.CODING, TaskType.REASONING]),
            ProviderConfig(Provider.ANTHROPIC, "claude-3-sonnet", 0.003, 1500, 100000,
                          [TaskType.REASONING, TaskType.CREATIVE]),
            ProviderConfig(Provider.GOOGLE, "gemini-1.5-flash", 0.00035, 500, 32768,
                          [TaskType.GENERAL, TaskType.SUMMARIZATION]),
        ]
        for config in defaults:
            self.register(config)

    def register(self, config: ProviderConfig) -> None:
        key = f"{config.provider.value}/{config.model}"
        self.providers[key] = config
        self.health[key] = ProviderHealth()
        logger.info(f"Provider registered: {key}")

    def get_healthy_providers(self) -> List[ProviderConfig]:
        return [
            config for key, config in self.providers.items()
            if self.health[key].is_healthy
        ]

    def update_health(self, provider: str, model: str,
                      latency_ms: float, success: bool) -> None:
        key = f"{provider}/{model}"
        if key in self.health:
            h = self.health[key]
            h.last_check = time.time()
            h.latency_ms = latency_ms
            h.check_count += 1
            if not success:
                h.error_rate = min(h.error_rate + 0.1, 1.0)
                if h.error_rate > 0.5:
                    h.is_healthy = False
                    logger.warning(f"Provider marked unhealthy: {key}")
            else:
                h.error_rate = max(h.error_rate - 0.05, 0.0)
                h.is_healthy = True

    def get_status(self) -> Dict:
        return {
            key: {
                "healthy": h.is_healthy,
                "latency_ms": h.latency_ms,
                "error_rate": h.error_rate
            }
            for key, h in self.health.items()
        }


if __name__ == "__main__":
    registry = ProviderRegistry()
    healthy = registry.get_healthy_providers()
    print(f"Healthy providers: {len(healthy)}")
    print(registry.get_status())
