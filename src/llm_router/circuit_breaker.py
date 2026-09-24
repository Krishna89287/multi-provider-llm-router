"""
A small per-provider circuit breaker.

If a provider fails repeatedly it gets marked "open" and the router skips it for
a cooldown period instead of hammering a service that is already struggling.
After the cooldown it is allowed back in. This is the standard circuit-breaker
pattern, kept deliberately simple.
"""
import time
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout_s: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, provider: str) -> bool:
        """True if `provider` is currently being skipped."""
        if provider not in self._opened_at:
            return False
        if time.monotonic() - self._opened_at[provider] >= self.reset_timeout_s:
            # Cooldown elapsed: give it another chance.
            del self._opened_at[provider]
            self._failures[provider] = 0
            logger.info("circuit for %s reset after cooldown", provider)
            return False
        return True

    def record_success(self, provider: str) -> None:
        self._failures[provider] = 0
        self._opened_at.pop(provider, None)

    def record_failure(self, provider: str) -> None:
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= self.failure_threshold:
            self._opened_at[provider] = time.monotonic()
            logger.warning(
                "circuit for %s opened after %d failures",
                provider,
                self._failures[provider],
            )

    def open_circuits(self) -> list[str]:
        return [p for p in self._opened_at if self.is_open(p)]
