"""
Provider clients.

Each provider knows how to turn a prompt into a completion and report how many
tokens it used. The OpenAI and Anthropic clients call the real APIs through the
official SDKs when an API key is present. When no key is set (CI, a quick local
demo, tests) they fall back to a MockProvider that returns a deterministic
canned response and counts tokens locally, so the router still runs end to end
without spending money. Responses carry an `is_mock` flag so nothing downstream
pretends a mock call was real.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass

from .tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    is_mock: bool


class ProviderError(Exception):
    """Raised when a provider call fails, so the router can fail over."""


class MockProvider:
    """Offline stand-in. Deterministic, no network, counts tokens for real."""

    def __init__(self, provider: str):
        self.provider = provider

    def complete(self, prompt: str, model: str, max_tokens: int = 256) -> Completion:
        text = (
            f"[mock:{self.provider}/{model}] This is a canned response used when "
            f"no API key is configured. It exists so the router can be run and "
            f"tested offline."
        )
        return Completion(
            text=text,
            input_tokens=count_tokens(prompt, model),
            output_tokens=count_tokens(text, model),
            is_mock=True,
        )


class OpenAIProvider:
    def __init__(self):
        from openai import OpenAI  # imported lazily so the SDK is optional

        self._client = OpenAI()

    def complete(self, prompt: str, model: str, max_tokens: int = 256) -> Completion:
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception as exc:  # normalise SDK errors for the router
            raise ProviderError(f"openai call failed: {exc}") from exc
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            is_mock=False,
        )


class AnthropicProvider:
    def __init__(self):
        import anthropic  # imported lazily so the SDK is optional

        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, model: str, max_tokens: int = 256) -> Completion:
        try:
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise ProviderError(f"anthropic call failed: {exc}") from exc
        return Completion(
            text="".join(block.text for block in resp.content if block.type == "text"),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            is_mock=False,
        )


# Maps provider name -> (env var that must be set, factory for the real client).
_REAL_CLIENTS = {
    "openai": ("OPENAI_API_KEY", OpenAIProvider),
    "anthropic": ("ANTHROPIC_API_KEY", AnthropicProvider),
}


def get_client(provider: str):
    """
    Return a client for `provider`. Uses the real SDK when the matching API key
    is set and the SDK is installed; otherwise returns a MockProvider and logs
    why, so it is obvious when you are running offline.
    """
    spec = _REAL_CLIENTS.get(provider)
    if spec is None:
        # e.g. a local Ollama model: no key, treat as mock for this demo.
        return MockProvider(provider)

    env_var, factory = spec
    if not os.getenv(env_var):
        logger.info("%s not set, using mock provider for %s", env_var, provider)
        return MockProvider(provider)
    try:
        return factory()
    except ImportError:
        logger.info("SDK for %s not installed, using mock provider", provider)
        return MockProvider(provider)
