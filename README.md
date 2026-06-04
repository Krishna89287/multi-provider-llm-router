# Multi-Provider LLM Router

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Krishna89287/multi-provider-llm-router/ci.yml?style=flat-square)](https://github.com/Krishna89287/multi-provider-llm-router/actions)

Intelligent LLM routing across OpenAI, Anthropic Claude, Google Gemini, and local models. Automatic fallback, cost optimization, and circuit breaker — unified behind a single API.

## Why Multi-Provider Routing

Every enterprise AI system depends on multiple LLM providers:

- OpenAI has an outage — route to Claude automatically
- GPT-4 is expensive for this classification task — route to Claude Haiku
- User needs maximum accuracy — route to GPT-4 or Claude Opus
- Running locally — route to Ollama/Mistral

This router handles all of it automatically, reducing costs by 40-60% while improving reliability.

## Quick Start

```bash
git clone https://github.com/Krishna89287/multi-provider-llm-router.git
cd multi-provider-llm-router
pip install -r requirements.txt
cp .env.example .env
# Add your API keys
python router/llm_router.py
```

## Usage

### Basic Routing

```python
from router.llm_router import LLMRouter

router = LLMRouter(strategy="cost_optimized")

response = router.route("Write a Python function to calculate fibonacci numbers")

print(f"Response: {response.content}")
print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
print(f"Cost: ${response.cost_usd:.6f}")
print(f"Latency: {response.latency_ms:.1f}ms")
print(f"Fallback used: {response.fallback_used}")
```

### Routing Strategies

```python
# Cost optimized — cheapest model that handles the task well
cost_router = LLMRouter(strategy="cost_optimized")

# Performance — fastest model
perf_router = LLMRouter(strategy="performance")

# Load balanced — distributes across providers
lb_router = LLMRouter(strategy="load_balanced")
```

### With Cost Limit

```python
router = LLMRouter(strategy="cost_optimized")

# Never spend more than $0.001 per request
response = router.route(
    "Classify this text as positive or negative",
    max_cost=0.001
)
```

### Task Classification

The router automatically classifies task type and picks the best model:

```python
router = LLMRouter()

tasks = [
    "Write a Python function to sort a list",           # -> gpt-4 (coding)
    "Summarize this 10-page document",                  # -> claude-3-sonnet (long context)
    "Classify this review as positive or negative",     # -> claude-3-haiku (cheap)
    "Analyze the philosophical implications of AI",     # -> gpt-4 or claude (reasoning)
]

for task in tasks:
    task_type = router.classify_task(task)
    provider = router.select_provider(task_type)
    print(f"Task: {task[:40]}... -> {provider.model}")
```

### Circuit Breaker

```python
from router.llm_router import CircuitBreaker

cb = CircuitBreaker(failure_threshold=5, reset_timeout=60)

# After 5 failures, circuit opens — requests skip that provider
# After 60 seconds, circuit resets automatically

cb.record_failure("openai")  # x5
print(cb.is_open("openai"))  # True — provider is down

# 60 seconds later
print(cb.is_open("openai"))  # False — circuit reset
```

### Provider Registry

```python
from providers.provider_registry import ProviderRegistry

registry = ProviderRegistry()

# Get all healthy providers
healthy = registry.get_healthy_providers()
print(f"Healthy providers: {len(healthy)}")

# Update health after a call
registry.update_health("openai", "gpt-4", latency_ms=1200, success=True)

# Get full status
status = registry.get_status()
print(status)
```

### Stats and Monitoring

```python
router = LLMRouter()

for prompt in your_prompts:
    router.route(prompt)

stats = router.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
print(f"Requests by provider: {stats['requests_by_provider']}")
print(f"Open circuits: {stats['open_circuits']}")
```

## Supported Providers and Models

| Provider | Models | Strengths |
|---|---|---|
| **OpenAI** | gpt-4, gpt-4-turbo, gpt-3.5-turbo | Coding, reasoning |
| **Anthropic** | claude-3-opus, claude-3-sonnet, claude-3-haiku | Long context, creative |
| **Google** | gemini-1.5-pro, gemini-1.5-flash | Speed, general |
| **Ollama** | mistral-7b, llama-3-8b | Free, local, private |

## Project Structure

```
multi-provider-llm-router/
├── router/
│   └── llm_router.py           # Core router with circuit breaker
├── providers/
│   └── provider_registry.py    # Provider health tracking
├── tests/
│   └── test_router.py
├── .env.example
├── requirements.txt
├── Makefile
└── README.md
```

## Running Tests

```bash
make test
```

## License

MIT — see [LICENSE](LICENSE)
