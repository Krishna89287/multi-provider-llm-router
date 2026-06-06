# Multi-Provider LLM Router

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Krishna89287/multi-provider-llm-router/daily-commit.yml?style=flat-square&label=Daily+Build)](https://github.com/Krishna89287/multi-provider-llm-router/actions)
[![GitHub stars](https://img.shields.io/github/stars/Krishna89287/multi-provider-llm-router?style=flat-square)](https://github.com/Krishna89287/multi-provider-llm-router)

Intelligent routing across OpenAI, Anthropic, Google and local models with circuit breaker

**Stack:** Python · FastAPI · OpenAI · Anthropic · Google Gemini · Ollama · Redis


## Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────┐
│         Task Classifier          │
│  coding/reasoning/creative/     │
│  summarization/classification   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│         LLM Router               │
│  Strategy: cost/performance/    │
│            load_balanced        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│       Circuit Breaker            │
│  Failure threshold: 5           │
│  Reset timeout: 60s             │
└─────────────────────────────────┘
    │
    ├──────┬──────┬──────┬────────┐
    ▼      ▼      ▼      ▼        ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│GPT-4 ││Claude││Gemini││GPT-3 ││Ollama│
│      ││Sonnet││Flash ││Turbo ││Local │
└──────┘└──────┘└──────┘└──────┘└──────┘
    │
    ▼
┌─────────────────────────────────┐
│      Provider Registry           │
│  Health Tracking · Stats        │
│  Cost per 1k tokens             │
└─────────────────────────────────┘
```



## Demo

```
$ python router/llm_router.py

LLMRouter initialized | strategy=cost_optimized

Test 1: "Write a Python fibonacci function"
  Task type: CODING → best: GPT-4 (coding strength)
  Provider: openai/gpt-4
  Cost: $0.000045 | Latency: 1240ms

Test 2: "Summarize this document in 3 bullets"
  Task type: SUMMARIZATION → best: Claude Haiku (cheap+fast)
  Provider: anthropic/claude-3-haiku
  Cost: $0.000003 | Latency: 580ms

Test 3: "Classify sentiment: positive or negative"
  Task type: CLASSIFICATION → best: Claude Haiku ($0.00025/1k)
  Provider: anthropic/claude-3-haiku
  Cost: $0.000002 | Latency: 340ms

Stats:
  total_requests: 3
  total_cost_usd: $0.000050
  savings_vs_gpt4_only: 78%
  open_circuits: []
```


## Quick Start

```bash
git clone https://github.com/Krishna89287/multi-provider-llm-router.git
cd multi-provider-llm-router
pip install -r requirements.txt
cp .env.example .env
make run
```

## Running Tests

```bash
make test
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Built by [Krishna Gove](https://github.com/Krishna89287) · [LinkedIn](https://www.linkedin.com/in/krishna-reddy-327463222)
