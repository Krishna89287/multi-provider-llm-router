# multi-provider-llm-router

> Stop using GPT-4 for everything

[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/Krishna89287/multi-provider-llm-router/daily-commit.yml?style=flat-square)](https://github.com/Krishna89287/multi-provider-llm-router/actions)

The first time I noticed that Claude Haiku handles classification just as well as GPT-4 at 1/50th the cost, I went back and looked at what we were actually sending to GPT-4. About 60% of it was simple tasks that a much cheaper model handles fine.

This router classifies each request and sends it to the right model. GPT-4 for complex reasoning, Claude Haiku for classification and summarisation, local Mistral for anything that shouldn't leave your machine. The circuit breaker handles provider outages automatically.

Running it for a month cut our LLM costs by about 65%.

## What it looks like running

![demo](docs/demo.svg)

## Getting started

```bash
git clone https://github.com/Krishna89287/multi-provider-llm-router
cd multi-provider-llm-router
pip install -r requirements.txt
cp .env.example .env
python router/llm_router.py
```

**Stack:** Python · FastAPI · OpenAI · Anthropic · Google Gemini · Ollama

---

Built by [Krishna Gove](https://github.com/Krishna89287), working on AI and cloud infrastructure in Munich.
