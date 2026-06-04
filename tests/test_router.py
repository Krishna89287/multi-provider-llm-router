"""Tests for multi-provider LLM router"""
import sys
sys.path.insert(0, '..')
from router.llm_router import LLMRouter, TaskType


def test_task_classification_coding():
    router = LLMRouter()
    task = router.classify_task("Write a Python function to sort a list")
    assert task == TaskType.CODING


def test_task_classification_summarization():
    router = LLMRouter()
    task = router.classify_task("Summarize this document in 3 bullet points")
    assert task == TaskType.SUMMARIZATION


def test_task_classification_general():
    router = LLMRouter()
    task = router.classify_task("Hello, how are you?")
    assert task == TaskType.GENERAL


def test_router_selects_provider():
    router = LLMRouter(strategy="cost_optimized")
    provider = router.select_provider(TaskType.GENERAL)
    assert provider is not None


def test_router_routes_request():
    router = LLMRouter()
    response = router.route("What is LangGraph?")
    assert response.content is not None
    assert response.provider is not None
    assert response.cost_usd >= 0


def test_circuit_breaker():
    from router.llm_router import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure("openai")
    assert cb.is_open("openai")
    cb.record_success("openai")
    assert not cb.is_open("openai")


def test_router_stats():
    router = LLMRouter()
    router.route("Test prompt")
    stats = router.get_stats()
    assert "total_requests" in stats
    assert stats["total_requests"] >= 1
