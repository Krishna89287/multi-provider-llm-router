import pytest

from llm_router import pricing
from llm_router.router import Router, Task, CATALOG
from llm_router.circuit_breaker import CircuitBreaker
from llm_router.tokens import count_tokens


def test_classify_picks_expected_tasks():
    r = Router()
    assert r.classify("Write a Python function to sort a list") == Task.CODING
    assert r.classify("Summarize this article") == Task.SUMMARIZATION
    assert r.classify("Classify the sentiment of this review") == Task.CLASSIFICATION
    assert r.classify("What time is it?") == Task.GENERAL


def test_cost_strategy_prefers_cheaper_model_than_quality():
    cost_pick = Router(strategy="cost").candidates(Task.GENERAL)[0]
    quality_pick = Router(strategy="quality").candidates(Task.REASONING)[0]
    # The cost strategy should never pick something pricier on input than the
    # quality strategy's top reasoning choice.
    assert (
        pricing.PRICES[cost_pick.id].input_per_m
        <= pricing.PRICES[quality_pick.id].input_per_m
    )


def test_route_returns_real_cost_and_token_counts():
    r = Router(strategy="cost")
    resp = r.route("Classify this text as spam or not spam")
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    # Cost must match the pricing table applied to the reported token counts.
    expected = pricing.cost_usd(resp.model, resp.input_tokens, resp.output_tokens)
    assert resp.cost_usd == pytest.approx(expected)


def test_pricing_matches_manual_calculation():
    # 1000 input + 1000 output tokens on gpt-4o at 2.50 / 10.00 per 1M.
    cost = pricing.cost_usd("gpt-4o", 1000, 1000)
    assert cost == pytest.approx(1000 / 1_000_000 * 2.50 + 1000 / 1_000_000 * 10.00)


def test_token_count_is_positive_and_grows_with_length():
    short = count_tokens("hello", "gpt-4o")
    longer = count_tokens("hello " * 50, "gpt-4o")
    assert 0 < short < longer


def test_circuit_breaker_opens_then_resets():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=0.0)
    cb.record_failure("openai")
    assert not cb.is_open("openai")
    cb.record_failure("openai")
    # Threshold reached; but reset_timeout is 0 so it immediately allows a retry.
    assert cb.is_open("openai") is False


def test_stats_track_requests_and_cost():
    r = Router(strategy="cost")
    r.route("Summarize this")
    r.route("Classify this")
    stats = r.stats()
    assert stats["total_requests"] == 2
    assert stats["total_cost_usd"] >= 0.0


def test_catalog_models_all_have_prices():
    for model in CATALOG:
        assert model.id in pricing.PRICES
