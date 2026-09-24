.PHONY: install demo benchmark test clean

install:
	pip install -e ".[dev]"

demo:
	python -m llm_router.cli --strategy cost

benchmark:
	python benchmark/benchmark.py

test:
	pytest -q

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
