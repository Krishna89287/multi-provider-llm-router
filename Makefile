.PHONY: install test lint clean run

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

run:
	python router/llm_router.py

lint:
	flake8 . --max-line-length=120 --exclude=.venv

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
