.PHONY: format lint test clean

format:
	black src scripts

lint:
	ruff check src scripts

test:
	pytest tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +