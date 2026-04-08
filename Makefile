.PHONY: clean install test

clean:
	rm -rf log/* data/* __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v
