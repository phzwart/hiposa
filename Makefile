.PHONY: help install install-dev test lint format clean docs build publish

help: ## Show this help message
	@echo "HiPoSa - Hierarchical Poisson Sampling"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

test: ## Run the test suite
	pytest tests/ -v --cov=hiposa --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linting checks
	flake8 hiposa/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy hiposa/ --ignore-missing-imports

format: ## Format code with black and isort
	black hiposa/ tests/ examples/
	isort hiposa/ tests/ examples/

check-format: ## Check if code is properly formatted
	black --check hiposa/ tests/ examples/
	isort --check-only hiposa/ tests/ examples/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docs: ## Build documentation
	cd docs && make html

build: ## Build the package
	python -m build

publish: ## Publish to PyPI (requires twine)
	twine upload dist/*

example: ## Run the basic usage example
	python examples/basic_usage.py

pre-commit: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

setup-dev: install-dev pre-commit ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"
