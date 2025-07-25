# Brain Go Brrr - Modern Python ML Project Makefile
# 2025 Best Practices with uv, ruff, and modern tooling

.PHONY: help install dev-install test test-fast test-cov lint format type-check quality check pre-commit clean build docs serve-docs train preprocess evaluate serve notebook mlflow dvc-setup docker-build docker-run benchmark profile

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m # No Color

# Project variables
PROJECT_NAME := brain-go-brrr
SRC_DIR := src/brain_go_brrr
TEST_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := config
NOTEBOOKS_DIR := notebooks

# Python and uv settings
PYTHON_VERSION := 3.11
UV := uv
PYTHON := $(UV) run python
PIP := $(UV) pip
PYTEST := $(UV) run pytest
RUFF := $(UV) run ruff

# Pytest options - can be overridden via environment
PYTEST_BASE_OPTS ?= -v
PYTEST_NO_PLUGINS ?= -p no:pytest_benchmark -p no:xdist
PYTEST_PARALLEL ?= -p xdist -n auto
MYPY := $(UV) run mypy
JUPYTER := $(UV) run jupyter
PRE_COMMIT := $(UV) run pre-commit

# Default target
help: ## Show this help message
	@echo "$(CYAN)Brain Go Brrr - Modern ML Project Commands$(NC)"
	@echo "$(YELLOW)================================================$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Development Workflow:$(NC)"
	@echo "  1. $(GREEN)make install$(NC)     - Install dependencies"
	@echo "  2. $(GREEN)make dev-install$(NC) - Install dev dependencies"
	@echo "  3. $(GREEN)make pre-commit$(NC)  - Setup pre-commit hooks"
	@echo "  4. $(GREEN)make test$(NC)        - Run tests"
	@echo "  5. $(GREEN)make check$(NC)       - Run all quality checks"

##@ Dependencies

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(UV) sync --no-dev

dev-install: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(UV) sync
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

update: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(UV) lock --upgrade

##@ Development

dev-setup: dev-install pre-commit ## Complete development setup
	@echo "$(GREEN)Development environment setup complete!$(NC)"

pre-commit: ## Setup pre-commit hooks
	@echo "$(GREEN)Setting up pre-commit hooks...$(NC)"
	$(PRE_COMMIT) install
	$(PRE_COMMIT) install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

notebook: ## Start Jupyter Lab
	@echo "$(GREEN)Starting Jupyter Lab...$(NC)"
	$(JUPYTER) lab --notebook-dir=$(NOTEBOOKS_DIR)

##@ Code Quality

lint: ## Run linting with ruff
	@echo "$(GREEN)Running linting checks...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) --fix
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	$(RUFF) format $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## Run full strict type checking (CI/pre-commit)
	@echo "$(CYAN)Running full type checks...$(NC)"
	@rm -rf .mypy_cache 2>/dev/null || true
	$(MYPY) src/brain_go_brrr
	@echo "$(GREEN)Type checking complete!$(NC)"

fast-type-check: ## Fast type checking for development (uses cache)
	@echo "$(CYAN)Running fast type checks...$(NC)"
	$(MYPY) --ignore-missing-imports src/brain_go_brrr
	@echo "$(GREEN)Fast type checking complete!$(NC)"

type-check-file: ## Check specific file: make type-check-file FILE=path/to/file.py
	@echo "$(CYAN)Type checking $(FILE)...$(NC)"
	$(MYPY) $(FILE)
	@echo "$(GREEN)File type checking complete!$(NC)"

quality: lint format type-check ## Run all code quality checks
	@echo "$(GREEN)All quality checks complete!$(NC)"

check: test quality ## Run all tests and quality checks
	@echo "$(GREEN)All checks passed!$(NC)"

##@ Testing

test: ## Run all tests (default - excludes benchmarks)
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTEST) $(TEST_DIR) $(PYTEST_BASE_OPTS) --ignore=tests/benchmarks $(PYTEST_NO_PLUGINS)

test-unit: ## Run unit tests only (fast)
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) tests/unit $(PYTEST_BASE_OPTS) -q $(PYTEST_NO_PLUGINS)

test-perf: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	$(PYTEST) tests/benchmarks $(PYTEST_BASE_OPTS) -m perf -p pytest_benchmark

test-parallel: ## Run tests in parallel (with xdist)
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) $(PYTEST_BASE_OPTS) $(PYTEST_PARALLEL) --ignore=tests/benchmarks

test-fast: test-unit  ## Legacy alias for test-unit

test-cov: ## Run fast tests with coverage (80% minimum)
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) \
		--cov=src/brain_go_brrr \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=80 \
		-m "not slow and not integration and not external"

test-integration: ## Run integration tests with timeout
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "integration" --timeout=900 --tb=short

test-all-cov: ## Run ALL tests with coverage report
	@echo "$(GREEN)Running all tests with full coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) \
		--cov=src/brain_go_brrr \
		--cov-report=term-missing \
		--cov-report=html \
		-m ""

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	$(PYTEST) $(TEST_DIR) -f -m "not slow"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTEST) $(TEST_DIR) --benchmark-only

##@ Machine Learning

train: ## Train EEGPT model
	@echo "$(GREEN)Starting model training...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli train

preprocess: ## Preprocess EEG data
	@echo "$(GREEN)Preprocessing EEG data...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli preprocess

evaluate: ## Evaluate trained model
	@echo "$(GREEN)Evaluating model...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli evaluate

serve: ## Serve model via API
	@echo "$(GREEN)Starting model server...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli serve

##@ MLOps

mlflow: ## Start MLflow tracking server
	@echo "$(GREEN)Starting MLflow server...$(NC)"
	$(UV) run mlflow server --host 0.0.0.0 --port 5000

dvc-setup: ## Initialize DVC for data versioning
	@echo "$(GREEN)Setting up DVC...$(NC)"
	$(UV) run dvc init
	@echo "$(GREEN)DVC initialized!$(NC)"

dvc-status: ## Check DVC status
	@echo "$(GREEN)Checking DVC status...$(NC)"
	$(UV) run dvc status

dvc-push: ## Push data to remote storage
	@echo "$(GREEN)Pushing data to remote storage...$(NC)"
	$(UV) run dvc push

dvc-pull: ## Pull data from remote storage
	@echo "$(GREEN)Pulling data from remote storage...$(NC)"
	$(UV) run dvc pull

##@ Documentation

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	$(UV) run mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Starting documentation server...$(NC)"
	$(UV) run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(GREEN)Deploying documentation...$(NC)"
	$(UV) run mkdocs gh-deploy

##@ Building & Packaging

build: ## Build package
	@echo "$(GREEN)Building package...$(NC)"
	$(UV) build

publish: ## Publish package to PyPI
	@echo "$(GREEN)Publishing package...$(NC)"
	$(UV) publish

##@ Docker

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):latest .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -it --rm -p 8000:8000 $(PROJECT_NAME):latest

docker-dev: ## Run Docker container in development mode
	@echo "$(GREEN)Running Docker container in dev mode...$(NC)"
	docker run -it --rm -v $(PWD):/app -p 8000:8000 $(PROJECT_NAME):latest

##@ Profiling & Debugging

profile: ## Run profiling on training
	@echo "$(GREEN)Running profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile_output.prof -m brain_go_brrr.cli train
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative').print_stats(20)"

debug: ## Run with debugger
	@echo "$(GREEN)Starting debug session...$(NC)"
	$(PYTHON) -m pdb -m brain_go_brrr.cli train

##@ Utilities

clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .tox/
	@echo "$(GREEN)Cleanup complete!$(NC)"

reset: clean ## Reset environment completely
	@echo "$(GREEN)Resetting environment...$(NC)"
	rm -rf .venv/
	$(UV) sync

version: ## Show version information
	@echo "$(GREEN)Version Information:$(NC)"
	@$(PYTHON) -m brain_go_brrr.cli version
	@echo "Python: $(shell python --version)"
	@echo "UV: $(shell uv --version)"

env-info: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python: $(shell python --version)"
	@echo "UV: $(shell uv --version)"
	@echo "Platform: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@$(PYTHON) -c "import sys; print(f'Python Path: {sys.executable}')"

##@ Git Hooks

pre-push: test quality ## Run pre-push checks
	@echo "$(GREEN)Running pre-push checks...$(NC)"
	@echo "$(GREEN)All pre-push checks passed!$(NC)"

ci: ## Run CI pipeline locally
	@echo "$(GREEN)Running CI pipeline...$(NC)"
	$(MAKE) clean
	$(MAKE) dev-install
	$(MAKE) check
	$(MAKE) build
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

##@ Examples

example-train: ## Run training example
	@echo "$(GREEN)Running training example...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli train --debug

example-preprocess: ## Run preprocessing example
	@echo "$(GREEN)Running preprocessing example...$(NC)"
	$(PYTHON) -m brain_go_brrr.cli preprocess data/raw data/processed --debug

# Help target should be first
.DEFAULT_GOAL := help
