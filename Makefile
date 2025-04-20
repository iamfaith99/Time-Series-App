# Time Series Forecasting Streamlit Application
# Makefile for standardized commands

.PHONY: run test lint format check setup-kernel

# Variables
POETRY = poetry
APP_NAME = "Time Series App.py"
KERNEL_NAME = time-series-app

# Run the application
run:
	@echo "Starting Streamlit application..."
	$(POETRY) run streamlit run $(APP_NAME)

# Ensure Jupyter kernel is set up to use Poetry's virtual environment
setup-kernel:
	@echo "Setting up Jupyter kernel to use Poetry's virtual environment..."
	$(POETRY) run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "Python ($(KERNEL_NAME))"
	@echo "Kernel setup complete."

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(POETRY) install
	@echo "Dependencies installed."

# Format code with black
format:
	@echo "Formatting code..."
	$(POETRY) run black $(APP_NAME)
	@echo "Formatting complete."

# Lint code with ruff
lint:
	@echo "Linting code..."
	$(POETRY) run ruff check $(APP_NAME)
	@echo "Linting complete."

# Type check with mypy
typecheck:
	@echo "Type checking..."
	./typecheck.sh
	@echo "Type checking complete."

# Run essential checks (format and lint)
check: format lint
	@echo "Essential checks completed."

# Run all checks including type checking (may have errors)
check-all: format lint typecheck
	@echo "All checks completed (type checking may have errors)."

# Help command
help:
	@echo "Available commands:"
	@echo "  make run        - Run the Streamlit application"
	@echo "  make install    - Install dependencies"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Lint code with ruff"
	@echo "  make typecheck  - Type check with mypy"
	@echo "  make check      - Run all checks"
	@echo "  make setup-kernel - Set up Jupyter kernel for Poetry environment"
