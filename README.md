# Time Series Forecasting Studio

A powerful, interactive Streamlit application for time series forecasting that implements multiple forecasting models using a functional and polymorphic approach.

## Features

- **CSV Upload & Preprocessing**: Upload your time series data and automatically handle date parsing, missing values, and outliers
- **Multiple Forecasting Models**: Compare results from various forecasting techniques:
  - Statistical models (ARIMA, SARIMA, Exponential Smoothing)
  - Machine learning models (Random Forest, Gradient Boosting, SVR)
  - Deep learning models (LSTM, TCN, NBEATS, DeepAR, TFT via Darts)
  - Prophet and NeuralProphet
- **Automatic Hyperparameter Optimization**: Quick optimization to find suitable parameters
- **Interactive Visualization**: Compare model performance with interactive Plotly charts
- **Metrics Comparison**: Evaluate models using RMSE, MAE, MAPE, and R²
- **Future Forecasting**: 
  - Generate and export forecasts for future periods
  - **Training Timer**: Displays start, end, and elapsed time for each model’s training
  - **Early Stopping**: Deep‑learning models halt automatically upon convergence
  - **TFT CPU Fallback**: Works around MPS NaN bug by running TFT on CPU
  - **Optimised Quick Mode**: TFT batch_size and N‑BEATS architecture tweaks for faster prototyping

## Implementation Highlights

- **Functional Programming Approach**: Pure functions, immutable data structures, and vectorized operations
- **Polymorphic Design**: Abstract base class with concrete implementations for each model type
- **Vectorized Operations**: Optimized performance with NumPy's vectorized operations
- **Type Annotations**: Comprehensive type hints for better code quality
- **Caching**: Streamlit caching for responsive UI

## Installation

This project uses Poetry for dependency management:

```bash
# Clone the repository
git clone https://github.com/iamfaith99/Time-Series-App.git
cd Time-Series-App

# Install dependencies
poetry install

# Install development dependencies (optional)
poetry install --with dev
```

## Usage

Run the application using the Makefile:

```bash
make run
```

Or directly with Poetry:

```bash
poetry run streamlit run "Time Series App.py"
```

## Development

This project follows professional Python development practices:

```bash
# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Run all checks
make check

# Set up Jupyter kernel for notebooks
make setup-kernel
```

## Requirements

- Python 3.9+
- Dependencies listed in pyproject.toml

## Optional Dependencies

Deep learning models require additional setup:
- PyTorch (automatically installed with `darts[pytorch]`)
- CUDA (for GPU acceleration, optional)

## License

[Your License]

## Author

Weldon T. Antoine III
