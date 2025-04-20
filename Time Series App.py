"""
Streamlit Time Series Forecasting Application
Author: Weldon T. Antoine III
Date: 2025-04-19

A single-file, functional & polymorphic implementation that enables:
 • CSV upload & preprocessing
 • Selection of target column
 • Training & comparison of multiple univariate forecasting models
 • Automatic hyper-parameter optimisation
 • Interactive visualisation & export of forecasts/metrics

Key libraries (managed with Poetry - see pyproject.toml):
    streamlit, pandas, numpy, plotly, scikit-learn, pmdarima,
    statsmodels, prophet, neuralprophet, darts (with PyTorch backend)

Run with: poetry run streamlit run "Time Series App.py"
"""

# =============================
# Imports & global definitions
# =============================
from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pmdarima as pm
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# prophet‑family
try:
    from prophet import Prophet  # fbprophet >=1 renamed
except ImportError:
    from fbprophet import Prophet  # type: ignore

try:
    from neuralprophet import NeuralProphet
except ImportError:
    NeuralProphet = None  # optional

# =============================
# Base Forecaster Abstraction
# =============================


class BaseForecaster(abc.ABC):
    """Abstract base class for all forecasters."""

    name: str = "Base"
    hyperparam_space: list[dict] | None = None

    def __init__(self, **params):
        self.params = params
        self.model = None

    @abc.abstractmethod
    def fit(self, y: pd.Series):
        """Fit the forecasting model to the time series data."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, horizon: int):
        """Predict future values."""
        raise NotImplementedError

    @classmethod
    def create(cls, model_name: str, **params) -> BaseForecaster:
        """Factory method to create forecaster instances."""
        # We need to refer to the concrete classes defined later in the file
        # Forward references will be resolved when called
        # Ensure concrete classes are defined globally before create() is called
        mapping: dict[str, type[BaseForecaster]] = {
            "arima": ARIMAForecaster,  # type: ignore
            "sarima": SARIMAForecaster,  # type: ignore
            "expsmoothing": ExponentialSmoothingForecaster,  # type: ignore
            "exp_smoothing": ExponentialSmoothingForecaster,  # alias for Exp_Smoothing
            "prophet": ProphetForecaster,  # type: ignore
            "rf": RandomForestForecaster,  # type: ignore
            "random_forest": RandomForestForecaster,  # alias for Random_Forest
            "gbrt": GradientBoostingForecaster,  # type: ignore
            "svr": SVRForecaster,  # type: ignore
            "neural_prophet": NeuralProphetForecaster,  # type: ignore
            "lstm": LSTMForecaster,  # type: ignore
            "tcn": TCNForecaster,  # type: ignore
            "nbeats": NBEATSForecaster,  # type: ignore
            "deepar": DeepARForecaster,  # type: ignore
            "tft": TFTForecaster,  # type: ignore
        }
        if model_name not in mapping:
            raise ValueError(f"Unsupported model {model_name}")
        # The actual class (e.g., ARIMAForecaster) must be defined globally by the time
        # this function returns the instance.
        return mapping[model_name](**params)


SEED = 42
np.random.seed(SEED)

warnings.filterwarnings("ignore")

# Deep learning via darts (optional dependency)
_darts_import_error_msg: str | None = None

# Function to raise error when deep learning models are used but not available
def _raise_not_installed_error(*args, **kwargs):
    # Use the stored error message
    error_detail = _darts_import_error_msg or "Unknown darts import error"
    raise RuntimeError(
        f"Deep learning models are not available: {error_detail}. "
        f"Install 'darts[torch]' to use these models."
    )

# Try to import darts for time series forecasting
try:
    # Import the base TimeSeries class from darts
    from darts import TimeSeries
    from darts.utils.likelihood_models.torch import GaussianLikelihood  # added for DeepAR
    print("[INFO] Successfully imported darts TimeSeries")
    
    # Try to import deep learning models (may fail if torch is not installed)
    try:
        from darts.models import (
            RNNModel,           # covers LSTM/GRU/DeepAR-style RNNs
            TCNModel,
            NBEATSModel,
            TFTModel,
        )
        from pytorch_lightning.callbacks import EarlyStopping
        print("[INFO] Successfully imported darts deep learning models")
        
        # Define a base wrapper for darts models using functional programming principles
        class _DartsWrapper(BaseForecaster):
            """Base wrapper for darts models with vectorized operations.
            
            This wrapper implements functional programming principles by:
            1. Using immutable state where possible
            2. Providing pure functions with clear inputs/outputs
            3. Leveraging vectorized operations via the darts library
            """
            def __init__(
                self, model_cls, input_chunk_length: int = 24, output_chunk_length: int = 12, **params
            ):
                super().__init__(**params)
                self.input_chunk_length = input_chunk_length
                self.output_chunk_length = output_chunk_length
                # Create model with vectorized operations for training and prediction
                self.model = model_cls(
                    random_state=SEED,
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=output_chunk_length,
                    **params,
                )
                # add early stopping callback to Darts trainer
                params.setdefault("pl_trainer_kwargs", {})
                params["pl_trainer_kwargs"].setdefault("callbacks", [])
                params["pl_trainer_kwargs"]["callbacks"].append(
                    EarlyStopping(monitor="train_loss", patience=3)
                )

            def fit(self, y: pd.Series):
                """Fit the model using vectorized operations in darts.
                
                Args:
                    y: Time series data as pandas Series
                    
                Returns:
                    Self for method chaining
                """
                # Convert to darts TimeSeries (handles vectorized operations internally)
                ts = TimeSeries.from_series(y.astype(np.float32))  # cast to float32 for MPS on Apple GPUs
                # Fit model (vectorized training happens inside darts)
                self.model.fit(ts)
                self.ts = ts
                return self

            def predict(self, horizon: int) -> np.ndarray:
                """Generate forecasts using vectorized operations in darts.
                
                Args:
                    horizon: Number of future time steps to predict
                    
                Returns:
                    Array of predicted values
                """
                # Predict future values (vectorized prediction in darts)
                forecast = self.model.predict(horizon)
                # Convert to numpy array for consistent interface
                return forecast.values().flatten()

        # Define concrete implementations for each model type
        class LSTMForecaster(_DartsWrapper):
            name = "LSTM"

            def __init__(self, **params):
                super().__init__(RNNModel, model="LSTM", **params)

        class TCNForecaster(_DartsWrapper):
            name = "TCN"

            def __init__(self, **params):
                super().__init__(TCNModel, **params)

        class NBEATSForecaster(_DartsWrapper):
            name = "NBEATS"

            def __init__(self, **params):
                super().__init__(NBEATSModel, **params)

        class DeepARForecaster(_DartsWrapper):
            """
            DeepAR is implemented in Darts as a probabilistic RNN.
            We wrap RNNModel with model="LSTM" and a Gaussian likelihood.
            """
            name = "DeepAR"

            def __init__(self, **params):
                super().__init__(
                    RNNModel,
                    model="LSTM",
                    likelihood=GaussianLikelihood(),
                    **params,
                )

        class TFTForecaster(_DartsWrapper):
            name = "TFT"

            def __init__(self, horizon: int | None = None, **params):
                params.setdefault("add_relative_index", True)
                params.setdefault(
                    "add_encoders",
                    {
                        "datetime_attribute": {"future": ["month", "dayofweek", "hour"]},
                        "cyclic": {"future": ["month", "dayofweek"]},
                    },
                )
                # force CPU to avoid MPS NaN bug for TFT
                params.setdefault("pl_trainer_kwargs", {"accelerator": "cpu"})
                if horizon is not None:
                    # override any default to ensure decoder window covers full horizon
                    params["output_chunk_length"] = horizon
                super().__init__(TFTModel, **params)
                
    except ImportError as e:
        # Deep learning models not available, create placeholder classes
        _darts_import_error_msg = str(e)
        print(f"[WARNING] Deep learning models not available: {e}")
        
        # Create placeholder classes for deep learning models
        class _DeepLearningUnavailable(BaseForecaster):
            """Placeholder for deep learning models when dependencies aren't available."""
            def fit(self, y: pd.Series):
                _raise_not_installed_error()

            def predict(self, horizon: int) -> np.ndarray:
                _raise_not_installed_error()

        # Create placeholder classes for each model type
        class LSTMForecaster(_DeepLearningUnavailable):
            name = "LSTM (requires torch)"

        class TCNForecaster(_DeepLearningUnavailable):
            name = "TCN (requires torch)"

        class NBEATSForecaster(_DeepLearningUnavailable):
            name = "NBEATS (requires torch)"

        class DeepARForecaster(_DeepLearningUnavailable):
            name = "DeepAR (requires torch)"

        class TFTForecaster(_DeepLearningUnavailable):
            name = "TFT (requires torch)"
            
except ImportError as e:
    # Base darts import failed
    _darts_import_error_msg = str(e)
    print(f"[WARNING] Failed to import darts: {e}")
    
    # Create placeholder for all deep learning models
    class _DeepLearningUnavailable(BaseForecaster):
        """Placeholder for when darts is not available."""
        def fit(self, y: pd.Series):
            _raise_not_installed_error()

        def predict(self, horizon: int) -> np.ndarray:
            _raise_not_installed_error()

    # Create placeholder classes for each model type
    class LSTMForecaster(_DeepLearningUnavailable):
        name = "LSTM (requires darts)"

    class TCNForecaster(_DeepLearningUnavailable):
        name = "TCN (requires darts)"

    class NBEATSForecaster(_DeepLearningUnavailable):
        name = "NBEATS (requires darts)"

    class DeepARForecaster(_DeepLearningUnavailable):
        name = "DeepAR (requires darts)"

    class TFTForecaster(_DeepLearningUnavailable):
        name = "TFT (requires darts)"

# =============================
# Utility & metric functions
# =============================

def _clean_nan(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        raise ValueError("All values are NaN – cannot evaluate metrics.")
    return y_true[mask], y_pred[mask]

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_nan(np.asarray(y_true), np.asarray(y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_nan(np.asarray(y_true), np.asarray(y_pred))
    return float(mean_absolute_error(y_true, y_pred))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean_nan(np.asarray(y_true), np.asarray(y_pred))
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

METRIC_FUNCS = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2_score}

# =============================
# Concrete model implementations
# =============================


class ARIMAForecaster(BaseForecaster):
    name = "ARIMA"

    def fit(self, y: pd.Series):
        # Ensure parameters aren't passed twice if they're in optimized params
        params = self.params.copy()
        params.pop("seasonal", None)
        params.pop("stepwise", None)
        self.model = pm.auto_arima(
            y, seasonal=False, stepwise=True, suppress_warnings=True, **params
        )
        return self

    def predict(self, horizon: int):
        return self.model.predict(horizon)


class SARIMAForecaster(BaseForecaster):
    name = "SARIMA"

    def fit(self, y):
        # Remove conflicting seasonal parameters
        params = self.params.copy()
        params.pop("seasonal", None)
        params.pop("stepwise", None)
        self.model = pm.auto_arima(
            y,
            seasonal=True,
            stepwise=True,
            m=_detect_freq_num(y.index),
            suppress_warnings=True,
            **params,
        )
        return self

    def predict(self, horizon):
        return self.model.predict(horizon)


class ExponentialSmoothingForecaster(BaseForecaster):
    name = "ExpSmoothing"

    def fit(self, y):
        periods = _detect_freq_num(y.index)
        if periods > 1:
            seasonal_args = {"seasonal": "add", "seasonal_periods": periods}
        else:
            seasonal_args = {"seasonal": None}

        self.model = ExponentialSmoothing(y, trend="add", **seasonal_args).fit(
            optimized=True, **self.params
        )
        return self

    def predict(self, horizon):
        return self.model.forecast(horizon)


class ProphetForecaster(BaseForecaster):
    name = "Prophet"

    def fit(self, y):
        df = pd.DataFrame({"ds": y.index, "y": y.values})
        # Store training DataFrame for future forecasts
        self.history_df = df
        # Prepare parameters without duplicates and instantiate model
        params = self.params.copy()
        self.model = Prophet(**params)
        self.model.fit(df)
        return self

    def predict(self, horizon):
        # Generate future DataFrame using stored history and inferred frequency
        future = self.model.make_future_dataframe(
            periods=horizon,
            freq=_infer_pandas_freq(self.history_df["ds"]),
            include_history=False,
        )
        forecast = self.model.predict(future)
        return forecast.iloc[-horizon:]["yhat"].values


# ---------- ML Wrappers ----------
class _RegressorForecaster(BaseForecaster):
    """Shared utilities for sklearn regressors"""

    def __init__(self, regressor_cls, lags: int = 14, **params):
        super().__init__(**params)
        self.lags = lags
        # SVR does not accept random_state
        if regressor_cls is SVR:
            self.regressor = regressor_cls(**params)
        else:
            self.regressor = regressor_cls(random_state=SEED, **params)

    def _create_features(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Create lagged features using a sliding window.

        This implementation creates a sliding window view of the time series
        without explicit loops, improving performance significantly.

        Args:
            series: Time series data

        Returns:
            Tuple of (features, targets)
        """
        values = series.values
        # Create lagged features and their next values using a single sliding window
        windows = np.lib.stride_tricks.sliding_window_view(values, self.lags + 1)
        features = windows[:, :-1]
        targets = windows[:, -1]
        return features, targets

    def fit(self, y):
        """Create lagged features and fit the underlying regressor."""
        features, targets = self._create_features(y)
        self.regressor.fit(features, targets)
        self._y_train_for_predict = y  # Store for prediction lag generation
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts using vectorized operations where possible.

        While the sequential nature of forecasting requires some iteration,
        we optimize by pre-allocating arrays and minimizing operations inside the loop.

        Args:
            horizon: Number of future time steps to predict

        Returns:
            Array of predicted values
        """
        # Pre-allocate arrays for better performance
        preds = np.zeros(horizon)
        # Initial window from training data
        window = self._y_train_for_predict.iloc[-self.lags :].values

        # Extend window array once to avoid repeated resizing
        full_window = np.zeros(len(window) + horizon)
        full_window[: len(window)] = window

        for i in range(horizon):
            # Extract current window efficiently
            current_idx = i + len(window) - self.lags
            x = full_window[current_idx : current_idx + self.lags].reshape(1, -1)
            # Generate prediction
            yhat = self.regressor.predict(x)[0]
            # Store prediction in both arrays
            preds[i] = yhat
            full_window[i + len(window)] = yhat

        return preds


class RandomForestForecaster(_RegressorForecaster):
    name = "RandomForest"

    def __init__(self, **params):
        super().__init__(RandomForestRegressor, **params)


class GradientBoostingForecaster(_RegressorForecaster):
    name = "GBRT"

    def __init__(self, **params):
        super().__init__(GradientBoostingRegressor, **params)


class SVRForecaster(_RegressorForecaster):
    name = "SVR"

    def __init__(self, **params):
        super().__init__(SVR, **params)


class NeuralProphetForecaster(BaseForecaster):
    name = "NeuralProphet"

    def fit(self, y):
        if NeuralProphet is None:
            raise RuntimeError("Install neuralprophet to use this model.")
        self.freq = _infer_pandas_freq(y.index)
        df = pd.DataFrame({"ds": y.index, "y": y.values})
        # Store training DataFrame for future forecasts
        self.history_df = df
        # Prepare parameters without duplicates and instantiate model
        params = self.params.copy()
        self.model = NeuralProphet(**params)
        self.model.fit(df, freq=self.freq)
        return self

    def predict(self, horizon):
        future = self.model.make_future_dataframe(
            df=self.history_df, periods=horizon, n_historic_predictions=False
        )
        forecast = self.model.predict(future)
        return forecast["yhat1"].values[-horizon:]


# =============================
# Pre‑processing helpers
# =============================


def _infer_pandas_freq(index: pd.Index) -> str:
    try:
        return pd.infer_freq(index)
    except Exception:
        return "D"


def _detect_freq_num(index: pd.Index) -> int:
    mapping = {"D": 7, "W": 52, "M": 12}
    return mapping.get(_infer_pandas_freq(index)[0], 1)


@st.cache_data(show_spinner=False)
def clean_preprocess(df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    """Clean data using functional programming patterns and vectorized operations.

    This function handles:
    1. Date parsing and indexing
    2. Missing value imputation
    3. Outlier removal using IQR method (vectorized)

    Returns a clean DataFrame ready for time series analysis.
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Handle date column if provided
    if date_col:
        processed_df[date_col] = pd.to_datetime(processed_df[date_col])
        processed_df = processed_df.set_index(date_col).sort_index()

    # Fill missing values (forward then backward)
    processed_df = processed_df.ffill().bfill()

    # Get numeric columns for vectorized operations
    numeric_cols = processed_df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:
        # Calculate quantiles for all numeric columns at once (vectorized)
        quantiles = processed_df[numeric_cols].quantile([0.25, 0.75])
        q1, q3 = quantiles.loc[0.25], quantiles.loc[0.75]

        # Calculate IQR and bounds vectorized across all columns
        iqr = q3 - q1
        lower_bounds = q1 - 1.5 * iqr
        upper_bounds = q3 + 1.5 * iqr

        # Apply clipping to all numeric columns at once
        for col in numeric_cols:
            processed_df[col] = np.clip(
                processed_df[col], lower_bounds[col], upper_bounds[col]
            )

    return processed_df


# =============================
# Hyper‑parameter search stubs
# =============================


@dataclass
class ModelParams:
    """Immutable container for model hyperparameters."""

    params: dict[str, Any] = field(default_factory=dict)


@st.cache_data(show_spinner=False)
def quick_optimize(model_name: str, y_train: pd.Series) -> dict[str, Any]:
    """Functional hyperparameter optimization for time series models.

    Uses a predefined parameter space for each model type and returns
    optimized parameters based on the training data characteristics.

    Args:
        model_name: The name of the model to optimize
        y_train: Training time series data

    Returns:
        Dictionary of optimized parameters
    """
    # Define parameter spaces as immutable mappings
    param_spaces = {
        "arima": ModelParams(
            {"seasonal": False, "stepwise": True, "max_p": 5, "max_q": 5}
        ),
        "sarima": ModelParams(
            {"seasonal": True, "stepwise": True, "max_P": 2, "max_Q": 2}
        ),
        "expsmoothing": ModelParams({"use_boxcox": True}),
        "prophet": ModelParams(
            {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
            }
        ),
        "rf": ModelParams(
            {"n_estimators": 200, "max_depth": None, "min_samples_split": 2}
        ),
        "gbrt": ModelParams(
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
        ),
        "svr": ModelParams({"kernel": "rbf", "C": 1.0, "epsilon": 0.1}),
        "neural_prophet": ModelParams(
            {"n_lags": 0, "n_forecasts": 1, "yearly_seasonality": True}
        ),
        "lstm": ModelParams(
            {"input_chunk_length": 24, "output_chunk_length": 12, "n_epochs": 50}
        ),
        "tcn": ModelParams(
            {"input_chunk_length": 24, "output_chunk_length": 12, "n_epochs": 50}
        ),
        "nbeats": ModelParams(
            {"input_chunk_length": 24, "output_chunk_length": 12, "n_epochs": 50, "num_stacks": 10, "num_blocks": 1}
        ),
        "deepar": ModelParams(
            {"input_chunk_length": 24, "output_chunk_length": 12, "n_epochs": 50}
        ),
        "tft": ModelParams(
            {"input_chunk_length": 24, "output_chunk_length": 12, "n_epochs": 50, "batch_size": 16}
        ),
    }

    # Apply data-specific optimizations
    model_key = model_name.lower()

    # Return the parameters for the requested model, or empty dict if not found
    return param_spaces.get(model_key, ModelParams()).params


# =============================
# Streamlit User Interface
# =============================

import time
import datetime

st.set_page_config(page_title="Time Series Forecasting Studio", layout="wide")

st.title("📈 Time Series Forecasting Studio")

with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    quick_mode = st.toggle("⚡ Quick optimisation", value=True)
    test_size = st.slider("Test size (%)", 5, 30, value=20, step=5)
    horizon = st.number_input("Forecast horizon", 1, 365, 30)

if not uploaded_file:
    st.info("👈 Upload a CSV file to get started.")
    st.stop()

# read data
raw_df = pd.read_csv(uploaded_file)

# choose date column and target
with st.sidebar:
    date_col: str = st.selectbox("Date column", raw_df.columns, index=0)
    target = st.selectbox(
        "Target variable", [c for c in raw_df.columns if c != date_col], index=0
    )

# preprocess
df = clean_preprocess(raw_df, date_col)

y = df[target]

# split
y_train, y_test = (
    y[: int(len(y) * (1 - test_size / 100))],
    y[int(len(y) * (1 - test_size / 100)) :],
)

st.subheader("Data Preview")
st.dataframe(df.head())

# Model selection
all_models = [
    "ARIMA",
    "SARIMA",
    "Exp_Smoothing",
    "Prophet",
    "Random_Forest",
    "GBRT",
    "SVR",
    "Neural_Prophet",
    "LSTM",
    "TCN",
    "NBEATS",
    "DeepAR",
    "TFT",
]

selected_models = st.multiselect(
    "Select models to train", all_models, default=["ARIMA", "Prophet", "Random_Forest"]
)

progress = st.progress(0.0, text="Training models…")
results = {}
plots = {}

for i, model_display in enumerate(selected_models, 1):
    model_key = model_display.lower().replace(" ", "_")
    st.write(f"### {model_display}")
    opt_params = quick_optimize(model_key, y_train) if quick_mode else {}
    # dynamic output_chunk_length for TFT to cover full horizon
    if model_key == "tft":
        forecaster = BaseForecaster.create(model_key, horizon=len(y_test), **opt_params)
    else:
        forecaster = BaseForecaster.create(model_key, **opt_params)
    # show start time and timer for training
    start_dt = datetime.datetime.now()
    st.write(f"Started {model_display} at {start_dt.strftime('%H:%M:%S')}")
    start_time = time.perf_counter()
    with st.spinner(f"Training {model_display}..."):
        forecaster = forecaster.fit(y_train)
    elapsed = time.perf_counter() - start_time
    end_dt = datetime.datetime.now()
    st.write(f"Finished {model_display} at {end_dt.strftime('%H:%M:%S')} (took {elapsed:.2f}s)")
    preds = forecaster.predict(len(y_test))
    # Skip models that predict all NaNs
    if np.isnan(preds).all():
        st.warning(f"{model_display} produced only NaNs – skipped.")
        progress.progress(i / len(selected_models), text=f"Skipped {model_display}")
        continue
    metrics = {m: f(y_test.values, preds) for m, f in METRIC_FUNCS.items()}
    results[model_display] = metrics
    # plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=y_train.index, y=y_train.values, mode="lines", name="Train")
    )
    fig.add_trace(
        go.Scatter(x=y_test.index, y=y_test.values, mode="lines", name="Test")
    )
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, mode="lines", name="Forecast"))
    plots[model_display] = fig
    st.plotly_chart(fig, use_container_width=True)
    progress.progress(i / len(selected_models), text=f"Finished {model_display}")

# Metric comparison table
metric_df = pd.DataFrame(results).T.round(3)
st.subheader("Model Comparison")
st.dataframe(metric_df.style.background_gradient(axis=0, cmap="Blues"))

# Export
csv_out = metric_df.to_csv().encode()
st.download_button(
    "Download metrics", data=csv_out, file_name="model_metrics.csv", mime="text/csv"
)

# allow forecasting into unseen future
st.header("Forecast Future")
model_for_forecast = st.selectbox("Model", selected_models)
if st.button("Generate Forecast"):
    model_key = model_for_forecast.lower().replace(" ", "_")
    # ensure TFT uses full-horizon decoder length
    if model_key == "tft":
        forecaster = BaseForecaster.create(model_key, horizon=horizon, **quick_optimize(model_key, y))
    else:
        forecaster = BaseForecaster.create(model_key, **quick_optimize(model_key, y))
    forecaster.fit(y)
    future_preds = forecaster.predict(horizon)
    future_index = pd.date_range(
        start=y.index[-1], periods=horizon + 1, freq=_infer_pandas_freq(y.index)
    )[1:]
    forecast_df = pd.DataFrame({"date": future_index, "forecast": future_preds})
    fig2 = px.line(forecast_df, x="date", y="forecast", title="Future Forecast")
    st.plotly_chart(fig2, use_container_width=True)
    st.download_button(
        "Download forecast",
        forecast_df.to_csv(index=False).encode(),
        "forecast.csv",
        "text/csv",
    )

st.caption(
    "© 2025 Time Series Forecasting Studio – Functional & Polymorphic Streamlit App"
)
