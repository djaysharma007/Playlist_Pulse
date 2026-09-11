"""
PlaylistPulse - Stream Count Regression

Target:
    log_stream_count

Models:
    1. Ordinary Least Squares (Normal Equation)
    2. Gradient Descent

Metrics:
    MAE, MSE, RMSE, R2

MLflow:
    Parameters, metrics, and experiment artifacts are logged.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "src" / "data" / "regression.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
LEARNING_RATE = 0.01
N_ITERATIONS = 2000


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "year",
]

TARGET_COLUMN = "log_stream_count"


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load and validate the regression dataset."""

    print(f"Loading regression data from: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Regression dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [
        "stream_count",
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove rows containing missing values
    df = df.dropna(
        subset=required_columns
    ).copy()

    # Verify log-transformed target consistency
    expected_log = np.log1p(df["stream_count"])

    max_error = np.max(
        np.abs(
            expected_log - df[TARGET_COLUMN]
        )
    )

    print(f"Samples loaded: {len(df)}")
    print(
        f"Maximum log-target consistency error: "
        f"{max_error:.8f}"
    )

    if max_error > 1e-6:
        raise ValueError(
            "log_stream_count does not match "
            "log1p(stream_count)."
        )

    return df


# ============================================================
# OLS / LEAST SQUARES
# ============================================================

def fit_ols(X, y):
    """
    Fit Ordinary Least Squares using the normal equation.

    A pseudo-inverse is used for numerical stability.
    """

    X_with_bias = np.column_stack(
        [np.ones(len(X)), X]
    )

    weights = (
        np.linalg.pinv(
            X_with_bias.T @ X_with_bias
        )
        @ X_with_bias.T
        @ y
    )

    return weights


def predict_ols(X, weights):
    """Generate predictions using the OLS model."""

    X_with_bias = np.column_stack(
        [np.ones(len(X)), X]
    )

    return X_with_bias @ weights


# ============================================================
# GRADIENT DESCENT
# ============================================================

def compute_cost(X, y, weights):
    """Compute the mean squared error cost."""

    predictions = X @ weights
    errors = predictions - y

    return np.mean(errors ** 2) / 2


def gradient_descent(
    X,
    y,
    learning_rate=LEARNING_RATE,
    n_iterations=N_ITERATIONS,
):
    """
    Fit linear regression using Gradient Descent.
    """

    weights = np.zeros(X.shape[1])
    cost_history = []

    m = len(y)

    for _ in range(n_iterations):

        predictions = X @ weights
        errors = predictions - y

        gradients = (
            X.T @ errors
        ) / m

        weights -= (
            learning_rate * gradients
        )

        cost = compute_cost(
            X,
            y,
            weights,
        )

        cost_history.append(cost)

    return weights, cost_history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, y_pred):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("PlaylistPulse - Stream Count Regression")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=float
    )

    y = df[TARGET_COLUMN].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    print(
        "\nTraining OLS / Least Squares..."
    )

    ols_weights = fit_ols(
        X_train_scaled,
        y_train,
    )

    ols_predictions = predict_ols(
        X_test_scaled,
        ols_weights,
    )

    ols_metrics = evaluate_model(
        y_test,
        ols_predictions,
    )

    print("\nOLS Results")

    for metric, value in ols_metrics.items():
        print(
            f"{metric}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Gradient Descent
    # --------------------------------------------------------

    print(
        "\nTraining Gradient Descent..."
    )

    X_train_gd = np.column_stack(
        [
            np.ones(len(X_train_scaled)),
            X_train_scaled,
        ]
    )

    X_test_gd = np.column_stack(
        [
            np.ones(len(X_test_scaled)),
            X_test_scaled,
        ]
    )

    gd_weights, cost_history = gradient_descent(
        X_train_gd,
        y_train,
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
    )

    gd_predictions = (
        X_test_gd @ gd_weights
    )

    gd_metrics = evaluate_model(
        y_test,
        gd_predictions,
    )

    print(
        "\nGradient Descent Results"
    )

    for metric, value in gd_metrics.items():
        print(
            f"{metric}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(
        {
            "Model": [
                "OLS",
                "Gradient Descent",
            ],
            "MAE": [
                ols_metrics["MAE"],
                gd_metrics["MAE"],
            ],
            "MSE": [
                ols_metrics["MSE"],
                gd_metrics["MSE"],
            ],
            "RMSE": [
                ols_metrics["RMSE"],
                gd_metrics["RMSE"],
            ],
            "R2": [
                ols_metrics["R2"],
                gd_metrics["R2"],
            ],
        }
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    print(
        "\nLogging experiment to MLflow..."
    )

    mlflow.set_experiment(
        "PlaylistPulse-Stream-Count-Regression"
    )

    with mlflow.start_run(
        run_name="OLS_vs_Gradient_Descent"
    ):

        # Parameters
        mlflow.log_param(
            "target",
            TARGET_COLUMN,
        )

        mlflow.log_param(
            "n_features",
            len(FEATURE_COLUMNS),
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE,
        )

        mlflow.log_param(
            "gd_iterations",
            N_ITERATIONS,
        )

        # OLS metrics
        mlflow.log_metrics(
            {
                "ols_mae": ols_metrics["MAE"],
                "ols_mse": ols_metrics["MSE"],
                "ols_rmse": ols_metrics["RMSE"],
                "ols_r2": ols_metrics["R2"],
            }
        )

        # Gradient Descent metrics
        mlflow.log_metrics(
            {
                "gd_mae": gd_metrics["MAE"],
                "gd_mse": gd_metrics["MSE"],
                "gd_rmse": gd_metrics["RMSE"],
                "gd_r2": gd_metrics["R2"],
            }
        )

        # ----------------------------------------------------
        # Save convergence history
        # ----------------------------------------------------

        convergence_path = (
            FIGURES_DIR
            / "gradient_descent_convergence.csv"
        )

        pd.DataFrame(
            {
                "iteration": np.arange(
                    1,
                    len(cost_history) + 1,
                ),
                "cost": cost_history,
            }
        ).to_csv(
            convergence_path,
            index=False,
        )

        mlflow.log_artifact(
            str(convergence_path)
        )

        print(
            "MLflow run logged successfully."
        )


if __name__ == "__main__":
    main()