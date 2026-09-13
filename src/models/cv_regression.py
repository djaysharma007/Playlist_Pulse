"""
PlaylistPulse - Cross-Validated Stream Count Regression

Purpose:
    Tune regularized regression models using cross-validation and
    select the regularization strength using the one-standard-error rule.

Models:
    1. Ordinary Least Squares
    2. Ridge Regression
    3. Lasso Regression

Target:
    log_stream_count

Evaluation:
    Cross-validation RMSE
    Test MAE
    Test MSE
    Test RMSE
    Test R2

MLflow:
    Cross-validation results, selected hyperparameters, final metrics,
    and model-selection artifacts are logged.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "src" / "data" / "regression.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

RIDGE_ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

LASSO_ALPHAS = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
]

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and validate the regression dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Regression dataset not found: {DATA_PATH}"
        )

    print(f"Loading regression data from:\n{DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [
        "stream_count",
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    if (data["stream_count"] < 0).any():
        raise ValueError(
            "stream_count contains negative values."
        )

    expected_log_target = np.log1p(
        data["stream_count"].to_numpy(dtype=float)
    )

    actual_log_target = data[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    consistency_error = np.max(
        np.abs(
            actual_log_target - expected_log_target
        )
    )

    if consistency_error > 1e-6:
        raise ValueError(
            "log_stream_count is inconsistent with "
            "log1p(stream_count). "
            f"Maximum error: {consistency_error}"
        )

    print(f"Samples loaded: {len(data)}")

    return data


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_ols():
    """Create the OLS pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )


def build_ridge(alpha):
    """Create a Ridge regression pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=alpha),
            ),
        ]
    )


def build_lasso(alpha):
    """Create a Lasso regression pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Lasso(
                    alpha=alpha,
                    max_iter=20000,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(model, X, y, cv):
    """
    Perform K-fold cross-validation.

    Negative RMSE is converted back to positive RMSE for readability.
    """

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        return_train_score=False,
    )

    rmse = -scores["test_rmse"]
    mae = -scores["test_mae"]
    r2 = scores["test_r2"]

    return {
        "RMSE_mean": rmse.mean(),
        "RMSE_std": rmse.std(ddof=1),
        "MAE_mean": mae.mean(),
        "R2_mean": r2.mean(),
    }


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def search_ridge(X, y, cv):
    """Evaluate Ridge across the candidate alpha values."""

    results = []

    for alpha in RIDGE_ALPHAS:

        model = build_ridge(alpha)

        metrics = run_cross_validation(
            model,
            X,
            y,
            cv,
        )

        results.append(
            {
                "model": "Ridge",
                "alpha": alpha,
                **metrics,
            }
        )

    return pd.DataFrame(results)


def search_lasso(X, y, cv):
    """Evaluate Lasso across the candidate alpha values."""

    results = []

    for alpha in LASSO_ALPHAS:

        model = build_lasso(alpha)

        metrics = run_cross_validation(
            model,
            X,
            y,
            cv,
        )

        results.append(
            {
                "model": "Lasso",
                "alpha": alpha,
                **metrics,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# One-standard-error rule
# ---------------------------------------------------------------------------

def select_one_standard_error(results):
    """
    Select the simplest model whose mean CV RMSE is within one standard
    error of the best-performing model.

    For regularized regression, a larger alpha means stronger
    regularization. Therefore, among acceptable candidates, we choose
    the largest alpha.
    """

    best_index = results["RMSE_mean"].idxmin()

    best_rmse = results.loc[
        best_index,
        "RMSE_mean",
    ]

    best_std = results.loc[
        best_index,
        "RMSE_std",
    ]

    n_folds = N_SPLITS

    standard_error = best_std / np.sqrt(n_folds)

    threshold = best_rmse + standard_error

    acceptable = results[
        results["RMSE_mean"] <= threshold
    ].copy()

    selected = acceptable.loc[
        acceptable["alpha"].idxmax()
    ]

    return {
        "selected_alpha": float(
            selected["alpha"]
        ),
        "best_rmse": float(best_rmse),
        "best_std": float(best_std),
        "standard_error": float(standard_error),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(model, X_train, y_train, X_test, y_test):
    """Train the model and evaluate it on the untouched test set."""

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    return {
        "MAE": mean_absolute_error(
            y_test,
            predictions,
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(
            y_test,
            predictions,
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():

    print("=" * 60)
    print("PlaylistPulse - Cross-Validated Stream Count Regression")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    data = load_data()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # -----------------------------------------------------------------------
    # Hold out test set
    # -----------------------------------------------------------------------

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print()
    print(f"Development samples: {len(X_development)}")
    print(f"Testing samples: {len(X_test)}")

    # -----------------------------------------------------------------------
    # Cross-validation configuration
    # -----------------------------------------------------------------------

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------------------
    # OLS baseline
    # -----------------------------------------------------------------------

    print()
    print("Evaluating OLS baseline...")

    ols_cv = run_cross_validation(
        build_ols(),
        X_development,
        y_development,
        cv,
    )

    # -----------------------------------------------------------------------
    # Ridge search
    # -----------------------------------------------------------------------

    print()
    print("Searching Ridge regularization strength...")

    ridge_results = search_ridge(
        X_development,
        y_development,
        cv,
    )

    ridge_best = ridge_results.loc[
        ridge_results["RMSE_mean"].idxmin()
    ]

    ridge_selection = select_one_standard_error(
        ridge_results
    )

    print(
        f"Best Ridge alpha by CV RMSE: "
        f"{ridge_best['alpha']}"
    )

    print(
        f"Ridge alpha selected by one-standard-error rule: "
        f"{ridge_selection['selected_alpha']}"
    )

    # -----------------------------------------------------------------------
    # Lasso search
    # -----------------------------------------------------------------------

    print()
    print("Searching Lasso regularization strength...")

    lasso_results = search_lasso(
        X_development,
        y_development,
        cv,
    )

    lasso_best = lasso_results.loc[
        lasso_results["RMSE_mean"].idxmin()
    ]

    lasso_selection = select_one_standard_error(
        lasso_results
    )

    print(
        f"Best Lasso alpha by CV RMSE: "
        f"{lasso_best['alpha']}"
    )

    print(
        f"Lasso alpha selected by one-standard-error rule: "
        f"{lasso_selection['selected_alpha']}"
    )

    # -----------------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------------

    print()
    print("Evaluating selected models on the test set...")

    ols_test = evaluate_test_set(
        build_ols(),
        X_development,
        y_development,
        X_test,
        y_test,
    )

    ridge_model = build_ridge(
        ridge_selection["selected_alpha"]
    )

    ridge_test = evaluate_test_set(
        ridge_model,
        X_development,
        y_development,
        X_test,
        y_test,
    )

    lasso_model = build_lasso(
        lasso_selection["selected_alpha"]
    )

    lasso_test = evaluate_test_set(
        lasso_model,
        X_development,
        y_development,
        X_test,
        y_test,
    )

    # -----------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "OLS",
                **ols_test,
            },
            {
                "Model": "Ridge",
                "Alpha": ridge_selection[
                    "selected_alpha"
                ],
                **ridge_test,
            },
            {
                "Model": "Lasso",
                "Alpha": lasso_selection[
                    "selected_alpha"
                ],
                **lasso_test,
            },
        ]
    )

    print()
    print("=" * 60)
    print("Cross-Validated Model Comparison")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # -----------------------------------------------------------------------
    # MLflow tracking
    # -----------------------------------------------------------------------

    mlflow.set_experiment(
        "PlaylistPulse-Cross-Validated-Regression"
    )

    with mlflow.start_run(
        run_name="Cross_Validated_OLS_Ridge_Lasso"
    ):

        mlflow.log_param(
            "target",
            TARGET_COLUMN,
        )

        mlflow.log_param(
            "n_splits",
            N_SPLITS,
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
            "ridge_best_cv_alpha",
            float(ridge_best["alpha"]),
        )

        mlflow.log_param(
            "ridge_one_se_alpha",
            ridge_selection[
                "selected_alpha"
            ],
        )

        mlflow.log_param(
            "lasso_best_cv_alpha",
            float(lasso_best["alpha"]),
        )

        mlflow.log_param(
            "lasso_one_se_alpha",
            lasso_selection[
                "selected_alpha"
            ],
        )

        for metric, value in ols_cv.items():
            mlflow.log_metric(
                f"ols_cv_{metric.lower()}",
                float(value),
            )

        for model_name, metrics in [
            ("ols", ols_test),
            ("ridge", ridge_test),
            ("lasso", lasso_test),
        ]:

            for metric, value in metrics.items():
                mlflow.log_metric(
                    f"{model_name}_test_{metric.lower()}",
                    float(value),
                )

        mlflow.log_text(
            ridge_results.to_csv(index=False),
            "ridge_cv_results.csv",
        )

        mlflow.log_text(
            lasso_results.to_csv(index=False),
            "lasso_cv_results.csv",
        )

        mlflow.log_text(
            comparison.to_csv(index=False),
            "final_model_comparison.csv",
        )

        print()
        print("MLflow run logged successfully.")

    print()
    print("Cross-validated regression pipeline completed.")


if __name__ == "__main__":
    main()