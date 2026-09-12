"""
PlaylistPulse - Regularized Stream Count Regression

Target:
    log_stream_count

Models:
    1. Ordinary Least Squares baseline
    2. Ridge Regression
    3. Lasso Regression

Evaluation:
    MAE
    MSE
    RMSE
    R2

The stream-count target is modeled in log space to reduce the
effect of the heavy-tailed distribution of stream counts.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "src" / "data" / "regression.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]

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


def load_data():
    """Load and validate the stream-count regression dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Regression dataset not found: {DATA_PATH}"
        )

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

    actual_log_target = data[TARGET_COLUMN].to_numpy(
        dtype=float
    )

    consistency_error = np.max(
        np.abs(actual_log_target - expected_log_target)
    )

    if consistency_error > 1e-6:
        raise ValueError(
            "log_stream_count is inconsistent with "
            "log1p(stream_count). "
            f"Maximum error: {consistency_error:.8f}"
        )

    return data


def evaluate_model(model, X_test, y_test):
    """Calculate regression evaluation metrics."""

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


def search_ridge_alpha(
    X_train,
    X_validation,
    y_train,
    y_validation,
):
    """Select Ridge alpha using validation RMSE."""

    results = []

    for alpha in RIDGE_ALPHAS:
        model = Ridge(
            alpha=alpha,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        mse = mean_squared_error(
            y_validation,
            predictions,
        )

        results.append(
            {
                "alpha": alpha,
                "RMSE": np.sqrt(mse),
            }
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["RMSE"].idxmin()
    ]

    best_alpha = float(
        best_row["alpha"]
    )

    return best_alpha, results_df


def search_lasso_alpha(
    X_train,
    X_validation,
    y_train,
    y_validation,
):
    """Select Lasso alpha using validation RMSE."""

    results = []

    for alpha in LASSO_ALPHAS:
        model = Lasso(
            alpha=alpha,
            max_iter=10000,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        mse = mean_squared_error(
            y_validation,
            predictions,
        )

        results.append(
            {
                "alpha": alpha,
                "RMSE": np.sqrt(mse),
            }
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["RMSE"].idxmin()
    ]

    best_alpha = float(
        best_row["alpha"]
    )

    return best_alpha, results_df


def main():
    print("=" * 60)
    print("PlaylistPulse - Regularized Stream Count Regression")
    print("=" * 60)

    print(
        f"\nLoading regression data from:\n{DATA_PATH}"
    )

    data = load_data()

    print(
        f"Samples loaded: {len(data)}"
    )

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # ---------------------------------------------------------
    # Development/Test split
    # ---------------------------------------------------------

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    # ---------------------------------------------------------
    # Training/Validation split
    # ---------------------------------------------------------

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_development,
            y_development,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ---------------------------------------------------------
    # Scale training/validation/test data
    # ---------------------------------------------------------

    selection_scaler = StandardScaler()

    X_train_scaled = selection_scaler.fit_transform(
        X_train
    )

    X_validation_scaled = selection_scaler.transform(
        X_validation
    )

    # ---------------------------------------------------------
    # OLS baseline
    # ---------------------------------------------------------

    print("\nTraining OLS baseline...")

    ols_model = LinearRegression()

    ols_model.fit(
        X_train_scaled,
        y_train,
    )

    # ---------------------------------------------------------
    # Ridge alpha selection
    # ---------------------------------------------------------

    print(
        "\nSelecting Ridge regularization strength..."
    )

    best_ridge_alpha, ridge_search = search_ridge_alpha(
        X_train_scaled,
        X_validation_scaled,
        y_train,
        y_validation,
    )

    print(
        f"Best Ridge alpha: {best_ridge_alpha}"
    )

    # ---------------------------------------------------------
    # Lasso alpha selection
    # ---------------------------------------------------------

    print(
        "\nSelecting Lasso regularization strength..."
    )

    best_lasso_alpha, lasso_search = search_lasso_alpha(
        X_train_scaled,
        X_validation_scaled,
        y_train,
        y_validation,
    )

    print(
        f"Best Lasso alpha: {best_lasso_alpha}"
    )

    # ---------------------------------------------------------
    # Refit models on complete development set
    # ---------------------------------------------------------

    final_scaler = StandardScaler()

    X_development_scaled = final_scaler.fit_transform(
        X_development
    )

    X_test_scaled = final_scaler.transform(
        X_test
    )

    # OLS
    ols_model = LinearRegression()

    ols_model.fit(
        X_development_scaled,
        y_development,
    )

    ols_metrics = evaluate_model(
        ols_model,
        X_test_scaled,
        y_test,
    )

    # Ridge
    ridge_model = Ridge(
        alpha=best_ridge_alpha,
    )

    ridge_model.fit(
        X_development_scaled,
        y_development,
    )

    ridge_metrics = evaluate_model(
        ridge_model,
        X_test_scaled,
        y_test,
    )

    # Lasso
    lasso_model = Lasso(
        alpha=best_lasso_alpha,
        max_iter=10000,
    )

    lasso_model.fit(
        X_development_scaled,
        y_development,
    )

    lasso_metrics = evaluate_model(
        lasso_model,
        X_test_scaled,
        y_test,
    )

    # ---------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "Model": "OLS",
                **ols_metrics,
            },
            {
                "Model": "Ridge",
                **ridge_metrics,
            },
            {
                "Model": "Lasso",
                **lasso_metrics,
            },
        ]
    )

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Coefficient sparsity
    # ---------------------------------------------------------

    lasso_zero_coefficients = np.sum(
        np.isclose(
            lasso_model.coef_,
            0.0,
            atol=1e-8,
        )
    )

    print(
        "\nLasso zero coefficients: "
        f"{lasso_zero_coefficients}/{len(FEATURE_COLUMNS)}"
    )

    # ---------------------------------------------------------
    # MLflow tracking
    # ---------------------------------------------------------

    mlflow.set_experiment(
        "PlaylistPulse-Stream-Count-Regularization"
    )

    with mlflow.start_run(
        run_name="OLS_Ridge_Lasso_Comparison"
    ):
        mlflow.log_param(
            "target",
            TARGET_COLUMN,
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "validation_size",
            VALIDATION_SIZE,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "ridge_alpha",
            best_ridge_alpha,
        )

        mlflow.log_param(
            "lasso_alpha",
            best_lasso_alpha,
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS),
        )

        for model_name, metrics in [
            ("ols", ols_metrics),
            ("ridge", ridge_metrics),
            ("lasso", lasso_metrics),
        ]:
            for metric_name, value in metrics.items():
                mlflow.log_metric(
                    f"{model_name}_{metric_name.lower()}",
                    value,
                )

        mlflow.log_metric(
            "lasso_zero_coefficients",
            int(lasso_zero_coefficients),
        )

        mlflow.log_text(
            results.to_csv(index=False),
            "model_comparison.csv",
        )

        mlflow.log_text(
            ridge_search.to_csv(index=False),
            "ridge_alpha_search.csv",
        )

        mlflow.log_text(
            lasso_search.to_csv(index=False),
            "lasso_alpha_search.csv",
        )

    print(
        "\nMLflow run logged successfully."
    )

    print(
        "\nRegularized regression pipeline completed."
    )


if __name__ == "__main__":
    main()