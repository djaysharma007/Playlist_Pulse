"""
PlaylistPulse - Cross-Validated Logistic Classification

Purpose:
    Tune Logistic Regression regularization using stratified
    cross-validation and select the regularization strength using
    the one-standard-error rule.

Input:
    src/data/classification.csv

Features:
    userId
    itemId
    timestamp

Target:
    y

Evaluation:
    Accuracy
    Precision
    Recall
    F1
    ROC-AUC

Model selection:
    Logistic Regression with different C values.

One-standard-error rule:
    For Logistic Regression, a smaller C applies stronger regularization
    and is treated as the simpler model. Among candidates within one
    standard error of the best CV score, the smallest C is selected.

Important:
    The semantic meaning of y is intentionally not assumed by this
    pipeline. It is treated only as a binary classification target.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "classification.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

CATEGORICAL_FEATURES = [
    "userId",
    "itemId",
]

NUMERICAL_FEATURES = [
    "timestamp",
]

TARGET_COLUMN = "y"


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_data():
    """Load and validate the classification dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Classification dataset not found: {DATA_PATH}"
        )

    print(f"Loading classification data from:\n{DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
        + [TARGET_COLUMN]
    )

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

    target_values = set(
        data[TARGET_COLUMN].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target y must be binary with values 0 and 1. "
            f"Found: {sorted(target_values)}"
        )

    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Target y must contain both classes."
        )

    print(f"Samples loaded: {len(data)}")
    print(
        f"Class distribution:\n"
        f"{data[TARGET_COLUMN].value_counts().sort_index()}"
    )

    return data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor():
    """Build the leakage-safe preprocessing pipeline."""

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(C):
    """Create the preprocessing and Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(model, X, y, cv):
    """Evaluate a model using stratified cross-validation."""

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        return_train_score=False,
    )

    return {
        "Accuracy_mean": scores[
            "test_accuracy"
        ].mean(),
        "Accuracy_std": scores[
            "test_accuracy"
        ].std(ddof=1),
        "Precision_mean": scores[
            "test_precision"
        ].mean(),
        "Recall_mean": scores[
            "test_recall"
        ].mean(),
        "F1_mean": scores[
            "test_f1"
        ].mean(),
        "F1_std": scores[
            "test_f1"
        ].std(ddof=1),
        "ROC_AUC_mean": scores[
            "test_roc_auc"
        ].mean(),
        "ROC_AUC_std": scores[
            "test_roc_auc"
        ].std(ddof=1),
    }


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def search_logistic_regression(X, y, cv):
    """Evaluate Logistic Regression across candidate C values."""

    results = []

    for C in C_VALUES:

        model = build_model(C)

        metrics = run_cross_validation(
            model,
            X,
            y,
            cv,
        )

        results.append(
            {
                "model": "LogisticRegression",
                "C": C,
                **metrics,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# One-standard-error rule
# ---------------------------------------------------------------------------

def select_one_standard_error(results):
    """
    Select the simplest Logistic Regression model whose mean CV F1
    is within one standard error of the best model.

    Higher F1 is better.

    Smaller C means stronger regularization and therefore a simpler
    model. Among acceptable candidates, the smallest C is selected.
    """

    best_index = results[
        "F1_mean"
    ].idxmax()

    best_f1 = results.loc[
        best_index,
        "F1_mean",
    ]

    best_std = results.loc[
        best_index,
        "F1_std",
    ]

    standard_error = (
        best_std
        / np.sqrt(N_SPLITS)
    )

    threshold = (
        best_f1
        - standard_error
    )

    acceptable = results[
        results["F1_mean"] >= threshold
    ].copy()

    selected = acceptable.loc[
        acceptable["C"].idxmin()
    ]

    return {
        "selected_C": float(
            selected["C"]
        ),
        "best_f1": float(
            best_f1
        ),
        "best_std": float(
            best_std
        ),
        "standard_error": float(
            standard_error
        ),
        "threshold": float(
            threshold
        ),
    }


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Fit the model and evaluate it on the untouched test set."""

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():

    print("=" * 60)
    print("PlaylistPulse - Cross-Validated Logistic Classification")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    data = load_data()

    X = data[
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
    ]

    y = data[TARGET_COLUMN]

    # -----------------------------------------------------------------------
    # Hold out test set
    # -----------------------------------------------------------------------

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print()
    print(
        f"Development samples: "
        f"{len(X_development)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # -----------------------------------------------------------------------
    # Stratified cross-validation
    # -----------------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------------------
    # Cross-validation search
    # -----------------------------------------------------------------------

    print()
    print(
        "Searching Logistic Regression "
        "regularization strength..."
    )

    cv_results = search_logistic_regression(
        X_development,
        y_development,
        cv,
    )

    # -----------------------------------------------------------------------
    # Best CV model
    # -----------------------------------------------------------------------

    best_cv = cv_results.loc[
        cv_results["F1_mean"].idxmax()
    ]

    print()
    print(
        f"Best C by CV F1: "
        f"{best_cv['C']}"
    )

    print(
        f"Best CV F1: "
        f"{best_cv['F1_mean']:.6f}"
    )

    # -----------------------------------------------------------------------
    # One-standard-error selection
    # -----------------------------------------------------------------------

    selection = select_one_standard_error(
        cv_results
    )

    print(
        f"One-standard-error selected C: "
        f"{selection['selected_C']}"
    )

    print(
        f"One-standard-error threshold: "
        f"{selection['threshold']:.6f}"
    )

    # -----------------------------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------------------------

    print()
    print(
        "Evaluating baseline and selected "
        "models on the test set..."
    )

    # Honest baseline: C = 1.0
    baseline_model = build_model(
        C=1.0
    )

    baseline_test = evaluate_test_set(
        baseline_model,
        X_development,
        y_development,
        X_test,
        y_test,
    )

    # Selected model
    selected_model = build_model(
        C=selection["selected_C"]
    )

    selected_test = evaluate_test_set(
        selected_model,
        X_development,
        y_development,
        X_test,
        y_test,
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression Baseline",
                "C": 1.0,
                **baseline_test,
            },
            {
                "Model": "Logistic Regression CV Selected",
                "C": selection[
                    "selected_C"
                ],
                **selected_test,
            },
        ]
    )

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("Final Test Set Comparison")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # -----------------------------------------------------------------------
    # Cross-validation results
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)

    print(
        cv_results[
            [
                "C",
                "Accuracy_mean",
                "F1_mean",
                "ROC_AUC_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # -----------------------------------------------------------------------
    # MLflow tracking
    # -----------------------------------------------------------------------

    mlflow.set_experiment(
        "PlaylistPulse-Cross-Validated-Classification"
    )

    with mlflow.start_run(
        run_name="Cross_Validated_Logistic_Regression"
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
            "best_cv_C",
            float(best_cv["C"]),
        )

        mlflow.log_param(
            "one_se_C",
            selection["selected_C"],
        )

        mlflow.log_metric(
            "best_cv_f1",
            float(best_cv["F1_mean"]),
        )

        mlflow.log_metric(
            "one_se_threshold",
            selection["threshold"],
        )

        for model_name, metrics in [
            (
                "baseline",
                baseline_test,
            ),
            (
                "cv_selected",
                selected_test,
            ),
        ]:

            for metric, value in metrics.items():

                mlflow.log_metric(
                    f"{model_name}_test_{metric.lower()}",
                    float(value),
                )

        mlflow.log_text(
            cv_results.to_csv(index=False),
            "logistic_cv_results.csv",
        )

        mlflow.log_text(
            comparison.to_csv(index=False),
            "final_model_comparison.csv",
        )

        print()
        print(
            "MLflow run logged successfully."
        )

    print()
    print(
        "Cross-validated classification "
        "pipeline completed."
    )


if __name__ == "__main__":
    main()