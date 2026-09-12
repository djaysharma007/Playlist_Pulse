"""
PlaylistPulse - Logistic Regression Classification

Target:
    y

Features:
    - userId
    - itemId
    - timestamp

Feature engineering:
    - One-hot encoding for userId and itemId
    - Standardized timestamp

Evaluation:
    Accuracy
    Precision
    Recall
    F1
    ROC-AUC
    Confusion Matrix

MLflow:
    Parameters, metrics, and model coefficients are logged.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Project paths and configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "src" / "data" / "classification.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

TARGET_COLUMN = "y"

CATEGORICAL_FEATURES = [
    "userId",
    "itemId",
]

NUMERICAL_FEATURES = [
    "timestamp",
]


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
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    # Target validation
    unique_targets = sorted(data[TARGET_COLUMN].unique())

    if unique_targets != [0, 1]:
        raise ValueError(
            f"Expected binary target values [0, 1], "
            f"found {unique_targets}"
        )

    # Numeric validation
    for column in CATEGORICAL_FEATURES + NUMERICAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    print(f"Samples loaded: {len(data)}")
    print()
    print("Target distribution:")
    print(data[TARGET_COLUMN].value_counts().sort_index())

    return data


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_preprocessor():
    """
    Build the preprocessing pipeline.

    userId and itemId are treated as categorical variables rather than
    continuous numerical quantities.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "timestamp",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X, y):
    """Calculate classification metrics."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y,
            probabilities,
        ),
    }

    return metrics, predictions, probabilities


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():

    print("=" * 60)
    print("PlaylistPulse - Logistic Regression Classification")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    data = load_data()

    X = data[
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    ]

    y = data[TARGET_COLUMN]

    # -----------------------------------------------------------------------
    # Train / test split
    # -----------------------------------------------------------------------

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Development → training / validation
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    print()
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Testing samples: {len(X_test)}")

    # -----------------------------------------------------------------------
    # Build preprocessing + logistic regression pipeline
    # -----------------------------------------------------------------------

    preprocessor = build_preprocessor()

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    print()
    print("Training logistic regression...")

    pipeline.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------------------------
    # Validation evaluation
    # -----------------------------------------------------------------------

    validation_metrics, _, _ = evaluate_model(
        pipeline,
        X_validation,
        y_validation,
    )

    print()
    print("=" * 60)
    print("Validation Metrics")
    print("=" * 60)

    for metric, value in validation_metrics.items():
        print(f"{metric:<12} {value:.6f}")

    # -----------------------------------------------------------------------
    # Final training
    #
    # After model selection, retrain on the complete development set
    # before evaluating on the untouched test set.
    # -----------------------------------------------------------------------

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print()
    print("Retraining on complete development set...")

    final_pipeline.fit(
        X_development,
        y_development,
    )

    # -----------------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------------

    test_metrics, test_predictions, test_probabilities = evaluate_model(
        final_pipeline,
        X_test,
        y_test,
    )

    print()
    print("=" * 60)
    print("Test Metrics")
    print("=" * 60)

    for metric, value in test_metrics.items():
        print(f"{metric:<12} {value:.6f}")

    # -----------------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    print()
    print("=" * 60)
    print("Confusion Matrix")
    print("=" * 60)

    print("                 Predicted")
    print("                 0      1")
    print(f"Actual  0      {cm[0, 0]:<6} {cm[0, 1]}")
    print(f"        1      {cm[1, 0]:<6} {cm[1, 1]}")

    # -----------------------------------------------------------------------
    # MLflow tracking
    # -----------------------------------------------------------------------

    mlflow.set_experiment(
        "PlaylistPulse-Logistic-Regression"
    )

    with mlflow.start_run(
        run_name="Logistic_Regression_Baseline"
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
            "solver",
            "liblinear",
        )

        mlflow.log_param(
            "max_iter",
            2000,
        )

        mlflow.log_param(
            "categorical_features",
            ",".join(CATEGORICAL_FEATURES),
        )

        mlflow.log_param(
            "numerical_features",
            ",".join(NUMERICAL_FEATURES),
        )

        mlflow.log_param(
            "training_samples",
            len(X_development),
        )

        mlflow.log_param(
            "test_samples",
            len(X_test),
        )

        for metric, value in test_metrics.items():
            mlflow.log_metric(
                metric,
                value,
            )

        mlflow.log_metric(
            "true_negatives",
            int(cm[0, 0]),
        )

        mlflow.log_metric(
            "false_positives",
            int(cm[0, 1]),
        )

        mlflow.log_metric(
            "false_negatives",
            int(cm[1, 0]),
        )

        mlflow.log_metric(
            "true_positives",
            int(cm[1, 1]),
        )

        print()
        print("MLflow run logged successfully.")

    print()
    print("Logistic regression pipeline completed.")


if __name__ == "__main__":
    main()