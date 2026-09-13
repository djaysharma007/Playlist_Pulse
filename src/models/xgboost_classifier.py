from pathlib import Path
import warnings

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "classification.csv"
)

CATEGORICAL_FEATURES = [
    "userId",
    "itemId",
]

NUMERICAL_FEATURES = [
    "timestamp",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

TARGET = "y"


def load_data():
    """Load and validate the classification dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Classification dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]

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
    ).copy()

    if data.empty:
        raise ValueError(
            "No valid samples remain after removing missing values."
        )

    unique_targets = sorted(
        data[TARGET].unique().tolist()
    )

    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(
            f"Target '{TARGET}' must contain only "
            f"binary values 0 and 1. "
            f"Found: {unique_targets}"
        )

    data[TARGET] = data[TARGET].astype(int)

    return data


def build_preprocessor():
    """Build leakage-safe feature preprocessing."""

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def evaluate_model(model, X, y):
    """Calculate classification metrics."""

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
        ),
    }


def train_xgboost(
    params,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Train XGBoost with validation-based early stopping."""

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    return model


def main():

    print("=" * 70)
    print("PlaylistPulse XGBoost Classification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    data = load_data()

    X = data[FEATURES]
    y = data[TARGET]

    print(
        f"\nSamples loaded: {len(data)}"
    )

    print("\nClass distribution:")
    print(
        y.value_counts()
        .sort_index()
    )

    # ------------------------------------------------------------------
    # Development / test split
    # ------------------------------------------------------------------

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # Training / validation split
    # ------------------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
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

    # ------------------------------------------------------------------
    # Fit preprocessing only on training data
    # ------------------------------------------------------------------

    print(
        "\nFitting feature preprocessing..."
    )

    preprocessor = build_preprocessor()

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    X_development_transformed = (
        preprocessor.transform(
            X_development
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    feature_names = np.asarray(
        preprocessor.get_feature_names_out(),
        dtype=str,
    )

    print(
        f"Transformed feature count: "
        f"{X_train_transformed.shape[1]}"
    )

    # ------------------------------------------------------------------
    # XGBoost hyperparameter search
    # ------------------------------------------------------------------

    configurations = [
        {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 700,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 700,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
        {
            "n_estimators": 700,
            "max_depth": 7,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
    ]

    print(
        f"\nEvaluating "
        f"{len(configurations)} "
        f"XGBoost configurations..."
    )

    validation_results = []

    for index, params in enumerate(
        configurations,
        start=1,
    ):

        model = train_xgboost(
            params,
            X_train_transformed,
            y_train,
            X_validation_transformed,
            y_validation,
        )

        metrics = evaluate_model(
            model,
            X_validation_transformed,
            y_validation,
        )

        best_iteration = (
            model.best_iteration
            if model.best_iteration is not None
            else params["n_estimators"] - 1
        )

        validation_results.append(
            {
                **params,
                "validation_accuracy": metrics[
                    "accuracy"
                ],
                "validation_precision": metrics[
                    "precision"
                ],
                "validation_recall": metrics[
                    "recall"
                ],
                "validation_f1": metrics[
                    "f1"
                ],
                "validation_roc_auc": metrics[
                    "roc_auc"
                ],
                "best_iteration": int(
                    best_iteration
                ),
            }
        )

        print(
            f"Configuration "
            f"{index}/{len(configurations)} completed | "
            f"F1={metrics['f1']:.6f} | "
            f"ROC-AUC={metrics['roc_auc']:.6f} | "
            f"Best iteration={best_iteration}"
        )

    validation_results_df = pd.DataFrame(
        validation_results
    )

    # ------------------------------------------------------------------
    # Select best configuration
    # ------------------------------------------------------------------

    validation_results_df = (
        validation_results_df
        .sort_values(
            by=[
                "validation_f1",
                "validation_roc_auc",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    best_configuration = (
        validation_results_df.iloc[0]
    )

    best_params = {
        "n_estimators": int(
            best_configuration[
                "n_estimators"
            ]
        ),
        "max_depth": int(
            best_configuration[
                "max_depth"
            ]
        ),
        "learning_rate": float(
            best_configuration[
                "learning_rate"
            ]
        ),
        "subsample": float(
            best_configuration[
                "subsample"
            ]
        ),
        "colsample_bytree": float(
            best_configuration[
                "colsample_bytree"
            ]
        ),
        "min_child_weight": int(
            best_configuration[
                "min_child_weight"
            ]
        ),
    }

    print(
        "\nBest validation configuration:"
    )

    for parameter, value in best_params.items():
        print(
            f"{parameter}: {value}"
        )

    print(
        f"Validation F1: "
        f"{best_configuration['validation_f1']:.6f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{best_configuration['validation_roc_auc']:.6f}"
    )

    print(
        f"Best boosting iteration: "
        f"{int(best_configuration['best_iteration'])}"
    )

    # ------------------------------------------------------------------
    # Final XGBoost model
    #
    # The validation split remains available for early stopping.
    # The test set remains untouched until final evaluation.
    # ------------------------------------------------------------------

    print(
        "\nTraining final XGBoost model..."
    )

    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
        **best_params,
    )

    final_model.fit(
        X_development_transformed,
        y_development,
        eval_set=[
            (
                X_validation_transformed,
                y_validation,
            )
        ],
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------

    xgb_metrics = evaluate_model(
        final_model,
        X_test_transformed,
        y_test,
    )

    final_best_iteration = (
        final_model.best_iteration
        if final_model.best_iteration is not None
        else best_params["n_estimators"] - 1
    )

    print(
        "\nEarly stopping results:"
    )

    print(
        f"Configured estimators: "
        f"{best_params['n_estimators']}"
    )

    print(
        f"Best boosting iteration: "
        f"{final_best_iteration}"
    )

    print(
        f"Effective trees: "
        f"{final_best_iteration + 1}"
    )

    # ------------------------------------------------------------------
    # Logistic Regression baseline
    # ------------------------------------------------------------------

    print(
        "\nTraining Logistic Regression baseline..."
    )

    logistic_regression = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    logistic_regression.fit(
        X_development_transformed,
        y_development,
    )

    logistic_metrics = evaluate_model(
        logistic_regression,
        X_test_transformed,
        y_test,
    )

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "XGBoost",
            ],
            "Accuracy": [
                logistic_metrics["accuracy"],
                xgb_metrics["accuracy"],
            ],
            "Precision": [
                logistic_metrics["precision"],
                xgb_metrics["precision"],
            ],
            "Recall": [
                logistic_metrics["recall"],
                xgb_metrics["recall"],
            ],
            "F1": [
                logistic_metrics["f1"],
                xgb_metrics["f1"],
            ],
            "ROC_AUC": [
                logistic_metrics["roc_auc"],
                xgb_metrics["roc_auc"],
            ],
        }
    )

    print(
        "\nTest-set model comparison:"
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nXGBoost confusion matrix:"
    )

    print(
        xgb_metrics["confusion_matrix"]
    )

    # ------------------------------------------------------------------
    # Native XGBoost feature importance
    # ------------------------------------------------------------------

    native_importance = (
        final_model.feature_importances_
    )

    native_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": native_importance,
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    print(
        "\nTop XGBoost feature importances:"
    )

    print(
        native_importance_df
        .head(15)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    # ------------------------------------------------------------------
    # SHAP analysis
    # ------------------------------------------------------------------

    print(
        "\nCalculating SHAP feature importance..."
    )

    shap_sample_size = min(
        1000,
        len(X_test_transformed),
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    shap_indices = rng.choice(
        len(X_test_transformed),
        size=shap_sample_size,
        replace=False,
    )

    X_shap = X_test_transformed[
        shap_indices
    ]

    explainer = shap.TreeExplainer(
        final_model
    )

    shap_values = explainer.shap_values(
        X_shap
    )

    shap_values_array = np.asarray(
        shap_values
    )

    # Handle possible extra dimensions returned
    # by different SHAP/XGBoost combinations.
    if shap_values_array.ndim == 3:
        shap_values_array = (
            shap_values_array[:, :, 0]
        )

    mean_abs_shap = np.mean(
        np.abs(shap_values_array),
        axis=0,
    )

    shap_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    print(
        "\nTop SHAP feature importances:"
    )

    print(
        shap_importance_df
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    mlflow.set_experiment(
        "PlaylistPulse-XGBoost"
    )

    with mlflow.start_run(
        run_name="XGBoost_Classification"
    ):

        for parameter, value in best_params.items():

            mlflow.log_param(
                parameter,
                value,
            )

        mlflow.log_param(
            "early_stopping_rounds",
            50,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        # Validation metrics
        mlflow.log_metric(
            "validation_accuracy",
            float(
                best_configuration[
                    "validation_accuracy"
                ]
            ),
        )

        mlflow.log_metric(
            "validation_precision",
            float(
                best_configuration[
                    "validation_precision"
                ]
            ),
        )

        mlflow.log_metric(
            "validation_recall",
            float(
                best_configuration[
                    "validation_recall"
                ]
            ),
        )

        mlflow.log_metric(
            "validation_f1",
            float(
                best_configuration[
                    "validation_f1"
                ]
            ),
        )

        mlflow.log_metric(
            "validation_roc_auc",
            float(
                best_configuration[
                    "validation_roc_auc"
                ]
            ),
        )

        # Early stopping
        mlflow.log_metric(
            "best_iteration",
            float(final_best_iteration),
        )

        mlflow.log_metric(
            "effective_trees",
            float(
                final_best_iteration + 1
            ),
        )

        # XGBoost test metrics
        mlflow.log_metric(
            "xgboost_test_accuracy",
            xgb_metrics["accuracy"],
        )

        mlflow.log_metric(
            "xgboost_test_precision",
            xgb_metrics["precision"],
        )

        mlflow.log_metric(
            "xgboost_test_recall",
            xgb_metrics["recall"],
        )

        mlflow.log_metric(
            "xgboost_test_f1",
            xgb_metrics["f1"],
        )

        mlflow.log_metric(
            "xgboost_test_roc_auc",
            xgb_metrics["roc_auc"],
        )

        # Logistic comparison
        mlflow.log_metric(
            "logistic_test_f1",
            logistic_metrics["f1"],
        )

        mlflow.log_metric(
            "logistic_test_roc_auc",
            logistic_metrics["roc_auc"],
        )

        # SHAP importance
        for _, row in (
            shap_importance_df
            .head(20)
            .iterrows()
        ):

            feature_name = row["feature"]

            safe_feature_name = (
                feature_name
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )

            mlflow.log_metric(
                f"shap_mean_abs_{safe_feature_name}",
                float(
                    row["mean_abs_shap"]
                ),
            )

        mlflow.xgboost.log_model(
            final_model,
            name="xgboost_model",
        )

        print(
            "\nMLflow run logged successfully."
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "XGBoost classification pipeline completed."
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal XGBoost metrics:"
    )

    print(
        f"Accuracy : "
        f"{xgb_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{xgb_metrics['precision']:.6f}"
    )

    print(
        f"Recall   : "
        f"{xgb_metrics['recall']:.6f}"
    )

    print(
        f"F1       : "
        f"{xgb_metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{xgb_metrics['roc_auc']:.6f}"
    )

    print(
        f"Best iteration: "
        f"{final_best_iteration}"
    )


if __name__ == "__main__":
    main()