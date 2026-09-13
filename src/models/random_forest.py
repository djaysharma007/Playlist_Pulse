from pathlib import Path
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "src" / "data" / "classification.csv"


CATEGORICAL_FEATURES = [
    "userId",
    "itemId",
]

NUMERICAL_FEATURES = [
    "timestamp",
]

TARGET = "y"


def load_data():
    """Load and validate the classification dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Classification dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
        + [TARGET]
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
    ).copy()

    if data.empty:
        raise ValueError("No valid samples remain after removing missing values.")

    unique_targets = sorted(data[TARGET].unique().tolist())

    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(
            f"Target '{TARGET}' must contain only binary values 0 and 1. "
            f"Found: {unique_targets}"
        )

    data[TARGET] = data[TARGET].astype(int)

    return data


def build_preprocessor():
    """Create leakage-safe preprocessing."""

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
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


def build_random_forest(
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
):
    """Build the Random Forest classification pipeline."""

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def evaluate_model(model, X, y):
    """Calculate classification metrics."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
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


def main():

    print("=" * 70)
    print("PlaylistPulse Random Forest Classification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    data = load_data()

    X = data[
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    ]

    y = data[TARGET]

    print(f"\nSamples loaded: {len(data)}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # ------------------------------------------------------------------
    # Development / test split
    # ------------------------------------------------------------------

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # Training / validation split
    # ------------------------------------------------------------------

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Testing samples: {len(X_test)}")

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    configurations = []

    n_estimators_values = [
        200,
        500,
    ]

    max_depth_values = [
        None,
        8,
        12,
    ]

    min_samples_split_values = [
        2,
        10,
    ]

    min_samples_leaf_values = [
        1,
        5,
    ]

    max_features_values = [
        "sqrt",
        "log2",
    ]

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            for min_samples_split in min_samples_split_values:
                for min_samples_leaf in min_samples_leaf_values:
                    for max_features in max_features_values:

                        configurations.append(
                            {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "min_samples_split": min_samples_split,
                                "min_samples_leaf": min_samples_leaf,
                                "max_features": max_features,
                            }
                        )

    print(
        f"\nEvaluating {len(configurations)} Random Forest configurations..."
    )

    validation_results = []

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):

        model = build_random_forest(
            n_estimators=configuration["n_estimators"],
            max_depth=configuration["max_depth"],
            min_samples_split=configuration["min_samples_split"],
            min_samples_leaf=configuration["min_samples_leaf"],
            max_features=configuration["max_features"],
        )

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        validation_results.append(
            {
                **configuration,
                "validation_accuracy": metrics["accuracy"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1": metrics["f1"],
                "validation_roc_auc": metrics["roc_auc"],
            }
        )

        if index % 10 == 0 or index == len(configurations):
            print(
                f"Evaluated {index}/{len(configurations)} configurations"
            )

    validation_results_df = pd.DataFrame(
        validation_results
    )

    # ------------------------------------------------------------------
    # Select best configuration
    # ------------------------------------------------------------------

    validation_results_df = validation_results_df.sort_values(
        by=[
            "validation_f1",
            "validation_roc_auc",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    best_configuration = validation_results_df.iloc[0]

    best_n_estimators = int(
        best_configuration["n_estimators"]
    )

    best_max_depth = best_configuration["max_depth"]

    if pd.isna(best_max_depth):
        best_max_depth = None
    else:
        best_max_depth = int(best_max_depth)

    best_min_samples_split = int(
        best_configuration["min_samples_split"]
    )

    best_min_samples_leaf = int(
        best_configuration["min_samples_leaf"]
    )

    best_max_features = str(
        best_configuration["max_features"]
    )

    print("\nBest validation configuration:")
    print(f"n_estimators: {best_n_estimators}")
    print(f"max_depth: {best_max_depth}")
    print(
        f"min_samples_split: {best_min_samples_split}"
    )
    print(
        f"min_samples_leaf: {best_min_samples_leaf}"
    )
    print(f"max_features: {best_max_features}")

    print(
        f"Validation F1: "
        f"{best_configuration['validation_f1']:.6f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{best_configuration['validation_roc_auc']:.6f}"
    )

    # ------------------------------------------------------------------
    # Final Random Forest
    # ------------------------------------------------------------------

    print("\nTraining final Random Forest...")

    final_random_forest = build_random_forest(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        max_features=best_max_features,
    )

    final_random_forest.fit(
        X_development,
        y_development,
    )

    random_forest_metrics = evaluate_model(
        final_random_forest,
        X_test,
        y_test,
    )

    # ------------------------------------------------------------------
    # OOB error
    # ------------------------------------------------------------------

    random_forest_classifier = (
        final_random_forest
        .named_steps["classifier"]
    )

    oob_score = random_forest_classifier.oob_score_
    oob_error = 1.0 - oob_score

    print("\nRandom Forest OOB results:")
    print(f"OOB score: {oob_score:.6f}")
    print(f"OOB error: {oob_error:.6f}")

    # ------------------------------------------------------------------
    # Logistic Regression baseline
    # ------------------------------------------------------------------

    print("\nTraining Logistic Regression baseline...")

    logistic_regression = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_regression.fit(
        X_development,
        y_development,
    )

    logistic_metrics = evaluate_model(
        logistic_regression,
        X_test,
        y_test,
    )

    # ------------------------------------------------------------------
    # Decision Tree comparison
    # ------------------------------------------------------------------

    print("\nTraining Decision Tree comparison model...")

    from sklearn.tree import DecisionTreeClassifier

    decision_tree = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_split=2,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    decision_tree.fit(
        X_development,
        y_development,
    )

    decision_tree_metrics = evaluate_model(
        decision_tree,
        X_test,
        y_test,
    )

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
            ],
            "Accuracy": [
                logistic_metrics["accuracy"],
                decision_tree_metrics["accuracy"],
                random_forest_metrics["accuracy"],
            ],
            "Precision": [
                logistic_metrics["precision"],
                decision_tree_metrics["precision"],
                random_forest_metrics["precision"],
            ],
            "Recall": [
                logistic_metrics["recall"],
                decision_tree_metrics["recall"],
                random_forest_metrics["recall"],
            ],
            "F1": [
                logistic_metrics["f1"],
                decision_tree_metrics["f1"],
                random_forest_metrics["f1"],
            ],
            "ROC_AUC": [
                logistic_metrics["roc_auc"],
                decision_tree_metrics["roc_auc"],
                random_forest_metrics["roc_auc"],
            ],
        }
    )

    print("\nTest-set model comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nRandom Forest confusion matrix:")
    print(
        random_forest_metrics["confusion_matrix"]
    )

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------

    print("\nCalculating permutation feature importance...")

    permutation = permutation_importance(
        final_random_forest,
        X_validation,
        y_validation,
        scoring="f1",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    permutation_importance_df = pd.DataFrame(
        {
            "feature": X_validation.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values(
        by="importance_mean",
        ascending=False,
    )

    print("\nPermutation feature importance:")
    print(
        permutation_importance_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    mlflow.set_experiment(
        "PlaylistPulse-Random-Forest"
    )

    with mlflow.start_run(
        run_name="Random_Forest_Classification"
    ):

        # Hyperparameters
        mlflow.log_param(
            "n_estimators",
            best_n_estimators,
        )

        mlflow.log_param(
            "max_depth",
            best_max_depth,
        )

        mlflow.log_param(
            "min_samples_split",
            best_min_samples_split,
        )

        mlflow.log_param(
            "min_samples_leaf",
            best_min_samples_leaf,
        )

        mlflow.log_param(
            "max_features",
            best_max_features,
        )

        mlflow.log_param(
            "bootstrap",
            True,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        # Validation metrics
        mlflow.log_metric(
            "validation_accuracy",
            float(best_configuration["validation_accuracy"]),
        )

        mlflow.log_metric(
            "validation_precision",
            float(best_configuration["validation_precision"]),
        )

        mlflow.log_metric(
            "validation_recall",
            float(best_configuration["validation_recall"]),
        )

        mlflow.log_metric(
            "validation_f1",
            float(best_configuration["validation_f1"]),
        )

        mlflow.log_metric(
            "validation_roc_auc",
            float(best_configuration["validation_roc_auc"]),
        )

        # OOB metrics
        mlflow.log_metric(
            "oob_score",
            float(oob_score),
        )

        mlflow.log_metric(
            "oob_error",
            float(oob_error),
        )

        # Random Forest test metrics
        mlflow.log_metric(
            "random_forest_test_accuracy",
            random_forest_metrics["accuracy"],
        )

        mlflow.log_metric(
            "random_forest_test_precision",
            random_forest_metrics["precision"],
        )

        mlflow.log_metric(
            "random_forest_test_recall",
            random_forest_metrics["recall"],
        )

        mlflow.log_metric(
            "random_forest_test_f1",
            random_forest_metrics["f1"],
        )

        mlflow.log_metric(
            "random_forest_test_roc_auc",
            random_forest_metrics["roc_auc"],
        )

        # Logistic Regression comparison
        mlflow.log_metric(
            "logistic_test_f1",
            logistic_metrics["f1"],
        )

        mlflow.log_metric(
            "logistic_test_roc_auc",
            logistic_metrics["roc_auc"],
        )

        # Decision Tree comparison
        mlflow.log_metric(
            "decision_tree_test_f1",
            decision_tree_metrics["f1"],
        )

        mlflow.log_metric(
            "decision_tree_test_roc_auc",
            decision_tree_metrics["roc_auc"],
        )

        # Permutation importance
        for _, row in permutation_importance_df.iterrows():

            feature_name = row["feature"]

            safe_feature_name = (
                feature_name
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )

            mlflow.log_metric(
                f"permutation_importance_{safe_feature_name}",
                float(row["importance_mean"]),
            )

        mlflow.sklearn.log_model(
            final_random_forest,
            name="random_forest_model",
        )

        print("\nMLflow run logged successfully.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Random Forest pipeline completed.")
    print("=" * 70)

    print("\nFinal Random Forest metrics:")

    print(
        f"Accuracy : "
        f"{random_forest_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{random_forest_metrics['precision']:.6f}"
    )

    print(
        f"Recall   : "
        f"{random_forest_metrics['recall']:.6f}"
    )

    print(
        f"F1       : "
        f"{random_forest_metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{random_forest_metrics['roc_auc']:.6f}"
    )

    print(
        f"OOB Score: "
        f"{oob_score:.6f}"
    )

    print(
        f"OOB Error: "
        f"{oob_error:.6f}"
    )


if __name__ == "__main__":
    main()