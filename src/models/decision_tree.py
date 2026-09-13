"""
PlaylistPulse - Decision Tree Classification

Purpose:
    Train and evaluate a Decision Tree classifier for the binary
    classification target in classification.csv.

Features:
    userId
    itemId
    timestamp

Target:
    y

Model selection:
    Validation-set tuning of tree complexity using:
        - max_depth
        - min_samples_split
        - min_samples_leaf

Evaluation:
    Accuracy
    Precision
    Recall
    F1
    ROC-AUC

Analysis:
    - Feature importance
    - Actual decision-tree split nodes
    - Top decision features

MLflow:
    Model parameters, validation results, test metrics, feature
    importance, and split analysis are logged.

Important:
    The semantic meaning of y is not assumed. It is treated only
    as a binary classification target.
"""

from pathlib import Path

import mlflow
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


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
VALIDATION_SIZE = 0.25

CATEGORICAL_FEATURES = [
    "userId",
    "itemId",
]

NUMERICAL_FEATURES = [
    "timestamp",
]

TARGET_COLUMN = "y"

MAX_DEPTH_VALUES = [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    None,
]

MIN_SAMPLES_SPLIT_VALUES = [
    2,
    10,
    20,
]

MIN_SAMPLES_LEAF_VALUES = [
    1,
    5,
    10,
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
            f"Target y must contain only 0 and 1. "
            f"Found: {sorted(target_values)}"
        )

    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Target y must contain both classes."
        )

    print(f"Samples loaded: {len(data)}")

    print("Class distribution:")
    print(
        data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor():
    """Create the preprocessing pipeline."""

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

def build_tree_model(
    max_depth,
    min_samples_split,
    min_samples_leaf,
):
    """Create the preprocessing and Decision Tree pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_logistic_baseline():
    """Create the Logistic Regression baseline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X,
    y,
):
    """Evaluate a classification model."""

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(
            y,
            predictions,
        ),
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


# ---------------------------------------------------------------------------
# Validation tuning
# ---------------------------------------------------------------------------

def tune_decision_tree(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Tune Decision Tree complexity using the validation set.

    F1 is used as the primary selection metric because it balances
    precision and recall for binary classification.
    """

    results = []

    total_candidates = (
        len(MAX_DEPTH_VALUES)
        * len(MIN_SAMPLES_SPLIT_VALUES)
        * len(MIN_SAMPLES_LEAF_VALUES)
    )

    print()
    print(
        f"Evaluating {total_candidates} "
        "Decision Tree configurations..."
    )

    for max_depth in MAX_DEPTH_VALUES:

        for min_samples_split in MIN_SAMPLES_SPLIT_VALUES:

            for min_samples_leaf in MIN_SAMPLES_LEAF_VALUES:

                model = build_tree_model(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
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

                results.append(
                    {
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        **metrics,
                    }
                )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "F1",
            "ROC_AUC",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best = results_df.iloc[0]

    print()
    print("Best validation configuration:")

    print(
        f"max_depth: "
        f"{best['max_depth']}"
    )

    print(
        f"min_samples_split: "
        f"{best['min_samples_split']}"
    )

    print(
        f"min_samples_leaf: "
        f"{best['min_samples_leaf']}"
    )

    print(
        f"Validation F1: "
        f"{best['F1']:.6f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{best['ROC_AUC']:.6f}"
    )

    return results_df, best


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def extract_feature_importance(model):
    """Extract feature importance from the fitted Decision Tree."""

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df[
            importance_df["importance"] > 0
        ]
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


# ---------------------------------------------------------------------------
# Actual tree split analysis
# ---------------------------------------------------------------------------

def extract_tree_splits(model):
    """
    Extract actual internal decision-tree split nodes.

    Each row represents an internal node where the tree made a
    decision based on a feature and threshold.
    """

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    tree = classifier.tree_

    split_rows = []

    for node_id in range(tree.node_count):

        feature_index = tree.feature[
            node_id
        ]

        if feature_index < 0:
            continue

        threshold = tree.threshold[
            node_id
        ]

        impurity = tree.impurity[
            node_id
        ]

        sample_count = tree.n_node_samples[
            node_id
        ]

        left_child = tree.children_left[
            node_id
        ]

        right_child = tree.children_right[
            node_id
        ]

        left_samples = tree.n_node_samples[
            left_child
        ]

        right_samples = tree.n_node_samples[
            right_child
        ]

        left_impurity = tree.impurity[
            left_child
        ]

        right_impurity = tree.impurity[
            right_child
        ]

        weighted_child_impurity = (
            (
                left_samples
                * left_impurity
            )
            + (
                right_samples
                * right_impurity
            )
        ) / sample_count

        impurity_decrease = (
            impurity
            - weighted_child_impurity
        )

        split_rows.append(
            {
                "node": node_id,
                "feature": feature_names[
                    feature_index
                ],
                "threshold": threshold,
                "samples": sample_count,
                "impurity": impurity,
                "impurity_decrease": impurity_decrease,
            }
        )

    split_df = pd.DataFrame(
        split_rows
    )

    if split_df.empty:
        return split_df

    return split_df.sort_values(
        by="impurity_decrease",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():

    print("=" * 60)
    print("PlaylistPulse - Decision Tree Classification")
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
    # Test split
    # -----------------------------------------------------------------------

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -----------------------------------------------------------------------
    # Validation split
    # -----------------------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_validation)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # -----------------------------------------------------------------------
    # Logistic Regression baseline
    # -----------------------------------------------------------------------

    print()
    print(
        "Training Logistic Regression baseline..."
    )

    baseline_model = build_logistic_baseline()

    baseline_model.fit(
        X_train,
        y_train,
    )

    baseline_validation = evaluate_model(
        baseline_model,
        X_validation,
        y_validation,
    )

    # -----------------------------------------------------------------------
    # Decision Tree tuning
    # -----------------------------------------------------------------------

    validation_results, best_config = (
        tune_decision_tree(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    # -----------------------------------------------------------------------
    # Safely extract selected parameters
    # -----------------------------------------------------------------------

    selected_max_depth = best_config[
        "max_depth"
    ]

    if pd.isna(selected_max_depth):
        selected_max_depth = None
    else:
        selected_max_depth = int(
            selected_max_depth
        )

    selected_min_samples_split = int(
        best_config[
            "min_samples_split"
        ]
    )

    selected_min_samples_leaf = int(
        best_config[
            "min_samples_leaf"
        ]
    )

    # -----------------------------------------------------------------------
    # Final Decision Tree
    # -----------------------------------------------------------------------

    final_tree = build_tree_model(
        max_depth=selected_max_depth,
        min_samples_split=selected_min_samples_split,
        min_samples_leaf=selected_min_samples_leaf,
    )

    # Fit only after hyperparameter selection.
    final_tree.fit(
        X_development,
        y_development,
    )

    # -----------------------------------------------------------------------
    # Final baseline
    # -----------------------------------------------------------------------

    baseline_model.fit(
        X_development,
        y_development,
    )

    # -----------------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------------

    print()
    print(
        "Evaluating final models "
        "on untouched test data..."
    )

    tree_test = evaluate_model(
        final_tree,
        X_test,
        y_test,
    )

    baseline_test = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression Baseline",
                **baseline_test,
            },
            {
                "Model": "Decision Tree",
                **tree_test,
            },
        ]
    )

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
    # Feature importance
    # -----------------------------------------------------------------------

    importance_df = extract_feature_importance(
        final_tree
    )

    print()
    print("=" * 60)
    print("Top Decision Tree Features")
    print("=" * 60)

    if importance_df.empty:

        print(
            "No non-zero feature importances found."
        )

    else:

        print(
            importance_df.head(15).to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

    # -----------------------------------------------------------------------
    # Actual split analysis
    # -----------------------------------------------------------------------

    split_df = extract_tree_splits(
        final_tree
    )

    print()
    print("=" * 60)
    print("Top Decision Tree Splits")
    print("=" * 60)

    if split_df.empty:

        print(
            "The fitted tree contains no internal split nodes."
        )

    else:

        print(
            split_df.head(15).to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

    # -----------------------------------------------------------------------
    # Tree statistics
    # -----------------------------------------------------------------------

    classifier = final_tree.named_steps[
        "classifier"
    ]

    print()
    print("=" * 60)
    print("Decision Tree Structure")
    print("=" * 60)

    print(
        f"Tree depth: "
        f"{classifier.get_depth()}"
    )

    print(
        f"Number of nodes: "
        f"{classifier.tree_.node_count}"
    )

    print(
        f"Number of leaves: "
        f"{classifier.get_n_leaves()}"
    )

    # -----------------------------------------------------------------------
    # MLflow tracking
    # -----------------------------------------------------------------------

    mlflow.set_experiment(
        "PlaylistPulse-Decision-Tree"
    )

    with mlflow.start_run(
        run_name="Decision_Tree_Classification"
    ):

        mlflow.log_param(
            "target",
            TARGET_COLUMN,
        )

        mlflow.log_param(
            "criterion",
            "gini",
        )

        mlflow.log_param(
            "max_depth",
            selected_max_depth,
        )

        mlflow.log_param(
            "min_samples_split",
            selected_min_samples_split,
        )

        mlflow.log_param(
            "min_samples_leaf",
            selected_min_samples_leaf,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_metric(
            "validation_f1",
            float(best_config["F1"]),
        )

        mlflow.log_metric(
            "validation_roc_auc",
            float(best_config["ROC_AUC"]),
        )

        for model_name, metrics in [
            (
                "baseline",
                baseline_test,
            ),
            (
                "decision_tree",
                tree_test,
            ),
        ]:

            for metric, value in metrics.items():

                mlflow.log_metric(
                    f"{model_name}_test_{metric.lower()}",
                    float(value),
                )

        mlflow.log_metric(
            "tree_depth",
            float(classifier.get_depth()),
        )

        mlflow.log_metric(
            "tree_nodes",
            float(classifier.tree_.node_count),
        )

        mlflow.log_metric(
            "tree_leaves",
            float(classifier.get_n_leaves()),
        )

        mlflow.log_text(
            validation_results.to_csv(
                index=False
            ),
            "decision_tree_validation_results.csv",
        )

        mlflow.log_text(
            comparison.to_csv(
                index=False
            ),
            "model_comparison.csv",
        )

        mlflow.log_text(
            importance_df.to_csv(
                index=False
            ),
            "feature_importance.csv",
        )

        mlflow.log_text(
            split_df.to_csv(
                index=False
            ),
            "tree_splits.csv",
        )

        print()
        print(
            "MLflow run logged successfully."
        )

    print()
    print(
        "Decision Tree classification pipeline completed."
    )


if __name__ == "__main__":
    main()