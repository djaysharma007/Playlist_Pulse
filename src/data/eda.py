from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ingest import (
    load_and_validate_data,
    load_and_validate_regression_data,
)


PLAYLIST_DATA_PATH = PROJECT_ROOT / "src" / "data" / "raw_playlist_data.csv"
CLASSIFICATION_DATA_PATH = PROJECT_ROOT / "src" / "data" / "classification.csv"
REGRESSION_DATA_PATH = PROJECT_ROOT / "src" / "data" / "regression.csv"

FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def calculate_iqr_outliers(df, columns):
    """
    Calculate IQR-based outlier counts for the specified columns.

    Returns:
        pd.DataFrame: Feature names, lower bounds, upper bounds,
                      and outlier counts.
    """
    results = []

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        results.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(outlier_mask.sum()),
            }
        )

    return pd.DataFrame(results)


def save_figure(filename):
    """
    Save the current matplotlib figure to the reports directory.
    """
    output_path = FIGURES_PATH / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# -------------------------------------------------------------------
# Playlist dataset analysis
# -------------------------------------------------------------------

def analyze_playlist_dataset():
    """
    Perform exploratory analysis on the original playlist dataset.
    """
    print("\n" + "=" * 70)
    print("PLAYLIST DATASET — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    df = load_and_validate_data(str(PLAYLIST_DATA_PATH))

    print("\n" + "-" * 70)
    print("DATASET DIMENSIONS")
    print("-" * 70)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "-" * 70)
    print("FEATURE NAMES AND DATA TYPES")
    print("-" * 70)
    print(df.dtypes)

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print("\nCategorical Features:")
    print(categorical_columns)

    print("\nNumerical Features:")
    print(numerical_columns)

    print("\n" + "-" * 70)
    print("MISSING VALUES AND DUPLICATES")
    print("-" * 70)

    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        print("Missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records: {duplicate_count}")

    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    print(df[numerical_columns].describe())

    print("\n" + "-" * 70)
    print("GENRE DISTRIBUTION")
    print("-" * 70)

    if "genre" in df.columns:
        genre_counts = df["genre"].value_counts(dropna=True)
        genre_percentages = (
            df["genre"]
            .value_counts(normalize=True, dropna=True)
            .mul(100)
        )

        print("\nGenre Counts:")
        print(genre_counts)

        print("\nGenre Percentages:")
        print(genre_percentages)

    # Correlation heatmap
    if len(numerical_columns) > 1:
        plt.figure(figsize=(12, 9))

        correlation_matrix = df[numerical_columns].corr()

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
        )

        plt.title("Feature Correlation Matrix")
        save_figure("correlation_heatmap.png")

    # Danceability vs energy
    if {"danceability", "energy"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))

        if "genre" in df.columns:
            sns.scatterplot(
                data=df,
                x="danceability",
                y="energy",
                hue="genre",
                alpha=0.7,
            )
        else:
            sns.scatterplot(
                data=df,
                x="danceability",
                y="energy",
                alpha=0.7,
            )

        plt.title("Danceability vs Energy")
        save_figure("danceability_energy_scatter.png")

    # Feature distributions
    distribution_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    for column in distribution_columns:
        if column not in df.columns:
            continue

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True,
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        save_figure(f"distribution_{column}.png")

    # Genre distribution
    if "genre" in df.columns:
        plt.figure(figsize=(12, 6))

        sns.countplot(
            data=df,
            x="genre",
            order=df["genre"].value_counts().index,
        )

        plt.title("Genre Distribution")
        plt.xlabel("Genre")
        plt.ylabel("Number of Tracks")
        plt.xticks(rotation=45, ha="right")

        save_figure("genre_distribution.png")

    # IQR outlier analysis
    print("\n" + "-" * 70)
    print("OUTLIER ANALYSIS — IQR METHOD")
    print("-" * 70)

    outlier_report = calculate_iqr_outliers(
        df,
        numerical_columns,
    )

    for _, row in outlier_report.iterrows():
        print(
            f"{row['feature']}: "
            f"{row['outlier_count']} outliers"
        )

    # Outlier boxplot
    if numerical_columns:
        plt.figure(figsize=(14, 8))

        sns.boxplot(
            data=df[numerical_columns],
            orient="h",
        )

        plt.title("Numerical Feature Outlier Analysis")

        save_figure("outliers_boxplot.png")


# -------------------------------------------------------------------
# Classification dataset analysis
# -------------------------------------------------------------------

def analyze_classification_dataset():
    """
    Perform exploratory analysis on the classification dataset.
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION DATASET — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    if not CLASSIFICATION_DATA_PATH.exists():
        print(
            f"Classification dataset not found: "
            f"{CLASSIFICATION_DATA_PATH}"
        )
        return

    classification_df = pd.read_csv(
        CLASSIFICATION_DATA_PATH
    )

    print("\n" + "-" * 70)
    print("DATASET DIMENSIONS")
    print("-" * 70)
    print(f"Rows: {classification_df.shape[0]}")
    print(f"Columns: {classification_df.shape[1]}")

    print("\n" + "-" * 70)
    print("FEATURES AND DATA TYPES")
    print("-" * 70)
    print(classification_df.dtypes)

    print("\nFeature Names:")
    print(classification_df.columns.tolist())

    print("\n" + "-" * 70)
    print("DATA QUALITY")
    print("-" * 70)

    missing_values = classification_df.isnull().sum()

    if missing_values.sum() > 0:
        print("Missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")

    duplicate_count = classification_df.duplicated().sum()
    print(f"Duplicate records: {duplicate_count}")

    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    print(classification_df.describe())

    target_column = "y"

    print("\n" + "-" * 70)
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    if target_column not in classification_df.columns:
        print("Target column 'y' not found.")
        return

    class_counts = (
        classification_df[target_column]
        .value_counts()
        .sort_index()
    )

    class_percentages = (
        classification_df[target_column]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    print("Target Column:", target_column)

    print("\nClass Counts:")
    print(class_counts)

    print("\nClass Percentages:")
    print(class_percentages)

    imbalance_difference = (
        class_percentages.max()
        - class_percentages.min()
    )

    if imbalance_difference <= 5:
        print("\nClass distribution is approximately balanced.")
    else:
        print("\nClass distribution shows imbalance.")

    # Target distribution
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=classification_df,
        x=target_column,
    )

    plt.title("Classification Target Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")

    save_figure("classification_target_distribution.png")

    # Correlation analysis
    classification_numeric = classification_df.select_dtypes(
        include=[np.number]
    )

    if len(classification_numeric.columns) > 1:
        plt.figure(figsize=(8, 6))

        correlation_matrix = classification_numeric.corr()

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
        )

        plt.title("Classification Dataset Correlation Matrix")

        save_figure(
            "classification_correlation_heatmap.png"
        )

    # Outlier analysis
    print("\n" + "-" * 70)
    print("CLASSIFICATION FEATURE OUTLIERS")
    print("-" * 70)

    feature_columns = [
        column
        for column in classification_numeric.columns
        if column != target_column
    ]

    if feature_columns:
        outlier_report = calculate_iqr_outliers(
            classification_df,
            feature_columns,
        )

        for _, row in outlier_report.iterrows():
            print(
                f"{row['feature']}: "
                f"{row['outlier_count']} outliers"
            )
    else:
        print(
            "No numerical feature columns available "
            "for outlier analysis."
        )


# -------------------------------------------------------------------
# Stream count regression dataset analysis
# -------------------------------------------------------------------

def analyze_regression_dataset():
    """
    Perform exploratory analysis on the stream-count regression dataset.
    """
    print("\n" + "=" * 70)
    print("STREAM COUNT REGRESSION DATASET — "
          "EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    df = load_and_validate_regression_data(
        str(REGRESSION_DATA_PATH)
    )

    print("\n" + "-" * 70)
    print("DATASET DIMENSIONS")
    print("-" * 70)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "-" * 70)
    print("FEATURES AND DATA TYPES")
    print("-" * 70)
    print(df.dtypes)

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print("\nCategorical Features:")
    print(categorical_columns)

    print("\nNumerical Features:")
    print(numerical_columns)

    print("\n" + "-" * 70)
    print("DATA QUALITY")
    print("-" * 70)

    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        print("Missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records: {duplicate_count}")

    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    print(df[numerical_columns].describe())

    # Target analysis
    print("\n" + "-" * 70)
    print("STREAM COUNT TARGET ANALYSIS")
    print("-" * 70)

    print("\nRaw Stream Count:")
    print(df["stream_count"].describe())

    print("\nLog Stream Count:")
    print(df["log_stream_count"].describe())

    expected_log_stream_count = np.log1p(
        df["stream_count"]
    )

    transformation_error = (
        expected_log_stream_count
        - df["log_stream_count"]
    ).abs()

    max_error = transformation_error.max()

    print("\nLog Transformation Validation")
    print(f"Maximum error: {max_error:.10f}")

    if max_error <= 1e-6:
        print("Transformation validated: log1p(stream_count).")
    else:
        print(
            "Warning: log_stream_count does not match "
            "log1p(stream_count)."
        )

    # Raw stream count distribution
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="stream_count",
        bins=50,
        kde=True,
    )

    plt.title("Distribution of Raw Stream Count")
    plt.xlabel("Stream Count")
    plt.ylabel("Frequency")

    save_figure("stream_count_distribution.png")

    # Log stream count distribution
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="log_stream_count",
        bins=50,
        kde=True,
    )

    plt.title("Distribution of Log Stream Count")
    plt.xlabel("log1p(Stream Count)")
    plt.ylabel("Frequency")

    save_figure("log_stream_count_distribution.png")

    # Raw vs log
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="stream_count",
        y="log_stream_count",
        alpha=0.6,
    )

    plt.title("Raw Stream Count vs Log Stream Count")
    plt.xlabel("Stream Count")
    plt.ylabel("log1p(Stream Count)")

    save_figure("stream_count_vs_log_stream_count.png")

    # Regression features
    regression_features = [
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

    available_features = [
        column
        for column in regression_features
        if column in df.columns
    ]

    correlation_data = df[
        available_features + ["log_stream_count"]
    ]

    correlation_matrix = correlation_data.corr()

    target_correlations = (
        correlation_matrix["log_stream_count"]
        .drop("log_stream_count")
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False,
        )
    )

    print("\n" + "-" * 70)
    print("CORRELATION WITH LOG STREAM COUNT")
    print("-" * 70)
    print(target_correlations)

    # Regression heatmap
    plt.figure(figsize=(13, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )

    plt.title(
        "Regression Features and Log Stream Count "
        "Correlation Matrix"
    )

    save_figure("regression_correlation_heatmap.png")

    # Top correlated features
    print("\n" + "-" * 70)
    print("FEATURE RELATIONSHIPS WITH LOG STREAM COUNT")
    print("-" * 70)

    top_features = (
        target_correlations.abs()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    print("Top correlated features:")
    print(list(top_features))

    for feature in top_features:
        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=df,
            x=feature,
            y="log_stream_count",
            alpha=0.6,
        )

        plt.title(
            f"{feature} vs Log Stream Count"
        )
        plt.xlabel(feature)
        plt.ylabel("log1p(Stream Count)")

        save_figure(
            f"{feature}_vs_log_stream_count.png"
        )

    # Outlier analysis
    print("\n" + "-" * 70)
    print("REGRESSION OUTLIER ANALYSIS — IQR METHOD")
    print("-" * 70)

    outlier_columns = (
        available_features
        + [
            "stream_count",
            "log_stream_count",
        ]
    )

    outlier_report = calculate_iqr_outliers(
        df,
        outlier_columns,
    )

    for _, row in outlier_report.iterrows():
        print(
            f"{row['feature']}: "
            f"{row['outlier_count']} outliers"
        )

    outlier_report_path = (
        FIGURES_PATH / "regression_outlier_report.csv"
    )

    outlier_report.to_csv(
        outlier_report_path,
        index=False,
    )

    print(f"Saved: {outlier_report_path}")

    # Regression boxplot
    plt.figure(figsize=(14, 9))

    sns.boxplot(
        data=df[
            available_features
            + ["log_stream_count"]
        ],
        orient="h",
    )

    plt.title("Regression Feature Outlier Analysis")

    save_figure("regression_outliers_boxplot.png")


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------

def main():
    """
    Execute all available exploratory analyses.
    """
    analyze_playlist_dataset()
    analyze_classification_dataset()
    analyze_regression_dataset()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEDA pipeline failed: {exc}")
        raise