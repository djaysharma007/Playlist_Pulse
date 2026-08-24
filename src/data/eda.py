import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Allow importing ingest.py when this file is executed directly
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ingest import load_and_validate_data


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PLAYLIST_DATA_PATH = os.path.join(
    "src", "data", "raw_playlist_data.csv"
)

CLASSIFICATION_DATA_PATH = os.path.join(
    "src", "data", "classification.csv"
)

FIGURES_PATH = os.path.join(
    "reports", "figures"
)

os.makedirs(FIGURES_PATH, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# PART A — PLAYLIST DATASET EDA
# ============================================================

def analyze_playlist_dataset():

    print("\n" + "=" * 60)
    print("PLAYLIST DATASET — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load Data using the Ingestion Pipeline
    # --------------------------------------------------------

    df = load_and_validate_data(PLAYLIST_DATA_PATH)

    print("\n" + "=" * 60)
    print("--- 1. DATASET DIMENSIONS ---")

    print(f"Total Rows (Samples): {df.shape[0]}")
    print(f"Total Columns (Metrics): {df.shape[1]}")

    # --------------------------------------------------------
    # 2. Feature Names & Data Types
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 2. FEATURE NAMES & DATA TYPES ---")

    print(df.dtypes)

    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(
        include=["object", "str", "string", "category"]
    ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print("\nCategorical Features:")
    print(categorical_columns)

    print("\nNumerical Features:")
    print(numerical_columns)

    # --------------------------------------------------------
    # 3. Missing Values & Duplicates
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 3. MISSING VALUES & DUPLICATES ---")

    missing_vals = df.isnull().sum()

    if missing_vals.sum() > 0:
        print("Missing Values per Column:")
        print(missing_vals[missing_vals > 0])
    else:
        print("No missing values found.")

    duplicates = df.duplicated().sum()

    print(f"Duplicate Records Count: {duplicates}")

    # --------------------------------------------------------
    # 4. Summary Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 4. SUMMARY STATISTICS (NUMERICAL FEATURES) ---")

    numerical_df = df.select_dtypes(include=[np.number])

    print(numerical_df.describe())

    # --------------------------------------------------------
    # 5. Categorical Distribution Analysis
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 5. CATEGORICAL DISTRIBUTION ANALYSIS ---")

    if "genre" in df.columns:

        genre_counts = df["genre"].value_counts(
            dropna=True
        )

        genre_percentages = df["genre"].value_counts(
            normalize=True,
            dropna=True
        ) * 100

        print("\nGenre Counts:")
        print(genre_counts)

        print("\nGenre Percentages:")
        print(genre_percentages)

    else:

        print("Genre column not found.")

    # --------------------------------------------------------
    # 6. Visualizations
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 6. GENERATING VISUALIZATIONS ---")

    # --------------------------------------------------------
    # A. Correlation Matrix
    # --------------------------------------------------------

    if len(numerical_df.columns) > 1:

        plt.figure(figsize=(12, 9))

        corr_matrix = numerical_df.corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
        )

        plt.title("Feature Correlation Matrix Heatmap")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_PATH,
                "correlation_heatmap.png"
            )
        )

        plt.close()

        print(
            "-> Saved correlation heatmap"
        )

    # --------------------------------------------------------
    # B. Danceability vs Energy Scatter Plot
    # --------------------------------------------------------

    if (
        "danceability" in df.columns
        and "energy" in df.columns
    ):

        plt.figure(figsize=(8, 6))

        if "genre" in df.columns:

            sns.scatterplot(
                data=df,
                x="danceability",
                y="energy",
                hue="genre",
                alpha=0.7
            )

        else:

            sns.scatterplot(
                data=df,
                x="danceability",
                y="energy",
                alpha=0.7
            )

        plt.title(
            "Danceability vs Energy"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_PATH,
                "danceability_energy_scatter.png"
            )
        )

        plt.close()

        print(
            "-> Saved danceability-energy scatter plot"
        )

    # --------------------------------------------------------
    # C. Feature Distribution Plots
    # --------------------------------------------------------

    distribution_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    for column in distribution_columns:

        if column in df.columns:

            plt.figure(figsize=(8, 5))

            sns.histplot(
                data=df,
                x=column,
                kde=True
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.tight_layout()

            filename = (
                f"distribution_{column}.png"
            )

            plt.savefig(
                os.path.join(
                    FIGURES_PATH,
                    filename
                )
            )

            plt.close()

            print(
                f"-> Saved distribution plot for {column}"
            )

    # --------------------------------------------------------
    # D. Genre Distribution
    # --------------------------------------------------------

    if "genre" in df.columns:

        plt.figure(figsize=(12, 6))

        sns.countplot(
            data=df,
            x="genre",
            order=df["genre"].value_counts().index
        )

        plt.title(
            "Genre Distribution"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_PATH,
                "genre_distribution.png"
            )
        )

        plt.close()

        print(
            "-> Saved genre distribution plot"
        )

    # --------------------------------------------------------
    # 7. Outlier Counts using IQR Method
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 7. OUTLIER COUNTS USING IQR METHOD ---")

    for column in numerical_df.columns:

        Q1 = numerical_df[column].quantile(0.25)
        Q3 = numerical_df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = (
            (numerical_df[column] < lower_bound)
            | (numerical_df[column] > upper_bound)
        ).sum()

        print(
            f"{column}: {outlier_count} outliers"
        )

    # --------------------------------------------------------
    # 8. Outlier Boxplot
    # --------------------------------------------------------

    if len(numerical_df.columns) > 0:

        plt.figure(figsize=(14, 8))

        sns.boxplot(
            data=numerical_df,
            orient="h"
        )

        plt.title(
            "Outlier Identification via Boxplots"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_PATH,
                "outliers_boxplot.png"
            )
        )

        plt.close()

        print(
            "-> Saved outlier boxplot"
        )

    print(
        "\nPlaylist dataset EDA completed successfully."
    )


# ============================================================
# PART B — CLASSIFICATION DATASET EDA
# ============================================================

def analyze_classification_dataset():

    print("\n" + "=" * 60)
    print("CLASSIFICATION DATASET — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load Classification Dataset
    # --------------------------------------------------------

    if not os.path.exists(
        CLASSIFICATION_DATA_PATH
    ):

        print(
            f"Classification dataset not found at "
            f"{CLASSIFICATION_DATA_PATH}"
        )

        return

    classification_df = pd.read_csv(
        CLASSIFICATION_DATA_PATH
    )

    print("\n" + "=" * 60)
    print("--- 1. CLASSIFICATION DATASET DIMENSIONS ---")

    print(
        f"Total Rows (Samples): "
        f"{classification_df.shape[0]}"
    )

    print(
        f"Total Columns (Metrics): "
        f"{classification_df.shape[1]}"
    )

    # --------------------------------------------------------
    # 2. Features & Data Types
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 2. FEATURES & DATA TYPES ---")

    print(
        classification_df.dtypes
    )

    print("\nFeature Names:")

    print(
        classification_df.columns.tolist()
    )

    # --------------------------------------------------------
    # 3. Missing Values & Duplicates
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 3. DATA QUALITY CHECK ---")

    missing_values = (
        classification_df.isnull().sum()
    )

    if missing_values.sum() > 0:

        print(
            "Missing Values per Column:"
        )

        print(
            missing_values[
                missing_values > 0
            ]
        )

    else:

        print(
            "No missing values found."
        )

    duplicates = (
        classification_df.duplicated().sum()
    )

    print(
        f"Duplicate Records Count: "
        f"{duplicates}"
    )

    # --------------------------------------------------------
    # 4. Statistical Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 4. SUMMARY STATISTICS ---")

    print(
        classification_df.describe()
    )

    # --------------------------------------------------------
    # 5. Classification Target Analysis
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- 5. CLASS IMBALANCE ANALYSIS ---")

    target_column = "y"

    if target_column in classification_df.columns:

        class_counts = (
            classification_df[
                target_column
            ].value_counts().sort_index()
        )

        class_percentages = (
            classification_df[
                target_column
            ].value_counts(
                normalize=True
            ).sort_index() * 100
        )

        print(
            f"Target Column: {target_column}"
        )

        print(
            "\nClass Counts:"
        )

        print(
            class_counts
        )

        print(
            "\nClass Percentages:"
        )

        print(
            class_percentages
        )

        # Determine whether the classes are balanced
        max_percentage = (
            class_percentages.max()
        )

        min_percentage = (
            class_percentages.min()
        )

        imbalance_difference = (
            max_percentage - min_percentage
        )

        print(
            "\nClass Distribution Assessment:"
        )

        if imbalance_difference <= 5:

            print(
                "The classification dataset "
                "is balanced."
            )

        else:

            print(
                "The classification dataset "
                "shows class imbalance."
            )

    else:

        print(
            "Target column 'y' not found."
        )

    # --------------------------------------------------------
    # 6. Target Class Visualization
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "--- 6. CLASSIFICATION VISUALIZATIONS ---"
    )

    if target_column in classification_df.columns:

        plt.figure(figsize=(7, 5))

        sns.countplot(
            data=classification_df,
            x=target_column
        )

        plt.title(
            "Classification Target Distribution"
        )

        plt.xlabel(
            "Class"
        )

        plt.ylabel(
            "Number of Samples"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_PATH,
                "classification_target_distribution.png"
            )
        )

        plt.close()

        print(
            "-> Saved classification target distribution"
        )

    # --------------------------------------------------------
    # 7. Classification Dataset Correlation
    # --------------------------------------------------------

    classification_numeric = (
        classification_df.select_dtypes(
            include=[np.number]
        )
    )

    if len(classification_numeric.columns) > 1:

        plt.figure(figsize=(8, 6))

        classification_corr = (
            classification_numeric.corr()
        )

        sns.heatmap(
            classification_corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
        )

        plt.title(
            "Classification Dataset Correlation Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_PATH,
                "classification_correlation_heatmap.png"
            )
        )

        plt.close()

        print(
            "-> Saved classification correlation heatmap"
        )

    # --------------------------------------------------------
    # 8. Classification Dataset Outlier Analysis
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "--- 8. CLASSIFICATION DATASET OUTLIERS ---"
    )

    # Exclude target column from feature outlier analysis
    feature_columns = [
        column
        for column in classification_numeric.columns
        if column != target_column
    ]

    classification_features = (
        classification_df[
            feature_columns
        ]
    )

    if len(feature_columns) > 0:

        for column in feature_columns:

            Q1 = (
                classification_features[
                    column
                ].quantile(0.25)
            )

            Q3 = (
                classification_features[
                    column
                ].quantile(0.75)
            )

            IQR = Q3 - Q1

            lower_bound = (
                Q1 - 1.5 * IQR
            )

            upper_bound = (
                Q3 + 1.5 * IQR
            )

            outlier_count = (
                (
                    classification_features[
                        column
                    ] < lower_bound
                )
                |
                (
                    classification_features[
                        column
                    ] > upper_bound
                )
            ).sum()

            print(
                f"{column}: "
                f"{outlier_count} outliers"
            )

    else:

        print(
            "No numerical feature columns "
            "available for outlier analysis."
        )

    print(
        "\nClassification dataset EDA "
        "completed successfully."
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "#" * 60)
    print("PLAYLIST_PULSE — LAB 3 EDA PIPELINE")
    print("#" * 60)

    try:

        # Analyze original playlist dataset
        analyze_playlist_dataset()

        # Analyze classification dataset
        analyze_classification_dataset()

        print("\n" + "#" * 60)
        print("EDA EXECUTION COMPLETE")
        print("#" * 60)

        print(
            "\nAll visualizations are stored in:"
        )

        print(
            "reports/figures/"
        )

    except Exception as e:

        print(
            "\nEDA execution failed:"
        )

        print(
            str(e)
        )