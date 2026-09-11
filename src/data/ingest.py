import os

import pandas as pd
import numpy as np


# ============================================================
# PLAYLIST DATASET INGESTION
# ============================================================

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load and validate the primary playlist dataset."""

    # Check whether the dataset exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Critical Error: Targeted data footprint not discovered at {file_path}"
        )

    # Display extraction progress
    print(
        f"Executing secure data extraction from: {file_path}"
    )

    # Load CSV into memory
    df = pd.read_csv(file_path)

    # Required columns for the primary playlist dataset
    required_columns = [
        "track_id",
        "name",
        "artist",
        "spotify_preview_url",
        "spotify_id",
        "tags",
        "genre",
        "year",
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
        "time_signature"
    ]

    # Find missing columns
    missing_cols = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    # Stop processing if schema is incorrect
    if missing_cols:
        raise ValueError(
            f"Schema Validation Failure: "
            f"Missing essential feature targets: {missing_cols}"
        )

    # Display successful ingestion
    print(
        f"Data ingestion resolved successfully. "
        f"Dimensions captured: {df.shape[0]} samples, "
        f"{df.shape[1]} metrics."
    )

    return df


# ============================================================
# REGRESSION DATASET INGESTION
# ============================================================

def load_and_validate_regression_data(
    file_path: str
) -> pd.DataFrame:
    """Load and validate the stream-count regression dataset."""

    # Check whether the dataset exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Critical Error: Regression dataset not discovered at {file_path}"
        )

    # Display extraction progress
    print(
        f"Executing regression data extraction from: {file_path}"
    )

    # Load CSV into memory
    df = pd.read_csv(file_path)

    # Required columns for the regression dataset
    required_columns = [
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
        "artist",
        "genre",
        "tags",
        "stream_count",
        "log_stream_count"
    ]

    # Find missing columns
    missing_cols = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    # Stop processing if schema is incorrect
    if missing_cols:
        raise ValueError(
            f"Regression Schema Validation Failure: "
            f"Missing essential feature targets: {missing_cols}"
        )

    # --------------------------------------------------------
    # Check for missing values
    # --------------------------------------------------------

    missing_values = df[required_columns].isnull().sum()

    missing_data = missing_values[
        missing_values > 0
    ]

    if not missing_data.empty:
        raise ValueError(
            "Regression Data Quality Failure: "
            f"Missing values detected:\n{missing_data}"
        )

    # --------------------------------------------------------
    # Validate numerical columns
    # --------------------------------------------------------

    numerical_columns = [
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
        "stream_count",
        "log_stream_count"
    ]

    for column in numerical_columns:
        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise TypeError(
                f"Regression Schema Validation Failure: "
                f"Column '{column}' must be numeric."
            )

    # --------------------------------------------------------
    # Validate stream count
    # --------------------------------------------------------

    if (df["stream_count"] < 0).any():
        raise ValueError(
            "Regression Target Validation Failure: "
            "stream_count cannot contain negative values."
        )

    # --------------------------------------------------------
    # Validate log transformation
    #
    # Expected:
    # log_stream_count = log1p(stream_count)
    # --------------------------------------------------------

    expected_log_stream_count = np.log1p(
        df["stream_count"]
    )

    maximum_error = (
        expected_log_stream_count
        - df["log_stream_count"]
    ).abs().max()

    if maximum_error > 1e-6:
        raise ValueError(
            "Regression Target Validation Failure: "
            "log_stream_count does not match "
            "log1p(stream_count). "
            f"Maximum error: {maximum_error}"
        )

    # --------------------------------------------------------
    # Display successful ingestion
    # --------------------------------------------------------

    print(
        f"Regression data ingestion resolved successfully. "
        f"Dimensions captured: {df.shape[0]} samples, "
        f"{df.shape[1]} metrics."
    )

    print(
        f"Log target validation passed. "
        f"Maximum error: {maximum_error:.10f}"
    )

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n" + "#" * 60)
    print("PLAYLIST_PULSE — DATA INGESTION PIPELINE")
    print("#" * 60)

    # --------------------------------------------------------
    # Primary Playlist Dataset
    # --------------------------------------------------------

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_playlist_data.csv"
    )

    try:
        raw_data = load_and_validate_data(
            DATA_PATH
        )
    except Exception as e:
        print(
            f"Ingestion lifecycle termination: {str(e)}"
        )

    # --------------------------------------------------------
    # Regression Dataset
    # --------------------------------------------------------

    REGRESSION_DATA_PATH = os.path.join(
        "src",
        "data",
        "regression.csv"
    )

    try:
        regression_data = (
            load_and_validate_regression_data(
                REGRESSION_DATA_PATH
            )
        )
    except Exception as e:
        print(
            f"Regression ingestion lifecycle termination: "
            f"{str(e)}"
        )