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
# CLASSIFICATION DATASET INGESTION
# ============================================================

def load_and_validate_classification_data(
    file_path: str
) -> pd.DataFrame:
    """Load and validate the PlaylistPulse skip classification dataset."""

    # Check whether the dataset exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Critical Error: Classification dataset not discovered at {file_path}"
        )

    # Display extraction progress
    print(
        f"Executing classification data extraction from: {file_path}"
    )

    # Load CSV into memory
    df = pd.read_csv(file_path)

    # Required columns for the classification dataset
    required_columns = [
        "userId",
        "itemId",
        "timestamp",

        # Audio features
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

        # Track metadata
        "artist",
        "genre",
        "tags",

        # Historical user / track information
        "user_play_count",
        "user_skip_rate",
        "track_play_count",
        "track_skip_rate",

        # Session information
        "session_position",

        # Time-based features
        "hour",
        "day_of_week",
        "month",
        "is_weekend",

        # Classification target
        "skip_within_30s"
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
            f"Classification Schema Validation Failure: "
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
            "Classification Data Quality Failure: "
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
        "user_play_count",
        "user_skip_rate",
        "track_play_count",
        "track_skip_rate",
        "session_position",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "skip_within_30s"
    ]

    for column in numerical_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise TypeError(
                f"Classification Schema Validation Failure: "
                f"Column '{column}' must be numeric."
            )

    # --------------------------------------------------------
    # Validate timestamp
    # --------------------------------------------------------

    parsed_timestamp = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if parsed_timestamp.isnull().any():
        raise ValueError(
            "Classification Data Quality Failure: "
            "Invalid timestamp values detected."
        )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    valid_target_values = {
        0,
        1
    }

    actual_target_values = set(
        df["skip_within_30s"].unique()
    )

    if not actual_target_values.issubset(
        valid_target_values
    ):
        raise ValueError(
            "Classification Target Validation Failure: "
            "skip_within_30s must contain only 0 and 1."
        )

    # --------------------------------------------------------
    # Validate post-playback columns are absent
    # --------------------------------------------------------
    # These columns contain information generated during or
    # after playback and must not be used for prediction.

    leakage_columns = [
        "skipped",
        "skip_time_ms",
        "duration_played_ms"
    ]

    present_leakage_columns = [
        column
        for column in leakage_columns
        if column in df.columns
    ]

    if present_leakage_columns:
        raise ValueError(
            "Classification Leakage Validation Failure: "
            "Post-playback columns detected: "
            f"{present_leakage_columns}"
        )

    # --------------------------------------------------------
    # Validate duplicate records
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(
            f"Warning: {duplicate_count} duplicate records detected."
        )

    # --------------------------------------------------------
    # Display target distribution
    # --------------------------------------------------------

    target_distribution = (
        df["skip_within_30s"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nClassification target distribution:"
    )

    print(
        target_distribution
    )

    # Display successful ingestion
    print(
        f"\nClassification data ingestion resolved successfully. "
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
    # Classification Dataset
    # --------------------------------------------------------

    CLASSIFICATION_DATA_PATH = os.path.join(
        "src",
        "data",
        "classification.csv"
    )

    try:

        classification_data = (
            load_and_validate_classification_data(
                CLASSIFICATION_DATA_PATH
            )
        )

    except Exception as e:

        print(
            f"Classification ingestion lifecycle termination: "
            f"{str(e)}"
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