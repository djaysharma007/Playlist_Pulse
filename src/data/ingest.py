import os
import pandas as pd


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    # Check whether the dataset exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Critical Error: Targeted data footprint not discovered at {file_path}"
        )

    # Display extraction progress
    print(f"Executing secure data extraction from: {file_path}")

    # Load CSV into memory
    df = pd.read_csv(file_path)

    # Required columns for Playlist_Pulse dataset
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
        col for col in required_columns
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


if __name__ == "__main__":
    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_playlist_data.csv"
    )

    try:
        raw_data = load_and_validate_data(DATA_PATH)

    except Exception as e:
        print(f"Ingestion lifecycle termination: {str(e)}")