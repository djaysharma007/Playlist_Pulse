import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# Import ingestion function
# ------------------------------------------------------------

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

sys.path.append(DATA_DIR)

from ingest import load_and_validate_data


def train_linear_regression_ls():

    print("\n" + "#" * 60)
    print("PLAYLIST_PULSE — LINEAR REGRESSION")
    print("STANDARD LEAST SQUARES METHOD")
    print("#" * 60)

    # ------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------

    DATA_PATH = os.path.join(
        DATA_DIR,
        "raw_playlist_data.csv"
    )

    print(f"\nExecuting secure data extraction from: {DATA_PATH}")

    df = load_and_validate_data(DATA_PATH)

    # ------------------------------------------------------------
    # 2. Select Features and Target
    # ------------------------------------------------------------
    #
    # Input dimensions L = 2
    #   Feature 1: danceability
    #   Feature 2: energy
    #
    # Output dimension M = 1
    #   Target: valence
    #
    # ------------------------------------------------------------

    feature_cols = [
        "danceability",
        "energy"
    ]

    target_col = "valence"

    required_columns = feature_cols + [target_col]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from dataset: "
            f"{missing_columns}"
        )

    # Remove rows containing missing values
    df_clean = df.dropna(
        subset=required_columns
    ).copy()

    X_raw = df_clean[feature_cols].values

    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print("\n" + "=" * 60)
    print("--- DATASET CONFIGURATION ---")
    print(f"Loaded {N} data points")
    print(f"Input dimension L: {X_raw.shape[1]}")
    print(f"Output dimension M: {y.shape[1]}")

    print("\nInput Features:")
    for feature in feature_cols:
        print(f"- {feature}")

    print(f"\nTarget Feature:")
    print(f"- {target_col}")

    # ------------------------------------------------------------
    # 3. Design Matrix
    # ------------------------------------------------------------
    #
    # X_design = [1, x1, x2]
    #
    # The first column of ones represents the intercept/bias.
    #
    # ------------------------------------------------------------

    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )

    print("\n" + "=" * 60)
    print("--- DESIGN MATRIX ---")
    print(f"Design matrix shape: {X_design.shape}")
    print("Design matrix contains:")
    print("[1, danceability, energy]")

    # ------------------------------------------------------------
    # 4. Standard Least Squares Method
    # ------------------------------------------------------------
    #
    # Normal Equation:
    #
    # w = (X^T X)^-1 X^T y
    #
    # If X^T X is singular, use the pseudo-inverse.
    #
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- STANDARD LEAST SQUARES CALCULATION ---")

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    XT_y = np.dot(
        X_design.T,
        y
    )

    try:

        XT_X_inv = np.linalg.inv(XT_X)

        print("Matrix inversion successful.")

    except np.linalg.LinAlgError:

        print(
            "Matrix is singular. "
            "Using pseudo-inverse instead."
        )

        XT_X_inv = np.linalg.pinv(XT_X)

    # Calculate optimal weights
    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # ------------------------------------------------------------
    # 5. Display Model Parameters
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- OPTIMAL MODEL PARAMETERS ---")

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.6f}"
    )

    print(
        f"Coefficient for {feature_cols[0]} (w1): "
        f"{w_optimal[1, 0]:.6f}"
    )

    print(
        f"Coefficient for {feature_cols[1]} (w2): "
        f"{w_optimal[2, 0]:.6f}"
    )

    # ------------------------------------------------------------
    # 6. Generate Predictions
    # ------------------------------------------------------------

    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # ------------------------------------------------------------
    # 7. Calculate Error Function
    # ------------------------------------------------------------
    #
    # E(w) = 0.5 * Sum((Xw - y)^2)
    #
    # ------------------------------------------------------------

    residuals = y_pred - y

    E_w = 0.5 * np.sum(
        residuals ** 2
    )

    SSE = np.sum(
        residuals ** 2
    )

    MSE = np.mean(
        residuals ** 2
    )

    RMSE = np.sqrt(MSE)

    print("\n" + "=" * 60)
    print("--- MODEL ERROR ---")

    print(
        f"Minimized Error E(w): "
        f"{E_w:.6f}"
    )

    print(
        f"Sum of Squared Errors (SSE): "
        f"{SSE:.6f}"
    )

    print(
        f"Mean Squared Error (MSE): "
        f"{MSE:.6f}"
    )

    print(
        f"Root Mean Squared Error (RMSE): "
        f"{RMSE:.6f}"
    )

    # ------------------------------------------------------------
    # 8. Calculate R² Score
    # ------------------------------------------------------------

    ss_total = np.sum(
        (y - np.mean(y)) ** 2
    )

    ss_residual = np.sum(
        (y - y_pred) ** 2
    )

    r2_score = 1 - (
        ss_residual / ss_total
    )

    print(
        f"R² Score: "
        f"{r2_score:.6f}"
    )

    # ------------------------------------------------------------
    # 9. Create Output Directory
    # ------------------------------------------------------------

    REPORT_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "reports",
            "figures"
        )
    )

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 10. Create 3D Regression Plane
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- GENERATING 3D REGRESSION VISUALIZATION ---")

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Actual data points
    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        alpha=0.5,
        label="Actual Data Points"
    )

    # ------------------------------------------------------------
    # Create meshgrid
    # ------------------------------------------------------------

    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        30
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        30
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )

    # ------------------------------------------------------------
    # Regression Plane
    #
    # y = w0 + w1*x1 + w2*x2
    # ------------------------------------------------------------

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3,
        edgecolor="none"
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ax.set_xlabel(
        "Danceability (Feature 1)"
    )

    ax.set_ylabel(
        "Energy (Feature 2)"
    )

    ax.set_zlabel(
        "Valence (Target)"
    )

    ax.set_title(
        "Linear Regression via Standard Least Squares\n"
        "(P=1, L=2, M=1)"
    )

    ax.legend()

    plt.tight_layout()

    # ------------------------------------------------------------
    # Save Visualization
    # ------------------------------------------------------------

    output_path = os.path.join(
        REPORT_DIR,
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n-> Successfully saved 3D regression plot to:"
    )

    print(output_path)

    # ------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("LINEAR REGRESSION EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train_linear_regression_ls()