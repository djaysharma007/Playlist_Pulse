import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ============================================================
# PLAYLIST_PULSE — GRADIENT DESCENT LINEAR REGRESSION
# ============================================================

# ------------------------------------------------------------
# Import validated ingestion function
# ------------------------------------------------------------

DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data"
    )
)

sys.path.append(DATA_DIR)

from ingest import load_and_validate_data


# ------------------------------------------------------------
# Dataset path
# ------------------------------------------------------------

DATA_PATH = os.path.join(
    DATA_DIR,
    "raw_playlist_data.csv"
)


# ============================================================
# 1. COST FUNCTION
# ============================================================

def compute_cost(X, y, w):
    """
    Computes the Mean Squared Error cost divided by 2.

    J(w) = (1 / 2m) * Sum((Xw - y)^2)
    """

    m = len(y)

    predictions = np.dot(
        X,
        w
    )

    errors = predictions - y

    cost = (
        1 / (2 * m)
    ) * np.sum(
        errors ** 2
    )

    return cost


# ============================================================
# 2. GRADIENT DESCENT
# ============================================================

def gradient_descent(
    X,
    y,
    w,
    alpha,
    num_iters
):
    """
    Implements Gradient Descent optimization
    completely from scratch using NumPy.
    """

    m = len(y)

    cost_history = []

    for i in range(num_iters):

        # ----------------------------------------------------
        # Calculate predictions
        # ----------------------------------------------------

        predictions = np.dot(
            X,
            w
        )

        # ----------------------------------------------------
        # Calculate errors
        # ----------------------------------------------------

        errors = predictions - y

        # ----------------------------------------------------
        # Calculate gradient
        #
        # Gradient = (1/m) * X^T * (Xw - y)
        # ----------------------------------------------------

        gradient = (
            1 / m
        ) * np.dot(
            X.T,
            errors
        )

        # ----------------------------------------------------
        # Update weights
        #
        # w = w - alpha * gradient
        # ----------------------------------------------------

        w = w - (
            alpha * gradient
        )

        # ----------------------------------------------------
        # Calculate and store cost
        # ----------------------------------------------------

        cost = compute_cost(
            X,
            y,
            w
        )

        cost_history.append(
            cost
        )

    return w, cost_history


# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================

def run_gradient_descent_experiment():

    print("\n" + "#" * 60)
    print("PLAYLIST_PULSE — LINEAR REGRESSION")
    print("GRADIENT DESCENT METHOD")
    print("#" * 60)

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("--- DATA LOADING ---")
    print("=" * 60)

    print(
        f"Loading dataset from:\n{DATA_PATH}"
    )

    df = load_and_validate_data(
        DATA_PATH
    )

    print(
        f"Dataset loaded successfully."
    )

    print(
        f"Total rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Select Features and Target
    #
    # Playlist_Pulse does not contain:
    # cgpa
    # salary_package_lpa
    #
    # Therefore:
    #
    # Input  = danceability
    # Output = valence
    # --------------------------------------------------------

    feature_cols = [
        "danceability"
    ]

    target_col = "valence"

    required_columns = (
        feature_cols
        + [target_col]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns are missing "
            f"from dataset: {missing_columns}"
        )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df_clean = df.dropna(
        subset=required_columns
    ).copy()

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    X_raw = df_clean[
        feature_cols
    ].values

    y_raw = df_clean[
        target_col
    ].values.reshape(
        -1,
        1
    )

    print("\n" + "=" * 60)
    print("--- DATASET CONFIGURATION ---")
    print("=" * 60)

    print(
        f"Valid samples: {len(df_clean)}"
    )

    print(
        f"Input feature: {feature_cols[0]}"
    )

    print(
        f"Target feature: {target_col}"
    )

    # --------------------------------------------------------
    # 80/20 Train-Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42
    )

    print("\n" + "=" * 60)
    print("--- TRAIN / TEST SPLIT ---")
    print("=" * 60)

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        "Training percentage: 80%"
    )

    print(
        "Testing percentage: 20%"
    )

    # --------------------------------------------------------
    # Feature Scaling
    #
    # Scaling is important for stable Gradient Descent.
    # --------------------------------------------------------

    scaler_x = StandardScaler()

    scaler_y = StandardScaler()

    # Fit only using training data

    X_train_scaled = scaler_x.fit_transform(
        X_train
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    )

    # Transform test data using training scalers

    X_test_scaled = scaler_x.transform(
        X_test
    )

    y_test_scaled = scaler_y.transform(
        y_test
    )

    print("\n" + "=" * 60)
    print("--- FEATURE SCALING ---")
    print("=" * 60)

    print(
        "StandardScaler applied to "
        "training features and target."
    )

    print(
        "Test data transformed using "
        "training-set scaling parameters."
    )

    # --------------------------------------------------------
    # Add Intercept / Bias Column
    #
    # Model:
    #
    # y = w0 + w1*x
    #
    # Design matrix:
    #
    # [1, x]
    # --------------------------------------------------------

    X_train_design = np.hstack(
        [
            np.ones(
                (
                    X_train_scaled.shape[0],
                    1
                )
            ),
            X_train_scaled
        ]
    )

    X_test_design = np.hstack(
        [
            np.ones(
                (
                    X_test_scaled.shape[0],
                    1
                )
            ),
            X_test_scaled
        ]
    )

    print("\n" + "=" * 60)
    print("--- DESIGN MATRIX ---")
    print("=" * 60)

    print(
        f"Training design matrix shape: "
        f"{X_train_design.shape}"
    )

    print(
        "Design matrix contains:"
    )

    print(
        "[1, danceability_scaled]"
    )

    # ========================================================
    # LEARNING RATE EXPERIMENT
    # ========================================================

    learning_rates = [
        0.001,
        0.01,
        0.1,
        0.5
    ]

    num_iterations = 1000

    # --------------------------------------------------------
    # Create reports directory
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results = {}

    # --------------------------------------------------------
    # Create cost-history plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    print("\n" + "=" * 60)
    print("--- LEARNING RATE EXPERIMENT ---")
    print("=" * 60)

    # --------------------------------------------------------
    # Run Gradient Descent for every learning rate
    # --------------------------------------------------------

    for alpha in learning_rates:

        print(
            f"\nRunning Gradient Descent "
            f"with alpha = {alpha}"
        )

        # Initialize weights to zero

        w_init = np.zeros(
            (
                X_train_design.shape[1],
                1
            )
        )

        # Run Gradient Descent

        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )

        # Store results

        results[alpha] = {
            "weights": w_opt,
            "history": cost_history
        }

        # Print final cost

        print(
            f"Initial cost: "
            f"{cost_history[0]:.6f}"
        )

        print(
            f"Final cost: "
            f"{cost_history[-1]:.6f}"
        )

        # Plot cost history

        plt.plot(
            cost_history,
            label=f"α = {alpha}"
        )

    # --------------------------------------------------------
    # Format Graph
    # --------------------------------------------------------

    plt.xlabel(
        "Iterations"
    )

    plt.ylabel(
        "Cost Function J(w)"
    )

    plt.title(
        "Effect of Learning Rate "
        "on Gradient Descent Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # --------------------------------------------------------
    # Save Graph
    # --------------------------------------------------------

    output_path = os.path.join(
        REPORT_DIR,
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 60)

    print(
        "Learning-rate comparison graph "
        "saved successfully:"
    )

    print(
        output_path
    )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    # Select learning rate for final model

    best_alpha = 0.1

    final_w = results[
        best_alpha
    ]["weights"]

    final_cost_history = results[
        best_alpha
    ]["history"]

    # --------------------------------------------------------
    # Display Parameters
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        f"--- CUSTOM GRADIENT DESCENT "
        f"PARAMETERS (α = {best_alpha}) ---"
    )
    print("=" * 60)

    print(
        f"Intercept (w0): "
        f"{final_w[0, 0]:.6f}"
    )

    print(
        f"Coefficient for danceability (w1): "
        f"{final_w[1, 0]:.6f}"
    )

    print(
        f"Final training cost: "
        f"{final_cost_history[-1]:.6f}"
    )

    # ========================================================
    # TEST SET PREDICTIONS
    # ========================================================

    # --------------------------------------------------------
    # Predictions on scaled test data
    # --------------------------------------------------------

    y_test_pred_scaled = np.dot(
        X_test_design,
        final_w
    )

    # --------------------------------------------------------
    # Convert predictions back to original valence scale
    # --------------------------------------------------------

    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled
    )

    # ========================================================
    # TEST PERFORMANCE
    # ========================================================

    test_errors = (
        y_test_pred - y_test
    )

    # Mean Squared Error

    test_mse = np.mean(
        test_errors ** 2
    )

    # Root Mean Squared Error

    test_rmse = np.sqrt(
        test_mse
    )

    # --------------------------------------------------------
    # R² Score
    # --------------------------------------------------------

    ss_res = np.sum(
        (y_test - y_test_pred) ** 2
    )

    ss_total = np.sum(
        (
            y_test
            - np.mean(y_test)
        ) ** 2
    )

    test_r2 = 1 - (
        ss_res / ss_total
    )

    print("\n" + "=" * 60)
    print("--- TEST SET PERFORMANCE ---")
    print("=" * 60)

    print(
        f"MSE:  {test_mse:.6f}"
    )

    print(
        f"RMSE: {test_rmse:.6f}"
    )

    print(
        f"R² Score: {test_r2:.6f}"
    )

    # ========================================================
    # SCIKIT-LEARN COMPARISON
    # ========================================================

    print("\n" + "=" * 60)
    print("--- SCIKIT-LEARN COMPARISON ---")
    print("=" * 60)

    # --------------------------------------------------------
    # Train Scikit-learn Linear Regression
    # --------------------------------------------------------

    sklearn_model = LinearRegression()

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    # --------------------------------------------------------
    # Extract Scikit-learn parameters
    # --------------------------------------------------------

    sklearn_intercept = (
        sklearn_model.intercept_[0]
    )

    sklearn_coefficient = (
        sklearn_model.coef_[0, 0]
    )

    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_intercept:.6f}"
    )

    print(
        f"Scikit-learn Coefficient: "
        f"{sklearn_coefficient:.6f}"
    )

    # ========================================================
    # PARAMETER COMPARISON
    # ========================================================

    intercept_difference = abs(
        final_w[0, 0]
        - sklearn_intercept
    )

    coefficient_difference = abs(
        final_w[1, 0]
        - sklearn_coefficient
    )

    print("\n" + "=" * 60)
    print("--- PARAMETER DIFFERENCE ---")
    print("=" * 60)

    print(
        f"Intercept difference: "
        f"{intercept_difference:.6f}"
    )

    print(
        f"Coefficient difference: "
        f"{coefficient_difference:.6f}"
    )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n" + "#" * 60)
    print("GRADIENT DESCENT EXECUTION COMPLETE")
    print("#" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_gradient_descent_experiment()