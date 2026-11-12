"""
Headless plotting helpers for EDA, model comparison and regression diagnostics.

Every plotting function saves a PNG, closes the figure and returns its path.

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


############################################################
# Project Modules
############################################################
import config


############################################################
# Function name : _finish
# Description   : Save and close a Matplotlib figure
# Input         : figure and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def _finish(fig, path: Path) -> Path:
    """Save a figure with consistent layout and release its resources."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


############################################################
# Function name : plot_target_distribution
# Description   : Plot the distribution of the sales target
# Input         : validated DataFrame and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_target_distribution(df: pd.DataFrame, path: Path) -> Path:
    """Plot the sales target distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = df[config.TARGET_COLUMN].to_numpy(dtype=float)
    ax.hist(values, bins=20, density=True, alpha=0.75, edgecolor="white")
    if len(values) > 1:
        from scipy.stats import gaussian_kde
        density = gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), 200)
        ax.plot(grid, density(grid), linewidth=2, label="KDE")
        ax.legend()
    ax.set_title("Distribution of Sales")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")
    return _finish(fig, path)


############################################################
# Function name : plot_feature_distributions
# Description   : Plot distributions of all advertising channels
# Input         : validated DataFrame and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_feature_distributions(df: pd.DataFrame, path: Path) -> Path:
    """Plot histograms for the three advertising expenditure features."""
    fig, axes = plt.subplots(1, len(config.FEATURE_COLUMNS), figsize=(14, 4))
    for ax, feature in zip(axes, config.FEATURE_COLUMNS):
        ax.hist(df[feature], bins=18, density=True, alpha=0.75, edgecolor="white")
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
    return _finish(fig, path)


############################################################
# Function name : plot_feature_target_relationships
# Description   : Plot each predictor against sales
# Input         : validated DataFrame and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_feature_target_relationships(df: pd.DataFrame, path: Path) -> Path:
    """Plot predictor-target scatterplots with fitted linear trend lines."""
    fig, axes = plt.subplots(1, len(config.FEATURE_COLUMNS), figsize=(15, 4.5))
    for ax, feature in zip(axes, config.FEATURE_COLUMNS):
        x_values = df[feature].to_numpy(dtype=float)
        y_values = df[config.TARGET_COLUMN].to_numpy(dtype=float)
        ax.scatter(x_values, y_values, alpha=0.7, s=28)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="darkred")
        ax.set_title(f"{feature} vs Sales")
    return _finish(fig, path)


############################################################
# Function name : plot_correlation_heatmap
# Description   : Plot correlations among predictors and target
# Input         : validated DataFrame and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> Path:
    """Plot the Pearson correlation matrix."""
    columns = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    matrix = df[columns].corr().to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), labels=columns, rotation=30, ha="right")
    ax.set_yticks(range(len(columns)), labels=columns)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.2f}",
                    ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Advertising Features and Sales — Correlation")
    return _finish(fig, path)


############################################################
# Function name : plot_model_comparison
# Description   : Compare cross-validation RMSE across models
# Input         : CV results DataFrame and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_model_comparison(cv_results: pd.DataFrame, path: Path) -> Path:
    """Plot mean cross-validation RMSE with fold standard deviation."""
    ordered = cv_results.sort_values("rmse_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        ordered["model"], ordered["rmse_mean"],
        xerr=ordered["rmse_std"], capsize=4,
    )
    ax.set_title("Regression Model Comparison (5-Fold CV)")
    ax.set_xlabel("Mean CV RMSE (lower is better)")
    ax.set_ylabel("Model")
    ax.invert_yaxis()
    return _finish(fig, path)


############################################################
# Function name : plot_actual_vs_predicted
# Description   : Compare actual and predicted holdout sales
# Input         : actual, predicted values and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_actual_vs_predicted(
    y_true, y_pred, path: Path
) -> Path:
    """Plot actual against predicted values with an ideal-fit reference."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(actual, predicted, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.plot([lower, upper], [lower, upper], "r--", label="Ideal prediction")
    ax.set_title("Actual vs Predicted Sales")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.legend()
    return _finish(fig, path)


############################################################
# Function name : plot_residual_distribution
# Description   : Plot the distribution of holdout residuals
# Input         : residuals and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_residual_distribution(residuals, path: Path) -> Path:
    """Plot residual distribution and its zero reference."""
    residuals = np.asarray(residuals, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=15, alpha=0.75, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", label="Zero error")
    ax.set_title("Distribution of Holdout Residuals")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.legend()
    return _finish(fig, path)


############################################################
# Function name : plot_residuals_vs_predicted
# Description   : Check residual patterns against predicted values
# Input         : predicted values, residuals and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_residuals_vs_predicted(
    y_pred, residuals, path: Path
) -> Path:
    """Plot residuals against fitted predictions."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, residuals, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuals vs Predicted Sales")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Residual")
    return _finish(fig, path)


############################################################
# Function name : plot_coefficients
# Description   : Visualise coefficients of an interpretable linear model
# Input         : fitted pipeline and destination path
# Output        : saved figure path or None
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_coefficients(pipeline, path: Path):
    """Plot original-unit coefficients for a fitted Linear Regression model."""
    if "model" not in pipeline.named_steps:
        return None
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return None

    coefficients = np.asarray(estimator.coef_, dtype=float).ravel()
    if "scaler" in pipeline.named_steps:
        scales = pipeline.named_steps["scaler"].scale_
        coefficients = coefficients / scales
    elif "polynomial" in pipeline.named_steps:
        return None

    if len(coefficients) != len(config.FEATURE_COLUMNS):
        return None

    ordered = pd.Series(coefficients, index=config.FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(ordered.index, ordered.values)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Linear Regression Coefficients (Original Feature Units)")
    ax.set_xlabel("Coefficient")
    return _finish(fig, path)


############################################################
# Function name : plot_model_coefficients
# Description   : Plot the largest absolute model coefficients
# Input         : coefficient DataFrame and destination path
# Output        : saved figure path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def plot_model_coefficients(
    coefficients: pd.DataFrame, path: Path, top_n: int = 10
) -> Path:
    """Plot up to top_n coefficients by absolute magnitude."""
    if coefficients.empty:
        raise ValueError("Coefficient table is empty.")

    selected = coefficients.head(top_n).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(selected["term"], selected["coefficient"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Largest Regression Coefficients")
    ax.set_xlabel("Coefficient in original expanded-feature units")
    ax.set_ylabel("Model term")
    return _finish(fig, path)
