"""
Regression metrics, holdout evaluation, comparison and report writers.

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


############################################################
# Project Modules
############################################################
import config


############################################################
# Function name : compute_metrics
# Description   : Calculate standard regression metrics
# Input         : actual and predicted target values
# Output        : dictionary of regression metrics
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
) -> Dict[str, float]:
    """Calculate MAE, MSE, RMSE and R-squared."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Evaluation arrays cannot be empty.")
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have equal lengths.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation arrays must contain finite values.")

    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


############################################################
# Function name : evaluate_on_holdout
# Description   : Evaluate the final fitted model on holdout data
# Input         : fitted pipeline, X_test and y_test
# Output        : metrics and prediction arrays
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def evaluate_on_holdout(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """Evaluate once on the untouched holdout set after model selection."""
    if X_test.empty or len(y_test) == 0:
        raise ValueError("Holdout data cannot be empty.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have equal lengths.")

    predictions = pipeline.predict(X_test)
    return {
        "metrics": compute_metrics(y_test, predictions),
        "n_test_rows": int(len(y_test)),
        "y_true": np.asarray(y_test, dtype=float).tolist(),
        "y_pred": np.asarray(predictions, dtype=float).tolist(),
        "residuals": (
            np.asarray(y_test, dtype=float) - np.asarray(predictions, dtype=float)
        ).tolist(),
    }


############################################################
# Function name : build_comparison_report
# Description   : Combine CV and holdout results for reporting
# Input         : CV DataFrame, selected family and holdout results
# Output        : comparison DataFrame
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def build_comparison_report(
    cv_results: pd.DataFrame,
    selected_family: str,
    holdout_results: Dict,
) -> pd.DataFrame:
    """Attach final holdout metrics only to the selected model row."""
    report = cv_results.copy()
    for metric, value in holdout_results["metrics"].items():
        report[f"holdout_{metric}"] = np.nan
        report.loc[report["family"] == selected_family, f"holdout_{metric}"] = value
    report["selected_model"] = report["family"] == selected_family
    return report


############################################################
# Function name : save_json
# Description   : Save a dictionary as indented JSON
# Input         : dictionary and destination path
# Output        : saved path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def save_json(obj: Dict, path: Path) -> Path:
    """Write JSON data after converting common NumPy values."""
############################################################
# Function name : convert
# Description   : Convert NumPy and container values into JSON-compatible objects.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def convert(value):
        """Convert NumPy and container values into JSON-compatible objects."""
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(obj), handle, indent=2, allow_nan=False)
    return path


############################################################
# Function name : save_csv
# Description   : Save a DataFrame without its index
# Input         : DataFrame and destination path
# Output        : saved path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV without its pandas index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


############################################################
# Function name : extract_model_coefficients
# Description   : Recover coefficients in original expanded-feature units
# Input         : fitted regression pipeline
# Output        : coefficient analysis DataFrame
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def extract_model_coefficients(pipeline) -> pd.DataFrame:
    """Extract coefficients and term names from a fitted linear estimator."""
    if "model" not in pipeline.named_steps:
        return pd.DataFrame(columns=["term", "coefficient"])

    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=["term", "coefficient"])

    if "polynomial" in pipeline.named_steps:
        transformer = pipeline.named_steps["polynomial"]
        terms = transformer.get_feature_names_out(config.FEATURE_COLUMNS)
    else:
        terms = np.asarray(config.FEATURE_COLUMNS, dtype=object)

    coefficients = np.asarray(estimator.coef_, dtype=float).ravel()
    if "scaler" in pipeline.named_steps:
        coefficients = coefficients / pipeline.named_steps["scaler"].scale_

    if len(terms) != len(coefficients):
        raise ValueError("Coefficient count does not match transformed feature names.")

    result = pd.DataFrame({
        "term": terms,
        "coefficient": coefficients,
    })
    result["absolute_coefficient"] = result["coefficient"].abs()
    result = result.sort_values("absolute_coefficient", ascending=False)
    result["interpretation_note"] = (
        "Associational model coefficient; not a causal advertising effect."
    )
    return result.reset_index(drop=True)
