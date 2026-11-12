"""
Regression pipelines, cross-validated comparison, tuning and persistence.

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


############################################################
# Project Modules
############################################################
import config


############################################################
# Function name : build_pipeline
# Description   : Build an unfitted regression pipeline
# Input         : model family
# Output        : scikit-learn Pipeline
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def build_pipeline(family: str) -> Pipeline:
    """Create a model pipeline with preprocessing contained inside CV."""
    if family == "dummy_mean":
        return Pipeline([("model", DummyRegressor(strategy="mean"))])

    if family == "linear_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    if family == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ])

    if family == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=20000, random_state=config.RANDOM_STATE)),
        ])

    if family == "polynomial":
        return Pipeline([
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ])

    raise ValueError(f"Unknown model family: {family!r}")


############################################################
# Function name : run_cv_comparison
# Description   : Compare regression families using training-only CV
# Input         : X_train and y_train
# Output        : DataFrame of CV metrics
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_cv_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Compare model families after tuning supported hyperparameters.

    All candidate searches use the same training-only folds. The holdout
    partition is not accessed by this function.
    """
    cv = KFold(
        n_splits=config.CV_N_SPLITS,
        shuffle=True,
        random_state=config.CV_RANDOM_STATE,
    )
    rows = []

    for family in config.MODEL_FAMILIES:
        pipeline = build_pipeline(family)
        parameter_grid = config.PARAM_GRIDS.get(family, {})

        if parameter_grid:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                scoring=config.CV_SCORING,
                refit=config.PRIMARY_METRIC,
                cv=cv,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            cv_results = search.cv_results_
            best_index = search.best_index_
            best_params = search.best_params_

            row = {
                "family": family,
                "model": config.FAMILY_LABELS[family],
                "is_baseline": family == config.BASELINE_FAMILY,
                "cv_folds": config.CV_N_SPLITS,
                "best_params": json.dumps(best_params, sort_keys=True),
            }

            for metric in config.CV_SCORING:
                mean_score = float(cv_results[f"mean_test_{metric}"][best_index])
                std_score = float(cv_results[f"std_test_{metric}"][best_index])
                if metric in config.LOWER_IS_BETTER:
                    mean_score = -mean_score
                row[f"{metric}_mean"] = mean_score
                row[f"{metric}_std"] = std_score
        else:
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=config.CV_SCORING,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            row = {
                "family": family,
                "model": config.FAMILY_LABELS[family],
                "is_baseline": family == config.BASELINE_FAMILY,
                "cv_folds": config.CV_N_SPLITS,
                "best_params": "{}",
            }

            for metric in config.CV_SCORING:
                values = np.asarray(scores[f"test_{metric}"], dtype=float)
                if metric in config.LOWER_IS_BETTER:
                    values = -values
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)

############################################################
# Function name : select_best_candidate
# Description   : Select the best non-baseline family using CV RMSE
# Input         : CV comparison DataFrame
# Output        : selected family and selection details
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def select_best_candidate(cv_results: pd.DataFrame) -> Dict:
    """Select lowest mean CV RMSE; use model simplicity as a tie-breaker."""
    metric = f"{config.PRIMARY_METRIC}_mean"
    candidates = cv_results.loc[
        cv_results["family"] != config.BASELINE_FAMILY
    ].copy()

    if candidates.empty or candidates[metric].isna().any():
        raise ValueError("No valid non-baseline model candidates are available.")

    candidates["complexity"] = candidates["family"].map(config.FAMILY_COMPLEXITY)
    selected = candidates.sort_values(
        [metric, "complexity"], ascending=[True, True]
    ).iloc[0]

    return {
        "family": str(selected["family"]),
        "model": str(selected["model"]),
        "cv_rmse_mean": float(selected[metric]),
        "cv_rmse_std": float(selected[f"{config.PRIMARY_METRIC}_std"]),
        "selection_metric": config.PRIMARY_METRIC,
        "best_params": json.loads(str(selected["best_params"])),
        "selection_method": (
            "Lowest tuned mean 5-fold training CV RMSE; model simplicity "
            "breaks ties. Holdout data is reserved for final evaluation."
        ),
    }


############################################################
# Function name : fit_selected_pipeline
# Description   : Refit the selected model configuration on all training data
# Input         : X_train, y_train, selected family and selected parameters
# Output        : fitted scikit-learn Pipeline and fit details
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def fit_selected_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    family: str,
    best_params: Dict = None,
):
    """Fit the selected, cross-validated configuration on all training data."""
    pipeline = build_pipeline(family)
    best_params = best_params or {}

    if best_params:
        pipeline.set_params(**best_params)

    pipeline.fit(X_train, y_train)
    return pipeline, {
        "best_params": best_params,
        "training_rows": int(len(X_train)),
        "selection_note": (
            "Hyperparameters were selected during the training-only "
            "cross-validation comparison; this step refits that configuration "
            "on the complete training partition."
        ),
    }

############################################################
# Function name : calculate_sha256
# Description   : Calculate SHA-256 checksum of a saved model
# Input         : model file path
# Output        : hexadecimal checksum
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def calculate_sha256(path: Path) -> str:
    """Calculate a SHA-256 checksum for artifact integrity metadata."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


############################################################
# Function name : build_metadata
# Description   : Create metadata for the selected fitted model
# Input         : fitted pipeline, training data and run details
# Output        : JSON-serialisable dictionary
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def build_metadata(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    family: str,
    tuning_details: Dict,
    n_test_rows: int,
) -> Dict:
    """Collect model, feature-range, split and environment information."""
    ranges = {
        column: {
            "min": float(X_train[column].min()),
            "max": float(X_train[column].max()),
            "mean": float(X_train[column].mean()),
            "std": float(X_train[column].std()),
        }
        for column in config.FEATURE_COLUMNS
    }

    return {
        "project": config.METADATA_PROJECT_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_family": family,
        "model_label": config.FAMILY_LABELS[family],
        "best_params": tuning_details.get("best_params", {}),
        "feature_order": list(config.FEATURE_COLUMNS),
        "target": config.TARGET_COLUMN,
        "training_ranges": ranges,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(n_test_rows),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "cv_n_splits": config.CV_N_SPLITS,
        "cv_random_state": config.CV_RANDOM_STATE,
        "model_file": config.MODEL_PATH.name,
        "model_sha256": None,
        "interpretation_note": (
            "Predictive model only; coefficients describe associations in this "
            "dataset and must not be interpreted as causal advertising effects."
        ),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }


############################################################
# Function name : save_pipeline
# Description   : Persist the fitted pipeline and JSON metadata
# Input         : fitted pipeline, metadata and output paths
# Output        : model path and metadata path
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def save_pipeline(
    pipeline: Pipeline,
    metadata: Dict,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> Tuple[Path, Path]:
    """Save the fitted pipeline and metadata, including model checksum."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    saved_metadata = dict(metadata)
    saved_metadata["model_sha256"] = calculate_sha256(model_path)
    saved_metadata["model_file"] = model_path.name

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(saved_metadata, handle, indent=2)

    return model_path, metadata_path


############################################################
# Function name : load_pipeline
# Description   : Load a saved model pipeline
# Input         : model file path
# Output        : fitted pipeline
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    """Load a previously saved trusted scikit-learn pipeline."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
