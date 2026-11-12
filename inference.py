"""
Inference utilities for the Advertising Sales Prediction project.

The saved scikit-learn pipeline performs polynomial feature generation and
scaling internally. Always pass the original three input features in the
configured order; never manually transform them before prediction.

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd


############################################################
# Project Modules
############################################################
import config
from model_training import calculate_sha256


class ModelArtifactError(RuntimeError):
    """Raised when saved model artifacts are missing or inconsistent."""


class PredictionInputError(ValueError):
    """Raised when prediction input does not match the expected schema."""


############################################################
# Function name : load_model_artifacts
# Description   : Load model and metadata after integrity checks
# Input         : model and metadata paths
# Output        : fitted pipeline and metadata dictionary
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def load_model_artifacts(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> Tuple[object, Dict]:
    """Load trusted artifacts and verify checksum and feature order."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.is_file():
        raise ModelArtifactError(f"Saved model not found: {model_path}")
    if not metadata_path.is_file():
        raise ModelArtifactError(f"Model metadata not found: {metadata_path}")

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"Could not read model metadata: {exc}") from exc

    expected_hash = metadata.get("model_sha256")
    if not expected_hash:
        raise ModelArtifactError("Metadata does not contain a model checksum.")

    actual_hash = calculate_sha256(model_path)
    if actual_hash != expected_hash:
        raise ModelArtifactError(
            "Model checksum mismatch. The model file may have changed."
        )

    if metadata.get("feature_order") != config.FEATURE_COLUMNS:
        raise ModelArtifactError(
            "Model metadata feature order does not match the current configuration."
        )

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model pipeline: {exc}") from exc

    if not hasattr(pipeline, "predict"):
        raise ModelArtifactError("Loaded artifact does not provide predict().")

    return pipeline, metadata


############################################################
# Function name : validate_prediction_input
# Description   : Validate a single or batch prediction DataFrame
# Input         : DataFrame and strict-schema flag
# Output        : validated DataFrame in configured feature order
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def validate_prediction_input(
    data: pd.DataFrame,
    strict_schema: bool = True,
) -> pd.DataFrame:
    """Validate columns and numeric values without silently imputing data."""
    if not isinstance(data, pd.DataFrame):
        raise PredictionInputError("Prediction input must be a pandas DataFrame.")
    if data.empty:
        raise PredictionInputError("Prediction input contains no rows.")
    if not data.columns.is_unique:
        raise PredictionInputError("Prediction input contains duplicate column names.")

    required = list(config.FEATURE_COLUMNS)
    missing = [column for column in required if column not in data.columns]
    extra = [column for column in data.columns if column not in required]

    if missing:
        raise PredictionInputError(f"Missing required feature columns: {missing}")
    if strict_schema and extra:
        raise PredictionInputError(f"Unexpected columns in prediction input: {extra}")

    validated = data.loc[:, required].copy()
    for column in required:
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise PredictionInputError(
                f"Feature '{column}' contains non-numeric values."
            ) from exc

    values = validated.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PredictionInputError(
            "Prediction input contains missing or non-finite values."
        )

    return validated.astype(float)


############################################################
# Function name : check_feature_ranges
# Description   : Flag values outside observed training ranges
# Input         : validated features and metadata
# Output        : list of human-readable range warnings
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def check_feature_ranges(
    data: pd.DataFrame,
    metadata: Dict,
) -> list:
    """Return warnings for values outside the training feature ranges."""
    ranges = metadata.get("training_ranges", {})
    warnings = []

    for feature in config.FEATURE_COLUMNS:
        if feature not in ranges:
            continue
        minimum = ranges[feature].get("min")
        maximum = ranges[feature].get("max")
        if minimum is None or maximum is None:
            continue

        observed_min = float(data[feature].min())
        observed_max = float(data[feature].max())
        if observed_min < minimum or observed_max > maximum:
            warnings.append(
                f"{feature}: input range [{observed_min:.3f}, {observed_max:.3f}] "
                f"extends beyond training range [{minimum:.3f}, {maximum:.3f}]. "
                "Predictions may be less reliable for extrapolated values."
            )

    return warnings


############################################################
# Function name : predict_single
# Description   : Generate a prediction for one advertising scenario
# Input         : fitted pipeline, metadata and feature values
# Output        : prediction, input echo and range warnings
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def predict_single(
    pipeline,
    metadata: Dict,
    values: Dict,
) -> Dict:
    """Predict sales for one row of TV, radio and newspaper inputs."""
    missing = [feature for feature in config.FEATURE_COLUMNS if feature not in values]
    if missing:
        raise PredictionInputError(f"Missing feature values: {missing}")

    frame = pd.DataFrame(
        [[values[feature] for feature in config.FEATURE_COLUMNS]],
        columns=config.FEATURE_COLUMNS,
    )
    validated = validate_prediction_input(frame)
    prediction = float(pipeline.predict(validated)[0])

    return {
        "prediction": prediction,
        "input": {key: float(validated.iloc[0][key]) for key in config.FEATURE_COLUMNS},
        "warnings": check_feature_ranges(validated, metadata),
    }


############################################################
# Function name : predict_batch
# Description   : Generate predictions for a batch of input rows
# Input         : fitted pipeline, metadata and batch DataFrame
# Output        : copied DataFrame with predicted sales column
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def predict_batch(
    pipeline,
    metadata: Dict,
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, list]:
    """Validate batch features, predict and append the predicted sales."""
    validated = validate_prediction_input(data, strict_schema=True)
    predictions = pipeline.predict(validated)

    result = validated.copy()
    result["predicted_sales"] = np.asarray(predictions, dtype=float)
    warnings = check_feature_ranges(validated, metadata)
    return result, warnings
