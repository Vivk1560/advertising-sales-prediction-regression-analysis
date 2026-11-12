"""
Dataset loading, schema validation, splitting and descriptive summaries.

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


############################################################
# Project Modules
############################################################
import config


class DataValidationError(ValueError):
    """Raised when the dataset does not satisfy the expected schema."""


############################################################
# Function name : read_data
# Description   : Read the Advertising CSV into a DataFrame
# Input         : path to the CSV file
# Output        : raw DataFrame
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def read_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the dataset from the specified path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


############################################################
# Function name : clean_dataset
# Description   : Remove the known redundant exported index column
# Input         : raw DataFrame
# Output        : cleaned DataFrame
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the redundant CSV index column without altering source data."""
    if config.INDEX_COLUMN in df.columns:
        df = df.drop(columns=[config.INDEX_COLUMN])
    return df.copy()


############################################################
# Function name : validate_dataset
# Description   : Validate schema, numeric values and data quality
# Input         : DataFrame
# Output        : validated DataFrame
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the expected columns and numerical observations."""
    if df is None or df.empty:
        raise DataValidationError("Dataset is empty.")

    if not df.columns.is_unique:
        raise DataValidationError("Dataset contains duplicate column names.")

    expected = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing = [column for column in expected if column not in df.columns]
    unexpected = [column for column in df.columns if column not in expected]

    if missing or unexpected:
        raise DataValidationError(
            f"Column mismatch. Missing: {missing}; unexpected: {unexpected}"
        )

    non_numeric = [
        column for column in expected
        if not pd.api.types.is_numeric_dtype(df[column])
        or pd.api.types.is_bool_dtype(df[column])
    ]
    if non_numeric:
        raise DataValidationError(f"Non-numeric columns found: {non_numeric}")

    values = df[expected].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError("Dataset contains missing or non-finite values.")

    if df.duplicated().any():
        raise DataValidationError("Dataset contains duplicate rows.")

    return df.copy()


############################################################
# Function name : split_features_target
# Description   : Separate predictors and target variable
# Input         : validated DataFrame
# Output        : X and y
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return independent variables and the sales target."""
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].copy()
    return X, y


############################################################
# Function name : make_train_test_split
# Description   : Create a reproducible holdout split
# Input         : X, y, test size and random state
# Output        : X_train, X_test, y_train, y_test
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible random train-test split for regression."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if len(X) < config.CV_N_SPLITS:
        raise ValueError("Not enough observations for configured cross-validation.")

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


############################################################
# Function name : summarize_dataset
# Description   : Generate descriptive and data-quality statistics
# Input         : validated DataFrame
# Output        : JSON-serialisable dictionary
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def summarize_dataset(df: pd.DataFrame) -> Dict:
    """Generate dataset dimensions, quality checks and descriptive statistics."""
    features = df[config.FEATURE_COLUMNS]
    q1 = features.quantile(0.25)
    q3 = features.quantile(0.75)
    iqr = q3 - q1
    outside = (features < q1 - 1.5 * iqr) | (features > q3 + 1.5 * iqr)
    correlations = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].corr()

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target": config.TARGET_COLUMN,
        "features": list(config.FEATURE_COLUMNS),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "iqr_outlier_counts": {
            column: int(outside[column].sum()) for column in config.FEATURE_COLUMNS
        },
        "iqr_outlier_note": "Descriptive IQR counts only; no observations are removed.",
        "correlation_with_target": {
            column: round(float(correlations.loc[column, config.TARGET_COLUMN]), 4)
            for column in config.FEATURE_COLUMNS
        },
        "describe": df.describe().T.round(4).to_dict(orient="index"),
    }
