"""
Centralised configuration for the Advertising Sales Prediction project.

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

from pathlib import Path

############################################################
# Paths
############################################################
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Advertising.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "advertising_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

############################################################
# Dataset
############################################################
INDEX_COLUMN = "Unnamed: 0"
TARGET_COLUMN = "sales"
FEATURE_COLUMNS = ["TV", "radio", "newspaper"]

############################################################
# Data splitting and cross-validation
############################################################
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42

############################################################
# Model selection
############################################################
# RMSE is measured in the target's units and penalises larger errors.
PRIMARY_METRIC = "rmse"
CV_SCORING = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
LOWER_IS_BETTER = ("mae", "mse", "rmse")

BASELINE_FAMILY = "dummy_mean"
MODEL_FAMILIES = [
    "dummy_mean",
    "linear_regression",
    "ridge",
    "lasso",
    "polynomial",
]
FAMILY_LABELS = {
    "dummy_mean": "Dummy Regressor (mean baseline)",
    "linear_regression": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "polynomial": "Polynomial Regression (degree 2)",
}
FAMILY_COMPLEXITY = {
    "dummy_mean": 0,
    "linear_regression": 1,
    "ridge": 2,
    "lasso": 2,
    "polynomial": 3,
}

# Search grids are intentionally compact for this 200-row dataset.
PARAM_GRIDS = {
    "ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "polynomial": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0],
    },
}

############################################################
# Reporting
############################################################
METADATA_PROJECT_NAME = (
    "Advertising Sales Prediction Using Regression and Regularization"
)
