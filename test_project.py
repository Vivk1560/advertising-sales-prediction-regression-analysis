"""
Automated quality checks for the Advertising Sales Prediction project.

Run from the project root:
    python -m unittest -v test_project.py

Tests use temporary files for artifact-failure scenarios and do not retrain
or overwrite the published model.

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


############################################################
# Project Modules
############################################################
import config
from data_loader import (
    DataValidationError,
    clean_dataset,
    make_train_test_split,
    read_data,
    split_features_target,
    validate_dataset,
)
from evaluation import compute_metrics
from inference import (
    ModelArtifactError,
    PredictionInputError,
    check_feature_ranges,
    load_model_artifacts,
    predict_batch,
    predict_single,
    validate_prediction_input,
)
from model_training import (
    build_pipeline,
    calculate_sha256,
    fit_selected_pipeline,
    select_best_candidate,
)


class ProjectTests(unittest.TestCase):
    """Verify data quality, regression workflow, artifacts and inference."""

    @classmethod
############################################################
# Function name : setUpClass
# Description   : Load the source dataset and trusted saved model once for this suit
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def setUpClass(cls):
        """Load the source dataset and trusted saved model once for this suite."""
        cls.dataset = validate_dataset(clean_dataset(read_data()))
        cls.pipeline, cls.metadata = load_model_artifacts()

    ############################################################
    # Dataset validation
    ############################################################

############################################################
# Function name : test_dataset_schema_and_quality
# Description   : Confirm the source dataset has the expected clean schema.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_dataset_schema_and_quality(self):
        """Confirm the source dataset has the expected clean schema."""
        self.assertEqual(self.dataset.shape, (200, 4))
        self.assertEqual(
            list(self.dataset.columns),
            config.FEATURE_COLUMNS + [config.TARGET_COLUMN],
        )
        self.assertFalse(self.dataset.isna().any().any())
        self.assertEqual(int(self.dataset.duplicated().sum()), 0)

############################################################
# Function name : test_clean_dataset_removes_exported_index
# Description   : Remove only the redundant exported CSV index column.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_clean_dataset_removes_exported_index(self):
        """Remove only the redundant exported CSV index column."""
        raw = read_data()
        cleaned = clean_dataset(raw)
        self.assertNotIn(config.INDEX_COLUMN, cleaned.columns)
        self.assertEqual(len(raw), len(cleaned))

############################################################
# Function name : test_dataset_rejects_missing_column
# Description   : Reject a training dataset with an incomplete feature schema.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_dataset_rejects_missing_column(self):
        """Reject a training dataset with an incomplete feature schema."""
        invalid = self.dataset.drop(columns=["radio"])
        with self.assertRaises(DataValidationError):
            validate_dataset(invalid)

############################################################
# Function name : test_dataset_rejects_missing_or_infinite_values
# Description   : Reject non-finite values rather than silently imputing them.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_dataset_rejects_missing_or_infinite_values(self):
        """Reject non-finite values rather than silently imputing them."""
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                invalid = self.dataset.copy()
                invalid.loc[0, "TV"] = value
                with self.assertRaises(DataValidationError):
                    validate_dataset(invalid)

############################################################
# Function name : test_dataset_rejects_duplicate_rows
# Description   : Reject duplicated observations in the training dataset.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_dataset_rejects_duplicate_rows(self):
        """Reject duplicated observations in the training dataset."""
        invalid = pd.concat([self.dataset, self.dataset.iloc[[0]]], ignore_index=True)
        with self.assertRaises(DataValidationError):
            validate_dataset(invalid)

############################################################
# Function name : test_reproducible_train_holdout_split
# Description   : Verify the configured 150/50 split and deterministic row membershi
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_reproducible_train_holdout_split(self):
        """Verify the configured 150/50 split and deterministic row membership."""
        X, y = split_features_target(self.dataset)
        first = make_train_test_split(X, y)
        second = make_train_test_split(X, y)
        X_train, X_test, y_train, y_test = first
        self.assertEqual((len(X_train), len(X_test)), (150, 50))
        self.assertEqual((len(y_train), len(y_test)), (150, 50))
        self.assertTrue(X_train.index.equals(second[0].index))
        self.assertTrue(X_test.index.equals(second[1].index))
        self.assertFalse(set(X_train.index) & set(X_test.index))

    ############################################################
    # Model configuration and evaluation
    ############################################################

############################################################
# Function name : test_all_model_families_build_pipelines
# Description   : Confirm every configured model family creates a valid pipeline.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_all_model_families_build_pipelines(self):
        """Confirm every configured model family creates a valid pipeline."""
        for family in config.MODEL_FAMILIES:
            with self.subTest(family=family):
                self.assertIsInstance(build_pipeline(family), Pipeline)
        with self.assertRaises(ValueError):
            build_pipeline("unknown_family")

############################################################
# Function name : test_model_selection_uses_lowest_cv_rmse
# Description   : Select the lowest-RMSE non-baseline candidate and its parameters.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_model_selection_uses_lowest_cv_rmse(self):
        """Select the lowest-RMSE non-baseline candidate and its parameters."""
        scores = pd.DataFrame([
            {"family": "dummy_mean", "model": "Dummy", "rmse_mean": 0.01,
             "rmse_std": 0.1, "best_params": "{}"},
            {"family": "ridge", "model": "Ridge", "rmse_mean": 1.2,
             "rmse_std": 0.1, "best_params": '{"model__alpha": 0.1}'},
            {"family": "polynomial", "model": "Polynomial", "rmse_mean": 0.7,
             "rmse_std": 0.2, "best_params": '{"model__alpha": 0.1}'},
        ])
        selected = select_best_candidate(scores)
        self.assertEqual(selected["family"], "polynomial")
        self.assertEqual(selected["best_params"], {"model__alpha": 0.1})

############################################################
# Function name : test_selected_pipeline_fits_training_data
# Description   : Fit a lightweight model on training rows without accessing holdout
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_selected_pipeline_fits_training_data(self):
        """Fit a lightweight model on training rows without accessing holdout."""
        X, y = split_features_target(self.dataset)
        X_train, X_test, y_train, _ = make_train_test_split(X, y)
        fitted, details = fit_selected_pipeline(X_train, y_train, "linear_regression")
        predictions = fitted.predict(X_test.iloc[:2])
        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(details["training_rows"], 150)

############################################################
# Function name : test_regression_metric_calculations
# Description   : Check MAE, MSE, RMSE and R-squared against a simple example.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_regression_metric_calculations(self):
        """Check MAE, MSE, RMSE and R-squared against a simple example."""
        metrics = compute_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["mse"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1 / 3))
        self.assertAlmostEqual(metrics["r2"], 0.5)
        with self.assertRaises(ValueError):
            compute_metrics([1, 2], [1])

    ############################################################
    # Saved model and metadata integrity
    ############################################################

############################################################
# Function name : test_saved_model_checksum_and_feature_order
# Description   : Check saved-model integrity, metadata and feature ordering.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_saved_model_checksum_and_feature_order(self):
        """Check saved-model integrity, metadata and feature ordering."""
        self.assertEqual(
            calculate_sha256(config.MODEL_PATH),
            self.metadata["model_sha256"],
        )
        self.assertEqual(self.metadata["feature_order"], config.FEATURE_COLUMNS)
        self.assertEqual(self.metadata["n_train_rows"], 150)
        self.assertEqual(self.metadata["n_test_rows"], 50)

############################################################
# Function name : test_artifact_loader_rejects_missing_model
# Description   : Raise a domain-specific error when the model file is absent.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_artifact_loader_rejects_missing_model(self):
        """Raise a domain-specific error when the model file is absent."""
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.joblib"
            with self.assertRaises(ModelArtifactError):
                load_model_artifacts(missing, config.METADATA_PATH)

############################################################
# Function name : test_artifact_loader_rejects_checksum_mismatch
# Description   : Reject altered metadata before attempting to unpickle a model.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_artifact_loader_rejects_checksum_mismatch(self):
        """Reject altered metadata before attempting to unpickle a model."""
        with tempfile.TemporaryDirectory() as folder:
            metadata = dict(self.metadata)
            metadata["model_sha256"] = "0" * 64
            path = Path(folder) / "metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ModelArtifactError, "checksum mismatch"):
                load_model_artifacts(config.MODEL_PATH, path)

############################################################
# Function name : test_artifact_loader_rejects_feature_order_mismatch
# Description   : Reject metadata that specifies a different feature order.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_artifact_loader_rejects_feature_order_mismatch(self):
        """Reject metadata that specifies a different feature order."""
        with tempfile.TemporaryDirectory() as folder:
            metadata = dict(self.metadata)
            metadata["feature_order"] = list(reversed(config.FEATURE_COLUMNS))
            path = Path(folder) / "metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ModelArtifactError, "feature order"):
                load_model_artifacts(config.MODEL_PATH, path)

    ############################################################
    # Single and batch inference
    ############################################################

############################################################
# Function name : test_single_prediction_is_finite
# Description   : Return a finite scalar and the original input values.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_single_prediction_is_finite(self):
        """Return a finite scalar and the original input values."""
        values = {"TV": 150, "radio": 25, "newspaper": 20}
        result = predict_single(self.pipeline, self.metadata, values)
        self.assertTrue(np.isfinite(result["prediction"]))
        self.assertEqual(result["input"], {key: float(value) for key, value in values.items()})
        self.assertEqual(result["warnings"], [])

############################################################
# Function name : test_batch_prediction_matches_single_prediction
# Description   : Verify batch and single inference use the same fitted pipeline.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_batch_prediction_matches_single_prediction(self):
        """Verify batch and single inference use the same fitted pipeline."""
        values = {"TV": 150, "radio": 25, "newspaper": 20}
        single = predict_single(self.pipeline, self.metadata, values)
        frame = pd.DataFrame([values, {"TV": 200, "radio": 30, "newspaper": 15}])
        batch, warnings = predict_batch(self.pipeline, self.metadata, frame)
        self.assertEqual(len(batch), 2)
        self.assertIn("predicted_sales", batch.columns)
        self.assertAlmostEqual(single["prediction"], batch.iloc[0]["predicted_sales"])
        self.assertEqual(warnings, [])

############################################################
# Function name : test_prediction_input_reorders_features
# Description   : Enforce configured model feature order irrespective of CSV order.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_prediction_input_reorders_features(self):
        """Enforce configured model feature order irrespective of CSV order."""
        frame = pd.DataFrame([{"newspaper": 20, "radio": 25, "TV": 150}])
        validated = validate_prediction_input(frame)
        self.assertEqual(list(validated.columns), config.FEATURE_COLUMNS)

############################################################
# Function name : test_prediction_input_rejects_missing_and_extra_columns
# Description   : Reject missing features and unexpected batch columns.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_prediction_input_rejects_missing_and_extra_columns(self):
        """Reject missing features and unexpected batch columns."""
        with self.assertRaises(PredictionInputError):
            validate_prediction_input(pd.DataFrame([{"TV": 150, "radio": 25}]))
        with self.assertRaises(PredictionInputError):
            validate_prediction_input(pd.DataFrame([{
                "TV": 150, "radio": 25, "newspaper": 20, "sales": 14
            }]))

############################################################
# Function name : test_prediction_input_rejects_bad_values
# Description   : Reject text, null and infinite feature values.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_prediction_input_rejects_bad_values(self):
        """Reject text, null and infinite feature values."""
        for value in ("invalid", np.nan, np.inf):
            with self.subTest(value=value):
                frame = pd.DataFrame([{"TV": value, "radio": 25, "newspaper": 20}])
                with self.assertRaises(PredictionInputError):
                    validate_prediction_input(frame)

############################################################
# Function name : test_prediction_input_rejects_empty_batch
# Description   : Reject empty batch files even when headers are valid.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_prediction_input_rejects_empty_batch(self):
        """Reject empty batch files even when headers are valid."""
        frame = pd.DataFrame(columns=config.FEATURE_COLUMNS)
        with self.assertRaises(PredictionInputError):
            predict_batch(self.pipeline, self.metadata, frame)

############################################################
# Function name : test_out_of_range_inputs_generate_warning
# Description   : Warn about extrapolation without silently clipping predictions.
# Input         : Arguments documented in the function signature
# Output        : Return value documented in the function signature
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
    def test_out_of_range_inputs_generate_warning(self):
        """Warn about extrapolation without silently clipping predictions."""
        frame = pd.DataFrame([{"TV": 10000, "radio": 25, "newspaper": 20}])
        warnings = check_feature_ranges(frame, self.metadata)
        self.assertTrue(any("TV:" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
