"""
End-to-end workflow for Advertising Sales Prediction.

Run from the project root:
    python main.py

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

import json
from pathlib import Path

import numpy as np
import pandas as pd


############################################################
# Project Modules
############################################################
import config
from data_loader import (
    clean_dataset,
    make_train_test_split,
    read_data,
    split_features_target,
    summarize_dataset,
    validate_dataset,
)
from evaluation import (
    build_comparison_report,
    evaluate_on_holdout,
    extract_model_coefficients,
    save_csv,
    save_json,
)
from model_training import (
    build_pipeline,
    build_metadata,
    run_cv_comparison,
    save_pipeline,
    select_best_candidate,
    fit_selected_pipeline,
)
from visualization import (
    plot_actual_vs_predicted,
    plot_coefficients,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_target_relationships,
    plot_model_comparison,
    plot_model_coefficients,
    plot_residual_distribution,
    plot_residuals_vs_predicted,
    plot_target_distribution,
)


############################################################
# Function name : prepare_directories
# Description   : Create model and output directories
# Input         : None
# Output        : None
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def prepare_directories() -> None:
    """Create all directories required by the workflow."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


############################################################
# Function name : run_data_analysis
# Description   : Load, clean, validate and summarise the dataset
# Input         : None
# Output        : cleaned DataFrame, features, target and summary
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_data_analysis():
    """Load the raw CSV, remove its exported index and validate the data."""
    print("\n[1/7] Loading and validating dataset...")
    raw_data = read_data()
    data = clean_dataset(raw_data)
    data = validate_dataset(data)

    X, y = split_features_target(data)
    summary = summarize_dataset(data)
    save_json(summary, config.REPORTS_DIR / "dataset_summary.json")

    print(f"Dataset shape after cleaning: {data.shape}")
    print(f"Observations: {len(data)}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {config.TARGET_COLUMN}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print(f"Correlation with sales: {summary['correlation_with_target']}")
    return data, X, y, summary


############################################################
# Function name : run_eda_visualizations
# Description   : Generate exploratory data analysis figures
# Input         : validated DataFrame
# Output        : dictionary of saved figure paths
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_eda_visualizations(data: pd.DataFrame) -> dict:
    """Generate a compact set of EDA figures."""
    print("\n[2/7] Generating EDA visualizations...")
    figures = {
        "target_distribution": plot_target_distribution(
            data, config.FIGURES_DIR / "target_distribution.png"
        ),
        "feature_distributions": plot_feature_distributions(
            data, config.FIGURES_DIR / "feature_distributions.png"
        ),
        "feature_target_relationships": plot_feature_target_relationships(
            data, config.FIGURES_DIR / "feature_target_relationships.png"
        ),
        "correlation_heatmap": plot_correlation_heatmap(
            data, config.FIGURES_DIR / "correlation_heatmap.png"
        ),
    }
    print(f"Generated {len(figures)} EDA figures.")
    return figures


############################################################
# Function name : run_model_comparison
# Description   : Run cross-validation and select a candidate
# Input         : training features and target
# Output        : CV results and selected candidate
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_model_comparison(X_train, y_train):
    """Compare model families using training data only."""
    print("\n[3/7] Comparing regression models with 5-fold CV...")
    cv_results = run_cv_comparison(X_train, y_train)
    save_csv(cv_results, config.REPORTS_DIR / "cv_model_comparison.csv")
    selected = select_best_candidate(cv_results)
    print(cv_results[
        ["model", "mae_mean", "rmse_mean", "rmse_std", "r2_mean"]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected candidate: {selected['model']}")
    return cv_results, selected


############################################################
# Function name : run_final_training
# Description   : Tune/refit the selected model on training data
# Input         : training data and selected candidate
# Output        : fitted pipeline and tuning details
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_final_training(X_train, y_train, selected):
    """Refit the selected family and its CV-chosen parameters on training data."""
    print("\n[4/7] Fitting the selected model configuration...")
    pipeline, tuning_details = fit_selected_pipeline(
        X_train,
        y_train,
        selected["family"],
        selected.get("best_params", {}),
    )
    print(f"Best parameters: {tuning_details.get('best_params', 'Not applicable')}")
    return pipeline, tuning_details


############################################################
# Function name : run_holdout_evaluation
# Description   : Evaluate final model and save holdout reports
# Input         : fitted pipeline, holdout data and CV results
# Output        : holdout results and comparison report
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_holdout_evaluation(pipeline, X_test, y_test, cv_results, selected):
    """Evaluate the final model once on the untouched holdout data."""
    print("\n[5/7] Evaluating on the untouched 25% holdout set...")
    results = evaluate_on_holdout(pipeline, X_test, y_test)
    report = build_comparison_report(
        cv_results, selected["family"], results
    )
    save_csv(report, config.REPORTS_DIR / "model_comparison_with_holdout.csv")
    save_json(results["metrics"], config.REPORTS_DIR / "holdout_metrics.json")

    print("Holdout metrics:")
    for metric, value in results["metrics"].items():
        print(f"  {metric.upper():<8}: {value:.4f}")
    return results, report


############################################################
# Function name : run_diagnostics
# Description   : Generate final-model diagnostic visualizations
# Input         : fitted pipeline and holdout results
# Output        : dictionary of saved figure paths
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_diagnostics(pipeline, results):
    """Generate actual-vs-predicted and residual diagnostic plots."""
    print("\n[6/7] Generating model diagnostics...")
    figures = {
        "actual_vs_predicted": plot_actual_vs_predicted(
            results["y_true"], results["y_pred"],
            config.FIGURES_DIR / "actual_vs_predicted.png",
        ),
        "residual_distribution": plot_residual_distribution(
            results["residuals"],
            config.FIGURES_DIR / "residual_distribution.png",
        ),
        "residuals_vs_predicted": plot_residuals_vs_predicted(
            results["y_pred"], results["residuals"],
            config.FIGURES_DIR / "residuals_vs_predicted.png",
        ),
    }
    coefficient_path = plot_coefficients(
        pipeline, config.FIGURES_DIR / "linear_coefficients.png"
    )
    if coefficient_path is not None:
        figures["linear_coefficients"] = coefficient_path
    return figures


############################################################
# Function name : run_workflow
# Description   : Execute the complete Part 1 machine learning workflow
# Input         : None
# Output        : final fitted pipeline and run summary
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def run_workflow():
    """Execute all Part 1 stages in the correct methodological order."""
    prepare_directories()

    data, X, y, summary = run_data_analysis()
    eda_figures = run_eda_visualizations(data)

    print("\n[3/7] Creating train-test split...")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    print(f"Training rows: {len(X_train)} | Holdout rows: {len(X_test)}")

    cv_results, selected = run_model_comparison(X_train, y_train)
    plot_model_comparison(
        cv_results, config.FIGURES_DIR / "model_comparison_cv.png"
    )
    pipeline, tuning_details = run_final_training(
        X_train, y_train, selected
    )

    holdout_results, comparison_report = run_holdout_evaluation(
        pipeline, X_test, y_test, cv_results, selected
    )
    diagnostic_figures = run_diagnostics(pipeline, holdout_results)

    coefficient_table = extract_model_coefficients(pipeline)
    if not coefficient_table.empty:
        save_csv(
            coefficient_table,
            config.REPORTS_DIR / "coefficient_analysis.csv",
        )
        diagnostic_figures["model_coefficients"] = plot_model_coefficients(
            coefficient_table,
            config.FIGURES_DIR / "model_coefficients.png",
        )

    metadata = build_metadata(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        family=selected["family"],
        tuning_details=tuning_details,
        n_test_rows=len(X_test),
    )
    metadata["selection"] = selected
    metadata["holdout_metrics"] = holdout_results["metrics"]
    model_path, metadata_path = save_pipeline(pipeline, metadata)

    summary_report = {
        "project": config.METADATA_PROJECT_NAME,
        "dataset_summary": summary,
        "selected_model": selected,
        "tuning_details": tuning_details,
        "holdout_metrics": holdout_results["metrics"],
        "artifacts": {
            "model": str(model_path.relative_to(config.PROJECT_ROOT)),
            "metadata": str(metadata_path.relative_to(config.PROJECT_ROOT)),
            "figures": [
                str(Path(path).relative_to(config.PROJECT_ROOT))
                for path in {**eda_figures, **diagnostic_figures}.values()
            ],
        },
    }
    save_json(summary_report, config.REPORTS_DIR / "run_summary.json")

    print("\n[7/7] Saving model and final reports...")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Reports directory: {config.REPORTS_DIR}")
    print(f"Figures directory: {config.FIGURES_DIR}")
    print("\nPart 1 ML workflow completed successfully.")
    return pipeline, summary_report


if __name__ == "__main__":
    run_workflow()
