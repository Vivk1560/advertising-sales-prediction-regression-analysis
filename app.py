"""
Interactive Streamlit dashboard for Advertising Sales Prediction.

Start from the project root:
    streamlit run app.py

Author : Vivaan Kukreja
Date   : 12/11/2026
"""

############################################################
# Required Python Packages
############################################################

import json
from pathlib import Path

import pandas as pd
import streamlit as st


############################################################
# Project Modules
############################################################
import config
from data_loader import clean_dataset, read_data, validate_dataset
from inference import (
    ModelArtifactError,
    PredictionInputError,
    load_model_artifacts,
    predict_batch,
    predict_single,
)

st.set_page_config(
    page_title="Advertising Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

############################################################
# Styling
############################################################
st.markdown(
    """
    <style>
    .stApp { background: #f6f7fb; }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stSidebar"] * { color: #eef2f7; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #14243b, #245b70);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin-bottom: .35rem; }
    .hero p { color: #d9e8ef; margin-bottom: 0; }
    .section-note { color: #5b6575; font-size: .92rem; }
    div[data-testid="stMetric"] {
        background: white; padding: 1rem; border-radius: 12px;
        border: 1px solid #e5e9f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


############################################################
# Function name : load_dataset
# Description   : Load and validate the project's source dataset
# Input         : None
# Output        : validated advertising dataset
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
@st.cache_data
def load_dataset():
    """Load and validate the project's source dataset."""
    return validate_dataset(clean_dataset(read_data()))


############################################################
# Function name : load_artifacts
# Description   : Load and cache the verified model artifacts
# Input         : None
# Output        : fitted pipeline and metadata dictionary
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
@st.cache_resource
def load_artifacts():
    """Load and cache the integrity-checked model pipeline and metadata."""
    return load_model_artifacts()


############################################################
# Function name : hero
# Description   : Display a styled page heading and subtitle
# Input         : page title and subtitle
# Output        : rendered Streamlit hero section
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def hero(title: str, subtitle: str) -> None:
    """Render a consistent title banner for each dashboard page."""
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


############################################################
# Function name : show_image
# Description   : Display a generated project figure when available
# Input         : figure filename and display caption
# Output        : rendered image or an informational message
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def show_image(filename: str, caption: str) -> None:
    """Display a figure from the project's output directory."""
    path = config.FIGURES_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found yet: {filename}. Run `python main.py` first.")


############################################################
# Function name : format_metric
# Description   : Format a numeric metric for dashboard display
# Input         : metric value and number of decimal places
# Output        : formatted metric string
# Author        : Vivaan Kukreja
# Date          : 12/11/2026
############################################################
def format_metric(value: float, digits: int = 3) -> str:
    """Format a metric value with thousands separators and fixed precision."""
    return f"{float(value):,.{digits}f}"


############################################################
# Sidebar and shared data/artifact status
############################################################
st.sidebar.title("Advertising Sales")
st.sidebar.caption("Regression & Regularization")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "EDA",
        "Model Comparison",
        "Diagnostics",
        "Sales Prediction",
        "Batch Prediction",
    ],
)
st.sidebar.divider()

try:
    dataset = load_dataset()
    data_error = None
except Exception as exc:
    dataset = None
    data_error = str(exc)

try:
    pipeline, metadata = load_artifacts()
    model_error = None
except Exception as exc:
    pipeline, metadata = None, None
    model_error = str(exc)

st.sidebar.markdown("**Project status**")
st.sidebar.write("Dataset:", "🟢 Available" if dataset is not None else "🔴 Missing")
st.sidebar.write("Saved model:", "🟢 Verified" if pipeline is not None else "🔴 Unavailable")
if metadata:
    st.sidebar.caption(f"Model: {metadata.get('model_label', 'Saved estimator')}")
    st.sidebar.caption(f"Created: {metadata.get('created_utc', 'Not recorded')[:10]}")
st.sidebar.divider()

if data_error:
    st.error(f"Dataset could not be loaded: {data_error}")
if model_error:
    st.warning(
        "The saved model is unavailable or failed integrity validation. "
        "Prediction pages require a successful Part 1 run."
    )
    with st.expander("Technical details"):
        st.code(model_error)


############################################################
# 1. Overview
############################################################
if page == "Overview":
    hero(
        "Advertising Sales Intelligence",
        "Explore advertising-channel data, evaluate regression models and "
        "estimate sales using a saved machine-learning pipeline.",
    )
    if dataset is not None:
        summary_path = config.REPORTS_DIR / "dataset_summary.json"
        comparison_path = config.REPORTS_DIR / "model_comparison_with_holdout.csv"
        metrics_path = config.REPORTS_DIR / "holdout_metrics.json"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", f"{len(dataset):,}")
        c2.metric("Advertising channels", len(config.FEATURE_COLUMNS))
        c3.metric("Missing values", int(dataset.isna().sum().sum()))
        c4.metric("Duplicate rows", int(dataset.duplicated().sum()))

        st.subheader("Project at a glance")
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown(
                """
                This project studies the relationship between three advertising
                channels — **TV, radio and newspaper** — and the observed sales
                target. It compares a mean baseline with linear, regularized and
                degree-2 polynomial regression models.

                The saved estimator is used for scenario predictions. These
                estimates are predictive associations, **not causal ROI claims**.
                """
            )
        with right:
            st.markdown("**Dataset preview**")
            st.dataframe(dataset.head(8), width="stretch", hide_index=True)

        st.subheader("Selected model performance")
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Holdout MAE", format_metric(metrics["mae"]))
            m2.metric("Holdout RMSE", format_metric(metrics["rmse"]))
            m3.metric("Holdout MSE", format_metric(metrics["mse"]))
            m4.metric("Holdout R²", format_metric(metrics["r2"]))
        else:
            st.info("Run `python main.py` to generate evaluation reports.")
        if metadata:
            st.info(
                f"Active saved model: **{metadata.get('model_label', 'Unknown')}**. "
                "The dashboard does not retrain the model."
            )
    else:
        st.warning("Dataset unavailable. Check data/Advertising.csv.")


############################################################
# 2. Exploratory Data Analysis
############################################################
elif page == "EDA":
    hero("Exploratory Data Analysis", "Understand distributions and feature relationships.")
    if dataset is not None:
        st.subheader("Feature distributions")
        show_image("feature_distributions.png", "Advertising channel distributions")
        show_image("target_distribution.png", "Observed sales distribution")

        st.subheader("Relationships with sales")
        show_image(
            "feature_target_relationships.png",
            "Scatterplots with simple linear trend lines",
        )
        show_image(
            "correlation_heatmap.png",
            "Pearson correlation matrix for predictors and sales",
        )

        st.subheader("Correlation with sales")
        correlations = dataset[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].corr()
        sales_corr = correlations[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
        st.bar_chart(sales_corr.sort_values(ascending=False))
        st.caption(
            "Correlation describes linear association in this dataset; "
            "it does not establish causation."
        )
        with st.expander("View descriptive statistics"):
            st.dataframe(dataset.describe().T, width="stretch")
    else:
        st.warning("EDA requires the source dataset.")


############################################################
# 3. Model Comparison
############################################################
elif page == "Model Comparison":
    hero("Model Comparison", "Compare training cross-validation results and final holdout metrics.")
    cv_path = config.REPORTS_DIR / "cv_model_comparison.csv"
    report_path = config.REPORTS_DIR / "model_comparison_with_holdout.csv"

    if cv_path.is_file():
        cv_results = pd.read_csv(cv_path)
        show_image("model_comparison_cv.png", "Five-fold cross-validation RMSE")
        st.subheader("Cross-validation results")
        display_cols = [
            "model", "best_params", "mae_mean", "mae_std", "rmse_mean",
            "rmse_std", "r2_mean", "r2_std",
        ]
        st.dataframe(
            cv_results[[col for col in display_cols if col in cv_results.columns]],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Regularized model families are compared using their best "
            "training-only CV hyperparameters. The displayed scores are "
            "selection scores, not an independent estimate of generalisation. "
            "Lower MAE/RMSE is better; higher R² is better."
        )
    else:
        st.info("CV report unavailable. Run `python main.py` first.")

    if report_path.is_file():
        st.subheader("Final holdout evaluation")
        report = pd.read_csv(report_path)
        holdout_cols = [
            "model", "selected_model", "holdout_mae",
            "holdout_rmse", "holdout_r2",
        ]
        st.dataframe(
            report[[col for col in holdout_cols if col in report.columns]],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Holdout metrics are populated for the selected model only. "
            "The holdout set was not used to choose the model."
        )


############################################################
# 4. Diagnostics
############################################################
elif page == "Diagnostics":
    hero("Model Diagnostics", "Inspect prediction fit, residual patterns and model terms.")
    if metadata:
        metrics = metadata.get("holdout_metrics", {})
        a, b = st.columns(2)
        a.metric("Holdout RMSE", format_metric(metrics.get("rmse", float("nan"))))
        b.metric("Holdout R²", format_metric(metrics.get("r2", float("nan"))))

    left, right = st.columns(2)
    with left:
        show_image("actual_vs_predicted.png", "Actual values versus model predictions")
    with right:
        show_image("residual_distribution.png", "Holdout residual distribution")
    show_image("residuals_vs_predicted.png", "Residuals versus predicted sales")

    coefficient_path = config.REPORTS_DIR / "coefficient_analysis.csv"
    if coefficient_path.is_file():
        st.subheader("Regression coefficient analysis")
        coefficients = pd.read_csv(coefficient_path)
        show_image("model_coefficients.png", "Largest absolute fitted coefficients")
        st.dataframe(coefficients, width="stretch", hide_index=True)
        st.caption(
            "Polynomial terms represent combinations of input features. "
            "Their coefficients should not be read as standalone causal effects."
        )
    st.info(
        "Residual plots help identify systematic error patterns. They do not, "
        "by themselves, prove that all regression assumptions are satisfied."
    )


############################################################
# 5. Single sales prediction
############################################################
elif page == "Sales Prediction":
    hero("Sales Prediction", "Enter advertising spend values to estimate the model's sales output.")
    if pipeline is None or metadata is None:
        st.warning("Prediction is unavailable until the saved model is verified.")
    else:
        st.markdown(
            "Enter values in the same units and conventions as the training dataset. "
            "The model does not infer currency or a business-specific unit."
        )
        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)
            tv = col1.number_input("TV advertising", min_value=0.0, value=150.0, step=1.0)
            radio = col2.number_input("Radio advertising", min_value=0.0, value=25.0, step=1.0)
            newspaper = col3.number_input(
                "Newspaper advertising", min_value=0.0, value=20.0, step=1.0
            )
            submitted = st.form_submit_button(
                "Predict sales", type="primary", width="stretch"
            )

        if submitted:
            try:
                result = predict_single(
                    pipeline,
                    metadata,
                    {"TV": tv, "radio": radio, "newspaper": newspaper},
                )
                st.success("Prediction generated successfully.")
                st.metric("Predicted sales", f"{result['prediction']:.3f}")
                st.subheader("Input scenario")
                st.dataframe(
                    pd.DataFrame([result["input"]]),
                    width="stretch",
                    hide_index=True,
                )
                for warning in result["warnings"]:
                    st.warning(warning)
                if not result["warnings"]:
                    st.caption("All inputs fall within the observed training ranges.")
                st.caption(
                    "This is a model estimate, not a guarantee or a causal estimate "
                    "of the impact of changing one advertising channel."
                )
            except PredictionInputError as exc:
                st.error(str(exc))


############################################################
# 6. Batch prediction
############################################################
elif page == "Batch Prediction":
    hero("Batch Prediction", "Upload a CSV of advertising scenarios and download model estimates.")
    if pipeline is None or metadata is None:
        st.warning("Batch prediction is unavailable until the saved model is verified.")
    else:
        st.markdown(
            "Your CSV must contain exactly these three feature columns: "
            "`TV`, `radio`, `newspaper`. Extra columns are rejected to prevent "
            "accidental use of identifiers or target leakage."
        )
        template = pd.DataFrame({
            "TV": [150.0, 230.1],
            "radio": [25.0, 37.8],
            "newspaper": [20.0, 69.2],
        })
        st.download_button(
            "Download input CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="advertising_prediction_template.csv",
            mime="text/csv",
        )
        uploaded = st.file_uploader("Upload prediction CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_data = pd.read_csv(uploaded)
                st.subheader("Uploaded data preview")
                st.dataframe(batch_data.head(10), width="stretch")
                if st.button("Generate batch predictions", type="primary"):
                    result, warnings = predict_batch(
                        pipeline, metadata, batch_data
                    )
                    st.session_state["batch_prediction_result"] = result
                    st.session_state["batch_prediction_warnings"] = warnings
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Could not read uploaded CSV: {exc}")
            except PredictionInputError as exc:
                st.error(str(exc))

        result = st.session_state.get("batch_prediction_result")
        if result is not None:
            st.subheader("Prediction results")
            st.dataframe(result, width="stretch", hide_index=True)
            st.metric("Rows predicted", f"{len(result):,}")
            st.download_button(
                "Download predictions CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="advertising_sales_predictions.csv",
                mime="text/csv",
            )
            for warning in st.session_state.get("batch_prediction_warnings", []):
                st.warning(warning)
            if not st.session_state.get("batch_prediction_warnings"):
                st.caption("All uploaded feature values fall within training ranges.")
