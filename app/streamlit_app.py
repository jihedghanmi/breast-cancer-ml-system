import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import load_data, split_data
from src.predict import DEFAULT_THRESHOLD, load_inference_artifacts, predict_patient

st.set_page_config(
    page_title="Breast Cancer Benchmark Classifier (Educational Demo)",
    layout="wide",
)


@st.cache_resource
def get_artifacts():
    return load_inference_artifacts()


@st.cache_data
def get_sample_patients():
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


pipe, explainer = get_artifacts()
X_test, y_test = get_sample_patients()

st.title("Breast Cancer Benchmark Classification Demo")
st.caption(
    "Educational machine learning exercise evaluated on the static UCI Wisconsin Breast Cancer Diagnostic benchmark dataset."
)

st.warning(
    "**Educational & Privacy Disclaimer:** This application is an unvalidated machine learning portfolio exercise built strictly for educational demonstration. "
    "Use built-in benchmark samples or synthetic numeric inputs only — **do not enter or upload real personal or patient medical data**. "
    "This software is not a medical device and must never be used for medical diagnosis, treatment, or clinical decision-making."
)

# Sidebar controls
st.sidebar.header("Benchmark Sample & Threshold")
st.sidebar.info(
    "Privacy Notice: Use benchmark samples or synthetic numbers only. Do not input real personal or patient medical data."
)

sample_idx = st.sidebar.number_input(
    "Load Benchmark Test Sample (0 - 113)",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1,
)
true_label = (
    "Positive Class: Malignant (1)"
    if y_test.iloc[sample_idx] == 1
    else "Negative Class: Benign (0)"
)
st.sidebar.write(f"**Benchmark Ground-Truth Label:** {true_label}")

threshold = st.sidebar.slider(
    "Classification Probability Threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help="Lower cutoffs increase positive-class Recall and lower Precision on the benchmark.",
)

# Pre-fill 30 benchmark numeric features from selected sample
default_row = X_test.iloc[sample_idx]
inputs = {}

st.subheader("Benchmark Feature Inputs (Sample or Synthetic Values Only)")
tab_mean, tab_error, tab_worst = st.tabs(
    ["Mean Features", "Error (SE) Features", "Worst Features"]
)

feature_groups = [
    (tab_mean, [c for c in X_test.columns if c.startswith("mean ")]),
    (tab_error, [c for c in X_test.columns if "error" in c]),
    (tab_worst, [c for c in X_test.columns if c.startswith("worst ")]),
]

for tab, cols in feature_groups:
    with tab:
        col_left, col_right = st.columns(2)
        for i, feature in enumerate(cols):
            target_col = col_left if i % 2 == 0 else col_right
            inputs[feature] = target_col.number_input(
                feature,
                value=float(default_row[feature]),
                format="%.5f",
                key=f"{sample_idx}_{feature}",
            )

# Preserve exact training column order before running inference
patient_df = pd.DataFrame([inputs])[X_test.columns.tolist()]
result = predict_patient(patient_df, pipe, explainer, threshold=threshold, top_n=8)

# Model Output Summary
st.divider()
st.subheader("Model Output Summary (Static Benchmark Pipeline)")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Class", result["label"])
col2.metric("Positive-Class Probability", f"{result['malignant_probability']:.2%}")
col3.metric("Active Threshold", f"{result['threshold']:.2f}")

if result["prediction"] == 1:
    st.warning(
        "Model output: Positive class (1) — predicted probability meets or exceeds the active threshold for these input values."
    )
else:
    st.success(
        "Model output: Negative class (0) — predicted probability is below the active threshold for these input values."
    )

# Local SHAP Feature Attribution
st.subheader("Local SHAP Feature Attribution (Model Interpretability)")
factors = result["top_factors"].iloc[::-1]
colors = ["#d62728" if val > 0 else "#1f77b4" for val in factors["shap_impact"]]

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(factors["feature"], factors["shap_impact"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(
    "SHAP Impact (+ pushes positive-class score up, - pushes score down)"
)
ax.set_title("Top Feature Contributions for Selected Input")
plt.tight_layout()

col_chart, col_table = st.columns([3, 2])
with col_chart:
    st.pyplot(fig)
    plt.close(fig)

with col_table:
    st.dataframe(result["top_factors"], use_container_width=True, hide_index=True)

st.caption(
    "Provided strictly 'as-is' for educational demonstration on a static public benchmark dataset. Not clinically validated; never use for medical decisions or with real patient data."
)