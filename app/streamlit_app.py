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

st.set_page_config(page_title="Breast Cancer Diagnostic Aid", layout="wide")


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

st.title("Breast Cancer Diagnostic Decision-Support System")
st.caption(
    "Logistic Regression pipeline with clinical threshold optimization and SHAP explainability."
)

# Sidebar controls
st.sidebar.header("Patient & Safety Settings")
sample_idx = st.sidebar.number_input(
    "Load Test Patient Sample (0 - 113)",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1,
)
true_label = "Malignant (1)" if y_test.iloc[sample_idx] == 1 else "Benign (0)"
st.sidebar.write(f"**True Test Diagnosis:** {true_label}")

threshold = st.sidebar.slider(
    "Clinical Safety Threshold",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help="Lower thresholds prioritize Recall (fewer missed cancers).",
)

# Pre-fill 30 cell measurements from selected sample
default_row = X_test.iloc[sample_idx]
inputs = {}

st.subheader("Cell Nuclei Measurements (Editable)")
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

# Preserve exact training column order before predicting
patient_df = pd.DataFrame([inputs])[X_test.columns.tolist()]
result = predict_patient(patient_df, pipe, explainer, threshold=threshold, top_n=8)

# Prediction Summary
st.divider()
st.subheader("Diagnostic Assessment")

col1, col2, col3 = st.columns(3)
col1.metric("Prediction", result["label"])
col2.metric("Malignant Probability", f"{result['malignant_probability']:.2%}")
col3.metric("Active Threshold", f"{result['threshold']:.2f}")

if result["prediction"] == 1:
    st.error(
        "High-risk pattern detected (exceeds clinical safety threshold). Recommend confirmatory follow-up."
    )
else:
    st.success(
        "Low-risk pattern detected (below clinical safety threshold)."
    )

# SHAP Local Explanation
st.subheader("Why Did the Model Predict This? (Local SHAP Drivers)")
factors = result["top_factors"].iloc[::-1]
colors = ["#d62728" if val > 0 else "#1f77b4" for val in factors["shap_impact"]]

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(factors["feature"], factors["shap_impact"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP Impact (+ pushes toward Malignant, - pushes toward Benign)")
ax.set_title("Top Feature Contributions for This Patient")
plt.tight_layout()

col_chart, col_table = st.columns([3, 2])
with col_chart:
    st.pyplot(fig)
    plt.close(fig)

with col_table:
    st.dataframe(result["top_factors"], use_container_width=True, hide_index=True)

st.info(
    "Disclaimer: Portfolio ML decision-support prototype on the UCI Wisconsin dataset. Not a certified medical device."
)