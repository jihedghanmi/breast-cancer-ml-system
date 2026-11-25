from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.data import load_data, split_data


def create_explainer(pipe, X_background):
    preprocessor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]
    X_bg_scaled = preprocessor.transform(X_background)
    return shap.LinearExplainer(classifier, X_bg_scaled)


def explain_patient(pipe, explainer, patient_df, top_n=5):
    patient_scaled = pipe.named_steps["preprocessor"].transform(patient_df)
    shap_vals = explainer.shap_values(patient_scaled)[0]

    explanation = pd.DataFrame(
        {
            "feature": patient_df.columns,
            "patient_value": patient_df.iloc[0].values,
            "shap_impact": shap_vals,
        }
    )
    explanation["abs_impact"] = explanation["shap_impact"].abs()
    explanation = explanation.sort_values("abs_impact", ascending=False).drop(
        columns="abs_impact"
    )
    return explanation.head(top_n)


def save_global_summary(
    pipe, explainer, X_data, output_path="reports/shap_summary.png"
):
    X_scaled = pipe.named_steps["preprocessor"].transform(X_data)
    shap_values = explainer.shap_values(X_scaled)

    shap.summary_plot(
        shap_values, X_scaled, feature_names=X_data.columns.tolist(), show=False
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved global SHAP summary plot to {output_path}")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipe = joblib.load("models/model.joblib")
    explainer = create_explainer(pipe, X_train)

    patient = X_test.iloc[[0]]
    malignant_prob = pipe.predict_proba(patient)[0, 1]
    decision = "MALIGNANT" if malignant_prob >= 0.30 else "BENIGN"

    print(f"Patient #0 Malignant Probability: {malignant_prob:.4f}")
    print(f"Decision (threshold=0.30):        {decision}")
    print(f"True Diagnosis (1=Mal, 0=Ben):    {y_test.iloc[0]}")
    print("\nTop 5 Features Driving This Prediction:")
    print(explain_patient(pipe, explainer, patient, top_n=5).to_string(index=False))

    print()
    save_global_summary(pipe, explainer, X_test)