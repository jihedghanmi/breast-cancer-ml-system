import joblib

from src.data import load_data, split_data
from src.explain import create_explainer, explain_patient

DEFAULT_THRESHOLD = 0.30


def load_inference_artifacts(model_path="models/model.joblib"):
    pipe = joblib.load(model_path)
    X, y = load_data()
    X_train, _, _, _ = split_data(X, y)
    explainer = create_explainer(pipe, X_train)
    return pipe, explainer


def predict_patient(
    patient_df, pipe, explainer, threshold=DEFAULT_THRESHOLD, top_n=5
):
    prob = float(pipe.predict_proba(patient_df)[0, 1])
    prediction = int(prob >= threshold)
    label = "Malignant" if prediction == 1 else "Benign"
    top_factors = explain_patient(pipe, explainer, patient_df, top_n=top_n)

    return {
        "prediction": prediction,
        "label": label,
        "malignant_probability": prob,
        "threshold": threshold,
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    pipe, explainer = load_inference_artifacts()
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)

    malignant_patient = X_test[y_test == 1].iloc[[0]]
    result = predict_patient(malignant_patient, pipe, explainer)

    print(f"Predicted Label:       {result['label']}")
    print(f"Malignant Probability: {result['malignant_probability']:.4f}")
    print(f"Decision Threshold:    {result['threshold']}")
    print("\nTop 5 Risk Drivers (SHAP):")
    print(result["top_factors"].to_string(index=False))