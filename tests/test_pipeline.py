from src.data import load_data, split_data
from src.predict import load_inference_artifacts, predict_patient
from src.preprocessing import build_preprocessor


def test_load_data_shape_and_target_mapping():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert len(y) == 569
    assert int(y.sum()) == 212  # 1 = Malignant (212 cases)


def test_preprocessor_handles_missing_values_without_leakage():
    X, y = load_data()
    X_train, X_test, _, _ = split_data(X, y)

    preprocessor = build_preprocessor(X_train.columns.tolist())
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Simulate a clinic input with a missing value
    patient_with_nan = X_test.iloc[[0]].copy()
    patient_with_nan.iloc[0, 0] = None
    patient_scaled = preprocessor.transform(patient_with_nan)

    assert X_train_scaled.shape == (455, 30)
    assert patient_scaled.shape == (1, 30)
    assert not any(val != val for val in patient_scaled[0])  # No NaNs remain


def test_predict_patient_output_contract():
    pipe, explainer = load_inference_artifacts()
    X, y = load_data()
    _, X_test, _, _ = split_data(X, y)

    result = predict_patient(X_test.iloc[[0]], pipe, explainer)

    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Benign", "Malignant")
    assert 0.0 <= result["malignant_probability"] <= 1.0
    assert len(result["top_factors"]) == 5