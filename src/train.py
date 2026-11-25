from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import load_data, split_data
from src.preprocessing import build_preprocessor


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["recall", "precision", "f1", "roc_auc"]
    results = []

    for name, model in get_models().items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train.columns.tolist())),
                ("classifier", model),
            ]
        )
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        results.append(
            {
                "model": name,
                "recall": scores["test_recall"].mean(),
                "precision": scores["test_precision"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
            }
        )

    return pd.DataFrame(results)


def tune_logistic_regression(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["recall", "precision", "f1", "roc_auc"]
    results = []

    for c_value in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=42)
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train.columns.tolist())),
                ("classifier", model),
            ]
        )
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        results.append(
            {
                "C": c_value,
                "recall": scores["test_recall"].mean(),
                "precision": scores["test_precision"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
            }
        )

    return pd.DataFrame(results)


def train_and_save_final_model(
    X_train, X_test, y_train, y_test, model_path="models/model.joblib"
):
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train.columns.tolist())),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"\nSaved final pipeline to {model_path}")

    return pipe


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("--- Model Comparison (5-Fold CV on Train) ---")
    print(evaluate_models(X_train, y_train))

    print("\n--- Logistic Regression Tuning (5-Fold CV on Train) ---")
    print(tune_logistic_regression(X_train, y_train))

    print("\n--- Final Test Evaluation (Unlocked Once) ---")
    train_and_save_final_model(X_train, X_test, y_train, y_test)