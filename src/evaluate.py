import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from src.data import load_data, split_data
from src.preprocessing import build_preprocessor
import joblib

def evaluate_thresholds(y_true, y_proba, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        rows.append({"threshold": t, "recall": recall_score(y_true, y_pred), "precision": precision_score(y_true, y_pred), "f1": f1_score(y_true, y_pred), "fn_missed": fn, "fp_false_alarm": fp})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = Pipeline(steps=[("preprocessor", build_preprocessor(X_train.columns.tolist())), ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_cv_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    print(evaluate_thresholds(y_train, train_cv_proba, [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]))

def evaluate_test_at_threshold(X_test, y_test, threshold=0.30, model_path="models/model.joblib"):
    pipe = joblib.load(model_path)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

if __name__ =="__main__":
    print("\n--- Final Test Evaluation at Threshold = 0.30 ---")
    evaluate_test_at_threshold(X_test, y_test, threshold=0.30)