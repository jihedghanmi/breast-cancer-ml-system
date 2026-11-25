# Breast Cancer Diagnostic ML System

End-to-end machine learning classification system predicting malignant vs. benign breast tumors from Fine Needle Aspirate (FNA) cell nuclei measurements using the UCI Wisconsin Breast Cancer Diagnostic dataset (569 samples, 30 numeric features).

Built with a production-first engineering workflow: explicit clinical risk framing, leakage-free `scikit-learn` preprocessing pipelines, cross-validated model selection, threshold safety tuning, model explainability, and a deployable inference service.

---

## 1. Problem Framing & Clinical Metric Strategy
In clinical decision support, classification errors carry asymmetric real-world costs:
* **Target Definition:** `1 = Malignant` (Positive class), `0 = Benign` (Negative class).
  * *Engineering Catch:* `scikit-learn`'s `load_breast_cancer()` defaults to `0 = malignant, 1 = benign`. Labels are explicitly remapped in `src/data.py` so evaluation metrics align with clinical detection.
* **Primary Metric — Recall:** A False Negative misses a malignant tumor and delays life-saving treatment, making Recall our top priority.
* **Guardrail Metric — Precision:** Prevents the model from trivially maximizing Recall by flagging healthy patients, controlling False Positives and unnecessary invasive follow-up tests.

---

## 2. Key EDA Findings (`notebooks/01_eda.ipynb`)
* **Dataset Integrity:** 569 samples, 30 real-valued cell nuclei measurements (`mean`, `error`, and `worst` statistics across 10 properties), 0 missing values.
* **Class Balance:** Moderately imbalanced — **37.3% Malignant (212)** vs. **62.7% Benign (357)**.
* **Feature Scale Disparity:** Features span vastly different magnitudes (e.g., `mean area` max `2501.0` vs. `mean smoothness` max `0.1634`), requiring standardization for linear models.
* **Multicollinearity:** Geometric size features (`radius`, `perimeter`, `area`) show ~`0.99` correlation. All 30 features are retained for baseline benchmarking before testing any feature reduction.

---

## 3. Leakage-Free Data & Preprocessing Architecture
To prevent data leakage during both training and live inference:
1. **Stratified Split First (`src/data.py`):** Splits data 80/20 (`455` train / `114` test) using `stratify=y` and `random_state=42`, preserving the ~37% malignant ratio across both sets (`0.3736` train vs. `0.3684` test).
2. **Modular Preprocessing Blueprint (`src/preprocessing.py`):** Builds an unfitted `ColumnTransformer` + `Pipeline` with:
   * `SimpleImputer(strategy="median")` — defensive production guard so unseen clinic inputs with missing values do not crash inference.
   * `StandardScaler()` — standardizes features using parameters learned **only** from `X_train` (`X_train_scaled` mean: `0.000000`, `X_test_scaled` mean: `-0.024009`).

---

## 4. Modeling & Evaluation Results (`src/train.py`)

### 5-Fold Stratified Cross-Validation Benchmark (`X_train` only)
| Model | Recall | Precision | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (`C=1.0`)** | **0.9529** | **0.9771** | **0.9640** | **0.9958** |
| Gradient Boosting | 0.9471 | 0.9646 | 0.9553 | 0.9908 |
| Random Forest (`n=200`) | 0.9353 | 0.9638 | 0.9489 | 0.9889 |

* **Key Insight:** On 455 clean, continuous numeric samples, regularized `LogisticRegression` outperformed tree ensembles across every metric.
* **Hyperparameter Tuning (`C` in `[0.01, 0.1, 1.0, 10.0, 100.0]`):** `C=0.01` underfit (`0.8647` Recall), `C=100.0` showed overfitting (`0.9485` Precision, `0.9914` ROC-AUC), and **`C=1.0`** achieved the strongest overall balance (`0.9529` Recall, `0.9640` F1, `0.9958` ROC-AUC).

### Final Held-Out Test Evaluation (`X_test`, 114 unseen patients, default `0.50` threshold)
* **Recall (Malignant):** `0.9286` (`39 / 42` malignant tumors caught; **`3` False Negatives**)
* **Precision (Malignant):** `0.9750` (`39 / 40` flagged cases malignant; **`1` False Positive**)
* **F1-Score (Malignant):** `0.9512` | **Accuracy:** `0.9649` | **ROC-AUC:** `0.9960`
* **Artifact Saved:** Full fitted `Pipeline` (imputer + scaler + classifier) exported to `models/model.joblib`.

---

## 5. Repository Structure
```text
breast-cancer-ml-system/
├── app/                   # Inference API / UI (planned in Phase 6)
├── models/
│   └── model.joblib       # Trained leakage-free sklearn Pipeline
├── notebooks/
│   └── 01_eda.ipynb       # Exploratory Data Analysis & correlation heatmap
├── src/
│   ├── __init__.py
│   ├── data.py            # Target remapping & stratified train/test split
│   ├── preprocessing.py   # Leakage-free ColumnTransformer blueprint
│   └── train.py           # CV benchmarking, tuning, test evaluation & export
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. How to Run Locally

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify data loading & stratified split:**
   ```bash
   python -m src.data
   ```
4. **Verify leakage-free preprocessing pipeline:**
   ```bash
   python -m src.preprocessing
   ```
5. **Run model benchmarking, tuning, evaluation & save pipeline:**
   ```bash
   python -m src.train
   ```

---

## 7. Project Roadmap
- [x] **Phase 1:** Problem Framing & Clinical Metric Selection
- [x] **Phase 2:** Exploratory Data Analysis & Multicollinearity Analysis
- [x] **Phase 3:** Leakage-Free Preprocessing Pipeline (`data.py` & `preprocessing.py`)
- [x] **Phase 4:** Baseline vs. Tree Ensemble Modeling (`train.py`)
- [ ] **Phase 5:** Threshold Tuning & SHAP Explainability
- [ ] **Phase 6:** Inference Service (`app/`), Unit Tests & Packaging