# Breast Cancer Diagnostic ML System

End-to-end machine learning classification system predicting malignant vs. benign breast tumors from Fine Needle Aspirate (FNA) cell nuclei measurements using the UCI Wisconsin Breast Cancer Diagnostic dataset (569 samples, 30 numeric features).

Built with a production-first engineering workflow: explicit clinical risk framing, leakage-free `scikit-learn` preprocessing pipelines, modular Python architecture, threshold safety tuning, model explainability, and a deployable inference service.

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

## 4. Repository Structure
```text
breast-cancer-ml-system/
├── app/                   # Inference API / UI (planned in Phase 6)
├── notebooks/
│   └── 01_eda.ipynb       # Exploratory Data Analysis & correlation heatmap
├── src/
│   ├── __init__.py
│   ├── data.py            # Target remapping & stratified train/test split
│   └── preprocessing.py   # Leakage-free ColumnTransformer blueprint
├── .gitignore
├── requirements.txt
└── README.md

---

## 6. Project Roadmap
- [x] **Phase 1:** Problem Framing & Clinical Metric Selection
- [x] **Phase 2:** Exploratory Data Analysis & Multicollinearity Analysis
- [x] **Phase 3:** Leakage-Free Preprocessing Pipeline (`data.py` & `preprocessing.py`)
- [ ] **Phase 4:** Baseline vs. Tree Ensemble Modeling (`train.py`)
- [ ] **Phase 5:** Threshold Tuning & SHAP Explainability
- [ ] **Phase 6:** Inference Service (`app/`), Unit Tests & Packaging