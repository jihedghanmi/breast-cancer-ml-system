# Breast Cancer Diagnostic ML System

End-to-end machine learning classification system predicting malignant vs. benign breast tumors from Fine Needle Aspirate (FNA) cell nuclei measurements (UCI Wisconsin Breast Cancer Diagnostic dataset, 569 samples, 30 features).

Built with a production-first mindset: leakage-free `scikit-learn` pipelines, clinical safety threshold optimization, SHAP explainability, modular Python code, and a deployable inference API.

## Problem Framing & Clinical Metrics
* **Target Definition:** `1 = Malignant` (Positive class), `0 = Benign` (Negative class). *(Note: `scikit-learn` defaults are reversed, so labels are explicitly remapped.)*
* **Primary Metric — Recall:** Missing a malignant tumor (False Negative) delays critical treatment, so maximizing Recall is the top priority.
* **Guardrail Metric — Precision:** Prevents excessive false alarms (False Positives) and unnecessary invasive follow-up procedures.

## Project Structure
* `notebooks/` — Exploratory Data Analysis (`01_eda.ipynb`)
* `src/` — Modular preprocessing, training, and inference pipelines *(in progress)*
* `app/` — Inference API / UI *(planned)*

## Roadmap
- [x] **Phase 1:** Problem Framing & Clinical Metric Selection
- [x] **Phase 2:** Exploratory Data Analysis & Multicollinearity Analysis
- [ ] **Phase 3:** Leakage-Free Preprocessing Pipeline
- [ ] **Phase 4:** Baseline vs. Tree Ensemble Modeling
- [ ] **Phase 5:** Threshold Tuning & SHAP Explainability
- [ ] **Phase 6:** Modular Refactoring, Inference App & Packaging