# 📊 PredictaBoard — Universal Binary Classification Dashboard

> **Live Demo:** [predictaboard.streamlit.app](https://predictaboard.streamlit.app)

A universal machine learning dashboard that accepts **any CSV file** and automatically trains and evaluates classification models on it — no code changes required per dataset.

---

## 🚀 What It Does

Most ML dashboards are hardcoded to one dataset. PredictaBoard solves this by:

- **Auto-detecting** column types (numeric vs categorical)
- **Dynamically fitting** a preprocessing pipeline on any uploaded CSV
- **Training 3 models** simultaneously — Random Forest, Logistic Regression, XGBoost
- **Evaluating** with confusion matrix, ROC curve, PR curve, and 5-fold cross-validation
- **Explaining** predictions using SHAP values
- **Deploying** on Streamlit Cloud with zero infrastructure setup

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **Universal CSV support** | Upload any binary classification dataset — zero code changes |
| **Auto preprocessing** | Detects numeric/categorical columns, handles missing values, scales and encodes automatically |
| **3 ML models** | Random Forest, Logistic Regression, XGBoost — switchable from sidebar |
| **Class imbalance handling** | Auto-detects imbalance and applies `class_weight='balanced'` / `scale_pos_weight` |
| **Full evaluation suite** | Confusion matrix, ROC curve, Precision-Recall curve, 5-fold CV AUC |
| **SHAP explainability** | Mean absolute SHAP bar chart + beeswarm plot for all 3 models |
| **Risk segmentation** | Low / Medium / High risk buckets with adjustable threshold slider |
| **Download predictions** | Export scored CSV with probabilities and risk labels |
| **Auto-drop ID columns** | Detects and removes columns where every value is unique |
| **Target validation** | Blocks non-binary columns with clear error messages |

---

## 🗂 Dashboard Tabs

1. **📋 Data Preview** — Raw data, missing value heatmap, column types and null counts
2. **🗂 Overview** — Class distribution, numeric distributions, correlation heatmap
3. **🔮 Predictions** — Risk buckets, probability histogram, threshold slider, downloadable CSV
4. **📈 Evaluation** — CV scores, confusion matrix, ROC curve, PR curve, model comparison
5. **💡 Insights** — Feature importances, numeric correlates, categorical breakdowns
6. **🧠 Explainability (SHAP)** — Mean SHAP impact chart, beeswarm plot, downloadable SHAP values

---

## 🛠 Tech Stack

```
Python          — Core language
Scikit-learn    — Preprocessing, Random Forest, Logistic Regression
XGBoost         — Gradient boosting model
SHAP            — Model explainability
Streamlit       — Web dashboard framework
Plotly          — Interactive visualizations
Pandas / NumPy  — Data manipulation
Streamlit Cloud — Free deployment
```

---

## 📁 Project Structure

```
PredictaBoard/
├── app.py                  # Entry point — main() and tab routing (~150 lines)
├── requirements.txt        # All dependencies
├── README.md
├── data/
│   ├── telecom_churn.csv   # Default bundled dataset
│   └── feature_names.json
└── src/
    ├── pipeline.py         # Dynamic preprocessing + model training
    ├── charts.py           # Data preview + overview tab
    ├── predictions.py      # Predictions tab
    ├── evaluation.py       # Confusion matrix, ROC, CV scores
    ├── insights.py         # Feature importance + breakdowns
    ├── explainability.py   # SHAP tab
    └── utils.py            # Shared constants and helpers
```

---

## ✅ Validated Datasets

Tested across 3 completely different domains without any code changes:

| Dataset | Domain | Rows | Target | AUC |
|---|---|---|---|---|
| Telecom Churn | Telecom | 10,000 | `churn` (0/1) | 0.87 |
| IBM HR Attrition | HR | 1,470 | `Attrition` (Yes/No) | 0.85 |
| Heart Disease UCI | Healthcare | 303 | `target` (0/1) | 0.91 |

---

## 🏃 Run Locally

```bash
# Clone the repo
git clone https://github.com/adithakur22/predictaboard.git
cd predictaboard

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 💡 How It Works

```
Upload CSV
    ↓
Auto-detect column types (numeric / categorical)
    ↓
Fit ColumnTransformer (StandardScaler + OneHotEncoder)
    ↓
Train RF + LR + XGBoost simultaneously
    ↓
5-fold cross-validation AUC
    ↓
Score all rows → risk buckets (Low / Medium / High)
    ↓
SHAP explainability → feature impact per prediction
```

---

## 📊 Model Comparison

| Model | Strength | Best for |
|---|---|---|
| Random Forest | Handles non-linear patterns, robust | General purpose |
| Logistic Regression | Fast, interpretable, good baseline | Linear relationships |
| XGBoost | Usually highest AUC, handles imbalance well | Best performance |

---

## 👨‍💻 Author

**Aditya Thakur**
B.E. Computer Science & Engineering (AI/ML) — Chandigarh University

[![GitHub](https://img.shields.io/badge/GitHub-adithakur22-black?logo=github)](https://github.com/adithakur22)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-aditya--thakur-blue?logo=linkedin)](https://linkedin.com/in/aditya-thakur)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-predictaboard.streamlit.app-red?logo=streamlit)](https://predictaboard.streamlit.app)

---
