# Customer Churn Prediction Dashboard

This project builds a synthetic telecom-style churn dataset, stores it in normalized SQLite tables, trains two scikit-learn classifiers (logistic regression and random forest), and serves an interactive **Streamlit** dashboard for exploration and batch scoring.

## Tech stack

Python, pandas, NumPy, scikit-learn, SQLite, Matplotlib, Seaborn, Plotly, Streamlit.

## Setup

From the `churn-prediction` directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run scripts in order

1. **Generate data** — creates `data/telecom_churn.csv` (10k rows, ~27% churn):

   ```bash
   python data/generate_data.py
   ```

2. **Load SQLite** — applies `db/schema.sql` and loads the CSV into `db/churn.db`:

   ```bash
   python db/load_db.py
   ```

3. **Preprocess** — imputation, one-hot encoding, scaling, train/test split, saves `.npy` files, `models/scaler.pkl` (numeric `StandardScaler` only), and `models/preprocessor.pkl` (full fitted pipeline used by `predict.py` and the dashboard):

   ```bash
   python src/preprocess.py
   ```

4. **Train** — fits logistic regression and random forest, prints metrics, saves `models/logistic_model.pkl` and `models/rf_model.pkl`:

   ```bash
   python src/train.py
   ```

5. **Optional** — exploratory notebook: open `notebooks/eda.ipynb` in Jupyter or VS Code.

6. **Dashboard**:

   ```bash
   streamlit run app.py
   ```

Use the sidebar to pick a model and optionally upload a CSV with the same feature columns as `telecom_churn.csv` (including `customer_id`; include `churn` if you want batch AUC-ROC on the Prediction tab).

## Project layout

- `data/generate_data.py` — synthetic dataset
- `db/schema.sql`, `db/load_db.py` — SQLite schema and loader
- `src/preprocess.py`, `src/train.py`, `src/predict.py` — ML pipeline
- `app.py` — Streamlit UI

## Reproducibility

`random_state=42` is used for splits and models where applicable.
