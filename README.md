# ML Feature Store + Model Serving Platform

## Live Demo
View Live Dashboard: https://ml-feature-store.streamlit.app/

## Architecture
Raw Data -> Feature Engineering -> Offline Feature Store -> Model Training -> Model Comparison -> Drift Monitoring

## Features
- Feature Store: Compute and store ML features for 1000-2000 customers
- Model Training: XGBoost, Random Forest, Logistic Regression
- Experiment Tracking: Custom JSON-based experiment logger (MLflow-style)
- Model Comparison: AUC, Accuracy, F1 Score, ROC curves
- Drift Monitoring: PSI and KS test based drift detection

## Tech Stack
- Feature Store: Custom SQLite-based offline store
- Training: Scikit-learn, XGBoost
- Experiment Tracking: MLflow (local) / Custom JSON (cloud)
- Monitoring: PSI + KS statistical tests
- Dashboard: Streamlit + Plotly

## How to Run Locally
git clone https://github.com/saimanjunathk/ml-feature-store
cd ml-feature-store
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py

## Status
- Feature Store - Done
- Model Training - Done
- Experiment Tracking - Done
- Model Comparison - Done
- Drift Monitoring - Done
- Live Dashboard - Done