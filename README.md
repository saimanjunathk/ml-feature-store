# ML Feature Store + Model Serving Platform

Production-grade ML platform with feature store, model registry, A/B testing and drift monitoring.

## Architecture
Raw Data -> Feature Store -> Model Training -> MLflow Registry -> FastAPI Serving -> Drift Monitoring

## Tech Stack
- Feature Store: Feast / Custom
- Training: Scikit-learn, XGBoost, PyTorch
- Registry: MLflow
- Serving: FastAPI
- Monitoring: Evidently
- Infra: Docker, Kubernetes

## Status
In Progress
