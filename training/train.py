# MODEL TRAINING WITH MLFLOW TRACKING
# MLflow tracks every experiment automatically:
# - Parameters (hyperparameters)
# - Metrics (accuracy, AUC, F1)
# - Artifacts (model files, plots)
# - Code version (git commit)
#
# This means you can compare 100 experiments and
# always reproduce any previous result

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    f1_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self, experiment_name: str = "churn_prediction"):

        # Set MLflow tracking URI
        # mlruns/ folder stores all experiment data locally
        mlflow.set_tracking_uri("mlruns")

        # Create or get experiment
        # An experiment groups related runs together
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

        logger.info(f"MLflow experiment: {experiment_name}")


    # This METHOD trains a model and logs everything to MLflow
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "xgboost",
        params: dict = None
    ) -> dict:

        # Split data into train and test sets
        # test_size=0.2 → 20% for testing, 80% for training
        # stratify=y → keep same churn ratio in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scale features
        scaler  = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Start MLflow run
        # Everything inside this block is automatically tracked
        with mlflow.start_run(run_name=f"{model_type}_run"):

            # Log parameters
            # mlflow.log_param saves a single key-value parameter
            mlflow.log_param("model_type",  model_type)
            mlflow.log_param("n_features",  X.shape[1])
            mlflow.log_param("n_samples",   X.shape[0])
            mlflow.log_param("test_size",   0.2)

            # Create model based on type
            if model_type == "xgboost":
                default_params = {
                    "n_estimators":  200,
                    "max_depth":     4,
                    "learning_rate": 0.05,
                    "subsample":     0.8,
                    "random_state":  42,
                    "eval_metric":   "logloss",
                    "verbosity":     0
                }
                if params:
                    default_params.update(params)
                model = XGBClassifier(**default_params)
                mlflow.log_params(default_params)

            elif model_type == "random_forest":
                default_params = {
                    "n_estimators": 100,
                    "max_depth":    6,
                    "random_state": 42
                }
                if params:
                    default_params.update(params)
                model = RandomForestClassifier(**default_params)
                mlflow.log_params(default_params)

            elif model_type == "logistic_regression":
                default_params = {"random_state": 42, "max_iter": 1000}
                model = LogisticRegression(**default_params)
                mlflow.log_params(default_params)

            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred      = model.predict(X_test_scaled)
            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc      = roc_auc_score(y_test, y_pred_prob)
            f1       = f1_score(y_test, y_pred)

            # Log metrics to MLflow
            # mlflow.log_metric saves a single key-value metric
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc",      auc)
            mlflow.log_metric("f1_score", f1)

            # Log model to MLflow
            # This saves the model as an artifact
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            # Get run ID for later reference
            run_id = mlflow.active_run().info.run_id

            logger.info(
                f"Model: {model_type} | "
                f"Accuracy: {accuracy:.4f} | "
                f"AUC: {auc:.4f} | "
                f"F1: {f1:.4f}"
            )

            return {
                "model_type": model_type,
                "run_id":     run_id,
                "accuracy":   accuracy,
                "auc":        auc,
                "f1":         f1,
                "model":      model,
                "scaler":     scaler,
                "X_test":     X_test_scaled,
                "y_test":     y_test,
                "y_pred":     y_pred,
                "y_pred_prob":y_pred_prob,
                "feature_names": X.columns.tolist()
            }


    # This METHOD trains multiple models and compares them
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> list:

        results = []
        for model_type in ["xgboost", "random_forest", "logistic_regression"]:
            logger.info(f"Training {model_type}...")
            result = self.train(X, y, model_type=model_type)
            results.append(result)

        # Sort by AUC score
        results.sort(key=lambda x: x["auc"], reverse=True)
        logger.info(f"Best model: {results[0]['model_type']} (AUC: {results[0]['auc']:.4f})")
        return results


    # This METHOD returns all MLflow runs as DataFrame
    def get_experiment_runs(self) -> pd.DataFrame:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return pd.DataFrame()
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return runs