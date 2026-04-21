import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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
        self.experiment_name = experiment_name
        self.runs = []
        # Use local JSON file instead of MLflow for cloud compatibility
        self.runs_file = "data/experiment_runs.json"
        os.makedirs("data", exist_ok=True)
        logger.info(f"Experiment tracker initialized: {experiment_name}")


    def _log_run(self, run_data: dict):
        # Load existing runs
        runs = []
        if os.path.exists(self.runs_file):
            try:
                with open(self.runs_file, "r") as f:
                    runs = json.load(f)
            except Exception:
                runs = []

        # Add new run
        runs.append(run_data)

        # Save back
        with open(self.runs_file, "w") as f:
            json.dump(runs, f, indent=2, default=str)


    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "xgboost",
        params: dict = None
    ) -> dict:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Create model
        if model_type == "xgboost":
            default_params = {
                "n_estimators":     200,
                "max_depth":        4,
                "learning_rate":    0.05,
                "subsample":        0.8,
                "random_state":     42,
                "verbosity":        0,
                "scale_pos_weight": int((y == 0).sum() / (y == 1).sum())
            }
            if params:
                default_params.update(params)
            model = XGBClassifier(**default_params)

        elif model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth":    6,
                "random_state": 42,
                "class_weight": "balanced"
            }
            if params:
                default_params.update(params)
            model = RandomForestClassifier(**default_params)

        elif model_type == "logistic_regression":
            default_params = {
                "random_state": 42,
                "max_iter":     1000,
                "class_weight": "balanced"
            }
            model = LogisticRegression(**default_params)

        # Train
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred      = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_pred_prob)
        f1       = f1_score(y_test, y_pred)

        # Log run
        run_data = {
            "run_id":     f"{model_type}_{datetime.now().strftime('%H%M%S')}",
            "model_type": model_type,
            "accuracy":   round(accuracy, 4),
            "auc":        round(auc, 4),
            "f1_score":   round(f1, 4),
            "n_samples":  len(X),
            "n_features": X.shape[1],
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._log_run(run_data)

        logger.info(
            f"Model: {model_type} | "
            f"Accuracy: {accuracy:.4f} | "
            f"AUC: {auc:.4f} | "
            f"F1: {f1:.4f}"
        )

        return {
            "model_type":    model_type,
            "run_id":        run_data["run_id"],
            "accuracy":      accuracy,
            "auc":           auc,
            "f1":            f1,
            "model":         model,
            "scaler":        scaler,
            "X_test":        X_test_scaled,
            "y_test":        y_test,
            "y_pred":        y_pred,
            "y_pred_prob":   y_pred_prob,
            "feature_names": X.columns.tolist()
        }


    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> list:
        results = []
        for model_type in ["xgboost", "random_forest", "logistic_regression"]:
            logger.info(f"Training {model_type}...")
            result = self.train(X, y, model_type=model_type)
            results.append(result)
        results.sort(key=lambda x: x["auc"], reverse=True)
        logger.info(f"Best model: {results[0]['model_type']} (AUC: {results[0]['auc']:.4f})")
        return results


    def get_experiment_runs(self) -> pd.DataFrame:
        if not os.path.exists(self.runs_file):
            return pd.DataFrame()
        try:
            with open(self.runs_file, "r") as f:
                runs = json.load(f)
            return pd.DataFrame(runs)
        except Exception:
            return pd.DataFrame()