"""
MLflow Experiment Tracking for Identity Risk Models

Provides utilities for tracking experiments, logging metrics,
and managing model versions.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np

# Default MLflow tracking URI (local)
DEFAULT_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "identity-risk-scoring"


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """
    Set up MLflow tracking.

    Args:
        tracking_uri: MLflow tracking URI (default: local mlruns/)
        experiment_name: Name of the experiment

    Returns:
        Experiment ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Use local tracking
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "identity-risk-agent", "team": "ml-platform"},
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_training_run(
    model,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    feature_columns: list[str],
    X_sample: Optional[np.ndarray] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Log a complete training run to MLflow.

    Args:
        model: Trained model (sklearn-compatible)
        metrics: Dictionary of metrics (e.g., auc, precision, recall)
        params: Dictionary of hyperparameters
        feature_columns: List of feature column names
        X_sample: Sample input for signature inference
        run_name: Optional name for the run
        tags: Optional tags for the run

    Returns:
        Run ID
    """
    if run_name is None:
        run_name = f"risk_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log feature columns as artifact
        feature_info = {
            "features": feature_columns,
            "n_features": len(feature_columns),
        }
        mlflow.log_dict(feature_info, "feature_info.json")

        # Infer model signature
        signature = None
        if X_sample is not None:
            try:
                y_pred = model.predict_proba(X_sample)[:, 1]
                signature = infer_signature(X_sample, y_pred)
            except Exception:
                pass

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name="identity-risk-model",
        )

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log additional metadata
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("timestamp", datetime.now().isoformat())

        return run.info.run_id


def log_evaluation_metrics(
    run_id: str,
    metrics: Dict[str, float],
    dataset_name: str = "test",
) -> None:
    """
    Log additional evaluation metrics to an existing run.

    Args:
        run_id: MLflow run ID
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset (e.g., "test", "validation")
    """
    with mlflow.start_run(run_id=run_id):
        prefixed_metrics = {f"{dataset_name}_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)


def log_confusion_matrix(
    run_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] = None,
) -> None:
    """
    Log confusion matrix as an artifact.

    Args:
        run_id: MLflow run ID
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=labels or ["Normal", "Fraud"],
        columns=labels or ["Pred Normal", "Pred Fraud"],
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(cm_df.to_dict(), "confusion_matrix.json")


def get_best_model(
    experiment_name: str = EXPERIMENT_NAME,
    metric: str = "auc",
    ascending: bool = False,
) -> Dict[str, Any]:
    """
    Get the best model from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        ascending: Sort ascending (True) or descending (False)

    Returns:
        Dictionary with run info and model URI
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs.iloc[0]
    return {
        "run_id": best_run["run_id"],
        "metrics": {
            k.replace("metrics.", ""): v
            for k, v in best_run.items()
            if k.startswith("metrics.")
        },
        "params": {
            k.replace("params.", ""): v
            for k, v in best_run.items()
            if k.startswith("params.")
        },
        "model_uri": f"runs:/{best_run['run_id']}/model",
    }


def load_model_from_run(run_id: str):
    """
    Load a model from an MLflow run.

    Args:
        run_id: MLflow run ID

    Returns:
        Loaded model
    """
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def load_production_model(model_name: str = "identity-risk-model"):
    """
    Load the production model from the model registry.

    Args:
        model_name: Registered model name

    Returns:
        Loaded model
    """
    model_uri = f"models:/{model_name}/Production"
    return mlflow.sklearn.load_model(model_uri)


class MLflowExperimentTracker:
    """
    Context manager for MLflow experiment tracking.

    Usage:
        with MLflowExperimentTracker("my_run") as tracker:
            tracker.log_param("learning_rate", 0.01)
            # ... train model ...
            tracker.log_metric("accuracy", 0.95)
            tracker.log_model(model)
    """

    def __init__(
        self,
        run_name: str,
        experiment_name: str = EXPERIMENT_NAME,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.run = None

    def __enter__(self):
        setup_mlflow(experiment_name=self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        return False

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model, artifact_path: str = "model", **kwargs) -> None:
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    @property
    def run_id(self) -> str:
        return self.run.info.run_id


if __name__ == "__main__":
    # Quick test
    print("Setting up MLflow...")
    experiment_id = setup_mlflow()
    print(f"Experiment ID: {experiment_id}")

    # Test logging
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)

    with MLflowExperimentTracker("test_run", tags={"test": "true"}) as tracker:
        tracker.log_params({"n_estimators": 10})
        tracker.log_metrics({"accuracy": 0.95, "auc": 0.92})
        tracker.log_model(model)
        print(f"Run ID: {tracker.run_id}")

    print("MLflow test complete! Run `mlflow ui` to view results.")
