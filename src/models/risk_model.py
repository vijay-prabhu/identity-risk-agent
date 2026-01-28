"""
Risk Scoring Model Training

Trains and evaluates models for identity risk scoring.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Feature columns used for modeling
FEATURE_COLUMNS = [
    "failed_logins_24h",
    "login_count_7d",
    "device_age_days",
    "is_new_device",
    "ip_reputation_score",
    "hour_of_day",
    "is_unusual_hour",
    "location_changed",
    "mfa_used",
    "vpn_detected",
    "success",
]

TARGET_COLUMN = "is_fraudulent"


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data for model training.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Select features that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    X = df[available_features].values
    y = df[target_col].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train set: {len(X_train)} samples ({y_train.mean()*100:.1f}% fraud)")
    print(f"Test set: {len(X_test)} samples ({y_test.mean()*100:.1f}% fraud)")

    return X_train, X_test, y_train, y_test, scaler


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest for unsupervised anomaly detection.

    Args:
        X_train: Training features
        contamination: Expected proportion of anomalies
        random_state: Random seed

    Returns:
        Trained IsolationForest model
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples="auto",
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train Random Forest classifier for supervised fraud detection.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",  # Handle class imbalance
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train Logistic Regression for interpretable baseline.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model with predict/predict_proba methods
        X_test: Test features
        y_test: Test labels
        model_name: Name for display

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print("=" * 50)

    # Handle IsolationForest differently
    if isinstance(model, IsolationForest):
        # IsolationForest returns -1 for anomalies, 1 for normal
        predictions = model.predict(X_test)
        y_pred = (predictions == -1).astype(int)  # Convert to 0/1
        y_proba = -model.score_samples(X_test)  # Higher score = more anomalous
        # Normalize to 0-1 range
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.5

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    # Print results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """
    Perform cross-validation on a model.

    Args:
        model: Sklearn-compatible model
        X: Features
        y: Labels
        cv: Number of folds

    Returns:
        Dictionary with CV scores
    """
    if isinstance(model, IsolationForest):
        print("Skipping CV for IsolationForest (unsupervised)")
        return {"cv_mean": 0, "cv_std": 0}

    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\n{cv}-Fold Cross-Validation ROC-AUC:")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    return {"cv_mean": scores.mean(), "cv_std": scores.std()}


class RiskScorer:
    """
    Combined risk scoring model for production use.

    Uses ensemble of models to produce a risk score.
    """

    def __init__(
        self,
        supervised_model: RandomForestClassifier,
        scaler: StandardScaler,
        feature_columns: list[str],
    ):
        self.supervised_model = supervised_model
        self.scaler = scaler
        self.feature_columns = feature_columns

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability of fraud."""
        X_scaled = self.scaler.transform(X)
        return self.supervised_model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary prediction."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, features: dict) -> dict:
        """
        Score a single login event.

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with risk_score and risk_level
        """
        # Create feature vector
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])

        # Get probability
        proba = self.predict_proba(X)[0]

        # Determine risk level
        if proba < 0.3:
            level = "low"
        elif proba < 0.6:
            level = "medium"
        elif proba < 0.8:
            level = "high"
        else:
            level = "critical"

        return {
            "risk_score": float(proba),
            "risk_level": level,
        }

    def save(self, path: str):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RiskScorer":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


def train_and_evaluate(
    df: pd.DataFrame,
    output_path: Optional[str] = "models/risk_model.pkl",
) -> Tuple[RiskScorer, dict]:
    """
    Full training pipeline: prepare data, train models, evaluate, save.

    Args:
        df: DataFrame with features and target
        output_path: Path to save model (None to skip)

    Returns:
        Trained RiskScorer and metrics dictionary
    """
    print("=" * 60)
    print("IDENTITY RISK MODEL TRAINING")
    print("=" * 60)

    # Prepare data
    print("\n1. Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Train models
    print("\n2. Training models...")

    print("\n  Training Isolation Forest (unsupervised)...")
    iso_model = train_isolation_forest(X_train)

    print("\n  Training Random Forest (supervised)...")
    rf_model = train_random_forest(X_train, y_train)

    print("\n  Training Logistic Regression (baseline)...")
    lr_model = train_logistic_regression(X_train, y_train)

    # Evaluate models
    print("\n3. Evaluating models...")

    iso_metrics = evaluate_model(iso_model, X_test, y_test, "Isolation Forest")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # Cross-validation for best model
    print("\n4. Cross-validation...")
    X_all = scaler.fit_transform(df[FEATURE_COLUMNS].values)
    y_all = df[TARGET_COLUMN].values
    cv_metrics = cross_validate_model(
        RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1
        ),
        X_all,
        y_all,
    )

    # Select best model (Random Forest typically performs best)
    best_model = rf_model
    best_metrics = rf_metrics

    # Feature importance
    print("\n5. Feature Importance (Random Forest):")
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(importance.to_string(index=False))

    # Create production scorer
    print("\n6. Creating production scorer...")
    scorer = RiskScorer(
        supervised_model=best_model,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
    )

    # Save model
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scorer.save(output_path)

    # Summary
    all_metrics = {
        "isolation_forest": iso_metrics,
        "random_forest": rf_metrics,
        "logistic_regression": lr_metrics,
        "cross_validation": cv_metrics,
        "best_model": "random_forest",
        "best_auc": best_metrics["auc"],
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Model: Random Forest")
    print(f"Test ROC-AUC: {best_metrics['auc']:.4f}")
    print(f"CV ROC-AUC: {cv_metrics['cv_mean']:.4f} (+/- {cv_metrics['cv_std']*2:.4f})")

    return scorer, all_metrics


if __name__ == "__main__":
    # Load features
    df = pd.read_parquet("data/features.parquet")
    print(f"Loaded {len(df)} samples")

    # Train and evaluate
    scorer, metrics = train_and_evaluate(df)

    # Test scorer
    print("\n\nTest Scoring:")
    sample_features = {
        "failed_logins_24h": 2,
        "login_count_7d": 5,
        "device_age_days": 0,
        "is_new_device": 1,
        "ip_reputation_score": 0.8,
        "hour_of_day": 3,
        "is_unusual_hour": 1,
        "location_changed": 1,
        "mfa_used": 0,
        "vpn_detected": 1,
        "success": 0,
    }
    result = scorer.score(sample_features)
    print(f"Suspicious login score: {result}")
