"""
Model Retraining DAG

Weekly DAG to retrain the risk scoring model with latest data.
Includes quality gates to ensure new model meets performance thresholds.
"""

from datetime import datetime, timedelta

from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from airflow import DAG

# Default arguments
default_args = {
    'owner': 'identity-risk-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# Quality thresholds
AUC_THRESHOLD = 0.85
MIN_TRAINING_SAMPLES = 5000


def collect_training_data(**context):
    """Collect recent feature data for training."""
    import sys
    from pathlib import Path

    import pandas as pd
    sys.path.insert(0, '/app')

    # Collect last 30 days of feature data
    feature_dir = Path('/app/data/features')
    feature_files = sorted(feature_dir.glob('features_*.parquet'))[-30:]

    if not feature_files:
        # Fall back to main features file
        df = pd.read_parquet('/app/data/features.parquet')
    else:
        dfs = [pd.read_parquet(f) for f in feature_files]
        df = pd.concat(dfs, ignore_index=True)

    print(f"Collected {len(df)} samples from {len(feature_files)} files")

    # Save combined training data
    output_path = '/app/data/training/train_data.parquet'
    Path('/app/data/training').mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return {
        'path': output_path,
        'n_samples': len(df),
        'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0.0,
    }


def check_data_quality(**context):
    """Check if we have enough quality data for training."""
    ti = context['ti']
    data_info = ti.xcom_pull(task_ids='collect_data')

    n_samples = data_info['n_samples']
    fraud_rate = data_info['fraud_rate']

    print(f"Training data: {n_samples} samples, {fraud_rate:.2%} fraud rate")

    # Check minimum samples
    if n_samples < MIN_TRAINING_SAMPLES:
        print(f"FAIL: Insufficient samples ({n_samples} < {MIN_TRAINING_SAMPLES})")
        return 'skip_training'

    # Check fraud rate bounds
    if fraud_rate < 0.01 or fraud_rate > 0.30:
        print(f"WARN: Unusual fraud rate: {fraud_rate:.2%}")

    return 'train_model'


def train_model(**context):
    """Train new risk scoring model."""
    import sys

    import mlflow
    import pandas as pd
    sys.path.insert(0, '/app')

    from src.features.feature_engineering import engineer_features
    from src.models.risk_model import train_and_evaluate

    ti = context['ti']
    data_info = ti.xcom_pull(task_ids='collect_data')

    # Load training data
    df = pd.read_parquet(data_info['path'])

    # Ensure features are engineered
    if 'ip_reputation_score' not in df.columns:
        df = engineer_features(df)

    # Train model with MLflow tracking
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment('risk_model_retrain')

    with mlflow.start_run(run_name=f"retrain_{context['execution_date'].strftime('%Y%m%d')}"):
        results = train_and_evaluate(df)

        # Log metrics
        mlflow.log_metric('auc', results['auc'])
        mlflow.log_metric('accuracy', results['accuracy'])
        mlflow.log_metric('precision', results['precision'])
        mlflow.log_metric('recall', results['recall'])
        mlflow.log_metric('f1', results['f1'])
        mlflow.log_metric('n_samples', len(df))

        # Log model
        mlflow.sklearn.log_model(
            results['scorer'].model,
            'model',
            registered_model_name='risk_scorer',
        )

        # Save model locally for evaluation
        model_path = '/app/models/candidate_model.pkl'
        results['scorer'].save(model_path)

        run_id = mlflow.active_run().info.run_id

    print(f"Model trained - AUC: {results['auc']:.4f}, Run ID: {run_id}")

    return {
        'auc': results['auc'],
        'accuracy': results['accuracy'],
        'model_path': model_path,
        'run_id': run_id,
    }


def evaluate_model(**context):
    """Evaluate model against quality gates."""
    ti = context['ti']
    train_results = ti.xcom_pull(task_ids='train_model')

    auc = train_results['auc']

    print(f"Model AUC: {auc:.4f}")
    print(f"Threshold: {AUC_THRESHOLD}")

    if auc >= AUC_THRESHOLD:
        print("PASS: Model meets quality threshold")
        return 'promote_model'
    else:
        print(f"FAIL: Model AUC {auc:.4f} below threshold {AUC_THRESHOLD}")
        return 'reject_model'


def promote_model(**context):
    """Promote candidate model to production."""
    import shutil
    import sys
    from pathlib import Path

    import mlflow
    sys.path.insert(0, '/app')

    ti = context['ti']
    train_results = ti.xcom_pull(task_ids='train_model')

    # Copy candidate to production
    src_path = train_results['model_path']
    dst_path = '/app/models/risk_model.pkl'

    # Backup current model
    backup_path = f"/app/models/backup/risk_model_{context['execution_date'].strftime('%Y%m%d')}.pkl"
    Path('/app/models/backup').mkdir(parents=True, exist_ok=True)

    if Path(dst_path).exists():
        shutil.copy(dst_path, backup_path)
        print(f"Backed up current model to {backup_path}")

    # Promote new model
    shutil.copy(src_path, dst_path)
    print("Promoted new model to production")

    # Transition model stage in MLflow
    mlflow.set_tracking_uri('http://mlflow:5000')
    client = mlflow.tracking.MlflowClient()

    try:
        # Get latest version
        versions = client.search_model_versions("name='risk_scorer'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                name='risk_scorer',
                version=latest.version,
                stage='Production',
            )
            print(f"Transitioned model version {latest.version} to Production")
    except Exception as e:
        print(f"Could not update MLflow model stage: {e}")

    return train_results['auc']


def reject_model(**context):
    """Handle rejected model."""
    ti = context['ti']
    train_results = ti.xcom_pull(task_ids='train_model')

    print(f"Model rejected - AUC: {train_results['auc']:.4f}")
    print("Current production model will continue to be used")

    # Could add alerting here
    return False


def notify_completion(**context):
    """Send completion notification."""
    ti = context['ti']

    # Check which branch was taken
    promote_result = ti.xcom_pull(task_ids='promote_model')
    ti.xcom_pull(task_ids='reject_model')

    if promote_result:
        print(f"Model retrain completed - New model deployed with AUC: {promote_result:.4f}")
    else:
        print("Model retrain completed - New model did not meet quality threshold")


# DAG definition
with DAG(
    dag_id='model_retrain',
    default_args=default_args,
    description='Weekly model retraining with quality gates',
    schedule_interval='0 4 * * 0',  # Run at 4 AM every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'weekly'],
) as dag:

    # Task 1: Collect training data
    collect_data = PythonOperator(
        task_id='collect_data',
        python_callable=collect_training_data,
        provide_context=True,
    )

    # Task 2: Check data quality (branching)
    check_quality = BranchPythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        provide_context=True,
    )

    # Task 3a: Skip training
    skip_training = EmptyOperator(
        task_id='skip_training',
    )

    # Task 3b: Train model
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )

    # Task 4: Evaluate model (branching)
    evaluate = BranchPythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )

    # Task 5a: Promote model
    promote = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model,
        provide_context=True,
    )

    # Task 5b: Reject model
    reject = PythonOperator(
        task_id='reject_model',
        python_callable=reject_model,
        provide_context=True,
    )

    # Task 6: Join and notify
    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
        trigger_rule='none_failed_min_one_success',
    )

    # DAG flow
    collect_data >> check_quality

    check_quality >> skip_training >> notify
    check_quality >> train >> evaluate

    evaluate >> promote >> notify
    evaluate >> reject >> notify
