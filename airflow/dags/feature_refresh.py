"""
Feature Refresh DAG

Daily DAG to refresh feature store with latest login events.
Runs feature engineering pipeline and materializes to online store.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


# Default arguments for all tasks
default_args = {
    'owner': 'identity-risk-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def generate_daily_events(**context):
    """Generate synthetic events for the day (simulates real data ingestion)."""
    import pandas as pd
    from datetime import datetime
    import sys
    sys.path.insert(0, '/app')

    from src.core.data_generator import generate_login_events

    # Generate events for the execution date
    execution_date = context['execution_date']
    n_events = 1000  # Daily event volume

    df = generate_login_events(
        n_events=n_events,
        fraud_rate=0.05,  # 5% fraud rate
        n_users=100,
        n_tenants=3,
    )

    # Add timestamp for the execution date
    df['event_date'] = execution_date.strftime('%Y-%m-%d')

    # Save to staging
    output_path = f'/app/data/staging/events_{execution_date.strftime("%Y%m%d")}.parquet'
    df.to_parquet(output_path, index=False)

    print(f"Generated {len(df)} events for {execution_date}")
    return output_path


def engineer_features(**context):
    """Run feature engineering on new events."""
    import pandas as pd
    import sys
    sys.path.insert(0, '/app')

    from src.features.feature_engineering import engineer_features

    # Get the staged events
    ti = context['ti']
    input_path = ti.xcom_pull(task_ids='generate_events')

    df = pd.read_parquet(input_path)
    df_features = engineer_features(df)

    # Save engineered features
    execution_date = context['execution_date']
    output_path = f'/app/data/features/features_{execution_date.strftime("%Y%m%d")}.parquet'
    df_features.to_parquet(output_path, index=False)

    print(f"Engineered features for {len(df_features)} events")
    return output_path


def update_feature_store(**context):
    """Materialize features to Feast online store."""
    import sys
    sys.path.insert(0, '/app')

    from datetime import datetime, timedelta
    from feast import FeatureStore

    # Initialize feature store
    store = FeatureStore(repo_path='/app/feature_store')

    # Materialize features to online store
    end_date = context['execution_date']
    start_date = end_date - timedelta(days=1)

    store.materialize(
        start_date=start_date,
        end_date=end_date,
    )

    print(f"Materialized features from {start_date} to {end_date}")


def validate_features(**context):
    """Run feature quality checks."""
    import pandas as pd
    import sys
    sys.path.insert(0, '/app')

    ti = context['ti']
    input_path = ti.xcom_pull(task_ids='engineer_features')

    df = pd.read_parquet(input_path)

    # Quality checks
    checks_passed = True
    checks = []

    # Check 1: No null values in key columns
    null_counts = df[['user_id', 'risk_score', 'is_fraud']].isnull().sum()
    if null_counts.sum() > 0:
        checks.append(f"FAIL: Null values found: {null_counts.to_dict()}")
        checks_passed = False
    else:
        checks.append("PASS: No null values in key columns")

    # Check 2: Risk score in valid range
    invalid_scores = (df['risk_score'] < 0) | (df['risk_score'] > 1)
    if invalid_scores.sum() > 0:
        checks.append(f"FAIL: {invalid_scores.sum()} invalid risk scores")
        checks_passed = False
    else:
        checks.append("PASS: All risk scores in [0, 1]")

    # Check 3: Fraud rate within expected bounds
    fraud_rate = df['is_fraud'].mean()
    if fraud_rate < 0.01 or fraud_rate > 0.20:
        checks.append(f"WARN: Unusual fraud rate: {fraud_rate:.2%}")
    else:
        checks.append(f"PASS: Fraud rate within bounds: {fraud_rate:.2%}")

    # Log results
    for check in checks:
        print(check)

    if not checks_passed:
        raise ValueError("Feature quality checks failed")

    return checks_passed


# DAG definition
with DAG(
    dag_id='feature_refresh',
    default_args=default_args,
    description='Daily feature store refresh pipeline',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['features', 'ml', 'daily'],
) as dag:

    # Task 1: Generate/ingest events
    generate_events = PythonOperator(
        task_id='generate_events',
        python_callable=generate_daily_events,
        provide_context=True,
    )

    # Task 2: Engineer features
    engineer = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_features,
        provide_context=True,
    )

    # Task 3: Validate features
    validate = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        provide_context=True,
    )

    # Task 4: Update feature store
    materialize = PythonOperator(
        task_id='update_feature_store',
        python_callable=update_feature_store,
        provide_context=True,
    )

    # Task 5: Cleanup staging
    cleanup = BashOperator(
        task_id='cleanup_staging',
        bash_command='find /app/data/staging -name "*.parquet" -mtime +7 -delete || true',
    )

    # DAG flow
    generate_events >> engineer >> validate >> materialize >> cleanup
