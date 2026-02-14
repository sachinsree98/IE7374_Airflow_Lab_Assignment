"""
ML Training Pipeline DAG
-------------------------
Periodically trains a Random Forest Regressor on the
California Housing dataset and logs evaluation metrics.

Tasks: load_data → preprocess_data → train_model → evaluate_model
"""

from datetime import datetime, timedelta
import json
import os

from airflow import DAG
from airflow.operators.python import PythonOperator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = "/opt/airflow/models"
os.makedirs(MODEL_DIR, exist_ok=True)

default_args = {
    "owner": "sachin",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


# ---------------------------------------------------------------------------
# Task 1: Load Data
# ---------------------------------------------------------------------------
def load_data(**context):
    """Fetch the California Housing dataset and push to XCom."""
    from sklearn.datasets import fetch_california_housing
    import pandas as pd

    data = fetch_california_housing(as_frame=True)
    df = data.frame

    context["ti"].xcom_push(key="dataset", value=df.to_json())
    context["ti"].xcom_push(key="feature_names", value=list(data.feature_names))
    context["ti"].xcom_push(key="target_name", value=data.target_names[0])

    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Features: {list(data.feature_names)}")
    print(f"   Target:   {data.target_names[0]}")


# ---------------------------------------------------------------------------
# Task 2: Preprocess Data
# ---------------------------------------------------------------------------
def preprocess_data(**context):
    """Train/test split + standard scaling (no data leakage)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib

    ti = context["ti"]

    # Pull raw data
    df = pd.read_json(ti.xcom_pull(key="dataset", task_ids="load_data"))
    target = ti.xcom_pull(key="target_name", task_ids="load_data")

    X = df.drop(columns=[target])
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler artifact
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Push splits via XCom
    cols = X.columns.tolist()
    ti.xcom_push(key="X_train", value=pd.DataFrame(X_train_scaled, columns=cols).to_json())
    ti.xcom_push(key="X_test", value=pd.DataFrame(X_test_scaled, columns=cols).to_json())
    ti.xcom_push(key="y_train", value=y_train.reset_index(drop=True).to_json())
    ti.xcom_push(key="y_test", value=y_test.reset_index(drop=True).to_json())

    print(f"✅ Preprocessing complete")
    print(f"   Train samples: {len(y_train)}")
    print(f"   Test samples:  {len(y_test)}")
    print(f"   Scaler saved:  {scaler_path}")


# ---------------------------------------------------------------------------
# Task 3: Train Model
# ---------------------------------------------------------------------------
def train_model(**context):
    """Train a Random Forest Regressor and save model artifacts."""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    ti = context["ti"]
    X_train = pd.read_json(ti.xcom_pull(key="X_train", task_ids="preprocess_data"))
    y_train = pd.read_json(
        ti.xcom_pull(key="y_train", task_ids="preprocess_data"), typ="series"
    )

    # Train
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Save with timestamp + a "latest" copy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"rf_model_{timestamp}.joblib")
    latest_path = os.path.join(MODEL_DIR, "rf_model_latest.joblib")

    joblib.dump(model, model_path)
    joblib.dump(model, latest_path)

    ti.xcom_push(key="model_path", value=model_path)

    print(f"✅ Model trained successfully")
    print(f"   Saved to: {model_path}")
    print(f"   Latest:   {latest_path}")


# ---------------------------------------------------------------------------
# Task 4: Evaluate Model
# ---------------------------------------------------------------------------
def evaluate_model(**context):
    """Evaluate the model and save metrics."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib

    ti = context["ti"]
    model_path = ti.xcom_pull(key="model_path", task_ids="train_model")
    X_test = pd.read_json(ti.xcom_pull(key="X_test", task_ids="preprocess_data"))
    y_test = pd.read_json(
        ti.xcom_pull(key="y_test", task_ids="preprocess_data"), typ="series"
    )

    # Predict
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "trained_at": datetime.now().isoformat(),
        "model_path": model_path,
    }

    # Save metrics JSON
    metrics_path = os.path.join(MODEL_DIR, "latest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    ti.xcom_push(key="metrics", value=metrics)

    print("=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MAE:       {metrics['mae']}")
    print(f"  RMSE:      {metrics['rmse']}")
    print(f"  R² Score:  {metrics['r2_score']}")
    print(f"  Timestamp: {metrics['trained_at']}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="Periodic ML model training on California Housing data",
    schedule_interval="@weekly",
    start_date=datetime(2025, 2, 1),
    catchup=False,
    tags=["ml", "training", "sklearn"],
) as dag:

    t1 = PythonOperator(task_id="load_data", python_callable=load_data)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t3 = PythonOperator(task_id="train_model", python_callable=train_model)
    t4 = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    # Define task dependencies
    t1 >> t2 >> t3 >> t4