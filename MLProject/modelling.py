import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature  # ✅ FIXED
import os

# Ambil parameter dari CLI
parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Autolog semua aktivitas ML
mlflow.sklearn.autolog()

# Mulai run nested agar tidak bentrok saat dijalankan dari `mlflow run`
with mlflow.start_run(nested=True):

    # Load dataset
    data = pd.read_csv("Supplement-Sales-Weekly-Expanded_preprocessing.csv")
    X = data.drop(columns=["Units Sold"])
    y = data["Units Sold"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Log model dengan signature
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_test.head(3),
        signature=signature
    )

    # Logging metrics
    mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))
    mlflow.log_metric("test_mae", mean_absolute_error(y_test, y_pred))

    # Output to console
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")
