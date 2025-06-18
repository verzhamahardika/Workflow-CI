import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import os
import joblib

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
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")  # Simpan model secara manual

    # Logging secara eksplisit
    mlflow.log_artifact("model/model.pkl", artifact_path="model")

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Logging metrics
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_r2_score", r2)
    mlflow.log_metric("test_mae", mae)

    # Print output
    print(f"MSE: {mse}")
    print(f"R² Score: {r2}")
