import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Logging parameterer
mlflow.log_param("test_size", args.test_size)
mlflow.log_param("random_state", args.random_state)

# Enable autologging
mlflow.sklearn.autolog()

# Load dataset
data = pd.read_csv("Supplement-Sales-Weekly-Expanded_preprocessing.csv")
X = data.drop(columns=["Units Sold"])
y = data["Units Sold"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ⬇️ Simpan model ke MLflow (agar muncul di path artifacts/model/)
mlflow.sklearn.log_model(model, artifact_path="model")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Log metrics
mlflow.log_metric("test_mse", mse)
mlflow.log_metric("test_r2_score", r2)
mlflow.log_metric("test_mae", mae)

# Print metrics
print(f"MSE: {mse}")
print(f"R² Score: {r2}")
