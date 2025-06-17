import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Logging argumen sebagai parameter
mlflow.log_param("test_size", args.test_size)
mlflow.log_param("random_state", args.random_state)

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load dataset
data = pd.read_csv("Supplement-Sales-Weekly-Expanded_preprocessing.csv")

# Fitur dan target
X = data.drop(columns=["Units Sold"])
y = data["Units Sold"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Logging manual
mlflow.log_metric("test_mse", mse)
mlflow.log_metric("test_r2_score", r2)
mlflow.log_metric("test_mae", mae)

print(f"MSE: {mse}")
print(f"R² Score: {r2}")
