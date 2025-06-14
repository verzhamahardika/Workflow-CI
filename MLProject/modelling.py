import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load dataset
data = pd.read_csv("Supplement-Sales-Weekly-Expanded_preprocessing.csv")

# Fitur dan target
X = data.drop(columns=["Units Sold"])
y = data["Units Sold"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi experiment MLflow
mlflow.set_experiment("Sales Forecasting")

with mlflow.start_run():
    # Model tanpa tuning
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi (opsional, karena autolog sudah mencatat)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Logging manual untuk metrik test
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_r2_score", r2)
    mlflow.log_metric("test_mae", mae)

    print(f"MSE: {mse}")
    print(f"R² Score: {r2}")
