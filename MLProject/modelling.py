import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

mlflow.set_experiment("sales_forecasting_experiment")

with mlflow.start_run():
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("random_state", args.random_state)

    mlflow.sklearn.autolog()

    data = pd.read_csv("Supplement-Sales-Weekly-Expanded_preprocessing.csv")
    X = data.drop(columns=["Units Sold"])
    y = data["Units Sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = RandomForestRegressor(random_state=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)

    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")
    print(f"MAE: {mae}")
