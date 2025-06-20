import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models.signature import infer_signature


def main(data_path: str, test_size: float, random_state: int):
    # 1) ── Load data ───────────────────────────────
    df = pd.read_csv(data_path)

    X = df.drop("Units Sold", axis=1)
    y = df["Units Sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2) ── Start an MLflow run ─────────────────────
    with mlflow.start_run():
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)

        # 3) ── Predictions & metrics ────────────────
        y_pred = model.predict(X_test)
        mse  = mean_squared_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)

        # 4) ── Log model & metrics explicitly ───────
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="rf_sales_regressor",
            input_example=X_test.head(3),
            signature=signature,
        )

        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_r2",  r2)
        mlflow.log_metric("test_mae", mae)

        # 5) ── Console feedback ─────────────────────
        print(f"MSE: {mse:.4f}")
        print(f"R² : {r2:.4f}")
        print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   type=str,   required=True,
                        help="Path ke CSV hasil preprocessing")
    parser.add_argument("--test_size",   type=float, default=0.2,
                        help="Porsi data uji (0‑1)")
    parser.add_argument("--random_state", type=int,   default=42,
                        help="Seed supaya hasil re‑producible")
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )
