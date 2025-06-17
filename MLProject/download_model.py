import mlflow
from mlflow.tracking import MlflowClient
import shutil
import os

# Dapatkan run ID terakhir
client = MlflowClient()
experiment = client.get_experiment_by_name("Default")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)
run_id = runs[0].info.run_id
print("Found run_id:", run_id)

# Unduh artefak model
downloaded_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
print("Model downloaded to:", downloaded_path)

# Pindahkan ke best_model/
if os.path.exists("best_model"):
    shutil.rmtree("best_model")
shutil.move(downloaded_path, "best_model")
