import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

run_id = "d9b9bf5b717d4106b8b2473162b01a0d"
experimento = mlflow.get_experiment_by_name("LightGBM_Anime")
exp_id = experimento.experiment_id

model_uri = None

# 1. Buscar en Logged Models de MLflow 3 dentro del experimento
if hasattr(mlflow, "search_logged_models"):
    modelos = mlflow.search_logged_models(experiment_ids=[exp_id])
    if not modelos.empty:
        coincidencia = modelos[modelos["source_run_id"] == run_id]
        fila = coincidencia.iloc[0] if not coincidencia.empty else modelos.iloc[0]
        model_uri = f"models:/{fila['model_id']}"

# 2. Respaldo para MLflow 2.x (artefactos clásicos del run)
if model_uri is None:
    artefactos = [a.path for a in client.list_artifacts(run_id)]
    if "modelo_lightgbm" in artefactos:
        model_uri = f"runs:/{run_id}/modelo_lightgbm"

if model_uri:
    print(f"Registrando modelo desde: {model_uri}")
    mlflow.register_model(model_uri=model_uri, name="AnimeLikeClassifier")
else:
    print("No se encontró ningún modelo en el experimento. Verifica que Anime.py esté guardado y vuelve a correrlo.")