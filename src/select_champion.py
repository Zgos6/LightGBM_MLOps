import os
import mlflow
from mlflow.tracking import MlflowClient


def seleccionar_campeon():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experimento = mlflow.get_experiment_by_name("LightGBM_Anime")
    if experimento is None:
        raise RuntimeError("No existe el experimento LightGBM_Anime en MLflow.")

    corridas = mlflow.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.roc_auc DESC", "metrics.f1_score DESC", "start_time DESC"],
    )

    if corridas.empty:
        raise RuntimeError("No se encontraron corridas finalizadas.")

    ganador = corridas.iloc[0]
    run_id = ganador["run_id"]
    print(f"Run ganador seleccionado: {run_id} (ROC-AUC: {ganador.get('metrics.roc_auc', 0):.4f})")

    model_uri = None

    # Compatibilidad con MLflow 3.x
    if hasattr(mlflow, "search_logged_models"):
        modelos = mlflow.search_logged_models(experiment_ids=[experimento.experiment_id])
        if not modelos.empty:
            coincidencia = modelos[modelos["source_run_id"] == run_id]
            fila = coincidencia.iloc[0] if not coincidencia.empty else modelos.iloc[0]
            model_uri = f"models:/{fila['model_id']}"

    # Respaldo para MLflow 2.x
    if model_uri is None:
        artefactos = [a.path for a in client.list_artifacts(run_id)]
        if "modelo_lightgbm" in artefactos:
            model_uri = f"runs:/{run_id}/modelo_lightgbm"

    if not model_uri:
        raise RuntimeError(f"El run ganador {run_id} no contiene ningún modelo guardado.")

    version_registrada = mlflow.register_model(
        model_uri=model_uri,
        name="AnimeLikeClassifier",
    )

    client.set_registered_model_alias(
        name="AnimeLikeClassifier",
        alias="champion",
        version=str(version_registrada.version),
    )
    print(f"Modelo registrado: AnimeLikeClassifier v{version_registrada.version} con alias @champion")


if __name__ == "__main__":
    seleccionar_campeon()