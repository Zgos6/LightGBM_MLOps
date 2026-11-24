import os
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
import lightgbm as lgb
import shap
import mlflow
import mlflow.lightgbm

BASE_DIR = Path(__file__).resolve().parent.parent


def entrenar_modelo():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("LightGBM_Anime")

    dataset_version = os.getenv("DATASET_VERSION", "v2")

    dvc_md5 = "no_encontrado"
    dvc_file = BASE_DIR / "Anime Dataset.dvc"
    if dvc_file.exists():
        match = re.search(r"md5:\s*([a-f0-9.]+)", dvc_file.read_text(encoding="utf-8"))
        if match:
            dvc_md5 = match.group(1)

    df_preparado = pd.read_parquet(BASE_DIR / "data" / "TablaFinalAnime.parquet")
    df_preparado["user_like_this"] = (df_preparado["rating"] >= 8).astype("int8")

    df_train, df_test = train_test_split(
        df_preparado,
        test_size=0.20,
        stratify=df_preparado["user_id"],
        random_state=42,
    )

    df_train["rating"] = df_train["rating"].astype("float32")
    df_train["main_genre"] = df_train["main_genre"].fillna("Unknown")
    df_test["main_genre"] = df_test["main_genre"].fillna("Unknown")

    # A. Variables en df_train (Leave-One-Out sin fuga de datos)
    user_sum = df_train.groupby("user_id")["rating"].transform("sum")
    user_cnt = df_train.groupby("user_id")["rating"].transform("count")

    df_train["user_rating_count"] = (user_cnt - 1).astype("int32")
    df_train["user_mean_rating"] = (user_sum - df_train["rating"]) / df_train["user_rating_count"]
    df_train["user_std_rating"] = df_train.groupby("user_id")["rating"].transform("std").fillna(0)

    ug_sum = df_train.groupby(["user_id", "main_genre"])["rating"].transform("sum")
    ug_cnt = df_train.groupby(["user_id", "main_genre"])["rating"].transform("count")

    df_train["user_genre_mean"] = (ug_sum - df_train["rating"]) / (ug_cnt - 1)
    df_train["user_genre_mean"] = (
        df_train["user_genre_mean"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df_train["user_mean_rating"])
    )

    # B. Trasladar perfil aprendido hacia df_test
    user_stats = df_train.groupby("user_id", as_index=False).agg(
        user_mean_rating=("rating", "mean"),
        user_std_rating=("rating", "std"),
        user_rating_count=("rating", "count"),
    )
    user_stats["user_std_rating"] = user_stats["user_std_rating"].fillna(0)

    genre_stats = df_train.groupby(["user_id", "main_genre"], as_index=False).agg(
        user_genre_mean=("rating", "mean")
    )

    df_test = df_test.merge(user_stats, on="user_id", how="left")
    df_test = df_test.merge(genre_stats, on=["user_id", "main_genre"], how="left")
    df_test["user_genre_mean"] = df_test["user_genre_mean"].fillna(df_test["user_mean_rating"])

    features = [
        "Type", "Episodes", "Source", "Rating", "release_year", "duration_min",
        "main_studio", "main_genre", "genre_count", "user_mean_rating",
        "user_std_rating", "user_rating_count", "user_genre_mean",
    ]
    target = "user_like_this"

    x_train, y_train = df_train[features].copy(), df_train[target]
    x_test, y_test = df_test[features].copy(), df_test[target]

    for col in ["Type", "Source", "Rating", "main_studio", "main_genre"]:
        x_train[col] = x_train[col].astype("category")
        x_test[col] = x_test[col].astype("category")

    # C. Entrenamiento y registro en MLflow
    with mlflow.start_run(run_name=f"lightgbm_{dataset_version}"):
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("dvc_md5", dvc_md5)

        modelo = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        )

        mlflow.log_params({
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "test_size": 0.20,
            "filas_preparadas": len(df_preparado),
            "usuarios_unicos": int(df_preparado["user_id"].nunique()),
        })

        modelo.fit(
            x_train, y_train,
            eval_set=[(x_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=50)],
        )

        y_pred = modelo.predict(x_test)
        y_prob = modelo.predict_proba(x_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        if modelo.best_iteration_:
            mlflow.log_metric("best_iteration", modelo.best_iteration_)

        print(f"\nROC-AUC: {auc:.4f} | F1-Score: {f1:.4f} | Accuracy: {acc:.4f}")
        print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

        # SHAP
        x_test_sample = x_test.sample(n=min(10_000, len(x_test)), random_state=42)
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(x_test_sample)

        if isinstance(shap_values, list):
            shap_matrix = shap_values[1]
        elif shap_values.ndim == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        importancia = pd.DataFrame({
            "variable": x_test_sample.columns,
            "impacto_shap": np.abs(shap_matrix).mean(axis=0),
        }).sort_values("impacto_shap", ascending=False)

        vars_usuario = ["user_mean_rating", "user_std_rating", "user_rating_count", "user_genre_mean"]
        importancia["grupo"] = np.where(
            importancia["variable"].isin(vars_usuario),
            "Exigencia del Usuario",
            "Características del Anime",
        )

        resumen_grupos = importancia.groupby("grupo")["impacto_shap"].sum()
        porcentaje_grupos = (resumen_grupos / resumen_grupos.sum()) * 100

        mlflow.log_metric("peso_exigencia_usuario_pct", float(porcentaje_grupos.get("Exigencia del Usuario", 0)))
        mlflow.log_metric("peso_caracteristicas_anime_pct", float(porcentaje_grupos.get("Características del Anime", 0)))

        shap_path = BASE_DIR / "shap_summary.png"
        shap.summary_plot(shap_matrix, x_test_sample, show=False)
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        mlflow.log_artifact(str(shap_path))
        if shap_path.exists():
            shap_path.unlink()

        mlflow.lightgbm.log_model(modelo, artifact_path="modelo_lightgbm")

    df_test["prob_le_guste"] = y_prob
    usuario_ejemplo = df_test["user_id"].iloc[0]
    cols_ver = ["Name", "English name", "main_genre", "rating", "user_mean_rating", "prob_le_guste"]
    print(f"\nPredicciones reales para el usuario {usuario_ejemplo}:")
    print(df_test[df_test["user_id"] == usuario_ejemplo][cols_ver].head(10).to_string(index=False))


if __name__ == "__main__":
    entrenar_modelo()