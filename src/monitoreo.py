import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
except ImportError:
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, ClassificationPreset

# 1. Cargar modelo entrenado y perfiles generados
modelo = joblib.load("models/lgbm_anime.pkl")
user_stats = pd.read_parquet("models/user_stats.parquet")[
    ["user_id", "user_mean_rating", "user_std_rating", "user_rating_count"]
]
user_genre_stats = pd.read_parquet("models/user_genre_stats.parquet")

ruta_parquet = "Anime Dataset/TablaFinalAnime.parquet"
if not os.path.exists(ruta_parquet):
    ruta_parquet = "data/TablaFinalAnime.parquet"

df = pd.read_parquet(ruta_parquet)
df["main_genre"] = df["main_genre"].astype(str).replace(["nan", "None", "<NA>"], "Unknown")
df["target"] = (df["rating"] >= 8).astype(int)

# Cruzar variables históricas de usuario (idéntico a Anime.py)
df = df.merge(user_stats, on="user_id", how="inner")
df = df.merge(user_genre_stats, on=["user_id", "main_genre"], how="left")
df["user_genre_mean"] = df["user_genre_mean"].fillna(df["user_mean_rating"])

FEATURES = [
    "Type", "Episodes", "Source", "Rating", "release_year",
    "duration_min", "main_studio", "main_genre", "genre_count",
    "user_mean_rating", "user_std_rating", "user_rating_count", "user_genre_mean"
]
CAT_COLS = ["Type", "Source", "Rating", "main_studio", "main_genre"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]


def preparar_categorias(datos: pd.DataFrame) -> pd.DataFrame:
    X = datos[FEATURES].copy()
    pandas_cats = getattr(getattr(modelo, "booster_", None), "pandas_categorical", None)
    for i, col in enumerate(CAT_COLS):
        if pandas_cats and i < len(pandas_cats):
            X[col] = pd.Categorical(X[col], categories=pandas_cats[i])
        else:
            X[col] = X[col].astype("category")
    return X


# 2. Separar Dataset de Referencia (70%) y Dataset Actual (30%)
corte = int(len(df) * 0.70)
ref_df = df.iloc[:corte].sample(n=min(12000, corte), random_state=42).copy()
curr_df = df.iloc[corte:].sample(n=min(8000, len(df) - corte), random_state=42).copy()

# Normalizar tipos estándar (evita errores de Int32/pd.NA en Evidently)
for col in NUM_COLS:
    mediana = float(ref_df[col].dropna().median())
    ref_df[col] = ref_df[col].astype("float64").fillna(mediana)
    curr_df[col] = curr_df[col].astype("float64").fillna(mediana)

for col in CAT_COLS:
    ref_df[col] = ref_df[col].astype(str).replace(["nan", "None", "<NA>"], "Unknown")
    curr_df[col] = curr_df[col].astype(str).replace(["nan", "None", "<NA>"], "Unknown")

# Simular cambio de comportamiento en producción (Data Drift) sobre datos actuales
np.random.seed(42)
curr_df["user_mean_rating"] = np.clip(curr_df["user_mean_rating"] - 1.35, 1.0, 10.0)
curr_df["user_genre_mean"] = np.clip(curr_df["user_genre_mean"] - 1.45, 1.0, 10.0)
curr_df["release_year"] = np.clip(curr_df["release_year"] + 5, 1970, 2020)
curr_df["Episodes"] = curr_df["Episodes"] * 1.4

# 3. Generar predicciones con LightGBM en ambos conjuntos
probs_ref = modelo.predict_proba(preparar_categorias(ref_df))[:, 1].astype("float64")
probs_curr = modelo.predict_proba(preparar_categorias(curr_df))[:, 1].astype("float64")


# 4. Calcular y comparar métricas de clasificación
def evaluar_metricas(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


m_ref = evaluar_metricas(ref_df["target"], probs_ref)
m_curr = evaluar_metricas(curr_df["target"], probs_curr)

print("=" * 62)
print("MONITOREO DE MODELO LIGHTGBM - REFERENCIA VS ACTUAL")
print("=" * 62)
print(f"Dataset de Referencia : {len(ref_df):,} registros")
print(f"Dataset Actual        : {len(curr_df):,} registros")
print("-" * 62)
for metrica in ["Accuracy", "F1-Score", "ROC-AUC"]:
    diff = m_curr[metrica] - m_ref[metrica]
    print(f"{metrica:<12} | Referencia: {m_ref[metrica]:.4f} | Actual: {m_curr[metrica]:.4f} | Cambio: {diff:+.4f}")
print("-" * 62)

# 5. Evaluar umbral de alerta por deterioro
caida_auc = m_ref["ROC-AUC"] - m_curr["ROC-AUC"]
if caida_auc > 0.03:
    print(f"[ALERTA POR DETERIORO] El ROC-AUC cayó {caida_auc:.4f} puntos (> 0.03).")
    print("[DECISIÓN RECOMENDADA] Se detectó Data Drift significativo en las preferencias")
    print("de calificación de usuarios y año de estreno. Se recomienda reentrenar LightGBM")
    print("con los datos recientes y versionar el nuevo dataset en DVC.")
else:
    print("[ESTADO NORMAL] El rendimiento del modelo se mantiene estable.")
print("=" * 62)

# 6. Preparar tablas limpias y generar Reporte HTML con Evidently
ev_ref = ref_df[FEATURES].copy()
ev_ref["target"] = ref_df["target"].astype(str)
ev_ref["0"] = 1.0 - probs_ref
ev_ref["1"] = probs_ref

ev_curr = curr_df[FEATURES].copy()
ev_curr["target"] = curr_df["target"].astype(str)
ev_curr["0"] = 1.0 - probs_curr
ev_curr["1"] = probs_curr

column_mapping = ColumnMapping(
    target="target",
    prediction=["0", "1"],
    pos_label="1",
    numerical_features=NUM_COLS,
    categorical_features=CAT_COLS,
)

reporte = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset(),
])

reporte.run(
    reference_data=ev_ref,
    current_data=ev_curr,
    column_mapping=column_mapping,
)

reporte.save_html("reporte_drift.html")
print("Reporte HTML generado exitosamente en: reporte_drift.html")