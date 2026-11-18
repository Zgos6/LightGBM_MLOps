import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import lightgbm as lgb
import shap

enrutamiento = "C:/Users/User/Documents/Mineria de datos/Proyecto MLOps/Anime Dataset"

# 1. Mantener las columnas utiles para la prediccion##
columnasValidas = ["MAL_ID","Genres","Type","Episodes","Aired","Studios","Source","Duration","Rating","Name","English name"]

dfAnime = pd.read_csv(
    f"{enrutamiento}/anime.csv", usecols=columnasValidas, na_values=["Unknown"]
)

# 2. Extraer año de estreno de los animes#
dfAnime["release_year"] = pd.to_numeric(dfAnime["Aired"].str.extract(r"(\d{4})")[0], errors="coerce").astype("Int32")

# 3. Convertir Duration a minutos numéricos#
dfAnime["duration_min"] = pd.to_numeric(dfAnime["Duration"].str.extract(r"(\d+)")[0], errors="coerce").astype("Int32")

# 4. Convertir Episodes a entero#
dfAnime["Episodes"] = pd.to_numeric(dfAnime["Episodes"], errors="coerce").astype("Int32")

# 5. Simplificar el nombre de las columnas# 
dfAnime["main_studio"] = dfAnime["Studios"].str.split(",").str[0].str.strip()
dfAnime["main_genre"] = dfAnime["Genres"].str.split(",").str[0].str.strip()
dfAnime["genre_count"] = (dfAnime["Genres"].str.split(",").str.len().fillna(0).astype("int8"))

# 6. Eliminar columnas de texto crudo ya procesadas y estandarizar ID para el cruce#
dfAnime.drop(columns=["Aired", "Duration", "Studios", "Genres"], inplace=True)
dfAnime.rename(columns={"MAL_ID": "anime_id"}, inplace=True)

# 7. Filtrar animes con año de estreno relevantes#
dfAnime = dfAnime[dfAnime["release_year"].between(1970, 2020)]

# 8. Muestrear po user_id y cruzar tablas#
dfRatings = pd.read_csv(f"{enrutamiento}/rating_complete.csv",dtype={"user_id": "int32", "anime_id": "int32", "rating": "int8"},)

# 9. Seleccionar ~6.6% de usuarios#
dfRatings = dfRatings[dfRatings["user_id"] % 15 == 0]

# 10. Filtrar usuarios activos#
conteoUsuarios = dfRatings["user_id"].value_counts()
usuariosActivos = conteoUsuarios[conteoUsuarios >= 20].index
dfRatings = dfRatings[dfRatings["user_id"].isin(usuariosActivos)]

# 11. Unir calificaciones con los animes#
dfPreparado = dfRatings.merge(dfAnime, on="anime_id", how="inner")

# 12. Guardar resultado#
archivoSalida = f"{enrutamiento}/TablaFinalAnime.parquet"
dfPreparado.to_parquet(archivoSalida, index=False)

print(f"Filas finales: {len(dfPreparado):,}")
print(f"Usuarios únicos conservados: {dfPreparado['user_id'].nunique():,}")
print(f"Columnas finales: {list(dfPreparado.columns)}")

# Crear la variable objetivo binaria (1 si rating >= 8, 0 en caso contrario)
dfPreparado['user_like_this'] = (dfPreparado['rating'] >= 8).astype('int8')

dfTrain, dfTest=train_test_split(
    dfPreparado,
    test_size=0.20,
    stratify=dfPreparado['user_id'],
    random_state=42
)

dfTrain['rating'] = dfTrain['rating'].astype('float32')
dfTrain['main_genre'] = dfTrain['main_genre'].fillna('Unknown')
dfTest['main_genre'] = dfTest['main_genre'].fillna('Unknown')

# ==============================================================================
# A. CALCULAR VARIABLES EN dfTrain (Leave-One-Out sin fuga de datos)
# ==============================================================================
user_sum = dfTrain.groupby('user_id')['rating'].transform('sum')
user_cnt = dfTrain.groupby('user_id')['rating'].transform('count')

dfTrain['user_rating_count'] = (user_cnt - 1).astype('int32')
dfTrain['user_mean_rating'] = (user_sum - dfTrain['rating']) / dfTrain['user_rating_count']
dfTrain['user_std_rating'] = dfTrain.groupby('user_id')['rating'].transform('std').fillna(0)

# Promedio por usuario y género específico (Leave-One-Out)
ug_sum = dfTrain.groupby(['user_id', 'main_genre'])['rating'].transform('sum')
ug_cnt = dfTrain.groupby(['user_id', 'main_genre'])['rating'].transform('count')

dfTrain['user_genre_mean'] = (ug_sum - dfTrain['rating']) / (ug_cnt - 1)

# Si era el único anime de ese género en train (división entre 0 -> inf/NaN), usar su promedio general
dfTrain['user_genre_mean'] = (
    dfTrain['user_genre_mean']
    .replace([np.inf, -np.inf], np.nan)
    .fillna(dfTrain['user_mean_rating'])
)

# ==============================================================================
# B. TRASLADAR PERFIL APRENDIDO HACIA dfTest (Sin tocar las notas de test)
# ==============================================================================
user_stats = dfTrain.groupby('user_id', as_index=False).agg(
    user_mean_rating=('rating', 'mean'),
    user_std_rating=('rating', 'std'),
    user_rating_count=('rating', 'count')
)
user_stats['user_std_rating'] = user_stats['user_std_rating'].fillna(0)

genre_stats = dfTrain.groupby(['user_id', 'main_genre'], as_index=False).agg(
    user_genre_mean=('rating', 'mean')
)

dfTest = dfTest.merge(user_stats, on='user_id', how='left')
dfTest = dfTest.merge(genre_stats, on=['user_id', 'main_genre'], how='left')

# Si en test aparece un género nuevo para ese usuario, respaldar con su promedio general
dfTest['user_genre_mean'] = dfTest['user_genre_mean'].fillna(dfTest['user_mean_rating'])

features=['Type','Episodes','Source','Rating','release_year','duration_min','main_studio','main_genre','genre_count','user_mean_rating','user_std_rating','user_rating_count','user_genre_mean']
target = 'user_like_this'

x_train = dfTrain[features].copy()
y_train = dfTrain[target]

x_test = dfTest[features].copy()
y_test = dfTest[target]

cat_cols = ['Type', 'Source', 'Rating', 'main_studio', 'main_genre']
for col in cat_cols:
    x_train[col] = x_train[col].astype('category')
    x_test[col] = x_test[col].astype('category')


# 1. Entrenar el clasificador LightGBM
modelo = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

modelo.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=50)]
)

# 2. Evaluar métricas en el 20% de prueba
y_pred = modelo.predict(x_test)
y_prob = modelo.predict_proba(x_test)[:, 1]

print(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# 3. Calcular valores SHAP (sobre 10,000 filas de prueba para ejecución rápida)
x_test_sample = x_test.sample(n=min(10_000, len(x_test)), random_state=42)
explainer = shap.TreeExplainer(modelo)
shap_values = explainer.shap_values(x_test_sample)

# Compatibilidad entre versiones de SHAP para clasificación binaria (clase 1 = le gustó)
if isinstance(shap_values, list):
    shap_matrix = shap_values[1]
elif shap_values.ndim == 3:
    shap_matrix = shap_values[:, :, 1]
else:
    shap_matrix = shap_values

# 4. Responder la pregunta: Exigencia del usuario vs. Características del anime
importancia = pd.DataFrame({
    'variable': x_test_sample.columns,
    'impacto_shap': np.abs(shap_matrix).mean(axis=0)
}).sort_values('impacto_shap', ascending=False)

vars_usuario = ['user_mean_rating', 'user_std_rating', 'user_rating_count', 'user_genre_mean']
importancia['grupo'] = np.where(
    importancia['variable'].isin(vars_usuario),
    'Exigencia del Usuario',
    'Características del Anime'
)

resumen_grupos = importancia.groupby('grupo')['impacto_shap'].sum()
porcentaje_grupos = (resumen_grupos / resumen_grupos.sum()) * 100

print("\n--- VEREDICTO: PESO RELATIVO EN LA DECISIÓN (%) ---")
print(porcentaje_grupos.round(2))
print("\n--- RANKING INDIVIDUAL DE VARIABLES ---")
print(importancia.to_string(index=False))

# Gráfico visual del impacto de cada variable
shap.summary_plot(shap_matrix, x_test_sample)

# 5. Inspeccionar predicciones con los nombres reales de los animes
dfTest['prob_le_guste'] = y_prob
usuario_ejemplo = dfTest['user_id'].iloc[0]
cols_ver = ['Name', 'English name', 'main_genre', 'rating', 'user_mean_rating', 'prob_le_guste']

print(f"\nPredicciones reales para el usuario {usuario_ejemplo}:")
print(dfTest[dfTest['user_id'] == usuario_ejemplo][cols_ver].head(10).to_string(index=False))