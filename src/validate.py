import os
import pandas as pd


def validar_y_preparar():
    enrutamiento = "Anime Dataset"
    ruta_anime = f"{enrutamiento}/anime.csv"
    ruta_ratings = f"{enrutamiento}/rating_complete.csv"

    assert os.path.exists(ruta_anime), f"No se encontró {ruta_anime}"
    assert os.path.exists(ruta_ratings), f"No se encontró {ruta_ratings}"

    columnas_validas = [
        "MAL_ID", "Genres", "Type", "Episodes", "Aired",
        "Studios", "Source", "Duration", "Rating", "Name", "English name"
    ]

    df_anime = pd.read_csv(ruta_anime, usecols=columnas_validas, na_values=["Unknown"])
    df_anime["release_year"] = pd.to_numeric(
        df_anime["Aired"].str.extract(r"(\d{4})")[0], errors="coerce"
    ).astype("Int32")
    df_anime["duration_min"] = pd.to_numeric(
        df_anime["Duration"].str.extract(r"(\d+)")[0], errors="coerce"
    ).astype("Int32")
    df_anime["Episodes"] = pd.to_numeric(df_anime["Episodes"], errors="coerce").astype("Int32")

    df_anime["main_studio"] = df_anime["Studios"].str.split(",").str[0].str.strip()
    df_anime["main_genre"] = df_anime["Genres"].str.split(",").str[0].str.strip()
    df_anime["genre_count"] = df_anime["Genres"].str.split(",").str.len().fillna(0).astype("int8")

    df_anime.drop(columns=["Aired", "Duration", "Studios", "Genres"], inplace=True)
    df_anime.rename(columns={"MAL_ID": "anime_id"}, inplace=True)
    df_anime = df_anime[df_anime["release_year"].between(1970, 2020)]

    df_ratings = pd.read_csv(
        ruta_ratings,
        dtype={"user_id": "int32", "anime_id": "int32", "rating": "int8"},
    )
    df_ratings = df_ratings[df_ratings["user_id"] % 15 == 0]

    conteo_usuarios = df_ratings["user_id"].value_counts()
    usuarios_activos = conteo_usuarios[conteo_usuarios >= 20].index
    df_ratings = df_ratings[df_ratings["user_id"].isin(usuarios_activos)]

    df_preparado = df_ratings.merge(df_anime, on="anime_id", how="inner")

    # Reglas de validación de calidad de datos
    assert len(df_preparado) > 0, "El dataset preparado quedó vacío."
    assert df_preparado[["user_id", "anime_id", "rating"]].isnull().sum().sum() == 0, "Hay nulos en columnas clave."
    assert df_preparado["rating"].between(1, 10).all(), "Hay calificaciones fuera del rango 1-10."

    os.makedirs("data", exist_ok=True)
    archivo_salida = "data/TablaFinalAnime.parquet"
    df_preparado.to_parquet(archivo_salida, index=False)

    print(f"Validación exitosa -> Filas: {len(df_preparado):,} | Usuarios: {df_preparado['user_id'].nunique():,}")


if __name__ == "__main__":
    validar_y_preparar()