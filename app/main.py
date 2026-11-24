import os
import random
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Anime Recommendation System - LightGBM",
    description="API y aplicación web conectada al perfil real de usuarios y catálogo de MyAnimeList.",
    version="3.1.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_anime.pkl")

try:
    modelo = joblib.load(MODEL_PATH)
    catalogo_df = pd.read_parquet("models/catalogo_animes.parquet")
    user_stats_df = pd.read_parquet("models/user_stats.parquet")
    user_genre_df = pd.read_parquet("models/user_genre_stats.parquet")
    user_watched_df = pd.read_parquet("models/user_watched.parquet")
    USER_STATS_MAP = user_stats_df.set_index("user_id").to_dict(orient="index")
    SAMPLE_USERS = sorted(user_stats_df["user_id"].head(40).tolist())
except Exception as e:
    modelo = None
    catalogo_df = None
    user_genre_df = None
    user_watched_df = None
    USER_STATS_MAP = {}
    SAMPLE_USERS = []
    print(f"Error cargando artefactos: {e}")

FEATURES = [
    "Type", "Episodes", "Source", "Rating", "release_year",
    "duration_min", "main_studio", "main_genre", "genre_count",
    "user_mean_rating", "user_std_rating", "user_rating_count", "user_genre_mean"
]
CAT_COLS = ["Type", "Source", "Rating", "main_studio", "main_genre"]


def aplicar_categorias_modelo(df: pd.DataFrame) -> pd.DataFrame:
    pandas_cats = getattr(getattr(modelo, "booster_", None), "pandas_categorical", None)
    for i, col in enumerate(CAT_COLS):
        if pandas_cats and i < len(pandas_cats):
            df[col] = pd.Categorical(df[col], categories=pandas_cats[i])
        else:
            df[col] = df[col].astype("category")
    return df


def calcular_nivel_afinidad(probabilidad: float) -> str:
    if probabilidad >= 0.75:
        return "Alta afinidad"
    elif probabilidad >= 0.45:
        return "Afinidad media"
    return "Baja afinidad"


class AnimeInput(BaseModel):
    Type: str = Field(..., example="TV")
    Episodes: int = Field(..., ge=1, example=24)
    Source: str = Field(..., example="Manga")
    Rating: str = Field(..., example="PG-13 - Teens 13 or older")
    release_year: int = Field(..., ge=1970, le=2025, example=2019)
    duration_min: int = Field(..., ge=1, example=24)
    main_studio: str = Field(..., example="ufotable")
    main_genre: str = Field(..., example="Action")
    genre_count: int = Field(..., ge=1, example=3)
    user_mean_rating: float = Field(..., ge=1.0, le=10.0, example=8.1)
    user_std_rating: float = Field(..., ge=0.0, example=1.15)
    user_rating_count: int = Field(..., ge=1, example=95)
    user_genre_mean: float = Field(..., ge=1.0, le=10.0, example=8.6)


class UserRecommendRequest(BaseModel):
    user_id: int = Field(..., example=15)
    genre: str = Field(default="Todos")
    anime_type: str = Field(default="Todos")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": modelo is not None,
        "total_users": len(USER_STATS_MAP),
        "total_animes": len(catalogo_df) if catalogo_df is not None else 0,
        "algorithm": "LightGBMClassifier"
    }


@app.get("/users/info")
def get_users_info():
    random_id = random.choice(list(USER_STATS_MAP.keys())) if USER_STATS_MAP else None
    return {"sample_users": SAMPLE_USERS, "random_user": random_id}


@app.post("/predict")
def predict(data: AnimeInput):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    df = pd.DataFrame([data.model_dump()])[FEATURES]
    df = aplicar_categorias_modelo(df)

    prediccion = int(modelo.predict(df)[0])
    probabilidad = float(modelo.predict_proba(df)[0][1])

    # Generar Top 3 de animes reales afines al perfil manual ingresado
    top3_recomendados = []
    if catalogo_df is not None:
        candidatos = catalogo_df[
            catalogo_df["main_genre"].str.lower() == data.main_genre.strip().lower()
        ].copy()

        if not candidatos.empty:
            por_tipo = candidatos[candidatos["Type"].str.lower() == data.Type.strip().lower()]
            if len(por_tipo) >= 3:
                candidatos = por_tipo.copy()

        if candidatos.empty:
            candidatos = catalogo_df.copy()

        candidatos["user_mean_rating"] = data.user_mean_rating
        candidatos["user_std_rating"] = data.user_std_rating
        candidatos["user_rating_count"] = data.user_rating_count
        candidatos["user_genre_mean"] = data.user_genre_mean

        X_cand = aplicar_categorias_modelo(candidatos[FEATURES].copy())
        candidatos["probabilidad"] = modelo.predict_proba(X_cand)[:, 1]
        top3 = candidatos.sort_values("probabilidad", ascending=False).head(3)

        for _, row in top3.iterrows():
            prob_item = float(row["probabilidad"])
            eng = row["English name"] if pd.notna(row["English name"]) else row["Name"]
            studio = row["main_studio"] if pd.notna(row["main_studio"]) else "Desconocido"
            top3_recomendados.append({
                "name": row["Name"],
                "english_name": eng,
                "type": row["Type"] if pd.notna(row["Type"]) else "N/A",
                "episodes": int(row["Episodes"]) if pd.notna(row["Episodes"]) else "?",
                "studio": studio,
                "genre": row["main_genre"],
                "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
                "probabilidad": round(prob_item * 100, 1),
                "affinity_level": calcular_nivel_afinidad(prob_item)
            })

    return {
        "user_like_this": prediccion,
        "resultado": "Le gustará (Rating >= 8)" if prediccion == 1 else "No le gustará (Rating < 8)",
        "probabilidad_gusto": round(probabilidad, 4),
        "affinity_level": calcular_nivel_afinidad(probabilidad),
        "top_3_recomendados": top3_recomendados
    }


@app.post("/recommend_user")
def recommend_for_user(req: UserRecommendRequest):
    if modelo is None or catalogo_df is None:
        raise HTTPException(status_code=500, detail="Artefactos no disponibles.")

    if req.user_id not in USER_STATS_MAP:
        raise HTTPException(status_code=404, detail=f"El user_id {req.user_id} no existe en la muestra de usuarios activos.")

    perfil = USER_STATS_MAP[req.user_id]

    # 1. Excluir animes que el usuario YA calificó/vio
    vistos = set(user_watched_df[user_watched_df["user_id"] == req.user_id]["anime_id"])
    candidatos = catalogo_df[~catalogo_df["anime_id"].isin(vistos)].copy()

    # 2. Aplicar filtros opcionales de género o formato
    if req.genre != "Todos":
        candidatos = candidatos[candidatos["main_genre"] == req.genre]
    if req.anime_type != "Todos":
        candidatos = candidatos[candidatos["Type"] == req.anime_type]

    if candidatos.empty:
        return {"perfil": perfil, "recomendaciones": []}

    # 3. Trasladar perfil aprendido del usuario (idéntico a la sección B de Anime.py)
    candidatos["user_mean_rating"] = perfil["user_mean_rating"]
    candidatos["user_std_rating"] = perfil["user_std_rating"]
    candidatos["user_rating_count"] = perfil["user_rating_count"]

    generos_usuario = user_genre_df[user_genre_df["user_id"] == req.user_id][["main_genre", "user_genre_mean"]]
    candidatos = candidatos.merge(generos_usuario, on="main_genre", how="left")
    candidatos["user_genre_mean"] = candidatos["user_genre_mean"].fillna(perfil["user_mean_rating"])

    # 4. Predecir probabilidad con LightGBM
    X_cand = aplicar_categorias_modelo(candidatos[FEATURES].copy())
    candidatos["probabilidad"] = modelo.predict_proba(X_cand)[:, 1]

    top10 = candidatos.sort_values("probabilidad", ascending=False).head(10)

    recomendaciones = []
    for _, row in top10.iterrows():
        prob = float(row["probabilidad"])
        eng = row["English name"] if pd.notna(row["English name"]) else row["Name"]
        studio = row["main_studio"] if pd.notna(row["main_studio"]) else "Desconocido"
        recomendaciones.append({
            "anime_id": int(row["anime_id"]),
            "name": row["Name"],
            "english_name": eng,
            "type": row["Type"] if pd.notna(row["Type"]) else "N/A",
            "episodes": int(row["Episodes"]) if pd.notna(row["Episodes"]) else "?",
            "studio": studio,
            "genre": row["main_genre"],
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A",
            "user_genre_mean_used": round(float(row["user_genre_mean"]), 2),
            "probabilidad": round(prob * 100, 1),
            "affinity_level": calcular_nivel_afinidad(prob)
        })

    return {
        "user_id": req.user_id,
        "perfil": {
            "user_mean_rating": round(float(perfil["user_mean_rating"]), 2),
            "user_std_rating": round(float(perfil["user_std_rating"]), 2),
            "user_rating_count": int(perfil["user_rating_count"]),
            "top_watched": perfil.get("top_watched", "Sin historial")
        },
        "recomendaciones": recomendaciones
    }


@app.get("/", response_class=HTMLResponse)
def web_interface():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Recomendador de Anime - LightGBM</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
            .container { max-width: 960px; margin: auto; background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }
            h1 { color: #38bdf8; margin: 0 0 0.4rem 0; font-size: 1.7rem; }
            p.sub { color: #94a3b8; margin: 0 0 1.5rem 0; }
            .tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; border-bottom: 1px solid #334155; padding-bottom: 0.8rem; }
            .tab-btn { background: #0f172a; color: #94a3b8; border: 1px solid #334155; padding: 0.6rem 1.2rem; border-radius: 8px; cursor: pointer; font-weight: 600; }
            .tab-btn.active { background: #0284c7; color: white; border-color: #0284c7; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; align-items: end; }
            label { display: block; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.3rem; }
            input, select { width: 100%; padding: 0.55rem; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #f8fafc; box-sizing: border-box; }
            .btn-row { display: flex; gap: 0.8rem; margin-top: 1.2rem; }
            button.submit-btn { flex: 2; padding: 0.85rem; background: #0284c7; color: white; border: none; border-radius: 8px; font-size: 1rem; font-weight: bold; cursor: pointer; }
            button.submit-btn:hover { background: #0369a1; }
            button.sec-btn { flex: 1; padding: 0.85rem; background: #334155; color: #f8fafc; border: none; border-radius: 8px; font-size: 0.95rem; font-weight: bold; cursor: pointer; }
            button.sec-btn:hover { background: #475569; }
            .profile-box { margin-top: 1.5rem; padding: 1.1rem 1.3rem; background: #0f172a; border: 1px solid #334155; border-left: 5px solid #a855f7; border-radius: 8px; display: none; }
            .profile-stats { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 0.5rem 0; font-size: 0.92rem; color: #e2e8f0; }
            .watched-list { font-size: 0.85rem; color: #94a3b8; margin-top: 0.4rem; }
            .card-list { display: flex; flex-direction: column; gap: 0.85rem; margin-top: 1.2rem; }
            .anime-card { background: #0f172a; border: 1px solid #334155; border-left: 5px solid #38bdf8; padding: 1rem 1.2rem; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; }
            .anime-title { font-size: 1.08rem; font-weight: bold; color: #f8fafc; margin: 0; }
            .anime-meta { font-size: 0.84rem; color: #94a3b8; margin: 0.3rem 0 0 0; }
            .score-box { text-align: right; min-width: 130px; }
            .score-num { font-size: 1.25rem; font-weight: bold; color: #38bdf8; }
            .badge { display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.8rem; font-weight: bold; margin-top: 0.3rem; }
            .like { background: #166534; color: #bbf7d0; }
            .mid { background: #854d0e; color: #fef08a; }
            .dislike { background: #991b1b; color: #fecaca; }
            #singleResult { margin-top: 1.5rem; padding: 1.2rem; border-radius: 8px; background: #0f172a; border-left: 5px solid #38bdf8; display: none; }
            .error-msg { color: #fca5a5; background: #7f1d1d; padding: 0.8rem; border-radius: 8px; margin-top: 1rem; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sistema de Recomendación Personalizada de Anime (LightGBM)</h1>
            <p class="sub">Selecciona un <b>user_id</b> real de MyAnimeList o evalúa un perfil manual para predecir cuáles animes calificará con &ge; 8.</p>

            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('rec', this)">Recomendador por Usuario Real (Top 10)</button>
                <button class="tab-btn" onclick="switchTab('single', this)">Evaluar Anime Manual + Top 3</button>
            </div>

            <!-- SECCIÓN 1: USUARIO REAL -->
            <div id="tab-rec">
                <div class="grid">
                    <div>
                        <label>Usuarios de muestra (user_id)</label>
                        <select id="userSelect" onchange="document.getElementById('userIdInput').value = this.value">
                            <option value="">Cargando usuarios...</option>
                        </select>
                    </div>
                    <div>
                        <label>O escribe cualquier user_id</label>
                        <input type="number" id="userIdInput" placeholder="Ej. 15" required>
                    </div>
                    <div>
                        <label>Filtrar por Género</label>
                        <select id="rec_genre">
                            <option value="Todos">Todos los géneros</option>
                            <option value="Action">Action</option>
                            <option value="Adventure">Adventure</option>
                            <option value="Comedy">Comedy</option>
                            <option value="Drama">Drama</option>
                            <option value="Fantasy">Fantasy</option>
                            <option value="Mystery">Mystery</option>
                            <option value="Romance">Romance</option>
                            <option value="Sci-Fi">Sci-Fi</option>
                            <option value="Slice of Life">Slice of Life</option>
                            <option value="Sports">Sports</option>
                        </select>
                    </div>
                    <div>
                        <label>Filtrar por Formato</label>
                        <select id="rec_type">
                            <option value="Todos">Todos (TV, Movie, OVA...)</option>
                            <option value="TV">TV (Serie)</option>
                            <option value="Movie">Movie (Película)</option>
                            <option value="OVA">OVA</option>
                        </select>
                    </div>
                </div>

                <div class="btn-row">
                    <button type="button" class="submit-btn" onclick="buscarRecomendaciones()">Predecir Top 10 Animes No Vistos</button>
                    <button type="button" class="sec-btn" onclick="usuarioAleatorio()">Usuario Aleatorio</button>
                </div>

                <div id="errorBox" class="error-msg"></div>

                <div id="profileBox" class="profile-box">
                    <strong style="color:#d8b4fe;" id="profTitle"></strong>
                    <div class="profile-stats" id="profStats"></div>
                    <div class="watched-list" id="profWatched"></div>
                </div>

                <div id="recList" class="card-list"></div>
            </div>

            <!-- SECCIÓN 2: EVALUADOR MANUAL + TOP 3 -->
            <div id="tab-single" style="display:none;">
                <form id="predictForm">
                    <div class="grid">
                        <div><label>Tipo (Type)</label><input type="text" id="Type" value="TV" required></div>
                        <div><label>Episodios</label><input type="number" id="Episodes" value="24" required></div>
                        <div><label>Fuente (Source)</label><input type="text" id="Source" value="Manga" required></div>
                        <div><label>Clasificación (Rating)</label><input type="text" id="Rating" value="PG-13 - Teens 13 or older" required></div>
                        <div><label>Año de estreno</label><input type="number" id="release_year" value="2019" required></div>
                        <div><label>Duración (min)</label><input type="number" id="duration_min" value="24" required></div>
                        <div><label>Estudio principal</label><input type="text" id="main_studio" value="ufotable" required></div>
                        <div><label>Género principal</label><input type="text" id="main_genre" value="Action" required></div>
                        <div><label>Cantidad de géneros</label><input type="number" id="genre_count" value="3" required></div>
                        <div><label>Promedio usuario (1-10)</label><input type="number" step="0.1" id="user_mean_rating" value="8.1" required></div>
                        <div><label>Desviación std usuario</label><input type="number" step="0.01" id="user_std_rating" value="1.15" required></div>
                        <div><label>Total animes vistos</label><input type="number" id="user_rating_count" value="95" required></div>
                        <div><label>Promedio en este género</label><input type="number" step="0.1" id="user_genre_mean" value="8.6" required></div>
                    </div>
                    <button type="submit" class="submit-btn" style="width:100%; margin-top:1.2rem;">Evaluar Anime y Ver Top 3 Similares</button>
                </form>

                <div id="singleResult">
                    <h3 id="resTitulo" style="margin:0 0 0.4rem 0;"></h3>
                    <p id="resProb" style="margin:0.2rem 0;"></p>
                    <span id="resBadge" class="badge"></span>
                </div>

                <div id="manualTop3Section" style="display:none; margin-top:1.5rem;">
                    <h3 style="color:#38bdf8; margin-bottom:0.6rem; font-size:1.1rem;">Top 3 Animes Reales Recomendados para este Perfil</h3>
                    <div id="manualTop3List" class="card-list" style="margin-top:0.5rem;"></div>
                </div>
            </div>
        </div>

        <script>
            async function cargarUsuariosIniciales() {
                const resp = await fetch("/users/info");
                const data = await resp.json();
                const sel = document.getElementById("userSelect");
                sel.innerHTML = "";
                data.sample_users.forEach(uid => {
                    const opt = document.createElement("option");
                    opt.value = uid;
                    opt.text = "Usuario #" + uid;
                    sel.appendChild(opt);
                });
                if (data.sample_users.length > 0) {
                    document.getElementById("userIdInput").value = data.sample_users[0];
                    buscarRecomendaciones();
                }
            }

            async function usuarioAleatorio() {
                const resp = await fetch("/users/info");
                const data = await resp.json();
                if (data.random_user) {
                    document.getElementById("userIdInput").value = data.random_user;
                    buscarRecomendaciones();
                }
            }

            async function buscarRecomendaciones() {
                const uid = parseInt(document.getElementById("userIdInput").value);
                const genre = document.getElementById("rec_genre").value;
                const anime_type = document.getElementById("rec_type").value;
                const errBox = document.getElementById("errorBox");
                errBox.style.display = "none";

                const resp = await fetch("/recommend_user", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ user_id: uid, genre: genre, anime_type: anime_type })
                });

                const data = await resp.json();
                if (!resp.ok) {
                    errBox.innerText = data.detail || "Error al consultar usuario.";
                    errBox.style.display = "block";
                    document.getElementById("profileBox").style.display = "none";
                    document.getElementById("recList").innerHTML = "";
                    return;
                }

                const pBox = document.getElementById("profileBox");
                pBox.style.display = "block";
                document.getElementById("profTitle").innerText = "Perfil Histórico Real — Usuario #" + data.user_id;
                document.getElementById("profStats").innerHTML = `
                    <span><b>Animes calificados:</b> ${data.perfil.user_rating_count}</span>
                    <span><b>Promedio histórico:</b> ${data.perfil.user_mean_rating} / 10</span>
                    <span><b>Exigencia (Std):</b> ${data.perfil.user_std_rating}</span>
                `;
                document.getElementById("profWatched").innerHTML = `<b>Favoritos ya vistos (excluidos):</b> ${data.perfil.top_watched}`;

                const container = document.getElementById("recList");
                container.innerHTML = "";
                data.recomendaciones.forEach((item, idx) => {
                    const cls = item.probabilidad >= 75 ? "like" : (item.probabilidad >= 45 ? "mid" : "dislike");
                    container.innerHTML += `
                        <div class="anime-card">
                            <div>
                                <p class="anime-title">#${idx + 1} ${item.name}</p>
                                <p class="anime-meta">Inglés: ${item.english_name} | Género: <b>${item.genre}</b> (Prom. usuario en género: ${item.user_genre_mean_used}) | Estudio: ${item.studio} | ${item.type} (${item.episodes} eps, ${item.release_year})</p>
                            </div>
                            <div class="score-box">
                                <div class="score-num">${item.probabilidad}%</div>
                                <span class="badge ${cls}">${item.affinity_level}</span>
                            </div>
                        </div>
                    `;
                });
            }

            function switchTab(tab, btn) {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-rec').style.display = tab === 'rec' ? 'block' : 'none';
                document.getElementById('tab-single').style.display = tab === 'single' ? 'block' : 'none';
            }

            document.getElementById("predictForm").addEventListener("submit", async function(e) {
                e.preventDefault();
                const payload = {
                    Type: document.getElementById("Type").value,
                    Episodes: parseInt(document.getElementById("Episodes").value),
                    Source: document.getElementById("Source").value,
                    Rating: document.getElementById("Rating").value,
                    release_year: parseInt(document.getElementById("release_year").value),
                    duration_min: parseInt(document.getElementById("duration_min").value),
                    main_studio: document.getElementById("main_studio").value,
                    main_genre: document.getElementById("main_genre").value,
                    genre_count: parseInt(document.getElementById("genre_count").value),
                    user_mean_rating: parseFloat(document.getElementById("user_mean_rating").value),
                    user_std_rating: parseFloat(document.getElementById("user_std_rating").value),
                    user_rating_count: parseInt(document.getElementById("user_rating_count").value),
                    user_genre_mean: parseFloat(document.getElementById("user_genre_mean").value)
                };
                const resp = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();

                const box = document.getElementById("singleResult");
                const badge = document.getElementById("resBadge");
                box.style.display = "block";
                document.getElementById("resTitulo").innerText = "Predicción del anime evaluado: " + data.resultado;
                document.getElementById("resProb").innerText = "Probabilidad estimada: " + (data.probabilidad_gusto * 100).toFixed(2) + "%";
                badge.innerText = data.affinity_level;
                badge.className = "badge " + (data.user_like_this === 1 ? "like" : "dislike");

                const top3Sec = document.getElementById("manualTop3Section");
                const top3List = document.getElementById("manualTop3List");
                top3List.innerHTML = "";

                if (data.top_3_recomendados && data.top_3_recomendados.length > 0) {
                    top3Sec.style.display = "block";
                    data.top_3_recomendados.forEach((item, idx) => {
                        const cls = item.probabilidad >= 75 ? "like" : (item.probabilidad >= 45 ? "mid" : "dislike");
                        top3List.innerHTML += `
                            <div class="anime-card">
                                <div>
                                    <p class="anime-title">#${idx + 1} ${item.name}</p>
                                    <p class="anime-meta">Inglés: ${item.english_name} | Género: <b>${item.genre}</b> | Estudio: ${item.studio} | ${item.type} (${item.episodes} eps, ${item.release_year})</p>
                                </div>
                                <div class="score-box">
                                    <div class="score-num">${item.probabilidad}%</div>
                                    <span class="badge ${cls}">${item.affinity_level}</span>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    top3Sec.style.display = "none";
                }
            });

            window.onload = cargarUsuariosIniciales;
        </script>
    </body>
    </html>
    """