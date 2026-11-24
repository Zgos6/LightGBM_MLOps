from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid():
    payload = {
        "Type": "TV",
        "Episodes": 24,
        "Source": "Manga",
        "Rating": "PG-13 - Teens 13 or older",
        "release_year": 2019,
        "duration_min": 24,
        "main_studio": "ufotable",
        "main_genre": "Action",
        "genre_count": 3,
        "user_mean_rating": 8.1,
        "user_std_rating": 1.15,
        "user_rating_count": 95,
        "user_genre_mean": 8.6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_like_this"] in [0, 1]
    assert "affinity_level" in data
    assert "top_3_recomendados" in data


def test_predict_invalid_data_returns_422():
    payload_invalido = {
        "Type": "TV",
        "Episodes": "dato_incorrecto",
        "Source": "Manga",
        "Rating": "PG-13",
        "release_year": 2019,
        "duration_min": 24,
        "main_studio": "ufotable",
        "main_genre": "Action",
        "genre_count": 3,
        "user_mean_rating": 8.1,
        "user_std_rating": 1.15,
        "user_rating_count": 95,
        "user_genre_mean": 8.6
    }
    response = client.post("/predict", json=payload_invalido)
    assert response.status_code == 422