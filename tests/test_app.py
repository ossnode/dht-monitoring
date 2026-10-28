import math

import pytest

import app as application


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    application.READINGS.clear()
    application.MODELS.clear()
    monkeypatch.setattr(application.MYSQL, "enabled", False)
    monkeypatch.setattr(application.MYSQL, "pool", None)
    monkeypatch.setattr(application.FIREBASE, "enabled", False)
    monkeypatch.setattr(application.FIREBASE, "reference", None)
    monkeypatch.setenv("DATABASES_REQUIRED", "false")
    yield


@pytest.fixture()
def client():
    application.app.config.update(TESTING=True)
    return application.app.test_client()


def seed(client, count=50):
    for i in range(count):
        response = client.post(
            "/data",
            json={"temperature": 24 + math.sin(i / 5) + i * 0.03, "humidity": 62 - i * 0.08 + math.cos(i / 6)},
        )
        assert response.status_code == 201


def test_pages_are_served(client):
    for route in ("/", "/mlr", "/mlp", "/third-model"):
        assert client.get(route).status_code == 200


def test_ingest_validates_and_buffers(client):
    bad = client.post("/data", json={"temperature": 20, "humidity": 120})
    assert bad.status_code == 400
    good = client.post("/data", json={"Temperature": 25, "Humidity": 60})
    assert good.status_code == 201
    body = client.get("/api/readings").get_json()
    assert body["count"] == 1
    assert body["readings"][0]["temperature"] == 25


def test_preprocessing(client):
    seed(client, 10)
    result = client.post("/api/preprocess", json={"operations": ["moving_average"], "window": 3})
    assert result.status_code == 200
    assert result.get_json()["summary"]["output"] == 10


def test_mlr_and_mlp(client):
    seed(client)
    mlr = client.post("/api/mlr/analyse", json={"target": "temperature", "lags": 3, "steps": 4})
    assert mlr.status_code == 200
    assert len(mlr.get_json()["future_predictions"]) == 4
    assert "equation" in mlr.get_json()

    trained = client.post("/api/mlp/train", json={"target": "humidity", "lags": 4, "epochs": 50, "hidden_layers": [8]})
    assert trained.status_code == 200
    model_id = trained.get_json()["model_id"]
    prediction = client.post("/api/mlp/predict", json={"model_id": model_id, "steps": 3})
    assert prediction.status_code == 200
    assert len(prediction.get_json()["predictions"]) == 3


def test_csv_export(client):
    seed(client, 3)
    response = client.get("/api/export/csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "temperature,humidity" in response.get_data(as_text=True).splitlines()[0]
