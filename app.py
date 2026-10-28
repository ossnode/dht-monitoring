"""Single Flask server for the ESP32 DHT11 monitoring demo.

The application owns ingestion, both database connections, preprocessing, and
all three model implementations.  Browser JavaScript is deliberately limited
to API calls and rendering.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/data": {"origins": "*"}})

BUFFER_SIZE = int(os.getenv("LIVE_BUFFER_SIZE", "5000"))
READINGS: deque[dict[str, Any]] = deque(maxlen=BUFFER_SIZE)
READINGS_LOCK = threading.RLock()
MODELS: dict[str, dict[str, Any]] = {}
MODELS_LOCK = threading.RLock()


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def json_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [json_safe(v) for v in list(value)]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


@dataclass
class StorageResult:
    enabled: bool
    ok: bool
    detail: str


class MySQLStore:
    def __init__(self) -> None:
        self.enabled = env_bool("MYSQL_ENABLED")
        self.required = env_bool("DATABASES_REQUIRED")
        self.pool = None
        self.error = "disabled"
        self.table = os.getenv("MYSQL_TABLE", "readings")
        if not self.table.replace("_", "").isalnum():
            raise RuntimeError("MYSQL_TABLE may contain only letters, numbers, and underscores")
        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            from mysql.connector.pooling import MySQLConnectionPool

            self.pool = MySQLConnectionPool(
                pool_name=f"dht_{uuid.uuid4().hex[:8]}",
                pool_size=int(os.getenv("MYSQL_POOL_SIZE", "5")),
                host=os.getenv("MYSQL_HOST", "127.0.0.1"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "dht_monitoring"),
                autocommit=False,
            )
            with self.pool.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    f"""CREATE TABLE IF NOT EXISTS `{self.table}` (
                    id CHAR(36) NOT NULL PRIMARY KEY,
                    timestamp_ms BIGINT NOT NULL,
                    recorded_at DATETIME(3) NOT NULL,
                    temperature DOUBLE NOT NULL,
                    humidity DOUBLE NOT NULL,
                    INDEX idx_timestamp_ms (timestamp_ms)
                    ) ENGINE=InnoDB"""
                )
                connection.commit()
                cursor.close()
            self.error = ""
        except Exception as exc:  # startup remains usable in offline demo mode
            self.pool = None
            self.error = str(exc)
            app.logger.warning("MySQL unavailable: %s", exc)

    def write(self, row: dict[str, Any]) -> StorageResult:
        if not self.enabled:
            return StorageResult(False, not self.required, "disabled")
        if self.pool is None:
            return StorageResult(True, False, self.error or "not connected")
        try:
            with self.pool.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    f"INSERT INTO `{self.table}` "
                    "(id, timestamp_ms, recorded_at, temperature, humidity) VALUES (%s,%s,%s,%s,%s)",
                    (
                        row["id"],
                        row["timestamp_ms"],
                        datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc).replace(tzinfo=None),
                        row["temperature"],
                        row["humidity"],
                    ),
                )
                connection.commit()
                cursor.close()
            return StorageResult(True, True, "saved")
        except Exception as exc:
            app.logger.exception("MySQL write failed")
            return StorageResult(True, False, str(exc))

    def read(self, limit: int, start: int | None, end: int | None) -> list[dict[str, Any]]:
        if self.pool is None:
            return []
        clauses, params = [], []
        if start is not None:
            clauses.append("timestamp_ms >= %s")
            params.append(start)
        if end is not None:
            clauses.append("timestamp_ms <= %s")
            params.append(end)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT id,timestamp_ms,temperature,humidity FROM `{self.table}`"
            f"{where} ORDER BY timestamp_ms DESC LIMIT %s"
        )
        params.append(limit)
        with self.pool.get_connection() as connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params)
            rows = list(reversed(cursor.fetchall()))
            cursor.close()
        return [canonical_row(row) for row in rows]


class FirebaseStore:
    def __init__(self) -> None:
        self.enabled = env_bool("FIREBASE_ENABLED")
        self.required = env_bool("DATABASES_REQUIRED")
        self.reference = None
        self.error = "disabled"
        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import firebase_admin
            from firebase_admin import credentials, db

            if not firebase_admin._apps:
                credential_json = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
                credential_file = os.getenv("FIREBASE_CREDENTIALS_FILE", "").strip()
                if credential_json:
                    credential = credentials.Certificate(json.loads(credential_json))
                elif credential_file:
                    credential = credentials.Certificate(credential_file)
                else:
                    credential = credentials.ApplicationDefault()
                firebase_admin.initialize_app(
                    credential, {"databaseURL": os.environ["FIREBASE_DATABASE_URL"]}
                )
            path = os.getenv("FIREBASE_PATH", "readings").strip("/") or "readings"
            self.reference = db.reference(path)
            self.error = ""
        except Exception as exc:
            self.reference = None
            self.error = str(exc)
            app.logger.warning("Firebase unavailable: %s", exc)

    def write(self, row: dict[str, Any]) -> StorageResult:
        if not self.enabled:
            return StorageResult(False, not self.required, "disabled")
        if self.reference is None:
            return StorageResult(True, False, self.error or "not connected")
        try:
            self.reference.child(row["id"]).set(row)
            return StorageResult(True, True, "saved")
        except Exception as exc:
            app.logger.exception("Firebase write failed")
            return StorageResult(True, False, str(exc))


MYSQL = MySQLStore()
FIREBASE = FirebaseStore()


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    timestamp_ms = int(row.get("timestamp_ms", row.get("time", row.get("t", 0))))
    return {
        "id": str(row.get("id", "")),
        "timestamp_ms": timestamp_ms,
        "timestamp": datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat(),
        "temperature": float(row.get("temperature", row.get("Temperature"))),
        "humidity": float(row.get("humidity", row.get("Humidity"))),
    }


def request_rows(payload: dict[str, Any], minimum: int = 3) -> list[dict[str, Any]]:
    raw = payload.get("readings")
    if raw is None:
        rows = []
        if MYSQL.pool is not None:
            try:
                rows = MYSQL.read(10000, None, None)
            except Exception as exc:
                app.logger.warning("Model data read failed; using live buffer: %s", exc)
        if not rows:
            with READINGS_LOCK:
                rows = list(READINGS)
    elif not isinstance(raw, list):
        raise ValueError("readings must be an array")
    else:
        rows = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"readings[{index}] must be an object")
            rows.append(canonical_row(item))
    if len(rows) < minimum:
        raise ValueError(f"At least {minimum} readings are required")
    return sorted(rows, key=lambda row: row["timestamp_ms"])


def series_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame[["id", "timestamp_ms", "temperature", "humidity"]].copy()
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="coerce")
    frame["humidity"] = pd.to_numeric(frame["humidity"], errors="coerce")
    return frame.sort_values("timestamp_ms").reset_index(drop=True)


def supervised(rows: list[dict[str, Any]], target: str, lags: int) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[float]]]:
    if target not in {"temperature", "humidity"}:
        raise ValueError("target must be 'temperature' or 'humidity'")
    if not 1 <= lags <= 24:
        raise ValueError("lags must be between 1 and 24")
    frame = series_frame(rows).dropna(subset=["temperature", "humidity"])
    if len(frame) < lags + 5:
        raise ValueError(f"At least {lags + 5} complete readings are required")
    other = "humidity" if target == "temperature" else "temperature"
    target_values = frame[target].to_numpy(dtype=float)
    other_values = frame[other].to_numpy(dtype=float)
    samples, labels = [], []
    names = [f"{target}_lag_{i}" for i in range(1, lags + 1)] + [
        f"{other}_lag_{i}" for i in range(1, lags + 1)
    ]
    for index in range(lags, len(frame)):
        samples.append(
            [target_values[index - i] for i in range(1, lags + 1)]
            + [other_values[index - i] for i in range(1, lags + 1)]
        )
        labels.append(target_values[index])
    windows = {
        "target": target_values[-lags:].tolist(),
        "other": other_values[-lags:].tolist(),
    }
    return np.asarray(samples), np.asarray(labels), names, windows


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    nonzero = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    return json_safe(
        {
            "mae": mean_absolute_error(actual, predicted),
            "rmse": math.sqrt(mean_squared_error(actual, predicted)),
            "r2": r2_score(actual, predicted) if len(actual) > 1 else None,
            "mape_percent": mape,
        }
    )


def split_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = min(max(int(len(x) * 0.8), 1), len(x) - 1)
    return x[:split], x[split:], y[:split], y[split:]


def iterative_predict(model: Any, scaler_x: Any, scaler_y: Any, windows: dict[str, list[float]], lags: int, steps: int) -> list[float]:
    target_window = list(windows["target"])
    other_window = list(windows["other"])
    predictions: list[float] = []
    for _ in range(steps):
        features = np.asarray(
            [target_window[-i] for i in range(1, lags + 1)]
            + [other_window[-i] for i in range(1, lags + 1)],
            dtype=float,
        ).reshape(1, -1)
        transformed = scaler_x.transform(features) if scaler_x is not None else features
        value = np.asarray(model.predict(transformed)).reshape(-1, 1)
        if scaler_y is not None:
            value = scaler_y.inverse_transform(value)
        prediction = float(value.ravel()[0])
        predictions.append(prediction)
        target_window.append(prediction)
        other_window.append(other_window[-1])
    return predictions


def model_response(model_id: str, kind: str, target: str, features: list[str], metrics: dict[str, Any], actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return json_safe(
        {
            "ok": True,
            "model_id": model_id,
            "model": kind,
            "target": target,
            "features": features,
            "metrics": metrics,
            "validation": {"actual": actual, "predicted": predicted},
        }
    )


@app.errorhandler(ValueError)
def invalid_input(exc: ValueError):
    return jsonify(ok=False, error=str(exc)), 400


@app.errorhandler(404)
def not_found(_exc: Any):
    return jsonify(ok=False, error="Not found"), 404


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/mlr")
def mlr_page():
    return render_template("mlr.html")


@app.get("/mlp")
def mlp_page():
    return render_template("mlp.html")


@app.get("/third-model")
def third_page():
    return render_template("third_model.html")


@app.get("/api/health")
def health():
    return jsonify(
        ok=True,
        buffer_size=len(READINGS),
        mysql={"enabled": MYSQL.enabled, "ready": MYSQL.pool is not None, "error": MYSQL.error},
        firebase={"enabled": FIREBASE.enabled, "ready": FIREBASE.reference is not None, "error": FIREBASE.error},
    )


@app.post("/data")
def ingest():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("A JSON object is required")
    temperature = json_number(payload.get("temperature", payload.get("Temperature")), "temperature")
    humidity = json_number(payload.get("humidity", payload.get("Humidity")), "humidity")
    if not -40 <= temperature <= 80:
        raise ValueError("temperature must be between -40 and 80 °C")
    if not 0 <= humidity <= 100:
        raise ValueError("humidity must be between 0 and 100%")
    now = datetime.now(timezone.utc)
    row = {
        "id": str(uuid.uuid4()),
        "timestamp_ms": int(now.timestamp() * 1000),
        "timestamp": now.isoformat(),
        "temperature": temperature,
        "humidity": humidity,
    }

    # Required order: MySQL, Firebase RTDB, live buffer, then HTTP visibility.
    mysql_result = MYSQL.write(row)
    firebase_result = FIREBASE.write(row)
    storage_ok = mysql_result.ok and firebase_result.ok
    if storage_ok or not env_bool("DATABASES_REQUIRED"):
        with READINGS_LOCK:
            READINGS.append(row)
    status = 201 if storage_ok else 503
    if not env_bool("DATABASES_REQUIRED"):
        status = 201
    return (
        jsonify(
            ok=storage_ok or not env_bool("DATABASES_REQUIRED"),
            reading=row,
            storage={
                "mysql": mysql_result.__dict__,
                "firebase": firebase_result.__dict__,
            },
        ),
        status,
    )


def filtered_buffer(limit: int, start: int | None, end: int | None) -> list[dict[str, Any]]:
    with READINGS_LOCK:
        rows = list(READINGS)
    if start is not None:
        rows = [row for row in rows if row["timestamp_ms"] >= start]
    if end is not None:
        rows = [row for row in rows if row["timestamp_ms"] <= end]
    return rows[-limit:]


@app.get("/api/readings")
def readings():
    limit = min(max(request.args.get("limit", 500, type=int), 1), 10000)
    start = request.args.get("start", type=int)
    end = request.args.get("end", type=int)
    source = request.args.get("source", "auto")
    rows: list[dict[str, Any]] = []
    used = "buffer"
    if source in {"auto", "mysql"} and MYSQL.pool is not None:
        try:
            rows = MYSQL.read(limit, start, end)
            used = "mysql"
        except Exception as exc:
            if source == "mysql":
                return jsonify(ok=False, error=str(exc)), 503
            app.logger.warning("MySQL read failed; using buffer: %s", exc)
    if not rows and source != "mysql":
        rows = filtered_buffer(limit, start, end)
        used = "buffer"
    return jsonify(ok=True, source=used, count=len(rows), readings=rows)


@app.post("/api/preprocess")
def preprocess():
    payload = request.get_json(silent=True) or {}
    rows = request_rows(payload, minimum=1)
    operations = payload.get("operations", ["drop_missing", "remove_duplicates"])
    if not isinstance(operations, list):
        raise ValueError("operations must be an array")
    frame = series_frame(rows)
    original = len(frame)
    notes: list[str] = []
    allowed = {"drop_missing", "remove_duplicates", "iqr", "zscore", "moving_average", "normalize", "standardize"}
    unknown = set(operations) - allowed
    if unknown:
        raise ValueError(f"Unknown preprocessing operation(s): {', '.join(sorted(unknown))}")
    if "drop_missing" in operations:
        frame = frame.dropna(subset=["temperature", "humidity"])
    if "remove_duplicates" in operations:
        frame = frame.drop_duplicates(subset=["timestamp_ms", "temperature", "humidity"])
    if "iqr" in operations and len(frame) >= 4:
        for column in ("temperature", "humidity"):
            q1, q3 = frame[column].quantile([0.25, 0.75])
            spread = q3 - q1
            frame = frame[frame[column].between(q1 - 1.5 * spread, q3 + 1.5 * spread)]
    if "zscore" in operations and len(frame) >= 3:
        numeric = frame[["temperature", "humidity"]]
        std = numeric.std(ddof=0).replace(0, 1)
        frame = frame[((numeric - numeric.mean()).abs() / std <= 3).all(axis=1)]
    window = min(max(int(payload.get("window", 3)), 1), 100)
    if "moving_average" in operations:
        frame[["temperature", "humidity"]] = frame[["temperature", "humidity"]].rolling(window, min_periods=1).mean()
        notes.append(f"Applied a {window}-reading moving average")
    if "normalize" in operations:
        numeric = frame[["temperature", "humidity"]]
        ranges = (numeric.max() - numeric.min()).replace(0, 1)
        frame[["temperature", "humidity"]] = (numeric - numeric.min()) / ranges
        notes.append("Values are now on a 0–1 scale")
    if "standardize" in operations:
        numeric = frame[["temperature", "humidity"]]
        std = numeric.std(ddof=0).replace(0, 1)
        frame[["temperature", "humidity"]] = (numeric - numeric.mean()) / std
        notes.append("Values are now z-scores")
    output = []
    for record in frame.to_dict("records"):
        record["timestamp"] = datetime.fromtimestamp(record["timestamp_ms"] / 1000, timezone.utc).isoformat()
        output.append(record)
    return jsonify(ok=True, summary={"input": original, "output": len(output), "removed": original - len(output), "notes": notes}, readings=json_safe(output))


@app.post("/api/mlr/analyse")
def mlr_analyse():
    payload = request.get_json(silent=True) or {}
    target = payload.get("target", "temperature")
    lags = int(payload.get("lags", 3))
    steps = min(max(int(payload.get("steps", 1)), 1), 100)
    rows = request_rows(payload, minimum=lags + 5)
    x, y, names, windows = supervised(rows, target, lags)
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = LinearRegression().fit(x_train, y_train)
    predicted = model.predict(x_test)
    equation_parts = [f"({coefficient:.6g} × {name})" for name, coefficient in zip(names, model.coef_)]
    future = iterative_predict(model, None, None, windows, lags, steps)
    response = model_response("mlr", "Multiple Linear Regression", target, names, metric_values(y_test, predicted), y_test, predicted)
    response.update(
        json_safe(
            {
                "equation": f"{target} = {model.intercept_:.6g} + " + " + ".join(equation_parts),
                "intercept": model.intercept_,
                "coefficients": dict(zip(names, model.coef_)),
                "future_predictions": future,
            }
        )
    )
    return jsonify(response)


@app.post("/api/mlp/train")
def mlp_train():
    payload = request.get_json(silent=True) or {}
    target = payload.get("target", "temperature")
    lags = int(payload.get("lags", 5))
    rows = request_rows(payload, minimum=lags + 5)
    x, y, names, windows = supervised(rows, target, lags)
    x_train, x_test, y_train, y_test = split_data(x, y)
    scaler = StandardScaler().fit(x_train)
    hidden = payload.get("hidden_layers", [32, 16])
    if not isinstance(hidden, list) or not hidden or len(hidden) > 5:
        raise ValueError("hidden_layers must be an array containing 1 to 5 layer sizes")
    hidden_tuple = tuple(min(max(int(size), 1), 512) for size in hidden)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_tuple,
        activation=payload.get("activation", "relu"),
        solver="adam",
        learning_rate_init=float(payload.get("learning_rate", 0.001)),
        max_iter=min(max(int(payload.get("epochs", 500)), 10), 5000),
        early_stopping=len(x_train) >= 20,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(scaler.transform(x_train), y_train)
    predicted = model.predict(scaler.transform(x_test))
    model_id = uuid.uuid4().hex
    with MODELS_LOCK:
        MODELS[model_id] = {"kind": "mlp", "model": model, "scaler_x": scaler, "scaler_y": None, "windows": windows, "lags": lags, "target": target}
    response = model_response(model_id, "Multilayer Perceptron", target, names, metric_values(y_test, predicted), y_test, predicted)
    response.update({"epochs_completed": model.n_iter_, "loss": json_safe(model.loss_), "hidden_layers": hidden_tuple})
    return jsonify(response)


def saved_model(payload: dict[str, Any], expected: str) -> tuple[str, dict[str, Any]]:
    model_id = str(payload.get("model_id", ""))
    with MODELS_LOCK:
        entry = MODELS.get(model_id)
    if not entry or entry["kind"] != expected:
        raise ValueError(f"Unknown {expected} model_id; train the model first")
    return model_id, entry


@app.post("/api/mlp/predict")
def mlp_predict():
    payload = request.get_json(silent=True) or {}
    model_id, entry = saved_model(payload, "mlp")
    steps = min(max(int(payload.get("steps", 1)), 1), 100)
    predictions = iterative_predict(entry["model"], entry["scaler_x"], None, entry["windows"], entry["lags"], steps)
    return jsonify(ok=True, model_id=model_id, target=entry["target"], predictions=json_safe(predictions))


@app.post("/api/third-model/train")
def third_train():
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        return jsonify(ok=False, error="TabNet is not installed. Run: pip install pytorch-tabnet torch"), 503
    payload = request.get_json(silent=True) or {}
    target = payload.get("target", "temperature")
    lags = int(payload.get("lags", 5))
    rows = request_rows(payload, minimum=lags + 8)
    x, y, names, windows = supervised(rows, target, lags)
    x_train, x_test, y_train, y_test = split_data(x, y)
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    x_train_scaled, x_test_scaled = scaler_x.transform(x_train).astype(np.float32), scaler_x.transform(x_test).astype(np.float32)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).astype(np.float32)
    model = TabNetRegressor(
        n_d=min(max(int(payload.get("decision_width", 8)), 4), 64),
        n_a=min(max(int(payload.get("attention_width", 8)), 4), 64),
        n_steps=min(max(int(payload.get("attention_steps", 3)), 1), 10),
        seed=42,
        verbose=0,
    )
    model.fit(
        x_train_scaled,
        y_train_scaled,
        eval_set=[(x_test_scaled, scaler_y.transform(y_test.reshape(-1, 1)).astype(np.float32))],
        eval_name=["validation"],
        eval_metric=["rmse"],
        max_epochs=min(max(int(payload.get("epochs", 100)), 10), 1000),
        patience=min(max(int(payload.get("patience", 20)), 3), 100),
        batch_size=min(max(int(payload.get("batch_size", 64)), 8), 2048),
        virtual_batch_size=min(max(int(payload.get("virtual_batch_size", 16)), 4), 256),
    )
    predicted_scaled = model.predict(x_test_scaled)
    predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
    model_id = uuid.uuid4().hex
    with MODELS_LOCK:
        MODELS[model_id] = {"kind": "tabnet", "model": model, "scaler_x": scaler_x, "scaler_y": scaler_y, "windows": windows, "lags": lags, "target": target}
    response = model_response(model_id, "TabNet", target, names, metric_values(y_test, predicted), y_test, predicted)
    response["feature_importance"] = json_safe(dict(zip(names, model.feature_importances_)))
    return jsonify(response)


@app.post("/api/third-model/predict")
def third_predict():
    payload = request.get_json(silent=True) or {}
    model_id, entry = saved_model(payload, "tabnet")
    steps = min(max(int(payload.get("steps", 1)), 1), 100)
    predictions = iterative_predict(entry["model"], entry["scaler_x"], entry["scaler_y"], entry["windows"], entry["lags"], steps)
    return jsonify(ok=True, model_id=model_id, target=entry["target"], predictions=json_safe(predictions))


@app.get("/api/export/csv")
def export_csv():
    limit = min(max(request.args.get("limit", 10000, type=int), 1), 100000)
    rows = []
    if MYSQL.pool is not None:
        try:
            rows = MYSQL.read(limit, None, None)
        except Exception as exc:
            app.logger.warning("CSV MySQL read failed; using buffer: %s", exc)
    if not rows:
        rows = filtered_buffer(limit, None, None)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["id", "timestamp_ms", "timestamp", "temperature", "humidity"])
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        stream.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=dht11-readings.csv"},
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        debug=env_bool("FLASK_DEBUG"),
        threaded=True,
    )
