# DHT11 Monitoring — single Flask server

This project uses one Python process to receive ESP32 readings, write the same reading to MySQL and Firebase Realtime Database, host the website, preprocess data, and run MLR, MLP, and TabNet models. There is no Node.js runtime or browser-side model code.

## Run it

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Edit `.env` with the MySQL credentials, Firebase RTDB URL, and Firebase Admin service-account path. Set `MYSQL_ENABLED=true`, `FIREBASE_ENABLED=true`, and `DATABASES_REQUIRED=true` for the complete dual-database demo.
4. Start the server:

   ```powershell
   python app.py
   ```

5. Open `http://localhost:8080`. Replace `YOUR_LAPTOP_IPV4` in `lab3esp32_firmware.ino`, then upload it to the ESP32.

The MySQL table is created automatically. Never place the Firebase service account in `static/` or a webpage.

## API

| Method and endpoint | Purpose |
| --- | --- |
| `GET /` | Main live monitor and model switcher |
| `GET /mlr` | Multiple Linear Regression interface |
| `GET /mlp` | Multilayer Perceptron interface |
| `GET /third-model` | TabNet interface |
| `POST /data` | Validate and ingest an ESP32 reading |
| `GET /api/readings` | Read buffered/MySQL data (`limit`, `start`, `end`) |
| `POST /api/preprocess` | Drop, filter, smooth, normalize, or standardize readings |
| `POST /api/mlr/analyse` | Fit MLR and return equation, metrics, validation, and forecast |
| `POST /api/mlp/train` | Train an MLP and return a `model_id` |
| `POST /api/mlp/predict` | Forecast with a trained MLP |
| `POST /api/third-model/train` | Train TabNet and return a `model_id` |
| `POST /api/third-model/predict` | Forecast with trained TabNet |
| `GET /api/export/csv` | Download readings as CSV |
| `GET /api/health` | Inspect server and database readiness |

Minimal ESP32 payload:

```json
{"temperature": 27, "humidity": 64}
```

In offline/demo mode (`DATABASES_REQUIRED=false`), disabled database steps are reported in the response and the validated reading is still buffered. In the final database demonstration, use `DATABASES_REQUIRED=true`; ingestion then becomes visible to the website only after both writes succeed.

## Preprocessing request

Operations are executed in the order `drop_missing`, `remove_duplicates`, `iqr`, `zscore`, `moving_average`, `normalize`, `standardize`, regardless of their array order. Send `readings` to process an explicit dataset or omit it to use the live buffer.

```json
{
  "operations": ["drop_missing", "remove_duplicates", "iqr", "moving_average"],
  "window": 3
}
```

MLR and model training accept `target` (`temperature` or `humidity`) and `lags`. Model state is held in process memory, so restart means retraining.

## Deploy on Render

Create a Render **Web Service** connected to this project's GitHub repository and use:

```text
Language: Python
Build command: pip install -r requirements.txt
Start command: gunicorn app:app --workers 1 --threads 4 --timeout 120
Plan: Free
```

Copy the variables from `.env.render.example` into the Render service's Environment settings, replacing `FIREBASE_DATABASE_URL` with the exact Realtime Database URL. Under **Environment > Secret Files**, create a file named `service-account.json` containing the complete Firebase service-account JSON. Render exposes it to the app at `/etc/secrets/service-account.json`.

The ESP32 must send JSON readings to:

```text
https://YOUR_RENDER_SERVICE.onrender.com/data
```

Use port `443` with `WiFiClientSecure`. The repository ignore rules exclude `.env`, virtual environments, caches, and Firebase service-account JSON files; never commit those secrets.
