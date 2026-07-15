# Task 2d Deployment Guide — Diabetes Risk API on Render

## What we are doing (the concept)

Your trained model currently lives in a file (`diabetes_model.pkl`) on your laptop. Nothing else can use it there. Deployment means putting it behind a **web API**: a program running on a server that listens for requests. Anyone (including your n8n workflow) can send patient measurements to a URL and get back a risk score as JSON. FastAPI is the Python framework that wraps the model; Render is the free cloud host that runs it 24/7.

## Step 1 — Create the GitHub repository (no coding needed)

1. Go to github.com → click **+** (top right) → **New repository**.
2. Name: `diabetes-risk-api`. Visibility: **Public** (Render free tier needs to read it). Tick **Add a README file**. Click **Create repository**.
3. On the repo page: **Add file → Upload files**.
4. Drag in these THREE files:
   - `app.py` (provided)
   - `requirements.txt` (provided)
   - `diabetes_model.pkl` (the file you downloaded from Colab)
5. Click **Commit changes**.

> Why requirements.txt matters: it tells the server exactly which library versions to install. scikit-learn is pinned to **1.6.1** — the version that trained your model. A different version might refuse to load the .pkl file.

## Step 2 — Deploy on Render

1. Go to dashboard.render.com → **New → Web Service**.
2. Connect your GitHub account if asked, then select the `diabetes-risk-api` repo.
3. Fill the form:
   - **Name:** `diabetes-risk-api` (this becomes part of your URL)
   - **Region:** Singapore (closest to Malaysia)
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Before clicking deploy, open **Advanced** (or Environment) and add an environment variable:
   - Key: `PYTHON_VERSION`  Value: `3.11.9`
   (This matches Colab's Python closely and avoids pickle compatibility surprises.)
5. Click **Deploy Web Service** and watch the log. First build takes 3–6 minutes. Success looks like: `Uvicorn running on http://0.0.0.0:10000` and a green "Live" badge.
6. Copy your service URL, e.g. `https://diabetes-risk-api.onrender.com`

## Step 3 — Test it (also your report screenshots)

1. **Health check:** open `https://YOUR-URL.onrender.com/` in a browser. You should see: `{"status":"ok","service":"diabetes-risk-api","model_loaded":true}`
2. **Interactive docs:** open `https://YOUR-URL.onrender.com/docs` — FastAPI auto-generates a test page. Expand **POST /predict** → **Try it out** → use this body → **Execute**:

```json
{
  "Pregnancies": 2,
  "Glucose": 168,
  "BloodPressure": 74,
  "SkinThickness": 33,
  "Insulin": 120,
  "BMI": 38.0,
  "DiabetesPedigreeFunction": 0.53,
  "Age": 45
}
```

Expected response shape:

```json
{
  "prediction": 1,
  "prediction_label": "Diabetes",
  "probability": 0.83,
  "risk_level": "High"
}
```

3. Also test a healthy-looking patient (e.g. Glucose 95, BMI 24, Age 23) and confirm you get `"risk_level": "Low"`.
4. **Screenshot for the report:** the /docs page showing a successful request + response (this is your "sample API endpoints" evidence), and the Render dashboard showing the service Live.

## API summary (for the report)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check — confirms service and model are loaded |
| POST | `/predict` | Accepts 8 clinical measurements (JSON), returns prediction, probability, and Low/Medium/High risk level |
| GET | `/docs` | Auto-generated interactive API documentation (Swagger UI) |

## Known limitation (write this in the report)

Render's free tier puts the service to sleep after ~15 minutes without traffic; the next request takes ~50 seconds while it wakes. Harmless for this assignment — the n8n workflow simply waits — but a production system would use a paid always-on instance.
