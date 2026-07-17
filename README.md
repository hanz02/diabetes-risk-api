# Diabetes Risk Screening — End-to-End Machine Learning Web App

Predict a patient's 5-year Type 2 diabetes risk from eight routine clinical measurements. This project covers the full machine learning lifecycle: data cleaning, exploratory analysis, model training and evaluation, deployment as a REST API, a browser-based frontend, and integration into an autonomous AI workflow.

## 🔗 Live Demo

**Try the web app here:** <!-- 👉 PASTE YOUR RENDER LIVE LINK BELOW 👈 -->

> **[ ADD YOUR RENDER URL HERE ]** — e.g. `https://diabetes-risk-api.onrender.com`

| Resource                          | Link                         |
| --------------------------------- | ---------------------------- |
| 🌐 Web app                        | `[ your Render URL ]/`       |
| 📘 Interactive API docs (Swagger) | `[ your Render URL ]/docs`   |
| ❤️ Service health check           | `[ your Render URL ]/health` |

> Note: the app is hosted on a free tier that sleeps after periods of inactivity. The first request after a while can take up to ~50 seconds while the service wakes up.

---

## Overview

Early detection of Type 2 diabetes dramatically reduces the risk of serious complications, but manual screening does not scale in busy clinics. This project builds an intelligent screening tool that takes a patient's measurements, predicts their diabetes risk with a trained machine learning model, and presents the result through a clean web interface and a programmatic API.

The same prediction API also powers an **autonomous agentic AI workflow** (built in n8n) that reads new patient records, calls the model, reasons over the result with a large language model, and automatically emails an alert to the care team for high-risk patients — all with no human in the loop.

## Key Features

- **Trained ML model** — a Logistic Regression pipeline selected via 5-fold cross-validation, achieving a test ROC-AUC of **0.81**.
- **Honest data preparation** — detects and repairs physiologically impossible values (e.g. a recorded blood pressure of zero) that are really disguised missing data.
- **REST API** — a FastAPI service with a `/predict` endpoint returning a prediction, probability, and Low / Medium / High risk band.
- **Web frontend** — a responsive single-page interface to enter measurements and view results, served by the same service.
- **Cloud deployment** — hosted on Render, callable from anywhere.
- **Agentic AI integration** — an n8n workflow that automates screening from data ingestion to notification.
- **Auto-generated API docs** — interactive Swagger UI at `/docs`.

## Architecture

```
 Web Frontend  ─┐
                ├──►  FastAPI service  ──►  Trained ML pipeline (.pkl)
 n8n workflow  ─┘        (Render)            imputer → scaler → model
                             │
                             ├─ GET  /         web app
                             ├─ GET  /health   health check
                             └─ POST /predict  risk prediction
```

The full agentic workflow adds five layers around the API: a **Trigger** (scheduled run), **Data Ingestion** (Google Sheets), **Prediction** (this API), an **AI Agent** (Google Gemini reasoning over the result), and an **Action/Output** layer (automated Gmail alert and sheet update).

## Tech Stack

| Area       | Tools                                                                                |
| ---------- | ------------------------------------------------------------------------------------ |
| Language   | Python 3.11                                                                          |
| Data & ML  | pandas, NumPy, scikit-learn                                                          |
| Model      | Logistic Regression (with median imputation + standard scaling in a single pipeline) |
| API        | FastAPI, Uvicorn, Pydantic                                                           |
| Frontend   | HTML, CSS, vanilla JavaScript (single file, no build step)                           |
| Deployment | Render (cloud), joblib (model serialization)                                         |
| Automation | n8n, Google Gemini, Google Sheets, Gmail                                             |

## The Dataset

The model is trained on the **Pima Indians Diabetes Database** (768 records, 8 features, binary outcome), a public dataset from the U.S. National Institute of Diabetes and Digestive and Kidney Diseases.

| Feature                  | Description                           |
| ------------------------ | ------------------------------------- |
| Pregnancies              | Number of times pregnant              |
| Glucose                  | Plasma glucose concentration (mg/dL)  |
| BloodPressure            | Diastolic blood pressure (mm Hg)      |
| SkinThickness            | Triceps skin fold thickness (mm)      |
| Insulin                  | 2-hour serum insulin (mu U/ml)        |
| BMI                      | Body mass index                       |
| DiabetesPedigreeFunction | Family-history-based likelihood score |
| Age                      | Age in years                          |

## Model Development Summary

- **Data cleaning:** impossible zero values in five columns were converted to missing and imputed with the column median. Features were standardized. All preprocessing lives inside a scikit-learn `Pipeline` to prevent data leakage during cross-validation.
- **Model selection:** Logistic Regression, Random Forest, and Gradient Boosting were compared with 5-fold stratified cross-validation, scored by ROC-AUC. Logistic Regression won (CV ROC-AUC 0.844) and was chosen for its performance and interpretability.
- **Test performance:** Accuracy 0.73, Recall 0.70, F1 0.65, ROC-AUC 0.81 on a held-out test set. Class weighting was used so the model favours catching true cases, which matters more than avoiding false alarms in a screening context.
- **Top predictors:** Glucose, BMI, and Pregnancies — consistent with clinical knowledge.

## API Reference

### `POST /predict`

Request body:

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

Response:

```json
{
  "prediction": 1,
  "prediction_label": "Diabetes",
  "probability": 0.8992,
  "risk_level": "High"
}
```

### `GET /health`

```json
{ "status": "ok", "service": "diabetes-risk-api", "model_loaded": true }
```

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --reload

# 3. Open the app
#    Web UI:   http://127.0.0.1:8000/
#    API docs: http://127.0.0.1:8000/docs
```

## Deploy to Render

1. Push this repository to GitHub.
2. On Render, create a **New Web Service** and connect the repo.
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Deploy, then paste your live URL into the **Live Demo** section at the top of this README.

## Project Structure

```
.
├── app.py               # FastAPI service: serves the web app + prediction API
├── index.html           # Web frontend (single file, no build step)
├── diabetes_model.pkl   # Trained scikit-learn pipeline
├── requirements.txt     # Python dependencies (scikit-learn pinned to training version)
├── DEPLOYMENT_GUIDE.md  # Step-by-step deployment notes
└── README.md
```

## What This Project Demonstrates

This is a complete, production-style machine learning project rather than a notebook experiment. It shows the ability to:

- Take a real, messy dataset and prepare it correctly, including catching disguised missing data that naive analysis would miss.
- Build a leakage-free modelling pipeline and select a model with proper cross-validation and appropriate metrics.
- Ship the model as a live, documented REST API and a usable web application.
- Integrate the model into an automated, agentic AI workflow that turns predictions into real actions.
- Reason about limitations and trade-offs (class imbalance, free-tier constraints, model generalisability) rather than just reporting a single accuracy number.

## Limitations

This is an academic screening aid and **not** a diagnostic tool. The dataset covers only adult female patients of Pima Indian heritage, so predictions may not generalise to other populations without retraining. A real clinical deployment would require broader data, regulatory approval, and a human in the loop.

## License

Released for educational and portfolio use.

## Author

**Han Zhe Khaw** — built as part of a Data Mining course project covering the full ML lifecycle from data preparation to deployment and agentic automation.
