"""
SWE402 Assignment - Task 2d: Model Deployment
Diabetes Risk Prediction API (FastAPI)

Loads the trained scikit-learn pipeline (imputer -> scaler -> logistic regression)
and exposes it as a REST API so external systems (e.g. the n8n agentic workflow)
can request predictions programmatically.
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Columns where a value of 0 is physiologically impossible and means "missing".
# We convert them to NaN so the pipeline's median imputer handles them,
# exactly as during training.
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

app = FastAPI(
    title='Diabetes Risk Prediction API',
    description='SWE402 Data Mining assignment - predicts 5-year diabetes risk '
                'from routine clinical measurements.',
    version='1.0.0',
)

model = joblib.load('diabetes_model.pkl')


class Patient(BaseModel):
    """One patient's clinical measurements (raw, unscaled values)."""
    Pregnancies: float = Field(ge=0, description='Number of pregnancies', examples=[2])
    Glucose: float = Field(ge=0, description='Plasma glucose concentration (mg/dL)', examples=[168])
    BloodPressure: float = Field(ge=0, description='Diastolic blood pressure (mm Hg)', examples=[74])
    SkinThickness: float = Field(ge=0, description='Triceps skin fold thickness (mm)', examples=[33])
    Insulin: float = Field(ge=0, description='2-hour serum insulin (mu U/ml)', examples=[120])
    BMI: float = Field(ge=0, description='Body mass index', examples=[38.0])
    DiabetesPedigreeFunction: float = Field(ge=0, description='Family history score', examples=[0.53])
    Age: float = Field(ge=0, description='Age in years', examples=[45])


def risk_level(probability: float) -> str:
    """Map probability to a coarse risk band the AI agent can reason over."""
    if probability >= 0.65:
        return 'High'
    if probability >= 0.35:
        return 'Medium'
    return 'Low'


@app.get('/')
def health():
    """Health check - used to verify the service is up."""
    return {'status': 'ok', 'service': 'diabetes-risk-api', 'model_loaded': model is not None}


@app.post('/predict')
def predict(patient: Patient):
    """Predict diabetes risk for a single patient."""
    data = pd.DataFrame([patient.model_dump()])[FEATURES]
    # Treat impossible zeros as missing, mirroring the training preprocessing
    data[ZERO_AS_MISSING] = data[ZERO_AS_MISSING].replace(0, np.nan)

    probability = float(model.predict_proba(data)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        'prediction': prediction,
        'prediction_label': 'Diabetes' if prediction == 1 else 'No Diabetes',
        'probability': round(probability, 4),
        'risk_level': risk_level(probability),
    }
