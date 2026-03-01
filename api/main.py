import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.preprocess import preprocess

MODEL_PATH = 'models/model.pkl'
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title='Heart Disease Predictor', lifespan=lifespan)


class PatientFeatures(BaseModel):
    Age: int                     = Field(..., ge=1, le=120)
    Sex: int                     = Field(..., ge=0, le=1)
    Chest_pain_type: int         = Field(..., alias='Chest pain type', ge=1, le=4)
    BP: int                      = Field(..., ge=0)
    Cholesterol: int             = Field(..., ge=0)
    FBS_over_120: int            = Field(..., alias='FBS over 120', ge=0, le=1)
    EKG_results: int             = Field(..., alias='EKG results', ge=0, le=2)
    Max_HR: int                  = Field(..., alias='Max HR', ge=0)
    Exercise_angina: int         = Field(..., alias='Exercise angina', ge=0, le=1)
    ST_depression: float         = Field(..., alias='ST depression', ge=0.0)
    Slope_of_ST: int             = Field(..., alias='Slope of ST', ge=1, le=3)
    Number_of_vessels_fluro: int = Field(..., alias='Number of vessels fluro', ge=0, le=3)
    Thallium: int                = Field(..., ge=0)

    model_config = {'populate_by_name': True}


@app.post('/predict')
def predict(patient: PatientFeatures):
    input_df = pd.DataFrame([patient.model_dump(by_alias=True)])
    X = preprocess(input_df)
    probability = float(model.predict_proba(X)[0, 1])
    return {
        'probability': probability,
        'prediction': 'Presence' if probability >= 0.5 else 'Absence',
    }
