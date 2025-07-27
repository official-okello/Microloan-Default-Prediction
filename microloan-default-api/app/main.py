from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

from app.model_utils import model, engineer_features
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Microloan Default Prediction API")

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request schema
class LoanApplicant(BaseModel):
    monthly_income: float
    utility_payment_timeliness: str
    has_previous_loan: int
    gender: str
    loan_amount: float

@app.post("/predict")
def predict_default(applicants: List[LoanApplicant]):
    # Convert to DataFrame
    df = pd.DataFrame([a.dict() for a in applicants])

    # Feature engineering
    df_transformed = engineer_features(df)

    # Predict
    preds = model.predict(df_transformed)
    return {"predictions": preds.tolist()}


# To run the FastAPI application, use the command:
# pip install fastapi uvicorn
# 'uvicorn app.main:app --reload'
# then visit  http://127.0.0.1:8000/docs