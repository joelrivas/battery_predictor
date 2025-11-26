"""API for battery life prediction using a pre-trained model."""

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

class BatteryInput(BaseModel):
    last_battery: float
    mean_battery: float
    min_battery: float
    max_battery: float
    count_events: float
    std_battery: float


model = joblib.load("models/battery_model.pkl")
print(model.feature_name_)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: BatteryInput):
    """Make a prediction using the pre-trained model."""
    
    if model is None:
        return {"error": "Model not loaded"}
    
    data = pd.DataFrame([payload.dict()])
    prediction = model.predict(data)
    return {"prediction": float(prediction[0])}
