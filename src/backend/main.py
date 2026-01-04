from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import joblib

from src.core.loader import load_data
from src.core.engineer import FeatureEngineer
from src.core.trainer import train_model
from xgboost import XGBClassifier

app = FastAPI(title="Feature Engineering Playground API")

ARTIFACTS = {
    "model": None,
    "features": [],
    "config": {}
}

class ExperimentRequest(BaseModel):
    use_polynomials: bool
    polynomial_cols: list[str] = ["tenure", "MonthlyCharges"]
    use_interaction: bool
    use_binning: bool

class PredictionRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str

@app.post("/run_experiment")
def run_experiment(config: ExperimentRequest):
    df_raw = load_data()
    engine = FeatureEngineer(df_raw)
    
    if config.use_binning: engine.bin_tenure()
    if config.use_interaction: engine.add_interaction("tenure", "MonthlyCharges")
    if config.use_polynomials: engine.add_polynomial(config.polynomial_cols)
    engine.process_categorical()
    
    df_processed = engine.get_data()
    metrics, importance = train_model(df_processed)
    
    X = df_processed.drop(columns=["Churn"])
    y = df_processed["Churn"]
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    ARTIFACTS["model"] = model
    ARTIFACTS["features"] = list(X.columns)
    ARTIFACTS["config"] = config.dict()
    
    return {"status": "success", "metrics": metrics, "top_features": importance}

@app.post("/predict_single")
def predict_single(data: PredictionRequest):
    if ARTIFACTS["model"] is None:
        raise HTTPException(status_code=400, detail="Please run an experiment first!")

    input_data = pd.DataFrame([data.dict()])
    
    config = ARTIFACTS["config"]
    engine = FeatureEngineer(input_data)
    
    if config["use_binning"]: engine.bin_tenure()
    if config["use_interaction"]: engine.add_interaction("tenure", "MonthlyCharges")
    if config["use_polynomials"]: engine.add_polynomial(config["polynomial_cols"])
    
    engine.process_categorical()
    
    engine.align_features(ARTIFACTS["features"])
    
    final_df = engine.get_data()
    
    prediction = ARTIFACTS["model"].predict(final_df)[0]
    probability = ARTIFACTS["model"].predict_proba(final_df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }