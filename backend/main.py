"""FastAPI backend for ICU-TwinAI."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from analytics.ml_models import predict_patient_risk, train_models
from digital_twin.patient_twin import ICUPatientTwin, PatientProfile
from sensor_stream.simulator import IOMTSensorSimulator
from synthetic_data.generate_synthetic_icu import save_synthetic_data
from utils.data_pipeline import (
    clean_dataset,
    load_csv,
    profile_dataset,
    save_cleaning_report,
)

app = FastAPI(title="ICU-TwinAI API", version="1.0.0")
simulator = IOMTSensorSimulator()


@app.get("/")
def root():
    return {
        "project": "ICU-TwinAI",
        "status": "online",
        "modules": ["data", "synthetic", "stream", "digital_twin", "ml"],
    }


@app.post("/dataset/profile")
async def dataset_profile(file: UploadFile = File(...)):
    df = load_csv(file.file)
    return profile_dataset(df).__dict__


@app.post("/dataset/clean")
async def dataset_clean(file: UploadFile = File(...)):
    df = load_csv(file.file)
    cleaned, report = clean_dataset(df)
    Path("outputs").mkdir(exist_ok=True)
    cleaned_path = Path("outputs/cleaned_dataset.csv")
    cleaned.to_csv(cleaned_path, index=False)
    save_cleaning_report(report)
    return {"cleaned_path": str(cleaned_path), "report": report}


@app.post("/synthetic/generate")
def generate_synthetic(rows: int = 100_000):
    path = save_synthetic_data(rows=rows)
    return {"path": str(path), "rows": rows}


@app.get("/stream/next")
def next_stream_reading():
    return simulator.next_reading().__dict__


@app.post("/digital-twin/simulate")
def simulate_twin(vitals: dict, age: int = 58):
    twin = ICUPatientTwin(PatientProfile(patient_id="API-PATIENT", age=age))
    return twin.update(vitals)


@app.post("/ml/train")
async def ml_train(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    df = pd.read_csv(tmp_path)
    metrics = train_models(df)
    return JSONResponse(metrics)


@app.post("/ml/predict")
def ml_predict(vitals: dict):
    return predict_patient_risk(vitals)
