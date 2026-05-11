"""Machine-learning training and prediction utilities for ICU clinical decision support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
    "glucose",
    "stress_index",
]


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for feature in FEATURES:
        if feature not in frame.columns:
            frame[feature] = 0.0
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(
            frame[feature].median()
        )
    if "critical_alert" not in frame.columns:
        risk = (
            np.maximum(frame["heart_rate"] - 105, 0)
            + np.maximum(94 - frame["spo2"], 0) * 5
            + np.maximum(frame["respiratory_rate"] - 24, 0) * 2
            + np.maximum(frame["temperature"] - 38, 0) * 8
        )
        frame["critical_alert"] = (risk > 25).astype(int)
    return frame


def train_models(
    df: pd.DataFrame, output_dir: str | Path = "models"
) -> Dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = prepare_training_frame(df)
    X = frame[FEATURES]
    y = frame["critical_alert"].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=120, random_state=42, class_weight="balanced"
        ),
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
        ),
    }
    metrics: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
        joblib.dump(model, output / f"{name}.joblib")
    return metrics


def predict_patient_risk(
    vitals: Dict[str, float], model_path: str | Path = "models/random_forest.joblib"
) -> Dict[str, float | str]:
    model = joblib.load(model_path)
    row = pd.DataFrame([{feature: vitals.get(feature, 0.0) for feature in FEATURES}])
    probability = (
        float(model.predict_proba(row)[0][1])
        if hasattr(model, "predict_proba")
        else float(model.predict(row)[0])
    )
    label = (
        "CRITICAL"
        if probability >= 0.65
        else "WATCH"
        if probability >= 0.35
        else "STABLE"
    )
    return {"critical_probability": round(probability, 4), "decision_state": label}
