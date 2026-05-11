"""Virtual patient digital twin for risk progression and health state simulation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np


@dataclass
class PatientProfile:
    patient_id: str
    age: int
    diagnosis: str = "Acute respiratory monitoring"
    comorbidity_index: float = 0.25


class ICUPatientTwin:
    def __init__(self, profile: PatientProfile):
        self.profile = profile
        self.history: List[Dict[str, float | str]] = []

    @staticmethod
    def compute_health_score(
        vitals: Dict[str, float], age: int, comorbidity_index: float
    ) -> Dict[str, float | str]:
        penalties = [
            max(vitals.get("heart_rate", 80) - 100, 0) * 0.35,
            max(94 - vitals.get("spo2", 98), 0) * 4.5,
            max(90 - vitals.get("systolic_bp", 120), 0) * 0.45,
            max(vitals.get("respiratory_rate", 18) - 22, 0) * 1.9,
            abs(vitals.get("temperature", 37) - 37.0) * 4.2,
            max(age - 65, 0) * 0.08,
            comorbidity_index * 12,
        ]
        risk = float(np.clip(sum(penalties), 0, 100))
        health_score = float(np.clip(100 - risk, 0, 100))
        stability_index = float(
            np.clip((health_score / 100) * (vitals.get("spo2", 98) / 100), 0, 1)
        )
        if risk >= 65:
            status = "CRITICAL"
        elif risk >= 35:
            status = "WATCH"
        else:
            status = "STABLE"
        return {
            "health_score": round(health_score, 2),
            "risk_score": round(risk, 2),
            "stability_index": round(stability_index, 3),
            "status": status,
        }

    def update(self, vitals: Dict[str, float]) -> Dict[str, object]:
        state = self.compute_health_score(
            vitals, self.profile.age, self.profile.comorbidity_index
        )
        snapshot = {**asdict(self.profile), **vitals, **state}
        self.history.append(snapshot)
        return snapshot

    def simulate_progression(
        self, initial_vitals: Dict[str, float], steps: int = 24
    ) -> List[Dict[str, object]]:
        rng = np.random.default_rng(12)
        vitals = initial_vitals.copy()
        results = []
        for _ in range(steps):
            vitals["heart_rate"] = float(
                vitals.get("heart_rate", 82) + rng.normal(0.8, 2.2)
            )
            vitals["spo2"] = float(vitals.get("spo2", 97) + rng.normal(-0.08, 0.55))
            vitals["systolic_bp"] = float(
                vitals.get("systolic_bp", 122) + rng.normal(0, 3.0)
            )
            vitals["respiratory_rate"] = float(
                vitals.get("respiratory_rate", 18) + rng.normal(0.1, 0.9)
            )
            vitals["temperature"] = float(
                vitals.get("temperature", 37) + rng.normal(0.02, 0.08)
            )
            results.append(self.update(vitals.copy()))
        return results
