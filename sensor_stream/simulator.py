"""IoMT edge sensor stream simulator for ICU vitals."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import random
from typing import Dict, Generator


@dataclass
class SensorReading:
    timestamp: str
    patient_id: str
    heart_rate: float
    spo2: float
    systolic_bp: float
    diastolic_bp: float
    respiratory_rate: float
    temperature: float
    alert_level: str
    alert_message: str


class IOMTSensorSimulator:
    def __init__(self, patient_id: str = "ICU-DEMO-001", seed: int = 7):
        self.patient_id = patient_id
        self.random = random.Random(seed)
        self.baseline = {
            "heart_rate": 82.0,
            "spo2": 97.0,
            "systolic_bp": 122.0,
            "diastolic_bp": 76.0,
            "respiratory_rate": 18.0,
            "temperature": 37.0,
        }

    def _next_value(self, key: str, volatility: float) -> float:
        self.baseline[key] += self.random.uniform(-volatility, volatility)
        return self.baseline[key]

    def _inject_event(self, values: Dict[str, float]) -> Dict[str, float]:
        if self.random.random() < 0.08:
            values["heart_rate"] += self.random.uniform(22, 48)
            values["spo2"] -= self.random.uniform(4, 11)
            values["respiratory_rate"] += self.random.uniform(5, 13)
        return values

    @staticmethod
    def classify_alert(values: Dict[str, float]) -> tuple[str, str]:
        if (
            values["spo2"] < 90
            or values["heart_rate"] > 135
            or values["systolic_bp"] < 85
        ):
            return "CRITICAL", "Immediate clinician review required"
        if (
            values["spo2"] < 94
            or values["heart_rate"] > 115
            or values["temperature"] > 38.4
        ):
            return "WARNING", "Patient trending toward instability"
        return "STABLE", "Vitals within monitored range"

    def next_reading(self) -> SensorReading:
        values = {
            "heart_rate": self._next_value("heart_rate", 3.0),
            "spo2": self._next_value("spo2", 0.8),
            "systolic_bp": self._next_value("systolic_bp", 4.0),
            "diastolic_bp": self._next_value("diastolic_bp", 2.5),
            "respiratory_rate": self._next_value("respiratory_rate", 1.1),
            "temperature": self._next_value("temperature", 0.12),
        }
        values = self._inject_event(values)
        level, message = self.classify_alert(values)
        rounded = {k: round(v, 2) for k, v in values.items()}
        return SensorReading(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            patient_id=self.patient_id,
            alert_level=level,
            alert_message=message,
            **rounded,
        )

    def stream(self) -> Generator[Dict[str, object], None, None]:
        while True:
            yield asdict(self.next_reading())
