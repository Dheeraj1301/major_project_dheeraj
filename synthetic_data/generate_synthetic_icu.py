"""Synthetic ICU data generator with noise, drift, missingness, and emergency spikes."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()


def _risk_score(hr, spo2, sbp, rr, temp, glucose, stress):
    risk = (
        np.maximum(hr - 100, 0) * 0.7
        + np.maximum(94 - spo2, 0) * 6.5
        + np.maximum(90 - sbp, 0) * 0.8
        + np.maximum(sbp - 150, 0) * 0.5
        + np.maximum(rr - 24, 0) * 2.5
        + np.abs(temp - 37.0) * 8
        + np.maximum(glucose - 160, 0) * 0.08
        + stress * 20
    )
    return np.clip(risk, 0, 100)


def generate_synthetic_icu_data(rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = pd.date_range("2026-01-01", periods=rows, freq="min")
    drift = np.linspace(0, 1.8, rows)
    patient_ids = [f"ICU-{rng.integers(1000, 9999)}" for _ in range(rows)]

    heart_rate = rng.normal(84, 14, rows) + drift * rng.choice([-1, 1], rows)
    spo2 = rng.normal(96, 2.2, rows) - drift * 0.8
    systolic_bp = rng.normal(122, 18, rows) + rng.normal(0, 4, rows)
    diastolic_bp = rng.normal(76, 10, rows)
    respiratory_rate = rng.normal(18, 4, rows) + drift * 0.6
    temperature = rng.normal(37.0, 0.55, rows) + rng.normal(0, 0.08, rows)
    ecg_variability = np.abs(rng.normal(42, 14, rows))
    glucose = rng.normal(122, 33, rows)
    stress_index = np.clip(rng.beta(2, 5, rows) + drift / 8, 0, 1)

    spike_count = max(1, rows // 100)
    spike_idx = rng.choice(rows, spike_count, replace=False)
    heart_rate[spike_idx] += rng.normal(38, 8, spike_count)
    spo2[spike_idx] -= rng.normal(8, 2, spike_count)
    respiratory_rate[spike_idx] += rng.normal(10, 3, spike_count)
    temperature[spike_idx] += rng.normal(1.2, 0.3, spike_count)
    stress_index[spike_idx] = np.clip(stress_index[spike_idx] + 0.45, 0, 1)

    risk = _risk_score(
        heart_rate,
        spo2,
        systolic_bp,
        respiratory_rate,
        temperature,
        glucose,
        stress_index,
    )
    critical = (risk >= 65).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": time,
            "patient_id": patient_ids,
            "patient_name": [fake.name() for _ in range(rows)],
            "age": rng.integers(18, 91, rows),
            "heart_rate": np.round(heart_rate, 1),
            "spo2": np.round(spo2, 1),
            "systolic_bp": np.round(systolic_bp, 1),
            "diastolic_bp": np.round(diastolic_bp, 1),
            "respiratory_rate": np.round(respiratory_rate, 1),
            "temperature": np.round(temperature, 2),
            "ecg_variability": np.round(ecg_variability, 2),
            "glucose": np.round(glucose, 1),
            "stress_index": np.round(stress_index, 3),
            "icu_risk_score": np.round(risk, 2),
            "critical_alert": critical,
        }
    )

    missing_mask = rng.random((rows, 7)) < 0.012
    vital_cols = [
        "heart_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "temperature",
        "glucose",
    ]
    for i, col in enumerate(vital_cols):
        df.loc[missing_mask[:, i], col] = np.nan
    return df


def save_synthetic_data(
    rows: int = 100_000,
    output_path: str | Path = "synthetic_data/synthetic_icu_data.csv",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_icu_data(rows=rows).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    path = save_synthetic_data()
    print(f"Saved synthetic ICU data to {path}")
