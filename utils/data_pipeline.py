"""Dataset ingestion, schema detection, and cleaning utilities for ICU-TwinAI."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

VITAL_ALIASES: Dict[str, Iterable[str]] = {
    "heart_rate": ["heart_rate", "hr", "pulse", "bpm"],
    "spo2": ["spo2", "oxygen", "oxygen_saturation", "o2sat", "sao2"],
    "systolic_bp": ["systolic_bp", "sbp", "sysbp", "blood_pressure_systolic"],
    "diastolic_bp": ["diastolic_bp", "dbp", "diasbp", "blood_pressure_diastolic"],
    "respiratory_rate": ["respiratory_rate", "resp_rate", "rr", "respiration"],
    "temperature": ["temperature", "temp", "body_temperature"],
    "glucose": ["glucose", "blood_glucose", "avg_glucose_level"],
}

NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "heart_rate": (45, 130),
    "spo2": (85, 100),
    "systolic_bp": (80, 180),
    "diastolic_bp": (45, 120),
    "respiratory_rate": (8, 35),
    "temperature": (34.5, 40.5),
    "glucose": (60, 260),
}


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    detected_vitals: Dict[str, str]
    outlier_counts: Dict[str, int]


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load CSV data from a file path or Streamlit uploaded file."""
    return pd.read_csv(path_or_buffer)


def detect_vital_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map canonical ICU vital names to dataset columns using flexible aliases."""
    lower_lookup = {str(col).strip().lower(): col for col in df.columns}
    detected: Dict[str, str] = {}
    for canonical, aliases in VITAL_ALIASES.items():
        for alias in aliases:
            if alias in lower_lookup:
                detected[canonical] = lower_lookup[alias]
                break
    return detected


def outlier_counts(df: pd.DataFrame, detected: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for canonical, column in detected.items():
        if column in df and pd.api.types.is_numeric_dtype(df[column]):
            low, high = NORMAL_RANGES[canonical]
            counts[column] = int(((df[column] < low) | (df[column] > high)).sum())
    return counts


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in df.columns if c not in numeric_columns]
    detected = detect_vital_columns(df)
    return DatasetProfile(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_values={c: int(v) for c, v in df.isna().sum().items()},
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        detected_vitals=detected,
        outlier_counts=outlier_counts(df, detected),
    )


def remove_sensor_noise(
    df: pd.DataFrame, columns: Iterable[str], window: int = 5
) -> pd.DataFrame:
    """Apply rolling median smoothing to numeric vital columns."""
    cleaned = df.copy()
    for column in columns:
        if column in cleaned and pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = (
                cleaned[column]
                .rolling(window=window, min_periods=1, center=True)
                .median()
            )
    return cleaned


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run a beginner-friendly but realistic cleaning workflow."""
    before = profile_dataset(df)
    cleaned = df.copy().drop_duplicates().reset_index(drop=True)

    numeric_columns = cleaned.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in cleaned.columns if c not in numeric_columns]

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    for column in categorical_columns:
        mode = cleaned[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        cleaned[column] = cleaned[column].fillna(fill_value)

    detected = detect_vital_columns(cleaned)
    for canonical, column in detected.items():
        if column in numeric_columns:
            low, high = NORMAL_RANGES[canonical]
            cleaned[column] = cleaned[column].clip(lower=low, upper=high)
    cleaned = remove_sensor_noise(cleaned, detected.values())

    for column in categorical_columns:
        cleaned[column] = LabelEncoder().fit_transform(cleaned[column].astype(str))

    if numeric_columns:
        cleaned[numeric_columns] = StandardScaler().fit_transform(
            cleaned[numeric_columns]
        )

    after = profile_dataset(cleaned)
    report = {
        "before": asdict(before),
        "after": asdict(after),
        "rows_removed": before.rows - after.rows,
        "missing_values_filled": sum(before.missing_values.values())
        - sum(after.missing_values.values()),
        "encoded_categorical_columns": categorical_columns,
        "normalized_numeric_columns": numeric_columns,
    }
    return cleaned, report


def save_cleaning_report(
    report: Dict[str, object], output_path: str | Path = "reports/cleaning_report.txt"
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ICU-TwinAI Cleaning Summary", "=" * 32]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
