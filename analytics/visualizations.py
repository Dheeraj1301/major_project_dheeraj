"""Visualization and report helpers for ICU-TwinAI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

VITAL_COLUMNS = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]


def health_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Digital Twin Health Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#25d0ab"},
                "steps": [
                    {"range": [0, 35], "color": "#4d111b"},
                    {"range": [35, 70], "color": "#514412"},
                    {"range": [70, 100], "color": "#113d36"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=300)
    return fig


def trend_graph(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in [c for c in VITAL_COLUMNS if c in df.columns]:
        fig.add_trace(go.Scatter(y=df[column], mode="lines", name=column))
    fig.update_layout(
        template="plotly_dark",
        title="ICU Vital Sign Trends",
        xaxis_title="Sample",
        yaxis_title="Value",
    )
    return fig


def save_correlation_heatmap(
    df: pd.DataFrame, output_path: str | Path = "outputs/correlation_heatmap.png"
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric.corr(), cmap="viridis", annot=False)
    plt.title("ICU Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def save_alert_badge(
    level: str, output_path: str | Path = "outputs/alert_badge.png"
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "STABLE": (52, 184, 121),
        "WARNING": (25, 170, 230),
        "CRITICAL": (47, 47, 220),
    }
    image = np.zeros((180, 520, 3), dtype=np.uint8)
    image[:] = (22, 28, 36)
    cv2.rectangle(image, (22, 42), (498, 138), colors.get(level, (255, 255, 255)), -1)
    cv2.putText(
        image,
        f"ICU ALERT: {level}",
        (45, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), image)
    return output


def write_analytics_report(
    df: pd.DataFrame,
    metrics: Dict[str, object] | None = None,
    output_path: str | Path = "reports/icu_analytics_report.txt",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ICU-TwinAI Analytics Report",
        "=" * 32,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
    ]
    available = [c for c in VITAL_COLUMNS if c in df.columns]
    if available:
        lines.append("\nVital Summary:")
        lines.append(df[available].describe().round(2).to_string())
    if metrics:
        lines.append("\nML Performance:")
        for name, result in metrics.items():
            lines.append(f"{name}: accuracy={result['accuracy']:.3f}")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
