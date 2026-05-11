"""Streamlit dashboard for ICU-TwinAI."""
# ruff: noqa: E402

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from analytics.ml_models import predict_patient_risk, train_models  # noqa: E402
from analytics.visualizations import (
    health_gauge,
    save_alert_badge,
    save_correlation_heatmap,
    trend_graph,
    write_analytics_report,
)  # noqa: E402
from digital_twin.patient_twin import ICUPatientTwin, PatientProfile  # noqa: E402
from sensor_stream.simulator import IOMTSensorSimulator  # noqa: E402
from synthetic_data.generate_synthetic_icu import (
    generate_synthetic_icu_data,
    save_synthetic_data,
)  # noqa: E402
from utils.data_pipeline import clean_dataset, profile_dataset, save_cleaning_report  # noqa: E402

st.set_page_config(page_title="ICU-TwinAI", page_icon="🫀", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #08111f 0%, #102b3f 48%, #071017 100%); color: #e8f6ff; }
    [data-testid="stSidebar"] { background-color: #07111f; }
    .metric-card { padding: 1rem; border-radius: 18px; background: rgba(37, 208, 171, 0.12); border: 1px solid rgba(37, 208, 171, 0.35); }
    .hero { padding: 1.3rem; border-radius: 24px; background: linear-gradient(90deg, rgba(37,208,171,.22), rgba(27,121,226,.18)); border: 1px solid rgba(255,255,255,.16); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def demo_data(rows: int = 1200) -> pd.DataFrame:
    return generate_synthetic_icu_data(rows=rows)


def render_profile(profile) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", profile.rows)
    c2.metric("Columns", profile.columns)
    c3.metric("Duplicates", profile.duplicate_rows)
    c4.metric("Detected vitals", len(profile.detected_vitals))
    st.json(profile.__dict__)


page = st.sidebar.radio(
    "Navigation",
    [
        "Home dashboard",
        "Dataset upload",
        "Data cleaning simulation",
        "Synthetic data generator",
        "Live ICU monitoring",
        "Digital twin simulation",
        "AI prediction dashboard",
        "Reports page",
    ],
)

st.sidebar.success("Cloud-edge ICU simulation active")
st.sidebar.caption(
    "Educational clinical decision support prototype. Not for real medical use."
)

if page == "Home dashboard":
    st.markdown(
        '<div class="hero"><h1>ICU-TwinAI</h1><h3>Cloud-Edge Digital Twin Framework for ICU Simulation & Clinical Decision Support</h3></div>',
        unsafe_allow_html=True,
    )
    df = demo_data()
    latest = df.iloc[-1]
    cols = st.columns(5)
    cols[0].metric("Heart rate", f"{latest.heart_rate:.1f} bpm")
    cols[1].metric("SpO₂", f"{latest.spo2:.1f}%")
    cols[2].metric("BP", f"{latest.systolic_bp:.0f}/{latest.diastolic_bp:.0f}")
    cols[3].metric("Risk", f"{latest.icu_risk_score:.1f}%")
    cols[4].metric("Alerts", int(df.critical_alert.sum()))
    st.plotly_chart(trend_graph(df.tail(180)), use_container_width=True)
    st.plotly_chart(
        health_gauge(100 - float(latest.icu_risk_score)), use_container_width=True
    )

elif page == "Dataset upload":
    st.header("CSV Dataset Pipeline")
    uploaded = st.file_uploader(
        "Upload ICU, MIMIC-style, or healthcare stroke CSV", type=["csv"]
    )
    df = pd.read_csv(uploaded) if uploaded else demo_data(600)
    st.dataframe(df.head(25), use_container_width=True)
    render_profile(profile_dataset(df))

elif page == "Data cleaning simulation":
    st.header("Cleaning & Preprocessing Simulation")
    uploaded = st.file_uploader(
        "Upload CSV for cleaning", type=["csv"], key="clean_upload"
    )
    df = pd.read_csv(uploaded) if uploaded else demo_data(700)
    cleaned, report = clean_dataset(df)
    save_cleaning_report(report)
    col1, col2 = st.columns(2)
    col1.subheader("Before")
    col1.write(report["before"])
    col2.subheader("After")
    col2.write(report["after"])
    st.dataframe(cleaned.head(25), use_container_width=True)
    st.success("Cleaning report saved to reports/cleaning_report.txt")

elif page == "Synthetic data generator":
    st.header("Large Synthetic ICU Dataset Generator")
    rows = st.slider("Rows", 1_000, 100_000, 10_000, step=1_000)
    if st.button("Generate synthetic ICU CSV"):
        path = save_synthetic_data(rows=rows)
        st.success(f"Saved {rows:,} rows to {path}")
    st.dataframe(demo_data(500).head(20), use_container_width=True)

elif page == "Live ICU monitoring":
    st.header("IoMT Sensor Streaming Simulation")
    simulator = IOMTSensorSimulator()
    samples = st.slider("Stream samples", 10, 120, 35)
    readings = []
    placeholder = st.empty()
    if st.button("Start simulated stream"):
        for _ in range(samples):
            reading = simulator.next_reading().__dict__
            readings.append(reading)
            with placeholder.container():
                st.metric("Alert", reading["alert_level"], reading["alert_message"])
                st.plotly_chart(
                    trend_graph(pd.DataFrame(readings)), use_container_width=True
                )
            time.sleep(0.03)
    else:
        readings = [simulator.next_reading().__dict__ for _ in range(samples)]
        st.plotly_chart(trend_graph(pd.DataFrame(readings)), use_container_width=True)

elif page == "Digital twin simulation":
    st.header("Virtual Patient Digital Twin")
    age = st.slider("Patient age", 18, 95, 62)
    profile = PatientProfile(
        patient_id="ICU-TWIN-001",
        age=age,
        comorbidity_index=st.slider("Comorbidity index", 0.0, 1.0, 0.35),
    )
    vitals = {
        "heart_rate": st.slider("Heart rate", 40, 170, 88),
        "spo2": st.slider("SpO₂", 70, 100, 96),
        "systolic_bp": st.slider("Systolic BP", 70, 210, 124),
        "respiratory_rate": st.slider("Respiratory rate", 6, 45, 20),
        "temperature": st.slider("Temperature", 34.0, 41.5, 37.2),
    }
    twin = ICUPatientTwin(profile)
    state = twin.update(vitals)
    st.plotly_chart(health_gauge(state["health_score"]), use_container_width=True)
    st.json(state)
    st.dataframe(
        pd.DataFrame(twin.simulate_progression(vitals, steps=24)),
        use_container_width=True,
    )

elif page == "AI prediction dashboard":
    st.header("AI Clinical Decision Support")
    df = demo_data(4000)
    if st.button("Train RandomForest + LogisticRegression"):
        metrics = train_models(df)
        write_analytics_report(df, metrics)
        st.success(
            "Models saved to models/ and ML report saved to reports/icu_analytics_report.txt"
        )
        st.json(metrics)
    vitals = {
        "heart_rate": st.number_input("Heart rate", value=112.0),
        "spo2": st.number_input("SpO₂", value=92.0),
        "systolic_bp": st.number_input("Systolic BP", value=104.0),
        "diastolic_bp": st.number_input("Diastolic BP", value=68.0),
        "respiratory_rate": st.number_input("Respiratory rate", value=25.0),
        "temperature": st.number_input("Temperature", value=38.2),
        "glucose": st.number_input("Glucose", value=166.0),
        "stress_index": st.number_input("Stress index", value=0.72),
    }
    model_path = PROJECT_ROOT / "models/random_forest.joblib"
    if model_path.exists():
        st.json(predict_patient_risk(vitals, model_path))
    else:
        st.info("Train models first to enable live prediction.")
    st.plotly_chart(
        px.histogram(
            df, x="icu_risk_score", color="critical_alert", template="plotly_dark"
        ),
        use_container_width=True,
    )

elif page == "Reports page":
    st.header("Reports & Visual Outputs")
    df = demo_data(1200)
    heatmap = save_correlation_heatmap(df)
    alert = save_alert_badge("WARNING")
    report = write_analytics_report(df)
    st.success("Generated report artifacts")
    st.write(
        {
            "analytics_report": str(report),
            "correlation_heatmap": str(heatmap),
            "alert_badge": str(alert),
        }
    )
    st.image(str(heatmap), caption="Correlation heatmap")
    st.image(str(alert), caption="OpenCV generated alert badge")
