# ICU-TwinAI

**Cloud-Edge Digital Twin Framework for ICU Simulation and Clinical Decision Support**

ICU-TwinAI is a modular AI/ML + Big Data healthcare prototype that looks and behaves like an industrial ICU monitoring platform while remaining beginner-friendly and fully runnable in VS Code.

> Educational prototype only. It is not a medical device and must not be used for real clinical decisions.

## Features

- CSV upload pipeline for ICU monitoring, MIMIC-style, and healthcare stroke datasets.
- Automatic schema detection for heart rate, SpO₂, blood pressure, respiration, temperature, and glucose columns.
- Cleaning workflow with missing value handling, duplicate removal, outlier clipping, categorical encoding, vital normalization, and sensor noise smoothing.
- Synthetic ICU data generation with 100,000-row support, drift, noise, missing sensor values, and emergency spikes.
- IoMT sensor stream simulation for edge-style real-time vital updates and alert states.
- Digital twin module for virtual patient health score, stability index, and risk progression.
- AI clinical decision support using RandomForestClassifier and LogisticRegression.
- Streamlit dark-themed dashboard with pages for upload, cleaning, synthetic data, live monitoring, twin simulation, AI prediction, and reports.
- FastAPI backend endpoints for dataset profiling, cleaning, streaming, synthetic data generation, digital twin simulation, and ML training/prediction.
- Optional big-data style large CSV chunk processing with a PySpark-compatible simulation fallback.

## Project Structure

```text
backend/            FastAPI app and API routes
frontend/           Streamlit healthcare dashboard
datasets/           Place Kaggle/MIMIC/stroke CSV files here
synthetic_data/     Synthetic ICU generator and generated CSV output
models/             Trained ML model artifacts
reports/            Cleaning, analytics, and ML reports
outputs/            Heatmaps, alert badges, cleaned datasets, batch summaries
utils/              Dataset profiling and cleaning utilities
digital_twin/       Virtual patient digital twin module
sensor_stream/      IoMT streaming simulator
analytics/          ML models, reports, visualizations, and big-data simulation
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
streamlit run frontend/app.py
```

### 5. Run the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

Open the API docs at <http://127.0.0.1:8000/docs>.

## Example Commands

Generate the default 100,000-row synthetic ICU dataset:

```bash
python synthetic_data/generate_synthetic_icu.py
```

Run a quick module validation:

```bash
python -m compileall backend frontend utils digital_twin sensor_stream analytics synthetic_data
```

Train models from Python:

```python
from synthetic_data.generate_synthetic_icu import generate_synthetic_icu_data
from analytics.ml_models import train_models

df = generate_synthetic_icu_data(rows=5000)
metrics = train_models(df)
print(metrics)
```

## Dataset Suggestions

Download beginner-friendly healthcare CSV datasets and place them in `datasets/`:

- Search Kaggle for **ICU Vital Signs Monitoring Dataset**.
- Search Kaggle for **MIMIC-III Demo Dataset** and use simplified CSV portions.
- Search Kaggle for **Healthcare Stroke Dataset** for tabular risk-prediction practice.

## Optional Big Data Support

Install PySpark if you want Spark flavor:

```bash
pip install pyspark
```

Then call `analytics.big_data.pyspark_preprocess_simulation("datasets/your_large_file.csv")`. If PySpark is unavailable, ICU-TwinAI automatically falls back to chunked Pandas processing.

## Reports and Outputs

The app can create:

- `reports/cleaning_report.txt`
- `reports/icu_analytics_report.txt`
- `outputs/correlation_heatmap.png`
- `outputs/alert_badge.png`
- `outputs/cleaned_dataset.csv`
- `synthetic_data/synthetic_icu_data.csv`
- `models/random_forest.joblib`
- `models/logistic_regression.joblib`

## Viva/Presentation Talking Points

- **Cloud-edge pipeline:** IoMT vitals are simulated at the edge and processed by backend/API and dashboard layers.
- **Digital twin:** The system maintains a virtual patient state and projects risk progression.
- **AIML:** Models classify emergency risk from multivariate ICU sensor features.
- **Big Data simulation:** Large CSV support and optional PySpark-style batch analytics demonstrate scalable healthcare preprocessing.
- **Clinical decision support:** Alerts, risk probabilities, and health scores are combined into an interpretable monitoring dashboard.
