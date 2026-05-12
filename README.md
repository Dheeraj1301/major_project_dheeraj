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


## Dashboard Pages: Complete Information, Code Flow, Algorithms, and Models

This section explains the full Streamlit dashboard page-by-page for report writing and viva preparation. The dashboard is implemented in `frontend/app.py`, while reusable logic is separated into modules under `utils/`, `synthetic_data/`, `sensor_stream/`, `digital_twin/`, and `analytics/`.

### Quick 8-Page Page-to-Module Map

| Page | What the user does | Main code modules that run | Algorithm/model used | Main output |
| --- | --- | --- | --- | --- |
| 1. Home dashboard | Opens the landing dashboard and views the latest ICU state. | `frontend/app.py`, `synthetic_data/generate_synthetic_icu.py`, `analytics/visualizations.py` | Synthetic time-series generation, rule-based ICU risk score, Plotly trend/gauge visualization | Latest vitals, alert count, trend chart, health gauge |
| 2. Dataset upload | Uploads a CSV or uses demo ICU data. | `frontend/app.py`, `utils/data_pipeline.py` | CSV ingestion, schema profiling, vital-column alias detection, missing/duplicate/outlier counting | Data preview and JSON dataset profile |
| 3. Data cleaning simulation | Uploads/loads data and runs preprocessing. | `frontend/app.py`, `utils/data_pipeline.py` | Duplicate removal, median/mode imputation, range clipping, rolling-median smoothing, `LabelEncoder`, `StandardScaler` | Cleaned dataframe and `reports/cleaning_report.txt` |
| 4. Synthetic data generator | Selects row count and generates a CSV. | `frontend/app.py`, `synthetic_data/generate_synthetic_icu.py` | NumPy random distributions, time-series drift, missingness injection, emergency spike injection, rule-based risk labeling | Preview data and `synthetic_data/synthetic_icu_data.csv` |
| 5. Live ICU monitoring | Starts or previews an IoMT sensor stream. | `frontend/app.py`, `sensor_stream/simulator.py`, `analytics/visualizations.py` | Random-walk streaming, probabilistic adverse-event injection, threshold alert classification | Live alert metric and vital trend graph |
| 6. Digital twin simulation | Sets patient age, comorbidity, and vitals. | `frontend/app.py`, `digital_twin/patient_twin.py`, `analytics/visualizations.py` | Digital-twin penalty scoring, clipped health/risk score, stability index, 24-step progression simulation | Health gauge, patient state JSON, progression table |
| 7. AI prediction dashboard | Trains models and enters vitals for prediction. | `frontend/app.py`, `analytics/ml_models.py`, `analytics/visualizations.py` | `RandomForestClassifier`, `LogisticRegression` pipeline with `StandardScaler`, train/test split, accuracy/report/confusion matrix | Saved `.joblib` models, ML report, live risk label/probability |
| 8. Reports page | Generates analytics artifacts. | `frontend/app.py`, `analytics/visualizations.py` | Seaborn correlation heatmap, Matplotlib export, OpenCV badge drawing, Pandas descriptive statistics | `reports/icu_analytics_report.txt`, heatmap PNG, alert badge PNG |

### Page 1: Home Dashboard

**Purpose:** The Home dashboard is the first overview screen for ICU-TwinAI. It demonstrates how a clinical monitoring platform can show the latest condition of a virtual ICU patient using generated ICU vitals.

**What happens on the page:**

1. Streamlit renders the hero title and project subtitle.
2. `demo_data()` creates a cached synthetic ICU dataset using `generate_synthetic_icu_data()`.
3. The latest row is selected as the current ICU state.
4. Metrics show heart rate, SpO₂, blood pressure, risk percentage, and total critical alerts.
5. `trend_graph()` plots recent vital trends.
6. `health_gauge()` shows a health score calculated as `100 - latest.icu_risk_score`.

**Code flow:** `frontend/app.py` → cached `demo_data()` → `synthetic_data.generate_synthetic_icu_data()` → latest-row metrics → `analytics.visualizations.trend_graph()` and `health_gauge()`.

**Algorithms used:** synthetic time-series generation, rule-based risk scoring, threshold-derived alert counting, and Plotly visualization.

**ML model usage:** No trained ML model runs on this page. The page uses generated risk labels and visual analytics only.

**Outputs:** dashboard metric cards, ICU vital trend chart, and digital-twin-style health gauge.

### Page 2: Dataset Upload

**Purpose:** This page lets a user upload a CSV dataset from ICU, MIMIC-style, or stroke/healthcare sources and immediately inspect whether the dataset is suitable for analysis.

**What happens on the page:**

1. The user uploads a CSV using Streamlit's file uploader; if no file is uploaded, demo synthetic data is used.
2. Pandas reads the CSV and displays the first 25 rows.
3. `profile_dataset()` calculates rows, columns, missing values, duplicate rows, numeric columns, categorical columns, detected vital signs, and outlier counts.
4. `render_profile()` displays summary metrics and the full profile as JSON.

**Code flow:** `frontend/app.py` → `pd.read_csv()` or `demo_data()` → `utils.data_pipeline.profile_dataset()` → `detect_vital_columns()` → `outlier_counts()` → `render_profile()`.

**Algorithms used:** schema profiling, flexible vital-name alias matching, duplicate counting, missing-value counting, numeric/categorical separation, and normal-range outlier counting.

**ML model usage:** No prediction model is trained here. This page prepares the user to understand input quality before cleaning or ML training.

**Outputs:** uploaded/demo dataframe preview and dataset profile JSON.

### Page 3: Data Cleaning Simulation

**Purpose:** This page demonstrates a full preprocessing pipeline that converts noisy raw healthcare data into a cleaner ML-ready table.

**What happens on the page:**

1. The user uploads a CSV or uses demo data.
2. `clean_dataset()` profiles the data before cleaning.
3. Duplicate rows are dropped.
4. Numeric missing values are filled with each column median.
5. Categorical missing values are filled with each column mode, or `Unknown` if no mode exists.
6. ICU vital columns are detected and clipped to medically reasonable normal ranges.
7. Vital signs are smoothed with a centered rolling median window to reduce sensor noise.
8. Categorical columns are encoded using `LabelEncoder`.
9. Numeric columns are standardized with `StandardScaler`.
10. The cleaned dataset is profiled again and a before/after report is saved.

**Code flow:** `frontend/app.py` → `clean_dataset()` → `profile_dataset()` before → duplicate removal → imputation → `detect_vital_columns()` → range clipping → `remove_sensor_noise()` → `LabelEncoder` → `StandardScaler` → `profile_dataset()` after → `save_cleaning_report()`.

**Algorithms used:** duplicate removal, median imputation, mode imputation, normal-range clipping, rolling median smoothing, label encoding, z-score standardization, and profiling comparison.

**ML model usage:** No classifier is trained on this page, but the preprocessing output is the type of data transformation normally required before ML training.

**Outputs:** before/after cleaning summaries, cleaned dataframe preview, and `reports/cleaning_report.txt`.

### Page 4: Synthetic Data Generator

**Purpose:** This page creates a large artificial ICU dataset so the system can be demonstrated even when real hospital data is unavailable.

**What happens on the page:**

1. The user chooses a row count from 1,000 to 100,000.
2. When the button is clicked, `save_synthetic_data()` generates and writes a CSV.
3. The generator creates timestamps, patient IDs, names, age, heart rate, SpO₂, blood pressure, respiratory rate, temperature, ECG variability, glucose, stress index, ICU risk score, and critical-alert labels.
4. Drift is added over time to imitate slow patient deterioration or recovery.
5. Emergency spikes are injected into a subset of rows.
6. A small amount of missingness is injected into vital columns to make cleaning realistic.

**Code flow:** `frontend/app.py` → row-count slider → `save_synthetic_data(rows)` → `generate_synthetic_icu_data()` → `_risk_score()` → CSV export.

**Algorithms used:** seeded NumPy random generation, Gaussian distributions for vitals, beta distribution for stress index, linear drift, probabilistic spike injection, random missing-value injection, rule-based risk scoring, and binary target generation.

**How synthetic data supports ML training:** The synthetic dataset includes both feature columns and the `critical_alert` target column. This allows Page 7 to train supervised classification models without needing protected real patient records.

**Outputs:** synthetic data preview and `synthetic_data/synthetic_icu_data.csv`.

### Page 5: Live ICU Monitoring

**Purpose:** This page simulates IoMT/edge-device streaming, where patient vitals arrive continuously and the dashboard updates alert status.

**What happens on the page:**

1. `IOMTSensorSimulator()` starts with baseline vitals.
2. The user selects how many stream samples to display.
3. If the stream button is clicked, the app repeatedly calls `next_reading()` and updates the placeholder container.
4. Each sensor value changes by a small random amount, creating a random-walk stream.
5. `_inject_event()` occasionally creates acute instability by increasing heart rate and respiratory rate while decreasing SpO₂.
6. `classify_alert()` assigns `STABLE`, `WARNING`, or `CRITICAL` using threshold rules.
7. `trend_graph()` visualizes the incoming readings.

**Code flow:** `frontend/app.py` → `IOMTSensorSimulator()` → `next_reading()` → `_next_value()` → `_inject_event()` → `classify_alert()` → Streamlit metric and Plotly trend graph.

**Algorithms used:** random-walk time-series simulation, probabilistic event injection, threshold-based triage classification, and live chart refresh.

**ML model usage:** No saved classifier is used here. The stream is rule-based to make edge alerting fast and interpretable.

**Outputs:** live alert level/message and streaming vital trend chart.

### Page 6: Digital Twin Simulation

**Purpose:** This page creates a virtual patient twin that updates its health state from patient profile values and current vitals.

**What happens on the page:**

1. The user selects age, comorbidity index, and current vital values.
2. `PatientProfile` stores patient-level context.
3. `ICUPatientTwin.update()` calls `compute_health_score()`.
4. The health score is calculated by subtracting weighted risk penalties from 100.
5. Risk penalties increase when heart rate, SpO₂, systolic BP, respiratory rate, temperature, age, or comorbidity become abnormal.
6. The stability index combines health score and SpO₂ into a 0-to-1 value.
7. Status is classified as `STABLE`, `WATCH`, or `CRITICAL` using risk thresholds.
8. `simulate_progression()` generates 24 future steps with small random changes to vitals.

**Code flow:** `frontend/app.py` → Streamlit sliders → `PatientProfile` → `ICUPatientTwin.update()` → `compute_health_score()` → `health_gauge()` → `simulate_progression()`.

**Algorithms used:** weighted penalty scoring, value clipping, threshold status classification, stochastic progression simulation, and gauge visualization.

**ML model usage:** This page uses a deterministic digital-twin model rather than a trained classifier. It is interpretable because each vital directly contributes to the risk penalty.

**Outputs:** health gauge, complete state JSON, and 24-step progression table.

### Page 7: AI Prediction Dashboard

**Purpose:** This page is the main supervised ML page. It trains clinical decision-support classifiers and uses saved models to estimate critical risk for manually entered vitals.

**ML input features:** `heart_rate`, `spo2`, `systolic_bp`, `diastolic_bp`, `respiratory_rate`, `temperature`, `glucose`, and `stress_index`.

**Target variable:** `critical_alert`, a binary label where `1` means critical-alert risk and `0` means non-critical/stable risk.

**What happens on the page:**

1. The page loads 4,000 rows of cached synthetic ICU demo data.
2. When the training button is clicked, `train_models(df)` prepares the training frame.
3. Missing feature columns are created if needed, and feature values are converted to numeric.
4. If `critical_alert` does not exist, it is generated from rule-based vital abnormalities.
5. `train_test_split()` separates data into 75% training and 25% testing using `random_state=42`; stratification is used when the target has enough examples in both classes.
6. Two models are trained: `RandomForestClassifier` and a `LogisticRegression` pipeline with `StandardScaler`.
7. Predictions on the test set are evaluated with accuracy, classification report, and confusion matrix.
8. The trained models are saved as `.joblib` files.
9. `write_analytics_report()` saves a text analytics report.
10. If `models/random_forest.joblib` exists, manually entered vitals are sent to `predict_patient_risk()`.
11. Prediction probability is converted to `STABLE`, `WATCH`, or `CRITICAL`.

**Code flow:** `frontend/app.py` → `demo_data(4000)` → training button → `analytics.ml_models.train_models()` → `prepare_training_frame()` → `train_test_split()` → model fitting/evaluation → `joblib.dump()` → `write_analytics_report()` → numeric vital inputs → `predict_patient_risk()`.

**Algorithms and models used:**

- `RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")` for robust non-linear classification.
- `LogisticRegression(max_iter=1000)` for interpretable linear classification.
- `StandardScaler` inside the logistic-regression pipeline to normalize feature scale.
- Accuracy score, classification report, and confusion matrix for evaluation.

**Saved model files:**

- `models/random_forest.joblib`
- `models/logistic_regression.joblib`

**Prediction outputs:** `critical_probability` and `decision_state`, where probability >= 0.65 is `CRITICAL`, probability >= 0.35 is `WATCH`, and lower probability is `STABLE`.

### Page 8: Reports Page

**Purpose:** This page generates report-ready visual and text artifacts for documentation, presentation, and viva evidence.

**What happens on the page:**

1. The page creates 1,200 rows of demo ICU data.
2. `save_correlation_heatmap(df)` selects numeric columns and computes a correlation matrix.
3. Seaborn renders the correlation heatmap.
4. Matplotlib saves the heatmap image to `outputs/correlation_heatmap.png`.
5. `save_alert_badge("WARNING")` uses OpenCV to create an alert-badge image.
6. `write_analytics_report(df)` writes rows, columns, and vital descriptive statistics.
7. Streamlit displays the generated artifact paths and images.

**Code flow:** `frontend/app.py` → `demo_data(1200)` → `save_correlation_heatmap()` → `save_alert_badge()` → `write_analytics_report()` → Streamlit image display.

**Algorithms used:** Pandas correlation calculation, Seaborn heatmap visualization, Matplotlib image export, OpenCV rectangle/text drawing, and Pandas descriptive statistics.

**Outputs:**

- `reports/icu_analytics_report.txt`
- `outputs/correlation_heatmap.png`
- `outputs/alert_badge.png`

### End-to-End Dashboard Flow

1. **Data source:** The app starts with uploaded CSV data or synthetic ICU data.
2. **Profiling:** The dataset is inspected for schema, missing values, duplicates, detected vitals, and outliers.
3. **Cleaning:** Missing values, duplicates, outliers, categorical columns, noise, and feature scale are handled.
4. **Simulation:** Synthetic ICU vitals and IoMT streams provide realistic demo data.
5. **Digital twin:** Patient vitals and profile data update a virtual patient state.
6. **ML training:** Clean/synthetic data trains RandomForest and LogisticRegression classifiers.
7. **Prediction:** Saved models classify live/manual vital inputs as stable, watch, or critical.
8. **Reporting:** Heatmaps, badges, text reports, cleaned datasets, synthetic datasets, and model artifacts are exported.

### Summary of Algorithms Used

| Area | Techniques used | Why it is used |
| --- | --- | --- |
| Dataset profiling | Missing count, duplicate count, numeric/categorical detection, alias-based vital detection, normal-range outlier count | Understand dataset quality and schema before processing |
| Cleaning/preprocessing | Median imputation, mode imputation, duplicate removal, range clipping, rolling median smoothing, label encoding, standard scaling | Prepare noisy healthcare data for analysis and ML |
| Synthetic data | Gaussian vital distributions, beta stress distribution, linear drift, emergency spikes, missingness injection, rule-based risk score | Provide realistic large-scale ICU data without exposing real patients |
| IoMT streaming | Random walk, event injection, threshold alerting | Simulate edge-device vitals and fast clinical alerts |
| Digital twin | Weighted penalty score, clipping, stability index, stochastic progression | Maintain an interpretable virtual patient state |
| Machine learning | `RandomForestClassifier`, `LogisticRegression`, `StandardScaler`, train/test split | Train and evaluate supervised critical-alert predictors |
| Evaluation | Accuracy, classification report, confusion matrix | Measure classification performance |
| Reporting | Correlation matrix, Seaborn heatmap, Matplotlib export, OpenCV image generation, descriptive statistics | Create visual and text artifacts for reports and viva |

## Viva/Presentation Talking Points

- **Cloud-edge pipeline:** IoMT vitals are simulated at the edge and processed by backend/API and dashboard layers.
- **Digital twin:** The system maintains a virtual patient state and projects risk progression.
- **AIML:** Models classify emergency risk from multivariate ICU sensor features.
- **Big Data simulation:** Large CSV support and optional PySpark-style batch analytics demonstrate scalable healthcare preprocessing.
- **Clinical decision support:** Alerts, risk probabilities, and health scores are combined into an interpretable monitoring dashboard.
