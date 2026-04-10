# 🧠 Advanced AI Predictive Intelligence Platform

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Docker-Enabled-2CA5E0.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Cloud-AWS_S3-FF9900.svg" alt="AWS S3">
  <img src="https://img.shields.io/badge/Monitoring-Prometheus_%7C_Grafana-orange.svg" alt="Monitoring">
  <img src="https://img.shields.io/badge/License-MIT-gray.svg" alt="License">
</div>

<br>

An **End-to-End Multimodal Artificial Intelligence Analytics Platform** designed to analyze the entire global economy. It ingests cross-domain data from 15+ external API sources, parses real-time signals, dynamically identifies market environments using Hidden Markov Models, executes multi-horizon asset price inferences using an adaptive Deep Learning Ensemble, and streams data scalelessly using AWS S3.

The platform is operationalized using a modern MLOps stack featuring Docker Compose, FastAPI, MLflow experiment tracking, and Prometheus/Grafana system monitoring.

---

## 🚀 Key Features

*   **🌐 15+ Data Domains Ingested**:
    *   **Financials:** Stocks with extensive OHLCV histories.
    *   **Crypto:** Top cryptocurrencies by market cap.
    *   **Macro-Economics (FRED):** GDP, Unemployment Rates, Inflation (CPI), Interest Rates.
    *   **Energy Supply (EIA):** Crude Oil, Natural Gas trends.

*   **☁️ Cloud-Native Inference Architecture**:
    *   Fully decoupled **AWS S3** streaming data backend loading tensors and model artifacts into memory dynamically, bypassing local disk bottlenecks.

*   **🤖 Adaptive Regime-Weighted Ensemble**: Predicts 1-Day, 5-Day, and 30-Day price targets utilizing 4 parallel PyTorch neural networks:
    *   **TFT (Temporal Fusion Transformer)**
    *   **Transformer (Self-Attention Sequence Modeling)**
    *   **LSTM (Long Short-Term Memory)**
    *   **GRU (Gated Recurrent Unit)**

*   **📉 Hidden Markov Model (HMM) Architecture**: Dynamically detects the underlying mathematical state of the market (Bull, Bear, Sideways, High Volatility, Trending).

*   **⚡ Modern MLOps Stack**:
    *   **FastAPI Backend**: Serving predictions and health checks.
    *   **MLflow**: Tracking model experimentation parameters and metrics.
    *   **Prometheus & Grafana**: Providing container-level and data-pipeline time-series observability.

*   **🖥️ 'Command Center' Streamlit Interface**: A hyper-aesthetic, data-rich user interface serving real-time analytics directly linked to cloud tensors.

---

## 📂 System Architecture Breakdown

| Component Layer | Directory | Description |
| :--- | :--- | :--- |
| **Data Collection & Processing** | `src/data_collection/`, `src/data_processing/`, `src/feature_engineering/` | Sanitizes NaNs, builds technical indicators, and stitches multivariable dataframes on date intersections |
| **Model Blueprints & Training** | `src/models/`, `src/training/` | The core PyTorch defining neural network depth and hyperparameter structures. |
| **Cloud Storage integration** | `src/cloud_storage/` | AWS S3 connectivity layer providing chunked, memory-streamed tensor fetching via `boto3`. |
| **API & Inference Pipeline** | `src/api/`, `src/pipelines/` | FastAPI interface routing requests to instantiated model checkpoints dynamically. |
| **UI Presentation** | `dashboard/` | Core Streamlit layouts transforming numeric tensors into visual dashboard interactions. |
| **Monitoring & Experimentation**| `monitoring/`, `docker-compose.yml` | Prometheus scraped metrics, Grafana dashboards, and MLflow setups. |

---

## 🛠️ Installation & Setup

### Environment Variables

Before starting the cluster, ensure you populate your `.env` file in the root directory:

```env
# Cloud Integration
USE_S3=True
MODEL_BUCKET_NAME=my-model-mlopsproj012
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# API Setup
API_KEY=predictive_intel_dev_key_2026
```

### 1. Bootstrapping Cloud Infrastructure

If this is a fresh setup and you need to upload local datasets and serialized models to S3:

```bash
python scripts/sync_to_s3.py
```

### 2. Docker Compose Deployment (Recommended)

Run the entire MLOps suite (Dashboard, API, Prometheus, Grafana, MLflow) in separated, fully-networked Docker containers.

```bash
docker-compose up --build -d
```

**Services Deployed on `localhost`:**
*   Dashboard: `http://localhost:8501`
*   FastAPI Docs: `http://localhost:8000/docs`
*   Grafana: `http://localhost:3000`
*   Prometheus: `http://localhost:9090`
*   MLflow: `http://localhost:5000`

### 3. Local Python Execution (Alternative)

1. **Python Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

---

## 💻 Dashboard Modules Tour

*   **🏠 Overview**: Real-time KPI summaries of system health, cloud latency metrics, and API statuses.
*   **🔮 Predictions**: Deep Learning Ensemble yields explicit price targets for multi-horizon periods, weighted contextually against Live VIX and Social Sentiment.
*   **🔎 Anomaly & Regimes**: Investigates historical price actions exhibiting statistically extreme bounds, revealing the discrete math driving the Hidden Markov State tracker.
*   **📈 Visualization Hub**: Custom graphical interfaces spanning from raw Stock candlestick explorers, correlation heatmaps, Crypto overviews, to weather data aggregations.
*   **🌎 Alternative Data**: Investigate the raw numerical signals dictating predictions across Federal Reserve indices, Government Energy data, GitHub developers, and Job vacancies.

---

## 📄 License & Usage Warnings

This platform offers complex mathematical and statistical modeling software designed **solely for educational and research simulations**.

> [!WARNING]
> Output inferences **DO NOT constitute certified financial advice**. Do not act on investment decisions generated by these machine learning predictions without thorough human diligence and adherence to local financial regulatory standards.

Distributed under the MIT License.
