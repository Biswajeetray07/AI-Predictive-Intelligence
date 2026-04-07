# 🧠 Advanced AI Predictive Intelligence Platform

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Architecture-Ensemble_ML-brightgreen.svg" alt="Architecture">
  <img src="https://img.shields.io/badge/NLP-DeBERTa--v3-blueviolet.svg" alt="NLP Engine">
  <img src="https://img.shields.io/badge/License-MIT-gray.svg" alt="License">
</div>

<br>

An **End-to-End Multimodal Artificial Intelligence Analytics Platform** designed to analyze the entire global economy. It ingests cross-domain data from 15+ external API sources, parses real-time signals through a robust DeBERTa-v3 NLP pipeline, dynamically identifies market environments using Hidden Markov Models, and executes multi-horizon asset price inferences using an adaptive Deep Learning Ensemble.

---

## 🚀 Key Features

*   **🌐 15+ Data Domains Ingested**:
    *   **Financials:** 500+ Stock Tickers with 10-year OHLCV histories.
    *   **Crypto:** Top 100 cryptocurrencies by market cap and volume.
    *   **Macro-Economics (FRED):** GDP, Unemployment Rates, Inflation (CPI), Interest Rates, and VIX Volatility.
    *   **Energy Supply (EIA):** Crude Oil, Natural Gas, Coal Production, and Electrical Demand grids.
    *   **Global Ecosystems:** 101 major supply chain hubs' climate data (Weather anomalies, temperature shifts).
    *   **Alternative Data:** Global patent filings, active job listings across multiple nations, population matrices, and Trade data.

*   **🗣️ Real-Time Social & NLP Engine (`src/pipelines/nlp`)**:
    *   DeBERTa-v3 architecture digests unstructured text from HackerNews, GitHub, StackExchange, and Mastodon.
    *   Determines categorical developer sentiment, quantifies engagement metrics, and builds structured numerical proxies for retail/institutional tech confidence.

*   **🤖 Adaptive Regime-Weighted Ensemble**: Predicts 1-Day, 5-Day, and 30-Day price targets utilizing 4 parallel PyTorch neural networks:
    *   **TFT (Temporal Fusion Transformer)**
    *   **Transformer (Self-Attention Sequence Modeling)**
    *   **LSTM (Long Short-Term Memory)**
    *   **GRU (Gated Recurrent Unit)**

*   **📉 Hidden Markov Model (HMM) Architecture**: Dynamically detects the underlying mathematical state of the market (Bull, Bear, Sideways, High Volatility, Trending). When regimes encounter a Z-Score perturbation > 2.5, predictions automatically re-weight to defensive configurations.

*   **🖥️ 'Command Center' Streamlit Interface**: A hyper-aesthetic, data-rich user interface serving real-time analytics.

---

## 📂 System Architecture Breakdown

| Component Layer | Directory | Description |
| :--- | :--- | :--- |
| **Data Collection** | `src/data_collection/` | Independent fetcher modules managing rate-limits and REST transactions for API providers (FRED, Yahoo, EIA). |
| **Pipelines & ETL** | `src/data_processing/` | Sanitizes NaNs, builds 17+ technical indicators (RSI, Bollinger Bands, MACD), and stitches multivariable dataframes on date intersections |
| **Model Blueprints** | `src/models/` | The isolated PyTorch tensor logic defining neural network depth and hyperparameter structures `(LSTM, GRU, TFT, Transformer)`. |
| **Inference Engine** | `src/pipelines/inference_pipeline.py` | Instantiates models from `.pt` serialized dictionaries, scales real-time data inputs via `MinMaxScaler`, and yields final target tensors. |
| **UI Presentation** | `dashboard/` | Core Streamlit layouts `(app.py, utils.py)` transforming numeric tensors into the Bento-Grid React-based dashboard visualizations. |
| **Automation** | `scripts/` | Shell and Python executables orchestrating the `pipeline -> train -> inference -> dashboard` sequence. |

---

## 🛠️ Installation & Setup

1. **Clone the Project Repository:**
   ```bash
   git clone https://github.com/Biswajeetray07/AI-Predictive-Intelligence.git
   cd AI-Predictive-Intelligence
   ```

2. **Initialize Python Environment:**
   *(Python 3.10+ required)*
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Bootstrapping the Pipeline:**
   Before launching the dashboard, you must generate the initial set of tensors from the underlying data processors.
   ```bash
   # Make automation scripts executable
   chmod +x scripts/*.sh

   # Option 1: Execute full training pipeline (takes time & API calls)
   python scripts/run_full_pipeline.py

   # Option 2: Bootstrap faster with Mock datasets for rapid development testing
   ./scripts/run_mock_pipeline.sh
   ```

5. **Start the Intelligence Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```
   *Dashboard handles its own hot-reloading logic. Visit `http://localhost:8502`.*

---

## 💻 Dashboard Modules Tour

*   **🏠 Overview**: Real-time KPI summaries of system health, active database volumes (CPU/RAM metrics), and comprehensive multi-domain data collection status.
*   **🔮 Predictions**: Pick any stock. The system forces a forward pass on the ensemble and returns explicit price targets for multi-horizon periods, weighted contextually against the Live VIX and Social Sentiment.
*   **🔎 Anomaly & Regimes**: Investigates historical price actions exhibiting statistically extreme bounds (Z-Score > 2.5), revealing the discrete math driving the Hidden Markov State tracker.
*   **📈 Visualization Hub**: 7 custom graphical interfaces spanning from raw Stock candlestick explorers, correlation heatmaps, Crypto overviews, to unique aggregations mapping global weather data against index movements.
*   **🌎 Alternative Data**: Investigate the raw numerical signals dictating predictions, split logically across Federal Reserve indices, Government Energy data, GitHub developer volume, and Job vacancy postings.

---

## 📄 License & Usage Warnings

This platform offers complex mathematical and statistical modeling software designed **solely for educational and research simulations**.

> [!WARNING]
> Output inferences **DO NOT constitute certified financial advice**. Do not act on investment decisions generated by these machine learning predictions without thorough human diligence and adherence to local financial regulatory standards.

Distributed under the MIT License.
