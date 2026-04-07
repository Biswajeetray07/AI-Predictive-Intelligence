# 🧠 AI Predictive Intelligence

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Architecture-Ensemble_ML-brightgreen.svg" alt="Architecture">
  <img src="https://img.shields.io/badge/License-MIT-gray.svg" alt="License">
</div>

<br>

An **End-to-End Multimodal Artificial Intelligence Platform** designed to analyze the entire global economy. It ingests cross-domain data from 15+ sources, parses real-time signals through a robust DeBERTa-v3 NLP pipeline, dynamically identifies market environments using Hidden Markov Models, and runs multi-horizon price inference using an adaptive Deep Learning Ensemble (Transformers, TFT, LSTM, GRU).

It's not just a trading bot — it's an **economic omniscience engine**.

---

## 🚀 Features at a Glance

*   **🌐 15+ Data Domains Ingested**: Stocks (Full OHLCV), Cryptocurrencies, Macro Economics (FRED), Global Trade, Labor Markets, Energy Demand (EIA), Global Weather anomalies, Patents, Academic Research (arXiv), and broad social metrics.
*   **🗣️ Real-Time Social & NLP Engine**: Scrapes developer communities and social platforms (GitHub, HackerNews, YouTube, Mastodon, StackExchange) extracting sentiment (-1 to +1 scale), event categorization (crashes, regulation, launches), and core topics using state-of-the-art `DeBERTa-v3` architecture.
*   **🤖 Adaptive Regime-Weighted Ensemble**: Predicts 1-Day, 5-Day, and 30-Day price targets utilizing 4 parallel neural networks:
    *   **TFT (Temporal Fusion Transformer)**
    *   **Transformer (Self-Attention)**
    *   **LSTM & GRU** (Recurrent memory)
*   **📉 Hidden Markov Model (HMM) Architecture**: Dynamically tracks mathematical market regime states (Bull, Bear, Sideways, High Volatility, Trending). If the VIX spikes or the regime detects high volatility, the system biases its model weights automatically toward conservative estimators.
*   **🖥️ 'Command Center' Streamlit Dashboard**: A hyper-aesthetic, data-rich user interface. Features an integrated explorer for visual heatmaps, alternative data statistics, and raw AI predictions.

---

## 📂 System Architecture Flow

The pipeline executes through robust stages, entirely verifiable from the UI:

1.  **Data Collection Layer (`src/data_collection/`)**: Interacts with disparate APIs (Yahoo Finance, Federal Reserve FRED, EIA, Adzuna, UN Data, NLP sources).
2.  **Processing & Features (`src/data_processing/`)**: Cleans inputs, normalizes time-series schemas, generating up to 17 unique technical indicators (RSI, MACD, Volatility ranges).
3.  **Model Training (`src/models/`)**: Generates optimized dataset permutations and builds separate PyTorch models, retaining testing architectures to validate zero-bias loss.
4.  **Inference Pipeline (`src/pipelines/`)**: Real-time evaluation script loading up the persisted models, aggregating live features, generating statistical target probabilities, and calculating aggregate prediction confidence.
5.  **Dashboards (`dashboard/app.py`)**: Interrogates the underlying `.npy` matrices, logs, and inference outputs to bring the predictions to life.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Biswajeetray07/AI-Predictive-Intelligence.git
   cd AI-Predictive-Intelligence
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file for API keys if utilizing external premium sources (FRED, EIA, etc) depending on customized configurations.

5. **Run the Initialization Scripts**:
   Generate mock or authentic training models based on configuration:
   ```bash
   # Make the runner executable
   chmod +x scripts/run_mock_pipeline.sh
   ./scripts/run_mock_pipeline.sh
   ```

6. **Launch the Intelligence Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   *Navigate to `http://localhost:8502/` to view the Predictive AI Command Center.*

---

## 📊 Dashboard Modules

*   **🏠 Overview**: Real-time KPI summaries of system health, active database volumes, and broad alternative data collection status.
*   **🔮 Predictions**: Select any processed asset (e.g., AAPL) and trigger a forward-pass inference. Displays target prices against immediate macroeconomic context (VIX, Crude, Social Sentiment).
*   **🔎 Anomaly & Regimes**: Investigates statistically extreme moves (Z-Score > 2.5) and exposes the historical identification logic of the Hidden Markov Model.
*   **📈 Visualization**: 7 custom graphing parameters. Stock explorers, Technical Indicator Heatmaps, Weather vs. Market overlays, and Economic indicator comparisons.
*   **🧠 Social & NLP**: Explores aggregated GitHub developer velocity, HN engagement, and resulting overall aggregated Sentiment Scores that drive the prediction confidence.
*   **🌎 Alternative Data Analytics**: Inspect raw, unadulterated global metrics—job vacancies, import/export numbers, energy output ratios.

---

## ⚠️ Disclaimer

This platform provides mathematical and statistical modeling algorithms meant for research and educational demonstrations. Output inferences are **not certified financial advice**. Do not make investment decisions generated exclusively by machine learning prediction models without human oversight and fiduciary consultation.

---

<div align="center">
  <b>Built for modern data science. Operationalized for real-world intelligence.</b>
</div>
