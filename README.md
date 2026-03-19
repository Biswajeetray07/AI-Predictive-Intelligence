<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🧠 AI Predictive Intelligence Platform</h1>

<p align="center">
  <b>An end-to-end multi-modal AI platform that fuses NLP sentiment analysis with time-series forecasting for financial market prediction.</b>
</p>

<p align="center">
  Built with PyTorch · DeBERTa · LSTM/GRU/Transformer/TFT · Optuna HPO · Streamlit Dashboard
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Pipeline Phases](#-pipeline-phases)
- [Dashboard](#-dashboard)
- [Model Details](#-model-details)
- [Configuration](#-configuration)
- [Docker Deployment](#-docker-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧭 Overview

**AI Predictive Intelligence** is a production-grade machine learning platform that combines two powerful AI paradigms:

1. **Natural Language Processing (NLP)** — Extracts sentiment, event signals, topic distributions, and named entities from financial news using a fine-tuned **DeBERTa-v3** multi-task model.
2. **Time-Series Forecasting** — Predicts market movements using an ensemble of **LSTM, GRU, Transformer, and Temporal Fusion Transformer (TFT)** models trained on 100+ engineered technical and macro-economic features.
3. **Fusion Layer** — A neural attention-based fusion model that combines NLP embeddings with time-series predictions to produce a unified, multi-horizon forecast with confidence intervals.

The platform collects data from **20+ real-world APIs** (Yahoo Finance, Alpha Vantage, NewsAPI, FRED, CoinGecko, GDELT, and more), processes it through a sophisticated feature engineering pipeline, and serves predictions through an interactive Streamlit dashboard.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION LAYER                      │
│  Yahoo Finance │ Alpha Vantage │ NewsAPI │ FRED │ CoinGecko    │
│  GDELT │ OpenSky │ Arxiv │ GitHub │ Google Trends │ ...        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   DATA PROCESSING LAYER                        │
│  Financial Processing │ News Processing │ Macro Processing     │
│  Social Processing │ Weather Processing │ External Datasets    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                      │
│  100+ Technical Indicators │ Regime Detection │ Feature Store  │
│  Feature Selection (Mutual Information + Correlation)          │
└────────────┬─────────────────────────────────┬──────────────────┘
             │                                 │
┌────────────▼──────────┐    ┌─────────────────▼──────────────────┐
│    NLP BRANCH         │    │      TIME-SERIES BRANCH            │
│  DeBERTa-v3 Base      │    │  LSTM │ GRU │ Transformer │ TFT   │
│  Multi-Task Head:     │    │  Weighted Ensemble                 │
│  • Sentiment          │    │  Walk-Forward Validation           │
│  • Event Detection    │    │  Optuna Hyperparameter Tuning      │
│  • Topic Modeling     │    │                                    │
│  • Entity Extraction  │    │                                    │
└────────────┬──────────┘    └─────────────────┬──────────────────┘
             │                                 │
┌────────────▼─────────────────────────────────▼──────────────────┐
│                    FUSION MODEL                                │
│  Multi-Head Attention │ Multi-Horizon (1d, 5d, 20d)            │
│  Confidence Calibration │ Direction + Magnitude Prediction     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   EVALUATION & SERVING                         │
│  Backtesting │ Sharpe Ratio │ Directional Accuracy             │
│  Drift Detection │ Streamlit Dashboard │ REST API              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Category | Details |
|---|---|
| **Data Collection** | 20+ API integrations with async fetching, rate limiting, and retry logic |
| **Feature Engineering** | 100+ technical indicators, regime detection, rolling statistics, cross-asset features |
| **NLP Pipeline** | Fine-tuned DeBERTa-v3 with 4-task heads (Sentiment/Events/Topics/Entities) |
| **Time-Series Models** | LSTM, GRU, Transformer, TFT with weighted ensemble averaging |
| **Fusion Model** | Neural attention-based fusion combining NLP + TimeSeries signals |
| **Hyperparameter Optimization** | Optuna-powered HPO with walk-forward cross-validation |
| **Dashboard** | Enterprise-grade Streamlit UI with 12 pages and dark SaaS theme |
| **Monitoring** | Prometheus + Grafana integration for production monitoring |
| **Deployment** | Dockerized with `docker-compose` for one-command deployment |

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+, HuggingFace Transformers |
| **NLP Model** | Microsoft DeBERTa-v3-Base |
| **Time-Series** | Custom LSTM, GRU, Transformer, Temporal Fusion Transformer |
| **Optimization** | Optuna, Walk-Forward Validation |
| **Data** | Pandas, NumPy, scikit-learn |
| **Visualization** | Plotly, Streamlit |
| **Experiment Tracking** | MLflow |
| **Monitoring** | Prometheus, Grafana |
| **Deployment** | Docker, Docker Compose |
| **Hardware Acceleration** | Apple MPS (Metal), CUDA, CPU fallback |

---

## 📁 Project Structure

```
AI-Predictive-Intelligence/
│
├── configs/                          # Model & pipeline configuration
│   ├── training_config.yaml          # Main training hyperparameters
│   ├── best_params.yaml              # Optuna-optimized parameters
│   └── best_training_config.yaml     # Best training configuration
│
├── dashboard/                        # Streamlit Dashboard UI
│   ├── app.py                        # Main 12-page dashboard application
│   ├── styles.py                     # Dark SaaS theme & Plotly templates
│   └── utils.py                      # Data loading & helper utilities
│
├── scripts/                          # Pipeline execution scripts
│   ├── run_full_pipeline.py          # Full 8-phase pipeline orchestrator
│   ├── run_phase_6_only.py           # Train models only
│   ├── run_phase_7_only.py           # Evaluate models only
│   ├── run_phase_8_only.py           # Generate predictions only
│   ├── run_training_only.py          # Quick training shortcut
│   ├── run_no_collection.py          # Skip data collection
│   ├── download_kaggle_datasets.py   # Kaggle dataset downloader
│   ├── generate_mock_nlp.py          # Mock NLP data generator
│   ├── generate_global_regime.py     # Market regime labeler
│   ├── bootstrap_data.py             # Initial data bootstrapper
│   └── advanced_scheduler.py         # Cron-based retraining scheduler
│
├── src/                              # Core source code
│   ├── data_collection/              # 20+ API collectors
│   │   ├── finance/                  # Yahoo Finance, Alpha Vantage, CoinGecko
│   │   ├── news/                     # NewsAPI, GDELT
│   │   ├── economy/                  # FRED economic data
│   │   ├── social_media/             # GitHub, Google Trends, HackerNews
│   │   ├── trade/                    # World Bank, UN Comtrade, OECD
│   │   ├── weather/                  # OpenWeather
│   │   ├── energy/                   # EIA energy data
│   │   ├── aviation/                 # OpenSky aviation data
│   │   ├── crypto/                   # Blockchain data
│   │   ├── research/                 # arXiv papers
│   │   ├── patents/                  # NASA patents
│   │   ├── population/               # UN population data
│   │   ├── jobs/                     # Adzuna, USAJobs
│   │   └── utils/                    # Rate limiter, retry handler, validator
│   │
│   ├── data_processing/              # Raw → processed data transforms
│   │   ├── financial_processing.py   # OHLCV + technical indicators
│   │   ├── news_processing.py        # NLP text preprocessing
│   │   ├── macro_processing.py       # Economic indicator processing
│   │   ├── social_processing.py      # Social media signal processing
│   │   ├── weather_processing.py     # Weather data normalization
│   │   ├── merge_datasets.py         # Multi-source data alignment
│   │   └── build_sequences.py        # Sequence generation for training
│   │
│   ├── feature_engineering/          # Feature creation & selection
│   │   ├── feature_generator.py      # 100+ technical feature generator
│   │   ├── feature_selection.py      # MI-based feature selection
│   │   ├── regime_detection/         # HMM-based market regime detection
│   │   └── feature_store/            # Feature versioning & storage
│   │
│   ├── models/                       # Neural network architectures
│   │   ├── timeseries/               # LSTM, GRU, Transformer, TFT
│   │   ├── nlp/                      # DeBERTa multi-task model
│   │   └── fusion/                   # Attention-based fusion model
│   │
│   ├── training/                     # Training loops & optimization
│   │   ├── timeseries/               # Time-series training pipeline
│   │   ├── nlp/                      # NLP fine-tuning pipeline
│   │   ├── fusion/                   # Fusion model training
│   │   └── optimization/             # Optuna HPO + walk-forward CV
│   │
│   ├── evaluation/                   # Model evaluation & analysis
│   │   ├── metrics.py                # Custom financial metrics
│   │   ├── backtest.py               # Walk-forward backtesting
│   │   ├── explainability.py         # SHAP / feature importance
│   │   └── monitoring/               # Drift detection & retraining triggers
│   │
│   ├── pipelines/                    # End-to-end pipeline orchestrators
│   │   ├── training_pipeline.py      # Full training orchestrator
│   │   └── inference_pipeline.py     # Production inference pipeline
│   │
│   ├── api/                          # REST API (FastAPI)
│   │   ├── app.py                    # API server endpoints
│   │   └── schemas.py                # Request/response Pydantic models
│   │
│   └── utils/                        # Shared utilities
│       ├── seed.py                   # Reproducibility (seed everything)
│       ├── mlflow_utils.py           # MLflow experiment tracking
│       └── pipeline_utils.py         # Common pipeline helpers
│
├── tests/                            # Unit & integration tests
│   ├── test_models.py                # Model architecture tests
│   ├── test_metrics.py               # Financial metrics tests
│   ├── test_data_processing.py       # Data pipeline tests
│   └── test_pipelines_integration.py # End-to-end integration tests
│
├── monitoring/                       # Production monitoring
│   ├── prometheus.yml                # Prometheus scrape configuration
│   └── grafana/                      # Grafana dashboard definitions
│
├── Dockerfile                        # Container image definition
├── docker-compose.yml                # Multi-service orchestration
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project metadata
├── run_phases.sh                     # Shell script pipeline runner
└── LICENSE                           # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip or conda
- (Optional) Apple Silicon Mac for MPS acceleration, or NVIDIA GPU for CUDA

### 1. Clone the Repository

```bash
git clone https://github.com/Biswajeetray07/AI-Predictive-Intelligence.git
cd AI-Predictive-Intelligence
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root with your API keys:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
FRED_API_KEY=your_key_here
COINGECKO_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
# ... add other API keys as needed
```

### 5. Run the Full Pipeline

```bash
python scripts/run_full_pipeline.py
```

Or run individual phases:

```bash
# Train models only (skip data collection)
python scripts/run_phase_6_only.py --hpo

# Evaluate trained models
python scripts/run_phase_7_only.py

# Generate predictions
python scripts/run_phase_8_only.py
```

### 6. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🔄 Pipeline Phases

The platform runs an **8-phase pipeline**, each phase building on the previous:

| Phase | Name | Description |
|:---:|---|---|
| 1 | **Data Collection** | Fetches data from 20+ APIs (finance, news, economic, social) |
| 2 | **Data Processing** | Cleans, normalizes, and aligns multi-source datasets |
| 3 | **Feature Engineering** | Generates 100+ technical indicators and market features |
| 4 | **Feature Selection** | Selects top features via Mutual Information and correlation analysis |
| 5 | **Sequence Building** | Creates windowed sequences for time-series model input |
| 6 | **Model Training** | Trains NLP (DeBERTa), TimeSeries (LSTM/GRU/TFT/Transformer), and Fusion models |
| 7 | **Evaluation** | Runs backtesting, computes Sharpe ratio, directional accuracy, and profit metrics |
| 8 | **Prediction** | Generates multi-horizon (1-day, 5-day, 20-day) forecasts with confidence bands |

---

## 📊 Dashboard

The Streamlit dashboard provides a **12-page enterprise-grade interface**:

| Page | Description |
|---|---|
| **Overview** | KPI cards, mini-charts, resource monitors, and running jobs panel |
| **Data Sources** | Data source health, record counts, and storage usage |
| **Data Pipeline** | Pipeline phase monitoring with status indicators |
| **Datasets** | Available ticker exploration with OHLCV charts |
| **Model Training** | Training progress, loss curves, and model registry |
| **Model Evaluation** | Performance metrics, confusion matrices, and metric comparisons |
| **Predictions** | Live price charts with forecast confidence bands |
| **Anomaly Detection** | Z-score anomaly detection with customizable thresholds |
| **Visualization** | Interactive workspace for custom chart creation |
| **System Monitor** | Real-time CPU, RAM, disk, and GPU utilization |
| **Logs** | Live pipeline log viewer with filtering |
| **Settings** | Theme, API key management, and model configuration |

---

## 🧬 Model Details

### NLP Model — DeBERTa-v3 Multi-Task

- **Base**: `microsoft/deberta-v3-base` (768-dim embeddings)
- **Architecture**: Shared encoder with 4 task-specific heads
- **Tasks**: Sentiment (3-class), Event Detection (binary), Topic Modeling (5-class), Entity Extraction
- **Training**: Fine-tuned with weighted multi-task loss, AdamW optimizer, cosine LR schedule
- **Layer Freezing**: Bottom 6 encoder layers frozen for faster training

### Time-Series Ensemble

| Model | Description |
|---|---|
| **LSTM** | 2-layer, 128-dim hidden, with dropout regularization |
| **GRU** | Lightweight alternative to LSTM for faster convergence |
| **Transformer** | Self-attention based model for capturing long-range dependencies |
| **TFT** | Temporal Fusion Transformer with variable selection and interpretable attention |

Ensemble combines predictions using **weighted averaging** (configurable weights in `configs/training_config.yaml`).

### Fusion Model

- **Input**: NLP embeddings (768-dim) + TimeSeries predictions
- **Architecture**: Multi-head attention (4 heads) → MLP [512 → 256 → 128] → Output
- **Output**: Multi-horizon predictions (1-day, 5-day, 20-day) with direction and magnitude
- **Optimization**: Optuna HPO with walk-forward cross-validation

---

## ⚙️ Configuration

All hyperparameters are centralized in `configs/training_config.yaml`:

```yaml
nlp_model:
  model_name: "microsoft/deberta-v3-base"
  max_length: 256
  batch_size: 16
  epochs: 5
  learning_rate: 5.0e-6

timeseries_model:
  sequence_length: 60
  hidden_dim: 128
  epochs: 80
  models: [lstm, gru, transformer, tft]

fusion_model:
  attention_heads: 4
  mlp_hidden: [512, 256, 128]
  dropout: 0.3
```

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d
```

This starts:
- **AI Platform** (model training + prediction API)
- **Streamlit Dashboard** (web UI on port 8501)
- **Prometheus** (metrics collection on port 9090)
- **Grafana** (monitoring dashboards on port 3000)

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_models.py -v
pytest tests/test_metrics.py -v
```

---

## 📈 Performance Metrics

The platform tracks the following financial metrics:

| Metric | Description |
|---|---|
| **Directional Accuracy** | % of correctly predicted price direction |
| **Sharpe Ratio** | Risk-adjusted return measure |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Profit Factor** | Ratio of gross profits to gross losses |
| **Win Rate** | Percentage of profitable predictions |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built by <a href="https://github.com/Biswajeetray07">Biswajeet Ray</a></b>
</p>
