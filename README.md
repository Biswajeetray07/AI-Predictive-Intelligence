# 🧠 AI Predictive Intelligence Platform

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/DeBERTa--v3-NLP_Engine-6366F1.svg?logo=huggingface&logoColor=white" alt="DeBERTa">
  <img src="https://img.shields.io/badge/AWS_S3-Cloud_Inference-FF9900.svg?logo=amazonaws&logoColor=white" alt="AWS S3">
  <img src="https://img.shields.io/badge/FastAPI-REST_API-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Optuna-Hyperparameter_Optimization-2980B9.svg" alt="Optuna">
  <img src="https://img.shields.io/badge/License-MIT-gray.svg" alt="License">
</div>

<br>

An **end-to-end multimodal AI analytics platform** designed to model the global economy. It ingests cross-domain data from **15+ external API sources** spanning finance, energy, trade, social media, research, patents, weather, and macroeconomics — then fuses these signals through a **regime-adaptive deep learning ensemble** to generate multi-horizon asset price forecasts.

The system dynamically detects market states using **Gaussian Hidden Markov Models**, adjusts ensemble weights in real-time, and enriches numerical predictions with **DeBERTa-v3 NLP sentiment and event analysis**. All model artifacts and data can be **streamed from AWS S3** for zero-disk cloud inference.

---

## Table of Contents

- [Core Architecture](#-core-architecture)
- [Multi-Domain Data Ingestion](#-multi-domain-data-ingestion)
- [Feature Engineering Pipeline](#-feature-engineering-pipeline)
- [Model Zoo — Deep Learning Ensemble](#-model-zoo--deep-learning-ensemble)
- [HMM Regime Detection](#-hmm-regime-detection--dynamic-ensemble-weighting)
- [Multi-Task NLP Engine](#-multi-task-nlp-engine-deberta-v3)
- [Deep Fusion Architecture](#-deep-fusion-architecture)
- [Cloud-Native S3 Inference](#%EF%B8%8F-cloud-native-s3-inference-architecture)
- [Hyperparameter Optimization](#-hyperparameter-optimization-optuna)
- [Evaluation & Backtesting](#-evaluation--backtesting-suite)
- [Model Explainability](#-model-explainability-shap--attention)
- [Drift Detection & Automated Retraining](#-drift-detection--automated-retraining)
- [Feature Store](#-feature-store-parquet--duckdb)
- [REST API](#-rest-api-fastapi)
- [Real-Time Dashboard](#-real-time-command-center-dashboard)
- [9-Phase Pipeline Orchestrator](#-9-phase-master-pipeline-orchestrator)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)
- [License & Disclaimer](#-license--disclaimer)

---

## 🏗 Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                                │
│  15+ APIs: Yahoo Finance · CoinGecko · FRED · EIA · NewsAPI · Reddit ·     │
│  GitHub · HackerNews · Mastodon · YouTube · arXiv · USPTO · OpenSky ·      │
│  World Bank · Adzuna · USAJobs · NASA · Alpha Vantage                      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                      FEATURE ENGINEERING LAYER                              │
│  FeatureGenerator → 11 derived indices (Trade, Energy, Innovation, ...)    │
│  RegimeDetector → 5-state HMM probability features                         │
│  FeatureSelection → MI + Variance + Correlation filtering                  │
│  FeatureStore → Versioned Parquet with DuckDB query engine                 │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                         MODEL TRAINING LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  LSTM         │  │  GRU         │  │ Transformer  │  │    TFT       │   │
│  │  Attention    │  │  Stacked     │  │ Causal Mask  │  │ Variable Sel │   │
│  │  Pooling      │  │  Deep GRU    │  │ Positional   │  │ Multi-Head   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                  │                  │           │
│         └────────┬────────┘──────────┬───────┘──────────┬──────┘           │
│                  ▼                   ▼                   ▼                  │
│         ┌────────────────────────────────────────────────────────┐          │
│         │  HMM REGIME-WEIGHTED ENSEMBLE (Dynamic α weights)     │          │
│         └──────────────────────┬─────────────────────────────────┘          │
│                                │                                            │
│  ┌─────────────────────────────▼──────┐  ┌──────────────────────────────┐  │
│  │  DeBERTa-v3 Multi-Task NLP Engine  │  │  RealTimeFeatureBuilder      │  │
│  │  • Sentiment (3-class)             │  │  • On-the-fly tensor         │  │
│  │  • Events (8-class)                │  │    construction from         │  │
│  │  • Topics (10-class)               │  │    raw data                  │  │
│  │  • Entities (6-class)              │  │  • Decoupled from training   │  │
│  └─────────────────────────────┬──────┘  └──────────────────────────────┘  │
│                                │                                            │
│           ┌────────────────────▼────────────────────┐                       │
│           │  DEEP FUSION MODEL (GMU + MLP Head)     │                       │
│           │  Multi-Horizon: 1d, 5d, 30d targets     │                       │
│           └────────────────────┬────────────────────┘                       │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                        SERVING LAYER                                        │
│  FastAPI REST API · Streamlit Dashboard · S3-Streamed Cloud Inference      │
│  Docker Compose Deployment · MLflow Experiment Tracking                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Multi-Domain Data Ingestion

The platform collects and fuses data from **15 distinct real-world domains** via automated API collectors, producing a unified dataset of **~1.6 million records**:

| Domain | Source | Records | Description |
|:---|:---|---:|:---|
| 📈 **Stocks** | Yahoo Finance | 1,369,796 | 502 tickers, 6-year OHLCV + 17 technical indicators |
| 🪙 **Crypto** | CoinGecko | 34,099 | Top 100 cryptocurrencies — price, volume, market cap |
| 🏛 **Macro-Economics** | FRED API | 134,724 | GDP, CPI, unemployment, interest rates, VIX, yield curves |
| ⚡ **Energy** | EIA (US DOE) | 6,364 | Crude oil, natural gas, coal, electricity supply series |
| 🌍 **Trade** | World Bank (WITS) | 2,548 | International trade flows and balances |
| 💬 **Social Media** | GitHub, HN, Reddit, YouTube, Mastodon | 12,445 | Developer activity, community sentiment, discussion volume |
| 📰 **News** | NewsAPI, GDELT | 217 | Headline sentiment, geopolitical events |
| 📄 **Research** | arXiv API | 8,003 | AI/ML/Finance research papers — title, abstract, categories |
| 🔬 **Patents** | USPTO (PatentsView) | 335 | Technology patent filings — innovation signals |
| ✈️ **Aviation** | OpenSky Network | 11,914 | Global flight traffic as economic mobility proxy |
| 💼 **Jobs** | Adzuna + USAJobs | 5,049 | Job market listings — demand signals |
| 🌤 **Weather** | WeatherAPI | 101 | 101 cities — temperature, humidity, economic weather impact |
| 👥 **Population** | World Bank | 6,480 | Demographic trends across 120 countries |
| 🔗 **Blockchain** | Blockchain.com | 5,806 | Bitcoin hash rate, transaction volume, network difficulty |
| 📊 **Alpha Vantage** | Alpha Vantage API | 200 | Supplementary financial indicators |

**Async data collection** is powered by a custom `AsyncFetcher` with configurable concurrency (20 simultaneous connections), exponential-backoff rate limiting, and automatic retries.

---

## ⚙️ Feature Engineering Pipeline

### Multi-Domain Feature Generator

The `FeatureGenerator` (`src/feature_engineering/feature_generator.py`) transforms raw data into **11 derived composite indices**:

| Index | Composition |
|:---|:---|
| **Trade Activity Index** | Import/export volumes, trade balance ratios |
| **Energy Market Index** | Oil/gas/coal/electricity price-weighted composite |
| **Innovation Index** | Patent filing rates, research publication velocity |
| **Labor Market Index** | Job listings, employment trends |
| **Social Sentiment Index** | GitHub stars, HN engagement, Reddit sentiment |
| **Aviation Activity Index** | Flight volumes as economic mobility proxy |
| **Blockchain Activity Index** | Hash rate, transaction volume, network activity |
| **Weather Impact Index** | Temperature anomalies, regional climate stress |
| **Population Dynamics Index** | Demographic growth, urbanization trends |
| **Macro Health Index** | Composite of GDP, CPI, unemployment, rates |
| **News Intensity Index** | Publication frequency, geopolitical signal strength |

### Automated Feature Selection

A 4-stage selection pipeline (`src/feature_engineering/feature_selection.py`) reduces dimensionality:

1. **Variance Filtering** — Remove near-zero variance features
2. **Correlation Pruning** — Eliminate redundant features (ρ > 0.85 threshold)
3. **Mutual Information Ranking** — Score features by MI with target variable
4. **Threshold Enforcement** — Keep minimum 20 features regardless of strict cuts

Output: a `selected_features.yaml` configuration consumed by the sequence builder.

### Regime Feature Augmentation

After base feature generation, Gaussian HMM-derived **regime probability columns** (`regime_prob_0..4`) are appended to the merged dataset, providing the model with explicit market-state awareness as input features.

---

## 🤖 Model Zoo — Deep Learning Ensemble

Four parallel PyTorch time-series models operate on 60-step sequences with 49 engineered features:

### 1. Temporal Fusion Transformer (TFT)
`src/models/timeseries/tft.py`

- **Variable Selection Network** — Learns per-timestep feature importance using GRN (Gated Residual Networks)
- **Multi-Head Interpretable Attention** — 4-head self-attention with interpretable weight extraction
- **Skip connections** and layer normalization for deep stability

### 2. Transformer Encoder
`src/models/timeseries/transformer.py`

- **Causal attention masking** — Prevents future data leakage during sequence modeling
- **Sinusoidal positional encoding** — Injects temporal ordering information
- **Configurable depth** — `d_model=128`, `nhead=4`, `num_layers=2`

### 3. LSTM with Attention Pooling
`src/models/timeseries/lstm.py`

- **Multi-layer unidirectional LSTM** — Captures long-range temporal dependencies
- **Learned attention pooling** — Weights hidden states by relevance instead of using only the final state
- Dropout regularization between layers

### 4. Stacked GRU
`src/models/timeseries/gru.py`

- **Efficient gated architecture** — Fewer parameters than LSTM with comparable performance
- **Deep stacking** — 2-layer GRU with inter-layer dropout
- Final hidden state projection to prediction head

### Ensemble Strategy

All four models are combined via a **regime-weighted ensemble** (see HMM section below). Default static weights from hyperparameter optimization:

```
LSTM: 0.25 | GRU: 0.20 | Transformer: 0.30 | TFT: 0.25
```

These weights are dynamically overridden at inference time based on the detected market regime.

---

## 📉 HMM Regime Detection & Dynamic Ensemble Weighting

`src/feature_engineering/regime_detection/regime_detector.py`

A **5-state Gaussian Hidden Markov Model** classifies the market into discrete regimes using return statistics and volatility:

| State | Label | Ensemble Bias |
|:---|:---|:---|
| 0 | **Bull Market** | Transformer & TFT weighted higher (momentum-capturing) |
| 1 | **Bear Market** | LSTM weighted higher (mean-reversion bias) |
| 2 | **Sideways/Range** | GRU weighted higher (noise-filtering) |
| 3 | **High Volatility** | TFT weighted highest (variable selection handles noise) |
| 4 | **Trending** | Equal weights (balanced ensemble) |

**Dynamic Weight Computation** (`Predictor._get_dynamic_regime_weights`):

At inference time, the regime detector processes recent price data to produce a probability vector `[p₀, p₁, p₂, p₃, p₄]` across all 5 states. The ensemble weights for each model are then computed as a **probability-weighted combination** of the regime-specific weight vectors:

```
w_final = Σᵢ pᵢ × w_regime_i
```

This means the ensemble smoothly transitions between regime-specific strategies — no hard switching.

---

## 🧬 Multi-Task NLP Engine (DeBERTa-v3)

`src/models/nlp/model.py`

A **multi-task DeBERTa-v3-base** model performs 4 simultaneous classification tasks on financial text:

```
                    ┌─────────────────────────────────┐
                    │     DeBERTa-v3-base Encoder      │
                    │  (Frozen bottom 6 layers)         │
                    └───────────┬──────────────────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼           │
   ┌────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐      │
   │Sentiment│ │ Events  │ │ Topics │ │ Entities │      │
   │ 3-class │ │ 8-class │ │10-class│ │  6-class │      │
   └────────┘ └─────────┘ └────────┘ └──────────┘      │
        │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┘
                                │
                    ┌───────────▼──────────────────────┐
                    │  Source + Temporal Embeddings     │
                    │  (day-of-week, month-of-year)    │
                    └──────────────────────────────────┘
```

**Key Innovations:**

- **Weak Supervision Label Generation** — Automatically generates training labels from unlabeled financial text using keyword-based heuristics, eliminating the need for expensive manual annotation
- **Source Embeddings** — Learned embeddings distinguish between data sources (Reddit vs. arXiv vs. NewsAPI), allowing the model to calibrate trust per source
- **Temporal Embeddings** — Day-of-week and month-of-year embeddings capture seasonal and cyclical patterns in financial sentiment
- **Task-Weighted Loss** — Configurable loss weights per task head: `sentiment=1.0, events=0.8, topics=0.5, entities=0.3`
- **Batched NLP Inference** — Optimized batch processing for 10-50x faster inference vs. per-sample processing

---

## 🔗 Deep Fusion Architecture

`src/models/fusion/fusion.py` + `src/models/fusion/multi_horizon_fusion.py`

The **Deep Fusion Model** unifies time-series and NLP embeddings via a **Gated Modality Unit (GMU)**:

```
Time-Series Embedding (128d) ──┐
                                ├──→ GMU (Learned Gate) ──→ Fused Vector ──→ MLP Head
NLP Embedding (768d) ──────────┘                                              │
                                                                    ┌─────────┼─────────┐
                                                                    ▼         ▼         ▼
                                                                  1-Day    5-Day    30-Day
                                                                 Target   Target   Target
```

**Gated Modality Unit (GMU):**

```python
gate = σ(W_ts · h_ts + W_nlp · h_nlp + b)   # Learned sigmoid gate
fused = gate ⊙ h_ts + (1 - gate) ⊙ h_nlp     # Soft modality fusion
```

The gate learns to dynamically balance how much weight to give time-series vs. NLP signals for each prediction. The fused representation passes through 4-head multi-head attention before entering the MLP prediction head (`512 → 256 → 128 → 3 horizons`).

---

## ☁️ Cloud-Native S3 Inference Architecture

`src/cloud_storage/aws_storage.py`

The entire inference pipeline is **fully decoupled from local disk**. Model checkpoints, scalers, test data, and feature matrices can all be streamed directly from S3 into memory:

```
S3 Bucket
├── saved_models/
│   ├── lstm_forecaster.pt        ← torch.load(BytesIO(s3.get_object()))
│   ├── gru_forecaster.pt
│   ├── transformer_forecaster.pt
│   ├── tft_forecaster.pt
│   ├── nlp_model.pt
│   ├── fusion_model.pt
│   └── hmm_regime_model.pkl
├── data/processed/model_inputs/
│   ├── X_test.npy                ← np.load(BytesIO(s3_stream))
│   ├── y_test.npy
│   └── metadata_test.csv
└── data/features/
    └── scalers.joblib
```

**Resolution Order:** Local filesystem (mmap for zero-copy) → S3 streaming fallback. Controlled via `USE_S3` environment variable.

**`SimpleStorageService` capabilities:**
- `read_pickle()` — Stream pickle objects from S3
- `read_numpy()` — Stream `.npy` arrays from S3
- `read_csv()` — Stream CSV into pandas DataFrame
- `read_model()` — Stream PyTorch `.pt` checkpoints
- `upload_file()` / `upload_directory()` — Push artifacts to cloud

---

## 🔬 Hyperparameter Optimization (Optuna)

`src/training/optimization/run_hyperopt.py`

A **6-phase Optuna optimization pipeline** discovers optimal architectures:

| Phase | Target | Trials | Strategy |
|:---|:---|:---|:---|
| **Phase 3** | LSTM architecture | 30 | TPE Sampler + Median Pruner |
| **Phase 3** | GRU architecture | 30 | TPE Sampler + Median Pruner |
| **Phase 3** | Transformer architecture | 30 | TPE Sampler + Median Pruner |
| **Phase 3** | TFT architecture | 30 | TPE Sampler + Median Pruner |
| **Phase 4** | NLP learning rate | — | Deferred to GPU cluster |
| **Phase 5** | Fusion model | — | Post-embedding generation |
| **Phase 6** | Ensemble weights | 30 | Softmax-normalized optimization |

**Search spaces include:** hidden dimensions, number of layers, dropout rates, learning rates, attention heads, and ensemble weight distributions.

Best discovered parameters are persisted to `configs/best_training_config.yaml` and automatically loaded by the training scripts.

---

## 📊 Evaluation & Backtesting Suite

### Comprehensive Metrics Engine
`src/evaluation/metrics.py`

| Category | Metrics |
|:---|:---|
| **Regression** | MSE, RMSE, MAE, R², MAPE |
| **Financial** | Directional Accuracy, Annualized Sharpe Ratio, Max Drawdown, Profit Factor, Calmar Ratio |
| **Statistical** | Diebold-Mariano significance test (p < 0.05, p < 0.01), paired t-test vs. baseline |

### Walk-Forward Backtest Engine
`src/evaluation/backtest.py`

- Runs full inference pipeline over the held-out test set
- Generates **per-ticker metric breakdowns** (top/bottom 5 performers)
- Outputs JSON summary + CSV trace + per-ticker CSV report to `evaluation/results/`
- Supports `--quick` mode for 500-sample validation

### Expanding-Window Cross-Validation
`src/evaluation/cross_validation.py`

Implements **proper temporal CV** to prevent look-ahead bias:

```
Fold 1: Train [0..200]   Val [200..260]   ← Smallest training window
Fold 2: Train [0..260]   Val [260..320]   ← Expanding forward
Fold 3: Train [0..320]   Val [320..380]
Fold 4: Train [0..380]   Val [380..440]
Fold 5: Train [0..440]   Val [440..500]   ← Largest training window
```

**No shuffling** — temporal ordering is strictly preserved. Runs across all 4 TS model architectures with aggregated mean ± std reporting.

---

## 🔍 Model Explainability (SHAP + Attention)

`src/evaluation/explainability.py`

### SHAP Deep Explainer
- Uses `shap.DeepExplainer` for gradient-based feature attribution on PyTorch models
- Computes mean |SHAP| importance rankings → top-k feature DataFrame
- Generates summary plots (`bar`, `dot`, `violin`) saved to disk

### Attention Weight Extraction
- Registers **forward hooks** on all `nn.MultiheadAttention` layers
- Captures attention weight matrices `[batch, heads, seq, seq]` during forward pass
- Works with Transformer, TFT, and Fusion models
- Enables visualization of which time steps and features drove each prediction

---

## 🚨 Drift Detection & Automated Retraining

### Statistical Drift Monitor
`src/evaluation/monitoring/drift_detection.py`

Monitors feature distributions using two statistical tests:

| Test | Purpose | Threshold |
|:---|:---|:---|
| **Population Stability Index (PSI)** | Detects distribution shift magnitude | PSI > 0.25 = drift |
| **Kolmogorov-Smirnov Test** | Detects distributional differences | p < 0.05 = drift |

### Automated Retraining Trigger
`src/evaluation/monitoring/retraining_trigger.py`

```
New Data → DriftMonitor.check() → Drift Report
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
              NO_DRIFT            WARNING          CRITICAL_DRIFT
            (continue)         (log + alert)      (trigger retrain)
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Sequential Retrain │
                                              │  1. TS Ensemble     │
                                              │  2. NLP Model       │
                                              │  3. Fusion Model    │
                                              └────────────────────┘
```

- **Critical drift** is triggered when >30% of monitored features exceed both PSI and KS thresholds
- Drift decisions are logged to **MLflow** and persisted as timestamped JSON reports in `logs/drift_reports/`

---

## 🏪 Feature Store (Parquet + DuckDB)

`src/feature_engineering/feature_store/store.py`

A **versioned, centralized feature storage** system:

```
data/feature_store/
├── market_features/
│   ├── v1/
│   │   ├── market_features.parquet
│   │   └── metadata.json           ← Schema, stats, timestamps
│   └── v2/
│       ├── market_features.parquet
│       └── metadata.json
└── social_features/
    └── v1/
        └── ...
```

**Capabilities:**
- **Versioned writes** — Every `save_features()` auto-increments version
- **Metadata tracking** — Row count, column schemas, dtypes, memory footprint
- **DuckDB SQL queries** — Run analytics across Parquet files: `store.query("SELECT * FROM 'market_features' WHERE close > 100")`
- **Version comparison** — Diff schema changes between versions (added/removed columns, row deltas)

---

## 🌐 REST API (FastAPI)

`src/api/app.py`

A production-grade FastAPI application with API key authentication via `X-API-Key` header:

| Endpoint | Method | Description |
|:---|:---|:---|
| `/health` | `GET` | System health + model/data load status |
| `/predict` | `POST` | Single-ticker multi-horizon prediction |
| `/predictions/batch` | `GET` | Batch predictions for multiple tickers & date ranges |
| `/predictions/report` | `GET` | Logged prediction accuracy report |
| `/info/models` | `GET` | Loaded model architecture details |
| `/info/data-sources` | `GET` | Connected data source inventory |
| `/version` | `GET` | API version metadata |
| `/docs` | `GET` | Interactive Swagger/OpenAPI documentation |

**Prediction Strategies (cascading fallback):**
1. **Test data cache lookup** — Pre-computed feature tensors from `X_test.npy` via mmap or S3
2. **Live feature builder** — `RealTimeFeatureBuilder` constructs feature tensors on-the-fly from raw data, fully decoupled from training artifacts

**Deployment options:**
```bash
# Development
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Production (4 workers)
gunicorn src.api.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 🖥 Real-Time Command Center Dashboard

`dashboard/app.py` — A **14-module Streamlit dashboard** (2,400+ lines) serving live analytics:

| Module | Description |
|:---|:---|
| 🏠 **Overview** | Real-time KPIs, system gauges, pipeline status, collected data summaries |
| 📡 **Data Sources** | Live API connection status, file counts, record volumes, storage footprint |
| ⚙️ **Data Pipeline** | 5-stage pipeline flow visualization with per-stage file/status tracking |
| 📊 **Datasets** | Interactive dataset explorer with schema inspection and sample previews |
| 🤖 **Model Training** | Training history, loss curves, checkpoint metadata |
| 📈 **Model Evaluation** | Backtest results, regression/financial metrics, per-ticker breakdowns |
| 🔮 **Predictions** | Live multi-horizon price targets with confidence intervals and NLP signals |
| 🚨 **Anomaly & Regimes** | HMM regime visualization, anomaly detection, state transition analysis |
| 📱 **Social & NLP** | Social media signal explorer, NLP label quality, sentiment distributions |
| 🌍 **Alternative Data** | Trade, energy, research, patent, weather, jobs, blockchain signal explorers |
| 📉 **Visualization** | Candlestick charts, correlation heatmaps, crypto overviews |
| 💻 **System Monitor** | CPU/RAM/Disk gauges, process metrics, S3 connection status |
| 📋 **Logs** | Live pipeline execution logs |
| 🛠 **Settings** | Configuration management |

**S3-aware:** The dashboard auto-detects `USE_S3` and seamlessly switches between local filesystem and S3-streamed data loading, enabling full cloud deployment on Streamlit Cloud.

---

## 🔄 9-Phase Master Pipeline Orchestrator

`scripts/run_full_pipeline.py`

A single command runs the entire ML lifecycle:

```bash
python scripts/run_full_pipeline.py [--hpo] [--hpo-trials N] [--skip-collection]
```

| Phase | Name | Description |
|:---|:---|:---|
| **1** | Data Collection | Collect from all 15+ API sources |
| **2** | Data Processing | Clean, parse, compute technical indicators per domain |
| **3** | Feature Engineering + Merge | Generate 11 derived indices, merge all datasets |
| **3c** | Regime Features | HMM regime probability columns appended to merged data |
| **4** | Sequence Generation | Build 60-step windowed numpy arrays (train/val/test splits) |
| **5** | HPO (opt-in) | Optuna hyperparameter optimization across all models |
| **6** | Full Training | Train TS ensemble → NLP DeBERTa → Deep Fusion (sequential) |
| **7** | Evaluation | Walk-forward backtesting + comprehensive metric reporting |
| **8** | Regime Detection | Generate global HMM states for inference-time ensembling |
| **9** | Drift Detection | PSI + KS drift check, automated retraining trigger |

**Device auto-detection:** CUDA GPU → Apple MPS (Metal) → CPU fallback.

---

## 🔁 CI/CD Pipeline

`.github/workflows/ci-cd.yml`

A **6-stage GitHub Actions pipeline**:

| Stage | Tools | Description |
|:---|:---|:---|
| **Lint** | Black, isort, flake8 | Code formatting and quality checks |
| **Test** | pytest + coverage | Unit tests with Codecov reporting |
| **Build** | Docker Buildx | Multi-stage Docker image build + push to GHCR |
| **Security** | Trivy | Vulnerability scanning with SARIF upload |
| **Integration** | httpx | API health endpoint integration testing |
| **Docs** | markdownlint | Documentation quality validation |

Triggers on push/PR to `main` and `develop` branches.

---

## 🛠 Installation & Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.1+ (with CUDA/MPS support recommended)
- Docker & Docker Compose (for containerized deployment)

### Environment Variables

Create a `.env` file in the project root:

```env
# Cloud Storage (optional — set USE_S3=False for local-only)
USE_S3=True
MODEL_BUCKET_NAME=my-model-mlopsproj012
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# API Security
API_KEY=predictive_intel_dev_key_2026

# Logging
LOG_LEVEL=INFO
```

### Option 1: Docker Compose (Recommended)

```bash
docker-compose up --build -d
```

**Services:**
| Service | URL |
|:---|:---|
| Dashboard | `http://localhost:8501` |
| FastAPI Docs | `http://localhost:8000/docs` |
| MLflow | `http://localhost:5000` |

### Option 2: Local Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the full pipeline
python scripts/run_full_pipeline.py --skip-collection

# Start the dashboard
streamlit run dashboard/app.py

# Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Bootstrap S3 (First-time cloud setup)

```bash
python scripts/sync_to_s3.py
```

---

## 📂 Project Structure

```
AI-Predictive-Intelligence/
├── configs/
│   ├── training_config.yaml          # Central training hyperparameters
│   ├── best_training_config.yaml     # Optuna-discovered optimal params
│   └── selected_features.yaml        # Automated feature selection output
├── dashboard/
│   ├── app.py                        # 14-module Streamlit dashboard (2400+ LOC)
│   ├── styles.py                     # CSS design system + Plotly templates
│   └── utils.py                      # Data loading utilities (S3-aware)
├── src/
│   ├── api/
│   │   ├── app.py                    # FastAPI REST API (CORS, auth, lazy loading)
│   │   └── schemas.py                # Pydantic request/response models
│   ├── cloud_storage/
│   │   └── aws_storage.py            # S3 streaming service (pickle, numpy, CSV, PT)
│   ├── data_collection/
│   │   ├── async_fetcher.py          # Concurrent HTTP client (aiohttp)
│   │   ├── stocks_yahoo.py           # Yahoo Finance OHLCV collector
│   │   ├── crypto_coingecko.py       # CoinGecko market data
│   │   ├── economic_fred.py          # FRED macro-economic indicators
│   │   ├── energy_eia.py             # EIA energy supply data
│   │   ├── social_*.py               # GitHub, HN, Reddit, Mastodon, YouTube
│   │   ├── research_arxiv.py         # arXiv paper crawler
│   │   └── ...                       # 15+ collectors total
│   ├── data_processing/
│   │   ├── financial_processing.py   # OHLCV + technical indicator computation
│   │   ├── merge_datasets.py         # Multi-domain dataset stitching
│   │   └── build_sequences.py        # Windowed numpy array generation
│   ├── feature_engineering/
│   │   ├── feature_generator.py      # 11 derived composite indices
│   │   ├── feature_selection.py      # MI + variance + correlation filtering
│   │   ├── feature_store/store.py    # Versioned Parquet + DuckDB
│   │   └── regime_detection/
│   │       ├── regime_detector.py    # 5-state Gaussian HMM
│   │       └── regime_features.py    # Regime probability feature generation
│   ├── models/
│   │   ├── timeseries/
│   │   │   ├── lstm.py               # LSTM + attention pooling
│   │   │   ├── gru.py                # Stacked GRU
│   │   │   ├── transformer.py        # Causal Transformer encoder
│   │   │   └── tft.py                # Temporal Fusion Transformer (VSN + GRN)
│   │   ├── nlp/
│   │   │   ├── model.py              # Multi-task DeBERTa-v3 (4 heads)
│   │   │   └── tokenizer.py          # HuggingFace tokenizer wrapper
│   │   └── fusion/
│   │       ├── fusion.py             # GMU-based deep fusion
│   │       └── multi_horizon_fusion.py  # 1d/5d/30d multi-horizon head
│   ├── pipelines/
│   │   ├── inference_pipeline.py     # Unified Predictor (regime weights + NLP batch)
│   │   └── feature_builder.py        # RealTimeFeatureBuilder (decoupled from training)
│   ├── training/
│   │   ├── timeseries/train.py       # TS ensemble training loop
│   │   ├── nlp/train.py              # DeBERTa multi-task training
│   │   ├── fusion/train.py           # Deep fusion training
│   │   └── optimization/
│   │       ├── run_hyperopt.py       # 6-phase Optuna orchestrator
│   │       ├── optuna_ts.py          # TS model objective functions
│   │       ├── optuna_nlp.py         # NLP objective functions
│   │       └── optuna_fusion.py      # Fusion objective functions
│   ├── evaluation/
│   │   ├── backtest.py               # Walk-forward backtesting
│   │   ├── cross_validation.py       # Expanding-window temporal CV
│   │   ├── explainability.py         # SHAP + attention extraction
│   │   ├── metrics.py                # Regression + financial + statistical metrics
│   │   └── monitoring/
│   │       ├── drift_detection.py    # PSI + KS drift monitor
│   │       └── retraining_trigger.py # Automated retraining orchestrator
│   └── validation/
│       └── backtest.py               # Production backtest engine
├── scripts/
│   ├── run_full_pipeline.py          # 9-phase master orchestrator
│   ├── sync_to_s3.py                 # S3 upload utility
│   └── generate_global_regime.py     # Global HMM state generator
├── tests/
│   ├── test_models.py                # Model architecture unit tests
│   ├── test_metrics.py               # Metric computation tests
│   ├── test_data_processing.py       # Data pipeline tests
│   ├── test_feature_selection.py     # Feature selection tests
│   ├── test_api_security.py          # API authentication tests
│   └── test_pipelines_integration.py # Integration tests
├── .github/workflows/
│   └── ci-cd.yml                     # 6-stage CI/CD pipeline
├── Dockerfile                        # Multi-stage production image
├── docker-compose.yml                # Full stack deployment
├── requirements.txt                  # 77 dependencies across 12 categories
├── data_manifest.json                # Source registry (15 sources, 1.6M records)
└── setup.py                          # Editable package installation
```

---

## 📄 License & Disclaimer

Distributed under the **MIT License**. Copyright © 2026 Biswajeet Ray.

> [!WARNING]
> This platform provides complex mathematical and statistical modeling software designed **solely for educational and research purposes**. Output inferences **DO NOT constitute certified financial advice**. Do not act on investment decisions generated by these machine learning predictions without thorough human due diligence and adherence to local financial regulatory standards.

---

<div align="center">
  <sub>Built with ❤️ using PyTorch, Transformers, and an unreasonable amount of APIs</sub>
</div>
