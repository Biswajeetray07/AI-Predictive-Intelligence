# AI Predictive Intelligence Platform

A multi-modal AI system for market prediction using time-series analysis, NLP intelligence, and deep fusion modeling.

## Features

- **27 Data Collectors** across 15 domains (finance, news, social media, economy, weather, etc.)
- **4-Model Time-Series Ensemble** — LSTM, GRU, Transformer (causal-masked), TFT
- **DeBERTa-v3 NLP Engine** — Multi-task learning for sentiment, events, topics, entities
- **Deep Fusion Model** — Attention-based modality gating with GMU
- **HMM Regime Detection** — 5-state market regime classification
- **MLflow Experiment Tracking** — Full training lifecycle management
- **Drift Monitoring** — PSI + KS-test based feature drift detection

## Quick Start

```bash
# Create and activate virtual environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/ -v

# Run training pipeline
python -m src.pipelines.training_pipeline
```

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## License

See [LICENSE](LICENSE) for details.