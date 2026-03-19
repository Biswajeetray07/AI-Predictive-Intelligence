# 🧠 Real-Time Dashboard Guide

## Overview

The **AI Predictive Intelligence Real-Time Dashboard** is a production-grade Streamlit application that connects to your FastAPI REST API and provides live monitoring, predictions, and drift detection.

## Features

✅ **Real-Time Monitoring**
- Live metrics from Prometheus (predictions/sec, latency, errors)
- API health status and performance tracking
- Auto-refreshing data with configurable intervals

✅ **Live Predictions**
- Make predictions on-demand for any ticker
- Get batch predictions for multiple assets
- View prediction confidence and signals

✅ **Model Management**
- View active models and their metrics
- Monitor model accuracy and performance
- Track model updates and versions

✅ **Drift Detection**
- Monitor feature drift using PSI (Population Stability Index)
- Detect prediction drift automatically
- Get alerts for data quality issues

✅ **Beautiful UI**
- Bloomberg Terminal / Palantir-inspired dark theme
- Responsive design for all devices
- Neon color scheme with glowing accents

## Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Start all services (API, Dashboard, Prometheus, Grafana, MLflow)
docker-compose up -d

# Dashboard will be available at http://localhost:8501
```

### Option 2: Local Development

```bash
# Terminal 1: Start the API
source venv/bin/activate
uvicorn src.api.app:app --reload --port 8000

# Terminal 2: Start the Dashboard
cd dashboard
bash start_dashboard.sh

# OR manually:
streamlit run app_realtime.py
```

### Option 3: Custom Installation

```bash
# Install dependencies
pip install streamlit plotly pandas numpy requests

# Run dashboard
streamlit run dashboard/app_realtime.py
```

## Dashboard Pages

### 🏠 Dashboard (Home)

Main command center with:
- **Real-time metrics**: Predictions/sec, latency, error rate, active models
- **Live predictions**: Get instant predictions for any ticker
- **Model info**: Active models with accuracy metrics
- **Performance timeline**: 24-hour API performance graph

### 🔮 Predictions

Make predictions with:
- Ticker/asset selection
- Date picker
- Full prediction response with price, confidence, and signal

**Example Response:**
```json
{
  "ticker": "AAPL",
  "date": "2026-03-19",
  "prediction": {
    "price": 182.45,
    "confidence": 0.87,
    "direction": "BUY",
    "signal_strength": "STRONG"
  },
  "models_used": ["LSTM", "GRU", "TFT"],
  "inference_time_ms": 127
}
```

### 📊 Metrics

Real-time performance metrics:
- **Predictions/sec**: Gauge chart showing throughput
- **Avg Latency**: Response time monitoring
- **Error Rate**: Error percentage tracking
- **Resource Usage**: CPU, memory, disk monitoring

### 🚨 Anomalies

Data quality and drift monitoring:
- Feature drift detection (PSI scores)
- Prediction drift monitoring
- Performance degradation alerts
- Recent alert history

### ⚙️ Settings

Configuration options:
- API and Prometheus URLs
- Service health checks
- Documentation links
- Quick access to external tools

## Architecture

```
┌─────────────────────┐
│  Streamlit Dashboard│  (Port 8501)
│  (app_realtime.py)  │
└──────────┬──────────┘
           │
    ┌──────┴──────┬────────────┬────────────┐
    │             │            │            │
    ▼             ▼            ▼            ▼
┌────────┐  ┌────────┐   ┌──────────┐  ┌──────────┐
│  FastAPI    │Prometheus│Grafana  │ MLflow
│  API     │ (9090)   │(3000)   │ (5000)
│(8000)   │          │         │
└────────┘  └────────┘   └──────────┘  └──────────┘
```

## Configuration

### Environment Variables

Create `.streamlit/secrets.toml` or use environment variables:

```toml
API_BASE_URL = "http://localhost:8000"
API_KEY = "predictive_intel_dev_key_2026"
PROMETHEUS_URL = "http://localhost:9090"
REFRESH_INTERVAL = 10  # seconds
```

### Streamlit Config

`~/.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#00F5D4"
backgroundColor = "#0B0F1A"
secondaryBackgroundColor = "#111827"
textColor = "#EAEAEA"
font = "sans serif"

[client]
showErrorDetails = false

[server]
port = 8501
headless = true
```

## API Integration

The dashboard connects to these API endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/predict` | POST | Get single prediction |
| `/predictions/batch` | GET | Get batch predictions |
| `/predictions/report` | GET | Get accuracy metrics |
| `/info/models` | GET | List active models |
| `/info/data-sources` | GET | List data sources |
| `/metrics` | GET | Get Prometheus metrics |

## Real-Time Features

### Auto-Refresh Mechanism

```python
# Dashboard auto-refreshes every N seconds
@st.cache_data(ttl=10)  # 10-second cache
def get_prometheus_metrics():
    # Fetch live metrics
    ...
```

### Live Prediction

```python
# Make prediction on-demand
prediction = get_live_prediction("AAPL")

# Response includes:
# - Predicted price
# - Confidence score
# - Buy/Hold/Sell signal
# - Models used
# - Inference time
```

### Drift Detection

```python
# Automatically monitor for drift
drift_report = {
    'feature_drift': {
        'AAPL_volume': {'psi': 0.08, 'status': 'NORMAL'},
        'market_sentiment': {'psi': 0.35, 'status': 'ALERT'},
    },
    'prediction_drift': {
        'psi': 0.14,
        'status': 'NORMAL',
    }
}
```

## Troubleshooting

### Dashboard Won't Start

```bash
# Check Streamlit installation
pip list | grep streamlit

# Reinstall if needed
pip install --upgrade streamlit

# Run with verbose logging
streamlit run dashboard/app_realtime.py --logger.level=debug
```

### API Connection Issues

**Error:** "API Server Offline"

```bash
# Check if API is running
curl -H "X-API-Key: predictive_intel_dev_key_2026" http://localhost:8000/health

# Start API if not running
docker-compose up -d api
# OR
uvicorn src.api.app:app --reload
```

### Metrics Not Loading

**Error:** Prometheus queries returning no data

```bash
# Check Prometheus is running
curl http://localhost:9090/-/healthy

# Start Prometheus
docker-compose up -d prometheus

# Verify API is sending metrics
curl http://localhost:8000/metrics
```

### Dashboard Slow/Laggy

```bash
# Increase cache TTL in app_realtime.py
@st.cache_data(ttl=30)  # Increase from 10
def get_prometheus_metrics():
    ...

# OR restart with better resources
docker-compose up -d --scale dashboard=1
```

## Performance Tuning

### Optimize Refresh Rate

```python
# Balance between freshness and performance
# Lower = More responsive, higher CPU
# Higher = Less responsive, lower CPU

# Recommended:
# - Development: 5-10 seconds
# - Production: 30-60 seconds

refresh_interval = st.slider("Refresh every (seconds)", 5, 60, 10)
```

### Cache Management

```python
# Cache expensive API calls
@st.cache_data(ttl=30)
def get_api_info():
    return requests.get(API_BASE_URL + "/info/models").json()

# Clear cache manually
if st.button("Refresh Metrics"):
    st.cache_data.clear()
    st.rerun()
```

### Database Optimization

For larger deployments:

1. **Use TimescaleDB** for metrics
2. **Enable Prometheus compression**
3. **Implement metric retention policies**

## Deployment

### Local Docker

```bash
docker-compose up -d dashboard

# Access at http://localhost:8501
```

### AWS EC2

```bash
# Install Docker
sudo yum install docker -y

# Clone and start
git clone <repo>
cd AI-Predictive-Intelligence
docker-compose up -d

# Access via: http://<EC2-PUBLIC-IP>:8501
```

### Kubernetes

```bash
kubectl apply -f k8s/dashboard-deployment.yaml

# Port forward
kubectl port-forward service/dashboard 8501:8501
```

### Production Nginx Proxy

```nginx
server {
    listen 80;
    server_name dashboard.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Advanced Usage

### Custom Pages

Add new pages to `app_realtime.py`:

```python
elif page == "🔧 Custom":
    st.markdown("<div class='section-title'>Custom Page</div>", unsafe_allow_html=True)
    
    # Your custom content here
    st.write("Custom page content")
```

### Real-Time Alerts

```python
if drift_alert:
    st.error(f"🚨 Feature drift detected: {feature_name}")
    # Send notification
    send_slack_alert(f"Drift in {feature_name}")
```

### Export Data

```python
# Export metrics to CSV
metrics_df = get_metrics_timeseries("predictions_total")
st.download_button(
    "Download Metrics",
    metrics_df.to_csv(),
    "metrics.csv"
)
```

## Monitoring Dashboard

For more detailed monitoring, use Grafana:

```
http://localhost:3000
- Username: admin
- Password: admin
```

Pre-configured dashboards:
- API Performance
- Model Metrics
- Data Quality
- System Resources

## Support & Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **MLflow**: http://localhost:5000

## Next Steps

1. ✅ Start the dashboard with `docker-compose up -d`
2. ✅ Access at http://localhost:8501
3. ✅ Make test predictions
4. ✅ Monitor real-time metrics
5. ✅ Set up alerts in Grafana
6. ✅ Deploy to production

---

**Status**: 🟢 Production Ready

Last Updated: 2026-03-19
