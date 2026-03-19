#!/bin/bash
# Quick Start Commands - AI Predictive Intelligence
# Copy and paste these commands to get started

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 AI PREDICTIVE INTELLIGENCE - QUICK START COMMANDS      ║"
echo "╚════════════════════════════════════════════════════════════╝"

# ========================================================================
# WEEK 1: PRODUCTION READINESS
# ========================================================================

echo ""
echo "📋 WEEK 1: Production Infrastructure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "1️⃣  LOCAL DEVELOPMENT SETUP"
echo ""
cat << 'EOF'
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run validation
python validate_week1_week2.py

# Start API (development mode with reload)
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start dashboard
streamlit run dashboard/app.py
EOF

echo ""
echo "2️⃣  DOCKER DEPLOYMENT"
echo ""
cat << 'EOF'
# Build all services
docker-compose build

# Start all services (API, Dashboard, Prometheus, Grafana, MLflow)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
EOF

echo ""
echo "3️⃣  TEST API ENDPOINTS"
echo ""
cat << 'EOF'
# Health check
curl -H "X-API-Key: predictive_intel_dev_key_2026" http://localhost:8000/health

# API documentation
# Open http://localhost:8000/docs in browser

# Get metrics (Prometheus format)
curl -H "X-API-Key: predictive_intel_dev_key_2026" http://localhost:8000/metrics

# Make a prediction (example)
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: predictive_intel_dev_key_2026" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "date": "2026-03-19"}'
EOF

# ========================================================================
# WEEK 2: MONITORING & OBSERVABILITY
# ========================================================================

echo ""
echo ""
echo "📊 WEEK 2: Monitoring & Observability"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "1️⃣  PROMETHEUS METRICS"
echo ""
cat << 'EOF'
# Access Prometheus dashboard
# http://localhost:9090

# Query examples:
# - Predictions per second: rate(predictions_total[5m])
# - Latency (P95): histogram_quantile(0.95, prediction_latency_seconds_bucket)
# - Feature drift: feature_psi
# - API uptime: up{job="api"}
EOF

echo ""
echo "2️⃣  GRAFANA DASHBOARDS"
echo ""
cat << 'EOF'
# Access Grafana
# http://localhost:3000
# Username: admin
# Password: admin123 (change in production!)

# Pre-configured dashboards:
# - Main Overview
# - Model Performance
# - Data Quality
# - System Health
EOF

echo ""
echo "3️⃣  DRIFT DETECTION"
echo ""
cat << 'EOF'
# Python example: Check for data drift
python -c "
from src.monitoring import DriftDetector
import pandas as pd

# Load baseline and new data
baseline = pd.read_csv('data/processed/merged/all_merged_dataset.csv')
new_data = baseline.sample(1000)  # Example

# Check drift
detector = DriftDetector(baseline.head(10000))
drift_report = detector.check_feature_drift(new_data)

# Summary
summary = detector.get_drift_summary(drift_report)
print(f'Drift severity: {summary[\"severity\"]}')
print(f'Drifted features: {summary[\"drifted_features_count\"]}')
"
EOF

echo ""
echo "4️⃣  STRUCTURED LOGGING"
echo ""
cat << 'EOF'
# View logs
tail -f logs/app_*.log

# Filter by level
grep "ERROR\|WARNING" logs/app_*.log

# JSON structured logs (set LOG_LEVEL=DEBUG for verbose)
export LOG_LEVEL=DEBUG
EOF

# ========================================================================
# TESTING
# ========================================================================

echo ""
echo ""
echo "✅ TESTING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
cat << 'EOF'
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py -v

# Integration tests
pytest tests/integration/ -v

# Coverage report
# Open htmlcov/index.html in browser

# CI/CD pipeline
# Push to GitHub → Actions tab shows workflow status
EOF

# ========================================================================
# PRODUCTION DEPLOYMENT
# ========================================================================

echo ""
echo ""
echo "🌐 PRODUCTION DEPLOYMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "AWS EC2:"
echo ""
cat << 'EOF'
# 1. Launch EC2 (Ubuntu 22.04, t3.large, 100GB storage)
# 2. SSH into instance
ssh -i key.pem ubuntu@<ip>

# 3. Setup Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh

# 4. Clone repo and deploy
git clone <repo>
cd AI-Predictive-Intelligence
docker-compose up -d

# 5. Setup Nginx reverse proxy (see DEPLOYMENT_GUIDE.md)
# 6. Setup SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot certonly --nginx -d your-domain.com
EOF

echo ""
echo "AWS ECS:"
echo ""
cat << 'EOF'
# 1. Build and push to ECR
docker build -t ai-predictive-intelligence .
docker tag ai-predictive-intelligence:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ai-predictive-intelligence:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ai-predictive-intelligence:latest

# 2. Create ECS cluster, task definition, and service (via Console or CLI)
EOF

# ========================================================================
# MONITORING SETUP
# ========================================================================

echo ""
echo ""
echo "📈 POST-DEPLOYMENT MONITORING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
cat << 'EOF'
# 1. Verify all services are running
docker-compose ps

# 2. Check API health
curl http://localhost:8000/health

# 3. Access monitoring dashboards
#    Prometheus: http://localhost:9090
#    Grafana: http://localhost:3000
#    MLflow: http://localhost:5000

# 4. Setup alerts in Grafana
#    Dashboard → Alert tab → Create alert
#    - Feature PSI > 0.25 (high drift)
#    - Prediction latency > 500ms
#    - API error rate > 5%

# 5. Setup integrations (Slack, PagerDuty, etc.)
#    Grafana → Configuration → Notification channels
EOF

# ========================================================================
# DIRECTORY STRUCTURE
# ========================================================================

echo ""
echo ""
echo "📁 PROJECT STRUCTURE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
cat << 'EOF'
AI-Predictive-Intelligence/
├── src/
│   ├── api/                      # ✨ NEW: REST API (FastAPI)
│   ├── monitoring/               # ✨ NEW: Drift detection, metrics, logging
│   ├── models/
│   ├── training/
│   ├── data_collection/
│   ├── data_processing/
│   └── ...
├── .github/workflows/            # ✨ NEW: GitHub Actions CI/CD
│   └── ci-cd.yml
├── monitoring/                   # ✨ NEW: Prometheus & Grafana config
│   ├── prometheus.yml
│   └── grafana/
├── dashboard/
├── configs/
├── saved_models/
├── logs/                         # ✨ NEW: Application logs
├── Dockerfile                    # Docker image
├── docker-compose.yml            # Docker orchestration
├── DEPLOYMENT_GUIDE.md           # ✨ NEW: Production deployment guide
├── validate_week1_week2.py       # ✨ NEW: Validation script
└── requirements.txt
EOF

# ========================================================================
# KEY FEATURES
# ========================================================================

echo ""
echo ""
echo "✨ WEEK 1 & 2 FEATURES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
cat << 'EOF'
✅ REST API (FastAPI)
   - Async inference endpoints
   - API key authentication
   - Batch prediction support
   - Auto-generated OpenAPI docs

✅ Monitoring & Observability
   - Prometheus metrics export
   - Grafana dashboards (pre-configured)
   - Structured JSON logging
   - Drift detection (PSI + KS-test)

✅ CI/CD Pipeline (GitHub Actions)
   - Code quality checks (Black, flake8)
   - Unit tests with coverage
   - Docker build & push
   - Security scanning (Trivy)
   - Integration tests
   - Automated deployment

✅ Docker Deployment
   - Multi-service orchestration
   - Health checks & auto-restart
   - Persistent volumes
   - Network isolation
   - Production-ready configuration

✅ Documentation
   - API documentation (Swagger UI)
   - Deployment guide (50+ pages)
   - Quick-start commands
   - Troubleshooting guide
   - Production checklist
EOF

# ========================================================================
# PERFORMANCE TARGETS
# ========================================================================

echo ""
echo ""
echo "🎯 PERFORMANCE TARGETS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
cat << 'EOF'
📊 API Metrics
   - Response time (P50): < 100ms
   - Response time (P95): < 500ms
   - Throughput: 150-200 requests/second
   - Uptime: > 99.5%
   - Error rate: < 0.1%

📈 Monitoring
   - Metrics collection: Every 15 seconds
   - Dashboard refresh: Real-time
   - Alert latency: < 1 minute
   - Log retention: 30 days

🔍 Data Quality
   - Feature drift detection: Enabled
   - Missing value rate: < 2%
   - Data staleness: < 24 hours
   - Prediction confidence: > 0.7
EOF

# ========================================================================
# NEXT STEPS
# ========================================================================

echo ""
echo ""
echo "🚀 NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
cat << 'EOF'
1. Run validation: python validate_week1_week2.py
2. Start local: uvicorn src.api.app:app --reload
3. Test API: curl http://localhost:8000/health
4. Docker: docker-compose up -d
5. Access dashboards:
   - API docs: http://localhost:8000/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin123)
6. Deploy to production (see DEPLOYMENT_GUIDE.md)
7. Setup monitoring alerts
8. Run tests: pytest tests/ -v

📖 For detailed instructions, see DEPLOYMENT_GUIDE.md
EOF

echo ""
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
