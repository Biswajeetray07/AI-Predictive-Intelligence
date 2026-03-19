#!/bin/bash
# Quick Reference - Real-Time Dashboard
# Run this to see all the commands at once

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                    🧠 DASHBOARD QUICK REFERENCE                          ║
╚════════════════════════════════════════════════════════════════════════════╝


🚀 START IMMEDIATELY
════════════════════════════════════════════════════════════════════════════

# Local Development (Best for Testing)
source venv/bin/activate
uvicorn src.api.app:app --reload &  # Terminal 1
cd dashboard && bash start_dashboard.sh  # Terminal 2

# Docker (Best for Production)
docker-compose up -d

# Manual
streamlit run dashboard/app_realtime.py


🌐 ACCESS URLS
════════════════════════════════════════════════════════════════════════════

Dashboard:          http://localhost:8501
API Docs:           http://localhost:8000/docs
Prometheus:         http://localhost:9090
Grafana:            http://localhost:3000
MLflow:             http://localhost:5000


📊 DASHBOARD PAGES
════════════════════════════════════════════════════════════════════════════

🏠 Dashboard        → Real-time metrics, predictions, performance
🔮 Predictions      → Make individual predictions
📊 Metrics          → Performance gauges and monitoring
🚨 Anomalies        → Drift detection and alerts
⚙️  Settings        → Configuration and service health


✅ TESTING
════════════════════════════════════════════════════════════════════════════

# Run all integration tests
python dashboard/test_integration.py

# Test API manually
curl -H "X-API-Key: predictive_intel_dev_key_2026" \
  http://localhost:8000/health

# Test Prometheus
curl http://localhost:9090/-/healthy

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: predictive_intel_dev_key_2026" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "date": "2026-03-19"}'


📚 DOCUMENTATION
════════════════════════════════════════════════════════════════════════════

DASHBOARD_SETUP.md       → Quick 3-step start guide
README_REALTIME.md       → Full documentation
DASHBOARD_COMPLETE.md    → Comprehensive summary
test_integration.py      → Test suite


🔧 CONFIGURATION
════════════════════════════════════════════════════════════════════════════

Edit dashboard/app_realtime.py (lines 18-20):

  API_BASE_URL = "http://localhost:8000"
  API_KEY = "predictive_intel_dev_key_2026"
  PROMETHEUS_URL = "http://localhost:9090"


📁 FILES CREATED
════════════════════════════════════════════════════════════════════════════

dashboard/app_realtime.py        (23 KB) - Main dashboard
dashboard/realtime_utils.py      (11 KB) - API utilities
dashboard/test_integration.py    (6.5 KB) - Tests
dashboard/start_dashboard.sh     (4.3 KB) - Startup script
dashboard/README_REALTIME.md     (9.6 KB) - Docs
DASHBOARD_SETUP.md                       - Quick start
DASHBOARD_COMPLETE.md                    - Summary


🎯 TYPICAL WORKFLOW
════════════════════════════════════════════════════════════════════════════

1. Start API
   source venv/bin/activate
   uvicorn src.api.app:app --reload

2. Start Dashboard
   cd dashboard && bash start_dashboard.sh

3. Open browser
   http://localhost:8501

4. Make predictions
   - Select ticker
   - Click "Get Prediction"
   - View result

5. Monitor metrics
   - Check real-time throughput
   - Monitor latency
   - View error rates

6. Check drift
   - Navigate to "Anomalies" page
   - Review feature drift
   - Check alerts


⚡ QUICK COMMANDS
════════════════════════════════════════════════════════════════════════════

Start everything:
  docker-compose up -d

Start dashboard only:
  cd dashboard && bash start_dashboard.sh

Stop all services:
  docker-compose down

View logs:
  docker-compose logs -f

Check service health:
  curl http://localhost:8000/health

Run tests:
  python dashboard/test_integration.py

Clear cache:
  # In dashboard, click "Refresh Metrics" button
  # OR manually: st.cache_data.clear()


🐛 TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════

API Offline:
  curl http://localhost:8000/health
  # If fails, start: uvicorn src.api.app:app --reload

Prometheus Down:
  curl http://localhost:9090/-/healthy
  # If fails, start: docker-compose up -d prometheus

Dashboard Won't Load:
  pip install --upgrade streamlit
  streamlit run dashboard/app_realtime.py --logger.level=debug

Port 8501 in Use:
  streamlit run dashboard/app_realtime.py --server.port=8502


📞 SUPPORT
════════════════════════════════════════════════════════════════════════════

Documentation:
  - DASHBOARD_SETUP.md (quick start)
  - README_REALTIME.md (full guide)
  - DASHBOARD_COMPLETE.md (complete reference)

Testing:
  - python dashboard/test_integration.py

API Documentation:
  - http://localhost:8000/docs


✨ FEATURES AT A GLANCE
════════════════════════════════════════════════════════════════════════════

Real-Time:
  ✓ Live predictions
  ✓ Auto-refreshing metrics
  ✓ Prometheus integration
  ✓ 5 interactive pages

Monitoring:
  ✓ Drift detection
  ✓ Performance tracking
  ✓ Error monitoring
  ✓ Health checks

Design:
  ✓ Dark theme
  ✓ Responsive UI
  ✓ Color-coded alerts
  ✓ Beautiful charts


🎊 NEXT STEPS
════════════════════════════════════════════════════════════════════════════

1. Start → bash dashboard/start_dashboard.sh
2. Open  → http://localhost:8501
3. Test  → Make a prediction
4. Monitor → Check real-time metrics
5. Deploy → Follow DASHBOARD_SETUP.md


════════════════════════════════════════════════════════════════════════════

Status: ✅ Production Ready
Version: 1.0 (Real-Time)
Last Updated: 2026-03-19

════════════════════════════════════════════════════════════════════════════

EOF
