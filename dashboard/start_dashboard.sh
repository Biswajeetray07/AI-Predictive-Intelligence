#!/bin/bash

# AI Predictive Intelligence - Dashboard Startup Script
# ═══════════════════════════════════════════════════════════════════════════════

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     AI Predictive Intelligence - Dashboard Launcher       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── Check Prerequisites ──────────────────────────────────────────

echo -e "${YELLOW}[1/4]${NC} Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

if ! command -v streamlit &> /dev/null; then
    echo -e "${YELLOW}⚠ Streamlit not found, installing...${NC}"
    pip install streamlit -q
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

# ─── Check Virtual Environment ────────────────────────────────────

echo -e "${YELLOW}[2/4]${NC} Setting up environment..."

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

echo -e "${GREEN}✓ Environment ready${NC}"

# ─── Install Dependencies ────────────────────────────────────────

echo -e "${YELLOW}[3/4]${NC} Installing dependencies..."

pip install -q streamlit plotly pandas numpy requests python-json-logger prometheus-client

echo -e "${GREEN}✓ Dependencies installed${NC}"

# ─── Check API Server ────────────────────────────────────────────

echo -e "${YELLOW}[4/4]${NC} Checking services..."

API_RUNNING=false
PROMETHEUS_RUNNING=false

if curl -s http://localhost:8000/health -H "X-API-Key: predictive_intel_dev_key_2026" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API Server${NC} (http://localhost:8000)"
    API_RUNNING=true
else
    echo -e "${RED}✗ API Server${NC} (http://localhost:8000) - Not running"
fi

if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Prometheus${NC} (http://localhost:9090)"
    PROMETHEUS_RUNNING=true
else
    echo -e "${YELLOW}⚠ Prometheus${NC} (http://localhost:9090) - Not running"
fi

echo ""

if [ "$API_RUNNING" = false ]; then
    echo -e "${YELLOW}⚠ Note: Start API server before using predictions:${NC}"
    echo ""
    echo -e "  ${BLUE}Option 1 (Local Development):${NC}"
    echo "  source venv/bin/activate"
    echo "  uvicorn src.api.app:app --reload"
    echo ""
    echo -e "  ${BLUE}Option 2 (Docker):${NC}"
    echo "  docker-compose up -d"
    echo ""
fi

# ─── Launch Dashboard ─────────────────────────────────────────────

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Starting Real-Time Dashboard                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo -e "${GREEN}🎉 Dashboard starting...${NC}"
echo ""
echo "📊 URL: http://localhost:8501"
echo "📡 API: http://localhost:8000"
echo "📈 Prometheus: http://localhost:9090"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

streamlit run dashboard/app_realtime.py --logger.level=error

deactivate
