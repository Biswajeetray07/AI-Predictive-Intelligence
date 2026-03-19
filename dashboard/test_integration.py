#!/usr/bin/env python3
"""
Dashboard Integration Test
Tests that the real-time dashboard can connect to the API.
"""

import sys
import os
import requests
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "predictive_intel_dev_key_2026"
PROMETHEUS_URL = "http://localhost:9090"

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_test(name, passed, message=""):
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {status} | {name}")
    if message:
        print(f"       | {message}")

def test_api_health():
    """Test API health endpoint."""
    print(f"\n{YELLOW}Testing API Health...{RESET}")
    try:
        resp = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
            headers={"X-API-Key": API_KEY}
        )
        passed = resp.status_code == 200
        print_test("API Health Check", passed, f"Status: {resp.status_code}")
        if passed:
            data = resp.json()
            print(f"       | Version: {data.get('version', 'N/A')}")
            print(f"       | Timestamp: {data.get('timestamp', 'N/A')}")
        return passed
    except Exception as e:
        print_test("API Health Check", False, str(e))
        return False

def test_api_models():
    """Test models endpoint."""
    print(f"\n{YELLOW}Testing Models Endpoint...{RESET}")
    try:
        resp = requests.get(
            f"{API_BASE_URL}/info/models",
            timeout=3,
            headers={"X-API-Key": API_KEY}
        )
        passed = resp.status_code == 200
        print_test("Get Models Info", passed, f"Status: {resp.status_code}")
        if passed:
            data = resp.json()
            models = data.get('models', [])
            print(f"       | Models Found: {len(models)}")
            for model in models[:3]:
                print(f"       |   - {model.get('name', 'Unknown')}")
        return passed
    except Exception as e:
        print_test("Get Models Info", False, str(e))
        return False

def test_api_data_sources():
    """Test data sources endpoint."""
    print(f"\n{YELLOW}Testing Data Sources Endpoint...{RESET}")
    try:
        resp = requests.get(
            f"{API_BASE_URL}/info/data-sources",
            timeout=3,
            headers={"X-API-Key": API_KEY}
        )
        passed = resp.status_code == 200
        print_test("Get Data Sources", passed, f"Status: {resp.status_code}")
        if passed:
            data = resp.json()
            sources = data.get('sources', [])
            print(f"       | Data Sources Found: {len(sources)}")
        return passed
    except Exception as e:
        print_test("Get Data Sources", False, str(e))
        return False

def test_prediction():
    """Test prediction endpoint."""
    print(f"\n{YELLOW}Testing Prediction Endpoint...{RESET}")
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            json={"ticker": "AAPL", "date": datetime.now().strftime("%Y-%m-%d")},
            timeout=10,
            headers={"X-API-Key": API_KEY}
        )
        passed = resp.status_code == 200
        print_test("Get Prediction", passed, f"Status: {resp.status_code}")
        if passed:
            data = resp.json()
            print(f"       | Ticker: {data.get('ticker', 'N/A')}")
            print(f"       | Inference Time: {data.get('inference_time_ms', 'N/A')}ms")
        return passed
    except Exception as e:
        print_test("Get Prediction", False, str(e))
        return False

def test_prometheus():
    """Test Prometheus connection."""
    print(f"\n{YELLOW}Testing Prometheus...{RESET}")
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/-/healthy",
            timeout=3
        )
        passed = resp.status_code == 200
        print_test("Prometheus Health", passed, f"Status: {resp.status_code}")
        
        # Try to query metrics
        resp2 = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "predictions_total"},
            timeout=3
        )
        metrics_ok = resp2.status_code == 200
        print_test("Query Metrics", metrics_ok, f"Status: {resp2.status_code}")
        
        return passed and metrics_ok
    except Exception as e:
        print_test("Prometheus Health", False, str(e))
        return False

def test_metrics_endpoint():
    """Test metrics endpoint."""
    print(f"\n{YELLOW}Testing Metrics Endpoint...{RESET}")
    try:
        resp = requests.get(
            f"{API_BASE_URL}/metrics",
            timeout=3,
            headers={"X-API-Key": API_KEY}
        )
        passed = resp.status_code == 200
        print_test("Get Prometheus Metrics", passed, f"Status: {resp.status_code}")
        if passed:
            lines = resp.text.split('\n')
            metric_lines = [l for l in lines if l and not l.startswith('#')]
            print(f"       | Metrics Found: {len(metric_lines)}")
        return passed
    except Exception as e:
        print_test("Get Prometheus Metrics", False, str(e))
        return False

def main():
    """Run all tests."""
    print_header("Real-Time Dashboard Integration Test")

    print(f"Testing Connection to:")
    print(f"  • API: {BLUE}{API_BASE_URL}{RESET}")
    print(f"  • Prometheus: {BLUE}{PROMETHEUS_URL}{RESET}")

    results = {
        "API Health": test_api_health(),
        "Models Info": test_api_models(),
        "Data Sources": test_api_data_sources(),
        "Prediction": test_prediction(),
        "Prometheus": test_prometheus(),
        "Metrics": test_metrics_endpoint(),
    }

    print_header("Test Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = f"{GREEN}✓{RESET}" if result else f"{RED}✗{RESET}"
        print(f"  {status} {test_name}")

    print(f"\n  Result: {passed}/{total} tests passed\n")

    if passed == total:
        print(f"{GREEN}✓ All tests passed! Dashboard is ready.{RESET}\n")
        print(f"Start the dashboard with:")
        print(f"  {BLUE}cd dashboard && bash start_dashboard.sh{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Check services are running.{RESET}\n")
        print(f"Start services with:")
        print(f"  {BLUE}docker-compose up -d{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
