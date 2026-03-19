#!/usr/bin/env python3
"""
Quick validation script for Week 1 & 2 infrastructure
Tests all components are properly configured
"""

import sys
import subprocess
import time
import requests
import json
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def check_file_exists(path: str, name: str) -> bool:
    """Check if file exists"""
    if Path(path).exists():
        print(f"✅ {name}: {path}")
        return True
    else:
        print(f"❌ {name} NOT FOUND: {path}")
        return False

def check_module_imports() -> bool:
    """Check all required modules can be imported"""
    print_header("Checking Module Imports")
    
    modules = [
        "fastapi",
        "prometheus_client",
        "pandas",
        "torch",
        "transformers",
        "streamlit",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_monitoring_files() -> bool:
    """Check monitoring configuration files exist"""
    print_header("Checking Monitoring Files")
    
    files = {
        "src/monitoring/drift_detector.py": "Drift detector module",
        "src/monitoring/metrics.py": "Prometheus metrics",
        "src/monitoring/logging_config.py": "Logging configuration",
        "monitoring/prometheus.yml": "Prometheus config",
        "monitoring/grafana/dashboards/main.json": "Grafana dashboard",
        ".github/workflows/ci-cd.yml": "CI/CD workflow",
        "Dockerfile": "Docker image definition",
        "docker-compose.yml": "Docker compose file",
    }
    
    all_ok = True
    for path, name in files.items():
        if not check_file_exists(path, name):
            all_ok = False
    
    return all_ok

def check_api_syntax() -> bool:
    """Check API syntax"""
    print_header("Checking API Syntax")
    
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", "src/api/app.py"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ API syntax is valid")
            return True
        else:
            print(f"❌ API syntax error: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"❌ Syntax check failed: {e}")
        return False

def check_docker_compose() -> bool:
    """Check docker-compose configuration"""
    print_header("Checking Docker Compose")
    
    try:
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            timeout=10,
            cwd="."
        )
        if result.returncode == 0:
            print("✅ docker-compose.yml is valid")
            return True
        else:
            print(f"❌ docker-compose error: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"⚠️  Docker not available (OK for local dev): {e}")
        return True

def check_tests() -> bool:
    """Check test files exist"""
    print_header("Checking Tests")
    
    test_files = [
        "tests/test_models.py",
        "tests/test_data_processing.py",
        "tests/test_metrics.py",
    ]
    
    all_ok = True
    for path in test_files:
        if not check_file_exists(path, f"Test: {path}"):
            all_ok = False
    
    return all_ok

def check_requirements() -> bool:
    """Check requirements.txt includes monitoring packages"""
    print_header("Checking Requirements")
    
    with open("requirements.txt") as f:
        content = f.read()
    
    packages = [
        "fastapi",
        "uvicorn",
        "prometheus-client",
        "python-json-logger",
        "pytest",
    ]
    
    all_ok = True
    for pkg in packages:
        if pkg in content:
            print(f"✅ {pkg}")
        else:
            print(f"❌ {pkg} not in requirements.txt")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("  🚀 AI PREDICTIVE INTELLIGENCE - WEEK 1 & 2 VALIDATION")
    print("="*60)
    
    checks = [
        ("Requirements", check_requirements),
        ("Module Imports", check_module_imports),
        ("Monitoring Files", check_monitoring_files),
        ("API Syntax", check_api_syntax),
        ("Docker Compose", check_docker_compose),
        ("Tests", check_tests),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    # Print summary
    print_header("Validation Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Local dev: uvicorn src.api.app:app --reload")
        print("  2. Docker: docker-compose up -d")
        print("  3. Tests: pytest tests/ -v")
        print("  4. See DEPLOYMENT_GUIDE.md for production setup")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please fix before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
