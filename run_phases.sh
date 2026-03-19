#!/bin/bash
# Quick Pipeline Runner - Run Phases One by One

set -e

PROJECT_ROOT="/Users/abrajput/Downloads/AI-Predictive-Intelligence"
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  AI PREDICTIVE INTELLIGENCE - INDIVIDUAL PHASE RUNNER             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Show menu
echo ""
echo "Choose which phase to run:"
echo "  1) Phase 6: Full Training (COMPLETED ✓)"
echo "  2) Phase 7: Evaluation & Backtesting"
echo "  3) Phase 8: HMM Regime Detection"
echo "  4) Run Phase 7 → Phase 8 (Sequential)"
echo "  5) Full Pipeline (All Phases 2-8)"
echo "  6) Exit"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
  1)
    echo "Running Phase 6..."
    python scripts/run_phase_6_only.py --hpo
    ;;
  2)
    echo "Running Phase 7..."
    python scripts/run_phase_7_only.py
    ;;
  3)
    echo "Running Phase 8..."
    python scripts/run_phase_8_only.py
    ;;
  4)
    echo "Running Phase 7 → Phase 8..."
    python scripts/run_phase_7_only.py
    echo ""
    echo "Phase 7 complete! Running Phase 8..."
    python scripts/run_phase_8_only.py
    ;;
  5)
    echo "Running Full Pipeline (Phases 2-8)..."
    python scripts/run_training_only.py --hpo
    ;;
  6)
    echo "Exiting..."
    exit 0
    ;;
  *)
    echo "Invalid choice!"
    exit 1
    ;;
esac

echo ""
echo "✅ Pipeline phase execution complete!"
