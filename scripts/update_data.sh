#!/bin/bash
# Daily World Cup 2026 data pull — martj42 results + openfootball fixtures.
# No API key required. Mirrors the stock-pipeline launchd pattern.
set -e
DIR="/Users/fred/Documents/scylla/World Cup 2026 Predictor"
cd "$DIR"

# Use the same interpreter as the stock pipeline (Anaconda ships pandas + requests).
/opt/anaconda3/bin/python3 scripts/update_data.py

echo "WC data pull complete — $(date)"
