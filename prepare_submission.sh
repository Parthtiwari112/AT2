#!/usr/bin/env bash
# Local cleanup script for AT2 submission

echo "Cleaning Python caches..."
find . -name "__pycache__" -exec rm -rf {} + || true
find . -name "*.pyc" -delete || true

echo "Removing .venv directories..."
find . -type d -name ".venv" -exec rm -rf {} + || true

echo "Ensuring models folders exist..."
mkdir -p experimentation_repo/models/rain_or_not
mkdir -p experimentation_repo/models/precipitation_fall
mkdir -p api_repo/app/models

echo "✅ Cleanup complete. Replace placeholder models with your trained models before final submission."

