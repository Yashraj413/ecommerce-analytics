#!/bin/bash
# E-Commerce Analytics Platform — Quick Start
echo "Installing dependencies..."
pip install -r requirements.txt -q

echo "Generating synthetic data (or skip and place Olist CSVs in data/raw/)..."
python src/generate_data.py

echo "Running full analytics pipeline..."
python src/etl.py
python src/rfm.py
python src/clv.py
python src/churn.py
python src/forecast.py
python src/product_performance.py
python src/generate_dashboard.py

echo ""
echo "Done! Open outputs/dashboard.html in your browser."    
