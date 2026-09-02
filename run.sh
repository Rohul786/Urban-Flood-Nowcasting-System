#!/bin/bash
echo "====================================================================="
echo "🌊 Starting FloodGuard AI Command Center (SIH 2026)"
echo "====================================================================="
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt --quiet
echo "Launching Streamlit..."
python3 -m streamlit run app.py
