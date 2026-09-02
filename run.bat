@echo off
title FloodGuard AI - Urban Flood Nowcasting & Emergency Response System
echo =====================================================================
echo 🌊 Starting FloodGuard AI Command Center (SIH 2026)
echo =====================================================================
echo Checking dependencies and launching Streamlit...
echo.

py -m pip install -r requirements.txt --quiet
py -m streamlit run app.py

pause
