#!/bin/bash

# Cybersecurity Intrusion Detection System - Quick Start Script
# This script sets up and runs the intrusion detection system

echo "CYBERSECURITY INTRUSION DETECTION SYSTEM"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run the setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import streamlit, pandas, numpy, sklearn, plotly, tensorflow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Some dependencies are missing. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies!"
        exit 1
    fi
fi

echo "Dependencies verified"

# Create necessary directories
echo "Creating directories..."
mkdir -p models logs data

# Run system test (optional)
echo ""
read -p "Run system validation test? (y/n): " run_test
if [ "$run_test" = "y" ] || [ "$run_test" = "Y" ]; then
    echo "Running system validation..."
    python -c "
from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering
print('Testing system components...')
try:
    ingestion = DataIngestion()
    data = ingestion.create_sample_data(100)
    feature_eng = FeatureEngineering()
    X, y = feature_eng.prepare_features_for_ml(data)
    print('System validation successful!')
except Exception as e:
    print(f'System validation failed: {e}')
    exit(1)
"
    if [ $? -ne 0 ]; then
        echo "System validation failed!"
        exit 1
    fi
fi

# Start the Streamlit application
echo ""
echo "🚀 Starting Cybersecurity Intrusion Detection Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🔍 Use Ctrl+C to stop the application"
echo ""

# Launch Streamlit
streamlit run app.py --server.port 8501 --server.headless true

echo ""
echo "Thanks for using the Cybersecurity Intrusion Detection System!"