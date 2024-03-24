# Cybersecurity Intrusion Detection System with Machine Learning

A comprehensive machine learning-based intrusion detection system that monitors network traffic in real-time to detect and alert on potential security threats. The system uses multiple ML models including Random Forest, Autoencoders, and LSTM networks to provide robust anomaly detection capabilities.

## Features

- **Multi-Model Approach**: Combines Random Forest, Autoencoder, and LSTM models for comprehensive threat detection
- **Real-Time Monitoring**: Continuous network traffic analysis with immediate alert generation
- **Interactive Dashboard**: Streamlit-based web interface for monitoring and management
- **Multiple Data Sources**: Supports CICIDS2017, NSL-KDD datasets, and PCAP files
- **Advanced Analytics**: Comprehensive reporting and trend analysis
- **Configurable Alerts**: Customizable severity levels and notification systems
- **Feature Engineering**: Automated extraction of network traffic patterns
- **Performance Monitoring**: Built-in system performance tracking

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space
- Linux/macOS/Windows

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cyber_detect
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Additional System Dependencies (Optional)

For PCAP file processing on Ubuntu/Debian:
```bash
sudo apt-get install libpcap-dev
```

For PCAP file processing on macOS:
```bash
brew install libpcap
```

## Quick Start

### 1. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### 2. Test the System

Run the comprehensive test suite:

```bash
python test_system.py
```

### 3. Basic Usage Workflow

1. **Upload Data**: Use the "Data Upload" tab to upload network traffic data (CSV or PCAP)
2. **Train Models**: Navigate to "Model Training" to train detection models
3. **Start Monitoring**: Go to "Real-time Monitoring" to begin threat detection
4. **View Alerts**: Check "Security Alerts" for detected threats
5. **Analyze Trends**: Use "Analytics" for comprehensive reporting

## Supported Data Formats

### CSV Files
- **CICIDS2017**: Canadian Institute for Cybersecurity IDS 2017 dataset
- **NSL-KDD**: Network Security Laboratory KDD dataset
- **Custom CSV**: Any CSV with network traffic features

### PCAP Files
- Standard packet capture files (.pcap, .pcapng)
- Automatic feature extraction from raw packets
- Support for various protocols (TCP, UDP, ICMP)

### Sample Data
- Built-in synthetic data generator for testing
- Configurable anomaly rates and traffic patterns

## Configuration

### Main Configuration (`config/config.py`)

```python
# Detection thresholds
DETECTION_CONFIG = {
    'anomaly_threshold': 0.7,
    'confidence_threshold': 0.5,
    'monitoring_interval': 1  # seconds
}

# Model parameters
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20
    },
    'autoencoder': {
        'encoding_dim': 32,
        'epochs': 100
    }
}
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # For GPU acceleration (optional)
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
```

## Machine Learning Models

### 1. Random Forest Classifier
- **Purpose**: Supervised classification of normal vs. malicious traffic
- **Features**: Ensemble of decision trees with feature importance ranking
- **Performance**: High accuracy with interpretable results

### 2. Autoencoder (Anomaly Detection)
- **Purpose**: Unsupervised anomaly detection based on reconstruction error
- **Architecture**: Dense neural network with bottleneck layer
- **Features**: Detects unknown attack patterns

### 3. LSTM (Long Short-Term Memory)
- **Purpose**: Sequence-based pattern recognition in network flows
- **Features**: Temporal pattern analysis for sophisticated attacks
- **Use Case**: Time-series anomaly detection

### 4. Ensemble Method
- **Combination**: Weighted average of all model predictions
- **Benefits**: Improved accuracy and reduced false positives
- **Customizable**: Adjustable model weights

## Alert System

### Severity Levels
- **🟢 Low**: Minor anomalies, routine monitoring
- **🟡 Medium**: Moderate threats requiring attention
- **🟠 High**: Serious threats needing immediate review
- **🔴 Critical**: Severe threats requiring urgent action

### Attack Types Detected
- **Port Scanning**: Reconnaissance attempts
- **DoS/DDoS Attacks**: Denial of service attempts
- **Brute Force**: Password cracking attempts
- **Data Exfiltration**: Unusual data transfer patterns
- **Unknown Anomalies**: Novel attack patterns

### Alert Features
- Real-time notifications
- Detailed threat information
- Source and destination analysis
- Confidence scoring
- Historical tracking

## Analytics and Reporting

### Real-Time Dashboard
- Live traffic monitoring
- Anomaly score tracking
- System performance metrics
- Alert status overview

### Historical Analysis
- Trend analysis over time
- Attack pattern identification
- Performance statistics
- Security posture assessment

### Custom Reports
- Exportable data and charts
- Configurable time ranges
- Filtering by severity and type
- Executive summaries

## Project Structure

```
cyber_detect/
├── app.py                 # Main Streamlit application
├── test_system.py         # Comprehensive test suite
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/
│   └── config.py         # System configuration
├── src/                  # Core modules
│   ├── data_ingestion.py # Data loading and preprocessing
│   ├── feature_engineering.py # Feature extraction
│   ├── ml_models.py      # Machine learning models
│   └── real_time_detector.py # Real-time monitoring
├── utils/
│   └── utils.py          # Utility functions
├── models/               # Trained model storage
├── data/                 # Dataset storage
└── logs/                 # Application logs
```

## Advanced Usage

### Custom Model Training

```python
from src.ml_models import MLModels
from src.feature_engineering import FeatureEngineering

# Initialize components
ml_models = MLModels()
feature_eng = FeatureEngineering()

# Prepare your data
X, y = feature_eng.prepare_features_for_ml(your_data)

# Train models with hyperparameter tuning
rf_results = ml_models.train_random_forest(X, y, hyperparameter_tuning=True)
```

### Real-Time Integration

```python
from src.real_time_detector import RealTimeAnomalyDetector


detector = RealTimeAnomalyDetector(models=ml_models, feature_engineer=feature_eng)

# Add custom alert callback
def custom_alert_handler(alert):
    print(f"THREAT DETECTED: {alert.description}")
    # Send to SIEM, email, Slack, etc.

detector.add_alert_callback(custom_alert_handler)
detector.start_monitoring()
```

### Custom Feature Engineering

```python
from src.feature_engineering import FeatureEngineering

feature_eng = FeatureEngineering()

# Extract custom features
network_features = feature_eng.extract_network_features(raw_data)
statistical_features = feature_eng.create_statistical_features(network_features)
final_features = feature_eng.create_feature_interactions(statistical_features)
```

## Testing

### Run All Tests
```bash
python test_system.py
```

### Individual Component Tests
```bash
# Test data ingestion
python -m src.data_ingestion

# Test feature engineering
python -m src.feature_engineering

# Test ML models
python -m src.ml_models

# Test real-time detector
python -m src.real_time_detector
```

### Performance Benchmarks
- **Data Processing**: ~1000 records/second
- **Feature Engineering**: ~500 records/second
- **Model Inference**: ~2000 predictions/second
- **Alert Generation**: <100ms latency

## Troubleshooting

### Common Issues

1. **TensorFlow GPU Issues**
   ```bash
   pip install tensorflow-cpu  # Use CPU version if GPU issues
   ```

2. **PCAP Processing Errors**
   ```bash
   sudo apt-get install libpcap-dev  # Install libpcap
   ```

3. **Memory Issues**
   - Reduce buffer sizes in config
   - Use smaller datasets for training
   - Enable data streaming

4. **Model Loading Errors**
   - Check model file paths
   - Verify TensorFlow/Keras compatibility
   - Retrain models if needed

### Logging

View detailed logs:
```bash
tail -f logs/integration_test.log
```

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### For Large Datasets
- Use data streaming instead of loading all data
- Implement batch processing
- Consider distributed computing frameworks

### For Real-Time Processing
- Optimize feature engineering pipeline
- Use compiled models (TensorFlow Lite)
- Implement caching strategies

### For Memory Constraints
- Reduce model complexity
- Use incremental learning
- Implement data sampling

## Security Considerations

- **Model Security**: Protect trained models from adversarial attacks
- **Data Privacy**: Implement data anonymization for sensitive networks
- **Access Control**: Secure dashboard access with authentication
- **Audit Logging**: Maintain comprehensive activity logs

## Deployment Options

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment
- Compatible with AWS, GCP, Azure
- Supports containerized deployment
- Scalable architecture

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

## Future Enhancements

- **Deep Learning Models**: Advanced neural network architectures
- **Federated Learning**: Distributed model training
- **Graph Neural Networks**: Network topology analysis
- **Automated Response**: Autonomous threat mitigation
- **Mobile App**: Mobile monitoring interface
- **API Integration**: RESTful API for external systems

## References

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Built with ❤️ for cybersecurity professionals and researchers.**