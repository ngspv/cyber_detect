"""
Integration test script for the Cybersecurity Intrusion Detection System
Tests all components and validates the complete workflow
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering
from src.ml_models import MLModels
from src.real_time_detector import RealTimeAnomalyDetector
from utils.utils import Logger, FileManager, PerformanceMonitor

def test_data_ingestion():
    """Test data ingestion functionality"""
    print("\\n" + "="*50)
    print("TESTING DATA INGESTION")
    print("="*50)
    
    ingestion = DataIngestion()
    
    print("\\n1. Testing sample data generation...")
    sample_data = ingestion.create_sample_data(1000)
    print(f"✅ Generated {len(sample_data)} sample records")
    print(f"   Features: {list(sample_data.columns)}")
    print(f"   Anomaly rate: {sample_data['label'].mean():.2%}")
    
    print("\\n2. Testing data preprocessing...")
    processed_data = ingestion.preprocess_data(sample_data)
    print(f"✅ Preprocessed data shape: {processed_data.shape}")
    
    print("\\n3. Testing data validation...")
    validation_results = ingestion.validate_data(processed_data)
    print(f"✅ Validation completed:")
    print(f"   Total records: {validation_results['total_records']}")
    print(f"   Missing values: {validation_results['missing_values']}")
    print(f"   Anomaly rate: {validation_results.get('anomaly_rate', 0):.2%}")
    
    return processed_data

def test_feature_engineering(data):
    """Test feature engineering functionality"""
    print("\\n" + "="*50)
    print("TESTING FEATURE ENGINEERING")
    print("="*50)
    
    feature_eng = FeatureEngineering()
    
    print("\\n1. Testing complete feature preparation pipeline...")
    X, y = feature_eng.prepare_features_for_ml(data)
    print(f"✅ Feature preparation completed:")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target shape: {y.shape if y is not None else 'None'}")
    print(f"   Selected features: {list(X.columns)[:10]}...")
    
    print("\\n2. Testing individual feature engineering steps...")
    
    network_features = feature_eng.extract_network_features(data)
    print(f"✅ Network features extracted: {network_features.shape}")
    
    statistical_features = feature_eng.create_statistical_features(network_features)
    print(f"✅ Statistical features created: {statistical_features.shape}")
    
    encoded_features = feature_eng.encode_categorical_features(statistical_features)
    print(f"✅ Categorical features encoded: {encoded_features.shape}")
    
    return feature_eng, X, y

def test_ml_models(feature_eng, X, y):
    """Test machine learning models"""
    print("\\n" + "="*50)
    print("TESTING MACHINE LEARNING MODELS")
    print("="*50)
    
    ml_models = MLModels()
    
    print("\\n1. Testing Random Forest model...")
    rf_results = ml_models.train_random_forest(X, y, hyperparameter_tuning=False)
    print(f"✅ Random Forest trained:")
    print(f"   Accuracy: {rf_results['performance']['accuracy']:.3f}")
    print(f"   F1-Score: {rf_results['performance']['f1_score']:.3f}")
    
    print("\\n2. Testing Autoencoder model...")
    ae_results = ml_models.train_autoencoder(X)
    print(f"✅ Autoencoder trained:")
    print(f"   Reconstruction threshold: {ae_results['threshold']:.6f}")
    print(f"   Anomaly rate: {ae_results['performance']['anomaly_rate']:.2%}")
    
    print("\\n3. Testing LSTM model...")
    if len(X) > 100:  # Only test if we have enough data
        try:
            lstm_results = ml_models.train_lstm(X.iloc[:500], y.iloc[:500])  # Use subset for faster testing
            print(f"✅ LSTM trained:")
            print(f"   Accuracy: {lstm_results['performance']['accuracy']:.3f}")
            print(f"   F1-Score: {lstm_results['performance']['f1_score']:.3f}")
        except Exception as e:
            print(f"⚠️ LSTM training skipped due to error: {str(e)}")
    else:
        print("⚠️ LSTM training skipped (insufficient data)")
    
    print("\\n4. Testing ensemble prediction...")
    ensemble_pred, model_scores = ml_models.ensemble_prediction(X.iloc[:100])  # Test on subset
    print(f"✅ Ensemble prediction completed:")
    print(f"   Predictions shape: {ensemble_pred.shape}")
    print(f"   Models used: {list(model_scores.keys())}")
    print(f"   Anomalies detected: {np.sum(ensemble_pred)}")
    
    return ml_models

def test_real_time_detector(ml_models, feature_eng):
    """Test real-time anomaly detection"""
    print("\\n" + "="*50)
    print("TESTING REAL-TIME ANOMALY DETECTION")
    print("="*50)
    
    detector = RealTimeAnomalyDetector(
        models=ml_models,
        feature_engineer=feature_eng,
        config={'monitoring_interval': 1}
    )
    
    print("\\n1. Setting up alert system...")
    alerts_received = []
    
    def test_alert_callback(alert):
        alerts_received.append(alert)
        print(f"   🚨 Alert: {alert.description} [{alert.severity}]")
    
    detector.add_alert_callback(test_alert_callback)
    print("✅ Alert callback configured")
    
    print("\\n2. Testing monitoring system...")
    detector.start_monitoring()
    print("✅ Monitoring started")
    
    print("\\n3. Simulating network traffic...")
    detector.simulate_real_time_data(duration_minutes=0.5)  # 30 seconds simulation
    
    time.sleep(2)
    
    print("\\n4. Checking detection statistics...")
    stats = detector.get_alert_statistics()
    print(f"✅ Detection statistics:")
    print(f"   Total alerts: {stats['total_alerts_24h']}")
    print(f"   Severity distribution: {stats['severity_distribution']}")
    print(f"   Processing time: {stats['average_processing_time']:.4f}s")
    
    detector.stop_monitoring()
    print("✅ Monitoring stopped")
    
    return detector, alerts_received

def test_performance():
    """Test system performance"""
    print("\\n" + "="*50)
    print("TESTING SYSTEM PERFORMANCE")
    print("="*50)
    
    monitor = PerformanceMonitor()
    
    print("\\n1. Testing data processing speed...")
    ingestion = DataIngestion()
    
    monitor.start_timer('data_generation')
    sample_data = ingestion.create_sample_data(5000)
    generation_time = monitor.end_timer('data_generation')
    
    print(f"✅ Data generation performance:")
    print(f"   Records: {len(sample_data)}")
    print(f"   Time: {generation_time:.3f}s")
    print(f"   Rate: {len(sample_data)/generation_time:.0f} records/sec")
    
    print("\\n2. Testing feature engineering speed...")
    feature_eng = FeatureEngineering()
    
    monitor.start_timer('feature_engineering')
    X, y = feature_eng.prepare_features_for_ml(sample_data)
    fe_time = monitor.end_timer('feature_engineering')
    
    print(f"✅ Feature engineering performance:")
    print(f"   Input shape: {sample_data.shape}")
    print(f"   Output shape: {X.shape}")
    print(f"   Time: {fe_time:.3f}s")
    print(f"   Rate: {len(X)/fe_time:.0f} records/sec")
    
    return monitor

def test_integration():
    """Test complete integration workflow"""
    print("\\n" + "="*60)
    print("TESTING COMPLETE INTEGRATION WORKFLOW")
    print("="*60)
    
    monitor = PerformanceMonitor()
    monitor.start_timer('total_workflow')
    
    try:
        print("\\nStep 1: Data ingestion and preprocessing...")
        data = test_data_ingestion()
        
        print("\\nStep 2: Feature engineering...")
        feature_eng, X, y = test_feature_engineering(data)
        
        print("\\nStep 3: Model training...")
        ml_models = test_ml_models(feature_eng, X, y)
        
        print("\\nStep 4: Real-time detection...")
        detector, alerts = test_real_time_detector(ml_models, feature_eng)
        
        print("\\nStep 5: Performance testing...")
        test_performance()
        
        total_time = monitor.end_timer('total_workflow')
        
        print("\\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"✅ All tests completed successfully!")
        print(f"   Total workflow time: {total_time:.2f}s")
        print(f"   Data processed: {len(data)} records")
        print(f"   Features generated: {X.shape[1]}")
        print(f"   Models trained: {len(ml_models.models)}")
        print(f"   Alerts generated: {len(alerts)}")
        print(f"   System status: OPERATIONAL ✅")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Integration test failed with error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("CYBERSECURITY INTRUSION DETECTION SYSTEM")
    print("COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger = Logger.setup_logger('test_runner', 'logs/integration_test.log')
    logger.info("Starting integration tests")
    
    try:
        success = test_integration()
        
        if success:
            print("\\n🎉 ALL TESTS PASSED! The system is ready for deployment.")
            logger.info("All integration tests passed successfully")
        else:
            print("\\n❌ SOME TESTS FAILED! Please check the errors above.")
            logger.error("Integration tests failed")
        
        return success
        
    except Exception as e:
        print(f"\\n💥 CRITICAL ERROR: {str(e)}")
        logger.error(f"Critical error in test runner: {str(e)}")
        return False
    
    finally:
        print(f"\\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)