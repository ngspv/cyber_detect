#!/usr/bin/env python3
"""
Quick test for the improved LSTM model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering
from src.ml_models import MLModels
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_lstm():
    print("Testing improved LSTM model...")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    data_ingestion = DataIngestion()
    data = data_ingestion.create_sample_data(1000)  # 1000 samples
    print(f"   Created {len(data)} samples")
    
    # 2. Feature engineering
    print("2. Processing features...")
    feature_eng = FeatureEngineering()
    X, y = feature_eng.prepare_features_for_ml(data)
    print(f"   Feature shape: {X.shape}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # 3. Train LSTM only
    print("3. Training LSTM model...")
    ml_models = MLModels()
    
    try:
        lstm_result = ml_models.train_lstm(X, y)
        
        print("\n✅ LSTM Training Results:")
        print(f"   Model type: {lstm_result['model_type']}")
        print(f"   Sequence length: {lstm_result['sequence_length']}")
        print(f"   Accuracy: {lstm_result['performance']['accuracy']:.4f}")
        print(f"   Precision: {lstm_result['performance']['precision']:.4f}")
        print(f"   Recall: {lstm_result['performance']['recall']:.4f}")
        print(f"   F1 Score: {lstm_result['performance']['f1_score']:.4f}")
        
        history = lstm_result['performance']['training_history']
        final_val_acc = history['val_accuracy'][-1] if 'val_accuracy' in history else 'N/A'
        final_val_loss = history['val_loss'][-1] if 'val_loss' in history else 'N/A'
        
        print(f"   Final validation accuracy: {final_val_acc}")
        print(f"   Final validation loss: {final_val_loss}")
        print(f"   Training epochs: {len(history['loss'])}")
        
        print("\n🎉 LSTM test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ LSTM test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lstm()