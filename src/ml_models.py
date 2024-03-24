"""
Machine Learning Models Module for Cybersecurity Intrusion Detection
Implements Random Forest, Autoencoder, and LSTM models for anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

class MLModels:
    """
    Machine Learning models for intrusion detection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize ML models
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        self.models = {}
        self.model_performance = {}
        
        self.default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'autoencoder': {
                'encoding_dim': 32,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'lstm': {
                'lstm_units': 32,
                'epochs': 30,
                'batch_size': 32,
                'sequence_length': 5,
                'learning_rate': 0.0001
            }
        }
        
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for ML models"""
        logger = logging.getLogger('MLModels')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train Random Forest classifier for intrusion detection
        
        Args:
            X: Feature DataFrame
            y: Target Series
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with model and performance metrics
        """
        try:
            self.logger.info("Training Random Forest classifier")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            if hyperparameter_tuning:
                self.logger.info("Performing hyperparameter tuning for Random Forest")
                
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                rf_base = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(
                    rf_base, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                best_params = grid_search.best_params_
                self.logger.info(f"Best parameters: {best_params}")
                rf_model = grid_search.best_estimator_
                
            else:
                rf_params = self.default_params['random_forest']
                rf_model = RandomForestClassifier(**rf_params)
                rf_model.fit(X_train, y_train)
            
            y_pred = rf_model.predict(X_test)
            
            y_pred_proba_full = rf_model.predict_proba(X_test)
            if y_pred_proba_full.shape[1] > 1:
                y_pred_proba = y_pred_proba_full[:, 1]
            else:
                y_pred_proba = y_pred_proba_full[:, 0]
            
            performance = self._calculate_performance_metrics(y_test, y_pred, y_pred_proba)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            model_info = {
                'model': rf_model,
                'performance': performance,
                'feature_importance': feature_importance,
                'model_type': 'random_forest'
            }
            
            self.models['random_forest'] = model_info
            self.model_performance['random_forest'] = performance
            
            self.logger.info(f"Random Forest training completed. Accuracy: {performance['accuracy']:.4f}")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {str(e)}")
            raise
    
    def train_autoencoder(self, X: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Train Autoencoder for anomaly detection
        
        Args:
            X: Feature DataFrame
            contamination: Expected proportion of anomalies
            
        Returns:
            Dictionary with model and performance metrics
        """
        try:
            self.logger.info("Training Autoencoder for anomaly detection")
            
            X_normal = X.copy()
            
            X_train, X_test = train_test_split(X_normal, test_size=0.2, random_state=42)
            
            autoencoder_params = self.default_params['autoencoder']
            input_dim = X_train.shape[1]
            encoding_dim = autoencoder_params['encoding_dim']
            
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(128, activation='relu')(input_layer)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(0.2)(encoded)
            encoded = Dense(64, activation='relu')(encoded)
            encoded = Dense(encoding_dim, activation='relu')(encoded)
            
            decoded = Dense(64, activation='relu')(encoded)
            decoded = Dropout(0.2)(decoded)
            decoded = Dense(128, activation='relu')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dense(input_dim, activation='linear')(decoded)
            
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(
                optimizer=Adam(learning_rate=autoencoder_params['learning_rate']),
                loss='mse'
            )
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            ]
            
            history = autoencoder.fit(
                X_train, X_train,
                epochs=autoencoder_params['epochs'],
                batch_size=autoencoder_params['batch_size'],
                validation_split=0.1,
                callbacks=callbacks,
                verbose=1
            )
            
            train_predictions = autoencoder.predict(X_train)
            train_errors = np.mean(np.square(X_train - train_predictions), axis=1)
            threshold = np.percentile(train_errors, (1 - contamination) * 100)
            
            test_predictions = autoencoder.predict(X_test)
            test_errors = np.mean(np.square(X_test - test_predictions), axis=1)
            
            test_anomalies = (test_errors > threshold).astype(int)
            
            performance = {
                'threshold': threshold,
                'mean_reconstruction_error': np.mean(test_errors),
                'std_reconstruction_error': np.std(test_errors),
                'anomaly_rate': np.mean(test_anomalies),
                'training_history': history.history
            }
            
            model_info = {
                'model': autoencoder,
                'threshold': threshold,
                'performance': performance,
                'model_type': 'autoencoder'
            }
            
            self.models['autoencoder'] = model_info
            self.model_performance['autoencoder'] = performance
            
            self.logger.info(f"Autoencoder training completed. Threshold: {threshold:.6f}")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training Autoencoder: {str(e)}")
            raise
    
    def train_lstm(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train LSTM model for sequence-based intrusion detection
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with model and performance metrics
        """
        try:
            self.logger.info("Training LSTM model for sequence-based detection")
            
            lstm_params = self.default_params['lstm']
            sequence_length = lstm_params['sequence_length']
            
            X_seq, y_seq = self._prepare_sequential_data(X, y, sequence_length)
            
            if len(X_seq) < 100:
                self.logger.warning(f"Limited sequential data ({len(X_seq)} samples). Using simpler LSTM.")
                return self._train_simple_lstm(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
            )
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            
            model = Sequential([
                LSTM(32, return_sequences=False, 
                     input_shape=(sequence_length, X.shape[1]),
                     dropout=0.3, recurrent_dropout=0.3),
                BatchNormalization(),
                Dense(16, activation='relu'),
                Dropout(0.4),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, mode='max'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7)
            ]
            
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
            
            history = model.fit(
                X_train_scaled, y_train,
                epochs=lstm_params['epochs'],
                batch_size=min(lstm_params['batch_size'], len(X_train) // 4),
                validation_data=(X_test_scaled, y_test),
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            performance = self._calculate_performance_metrics(y_test, y_pred, y_pred_proba.flatten())
            performance['training_history'] = history.history
            performance['sequence_length'] = sequence_length
            
            model_info = {
                'model': model,
                'scaler': scaler,
                'performance': performance,
                'sequence_length': sequence_length,
                'model_type': 'lstm'
            }
            
            self.models['lstm'] = model_info
            self.model_performance['lstm'] = performance
            
            self.logger.info(f"LSTM training completed. Accuracy: {performance['accuracy']:.4f}")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training LSTM: {str(e)}")
            raise
    
    def _train_simple_lstm(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train a simple LSTM for cases with limited sequential data
        """
        try:
            self.logger.info("Training simple LSTM model (limited sequential data)")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train_reshaped = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_reshaped = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.values).reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_scaled = scaler.transform(X_test.values).reshape(X_test.shape[0], 1, X_test.shape[1])
            
            model = Sequential([
                LSTM(16, input_shape=(1, X.shape[1]), dropout=0.2),
                Dense(8, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max')
            ]
            
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
            
            history = model.fit(
                X_train_scaled, y_train,
                epochs=30,
                batch_size=32,
                validation_data=(X_test_scaled, y_test),
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            performance = self._calculate_performance_metrics(y_test, y_pred, y_pred_proba.flatten())
            performance['training_history'] = history.history
            performance['sequence_length'] = 1
            
            model_info = {
                'model': model,
                'scaler': scaler,
                'performance': performance,
                'sequence_length': 1,
                'model_type': 'lstm_simple'
            }
            
            self.models['lstm'] = model_info
            self.model_performance['lstm'] = performance
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training simple LSTM: {str(e)}")
            raise
    
    def _prepare_sequential_data(self, X: pd.DataFrame, y: pd.Series, 
                               sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM training with better handling
        
        Args:
            X: Feature DataFrame
            y: Target Series
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (sequential features, sequential targets)
        """
        if len(X) < sequence_length * 2:
            self.logger.warning(f"Dataset too small for sequence length {sequence_length}")
            sequence_length = max(1, len(X) // 3)
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            seq_data = X.iloc[i-sequence_length:i].values
            
            if seq_data.shape[0] == sequence_length:
                sequences.append(seq_data)
                targets.append(y.iloc[i])
        
        if len(sequences) == 0:
            sequences = [X.iloc[i:i+1].values for i in range(len(X))]
            targets = y.values.tolist()
            sequence_length = 1
        
        sequences_array = np.array(sequences)
        targets_array = np.array(targets)
        
        self.logger.info(f"Created {len(sequences_array)} sequences of length {sequence_length}")
        return sequences_array, targets_array
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
            
            unique_classes = len(np.unique(y_true))
            if unique_classes > 1 and len(y_pred_proba) > 0:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                except ValueError:
                    metrics['roc_auc'] = 0.5
            else:
                metrics['roc_auc'] = 0.5
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.5
            }
    
    def ensemble_prediction(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions using all trained models
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (ensemble predictions, individual model scores)
        """
        try:
            self.logger.info("Making ensemble predictions")
            
            predictions = {}
            scores = {}
            
            if 'random_forest' in self.models:
                rf_model = self.models['random_forest']['model']
                rf_pred_proba_full = rf_model.predict_proba(X)
                
                if rf_pred_proba_full.shape[1] > 1:
                    rf_pred = rf_pred_proba_full[:, 1]
                else:
                    rf_pred = rf_pred_proba_full[:, 0]
                
                predictions['random_forest'] = rf_pred
                scores['random_forest'] = rf_pred
            
            if 'autoencoder' in self.models:
                ae_model = self.models['autoencoder']['model']
                ae_threshold = self.models['autoencoder']['threshold']
                
                ae_pred = ae_model.predict(X)
                reconstruction_errors = np.mean(np.square(X - ae_pred), axis=1)
                ae_scores = (reconstruction_errors - ae_threshold) / ae_threshold
                ae_scores = np.clip(ae_scores, 0, 1)  # Normalize to [0, 1]
                
                predictions['autoencoder'] = ae_scores
                scores['autoencoder'] = ae_scores
            
            if 'lstm' in self.models and len(X) >= self.models['lstm']['sequence_length']:
                lstm_model = self.models['lstm']['model']
                sequence_length = self.models['lstm']['sequence_length']
                
                X_seq = []
                for i in range(sequence_length, len(X)):
                    X_seq.append(X.iloc[i-sequence_length:i].values)
                X_seq = np.array(X_seq)
                
                if len(X_seq) > 0:
                    lstm_pred = lstm_model.predict(X_seq).flatten()
                    
                    lstm_scores = np.zeros(len(X))
                    lstm_scores[sequence_length:] = lstm_pred
                    
                    predictions['lstm'] = lstm_scores
                    scores['lstm'] = lstm_scores
            
            if predictions:
                ensemble_scores = np.mean(list(predictions.values()), axis=0)
                ensemble_pred = (ensemble_scores > 0.5).astype(int)
                
                self.logger.info(f"Ensemble prediction completed using {len(predictions)} models")
                return ensemble_pred, scores
            else:
                self.logger.warning("No trained models available for ensemble prediction")
                return np.zeros(len(X)), {}
                
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {str(e)}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results for each model
        """
        evaluation_results = {}
        
        for model_name, model_info in self.models.items():
            try:
                self.logger.info(f"Evaluating {model_name} model")
                
                if model_name == 'random_forest':
                    y_pred = model_info['model'].predict(X_test)
                    y_pred_proba = model_info['model'].predict_proba(X_test)[:, 1]
                    
                elif model_name == 'autoencoder':
                    ae_pred = model_info['model'].predict(X_test)
                    reconstruction_errors = np.mean(np.square(X_test - ae_pred), axis=1)
                    threshold = model_info['threshold']
                    y_pred = (reconstruction_errors > threshold).astype(int)
                    y_pred_proba = reconstruction_errors / np.max(reconstruction_errors)
                    
                elif model_name == 'lstm':
                    sequence_length = model_info['sequence_length']
                    if len(X_test) >= sequence_length:
                        X_seq, y_seq = self._prepare_sequential_data(X_test, y_test, sequence_length)
                        y_pred_proba = model_info['model'].predict(X_seq).flatten()
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        y_test = y_seq  # Use sequential targets
                    else:
                        continue
                
                metrics = self._calculate_performance_metrics(y_test, y_pred, y_pred_proba)
                evaluation_results[model_name] = metrics
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return evaluation_results
    
    def save_models(self, save_dir: str) -> None:
        """
        Save all trained models
        
        Args:
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model_info in self.models.items():
            try:
                if model_name in ['autoencoder', 'lstm']:
                    model_path = os.path.join(save_dir, f"{model_name}_model.h5")
                    model_info['model'].save(model_path)
                    
                    info_path = os.path.join(save_dir, f"{model_name}_info.pkl")
                    model_info_copy = model_info.copy()
                    del model_info_copy['model']  # Remove model object
                    joblib.dump(model_info_copy, info_path)
                    
                else:
                    model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
                    joblib.dump(model_info, model_path)
                
                self.logger.info(f"Saved {model_name} model to {save_dir}")
                
            except Exception as e:
                self.logger.error(f"Error saving {model_name}: {str(e)}")
    
    def load_models(self, save_dir: str) -> None:
        """
        Load saved models
        
        Args:
            save_dir: Directory containing saved models
        """
        import os
        from tensorflow.keras.models import load_model
        
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'autoencoder': 'autoencoder_model.h5',
            'lstm': 'lstm_model.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(save_dir, filename)
            
            if os.path.exists(model_path):
                try:
                    if model_name in ['autoencoder', 'lstm']:
                        model = load_model(model_path)
                        info_path = os.path.join(save_dir, f"{model_name}_info.pkl")
                        model_info = joblib.load(info_path)
                        model_info['model'] = model
                    else:
                        model_info = joblib.load(model_path)
                    
                    self.models[model_name] = model_info
                    self.logger.info(f"Loaded {model_name} model from {save_dir}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading {model_name}: {str(e)}")

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from feature_engineering import FeatureEngineering
    
    ingestion = DataIngestion()
    sample_data = ingestion.create_sample_data(1000)
    
    feature_eng = FeatureEngineering()
    X, y = feature_eng.prepare_features_for_ml(sample_data)
    
    ml_models = MLModels()
    
    print("Training Random Forest...")
    rf_results = ml_models.train_random_forest(X, y)
    
    print("\\nTraining Autoencoder...")
    ae_results = ml_models.train_autoencoder(X)
    
    print("\\nTraining LSTM...")
    lstm_results = ml_models.train_lstm(X, y)
    
    print("\\nModel Performance Summary:")
    for model_name, performance in ml_models.model_performance.items():
        print(f"{model_name}: {performance}")