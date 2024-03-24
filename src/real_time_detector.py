"""
Real-time Anomaly Detection Module for Cybersecurity Intrusion Detection
Handles real-time monitoring, anomaly scoring, alert generation, and threat classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import threading
import time
import queue
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class Alert:
    """Class to represent security alerts"""
    
    def __init__(self, alert_id: str, timestamp: datetime, severity: str, 
                 alert_type: str, description: str, source_ip: str = None,
                 destination_ip: str = None, confidence: float = 0.0,
                 additional_info: Dict = None):
        self.alert_id = alert_id
        self.timestamp = timestamp
        self.severity = severity
        self.alert_type = alert_type
        self.description = description
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.confidence = confidence
        self.additional_info = additional_info or {}
        self.status = 'new'
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'alert_type': self.alert_type,
            'description': self.description,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'confidence': self.confidence,
            'additional_info': self.additional_info,
            'status': self.status
        }

class RealTimeAnomalyDetector:
    """
    Real-time anomaly detection system for network intrusion detection
    """
    
    def __init__(self, models: Dict, feature_engineer, config: Dict = None):
        """
        Initialize real-time anomaly detector
        
        Args:
            models: Dictionary of trained ML models
            feature_engineer: Feature engineering instance
            config: Configuration dictionary
        """
        self.models = models
        self.feature_engineer = feature_engineer
        self.config = config or {}
        self.logger = self._setup_logger()
        self.monitoring_start_time = None
        
        self.data_buffer = deque(maxlen=self.config.get('buffer_size', 10000))
        self.alerts = []
        self.alert_queue = queue.Queue()
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks = []
        
        self.thresholds = {
            'anomaly_score': self.config.get('anomaly_threshold', 0.7),
            'confidence': self.config.get('confidence_threshold', 0.5),
            'severity_thresholds': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9
            }
        }
        
        self.stats = {
            'total_packets': 0,
            'anomalies_detected': 0,
            'alerts_generated': 0,
            'false_positives': 0,
            'detection_rate': 0.0,
            'processing_time': deque(maxlen=1000)
        }
        
        self.time_windows = {
            'short': timedelta(minutes=5),
            'medium': timedelta(minutes=30),
            'long': timedelta(hours=2)
        }
        
        self.attack_patterns = {
            'port_scan': {'window': 60, 'threshold': 20},
            'dos_attack': {'window': 10, 'threshold': 1000},
            'brute_force': {'window': 300, 'threshold': 10},
            'data_exfiltration': {'window': 600, 'threshold': 100000}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for real-time detector"""
        logger = logging.getLogger('RealTimeDetector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_start_time = datetime.now()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                if len(self.data_buffer) > 0:
                    recent_data = list(self.data_buffer)[-100:]
                    df = pd.DataFrame(recent_data)
                    
                    if not df.empty:
                        anomalies = self.detect_anomalies(df)
                        
                        for anomaly in anomalies:
                            self._generate_alert(anomaly)
                
                time.sleep(self.config.get('monitoring_interval', 1))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def add_data(self, data: Dict) -> None:
        """
        Add new network data for real-time analysis
        
        Args:
            data: Dictionary with network traffic data
        """
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        self.data_buffer.append(data)
        self.stats['total_packets'] += 1
        self.logger.info(f"Data added to buffer: {data}, Total packets: {self.stats['total_packets']}")
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies in real-time data
        
        Args:
            df: DataFrame with network traffic data
            
        Returns:
            List of detected anomalies
        """
        try:
            start_time = time.time()
            anomalies = []
            
            if df.empty:
                return anomalies
            
            X, _ = self.feature_engineer.prepare_features_for_ml(df, fit=False)
            
            if X.empty:
                return anomalies
            
            if hasattr(self.models, 'ensemble_prediction'):
                predictions, model_scores = self.models.ensemble_prediction(X)
            else:
                predictions, model_scores = self._get_model_predictions(X)

            if model_scores:
                if isinstance(model_scores, dict) and len(model_scores) > 0:
                    try:
                        self.last_anomaly_scores = np.mean(list(model_scores.values()), axis=0)
                    except Exception:
                        self.last_anomaly_scores = list(model_scores.values())[0]
                else:
                    self.last_anomaly_scores = [0]*len(X)
            else:
                self.last_anomaly_scores = [0]*len(X)
            
            for i, (idx, row) in enumerate(df.iterrows()):
                if i < len(predictions) and predictions[i] == 1:
                    anomaly_info = {
                        'index': idx,
                        'timestamp': row.get('timestamp', time.time()),
                        'data': row.to_dict(),
                        'anomaly_score': np.mean(list(model_scores.values()))[i] if model_scores else 0.5,
                        'model_scores': {k: v[i] for k, v in model_scores.items() if i < len(v)},
                        'severity': self._calculate_severity(model_scores, i),
                        'attack_type': self._classify_attack_type(row)
                    }
                    anomalies.append(anomaly_info)
            
            processing_time = time.time() - start_time
            self.stats['processing_time'].append(processing_time)
            self.stats['anomalies_detected'] += len(anomalies)
            
            if len(anomalies) > 0:
                self.logger.info(f"Detected {len(anomalies)} anomalies in {processing_time:.3f} seconds")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def _get_model_predictions(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Get predictions from individual models
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (ensemble predictions, model scores)
        """
        predictions = {}
        scores = {}
        
        if 'random_forest' in self.models:
            try:
                rf_model = self.models['random_forest']['model']
                rf_pred = rf_model.predict_proba(X)[:, 1]
                predictions['random_forest'] = (rf_pred > 0.5).astype(int)
                scores['random_forest'] = rf_pred
            except Exception as e:
                self.logger.warning(f"Error with Random Forest prediction: {str(e)}")
        
        if 'autoencoder' in self.models:
            try:
                ae_model = self.models['autoencoder']['model']
                ae_threshold = self.models['autoencoder']['threshold']
                
                ae_pred = ae_model.predict(X)
                reconstruction_errors = np.mean(np.square(X.values - ae_pred), axis=1)
                ae_anomalies = (reconstruction_errors > ae_threshold).astype(int)
                ae_scores = reconstruction_errors / np.max(reconstruction_errors)
                
                predictions['autoencoder'] = ae_anomalies
                scores['autoencoder'] = ae_scores
            except Exception as e:
                self.logger.warning(f"Error with Autoencoder prediction: {str(e)}")
        
        if predictions:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            ensemble_pred = (ensemble_pred > 0.5).astype(int)
            return ensemble_pred, scores
        else:
            return np.zeros(len(X)), {}
    
    def _calculate_severity(self, model_scores: Dict, index: int) -> str:
        """
        Calculate severity level based on model scores
        
        Args:
            model_scores: Dictionary of model scores
            index: Index of the data point
            
        Returns:
            Severity level string
        """
        if not model_scores:
            return 'low'
        
        scores = []
        for model_name, score_array in model_scores.items():
            if index < len(score_array):
                scores.append(score_array[index])
        
        if not scores:
            return 'low'
        
        avg_score = np.mean(scores)
        
        thresholds = self.thresholds['severity_thresholds']
        if avg_score >= thresholds['critical']:
            return 'critical'
        elif avg_score >= thresholds['high']:
            return 'high'
        elif avg_score >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _classify_attack_type(self, data_row: pd.Series) -> str:
        """
        Classify the type of attack based on network features
        
        Args:
            data_row: Series with network data
            
        Returns:
            Attack type classification
        """
        try:
            if self._detect_port_scan(data_row):
                return 'port_scan'
            
            if self._detect_dos_attack(data_row):
                return 'dos_attack'
            
            if self._detect_brute_force(data_row):
                return 'brute_force'
            
            if self._detect_data_exfiltration(data_row):
                return 'data_exfiltration'
            
            return 'unknown_anomaly'
            
        except Exception as e:
            self.logger.warning(f"Error classifying attack type: {str(e)}")
            return 'unknown'
    
    def _detect_port_scan(self, data_row: pd.Series) -> bool:
        """Detect port scanning patterns"""
        if 'dst_port' in data_row and 'count' in data_row:
            return data_row.get('count', 0) > 20 and data_row.get('srv_count', 0) < 5
        return False
    
    def _detect_dos_attack(self, data_row: pd.Series) -> bool:
        """Detect DoS attack patterns"""
        return data_row.get('count', 0) > 100 and data_row.get('duration', 0) < 1
    
    def _detect_brute_force(self, data_row: pd.Series) -> bool:
        """Detect brute force attack patterns"""
        return data_row.get('num_failed_logins', 0) > 5
    
    def _detect_data_exfiltration(self, data_row: pd.Series) -> bool:
        """Detect data exfiltration patterns"""
        total_bytes = data_row.get('src_ip', 0) + data_row.get('dst_bytes', 0)
        return total_bytes > 10000 and data_row.get('dst_bytes', 0) > data_row.get('src_bytes', 0)
    
    def _generate_alert(self, anomaly: Dict) -> None:
        """
        Generate security alert for detected anomaly
        
        Args:
            anomaly: Dictionary with anomaly information
        """
        try:
            alert_id = f"ALT_{int(time.time())}_{len(self.alerts)}"
            timestamp = datetime.fromtimestamp(anomaly['timestamp'])
            
            data = anomaly['data']
            source_ip = data.get('src_ip', 'unknown')
            dest_ip = data.get('dst_ip', 'unknown')
            
            attack_type = anomaly.get('attack_type', 'unknown')
            confidence = anomaly.get('anomaly_score', 0.0)
            description = f"{attack_type.replace('_', ' ').title()} detected from {source_ip} to {dest_ip}"
            
            alert = Alert(
                alert_id=alert_id,
                timestamp=timestamp,
                severity=anomaly['severity'],
                alert_type=attack_type,
                description=description,
                source_ip=source_ip,
                destination_ip=dest_ip,
                confidence=confidence,
                additional_info={
                    'model_scores': anomaly.get('model_scores', {}),
                    'raw_data': data
                }
            )
            
            self.alerts.append(alert)
            self.alert_queue.put(alert)
            self.stats['alerts_generated'] += 1
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
            
            self.logger.warning(f"ALERT: {alert.description} (Severity: {alert.severity})")
            
        except Exception as e:
            self.logger.error(f"Error generating alert: {str(e)}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add callback function to be called when alert is generated
        
        Args:
            callback: Function to call with alert object
        """
        self.alert_callbacks.append(callback)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """
        Get alerts from the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
        return recent_alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive alert statistics
        
        Returns:
            Dictionary with alert statistics
        """
        recent_alerts = self.get_recent_alerts(24)
        
        severity_counts = {}
        for severity in ['low', 'medium', 'high', 'critical']:
            severity_counts[severity] = sum(1 for alert in recent_alerts if alert.severity == severity)
        
        attack_type_counts = {}
        for alert in recent_alerts:
            attack_type = alert.alert_type
            attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1
        
        return {
            'total_alerts_24h': len(recent_alerts),
            'severity_distribution': severity_counts,
            'attack_type_distribution': attack_type_counts,
            'detection_rate': self.stats['detection_rate'],
            'average_processing_time': np.mean(self.stats['processing_time']) if self.stats['processing_time'] else 0,
            'total_packets_processed': self.stats['total_packets'],
            'total_anomalies_detected': self.stats['anomalies_detected']
        }
    
    def update_thresholds(self, new_thresholds: Dict) -> None:
        """
        Update anomaly detection thresholds
        
        Args:
            new_thresholds: Dictionary with new threshold values
        """
        self.thresholds.update(new_thresholds)
        self.logger.info(f"Updated thresholds: {self.thresholds}")
    
    def simulate_real_time_data(self, duration_minutes: int = 5) -> None:
        """
        Simulate real-time network data for testing
        
        Args:
            duration_minutes: Duration to simulate data
        """
        self.logger.info(f"Starting {duration_minutes}-minute real-time simulation")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        normal_patterns = [
            {'protocol_type': 'tcp', 'src_port': 80, 'dst_port': 443, 'packet_size': 1500},
            {'protocol_type': 'udp', 'src_port': 53, 'dst_port': 12345, 'packet_size': 512},
            {'protocol_type': 'tcp', 'src_port': 22, 'dst_port': 54321, 'packet_size': 800}
        ]
        
        anomaly_patterns = [
            {'protocol_type': 'tcp', 'count': 1000, 'duration': 0.1, 'attack_type': 'dos'},
            {'protocol_type': 'tcp', 'dst_port': 22, 'num_failed_logins': 10, 'attack_type': 'brute_force'},
            {'protocol_type': 'tcp', 'src_bytes': 50000, 'dst_bytes': 5000, 'attack_type': 'exfiltration'}
        ]
        
        while time.time() < end_time:
            if np.random.random() < 0.9:
                pattern = np.random.choice(normal_patterns)
                data = {
                    'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
                    'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
                    'timestamp': time.time(),
                    **pattern,
                    'src_bytes': np.random.randint(100, 2000),
                    'dst_bytes': np.random.randint(100, 2000),
                    'duration': np.random.exponential(1),
                    'count': np.random.randint(1, 10)
                }
            else:
                pattern = np.random.choice(anomaly_patterns)
                data = {
                    'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
                    'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
                    'timestamp': time.time(),
                    **pattern,
                    'label': 1
                }
                for field in ['src_bytes', 'dst_bytes', 'duration', 'count', 'packet_size']:
                    if field not in data:
                        data[field] = np.random.randint(1, 1000)
        
        while time.time() < end_time:
            if np.random.random() < 0.9:
                pattern = np.random.choice(normal_patterns)
                data = {
                    'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
                    'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
                    'timestamp': time.time(),
                    **pattern,
                    'src_bytes': np.random.randint(100, 2000),
                    'dst_bytes': np.random.randint(100, 2000),
                    'duration': np.random.exponential(1),
                    'count': np.random.randint(1, 10)
                }
            else:
                pattern = np.random.choice(anomaly_patterns)
                data = {
                    'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
                    'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
                    'timestamp': time.time(),
                    **pattern,
                    'label': 1  # Mark as anomaly for testing
                }
                for field in ['src_bytes', 'dst_bytes', 'duration', 'count', 'packet_size']:
                    if field not in data:
                        data[field] = np.random.randint(1, 1000)
            
            self.add_data(data)
            time.sleep(0.1)
        
        self.logger.info("Real-time simulation completed")

if __name__ == "__main__":
    class MockMLModels:
        def ensemble_prediction(self, X):
            predictions = np.random.choice([0, 1], size=len(X), p=[0.9, 0.1])
            scores = {'mock_model': np.random.random(len(X))}
            return predictions, scores
    
    class MockFeatureEngineering:
        def prepare_features_for_ml(self, df, fit=False):
            return df.select_dtypes(include=[np.number]), None
    
    mock_models = MockMLModels()
    mock_feature_eng = MockFeatureEngineering()
    
    detector = RealTimeAnomalyDetector(
        models=mock_models,
        feature_engineer=mock_feature_eng,
        config={'monitoring_interval': 2}
    )
    
    def print_alert(alert: Alert):
        print(f"ALERT: {alert.description} [{alert.severity}]")
    
    detector.add_alert_callback(print_alert)
    
    detector.start_monitoring()
    
    print("Simulating real-time data...")
    detector.simulate_real_time_data(duration_minutes=1)
    
    detector.stop_monitoring()
    
    stats = detector.get_alert_statistics()
    print("\\nAlert Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")