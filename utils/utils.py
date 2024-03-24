"""
Utility functions for the Cybersecurity Intrusion Detection System
Common helper functions used across different modules
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import hashlib
import yaml

class Logger:
    """Enhanced logging utility"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: str = 'INFO') -> logging.Logger:
        """
        Set up logger with file and console handlers
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

class FileManager:
    """File management utilities"""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Create directory if it doesn't exist"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict, file_path: str) -> None:
        """Save data to JSON file"""
        FileManager.ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(file_path: str) -> Dict:
        """Load data from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(obj: Any, file_path: str) -> None:
        """Save object to pickle file"""
        FileManager.ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """Load object from pickle file"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_yaml(data: Dict, file_path: str) -> None:
        """Save data to YAML file"""
        FileManager.ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict:
        """Load data from YAML file"""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Get MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)

class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def convert_bytes_to_mb(bytes_value: int) -> float:
        """Convert bytes to megabytes"""
        return bytes_value / (1024 * 1024)
    
    @staticmethod
    def convert_mb_to_bytes(mb_value: float) -> int:
        """Convert megabytes to bytes"""
        return int(mb_value * 1024 * 1024)
    
    @staticmethod
    def format_number(number: Union[int, float], precision: int = 2) -> str:
        """Format number with thousands separators"""
        if isinstance(number, float):
            return f"{number:,.{precision}f}"
        return f"{number:,}"
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division that handles division by zero"""
        return numerator / denominator if denominator != 0 else default
    
    @staticmethod
    def normalize_ip_address(ip: str) -> str:
        """Normalize IP address format"""
        try:
            parts = ip.split('.')
            if len(parts) == 4:
                return '.'.join([str(int(part)) for part in parts])
        except:
            pass
        return ip
    
    @staticmethod
    def is_private_ip(ip: str) -> bool:
        """Check if IP address is private"""
        try:
            parts = [int(x) for x in ip.split('.')]
            if len(parts) != 4:
                return False
            
            if parts[0] == 10:
                return True
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return True
            if parts[0] == 192 and parts[1] == 168:
                return True
            
            return False
        except:
            return False
    
    @staticmethod
    def calculate_entropy(data: List[Any]) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        counts = {}
        for item in data:
            counts[item] = counts.get(item, 0) + 1
        
        entropy = 0.0
        length = len(data)
        for count in counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy

class TimeUtils:
    """Time and date utilities"""
    
    @staticmethod
    def get_current_timestamp() -> float:
        """Get current Unix timestamp"""
        return datetime.now().timestamp()
    
    @staticmethod
    def timestamp_to_datetime(timestamp: float) -> datetime:
        """Convert Unix timestamp to datetime"""
        return datetime.fromtimestamp(timestamp)
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> float:
        """Convert datetime to Unix timestamp"""
        return dt.timestamp()
    
    @staticmethod
    def format_timestamp(timestamp: float, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Format timestamp as string"""
        return datetime.fromtimestamp(timestamp).strftime(format_str)
    
    @staticmethod
    def get_time_ago(timestamp: float) -> str:
        """Get human-readable time ago string"""
        now = datetime.now()
        past = datetime.fromtimestamp(timestamp)
        diff = now - past
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return f"{diff.seconds} second{'s' if diff.seconds != 1 else ''} ago"
    
    @staticmethod
    def get_time_range(hours_back: int = 24) -> tuple:
        """Get time range from hours back to now"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        return start_time, end_time

class SecurityUtils:
    """Security-related utilities"""
    
    @staticmethod
    def classify_severity(score: float, thresholds: Dict[str, float] = None) -> str:
        """
        Classify severity based on score
        
        Args:
            score: Anomaly score (0-1)
            thresholds: Custom severity thresholds
            
        Returns:
            Severity level string
        """
        if thresholds is None:
            thresholds = {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9
            }
        
        if score >= thresholds['critical']:
            return 'critical'
        elif score >= thresholds['high']:
            return 'high'
        elif score >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    @staticmethod
    def get_severity_color(severity: str) -> str:
        """Get color code for severity level"""
        colors = {
            'low': '#4CAF50',      # Green
            'medium': '#9C27B0',   # Purple
            'high': '#FF9800',     # Orange
            'critical': '#F44336'  # Red
        }
        return colors.get(severity, '#9E9E9E')  # Gray default
    
    @staticmethod
    def is_suspicious_port(port: int) -> bool:
        """Check if port is commonly used in attacks"""
        suspicious_ports = {
            22, 23, 135, 139, 445, 1433, 1521, 3389, 5432, 5900
        }
        return port in suspicious_ports
    
    @staticmethod
    def classify_protocol_risk(protocol: str) -> str:
        """Classify protocol risk level"""
        high_risk = ['telnet', 'ftp', 'http', 'snmp']
        medium_risk = ['ssh', 'smtp', 'pop3', 'imap']
        
        protocol_lower = protocol.lower()
        
        if protocol_lower in high_risk:
            return 'high'
        elif protocol_lower in medium_risk:
            return 'medium'
        else:
            return 'low'

class ModelUtils:
    """Machine learning model utilities"""
    
    @staticmethod
    def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive model metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None and len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    def ensemble_predictions(predictions: Dict[str, np.ndarray], 
                           weights: Dict[str, float] = None) -> np.ndarray:
        """Combine predictions from multiple models"""
        if not predictions:
            return np.array([])
        
        if weights is None:
            weights = {model: 1.0 for model in predictions.keys()}
        
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        ensemble = np.zeros(len(list(predictions.values())[0]))
        for model, pred in predictions.items():
            ensemble += weights.get(model, 0) * pred
        
        return ensemble

class ConfigManager:
    """Configuration management utilities"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_path.endswith('.json'):
            self.config = FileManager.load_json(self.config_path)
        elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
            self.config = FileManager.load_yaml(self.config_path)
        return self.config
    
    def save_config(self) -> None:
        """Save configuration to file"""
        if not self.config_path:
            raise ValueError("No config path specified")
        
        if self.config_path.endswith('.json'):
            FileManager.save_json(self.config, self.config_path)
        elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
            FileManager.save_yaml(self.config, self.config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict) -> None:
        """Update configuration with new values"""
        def deep_update(original, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    deep_update(original[key], value)
                else:
                    original[key] = value
        
        deep_update(self.config, updates)

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str) -> None:
        """Start timing operation"""
        self.metrics[name] = {'start_time': datetime.now()}
    
    def end_timer(self, name: str) -> float:
        """End timing operation and return duration"""
        if name in self.metrics and 'start_time' in self.metrics[name]:
            duration = (datetime.now() - self.metrics[name]['start_time']).total_seconds()
            self.metrics[name]['duration'] = duration
            return duration
        return 0.0
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add custom metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            'value': value,
            'timestamp': datetime.now()
        })
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        summary = {}
        for name, data in self.metrics.items():
            if isinstance(data, dict) and 'duration' in data:
                summary[name] = data['duration']
            elif isinstance(data, list):
                values = [item['value'] for item in data if isinstance(item, dict)]
                if values:
                    summary[name] = {
                        'count': len(values),
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        return summary

if __name__ == "__main__":
    logger = Logger.setup_logger('test', 'logs/test.log')
    logger.info("Testing utilities")
    
    test_data = {'test': 'data', 'number': 123}
    FileManager.save_json(test_data, 'test_output.json')
    loaded_data = FileManager.load_json('test_output.json')
    print(f"JSON test: {loaded_data}")
    
    print(f"File size: {DataUtils.format_number(1234567)} bytes")
    print(f"Is private IP: {DataUtils.is_private_ip('192.168.1.1')}")
    
    now = TimeUtils.get_current_timestamp()
    print(f"Current time: {TimeUtils.format_timestamp(now)}")
    
    print(f"Severity: {SecurityUtils.classify_severity(0.8)}")
    print(f"Suspicious port: {SecurityUtils.is_suspicious_port(22)}")
    
    if os.path.exists('test_output.json'):
        os.remove('test_output.json')