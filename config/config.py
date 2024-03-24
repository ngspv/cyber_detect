DATA_CONFIG = {
    'max_file_size_mb': 500,
    'supported_formats': ['csv', 'pcap', 'pcapng'],
    'max_packets_pcap': 50000,
    'buffer_size': 10000
}

FEATURE_CONFIG = {
    'statistical_window_size': 100,
    'max_interactions': 20,
    'feature_selection_k': 50,
    'normalization_method': 'standard'
}

MODEL_CONFIG = {
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
        'learning_rate': 0.001,
        'validation_split': 0.1
    },
    'lstm': {
        'lstm_units': 50,
        'epochs': 50,
        'batch_size': 32,
        'sequence_length': 10,
        'learning_rate': 0.001
    }
}

DETECTION_CONFIG = {
    'monitoring_interval': 1,  # seconds
    'anomaly_threshold': 0.7,
    'confidence_threshold': 0.5,
    'buffer_size': 10000,
    'severity_thresholds': {
        'low': 0.3,
        'medium': 0.5,
        'high': 0.7,
        'critical': 0.9
    }
}

ALERT_CONFIG = {
    'max_alerts_stored': 10000,
    'alert_retention_days': 30,
    'email_notifications': False,
    'slack_notifications': False,
    'webhook_url': None
}

UI_CONFIG = {
    'page_title': 'Cybersecurity Intrusion Detection',
    'page_icon': '🛡️',
    'layout': 'wide',
    'theme': 'light',
    'refresh_interval': 5  # seconds
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'max_log_size_mb': 100,
    'backup_count': 5
}

ATTACK_PATTERNS = {
    'port_scan': {
        'window_seconds': 60,
        'threshold_ports': 20,
        'min_packets': 50
    },
    'dos_attack': {
        'window_seconds': 10,
        'threshold_packets': 1000,
        'min_packet_rate': 100
    },
    'brute_force': {
        'window_seconds': 300,
        'threshold_attempts': 10,
        'target_ports': [22, 23, 21, 3389]
    },
    'data_exfiltration': {
        'window_seconds': 600,
        'threshold_bytes': 100000000,  # 100MB
        'suspicious_ratio': 0.1
    }
}

MODEL_PATHS = {
    'model_directory': './models',
    'random_forest': 'random_forest_model.pkl',
    'autoencoder': 'autoencoder_model.h5',
    'lstm': 'lstm_model.h5',
    'feature_scaler': 'feature_scaler.pkl',
    'label_encoders': 'label_encoders.pkl'
}

DATABASE_CONFIG = {
    'type': 'sqlite',
    'host': 'localhost',
    'port': 5432,
    'database': 'cyber_detect.db',
    'username': None,
    'password': None
}

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'cors_enabled': True
}