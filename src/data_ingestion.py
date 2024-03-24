"""
Data Ingestion Module for Cybersecurity Intrusion Detection
Handles loading and preprocessing of network traffic data from various sources
including CICIDS2017, NSL-KDD datasets, and real-time packet capture
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from scapy.all import rdpcap, Packet
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataIngestion:
    """
    Handles data loading and preprocessing for network intrusion detection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize data ingestion module
        
        Args:
            config: Configuration dictionary with data paths and settings
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        self.expected_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        
        self.attack_labels = {
            'normal': 0,
            'benign': 0,
            'anomaly': 1,
            'malicious': 1,
            'dos': 1, 'ddos': 1, 'syn_flood': 1, 'smurf': 1,
            'probe': 1, 'portsweep': 1, 'ipsweep': 1, 'nmap': 1,
            'u2r': 1, 'buffer_overflow': 1, 'rootkit': 1,
            'r2l': 1, 'ftp_write': 1, 'guess_passwd': 1, 'imap': 1
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for data ingestion"""
        logger = logging.getLogger('DataIngestion')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_csv_data(self, file_path: str, has_header: bool = True) -> pd.DataFrame:
        """
        Load network traffic data from CSV file
        
        Args:
            file_path: Path to CSV file
            has_header: Whether CSV has column headers
            
        Returns:
            DataFrame with network traffic data
        """
        try:
            self.logger.info(f"Loading CSV data from {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if has_header:
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(file_path, header=None)
                if len(df.columns) == len(self.expected_features) + 1:
                    df.columns = self.expected_features + ['label']
                else:
                    df.columns = [f'feature_{i}' for i in range(len(df.columns)-1)] + ['label']
            
            self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def load_pcap_data(self, file_path: str, max_packets: int = 10000) -> pd.DataFrame:
        """
        Load network traffic data from PCAP file
        
        Args:
            file_path: Path to PCAP file
            max_packets: Maximum number of packets to process
            
        Returns:
            DataFrame with extracted network features
        """
        try:
            self.logger.info(f"Loading PCAP data from {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            packets = rdpcap(file_path)
            
            if len(packets) > max_packets:
                packets = packets[:max_packets]
                self.logger.warning(f"Limited to {max_packets} packets for processing")
            
            features = []
            for packet in packets:
                feature_dict = self._extract_packet_features(packet)
                features.append(feature_dict)
            
            df = pd.DataFrame(features)
            self.logger.info(f"Extracted features from {len(df)} packets")
            
            df['label'] = 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading PCAP data: {str(e)}")
            raise
    
    def _extract_packet_features(self, packet: Packet) -> Dict:
        """
        Extract network features from a single packet
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            features['packet_size'] = len(packet)
            features['timestamp'] = float(packet.time) if hasattr(packet, 'time') else 0
            
            if packet.haslayer('IP'):
                ip_layer = packet['IP']
                features['src_ip'] = str(ip_layer.src)
                features['dst_ip'] = str(ip_layer.dst)
                features['ip_len'] = ip_layer.len
                features['ip_flags'] = ip_layer.flags
                features['ip_ttl'] = ip_layer.ttl
                features['protocol'] = ip_layer.proto
            else:
                features.update({
                    'src_ip': '0.0.0.0', 'dst_ip': '0.0.0.0',
                    'ip_len': 0, 'ip_flags': 0, 'ip_ttl': 0, 'protocol': 0
                })
            
            if packet.haslayer('TCP'):
                tcp_layer = packet['TCP']
                features['src_port'] = tcp_layer.sport
                features['dst_port'] = tcp_layer.dport
                features['tcp_flags'] = tcp_layer.flags
                features['tcp_window'] = tcp_layer.window
                features['tcp_seq'] = tcp_layer.seq
                features['tcp_ack'] = tcp_layer.ack
            else:
                features.update({
                    'src_port': 0, 'dst_port': 0, 'tcp_flags': 0,
                    'tcp_window': 0, 'tcp_seq': 0, 'tcp_ack': 0
                })
            
            if packet.haslayer('UDP'):
                udp_layer = packet['UDP']
                features['src_port'] = udp_layer.sport
                features['dst_port'] = udp_layer.dport
                features['udp_len'] = udp_layer.len
            elif 'udp_len' not in features:
                features['udp_len'] = 0
            
            protocol_map = {1: 'icmp', 6: 'tcp', 17: 'udp'}
            features['protocol_type'] = protocol_map.get(features.get('protocol', 0), 'other')
            
        except Exception as e:
            self.logger.warning(f"Error extracting packet features: {str(e)}")
            features = {
                'packet_size': 0, 'timestamp': 0, 'src_ip': '0.0.0.0',
                'dst_ip': '0.0.0.0', 'src_port': 0, 'dst_port': 0,
                'protocol': 0, 'protocol_type': 'other'
            }
        
        return features
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data for machine learning
        
        Args:
            df: Raw DataFrame with network traffic data
            
        Returns:
            Preprocessed DataFrame ready for ML models
        """
        try:
            self.logger.info("Preprocessing network traffic data")
            df_processed = df.copy()
            
            df_processed = df_processed.fillna(0)
            
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in ['label']:  # Don't encode the target variable yet
                    df_processed[col] = pd.Categorical(df_processed[col]).codes
            
            if 'label' in df_processed.columns:
                df_processed['label'] = self._standardize_labels(df_processed['label'])
            
            df_processed = df_processed.replace([np.inf, -np.inf], 0)
            
            ip_columns = ['src_ip', 'dst_ip']
            for col in ip_columns:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].apply(self._ip_to_int)
            
            self.logger.info(f"Preprocessing completed. Shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def _standardize_labels(self, labels: pd.Series) -> pd.Series:
        """
        Standardize labels to binary classification (0: normal, 1: attack)
        
        Args:
            labels: Series with original labels
            
        Returns:
            Series with standardized binary labels
        """
        standardized = labels.copy()
        
        standardized = standardized.astype(str).str.lower().str.strip()
        
        binary_labels = []
        for label in standardized:
            binary_labels.append(self.attack_labels.get(label, 1))  # Default to attack if unknown
        
        return pd.Series(binary_labels, index=labels.index)
    
    def _ip_to_int(self, ip_str: str) -> int:
        """
        Convert IP address string to integer
        
        Args:
            ip_str: IP address string
            
        Returns:
            Integer representation of IP address
        """
        try:
            if pd.isna(ip_str) or ip_str == '0.0.0.0':
                return 0
            
            parts = str(ip_str).split('.')
            if len(parts) != 4:
                return 0
            
            return sum(int(part) << (8 * (3 - i)) for i, part in enumerate(parts))
        except:
            return 0
    
    def create_sample_data(self, n_samples: int = 1000, anomaly_rate: int = 0.2) -> pd.DataFrame:
        """
        Create sample network traffic data for testing
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic network traffic data
        """
        np.random.seed(42)
        
        data = {
            'duration': np.random.exponential(1, n_samples),
            'src_bytes': np.random.exponential(100, n_samples),
            'dst_bytes': np.random.exponential(100, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'src_port': np.random.randint(1, 65536, n_samples),
            'dst_port': np.random.randint(1, 65536, n_samples),
            'packet_size': np.random.randint(64, 1500, n_samples),
            'tcp_flags': np.random.randint(0, 256, n_samples),
            'count': np.random.randint(1, 100, n_samples),
            'srv_count': np.random.randint(1, 100, n_samples)
        }
        
        n_anomalies = int(n_samples * anomaly_rate)
        anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            data['src_bytes'][idx] *= 10  # Unusually large transfers
            data['count'][idx] *= 5       # High connection counts
            data['dst_bytes'][idx] *= 3   # Large destination bytes
        
        labels = np.zeros(n_samples)
        labels[anomaly_indices] = 1
        data['label'] = labels
        
        df = pd.DataFrame(data)
        
        unique_labels = df['label'].unique()
        if len(unique_labels) < 2:
            n_force_anomalies = max(1, n_samples // 10)  # At least 10% anomalies
            force_indices = np.random.choice(n_samples, size=n_force_anomalies, replace=False)
            df.loc[force_indices, 'label'] = 1
        
        self.logger.info(f"Created {n_samples} sample records with {np.sum(df['label'])} anomalies")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the loaded data and return statistics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            validation_results['label_distribution'] = label_counts.to_dict()
            validation_results['anomaly_rate'] = label_counts.get(1, 0) / len(df)
        
        validation_results['data_quality'] = {
            'has_negative_values': (df.select_dtypes(include=[np.number]) < 0).any().any(),
            'has_infinite_values': df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any(),
            'has_null_values': df.isnull().any().any()
        }
        
        self.logger.info(f"Data validation completed: {validation_results['total_records']} records")
        return validation_results

if __name__ == "__main__":
    ingestion = DataIngestion()
    
    sample_data = ingestion.create_sample_data(1000)
    print("Sample data created:")
    print(sample_data.head())
    
    processed_data = ingestion.preprocess_data(sample_data)
    print("\\nProcessed data:")
    print(processed_data.head())
    
    validation_results = ingestion.validate_data(processed_data)
    print("\\nValidation results:")
    print(json.dumps(validation_results, indent=2))