"""
Feature Engineering Module for Cybersecurity Intrusion Detection
Handles feature extraction, transformation, and preparation for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Handles feature engineering for network intrusion detection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature engineering module
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'numerical': StandardScaler()
        }
        
        self.label_encoders = {}
        
        self.feature_selectors = {}
        
        self.statistical_features = [
            'mean', 'std', 'min', 'max', 'median', 'skew', 'kurt'
        ]
        
        self.feature_categories = {
            'basic': ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes'],
            'content': ['hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell'],
            'time': ['count', 'srv_count', 'serror_rate', 'srv_serror_rate'],
            'host': ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate']
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for feature engineering"""
        logger = logging.getLogger('FeatureEngineering')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive network features from raw data
        
        Args:
            df: DataFrame with raw network data
            
        Returns:
            DataFrame with extracted features
        """
        try:
            self.logger.info("Extracting network features")
            feature_df = df.copy()
            
            if 'packet_size' in feature_df.columns:
                feature_df['packet_size_log'] = np.log1p(feature_df['packet_size'])
                feature_df['packet_size_squared'] = feature_df['packet_size'] ** 2
            
            if 'src_bytes' in feature_df.columns and 'dst_bytes' in feature_df.columns:
                feature_df['total_bytes'] = feature_df['src_bytes'] + feature_df['dst_bytes']
                feature_df['byte_ratio'] = np.where(
                    feature_df['dst_bytes'] > 0,
                    feature_df['src_bytes'] / feature_df['dst_bytes'],
                    feature_df['src_bytes']
                )
                feature_df['bytes_per_second'] = np.where(
                    feature_df['duration'] > 0,
                    feature_df['total_bytes'] / feature_df['duration'],
                    feature_df['total_bytes']
                )
            
            if 'count' in feature_df.columns and 'srv_count' in feature_df.columns:
                feature_df['srv_ratio'] = np.where(
                    feature_df['count'] > 0,
                    feature_df['srv_count'] / feature_df['count'],
                    0
                )
            
            if 'src_port' in feature_df.columns and 'dst_port' in feature_df.columns:
                feature_df['is_well_known_src_port'] = (feature_df['src_port'] <= 1024).astype(int)
                feature_df['is_well_known_dst_port'] = (feature_df['dst_port'] <= 1024).astype(int)
                feature_df['port_diff'] = abs(feature_df['src_port'] - feature_df['dst_port'])
            
            if 'protocol_type' in feature_df.columns:
                feature_df['is_tcp'] = (feature_df['protocol_type'] == 'tcp').astype(int)
                feature_df['is_udp'] = (feature_df['protocol_type'] == 'udp').astype(int)
                feature_df['is_icmp'] = (feature_df['protocol_type'] == 'icmp').astype(int)
            
            if 'timestamp' in feature_df.columns:
                feature_df['hour'] = pd.to_datetime(feature_df['timestamp'], unit='s').dt.hour
                feature_df['day_of_week'] = pd.to_datetime(feature_df['timestamp'], unit='s').dt.dayofweek
                feature_df['is_weekend'] = (feature_df['day_of_week'] >= 5).astype(int)
            
            if 'tcp_flags' in feature_df.columns:
                feature_df['has_syn'] = ((feature_df['tcp_flags'] & 2) > 0).astype(int)
                feature_df['has_ack'] = ((feature_df['tcp_flags'] & 16) > 0).astype(int)
                feature_df['has_fin'] = ((feature_df['tcp_flags'] & 1) > 0).astype(int)
                feature_df['has_rst'] = ((feature_df['tcp_flags'] & 4) > 0).astype(int)
            
            self.logger.info(f"Feature extraction completed. New shape: {feature_df.shape}")
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {str(e)}")
            raise
    
    def create_statistical_features(self, df: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
        """
        Create statistical features using rolling windows
        
        Args:
            df: DataFrame with network data
            window_size: Size of rolling window for statistics
            
        Returns:
            DataFrame with statistical features
        """
        try:
            self.logger.info(f"Creating statistical features with window size {window_size}")
            feature_df = df.copy()
            
            numerical_cols = feature_df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != 'label']
            
            for col in numerical_cols[:10]:
                try:
                    feature_df[f'{col}_rolling_mean'] = feature_df[col].rolling(
                        window=min(window_size, len(feature_df)), min_periods=1
                    ).mean()
                    
                    feature_df[f'{col}_rolling_std'] = feature_df[col].rolling(
                        window=min(window_size, len(feature_df)), min_periods=1
                    ).std().fillna(0)
                    
                    feature_df[f'{col}_deviation'] = abs(
                        feature_df[col] - feature_df[f'{col}_rolling_mean']
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Error creating rolling features for {col}: {str(e)}")
                    continue
            
            if 'src_ip' in feature_df.columns:
                src_ip_counts = feature_df['src_ip'].value_counts()
                feature_df['src_ip_frequency'] = feature_df['src_ip'].map(src_ip_counts)
                
                dst_ip_counts = feature_df['dst_ip'].value_counts()
                feature_df['dst_ip_frequency'] = feature_df['dst_ip'].map(dst_ip_counts)
            
            self.logger.info(f"Statistical features created. New shape: {feature_df.shape}")
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating statistical features: {str(e)}")
            return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features for machine learning
        
        Args:
            df: DataFrame with categorical features
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        try:
            self.logger.info("Encoding categorical features")
            encoded_df = df.copy()
            
            categorical_cols = encoded_df.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if col != 'label']
            
            for col in categorical_cols:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    unique_values = encoded_df[col].unique()
                    try:
                        self.label_encoders[col].fit(unique_values)
                        encoded_df[col] = self.label_encoders[col].transform(encoded_df[col])
                    except Exception:
                        encoded_df[col] = pd.Categorical(encoded_df[col]).codes
                else:
                    if col in self.label_encoders:
                        seen_values = set(self.label_encoders[col].classes_)
                        encoded_df[col] = encoded_df[col].apply(
                            lambda x: x if x in seen_values else 'unknown'
                        )
                        
                        if 'unknown' not in seen_values:
                            current_classes = list(self.label_encoders[col].classes_)
                            current_classes.append('unknown')
                            self.label_encoders[col].classes_ = np.array(current_classes)
                        
                        encoded_df[col] = self.label_encoders[col].transform(encoded_df[col])
                    else:
                        encoded_df[col] = pd.Categorical(encoded_df[col]).codes
            
            self.logger.info(f"Categorical encoding completed for {len(categorical_cols)} features")
            return encoded_df
            
        except Exception as e:
            self.logger.error(f"Error in categorical encoding: {str(e)}")
            raise
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard', fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df: DataFrame with features to normalize
            method: Normalization method ('standard', 'minmax')
            fit: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            DataFrame with normalized features
        """
        try:
            self.logger.info(f"Normalizing features using {method} method")
            normalized_df = df.copy()
            
            numerical_cols = normalized_df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != 'label']
            
            if len(numerical_cols) == 0:
                self.logger.warning("No numerical columns found for normalization")
                return normalized_df
            
            scaler = self.scalers.get(method, StandardScaler())
            
            if fit:
                normalized_data = scaler.fit_transform(normalized_df[numerical_cols])
                self.scalers[method] = scaler
            else:
                normalized_data = scaler.transform(normalized_df[numerical_cols])
            
            normalized_df[numerical_cols] = normalized_data
            
            if fit:
                self.set_feature_names(list(normalized_df.columns))
            if not fit:
                normalized_df = self.align_features(normalized_df)
            
            self.logger.info(f"Normalization completed for {len(numerical_cols)} features")
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Error in feature normalization: {str(e)}")
            raise
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', 
                       k: int = 20, fit: bool = True) -> pd.DataFrame:
        """
        Select most important features
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method ('mutual_info', 'f_classif')
            k: Number of features to select
            fit: Whether to fit selector
            
        Returns:
            DataFrame with selected features
        """
        try:
            self.logger.info(f"Selecting {k} features using {method} method")
            
            score_func = mutual_info_classif if method == 'mutual_info' else f_classif
            
            if fit:
                selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
                X_selected = selector.fit_transform(X, y)
                
                self.feature_selectors[method] = selector
                selected_features = X.columns[selector.get_support()]
                
                X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                
                self.logger.info(f"Selected features: {list(selected_features)}")
                
            else:
                if method in self.feature_selectors:
                    selector = self.feature_selectors[method]
                    X_selected = selector.transform(X)
                    selected_features = X.columns[selector.get_support()]
                    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                else:
                    self.logger.warning(f"No fitted selector found for {method}, returning original features")
                    X_selected_df = X
            
            return X_selected_df
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return X
    
    def create_feature_interactions(self, df: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """
        Create feature interactions for better model performance
        
        Args:
            df: DataFrame with features
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with interaction features
        """
        try:
            self.logger.info(f"Creating {max_interactions} feature interactions")
            interaction_df = df.copy()
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != 'label']
            
            if len(numerical_cols) < 2:
                self.logger.warning("Need at least 2 numerical columns for interactions")
                return interaction_df
            
            important_features = numerical_cols[:5]
            interaction_count = 0
            
            for i, feat1 in enumerate(important_features):
                for feat2 in important_features[i+1:]:
                    if interaction_count >= max_interactions:
                        break
                    
                    try:
                        interaction_df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                        
                        ratio_col = f'{feat1}_div_{feat2}'
                        interaction_df[ratio_col] = np.where(
                            df[feat2] != 0,
                            df[feat1] / df[feat2],
                            0
                        )
                        
                        interaction_count += 2
                        
                    except Exception as e:
                        self.logger.warning(f"Error creating interaction {feat1}_{feat2}: {str(e)}")
                        continue
                
                if interaction_count >= max_interactions:
                    break
            
            self.logger.info(f"Created {interaction_count} interaction features")
            return interaction_df
            
        except Exception as e:
            self.logger.error(f"Error creating feature interactions: {str(e)}")
            return df
    
    def prepare_features_for_ml(self, df: pd.DataFrame, target_col: str = 'label', 
                               fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete feature preparation pipeline for machine learning
        
        Args:
            df: Raw DataFrame
            target_col: Name of target column
            fit: Whether to fit transformers
            
        Returns:
            Tuple of (prepared features DataFrame, target Series)
        """
        try:
            self.logger.info("Starting complete feature preparation pipeline")
            
            if target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[target_col]
            else:
                X = df.copy()
                y = None
                self.logger.warning(f"Target column '{target_col}' not found")
            
            X = self.extract_network_features(X)
            
            X = self.create_statistical_features(X)
            
            X = self.encode_categorical_features(X, fit=fit)
            
            X = self.create_feature_interactions(X)
            
            X = self.normalize_features(X, method='standard', fit=fit)
            
            if y is not None and fit and len(X) > 0:
                X = self.select_features(X, y, k=min(50, X.shape[1]), fit=fit)
            
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            self.logger.info(f"Feature preparation completed. Final shape: {X.shape}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation pipeline: {str(e)}")
            raise
    
    def get_feature_importance_summary(self, feature_names: List[str], 
                                     importance_scores: np.ndarray) -> pd.DataFrame:
        """
        Create feature importance summary
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            
        Returns:
            DataFrame with feature importance summary
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def set_feature_names(self, feature_names: list):
        """Set the feature names to enforce for transform/inference."""
        self.feature_names_ = feature_names

    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align columns to match training features, filling missing with 0 and dropping extras."""
        if not hasattr(self, 'feature_names_'):
            return df
        aligned = df.reindex(columns=self.feature_names_, fill_value=0)
        return aligned

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    sample_data = ingestion.create_sample_data(1000)
    
    feature_eng = FeatureEngineering()
    
    X, y = feature_eng.prepare_features_for_ml(sample_data)
    
    print(f"Prepared features shape: {X.shape}")
    print(f"Target shape: {y.shape if y is not None else 'None'}")
    print("\\nFeature columns:")
    print(list(X.columns)[:20])