import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import io
import sys
import os
from streamlit_autorefresh import st_autorefresh

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_ingestion import DataIngestion
    from src.feature_engineering import FeatureEngineering
    from src.ml_models import MLModels
    from src.real_time_detector import RealTimeAnomalyDetector, Alert
    from src.packet_capture_thread import PacketCaptureThread
except ImportError:
    from data_ingestion import DataIngestion
    from feature_engineering import FeatureEngineering
    from ml_models import MLModels
    from real_time_detector import RealTimeAnomalyDetector, Alert

st.set_page_config(
    page_title="Cybersecurity Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.alert-critical {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.alert-high {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.alert-medium {
    background-color: #f3e5f5;
    border-left: 5px solid #9c27b0;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.alert-low {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

class CyberSecurityDashboard:
    """Main dashboard class for the cybersecurity application"""
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        if 'feature_engineering' not in st.session_state or st.session_state.feature_engineering is None:
            st.session_state.feature_engineering = FeatureEngineering()
        self.feature_engineering = st.session_state.feature_engineering
        self.ml_models = MLModels()
        self.detector = None
        
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'packet_capture_thread' not in st.session_state:
            st.session_state.packet_capture_thread = None
        if 'packet_capture_active' not in st.session_state:
            st.session_state.packet_capture_active = False
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
    
    def main(self):
        """Main dashboard function"""
        st.title("Cybersecurity Intrusion Detection System")
        st.markdown("---")
        
        self.render_sidebar()
        
        page = st.session_state.get('current_page', 'overview')
        
        if page == 'overview':
            self.render_overview()
        elif page == 'data_upload':
            self.render_data_upload()
        elif page == 'training':
            self.render_model_training()
        elif page == 'monitoring':
            self.render_real_time_monitoring()
        elif page == 'alerts':
            self.render_alerts()
        elif page == 'analytics':
            self.render_analytics()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.title("Navigation")
            
            pages = {
                'overview': 'Overview',
                'data_upload': 'Data Upload',
                'training': 'Model Training',
                'monitoring': 'Real-time Monitoring',
                'alerts': 'Security Alerts',
                'analytics': 'Analytics'
            }
            
            selected_page = st.radio("Select Page", list(pages.values()))
            st.session_state.current_page = [k for k, v in pages.items() if v == selected_page][0]
            
            st.markdown("---")
            
            st.subheader("System Status")
            
            if st.session_state.models_trained:
                st.success("Models Trained")
            else:
                st.warning("Models Not Trained")
            
            if st.session_state.monitoring_active:
                st.success("Monitoring Active")
            else:
                st.info("ℹMonitoring Inactive")
            
            if st.session_state.alerts:
                st.metric("Active Alerts", len(st.session_state.alerts))
    
    def render_overview(self):
        """Render overview dashboard"""
        st.header("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Packets Processed",
                value=st.session_state.get('total_packets', 0),
                delta=st.session_state.get('packet_delta', 0)
            )
        
        with col2:
            st.metric(
                label="Threats Detected",
                value=len(st.session_state.alerts),
                delta=st.session_state.get('threat_delta', 0)
            )
        
        with col3:
            st.metric(
                label="Detection Accuracy",
                value=f"{st.session_state.get('accuracy', 0):.1%}",
                delta=f"{st.session_state.get('accuracy_delta', 0):.1%}"
            )
        
        with col4:
            st.metric(
                label="System Health",
                value="Healthy" if st.session_state.models_trained else "Setup Required"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Threat Detection Timeline")
            if st.session_state.alerts:
                alert_df = pd.DataFrame([alert.to_dict() for alert in st.session_state.alerts])
                alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
                
                fig = px.histogram(
                    alert_df, 
                    x='timestamp', 
                    color='severity',
                    title="Alerts Over Time",
                    color_discrete_map={
                        'critical': '#f44336',
                        'high': '#ff9800',
                        'medium': '#9c27b0',
                        'low': '#4caf50'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No alerts to display. Start monitoring to see real-time data.")
        
        with col2:
            st.subheader("Attack Type Distribution")
            if st.session_state.alerts:
                alert_df = pd.DataFrame([alert.to_dict() for alert in st.session_state.alerts])
                attack_counts = alert_df['alert_type'].value_counts()
                
                fig = px.pie(
                    values=attack_counts.values,
                    names=attack_counts.index,
                    title="Attack Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload data and train models to see attack distribution.")
        
        st.subheader("Recent Alerts Summary")
        if st.session_state.alerts:
            recent_alerts = sorted(st.session_state.alerts, key=lambda x: x.timestamp, reverse=True)[:5]
            
            for alert in recent_alerts:
                severity_class = f"alert-{alert.severity}"
                st.markdown(f"""
                <div class="{severity_class}">
                    <strong>{alert.alert_type.replace('_', ' ').title()}</strong> - {alert.severity.upper()}<br>
                    <small>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {alert.description}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent alerts to display.")
    
    def render_data_upload(self):
        """Render data upload interface"""
        st.header("Data Upload & Management")
        
        tab1, tab2, tab3 = st.tabs(["Upload Data", "Data Preview", "Data Statistics"])
        
        with tab1:
            st.subheader("Upload Network Traffic Data")
            
            upload_type = st.radio(
                "Select data source:",
                ["Upload CSV File", "Upload PCAP File", "Generate Sample Data"]
            )
            
            if upload_type == "Upload CSV File":
                uploaded_file = st.file_uploader(
                    "Choose a CSV file",
                    type=['csv'],
                    help="Upload network traffic data in CSV format (CICIDS2017, NSL-KDD, or similar)"
                )
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.current_data = df
                        
                        st.success(f"Successfully loaded {len(df)} records with {len(df.columns)} features")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Records", len(df))
                            st.metric("Features", len(df.columns))
                        with col2:
                            if 'label' in df.columns:
                                anomaly_rate = df['label'].mean()
                                st.metric("Anomaly Rate", f"{anomaly_rate:.2%}")
                        
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
            
            elif upload_type == "Upload PCAP File":
                uploaded_file = st.file_uploader(
                    "Choose a PCAP file",
                    type=['pcap', 'pcapng'],
                    help="Upload packet capture file for analysis"
                )
                
                if uploaded_file is not None:
                    with st.spinner("Processing PCAP file..."):
                        try:
                            temp_path = f"/tmp/{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            df = self.data_ingestion.load_pcap_data(temp_path, max_packets=1000)
                            st.session_state.current_data = df
                            
                            st.success(f"Successfully processed {len(df)} packets")
                            
                            os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"Error processing PCAP file: {str(e)}")
            
            else:  # Generate Sample Data
                col1, col2 = st.columns(2)
                with col1:
                    n_samples = st.number_input("Number of samples", min_value=100, max_value=10000, value=1000)
                with col2:
                    anomaly_rate = st.slider("Anomaly rate", min_value=0.01, max_value=0.5, value=0.1)
                
                if st.button("Generate Sample Data"):
                    with st.spinner("Generating sample data..."):
                        df = self.data_ingestion.create_sample_data(n_samples, anomaly_rate)
                        st.session_state.current_data = df
                        
                        st.success(f"Generated {len(df)} sample records")
        
        with tab2:
            st.subheader("Data Preview")
            if st.session_state.current_data is not None:
                df = st.session_state.current_data
                
                st.dataframe(df.head(100), use_container_width=True)
                
                st.subheader("Data Information")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
                
            else:
                st.info("No data loaded. Please upload data in the Upload Data tab.")
        
        with tab3:
            st.subheader("Data Statistics")
            if st.session_state.current_data is not None:
                df = st.session_state.current_data
                
                validation_results = self.data_ingestion.validate_data(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(validation_results)
                
                with col2:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        selected_feature = st.selectbox("Select feature for distribution", numeric_cols)
                        
                        fig = px.histogram(df, x=selected_feature, title=f"Distribution of {selected_feature}")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data loaded. Please upload data first.")
    
    def render_model_training(self):
        """Render model training interface"""
        st.header("Machine Learning Model Training")
        
        if st.session_state.current_data is None:
            st.warning("Please upload data first before training models.")
            return
        
        tab1, tab2, tab3 = st.tabs(["Train Models", "Model Performance", "Model Management"])
        
        with tab1:
            st.subheader("Train Detection Models")
            
            df = st.session_state.current_data
            
            st.write("Select models to train:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                train_rf = st.checkbox("Random Forest", value=True)
            with col2:
                train_ae = st.checkbox("Autoencoder", value=True)
            with col3:
                train_lstm = st.checkbox("LSTM", value=True)
            
            st.subheader("Training Options")
            col1, col2 = st.columns(2)
            
            with col1:
                hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
            with col2:
                test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.4, value=0.2)
            
            if st.button("Start Training", type="primary"):
                with st.spinner("Training models... This may take several minutes."):
                    try:
                        X, y = self.feature_engineering.prepare_features_for_ml(df)
                        
                        training_results = {}
                        
                        if train_rf:
                            st.write("Training Random Forest...")
                            rf_results = self.ml_models.train_random_forest(
                                X, y, hyperparameter_tuning=hyperparameter_tuning
                            )
                            training_results['random_forest'] = rf_results
                        
                        if train_ae:
                            st.write("Training Autoencoder...")
                            ae_results = self.ml_models.train_autoencoder(X)
                            training_results['autoencoder'] = ae_results
                        
                        if train_lstm:
                            st.write("Training LSTM...")
                            lstm_results = self.ml_models.train_lstm(X, y)
                            training_results['lstm'] = lstm_results
                        
                        st.session_state.training_results = training_results
                        st.session_state.models_trained = True
                        
                        st.success("Model training completed!")
                        
                        st.subheader("Training Results Summary")
                        for model_name, results in training_results.items():
                            performance = results['performance']
                            if 'accuracy' in performance:
                                st.metric(
                                    f"{model_name.title()} Accuracy",
                                    f"{performance['accuracy']:.3f}"
                                )
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
        
        with tab2:
            st.subheader("Model Performance Analysis")
            
            if st.session_state.models_trained:
                training_results = st.session_state.get('training_results', {})
                
                performance_data = []
                for model_name, results in training_results.items():
                    perf = results['performance']
                    if 'accuracy' in perf:
                        performance_data.append({
                            'Model': model_name.title(),
                            'Accuracy': perf['accuracy'],
                            'Precision': perf.get('precision', 0),
                            'Recall': perf.get('recall', 0),
                            'F1-Score': perf.get('f1_score', 0)
                        })
                
                if performance_data:
                    perf_df = pd.DataFrame(performance_data)
                    
                    fig = go.Figure()
                    
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    for metric in metrics:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=perf_df['Model'],
                            y=perf_df[metric]
                        ))
                    
                    fig.update_layout(
                        title="Model Performance Comparison",
                        xaxis_title="Models",
                        yaxis_title="Score",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(perf_df, use_container_width=True)
                
                if 'random_forest' in training_results:
                    st.subheader("Feature Importance (Random Forest)")
                    importance_df = training_results['random_forest']['feature_importance']
                    
                    fig = px.bar(
                        importance_df.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 15 Most Important Features"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trained models available. Please train models first.")
        
        with tab3:
            st.subheader("Model Management")
            
            if st.session_state.models_trained:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Save Models"):
                        try:
                            save_dir = "./models"
                            self.ml_models.save_models(save_dir)
                            st.success(f"Models saved to {save_dir}")
                        except Exception as e:
                            st.error(f"Error saving models: {str(e)}")
                
                with col2:
                    if st.button("Load Models"):
                        try:
                            save_dir = "./models"
                            self.ml_models.load_models(save_dir)
                            st.success(f"Models loaded from {save_dir}")
                            st.session_state.models_trained = True
                        except Exception as e:
                            st.error(f"Error loading models: {str(e)}")
                
                st.subheader("Model Information")
                for model_name in self.ml_models.models.keys():
                    with st.expander(f"{model_name.title()} Model Details"):
                        model_info = self.ml_models.models[model_name]
                        st.json({
                            'Model Type': model_info['model_type'],
                            'Performance': model_info['performance']
                        })
            else:
                st.info("No models available. Please train models first.")
    
    def render_real_time_monitoring(self):
        st.header("Real-time Network Monitoring")
        st_autorefresh(interval=2000)
        
        if not st.session_state.models_trained:
            st.warning("Please train models first before starting monitoring.")
            return
        
        if 'detector' not in st.session_state or st.session_state.detector is None:
            st.session_state.detector = RealTimeAnomalyDetector(
                models=self.ml_models,
                feature_engineer=st.session_state.feature_engineering,
                config={'monitoring_interval': 2}
            )
            def add_alert_to_session(alert):
                st.session_state.alerts.append(alert)
            st.session_state.detector.add_alert_callback(add_alert_to_session)
        self.detector = st.session_state.detector
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Start Monitoring", type="primary"):
                self.detector.start_monitoring()
                st.session_state.monitoring_active = True
                st.success("Monitoring started")
        with col2:
            if st.button("Stop Monitoring"):
                if self.detector:
                    self.detector.stop_monitoring()
                st.session_state.monitoring_active = False
                st.info("ℹMonitoring stopped")
        with col3:
            if st.button("Simulate Data"):
                if self.detector:
                    with st.spinner("Simulating network traffic..."):
                        self.detector.simulate_real_time_data(duration_minutes=1)
                    st.success("Simulation completed")
        with col4:
            if not st.session_state.packet_capture_active:
                if st.button("Start Packet Capture", type="primary"):
                    if st.session_state.packet_capture_thread is None:
                        st.session_state.packet_capture_thread = PacketCaptureThread(self.detector, iface='eth0')
                    st.session_state.packet_capture_thread.start()
                    st.session_state.packet_capture_active = True
                    st.success("Packet capture started on eth0")
            else:
                if st.button("Stop Packet Capture"):
                    if st.session_state.packet_capture_thread is not None:
                        st.session_state.packet_capture_thread.stop()
                    st.session_state.packet_capture_active = False
                    st.info("ℹPacket capture stopped")
        
        if st.session_state.monitoring_active:
            current_packets = self.detector.stats.get('total_packets', 0) if self.detector else 0
            if current_packets != st.session_state.get('last_packet_count', 0):
                st.session_state.last_packet_count = current_packets
                st.session_state.last_update = time.time()
                st.rerun()
            
            col_refresh = st.columns([1, 4])[0]
            with col_refresh:
                if st.button("Refresh Dashboard"):
                    st.rerun()
            
            placeholder = st.empty()
            with placeholder.container():
                stats = self.detector.stats if self.detector else {}
                total_packets = stats.get('total_packets', 0)
                alerts_generated = stats.get('alerts_generated', 0)
                if self.detector and getattr(self.detector, 'monitoring_start_time', None):
                    uptime_seconds = int((datetime.now() - self.detector.monitoring_start_time).total_seconds())
                else:
                    uptime_seconds = 0
                uptime_str = str(timedelta(seconds=uptime_seconds)) if uptime_seconds else "--"

                # --- Real-time packets/sec calculation ---
                buffer = list(self.detector.data_buffer) if self.detector else []
                if 'recent_packet_times' not in st.session_state:
                    st.session_state.recent_packet_times = []
                if buffer:
                    last_packet_time = buffer[-1]['timestamp'] if 'timestamp' in buffer[-1] else None
                    if last_packet_time:
                        if not st.session_state.recent_packet_times or last_packet_time > st.session_state.recent_packet_times[-1]:
                            st.session_state.recent_packet_times.append(last_packet_time)
                now_ts = datetime.now().timestamp()
                st.session_state.recent_packet_times = [t for t in st.session_state.recent_packet_times if now_ts - t <= 1.0]
                packets_per_sec = len(st.session_state.recent_packet_times)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Status", "🟢 Active")
                with col2:
                    st.metric("Packets/sec", packets_per_sec)
                with col3:
                    st.metric("Threats", alerts_generated)
                with col4:
                    st.metric("Uptime", uptime_str)

                st.subheader("Live Traffic Analysis")
                buffer = list(self.detector.data_buffer) if self.detector else []
                if buffer:
                    df = pd.DataFrame(buffer)
                    
                    numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'src_port', 'dst_port', 'packet_size', 'tcp_flags', 'count', 'srv_count']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                    if 'timestamp' in df:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                        df = df.sort_values('timestamp')
                        traffic_counts = df.set_index('timestamp').resample('1T').size()
                        if hasattr(self.detector, 'last_anomaly_scores'):
                            anomaly_scores = self.detector.last_anomaly_scores
                        else:
                            anomaly_scores = [0]*len(df)
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=['Network Traffic Volume', 'Anomaly Scores'],
                            vertical_spacing=0.1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=traffic_counts.index,
                                y=traffic_counts.values,
                                mode='lines+markers',
                                name='Packets',
                                line=dict(color='blue')
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df['timestamp'],
                                y=anomaly_scores,
                                mode='lines+markers',
                                name='Anomaly Score',
                                line=dict(color='red')
                            ),
                            row=2, col=1
                        )
                        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Threshold", row=2, col=1)
                        fig.update_layout(height=500, title_text="Real-time Network Monitoring")
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("Recent Captured Packets")
                        display_cols = [col for col in ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol_type', 'packet_size'] if col in df.columns]
                        if display_cols:
                            st.dataframe(df[display_cols].sort_values('timestamp', ascending=False).head(20), use_container_width=True)
                        else:
                            st.info("No packet details available yet.")
                    else:
                        st.info("No timestamp data available in buffer.")
                else:
                    st.info("No real-time data in buffer yet. Start simulation or feed live data.")
                st.subheader("Debug: Raw Buffer")
                if buffer:
                    st.write(pd.DataFrame(buffer).tail(10))
                else:
                    st.info("No packets in buffer.")

                st.subheader("Live Alerts")
                alerts = st.session_state.alerts if hasattr(st.session_state, 'alerts') else []
                if alerts:
                    alerts_df = pd.DataFrame([a.to_dict() if hasattr(a, 'to_dict') else dict(a) for a in alerts])
                    alerts_df = alerts_df.sort_values('timestamp', ascending=False)
                    st.dataframe(alerts_df[['timestamp', 'severity', 'alert_type', 'description', 'source_ip', 'destination_ip', 'confidence']].head(10), use_container_width=True)
                else:
                    st.info("No alerts generated yet. Start monitoring or simulate data.")
        
        st.subheader("Monitoring Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Thresholds")
            anomaly_threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.7, 0.01)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
            
            if st.button("Update Thresholds"):
                if self.detector:
                    self.detector.update_thresholds({
                        'anomaly_score': anomaly_threshold,
                        'confidence': confidence_threshold
                    })
                    st.success("Thresholds updated")
        
        with col2:
            st.subheader("Alert Settings")
            alert_severity = st.multiselect(
                "Alert Severity Levels",
                ['low', 'medium', 'high', 'critical'],
                default=['medium', 'high', 'critical']
            )
            
            email_alerts = st.checkbox("Enable Email Alerts")
            slack_alerts = st.checkbox("Enable Slack Notifications")
    
    def render_alerts(self):
        """Render security alerts interface"""
        st.header("Security Alerts Management")
        
        if not st.session_state.alerts:
            st.info("No alerts available. Start monitoring to generate alerts.")
            return
        
        alerts_df = pd.DataFrame([alert.to_dict() for alert in st.session_state.alerts])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", len(alerts_df))
        with col2:
            critical_count = len(alerts_df[alerts_df['severity'] == 'critical'])
            st.metric("Critical Alerts", critical_count)
        with col3:
            high_count = len(alerts_df[alerts_df['severity'] == 'high'])
            st.metric("High Priority", high_count)
        with col4:
            avg_confidence = alerts_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        st.subheader("Filter Alerts")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "Severity",
                options=['low', 'medium', 'high', 'critical'],
                default=['medium', 'high', 'critical']
            )
        
        with col2:
            attack_types = alerts_df['alert_type'].unique()
            type_filter = st.multiselect(
                "Attack Type",
                options=attack_types,
                default=attack_types
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Range",
                ['Last Hour', 'Last 6 Hours', 'Last 24 Hours', 'All Time']
            )
        
        filtered_alerts = alerts_df[
            (alerts_df['severity'].isin(severity_filter)) &
            (alerts_df['alert_type'].isin(type_filter))
        ]
        
        st.subheader("Alert Timeline")
        if not filtered_alerts.empty:
            filtered_alerts['timestamp'] = pd.to_datetime(filtered_alerts['timestamp'])
            
            fig = px.timeline(
                filtered_alerts.sort_values('timestamp'),
                x_start='timestamp',
                x_end='timestamp',
                y='alert_type',
                color='severity',
                title="Security Alerts Timeline",
                color_discrete_map={
                    'critical': '#f44336',
                    'high': '#ff9800',
                    'medium': '#9c27b0',
                    'low': '#4caf50'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Alert Details")
        
        for i, alert in enumerate(filtered_alerts.to_dict('records')):
            severity_class = f"alert-{alert['severity']}"
            
            with st.expander(f"Alert #{i+1} - {alert['alert_type'].replace('_', ' ').title()} [{alert['severity'].upper()}]"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Timestamp:** {alert['timestamp']}")
                    st.write(f"**Source IP:** {alert['source_ip']}")
                    st.write(f"**Destination IP:** {alert['destination_ip']}")
                    st.write(f"**Confidence:** {alert['confidence']:.3f}")
                
                with col2:
                    st.write(f"**Description:** {alert['description']}")
                    st.write(f"**Status:** {alert['status']}")
                    
                    if st.button(f"Acknowledge Alert #{i+1}"):
                        st.success(f"Alert #{i+1} acknowledged")
                    
                    if st.button(f"Mark Resolved #{i+1}"):
                        st.success(f"Alert #{i+1} marked as resolved")
                
                if alert['additional_info']:
                    st.write("**Additional Information:**")
                    st.json(alert['additional_info'])
    
    def render_analytics(self):
        """Render analytics and reporting interface"""
        st.header("Security Analytics & Reporting")
        
        if not st.session_state.alerts:
            st.info("No data available for analytics. Please generate some alerts first.")
            return
        
        alerts_df = pd.DataFrame([alert.to_dict() for alert in st.session_state.alerts])
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        tab1, tab2, tab3 = st.tabs(["Overview Analytics", "Attack Analysis", "Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Alert Severity Distribution")
                severity_counts = alerts_df['severity'].value_counts()
                
                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Alert Distribution by Severity",
                    color_discrete_map={
                        'critical': '#f44336',
                        'high': '#ff9800',
                        'medium': '#9c27b0',
                        'low': '#4caf50'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Attack Type Analysis")
                attack_counts = alerts_df['alert_type'].value_counts()
                
                fig = px.bar(
                    x=attack_counts.index,
                    y=attack_counts.values,
                    title="Attacks by Type"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Detailed Attack Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Source IPs**")
                src_ip_counts = alerts_df['source_ip'].value_counts().head(10)
                st.dataframe(src_ip_counts)
            
            with col2:
                st.write("**Top Target IPs**")
                dst_ip_counts = alerts_df['destination_ip'].value_counts().head(10)
                st.dataframe(dst_ip_counts)
            
            st.subheader("Detection Confidence Analysis")
            fig = px.histogram(
                alerts_df,
                x='confidence',
                nbins=20,
                title="Distribution of Alert Confidence Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Security Trends")
            
            alerts_df['hour'] = alerts_df['timestamp'].dt.hour
            alerts_df['day'] = alerts_df['timestamp'].dt.date
            
            hourly_counts = alerts_df['hour'].value_counts().sort_index()
            
            fig = px.line(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Alert Frequency by Hour of Day"
            )
            fig.update_xaxes(title="Hour of Day")
            fig.update_yaxes(title="Number of Alerts")
            st.plotly_chart(fig, use_container_width=True)
            
            daily_counts = alerts_df['day'].value_counts().sort_index()
            
            fig = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Alert Frequency Over Time"
            )
            fig.update_xaxes(title="Date")
            fig.update_yaxes(title="Number of Alerts")
            st.plotly_chart(fig, use_container_width=True)

def main():
    dashboard = CyberSecurityDashboard()
    dashboard.main()

if __name__ == "__main__":
    main()