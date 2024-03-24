"""
Packet Capture Module for Real-Time Intrusion Detection
Captures live network packets using pyshark and feeds them to the anomaly detector
"""

import pyshark
import pandas as pd
from datetime import datetime
import logging

class PacketCapture:
    def __init__(self, detector, iface=None, filter_str=None):
        self.detector = detector
        self.iface = iface
        self.filter_str = filter_str
        self.capture = None
        self.running = False
        self.logger = logging.getLogger('PacketCapture')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _process_packet(self, pkt):
        try:
            data = {
                'duration': 0.0,
                'src_bytes': float(pkt.length) if hasattr(pkt, 'length') else 0.0,
                'dst_bytes': 0.0,
                'protocol_type': pkt.transport_layer.lower() if pkt.transport_layer else 'unknown',
                'src_port': int(pkt.tcp.srcport) if hasattr(pkt, 'tcp') and hasattr(pkt.tcp, 'srcport') else (int(pkt.udp.srcport) if hasattr(pkt, 'udp') and hasattr(pkt.udp, 'srcport') else 0),
                'dst_port': int(pkt.tcp.dstport) if hasattr(pkt, 'tcp') and hasattr(pkt.tcp, 'dstport') else (int(pkt.udp.dstport) if hasattr(pkt, 'udp') and hasattr(pkt.udp, 'dstport') else 0),
                'packet_size': int(pkt.length),
                'tcp_flags': int(pkt.tcp.flags, 16) if hasattr(pkt, 'tcp') and pkt.tcp.flags else 0,
                'count': 1,
                'srv_count': 1,
                'timestamp': pkt.sniff_time.timestamp(),
                'src_ip': pkt.ip.src if hasattr(pkt, 'ip') else None,
                'dst_ip': pkt.ip.dst if hasattr(pkt, 'ip') else None
            }
            self.detector.add_data(data)
            self.logger.info(f"Packet captured: {data}")
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")

    def start(self):
        self.running = True
        self.capture = pyshark.LiveCapture(interface=self.iface, bpf_filter=self.filter_str)
        self.logger.info("Starting packet capture...")
        for pkt in self.capture.sniff_continuously():
            if not self.running:
                break
            self._process_packet(pkt)
        self.capture.close()
        self.logger.info("Packet capture stopped.")

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.close()
