"""
Background thread for packet capture to be used with Streamlit
"""
import threading
from src.packet_capture import PacketCapture

class PacketCaptureThread:
    def __init__(self, detector, iface=None, filter_str=None):
        self.capture = PacketCapture(detector, iface=iface, filter_str=filter_str)
        self.thread = None

    def start(self):
        if not self.capture.running:
            self.thread = threading.Thread(target=self.capture.start, daemon=True)
            self.thread.start()

    def stop(self):
        self.capture.stop()
        if self.thread:
            self.thread.join(timeout=2)
