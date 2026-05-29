import sys
import cv2
import numpy as np
import os
import time
import threading
from queue import Queue
from threading import Event

try:
    import tflite_runtime.interpreter as tflite
    using_tflite_runtime = True
except ImportError:
    try:
        import tensorflow as tf
        using_tflite_runtime = False
    except ImportError:
        print("[CRITICAL] Neither tflite-runtime nor tensorflow found.")
        sys.exit(1)

import serial
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap

# ==================================================
# CONFIGURATION
# ==================================================
DEFAULT_MODEL_PATH = "/home/plantdisease/Documents/models/mobilenetv2_plant.tflite"
DEFAULT_LABEL_PATH = "/home/plantdisease/Documents/models/labels.txt"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
LABEL_PATH = os.environ.get("LABEL_PATH", DEFAULT_LABEL_PATH)
THRESHOLD = 0.6

# ==================================================
# WORKER SIGNALS
# ==================================================
class WorkerSignals(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = Event()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.ser = None

# ==================================================
# MAIN WINDOW
# ==================================================
class PlantDiseaseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plant Disease Detection Rover")
        self.resize(1024, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 1. Camera Section (70%)
        self.camera_label = QLabel("Camera Loading...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        self.layout.addWidget(self.camera_label, 70)

        # 2. Control Section (30%)
        self.control_panel = QWidget()
        self.control_panel.setStyleSheet("background-color: #222; color: white; border-left: 2px solid #444;")
        self.control_layout = QVBoxLayout(self.control_panel)
        self.layout.addWidget(self.control_panel, 30)

        self.setup_ui()
        self.setup_workers()

        # Timer to update UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)

        # Auto-pump state
        self.pump_active = False
        self.last_pump_time = 0

    def setup_ui(self):
        # Summary / Info
        self.info_group = QWidget()
        self.info_layout = QVBoxLayout(self.info_group)
        self.result_label = QLabel("Waiting for Inference...")
        self.result_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #55efc4;")
        self.conf_bar = QLabel("Confidence: 0%")
        self.conf_bar.setStyleSheet("font-size: 10pt; color: #81ecec;")
        self.info_layout.addWidget(self.result_label)
        self.info_layout.addWidget(self.conf_bar)
        self.control_layout.addWidget(self.info_group)

        self.control_layout.addSpacing(20)
        
        # Movement Controls
        self.control_layout.addWidget(QLabel("ROVER CONTROLS"))
        grid = QGridLayout()
        self.buttons = {}
        controls = [('F', 0, 1), ('L', 1, 0), ('STOP', 1, 1), ('R', 1, 2), ('B', 2, 1)]
        for label, r, c in controls:
            btn = QPushButton(label)
            btn.setMinimumHeight(60)
            btn.setStyleSheet("""
                QPushButton { background-color: #34495e; color: white; border-radius: 8px; font-weight: bold; border: 1px solid #2c3e50; }
                QPushButton:pressed { background-color: #2c3e50; }
            """)
            grid.addWidget(btn, r, c)
            self.buttons[label] = btn

        self.control_layout.addLayout(grid)
        self.control_layout.addSpacing(20)

        # Servos
        self.control_layout.addWidget(QLabel("CAMERA GIMBAL"))
        self.pan_slider = self.create_slider("PAN")
        self.tilt_slider = self.create_slider("TILT")

        # Pump
        self.btn_pump = QPushButton("START PUMP")
        self.btn_pump.setCheckable(True)
        self.btn_pump.setMinimumHeight(60)
        self.btn_pump.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; border-radius: 8px; font-weight: bold; font-size: 14pt; }
            QPushButton:checked { background-color: #c0392b; }
        """)
        self.control_layout.addWidget(self.btn_pump)

        self.control_layout.addSpacing(20)

        # Soil Sensor Data
        self.control_layout.addWidget(QLabel("SOIL SENSOR (MODBUS)"))
        self.soil_widgets = {}
        metrics = ["Moisture", "Temperature", "EC", "pH", "N", "P", "K"]
        soil_grid = QGridLayout()
        for i, m in enumerate(metrics):
            lbl = QLabel(f"{m}: --")
            lbl.setStyleSheet("background: #1e1e1e; padding: 5px; border-radius: 4px; border: 1px solid #333;")
            soil_grid.addWidget(lbl, i // 2, i % 2)
            self.soil_widgets[m] = lbl
        self.control_layout.addLayout(soil_grid)
        
        self.control_layout.addStretch()

        # Connections
        self.buttons['F'].pressed.connect(lambda: self.send_cmd("F"))
        self.buttons['B'].pressed.connect(lambda: self.send_cmd("B"))
        self.buttons['L'].pressed.connect(lambda: self.send_cmd("L"))
        self.buttons['R'].pressed.connect(lambda: self.send_cmd("R"))
        for b in ['F', 'B', 'L', 'R']: self.buttons[b].released.connect(lambda: self.send_cmd("S"))
        self.buttons['STOP'].clicked.connect(lambda: self.send_cmd("S"))
        
        self.btn_pump.toggled.connect(self.toggle_pump)
        self.pan_slider.valueChanged.connect(lambda v: self.send_cmd(f"PAN:{v}"))
        self.tilt_slider.valueChanged.connect(lambda v: self.send_cmd(f"TILT:{v}"))

    def create_slider(self, label):
        self.control_layout.addWidget(QLabel(label))
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 180)
        s.setValue(90)
        s.setStyleSheet("""
            QSlider::handle:horizontal { background: #3498db; width: 18px; border-radius: 9px; }
            QSlider::groove:horizontal { background: #444; height: 8px; border-radius: 4px; }
        """)
        self.control_layout.addWidget(s)
        return s

    def toggle_pump(self, state):
        if state:
            self.btn_pump.setText("STOP PUMP")
            self.send_cmd("PUMP:ON")
        else:
            self.btn_pump.setText("START PUMP")
            self.send_cmd("PUMP:OFF")

    def trigger_auto_pump(self):
        print("[AUTO] Tomato detected! Starting 3s pump...")
        self.pump_active = True
        self.btn_pump.setChecked(True) # Visual feedback
        self.toggle_pump(True)
        
        # Turn off after 3 seconds
        QTimer.singleShot(3000, self.stop_auto_pump)

    def stop_auto_pump(self):
        print("[AUTO] 3s complete. Stopping pump.")
        self.toggle_pump(False)
        self.btn_pump.setChecked(False)
        self.pump_active = False
        self.last_pump_time = time.time()
        
    def setup_workers(self):
        self.stop_event = Event()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        
        # Load labels and model
        with open(LABEL_PATH, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        if using_tflite_runtime:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        # Threads
        self.cam_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.inf_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.ser_thread = threading.Thread(target=self.serial_worker, daemon=True)
        
        self.cam_thread.start()
        self.inf_thread.start()
        self.ser_thread.start()

    def send_cmd(self, cmd):
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            try:
                self.ser.write((cmd + "\n").encode())
                print(f"[UI] Sent: {cmd}")
            except:
                pass

    def camera_worker(self):
        cap = cv2.VideoCapture(0)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
        cap.release()

    def inference_worker(self):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except:
                continue
            
            # 1. Classification
            img = cv2.resize(frame, (w, h))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            self.interpreter.set_tensor(input_details[0]['index'], img)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_details[0]['index'])
            
            pred = np.argmax(output)
            confidence = np.max(output)
            label = self.labels[pred]
            
            # 2. Tomato Filter and Bounding Box
            bbox = None
            is_tomato = label.startswith("tomato") and confidence > THRESHOLD
            
            if is_tomato:
                # Find the leaf using green color masking
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 500:
                        bbox = cv2.boundingRect(largest)
            else:
                label = "Not a Tomato Leaf"
            
            self.result_queue.put({
                'frame': frame,
                'label': label,
                'conf': confidence if is_tomato else 0,
                'bbox': bbox,
                'is_tomato': is_tomato
            })

    def serial_worker(self):
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
            while not self.stop_event.is_set():
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # Logic to identify soil data (e.g., M:15.2% T:25.1C)
                        self.parse_serial(line)
                time.sleep(0.01)
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")

    def parse_serial(self, line):
        # Example espSerial print: "M:%.1f%% T:%.1fC EC:%.0f pH:%.1f N:%u P:%u K:%u"
        if "M:" in line and "T:" in line:
            self.soil_line = line # Store for UI update
        elif "SOIL" in line:
            print(f"[ESP] {line}")
            
    def update_ui(self):
        try:
            result = self.result_queue.get_nowait()
            frame = result['frame']
            label = result['label']
            conf = result['conf']
            bbox = result['bbox']
            is_tomato = result.get('is_tomato', False)

            # Update Sidebar Text
            self.result_label.setText(label.upper())
            if is_tomato:
                self.conf_bar.setText(f"Confidence: {conf*100:.1f}%")
                # Auto-Pump Logic (only if not already pumping and cooldown passed)
                if not getattr(self, 'pump_active', False):
                    now = time.time()
                    last_time = getattr(self, 'last_pump_time', 0)
                    if now - last_time > 10: # 10s cooldown
                        self.trigger_auto_pump()
            else:
                self.conf_bar.setText("") # Hide confidence for non-tomato

            # Draw bbox on frame (optional check: if you REALLY want it off camera, comment this)
            if bbox:
                x, y, wb, hb = bbox
                cv2.rectangle(frame, (x, int(y)), (x + int(wb), int(y) + int(hb)), (0, 255, 0), 2)

            # Update Soil Data if available
            if hasattr(self, 'soil_line'):
                # M:12.3% T:25.0C EC:100 pH:6.5 N:50 P:30 K:40
                parts = self.soil_line.split()
                for p in parts:
                    if ':' in p:
                        key, val = p.split(':')
                        if key == 'M': self.soil_widgets['Moisture'].setText(f"Moisture: {val}")
                        elif key == 'T': self.soil_widgets['Temperature'].setText(f"Temp: {val}")
                        elif key == 'EC': self.soil_widgets['EC'].setText(f"EC: {val}")
                        elif key == 'pH': self.soil_widgets['pH'].setText(f"pH: {val}")
                        elif key == 'N': self.soil_widgets['N'].setText(f"N: {val}")
                        elif key == 'P': self.soil_widgets['P'].setText(f"P: {val}")
                        elif key == 'K': self.soil_widgets['K'].setText(f"K: {val}")

            # Convert to Pixmap
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
                self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        except:
            pass

    def closeEvent(self, event):
        self.stop_event.set()
        if hasattr(self, 'ser') and self.ser:
            self.ser.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlantDiseaseGUI()
    window.show()
    sys.exit(app.exec())
