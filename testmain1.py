import os
import cv2
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
    using_tflite_runtime = True
except ImportError:
    import tensorflow as tf
    using_tflite_runtime = False
import serial
import time
import threading
from queue import Queue
from threading import Event

# ==================================================
# CONFIGURATION
# ==================================================
DEFAULT_MODEL_PATH = "/home/plantdisease/Documents/models/mobilenetv2_plant.tflite"
DEFAULT_LABEL_PATH = "/home/plantdisease/Documents/models/labels.txt"

MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
LABEL_PATH = os.environ.get("LABEL_PATH", DEFAULT_LABEL_PATH)
THRESHOLD = 0.6
FRAME_QUEUE_SIZE = 2
INFERENCE_QUEUE_SIZE = 2

# ==================================================
# GLOBAL STATE
# ==================================================
frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
inference_queue = Queue(maxsize=INFERENCE_QUEUE_SIZE)
stop_event = Event()
ser = None

# ==================================================
# SERIAL WORKER
# ==================================================
def serial_worker():
    """Handles serial communication with ESP32"""
    global ser
    try:
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        time.sleep(2)
        print("[SERIAL] Connected to ESP32")
        
        # Initialization sequence
        print("[INIT] Pan-tilt initialization...")
        send("PAN:90")
        send("TILT:90")
        time.sleep(1)
        
        # Sweep scan
        for angle in range(60, 121, 10):
            send(f"PAN:{angle}")
            send(f"TILT:{angle}")
            time.sleep(0.3)
        
        for angle in range(120, 59, -10):
            send(f"PAN:{angle}")
            send(f"TILT:{angle}")
            time.sleep(0.3)
        
        # Return center
        send("PAN:90")
        send("TILT:90")
        time.sleep(1)
        print("[INIT] Pan-tilt initialization done")
        
        # Move forward
        print("[INIT] Moving forward 5 seconds")
        send("F")
        time.sleep(5)
        send("S")
        
    except Exception as e:
        print(f"[SERIAL ERROR] {e}")
    finally:
        if ser:
            ser.close()

def send(cmd):
    """Send command to ESP32 via serial"""
    if ser and ser.is_open:
        try:
            ser.write((cmd + "\n").encode())
            print(f"[SERIAL] {cmd}")
        except Exception as e:
            print(f"[SERIAL ERROR] Failed to send '{cmd}': {e}")

# ==================================================
# CAMERA WORKER
# ==================================================
def camera_worker():
    """Captures frames from camera"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAMERA ERROR] Cannot open camera")
        stop_event.set()
        return
    
    print("[CAMERA] Camera opened")
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA] End of stream")
                break
            
            # Drop old frames if queue is full
            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                frame_queue.get()  # Drop oldest
                frame_queue.put(frame)
    finally:
        cap.release()
        print("[CAMERA] Camera released")

# ==================================================
# INFERENCE WORKER
# ==================================================
def inference_worker(h, w, interpreter, input_details, output_details, labels):
    """Runs ML inference on frames"""
    print("[INFERENCE] Worker started")
    
    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except:
                continue
            
            # 1. Classification
            img = cv2.resize(frame, (w, h))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            pred = np.argmax(output)
            confidence = np.max(output)
            label = labels[pred]
            
            # 2. Bounding Box Detection (Color-based)
            bbox = None
            if label.startswith("tomato") and confidence > THRESHOLD:
                result = f"{label} ({confidence:.2f})"
                color = (0, 255, 0)
                
                # Find the leaf using green color masking
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Green range (standard values)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Find the largest contour
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 500: # Min area filter
                        bbox = cv2.boundingRect(largest)
            else:
                result = "not a tomato leaf"
                color = (0, 0, 255)
            
            result_data = {
                'frame': frame,
                'result': result,
                'color': color,
                'bbox': bbox
            }
            
            # Drop old results if queue is full
            if not inference_queue.full():
                inference_queue.put(result_data)
            else:
                inference_queue.get()  # Drop oldest
                inference_queue.put(result_data)
    
    except Exception as e:
        print(f"[INFERENCE ERROR] {e}")
    finally:
        print("[INFERENCE] Worker stopped")

# ==================================================
# DISPLAY WORKER
# ==================================================
def display_worker():
    """Displays annotated frames"""
    print("[DISPLAY] Worker started")
    
    try:
        while not stop_event.is_set():
            try:
                result_data = inference_queue.get(timeout=1)
            except:
                continue
            
            frame = result_data['frame']
            result = result_data['result']
            color = result_data['color']
            bbox = result_data.get('bbox')
            
            # Draw bounding box if it exists
            if bbox is not None and len(bbox) == 4:
                x, y, wb, hb = bbox
                # Ensure they are integers
                x, y, wb, hb = int(x), int(y), int(wb), int(hb)
                cv2.rectangle(frame, (x, y), (x + wb, y + hb), color, 2)
                cv2.putText(frame, "LEAF", (x, max(0, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, result, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Tomato Disease Detection", frame)
            
            if cv2.waitKey(1) == 27:  # ESC key
                stop_event.set()
                break
    
    except Exception as e:
        print(f"[DISPLAY ERROR] {e}")
    finally:
        cv2.destroyAllWindows()
        print("[DISPLAY] Worker stopped")

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    try:
        # Load model
        print("[MAIN] Loading model...")
        if not os.path.exists(MODEL_PATH):
            print(f"[MAIN ERROR] Model file not found at {MODEL_PATH}")
            import sys
            sys.exit(1)
        
        if not os.path.exists(LABEL_PATH):
            print(f"[MAIN ERROR] Label file not found at {LABEL_PATH}")
            import sys
            sys.exit(1)

        with open(LABEL_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        
        if using_tflite_runtime:
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        else:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        h = input_details[0]['shape'][1]
        w = input_details[0]['shape'][2]
        print(f"[MAIN] Model loaded. Input size: {w}x{h}")
        
        # Start worker threads
        print("[MAIN] Starting workers...")
        serial_thread = threading.Thread(target=serial_worker, daemon=True)
        camera_thread = threading.Thread(target=camera_worker, daemon=True)
        inference_thread = threading.Thread(target=inference_worker, 
                                           args=(h, w, interpreter, input_details, output_details, labels),
                                           daemon=True)
        display_thread = threading.Thread(target=display_worker, daemon=True)
        
        serial_thread.start()
        time.sleep(1)  # Let serial init
        camera_thread.start()
        inference_thread.start()
        display_thread.start()
        
        print("[MAIN] All workers started. Press ESC in window to stop.")
        
        # Wait for stop signal
        display_thread.join()
        stop_event.set()
        
        # Wait for all threads
        camera_thread.join(timeout=2)
        inference_thread.join(timeout=2)
        serial_thread.join(timeout=2)
        
        print("[MAIN] All workers stopped. Shutdown complete.")
    
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
    finally:
        stop_event.set()
        if ser:
            ser.close()

