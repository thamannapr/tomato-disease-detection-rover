import serial
import time

ser = serial.Serial('/dev/ttyUSB0',115200,timeout=1)

time.sleep(2)  # allow ESP32 reset

# --- INIT ---
ser.write(b'PAN:90\n')
ser.write(b'TILT:90\n')
time.sleep(1)

# --- FORWARD ---
ser.write(b'F\n')
time.sleep(2)

# --- BACKWARD ---
ser.write(b'B\n')
time.sleep(2)

# --- STOP ---
ser.write(b'S\n')
time.sleep(1)

# --- PAN SWEEP (90 -> 180) ---
for angle in range(90,181,5):
    ser.write(f"PAN:{angle}\n".encode())
    time.sleep(0.05)

# --- PAN SWEEP (180 -> 0) ---
for angle in range(180,-1,-5):
    ser.write(f"PAN:{angle}\n".encode())
    time.sleep(0.05)

# --- PAN RETURN (0 -> 90) ---
for angle in range(0,91,5):
    ser.write(f"PAN:{angle}\n".encode())
    time.sleep(0.05)

# --- TILT SWEEP (90 -> 120) ---
for angle in range(90,121,3):
    ser.write(f"TILT:{angle}\n".encode())
    time.sleep(0.05)

# --- TILT SWEEP (120 -> 60) ---
for angle in range(120,59,-3):
    ser.write(f"TILT:{angle}\n".encode())
    time.sleep(0.05)

# --- TILT RETURN (60 -> 90) ---
for angle in range(60,91,3):
    ser.write(f"TILT:{angle}\n".encode())
    time.sleep(0.05)

ser.close()