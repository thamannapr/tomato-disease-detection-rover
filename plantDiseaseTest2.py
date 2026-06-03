import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "/home/plantdisease/Documents/models/mobilenetv2_plant.tflite"
LABEL_PATH = "/home/plantdisease/Documents/models/labels.txt"

THRESHOLD = 0.6

# Load labels
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

h = input_details[0]['shape'][1]
w = input_details[0]['shape'][2]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame,(w,h))
    img = img.astype(np.float32)/255.0
    img = np.expand_dims(img,axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    pred = np.argmax(output)
    confidence = np.max(output)

    label = labels[pred]

    # Tomato filter
    if label.startswith("tomato") and confidence > THRESHOLD:
        result = f"{label} ({confidence:.2f})"
        color = (0,255,0)
    else:
        result = "not a tomato leaf"
        color = (0,0,255)

    cv2.putText(frame,result,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("Tomato Disease Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()