import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "/home/plantdisease/Documents/models/mobilenetv2_plant.tflite"
LABEL_PATH = "/home/plantdisease/Documents/models/labels.txt"

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
    class_name = labels[pred]

    cv2.putText(frame,class_name,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Plant Disease Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()