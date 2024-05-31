
from ultralytics import YOLO
model = YOLO('./best.pt')
results = model(source=0, show=True, conf=0.3, save=True)

'''
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# Ensure the ultralytics module is installed: pip install ultralytics
from ultralytics import YOLO

# Load YOLO model
model = YOLO('./best.pt')

# Load classification model
classification_model_path = './efficientnet_model.h5'
classification_model = load_model(classification_model_path)

# Function to preprocess image for YOLOv8
def preprocess_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img.transpose((2, 0, 1))  # Transpose to (C, H, W) format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1] and convert to float32
    return img

# Function to run YOLOv8 with classification on camera feed
def run_yolo_with_classification(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess image for YOLO model
        img = preprocess_image(frame)

        # Perform YOLOv8 inference
        results = model(img)

        # Process each detected object
        for item in results.xywh[0]:
            bbox = item[:4]  # Get bounding box coordinates
            bbox = [int(b) for b in bbox]  # Convert to integers
            crop_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]  # Crop bounding box area

            # Ensure crop_img has valid shape
            if crop_img.size == 0:
                continue

            crop_img = cv2.resize(crop_img, (224, 224))  # Resize image for classification model
            crop_img = preprocess_input(crop_img)  # Preprocess input for classification model

            # Perform classification
            pred_class = classification_model.predict(np.expand_dims(crop_img, axis=0))
            pred_label = 'eyes_opened' if pred_class[0] < 0.5 else 'eyes_closed'  # Example logic, adjust as per your model

            # Draw bounding box and classification label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, pred_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with bounding boxes and classifications
        cv2.imshow('YOLOv8 with Classification', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run YOLOv8 with classification on the local camera
run_yolo_with_classification()
'''

