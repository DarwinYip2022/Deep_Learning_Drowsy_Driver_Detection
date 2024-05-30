import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

# Load your object detection model
model = tf.keras.models.load_model("Object_Detection/detect_model.h5")

# Path to the new image
new_image_path = "PHOTO-2024-05-29-19-11-53.jpg"

# Read the new image
new_image = cv2.imread(new_image_path)
if new_image is None:
    print(f"Error: Unable to load image from {new_image_path}")
else:
    # Preprocess the new image (resize, normalize, etc.)
    preprocessed_new_image = cv2.resize(new_image, (244, 244))  # Resize to match the input size of your model
    
    # Make prediction on the preprocessed new image
    prediction = model.predict(preprocessed_new_image)
    print(prediction)
    # Extract bounding box coordinates from the prediction
    x, y, w, h = prediction[0][:4]  # Assuming the first four elements represent bounding box coordinates

    # Draw bounding box on the image
    cv2.rectangle(new_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Display the image with bounding box
    cv2.imshow("Image with Bounding Box", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
