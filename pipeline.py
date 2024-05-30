import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import cv2 as cv
import numpy as np


input_size = 244

# Load your object detection model
class_model = tf.keras.models.load_model("Object_Classification/class_model.h5")

# Path to the new image
new_image_path = "PHOTO-2024-05-29-19-11-53.jpg"

# Read the new image
new_image = cv2.imread(new_image_path)
if new_image is None:
    print(f"Error: Unable to load image from {new_image_path}")
else:

    # Load the new image
    image_path = "/Users/sameeraboppana/Desktop/DL_Project/PHOTO-2024-05-29-19-11-53.jpg" 
    new_image = cv.imread(image_path)

    # Check if the image was loaded successfully
    if new_image is None:
        print("Error: Unable to load image.")
    else:
        # Preprocess the new image (resize, normalize, etc.)
        preprocessed_new_image = cv2.resize(new_image, (244, 244))  # Resize to match the input size of your model
        
        # Make prediction on the preprocessed new image
        prediction = class_model.predict(np.expand_dims(preprocessed_new_image, axis=0))
        
        # Convert prediction to class label
        predicted_label = np.argmax(prediction)
        
        print("Predicted label for the new image:", predicted_label)