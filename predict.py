import cv2
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load saved model
model = load_model("model_resnet50_unknown_class.h5")

# Class names
dataset_path = r"C:\project\medicinal-plant-detection\Datasets\Medicinal plant dataset"

class_names = os.listdir(dataset_path)
# Image path
img_path = r"C:\project\medicinal-plant-detection\Datasets\Medicinal plant dataset\Neem\1791.jpg"

# Check image exists
print("Image Exists:", os.path.exists(img_path))

# Read image
img = cv2.imread(img_path)

if img is None:
    print("Image not found!")
    exit()

# Resize image
img = cv2.resize(img, (224,224))

# Preprocess
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# Prediction
prediction = model.predict(x)

predicted_class = np.argmax(prediction)

print("Predicted Class Index:", predicted_class)

# Confidence
confidence = np.max(prediction) * 100

print("Confidence:", confidence)

# Print class name
print("Predicted Plant:", class_names[predicted_class])