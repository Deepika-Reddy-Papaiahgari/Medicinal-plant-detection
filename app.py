import os
import cv2
import numpy as np


from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load trained model
model = load_model("models/model_resnet50_unknown_class.h5")

# Dataset classes
dataset_path = r"C:\project\medicinal-plant-detection\Datasets\Medicinal plant dataset"

class_names = os.listdir(dataset_path)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["image"]
        if file:

            filename = file.filename

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            # Read image
            img = cv2.imread(filepath)

            img = cv2.resize(img, (224,224))

            x = np.expand_dims(img, axis=0)

            x = preprocess_input(x)

            result = model.predict(x)

            predicted_index = np.argmax(result)

            confidence = np.max(result) * 100

            prediction = f"{class_names[predicted_index]} ({confidence:.2f}%)"

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=filepath
            )

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)