<<<<<<< HEAD
# Medicinal Plant Classification Web Application using ResNet50

## Overview

This project is a Deep Learning based web application that classifies medicinal plant leaf images using the ResNet50 Convolutional Neural Network (CNN) architecture.

The system allows users to upload a leaf image through a web interface and predicts the medicinal plant name with confidence score.

The project is built using:

* TensorFlow/Keras
* ResNet50 Transfer Learning
* Flask Web Framework
* OpenCV
* Python

---

## Features

* Upload leaf images through website
* Medicinal plant classification
* Transfer Learning using ResNet50
* Confidence score prediction
* Unknown leaf detection using thresholding
* Simple and user-friendly web interface
* Real-time image prediction

---

## Technologies Used

* Python
* TensorFlow
* Keras
* Flask
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Dataset

The model is trained on a medicinal plant dataset containing multiple medicinal plant leaf classes.

Example dataset structure:

```text id="k3l1ma"
Datasets/
└── Medicinal plant dataset/
    ├── Neem/
    ├── Tulsi/
    ├── AloeVera/
    ├── Mint/
    └── ...
```

---

## Model Architecture

* ResNet50 pretrained CNN model
* Transfer Learning approach
* Dense neural network layers
* Softmax activation for multiclass classification

---

## Project Structure

```text id="s7k0pn"
medicinal-plant-detection/
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── Datasets/
├── app.py
├── predict.py
├── resnet_50_finalised_training_code.py
├── model_resnet50_unknown_class.h5
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash id="k2p5dz"
git clone https://github.com/YOUR_USERNAME/medicinal-plant-detection.git
cd medicinal-plant-detection
```

---

### Create Virtual Environment

```bash id="x4l2oe"
python -m venv venv
```

---

### Activate Virtual Environment

#### Windows

```bash id="m1p8cw"
venv\Scripts\activate
```

---

### Install Dependencies

```bash id="v6r1hs"
pip install -r requirements.txt
```

---

## Run Training

```bash id="z5y8gt"
python resnet_50_finalised_training_code.py
```

---

## Run Prediction

```bash id="h8q4nj"
python predict.py
```

---

## Run Flask Web Application

```bash id="n2u7ra"
python app.py
```

---

## Open Website

Open browser and visit:

```text id="b4f0ly"
http://127.0.0.1:5000
```

---

## Website Workflow

1. Upload medicinal plant leaf image
2. Model preprocesses image
3. ResNet50 predicts plant class
4. Website displays prediction and confidence score

---

## Results

* Successfully classified medicinal plant leaf images
* Generated prediction confidence scores
* Visualized training accuracy and loss
* Built functional Flask web application

---

## Future Improvements

* Mobile responsive UI
* Real-time camera detection
* Plant disease identification
* Cloud deployment
* Android application integration
* GPU acceleration support

---

## Author

Deepika Reddy Papaiahgari

---

## License

This project is developed for educational and research purposes.
=======
# Medicinal-plant-detection
Medicinal Plant Detection using Deep Learning with TensorFlow/Keras. This project uses pretrained CNN models like ResNet50 and VGG16 with transfer learning to classify medicinal plants from leaf images. Includes image preprocessing, data augmentation, model training, evaluation, and Flask-based web deployment.
>>>>>>> c8498ff481eaec749b4b42b08fb9e0772cfcd1c4
