from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained model
model = tf.keras.models.load_model("blood_group_cnn.h5")

# Define blood group labels
blood_group_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Initialize Flask app
app = Flask(__name__)

# Serve the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img = img.reshape(1, 128, 128, 1)  # Reshape for model
    return img

# API to handle image upload & prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)

    # Preprocess & Predict
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)

    return jsonify({"blood_group": blood_group_labels[predicted_label]})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
