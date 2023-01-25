#!/usr/bin/env python
# coding: utf-8

import io

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import logging
logging.basicConfig(level=logging.DEBUG)

model = tf.keras.models.load_model('model/retinal-oct.h5')

def prepare_image(img, from_api=True):
    """
    prepares the image for the api call
    """
    if from_api:
        img = Image.open(io.BytesIO(img)).convert('RGB')
    else:
        img = Image.open(img).convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    """predicts the result"""
    return np.argmax(model.predict(img)[0])

app = Flask(__name__, template_folder="templates")

@app.route('/oct', methods=['GET'])
def oct():
    return render_template("predict.html")

@app.route('/show_prediction', methods=["GET", "POST"])
def show_prediction():
    image = request.files.get("image_import")
    image = prepare_image(image, from_api=False)
    pred = predict_result(image)
    labels = ["CNV", "DME", "DRUSEN", "NORMAL"]
    pred = labels[pred]
    return render_template("predict.html", pred=str(pred))

@app.route('/predict', methods=['POST'])
def infer_image():
    logging.info(str(request.files))
    
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')
    
    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return str(predict_result(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Retinal OCT prediction API'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
