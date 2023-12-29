import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import generate_caption as GC

import os
import tensorflow as tf
import base64
from flask import Flask, render_template, url_for, request
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.ops.gen_array_ops import Concat
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# Loading the model
model = load_model("model3.h5", compile=False)
app = Flask(__name__)

# default home page or route
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/Prediction')
def Prediction():
    return render_template('Prediction.html')


@app.route('/PredictCaption', methods=["GET", "POST"])
def upload():

    if request.method == "POST":
        file = request.files['image']
        # getting the current path i.e where app.py is present
        basepath = os.path.dirname(__file__)
        print("current path",basepath)
        # from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        filepath = os.path.join(basepath, 'uploads', file.filename)
        print("upload folder is",filepath)
        file.save(filepath)
        
        captions = GC.generate_captions(filepath)
        
    with open(filepath, 'rb') as uploadedfile:
        img_base64 = base64.b64encode(uploadedfile.read()).decode()
    return render_template('Prediction.html', prediction=str(captions), image=img_base64)

""" Running our application """
if __name__ == '__main__':
    app.run(debug=True , port= 1100)
