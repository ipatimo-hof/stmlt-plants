import streamlit as st
import requests
import io
import pandas as pd

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from PIL.ExifTags import IFD
from PIL import ImageOps
import os

def predict_plant(image):
    # Convert the image to RGB (3 channels)
    image = image.convert('RGB')
    
    # Convert the image to a numpy array and add a batch dimension
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    
    # Convert the image to a float tensor and normalize it
    image = tf.cast(image, tf.float32) / 255.0
    
    # Pass the image to the model and get the output tensor
    output = m(image)
    
    # Get the top 3 predictions
    top_3_predictions = tf.math.top_k(output, k=3)
    predicted_classes = top_3_predictions.indices.numpy()[0]
    predicted_probabilities = tf.nn.softmax(top_3_predictions.values).numpy()[0]
    
    # Load the class names
    # Read the csv file with the classifier and get the category names as a dictionary
    df = pd.read_csv('classifier.csv')
    categories = dict(zip(df['id'], df['name']))
    
    # Get the names of the categories from the IDs and probabilities
    names_and_probabilities = [(categories[pred], prob) for pred, prob in zip(predicted_classes, predicted_probabilities)]
    
    # Return the predicted class names and probabilities
    return names_and_probabilities

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

st.set_page_config(layout="wide", page_title="Plant Recognizer")


st.image('header.png', use_column_width=True)
st.write("## Upload image :gear:")

MODEL_URL = 'https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1'

@st.cache_resource
def load_model():
    return hub.KerasLayer(MODEL_URL)

from PIL.ExifTags import TAGS

# ... The rest of your code ...

uploaded_file = st.file_uploader("Upload an image or make a photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the TensorFlow Hub model
    m = load_model()

    image = Image.open(uploaded_file)
    image = correct_image_orientation(image)
    max_size = (224, 224)
    image.thumbnail(max_size)
    # Pad the image to the desired size while maintaining aspect ratio
    width, height = image.size
    delta_w = max_size[0] - width
    delta_h = max_size[1] - height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    image = ImageOps.expand(image, padding)
    with st.spinner("Waiting for model inference..."):
        class_name = predict_plant(image)

    display_results(image, class_name)
else:
    st.write("Please upload an image to get started!")
