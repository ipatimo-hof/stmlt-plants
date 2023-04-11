import streamlit as st
import requests
import io
import pandas as pd

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

st.set_page_config(layout="wide", page_title="Plant Recognizer")

st.write("## Recognize plants in your images")

st.sidebar.write("## Upload and recognize :gear:")

def predict_plant(image):
    # Convert the image to a numpy array and add a batch dimension
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # Convert the image to a float tensor and normalize it
    image = tf.cast(image, tf.float32) / 255.0
    # Pass the image to the model and get the output tensor
    output = m(image)
    predicted_class = tf.argmax(output, axis=1)[0].numpy()
    # Load the class names
    # Read the csv file with the classifier and get the category names as a list
    df = pd.read_csv('classifier.csv')
    categories = df['name'].tolist()

    # Get the name of the category from the index and print it
    name = categories[predicted_class]
    # Return the predicted class name
    return name

def display_results(image, class_name):
    st.write(f"The plant in the image is a {class_name}.")
    st.write("Uploaded image:")
    st.image(image, width=400)

# Load the TensorFlow Hub model
m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    with st.spinner("Waiting for model inference..."):
        class_name = predict_plant(image)

    display_results(image, class_name)
else:
    st.write("Please upload an image to get started.")
