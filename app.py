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

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

st.set_page_config(layout="wide", page_title="Plant Recognizer")

st.write("## Recognize plants ")

#st.sidebar.write("## Upload image :gear:")
st.write("## Upload image !:gear:")
@st.cache_resource
def load_model():
    return hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

def correct_image_orientation(image):
    try:
        exif = image._getexif()
        orientation = IFD.Orientation.value
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
            st.write("Rotation 180")
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
            st.write("Rotation 270")
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
            st.write("Rotation 90")
    except (AttributeError, KeyError, IndexError):
        # Cases: image don't have getexif
        st.write("no EXIF data found")
        image = image.rotate(270, expand=True)
        pass
    return image

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
    # Read the csv file with the classifier and get the category names as a dictionary
    df = pd.read_csv('classifier.csv')
    categories = dict(zip(df['id'], df['name']))

    # Get the name of the category from the ID and print it
    name = categories[predicted_class]
    # Return the predicted class name
    return name


def display_results(image, class_name):
    st.write(f"The plant in the image is a {class_name}.")
    st.write("Uploaded image:")
    st.image(image, width=400)

# Load the TensorFlow Hub model

m = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
#uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
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
