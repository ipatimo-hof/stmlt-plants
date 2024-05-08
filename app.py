import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import ImageOps
import os

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

st.set_page_config(layout="wide", page_title="Plant Recognizer")
st.image('header.png', use_column_width=True)
st.write("# Pflanzen auf dem Gründach erkennen V.0.16")

# Load the TensorFlow Hub model
@st.cache(allow_output_mutation=True)
def load_model():
    return hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

model = load_model()

def correct_image_orientation(image):
    try:
        exif = image._getexif()
        orientation_key = [key for key, value in ExifTags.TAGS.items() if value == 'Orientation'][0]
        if orientation_key and orientation_key in exif:
            if exif[orientation_key] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation_key] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation_key] == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        st.error(f"EXIF extraction failed with error: {e}")
    return image

def predict_plant(image):
    image = image.convert('RGB').resize((224, 224))
    image_np = tf.keras.preprocessing.image.img_to_array(image)
    image_np = tf.expand_dims(image_np, 0) / 255.0
    output = model(image_np)
    top_3_predictions = tf.math.top_k(output, k=3)
    df = pd.read_csv('classifier.csv')
    categories = dict(zip(df['id'], df['name']))
    return [(categories[i], prob) for i, prob in zip(top_3_predictions.indices.numpy()[0], tf.nn.softmax(top_3_predictions.values).numpy()[0])]

def display_results(image, names_and_probabilities):
    st.image(image, width=400)
    for name, prob in names_and_probabilities:
        st.markdown(f"**{name}**: {prob * 100:.2f}% chance")

# Handling camera input with session_state
if 'captured_image' not in st.session_state:
    st.session_state['captured_image'] = None

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    st.session_state['captured_image'] = camera_image

uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.session_state['captured_image'] = uploaded_file

if st.session_state['captured_image'] is not None:
    image = Image.open(st.session_state['captured_image'])
    image = correct_image_orientation(image)
    results = predict_plant(image)
    display_results(image, results)

st.write("Ungeeignete Anwendungsfälle:")
st.write("1. Diese App eignet sich nicht zur Bestimmung, ob eine Pflanze essbar, giftig oder toxisch ist.")
st.write("2. Diese App eignet sich nicht zur Bestimmung, ob die Pflanze auf dem Bild medizinische Anwendungen hat.")
st.write("3. Diese App eignet sich nicht zur Bestimmung des Standorts des Benutzers basierend auf den sichtbaren Pflanzen.")
