import streamlit as st
from PIL import Image
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import ImageOps
import os

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
st.set_page_config(layout="wide", page_title="Plant Recognizer")

if 'captured_image' not in st.session_state:
    st.session_state['captured_image'] = None

st.image('header.png', use_column_width=True)
st.write("# Pflanzen auf dem Gründach erkennen V.0.16")

@st.cache(allow_output_mutation=True)
def load_model():
    return hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

model = load_model()

def correct_image_orientation(image):
    try:
        exif = image.getexif()
        orientation_key = 274  # Default EXIF Orientation Tag
        if orientation_key in exif:
            orientation = exif[orientation_key]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        st.error(f"Error in processing EXIF data: {e}")
    return image

def predict_plant(image):
    image = image.convert('RGB').resize((224, 224))
    image_np = np.array(image)
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32) / 255.0
    output = model(tf.expand_dims(image_tensor, 0))
    top_3_predictions = tf.math.top_k(output, k=3)
    df = pd.read_csv('classifier.csv')
    categories = dict(zip(df['id'], df['name']))
    return [(categories[i], prob) for i, prob in zip(top_3_predictions.indices.numpy()[0], tf.nn.softmax(top_3_predictions.values).numpy()[0])]

def display_results(image, results):
    st.image(image, width=400)
    for name, prob in results:
        st.markdown(f"**{name}**: {prob * 100:.2f}% chance")

def handle_camera_input():
    camera_image = st.camera_input("Bild aufnehmen")
    if camera_image:
        st.session_state['captured_image'] = camera_image

if st.button("Kamera verwenden"):
    handle_camera_input()

if st.session_state['captured_image'] is not None:
    image = Image.open(st.session_state['captured_image'])
    image = correct_image_orientation(image)
    results = predict_plant(image)
    display_results(image, results)

st.write("Ungeeignete Anwendungsfälle:")
st.write("1. Diese App eignet sich nicht zur Bestimmung, ob eine Pflanze essbar, giftig oder toxisch ist.")
st.write("2. Diese App eignet sich nicht zur Bestimmung, ob die Pflanze auf dem Bild medizinische Anwendungen hat.")
st.write("3. Diese App eignet sich nicht zur Bestimmung des Standorts des Benutzers basierend auf den sichtbaren Pflanzen.")
