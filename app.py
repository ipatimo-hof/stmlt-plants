import streamlit as st
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Setze die Seitenkonfiguration
st.set_page_config(layout="wide", page_title="Pflanzenerkenner")

st.image('header.png', use_column_width=True)
st.write("# Pflanzen auf dem Gründach erkennen V.0.15")

# Funktion zum Laden des Modells
@st.cache(allow_output_mutation=True)
def load_model():
    return hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

model = load_model()

# Funktion zur Korrektur der Bildausrichtung basierend auf EXIF-Daten
def correct_image_orientation(image):
    try:
        exif = image.getexif()
        orientation_key = 274 # Standard-EXIF-Tag für Orientierung
        if exif and orientation_key in exif:
            if exif[orientation_key] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation_key] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation_key] == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung der EXIF-Daten: {e}")
    return image

# Funktion zur Vorhersage der Pflanzenart
def predict_plant(image):
    image = image.convert('RGB').resize((224, 224))
    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=0) / 255.0
    output = model(tf.convert_to_tensor(image_np, dtype=tf.float32))
    top_3 = tf.math.top_k(output, k=3)
    df = pd.read_csv('classifier.csv')
    categories = dict(zip(df['id'], df['name']))
    names_probs = [(categories[i], prob) for i, prob in zip(top_3.indices.numpy()[0], tf.nn.softmax(top_3.values).numpy()[0])]
    return names_probs

# Funktion zur Anzeige der Ergebnisse
def display_results(image, results):
    st.image(image, width=400)
    for name, prob in results:
        st.markdown(f"**{name}**: {prob * 100:.2f}% Wahrscheinlichkeit")

# Kameraeingabe zum Erfassen eines Bildes vom Gerät
camera_image = st.camera_input("Bild aufnehmen")

# Datei-Uploader zum Hochladen einer Bilddatei
uploaded_file = st.file_uploader("Oder laden Sie eine Bilddatei hoch", type=["png", "jpg", "jpeg"])

# Verarbeitung des Bildes von Kamera oder Upload
if camera_image or uploaded_file:
    image = Image.open(camera_image if camera_image else uploaded_file)
    image = correct_image_orientation(image)
    results = predict_plant(image)
    display_results(image, results)
else:
    st.write("Bitte laden Sie ein Bild hoch oder nehmen Sie ein Foto auf, um die Erkennung zu starten.")

st.write("Hinweis: Diese Anwendung kann nicht bestimmen, ob eine Pflanze essbar, giftig oder medizinisch verwendbar ist.")

st.write("Ungeeignete Anwendungsfälle:")
st.write("1. Diese App eignet sich nicht zur Bestimmung, ob eine Pflanze essbar, giftig oder toxisch ist.")
st.write("2. Diese App eignet sich nicht zur Bestimmung, ob die Pflanze auf dem Bild medizinische Anwendungen hat.")
st.write("3. Diese App eignet sich nicht zur Bestimmung des Standorts des Benutzers basierend auf den sichtbaren Pflanzen.")

