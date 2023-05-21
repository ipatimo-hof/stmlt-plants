import requests
import io
import pandas as pd
import streamlit as st
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


st.image('header.png', use_column_width=True)

st.write("## Pflanzenerkennung V.0.15")
@st.cache_resource
def load_model():
    return hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

from PIL.ExifTags import TAGS

def correct_image_orientation(image):
    try:
        exif = image.getexif()
        if exif is not None:
            orientation_key = None
            for key, value in TAGS.items():
                if value == 'Orientation':
                    orientation_key = key
                    break
            if orientation_key and orientation_key in exif:
              #  st.write(f"EXIF Orientation Value: {exif[orientation_key]}")
                if exif[orientation_key] == 3:
                    image = image.rotate(180, expand=True)
                  #  st.write("Rotation 180")
                elif exif[orientation_key] == 6:
                    image = image.rotate(270, expand=True)
                   # st.write("Rotation 270")
                elif exif[orientation_key] == 8:
                    image = image.rotate(90, expand=True)
                   # st.write("Rotation 90")
            else:
                ex_o=1
                #st.write("No 'Orientation' tag in EXIF data")
        else:
            ex_o=2
            #st.write("No EXIF data found")
            image = image.rotate(270, expand=True)
    except (AttributeError, KeyError) as e:
        #st.write(f"EXIF extraction failed with error: {str(e)}")
        pass
    except IndexError as e:
        st.write(f"IndexError during EXIF extraction: {str(e)}")
        pass
    return image

def predict_plant(image):
    # Convert the image to RGB
    image = image.convert('RGB')

    # Resize the image to the expected input size
    image = image.resize((224, 224))

    # Convert the image to a numpy array and add a batch dimension
    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=0)

    # Convert the image to a float tensor and normalize it
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32) / 255.0

    # Pass the image to the model and get the output tensor
    output = m(image_tensor)

    # Get the top 3 predictions
    top_3_predictions = tf.math.top_k(output, k=3)
    predicted_classes = top_3_predictions.indices.numpy()[0]
    predicted_probabilities = tf.nn.softmax(top_3_predictions.values).numpy()[0]

    # Load the class names
    df = pd.read_csv('classifier.csv')
    categories = dict(zip(df['id'], df['name']))

    # Get the names of the categories from the IDs and probabilities
    names_and_probabilities = [(categories[pred], prob) for pred, prob in zip(predicted_classes, predicted_probabilities)]

    # Return the predicted class names and probabilities
    return names_and_probabilities

def display_results(image, names_and_probabilities):
    # Read the list of "bad" plants from the file
    with open('plants.txt', 'r') as file:
        bad_plants = [plant.strip().lower() for plant in file.read().split(',')]
    

    for name, prob in names_and_probabilities:
        name = name.lower()
        output = f"Die Pflanze auf dem Bild ist möglicherweise eine {name} mit einer Wahrscheinlichkeit von {prob*100:.2f}%."
        if name in bad_plants:
            output += " Diese Pflanze ist eine Gefahr für das Gründach!"
            output = f"<span style='color:red'>{output}</span>"
        st.markdown(output, unsafe_allow_html=True)
        st.markdown(f"[Mehr über {name}](https://www.wikipedia.org/wiki/{name.replace(' ', '_')})")
    st.image(image, width=400)

# Load the TensorFlow Hub model

m = load_model()

uploaded_file = st.file_uploader("       ", type=["png", "jpg", "jpeg"])
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
    st.write("Bitte laden Sie ein Bild hoch, um loszulegen!")
st.write("Ungeeignete Anwendungsfälle:")
st.write("1. Diese App eignet sich nicht zur Bestimmung, ob eine Pflanze essbar, giftig oder toxisch ist.")
st.write("2. Diese App eignet sich nicht zur Bestimmung, ob die Pflanze auf dem Bild medizinische Anwendungen hat.")
st.write("3. Diese App eignet sich nicht zur Bestimmung des Standorts des Benutzers basierend auf den sichtbaren Pflanzen.")

