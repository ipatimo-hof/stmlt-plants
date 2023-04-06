import streamlit as st
import requests
import io
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(layout="wide", page_title="Plant Recognizer")

st.write("## Recognize plants in your images")
st.write(
    ":seedling: Try uploading an image to see the plant species identified. This code is open source and available [here](https://github.com/tyler-simons/PlantRecognizer) on GitHub. Special thanks to [TensorFlow Hub](https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1) :grin:"
)
st.sidebar.write("## Upload and recognize :gear:")

# Load the TensorFlow Hub model
m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')


def predict_plant(image_bytes):
    # Load the image and preprocess it
    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image /= 255.
    # Predict the plant using the TensorFlow Hub model
    result = m(tf.expand_dims(image, axis=0))
    # Get the class label with the highest probability
    predicted_class = tf.argmax(result, axis=1)[0]
    # Load the class names
    classes_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    classes_file = requests.get(classes_url).text
    class_names = classes_file.split("\n")
    # Return the predicted class name
    return class_names[predicted_class]


def display_results(image, class_name):
    st.write(f"The plant in the image is a {class_name}.")
    st.write("Uploaded image:")
    st.image(image, width=400)


uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Analyzing...")
    with st.spinner("Waiting for model inference..."):
        class_name = predict_plant(uploaded_file.read())
    display_results(image, class_name)
else:
    st.write("Please upload an image to get started.")    
