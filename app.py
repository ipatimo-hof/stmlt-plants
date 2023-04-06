import streamlit as st
import requests
import io



import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image



st.set_page_config(layout="wide", page_title="Plant Recognizer")

st.write("## Recognize plants in your images")

st.sidebar.write("## Upload and recognize :gear:")

# Load the TensorFlow Hub model


m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

def predict_plant(image_bytes):
    # Convert the image to a numpy array and add a batch dimension
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # Convert the image to a float tensor and normalize it
    image = tf.cast(image, tf.float32) / 255.0
    # Pass the image to the model and get the output tensor
    output = m(image)
    predicted_class = tf.argmax(output, axis=1)[0]
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
    image = image.resize((224, 224))
    with st.spinner("Waiting for model inference..."):
        
        class_name = predict_plant(image)

    display_results(image, class_name)
else:
    st.write("Please upload an image to get started.")    

'''




# Load the image and resize it
image = Image.open('C:/Users/ptimofeev/Pictures/plants/f2.jpg')
image = image.resize((224, 224))



# Find the index of the category with the highest logit value and print it
index = np.argmax(output)
print(index)
'''
