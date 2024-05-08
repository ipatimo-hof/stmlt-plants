import streamlit as st
from PIL import Image
import os

def save_image(image_file):
    if image_file is not None:
        # Save the image to a temporary file
        with open("temp_image.png", "wb") as f:
            f.write(image_file.getvalue())
        return "temp_image.png"
    return None

def load_and_display_image(image_path):
    if image_path is not None and os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Captured Image")
        os.remove(image_path)  # Optionally, delete the file after displaying

def handle_camera_input():
    camera_image = st.camera_input("Capture Image")
    if camera_image:
        image_path = save_image(camera_image)
        st.session_state['image_path'] = image_path

if 'image_path' not in st.session_state:
    st.session_state['image_path'] = None

if st.button("Capture Image"):
    handle_camera_input()

if st.session_state['image_path']:
    load_and_display_image(st.session_state['image_path'])
