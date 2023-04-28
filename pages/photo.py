import streamlit as st
from PIL import Image
import base64
import io
import os

st.set_page_config(layout="wide", page_title="Camera Image Capture")

st.write("## Capture an image with your camera")

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "camera_input.html"), "r") as f:
    html_template = f.read()

st.markdown(html_template, unsafe_allow_html=True)

if "image-captured" not in st.session_state:
    st.session_state.image_captured = None

if st.session_state.image_captured is not None:
    # Convert the image to PIL Image
    img_data = base64.b64decode(st.session_state.image_captured.split(",")[1])
    img = Image.open(io.BytesIO(img_data))

    # Display the captured image
    st.image(img, caption="Captured Image", use_column_width=True)

# Handle image data received from JavaScript
st.write(" ", unsafe_allow_html=True)