import streamlit as st
from PIL import Image

def handle_camera_input():
    # Capture the image from the camera
    camera_image = st.camera_input("Bild aufnehmen")
    
    # Check if an image has been captured
    if camera_image:
        # Since camera_image is a file-like object, we can directly open it with PIL
        with Image.open(camera_image) as img:
            # Process the image (e.g., display it or perform transformations)
            st.image(img, caption='Captured Image')

# Button to activate camera input
if st.button("Capture Image"):
    handle_camera_input()
