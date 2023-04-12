import streamlit as st
from PIL import Image
import base64
import io

st.set_page_config(layout="wide", page_title="Camera Image Capture")

st.write("## Capture an image with your camera")

with open("camera_input.html", "r") as f:
    html_template = f.read()

st.write(html_template, unsafe_allow_html=True)

if "image-captured" not in st.session_state:
    st.session_state.image_captured = None

if st.session_state.image_captured is not None:
    # Convert the image to PIL Image
    img_data = base64.b64decode(st.session_state.image_captured.split(",")[1])
    img = Image.open(io.BytesIO(img_data))

    # Display the captured image
    st.image(img, caption="Captured Image", use_column_width=True)

# Handle image data received from JavaScript
st.write(
    """
    <script>
    window.addEventListener("message", (event) => {
      if (event.data.type === "image-captured") {
        const img_data = event.data.data;
        window.Streamlit.setComponentValue("image-captured", img_data);
      }
    }, false);
    </script>
""",
    unsafe_allow_html=True,
)
