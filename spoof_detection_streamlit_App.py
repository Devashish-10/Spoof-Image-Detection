import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import gdown

#  Must be first Streamlit call
st.set_page_config(page_title="Spoof Detection System", layout="centered")

# Constants
FILE_ID = "1-W3XEcLKsce_ULy6BMgEkFdqxIIUoOxk"
MODEL_PATH = "spook_classifier_model.h5"
img_height, img_width = 150, 150

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Image prediction function
def predict_image(image):
    image = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "REAL" if prediction[0][0] >= 0.5 else "SPOOF"

# UI Elements
st.title(" Spoof Detection System")
st.markdown("### Detect whether an image is **REAL** or **SPOOFED** using a trained deep learning model.")

# Tabs for image upload and webcam
tab1, tab2 = st.tabs([" Upload Image", " Use Webcam"])

# Image Upload Tab
with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label = predict_image(image)
        st.markdown(f"###  Prediction: `{label}`")

# Webcam Tab
with tab2:
    st.subheader("Live Webcam Prediction")
    
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    toggle = st.button("Start Webcam" if not st.session_state.camera_active else "Stop Webcam")

    if toggle:
        st.session_state.camera_active = not st.session_state.camera_active

    FRAME_WINDOW = st.empty()

    if st.session_state.camera_active:
        camera = cv2.VideoCapture(0)
        while st.session_state.camera_active:
            ret, frame = camera.read()
            if not ret:
                st.error("Unable to access webcam.")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            label = predict_image(pil_img)

            # Display label on frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if label == "REAL" else (0, 0, 255), 2)

            FRAME_WINDOW.image(frame, channels="BGR")

        camera.release()
        cv2.destroyAllWindows()
