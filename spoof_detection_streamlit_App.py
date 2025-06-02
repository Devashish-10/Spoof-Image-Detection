import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load trained model
model = tf.keras.models.load_model("spook_classifier_model.h5")
img_height, img_width = 150, 150

# Prediction function
def predict_image(image):
    image = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "REAL" if prediction[0][0] >= 0.5 else "SPOOF"

# UI setup
st.set_page_config(page_title="Spoof Detection System", layout="centered")
st.title("Spoof Detection System")
st.markdown("### Detect whether an image is **REAL** or **SPOOFED** using a trained deep learning model.")

# Tabs: Upload Image or Webcam
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Use Webcam"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label = predict_image(image)
        st.markdown(f"### Prediction: `{label}`")

with tab2:
    st.subheader("Live Webcam Prediction")
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # Convert to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        label = predict_image(pil_img)

        # Overlay prediction (without confidence)
        cv2.putText(frame, f"{label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if label == "REAL" else (0,0,255), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    else:
        camera.release()
        cv2.destroyAllWindows()
