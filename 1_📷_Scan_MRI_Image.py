import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Detect TBC in MRI Image",
    page_icon="📷"
)

# Function to load the model from Google Drive
def load_model_from_drive(model_path):
    load_path = '/content/self_care_1_mod.h5' + model_path
    model = load_model(load_path)
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Function to make predictions using the loaded model
def make_prediction(image):
    model = load_model('./content/self_care_1.h5')
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    class_label = 'TBC' if prediction > 0.5 else 'NORMAL'
    return class_label

# Streamlit App
st.title("Detect TBC in MRI Image")
st.write('Upload Lung MRI Scan.')

uploaded_file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('SCAN!'):
        predicted_label = make_prediction(image)
        st.success(f'Prediction: {predicted_label}')

# st.sidebar.success("Select a page above.")