import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import rag_chatbot_lib as glib

# Page configuration
st.set_page_config(
    page_title="Detect TBC in MRI Image"
)

# Function to load the model
def load_model_from_drive(model_path):
    load_path = '/self_care_1.h5' + model_path
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
    model = load_model('./self_care_1.h5')
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    class_label = 'TBC' if prediction > 0.5 else 'NORMAL'
    
    if class_label == "NORMAL":
        st.session_state.chat_history.append({"role": "assistant", "text": f"Gambar MRI Scan yang diunggah adalah paru-paru {class_label}!"})
        # st.session_state.chat_history.append({"role": "user", "text": input_text})
        chat_response = glib.get_rag_image_response(memory=st.session_state.memory, index=st.session_state.vector_index, class_label=class_label)
        st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
    
    if class_label == "TBC":
        st.session_state.chat_history.append({"role": "assistant", "text": f"Gambar MRI Scan yanng diunggah adalah paru-paru {class_label}!"})
        chat_response = glib.get_rag_image_response(memory=st.session_state.memory, index=st.session_state.vector_index, class_label=class_label)
        st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
        
    return class_label

# Streamlit App
st.title("Deteksi TBC Dalam Gambar Scan MRI")

uploaded_file = st.file_uploader('Unggah Gambar MRI Scan', type=['jpg', 'jpeg', 'png'])

predicted_label = ""

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Scan MRI Yang Diunggah', width=150)

    predicted_label = make_prediction(image)
    # if predicted_label == "NORMAL":
    #     st.session_state.chat_history.append({"role": "assistant", "text": f"Gambar MRI Scan yang diunggah adalah paru-paru {predicted_label}."})
    #     # st.session_state.chat_history.append({"role": "user", "text": input_text})
    #     chat_response = glib.get_rag_image_response(memory=st.session_state.memory, index=st.session_state.vector_index, class_label=predicted_label)
    #     st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
    
    # if predicted_label == "TBC":
    #     st.session_state.chat_history.append({"role": "assistant", "text": f"Gambar MRI Scan yanng diunggah adalah paru-paru {predicted_label} "})
    #     chat_response = glib.get_rag_image_response(memory=st.session_state.memory, index=st.session_state.vector_index, class_label=predicted_label)
    #     st.session_state.chat_history.append({"role": "assistant", "text": chat_response})

# Chatbot Integration
if 'memory' not in st.session_state:
    st.session_state.memory = glib.get_memory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_index' not in st.session_state:
    with st.spinner("Indexing document..."):
        st.session_state.vector_index = glib.get_index()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("Chat with your bot here")

if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)

    st.session_state.chat_history.append({"role": "user", "text": input_text})

    chat_response = glib.get_rag_chat_response(input_text=input_text, memory=st.session_state.memory, index=st.session_state.vector_index)

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
