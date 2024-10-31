import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load models
def load_model_svc():
    with open("svm_classifier.pkl", "rb") as file:
        svm_classifier = joblib.load(file)
    return svm_classifier

def load_features():
    feature_extractor = load_model("mobilenet_feature_extractor.keras")
    return feature_extractor

def load_scaler():
    with open("scaler.pkl", "rb") as file:
        scaler = joblib.load(file)
    return scaler

svm_classifier = load_model_svc()
feature_extractor = load_features()
scaler = load_scaler()



# CSS for consistent styling
def apply_custom_css():
    st.markdown("""
        <style>
        .header { font-size:2.2em; font-weight: bold; color: #4B0090; margin-bottom: 15px; }
        .subheader { font-size:1.2em; color: #808080; margin-top: 10px; }
        .uploaded { color: #007ACC; font-style: italic; }
        </style>
        """, unsafe_allow_html=True)

def show_predict_page():
    apply_custom_css()  # Apply custom CSS on each run for consistency

    st.markdown('<div class="header">Hand Sign Classifier</div>', unsafe_allow_html=True)
    st.image("signs.jpg", caption="Reference: Hand Signs Associated with Numbers")


    # File uploader, camera input, and URL input options
    st.sidebar.subheader("Upload, Capture, or Enter URL")
    st.sidebar.write("You can upload a file, capture an image, or enter an image URL to classify the hand sign.")

    with st.expander("Choose Input Method", expanded=True):
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        image_url = st.text_input("Or enter an image URL")
        # camera_image = st.camera_input("Or capture from camera")

    # Initialize the image variable
    image = None

    # Check if an image file is uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((224, 224))
        st.markdown('<p class="uploaded">Image uploaded from file:</p>', unsafe_allow_html=True)

    # # Check if an image is captured from camera
    # elif camera_image is not None:
    #     image = Image.open(camera_image).resize((224, 224))
    #     st.markdown('<p class="uploaded">Image captured from camera:</p>', unsafe_allow_html=True)

    # Check if a URL is provided
    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).resize((224, 224))
            st.markdown('<p class="uploaded">Image loaded from URL:</p>', unsafe_allow_html=True)
        except Exception:
            st.error("Error loading image from URL. Please check the URL format.")

    # Display and classify the image if available
    if image:
        st.image(image, caption="Input Image", width=150)  # Display smaller size

        if st.button("Classify"):
            with st.spinner("Classifying..."):
                img_array = img_to_array(image)  
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                img_features = feature_extractor.predict(img_array)
                img_features = img_features.reshape(1, -1)
                img_features = scaler.transform(img_features)
                predicted_class = svm_classifier.predict(img_features)

            st.subheader(f'Predicted hand Digit : {predicted_class[0]}')
