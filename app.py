import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os
import tensorflow_hub as hub

# Download model if not present
file_id = "1CtBcbYXQtj77TdKccndc7r6FaAxv7m6T"
download_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "dog_breed_model.h5"

if not os.path.exists(model_path):
    st.write("Downloading model...")
    gdown.download(download_url, model_path, quiet=False)
    st.write("Download complete!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

# Dog breeds list
dog_breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
 'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel',
 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
 'doberman', 'english_foxhound', 'english_setter', 'english_springer',
 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog',
 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer',
 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees',
 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter',
 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug',
 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki',
 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound',
 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky',
 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff',
 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound',
 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier',
 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier'] 

IMG_SIZE = 224

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# --- CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #87CEEB;
    }
    .title {
        color: #003366;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 1px 1px 2px #a4b0be;
    }
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .result-box {
        background: #70a1ff;
        border-radius: 15px;
        padding: 20px;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 20px rgba(112, 161, 255, 0.5);
        margin-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background: #2ed573 !important;
    }
    .clear-btn button {
        background-color: #ff4757;
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        margin-top: 1.5rem;
        width: 100%;
        box-shadow: 0 4px 12px rgba(255, 71, 87, 0.4);
        transition: background-color 0.3s ease;
    }
    .clear-btn button:hover {
        background-color: #ff6b81;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "prediction" not in st.session_state:
    st.session_state.prediction = None

st.markdown('<div class="title">🐕 Dog Breed Classifier 🐾</div>', unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model = load_model()

# Session state for uploaded file
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# File uploader
uploaded_file = st.file_uploader(
    "Choose a dog image...", 
    type=["jpg", "jpeg", "png"], 
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

# If image exists, show it & classify
if st.session_state.uploaded_file:
    image = Image.open(st.session_state.uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True, output_format="auto")
    
    with col2:
        with st.spinner("Classifying image..."):
            input_img = preprocess_image(image)
            prediction = model.predict(input_img)

        predicted_index = np.argmax(prediction[0])
        predicted_breed = dog_breeds[predicted_index].replace('_', ' ').title()
        confidence = float(prediction[0][predicted_index])
        confidence_percent = confidence * 100

        # Always show predicted breed
        st.markdown(
            f'<div class="result-box">🏷️ Predicted Breed: {predicted_breed}</div>',
            unsafe_allow_html=True
        )

        # Color logic based on confidence
        if confidence_percent >= 80:
            confidence_color = "#2ed573"  # green
            confidence_text = f"📊 Confidence: {confidence_percent:.2f}%"
        else:
            confidence_color = "#ff4757"  # red
            confidence_text = f"📊 Confidence: {confidence_percent:.2f}%<br>❌ Unable to predict"

        st.markdown(
            f'<div class="result-box" style="background:{confidence_color}; margin-top:0.5rem;">{confidence_text}</div>',
            unsafe_allow_html=True
        )
        st.progress(confidence)


    # Clear Image button
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("Clear Image"):
        st.session_state.uploaded_file = None
        st.session_state.prediction = None
        st.session_state.uploader_key += 1  # Force uploader to refresh
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
