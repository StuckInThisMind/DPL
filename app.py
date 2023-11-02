import streamlit as st
import joblib
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

# Load your trained machine learning model
model_file = 'hb.pkl'
model = joblib.load(model_file)

# Create a Streamlit web app
st.title('Audio Classification Model Deployment')
st.sidebar.header('Upload an Audio File')

# Define file upload widget
audio_file = st.sidebar.file_uploader('Choose an audio file', type=['wav', 'mp3'])

# Mapping for numerical predictions to labels
label_mapping = {4: "normal", 3: "murmur", 0: "artifact", 2: "extrastole", 1: "extrahls"}

# Create a function to make predictions and map the labels
def predict(audio_file):
    if audio_file:
        # Load and process the uploaded audio file
        audio_data, sr = sf.read(audio_file)
        features = extract_audio_features(audio_data, sr)
        prediction = model.predict([features])
        mapped_prediction = label_mapping.get(prediction[0], "Unknown")
        return mapped_prediction
    return None

# Extract audio features
def extract_audio_features(audio_data, sr, low_pass_cutoff=195):
    x_filtered = butter_lowpass_filter(audio_data, low_pass_cutoff, sr)
    mfcc_features = np.mean(librosa.feature.mfcc(y=x_filtered, sr=sr, n_mfcc=128), axis=1)
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(x_filtered))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=x_filtered, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=x_filtered, sr=sr))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=x_filtered, sr=sr))
    feature_list = [*mfcc_features, zero_crossings, spectral_centroid, spectral_rolloff, chroma_stft]
    return feature_list

# Display the prediction
st.subheader('Prediction')
if st.button('Make Prediction'):
    prediction = predict(audio_file)
    if prediction is not None:
        st.write('The model predicts:', prediction)
