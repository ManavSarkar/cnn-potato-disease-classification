import streamlit as st
import tensorflow as tf
from keras import models, layers
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
model = load_model('potato_classification.h5')


st.title('Potato Disease Classification')

st.write('This is a simple web app to classify potato diseases using a Convolutional Neural Network (CNN).')

st.write('Upload an image of a potato leaf and the model will classify it as one of the following diseases: Early Blight, Late Blight, or Healthy.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif"])
# Define the disease classes
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Function to make predictions
def predict(image):
    # resize the image to 256, 256
    image = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # if img_array.shape[-1] == 4:
    #     img_array = img_array[..., :3]
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, confidence = predict(image)
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
    