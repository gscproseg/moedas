import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# Define a function to make predictions
def predict_class(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

st.title("Classificador de Imagens em Tempo Real")

# Use Streamlit's camera input
camera = st.camera_input("Capture uma imagem")

if camera:
    # Read the image from the camera
    image = np.array(camera)
    st.image(image, channels="BGR", caption="Imagem Capturada")

    # Make a prediction
    class_name, confidence_score = predict_class(image)

    # Display the result
    st.write(f"Classificação: {class_name}")
    st.write(f"Confiança: {confidence_score:.2f}")

    # Optionally, you can add a button to capture the image
    if st.button('Capturar Imagem'):
        st.write("Imagem Capturada")
        # Save or process the captured image if needed

else:
    st.text("Nenhuma imagem capturada")
