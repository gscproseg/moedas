import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import os

# Caminho dos arquivos
model_path = "./keras_Model.h5"
labels_path = "./labels.txt"

# Verifique se o arquivo do modelo e o arquivo de labels existem
if not os.path.isfile(model_path):
    st.error(f"Arquivo do modelo não encontrado: {model_path}")
if not os.path.isfile(labels_path):
    st.error(f"Arquivo de labels não encontrado: {labels_path}")

# Carregar o modelo
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

# Carregar as labels
try:
    with open(labels_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
except Exception as e:
    st.error(f"Erro ao carregar o arquivo de labels: {e}")

# Função para preprocessar a imagem
def preprocess_image(image):
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    image = np.array(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# Função para fazer previsões
def predict_class(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

st.title("Classificador de Imagens em Tempo Real")

# Usar a entrada da câmera do Streamlit
camera = st.camera_input("Capture uma imagem")

if camera:
    # Ler a imagem da câmera
    image = Image.open(camera)
    st.image(image, caption="Imagem Capturada")

    # Fazer um
