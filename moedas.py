import streamlit as st
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

# Verificar versões
st.write("TensorFlow version:", tf.__version__)
st.write("Keras version:", tf.keras.__version__)

# Função para carregar o modelo
@st.cache_resource
def load_keras_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        st.write("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função para carregar rótulos
def load_labels(label_path):
    try:
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except Exception as e:
        st.error(f"Erro ao carregar os rótulos: {e}")
        return []

# Função para processar a imagem
def preprocess_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# Função principal para exibir o vídeo e previsões
def main():
    st.title("Detecção de Moeda com Modelo Keras")

    # Carregar o modelo e rótulos
    model_path = "keras_model.h5"
    label_path = "labels.txt"
    model = load_keras_model(model_path)
    labels = load_labels(label_path)

    if model is None or not labels:
        st.stop()

    # Configurar a câmera
    camera = cv2.VideoCapture(0)

    stframe = st.empty()

    while True:
        # Capturar imagem da câmera
        ret, image = camera.read()

        if not ret:
            st.error("Não foi possível acessar a câmera.")
            break

        # Exibir a imagem da câmera
        stframe.image(image, channels="BGR", use_column_width=True)

        # Processar e fazer previsões
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = labels[index]
        confidence_score = prediction[0][index]

        # Exibir a previsão e a pontuação de confiança
        st.write(f"Classe: {class_name}")
        st.write(f"Pontuação de Confiança: {confidence_score * 100:.2f}%")

        # Atualizar a imagem a cada 1 segundo
        cv2.waitKey(1000)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
