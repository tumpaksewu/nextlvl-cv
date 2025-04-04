import streamlit as st
from streamlit_extras.let_it_rain import rain
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import models, transforms
import time
import requests
from io import BytesIO
import os


st.image("shipgif.gif", width=1000)

MODEL_PATH = "ships.pt"
model = YOLO (MODEL_PATH)

# Интерфейс для загрузки изображений
uploaded_files = st.file_uploader("📁 Загрузка изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_url = st.text_input("🔗 Или вставьте URL изображения")

images_to_classify = []

# Обработка ссылки
if image_url:
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Проверка на ошибки HTTP
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images_to_classify.append(("URL", image))
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения по ссылке: {e}")

    # Обработка загруженных файлов
if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        images_to_classify.append((file.name, image))

    # Классификация и отображение изображений
if images_to_classify:
    for name, image in images_to_classify:
        # Выполнение предсказания с помощью модели YOLO
        results = model(image)

        # Отображение аннотированного изображения
        st.subheader(f"⚓{name}")
        annotated_image = results[0].plot()  # Получение аннотированного изображения
        st.image(annotated_image, width=700)  # Отображение изображения в Streamlit

 