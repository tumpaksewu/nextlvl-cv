import streamlit as st
import time
import torch
import requests
import numpy as np
import cv2

# from's
from io import BytesIO
from PIL import Image
from ultralytics import YOLO


# Отображение заголовка и изображения
st.image("forestgif.gif", width=1000)


MODEL_PATH = "models/forest.pt"
model = YOLO(MODEL_PATH)

# Интерфейс для загрузки изображений
uploaded_files = st.file_uploader(
    "📁 Загрузка изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
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


# model = load_model(MODEL_PATH)

# # Интерфейс для загрузки изображений
# uploaded_files = st.file_uploader("📁 Загрузка изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
# image_url = st.text_input("🔗 Или вставьте URL изображения")

# images_to_process = []

# # Обработка ссылки
# if image_url:
#     try:
#         response = requests.get(image_url)
#         response.raise_for_status()  # Проверка на ошибки HTTP
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#         images_to_process.append(("URL", image))
#     except Exception as e:
#         st.error(f"Ошибка при загрузке изображения по ссылке: {e}")

# # Обработка загруженных файлов
# if uploaded_files:
#     for file in uploaded_files:
#         image = Image.open(file).convert("RGB")
#         images_to_process.append((file.name, image))

# # Обработка изображений
# if images_to_process:
#     for name, image in images_to_process:
#         st.subheader(f"⚓ {name}")

#         # Преобразование изображения в тензор
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),  # Приводим изображение к размеру, на котором обучалась модель
#             transforms.ToTensor(),         # Преобразуем в тензор
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
#         ])
#         input_tensor = transform(image).unsqueeze(0).to(device)  # Добавляем батч-измерение

#         # Предсказание маски сегментации
#         with torch.no_grad():
#             output = model(input_tensor)
#             mask = torch.sigmoid(output)  # Применяем сигмоиду для бинарной сегментации
#             mask = (mask > 0.5).float()   # Превращаем вероятности в бинарную маску

#         # Отображение исходного изображения
#         st.image(image, caption="Исходное изображение", width=700)

#         # Отображение маски сегментации
#         mask_np = mask.squeeze().cpu().numpy()
#         st.image(mask_np, caption="Маска сегментации", width=700, clamp=True)

#         # Наложение маски на исходное изображение
#         overlay = np.array(image.resize((256, 256)))  # Масштабируем изображение до размера маски
#         overlay[mask_np > 0] = [255, 0, 0]  # Красный цвет для сегментированных областей
#         st.image(overlay, caption="Наложение маски", width=700)
