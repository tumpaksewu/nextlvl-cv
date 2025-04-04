import streamlit as st
from streamlit_extras.let_it_rain import rain
from ultralytics import YOLO
from ultralytics import solutions
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import models, transforms
import time
import requests
from io import BytesIO
import os


# Custom navigation
page = st.sidebar.selectbox("Navigate", ["🦳", "👽"])


if page == "🦳":
    st.image("faceblurgif.gif", width=1000)

    MODEL_PATH = "models/faces.pt"
    model = YOLO(MODEL_PATH)

    # Init ObjectBlurrer with your 'yolo12' model
    blurrer = solutions.ObjectBlurrer(
        model=MODEL_PATH,
        blur_ratio=0.15,
        line_width=1,
        classes=[0],  # Assuming class '0' corresponds to faces in your model
    )

    # Интерфейс для загрузки изображений
    uploaded_files = st.file_uploader(
        "📁 Загрузка изображений",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    image_url = st.text_input("🔗 Или вставьте URL изображения")

    images_to_process = []

    # Обработка ссылки
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Проверка на ошибки HTTP
            image = Image.open(BytesIO(response.content)).convert("RGB")
            images_to_process.append(("URL", image))
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения по ссылке: {e}")

    # Обработка загруженных файлов
    STANDARD_SIZE = (640, 640)  # Set a standard size to resize images to
    STANDARD_CHANNELS = 3  # Ensure all images have 3 channels (BGR)

    if uploaded_files:
        for file in uploaded_files:
            try:
                image = Image.open(file).convert(
                    "RGB"
                )  # Convert to RGB to standardize format
                image = image.resize(STANDARD_SIZE)  # Resize to standard size
                open_cv_image = np.array(image)
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                if open_cv_image.shape[2] != STANDARD_CHANNELS:
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGRA2BGR)

                images_to_process.append((file.name, open_cv_image))
            except Exception as e:
                st.warning(f"Failed to process {file.name}: {e}")

    # Классификация и отображение изображений
    if images_to_process:
        for name, image in images_to_process:
            st.subheader(f"⚓ {name}")

            # Конвертация изображения из PIL в OpenCV формат
            open_cv_image = np.array(image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            # Выполнение размытия лиц с использованием ObjectBlurrer
            blurrer = solutions.ObjectBlurrer(
                model=MODEL_PATH,
                blur_ratio=0.15,
                line_width=1,
                classes=[0],
            )

            # Выполнение размытия лиц с использованием ObjectBlurrer
            results = blurrer(open_cv_image)

            # Получение обработанного изображения
            blurred_image = results.plot_im

            # Конвертация обратно в формат PIL для отображения
            blurred_image_pil = Image.fromarray(
                cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
            )

            # Отображение результата
            st.image(blurred_image_pil, width=700)

            # Сохранение и предоставление ссылки на скачивание
            output_path = f"{name}_face_blurred.jpg"
            blurred_image_pil.save(output_path)
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Скачать размытую картинку",
                    data=file,
                    file_name=output_path,
                    mime="image/jpeg",
                )


elif page == "👽":
    correct_pass = False

    st.markdown(
        """
        <style>
            /* Change the background color of the entire app */
            .stApp {
                background-color: black !important;
            }

            /* Change the font color for all text */
            body {
                color: green !important;
            }

            /* Optional: Style headers (h1, h2, etc.) */
            h1, h2, h3, h4, h5, h6, div {
                color: limegreen !important; /* Brighter green for headings */
            }

            /* Optional: Style buttons */
            .stButton > button {
                background-color: green !important;
                color: white !important;
                border-radius: 5px;
            }

            /* Optional: Style input widgets */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                background-color: white !important;
                color: green !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    def alien():
        rain(
            emoji="👨‍💻",
            font_size=50,
            falling_speed=6,
            animation_length="infinite",
        )

    def clown():
        rain(
            emoji="🤡",
            font_size=100,
            falling_speed=4,
            animation_length=15,
        )

    st.image("matrixgif.gif", width=1000)

    # Initialize session state variables
    if "show_password_input" not in st.session_state:
        st.session_state.show_password_input = False

    # Function to simulate password validation
    def check_password(password):
        # Replace "your_secure_password" with your actual password
        correct_password = "ValueError"
        return password == correct_password

    # Main app logic

    st.title("ENTER THE MATRIX")

    # Button to trigger password input
    if st.button("Я ГОТОВ К ПРАВДЕ"):
        st.session_state.show_password_input = True

    # Show password input only if the button has been pressed
    if st.session_state.show_password_input:
        password = st.text_input(
            "**ВВЕДИ ПАРОЛЬ ДЛЯ ИЗБРАННЫХ И ВСТРЕТЬСЯ С ИЗБРАННЫМ ЛИЦОМ К ЛИЦУ**",
            type="password",
        )

        # Check if the user has entered a password
        if password:
            if check_password(password):
                st.success("Access Granted!")
                # Add your protected content here
                st.markdown(
                    "<p style='font-size: 40px;'>👨‍💻Welcome to the Matrix!</p>",
                    unsafe_allow_html=True,
                )
                st.write("This is some secret content.")
                alien()
                correct_pass = True
            else:
                st.error("ТЫ НЕ ДОСТОИН.")
                clown()
                correct_pass = False

    if correct_pass == True:
        MODEL_PATH = "models/faces.pt"
        model = YOLO(MODEL_PATH)

        # Load the mask image
        mask = cv2.imread(
            "tim.png", cv2.IMREAD_UNCHANGED
        )  # Load with alpha channel if present
        if mask is None:
            st.error("The mask image 'tim.png' could not be found or loaded.")
            st.stop()

        # Интерфейс для загрузки изображений
        uploaded_files = st.file_uploader(
            "📁 Загрузка изображений",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        image_url = st.text_input("🔗 Или вставьте URL изображения")

        images_to_process = []

        # Обработка ссылки
        if image_url:
            try:
                response = requests.get(image_url)
                response.raise_for_status()  # Проверка на ошибки HTTP
                image = Image.open(BytesIO(response.content)).convert("RGB")
                images_to_process.append(("URL", image))
            except Exception as e:
                st.error(f"Ошибка при загрузке изображения по ссылке: {e}")

        # Обработка загруженных файлов
        STANDARD_SIZE = (640, 640)  # Set a standard size to resize images to
        STANDARD_CHANNELS = 3  # Ensure all images have 3 channels (BGR)

        images_to_process = []

        if uploaded_files:
            for file in uploaded_files:
                try:
                    image = Image.open(file).convert(
                        "RGB"
                    )  # Convert to RGB to standardize format
                    image = image.resize(STANDARD_SIZE)  # Resize to standard size
                    open_cv_image = np.array(image)
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                    if open_cv_image.shape[2] != STANDARD_CHANNELS:
                        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGRA2BGR)

                    images_to_process.append((file.name, open_cv_image))
                except Exception as e:
                    st.warning(f"Failed to process {file.name}: {e}")

        # Классификация и отображение изображений
        if images_to_process:
            for name, image in images_to_process:
                st.subheader(f"⚓ {name}")

                # Конвертация изображения из PIL в OpenCV формат
                open_cv_image = np.array(image)
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                # Выполнение предсказания с помощью модели YOLO
                results = model(open_cv_image)[0]

                # Получение размеров маски
                mask_h, mask_w = mask.shape[:2]

                # Обработка каждой обнаруженной области (лиц)
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox_w, bbox_h = x2 - x1, y2 - y1

                    # Масштабирование маски для соответствия размеру обнаруженной области
                    scale_factor = min(bbox_w / mask_w, bbox_h / mask_h) * 2.2
                    new_size = (int(mask_w * scale_factor), int(mask_h * scale_factor))
                    resized_mask = cv2.resize(
                        mask, new_size, interpolation=cv2.INTER_AREA
                    )

                    rm_h, rm_w = resized_mask.shape[:2]
                    center_x, center_y = x1 + bbox_w // 2, y1 + bbox_h // 2
                    top_left_x = center_x - rm_w // 2
                    top_left_y = center_y - rm_h // 2

                    # Предотвращение выхода маски за пределы изображения
                    top_left_x = max(0, top_left_x)
                    top_left_y = max(0, top_left_y)
                    bottom_right_x = min(open_cv_image.shape[1], top_left_x + rm_w)
                    bottom_right_y = min(open_cv_image.shape[0], top_left_y + rm_h)
                    resized_mask = resized_mask[
                        : bottom_right_y - top_left_y, : bottom_right_x - top_left_x
                    ]

                    # Обработка альфа-канала (если он существует)
                    if resized_mask.shape[2] == 4:
                        alpha_mask = resized_mask[:, :, 3] / 255.0
                        mask_rgb = resized_mask[:, :, :3]
                    else:
                        alpha_mask = np.ones((rm_h, rm_w))
                        mask_rgb = resized_mask

                    roi = open_cv_image[
                        top_left_y:bottom_right_y, top_left_x:bottom_right_x
                    ]

                    # Наложение маски на изображение
                    for c in range(3):  # Для каждого канала (B, G, R)
                        roi[:, :, c] = (
                            alpha_mask * mask_rgb[:, :, c]
                            + (1 - alpha_mask) * roi[:, :, c]
                        ).astype(np.uint8)

                    open_cv_image[
                        top_left_y:bottom_right_y, top_left_x:bottom_right_x
                    ] = roi

                # Конвертация обратно в формат PIL для отображения
                output_image = Image.fromarray(
                    cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
                )
                st.image(output_image, width=700)

                # Сохранение и предоставление ссылки на скачивание
                output_path = f"{name}_modified.jpg"
                output_image.save(output_path)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 Скачать изменённое изображение",
                        data=file,
                        file_name=output_path,
                        mime="image/jpeg",
                    )
