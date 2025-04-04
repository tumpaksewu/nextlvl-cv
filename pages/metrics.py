import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
from ultralytics import YOLO


plt.style.use("cyberpunk")

page = st.sidebar.selectbox("Navigate", ["🦳 FACES", "⚓ BOATS", "🌳 FOREST"])

if page == "⚓ BOATS":
    # Загрузка модели YOLO12
    @st.cache_resource
    def load_model():
        model_path = "models/boats.pt"

        # Проверяем, существует ли файл
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            st.stop()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_b = YOLO("models/boats.pt")
        model_b.to(device)
        return model_b

    model_b = load_model()

    # Чтение данных из results.csv
    @st.cache_data
    def load_results():
        results_path = "models/boats.csv"

        # Проверяем, существует ли файл
        if not os.path.exists(results_path):
            st.error(f"Файл результатов не найден: {results_path}")
            st.stop()

        results = pd.read_csv(results_path)
        return results

    results = load_results()

    # Построение графиков метрик
    def plot_metrics(results):
        fig, ax = plt.subplots(4, 1, figsize=(10, 20))

        # График Loss
        ax[0].plot(
            results["epoch"],
            results["train/box_loss"],
            label="Train Box Loss",
            color="red",
        )
        ax[0].plot(
            results["epoch"],
            results["val/box_loss"],
            label="Validation Box Loss",
            color="orange",
        )
        ax[0].set_title("Box Loss over Epochs")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # График Precision и Recall
        ax[1].plot(
            results["epoch"],
            results["metrics/precision(B)"],
            label="Precision",
            color="blue",
        )
        ax[1].plot(
            results["epoch"],
            results["metrics/recall(B)"],
            label="Recall",
            color="green",
        )
        ax[1].set_title("Precision and Recall over Epochs")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Value")
        ax[1].legend()

        # График mAP
        ax[2].plot(
            results["epoch"], results["metrics/mAP50(B)"], label="mAP50", color="purple"
        )
        ax[2].plot(
            results["epoch"],
            results["metrics/mAP50-95(B)"],
            label="mAP50-95",
            color="brown",
        )
        ax[2].set_title("mAP over Epochs")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("mAP")
        ax[2].legend()

        # График Learning Rates
        ax[3].plot(
            results["epoch"],
            results["lr/pg0"],
            label="Learning Rate (pg0)",
            color="teal",
        )
        ax[3].plot(
            results["epoch"],
            results["lr/pg1"],
            label="Learning Rate (pg1)",
            color="pink",
        )
        ax[3].plot(
            results["epoch"],
            results["lr/pg2"],
            label="Learning Rate (pg2)",
            color="gray",
        )
        ax[3].set_title("Learning Rates over Epochs")
        ax[3].set_xlabel("Epoch")
        ax[3].set_ylabel("Learning Rate")
        ax[3].legend()

        plt.tight_layout()
        mplcyberpunk.add_glow_effects(ax[0], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[1], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[2], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[3], gradient_fill=True)
        return fig

    # Расчет метрик для загруженного изображения
    def calculate_image_metrics(image, model_b):
        with torch.no_grad():
            # Получаем результаты инференса
            results = model_b(image)

            # Извлекаем bounding boxes и количество объектов
            boxes = results[0].boxes
            num_objects = len(boxes)

            # Извлекаем mAP50 и mAP50-95 из результатов
            if hasattr(results[0], "mAP50"):
                mAP50 = results[0].mAP50  # mAP для IoU порога 0.5
                mAP50_95 = results[0].mAP50_95  # mAP для диапазона IoU [0.5:0.95]
            else:
                mAP50, mAP50_95 = 0.0, 0.0

            # Преобразование bounding boxes в формат для отображения
            plotted_image = results[
                0
            ].plot()  # Встроенный метод для отображения bounding boxes

            return num_objects, mAP50, mAP50_95, plotted_image

    # Интерфейс Streamlit
    st.title("Metrics Dashboard")

    # Отображение графиков
    st.header("Training Metrics")
    fig = plot_metrics(results)
    st.pyplot(fig)

    # Загрузка изображений или URL
    st.header("Ищем спасение в море")
    uploaded_file = st.file_uploader(
        "Загрузи изображение...", type=["jpg", "jpeg", "png"]
    )
    image_url = st.text_input("URL:")

    if uploaded_file is not None:
        image_b = Image.open(uploaded_file).convert("RGB")
    elif image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_b = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Ошибка URL: {e}")
            image_b = None
    else:
        image_b = None

    # Обработка изображения
    if image_b is not None:
        st.image(image_b, caption="Загруженное изображение", width=500)
        st.subheader("Найдено спасение")
        num_objects, mAP50, mAP50_95, plotted_image = calculate_image_metrics(
            image_b, model_b
        )
        st.write(f"Кол-во кораблей для эвакуации от сталактита: {num_objects}")

        # Отображение изображения с bounding boxes
        st.image(plotted_image, caption="Image with Bounding Boxes", width=500)


elif page == "🦳 FACES":
    # Загрузка модели YOLO12
    @st.cache_resource
    def load_model():
        model_path = "models/faces.pt"

        # Проверяем, существует ли файл
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            st.stop()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_f = YOLO("models/faces.pt")
        model_f.to(device)
        return model_f

    model_f = load_model()

    # Чтение данных из results.csv
    @st.cache_data
    def load_results():
        results_path = "models/faces.csv"

        # Проверяем, существует ли файл
        if not os.path.exists(results_path):
            st.error(f"Файл результатов не найден: {results_path}")
            st.stop()

        results = pd.read_csv(results_path)
        return results

    results = load_results()

    # Построение графиков метрик
    def plot_metrics(results):
        fig, ax = plt.subplots(4, 1, figsize=(10, 20))

        # График Loss
        ax[0].plot(
            results["epoch"],
            results["train/box_loss"],
            label="Train Box Loss",
            color="red",
        )
        ax[0].plot(
            results["epoch"],
            results["val/box_loss"],
            label="Validation Box Loss",
            color="orange",
        )
        ax[0].set_title("Box Loss over Epochs")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # График Precision и Recall
        ax[1].plot(
            results["epoch"],
            results["metrics/precision(B)"],
            label="Precision",
            color="blue",
        )
        ax[1].plot(
            results["epoch"],
            results["metrics/recall(B)"],
            label="Recall",
            color="green",
        )
        ax[1].set_title("Precision and Recall over Epochs")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Value")
        ax[1].legend()

        # График mAP
        ax[2].plot(
            results["epoch"], results["metrics/mAP50(B)"], label="mAP50", color="purple"
        )
        ax[2].plot(
            results["epoch"],
            results["metrics/mAP50-95(B)"],
            label="mAP50-95",
            color="brown",
        )
        ax[2].set_title("mAP over Epochs")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("mAP")
        ax[2].legend()

        # График Learning Rates
        ax[3].plot(
            results["epoch"],
            results["lr/pg0"],
            label="Learning Rate (pg0)",
            color="teal",
        )
        ax[3].plot(
            results["epoch"],
            results["lr/pg1"],
            label="Learning Rate (pg1)",
            color="pink",
        )
        ax[3].plot(
            results["epoch"],
            results["lr/pg2"],
            label="Learning Rate (pg2)",
            color="gray",
        )
        ax[3].set_title("Learning Rates over Epochs")
        ax[3].set_xlabel("Epoch")
        ax[3].set_ylabel("Learning Rate")
        ax[3].legend()

        plt.tight_layout()
        mplcyberpunk.add_glow_effects(ax[0], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[1], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[2], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[3], gradient_fill=True)
        return fig

    # Расчет метрик для загруженного изображения
    def calculate_image_metrics(image, model):
        with torch.no_grad():
            # Получаем результаты инференса
            results = model_f(image)

            # Извлекаем bounding boxes и количество объектов
            boxes = results[0].boxes
            num_objects = len(boxes)

            # Извлекаем mAP50 и mAP50-95 из результатов
            if hasattr(results[0], "mAP50"):
                mAP50 = results[0].mAP50  # mAP для IoU порога 0.5
                mAP50_95 = results[0].mAP50_95  # mAP для диапазона IoU [0.5:0.95]
            else:
                mAP50, mAP50_95 = 0.0, 0.0

            # Преобразование bounding boxes в формат для отображения
            plotted_image = results[
                0
            ].plot()  # Встроенный метод для отображения bounding boxes

            return num_objects, mAP50, mAP50_95, plotted_image

    # Интерфейс Streamlit
    st.title("Metrics Dashboard")

    # Отображение графиков
    st.header("Training Metrics")
    fig = plot_metrics(results)
    st.pyplot(fig)

    # Загрузка изображений или URL
    st.header("Загрузи изображение")
    uploaded_file = st.file_uploader(
        "Загрузи изображение...", type=["jpg", "jpeg", "png"]
    )
    image_url = st.text_input("URL:")

    if uploaded_file is not None:
        image_f = Image.open(uploaded_file).convert("RGB")
    elif image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_f = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Ошибка URL: {e}")
            image_f = None
    else:
        image_f = None

    # Обработка изображения
    if image_f is not None:
        st.image(image_f, caption="Подозревамый пострадавший", width=500)
        st.subheader("Результат заражения")
        num_objects, mAP50, mAP50_95, plotted_image = calculate_image_metrics(
            image_f, model_f
        )
        st.write(f"Пострадавших от сталактита: {num_objects}")

        # Отображение изображения с bounding boxes
        st.image(plotted_image, caption="Image with Bounding Boxes", width=500)

elif page == "🌳 FOREST":
    # Загрузка модели YOLO12
    @st.cache_resource
    def load_model():
        model_path = "models/forest.pt"

        # Проверяем, существует ли файл
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            st.stop()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_fo = YOLO("models/forest.pt")
        model_fo.to(device)
        return model_fo

    model_fo = load_model()

    # Чтение данных из results.csv
    @st.cache_data
    def load_results():
        results_path = "models/forest.csv"

        # Проверяем, существует ли файл
        if not os.path.exists(results_path):
            st.error(f"Файл результатов не найден: {results_path}")
            st.stop()

        results = pd.read_csv(results_path)
        return results

    results = load_results()

    # Построение графиков метрик
    def plot_metrics(results):
        fig, ax = plt.subplots(4, 1, figsize=(10, 20))

        # График Loss
        ax[0].plot(
            results["epoch"],
            results["train/box_loss"],
            label="Train Box Loss",
            color="red",
        )
        ax[0].plot(
            results["epoch"],
            results["val/box_loss"],
            label="Validation Box Loss",
            color="orange",
        )
        ax[0].set_title("Box Loss over Epochs")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # График Precision и Recall
        ax[1].plot(
            results["epoch"],
            results["metrics/precision(B)"],
            label="Precision",
            color="blue",
        )
        ax[1].plot(
            results["epoch"],
            results["metrics/recall(B)"],
            label="Recall",
            color="green",
        )
        ax[1].set_title("Precision and Recall over Epochs")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Value")
        ax[1].legend()

        # График mAP
        ax[2].plot(
            results["epoch"], results["metrics/mAP50(B)"], label="mAP50", color="purple"
        )
        ax[2].plot(
            results["epoch"],
            results["metrics/mAP50-95(B)"],
            label="mAP50-95",
            color="brown",
        )
        ax[2].set_title("mAP over Epochs")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("mAP")
        ax[2].legend()

        # График Learning Rates
        ax[3].plot(
            results["epoch"],
            results["lr/pg0"],
            label="Learning Rate (pg0)",
            color="teal",
        )
        ax[3].plot(
            results["epoch"],
            results["lr/pg1"],
            label="Learning Rate (pg1)",
            color="pink",
        )
        ax[3].plot(
            results["epoch"],
            results["lr/pg2"],
            label="Learning Rate (pg2)",
            color="gray",
        )
        ax[3].set_title("Learning Rates over Epochs")
        ax[3].set_xlabel("Epoch")
        ax[3].set_ylabel("Learning Rate")
        ax[3].legend()
        plt.tight_layout()
        mplcyberpunk.add_glow_effects(ax[0], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[1], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[2], gradient_fill=True)
        mplcyberpunk.add_glow_effects(ax[3], gradient_fill=True)
        return fig

    # Расчет метрик для загруженного изображения
    def calculate_image_metrics(image, model_fo):
        with torch.no_grad():
            # Получаем результаты инференса
            results = model_fo(image_fo)

            # Извлекаем bounding boxes и количество объектов
            boxes = results[0].boxes
            num_objects = len(boxes)

            # Извлекаем mAP50 и mAP50-95 из результатов
            if hasattr(results[0], "mAP50"):
                mAP50 = results[0].mAP50  # mAP для IoU порога 0.5
                mAP50_95 = results[0].mAP50_95  # mAP для диапазона IoU [0.5:0.95]
            else:
                mAP50, mAP50_95 = 0.0, 0.0

            # Преобразование bounding boxes в формат для отображения
            plotted_image = results[
                0
            ].plot()  # Встроенный метод для отображения bounding boxes

            return num_objects, mAP50, mAP50_95, plotted_image

    # Интерфейс Streamlit
    st.title("Metrics Dashboard")

    # Отображение графиков
    st.header("Training Metrics")
    fig = plot_metrics(results)
    st.pyplot(fig)
    mplcyberpunk.add_glow_effects()

    # Загрузка изображений или URL
    st.header("Загрузи изображение")
    uploaded_file = st.file_uploader(
        "Загрузи изображение...", type=["jpg", "jpeg", "png"]
    )
    image_url = st.text_input("URL:")

    if uploaded_file is not None:
        image_fo = Image.open(uploaded_file).convert("RGB")
    elif image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_fo = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Ошибка URL: {e}")
            image_fo = None
    else:
        image_fo = None

    # Обработка изображения
    if image_fo is not None:
        st.image(image_fo, caption="Поиск локации свежего воздуха", width=500)
        st.subheader("Зона безопасного дыхания")
        num_objects, mAP50, mAP50_95, plotted_image = calculate_image_metrics(
            image_fo, model_fo
        )
        st.write(f"Кол-во безопасных локаций: {num_objects}")

        # Отображение изображения с bounding boxes
        st.image(plotted_image, caption="Image with Bounding Boxes", width=500)
