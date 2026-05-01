import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Настройки приложения

st.set_page_config(
    page_title="Itten Color MVP",
    page_icon=None,
    layout="centered"
)
st.title("MVP: извлечение из изображений паттернов Иттена")
st.write(
    "Загрузите изображение. Приложение выделит 5 доминирующих цветов "
    "с помощью K-means и покажет таблицу HEX + доли цветов."
)

# 2. Вспомогательные функции

def rgb_to_hex(rgb):
    #Переводит цвет из RGB-формата в HEX.
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def prepare_image(image, max_size=300):
    # Подготавливает изображение для анализа.
    # Что делает:
    # 1. Переводит изображение в RGB.
    # 2. Уменьшает изображение, чтобы K-means работал быстрее.
    # 3. Превращает картинку в массив пикселей.

    image = image.convert("RGB")
    # Уменьшает изображение для скорости
    image.thumbnail((max_size, max_size))
    image_array = np.array(image)
    # Превращает изображение в таблицу пикселей
    pixels = image_array.reshape(-1, image_array.shape[-1])

    return image, pixels

def extract_dominant_colors(pixels, n_colors=5):

    # Выделяет n_colors доминирующих цветов с помощью K-means.
    # pixels — массив вида:
    # [
    #     [R, G, B],
    #     [R, G, B],
    #     ...
    # ]

    kmeans = KMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init="auto"
    )

    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    # Считает, сколько пикселей попало в каждый кластер
    counts = np.bincount(labels, minlength=n_colors)
    # Доля каждого цвета
    proportions = counts / counts.sum()
    # Округляет цвета до целых значений RGB
    colors = colors.astype(int)

    return colors, proportions

def make_palette_table(colors, proportions):
    # Создаёт таблицу с цветами: номер цвета, HEX, RGB, доля, процент

    rows = []

    for i, (color, proportion) in enumerate(zip(colors, proportions), start=1):
        rgb = color.tolist()
        hex_color = rgb_to_hex(rgb)

        rows.append({
            "№": i,
            "HEX": hex_color,
            "RGB": f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
            "Доля": round(float(proportion), 4),
            "Процент": round(float(proportion) * 100, 2)
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(by="Доля", ascending=False).reset_index(drop=True)

    df["№"] = range(1, len(df) + 1)
    return df

def plot_color_bar_chart(df):
    # Строит bar chart:
    # - по оси X — HEX-коды цветов;
    # - по оси Y — процент цвета в изображении;
    # - цвет каждого столбца соответствует самому цвету.

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        df["HEX"],
        df["Процент"],
        color=df["HEX"]
    )

    ax.set_xlabel("Цвет")
    ax.set_ylabel("Доля в изображении, %")
    ax.set_title("5 доминирующих цветов")

    for i, value in enumerate(df["Процент"]):
        ax.text(
            i,
            value + 0.5,
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=30)
    plt.tight_layout()

    return fig


def show_color_swatches(df):
    # Показывает цветовые плашки прямо в Streamlit.

    st.subheader("Палитра")
    df = df.sort_values(by="Доля", ascending=False)

    cols = st.columns(len(df))

    for col, (_, row) in zip(cols, df.iterrows()):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color: {row['HEX']};
                    height: 80px;
                    border-radius: 8px;
                    border: 1px solid #cccccc;
                "></div>
                <p style="text-align:center; font-size:14px;">
                    {row['HEX']}<br>
                    {row['Процент']}%
                </p>
                """,
                unsafe_allow_html=True
            )

# 3. Основной интерфейс

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Исходное изображение")
    st.image(image, use_container_width=True)

    prepared_image, pixels = prepare_image(image)

    colors, proportions = extract_dominant_colors(
        pixels=pixels,
        n_colors=5
    )

    df = make_palette_table(colors, proportions)

    show_color_swatches(df)

    st.subheader("Таблица HEX + доли")
    st.dataframe(df, use_container_width=True)

    st.subheader("Bar chart")
    fig = plot_color_bar_chart(df)
    st.pyplot(fig)

else:
    st.info("Пока изображение не загружено.")