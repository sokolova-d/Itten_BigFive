import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2lab

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

def hue_distance(h1, h2):
    # Циклическое расстояние между оттенками hue.
    d = abs(h1 - h2)
    d = min(d, 1 - d)
    return d * 360

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

def select_diverse_colors(colors, proportions, n_final=5,
    min_hue_dist=20, min_rgb_dist=45, min_saturation=0.15):
    # Выбирает n_final цветов из списка кандидатов так,
    # чтобы они были более разнообразными.

    hsv_colors = rgb_to_hsv(colors[np.newaxis, :, :] / 255.0)[0]
    order = np.argsort(proportions)[::-1]
    selected_indices = []

    for idx in order:
        if len(selected_indices) == 0:
            selected_indices.append(idx)
            continue

        current_rgb = colors[idx]
        current_hsv = hsv_colors[idx]

        too_similar = False

        for selected_idx in selected_indices:
            selected_rgb = colors[selected_idx]
            selected_hsv = hsv_colors[selected_idx]

            rgb_dist = np.linalg.norm(current_rgb - selected_rgb)

            # Если оба цвета достаточно насыщенные, сравниваем по hue
            if current_hsv[1] >= min_saturation and selected_hsv[1] >= min_saturation:
                h_dist = hue_distance(current_hsv[0], selected_hsv[0])
                if h_dist < min_hue_dist:
                    too_similar = True
                    break
            else:
                if rgb_dist < min_rgb_dist:
                    too_similar = True
                    break

        if not too_similar:
            selected_indices.append(idx)

        if len(selected_indices) == n_final:
            break
    # добираем цвета по доле, если их не хватает
    if len(selected_indices) < n_final:
        for idx in order:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) == n_final:
                break

    return selected_indices

def extract_dominant_colors(pixels, n_colors=5, n_candidates=10):

    # Выделяет n_colors доминирующих цветов с помощью K-means.
    # pixels — массив вида:
    # [
    #     [R, G, B],
    #     [R, G, B],
    #     ...
    # ]

    kmeans = KMeans(
        n_clusters=n_candidates,
        random_state=42,
        n_init="auto"
    )

    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    # Считает, сколько пикселей попало в каждый кластер
    counts = np.bincount(labels, minlength=n_candidates)
    # Доля каждого цвета
    proportions = counts / counts.sum()

    # Выбираем более разнообразные цвета
    selected_indices = select_diverse_colors(
        colors=colors,
        proportions=proportions,
        n_final=n_colors,
        min_hue_dist=20,
        min_rgb_dist=45,
        min_saturation=0.15
    )

    final_colors = colors[selected_indices]
    final_proportions = proportions[selected_indices]

    order = np.argsort(final_proportions)[::-1]
    final_colors = final_colors[order]
    final_proportions = final_proportions[order]

    return final_colors, final_proportions

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

    coverage = df["Доля"].sum()
    st.write(f"Суммарное покрытие выбранных цветов: {coverage * 100:.2f}%")

    st.subheader("Bar chart")
    fig = plot_color_bar_chart(df)
    st.pyplot(fig)

else:
    st.info("Пока изображение не загружено.")