import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2lab

st.set_page_config(
    page_title="Itten Color MVP",
    page_icon=None,
    layout="centered"
)
st.title("MVP: извлечение из изображений паттернов Иттена")
st.write(
    "Загрузите изображение. Приложение выделит 5 доминирующих цветов "
    "с помощью K-means и покажет таблицу HEX + доли цветов + HSV + CIE Lab."
)

def rgb_to_hex(rgb):
    #Переводит цвет из RGB-формата в HEX.
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def hue_distance(h1, h2):
    # Циклическое расстояние между оттенками hue.
    d = abs(h1 - h2)
    d = min(d, 1 - d)
    return d * 360

def hue_distance_degrees(h1, h2):
    # Циклическое расстояние между hue в градусах.
    d = abs(h1 - h2)
    return min(d, 360 - d)

def distance_to_template(hue, template_angles, rotation):
    # Расстояние от hue до ближайшего угла шаблона.
    distances = []
    for angle in template_angles:
        rotated_angle = (angle + rotation) % 360
        distances.append(hue_distance_degrees(hue, rotated_angle))

    return min(distances)

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
    candidate_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    # Считает, сколько пикселей попало в каждый кластер
    counts = np.bincount(labels, minlength=n_candidates)
    # Доля каждого цвета
    candidate_proportions = counts / counts.sum()

    # Выбираем более разнообразные цвета
    selected_indices = select_diverse_colors(
        colors=candidate_colors,
        proportions=candidate_proportions,
        n_final=n_colors,
        min_hue_dist=20,
        min_rgb_dist=45,
        min_saturation=0.15
    )

    palette_colors = candidate_colors[selected_indices]
    palette_proportions = candidate_proportions[selected_indices]
    # Для 5 цветов
    palette_order = np.argsort(palette_proportions)[::-1]
    palette_colors = palette_colors[palette_order]
    palette_proportions = palette_proportions[palette_order]
    # Для 10 цветов
    candidate_order = np.argsort(candidate_proportions)[::-1]
    candidate_colors = candidate_colors[candidate_order]
    candidate_proportions = candidate_proportions[candidate_order]

    return palette_colors, palette_proportions, candidate_colors, candidate_proportions

def analyze_itten_contrasts_hsv_lab(df):
    # Rule-based анализ контрастов Иттена.

    hues = df["Hue"].to_numpy()
    saturations = df["Saturation"].to_numpy()
    values = df["Value"].to_numpy()

    l_values = df["L*"].to_numpy()
    chroma_values = df["Chroma"].to_numpy()

    proportions = df["Доля"].to_numpy()
    hex_colors = df["HEX"].to_numpy()

    results = {}

    # Берём только достаточно насыщенные цвета для hue-анализа
    saturated_mask = saturations >= 0.2

    saturated_hues = hues[saturated_mask]
    saturated_props = proportions[saturated_mask]
    saturated_sats = saturations[saturated_mask]
    saturated_hex = hex_colors[saturated_mask]
    saturated_chroma = chroma_values[saturated_mask]

    # Hue contrast
    if len(saturated_hues) >= 2:
        hue_distances = []

        for i in range(len(saturated_hues)):
            for j in range(i + 1, len(saturated_hues)):
                hue_distances.append(
                    hue_distance_degrees(saturated_hues[i], saturated_hues[j])
                )

        hue_range = max(hue_distances)

        if hue_range < 30:
            level = "слабый"
        elif hue_range < 90:
            level = "умеренный"
        elif hue_range < 150:
            level = "выраженный"
        else:
            level = "сильный"

        results["Контраст цветового тона"] = (
            f"{level}: максимальное расстояние между насыщенными оттенками {hue_range:.1f}°."
        )
    else:
        results["Контраст цветового тона"] = (
            "Не определяется: в палитре недостаточно насыщенных цветов."
        )

    # Light-dark
    light_dark_range = l_values.max() - l_values.min()

    if light_dark_range < 15:
        level = "слабый"
    elif light_dark_range < 35:
        level = "умеренный"
    else:
        level = "сильный"

    results["Светло-тёмный контраст"] = (
        f"{level}: разброс светлоты L* = {light_dark_range:.1f}."
    )

    # Warm-cool
    warm_weight = 0
    cool_weight = 0

    for h, s, p in zip(hues, saturations, proportions):
        weight = p * s

        if (0 <= h <= 70) or (330 <= h <= 360):
            warm_weight += weight
        elif 160 <= h <= 280:
            cool_weight += weight

    warm_cool_score = min(warm_weight, cool_weight)

    if warm_weight > 0 and cool_weight > 0:
        results["Тепло-холодный контраст"] = (
            f"есть: тёплый вклад = {warm_weight:.3f}, холодный вклад = {cool_weight:.3f}."
        )
    else:
        results["Тепло-холодный контраст"] = (
            f"слабый: тёплый вклад = {warm_weight:.3f}, холодный вклад = {cool_weight:.3f}."
        )

    # Saturation
    saturation_range = saturations.max() - saturations.min()
    chroma_range = chroma_values.max() - chroma_values.min()

    if chroma_range < 20:
        level = "слабый"
    elif chroma_range < 45:
        level = "умеренный"
    else:
        level = "сильный"

    results["Контраст насыщенности / chroma"] = (
        f"{level}: разброс HSV Saturation = {saturation_range:.3f}, "
        f"разброс Lab Chroma = {chroma_range:.1f}."
    )

    # Complementary
    best_pair = None
    best_distance = None
    best_score = 0

    if len(saturated_hues) >= 2:
        for i in range(len(saturated_hues)):
            for j in range(i + 1, len(saturated_hues)):
                d = hue_distance_degrees(saturated_hues[i], saturated_hues[j])

                closeness = 1 - abs(d - 180) / 180

                chroma_weight_i = 1 + saturated_chroma[i] / 100
                chroma_weight_j = 1 + saturated_chroma[j] / 100

                score = (
                    closeness
                    * saturated_props[i]
                    * saturated_props[j]
                    * saturated_sats[i]
                    * saturated_sats[j]
                    * chroma_weight_i
                    * chroma_weight_j
                )
                if score > best_score:
                    best_score = score
                    best_distance = d
                    best_pair = (saturated_hex[i], saturated_hex[j])

    if best_pair is not None and best_distance >= 150:
        results["Дополнительный контраст"] = (
            f"найдена близкая к дополнительной пара: "
            f"{best_pair[0]} и {best_pair[1]}, расстояние {best_distance:.1f}°."
        )
    else:
        results["Дополнительный контраст"] = (
            "явная дополнительная пара не найдена."
        )

    # Extension
    visual_weights = proportions * (1 + chroma_values / 100)

    if visual_weights.sum() > 0:
        q = visual_weights / visual_weights.sum()
        entropy = -np.sum(q * np.log(q + 1e-9))
        entropy_norm = entropy / np.log(len(q))
        extension_score = 1 - entropy_norm
    else:
        extension_score = 0

    if extension_score < 0.2:
        level = "слабый"
    elif extension_score < 0.45:
        level = "умеренный"
    else:
        level = "сильный"

    dominant_color = df.loc[df["Доля"].idxmax(), "HEX"]
    dominant_percent = df.loc[df["Доля"].idxmax(), "Процент"]

    results["Контраст распространения"] = (
        f"{level}: доминирующий цвет {dominant_color} занимает {dominant_percent}% изображения. "
        f"Оценка с учётом chroma = {extension_score:.3f}."
    )

    return results

def fit_harmony_scheme_hsv_lab(df):
    # Подбирает цветовую схему по HSV Hue.
    schemes = {
        "Монохромная": [0],
        "Аналоговая": [-30, 0, 30],
        "Комплиментарная": [0, 180],
        "Триада": [0, 120, 240],
        "Тетрада": [0, 90, 180, 270],
        "Раздельно-комплиментарная": [0, 150, 210]
    }

    df_sat = df[df["Saturation"] >= 0.2].copy()

    if len(df_sat) == 0:
        return {
            "scheme": "Не определяется",
            "description": "В палитре нет достаточно насыщенных цветов для анализа схемы."
        }

    hues = df_sat["Hue"].to_numpy()
    proportions = df_sat["Доля"].to_numpy()
    saturations = df_sat["Saturation"].to_numpy()
    chroma_values = df_sat["Chroma"].to_numpy()

    weights = proportions * saturations * (1 + chroma_values / 100)

    if weights.sum() > 0:
        weights = weights / weights.sum()

    best_scheme = None
    best_score = float("inf")
    best_rotation = None

    for scheme_name, template_angles in schemes.items():
        for rotation in range(360):
            error = 0

            for h, w in zip(hues, weights):
                error += w * distance_to_template(h, template_angles, rotation)

            if error < best_score:
                best_score = error
                best_scheme = scheme_name
                best_rotation = rotation

    if best_score < 15:
        confidence = "высокая"
    elif best_score < 35:
        confidence = "средняя"
    else:
        confidence = "низкая"

    return {
        "scheme": best_scheme,
        "score": round(float(best_score), 2),
        "rotation": best_rotation,
        "confidence": confidence,
        "description": (
            f"Ближайшая цветовая схема: {best_scheme}. "
            f"Средняя ошибка fitting = {best_score:.1f}°. "
            f"Уверенность: {confidence}."
        )
    }

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

def add_color_space_columns(df):
    # Добавляет к таблице палитры признаки HSV и CIE Lab.

    rgb_values = []

    for rgb_text in df["RGB"]:
        numbers = rgb_text.replace("rgb(", "").replace(")", "").split(",")
        rgb = [int(x.strip()) for x in numbers]
        rgb_values.append(rgb)

    rgb_array = np.array(rgb_values) / 255.0

    hsv_array = rgb_to_hsv(rgb_array[np.newaxis, :, :])[0]
    lab_array = rgb2lab(rgb_array[np.newaxis, :, :])[0]

    l_values = lab_array[:, 0]
    a_values = lab_array[:, 1]
    b_values = lab_array[:, 2]

    chroma_values = np.sqrt(a_values ** 2 + b_values ** 2)

    df = df.copy()

    df["Hue"] = np.round(hsv_array[:, 0] * 360, 2)
    df["Saturation"] = np.round(hsv_array[:, 1], 3)
    df["Value"] = np.round(hsv_array[:, 2], 3)

    df["L*"] = np.round(l_values, 2)
    df["a*"] = np.round(a_values, 2)
    df["b*"] = np.round(b_values, 2)
    df["Chroma"] = np.round(chroma_values, 2)

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
                <p style="text-align:center; font-size:12px;">
                    {row['HEX']}<br>
                    {row['Процент']}%
                </p>
                """,
                unsafe_allow_html=True
            )

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Исходное изображение")
    st.image(image, use_container_width=True)

    prepared_image, pixels = prepare_image(image)

    palette_colors, palette_proportions, analysis_colors, analysis_proportions = extract_dominant_colors(
        pixels=pixels,
        n_colors=5,
        n_candidates=10
    )

    df_palette = make_palette_table(palette_colors, palette_proportions)
    df_palette = add_color_space_columns(df_palette)

    df_analysis = make_palette_table(analysis_colors, analysis_proportions)
    df_analysis = add_color_space_columns(df_analysis)

    show_color_swatches(df_palette)

    st.subheader("Таблица HEX + доли + HSV + CIE Lab")
    st.dataframe(df_palette, use_container_width=True)

    coverage = df_palette["Доля"].sum()
    st.write(f"Суммарное покрытие выбранных цветов: {coverage * 100:.2f}%")

    st.subheader("Bar chart")
    fig = plot_color_bar_chart(df_palette)
    st.pyplot(fig)

    with st.expander("10 цветов, по которым считаются признаки Иттена"):
        show_color_swatches(df_analysis)
        st.dataframe(df_analysis, use_container_width=True)
        analysis_coverage = df_analysis["Доля"].sum()
        if analysis_coverage * 100 > 100:
            analysis_coverage = 100.0
        else:
            analysis_coverage *= 100
        st.write(f"Суммарное покрытие 10 цветов: {analysis_coverage:.2f}%")

    st.subheader("Контрасты Иттена по изображению")

    contrasts = analyze_itten_contrasts_hsv_lab(df_analysis)

    for contrast_name, description in contrasts.items():
        st.markdown(f"**{contrast_name}**")
        st.write(description)

    st.subheader("Цветовая схема фотографии по 10 цветам")

    scheme_result = fit_harmony_scheme_hsv_lab(df_analysis)
    st.write(scheme_result["description"])

else:
    st.info("Пока изображение не загружено.")