import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2lab
import json

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

    descriptions = {}
    features = {}

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

        descriptions["Контраст цветового тона"] = (
            f"{level}: максимальное расстояние между насыщенными оттенками {hue_range:.1f}°."
        )
        features["hue_contrast"] = hue_range
    else:
        descriptions["Контраст цветового тона"] = (
            "Не определяется: в палитре недостаточно насыщенных цветов."
        )
        features["hue_contrast"] = 0.0

    # Light-dark
    light_dark_range = l_values.max() - l_values.min()

    if light_dark_range < 15:
        level = "слабый"
    elif light_dark_range < 35:
        level = "умеренный"
    else:
        level = "сильный"

    descriptions["Светло-тёмный контраст"] = (
        f"{level}: разброс светлоты L* = {light_dark_range:.1f}."
    )
    features["light_dark_contrast"] = light_dark_range

    # Warm-cool
    warm_weight = 0
    cool_weight = 0

    for h, s, p in zip(hues, saturations, proportions):
        weight = p * s

        if (h >= 330) or (h < 150):
            warm_weight += weight
        else:
            cool_weight += weight

    total_temp_weight = warm_weight + cool_weight

    if total_temp_weight > 0:
        warm_share = warm_weight / total_temp_weight
        cool_share = cool_weight / total_temp_weight

        warm_cool_balance_score = 1 - abs(warm_share - cool_share)
        warm_cool_direction = warm_share - cool_share
    else:
        warm_share = 0
        cool_share = 0
        warm_cool_balance_score = 0
        warm_cool_direction = 0

    if warm_cool_balance_score < 0.25:
        level = "слабый"
    elif warm_cool_balance_score < 0.6:
        level = "умеренный"
    else:
        level = "сильный"

    descriptions["Тепло-холодный контраст"] = (
        f"{level}: тёплый вклад = {warm_share:.3f}, холодный вклад = {cool_share:.3f}."
        f"Баланс тёплого и холодного = {warm_cool_balance_score:.3f}."
    )
    features["warm_cool_contrast"] = warm_cool_balance_score

    # Saturation
    saturation_range = saturations.max() - saturations.min()
    chroma_range = chroma_values.max() - chroma_values.min()

    if chroma_range < 20:
        level = "слабый"
    elif chroma_range < 45:
        level = "умеренный"
    else:
        level = "сильный"

    descriptions["Насыщенность"] = (
        f"{level}: разброс HSV Saturation = {saturation_range:.3f}, "
        f"разброс Lab Chroma = {chroma_range:.1f}."
    )
    features["saturation_contrast"] = chroma_range

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
        descriptions["Комплиментарность"] = (
            f"найдена близкая к дополнительной пара: "
            f"{best_pair[0]} и {best_pair[1]}, расстояние {best_distance:.1f}°."
        )
        complementary_pair_found = 1
    else:
        descriptions["Комплиментарность"] = (
            "явная дополнительная пара не найдена."
        )
        complementary_pair_found = 0

    if best_distance is None:
        best_distance_value = 0
        complementary_distance_error = 180
    else:
        best_distance_value = best_distance
        complementary_distance_error = abs(best_distance - 180)

    features["complementary_contrast"] = best_score

    # Extension
    visual_weights = proportions * (1 + chroma_values / 100)

    if visual_weights.sum() > 0:
        q = visual_weights / visual_weights.sum()
        entropy = -np.sum(q * np.log(q + 1e-9))
        entropy_norm = entropy / np.log(len(q))
        extension_score = 1 - entropy_norm
    else:
        entropy = 0
        entropy_norm = 0
        extension_score = 0

    if extension_score < 0.2:
        level = "слабый"
    elif extension_score < 0.45:
        level = "умеренный"
    else:
        level = "сильный"

    dominant_color = df.loc[df["Доля"].idxmax(), "HEX"]
    dominant_percent = df.loc[df["Доля"].idxmax(), "Процент"]

    descriptions["Контраст распространения"] = (
        f"{level}: доминирующий цвет {dominant_color} занимает {dominant_percent}% изображения. "
        f"Оценка с учётом chroma = {extension_score:.3f}."
    )
    features["extension_contrast"] = extension_score

    return descriptions, features

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
            "fitting_error": None,
            "rotation": None,
            "confidence": "не определяется",
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

def load_bigfive_correlations(json_path="itten_correlations_for_code.json"):
    """
    Загружает реальные корреляции между признаками Иттена и Big Five из JSON-файла.

    Файл itten_correlations_for_code.json должен лежать в той же папке,
    что и main.py.
    """

    with open(json_path, "r", encoding="utf-8") as file:
        correlations = json.load(file)

    # В файле сейчас написано "openess to experience".
    # Здесь приводим название к более удобному ключу.
    if "openess to experience" in correlations:
        correlations["openness_to_experience"] = correlations.pop("openess to experience")

    return correlations


def normalize_image_features_for_bigfive(image_features):
    # Приводит признаки изображения к шкале -1..1.

    hue = image_features.get("hue_contrast", 0)
    light_dark = image_features.get("light_dark_contrast", 0)
    warm_cool = image_features.get("warm_cool_contrast", 0)
    saturation = image_features.get("saturation_contrast", 0)
    complementary = image_features.get("complementary_contrast", 0)
    extension = image_features.get("extension_contrast", 0)
    scheme_error = image_features.get("scheme_fitting_error", 180)

    if scheme_error is None:
        scheme_error = 180

    # Сначала приводим признаки к диапазону 0..1.
    normalized_0_1 = {
        "hue_contrast": min(hue / 180, 1),
        "light_dark_contrast": min(light_dark / 100, 1),
        "warm_cool_contrast": min(warm_cool, 1),
        "saturation_contrast": min(saturation / 100, 1),
        "complementary_contrast": min(complementary / 0.05, 1),
        "extension_contrast": min(extension, 1),
        "scheme_harmony": max(0, 1 - scheme_error / 180)
    }

    # Потом центрируем: 0..1 превращается в -1..1.
    centered = {}

    for feature_name, value in normalized_0_1.items():
        centered[feature_name] = (value - 0.5) * 2

    return centered

def estimate_bigfive_0_100(image_features):
    # Рассчитывает демонстрационный профиль Big Five в диапазоне 0..100
    # на основе реальных корреляций из JSON.

    normalized_features = normalize_image_features_for_bigfive(image_features)
    correlations = load_bigfive_correlations()

    trait_labels = {
        "extraversion": "Extraversion / экстраверсия",
        "agreeableness": "Agreeableness / привязанность",
        "conscientiousness": "Conscientiousness / самоконтроль",
        "neuroticism": "Neuroticism / нейротизм",
        "openness_to_experience": "Openness to experience / экспрессивность"
    }

    results = []

    for trait, weights in correlations.items():
        weighted_sum = 0
        abs_weight_sum = 0

        for feature_name, weight in weights.items():
            feature_value = normalized_features.get(feature_name, 0)

            weighted_sum += feature_value * weight
            abs_weight_sum += abs(weight)

        # Перевод из диапазона примерно -1..1 в диапазон 0..100.
        score_0_100 = 50 + weighted_sum * 160

        # Ограничиваем диапазон.
        score_0_100 = min(max(score_0_100, 0), 100)

        results.append({
            "trait_key": trait,
            "Признак": trait_labels[trait],
            "Значение 0–100": round(score_0_100, 1)
        })

    return pd.DataFrame(results)

def get_level_by_score(score):
    # Переводит численное значение 0..100 в уровень.

    if score < 35:
        return "low"
    elif score < 65:
        return "middle"
    else:
        return "high"

def level_to_russian(level):
    levels = {
        "low": "низкий",
        "middle": "средний",
        "high": "высокий"
    }

    return levels.get(level, level)

def generate_bigfive_text_summary(df_bigfive, image_features):
    # Формирует итоговый текстовый вывод о человеке
    # по демонстрационной модели Big Five.

    trait_info = {
        "extraversion": {
            "name": "экстраверсия",
            "definition": "Экстраверсия характеризует высокую вовлечённость внешним миром.",
            "high": (
                "Скорее всего, человек выглядит более открытым к общению, активным "
                "и ориентированным на внешний мир. Ему может быть свойственно стремление "
                "быть заметным и проявлять себя в социальной среде."
            ),
            "middle": (
                "Скорее всего, человек сочетает черты интровертности и экстравертности: "
                "может быть общительным в комфортной среде, но не обязательно стремится "
                "постоянно находиться в центре внимания."
            ),
            "low": (
                "Скорее всего, человек более интровертный, сдержанный и спокойный. "
                "Ему может быть комфортнее в одиночестве или в кругу нескольких близких людей, "
                "чем в большом окружении."
            )
        },
        "agreeableness": {
            "name": "привязанность",
            "definition": (
                "Привязанность отражает отношение человека к сотрудничеству, доверию "
                "и социальной гармонии."
            ),
            "high": (
                "Скорее всего, человек склонен к сотрудничеству, доверию и вниманию "
                "к другим людям. Он может производить впечатление приятного, мягкого "
                "и сговорчивого человека."
            ),
            "middle": (
                "Скорее всего, человек может быть дружелюбным и готовым к сотрудничеству, "
                "но при этом способен отстаивать свои интересы и сохранять дистанцию."
            ),
            "low": (
                "Скорее всего, человек меньше склонен к уступчивости и доверию. "
                "Ему могут быть ближе самостоятельность, обособленность, критичность "
                "или осторожность в отношениях с другими."
            )
        },
        "conscientiousness": {
            "name": "самоконтроль",
            "definition": (
                "Самоконтроль определяет способность человека контролировать, управлять "
                "и направлять свои импульсы."
            ),
            "high": (
                "Скорее всего, человек склонен к ответственности, организованности "
                "и постановке чётких целей. Он может восприниматься как надёжный "
                "и трудолюбивый."
            ),
            "middle": (
                "Скорее всего, человек способен быть организованным, когда это нужно, "
                "но не всегда стремится к жёсткому контролю и строгому планированию."
            ),
            "low": (
                "Скорее всего, человек меньше склонен к строгой организованности "
                "и постоянному самоконтролю. В поведении могут сильнее проявляться "
                "спонтанность, импульсивность или беспечность."
            )
        },
        "neuroticism": {
            "name": "нейротизм",
            "definition": (
                "Нейротизм определяет склонность человека переживать негативные эмоции."
            ),
            "high": (
                "Скорее всего, человек более чувствительный и эмоциональный. "
                "Он может легче расстраиваться, сильнее реагировать на напряжённые ситуации "
                "и чаще переживать тревогу или внутреннюю неустойчивость."
            ),
            "middle": (
                "Скорее всего, человек имеет умеренную эмоциональную чувствительность: "
                "может переживать стресс, но не обязательно постоянно находится "
                "в состоянии тревоги или напряжения."
            ),
            "low": (
                "Скорее всего, человек более стрессоустойчивый и эмоционально стабильный. "
                "Он может спокойнее реагировать на трудности и меньше склонен "
                "к тревожности."
            )
        },
        "openness_to_experience": {
            "name": "экспрессивность",
            "definition": (
                "Экспрессивность описывает стиль мышления и отличает более творческих, "
                "любознательных людей от более практичных и приземлённых."
            ),
            "high": (
                "Скорее всего, человек любит новизну, разнообразие и перемены. "
                "Ему могут быть свойственны любознательность, интерес к новым идеям, "
                "изобретательность и творческое мышление."
            ),
            "middle": (
                "Скорее всего, человек сочетает интерес к новому с практичностью. "
                "Он может быть открыт к новым идеям, но при этом сохранять реалистичный "
                "и прагматичный взгляд."
            ),
            "low": (
                "Скорее всего, человек больше склонен к практичности, прагматичности "
                "и реализму. Ему могут быть ближе устойчивые взгляды, понятные решения "
                "и меньшая потребность в экспериментах."
            )
        }
    }

    # Сортируем, чтобы найти самые выраженные черты.
    df_sorted = df_bigfive.sort_values(by="Значение 0–100", ascending=False)

    top_trait = df_sorted.iloc[0]
    low_trait = df_sorted.iloc[-1]

    top_key = top_trait["trait_key"]
    low_key = low_trait["trait_key"]

    top_score = top_trait["Значение 0–100"]
    low_score = low_trait["Значение 0–100"]

    top_level = get_level_by_score(top_score)
    low_level = get_level_by_score(low_score)

    other_traits = df_sorted.iloc[1:-1]

    other_traits_text_parts = []

    for _, row in other_traits.iterrows():
        trait_key = row["trait_key"]
        trait_score = row["Значение 0–100"]
        trait_level = get_level_by_score(trait_score)

        other_traits_text_parts.append(
            f"Шкала «{trait_info[trait_key]['name']}» — "
            f"{trait_score:.1f}/100, уровень: {level_to_russian(trait_level)}. "
            f"{trait_info[trait_key][trait_level]}"
        )

    other_traits_text = "\n\n".join(other_traits_text_parts)

    hue = image_features.get("hue_contrast", 0)
    light_dark = image_features.get("light_dark_contrast", 0)
    warm_cool = image_features.get("warm_cool_contrast", 0)
    saturation = image_features.get("saturation_contrast", 0)
    complementary = image_features.get("complementary_contrast", 0)
    extension = image_features.get("extension_contrast", 0)
    scheme = image_features.get("scheme", "не определяется")
    scheme_error = image_features.get("scheme_fitting_error", None)

    image_evidence = []

    if hue >= 120:
        image_evidence.append(f"сильный контраст цветового тона ({hue:.1f}°)")
    elif hue >= 60:
        image_evidence.append(f"умеренный контраст цветового тона ({hue:.1f}°)")

    if light_dark >= 35:
        image_evidence.append(f"выраженный светло-тёмный контраст ({light_dark:.1f})")

    if warm_cool >= 0.6:
        image_evidence.append(f"заметный тепло-холодный контраст ({warm_cool:.2f})")

    if saturation >= 45:
        image_evidence.append(f"высокий контраст насыщенности ({saturation:.1f})")
    elif saturation >= 20:
        image_evidence.append(f"умеренный контраст насыщенности ({saturation:.1f})")

    if complementary >= 0.02:
        image_evidence.append(f"выраженная комплиментарность ({complementary:.3f})")

    if extension >= 0.4:
        image_evidence.append(f"сильное доминирование отдельных цветов ({extension:.2f})")

    if len(image_evidence) == 0:
        evidence_text = (
            "По цветовым признакам аватарка выглядит относительно нейтральной: "
            "без резко выраженных контрастов Иттена."
        )
    else:
        evidence_text = (
            "Судя по рассчитанным признакам Иттена, в аватарке выделяются: "
            + ", ".join(image_evidence)
            + "."
        )

    if scheme_error is not None:
        evidence_text += (
            f" Ближайшая цветовая схема — «{scheme}» "
            f"(ошибка fitting = {scheme_error:.1f}°)."
        )

    summary = (
        f"{evidence_text}\n\n"
        f"Самая выраженная шкала — «{trait_info[top_key]['name']}» "
        f"({top_score:.1f}/100, уровень: {level_to_russian(top_level)}). "
        f"{trait_info[top_key]['definition']} "
        f"{trait_info[top_key][top_level]}\n\n"

        f"Наименее выраженная шкала — «{trait_info[low_key]['name']}» "
        f"({low_score:.1f}/100, уровень: {level_to_russian(low_level)}). "
        f"{trait_info[low_key]['definition']} "
        f"{trait_info[low_key][low_level]}\n\n"

        f"Остальные шкалы:\n\n"
        f"{other_traits_text}\n\n"

        f"Важно: этот вывод основан на цветовых признаках аватарки и корреляционных весах. "
        f"Он не является психологической диагностикой."
    )

    return summary

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

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

    contrast_descriptions, contrast_features = analyze_itten_contrasts_hsv_lab(df_analysis)

    scheme_result = fit_harmony_scheme_hsv_lab(df_analysis)

    image_features = {
        **contrast_features,
        "scheme": scheme_result["scheme"],
        "scheme_fitting_error": scheme_result["score"],
        "scheme_harmony": -scheme_result["score"],
        "scheme_rotation": scheme_result["rotation"],
        "scheme_confidence": scheme_result["confidence"]
    }

    df_features = pd.DataFrame([image_features])

    df_bigfive = estimate_bigfive_0_100(image_features)

    show_color_swatches(df_palette)

    st.subheader("Профиль Big Five по цветовым признакам аватарки")

    fig_bigfive, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        df_bigfive["Признак"],
        df_bigfive["Значение 0–100"]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Значение, 0–100")
    ax.set_title("Профиль Big Five по цветовым признакам")

    for i, value in enumerate(df_bigfive["Значение 0–100"]):
        ax.text(
            i,
            value + 1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    st.pyplot(fig_bigfive)

    st.subheader("Итоговая интерпретация")

    personality_summary = generate_bigfive_text_summary(
        df_bigfive,
        image_features
    )

    st.write(personality_summary)

    with st.expander("Исходное изображение"):
        st.image(image, use_container_width=True)

    with st.expander("Таблица 5 доминирующих цветов"):
        st.dataframe(df_palette, use_container_width=True)

        coverage = df_palette["Доля"].sum()
        st.write(f"Суммарное покрытие выбранных цветов: {coverage * 100:.2f}%")

    with st.expander("Bar chart 5 доминирующих цветов"):
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

    with st.expander("Контрасты Иттена по изображению"):
        for contrast_name, description in contrast_descriptions.items():
            st.markdown(f"**{contrast_name}**")
            st.write(description)

    with st.expander("Цветовая схема фотографии по 10 цветам"):
        st.write(scheme_result["description"])

    with st.expander("Численные признаки для дальнейшего анализа"):
        st.dataframe(df_features, use_container_width=True)

    with st.expander("Таблица значений Big Five"):
        st.write(
            "Профиль рассчитан на основе признаков Иттена и реальных корреляционных весов. "
            "Значения приведены к диапазону 0–100: низкие значения означают слабую "
            "выраженность признака, высокие — сильную выраженность."
        )
        st.dataframe(
            df_bigfive[["Признак", "Значение 0–100"]],
            use_container_width=True
        )

else:
    st.info("Пока изображение не загружено.")