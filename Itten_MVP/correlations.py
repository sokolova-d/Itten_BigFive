from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.cluster import KMeans
from matplotlib.colors import rgb_to_hsv
from skimage.color import rgb2lab

CSV_PATH = "ava_bf_research_student_pack/st-data-bf-anon.csv"
IMAGES_DIR = "ava_bf_research_student_pack/BF_photo"

IMAGE_EXTENSION = ".jpg"

BIG_FIVE_COLUMNS = {
    "BF1": "extraversion",
    "BF2": "agreeableness",
    "BF3": "conscientiousness",
    "BF4": "neuroticism",
    "BF5": "openess to experience"
}

ITTEN_FEATURES_FOR_CODE = [
    "hue_contrast",
    "light_dark_contrast",
    "warm_cool_contrast",
    "saturation_contrast",
    "complementary_contrast",
    "extension_contrast",
    "scheme_harmony"
]

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def hue_distance(h1, h2):
    d = abs(h1 - h2)
    d = min(d, 1 - d)
    return d * 360


def hue_distance_degrees(h1, h2):
    d = abs(h1 - h2)
    return min(d, 360 - d)


def distance_to_template(hue, template_angles, rotation):
    distances = []
    for angle in template_angles:
        rotated_angle = (angle + rotation) % 360
        distances.append(hue_distance_degrees(hue, rotated_angle))

    return min(distances)

def prepare_image(image, max_size=300):
    image = image.convert("RGB")

    image.thumbnail((max_size, max_size))
    image_array = np.array(image)

    pixels = image_array.reshape(-1, image_array.shape[-1])

    return image, pixels

def select_diverse_colors(colors, proportions, n_final=5,
    min_hue_dist=20, min_rgb_dist=45, min_saturation=0.15):

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

    if len(selected_indices) < n_final:
        for idx in order:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) == n_final:
                break

    return selected_indices


def extract_dominant_colors(pixels, n_colors=5, n_candidates=10):

    kmeans = KMeans(
        n_clusters=n_candidates,
        random_state=42,
        n_init="auto"
    )

    kmeans.fit(pixels)
    candidate_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    counts = np.bincount(labels, minlength=n_candidates)

    candidate_proportions = counts / counts.sum()

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

    palette_order = np.argsort(palette_proportions)[::-1]
    palette_colors = palette_colors[palette_order]
    palette_proportions = palette_proportions[palette_order]

    candidate_order = np.argsort(candidate_proportions)[::-1]
    candidate_colors = candidate_colors[candidate_order]
    candidate_proportions = candidate_proportions[candidate_order]

    return palette_colors, palette_proportions, candidate_colors, candidate_proportions

def analyze_itten_contrasts_hsv_lab(df):

    hues = df["Hue"].to_numpy()
    saturations = df["Saturation"].to_numpy()

    l_values = df["L*"].to_numpy()
    chroma_values = df["Chroma"].to_numpy()

    proportions = df["Доля"].to_numpy()
    hex_colors = df["HEX"].to_numpy()

    features = {}

    saturated_mask = saturations >= 0.2

    saturated_hues = hues[saturated_mask]
    saturated_props = proportions[saturated_mask]
    saturated_sats = saturations[saturated_mask]
    saturated_chroma = chroma_values[saturated_mask]

    if len(saturated_hues) >= 2:
        hue_distances = []

        for i in range(len(saturated_hues)):
            for j in range(i + 1, len(saturated_hues)):
                hue_distances.append(
                    hue_distance_degrees(saturated_hues[i], saturated_hues[j])
                )

        hue_range = max(hue_distances)
        features["hue_contrast"] = hue_range
    else:
        features["hue_contrast"] = 0.0

    light_dark_range = l_values.max() - l_values.min()
    features["light_dark_contrast"] = light_dark_range

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
    else:
        warm_cool_balance_score = 0

    features["warm_cool_contrast"] = warm_cool_balance_score

    chroma_range = chroma_values.max() - chroma_values.min()
    features["saturation_contrast"] = chroma_range

    best_score = 0

    if len(saturated_hues) >= 2:
        for i in range(len(saturated_hues)):
            for j in range(i + 1, len(saturated_hues)):
                d = hue_distance_degrees(
                    saturated_hues[i],
                    saturated_hues[j]
                )

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

    features["complementary_contrast"] = best_score

    visual_weights = proportions * (1 + chroma_values / 100)

    if visual_weights.sum() > 0:
        q = visual_weights / visual_weights.sum()
        entropy = -np.sum(q * np.log(q + 1e-9))
        entropy_norm = entropy / np.log(len(q))
        extension_score = 1 - entropy_norm
    else:
        extension_score = 0

    features["extension_contrast"] = extension_score

    return features

def fit_harmony_scheme_hsv_lab(df):
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
            "scheme_fitting_error": np.nan,
            "scheme_rotation": np.nan,
            "scheme_confidence": "не определяется"
        }

    hues = df_sat["Hue"].to_numpy()
    proportions = df_sat["Доля"].to_numpy()
    saturations = df_sat["Saturation"].to_numpy()
    chroma_values = df_sat["Chroma"].to_numpy()

    weights = proportions * saturations * (1 + chroma_values / 100)

    if weights.sum() > 0:
        weights = weights / weights.sum()

    best_scheme = None
    best_error = float("inf")
    best_rotation = None

    for scheme_name, template_angles in schemes.items():
        for rotation in range(360):
            error = 0

            for h, w in zip(hues, weights):
                error += w * distance_to_template(hue=h, template_angles=template_angles, rotation=rotation)

            if error < best_error:
                best_error = error
                best_scheme = scheme_name
                best_rotation = rotation

    if best_error < 15:
        confidence = "высокая"
    elif best_error < 35:
        confidence = "средняя"
    else:
        confidence = "низкая"

    return {
        "scheme": best_scheme,
        "scheme_fitting_error": round(float(best_error), 2),
        "scheme_rotation": best_rotation,
        "scheme_confidence": confidence
    }

def make_palette_table(colors, proportions):
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

    df = df.sort_values(
        by="Доля",
        ascending=False
    ).reset_index(drop=True)

    df["№"] = range(1, len(df) + 1)

    return df

def add_color_space_columns(df):

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

def normalize_id(image_id):
    if pd.isna(image_id):
        return None

    image_id = str(image_id).strip()

    if image_id.endswith(".0"):
        image_id = image_id[:-2]

    return image_id


def find_image_path(images_dir, image_id):
    image_id = normalize_id(image_id)

    if image_id is None:
        return None

    path = Path(images_dir) / f"{image_id}{IMAGE_EXTENSION}"

    if path.exists():
        return path

    return None

def extract_features_from_image(image_path):
    image = Image.open(image_path)

    _, pixels = prepare_image(image)

    palette_colors, palette_proportions, analysis_colors, analysis_proportions = extract_dominant_colors(
        pixels=pixels,
        n_colors=5,
        n_candidates=10
    )

    df_palette = make_palette_table(
        colors=palette_colors,
        proportions=palette_proportions
    )
    df_palette = add_color_space_columns(df_palette)

    df_analysis = make_palette_table(
        colors=analysis_colors,
        proportions=analysis_proportions
    )
    df_analysis = add_color_space_columns(df_analysis)

    itten_features = analyze_itten_contrasts_hsv_lab(df_analysis)

    scheme_result = fit_harmony_scheme_hsv_lab(df_analysis)

    features = {}

    features.update(itten_features)

    features["scheme_harmony"] = (-scheme_result["scheme_fitting_error"])
    for i, row in df_palette.iterrows():
        number = i + 1

        features[f"palette_{number}_share"] = row["Доля"]
        features[f"palette_{number}_hue"] = row["Hue"]
        features[f"palette_{number}_saturation"] = row["Saturation"]
        features[f"palette_{number}_value"] = row["Value"]
        features[f"palette_{number}_lightness"] = row["L*"]
        features[f"palette_{number}_chroma"] = row["Chroma"]

    return features

def build_features_table(csv_path, images_dir):
    df = pd.read_csv(csv_path)

    rows = []

    for _, row in df.iterrows():
        image_id = row["id"]

        image_path = find_image_path(
            images_dir=images_dir,
            image_id=image_id
        )

        if image_path is None:
            print(f"id={image_id}: фото не найдено")
            continue

        try:
            image_features = extract_features_from_image(image_path)

            result_row = {
                "id": normalize_id(image_id)
            }

            for bf_column, trait_name in BIG_FIVE_COLUMNS.items():
                result_row[trait_name] = row[bf_column]

            result_row.update(image_features)

            rows.append(result_row)

            print(f"id={image_id}: обработано")

        except Exception as error:
            print(f"id={image_id}: ошибка — {error}")

    return pd.DataFrame(rows)

def calculate_spearman_correlations(df_features):

    traits = list(BIG_FIVE_COLUMNS.values())

    rows = []

    for trait in traits:
        for feature in ITTEN_FEATURES_FOR_CODE:
            if feature not in df_features.columns:
                rows.append({
                    "trait": trait,
                    "feature": feature,
                    "spearman_correlation": np.nan
                })
                continue

            corr = df_features[trait].corr(
                df_features[feature],
                method="spearman"
            )

            rows.append({
                "trait": trait,
                "feature": feature,
                "spearman_correlation": corr
            })

    return pd.DataFrame(rows)

def make_code_dict(df_corr):

    result = {}

    for trait in BIG_FIVE_COLUMNS.values():
        result[trait] = {}

        for feature in ITTEN_FEATURES_FOR_CODE:
            value = df_corr.loc[
                (df_corr["trait"] == trait)
                & (df_corr["feature"] == feature),
                "spearman_correlation"
            ]

            if len(value) == 0 or pd.isna(value.iloc[0]):
                result[trait][feature] = None
            else:
                result[trait][feature] = round(float(value.iloc[0]), 2)

    return result

def main():
    df_features = build_features_table(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR
    )

    if len(df_features) == 0:
        print("Не удалось обработать ни одной фотографии.")
        return

    df_corr_spearman = calculate_spearman_correlations(df_features)

    correlations_dict = make_code_dict(df_corr_spearman)

    with open("itten_correlations_for_code.json", "w", encoding="utf-8") as file:
        json.dump(
            correlations_dict,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("Словарь корреляций для вставки в код:")
    print()
    print("return ")
    print(json.dumps(correlations_dict, indent=4, ensure_ascii=False))

    print()
    print("Готово. Создан только один файл:")
    print("itten_correlations_for_code.json")


if __name__ == "__main__":
    main()