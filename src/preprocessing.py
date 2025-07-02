import re                     
import time                   
import numpy as np
from collections import Counter

def parse_personal_info(txt_path):
    info = {"Age": None, "Height": None, "Weight": None, "Gender": None, "Dominant hand": None}
    alias_map = {
        "age": "Age", "height": "Height", "height(cm)": "Height",
        "weight": "Weight", "weight(kg)": "Weight", "gender": "Gender",
        "dominanthand": "Dominant hand", "dominant hand": "Dominant hand"
    }
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            if ":" not in raw:
                continue
            key_raw, value_raw = map(str.strip, raw.split(":", 1))
            key_norm = key_raw.lower().replace(" ", "")
            if key_norm in alias_map:
                std_key = alias_map[key_norm]
                info[std_key] = value_raw.strip()
    for num_key in ("Age", "Height", "Weight"):
        if info[num_key]:
            m = re.search(r"\d+", info[num_key])
            if m:
                info[num_key] = int(m.group())
    return info

def extract_features_with_majority_label(data, window_size=42000, window_shift=175, verbose=False):
    start_time = time.time()
    chest = data['signal']['chest']
    labels = data['label']
    n_samples = len(labels)
    total_windows = (n_samples - window_size) // window_shift

    features_list = []
    labels_list = []

    for idx, start in enumerate(range(0, n_samples - window_size, window_shift)):
        end = start + window_size
        window_labels = labels[start:end]
        majority_label = Counter(window_labels).most_common(1)[0][0]

        win_acc = chest['ACC'][start:end]
        win_eda = chest['EDA'][start:end]
        win_tmp = chest['Temp'][start:end]

        acc_mean = np.mean(win_acc)
        acc_std = np.std(win_acc)
        acc_maxx = np.max(win_acc[:, 0])
        acc_maxy = np.max(win_acc[:, 1])
        acc_maxz = np.max(win_acc[:, 2])

        eda_max = np.max(win_eda)
        eda_min = np.min(win_eda)
        eda_mean = np.mean(win_eda)
        eda_range = eda_max - eda_min
        eda_std = np.std(win_eda)
        eda_slope = np.polyfit(np.arange(window_size), win_eda, 1)[0]

        tmp_max = np.max(win_tmp)
        tmp_min = np.min(win_tmp)
        tmp_mean = np.mean(win_tmp)
        tmp_range = tmp_max - tmp_min
        tmp_std = np.std(win_tmp)

        feature_vector = [
            acc_mean, acc_std, acc_maxx, acc_maxy, acc_maxz,
            eda_max, eda_min, eda_mean, eda_range, eda_std, eda_slope,
            tmp_max, tmp_min, tmp_mean, tmp_range, tmp_std
        ]

        features_list.append(feature_vector)
        labels_list.append(majority_label)

        if verbose and idx % (total_windows // 20 + 1) == 0:
            percent = (idx / total_windows) * 100
            print(f"  Progreso: {idx}/{total_windows} ventanas ({percent:.1f}%)")

    elapsed = time.time() - start_time
    if verbose:
        print(f"Extracción finalizada en {elapsed:.2f} segundos.")
    return np.array(features_list), np.array(labels_list)
