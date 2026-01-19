# Code/preprocessing/preprocess_t2.py

import os
import numpy as np
import pandas as pd
import pydicom
import cv2


# ----------------------------
# DICOM loading
# ----------------------------
def load_t2_dicom_series(series_dir):
    slices = []

    for f in os.listdir(series_dir):
        if not f.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, f))
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except:
            continue

    if len(slices) == 0:
        return None

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    return volume


# ----------------------------
# Preprocessing steps
# ----------------------------
def find_t2_series_folder(session_dir):
    """
    Automatically find the T2 MRI series folder inside a session directory.
    """
    for d in os.listdir(session_dir):
        series_path = os.path.join(session_dir, d)
        if not os.path.isdir(series_path):
            continue

        name = d.lower()

        # Strong T2 indicators
        if (
            "t2" in name
            or "fse" in name
            or "fast spin echo" in name
        ):
            # Exclude derived maps
            if any(x in name for x in ["adc", "trace", "average", "fractional"]):
                continue

            return series_path

    return None


def resize_volume(volume, target_hw=(128, 128)):
    return np.stack([
        cv2.resize(v, target_hw, interpolation=cv2.INTER_LINEAR)
        for v in volume
    ])


def normalize_volume(volume):
    mean, std = volume.mean(), volume.std()
    return volume if std < 1e-6 else (volume - mean) / std


def fix_depth(volume, target_depth=64):
    d = volume.shape[0]
    if d > target_depth:
        start = (d - target_depth) // 2
        return volume[start:start + target_depth]
    elif d < target_depth:
        pad = target_depth - d
        return np.pad(volume, ((pad//2, pad - pad//2), (0,0), (0,0)))
    return volume


def preprocess_series(series_dir):
    vol = load_t2_dicom_series(series_dir)
    if vol is None:
        return None
    vol = resize_volume(vol)
    vol = normalize_volume(vol)
    vol = fix_depth(vol)
    return vol


# ----------------------------
# Main execution
# ----------------------------
def main():
    df = pd.read_csv("Data/final_T2_mutation_dataset.csv")

    out_dir = "Data/processed/T2"
    os.makedirs(out_dir, exist_ok=True)

    records = []

    for _, row in df.iterrows():
        patient = row["Patient"]
        session = row["Session"]

        session_dir = f"Data/MRI/raw/{patient}/{session}"
        series_dir = find_t2_series_folder(session_dir)

        if series_dir is None:
            print(f"⚠️ No T2 series found for {patient} {session}")
            continue

        volume = preprocess_series(series_dir)
        if volume is None:
            continue

        fname = f"{patient}_{session}.npy"
        np.save(os.path.join(out_dir, fname), volume)

        records.append({
            "patient": patient,
            "session": session,
            "path": fname,
            "TP53": row["TP53_mut"],
            "PTEN": row["PTEN_mut"],
            "IDH": row["IDH_mut"],
            "EGFR": row["EGFR_mut"]
        })
        

    pd.DataFrame(records).to_csv(
        "Data/processed/metadata.csv", index=False
    )



if __name__ == "__main__":
    main()
