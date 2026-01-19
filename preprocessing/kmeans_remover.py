import os
import numpy as np
from sklearn.cluster import KMeans
from utils.mri_utils import load_dicom_series_safe


# =========================================================
# Metrics
# =========================================================
def compute_black_ratio_per_slice(volume, percentile=5):
    ratios = []
    for img in volume:
        img = img.astype(np.float32)
        thresh = np.percentile(img, percentile)
        ratios.append(np.mean(img <= thresh))
    return np.array(ratios)


# =========================================================
# Core processing (ONE correct function)
# =========================================================
def process_series_with_kmeans_and_save(
    series_dir: str,
    patient: str,
    session: str,
    output_root: str,
    percentile: int = 5,
    n_clusters: int = 3,
):
    """
    Load a DICOM series, apply KMeans black-ratio filtering,
    and save filtered slices as .npy
    """

    series_name = os.path.basename(series_dir)

    volume = load_dicom_series_safe(series_dir)
    if volume is None or len(volume) < n_clusters:
        print(f"⚠️ Skipping empty/short series: {patient} | {series_name}")
        return

    # --- compute black ratios ---
    black_ratios = compute_black_ratio_per_slice(volume, percentile)
    X = black_ratios.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    centers = kmeans.cluster_centers_.flatten()

    # keep middle cluster (brain slices)
    mid_cluster = np.argsort(centers)[1]
    mask = kmeans.labels_ == mid_cluster

    kept_volume = volume[mask]

    if kept_volume.shape[0] == 0:
        print(f"⚠️ No slices kept: {patient} | {series_name}")
        return

    # --- output directory ---
    out_dir = os.path.join(
        output_root,
        patient,
        session,
        series_name,
    )
    os.makedirs(out_dir, exist_ok=True)

    # --- save slices ---
    for i, img in enumerate(kept_volume):
        np.save(os.path.join(out_dir, f"slice_{i:03d}.npy"), img)

    print(
        f"✅ {patient} | {series_name} | "
        f"kept {kept_volume.shape[0]}/{volume.shape[0]} slices | "
        f"clusters={np.round(np.sort(centers), 3)}"
    )


# =========================================================
# Optional: full-folder batch processing
# =========================================================
def process_all_mri(
    raw_root="data/MRI/raw",
    output_root="data/processed/black_ratio",
    percentile=5,
    n_clusters=3,
):
    """
    Process ALL MRI series in raw_root
    """
    os.makedirs(output_root, exist_ok=True)
    series_count = 0

    for patient in sorted(os.listdir(raw_root)):
        patient_dir = os.path.join(raw_root, patient)
        if not os.path.isdir(patient_dir):
            continue

        for session in os.listdir(patient_dir):
            session_dir = os.path.join(patient_dir, session)
            if not os.path.isdir(session_dir):
                continue

            for series in os.listdir(session_dir):
                series_dir = os.path.join(session_dir, series)
                if not os.path.isdir(series_dir):
                    continue

                # accept ANY folder with DICOMs
                if not any(f.lower().endswith(".dcm") for f in os.listdir(series_dir)):
                    continue

                process_series_with_kmeans_and_save(
                    series_dir=series_dir,
                    patient=patient,
                    session=session,
                    output_root=output_root,
                    percentile=percentile,
                    n_clusters=n_clusters,
                )

                series_count += 1

    print(f"\n✅ Finished preprocessing {series_count} MRI series")
