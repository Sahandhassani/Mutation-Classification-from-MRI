import os
import pandas as pd
from collections import defaultdict
from typing import Optional


# =========================================================
# Helpers
# =========================================================
def normalize_modality(name: str) -> Optional[str]:
    name = name.lower()

    if "axi" in name and "t1" in name:
        return "axial_T1"
    if "cor" in name and "t1" in name:
        return "coronal_T1"
    if "sag" in name and "t1" in name:
        return "sagittal_T1"

    if "axi" in name and "t2" in name:
        return "axial_T2"
    if "cor" in name and "t2" in name:
        return "coronal_T2"
    if "sag" in name and "t2" in name:
        return "sagittal_T2"

    if "axi" in name and "flair" in name:
        return "axial_FLAIR"
    if "cor" in name and "flair" in name:
        return "coronal_FLAIR"
    if "sag" in name and "flair" in name:
        return "sagittal_FLAIR"

    return None


def count_dicom_files(folder_path: str) -> int:
    return sum(
        1 for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    )


EXPECTED_COLUMNS = [
    "Patient",
    "Session",
    "axial_T1", "coronal_T1", "sagittal_T1",
    "axial_T2", "coronal_T2", "sagittal_T2",
    "axial_FLAIR", "coronal_FLAIR", "sagittal_FLAIR",
]


# =========================================================
# MRI indexing
# =========================================================
def build_mri_first_session_index(mri_root: str) -> pd.DataFrame:
    records = []

    for patient in sorted(os.listdir(mri_root)):
        patient_dir = os.path.join(mri_root, patient)
        if not os.path.isdir(patient_dir):
            continue

        sessions = sorted(
            s for s in os.listdir(patient_dir)
            if os.path.isdir(os.path.join(patient_dir, s))
        )
        if not sessions:
            continue

        first_session = sessions[0]
        session_dir = os.path.join(patient_dir, first_session)

        modality_counts = defaultdict(int)

        for folder in os.listdir(session_dir):
            folder_path = os.path.join(session_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            modality = normalize_modality(folder)
            if modality is None:
                continue

            modality_counts[modality] += count_dicom_files(folder_path)

        record = {col: 0 for col in EXPECTED_COLUMNS}
        record["Patient"] = patient
        record["Session"] = first_session

        for mod, count in modality_counts.items():
            record[mod] = count

        records.append(record)

    df = pd.DataFrame(records, columns=EXPECTED_COLUMNS)
    df.fillna(0, inplace=True)
    return df


# =========================================================
# Mutation + merge
# =========================================================
def build_merged_mri_mutation_df(
    mri_root: str,
    mutation_csv: str,
) -> pd.DataFrame:

    mri_df = build_mri_first_session_index(mri_root)

    mut_df = pd.read_csv(mutation_csv)

    # IDH mutation definition
    mut_df["IDH_mut"] = (
        (mut_df["IDH1_mut"] == 1) |
        (mut_df["IDH2_mut"] == 1)
    ).astype(int)

    merged_df = mri_df.merge(
        mut_df,
        left_on="Patient",
        right_on="TCGA_ID",
        how="inner"
    ).drop(columns=["TCGA_ID"])

    return merged_df

# =========================================================
# Script entry
# =========================================================
if __name__ == "__main__":
    MRI_ROOT = "data/MRI/raw"
    MUTATION_CSV = "Data/mutation/TCGA_GBM_mutation_labels_02.csv"
    OUTPUT_PATH = "data/MRI/mri_first_session_with_mutations.csv"

    merged_df = build_merged_mri_mutation_df(
        mri_root=MRI_ROOT,
        mutation_csv=MUTATION_CSV,
    )

    merged_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved merged dataframe → {OUTPUT_PATH}")
    print(f"Patients: {merged_df['Patient'].nunique()}")
    print(f"Rows: {len(merged_df)}")
