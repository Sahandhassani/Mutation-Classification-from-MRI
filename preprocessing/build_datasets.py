import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
from typing import List
import random

# ======================================================
# Slice-level Dataset
# ======================================================
class MRISliceDataset(Dataset):
    def __init__(self, paths, labels):
        assert len(paths) == len(labels)
        self.paths = list(paths)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.load(self.paths[idx], mmap_mode="r")
        img = np.nan_to_num(img.astype(np.float32))
        img = (img - img.mean()) / (img.std() + 1e-6)

        img = torch.from_numpy(img).unsqueeze(0)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        label = torch.from_numpy(self.labels[idx])
        return img, label


# ======================================================
# Build slice index (patient aware)
# ======================================================
def build_slice_index(processed_root, merged_df, label_cols,
                      max_slices_per_patient=100, seed=42):

    rng = np.random.RandomState(seed)
    label_df = merged_df.set_index("Patient")[label_cols]

    records = []

    for patient in sorted(os.listdir(processed_root)):
        pdir = os.path.join(processed_root, patient)
        if not os.path.isdir(pdir) or patient not in label_df.index:
            continue

        labels = label_df.loc[patient].values.astype(np.float32)

        slice_paths = []
        for root, _, files in os.walk(pdir):
            for f in files:
                if f.endswith(".npy"):
                    slice_paths.append(os.path.join(root, f))

        if len(slice_paths) == 0:
            continue

        if len(slice_paths) > max_slices_per_patient:
            slice_paths = rng.choice(slice_paths, max_slices_per_patient, replace=False)

        for sp in slice_paths:
            records.append({
                "patient": patient,
                "path": sp,
                "labels": labels
            })

    df = pd.DataFrame(records)
    print(f"✅ Total slices collected: {len(df)}")
    return df


# ======================================================
# Patient-wise TEST split with label coverage
# ======================================================
def patient_wise_test_split(slice_df, label_cols, test_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)

    patient_labels = (
        slice_df
        .drop_duplicates("patient")
        .set_index("patient")["labels"]
        .apply(np.asarray)
    )

    patients = list(patient_labels.index)
    rng.shuffle(patients)

    num_labels = len(label_cols)
    covered = np.zeros(num_labels, dtype=bool)
    test_patients = []

    # ---- ensure each label appears in TEST ----
    for p in patients:
        y = patient_labels.loc[p]
        if np.any((y == 1) & (~covered)):
            test_patients.append(p)
            covered |= (y == 1)
        if covered.all():
            break

    if not covered.all():
        raise RuntimeError("❌ Cannot guarantee at least one positive per label in TEST")

    desired_test_size = max(len(test_patients),
                            int(len(patients) * test_ratio))

    for p in patients:
        if p not in test_patients and len(test_patients) < desired_test_size:
            test_patients.append(p)

    train_patients = [p for p in patients if p not in test_patients]

    return train_patients, test_patients


# ======================================================
# Main builder
# ======================================================
def build_train_val_test_datasets(
    merged_df,
    processed_root,
    label_cols,
    max_slices_per_patient=50,
    seed=42,
    val_ratio=0.15
):
    # 1️⃣ slice index
    slice_df = build_slice_index(
        processed_root,
        merged_df,
        label_cols,
        max_slices_per_patient,
        seed
    )

    # 2️⃣ patient-wise TEST split
    train_patients, test_patients = patient_wise_test_split(
        slice_df,
        label_cols,
        test_ratio=0.15,
        seed=seed
    )

    train_df = slice_df[slice_df.patient.isin(train_patients)]
    test_df  = slice_df[slice_df.patient.isin(test_patients)]

    # 3️⃣ shuffle slices independently
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 4️⃣ split TRAIN slices → TRAIN / VAL
    val_size = int(len(train_df) * val_ratio)
    val_df   = train_df.iloc[:val_size]
    train_df = train_df.iloc[val_size:]

    # ---- sanity prints ----
    def print_stats(name, df):
        print(f"\n{name} SET")
        print(f"Patients: {df.patient.nunique()}")
        print(f"Slices: {len(df)}")
        pats = df.drop_duplicates("patient")["labels"].apply(np.asarray)
        pats = np.stack(pats.values)
        for i, c in enumerate(label_cols):
            print(f"  {c}: count={int(pats[:, i].sum())}")

    print_stats("TRAIN", train_df)
    print_stats("VAL", val_df)
    print_stats("TEST", test_df)

    return (
        MRISliceDataset(train_df["path"], np.stack(train_df["labels"])),
        MRISliceDataset(val_df["path"],   np.stack(val_df["labels"])),
        MRISliceDataset(test_df["path"],  np.stack(test_df["labels"])),
    )
