import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset

class GBMMRIDataset(Dataset):
    def __init__(self, mri_root, label_csv, modality, transform=None):
        self.mri_root = mri_root
        self.labels = pd.read_csv(label_csv)
        self.modality = modality
        self.transform = transform

        self.patient_ids = self.labels["TCGA_ID"].values
        self.label_map = dict(
            zip(self.labels["TCGA_ID"], self.labels["IDH_mut"])
        )

    def __len__(self):
        return len(self.patient_ids)

    def load_mri(self, patient_id):
        patient_dir = os.path.join(self.mri_root, patient_id)

        # example: choose first timepoint
        timepoints = sorted(os.listdir(patient_dir))
        tp_dir = os.path.join(patient_dir, timepoints[0])

        modality_dir = os.path.join(tp_dir, self.modality)
        volume_path = os.listdir(modality_dir)[0]

        volume = nib.load(
            os.path.join(modality_dir, volume_path)
        ).get_fdata()

        volume = np.expand_dims(volume, axis=0)  # [C, D, H, W]
        return volume.astype(np.float32)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        volume = self.load_mri(pid)
        label = self.label_map[pid]

        if self.transform:
            volume = self.transform(volume)

        return torch.tensor(volume), torch.tensor(label)
