Mutation Label Classification from Medical Images Using Computer Vision Models

Author: Sahand Hassanizorgabad
Affiliation: Department of Computational Sciences and Engineering, Koç University (GSSE)
Email: shassanizorgabad22@ku.edu.tr
Date: January 18, 2026

📌 Project Overview
Glioblastoma multiforme (GBM) is a highly aggressive brain tumor in which molecular alterations play a critical role in diagnosis, prognosis, and treatment planning. In current clinical practice, these molecular alterations are identified through invasive biopsy and tissue sampling.
This project investigates whether clinically relevant molecular mutation labels—specifically TP53, PTEN, EGFR, and IDH—can be predicted non-invasively from MRI scans using modern computer vision models. Molecular labels are used only as supervision and are not provided as model inputs.
The ultimate goal is to explore the feasibility of reducing reliance on biopsy by extracting molecular insights directly from MRI, thereby supporting non-invasive clinical decision-making.

🧠 Models Implemented
The following 2D vision-based models are implemented and evaluated:
("AlexNet2D", AlexNet2D(num_labels=num_labels)),
("SENet", SENet(num_labels=num_labels)),
("ConvNeXtV2_2D", ConvNeXtV2_2D(in_chans=1, num_classes=num_labels)),
("VisionLSTM", VisionLSTM(img_size=224, patch_size=16, in_chans=1, embed_dim=256, hidden_dim=256, depth=6, num_labels=num_labels)),
("ViT", VisionTransformer(num_labels=num_labels)),

📂 Dataset Description (TCGA)
1-MRI Data
MRI data are obtained from The Cancer Imaging Archive (TCIA)
Dataset: TCGA-GBM
Includes multi-sequence, multi-plane MRI volumes (T1, T2, FLAIR)
Stored in DICOM format

2-Mutation Labels
Molecular mutation labels are obtained from The Cancer Genome Atlas (TCGA)
Genomic data include mutation annotations for 371 GBM patients
After patient-level matching with MRI data, 157 patients have both MRI and mutation labels
Due to download constraints, a subset is used in experiments

3-Target Mutations
TP53
PTEN
EGFR
IDH
Each mutation is treated as a binary classification task.

⚠️ Important: Data Requirement & Folder Structure
❗ You must download the MRI data manually
This repository does NOT include MRI data.
To run the code, you must:
Download TCGA-GBM MRI data from TCIA
Place the raw DICOM files in the following directory:
data/
└── mri/
    └── raw/
        ├── TCGA-02-0001/
        │   └── session_1/
        │       └── series_1/
        │           ├── image001.dcm
        │           ├── image002.dcm
        │           └── ...
        ├── TCGA-02-0002/
        └── ...
✅ The preprocessing pipeline assumes this structure.
❌ The code will not run without MRI data placed in data/mri/raw.

🧹 Preprocessing Pipeline
Edge Slice Removal via Unsupervised Clustering
MRI volumes often contain uninformative slices at the beginning and end of sequences. To remove these:
Black pixel ratio is computed per slice using the 5th percentile intensity threshold
Each slice is represented by a single scalar (background dominance)
K-Means clustering (k=3) is applied:
background-dominated slices
brain-dominated slices
transitional slices
The middle cluster is retained
Edge/background slices are discarded automatically
This improves data quality and robustness, especially in low-sample regimes.


🏗 Methodology Summary
MRI slices from all planes (axial, sagittal, coronal) are used
Due to limited data, all slices are aggregated rather than modeled separately
Models are trained end-to-end using only MRI images
Mutation labels are used strictly as ground truth supervision
Evaluation is performed mutation-wise and averaged across labels

🚀 Running the Code
Download MRI data from TCIA
Place DICOM files in data/mri/raw
Install dependencies
Run training scripts for the desired model
Results are saved under:
experiments/
└── ModelName/
    ├── history.csv
    ├── results.json
    └── plots/

📊 Outputs
Per-epoch training and validation metrics
Mutation-wise accuracy (TP53, PTEN, EGFR, IDH)
Mean accuracy and loss
CSV and JSON summaries for reproducibility
Plots for convergence and comparison across models

🔮 Limitations & Future Work
Limitations
Limited number of matched MRI–genomic samples
All planes and sequences are combined rather than modeled independently
Slice-level modeling instead of full 3D volumes
Future Work
Acquire full TCGA MRI dataset
Train 3D CNN and transformer-based models
Separate modeling per MRI plane and sequence
Fuse predictions at the patient level
