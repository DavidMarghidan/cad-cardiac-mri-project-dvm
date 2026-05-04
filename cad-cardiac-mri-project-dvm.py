#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
#
# if os.path.exists("/kaggle/input"):
#     base_path = "/kaggle/input"
# else:
#     base_path = r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset'  # local folder in your project
#
# for dirname, _, filenames in os.walk(base_path):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# =============================
# 🧠 CAD Detection from Cardiac MRI - Kaggle Notebook
# =============================

# Install (if needed)
# !pip install timm

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =============================
# CONFIG
# =============================

DATASET_PATH = "/kaggle/input/cad-cardiac-mri-dataset"

if os.path.exists("/kaggle/input"):
    DATASET_PATH = "/kaggle/input"
else:
    DATASET_PATH = r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset'  # local folder in your project

# for dirname, _, filenames in os.walk(DATASET_PATH):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

IMG_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# DATASET
# =============================

class MRIDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = np.stack([img]*3, axis=-1)

        if self.transform:
            img = self.transform(img)

        return img, label, patient

# =============================
# LOAD DATA
# =============================

# def load_samples(root_dir):
#     samples = []
#     for patient in os.listdir(root_dir):
#         patient_path = os.path.join(root_dir, patient)
#         if not os.path.isdir(patient_path):
#             continue
#
#         label = 1 if "CAD" in patient else 0
#
#         for img in os.listdir(patient_path):
#             samples.append((os.path.join(patient_path, img), label, patient))
#
#     return samples

def load_samples(root_dir):
    samples = []

    for label_name in ["Normal", "Sick"]:
        class_path = os.path.join(root_dir, label_name)

        if not os.path.exists(class_path):
            continue

        label = 0 if label_name == "Normal" else 1

        for patient in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient)

            if not os.path.isdir(patient_path):
                continue

            # ID pacient unic
            patient_id = f"{label_name}_{patient}"

            # cauta recursiv foldere "series..."
            for root, dirs, files in os.walk(patient_path):

                # verifică dacă folderul curent e de tip series
                if os.path.basename(root).startswith("series") or os.path.basename(root).startswith("SR_"):

                    for file in files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            img_path = os.path.join(root, file)
                            samples.append((img_path, label, patient_id))

    return samples


samples = load_samples(DATASET_PATH)

# =============================
# TRANSFORMS
# =============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =============================
# SEGMENTATION (SIMPLIFIED U-NET)
# =============================

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer4(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

# =============================
# FEATURE EXTRACTOR
# =============================

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)

feature_extractor = FeatureExtractor().to(DEVICE)
feature_extractor.eval()

# =============================
# SLICE FILTERING
# =============================

def slice_score(img):
    return img.std()

# =============================
# FEATURE EXTRACTION LOOP
# =============================

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_features = []
    all_labels = []
    all_patients = []

    with torch.no_grad():
        for imgs, labels, patients in tqdm(loader):
            imgs = imgs.to(DEVICE).float()
            feats = feature_extractor(imgs)

            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_patients.extend(patients)

    return np.vstack(all_features), np.array(all_labels), np.array(all_patients)

# =============================
# GROUP BY PATIENT
# =============================

def aggregate_patient(features, labels, patients):
    patient_dict = {}

    for f, l, p in zip(features, labels, patients):
        if p not in patient_dict:
            patient_dict[p] = {"features": [], "label": l}

        patient_dict[p]["features"].append(f)

    X, y = [], []

    for p in patient_dict:
        feats = np.array(patient_dict[p]["features"])

        # attention-like mean
        agg = feats.mean(axis=0)

        X.append(agg)
        y.append(patient_dict[p]["label"])

    return np.array(X), np.array(y)

# =============================
# CROSS VALIDATION
# =============================

patients = [s[2] for s in samples]

print("Numar pacienti:", len(set(patients)))
print("Numar imagini:", len(samples))

unique_patients = list(set(patients))
n_splits = min(5, len(unique_patients))

# gkf = GroupKFold(n_splits=n_splits)

aucs = []

# for fold, (train_idx, val_idx) in enumerate(gkf.split(samples, groups=patients)):
#     print(f"\n===== FOLD {fold} =====")
#
#     train_samples = [samples[i] for i in train_idx]
#     val_samples = [samples[i] for i in val_idx]

# balanced selection
normal_patients = [p for p in patients if p.startswith("Normal")]
sick_patients = [p for p in patients if p.startswith("Sick")]

train_patients = [normal_patients[0], sick_patients[0]]
test_patients  = [normal_patients[1], sick_patients[1]]

print("Train patients:", train_patients)
print("Test patients:", test_patients)

train_samples = [s for s in samples if s[2] in train_patients]
val_samples  = [s for s in samples if s[2] in test_patients]


train_ds = MRIDataset(train_samples, transform)
val_ds = MRIDataset(val_samples, transform)
# FIX this error - test commit
X_train, y_train, p_train = extract_features(train_ds)
X_val, y_val, p_val = extract_features(val_ds)

X_train, y_train = aggregate_patient(X_train, y_train, p_train)
X_val, y_val = aggregate_patient(X_val, y_val, p_val)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, preds)

print("AUC:", auc)
aucs.append(auc)

# TEST2

################################## end for


print("\nFINAL AUC:", np.mean(aucs))

# =============================
# OPTIONAL IMPROVEMENTS
# =============================

# 1. Replace mean aggregation with attention pooling
# 2. Add real segmentation masks
# 3. Use TTA (test-time augmentation)
# 4. Try SVM / XGBoost
# 5. Keep top-K slices instead of all

print("Done!")
