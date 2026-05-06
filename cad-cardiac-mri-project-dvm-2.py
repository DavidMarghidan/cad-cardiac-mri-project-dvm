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


# Import sistem de fișiere
import os

# OpenCV pentru citirea și procesarea imaginilor medicale
import cv2

# NumPy pentru operații numerice
import numpy as np

# tqdm pentru bare de progres în loop-uri
from tqdm import tqdm

# PyTorch – framework principal de deep learning
import torch

# Module de bază pentru rețele neuronale
import torch.nn as nn

# Dataset și DataLoader pentru batching
from torch.utils.data import Dataset, DataLoader

# Transformări standard pentru imagini
import torchvision.transforms as transforms

# Modele pretrained (ResNet, EfficientNet etc.)
from torchvision import models

# Clasificator liniar
from sklearn.linear_model import LogisticRegression

# Metrică AUC pentru evaluare binară
from sklearn.metrics import roc_auc_score

import torch.nn.functional as F

# =============================
# MRI SLICES (2D) – CONFIG
# =============================

# Dimensiunea standard de input pentru CNN-uri ImageNet
IMG_SIZE = 224

# Număr de imagini procesate simultan
BATCH_SIZE = 8

# Selectare automată GPU / CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path standard Kaggle către dataset
DATASET_PATH = "/kaggle/input/cad-cardiac-mri-dataset"

if os.path.exists("/kaggle/input"):
    DATASET_PATH = "/kaggle/input"
else:
    DATASET_PATH = r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset'  # local folder in your project


# =============================
# MRI SLICES (2D) – DATASET
# =============================

class MRIDataset(Dataset):
    """
    Dataset PyTorch custom:
    returnează (imagine, label, patient_id)
    """

    def __init__(self, samples, transform=None):
        self.samples = samples          # listă de tupluri (path, label, patient)
        self.transform = transform      # transformări pentru image preprocessing

    def __len__(self):
        return len(self.samples)        # numărul total de imagini

    def __getitem__(self, idx):
        # Extragem informațiile corespunzătoare imaginii idx
        img_path, label, patient = self.samples[idx]

        # Citim imaginea ca grayscale (RMN este 1 canal)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Redimensionăm la dimensiunea standard CNN
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Convertim 1 canal → 3 canale (ImageNet compatibility)
        img = np.stack([img] * 3, axis=-1)

        # Aplicăm transformări PyTorch
        if self.transform:
            img = self.transform(img)

        # Returnăm imaginea + eticheta + ID pacient
        return img, label, patient

# =============================
# LOAD DATA
# =============================

def load_samples(root_dir):
    """
    Parcurge structura:
    Normal/Patient_ID/series_x/*.png
    Sick/Patient_ID/series_y/*.png
    """
    samples = []

    # Iterăm cele două clase
    for class_name in ["Normal", "Sick"]:
        label = 0 if class_name == "Normal" else 1
        class_path = os.path.join(root_dir, class_name)

        # Iterăm pacienții
        for patient in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient)

            # Construim un ID unic pe pacient
            patient_id = f"{class_name}_{patient}"

            # Parcurgere recursivă după imagini
            for root, _, files in os.walk(patient_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        samples.append(
                            (os.path.join(root, f), label, patient_id)
                        )

    return samples

# Încărcăm toate imaginile
samples = load_samples(DATASET_PATH)

# =============================
# MRI SLICES (2D) – PREPROCESSING
# =============================

# Pipeline standard de normalizare
transform = transforms.Compose([
    transforms.ToTensor(),                         # HWC → CHW și [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)         # standardizare ImageNet-like
])

# =============================
# ATTENTION U-NET – HEART SEGMENTATION
# =============================

class AttentionBlock(nn.Module):
    """
    Attention Gate – evidențiază regiuni relevante (inima)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)     # proiecție gate
        self.W_x = nn.Conv2d(F_l, F_int, 1)     # proiecție skip
        self.psi = nn.Conv2d(F_int, 1, 1)       # mască attention
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # Computăm attention map
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.sigmoid(self.psi(psi))

        # Maskăm feature map-ul
        return x * psi

class AttentionUNet(nn.Module):
    """
    U-Net simplificat cu backbone ResNet18
    """
    def __init__(self):
        super().__init__()

        # Encoder pretrained
        self.encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder.fc = nn.Identity()          # eliminăm clasificatorul

        # Decoder minimal
        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.att = AttentionBlock(256, 256, 128)
        self.conv_out = nn.Conv2d(256, 1, 1)     # output: mască binară

    def forward(self, x):
        # Forward encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # Attention + output
        g = self.conv1(x)
        x = self.att(g, g)
        x = self.conv_out(x)

        # Sigmoid → probabilitate pixel-wise
        return torch.sigmoid(x)

    def forward(self, x):
        input_size = x.shape[-2:]  # (224, 224)
        # Forward encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # Attention + output
        g = self.conv1(x)
        x = self.att(g, g)
        x = self.conv_out(x)

        # Sigmoid → probabilitate pixel-wise
        x = torch.sigmoid(x)

        # ✅ UPSAMPLE LA DIMENSIUNE ORIGINALĂ
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


# Inițializare model segmentare
unet = AttentionUNet().to(DEVICE)
unet.eval()     # nu antrenăm, doar inferență

def apply_mask(img, mask):
    """
    Aplică masca asupra imaginii → ROI cardiac
    """
    mask = (mask > 0.5).float()
    mask = mask.repeat(1, 3, 1, 1)  # ✅ match RGB
    return img * mask


# =============================
# EFFICIENTNET-B0 – FEATURE EXTRACTION
# =============================

class FeatureExtractor(nn.Module):
    """
    CNN pentru extragere de caracteristici (1280D)
    """
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Identity()    # păstrăm doar feature extractor

    def forward(self, x):
        return self.model(x)

# Inițializare model feature extraction
feature_extractor = FeatureExtractor().to(DEVICE)
feature_extractor.eval()

# =============================
# SLICE FILTERING
# =============================

def slice_score(roi):
    """
    Heuristic: deviația standard ≈ informație utilă
    """
    return roi.std().item()

# =============================
# FEATURE VECTORS PER SLICE (1280D)
# =============================

def extract_features(dataset):
    """
    Pipeline complet pe slice:
    MRI → segmentare → ROI → features → filtrare
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_features, all_labels, all_patients = [], [], []

    with torch.no_grad():
        for imgs, labels, patients in tqdm(loader):

            imgs = imgs.to(DEVICE)

            # Segmentare
            masks = unet(imgs)

            # ROI
            roi_imgs = apply_mask(imgs, masks)

            # Feature extraction
            feats = feature_extractor(roi_imgs)

            # Filtrare slice-uri slabe
            scores = torch.std(roi_imgs, dim=[1,2,3])
            keep = scores > scores.median()

            feats = feats[keep]
            labels = labels[keep.cpu().numpy()]
            patients = np.array(patients)[keep.cpu().numpy()]

            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels)
            all_patients.extend(patients)

    return np.vstack(all_features), np.array(all_labels), np.array(all_patients)

# =============================
# PATIENT-LEVEL AGGREGATION – BAYESIAN FUSION
# =============================

def bayesian_fusion(slice_probs):
    """
    Combina probabilitățile slice-urilor într-o
    probabilitate finală per pacient
    """
    eps = 1e-6
    slice_probs = np.clip(slice_probs, eps, 1-eps)
    log_odds = np.log(slice_probs / (1 - slice_probs))
    return 1 / (1 + np.exp(-log_odds.mean()))

def aggregate_patient(features, labels, patients, clf):
    """
    Slice-level → patient-level prediction
    """
    patient_probs, patient_labels = {}, {}

    slice_probs = clf.predict_proba(features)[:,1]

    for prob, label, patient in zip(slice_probs, labels, patients):
        patient_probs.setdefault(patient, []).append(prob)
        patient_labels[patient] = label

    X, y = [], []
    for p in patient_probs:
        X.append(bayesian_fusion(np.array(patient_probs[p])))
        y.append(patient_labels[p])

    return np.array(X), np.array(y)

# =============================
# FINAL CLASSIFIER – CAD PREDICTION
# =============================

# Separăm pacienți Normal / Sick
patients = list(set([s[2] for s in samples]))
normal = [p for p in patients if p.startswith("Normal")]
sick = [p for p in patients if p.startswith("Sick")]

# Split manual (dataset mic)
train_patients = normal[:2] + sick[:2]
test_patients  = normal[2:4] + sick[2:4]

# Construim seturile
train_samples = [s for s in samples if s[2] in train_patients]
test_samples  = [s for s in samples if s[2] in test_patients]

train_samples = train_samples[0:11]
test_samples = test_samples[0:11]

train_ds = MRIDataset(train_samples, transform)
test_ds  = MRIDataset(test_samples, transform)

# Feature extraction
X_train, y_train, p_train = extract_features(train_ds)
X_test,  y_test,  p_test  = extract_features(test_ds)

# Clasificator final
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Agregare per pacient + evaluare
Xp_test, yp_test = aggregate_patient(X_test, y_test, p_test, clf)
auc = roc_auc_score(yp_test, Xp_test)

print("PATIENT-LEVEL AUC:", auc)
print("Done!")
