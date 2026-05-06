# %% ============================================================
# 🧠 CAD Detection from Cardiac MRI – Full Pipeline (Single File)
# ============================================================

# Pipeline:
# MRI slice (2D)
#    ↓
# Attention U-Net (segmentare inimă)
#    ↓
# Masked ROI
#    ↓
# EfficientNet-B0 (pretrained)
#    ↓
# 1280D slice feature
#    ↓
# Slice filtering
#    ↓
# Bayesian fusion (patient-level)
#    ↓
# Logistic Regression / SVM


# =============================
# IMPORTS
# =============================

import os  # OS interaction (files, paths)
import cv2  # Image processing (MRI loading)
import numpy as np  # Numerical operations
from tqdm import tqdm  # Progress bar

import torch  # Deep Learning framework
import torch.nn as nn  # Neural networks
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision import models  # Pretrained CNNs

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch.nn.functional as F  # Functional ops (upsampling)

# =============================
# CONFIGURATION
# =============================

IMG_SIZE = 224  # CNN input size (standard pentru modele ImageNet)
BATCH_SIZE = 8  # Batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset path (Kaggle vs local)
if os.path.exists("/kaggle/input"):
    DATASET_PATH = "/kaggle/input"
else:
    DATASET_PATH = r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset'


# =============================
# PIPELINE STEP 1
# MRI slice (2D) – DATASET
# =============================

class MRIDataset(Dataset):
    """
    PIPELINE STEP 1:
    Loads MRI slices (2D images)
    Returns: (image, label, patient_id)
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient = self.samples[idx]

        # ✅ Load MRI slice (grayscale)
        # MRI-urile sunt de obicei stocate ca imagini în tonuri de gri (1 canal)
        # deoarece informația este bazată pe intensitate (semnal RMN), nu pe culoare
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # ✅ Resize to CNN input size
        # Redimensionarea normalizează dimensiunile tuturor imaginilor
        # Important pentru batching și pentru compatibilitatea cu CNN-uri preantrenate
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # ✅ Convert 1-channel → 3-channel (ImageNet compatibility)
        # Rețelele preantrenate (ResNet, EfficientNet) așteaptă 3 canale (RGB)
        # Aici replicăm canalul grayscale pe 3 canale (nu adăugăm informație nouă)
        img = np.stack([img] * 3, axis=-1)

        # ✅ Transformări suplimentare:
        # - conversie la tensor
        # - normalizare (important pentru stabilitatea rețelei)
        if self.transform:
            img = self.transform(img)

        return img, label, patient


# =============================
# LOAD DATA
# =============================

def load_samples(root_dir):
    """
    Loads data from:
    Normal/Patient/...
    Sick/Patient/...

    Structura ierarhică permite:
    - etichetare automată (Normal vs Sick)
    - grupare pe pacient (important pentru agregare ulterioară)
    """
    samples = []

    for class_name in ["Normal", "Sick"]:
        label = 0 if class_name == "Normal" else 1
        class_path = os.path.join(root_dir, class_name)

        for patient in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient)

            # ID unic per pacient (folosit ulterior la fusion)
            patient_id = f"{class_name}_{patient}"

            for root, _, files in os.walk(patient_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        samples.append(
                            (os.path.join(root, f), label, patient_id)
                        )

    return samples


samples = load_samples(DATASET_PATH)

# =============================
# PREPROCESSING
# =============================

transform = transforms.Compose([
    # ✅ Conversie imagine -> tensor PyTorch (HWC -> CHW, valori [0,1])
    transforms.ToTensor(),

    # ✅ Normalizare:
    # Centering datelor în jurul valorii 0 și scalare → stabilitate numerică
    # Valorile [0.5, 0.5, 0.5] sunt aproximative pentru date generice
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


# =============================
# PIPELINE STEP 2
# Attention U-Net – HEART SEGMENTATION
# =============================

class AttentionBlock(nn.Module):
    """
    Attention Gate: focuses on heart region

    Rol:
    - suprimă zone irelevante (plămâni, fundal)
    - evidențiază structuri relevante (miocard, ventriculi)
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Conv2d(F_int, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # combină context global (g) cu informație locală (x)
        psi = self.relu(self.W_g(g) + self.W_x(x))

        # output între 0 și 1 → hartă de atenție (probabilitate)
        psi = self.sigmoid(self.psi(psi))

        # multiplicare → filtrare spațială
        return x * psi


class AttentionUNet(nn.Module):
    """
    PIPELINE STEP 2:
    Segment heart from MRI

    Output:
    - masca binară (probabilistică) a inimii
    """

    def __init__(self):
        super().__init__()

        # Encoder preantrenat → extrage caracteristici relevante
        self.encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder.fc = nn.Identity()

        # Reducere dimensională + pregătire pentru segmentare
        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)

        self.att = AttentionBlock(256, 256, 128)

        # Output: 1 canal = probabilitate pixel apartine inimii
        self.conv_out = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # ✅ Encoder
        # extrage caracteristici ierarhice (low-level → high-level)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # ✅ Attention
        g = self.conv1(x)
        x = self.att(g, g)

        # ✅ Segmentation mask
        x = self.conv_out(x)

        # sigmoid → probabilitate pixel ∈ [0,1]
        x = torch.sigmoid(x)

        # ✅ Resize to original image
        # important: aliniază masca cu imaginea inițială (pentru ROI corect)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


# Initialize segmentation model
unet = AttentionUNet().to(DEVICE)
unet.eval()


# =============================
# PIPELINE STEP 3
# Masked ROI
# =============================

def apply_mask(img, mask):
    """
    Apply segmentation mask → ROI extraction

    ROI = Region of Interest (zona inimii)
    """

    # ✅ Binarizare mască:
    # transformă probabilitățile în 0/1
    mask = (mask > 0.5).float()

    # ✅ Extindem masca la 3 canale pentru compatibilitate cu imaginea
    mask = mask.repeat(1, 3, 1, 1)

    # ✅ Keep only heart region:
    # Pixelii din afara inimii devin 0 → reducere zgomot + focus pe regiune relevantă
    return img * mask


# =============================
# PIPELINE STEP 4
# EfficientNet-B0 – FEATURE EXTRACTION
# =============================

class FeatureExtractor(nn.Module):
    """
    Extracts 1280D feature vector per slice

    Reprezentare compactă a informației vizuale:
    - textură miocard
    - forme
    - variații de intensitate
    """

    def __init__(self):
        super().__init__()

        # Model preantrenat pe ImageNet → transfer learning
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Eliminăm layerul de clasificare
        self.model.classifier = nn.Identity()

    def forward(self, x):
        # ✅ Output: vector 1280D (embedding imagine)
        return self.model(x)


feature_extractor = FeatureExtractor().to(DEVICE)
feature_extractor.eval()


# =============================
# PIPELINE STEP 5 + 6
# Feature Extraction + Slice Filtering
# =============================

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_features, all_labels, all_patients = [], [], []

    with torch.no_grad():
        for imgs, labels, patients in tqdm(loader):
            imgs = imgs.to(DEVICE)

            # ✅ STEP 2: segmentation
            masks = unet(imgs)

            # ✅ STEP 3: ROI
            roi_imgs = apply_mask(imgs, masks)

            # ✅ STEP 4: feature extraction (1280D)
            feats = feature_extractor(roi_imgs)

            # =============================
            # PIPELINE STEP 5
            # Slice filtering
            # =============================

            # ✅ Variabilitate intensitate (std dev):
            # slice-urile informative au variații mai mari (structură vizibilă)
            scores = torch.std(roi_imgs, dim=[1, 2, 3])

            # ✅ Filtrăm slice-urile slabe:
            # ex: regiuni fără inimă sau cu semnal slab
            keep = scores > scores.median()

            feats = feats[keep]
            labels = labels[keep.cpu().numpy()]
            patients = np.array(patients)[keep.cpu().numpy()]

            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels)
            all_patients.extend(patients)

    return np.vstack(all_features), np.array(all_labels), np.array(all_patients)


# =============================
# PIPELINE STEP 7
# Bayesian Fusion (patient-level)
# =============================

def bayesian_fusion(slice_probs):
    eps = 1e-6

    # Evităm log(0)
    slice_probs = np.clip(slice_probs, eps, 1 - eps)

    # ✅ Convert probabilities → log-odds
    # transformare utilă pentru combinarea probabilităților independente
    log_odds = np.log(slice_probs / (1 - slice_probs))

    # ✅ Aggregate evidence (media log-odds)
    return 1 / (1 + np.exp(-log_odds.mean()))


def aggregate_patient(features, labels, patients, clf):
    patient_probs, patient_labels = {}, {}
    slice_probs = clf.predict_proba(features)[:, 1]

    for prob, label, patient in zip(slice_probs, labels, patients):
        patient_probs.setdefault(patient, []).append(prob)
        patient_labels[patient] = label

    X, y = [], []
    for p in patient_probs:
        X.append(bayesian_fusion(np.array(patient_probs[p])))
        y.append(patient_labels[p])

    return np.array(X), np.array(y)


# =============================
# PIPELINE STEP 8
# Final Classifier
# =============================

patients = list(set([s[2] for s in samples]))

normal = [p for p in patients if p.startswith("Normal")]
sick = [p for p in patients if p.startswith("Sick")]

train_patients = normal[:1] + sick[:1]
test_patients = normal[1:2] + sick[1:2]

train_samples = [s for s in samples if s[2] in train_patients]
test_samples = [s for s in samples if s[2] in test_patients]

train_ds = MRIDataset(train_samples, transform)
test_ds = MRIDataset(test_samples, transform)

# ✅ Extract features
X_train, y_train, p_train = extract_features(train_ds)
X_test, y_test, p_test = extract_features(test_ds)

# ✅ Logistic Regression (can replace with SVM easily)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ✅ Patient-level prediction
Xp_test, yp_test = aggregate_patient(X_test, y_test, p_test, clf)

# ✅ Evaluation
auc = roc_auc_score(yp_test, Xp_test)

print("PATIENT-LEVEL AUC:", auc)
print("Done!")
