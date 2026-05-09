#%% ============================================================
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

import os                                # OS interaction (files, paths) → used to traverse directories and manage dataset structure
import cv2                               # Image processing (MRI loading) → OpenCV is efficient for reading grayscale medical images
import numpy as np                       # Numerical operations → core library for matrix/image manipulation
from tqdm import tqdm                    # Progress bar → useful for monitoring long dataset iterations

import torch                             # Deep Learning framework → tensor computations and GPU acceleration
import torch.nn as nn                    # Neural networks → layers and modules
from torch.utils.data import Dataset, DataLoader  # Data utilities → batching, shuffling, parallel loading

import torchvision.transforms as transforms        # Preprocessing utilities → normalization, tensor conversion
from torchvision import models           # Pretrained CNNs → ImageNet-based feature extractors

from sklearn.linear_model import LogisticRegression   # Final classical ML classifier
from sklearn.metrics import roc_auc_score             # Evaluation metric → sensitive to ranking quality

import torch.nn.functional as F          # Functional ops (upsampling) → used for resizing segmentation masks

import matplotlib.pyplot as plt

# =============================
# CONFIGURATION
# =============================

IMG_SIZE = 224                           # CNN input size → standard for ImageNet models (ensures compatibility)
BATCH_SIZE = 8                           # Batch size → trade-off between memory usage and training speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # Automatically select GPU if available

# Dataset path (Kaggle vs local)
if os.path.exists("/kaggle/input"):      # Detect if running in Kaggle environment
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

    IMPORTANT (Medical Imaging Context):
    - Each slice is a cross-sectional view of the heart
    - Slices may contain partial or no heart → motivates later filtering
    """

    def __init__(self, samples, transform=None):
        self.samples = samples            # List of (path, label, patient)
        self.transform = transform        # Image preprocessing pipeline

    def __len__(self):
        return len(self.samples)          # Total number of slices

    def __getitem__(self, idx):
        img_path, label, patient = self.samples[idx]   # Extract sample metadata

        # ✅ Load MRI slice (grayscale)
        # MRI images encode tissue intensity (not color), so grayscale is correct representation
        # Pixel intensity ~ proton density / relaxation properties
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # ✅ Resize to CNN input size
        # Standardization is critical:
        # - CNNs require fixed resolution
        # - spatial scale consistency improves generalization
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # ✅ Convert 1-channel → 3-channel (ImageNet compatibility)
        # Pretrained networks expect RGB-like input (3 channels)
        # Replication does NOT create new information, only adapts format
        img = np.stack([img] * 3, axis=-1)

        # ✅ Apply transformations:
        # Includes normalization → shifts intensity distribution
        if self.transform:
            img = self.transform(img)

        return img, label, patient        # Return tensor + metadata

# =============================
# LOAD DATA
# =============================

def load_samples(root_dir):
    """
    Loads data from:
    Normal/Patient/...
    Sick/Patient/...

    Image Processing Insight:
    - Organizing by patient allows slice aggregation later
    - Important because diagnosis is patient-level, not slice-level
    """
    samples = []

    for class_name in ["Normal", "Sick"]:   # Binary classification
        label = 0 if class_name == "Normal" else 1   # Encode label numerically
        class_path = os.path.join(root_dir, class_name)

        for patient in os.listdir(class_path):   # Iterate patients
            patient_path = os.path.join(class_path, patient)

            patient_id = f"{class_name}_{patient}"   # Unique global ID

            for root, _, files in os.walk(patient_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        # Store full path + metadata
                        samples.append(
                            (os.path.join(root, f), label, patient_id)
                        )

    return samples

samples = load_samples(DATASET_PATH)   # Load all dataset slices

# =============================
# PREPROCESSING
# =============================

transform = transforms.Compose([
    transforms.ToTensor(),                     # Convert to tensor → also scales [0,255] to [0,1]

    # Normalize
    # Centers distribution around 0, improves gradient flow
    # Important in medical imaging due to varying intensity distributions
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =============================
# PIPELINE STEP 2
# Attention U-Net – HEART SEGMENTATION
# =============================

class AttentionBlock(nn.Module):
    """
    Attention Gate: focuses on heart region

    Image Processing Insight:
    - MRI slices contain noise, ribs, lungs, background
    - Attention suppresses irrelevant spatial areas
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)    # Linear projection of gating signal
        self.W_x = nn.Conv2d(F_l, F_int, 1)    # Linear projection of input feature map
        self.psi = nn.Conv2d(F_int, 1, 1)      # Produces attention map
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # Combine global and local features
        psi = self.relu(self.W_g(g) + self.W_x(x))

        # Convert to attention weights [0,1]
        psi = self.sigmoid(self.psi(psi))

        return x * psi   # Element-wise masking of features

# =============================
# Attention UNet
# =============================

class AttentionUNet(nn.Module):
    """
    PIPELINE STEP 2:
    Segment heart from MRI

    Produces:
    - Soft segmentation mask (probability map)
    """

    def __init__(self):
        super().__init__()

        self.encoder = models.resnet18(weights="IMAGENET1K_V1")   # Pretrained encoder
        self.encoder.fc = nn.Identity()   # Remove classification layer

        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)  # Feature refinement
        self.att = AttentionBlock(256, 256, 128)        # Attention gating
        self.conv_out = nn.Conv2d(256, 1, 1)            # Output mask

    def forward(self, x):
        input_size = x.shape[-2:]   # Save original resolution

        # ✅ Encoder feature extraction
        x = self.encoder.conv1(x)   # Initial convolution
        x = self.encoder.bn1(x)     # Normalize activations
        x = self.encoder.relu(x)    # Non-linearity
        x = self.encoder.maxpool(x) # Downsampling

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)  # Deep semantic features

        # ✅ Attention
        g = self.conv1(x)
        x = self.att(g, g)

        # ✅ Segmentation mask
        x = self.conv_out(x)        # Linear projection
        x = torch.sigmoid(x)        # Pixel-wise probability

        # ✅ Resize mask back to input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x

# Initialize segmentation model
unet = AttentionUNet().to(DEVICE)   # Move to GPU/CPU
unet.eval()                         # Inference mode (no dropout, no BN updates)

# =============================
# PIPELINE STEP 3
# Masked ROI
# =============================

def apply_mask(img, mask):
    """
    Apply segmentation mask → ROI extraction

    Image Processing Insight:
    - Removes irrelevant tissue/background
    - Forces model to focus on myocardium and chambers
    """

    mask = (mask > 0.5).float()   # Threshold → binary segmentation

    mask = mask.repeat(1, 3, 1, 1)   # Match RGB channels

    # ✅ Keep only heart region
    return img * mask   # Zero out non-heart pixels



def denormalize(img):
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)


def debug_visualization(imgs, masks, roi_imgs, scores, batch_idx, max_show=1):
    imgs = imgs.cpu()
    masks = masks.cpu()
    roi_imgs = roi_imgs.cpu()
    scores = scores.cpu()

    os.makedirs("debug_output", exist_ok=True)

    for i in range(min(max_show, imgs.shape[0])):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # Original
        ax[0].imshow(denormalize(imgs[i]).permute(1, 2, 0).numpy())
        ax[0].set_title("Original")
        ax[0].axis("off")

        # Mask
        ax[1].imshow(masks[i][0], cmap='gray')
        ax[1].set_title("Mask")
        ax[1].axis("off")

        # ROI
        ax[2].imshow(denormalize(roi_imgs[i]).permute(1, 2, 0).numpy())
        ax[2].set_title(f"ROI (std={scores[i]:.3f})")
        ax[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"debug_output/batch{batch_idx}_img{i}.png")
        plt.close()



# =============================
# PIPELINE STEP 4
# EfficientNet-B0 – FEATURE EXTRACTION
# =============================

class FeatureExtractor(nn.Module):
    """
    Extracts 1280D feature vector per slice

    Image Processing Insight:
    - Encodes texture, shape, spatial patterns
    - Learned from large-scale ImageNet dataset
    """

    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")  # Load pretrained weights
        self.model.classifier = nn.Identity()   # Remove classification head

    def forward(self, x):
        return self.model(x)   # Output embedding vector

feature_extractor = FeatureExtractor().to(DEVICE)
feature_extractor.eval()

# =============================
# PIPELINE STEP 5 + 6
# Feature Extraction + Slice Filtering
# =============================

def extract_features(dataset, debug=False):

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_features, all_labels, all_patients = [], [], []

    with torch.no_grad():   # Disable gradients → faster inference
        for batch_idx, (imgs, labels, patients) in enumerate(tqdm(loader)):

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

            # Compute intensity variability
            # Low std → flat image (likely no heart or poor quality)
            scores = torch.std(roi_imgs, dim=[1,2,3])


            # ✅ DEBUG VISUAL (doar primele batch-uri)
            if debug and batch_idx == 0:
                debug_visualization(imgs, masks, roi_imgs, scores, batch_idx)

            # ✅ Keep informative slices only
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

    slice_probs = np.clip(slice_probs, eps, 1-eps)

    # ✅ Convert probabilities → log-odds
    log_odds = np.log(slice_probs / (1 - slice_probs))

    # ✅ Aggregate evidence
    return 1 / (1 + np.exp(-log_odds.mean()))

def aggregate_patient(features, labels, patients, clf):

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
# PIPELINE STEP 8
# Final Classifier
# =============================

patients = list(set([s[2] for s in samples]))

normal = [p for p in patients if p.startswith("Normal")]
sick = [p for p in patients if p.startswith("Sick")]

train_patients = normal[:1] + sick[:1]
test_patients  = normal[1:2] + sick[1:2]

train_samples = [s for s in samples if s[2] in train_patients]
test_samples  = [s for s in samples if s[2] in test_patients]

# train_samples = train_samples[0:100]
# test_samples = test_samples[0:100]

train_ds = MRIDataset(train_samples, transform)
test_ds  = MRIDataset(test_samples, transform)

# ✅ Extract features
X_train, y_train, p_train = extract_features(train_ds, debug=True)
X_test,  y_test,  p_test  = extract_features(test_ds, debug=True)


# ✅ Logistic Regression (can replace with SVM easily)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ✅ Patient-level prediction
Xp_test, yp_test = aggregate_patient(X_test, y_test, p_test, clf)

# ✅ Evaluation
auc = roc_auc_score(yp_test, Xp_test)

print("PATIENT-LEVEL AUC:", auc)
print("Done!")