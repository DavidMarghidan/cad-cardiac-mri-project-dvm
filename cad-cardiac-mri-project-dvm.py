#%% ============================================================
# 🧠 CAD Detection from Cardiac MRI – Full Pipeline (Single File)
# ============================================================

# ============================================================================
# OVERVIEW
# ============================================================================
#
# This script implements a FULL END-TO-END CAD (Coronary Artery Disease)
# detection pipeline using Cardiac MRI slices.
#
# The architecture combines:
#
#   1. Deep Learning segmentation
#   2. ROI extraction
#   3. Deep feature extraction
#   4. Classical Machine Learning classification
#   5. Patient-level probabilistic fusion
#
# The pipeline is intentionally modular so every stage can later be replaced
# independently:
#
#   - Segmentation model → MedSAM Pretrained 2D model / U-Net / SAM / MedSAM
#   - Backbone → EfficientNet / ConvNeXt / ViT
#   - Fusion → Bayesian / Mean / Attention pooling
#   - Final classifier → LR / SVM / XGBoost
#
# ============================================================================
# PIPELINE FLOW
# ============================================================================
#
# MRI slice (2D)
#    ↓
# MedSAM Pretrained 2D segmentation model (segmentare inimă)
#    ↓
# Soft probability mask
#    ↓
# ROI extraction (heart isolation)
#    ↓
# EfficientNet-B0 (ImageNet pretrained)
#    ↓
# 1280D feature vector / slice
#    ↓
# Slice quality filtering
#    ↓
# Bayesian patient fusion
#    ↓
# Logistic Regression / SVM
#    ↓
# Patient-level CAD prediction
#
# ============================================================================
# WHY THIS PIPELINE?
# ============================================================================
#
# Medical MRI datasets are usually:
#
#   - Small
#   - Noisy
#   - Highly variable between patients
#   - Difficult to annotate
#
# Therefore:
#
#   - Transfer learning improves generalization
#   - Segmentation reduces background noise
#   - Slice filtering removes uninformative views
#   - Patient-level aggregation stabilizes predictions
#
# The pipeline mimics radiologist reasoning:
#
#   "Inspect multiple slices → combine evidence → decide diagnosis"
#
# ============================================================================


# =============================
# IMPORTS
# =============================

import os
# OS interaction (files, paths)
# Used for:
#   - traversing dataset folders
#   - building portable file paths
#   - checking execution environment (Kaggle vs local)

import cv2
# OpenCV image processing library
# Efficient for:
#   - grayscale MRI loading
#   - image resizing
#   - low-level pixel manipulation

import numpy as np
# Core numerical computation library
# Used heavily for:
#   - matrix operations
#   - probability fusion
#   - feature aggregation
#   - tensor-like preprocessing

from tqdm import tqdm
# Progress visualization utility
# Helpful for:
#   - monitoring extraction progress
#   - debugging long-running inference pipelines

import torch
# PyTorch Deep Learning framework
# Provides:
#   - GPU acceleration
#   - automatic differentiation
#   - tensor computations

import torch.nn as nn
# Neural network building blocks:
#   - convolution layers
#   - activation functions
#   - modules

from torch.utils.data import Dataset, DataLoader
# Dataset utilities:
#   - batching
#   - shuffling
#   - multiprocessing data loading

import torchvision.transforms as transforms
# Image preprocessing utilities:
#   - tensor conversion
#   - normalization
#   - augmentation (optional)

from torchvision import models
# Access to pretrained ImageNet architectures:
#   - ResNet
#   - EfficientNet
#   - ViT
# etc.

from sklearn.linear_model import LogisticRegression
# Classical ML classifier
# Used after deep feature extraction

from sklearn.metrics import roc_auc_score
# Evaluation metric:
# ROC-AUC measures ranking quality
# Extremely important for medical binary classification

import torch.nn.functional as F
# Functional API
# Used here mainly for:
#   - interpolation
#   - resizing masks

import matplotlib.pyplot as plt
# Visualization utility for:
#   - debugging masks
#   - ROI inspection
#   - sanity checks


from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download


# =============================
# CONFIGURATION
# =============================

IMG_SIZE = 224
# CNN input size
#
# Why 224?
#   - Standard ImageNet resolution
#   - Compatible with pretrained EfficientNet/ResNet
#   - Good trade-off between:
#       detail retention
#       GPU memory usage
#
# WARNING:
# Excessive resizing may distort anatomy.

BATCH_SIZE = 8
# Number of slices processed simultaneously
#
# Trade-off:
#   Larger batch:
#       + faster GPU utilization
#       - higher VRAM usage
#
#   Smaller batch:
#       + lower memory usage
#       - slower training/inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Automatically select:
#
#   CUDA GPU → if available
#   CPU      → fallback
#
# GPU acceleration is critical for:
#   - segmentation
#   - feature extraction
#   - large datasets

# Dataset path (Kaggle vs local)
if os.path.exists("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"):
    # Detect Kaggle execution environment
    DATASET_PATH = "/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"
else:
    # Local Windows path
    DATASET_PATH = r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset'


# =============================
# PIPELINE STEP 1
# MRI slice (2D) – DATASET
# =============================

class MRIDataset(Dataset):
    """
    PIPELINE STEP 1:
    Loads MRI slices (2D images)

    Returns:
        (image, label, patient_id)

    =========================================================================
    MEDICAL IMAGING CONTEXT
    =========================================================================

    Each MRI slice represents a 2D cross-sectional view of the thoracic area.

    Important observations:

        - Some slices contain full heart anatomy
        - Some contain only partial heart
        - Some may contain almost no useful anatomy

    This motivates later:
        → segmentation
        → slice filtering

    =========================================================================
    WHY SLICE-LEVEL PROCESSING?
    =========================================================================

    MRI volumes are 3D, but:
        - 2D CNNs are simpler
        - Require less GPU memory
        - Easier to train on small datasets

    Later we aggregate slice predictions into patient-level predictions.
    """

    def __init__(self, samples, transform=None):

        self.samples = samples
        # List containing:
        #   (img_path, label, patient_id)

        self.transform = transform
        # Preprocessing pipeline:
        #   tensor conversion
        #   normalization
        #   augmentation (optional)

    def __len__(self):

        return len(self.samples)
        # Total number of MRI slices

    def __getitem__(self, idx):

        img_path, label, patient = self.samples[idx]

        # =========================================================
        # LOAD MRI SLICE
        # =========================================================

        # ✅ MRI images are grayscale by nature
        #
        # Pixel intensities encode:
        #   - tissue density
        #   - relaxation properties
        #   - proton behavior
        #
        # Unlike RGB images:
        #   intensity has PHYSICAL meaning.
        #
        # IMPORTANT:
        # MRI intensity is NOT standardized between scanners.
        #
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # =========================================================
        # RESIZE
        # =========================================================

        # CNNs require fixed spatial dimensions.
        #
        # Resizing ensures:
        #   - batch compatibility
        #   - consistent receptive fields
        #   - stable transfer learning behavior
        #
        # Potential downside:
        #   anatomical distortion if aspect ratio changes.
        #
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # =========================================================
        # CONVERT 1-CHANNEL → 3-CHANNEL
        # =========================================================

        # ImageNet pretrained models expect RGB input.
        #
        # Since MRI is grayscale:
        #   we replicate the same channel 3 times.
        #
        # NOTE:
        # This DOES NOT create new information.
        #
        img = np.stack([img] * 3, axis=-1)

        # =========================================================
        # PREPROCESSING
        # =========================================================

        # Includes:
        #   - tensor conversion
        #   - normalization
        #
        if self.transform:
            img = self.transform(img)

        return img, label, patient


# =============================
# LOAD DATA
# =============================

def load_samples(root_dir):
    """
    Expected structure:

        Normal/
            Patient001/
            Patient002/

        Sick/
            Patient001/
            Patient002/

    =========================================================================
    IMPORTANT DESIGN CHOICE
    =========================================================================

    We organize samples PER PATIENT because:

        diagnosis is PATIENT-LEVEL
        NOT slice-level.

    This prevents:
        - data leakage
        - train/test contamination

    =========================================================================
    WHY THIS MATTERS
    =========================================================================

    If slices from the SAME patient appear in:
        train AND test

    then:
        model memorizes patient-specific patterns,
        producing unrealistically high performance.
    """

    samples = []

    for class_name in ["Normal", "Sick"]:

        # Binary encoding
        label = 0 if class_name == "Normal" else 1

        class_path = os.path.join(root_dir, class_name)

        for patient in os.listdir(class_path):

            patient_path = os.path.join(class_path, patient)

            # Build globally unique patient ID
            patient_id = f"{class_name}_{patient}"

            for root, _, files in os.walk(patient_path):

                for f in files:

                    if f.lower().endswith((".png", ".jpg", ".jpeg")):

                        # Store:
                        #   image path
                        #   class label
                        #   patient identifier
                        #
                        samples.append(
                            (os.path.join(root, f), label, patient_id)
                        )

    return samples


samples = load_samples(DATASET_PATH)


# =============================
# PREPROCESSING
# =============================

transform = transforms.Compose([

    transforms.ToTensor(),
    # Converts:
    #   HWC → CHW
    #
    # Also rescales:
    #   [0,255] → [0,1]

    transforms.Normalize([0.5]*3, [0.5]*3)
    # Normalize intensities:
    #
    # Formula:
    #   x_norm = (x - mean) / std
    #
    # Here:
    #   mean = 0.5
    #   std  = 0.5
    #
    # Result:
    #   range becomes approximately [-1,1]
    #
    # Benefits:
    #   - faster convergence
    #   - stable gradients
    #   - improved transfer learning
])

# =============================
# PIPELINE STEP 2
# MEDSAM PRETRAINED SEGMENTATION MODEL
# =============================


class MedSAMPretrainedSegmentationModel(nn.Module):
    """
    PIPELINE STEP 2:
    MedSAM-based cardiac ROI extraction.

    =========================================================================
    WHY MEDSAM?
    =========================================================================

    The original ACDC segmentation network was trained exclusively on ACDC
    cardiac MRI data.

    The CAD Cardiac MRI Dataset contains substantially different image
    characteristics:

        - larger field of view
        - coronal acquisitions
        - surrounding thoracic organs
        - different scanner settings

    As a consequence, the ACDC model often predicts only background.

    MedSAM is a foundation segmentation model built upon Segment Anything and
    adapted to medical imaging.

    It generalizes significantly better to unseen medical datasets.

    =========================================================================
    INPUT
    =========================================================================

        [B,3,224,224]

    identical to the original pipeline.

    =========================================================================
    OUTPUT
    =========================================================================

        [B,1,224,224]

    soft ROI mask.

    =========================================================================
    """

    def __init__(self):
        super().__init__()

        self.model_dir = os.path.join(
            "pretrained_models",
            "MedSAM"
        )

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )



        # checkpoint_path = hf_hub_download(
        #     repo_id="wanglab/medsam-vit-b",
        #     filename="medsam_vit_b.pth",
        #     local_dir=self.model_dir
        # )

        # =========================================================
        # LOAD LOCAL MEDSAM CHECKPOINT
        # =========================================================

        checkpoint_path = "./pretrained_models/MedSAM/medsam_vit_b.pth"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"MedSAM checkpoint not found:\n{checkpoint_path}"
            )

        # =========================================================
        # CREATE SAM MODEL
        # =========================================================

        self.sam = sam_model_registrycheckpoint = checkpoint_path

        self.sam.to(DEVICE)

        self.sam.eval()

        self.predictor = SamPredictor(
            self.sam
        )

    def forward(self, x):
        batch_masks = []

        for img_tensor in x:
            # =====================================================
            # REVERSE NORMALIZATION
            # =====================================================

            img = img_tensor.detach().cpu()

            img = img * 0.5 + 0.5

            img = img.permute(
                1,
                2,
                0
            ).numpy()

            img = np.clip(
                img,
                0,
                1
            )

            img = (
                    img * 255
            ).astype(
                np.uint8
            )

            # =====================================================
            # MEDSAM IMAGE ENCODING
            # =====================================================

            self.predictor.set_image(img)

            h, w = img.shape[:2]

            # =====================================================
            # AUTOMATIC CARDIAC BOX PROMPT
            # =====================================================

            #
            # The CAD dataset usually places the
            # heart near the central thoracic area.
            #
            # A box prompt produces much better
            # masks than a single positive point.
            #

            input_box = np.array([
                w * 0.25,
                h * 0.05,
                w * 0.75,
                h * 0.60
            ])

            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=True
            )

            best_mask = masks[
                np.argmax(scores)
            ]

            best_mask = torch.tensor(
                best_mask,
                dtype=torch.float32
            )

            best_mask = best_mask.unsqueeze(0)

            batch_masks.append(
                best_mask
            )

        batch_masks = torch.stack(
            batch_masks,
            dim=0
        )

        batch_masks = batch_masks.to(
            DEVICE
        )

        return batch_masks


unet = MedSAMPretrainedSegmentationModel().to(
    DEVICE
)

unet.eval()


# =============================
# PIPELINE STEP 3
# Masked ROI
# =============================

def apply_mask(img, mask):
    """
    Apply segmentation mask → ROI extraction

    =========================================================================
    ROI = REGION OF INTEREST
    =========================================================================

    Goal:
        isolate cardiac anatomy
        suppress irrelevant tissue

    =========================================================================
    WHY ROI EXTRACTION?
    =========================================================================

    CNNs are sensitive to:
        - irrelevant texture
        - scanner borders
        - anatomy outside target organ

    ROI masking improves:
        - signal-to-noise ratio
        - feature quality
        - explainability

    =========================================================================
    OUTPUT
    =========================================================================

    Returns:
        image where only heart pixels remain visible
    """

    # Convert soft probabilities → binary mask
    mask = (mask > 0.5).float()

    # Duplicate mask across RGB channels
    mask = mask.repeat(1, 3, 1, 1)

    # Zero-out non-heart pixels
    return img * mask


def denormalize(img):

    # Reverse normalization for visualization
    img = img * 0.5 + 0.5

    return img.clamp(0, 1)


def debug_visualization(
    imgs,
    masks,
    roi_imgs,
    scores,
    batch_idx,
    max_show=1
):
    """
    Visualization helper for debugging.

    Saves:
        - original image
        - segmentation mask
        - extracted ROI

    Useful for:
        - verifying segmentation quality
        - detecting preprocessing bugs
        - sanity checking filtering logic
    """

    imgs = imgs.cpu()
    masks = masks.cpu()
    roi_imgs = roi_imgs.cpu()
    scores = scores.cpu()

    os.makedirs("debug_output", exist_ok=True)

    for i in range(min(max_show, imgs.shape[0])):

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # =====================================================
        # ORIGINAL IMAGE
        # =====================================================

        ax[0].imshow(
            denormalize(imgs[i]).permute(1, 2, 0).numpy()
        )

        ax[0].set_title("Original")
        ax[0].axis("off")

        # =====================================================
        # SEGMENTATION MASK
        # =====================================================

        ax[1].imshow(masks[i][0], cmap='gray')

        ax[1].set_title("Mask")
        ax[1].axis("off")

        # =====================================================
        # ROI
        # =====================================================

        ax[2].imshow(
            denormalize(roi_imgs[i]).permute(1, 2, 0).numpy()
        )

        ax[2].set_title(f"ROI (std={scores[i]:.3f})")
        ax[2].axis("off")

        plt.tight_layout()

        plt.savefig(
            f"debug_output/batch{batch_idx}_img{i}.png"
        )

        plt.close()


# =============================
# PIPELINE STEP 4
# EfficientNet-B0 – FEATURE EXTRACTION
# =============================

class FeatureExtractor(nn.Module):
    """
    Extracts deep feature embeddings from ROI images.

    Output:
        1280-dimensional feature vector

    =========================================================================
    WHAT ARE FEATURES?
    =========================================================================

    Learned representations encoding:
        - texture
        - anatomical structure
        - shape
        - intensity distribution
        - pathological patterns

    =========================================================================
    WHY TRANSFER LEARNING?
    =========================================================================

    Medical datasets are usually small.

    Using ImageNet-pretrained networks:
        - accelerates convergence
        - improves generalization
        - reduces overfitting

    Even though ImageNet is natural-image based,
    early CNN filters remain highly useful.
    """

    def __init__(self):

        super().__init__()

        self.model = models.efficientnet_b0(
            weights="IMAGENET1K_V1"
        )

        # Remove classifier head
        self.model.classifier = nn.Identity()

    def forward(self, x):

        # Returns embedding vector
        return self.model(x)


feature_extractor = FeatureExtractor().to(DEVICE)
feature_extractor.eval()


# =============================
# PIPELINE STEP 5 + 6
# Feature Extraction + Slice Filtering
# =============================

def extract_features(dataset, debug=False):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_features, all_labels, all_patients = [], [], []

    # Disable gradient computation
    #
    # Benefits:
    #   - lower memory usage
    #   - faster inference
    #
    with torch.no_grad():

        for batch_idx, (imgs, labels, patients) in enumerate(tqdm(loader)):

            imgs = imgs.to(DEVICE)

            # =====================================================
            # STEP 2: SEGMENTATION
            # =====================================================

            masks = unet(imgs)

            # =====================================================
            # STEP 3: ROI EXTRACTION
            # =====================================================

            roi_imgs = apply_mask(imgs, masks)

            # =====================================================
            # STEP 4: FEATURE EXTRACTION
            # =====================================================

            feats = feature_extractor(roi_imgs)

            # =====================================================
            # PIPELINE STEP 5
            # SLICE FILTERING
            # =====================================================

            # Compute image intensity variability
            #
            # Intuition:
            #
            # Low standard deviation:
            #   → flat / empty image
            #   → weak anatomical information
            #
            # High standard deviation:
            #   → richer anatomical structure
            #
            scores = torch.std(roi_imgs, dim=[1,2,3])

            # =====================================================
            # DEBUG VISUALIZATION
            # =====================================================

            if debug: # and batch_idx == 0:

                debug_visualization(
                    imgs,
                    masks,
                    roi_imgs,
                    scores,
                    batch_idx
                )

            # =====================================================
            # KEEP MOST INFORMATIVE SLICES
            # =====================================================

            # Retain only slices above median variability
            #
            # This removes:
            #   - near-empty slices
            #   - poor quality scans
            #   - low-information anatomy
            #
            keep = scores > scores.median()

            feats = feats[keep]

            labels = labels[keep.cpu().numpy()]

            patients = np.array(patients)[keep.cpu().numpy()]

            all_features.append(feats.cpu().numpy())

            all_labels.extend(labels)

            all_patients.extend(patients)

    return (
        np.vstack(all_features),
        np.array(all_labels),
        np.array(all_patients)
    )


# =============================
# PIPELINE STEP 7
# Bayesian Fusion (patient-level)
# =============================

def bayesian_fusion(slice_probs):
    """
    Combines slice probabilities into ONE patient probability.

    =========================================================================
    WHY FUSION?
    =========================================================================

    CAD diagnosis is patient-level.

    A single slice may be:
        - noisy
        - ambiguous
        - incomplete

    Combining multiple slices improves robustness.

    =========================================================================
    WHY LOG-ODDS?
    =========================================================================

    Direct averaging of probabilities is unstable.

    Log-odds aggregation:
        - accumulates evidence better
        - handles confidence more naturally
        - behaves more probabilistically
    """

    eps = 1e-6

    # Prevent numerical instability
    slice_probs = np.clip(slice_probs, eps, 1-eps)

    # Convert probability → log-odds
    #
    # log(p / (1-p))
    #
    log_odds = np.log(slice_probs / (1 - slice_probs))

    # Average evidence and reconvert to probability
    return 1 / (1 + np.exp(-log_odds.mean()))


def aggregate_patient(features, labels, patients, clf):
    """
    Aggregates slice-level predictions into patient-level predictions.
    """

    patient_probs, patient_labels = {}, {}

    # Slice-level probabilities
    slice_probs = clf.predict_proba(features)[:,1]

    for prob, label, patient in zip(slice_probs, labels, patients):

        patient_probs.setdefault(patient, []).append(prob)

        patient_labels[patient] = label

    X, y = [], []

    for p in patient_probs:

        # Bayesian aggregation
        X.append(
            bayesian_fusion(np.array(patient_probs[p]))
        )

        y.append(patient_labels[p])

    return np.array(X), np.array(y)


# =============================
# PIPELINE STEP 8
# Final Classifier
# =============================

# =============================================================
# BUILD PATIENT LIST
# =============================================================

patients = list(set([s[2] for s in samples]))

normal = [p for p in patients if p.startswith("Normal")]
sick = [p for p in patients if p.startswith("Sick")]

# =============================================================
# TRAIN / TEST SPLIT
# =============================================================

# IMPORTANT:
# Split is PATIENT-LEVEL.
#
# NEVER split by slices directly.
#
train_patients = normal[:1] + sick[:1]

test_patients  = normal[1:2] + sick[1:2]

train_samples = [
    s for s in samples
    if s[2] in train_patients
]

test_samples = [
    s for s in samples
    if s[2] in test_patients
]

# Optional debugging subset
# train_samples = train_samples[0:100]
# test_samples = test_samples[0:100]

train_ds = MRIDataset(train_samples, transform)
test_ds  = MRIDataset(test_samples, transform)

# =============================================================
# FEATURE EXTRACTION
# =============================================================

X_train, y_train, p_train = extract_features(
    train_ds,
    debug=True
)

X_test, y_test, p_test = extract_features(
    test_ds,
    debug=True
)

# =============================================================
# CLASSICAL ML CLASSIFIER
# =============================================================

# Logistic Regression chosen because:
#
#   + simple
#   + interpretable
#   + robust on small datasets
#   + fast training
#
# Can easily replace with:
#   - SVM
#   - XGBoost
#   - Random Forest
#
clf = LogisticRegression(max_iter=1000)

clf.fit(X_train, y_train)

# =============================================================
# PATIENT-LEVEL PREDICTION
# =============================================================

Xp_test, yp_test = aggregate_patient(
    X_test,
    y_test,
    p_test,
    clf
)

# =============================================================
# EVALUATION
# =============================================================

# ROC-AUC:
#
#   1.0 → perfect
#   0.5 → random guessing
#
# In medical imaging:
# ROC-AUC is preferred over accuracy because:
#   - datasets are often imbalanced
#   - ranking quality matters
#
auc = roc_auc_score(yp_test, Xp_test)

print("PATIENT-LEVEL AUC:", auc)

print("Done!")