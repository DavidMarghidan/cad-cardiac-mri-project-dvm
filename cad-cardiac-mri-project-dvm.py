#%% ============================================================
# 🧠 CAD Detection from Cardiac MRI – Full Series-Level Pipeline (Single File)
# ============================================================

# ============================================================================
# OVERVIEW
# ============================================================================
#
# This script implements a FULL END-TO-END CAD (Coronary Artery Disease)
# series-level classification pipeline using Cardiac MRI slices.
#
# The architecture combines:
#
#   1. Deep Learning segmentation
#   2. ROI extraction
#   3. Deep feature extraction
#   4. Classical Machine Learning classification
#   5. Series-level probabilistic fusion
#
# The pipeline is intentionally modular so every stage can later be replaced
# independently:
#
#   - Segmentation model → U-Net / Attention U-Net / SAM / MedSAM
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
# Attention U-Net (segmentare inimă)
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
# Bayesian series fusion
#    ↓
# Logistic Regression / SVM
#    ↓
# Series-level CAD-associated prediction
#
# ============================================================================
# WHY THIS PIPELINE?
# ============================================================================
#
# Medical MRI datasets are usually:
#
#   - Small
#   - Noisy
#   - Highly variable between acquisitions and series
#   - Difficult to annotate
#
# Therefore:
#
#   - Transfer learning improves generalization
#   - Segmentation reduces background noise
#   - Slice filtering removes uninformative views
#   - Series-level aggregation stabilizes predictions
#
# The pipeline mimics radiologist reasoning:
#
#   "Inspect multiple slices from one series → combine evidence → classify the series"
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
        (image, label, series_id)

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

    Later we aggregate slice predictions into series-level predictions.
    """

    def __init__(self, samples, transform=None):

        self.samples = samples
        # List containing:
        #   (img_path, label, series_id)

        self.transform = transform
        # Preprocessing pipeline:
        #   tensor conversion
        #   normalization
        #   augmentation (optional)

    def __len__(self):

        return len(self.samples)
        # Total number of MRI slices

    def __getitem__(self, idx):

        img_path, label, series = self.samples[idx]

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

        return img, label, series


# =============================
# LOAD DATA
# =============================

def load_samples(root_dir):
    """
    Expected structure:

        Normal/
            Directory_1/
                series0001-Body/
                series0002-Body/
                ...

        Sick/
            Directory_17/
                SR_1/
                SR_2/
                ...

    =========================================================================
    IMPORTANT DESIGN CHOICE
    =========================================================================

    We organize samples PER FOLDER-SERIES because the public JPEG release
    does not expose a reliable patient identifier.

    Therefore:
        classification is SERIES-LEVEL
        NOT patient-level.

    For leakage-resistant evaluation, all folder-series from the same
    Directory_* are assigned to the same train/test partition.

    This prevents:
        - exact duplicate images from crossing train/test boundaries
        - closely related series from the same acquisition directory
          being evaluated as independent train/test observations

    =========================================================================
    WHY THIS MATTERS
    =========================================================================

    If duplicate or strongly related series from the SAME Directory_* appear in:
        train AND test

    then:
        the model may memorize acquisition/export-specific patterns,
        producing unrealistically high performance.
    """

    samples = []
    series_to_directory = {}

    for class_name in ["Normal", "Sick"]:

        # Binary encoding
        label = 0 if class_name == "Normal" else 1

        class_path = os.path.join(root_dir, class_name)

        for directory in sorted(os.listdir(class_path)):

            directory_path = os.path.join(class_path, directory)

            if not os.path.isdir(directory_path):
                continue

            # Build globally unique acquisition-directory ID
            directory_id = f"{class_name}/{directory}"

            for root, _, files in os.walk(directory_path):

                if not any(
                    f.lower().endswith((".png", ".jpg", ".jpeg"))
                    for f in files
                ):
                    continue

                series_id = os.path.relpath(root, root_dir).replace(os.sep, "/")
                series_to_directory[series_id] = directory_id

                for f in sorted(files):

                    if f.lower().endswith((".png", ".jpg", ".jpeg")):

                        # Store:
                        #   image path
                        #   class label
                        #   series identifier
                        #
                        samples.append(
                            (os.path.join(root, f), label, series_id)
                        )

    return samples, series_to_directory


samples, series_to_directory = load_samples(DATASET_PATH)


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
# Attention U-Net – HEART SEGMENTATION
# =============================

class AttentionBlock(nn.Module):
    """
    Attention Gate

    Goal:
        Focus only on relevant anatomical regions.

    =========================================================================
    WHY ATTENTION?
    =========================================================================

    MRI slices contain:
        - lungs
        - ribs
        - fat
        - scanner artifacts
        - background

    Attention suppresses irrelevant spatial regions and highlights:
        - myocardium
        - ventricles
        - atria

    =========================================================================
    CONCEPT
    =========================================================================

    Produces spatial weights in [0,1].

    High weight:
        important region

    Low weight:
        irrelevant region
    """

    def __init__(self, F_g, F_l, F_int):

        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, 1)
        # Gating signal projection

        self.W_x = nn.Conv2d(F_l, F_int, 1)
        # Local feature projection

        self.psi = nn.Conv2d(F_int, 1, 1)
        # Final attention map generator

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):

        # Combine global + local context
        psi = self.relu(self.W_g(g) + self.W_x(x))

        # Convert activations → probabilities
        psi = self.sigmoid(self.psi(psi))

        # Apply spatial weighting
        return x * psi


# =============================
# Attention UNet
# =============================

class AttentionUNet(nn.Module):
    """
    PIPELINE STEP 2:
    Heart segmentation network

    Output:
        Soft probability segmentation mask

    =========================================================================
    WHY SEGMENTATION FIRST?
    =========================================================================

    Feature extraction on full MRI slices introduces:
        - irrelevant anatomy
        - noise
        - scanner artifacts

    Segmentation constrains the classifier to:
        HEART REGION ONLY

    This usually improves:
        - robustness
        - explainability
        - generalization
    """

    def __init__(self):

        super().__init__()

        self.encoder = models.resnet18(weights="IMAGENET1K_V1")

        # Remove classification head
        self.encoder.fc = nn.Identity()

        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        # Feature refinement

        self.att = AttentionBlock(256, 256, 128)
        # Spatial attention gate

        self.conv_out = nn.Conv2d(256, 1, 1)
        # Final 1-channel segmentation mask

    def forward(self, x):

        input_size = x.shape[-2:]

        # =====================================================
        # ENCODER
        # =====================================================

        # Initial feature extraction
        x = self.encoder.conv1(x)

        # Batch normalization stabilizes activations
        x = self.encoder.bn1(x)

        # Non-linear activation
        x = self.encoder.relu(x)

        # Spatial downsampling
        x = self.encoder.maxpool(x)

        # Residual feature extraction
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)

        # Deep semantic features
        x = self.encoder.layer4(x)

        # =====================================================
        # ATTENTION
        # =====================================================

        g = self.conv1(x)

        x = self.att(g, g)

        # =====================================================
        # SEGMENTATION HEAD
        # =====================================================

        x = self.conv_out(x)

        # Convert logits → probabilities
        x = torch.sigmoid(x)

        # =====================================================
        # UPSAMPLING
        # =====================================================

        # Resize mask back to original resolution
        #
        # Bilinear interpolation:
        #   smooth
        #   computationally efficient
        #
        x = F.interpolate(
            x,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )

        return x


# Initialize segmentation model
unet = AttentionUNet().to(DEVICE)

# Inference mode:
#   disables dropout
#   freezes batchnorm updates
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

    all_features, all_labels, all_series = [], [], []

    # Disable gradient computation
    #
    # Benefits:
    #   - lower memory usage
    #   - faster inference
    #
    with torch.no_grad():

        for batch_idx, (imgs, labels, series) in enumerate(tqdm(loader)):

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

            if debug and batch_idx == 0:

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

            series = np.array(series)[keep.cpu().numpy()]

            all_features.append(feats.cpu().numpy())

            all_labels.extend(labels)

            all_series.extend(series)

    return (
        np.vstack(all_features),
        np.array(all_labels),
        np.array(all_series)
    )


# =============================
# PIPELINE STEP 7
# Bayesian Fusion (series-level)
# =============================

def bayesian_fusion(slice_probs):
    """
    Combines slice probabilities into ONE series-level probability.

    =========================================================================
    WHY FUSION?
    =========================================================================

    The evaluation target in this public release is series-level.

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


def aggregate_series(features, labels, series, clf):
    """
    Aggregates slice-level predictions into series-level predictions.
    """

    series_probs, series_labels = {}, {}

    # Slice-level probabilities
    slice_probs = clf.predict_proba(features)[:,1]

    for prob, label, series_id in zip(slice_probs, labels, series):

        series_probs.setdefault(series_id, []).append(prob)

        series_labels[series_id] = label

    X, y = [], []

    for series_id in series_probs:

        # Bayesian aggregation
        X.append(
            bayesian_fusion(np.array(series_probs[series_id]))
        )

        y.append(series_labels[series_id])

    return np.array(X), np.array(y)


# =============================
# PIPELINE STEP 8
# Final Classifier
# =============================

# =============================================================
# BUILD SERIES / DIRECTORY LIST
# =============================================================

series = sorted(set([s[2] for s in samples]))

directories = sorted(set(series_to_directory.values()))
normal_directories = [d for d in directories if d.startswith("Normal/")]
sick_directories = [d for d in directories if d.startswith("Sick/")]

rng = np.random.RandomState(42)
normal_directories = list(rng.permutation(normal_directories))
sick_directories = list(rng.permutation(sick_directories))

normal_test_count = max(1, int(round(len(normal_directories) * 0.2)))
sick_test_count = max(1, int(round(len(sick_directories) * 0.2)))

# =============================================================
# TRAIN / TEST SPLIT
# =============================================================

# IMPORTANT:
# Split is DIRECTORY-GROUPED for SERIES-LEVEL evaluation.
#
# NEVER split by slices or individual series directly because exact duplicate
# images were observed between series inside the same Directory_*.
#
train_directories = (
    normal_directories[normal_test_count:]
    + sick_directories[sick_test_count:]
)

test_directories = (
    normal_directories[:normal_test_count]
    + sick_directories[:sick_test_count]
)

train_series = [
    series_id for series_id in series
    if series_to_directory[series_id] in train_directories
]

test_series = [
    series_id for series_id in series
    if series_to_directory[series_id] in test_directories
]

train_samples = [
    s for s in samples
    if s[2] in train_series
]

test_samples = [
    s for s in samples
    if s[2] in test_series
]

# Optional debugging subset
# train_samples = train_samples[0:100]
# test_samples = test_samples[0:100]

train_ds = MRIDataset(train_samples, transform)
test_ds  = MRIDataset(test_samples, transform)

# =============================================================
# FEATURE EXTRACTION
# =============================================================

X_train, y_train, s_train = extract_features(
    train_ds,
    debug=True
)

X_test, y_test, s_test = extract_features(
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
# SERIES-LEVEL PREDICTION
# =============================================================

Xs_test, ys_test = aggregate_series(
    X_test,
    y_test,
    s_test,
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
auc = roc_auc_score(ys_test, Xs_test)

print("SERIES-LEVEL AUC:", auc)

print("Done!")
