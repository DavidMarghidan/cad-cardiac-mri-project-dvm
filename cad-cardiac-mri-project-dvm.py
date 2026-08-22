#%% ============================================================
# 🧠 CAD Detection from Cardiac MRI – MONAI Patient-Level Pipeline (Single File)
# ============================================================

# ============================================================================
# OVERVIEW
# ============================================================================
#
# This script implements a FULL END-TO-END CAD (Coronary Artery Disease)
# patient-level classification pipeline using 2D Cardiac MRI JPEG images from:
#
#   CAD Cardiac MRI Dataset
#   https://www.kaggle.com/datasets/danialsharifrazi/cad-cardiac-mri-dataset/data
#
# PATIENT IDENTIFIER USED BY THIS IMPLEMENTATION (VALIDATED FOR THIS RELEASE):
#
#   patient_id = Directory_*
#
# The parent folder supplies the patient-level class label:
#
#   Normal/Directory_* -> label 0
#   Sick/Directory_*   -> label 1
#
# IMPORTANT: SR_* / series* folders are NOT treated as patients. They are
# imaging-series containers belonging to the Directory_* patient. All images
# and all series from one Directory_* remain together in every train/validation
# split and are fused back to one final patient-level prediction.
#
# The architecture combines:
#
#   1. Patient-level grouping and leakage-safe evaluation
#   2. Pretrained MONAI cardiac ventricular segmentation (SAX-specific)
#   3. Confidence-gated soft ROI extraction with full-image fallback
#   4. ImageNet-pretrained EfficientNet-B0 feature extraction
#   5. Optional series-local slice-quality weighting (disabled by default)
#   6. Hierarchically balanced Logistic Regression on frozen embeddings
#      (class → patient → series → slice)
#   7. Hierarchical probabilistic fusion: slices → series → patient
#   8. Stratified K-fold evaluation defined directly on Directory_* patients
#   9. Pooled out-of-fold AUC with a patient-level bootstrap confidence interval
#
# IMPORTANT METHODOLOGICAL CHANGES:
# The original version used one 80/20 split and discarded roughly half of the
# slices using a standard-deviation cutoff. This revision instead:
#
#   - performs cross-validation on the validated Directory_* patient units;
#   - retains every readable slice by default;
#   - keeps the standard-deviation score only as an OPTIONAL heuristic ablation;
#   - balances training so that a patient with many series/slices cannot dominate;
#   - fits StandardScaler with the SAME hierarchical sample weights as the
#     classifier, avoiding a subtle slice-count bias in feature standardization;
#   - reports a patient-level bootstrap CI around the pooled OOF ROC-AUC.
#
# The Scientific Reports paper reports 1,224 original participants, but this
# script does NOT infer the number of computational patient units from that
# paper-level count. Under the validated mapping requested here, the number of
# patients used by the code is exactly the number of discovered Directory_*
# folders containing images. This distinction is important for correct claims
# about sample size and statistical uncertainty.
#
# The segmentation stage uses the official MONAI Model Zoo bundle:
#
#   ventricular_short_axis_3label, version 0.3.5
#
# The bundle contains a pretrained 2D residual U-Net (MONAI UNet with residual
# units) that produces four output channels:
#
#   0 = background
#   1 = left-ventricular blood pool
#   2 = left-ventricular myocardium
#   3 = right-ventricular blood pool
#
# IMPORTANT DOMAIN LIMITATION:
# The MONAI model was trained for 2D short-axis cardiac MR images. The CAD
# dataset also contains heterogeneous series, including long-axis images,
# localizers, derived exports, and possibly other acquisition types. For that
# reason, this script does NOT trust every segmentation unconditionally.
# A plausibility gate checks mask size and confidence. If a mask is implausible,
# the full image is used instead of a potentially destructive ROI mask.
#
# Required packages:
#
#   pip install monai==1.6.0 huggingface_hub
#   pip install torch torchvision opencv-python numpy scikit-learn matplotlib tqdm
#
# NOTE ABOUT THE PRETRAINED SEGMENTER:
# The MONAI bundle metadata identifies version 0.3.5 and records the original
# bundle environment as MONAI 1.3.0 / PyTorch 1.13.0. The network definition is
# verified from the official bundle configuration. The current script can use a
# newer MONAI version to build/load the network, but the exact checkpoint and
# architecture should be kept pinned for reproducibility.
#
# On first execution, the script downloads the pinned MONAI bundle if needed,
# imports MONAI once, and exports the segmenter to a local TorchScript cache.
# Later executions load that cache directly and do not import MONAI at startup.
# Set AUTO_DOWNLOAD_MONAI_BUNDLE = False if the original bundle must be supplied
# manually or the machine has no internet access.
#
# ============================================================================
# PIPELINE FLOW
# ============================================================================
#
# Raw MRI JPEG slice
#    ↓
# Min-max intensity scaling to [0,1]
#    ↓
# Aspect-ratio-preserving zero padding to 256×256
#    ↓
# Pretrained MONAI residual U-Net
#    ↓
# Four-class ventricular probability map
#    ↓
# Cardiac probability = LV pool + myocardium + RV pool
#    ↓
# Mask plausibility / confidence check
#    ↓
# Confidence-gated soft ROI
#    ↓
# EfficientNet-B0 ImageNet normalization
#    ↓
# EfficientNet-B0 feature encoding
#    ↓
# 1280D feature vector / readable slice
#    ↓
# Logistic Regression slice probabilities
#    ↓
# Log-odds fusion across readable slices inside each imaging series
#    ↓
# Log-odds fusion across all series belonging to one Directory_* patient
#    ↓
# Patient-level CAD-associated probability
#
# ============================================================================
# WHY THIS PIPELINE?
# ============================================================================
#
# Medical MRI datasets are usually:
#
#   - Small relative to natural-image datasets
#   - Noisy
#   - Heterogeneous across scanners, protocols, views, and exports
#   - Difficult and expensive to annotate
#
# Therefore:
#
#   - A cardiac-MRI-pretrained segmenter is preferable to a randomly
#     initialized segmentation head.
#   - Soft ROI weighting reduces irrelevant anatomy without deleting all
#     contextual information.
#   - Confidence gating protects the pipeline when the pretrained segmenter is
#     applied outside its original short-axis domain.
#   - Transfer learning improves feature quality when labeled CAD data are
#     limited.
#   - All slices are retained by default; optional quality weights are computed
#     only inside their own series and are treated as a heuristic, not a clinical
#     image-quality measurement.
#   - Hierarchical aggregation prevents a very long series from dominating the
#     patient simply because it contains more exported JPEG frames.
#   - Hierarchical training weights give equal nominal influence to patients and
#     equal nominal influence to the series inside each patient.
#   - Patient-level splitting prevents images or series from the same
#     Directory_* patient from appearing in both training and validation folds.
#
# The pipeline approximates the following reasoning process:
#
#   "Localize cardiac anatomy when reliable → inspect informative slices from
#    every series → combine series evidence → classify the patient."
#
# This is an exploratory patient-level classifier under the validated dataset
# mapping that every Directory_* is one patient. The top-level Normal/Sick
# folder supplies the patient-level class label. When that label is propagated
# to every slice during slice-level classifier training, the slice supervision
# is weak because individual slices are not independently annotated for CAD.
#
# ============================================================================


# =============================
# IMPORTS
# =============================

import hashlib
# Used to verify the SHA-256 checksum of the downloaded MONAI checkpoint.
# Checksum verification makes the experiment more reproducible and helps detect
# corrupted or unintended model files.

import os
# OS interaction (files, paths, environment variables).
# Used for:
#   - traversing dataset folders
#   - building portable paths
#   - detecting Kaggle versus local execution
#   - reading optional bundle path overrides

from pathlib import Path
# Object-oriented path manipulation.
# Used to manage the MONAI bundle directory and checkpoint files safely.

import cv2
# OpenCV image processing library.
# Used for:
#   - grayscale JPEG loading
#   - aspect-ratio-preserving resizing
#   - creating fixed-size inputs for MONAI and EfficientNet

import numpy as np
# Core numerical computation library.
# Used for:
#   - image preprocessing
#   - probability fusion
#   - feature and label arrays
#   - deterministic patient shuffling

from tqdm import tqdm
# Progress visualization utility.
# Useful for monitoring segmentation and feature extraction over many slices.

import torch
# PyTorch Deep Learning framework.
# Provides:
#   - tensor computation
#   - GPU acceleration
#   - model inference

import torch.nn as nn
# Neural-network module API.
# Used for the EfficientNet feature-extractor wrapper.

import torch.nn.functional as F
# Functional tensor operations.
# Used for:
#   - softmax over MONAI output channels
#   - mask dilation with max pooling
#   - mask resizing from 256×256 to 224×224

from torch.utils.data import Dataset, DataLoader
# Dataset utilities for batching and deterministic inference.

import torchvision.transforms as transforms
# Converts NumPy HWC arrays to PyTorch CHW tensors.
# EfficientNet normalization is intentionally applied AFTER ROI extraction.

from torchvision import models
# Provides ImageNet-pretrained EfficientNet-B0.

# IMPORTANT STARTUP OPTIMIZATION:
#
# MONAI is intentionally NOT imported at module startup.
#
# On the first run, build_monai_segmenter() imports MONAI lazily, constructs
# the official pretrained UNet, and exports it once to a TorchScript cache.
# On all later runs, the segmentation network is loaded directly with
# torch.jit.load(), so neither monai.networks.nets nor monai.bundle.scripts is
# imported at startup. This removes the slow MONAI import from normal runs.
#
# The MONAI bundle downloader remains lazy as well and is imported only if the
# original model.pt checkpoint is missing.

from sklearn.linear_model import LogisticRegression
# Classical ML classifier trained on EfficientNet slice embeddings.

from sklearn.metrics import roc_auc_score
# ROC-AUC evaluation metric for binary patient-level ranking.

from sklearn.model_selection import StratifiedKFold
# Cross-validation is defined on a table containing exactly one row per
# Directory_* patient. Because slices are mapped to folds only AFTER patients
# are split, an additional group variable is unnecessary here.

from sklearn.pipeline import Pipeline
# Bundles feature standardization and Logistic Regression so the scaler is fit
# ONLY on the training patients of each fold.

from sklearn.preprocessing import StandardScaler
# Standardizes the 1280-dimensional EfficientNet embeddings inside each fold.

import matplotlib.pyplot as plt
# Visualization utility for inspecting:
#   - the padded input
#   - the MONAI cardiac probability map
#   - the confidence-gated soft ROI


# =============================
# CONFIGURATION
# =============================

IMG_SIZE = 224
# EfficientNet-B0 input resolution.
#
# The original JPEG is first placed inside a 256×256 zero-padded MONAI canvas.
# The complete canvas is then resized to 224×224 for EfficientNet. Because the
# whole square canvas is resized uniformly, the MONAI mask and EfficientNet
# image remain spatially aligned.

MONAI_INPUT_SIZE = 256
# Input size used to train the official ventricular_short_axis_3label bundle.
# Images smaller than this size are centered and zero-padded instead of being
# enlarged. This follows the bundle documentation, which states that many
# training images were smaller than 256×256 and were zero-padded.

BATCH_SIZE = 8
# Number of slices processed simultaneously.
#
# Both MONAI segmentation and EfficientNet inference are performed for every
# batch, so reduce this value if GPU memory is insufficient. Increase it only
# after checking GPU memory; batch size changes throughput, not predictions.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Automatically select CUDA when available; otherwise use CPU.

RANDOM_SEED = 42
# Fixed seed for repeatable fold assignment and bootstrap resampling. The frozen
# inference networks contain no dropout at evaluation time, but exact bitwise
# reproducibility can still depend on hardware/library kernels.

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

USE_SLICE_QUALITY_WEIGHTS = False
# IMPORTANT DEFAULT:
# Standard deviation after ROI weighting is NOT a validated MRI quality metric.
# Therefore the publication-safe default is equal slice weights. Set this True
# only for a predeclared ablation after verifying that the heuristic is useful.

SLICE_QUALITY_MIN_WEIGHT = 0.25
# Lower bound used only when USE_SLICE_QUALITY_WEIGHTS=True.

DEBUG_VISUALIZATION = True
DEBUG_MAX_BATCHES = 5
# Saving figures for every batch can create thousands of PNG files and dominate
# runtime. Enable visual QC deliberately; only the first DEBUG_MAX_BATCHES are
# saved during a full extraction run.

# ---------------------------------------------------------------------------
# MONAI BUNDLE CONFIGURATION
# ---------------------------------------------------------------------------

MONAI_BUNDLE_NAME = "ventricular_short_axis_3label"
# Official MONAI Model Zoo bundle used for cardiac segmentation.

MONAI_BUNDLE_VERSION = "0.3.5"
# Pinned bundle version for reproducibility.

MONAI_DOWNLOAD_SOURCE = "monaihosting"
# Official MONAI-hosted bundle source. Current MONAI releases resolve this
# source through the MONAI model hosting infrastructure.

AUTO_DOWNLOAD_MONAI_BUNDLE = True
# When True, the bundle is downloaded automatically if model.pt is missing.
# Set to False for an offline environment and copy the bundle manually to:
#
#   <MONAI_BUNDLE_DIR>/ventricular_short_axis_3label/models/model.pt

VERIFY_MONAI_CHECKPOINT_SHA256 = False
# No unverifiable model.pt digest is hard-coded. If you have an authoritative
# SHA-256 for the exact local checkpoint used in your experiment, place it in
# MONAI_MODEL_SHA256 and enable this flag. Otherwise reproducibility should be
# documented by bundle name, bundle version, software versions and the archived
# checkpoint itself.

MONAI_MODEL_SHA256 = ""
# Optional SHA-256 for a locally supplied model.pt. Leave empty to skip.

MONAI_ROI_DILATION_KERNEL = 31
# Expands the predicted ventricular structures to retain a margin around the
# myocardium. Must be an odd positive integer so output size remains unchanged.
# 17  → extindere mică
# 31  → extindere moderată
# 41  → extindere mare
# 51  → foarte mare

MONAI_BACKGROUND_WEIGHT = 0.15
# Soft ROI background retention.
#
# A value of 0 would remove all pixels outside the predicted cardiac region.
# That is risky under domain shift. A value of 0.15 keeps 15% of the original
# background signal while emphasizing the predicted heart region.
# 0.00 = exterior complet eliminat
# 0.15 = exterior foarte atenuat       ← actual
# 0.30 = păstrează destul context
# 0.40 = păstrează mult context
# 1.00 = practic fără ROI

MONAI_MIN_HEART_AREA_RATIO = 0.003
MONAI_MAX_HEART_AREA_RATIO = 0.50
MONAI_MIN_PEAK_HEART_PROBABILITY = 0.50
# Initial plausibility thresholds for deciding whether to trust a MONAI mask.
#
# These are safeguards, not clinically validated constants. They should be
# calibrated on a manually reviewed subset of this CAD dataset. A slice falls
# back to the full image when:
#   - the predicted heart is nearly empty,
#   - the predicted heart occupies implausibly much of the field of view, or
#   - no pixel receives sufficient non-background probability.

# ---------------------------------------------------------------------------
# EFFICIENTNET NORMALIZATION
# ---------------------------------------------------------------------------

EFFICIENTNET_MEAN = (0.485, 0.456, 0.406)
EFFICIENTNET_STD = (0.229, 0.224, 0.225)
# Official ImageNet normalization associated with torchvision's pretrained
# EfficientNet-B0 weights. It is applied after soft ROI extraction so masking
# operates on interpretable [0,1] image intensities.

# ---------------------------------------------------------------------------
# MONAI BUNDLE LOCATION
# ---------------------------------------------------------------------------

if os.path.exists("/kaggle/working"):
    default_bundle_parent = Path("/kaggle/working/monai_bundles")
else:
    try:
        default_bundle_parent = Path(__file__).resolve().parent / "monai_bundles"
    except NameError:
        # __file__ is unavailable in some notebook environments.
        default_bundle_parent = Path.cwd() / "monai_bundles"

MONAI_BUNDLE_DIR = Path(
    os.environ.get("MONAI_BUNDLE_DIR", str(default_bundle_parent))
)
# The MONAI_BUNDLE_DIR environment variable can override the default location.

MONAI_TORCHSCRIPT_PATH = Path(
    os.environ.get(
        "MONAI_TORCHSCRIPT_PATH",
        str(
            MONAI_BUNDLE_DIR
            / f"{MONAI_BUNDLE_NAME}_{MONAI_BUNDLE_VERSION}_torchscript.pt"
        ),
    )
)
# Fast-start cache of the complete pretrained segmentation network.
#
# First run:
#   model.pt -> lazy MONAI import -> construct UNet -> TorchScript export
#
# Later runs:
#   TorchScript cache -> torch.jit.load()
#
# MONAI itself is therefore not imported during normal later executions.

FORCE_REBUILD_MONAI_TORCHSCRIPT = False
# Set True only when intentionally rebuilding the cached TorchScript model,
# for example after changing the MONAI architecture or checkpoint version.

# ---------------------------------------------------------------------------
# DATASET LOCATION
# ---------------------------------------------------------------------------

if os.path.exists("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"):
    DATASET_PATH = "/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"
else:
    DATASET_PATH = r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset"


# =============================
# IMAGE PREPROCESSING HELPERS
# =============================

def scale_intensity_0_1(image):
    """
    Scale one grayscale MRI JPEG to float32 values in [0,1].

    MONAI's official training configuration applies ScaleIntensity to the input.
    JPEG images in this dataset do not share a physically standardized MRI
    intensity scale, so each image is normalized independently.

    A constant image is converted to zeros to avoid division by zero.
    """

    image = image.astype(np.float32)

    minimum = float(image.min())
    maximum = float(image.max())

    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)

    return (image - minimum) / (maximum - minimum)


def zero_pad_to_monai_canvas(image):
    """
    Place a 2D image inside a centered 256×256 MONAI input canvas.

    Design rules:

        1. Preserve aspect ratio.
        2. Do not enlarge images already smaller than 256×256.
        3. Fill unused pixels with zero.
        4. If an image is larger than 256×256, downscale it only enough to fit.

    The official bundle documentation states that training volumes had spatial
    size 256×256 and that smaller images were zero-padded; differing dimensions
    should otherwise be cropped or padded to that size. For this JPEG pipeline,
    isotropic downscaling of an oversized image is a deliberate adaptation that
    preserves the complete field of view. It is NOT identical to the official
    bundle preprocessing and should be reported if oversized inputs occur.
    """

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, received shape {image.shape}."
        )

    height, width = image.shape

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {image.shape}.")

    scale = min(
        1.0,
        MONAI_INPUT_SIZE / height,
        MONAI_INPUT_SIZE / width,
    )

    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))

    if resized_height != height or resized_width != width:
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    canvas = np.zeros(
        (MONAI_INPUT_SIZE, MONAI_INPUT_SIZE),
        dtype=np.float32,
    )

    top = (MONAI_INPUT_SIZE - resized_height) // 2
    left = (MONAI_INPUT_SIZE - resized_width) // 2

    canvas[
        top:top + resized_height,
        left:left + resized_width,
    ] = resized

    return canvas


# =============================
# PIPELINE STEP 1
# MRI SLICE (2D) – DATASET
# =============================

class MRIDataset(Dataset):
    """
    PIPELINE STEP 1:
    Load one JPEG MRI slice and construct two aligned inputs.

    Returns:
        classification_image:
            Tensor [3, 224, 224], values in [0,1].
            Used by EfficientNet after ROI extraction and ImageNet
            normalization.

        monai_image:
            Tensor [1, 256, 256], values in [0,1].
            Used by the pretrained MONAI cardiac segmenter.

        label:
            0 for Normal, 1 for Sick.

        patient_id:
            The validated patient identifier itself, for example Directory_24.

        series_id:
            Patient-scoped imaging-series identifier, for example
            Directory_24/SR_3.

    =========================================================================
    WHY TWO INPUT TENSORS?
    =========================================================================

    The two networks were trained with different input conventions:

        MONAI cardiac segmenter:
            - one grayscale channel
            - 256×256
            - intensity range [0,1]

        EfficientNet-B0:
            - three channels
            - 224×224
            - ImageNet mean/std normalization

    Reusing one already-normalized tensor for both networks would violate at
    least one model's expected input distribution. Therefore, segmentation and
    classification preprocessing are kept explicitly separate.

    =========================================================================
    SPATIAL ALIGNMENT
    =========================================================================

    The original MRI is first placed in a 256×256 square canvas. The 224×224
    classifier image is created by uniformly resizing that complete square.
    Consequently, a MONAI probability map resized from 256×256 to 224×224 aligns
    with the EfficientNet image without requiring DICOM geometry metadata.
    """

    def __init__(self, samples, transform=None):

        self.samples = samples
        # List containing:
        #   (image_path, binary_label, patient_id, series_id)

        self.transform = transform
        # Classifier-side conversion from NumPy HWC to PyTorch CHW.
        # ImageNet normalization is deliberately deferred until after ROI
        # weighting.

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label, patient_id, series_id = self.samples[idx]

        # =========================================================
        # LOAD RAW GRAYSCALE MRI JPEG
        # =========================================================

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(
                f"OpenCV could not read the MRI image: {img_path}"
            )

        # =========================================================
        # MONAI INTENSITY PREPROCESSING
        # =========================================================

        image = scale_intensity_0_1(image)

        # Preserve native size when possible and zero-pad to 256×256.
        monai_canvas = zero_pad_to_monai_canvas(image)

        # MONAI expects [channel, height, width] with one channel.
        monai_image = torch.from_numpy(monai_canvas).unsqueeze(0)

        # =========================================================
        # EFFICIENTNET SPATIAL PREPROCESSING
        # =========================================================

        # Resize the complete square canvas to EfficientNet resolution.
        # This retains exact alignment with the MONAI output mask.
        classification_gray = cv2.resize(
            monai_canvas,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # ImageNet-pretrained models expect three channels. Replicating a
        # grayscale channel does not add information, but makes the tensor
        # compatible with the pretrained first convolution.
        classification_image = np.stack(
            [classification_gray] * 3,
            axis=-1,
        )

        if self.transform:
            classification_image = self.transform(classification_image)
        else:
            classification_image = torch.from_numpy(
                classification_image
            ).permute(2, 0, 1)

        return (
            classification_image,
            monai_image,
            label,
            patient_id,
            series_id,
        )


# =============================
# LOAD DATA
# =============================

def load_samples(root_dir):
    """
    Discover MRI images and attach patient + series identifiers.

    DATASET COHORT VS. COMPUTATIONAL PATIENT UNIT
    ----------------------------------------------
    The accompanying Scientific Reports paper reports 1,224 original
    participants (722 healthy, 502 CAD) and 63,648 CMR images. Those paper-level
    numbers are descriptive metadata; they are NOT used by this function to
    manufacture patient IDs or expected folder counts.

    PATIENT UNIT -- VALIDATED FOR THIS DATASET RELEASE
    --------------------------------------------------
    ``patient_id`` is exactly the immediate ``Directory_*`` folder name.
    ``SR_*`` and ``series*`` child folders remain imaging-series containers and
    never become patients. The Normal/Sick parent folder supplies the class
    label but is not part of patient_id.

    The function explicitly rejects a Directory_* name that appears under both
    Normal and Sick, because patient_id must be globally unambiguous.

    SERIES UNIT
    -----------
    The first directory level below the patient is treated as the imaging
    series. This is safer than using ``os.walk(root)`` as the series definition:
    nested folders inside a series should not silently become separate series.

    Example:

        Sick/Directory_24/SR_3/image001.jpg
        Sick/Directory_24/SR_3/subfolder/image002.jpg

    Both images remain in the same series identifier:

        Directory_24/SR_3

    Images directly inside Directory_* receive the special series ID
    ``<patient_id>/__ROOT__``.

    The function returns:
        (image_path, binary_label, patient_id, series_id)
    """

    samples = []
    discovered_patients = set()
    patient_to_class = {}

    for class_name in ["Normal", "Sick"]:

        label = 0 if class_name == "Normal" else 1
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            raise FileNotFoundError(
                f"Missing expected class directory: {class_path}"
            )

        for directory in sorted(os.listdir(class_path)):

            patient_path = os.path.join(class_path, directory)

            if not os.path.isdir(patient_path):
                continue

            if not directory.lower().startswith("directory_"):
                continue

            # The user-validated patient unit is Directory_* itself.
            # Do NOT derive a patient from SR_*, series names, filenames or
            # duplicate components.
            patient_id = directory

            previous_class = patient_to_class.get(patient_id)
            if previous_class is not None and previous_class != class_name:
                raise RuntimeError(
                    f"Patient identifier {patient_id} appears under both "
                    f"{previous_class} and {class_name}. Directory_* must be "
                    "globally unique when it is used as patient_id."
                )

            patient_to_class[patient_id] = class_name
            discovered_patients.add(patient_id)

            # ---------------------------------------------------------
            # First collect images directly inside the patient folder.
            # ---------------------------------------------------------
            root_images = [
                filename
                for filename in sorted(os.listdir(patient_path))
                if filename.lower().endswith((".png", ".jpg", ".jpeg"))
                and os.path.isfile(os.path.join(patient_path, filename))
            ]

            root_series_id = f"{patient_id}/__ROOT__"

            for filename in root_images:
                samples.append(
                    (
                        os.path.join(patient_path, filename),
                        label,
                        patient_id,
                        root_series_id,
                    )
                )

            # ---------------------------------------------------------
            # Find immediate child directories. Each child directory is
            # treated as one imaging series. os.walk is still used to
            # collect images recursively inside that series.
            # ---------------------------------------------------------
            child_series_dirs = [
                child
                for child in sorted(os.listdir(patient_path))
                if os.path.isdir(os.path.join(patient_path, child))
            ]

            for series_name in child_series_dirs:

                series_path = os.path.join(patient_path, series_name)
                series_id = f"{patient_id}/{series_name}"

                found_images = False

                for root, _, files in os.walk(series_path):

                    for filename in sorted(files):

                        if not filename.lower().endswith(
                            (".png", ".jpg", ".jpeg")
                        ):
                            continue

                        found_images = True

                        samples.append(
                            (
                                os.path.join(root, filename),
                                label,
                                patient_id,
                                series_id,
                            )
                        )

                if not found_images:
                    # Empty series folders are ignored rather than becoming
                    # artificial series with zero observations.
                    continue

    if not samples:
        raise RuntimeError(
            f"No MRI images were discovered under dataset path: {root_dir}"
        )

    if not discovered_patients:
        raise RuntimeError(
            "No Directory_* patient folders were discovered. "
            "Verify the dataset path and folder structure."
        )

    # -------------------------------------------------------------
    # Sanity checks that should fail early rather than silently
    # contaminating a publication experiment.
    # -------------------------------------------------------------
    patient_to_label = {}

    for _, label, patient_id, _ in samples:

        previous = patient_to_label.get(patient_id)

        if previous is not None and previous != label:
            raise RuntimeError(
                f"Patient {patient_id} appears with both class labels."
            )

        patient_to_label[patient_id] = label

    print("Dataset discovery summary")
    print(f"  Patients: {len(patient_to_label)}")
    print(
        "  Normal patients:",
        sum(label == 0 for label in patient_to_label.values()),
    )
    print(
        "  Sick patients:",
        sum(label == 1 for label in patient_to_label.values()),
    )
    print(f"  Images: {len(samples)}")
    print(f"  Series: {len(set(sample[3] for sample in samples))}")

    return samples


# =============================
# CLASSIFIER-SIDE TENSOR CONVERSION
# =============================

transform = transforms.Compose([
    transforms.ToTensor(),
])
# ToTensor converts:
#
#   NumPy HWC → PyTorch CHW
#
# The source array is already float32 in [0,1], so no additional 255 division
# is needed. EfficientNet mean/std normalization is performed only after the
# MONAI-derived ROI has been applied.


# =============================
# PIPELINE STEP 2
# PRETRAINED MONAI CARDIAC SEGMENTATION
# =============================

def sha256_file(path):
    """Calculate a file's SHA-256 digest without loading it fully into memory."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def locate_monai_bundle_root():
    """
    Find the extracted bundle directory containing models/model.pt.

    The normal location is:

        MONAI_BUNDLE_DIR/
            ventricular_short_axis_3label/
                models/model.pt
                configs/train.json

    A recursive fallback is included because bundle download behavior can vary
    slightly between MONAI versions and storage sources.
    """

    direct_root = MONAI_BUNDLE_DIR / MONAI_BUNDLE_NAME
    direct_model = direct_root / "models" / "model.pt"

    if direct_model.is_file():
        return direct_root

    if MONAI_BUNDLE_DIR.exists():
        for model_path in MONAI_BUNDLE_DIR.rglob("model.pt"):
            candidate_root = model_path.parent.parent

            if (
                candidate_root.name == MONAI_BUNDLE_NAME
                or (candidate_root / "configs" / "train.json").is_file()
            ):
                return candidate_root

    return direct_root


def ensure_monai_bundle():
    """
    Ensure that the pinned pretrained MONAI bundle is available locally.

    The bundle is downloaded only when model.pt is absent. A clear error is
    raised when automatic download is disabled or fails.
    """

    bundle_root = locate_monai_bundle_root()
    model_path = bundle_root / "models" / "model.pt"

    if model_path.is_file():
        print(f"Using cached MONAI checkpoint: {model_path}")
        return bundle_root

    if not AUTO_DOWNLOAD_MONAI_BUNDLE:
        raise FileNotFoundError(
            "The MONAI checkpoint was not found and automatic download is "
            "disabled. Expected a bundle containing models/model.pt under: "
            f"{MONAI_BUNDLE_DIR}"
        )

    # Import the heavy MONAI downloader only when the checkpoint is absent.
    # This path is normally executed only on the first run.
    try:
        from monai.bundle.scripts import download as download_monai_bundle
    except ImportError as exc:
        raise ImportError(
            "The MONAI checkpoint is missing and the bundle downloader could "
            "not be imported. Install the downloader dependency with: "
            "pip install monai==1.6.0 huggingface_hub"
        ) from exc

    MONAI_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "MONAI checkpoint not found locally. Downloading bundle once: "
        f"{MONAI_BUNDLE_NAME} version {MONAI_BUNDLE_VERSION}..."
    )

    try:
        download_monai_bundle(
            name=MONAI_BUNDLE_NAME,
            version=MONAI_BUNDLE_VERSION,
            bundle_dir=str(MONAI_BUNDLE_DIR),
            source=MONAI_DOWNLOAD_SOURCE,
            progress=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Automatic MONAI bundle download failed. Verify internet access "
            "and install the optional downloader dependency with: "
            "pip install huggingface_hub. The equivalent CLI command is: "
            "python -m monai.bundle download "
            f"--name {MONAI_BUNDLE_NAME} "
            f"--version {MONAI_BUNDLE_VERSION} "
            f"--bundle_dir \"{MONAI_BUNDLE_DIR}\" "
            f"--source {MONAI_DOWNLOAD_SOURCE}"
        ) from exc

    bundle_root = locate_monai_bundle_root()
    model_path = bundle_root / "models" / "model.pt"

    if not model_path.is_file():
        raise FileNotFoundError(
            "MONAI reported a completed bundle download, but model.pt could "
            f"not be located under {MONAI_BUNDLE_DIR}."
        )

    print(f"MONAI bundle cached for future runs: {model_path}")
    return bundle_root


def load_checkpoint_state_dict(path):
    """
    Load an official PyTorch checkpoint as weights only.

    weights_only=True reduces the risk associated with arbitrary pickle object
    loading. The fallback exists for older PyTorch releases that do not expose
    the weights_only argument.
    """

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if isinstance(checkpoint, dict):
        # MONAI bundles commonly save the network through CheckpointSaver with
        # save_dict={"model": network}. Depending on Ignite/MONAI version, the
        # resulting file may therefore be either a raw state_dict or a mapping
        # containing a "model" key. Support both formats explicitly.
        for key in (
            "model",
            "state_dict",
            "model_state_dict",
            "network_state_dict",
        ):
            nested = checkpoint.get(key)

            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a state-dictionary checkpoint, got {type(checkpoint)}."
        )

    return checkpoint


def build_monai_segmenter():
    """
    Load the pretrained MONAI ventricular segmenter with a fast-start cache.

    NORMAL RUNS
    ------------
    If MONAI_TORCHSCRIPT_PATH already exists, load it directly with
    torch.jit.load(). No MONAI import is performed.

    FIRST RUN ONLY
    --------------
    If the TorchScript cache does not exist:

        1. Ensure the official MONAI bundle/checkpoint is present.
        2. Import MONAI UNet lazily.
        3. Build the exact official network architecture.
        4. Load the pretrained checkpoint.
        5. Trace and validate the network on the fixed 256x256 input used by
           this pipeline.
        6. Save a TorchScript model for all future runs.

    This keeps the same pretrained segmentation network while avoiding the
    expensive ``from monai.networks.nets import UNet`` import after the cache
    has been created once.
    """

    # =========================================================
    # FAST PATH: NO MONAI IMPORT
    # =========================================================

    if (
        MONAI_TORCHSCRIPT_PATH.is_file()
        and not FORCE_REBUILD_MONAI_TORCHSCRIPT
    ):
        try:
            network = torch.jit.load(
                str(MONAI_TORCHSCRIPT_PATH),
                map_location=DEVICE,
            )
            network.eval()

            print(
                "Loaded cached MONAI TorchScript segmenter (no MONAI import): "
                f"{MONAI_TORCHSCRIPT_PATH}"
            )

            return network

        except Exception as exc:
            print(
                "Cached MONAI TorchScript model could not be loaded. "
                "It will be rebuilt once from the official checkpoint. "
                f"Reason: {exc}"
            )

    # =========================================================
    # FIRST-RUN / REBUILD PATH
    # =========================================================

    bundle_root = ensure_monai_bundle()
    model_path = bundle_root / "models" / "model.pt"

    if VERIFY_MONAI_CHECKPOINT_SHA256 and MONAI_MODEL_SHA256:
        actual_hash = sha256_file(model_path)

        if actual_hash.lower() != MONAI_MODEL_SHA256.lower():
            raise RuntimeError(
                "MONAI model.pt SHA-256 mismatch. "
                f"Expected {MONAI_MODEL_SHA256}, got {actual_hash}."
            )

    print(
        "TorchScript cache not found. Importing MONAI once to build the "
        "pretrained segmenter..."
    )

    try:
        # Deliberately lazy. This expensive import happens only when the
        # TorchScript cache must be created or rebuilt.
        from monai.networks.nets import UNet as MONAIUNet
    except ImportError as exc:
        raise ImportError(
            "MONAI is required only to create the segmentation cache the first "
            "time. Install it with: pip install monai==1.6.0 huggingface_hub"
        ) from exc

    network = MONAIUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    state_dict = load_checkpoint_state_dict(model_path)

    # strict=True guarantees that architecture and checkpoint keys match.
    network.load_state_dict(state_dict, strict=True)
    network.eval()
    network.requires_grad_(False)

    # Trace on CPU using the exact fixed spatial input used throughout this
    # pipeline. Keeping export on CPU avoids CUDA-specific serialization.
    example_input = torch.zeros(
        1,
        1,
        MONAI_INPUT_SIZE,
        MONAI_INPUT_SIZE,
        dtype=torch.float32,
    )

    network = network.cpu()

    print(
        "Creating MONAI TorchScript cache. This is done only once..."
    )

    with torch.inference_mode():
        reference_output = network(example_input)

        traced_network = torch.jit.trace(
            network,
            example_input,
            strict=False,
        )

        traced_output = traced_network(example_input)

    # Verify that tracing preserved the numerical output before saving it.
    if not torch.allclose(
        reference_output,
        traced_output,
        rtol=1e-4,
        atol=1e-5,
    ):
        max_difference = float(
            (reference_output - traced_output).abs().max().item()
        )
        raise RuntimeError(
            "TorchScript validation failed: traced MONAI output differs from "
            f"the original network (max abs difference={max_difference:.6g})."
        )

    # Freeze inference-only graph where supported.
    try:
        traced_network = torch.jit.freeze(traced_network.eval())
    except Exception:
        traced_network.eval()

    MONAI_TORCHSCRIPT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    traced_network.save(str(MONAI_TORCHSCRIPT_PATH))

    print(
        "Saved MONAI TorchScript cache for future fast starts: "
        f"{MONAI_TORCHSCRIPT_PATH}"
    )

    # Use the cached representation immediately, including on the first run.
    network = torch.jit.load(
        str(MONAI_TORCHSCRIPT_PATH),
        map_location=DEVICE,
    )
    network.eval()

    return network


monai_segmenter = build_monai_segmenter()


# =============================
# PIPELINE STEP 3
# MONAI PROBABILITY MAP + SOFT ROI
# =============================

@torch.inference_mode()
def predict_monai_heart_masks(monai_images, classifier_size):
    """
    Predict cardiac probability maps and mask-quality indicators.

    Args:
        monai_images:
            Tensor [B,1,256,256] in [0,1].

        classifier_size:
            Spatial size of the aligned EfficientNet images, normally
            (224,224).

    Returns:
        roi_probability:
            Dilated soft cardiac probability map aligned to classifier images.

        hard_mask:
            Binary argmax-derived cardiac mask aligned to classifier images.

        valid_mask:
            Boolean tensor indicating whether the segmentation passes initial
            plausibility tests.

        area_ratio:
            Fraction of MONAI input pixels assigned to a cardiac class.

        peak_probability:
            Maximum non-background cardiac probability in each image.

    =========================================================================
    OUTPUT INTERPRETATION
    =========================================================================

    MONAI returns logits for four mutually exclusive classes. Softmax converts
    them into probabilities. The cardiac probability used for ROI extraction is:

        P(heart) = P(LV blood pool)
                 + P(LV myocardium)
                 + P(RV blood pool)

    Because the model is short-axis-specific, the hard mask is screened before
    it is trusted. Invalid masks trigger a full-image fallback later.
    """

    logits = monai_segmenter(monai_images)

    if logits.ndim != 4 or logits.shape[1] != 4:
        raise RuntimeError(
            "Unexpected MONAI output shape. Expected [B,4,H,W], got "
            f"{tuple(logits.shape)}."
        )

    class_probabilities = torch.softmax(logits, dim=1)

    heart_probability = class_probabilities[:, 1:, :, :].sum(
        dim=1,
        keepdim=True,
    ).clamp(0.0, 1.0)

    class_map = torch.argmax(
        class_probabilities,
        dim=1,
        keepdim=True,
    )

    hard_mask_256 = (class_map > 0).float()

    area_ratio = hard_mask_256.mean(dim=(1, 2, 3))
    peak_probability = heart_probability.amax(dim=(1, 2, 3))

    valid_mask = (
        (area_ratio >= MONAI_MIN_HEART_AREA_RATIO)
        & (area_ratio <= MONAI_MAX_HEART_AREA_RATIO)
        & (peak_probability >= MONAI_MIN_PEAK_HEART_PROBABILITY)
    )

    # Max pooling expands the ROI around the predicted ventricles and
    # myocardium. This prevents a narrow segmentation from cutting away nearby
    # diagnostically useful cardiac tissue.
    roi_probability_256 = F.max_pool2d(
        heart_probability,
        kernel_size=MONAI_ROI_DILATION_KERNEL,
        stride=1,
        padding=MONAI_ROI_DILATION_KERNEL // 2,
    )

    hard_mask_256 = F.max_pool2d(
        hard_mask_256,
        kernel_size=MONAI_ROI_DILATION_KERNEL,
        stride=1,
        padding=MONAI_ROI_DILATION_KERNEL // 2,
    )

    roi_probability = F.interpolate(
        roi_probability_256,
        size=classifier_size,
        mode="bilinear",
        align_corners=False,
    )

    hard_mask = F.interpolate(
        hard_mask_256,
        size=classifier_size,
        mode="nearest",
    )

    return (
        roi_probability,
        hard_mask,
        valid_mask,
        area_ratio,
        peak_probability,
    )


def apply_confidence_gated_soft_roi(images, roi_probability, valid_mask):
    """
    Apply MONAI-derived soft ROI weighting with per-slice fallback.

    For a valid segmentation:

        ROI_weight = background_weight
                   + (1 - background_weight) × P(heart)

        ROI_image = original_image × ROI_weight

    Therefore, predicted cardiac pixels retain nearly full intensity while
    background pixels are attenuated but not completely erased.

    For an invalid segmentation, the original full image is returned. This is
    critical because the pretrained model is intended for short-axis MRI and
    may not generalize to every series in the CAD JPEG release.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Expected classifier images [B,3,H,W], got {tuple(images.shape)}."
        )

    if roi_probability.ndim != 4 or roi_probability.shape[1] != 1:
        raise ValueError(
            "Expected ROI probability [B,1,H,W], got "
            f"{tuple(roi_probability.shape)}."
        )

    roi_probability = roi_probability.clamp(0.0, 1.0)

    roi_weight = (
        MONAI_BACKGROUND_WEIGHT
        + (1.0 - MONAI_BACKGROUND_WEIGHT) * roi_probability
    )

    roi_weight = roi_weight.repeat(1, 3, 1, 1)

    weighted_images = images * roi_weight

    valid_mask = valid_mask.view(-1, 1, 1, 1)

    # torch.where applies the decision independently to every slice in a batch.
    return torch.where(valid_mask, weighted_images, images)


def normalize_for_efficientnet(images):
    """
    Apply the official ImageNet normalization for EfficientNet-B0.

    Input images must already be float tensors in [0,1]. ROI extraction is
    intentionally completed before this step.
    """

    mean = torch.tensor(
        EFFICIENTNET_MEAN,
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        EFFICIENTNET_STD,
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)

    return (images - mean) / std


def debug_visualization(
    images,
    roi_probability,
    hard_mask,
    roi_images,
    scores,
    valid_masks,
    area_ratios,
    peak_probabilities,
    batch_idx,
    max_show=2,
):
    """
    Save MONAI segmentation and ROI sanity-check figures.

    Each row shows:

        1. Aligned input image
        2. Soft MONAI cardiac probability map
        3. Dilated hard cardiac mask
        4. Confidence-gated ROI used by EfficientNet

    The titles also report:

        - mask plausibility status
        - predicted cardiac area ratio
        - peak cardiac probability
        - ROI intensity standard deviation

    Visual review is mandatory before treating the pretrained masks as valid on
    this heterogeneous CAD dataset.
    """

    images = images.detach().cpu()
    roi_probability = roi_probability.detach().cpu()
    hard_mask = hard_mask.detach().cpu()
    roi_images = roi_images.detach().cpu()
    scores = scores.detach().cpu()
    valid_masks = valid_masks.detach().cpu()
    area_ratios = area_ratios.detach().cpu()
    peak_probabilities = peak_probabilities.detach().cpu()

    os.makedirs("debug_output", exist_ok=True)

    for i in range(min(max_show, images.shape[0])):

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(
            images[i].permute(1, 2, 0).numpy(),
            vmin=0.0,
            vmax=1.0,
        )
        axes[0].set_title("Aligned input")
        axes[0].axis("off")

        axes[1].imshow(
            roi_probability[i, 0].numpy(),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1].set_title(
            "MONAI P(heart)\n"
            f"peak={peak_probabilities[i]:.3f}"
        )
        axes[1].axis("off")

        axes[2].imshow(
            hard_mask[i, 0].numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[2].set_title(
            f"Mask valid={bool(valid_masks[i])}\n"
            f"area={area_ratios[i]:.4f}"
        )
        axes[2].axis("off")

        axes[3].imshow(
            roi_images[i].permute(1, 2, 0).numpy(),
            vmin=0.0,
            vmax=1.0,
        )
        axes[3].set_title(f"ROI / fallback\nstd={scores[i]:.3f}")
        axes[3].axis("off")

        plt.tight_layout()

        plt.savefig(
            f"debug_output/batch{batch_idx}_img{i}.png",
            dpi=150,
        )

        plt.close(fig)


# =============================
# PIPELINE STEP 4
# EFFICIENTNET-B0 FEATURE EXTRACTION
# =============================

class FeatureExtractor(nn.Module):
    """
    Extract 1280-dimensional embeddings from ROI-weighted MRI slices.

    EfficientNet-B0 is initialized with ImageNet pretrained weights. The
    classification head is removed so the output is the penultimate feature
    representation rather than one of the 1000 ImageNet classes.

    MONAI and EfficientNet have complementary roles:

        MONAI:
            anatomical localization of ventricular structures

        EfficientNet:
            generic feature encoding for the downstream CAD-associated
            classifier

    The ImageNet model is not itself a CAD classifier.
    """

    def __init__(self):

        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT

        self.model = models.efficientnet_b0(
            weights=weights,
        )

        self.model.classifier = nn.Identity()

    def forward(self, x):

        return self.model(x)


feature_extractor = FeatureExtractor().to(DEVICE)
feature_extractor.eval()
feature_extractor.requires_grad_(False)


# =============================
# PIPELINE STEP 5 + 6
# FEATURE EXTRACTION + OPTIONAL SLICE WEIGHTING
# =============================

def _normalize_quality_weights(scores, minimum_weight=SLICE_QUALITY_MIN_WEIGHT):
    """
    Convert a heuristic per-slice score into bounded weights within ONE series.

    Why weights instead of deleting slices?
    ---------------------------------------
    A hard "top 50%" rule can remove diagnostically useful slices and makes the
    result depend on an arbitrary cutoff. If this optional heuristic is enabled,
    every slice is still retained and only its relative influence is changed.

    IMPORTANT:
    The score used below is image-intensity standard deviation after ROI
    weighting. It can respond to anatomy, ROI size, noise and contrast; it is
    NOT a validated clinical MRI quality score. For that reason
    USE_SLICE_QUALITY_WEIGHTS=False is the default.
    """

    scores = np.asarray(scores, dtype=np.float64)

    if scores.size == 1:
        return np.ones(1, dtype=np.float64)

    # Robust normalization using the 25th and 75th percentiles. This is less
    # sensitive to one extremely noisy slice than min-max scaling.
    q25, q75 = np.percentile(scores, [25, 75])
    scale = max(q75 - q25, 1e-8)

    normalized = (scores - q25) / scale
    normalized = np.clip(normalized, 0.0, 1.0)

    weights = minimum_weight + (1.0 - minimum_weight) * normalized

    # Mean=1 means the absolute magnitude of weights does not change the
    # effective sample-size scale for the classifier.
    weights /= max(weights.mean(), 1e-8)

    return weights


def extract_features(dataset, debug=False):
    """
    Run the frozen image-processing pipeline and return one embedding per slice.

    PIPELINE:
        JPEG
          ↓
        per-image intensity scaling
          ↓
        MONAI-compatible 256×256 canvas
          ↓
        MONAI ventricular segmentation
          ↓
        confidence gate
          ↓
        soft cardiac ROI / full-image fallback
          ↓
        EfficientNet-B0
          ↓
        1280-D embedding
          ↓
        optional series-local quality weight (default = 1)

    No class labels are used to generate features or quality weights.

    This function intentionally keeps ALL readable slices. The previous
    implementation selected slices above the per-series median standard
    deviation. That is a brittle hard filter because high standard deviation can
    also reflect noise, artefact, ROI size or extracardiac anatomy.

    By default every slice receives weight 1. The standard-deviation heuristic
    can be enabled only as an explicit ablation with
    USE_SLICE_QUALITY_WEIGHTS=True.

    Returns:
        features:
            [N, 1280] EfficientNet embeddings.
        labels:
            [N] binary patient labels.
        patient_ids:
            [N] patient identifiers.
        series_ids:
            [N] series identifiers.
        quality_weights:
            [N] non-negative slice weights, normalized within each series.
        monai_valid:
            [N] boolean segmentation-gate result.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    all_features = []
    all_labels = []
    all_patients = []
    all_series = []
    all_scores = []
    all_monai_valid = []

    with torch.inference_mode():

        for batch_idx, batch in enumerate(tqdm(loader)):

            (
                images,
                monai_images,
                labels,
                patient_ids,
                series_ids,
            ) = batch

            images = images.to(DEVICE, non_blocking=True)
            monai_images = monai_images.to(DEVICE, non_blocking=True)

            (
                roi_probability,
                hard_mask,
                valid_mask,
                area_ratio,
                peak_probability,
            ) = predict_monai_heart_masks(
                monai_images,
                classifier_size=images.shape[-2:],
            )

            roi_images = apply_confidence_gated_soft_roi(
                images,
                roi_probability,
                valid_mask,
            )

            efficientnet_inputs = normalize_for_efficientnet(roi_images)

            features = feature_extractor(efficientnet_inputs)

            # ---------------------------------------------------------
            # Slice-quality signal
            # ---------------------------------------------------------
            # Standard deviation is only a proxy for image information. It is
            # NOT a clinical quality metric and must not be described as one.
            #
            # We calculate it before ImageNet normalization because the [0,1]
            # intensity scale is easier to interpret.
            scores = torch.std(
                roi_images,
                dim=(1, 2, 3),
            )

            if debug and batch_idx < DEBUG_MAX_BATCHES:
                debug_visualization(
                    images,
                    roi_probability,
                    hard_mask,
                    roi_images,
                    scores,
                    valid_mask,
                    area_ratio,
                    peak_probability,
                    batch_idx,
                )

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_patients.extend(list(patient_ids))
            all_series.extend(list(series_ids))
            all_scores.extend(scores.cpu().numpy().tolist())
            all_monai_valid.extend(valid_mask.cpu().numpy().tolist())

    if not all_features:
        raise RuntimeError("No slice features were extracted.")

    features = np.vstack(all_features)
    labels = np.asarray(all_labels, dtype=np.int64)
    patient_ids = np.asarray(all_patients)
    series_ids = np.asarray(all_series)
    scores = np.asarray(all_scores, dtype=np.float32)
    monai_valid = np.asarray(all_monai_valid, dtype=bool)

    if not (
        len(features)
        == len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(scores)
        == len(monai_valid)
    ):
        raise RuntimeError(
            "Feature extraction produced arrays with inconsistent lengths."
        )

    # -------------------------------------------------------------
    # OPTIONAL QUALITY WEIGHTS
    # -------------------------------------------------------------
    # Equal slice weights are the default because standard deviation is only a
    # heuristic. If explicitly enabled, normalization is performed independently
    # inside each series so scanners/exports with different contrast do not get a
    # global advantage merely because of their intensity distribution.
    if USE_SLICE_QUALITY_WEIGHTS:
        quality_weights = np.zeros(len(scores), dtype=np.float64)

        for series_id in np.unique(series_ids):
            indices = np.flatnonzero(series_ids == series_id)
            quality_weights[indices] = _normalize_quality_weights(
                scores[indices]
            )
    else:
        quality_weights = np.ones(len(scores), dtype=np.float64)

    # Every patient must still have at least one series and one slice.
    if set(patient_ids.tolist()) != set(
        patient_ids[quality_weights > 0].tolist()
    ):
        raise RuntimeError(
            "At least one patient received no positive slice weight."
        )

    valid_rate = float(monai_valid.mean())

    print(
        "MONAI plausible-mask rate: "
        f"{int(monai_valid.sum())}/{len(monai_valid)} ({valid_rate:.2%})"
    )
    print(
        "All extracted slices retained; "
        f"quality weighting={'enabled' if USE_SLICE_QUALITY_WEIGHTS else 'disabled'}; "
        f"mean weight={quality_weights.mean():.3f}"
    )

    return (
        features,
        labels,
        patient_ids,
        series_ids,
        quality_weights,
        monai_valid,
    )


# =============================
# PIPELINE STEP 7
# HIERARCHICAL FUSION (SLICE → SERIES → PATIENT)
# =============================

def weighted_log_odds_fusion(probabilities, weights=None):
    """
    Fuse probabilities in log-odds space, optionally weighted.

    Mathematical form:

        logit(p_fused) = Σ w_i * logit(p_i) / Σ w_i

    This is preferable to multiplying probabilities directly because the
    number of slices/series does not automatically force the fused result
    toward 0 or 1.

    LIMITATION:
    The individual Logistic Regression probabilities are not guaranteed to be
    perfectly calibrated. Therefore this fusion should be described as a
    probabilistic aggregation rule, not as a formally calibrated posterior.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.size == 0:
        raise ValueError("Cannot fuse an empty probability collection.")

    if weights is None:
        weights = np.ones_like(probabilities, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if probabilities.shape != weights.shape:
        raise ValueError("probabilities and weights must have the same shape.")

    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be non-negative and not all zero.")

    eps = 1e-6
    probabilities = np.clip(probabilities, eps, 1 - eps)

    log_odds = np.log(probabilities / (1.0 - probabilities))

    weighted_mean_log_odds = float(
        np.average(log_odds, weights=weights)
    )

    return 1.0 / (1.0 + np.exp(-weighted_mean_log_odds))


def aggregate_patients(
    features,
    labels,
    patient_ids,
    series_ids,
    quality_weights,
    clf,
):
    """
    Aggregate slice predictions hierarchically:

        slice probabilities
              ↓
        weighted series probability
              ↓
        equal-weight series probabilities
              ↓
        patient probability

    WHY TWO LEVELS?
    ---------------
    A patient may have different numbers of images in different series. If all
    slices were fused directly, a long series would dominate the patient merely
    because it contains more frames.

    The revised strategy therefore:
        1. weights slices by within-series quality;
        2. gives each series one fused probability;
        3. gives each series equal influence at patient level.

    This is still a hand-designed fusion rule. For a stronger publication,
    compare it prospectively with a learned attention-pooling model.
    """

    if not (
        len(features)
        == len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(quality_weights)
    ):
        raise ValueError("All aggregation arrays must have the same length.")

    slice_probabilities = clf.predict_proba(features)[:, 1]

    patient_series_probabilities = {}
    patient_labels = {}

    for probability, label, patient_id, series_id, quality_weight in zip(
        slice_probabilities,
        labels,
        patient_ids,
        series_ids,
        quality_weights,
    ):

        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)

        existing_label = patient_labels.get(patient_id)

        if existing_label is not None and existing_label != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent labels."
            )

        patient_labels[patient_id] = label

        patient_series_probabilities.setdefault(patient_id, {})
        patient_series_probabilities[patient_id].setdefault(
            series_id,
            {"probabilities": [], "weights": []},
        )

        patient_series_probabilities[patient_id][series_id][
            "probabilities"
        ].append(float(probability))

        patient_series_probabilities[patient_id][series_id][
            "weights"
        ].append(float(quality_weight))

    fused_patient_probabilities = []
    fused_patient_labels = []
    fused_patient_ids = []

    for patient_id in sorted(patient_series_probabilities):

        series_probability_values = []

        for series_id in sorted(patient_series_probabilities[patient_id]):

            series_data = patient_series_probabilities[patient_id][series_id]

            series_probability = weighted_log_odds_fusion(
                series_data["probabilities"],
                series_data["weights"],
            )

            series_probability_values.append(series_probability)

        # Equal weighting of series prevents one acquisition type from
        # dominating simply because it contains more frames.
        patient_probability = weighted_log_odds_fusion(
            np.asarray(series_probability_values, dtype=np.float64)
        )

        fused_patient_probabilities.append(patient_probability)
        fused_patient_labels.append(patient_labels[patient_id])
        fused_patient_ids.append(patient_id)

    return (
        np.asarray(fused_patient_probabilities, dtype=np.float64),
        np.asarray(fused_patient_labels, dtype=np.int64),
        np.asarray(fused_patient_ids),
    )


def compute_hierarchical_training_weights(
    labels,
    patient_ids,
    series_ids,
    quality_weights=None,
):
    """
    Build training weights aligned with the evaluation hierarchy.

    Desired influence structure:

        class -> patient -> series -> slice

    For slice i belonging to series s of patient p and class c:

        w_i = (1 / N_patients_in_class_c)
              * (1 / N_series_for_patient_p)
              * (q_i / sum(q_j for j in series_s))

    where q_i is the optional within-series quality weight. If quality weighting
    is disabled, q_i = 1 and all slices inside a series share that series' total
    training influence equally.

    This gives two important invariances:

        1. A patient with more exported JPEGs does not dominate training.
        2. A patient with one very long series does not let that series dominate
           over the patient's shorter series.

    The class factor gives Normal and Sick equal total nominal weight even when
    the number of Directory_* patients differs between classes.

    The returned vector is finally rescaled to mean 1. Rescaling does not alter
    relative influence; it only keeps numerical magnitudes convenient for
    scikit-learn.
    """

    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    series_ids = np.asarray(series_ids)

    if not (len(labels) == len(patient_ids) == len(series_ids)):
        raise ValueError(
            "labels, patient_ids and series_ids must have identical lengths."
        )

    if quality_weights is None:
        quality_weights = np.ones(len(labels), dtype=np.float64)
    else:
        quality_weights = np.asarray(quality_weights, dtype=np.float64)

    if len(quality_weights) != len(labels):
        raise ValueError("quality_weights length does not match labels.")

    if np.any(quality_weights < 0):
        raise ValueError("quality_weights must be non-negative.")

    patient_to_label = {}
    patient_to_series = {}
    series_to_patient = {}

    for label, patient_id, series_id in zip(labels, patient_ids, series_ids):
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)

        previous_label = patient_to_label.get(patient_id)
        if previous_label is not None and previous_label != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent labels."
            )
        patient_to_label[patient_id] = label

        previous_patient = series_to_patient.get(series_id)
        if previous_patient is not None and previous_patient != patient_id:
            raise RuntimeError(
                f"Series {series_id} is assigned to more than one patient."
            )
        series_to_patient[series_id] = patient_id
        patient_to_series.setdefault(patient_id, set()).add(series_id)

    class_to_patient_count = {}
    for label in patient_to_label.values():
        class_to_patient_count[label] = (
            class_to_patient_count.get(label, 0) + 1
        )

    if set(class_to_patient_count) != {0, 1}:
        raise RuntimeError(
            "Training fold must contain both Normal and Sick patients."
        )

    series_quality_sum = {}
    for series_id, quality_weight in zip(series_ids, quality_weights):
        series_id = str(series_id)
        series_quality_sum[series_id] = (
            series_quality_sum.get(series_id, 0.0)
            + float(quality_weight)
        )

    weights = np.empty(len(labels), dtype=np.float64)

    for index, (label, patient_id, series_id) in enumerate(
        zip(labels, patient_ids, series_ids)
    ):
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)

        n_patient_series = len(patient_to_series[patient_id])
        if n_patient_series <= 0:
            raise RuntimeError(
                f"Patient {patient_id} unexpectedly has no series."
            )

        normalized_slice_quality = (
            float(quality_weights[index])
            / max(series_quality_sum[series_id], 1e-12)
        )

        weights[index] = (
            normalized_slice_quality
            / n_patient_series
            / class_to_patient_count[label]
        )

    if not np.any(weights > 0):
        raise RuntimeError("All computed training weights are zero.")

    weights *= len(weights) / max(weights.sum(), 1e-12)

    return weights


# =============================
# PIPELINE STEP 8
# PATIENT-LEVEL STRATIFIED K-FOLD EVALUATION
# =============================

# =============================================================
# WHY K-FOLD INSTEAD OF ONE 80/20 SPLIT?
# =============================================================
#
# The original implementation used one random 80/20 holdout. The stability of
# such a result depends on the ACTUAL number of Directory_* patient units found
# at runtime; the paper's 1,224-participant cohort count must not be substituted
# for that folder-derived sample size.
#
# For a research paper, a stronger default is:
#
#   StratifiedKFold(n_splits=5) ON THE PATIENT TABLE
#
# where every row of that table is one validated Directory_* patient.
# Stratification tries to preserve the Normal/Sick proportion. Slices are not
# passed to the splitter at all; they inherit the fold of their patient later.
# This is simpler and more explicit than using a group splitter with one unique
# group per already-aggregated patient row.
#
# The validation prediction for every patient is therefore OUT-OF-FOLD (OOF):
# the classifier has never been trained on that patient's slices.
#
# IMPORTANT:
# The MONAI segmenter and EfficientNet are frozen pretrained models. Feature
# extraction itself does not use CAD labels. Nevertheless, the classifier,
# scaler, and any future hyperparameter tuning MUST be fitted inside each
# training fold only.
# =============================================================


def bootstrap_patient_auc_ci(
    labels,
    probabilities,
    n_bootstrap=2000,
    confidence=0.95,
    random_state=RANDOM_SEED,
):
    """
    Estimate a stratified patient-level bootstrap CI for ROC-AUC.

    Resampling is performed independently within Normal and Sick patients so
    every bootstrap replicate contains both classes. This CI quantifies sampling
    variability of the pooled OOF predictions; it does NOT replace external
    validation and does not capture every source of model-selection uncertainty.
    """

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length.")

    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)

    if len(class0) == 0 or len(class1) == 0:
        raise ValueError("Bootstrap AUC requires both classes.")

    rng = np.random.default_rng(random_state)
    bootstrap_aucs = np.empty(n_bootstrap, dtype=np.float64)

    for bootstrap_index in range(n_bootstrap):
        sampled0 = rng.choice(class0, size=len(class0), replace=True)
        sampled1 = rng.choice(class1, size=len(class1), replace=True)
        sampled = np.concatenate([sampled0, sampled1])

        bootstrap_aucs[bootstrap_index] = roc_auc_score(
            labels[sampled],
            probabilities[sampled],
        )

    alpha = 1.0 - confidence
    lower = float(np.quantile(bootstrap_aucs, alpha / 2.0))
    upper = float(np.quantile(bootstrap_aucs, 1.0 - alpha / 2.0))

    return lower, upper


def print_fold_summary(
    fold_index,
    y_train,
    y_valid,
    patient_train,
    patient_valid,
):
    """Print patient-level rather than slice-level fold statistics."""

    train_patient_labels = {}
    valid_patient_labels = {}

    for label, patient_id in zip(y_train, patient_train):
        train_patient_labels[str(patient_id)] = int(label)

    for label, patient_id in zip(y_valid, patient_valid):
        valid_patient_labels[str(patient_id)] = int(label)

    print(f"\n========== FOLD {fold_index} ==========")
    print(
        f"Train patients: {len(train_patient_labels)} "
        f"(Normal={sum(v == 0 for v in train_patient_labels.values())}, "
        f"Sick={sum(v == 1 for v in train_patient_labels.values())})"
    )
    print(
        f"Validation patients: {len(valid_patient_labels)} "
        f"(Normal={sum(v == 0 for v in valid_patient_labels.values())}, "
        f"Sick={sum(v == 1 for v in valid_patient_labels.values())})"
    )


# -------------------------------------------------------------
# Load the complete dataset ONCE.
# -------------------------------------------------------------
#
# Feature extraction is label-independent and expensive. We therefore extract
# the frozen MONAI/EfficientNet embeddings once and reuse them in each fold.
#
# This does NOT let the Logistic Regression see validation labels.
# The scaler and classifier are still fitted separately inside each fold.
# -------------------------------------------------------------

samples = load_samples(DATASET_PATH)

all_dataset = MRIDataset(
    samples,
    transform,
)

(
    X_all,
    y_all,
    patient_all,
    series_all,
    quality_weights_all,
    monai_valid_all,
) = extract_features(
    all_dataset,
    debug=DEBUG_VISUALIZATION,
)

# -------------------------------------------------------------
# Build ONE patient-level label table.
# -------------------------------------------------------------

patient_labels = {}

for label, patient_id in zip(y_all, patient_all):

    patient_id = str(patient_id)
    label = int(label)

    previous = patient_labels.get(patient_id)

    if previous is not None and previous != label:
        raise RuntimeError(
            f"Patient {patient_id} occurs with conflicting labels."
        )

    patient_labels[patient_id] = label

all_patient_ids = np.asarray(sorted(patient_labels))
all_patient_labels = np.asarray(
    [patient_labels[patient_id] for patient_id in all_patient_ids],
    dtype=np.int64,
)

if np.unique(all_patient_labels).size != 2:
    raise RuntimeError(
        "The dataset must contain both Normal and Sick patients."
    )

print("\nFinal patient-level dataset")
print(f"  Patients: {len(all_patient_ids)}")
print(
    "  Normal:",
    int(np.sum(all_patient_labels == 0)),
)
print(
    "  Sick:",
    int(np.sum(all_patient_labels == 1)),
)
print(f"  Slice embeddings: {len(X_all)}")

if len(all_patient_ids) < 100:
    print(
        "WARNING: fewer than 100 Directory_* patient units were discovered. "
        "Fold AUCs can be highly variable; emphasize pooled OOF performance, "
        "confidence intervals and external validation rather than a single "
        "fold or a single holdout split."
    )

# -------------------------------------------------------------
# Map each slice to its patient-level fold.
# -------------------------------------------------------------
#
# StratifiedKFold is run on the patient table, not on individual slices.
# This is important: if we passed every image as a separate sample, a patient
# with 100 slices could appear in both train and validation.
# -------------------------------------------------------------

N_SPLITS = 5
CV_RANDOM_STATE = RANDOM_SEED

if len(all_patient_ids) < N_SPLITS:
    raise RuntimeError(
        f"Need at least {N_SPLITS} patients for {N_SPLITS}-fold CV."
    )

class_patient_counts = np.bincount(all_patient_labels, minlength=2)

if int(class_patient_counts.min()) < N_SPLITS:
    raise RuntimeError(
        f"{N_SPLITS}-fold stratified CV requires at least {N_SPLITS} "
        "Directory_* patients in EACH class. Found "
        f"Normal={int(class_patient_counts[0])}, "
        f"Sick={int(class_patient_counts[1])}."
    )

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=CV_RANDOM_STATE,
)

# OOF patient predictions are stored here. Each patient should be assigned
# exactly once by the cross-validation procedure.
oof_probability_by_patient = {}
oof_label_by_patient = {}

# -------------------------------------------------------------
# IMPORTANT:
# We need a slice mask for every fold because X_all contains individual slices
# while CV splits are defined on patient IDs.
# -------------------------------------------------------------

for fold_index, (train_patient_idx, valid_patient_idx) in enumerate(
    cv.split(
        all_patient_ids,
        all_patient_labels,
    ),
    start=1,
):

    train_patient_fold = set(
        all_patient_ids[train_patient_idx].tolist()
    )

    valid_patient_fold = set(
        all_patient_ids[valid_patient_idx].tolist()
    )

    # Explicit leakage guard.
    overlap = train_patient_fold.intersection(valid_patient_fold)

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected in fold {fold_index}: "
            f"{sorted(overlap)}"
        )

    train_slice_mask = np.isin(
        patient_all,
        list(train_patient_fold),
    )

    valid_slice_mask = np.isin(
        patient_all,
        list(valid_patient_fold),
    )

    X_train = X_all[train_slice_mask]
    y_train = y_all[train_slice_mask]
    patient_train = patient_all[train_slice_mask]
    series_train = series_all[train_slice_mask]
    quality_train = quality_weights_all[train_slice_mask]

    X_valid = X_all[valid_slice_mask]
    y_valid = y_all[valid_slice_mask]
    patient_valid = patient_all[valid_slice_mask]
    series_valid = series_all[valid_slice_mask]
    quality_valid = quality_weights_all[valid_slice_mask]

    print_fold_summary(
        fold_index,
        y_train,
        y_valid,
        patient_train,
        patient_valid,
    )

    # ---------------------------------------------------------
    # TRAINING WEIGHTS
    # ---------------------------------------------------------
    #
    # Each class receives equal total nominal weight; inside a class, every
    # patient receives equal total influence; inside a patient, every series
    # receives equal total influence. Optional slice-quality weights redistribute
    # influence only WITHIN a series.
    # ---------------------------------------------------------

    training_sample_weights = compute_hierarchical_training_weights(
        y_train,
        patient_train,
        series_train,
        quality_weights=quality_train,
    )

    # ---------------------------------------------------------
    # SCALER + CLASSIFIER
    # ---------------------------------------------------------
    #
    # StandardScaler is FIT ONLY on the training fold and receives the same
    # hierarchical sample weights as Logistic Regression. This prevents both
    # validation leakage AND slice-count imbalance from influencing the fitted
    # mean/variance.
    #
    # Logistic Regression is intentionally kept simple because the primary
    # objective here is to test the imaging representation and patient-level
    # aggregation, not to overfit a high-capacity classifier to ~1,000 patients.
    # ---------------------------------------------------------

    classifier = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="liblinear",
                ),
            ),
        ]
    )

    classifier.fit(
        X_train,
        y_train,
        # IMPORTANT: StandardScaler must be weighted too. Otherwise a patient
        # with many slices would still dominate the training-fold mean/variance
        # even though Logistic Regression itself is patient/series balanced.
        scaler__sample_weight=training_sample_weights,
        logreg__sample_weight=training_sample_weights,
    )

    # ---------------------------------------------------------
    # PATIENT-LEVEL VALIDATION
    # ---------------------------------------------------------

    (
        patient_probabilities,
        patient_ground_truth,
        evaluated_patient_ids,
    ) = aggregate_patients(
        X_valid,
        y_valid,
        patient_valid,
        series_valid,
        quality_valid,
        classifier,
    )

    if len(evaluated_patient_ids) != len(valid_patient_fold):
        raise RuntimeError(
            f"Fold {fold_index}: some validation patients have no prediction."
        )

    # Every patient must be assigned exactly once.
    for patient_id, label, probability in zip(
        evaluated_patient_ids,
        patient_ground_truth,
        patient_probabilities,
    ):

        patient_id = str(patient_id)

        if patient_id in oof_probability_by_patient:
            raise RuntimeError(
                f"Patient {patient_id} received more than one OOF prediction."
            )

        oof_probability_by_patient[patient_id] = float(probability)
        oof_label_by_patient[patient_id] = int(label)

    if np.unique(patient_ground_truth).size == 2:
        fold_auc = roc_auc_score(
            patient_ground_truth,
            patient_probabilities,
        )
        print(f"Fold {fold_index} patient-level AUC: {fold_auc:.4f}")
    else:
        print(
            f"Fold {fold_index}: validation fold contains only one class; "
            "fold-specific AUC is undefined."
        )

# =============================================================
# FINAL OUT-OF-FOLD PATIENT-LEVEL EVALUATION
# =============================================================

if set(oof_probability_by_patient) != set(all_patient_ids.tolist()):
    missing = sorted(
        set(all_patient_ids.tolist())
        - set(oof_probability_by_patient)
    )
    raise RuntimeError(
        "Some patients did not receive an out-of-fold prediction: "
        f"{missing}"
    )

evaluated_patient_ids = np.asarray(
    sorted(oof_probability_by_patient)
)

patient_probabilities = np.asarray(
    [
        oof_probability_by_patient[patient_id]
        for patient_id in evaluated_patient_ids
    ],
    dtype=np.float64,
)

patient_ground_truth = np.asarray(
    [
        oof_label_by_patient[patient_id]
        for patient_id in evaluated_patient_ids
    ],
    dtype=np.int64,
)

# The pooled OOF AUC is the main performance number. It uses one prediction
# per patient and therefore does not let patients with more slices contribute
# multiple times to the ROC curve.
if np.unique(patient_ground_truth).size != 2:
    raise RuntimeError(
        "Pooled patient-level ROC-AUC requires both Normal and Sick patients."
    )

auc = roc_auc_score(
    patient_ground_truth,
    patient_probabilities,
)

auc_ci_lower, auc_ci_upper = bootstrap_patient_auc_ci(
    patient_ground_truth,
    patient_probabilities,
)

print("\n============================================================")
print("FINAL OUT-OF-FOLD PATIENT-LEVEL RESULT")
print("============================================================")
print(f"Patients evaluated: {len(evaluated_patient_ids)}")
print(f"Patient-level OOF AUC: {auc:.6f}")
print(
    "Patient-level bootstrap 95% CI: "
    f"[{auc_ci_lower:.6f}, {auc_ci_upper:.6f}]"
)

print("\nPATIENT-LEVEL OOF PREDICTIONS:")

for patient_id, label, probability in zip(
    evaluated_patient_ids,
    patient_ground_truth,
    patient_probabilities,
):
    print(
        f"  {patient_id}: "
        f"true_label={int(label)}, "
        f"CAD_probability={float(probability):.6f}"
    )

print("\nDone!")

# =============================================================
# RECOMMENDED NEXT EXPERIMENTS
# =============================================================
#
# For a publication-quality study, do not report only this one AUC. The next
# experiments should be:
#
# 1. Ablation: full image vs MONAI ROI.
# 2. Ablation: equal slice weights (default) vs the optional quality heuristic.
# 3. Ablation: mean probability vs log-odds fusion.
# 4. Compare slice-classifier + fusion against hierarchical EMBEDDING pooling
#    followed by a classifier trained directly at patient level. This removes
#    slice-level pseudo-replication and is an especially important baseline.
# 5. Compare Logistic Regression vs linear SVM.
# 6. Compare frozen ImageNet EfficientNet-B0 with a cardiac-MRI-pretrained
#    encoder if an appropriate public checkpoint is available.
# 7. Evaluate each sequence/view separately only if sequence/view identity can
#    be recovered reliably from the released files.
# 8. Audit exact/near-duplicate images ACROSS Directory_* patients before any
#    final publication split; cross-patient duplicates can leak visual content
#    even when patient IDs themselves never cross folds.
# 9. Tune ROI-gate thresholds, classifier C, calibration or decision thresholds
#    only inside training data (nested CV if tuned quantitatively).
# 10. Report sensitivity, specificity, PPV, NPV and confidence intervals using
#     a threshold selected without looking at the validation fold.
# 11. Perform external validation on an independent hospital dataset if possible.
#
# MOST IMPORTANT:
# The MONAI segmenter is designed for 2D short-axis cardiac MR images. The
# published CAD dataset contains heterogeneous CMR acquisitions including
# LGE, Perfusion, T2-weighted and SSFP, with long- and short-axis views.
# Therefore the segmentation mask should NOT be presented as valid anatomy
# segmentation for every image in the dataset. The confidence gate and
# fallback are safeguards, not proof of anatomical correctness.
#
# The Scientific Reports dataset paper reports 63,648 images and 1,224
# participants (722 healthy, 502 CAD), and explicitly states that four sequence
# families and both long- and short-axis planes were used. This domain mismatch
# is one of the most important limitations of the pipeline and should be
# discussed in any manuscript.

