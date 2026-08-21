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
# PATIENT IDENTIFIER USED BY THIS IMPLEMENTATION:
#
#   Normal/Directory_* = one Normal patient
#   Sick/Directory_*   = one Sick patient
#
# Every child folder such as SR_10 or series0003-Body is treated as one imaging
# series belonging to that patient. All images and all series from the same
# Directory_* are kept together throughout splitting, training, aggregation,
# and evaluation.
#
# The architecture combines:
#
#   1. Pretrained MONAI cardiac MRI segmentation
#   2. Confidence-gated soft ROI extraction
#   3. ImageNet-pretrained EfficientNet-B0 feature extraction
#   4. Series-aware slice quality filtering
#   5. Patient-balanced classical Machine Learning classification
#   6. Hierarchical probabilistic fusion: slices → series → patient
#   7. Patient-stratified train/test evaluation
#
# The segmentation stage uses the official MONAI Model Zoo bundle:
#
#   ventricular_short_axis_3label, version 0.3.5
#
# The bundle contains a pretrained 2D residual U-Net that produces four output
# channels:
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
# On first execution, the script downloads the pinned MONAI bundle into the
# local monai_bundles directory. Set AUTO_DOWNLOAD_MONAI_BUNDLE = False if the
# bundle must be supplied manually or the machine has no internet access.
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
# 1280D feature vector / retained slice
#    ↓
# Logistic Regression slice probabilities
#    ↓
# Log-odds fusion across retained slices inside each imaging series
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
#   - Slice filtering is performed independently inside each series so that a
#     batch boundary cannot remove an entire series or patient.
#   - Hierarchical aggregation prevents a very long series from dominating the
#     patient simply because it contains more exported JPEG frames.
#   - Patient-balanced sample weights prevent patients with more retained
#     slices from dominating the slice-level classifier.
#   - Patient-level splitting prevents images or series from the same
#     Directory_* patient from appearing in both train and test sets.
#
# The pipeline approximates the following reasoning process:
#
#   "Localize cardiac anatomy when reliable → inspect informative slices from
#    every series → combine series evidence → classify the patient."
#
# This is an exploratory patient-level classifier under the explicit dataset
# mapping that every Directory_* is one patient. The top-level Normal/Sick
# folder supplies the weak patient label.
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

try:
    from monai.bundle.scripts import download as download_monai_bundle
    from monai.networks.nets import UNet as MONAIUNet
except ImportError as exc:
    raise ImportError(
        "MONAI is required for the pretrained cardiac segmentation stage. "
        "Install it with: pip install monai==1.6.0 huggingface_hub"
    ) from exc
# MONAI provides:
#   - the residual 2D U-Net architecture used by the official bundle
#   - the bundle download utility
#
# The architecture below is instantiated directly from the official bundle
# configuration. This avoids depending on unrelated training-only components
# in train.json while still loading the official pretrained weights.

from sklearn.linear_model import LogisticRegression
# Classical ML classifier trained on EfficientNet slice embeddings.

from sklearn.metrics import roc_auc_score
# ROC-AUC evaluation metric for binary patient-level ranking.

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
# batch, so reduce this value if GPU memory is insufficient.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Automatically select CUDA when available; otherwise use CPU.

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

VERIFY_MONAI_CHECKPOINT_SHA256 = True
# Verify the checkpoint against the hash published for the pinned official
# model file. Set to False only when intentionally using another compatible
# checkpoint.

MONAI_MODEL_SHA256 = (
    "464ca796028831f6c9e2b1cdaebe9af002fc1d7f494f7a89a63f2079e38837a1"
)
# SHA-256 of the official model.pt stored in the MONAI bundle repository.

MONAI_ROI_DILATION_KERNEL = 17
# Expands the predicted ventricular structures to retain a margin around the
# myocardium. Must be an odd positive integer so output size remains unchanged.

MONAI_BACKGROUND_WEIGHT = 0.15
# Soft ROI background retention.
#
# A value of 0 would remove all pixels outside the predicted cardiac region.
# That is risky under domain shift. A value of 0.15 keeps 15% of the original
# background signal while emphasizing the predicted heart region.

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
        3. Downscale only when an image exceeds the MONAI input dimensions.
        4. Fill unused pixels with zero.

    The official bundle was trained with many smaller images that had been
    zero-padded to 256×256. Avoiding unnecessary upscaling follows that training
    convention and reduces interpolation-induced anatomical distortion.
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
            Globally unique Directory_* patient identifier, for example
            Normal/Directory_1 or Sick/Directory_24.

        series_id:
            Globally unique relative path of the imaging series belonging to
            the patient, for example Sick/Directory_24/SR_3.

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
    Discover every MRI image and attach both patient and series identifiers.

    Expected high-level structure:

        Normal/
            Directory_1/                 <- one Normal patient
                series0001-Body/          <- one imaging series
                series0002-Body/
                ...

        Sick/
            Directory_17/                <- one Sick patient
                SR_1/                     <- one imaging series
                SR_2/
                ...

    =========================================================================
    PATIENT-LEVEL DESIGN CHOICE
    =========================================================================

    This implementation uses the mapping supplied for this dataset:

        one immediate Directory_* folder = one patient

    The patient_id is the class-qualified relative path, for example:

        Normal/Directory_1
        Sick/Directory_24

    Qualifying the identifier with Normal/Sick avoids any accidental collision
    if a Directory number is ever reused between the two class folders.

    Every image-containing child folder receives a unique series_id based on
    its full relative path. Images placed directly inside a Directory_* folder
    are also supported; in that case, patient_id and series_id are identical.

    The sample tuple is:

        (image_path, label, patient_id, series_id)

    The patient identifier is later used for train/test splitting and final
    evaluation. The series identifier is retained for two-stage aggregation:

        slice probabilities → series probability → patient probability
    """

    samples = []
    discovered_patients = set()

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

            # The dataset convention used here is that only Directory_* folders
            # represent patients. Other auxiliary folders are ignored.
            if not directory.lower().startswith("directory_"):
                continue

            patient_id = f"{class_name}/{directory}"
            patient_image_count = 0

            for root, _, files in os.walk(patient_path):

                image_files = [
                    filename
                    for filename in files
                    if filename.lower().endswith((".png", ".jpg", ".jpeg"))
                ]

                if not image_files:
                    continue

                series_id = os.path.relpath(root, root_dir).replace(os.sep, "/")

                for filename in sorted(image_files):

                    samples.append(
                        (
                            os.path.join(root, filename),
                            label,
                            patient_id,
                            series_id,
                        )
                    )
                    patient_image_count += 1

            if patient_image_count > 0:
                discovered_patients.add(patient_id)

    if not samples:
        raise RuntimeError(
            f"No MRI images were discovered under dataset path: {root_dir}"
        )

    if not discovered_patients:
        raise RuntimeError(
            "Images were found, but no image-containing Directory_* patient "
            "folders were discovered. Verify the dataset directory structure."
        )

    return samples


samples = load_samples(DATASET_PATH)


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
        return bundle_root

    if not AUTO_DOWNLOAD_MONAI_BUNDLE:
        raise FileNotFoundError(
            "The MONAI checkpoint was not found and automatic download is "
            "disabled. Expected a bundle containing models/model.pt under: "
            f"{MONAI_BUNDLE_DIR}"
        )

    MONAI_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Downloading MONAI bundle "
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
        for key in ("state_dict", "model_state_dict", "network_state_dict"):
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
    Build and initialize the official pretrained MONAI ventricular segmenter.

    The architecture exactly matches the official bundle configuration:

        MONAI UNet
        spatial_dims = 2
        in_channels  = 1
        out_channels = 4
        channels     = (16, 32, 64, 128, 256)
        strides      = (2, 2, 2, 2)
        num_res_units = 2

    Unlike the removed custom Attention U-Net, every segmentation layer in this
    model receives pretrained cardiac-MRI weights.
    """

    bundle_root = ensure_monai_bundle()
    model_path = bundle_root / "models" / "model.pt"

    if VERIFY_MONAI_CHECKPOINT_SHA256:
        actual_hash = sha256_file(model_path)

        if actual_hash.lower() != MONAI_MODEL_SHA256.lower():
            raise RuntimeError(
                "MONAI model.pt SHA-256 mismatch. "
                f"Expected {MONAI_MODEL_SHA256}, got {actual_hash}. "
                "Delete the bundle and download the pinned version again, or "
                "disable verification only when intentionally using a different "
                "compatible checkpoint."
            )

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

    network = network.to(DEVICE)
    network.eval()
    network.requires_grad_(False)

    print(f"Loaded pretrained MONAI segmenter from: {model_path}")

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
# FEATURE EXTRACTION + SLICE FILTERING
# =============================

def extract_features(dataset, debug=False):
    """
    Run MONAI segmentation, ROI construction, EfficientNet encoding, and
    series-aware low-information slice filtering.

    Returns:
        features:
            Matrix [retained_slices, 1280].

        labels:
            Binary patient label repeated for each retained slice.

        patient_ids:
            Directory_* patient identifier for each retained slice.

        series_ids:
            Imaging-series identifier for each retained slice.

    Slice-quality filtering is applied AFTER all batches have been encoded and
    independently inside every series. The top half by ROI intensity standard
    deviation is retained in each series. This makes the result independent of
    DataLoader batch boundaries and guarantees that every non-empty series, and
    therefore every patient, keeps at least one slice.

    The function also prints the proportion of slices whose MONAI masks passed
    the plausibility gate. A very low valid-mask rate is a warning that the
    pretrained short-axis segmenter is strongly out of domain for this subset.
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

    total_masks = 0
    valid_masks_count = 0

    with torch.inference_mode():

        for batch_idx, batch in enumerate(tqdm(loader)):

            (
                images,
                monai_images,
                labels,
                patient_ids,
                series_ids,
            ) = batch

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            monai_images = monai_images.to(
                DEVICE,
                non_blocking=True,
            )

            # =====================================================
            # STEP 2: PRETRAINED MONAI SEGMENTATION
            # =====================================================

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

            total_masks += int(valid_mask.numel())
            valid_masks_count += int(valid_mask.sum().item())

            # =====================================================
            # STEP 3: CONFIDENCE-GATED SOFT ROI
            # =====================================================

            roi_images = apply_confidence_gated_soft_roi(
                images,
                roi_probability,
                valid_mask,
            )

            # =====================================================
            # STEP 4: EFFICIENTNET FEATURE EXTRACTION
            # =====================================================

            efficientnet_inputs = normalize_for_efficientnet(
                roi_images
            )

            features = feature_extractor(efficientnet_inputs)

            # =====================================================
            # STEP 5: SLICE QUALITY SCORING
            # =====================================================

            # Standard deviation is measured before ImageNet normalization.
            # Low variability often indicates an empty, flat, or weakly
            # informative image. This remains only a simple heuristic and can
            # later be replaced with a learned quality model.
            scores = torch.std(
                roi_images,
                dim=(1, 2, 3),
            )

            # =====================================================
            # DEBUG VISUALIZATION
            # =====================================================

            if debug:  # Add "and batch_idx == 0" to save only the first batch.
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

            # Store every encoded slice first. Filtering is performed per
            # series after all batches are available, not per arbitrary batch.
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_patients.extend(list(patient_ids))
            all_series.extend(list(series_ids))
            all_scores.extend(scores.cpu().numpy().tolist())

    if not all_features:
        raise RuntimeError(
            "No slice features were extracted. Inspect image loading, MONAI "
            "masks, and EfficientNet inference."
        )

    features = np.vstack(all_features)
    labels = np.asarray(all_labels, dtype=np.int64)
    patient_ids = np.asarray(all_patients)
    series_ids = np.asarray(all_series)
    scores = np.asarray(all_scores, dtype=np.float32)

    if not (
        len(features)
        == len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(scores)
    ):
        raise RuntimeError(
            "Feature extraction produced arrays with inconsistent lengths."
        )

    # =============================================================
    # SERIES-AWARE SLICE QUALITY FILTERING
    # =============================================================

    keep = np.zeros(len(scores), dtype=bool)

    for series_id in np.unique(series_ids):

        series_indices = np.flatnonzero(series_ids == series_id)
        series_scores = scores[series_indices]

        threshold = np.median(series_scores)
        retained_indices = series_indices[series_scores >= threshold]

        if retained_indices.size == 0:
            # Defensive fallback for pathological numerical cases.
            retained_indices = np.asarray(
                [series_indices[int(np.argmax(series_scores))]]
            )

        keep[retained_indices] = True

    if not bool(keep.any()):
        raise RuntimeError(
            "No slice features were retained after series-aware filtering."
        )

    retained_patients = set(patient_ids[keep].tolist())
    all_patients_set = set(patient_ids.tolist())

    if retained_patients != all_patients_set:
        missing = sorted(all_patients_set - retained_patients)
        raise RuntimeError(
            "Slice filtering removed every image from one or more patients: "
            f"{missing}"
        )

    valid_rate = valid_masks_count / max(total_masks, 1)

    print(
        "MONAI plausible-mask rate: "
        f"{valid_masks_count}/{total_masks} ({valid_rate:.2%})"
    )
    print(
        "Series-aware slice retention: "
        f"{int(keep.sum())}/{len(keep)} ({float(keep.mean()):.2%})"
    )

    return (
        features[keep],
        labels[keep],
        patient_ids[keep],
        series_ids[keep],
    )


# =============================
# PIPELINE STEP 7
# HIERARCHICAL FUSION (SLICE → SERIES → PATIENT)
# =============================

def log_odds_fusion(probabilities):
    """
    Fuse a collection of probabilities by averaging their log-odds.

    Averaging, rather than summing, prevents the number of elements alone from
    making the fused result artificially extreme. The same function is used at
    both hierarchy levels:

        retained slice probabilities → one series probability
        series probabilities         → one patient probability
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.size == 0:
        raise ValueError("Cannot fuse an empty probability collection.")

    eps = 1e-6
    probabilities = np.clip(probabilities, eps, 1 - eps)

    log_odds = np.log(probabilities / (1 - probabilities))
    mean_log_odds = float(log_odds.mean())

    return 1.0 / (1.0 + np.exp(-mean_log_odds))


def aggregate_patients(
    features,
    labels,
    patient_ids,
    series_ids,
    clf,
):
    """
    Aggregate retained slice predictions into one probability per patient.

    Hierarchy:

        1. Predict one CAD-associated probability for every retained slice.
        2. Fuse all retained slices belonging to the same imaging series.
        3. Fuse all series probabilities belonging to the same Directory_*
           patient.

    The second fusion level gives each imaging series one vote regardless of
    how many JPEG frames it contains. This avoids over-weighting a long cine
    series relative to a short series from the same patient.
    """

    slice_probabilities = clf.predict_proba(features)[:, 1]

    patient_series_probabilities = {}
    patient_labels = {}

    for probability, label, patient_id, series_id in zip(
        slice_probabilities,
        labels,
        patient_ids,
        series_ids,
    ):

        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)

        existing_label = patient_labels.get(patient_id)

        if existing_label is not None and existing_label != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent labels: "
                f"{existing_label} and {label}."
            )

        patient_labels[patient_id] = label

        patient_series_probabilities.setdefault(patient_id, {})
        patient_series_probabilities[patient_id].setdefault(series_id, [])
        patient_series_probabilities[patient_id][series_id].append(
            float(probability)
        )

    fused_patient_probabilities = []
    fused_patient_labels = []
    fused_patient_ids = []

    for patient_id in sorted(patient_series_probabilities):

        series_probability_values = []

        for series_id in sorted(patient_series_probabilities[patient_id]):

            retained_slice_probabilities = np.asarray(
                patient_series_probabilities[patient_id][series_id],
                dtype=np.float64,
            )

            series_probability_values.append(
                log_odds_fusion(retained_slice_probabilities)
            )

        patient_probability = log_odds_fusion(
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


def compute_patient_balanced_sample_weights(labels, patient_ids):
    """
    Give every patient equal total influence within its class and give both
    classes equal total influence during slice-level Logistic Regression.

    Without these weights, a patient with thousands of retained slices would
    contribute far more optimization weight than a patient with fewer slices.

    For one slice from patient p with class c:

        weight = 1 / (number_of_patients_in_class_c × slices_from_patient_p)

    The weights are finally rescaled to have mean 1, which does not change their
    relative effect but keeps their numerical scale conventional.
    """

    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)

    if len(labels) != len(patient_ids):
        raise ValueError("labels and patient_ids must have identical lengths.")

    patient_to_label = {}
    patient_to_slice_count = {}

    for label, patient_id in zip(labels, patient_ids):

        patient_id = str(patient_id)
        label = int(label)

        existing_label = patient_to_label.get(patient_id)

        if existing_label is not None and existing_label != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent training labels."
            )

        patient_to_label[patient_id] = label
        patient_to_slice_count[patient_id] = (
            patient_to_slice_count.get(patient_id, 0) + 1
        )

    class_to_patient_count = {}

    for label in patient_to_label.values():
        class_to_patient_count[label] = class_to_patient_count.get(label, 0) + 1

    if set(class_to_patient_count) != {0, 1}:
        raise RuntimeError(
            "Training data must contain at least one Normal and one Sick "
            "patient."
        )

    weights = np.empty(len(labels), dtype=np.float64)

    for index, (label, patient_id) in enumerate(zip(labels, patient_ids)):

        patient_id = str(patient_id)
        label = int(label)

        weights[index] = 1.0 / (
            class_to_patient_count[label]
            * patient_to_slice_count[patient_id]
        )

    weights *= len(weights) / weights.sum()

    return weights


# =============================
# PIPELINE STEP 8
# PATIENT-LEVEL TRAIN / TEST EVALUATION
# =============================

# =============================================================
# BUILD PATIENT AND SERIES LISTS
# =============================================================

patient_labels = {}

for _, label, patient_id, _ in samples:

    existing_label = patient_labels.get(patient_id)

    if existing_label is not None and existing_label != label:
        raise RuntimeError(
            f"Patient {patient_id} occurs with conflicting labels."
        )

    patient_labels[patient_id] = int(label)

normal_patients = sorted(
    patient_id
    for patient_id, label in patient_labels.items()
    if label == 0
)

sick_patients = sorted(
    patient_id
    for patient_id, label in patient_labels.items()
    if label == 1
)

if len(normal_patients) < 2 or len(sick_patients) < 2:
    raise RuntimeError(
        "A patient-level holdout split requires at least two Normal and two "
        "Sick Directory_* patients."
    )

rng = np.random.RandomState(42)
normal_patients = list(rng.permutation(normal_patients))
sick_patients = list(rng.permutation(sick_patients))


def calculate_test_patient_count(number_of_patients, test_fraction=0.2):
    """Choose a non-empty test subset while preserving training patients."""

    return min(
        max(1, int(round(number_of_patients * test_fraction))),
        number_of_patients - 1,
    )


normal_test_count = calculate_test_patient_count(len(normal_patients))
sick_test_count = calculate_test_patient_count(len(sick_patients))

# =============================================================
# PATIENT-LEVEL TRAIN / TEST SPLIT
# =============================================================

# IMPORTANT:
# The split unit is the Directory_* patient, never an image and never a series.
# Consequently, every image and every series belonging to one patient remains
# entirely in either train or test. This is the central leakage-prevention rule
# required for patient-level evaluation.

train_patients = (
    normal_patients[normal_test_count:]
    + sick_patients[sick_test_count:]
)

test_patients = (
    normal_patients[:normal_test_count]
    + sick_patients[:sick_test_count]
)

train_patient_set = set(train_patients)
test_patient_set = set(test_patients)

patient_overlap = train_patient_set.intersection(test_patient_set)

if patient_overlap:
    raise RuntimeError(
        "Patient leakage detected between train and test: "
        f"{sorted(patient_overlap)}"
    )

train_samples = [
    sample
    for sample in samples
    if sample[2] in train_patient_set
]

test_samples = [
    sample
    for sample in samples
    if sample[2] in test_patient_set
]

train_series = sorted(set(sample[3] for sample in train_samples))
test_series = sorted(set(sample[3] for sample in test_samples))

series_overlap = set(train_series).intersection(test_series)

if series_overlap:
    raise RuntimeError(
        "Series leakage detected between train and test: "
        f"{sorted(series_overlap)}"
    )

print(f"Training patients: {len(train_patients)}")
print(f"Testing patients:  {len(test_patients)}")
print(f"Training series:   {len(train_series)}")
print(f"Testing series:    {len(test_series)}")
print(f"Training images:   {len(train_samples)}")
print(f"Testing images:    {len(test_samples)}")

# Optional debugging subset:
# Do not truncate with train_samples[0:N] or test_samples[0:N] for a real
# patient-level experiment because doing so can produce partial patients.
# For debugging, select a small list of complete patient IDs instead.

train_dataset = MRIDataset(
    train_samples,
    transform,
)

test_dataset = MRIDataset(
    test_samples,
    transform,
)

# =============================================================
# FEATURE EXTRACTION
# =============================================================

(
    X_train,
    y_train,
    patient_train,
    series_train,
) = extract_features(
    train_dataset,
    debug=True,
)

(
    X_test,
    y_test,
    patient_test,
    series_test,
) = extract_features(
    test_dataset,
    debug=True,
)

# =============================================================
# PATIENT-BALANCED CLASSICAL ML CLASSIFIER
# =============================================================

# Logistic Regression remains a slice-level classifier on frozen EfficientNet
# embeddings, but patient-balanced sample weights ensure that:
#
#   - every patient has equal total influence inside its class;
#   - Normal and Sick patients have equal total class influence;
#   - a patient with more exported JPEG frames cannot dominate optimization.

classifier = LogisticRegression(
    max_iter=1000,
)

training_sample_weights = compute_patient_balanced_sample_weights(
    y_train,
    patient_train,
)

classifier.fit(
    X_train,
    y_train,
    sample_weight=training_sample_weights,
)

# =============================================================
# PATIENT-LEVEL PREDICTION
# =============================================================

(
    patient_probabilities,
    patient_ground_truth,
    evaluated_patient_ids,
) = aggregate_patients(
    X_test,
    y_test,
    patient_test,
    series_test,
    classifier,
)

# =============================================================
# PATIENT-LEVEL EVALUATION
# =============================================================

# ROC-AUC:
#
#   1.0 = perfect patient ranking
#   0.5 = random patient ranking
#
# Each test patient contributes exactly one probability and one ground-truth
# label, regardless of how many series or slices are stored in Directory_*.

if np.unique(patient_ground_truth).size != 2:
    raise RuntimeError(
        "Patient-level ROC-AUC requires both Normal and Sick patients in the "
        "test set."
    )

auc = roc_auc_score(
    patient_ground_truth,
    patient_probabilities,
)

print("PATIENT-LEVEL PREDICTIONS:")

for patient_id, label, probability in zip(
    evaluated_patient_ids,
    patient_ground_truth,
    patient_probabilities,
):
    print(
        f"  {patient_id}: true_label={int(label)}, "
        f"CAD_probability={float(probability):.6f}"
    )

print("PATIENT-LEVEL AUC:", auc)
print("Done!")
