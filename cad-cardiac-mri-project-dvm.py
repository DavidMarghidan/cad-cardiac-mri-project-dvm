#%% ============================================================
# 🧠 CAD Detection from Cardiac MRI – MONAI Patient-Level Pipeline (Single File)
# ============================================================

# ============================================================================
# OVERVIEW
# ============================================================================
#
# This script implements a COMPLETE EXECUTION PIPELINE for exploratory
# patient-level CAD (Coronary Artery Disease) classification from 2D cardiac
# MRI JPEG exports. It is "end-to-end" only in the operational sense that it
# runs from files to patient scores; it is NOT an end-to-end jointly trained
# neural network because MONAI and EfficientNet remain frozen:
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
# IMPORTANT: SR_* / series* folders are NOT treated as patients. They are used
# as folder-defined SERIES PROXIES belonging to the Directory_* patient. Because
# the release contains JPEG files rather than the original DICOM metadata, these
# folder names must not be described as validated DICOM SeriesInstanceUIDs.
# All images and all series proxies from one Directory_* remain together in
# every train/validation split and are combined into one patient-level score.
#
# The architecture combines:
#
#   1. Patient-level grouping with patient_id fixed to Directory_*
#   2. Optional pretrained MONAI ventricular segmentation (short-axis-specific)
#   3. Confidence-gated soft ROI extraction with full-image fallback
#   4. Explicitly pinned ImageNet EfficientNet-B0 feature extraction
#   5. Optional series-local slice-quality weighting (disabled by default)
#   6. Recommended direct patient-level training after hierarchical EMBEDDING
#      pooling: slices → series proxy → patient
#   7. Optional legacy weakly supervised slice classifier followed by
#      hierarchical probability fusion, retained as an ablation
#   8. Stratified K-fold evaluation defined directly on Directory_* patients
#   9. Pooled out-of-fold AUC with a patient-level bootstrap confidence interval
#  10. Exact decoded-pixel duplicate auditing across Directory_* patients
#  11. Reusable feature caching and machine-readable QC/result files
#
# IMPORTANT METHODOLOGICAL CHANGES:
# The original version used one 80/20 split and discarded roughly half of the
# slices using a standard-deviation cutoff. This revision instead:
#
#   - performs cross-validation on the validated Directory_* patient units;
#   - keeps every successfully decoded slice by default; an unreadable file
#     raises an explicit error instead of being silently omitted;
#   - keeps standard deviation only as an OPTIONAL, non-clinical heuristic;
#   - uses direct patient-level embedding pooling as the recommended default,
#     thereby avoiding repeated patient labels being treated as independent
#     slice observations;
#   - retains the prior slice-classifier/fusion method only as a selectable
#     weakly supervised ablation;
#   - when the slice method is selected, scales sample-weight mass to the number
#     of training patients, because globally multiplying sample weights changes
#     the effective regularization of a regularized Logistic Regression model;
#   - reports a patient-level bootstrap CI around the pooled OOF ROC-AUC;
#   - saves OOF predictions, fold assignments and MONAI QC summaries.
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
# Required packages for the normal path:
#
#   pip install huggingface_hub
#   pip install torch torchvision opencv-python numpy "scikit-learn>=1.0" matplotlib tqdm
#
# MONAI itself is only required for the exceptional fallback that reconstructs
# the network from ``models/model.pt`` when the official ``models/model.ts``
# artifact cannot be loaded:
#
#   pip install monai==1.6.0
#
# StandardScaler.fit(sample_weight=...) is required. Use mutually compatible
# torch/torchvision builds for the installed CUDA runtime.
#
# NOTE ABOUT THE PRETRAINED SEGMENTER:
# The bundle metadata identifies version 0.3.5 and records the original bundle
# environment as MONAI 1.3.0 / PyTorch 1.13.0. The Hugging Face repository is
# pinned to a specific commit and the official model.ts/model.pt digests are
# checked before use. The official TorchScript artifact is the preferred path:
# it can be loaded directly by PyTorch and avoids both the slow MONAI import and
# a locally re-traced copy during ordinary first and later executions.
#
# Set AUTO_DOWNLOAD_MONAI_BUNDLE = False for an offline environment and place
# the pinned bundle files under the configured MONAI bundle directory.
#
# ============================================================================
# PIPELINE FLOW
# ============================================================================
#
# Raw MRI JPEG slice
#    ↓
# Per-image min-max intensity scaling to [0,1]
#    ↓
# Aspect-ratio-preserving placement in a 256×256 zero-padded canvas
#    ↓
# Optional pretrained MONAI residual U-Net
#    ↓
# Confidence-gated soft ROI or unchanged full-image fallback
#    ↓
# Custom 256→224 whole-canvas resize + ImageNet mean/std normalization
#    ↓
# Frozen EfficientNet-B0 feature encoding
#    ↓
# 1280D feature vector per successfully decoded slice
#    ↓
# Hierarchical embedding pooling inside each folder-defined series proxy
#    ↓
# Equal-weight pooling across a Directory_* patient's series proxies
#    ↓
# Logistic Regression trained on one vector per Directory_* patient
#    ↓
# Out-of-fold patient-level CAD-associated MODEL SCORE
#
# Optional legacy ablation:
#   slice Logistic Regression → series log-odds fusion → patient log-odds fusion
#
# ============================================================================
# DETAILED DATA CONTRACT AND SHAPE TRACE
# ============================================================================
#
# The pipeline passes a small number of clearly defined objects from one stage
# to the next. Keeping these contracts explicit makes it easier to debug shape,
# grouping, leakage, and caching errors:
#
#   A. ``samples`` -- Python list created by ``load_samples``
#
#      Each element is:
#
#          (image_path, label, patient_id, series_id)
#
#      where ``patient_id`` is always Directory_* and ``series_id`` is a
#      patient-scoped folder proxy such as Directory_24/SR_3. The label is
#      metadata only during frozen image processing; it is never supplied to
#      MONAI or EfficientNet.
#
#   B. One ``MRIDataset`` item -- tensors plus immutable metadata
#
#          classification_image : [3, 224, 224], float, initially in [0,1]
#          monai_image          : [1, 256, 256], float, in [0,1]
#          label                : scalar 0 or 1
#          patient_id           : Directory_* string
#          series_id            : patient-scoped folder-proxy string
#          sample_index         : deterministic position in ``samples``
#          decoded_pixel_hash   : exact decoded-pixel SHA-256 string
#
#   C. One DataLoader batch -- the same objects with a leading batch dimension
#
#          images       : [B, 3, 224, 224]
#          monai_images : [B, 1, 256, 256]
#
#   D. MONAI outputs -- used only when ``USE_MONAI_ROI=True``
#
#          logits                  : [B, 4, 256, 256]
#          heart probability       : [B, 1, 256, 256]
#          aligned ROI probability : [B, 1, 224, 224]
#          valid_mask              : [B], one plausibility decision per slice
#
#      A failed plausibility check does not delete the slice. It selects the
#      unmodified full image for that slice.
#
#   E. EfficientNet output
#
#          slice embedding : [B, 1280]
#
#      The 1000-class ImageNet head is removed. These vectors are frozen image
#      descriptors, not CAD predictions.
#
#   F. Cached extraction arrays -- one row per decoded JPEG slice
#
#      Features, labels, patient IDs, series IDs, optional quality weights,
#      MONAI QC values, slice scores, and decoded-pixel hashes are stored with
#      one-to-one row alignment. Changing any feature-affecting setting changes
#      the cache fingerprint.
#
#   G. Recommended supervised input
#
#      Slice embeddings are pooled within each series proxy, then equally across
#      all series proxies of one Directory_* patient. The classifier therefore
#      receives exactly one vector and one label per patient.
#
#   H. Evaluation output
#
#      Every Directory_* patient receives exactly one out-of-fold score and one
#      fold identifier. Pooled ROC-AUC and a stratified patient bootstrap
#      confidence interval are calculated from those patient rows.
#
# TRAINED VERSUS FROZEN COMPONENTS
# --------------------------------
#
#   Frozen / inference-only:
#       - MONAI ventricular segmenter
#       - ImageNet EfficientNet-B0 encoder
#       - deterministic image preprocessing and pooling rules
#
#   Fitted separately inside every training fold:
#       - StandardScaler
#       - optional PCA
#       - Logistic Regression
#
# This distinction is central to leakage control. Validation-patient labels and
# embeddings are never used to fit fold-local preprocessing or classification.
#
# ============================================================================
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
#   - Hierarchical aggregation prevents a very long folder-defined series proxy
#     from dominating simply because it contains more exported JPEG frames.
#   - The recommended classifier receives one pooled embedding per patient, so
#     the effective labeled sample size is the number of Directory_* folders,
#     not the number of JPEG slices.
#   - The optional legacy slice classifier uses hierarchical weights so patients
#     and their folder-defined series proxies receive controlled nominal influence.
#   - Patient-level splitting prevents images or series proxies from the same
#     Directory_* patient from appearing in both training and validation folds.
#
# The pipeline approximates the following reasoning process:
#
#   "Localize cardiac anatomy when reliable → inspect informative slices from
#    every folder proxy → combine patient evidence → classify the patient."
#
# This is an exploratory patient-level classifier under the validated dataset
# mapping that every Directory_* is one patient. The top-level Normal/Sick
# folder supplies the patient-level class label. The recommended strategy trains
# directly on one pooled vector per Directory_* patient. If the optional legacy
# strategy propagates that label to every slice, its supervision is weak because
# individual slices are not independently annotated for CAD and are strongly
# correlated within the same patient.
#
# ============================================================================


# =============================
# IMPORTS
# =============================

import csv
# Standard-library CSV writer used for reproducible machine-readable outputs.

import hashlib
# Used for checkpoint verification and feature-cache fingerprints.
# Checksum verification makes the experiment more reproducible and helps detect
# corrupted or unintended model files.

import json
# Serializes run configuration, software versions and evaluation summaries.

import os
# OS interaction (files, paths, environment variables).
# Used for:
#   - traversing dataset folders
#   - building portable paths
#   - detecting Kaggle versus local execution
#   - reading optional bundle path overrides

import platform
# Records operating-system and Python runtime information for reproducibility.

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

import torchvision
# Used only to report the exact torchvision version in the run metadata.

import torchvision.transforms as transforms
# Converts NumPy HWC arrays to PyTorch CHW tensors.
# EfficientNet normalization is intentionally applied AFTER ROI extraction.

from torchvision import models
# Provides ImageNet-pretrained EfficientNet-B0.

# IMPORTANT STARTUP OPTIMIZATION:
#
# MONAI is intentionally NOT imported at module startup or on the normal model
# loading path. The pinned bundle already provides ``models/model.ts``; the
# script downloads that file with huggingface_hub and loads it directly through
# ``torch.jit.load``. MONAI UNet is imported lazily only if the official
# TorchScript artifact cannot be used and reconstruction from model.pt is needed.

import sklearn
# Used to record the exact scikit-learn version in the run metadata.

from sklearn.decomposition import PCA
# Optional fold-local dimensionality reduction for the very small patient cohort.

from sklearn.linear_model import LogisticRegression
# Linear probabilistic classifier. By default it is trained on one pooled
# embedding per Directory_* patient; slice-level training is an optional ablation.

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

# ---------------------------------------------------------------------------
# EXECUTION / EVALUATION STRATEGY
# ---------------------------------------------------------------------------

CLASSIFICATION_STRATEGY = "patient_embedding"
# Recommended default: "patient_embedding".
#
#   patient_embedding:
#       weighted mean of slice embeddings inside each series proxy, followed by
#       an equal mean across the patient's series proxies. Logistic Regression
#       is then trained on exactly one vector per Directory_* patient.
#
#   slice_probability_fusion:
#       legacy weakly supervised approach: train Logistic Regression on slice
#       embeddings with hierarchical weights, then fuse slice probabilities to
#       series and patient scores. This remains useful as a declared ablation,
#       but it does not remove within-patient pseudo-replication from fitting.

N_SPLITS = 5
CV_RANDOM_STATE = RANDOM_SEED
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000
USE_PATIENT_PCA = True
PATIENT_PCA_EXPLAINED_VARIANCE = 0.95
BOOTSTRAP_REPLICATES = 2000
# Patient-level PCA is fitted ONLY on each training fold and can retain at most
# n_train_patients - 1 components. It reduces the 1280D/very-small-N mismatch,
# but remains an analysis choice that must be fixed before OOF evaluation.
# All settings above are fixed BEFORE evaluation. If tuned using performance,
# tuning must occur inside an inner patient-level CV loop, never on OOF results.

USE_MONAI_ROI = True
# Set False for the required full-image ablation. When False, MONAI is not built
# or executed and the entire aligned image is passed to EfficientNet.

USE_CUDA_AMP = True
# Mixed-precision inference can substantially improve GPU throughput. Logits and
# stored embeddings are converted back to float32 before downstream processing.
# Set False when exact numerical comparability across hardware is more important.

DATALOADER_NUM_WORKERS = 0
# Zero is the safest cross-platform default, especially on Windows notebooks.
# Because execution is protected by ``if __name__ == "__main__"``, this can be
# increased after testing local RAM, storage throughput and multiprocessing.

USE_FEATURE_CACHE = True
FORCE_REBUILD_FEATURE_CACHE = False
FEATURE_CACHE_SCHEMA_VERSION = "2026-08-22-v4"
# Frozen MONAI/EfficientNet extraction is the expensive stage. A cache keyed by
# dataset file metadata and all feature-affecting settings avoids repeating it.
# Increment FEATURE_CACHE_SCHEMA_VERSION after changing extraction semantics.

AUDIT_EXACT_DECODED_PIXEL_DUPLICATES = True
FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES = False
# During the same pass that decodes each image, the code hashes the raw decoded
# grayscale pixel matrix (including its shape). This detects exact pixel copies
# even when JPEG container metadata differs. It does NOT detect near-duplicates
# or re-encoded copies whose decoded pixels changed slightly. Cross-patient
# duplicate groups are written to CSV and trigger a prominent warning. Set the
# failure flag True for a strict publication run after deciding how such groups
# will be handled without redefining patient_id away from Directory_*.

USE_SLICE_QUALITY_WEIGHTS = False
# IMPORTANT DEFAULT:
# Standard deviation after ROI weighting is NOT a validated MRI quality metric.
# Therefore the publication-safe default is equal slice weights. Set this True
# only for a predeclared ablation after verifying that the heuristic is useful.

SLICE_QUALITY_MIN_WEIGHT = 0.25
# Lower bound used only when USE_SLICE_QUALITY_WEIGHTS=True.

DEBUG_VISUALIZATION = False
DEBUG_INDICES = "10%"
# Select debug figures from the complete image list. Use "full" to save a
# figure for every image, or a percentage such as "10%" to save an evenly
# spaced, deterministic 10% of all images. Numeric percentages (for example,
# 10 or 2.5) are also accepted. These figures are qualitative QC only. ROI-gate
# thresholds must not be repeatedly adjusted after inspecting OOF performance.

# ---------------------------------------------------------------------------
# MONAI BUNDLE CONFIGURATION
# ---------------------------------------------------------------------------

MONAI_BUNDLE_NAME = "ventricular_short_axis_3label"
# Official MONAI Model Zoo bundle used for cardiac segmentation.

MONAI_BUNDLE_VERSION = "0.3.5"
# Human-readable bundle version declared by configs/metadata.json.

MONAI_HF_REPO_ID = "MONAI/ventricular_short_axis_3label"
MONAI_HF_REVISION = "eefc17c8e002cc8a567bbfce8f02d7d3116408f4"
# The immutable Hugging Face commit corresponding to the pinned metadata
# release. Pinning a commit, rather than downloading ``main``, prevents a future
# repository update from silently changing the files used by the experiment.

AUTO_DOWNLOAD_MONAI_BUNDLE = True
# When True, missing pinned files are downloaded through huggingface_hub.
# Set False for an offline machine and place at least these files manually:
#
#   <MONAI_BUNDLE_DIR>/ventricular_short_axis_3label/models/model.ts
#   <MONAI_BUNDLE_DIR>/ventricular_short_axis_3label/configs/metadata.json
#
# ``models/model.pt`` and ``configs/train.json`` are needed only for the
# reconstruction fallback described below.

VERIFY_MONAI_ARTIFACT_SHA256 = True
# Official SHA-256 digests published with the pinned Hugging Face artifacts.
# Verification is enabled by default because a wrong or corrupted segmentation
# model would invalidate both the ROI and the frozen feature cache.

MONAI_OFFICIAL_TORCHSCRIPT_SHA256 = (
    "27d5532401fa6c1883872fa21635adbb7615981e7f385d0c58dd75b355e340b3"
)
MONAI_MODEL_SHA256 = (
    "464ca796028831f6c9e2b1cdaebe9af002fc1d7f494f7a89a63f2079e38837a1"
)

MONAI_ROI_DILATION_KERNEL = 31
# Expands the predicted ventricular structures to retain a margin around the
# myocardium. Must be an odd positive integer so output size remains unchanged.
# 17  → small spatial expansion
# 31  → moderate spatial expansion (current setting)
# 41  → large spatial expansion
# 51  → very large spatial expansion

MONAI_BACKGROUND_WEIGHT = 0.15
# Soft ROI background retention.
#
# A value of 0 would remove all pixels outside the predicted cardiac region.
# That is risky under domain shift. A value of 0.15 keeps 15% of the original
# background signal while emphasizing the predicted heart region.
# 0.00 = background completely removed
# 0.15 = background strongly attenuated  ← current setting
# 0.30 = retains a moderate amount of context
# 0.40 = retains substantial context
# 1.00 = effectively no ROI attenuation

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

EFFICIENTNET_WEIGHTS_NAME = "IMAGENET1K_V1"
# Explicit enum name, rather than DEFAULT, prevents a future torchvision release
# from silently changing the pretrained checkpoint selected by this experiment.

EFFICIENTNET_MEAN = (0.485, 0.456, 0.406)
EFFICIENTNET_STD = (0.229, 0.224, 0.225)
# Mean/std values associated with EfficientNet_B0_Weights.IMAGENET1K_V1.
# Important: torchvision's complete reference transform also resizes to 256 and
# center-crops to 224. This pipeline deliberately resizes the COMPLETE aligned
# 256×256 canvas to 224×224 to avoid discarding peripheral MRI content and to
# preserve simple MONAI-mask alignment. Therefore only the normalization values
# and checkpoint are official; the spatial preprocessing is a documented custom
# adaptation that should be included in the methods and ablated if necessary.

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
            / (
                f"{MONAI_BUNDLE_NAME}_{MONAI_BUNDLE_VERSION}_"
                f"{MONAI_HF_REVISION[:8]}_fallback_torchscript.pt"
            )
        ),
    )
)
# Local TorchScript cache used ONLY by the reconstruction fallback:
#
#   official model.pt -> lazy MONAI import -> strict state-dict loading
#                     -> validated local TorchScript export
#
# The normal path loads the official bundle file ``models/model.ts`` directly,
# so this fallback cache usually never has to be created.

FORCE_REBUILD_MONAI_TORCHSCRIPT = False
# False (normal): use the verified official model.ts; consult a local fallback
# cache only when that official artifact cannot execute on the current PyTorch.
# True: deliberately ignore both TorchScript files and rebuild from model.pt.
# This is intended for diagnostics, not routine execution.

MONAI_RUNTIME_SOURCE = "not_loaded"
MONAI_RUNTIME_ARTIFACT_PATH = None
# Populated by load_and_validate_torchscript_segmenter() so run_metadata.json
# records the artifact actually executed, not merely the files present on disk.

# ---------------------------------------------------------------------------
# DATASET LOCATION
# ---------------------------------------------------------------------------

if os.path.exists("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"):
    DATASET_PATH = "/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"
else:
    DATASET_PATH = r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset"

if os.path.exists("/kaggle/working"):
    OUTPUT_ROOT = Path("/kaggle/working/cad_patient_pipeline_outputs")
else:
    try:
        OUTPUT_ROOT = (
            Path(__file__).resolve().parent / "cad_patient_pipeline_outputs"
        )
    except NameError:
        OUTPUT_ROOT = Path.cwd() / "cad_patient_pipeline_outputs"

# A deterministic configuration tag prevents one ablation from silently
# overwriting another. The shared feature cache remains outside the run folder
# because patient-level classifier settings do not change frozen embeddings.
_run_identity = {
    "strategy": CLASSIFICATION_STRATEGY,
    "random_seed": RANDOM_SEED,
    "n_splits": N_SPLITS,
    "cv_seed": CV_RANDOM_STATE,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "logistic_c": LOGISTIC_C,
    "logistic_max_iter": LOGISTIC_MAX_ITER,
    "use_patient_pca": USE_PATIENT_PCA,
    "pca_variance": PATIENT_PCA_EXPLAINED_VARIANCE,
    "img_size": IMG_SIZE,
    "monai_input_size": MONAI_INPUT_SIZE,
    "use_monai_roi": USE_MONAI_ROI,
    "monai_bundle": MONAI_BUNDLE_NAME,
    "monai_bundle_version": MONAI_BUNDLE_VERSION,
    "monai_hf_revision": MONAI_HF_REVISION,
    "monai_model_ts_sha256": MONAI_OFFICIAL_TORCHSCRIPT_SHA256,
    "force_rebuild_monai_torchscript": FORCE_REBUILD_MONAI_TORCHSCRIPT,
    "roi_dilation": MONAI_ROI_DILATION_KERNEL,
    "roi_background": MONAI_BACKGROUND_WEIGHT,
    "roi_min_area": MONAI_MIN_HEART_AREA_RATIO,
    "roi_max_area": MONAI_MAX_HEART_AREA_RATIO,
    "roi_min_peak": MONAI_MIN_PEAK_HEART_PROBABILITY,
    "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
    "use_cuda_amp": USE_CUDA_AMP,
    "batch_size": BATCH_SIZE,
    "use_quality_weights": USE_SLICE_QUALITY_WEIGHTS,
    "quality_min_weight": SLICE_QUALITY_MIN_WEIGHT,
}
RUN_CONFIGURATION_TAG = hashlib.sha256(
    json.dumps(_run_identity, sort_keys=True).encode("utf-8")
).hexdigest()[:10]
RUN_NAME = f"{CLASSIFICATION_STRATEGY}__{RUN_CONFIGURATION_TAG}"

OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
FEATURE_CACHE_ROOT = OUTPUT_ROOT / "feature_cache"

# Keep every debug image directly in one flat folder, independent of the
# configuration-specific result and cache directories.
if os.path.exists("/kaggle/working"):
    DEBUG_OUTPUT_DIR = Path("/kaggle/working/debug_output")
else:
    try:
        DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "debug_output"
    except NameError:
        DEBUG_OUTPUT_DIR = Path.cwd() / "debug_output"


# =============================
# CONFIGURATION VALIDATION
# =============================

# This validation function is intentionally called before dataset scanning,
# model loading, downloading, GPU allocation, or feature extraction. A malformed
# setting should fail immediately, before the pipeline spends time or creates
# cache/output files that could later be mistaken for a valid experiment.
def validate_configuration():
    """Fail early for settings that would create invalid or ambiguous runs."""


    # ------------------------------------------------------------------
    # 1. Validate the top-level supervised-learning strategy.
    # ------------------------------------------------------------------
    # Only named, reviewed strategies are accepted. Rejecting arbitrary strings
    # prevents a typo from silently selecting an unintended branch later.

    valid_strategies = {
        "patient_embedding",
        "slice_probability_fusion",
    }

    if CLASSIFICATION_STRATEGY not in valid_strategies:
        raise ValueError(
            "CLASSIFICATION_STRATEGY must be one of "
            f"{sorted(valid_strategies)}, got {CLASSIFICATION_STRATEGY!r}."
        )

    # ------------------------------------------------------------------
    # 2. Validate ROI geometry and probability-gate settings.
    # ------------------------------------------------------------------
    # The dilation kernel must be odd because max-pooling uses symmetric
    # padding of ``kernel // 2`` and is expected to preserve H×W dimensions.
    if MONAI_ROI_DILATION_KERNEL <= 0 or MONAI_ROI_DILATION_KERNEL % 2 == 0:
        raise ValueError(
            "MONAI_ROI_DILATION_KERNEL must be a positive odd integer."
        )

    if not 0.0 <= MONAI_BACKGROUND_WEIGHT <= 1.0:
        raise ValueError("MONAI_BACKGROUND_WEIGHT must lie in [0,1].")

    if not (
        0.0 <= MONAI_MIN_HEART_AREA_RATIO
        < MONAI_MAX_HEART_AREA_RATIO
        <= 1.0
    ):
        raise ValueError("Invalid MONAI heart-area plausibility interval.")

    if not 0.0 <= MONAI_MIN_PEAK_HEART_PROBABILITY <= 1.0:
        raise ValueError(
            "MONAI_MIN_PEAK_HEART_PROBABILITY must lie in [0,1]."
        )

    # ------------------------------------------------------------------
    # 3. Validate the spatial contracts of both pretrained networks.
    # ------------------------------------------------------------------
    # The reviewed preprocessing, mask alignment, and feature-cache semantics
    # assume MONAI 256×256 input and EfficientNet 224×224 input.
    if IMG_SIZE <= 0 or MONAI_INPUT_SIZE <= 0:
        raise ValueError("Image sizes must be strictly positive.")

    if IMG_SIZE != 224:
        raise ValueError(
            "This reviewed pipeline is spatially documented and cache-keyed "
            "for EfficientNet-B0 at 224x224. Keep IMG_SIZE=224 unless all "
            "preprocessing assumptions are revalidated."
        )

    if MONAI_INPUT_SIZE != 256:
        raise ValueError(
            "The pinned ventricular_short_axis_3label bundle documents a "
            "256x256 input. Keep MONAI_INPUT_SIZE=256."
        )

    # ------------------------------------------------------------------
    # 4. Validate execution controls that affect batching and data loading.
    # ------------------------------------------------------------------
    # These checks prevent impossible DataLoader configurations. Batch size
    # affects throughput and memory consumption, but not the patient grouping.
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be strictly positive.")

    if DATALOADER_NUM_WORKERS < 0:
        raise ValueError("DATALOADER_NUM_WORKERS cannot be negative.")

    # Parse the debug selector now so an invalid value is detected before
    # the expensive extraction loop starts. The function returns a normalized
    # percentage, but validation needs only to prove that parsing succeeds.
    _parse_debug_indices(DEBUG_INDICES)

    # ------------------------------------------------------------------
    # 5. Validate optional heuristic weighting and duplicate-audit policies.
    # ------------------------------------------------------------------
    # A positive lower bound guarantees that enabling the heuristic does not
    # silently delete a slice by assigning it exactly zero influence.
    if not 0.0 < SLICE_QUALITY_MIN_WEIGHT <= 1.0:
        raise ValueError(
            "SLICE_QUALITY_MIN_WEIGHT must lie in (0,1]."
        )

    if (
        FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES
        and not AUDIT_EXACT_DECODED_PIXEL_DUPLICATES
    ):
        raise ValueError(
            "The strict duplicate policy requires duplicate auditing to be "
            "enabled."
        )

    # ------------------------------------------------------------------
    # 6. Validate model provenance identifiers before any network is loaded.
    # ------------------------------------------------------------------
    # SHA-256 strings must be syntactically valid hexadecimal digests. The
    # actual file content is checked later, after the artifact is located.
    if VERIFY_MONAI_ARTIFACT_SHA256:
        for digest_name, digest_value in (
            (
                "MONAI_OFFICIAL_TORCHSCRIPT_SHA256",
                MONAI_OFFICIAL_TORCHSCRIPT_SHA256,
            ),
            ("MONAI_MODEL_SHA256", MONAI_MODEL_SHA256),
        ):
            if len(digest_value) != 64 or any(
                character not in "0123456789abcdefABCDEF"
                for character in digest_value
            ):
                raise ValueError(
                    f"{digest_name} must be a 64-character hexadecimal "
                    "digest when verification is enabled."
                )

    if not MONAI_HF_REPO_ID or not MONAI_HF_REVISION:
        raise ValueError(
            "MONAI_HF_REPO_ID and MONAI_HF_REVISION must be non-empty."
        )

    # ------------------------------------------------------------------
    # 7. Validate the explicitly pinned EfficientNet checkpoint enum.
    # ------------------------------------------------------------------
    # Using a named enum rather than ``DEFAULT`` prevents a future torchvision
    # release from silently substituting a different pretrained checkpoint.
    if EFFICIENTNET_WEIGHTS_NAME not in (
        models.EfficientNet_B0_Weights.__members__
    ):
        raise ValueError(
            f"Unknown EfficientNet-B0 weight enum: "
            f"{EFFICIENTNET_WEIGHTS_NAME!r}."
        )

    # ------------------------------------------------------------------
    # 8. Validate fold-local classifier and uncertainty settings.
    # ------------------------------------------------------------------
    # These are basic syntactic checks. Class-specific patient counts are only
    # known after dataset discovery and are checked inside cross-validation.
    if N_SPLITS < 2:
        raise ValueError("N_SPLITS must be at least 2.")

    if LOGISTIC_C <= 0:
        raise ValueError("LOGISTIC_C must be strictly positive.")

    if LOGISTIC_MAX_ITER <= 0:
        raise ValueError("LOGISTIC_MAX_ITER must be strictly positive.")

    if not 0.0 < PATIENT_PCA_EXPLAINED_VARIANCE < 1.0:
        raise ValueError(
            "PATIENT_PCA_EXPLAINED_VARIANCE must lie strictly in (0,1) "
            "when it is passed to PCA as an explained-variance fraction."
        )

    if BOOTSTRAP_REPLICATES <= 0:
        raise ValueError("BOOTSTRAP_REPLICATES must be strictly positive.")


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

    # Convert before subtraction/division. Performing these operations on
    # uint8 could wrap negative values and would not preserve fractional output.
    image = image.astype(np.float32)

    # Compute the dynamic range independently for this JPEG. MRI JPEG
    # exports do not preserve a shared physical intensity unit across images.
    minimum = float(image.min())
    maximum = float(image.max())

    # A constant image has zero dynamic range. Returning an all-zero array
    # is deterministic and avoids a division-by-zero NaN propagation.
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)

    # Linear min-max scaling maps the darkest pixel to 0 and the brightest
    # pixel to 1 while preserving within-image intensity ordering.
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

    # Step 1: enforce the grayscale two-dimensional input contract. This
    # catches accidental H×W×C input before any geometry calculation.
    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, received shape {image.shape}."
        )

    # Step 2: read native geometry. These dimensions determine whether
    # downscaling is necessary; smaller images are never enlarged here.
    height, width = image.shape

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {image.shape}.")

    # Step 3: choose one isotropic scale factor for both axes. The leading
    # 1.0 caps the factor, so the function can downscale but cannot upsample.
    scale = min(
        1.0,
        MONAI_INPUT_SIZE / height,
        MONAI_INPUT_SIZE / width,
    )

    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))

    # Step 4: resize only when at least one dimension exceeds the canvas.
    # INTER_AREA is appropriate for downsampling and reduces aliasing.
    if resized_height != height or resized_width != width:
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    # Step 5: allocate the fixed MONAI canvas. Zero corresponds to the
    # minimum of the already normalized intensity range and acts as padding.
    canvas = np.zeros(
        (MONAI_INPUT_SIZE, MONAI_INPUT_SIZE),
        dtype=np.float32,
    )

    # Step 6: compute centered integer offsets. An odd unused margin leaves
    # one extra padding pixel on the bottom or right, which is deterministic.
    top = (MONAI_INPUT_SIZE - resized_height) // 2
    left = (MONAI_INPUT_SIZE - resized_width) // 2

    # Step 7: copy the complete resized field of view into the centered
    # region. No cropping is performed by this helper.
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
            Patient-scoped folder-defined series proxy, for example
            Directory_24/SR_3. This is not a recovered DICOM UID.

        sample_index:
            Stable integer index in ``samples``. It is used only to select
            deterministic debug examples from the complete image list.

        decoded_pixel_hash:
            SHA-256 of the decoded uint8 grayscale pixel matrix plus its shape.
            It supports an exact-pixel duplicate audit; it is not a perceptual
            or near-duplicate hash.

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


        # Store only lightweight paths and metadata. JPEG decoding is deferred
        # to ``__getitem__`` so DataLoader controls when each image is read.

        self.samples = samples
        # List containing:
        #   (image_path, binary_label, patient_id, series_id)

        self.transform = transform
        # Classifier-side conversion from NumPy HWC to PyTorch CHW.
        # ImageNet normalization is deliberately deferred until after ROI
        # weighting.

    def __len__(self):


        # DataLoader uses this exact count to determine epoch/batch coverage.
        # Because ``shuffle=False`` during extraction, indices remain aligned
        # with ``samples`` and with deterministic debug selection.

        return len(self.samples)

    def __getitem__(self, idx):


        # Resolve the immutable metadata row first. The class label and grouping
        # identifiers are returned unchanged; they do not influence pixels.

        img_path, label, patient_id, series_id = self.samples[idx]

        # =========================================================
        # LOAD RAW GRAYSCALE MRI JPEG
        # =========================================================

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(
                f"OpenCV could not read the MRI image: {img_path}"
            )

        # Hash the decoded uint8 matrix BEFORE any normalization or resizing.
        # Shape is included so two byte streams with different geometry cannot
        # collide merely because their flattened bytes happen to match.
        # The duplicate hash is deliberately computed on decoded pixels rather
        # than JPEG file bytes. Two JPEG containers with different metadata but
        # exactly identical decoded matrices therefore receive the same hash.
        pixel_digest = hashlib.sha256()
        pixel_digest.update(
            np.asarray(image.shape, dtype=np.int32).tobytes()
        )
        pixel_digest.update(image.tobytes(order="C"))
        decoded_pixel_hash = pixel_digest.hexdigest()

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

        # Convert HWC NumPy layout to CHW tensor layout. No ImageNet
        # normalization is applied here because ROI weighting must operate on
        # interpretable [0,1] intensities first.
        if self.transform:
            classification_image = self.transform(classification_image)
        else:
            classification_image = torch.from_numpy(
                classification_image
            ).permute(2, 0, 1)

        # Return every tensor and metadata field in a fixed order. The default
        # PyTorch collate function stacks tensors and keeps strings as lists,
        # which is exactly what ``extract_features`` expects.
        return (
            classification_image,
            monai_image,
            label,
            patient_id,
            series_id,
            idx,
            decoded_pixel_hash,
        )


# =============================
# LOAD DATA
# =============================

def load_samples(root_dir):
    """
    Discover MRI images and attach patient + folder-defined series identifiers.

    DATASET COHORT VS. COMPUTATIONAL PATIENT UNIT
    ----------------------------------------------
    The accompanying Scientific Reports paper reports 1,224 original
    participants (722 healthy, 502 CAD) and 63,648 CMR images. Those paper-level
    numbers are descriptive metadata; they are NOT used by this function to
    manufacture patient IDs or expected folder counts.

    PATIENT UNIT -- VALIDATED FOR THIS DATASET RELEASE
    --------------------------------------------------
    ``patient_id`` is exactly the immediate ``Directory_*`` folder name.
    ``SR_*`` and ``series*`` child folders remain folder-defined containers and
    never become patients. The Normal/Sick parent folder supplies the class
    label but is not part of patient_id.

    The function explicitly rejects a Directory_* name that appears under both
    Normal and Sick, because patient_id must be globally unambiguous.

    SERIES PROXY -- NOT A VALIDATED DICOM SERIES UID
    --------------------------------------------------
    The first directory level below the patient is treated as an operational
    series proxy. This preserves the released folder organization, but JPEG
    exports do not expose enough DICOM metadata to prove that each child folder
    is exactly one acquisition SeriesInstanceUID. Nested folders inside one
    immediate child are intentionally kept in the same proxy.

    Example:

        Sick/Directory_24/SR_3/image001.jpg
        Sick/Directory_24/SR_3/subfolder/image002.jpg

    Both images remain in the same series-proxy identifier:

        Directory_24/SR_3

    This identifier means "patient Directory_24, child folder SR_3"; it must not
    be interpreted as a DICOM series UID. Images directly inside Directory_*
    receive the special series ID
    ``<patient_id>/__ROOT__``.

    The function returns:
        (image_path, binary_label, patient_id, series_id)
    """

    # ``samples`` preserves one row per discovered image. The two maps/sets
    # below enforce global patient identity and label consistency independently
    # of how many series folders or images each patient contains.
    samples = []
    discovered_patients = set()
    patient_to_class = {}

    # Stage 1: traverse the two expected top-level class directories in a
    # fixed order. Sorting at every lower level makes discovery deterministic.
    for class_name in ["Normal", "Sick"]:

        label = 0 if class_name == "Normal" else 1
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            raise FileNotFoundError(
                f"Missing expected class directory: {class_path}"
            )

        # Stage 2: inspect immediate children of the class directory. Only
        # Directory_* folders are eligible to become computational patients.
        for directory in sorted(os.listdir(class_path)):

            patient_path = os.path.join(class_path, directory)

            if not os.path.isdir(patient_path):
                continue

            if (directory.lower() != "directory_1" and directory.lower() != "directory_17"
                    # and directory.lower() != "directory_2" and directory.lower() != "directory_18"
            ):
                continue

            # if not directory.lower().startswith("directory_"):
            #     continue

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

            # Record the globally unique patient-to-class relation before any
            # image rows are appended. This makes a cross-class collision fail
            # deterministically at the first conflicting folder.
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
            # Find immediate child directories. Each child is treated as one
            # folder-defined series PROXY. os.walk is used only to collect
            # images recursively without splitting nested subfolders again.
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

                for root, nested_dirs, files in os.walk(series_path):

                    # os.walk does not guarantee directory order. Sorting it
                    # makes sample order, cache fingerprints and debug indices
                    # reproducible across filesystems.
                    nested_dirs.sort()

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
                    # Empty child folders are ignored rather than becoming
                    # artificial series proxies with zero observations.
                    continue

    # Stage 3: perform cohort-level existence checks after traversal. Empty
    # datasets or an incorrect root path must never continue into model loading.
    if not samples:
        raise RuntimeError(
            f"No MRI images were discovered under dataset path: {root_dir}"
        )

    if not discovered_patients:
        raise RuntimeError(
            "No Directory_* patient folders were discovered. "
            "Verify the dataset path and folder structure."
        )

    # Stage 4: verify the final image-level table. This second check ensures
    # that every appended row for one patient carries the same binary label.
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

    # Stage 5: print counts at the patient, image, and folder-proxy levels.
    # The patient count is the effective labeled sample size used by evaluation.
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
    print(f"  Series proxies: {len(set(sample[3] for sample in samples))}")

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

    # Stream the artifact in 1 MiB chunks. This keeps memory usage bounded
    # even for a large checkpoint and produces the same digest as one-shot read.
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def locate_monai_bundle_root():
    """Locate the requested bundle root without accepting an unrelated model.

    Preferred layout::

        MONAI_BUNDLE_DIR/
            ventricular_short_axis_3label/
                models/model.ts
                configs/metadata.json

    ``models/model.pt`` may also be present for the reconstruction fallback.
    Older MONAI download utilities can produce a slightly different nesting
    layout, so a deterministic recursive search is retained. A candidate is
    accepted only when the requested bundle slug occurs in its path.
    """

    # First try the canonical layout because it is unambiguous and avoids a
    # potentially expensive recursive search through old bundle directories.
    direct_root = MONAI_BUNDLE_DIR / MONAI_BUNDLE_NAME

    if any(
        (direct_root / relative_path).is_file()
        for relative_path in (
            Path("models/model.ts"),
            Path("models/model.pt"),
        )
    ):
        return direct_root

    # If the canonical location is absent, search deterministic legacy
    # layouts. Candidate paths still must contain the requested bundle slug.
    if MONAI_BUNDLE_DIR.exists():
        candidate_artifacts = []

        # Prefer the official TorchScript artifact when several old layouts
        # coexist, then fall back to the state-dict checkpoint.
        for filename in ("model.ts", "model.pt"):
            candidate_artifacts.extend(
                sorted(MONAI_BUNDLE_DIR.rglob(filename))
            )

        for artifact_path in candidate_artifacts:
            if artifact_path.parent.name != "models":
                continue

            candidate_root = artifact_path.parent.parent

            if MONAI_BUNDLE_NAME in candidate_root.parts:
                return candidate_root

    return direct_root


def validate_monai_bundle_metadata(bundle_root):
    """Validate official metadata against the pinned model assumptions.

    The normal download requests metadata.json explicitly. A manually supplied
    fallback can omit it, but that weakens provenance; therefore a missing file
    emits a warning rather than silently pretending that the version was
    verified. Present metadata must match the pinned version and I/O contract.
    """

    # Metadata validation checks declared provenance and tensor contracts;
    # it does not replace cryptographic verification of the model file itself.
    metadata_path = bundle_root / "configs" / "metadata.json"

    if not metadata_path.is_file():
        print(
            "WARNING: MONAI metadata.json is missing. Artifact SHA-256 and "
            "runtime output shape can still be checked, but the declared "
            "bundle version and I/O metadata cannot be verified."
        )
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not parse MONAI bundle metadata: {metadata_path}"
        ) from exc

    version = str(metadata.get("version", ""))
    if version != MONAI_BUNDLE_VERSION:
        raise RuntimeError(
            "MONAI bundle version mismatch: "
            f"expected {MONAI_BUNDLE_VERSION}, metadata reports {version!r}."
        )

    network_format = metadata.get("network_data_format", {})
    input_image = network_format.get("inputs", {}).get("image", {})
    output_pred = network_format.get("outputs", {}).get("pred", {})

    declared_input_channels = input_image.get("num_channels")
    declared_input_shape = input_image.get("spatial_shape")
    declared_output_channels = output_pred.get("num_channels")
    declared_output_shape = output_pred.get("spatial_shape")

    if declared_input_channels not in (None, 1):
        raise RuntimeError(
            "Unexpected MONAI metadata input-channel count: "
            f"{declared_input_channels}."
        )
    if declared_input_shape not in (None, [256, 256], (256, 256)):
        raise RuntimeError(
            "Unexpected MONAI metadata input spatial shape: "
            f"{declared_input_shape}."
        )
    if declared_output_channels not in (None, 4):
        raise RuntimeError(
            "Unexpected MONAI metadata output-channel count: "
            f"{declared_output_channels}."
        )
    if declared_output_shape not in (None, [256, 256], (256, 256)):
        raise RuntimeError(
            "Unexpected MONAI metadata output spatial shape: "
            f"{declared_output_shape}."
        )


def validate_monai_train_config(bundle_root):
    """Check that fallback reconstruction matches the official train config."""

    # The fallback architecture is hard-coded below, so train.json is used
    # as an independent contract check before official weights are loaded.
    train_path = bundle_root / "configs" / "train.json"

    if not train_path.is_file():
        raise FileNotFoundError(
            "MONAI fallback reconstruction requires configs/train.json so the "
            f"hard-coded architecture can be checked: {train_path}"
        )

    try:
        train_config = json.loads(train_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not parse MONAI train configuration: {train_path}"
        ) from exc

    network_def = train_config.get("network_def", {})
    expected = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 4,
        "channels": [16, 32, 64, 128, 256],
        "strides": [2, 2, 2, 2],
        "num_res_units": 2,
    }

    for key, expected_value in expected.items():
        actual_value = network_def.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                "MONAI train.json architecture mismatch for "
                f"{key!r}: expected {expected_value!r}, got {actual_value!r}."
            )


def verify_monai_artifact_sha256(path, expected_sha256, artifact_label):
    """Verify a pinned MONAI artifact before it influences extracted features."""

    if not VERIFY_MONAI_ARTIFACT_SHA256:
        return None

    if not expected_sha256:
        raise RuntimeError(
            f"SHA-256 verification is enabled but no digest is configured for "
            f"{artifact_label}."
        )

    # Calculate the digest immediately before use. A cached filename alone
    # is not sufficient evidence that the intended bytes are present.
    actual_sha256 = sha256_file(path)

    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"{artifact_label} SHA-256 mismatch. Expected "
            f"{expected_sha256}, got {actual_sha256}. Delete the local file "
            "and download the pinned artifact again."
        )

    return actual_sha256


def ensure_monai_bundle(required_relative_paths=None):
    """Ensure selected files from the immutable MONAI bundle are local.

    The ordinary path requests only the official TorchScript model and metadata,
    avoiding both a MONAI import and an unnecessary duplicate checkpoint
    download. The state dict and train configuration are downloaded later only
    if fallback reconstruction is actually needed.

    ``huggingface_hub.snapshot_download`` is pinned to ``MONAI_HF_REVISION`` and
    receives ``allow_patterns`` so unrelated repository files are not fetched.
    """

    # Normalize the requested file list first. The ordinary execution path
    # intentionally requests only model.ts and metadata.json.
    if required_relative_paths is None:
        required_relative_paths = (
            "models/model.ts",
            "configs/metadata.json",
        )

    required_relative_paths = tuple(
        dict.fromkeys(str(path) for path in required_relative_paths)
    )

    # Resolve a local candidate and identify exactly which requested files
    # are absent before deciding whether network access is necessary.
    bundle_root = locate_monai_bundle_root()
    missing = [
        relative_path
        for relative_path in required_relative_paths
        if not (bundle_root / relative_path).is_file()
    ]

    if not missing:
        validate_monai_bundle_metadata(bundle_root)
        print(f"Using cached pinned MONAI bundle files: {bundle_root}")
        return bundle_root

    if not AUTO_DOWNLOAD_MONAI_BUNDLE:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Required pinned MONAI bundle files are missing and automatic "
            f"download is disabled under {bundle_root}:\n  - {formatted}"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Pinned MONAI files are missing. Install the lightweight download "
            "dependency with: pip install huggingface_hub"
        ) from exc

    bundle_root.mkdir(parents=True, exist_ok=True)

    print(
        "Downloading missing MONAI bundle files from the pinned repository "
        f"revision {MONAI_HF_REVISION[:8]}: {', '.join(missing)}"
    )

    try:
        snapshot_download(
            repo_id=MONAI_HF_REPO_ID,
            revision=MONAI_HF_REVISION,
            local_dir=str(bundle_root),
            allow_patterns=list(required_relative_paths),
        )
    except Exception as exc:
        raise RuntimeError(
            "Automatic pinned MONAI download failed. Verify internet access "
            f"and repository availability for {MONAI_HF_REPO_ID} at revision "
            f"{MONAI_HF_REVISION}. For offline execution, copy the requested "
            f"files manually under {bundle_root}."
        ) from exc

    still_missing = [
        relative_path
        for relative_path in required_relative_paths
        if not (bundle_root / relative_path).is_file()
    ]

    if still_missing:
        formatted = "\n  - ".join(still_missing)
        raise FileNotFoundError(
            "The pinned download call completed, but required MONAI files "
            f"remain missing under {bundle_root}:\n  - {formatted}"
        )

    validate_monai_bundle_metadata(bundle_root)
    print(f"Pinned MONAI files cached for future runs: {bundle_root}")
    return bundle_root


def load_and_validate_torchscript_segmenter(path, source_description):
    """Load TorchScript and validate its fixed inference contract immediately."""

    global MONAI_RUNTIME_SOURCE, MONAI_RUNTIME_ARTIFACT_PATH

    # Load directly onto the selected runtime device. The subsequent zero-
    # input test validates executability, output type, shape, and finite values.
    network = torch.jit.load(
        str(path),
        map_location=DEVICE,
    )
    network.eval()

    try:
        network.requires_grad_(False)
    except (AttributeError, RuntimeError):
        # Some TorchScript module types do not expose this mutator. Inference is
        # still protected globally by torch.inference_mode in the caller path.
        pass

    example_input = torch.zeros(
        1,
        1,
        MONAI_INPUT_SIZE,
        MONAI_INPUT_SIZE,
        device=DEVICE,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        example_output = network(example_input)

    if not isinstance(example_output, torch.Tensor):
        raise RuntimeError(
            f"{source_description} returned {type(example_output)} instead of "
            "a tensor."
        )

    expected_shape = (
        1,
        4,
        MONAI_INPUT_SIZE,
        MONAI_INPUT_SIZE,
    )

    if tuple(example_output.shape) != expected_shape:
        raise RuntimeError(
            f"{source_description} output shape mismatch: expected "
            f"{expected_shape}, got {tuple(example_output.shape)}."
        )

    if not torch.isfinite(example_output).all():
        raise RuntimeError(
            f"{source_description} produced non-finite values on a zero-input "
            "sanity check."
        )

    MONAI_RUNTIME_SOURCE = str(source_description)
    MONAI_RUNTIME_ARTIFACT_PATH = str(Path(path).resolve())

    print(f"Loaded and validated {source_description}: {path}")
    return network


def load_checkpoint_state_dict(path):
    """
    Load an official PyTorch checkpoint as weights only.

    weights_only=True prevents arbitrary objects from being reconstructed from
    the checkpoint. This script deliberately refuses an old PyTorch release that
    lacks the argument instead of silently falling back to unrestricted pickle
    loading. Upgrade PyTorch if this call is unsupported.
    """

    # Use restricted ``weights_only`` deserialization. A checkpoint is data,
    # not trusted executable Python, and unrestricted pickle loading is refused.
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not support torch.load(..., "
            "weights_only=True). Upgrade PyTorch rather than loading the "
            "checkpoint through unrestricted pickle deserialization."
        ) from exc

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
    """Load the pinned ventricular segmenter without a normal MONAI import.

    Loading order when ``FORCE_REBUILD_MONAI_TORCHSCRIPT=False``:

        1. The official pinned bundle artifact ``models/model.ts``.
        2. A previously generated local fallback TorchScript cache, but only if
           the official artifact cannot be executed by the current PyTorch.
        3. Reconstruction from the pinned ``models/model.pt`` state dict.

    Step 1 is the deterministic normal path for both first and later runs. It
    avoids importing MONAI entirely. Step 3 is retained only for compatibility
    recovery and validates the hard-coded architecture against train.json before
    strict weight loading and local TorchScript export.
    """

    # Preserve the original official-artifact failure so a later fallback
    # error can report useful context without hiding the primary problem.
    official_failure = None

    # =========================================================
    # NORMAL PATH: OFFICIAL model.ts, NO MONAI IMPORT
    # =========================================================

    if not FORCE_REBUILD_MONAI_TORCHSCRIPT:
        bundle_root = ensure_monai_bundle(
            (
                "models/model.ts",
                "configs/metadata.json",
            )
        )
        official_torchscript_path = bundle_root / "models" / "model.ts"

        verify_monai_artifact_sha256(
            official_torchscript_path,
            MONAI_OFFICIAL_TORCHSCRIPT_SHA256,
            "Official MONAI models/model.ts",
        )

        try:
            return load_and_validate_torchscript_segmenter(
                official_torchscript_path,
                "official pinned MONAI TorchScript segmenter",
            )
        except Exception as exc:
            official_failure = exc
            print(
                "WARNING: the official pinned model.ts passed file provenance "
                "checks but could not be executed by this PyTorch build. "
                "A previously reconstructed local cache will be tried next. "
                f"Reason: {exc}"
            )

    # =========================================================
    # OPTIONAL LOCAL FALLBACK CACHE: NO MONAI IMPORT
    # =========================================================

    if (
        MONAI_TORCHSCRIPT_PATH.is_file()
        and not FORCE_REBUILD_MONAI_TORCHSCRIPT
    ):
        try:
            return load_and_validate_torchscript_segmenter(
                MONAI_TORCHSCRIPT_PATH,
                "locally reconstructed MONAI TorchScript cache",
            )
        except Exception as exc:
            print(
                "WARNING: local fallback TorchScript cache could not be used; "
                "strict reconstruction from model.pt will be attempted. "
                f"Reason: {exc}"
            )

    # =========================================================
    # EXCEPTIONAL FALLBACK: model.pt + LAZY MONAI IMPORT
    # =========================================================

    bundle_root = ensure_monai_bundle(
        (
            "models/model.pt",
            "configs/train.json",
            "configs/metadata.json",
        )
    )
    model_path = bundle_root / "models" / "model.pt"

    verify_monai_artifact_sha256(
        model_path,
        MONAI_MODEL_SHA256,
        "Official MONAI models/model.pt",
    )
    validate_monai_train_config(bundle_root)

    print(
        "Reconstructing the pinned MONAI UNet from model.pt. This exceptional "
        "fallback imports MONAI once and creates a validated local TorchScript "
        "cache."
    )

    try:
        from monai.networks.nets import UNet as MONAIUNet
    except ImportError as exc:
        reason = (
            f" Official model.ts failure: {official_failure}"
            if official_failure is not None
            else ""
        )
        raise ImportError(
            "Fallback reconstruction requires MONAI. Install it with: "
            f"pip install monai==1.6.0.{reason}"
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

    # strict=True fails on missing/unexpected keys instead of silently running a
    # partially initialized segmentation network.
    network.load_state_dict(state_dict, strict=True)
    network.eval()
    network.requires_grad_(False)

    # Trace on CPU using the exact fixed spatial input used by this pipeline.
    # CPU export avoids serializing a device-specific CUDA graph.
    example_input = torch.zeros(
        1,
        1,
        MONAI_INPUT_SIZE,
        MONAI_INPUT_SIZE,
        dtype=torch.float32,
    )

    network = network.cpu()

    with torch.inference_mode():
        reference_output = network(example_input)

        traced_network = torch.jit.trace(
            network,
            example_input,
            strict=False,
        )

        traced_output = traced_network(example_input)

    if tuple(reference_output.shape) != (
        1,
        4,
        MONAI_INPUT_SIZE,
        MONAI_INPUT_SIZE,
    ):
        raise RuntimeError(
            "Reconstructed MONAI network produced an unexpected output shape: "
            f"{tuple(reference_output.shape)}."
        )

    # Verify that tracing preserved numerical output before the graph is saved.
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
            "Fallback TorchScript validation failed: traced output differs "
            "from the reconstructed network (max abs difference="
            f"{max_difference:.6g})."
        )

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
        "Saved validated fallback MONAI TorchScript cache: "
        f"{MONAI_TORCHSCRIPT_PATH}"
    )

    return load_and_validate_torchscript_segmenter(
        MONAI_TORCHSCRIPT_PATH,
        "newly reconstructed MONAI TorchScript cache",
    )


# =============================
# PIPELINE STEP 3
# MONAI PROBABILITY MAP + SOFT ROI
# =============================

@torch.inference_mode()
def predict_monai_heart_masks(
    monai_images,
    classifier_size,
    monai_segmenter,
):
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

        mean_foreground_probability:
            Mean non-background probability over pixels assigned by argmax to a
            cardiac class. This is recorded for QC but is not treated as proof
            of anatomical correctness and is not currently part of the gate.

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

    # Step 1: run frozen segmentation inference. ``torch.inference_mode``
    # disables autograd bookkeeping and guarantees this function cannot train
    # or update the MONAI network.
    logits = monai_segmenter(monai_images)

    if logits.ndim != 4 or logits.shape[1] != 4:
        raise RuntimeError(
            "Unexpected MONAI output shape. Expected [B,4,H,W], got "
            f"{tuple(logits.shape)}."
        )

    # Softmax is intentionally evaluated in float32 even when CUDA AMP is
    # enabled, because probability/gating calculations are more numerically
    # stable than in float16.
    # Step 2: convert mutually exclusive class logits to probabilities.
    # Float32 is used for the softmax and all gate statistics even under AMP.
    class_probabilities = torch.softmax(logits.float(), dim=1)

    # Step 3: combine all three foreground anatomy classes into one soft
    # cardiac probability. Background channel 0 is intentionally excluded.
    heart_probability = class_probabilities[:, 1:, :, :].sum(
        dim=1,
        keepdim=True,
    ).clamp(0.0, 1.0)

    # Step 4: derive a hard class assignment only for plausibility/QC. The
    # soft probability, not this binary mask, controls ROI intensity weighting.
    class_map = torch.argmax(
        class_probabilities,
        dim=1,
        keepdim=True,
    )

    hard_mask_256 = (class_map > 0).float()

    # Step 5: calculate per-slice diagnostics on the original 256×256 mask.
    # These metrics describe the model output; they are not accuracy estimates.
    area_ratio = hard_mask_256.mean(dim=(1, 2, 3))
    peak_probability = heart_probability.amax(dim=(1, 2, 3))

    foreground_pixel_count = hard_mask_256.sum(dim=(1, 2, 3))
    mean_foreground_probability = (
        (heart_probability * hard_mask_256).sum(dim=(1, 2, 3))
        / foreground_pixel_count.clamp_min(1.0)
    )

    # Step 6: apply the predeclared plausibility gate independently to every
    # slice. The result is a Boolean selector for ROI versus full-image fallback.
    valid_mask = (
        (area_ratio >= MONAI_MIN_HEART_AREA_RATIO)
        & (area_ratio <= MONAI_MAX_HEART_AREA_RATIO)
        & (peak_probability >= MONAI_MIN_PEAK_HEART_PROBABILITY)
    )

    # Max pooling expands the ROI around the predicted ventricles and
    # myocardium. This prevents a narrow segmentation from cutting away nearby
    # diagnostically useful cardiac tissue.
    # Step 7: dilate both soft and hard masks with stride-one max pooling.
    # The operation adds a contextual margin around the predicted ventricles.
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

    # Step 8: map masks from MONAI coordinates to the aligned 224×224
    # EfficientNet canvas. Bilinear interpolation preserves a smooth soft map.
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
        mean_foreground_probability,
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
    may not generalize to every image type in the CAD JPEG release.
    """

    # Step 1: validate image and mask tensor contracts before relying on
    # broadcasting. Shape errors here usually indicate preprocessing mismatch.
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

    # Step 2: transform probability into an attenuation field. A pixel with
    # P(heart)=0 retains ``MONAI_BACKGROUND_WEIGHT`` of its intensity, while a
    # pixel with P(heart)=1 retains its full intensity.
    roi_weight = (
        MONAI_BACKGROUND_WEIGHT
        + (1.0 - MONAI_BACKGROUND_WEIGHT) * roi_probability
    )

    # Step 3: replicate the one-channel spatial weight over the three
    # identical grayscale channels expected by EfficientNet.
    roi_weight = roi_weight.repeat(1, 3, 1, 1)

    weighted_images = images * roi_weight

    # Step 4: reshape the per-slice gate so it broadcasts over channel and
    # spatial dimensions without mixing decisions between batch elements.
    valid_mask = valid_mask.view(-1, 1, 1, 1)

    # torch.where applies the decision independently to every slice in a batch.
    return torch.where(valid_mask, weighted_images, images)


def normalize_for_efficientnet(images):
    """
    Apply the official ImageNet normalization for EfficientNet-B0.

    Input images must already be float tensors in [0,1]. ROI extraction is
    intentionally completed before this step.
    """

    # Construct per-channel constants on the same device and with the same
    # dtype as the input, avoiding implicit CPU/GPU copies or dtype promotion.
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
    mean_foreground_probabilities,
    labels,
    patient_ids,
    series_ids,
    sample_indices,
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
        - mean foreground confidence
        - ROI intensity standard deviation

    Examples are selected deterministically from the complete image list using
    DEBUG_INDICES (all images or a configured percentage).
    Visual review is mandatory before treating the pretrained masks as useful
    ROI proposals on this heterogeneous dataset. It still does not constitute a
    quantitative segmentation validation because no ground-truth masks exist.
    """

    images = images.detach().cpu()
    roi_probability = roi_probability.detach().cpu()
    hard_mask = hard_mask.detach().cpu()
    roi_images = roi_images.detach().cpu()
    scores = scores.detach().cpu()
    valid_masks = valid_masks.detach().cpu()
    area_ratios = area_ratios.detach().cpu()
    peak_probabilities = peak_probabilities.detach().cpu()
    mean_foreground_probabilities = (
        mean_foreground_probabilities.detach().cpu()
    )

    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(images.shape[0]):

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
            f"peak={peak_probabilities[i]:.3f}, "
            f"mean_fg={mean_foreground_probabilities[i]:.3f}"
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

        safe_series = str(series_ids[i]).replace("/", "__").replace("\\", "__")
        output_name = (
            f"label{int(labels[i])}_{patient_ids[i]}_{safe_series}_"
            f"sample{int(sample_indices[i])}.png"
        )

        plt.savefig(
            DEBUG_OUTPUT_DIR / output_name,
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


        # Resolve the exact pretrained-weight enum configured for this run. The
        # enum also documents the matching ImageNet normalization constants.

        super().__init__()

        weights = models.EfficientNet_B0_Weights[
            EFFICIENTNET_WEIGHTS_NAME
        ]

        self.model = models.efficientnet_b0(weights=weights)

        # Replace only the final classifier. The convolutional backbone and
        # global pooling remain intact, so forward() returns a 1280-D vector.
        self.model.classifier = nn.Identity()

    def forward(self, x):


        # No additional trainable layer is introduced here. Input tensors pass
        # directly through the frozen EfficientNet feature encoder.

        return self.model(x)



# =============================
# PIPELINE STEP 5 + 6
# FEATURE EXTRACTION + OPTIONAL SLICE WEIGHTING
# =============================

def _normalize_quality_weights(scores, minimum_weight=SLICE_QUALITY_MIN_WEIGHT):
    """
    Convert a heuristic per-slice score into bounded weights within ONE series proxy.

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

    # Mean=1 is only a convenient local convention. The downstream hierarchy
    # renormalizes weights inside each series proxy, so this common scale cancels
    # and is not relied upon to define Logistic Regression regularization.
    weights /= max(weights.mean(), 1e-8)

    return weights


def _parse_debug_indices(selection):
    """Return ``None`` for full selection or a percentage in [0, 100]."""

    if isinstance(selection, str):
        normalized = selection.strip().lower()
        if normalized == "full":
            return None
        if not normalized.endswith("%"):
            raise ValueError(
                'DEBUG_INDICES must be "full" or a percentage such as "10%".'
            )
        numeric_value = normalized[:-1].strip()
    elif (
        isinstance(selection, (int, float))
        and not isinstance(selection, bool)
    ):
        numeric_value = selection
    else:
        raise ValueError(
            'DEBUG_INDICES must be "full" or a percentage such as "10%".'
        )

    try:
        percentage = float(numeric_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            'DEBUG_INDICES must be "full" or a percentage such as "10%".'
        ) from error

    if not np.isfinite(percentage) or not 0.0 <= percentage <= 100.0:
        raise ValueError("DEBUG_INDICES percentage must lie in [0, 100].")

    return percentage


def choose_debug_sample_indices(samples, selection):
    """Choose all images or a deterministic percentage of the full image list."""

    percentage = _parse_debug_indices(selection)
    n_images = len(samples)

    if n_images == 0:
        return set()

    if percentage is None or percentage == 100.0:
        return set(range(n_images))

    if percentage == 0.0:
        return set()

    n_select = min(
        n_images,
        max(1, int(np.ceil(n_images * percentage / 100.0))),
    )
    return set(
        np.linspace(0, n_images - 1, num=n_select, dtype=int).tolist()
    )


def extract_features(
    dataset,
    monai_segmenter,
    feature_extractor,
    debug=False,
):
    """
    Run the frozen image-processing pipeline and return one embedding per slice.

    PIPELINE:
        JPEG
          ↓
        per-image intensity scaling
          ↓
        MONAI-compatible 256×256 canvas
          ↓
        optional MONAI ventricular segmentation
          ↓
        confidence gate when MONAI ROI is enabled
          ↓
        soft cardiac ROI / full-image fallback, or full-image ablation
          ↓
        EfficientNet-B0
          ↓
        1280-D embedding
          ↓
        optional series-local quality weight (default = 1)

    Class labels are never inputs to MONAI, EfficientNet, ROI gating or quality
    weighting. They are carried as metadata for debug figures, QC summaries,
    supervised fitting and stratified evaluation.

    This function retains every slice that is successfully decoded. An
    unreadable discovered file raises an explicit error instead of being silently
    skipped. The previous implementation selected slices above the per-series median standard
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
            [N] folder-defined series-proxy identifiers.
        quality_weights:
            [N] non-negative slice weights, normalized within each series proxy.
        monai_valid:
            [N] boolean segmentation-gate result.
        area_ratios, peak_probabilities, mean_foreground_probabilities:
            [N] MONAI QC signals. They are descriptive diagnostics, not
            segmentation-accuracy metrics.
        slice_scores:
            [N] ROI/full-image standard deviation used only by the optional
            heuristic weighting ablation.
        decoded_pixel_hashes:
            [N] exact decoded-grayscale-pixel SHA-256 strings used only for
            duplicate auditing.
    """

    # ------------------------------------------------------------------
    # Extraction stage 1: build a deterministic, non-shuffled DataLoader.
    # ------------------------------------------------------------------
    # Stable ordering is required because every returned array must stay aligned
    # with ``dataset.samples`` and its decoded-pixel duplicate hashes.
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(DATALOADER_NUM_WORKERS > 0),
    )

    # ------------------------------------------------------------------
    # Extraction stage 2: allocate append-only collectors.
    # ------------------------------------------------------------------
    # Features are accumulated batch-wise to avoid preallocating a potentially
    # very large matrix before the first successful model inference.
    all_features = []
    all_labels = []
    all_patients = []
    all_series = []
    all_scores = []
    all_monai_valid = []
    all_area_ratios = []
    all_peak_probabilities = []
    all_mean_foreground_probabilities = []
    all_decoded_pixel_hashes = []

    # ------------------------------------------------------------------
    # Extraction stage 3: precompute deterministic qualitative-QC indices.
    # ------------------------------------------------------------------
    # Debug selection is based on global sample indices, not batch positions, so
    # changing batch size does not change which images are visualized.
    debug_indices = (
        choose_debug_sample_indices(
            dataset.samples,
            DEBUG_INDICES,
        )
        if debug and USE_MONAI_ROI
        else set()
    )

    # Mixed precision is enabled only on CUDA. CPU execution remains
    # float32 because the chosen autocast device type below is explicitly CUDA.
    autocast_enabled = USE_CUDA_AMP and DEVICE == "cuda"

    with torch.inference_mode():

        # ----------------------------------------------------------------
        # Extraction stage 4: process every discovered slice exactly once.
        # ----------------------------------------------------------------
        # ``batch_idx`` is retained for traceability even though sample identity
        # is carried by the stable ``sample_indices`` tensor.
        for batch_idx, batch in enumerate(tqdm(loader)):

            (
                images,
                monai_images,
                labels,
                patient_ids,
                series_ids,
                sample_indices,
                decoded_pixel_hashes,
            ) = batch

            # Move only tensors to the runtime device. Patient/series strings and
            # decoded hashes stay on the host because networks do not consume them.
            images = images.to(DEVICE, non_blocking=True)
            monai_images = monai_images.to(DEVICE, non_blocking=True)

            # -------------------------------------------------------------
            # Extraction stage 5: create the classifier image for this batch.
            # -------------------------------------------------------------
            # ROI mode runs MONAI, evaluates plausibility, and chooses soft ROI
            # or full-image fallback per slice. Ablation mode bypasses MONAI.
            if USE_MONAI_ROI:
                if monai_segmenter is None:
                    raise RuntimeError(
                        "USE_MONAI_ROI=True but no MONAI segmenter was supplied."
                    )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    (
                        roi_probability,
                        hard_mask,
                        valid_mask,
                        area_ratio,
                        peak_probability,
                        mean_foreground_probability,
                    ) = predict_monai_heart_masks(
                        monai_images,
                        classifier_size=images.shape[-2:],
                        monai_segmenter=monai_segmenter,
                    )

                    roi_images = apply_confidence_gated_soft_roi(
                        images,
                        roi_probability,
                        valid_mask,
                    )
            else:
                batch_size = images.shape[0]
                roi_probability = torch.ones(
                    batch_size,
                    1,
                    *images.shape[-2:],
                    device=images.device,
                    dtype=images.dtype,
                )
                hard_mask = roi_probability.clone()
                valid_mask = torch.zeros(
                    batch_size,
                    device=images.device,
                    dtype=torch.bool,
                )
                area_ratio = torch.full(
                    (batch_size,),
                    float("nan"),
                    device=images.device,
                )
                peak_probability = torch.full_like(area_ratio, float("nan"))
                mean_foreground_probability = torch.full_like(
                    area_ratio,
                    float("nan"),
                )
                roi_images = images

            # -------------------------------------------------------------
            # Extraction stage 6: normalize and encode every retained image.
            # -------------------------------------------------------------
            # ROI weighting occurs before normalization. EfficientNet returns one
            # frozen 1280-D embedding per input slice.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                efficientnet_inputs = normalize_for_efficientnet(roi_images)
                features = feature_extractor(efficientnet_inputs)

            features = features.float()

            # ---------------------------------------------------------
            # Slice-quality signal
            # ---------------------------------------------------------
            # Standard deviation is only a proxy for image information. It is
            # NOT a clinical quality metric and must not be described as one.
            #
            # We calculate it before ImageNet normalization because the [0,1]
            # intensity scale is easier to interpret.
            # -------------------------------------------------------------
            # Extraction stage 7: calculate the optional heuristic score.
            # -------------------------------------------------------------
            # This statistic is always recorded for reproducibility, even when it
            # is not used to alter slice influence.
            scores = torch.std(
                roi_images,
                dim=(1, 2, 3),
            )

            # -------------------------------------------------------------
            # Extraction stage 8: save selected qualitative ROI figures.
            # -------------------------------------------------------------
            # Selection and plotting do not feed back into model inputs or labels.
            if debug_indices:
                selected_positions = [
                    position
                    for position, sample_index in enumerate(
                        sample_indices.tolist()
                    )
                    if int(sample_index) in debug_indices
                ]

                if selected_positions:
                    debug_visualization(
                        images[selected_positions],
                        roi_probability[selected_positions],
                        hard_mask[selected_positions],
                        roi_images[selected_positions],
                        scores[selected_positions],
                        valid_mask[selected_positions],
                        area_ratio[selected_positions],
                        peak_probability[selected_positions],
                        mean_foreground_probability[selected_positions],
                        labels[selected_positions],
                        [patient_ids[i] for i in selected_positions],
                        [series_ids[i] for i in selected_positions],
                        sample_indices[selected_positions],
                    )

            # -------------------------------------------------------------
            # Extraction stage 9: return tensors to CPU and append aligned rows.
            # -------------------------------------------------------------
            # Every collector receives exactly one entry per batch element.
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_patients.extend(list(patient_ids))
            all_series.extend(list(series_ids))
            all_scores.extend(scores.cpu().numpy().tolist())
            all_monai_valid.extend(valid_mask.cpu().numpy().tolist())
            all_area_ratios.extend(area_ratio.cpu().numpy().tolist())
            all_peak_probabilities.extend(
                peak_probability.cpu().numpy().tolist()
            )
            all_mean_foreground_probabilities.extend(
                mean_foreground_probability.cpu().numpy().tolist()
            )
            all_decoded_pixel_hashes.extend(list(decoded_pixel_hashes))

    # ------------------------------------------------------------------
    # Extraction stage 10: finalize batch collectors into NumPy arrays.
    # ------------------------------------------------------------------
    # A hard failure is preferable to returning empty arrays that might create
    # misleading downstream output files.
    if not all_features:
        raise RuntimeError("No slice features were extracted.")

    features = np.vstack(all_features)
    labels = np.asarray(all_labels, dtype=np.int64)
    patient_ids = np.asarray(all_patients)
    series_ids = np.asarray(all_series)
    scores = np.asarray(all_scores, dtype=np.float32)
    monai_valid = np.asarray(all_monai_valid, dtype=bool)
    area_ratios = np.asarray(all_area_ratios, dtype=np.float32)
    peak_probabilities = np.asarray(
        all_peak_probabilities,
        dtype=np.float32,
    )
    mean_foreground_probabilities = np.asarray(
        all_mean_foreground_probabilities,
        dtype=np.float32,
    )
    decoded_pixel_hashes = np.asarray(all_decoded_pixel_hashes)

    # Verify one-to-one row alignment across features, grouping metadata,
    # QC diagnostics, heuristic scores, and duplicate hashes.
    if not (
        len(features)
        == len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(scores)
        == len(monai_valid)
        == len(area_ratios)
        == len(peak_probabilities)
        == len(mean_foreground_probabilities)
        == len(decoded_pixel_hashes)
    ):
        raise RuntimeError(
            "Feature extraction produced arrays with inconsistent lengths."
        )

    # -------------------------------------------------------------
    # OPTIONAL QUALITY WEIGHTS
    # -------------------------------------------------------------
    # Equal slice weights are the default because standard deviation is only a
    # heuristic. If explicitly enabled, normalization is performed independently
    # inside each series proxy so scanners/exports with different contrast do not get a
    # global advantage merely because of their intensity distribution.
    # ------------------------------------------------------------------
    # Extraction stage 11: derive final per-slice influence weights.
    # ------------------------------------------------------------------
    # Equal weights are the reviewed default. The optional heuristic is
    # normalized separately within each patient-scoped series proxy.
    if USE_SLICE_QUALITY_WEIGHTS:
        quality_weights = np.zeros(len(scores), dtype=np.float64)

        for series_id in np.unique(series_ids):
            indices = np.flatnonzero(series_ids == series_id)
            quality_weights[indices] = _normalize_quality_weights(
                scores[indices]
            )
    else:
        quality_weights = np.ones(len(scores), dtype=np.float64)

    # Every patient must still have at least one series proxy and one slice.
    if set(patient_ids.tolist()) != set(
        patient_ids[quality_weights > 0].tolist()
    ):
        raise RuntimeError(
            "At least one patient received no positive slice weight."
        )

    valid_rate = float(monai_valid.mean()) if USE_MONAI_ROI else float("nan")

    if USE_MONAI_ROI:
        print(
            "MONAI plausible-mask rate: "
            f"{int(monai_valid.sum())}/{len(monai_valid)} ({valid_rate:.2%})"
        )
    else:
        print("MONAI ROI disabled: full-image EfficientNet ablation is active.")
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
        area_ratios,
        peak_probabilities,
        mean_foreground_probabilities,
        scores,
        decoded_pixel_hashes,
    )


# =============================
# PIPELINE STEP 7
# HIERARCHICAL FUSION (SLICE → SERIES → PATIENT)
# =============================

def weighted_log_odds_fusion(probabilities, weights=None):
    """
    Fuse probability-like model outputs in log-odds space, optionally weighted.

    Mathematical form:

        logit(p_fused) = Σ w_i * logit(p_i) / Σ w_i

    This is preferable to multiplying probabilities directly because the
    number of slices/series proxies does not automatically force the fused result
    toward 0 or 1.

    LIMITATION:
    The individual Logistic Regression ``predict_proba`` outputs are not
    guaranteed to be calibrated on new clinical data. Therefore the fused value
    is an aggregation score on a 0-to-1 scale, not a validated CAD posterior.
    """

    # Convert inputs to stable float64 NumPy arrays before clipping and
    # logarithms. Fusion is a lightweight CPU operation, independent of GPU AMP.
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.size == 0:
        raise ValueError("Cannot fuse an empty probability collection.")

    if weights is None:
        weights = np.ones_like(probabilities, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if probabilities.shape != weights.shape:
        raise ValueError("probabilities and weights must have the same shape.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probability-like inputs must be finite.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("fusion weights must be finite.")

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
        weighted series-proxy score
              ↓
        equal-weight series-proxy scores
              ↓
        patient score

    WHY TWO LEVELS?
    ---------------
    A patient may have different numbers of images in different folder
    proxies. If all slices were fused directly, a long proxy would dominate the
    patient merely
    because it contains more frames.

    The revised strategy therefore:
        1. weights slices by within-proxy quality;
        2. gives each series proxy one fused score;
        3. gives each series proxy equal influence at patient level.

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

    # Stage 1: obtain one positive-class model score per validation slice.
    # These scores are produced by a classifier fitted only on training patients.
    slice_probabilities = clf.predict_proba(features)[:, 1]

    # Stage 2: build a nested patient -> series -> slice-evidence structure
    # while rechecking that each patient has one consistent ground-truth label.
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

    # Stage 3: reduce each nested structure in two levels. First fuse slices
    # inside a series proxy; then fuse one score per proxy at patient level.
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

        # Equal weighting of folder proxies prevents one long child folder from
        # dominating simply because it contains more frames. It does not prove
        # that the proxies correspond to distinct acquisition types.
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

        class -> patient -> series proxy -> slice

    For slice i belonging to series proxy s of patient p and class c:

        w_i = (1 / N_patients_in_class_c)
              * (1 / N_series_for_patient_p)
              * (q_i / sum(q_j for j in series_s))

    where q_i is the optional within-proxy quality weight. If quality weighting
    is disabled, q_i = 1 and all slices inside a proxy share that proxy's total
    training influence equally.

    This gives two important invariances:

        1. A patient with more exported JPEGs does not dominate training.
        2. A patient with one very long proxy does not let that proxy dominate
           over the patient's shorter proxies.

    The class factor gives Normal and Sick equal total nominal weight even when
    the number of Directory_* patients differs between classes.

    GLOBAL WEIGHT SCALE MATTERS FOR REGULARIZED LOGISTIC REGRESSION
    ----------------------------------------------------------------
    Multiplying every sample weight by the same constant leaves weighted means
    unchanged, but it changes the data-loss/regularization balance of a model
    fitted with fixed ``C``. Therefore the final total weight is set to the
    number of unique training patients, making the effective regularization
    invariant to how many JPEG slices happen to be exported.
    """

    # Normalize all metadata arrays before building the class/patient/series
    # hierarchy. No ordering assumption is required for the weight calculation.
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

    if not np.all(np.isfinite(quality_weights)):
        raise ValueError("quality_weights must contain only finite values.")

    if np.any(quality_weights < 0):
        raise ValueError("quality_weights must be non-negative.")

    # Build explicit maps that enforce: one label per patient, one patient
    # per series-proxy identifier, and a known set of proxies per patient.
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
                f"Series proxy {series_id} is assigned to more than one patient."
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

    zero_weight_series = sorted(
        series_id
        for series_id, total in series_quality_sum.items()
        if total <= 0.0
    )
    if zero_weight_series:
        raise RuntimeError(
            "Every series proxy must have positive total slice weight. "
            f"Invalid series: {zero_weight_series[:10]}"
        )

    # Allocate the final slice-weight vector only after all hierarchy and
    # positive-total checks have succeeded.
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
                f"Patient {patient_id} unexpectedly has no series proxy."
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

    n_training_patients = len(patient_to_label)
    weights *= n_training_patients / max(weights.sum(), 1e-12)

    return weights


def aggregate_patient_embeddings(
    features,
    labels,
    patient_ids,
    series_ids,
    quality_weights,
):
    """Pool frozen slice embeddings to exactly one embedding per patient.

    Hierarchy:

        slice embeddings --weighted mean within series proxy--> series embedding
        series embeddings --------equal mean across proxies----> patient embedding

    This is the recommended default because the supervised classifier is fitted
    on one observation per validated Directory_* patient. Equal series-proxy
    weighting prevents a folder containing many JPEG frames from dominating.
    The series unit remains an operational folder proxy, not a DICOM UID.
    """

    # Convert all inputs to predictable NumPy dtypes and verify finiteness
    # before any weighted mean can hide an invalid value.
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    series_ids = np.asarray(series_ids)
    quality_weights = np.asarray(quality_weights, dtype=np.float64)

    if not np.all(np.isfinite(features)):
        raise ValueError("Patient-pooling features must be finite.")
    if not np.all(np.isfinite(quality_weights)):
        raise ValueError("Patient-pooling weights must be finite.")

    if not (
        len(features)
        == len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(quality_weights)
    ):
        raise ValueError("Patient-embedding arrays must have equal lengths.")

    patient_to_label = {}
    patient_to_series_indices = {}

    for index, (label, patient_id, series_id) in enumerate(
        zip(labels, patient_ids, series_ids)
    ):
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)

        previous = patient_to_label.get(patient_id)
        if previous is not None and previous != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent labels."
            )

        patient_to_label[patient_id] = label
        patient_to_series_indices.setdefault(patient_id, {})
        patient_to_series_indices[patient_id].setdefault(series_id, []).append(
            index
        )

    # Pool in deterministic sorted patient/series order so the returned
    # patient table can be compared directly with the CV patient table.
    pooled_embeddings = []
    pooled_labels = []
    pooled_patient_ids = []

    for patient_id in sorted(patient_to_series_indices):
        series_embeddings = []

        for series_id in sorted(patient_to_series_indices[patient_id]):
            indices = np.asarray(
                patient_to_series_indices[patient_id][series_id],
                dtype=np.int64,
            )
            weights = quality_weights[indices]

            if np.any(weights < 0) or not np.any(weights > 0):
                raise RuntimeError(
                    f"Series proxy {series_id} has invalid slice weights."
                )

            series_embedding = np.average(
                features[indices],
                axis=0,
                weights=weights,
            )
            series_embeddings.append(series_embedding)

        patient_embedding = np.mean(
            np.stack(series_embeddings, axis=0),
            axis=0,
        ).astype(np.float32, copy=False)

        pooled_embeddings.append(patient_embedding)
        pooled_labels.append(patient_to_label[patient_id])
        pooled_patient_ids.append(patient_id)

    return (
        np.stack(pooled_embeddings, axis=0),
        np.asarray(pooled_labels, dtype=np.int64),
        np.asarray(pooled_patient_ids),
    )


def compute_balanced_patient_weights(labels):
    """Give each class equal total weight while keeping total mass = patients."""

    labels = np.asarray(labels, dtype=np.int64)
    # Count patient rows, not slices. Each class will receive half of the
    # total supervised weight mass inside the current training fold.
    counts = np.bincount(labels, minlength=2)

    if int(counts.min()) <= 0:
        raise RuntimeError("Training fold must contain both classes.")

    n_patients = len(labels)
    weights = np.asarray(
        [n_patients / (2.0 * counts[int(label)]) for label in labels],
        dtype=np.float64,
    )

    return weights


def build_classifier():
    """Create a fresh fold-local preprocessing/classification pipeline.

    For the recommended patient-embedding strategy, PCA is optionally fitted
    inside the training fold. For the legacy slice strategy it is omitted to
    avoid an expensive dense PCA over tens of thousands of correlated slices.
    """

    # Start with fold-local standardization. The scaler is fitted anew for
    # every fold and therefore never sees validation-patient embeddings.
    steps = [("scaler", StandardScaler())]

    # PCA is available only for the one-row-per-patient strategy. Fitting a
    # dense PCA to tens of thousands of correlated slice rows is intentionally
    # avoided in the legacy branch.
    if CLASSIFICATION_STRATEGY == "patient_embedding" and USE_PATIENT_PCA:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=PATIENT_PCA_EXPLAINED_VARIANCE,
                    svd_solver="full",
                ),
            )
        )

    steps.append(
        (
            "logreg",
            LogisticRegression(
                C=LOGISTIC_C,
                max_iter=LOGISTIC_MAX_ITER,
                solver="liblinear",
                random_state=RANDOM_SEED,
            ),
        )
    )

    return Pipeline(steps=steps)


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
# For this small Directory_* cohort, one defensible exploratory default is:
#
#   StratifiedKFold(n_splits=5) ON THE PATIENT TABLE
#
# Five folds still leave only a few validation patients per fold. Fold-specific
# AUCs are therefore highly quantized and unstable; the pooled OOF result and its
# broad uncertainty interval are more informative, but external validation is
# still essential.
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
    n_bootstrap=BOOTSTRAP_REPLICATES,
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

    # Work at the patient-row level. Each element must correspond to one
    # unique Directory_* patient's OOF prediction.
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length.")

    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)

    if len(class0) == 0 or len(class1) == 0:
        raise ValueError("Bootstrap AUC requires both classes.")

    # Use a local random generator so bootstrap reproducibility does not
    # depend on unrelated NumPy random calls elsewhere in the program.
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

    # Collapse repeated slice rows to patient dictionaries before printing.
    # This prevents large series from inflating the displayed fold sample size.
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


# =============================
# FEATURE CACHE + OUTPUT HELPERS
# =============================

def feature_cache_fingerprint(samples, dataset_root):
    """Hash file metadata plus every setting that changes frozen embeddings."""

    # The fingerprint combines extraction semantics with the discovered file
    # inventory. Classifier-only settings are deliberately excluded because they
    # do not change frozen slice embeddings.
    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()

    settings = {
        "schema": FEATURE_CACHE_SCHEMA_VERSION,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "monai_input_size": MONAI_INPUT_SIZE,
        "use_monai_roi": USE_MONAI_ROI,
        "monai_bundle": MONAI_BUNDLE_NAME,
        "monai_bundle_version": MONAI_BUNDLE_VERSION,
        "monai_hf_repo_id": MONAI_HF_REPO_ID,
        "monai_hf_revision": MONAI_HF_REVISION,
        "monai_official_torchscript_sha256": (
            MONAI_OFFICIAL_TORCHSCRIPT_SHA256
        ),
        "monai_model_sha256": MONAI_MODEL_SHA256,
        "verify_monai_artifact_sha256": VERIFY_MONAI_ARTIFACT_SHA256,
        "force_rebuild_monai_torchscript": (
            FORCE_REBUILD_MONAI_TORCHSCRIPT
        ),
        "roi_dilation": MONAI_ROI_DILATION_KERNEL,
        "background_weight": MONAI_BACKGROUND_WEIGHT,
        "min_area": MONAI_MIN_HEART_AREA_RATIO,
        "max_area": MONAI_MAX_HEART_AREA_RATIO,
        "min_peak": MONAI_MIN_PEAK_HEART_PROBABILITY,
        "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
        "efficientnet_mean": EFFICIENTNET_MEAN,
        "efficientnet_std": EFFICIENTNET_STD,
        "quality_weights_enabled": USE_SLICE_QUALITY_WEIGHTS,
        "quality_min_weight": SLICE_QUALITY_MIN_WEIGHT,
        "use_cuda_amp": USE_CUDA_AMP,
        "device_type": DEVICE,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "opencv_version": cv2.__version__,
    }
    digest.update(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    )

    for image_path, label, patient_id, series_id in samples:
        path = Path(image_path)
        stat = path.stat()

        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            relative = path.resolve()

        record = (
            f"{relative.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{int(label)}|{patient_id}|{series_id}\n"
        )
        digest.update(record.encode("utf-8"))

    return digest.hexdigest()


# Centralizing cache filenames prevents load/save order drift. The returned
# dictionary is used by both functions, so one renamed array cannot be silently
# written under one name and read under another.
def _cache_array_paths(cache_dir):
    names = [
        "features",
        "labels",
        "patient_ids",
        "series_ids",
        "quality_weights",
        "monai_valid",
        "area_ratios",
        "peak_probabilities",
        "mean_foreground_probabilities",
        "slice_scores",
        "decoded_pixel_hashes",
    ]
    return {name: cache_dir / f"{name}.npy" for name in names}


def load_feature_cache(cache_dir, expected_fingerprint):
    """Load a complete cache only when its metadata and arrays all match."""

    # Metadata is the cache completion marker. A directory containing only a
    # subset of arrays is treated as invalid and extraction is rerun.
    metadata_path = cache_dir / "metadata.json"
    paths = _cache_array_paths(cache_dir)

    if not metadata_path.is_file() or not all(
        path.is_file() for path in paths.values()
    ):
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if metadata.get("fingerprint") != expected_fingerprint:
        return None

    print(f"Loading frozen-feature cache: {cache_dir}")

    return tuple(
        np.load(paths[name], allow_pickle=False)
        for name in [
            "features",
            "labels",
            "patient_ids",
            "series_ids",
            "quality_weights",
            "monai_valid",
            "area_ratios",
            "peak_probabilities",
            "mean_foreground_probabilities",
            "slice_scores",
            "decoded_pixel_hashes",
        ]
    )


def save_feature_cache(cache_dir, fingerprint, arrays):
    """Save arrays first and write metadata last as the completion marker."""

    # Create the cache directory, write every array, then write metadata last.
    # This ordering reduces the risk of accepting an interrupted partial cache.
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_array_paths(cache_dir)
    names = [
        "features",
        "labels",
        "patient_ids",
        "series_ids",
        "quality_weights",
        "monai_valid",
        "area_ratios",
        "peak_probabilities",
        "mean_foreground_probabilities",
        "slice_scores",
        "decoded_pixel_hashes",
    ]

    if len(arrays) != len(names):
        raise ValueError(
            f"Expected {len(names)} feature-cache arrays, got {len(arrays)}."
        )

    for name, array in zip(names, arrays):
        np.save(paths[name], np.asarray(array), allow_pickle=False)

    metadata = {
        "fingerprint": fingerprint,
        "schema": FEATURE_CACHE_SCHEMA_VERSION,
        "n_slices": int(len(arrays[0])),
        "feature_dimension": int(arrays[0].shape[1]),
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved frozen-feature cache: {cache_dir}")


def audit_exact_decoded_pixel_duplicates(
    output_path,
    samples,
    decoded_pixel_hashes,
):
    """Audit exact decoded-pixel copies without changing patient identity.

    The hash is calculated from the decoded uint8 grayscale matrix and its
    shape before normalization. Therefore it catches exact pixel equality even
    when JPEG metadata differs, but it does not catch perceptually similar or
    slightly re-encoded images. A duplicate that spans two Directory_* patients
    can leak image content across patient-level folds; a duplicate spanning both
    labels is an even stronger indication of dataset/export contamination.

    The function only reports or optionally aborts. It never merges patients or
    changes ``patient_id = Directory_*``.
    """

    # Row alignment is essential: every decoded hash must refer to the same
    # image metadata row at the corresponding ``samples`` index.
    if len(samples) != len(decoded_pixel_hashes):
        raise ValueError(
            "Duplicate-audit hashes must align one-to-one with samples."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group sample indices by exact decoded-pixel digest. This operation does
    # not alter samples, patient IDs, folds, or feature arrays.
    hash_to_indices = {}
    for index, pixel_hash in enumerate(decoded_pixel_hashes):
        hash_to_indices.setdefault(str(pixel_hash), []).append(index)

    duplicate_groups = [
        (pixel_hash, indices)
        for pixel_hash, indices in sorted(hash_to_indices.items())
        if len(indices) > 1
    ]

    fieldnames = [
        "group_id",
        "decoded_pixel_sha256",
        "n_images",
        "n_patients",
        "n_labels",
        "cross_patient",
        "cross_label",
        "image_path",
        "label",
        "patient_id",
        "series_id",
    ]

    # Build one CSV row per image in every duplicate group and aggregate
    # high-level counts for warnings and run_summary.json.
    rows = []
    cross_patient_groups = 0
    cross_label_groups = 0
    cross_patient_images = 0
    affected_cross_patient_ids = set()

    for group_number, (pixel_hash, indices) in enumerate(
        duplicate_groups,
        start=1,
    ):
        patients = {str(samples[index][2]) for index in indices}
        labels = {int(samples[index][1]) for index in indices}
        cross_patient = len(patients) > 1
        cross_label = len(labels) > 1

        if cross_patient:
            cross_patient_groups += 1
            cross_patient_images += len(indices)
            affected_cross_patient_ids.update(patients)
        if cross_label:
            cross_label_groups += 1

        for index in indices:
            image_path, label, patient_id, series_id = samples[index]
            rows.append(
                {
                    "group_id": group_number,
                    "decoded_pixel_sha256": pixel_hash,
                    "n_images": len(indices),
                    "n_patients": len(patients),
                    "n_labels": len(labels),
                    "cross_patient": int(cross_patient),
                    "cross_label": int(cross_label),
                    "image_path": str(image_path),
                    "label": int(label),
                    "patient_id": str(patient_id),
                    "series_id": str(series_id),
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "enabled": True,
        "definition": "exact decoded uint8 grayscale pixels plus shape",
        "duplicate_groups": int(len(duplicate_groups)),
        "duplicate_images": int(sum(len(indices) for _, indices in duplicate_groups)),
        "cross_patient_duplicate_groups": int(cross_patient_groups),
        "cross_patient_duplicate_images": int(cross_patient_images),
        "cross_label_duplicate_groups": int(cross_label_groups),
        "affected_cross_patient_ids": sorted(affected_cross_patient_ids),
        "csv_path": str(output_path),
    }

    print(
        "Exact decoded-pixel duplicate audit: "
        f"groups={summary['duplicate_groups']}, "
        f"cross-patient groups={cross_patient_groups}, "
        f"cross-label groups={cross_label_groups}"
    )

    if cross_patient_groups:
        print(
            "WARNING: exact decoded-pixel copies occur across Directory_* "
            "patients. A patient-only split does not by itself prevent that "
            "visual-content leakage. Inspect the duplicate CSV before "
            "reporting publication results."
        )

    if cross_label_groups:
        print(
            "WARNING: at least one exact decoded-pixel duplicate group spans "
            "both Normal and Sick labels. This requires dataset-level review."
        )

    if cross_patient_groups and FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES:
        raise RuntimeError(
            "Cross-patient exact decoded-pixel duplicates were found. "
            f"See {output_path}. The strict duplicate policy is enabled."
        )

    return summary


def write_monai_qc_summary(
    output_path,
    labels,
    patient_ids,
    series_ids,
    monai_valid,
    area_ratios,
    peak_probabilities,
    mean_foreground_probabilities,
):
    """Save patient-level MONAI gate diagnostics without claiming Dice accuracy."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate slice-level MONAI gate signals to one descriptive QC row per
    # patient. Median statistics reduce sensitivity to one extreme slice.
    rows = []
    for patient_id in sorted(set(map(str, patient_ids))):
        indices = np.flatnonzero(patient_ids == patient_id)
        label_values = np.unique(labels[indices])

        if len(label_values) != 1:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent QC labels."
            )

        valid_values = monai_valid[indices]
        rows.append(
            {
                "patient_id": patient_id,
                "label": int(label_values[0]),
                "n_slices": int(len(indices)),
                "n_series_proxies": int(len(np.unique(series_ids[indices]))),
                "plausible_masks": int(valid_values.sum()),
                "plausible_mask_rate": (
                    float(valid_values.mean()) if USE_MONAI_ROI else ""
                ),
                "median_area_ratio": (
                    float(np.nanmedian(area_ratios[indices]))
                    if USE_MONAI_ROI
                    else ""
                ),
                "median_peak_probability": (
                    float(np.nanmedian(peak_probabilities[indices]))
                    if USE_MONAI_ROI
                    else ""
                ),
                "median_mean_foreground_probability": (
                    float(
                        np.nanmedian(
                            mean_foreground_probabilities[indices]
                        )
                    )
                    if USE_MONAI_ROI
                    else ""
                ),
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if USE_MONAI_ROI:
        for label, class_name in ((0, "Normal"), (1, "Sick")):
            indices = np.flatnonzero(labels == label)
            rate = float(monai_valid[indices].mean())
            print(
                f"MONAI plausible-mask rate in {class_name}: "
                f"{int(monai_valid[indices].sum())}/{len(indices)} "
                f"({rate:.2%})"
            )


def run_patient_level_cross_validation(
    X_all,
    y_all,
    patient_all,
    series_all,
    quality_weights_all,
):
    """Return one out-of-fold score per Directory_* patient."""

    # Evaluation stage 1: construct the authoritative one-row-per-patient
    # label table from slice metadata and reject any conflicting labels.
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
        raise RuntimeError("The dataset must contain both classes.")

    class_patient_counts = np.bincount(all_patient_labels, minlength=2)
    if int(class_patient_counts.min()) < N_SPLITS:
        raise RuntimeError(
            f"{N_SPLITS}-fold stratified CV requires at least {N_SPLITS} "
            "Directory_* patients in each class. Found "
            f"Normal={int(class_patient_counts[0])}, "
            f"Sick={int(class_patient_counts[1])}."
        )

    print("\nFinal patient-level dataset")
    print(f"  Patients: {len(all_patient_ids)}")
    print(f"  Normal: {int(class_patient_counts[0])}")
    print(f"  Sick: {int(class_patient_counts[1])}")
    print(f"  Slice embeddings: {len(X_all)}")
    print(f"  Classification strategy: {CLASSIFICATION_STRATEGY}")

    if len(all_patient_ids) < 100:
        print(
            "WARNING: fewer than 100 Directory_* patient units were found. "
            "The effective labeled sample size is the patient count, not the "
            "slice count; expect wide uncertainty and split sensitivity."
        )

    # Evaluation stage 2: precompute deterministic patient embeddings for the
    # recommended branch. Pooling is label-free and uses all slices of a patient,
    # but each patient is later assigned wholly to one fold.
    if CLASSIFICATION_STRATEGY == "patient_embedding":
        (
            patient_embeddings,
            embedding_labels,
            embedding_patient_ids,
        ) = aggregate_patient_embeddings(
            X_all,
            y_all,
            patient_all,
            series_all,
            quality_weights_all,
        )

        if not np.array_equal(embedding_patient_ids, all_patient_ids):
            raise RuntimeError(
                "Pooled patient embedding order does not match patient table."
            )
        if not np.array_equal(embedding_labels, all_patient_labels):
            raise RuntimeError(
                "Pooled patient labels do not match patient table."
            )
    else:
        patient_embeddings = None

    # Evaluation stage 3: define patient-level folds. Stratification preserves
    # class proportions as closely as possible; no slice enters the splitter.
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    oof_score_by_patient = {}
    oof_label_by_patient = {}
    oof_fold_by_patient = {}
    fold_aucs = []

    # Evaluation stage 4: fit a completely fresh scaler/PCA/classifier in
    # every fold and generate scores only for that fold's held-out patients.
    for fold_index, (train_idx, valid_idx) in enumerate(
        cv.split(all_patient_ids, all_patient_labels),
        start=1,
    ):
        # Convert index arrays to explicit patient sets and verify disjointness
        # before selecting any patient embeddings or slice rows.
        train_patients = set(all_patient_ids[train_idx].tolist())
        valid_patients = set(all_patient_ids[valid_idx].tolist())

        overlap = train_patients.intersection(valid_patients)
        if overlap:
            raise RuntimeError(
                f"Patient leakage in fold {fold_index}: {sorted(overlap)}"
            )

        # Branch A (recommended): one training and one validation row per
        # patient. Branch B (legacy): select all slices by patient membership,
        # fit with hierarchical weights, then fuse validation slice scores.
        if CLASSIFICATION_STRATEGY == "patient_embedding":
            X_train = patient_embeddings[train_idx]
            y_train = all_patient_labels[train_idx]
            X_valid = patient_embeddings[valid_idx]
            y_valid_patient = all_patient_labels[valid_idx]
            evaluated_patient_ids = all_patient_ids[valid_idx]

            print(f"\n========== FOLD {fold_index} ==========")
            print(
                f"Train patients: {len(train_idx)} "
                f"(Normal={int(np.sum(y_train == 0))}, "
                f"Sick={int(np.sum(y_train == 1))})"
            )
            print(
                f"Validation patients: {len(valid_idx)} "
                f"(Normal={int(np.sum(y_valid_patient == 0))}, "
                f"Sick={int(np.sum(y_valid_patient == 1))})"
            )

            training_weights = compute_balanced_patient_weights(y_train)
            classifier = build_classifier()
            classifier.fit(
                X_train,
                y_train,
                # One row already equals one patient, so the unsupervised
                # scaler gives every training patient equal influence. Class
                # balancing is applied only to the supervised classifier. PCA
                # is likewise unweighted because sklearn PCA has no sample-
                # weight argument.
                scaler__sample_weight=np.ones(len(y_train), dtype=np.float64),
                logreg__sample_weight=training_weights,
            )
            patient_scores = classifier.predict_proba(X_valid)[:, 1]
            patient_ground_truth = y_valid_patient

        else:
            train_slice_mask = np.isin(patient_all, list(train_patients))
            valid_slice_mask = np.isin(patient_all, list(valid_patients))

            if np.any(train_slice_mask & valid_slice_mask):
                raise RuntimeError("Slice masks overlap across CV partitions.")
            if not np.all(train_slice_mask | valid_slice_mask):
                raise RuntimeError("Some slices were not assigned to a fold side.")

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

            training_weights = compute_hierarchical_training_weights(
                y_train,
                patient_train,
                series_train,
                quality_weights=quality_train,
            )
            classifier = build_classifier()
            classifier.fit(
                X_train,
                y_train,
                scaler__sample_weight=training_weights,
                logreg__sample_weight=training_weights,
            )

            (
                patient_scores,
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

            if set(map(str, evaluated_patient_ids)) != valid_patients:
                raise RuntimeError(
                    f"Fold {fold_index}: validation patient set mismatch."
                )

        # Register exactly one OOF score per evaluated patient. Duplicate
        # insertion is treated as an implementation error and stops the run.
        for patient_id, label, score in zip(
            evaluated_patient_ids,
            patient_ground_truth,
            patient_scores,
        ):
            patient_id = str(patient_id)
            if patient_id in oof_score_by_patient:
                raise RuntimeError(
                    f"Patient {patient_id} received multiple OOF scores."
                )
            oof_score_by_patient[patient_id] = float(score)
            oof_label_by_patient[patient_id] = int(label)
            oof_fold_by_patient[patient_id] = int(fold_index)

        fold_auc = roc_auc_score(patient_ground_truth, patient_scores)
        fold_aucs.append(float(fold_auc))
        print(f"Fold {fold_index} patient-level AUC: {fold_auc:.4f}")

    if set(oof_score_by_patient) != set(all_patient_ids.tolist()):
        missing = sorted(
            set(all_patient_ids.tolist()) - set(oof_score_by_patient)
        )
        raise RuntimeError(
            f"Patients missing OOF predictions: {missing}"
        )

    # Evaluation stage 5: rebuild the complete OOF patient table in stable
    # sorted order after proving that every discovered patient appears once.
    evaluated_patient_ids = np.asarray(sorted(oof_score_by_patient))
    patient_scores = np.asarray(
        [oof_score_by_patient[p] for p in evaluated_patient_ids],
        dtype=np.float64,
    )
    patient_ground_truth = np.asarray(
        [oof_label_by_patient[p] for p in evaluated_patient_ids],
        dtype=np.int64,
    )
    patient_folds = np.asarray(
        [oof_fold_by_patient[p] for p in evaluated_patient_ids],
        dtype=np.int64,
    )

    return (
        evaluated_patient_ids,
        patient_ground_truth,
        patient_scores,
        patient_folds,
        fold_aucs,
    )


def write_oof_predictions(
    output_path,
    patient_ids,
    labels,
    scores,
    folds,
):
    """Write one transparent out-of-fold result row per Directory_* patient."""

    # Create a machine-readable table whose unit is explicitly one patient.
    # The score column is named uncalibrated to prevent clinical overinterpretation.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "patient_id",
                "true_label",
                "oof_fold",
                "uncalibrated_cad_score",
            ]
        )
        for row in zip(patient_ids, labels, folds, scores):
            writer.writerow(
                [str(row[0]), int(row[1]), int(row[2]), float(row[3])]
            )


def collect_run_metadata():
    """Capture configuration, software versions and available model hashes."""

    # Record both configured expectations and artifacts actually present at
    # the end of the run. This supports later provenance and cache investigations.
    bundle_root = locate_monai_bundle_root()
    official_torchscript_path = bundle_root / "models" / "model.ts"
    checkpoint_path = bundle_root / "models" / "model.pt"

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "opencv": cv2.__version__,
        "device": DEVICE,
        "classification_strategy": CLASSIFICATION_STRATEGY,
        "run_name": RUN_NAME,
        "run_configuration_tag": RUN_CONFIGURATION_TAG,
        "output_directory": str(OUTPUT_DIR),
        "patient_definition": "Directory_*",
        "series_definition": "immediate child folder proxy; not DICOM UID",
        "n_splits": N_SPLITS,
        "cv_random_state": CV_RANDOM_STATE,
        "logistic_c": LOGISTIC_C,
        "use_patient_pca": USE_PATIENT_PCA,
        "patient_pca_explained_variance": PATIENT_PCA_EXPLAINED_VARIANCE,
        "use_monai_roi": USE_MONAI_ROI,
        "monai_bundle_name": MONAI_BUNDLE_NAME,
        "monai_bundle_version": MONAI_BUNDLE_VERSION,
        "monai_hf_repo_id": MONAI_HF_REPO_ID,
        "monai_hf_revision": MONAI_HF_REVISION,
        "verify_monai_artifact_sha256": VERIFY_MONAI_ARTIFACT_SHA256,
        "monai_expected_model_ts_sha256": (
            MONAI_OFFICIAL_TORCHSCRIPT_SHA256
        ),
        "monai_expected_model_pt_sha256": MONAI_MODEL_SHA256,
        "monai_runtime_source": MONAI_RUNTIME_SOURCE,
        "monai_runtime_artifact_path": MONAI_RUNTIME_ARTIFACT_PATH,
        "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
        "debug_visualization": DEBUG_VISUALIZATION,
        "debug_indices": DEBUG_INDICES,
        "debug_output_directory": str(DEBUG_OUTPUT_DIR),
        "use_slice_quality_weights": USE_SLICE_QUALITY_WEIGHTS,
        "audit_exact_decoded_pixel_duplicates": (
            AUDIT_EXACT_DECODED_PIXEL_DUPLICATES
        ),
        "fail_on_cross_patient_exact_duplicates": (
            FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES
        ),
        "feature_cache_schema": FEATURE_CACHE_SCHEMA_VERSION,
    }

    if official_torchscript_path.is_file():
        metadata["monai_actual_model_ts_sha256"] = sha256_file(
            official_torchscript_path
        )
    else:
        metadata["monai_actual_model_ts_sha256"] = None

    if checkpoint_path.is_file():
        metadata["monai_actual_model_pt_sha256"] = sha256_file(
            checkpoint_path
        )
    else:
        metadata["monai_actual_model_pt_sha256"] = None

    if MONAI_TORCHSCRIPT_PATH.is_file():
        metadata["monai_fallback_torchscript_sha256"] = sha256_file(
            MONAI_TORCHSCRIPT_PATH
        )
    else:
        metadata["monai_fallback_torchscript_sha256"] = None

    return metadata


def main():
    """Run the complete pipeline with hard-coded configuration above."""

    # ======================================================================
    # MAIN STAGE 1 -- VALIDATE THE COMPLETE RUN CONFIGURATION
    # ======================================================================
    # This call checks strategy names, tensor-size assumptions, ROI parameters,
    # checkpoint identifiers, duplicate-audit dependencies, CV settings, and
    # classifier hyperparameters. It must happen before any expensive I/O,
    # download, model initialization, GPU work, or output generation.
    #
    # Input:
    #     module-level constants defined in CONFIGURATION.
    # Output:
    #     no returned value; success means the configuration is internally
    #     coherent, while any invalid setting raises an explicit exception.
    validate_configuration()

    # ======================================================================
    # MAIN STAGE 2 -- CREATE THE CONFIGURATION-SPECIFIC OUTPUT DIRECTORY
    # ======================================================================
    # RUN_NAME contains a deterministic hash of analysis settings, so results
    # from different ablations are written to separate directories. ``parents``
    # creates missing upper directories and ``exist_ok`` allows a repeated run
    # to reuse the same destination without deleting existing files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # MAIN STAGE 3 -- DISCOVER IMAGES AND BUILD THE SLICE METADATA TABLE
    # ======================================================================
    # ``load_samples`` walks Normal/ and Sick/, accepts Directory_* as the
    # validated patient unit, assigns child folders as patient-scoped series
    # proxies, and returns one metadata tuple per image. It also rejects missing
    # class directories, empty datasets, and patient/label inconsistencies.
    #
    # Output row:
    #     (image_path, binary_label, patient_id, series_id)
    #
    # No JPEG is decoded and no neural network is loaded in this stage.
    samples = load_samples(DATASET_PATH)

    # ======================================================================
    # MAIN STAGE 4 -- COMPUTE THE FROZEN-FEATURE CACHE IDENTITY
    # ======================================================================
    # The fingerprint combines:
    #   - every discovered file's relative path, size, modification timestamp,
    #     label, patient ID, and series-proxy ID;
    #   - all settings and software versions that can change frozen embeddings.
    #
    # Consequently, a changed dataset or extraction setting points to a new
    # cache rather than silently reusing incompatible features. The first 16
    # hexadecimal characters provide a compact directory name, while the full
    # digest remains stored in cache/run metadata.
    fingerprint = feature_cache_fingerprint(samples, DATASET_PATH)
    cache_dir = FEATURE_CACHE_ROOT / fingerprint[:16]

    # ======================================================================
    # MAIN STAGE 5 -- TRY TO REUSE A COMPLETE, MATCHING FEATURE CACHE
    # ======================================================================
    # Feature extraction is the expensive stage because every slice may pass
    # through MONAI and EfficientNet. Reusing it is safe only when:
    #   1. caching is enabled;
    #   2. a forced rebuild was not requested;
    #   3. metadata contains the exact expected full fingerprint;
    #   4. every required NumPy array is present.
    #
    # ``extracted`` remains None when any condition fails, which routes execution
    # to fresh model inference below.
    extracted = None
    if USE_FEATURE_CACHE and not FORCE_REBUILD_FEATURE_CACHE:
        extracted = load_feature_cache(cache_dir, fingerprint)

    # ======================================================================
    # MAIN STAGE 6 -- BUILD DATASET/MODELS AND EXTRACT FROZEN SLICE FEATURES
    # ======================================================================
    # This block executes only when a valid feature cache was not loaded.
    if extracted is None:

        # 6A. Create the lazy PyTorch Dataset.
        #     The object stores paths/metadata; individual JPEGs are decoded and
        #     converted to aligned 256×256 and 224×224 tensors on demand.
        all_dataset = MRIDataset(samples, transform)

        # 6B. Load the pinned MONAI segmenter only when ROI mode is enabled.
        #     Normal execution uses the verified official model.ts directly.
        #     Setting USE_MONAI_ROI=False deliberately skips all MONAI loading
        #     and creates the required full-image ablation.
        monai_segmenter = (
            build_monai_segmenter() if USE_MONAI_ROI else None
        )

        # 6C. Initialize the explicitly pinned ImageNet EfficientNet-B0 encoder.
        #     The final ImageNet classifier is removed by FeatureExtractor.
        #     eval() disables training behavior, and requires_grad_(False)
        #     prevents accidental gradient computation or parameter updates.
        feature_extractor = FeatureExtractor().to(DEVICE)
        feature_extractor.eval()
        feature_extractor.requires_grad_(False)

        # 6D. Process every decoded slice in deterministic DataLoader order:
        #       JPEG decode and exact-pixel hash
        #         -> per-image [0,1] scaling
        #         -> centered 256×256 MONAI canvas
        #         -> optional MONAI probability map and plausibility gate
        #         -> soft ROI or unchanged full-image fallback
        #         -> ImageNet normalization
        #         -> frozen 1280-D EfficientNet embedding
        #         -> QC values and optional within-series quality weight.
        #
        # Labels travel only as metadata and are not inputs to either network.
        extracted = extract_features(
            all_dataset,
            monai_segmenter=monai_segmenter,
            feature_extractor=feature_extractor,
            debug=DEBUG_VISUALIZATION,
        )

        # 6E. Persist the aligned extraction arrays for future runs. Metadata is
        #     written last as a completion marker, so a partially interrupted
        #     cache is not accepted by ``load_feature_cache``.
        if USE_FEATURE_CACHE:
            save_feature_cache(cache_dir, fingerprint, extracted)

    # ======================================================================
    # MAIN STAGE 7 -- UNPACK THE ALIGNED SLICE-LEVEL EXTRACTION ARRAYS
    # ======================================================================
    # Every array below has exactly one row/value per successfully decoded image:
    #
    #   X_all                              [N,1280] frozen embeddings
    #   y_all                              [N] patient labels repeated as metadata
    #   patient_all                        [N] Directory_* identifiers
    #   series_all                         [N] folder-defined series proxies
    #   quality_weights_all                [N] equal/default or heuristic weights
    #   monai_valid_all                    [N] ROI-gate Boolean decisions
    #   area_ratios_all                    [N] hard-mask area ratios
    #   peak_probabilities_all             [N] peak P(heart)
    #   mean_foreground_probabilities_all  [N] descriptive foreground confidence
    #   slice_scores_all                   [N] ROI/full-image intensity std
    #   decoded_pixel_hashes_all           [N] exact decoded-pixel hashes
    #
    # Their one-to-one alignment is checked inside ``extract_features`` or is
    # inherited from a cache that was produced by the same function.
    (
        X_all,
        y_all,
        patient_all,
        series_all,
        quality_weights_all,
        monai_valid_all,
        area_ratios_all,
        peak_probabilities_all,
        mean_foreground_probabilities_all,
        slice_scores_all,
        decoded_pixel_hashes_all,
    ) = extracted

    # The raw heuristic score is no longer needed in the current execution path.
    # It remains saved in the cache so a later predeclared quality-weighting
    # ablation can be reproduced without rerunning the frozen neural networks.
    del slice_scores_all  # retained in cache for reproducible quality ablations

    # ======================================================================
    # MAIN STAGE 8 -- AUDIT EXACT DECODED-PIXEL DUPLICATES
    # ======================================================================
    # The audit groups images whose decoded uint8 grayscale matrix and shape are
    # exactly identical. It reports within-patient, cross-patient, and cross-label
    # groups without changing patient identity or deleting images.
    #
    # Cross-patient equality matters because patient-level splitting alone cannot
    # prevent the same visual content from appearing in training and validation.
    if AUDIT_EXACT_DECODED_PIXEL_DUPLICATES:
        duplicate_audit_summary = audit_exact_decoded_pixel_duplicates(
            OUTPUT_DIR / "exact_decoded_pixel_duplicate_groups.csv",
            samples,
            decoded_pixel_hashes_all,
        )
    else:
        # Preserve a stable summary schema even when the optional audit is off.
        # Explicit None values distinguish "not measured" from a measured zero.
        duplicate_audit_summary = {
            "enabled": False,
            "duplicate_groups": None,
            "cross_patient_duplicate_groups": None,
            "cross_label_duplicate_groups": None,
        }

    # ======================================================================
    # MAIN STAGE 9 -- WRITE PATIENT-LEVEL MONAI QUALITY-CONTROL SUMMARIES
    # ======================================================================
    # Slice-level gate values are aggregated by Directory_* patient into a CSV
    # containing slice count, series-proxy count, plausible-mask count/rate, and
    # median diagnostics. These are QC descriptors only; without ground-truth
    # masks they must not be interpreted as Dice or segmentation accuracy.
    write_monai_qc_summary(
        OUTPUT_DIR / "monai_qc_by_patient.csv",
        y_all,
        patient_all,
        series_all,
        monai_valid_all,
        area_ratios_all,
        peak_probabilities_all,
        mean_foreground_probabilities_all,
    )

    # ======================================================================
    # MAIN STAGE 10 -- RUN DIRECTORY_*-LEVEL STRATIFIED CROSS-VALIDATION
    # ======================================================================
    # The splitter receives a table with one row per patient. All slices and all
    # series proxies of a patient inherit the same fold. For each fold, a fresh
    # scaler, optional PCA, and Logistic Regression are fitted on training
    # patients only. The function returns exactly one out-of-fold score per
    # Directory_* patient plus the fold-specific AUC values.
    (
        evaluated_patient_ids,
        patient_ground_truth,
        patient_scores,
        patient_folds,
        fold_aucs,
    ) = run_patient_level_cross_validation(
        X_all,
        y_all,
        patient_all,
        series_all,
        quality_weights_all,
    )

    # ======================================================================
    # MAIN STAGE 11 -- CALCULATE THE POOLED OOF AUC AND PATIENT BOOTSTRAP CI
    # ======================================================================
    # The primary ranking metric is computed from the complete set of held-out
    # patient predictions, not from training predictions or individual slices.
    auc = roc_auc_score(patient_ground_truth, patient_scores)

    # The bootstrap resamples Normal and Sick patient rows separately with
    # replacement, guaranteeing that each replicate contains both classes. The
    # interval quantifies sampling variability of these OOF rows but does not
    # replace independent external validation.
    auc_ci_lower, auc_ci_upper = bootstrap_patient_auc_ci(
        patient_ground_truth,
        patient_scores,
    )

    # ======================================================================
    # MAIN STAGE 12 -- SAVE ONE TRANSPARENT OOF ROW PER PATIENT
    # ======================================================================
    # This CSV is the auditable basis for the pooled AUC. It records patient ID,
    # true label, held-out fold, and the uncalibrated positive-class model score.
    write_oof_predictions(
        OUTPUT_DIR / "patient_oof_predictions.csv",
        evaluated_patient_ids,
        patient_ground_truth,
        patient_scores,
        patient_folds,
    )

    # ======================================================================
    # MAIN STAGE 13 -- ASSEMBLE AND SAVE COMPLETE RUN METADATA
    # ======================================================================
    # ``collect_run_metadata`` records software versions, device, strategy,
    # model provenance, configured hashes, artifact hashes when present, cache
    # schema, and output locations. Evaluation results and duplicate-audit counts
    # are then added to the same JSON object.
    summary = collect_run_metadata()
    summary.update(
        {
            "patients_evaluated": int(len(evaluated_patient_ids)),
            "normal_patients": int(np.sum(patient_ground_truth == 0)),
            "sick_patients": int(np.sum(patient_ground_truth == 1)),
            "patient_oof_auc": float(auc),
            "patient_oof_auc_ci_95": [
                float(auc_ci_lower),
                float(auc_ci_upper),
            ],
            "fold_aucs": [float(value) for value in fold_aucs],
            "feature_cache_fingerprint": fingerprint,
            "exact_duplicate_audit": duplicate_audit_summary,
        }
    )

    # JSON is human-readable, machine-readable, UTF-8 encoded, and key-sorted to
    # make differences between runs easier to inspect in version control.
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # ======================================================================
    # MAIN STAGE 14 -- PRINT THE PRIMARY PATIENT-LEVEL RESULT
    # ======================================================================
    # The console summary intentionally repeats the effective patient count and
    # explicitly warns that the scores are not externally calibrated clinical
    # probabilities. The saved CSV/JSON files remain the authoritative records.
    print("\n============================================================")
    print("FINAL OUT-OF-FOLD PATIENT-LEVEL RESULT")
    print("============================================================")
    print(f"Patients evaluated: {len(evaluated_patient_ids)}")
    print(f"Patient-level OOF AUC: {auc:.6f}")
    print(
        "Patient-level bootstrap 95% CI: "
        f"[{auc_ci_lower:.6f}, {auc_ci_upper:.6f}]"
    )
    print(
        "Scores are model outputs from Logistic Regression and are not "
        "claimed to be externally calibrated clinical probabilities."
    )

    # ======================================================================
    # MAIN STAGE 15 -- PRINT EVERY PATIENT'S AUDITABLE OOF SCORE
    # ======================================================================
    # Iterating over aligned arrays makes it possible to compare console output
    # directly with patient_oof_predictions.csv. Each patient appears once.
    print("\nPATIENT-LEVEL OOF SCORES:")
    for patient_id, label, fold, score in zip(
        evaluated_patient_ids,
        patient_ground_truth,
        patient_folds,
        patient_scores,
    ):
        print(
            f"  {patient_id}: true_label={int(label)}, "
            f"fold={int(fold)}, uncalibrated_CAD_score={float(score):.6f}"
        )

    # ======================================================================
    # MAIN STAGE 16 -- REPORT THE OUTPUT LOCATION AND NORMAL COMPLETION
    # ======================================================================
    # Reaching this point means validation, extraction/cache loading, audits,
    # cross-validation, metrics, and all principal output writes completed
    # without raising an exception.
    print(f"\nOutputs saved under: {OUTPUT_DIR}")
    print("Done!")


# The execution guard is essential on platforms that use process spawning
# (notably Windows). It prevents DataLoader worker processes from recursively
# rerunning the full script when this file is imported. It also permits the
# functions/classes to be imported for tests without automatically starting the
# expensive experiment.
if __name__ == "__main__":
    main()

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
# 4. The recommended default already uses hierarchical embedding pooling.
#    Report the legacy slice-classifier/fusion strategy only as an ablation and
#    discuss its repeated-label/pseudo-replication limitation explicitly.
# 5. Compare Logistic Regression vs linear SVM.
# 6. Compare frozen ImageNet EfficientNet-B0 with a cardiac-MRI-pretrained
#    encoder if an appropriate public checkpoint is available.
# 7. Evaluate each sequence/view separately only if sequence/view identity can
#    be recovered reliably from the released files.
# 8. Audit exact/near-duplicate images ACROSS Directory_* patients before any
#    final publication split; cross-patient duplicates can leak visual content
#    even when patient IDs themselves never cross folds.
# 9. Tune ROI-gate thresholds, classifier C, calibration or decision thresholds
#    only inside training data (nested patient-level CV if tuned quantitatively).
# 10. Compare the class-specific MONAI gate rates. A large Normal/Sick difference
#     may indicate protocol/export confounding rather than anatomical usefulness.
# 11. Report sensitivity, specificity, PPV, NPV and confidence intervals using
#     a threshold selected without looking at the validation fold.
# 12. Perform external validation on an independent hospital dataset if possible.
# 13. Audit image dimensions, borders, compression and acquisition/export style
#     by class; the model must not be allowed to classify dataset provenance.
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
