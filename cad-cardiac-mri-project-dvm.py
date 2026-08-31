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


# ============================================================================
# MULTI-EXPERIMENT EXTENSION
# ============================================================================
#
# The implementation below extends the original patient-level pipeline into a
# controlled experiment suite without changing the validated patient definition
# or the frozen MONAI/EfficientNet preprocessing contract.
#
# The suite evaluates the primary pipeline together with declared ablations and
# negative controls using the SAME duplicate-aware patient folds. This makes the
# comparison scientifically interpretable rather than a collection of unrelated
# train/test runs.
#
# Default experiments:
#
#   B0  MONAI ROI + hierarchical embedding pooling + Logistic Regression + PCA
#   A1  Full image instead of MONAI ROI
#   C1  Border-only negative control
#   C2  Outside-MONAI-mask negative control
#   C3  Export/provenance metadata-only classifier
#   C4  MONAI-QC-only classifier
#   A2  Flat slice-to-patient embedding pooling
#   A3  Legacy slice classifier + log-odds fusion
#   A4  Legacy slice classifier + mean-probability fusion
#   A5  Linear SVM instead of Logistic Regression
#   A6  Optional slice-quality heuristic instead of equal slice weights
#   A7  PCA disabled
#
# The expensive frozen image processing is shared through one multi-view feature
# bank. Full-image and MONAI-ROI embeddings are extracted in the same pass, while
# classifier, PCA, weighting and fusion choices are evaluated only inside the
# fold-local supervised stage. Hyperparameter C and decision thresholds are
# selected only from inner out-of-fold predictions of the outer training cohort.
#
# Negative controls are intentionally retained: if border/provenance/MONAI-QC
# variables predict the label strongly, a high CAD AUC cannot safely be
# interpreted as purely anatomical signal.
#
# ============================================================================

# IMPORTS
# =============================

import csv
# Standard-library CSV writer used for reproducible machine-readable outputs.

import shutil
# Removes incomplete feature-bank directories before a deliberate rebuild.

import sys
# Redirects ordinary print() and tqdm output to both console and log file.

import traceback
# Saves visible stack traces when one experiment fails while the suite continues.

from collections import defaultdict
# Efficient grouping for duplicate candidates and patient/series aggregation.

from dataclasses import asdict, dataclass
# Immutable experiment configurations and reproducible JSON serialization.

from itertools import combinations
# Creates patient-pair edges inside cross-patient duplicate groups.


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

import time
# High-resolution wall-clock timing used by the console progress messages.
# ``time.perf_counter`` is monotonic and is appropriate for measuring stage,
# model-loading, cache, fold, and end-to-end execution durations.

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

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
# Patient-level discrimination, calibration and threshold metrics.

from sklearn.model_selection import StratifiedKFold
# Standard patient-level splitter used when every duplicate component contains
# exactly one patient. StratifiedGroupKFold is imported lazily when confirmed
# cross-patient duplicate components must remain together.

from sklearn.pipeline import Pipeline
# Bundles fold-local standardization, optional PCA, and the selected linear
# classifier so no validation-patient feature is used to fit preprocessing.

from sklearn.preprocessing import StandardScaler
# Standardizes embeddings or tabular controls inside each fold.

from sklearn.svm import LinearSVC
# Linear maximum-margin classifier used in the declared SVM ablation.

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
# MULTI-EXPERIMENT SUITE CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable description of one auditable experiment in the suite."""

    experiment_id: str
    description: str
    feature_mode: str
    strategy: str
    pooling_strategy: str = "hierarchical"
    weighting_mode: str = "equal"
    classifier_type: str = "logistic_regression"
    fusion_method: str | None = None
    use_pca: bool = True
    tune_c: bool = True
    fixed_c: float = 1.0
    role: str = "ablation"
    enabled: bool = True


# Every enabled experiment is launched automatically by main(). To run a
# smaller preliminary suite, set EXPERIMENTS_TO_RUN to a tuple of IDs. Keep it
# as None to run every configuration whose enabled field is True.
EXPERIMENTS_TO_RUN = None

BASELINE_EXPERIMENT_ID = "B0_ROI_HIER_LR_PCA"

EXPERIMENT_REGISTRY = (
    ExperimentConfig(
        experiment_id="B0_ROI_HIER_LR_PCA",
        description=(
            "Recommended baseline: confidence-gated MONAI ROI, equal slice "
            "weights, hierarchical embedding pooling, PCA and Logistic Regression."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="baseline",
    ),
    ExperimentConfig(
        experiment_id="A1_FULL_HIER_LR_PCA",
        description="Full-image ablation with all downstream settings unchanged.",
        feature_mode="full_image",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
    ),
    ExperimentConfig(
        experiment_id="C1_BORDER_ONLY_HIER_LR_PCA",
        description=(
            "Negative control retaining only the outer image border; a high AUC "
            "would indicate export/style shortcut risk."
        ),
        feature_mode="border_only",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA",
        description=(
            "Negative control retaining signal outside the dilated MONAI soft "
            "mask. The mask is applied unconditionally and remains a proxy, not "
            "validated anatomy."
        ),
        feature_mode="outside_heart",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C3_EXPORT_PROVENANCE_ONLY_LR",
        description=(
            "Patient classifier using only image dimensions, border/intensity "
            "statistics, sharpness and file-size/compression proxies."
        ),
        feature_mode="provenance_only",
        strategy="patient_tabular",
        pooling_strategy="patient_tabular",
        weighting_mode="not_applicable",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C4_MONAI_QC_ONLY_LR",
        description=(
            "Patient classifier using only MONAI gate/QC statistics. A high AUC "
            "would suggest protocol or sequence confounding."
        ),
        feature_mode="monai_qc_only",
        strategy="patient_tabular",
        pooling_strategy="patient_tabular",
        weighting_mode="not_applicable",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="A2_ROI_FLAT_LR_PCA",
        description=(
            "Flat slice-to-patient embedding mean; long series can dominate "
            "because the series hierarchy is intentionally removed."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="flat",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
    ),
    ExperimentConfig(
        experiment_id="A3_ROI_LEGACY_LOGODDS_LR",
        description=(
            "Legacy weakly supervised slice classifier with hierarchical "
            "log-odds fusion; retained only as an ablation."
        ),
        feature_mode="monai_roi",
        strategy="slice_probability_fusion",
        pooling_strategy="probability_fusion",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        fusion_method="log_odds",
        use_pca=False,
        tune_c=False,
        fixed_c=1.0,
    ),
    ExperimentConfig(
        experiment_id="A4_ROI_LEGACY_MEANPROB_LR",
        description=(
            "Same legacy slice classifier as A3, but mean-probability fusion "
            "replaces log-odds fusion."
        ),
        feature_mode="monai_roi",
        strategy="slice_probability_fusion",
        pooling_strategy="probability_fusion",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        fusion_method="mean_probability",
        use_pca=False,
        tune_c=False,
        fixed_c=1.0,
    ),
    ExperimentConfig(
        experiment_id="A5_ROI_HIER_LINEAR_SVM_PCA",
        description=(
            "Linear SVM ablation. Fold-local sigmoid calibration is learned only "
            "from inner out-of-fold training scores."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="linear_svm",
        use_pca=True,
        tune_c=True,
    ),
    ExperimentConfig(
        experiment_id="A6_ROI_HIER_LR_QUALITY_PCA",
        description=(
            "Optional series-local intensity-standard-deviation weighting "
            "heuristic; every slice remains included."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="quality",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
    ),
    ExperimentConfig(
        experiment_id="A7_ROI_HIER_LR_NO_PCA",
        description="PCA-off ablation with all other baseline settings unchanged.",
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
    ),
)

N_SPLITS = 5
CV_RANDOM_STATE = RANDOM_SEED
INNER_CV_SPLITS = 3
INNER_CV_RANDOM_STATE = RANDOM_SEED + 1000

CLASSIFIER_C_GRID = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_MAX_ITER = 4000
SVM_MAX_ITER = 20000
PATIENT_PCA_EXPLAINED_VARIANCE = 0.95

THRESHOLD_SELECTION_METHOD = "youden"
# Supported values:
#   "youden"           -> maximize sensitivity + specificity - 1
#   "target_sensitivity" -> choose the largest training-only threshold that
#                           reaches TARGET_SENSITIVITY when possible
TARGET_SENSITIVITY = 0.90

BOOTSTRAP_REPLICATES = 2000
PAIRED_BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95

USE_CUDA_AMP = True
DATALOADER_NUM_WORKERS = 0
ENABLE_DETAILED_PROGRESS_PRINTS = True
PROGRESS_PRINT_EVERY_N_BATCHES = 25

USE_FEATURE_CACHE = True
FORCE_REBUILD_FEATURE_CACHE = False
FEATURE_CACHE_SCHEMA_VERSION = "2026-08-29-multi-experiment-v1"
EFFICIENTNET_FEATURE_DIM = 1280

SLICE_QUALITY_MIN_WEIGHT = 0.25
# The quality signal remains a non-clinical heuristic. It is computed once from
# the confidence-gated ROI/fallback image and activated only by experiment A6.

BORDER_WIDTH_FRACTION = 0.15
# C1 retains only this outer fraction on each side of the 224x224 image.

AUDIT_EXACT_DECODED_PIXEL_DUPLICATES = True
AUDIT_PERCEPTUAL_NEAR_DUPLICATES = True
PHASH_HAMMING_THRESHOLD = 3
PHASH_BUCKET_BITS = 16
PHASH_MAX_BUCKET_SIZE = 300
PHASH_MAX_IMAGE_PAIR_CANDIDATES = 1_000_000

GROUP_SPLITS_BY_EXACT_DUPLICATES = True
GROUP_SPLITS_BY_PHASH_CANDIDATES = False
# Perceptual candidates are not automatically treated as confirmed duplicates
# by default. Set True only after reviewing the saved candidate table.

FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES = False
FAIL_ON_CROSS_LABEL_EXACT_DUPLICATES = False
FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS = False

SHORTCUT_WARNING_AUC = 0.65
MONAI_GATE_RATE_DIFFERENCE_WARNING = 0.20

DEBUG_VISUALIZATION = False
DEBUG_INDICES = "10%"

RUN_EXTERNAL_VALIDATION = False
EXTERNAL_DATASET_PATH = None
# External validation is deliberately disabled until an independent dataset with
# a documented compatible Normal/Sick endpoint and patient mapping is supplied.
# The suite records an explicit SKIPPED status rather than pretending that
# internal cross-validation is external validation.

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

# A deterministic suite tag prevents different registries or evaluation rules
# from silently overwriting one another. Frozen feature caches remain outside
# the suite folder because classifier/pooling settings do not change embeddings.
_selected_experiments_for_identity = [
    asdict(experiment)
    for experiment in EXPERIMENT_REGISTRY
    if experiment.enabled
    and (
        EXPERIMENTS_TO_RUN is None
        or experiment.experiment_id in set(EXPERIMENTS_TO_RUN)
    )
]

_suite_identity = {
    "experiments": _selected_experiments_for_identity,
    "random_seed": RANDOM_SEED,
    "n_splits": N_SPLITS,
    "inner_splits": INNER_CV_SPLITS,
    "c_grid": CLASSIFIER_C_GRID,
    "threshold_method": THRESHOLD_SELECTION_METHOD,
    "target_sensitivity": TARGET_SENSITIVITY,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "paired_bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
    "pca_variance": PATIENT_PCA_EXPLAINED_VARIANCE,
    "img_size": IMG_SIZE,
    "monai_input_size": MONAI_INPUT_SIZE,
    "monai_bundle": MONAI_BUNDLE_NAME,
    "monai_bundle_version": MONAI_BUNDLE_VERSION,
    "monai_hf_revision": MONAI_HF_REVISION,
    "roi_dilation": MONAI_ROI_DILATION_KERNEL,
    "roi_background": MONAI_BACKGROUND_WEIGHT,
    "roi_min_area": MONAI_MIN_HEART_AREA_RATIO,
    "roi_max_area": MONAI_MAX_HEART_AREA_RATIO,
    "roi_min_peak": MONAI_MIN_PEAK_HEART_PROBABILITY,
    "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
    "border_width_fraction": BORDER_WIDTH_FRACTION,
    "group_exact_duplicates": GROUP_SPLITS_BY_EXACT_DUPLICATES,
    "group_phash_candidates": GROUP_SPLITS_BY_PHASH_CANDIDATES,
}

SUITE_CONFIGURATION_TAG = hashlib.sha256(
    json.dumps(_suite_identity, sort_keys=True).encode("utf-8")
).hexdigest()[:10]
SUITE_NAME = f"multi_experiment_suite__{SUITE_CONFIGURATION_TAG}"

OUTPUT_DIR = OUTPUT_ROOT / SUITE_NAME
FEATURE_CACHE_ROOT = OUTPUT_ROOT / "feature_bank_cache"
CONSOLE_LOG_PATH = OUTPUT_DIR / "console_output.log"

if os.path.exists("/kaggle/working"):
    DEBUG_OUTPUT_DIR = Path("/kaggle/working/debug_output")
else:
    try:
        DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "debug_output"
    except NameError:
        DEBUG_OUTPUT_DIR = Path.cwd() / "debug_output"

# =============================
# EXECUTION STATUS + TIMING HELPERS
# =============================

PIPELINE_STAGE_COUNT = 11
# The number matches the suite orchestration stages inside ``main``.


def _synchronize_timing_device():
    """Synchronize CUDA before reading a timer when GPU work may be pending."""

    # CUDA kernels are normally asynchronous relative to Python. Without an
    # explicit synchronization, a timer can stop before the GPU has completed
    # the operation being measured. CPU execution requires no synchronization.
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _format_elapsed_time(seconds):
    """Format a duration for readable live console output."""

    seconds = max(0.0, float(seconds))

    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"

    if seconds < 60.0:
        return f"{seconds:.2f} s"

    minutes, remaining_seconds = divmod(seconds, 60.0)

    if minutes < 60.0:
        return f"{int(minutes)} min {remaining_seconds:.1f} s"

    hours, remaining_minutes = divmod(minutes, 60.0)
    return (
        f"{int(hours)} h {int(remaining_minutes)} min "
        f"{remaining_seconds:.1f} s"
    )


def _print_stage_start(stage_number, title, expected_workload):
    """Print a visible stage header and return a high-resolution start time."""

    print("\n" + "=" * 78, flush=True)
    print(
        f"[PIPELINE {stage_number:02d}/{PIPELINE_STAGE_COUNT:02d}] "
        f"START: {title}",
        flush=True,
    )
    print(
        f"[PIPELINE {stage_number:02d}/{PIPELINE_STAGE_COUNT:02d}] "
        f"Expected workload: {expected_workload}",
        flush=True,
    )
    print("=" * 78, flush=True)

    _synchronize_timing_device()
    return time.perf_counter()


def _print_stage_complete(stage_number, title, started_at, details=None):
    """Print measured stage duration and return it in seconds."""

    _synchronize_timing_device()
    elapsed = time.perf_counter() - started_at

    print(
        f"[PIPELINE {stage_number:02d}/{PIPELINE_STAGE_COUNT:02d}] "
        f"COMPLETED: {title} in {_format_elapsed_time(elapsed)}",
        flush=True,
    )

    if details:
        print(
            f"[PIPELINE {stage_number:02d}/{PIPELINE_STAGE_COUNT:02d}] "
            f"{details}",
            flush=True,
        )

    return float(elapsed)


def _print_stage_skipped(stage_number, title, reason):
    """Print an explicit message when a stage is safely bypassed."""

    print("\n" + "=" * 78, flush=True)
    print(
        f"[PIPELINE {stage_number:02d}/{PIPELINE_STAGE_COUNT:02d}] "
        f"SKIPPED: {title}",
        flush=True,
    )
    print(
        f"[PIPELINE {stage_number:02d}/{PIPELINE_STAGE_COUNT:02d}] "
        f"Reason: {reason}",
        flush=True,
    )
    print("=" * 78, flush=True)


def _print_detail(message):
    """Print a flushed substage message when detailed logging is enabled."""

    if ENABLE_DETAILED_PROGRESS_PRINTS:
        print(f"[DETAIL] {message}", flush=True)


def _print_timing_summary(stage_durations, total_elapsed):
    """Print one compact end-of-run timing table in execution order."""

    print("\n" + "-" * 78, flush=True)
    print("EXECUTION TIMING SUMMARY", flush=True)
    print("-" * 78, flush=True)

    for stage_key, duration in stage_durations.items():
        print(
            f"  {stage_key:<52} {_format_elapsed_time(duration):>20}",
            flush=True,
        )

    print("-" * 78, flush=True)
    print(
        f"  {'TOTAL PIPELINE RUNTIME':<52} "
        f"{_format_elapsed_time(total_elapsed):>20}",
        flush=True,
    )
    print("-" * 78, flush=True)

# =============================
# CONFIGURATION VALIDATION
# =============================

# This validation function is intentionally called before dataset scanning,
# model loading, downloading, GPU allocation, or feature extraction. A malformed
# setting should fail immediately, before the pipeline spends time or creates
# cache/output files that could later be mistaken for a valid experiment.
def get_enabled_experiments():
    """Return the ordered experiment list selected by hard-coded configuration."""

    selected_ids = None if EXPERIMENTS_TO_RUN is None else set(EXPERIMENTS_TO_RUN)
    experiments = [
        experiment
        for experiment in EXPERIMENT_REGISTRY
        if experiment.enabled
        and (selected_ids is None or experiment.experiment_id in selected_ids)
    ]

    if selected_ids is not None:
        known_ids = {experiment.experiment_id for experiment in EXPERIMENT_REGISTRY}
        unknown = sorted(selected_ids - known_ids)
        if unknown:
            raise ValueError(
                "EXPERIMENTS_TO_RUN contains unknown experiment IDs: "
                f"{unknown}."
            )

    return experiments


def validate_configuration():
    """Fail before expensive work when the experiment suite is inconsistent."""

    experiments = get_enabled_experiments()
    if not experiments:
        raise ValueError("At least one experiment must be enabled.")

    experiment_ids = [experiment.experiment_id for experiment in experiments]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Experiment IDs must be unique.")

    if BASELINE_EXPERIMENT_ID not in experiment_ids:
        raise ValueError(
            "The configured baseline must be present in the enabled suite: "
            f"{BASELINE_EXPERIMENT_ID!r}."
        )

    valid_feature_modes = {
        "monai_roi",
        "full_image",
        "border_only",
        "outside_heart",
        "provenance_only",
        "monai_qc_only",
    }
    valid_strategies = {
        "patient_embedding",
        "slice_probability_fusion",
        "patient_tabular",
    }
    valid_pooling = {
        "hierarchical",
        "flat",
        "probability_fusion",
        "patient_tabular",
    }
    valid_weighting = {"equal", "quality", "not_applicable"}
    valid_classifiers = {"logistic_regression", "linear_svm"}
    valid_fusion = {None, "log_odds", "mean_probability"}

    for experiment in experiments:
        if experiment.feature_mode not in valid_feature_modes:
            raise ValueError(
                f"{experiment.experiment_id}: invalid feature mode "
                f"{experiment.feature_mode!r}."
            )
        if experiment.strategy not in valid_strategies:
            raise ValueError(
                f"{experiment.experiment_id}: invalid strategy "
                f"{experiment.strategy!r}."
            )
        if experiment.pooling_strategy not in valid_pooling:
            raise ValueError(
                f"{experiment.experiment_id}: invalid pooling strategy "
                f"{experiment.pooling_strategy!r}."
            )
        if experiment.weighting_mode not in valid_weighting:
            raise ValueError(
                f"{experiment.experiment_id}: invalid weighting mode "
                f"{experiment.weighting_mode!r}."
            )
        if experiment.classifier_type not in valid_classifiers:
            raise ValueError(
                f"{experiment.experiment_id}: invalid classifier "
                f"{experiment.classifier_type!r}."
            )
        if experiment.fusion_method not in valid_fusion:
            raise ValueError(
                f"{experiment.experiment_id}: invalid fusion method "
                f"{experiment.fusion_method!r}."
            )
        if experiment.fixed_c <= 0:
            raise ValueError(
                f"{experiment.experiment_id}: fixed_c must be positive."
            )

        if experiment.strategy == "slice_probability_fusion":
            if experiment.classifier_type != "logistic_regression":
                raise ValueError(
                    f"{experiment.experiment_id}: the reviewed legacy branch "
                    "supports Logistic Regression only."
                )
            if experiment.fusion_method is None:
                raise ValueError(
                    f"{experiment.experiment_id}: legacy probability fusion "
                    "requires an explicit fusion method."
                )
            if experiment.use_pca:
                raise ValueError(
                    f"{experiment.experiment_id}: PCA is intentionally disabled "
                    "for the legacy slice-level branch."
                )

        if experiment.strategy == "patient_tabular":
            if experiment.feature_mode not in {
                "provenance_only",
                "monai_qc_only",
            }:
                raise ValueError(
                    f"{experiment.experiment_id}: patient_tabular requires a "
                    "tabular feature mode."
                )

        if experiment.classifier_type == "linear_svm" and (
            experiment.strategy != "patient_embedding"
        ):
            raise ValueError(
                f"{experiment.experiment_id}: Linear SVM is currently reviewed "
                "only for one-row-per-patient inputs."
            )

    if IMG_SIZE != 224:
        raise ValueError(
            "This reviewed EfficientNet-B0 pipeline requires IMG_SIZE=224."
        )
    if MONAI_INPUT_SIZE != 256:
        raise ValueError(
            "The pinned ventricular bundle requires MONAI_INPUT_SIZE=256."
        )
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")
    if DATALOADER_NUM_WORKERS < 0:
        raise ValueError("DATALOADER_NUM_WORKERS cannot be negative.")
    if PROGRESS_PRINT_EVERY_N_BATCHES <= 0:
        raise ValueError("PROGRESS_PRINT_EVERY_N_BATCHES must be positive.")

    if N_SPLITS < 2 or INNER_CV_SPLITS < 2:
        raise ValueError("Outer and inner CV must each contain at least 2 folds.")
    if not CLASSIFIER_C_GRID or any(value <= 0 for value in CLASSIFIER_C_GRID):
        raise ValueError("CLASSIFIER_C_GRID must contain positive values.")
    if LOGISTIC_MAX_ITER <= 0 or SVM_MAX_ITER <= 0:
        raise ValueError("Classifier iteration limits must be positive.")
    if not 0.0 < PATIENT_PCA_EXPLAINED_VARIANCE < 1.0:
        raise ValueError(
            "PATIENT_PCA_EXPLAINED_VARIANCE must lie strictly in (0,1)."
        )

    if THRESHOLD_SELECTION_METHOD not in {"youden", "target_sensitivity"}:
        raise ValueError(
            "THRESHOLD_SELECTION_METHOD must be 'youden' or "
            "'target_sensitivity'."
        )
    if not 0.0 < TARGET_SENSITIVITY <= 1.0:
        raise ValueError("TARGET_SENSITIVITY must lie in (0,1].")
    if BOOTSTRAP_REPLICATES <= 0 or PAIRED_BOOTSTRAP_REPLICATES <= 0:
        raise ValueError("Bootstrap replicate counts must be positive.")
    if not 0.0 < BOOTSTRAP_CONFIDENCE < 1.0:
        raise ValueError("BOOTSTRAP_CONFIDENCE must lie strictly in (0,1).")

    if not 0.0 < SLICE_QUALITY_MIN_WEIGHT <= 1.0:
        raise ValueError("SLICE_QUALITY_MIN_WEIGHT must lie in (0,1].")
    if not 0.0 < BORDER_WIDTH_FRACTION < 0.5:
        raise ValueError("BORDER_WIDTH_FRACTION must lie in (0,0.5).")

    if PHASH_BUCKET_BITS <= 0 or 64 % PHASH_BUCKET_BITS != 0:
        raise ValueError("PHASH_BUCKET_BITS must divide 64 exactly.")
    phash_chunks = 64 // PHASH_BUCKET_BITS
    if PHASH_HAMMING_THRESHOLD < 0:
        raise ValueError("PHASH_HAMMING_THRESHOLD cannot be negative.")
    if PHASH_HAMMING_THRESHOLD >= phash_chunks:
        raise ValueError(
            "The current exact-chunk LSH audit guarantees candidate recall only "
            "when PHASH_HAMMING_THRESHOLD is smaller than the number of chunks."
        )
    if PHASH_MAX_BUCKET_SIZE <= 1:
        raise ValueError("PHASH_MAX_BUCKET_SIZE must exceed 1.")
    if PHASH_MAX_IMAGE_PAIR_CANDIDATES <= 0:
        raise ValueError("PHASH_MAX_IMAGE_PAIR_CANDIDATES must be positive.")

    if FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES and (
        not AUDIT_EXACT_DECODED_PIXEL_DUPLICATES
    ):
        raise ValueError(
            "The strict exact-duplicate policy requires the exact audit."
        )
    if GROUP_SPLITS_BY_EXACT_DUPLICATES and (
        not AUDIT_EXACT_DECODED_PIXEL_DUPLICATES
    ):
        raise ValueError(
            "Exact-duplicate-aware folds require exact duplicate auditing."
        )
    if GROUP_SPLITS_BY_PHASH_CANDIDATES and (
        not AUDIT_PERCEPTUAL_NEAR_DUPLICATES
    ):
        raise ValueError(
            "Perceptual-candidate grouping requires the perceptual audit."
        )

    if MONAI_ROI_DILATION_KERNEL <= 0 or (
        MONAI_ROI_DILATION_KERNEL % 2 == 0
    ):
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
                    f"{digest_name} must be a 64-character hexadecimal digest."
                )

    if EFFICIENTNET_WEIGHTS_NAME not in (
        models.EfficientNet_B0_Weights.__members__
    ):
        raise ValueError(
            f"Unknown EfficientNet-B0 weight enum: "
            f"{EFFICIENTNET_WEIGHTS_NAME!r}."
        )

    _parse_debug_indices(DEBUG_INDICES)

    if RUN_EXTERNAL_VALIDATION and not EXTERNAL_DATASET_PATH:
        raise ValueError(
            "RUN_EXTERNAL_VALIDATION=True requires EXTERNAL_DATASET_PATH."
        )


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
# MRI SLICE DATASET + PROVENANCE FEATURES
# =============================

PROVENANCE_FEATURE_NAMES = (
    "native_height",
    "native_width",
    "aspect_ratio_width_over_height",
    "file_size_bytes",
    "bytes_per_native_pixel",
    "raw_mean_intensity",
    "raw_std_intensity",
    "raw_entropy_bits",
    "near_black_fraction",
    "near_white_fraction",
    "border_mean_intensity",
    "border_std_intensity",
    "border_near_black_fraction",
    "center_mean_intensity",
    "edge_pixel_fraction",
    "laplacian_variance",
)


def compute_dct_perceptual_hash(image):
    """Return a deterministic 64-bit DCT perceptual hash as 16 hex digits."""

    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct_values = cv2.dct(resized.astype(np.float32))
    low_frequency = dct_values[:8, :8].reshape(-1)

    # Exclude the DC coefficient when choosing the threshold because it mostly
    # reflects global brightness. The DC bit itself remains in the 64-bit hash.
    median = float(np.median(low_frequency[1:]))
    bits = low_frequency > median

    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))

    return f"{value:016x}"


def compute_image_provenance_features(image, image_path):
    """Extract label-free export/style features from the native uint8 image."""

    if image.ndim != 2:
        raise ValueError(
            f"Provenance extraction expects a 2D image, got {image.shape}."
        )

    height, width = image.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid native image shape: {image.shape}.")

    image_float = image.astype(np.float32)
    file_size = float(Path(image_path).stat().st_size)
    pixel_count = float(height * width)

    histogram = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram / max(histogram.sum(), 1.0)
    nonzero = probabilities > 0
    entropy = float(
        -np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))
    )

    border_width = max(1, int(round(min(height, width) * 0.10)))
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border_pixels = image_float[border_mask]

    if height > 2 * border_width and width > 2 * border_width:
        center_pixels = image_float[
            border_width:height - border_width,
            border_width:width - border_width,
        ]
    else:
        center_pixels = image_float

    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    laplacian = cv2.Laplacian(image_float, cv2.CV_32F)

    features = np.asarray(
        [
            float(height),
            float(width),
            float(width / height),
            file_size,
            float(file_size / pixel_count),
            float(image_float.mean()),
            float(image_float.std()),
            entropy,
            float(np.mean(image <= 5)),
            float(np.mean(image >= 250)),
            float(border_pixels.mean()),
            float(border_pixels.std()),
            float(np.mean(border_pixels <= 5)),
            float(center_pixels.mean()),
            float(np.mean(edges > 0)),
            float(laplacian.var()),
        ],
        dtype=np.float32,
    )

    if len(features) != len(PROVENANCE_FEATURE_NAMES):
        raise RuntimeError("Provenance feature-name and value counts differ.")
    if not np.all(np.isfinite(features)):
        raise RuntimeError(
            f"Non-finite provenance feature generated for {image_path}."
        )

    return features


class MRIDataset(Dataset):
    """Load one JPEG and return aligned network inputs plus audit metadata.

    Each item contains:

        classification_image : [3,224,224] float tensor in [0,1]
        monai_image          : [1,256,256] float tensor in [0,1]
        label                : scalar 0/1 metadata
        patient_id           : validated Directory_* identifier
        series_id            : patient-scoped folder proxy
        sample_index         : deterministic position in the discovered list
        decoded_pixel_hash   : exact SHA-256 of native decoded pixels + shape
        perceptual_hash      : 64-bit DCT pHash candidate key
        provenance_features  : label-free native export/style feature vector

    The class label is never used to construct pixels, masks, hashes, or
    provenance features.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id, series_id = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(
                f"OpenCV could not read the MRI image: {img_path}"
            )

        pixel_digest = hashlib.sha256()
        pixel_digest.update(np.asarray(image.shape, dtype=np.int32).tobytes())
        pixel_digest.update(image.tobytes(order="C"))
        decoded_pixel_hash = pixel_digest.hexdigest()

        perceptual_hash = compute_dct_perceptual_hash(image)
        provenance_features = compute_image_provenance_features(
            image,
            img_path,
        )

        image = scale_intensity_0_1(image)
        monai_canvas = zero_pad_to_monai_canvas(image)
        monai_image = torch.from_numpy(monai_canvas).unsqueeze(0)

        classification_gray = cv2.resize(
            monai_canvas,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
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
            idx,
            decoded_pixel_hash,
            perceptual_hash,
            torch.from_numpy(provenance_features),
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

    discovery_started_at = time.perf_counter()
    print(
        f"[DATA] Starting dataset discovery under: {root_dir}",
        flush=True,
    )
    print(
        "[DATA] This stage scans folders and filenames only; JPEG pixels are "
        "decoded later during feature extraction.",
        flush=True,
    )

    # ``samples`` preserves one row per discovered image. The two maps/sets
    # below enforce global patient identity and label consistency independently
    # of how many series folders or images each patient contains.
    samples = []
    discovered_patients = set()
    patient_to_class = {}

    # Stage 1: traverse the two expected top-level class directories in a
    # fixed order. Sorting at every lower level makes discovery deterministic.
    for class_name in ["Normal", "Sick"]:

        class_started_at = time.perf_counter()
        class_start_images = len(samples)
        class_start_patients = len(discovered_patients)
        print(
            f"[DATA] Scanning class folder: {class_name}",
            flush=True,
        )

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

            # # TEMP - Only for test
            # if (directory.lower() != "directory_1" and directory.lower() != "directory_17"
            #         and directory.lower() != "directory_2" and directory.lower() != "directory_18"
            # ):
            #     continue

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

        class_elapsed = time.perf_counter() - class_started_at
        print(
            f"[DATA] Finished {class_name}: "
            f"patients_added={len(discovered_patients) - class_start_patients}, "
            f"images_added={len(samples) - class_start_images}, "
            f"elapsed={_format_elapsed_time(class_elapsed)}",
            flush=True,
        )

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
    print(
        "[DATA] Dataset discovery completed in "
        f"{_format_elapsed_time(time.perf_counter() - discovery_started_at)}",
        flush=True,
    )

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

    ensure_started_at = time.perf_counter()
    _print_detail(
        "Checking whether the pinned MONAI bundle files are already present."
    )

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
    _print_detail(f"MONAI bundle root resolved to: {bundle_root}")
    missing = [
        relative_path
        for relative_path in required_relative_paths
        if not (bundle_root / relative_path).is_file()
    ]

    if not missing:
        validate_monai_bundle_metadata(bundle_root)
        elapsed = time.perf_counter() - ensure_started_at
        print(
            f"[MODEL][MONAI] Required pinned files are already cached at "
            f"{bundle_root} (check completed in {_format_elapsed_time(elapsed)}).",
            flush=True,
        )
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
        "[MODEL][MONAI] Downloading missing bundle files from the pinned "
        f"repository revision {MONAI_HF_REVISION[:8]}: {', '.join(missing)}",
        flush=True,
    )
    print(
        "[MODEL][MONAI] This can be slow on the first run and depends on the "
        "network connection. Later runs should reuse the local files.",
        flush=True,
    )
    download_started_at = time.perf_counter()

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

    print(
        "[MODEL][MONAI] Download call completed in "
        f"{_format_elapsed_time(time.perf_counter() - download_started_at)}.",
        flush=True,
    )

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
    elapsed = time.perf_counter() - ensure_started_at
    print(
        f"[MODEL][MONAI] Pinned files are ready and cached at {bundle_root}. "
        f"Total preparation time: {_format_elapsed_time(elapsed)}",
        flush=True,
    )
    return bundle_root


def load_and_validate_torchscript_segmenter(path, source_description):
    """Load TorchScript and validate its fixed inference contract immediately."""

    global MONAI_RUNTIME_SOURCE, MONAI_RUNTIME_ARTIFACT_PATH

    load_started_at = time.perf_counter()
    print(
        f"[MODEL][MONAI] Loading {source_description} from: {path}",
        flush=True,
    )
    print(
        f"[MODEL][MONAI] Target device: {DEVICE}. A zero-input inference "
        "sanity check will run immediately after loading.",
        flush=True,
    )

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

    _print_detail("Running MONAI zero-input shape/finite-value validation.")
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

    _synchronize_timing_device()
    elapsed = time.perf_counter() - load_started_at
    print(
        f"[MODEL][MONAI] Loaded and validated {source_description} in "
        f"{_format_elapsed_time(elapsed)}; output_shape={tuple(example_output.shape)}.",
        flush=True,
    )
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

    build_started_at = time.perf_counter()
    print(
        "[MODEL][MONAI] Preparing the pretrained ventricular segmenter.",
        flush=True,
    )
    print(
        "[MODEL][MONAI] Preferred path: verified official model.ts without "
        "importing the MONAI Python package.",
        flush=True,
    )

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
            network = load_and_validate_torchscript_segmenter(
                official_torchscript_path,
                "official pinned MONAI TorchScript segmenter",
            )
            print(
                "[MODEL][MONAI] Segmenter preparation completed through the "
                f"preferred official path in "
                f"{_format_elapsed_time(time.perf_counter() - build_started_at)}.",
                flush=True,
            )
            return network
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
            network = load_and_validate_torchscript_segmenter(
                MONAI_TORCHSCRIPT_PATH,
                "locally reconstructed MONAI TorchScript cache",
            )
            print(
                "[MODEL][MONAI] Segmenter preparation completed through the "
                f"local fallback cache in "
                f"{_format_elapsed_time(time.perf_counter() - build_started_at)}.",
                flush=True,
            )
            return network
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

    monai_import_started_at = time.perf_counter()
    print(
        "[MODEL][MONAI] Importing monai.networks.nets.UNet for the exceptional "
        "fallback. This import can take noticeably longer on some systems.",
        flush=True,
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

    print(
        "[MODEL][MONAI] MONAI import completed in "
        f"{_format_elapsed_time(time.perf_counter() - monai_import_started_at)}.",
        flush=True,
    )

    network = MONAIUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    checkpoint_started_at = time.perf_counter()
    _print_detail(f"Loading MONAI fallback state dictionary: {model_path}")
    state_dict = load_checkpoint_state_dict(model_path)
    _print_detail(
        "MONAI fallback checkpoint loaded in "
        f"{_format_elapsed_time(time.perf_counter() - checkpoint_started_at)}."
    )

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

    trace_started_at = time.perf_counter()
    print(
        "[MODEL][MONAI] Tracing and validating the reconstructed network. "
        "This is a one-time fallback cost when the resulting cache is reused.",
        flush=True,
    )

    with torch.inference_mode():
        reference_output = network(example_input)

        traced_network = torch.jit.trace(
            network,
            example_input,
            strict=False,
        )

        traced_output = traced_network(example_input)

    print(
        "[MODEL][MONAI] Trace creation and first validation inference completed "
        f"in {_format_elapsed_time(time.perf_counter() - trace_started_at)}.",
        flush=True,
    )

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

    network = load_and_validate_torchscript_segmenter(
        MONAI_TORCHSCRIPT_PATH,
        "newly reconstructed MONAI TorchScript cache",
    )
    print(
        "[MODEL][MONAI] Exceptional fallback preparation completed in "
        f"{_format_elapsed_time(time.perf_counter() - build_started_at)}.",
        flush=True,
    )
    return network


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

        initialization_started_at = time.perf_counter()
        print(
            f"[MODEL][EfficientNet] Loading EfficientNet-B0 weights "
            f"{EFFICIENTNET_WEIGHTS_NAME}.",
            flush=True,
        )
        print(
            "[MODEL][EfficientNet] The first run may download the ImageNet "
            "checkpoint; later runs should use the torchvision cache.",
            flush=True,
        )

        weights = models.EfficientNet_B0_Weights[
            EFFICIENTNET_WEIGHTS_NAME
        ]

        self.model = models.efficientnet_b0(weights=weights)

        # Replace only the final classifier. The convolutional backbone and
        # global pooling remain intact, so forward() returns a 1280-D vector.
        self.model.classifier = nn.Identity()

        print(
            "[MODEL][EfficientNet] Backbone initialized and ImageNet "
            f"classifier removed in "
            f"{_format_elapsed_time(time.perf_counter() - initialization_started_at)}.",
            flush=True,
        )

    def forward(self, x):


        # No additional trainable layer is introduced here. Input tensors pass
        # directly through the frozen EfficientNet feature encoder.

        return self.model(x)



# =============================
# PIPELINE STEP 5
# MULTI-VIEW FEATURE BANK
# =============================


def _normalize_quality_weights(scores, minimum_weight=SLICE_QUALITY_MIN_WEIGHT):
    """Convert one series proxy's heuristic scores into positive mean-one weights."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot normalize an empty quality-score collection.")
    if scores.size == 1:
        return np.ones(1, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Quality scores must be finite.")

    q25, q75 = np.percentile(scores, [25, 75])
    scale = max(float(q75 - q25), 1e-8)
    normalized = np.clip((scores - q25) / scale, 0.0, 1.0)
    weights = minimum_weight + (1.0 - minimum_weight) * normalized
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def _parse_debug_indices(selection):
    """Return None for full selection or a percentage in [0,100]."""

    if isinstance(selection, str):
        normalized = selection.strip().lower()
        if normalized == "full":
            return None
        if not normalized.endswith("%"):
            raise ValueError(
                'DEBUG_INDICES must be "full" or a percentage such as "10%".'
            )
        numeric_value = normalized[:-1].strip()
    elif isinstance(selection, (int, float)) and not isinstance(selection, bool):
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
        raise ValueError("DEBUG_INDICES percentage must lie in [0,100].")
    return percentage


def choose_debug_sample_indices(samples, selection):
    """Choose all images or an evenly spaced deterministic percentage."""

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
    return set(np.linspace(0, n_images - 1, n_select, dtype=int).tolist())


def required_efficientnet_feature_modes(experiments):
    """Return ordered image modes needed by the selected registry."""

    canonical_order = (
        "monai_roi",
        "full_image",
        "border_only",
        "outside_heart",
    )
    requested = {
        experiment.feature_mode
        for experiment in experiments
        if experiment.feature_mode in canonical_order
    }
    return tuple(mode for mode in canonical_order if mode in requested)


def feature_bank_fingerprint(samples, dataset_root):
    """Hash the dataset inventory and every setting that changes cached features."""

    started_at = time.perf_counter()
    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()

    settings = {
        "schema": FEATURE_CACHE_SCHEMA_VERSION,
        "img_size": IMG_SIZE,
        "monai_input_size": MONAI_INPUT_SIZE,
        "monai_bundle": MONAI_BUNDLE_NAME,
        "monai_bundle_version": MONAI_BUNDLE_VERSION,
        "monai_hf_revision": MONAI_HF_REVISION,
        "monai_model_ts_sha256": MONAI_OFFICIAL_TORCHSCRIPT_SHA256,
        "roi_dilation": MONAI_ROI_DILATION_KERNEL,
        "roi_background": MONAI_BACKGROUND_WEIGHT,
        "roi_min_area": MONAI_MIN_HEART_AREA_RATIO,
        "roi_max_area": MONAI_MAX_HEART_AREA_RATIO,
        "roi_min_peak": MONAI_MIN_PEAK_HEART_PROBABILITY,
        "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
        "efficientnet_mean": EFFICIENTNET_MEAN,
        "efficientnet_std": EFFICIENTNET_STD,
        "border_width_fraction": BORDER_WIDTH_FRACTION,
        "provenance_features": PROVENANCE_FEATURE_NAMES,
        "perceptual_hash": "32x32_DCT_8x8_64bit_v1",
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))

    progress_step = max(1, len(samples) // 10)
    for index, (image_path, label, patient_id, series_id) in enumerate(
        samples,
        start=1,
    ):
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

        if ENABLE_DETAILED_PROGRESS_PRINTS and (
            index == 1 or index % progress_step == 0 or index == len(samples)
        ):
            print(
                f"[FEATURE BANK FINGERPRINT] {index}/{len(samples)} files "
                f"({100.0 * index / len(samples):.0f}%)",
                flush=True,
            )

    fingerprint = digest.hexdigest()
    print(
        "[FEATURE BANK] Fingerprint completed in "
        f"{_format_elapsed_time(time.perf_counter() - started_at)}: "
        f"{fingerprint[:16]}...",
        flush=True,
    )
    return fingerprint


def _feature_bank_shared_paths(cache_dir):
    names = (
        "labels",
        "patient_ids",
        "series_ids",
        "sample_indices",
        "decoded_pixel_hashes",
        "perceptual_hashes",
        "provenance_features",
        "monai_valid",
        "area_ratios",
        "peak_probabilities",
        "mean_foreground_probabilities",
        "roi_slice_scores",
    )
    return {name: cache_dir / f"{name}.npy" for name in names}


def _feature_mode_path(cache_dir, mode):
    return cache_dir / f"features__{mode}.npy"


def load_feature_bank(cache_dir, expected_fingerprint, required_modes):
    """Load a complete matching feature bank through memory maps."""

    metadata_path = cache_dir / "metadata.json"
    shared_paths = _feature_bank_shared_paths(cache_dir)

    if not metadata_path.is_file():
        return None
    if not all(path.is_file() for path in shared_paths.values()):
        return None
    if not all(_feature_mode_path(cache_dir, mode).is_file() for mode in required_modes):
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if metadata.get("fingerprint") != expected_fingerprint:
        return None
    if not set(required_modes).issubset(set(metadata.get("completed_modes", []))):
        return None

    # Preserve model provenance even when the expensive feature bank is reused
    # and the MONAI network itself is not loaded in the current process.
    global MONAI_RUNTIME_SOURCE, MONAI_RUNTIME_ARTIFACT_PATH
    MONAI_RUNTIME_SOURCE = metadata.get(
        "monai_runtime_source",
        MONAI_RUNTIME_SOURCE,
    )
    MONAI_RUNTIME_ARTIFACT_PATH = metadata.get(
        "monai_runtime_artifact_path",
        MONAI_RUNTIME_ARTIFACT_PATH,
    )

    print(f"[FEATURE BANK] Loading cache: {cache_dir}", flush=True)
    started_at = time.perf_counter()

    bank = {
        "cache_dir": cache_dir,
        "fingerprint": expected_fingerprint,
        "metadata": metadata,
        "features": {
            mode: np.load(
                _feature_mode_path(cache_dir, mode),
                mmap_mode="r",
                allow_pickle=False,
            )
            for mode in required_modes
        },
    }
    for name, path in shared_paths.items():
        bank[name] = np.load(path, mmap_mode="r", allow_pickle=False)

    n_slices = len(bank["labels"])
    for mode, features in bank["features"].items():
        if features.shape != (n_slices, EFFICIENTNET_FEATURE_DIM):
            raise RuntimeError(
                f"Feature bank mode {mode!r} has unexpected shape "
                f"{features.shape}."
            )

    print(
        "[FEATURE BANK] Cache hit loaded in "
        f"{_format_elapsed_time(time.perf_counter() - started_at)}; "
        f"slices={n_slices}, modes={list(required_modes)}.",
        flush=True,
    )
    return bank


def create_border_only_images(images):
    """Retain only a fixed outer border and set the central region to zero."""

    height, width = images.shape[-2:]
    border = max(1, int(round(min(height, width) * BORDER_WIDTH_FRACTION)))
    border_mask = torch.ones(
        1,
        1,
        height,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    if height > 2 * border and width > 2 * border:
        border_mask[:, :, border:height - border, border:width - border] = 0.0
    return images * border_mask


def create_outside_monai_mask_images(images, roi_probability):
    """Retain signal outside the dilated soft MONAI probability map.

    The inverse map is applied even when the plausibility gate is false. This is
    deliberate: the control asks whether non-ROI/export context is predictive,
    but it must not be described as a validated extracardiac-anatomy mask.
    """

    outside_weight = (1.0 - roi_probability.clamp(0.0, 1.0)).repeat(1, 3, 1, 1)
    return images * outside_weight


def extract_feature_bank(
    dataset,
    monai_segmenter,
    feature_extractor,
    required_modes,
    cache_dir,
    fingerprint,
    debug=False,
):
    """Extract all requested frozen image variants in one dataset pass.

    JPEG decoding and MONAI inference occur once per batch. Each requested image
    variant is then encoded sequentially by the same frozen EfficientNet-B0. The
    resulting feature matrices are written directly to .npy memory maps, so four
    1280-D variants do not have to reside in RAM simultaneously.
    """

    if not required_modes:
        raise ValueError("At least one EfficientNet feature mode is required.")

    need_monai = any(
        mode in {"monai_roi", "outside_heart"}
        for mode in required_modes
    )
    if need_monai and monai_segmenter is None:
        raise RuntimeError(
            "MONAI-dependent feature modes were requested without a segmenter."
        )

    if cache_dir.exists():
        print(
            f"[FEATURE BANK] Removing incomplete or deliberately rebuilt cache: "
            f"{cache_dir}",
            flush=True,
        )
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_slices = len(dataset)
    if n_slices <= 0:
        raise RuntimeError("Cannot extract a feature bank from an empty dataset.")

    feature_maps = {
        mode: np.lib.format.open_memmap(
            _feature_mode_path(cache_dir, mode),
            mode="w+",
            dtype=np.float32,
            shape=(n_slices, EFFICIENTNET_FEATURE_DIM),
        )
        for mode in required_modes
    }

    labels_array = np.empty(n_slices, dtype=np.int64)
    sample_indices_array = np.empty(n_slices, dtype=np.int64)
    provenance_array = np.empty(
        (n_slices, len(PROVENANCE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    monai_valid_array = np.zeros(n_slices, dtype=bool)
    area_ratio_array = np.full(n_slices, np.nan, dtype=np.float32)
    peak_probability_array = np.full(n_slices, np.nan, dtype=np.float32)
    mean_foreground_array = np.full(n_slices, np.nan, dtype=np.float32)
    roi_slice_score_array = np.full(n_slices, np.nan, dtype=np.float32)

    patient_ids_values = [None] * n_slices
    series_ids_values = [None] * n_slices
    decoded_hash_values = [None] * n_slices
    perceptual_hash_values = [None] * n_slices

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(DATALOADER_NUM_WORKERS > 0),
    )

    total_batches = len(loader)
    progress_interval = max(
        1,
        PROGRESS_PRINT_EVERY_N_BATCHES,
        total_batches // 20,
    )
    debug_indices = (
        choose_debug_sample_indices(dataset.samples, DEBUG_INDICES)
        if debug and need_monai
        else set()
    )
    autocast_enabled = USE_CUDA_AMP and DEVICE == "cuda"
    extraction_started_at = time.perf_counter()
    processed_slices = 0

    print(
        "[FEATURE BANK] Starting multi-view frozen extraction.",
        flush=True,
    )
    print(
        f"[FEATURE BANK] slices={n_slices}, batches={total_batches}, "
        f"batch_size={BATCH_SIZE}, modes={list(required_modes)}, "
        f"MONAI_required={need_monai}, device={DEVICE}",
        flush=True,
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(
                loader,
                desc="Feature bank",
                unit="batch",
                dynamic_ncols=True,
                file=sys.stdout,
            ),
            start=1,
        ):
            batch_started_at = time.perf_counter()
            (
                images,
                monai_images,
                labels,
                patient_ids,
                series_ids,
                sample_indices,
                decoded_pixel_hashes,
                perceptual_hashes,
                provenance_features,
            ) = batch

            index_values = sample_indices.detach().cpu().numpy().astype(np.int64)
            images = images.to(DEVICE, non_blocking=True)
            monai_images = monai_images.to(DEVICE, non_blocking=True)

            if need_monai:
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
                roi_probability = torch.zeros(
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

            variants = {}
            if "monai_roi" in required_modes:
                variants["monai_roi"] = roi_images
            if "full_image" in required_modes:
                variants["full_image"] = images
            if "border_only" in required_modes:
                variants["border_only"] = create_border_only_images(images)
            if "outside_heart" in required_modes:
                variants["outside_heart"] = create_outside_monai_mask_images(
                    images,
                    roi_probability,
                )

            for mode in required_modes:
                mode_images = variants[mode]
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    network_input = normalize_for_efficientnet(mode_images)
                    batch_features = feature_extractor(network_input).float()

                if batch_features.shape != (
                    len(index_values),
                    EFFICIENTNET_FEATURE_DIM,
                ):
                    raise RuntimeError(
                        f"Unexpected EfficientNet feature shape for {mode}: "
                        f"{tuple(batch_features.shape)}."
                    )

                feature_maps[mode][index_values, :] = (
                    batch_features.detach().cpu().numpy()
                )

            roi_scores = torch.std(roi_images, dim=(1, 2, 3))

            labels_array[index_values] = labels.detach().cpu().numpy()
            sample_indices_array[index_values] = index_values
            provenance_array[index_values, :] = (
                provenance_features.detach().cpu().numpy()
            )
            monai_valid_array[index_values] = valid_mask.detach().cpu().numpy()
            area_ratio_array[index_values] = area_ratio.detach().cpu().numpy()
            peak_probability_array[index_values] = (
                peak_probability.detach().cpu().numpy()
            )
            mean_foreground_array[index_values] = (
                mean_foreground_probability.detach().cpu().numpy()
            )
            roi_slice_score_array[index_values] = roi_scores.detach().cpu().numpy()

            for local_position, global_index in enumerate(index_values.tolist()):
                patient_ids_values[global_index] = str(patient_ids[local_position])
                series_ids_values[global_index] = str(series_ids[local_position])
                decoded_hash_values[global_index] = str(
                    decoded_pixel_hashes[local_position]
                )
                perceptual_hash_values[global_index] = str(
                    perceptual_hashes[local_position]
                )

            if debug_indices:
                selected_positions = [
                    position
                    for position, global_index in enumerate(index_values.tolist())
                    if global_index in debug_indices
                ]
                if selected_positions:
                    debug_visualization(
                        images[selected_positions],
                        roi_probability[selected_positions],
                        hard_mask[selected_positions],
                        roi_images[selected_positions],
                        roi_scores[selected_positions],
                        valid_mask[selected_positions],
                        area_ratio[selected_positions],
                        peak_probability[selected_positions],
                        mean_foreground_probability[selected_positions],
                        labels[selected_positions],
                        [patient_ids[i] for i in selected_positions],
                        [series_ids[i] for i in selected_positions],
                        sample_indices[selected_positions],
                    )

            processed_slices += len(index_values)
            if (
                batch_index == 1
                or batch_index % progress_interval == 0
                or batch_index == total_batches
            ):
                _synchronize_timing_device()
                elapsed = time.perf_counter() - extraction_started_at
                rate = processed_slices / max(elapsed, 1e-8)
                remaining = n_slices - processed_slices
                eta = remaining / max(rate, 1e-8)
                print(
                    f"[FEATURE BANK] {processed_slices}/{n_slices} slices "
                    f"({100.0 * processed_slices / n_slices:.1f}%) | "
                    f"batch={batch_index}/{total_batches} | "
                    f"rate={rate:.2f} slices/s | ETA={_format_elapsed_time(eta)} | "
                    f"latest_batch={_format_elapsed_time(time.perf_counter() - batch_started_at)}",
                    flush=True,
                )

    if any(value is None for value in patient_ids_values):
        raise RuntimeError("At least one patient ID was not written to the bank.")
    if any(value is None for value in decoded_hash_values):
        raise RuntimeError("At least one decoded hash was not written to the bank.")
    if not np.array_equal(sample_indices_array, np.arange(n_slices)):
        raise RuntimeError("Feature-bank sample indices are incomplete or reordered.")

    for feature_map in feature_maps.values():
        feature_map.flush()
    del feature_maps

    shared_paths = _feature_bank_shared_paths(cache_dir)
    shared_arrays = {
        "labels": labels_array,
        "patient_ids": np.asarray(patient_ids_values),
        "series_ids": np.asarray(series_ids_values),
        "sample_indices": sample_indices_array,
        "decoded_pixel_hashes": np.asarray(decoded_hash_values),
        "perceptual_hashes": np.asarray(perceptual_hash_values),
        "provenance_features": provenance_array,
        "monai_valid": monai_valid_array,
        "area_ratios": area_ratio_array,
        "peak_probabilities": peak_probability_array,
        "mean_foreground_probabilities": mean_foreground_array,
        "roi_slice_scores": roi_slice_score_array,
    }
    for name, array in shared_arrays.items():
        np.save(shared_paths[name], np.asarray(array), allow_pickle=False)

    metadata = {
        "fingerprint": fingerprint,
        "schema": FEATURE_CACHE_SCHEMA_VERSION,
        "completed_modes": list(required_modes),
        "n_slices": int(n_slices),
        "feature_dimension": EFFICIENTNET_FEATURE_DIM,
        "provenance_feature_names": list(PROVENANCE_FEATURE_NAMES),
        "monai_runtime_source": MONAI_RUNTIME_SOURCE,
        "monai_runtime_artifact_path": MONAI_RUNTIME_ARTIFACT_PATH,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _synchronize_timing_device()
    print(
        "[FEATURE BANK] Extraction and cache write completed in "
        f"{_format_elapsed_time(time.perf_counter() - extraction_started_at)}.",
        flush=True,
    )

    loaded = load_feature_bank(cache_dir, fingerprint, required_modes)
    if loaded is None:
        raise RuntimeError("Freshly written feature bank could not be reloaded.")
    return loaded


def load_or_extract_feature_bank(samples, required_modes, fingerprint, cache_dir):
    """Load a matching bank or perform one fresh multi-view extraction."""

    if USE_FEATURE_CACHE and not FORCE_REBUILD_FEATURE_CACHE:
        bank = load_feature_bank(cache_dir, fingerprint, required_modes)
        if bank is not None:
            return bank, "HIT"

    print(
        "[FEATURE BANK] Cache miss or forced rebuild; neural-network inference "
        "will run now.",
        flush=True,
    )
    dataset = MRIDataset(samples, transform)

    need_monai = any(
        mode in {"monai_roi", "outside_heart"}
        for mode in required_modes
    )
    monai_segmenter = build_monai_segmenter() if need_monai else None

    feature_extractor = FeatureExtractor().to(DEVICE)
    feature_extractor.eval()
    feature_extractor.requires_grad_(False)

    runtime_cache_dir = (
        cache_dir
        if USE_FEATURE_CACHE
        else OUTPUT_DIR / "_runtime_feature_bank"
    )
    bank = extract_feature_bank(
        dataset=dataset,
        monai_segmenter=monai_segmenter,
        feature_extractor=feature_extractor,
        required_modes=required_modes,
        cache_dir=runtime_cache_dir,
        fingerprint=fingerprint,
        debug=DEBUG_VISUALIZATION,
    )

    del feature_extractor
    del monai_segmenter
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return bank, "MISS"

# =============================
# PIPELINE STEP 6
# DUPLICATE, PROVENANCE AND MONAI-QC AUDITS
# =============================


class UnionFind:
    """Small deterministic disjoint-set structure for patient components."""

    def __init__(self, values):
        self.parent = {str(value): str(value) for value in values}
        self.rank = {str(value): 0 for value in values}

    def find(self, value):
        value = str(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return

        rank_first = self.rank[root_first]
        rank_second = self.rank[root_second]
        if rank_first < rank_second:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if rank_first == rank_second:
            self.rank[root_first] += 1


def audit_exact_decoded_pixel_duplicates(
    output_path,
    samples,
    decoded_pixel_hashes,
):
    """Write exact duplicate groups and return patient edges for safe splitting."""

    if len(samples) != len(decoded_pixel_hashes):
        raise ValueError("Exact duplicate hashes do not align with samples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hash_to_indices = defaultdict(list)
    for index, pixel_hash in enumerate(decoded_pixel_hashes):
        hash_to_indices[str(pixel_hash)].append(index)

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
    rows = []
    patient_edges = set()
    affected_patients = set()
    cross_patient_groups = 0
    cross_label_groups = 0

    for group_number, (pixel_hash, indices) in enumerate(
        duplicate_groups,
        start=1,
    ):
        patients = sorted({str(samples[index][2]) for index in indices})
        labels = sorted({int(samples[index][1]) for index in indices})
        cross_patient = len(patients) > 1
        cross_label = len(labels) > 1

        if cross_patient:
            cross_patient_groups += 1
            affected_patients.update(patients)
            for first, second in combinations(patients, 2):
                patient_edges.add(tuple(sorted((first, second))))
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
        "definition": "exact decoded uint8 grayscale pixels plus native shape",
        "duplicate_groups": int(len(duplicate_groups)),
        "duplicate_images": int(sum(len(indices) for _, indices in duplicate_groups)),
        "cross_patient_duplicate_groups": int(cross_patient_groups),
        "cross_label_duplicate_groups": int(cross_label_groups),
        "patient_edges": int(len(patient_edges)),
        "affected_cross_patient_ids": sorted(affected_patients),
        "csv_path": str(output_path),
    }

    print(
        "[EXACT DUPLICATES] "
        f"groups={summary['duplicate_groups']}, "
        f"cross_patient_groups={cross_patient_groups}, "
        f"cross_label_groups={cross_label_groups}, "
        f"patient_edges={len(patient_edges)}",
        flush=True,
    )

    if cross_patient_groups:
        print(
            "[WARNING] Exact visual content occurs across Directory_* patients. "
            "Those patients will be kept in the same outer/inner fold when "
            "GROUP_SPLITS_BY_EXACT_DUPLICATES=True.",
            flush=True,
        )
    if cross_label_groups:
        print(
            "[WARNING] At least one exact duplicate group spans Normal and Sick. "
            "This requires dataset-level investigation before publication.",
            flush=True,
        )

    if cross_patient_groups and FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES:
        raise RuntimeError(
            f"Cross-patient exact duplicates found; inspect {output_path}."
        )
    if cross_label_groups and FAIL_ON_CROSS_LABEL_EXACT_DUPLICATES:
        raise RuntimeError(
            f"Cross-label exact duplicates found; inspect {output_path}."
        )

    return summary, patient_edges


def _phash_hamming_distance(first_hash, second_hash):
    return (int(str(first_hash), 16) ^ int(str(second_hash), 16)).bit_count()


def audit_perceptual_near_duplicate_candidates(
    output_path,
    samples,
    perceptual_hashes,
    decoded_pixel_hashes,
):
    """Find cross-patient low-Hamming pHash candidates through exact-chunk LSH.

    With four 16-bit chunks and Hamming threshold 3, any qualifying 64-bit pair
    must share at least one complete chunk. This creates a manageable candidate
    set without an O(N^2) all-pairs comparison. pHash similarity is only a
    screening signal; the CSV requires visual/manual confirmation.
    """

    if not (
        len(samples) == len(perceptual_hashes) == len(decoded_pixel_hashes)
    ):
        raise ValueError("Perceptual audit arrays do not align with samples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hash_values = [int(str(value), 16) for value in perceptual_hashes]
    chunk_mask = (1 << PHASH_BUCKET_BITS) - 1
    chunk_count = 64 // PHASH_BUCKET_BITS

    buckets = defaultdict(list)
    for index, value in enumerate(hash_values):
        for chunk_index in range(chunk_count):
            chunk = (value >> (chunk_index * PHASH_BUCKET_BITS)) & chunk_mask
            buckets[(chunk_index, chunk)].append(index)

    seen_image_pairs = set()
    patient_pair_data = {}
    skipped_large_buckets = 0
    examined_image_pairs = 0
    truncated = False

    for bucket_key in sorted(buckets):
        indices = buckets[bucket_key]
        if len(indices) > PHASH_MAX_BUCKET_SIZE:
            skipped_large_buckets += 1
            continue

        for first_position in range(len(indices)):
            first_index = indices[first_position]
            first_patient = str(samples[first_index][2])

            for second_position in range(first_position + 1, len(indices)):
                second_index = indices[second_position]
                second_patient = str(samples[second_index][2])
                if first_patient == second_patient:
                    continue

                image_pair = (
                    min(first_index, second_index),
                    max(first_index, second_index),
                )
                if image_pair in seen_image_pairs:
                    continue
                seen_image_pairs.add(image_pair)
                examined_image_pairs += 1

                if examined_image_pairs > PHASH_MAX_IMAGE_PAIR_CANDIDATES:
                    truncated = True
                    break

                distance = (
                    hash_values[first_index] ^ hash_values[second_index]
                ).bit_count()
                if distance > PHASH_HAMMING_THRESHOLD:
                    continue

                patient_pair = tuple(sorted((first_patient, second_patient)))
                first_label = int(samples[first_index][1])
                second_label = int(samples[second_index][1])
                exact_pixels = (
                    str(decoded_pixel_hashes[first_index])
                    == str(decoded_pixel_hashes[second_index])
                )

                record = patient_pair_data.get(patient_pair)
                if record is None:
                    record = {
                        "patient_id_a": patient_pair[0],
                        "patient_id_b": patient_pair[1],
                        "label_a": (
                            first_label
                            if first_patient == patient_pair[0]
                            else second_label
                        ),
                        "label_b": (
                            second_label
                            if second_patient == patient_pair[1]
                            else first_label
                        ),
                        "cross_label": int(first_label != second_label),
                        "minimum_phash_hamming_distance": int(distance),
                        "candidate_image_pairs": 0,
                        "contains_exact_pixel_pair": int(exact_pixels),
                        "example_image_a": str(samples[first_index][0]),
                        "example_image_b": str(samples[second_index][0]),
                        "example_phash_a": str(perceptual_hashes[first_index]),
                        "example_phash_b": str(perceptual_hashes[second_index]),
                    }
                    patient_pair_data[patient_pair] = record

                record["candidate_image_pairs"] += 1
                record["contains_exact_pixel_pair"] = int(
                    bool(record["contains_exact_pixel_pair"]) or exact_pixels
                )
                if distance < record["minimum_phash_hamming_distance"]:
                    record["minimum_phash_hamming_distance"] = int(distance)
                    record["example_image_a"] = str(samples[first_index][0])
                    record["example_image_b"] = str(samples[second_index][0])
                    record["example_phash_a"] = str(
                        perceptual_hashes[first_index]
                    )
                    record["example_phash_b"] = str(
                        perceptual_hashes[second_index]
                    )

            if truncated:
                break
        if truncated:
            break

    rows = sorted(
        patient_pair_data.values(),
        key=lambda row: (
            row["minimum_phash_hamming_distance"],
            -row["candidate_image_pairs"],
            row["patient_id_a"],
            row["patient_id_b"],
        ),
    )

    fieldnames = [
        "patient_id_a",
        "patient_id_b",
        "label_a",
        "label_b",
        "cross_label",
        "minimum_phash_hamming_distance",
        "candidate_image_pairs",
        "contains_exact_pixel_pair",
        "example_image_a",
        "example_image_b",
        "example_phash_a",
        "example_phash_b",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    candidate_edges = {
        tuple(sorted((row["patient_id_a"], row["patient_id_b"])))
        for row in rows
    }
    summary = {
        "enabled": True,
        "definition": (
            "64-bit DCT pHash screening; cross-patient pairs with Hamming "
            f"distance <= {PHASH_HAMMING_THRESHOLD}"
        ),
        "patient_pair_candidates": int(len(rows)),
        "cross_label_patient_pair_candidates": int(
            sum(row["cross_label"] for row in rows)
        ),
        "examined_unique_image_pairs": int(
            min(examined_image_pairs, PHASH_MAX_IMAGE_PAIR_CANDIDATES)
        ),
        "skipped_large_lsh_buckets": int(skipped_large_buckets),
        "candidate_search_truncated": bool(truncated),
        "csv_path": str(output_path),
    }

    print(
        "[PERCEPTUAL DUPLICATES] "
        f"patient_pair_candidates={len(rows)}, "
        f"cross_label={summary['cross_label_patient_pair_candidates']}, "
        f"examined_image_pairs={summary['examined_unique_image_pairs']}, "
        f"skipped_large_buckets={skipped_large_buckets}, "
        f"truncated={truncated}",
        flush=True,
    )
    if rows:
        print(
            "[WARNING] Perceptual-hash matches are screening candidates, not "
            "confirmed duplicates. Review the saved example pairs manually.",
            flush=True,
        )

    return summary, candidate_edges


def build_patient_label_table(labels, patient_ids):
    """Return sorted unique patient IDs and one consistent label per patient."""

    patient_to_label = {}
    for label, patient_id in zip(labels, patient_ids):
        patient_id = str(patient_id)
        label = int(label)
        previous = patient_to_label.get(patient_id)
        if previous is not None and previous != label:
            raise RuntimeError(
                f"Patient {patient_id} appears with conflicting labels."
            )
        patient_to_label[patient_id] = label

    ordered_ids = np.asarray(sorted(patient_to_label))
    ordered_labels = np.asarray(
        [patient_to_label[patient_id] for patient_id in ordered_ids],
        dtype=np.int64,
    )
    return ordered_ids, ordered_labels


def build_duplicate_component_map(patient_ids, exact_edges, phash_edges):
    """Create deterministic patient components used as split groups."""

    patient_ids = [str(value) for value in patient_ids]
    union_find = UnionFind(patient_ids)

    if GROUP_SPLITS_BY_EXACT_DUPLICATES:
        for first, second in exact_edges:
            union_find.union(first, second)

    if GROUP_SPLITS_BY_PHASH_CANDIDATES:
        for first, second in phash_edges:
            union_find.union(first, second)

    root_to_members = defaultdict(list)
    for patient_id in patient_ids:
        root_to_members[union_find.find(patient_id)].append(patient_id)

    patient_to_component = {}
    for members in root_to_members.values():
        members = sorted(members)
        digest = hashlib.sha256("|".join(members).encode("utf-8")).hexdigest()[:10]
        component_id = f"COMP_{digest}"
        for patient_id in members:
            patient_to_component[patient_id] = component_id

    return patient_to_component


def assign_stratified_patient_folds(
    patient_ids,
    labels,
    group_ids,
    n_splits,
    random_state,
):
    """Assign deterministic patient folds, respecting duplicate components."""

    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels, dtype=np.int64)
    group_ids = np.asarray(group_ids)

    if not (len(patient_ids) == len(labels) == len(group_ids)):
        raise ValueError("Patient IDs, labels and group IDs must align.")
    if np.unique(labels).size != 2:
        raise RuntimeError("Fold assignment requires both classes.")
    class_counts = np.bincount(labels, minlength=2)
    if int(class_counts.min()) < n_splits:
        raise RuntimeError(
            f"{n_splits}-fold CV needs at least {n_splits} patients in each "
            f"class; found Normal={class_counts[0]}, Sick={class_counts[1]}."
        )
    if len(np.unique(group_ids)) < n_splits:
        raise RuntimeError(
            f"Only {len(np.unique(group_ids))} duplicate components are "
            f"available for {n_splits} folds."
        )

    all_groups_unique = len(np.unique(group_ids)) == len(group_ids)

    for attempt in range(100):
        attempt_seed = int(random_state + attempt)
        fold_numbers = np.zeros(len(patient_ids), dtype=np.int64)

        if all_groups_unique:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=attempt_seed,
            )
            split_iterator = splitter.split(patient_ids, labels)
        else:
            try:
                from sklearn.model_selection import StratifiedGroupKFold
            except ImportError as exc:
                raise RuntimeError(
                    "Cross-patient duplicate components require "
                    "StratifiedGroupKFold. Install scikit-learn>=1.1."
                ) from exc

            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=attempt_seed,
            )
            split_iterator = splitter.split(
                np.zeros((len(patient_ids), 1), dtype=np.float32),
                labels,
                groups=group_ids,
            )

        for fold_index, (_, valid_indices) in enumerate(
            split_iterator,
            start=1,
        ):
            fold_numbers[valid_indices] = fold_index

        valid_assignment = True
        for fold_index in range(1, n_splits + 1):
            fold_labels = labels[fold_numbers == fold_index]
            if len(fold_labels) == 0 or np.unique(fold_labels).size != 2:
                valid_assignment = False
                break

        if valid_assignment:
            return fold_numbers

    raise RuntimeError(
        "Could not construct duplicate-aware stratified folds containing both "
        "classes after 100 deterministic attempts. Reduce N_SPLITS or inspect "
        "large/cross-label duplicate components."
    )


def build_patient_fold_manifest(
    labels,
    patient_ids,
    exact_edges,
    phash_edges,
):
    """Build the single authoritative outer-fold assignment for every experiment."""

    ordered_ids, ordered_labels = build_patient_label_table(labels, patient_ids)
    patient_to_component = build_duplicate_component_map(
        ordered_ids,
        exact_edges,
        phash_edges,
    )
    group_ids = np.asarray(
        [patient_to_component[patient_id] for patient_id in ordered_ids]
    )

    folds = assign_stratified_patient_folds(
        ordered_ids,
        ordered_labels,
        group_ids,
        N_SPLITS,
        CV_RANDOM_STATE,
    )

    component_sizes = {
        component_id: int(np.sum(group_ids == component_id))
        for component_id in np.unique(group_ids)
    }
    rows = []
    for patient_id, label, component_id, outer_fold in zip(
        ordered_ids,
        ordered_labels,
        group_ids,
        folds,
    ):
        rows.append(
            {
                "patient_id": str(patient_id),
                "true_label": int(label),
                "duplicate_component_id": str(component_id),
                "duplicate_component_size": component_sizes[str(component_id)],
                "outer_fold": int(outer_fold),
            }
        )

    for fold_index in range(1, N_SPLITS + 1):
        fold_rows = [row for row in rows if row["outer_fold"] == fold_index]
        print(
            f"[FOLD MANIFEST] fold={fold_index}: patients={len(fold_rows)}, "
            f"Normal={sum(row['true_label'] == 0 for row in fold_rows)}, "
            f"Sick={sum(row['true_label'] == 1 for row in fold_rows)}, "
            f"components={len(set(row['duplicate_component_id'] for row in fold_rows))}",
            flush=True,
        )

    return rows


def write_patient_fold_manifest(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_cohort_manifest(output_path, samples, bank, fold_manifest_rows):
    """Write one auditable row per image including folds and provenance fields."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fold_by_patient = {
        row["patient_id"]: row["outer_fold"] for row in fold_manifest_rows
    }
    component_by_patient = {
        row["patient_id"]: row["duplicate_component_id"]
        for row in fold_manifest_rows
    }

    fieldnames = [
        "sample_index",
        "image_path",
        "true_label",
        "patient_id",
        "series_id",
        "outer_fold",
        "duplicate_component_id",
        "decoded_pixel_sha256",
        "perceptual_hash",
        "monai_gate_valid",
        "monai_area_ratio",
        "monai_peak_probability",
        "monai_mean_foreground_probability",
        "roi_slice_std_score",
        *PROVENANCE_FEATURE_NAMES,
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, sample in enumerate(samples):
            image_path, label, patient_id, series_id = sample
            row = {
                "sample_index": index,
                "image_path": str(image_path),
                "true_label": int(label),
                "patient_id": str(patient_id),
                "series_id": str(series_id),
                "outer_fold": int(fold_by_patient[str(patient_id)]),
                "duplicate_component_id": component_by_patient[str(patient_id)],
                "decoded_pixel_sha256": str(bank["decoded_pixel_hashes"][index]),
                "perceptual_hash": str(bank["perceptual_hashes"][index]),
                "monai_gate_valid": int(bool(bank["monai_valid"][index])),
                "monai_area_ratio": float(bank["area_ratios"][index]),
                "monai_peak_probability": float(
                    bank["peak_probabilities"][index]
                ),
                "monai_mean_foreground_probability": float(
                    bank["mean_foreground_probabilities"][index]
                ),
                "roi_slice_std_score": float(bank["roi_slice_scores"][index]),
            }
            for feature_name, value in zip(
                PROVENANCE_FEATURE_NAMES,
                bank["provenance_features"][index],
            ):
                row[feature_name] = float(value)
            writer.writerow(row)


def _patient_indices(patient_ids):
    mapping = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        mapping[str(patient_id)].append(index)
    return mapping


def aggregate_patient_provenance_features(bank):
    """Aggregate native export/style features to one row per patient."""

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"])
    series_ids = np.asarray(bank["series_ids"])
    provenance = np.asarray(bank["provenance_features"], dtype=np.float32)

    mapping = _patient_indices(patient_ids)
    ordered_patients = np.asarray(sorted(mapping))
    patient_labels = []
    rows = []

    output_names = ["n_slices", "n_series_proxies", "mean_series_length", "max_series_length"]
    for feature_name in PROVENANCE_FEATURE_NAMES:
        output_names.extend(
            [
                f"mean__{feature_name}",
                f"median__{feature_name}",
                f"std__{feature_name}",
            ]
        )

    for patient_id in ordered_patients:
        indices = np.asarray(mapping[str(patient_id)], dtype=np.int64)
        label_values = np.unique(labels[indices])
        if len(label_values) != 1:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent provenance labels."
            )
        patient_labels.append(int(label_values[0]))

        unique_series, series_counts = np.unique(
            series_ids[indices],
            return_counts=True,
        )
        del unique_series
        feature_values = provenance[indices]
        row = [
            float(len(indices)),
            float(len(series_counts)),
            float(np.mean(series_counts)),
            float(np.max(series_counts)),
        ]
        for feature_index in range(feature_values.shape[1]):
            column = feature_values[:, feature_index]
            row.extend(
                [
                    float(np.mean(column)),
                    float(np.median(column)),
                    float(np.std(column)),
                ]
            )
        rows.append(row)

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(patient_labels, dtype=np.int64)
    return X, y, ordered_patients, tuple(output_names)


def aggregate_patient_monai_qc_features(bank):
    """Aggregate MONAI gate diagnostics to one label-free patient feature row."""

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"])
    valid = np.asarray(bank["monai_valid"], dtype=bool)
    area = np.asarray(bank["area_ratios"], dtype=np.float32)
    peak = np.asarray(bank["peak_probabilities"], dtype=np.float32)
    foreground = np.asarray(
        bank["mean_foreground_probabilities"],
        dtype=np.float32,
    )

    mapping = _patient_indices(patient_ids)
    ordered_patients = np.asarray(sorted(mapping))
    patient_labels = []
    rows = []
    feature_names = (
        "plausible_mask_rate",
        "median_area_ratio",
        "std_area_ratio",
        "median_peak_probability",
        "std_peak_probability",
        "median_mean_foreground_probability",
        "std_mean_foreground_probability",
    )

    for patient_id in ordered_patients:
        indices = np.asarray(mapping[str(patient_id)], dtype=np.int64)
        label_values = np.unique(labels[indices])
        if len(label_values) != 1:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent MONAI-QC labels."
            )
        patient_labels.append(int(label_values[0]))
        rows.append(
            [
                float(np.mean(valid[indices])),
                float(np.nanmedian(area[indices])),
                float(np.nanstd(area[indices])),
                float(np.nanmedian(peak[indices])),
                float(np.nanstd(peak[indices])),
                float(np.nanmedian(foreground[indices])),
                float(np.nanstd(foreground[indices])),
            ]
        )

    X = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(X)):
        raise RuntimeError(
            "MONAI-QC patient features contain non-finite values."
        )
    y = np.asarray(patient_labels, dtype=np.int64)
    return X, y, ordered_patients, feature_names


def bootstrap_difference_in_means(values, labels, n_bootstrap, random_state):
    """Patient-level stratified bootstrap CI for class-1 minus class-0 mean."""

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    if len(class0) == 0 or len(class1) == 0:
        raise ValueError("Both classes are required for a mean difference.")

    rng = np.random.default_rng(random_state)
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled0 = rng.choice(class0, size=len(class0), replace=True)
        sampled1 = rng.choice(class1, size=len(class1), replace=True)
        differences[index] = (
            float(np.mean(values[sampled1])) - float(np.mean(values[sampled0]))
        )

    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    return (
        float(np.quantile(differences, alpha / 2.0)),
        float(np.quantile(differences, 1.0 - alpha / 2.0)),
    )


def write_monai_qc_outputs(output_dir, bank):
    """Write patient-level gate statistics and class-specific comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    X_qc, y_qc, patient_ids, feature_names = (
        aggregate_patient_monai_qc_features(bank)
    )

    rows = []
    for patient_id, label, values in zip(patient_ids, y_qc, X_qc):
        row = {"patient_id": str(patient_id), "true_label": int(label)}
        for feature_name, value in zip(feature_names, values):
            row[feature_name] = float(value)
        rows.append(row)

    patient_csv = output_dir / "monai_qc_by_patient.csv"
    with open(patient_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gate_rates = X_qc[:, feature_names.index("plausible_mask_rate")]
    normal_mean = float(np.mean(gate_rates[y_qc == 0]))
    sick_mean = float(np.mean(gate_rates[y_qc == 1]))
    difference = sick_mean - normal_mean
    ci_lower, ci_upper = bootstrap_difference_in_means(
        gate_rates,
        y_qc,
        BOOTSTRAP_REPLICATES,
        RANDOM_SEED + 701,
    )

    slice_labels = np.asarray(bank["labels"], dtype=np.int64)
    slice_valid = np.asarray(bank["monai_valid"], dtype=bool)
    slice_normal_rate = float(np.mean(slice_valid[slice_labels == 0]))
    slice_sick_rate = float(np.mean(slice_valid[slice_labels == 1]))

    comparison = {
        "patient_level_normal_mean_gate_rate": normal_mean,
        "patient_level_sick_mean_gate_rate": sick_mean,
        "patient_level_sick_minus_normal_difference": difference,
        "patient_level_difference_ci": [ci_lower, ci_upper],
        "slice_level_normal_gate_rate_descriptive": slice_normal_rate,
        "slice_level_sick_gate_rate_descriptive": slice_sick_rate,
        "warning_threshold_absolute_difference": (
            MONAI_GATE_RATE_DIFFERENCE_WARNING
        ),
        "warning_triggered": bool(
            abs(difference) >= MONAI_GATE_RATE_DIFFERENCE_WARNING
        ),
        "interpretation": (
            "A large class difference may reflect sequence/protocol/export "
            "confounding; it is not segmentation-accuracy evidence."
        ),
    }
    (output_dir / "monai_gate_class_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "[MONAI QC] Patient-level mean plausible-mask rate: "
        f"Normal={normal_mean:.3f}, Sick={sick_mean:.3f}, "
        f"difference={difference:+.3f}, "
        f"95% CI=[{ci_lower:+.3f}, {ci_upper:+.3f}]",
        flush=True,
    )
    if comparison["warning_triggered"]:
        print(
            "[WARNING] The class-specific MONAI gate-rate difference exceeds "
            f"{MONAI_GATE_RATE_DIFFERENCE_WARNING:.0%}. Protocol/export "
            "confounding must be investigated.",
            flush=True,
        )

    return comparison, (X_qc, y_qc, patient_ids, feature_names)


def write_patient_tabular_features(output_path, X, y, patient_ids, feature_names):
    """Write one patient row for a provenance or MONAI-QC control matrix."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "true_label", *feature_names]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id, label, values in zip(patient_ids, y, X):
            row = {"patient_id": str(patient_id), "true_label": int(label)}
            row.update(
                {
                    feature_name: float(value)
                    for feature_name, value in zip(feature_names, values)
                }
            )
            writer.writerow(row)

# =============================
# PIPELINE STEP 7
# WEIGHTING, POOLING AND LEGACY FUSION
# =============================


def build_slice_weights(roi_slice_scores, series_ids, weighting_mode):
    """Create equal or series-local heuristic weights without rerunning images."""

    series_ids = np.asarray(series_ids)
    if weighting_mode == "equal":
        return np.ones(len(series_ids), dtype=np.float64)
    if weighting_mode != "quality":
        raise ValueError(f"Unsupported slice weighting mode: {weighting_mode!r}.")

    scores = np.asarray(roi_slice_scores, dtype=np.float64)
    if len(scores) != len(series_ids):
        raise ValueError("Quality scores and series IDs must align.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Quality weighting requires finite ROI slice scores.")

    weights = np.zeros(len(scores), dtype=np.float64)
    for series_id in np.unique(series_ids):
        indices = np.flatnonzero(series_ids == series_id)
        weights[indices] = _normalize_quality_weights(scores[indices])

    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise RuntimeError("Quality weighting produced invalid slice weights.")
    return weights


def aggregate_patient_embeddings(
    features,
    labels,
    patient_ids,
    series_ids,
    slice_weights,
    pooling_strategy,
):
    """Pool slice embeddings to one vector per Directory_* patient."""

    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    series_ids = np.asarray(series_ids)
    slice_weights = np.asarray(slice_weights, dtype=np.float64)

    if not (
        len(features)
        == len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(slice_weights)
    ):
        raise ValueError("Embedding-pooling arrays must have equal lengths.")
    if pooling_strategy not in {"hierarchical", "flat"}:
        raise ValueError(
            f"Unsupported patient embedding pooling: {pooling_strategy!r}."
        )
    if np.any(slice_weights < 0) or not np.all(np.isfinite(slice_weights)):
        raise ValueError("Slice weights must be finite and non-negative.")

    patient_to_label = {}
    patient_to_indices = defaultdict(list)
    patient_to_series_indices = defaultdict(lambda: defaultdict(list))

    for index, (label, patient_id, series_id) in enumerate(
        zip(labels, patient_ids, series_ids)
    ):
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)
        previous = patient_to_label.get(patient_id)
        if previous is not None and previous != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent labels during pooling."
            )
        patient_to_label[patient_id] = label
        patient_to_indices[patient_id].append(index)
        patient_to_series_indices[patient_id][series_id].append(index)

    ordered_patients = np.asarray(sorted(patient_to_label))
    pooled_features = []
    pooled_labels = []

    for patient_id in ordered_patients:
        if pooling_strategy == "flat":
            indices = np.asarray(patient_to_indices[patient_id], dtype=np.int64)
            weights = slice_weights[indices]
            if not np.any(weights > 0):
                raise RuntimeError(
                    f"Patient {patient_id} has no positive flat-pooling weight."
                )
            patient_embedding = np.average(
                features[indices],
                axis=0,
                weights=weights,
            )
        else:
            series_embeddings = []
            for series_id in sorted(patient_to_series_indices[patient_id]):
                indices = np.asarray(
                    patient_to_series_indices[patient_id][series_id],
                    dtype=np.int64,
                )
                weights = slice_weights[indices]
                if not np.any(weights > 0):
                    raise RuntimeError(
                        f"Series proxy {series_id} has no positive pooling weight."
                    )
                series_embeddings.append(
                    np.average(
                        features[indices],
                        axis=0,
                        weights=weights,
                    )
                )
            patient_embedding = np.mean(
                np.stack(series_embeddings, axis=0),
                axis=0,
            )

        pooled_features.append(
            np.asarray(patient_embedding, dtype=np.float32)
        )
        pooled_labels.append(patient_to_label[patient_id])

    X_patient = np.stack(pooled_features, axis=0)
    y_patient = np.asarray(pooled_labels, dtype=np.int64)
    if not np.all(np.isfinite(X_patient)):
        raise RuntimeError("Pooled patient embeddings contain non-finite values.")
    return X_patient, y_patient, ordered_patients


def compute_balanced_patient_weights(labels):
    """Give Normal and Sick equal supervised mass at patient level."""

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if int(counts.min()) <= 0:
        raise RuntimeError("A training partition must contain both classes.")
    n_patients = len(labels)
    return np.asarray(
        [n_patients / (2.0 * counts[int(label)]) for label in labels],
        dtype=np.float64,
    )


def compute_hierarchical_training_weights(
    labels,
    patient_ids,
    series_ids,
    quality_weights,
):
    """Control class, patient, series and slice influence in legacy fitting."""

    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    series_ids = np.asarray(series_ids)
    quality_weights = np.asarray(quality_weights, dtype=np.float64)

    if not (
        len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(quality_weights)
    ):
        raise ValueError("Legacy training arrays must have equal lengths.")
    if np.any(quality_weights < 0) or not np.all(np.isfinite(quality_weights)):
        raise ValueError("Legacy quality weights are invalid.")

    patient_to_label = {}
    patient_to_series = defaultdict(set)
    series_quality_sum = defaultdict(float)

    for label, patient_id, series_id, quality_weight in zip(
        labels,
        patient_ids,
        series_ids,
        quality_weights,
    ):
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)
        previous = patient_to_label.get(patient_id)
        if previous is not None and previous != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent legacy labels."
            )
        patient_to_label[patient_id] = label
        patient_to_series[patient_id].add(series_id)
        series_quality_sum[series_id] += float(quality_weight)

    class_patient_counts = np.bincount(
        np.asarray(list(patient_to_label.values()), dtype=np.int64),
        minlength=2,
    )
    if int(class_patient_counts.min()) <= 0:
        raise RuntimeError("Legacy training fold must contain both classes.")

    weights = np.empty(len(labels), dtype=np.float64)
    for index, (label, patient_id, series_id, quality_weight) in enumerate(
        zip(labels, patient_ids, series_ids, quality_weights)
    ):
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)
        weights[index] = (
            float(quality_weight)
            / max(series_quality_sum[series_id], 1e-12)
            / len(patient_to_series[patient_id])
            / class_patient_counts[label]
        )

    n_patients = len(patient_to_label)
    weights *= n_patients / max(float(weights.sum()), 1e-12)
    return weights


def weighted_log_odds_fusion(probabilities, weights=None):
    """Fuse finite probabilities through a weighted mean of clipped logits."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.size == 0:
        raise ValueError("Cannot fuse an empty probability collection.")
    if weights is None:
        weights = np.ones_like(probabilities, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if probabilities.shape != weights.shape:
        raise ValueError("Probabilities and fusion weights must share a shape.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Fusion probabilities must be finite.")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Fusion weights must be non-negative and not all zero.")

    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(probabilities / (1.0 - probabilities))
    fused_logit = float(np.average(logits, weights=weights))
    return float(1.0 / (1.0 + np.exp(-fused_logit)))


def fuse_probabilities(probabilities, weights, method):
    """Apply the declared mean-probability or log-odds fusion rule."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if method == "log_odds":
        return weighted_log_odds_fusion(probabilities, weights)
    if method == "mean_probability":
        if probabilities.size == 0:
            raise ValueError("Cannot fuse an empty probability collection.")
        if probabilities.shape != weights.shape:
            raise ValueError("Mean-fusion arrays must share a shape.")
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("Mean-fusion weights are invalid.")
        return float(np.average(probabilities, weights=weights))
    raise ValueError(f"Unknown probability fusion method: {method!r}.")


def aggregate_legacy_slice_probabilities(
    slice_probabilities,
    labels,
    patient_ids,
    series_ids,
    quality_weights,
    fusion_method,
):
    """Fuse validation slice scores to series proxies and then patients."""

    if not (
        len(slice_probabilities)
        == len(labels)
        == len(patient_ids)
        == len(series_ids)
        == len(quality_weights)
    ):
        raise ValueError("Legacy aggregation arrays must have equal lengths.")

    nested = defaultdict(lambda: defaultdict(lambda: {"p": [], "w": []}))
    patient_to_label = {}
    for probability, label, patient_id, series_id, weight in zip(
        slice_probabilities,
        labels,
        patient_ids,
        series_ids,
        quality_weights,
    ):
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)
        previous = patient_to_label.get(patient_id)
        if previous is not None and previous != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent aggregation labels."
            )
        patient_to_label[patient_id] = label
        nested[patient_id][series_id]["p"].append(float(probability))
        nested[patient_id][series_id]["w"].append(float(weight))

    ordered_patients = np.asarray(sorted(nested))
    patient_scores = []
    patient_labels = []
    for patient_id in ordered_patients:
        series_scores = []
        for series_id in sorted(nested[patient_id]):
            series_data = nested[patient_id][series_id]
            series_scores.append(
                fuse_probabilities(
                    series_data["p"],
                    series_data["w"],
                    fusion_method,
                )
            )
        patient_scores.append(
            fuse_probabilities(
                series_scores,
                np.ones(len(series_scores), dtype=np.float64),
                fusion_method,
            )
        )
        patient_labels.append(patient_to_label[patient_id])

    return (
        np.asarray(patient_scores, dtype=np.float64),
        np.asarray(patient_labels, dtype=np.int64),
        ordered_patients,
    )

# =============================
# PIPELINE STEP 8
# NESTED PATIENT-LEVEL CROSS-VALIDATION
# =============================


def prepare_experiment_data(experiment, bank, tabular_feature_sets):
    """Build the fixed data representation used by one experiment."""

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"])
    series_ids = np.asarray(bank["series_ids"])

    if experiment.strategy == "patient_tabular":
        X, y, patients, feature_names = tabular_feature_sets[
            experiment.feature_mode
        ]
        return {
            "unit": "patient",
            "X": np.asarray(X, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int64),
            "patient_ids": np.asarray(patients),
            "feature_names": tuple(feature_names),
        }

    features = bank["features"][experiment.feature_mode]
    slice_weights = build_slice_weights(
        bank["roi_slice_scores"],
        series_ids,
        experiment.weighting_mode,
    )

    if experiment.strategy == "patient_embedding":
        X, y, patients = aggregate_patient_embeddings(
            features=features,
            labels=labels,
            patient_ids=patient_ids,
            series_ids=series_ids,
            slice_weights=slice_weights,
            pooling_strategy=experiment.pooling_strategy,
        )
        return {
            "unit": "patient",
            "X": X,
            "y": y,
            "patient_ids": patients,
            "feature_names": tuple(
                f"embedding_{index:04d}"
                for index in range(X.shape[1])
            ),
        }

    if experiment.strategy == "slice_probability_fusion":
        return {
            "unit": "slice",
            "X": features,
            "y": labels,
            "patient_ids": patient_ids,
            "series_ids": series_ids,
            "slice_weights": slice_weights,
        }

    raise ValueError(
        f"Unsupported experiment strategy: {experiment.strategy!r}."
    )


def build_patient_classifier(experiment, c_value):
    """Create a fresh scaler/PCA/linear classifier for patient rows."""

    steps = [("scaler", StandardScaler())]
    if experiment.use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=PATIENT_PCA_EXPLAINED_VARIANCE,
                    svd_solver="full",
                ),
            )
        )

    if experiment.classifier_type == "logistic_regression":
        classifier = LogisticRegression(
            C=float(c_value),
            max_iter=LOGISTIC_MAX_ITER,
            solver="liblinear",
            random_state=RANDOM_SEED,
        )
    elif experiment.classifier_type == "linear_svm":
        classifier = LinearSVC(
            C=float(c_value),
            max_iter=SVM_MAX_ITER,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(
            f"Unknown classifier type: {experiment.classifier_type!r}."
        )

    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)


def fit_patient_classifier(model, X_train, y_train):
    """Fit patient-row preprocessing and classifier with balanced labels."""

    y_train = np.asarray(y_train, dtype=np.int64)
    patient_weights = compute_balanced_patient_weights(y_train)
    model.fit(
        X_train,
        y_train,
        scaler__sample_weight=np.ones(len(y_train), dtype=np.float64),
        classifier__sample_weight=patient_weights,
    )
    return model


def patient_model_raw_scores(model, experiment, X_valid):
    """Return probabilities for LR and margins for Linear SVM."""

    if experiment.classifier_type == "logistic_regression":
        return np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)
    return np.asarray(model.decision_function(X_valid), dtype=np.float64)


def build_legacy_slice_classifier(c_value):
    """Create the reviewed legacy slice-level Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=LOGISTIC_MAX_ITER,
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def fit_predict_raw_for_patient_sets(
    experiment,
    prepared,
    train_patients,
    valid_patients,
    c_value,
):
    """Fit one fold-local model and return held-out patient scores."""

    train_patients = set(map(str, train_patients))
    valid_patients = set(map(str, valid_patients))
    overlap = train_patients.intersection(valid_patients)
    if overlap:
        raise RuntimeError(f"Train/validation patient overlap: {sorted(overlap)}")

    if prepared["unit"] == "patient":
        patient_ids = np.asarray(prepared["patient_ids"])
        train_mask = np.isin(patient_ids, list(train_patients))
        valid_mask = np.isin(patient_ids, list(valid_patients))

        if not np.any(train_mask) or not np.any(valid_mask):
            raise RuntimeError("A patient-level fold side is empty.")

        model = build_patient_classifier(experiment, c_value)
        fit_patient_classifier(
            model,
            prepared["X"][train_mask],
            prepared["y"][train_mask],
        )
        scores = patient_model_raw_scores(
            model,
            experiment,
            prepared["X"][valid_mask],
        )
        return (
            scores,
            np.asarray(prepared["y"][valid_mask], dtype=np.int64),
            np.asarray(patient_ids[valid_mask]),
        )

    slice_patient_ids = np.asarray(prepared["patient_ids"])
    train_mask = np.isin(slice_patient_ids, list(train_patients))
    valid_mask = np.isin(slice_patient_ids, list(valid_patients))
    if not np.any(train_mask) or not np.any(valid_mask):
        raise RuntimeError("A legacy slice-level fold side is empty.")

    training_weights = compute_hierarchical_training_weights(
        prepared["y"][train_mask],
        prepared["patient_ids"][train_mask],
        prepared["series_ids"][train_mask],
        prepared["slice_weights"][train_mask],
    )
    model = build_legacy_slice_classifier(c_value)
    model.fit(
        prepared["X"][train_mask],
        prepared["y"][train_mask],
        scaler__sample_weight=training_weights,
        classifier__sample_weight=training_weights,
    )

    slice_probabilities = model.predict_proba(prepared["X"][valid_mask])[:, 1]
    return aggregate_legacy_slice_probabilities(
        slice_probabilities=slice_probabilities,
        labels=prepared["y"][valid_mask],
        patient_ids=prepared["patient_ids"][valid_mask],
        series_ids=prepared["series_ids"][valid_mask],
        quality_weights=prepared["slice_weights"][valid_mask],
        fusion_method=experiment.fusion_method,
    )


def create_inner_fold_assignment(
    train_patient_ids,
    train_labels,
    patient_to_group,
    outer_fold_index,
):
    """Construct duplicate-aware inner folds, reducing count only if necessary."""

    train_patient_ids = np.asarray(train_patient_ids)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    groups = np.asarray(
        [patient_to_group[str(patient_id)] for patient_id in train_patient_ids]
    )

    maximum_splits = min(
        INNER_CV_SPLITS,
        int(np.bincount(train_labels, minlength=2).min()),
        len(np.unique(groups)),
    )
    last_error = None
    for n_splits in range(maximum_splits, 1, -1):
        try:
            folds = assign_stratified_patient_folds(
                train_patient_ids,
                train_labels,
                groups,
                n_splits,
                INNER_CV_RANDOM_STATE + 100 * outer_fold_index,
            )
            return folds, n_splits
        except RuntimeError as error:
            last_error = error

    raise RuntimeError(
        "Could not create at least two valid inner folds for training-only "
        f"selection. Last error: {last_error}"
    )


def collect_inner_oof_raw_scores(
    experiment,
    prepared,
    outer_train_patient_ids,
    outer_train_labels,
    patient_to_group,
    outer_fold_index,
    c_value,
):
    """Generate training-only patient OOF scores for C/threshold selection."""

    inner_folds, actual_inner_splits = create_inner_fold_assignment(
        outer_train_patient_ids,
        outer_train_labels,
        patient_to_group,
        outer_fold_index,
    )

    score_by_patient = {}
    label_by_patient = {}
    for inner_fold in range(1, actual_inner_splits + 1):
        inner_train_patients = outer_train_patient_ids[inner_folds != inner_fold]
        inner_valid_patients = outer_train_patient_ids[inner_folds == inner_fold]

        scores, labels, patients = fit_predict_raw_for_patient_sets(
            experiment=experiment,
            prepared=prepared,
            train_patients=inner_train_patients,
            valid_patients=inner_valid_patients,
            c_value=c_value,
        )
        for patient_id, label, score in zip(patients, labels, scores):
            patient_id = str(patient_id)
            if patient_id in score_by_patient:
                raise RuntimeError(
                    f"Patient {patient_id} received multiple inner OOF scores."
                )
            score_by_patient[patient_id] = float(score)
            label_by_patient[patient_id] = int(label)

    ordered_patients = np.asarray(sorted(map(str, outer_train_patient_ids)))
    if set(score_by_patient) != set(ordered_patients.tolist()):
        missing = sorted(set(ordered_patients.tolist()) - set(score_by_patient))
        raise RuntimeError(f"Missing inner OOF patients: {missing}")

    labels = np.asarray(
        [label_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.int64,
    )
    scores = np.asarray(
        [score_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.float64,
    )
    return ordered_patients, labels, scores, actual_inner_splits


def fit_sigmoid_calibrator(raw_scores, labels):
    """Fit a one-dimensional sigmoid only on inner OOF training scores."""

    raw_scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int64)
    calibrator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_SEED,
    )
    calibrator.fit(
        raw_scores,
        labels,
        sample_weight=compute_balanced_patient_weights(labels),
    )
    return calibrator


def apply_sigmoid_calibrator(calibrator, raw_scores):
    return np.asarray(
        calibrator.predict_proba(
            np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1)
        )[:, 1],
        dtype=np.float64,
    )


def select_decision_threshold(labels, probabilities):
    """Select a threshold using only inner OOF training predictions."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        probabilities,
    )
    finite = np.isfinite(thresholds)
    false_positive_rate = false_positive_rate[finite]
    true_positive_rate = true_positive_rate[finite]
    thresholds = thresholds[finite]

    if len(thresholds) == 0:
        return 0.5

    if THRESHOLD_SELECTION_METHOD == "youden":
        statistic = true_positive_rate - false_positive_rate
        best_value = float(np.max(statistic))
        candidates = np.flatnonzero(
            np.isclose(statistic, best_value, rtol=0.0, atol=1e-12)
        )
        # A larger threshold is the conservative deterministic tie-break.
        return float(np.max(thresholds[candidates]))

    eligible = np.flatnonzero(true_positive_rate >= TARGET_SENSITIVITY)
    if len(eligible) == 0:
        best_index = int(np.argmax(true_positive_rate))
        return float(thresholds[best_index])

    # Among thresholds meeting the sensitivity target, minimize FPR; use the
    # largest threshold as a deterministic secondary tie-break.
    eligible_fpr = false_positive_rate[eligible]
    minimum_fpr = float(np.min(eligible_fpr))
    best = eligible[np.isclose(eligible_fpr, minimum_fpr, atol=1e-12)]
    return float(np.max(thresholds[best]))


def experiment_candidate_c_values(experiment):
    if experiment.tune_c:
        return tuple(sorted(set(map(float, CLASSIFIER_C_GRID))))
    return (float(experiment.fixed_c),)


def select_c_and_training_threshold(
    experiment,
    prepared,
    outer_train_patient_ids,
    outer_train_labels,
    patient_to_group,
    outer_fold_index,
):
    """Select C and threshold exclusively from the outer-training cohort."""

    candidate_results = []
    for c_value in experiment_candidate_c_values(experiment):
        patients, labels, raw_scores, inner_splits = collect_inner_oof_raw_scores(
            experiment=experiment,
            prepared=prepared,
            outer_train_patient_ids=outer_train_patient_ids,
            outer_train_labels=outer_train_labels,
            patient_to_group=patient_to_group,
            outer_fold_index=outer_fold_index,
            c_value=c_value,
        )
        inner_auc = float(roc_auc_score(labels, raw_scores))
        candidate_results.append(
            {
                "c_value": float(c_value),
                "patient_ids": patients,
                "labels": labels,
                "raw_scores": raw_scores,
                "inner_splits": int(inner_splits),
                "inner_auc": inner_auc,
            }
        )
        print(
            f"[INNER CV][{experiment.experiment_id}][outer={outer_fold_index}] "
            f"C={c_value:g}, pooled_inner_AUC={inner_auc:.4f}",
            flush=True,
        )

    selected = sorted(
        candidate_results,
        key=lambda result: (-result["inner_auc"], result["c_value"]),
    )[0]

    if experiment.classifier_type == "linear_svm":
        calibrator = fit_sigmoid_calibrator(
            selected["raw_scores"],
            selected["labels"],
        )
        inner_probabilities = apply_sigmoid_calibrator(
            calibrator,
            selected["raw_scores"],
        )
    else:
        calibrator = None
        inner_probabilities = np.clip(
            selected["raw_scores"],
            0.0,
            1.0,
        )

    threshold = select_decision_threshold(
        selected["labels"],
        inner_probabilities,
    )
    return {
        "selected_c": selected["c_value"],
        "inner_auc": selected["inner_auc"],
        "inner_splits": selected["inner_splits"],
        "inner_probabilities": inner_probabilities,
        "inner_labels": selected["labels"],
        "threshold": float(threshold),
        "calibrator": calibrator,
        "candidate_results": [
            {
                "c_value": result["c_value"],
                "inner_auc": result["inner_auc"],
            }
            for result in candidate_results
        ],
    }

# =============================
# PIPELINE STEP 9
# PATIENT-LEVEL METRICS AND CONFIDENCE INTERVALS
# =============================


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_binary_patient_metrics(labels, probabilities, predictions):
    """Compute threshold-free and threshold-dependent patient metrics."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if not (len(labels) == len(probabilities) == len(predictions)):
        raise ValueError("Metric arrays must have equal lengths.")
    if np.unique(labels).size != 2:
        raise ValueError("Patient metrics require both classes.")

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    ppv = _safe_divide(tp, tp + fp)
    npv = _safe_divide(tn, tn + fn)

    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": float(
            np.nanmean([sensitivity, specificity])
        ),
        "accuracy": float(np.mean(predictions == labels)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def bootstrap_patient_metric_intervals(
    labels,
    probabilities,
    predictions,
    n_bootstrap=BOOTSTRAP_REPLICATES,
    confidence=BOOTSTRAP_CONFIDENCE,
    random_state=RANDOM_SEED,
):
    """Stratified patient bootstrap intervals for all reported metrics."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    if len(class0) == 0 or len(class1) == 0:
        raise ValueError("Bootstrap intervals require both classes.")

    metric_names = (
        "auc",
        "auprc",
        "brier_score",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "balanced_accuracy",
        "accuracy",
    )
    values = {name: [] for name in metric_names}
    rng = np.random.default_rng(random_state)

    for _ in range(n_bootstrap):
        sampled0 = rng.choice(class0, size=len(class0), replace=True)
        sampled1 = rng.choice(class1, size=len(class1), replace=True)
        sampled = np.concatenate([sampled0, sampled1])
        metrics = compute_binary_patient_metrics(
            labels[sampled],
            probabilities[sampled],
            predictions[sampled],
        )
        for name in metric_names:
            value = metrics[name]
            if np.isfinite(value):
                values[name].append(float(value))

    alpha = 1.0 - confidence
    intervals = {}
    for name, metric_values in values.items():
        if not metric_values:
            intervals[name] = [None, None]
            continue
        array = np.asarray(metric_values, dtype=np.float64)
        intervals[name] = [
            float(np.quantile(array, alpha / 2.0)),
            float(np.quantile(array, 1.0 - alpha / 2.0)),
        ]
    return intervals


def run_one_experiment(
    experiment,
    prepared,
    fold_manifest_rows,
    output_dir,
):
    """Run all outer folds and save exactly one OOF row per patient."""

    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(asdict(experiment), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    patient_to_fold = {
        row["patient_id"]: int(row["outer_fold"])
        for row in fold_manifest_rows
    }
    patient_to_label = {
        row["patient_id"]: int(row["true_label"])
        for row in fold_manifest_rows
    }
    patient_to_group = {
        row["patient_id"]: row["duplicate_component_id"]
        for row in fold_manifest_rows
    }
    all_patients = np.asarray(sorted(patient_to_fold))
    all_labels = np.asarray(
        [patient_to_label[patient_id] for patient_id in all_patients],
        dtype=np.int64,
    )

    prepared_patients = set(map(str, prepared["patient_ids"]))
    if prepared_patients != set(all_patients.tolist()):
        missing = sorted(set(all_patients.tolist()) - prepared_patients)
        extra = sorted(prepared_patients - set(all_patients.tolist()))
        raise RuntimeError(
            f"Prepared patient table mismatch. Missing={missing}, extra={extra}."
        )

    score_by_patient = {}
    prediction_by_patient = {}
    threshold_by_patient = {}
    selected_c_by_patient = {}
    fold_rows = []

    print("\n" + "=" * 100, flush=True)
    print(f"[EXPERIMENT] START: {experiment.experiment_id}", flush=True)
    print(f"[EXPERIMENT] {experiment.description}", flush=True)
    print(
        f"[EXPERIMENT] feature_mode={experiment.feature_mode}, "
        f"strategy={experiment.strategy}, pooling={experiment.pooling_strategy}, "
        f"weighting={experiment.weighting_mode}, "
        f"classifier={experiment.classifier_type}, PCA={experiment.use_pca}, "
        f"fusion={experiment.fusion_method}",
        flush=True,
    )
    print("=" * 100, flush=True)

    for outer_fold in range(1, N_SPLITS + 1):
        fold_started_at = time.perf_counter()
        train_patients = np.asarray(
            [
                patient_id
                for patient_id in all_patients
                if patient_to_fold[str(patient_id)] != outer_fold
            ]
        )
        valid_patients = np.asarray(
            [
                patient_id
                for patient_id in all_patients
                if patient_to_fold[str(patient_id)] == outer_fold
            ]
        )
        train_labels = np.asarray(
            [patient_to_label[str(patient_id)] for patient_id in train_patients],
            dtype=np.int64,
        )
        valid_labels_expected = np.asarray(
            [patient_to_label[str(patient_id)] for patient_id in valid_patients],
            dtype=np.int64,
        )

        print(
            f"\n[EXPERIMENT {experiment.experiment_id}] OUTER FOLD "
            f"{outer_fold}/{N_SPLITS}",
            flush=True,
        )
        print(
            f"  train_patients={len(train_patients)} "
            f"(Normal={int(np.sum(train_labels == 0))}, "
            f"Sick={int(np.sum(train_labels == 1))})",
            flush=True,
        )
        print(
            f"  valid_patients={len(valid_patients)} "
            f"(Normal={int(np.sum(valid_labels_expected == 0))}, "
            f"Sick={int(np.sum(valid_labels_expected == 1))})",
            flush=True,
        )

        selection = select_c_and_training_threshold(
            experiment=experiment,
            prepared=prepared,
            outer_train_patient_ids=train_patients,
            outer_train_labels=train_labels,
            patient_to_group=patient_to_group,
            outer_fold_index=outer_fold,
        )
        print(
            f"[OUTER {outer_fold}] selected_C={selection['selected_c']:g}, "
            f"inner_AUC={selection['inner_auc']:.4f}, "
            f"training_only_threshold={selection['threshold']:.6f}, "
            f"inner_splits={selection['inner_splits']}",
            flush=True,
        )

        raw_scores, valid_labels, evaluated_patients = (
            fit_predict_raw_for_patient_sets(
                experiment=experiment,
                prepared=prepared,
                train_patients=train_patients,
                valid_patients=valid_patients,
                c_value=selection["selected_c"],
            )
        )

        if experiment.classifier_type == "linear_svm":
            probabilities = apply_sigmoid_calibrator(
                selection["calibrator"],
                raw_scores,
            )
        else:
            probabilities = np.clip(raw_scores, 0.0, 1.0)

        evaluated_patients = np.asarray(evaluated_patients)
        sort_order = np.argsort(evaluated_patients.astype(str))
        evaluated_patients = evaluated_patients[sort_order]
        valid_labels = np.asarray(valid_labels, dtype=np.int64)[sort_order]
        probabilities = np.asarray(probabilities, dtype=np.float64)[sort_order]

        expected_order = np.asarray(sorted(map(str, valid_patients)))
        if not np.array_equal(
            evaluated_patients.astype(str),
            expected_order.astype(str),
        ):
            raise RuntimeError(
                f"Outer fold {outer_fold}: evaluated patient IDs do not match "
                "the shared fold manifest."
            )
        expected_labels_ordered = np.asarray(
            [patient_to_label[str(patient_id)] for patient_id in expected_order],
            dtype=np.int64,
        )
        if not np.array_equal(valid_labels, expected_labels_ordered):
            raise RuntimeError(
                f"Outer fold {outer_fold}: validation labels do not match the "
                "shared fold manifest."
            )

        predictions = (
            probabilities >= float(selection["threshold"])
        ).astype(np.int64)
        fold_auc = float(roc_auc_score(valid_labels, probabilities))

        for patient_id, label, probability, prediction in zip(
            evaluated_patients,
            valid_labels,
            probabilities,
            predictions,
        ):
            patient_id = str(patient_id)
            if patient_id in score_by_patient:
                raise RuntimeError(
                    f"Patient {patient_id} received multiple outer OOF scores."
                )
            if int(label) != patient_to_label[patient_id]:
                raise RuntimeError(
                    f"Patient {patient_id} received an inconsistent OOF label."
                )
            score_by_patient[patient_id] = float(probability)
            prediction_by_patient[patient_id] = int(prediction)
            threshold_by_patient[patient_id] = float(selection["threshold"])
            selected_c_by_patient[patient_id] = float(selection["selected_c"])

        fold_row = {
            "outer_fold": outer_fold,
            "n_train_patients": int(len(train_patients)),
            "n_valid_patients": int(len(valid_patients)),
            "selected_c": float(selection["selected_c"]),
            "inner_pooled_auc": float(selection["inner_auc"]),
            "inner_cv_splits": int(selection["inner_splits"]),
            "selected_threshold": float(selection["threshold"]),
            "outer_fold_auc": fold_auc,
            "runtime_seconds": float(time.perf_counter() - fold_started_at),
            "candidate_c_results_json": json.dumps(
                selection["candidate_results"],
                sort_keys=True,
            ),
        }
        fold_rows.append(fold_row)
        print(
            f"[OUTER {outer_fold}] held-out AUC={fold_auc:.4f}; "
            f"runtime={_format_elapsed_time(fold_row['runtime_seconds'])}",
            flush=True,
        )

    if set(score_by_patient) != set(all_patients.tolist()):
        missing = sorted(set(all_patients.tolist()) - set(score_by_patient))
        raise RuntimeError(f"Patients missing final OOF scores: {missing}")

    ordered_patients = np.asarray(sorted(score_by_patient))
    labels = np.asarray(
        [patient_to_label[patient_id] for patient_id in ordered_patients],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [score_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.float64,
    )
    predictions = np.asarray(
        [prediction_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.int64,
    )
    folds = np.asarray(
        [patient_to_fold[patient_id] for patient_id in ordered_patients],
        dtype=np.int64,
    )
    thresholds = np.asarray(
        [threshold_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.float64,
    )
    selected_cs = np.asarray(
        [selected_c_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.float64,
    )

    metrics = compute_binary_patient_metrics(
        labels,
        probabilities,
        predictions,
    )
    intervals = bootstrap_patient_metric_intervals(
        labels,
        probabilities,
        predictions,
        random_state=RANDOM_SEED
        + int(hashlib.sha256(experiment.experiment_id.encode()).hexdigest()[:8], 16),
    )
    runtime_seconds = float(time.perf_counter() - started_at)

    prediction_path = output_dir / "patient_oof_predictions.csv"
    with open(prediction_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "experiment_id",
            "patient_id",
            "true_label",
            "outer_fold",
            "oof_score",
            "training_only_threshold",
            "predicted_label",
            "selected_c",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in zip(
            ordered_patients,
            labels,
            folds,
            probabilities,
            thresholds,
            predictions,
            selected_cs,
        ):
            writer.writerow(
                {
                    "experiment_id": experiment.experiment_id,
                    "patient_id": str(row[0]),
                    "true_label": int(row[1]),
                    "outer_fold": int(row[2]),
                    "oof_score": float(row[3]),
                    "training_only_threshold": float(row[4]),
                    "predicted_label": int(row[5]),
                    "selected_c": float(row[6]),
                }
            )

    fold_path = output_dir / "fold_metrics.csv"
    with open(fold_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    summary = {
        "status": "OK",
        "experiment": asdict(experiment),
        "n_patients": int(len(ordered_patients)),
        "normal_patients": int(np.sum(labels == 0)),
        "sick_patients": int(np.sum(labels == 1)),
        "metrics": metrics,
        "confidence_intervals": intervals,
        "fold_metrics": fold_rows,
        "runtime_seconds": runtime_seconds,
        "prediction_csv": str(prediction_path),
        "fold_metrics_csv": str(fold_path),
        "score_interpretation": (
            "Cross-validated model score; not an externally validated clinical "
            "probability. SVM scores receive fold-local sigmoid calibration "
            "from inner OOF training predictions."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    auc_ci = intervals["auc"]
    print(
        f"[EXPERIMENT] COMPLETED: {experiment.experiment_id} | "
        f"AUC={metrics['auc']:.4f} "
        f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}] | "
        f"AUPRC={metrics['auprc']:.4f} | "
        f"Sensitivity={metrics['sensitivity']:.4f} | "
        f"Specificity={metrics['specificity']:.4f} | "
        f"runtime={_format_elapsed_time(runtime_seconds)}",
        flush=True,
    )

    return {
        "status": "OK",
        "config": experiment,
        "summary": summary,
        "patient_ids": ordered_patients,
        "labels": labels,
        "probabilities": probabilities,
        "predictions": predictions,
        "folds": folds,
        "thresholds": thresholds,
        "selected_cs": selected_cs,
    }

# =============================
# PIPELINE STEP 10
# PAIRED COMPARISONS AND MASTER REPORTS
# =============================


def paired_auc_difference_interval(
    baseline_labels,
    baseline_scores,
    comparison_scores,
    n_bootstrap=PAIRED_BOOTSTRAP_REPLICATES,
    confidence=BOOTSTRAP_CONFIDENCE,
    random_state=RANDOM_SEED,
):
    """Paired patient bootstrap for comparison AUC minus baseline AUC."""

    labels = np.asarray(baseline_labels, dtype=np.int64)
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
    comparison_scores = np.asarray(comparison_scores, dtype=np.float64)
    if not (
        len(labels) == len(baseline_scores) == len(comparison_scores)
    ):
        raise ValueError("Paired AUC arrays must have equal lengths.")

    observed = float(
        roc_auc_score(labels, comparison_scores)
        - roc_auc_score(labels, baseline_scores)
    )
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(random_state)
    differences = np.empty(n_bootstrap, dtype=np.float64)

    for index in range(n_bootstrap):
        sampled0 = rng.choice(class0, size=len(class0), replace=True)
        sampled1 = rng.choice(class1, size=len(class1), replace=True)
        sampled = np.concatenate([sampled0, sampled1])
        differences[index] = (
            roc_auc_score(labels[sampled], comparison_scores[sampled])
            - roc_auc_score(labels[sampled], baseline_scores[sampled])
        )

    alpha = 1.0 - confidence
    return {
        "delta_auc": observed,
        "delta_auc_ci_lower": float(
            np.quantile(differences, alpha / 2.0)
        ),
        "delta_auc_ci_upper": float(
            np.quantile(differences, 1.0 - alpha / 2.0)
        ),
        "bootstrap_probability_delta_above_zero": float(
            np.mean(differences > 0.0)
        ),
    }


def compare_experiments_to_baseline(results):
    """Return one paired AUC comparison row for every successful experiment."""

    successful = {
        result["config"].experiment_id: result
        for result in results
        if result.get("status") == "OK"
    }
    if BASELINE_EXPERIMENT_ID not in successful:
        print(
            "[COMPARISON] Baseline failed or is missing; paired comparisons "
            "cannot be calculated.",
            flush=True,
        )
        return []

    baseline = successful[BASELINE_EXPERIMENT_ID]
    baseline_order = np.argsort(baseline["patient_ids"].astype(str))
    baseline_patients = baseline["patient_ids"][baseline_order].astype(str)
    baseline_labels = baseline["labels"][baseline_order]
    baseline_scores = baseline["probabilities"][baseline_order]

    rows = []
    for experiment_id, result in successful.items():
        order = np.argsort(result["patient_ids"].astype(str))
        patients = result["patient_ids"][order].astype(str)
        labels = result["labels"][order]
        scores = result["probabilities"][order]
        if not np.array_equal(patients, baseline_patients):
            raise RuntimeError(
                f"Experiment {experiment_id} has a different patient set from "
                "the baseline."
            )
        if not np.array_equal(labels, baseline_labels):
            raise RuntimeError(
                f"Experiment {experiment_id} has labels inconsistent with the "
                "baseline."
            )

        comparison = paired_auc_difference_interval(
            baseline_labels,
            baseline_scores,
            scores,
            random_state=RANDOM_SEED
            + int(hashlib.sha256(experiment_id.encode()).hexdigest()[:8], 16),
        )
        row = {
            "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
            "comparison_experiment_id": experiment_id,
            **comparison,
        }
        rows.append(row)

    return sorted(rows, key=lambda row: row["comparison_experiment_id"])


def result_summary_row(result, comparison_lookup):
    experiment = result["config"]
    summary = result["summary"]
    metrics = summary["metrics"]
    intervals = summary["confidence_intervals"]
    comparison = comparison_lookup.get(experiment.experiment_id, {})

    return {
        "experiment_id": experiment.experiment_id,
        "status": result["status"],
        "role": experiment.role,
        "description": experiment.description,
        "feature_mode": experiment.feature_mode,
        "strategy": experiment.strategy,
        "pooling_strategy": experiment.pooling_strategy,
        "weighting_mode": experiment.weighting_mode,
        "classifier_type": experiment.classifier_type,
        "fusion_method": experiment.fusion_method or "",
        "use_pca": int(experiment.use_pca),
        "n_patients": summary["n_patients"],
        "auc": metrics["auc"],
        "auc_ci_lower": intervals["auc"][0],
        "auc_ci_upper": intervals["auc"][1],
        "auprc": metrics["auprc"],
        "auprc_ci_lower": intervals["auprc"][0],
        "auprc_ci_upper": intervals["auprc"][1],
        "sensitivity": metrics["sensitivity"],
        "sensitivity_ci_lower": intervals["sensitivity"][0],
        "sensitivity_ci_upper": intervals["sensitivity"][1],
        "specificity": metrics["specificity"],
        "specificity_ci_lower": intervals["specificity"][0],
        "specificity_ci_upper": intervals["specificity"][1],
        "ppv": metrics["ppv"],
        "ppv_ci_lower": intervals["ppv"][0],
        "ppv_ci_upper": intervals["ppv"][1],
        "npv": metrics["npv"],
        "npv_ci_lower": intervals["npv"][0],
        "npv_ci_upper": intervals["npv"][1],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "brier_score": metrics["brier_score"],
        "delta_auc_vs_baseline": comparison.get("delta_auc"),
        "delta_auc_ci_lower": comparison.get("delta_auc_ci_lower"),
        "delta_auc_ci_upper": comparison.get("delta_auc_ci_upper"),
        "runtime_seconds": summary["runtime_seconds"],
    }


def write_master_outputs(results, failed_results, paired_rows, comparison_dir):
    """Write the cross-experiment CSV/JSON files used for final comparison."""

    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_lookup = {
        row["comparison_experiment_id"]: row for row in paired_rows
    }
    summary_rows = [
        result_summary_row(result, comparison_lookup)
        for result in results
        if result.get("status") == "OK"
    ]
    summary_rows.sort(key=lambda row: (-row["auc"], row["experiment_id"]))

    summary_path = comparison_dir / "experiment_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    prediction_rows = []
    for result in results:
        if result.get("status") != "OK":
            continue
        experiment_id = result["config"].experiment_id
        for row in zip(
            result["patient_ids"],
            result["labels"],
            result["folds"],
            result["probabilities"],
            result["thresholds"],
            result["predictions"],
            result["selected_cs"],
        ):
            prediction_rows.append(
                {
                    "experiment_id": experiment_id,
                    "patient_id": str(row[0]),
                    "true_label": int(row[1]),
                    "outer_fold": int(row[2]),
                    "oof_score": float(row[3]),
                    "training_only_threshold": float(row[4]),
                    "predicted_label": int(row[5]),
                    "selected_c": float(row[6]),
                }
            )

    prediction_path = comparison_dir / "patient_predictions_all_experiments.csv"
    if prediction_rows:
        with open(prediction_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)

    paired_path = comparison_dir / "paired_auc_comparisons.csv"
    if paired_rows:
        with open(paired_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    else:
        paired_path.write_text(
            "baseline_experiment_id,comparison_experiment_id,delta_auc," 
            "delta_auc_ci_lower,delta_auc_ci_upper," 
            "bootstrap_probability_delta_above_zero\n",
            encoding="utf-8",
        )

    report = {
        "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "successful_experiments": summary_rows,
        "failed_experiments": failed_results,
        "paired_auc_comparisons": paired_rows,
        "interpretation_rules": {
            "negative_control_warning_auc": SHORTCUT_WARNING_AUC,
            "delta_auc_ci": (
                "A paired CI containing zero does not establish a reliable "
                "difference between configurations."
            ),
            "clinical_warning": (
                "All scores are exploratory and require independent external "
                "validation before clinical interpretation."
            ),
        },
    }
    (comparison_dir / "final_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_rows


def print_final_comparison(summary_rows):
    """Print a compact ranking and explicit shortcut warnings."""

    print("\n" + "=" * 126, flush=True)
    print("FINAL MULTI-EXPERIMENT PATIENT-LEVEL COMPARISON", flush=True)
    print("=" * 126, flush=True)
    header = (
        f"{'Rank':<5} {'Experiment':<38} {'Role':<17} "
        f"{'AUC [95% CI]':<25} {'Delta AUC vs B0':<22} "
        f"{'Sens.':>7} {'Spec.':>7}"
    )
    print(header, flush=True)
    print("-" * 126, flush=True)

    for rank, row in enumerate(summary_rows, start=1):
        auc_text = (
            f"{row['auc']:.4f} "
            f"[{row['auc_ci_lower']:.4f}, {row['auc_ci_upper']:.4f}]"
        )
        if row["delta_auc_vs_baseline"] is None:
            delta_text = "baseline / unavailable"
        else:
            delta_text = (
                f"{row['delta_auc_vs_baseline']:+.4f} "
                f"[{row['delta_auc_ci_lower']:+.4f}, "
                f"{row['delta_auc_ci_upper']:+.4f}]"
            )
        print(
            f"{rank:<5} {row['experiment_id']:<38} {row['role']:<17} "
            f"{auc_text:<25} {delta_text:<22} "
            f"{row['sensitivity']:>7.3f} {row['specificity']:>7.3f}",
            flush=True,
        )

    print("-" * 126, flush=True)
    print(
        "All rows use the same outer patient-fold manifest. Thresholds were "
        "selected only from inner OOF training predictions.",
        flush=True,
    )

    negative_controls = [
        row
        for row in summary_rows
        if row["role"] == "negative_control"
        and row["auc"] >= SHORTCUT_WARNING_AUC
    ]
    if negative_controls:
        print("\nSHORTCUT / CONFOUNDING WARNINGS", flush=True)
        for row in negative_controls:
            print(
                f"[WARNING] {row['experiment_id']} achieved AUC={row['auc']:.4f} "
                f">= {SHORTCUT_WARNING_AUC:.2f}. The primary model may be using "
                "non-anatomical dataset/protocol/export information.",
                flush=True,
            )
    else:
        print(
            "\n[CONTROL CHECK] No enabled negative control crossed the configured "
            f"AUC warning threshold of {SHORTCUT_WARNING_AUC:.2f}.",
            flush=True,
        )
    print("=" * 126, flush=True)

# =============================
# PIPELINE STEP 11
# SUITE ORCHESTRATION, LOGGING AND FINALIZATION
# =============================


def write_tabular_class_summary(output_path, X, y, feature_names):
    """Write descriptive patient-level feature means/medians by class."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        normal_values = X[y == 0, feature_index]
        sick_values = X[y == 1, feature_index]
        rows.append(
            {
                "feature_name": feature_name,
                "normal_mean": float(np.mean(normal_values)),
                "sick_mean": float(np.mean(sick_values)),
                "sick_minus_normal_mean": float(
                    np.mean(sick_values) - np.mean(normal_values)
                ),
                "normal_median": float(np.median(normal_values)),
                "sick_median": float(np.median(sick_values)),
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_suite_metadata(
    samples,
    fingerprint,
    cache_status,
    exact_summary,
    phash_summary,
    monai_gate_comparison,
    successful_results,
    failed_results,
    total_runtime,
):
    """Collect software, model, data, audit and completion metadata."""

    patient_ids, patient_labels = build_patient_label_table(
        [sample[1] for sample in samples],
        [sample[2] for sample in samples],
    )
    bundle_root = locate_monai_bundle_root()
    official_torchscript_path = bundle_root / "models" / "model.ts"

    metadata = {
        "suite_name": SUITE_NAME,
        "suite_configuration_tag": SUITE_CONFIGURATION_TAG,
        "output_directory": str(OUTPUT_DIR),
        "console_log": str(CONSOLE_LOG_PATH),
        "dataset_path": str(DATASET_PATH),
        "patient_definition": "Directory_*",
        "series_definition": "immediate child folder proxy; not DICOM UID",
        "n_images": int(len(samples)),
        "n_patients": int(len(patient_ids)),
        "normal_patients": int(np.sum(patient_labels == 0)),
        "sick_patients": int(np.sum(patient_labels == 1)),
        "feature_bank_fingerprint": fingerprint,
        "feature_bank_cache_status": cache_status,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "opencv": cv2.__version__,
        "device": DEVICE,
        "monai_runtime_source": MONAI_RUNTIME_SOURCE,
        "monai_runtime_artifact_path": MONAI_RUNTIME_ARTIFACT_PATH,
        "monai_hf_revision": MONAI_HF_REVISION,
        "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
        "exact_duplicate_audit": exact_summary,
        "perceptual_duplicate_audit": phash_summary,
        "monai_gate_comparison": monai_gate_comparison,
        "successful_experiment_ids": [
            result["config"].experiment_id for result in successful_results
        ],
        "failed_experiments": failed_results,
        "total_runtime_seconds": float(total_runtime),
        "external_validation": {
            "requested": bool(RUN_EXTERNAL_VALIDATION),
            "status": (
                "BLOCKED_REQUIRES_VERIFIED_EXTERNAL_DATA_ADAPTER"
                if RUN_EXTERNAL_VALIDATION
                else "SKIPPED_NOT_CONFIGURED"
            ),
            "dataset_path": EXTERNAL_DATASET_PATH,
        },
    }
    if official_torchscript_path.is_file():
        metadata["monai_actual_model_ts_sha256"] = sha256_file(
            official_torchscript_path
        )
    else:
        metadata["monai_actual_model_ts_sha256"] = None
    return metadata


def write_suite_configuration(output_path, experiments):
    """Save every predeclared setting before model evaluation starts."""

    configuration = {
        "suite_name": SUITE_NAME,
        "suite_configuration_tag": SUITE_CONFIGURATION_TAG,
        "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "experiments": [asdict(experiment) for experiment in experiments],
        "outer_cv_splits": N_SPLITS,
        "inner_cv_splits": INNER_CV_SPLITS,
        "classifier_c_grid": list(CLASSIFIER_C_GRID),
        "threshold_selection_method": THRESHOLD_SELECTION_METHOD,
        "target_sensitivity": TARGET_SENSITIVITY,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "patient_pca_explained_variance": (
            PATIENT_PCA_EXPLAINED_VARIANCE
        ),
        "patient_definition": "Directory_*",
        "group_splits_by_exact_duplicates": (
            GROUP_SPLITS_BY_EXACT_DUPLICATES
        ),
        "group_splits_by_phash_candidates": (
            GROUP_SPLITS_BY_PHASH_CANDIDATES
        ),
        "external_validation_requested": RUN_EXTERNAL_VALIDATION,
        "external_dataset_path": EXTERNAL_DATASET_PATH,
    }
    output_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main():
    """Run every enabled experiment and compare the patient-level results."""

    pipeline_started_at = time.perf_counter()
    stage_durations = {}
    experiments = get_enabled_experiments()

    print("\n" + "#" * 100, flush=True)
    print("CAD CARDIAC MRI — MULTI-EXPERIMENT PATIENT-LEVEL SUITE", flush=True)
    print("#" * 100, flush=True)
    print(f"[SUITE] Dataset: {DATASET_PATH}", flush=True)
    print(f"[SUITE] Output: {OUTPUT_DIR}", flush=True)
    print(f"[SUITE] Device: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(
            f"[SUITE] CUDA device: {torch.cuda.get_device_name(0)}",
            flush=True,
        )
    print(
        f"[SUITE] Enabled experiments ({len(experiments)}): "
        + ", ".join(experiment.experiment_id for experiment in experiments),
        flush=True,
    )
    print(
        "[SUITE] Patient definition is fixed: patient_id = Directory_*.",
        flush=True,
    )

    stage_started = _print_stage_start(
        1,
        "Validate suite configuration",
        "Fast; no image scanning or model loading.",
    )
    validate_configuration()
    stage_durations["01 Configuration validation"] = _print_stage_complete(
        1,
        "Validate suite configuration",
        stage_started,
        "All registry, CV, audit and model settings passed validation.",
    )

    stage_started = _print_stage_start(
        2,
        "Create output structure and save predeclared configuration",
        "Fast filesystem and JSON operations.",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The suite directory is deterministic for a given configuration. Remove
    # stale run-specific subdirectories before a rerun so an old successful
    # summary cannot be mistaken for the result of a newly failed experiment.
    for subdirectory_name in (
        "manifests",
        "audits",
        "experiments",
        "comparison",
    ):
        directory = OUTPUT_DIR / subdirectory_name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    write_suite_configuration(
        OUTPUT_DIR / "suite_configuration.json",
        experiments,
    )
    stage_durations["02 Output structure"] = _print_stage_complete(
        2,
        "Create output structure and save predeclared configuration",
        stage_started,
        f"Configuration: {OUTPUT_DIR / 'suite_configuration.json'}",
    )

    stage_started = _print_stage_start(
        3,
        "Discover Directory_* patients and image rows",
        "Depends on filesystem and folder count; no pixel decoding yet.",
    )
    samples = load_samples(DATASET_PATH)
    stage_durations["03 Dataset discovery"] = _print_stage_complete(
        3,
        "Discover Directory_* patients and image rows",
        stage_started,
        f"Discovered {len(samples)} image rows.",
    )

    stage_started = _print_stage_start(
        4,
        "Load or build the shared multi-view feature bank",
        "Potentially the longest stage on a cache miss; all required image "
        "variants are encoded and cached.",
    )
    required_modes = required_efficientnet_feature_modes(experiments)
    fingerprint = feature_bank_fingerprint(samples, DATASET_PATH)
    cache_dir = FEATURE_CACHE_ROOT / fingerprint[:16]
    bank, cache_status = load_or_extract_feature_bank(
        samples,
        required_modes,
        fingerprint,
        cache_dir,
    )
    stage_durations["04 Shared feature bank"] = _print_stage_complete(
        4,
        "Load or build the shared multi-view feature bank",
        stage_started,
        f"Cache={cache_status}; modes={list(required_modes)}; "
        f"path={bank['cache_dir']}",
    )

    stage_started = _print_stage_start(
        5,
        "Audit exact and perceptual cross-patient duplicates",
        "Hash grouping and pHash candidate search; no neural-network inference.",
    )
    if AUDIT_EXACT_DECODED_PIXEL_DUPLICATES:
        exact_summary, exact_edges = audit_exact_decoded_pixel_duplicates(
            OUTPUT_DIR / "audits" / "exact_decoded_pixel_duplicate_groups.csv",
            samples,
            bank["decoded_pixel_hashes"],
        )
    else:
        exact_summary = {"enabled": False}
        exact_edges = set()

    if AUDIT_PERCEPTUAL_NEAR_DUPLICATES:
        phash_summary, phash_edges = audit_perceptual_near_duplicate_candidates(
            OUTPUT_DIR
            / "audits"
            / "perceptual_near_duplicate_patient_pairs.csv",
            samples,
            bank["perceptual_hashes"],
            bank["decoded_pixel_hashes"],
        )
    else:
        phash_summary = {"enabled": False}
        phash_edges = set()

    stage_durations["05 Duplicate audits"] = _print_stage_complete(
        5,
        "Audit exact and perceptual cross-patient duplicates",
        stage_started,
        f"Exact patient edges={len(exact_edges)}; "
        f"perceptual candidate edges={len(phash_edges)}.",
    )

    stage_started = _print_stage_start(
        6,
        "Create one duplicate-aware outer-fold and cohort manifest",
        "Fast patient-level grouping plus one large CSV write.",
    )
    fold_manifest_rows = build_patient_fold_manifest(
        bank["labels"],
        bank["patient_ids"],
        exact_edges,
        phash_edges,
    )
    write_patient_fold_manifest(
        OUTPUT_DIR / "manifests" / "patient_fold_manifest.csv",
        fold_manifest_rows,
    )
    write_cohort_manifest(
        OUTPUT_DIR / "manifests" / "cohort_manifest.csv",
        samples,
        bank,
        fold_manifest_rows,
    )
    stage_durations["06 Manifests"] = _print_stage_complete(
        6,
        "Create one duplicate-aware outer-fold and cohort manifest",
        stage_started,
        "Every experiment will reuse these exact outer folds.",
    )

    stage_started = _print_stage_start(
        7,
        "Build and save provenance and MONAI-QC patient controls",
        "Patient-level aggregation and descriptive audit files.",
    )
    provenance_set = aggregate_patient_provenance_features(bank)
    monai_gate_comparison, monai_qc_set = write_monai_qc_outputs(
        OUTPUT_DIR / "audits",
        bank,
    )
    write_patient_tabular_features(
        OUTPUT_DIR / "audits" / "patient_provenance_features.csv",
        *provenance_set,
    )
    write_tabular_class_summary(
        OUTPUT_DIR / "audits" / "provenance_class_summary.csv",
        provenance_set[0],
        provenance_set[1],
        provenance_set[3],
    )
    write_tabular_class_summary(
        OUTPUT_DIR / "audits" / "monai_qc_class_summary.csv",
        monai_qc_set[0],
        monai_qc_set[1],
        monai_qc_set[3],
    )
    tabular_feature_sets = {
        "provenance_only": provenance_set,
        "monai_qc_only": monai_qc_set,
    }
    stage_durations["07 QC/provenance controls"] = _print_stage_complete(
        7,
        "Build and save provenance and MONAI-QC patient controls",
        stage_started,
        "Patient-level control matrices are ready for C3 and C4.",
    )

    stage_started = _print_stage_start(
        8,
        "Run every enabled experiment on the shared folds",
        "Several fold-local fits. Patient-embedding experiments are fast; "
        "legacy slice classifiers are substantially heavier.",
    )
    successful_results = []
    failed_results = []
    prepared_cache = {}

    for experiment_index, experiment in enumerate(experiments, start=1):
        print(
            f"\n[SUITE] Launching experiment {experiment_index}/{len(experiments)}: "
            f"{experiment.experiment_id}",
            flush=True,
        )
        experiment_output = OUTPUT_DIR / "experiments" / experiment.experiment_id
        preparation_key = (
            experiment.feature_mode,
            experiment.strategy,
            experiment.pooling_strategy,
            experiment.weighting_mode,
        )

        try:
            if preparation_key not in prepared_cache:
                preparation_started = time.perf_counter()
                prepared_cache[preparation_key] = prepare_experiment_data(
                    experiment,
                    bank,
                    tabular_feature_sets,
                )
                print(
                    f"[SUITE] Prepared data representation {preparation_key} in "
                    f"{_format_elapsed_time(time.perf_counter() - preparation_started)}.",
                    flush=True,
                )

            result = run_one_experiment(
                experiment=experiment,
                prepared=prepared_cache[preparation_key],
                fold_manifest_rows=fold_manifest_rows,
                output_dir=experiment_output,
            )
            successful_results.append(result)

        except Exception as error:
            experiment_output.mkdir(parents=True, exist_ok=True)
            failure = {
                "experiment_id": experiment.experiment_id,
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
            failed_results.append(failure)
            (experiment_output / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                f"[EXPERIMENT] FAILED: {experiment.experiment_id}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            traceback.print_exc(file=sys.stdout)

            if FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS:
                raise

    stage_durations["08 Experiment execution"] = _print_stage_complete(
        8,
        "Run every enabled experiment on the shared folds",
        stage_started,
        f"Successful={len(successful_results)}, failed={len(failed_results)}.",
    )

    stage_started = _print_stage_start(
        9,
        "Calculate paired comparisons and write master result files",
        "Patient-level bootstrap comparisons and CSV/JSON consolidation.",
    )
    paired_rows = compare_experiments_to_baseline(successful_results)
    summary_rows = write_master_outputs(
        successful_results,
        failed_results,
        paired_rows,
        OUTPUT_DIR / "comparison",
    )
    stage_durations["09 Master comparisons"] = _print_stage_complete(
        9,
        "Calculate paired comparisons and write master result files",
        stage_started,
        f"Summary rows={len(summary_rows)}; paired rows={len(paired_rows)}.",
    )

    stage_started = _print_stage_start(
        10,
        "Record external-validation status",
        "Immediate unless a verified independent-data adapter is later added.",
    )
    if RUN_EXTERNAL_VALIDATION:
        external_status = {
            "status": "BLOCKED_REQUIRES_VERIFIED_EXTERNAL_DATA_ADAPTER",
            "dataset_path": EXTERNAL_DATASET_PATH,
            "reason": (
                "The external cohort's endpoint, patient mapping, sequence "
                "contract and preprocessing compatibility must be validated "
                "before this script can apply a locked model."
            ),
        }
        print(
            "[EXTERNAL VALIDATION] BLOCKED: a dataset-specific verified adapter "
            "has not been defined. No external labels were inspected or used.",
            flush=True,
        )
    else:
        external_status = {
            "status": "SKIPPED_NOT_CONFIGURED",
            "dataset_path": None,
            "reason": "No independent hospital dataset was configured.",
        }
        print(
            "[EXTERNAL VALIDATION] SKIPPED: no independent dataset configured.",
            flush=True,
        )
    (OUTPUT_DIR / "external_validation_status.json").write_text(
        json.dumps(external_status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    stage_durations["10 External validation status"] = _print_stage_complete(
        10,
        "Record external-validation status",
        stage_started,
        external_status["status"],
    )

    stage_started = _print_stage_start(
        11,
        "Print final comparison and save suite metadata",
        "Console table, warnings, metadata JSON and timing summary.",
    )
    print_final_comparison(summary_rows)

    total_runtime = time.perf_counter() - pipeline_started_at
    metadata = collect_suite_metadata(
        samples=samples,
        fingerprint=fingerprint,
        cache_status=cache_status,
        exact_summary=exact_summary,
        phash_summary=phash_summary,
        monai_gate_comparison=monai_gate_comparison,
        successful_results=successful_results,
        failed_results=failed_results,
        total_runtime=total_runtime,
    )
    (OUTPUT_DIR / "suite_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    stage_durations["11 Final reporting"] = _print_stage_complete(
        11,
        "Print final comparison and save suite metadata",
        stage_started,
        f"Master summary: {OUTPUT_DIR / 'comparison' / 'experiment_summary.csv'}",
    )

    total_runtime = time.perf_counter() - pipeline_started_at
    _print_timing_summary(stage_durations, total_runtime)
    print(f"\n[SUITE] Outputs saved under: {OUTPUT_DIR}", flush=True)
    print(f"[SUITE] Console log: {CONSOLE_LOG_PATH}", flush=True)
    print(
        f"[SUITE] Completed with {len(successful_results)} successful and "
        f"{len(failed_results)} failed experiments in "
        f"{_format_elapsed_time(total_runtime)}.",
        flush=True,
    )
    print("Done!", flush=True)


class TeeStream:
    """Write text simultaneously to the original stream and a shared log file."""

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, text):
        self.console_stream.write(text)
        self.log_stream.write(text)
        return len(text)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())


def run_with_console_logging():
    """Run main() while preserving every print() and traceback in one log."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(CONSOLE_LOG_PATH, "w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            main()
        except Exception:
            print("\n[SUITE] FATAL ERROR", flush=True)
            traceback.print_exc(file=sys.stdout)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    run_with_console_logging()

# ============================================================================
# EXPERIMENTS DELIBERATELY NOT AUTOMATED IN THIS FILE
# ============================================================================
#
# 1. A cardiac-MRI-pretrained encoder comparison requires a public checkpoint
#    whose exact 2D/temporal input contract can be reconstructed from these
#    released files. Repeating one JPEG as a fake cine clip is not valid.
# 2. Sequence/view-specific evaluation requires reliable blinded recovery or
#    annotation of sequence/view identity. SR_* and series* names alone are not
#    treated as validated sequence labels.
# 3. True external validation requires an independent cohort adapter with a
#    comparable CAD endpoint, patient unit and locked preprocessing contract.
#
# These are recorded as scientific next steps rather than silently approximated.
