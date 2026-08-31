#%% ============================================================
# 🧠 CAD Detection from Cardiac MRI – Deconfounded MONAI Patient-Level Multi-Experiment Pipeline (Single File)
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
#  12. One shared multi-view feature bank for original and label-blind
#      standardized image/ROI/negative-control representations
#  13. Exact-duplicate-aware patient folds plus perceptual near-duplicate
#      candidate auditing
#  14. Nested patient-level cross-validation for classifier C and a decision
#      threshold selected only from outer-training data
#  15. Logistic Regression, Linear SVM, pooling, weighting, PCA and legacy
#      probability-fusion ablations executed through one experiment registry
#  16. Paired patient-bootstrap comparisons and class-specific provenance /
#      MONAI-gate negative controls
#  17. Label-blind removal of only consecutive dark, nearly uniform native
#      padding followed by robust 1st/99th-percentile intensity scaling
#  18. A new standardized primary baseline plus narrow-border, corner, detected-
#      padding, center-crop, strict-ROI and outside-bounding-box controls
#  19. Separate MONAI inference and QC on original versus standardized canvases
#  20. Conservative C selection: the smallest C within a predeclared inner-AUC
#      tolerance of the best candidate is chosen
#  21. Repeated nested patient-level CV to quantify outer-split sensitivity
#  22. A patient-label permutation test that repeats the full nested fitting path
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
#   - reports patient-level bootstrap confidence intervals for discrimination
#     and threshold-dependent metrics;
#   - selects classifier C and the operating threshold only inside the
#     outer-training cohort through duplicate-aware inner CV;
#   - uses one authoritative outer-fold manifest for every experiment;
#   - preserves the original min-max/full-canvas pipeline as a historical
#     reference, while making the label-blind standardized ROI branch the new
#     primary baseline after shortcut controls exposed export confounding;
#   - evaluates whether apparent performance survives narrow-border, corner,
#     detected-padding, center-crop, strict-ROI and outside-box controls;
#   - repeats the primary nested CV across multiple deterministic outer splits;
#   - repeats the complete patient-level fitting path after patient-label
#     permutation to obtain an empirical null AUC distribution;
#   - saves OOF predictions, fold assignments, paired comparisons, duplicate
#     audits, provenance/standardization controls and original/standardized
#     MONAI QC summaries.
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
# Required runtime for the normal path:
#
#   Python 3.10 or newer
#   pip install huggingface_hub
#   pip install torch torchvision opencv-python numpy "scikit-learn>=1.1" matplotlib tqdm
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
# PRIMARY DECONFOUNDED BASELINE PIPELINE FLOW
# ============================================================================
#
# Raw MRI JPEG slice
#    ↓
# Detect consecutive edge rows/columns that are BOTH dark and nearly uniform
# using one fixed label-blind rule with conservative crop safety limits
#    ↓
# Remove only the accepted native padding; preserve the complete retained field
# of view and save crop/padding geometry as QC metadata
#    ↓
# Robust 1st/99th-percentile intensity scaling to [0,1]
#    ↓
# Aspect-ratio-preserving placement in a 256×256 zero-padded canvas
#    ↓
# Pinned pretrained MONAI residual U-Net on the standardized canvas
#    ↓
# Confidence-gated soft ROI or standardized full-image fallback
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
# Fold-local PCA + Logistic Regression trained on one vector per patient
#    ↓
# Nested-CV out-of-fold patient-level CAD-associated MODEL SCORE
#
# Historical B0 reference:
#   original per-image min-max scaling + original full canvas + MONAI soft ROI
#
# Optional legacy ablation:
#   slice Logistic Regression → series fusion → patient fusion
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
#          classification_image          : original [3,224,224] min-max view
#          monai_image                   : original [1,256,256] MONAI canvas
#          standardized_classification   : standardized [3,224,224] view
#          standardized_monai_image      : standardized [1,256,256] canvas
#          detected_padding_image        : [3,224,224] binary padding control
#          label                         : scalar 0 or 1
#          patient_id                    : Directory_* string
#          series_id                     : patient-scoped folder-proxy string
#          sample_index                  : deterministic position in ``samples``
#          decoded_pixel_hash            : exact decoded-pixel SHA-256 string
#          perceptual_hash               : 64-bit DCT pHash candidate key
#          provenance_features          : native export/style feature vector
#          standardization_features     : crop/padding/robust-range QC vector
#
#   C. One DataLoader batch -- the same objects with a leading batch dimension
#
#          images                       : original [B,3,224,224]
#          monai_images                 : original [B,1,256,256]
#          standardized_images          : standardized [B,3,224,224]
#          standardized_monai_images    : standardized [B,1,256,256]
#          detected_padding_images      : binary-control [B,3,224,224]
#
#   D. MONAI outputs -- produced when a selected experiment requires MONAI
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
#      Original full/ROI/border/outside-mask embeddings and standardized
#      full/ROI/narrow-border/corner/center-crop/strict-ROI/outside-box
#      embeddings are stored in one shared feature bank. Labels, patient IDs,
#      series IDs, original and standardized MONAI QC, ROI slice scores, exact/
#      perceptual hashes, provenance features and standardization QC remain in
#      one-to-one row alignment. Any feature-affecting setting changes the
#      feature-bank fingerprint.
#
#   G. Recommended supervised input
#
#      Slice embeddings are pooled within each series proxy, then equally across
#      all series proxies of one Directory_* patient. The classifier therefore
#      receives exactly one vector and one label per patient.
#
#   H. Evaluation output
#
#      Every Directory_* patient receives exactly one outer out-of-fold score,
#      fold identifier, training-only threshold and predicted label for every
#      experiment. ROC-AUC, AUPRC, Brier score, sensitivity, specificity, PPV,
#      NPV, F1 and patient-bootstrap confidence intervals are calculated from
#      those patient rows. Experiment differences use paired resampling of the
#      same patients.
#
# TRAINED VERSUS FROZEN COMPONENTS
# --------------------------------
#
#   Frozen / inference-only:
#       - MONAI ventricular segmenter
#       - ImageNet EfficientNet-B0 encoder
#       - deterministic image preprocessing and pooling rules
#
#   Fitted separately inside every outer/inner training partition:
#       - StandardScaler
#       - optional PCA
#       - Logistic Regression or Linear SVM
#       - optional one-dimensional sigmoid calibrator for SVM margins
#       - classifier C and the decision threshold selected by inner OOF data
#
# This distinction is central to leakage control. Validation-patient labels and
# embeddings are never used to fit fold-local preprocessing, classification,
# calibration, hyperparameter selection or threshold selection.
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
#   - Exact cross-patient duplicate components can be kept in the same fold,
#     because patient-only splitting does not prevent copied visual content
#     from crossing partitions.
#   - Original and standardized border-only, corner-only, padding-only,
#     outside-mask/outside-box, provenance-only, standardization-QC and MONAI-QC
#     controls test whether apparent performance can be explained by
#     non-anatomical shortcuts or protocol/export differences.
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
# The detailed flow above describes the NEW STANDARDIZED BASELINE B1. The
# original sixteen experiments are retained unchanged for direct historical
# comparison, and twelve targeted deconfounding experiments are added:
#
#   ORIGINAL / HISTORICAL SUITE
#   B0  Original MONAI ROI + hierarchical pooling + Logistic Regression + PCA
#   A1  Original full image instead of original MONAI ROI
#   C1  Original 15% border-only negative control
#   C2  Original outside-MONAI-mask negative control
#   C3  Export/provenance metadata-only classifier
#   C4  Original MONAI-QC-only classifier
#   A2  Flat slice-to-patient embedding pooling
#   A3  Legacy slice classifier + log-odds fusion
#   A4  Legacy slice classifier + mean-probability fusion
#   A5  Linear SVM instead of Logistic Regression
#   A6  Optional series-local slice-quality weighting heuristic
#   A7  PCA disabled while C remains selected inside training data
#   A8  No-PCA, fixed-C patient model matched to the legacy branch
#   R1/R2/R3  Deterministic 10%/25%/50% within-series slice dropout
#
#   DECONFOUNDING EXTENSION
#   B1  Label-blind standardized MONAI ROI (new primary baseline)
#   A9  Standardized full image
#   C5  Standardized outer 5% only
#   C6  Standardized outer 10% only
#   C7  Detected native/canvas padding mask only
#   C8  Standardized corners only
#   C9  MONAI-area-matched standardized center crop
#   A10 Standardized MONAI soft ROI with zero background
#   A11 Standardized MONAI bounding-box crop with fixed context
#   C10 Standardized signal outside a larger MONAI bounding box
#   C11 Standardization geometry/QC features only
#   C12 Standardized MONAI gate/QC features only
#
# Every enabled experiment uses the SAME duplicate-aware outer patient-fold
# manifest. Classifier C and the decision threshold are selected only from each
# outer-training cohort through inner patient-level OOF predictions. Linear-SVM
# margins are calibrated by a sigmoid fitted only to inner OOF training scores.
# The new primary baseline is additionally rerun across multiple outer-split
# seeds and through patient-label permutation; no favorable split is selected.
#
# A single script launch does NOT mean one classifier represents every ablation.
# It means the expensive operations are shared correctly:
#
#   JPEG decoding + MONAI inference + frozen EfficientNet encoding
#       ↓
#   one cached multi-view feature bank
#       ↓
#   several independent fold-local classifiers and aggregation rules
#       ↓
#   one paired patient-level comparison report
#
# The following experiments remain deliberately external to this file until a
# valid input contract is available:
#
#   - a cardiac-MRI-pretrained encoder requiring verified sequence/frame order;
#   - sequence/view-specific analysis requiring reliable blinded annotation;
#   - true external validation requiring an independent cohort adapter and a
#     comparable patient-level CAD endpoint.
#
# ============================================================================
# PRINCIPAL OUTPUT PACKAGE
# ============================================================================
#
#   console_output.log
#   suite_configuration.json
#   manifests/cohort_manifest.csv
#   manifests/patient_fold_manifest.csv
#   audits/exact_decoded_pixel_duplicate_groups.csv
#   audits/perceptual_near_duplicate_patient_pairs.csv
#   audits/monai_qc_by_patient.csv
#   audits/monai_gate_class_comparison.json
#   audits/patient_provenance_features.csv
#   audits/patient_standardization_features.csv
#   audits/standardized_monai_qc_by_patient.csv
#   audits/standardized_monai_gate_class_comparison.json
#   experiments/<experiment_id>/patient_oof_predictions.csv
#   experiments/<experiment_id>/fold_metrics.csv
#   experiments/<experiment_id>/summary.json
#   comparison/experiment_summary.csv
#   comparison/patient_predictions_all_experiments.csv
#   comparison/paired_auc_comparisons.csv
#   comparison/paired_primary_ablation_comparisons.csv
#   comparison/failed_experiments.csv
#   comparison/final_report.json
#   stability/repeated_nested_cv_runs.csv
#   stability/repeated_nested_cv_oof_predictions.csv
#   stability/patient_score_stability.csv
#   stability/repeated_nested_cv_summary.json
#   permutation/patient_label_permutation_auc.csv
#   permutation/patient_label_permutation_summary.json
#
# All ordinary print() messages, tqdm progress and tracebacks are duplicated to
# both the live console and ``console_output.log``.
#
# ============================================================================

# =============================
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
    slice_dropout_rate: float = 0.0
    # Optional deterministic robustness perturbation applied before pooling.
    # At least one slice is retained in every series proxy.
    role: str = "ablation"
    enabled: bool = True


# Every enabled experiment is launched automatically by main(). To run a
# smaller preliminary suite, set EXPERIMENTS_TO_RUN to a tuple of IDs. Keep it
# as None to run every configuration whose enabled field is True.
EXPERIMENTS_TO_RUN = None

BASELINE_EXPERIMENT_ID = "B1_STANDARDIZED_ROI_HIER_LR_PCA"

EXPERIMENT_REGISTRY = (
    ExperimentConfig(
        experiment_id="B0_ROI_HIER_LR_PCA",
        description=(
            "Historical original-canvas reference: confidence-gated MONAI ROI, "
            "equal slice weights, hierarchical embedding pooling, PCA and "
            "Logistic Regression."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="historical_baseline",
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
            "Patient classifier using only the conservative export/provenance "
            "subset: native geometry, file-size-per-pixel, padding and border "
            "statistics. Central intensity, entropy and sharpness are excluded."
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
    ExperimentConfig(
        experiment_id="A8_ROI_HIER_LR_NO_PCA_FIXEDC",
        description=(
            "Matched patient-embedding reference for the legacy slice-classifier "
            "ablation: no PCA and fixed C=1.0, so A8 versus A3 isolates the "
            "training/pooling strategy rather than changing PCA or C selection."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=False,
        fixed_c=1.0,
        role="matched_reference",
    ),
    # ------------------------------------------------------------------
    # DECONFOUNDING EXTENSION
    # ------------------------------------------------------------------
    # These experiments were added after the first complete suite showed that
    # border-only, outside-mask, and provenance-only controls retained
    # substantial predictive signal. They preserve the original B0 result for
    # historical comparison but introduce a new label-blind standardized
    # baseline and controls that isolate padding, corners, central cropping,
    # stricter ROI removal, and preprocessing geometry.
    ExperimentConfig(
        experiment_id="B1_STANDARDIZED_ROI_HIER_LR_PCA",
        description=(
            "Primary deconfounded baseline: label-blind dark-padding removal, "
            "robust percentile intensity scaling, standardized MONAI ROI, "
            "hierarchical embedding pooling, PCA and Logistic Regression."
        ),
        feature_mode="standardized_monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="baseline",
    ),
    ExperimentConfig(
        experiment_id="A9_STANDARDIZED_FULL_HIER_LR_PCA",
        description=(
            "Label-blind standardized full-image ablation with all downstream "
            "settings matched to the deconfounded baseline."
        ),
        feature_mode="standardized_full_image",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
    ),
    ExperimentConfig(
        experiment_id="C5_STANDARDIZED_BORDER05_HIER_LR_PCA",
        description=(
            "Negative control retaining only the outer 5% of the standardized "
            "image. High AUC indicates residual padding/export shortcut risk."
        ),
        feature_mode="standardized_border_05",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C6_STANDARDIZED_BORDER10_HIER_LR_PCA",
        description=(
            "Negative control retaining only the outer 10% of the standardized "
            "image. This is stricter than the original 15% border control."
        ),
        feature_mode="standardized_border_10",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C7_DETECTED_PADDING_MASK_HIER_LR_PCA",
        description=(
            "Negative control using only a binary mask of label-blind detected "
            "native dark padding plus pipeline-added canvas padding."
        ),
        feature_mode="detected_padding_mask",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C8_STANDARDIZED_CORNERS_HIER_LR_PCA",
        description=(
            "Negative control retaining only four standardized-image corners; "
            "it targets scanner overlays, crop geometry, and export templates."
        ),
        feature_mode="standardized_corners",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA",
        description=(
            "Area-matched central-crop control. It tests whether MONAI adds "
            "anatomical localization beyond simply concentrating on the center."
        ),
        feature_mode="standardized_center_crop",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="localization_control",
    ),
    ExperimentConfig(
        experiment_id="A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA",
        description=(
            "Strict standardized soft ROI with zero background for valid MONAI "
            "masks and full-image fallback for invalid masks."
        ),
        feature_mode="standardized_roi_zero_background",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
    ),
    ExperimentConfig(
        experiment_id="A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA",
        description=(
            "Strict standardized crop around the dilated MONAI hard-mask "
            "bounding box with a fixed label-blind context margin."
        ),
        feature_mode="standardized_roi_bbox",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
    ),
    ExperimentConfig(
        experiment_id="C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA",
        description=(
            "Strict negative control retaining only pixels outside an enlarged "
            "MONAI bounding box; invalid masks yield a zero image rather than a "
            "full-image fallback."
        ),
        feature_mode="standardized_outside_large_bbox",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C11_STANDARDIZATION_QC_ONLY_LR",
        description=(
            "Patient classifier using only label-blind crop fractions and robust "
            "intensity-scaling limits generated by standardization."
        ),
        feature_mode="standardization_qc_only",
        strategy="patient_tabular",
        pooling_strategy="patient_tabular",
        weighting_mode="not_applicable",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C12_STANDARDIZED_MONAI_QC_ONLY_LR",
        description=(
            "Patient classifier using only MONAI gate/QC statistics produced "
            "after label-blind standardization."
        ),
        feature_mode="standardized_monai_qc_only",
        strategy="patient_tabular",
        pooling_strategy="patient_tabular",
        weighting_mode="not_applicable",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="R1_ROI_HIER_LR_PCA_DROP10",
        description=(
            "Exploratory robustness control after deterministic 10% slice "
            "dropout within each series proxy; at least one slice per proxy is retained."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        slice_dropout_rate=0.10,
        role="robustness",
    ),
    ExperimentConfig(
        experiment_id="R2_ROI_HIER_LR_PCA_DROP25",
        description=(
            "Exploratory robustness control after deterministic 25% slice "
            "dropout within each series proxy; at least one slice per proxy is retained."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        slice_dropout_rate=0.25,
        role="robustness",
    ),
    ExperimentConfig(
        experiment_id="R3_ROI_HIER_LR_PCA_DROP50",
        description=(
            "Exploratory robustness control after deterministic 50% slice "
            "dropout within each series proxy; at least one slice per proxy is retained."
        ),
        feature_mode="monai_roi",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        slice_dropout_rate=0.50,
        role="robustness",
    ),
)

# These comparisons are declared before evaluation. The first identifier is the
# reference and the second is the changed configuration, so Delta-AUROC is
# calculated as ``comparison - reference``. A8 exists specifically to make the
# legacy strategy comparison clean: A8 and A3 use the same ROI features, equal
# weights, Logistic Regression, no PCA, and fixed C=1.0.
PRIMARY_ABLATION_COMPARISONS = (
    (
        "ORIGINAL_ROI_VS_LABEL_BLIND_STANDARDIZED_ROI",
        "B0_ROI_HIER_LR_PCA",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "How does label-blind padding removal and robust scaling change the original ROI baseline?",
    ),
    (
        "STANDARDIZED_ROI_VS_STANDARDIZED_FULL_IMAGE",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "A9_STANDARDIZED_FULL_HIER_LR_PCA",
        "After export standardization, does MONAI ROI still improve over the full image?",
    ),
    (
        "STANDARDIZED_ROI_VS_AREA_MATCHED_CENTER_CROP",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA",
        "Does MONAI localization outperform a similarly sized central crop?",
    ),
    (
        "STANDARDIZED_SOFT_ROI_VS_ZERO_BACKGROUND",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA",
        "Is retaining 15 percent background context necessary after standardization?",
    ),
    (
        "STANDARDIZED_SOFT_ROI_VS_BOUNDING_BOX_CROP",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA",
        "Does a stricter MONAI bounding-box crop preserve useful patient signal?",
    ),
    (
        "STANDARDIZED_ROI_VS_OUTSIDE_LARGE_BOUNDING_BOX",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA",
        "How much predictive signal remains after removing an enlarged MONAI region?",
    ),
    (
        "STANDARDIZED_ROI_VS_STANDARDIZED_BORDER_05",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "C5_STANDARDIZED_BORDER05_HIER_LR_PCA",
        "Can the outer 5 percent of standardized images still classify the cohort?",
    ),
    (
        "STANDARDIZED_ROI_VS_STANDARDIZED_BORDER_10",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "C6_STANDARDIZED_BORDER10_HIER_LR_PCA",
        "Can the outer 10 percent of standardized images still classify the cohort?",
    ),
    (
        "STANDARDIZED_ROI_VS_DETECTED_PADDING_MASK",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "C7_DETECTED_PADDING_MASK_HIER_LR_PCA",
        "Does detected padding geometry alone encode the class label?",
    ),
    (
        "STANDARDIZED_ROI_VS_STANDARDIZED_CORNERS",
        "B1_STANDARDIZED_ROI_HIER_LR_PCA",
        "C8_STANDARDIZED_CORNERS_HIER_LR_PCA",
        "Do image corners retain scanner/export shortcut information?",
    ),
    (
        "ROI_VS_FULL_IMAGE",
        "B0_ROI_HIER_LR_PCA",
        "A1_FULL_HIER_LR_PCA",
        "Does confidence-gated MONAI ROI improve patient-level discrimination?",
    ),
    (
        "HIERARCHICAL_VS_FLAT_POOLING",
        "B0_ROI_HIER_LR_PCA",
        "A2_ROI_FLAT_LR_PCA",
        "Does series-aware hierarchical embedding pooling improve over a flat slice mean?",
    ),
    (
        "PATIENT_EMBEDDING_VS_LEGACY_SLICE_CLASSIFIER",
        "A8_ROI_HIER_LR_NO_PCA_FIXEDC",
        "A3_ROI_LEGACY_LOGODDS_LR",
        "What changes when one-patient-row training is replaced by repeated-label slice training?",
    ),
    (
        "LOGODDS_VS_MEAN_PROBABILITY_FUSION",
        "A3_ROI_LEGACY_LOGODDS_LR",
        "A4_ROI_LEGACY_MEANPROB_LR",
        "Within the same legacy slice classifier, does mean-probability fusion differ from log-odds fusion?",
    ),
    (
        "LOGISTIC_REGRESSION_VS_LINEAR_SVM",
        "B0_ROI_HIER_LR_PCA",
        "A5_ROI_HIER_LINEAR_SVM_PCA",
        "Does the result depend on the selected linear classifier family?",
    ),
    (
        "EQUAL_VS_QUALITY_SLICE_WEIGHTS",
        "B0_ROI_HIER_LR_PCA",
        "A6_ROI_HIER_LR_QUALITY_PCA",
        "Does the optional series-local intensity-variation heuristic improve pooling?",
    ),
    (
        "PCA_ON_VS_PCA_OFF",
        "B0_ROI_HIER_LR_PCA",
        "A7_ROI_HIER_LR_NO_PCA",
        "Does fold-local PCA improve the high-dimensional small-patient setting?",
    ),
    (
        "ROBUSTNESS_AFTER_10_PERCENT_SLICE_DROPOUT",
        "B0_ROI_HIER_LR_PCA",
        "R1_ROI_HIER_LR_PCA_DROP10",
        "How stable is the patient model after deterministic 10% within-series slice removal?",
    ),
    (
        "ROBUSTNESS_AFTER_25_PERCENT_SLICE_DROPOUT",
        "B0_ROI_HIER_LR_PCA",
        "R2_ROI_HIER_LR_PCA_DROP25",
        "How stable is the patient model after deterministic 25% within-series slice removal?",
    ),
    (
        "ROBUSTNESS_AFTER_50_PERCENT_SLICE_DROPOUT",
        "B0_ROI_HIER_LR_PCA",
        "R3_ROI_HIER_LR_PCA_DROP50",
        "How stable is the patient model after deterministic 50% within-series slice removal?",
    ),
)

N_SPLITS = 5
CV_RANDOM_STATE = RANDOM_SEED
INNER_CV_SPLITS = 3
INNER_CV_RANDOM_STATE = RANDOM_SEED + 1000

CLASSIFIER_C_GRID = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_MAX_ITER = 4000
SVM_MAX_ITER = 20000
SVM_CALIBRATION_C = 1.0
# The sigmoid calibrator is fitted on inner OOF patient scores without class
# balancing, so it targets the prevalence of the outer-training cohort rather
# than an artificial 50/50 prior. The amount of regularization is predeclared.
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
FEATURE_CACHE_SCHEMA_VERSION = "2026-08-31-deconfounding-v3"
EFFICIENTNET_FEATURE_DIM = 1280
FEATURE_MODES_PER_ENCODER_CALL = 4
# Several image variants can be concatenated along the batch dimension and
# encoded by one EfficientNet call. Four modes at a time is a conservative T4
# default: it reduces Python/kernel-launch overhead without materializing all
# fourteen views simultaneously. Lower this value if GPU memory is insufficient.

SLICE_QUALITY_MIN_WEIGHT = 0.25
# The quality signal remains a non-clinical heuristic. It is computed once from
# the confidence-gated ROI/fallback image and activated only by experiment A6.

BORDER_WIDTH_FRACTION = 0.15
# C1 retains only this outer fraction on each side of the original 224x224
# image so the first suite result remains directly reproducible.

STANDARDIZED_BORDER_WIDTH_FRACTIONS = (0.05, 0.10)
STANDARDIZED_CORNER_WIDTH_FRACTION = 0.15
CENTER_CROP_FALLBACK_FRACTION = 0.60
MONAI_BBOX_CONTEXT_FRACTION = 0.15
OUTSIDE_MONAI_BBOX_CONTEXT_FRACTION = 0.30
# New controls use narrower borders, isolated corners, an area-matched center
# crop, and strict MONAI bounding-box views. All fractions are fixed before
# evaluation and are independent of labels and OOF performance.

STANDARDIZATION_LOWER_PERCENTILE = 1.0
STANDARDIZATION_UPPER_PERCENTILE = 99.0
STANDARDIZATION_DARK_LINE_MAX_MEAN = 12.0
STANDARDIZATION_DARK_LINE_MAX_STD = 4.0
STANDARDIZATION_DARK_PIXEL_MAX_VALUE = 20
STANDARDIZATION_DARK_PIXEL_MIN_FRACTION = 0.98
STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE = 0.20
STANDARDIZATION_MIN_RETAINED_FRACTION = 0.60
STANDARDIZATION_MIN_PADDING_RUN = 2
STANDARDIZATION_FEATURE_NAMES = (
    "crop_applied",
    "crop_top_fraction",
    "crop_bottom_fraction",
    "crop_left_fraction",
    "crop_right_fraction",
    "retained_height_fraction",
    "retained_width_fraction",
    "detected_padding_fraction",
    "robust_lower_intensity_0_1",
    "robust_upper_intensity_0_1",
    "robust_dynamic_range_0_1",
)
# Label-blind standardization removes only consecutive edge rows/columns that
# are nearly uniform and dark. Safety limits prevent aggressive cropping. The
# retained image is scaled with robust 1st/99th percentiles before the existing
# aspect-ratio-preserving 256x256 canvas operation.

C_SELECTION_AUC_TOLERANCE = 0.01
# Select the smallest (most regularized) C whose inner AUC is within this
# absolute tolerance of the best candidate. This prevents tiny inner-CV
# differences from repeatedly choosing the least regularized edge of the grid.

RUN_REPEATED_NESTED_CV_STABILITY = True
REPEATED_NESTED_CV_REPEATS = 10
REPEATED_NESTED_CV_RANDOM_STATE = RANDOM_SEED + 20_000
STABILITY_EXPERIMENT_ID = BASELINE_EXPERIMENT_ID

RUN_PATIENT_LABEL_PERMUTATION_TEST = True
LABEL_PERMUTATION_REPLICATES = 200
LABEL_PERMUTATION_RANDOM_STATE = RANDOM_SEED + 40_000
PERMUTATION_EXPERIMENT_ID = BASELINE_EXPERIMENT_ID
# Stability repeats and label permutation operate only on the selected primary
# patient-embedding baseline after the shared feature bank is created. They do
# not rerun MONAI or EfficientNet.

AUDIT_EXACT_DECODED_PIXEL_DUPLICATES = True
AUDIT_PERCEPTUAL_NEAR_DUPLICATES = True
PHASH_HAMMING_THRESHOLD = 3
PHASH_USE_COMPLETE_BK_TREE_AUDIT = True
# The reviewed audit searches unique 64-bit hashes with a complete BK-tree
# radius query. It does not skip large buckets or truncate at an arbitrary
# image-pair count. pHash matches remain screening candidates, not verdicts.

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
    "primary_ablation_comparisons": PRIMARY_ABLATION_COMPARISONS,
    "random_seed": RANDOM_SEED,
    "cv_random_state": CV_RANDOM_STATE,
    "inner_cv_random_state": INNER_CV_RANDOM_STATE,
    "n_splits": N_SPLITS,
    "inner_splits": INNER_CV_SPLITS,
    "c_grid": CLASSIFIER_C_GRID,
    "c_selection_auc_tolerance": C_SELECTION_AUC_TOLERANCE,
    "logistic_max_iter": LOGISTIC_MAX_ITER,
    "svm_max_iter": SVM_MAX_ITER,
    "svm_calibration_c": SVM_CALIBRATION_C,
    "threshold_method": THRESHOLD_SELECTION_METHOD,
    "target_sensitivity": TARGET_SENSITIVITY,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "paired_bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
    "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
    "pca_variance": PATIENT_PCA_EXPLAINED_VARIANCE,
    "slice_quality_min_weight": SLICE_QUALITY_MIN_WEIGHT,
    "feature_cache_schema": FEATURE_CACHE_SCHEMA_VERSION,
    "batch_size": BATCH_SIZE,
    "use_cuda_amp": USE_CUDA_AMP,
    "device_type": DEVICE,
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
    "standardized_border_width_fractions": STANDARDIZED_BORDER_WIDTH_FRACTIONS,
    "standardized_corner_width_fraction": STANDARDIZED_CORNER_WIDTH_FRACTION,
    "center_crop_fallback_fraction": CENTER_CROP_FALLBACK_FRACTION,
    "monai_bbox_context_fraction": MONAI_BBOX_CONTEXT_FRACTION,
    "outside_monai_bbox_context_fraction": (
        OUTSIDE_MONAI_BBOX_CONTEXT_FRACTION
    ),
    "standardization_lower_percentile": STANDARDIZATION_LOWER_PERCENTILE,
    "standardization_upper_percentile": STANDARDIZATION_UPPER_PERCENTILE,
    "standardization_dark_line_max_mean": (
        STANDARDIZATION_DARK_LINE_MAX_MEAN
    ),
    "standardization_dark_line_max_std": STANDARDIZATION_DARK_LINE_MAX_STD,
    "standardization_dark_pixel_max_value": (
        STANDARDIZATION_DARK_PIXEL_MAX_VALUE
    ),
    "standardization_dark_pixel_min_fraction": (
        STANDARDIZATION_DARK_PIXEL_MIN_FRACTION
    ),
    "standardization_max_crop_fraction_per_side": (
        STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE
    ),
    "standardization_min_retained_fraction": (
        STANDARDIZATION_MIN_RETAINED_FRACTION
    ),
    "standardization_min_padding_run": STANDARDIZATION_MIN_PADDING_RUN,
    "standardization_feature_names": STANDARDIZATION_FEATURE_NAMES,
    "provenance_classifier_schema": "geometry_file_padding_border_v1",
    "audit_exact_duplicates": AUDIT_EXACT_DECODED_PIXEL_DUPLICATES,
    "audit_perceptual_duplicates": AUDIT_PERCEPTUAL_NEAR_DUPLICATES,
    "phash_hamming_threshold": PHASH_HAMMING_THRESHOLD,
    "phash_use_complete_bk_tree_audit": PHASH_USE_COMPLETE_BK_TREE_AUDIT,
    "group_exact_duplicates": GROUP_SPLITS_BY_EXACT_DUPLICATES,
    "group_phash_candidates": GROUP_SPLITS_BY_PHASH_CANDIDATES,
    "fail_on_cross_patient_exact_duplicates": (
        FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES
    ),
    "fail_on_cross_label_exact_duplicates": (
        FAIL_ON_CROSS_LABEL_EXACT_DUPLICATES
    ),
    "shortcut_warning_auc": SHORTCUT_WARNING_AUC,
    "monai_gate_rate_difference_warning": (
        MONAI_GATE_RATE_DIFFERENCE_WARNING
    ),
    "run_repeated_nested_cv_stability": RUN_REPEATED_NESTED_CV_STABILITY,
    "repeated_nested_cv_repeats": REPEATED_NESTED_CV_REPEATS,
    "repeated_nested_cv_random_state": REPEATED_NESTED_CV_RANDOM_STATE,
    "stability_experiment_id": STABILITY_EXPERIMENT_ID,
    "run_patient_label_permutation_test": (
        RUN_PATIENT_LABEL_PERMUTATION_TEST
    ),
    "label_permutation_replicates": LABEL_PERMUTATION_REPLICATES,
    "label_permutation_random_state": LABEL_PERMUTATION_RANDOM_STATE,
    "permutation_experiment_id": PERMUTATION_EXPERIMENT_ID,
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

PIPELINE_STAGE_COUNT = 13
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

    # Validate the predeclared scientific comparison registry against the full
    # experiment catalog. A reduced preliminary run may omit one side of a pair;
    # that pair will be saved as SKIPPED rather than being silently redefined.
    known_experiment_ids = {
        experiment.experiment_id for experiment in EXPERIMENT_REGISTRY
    }
    comparison_names = [row[0] for row in PRIMARY_ABLATION_COMPARISONS]
    if len(comparison_names) != len(set(comparison_names)):
        raise ValueError("Primary ablation comparison names must be unique.")
    for comparison_name, reference_id, comparison_id, _ in (
        PRIMARY_ABLATION_COMPARISONS
    ):
        if reference_id not in known_experiment_ids:
            raise ValueError(
                f"{comparison_name}: unknown reference experiment {reference_id!r}."
            )
        if comparison_id not in known_experiment_ids:
            raise ValueError(
                f"{comparison_name}: unknown comparison experiment {comparison_id!r}."
            )
        if reference_id == comparison_id:
            raise ValueError(
                f"{comparison_name}: reference and comparison must differ."
            )

    if BASELINE_EXPERIMENT_ID not in experiment_ids:
        raise ValueError(
            "The configured baseline must be present in the enabled suite: "
            f"{BASELINE_EXPERIMENT_ID!r}."
        )

    if RUN_REPEATED_NESTED_CV_STABILITY or RUN_PATIENT_LABEL_PERMUTATION_TEST:
        enabled_by_id = {
            experiment.experiment_id: experiment for experiment in experiments
        }
        requested_analysis_ids = set()
        if RUN_REPEATED_NESTED_CV_STABILITY:
            requested_analysis_ids.add(STABILITY_EXPERIMENT_ID)
        if RUN_PATIENT_LABEL_PERMUTATION_TEST:
            requested_analysis_ids.add(PERMUTATION_EXPERIMENT_ID)

        for analysis_experiment_id in sorted(requested_analysis_ids):
            if analysis_experiment_id not in enabled_by_id:
                raise ValueError(
                    "The stability/permutation experiment must be enabled: "
                    f"{analysis_experiment_id!r}."
                )
            analysis_experiment = enabled_by_id[analysis_experiment_id]
            if analysis_experiment.strategy != "patient_embedding":
                raise ValueError(
                    "Stability and patient-label permutation are reviewed only "
                    "for patient-embedding experiments."
                )
            if analysis_experiment.classifier_type not in {
                "logistic_regression",
                "linear_svm",
            }:
                raise ValueError(
                    "Unsupported classifier for stability/permutation analysis."
                )

    valid_feature_modes = {
        "monai_roi",
        "full_image",
        "border_only",
        "outside_heart",
        "standardized_monai_roi",
        "standardized_full_image",
        "standardized_border_05",
        "standardized_border_10",
        "detected_padding_mask",
        "standardized_corners",
        "standardized_center_crop",
        "standardized_roi_zero_background",
        "standardized_roi_bbox",
        "standardized_outside_large_bbox",
        "provenance_only",
        "monai_qc_only",
        "standardization_qc_only",
        "standardized_monai_qc_only",
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
        if not 0.0 <= experiment.slice_dropout_rate < 1.0:
            raise ValueError(
                f"{experiment.experiment_id}: slice_dropout_rate must lie "
                "in [0,1)."
            )
        if experiment.slice_dropout_rate > 0.0 and (
            experiment.strategy != "patient_embedding"
            or experiment.feature_mode not in {
                "monai_roi",
                "full_image",
                "border_only",
                "outside_heart",
                "standardized_monai_roi",
                "standardized_full_image",
                "standardized_roi_zero_background",
                "standardized_roi_bbox",
            }
        ):
            raise ValueError(
                f"{experiment.experiment_id}: slice dropout is reviewed only "
                "for image-based patient-embedding experiments."
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
                "standardization_qc_only",
                "standardized_monai_qc_only",
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
    if SVM_CALIBRATION_C <= 0:
        raise ValueError("SVM_CALIBRATION_C must be positive.")
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
    if any(
        not 0.0 < float(value) < 0.5
        for value in STANDARDIZED_BORDER_WIDTH_FRACTIONS
    ):
        raise ValueError(
            "STANDARDIZED_BORDER_WIDTH_FRACTIONS must lie in (0,0.5)."
        )
    if not 0.0 < STANDARDIZED_CORNER_WIDTH_FRACTION < 0.5:
        raise ValueError(
            "STANDARDIZED_CORNER_WIDTH_FRACTION must lie in (0,0.5)."
        )
    if not 0.0 < CENTER_CROP_FALLBACK_FRACTION <= 1.0:
        raise ValueError(
            "CENTER_CROP_FALLBACK_FRACTION must lie in (0,1]."
        )
    if MONAI_BBOX_CONTEXT_FRACTION < 0.0 or (
        OUTSIDE_MONAI_BBOX_CONTEXT_FRACTION < 0.0
    ):
        raise ValueError("MONAI bounding-box context fractions cannot be negative.")
    if not (
        0.0 <= STANDARDIZATION_LOWER_PERCENTILE
        < STANDARDIZATION_UPPER_PERCENTILE <= 100.0
    ):
        raise ValueError("Invalid robust standardization percentile interval.")
    if not 0.0 < STANDARDIZATION_MIN_RETAINED_FRACTION <= 1.0:
        raise ValueError(
            "STANDARDIZATION_MIN_RETAINED_FRACTION must lie in (0,1]."
        )
    if not 0.0 <= STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE < 0.5:
        raise ValueError(
            "STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE must lie in [0,0.5)."
        )
    if STANDARDIZATION_MIN_PADDING_RUN < 1:
        raise ValueError("STANDARDIZATION_MIN_PADDING_RUN must be positive.")
    if C_SELECTION_AUC_TOLERANCE < 0.0:
        raise ValueError("C_SELECTION_AUC_TOLERANCE cannot be negative.")
    if REPEATED_NESTED_CV_REPEATS <= 0:
        raise ValueError("REPEATED_NESTED_CV_REPEATS must be positive.")
    if LABEL_PERMUTATION_REPLICATES <= 0:
        raise ValueError("LABEL_PERMUTATION_REPLICATES must be positive.")
    if FEATURE_MODES_PER_ENCODER_CALL <= 0:
        raise ValueError("FEATURE_MODES_PER_ENCODER_CALL must be positive.")

    if PHASH_HAMMING_THRESHOLD < 0 or PHASH_HAMMING_THRESHOLD > 64:
        raise ValueError("PHASH_HAMMING_THRESHOLD must lie in [0,64].")
    if not PHASH_USE_COMPLETE_BK_TREE_AUDIT:
        raise ValueError(
            "This reviewed version requires the complete BK-tree pHash audit. "
            "Keep PHASH_USE_COMPLETE_BK_TREE_AUDIT=True."
        )

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



def _is_dark_uniform_edge_line(line):
    """Return True when one native edge row/column is almost uniformly dark.

    The rule is intentionally simple, deterministic, and label-blind. It is
    designed to identify scanner/export padding, not to segment anatomy. A line
    must satisfy all three conditions: low mean, low standard deviation, and a
    very high fraction of pixels below a fixed uint8 intensity threshold.
    """

    values = np.asarray(line, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return False

    return bool(
        float(values.mean()) <= STANDARDIZATION_DARK_LINE_MAX_MEAN
        and float(values.std()) <= STANDARDIZATION_DARK_LINE_MAX_STD
        and float(
            np.mean(values <= STANDARDIZATION_DARK_PIXEL_MAX_VALUE)
        ) >= STANDARDIZATION_DARK_PIXEL_MIN_FRACTION
    )


def detect_label_blind_dark_padding_bounds(image):
    """Detect conservative dark-uniform padding bounds on a native uint8 image.

    Returns ``(top, bottom, left, right)`` using NumPy slicing semantics, where
    ``bottom`` and ``right`` are exclusive. Only consecutive qualifying lines
    beginning at the four image edges can be removed. The detector cannot crop
    more than a fixed fraction from any side and must retain at least the
    configured fraction of the original height and width.

    This operation never reads the Normal/Sick label, patient ID, series ID,
    model score, or fold assignment. Its purpose is to reduce obvious export
    padding while preserving a separately auditable record of the crop geometry.
    """

    if image.ndim != 2:
        raise ValueError(
            f"Padding detection expects a 2D grayscale image, got {image.shape}."
        )

    height, width = image.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid native image shape: {image.shape}.")

    minimum_height = max(
        8,
        int(np.ceil(height * STANDARDIZATION_MIN_RETAINED_FRACTION)),
    )
    minimum_width = max(
        8,
        int(np.ceil(width * STANDARDIZATION_MIN_RETAINED_FRACTION)),
    )
    maximum_vertical_crop = int(
        np.floor(height * STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE)
    )
    maximum_horizontal_crop = int(
        np.floor(width * STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE)
    )

    top_crop = 0
    while (
        top_crop < maximum_vertical_crop
        and height - (top_crop + 1) >= minimum_height
        and _is_dark_uniform_edge_line(image[top_crop, :])
    ):
        top_crop += 1

    bottom_crop = 0
    while (
        bottom_crop < maximum_vertical_crop
        and height - top_crop - (bottom_crop + 1) >= minimum_height
        and _is_dark_uniform_edge_line(image[height - 1 - bottom_crop, :])
    ):
        bottom_crop += 1

    left_crop = 0
    while (
        left_crop < maximum_horizontal_crop
        and width - (left_crop + 1) >= minimum_width
        and _is_dark_uniform_edge_line(image[:, left_crop])
    ):
        left_crop += 1

    right_crop = 0
    while (
        right_crop < maximum_horizontal_crop
        and width - left_crop - (right_crop + 1) >= minimum_width
        and _is_dark_uniform_edge_line(image[:, width - 1 - right_crop])
    ):
        right_crop += 1

    # One isolated dark line can occur naturally or through interpolation. The
    # minimum-run rule avoids declaring it an export border without repetition.
    if top_crop < STANDARDIZATION_MIN_PADDING_RUN:
        top_crop = 0
    if bottom_crop < STANDARDIZATION_MIN_PADDING_RUN:
        bottom_crop = 0
    if left_crop < STANDARDIZATION_MIN_PADDING_RUN:
        left_crop = 0
    if right_crop < STANDARDIZATION_MIN_PADDING_RUN:
        right_crop = 0

    if height - top_crop - bottom_crop < minimum_height:
        top_crop = 0
        bottom_crop = 0
    if width - left_crop - right_crop < minimum_width:
        left_crop = 0
        right_crop = 0

    top = int(top_crop)
    bottom = int(height - bottom_crop)
    left = int(left_crop)
    right = int(width - right_crop)

    if bottom <= top or right <= left:
        # This should be impossible under the safety checks, but returning the
        # full image is safer than allowing an invalid crop to propagate.
        return 0, height, 0, width

    return top, bottom, left, right


def robust_scale_intensity_0_1(image):
    """Scale a cropped grayscale image with fixed robust percentiles.

    The ordinary baseline keeps the original per-image min-max transformation
    for direct reproducibility. The deconfounded branch instead clips to the
    predeclared 1st/99th percentile limits so a small number of bright text
    pixels or compression outliers cannot define the complete dynamic range.

    Returns the scaled image plus the two native-intensity percentile values so
    the transformation can be audited at image and patient level.
    """

    image_float = np.asarray(image, dtype=np.float32)
    if image_float.ndim != 2 or image_float.size == 0:
        raise ValueError(
            "Robust intensity scaling requires a non-empty 2D grayscale image."
        )

    lower = float(
        np.percentile(image_float, STANDARDIZATION_LOWER_PERCENTILE)
    )
    upper = float(
        np.percentile(image_float, STANDARDIZATION_UPPER_PERCENTILE)
    )

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        minimum = float(image_float.min())
        maximum = float(image_float.max())
        if maximum <= minimum:
            return np.zeros_like(image_float, dtype=np.float32), minimum, maximum
        lower, upper = minimum, maximum

    scaled = np.clip(image_float, lower, upper)
    scaled = (scaled - lower) / max(upper - lower, 1e-8)
    return scaled.astype(np.float32, copy=False), lower, upper


def zero_pad_binary_mask_to_monai_canvas(mask):
    """Place a binary native mask in the same 256x256 geometry as its image.

    The canvas is initialized to one because pixels introduced by the pipeline's
    own square padding are also padding. Inside the transformed native field of
    view, the supplied mask marks detected native padding with one and retained
    content with zero. Nearest-neighbor interpolation preserves this binary
    interpretation when an oversized source image must be downscaled.
    """

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D padding mask, got {mask.shape}.")

    height, width = mask.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid padding-mask dimensions: {mask.shape}.")

    scale = min(
        1.0,
        MONAI_INPUT_SIZE / height,
        MONAI_INPUT_SIZE / width,
    )
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))

    if resized_height != height or resized_width != width:
        resized = cv2.resize(
            mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized = mask

    canvas = np.ones(
        (MONAI_INPUT_SIZE, MONAI_INPUT_SIZE),
        dtype=np.float32,
    )
    top = (MONAI_INPUT_SIZE - resized_height) // 2
    left = (MONAI_INPUT_SIZE - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return np.clip(canvas, 0.0, 1.0)


def build_label_blind_standardized_image(image):
    """Create the standardized image, padding mask, and audit feature vector.

    Processing order:

        native uint8 JPEG
            -> conservative dark-uniform edge detection
            -> crop only the detected edge runs
            -> robust fixed-percentile intensity scaling
            -> separate binary padding-geometry mask

    The returned standardized image is not yet resized or padded; the existing
    ``zero_pad_to_monai_canvas`` function performs that shared geometry step.
    """

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"Standardization expects a 2D grayscale image, got {image.shape}."
        )

    height, width = image.shape
    top, bottom, left, right = detect_label_blind_dark_padding_bounds(image)
    cropped = image[top:bottom, left:right]
    scaled, lower, upper = robust_scale_intensity_0_1(cropped)

    native_padding_mask = np.ones((height, width), dtype=np.float32)
    native_padding_mask[top:bottom, left:right] = 0.0
    padding_canvas = zero_pad_binary_mask_to_monai_canvas(native_padding_mask)

    top_fraction = float(top / height)
    bottom_fraction = float((height - bottom) / height)
    left_fraction = float(left / width)
    right_fraction = float((width - right) / width)
    retained_height_fraction = float((bottom - top) / height)
    retained_width_fraction = float((right - left) / width)
    detected_padding_fraction = float(
        1.0 - retained_height_fraction * retained_width_fraction
    )

    features = np.asarray(
        [
            float(any(value > 0 for value in (top, height - bottom, left, width - right))),
            top_fraction,
            bottom_fraction,
            left_fraction,
            right_fraction,
            retained_height_fraction,
            retained_width_fraction,
            detected_padding_fraction,
            float(lower / 255.0),
            float(upper / 255.0),
            float(max(upper - lower, 0.0) / 255.0),
        ],
        dtype=np.float32,
    )

    if len(features) != len(STANDARDIZATION_FEATURE_NAMES):
        raise RuntimeError(
            "Standardization feature-name and value counts differ."
        )
    if not np.all(np.isfinite(features)):
        raise RuntimeError("Standardization features contain non-finite values.")

    return scaled, padding_canvas, features


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


# The complete list above is written to the cohort manifest for descriptive
# audit. The provenance-only classifier intentionally uses a narrower subset
# that is dominated by export geometry, file/container properties, padding, and
# border style. Global intensity, center intensity, entropy, edge density, and
# sharpness are excluded from that classifier because they can contain genuine
# anatomical or pathology-related signal and would make the term "provenance
# only" too strong.
PROVENANCE_CLASSIFIER_FEATURE_NAMES = (
    "native_height",
    "native_width",
    "aspect_ratio_width_over_height",
    "file_size_bytes",
    "bytes_per_native_pixel",
    "near_black_fraction",
    "near_white_fraction",
    "border_mean_intensity",
    "border_std_intensity",
    "border_near_black_fraction",
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
    """
    PIPELINE STEP 1:
    Load one JPEG MRI slice and construct two spatially aligned network inputs
    plus the label-free audit metadata needed by the multi-experiment suite.

    Returns:
        classification_image:
            Tensor [3, 224, 224], values in [0,1].
            Used by EfficientNet only after the selected image representation
            has been created and ImageNet normalization has been applied.

        monai_image:
            Tensor [1, 256, 256], values in [0,1].
            Original min-max baseline input for the MONAI segmenter.

        standardized_classification_image:
            Tensor [3, 224, 224], values in [0,1]. Label-blind padding removal
            and robust percentile scaling are applied before square placement.

        standardized_monai_image:
            Tensor [1, 256, 256], values in [0,1]. Spatially aligned MONAI input
            for the deconfounded standardized branch.

        detected_padding_image:
            Tensor [3, 224, 224], binary. It contains only detected native and
            pipeline-added padding geometry for a strict negative control.

        label:
            0 for Normal, 1 for Sick. The label travels as metadata only and is
            never supplied to MONAI, EfficientNet, image hashes, or provenance
            feature extraction.

        patient_id:
            The validated patient identifier itself, for example Directory_24.

        series_id:
            Patient-scoped folder-defined series proxy, for example
            Directory_24/SR_3. This is not a recovered DICOM UID.

        sample_index:
            Stable integer position in ``samples``. It preserves one-to-one row
            alignment across feature-bank arrays and deterministic debug images.

        decoded_pixel_hash:
            SHA-256 of the decoded uint8 grayscale pixel matrix plus its native
            shape. It supports an exact-pixel duplicate audit; it is not a
            perceptual or near-duplicate hash.

        perceptual_hash:
            A deterministic 64-bit DCT pHash used only to generate candidate
            cross-patient near-duplicate pairs. A small Hamming distance is a
            screening signal and is not proof that two images are duplicates.

        provenance_features:
            Label-free native image/export features such as dimensions, file
            size, intensity distribution, border statistics, edge fraction and
            sharpness. These features support the provenance-only negative
            control; they are not fed to the image encoder.

        standardization_features:
            Label-free crop fractions and robust intensity limits generated by
            the standardized branch. They support a separate QC-only control.

    =========================================================================
    WHY TWO INPUT TENSORS?
    =========================================================================

    The two pretrained networks were built around different input conventions:

        MONAI cardiac segmenter:
            - one grayscale channel
            - 256×256
            - intensity range [0,1]

        EfficientNet-B0:
            - three channels
            - 224×224
            - ImageNet mean/std normalization

    Reusing one already-normalized tensor for both networks would violate at
    least one model's expected input distribution. Therefore segmentation and
    classification preprocessing remain explicitly separate.

    =========================================================================
    SPATIAL ALIGNMENT
    =========================================================================

    The original MRI is first placed in a 256×256 square canvas. The 224×224
    classifier image is created by uniformly resizing that complete square.
    Consequently, a MONAI probability map resized from 256×256 to 224×224 aligns
    with the EfficientNet image without requiring DICOM geometry metadata.

    =========================================================================
    WHY AUDIT METADATA IS CREATED DURING THE SAME DECODE?
    =========================================================================

    Native image dimensions, exact hashes, pHash and export-style features must
    be calculated before normalization/resizing destroys that information.
    Computing them here avoids a second complete JPEG-decoding pass and keeps
    every audit row aligned with the frozen feature-bank row for the same file.
    """

    def __init__(self, samples, transform=None):

        # Store only lightweight paths and immutable metadata. JPEG decoding is
        # deferred to ``__getitem__`` so DataLoader controls when each image is
        # read and can use worker processes if that option is enabled later.
        self.samples = samples
        # List containing:
        #   (image_path, binary_label, patient_id, series_id)

        self.transform = transform
        # Classifier-side HWC NumPy -> CHW PyTorch conversion. ImageNet
        # normalization is deliberately deferred until after full/ROI/control
        # image construction.

    def __len__(self):

        # DataLoader uses this exact count to determine complete batch coverage.
        # Because extraction uses ``shuffle=False``, indices remain aligned with
        # ``samples`` and all feature-bank arrays.
        return len(self.samples)

    def __getitem__(self, idx):

        # Resolve the immutable metadata row first. Neither the binary label nor
        # the grouping identifiers influence pixels, masks, hashes, or audit
        # features.
        img_path, label, patient_id, series_id = self.samples[idx]

        # =========================================================
        # LOAD RAW GRAYSCALE MRI JPEG
        # =========================================================

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # An unreadable file is a dataset error. Silently skipping it would
            # change patient/series composition and could make cache rows drift.
            raise FileNotFoundError(
                f"OpenCV could not read the MRI image: {img_path}"
            )

        # =========================================================
        # EXACT AND PERCEPTUAL DUPLICATE KEYS
        # =========================================================

        # Hash the decoded uint8 matrix BEFORE any normalization or resizing.
        # Shape is included so two flattened byte streams with different native
        # geometry cannot be treated as equal merely because their bytes match.
        # Hashing decoded pixels, rather than JPEG container bytes, detects files
        # with different container metadata but exactly equal decoded matrices.
        pixel_digest = hashlib.sha256()
        pixel_digest.update(
            np.asarray(image.shape, dtype=np.int32).tobytes()
        )
        pixel_digest.update(image.tobytes(order="C"))
        decoded_pixel_hash = pixel_digest.hexdigest()

        # pHash deliberately tolerates small low-frequency changes and is used
        # only to generate review candidates. It must not be interpreted as a
        # validated duplicate decision without inspecting the saved examples.
        perceptual_hash = compute_dct_perceptual_hash(image)

        # =========================================================
        # NATIVE EXPORT / PROVENANCE FEATURES
        # =========================================================

        # These values are calculated on the unmodified uint8 image and file.
        # They make it possible to ask whether image dimensions, borders,
        # compression proxies or acquisition/export style alone predict class.
        provenance_features = compute_image_provenance_features(
            image,
            img_path,
        )

        # =========================================================
        # ORIGINAL AND LABEL-BLIND STANDARDIZED PREPROCESSING
        # =========================================================

        # Preserve the original baseline transformation exactly so the first
        # 16-experiment suite remains reproducible and can be compared directly
        # with the new deconfounded branch.
        original_scaled_image = scale_intensity_0_1(image)
        monai_canvas = zero_pad_to_monai_canvas(original_scaled_image)
        monai_image = torch.from_numpy(monai_canvas).unsqueeze(0)

        # Build a second, label-blind representation that removes only
        # conservative dark-uniform edge runs and uses fixed robust percentiles.
        # Crop geometry and intensity limits are returned separately for the
        # standardization-QC-only negative control.
        (
            standardized_scaled_image,
            detected_padding_canvas,
            standardization_features,
        ) = build_label_blind_standardized_image(image)
        standardized_monai_canvas = zero_pad_to_monai_canvas(
            standardized_scaled_image
        )
        standardized_monai_image = torch.from_numpy(
            standardized_monai_canvas
        ).unsqueeze(0)

        # =========================================================
        # EFFICIENTNET SPATIAL PREPROCESSING
        # =========================================================

        # Resize both complete square canvases to EfficientNet resolution. Each
        # 224x224 image remains aligned with the MONAI output generated from its
        # corresponding 256x256 canvas.
        classification_gray = cv2.resize(
            monai_canvas,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        standardized_classification_gray = cv2.resize(
            standardized_monai_canvas,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        detected_padding_gray = cv2.resize(
            detected_padding_canvas,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        # ImageNet-pretrained models expect three channels. Replicating a
        # grayscale channel adds no information but satisfies the pretrained
        # first convolution's input contract. The padding control is a binary
        # geometry image, not an anatomical MRI representation.
        classification_image = np.stack(
            [classification_gray] * 3,
            axis=-1,
        )
        standardized_classification_image = np.stack(
            [standardized_classification_gray] * 3,
            axis=-1,
        )
        detected_padding_image = np.stack(
            [detected_padding_gray] * 3,
            axis=-1,
        )

        # Convert HWC NumPy arrays to CHW tensors. ImageNet normalization is
        # still deferred until after the requested ROI/control view is created.
        if self.transform:
            classification_image = self.transform(classification_image)
            standardized_classification_image = self.transform(
                standardized_classification_image
            )
            detected_padding_image = self.transform(detected_padding_image)
        else:
            classification_image = torch.from_numpy(
                classification_image
            ).permute(2, 0, 1)
            standardized_classification_image = torch.from_numpy(
                standardized_classification_image
            ).permute(2, 0, 1)
            detected_padding_image = torch.from_numpy(
                detected_padding_image
            ).permute(2, 0, 1)

        # The default PyTorch collate function stacks tensors and keeps strings
        # as ordered lists. ``sample_index`` later places every result into its
        # deterministic feature-bank row even if DataLoader workers are used.
        return (
            classification_image,
            monai_image,
            standardized_classification_image,
            standardized_monai_image,
            detected_padding_image,
            label,
            patient_id,
            series_id,
            idx,
            decoded_pixel_hash,
            perceptual_hash,
            torch.from_numpy(provenance_features),
            torch.from_numpy(standardization_features),
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


def apply_confidence_gated_soft_roi(
    images,
    roi_probability,
    valid_mask,
    background_weight=None,
):
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

    # ``None`` preserves the original reviewed baseline. Explicit values allow
    # predeclared strict-ROI ablations without mutating the global configuration.
    if background_weight is None:
        background_weight = MONAI_BACKGROUND_WEIGHT
    background_weight = float(background_weight)
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError("background_weight must lie in [0,1].")

    # Step 2: transform probability into an attenuation field. A pixel with
    # P(heart)=0 retains the selected background weight, while a pixel with
    # P(heart)=1 retains its full intensity.
    roi_weight = (
        background_weight
        + (1.0 - background_weight) * roi_probability
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
    """
    Convert a heuristic per-slice score into bounded weights within ONE series
    proxy.

    WHY WEIGHTS INSTEAD OF DELETING SLICES?
    -----------------------------------------
    A hard top-percentage or median cutoff can remove diagnostically useful
    slices and makes the result depend on an arbitrary exclusion rule. When the
    optional quality ablation is enabled, every slice remains present and only
    its relative influence inside its own series proxy changes.

    IMPORTANT LIMITATION
    --------------------
    The score is the image-intensity standard deviation after confidence-gated
    ROI weighting/full-image fallback. It can respond to anatomy, contrast,
    noise, mask size and export style; it is NOT a validated clinical MRI image-
    quality score. Equal weights therefore remain the publication-safe baseline.

    The 25th and 75th percentiles provide a robust local scale, weights are
    clipped to a positive interval, and the final mean-one normalization is only
    a convenient within-series convention. Downstream pooling renormalizes the
    values again, so their common scale does not define model regularization.
    """

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
        # Original representations retained for direct comparison.
        "monai_roi",
        "full_image",
        "border_only",
        "outside_heart",
        # Label-blind standardized/deconfounding representations.
        "standardized_monai_roi",
        "standardized_full_image",
        "standardized_border_05",
        "standardized_border_10",
        "detected_padding_mask",
        "standardized_corners",
        "standardized_center_crop",
        "standardized_roi_zero_background",
        "standardized_roi_bbox",
        "standardized_outside_large_bbox",
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
        "batch_size": BATCH_SIZE,
        "use_cuda_amp": USE_CUDA_AMP,
        "device_type": DEVICE,
        "img_size": IMG_SIZE,
        "monai_input_size": MONAI_INPUT_SIZE,
        "monai_bundle": MONAI_BUNDLE_NAME,
        "monai_bundle_version": MONAI_BUNDLE_VERSION,
        "monai_hf_revision": MONAI_HF_REVISION,
        "monai_model_ts_sha256": MONAI_OFFICIAL_TORCHSCRIPT_SHA256,
        "monai_model_pt_sha256": MONAI_MODEL_SHA256,
        "roi_dilation": MONAI_ROI_DILATION_KERNEL,
        "roi_background": MONAI_BACKGROUND_WEIGHT,
        "roi_min_area": MONAI_MIN_HEART_AREA_RATIO,
        "roi_max_area": MONAI_MAX_HEART_AREA_RATIO,
        "roi_min_peak": MONAI_MIN_PEAK_HEART_PROBABILITY,
        "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
        "efficientnet_mean": EFFICIENTNET_MEAN,
        "efficientnet_std": EFFICIENTNET_STD,
        "border_width_fraction": BORDER_WIDTH_FRACTION,
        "standardized_border_width_fractions": STANDARDIZED_BORDER_WIDTH_FRACTIONS,
        "standardized_corner_width_fraction": STANDARDIZED_CORNER_WIDTH_FRACTION,
        "center_crop_fallback_fraction": CENTER_CROP_FALLBACK_FRACTION,
        "monai_bbox_context_fraction": MONAI_BBOX_CONTEXT_FRACTION,
        "outside_monai_bbox_context_fraction": (
            OUTSIDE_MONAI_BBOX_CONTEXT_FRACTION
        ),
        "standardization_lower_percentile": STANDARDIZATION_LOWER_PERCENTILE,
        "standardization_upper_percentile": STANDARDIZATION_UPPER_PERCENTILE,
        "standardization_dark_line_max_mean": (
            STANDARDIZATION_DARK_LINE_MAX_MEAN
        ),
        "standardization_dark_line_max_std": (
            STANDARDIZATION_DARK_LINE_MAX_STD
        ),
        "standardization_dark_pixel_max_value": (
            STANDARDIZATION_DARK_PIXEL_MAX_VALUE
        ),
        "standardization_dark_pixel_min_fraction": (
            STANDARDIZATION_DARK_PIXEL_MIN_FRACTION
        ),
        "standardization_max_crop_fraction_per_side": (
            STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE
        ),
        "standardization_min_retained_fraction": (
            STANDARDIZATION_MIN_RETAINED_FRACTION
        ),
        "standardization_min_padding_run": STANDARDIZATION_MIN_PADDING_RUN,
        "standardization_features": STANDARDIZATION_FEATURE_NAMES,
        "provenance_features": PROVENANCE_FEATURE_NAMES,
        "provenance_classifier_features": (
            PROVENANCE_CLASSIFIER_FEATURE_NAMES
        ),
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
        "standardization_features",
        "monai_valid",
        "area_ratios",
        "peak_probabilities",
        "mean_foreground_probabilities",
        "roi_slice_scores",
        "standardized_monai_valid",
        "standardized_area_ratios",
        "standardized_peak_probabilities",
        "standardized_mean_foreground_probabilities",
        "standardized_roi_slice_scores",
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


def create_border_only_images(images, border_fraction=BORDER_WIDTH_FRACTION):
    """Retain only a fixed outer border and set the central region to zero."""

    border_fraction = float(border_fraction)
    if not 0.0 < border_fraction < 0.5:
        raise ValueError("border_fraction must lie in (0,0.5).")

    height, width = images.shape[-2:]
    border = max(1, int(round(min(height, width) * border_fraction)))
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


def create_corner_only_images(
    images,
    corner_fraction=STANDARDIZED_CORNER_WIDTH_FRACTION,
):
    """Retain only four square image corners as an export-template control."""

    corner_fraction = float(corner_fraction)
    if not 0.0 < corner_fraction < 0.5:
        raise ValueError("corner_fraction must lie in (0,0.5).")

    height, width = images.shape[-2:]
    corner_height = max(1, int(round(height * corner_fraction)))
    corner_width = max(1, int(round(width * corner_fraction)))
    mask = torch.zeros(
        1,
        1,
        height,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    mask[:, :, :corner_height, :corner_width] = 1.0
    mask[:, :, :corner_height, width - corner_width:] = 1.0
    mask[:, :, height - corner_height:, :corner_width] = 1.0
    mask[:, :, height - corner_height:, width - corner_width:] = 1.0
    return images * mask


def _hard_mask_bounding_box(mask_2d):
    """Return inclusive-exclusive bounding-box coordinates or None."""

    positions = torch.nonzero(mask_2d > 0.5, as_tuple=False)
    if positions.numel() == 0:
        return None
    top = int(positions[:, 0].min().item())
    bottom = int(positions[:, 0].max().item()) + 1
    left = int(positions[:, 1].min().item())
    right = int(positions[:, 1].max().item()) + 1
    return top, bottom, left, right


def _resize_single_crop(image, top, bottom, left, right, output_size):
    """Crop one CHW tensor, square-pad it, and resize without aspect distortion.

    Directly stretching a rectangular cardiac crop to 224x224 would introduce a
    new geometry artefact and make the center-crop/bounding-box controls harder
    to interpret. The cropped field is therefore centered in a zero-valued
    square first, then resized uniformly to the requested output dimensions.
    """

    height, width = image.shape[-2:]
    top = int(np.clip(top, 0, height - 1))
    bottom = int(np.clip(bottom, top + 1, height))
    left = int(np.clip(left, 0, width - 1))
    right = int(np.clip(right, left + 1, width))
    crop = image.unsqueeze(0)[:, :, top:bottom, left:right]

    crop_height, crop_width = crop.shape[-2:]
    square_side = max(crop_height, crop_width)
    pad_height = square_side - crop_height
    pad_width = square_side - crop_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    square_crop = F.pad(
        crop,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0,
    )
    return F.interpolate(
        square_crop,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0]


def create_center_crop_area_matched_images(
    images,
    hard_mask,
    valid_mask,
):
    """Create a central crop with the same HxW as each MONAI mask bounding box.

    This control separates the effect of anatomical localization from the much
    simpler effect of zooming into the image center. Invalid/empty masks use one
    fixed predeclared crop fraction rather than a label- or score-dependent rule.
    """

    height, width = images.shape[-2:]
    output = []
    for index in range(images.shape[0]):
        box = None
        if bool(valid_mask[index].item()):
            box = _hard_mask_bounding_box(hard_mask[index, 0])

        if box is None:
            crop_height = max(2, int(round(height * CENTER_CROP_FALLBACK_FRACTION)))
            crop_width = max(2, int(round(width * CENTER_CROP_FALLBACK_FRACTION)))
        else:
            top, bottom, left, right = box
            crop_height = max(2, bottom - top)
            crop_width = max(2, right - left)

        center_y = height // 2
        center_x = width // 2
        top = center_y - crop_height // 2
        left = center_x - crop_width // 2
        bottom = top + crop_height
        right = left + crop_width

        # Shift the box back inside the image without changing its size where
        # possible. Final clipping occurs in _resize_single_crop.
        if top < 0:
            bottom -= top
            top = 0
        if left < 0:
            right -= left
            left = 0
        if bottom > height:
            top -= bottom - height
            bottom = height
        if right > width:
            left -= right - width
            right = width

        output.append(
            _resize_single_crop(
                images[index],
                top,
                bottom,
                left,
                right,
                (height, width),
            )
        )

    return torch.stack(output, dim=0)


def create_monai_bounding_box_crop_images(
    images,
    hard_mask,
    valid_mask,
    context_fraction=MONAI_BBOX_CONTEXT_FRACTION,
):
    """Crop around a valid MONAI hard-mask box and resize to EfficientNet size.

    Invalid masks preserve the full standardized image, matching the baseline's
    safety principle. The context fraction is fixed and label-blind.
    """

    context_fraction = float(context_fraction)
    if context_fraction < 0.0:
        raise ValueError("context_fraction cannot be negative.")

    height, width = images.shape[-2:]
    output = []
    for index in range(images.shape[0]):
        box = None
        if bool(valid_mask[index].item()):
            box = _hard_mask_bounding_box(hard_mask[index, 0])

        if box is None:
            output.append(images[index])
            continue

        top, bottom, left, right = box
        box_height = bottom - top
        box_width = right - left
        margin = int(round(max(box_height, box_width) * context_fraction))
        output.append(
            _resize_single_crop(
                images[index],
                top - margin,
                bottom + margin,
                left - margin,
                right + margin,
                (height, width),
            )
        )

    return torch.stack(output, dim=0)


def create_outside_monai_bounding_box_images(
    images,
    hard_mask,
    valid_mask,
    context_fraction=OUTSIDE_MONAI_BBOX_CONTEXT_FRACTION,
):
    """Retain only pixels outside an enlarged valid MONAI bounding box.

    Invalid or empty masks yield an all-zero control image. Returning the full
    image in those cases would allow fallback frequency itself to inject the
    complete anatomy/export signal into a supposedly outside-region control.
    """

    context_fraction = float(context_fraction)
    if context_fraction < 0.0:
        raise ValueError("context_fraction cannot be negative.")

    height, width = images.shape[-2:]
    output = torch.zeros_like(images)
    for index in range(images.shape[0]):
        if not bool(valid_mask[index].item()):
            continue
        box = _hard_mask_bounding_box(hard_mask[index, 0])
        if box is None:
            continue

        top, bottom, left, right = box
        box_height = bottom - top
        box_width = right - left
        margin = int(round(max(box_height, box_width) * context_fraction))
        top = max(0, top - margin)
        bottom = min(height, bottom + margin)
        left = max(0, left - margin)
        right = min(width, right + margin)

        output[index] = images[index]
        output[index, :, top:bottom, left:right] = 0.0

    return output


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
    """
    Run the shared frozen image-processing stage and build one aligned feature
    bank for every image representation required by the experiment registry.

    PIPELINE FOR EACH DISCOVERED JPEG
    ---------------------------------

        native JPEG decode
            -> exact decoded-pixel hash + DCT pHash + provenance features
            -> per-image intensity scaling to [0,1]
            -> centered 256x256 MONAI canvas
            -> optional MONAI inference on original and standardized canvases
            -> original and standardized ROI / image / shortcut-control views
            -> ImageNet normalization performed separately for each view
            -> frozen EfficientNet-B0 1280-D embedding per requested view
            -> one deterministic row in every feature-bank array

    WHY ONE DATASET PASS?
    ---------------------
    Full-image, ROI and negative-control ablations must use exactly the same JPEG
    files and metadata rows. Decoding the JPEG and running MONAI once per batch
    avoids repeated I/O and repeated segmentation while preserving a controlled
    comparison. EfficientNet is still executed separately for each requested
    image representation because each view has different pixels.

    WHY MEMORY-MAPPED FEATURE MATRICES?
    -----------------------------------
    A 63,648 x 1,280 float32 matrix is roughly 311 MiB. This deconfounding
    suite can create fourteen such representations, so keeping them plus
    intermediate tensors in ordinary RAM is unnecessary. Each matrix is
    written incrementally to a NumPy .npy memory map, flushed, and reopened read-
    only after the metadata completion marker is written.

    LABEL-LEAKAGE CONTRACT
    ----------------------
    Labels are carried only as aligned metadata. They are not inputs to MONAI,
    EfficientNet, mask gating, quality scoring, hashes, or provenance features.
    All supervised fitting occurs later inside patient-level CV folds.
    """

    # ------------------------------------------------------------------
    # Feature-bank stage 1: validate the requested view contract.
    # ------------------------------------------------------------------
    # Tabular controls do not create EfficientNet matrices and therefore are
    # absent from ``required_modes``. At least one image experiment must remain.
    if not required_modes:
        raise ValueError("At least one EfficientNet feature mode is required.")

    # MONAI is loaded only when an enabled image view actually needs its mask.
    # The full-image and border-only controls can otherwise run without MONAI.
    original_monai_modes = {"monai_roi", "outside_heart"}
    standardized_monai_modes = {
        "standardized_monai_roi",
        "standardized_center_crop",
        "standardized_roi_zero_background",
        "standardized_roi_bbox",
        "standardized_outside_large_bbox",
    }
    need_original_monai = any(
        mode in original_monai_modes for mode in required_modes
    )
    need_standardized_monai = any(
        mode in standardized_monai_modes for mode in required_modes
    )
    need_monai = need_original_monai or need_standardized_monai
    if need_monai and monai_segmenter is None:
        raise RuntimeError(
            "MONAI-dependent feature modes were requested without a segmenter."
        )

    # ------------------------------------------------------------------
    # Feature-bank stage 2: create a clean cache transaction directory.
    # ------------------------------------------------------------------
    # metadata.json is written only after every array succeeds. Therefore an
    # existing incomplete directory is safe to remove before a rebuild.
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

    # ------------------------------------------------------------------
    # Feature-bank stage 3: preallocate disk-backed embedding matrices.
    # ------------------------------------------------------------------
    # Each row index is the immutable sample_index. Direct indexed writes avoid
    # accumulating hundreds of megabytes of Python lists before np.vstack.
    feature_maps = {
        mode: np.lib.format.open_memmap(
            _feature_mode_path(cache_dir, mode),
            mode="w+",
            dtype=np.float32,
            shape=(n_slices, EFFICIENTNET_FEATURE_DIM),
        )
        for mode in required_modes
    }

    # Shared arrays below contain one value/vector per slice and remain aligned
    # with every feature matrix. Small numeric arrays stay in RAM during the
    # pass; large embedding matrices are already memory mapped.
    labels_array = np.empty(n_slices, dtype=np.int64)
    sample_indices_array = np.empty(n_slices, dtype=np.int64)
    provenance_array = np.empty(
        (n_slices, len(PROVENANCE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    standardization_array = np.empty(
        (n_slices, len(STANDARDIZATION_FEATURE_NAMES)),
        dtype=np.float32,
    )

    # Original-baseline MONAI QC arrays remain available for historical C4.
    monai_valid_array = np.zeros(n_slices, dtype=bool)
    area_ratio_array = np.full(n_slices, np.nan, dtype=np.float32)
    peak_probability_array = np.full(n_slices, np.nan, dtype=np.float32)
    mean_foreground_array = np.full(n_slices, np.nan, dtype=np.float32)
    roi_slice_score_array = np.full(n_slices, np.nan, dtype=np.float32)

    # The deconfounded branch receives its own MONAI inference/QC arrays because
    # label-blind cropping and robust scaling can legitimately change masks.
    standardized_monai_valid_array = np.zeros(n_slices, dtype=bool)
    standardized_area_ratio_array = np.full(
        n_slices, np.nan, dtype=np.float32
    )
    standardized_peak_probability_array = np.full(
        n_slices, np.nan, dtype=np.float32
    )
    standardized_mean_foreground_array = np.full(
        n_slices, np.nan, dtype=np.float32
    )
    standardized_roi_slice_score_array = np.full(
        n_slices, np.nan, dtype=np.float32
    )

    patient_ids_values = [None] * n_slices
    series_ids_values = [None] * n_slices
    decoded_hash_values = [None] * n_slices
    perceptual_hash_values = [None] * n_slices

    # ------------------------------------------------------------------
    # Feature-bank stage 4: build a deterministic non-shuffled DataLoader.
    # ------------------------------------------------------------------
    # ``sample_index`` is still used for indexed writes, so increasing worker
    # count later cannot silently reorder the cache rows.
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

    # ------------------------------------------------------------------
    # Feature-bank stage 5: process every discovered image exactly once.
    # ------------------------------------------------------------------
    # Inference mode disables gradients for both frozen networks. CUDA autocast
    # affects only network inference; probabilities and stored embeddings are
    # converted back to float32 before QC/downstream analysis.
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
                standardized_images,
                standardized_monai_images,
                detected_padding_images,
                labels,
                patient_ids,
                series_ids,
                sample_indices,
                decoded_pixel_hashes,
                perceptual_hashes,
                provenance_features,
                standardization_features,
            ) = batch

            # Stable global indices identify the target rows in every output
            # array. Only network tensors are moved to the runtime device; IDs,
            # hashes and provenance values remain host-side metadata.
            index_values = sample_indices.detach().cpu().numpy().astype(np.int64)
            images = images.to(DEVICE, non_blocking=True)
            monai_images = monai_images.to(DEVICE, non_blocking=True)
            standardized_images = standardized_images.to(
                DEVICE, non_blocking=True
            )
            standardized_monai_images = standardized_monai_images.to(
                DEVICE, non_blocking=True
            )
            detected_padding_images = detected_padding_images.to(
                DEVICE, non_blocking=True
            )

            # -------------------------------------------------------------
            # Feature-bank stage 6: run MONAI separately on the original and
            # standardized canvases only when enabled views require them.
            # -------------------------------------------------------------
            # The two branches must not share masks because dark-padding removal
            # and robust scaling can legitimately alter the segmenter's output.
            # Both use the same pinned network, gate thresholds and dilation.
            batch_size = images.shape[0]

            if need_original_monai:
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

            if need_standardized_monai:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    (
                        standardized_roi_probability,
                        standardized_hard_mask,
                        standardized_valid_mask,
                        standardized_area_ratio,
                        standardized_peak_probability,
                        standardized_mean_foreground_probability,
                    ) = predict_monai_heart_masks(
                        standardized_monai_images,
                        classifier_size=standardized_images.shape[-2:],
                        monai_segmenter=monai_segmenter,
                    )
                    standardized_roi_images = apply_confidence_gated_soft_roi(
                        standardized_images,
                        standardized_roi_probability,
                        standardized_valid_mask,
                    )
            else:
                standardized_roi_probability = torch.zeros(
                    batch_size,
                    1,
                    *standardized_images.shape[-2:],
                    device=standardized_images.device,
                    dtype=standardized_images.dtype,
                )
                standardized_hard_mask = standardized_roi_probability.clone()
                standardized_valid_mask = torch.zeros(
                    batch_size,
                    device=standardized_images.device,
                    dtype=torch.bool,
                )
                standardized_area_ratio = torch.full(
                    (batch_size,),
                    float("nan"),
                    device=standardized_images.device,
                )
                standardized_peak_probability = torch.full_like(
                    standardized_area_ratio,
                    float("nan"),
                )
                standardized_mean_foreground_probability = torch.full_like(
                    standardized_area_ratio,
                    float("nan"),
                )
                standardized_roi_images = standardized_images

            # -------------------------------------------------------------
            # Feature-bank stage 7: construct only the enabled image views.
            # -------------------------------------------------------------
            # Original controls remain available for exact reproduction. New
            # standardized controls isolate narrow borders, corners, padding
            # geometry, center cropping, stricter ROI removal, and exterior
            # signal after label-blind export normalization.
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

            if "standardized_monai_roi" in required_modes:
                variants["standardized_monai_roi"] = standardized_roi_images
            if "standardized_full_image" in required_modes:
                variants["standardized_full_image"] = standardized_images
            if "standardized_border_05" in required_modes:
                variants["standardized_border_05"] = create_border_only_images(
                    standardized_images,
                    border_fraction=STANDARDIZED_BORDER_WIDTH_FRACTIONS[0],
                )
            if "standardized_border_10" in required_modes:
                variants["standardized_border_10"] = create_border_only_images(
                    standardized_images,
                    border_fraction=STANDARDIZED_BORDER_WIDTH_FRACTIONS[1],
                )
            if "detected_padding_mask" in required_modes:
                variants["detected_padding_mask"] = detected_padding_images
            if "standardized_corners" in required_modes:
                variants["standardized_corners"] = create_corner_only_images(
                    standardized_images
                )
            if "standardized_center_crop" in required_modes:
                variants["standardized_center_crop"] = (
                    create_center_crop_area_matched_images(
                        standardized_images,
                        standardized_hard_mask,
                        standardized_valid_mask,
                    )
                )
            if "standardized_roi_zero_background" in required_modes:
                variants["standardized_roi_zero_background"] = (
                    apply_confidence_gated_soft_roi(
                        standardized_images,
                        standardized_roi_probability,
                        standardized_valid_mask,
                        background_weight=0.0,
                    )
                )
            if "standardized_roi_bbox" in required_modes:
                variants["standardized_roi_bbox"] = (
                    create_monai_bounding_box_crop_images(
                        standardized_images,
                        standardized_hard_mask,
                        standardized_valid_mask,
                    )
                )
            if "standardized_outside_large_bbox" in required_modes:
                variants["standardized_outside_large_bbox"] = (
                    create_outside_monai_bounding_box_images(
                        standardized_images,
                        standardized_hard_mask,
                        standardized_valid_mask,
                    )
                )

            # -------------------------------------------------------------
            # Feature-bank stage 8: encode requested views in small mode chunks.
            # -------------------------------------------------------------
            # Each view still receives an independent 1280-D embedding, but up
            # to FEATURE_MODES_PER_ENCODER_CALL views are concatenated along the
            # batch dimension before one frozen EfficientNet forward pass. In
            # evaluation mode, EfficientNet batch-normalization uses fixed
            # running statistics, so one view cannot change another view's
            # representation. Chunking reduces Python and CUDA launch overhead
            # while bounding peak memory on a Tesla T4-class GPU.
            mode_names = list(required_modes)
            for chunk_start in range(
                0,
                len(mode_names),
                FEATURE_MODES_PER_ENCODER_CALL,
            ):
                chunk_modes = mode_names[
                    chunk_start:chunk_start + FEATURE_MODES_PER_ENCODER_CALL
                ]
                concatenated_images = torch.cat(
                    [variants[mode] for mode in chunk_modes],
                    dim=0,
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    network_input = normalize_for_efficientnet(
                        concatenated_images
                    )
                    concatenated_features = feature_extractor(
                        network_input
                    ).float()

                expected_rows = len(index_values) * len(chunk_modes)
                if concatenated_features.shape != (
                    expected_rows,
                    EFFICIENTNET_FEATURE_DIM,
                ):
                    raise RuntimeError(
                        "Unexpected concatenated EfficientNet feature shape for "
                        f"modes {chunk_modes}: "
                        f"{tuple(concatenated_features.shape)}."
                    )

                split_features = torch.split(
                    concatenated_features,
                    len(index_values),
                    dim=0,
                )
                for mode, batch_features in zip(chunk_modes, split_features):
                    feature_maps[mode][index_values, :] = (
                        batch_features.detach().cpu().numpy()
                    )

                # Release the largest temporary tensors before MONAI-QC and
                # metadata arrays are copied back to the host.
                del concatenated_images, network_input, concatenated_features

            # -------------------------------------------------------------
            # Feature-bank stage 9: record the optional heuristic and QC data.
            # -------------------------------------------------------------
            # ROI/fallback standard deviation is saved for A6 but does not alter
            # the default baseline. Every shared value uses the same row indices.
            roi_scores = torch.std(roi_images, dim=(1, 2, 3))
            standardized_roi_scores = torch.std(
                standardized_roi_images, dim=(1, 2, 3)
            )

            labels_array[index_values] = labels.detach().cpu().numpy()
            sample_indices_array[index_values] = index_values
            provenance_array[index_values, :] = (
                provenance_features.detach().cpu().numpy()
            )
            standardization_array[index_values, :] = (
                standardization_features.detach().cpu().numpy()
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

            standardized_monai_valid_array[index_values] = (
                standardized_valid_mask.detach().cpu().numpy()
            )
            standardized_area_ratio_array[index_values] = (
                standardized_area_ratio.detach().cpu().numpy()
            )
            standardized_peak_probability_array[index_values] = (
                standardized_peak_probability.detach().cpu().numpy()
            )
            standardized_mean_foreground_array[index_values] = (
                standardized_mean_foreground_probability.detach().cpu().numpy()
            )
            standardized_roi_slice_score_array[index_values] = (
                standardized_roi_scores.detach().cpu().numpy()
            )

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

    # ------------------------------------------------------------------
    # Feature-bank stage 10: prove complete one-to-one row coverage.
    # ------------------------------------------------------------------
    # A partially written matrix must never receive a completion marker or be
    # accepted as a valid cache on a later run.
    if any(value is None for value in patient_ids_values):
        raise RuntimeError("At least one patient ID was not written to the bank.")
    if any(value is None for value in decoded_hash_values):
        raise RuntimeError("At least one decoded hash was not written to the bank.")
    if not np.array_equal(sample_indices_array, np.arange(n_slices)):
        raise RuntimeError("Feature-bank sample indices are incomplete or reordered.")

    # Flush large disk-backed matrices before writing the smaller shared arrays
    # and metadata. metadata.json remains the final completion marker.
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
        "standardization_features": standardization_array,
        "monai_valid": monai_valid_array,
        "area_ratios": area_ratio_array,
        "peak_probabilities": peak_probability_array,
        "mean_foreground_probabilities": mean_foreground_array,
        "roi_slice_scores": roi_slice_score_array,
        "standardized_monai_valid": standardized_monai_valid_array,
        "standardized_area_ratios": standardized_area_ratio_array,
        "standardized_peak_probabilities": (
            standardized_peak_probability_array
        ),
        "standardized_mean_foreground_probabilities": (
            standardized_mean_foreground_array
        ),
        "standardized_roi_slice_scores": (
            standardized_roi_slice_score_array
        ),
    }
    for name, array in shared_arrays.items():
        np.save(shared_paths[name], np.asarray(array), allow_pickle=False)

    # ------------------------------------------------------------------
    # Feature-bank stage 11: write provenance metadata last and re-open cache.
    # ------------------------------------------------------------------
    metadata = {
        "fingerprint": fingerprint,
        "schema": FEATURE_CACHE_SCHEMA_VERSION,
        "completed_modes": list(required_modes),
        "n_slices": int(n_slices),
        "feature_dimension": EFFICIENTNET_FEATURE_DIM,
        "provenance_feature_names": list(PROVENANCE_FEATURE_NAMES),
        "standardization_feature_names": list(
            STANDARDIZATION_FEATURE_NAMES
        ),
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
        mode in {
            "monai_roi",
            "outside_heart",
            "standardized_monai_roi",
            "standardized_center_crop",
            "standardized_roi_zero_background",
            "standardized_roi_bbox",
            "standardized_outside_large_bbox",
        }
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


class _HammingBKTree:
    """Minimal BK-tree for exact radius search over 64-bit integer hashes."""

    def __init__(self):
        self.root = None

    @staticmethod
    def _distance(first, second):
        return int(first ^ second).bit_count()

    def add(self, value):
        value = int(value)
        if self.root is None:
            self.root = [value, {}]
            return

        node = self.root
        while True:
            distance = self._distance(value, node[0])
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = [value, {}]
                return
            node = child

    def query(self, value, maximum_distance):
        if self.root is None:
            return []

        value = int(value)
        maximum_distance = int(maximum_distance)
        matches = []
        stack = [self.root]
        while stack:
            node_value, children = stack.pop()
            distance = self._distance(value, node_value)
            if distance <= maximum_distance:
                matches.append((node_value, distance))

            lower = distance - maximum_distance
            upper = distance + maximum_distance
            for edge_distance, child in children.items():
                if lower <= edge_distance <= upper:
                    stack.append(child)

        return matches


def audit_perceptual_near_duplicate_candidates(
    output_path,
    samples,
    perceptual_hashes,
    decoded_pixel_hashes,
):
    """Find all cross-patient pHash candidates within the configured radius.

    The first suite used exact-chunk LSH with emergency bucket and pair caps.
    Those safeguards made runtime predictable, but the audit could skip large
    buckets and stop after one million pairs. The revised default constructs a
    BK-tree over UNIQUE 64-bit pHash values and performs an exact Hamming-radius
    search. Candidate evidence is aggregated by patient and hash group, so large
    within-patient duplicate sets do not require enumerating every image pair.

    pHash remains a screening method, not a duplicate verdict. Every saved pair
    still requires manual inspection or a stronger image-similarity review.
    """

    if not (
        len(samples) == len(perceptual_hashes) == len(decoded_pixel_hashes)
    ):
        raise ValueError("Perceptual audit arrays do not align with samples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hash_values = np.asarray(
        [int(str(value), 16) for value in perceptual_hashes],
        dtype=np.uint64,
    )

    # Group each unique pHash by patient. Counts are retained so the output can
    # report how many image-pair combinations support one patient-pair flag
    # without materializing all combinations in memory.
    hash_to_patient_data = defaultdict(dict)
    for index, hash_value in enumerate(hash_values.tolist()):
        patient_id = str(samples[index][2])
        label = int(samples[index][1])
        patient_record = hash_to_patient_data[int(hash_value)].get(patient_id)
        if patient_record is None:
            patient_record = {
                "label": label,
                "count": 0,
                "example_index": index,
                "decoded_hashes": set(),
            }
            hash_to_patient_data[int(hash_value)][patient_id] = patient_record
        elif int(patient_record["label"]) != label:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent pHash-audit labels."
            )
        patient_record["count"] += 1
        patient_record["decoded_hashes"].add(
            str(decoded_pixel_hashes[index])
        )

    unique_hashes = sorted(hash_to_patient_data)
    tree = _HammingBKTree()
    for hash_value in unique_hashes:
        tree.add(hash_value)

    patient_pair_data = {}
    examined_unique_hash_pairs = 0
    supporting_image_pair_count = 0
    search_started_at = time.perf_counter()
    search_progress_interval = max(1, len(unique_hashes) // 10)
    print(
        "[PERCEPTUAL DUPLICATES] Starting complete BK-tree radius search over "
        f"{len(unique_hashes)} unique pHash values.",
        flush=True,
    )

    for hash_index, first_hash in enumerate(unique_hashes, start=1):
        for second_hash, distance in tree.query(
            first_hash,
            PHASH_HAMMING_THRESHOLD,
        ):
            # Process each unordered unique-hash pair exactly once. Self-pairs
            # are retained because one pHash value can occur in several patients.
            if second_hash < first_hash:
                continue
            examined_unique_hash_pairs += 1

            first_patients = hash_to_patient_data[first_hash]
            second_patients = hash_to_patient_data[second_hash]

            for first_patient, first_data in first_patients.items():
                for second_patient, second_data in second_patients.items():
                    if first_patient == second_patient:
                        continue

                    patient_pair = tuple(
                        sorted((str(first_patient), str(second_patient)))
                    )

                    # When both sides refer to the same pHash group, symmetric
                    # patient combinations would otherwise be counted twice.
                    if first_hash == second_hash and first_patient > second_patient:
                        continue

                    if patient_pair[0] == first_patient:
                        data_a, data_b = first_data, second_data
                        hash_a, hash_b = first_hash, second_hash
                    else:
                        data_a, data_b = second_data, first_data
                        hash_a, hash_b = second_hash, first_hash

                    candidate_count = int(data_a["count"] * data_b["count"])
                    supporting_image_pair_count += candidate_count
                    exact_pixels = bool(
                        data_a["decoded_hashes"].intersection(
                            data_b["decoded_hashes"]
                        )
                    )

                    example_index_a = int(data_a["example_index"])
                    example_index_b = int(data_b["example_index"])
                    record = patient_pair_data.get(patient_pair)
                    if record is None:
                        record = {
                            "patient_id_a": patient_pair[0],
                            "patient_id_b": patient_pair[1],
                            "label_a": int(data_a["label"]),
                            "label_b": int(data_b["label"]),
                            "cross_label": int(
                                int(data_a["label"]) != int(data_b["label"])
                            ),
                            "minimum_phash_hamming_distance": int(distance),
                            "candidate_image_pairs": 0,
                            "contains_exact_pixel_pair": int(exact_pixels),
                            "example_image_a": str(samples[example_index_a][0]),
                            "example_image_b": str(samples[example_index_b][0]),
                            "example_phash_a": f"{int(hash_a):016x}",
                            "example_phash_b": f"{int(hash_b):016x}",
                        }
                        patient_pair_data[patient_pair] = record

                    record["candidate_image_pairs"] += candidate_count
                    record["contains_exact_pixel_pair"] = int(
                        bool(record["contains_exact_pixel_pair"]) or exact_pixels
                    )
                    if int(distance) < int(
                        record["minimum_phash_hamming_distance"]
                    ):
                        record["minimum_phash_hamming_distance"] = int(distance)
                        record["example_image_a"] = str(
                            samples[example_index_a][0]
                        )
                        record["example_image_b"] = str(
                            samples[example_index_b][0]
                        )
                        record["example_phash_a"] = f"{int(hash_a):016x}"
                        record["example_phash_b"] = f"{int(hash_b):016x}"

        if (
            hash_index == 1
            or hash_index % search_progress_interval == 0
            or hash_index == len(unique_hashes)
        ):
            elapsed = time.perf_counter() - search_started_at
            rate = hash_index / max(elapsed, 1e-12)
            remaining = len(unique_hashes) - hash_index
            eta = remaining / max(rate, 1e-12)
            print(
                "[PERCEPTUAL DUPLICATES] "
                f"{hash_index}/{len(unique_hashes)} hashes "
                f"({100.0 * hash_index / max(len(unique_hashes), 1):.0f}%) | "
                f"elapsed={_format_elapsed_time(elapsed)} | "
                f"ETA={_format_elapsed_time(eta)}",
                flush=True,
            )

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
            "Complete unique-hash BK-tree search over 64-bit DCT pHash; "
            f"cross-patient pairs with Hamming distance <= {PHASH_HAMMING_THRESHOLD}"
        ),
        "search_method": "complete_bk_tree_unique_phash_values",
        "unique_phash_values": int(len(unique_hashes)),
        "examined_unique_hash_pairs_within_radius": int(
            examined_unique_hash_pairs
        ),
        "supporting_image_pair_combinations": int(
            supporting_image_pair_count
        ),
        "patient_pair_candidates": int(len(rows)),
        "cross_label_patient_pair_candidates": int(
            sum(row["cross_label"] for row in rows)
        ),
        "skipped_large_lsh_buckets": 0,
        "candidate_search_truncated": False,
        "search_runtime_seconds": float(
            time.perf_counter() - search_started_at
        ),
        "csv_path": str(output_path),
    }

    print(
        "[PERCEPTUAL DUPLICATES] "
        f"method=complete_BK_tree, unique_hashes={len(unique_hashes)}, "
        f"patient_pair_candidates={len(rows)}, "
        f"cross_label={summary['cross_label_patient_pair_candidates']}, "
        "skipped_large_buckets=0, truncated=False",
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
        "standardized_monai_gate_valid",
        "standardized_monai_area_ratio",
        "standardized_monai_peak_probability",
        "standardized_monai_mean_foreground_probability",
        "standardized_roi_slice_std_score",
        *PROVENANCE_FEATURE_NAMES,
        *STANDARDIZATION_FEATURE_NAMES,
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
                "standardized_monai_gate_valid": int(
                    bool(bank["standardized_monai_valid"][index])
                ),
                "standardized_monai_area_ratio": float(
                    bank["standardized_area_ratios"][index]
                ),
                "standardized_monai_peak_probability": float(
                    bank["standardized_peak_probabilities"][index]
                ),
                "standardized_monai_mean_foreground_probability": float(
                    bank["standardized_mean_foreground_probabilities"][index]
                ),
                "standardized_roi_slice_std_score": float(
                    bank["standardized_roi_slice_scores"][index]
                ),
            }
            for feature_name, value in zip(
                PROVENANCE_FEATURE_NAMES,
                bank["provenance_features"][index],
            ):
                row[feature_name] = float(value)
            for feature_name, value in zip(
                STANDARDIZATION_FEATURE_NAMES,
                bank["standardization_features"][index],
            ):
                row[feature_name] = float(value)
            writer.writerow(row)


def _patient_indices(patient_ids):
    mapping = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        mapping[str(patient_id)].append(index)
    return mapping


def aggregate_patient_provenance_features(bank):
    """Aggregate conservative export/provenance features per patient.

    The full image-level audit remains available in ``cohort_manifest.csv``.
    This classifier matrix excludes central/global texture features that might
    encode disease-related anatomy; it therefore provides a more defensible
    test of whether folder/export structure, dimensions, compression proxies,
    padding, or border style alone can separate Normal and Sick patients.
    """

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"])
    series_ids = np.asarray(bank["series_ids"])
    provenance = np.asarray(bank["provenance_features"], dtype=np.float32)

    selected_feature_indices = np.asarray(
        [
            PROVENANCE_FEATURE_NAMES.index(feature_name)
            for feature_name in PROVENANCE_CLASSIFIER_FEATURE_NAMES
        ],
        dtype=np.int64,
    )
    provenance = provenance[:, selected_feature_indices]

    mapping = _patient_indices(patient_ids)
    ordered_patients = np.asarray(sorted(mapping))
    patient_labels = []
    rows = []

    output_names = [
        "n_slices",
        "n_series_proxies",
        "mean_series_length",
        "max_series_length",
    ]
    for feature_name in PROVENANCE_CLASSIFIER_FEATURE_NAMES:
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



def aggregate_patient_standardization_features(bank):
    """Aggregate label-blind crop/scaling diagnostics to one patient row.

    This negative-control matrix contains no EfficientNet embedding, MONAI
    probability, image texture, filename, or class label. It asks whether the
    amount of detected padding and the robust intensity limits themselves are
    systematically different between Normal and Sick exports.
    """

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"])
    values = np.asarray(bank["standardization_features"], dtype=np.float32)

    mapping = _patient_indices(patient_ids)
    ordered_patients = np.asarray(sorted(mapping))
    patient_labels = []
    rows = []
    output_names = []
    for feature_name in STANDARDIZATION_FEATURE_NAMES:
        output_names.extend(
            [
                f"mean__{feature_name}",
                f"median__{feature_name}",
                f"std__{feature_name}",
                f"max__{feature_name}",
            ]
        )

    for patient_id in ordered_patients:
        indices = np.asarray(mapping[str(patient_id)], dtype=np.int64)
        label_values = np.unique(labels[indices])
        if len(label_values) != 1:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent standardization labels."
            )
        patient_labels.append(int(label_values[0]))

        patient_values = values[indices]
        row = []
        for feature_index in range(patient_values.shape[1]):
            column = patient_values[:, feature_index]
            row.extend(
                [
                    float(np.mean(column)),
                    float(np.median(column)),
                    float(np.std(column)),
                    float(np.max(column)),
                ]
            )
        rows.append(row)

    X = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(X)):
        raise RuntimeError(
            "Standardization-QC patient features contain non-finite values."
        )
    y = np.asarray(patient_labels, dtype=np.int64)
    return X, y, ordered_patients, tuple(output_names)


def aggregate_patient_standardized_monai_qc_features(bank):
    """Aggregate MONAI QC after label-blind preprocessing to one patient row."""

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"])
    valid = np.asarray(bank["standardized_monai_valid"], dtype=bool)
    area = np.asarray(bank["standardized_area_ratios"], dtype=np.float32)
    peak = np.asarray(
        bank["standardized_peak_probabilities"], dtype=np.float32
    )
    foreground = np.asarray(
        bank["standardized_mean_foreground_probabilities"],
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
                f"Patient {patient_id} has inconsistent standardized MONAI-QC labels."
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
            "Standardized MONAI-QC patient features contain non-finite values."
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


def write_standardized_monai_qc_outputs(output_dir, bank):
    """Write the same gate audit after label-blind image standardization."""

    output_dir.mkdir(parents=True, exist_ok=True)
    X_qc, y_qc, patient_ids, feature_names = (
        aggregate_patient_standardized_monai_qc_features(bank)
    )

    rows = []
    for patient_id, label, values in zip(patient_ids, y_qc, X_qc):
        row = {"patient_id": str(patient_id), "true_label": int(label)}
        for feature_name, value in zip(feature_names, values):
            row[feature_name] = float(value)
        rows.append(row)

    patient_csv = output_dir / "standardized_monai_qc_by_patient.csv"
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
        RANDOM_SEED + 1701,
    )

    slice_labels = np.asarray(bank["labels"], dtype=np.int64)
    slice_valid = np.asarray(bank["standardized_monai_valid"], dtype=bool)
    slice_normal_rate = float(np.mean(slice_valid[slice_labels == 0]))
    slice_sick_rate = float(np.mean(slice_valid[slice_labels == 1]))

    comparison = {
        "preprocessing_branch": "label_blind_standardized",
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
            "A large class difference after label-blind standardization may "
            "still reflect sequence/protocol differences; it is not "
            "segmentation-accuracy evidence."
        ),
    }
    (
        output_dir / "standardized_monai_gate_class_comparison.json"
    ).write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "[STANDARDIZED MONAI QC] Patient-level mean plausible-mask rate: "
        f"Normal={normal_mean:.3f}, Sick={sick_mean:.3f}, "
        f"difference={difference:+.3f}, "
        f"95% CI=[{ci_lower:+.3f}, {ci_upper:+.3f}]",
        flush=True,
    )
    if comparison["warning_triggered"]:
        print(
            "[WARNING] The standardized class-specific MONAI gate-rate "
            "difference exceeds the configured threshold.",
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
    """
    Pool frozen slice embeddings to exactly one vector per Directory_* patient.

    RECOMMENDED HIERARCHICAL STRATEGY
    ---------------------------------

        slice embeddings --weighted mean within series proxy--> series embedding
        series embeddings --------equal mean across proxies----> patient embedding

    This is the recommended default because the supervised classifier receives
    one observation and one label per validated Directory_* patient. Equal
    series-proxy weighting prevents a folder with many exported JPEG frames from
    dominating merely because it is long. The series unit remains an operational
    folder proxy, not a recovered DICOM SeriesInstanceUID.

    FLAT-POOLING ABLATION
    ---------------------

        all patient slices --one weighted mean--> patient embedding

    Flat pooling intentionally removes the series hierarchy. With equal slice
    weights, a long folder proxy receives proportionally more influence. It is
    retained only to measure whether the hierarchical design materially changes
    patient-level performance.

    Both branches are deterministic and label-free. Labels are checked only for
    patient consistency and later supervised fitting.
    """

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
    """
    Build legacy slice-classifier training weights aligned with the evaluation
    hierarchy.

    Desired nominal influence structure:

        class -> patient -> series proxy -> slice

    For slice i belonging to series proxy s of patient p and class c:

        w_i = (1 / N_patients_in_class_c)
              * (1 / N_series_for_patient_p)
              * (q_i / sum(q_j for j in series_s))

    where q_i is either 1 or the optional series-local heuristic weight. This
    gives two useful invariances:

        1. A patient with more exported JPEG files does not automatically
           dominate fitting.
        2. One unusually long folder proxy does not dominate the patient's
           shorter proxies.

    The class term gives Normal and Sick equal total nominal supervised mass.

    IMPORTANT: WEIGHTS DO NOT REMOVE PSEUDO-REPLICATION
    ----------------------------------------------------
    Every slice still carries its patient's repeated Normal/Sick label, and
    slices from one patient remain strongly correlated. The weights control
    nominal influence but do not turn slices into independent observations. The
    legacy branch must therefore be reported only as an ablation.

    GLOBAL WEIGHT SCALE AND REGULARIZATION
    --------------------------------------
    Multiplying all sample weights by a constant changes the data-loss versus
    regularization balance of Logistic Regression at fixed C. The final vector
    is therefore rescaled so its total mass equals the number of unique training
    patients, making effective regularization independent of JPEG count.
    """

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
    """
    Fuse finite probability-like model outputs in log-odds space.

    Mathematical form:

        logit(p_fused) = sum_i w_i * logit(p_i) / sum_i w_i

    Averaging logits avoids directly multiplying many probabilities, which
    would make the number of slices or series proxies force the output toward 0
    or 1. Inputs are clipped before the logarithm for numerical stability.

    LIMITATION
    ----------
    Logistic Regression slice outputs are not guaranteed to be calibrated on a
    new clinical population. The fused result is therefore an aggregation score
    on a 0-to-1 scale, not a validated posterior probability of CAD. Mean-
    probability fusion is retained as a controlled legacy ablation.
    """

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


def deterministic_series_preserving_slice_dropout_mask(
    patient_ids,
    series_ids,
    dropout_rate,
):
    """Return a reproducible label-blind slice-retention mask.

    The attached master script included a useful random slice-dropout robustness
    experiment, but one unrestricted patient-level draw can remove an entire
    short series proxy and then conflate missing slices with missing series. This
    implementation samples independently inside each patient-scoped series proxy
    and always retains at least one slice. The seed is derived from the global
    seed, dropout rate, and series ID through SHA-256, so results are independent
    of Python hash randomization, array traversal order, and batch size.

    The function never reads labels, image pixels, model scores, or fold IDs.
    It therefore cannot select slices according to outcome or validation
    performance. The resulting robustness analyses remain exploratory because
    each rate uses one predeclared deterministic realization.
    """

    patient_ids = np.asarray(patient_ids).astype(str)
    series_ids = np.asarray(series_ids).astype(str)
    if len(patient_ids) != len(series_ids):
        raise ValueError("patient_ids and series_ids must have equal lengths.")
    if not 0.0 <= float(dropout_rate) < 1.0:
        raise ValueError("dropout_rate must lie in [0,1).")

    keep = np.ones(len(series_ids), dtype=bool)
    if float(dropout_rate) == 0.0:
        return keep

    for series_id in np.unique(series_ids):
        indices = np.flatnonzero(series_ids == series_id)
        series_patients = np.unique(patient_ids[indices])
        if len(series_patients) != 1:
            raise RuntimeError(
                f"Series proxy {series_id} is assigned to multiple patients."
            )

        # floor(rate*N) matches the attached experiment's intended fraction,
        # while min(..., N-1) preserves at least one representative slice.
        n_drop = min(
            int(np.floor(len(indices) * float(dropout_rate))),
            max(0, len(indices) - 1),
        )
        if n_drop == 0:
            continue

        seed_material = (
            f"{RANDOM_SEED}|{float(dropout_rate):.8f}|{series_id}"
        ).encode("utf-8")
        local_seed = int(
            hashlib.sha256(seed_material).hexdigest()[:16],
            16,
        ) % (2 ** 32)
        rng = np.random.default_rng(local_seed)
        dropped = rng.choice(indices, size=n_drop, replace=False)
        keep[dropped] = False

    # The per-series retention rule should guarantee patient coverage. Keep the
    # explicit check because losing a patient would make OOF sets incomparable.
    retained_patients = set(patient_ids[keep].tolist())
    original_patients = set(patient_ids.tolist())
    if retained_patients != original_patients:
        missing = sorted(original_patients - retained_patients)
        raise RuntimeError(
            f"Slice dropout removed all observations for patients: {missing}."
        )

    return keep


def prepare_experiment_data(experiment, bank, tabular_feature_sets):
    """Build the fixed, label-blind data representation for one experiment."""

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
            "slice_dropout_rate": 0.0,
            "n_source_slices": None,
            "n_retained_slices": None,
        }

    features = bank["features"][experiment.feature_mode]
    roi_slice_scores = np.asarray(bank["roi_slice_scores"])
    n_source_slices = int(len(labels))

    # Robustness variants use the same frozen feature bank; only a deterministic
    # subset of aligned rows is retained before weighting and pooling. The mask
    # is generated without labels and is independent of outer-fold assignment.
    retention_mask = deterministic_series_preserving_slice_dropout_mask(
        patient_ids,
        series_ids,
        experiment.slice_dropout_rate,
    )
    if experiment.slice_dropout_rate > 0.0:
        features = features[retention_mask]
        labels = labels[retention_mask]
        patient_ids = patient_ids[retention_mask]
        series_ids = series_ids[retention_mask]
        roi_slice_scores = roi_slice_scores[retention_mask]
        print(
            f"[ROBUSTNESS] {experiment.experiment_id}: retained "
            f"{int(retention_mask.sum())}/{len(retention_mask)} slices "
            f"after {experiment.slice_dropout_rate:.0%} within-series dropout.",
            flush=True,
        )

    slice_weights = build_slice_weights(
        roi_slice_scores,
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
            "slice_dropout_rate": float(experiment.slice_dropout_rate),
            "n_source_slices": n_source_slices,
            "n_retained_slices": int(len(labels)),
        }

    if experiment.strategy == "slice_probability_fusion":
        return {
            "unit": "slice",
            "X": features,
            "y": labels,
            "patient_ids": patient_ids,
            "series_ids": series_ids,
            "slice_weights": slice_weights,
            # A3 and A4 differ only in probability fusion. This process-local
            # signature lets them reuse identical fold-local slice classifier
            # predictions without storing models or changing scientific output.
            "legacy_cache_signature": (
                experiment.feature_mode,
                experiment.weighting_mode,
                int(features.shape[0]),
                int(features.shape[1]),
            ),
            "slice_dropout_rate": float(experiment.slice_dropout_rate),
            "n_source_slices": n_source_slices,
            "n_retained_slices": int(len(labels)),
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
    legacy_prediction_cache=None,
):
    """Fit one fold-local model and return held-out patient scores.

    For the legacy slice-classifier branch, A3 and A4 have the same training
    data, weights, classifier and C; only the final fusion rule differs. A
    process-local cache can therefore reuse the held-out slice probabilities
    produced by the first variant. This reduces runtime without sharing fitted
    information across folds and without changing any patient-level result.
    """

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

    # The cache key deliberately excludes ``fusion_method`` because fusion is
    # applied only after the identical held-out slice probabilities are known.
    # Sorted patient IDs make the key independent of set iteration order.
    legacy_cache_key = (
        prepared.get("legacy_cache_signature"),
        float(c_value),
        tuple(sorted(train_patients)),
        tuple(sorted(valid_patients)),
    )

    cached = (
        legacy_prediction_cache.get(legacy_cache_key)
        if legacy_prediction_cache is not None
        else None
    )

    if cached is None:
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

        cached = {
            "slice_probabilities": np.asarray(
                model.predict_proba(prepared["X"][valid_mask])[:, 1],
                dtype=np.float64,
            ),
            "labels": np.asarray(
                prepared["y"][valid_mask],
                dtype=np.int64,
            ),
            "patient_ids": np.asarray(
                prepared["patient_ids"][valid_mask]
            ),
            "series_ids": np.asarray(
                prepared["series_ids"][valid_mask]
            ),
            "quality_weights": np.asarray(
                prepared["slice_weights"][valid_mask],
                dtype=np.float64,
            ),
        }
        if legacy_prediction_cache is not None:
            legacy_prediction_cache[legacy_cache_key] = cached
    else:
        print(
            "[LEGACY CACHE] Reusing fold-local slice probabilities; only "
            f"{experiment.fusion_method} fusion is recomputed.",
            flush=True,
        )

    return aggregate_legacy_slice_probabilities(
        slice_probabilities=cached["slice_probabilities"],
        labels=cached["labels"],
        patient_ids=cached["patient_ids"],
        series_ids=cached["series_ids"],
        quality_weights=cached["quality_weights"],
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
    legacy_prediction_cache=None,
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
            legacy_prediction_cache=legacy_prediction_cache,
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
        C=SVM_CALIBRATION_C,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_SEED,
    )
    # Do not class-balance probability calibration. Balancing would force an
    # artificial 50/50 prior and distort probability-dependent metrics. Each
    # inner OOF row already represents exactly one training patient.
    calibrator.fit(raw_scores, labels)
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
    legacy_prediction_cache=None,
    verbose=True,
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
            legacy_prediction_cache=legacy_prediction_cache,
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
        if verbose:
            print(
                f"[INNER CV][{experiment.experiment_id}][outer={outer_fold_index}] "
                f"C={c_value:g}, pooled_inner_AUC={inner_auc:.4f}",
                flush=True,
            )

    # Prefer stronger regularization when several C values are effectively
    # indistinguishable inside the small outer-training cohort. The best AUC is
    # identified first, then the smallest C within the predeclared absolute
    # tolerance is selected. This is a deterministic one-standard-error-like
    # rule without estimating unstable fold-level standard errors from 24 rows.
    best_inner_auc = max(result["inner_auc"] for result in candidate_results)
    eligible_results = [
        result
        for result in candidate_results
        if result["inner_auc"] >= best_inner_auc - C_SELECTION_AUC_TOLERANCE
    ]
    selected = sorted(
        eligible_results,
        key=lambda result: result["c_value"],
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
        "best_candidate_inner_auc": float(best_inner_auc),
        "c_selection_auc_tolerance": float(C_SELECTION_AUC_TOLERANCE),
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
    # Compute F1 directly from the confusion-matrix counts. This avoids a
    # 0/0 intermediate when both precision and sensitivity are zero and follows
    # the standard binary definition 2*TP / (2*TP + FP + FN).
    f1 = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)

    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
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
        "f1",
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
    legacy_prediction_cache=None,
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
        f"fusion={experiment.fusion_method}, "
        f"slice_dropout={experiment.slice_dropout_rate:.0%}",
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
            legacy_prediction_cache=legacy_prediction_cache,
        )
        print(
            f"[OUTER {outer_fold}] selected_C={selection['selected_c']:g}, "
            f"selected_inner_AUC={selection['inner_auc']:.4f}, "
            f"best_candidate_inner_AUC="
            f"{selection['best_candidate_inner_auc']:.4f}, "
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
                legacy_prediction_cache=legacy_prediction_cache,
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
        fold_metric_values = compute_binary_patient_metrics(
            valid_labels,
            probabilities,
            predictions,
        )
        fold_auc = float(fold_metric_values["auc"])

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
            "best_candidate_inner_auc": float(
                selection["best_candidate_inner_auc"]
            ),
            "c_selection_auc_tolerance": float(
                selection["c_selection_auc_tolerance"]
            ),
            "inner_cv_splits": int(selection["inner_splits"]),
            "selected_threshold": float(selection["threshold"]),
            "outer_fold_auc": fold_auc,
            "outer_fold_auprc": float(fold_metric_values["auprc"]),
            "outer_fold_sensitivity": float(
                fold_metric_values["sensitivity"]
            ),
            "outer_fold_specificity": float(
                fold_metric_values["specificity"]
            ),
            "outer_fold_ppv": float(fold_metric_values["ppv"]),
            "outer_fold_npv": float(fold_metric_values["npv"]),
            "outer_fold_f1": float(fold_metric_values["f1"]),
            "outer_fold_balanced_accuracy": float(
                fold_metric_values["balanced_accuracy"]
            ),
            "outer_fold_brier_score": float(
                fold_metric_values["brier_score"]
            ),
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
        f"F1={metrics['f1']:.4f} | "
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
        "prepared_metadata": {
            "slice_dropout_rate": prepared.get("slice_dropout_rate", 0.0),
            "n_source_slices": prepared.get("n_source_slices"),
            "n_retained_slices": prepared.get("n_retained_slices"),
        },
    }

# =============================
# PIPELINE STEP 10
# REPEATED NESTED CV + PATIENT-LABEL PERMUTATION
# =============================


def _build_fold_manifest_rows_for_seed(
    patient_ids,
    patient_labels,
    patient_to_group,
    random_state,
):
    """Create a duplicate-aware fold manifest for one stability/permutation run."""

    patient_ids = np.asarray(patient_ids).astype(str)
    patient_labels = np.asarray(patient_labels, dtype=np.int64)
    group_ids = np.asarray(
        [patient_to_group[str(patient_id)] for patient_id in patient_ids]
    )
    fold_numbers = assign_stratified_patient_folds(
        patient_ids,
        patient_labels,
        group_ids,
        N_SPLITS,
        int(random_state),
    )

    group_sizes = {
        str(group_id): int(np.sum(group_ids == group_id))
        for group_id in np.unique(group_ids)
    }
    return [
        {
            "patient_id": str(patient_id),
            "true_label": int(label),
            "duplicate_component_id": str(group_id),
            "duplicate_component_size": int(group_sizes[str(group_id)]),
            "outer_fold": int(fold),
        }
        for patient_id, label, group_id, fold in zip(
            patient_ids,
            patient_labels,
            group_ids,
            fold_numbers,
        )
    ]


def run_nested_cv_auc_only(
    experiment,
    prepared,
    fold_manifest_rows,
    verbose=False,
):
    """Run the complete nested fitting path and return OOF scores without files.

    This lightweight helper is used by repeated split stability and patient-label
    permutation. It still selects C inside outer-training data and still fits the
    outer-fold model exactly as the main experiment does; it omits threshold
    metrics, bootstrap intervals, per-fold CSV files, and verbose console tables.
    """

    if prepared["unit"] != "patient":
        raise ValueError(
            "Stability/permutation evaluation is reviewed only for one-row-per-"
            "patient experiment representations."
        )

    patient_to_fold = {
        str(row["patient_id"]): int(row["outer_fold"])
        for row in fold_manifest_rows
    }
    patient_to_label = {
        str(row["patient_id"]): int(row["true_label"])
        for row in fold_manifest_rows
    }
    patient_to_group = {
        str(row["patient_id"]): str(row["duplicate_component_id"])
        for row in fold_manifest_rows
    }
    all_patients = np.asarray(sorted(patient_to_fold))

    score_by_patient = {}
    fold_by_patient = {}
    selected_c_by_patient = {}

    for outer_fold in range(1, N_SPLITS + 1):
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

        selection = select_c_and_training_threshold(
            experiment=experiment,
            prepared=prepared,
            outer_train_patient_ids=train_patients,
            outer_train_labels=train_labels,
            patient_to_group=patient_to_group,
            outer_fold_index=outer_fold,
            legacy_prediction_cache=None,
            verbose=verbose,
        )
        raw_scores, valid_labels, evaluated_patients = (
            fit_predict_raw_for_patient_sets(
                experiment=experiment,
                prepared=prepared,
                train_patients=train_patients,
                valid_patients=valid_patients,
                c_value=selection["selected_c"],
                legacy_prediction_cache=None,
            )
        )

        if experiment.classifier_type == "linear_svm":
            probabilities = apply_sigmoid_calibrator(
                selection["calibrator"],
                raw_scores,
            )
        else:
            probabilities = np.clip(raw_scores, 0.0, 1.0)

        for patient_id, label, probability in zip(
            evaluated_patients,
            valid_labels,
            probabilities,
        ):
            patient_id = str(patient_id)
            if int(label) != patient_to_label[patient_id]:
                raise RuntimeError(
                    f"Lightweight nested CV label mismatch for {patient_id}."
                )
            if patient_id in score_by_patient:
                raise RuntimeError(
                    f"Patient {patient_id} received multiple lightweight OOF scores."
                )
            score_by_patient[patient_id] = float(probability)
            fold_by_patient[patient_id] = int(outer_fold)
            selected_c_by_patient[patient_id] = float(selection["selected_c"])

    if set(score_by_patient) != set(all_patients.tolist()):
        missing = sorted(set(all_patients.tolist()) - set(score_by_patient))
        raise RuntimeError(
            f"Lightweight nested CV is missing OOF patients: {missing}."
        )

    ordered_patients = np.asarray(sorted(score_by_patient))
    labels = np.asarray(
        [patient_to_label[patient_id] for patient_id in ordered_patients],
        dtype=np.int64,
    )
    scores = np.asarray(
        [score_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.float64,
    )
    folds = np.asarray(
        [fold_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.int64,
    )
    selected_cs = np.asarray(
        [selected_c_by_patient[patient_id] for patient_id in ordered_patients],
        dtype=np.float64,
    )
    return ordered_patients, labels, scores, folds, selected_cs


def run_repeated_nested_cv_stability(
    experiment,
    prepared,
    base_fold_manifest_rows,
    output_dir,
):
    """Repeat the primary nested CV over several deterministic outer splits."""

    output_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = np.asarray(prepared["patient_ids"]).astype(str)
    patient_labels = np.asarray(prepared["y"], dtype=np.int64)
    order = np.argsort(patient_ids)
    patient_ids = patient_ids[order]
    patient_labels = patient_labels[order]

    patient_to_group = {
        str(row["patient_id"]): str(row["duplicate_component_id"])
        for row in base_fold_manifest_rows
    }

    run_rows = []
    oof_rows = []
    score_values_by_patient = defaultdict(list)

    print(
        f"[STABILITY] Running {REPEATED_NESTED_CV_REPEATS} repeated nested "
        f"patient-level splits for {experiment.experiment_id}.",
        flush=True,
    )

    for repeat_index in range(REPEATED_NESTED_CV_REPEATS):
        seed = REPEATED_NESTED_CV_RANDOM_STATE + repeat_index
        fold_rows = _build_fold_manifest_rows_for_seed(
            patient_ids,
            patient_labels,
            patient_to_group,
            seed,
        )
        (
            ordered_patients,
            labels,
            scores,
            folds,
            selected_cs,
        ) = run_nested_cv_auc_only(
            experiment,
            prepared,
            fold_rows,
            verbose=False,
        )

        auc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        run_rows.append(
            {
                "repeat_index": int(repeat_index + 1),
                "outer_cv_random_state": int(seed),
                "auc": auc,
                "auprc": auprc,
                "median_selected_c": float(np.median(selected_cs)),
            }
        )
        for patient_id, label, fold, score, selected_c in zip(
            ordered_patients,
            labels,
            folds,
            scores,
            selected_cs,
        ):
            score_values_by_patient[str(patient_id)].append(float(score))
            oof_rows.append(
                {
                    "repeat_index": int(repeat_index + 1),
                    "outer_cv_random_state": int(seed),
                    "patient_id": str(patient_id),
                    "true_label": int(label),
                    "outer_fold": int(fold),
                    "oof_score": float(score),
                    "selected_c": float(selected_c),
                }
            )

        print(
            f"[STABILITY] repeat={repeat_index + 1}/"
            f"{REPEATED_NESTED_CV_REPEATS}, seed={seed}, "
            f"AUC={auc:.4f}, AUPRC={auprc:.4f}",
            flush=True,
        )

    auc_values = np.asarray([row["auc"] for row in run_rows], dtype=np.float64)
    patient_rows = []
    label_by_patient = {
        str(patient_id): int(label)
        for patient_id, label in zip(patient_ids, patient_labels)
    }
    for patient_id in sorted(score_values_by_patient):
        values = np.asarray(score_values_by_patient[patient_id], dtype=np.float64)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "true_label": int(label_by_patient[patient_id]),
                "mean_oof_score": float(np.mean(values)),
                "std_oof_score": float(np.std(values)),
                "minimum_oof_score": float(np.min(values)),
                "maximum_oof_score": float(np.max(values)),
                "score_range": float(np.max(values) - np.min(values)),
            }
        )

    summary = {
        "status": "OK",
        "experiment_id": experiment.experiment_id,
        "repeats": int(REPEATED_NESTED_CV_REPEATS),
        "auc_mean": float(np.mean(auc_values)),
        "auc_median": float(np.median(auc_values)),
        "auc_std": float(np.std(auc_values)),
        "auc_q25": float(np.quantile(auc_values, 0.25)),
        "auc_q75": float(np.quantile(auc_values, 0.75)),
        "auc_minimum": float(np.min(auc_values)),
        "auc_maximum": float(np.max(auc_values)),
        "interpretation": (
            "All repeats are reported. No split is selected according to AUC. "
            "This measures split sensitivity and does not replace external validation."
        ),
    }

    with open(
        output_dir / "repeated_nested_cv_runs.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    with open(
        output_dir / "repeated_nested_cv_oof_predictions.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(oof_rows[0]))
        writer.writeheader()
        writer.writerows(oof_rows)
    with open(
        output_dir / "patient_score_stability.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)
    (output_dir / "repeated_nested_cv_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "[STABILITY] AUC median="
        f"{summary['auc_median']:.4f}, IQR=[{summary['auc_q25']:.4f}, "
        f"{summary['auc_q75']:.4f}], range=[{summary['auc_minimum']:.4f}, "
        f"{summary['auc_maximum']:.4f}]",
        flush=True,
    )
    return summary


def run_patient_label_permutation_test(
    experiment,
    prepared,
    base_fold_manifest_rows,
    observed_auc,
    output_dir,
):
    """Estimate a patient-level empirical null distribution for nested-CV AUC."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if prepared["unit"] != "patient":
        raise ValueError(
            "Patient-label permutation requires one-row-per-patient prepared data."
        )

    patient_ids = np.asarray(prepared["patient_ids"]).astype(str)
    original_labels = np.asarray(prepared["y"], dtype=np.int64)
    order = np.argsort(patient_ids)
    patient_ids = patient_ids[order]
    original_labels = original_labels[order]
    original_X = np.asarray(prepared["X"])[order]

    patient_to_group = {
        str(row["patient_id"]): str(row["duplicate_component_id"])
        for row in base_fold_manifest_rows
    }
    rng = np.random.default_rng(LABEL_PERMUTATION_RANDOM_STATE)
    rows = []
    progress_interval = max(1, LABEL_PERMUTATION_REPLICATES // 10)

    print(
        f"[PERMUTATION] Running {LABEL_PERMUTATION_REPLICATES} patient-label "
        f"permutations for {experiment.experiment_id}.",
        flush=True,
    )

    for permutation_index in range(LABEL_PERMUTATION_REPLICATES):
        permuted_labels = rng.permutation(original_labels)
        permuted_prepared = dict(prepared)
        permuted_prepared["patient_ids"] = patient_ids
        permuted_prepared["X"] = original_X
        permuted_prepared["y"] = permuted_labels

        fold_seed = LABEL_PERMUTATION_RANDOM_STATE + permutation_index + 1
        fold_rows = _build_fold_manifest_rows_for_seed(
            patient_ids,
            permuted_labels,
            patient_to_group,
            fold_seed,
        )
        _, labels, scores, _, _ = run_nested_cv_auc_only(
            experiment,
            permuted_prepared,
            fold_rows,
            verbose=False,
        )
        auc = float(roc_auc_score(labels, scores))
        rows.append(
            {
                "permutation_index": int(permutation_index + 1),
                "outer_cv_random_state": int(fold_seed),
                "permuted_auc": auc,
            }
        )

        if (
            permutation_index == 0
            or (permutation_index + 1) % progress_interval == 0
            or permutation_index + 1 == LABEL_PERMUTATION_REPLICATES
        ):
            print(
                f"[PERMUTATION] {permutation_index + 1}/"
                f"{LABEL_PERMUTATION_REPLICATES} complete; latest AUC={auc:.4f}",
                flush=True,
            )

    null_aucs = np.asarray(
        [row["permuted_auc"] for row in rows], dtype=np.float64
    )
    empirical_p_value = float(
        (1 + np.sum(null_aucs >= float(observed_auc)))
        / (LABEL_PERMUTATION_REPLICATES + 1)
    )
    summary = {
        "status": "OK",
        "experiment_id": experiment.experiment_id,
        "observed_auc": float(observed_auc),
        "permutation_replicates": int(LABEL_PERMUTATION_REPLICATES),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_median": float(np.median(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
        "null_auc_q025": float(np.quantile(null_aucs, 0.025)),
        "null_auc_q975": float(np.quantile(null_aucs, 0.975)),
        "empirical_one_sided_p_value": empirical_p_value,
        "permutation_unit": "Directory_* patient labels",
        "interpretation": (
            "The complete patient-level nested fitting procedure is repeated "
            "after label permutation. This is an implementation/signal sanity "
            "test, not a substitute for external validation."
        ),
    }

    with open(
        output_dir / "patient_label_permutation_auc.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "patient_label_permutation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "[PERMUTATION] observed AUC="
        f"{observed_auc:.4f}, null median={summary['null_auc_median']:.4f}, "
        f"empirical p={empirical_p_value:.6f}",
        flush=True,
    )
    return summary


# =============================
# PIPELINE STEP 11
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


def compare_predeclared_ablation_pairs(results):
    """Calculate the scientifically matched paired comparisons declared above.

    Unlike the ranking table, which compares every successful experiment with
    the main baseline for orientation, this table uses the reference chosen for
    each scientific question. In particular, the legacy slice-classifier branch
    is compared with A8, which matches no-PCA and fixed-C settings, and the two
    fusion rules are compared directly with one another.
    """

    successful = {
        result["config"].experiment_id: result
        for result in results
        if result.get("status") == "OK"
    }
    rows = []

    for (
        comparison_name,
        reference_id,
        comparison_id,
        scientific_question,
    ) in PRIMARY_ABLATION_COMPARISONS:
        missing = [
            experiment_id
            for experiment_id in (reference_id, comparison_id)
            if experiment_id not in successful
        ]
        if missing:
            rows.append(
                {
                    "comparison_name": comparison_name,
                    "status": "SKIPPED_MISSING_EXPERIMENT",
                    "reference_experiment_id": reference_id,
                    "comparison_experiment_id": comparison_id,
                    "scientific_question": scientific_question,
                    "reference_auc": None,
                    "comparison_auc": None,
                    "delta_auc": None,
                    "delta_auc_ci_lower": None,
                    "delta_auc_ci_upper": None,
                    "bootstrap_probability_delta_above_zero": None,
                    "missing_experiments": ";".join(missing),
                }
            )
            continue

        reference = successful[reference_id]
        comparison = successful[comparison_id]
        reference_order = np.argsort(reference["patient_ids"].astype(str))
        comparison_order = np.argsort(comparison["patient_ids"].astype(str))
        reference_patients = reference["patient_ids"][reference_order].astype(str)
        comparison_patients = comparison["patient_ids"][comparison_order].astype(str)
        reference_labels = reference["labels"][reference_order]
        comparison_labels = comparison["labels"][comparison_order]

        if not np.array_equal(reference_patients, comparison_patients):
            raise RuntimeError(
                f"Primary comparison {comparison_name} uses different patient sets."
            )
        if not np.array_equal(reference_labels, comparison_labels):
            raise RuntimeError(
                f"Primary comparison {comparison_name} uses inconsistent labels."
            )

        reference_scores = reference["probabilities"][reference_order]
        comparison_scores = comparison["probabilities"][comparison_order]
        interval = paired_auc_difference_interval(
            reference_labels,
            reference_scores,
            comparison_scores,
            random_state=RANDOM_SEED
            + int(
                hashlib.sha256(comparison_name.encode()).hexdigest()[:8],
                16,
            ),
        )
        rows.append(
            {
                "comparison_name": comparison_name,
                "status": "OK",
                "reference_experiment_id": reference_id,
                "comparison_experiment_id": comparison_id,
                "scientific_question": scientific_question,
                "reference_auc": float(
                    roc_auc_score(reference_labels, reference_scores)
                ),
                "comparison_auc": float(
                    roc_auc_score(reference_labels, comparison_scores)
                ),
                **interval,
                "missing_experiments": "",
            }
        )

    return rows


def print_primary_ablation_comparisons(rows):
    """Print the matched Delta-AUROC table separately from the ranking table."""

    table_width = 180
    print("\n" + "=" * table_width, flush=True)
    print("PREDECLARED MATCHED ABLATION COMPARISONS", flush=True)
    print("=" * table_width, flush=True)
    print(
        f"{'Comparison':<49} {'Reference -> Changed':<76} "
        f"{'Delta AUC [95% CI]':<29} {'Status':<22}",
        flush=True,
    )
    print("-" * table_width, flush=True)
    for row in rows:
        direction = (
            f"{row['reference_experiment_id']} -> "
            f"{row['comparison_experiment_id']}"
        )
        if row["status"] == "OK":
            delta_text = (
                f"{row['delta_auc']:+.4f} "
                f"[{row['delta_auc_ci_lower']:+.4f}, "
                f"{row['delta_auc_ci_upper']:+.4f}]"
            )
        else:
            delta_text = "not calculated"
        print(
            f"{row['comparison_name']:<49} {direction:<76} "
            f"{delta_text:<29} {row['status']:<22}",
            flush=True,
        )
    print("-" * table_width, flush=True)
    print(
        "Delta AUC is changed configuration minus its declared reference. "
        "A confidence interval containing zero does not establish a reliable "
        "difference.",
        flush=True,
    )
    print("=" * table_width, flush=True)


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
        "slice_dropout_rate": float(experiment.slice_dropout_rate),
        "n_source_slices": result["prepared_metadata"].get(
            "n_source_slices"
        ),
        "n_retained_slices": result["prepared_metadata"].get(
            "n_retained_slices"
        ),
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
        "f1": metrics["f1"],
        "f1_ci_lower": intervals["f1"][0],
        "f1_ci_upper": intervals["f1"][1],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "brier_score": metrics["brier_score"],
        "mean_training_only_threshold": float(
            np.mean(result["thresholds"])
        ),
        "median_selected_c": float(np.median(result["selected_cs"])),
        "delta_auc_vs_baseline": comparison.get("delta_auc"),
        "delta_auc_ci_lower": comparison.get("delta_auc_ci_lower"),
        "delta_auc_ci_upper": comparison.get("delta_auc_ci_upper"),
        "runtime_seconds": summary["runtime_seconds"],
    }


def write_master_outputs(
    results,
    failed_results,
    paired_rows,
    primary_ablation_rows,
    comparison_dir,
):
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
    else:
        # Keep a visible completion artifact even when every experiment fails.
        # The detailed error records remain in failed_experiments.csv and in
        # each experiment's own failure.json file.
        summary_path.write_text(
            "experiment_id,status,role,description\n",
            encoding="utf-8",
        )

    # Failures are saved in a dedicated flat CSV in addition to final_report.json
    # so spreadsheet inspection does not require parsing nested JSON. Tracebacks
    # are preserved because they are often necessary to distinguish a data
    # problem from an optional experiment or library failure.
    failure_path = comparison_dir / "failed_experiments.csv"
    failure_fields = [
        "experiment_id",
        "status",
        "error_type",
        "error_message",
        "traceback",
    ]
    with open(failure_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=failure_fields)
        writer.writeheader()
        for failure in failed_results:
            writer.writerow(
                {field: failure.get(field, "") for field in failure_fields}
            )

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
    prediction_fields = [
        "experiment_id",
        "patient_id",
        "true_label",
        "outer_fold",
        "oof_score",
        "training_only_threshold",
        "predicted_label",
        "selected_c",
    ]
    with open(prediction_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    primary_paired_path = (
        comparison_dir / "paired_primary_ablation_comparisons.csv"
    )
    primary_fields = [
        "comparison_name",
        "status",
        "reference_experiment_id",
        "comparison_experiment_id",
        "scientific_question",
        "reference_auc",
        "comparison_auc",
        "delta_auc",
        "delta_auc_ci_lower",
        "delta_auc_ci_upper",
        "bootstrap_probability_delta_above_zero",
        "missing_experiments",
    ]
    with open(primary_paired_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=primary_fields)
        writer.writeheader()
        for row in primary_ablation_rows:
            writer.writerow({field: row.get(field, "") for field in primary_fields})

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
        "paired_auc_comparisons_vs_baseline": paired_rows,
        "paired_primary_ablation_comparisons": primary_ablation_rows,
        "interpretation_rules": {
            "negative_control_warning_auc": SHORTCUT_WARNING_AUC,
            "delta_auc_ci": (
                "A paired CI containing zero does not establish a reliable "
                "difference between configurations."
            ),
            "calibration_warning": (
                "Logistic-Regression outputs are class-weighted model scores, "
                "not externally calibrated clinical probabilities. Linear-SVM "
                "margins receive training-only sigmoid calibration. Brier scores "
                "must therefore be interpreted as exploratory."
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


def print_final_comparison(summary_rows, failed_results=None):
    """Print a compact ranking, shortcut warnings, and failed experiments."""

    if failed_results is None:
        failed_results = []

    print("\n" + "=" * 126, flush=True)
    print("FINAL MULTI-EXPERIMENT PATIENT-LEVEL COMPARISON", flush=True)
    print("=" * 126, flush=True)
    header = (
        f"{'Rank':<5} {'Experiment':<38} {'Role':<17} "
        f"{'AUC [95% CI]':<25} {'Delta AUC vs baseline':<22} "
        f"{'Sens.':>7} {'Spec.':>7} {'F1':>7}"
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
            f"{row['sensitivity']:>7.3f} {row['specificity']:>7.3f} "
            f"{row['f1']:>7.3f}",
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

    if failed_results:
        print("\nFAILED EXPERIMENTS", flush=True)
        for failure in failed_results:
            print(
                f"[FAILED] {failure.get('experiment_id', 'unknown')}: "
                f"{failure.get('error_type', 'Error')}: "
                f"{failure.get('error_message', '')}",
                flush=True,
            )
        print(
            "Detailed tracebacks: comparison/failed_experiments.csv and each "
            "experiment's failure.json.",
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
    standardized_monai_gate_comparison,
    stability_summary,
    permutation_summary,
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
        "feature_cache_schema": FEATURE_CACHE_SCHEMA_VERSION,
        "feature_modes_per_encoder_call": FEATURE_MODES_PER_ENCODER_CALL,
        "outer_cv_splits": N_SPLITS,
        "outer_cv_random_state": CV_RANDOM_STATE,
        "inner_cv_splits": INNER_CV_SPLITS,
        "inner_cv_random_state": INNER_CV_RANDOM_STATE,
        "classifier_c_grid": list(CLASSIFIER_C_GRID),
        "c_selection_auc_tolerance": C_SELECTION_AUC_TOLERANCE,
        "svm_calibration_c": SVM_CALIBRATION_C,
        "threshold_selection_method": THRESHOLD_SELECTION_METHOD,
        "target_sensitivity": TARGET_SENSITIVITY,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "use_cuda_amp": USE_CUDA_AMP,
        "batch_size": BATCH_SIZE,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
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
        "monai_expected_model_ts_sha256": (
            MONAI_OFFICIAL_TORCHSCRIPT_SHA256
        ),
        "monai_expected_model_pt_sha256": MONAI_MODEL_SHA256,
        "efficientnet_weights": EFFICIENTNET_WEIGHTS_NAME,
        "provenance_classifier_feature_names": list(
            PROVENANCE_CLASSIFIER_FEATURE_NAMES
        ),
        "standardization_feature_names": list(
            STANDARDIZATION_FEATURE_NAMES
        ),
        "label_blind_standardization": {
            "lower_percentile": STANDARDIZATION_LOWER_PERCENTILE,
            "upper_percentile": STANDARDIZATION_UPPER_PERCENTILE,
            "dark_line_max_mean": STANDARDIZATION_DARK_LINE_MAX_MEAN,
            "dark_line_max_std": STANDARDIZATION_DARK_LINE_MAX_STD,
            "dark_pixel_max_value": STANDARDIZATION_DARK_PIXEL_MAX_VALUE,
            "dark_pixel_min_fraction": (
                STANDARDIZATION_DARK_PIXEL_MIN_FRACTION
            ),
            "max_crop_fraction_per_side": (
                STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE
            ),
            "minimum_retained_fraction": (
                STANDARDIZATION_MIN_RETAINED_FRACTION
            ),
            "minimum_padding_run": STANDARDIZATION_MIN_PADDING_RUN,
        },
        "exact_duplicate_audit": exact_summary,
        "perceptual_duplicate_audit": phash_summary,
        "monai_gate_comparison": monai_gate_comparison,
        "standardized_monai_gate_comparison": (
            standardized_monai_gate_comparison
        ),
        "repeated_nested_cv_stability": stability_summary,
        "patient_label_permutation_test": permutation_summary,
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
    checkpoint_path = bundle_root / "models" / "model.pt"

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


def write_suite_configuration(output_path, experiments):
    """Save every predeclared setting before model evaluation starts."""

    configuration = {
        "suite_name": SUITE_NAME,
        "suite_configuration_tag": SUITE_CONFIGURATION_TAG,
        "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "experiments": [asdict(experiment) for experiment in experiments],
        "primary_ablation_comparisons": [
            {
                "comparison_name": comparison_name,
                "reference_experiment_id": reference_id,
                "comparison_experiment_id": comparison_id,
                "scientific_question": scientific_question,
            }
            for (
                comparison_name,
                reference_id,
                comparison_id,
                scientific_question,
            ) in PRIMARY_ABLATION_COMPARISONS
        ],
        "outer_cv_splits": N_SPLITS,
        "outer_cv_random_state": CV_RANDOM_STATE,
        "inner_cv_splits": INNER_CV_SPLITS,
        "inner_cv_random_state": INNER_CV_RANDOM_STATE,
        "classifier_c_grid": list(CLASSIFIER_C_GRID),
        "c_selection_auc_tolerance": C_SELECTION_AUC_TOLERANCE,
        "logistic_max_iter": LOGISTIC_MAX_ITER,
        "svm_max_iter": SVM_MAX_ITER,
        "svm_calibration_c": SVM_CALIBRATION_C,
        "threshold_selection_method": THRESHOLD_SELECTION_METHOD,
        "target_sensitivity": TARGET_SENSITIVITY,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "patient_pca_explained_variance": (
            PATIENT_PCA_EXPLAINED_VARIANCE
        ),
        "batch_size": BATCH_SIZE,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "use_cuda_amp": USE_CUDA_AMP,
        "feature_cache_schema": FEATURE_CACHE_SCHEMA_VERSION,
        "feature_modes_per_encoder_call": FEATURE_MODES_PER_ENCODER_CALL,
        "slice_quality_min_weight": SLICE_QUALITY_MIN_WEIGHT,
        "border_width_fraction": BORDER_WIDTH_FRACTION,
        "standardized_border_width_fractions": list(
            STANDARDIZED_BORDER_WIDTH_FRACTIONS
        ),
        "standardized_corner_fraction": STANDARDIZED_CORNER_WIDTH_FRACTION,
        "center_crop_fallback_fraction": CENTER_CROP_FALLBACK_FRACTION,
        "monai_bbox_context_fraction": MONAI_BBOX_CONTEXT_FRACTION,
        "outside_monai_bbox_context_fraction": (
            OUTSIDE_MONAI_BBOX_CONTEXT_FRACTION
        ),
        "label_blind_standardization": {
            "lower_percentile": STANDARDIZATION_LOWER_PERCENTILE,
            "upper_percentile": STANDARDIZATION_UPPER_PERCENTILE,
            "dark_line_max_mean": STANDARDIZATION_DARK_LINE_MAX_MEAN,
            "dark_line_max_std": STANDARDIZATION_DARK_LINE_MAX_STD,
            "dark_pixel_max_value": STANDARDIZATION_DARK_PIXEL_MAX_VALUE,
            "dark_pixel_min_fraction": (
                STANDARDIZATION_DARK_PIXEL_MIN_FRACTION
            ),
            "max_crop_fraction_per_side": (
                STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE
            ),
            "minimum_retained_fraction": (
                STANDARDIZATION_MIN_RETAINED_FRACTION
            ),
            "minimum_padding_run": STANDARDIZATION_MIN_PADDING_RUN,
            "feature_names": list(STANDARDIZATION_FEATURE_NAMES),
        },
        "provenance_classifier_feature_names": list(
            PROVENANCE_CLASSIFIER_FEATURE_NAMES
        ),
        "patient_definition": "Directory_*",
        "audit_exact_decoded_pixel_duplicates": (
            AUDIT_EXACT_DECODED_PIXEL_DUPLICATES
        ),
        "audit_perceptual_near_duplicates": (
            AUDIT_PERCEPTUAL_NEAR_DUPLICATES
        ),
        "phash_hamming_threshold": PHASH_HAMMING_THRESHOLD,
        "phash_search_method": (
            "complete_bk_tree_unique_phash_values"
            if PHASH_USE_COMPLETE_BK_TREE_AUDIT
            else "unsupported"
        ),
        "phash_candidate_interpretation": (
            "screening candidates requiring manual or stronger similarity review"
        ),
        "group_splits_by_exact_duplicates": (
            GROUP_SPLITS_BY_EXACT_DUPLICATES
        ),
        "group_splits_by_phash_candidates": (
            GROUP_SPLITS_BY_PHASH_CANDIDATES
        ),
        "fail_on_cross_patient_exact_duplicates": (
            FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES
        ),
        "fail_on_cross_label_exact_duplicates": (
            FAIL_ON_CROSS_LABEL_EXACT_DUPLICATES
        ),
        "shortcut_warning_auc": SHORTCUT_WARNING_AUC,
        "monai_gate_rate_difference_warning": (
            MONAI_GATE_RATE_DIFFERENCE_WARNING
        ),
        "repeated_nested_cv_stability": {
            "enabled": RUN_REPEATED_NESTED_CV_STABILITY,
            "experiment_id": STABILITY_EXPERIMENT_ID,
            "repeats": REPEATED_NESTED_CV_REPEATS,
            "random_state": REPEATED_NESTED_CV_RANDOM_STATE,
        },
        "patient_label_permutation_test": {
            "enabled": RUN_PATIENT_LABEL_PERMUTATION_TEST,
            "experiment_id": PERMUTATION_EXPERIMENT_ID,
            "replicates": LABEL_PERMUTATION_REPLICATES,
            "random_state": LABEL_PERMUTATION_RANDOM_STATE,
        },
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

    # ======================================================================
    # MAIN STAGE 1 -- VALIDATE THE COMPLETE MULTI-EXPERIMENT CONFIGURATION
    # ======================================================================
    # Validation occurs before dataset scanning, model downloads, GPU
    # allocation, cache creation, or experiment fitting. In addition to the
    # original image-pipeline checks, this suite validates the experiment
    # registry, outer/inner patient-level CV settings, classifier-C grid,
    # training-only threshold policy, duplicate-audit settings, negative-control
    # definitions, and the requested feature-bank modes. Failing here prevents a
    # malformed configuration from producing partially valid-looking outputs.
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

    # ======================================================================
    # MAIN STAGE 2 -- CREATE A CLEAN OUTPUT PACKAGE AND FREEZE THE PROTOCOL
    # ======================================================================
    # Every enabled experiment and all shared analysis choices are serialized
    # before any result is calculated. This makes the run auditable and prevents
    # a later result-dependent change from being mistaken for a predeclared
    # setting. Only run-specific subdirectories are removed on an intentional
    # rerun; the shared frozen-feature cache remains outside OUTPUT_DIR and is
    # reused only when its complete fingerprint matches.
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
        "stability",
        "permutation",
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

    # ======================================================================
    # MAIN STAGE 3 -- DISCOVER THE RELEASED COHORT WITHOUT DECODING PIXELS
    # ======================================================================
    # ``load_samples`` preserves the validated computational patient unit
    # ``Directory_*`` and the original detailed series-proxy interpretation.
    # It scans Normal/ and Sick/, rejects cross-class patient-ID collisions,
    # assigns every image to one patient-scoped immediate-child-folder proxy,
    # and returns one deterministic metadata row per image. No label is supplied
    # to MONAI or EfficientNet, and no JPEG is decoded in this stage.
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

    # ======================================================================
    # MAIN STAGE 4 -- LOAD OR EXTRACT ONE SHARED MULTI-VIEW FEATURE BANK
    # ======================================================================
    # This is the expensive image-processing stage. On a cache miss, each JPEG
    # is decoded once. MONAI is run on the original canvas and, separately, on
    # the label-blind standardized canvas because padding removal and robust
    # scaling can legitimately change its probability map. All enabled image
    # variants are then derived from those aligned tensors/masks. EfficientNet
    # embeddings are written to memory-mapped arrays for the original views and
    # for standardized full, ROI, narrow-border, corners, detected padding,
    # center-crop, strict-ROI and outside-bounding-box controls. Small groups of
    # modes are concatenated per encoder call to reduce launch overhead.
    #
    # Downstream pooling, classifier, PCA, weighting, and fusion ablations reuse
    # these frozen arrays and therefore do not repeat neural inference. The cache
    # fingerprint includes dataset metadata and every setting capable of changing
    # the frozen representations, including AMP/device semantics.
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

    # ======================================================================
    # MAIN STAGE 5 -- AUDIT EXACT AND NEAR-DUPLICATE VISUAL CONTENT
    # ======================================================================
    # The original exact decoded-pixel audit is retained. The suite additionally
    # generates perceptual-hash candidates for images whose pixels may differ
    # slightly after JPEG recompression or small export changes. Exact matches
    # are trusted as deterministic duplicate evidence; pHash matches remain
    # candidates unless they are manually verified. Both audits occur before
    # the final fold manifest is created because cross-patient visual duplicates
    # can leak content even when Directory_* identities themselves never cross
    # train/validation boundaries.
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

    # ======================================================================
    # MAIN STAGE 6 -- FREEZE ONE DUPLICATE-AWARE PATIENT FOLD MANIFEST
    # ======================================================================
    # All experiments must evaluate the same held-out patients. Connected
    # components induced by accepted duplicate edges are therefore treated as
    # split groups while ``patient_id`` remains exactly Directory_*. The output
    # manifests record the patient label, duplicate component, outer fold, image
    # path, series proxy, image geometry, hashes, and MONAI diagnostics. This
    # explicit manifest is safer than allowing every experiment to call a random
    # splitter independently and enables paired patient-level comparisons.
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

    # ======================================================================
    # MAIN STAGE 7 -- BUILD NON-ANATOMICAL CONFOUNDING CONTROLS
    # ======================================================================
    # Patient-level provenance controls use a conservative subset dominated by
    # geometry, file size, padding and border structure rather than central
    # anatomical texture. Standardization-QC controls use only crop/padding and
    # robust-range metadata produced by the fixed label-blind preprocessing.
    # Original and standardized MONAI-QC controls summarize gate rate, mask
    # area, confidence and fallback behavior. All tables are saved descriptively
    # and evaluated through the same outer folds. A high control AUC indicates
    # that Normal/Sick may be predictable from acquisition/export/protocol
    # provenance rather than exclusively from CAD-related anatomy.
    stage_started = _print_stage_start(
        7,
        "Build and save provenance, standardization and MONAI-QC controls",
        "Patient-level aggregation and descriptive audit files.",
    )
    provenance_set = aggregate_patient_provenance_features(bank)
    standardization_set = aggregate_patient_standardization_features(bank)
    monai_gate_comparison, monai_qc_set = write_monai_qc_outputs(
        OUTPUT_DIR / "audits",
        bank,
    )
    (
        standardized_monai_gate_comparison,
        standardized_monai_qc_set,
    ) = write_standardized_monai_qc_outputs(
        OUTPUT_DIR / "audits",
        bank,
    )
    write_patient_tabular_features(
        OUTPUT_DIR / "audits" / "patient_provenance_features.csv",
        *provenance_set,
    )
    write_patient_tabular_features(
        OUTPUT_DIR / "audits" / "patient_standardization_features.csv",
        *standardization_set,
    )
    write_tabular_class_summary(
        OUTPUT_DIR / "audits" / "provenance_class_summary.csv",
        provenance_set[0],
        provenance_set[1],
        provenance_set[3],
    )
    write_tabular_class_summary(
        OUTPUT_DIR / "audits" / "standardization_class_summary.csv",
        standardization_set[0],
        standardization_set[1],
        standardization_set[3],
    )
    write_tabular_class_summary(
        OUTPUT_DIR / "audits" / "monai_qc_class_summary.csv",
        monai_qc_set[0],
        monai_qc_set[1],
        monai_qc_set[3],
    )
    write_tabular_class_summary(
        OUTPUT_DIR / "audits" / "standardized_monai_qc_class_summary.csv",
        standardized_monai_qc_set[0],
        standardized_monai_qc_set[1],
        standardized_monai_qc_set[3],
    )
    tabular_feature_sets = {
        "provenance_only": provenance_set,
        "monai_qc_only": monai_qc_set,
        "standardization_qc_only": standardization_set,
        "standardized_monai_qc_only": standardized_monai_qc_set,
    }
    stage_durations["07 QC/provenance controls"] = _print_stage_complete(
        7,
        "Build and save provenance, standardization and MONAI-QC controls",
        stage_started,
        "Patient-level control matrices are ready for provenance, original/"
        "standardized MONAI QC, and standardization-geometry controls.",
    )

    # ======================================================================
    # MAIN STAGE 8 -- RUN THE PREDECLARED EXPERIMENT REGISTRY
    # ======================================================================
    # Each experiment receives the same frozen outer-fold assignments. Any
    # quantitatively selected classifier C, SVM sigmoid calibration, or decision
    # threshold is learned only from inner out-of-fold predictions belonging to
    # the current outer-training cohort. The outer-validation fold remains
    # untouched until the configuration for that fold is locked. Prepared
    # representations are cached in memory, and identical fold-local legacy
    # slice probabilities are reused between mean-probability and log-odds
    # fusion so that the fusion ablation changes only the aggregation rule.
    #
    # One experiment failure is written to its own failure.json and does not
    # erase successful results unless FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS=True.
    stage_started = _print_stage_start(
        8,
        "Run every enabled experiment on the shared folds",
        "Several fold-local fits. Patient-embedding experiments are fast; "
        "legacy slice classifiers are substantially heavier.",
    )
    successful_results = []
    failed_results = []
    prepared_cache = {}
    legacy_prediction_cache = {}

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
            experiment.slice_dropout_rate,
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
                legacy_prediction_cache=legacy_prediction_cache,
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

    # ======================================================================
    # MAIN STAGE 9 -- COMPARE EXPERIMENTS ON IDENTICAL PATIENT RESAMPLES
    # ======================================================================
    # Stand-alone AUC confidence intervals do not establish whether two models
    # differ. The suite therefore aligns each successful experiment with the
    # baseline by patient ID and computes paired bootstrap distributions of
    # Delta-AUROC using the same resampled patients for both score vectors. It
    # then consolidates experiment summaries, all OOF patient predictions,
    # paired comparisons, and failures into CSV/JSON files for later tables,
    # plots, and manuscript/poster preparation.
    stage_started = _print_stage_start(
        9,
        "Calculate paired comparisons and write master result files",
        "Patient-level bootstrap comparisons and CSV/JSON consolidation.",
    )
    paired_rows = compare_experiments_to_baseline(successful_results)
    primary_ablation_rows = compare_predeclared_ablation_pairs(
        successful_results
    )
    summary_rows = write_master_outputs(
        successful_results,
        failed_results,
        paired_rows,
        primary_ablation_rows,
        OUTPUT_DIR / "comparison",
    )
    stage_durations["09 Master comparisons"] = _print_stage_complete(
        9,
        "Calculate paired comparisons and write master result files",
        stage_started,
        f"Summary rows={len(summary_rows)}; baseline-paired rows={len(paired_rows)}; "
        f"matched-ablation rows={len(primary_ablation_rows)}.",
    )

    # ======================================================================
    # MAIN STAGE 10 -- REPEAT THE PRIMARY NESTED CV ACROSS OUTER SPLITS
    # ======================================================================
    # A single five-fold assignment is statistically fragile when the effective
    # sample size is only the number of Directory_* folders. This stage repeats
    # the complete nested patient-level fitting path over several deterministic
    # outer split seeds. It reuses cached patient embeddings, never chooses the
    # best split, and reports the full AUC distribution plus per-patient score
    # variability. No MONAI or EfficientNet inference is repeated.
    stage_started = _print_stage_start(
        10,
        "Run repeated nested-CV split-stability analysis",
        "Several fast patient-level reruns; frozen image features are reused.",
    )
    stability_summary = {
        "status": "SKIPPED_DISABLED",
        "experiment_id": STABILITY_EXPERIMENT_ID,
    }
    stability_result = next(
        (
            result
            for result in successful_results
            if result["config"].experiment_id == STABILITY_EXPERIMENT_ID
        ),
        None,
    )
    stability_experiment = next(
        experiment
        for experiment in experiments
        if experiment.experiment_id == STABILITY_EXPERIMENT_ID
    )
    stability_preparation_key = (
        stability_experiment.feature_mode,
        stability_experiment.strategy,
        stability_experiment.pooling_strategy,
        stability_experiment.weighting_mode,
        stability_experiment.slice_dropout_rate,
    )

    if RUN_REPEATED_NESTED_CV_STABILITY and stability_result is not None:
        stability_summary = run_repeated_nested_cv_stability(
            experiment=stability_experiment,
            prepared=prepared_cache[stability_preparation_key],
            base_fold_manifest_rows=fold_manifest_rows,
            output_dir=OUTPUT_DIR / "stability",
        )
    elif RUN_REPEATED_NESTED_CV_STABILITY:
        stability_summary = {
            "status": "SKIPPED_PRIMARY_EXPERIMENT_FAILED",
            "experiment_id": PERMUTATION_EXPERIMENT_ID,
        }
        (OUTPUT_DIR / "stability" / "repeated_nested_cv_summary.json").write_text(
            json.dumps(stability_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[STABILITY] SKIPPED because the configured primary experiment "
            "did not complete successfully.",
            flush=True,
        )
    else:
        (OUTPUT_DIR / "stability" / "repeated_nested_cv_summary.json").write_text(
            json.dumps(stability_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("[STABILITY] SKIPPED by configuration.", flush=True)

    stage_durations["10 Repeated nested CV"] = _print_stage_complete(
        10,
        "Run repeated nested-CV split-stability analysis",
        stage_started,
        stability_summary.get("status", "OK"),
    )

    # ======================================================================
    # MAIN STAGE 11 -- PATIENT-LABEL PERMUTATION SANITY TEST
    # ======================================================================
    # Labels are permuted only at the Directory_* patient level. For every
    # replicate, duplicate-aware outer folds are rebuilt and the full nested C
    # selection/fitting procedure is rerun. A null AUC distribution centered
    # near 0.5 supports the absence of an obvious implementation-level label
    # leak; it does not resolve dataset provenance confounding or replace an
    # independent hospital cohort.
    stage_started = _print_stage_start(
        11,
        "Run patient-label permutation sanity test",
        "Many lightweight patient-level fits; no neural-network inference.",
    )
    permutation_summary = {
        "status": "SKIPPED_DISABLED",
        "experiment_id": PERMUTATION_EXPERIMENT_ID,
    }
    permutation_result = next(
        (
            result
            for result in successful_results
            if result["config"].experiment_id == PERMUTATION_EXPERIMENT_ID
        ),
        None,
    )
    permutation_experiment = next(
        experiment
        for experiment in experiments
        if experiment.experiment_id == PERMUTATION_EXPERIMENT_ID
    )
    permutation_preparation_key = (
        permutation_experiment.feature_mode,
        permutation_experiment.strategy,
        permutation_experiment.pooling_strategy,
        permutation_experiment.weighting_mode,
        permutation_experiment.slice_dropout_rate,
    )

    if RUN_PATIENT_LABEL_PERMUTATION_TEST and permutation_result is not None:
        observed_auc = float(permutation_result["summary"]["metrics"]["auc"])
        permutation_summary = run_patient_label_permutation_test(
            experiment=permutation_experiment,
            prepared=prepared_cache[permutation_preparation_key],
            base_fold_manifest_rows=fold_manifest_rows,
            observed_auc=observed_auc,
            output_dir=OUTPUT_DIR / "permutation",
        )
    elif RUN_PATIENT_LABEL_PERMUTATION_TEST:
        permutation_summary = {
            "status": "SKIPPED_PRIMARY_EXPERIMENT_FAILED",
            "experiment_id": PERMUTATION_EXPERIMENT_ID,
        }
        (
            OUTPUT_DIR / "permutation" / "patient_label_permutation_summary.json"
        ).write_text(
            json.dumps(permutation_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[PERMUTATION] SKIPPED because the configured primary experiment "
            "did not complete successfully.",
            flush=True,
        )
    else:
        (
            OUTPUT_DIR / "permutation" / "patient_label_permutation_summary.json"
        ).write_text(
            json.dumps(permutation_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("[PERMUTATION] SKIPPED by configuration.", flush=True)

    stage_durations["11 Label permutation"] = _print_stage_complete(
        11,
        "Run patient-label permutation sanity test",
        stage_started,
        permutation_summary.get("status", "OK"),
    )

    # ======================================================================
    # MAIN STAGE 12 -- RECORD, BUT DO NOT FABRICATE, EXTERNAL VALIDATION
    # ======================================================================
    # External validation cannot be made valid merely by pointing the script at
    # an arbitrary cardiac dataset. A verified adapter must first establish a
    # comparable CAD endpoint, one independent patient unit, image/sequence
    # compatibility, and a locked preprocessing contract. Until such an adapter
    # exists, the suite writes an explicit SKIPPED/BLOCKED status instead of
    # silently adapting parameters after inspecting external labels.
    stage_started = _print_stage_start(
        12,
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
    stage_durations["12 External validation status"] = _print_stage_complete(
        12,
        "Record external-validation status",
        stage_started,
        external_status["status"],
    )

    # ======================================================================
    # MAIN STAGE 13 -- PRINT THE FINAL TABLE AND SAVE COMPLETE PROVENANCE
    # ======================================================================
    # The final console table ranks successful experiments but also prints
    # explicit shortcut warnings for negative controls above the configured AUC
    # threshold. The metadata JSON records software versions, model hashes,
    # feature-bank identity, patient counts, audit summaries, successful and
    # failed experiments, and total runtime. All ordinary print() output and
    # tracebacks are simultaneously preserved in console_output.log.
    stage_started = _print_stage_start(
        13,
        "Print final comparison and save suite metadata",
        "Console table, warnings, metadata JSON and timing summary.",
    )
    print_final_comparison(summary_rows, failed_results)
    print_primary_ablation_comparisons(primary_ablation_rows)

    total_runtime = time.perf_counter() - pipeline_started_at
    metadata = collect_suite_metadata(
        samples=samples,
        fingerprint=fingerprint,
        cache_status=cache_status,
        exact_summary=exact_summary,
        phash_summary=phash_summary,
        monai_gate_comparison=monai_gate_comparison,
        standardized_monai_gate_comparison=(
            standardized_monai_gate_comparison
        ),
        stability_summary=stability_summary,
        permutation_summary=permutation_summary,
        successful_results=successful_results,
        failed_results=failed_results,
        total_runtime=total_runtime,
    )
    (OUTPUT_DIR / "suite_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    stage_durations["13 Final reporting"] = _print_stage_complete(
        13,
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
