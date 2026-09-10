#%% ============================================================
# 🧠 CAD Detection from Cardiac MRI – Focused Research Validation Pipeline V7.2 strict_balanced_fixed (Single File)
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
# The focused architecture combines:
#
#   1. Patient identity fixed to Directory_*; every slice and series proxy from
#      one patient remains in one train/validation partition
#   2. Conservative label-blind removal of dark uniform native borders
#   3. Fixed-content 240-in-256 geometry with separate intensity contracts for
#      MONAI, historical classifier views and post-localization regional scaling
#   4. A pinned pretrained MONAI ventricular segmenter used only for anatomical
#      localization and fixed plausibility-gate diagnostics
#   5. A17 hard cardiac support with additional fixed dilation, content-mask
#      intersection, center fallback and no soft-confidence intensity modulation
#   6. Region-only robust intensity normalization after the final visible support
#      is defined, preventing excluded pixels from controlling its intensity scale
#   7. Explicitly pinned ImageNet EfficientNet-B0 feature extraction with the
#      1000-class head removed
#   8. Hierarchical label-free pooling: slices -> folder-defined series proxy ->
#      Directory_* patient, followed by one supervised row per patient
#   9. Duplicate-aware nested patient-level cross-validation for classifier C and
#      an operating threshold selected only from outer-training data
#  10. One locked primary candidate, one valid-only sensitivity model, one exact
#      within-patient deduplication model and one 50% slice-dropout stress test
#  11. Three compact references: locked historical A12, standardized full image
#      A9 and shape-suppressed fixed-FOV localization A16
#  12. Three exact A17-matched controls: support-only C31, intensity-shuffled C32
#      and independently normalized exact complement C33
#  13. Two broader extracardiac controls: conservative outside-whole-heart C28
#      and MONAI-independent fixed periphery C29
#  14. Four low-cost protocol/export controls: broad provenance, series count,
#      native geometry and file-size/compression proxies
#  15. Exact decoded-pixel auditing, complete pHash candidate search, blinded
#      review panels and one authoritative patient fold manifest
#  16. Fifty repeated nested-CV split seeds for every active experiment and
#      same-seed paired delta-AUROC distributions
#  17. A 1,000-replicate patient-label permutation test for locked A17
#  18. Nested candidate identity plus C selection among A17/A20/A12 and a
#      selection-adjusted patient-label permutation test
#  19. Optional blinded sequence/view-balanced analysis after manual annotation;
#      sequence and view are never inferred from SR_*/series* names
#  20. An explicit blocked/skipped external-validation stage until a compatible
#      independent patient-level CAD cohort adapter is supplied
#  21. A compact research scorecard that separates internal association,
#      mechanistic controls, residual confounding and missing validation evidence
#
# IMPORTANT METHODOLOGICAL CHANGES:
# The earliest version used one 80/20 split and discarded many slices using an
# intensity-standard-deviation cutoff. The focused version instead:
#
#   - evaluates only at the validated Directory_* patient level;
#   - keeps every successfully decoded slice in A17 and treats unreadable files
#     as explicit errors rather than silent exclusions;
#   - separates MONAI localization from MRI intensity classification so soft
#     segmenter confidence is never multiplied into A17 brightness;
#   - calculates robust intensity limits only from pixels that remain visible in
#     each final regional representation;
#   - trains on one pooled embedding per patient, avoiding repeated slice labels
#     as nominally independent supervised observations;
#   - fits StandardScaler, PCA, Logistic Regression, C and threshold separately
#     inside every outer/inner training partition;
#   - preserves A12 as an independently locked historical reference instead of
#     redefining the baseline after observing A17;
#   - keeps A9 and A16 because they isolate full-image signal and support-outline
#     dependence without recreating a large localization leaderboard;
#   - uses A20, A21 and R4 to test invalid-mask fallback, exact repeated exports
#     and severe slice subsampling with the same A17 representation;
#   - derives A17, C31, C32 and C33 from one exact support tensor so their spatial
#     contracts cannot drift across implementations;
#   - checks whether A17 adds information beyond exact support geometry, beyond
#     the exact visible intensity multiset, and beyond its exact complement;
#   - retains broader extracardiac and metadata-only controls because a large
#     candidate-control gap does not eliminate residual cohort confounding;
#   - runs the same compact panel over 50 split seeds and reports median/IQR,
#     rather than choosing a model from one favorable split;
#   - repeats patient-label permutation for A17 and repeats the compact candidate
#     selection rule inside a second selection-adjusted permutation null;
#   - writes blinded pHash and series-review material before any optional manual
#     sequence/view analysis;
#   - keeps the sequence/view stage disabled until a reviewer supplies explicit
#     annotations, and keeps external validation blocked until a compatible
#     independent cohort is documented;
#   - stores machine-readable predictions, manifests, controls, paired deltas,
#     stability summaries and a concise claim-boundary scorecard;
#   - retains detailed implementation comments while removing redundant legacy
#     experiment definitions that no longer answer a distinct scientific question.
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
# A plausibility gate checks mask size and confidence. For A17, an implausible
# mask activates one fixed central-square support rather than a full-image
# fallback; A20 removes the same rows as a predeclared sensitivity analysis.
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
# PRIMARY FOCUSED A17 PIPELINE FLOW
# ============================================================================
#
# Raw grayscale cardiac MRI JPEG slice
#    ↓
# Detect and conservatively remove only consecutive dark, nearly uniform native
# edge runs; record every crop and geometry quantity as QC metadata
#    ↓
# Build three spatially aligned views of the same retained content:
#   - min-max 256×256 canvas for the pretrained MONAI segmenter
#   - globally robust-scaled historical classifier canvas for A9/A12
#   - raw uint8/255 classifier canvas for post-localization regional scaling
#    ↓
# Run the pinned MONAI ventricular segmenter on the standardized min-max canvas
#    ↓
# Convert ventricular channels to a hard mask, apply the fixed plausibility gate
# and fixed additional dilation
#    ↓
# Build one exact A17 support:
#   valid MONAI mask → dilated binary support
#   invalid MONAI mask → fixed central square fallback
#   both cases → intersect with retained non-padding content
#    ↓
# Estimate 1st/99th-percentile limits only from visible A17-support pixels,
# enforce a minimum dynamic range, rescale to [0,1], and set the complement to 0
#    ↓
# Apply ImageNet normalization and frozen EfficientNet-B0 encoding
#    ↓
# 1280-dimensional embedding per successfully decoded slice
#    ↓
# Weighted/equal slice mean inside each folder-defined series proxy
#    ↓
# Equal mean across all series proxies belonging to one Directory_* patient
#    ↓
# Fold-local StandardScaler → PCA → Logistic Regression on one row per patient
#    ↓
# Inner patient-level OOF selection of C and decision threshold
#    ↓
# One untouched outer-fold score for each patient
#    ↓
# Pooled OOF metrics, patient bootstrap intervals, repeated split-seed stability,
# exact matched controls, permutation tests and a focused evidence scorecard
#
# Direct A17 sensitivity branches:
#   A20 → same representation, only MONAI gate-valid source rows
#   A21 → same representation after exact within-patient decoded-pixel dedup
#   R4  → same representation after deterministic 50% within-series dropout
#
# Mechanistic controls sharing the exact A17 support contract:
#   C31 → support geometry only, no MRI intensity
#   C32 → exact support and exact intensity multiset, spatial arrangement shuffled
#   C33 → independently normalized exact retained-content complement
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
#          standardized_classification   : robust [3,224,224] historical view
#          standardized_monai_image      : standardized [1,256,256] MONAI canvas
#          standardized_raw_image        : raw uint8/255 [3,224,224] V7 view
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
#          standardized_images          : robust historical [B,3,224,224]
#          standardized_monai_images    : standardized [B,1,256,256]
#          standardized_raw_images      : raw uint8/255 [B,3,224,224]
#          detected_padding_images      : binary-control [B,3,224,224]
#
#   D. MONAI outputs -- produced when a selected experiment requires MONAI
#
#          logits                  : [B, 4, 256, 256]
#          heart probability       : [B, 1, 256, 256]
#          aligned ROI probability : [B, 1, 224, 224]
#          valid_mask              : [B], one plausibility decision per slice
#
#      A failed plausibility check does not use validation labels. A17 replaces
#      the unreliable support with one fixed central square; A20 excludes the
#      same gate-invalid row as a sensitivity analysis.
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
#      The focused run stores only the nine image feature modes required by its
#      active experiments: standardized full image, historical A12, fixed-FOV
#      A16, A17, C31, C32, C33, conservative outside-heart and fixed periphery.
#      Labels, patient IDs, series IDs, MONAI QC, exact/perceptual hashes,
#      provenance features and standardization QC remain in one-to-one row
#      alignment. Any feature-affecting setting changes the cache fingerprint.
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
#       - PCA retaining the predeclared explained-variance fraction
#       - Logistic Regression
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
# The effective labeled sample is 30 Directory_* patients, not 63,425
# independent JPEG slices. A compact linear patient-level model is therefore
# more defensible than a large trainable neural classifier on repeated labels.
# The frozen segmenter and encoder provide localization and representation;
# supervised fitting occurs only after all slice information has been pooled to
# one row per patient.
#
# The focused panel is built around falsifiable questions rather than a search
# for the largest internal AUROC:
#
#   - Does cardiac localization add information over a standardized full image?
#   - Does visible support shape itself classify the cohort?
#   - Do MRI intensities add information beyond that exact shape?
#   - Does intact spatial organization add information beyond the same intensity
#     histogram and support?
#   - Is the support more predictive than its exact complement and conservative
#     image periphery?
#   - Does performance survive exact deduplication and severe slice removal?
#   - Can acquisition/export variables classify the same labels?
#   - Is the result stable to patient split seed and patient-label permutation?
#   - Does candidate selection remain valid when model identity is chosen only
#     inside outer training?
#
# This structure is suitable for a concise research supplement because a future
# modification is not accepted merely for a higher AUC. It must improve or
# preserve the locked candidate while surviving the same references, matched
# controls, robustness tests and nuisance analyses. The sequence/view and
# external-validation stages make missing evidence explicit rather than silently
# approximating it.
#
# ============================================================================
# FOCUSED EXPERIMENT PANEL
# ============================================================================
#
# Only sixteen experiments are executable in this file. Each retained row has a
# distinct role in the evidence chain:
#
#   PRIMARY AND SENSITIVITY
#   A17  Locked hard-support, region-normalized candidate
#   A20  A17 restricted to MONAI gate-valid rows
#   A21  A17 after exact within-patient decoded-pixel deduplication
#   R4   A17 after deterministic 50% within-series slice dropout
#
#   REFERENCES
#   A12  Independently locked historical strict-ROI reference
#   A16  Fixed-FOV heart-centered crop that suppresses exact support silhouette
#   A9   Label-blind standardized full-image reference
#
#   EXACT A17-MATCHED REPRESENTATION CONTROLS
#   C31  Exact A17 support with MRI intensities removed
#   C32  Exact support and visible intensity multiset with spatial order shuffled
#   C33  Independently normalized exact retained-content complement
#
#   BROADER EXTRACARDIAC CONTROLS
#   C28  Conservative outside-whole-heart proxy
#   C29  MONAI-independent fixed periphery
#
#   PROTOCOL / EXPORT CONTROLS
#   C3   Broad conservative export/provenance feature set
#   C17  Number of folder-defined series proxies only
#   C19  Native height, width and aspect-ratio summaries only
#   C20  File-size and bytes-per-native-pixel summaries only
#
# Earlier broad suites tested legacy slice classifiers, SVM, PCA-off settings,
# many overlapping crop sizes, soft-map decompositions and multiple dropout
# severities. Their run artifacts remain part of the research history, but those
# configurations are intentionally absent from this executable registry because
# they no longer answer a distinct central question.
#
# FUTURE MODIFICATION CONTRACT
# ----------------------------
# A future representation must be added explicitly to EXPERIMENT_REGISTRY and
# FUTURE_CANDIDATE_EXPERIMENT_IDS before execution. The suite then includes it in
# the active run, repeated CV, generic comparisons with A17/A12 and nested
# candidate selection. Any new masking/support rule must also declare a matched
# control in FUTURE_CONTROL_EXPERIMENT_IDS and an explicit scientific comparison;
# otherwise its anatomical interpretation remains unsupported.
#
# The focused design reduces GPU work to nine image feature modes while retaining
# all low-cost patient metadata controls. It does not claim that a small number
# of experiments removes the need for external validation.
#
# ============================================================================
# PRINCIPAL OUTPUT PACKAGE
# ============================================================================
#
#   console_output.log
#   suite_configuration.json
#   suite_run_metadata.json
#   manifests/cohort_manifest.csv
#   manifests/patient_fold_manifest.csv
#   audits/exact_decoded_pixel_duplicate_groups.csv
#   audits/perceptual_near_duplicate_patient_pairs.csv
#   audits/phash_review_panels/*.png
#   audits/series_annotation_template.csv
#   audits/series_annotation_label_key.csv
#   audits/focused_row_contract.json
#   audits/monai_qc_availability.json
#   audits/monai_gate_class_comparison.json
#   audits/monai_qc_by_patient.csv                     (only when computed)
#   audits/standardized_monai_qc_availability.json
#   audits/standardized_monai_qc_by_patient.csv        (only when computed)
#   audits/patient_provenance_features.csv
#   experiments/<experiment_id>/patient_oof_predictions.csv
#   experiments/<experiment_id>/fold_metrics.csv
#   experiments/<experiment_id>/summary.json
#   comparison/experiment_summary.csv
#   comparison/patient_predictions_all_experiments.csv
#   comparison/paired_auc_comparisons.csv
#   comparison/paired_primary_ablation_comparisons.csv
#   comparison/final_report.json
#   comparison/focused_research_scorecard.json
#   comparison/focused_research_scorecard.csv
#   comparison/focused_research_evidence_summary.md
#   stability/<experiment_id>/repeated_nested_cv_runs.csv
#   stability/<experiment_id>/repeated_nested_cv_oof_predictions.csv
#   stability/<experiment_id>/patient_score_stability.csv
#   stability/<experiment_id>/repeated_nested_cv_summary.json
#   stability/focused_repeated_cv_ranking.csv
#   stability/repeated_nested_cv_paired_deltas.csv
#   stability/repeated_nested_cv_paired_comparisons.csv
#   model_selection/candidate_selection_oof_predictions.csv
#   model_selection/candidate_selection_outer_folds.csv
#   model_selection/candidate_selection_summary.json
#   model_selection/repeated_candidate_selection_runs.csv
#   model_selection/selection_adjusted_permutation/*.csv|json
#   permutation/<experiment_id>/patient_label_permutation_auc.csv
#   permutation/<experiment_id>/patient_label_permutation_summary.json
#   sequence_view_analysis/<selection_name>/...  (only after annotation)
#   external_validation_status.json
#
# All ordinary print(), tqdm progress and tracebacks are duplicated to both the
# live console and ``console_output.log``.
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

from dataclasses import asdict, dataclass, replace
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

import math
# Greatest-common-divisor calculations for the deterministic within-support
# intensity permutation used by the spatial-information control.

# cuBLAS uses this policy when deterministic CUDA matrix multiplication is
# requested. It is set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
# KMeans is used only by the predeclared label-free series-family clustering stage.
from sklearn.cluster import KMeans
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

# The focused validation suite is intended to produce a stable, reportable
# result. Disable autotuner-dependent kernel selection and TF32, and request
# deterministic PyTorch algorithms whenever the runtime provides them.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

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
    deduplicate_exact_within_patient: bool = False
    # When True, repeated decoded-pixel copies are collapsed inside each
    # Directory_* patient before weighting and pooling. A deterministic
    # canonical row is retained for each (patient_id, decoded-pixel SHA-256)
    # group, so the experiment does not redefine patient identity or use labels.
    slice_filter: str = "all"
    # Supported values:
    #   "all" -> retain every decoded slice;
    #   "standardized_monai_valid" -> retain only slices whose standardized
    #       MONAI output passes the fixed plausibility gate. Candidate/control
    #       pairs that use this option receive exactly the same source rows.
    role: str = "ablation"
    enabled: bool = True


# ---------------------------------------------------------------------------
# FOCUSED RESEARCH PANEL
# ---------------------------------------------------------------------------
FOCUSED_SUITE_VERSION = '7.2-SB1.1'

# ---------------------------------------------------------------------------
# AUTOMATIC CROSS-CLASS SERIES HARMONIZATION — V7.2
# ---------------------------------------------------------------------------
#
# The released JPEG cohort contains folder-defined series proxies rather than
# validated DICOM SeriesInstanceUIDs or sequence/view labels. The previous
# focused run demonstrated that export geometry, file size, number of series,
# support masks and extracardiac pixels remain predictive. V7.2 therefore adds
# one PREDECLARED cohort-harmonization stage before any MONAI/EfficientNet
# extraction:
#
#   all folder-defined series proxies
#       -> label-free visual/layout descriptor per series
#       -> deterministic robust scaling
#       -> label-free K-means clustering at several resolutions
#       -> retain series whose cluster has patient support in BOTH classes
#          at a majority of resolutions
#       -> remove class-exclusive, rare and cluster-distance-outlying series
#
# The descriptor and K-means fit never receive class labels. Class labels are
# used only AFTER clustering to ask whether a discovered series family occurs
# in both Normal and Sick. No AUC, classifier score, fold result or outcome-
# dependent threshold is used to tune the selection.
#
# IMPORTANT STATISTICAL LIMITATION:
# This default is a GLOBAL COHORT-HARMONIZATION SENSITIVITY ANALYSIS. Because
# both class folders are consulted to define shared cluster support before CV,
# its downstream AUC must not be described as independent external validation.
# A publication-grade confirmation should repeat the harmonization fold-locally
# or, preferably, lock this rule and evaluate it on a separate cohort.

AUTOMATIC_CROSS_CLASS_SERIES_HARMONIZATION = True
SERIES_HARMONIZATION_DESCRIPTOR_VERSION = "series-layout-dct-hist-v3-focused"
SERIES_HARMONIZATION_POLICY_VERSION = "multires-shared-crossclass-v3-focused"
SERIES_HARMONIZATION_SCOPE = "global_cohort_sensitivity_not_external_validation"

SERIES_HARMONIZATION_SAMPLE_FRAMES_PER_SERIES = 7
# Deterministic equally spaced frame positions are sampled. Every retained image
# is still processed by the main feature bank; sampling is used only to build a
# lightweight series-family descriptor.

SERIES_HARMONIZATION_DESCRIPTOR_IMAGE_SIZE = 64
SERIES_HARMONIZATION_INTENSITY_HISTOGRAM_BINS = 16
SERIES_HARMONIZATION_DCT_SIDE = 4
# Each sampled frame contributes native layout/export statistics, robust
# intensity statistics, a normalized intensity histogram, low-frequency DCT
# coefficients, gradient summaries, and central/peripheral intensity summaries.

SERIES_HARMONIZATION_CLUSTER_COUNTS = (53, 71, 89)
SERIES_HARMONIZATION_KMEANS_N_INIT = 20
SERIES_HARMONIZATION_RANDOM_STATE = RANDOM_SEED + 7300
# Three nearby resolutions reduce dependence on one arbitrary cluster count.
# Values are fixed before model evaluation and are never selected by AUROC.

SERIES_HARMONIZATION_MIN_NORMAL_PATIENTS_PER_CLUSTER = 2
SERIES_HARMONIZATION_MIN_SICK_PATIENTS_PER_CLUSTER = 2
SERIES_HARMONIZATION_MIN_NORMAL_SERIES_PER_CLUSTER = 2
SERIES_HARMONIZATION_MIN_SICK_SERIES_PER_CLUSTER = 2
SERIES_HARMONIZATION_MIN_SHARED_RESOLUTIONS = 2
# A series is retained only when its assigned cluster is represented by at least
# two Normal patients and two Sick patients (and at least two series per class)
# in at least two of the three clustering resolutions.

SERIES_HARMONIZATION_CLUSTER_OUTLIER_MAD_MULTIPLIER = 5.0
SERIES_HARMONIZATION_CLUSTER_OUTLIER_MIN_QUANTILE = 0.99
# Within a shared cluster, an extreme descriptor-distance outlier can still be
# removed. The cutoff is the more permissive of median+5*MAD and the empirical
# 99th percentile; small clusters therefore are not over-pruned.

SERIES_HARMONIZATION_ALLOW_PATIENT_COVERAGE_RESCUE = True
SERIES_HARMONIZATION_MAX_COVERAGE_RESCUES = 5
# The validated patient unit remains Directory_*. If filtering would remove all
# series from one patient, the most cross-class-supported, least-outlying series
# for that patient is retained and marked as a coverage rescue. Every rescue is
# explicit in the audit. The pipeline stops if more than the predeclared maximum
# is required. The prior comparable run required zero rescues.

SERIES_HARMONIZATION_REQUIRE_ALL_30_PATIENTS = True
SERIES_HARMONIZATION_EXPECTED_PATIENTS = 30
SERIES_HARMONIZATION_EXPECTED_NORMAL_PATIENTS = 16
SERIES_HARMONIZATION_EXPECTED_SICK_PATIENTS = 14
# These counts are validated for this dataset release. They protect against a
# silent change in patient definition or accidental loss of a Directory_*.

FOCUSED_SUITE_PROFILE = "compact_reproducibility_and_validity_evidence"
MONAI_QC_MISSING_BRANCH_POLICY = "explicit_skip_without_imputation_v1"
# The focused V7 panel requests only standardized image representations. The
# original-canvas MONAI branch is therefore intentionally not executed during
# feature extraction. Missing QC from an unexecuted branch must be reported as
# SKIPPED_NOT_COMPUTED; it must never be converted to zeros or imputed medians,
# because those invented values would create a misleading QC classifier.

# This profile is designed for a concise research supplement or portfolio. It
# prioritizes falsifiable controls and robustness over a large model leaderboard.
# The executable registry contains only the sixteen essential experiments plus
# explicitly declared future additions. Earlier broad-run outputs remain the
# historical archive; redundant configurations are not carried into this file.

# Detailed comments are retained for every active representation, audit and
# evaluation contract. This reduces redundant model fitting and feature
# extraction while preserving every experiment needed to challenge a future
# modification.
#
# To test a new image representation:
#   1. add its ExperimentConfig to EXPERIMENT_REGISTRY;
#   2. add its ID to FUTURE_CANDIDATE_EXPERIMENT_IDS;
#   3. add any representation-specific negative controls to
#      FUTURE_CONTROL_EXPERIMENT_IDS;
#   4. map each future candidate to at least one active matched control in
#      FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.
# The suite will automatically execute future candidates, include them in
# repeated CV and add them to the nested candidate-selection audit.

V4_LOCKED_REFERENCE_EXPERIMENT_ID = (
    "A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA"
)
PRIMARY_CANDIDATE_EXPERIMENT_ID = (
    "A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA"
)
VALID_ONLY_CANDIDATE_EXPERIMENT_ID = (
    "A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA"
)
DEVELOPMENT_BASELINE_EXPERIMENT_ID = "A9_STANDARDIZED_FULL_HIER_LR_PCA"
BASELINE_EXPERIMENT_ID = PRIMARY_CANDIDATE_EXPERIMENT_ID

FUTURE_CANDIDATE_EXPERIMENT_IDS = ()
FUTURE_CONTROL_EXPERIMENT_IDS = ()
FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE = {}
# Every future candidate must have one dictionary entry whose value is a tuple
# of one or more active control IDs. A candidate that reuses A17's exact support
# may map to C31/C32/C33; a candidate with a new support or crop contract should
# define new controls. This requirement prevents a future AUC improvement from
# being evaluated without a representation-specific falsification test.

ESSENTIAL_EXPERIMENT_IDS = (
    # Main result and direct sensitivity analyses.
    PRIMARY_CANDIDATE_EXPERIMENT_ID,
    VALID_ONLY_CANDIDATE_EXPERIMENT_ID,
    "A21_HARD_SUPPORT_REGION_NORM_DEDUP_HIER_LR_PCA",
    "R4_HARD_SUPPORT_REGION_NORM_DROP50_HIER_LR_PCA",
    # Historical and localization references.
    V4_LOCKED_REFERENCE_EXPERIMENT_ID,
    "A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA",
    DEVELOPMENT_BASELINE_EXPERIMENT_ID,
    # Exact candidate-matched and extracardiac controls.
    "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
    "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
    "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
    "C28_OUTSIDE_WHOLE_HEART_REGION_NORM_HIER_LR_PCA",
    "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
    # Low-cost protocol/export confounder controls.
    "C3_EXPORT_PROVENANCE_ONLY_LR",
    "C17_N_SERIES_ONLY_LR",
    "C19_NATIVE_GEOMETRY_ONLY_LR",
    "C20_FILE_SIZE_ONLY_LR",
)

EXPERIMENTS_TO_RUN = ('A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA', 'A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA', 'A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA', 'A9_STANDARDIZED_FULL_HIER_LR_PCA', 'C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA', 'C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA', 'C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA', 'C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA', 'C17_N_SERIES_ONLY_LR', 'C19_NATIVE_GEOMETRY_ONLY_LR', 'C20_FILE_SIZE_ONLY_LR')

EXPERIMENT_REGISTRY = (
    # ======================================================================
    # PRIMARY CANDIDATE AND DIRECT SENSITIVITY ANALYSES
    # ======================================================================
    # These four experiments answer whether the principal A17 result survives
    # three concrete perturbations: rejecting MONAI-invalid rows, collapsing
    # exact repeated exports, and removing half of the slices inside every
    # series proxy. They intentionally keep the encoder, pooling hierarchy,
    # PCA, classifier family, hyperparameter search and outer folds unchanged.
    ExperimentConfig(
        experiment_id="A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA",
        description=(
            "Prospectively locked focused candidate: binary dilated cardiac "
            "support, fixed-center fallback, content-mask intersection and "
            "region-only robust scaling. MONAI soft confidence is not "
            "multiplied into MRI intensity."
        ),
        feature_mode="standardized_hard_support_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="primary_candidate",
    ),
    ExperimentConfig(
        experiment_id="A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA",
        description=(
            "A17 restricted to standardized MONAI gate-valid slices. This "
            "tests whether the main result depends on fixed-center fallback "
            "slices."
        ),
        feature_mode="standardized_hard_support_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        slice_filter="standardized_monai_valid",
        role="candidate_sensitivity",
    ),
    ExperimentConfig(
        experiment_id="A21_HARD_SUPPORT_REGION_NORM_DEDUP_HIER_LR_PCA",
        description=(
            "A17 after deterministic exact decoded-pixel deduplication inside "
            "each Directory_* patient. This verifies that repeated identical "
            "exports do not create the candidate's apparent performance."
        ),
        feature_mode="standardized_hard_support_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        deduplicate_exact_within_patient=True,
        role="candidate_sensitivity",
    ),
    ExperimentConfig(
        experiment_id="R4_HARD_SUPPORT_REGION_NORM_DROP50_HIER_LR_PCA",
        description=(
            "A17 after deterministic 50% within-series slice dropout, with at "
            "least one slice retained per series proxy. This is the single "
            "high-severity sampling robustness test kept by the focused suite."
        ),
        feature_mode="standardized_hard_support_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        slice_dropout_rate=0.50,
        role="robustness",
    ),

    # ======================================================================
    # HISTORICAL AND LOCALIZATION REFERENCES
    # ======================================================================
    # A12 is retained because it was locked before A17 was observed. A9 is the
    # fully standardized full-image reference, and A16 suppresses the visible
    # support silhouette by using a fixed-size heart-centered square crop. This
    # compact trio allows a future representation to be compared against:
    #   - the prior locked decision;
    #   - no cardiac masking at all;
    #   - localization without exposing exact mask shape.
    ExperimentConfig(
        experiment_id="A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA",
        description=(
            "Locked V4 reference: zero-background standardized MONAI ROI for "
            "valid masks, with a fixed 60% center crop rather than the full "
            "image when the gate fails."
        ),
        feature_mode="standardized_roi_zero_bg_center_fallback",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="historical_locked_reference",
    ),
    ExperimentConfig(
        experiment_id="A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA",
        description=(
            "Shape-suppressed localization reference: a fixed-size square field "
            "of view is centered on the valid MONAI hard-mask centroid, with an "
            "image-center fallback. Intensities are normalized only inside the "
            "retained non-padding crop, and the crop is resized without exposing "
            "the predicted support silhouette."
        ),
        feature_mode="standardized_heart_centered_fixed_fov_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="localization_reference",
    ),
    ExperimentConfig(
        experiment_id="A9_STANDARDIZED_FULL_HIER_LR_PCA",
        description=(
            "Label-blind standardized full-image reference with the same "
            "hierarchical pooling, PCA and Logistic Regression as A17."
        ),
        feature_mode="standardized_full_image",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="full_image_reference",
    ),

    # ======================================================================
    # EXACT A17-MATCHED REPRESENTATION CONTROLS
    # ======================================================================
    # These controls form the mechanistic core of the focused suite. C31 asks
    # whether exact support geometry alone predicts the label. C32 preserves
    # that support and every visible normalized intensity but destroys their
    # original spatial arrangement. C33 uses the exact retained-content
    # complement and independent regional scaling. All three share A17's source
    # rows, patient grouping, pooling and downstream classifier path.
    ExperimentConfig(
        experiment_id="C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
        description=(
            "Exact binary support used by A17, including dilation, content-mask "
            "intersection and fixed-center fallback, but with all MRI intensity "
            "removed. It isolates support geometry."
        ),
        feature_mode="standardized_a17_exact_support_mask_only",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="segmentation_representation_control",
    ),
    ExperimentConfig(
        experiment_id="C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
        description=(
            "A17 with its exact support and exact visible intensity multiset "
            "preserved, while a deterministic decoded-pixel-hash permutation "
            "destroys the original spatial arrangement. It isolates spatial "
            "anatomy and texture beyond support and histogram."
        ),
        feature_mode="standardized_a17_support_intensity_affine_shuffled",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="segmentation_representation_control",
    ),
    ExperimentConfig(
        experiment_id="C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
        description=(
            "Exact retained-content complement of A17's support, with "
            "independent region-only robust scaling. It uses the same slices, "
            "pooling, folds and classifier path; only the visible spatial region "
            "changes."
        ),
        feature_mode="standardized_a17_exact_support_complement_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),

    # ======================================================================
    # BROADER EXTRACARDIAC CONTROLS
    # ======================================================================
    # C28 deliberately removes a larger, conservative central cardiac proxy;
    # C29 uses no MONAI geometry at all. Together with C33 they distinguish
    # residual cardiac content from scanner, protocol, positioning or export
    # signal in the image periphery.
    ExperimentConfig(
        experiment_id="C28_OUTSIDE_WHOLE_HEART_REGION_NORM_HIER_LR_PCA",
        description=(
            "Conservative extracardiac control. It removes the union of a large "
            "fixed central square, a mask-centered square and a substantially "
            "expanded ventricular bounding box, then normalizes only the pixels "
            "that remain. This is a proxy, not ground-truth whole-heart anatomy."
        ),
        feature_mode="standardized_outside_whole_heart_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
        description=(
            "MONAI-independent peripheral control retaining only pixels outside "
            "a fixed central square and computing its robust scaling only in "
            "that periphery. It tests scanner and export signal independently "
            "of mask shape."
        ),
        feature_mode="standardized_fixed_periphery_region_norm",
        strategy="patient_embedding",
        pooling_strategy="hierarchical",
        weighting_mode="equal",
        classifier_type="logistic_regression",
        use_pca=True,
        tune_c=True,
        role="negative_control",
    ),

    # ======================================================================
    # LOW-COST PROTOCOL AND EXPORT CONFOUNDER CONTROLS
    # ======================================================================
    # These one-row-per-patient controls do not require extra neural inference.
    # They remain essential because a high candidate AUROC is not persuasive if
    # class labels can also be recovered from image geometry, file compression,
    # export provenance or the number of folder-defined series proxies.
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
        experiment_id="C17_N_SERIES_ONLY_LR",
        description=(
            "Negative control using only the number of folder-defined series "
            "proxies per patient."
        ),
        feature_mode="n_series_only",
        strategy="patient_tabular",
        pooling_strategy="patient_tabular",
        weighting_mode="not_applicable",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C19_NATIVE_GEOMETRY_ONLY_LR",
        description=(
            "Negative control using only patient-aggregated native height, "
            "width and aspect-ratio features; image counts and file size are "
            "excluded."
        ),
        feature_mode="native_geometry_only",
        strategy="patient_tabular",
        pooling_strategy="patient_tabular",
        weighting_mode="not_applicable",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
        role="negative_control",
    ),
    ExperimentConfig(
        experiment_id="C20_FILE_SIZE_ONLY_LR",
        description=(
            "Negative control using only patient-aggregated file-size and "
            "bytes-per-native-pixel statistics; geometry and image counts are "
            "excluded."
        ),
        feature_mode="file_size_only",
        strategy="patient_tabular",
        pooling_strategy="patient_tabular",
        weighting_mode="not_applicable",
        classifier_type="logistic_regression",
        use_pca=False,
        tune_c=True,
        role="negative_control",
    ),
)

# The broad comparison catalog from earlier versions is intentionally archived.
# Every active V7 comparison is declared below with an explicit reference,
# changed configuration and research question, so Delta-AUROC is always
# calculated as ``changed - reference``. Historical legacy slice-classifier and
# classifier-family comparisons remain available in prior run artifacts but are
# not rerun by this compact suite.
ARCHIVED_FULL_ABLATION_COMPARISONS = ()
# Earlier development versions also tested legacy slice-level classifiers,
# Linear SVM, PCA-off variants, multiple center-crop sizes, soft-mask
# decomposition, 10%/25% dropout and several overlapping border/corner controls.
# Their numerical results remain in the earlier run packages and are not erased
# from the research history. They are deliberately not executable here because:
#
#   1. they no longer change the central claim after A17 was prospectively
#      confirmed against exact support, shuffled-intensity and complement
#      controls;
#   2. several answer nearly identical questions and inflate a model leaderboard
#      without providing a new falsification route;
#   3. legacy slice-classifier branches are computationally expensive while the
#      effective labeled sample remains 30 patients;
#   4. a concise portfolio artifact is stronger when every retained experiment
#      has a clear scientific purpose and a predeclared interpretation;
#   5. future candidates can be registered explicitly through
#      FUTURE_CANDIDATE_EXPERIMENT_IDS and receive the same patient-level CV,
#      repeated-split, permutation and confounder framework.
#
# The detailed preprocessing, duplicate auditing, fold construction, MONAI QC,
# nested fitting, bootstrap intervals, sequence/view adapter and external-
# validation boundary comments remain in this source. Only redundant experiment
# definitions and redundant comparison tuples have been removed.


# ---------------------------------------------------------------------------
# ACTIVE FOCUSED COMPARISONS
# ---------------------------------------------------------------------------
# Earlier comparison catalogs remain in prior run packages. The executable list
# below contains only questions that materially affect the validity of the
# current candidate or a future challenger. Delta-AUROC is always
# ``changed - reference``.
FOCUSED_BASE_COMPARISONS = (
    (
        "A12_REFERENCE_VS_A17_PRIMARY",
        V4_LOCKED_REFERENCE_EXPERIMENT_ID,
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does the prospectively locked hard-support candidate improve over the locked V4 reference?",
    ),
    (
        "A9_FULL_IMAGE_VS_A17_PRIMARY",
        DEVELOPMENT_BASELINE_EXPERIMENT_ID,
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does cardiac localization outperform the standardized full-image representation?",
    ),
    (
        "A16_SHAPE_SUPPRESSED_CROP_VS_A17_SUPPORT",
        "A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does exposing the exact support improve over a fixed heart-centered crop that suppresses the support silhouette?",
    ),
    (
        "A17_ALL_SLICES_VS_A20_VALID_ONLY",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        VALID_ONLY_CANDIDATE_EXPERIMENT_ID,
        "Does removing gate-invalid fallback slices materially change the hard-support candidate?",
    ),
    (
        "A17_VS_A21_EXACT_DEDUP",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "A21_HARD_SUPPORT_REGION_NORM_DEDUP_HIER_LR_PCA",
        "Does exact within-patient decoded-pixel deduplication change the main candidate?",
    ),
    (
        "A17_VS_R4_DROP50",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "R4_HARD_SUPPORT_REGION_NORM_DROP50_HIER_LR_PCA",
        "Does the candidate retain discrimination after deterministic removal of half the slices in every series proxy?",
    ),
    (
        "C31_EXACT_SUPPORT_MASK_ONLY_VS_A17",
        "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "How much does MRI intensity add beyond A17's exact support geometry?",
    ),
    (
        "C32_SHUFFLED_INTENSITY_VS_A17",
        "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "How much does intact spatial anatomy and texture add beyond the same support and intensity multiset?",
    ),
    (
        "C33_EXACT_COMPLEMENT_VS_A17",
        "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "How much more predictive is A17's exact support than its independently normalized complement?",
    ),
    (
        "C28_CONSERVATIVE_OUTSIDE_VS_A17",
        "C28_OUTSIDE_WHOLE_HEART_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does the candidate outperform a conservative outside-whole-heart proxy?",
    ),
    (
        "C29_FIXED_PERIPHERY_VS_A17",
        "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does the candidate outperform a MONAI-independent fixed periphery?",
    ),
    (
        "C3_EXPORT_PROVENANCE_VS_A17",
        "C3_EXPORT_PROVENANCE_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does A17 outperform a classifier using only export/provenance metadata?",
    ),
    (
        "C17_SERIES_COUNT_VS_A17",
        "C17_N_SERIES_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does A17 outperform the number of exported series proxies alone?",
    ),
    (
        "C19_NATIVE_GEOMETRY_VS_A17",
        "C19_NATIVE_GEOMETRY_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does A17 outperform native image geometry alone?",
    ),
    (
        "C20_FILE_SIZE_VS_A17",
        "C20_FILE_SIZE_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Does A17 outperform file-size and compression proxies alone?",
    ),
)

# Every future candidate receives two generic paired comparisons automatically.
# Representation-specific controls still require an explicit comparison because
# only the researcher can define what is truly matched for a new spatial contract.
FUTURE_CANDIDATE_COMPARISONS = tuple(
    comparison
    for experiment_id in FUTURE_CANDIDATE_EXPERIMENT_IDS
    for comparison in (
        (
            f"A17_PRIMARY_VS_FUTURE__{experiment_id}",
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            experiment_id,
            "Does the future candidate improve over the locked focused-suite candidate?",
        ),
        (
            f"A12_REFERENCE_VS_FUTURE__{experiment_id}",
            V4_LOCKED_REFERENCE_EXPERIMENT_ID,
            experiment_id,
            "Does the future candidate improve over the independent historical reference?",
        ),
        (
            f"A9_FULL_IMAGE_VS_FUTURE__{experiment_id}",
            DEVELOPMENT_BASELINE_EXPERIMENT_ID,
            experiment_id,
            "Does the future candidate improve over the standardized full-image reference?",
        ),
    )
)

FUTURE_MATCHED_CONTROL_COMPARISONS = tuple(
    (
        f"FUTURE_CONTROL__{control_id}__VS__{candidate_id}",
        control_id,
        candidate_id,
        "Does the future candidate outperform its predeclared matched control?",
    )
    for candidate_id in FUTURE_CANDIDATE_EXPERIMENT_IDS
    for control_id in FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.get(
        candidate_id, ()
    )
)

PRIMARY_ABLATION_COMPARISONS = (('A12_REFERENCE_VS_A17_PRIMARY', 'A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the prospectively locked hard-support candidate remain stronger than the independently established A12 reference after strict series balancing?'), ('A9_FULL_IMAGE_VS_A17_PRIMARY', 'A9_STANDARDIZED_FULL_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does cardiac localization retain value over the standardized full image after strict series balancing?'), ('A16_SHAPE_SUPPRESSED_CROP_VS_A17_SUPPORT', 'A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does A17 outperform a heart-centred crop that does not expose the exact support silhouette?'), ('A17_ALL_SLICES_VS_A20_VALID_ONLY', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA', 'Does the candidate depend on slices receiving the fixed-centre fallback after an invalid standardized MONAI gate?'), ('C31_EXACT_SUPPORT_MASK_ONLY_VS_A17', 'C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', "How much does MRI intensity add beyond A17's exact support geometry?"), ('C32_SHUFFLED_INTENSITY_VS_A17', 'C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'How much does intact spatial texture add beyond the same support and visible intensity multiset?'), ('C33_EXACT_COMPLEMENT_VS_A17', 'C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the exact A17 support remain more predictive than its independently normalized retained-content complement?'), ('C29_FIXED_PERIPHERY_VS_A17', 'C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the cardiac candidate remain stronger than a MONAI-independent fixed peripheral image control?'), ('C17_SERIES_COUNT_VS_A17', 'C17_N_SERIES_ONLY_LR', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does exact per-patient series balancing remove the former series-count shortcut while preserving the cardiac candidate?'), ('C19_NATIVE_GEOMETRY_VS_A17', 'C19_NATIVE_GEOMETRY_ONLY_LR', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the candidate remain stronger than native image geometry after stricter cross-class overlap selection?'), ('C20_FILE_SIZE_VS_A17', 'C20_FILE_SIZE_ONLY_LR', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the candidate remain stronger than file-size and compression proxies after stricter cross-class overlap selection?'))

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

USE_CUDA_AMP = False
DATALOADER_NUM_WORKERS = 0
ENABLE_DETAILED_PROGRESS_PRINTS = True
PROGRESS_PRINT_EVERY_N_BATCHES = 25

USE_FEATURE_CACHE = True
FORCE_REBUILD_FEATURE_CACHE = False
FEATURE_CACHE_SCHEMA_VERSION = "2026-09-09-focused-research-v7-v1"
EFFICIENTNET_FEATURE_DIM = 1280
FEATURE_MODES_PER_ENCODER_CALL = 4
# Several image variants can be concatenated along the batch dimension and
# encoded by one EfficientNet call. Four modes at a time is a conservative T4
# default: it reduces Python/kernel-launch overhead without materializing all
# nine focused image views simultaneously. Lower this value if GPU memory is
# insufficient.

SLICE_QUALITY_MIN_WEIGHT = 0.25
# This generic helper constant is retained for compatibility with the reusable
# preparation functions, but no quality-weighted experiment is active in V7.

BORDER_WIDTH_FRACTION = 0.15
# C1 retains only this outer fraction on each side of the original 224x224
# image so the first suite result remains directly reproducible.

STANDARDIZED_BORDER_WIDTH_FRACTIONS = (0.05, 0.10)
STANDARDIZED_CORNER_WIDTH_FRACTION = 0.15
CENTER_CROP_FALLBACK_FRACTION = 0.60
FIXED_CENTER_CROP_FRACTIONS = (0.50, 0.60, 0.70)
STANDARDIZED_CONTENT_LONG_SIDE = 240
FIXED_CHUNK_SIZE = 20
MONAI_BBOX_CONTEXT_FRACTION = 0.15
OUTSIDE_MONAI_BBOX_CONTEXT_FRACTION = 0.30

# ---------------------------------------------------------------------------
# FOCUSED V7 CARDIAC / EXTRACARDIAC REPRESENTATIONS
# ---------------------------------------------------------------------------
FOCUSED_FIXED_HEART_FOV_FRACTION = 0.65
FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL = 15
FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION = 0.75
FOCUSED_WHOLE_HEART_EXCLUSION_BBOX_CONTEXT_FRACTION = 0.65
FOCUSED_FIXED_PERIPHERY_EXCLUSION_FRACTION = 0.75

FOCUSED_REGION_NORM_LOWER_PERCENTILE = 1.0
FOCUSED_REGION_NORM_UPPER_PERCENTILE = 99.0
FOCUSED_REGION_NORM_MIN_PIXELS = 64
FOCUSED_REGION_NORM_HISTOGRAM_BINS = 256
FOCUSED_REGION_NORM_MIN_DYNAMIC_RANGE = 8.0 / 255.0
# Intensities are estimated only from pixels that remain visible in the final
# representation. Consequently, excluded cardiac pixels cannot define the
# contrast of an extracardiac control and vice versa. A fixed minimum dynamic
# range prevents nearly uniform JPEG backgrounds from being amplified to the
# complete [0,1] interval.

FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION = (
    "sha256-affine-visible-index-v1"
)
# C32 uses a hash-derived affine permutation that is bijective over the visible
# support. It preserves the exact per-image intensity multiset and support but
# destroys the original spatial/anatomical arrangement.
# The standardized branch now resizes the longest retained content dimension
# to exactly 240 pixels, including upsampling when needed, and centers it in a
# 256x256 canvas. This removes the previous dependence of pipeline-added
# padding on native pixel dimensions and on how much dark border was removed.
# Aspect ratio is still preserved and therefore remains explicitly audited.
# Fixed 50%, 60%, and 70% center crops are independent of MONAI geometry.
# FIXED_CHUNK_SIZE supports a folder-independent pooling ablation. All values
# are fixed before evaluation and are independent of labels and OOF performance.

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
    "fixed_content_height_fraction_of_canvas",
    "fixed_content_width_fraction_of_canvas",
    "fixed_pipeline_padding_fraction",
)
# Label-blind standardization removes only consecutive edge rows/columns that
# are nearly uniform and dark. Safety limits prevent aggressive cropping. The
# retained image is converted into two intensity views on the same fixed
# geometry: min-max scaling for MONAI and robust 1st/99th-percentile scaling for
# EfficientNet. The longest retained side is always resized to 240 pixels and
# centered in a 256x256 canvas, so pipeline padding no longer depends on native
# resolution or on the amount of detected border removal.

MONAI_SOFT_HISTOGRAM_BINS = 64
# The distribution-only control converts each gate-valid soft MONAI map into a
# fixed 64-bin cumulative-distribution image. It retains the empirical
# probability distribution but removes every original pixel coordinate and all
# connected mask morphology. The value is fixed before evaluation.

MONAI_SOFT_BLOCK_SHUFFLE_GRID = 14
MONAI_SOFT_BLOCK_SHUFFLE_VERSION = "sha256-per-image-14x14-v1"
# The 224x224 map is divided into a 14x14 grid of 16x16 blocks. A deterministic
# per-image permutation is derived from the exact decoded-pixel SHA-256, never
# from label, patient ID, series ID, fold, or model score. Within-block soft-map
# texture and the complete probability histogram are preserved, while global
# shape and location are destroyed. This remains a diagnostic control rather
# than a natural-image preprocessing recommendation.

MONAI_CANONICAL_MASK_CONTENT_FRACTION = 0.75
# Canonicalized mask controls crop to the gate-valid hard-mask bounding box,
# preserve relative morphology, square-pad without anisotropic distortion, and
# place the result at one fixed centered scale occupying 75% of the 224x224
# canvas. Absolute mask location and original extent are therefore removed.

RUN_ANNOTATED_SERIES_SUBSET_ANALYSIS = False
SERIES_ANNOTATION_INPUT_PATH = None
ANNOTATED_SERIES_SELECTION_NAME = "heart_nonlocalizer_nonderived"
ANNOTATED_SERIES_REQUIRE_CONTAINS_HEART = True
ANNOTATED_SERIES_EXCLUDE_LOCALIZERS = True
ANNOTATED_SERIES_EXCLUDE_DERIVED_EXPORTS = True
ANNOTATED_SERIES_ALLOWED_SEQUENCE_TYPES = ()
ANNOTATED_SERIES_ALLOWED_VIEW_TYPES = ()
ANNOTATED_SERIES_ALLOWED_CONFIDENCE_VALUES = ("high", "medium")
ANNOTATED_SERIES_USE_SEQUENCE_VIEW_BALANCED_POOLING = True
# After blinded annotations are completed, the focused sensitivity analysis
# aggregates slices -> series proxy -> explicit (sequence_type, view_type) cell
# -> patient, assigning equal weight to each retained cell. This prevents a
# protocol with many exported folders from dominating solely through frequency.

_FUTURE_IMAGE_EXPERIMENT_IDS = tuple(
    experiment_id
    for experiment_id in (
        FUTURE_CANDIDATE_EXPERIMENT_IDS + FUTURE_CONTROL_EXPERIMENT_IDS
    )
    if any(
        experiment.experiment_id == experiment_id
        and experiment.strategy == "patient_embedding"
        for experiment in EXPERIMENT_REGISTRY
    )
)

ANNOTATED_SERIES_EXPERIMENT_IDS = ('A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA', 'A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA', 'A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA', 'A9_STANDARDIZED_FULL_HIER_LR_PCA', 'C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA', 'C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA', 'C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA', 'C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA')
# The pipeline never infers sequence or view from SR_*/series* folder names.
# When this optional stage is enabled, the user must point to a completed copy
# of the blinded series_annotation_template.csv stored outside the current run
# directory. Only explicitly annotated series proxies are retained, and a fresh
# patient-level fold manifest is created for the retained cohort. The default is
# disabled because empty annotation columns cannot support valid filtering.

C_SELECTION_AUC_TOLERANCE = 0.01
# Select the smallest (most regularized) C whose inner AUC is within this
# absolute tolerance of the best candidate. This prevents tiny inner-CV
# differences from repeatedly choosing the least regularized edge of the grid.

RUN_REPEATED_NESTED_CV_STABILITY = True
REPEATED_NESTED_CV_REPEATS = 50
REPEATED_NESTED_CV_RANDOM_STATE = RANDOM_SEED + 20_000
STABILITY_EXPERIMENT_IDS = ('A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA', 'A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA', 'A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA', 'A9_STANDARDIZED_FULL_HIER_LR_PCA', 'C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA', 'C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA', 'C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA', 'C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA', 'C17_N_SERIES_ONLY_LR', 'C19_NATIVE_GEOMETRY_ONLY_LR', 'C20_FILE_SIZE_ONLY_LR')

# Every active repeated comparison uses the exact same outer split seeds. The
# first model is the reference and the second is the changed configuration, so
# a positive delta means the SECOND named configuration performed better on that
# repeat. The archived broad comparison panel is not rerun in V7.
ARCHIVED_REPEATED_STABILITY_COMPARISONS = ()
# The earlier 19- and 30-model stability panels remain documented in their
# original run outputs. Repeating them in this focused artifact would add many
# thousands of small nested fits without changing the questions needed to
# evaluate A17 or a future candidate. The active panel below retains one
# comparison for each independent validity threat: prior locked reference,
# full-image signal, support-silhouette dependence, invalid-mask fallback,
# exact duplicate exports, severe slice subsampling, support geometry,
# intensity distribution, exact complement, conservative periphery, export
# provenance, series count, native geometry and compression/file size.

FOCUSED_REPEATED_STABILITY_COMPARISONS = (
    (
        "A12_REFERENCE_VS_A17_PRIMARY",
        V4_LOCKED_REFERENCE_EXPERIMENT_ID,
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Historical reference versus the prospectively locked focused candidate.",
    ),
    (
        "A9_FULL_IMAGE_VS_A17_PRIMARY",
        DEVELOPMENT_BASELINE_EXPERIMENT_ID,
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Standardized full image versus cardiac hard support.",
    ),
    (
        "A16_FIXED_FOV_VS_A17_SUPPORT",
        "A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Support-silhouette-suppressed crop versus exact hard support.",
    ),
    (
        "A17_ALL_VS_A20_VALID_ONLY",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        VALID_ONLY_CANDIDATE_EXPERIMENT_ID,
        "All slices versus gate-valid slices only.",
    ),
    (
        "A17_VS_A21_EXACT_DEDUP",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "A21_HARD_SUPPORT_REGION_NORM_DEDUP_HIER_LR_PCA",
        "All exact exports versus one canonical copy within each patient.",
    ),
    (
        "A17_VS_R4_DROP50",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "R4_HARD_SUPPORT_REGION_NORM_DROP50_HIER_LR_PCA",
        "All slices versus deterministic 50 percent within-series dropout.",
    ),
    (
        "C31_EXACT_SUPPORT_ONLY_VS_A17",
        "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Exact support geometry only versus support plus MRI intensity.",
    ),
    (
        "C32_SHUFFLED_INTENSITY_VS_A17",
        "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Support and intensity multiset without intact spatial anatomy versus A17.",
    ),
    (
        "C33_EXACT_COMPLEMENT_VS_A17",
        "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Exact independently normalized complement versus A17 support.",
    ),
    (
        "C28_CONSERVATIVE_OUTSIDE_VS_A17",
        "C28_OUTSIDE_WHOLE_HEART_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Conservative extracardiac proxy versus A17.",
    ),
    (
        "C29_FIXED_PERIPHERY_VS_A17",
        "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "MONAI-independent periphery versus A17.",
    ),
    (
        "C3_EXPORT_PROVENANCE_VS_A17",
        "C3_EXPORT_PROVENANCE_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Export/provenance metadata only versus A17.",
    ),
    (
        "C17_SERIES_COUNT_VS_A17",
        "C17_N_SERIES_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Number of exported series proxies only versus A17.",
    ),
    (
        "C19_NATIVE_GEOMETRY_VS_A17",
        "C19_NATIVE_GEOMETRY_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "Native image geometry only versus A17.",
    ),
    (
        "C20_FILE_SIZE_VS_A17",
        "C20_FILE_SIZE_ONLY_LR",
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "File-size/compression proxies only versus A17.",
    ),
)

FUTURE_REPEATED_STABILITY_COMPARISONS = tuple(
    comparison
    for experiment_id in FUTURE_CANDIDATE_EXPERIMENT_IDS
    for comparison in (
        (
            f"A17_PRIMARY_VS_FUTURE__{experiment_id}",
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            experiment_id,
            "Future candidate versus the locked focused candidate.",
        ),
        (
            f"A12_REFERENCE_VS_FUTURE__{experiment_id}",
            V4_LOCKED_REFERENCE_EXPERIMENT_ID,
            experiment_id,
            "Future candidate versus the independent historical reference.",
        ),
        (
            f"A9_FULL_IMAGE_VS_FUTURE__{experiment_id}",
            DEVELOPMENT_BASELINE_EXPERIMENT_ID,
            experiment_id,
            "Future candidate versus the standardized full-image reference.",
        ),
    )
)

FUTURE_MATCHED_CONTROL_REPEATED_COMPARISONS = tuple(
    (
        f"FUTURE_CONTROL__{control_id}__VS__{candidate_id}",
        control_id,
        candidate_id,
        "Future candidate versus its predeclared matched control.",
    )
    for candidate_id in FUTURE_CANDIDATE_EXPERIMENT_IDS
    for control_id in FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.get(
        candidate_id, ()
    )
    if control_id in STABILITY_EXPERIMENT_IDS
)

REPEATED_STABILITY_COMPARISONS = (('A12_REFERENCE_VS_A17_PRIMARY', 'A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the prospectively locked hard-support candidate remain stronger than the independently established A12 reference after strict series balancing?'), ('A9_FULL_IMAGE_VS_A17_PRIMARY', 'A9_STANDARDIZED_FULL_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does cardiac localization retain value over the standardized full image after strict series balancing?'), ('A16_SHAPE_SUPPRESSED_CROP_VS_A17_SUPPORT', 'A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does A17 outperform a heart-centred crop that does not expose the exact support silhouette?'), ('A17_ALL_SLICES_VS_A20_VALID_ONLY', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA', 'Does the candidate depend on slices receiving the fixed-centre fallback after an invalid standardized MONAI gate?'), ('C31_EXACT_SUPPORT_MASK_ONLY_VS_A17', 'C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', "How much does MRI intensity add beyond A17's exact support geometry?"), ('C32_SHUFFLED_INTENSITY_VS_A17', 'C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'How much does intact spatial texture add beyond the same support and visible intensity multiset?'), ('C33_EXACT_COMPLEMENT_VS_A17', 'C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the exact A17 support remain more predictive than its independently normalized retained-content complement?'), ('C29_FIXED_PERIPHERY_VS_A17', 'C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the cardiac candidate remain stronger than a MONAI-independent fixed peripheral image control?'), ('C17_SERIES_COUNT_VS_A17', 'C17_N_SERIES_ONLY_LR', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does exact per-patient series balancing remove the former series-count shortcut while preserving the cardiac candidate?'), ('C19_NATIVE_GEOMETRY_VS_A17', 'C19_NATIVE_GEOMETRY_ONLY_LR', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the candidate remain stronger than native image geometry after stricter cross-class overlap selection?'), ('C20_FILE_SIZE_VS_A17', 'C20_FILE_SIZE_ONLY_LR', 'A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'Does the candidate remain stronger than file-size and compression proxies after stricter cross-class overlap selection?'))


RUN_PATIENT_LABEL_PERMUTATION_TEST = True
LABEL_PERMUTATION_REPLICATES = 1000
LABEL_PERMUTATION_RANDOM_STATE = RANDOM_SEED + 40_000
PERMUTATION_EXPERIMENT_IDS = ('A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA',)
# The ordinary null repeats the complete patient-level nested fitting path for
# the prospectively locked A17 representation.

RUN_NESTED_MODEL_SELECTION_AUDIT = True
RUN_REPEATED_MODEL_SELECTION_STABILITY = True
MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS = ('A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA', 'A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA', 'A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA')
MODEL_SELECTION_AUC_TOLERANCE = 0.01
# Representation identity and classifier C are selected only inside each outer
# training cohort. Candidates within 0.01 AUC of the best inner result follow
# the fixed priority order above, preventing tiny inner-CV fluctuations from
# repeatedly changing the chosen representation.

RUN_SELECTION_ADJUSTED_PERMUTATION_TEST = True
SELECTION_ADJUSTED_PERMUTATION_REPLICATES = 500
SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE = RANDOM_SEED + 60_000
# This second null repeats candidate identity and C selection after every
# patient-label permutation. It corrects for the compact declared candidate
# family, not for every historical exploratory idea retained in the archive.

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
FOCUSED_REPEATED_SUPPORT_FRACTION = 0.80
FOCUSED_ROBUSTNESS_AUC_TOLERANCE = 0.02
# Evidence labels in the compact research scorecard use only these predeclared
# descriptive rules. They are not clinical non-inferiority margins or formal
# hypothesis tests. A claim is marked stable across split seeds only when the
# median paired delta has the expected sign and at least 80% of repeated splits
# agree. Perturbation robustness is marked only when the absolute median AUC
# change is no more than 0.02.

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
    "focused_suite_version": FOCUSED_SUITE_VERSION,
    "focused_suite_profile": FOCUSED_SUITE_PROFILE,
    "monai_qc_missing_branch_policy": MONAI_QC_MISSING_BRANCH_POLICY,
    "experiments": _selected_experiments_for_identity,
    "essential_experiment_ids": ESSENTIAL_EXPERIMENT_IDS,
    "future_candidate_experiment_ids": FUTURE_CANDIDATE_EXPERIMENT_IDS,
    "future_control_experiment_ids": FUTURE_CONTROL_EXPERIMENT_IDS,
    "future_matched_control_ids_by_candidate": (
        FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE
    ),
    "primary_candidate_experiment_id": PRIMARY_CANDIDATE_EXPERIMENT_ID,
    "valid_only_candidate_experiment_id": VALID_ONLY_CANDIDATE_EXPERIMENT_ID,
    "v4_locked_reference_experiment_id": (
        V4_LOCKED_REFERENCE_EXPERIMENT_ID
    ),
    "development_baseline_experiment_id": DEVELOPMENT_BASELINE_EXPERIMENT_ID,
    "primary_ablation_comparisons": PRIMARY_ABLATION_COMPARISONS,
    "repeated_stability_comparisons": REPEATED_STABILITY_COMPARISONS,
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
    "deterministic_execution": {
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32": False,
    },
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
    "fixed_center_crop_fractions": FIXED_CENTER_CROP_FRACTIONS,
    "standardized_content_long_side": STANDARDIZED_CONTENT_LONG_SIDE,
    "fixed_chunk_size": FIXED_CHUNK_SIZE,
    "focused_fixed_heart_fov_fraction": FOCUSED_FIXED_HEART_FOV_FRACTION,
    "focused_hard_support_extra_dilation_kernel": (
        FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL
    ),
    "focused_whole_heart_exclusion_center_fraction": (
        FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION
    ),
    "focused_whole_heart_exclusion_bbox_context_fraction": (
        FOCUSED_WHOLE_HEART_EXCLUSION_BBOX_CONTEXT_FRACTION
    ),
    "focused_fixed_periphery_exclusion_fraction": (
        FOCUSED_FIXED_PERIPHERY_EXCLUSION_FRACTION
    ),
    "focused_region_norm_lower_percentile": (
        FOCUSED_REGION_NORM_LOWER_PERCENTILE
    ),
    "focused_region_norm_upper_percentile": (
        FOCUSED_REGION_NORM_UPPER_PERCENTILE
    ),
    "focused_region_norm_min_pixels": FOCUSED_REGION_NORM_MIN_PIXELS,
    "focused_region_norm_histogram_bins": (
        FOCUSED_REGION_NORM_HISTOGRAM_BINS
    ),
    "focused_region_norm_min_dynamic_range": (
        FOCUSED_REGION_NORM_MIN_DYNAMIC_RANGE
    ),
    "focused_support_intensity_permutation_version": (
        FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION
    ),
    "monai_soft_histogram_bins": MONAI_SOFT_HISTOGRAM_BINS,
    "monai_soft_block_shuffle_grid": MONAI_SOFT_BLOCK_SHUFFLE_GRID,
    "monai_soft_block_shuffle_version": MONAI_SOFT_BLOCK_SHUFFLE_VERSION,
    "monai_canonical_mask_content_fraction": (
        MONAI_CANONICAL_MASK_CONTENT_FRACTION
    ),
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
    "stability_experiment_ids": STABILITY_EXPERIMENT_IDS,
    "run_patient_label_permutation_test": (
        RUN_PATIENT_LABEL_PERMUTATION_TEST
    ),
    "label_permutation_replicates": LABEL_PERMUTATION_REPLICATES,
    "label_permutation_random_state": LABEL_PERMUTATION_RANDOM_STATE,
    "permutation_experiment_ids": PERMUTATION_EXPERIMENT_IDS,
    "run_nested_model_selection_audit": RUN_NESTED_MODEL_SELECTION_AUDIT,
    "run_repeated_model_selection_stability": (
        RUN_REPEATED_MODEL_SELECTION_STABILITY
    ),
    "model_selection_candidate_experiment_ids": (
        MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
    ),
    "model_selection_auc_tolerance": MODEL_SELECTION_AUC_TOLERANCE,
    "run_selection_adjusted_permutation_test": (
        RUN_SELECTION_ADJUSTED_PERMUTATION_TEST
    ),
    "selection_adjusted_permutation_replicates": (
        SELECTION_ADJUSTED_PERMUTATION_REPLICATES
    ),
    "selection_adjusted_permutation_random_state": (
        SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE
    ),
    "focused_repeated_support_fraction": (
        FOCUSED_REPEATED_SUPPORT_FRACTION
    ),
    "focused_robustness_auc_tolerance": (
        FOCUSED_ROBUSTNESS_AUC_TOLERANCE
    ),
    "run_annotated_series_subset_analysis": (
        RUN_ANNOTATED_SERIES_SUBSET_ANALYSIS
    ),
    "series_annotation_input_path": SERIES_ANNOTATION_INPUT_PATH,
    "annotated_series_selection_name": ANNOTATED_SERIES_SELECTION_NAME,
    "annotated_series_require_contains_heart": (
        ANNOTATED_SERIES_REQUIRE_CONTAINS_HEART
    ),
    "annotated_series_exclude_localizers": (
        ANNOTATED_SERIES_EXCLUDE_LOCALIZERS
    ),
    "annotated_series_exclude_derived_exports": (
        ANNOTATED_SERIES_EXCLUDE_DERIVED_EXPORTS
    ),
    "annotated_series_allowed_sequence_types": (
        ANNOTATED_SERIES_ALLOWED_SEQUENCE_TYPES
    ),
    "annotated_series_allowed_view_types": ANNOTATED_SERIES_ALLOWED_VIEW_TYPES,
    "annotated_series_allowed_confidence_values": (
        ANNOTATED_SERIES_ALLOWED_CONFIDENCE_VALUES
    ),
    "annotated_series_use_sequence_view_balanced_pooling": (
        ANNOTATED_SERIES_USE_SEQUENCE_VIEW_BALANCED_POOLING
    ),
    "annotated_series_experiment_ids": ANNOTATED_SERIES_EXPERIMENT_IDS,
}

SUITE_CONFIGURATION_TAG = hashlib.sha256(
    json.dumps(_suite_identity, sort_keys=True).encode("utf-8")
).hexdigest()[:10]
SUITE_NAME = f"focused_research_suite__{SUITE_CONFIGURATION_TAG}"

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

PIPELINE_STAGE_COUNT = 14
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
    if MONAI_QC_MISSING_BRANCH_POLICY != (
        "explicit_skip_without_imputation_v1"
    ):
        raise ValueError(
            "Unsupported MONAI_QC_MISSING_BRANCH_POLICY. Missing branches "
            "must be skipped explicitly and must not be imputed."
        )

    experiment_ids = [experiment.experiment_id for experiment in experiments]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Experiment IDs must be unique.")

    # Validate the predeclared scientific comparison registry against the full
    # experiment catalog. A reduced preliminary run may omit one side of a pair;
    # that pair will be saved as SKIPPED rather than being silently redefined.
    known_experiment_ids = {
        experiment.experiment_id for experiment in EXPERIMENT_REGISTRY
    }

    # The focused artifact must not silently drift back into a broad model
    # leaderboard. Every registry entry is either part of the essential panel or
    # an explicitly declared future candidate/control. Adding an experiment to
    # only one location is treated as a configuration error before any image is
    # decoded.
    declared_active_ids = set(ESSENTIAL_EXPERIMENT_IDS).union(
        FUTURE_CANDIDATE_EXPERIMENT_IDS,
        FUTURE_CONTROL_EXPERIMENT_IDS,
    )
    undeclared_registry_ids = sorted(
        known_experiment_ids - declared_active_ids
    )
    missing_registry_ids = sorted(
        declared_active_ids - known_experiment_ids
    )
    if undeclared_registry_ids:
        raise ValueError(
            "Focused registry contains experiments not declared as essential "
            f"or future additions: {undeclared_registry_ids}."
        )
    if missing_registry_ids:
        raise ValueError(
            "Focused experiment lists contain IDs missing from the registry: "
            f"{missing_registry_ids}."
        )
    if set(FUTURE_CANDIDATE_EXPERIMENT_IDS).intersection(
        FUTURE_CONTROL_EXPERIMENT_IDS
    ):
        raise ValueError(
            "An experiment cannot be both a future candidate and future control."
        )
    registry_by_id = {
        experiment.experiment_id: experiment
        for experiment in EXPERIMENT_REGISTRY
    }
    for experiment_id in FUTURE_CANDIDATE_EXPERIMENT_IDS:
        experiment = registry_by_id[experiment_id]
        if experiment.strategy != "patient_embedding":
            raise ValueError(
                f"{experiment_id}: future candidates must produce one image-"
                "derived patient embedding before classification."
            )
        if experiment.role in {
            "negative_control",
            "segmentation_representation_control",
        }:
            raise ValueError(
                f"{experiment_id}: a future candidate cannot carry a control role."
            )

    unknown_matched_candidate_keys = sorted(
        set(FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE)
        - set(FUTURE_CANDIDATE_EXPERIMENT_IDS)
    )
    if unknown_matched_candidate_keys:
        raise ValueError(
            "FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE contains keys that are not "
            f"declared future candidates: {unknown_matched_candidate_keys}."
        )
    for candidate_id in FUTURE_CANDIDATE_EXPERIMENT_IDS:
        control_ids = tuple(
            FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.get(candidate_id, ())
        )
        if not control_ids:
            raise ValueError(
                f"{candidate_id}: every future candidate must declare at least "
                "one representation-matched control."
            )
        if len(control_ids) != len(set(control_ids)):
            raise ValueError(
                f"{candidate_id}: matched control IDs must be unique."
            )
        for control_id in control_ids:
            if control_id not in registry_by_id:
                raise ValueError(
                    f"{candidate_id}: unknown matched control {control_id!r}."
                )
            if control_id not in declared_active_ids:
                raise ValueError(
                    f"{candidate_id}: matched control {control_id!r} must be "
                    "active as essential or future control."
                )
            if control_id == candidate_id:
                raise ValueError(
                    f"{candidate_id}: candidate cannot be its own control."
                )
            if registry_by_id[control_id].role not in {
                "negative_control",
                "segmentation_representation_control",
            }:
                raise ValueError(
                    f"{candidate_id}: {control_id!r} must carry a control role."
                )
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

    repeated_comparison_names = [
        row[0] for row in REPEATED_STABILITY_COMPARISONS
    ]
    if len(repeated_comparison_names) != len(set(repeated_comparison_names)):
        raise ValueError(
            "Repeated-stability comparison names must be unique."
        )
    for comparison_name, reference_id, comparison_id, _ in (
        REPEATED_STABILITY_COMPARISONS
    ):
        if reference_id not in known_experiment_ids:
            raise ValueError(
                f"{comparison_name}: unknown repeated-CV reference "
                f"{reference_id!r}."
            )
        if comparison_id not in known_experiment_ids:
            raise ValueError(
                f"{comparison_name}: unknown repeated-CV comparison "
                f"{comparison_id!r}."
            )
        if reference_id == comparison_id:
            raise ValueError(
                f"{comparison_name}: repeated-CV models must differ."
            )
        if RUN_REPEATED_NESTED_CV_STABILITY and (
            reference_id not in STABILITY_EXPERIMENT_IDS
            or comparison_id not in STABILITY_EXPERIMENT_IDS
        ):
            raise ValueError(
                f"{comparison_name}: both models must be present in "
                "STABILITY_EXPERIMENT_IDS."
            )

    if not PERMUTATION_EXPERIMENT_IDS:
        raise ValueError(
            "PERMUTATION_EXPERIMENT_IDS must contain at least one experiment."
        )
    if len(PERMUTATION_EXPERIMENT_IDS) != len(
        set(PERMUTATION_EXPERIMENT_IDS)
    ):
        raise ValueError(
            "PERMUTATION_EXPERIMENT_IDS must not contain duplicates."
        )

    if (
        RUN_REPEATED_NESTED_CV_STABILITY
        or RUN_PATIENT_LABEL_PERMUTATION_TEST
        or RUN_NESTED_MODEL_SELECTION_AUDIT
        or RUN_SELECTION_ADJUSTED_PERMUTATION_TEST
    ):
        enabled_by_id = {
            experiment.experiment_id: experiment for experiment in experiments
        }
        requested_analysis_ids = set()
        if RUN_REPEATED_NESTED_CV_STABILITY:
            requested_analysis_ids.update(STABILITY_EXPERIMENT_IDS)
        if RUN_PATIENT_LABEL_PERMUTATION_TEST:
            requested_analysis_ids.update(PERMUTATION_EXPERIMENT_IDS)
        if RUN_NESTED_MODEL_SELECTION_AUDIT or (
            RUN_SELECTION_ADJUSTED_PERMUTATION_TEST
        ):
            requested_analysis_ids.update(
                MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
            )

        for analysis_experiment_id in sorted(requested_analysis_ids):
            if analysis_experiment_id not in enabled_by_id:
                raise ValueError(
                    "The requested focused analysis experiment must be enabled: "
                    f"{analysis_experiment_id!r}."
                )
            analysis_experiment = enabled_by_id[analysis_experiment_id]
            if analysis_experiment.classifier_type not in {
                "logistic_regression",
                "linear_svm",
            }:
                raise ValueError(
                    "Unsupported classifier for stability/permutation analysis."
                )

        # Repeated split stability can also evaluate one-row-per-patient tabular
        # nuisance controls. Ordinary label permutation and candidate selection
        # remain restricted to image-derived patient embeddings because their
        # purpose is to validate the candidate modelling path, not to promote a
        # metadata-only control as a candidate disease detector.
        for experiment_id in STABILITY_EXPERIMENT_IDS:
            experiment = enabled_by_id[experiment_id]
            if experiment.strategy not in {
                "patient_embedding",
                "patient_tabular",
            }:
                raise ValueError(
                    f"{experiment_id}: repeated stability requires one row per "
                    "patient after preparation."
                )
        for experiment_id in set(PERMUTATION_EXPERIMENT_IDS).union(
            MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
        ):
            experiment = enabled_by_id[experiment_id]
            if experiment.strategy != "patient_embedding":
                raise ValueError(
                    f"{experiment_id}: permutation/model selection is reserved "
                    "for image-derived patient-embedding candidates."
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
        "standardized_fixed_center_50",
        "standardized_fixed_center_60",
        "standardized_fixed_center_70",
        "standardized_roi_zero_background",
        "standardized_roi_zero_bg_center_fallback",
        "standardized_roi_bbox",
        "standardized_outside_large_bbox",
        "standardized_soft_monai_mask_only",
        "standardized_hard_monai_mask_only",
        "standardized_monai_bbox_mask_only",
        "standardized_soft_monai_histogram_only",
        "standardized_soft_monai_block_shuffled",
        "standardized_canonical_hard_monai_mask_only",
        "standardized_canonical_soft_monai_mask_only",
        "standardized_heart_centered_fixed_fov_region_norm",
        "standardized_hard_support_region_norm",
        "standardized_a17_exact_support_mask_only",
        "standardized_a17_support_intensity_affine_shuffled",
        "standardized_a17_exact_support_complement_region_norm",
        "standardized_outside_whole_heart_region_norm",
        "standardized_fixed_periphery_region_norm",
        "provenance_only",
        "monai_qc_only",
        "standardization_qc_only",
        "standardized_monai_qc_only",
        "n_slices_only",
        "n_series_only",
        "series_length_only",
        "native_geometry_only",
        "file_size_only",
    }
    valid_strategies = {
        "patient_embedding",
        "slice_probability_fusion",
        "patient_tabular",
    }
    valid_pooling = {
        "hierarchical",
        "sequence_view_balanced",
        "flat",
        "fixed_chunk",
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
        if experiment.slice_filter not in {
            "all",
            "standardized_monai_valid",
        }:
            raise ValueError(
                f"{experiment.experiment_id}: invalid slice_filter "
                f"{experiment.slice_filter!r}."
            )
        if experiment.strategy == "patient_tabular" and (
            experiment.slice_filter != "all"
        ):
            raise ValueError(
                f"{experiment.experiment_id}: patient-tabular experiments "
                "cannot use image-row slice filters."
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
                "standardized_roi_zero_bg_center_fallback",
                "standardized_roi_bbox",
                "standardized_hard_support_region_norm",
            }
        ):
            raise ValueError(
                f"{experiment.experiment_id}: slice dropout is reviewed only "
                "for image-based patient-embedding experiments."
            )

        if experiment.deduplicate_exact_within_patient and (
            experiment.strategy != "patient_embedding"
        ):
            raise ValueError(
                f"{experiment.experiment_id}: exact within-patient "
                "deduplication is reviewed only for patient-embedding "
                "experiments."
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
                "n_slices_only",
                "n_series_only",
                "series_length_only",
                "native_geometry_only",
                "file_size_only",
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
    if not 1 <= STANDARDIZED_CONTENT_LONG_SIDE <= MONAI_INPUT_SIZE:
        raise ValueError(
            "STANDARDIZED_CONTENT_LONG_SIDE must lie in [1, MONAI_INPUT_SIZE]."
        )
    if FIXED_CHUNK_SIZE <= 0:
        raise ValueError("FIXED_CHUNK_SIZE must be strictly positive.")
    for setting_name, setting_value in (
        ("FOCUSED_FIXED_HEART_FOV_FRACTION", FOCUSED_FIXED_HEART_FOV_FRACTION),
        (
            "FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION",
            FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION,
        ),
        (
            "FOCUSED_FIXED_PERIPHERY_EXCLUSION_FRACTION",
            FOCUSED_FIXED_PERIPHERY_EXCLUSION_FRACTION,
        ),
    ):
        if not 0.0 < float(setting_value) <= 1.0:
            raise ValueError(f"{setting_name} must lie in (0,1].")
    if FOCUSED_WHOLE_HEART_EXCLUSION_BBOX_CONTEXT_FRACTION < 0.0:
        raise ValueError(
            "FOCUSED_WHOLE_HEART_EXCLUSION_BBOX_CONTEXT_FRACTION cannot be negative."
        )
    if FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL <= 0 or (
        FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL % 2 == 0
    ):
        raise ValueError(
            "FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL must be a positive odd integer."
        )
    if not (
        0.0 <= FOCUSED_REGION_NORM_LOWER_PERCENTILE
        < FOCUSED_REGION_NORM_UPPER_PERCENTILE <= 100.0
    ):
        raise ValueError("Invalid focused region-normalization percentiles.")
    if FOCUSED_REGION_NORM_MIN_PIXELS < 1:
        raise ValueError("FOCUSED_REGION_NORM_MIN_PIXELS must be positive.")
    if FOCUSED_REGION_NORM_HISTOGRAM_BINS < 2:
        raise ValueError(
            "FOCUSED_REGION_NORM_HISTOGRAM_BINS must be at least 2."
        )
    if not 0.0 < FOCUSED_REGION_NORM_MIN_DYNAMIC_RANGE <= 1.0:
        raise ValueError(
            "FOCUSED_REGION_NORM_MIN_DYNAMIC_RANGE must lie in (0,1]."
        )
    if not str(FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION).strip():
        raise ValueError(
            "FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION must be non-empty."
        )
    if USE_CUDA_AMP:
        raise ValueError(
            "The focused reportable run requires USE_CUDA_AMP=False."
        )
    if len(FIXED_CENTER_CROP_FRACTIONS) != 3 or any(
        not 0.0 < float(value) <= 1.0
        for value in FIXED_CENTER_CROP_FRACTIONS
    ):
        raise ValueError(
            "FIXED_CENTER_CROP_FRACTIONS must contain three values in (0,1]."
        )
    if C_SELECTION_AUC_TOLERANCE < 0.0:
        raise ValueError("C_SELECTION_AUC_TOLERANCE cannot be negative.")
    if REPEATED_NESTED_CV_REPEATS <= 0:
        raise ValueError("REPEATED_NESTED_CV_REPEATS must be positive.")
    if LABEL_PERMUTATION_REPLICATES <= 0:
        raise ValueError("LABEL_PERMUTATION_REPLICATES must be positive.")
    if not MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS:
        raise ValueError(
            "MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS must not be empty."
        )
    if len(MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS) != len(
        set(MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS)
    ):
        raise ValueError(
            "MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS must be unique."
        )
    if MODEL_SELECTION_AUC_TOLERANCE < 0.0:
        raise ValueError("MODEL_SELECTION_AUC_TOLERANCE cannot be negative.")
    if SELECTION_ADJUSTED_PERMUTATION_REPLICATES <= 0:
        raise ValueError(
            "SELECTION_ADJUSTED_PERMUTATION_REPLICATES must be positive."
        )
    if not 0.5 < FOCUSED_REPEATED_SUPPORT_FRACTION <= 1.0:
        raise ValueError(
            "FOCUSED_REPEATED_SUPPORT_FRACTION must lie in (0.5,1]."
        )
    if FOCUSED_ROBUSTNESS_AUC_TOLERANCE < 0.0:
        raise ValueError(
            "FOCUSED_ROBUSTNESS_AUC_TOLERANCE cannot be negative."
        )
    if RUN_SELECTION_ADJUSTED_PERMUTATION_TEST and not (
        RUN_NESTED_MODEL_SELECTION_AUDIT
    ):
        raise ValueError(
            "Selection-adjusted permutation requires nested model selection."
        )
    if FEATURE_MODES_PER_ENCODER_CALL <= 0:
        raise ValueError("FEATURE_MODES_PER_ENCODER_CALL must be positive.")

    # The four V4 segmentation-representation controls use fixed synthetic
    # encodings. Validate their spatial/distribution settings before the very
    # expensive feature-bank extraction begins.
    if MONAI_SOFT_HISTOGRAM_BINS < 2:
        raise ValueError("MONAI_SOFT_HISTOGRAM_BINS must be at least 2.")
    if MONAI_SOFT_BLOCK_SHUFFLE_GRID <= 1:
        raise ValueError(
            "MONAI_SOFT_BLOCK_SHUFFLE_GRID must be greater than 1."
        )
    if IMG_SIZE % MONAI_SOFT_BLOCK_SHUFFLE_GRID != 0:
        raise ValueError(
            "IMG_SIZE must be divisible by MONAI_SOFT_BLOCK_SHUFFLE_GRID so "
            "block shuffling preserves every pixel exactly."
        )
    if not 0.0 < MONAI_CANONICAL_MASK_CONTENT_FRACTION <= 1.0:
        raise ValueError(
            "MONAI_CANONICAL_MASK_CONTENT_FRACTION must lie in (0,1]."
        )
    if not str(MONAI_SOFT_BLOCK_SHUFFLE_VERSION).strip():
        raise ValueError(
            "MONAI_SOFT_BLOCK_SHUFFLE_VERSION must be a non-empty string."
        )

    # Sequence/view analysis is optional because the released JPEG folders do
    # not expose validated DICOM sequence metadata. When enabled, it must be
    # driven by a completed external copy of the blinded annotation template.
    if RUN_ANNOTATED_SERIES_SUBSET_ANALYSIS:
        if not SERIES_ANNOTATION_INPUT_PATH:
            raise ValueError(
                "RUN_ANNOTATED_SERIES_SUBSET_ANALYSIS=True requires "
                "SERIES_ANNOTATION_INPUT_PATH."
            )
        annotation_path = Path(SERIES_ANNOTATION_INPUT_PATH).expanduser()
        if not annotation_path.is_file():
            raise FileNotFoundError(
                "Completed series annotation CSV was not found: "
                f"{annotation_path}"
            )
        try:
            annotation_path.resolve().relative_to(OUTPUT_DIR.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(
                "SERIES_ANNOTATION_INPUT_PATH must point to a completed copy "
                "stored outside the current run directory. Stage 2 cleans the "
                "run-specific output package before execution."
            )
        if not str(ANNOTATED_SERIES_SELECTION_NAME).strip():
            raise ValueError(
                "ANNOTATED_SERIES_SELECTION_NAME must be non-empty."
            )
        if any(
            character in str(ANNOTATED_SERIES_SELECTION_NAME)
            for character in ("/", "\\")
        ):
            raise ValueError(
                "ANNOTATED_SERIES_SELECTION_NAME must be a safe directory "
                "name without path separators."
            )
        if not ANNOTATED_SERIES_EXPERIMENT_IDS:
            raise ValueError(
                "ANNOTATED_SERIES_EXPERIMENT_IDS must not be empty when the "
                "optional annotated-series analysis is enabled."
            )
        if len(ANNOTATED_SERIES_EXPERIMENT_IDS) != len(
            set(ANNOTATED_SERIES_EXPERIMENT_IDS)
        ):
            raise ValueError(
                "ANNOTATED_SERIES_EXPERIMENT_IDS must not contain duplicates."
            )
        enabled_by_id = {
            experiment.experiment_id: experiment for experiment in experiments
        }
        for experiment_id in ANNOTATED_SERIES_EXPERIMENT_IDS:
            if experiment_id not in enabled_by_id:
                raise ValueError(
                    "Annotated-series experiment must be enabled in the main "
                    f"suite: {experiment_id!r}."
                )
            if enabled_by_id[experiment_id].strategy != "patient_embedding":
                raise ValueError(
                    "Annotated-series subset analysis supports image-based "
                    "patient-embedding experiments only."
                )
        if not ANNOTATED_SERIES_ALLOWED_CONFIDENCE_VALUES:
            raise ValueError(
                "ANNOTATED_SERIES_ALLOWED_CONFIDENCE_VALUES must contain at "
                "least one accepted confidence label."
            )

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

    focused_contract_summary = validate_focused_transform_contract()
    if focused_contract_summary.get("status") != "PASS":
        raise RuntimeError("Focused A17/C31/C32/C33 transform contract failed.")

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


def _fixed_content_canvas_geometry(height, width):
    """Return deterministic fixed-content geometry for a 256x256 canvas.

    The longest retained image side is resized to exactly
    ``STANDARDIZED_CONTENT_LONG_SIDE`` pixels. Unlike the historical MONAI
    helper, this transformation deliberately allows both downsampling and
    upsampling. Consequently, native image resolution and the number of removed
    dark-border pixels cannot change the final content scale. Aspect ratio is
    preserved and remains auditable through the final content width/height.
    """

    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Fixed-content geometry requires positive dimensions, got "
            f"{height}x{width}."
        )
    if not 1 <= STANDARDIZED_CONTENT_LONG_SIDE <= MONAI_INPUT_SIZE:
        raise ValueError(
            "STANDARDIZED_CONTENT_LONG_SIDE must lie between 1 and "
            "MONAI_INPUT_SIZE."
        )

    scale = float(STANDARDIZED_CONTENT_LONG_SIDE / max(height, width))
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))

    # Rounding can move the nominal longest side by one pixel. Force it back to
    # the predeclared target without changing the aspect-ratio calculation for
    # the shorter side.
    if height >= width:
        resized_height = int(STANDARDIZED_CONTENT_LONG_SIDE)
    else:
        resized_width = int(STANDARDIZED_CONTENT_LONG_SIDE)

    resized_height = min(resized_height, MONAI_INPUT_SIZE)
    resized_width = min(resized_width, MONAI_INPUT_SIZE)
    top = (MONAI_INPUT_SIZE - resized_height) // 2
    left = (MONAI_INPUT_SIZE - resized_width) // 2
    return resized_height, resized_width, top, left, scale


def resize_to_fixed_content_canvas(image):
    """Resize retained content to a fixed scale and center it in 256x256.

    Downsampling uses INTER_AREA. Upsampling uses INTER_CUBIC because the input
    is a continuous grayscale image rather than a categorical mask. The same
    geometry is used for the MONAI min-max image and the robustly scaled
    EfficientNet image, so their masks and pixels remain spatially aligned.
    """

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.size == 0:
        raise ValueError(
            "Fixed-content resizing requires a non-empty 2D grayscale image."
        )

    height, width = image.shape
    resized_height, resized_width, top, left, scale = (
        _fixed_content_canvas_geometry(height, width)
    )
    if resized_height != height or resized_width != width:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
    else:
        resized = image

    # Cubic interpolation can overshoot slightly beyond the source [0,1]
    # range. Clipping preserves the documented network-input contract.
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)

    canvas = np.zeros(
        (MONAI_INPUT_SIZE, MONAI_INPUT_SIZE),
        dtype=np.float32,
    )
    canvas[
        top:top + resized_height,
        left:left + resized_width,
    ] = resized
    return canvas, resized_height, resized_width


def build_fixed_content_padding_canvas(height, width):
    """Return only the padding geometry of the fixed-content canvas.

    Pixels occupied by retained image content are zero; pixels introduced by
    the standardized 256x256 canvas are one. Native dark borders are removed
    before this operation and are audited separately through crop-fraction
    features. This prevents their geometry from being reintroduced into the
    supposedly standardized image representation.
    """

    resized_height, resized_width, top, left, _ = (
        _fixed_content_canvas_geometry(height, width)
    )
    canvas = np.ones(
        (MONAI_INPUT_SIZE, MONAI_INPUT_SIZE),
        dtype=np.float32,
    )
    canvas[
        top:top + resized_height,
        left:left + resized_width,
    ] = 0.0
    return canvas, resized_height, resized_width


def build_label_blind_standardized_image(image):
    """Create aligned MONAI/classifier canvases and standardization audits.

    Processing order:

        native uint8 JPEG
            -> conservative dark-uniform edge detection
            -> crop only the detected edge runs
            -> min-max scaling for the pretrained MONAI segmenter
            -> robust fixed-percentile scaling for historical EfficientNet views
            -> unscaled uint8/255 intensity for focused regional normalization
            -> identical fixed-content 240-in-256 geometry for all views
            -> binary fixed-canvas padding control

    The Normal/Sick label, patient ID, series ID, fold and model scores are not
    used. Separating intensity transforms preserves the segmenter's closer
    training-time contract while still preventing bright text/compression
    outliers from defining the classifier's complete dynamic range.

    Returns:
        robust_classifier_canvas:
            256x256 robustly scaled image used to build EfficientNet views.
        monai_minmax_canvas:
            256x256 min-max image used only by the MONAI segmenter.
        raw_classifier_canvas:
            256x256 retained uint8 intensity divided by 255, with no global
            per-image min-max or percentile transformation. The focused A16/A17
            and outside controls normalize only after their final visible region
            has been defined.
        fixed_padding_canvas:
            Binary geometry-only control after content-size normalization.
        features:
            Label-blind crop, intensity and final-canvas audit features.
    """

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"Standardization expects a 2D grayscale image, got {image.shape}."
        )

    height, width = image.shape
    top, bottom, left, right = detect_label_blind_dark_padding_bounds(image)
    cropped = image[top:bottom, left:right]
    if cropped.size == 0:
        raise RuntimeError("Label-blind standardization produced an empty crop.")

    robust_scaled, lower, upper = robust_scale_intensity_0_1(cropped)
    monai_scaled = scale_intensity_0_1(cropped)
    raw_scaled = cropped.astype(np.float32) / 255.0

    robust_canvas, resized_height, resized_width = (
        resize_to_fixed_content_canvas(robust_scaled)
    )
    monai_canvas, monai_height, monai_width = (
        resize_to_fixed_content_canvas(monai_scaled)
    )
    raw_canvas, raw_height, raw_width = (
        resize_to_fixed_content_canvas(raw_scaled)
    )
    if (
        (resized_height, resized_width) != (monai_height, monai_width)
        or (resized_height, resized_width) != (raw_height, raw_width)
    ):
        raise RuntimeError(
            "MONAI, robust-classifier and raw standardized geometries are "
            "misaligned."
        )

    padding_canvas, padding_height, padding_width = (
        build_fixed_content_padding_canvas(*cropped.shape)
    )
    if (resized_height, resized_width) != (padding_height, padding_width):
        raise RuntimeError(
            "Fixed padding control does not align with standardized content."
        )

    top_fraction = float(top / height)
    bottom_fraction = float((height - bottom) / height)
    left_fraction = float(left / width)
    right_fraction = float((width - right) / width)
    retained_height_fraction = float((bottom - top) / height)
    retained_width_fraction = float((right - left) / width)
    detected_padding_fraction = float(
        1.0 - retained_height_fraction * retained_width_fraction
    )
    content_height_fraction = float(resized_height / MONAI_INPUT_SIZE)
    content_width_fraction = float(resized_width / MONAI_INPUT_SIZE)
    pipeline_padding_fraction = float(
        1.0
        - (resized_height * resized_width)
        / float(MONAI_INPUT_SIZE * MONAI_INPUT_SIZE)
    )

    features = np.asarray(
        [
            float(
                any(
                    value > 0
                    for value in (
                        top,
                        height - bottom,
                        left,
                        width - right,
                    )
                )
            ),
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
            content_height_fraction,
            content_width_fraction,
            pipeline_padding_fraction,
        ],
        dtype=np.float32,
    )

    if len(features) != len(STANDARDIZATION_FEATURE_NAMES):
        raise RuntimeError(
            "Standardization feature-name and value counts differ."
        )
    if not np.all(np.isfinite(features)):
        raise RuntimeError("Standardization features contain non-finite values.")

    return robust_canvas, monai_canvas, raw_canvas, padding_canvas, features


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
    Load one JPEG MRI slice and construct aligned MONAI, historical classifier,
    focused raw-classifier, and padding-control inputs plus label-free metadata.

    Returns:
        classification_image:
            Tensor [3, 224, 224], values in [0,1].
            Used by EfficientNet only after the selected image representation
            has been created and ImageNet normalization has been applied.

        monai_image:
            Tensor [1, 256, 256], values in [0,1].
            Original min-max baseline input for the MONAI segmenter.

        standardized_classification_image:
            Tensor [3, 224, 224], values in [0,1]. Label-blind native padding
            removal, robust percentile scaling, and fixed 240-in-256 content
            geometry are applied before the EfficientNet resize.

        standardized_monai_image:
            Tensor [1, 256, 256], values in [0,1]. It uses min-max scaling on
            the same retained crop and exactly the same fixed geometry, keeping
            MONAI closer to its documented intensity contract.

        standardized_raw_classification_image:
            Tensor [3, 224, 224], values in [0,1], derived from retained native
            uint8 intensity divided by 255 without global percentile scaling.
            Focused candidate/control branches compute their intensity limits
            only after the final visible cardiac or extracardiac region is known.

        detected_padding_image:
            Tensor [3, 224, 224], binary. It contains only the standardized
            fixed-canvas padding geometry; removed native padding is represented
            separately by tabular crop-fraction audit features.

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
    WHY SEPARATE ALIGNED INPUT TENSORS?
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
    least one model's expected input distribution. Focused regional controls
    additionally need retained raw intensity before region selection. Therefore
    segmentation, historical classification and focused regional preprocessing
    remain separate while sharing exactly one spatial geometry.

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

        # Build a second, label-blind branch on one shared crop but with two
        # intensity contracts. MONAI receives a min-max scaled image, while
        # EfficientNet receives robust 1st/99th-percentile scaling. Both are
        # resized so the retained content's longest side is exactly 240 pixels
        # and centered in the same 256x256 canvas. This removes the former
        # dependence of pipeline-added padding on native resolution and crop size.
        (
            standardized_classifier_canvas,
            standardized_monai_canvas,
            standardized_raw_canvas,
            detected_padding_canvas,
            standardization_features,
        ) = build_label_blind_standardized_image(image)
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
            standardized_classifier_canvas,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        standardized_raw_classification_gray = cv2.resize(
            standardized_raw_canvas,
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
        standardized_raw_classification_image = np.stack(
            [standardized_raw_classification_gray] * 3,
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
            standardized_raw_classification_image = self.transform(
                standardized_raw_classification_image
            )
            detected_padding_image = self.transform(detected_padding_image)
        else:
            classification_image = torch.from_numpy(
                classification_image
            ).permute(2, 0, 1)
            standardized_classification_image = torch.from_numpy(
                standardized_classification_image
            ).permute(2, 0, 1)
            standardized_raw_classification_image = torch.from_numpy(
                standardized_raw_classification_image
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
            standardized_raw_classification_image,
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
        "standardized_fixed_center_50",
        "standardized_fixed_center_60",
        "standardized_fixed_center_70",
        "standardized_roi_zero_background",
        "standardized_roi_zero_bg_center_fallback",
        "standardized_roi_bbox",
        "standardized_outside_large_bbox",
        "standardized_soft_monai_mask_only",
        "standardized_hard_monai_mask_only",
        "standardized_monai_bbox_mask_only",
        "standardized_soft_monai_histogram_only",
        "standardized_soft_monai_block_shuffled",
        "standardized_canonical_hard_monai_mask_only",
        "standardized_canonical_soft_monai_mask_only",
        "standardized_heart_centered_fixed_fov_region_norm",
        "standardized_hard_support_region_norm",
        "standardized_a17_exact_support_mask_only",
        "standardized_a17_support_intensity_affine_shuffled",
        "standardized_a17_exact_support_complement_region_norm",
        "standardized_outside_whole_heart_region_norm",
        "standardized_fixed_periphery_region_norm",
    )
    requested = {
        experiment.feature_mode
        for experiment in experiments
        if experiment.feature_mode in canonical_order
    }
    return tuple(mode for mode in canonical_order if mode in requested)


# MONAI inference is requested not only by image representations but also by a
# future tabular QC-only experiment. Keeping this requirement in one helper
# prevents a reduced registry from silently leaving the requested QC branch as
# an all-NaN sentinel.
ORIGINAL_MONAI_IMAGE_MODES = frozenset({"monai_roi", "outside_heart"})
STANDARDIZED_MONAI_IMAGE_MODES = frozenset(
    {
        "standardized_monai_roi",
        "standardized_center_crop",
        "standardized_roi_zero_background",
        "standardized_roi_zero_bg_center_fallback",
        "standardized_roi_bbox",
        "standardized_outside_large_bbox",
        "standardized_soft_monai_mask_only",
        "standardized_hard_monai_mask_only",
        "standardized_monai_bbox_mask_only",
        "standardized_soft_monai_histogram_only",
        "standardized_soft_monai_block_shuffled",
        "standardized_canonical_hard_monai_mask_only",
        "standardized_canonical_soft_monai_mask_only",
        "standardized_heart_centered_fixed_fov_region_norm",
        "standardized_hard_support_region_norm",
        "standardized_a17_exact_support_mask_only",
        "standardized_a17_support_intensity_affine_shuffled",
        "standardized_a17_exact_support_complement_region_norm",
        "standardized_outside_whole_heart_region_norm",
    }
)


def required_monai_qc_branches(experiments):
    """Return which original/standardized MONAI inference branches are needed."""

    experiments = tuple(experiments)
    image_modes = set(required_efficientnet_feature_modes(experiments))
    all_feature_modes = {experiment.feature_mode for experiment in experiments}
    return {
        "original": bool(
            image_modes.intersection(ORIGINAL_MONAI_IMAGE_MODES)
            or "monai_qc_only" in all_feature_modes
        ),
        "standardized": bool(
            image_modes.intersection(STANDARDIZED_MONAI_IMAGE_MODES)
            or "standardized_monai_qc_only" in all_feature_modes
        ),
    }




# =============================================================================
# AUTOMATIC CROSS-CLASS SERIES HARMONIZATION — V7.2 IMPLEMENTATION
# =============================================================================


def _series_harmonization_evenly_spaced_indices(n_items, maximum_items):
    """Return deterministic distinct positions spanning an ordered series."""

    n_items = int(n_items)
    maximum_items = int(maximum_items)
    if n_items <= 0 or maximum_items <= 0:
        raise ValueError("Series sampling requires positive item counts.")
    if n_items <= maximum_items:
        return np.arange(n_items, dtype=np.int64)
    positions = np.rint(
        np.linspace(0, n_items - 1, maximum_items, dtype=np.float64)
    ).astype(np.int64)
    return np.unique(positions)


def _series_harmonization_entropy_from_histogram(histogram):
    """Return Shannon entropy in bits for a normalized nonnegative histogram."""

    histogram = np.asarray(histogram, dtype=np.float64)
    total = float(histogram.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    probabilities = histogram / total
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log2(probabilities[positive])))


def _series_harmonization_frame_descriptor(image_path):
    """Build one label-free descriptor from a sampled grayscale JPEG frame.

    The descriptor deliberately excludes patient ID, class label, series-folder
    text, model scores and all fitted CAD-classifier quantities. Native geometry
    and file-size-per-pixel are included because they help identify export/layout
    families that should be represented in both classes rather than appearing
    only on one side of the cohort.
    """

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(
            "Series harmonization could not decode a sampled JPEG: "
            f"{path}. The main pipeline also treats unreadable files as errors."
        )
    if image.ndim != 2 or image.size == 0:
        raise RuntimeError(
            f"Invalid sampled JPEG shape for series harmonization: {path}: "
            f"{image.shape}."
        )

    height, width = image.shape
    n_pixels = max(1, int(height) * int(width))
    file_size = int(path.stat().st_size)
    interpolation = (
        cv2.INTER_AREA
        if max(height, width) >= SERIES_HARMONIZATION_DESCRIPTOR_IMAGE_SIZE
        else cv2.INTER_CUBIC
    )
    resized = cv2.resize(
        image,
        (
            SERIES_HARMONIZATION_DESCRIPTOR_IMAGE_SIZE,
            SERIES_HARMONIZATION_DESCRIPTOR_IMAGE_SIZE,
        ),
        interpolation=interpolation,
    ).astype(np.float32)

    lower = float(np.percentile(resized, 1.0))
    upper = float(np.percentile(resized, 99.0))
    if upper <= lower:
        scaled = np.zeros_like(resized, dtype=np.float32)
    else:
        scaled = np.clip((resized - lower) / (upper - lower), 0.0, 1.0)

    histogram, _ = np.histogram(
        scaled,
        bins=SERIES_HARMONIZATION_INTENSITY_HISTOGRAM_BINS,
        range=(0.0, 1.0),
    )
    histogram = histogram.astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)

    centered = scaled - float(np.mean(scaled))
    dct = cv2.dct(centered.astype(np.float32))
    dct_patch = dct[
        :SERIES_HARMONIZATION_DCT_SIDE,
        :SERIES_HARMONIZATION_DCT_SIDE,
    ].reshape(-1)
    # Exclude the DC term because the centered image makes it numerically
    # redundant with the explicitly recorded intensity mean.
    dct_features = dct_patch[1:].astype(np.float64)

    gradient_x = cv2.Sobel(scaled, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(scaled, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)

    size = SERIES_HARMONIZATION_DESCRIPTOR_IMAGE_SIZE
    margin = max(1, size // 8)
    center_start = size // 4
    center_end = size - center_start
    center = scaled[center_start:center_end, center_start:center_end]
    border_mask = np.ones((size, size), dtype=bool)
    border_mask[margin:size - margin, margin:size - margin] = False
    border = scaled[border_mask]
    quadrants = (
        scaled[: size // 2, : size // 2],
        scaled[: size // 2, size // 2 :],
        scaled[size // 2 :, : size // 2],
        scaled[size // 2 :, size // 2 :],
    )

    scalar_features = np.asarray(
        [
            np.log1p(float(height)),
            np.log1p(float(width)),
            float(width) / max(float(height), 1.0),
            np.log1p(float(file_size)),
            np.log1p(float(file_size) / float(n_pixels)),
            lower / 255.0,
            upper / 255.0,
            (upper - lower) / 255.0,
            float(np.mean(scaled)),
            float(np.std(scaled)),
            float(np.quantile(scaled, 0.10)),
            float(np.quantile(scaled, 0.50)),
            float(np.quantile(scaled, 0.90)),
            float(np.mean(scaled > 0.02)),
            float(np.mean(scaled > 0.98)),
            _series_harmonization_entropy_from_histogram(histogram),
            float(np.mean(gradient)),
            float(np.std(gradient)),
            float(np.quantile(gradient, 0.90)),
            float(np.mean(gradient > 0.15)),
            float(np.mean(center)),
            float(np.std(center)),
            float(np.mean(border)),
            float(np.std(border)),
            *[float(np.mean(region)) for region in quadrants],
        ],
        dtype=np.float64,
    )
    descriptor = np.concatenate(
        [scalar_features, histogram, dct_features],
        axis=0,
    )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError(
            f"Non-finite sampled-frame descriptor for {path}."
        )
    return descriptor


def _series_harmonization_descriptor_names():
    """Return stable names aligned with the sampled-frame descriptor."""

    scalar_names = [
        "log_native_height",
        "log_native_width",
        "native_aspect_ratio",
        "log_file_size_bytes",
        "log_bytes_per_native_pixel",
        "robust_lower_intensity_0_1",
        "robust_upper_intensity_0_1",
        "robust_dynamic_range_0_1",
        "scaled_mean",
        "scaled_std",
        "scaled_q10",
        "scaled_q50",
        "scaled_q90",
        "scaled_fraction_above_002",
        "scaled_fraction_above_098",
        "scaled_histogram_entropy_bits",
        "gradient_mean",
        "gradient_std",
        "gradient_q90",
        "gradient_fraction_above_015",
        "center_mean",
        "center_std",
        "border_mean",
        "border_std",
        "quadrant_tl_mean",
        "quadrant_tr_mean",
        "quadrant_bl_mean",
        "quadrant_br_mean",
    ]
    histogram_names = [
        f"histogram_bin_{index:02d}"
        for index in range(SERIES_HARMONIZATION_INTENSITY_HISTOGRAM_BINS)
    ]
    dct_names = []
    for row in range(SERIES_HARMONIZATION_DCT_SIDE):
        for column in range(SERIES_HARMONIZATION_DCT_SIDE):
            if row == 0 and column == 0:
                continue
            dct_names.append(f"centered_dct_r{row}_c{column}")
    return tuple(scalar_names + histogram_names + dct_names)


def _series_harmonization_build_series_table(samples):
    """Group immutable image rows into patient-scoped folder-series proxies."""

    grouped = defaultdict(list)
    labels_by_series = {}
    patients_by_series = {}
    for sample in samples:
        if len(sample) != 4:
            raise ValueError(
                "Every sample must be (image_path, label, patient_id, series_id)."
            )
        image_path, label, patient_id, series_id = sample
        patient_id = str(patient_id)
        series_id = str(series_id)
        label = int(label)
        existing_label = labels_by_series.get(series_id)
        existing_patient = patients_by_series.get(series_id)
        if existing_label is not None and existing_label != label:
            raise RuntimeError(f"Series {series_id} has inconsistent labels.")
        if existing_patient is not None and existing_patient != patient_id:
            raise RuntimeError(
                f"Series proxy {series_id} crosses patients: "
                f"{existing_patient} versus {patient_id}."
            )
        labels_by_series[series_id] = label
        patients_by_series[series_id] = patient_id
        grouped[series_id].append(sample)

    records = []
    for series_id in sorted(grouped):
        rows = sorted(grouped[series_id], key=lambda item: str(item[0]))
        records.append(
            {
                "series_id": series_id,
                "patient_id": patients_by_series[series_id],
                "label": labels_by_series[series_id],
                "samples": rows,
                "n_images": int(len(rows)),
            }
        )
    if not records:
        raise RuntimeError("Series harmonization received no series proxies.")
    return records


def _series_harmonization_build_descriptors(series_records):
    """Return robust series descriptors from deterministic sampled frames."""

    frame_names = _series_harmonization_descriptor_names()
    descriptors = []
    audit_rows = []
    progress_step = max(1, len(series_records) // 20)
    print(
        "[SERIES HARMONIZATION] Building label-free descriptors for "
        f"{len(series_records)} folder-defined series proxies.",
        flush=True,
    )

    for series_index, record in enumerate(series_records, start=1):
        samples = record["samples"]
        positions = _series_harmonization_evenly_spaced_indices(
            len(samples),
            SERIES_HARMONIZATION_SAMPLE_FRAMES_PER_SERIES,
        )
        frame_matrix = np.stack(
            [
                _series_harmonization_frame_descriptor(samples[position][0])
                for position in positions
            ],
            axis=0,
        )
        frame_median = np.median(frame_matrix, axis=0)
        frame_iqr = np.quantile(frame_matrix, 0.75, axis=0) - np.quantile(
            frame_matrix, 0.25, axis=0
        )
        frame_range = np.max(frame_matrix, axis=0) - np.min(
            frame_matrix, axis=0
        )
        series_prefix = np.asarray(
            [
                np.log1p(float(record["n_images"])),
                float(len(positions)) / float(record["n_images"]),
            ],
            dtype=np.float64,
        )
        descriptor = np.concatenate(
            [series_prefix, frame_median, frame_iqr, frame_range],
            axis=0,
        )
        if not np.all(np.isfinite(descriptor)):
            raise RuntimeError(
                f"Non-finite series descriptor for {record['series_id']}."
            )
        descriptors.append(descriptor)
        audit_rows.append(
            {
                "series_id": record["series_id"],
                "patient_id": record["patient_id"],
                "label": int(record["label"]),
                "n_images": int(record["n_images"]),
                "n_sampled_images": int(len(positions)),
                "sampled_positions": ";".join(map(str, positions.tolist())),
            }
        )
        if (
            series_index == 1
            or series_index % progress_step == 0
            or series_index == len(series_records)
        ):
            print(
                f"[SERIES HARMONIZATION] descriptors={series_index}/"
                f"{len(series_records)} "
                f"({100.0 * series_index / len(series_records):.1f}%)",
                flush=True,
            )

    descriptor_names = (
        "log_series_image_count",
        "sampled_fraction",
        *[f"median__{name}" for name in frame_names],
        *[f"iqr__{name}" for name in frame_names],
        *[f"range__{name}" for name in frame_names],
    )
    matrix = np.asarray(descriptors, dtype=np.float64)
    if matrix.shape != (len(series_records), len(descriptor_names)):
        raise RuntimeError(
            "Series descriptor/name shape mismatch: "
            f"matrix={matrix.shape}, names={len(descriptor_names)}."
        )
    return matrix, descriptor_names, audit_rows


def _series_harmonization_robust_standardize(descriptor_matrix):
    """Robustly scale descriptor columns without using class labels."""

    matrix = np.asarray(descriptor_matrix, dtype=np.float64)
    center = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    iqr_scale = (q75 - q25) / 1.349
    mad_scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    scale = np.maximum(iqr_scale, mad_scale)
    active = np.isfinite(scale) & (scale > 1e-8)
    if not np.any(active):
        raise RuntimeError(
            "All series-descriptor columns are constant; harmonization cannot run."
        )
    scaled = (matrix[:, active] - center[active]) / scale[active]
    scaled = np.clip(scaled, -8.0, 8.0)
    if not np.all(np.isfinite(scaled)):
        raise RuntimeError("Robustly scaled series descriptors are non-finite.")
    return scaled.astype(np.float64), active, center, scale


def _series_harmonization_cluster_distance_inliers(distances, cluster_labels):
    """Return conservative label-free inlier decisions within each cluster."""

    distances = np.asarray(distances, dtype=np.float64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    inlier = np.ones(len(distances), dtype=bool)
    thresholds = {}
    for cluster_id in sorted(np.unique(cluster_labels)):
        indices = np.flatnonzero(cluster_labels == cluster_id)
        values = distances[indices]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_threshold = median + (
            SERIES_HARMONIZATION_CLUSTER_OUTLIER_MAD_MULTIPLIER
            * 1.4826
            * mad
        )
        empirical_threshold = float(
            np.quantile(
                values,
                SERIES_HARMONIZATION_CLUSTER_OUTLIER_MIN_QUANTILE,
            )
        )
        threshold = max(robust_threshold, empirical_threshold)
        # A zero-MAD cluster can contain many exactly repeated descriptors. The
        # empirical quantile remains a safe finite threshold in that case.
        if not np.isfinite(threshold):
            raise RuntimeError(
                f"Non-finite cluster-distance threshold for cluster {cluster_id}."
            )
        inlier[indices] = values <= threshold + 1e-12
        thresholds[int(cluster_id)] = threshold
    return inlier, thresholds


def _series_harmonization_fit_multiresolution_clusters(
    scaled_descriptors,
    series_records,
):
    """Fit unlabeled clusters, then identify cluster support in both classes."""

    n_series = len(series_records)
    requested_counts = tuple(
        int(value) for value in SERIES_HARMONIZATION_CLUSTER_COUNTS
    )
    cluster_counts = tuple(
        value for value in requested_counts if 2 <= value < n_series
    )
    if len(cluster_counts) < 2:
        raise RuntimeError(
            "Series harmonization needs at least two valid clustering "
            f"resolutions; requested={requested_counts}, n_series={n_series}."
        )

    labels = np.asarray(
        [int(record["label"]) for record in series_records],
        dtype=np.int64,
    )
    patient_ids = np.asarray(
        [str(record["patient_id"]) for record in series_records]
    )
    shared_votes = np.zeros(n_series, dtype=np.int64)
    distance_inlier_votes = np.zeros(n_series, dtype=np.int64)
    cluster_assignments = {}
    cluster_shared_flags = {}
    cluster_inlier_flags = {}
    assigned_distances = {}
    cluster_summary_rows = []

    print(
        "[SERIES HARMONIZATION] Fitting label-free multi-resolution "
        f"clusters: {list(cluster_counts)}.",
        flush=True,
    )

    for resolution_index, n_clusters in enumerate(cluster_counts):
        model = KMeans(
            n_clusters=n_clusters,
            random_state=(
                SERIES_HARMONIZATION_RANDOM_STATE + resolution_index
            ),
            n_init=SERIES_HARMONIZATION_KMEANS_N_INIT,
            algorithm="lloyd",
        )
        assignment = model.fit_predict(scaled_descriptors).astype(np.int64)
        distances_to_centers = model.transform(scaled_descriptors)
        own_distance = distances_to_centers[
            np.arange(n_series), assignment
        ].astype(np.float64)
        inlier, distance_thresholds = (
            _series_harmonization_cluster_distance_inliers(
                own_distance,
                assignment,
            )
        )

        shared_by_cluster = {}
        for cluster_id in sorted(np.unique(assignment)):
            indices = np.flatnonzero(assignment == cluster_id)
            normal_indices = indices[labels[indices] == 0]
            sick_indices = indices[labels[indices] == 1]
            normal_patients = len(set(patient_ids[normal_indices].tolist()))
            sick_patients = len(set(patient_ids[sick_indices].tolist()))
            normal_series = int(len(normal_indices))
            sick_series = int(len(sick_indices))
            shared = bool(
                normal_patients
                >= SERIES_HARMONIZATION_MIN_NORMAL_PATIENTS_PER_CLUSTER
                and sick_patients
                >= SERIES_HARMONIZATION_MIN_SICK_PATIENTS_PER_CLUSTER
                and normal_series
                >= SERIES_HARMONIZATION_MIN_NORMAL_SERIES_PER_CLUSTER
                and sick_series
                >= SERIES_HARMONIZATION_MIN_SICK_SERIES_PER_CLUSTER
            )
            shared_by_cluster[int(cluster_id)] = shared
            cluster_summary_rows.append(
                {
                    "resolution_n_clusters": int(n_clusters),
                    "cluster_id": int(cluster_id),
                    "total_series": int(len(indices)),
                    "normal_series": normal_series,
                    "sick_series": sick_series,
                    "normal_unique_patients": int(normal_patients),
                    "sick_unique_patients": int(sick_patients),
                    "shared_cross_class_support": int(shared),
                    "distance_threshold": float(
                        distance_thresholds[int(cluster_id)]
                    ),
                    "distance_outlier_series": int(
                        np.sum(~inlier[indices])
                    ),
                }
            )

        shared_for_row = np.asarray(
            [shared_by_cluster[int(value)] for value in assignment],
            dtype=bool,
        )
        shared_votes += (shared_for_row & inlier).astype(np.int64)
        distance_inlier_votes += inlier.astype(np.int64)
        cluster_assignments[int(n_clusters)] = assignment
        cluster_shared_flags[int(n_clusters)] = shared_for_row
        cluster_inlier_flags[int(n_clusters)] = inlier
        assigned_distances[int(n_clusters)] = own_distance

    return {
        "cluster_counts": cluster_counts,
        "shared_votes": shared_votes,
        "distance_inlier_votes": distance_inlier_votes,
        "cluster_assignments": cluster_assignments,
        "cluster_shared_flags": cluster_shared_flags,
        "cluster_inlier_flags": cluster_inlier_flags,
        "assigned_distances": assigned_distances,
        "cluster_summary_rows": cluster_summary_rows,
    }


def _series_harmonization_select_and_rescue(series_records, clustering):
    """Select majority-shared series and transparently preserve patient coverage."""

    shared_votes = np.asarray(clustering["shared_votes"], dtype=np.int64)
    retained = (
        shared_votes >= SERIES_HARMONIZATION_MIN_SHARED_RESOLUTIONS
    )
    rescued = np.zeros(len(series_records), dtype=bool)
    patient_to_indices = defaultdict(list)
    for index, record in enumerate(series_records):
        patient_to_indices[str(record["patient_id"])].append(index)

    rescue_rows = []
    for patient_id in sorted(patient_to_indices):
        indices = np.asarray(patient_to_indices[patient_id], dtype=np.int64)
        if np.any(retained[indices]):
            continue
        if not SERIES_HARMONIZATION_ALLOW_PATIENT_COVERAGE_RESCUE:
            raise RuntimeError(
                "Series harmonization removed every series for patient "
                f"{patient_id}. Enable the audited coverage-rescue policy or "
                "relax only predeclared support thresholds."
            )

        # Prefer the greatest number of shared-cluster votes. Break ties using
        # the lowest mean normalized assigned-centre distance and then series ID.
        candidate_rows = []
        for index in indices:
            normalized_distances = []
            for n_clusters in clustering["cluster_counts"]:
                distances = clustering["assigned_distances"][n_clusters]
                median_distance = max(float(np.median(distances)), 1e-8)
                normalized_distances.append(
                    float(distances[index]) / median_distance
                )
            candidate_rows.append(
                (
                    -int(shared_votes[index]),
                    float(np.mean(normalized_distances)),
                    str(series_records[index]["series_id"]),
                    int(index),
                )
            )
        candidate_rows.sort()
        selected_index = candidate_rows[0][-1]
        retained[selected_index] = True
        rescued[selected_index] = True
        rescue_rows.append(
            {
                "patient_id": patient_id,
                "series_id": series_records[selected_index]["series_id"],
                "shared_resolution_votes": int(
                    shared_votes[selected_index]
                ),
                "reason": "patient_coverage_rescue",
            }
        )

    if len(rescue_rows) > SERIES_HARMONIZATION_MAX_COVERAGE_RESCUES:
        raise RuntimeError(
            "Series harmonization required too many patient-coverage rescues: "
            f"{len(rescue_rows)} > "
            f"{SERIES_HARMONIZATION_MAX_COVERAGE_RESCUES}. This indicates that "
            "the overlap policy is not appropriate for the discovered release."
        )
    return retained, rescued, rescue_rows


def _series_harmonization_write_csv(path, rows, fieldnames=None):
    """Write deterministic CSV output, including an empty header-only case."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        if not fieldnames:
            file.write("")
            return
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def harmonize_cross_class_series(samples, output_dir):
    """Retain cross-class-comparable series and remove atypical proxies.

    Returns
    -------
    filtered_samples, summary
        ``filtered_samples`` preserves the original sample order. ``summary`` is
        also saved to JSON and includes the deterministic selection signature.

    Statistical interpretation
    --------------------------
    Descriptor construction and clustering are outcome-blind. The final shared-
    support test uses the known Normal/Sick folders globally, so this is a
    class-aware cohort-curation sensitivity analysis. It is intended to test
    whether A17 survives removal of class-exclusive series families, not to
    replace external validation.
    """

    if not AUTOMATIC_CROSS_CLASS_SERIES_HARMONIZATION:
        summary = {
            "status": "SKIPPED_DISABLED",
            "enabled": False,
            "n_images_before": int(len(samples)),
            "n_images_after": int(len(samples)),
        }
        return list(samples), summary

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series_records = _series_harmonization_build_series_table(samples)
    descriptor_matrix, descriptor_names, base_audit_rows = (
        _series_harmonization_build_descriptors(series_records)
    )
    scaled, active_columns, robust_center, robust_scale = (
        _series_harmonization_robust_standardize(descriptor_matrix)
    )
    clustering = _series_harmonization_fit_multiresolution_clusters(
        scaled,
        series_records,
    )
    retained, rescued, rescue_rows = (
        _series_harmonization_select_and_rescue(
            series_records,
            clustering,
        )
    )

    retained_series_ids = {
        str(series_records[index]["series_id"])
        for index in np.flatnonzero(retained)
    }
    filtered_samples = [
        sample for sample in samples if str(sample[3]) in retained_series_ids
    ]

    before_patients = sorted({str(sample[2]) for sample in samples})
    after_patients = sorted({str(sample[2]) for sample in filtered_samples})
    before_label_by_patient = {}
    after_label_by_patient = {}
    for sample in samples:
        before_label_by_patient[str(sample[2])] = int(sample[1])
    for sample in filtered_samples:
        after_label_by_patient[str(sample[2])] = int(sample[1])

    if SERIES_HARMONIZATION_REQUIRE_ALL_30_PATIENTS:
        if len(before_patients) != SERIES_HARMONIZATION_EXPECTED_PATIENTS:
            raise RuntimeError(
                "Unexpected pre-harmonization patient count: "
                f"{len(before_patients)} != "
                f"{SERIES_HARMONIZATION_EXPECTED_PATIENTS}."
            )
        normal_before = sum(
            label == 0 for label in before_label_by_patient.values()
        )
        sick_before = sum(
            label == 1 for label in before_label_by_patient.values()
        )
        if (
            normal_before != SERIES_HARMONIZATION_EXPECTED_NORMAL_PATIENTS
            or sick_before != SERIES_HARMONIZATION_EXPECTED_SICK_PATIENTS
        ):
            raise RuntimeError(
                "Unexpected class-specific patient counts before harmonization: "
                f"Normal={normal_before}, Sick={sick_before}."
            )
        if after_patients != before_patients:
            missing = sorted(set(before_patients) - set(after_patients))
            raise RuntimeError(
                "Series harmonization changed the validated Directory_* cohort; "
                f"missing patients={missing}."
            )

    if not filtered_samples:
        raise RuntimeError("Series harmonization retained no image rows.")
    if len({int(sample[1]) for sample in filtered_samples}) != 2:
        raise RuntimeError(
            "Series harmonization did not retain both Normal and Sick classes."
        )

    selection_hasher = hashlib.sha256()
    selection_hasher.update(
        SERIES_HARMONIZATION_DESCRIPTOR_VERSION.encode("utf-8")
    )
    selection_hasher.update(
        SERIES_HARMONIZATION_POLICY_VERSION.encode("utf-8")
    )
    for index, record in enumerate(series_records):
        cluster_tokens = [
            f"k{n_clusters}={int(clustering['cluster_assignments'][n_clusters][index])}"
            for n_clusters in clustering["cluster_counts"]
        ]
        selection_hasher.update(
            (
                f"{record['series_id']}|{int(retained[index])}|"
                f"{int(rescued[index])}|"
                f"{int(clustering['shared_votes'][index])}|"
                + "|".join(cluster_tokens)
                + "\n"
            ).encode("utf-8")
        )
    selection_signature = selection_hasher.hexdigest()

    manifest_rows = []
    blinded_rows = []
    label_key_rows = []
    for index, (record, base_row) in enumerate(
        zip(series_records, base_audit_rows)
    ):
        row = {
            **base_row,
            "class_name": "Normal" if int(record["label"]) == 0 else "Sick",
            "shared_resolution_votes": int(
                clustering["shared_votes"][index]
            ),
            "distance_inlier_resolution_votes": int(
                clustering["distance_inlier_votes"][index]
            ),
            "retained": int(retained[index]),
            "coverage_rescue": int(rescued[index]),
            "decision_reason": (
                "patient_coverage_rescue"
                if rescued[index]
                else (
                    "shared_cross_class_series_family"
                    if retained[index]
                    else "atypical_or_class_exclusive_series_family"
                )
            ),
        }
        for n_clusters in clustering["cluster_counts"]:
            row[f"cluster_k{n_clusters}"] = int(
                clustering["cluster_assignments"][n_clusters][index]
            )
            row[f"shared_cluster_k{n_clusters}"] = int(
                clustering["cluster_shared_flags"][n_clusters][index]
            )
            row[f"distance_inlier_k{n_clusters}"] = int(
                clustering["cluster_inlier_flags"][n_clusters][index]
            )
            row[f"assigned_distance_k{n_clusters}"] = float(
                clustering["assigned_distances"][n_clusters][index]
            )
        manifest_rows.append(row)

        blinded_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"label", "class_name"}
            }
        )
        label_key_rows.append(
            {
                "series_id": record["series_id"],
                "patient_id": record["patient_id"],
                "label": int(record["label"]),
                "class_name": (
                    "Normal" if int(record["label"]) == 0 else "Sick"
                ),
            }
        )

    descriptor_rows = []
    for index, record in enumerate(series_records):
        descriptor_rows.append(
            {
                "series_id": record["series_id"],
                "patient_id": record["patient_id"],
                **{
                    descriptor_names[column]: float(
                        descriptor_matrix[index, column]
                    )
                    for column in range(len(descriptor_names))
                },
            }
        )

    patient_coverage_rows = []
    for patient_id in before_patients:
        original_indices = [
            index
            for index, record in enumerate(series_records)
            if str(record["patient_id"]) == patient_id
        ]
        kept_indices = [index for index in original_indices if retained[index]]
        patient_coverage_rows.append(
            {
                "patient_id": patient_id,
                "label": int(before_label_by_patient[patient_id]),
                "class_name": (
                    "Normal"
                    if int(before_label_by_patient[patient_id]) == 0
                    else "Sick"
                ),
                "series_before": int(len(original_indices)),
                "series_after": int(len(kept_indices)),
                "series_removed": int(
                    len(original_indices) - len(kept_indices)
                ),
                "images_before": int(
                    sum(series_records[index]["n_images"] for index in original_indices)
                ),
                "images_after": int(
                    sum(series_records[index]["n_images"] for index in kept_indices)
                ),
                "coverage_rescues": int(
                    sum(bool(rescued[index]) for index in kept_indices)
                ),
            }
        )

    active_descriptor_names = [
        name
        for name, is_active in zip(descriptor_names, active_columns)
        if bool(is_active)
    ]
    summary = {
        "status": "OK",
        "enabled": True,
        "descriptor_version": SERIES_HARMONIZATION_DESCRIPTOR_VERSION,
        "policy_version": SERIES_HARMONIZATION_POLICY_VERSION,
        "scope": SERIES_HARMONIZATION_SCOPE,
        "statistical_warning": (
            "Shared class support is determined globally after label-free "
            "clustering. Treat downstream metrics as a cohort-harmonization "
            "sensitivity analysis, not independent external validation."
        ),
        "cluster_counts": list(clustering["cluster_counts"]),
        "minimum_shared_resolutions": int(
            SERIES_HARMONIZATION_MIN_SHARED_RESOLUTIONS
        ),
        "minimum_normal_patients_per_cluster": int(
            SERIES_HARMONIZATION_MIN_NORMAL_PATIENTS_PER_CLUSTER
        ),
        "minimum_sick_patients_per_cluster": int(
            SERIES_HARMONIZATION_MIN_SICK_PATIENTS_PER_CLUSTER
        ),
        "n_descriptor_columns": int(descriptor_matrix.shape[1]),
        "n_active_descriptor_columns": int(scaled.shape[1]),
        "active_descriptor_names": active_descriptor_names,
        "n_series_before": int(len(series_records)),
        "n_series_after": int(np.sum(retained)),
        "n_series_removed": int(len(series_records) - np.sum(retained)),
        "n_images_before": int(len(samples)),
        "n_images_after": int(len(filtered_samples)),
        "n_images_removed": int(len(samples) - len(filtered_samples)),
        "retained_image_fraction": float(len(filtered_samples) / len(samples)),
        "n_patients_before": int(len(before_patients)),
        "n_patients_after": int(len(after_patients)),
        "normal_patients_after": int(
            sum(label == 0 for label in after_label_by_patient.values())
        ),
        "sick_patients_after": int(
            sum(label == 1 for label in after_label_by_patient.values())
        ),
        "coverage_rescue_count": int(len(rescue_rows)),
        "coverage_rescues": rescue_rows,
        "selection_signature_sha256": selection_signature,
        "feature_cache_note": (
            "The filtered sample inventory changes the ordinary feature-bank "
            "fingerprint, so unfiltered V7.1 embeddings cannot be reused as if "
            "they represented the harmonized cohort."
        ),
    }

    _series_harmonization_write_csv(
        output_dir / "series_harmonization_manifest.csv",
        manifest_rows,
    )
    _series_harmonization_write_csv(
        output_dir / "series_harmonization_manifest_blinded.csv",
        blinded_rows,
    )
    _series_harmonization_write_csv(
        output_dir / "series_harmonization_label_key.csv",
        label_key_rows,
    )
    _series_harmonization_write_csv(
        output_dir / "series_harmonization_cluster_class_support.csv",
        clustering["cluster_summary_rows"],
    )
    _series_harmonization_write_csv(
        output_dir / "series_harmonization_patient_coverage.csv",
        patient_coverage_rows,
    )
    _series_harmonization_write_csv(
        output_dir / "series_harmonization_descriptors.csv",
        descriptor_rows,
    )
    (output_dir / "series_harmonization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "series_harmonization_configuration.json").write_text(
        json.dumps(
            {
                "descriptor_version": SERIES_HARMONIZATION_DESCRIPTOR_VERSION,
                "policy_version": SERIES_HARMONIZATION_POLICY_VERSION,
                "scope": SERIES_HARMONIZATION_SCOPE,
                "sample_frames_per_series": (
                    SERIES_HARMONIZATION_SAMPLE_FRAMES_PER_SERIES
                ),
                "descriptor_image_size": (
                    SERIES_HARMONIZATION_DESCRIPTOR_IMAGE_SIZE
                ),
                "histogram_bins": (
                    SERIES_HARMONIZATION_INTENSITY_HISTOGRAM_BINS
                ),
                "dct_side": SERIES_HARMONIZATION_DCT_SIDE,
                "cluster_counts": list(
                    SERIES_HARMONIZATION_CLUSTER_COUNTS
                ),
                "kmeans_n_init": SERIES_HARMONIZATION_KMEANS_N_INIT,
                "random_state": SERIES_HARMONIZATION_RANDOM_STATE,
                "minimum_normal_patients_per_cluster": (
                    SERIES_HARMONIZATION_MIN_NORMAL_PATIENTS_PER_CLUSTER
                ),
                "minimum_sick_patients_per_cluster": (
                    SERIES_HARMONIZATION_MIN_SICK_PATIENTS_PER_CLUSTER
                ),
                "minimum_normal_series_per_cluster": (
                    SERIES_HARMONIZATION_MIN_NORMAL_SERIES_PER_CLUSTER
                ),
                "minimum_sick_series_per_cluster": (
                    SERIES_HARMONIZATION_MIN_SICK_SERIES_PER_CLUSTER
                ),
                "minimum_shared_resolutions": (
                    SERIES_HARMONIZATION_MIN_SHARED_RESOLUTIONS
                ),
                "cluster_outlier_mad_multiplier": (
                    SERIES_HARMONIZATION_CLUSTER_OUTLIER_MAD_MULTIPLIER
                ),
                "cluster_outlier_min_quantile": (
                    SERIES_HARMONIZATION_CLUSTER_OUTLIER_MIN_QUANTILE
                ),
                "coverage_rescue_enabled": (
                    SERIES_HARMONIZATION_ALLOW_PATIENT_COVERAGE_RESCUE
                ),
                "maximum_coverage_rescues": (
                    SERIES_HARMONIZATION_MAX_COVERAGE_RESCUES
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("[SERIES HARMONIZATION] Selection summary", flush=True)
    print(
        f"  Series: {len(series_records)} -> {int(np.sum(retained))} "
        f"(removed {len(series_records) - int(np.sum(retained))})",
        flush=True,
    )
    print(
        f"  Images: {len(samples)} -> {len(filtered_samples)} "
        f"({100.0 * len(filtered_samples) / len(samples):.1f}% retained)",
        flush=True,
    )
    print(
        f"  Patients: {len(before_patients)} -> {len(after_patients)}; "
        f"Normal={summary['normal_patients_after']}, "
        f"Sick={summary['sick_patients_after']}",
        flush=True,
    )
    print(
        f"  Coverage rescues: {len(rescue_rows)}; selection signature="
        f"{selection_signature[:16]}...",
        flush=True,
    )
    print(
        "[SERIES HARMONIZATION][STATISTICAL WARNING] Shared cluster support "
        "used both class folders globally. Interpret the filtered run as a "
        "predeclared harmonization sensitivity analysis, not external validation.",
        flush=True,
    )
    return filtered_samples, summary


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
        "fixed_center_crop_fractions": FIXED_CENTER_CROP_FRACTIONS,
        "standardized_content_long_side": STANDARDIZED_CONTENT_LONG_SIDE,
        "fixed_chunk_size": FIXED_CHUNK_SIZE,
        "focused_fixed_heart_fov_fraction": FOCUSED_FIXED_HEART_FOV_FRACTION,
        "focused_hard_support_extra_dilation_kernel": (
            FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL
        ),
        "focused_whole_heart_exclusion_center_fraction": (
            FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION
        ),
        "focused_whole_heart_exclusion_bbox_context_fraction": (
            FOCUSED_WHOLE_HEART_EXCLUSION_BBOX_CONTEXT_FRACTION
        ),
        "focused_fixed_periphery_exclusion_fraction": (
            FOCUSED_FIXED_PERIPHERY_EXCLUSION_FRACTION
        ),
        "focused_region_norm_lower_percentile": (
            FOCUSED_REGION_NORM_LOWER_PERCENTILE
        ),
        "focused_region_norm_upper_percentile": (
            FOCUSED_REGION_NORM_UPPER_PERCENTILE
        ),
        "focused_region_norm_min_pixels": FOCUSED_REGION_NORM_MIN_PIXELS,
        "focused_region_norm_histogram_bins": (
            FOCUSED_REGION_NORM_HISTOGRAM_BINS
        ),
        "focused_region_norm_min_dynamic_range": (
            FOCUSED_REGION_NORM_MIN_DYNAMIC_RANGE
        ),
        "focused_support_intensity_permutation_version": (
            FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION
        ),
        "monai_soft_histogram_bins": MONAI_SOFT_HISTOGRAM_BINS,
        "monai_soft_block_shuffle_grid": MONAI_SOFT_BLOCK_SHUFFLE_GRID,
        "monai_soft_block_shuffle_version": MONAI_SOFT_BLOCK_SHUFFLE_VERSION,
        "monai_canonical_mask_content_fraction": (
            MONAI_CANONICAL_MASK_CONTENT_FRACTION
        ),
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

    requested_qc_branches = required_monai_qc_branches(
        get_enabled_experiments()
    )
    for standardized, requirement_key in ((False, "original"), (True, "standardized")):
        if not requested_qc_branches[requirement_key]:
            continue
        availability = inspect_monai_qc_branch_availability(
            bank, standardized=standardized
        )
        if availability["status"] != "AVAILABLE":
            print(
                "[FEATURE BANK] Cached MONAI-QC branch does not satisfy the "
                f"current registry ({requirement_key}: "
                f"{availability['status']}); a clean rebuild is required.",
                flush=True,
            )
            return None

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


def create_fixed_fraction_center_crop_images(images, crop_fraction):
    """Create a MONAI-independent square center crop at a fixed fraction.

    ``crop_fraction`` is applied to the standardized 224x224 image dimensions
    and is fixed before evaluation. No MONAI mask, gate value, label, patient ID,
    fold assignment, or classifier score influences the crop. The selected
    square is resized back to the full EfficientNet input size.
    """

    crop_fraction = float(crop_fraction)
    if not 0.0 < crop_fraction <= 1.0:
        raise ValueError("crop_fraction must lie in (0,1].")

    height, width = images.shape[-2:]
    crop_side = max(2, int(round(min(height, width) * crop_fraction)))
    center_y = height // 2
    center_x = width // 2
    top = center_y - crop_side // 2
    left = center_x - crop_side // 2
    bottom = top + crop_side
    right = left + crop_side

    output = [
        _resize_single_crop(
            images[index],
            top,
            bottom,
            left,
            right,
            (height, width),
        )
        for index in range(images.shape[0])
    ]
    return torch.stack(output, dim=0)


def apply_zero_background_roi_with_fixed_center_fallback(
    images,
    roi_probability,
    valid_mask,
    fallback_fraction=CENTER_CROP_FALLBACK_FRACTION,
):
    """Use strict zero-background ROI or a fixed center crop when invalid.

    Earlier development variants used a complete standardized-image fallback
    when MONAI gating failed. That was operationally safe but could reintroduce
    full export and border information into an otherwise strict ROI experiment.
    This retained historical-reference helper instead uses a fixed,
    MONAI-independent center crop for invalid masks. It keeps every slice and
    makes fallback content deterministic and label-blind.
    """

    strict_roi = apply_confidence_gated_soft_roi(
        images,
        roi_probability,
        valid_mask,
        background_weight=0.0,
    )
    fixed_center = create_fixed_fraction_center_crop_images(
        images,
        fallback_fraction,
    )
    selector = valid_mask.view(-1, 1, 1, 1)
    return torch.where(selector, strict_roi, fixed_center)


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



def _focused_fixed_square_bounds(height, width, center_y, center_x, fraction):
    """Return a fixed-size in-bounds square around a floating-point center.

    The square side depends only on the predeclared fraction and image size. It
    does not depend on class labels, fold membership, mask area or model score.
    """

    height = int(height)
    width = int(width)
    fraction = float(fraction)
    if height <= 0 or width <= 0:
        raise ValueError("Square bounds require positive image dimensions.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Square fraction must lie in (0,1].")

    side = max(2, int(round(min(height, width) * fraction)))
    side = min(side, height, width)
    top = int(round(float(center_y) - side / 2.0))
    left = int(round(float(center_x) - side / 2.0))
    top = int(np.clip(top, 0, height - side))
    left = int(np.clip(left, 0, width - side))
    return top, top + side, left, left + side


def _focused_mask_centroid_or_image_center(mask_2d, use_mask):
    """Return a hard-mask centroid or the deterministic geometric center."""

    height, width = mask_2d.shape[-2:]
    if use_mask:
        positions = torch.nonzero(mask_2d > 0.5, as_tuple=False)
        if positions.numel() > 0:
            return (
                float(positions[:, 0].float().mean().item()),
                float(positions[:, 1].float().mean().item()),
            )
    return (height - 1) / 2.0, (width - 1) / 2.0


def _focused_robust_scale_visible_regions(images, visible_masks):
    """Batched robust scaling calculated only from final visible pixels.

    ``images`` contains the raw standardized uint8/255 view before global
    percentile scaling. A masked 256-bin histogram estimates the configured
    lower and upper quantiles separately for each image. Pixels outside the
    final mask are then set to zero. This ordering prevents excluded cardiac
    pixels from setting the contrast of an outside-region control and prevents
    outside pixels from setting the contrast of a cardiac candidate.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Region scaling expects [B,3,H,W], got {tuple(images.shape)}."
        )
    if visible_masks.ndim == 3:
        visible_masks = visible_masks.unsqueeze(1)
    if visible_masks.ndim != 4 or visible_masks.shape[1] != 1:
        raise ValueError("visible_masks must have shape [B,1,H,W].")
    if images.shape[0] != visible_masks.shape[0] or tuple(
        images.shape[-2:]
    ) != tuple(visible_masks.shape[-2:]):
        raise ValueError("Visible masks and image tensors are not aligned.")

    bins = int(FOCUSED_REGION_NORM_HISTOGRAM_BINS)
    mask = visible_masks > 0.5
    mask_flat = mask[:, 0].reshape(images.shape[0], -1)
    counts = mask_flat.sum(dim=1).to(torch.long)

    # The input channels are identical grayscale copies, so one channel is
    # sufficient for percentile estimation and limits are broadcast to all three.
    values = images[:, 0].float().clamp(0.0, 1.0)
    bin_indices = torch.round(values * float(bins - 1)).to(torch.long)
    bin_indices = bin_indices.clamp_(0, bins - 1).reshape(images.shape[0], -1)

    histogram = torch.zeros(
        images.shape[0], bins, device=images.device, dtype=torch.float32
    )
    histogram.scatter_add_(
        1, bin_indices, mask_flat.to(torch.float32)
    )
    cumulative = torch.cumsum(histogram, dim=1)

    safe_counts = counts.clamp_min(1)
    lower_rank = (
        torch.floor(
            (FOCUSED_REGION_NORM_LOWER_PERCENTILE / 100.0)
            * (safe_counts - 1).to(torch.float32)
        ).to(torch.long)
        + 1
    )
    upper_rank = (
        torch.floor(
            (FOCUSED_REGION_NORM_UPPER_PERCENTILE / 100.0)
            * (safe_counts - 1).to(torch.float32)
        ).to(torch.long)
        + 1
    )
    lower_bins = (
        cumulative >= lower_rank.unsqueeze(1).to(cumulative.dtype)
    ).to(torch.int64).argmax(dim=1)
    upper_bins = (
        cumulative >= upper_rank.unsqueeze(1).to(cumulative.dtype)
    ).to(torch.int64).argmax(dim=1)

    lower = lower_bins.to(torch.float32) / float(bins - 1)
    upper = upper_bins.to(torch.float32) / float(bins - 1)
    valid_rows = (
        (counts >= int(FOCUSED_REGION_NORM_MIN_PIXELS))
        & torch.isfinite(lower)
        & torch.isfinite(upper)
        & (upper > lower)
    )

    lower = lower.view(-1, 1, 1, 1)
    upper = upper.view(-1, 1, 1, 1)
    denominator = (upper - lower).clamp_min(
        float(FOCUSED_REGION_NORM_MIN_DYNAMIC_RANGE)
    )
    scaled = ((images.float() - lower) / denominator).clamp(0.0, 1.0)
    scaled = scaled * mask.to(scaled.dtype)
    scaled = scaled * valid_rows.view(-1, 1, 1, 1).to(scaled.dtype)
    return scaled.to(images.dtype)


def create_focused_heart_centered_fixed_fov_images(
    raw_images,
    hard_mask,
    valid_mask,
    content_mask,
    crop_fraction=FOCUSED_FIXED_HEART_FOV_FRACTION,
):
    """Create A16: fixed heart-centered FOV without a visible mask silhouette.

    The square has a fixed side length and is centered on the valid hard-mask
    centroid; gate-invalid slices use the image center. It is normalized only
    from retained non-padding pixels inside the square and resized directly to
    the encoder size. Because no zero-background support outline is preserved,
    A16 is a useful shape-suppressed localization reference for future models.
    """

    if raw_images.ndim != 4 or raw_images.shape[1] != 3:
        raise ValueError("raw_images must have shape [B,3,H,W].")
    height, width = raw_images.shape[-2:]
    crop_bounds = []
    visible = torch.zeros(
        raw_images.shape[0], 1, height, width,
        device=raw_images.device, dtype=raw_images.dtype,
    )
    for index in range(raw_images.shape[0]):
        center_y, center_x = _focused_mask_centroid_or_image_center(
            hard_mask[index, 0], bool(valid_mask[index].item())
        )
        top, bottom, left, right = _focused_fixed_square_bounds(
            height, width, center_y, center_x, crop_fraction
        )
        crop_bounds.append((top, bottom, left, right))
        visible[index, 0, top:bottom, left:right] = 1.0

    visible *= content_mask
    scaled_full = _focused_robust_scale_visible_regions(raw_images, visible)
    output = []
    for index, (top, bottom, left, right) in enumerate(crop_bounds):
        crop = scaled_full[index, :, top:bottom, left:right]
        output.append(
            F.interpolate(
                crop.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return torch.stack(output, dim=0)


def create_focused_a17_exact_support_mask(hard_mask, valid_mask, content_mask):
    """Return the exact final support shared by A17, C31, C32 and C33.

    Valid slices use the standardized MONAI hard mask after one fixed additional
    dilation. Invalid slices use the fixed central-square fallback. The result
    is intersected with retained image content so square-canvas padding is never
    part of the candidate or its exact matched controls.
    """

    if hard_mask.ndim != 4 or hard_mask.shape[1] != 1:
        raise ValueError("hard_mask must have shape [B,1,H,W].")
    if content_mask.shape != hard_mask.shape:
        raise ValueError("hard_mask and content_mask must have equal shapes.")
    if valid_mask.ndim != 1 or valid_mask.shape[0] != hard_mask.shape[0]:
        raise ValueError("valid_mask must contain one value per image.")

    kernel = int(FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL)
    support = (hard_mask > 0.5).to(hard_mask.dtype)
    support = F.max_pool2d(
        support, kernel_size=kernel, stride=1, padding=kernel // 2
    )
    batch_size, _, height, width = support.shape
    final_support = torch.zeros_like(support)
    for index in range(batch_size):
        if bool(valid_mask[index].item()) and bool(
            torch.any(support[index, 0] > 0.5).item()
        ):
            final_support[index, 0] = support[index, 0]
        else:
            top, bottom, left, right = _focused_fixed_square_bounds(
                height,
                width,
                (height - 1) / 2.0,
                (width - 1) / 2.0,
                FOCUSED_FIXED_HEART_FOV_FRACTION,
            )
            final_support[index, 0, top:bottom, left:right] = 1.0
    return (
        final_support * (content_mask > 0.5).to(final_support.dtype)
    ).clamp(0.0, 1.0)


def create_focused_a17_images(
    raw_images,
    hard_mask,
    valid_mask,
    content_mask,
    exact_support=None,
):
    """Create A17 from raw MRI intensities and the exact binary support."""

    if exact_support is None:
        exact_support = create_focused_a17_exact_support_mask(
            hard_mask, valid_mask, content_mask
        )
    return _focused_robust_scale_visible_regions(raw_images, exact_support)


def create_focused_a17_support_only_images(exact_support):
    """Create C31 by removing all MRI intensities from A17's support."""

    if exact_support.ndim != 4 or exact_support.shape[1] != 1:
        raise ValueError("exact_support must have shape [B,1,H,W].")
    return exact_support.clamp(0.0, 1.0).repeat(1, 3, 1, 1)


def _focused_hash_coprime_affine_parameters(pixel_hash, n_values):
    """Return a deterministic non-identity affine permutation of n positions."""

    n_values = int(n_values)
    if n_values <= 1:
        return 1, 0
    token = str(pixel_hash).strip().lower()
    if len(token) < 32 or any(c not in "0123456789abcdef" for c in token[:32]):
        raise ValueError(f"Invalid decoded-pixel SHA-256 value: {pixel_hash!r}.")

    candidate = 2 + (int(token[:16], 16) % max(1, n_values - 2))
    candidate %= n_values
    if candidate == 0:
        candidate = 1
    start = candidate
    while math.gcd(candidate, n_values) != 1:
        candidate += 1
        if candidate >= n_values:
            candidate = 1
        if candidate == start:
            raise RuntimeError("Could not construct a coprime permutation.")
    offset = int(token[16:32], 16) % n_values
    if candidate == 1 and offset == 0:
        offset = 1
    return int(candidate), int(offset)


def create_focused_a17_shuffled_intensity_images(
    a17_images,
    exact_support,
    decoded_pixel_hashes,
):
    """Create C32: preserve support and intensity multiset, destroy position.

    A hash-derived affine bijection permutes normalized grayscale values only
    among visible support coordinates. Each value is used exactly once, the
    three channels remain identical, and pixels outside the support stay zero.
    """

    if a17_images.ndim != 4 or a17_images.shape[1] != 3:
        raise ValueError("a17_images must have shape [B,3,H,W].")
    if exact_support.ndim != 4 or exact_support.shape[1] != 1:
        raise ValueError("exact_support must have shape [B,1,H,W].")
    if len(decoded_pixel_hashes) != a17_images.shape[0]:
        raise ValueError("One decoded-pixel hash is required per image.")

    output = torch.zeros_like(a17_images)
    for index in range(a17_images.shape[0]):
        positions = torch.nonzero(
            exact_support[index, 0].reshape(-1) > 0.5,
            as_tuple=False,
        ).reshape(-1)
        n_values = int(positions.numel())
        if n_values == 0:
            continue
        source_values = a17_images[index, 0].reshape(-1)[positions]
        multiplier, offset = _focused_hash_coprime_affine_parameters(
            decoded_pixel_hashes[index], n_values
        )
        destination_rank = torch.arange(
            n_values, device=a17_images.device, dtype=torch.long
        )
        source_rank = (destination_rank * multiplier + offset) % n_values
        shuffled = source_values[source_rank]
        output[index].reshape(3, -1)[:, positions] = shuffled.unsqueeze(0)
    return output


def create_focused_a17_complement_images(
    raw_images,
    exact_support,
    content_mask,
):
    """Create C33 from the exact independently normalized A17 complement."""

    if exact_support.shape != content_mask.shape:
        raise ValueError("exact_support and content_mask must align.")
    visible = (
        (content_mask > 0.5) & (exact_support <= 0.5)
    ).to(raw_images.dtype)
    return _focused_robust_scale_visible_regions(raw_images, visible)


def _focused_fixed_central_square_mask(
    batch_size, height, width, fraction, device, dtype
):
    """Return one predeclared central square mask for the whole batch."""

    top, bottom, left, right = _focused_fixed_square_bounds(
        height,
        width,
        (height - 1) / 2.0,
        (width - 1) / 2.0,
        fraction,
    )
    mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    mask[:, :, top:bottom, left:right] = 1.0
    return mask


def create_focused_conservative_whole_heart_exclusion(hard_mask, valid_mask):
    """Build the conservative exclusion proxy used by C28.

    It is the union of a fixed central square, a square centered on the valid
    hard-mask centroid, and a substantially expanded ventricular bounding box.
    It is intentionally described as a proxy because the ventricular segmenter
    does not produce ground-truth whole-heart masks for all views.
    """

    batch_size, _, height, width = hard_mask.shape
    exclusion = _focused_fixed_central_square_mask(
        batch_size,
        height,
        width,
        FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION,
        hard_mask.device,
        hard_mask.dtype,
    )
    for index in range(batch_size):
        if not bool(valid_mask[index].item()):
            continue
        box = _hard_mask_bounding_box(hard_mask[index, 0])
        if box is None:
            continue
        center_y, center_x = _focused_mask_centroid_or_image_center(
            hard_mask[index, 0], True
        )
        top, bottom, left, right = _focused_fixed_square_bounds(
            height,
            width,
            center_y,
            center_x,
            FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION,
        )
        exclusion[index, 0, top:bottom, left:right] = 1.0

        box_top, box_bottom, box_left, box_right = box
        box_height = box_bottom - box_top
        box_width = box_right - box_left
        margin = int(
            round(
                max(box_height, box_width)
                * FOCUSED_WHOLE_HEART_EXCLUSION_BBOX_CONTEXT_FRACTION
            )
        )
        box_top = max(0, box_top - margin)
        box_bottom = min(height, box_bottom + margin)
        box_left = max(0, box_left - margin)
        box_right = min(width, box_right + margin)
        exclusion[index, 0, box_top:box_bottom, box_left:box_right] = 1.0
    return exclusion.clamp(0.0, 1.0)


def create_focused_outside_whole_heart_images(
    raw_images, hard_mask, valid_mask, content_mask
):
    """Create C28 by retaining only a conservative extracardiac proxy."""

    exclusion = create_focused_conservative_whole_heart_exclusion(
        hard_mask, valid_mask
    )
    visible = (1.0 - exclusion) * content_mask
    return _focused_robust_scale_visible_regions(raw_images, visible)


def create_focused_fixed_periphery_images(raw_images, content_mask):
    """Create C29: MONAI-independent fixed periphery with local scaling."""

    batch_size, _, height, width = raw_images.shape
    exclusion = _focused_fixed_central_square_mask(
        batch_size,
        height,
        width,
        FOCUSED_FIXED_PERIPHERY_EXCLUSION_FRACTION,
        raw_images.device,
        raw_images.dtype,
    )
    visible = (1.0 - exclusion) * content_mask
    return _focused_robust_scale_visible_regions(raw_images, visible)


def validate_focused_transform_contract():
    """Run a fast synthetic contract test for the focused A17 control family.

    This test executes before dataset scanning or model loading. It verifies the
    scientific invariants that make C31-C33 interpretable: one shared support,
    exact intensity-multiset preservation in C32, non-overlap of A17 and C33,
    and independence of each regional normalization from excluded pixels.
    """

    height = width = 64
    batch_size = 3
    base = torch.linspace(
        0.0,
        1.0,
        steps=height * width,
        dtype=torch.float32,
    ).reshape(1, 1, height, width)
    raw_images = base.repeat(batch_size, 3, 1, 1)
    raw_images[1] = torch.flip(raw_images[1], dims=[2])
    raw_images[2] = torch.roll(raw_images[2], shifts=7, dims=2)

    hard_mask = torch.zeros(batch_size, 1, height, width)
    hard_mask[0, 0, 18:38, 20:42] = 1.0
    hard_mask[1, 0, 24:45, 10:31] = 1.0
    # The third row deliberately has no trusted mask and exercises the fixed
    # central fallback used by A17 and every exact matched control.
    valid_mask = torch.tensor([True, True, False], dtype=torch.bool)
    content_mask = torch.ones(batch_size, 1, height, width)
    content_mask[:, :, :3, :] = 0.0
    content_mask[:, :, -3:, :] = 0.0
    content_mask[:, :, :, :4] = 0.0
    content_mask[:, :, :, -4:] = 0.0

    exact_support = create_focused_a17_exact_support_mask(
        hard_mask,
        valid_mask,
        content_mask,
    )
    a17 = create_focused_a17_images(
        raw_images,
        hard_mask,
        valid_mask,
        content_mask,
        exact_support=exact_support,
    )
    c31 = create_focused_a17_support_only_images(exact_support)
    hashes = [
        hashlib.sha256(f"focused-contract-{index}".encode("utf-8")).hexdigest()
        for index in range(batch_size)
    ]
    c32 = create_focused_a17_shuffled_intensity_images(
        a17,
        exact_support,
        hashes,
    )
    c32_repeat = create_focused_a17_shuffled_intensity_images(
        a17,
        exact_support,
        hashes,
    )
    c33 = create_focused_a17_complement_images(
        raw_images,
        exact_support,
        content_mask,
    )

    support_three_channels = exact_support.repeat(1, 3, 1, 1)
    if not torch.equal(c31[:, 0:1] > 0.5, exact_support > 0.5):
        raise RuntimeError("C31 does not reproduce A17's exact support.")
    if torch.count_nonzero(a17 * (1.0 - support_three_channels)).item() != 0:
        raise RuntimeError("A17 contains non-zero values outside its support.")
    if torch.count_nonzero(c32 * (1.0 - support_three_channels)).item() != 0:
        raise RuntimeError("C32 contains non-zero values outside A17 support.")
    if torch.count_nonzero(c33 * support_three_channels).item() != 0:
        raise RuntimeError("C33 overlaps A17's exact support.")
    if not torch.equal(c32, c32_repeat):
        raise RuntimeError("C32's hash-derived permutation is not deterministic.")

    for index in range(batch_size):
        positions = exact_support[index, 0].reshape(-1) > 0.5
        original_values = torch.sort(
            a17[index, 0].reshape(-1)[positions]
        ).values
        shuffled_values = torch.sort(
            c32[index, 0].reshape(-1)[positions]
        ).values
        if not torch.equal(original_values, shuffled_values):
            raise RuntimeError(
                "C32 does not preserve the exact visible intensity multiset."
            )
        if (
            torch.unique(original_values).numel() > 1
            and torch.equal(
                a17[index, 0].reshape(-1)[positions],
                c32[index, 0].reshape(-1)[positions],
            )
        ):
            raise RuntimeError("C32 failed to destroy spatial intensity order.")

    # Alter only excluded A17 pixels: A17 must remain unchanged.
    outside_support = (
        (content_mask > 0.5) & (exact_support <= 0.5)
    ).repeat(1, 3, 1, 1)
    outside_changed = raw_images.clone()
    outside_changed[outside_support] = 1.0 - outside_changed[outside_support]
    a17_after_outside_change = create_focused_a17_images(
        outside_changed,
        hard_mask,
        valid_mask,
        content_mask,
        exact_support=exact_support,
    )
    if not torch.equal(a17, a17_after_outside_change):
        raise RuntimeError("Excluded complement pixels changed A17 normalization.")

    # Alter only excluded C33 pixels: C33 must remain unchanged.
    inside_changed = raw_images.clone()
    inside_selector = support_three_channels > 0.5
    inside_changed[inside_selector] = 1.0 - inside_changed[inside_selector]
    c33_after_inside_change = create_focused_a17_complement_images(
        inside_changed,
        exact_support,
        content_mask,
    )
    if not torch.equal(c33, c33_after_inside_change):
        raise RuntimeError("Excluded A17 pixels changed C33 normalization.")

    content_boolean = content_mask > 0.5
    complement_boolean = content_boolean & (exact_support <= 0.5)
    if not torch.equal(
        (exact_support > 0.5) | complement_boolean,
        content_boolean,
    ):
        raise RuntimeError("A17 support and C33 complement do not partition content.")

    return {
        "status": "PASS",
        "batch_size": int(batch_size),
        "support_pixel_counts": [
            int(value)
            for value in exact_support.flatten(1).sum(dim=1).tolist()
        ],
        "permutation_version": FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION,
    }


def create_soft_monai_mask_only_images(roi_probability, valid_mask):
    """Encode only the standardized soft MONAI map as a three-channel image.

    MRI intensities are deliberately excluded. Gate-valid slices retain the
    dilated soft probability map; gate-invalid slices become all zero so the
    control cannot receive the full image through a fallback branch. A high
    patient-level AUC would indicate that mask shape, position, extent, or
    confidence alone is associated with the released Normal/Sick label.
    """

    if roi_probability.ndim != 4 or roi_probability.shape[1] != 1:
        raise ValueError(
            "Soft MONAI mask-only input must have shape [B,1,H,W]."
        )
    selector = valid_mask.view(-1, 1, 1, 1).to(roi_probability.dtype)
    soft_map = roi_probability.clamp(0.0, 1.0) * selector
    return soft_map.repeat(1, 3, 1, 1)


def create_hard_monai_mask_only_images(hard_mask, valid_mask):
    """Encode only the dilated standardized hard MONAI mask.

    This control removes both MRI intensity and soft confidence. It preserves
    only mask morphology and spatial position for gate-valid slices. Invalid
    slices are all zero, which keeps gate failure from exposing full-image
    anatomy or export style.
    """

    if hard_mask.ndim != 4 or hard_mask.shape[1] != 1:
        raise ValueError(
            "Hard MONAI mask-only input must have shape [B,1,H,W]."
        )
    selector = valid_mask.view(-1, 1, 1, 1).to(hard_mask.dtype)
    binary_mask = (hard_mask > 0.5).to(hard_mask.dtype) * selector
    return binary_mask.repeat(1, 3, 1, 1)


def create_monai_bbox_mask_only_images(hard_mask, valid_mask):
    """Encode only MONAI bounding-box geometry as a binary image.

    One filled rectangle is drawn around each gate-valid dilated hard mask. MRI
    intensities and within-box mask morphology are discarded. The experiment
    therefore tests whether box location and extent alone encode cohort
    provenance or class. Invalid/empty masks yield an all-zero image.
    """

    if hard_mask.ndim != 4 or hard_mask.shape[1] != 1:
        raise ValueError(
            "MONAI bounding-box mask input must have shape [B,1,H,W]."
        )

    batch_size, _, height, width = hard_mask.shape
    output = torch.zeros(
        batch_size,
        3,
        height,
        width,
        device=hard_mask.device,
        dtype=hard_mask.dtype,
    )
    for index in range(batch_size):
        if not bool(valid_mask[index].item()):
            continue
        box = _hard_mask_bounding_box(hard_mask[index, 0])
        if box is None:
            continue
        top, bottom, left, right = box
        output[index, :, top:bottom, left:right] = 1.0
    return output



def create_soft_monai_histogram_only_images(
    roi_probability,
    valid_mask,
    histogram_bins=MONAI_SOFT_HISTOGRAM_BINS,
):
    """Encode only the soft-map probability distribution as a CDF image.

    This segmentation-derived control deliberately removes every original
    spatial coordinate. For each gate-valid slice, the dilated MONAI
    probabilities are summarized into a fixed-bin histogram and cumulative
    distribution function (CDF). The one-dimensional CDF is then repeated over
    image rows and copied to three channels so the same frozen EfficientNet
    encoder can be used. Gate-invalid slices remain all zero, matching the
    archived soft-map control contract from the broad development suite.

    The resulting image is synthetic and must not be interpreted anatomically.
    Its purpose is to test whether the empirical confidence/probability
    distribution alone can explain the high performance of the intact soft-map
    representation. Original mask location, connected components, contour shape,
    and local spatial adjacency are absent.
    """

    if roi_probability.ndim != 4 or roi_probability.shape[1] != 1:
        raise ValueError(
            "Soft MONAI histogram input must have shape [B,1,H,W]."
        )
    histogram_bins = int(histogram_bins)
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2.")

    batch_size, _, height, width = roi_probability.shape
    output = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=roi_probability.device,
        dtype=roi_probability.dtype,
    )

    for index in range(batch_size):
        if not bool(valid_mask[index].item()):
            continue

        values = roi_probability[index, 0].float().clamp(0.0, 1.0)
        histogram = torch.histc(
            values,
            bins=histogram_bins,
            min=0.0,
            max=1.0,
        )
        total = histogram.sum()
        if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
            continue

        cdf = torch.cumsum(histogram, dim=0) / total
        cdf_line = F.interpolate(
            cdf.view(1, 1, histogram_bins),
            size=width,
            mode="linear",
            align_corners=False,
        ).view(width)
        output[index, 0] = cdf_line.view(1, width).expand(height, width)

    return output.repeat(1, 3, 1, 1).to(roi_probability.dtype)


def create_block_shuffled_soft_monai_mask_only_images(
    roi_probability,
    valid_mask,
    decoded_pixel_hashes,
    block_grid=MONAI_SOFT_BLOCK_SHUFFLE_GRID,
):
    """Destroy global soft-mask shape/location while preserving local blocks.

    The gate-valid 224x224 probability map is divided into a fixed block grid.
    Blocks are permuted independently for each image using a deterministic seed
    derived from the image's exact decoded-pixel SHA-256. No label, patient ID,
    series ID, fold assignment, or model score contributes to the permutation.

    This preserves:
        - the complete soft-probability histogram;
        - all pixel values;
        - local texture within each block.

    It destroys:
        - the original global mask contour;
        - original absolute location;
        - long-range spatial relationships between blocks.

    The control is complementary to the CDF/histogram-only representation. A
    high score here but a lower histogram-only score suggests that local
    within-block confidence texture contributes beyond the marginal
    distribution. Gate-invalid slices remain all zero.
    """

    if roi_probability.ndim != 4 or roi_probability.shape[1] != 1:
        raise ValueError(
            "Block-shuffled soft MONAI input must have shape [B,1,H,W]."
        )

    batch_size, _, height, width = roi_probability.shape
    block_grid = int(block_grid)
    if block_grid <= 1:
        raise ValueError("block_grid must be greater than 1.")
    if height % block_grid != 0 or width % block_grid != 0:
        raise ValueError(
            f"Soft-map size {(height, width)} must be divisible by block_grid="
            f"{block_grid}."
        )
    if len(decoded_pixel_hashes) != batch_size:
        raise ValueError(
            "decoded_pixel_hashes must contain one value per soft MONAI map."
        )

    selector = valid_mask.view(-1, 1, 1, 1).to(roi_probability.dtype)
    soft_maps = roi_probability.clamp(0.0, 1.0) * selector
    output = torch.zeros_like(soft_maps)
    block_height = height // block_grid
    block_width = width // block_grid
    n_blocks = block_grid * block_grid

    for index in range(batch_size):
        if not bool(valid_mask[index].item()):
            continue

        seed_material = (
            f"{MONAI_SOFT_BLOCK_SHUFFLE_VERSION}|"
            f"{str(decoded_pixel_hashes[index])}"
        ).encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16) % (
            2 ** 32
        )
        permutation = np.random.default_rng(seed).permutation(n_blocks)
        permutation_tensor = torch.as_tensor(
            permutation,
            dtype=torch.long,
            device=roi_probability.device,
        )

        blocks = (
            soft_maps[index, 0]
            .reshape(block_grid, block_height, block_grid, block_width)
            .permute(0, 2, 1, 3)
            .reshape(n_blocks, block_height, block_width)
        )
        shuffled_blocks = blocks.index_select(0, permutation_tensor)
        shuffled_map = (
            shuffled_blocks
            .reshape(block_grid, block_grid, block_height, block_width)
            .permute(0, 2, 1, 3)
            .reshape(height, width)
        )
        output[index, 0] = shuffled_map

    return output.repeat(1, 3, 1, 1)


def _create_canonicalized_monai_map_only_images(
    source_map,
    hard_mask,
    valid_mask,
    *,
    binary,
    content_fraction=MONAI_CANONICAL_MASK_CONTENT_FRACTION,
):
    """Canonicalize one MONAI-derived map to fixed centered location and scale.

    The gate-valid hard mask defines the source bounding box. The corresponding
    hard or soft map is cropped to that box, square-padded without anisotropic
    stretching, resized to one predeclared target side, and centered in the
    original classifier canvas. This removes absolute position and original
    extent while preserving relative contour geometry; the soft variant also
    preserves within-mask confidence gradients. Invalid/empty masks produce an
    all-zero image.
    """

    if source_map.ndim != 4 or source_map.shape[1] != 1:
        raise ValueError("Canonical source_map must have shape [B,1,H,W].")
    if hard_mask.shape != source_map.shape:
        raise ValueError(
            "Canonical source_map and hard_mask must have identical shapes."
        )

    content_fraction = float(content_fraction)
    if not 0.0 < content_fraction <= 1.0:
        raise ValueError("content_fraction must lie in (0,1].")

    batch_size, _, height, width = source_map.shape
    target_side = max(
        2,
        int(round(min(height, width) * content_fraction)),
    )
    target_top = (height - target_side) // 2
    target_left = (width - target_side) // 2
    output = torch.zeros_like(source_map)

    for index in range(batch_size):
        if not bool(valid_mask[index].item()):
            continue
        box = _hard_mask_bounding_box(hard_mask[index, 0])
        if box is None:
            continue

        top, bottom, left, right = box
        crop = source_map[index:index + 1, :, top:bottom, left:right]
        if crop.numel() == 0:
            continue
        if binary:
            crop = (crop > 0.5).to(source_map.dtype)
        else:
            crop = crop.clamp(0.0, 1.0)

        crop_height = int(crop.shape[-2])
        crop_width = int(crop.shape[-1])
        square_side = max(crop_height, crop_width)
        square = torch.zeros(
            1,
            1,
            square_side,
            square_side,
            device=source_map.device,
            dtype=source_map.dtype,
        )
        square_top = (square_side - crop_height) // 2
        square_left = (square_side - crop_width) // 2
        square[
            :,
            :,
            square_top:square_top + crop_height,
            square_left:square_left + crop_width,
        ] = crop

        if binary:
            resized = F.interpolate(
                square,
                size=(target_side, target_side),
                mode="nearest",
            )
            resized = (resized > 0.5).to(source_map.dtype)
        else:
            resized = F.interpolate(
                square,
                size=(target_side, target_side),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

        output[
            index:index + 1,
            :,
            target_top:target_top + target_side,
            target_left:target_left + target_side,
        ] = resized

    return output.repeat(1, 3, 1, 1)


def create_canonicalized_hard_monai_mask_only_images(
    hard_mask,
    valid_mask,
):
    """Retain relative hard-mask shape while removing location and scale."""

    return _create_canonicalized_monai_map_only_images(
        hard_mask,
        hard_mask,
        valid_mask,
        binary=True,
    )


def create_canonicalized_soft_monai_mask_only_images(
    roi_probability,
    hard_mask,
    valid_mask,
):
    """Retain canonical soft morphology/confidence without absolute geometry."""

    return _create_canonicalized_monai_map_only_images(
        roi_probability,
        hard_mask,
        valid_mask,
        binary=False,
    )


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
    A 63,425 x 1,280 float32 matrix is roughly 310 MiB. The focused suite
    creates nine such representations, so keeping them plus
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

    # MONAI is loaded only when an enabled image view or a declared tabular
    # MONAI-QC control requires that preprocessing branch. The focused V7.1
    # panel needs only the standardized branch; the original branch therefore
    # remains intentionally uncomputed and is reported as SKIPPED downstream.
    monai_requirements = required_monai_qc_branches(
        get_enabled_experiments()
    )
    need_original_monai = bool(monai_requirements["original"])
    need_standardized_monai = bool(monai_requirements["standardized"])
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
    # label-blind cropping and fixed content geometry can legitimately change
    # masks. MONAI itself receives min-max scaling, not the robust classifier view.
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
        f"MONAI_required={need_monai}, "
        f"original_MONAI_QC={need_original_monai}, "
        f"standardized_MONAI_QC={need_standardized_monai}, "
        f"device={DEVICE}",
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
                standardized_raw_images,
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
            standardized_raw_images = standardized_raw_images.to(
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
            standardized_content_mask = (
                1.0 - detected_padding_images[:, 0:1]
            ).clamp(0.0, 1.0)
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
            if "standardized_fixed_center_50" in required_modes:
                variants["standardized_fixed_center_50"] = (
                    create_fixed_fraction_center_crop_images(
                        standardized_images,
                        FIXED_CENTER_CROP_FRACTIONS[0],
                    )
                )
            if "standardized_fixed_center_60" in required_modes:
                variants["standardized_fixed_center_60"] = (
                    create_fixed_fraction_center_crop_images(
                        standardized_images,
                        FIXED_CENTER_CROP_FRACTIONS[1],
                    )
                )
            if "standardized_fixed_center_70" in required_modes:
                variants["standardized_fixed_center_70"] = (
                    create_fixed_fraction_center_crop_images(
                        standardized_images,
                        FIXED_CENTER_CROP_FRACTIONS[2],
                    )
                )
            if "standardized_roi_zero_bg_center_fallback" in required_modes:
                variants["standardized_roi_zero_bg_center_fallback"] = (
                    apply_zero_background_roi_with_fixed_center_fallback(
                        standardized_images,
                        standardized_roi_probability,
                        standardized_valid_mask,
                        fallback_fraction=CENTER_CROP_FALLBACK_FRACTION,
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
            if (
                "standardized_heart_centered_fixed_fov_region_norm"
                in required_modes
            ):
                variants[
                    "standardized_heart_centered_fixed_fov_region_norm"
                ] = create_focused_heart_centered_fixed_fov_images(
                    standardized_raw_images,
                    standardized_hard_mask,
                    standardized_valid_mask,
                    standardized_content_mask,
                )

            focused_a17_modes = {
                "standardized_hard_support_region_norm",
                "standardized_a17_exact_support_mask_only",
                "standardized_a17_support_intensity_affine_shuffled",
                "standardized_a17_exact_support_complement_region_norm",
            }
            if any(mode in required_modes for mode in focused_a17_modes):
                exact_support = create_focused_a17_exact_support_mask(
                    standardized_hard_mask,
                    standardized_valid_mask,
                    standardized_content_mask,
                )
                a17_images = None
                if (
                    "standardized_hard_support_region_norm" in required_modes
                    or "standardized_a17_support_intensity_affine_shuffled"
                    in required_modes
                ):
                    a17_images = create_focused_a17_images(
                        standardized_raw_images,
                        standardized_hard_mask,
                        standardized_valid_mask,
                        standardized_content_mask,
                        exact_support=exact_support,
                    )
                if "standardized_hard_support_region_norm" in required_modes:
                    variants["standardized_hard_support_region_norm"] = a17_images
                if "standardized_a17_exact_support_mask_only" in required_modes:
                    variants["standardized_a17_exact_support_mask_only"] = (
                        create_focused_a17_support_only_images(exact_support)
                    )
                if (
                    "standardized_a17_support_intensity_affine_shuffled"
                    in required_modes
                ):
                    variants[
                        "standardized_a17_support_intensity_affine_shuffled"
                    ] = create_focused_a17_shuffled_intensity_images(
                        a17_images, exact_support, decoded_pixel_hashes
                    )
                if (
                    "standardized_a17_exact_support_complement_region_norm"
                    in required_modes
                ):
                    variants[
                        "standardized_a17_exact_support_complement_region_norm"
                    ] = create_focused_a17_complement_images(
                        standardized_raw_images,
                        exact_support,
                        standardized_content_mask,
                    )

            if "standardized_outside_whole_heart_region_norm" in required_modes:
                variants["standardized_outside_whole_heart_region_norm"] = (
                    create_focused_outside_whole_heart_images(
                        standardized_raw_images,
                        standardized_hard_mask,
                        standardized_valid_mask,
                        standardized_content_mask,
                    )
                )
            if "standardized_fixed_periphery_region_norm" in required_modes:
                variants["standardized_fixed_periphery_region_norm"] = (
                    create_focused_fixed_periphery_images(
                        standardized_raw_images, standardized_content_mask
                    )
                )

            if "standardized_soft_monai_mask_only" in required_modes:
                variants["standardized_soft_monai_mask_only"] = (
                    create_soft_monai_mask_only_images(
                        standardized_roi_probability,
                        standardized_valid_mask,
                    )
                )
            if "standardized_hard_monai_mask_only" in required_modes:
                variants["standardized_hard_monai_mask_only"] = (
                    create_hard_monai_mask_only_images(
                        standardized_hard_mask,
                        standardized_valid_mask,
                    )
                )
            if "standardized_monai_bbox_mask_only" in required_modes:
                variants["standardized_monai_bbox_mask_only"] = (
                    create_monai_bbox_mask_only_images(
                        standardized_hard_mask,
                        standardized_valid_mask,
                    )
                )
            if "standardized_soft_monai_histogram_only" in required_modes:
                variants["standardized_soft_monai_histogram_only"] = (
                    create_soft_monai_histogram_only_images(
                        standardized_roi_probability,
                        standardized_valid_mask,
                    )
                )
            if "standardized_soft_monai_block_shuffled" in required_modes:
                variants["standardized_soft_monai_block_shuffled"] = (
                    create_block_shuffled_soft_monai_mask_only_images(
                        standardized_roi_probability,
                        standardized_valid_mask,
                        decoded_pixel_hashes,
                    )
                )
            if "standardized_canonical_hard_monai_mask_only" in required_modes:
                variants["standardized_canonical_hard_monai_mask_only"] = (
                    create_canonicalized_hard_monai_mask_only_images(
                        standardized_hard_mask,
                        standardized_valid_mask,
                    )
                )
            if "standardized_canonical_soft_monai_mask_only" in required_modes:
                variants["standardized_canonical_soft_monai_mask_only"] = (
                    create_canonicalized_soft_monai_mask_only_images(
                        standardized_roi_probability,
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
            # ROI/fallback standard deviation remains in the reusable feature-bank
            # audit schema for compatibility with earlier weighting studies, but
            # no active focused experiment uses it to change slice influence.
            # Every shared value uses the same row indices.
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
        "original_monai_qc_computed": bool(need_original_monai),
        "standardized_monai_qc_computed": bool(
            need_standardized_monai
        ),
        "monai_qc_missing_branch_policy": (
            MONAI_QC_MISSING_BRANCH_POLICY
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

    # Keep this set synchronized with every representation that consumes a
    # MONAI probability map, hard mask, gate decision, or bounding box. This is
    # especially important for reduced or future runs that enable only one
    # localized candidate or one mask-derived control; such a run must still load
    # the segmenter without relying on another incidental MONAI-dependent mode.
    monai_dependent_modes = {
        "monai_roi",
        "outside_heart",
        "standardized_monai_roi",
        "standardized_center_crop",
        "standardized_roi_zero_background",
        "standardized_roi_zero_bg_center_fallback",
        "standardized_roi_bbox",
        "standardized_outside_large_bbox",
        "standardized_soft_monai_mask_only",
        "standardized_hard_monai_mask_only",
        "standardized_monai_bbox_mask_only",
        "standardized_soft_monai_histogram_only",
        "standardized_soft_monai_block_shuffled",
        "standardized_canonical_hard_monai_mask_only",
        "standardized_canonical_soft_monai_mask_only",
        "standardized_heart_centered_fixed_fov_region_norm",
        "standardized_hard_support_region_norm",
        "standardized_a17_exact_support_mask_only",
        "standardized_a17_support_intensity_affine_shuffled",
        "standardized_a17_exact_support_complement_region_norm",
        "standardized_outside_whole_heart_region_norm",
    }
    need_monai = any(
        mode in monai_dependent_modes for mode in required_modes
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


def _prepare_phash_review_tile(image_path, title, tile_size=384):
    """Load one candidate image into a square grayscale review tile."""

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = np.zeros((tile_size, tile_size), dtype=np.uint8)
        cv2.putText(
            image,
            "READ FAILED",
            (20, tile_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255,
            2,
            cv2.LINE_AA,
        )
    else:
        height, width = image.shape
        scale = min(tile_size / max(height, 1), tile_size / max(width, 1))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        canvas = np.zeros((tile_size, tile_size), dtype=np.uint8)
        top = (tile_size - resized_height) // 2
        left = (tile_size - resized_width) // 2
        canvas[top:top + resized_height, left:left + resized_width] = resized
        image = canvas

    header_height = 58
    tile = np.zeros((tile_size + header_height, tile_size, 3), dtype=np.uint8)
    tile[header_height:, :, :] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        tile,
        str(title)[:58],
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        Path(str(image_path)).name[:58],
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return tile


def write_phash_review_panels(rows, output_directory):
    """Create side-by-side panels and blank review fields for pHash candidates."""

    output_directory.mkdir(parents=True, exist_ok=True)
    for pair_index, row in enumerate(rows, start=1):
        # Keep outcome labels off the review image so visual duplicate
        # adjudication can be performed blinded. Labels remain in the CSV for
        # later merge after the reviewer records a status.
        title_a = (
            f"A: {row['patient_id_a']} | pHash={row['example_phash_a']}"
        )
        title_b = (
            f"B: {row['patient_id_b']} | pHash={row['example_phash_b']}"
        )
        tile_a = _prepare_phash_review_tile(row["example_image_a"], title_a)
        tile_b = _prepare_phash_review_tile(row["example_image_b"], title_b)
        panel = np.concatenate([tile_a, tile_b], axis=1)
        footer = np.zeros((70, panel.shape[1], 3), dtype=np.uint8)
        footer_text = (
            f"distance={row['minimum_phash_hamming_distance']} | "
            f"candidate_pairs={row['candidate_image_pairs']}"
        )
        cv2.putText(
            footer,
            footer_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            footer,
            "Manual status: confirmed_near_duplicate / generic_template / "
            "independent / uncertain",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        panel = np.concatenate([panel, footer], axis=0)
        safe_a = str(row["patient_id_a"]).replace("/", "_")
        safe_b = str(row["patient_id_b"]).replace("/", "_")
        panel_path = output_directory / (
            f"pair_{pair_index:03d}__{safe_a}__{safe_b}.png"
        )
        if not cv2.imwrite(str(panel_path), panel):
            raise RuntimeError(
                f"Could not write pHash review panel: {panel_path}"
            )
        row["review_panel_path"] = str(panel_path)
        row["manual_review_status"] = ""
        row["manual_review_notes"] = ""


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

    review_panel_directory = output_path.parent / "phash_review_panels"
    write_phash_review_panels(rows, review_panel_directory)

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
        "review_panel_path",
        "manual_review_status",
        "manual_review_notes",
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
        "review_panel_directory": str(review_panel_directory),
        "manual_review_status_values": [
            "confirmed_near_duplicate",
            "generic_template_or_localizer",
            "independent_image",
            "uncertain",
        ],
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
    """Write one auditable row per image including folds and provenance fields.

    An uncomputed MONAI branch is represented by an explicit branch-status
    column and empty QC cells, not by ``False`` gate values plus ``nan`` numeric
    cells. The latter would incorrectly imply that inference ran and every mask
    failed. Partial non-finite branches remain errors because they can indicate a
    damaged cache or numerical failure.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_qc_availability = inspect_monai_qc_branch_availability(
        bank, standardized=False
    )
    standardized_qc_availability = inspect_monai_qc_branch_availability(
        bank, standardized=True
    )
    for availability, standardized in (
        (original_qc_availability, False),
        (standardized_qc_availability, True),
    ):
        if availability["status"] == "PARTIAL_NONFINITE":
            _require_available_monai_qc_branch(
                bank, standardized=standardized
            )

    original_qc_available = bool(original_qc_availability["available"])
    standardized_qc_available = bool(
        standardized_qc_availability["available"]
    )

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
        "original_monai_qc_status",
        "monai_gate_valid",
        "monai_area_ratio",
        "monai_peak_probability",
        "monai_mean_foreground_probability",
        "roi_slice_std_score",
        "standardized_monai_qc_status",
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
                "original_monai_qc_status": original_qc_availability[
                    "status"
                ],
                "monai_gate_valid": (
                    int(bool(bank["monai_valid"][index]))
                    if original_qc_available
                    else ""
                ),
                "monai_area_ratio": (
                    float(bank["area_ratios"][index])
                    if original_qc_available
                    else ""
                ),
                "monai_peak_probability": (
                    float(bank["peak_probabilities"][index])
                    if original_qc_available
                    else ""
                ),
                "monai_mean_foreground_probability": (
                    float(bank["mean_foreground_probabilities"][index])
                    if original_qc_available
                    else ""
                ),
                "roi_slice_std_score": (
                    float(bank["roi_slice_scores"][index])
                    if original_qc_available
                    else ""
                ),
                "standardized_monai_qc_status": (
                    standardized_qc_availability["status"]
                ),
                "standardized_monai_gate_valid": (
                    int(bool(bank["standardized_monai_valid"][index]))
                    if standardized_qc_available
                    else ""
                ),
                "standardized_monai_area_ratio": (
                    float(bank["standardized_area_ratios"][index])
                    if standardized_qc_available
                    else ""
                ),
                "standardized_monai_peak_probability": (
                    float(bank["standardized_peak_probabilities"][index])
                    if standardized_qc_available
                    else ""
                ),
                "standardized_monai_mean_foreground_probability": (
                    float(
                        bank[
                            "standardized_mean_foreground_probabilities"
                        ][index]
                    )
                    if standardized_qc_available
                    else ""
                ),
                "standardized_roi_slice_std_score": (
                    float(bank["standardized_roi_slice_scores"][index])
                    if standardized_qc_available
                    else ""
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


def write_series_annotation_template(output_path, samples):
    """Write a blinded-ready manual sequence/view annotation template.

    JPEG folder names are not treated as validated DICOM series identities and
    this function does not infer sequence or view labels from SR_*/series* names.
    It records deterministic first, middle, and last image paths for every
    folder-defined proxy so a radiologist or trained reviewer can annotate the
    released structure without inspecting model predictions.
    """

    grouped = defaultdict(list)
    patient_label = {}
    for image_path, label, patient_id, series_id in samples:
        patient_id = str(patient_id)
        series_id = str(series_id)
        previous = patient_label.get(patient_id)
        if previous is not None and int(previous) != int(label):
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent series-template labels."
            )
        patient_label[patient_id] = int(label)
        grouped[(patient_id, series_id)].append(str(image_path))

    rows = []
    for patient_id, series_id in sorted(grouped):
        paths = sorted(grouped[(patient_id, series_id)])
        middle_index = len(paths) // 2
        rows.append(
            {
                "patient_id": patient_id,
                "series_proxy_id": series_id,
                "n_images": int(len(paths)),
                "first_image_path": paths[0],
                "middle_image_path": paths[middle_index],
                "last_image_path": paths[-1],
                "sequence_type": "",
                "view_type": "",
                "contains_heart": "",
                "is_localizer": "",
                "is_derived_export": "",
                "monai_mask_anatomically_plausible": "",
                "annotation_confidence": "",
                "reviewer": "",
                "notes": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Keep labels in a separate key so the annotation table can be given to a
    # reviewer without exposing Normal/Sick outcome. The key is merged only
    # after sequence/view annotation is complete.
    label_key_path = output_path.with_name(
        "series_annotation_label_key.csv"
    )
    key_rows = [
        {
            "patient_id": patient_id,
            "true_label": int(patient_label[patient_id]),
        }
        for patient_id in sorted(patient_label)
    ]
    with open(label_key_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    print(
        f"[SERIES REVIEW] Wrote {len(rows)} blinded folder-proxy rows: "
        f"{output_path}",
        flush=True,
    )
    print(
        f"[SERIES REVIEW] Label key kept separately: {label_key_path}",
        flush=True,
    )
    return rows


def _normalize_annotation_token(value):
    """Normalize one manually entered annotation token for exact matching."""

    return str(value or "").strip().casefold()


def _parse_required_annotation_boolean(value, field_name, row_number):
    """Parse a required blinded annotation Boolean without guessing.

    Manual CSV editors can serialize Boolean choices in several equivalent
    forms. Accepted positive values are ``1/true/yes/y`` and accepted negative
    values are ``0/false/no/n``. Blank or unfamiliar values return ``None`` so
    the row is counted as incomplete rather than silently coerced.
    """

    normalized = _normalize_annotation_token(value)
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    if normalized == "":
        return None

    print(
        f"[SERIES SUBSET] Row {row_number}: unrecognized {field_name}="
        f"{value!r}; treating the row as incomplete.",
        flush=True,
    )
    return None


def build_annotated_series_selection(annotation_path, bank):
    """Create one feature-bank row mask from completed blinded annotations.

    The function never infers sequence/view from ``SR_*`` or ``series*`` names.
    It reads only explicit reviewer entries from the completed CSV, applies the
    predeclared label-blind inclusion rules, verifies that the series identifiers
    belong to the current feature bank, and then expands selected series proxies
    to their aligned image rows.

    Rows with incomplete required annotations are reported and excluded. A
    stale annotation file containing unknown series identifiers fails loudly,
    because silently ignoring them could mix annotations from another dataset
    version. Sequence and view filters are optional; an empty configured tuple
    means that all explicitly annotated values are eligible.
    """

    annotation_path = Path(annotation_path).expanduser().resolve()
    required_columns = {
        "patient_id",
        "series_proxy_id",
        "contains_heart",
        "is_localizer",
        "is_derived_export",
        "annotation_confidence",
        "sequence_type",
        "view_type",
    }

    with open(annotation_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        actual_columns = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - actual_columns)
        if missing_columns:
            raise ValueError(
                "Completed series annotation CSV is missing required columns: "
                f"{missing_columns}."
            )
        annotation_rows = list(reader)

    if not annotation_rows:
        raise ValueError("Completed series annotation CSV contains no data rows.")

    bank_series_ids = np.asarray(bank["series_ids"]).astype(str)
    bank_patient_ids = np.asarray(bank["patient_ids"]).astype(str)
    if len(bank_series_ids) != len(bank_patient_ids):
        raise RuntimeError(
            "Feature-bank series and patient arrays have different lengths."
        )
    series_to_patient = {}
    for series_id, patient_id in zip(bank_series_ids, bank_patient_ids):
        previous = series_to_patient.get(str(series_id))
        if previous is not None and previous != str(patient_id):
            raise RuntimeError(
                f"Series proxy {series_id!r} maps to multiple patients in the "
                "feature bank."
            )
        series_to_patient[str(series_id)] = str(patient_id)
    available_series = set(series_to_patient)
    allowed_sequences = {
        _normalize_annotation_token(value)
        for value in ANNOTATED_SERIES_ALLOWED_SEQUENCE_TYPES
        if _normalize_annotation_token(value)
    }
    allowed_views = {
        _normalize_annotation_token(value)
        for value in ANNOTATED_SERIES_ALLOWED_VIEW_TYPES
        if _normalize_annotation_token(value)
    }
    allowed_confidences = {
        _normalize_annotation_token(value)
        for value in ANNOTATED_SERIES_ALLOWED_CONFIDENCE_VALUES
        if _normalize_annotation_token(value)
    }

    seen_series = set()
    selected_series = set()
    selected_annotation_rows = []
    incomplete_rows = 0
    excluded_by_confidence = 0
    excluded_by_clinical_rules = 0
    excluded_by_sequence_or_view = 0
    unknown_series = []

    for row_number, row in enumerate(annotation_rows, start=2):
        series_id = str(row.get("series_proxy_id", "")).strip()
        if not series_id:
            incomplete_rows += 1
            continue
        if series_id in seen_series:
            raise ValueError(
                "Completed annotation CSV contains a duplicate "
                f"series_proxy_id at row {row_number}: {series_id!r}."
            )
        seen_series.add(series_id)

        if series_id not in available_series:
            unknown_series.append(series_id)
            continue

        annotated_patient_id = str(row.get("patient_id", "")).strip()
        expected_patient_id = series_to_patient[series_id]
        if not annotated_patient_id:
            incomplete_rows += 1
            continue
        if annotated_patient_id != expected_patient_id:
            raise ValueError(
                f"Annotation row {row_number} maps series {series_id!r} to "
                f"patient {annotated_patient_id!r}, but the current feature "
                f"bank maps it to {expected_patient_id!r}."
            )

        contains_heart = _parse_required_annotation_boolean(
            row.get("contains_heart"), "contains_heart", row_number
        )
        is_localizer = _parse_required_annotation_boolean(
            row.get("is_localizer"), "is_localizer", row_number
        )
        is_derived = _parse_required_annotation_boolean(
            row.get("is_derived_export"), "is_derived_export", row_number
        )
        confidence = _normalize_annotation_token(
            row.get("annotation_confidence")
        )
        sequence_type = _normalize_annotation_token(row.get("sequence_type"))
        view_type = _normalize_annotation_token(row.get("view_type"))

        if (
            contains_heart is None
            or is_localizer is None
            or is_derived is None
            or not confidence
            or (
                ANNOTATED_SERIES_USE_SEQUENCE_VIEW_BALANCED_POOLING
                and (not sequence_type or not view_type)
            )
        ):
            incomplete_rows += 1
            continue

        if confidence not in allowed_confidences:
            excluded_by_confidence += 1
            continue

        if (
            ANNOTATED_SERIES_REQUIRE_CONTAINS_HEART
            and not contains_heart
        ):
            excluded_by_clinical_rules += 1
            continue
        if ANNOTATED_SERIES_EXCLUDE_LOCALIZERS and is_localizer:
            excluded_by_clinical_rules += 1
            continue
        if ANNOTATED_SERIES_EXCLUDE_DERIVED_EXPORTS and is_derived:
            excluded_by_clinical_rules += 1
            continue

        if allowed_sequences and sequence_type not in allowed_sequences:
            excluded_by_sequence_or_view += 1
            continue
        if allowed_views and view_type not in allowed_views:
            excluded_by_sequence_or_view += 1
            continue

        selected_series.add(series_id)
        selected_annotation_rows.append(
            {
                "patient_id": annotated_patient_id,
                "series_proxy_id": series_id,
                "sequence_type": str(row.get("sequence_type", "")).strip(),
                "view_type": str(row.get("view_type", "")).strip(),
                "contains_heart": int(contains_heart),
                "is_localizer": int(is_localizer),
                "is_derived_export": int(is_derived),
                "annotation_confidence": str(
                    row.get("annotation_confidence", "")
                ).strip(),
                "reviewer": str(row.get("reviewer", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
            }
        )

    if unknown_series:
        examples = unknown_series[:10]
        raise ValueError(
            "The completed annotation CSV contains series_proxy_id values that "
            "are absent from the current feature bank. This usually means the "
            "annotation belongs to another dataset/run. Examples: "
            f"{examples}; total_unknown={len(unknown_series)}."
        )
    if not selected_series:
        raise ValueError(
            "The completed annotations and configured inclusion rules retained "
            "no series proxies."
        )

    row_mask = np.isin(bank_series_ids, np.asarray(sorted(selected_series)))
    labels = np.asarray(bank["labels"], dtype=np.int64)[row_mask]
    patient_ids = np.asarray(bank["patient_ids"]).astype(str)[row_mask]
    selected_patient_ids, selected_patient_labels = build_patient_label_table(
        labels,
        patient_ids,
    )
    class_counts = {
        0: int(np.sum(selected_patient_labels == 0)),
        1: int(np.sum(selected_patient_labels == 1)),
    }
    if min(class_counts.values()) < N_SPLITS:
        raise ValueError(
            "Annotated-series subset must retain at least N_SPLITS patients in "
            "each class. Retained counts: "
            f"Normal={class_counts[0]}, Sick={class_counts[1]}, "
            f"N_SPLITS={N_SPLITS}."
        )

    series_to_pooling_cell = {
        str(row["series_proxy_id"]): (
            f"{_normalize_annotation_token(row['sequence_type'])}::"
            f"{_normalize_annotation_token(row['view_type'])}"
        )
        for row in selected_annotation_rows
    }
    selected_cell_counts = defaultdict(int)
    for cell in series_to_pooling_cell.values():
        selected_cell_counts[str(cell)] += 1

    summary = {
        "status": "READY",
        "annotation_path": str(annotation_path),
        "selection_name": str(ANNOTATED_SERIES_SELECTION_NAME),
        "n_annotation_rows": int(len(annotation_rows)),
        "n_feature_bank_series": int(len(available_series)),
        "n_selected_series": int(len(selected_series)),
        "n_selected_image_rows": int(np.sum(row_mask)),
        "n_selected_patients": int(len(selected_patient_ids)),
        "sequence_view_balanced_pooling_enabled": bool(
            ANNOTATED_SERIES_USE_SEQUENCE_VIEW_BALANCED_POOLING
        ),
        "n_selected_sequence_view_cells": int(len(selected_cell_counts)),
        "selected_sequence_view_cell_series_counts": dict(
            sorted(selected_cell_counts.items())
        ),
        "normal_patients": class_counts[0],
        "sick_patients": class_counts[1],
        "n_incomplete_rows_excluded": int(incomplete_rows),
        "n_rows_excluded_by_confidence": int(excluded_by_confidence),
        "n_rows_excluded_by_clinical_rules": int(
            excluded_by_clinical_rules
        ),
        "n_rows_excluded_by_sequence_or_view": int(
            excluded_by_sequence_or_view
        ),
        "require_contains_heart": bool(
            ANNOTATED_SERIES_REQUIRE_CONTAINS_HEART
        ),
        "exclude_localizers": bool(ANNOTATED_SERIES_EXCLUDE_LOCALIZERS),
        "exclude_derived_exports": bool(
            ANNOTATED_SERIES_EXCLUDE_DERIVED_EXPORTS
        ),
        "allowed_sequence_types": list(
            ANNOTATED_SERIES_ALLOWED_SEQUENCE_TYPES
        ),
        "allowed_view_types": list(ANNOTATED_SERIES_ALLOWED_VIEW_TYPES),
        "allowed_confidence_values": list(
            ANNOTATED_SERIES_ALLOWED_CONFIDENCE_VALUES
        ),
    }
    return (
        row_mask,
        selected_annotation_rows,
        series_to_pooling_cell,
        summary,
    )


def run_annotated_series_subset_analysis(
    bank,
    experiments,
    tabular_feature_sets,
    base_fold_manifest_rows,
):
    """Optionally rerun selected image experiments on annotated series only.

    This is a secondary sensitivity analysis, not a replacement for the locked
    full-cohort result. It can run only after a blinded reviewer completes the
    exported series annotation template. The selected image rows are filtered
    before pooling; a fresh patient-level fold manifest is then constructed for
    the retained patient set using the same duplicate-component constraints.
    No annotation field is derived from class labels or model predictions.
    """

    status_path = OUTPUT_DIR / "audits" / "series_annotation_analysis_status.json"
    if not RUN_ANNOTATED_SERIES_SUBSET_ANALYSIS:
        summary = {
            "status": "SKIPPED_DISABLED",
            "annotation_path": SERIES_ANNOTATION_INPUT_PATH,
            "selection_name": ANNOTATED_SERIES_SELECTION_NAME,
            "reason": (
                "Enable RUN_ANNOTATED_SERIES_SUBSET_ANALYSIS only after a "
                "blinded reviewer completes a copy of "
                "series_annotation_template.csv."
            ),
        }
        status_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[SERIES SUBSET] SKIPPED: no completed annotation CSV was enabled.",
            flush=True,
        )
        return summary

    (
        row_mask,
        selected_rows,
        series_to_pooling_cell,
        selection_summary,
    ) = build_annotated_series_selection(
        SERIES_ANNOTATION_INPUT_PATH,
        bank,
    )
    selection_root = (
        OUTPUT_DIR
        / "sequence_view_analysis"
        / str(ANNOTATED_SERIES_SELECTION_NAME)
    )
    experiment_root = selection_root / "experiments"
    comparison_root = selection_root / "comparison"
    selection_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    selected_series_path = selection_root / "selected_series_proxies.csv"
    with open(selected_series_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)

    selected_labels = np.asarray(bank["labels"], dtype=np.int64)[row_mask]
    selected_patients = np.asarray(bank["patient_ids"]).astype(str)[row_mask]
    patient_ids, patient_labels = build_patient_label_table(
        selected_labels,
        selected_patients,
    )
    base_group_by_patient = {
        str(row["patient_id"]): str(row["duplicate_component_id"])
        for row in base_fold_manifest_rows
    }
    missing_group_patients = sorted(
        set(patient_ids.tolist()) - set(base_group_by_patient)
    )
    if missing_group_patients:
        raise RuntimeError(
            "Annotated-series patients are missing duplicate-component IDs: "
            f"{missing_group_patients}."
        )
    subset_fold_rows = _build_fold_manifest_rows_for_seed(
        patient_ids,
        patient_labels,
        base_group_by_patient,
        CV_RANDOM_STATE + 70_000,
    )
    write_patient_fold_manifest(
        selection_root / "patient_fold_manifest.csv",
        subset_fold_rows,
    )

    experiments_by_id = {
        experiment.experiment_id: experiment for experiment in experiments
    }
    successful_results = []
    failed_results = []
    prepared_cache = {}

    for index, experiment_id in enumerate(
        ANNOTATED_SERIES_EXPERIMENT_IDS,
        start=1,
    ):
        experiment = experiments_by_id[experiment_id]
        analysis_experiment = experiment
        if (
            ANNOTATED_SERIES_USE_SEQUENCE_VIEW_BALANCED_POOLING
            and experiment.pooling_strategy == "hierarchical"
        ):
            analysis_experiment = replace(
                experiment,
                pooling_strategy="sequence_view_balanced",
                description=(
                    experiment.description
                    + " In this blinded annotated sensitivity analysis, series "
                    + "proxies are additionally balanced across explicit "
                    + "(sequence_type, view_type) cells."
                ),
            )
        print(
            f"[SERIES SUBSET] Experiment {index}/"
            f"{len(ANNOTATED_SERIES_EXPERIMENT_IDS)}: {experiment_id}; "
            f"pooling={analysis_experiment.pooling_strategy}",
            flush=True,
        )
        output_dir = experiment_root / experiment_id
        preparation_key = experiment_preparation_cache_key(analysis_experiment)
        try:
            if preparation_key not in prepared_cache:
                prepared_cache[preparation_key] = prepare_experiment_data(
                    analysis_experiment,
                    bank,
                    tabular_feature_sets,
                    row_mask=row_mask,
                    series_to_pooling_cell=series_to_pooling_cell,
                )
            result = run_one_experiment(
                experiment=analysis_experiment,
                prepared=prepared_cache[preparation_key],
                fold_manifest_rows=subset_fold_rows,
                output_dir=output_dir,
                legacy_prediction_cache=None,
            )
            successful_results.append(result)
        except Exception as error:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "experiment_id": experiment_id,
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
            failed_results.append(failure)
            (output_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                f"[SERIES SUBSET] FAILED {experiment_id}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            traceback.print_exc(file=sys.stdout)

    paired_rows = compare_experiments_to_baseline(successful_results)
    primary_rows = compare_predeclared_ablation_pairs(successful_results)
    summary_rows = write_master_outputs(
        successful_results,
        failed_results,
        paired_rows,
        primary_rows,
        comparison_root,
    )

    summary = {
        **selection_summary,
        "status": (
            "OK" if not failed_results else "COMPLETED_WITH_FAILURES"
        ),
        "selected_series_csv": str(selected_series_path),
        "fold_manifest": str(selection_root / "patient_fold_manifest.csv"),
        "comparison_directory": str(comparison_root),
        "requested_experiment_ids": list(ANNOTATED_SERIES_EXPERIMENT_IDS),
        "successful_experiment_ids": [
            result["config"].experiment_id for result in successful_results
        ],
        "failed_experiments": failed_results,
        "experiment_summary_rows": summary_rows,
        "interpretation": (
            "This sensitivity analysis uses manually annotated folder proxies "
            "only and, when enabled, equal weighting across explicit sequence/"
            "view cells. It does not create validated DICOM SeriesInstanceUIDs "
            "and does not replace independent external validation."
        ),
    }
    status_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[SERIES SUBSET] Completed annotated series/view sensitivity analysis: "
        f"patients={selection_summary['n_selected_patients']}, "
        f"series={selection_summary['n_selected_series']}, "
        f"successful={len(successful_results)}, failed={len(failed_results)}.",
        flush=True,
    )
    return summary


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
    decoded_pixel_hashes = np.asarray(bank["decoded_pixel_hashes"])
    sample_indices = np.asarray(bank["sample_indices"], dtype=np.int64)
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


def subset_patient_tabular_feature_set(feature_set, selected_feature_names):
    """Return one named subset of an already aligned patient feature matrix.

    This helper is used to decompose the broad provenance-only control into
    single-family experiments. Every subset retains exactly the same patient
    order and labels; only columns change. This avoids rerunning image decoding
    and makes it possible to identify whether slice counts, folder counts,
    native geometry, or file-size proxies are responsible for classification.
    """

    X, y, patient_ids, feature_names = feature_set
    feature_names = tuple(feature_names)
    selected_feature_names = tuple(selected_feature_names)
    missing = [name for name in selected_feature_names if name not in feature_names]
    if missing:
        raise ValueError(
            f"Requested tabular control features are missing: {missing}."
        )
    indices = np.asarray(
        [feature_names.index(name) for name in selected_feature_names],
        dtype=np.int64,
    )
    subset = np.asarray(X, dtype=np.float32)[:, indices]
    if subset.ndim != 2 or subset.shape[1] != len(selected_feature_names):
        raise RuntimeError("Patient tabular feature subsetting produced an invalid shape.")
    return (
        subset,
        np.asarray(y, dtype=np.int64),
        np.asarray(patient_ids),
        selected_feature_names,
    )


def build_provenance_component_control_sets(provenance_set):
    """Create predeclared single-family export/provenance controls.

    The broad C3 matrix can obtain a high AUC without revealing which released
    dataset property carries the label. These narrower controls separate:

        - total number of exported slices;
        - number of folder-defined series proxies;
        - mean and maximum proxy length;
        - native height/width/aspect-ratio summaries;
        - file-size and bytes-per-pixel summaries.

    None of these matrices contains EfficientNet embeddings, MONAI outputs, or
    central image texture.
    """

    feature_names = tuple(provenance_set[3])
    geometry_names = tuple(
        name
        for name in feature_names
        if any(
            token in name
            for token in (
                "native_height",
                "native_width",
                "aspect_ratio_width_over_height",
            )
        )
    )
    file_size_names = tuple(
        name
        for name in feature_names
        if any(
            token in name
            for token in (
                "file_size_bytes",
                "bytes_per_native_pixel",
            )
        )
    )

    return {
        "n_slices_only": subset_patient_tabular_feature_set(
            provenance_set,
            ("n_slices",),
        ),
        "n_series_only": subset_patient_tabular_feature_set(
            provenance_set,
            ("n_series_proxies",),
        ),
        "series_length_only": subset_patient_tabular_feature_set(
            provenance_set,
            ("mean_series_length", "max_series_length"),
        ),
        "native_geometry_only": subset_patient_tabular_feature_set(
            provenance_set,
            geometry_names,
        ),
        "file_size_only": subset_patient_tabular_feature_set(
            provenance_set,
            file_size_names,
        ),
    }


def inspect_monai_qc_branch_availability(bank, standardized=False):
    """Inspect whether one MONAI-QC branch was actually computed.

    The compact focused suite needs standardized MONAI masks for A17 and its
    controls, but no active experiment needs the historical original-canvas
    MONAI branch. During extraction the unused original QC arrays are therefore
    deliberately stored as all-NaN sentinels. Treating those sentinels as failed
    measurements would be incorrect, while replacing them with zeros would
    fabricate a QC signal. This helper distinguishes three cases:

      AVAILABLE
          every slice has finite area/confidence/foreground diagnostics;
      SKIPPED_NOT_COMPUTED
          all three diagnostic arrays contain no finite values, which is the
          expected sentinel state for an intentionally unexecuted branch;
      PARTIAL_NONFINITE
          some but not all diagnostics are finite. This indicates numerical or
          cache corruption and remains a hard error rather than being imputed.
    """

    prefix = "standardized_" if standardized else ""
    branch_name = "standardized" if standardized else "original"
    key_map = {
        "valid": f"{prefix}monai_valid",
        "area": f"{prefix}area_ratios",
        "peak": f"{prefix}peak_probabilities",
        "foreground": f"{prefix}mean_foreground_probabilities",
    }

    missing_keys = [key for key in key_map.values() if key not in bank]
    if missing_keys:
        raise RuntimeError(
            f"{branch_name.capitalize()} MONAI-QC arrays are missing from the "
            f"feature bank: {missing_keys}."
        )

    n_rows = int(len(np.asarray(bank["labels"])))
    arrays = {
        name: np.asarray(bank[key])
        for name, key in key_map.items()
    }
    for name, array in arrays.items():
        if array.ndim != 1 or len(array) != n_rows:
            raise RuntimeError(
                f"{branch_name.capitalize()} MONAI-QC array {name!r} has "
                f"shape {array.shape}; expected ({n_rows},)."
            )

    finite = {
        name: np.isfinite(np.asarray(arrays[name], dtype=np.float64))
        for name in ("area", "peak", "foreground")
    }
    row_all_finite = finite["area"] & finite["peak"] & finite["foreground"]
    row_any_finite = finite["area"] | finite["peak"] | finite["foreground"]
    n_all_finite = int(np.sum(row_all_finite))
    n_any_finite = int(np.sum(row_any_finite))

    metadata_key = (
        "standardized_monai_qc_computed"
        if standardized
        else "original_monai_qc_computed"
    )
    metadata_value = bank.get("metadata", {}).get(metadata_key)

    if n_all_finite == n_rows:
        status = "AVAILABLE"
        available = True
        reason = (
            f"All {n_rows} slice rows contain finite {branch_name} MONAI-QC "
            "diagnostics."
        )
    elif n_any_finite == 0:
        status = "SKIPPED_NOT_COMPUTED"
        available = False
        reason = (
            f"The {branch_name} MONAI inference branch was not executed by the "
            "active focused feature modes; its saved QC arrays contain the "
            "documented all-NaN sentinel values."
        )
    else:
        status = "PARTIAL_NONFINITE"
        available = False
        reason = (
            f"Only {n_all_finite}/{n_rows} rows have all finite diagnostics "
            f"and {n_any_finite}/{n_rows} have at least one finite diagnostic."
        )

    return {
        "status": status,
        "available": bool(available),
        "preprocessing_branch": branch_name,
        "n_slice_rows": n_rows,
        "n_rows_all_diagnostics_finite": n_all_finite,
        "n_rows_any_diagnostic_finite": n_any_finite,
        "n_nonfinite_area": int(np.sum(~finite["area"])),
        "n_nonfinite_peak": int(np.sum(~finite["peak"])),
        "n_nonfinite_foreground": int(np.sum(~finite["foreground"])),
        "metadata_declared_computed": metadata_value,
        "missing_branch_policy": MONAI_QC_MISSING_BRANCH_POLICY,
        "reason": reason,
    }


def _require_available_monai_qc_branch(bank, standardized=False):
    """Return availability metadata or fail on partial/non-finite corruption."""

    availability = inspect_monai_qc_branch_availability(
        bank, standardized=standardized
    )
    if availability["status"] == "PARTIAL_NONFINITE":
        branch_name = availability["preprocessing_branch"]
        raise RuntimeError(
            f"{branch_name.capitalize()} MONAI-QC arrays are partially "
            "non-finite. This is not the expected all-NaN sentinel for an "
            "uncomputed branch and may indicate numerical failure or an "
            f"incomplete/corrupt cache. Diagnostics: {availability}."
        )
    if not availability["available"]:
        branch_name = availability["preprocessing_branch"]
        raise RuntimeError(
            f"{branch_name.capitalize()} MONAI-QC was not computed. "
            f"Diagnostics: {availability}."
        )
    return availability


def _aggregate_available_patient_monai_qc_features(bank, standardized=False):
    """Aggregate one fully available MONAI-QC branch to patient rows.

    No NaN-aware statistic is used here. Once a branch is declared AVAILABLE,
    every slice-level diagnostic must be finite. This prevents NumPy's
    ``All-NaN slice`` warnings and prevents silent omission or imputation of
    failed measurements.
    """

    _require_available_monai_qc_branch(bank, standardized=standardized)
    prefix = "standardized_" if standardized else ""
    branch_name = "standardized" if standardized else "original"

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"])
    valid = np.asarray(bank[f"{prefix}monai_valid"], dtype=bool)
    area = np.asarray(bank[f"{prefix}area_ratios"], dtype=np.float32)
    peak = np.asarray(bank[f"{prefix}peak_probabilities"], dtype=np.float32)
    foreground = np.asarray(
        bank[f"{prefix}mean_foreground_probabilities"],
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
                f"Patient {patient_id} has inconsistent {branch_name} "
                "MONAI-QC labels."
            )

        patient_area = area[indices]
        patient_peak = peak[indices]
        patient_foreground = foreground[indices]
        if not (
            np.all(np.isfinite(patient_area))
            and np.all(np.isfinite(patient_peak))
            and np.all(np.isfinite(patient_foreground))
        ):
            raise RuntimeError(
                f"Patient {patient_id} contains non-finite {branch_name} "
                "MONAI-QC values after branch availability validation."
            )

        patient_labels.append(int(label_values[0]))
        rows.append(
            [
                float(np.mean(valid[indices])),
                float(np.median(patient_area)),
                float(np.std(patient_area)),
                float(np.median(patient_peak)),
                float(np.std(patient_peak)),
                float(np.median(patient_foreground)),
                float(np.std(patient_foreground)),
            ]
        )

    X = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(X)):
        raise RuntimeError(
            f"{branch_name.capitalize()} MONAI-QC patient features contain "
            "non-finite values after validated aggregation."
        )
    y = np.asarray(patient_labels, dtype=np.int64)
    return X, y, ordered_patients, feature_names


def aggregate_patient_monai_qc_features(bank):
    """Aggregate original-canvas MONAI QC when that branch was computed."""

    return _aggregate_available_patient_monai_qc_features(
        bank, standardized=False
    )


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
    """Aggregate standardized MONAI QC when that branch was computed."""

    return _aggregate_available_patient_monai_qc_features(
        bank, standardized=True
    )


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


def _write_monai_qc_unavailable_status(output_dir, availability, standardized):
    """Write an explicit machine-readable SKIPPED status for missing QC."""

    prefix = "standardized_" if standardized else ""
    branch_name = availability["preprocessing_branch"]
    comparison = {
        "status": "SKIPPED_NOT_COMPUTED",
        "preprocessing_branch": branch_name,
        "reason": availability["reason"],
        "availability": availability,
        "patient_level_normal_mean_gate_rate": None,
        "patient_level_sick_mean_gate_rate": None,
        "patient_level_sick_minus_normal_difference": None,
        "patient_level_difference_ci": None,
        "slice_level_normal_gate_rate_descriptive": None,
        "slice_level_sick_gate_rate_descriptive": None,
        "warning_threshold_absolute_difference": (
            MONAI_GATE_RATE_DIFFERENCE_WARNING
        ),
        "warning_triggered": False,
        "interpretation": (
            "This branch was intentionally not computed by the focused feature "
            "panel. No zeros, medians, or values from the other preprocessing "
            "branch were substituted."
        ),
    }
    (output_dir / f"{prefix}monai_qc_availability.json").write_text(
        json.dumps(availability, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / f"{prefix}monai_gate_class_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[{branch_name.upper()} MONAI QC] SKIPPED_NOT_COMPUTED: "
        f"{availability['reason']}",
        flush=True,
    )
    return comparison, None


def write_monai_qc_outputs(output_dir, bank):
    """Write original-canvas MONAI QC, or explicitly skip if uncomputed."""

    output_dir.mkdir(parents=True, exist_ok=True)
    availability = inspect_monai_qc_branch_availability(
        bank, standardized=False
    )
    if availability["status"] == "SKIPPED_NOT_COMPUTED":
        return _write_monai_qc_unavailable_status(
            output_dir, availability, standardized=False
        )
    if availability["status"] != "AVAILABLE":
        _require_available_monai_qc_branch(bank, standardized=False)

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
        "status": "OK",
        "preprocessing_branch": "original",
        "availability": availability,
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
    (output_dir / "monai_qc_availability.json").write_text(
        json.dumps(availability, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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
    """Write standardized MONAI QC, or explicitly skip if uncomputed."""

    output_dir.mkdir(parents=True, exist_ok=True)
    availability = inspect_monai_qc_branch_availability(
        bank, standardized=True
    )
    if availability["status"] == "SKIPPED_NOT_COMPUTED":
        return _write_monai_qc_unavailable_status(
            output_dir, availability, standardized=True
        )
    if availability["status"] != "AVAILABLE":
        _require_available_monai_qc_branch(bank, standardized=True)

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
        "status": "OK",
        "preprocessing_branch": "label_blind_standardized",
        "availability": availability,
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
            "A large class difference after standardization may still reflect "
            "sequence/protocol/export confounding; it is not segmentation-"
            "accuracy evidence."
        ),
    }
    (output_dir / "standardized_monai_qc_availability.json").write_text(
        json.dumps(availability, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "standardized_monai_gate_class_comparison.json").write_text(
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
            f"difference exceeds {MONAI_GATE_RATE_DIFFERENCE_WARNING:.0%}. "
            "Protocol/export confounding must be investigated.",
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
    series_to_pooling_cell=None,
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

    FIXED-CHUNK POOLING ABLATION
    --------------------------------

        deterministic series-independent hash order -> FIXED_CHUNK_SIZE chunks
        chunk means -----------------------> equal patient mean

    This branch ignores folder-proxy boundaries while preventing one long run
    of slices from receiving unlimited influence. It is designed specifically
    to test whether arbitrary SR_*/series* export partitioning is predictive.

    All branches are deterministic and label-free. Labels are checked only for
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
    if pooling_strategy not in {
        "hierarchical",
        "sequence_view_balanced",
        "flat",
        "fixed_chunk",
    }:
        raise ValueError(
            f"Unsupported patient embedding pooling: {pooling_strategy!r}."
        )
    if np.any(slice_weights < 0) or not np.all(np.isfinite(slice_weights)):
        raise ValueError("Slice weights must be finite and non-negative.")
    if pooling_strategy == "sequence_view_balanced" and not (
        series_to_pooling_cell
    ):
        raise ValueError(
            "sequence_view_balanced pooling requires explicit completed "
            "series-to-(sequence,view) annotations."
        )

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
        elif pooling_strategy == "fixed_chunk":
            # Ignore SR_*/series* folder boundaries and partition the patient's
            # deterministic series-independent hash order into equal maximum-
            # size chunks. Each chunk receives one embedding and equal patient-level
            # influence. This tests whether folder-defined export partitioning,
            # rather than a true acquisition hierarchy, drives performance.
            indices = np.asarray(patient_to_indices[patient_id], dtype=np.int64)
            # A deterministic SHA-256 ordering deliberately ignores series IDs
            # and prevents contiguous filesystem/folder order from recreating
            # the released proxy boundaries inside the chunks.
            ordering_keys = [
                hashlib.sha256(
                    f"{RANDOM_SEED}|fixed_chunk|{patient_id}|{int(index)}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                for index in indices
            ]
            indices = indices[np.argsort(np.asarray(ordering_keys))]
            chunk_embeddings = []
            for start in range(0, len(indices), FIXED_CHUNK_SIZE):
                chunk_indices = indices[start:start + FIXED_CHUNK_SIZE]
                weights = slice_weights[chunk_indices]
                if not np.any(weights > 0):
                    raise RuntimeError(
                        f"Patient {patient_id} has a fixed chunk with no positive weight."
                    )
                chunk_embeddings.append(
                    np.average(
                        features[chunk_indices],
                        axis=0,
                        weights=weights,
                    )
                )
            patient_embedding = np.mean(
                np.stack(chunk_embeddings, axis=0),
                axis=0,
            )
        else:
            series_embeddings = {}
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
                series_embeddings[series_id] = np.average(
                    features[indices],
                    axis=0,
                    weights=weights,
                )

            if pooling_strategy == "sequence_view_balanced":
                cell_to_series_embeddings = defaultdict(list)
                for series_id, series_embedding in series_embeddings.items():
                    cell = series_to_pooling_cell.get(str(series_id))
                    if not cell:
                        raise RuntimeError(
                            "Missing explicit sequence/view cell for selected "
                            f"series proxy {series_id!r}."
                        )
                    cell_to_series_embeddings[str(cell)].append(series_embedding)
                cell_embeddings = [
                    np.mean(
                        np.stack(cell_to_series_embeddings[cell], axis=0),
                        axis=0,
                    )
                    for cell in sorted(cell_to_series_embeddings)
                ]
                patient_embedding = np.mean(
                    np.stack(cell_embeddings, axis=0),
                    axis=0,
                )
            else:
                patient_embedding = np.mean(
                    np.stack(
                        [
                            series_embeddings[series_id]
                            for series_id in sorted(series_embeddings)
                        ],
                        axis=0,
                    ),
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


def deterministic_exact_within_patient_deduplication_mask(
    patient_ids,
    series_ids,
    decoded_pixel_hashes,
    sample_indices,
):
    """Keep one deterministic row per exact decoded image within a patient.

    Exact duplicates inside one Directory_* cannot leak across outer folds, but
    repeated exports can still receive repeated influence during series and
    patient pooling. This helper collapses every

        (patient_id, decoded_pixel_sha256)

    group to one canonical row selected without labels or model scores. The
    canonical occurrence is the lexicographically smallest series proxy and,
    within that proxy, the smallest immutable sample index. If the same decoded
    image appears in several proxy folders, this rule assigns the retained
    representative to one deterministic proxy rather than counting it several
    times. Patient identity remains exactly Directory_*.
    """

    # Convert all grouping fields to one-dimensional NumPy arrays first.
    # The explicit dimensionality check is important: indexing a 2-D array can
    # return another ndarray, and ndarrays are not hashable set elements.
    patient_ids = np.asarray(patient_ids, dtype=str)
    series_ids = np.asarray(series_ids, dtype=str)
    decoded_pixel_hashes = np.asarray(decoded_pixel_hashes, dtype=str)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)

    for array_name, array in (
        ("patient_ids", patient_ids),
        ("series_ids", series_ids),
        ("decoded_pixel_hashes", decoded_pixel_hashes),
        ("sample_indices", sample_indices),
    ):
        if array.ndim != 1:
            raise ValueError(
                f"{array_name} must be one-dimensional for exact "
                f"within-patient deduplication; received shape {array.shape}."
            )

    if not (
        len(patient_ids)
        == len(series_ids)
        == len(decoded_pixel_hashes)
        == len(sample_indices)
    ):
        raise ValueError(
            "Exact-deduplication arrays must have identical lengths."
        )

    keep = np.zeros(len(patient_ids), dtype=bool)
    order = np.lexsort(
        (
            sample_indices,
            series_ids,
            decoded_pixel_hashes,
            patient_ids,
        )
    )

    # NumPy's static type stubs allow scalar indexing to be inferred as either
    # a scalar or an ndarray. Convert each value explicitly to a native Python
    # string before constructing the set key. The key is therefore guaranteed
    # to be hashable both for the type checker and at runtime.
    seen: set[tuple[str, str]] = set()
    for raw_index in order:
        index = int(raw_index)
        key: tuple[str, str] = (
            str(patient_ids[index]),
            str(decoded_pixel_hashes[index]),
        )
        if key in seen:
            continue
        seen.add(key)
        keep[index] = True

    original_patients = {str(value) for value in patient_ids.tolist()}
    retained_patients = {str(value) for value in patient_ids[keep].tolist()}
    if retained_patients != original_patients:
        missing = sorted(original_patients - retained_patients)
        raise RuntimeError(
            "Exact within-patient deduplication removed all rows for "
            f"patients: {missing}."
        )
    return keep


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


def _sample_index_contract_sha256(sample_indices):
    """Hash the ordered immutable feature-bank row identifiers.

    The feature-bank fingerprint identifies the source dataset and preprocessing
    contract. Within that bank, the ordered sample-index digest proves whether
    two experiments used exactly the same slice rows after filtering, exact
    deduplication and deterministic dropout, without serializing tens of
    thousands of row identifiers into each experiment summary.
    """

    values = np.asarray(sample_indices, dtype="<i8")
    if values.ndim != 1:
        raise ValueError("sample_indices must be one-dimensional.")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def experiment_preparation_cache_key(experiment):
    """Return the exact label-blind representation cache key for an experiment."""

    return (
        experiment.feature_mode,
        experiment.strategy,
        experiment.pooling_strategy,
        experiment.weighting_mode,
        float(experiment.slice_dropout_rate),
        bool(experiment.deduplicate_exact_within_patient),
        str(experiment.slice_filter),
    )


def prepare_experiment_data(
    experiment,
    bank,
    tabular_feature_sets,
    row_mask=None,
    series_to_pooling_cell=None,
):
    """Build one fixed, label-blind representation for an experiment.

    Manual sequence/view filtering and the optional common MONAI-valid filter
    are applied on the shared feature-bank row axis. Features, labels, patient
    IDs, series IDs, hashes, sample indices and slice scores are filtered by the
    same Boolean mask before deduplication, robustness dropout or pooling. This
    preserves the patient-level matched-comparison contract.
    """

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"]).astype(str)
    series_ids = np.asarray(bank["series_ids"]).astype(str)
    decoded_pixel_hashes = np.asarray(bank["decoded_pixel_hashes"]).astype(str)
    sample_indices = np.asarray(bank["sample_indices"], dtype=np.int64)

    if experiment.strategy == "patient_tabular":
        if row_mask is not None:
            raise ValueError(
                "A slice/series row mask cannot be applied to patient-tabular "
                "controls. Annotated-series analysis is restricted to image "
                "patient-embedding experiments."
            )
        if experiment.slice_filter != "all":
            raise ValueError(
                "Patient-tabular controls cannot use an image-row slice filter."
            )
        X, y, patients, feature_names = tabular_feature_sets[
            experiment.feature_mode
        ]
        return {
            "unit": "patient",
            "X": np.asarray(X, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int64),
            "patient_ids": np.asarray(patients).astype(str),
            "feature_names": tuple(feature_names),
            "slice_dropout_rate": 0.0,
            "slice_filter": "all",
            "n_source_slices": None,
            "n_after_slice_filter": None,
            "n_retained_slices": None,
            "retained_sample_indices_sha256": None,
        }

    features = bank["features"][experiment.feature_mode]
    if experiment.feature_mode.startswith("standardized_"):
        roi_slice_scores = np.asarray(
            bank["standardized_roi_slice_scores"], dtype=np.float64
        )
    else:
        roi_slice_scores = np.asarray(
            bank["roi_slice_scores"], dtype=np.float64
        )

    eligible = np.ones(len(labels), dtype=bool)
    series_subset_applied = row_mask is not None
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if row_mask.ndim != 1 or len(row_mask) != len(labels):
            raise ValueError(
                "Annotated-series row_mask must be one-dimensional and align "
                "with every feature-bank row."
            )
        if not np.any(row_mask):
            raise ValueError("Annotated-series filtering retained no image rows.")
        eligible &= row_mask

    expected_patients = set(patient_ids[eligible].tolist())
    n_source_slices = int(np.sum(eligible))
    combined_mask = eligible.copy()
    if experiment.slice_filter == "standardized_monai_valid":
        monai_valid = np.asarray(bank["standardized_monai_valid"], dtype=bool)
        if monai_valid.shape != combined_mask.shape:
            raise RuntimeError(
                "standardized_monai_valid does not align with feature-bank rows."
            )
        combined_mask &= monai_valid
    elif experiment.slice_filter != "all":
        raise ValueError(
            f"Unsupported slice_filter: {experiment.slice_filter!r}."
        )

    if not np.any(combined_mask):
        raise ValueError(
            f"{experiment.experiment_id}: row and slice filters retained no images."
        )
    retained_filter_patients = set(patient_ids[combined_mask].tolist())
    if retained_filter_patients != expected_patients:
        missing = sorted(expected_patients - retained_filter_patients)
        raise RuntimeError(
            f"{experiment.experiment_id}: filtering removed every row for "
            f"patients {missing}."
        )

    if not np.all(combined_mask):
        features = features[combined_mask]
        labels = labels[combined_mask]
        patient_ids = patient_ids[combined_mask]
        series_ids = series_ids[combined_mask]
        roi_slice_scores = roi_slice_scores[combined_mask]
        decoded_pixel_hashes = decoded_pixel_hashes[combined_mask]
        sample_indices = sample_indices[combined_mask]

    n_after_slice_filter = int(len(labels))
    if experiment.slice_filter != "all":
        print(
            f"[SLICE FILTER] {experiment.experiment_id}: retained "
            f"{n_after_slice_filter}/{n_source_slices} rows using "
            f"{experiment.slice_filter!r}.",
            flush=True,
        )

    n_exact_duplicate_rows_removed = 0
    if experiment.deduplicate_exact_within_patient:
        keep = deterministic_exact_within_patient_deduplication_mask(
            patient_ids,
            series_ids,
            decoded_pixel_hashes,
            sample_indices,
        )
        n_exact_duplicate_rows_removed = int(len(keep) - int(keep.sum()))
        features = features[keep]
        labels = labels[keep]
        patient_ids = patient_ids[keep]
        series_ids = series_ids[keep]
        roi_slice_scores = roi_slice_scores[keep]
        decoded_pixel_hashes = decoded_pixel_hashes[keep]
        sample_indices = sample_indices[keep]
        print(
            f"[EXACT DEDUP] {experiment.experiment_id}: retained "
            f"{int(keep.sum())}/{len(keep)} slices; removed="
            f"{n_exact_duplicate_rows_removed} repeated exact exports.",
            flush=True,
        )

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
        decoded_pixel_hashes = decoded_pixel_hashes[retention_mask]
        sample_indices = sample_indices[retention_mask]
        print(
            f"[ROBUSTNESS] {experiment.experiment_id}: retained "
            f"{int(retention_mask.sum())}/{len(retention_mask)} slices after "
            f"{experiment.slice_dropout_rate:.0%} within-series dropout.",
            flush=True,
        )

    retained_sample_indices_sha256 = _sample_index_contract_sha256(
        sample_indices
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
            series_to_pooling_cell=series_to_pooling_cell,
        )
        return {
            "unit": "patient",
            "X": X,
            "y": y,
            "patient_ids": patients,
            "feature_names": tuple(
                f"embedding_{index:04d}" for index in range(X.shape[1])
            ),
            "slice_dropout_rate": float(experiment.slice_dropout_rate),
            "slice_filter": str(experiment.slice_filter),
            "deduplicate_exact_within_patient": bool(
                experiment.deduplicate_exact_within_patient
            ),
            "n_exact_duplicate_rows_removed": int(
                n_exact_duplicate_rows_removed
            ),
            "n_source_slices": n_source_slices,
            "n_after_slice_filter": n_after_slice_filter,
            "n_retained_slices": int(len(labels)),
            "retained_sample_indices_sha256": (
                retained_sample_indices_sha256
            ),
            "annotated_series_subset_applied": bool(series_subset_applied),
        }

    if experiment.strategy == "slice_probability_fusion":
        return {
            "unit": "slice",
            "X": features,
            "y": labels,
            "patient_ids": patient_ids,
            "series_ids": series_ids,
            "slice_weights": slice_weights,
            "legacy_cache_signature": (
                experiment.feature_mode,
                experiment.weighting_mode,
                experiment.slice_filter,
                int(features.shape[0]),
                int(features.shape[1]),
            ),
            "slice_dropout_rate": float(experiment.slice_dropout_rate),
            "slice_filter": str(experiment.slice_filter),
            "deduplicate_exact_within_patient": bool(
                experiment.deduplicate_exact_within_patient
            ),
            "n_exact_duplicate_rows_removed": int(
                n_exact_duplicate_rows_removed
            ),
            "n_source_slices": n_source_slices,
            "n_after_slice_filter": n_after_slice_filter,
            "n_retained_slices": int(len(labels)),
            "retained_sample_indices_sha256": (
                retained_sample_indices_sha256
            ),
            "annotated_series_subset_applied": bool(series_subset_applied),
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

    The focused registry uses patient-level pooled embeddings only. The generic
    legacy slice-probability branch is retained in the implementation so older
    archived configurations can still be reproduced from this code lineage.
    When two such archived variants share training data, weights, classifier and
    C and differ only in fusion, a process-local cache can reuse held-out slice
    probabilities without sharing fitted information across folds.
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
        f"slice_dropout={experiment.slice_dropout_rate:.0%}, "
        f"slice_filter={experiment.slice_filter}, "
        f"exact_within_patient_dedup="
        f"{experiment.deduplicate_exact_within_patient}",
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
        "prepared_data": {
            "slice_dropout_rate": prepared.get("slice_dropout_rate", 0.0),
            "deduplicate_exact_within_patient": prepared.get(
                "deduplicate_exact_within_patient", False
            ),
            "n_exact_duplicate_rows_removed": prepared.get(
                "n_exact_duplicate_rows_removed", 0
            ),
            "n_source_slices": prepared.get("n_source_slices"),
            "n_retained_slices": prepared.get("n_retained_slices"),
            "annotated_series_subset_applied": bool(
                prepared.get("annotated_series_subset_applied", False)
            ),
        },
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
            "deduplicate_exact_within_patient": prepared.get(
                "deduplicate_exact_within_patient", False
            ),
            "n_exact_duplicate_rows_removed": prepared.get(
                "n_exact_duplicate_rows_removed", 0
            ),
            "n_source_slices": prepared.get("n_source_slices"),
            "n_after_slice_filter": prepared.get("n_after_slice_filter"),
            "n_retained_slices": prepared.get("n_retained_slices"),
            "slice_filter": prepared.get("slice_filter", "all"),
            "retained_sample_indices_sha256": prepared.get(
                "retained_sample_indices_sha256"
            ),
            "annotated_series_subset_applied": bool(
                prepared.get("annotated_series_subset_applied", False)
            ),
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
    """Repeat one selected nested-CV experiment over deterministic outer splits."""

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


def compare_repeated_nested_cv_pairs(stability_root, comparisons):
    """Compare selected experiments on the exact same repeated split seeds.

    Each experiment's repeated_nested_cv_runs.csv contains one AUC/AUPRC row
    for every predeclared outer-CV random state. Aligning those rows by seed
    gives a genuinely paired split-sensitivity comparison. The empirical 2.5th
    and 97.5th percentiles below describe the distribution of per-seed deltas;
    they are not presented as an independent-sample confidence interval because
    all repeats reuse the same small cohort.
    """

    stability_root = Path(stability_root)
    per_repeat_rows = []
    summary_rows = []

    def read_runs(experiment_id):
        path = (
            stability_root
            / experiment_id
            / "repeated_nested_cv_runs.csv"
        )
        if not path.is_file():
            return None, path
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        by_seed = {
            int(row["outer_cv_random_state"]): {
                "repeat_index": int(row["repeat_index"]),
                "auc": float(row["auc"]),
                "auprc": float(row["auprc"]),
            }
            for row in rows
        }
        return by_seed, path

    for comparison_name, reference_id, comparison_id, question in comparisons:
        reference_runs, reference_path = read_runs(reference_id)
        comparison_runs, comparison_path = read_runs(comparison_id)

        if reference_runs is None or comparison_runs is None:
            summary_rows.append(
                {
                    "comparison_name": comparison_name,
                    "reference_experiment_id": reference_id,
                    "comparison_experiment_id": comparison_id,
                    "scientific_question": question,
                    "status": "SKIPPED_MISSING_STABILITY_RUNS",
                    "n_common_repeats": 0,
                    "delta_auc_mean": "",
                    "delta_auc_median": "",
                    "delta_auc_std": "",
                    "delta_auc_q025": "",
                    "delta_auc_q25": "",
                    "delta_auc_q75": "",
                    "delta_auc_q975": "",
                    "delta_auc_minimum": "",
                    "delta_auc_maximum": "",
                    "comparison_better_fraction": "",
                    "reference_better_fraction": "",
                    "tie_fraction": "",
                    "reference_runs_csv": str(reference_path),
                    "comparison_runs_csv": str(comparison_path),
                }
            )
            continue

        common_seeds = sorted(set(reference_runs).intersection(comparison_runs))
        if not common_seeds:
            raise RuntimeError(
                f"No common repeated-CV seeds for {comparison_name}."
            )

        auc_deltas = []
        for seed in common_seeds:
            reference_row = reference_runs[seed]
            comparison_row = comparison_runs[seed]
            delta_auc = comparison_row["auc"] - reference_row["auc"]
            delta_auprc = comparison_row["auprc"] - reference_row["auprc"]
            auc_deltas.append(delta_auc)
            per_repeat_rows.append(
                {
                    "comparison_name": comparison_name,
                    "reference_experiment_id": reference_id,
                    "comparison_experiment_id": comparison_id,
                    "outer_cv_random_state": int(seed),
                    "reference_repeat_index": int(
                        reference_row["repeat_index"]
                    ),
                    "comparison_repeat_index": int(
                        comparison_row["repeat_index"]
                    ),
                    "reference_auc": float(reference_row["auc"]),
                    "comparison_auc": float(comparison_row["auc"]),
                    "delta_auc": float(delta_auc),
                    "reference_auprc": float(reference_row["auprc"]),
                    "comparison_auprc": float(comparison_row["auprc"]),
                    "delta_auprc": float(delta_auprc),
                }
            )

        deltas = np.asarray(auc_deltas, dtype=np.float64)
        summary_rows.append(
            {
                "comparison_name": comparison_name,
                "reference_experiment_id": reference_id,
                "comparison_experiment_id": comparison_id,
                "scientific_question": question,
                "status": "OK",
                "n_common_repeats": int(len(deltas)),
                "delta_auc_mean": float(np.mean(deltas)),
                "delta_auc_median": float(np.median(deltas)),
                "delta_auc_std": float(np.std(deltas)),
                "delta_auc_q025": float(np.quantile(deltas, 0.025)),
                "delta_auc_q25": float(np.quantile(deltas, 0.25)),
                "delta_auc_q75": float(np.quantile(deltas, 0.75)),
                "delta_auc_q975": float(np.quantile(deltas, 0.975)),
                "delta_auc_minimum": float(np.min(deltas)),
                "delta_auc_maximum": float(np.max(deltas)),
                "comparison_better_fraction": float(np.mean(deltas > 0.0)),
                "reference_better_fraction": float(np.mean(deltas < 0.0)),
                "tie_fraction": float(np.mean(deltas == 0.0)),
                "reference_runs_csv": str(reference_path),
                "comparison_runs_csv": str(comparison_path),
            }
        )

    per_repeat_path = (
        stability_root / "repeated_nested_cv_paired_deltas.csv"
    )
    summary_path = (
        stability_root / "repeated_nested_cv_paired_comparisons.csv"
    )
    if per_repeat_rows:
        with open(per_repeat_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(per_repeat_rows[0])
            )
            writer.writeheader()
            writer.writerows(per_repeat_rows)
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_payload = {
        "status": "OK",
        "delta_definition": "comparison AUC minus reference AUC on the same seed",
        "repeats_are_descriptive_not_independent_samples": True,
        "comparisons": summary_rows,
        "per_repeat_csv": str(per_repeat_path),
        "summary_csv": str(summary_path),
    }
    (
        stability_root / "repeated_nested_cv_paired_comparisons.json"
    ).write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n[STABILITY] PAIRED REPEATED-CV COMPARISONS", flush=True)
    for row in summary_rows:
        if row["status"] != "OK":
            print(
                f"[STABILITY][PAIRED] {row['comparison_name']}: "
                f"{row['status']}",
                flush=True,
            )
            continue
        print(
            f"[STABILITY][PAIRED] {row['comparison_name']}: "
            f"median_delta={row['delta_auc_median']:+.4f}, "
            f"IQR=[{row['delta_auc_q25']:+.4f}, "
            f"{row['delta_auc_q75']:+.4f}], "
            f"comparison_better={row['comparison_better_fraction']:.1%}",
            flush=True,
        )
    return json_payload


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




def _clone_prepared_with_patient_label_mapping(prepared, label_by_patient):
    """Clone one patient-level representation with a new label assignment.

    Frozen image embeddings, patient IDs, pooling and every label-blind
    preprocessing output remain unchanged. Only the one-row-per-patient target
    vector is replaced. This is the correct unit for the selection-adjusted
    patient-label permutation null.
    """

    if prepared.get("unit") != "patient":
        raise ValueError(
            "Candidate-family selection requires one prepared row per patient."
        )
    patient_ids = np.asarray(prepared["patient_ids"]).astype(str)
    normalized_mapping = {
        str(patient_id): int(label)
        for patient_id, label in label_by_patient.items()
    }
    missing = sorted(set(patient_ids.tolist()) - set(normalized_mapping))
    if missing:
        raise ValueError(f"Missing replacement labels for patients: {missing}.")

    cloned = dict(prepared)
    cloned["patient_ids"] = patient_ids
    cloned["y"] = np.asarray(
        [normalized_mapping[patient_id] for patient_id in patient_ids],
        dtype=np.int64,
    )
    return cloned


def _run_nested_candidate_selection_once(
    candidate_experiment_ids,
    experiments_by_id,
    prepared_by_id,
    fold_manifest_rows,
    verbose=False,
):
    """Select representation and C inside outer training, then score outer test.

    Candidate identity is never chosen from outer-validation performance. Each
    candidate first executes its ordinary inner C-selection path inside the
    current outer-training cohort. Candidates within the predeclared
    ``MODEL_SELECTION_AUC_TOLERANCE`` of the best selected inner AUC are resolved
    by the fixed priority order in ``candidate_experiment_ids``. The selected
    candidate is then fitted once on the complete outer-training cohort and
    applied to outer validation.
    """

    candidate_experiment_ids = tuple(candidate_experiment_ids)
    if not candidate_experiment_ids:
        raise ValueError("At least one candidate is required for model selection.")

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

    # Every candidate must represent the same patient cohort and labels. Slice
    # filtering is allowed, but it must not remove every retained row for any
    # patient before patient-level pooling.
    for experiment_id in candidate_experiment_ids:
        if experiment_id not in experiments_by_id:
            raise KeyError(f"Unknown candidate experiment: {experiment_id}.")
        if experiment_id not in prepared_by_id:
            raise KeyError(f"Prepared candidate data missing: {experiment_id}.")
        prepared = prepared_by_id[experiment_id]
        if prepared.get("unit") != "patient":
            raise ValueError(
                f"{experiment_id}: candidate selection requires patient rows."
            )
        candidate_patients = np.asarray(prepared["patient_ids"]).astype(str)
        if set(candidate_patients.tolist()) != set(all_patients.tolist()):
            raise RuntimeError(
                f"{experiment_id}: candidate patient cohort does not match the "
                "outer-fold manifest."
            )
        candidate_label_by_patient = {
            str(patient_id): int(label)
            for patient_id, label in zip(candidate_patients, prepared["y"])
        }
        mismatched = [
            patient_id
            for patient_id in all_patients
            if candidate_label_by_patient[str(patient_id)]
            != patient_to_label[str(patient_id)]
        ]
        if mismatched:
            raise RuntimeError(
                f"{experiment_id}: prepared labels disagree for {mismatched}."
            )

    score_by_patient = {}
    prediction_by_patient = {}
    threshold_by_patient = {}
    fold_by_patient = {}
    selected_model_by_patient = {}
    selected_c_by_patient = {}
    selected_inner_auc_by_patient = {}
    outer_fold_rows = []

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

        candidate_selections = {}
        for priority_index, experiment_id in enumerate(candidate_experiment_ids):
            experiment = experiments_by_id[experiment_id]
            selection = select_c_and_training_threshold(
                experiment=experiment,
                prepared=prepared_by_id[experiment_id],
                outer_train_patient_ids=train_patients,
                outer_train_labels=train_labels,
                patient_to_group=patient_to_group,
                outer_fold_index=outer_fold,
                legacy_prediction_cache=None,
                verbose=False,
            )
            candidate_selections[experiment_id] = {
                "priority_index": int(priority_index),
                "selection": selection,
                "selected_inner_auc": float(selection["inner_auc"]),
            }

        best_inner_auc = max(
            row["selected_inner_auc"]
            for row in candidate_selections.values()
        )
        eligible_ids = [
            experiment_id
            for experiment_id in candidate_experiment_ids
            if candidate_selections[experiment_id]["selected_inner_auc"]
            >= best_inner_auc - MODEL_SELECTION_AUC_TOLERANCE
        ]
        # ``eligible_ids`` follows the immutable candidate priority order.
        selected_experiment_id = eligible_ids[0]
        selected_experiment = experiments_by_id[selected_experiment_id]
        selected = candidate_selections[selected_experiment_id]["selection"]

        raw_scores, valid_labels, evaluated_patients = (
            fit_predict_raw_for_patient_sets(
                experiment=selected_experiment,
                prepared=prepared_by_id[selected_experiment_id],
                train_patients=train_patients,
                valid_patients=valid_patients,
                c_value=selected["selected_c"],
                legacy_prediction_cache=None,
            )
        )
        if selected_experiment.classifier_type == "linear_svm":
            probabilities = apply_sigmoid_calibrator(
                selected["calibrator"], raw_scores
            )
        else:
            probabilities = np.clip(raw_scores, 0.0, 1.0)
        predictions = (
            probabilities >= float(selected["threshold"])
        ).astype(np.int64)

        for patient_id, label, score, prediction in zip(
            evaluated_patients,
            valid_labels,
            probabilities,
            predictions,
        ):
            patient_id = str(patient_id)
            if int(label) != patient_to_label[patient_id]:
                raise RuntimeError(
                    f"Candidate-selection label mismatch for {patient_id}."
                )
            if patient_id in score_by_patient:
                raise RuntimeError(
                    f"Patient {patient_id} received multiple selected-model scores."
                )
            score_by_patient[patient_id] = float(score)
            prediction_by_patient[patient_id] = int(prediction)
            threshold_by_patient[patient_id] = float(selected["threshold"])
            fold_by_patient[patient_id] = int(outer_fold)
            selected_model_by_patient[patient_id] = selected_experiment_id
            selected_c_by_patient[patient_id] = float(selected["selected_c"])
            selected_inner_auc_by_patient[patient_id] = float(
                selected["inner_auc"]
            )

        candidate_auc_map = {
            experiment_id: float(
                candidate_selections[experiment_id]["selected_inner_auc"]
            )
            for experiment_id in candidate_experiment_ids
        }
        outer_fold_rows.append(
            {
                "outer_fold": int(outer_fold),
                "train_patients": int(len(train_patients)),
                "valid_patients": int(len(valid_patients)),
                "selected_experiment_id": selected_experiment_id,
                "selected_c": float(selected["selected_c"]),
                "selected_inner_auc": float(selected["inner_auc"]),
                "best_candidate_inner_auc": float(best_inner_auc),
                "training_only_threshold": float(selected["threshold"]),
                "candidate_inner_auc_json": json.dumps(
                    candidate_auc_map, sort_keys=True
                ),
            }
        )
        if verbose:
            print(
                f"[MODEL SELECTION] outer={outer_fold}, selected="
                f"{selected_experiment_id}, inner_AUC="
                f"{selected['inner_auc']:.4f}, C={selected['selected_c']:g}",
                flush=True,
            )

    if set(score_by_patient) != set(all_patients.tolist()):
        missing = sorted(set(all_patients.tolist()) - set(score_by_patient))
        raise RuntimeError(
            f"Candidate-family nested selection is missing patients: {missing}."
        )

    ordered_patients = np.asarray(sorted(score_by_patient))
    return {
        "patient_ids": ordered_patients,
        "labels": np.asarray(
            [patient_to_label[patient_id] for patient_id in ordered_patients],
            dtype=np.int64,
        ),
        "scores": np.asarray(
            [score_by_patient[patient_id] for patient_id in ordered_patients],
            dtype=np.float64,
        ),
        "predictions": np.asarray(
            [prediction_by_patient[patient_id] for patient_id in ordered_patients],
            dtype=np.int64,
        ),
        "thresholds": np.asarray(
            [threshold_by_patient[patient_id] for patient_id in ordered_patients],
            dtype=np.float64,
        ),
        "folds": np.asarray(
            [fold_by_patient[patient_id] for patient_id in ordered_patients],
            dtype=np.int64,
        ),
        "selected_experiment_ids": np.asarray(
            [selected_model_by_patient[patient_id] for patient_id in ordered_patients]
        ),
        "selected_cs": np.asarray(
            [selected_c_by_patient[patient_id] for patient_id in ordered_patients],
            dtype=np.float64,
        ),
        "selected_inner_aucs": np.asarray(
            [
                selected_inner_auc_by_patient[patient_id]
                for patient_id in ordered_patients
            ],
            dtype=np.float64,
        ),
        "outer_fold_rows": outer_fold_rows,
    }


def run_nested_candidate_selection_audit(
    candidate_experiment_ids,
    experiments_by_id,
    prepared_by_id,
    fold_manifest_rows,
    output_dir,
):
    """Evaluate the complete predeclared representation-plus-C procedure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result = _run_nested_candidate_selection_once(
        candidate_experiment_ids,
        experiments_by_id,
        prepared_by_id,
        fold_manifest_rows,
        verbose=True,
    )
    metrics = compute_binary_patient_metrics(
        result["labels"], result["scores"], result["predictions"]
    )
    intervals = bootstrap_patient_metric_intervals(
        result["labels"],
        result["scores"],
        result["predictions"],
        random_state=RANDOM_SEED + 71_000,
    )
    selected_fold_ids = [
        row["selected_experiment_id"] for row in result["outer_fold_rows"]
    ]
    selection_counts = {
        experiment_id: int(selected_fold_ids.count(experiment_id))
        for experiment_id in candidate_experiment_ids
    }

    oof_rows = []
    for row in zip(
        result["patient_ids"],
        result["labels"],
        result["folds"],
        result["scores"],
        result["thresholds"],
        result["predictions"],
        result["selected_experiment_ids"],
        result["selected_cs"],
        result["selected_inner_aucs"],
    ):
        oof_rows.append(
            {
                "patient_id": str(row[0]),
                "true_label": int(row[1]),
                "outer_fold": int(row[2]),
                "oof_score": float(row[3]),
                "training_only_threshold": float(row[4]),
                "predicted_label": int(row[5]),
                "selected_experiment_id": str(row[6]),
                "selected_c": float(row[7]),
                "selected_inner_auc": float(row[8]),
            }
        )

    with open(
        output_dir / "candidate_selection_oof_predictions.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(oof_rows[0]))
        writer.writeheader()
        writer.writerows(oof_rows)
    with open(
        output_dir / "candidate_selection_outer_folds.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(result["outer_fold_rows"][0])
        )
        writer.writeheader()
        writer.writerows(result["outer_fold_rows"])

    summary = {
        "status": "OK",
        "candidate_experiment_ids_in_priority_order": list(
            candidate_experiment_ids
        ),
        "model_selection_auc_tolerance": float(MODEL_SELECTION_AUC_TOLERANCE),
        "selection_rule": (
            "Inside each outer-training cohort, select C separately for each "
            "candidate, then choose the first predeclared candidate whose "
            "selected inner OOF AUC is within the configured tolerance of the "
            "best candidate. Evaluate once on outer validation."
        ),
        "n_patients": int(len(result["labels"])),
        "metrics": metrics,
        "confidence_intervals": intervals,
        "selected_model_counts_across_outer_folds": selection_counts,
        "outer_fold_rows": result["outer_fold_rows"],
        "observed_selection_aware_auc": float(metrics["auc"]),
        "interpretation": (
            "This is internal validation of the predeclared candidate-selection "
            "procedure. It is not an independent external cohort."
        ),
    }
    (output_dir / "candidate_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[MODEL SELECTION] selection-aware OOF AUC="
        f"{metrics['auc']:.4f}; selected folds={selection_counts}",
        flush=True,
    )
    return summary


def run_repeated_nested_candidate_selection_stability(
    candidate_experiment_ids,
    experiments_by_id,
    prepared_by_id,
    base_fold_manifest_rows,
    output_dir,
):
    """Repeat the complete model-level selection procedure over outer seeds."""

    output_dir.mkdir(parents=True, exist_ok=True)
    first_prepared = prepared_by_id[candidate_experiment_ids[0]]
    patient_ids = np.asarray(first_prepared["patient_ids"]).astype(str)
    labels = np.asarray(first_prepared["y"], dtype=np.int64)
    order = np.argsort(patient_ids)
    patient_ids = patient_ids[order]
    labels = labels[order]
    patient_to_group = {
        str(row["patient_id"]): str(row["duplicate_component_id"])
        for row in base_fold_manifest_rows
    }

    rows = []
    aggregate_selection_counts = {
        experiment_id: 0 for experiment_id in candidate_experiment_ids
    }
    for repeat_index in range(REPEATED_NESTED_CV_REPEATS):
        seed = REPEATED_NESTED_CV_RANDOM_STATE + repeat_index
        fold_rows = _build_fold_manifest_rows_for_seed(
            patient_ids, labels, patient_to_group, seed
        )
        selected = _run_nested_candidate_selection_once(
            candidate_experiment_ids,
            experiments_by_id,
            prepared_by_id,
            fold_rows,
            verbose=False,
        )
        auc = float(roc_auc_score(selected["labels"], selected["scores"]))
        auprc = float(
            average_precision_score(selected["labels"], selected["scores"])
        )
        fold_models = [
            row["selected_experiment_id"]
            for row in selected["outer_fold_rows"]
        ]
        per_repeat_counts = {
            experiment_id: int(fold_models.count(experiment_id))
            for experiment_id in candidate_experiment_ids
        }
        for experiment_id, count in per_repeat_counts.items():
            aggregate_selection_counts[experiment_id] += int(count)
        rows.append(
            {
                "repeat_index": int(repeat_index + 1),
                "outer_cv_random_state": int(seed),
                "selection_aware_auc": auc,
                "selection_aware_auprc": auprc,
                "selected_model_counts_json": json.dumps(
                    per_repeat_counts, sort_keys=True
                ),
            }
        )
        print(
            f"[MODEL SELECTION][STABILITY] repeat={repeat_index + 1}/"
            f"{REPEATED_NESTED_CV_REPEATS}, AUC={auc:.4f}",
            flush=True,
        )

    auc_values = np.asarray(
        [row["selection_aware_auc"] for row in rows], dtype=np.float64
    )
    summary = {
        "status": "OK",
        "repeats": int(REPEATED_NESTED_CV_REPEATS),
        "candidate_experiment_ids_in_priority_order": list(
            candidate_experiment_ids
        ),
        "auc_mean": float(np.mean(auc_values)),
        "auc_median": float(np.median(auc_values)),
        "auc_std": float(np.std(auc_values)),
        "auc_q25": float(np.quantile(auc_values, 0.25)),
        "auc_q75": float(np.quantile(auc_values, 0.75)),
        "auc_minimum": float(np.min(auc_values)),
        "auc_maximum": float(np.max(auc_values)),
        "selected_model_counts_across_all_repeat_folds": (
            aggregate_selection_counts
        ),
        "interpretation": (
            "Repeated outer seeds describe split sensitivity; they are not "
            "independent samples or external validation."
        ),
    }
    with open(
        output_dir / "repeated_candidate_selection_runs.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "repeated_candidate_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def run_selection_adjusted_patient_label_permutation_test(
    candidate_experiment_ids,
    experiments_by_id,
    prepared_by_id,
    base_fold_manifest_rows,
    observed_auc,
    output_dir,
):
    """Build a null that repeats candidate identity and C selection each time."""

    output_dir.mkdir(parents=True, exist_ok=True)
    first_prepared = prepared_by_id[candidate_experiment_ids[0]]
    patient_ids = np.asarray(first_prepared["patient_ids"]).astype(str)
    original_labels = np.asarray(first_prepared["y"], dtype=np.int64)
    order = np.argsort(patient_ids)
    patient_ids = patient_ids[order]
    original_labels = original_labels[order]
    patient_to_group = {
        str(row["patient_id"]): str(row["duplicate_component_id"])
        for row in base_fold_manifest_rows
    }
    rng = np.random.default_rng(SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE)
    rows = []
    progress_interval = max(1, SELECTION_ADJUSTED_PERMUTATION_REPLICATES // 10)

    print(
        f"[SELECTION-ADJUSTED PERMUTATION] Running "
        f"{SELECTION_ADJUSTED_PERMUTATION_REPLICATES} patient-label "
        "permutations with candidate and C selection repeated inside every "
        "outer-training fold.",
        flush=True,
    )

    for permutation_index in range(SELECTION_ADJUSTED_PERMUTATION_REPLICATES):
        permuted_labels = rng.permutation(original_labels)
        label_by_patient = {
            str(patient_id): int(label)
            for patient_id, label in zip(patient_ids, permuted_labels)
        }
        permuted_prepared_by_id = {
            experiment_id: _clone_prepared_with_patient_label_mapping(
                prepared_by_id[experiment_id], label_by_patient
            )
            for experiment_id in candidate_experiment_ids
        }
        # Reuse the same splitter random state for every null replicate. Fold
        # membership can still change because stratification sees permuted
        # labels, but no extra variation is injected by changing the seed.
        fold_rows = _build_fold_manifest_rows_for_seed(
            patient_ids,
            permuted_labels,
            patient_to_group,
            CV_RANDOM_STATE,
        )
        selected = _run_nested_candidate_selection_once(
            candidate_experiment_ids,
            experiments_by_id,
            permuted_prepared_by_id,
            fold_rows,
            verbose=False,
        )
        auc = float(roc_auc_score(selected["labels"], selected["scores"]))
        fold_models = [
            row["selected_experiment_id"]
            for row in selected["outer_fold_rows"]
        ]
        rows.append(
            {
                "permutation_index": int(permutation_index + 1),
                "selection_aware_permuted_auc": auc,
                "selected_model_counts_json": json.dumps(
                    {
                        experiment_id: int(fold_models.count(experiment_id))
                        for experiment_id in candidate_experiment_ids
                    },
                    sort_keys=True,
                ),
            }
        )
        if (
            permutation_index == 0
            or (permutation_index + 1) % progress_interval == 0
            or permutation_index + 1
            == SELECTION_ADJUSTED_PERMUTATION_REPLICATES
        ):
            print(
                f"[SELECTION-ADJUSTED PERMUTATION] {permutation_index + 1}/"
                f"{SELECTION_ADJUSTED_PERMUTATION_REPLICATES}; latest AUC="
                f"{auc:.4f}",
                flush=True,
            )

    null_aucs = np.asarray(
        [row["selection_aware_permuted_auc"] for row in rows],
        dtype=np.float64,
    )
    p_value = float(
        (1 + np.sum(null_aucs >= float(observed_auc)))
        / (SELECTION_ADJUSTED_PERMUTATION_REPLICATES + 1)
    )
    summary = {
        "status": "OK",
        "candidate_experiment_ids_in_priority_order": list(
            candidate_experiment_ids
        ),
        "observed_selection_aware_auc": float(observed_auc),
        "permutation_replicates": int(
            SELECTION_ADJUSTED_PERMUTATION_REPLICATES
        ),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_median": float(np.median(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
        "null_auc_q025": float(np.quantile(null_aucs, 0.025)),
        "null_auc_q975": float(np.quantile(null_aucs, 0.975)),
        "empirical_one_sided_p_value": p_value,
        "permutation_unit": "Directory_* patient labels",
        "selection_repeated_inside_each_permutation": True,
        "interpretation": (
            "This null repeats only the compact predeclared focused candidate "
            "family. It does not retroactively correct every historical idea."
        ),
    }
    with open(
        output_dir / "selection_adjusted_permutation_auc.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selection_adjusted_permutation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[SELECTION-ADJUSTED PERMUTATION] observed AUC="
        f"{observed_auc:.4f}, null median={summary['null_auc_median']:.4f}, "
        f"empirical p={p_value:.6f}",
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

    Unlike the orientation table, which compares every successful experiment
    with the primary candidate, this table uses the predeclared reference chosen
    for each scientific question. The focused registry therefore reports only
    direct candidate, reference, robustness and falsification comparisons.
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
        "slice_filter": str(experiment.slice_filter),
        "is_primary_candidate": int(
            experiment.experiment_id == PRIMARY_CANDIDATE_EXPERIMENT_ID
        ),
        "is_future_candidate": int(
            experiment.experiment_id in FUTURE_CANDIDATE_EXPERIMENT_IDS
        ),
        "deduplicate_exact_within_patient": int(
            experiment.deduplicate_exact_within_patient
        ),
        "n_exact_duplicate_rows_removed": result["prepared_metadata"].get(
            "n_exact_duplicate_rows_removed", 0
        ),
        "n_source_slices": result["prepared_metadata"].get(
            "n_source_slices"
        ),
        "n_after_slice_filter": result["prepared_metadata"].get(
            "n_after_slice_filter"
        ),
        "n_retained_slices": result["prepared_metadata"].get(
            "n_retained_slices"
        ),
        "retained_sample_indices_sha256": result["prepared_metadata"].get(
            "retained_sample_indices_sha256"
        ),
        "annotated_series_subset_applied": int(
            bool(
                result["prepared_metadata"].get(
                    "annotated_series_subset_applied", False
                )
            )
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


def write_focused_row_contract_audit(successful_results, output_path):
    """Verify that candidate-matched controls use the intended source rows.

    A17, C31, C32 and C33 must share the same ordered sample-index digest and
    patient table. A20, A21 and R4 intentionally alter the row set, but must
    retain the same patient cohort. A mismatch in a completed experiment is a
    fatal scientific-contract error rather than a warning.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_by_id = {
        result["config"].experiment_id: result
        for result in successful_results
        if result.get("status") == "OK"
    }
    exact_ids = (
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
        "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
        "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
    )
    missing_exact = [
        experiment_id
        for experiment_id in exact_ids
        if experiment_id not in result_by_id
    ]
    if missing_exact:
        summary = {
            "status": "INCOMPLETE_EXPERIMENT_FAILURE",
            "missing_exact_contract_experiments": missing_exact,
        }
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    reference = result_by_id[PRIMARY_CANDIDATE_EXPERIMENT_ID]
    reference_patients = np.asarray(reference["patient_ids"]).astype(str)
    reference_labels = np.asarray(reference["labels"], dtype=np.int64)
    reference_metadata = reference["prepared_metadata"]
    reference_hash = reference_metadata.get("retained_sample_indices_sha256")
    reference_rows = reference_metadata.get("n_retained_slices")
    exact_rows = []

    for experiment_id in exact_ids:
        result = result_by_id[experiment_id]
        metadata = result["prepared_metadata"]
        patient_ids = np.asarray(result["patient_ids"]).astype(str)
        labels = np.asarray(result["labels"], dtype=np.int64)
        row = {
            "experiment_id": experiment_id,
            "n_patients": int(len(patient_ids)),
            "n_retained_slices": metadata.get("n_retained_slices"),
            "retained_sample_indices_sha256": metadata.get(
                "retained_sample_indices_sha256"
            ),
            "same_patient_order_as_a17": bool(
                np.array_equal(patient_ids, reference_patients)
            ),
            "same_labels_as_a17": bool(
                np.array_equal(labels, reference_labels)
            ),
            "same_rows_as_a17": bool(
                metadata.get("n_retained_slices") == reference_rows
                and metadata.get("retained_sample_indices_sha256")
                == reference_hash
            ),
        }
        exact_rows.append(row)
        if not (
            row["same_patient_order_as_a17"]
            and row["same_labels_as_a17"]
            and row["same_rows_as_a17"]
        ):
            raise RuntimeError(
                "Focused exact-control row contract failed for "
                f"{experiment_id}: {row}."
            )

    sensitivity_rows = []
    for experiment_id in (
        VALID_ONLY_CANDIDATE_EXPERIMENT_ID,
        "A21_HARD_SUPPORT_REGION_NORM_DEDUP_HIER_LR_PCA",
        "R4_HARD_SUPPORT_REGION_NORM_DROP50_HIER_LR_PCA",
    ):
        result = result_by_id.get(experiment_id)
        if result is None:
            sensitivity_rows.append(
                {"experiment_id": experiment_id, "status": "UNAVAILABLE"}
            )
            continue
        patient_ids = np.asarray(result["patient_ids"]).astype(str)
        labels = np.asarray(result["labels"], dtype=np.int64)
        metadata = result["prepared_metadata"]
        same_patients = np.array_equal(patient_ids, reference_patients)
        same_labels = np.array_equal(labels, reference_labels)
        retained_rows = metadata.get("n_retained_slices")
        if not same_patients or not same_labels:
            raise RuntimeError(
                f"{experiment_id} changed the patient cohort or labels."
            )
        if retained_rows is not None and reference_rows is not None and (
            retained_rows > reference_rows
        ):
            raise RuntimeError(
                f"{experiment_id} retained more rows than A17 unexpectedly."
            )
        sensitivity_rows.append(
            {
                "experiment_id": experiment_id,
                "status": "PASS",
                "same_patient_order_as_a17": bool(same_patients),
                "same_labels_as_a17": bool(same_labels),
                "n_retained_slices": retained_rows,
                "retained_sample_indices_sha256": metadata.get(
                    "retained_sample_indices_sha256"
                ),
            }
        )

    summary = {
        "status": "PASS",
        "feature_bank_row_identity": (
            "Ordered immutable sample_index SHA-256 within one feature-bank "
            "fingerprint."
        ),
        "exact_a17_control_family": exact_rows,
        "intentional_row_perturbations": sensitivity_rows,
    }
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[FOCUSED ROW CONTRACT] A17/C31/C32/C33 share {reference_rows} "
        "ordered source rows and one patient cohort.",
        flush=True,
    )
    return summary


def write_stability_ranking(summary_rows, stability_summaries, output_path):
    """Write the preferred ranking based on median repeated nested-CV AUC."""

    summary_lookup = {row["experiment_id"]: row for row in summary_rows}
    rows = []
    for experiment_id in STABILITY_EXPERIMENT_IDS:
        single = summary_lookup.get(experiment_id, {})
        stability = stability_summaries.get(experiment_id, {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "role": single.get("role", ""),
                "is_primary_candidate": int(
                    experiment_id == PRIMARY_CANDIDATE_EXPERIMENT_ID
                ),
                "is_future_candidate": int(
                    experiment_id in FUTURE_CANDIDATE_EXPERIMENT_IDS
                ),
                "single_split_auc": single.get("auc", ""),
                "repeated_cv_status": stability.get(
                    "status", "NOT_AVAILABLE"
                ),
                "repeated_auc_median": stability.get("auc_median", ""),
                "repeated_auc_q25": stability.get("auc_q25", ""),
                "repeated_auc_q75": stability.get("auc_q75", ""),
                "repeated_auc_mean": stability.get("auc_mean", ""),
                "repeated_auc_std": stability.get("auc_std", ""),
                "repeated_auc_minimum": stability.get("auc_minimum", ""),
                "repeated_auc_maximum": stability.get("auc_maximum", ""),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["repeated_cv_status"] == "OK" else 1,
            -float(row["repeated_auc_median"])
            if row["repeated_cv_status"] == "OK"
            else float("inf"),
            row["experiment_id"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[FOCUSED STABILITY RANKING] median repeated nested-CV AUC", flush=True)
    for rank, row in enumerate(rows, start=1):
        if row["repeated_cv_status"] != "OK":
            continue
        marker = " [PRIMARY]" if row["is_primary_candidate"] else ""
        print(
            f"  {rank:02d}. {row['experiment_id']}{marker}: median="
            f"{float(row['repeated_auc_median']):.4f}, IQR=["
            f"{float(row['repeated_auc_q25']):.4f}, "
            f"{float(row['repeated_auc_q75']):.4f}]",
            flush=True,
        )
    return rows


def _focused_repeated_comparison_status(row, mode):
    """Assign a transparent descriptive status to one repeated comparison."""

    if not row or row.get("status") != "OK":
        return "NOT_AVAILABLE"
    median_delta = float(row["delta_auc_median"])
    better_fraction = float(row["comparison_better_fraction"])
    reference_better_fraction = float(row["reference_better_fraction"])

    if mode == "robustness":
        if abs(median_delta) <= FOCUSED_ROBUSTNESS_AUC_TOLERANCE:
            return "STABLE_WITHIN_PREDECLARED_AUC_TOLERANCE"
        if median_delta > 0.0:
            return "PERTURBATION_IMPROVED_MEDIAN_AUC"
        return "SENSITIVE_TO_PERTURBATION"

    if (
        median_delta > 0.0
        and better_fraction >= FOCUSED_REPEATED_SUPPORT_FRACTION
    ):
        return "SUPPORTED_ACROSS_SPLIT_SEEDS"
    if (
        median_delta < 0.0
        and reference_better_fraction >= FOCUSED_REPEATED_SUPPORT_FRACTION
    ):
        return "EVIDENCE_FAVORS_REFERENCE"
    return "MIXED_SPLIT_SENSITIVITY"


def print_focused_candidate_selection_summary(
    model_selection_summary,
    selection_adjusted_permutation_summary,
):
    """Print the compact procedure-level validation result."""

    print("\nFOCUSED CANDIDATE-SELECTION AUDIT", flush=True)
    if model_selection_summary.get("status") == "OK":
        metrics = model_selection_summary.get("metrics", {})
        print(
            "[MODEL SELECTION] OOF AUC="
            f"{float(metrics.get('auc', float('nan'))):.4f}; selections="
            f"{model_selection_summary.get('selected_model_counts_across_outer_folds', {})}.",
            flush=True,
        )
        repeated = model_selection_summary.get(
            "repeated_candidate_selection_stability", {}
        )
        if repeated.get("status") == "OK":
            print(
                "[MODEL SELECTION] repeated median AUC="
                f"{float(repeated['auc_median']):.4f}, IQR=["
                f"{float(repeated['auc_q25']):.4f}, "
                f"{float(repeated['auc_q75']):.4f}].",
                flush=True,
            )
    else:
        print(
            f"[MODEL SELECTION] status={model_selection_summary.get('status', 'UNKNOWN')}.",
            flush=True,
        )

    if selection_adjusted_permutation_summary.get("status") == "OK":
        print(
            "[SELECTION-ADJUSTED PERMUTATION] observed AUC="
            f"{float(selection_adjusted_permutation_summary['observed_selection_aware_auc']):.4f}; "
            "null median="
            f"{float(selection_adjusted_permutation_summary['null_auc_median']):.4f}; "
            "empirical p="
            f"{float(selection_adjusted_permutation_summary['empirical_one_sided_p_value']):.6f}.",
            flush=True,
        )
    else:
        print(
            "[SELECTION-ADJUSTED PERMUTATION] status="
            f"{selection_adjusted_permutation_summary.get('status', 'UNKNOWN')}.",
            flush=True,
        )


def write_focused_research_scorecard(
    summary_rows,
    stability_summary,
    primary_ablation_rows,
    model_selection_summary,
    permutation_summary,
    series_annotation_summary,
    external_status,
    output_dir,
):
    """Write a concise evidence map suitable for a research supplement.

    The scorecard does not convert internal AUROC into a clinical or admissions
    claim. It records what each essential experiment can and cannot establish,
    identifies residual confounding, and exposes missing sequence/view and
    external-validation evidence.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lookup = {row["experiment_id"]: row for row in summary_rows}
    single_pair_lookup = {
        row["comparison_name"]: row for row in primary_ablation_rows
    }
    repeated_payload = stability_summary.get("paired_comparisons", {})
    repeated_pair_lookup = {
        row["comparison_name"]: row
        for row in repeated_payload.get("comparisons", [])
    }
    repeated_experiment_lookup = stability_summary.get(
        "experiment_summaries", {}
    )

    def comparison_row(
        evidence_id,
        question,
        comparison_name,
        repeated_name,
        mode="superiority",
    ):
        single = single_pair_lookup.get(comparison_name, {})
        repeated = repeated_pair_lookup.get(repeated_name, {})
        reference_id = single.get(
            "reference_experiment_id", repeated.get("reference_experiment_id", "")
        )
        changed_id = single.get(
            "comparison_experiment_id", repeated.get("comparison_experiment_id", "")
        )
        reference_stability = repeated_experiment_lookup.get(reference_id, {})
        changed_stability = repeated_experiment_lookup.get(changed_id, {})
        return {
            "evidence_id": evidence_id,
            "research_question": question,
            "reference_experiment_id": reference_id,
            "changed_experiment_id": changed_id,
            "single_split_reference_auc": single.get("reference_auc", ""),
            "single_split_changed_auc": single.get("comparison_auc", ""),
            "single_split_delta_auc": single.get("delta_auc", ""),
            "single_split_delta_ci_lower": single.get(
                "delta_auc_ci_lower", ""
            ),
            "single_split_delta_ci_upper": single.get(
                "delta_auc_ci_upper", ""
            ),
            "repeated_reference_median_auc": reference_stability.get(
                "auc_median", ""
            ),
            "repeated_changed_median_auc": changed_stability.get(
                "auc_median", ""
            ),
            "repeated_median_delta_auc": repeated.get(
                "delta_auc_median", ""
            ),
            "repeated_delta_q25": repeated.get("delta_auc_q25", ""),
            "repeated_delta_q75": repeated.get("delta_auc_q75", ""),
            "changed_better_fraction": repeated.get(
                "comparison_better_fraction", ""
            ),
            "evidence_status": _focused_repeated_comparison_status(
                repeated, mode
            ),
            "interpretation_scope": (
                "Internal patient-level repeated nested CV on the same released "
                "30-patient cohort; not external or clinical validation."
            ),
        }

    rows = [
        comparison_row(
            "E01_PRIMARY_VS_HISTORICAL_REFERENCE",
            "Does A17 improve over locked historical A12?",
            "A12_REFERENCE_VS_A17_PRIMARY",
            "A12_REFERENCE_VS_A17_PRIMARY",
        ),
        comparison_row(
            "E02_LOCALIZATION_VS_FULL_IMAGE",
            "Does cardiac localization outperform the standardized full image?",
            "A9_FULL_IMAGE_VS_A17_PRIMARY",
            "A9_FULL_IMAGE_VS_A17_PRIMARY",
        ),
        comparison_row(
            "E03_SUPPORT_SILHOUETTE",
            "Does A17 outperform a fixed-FOV crop that suppresses the support outline?",
            "A16_SHAPE_SUPPRESSED_CROP_VS_A17_SUPPORT",
            "A16_FIXED_FOV_VS_A17_SUPPORT",
        ),
        comparison_row(
            "E04_GATE_INVALID_FALLBACK",
            "Is A17 stable after excluding gate-invalid slices?",
            "A17_ALL_SLICES_VS_A20_VALID_ONLY",
            "A17_ALL_VS_A20_VALID_ONLY",
            mode="robustness",
        ),
        comparison_row(
            "E05_EXACT_WITHIN_PATIENT_DUPLICATES",
            "Is A17 stable after exact within-patient deduplication?",
            "A17_VS_A21_EXACT_DEDUP",
            "A17_VS_A21_EXACT_DEDUP",
            mode="robustness",
        ),
        comparison_row(
            "E06_HALF_SLICE_DROPOUT",
            "Is A17 stable after removing half the slices inside each series proxy?",
            "A17_VS_R4_DROP50",
            "A17_VS_R4_DROP50",
            mode="robustness",
        ),
        comparison_row(
            "E07_INTENSITY_BEYOND_SUPPORT_GEOMETRY",
            "Does MRI intensity add information beyond A17's exact support geometry?",
            "C31_EXACT_SUPPORT_MASK_ONLY_VS_A17",
            "C31_EXACT_SUPPORT_ONLY_VS_A17",
        ),
        comparison_row(
            "E08_SPATIAL_TEXTURE_BEYOND_HISTOGRAM",
            "Does intact spatial texture add information beyond support and intensity multiset?",
            "C32_SHUFFLED_INTENSITY_VS_A17",
            "C32_SHUFFLED_INTENSITY_VS_A17",
        ),
        comparison_row(
            "E09_INSIDE_VS_EXACT_COMPLEMENT",
            "Does A17 outperform its independently normalized exact complement?",
            "C33_EXACT_COMPLEMENT_VS_A17",
            "C33_EXACT_COMPLEMENT_VS_A17",
        ),
        comparison_row(
            "E10_INSIDE_VS_CONSERVATIVE_OUTSIDE",
            "Does A17 outperform a conservative outside-whole-heart proxy?",
            "C28_CONSERVATIVE_OUTSIDE_VS_A17",
            "C28_CONSERVATIVE_OUTSIDE_VS_A17",
        ),
        comparison_row(
            "E11_INSIDE_VS_FIXED_PERIPHERY",
            "Does A17 outperform a MONAI-independent fixed periphery?",
            "C29_FIXED_PERIPHERY_VS_A17",
            "C29_FIXED_PERIPHERY_VS_A17",
        ),
        comparison_row(
            "E12_EXPORT_PROVENANCE",
            "Does A17 outperform export/provenance metadata alone?",
            "C3_EXPORT_PROVENANCE_VS_A17",
            "C3_EXPORT_PROVENANCE_VS_A17",
        ),
        comparison_row(
            "E13_SERIES_COUNT",
            "Does A17 outperform exported series count alone?",
            "C17_SERIES_COUNT_VS_A17",
            "C17_SERIES_COUNT_VS_A17",
        ),
        comparison_row(
            "E14_NATIVE_GEOMETRY",
            "Does A17 outperform native geometry alone?",
            "C19_NATIVE_GEOMETRY_VS_A17",
            "C19_NATIVE_GEOMETRY_VS_A17",
        ),
        comparison_row(
            "E15_FILE_SIZE",
            "Does A17 outperform file-size/compression proxies alone?",
            "C20_FILE_SIZE_VS_A17",
            "C20_FILE_SIZE_VS_A17",
        ),
    ]

    # Future candidates are appended without rewriting the fixed evidence IDs.
    # Their generic references are created automatically; representation-specific
    # controls come from FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.
    for future_index, candidate_id in enumerate(
        FUTURE_CANDIDATE_EXPERIMENT_IDS,
        start=1,
    ):
        safe_candidate = "".join(
            character if character.isalnum() else "_"
            for character in candidate_id
        )
        rows.extend(
            [
                comparison_row(
                    f"F{future_index:02d}_VS_A17",
                    f"Does {candidate_id} improve over locked A17?",
                    f"A17_PRIMARY_VS_FUTURE__{candidate_id}",
                    f"A17_PRIMARY_VS_FUTURE__{candidate_id}",
                ),
                comparison_row(
                    f"F{future_index:02d}_VS_A12",
                    f"Does {candidate_id} improve over locked historical A12?",
                    f"A12_REFERENCE_VS_FUTURE__{candidate_id}",
                    f"A12_REFERENCE_VS_FUTURE__{candidate_id}",
                ),
                comparison_row(
                    f"F{future_index:02d}_VS_A9_FULL",
                    f"Does {candidate_id} improve over the standardized full image?",
                    f"A9_FULL_IMAGE_VS_FUTURE__{candidate_id}",
                    f"A9_FULL_IMAGE_VS_FUTURE__{candidate_id}",
                ),
            ]
        )
        for control_index, control_id in enumerate(
            FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.get(candidate_id, ()),
            start=1,
        ):
            comparison_name = (
                f"FUTURE_CONTROL__{control_id}__VS__{candidate_id}"
            )
            rows.append(
                comparison_row(
                    f"F{future_index:02d}_MATCHED_{control_index:02d}_{safe_candidate}",
                    f"Does {candidate_id} outperform matched control {control_id}?",
                    comparison_name,
                    comparison_name,
                )
            )

    # Mark residual shortcut signal separately from the candidate-control gap.
    nuisance_ids = tuple(
        dict.fromkeys(
            (
                "C28_OUTSIDE_WHOLE_HEART_REGION_NORM_HIER_LR_PCA",
                "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
                "C3_EXPORT_PROVENANCE_ONLY_LR",
                "C17_N_SERIES_ONLY_LR",
                "C19_NATIVE_GEOMETRY_ONLY_LR",
                "C20_FILE_SIZE_ONLY_LR",
            )
            + tuple(
                experiment_id
                for experiment_id in FUTURE_CONTROL_EXPERIMENT_IDS
                if summary_lookup.get(experiment_id, {}).get("role")
                == "negative_control"
            )
        )
    )
    nuisance_rows = []
    for experiment_id in nuisance_ids:
        single = summary_lookup.get(experiment_id, {})
        stability = repeated_experiment_lookup.get(experiment_id, {})
        value = stability.get("auc_median", single.get("auc"))
        warning = bool(
            value not in (None, "") and float(value) >= SHORTCUT_WARNING_AUC
        )
        nuisance_rows.append(
            {
                "experiment_id": experiment_id,
                "single_split_auc": single.get("auc", ""),
                "repeated_median_auc": stability.get("auc_median", ""),
                "warning_threshold": float(SHORTCUT_WARNING_AUC),
                "residual_signal_warning": warning,
            }
        )

    primary = summary_lookup.get(PRIMARY_CANDIDATE_EXPERIMENT_ID, {})
    primary_stability = repeated_experiment_lookup.get(
        PRIMARY_CANDIDATE_EXPERIMENT_ID, {}
    )
    ordinary_permutation = permutation_summary.get(
        "experiment_summaries", {}
    ).get(PRIMARY_CANDIDATE_EXPERIMENT_ID, {})
    adjusted_permutation = permutation_summary.get(
        "selection_adjusted_candidate_family", {}
    )

    scorecard = {
        "status": "COMPLETE" if primary else "INCOMPLETE_PRIMARY_FAILED",
        "purpose": (
            "Compact reproducibility and validity evidence for a research "
            "supplement or portfolio. It must not be presented as clinical "
            "validation, publication, or proof of CAD-specific causality."
        ),
        "primary_candidate_experiment_id": PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "active_experiment_ids": list(EXPERIMENTS_TO_RUN),
        "future_candidate_experiment_ids": list(
            FUTURE_CANDIDATE_EXPERIMENT_IDS
        ),
        "future_control_experiment_ids": list(FUTURE_CONTROL_EXPERIMENT_IDS),
        "future_matched_control_ids_by_candidate": {
            str(candidate_id): list(control_ids)
            for candidate_id, control_ids in (
                FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.items()
            )
        },
        "primary_single_split_metrics": {
            key: primary.get(key)
            for key in (
                "auc",
                "auc_ci_lower",
                "auc_ci_upper",
                "auprc",
                "sensitivity",
                "specificity",
                "f1",
            )
        },
        "primary_repeated_cv": primary_stability,
        "ordinary_patient_label_permutation": ordinary_permutation,
        "nested_candidate_selection": model_selection_summary,
        "selection_adjusted_patient_label_permutation": adjusted_permutation,
        "paired_evidence": rows,
        "nuisance_control_signal": nuisance_rows,
        "residual_confounding_warning": bool(
            any(row["residual_signal_warning"] for row in nuisance_rows)
        ),
        "sequence_view_analysis": series_annotation_summary,
        "external_validation": external_status,
        "claim_boundary": (
            "The suite can establish robust internal association with released "
            "Normal/Sick labels and test specified shortcut hypotheses. It cannot "
            "establish CAD-specific clinical validity without verified sequence/"
            "view analysis and an independent external cohort."
        ),
    }

    json_path = output_dir / "focused_research_scorecard.json"
    csv_path = output_dir / "focused_research_scorecard.csv"
    markdown_path = output_dir / "focused_research_evidence_summary.md"
    json_path.write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_lines = [
        "# Focused CAD Cardiac MRI Research Evidence Summary",
        "",
        f"- Primary candidate: `{PRIMARY_CANDIDATE_EXPERIMENT_ID}`",
        f"- Active experiments: {len(EXPERIMENTS_TO_RUN)}",
        f"- Patients: {primary.get('n_patients', 'unavailable')}",
        f"- Single-split pooled OOF AUROC: {primary.get('auc', 'unavailable')}",
        f"- Repeated-CV median AUROC: {primary_stability.get('auc_median', 'unavailable')}",
        f"- Residual confounding warning: {scorecard['residual_confounding_warning']}",
        f"- Sequence/view analysis status: {series_annotation_summary.get('status', 'UNKNOWN')}",
        f"- External validation status: {external_status.get('status', 'UNKNOWN')}",
        "",
        "## Evidence questions",
        "",
        "| Evidence | Status | Repeated median ΔAUROC | Changed better fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["repeated_median_delta_auc"]
        fraction = row["changed_better_fraction"]
        markdown_lines.append(
            f"| {row['evidence_id']} | {row['evidence_status']} | "
            f"{delta if delta != '' else 'NA'} | "
            f"{fraction if fraction != '' else 'NA'} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Required interpretation boundary",
            "",
            scorecard["claim_boundary"],
            "",
            "Repeated split runs reuse the same small cohort and are descriptive "
            "sensitivity analyses, not independent replications.",
        ]
    )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(
        f"[FOCUSED SCORECARD] Wrote {json_path}, {csv_path}, and "
        f"{markdown_path}.",
        flush=True,
    )
    return scorecard


def augment_focused_final_report(
    comparison_dir,
    focused_scorecard,
    stability_summary,
    model_selection_summary,
    permutation_summary,
    series_annotation_summary,
    external_status,
    focused_row_contract_summary,
):
    """Attach late-stage validation evidence to the stage-9 master report.

    ``write_master_outputs`` runs before repeated CV, permutation, optional
    sequence/view analysis and external-validation status are available. This
    function performs one deterministic final merge so reviewers can inspect a
    single JSON file without losing the standalone detailed artifacts.
    """

    report_path = Path(comparison_dir) / "final_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}
    report.update(
        {
            "focused_research_scorecard": focused_scorecard,
            "repeated_nested_cv_stability": stability_summary,
            "nested_candidate_selection": model_selection_summary,
            "patient_label_permutation_tests": permutation_summary,
            "focused_row_contract": focused_row_contract_summary,
            "annotated_sequence_view_analysis": series_annotation_summary,
            "external_validation": external_status,
            "preferred_evidence_order": [
                "A17 repeated-CV median and IQR",
                "same-seed A17 versus A12/A9/A16 deltas",
                "A17 versus C31/C32/C33 exact matched controls",
                "A17 versus C28/C29 and metadata nuisance controls",
                "A20/A21/R4 robustness",
                "nested candidate-selection performance",
                "ordinary and selection-adjusted permutation tests",
                "annotated sequence/view result",
                "independent external validation",
            ],
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


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
        "suite_type": "focused_research_validation",
        "focused_suite_version": FOCUSED_SUITE_VERSION,
        "focused_suite_profile": FOCUSED_SUITE_PROFILE,
        "active_experiment_count": int(len(EXPERIMENTS_TO_RUN)),
        "active_experiment_ids": list(EXPERIMENTS_TO_RUN),
        "primary_candidate_experiment_id": PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "valid_only_candidate_experiment_id": (
            VALID_ONLY_CANDIDATE_EXPERIMENT_ID
        ),
        "historical_locked_reference_experiment_id": (
            V4_LOCKED_REFERENCE_EXPERIMENT_ID
        ),
        "full_image_reference_experiment_id": (
            DEVELOPMENT_BASELINE_EXPERIMENT_ID
        ),
        "successful_experiments": summary_rows,
        "failed_experiments": failed_results,
        "paired_auc_comparisons_vs_primary_candidate": paired_rows,
        "predeclared_focused_comparisons": primary_ablation_rows,
        "experiment_groups": {
            "candidate_and_sensitivity": [
                PRIMARY_CANDIDATE_EXPERIMENT_ID,
                VALID_ONLY_CANDIDATE_EXPERIMENT_ID,
                "A21_HARD_SUPPORT_REGION_NORM_DEDUP_HIER_LR_PCA",
                "R4_HARD_SUPPORT_REGION_NORM_DROP50_HIER_LR_PCA",
            ],
            "historical_and_localization_references": [
                V4_LOCKED_REFERENCE_EXPERIMENT_ID,
                "A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA",
                DEVELOPMENT_BASELINE_EXPERIMENT_ID,
            ],
            "exact_candidate_matched_controls": [
                "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
                "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
                "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
            ],
            "broader_extracardiac_controls": [
                "C28_OUTSIDE_WHOLE_HEART_REGION_NORM_HIER_LR_PCA",
                "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
            ],
            "protocol_and_export_controls": [
                "C3_EXPORT_PROVENANCE_ONLY_LR",
                "C17_N_SERIES_ONLY_LR",
                "C19_NATIVE_GEOMETRY_ONLY_LR",
                "C20_FILE_SIZE_ONLY_LR",
            ],
        },
        "interpretation_rules": {
            "negative_control_warning_auc": SHORTCUT_WARNING_AUC,
            "single_split_delta_auc_ci": (
                "A paired bootstrap interval containing zero does not establish "
                "a reliable difference. Repeated same-seed split sensitivity "
                "must also be inspected."
            ),
            "exact_a17_control_contract": (
                "A17, C31, C32 and C33 must use the same source rows, patient "
                "order and exact final support definition. C31 removes MRI "
                "intensity, C32 preserves the visible intensity multiset while "
                "destroying spatial arrangement, and C33 retains the independently "
                "normalized exact support complement."
            ),
            "nuisance_control_rule": (
                "A candidate-control gap does not erase residual confounding. "
                "Any negative control at or above the warning threshold must be "
                "reported as evidence that the released cohort remains predictable "
                "from protocol, export, geometry or extracardiac information."
            ),
            "repeated_cv_rule": (
                "The 50 split-seed runs are descriptive sensitivity analyses on "
                "the same small cohort, not 50 independent replications."
            ),
            "calibration_warning": (
                "Class-weighted Logistic-Regression outputs are model scores, "
                "not externally calibrated clinical probabilities."
            ),
            "sequence_view_requirement": (
                "CAD-specific interpretation requires the blinded sequence/view "
                "subset analysis with localizers and derived exports removed; "
                "folder names are never treated as validated sequence labels."
            ),
            "external_validation_requirement": (
                "Generalization requires a compatible independent cohort with a "
                "documented patient unit, endpoint and locked preprocessing adapter."
            ),
            "claim_boundary": (
                "The focused suite evaluates robust internal association with the "
                "released Normal/Sick labels and falsifies specified shortcut "
                "hypotheses. It does not by itself establish CAD-specific clinical "
                "validity, causality, publication status or external generalization."
            ),
        },
        "future_candidate_protocol": {
            "registration_list": "FUTURE_CANDIDATE_EXPERIMENT_IDS",
            "control_list": "FUTURE_CONTROL_EXPERIMENT_IDS",
            "required_generic_comparisons": [
                "future candidate versus locked A17",
                "future candidate versus independent A12 reference",
            ],
            "additional_requirement": (
                "Every representation with a new support or masking contract must "
                "define its own exact matched control before results are inspected."
            ),
        },
    }
    (comparison_dir / "final_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_rows


def print_final_comparison(summary_rows, failed_results=None):
    """Print the compact research panel in scientifically meaningful groups.

    The focused artifact intentionally avoids a single mixed leaderboard. A
    negative control with a high AUC is a warning, not a candidate model, and a
    robustness perturbation is interpreted relative to A17 rather than by its
    absolute rank. Every successful active experiment appears exactly once.
    """

    if failed_results is None:
        failed_results = []

    rows_by_id = {row["experiment_id"]: row for row in summary_rows}
    candidate_ids = {
        PRIMARY_CANDIDATE_EXPERIMENT_ID,
        VALID_ONLY_CANDIDATE_EXPERIMENT_ID,
        "A21_HARD_SUPPORT_REGION_NORM_DEDUP_HIER_LR_PCA",
        "R4_HARD_SUPPORT_REGION_NORM_DROP50_HIER_LR_PCA",
    }.union(FUTURE_CANDIDATE_EXPERIMENT_IDS)
    reference_ids = {
        V4_LOCKED_REFERENCE_EXPERIMENT_ID,
        "A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA",
        DEVELOPMENT_BASELINE_EXPERIMENT_ID,
    }
    exact_control_ids = {
        "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
        "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
        "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
    }
    extracardiac_ids = {
        "C28_OUTSIDE_WHOLE_HEART_REGION_NORM_HIER_LR_PCA",
        "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
    }
    protocol_ids = {
        "C3_EXPORT_PROVENANCE_ONLY_LR",
        "C17_N_SERIES_ONLY_LR",
        "C19_NATIVE_GEOMETRY_ONLY_LR",
        "C20_FILE_SIZE_ONLY_LR",
    }

    assigned_ids = (
        candidate_ids
        | reference_ids
        | exact_control_ids
        | extracardiac_ids
        | protocol_ids
    )
    additional_control_ids = {
        experiment_id
        for experiment_id in FUTURE_CONTROL_EXPERIMENT_IDS
        if experiment_id not in assigned_ids
    }
    assigned_ids |= additional_control_ids

    # Any future extension not recognized by the expected groups is shown in an
    # explicit final section instead of being silently dropped from the console.
    unclassified_ids = set(rows_by_id) - assigned_ids

    group_specs = (
        (
            "PRIMARY CANDIDATE AND DIRECT SENSITIVITY ANALYSES",
            candidate_ids,
        ),
        (
            "HISTORICAL AND LOCALIZATION REFERENCES",
            reference_ids,
        ),
        (
            "EXACT A17-MATCHED REPRESENTATION CONTROLS",
            exact_control_ids,
        ),
        (
            "BROADER EXTRACARDIAC CONTROLS",
            extracardiac_ids,
        ),
        (
            "PROTOCOL / EXPORT / GEOMETRY CONTROLS",
            protocol_ids,
        ),
        (
            "ADDITIONAL FUTURE CONTROLS",
            additional_control_ids,
        ),
        (
            "UNCLASSIFIED ACTIVE EXTENSIONS — REVIEW CONFIGURATION",
            unclassified_ids,
        ),
    )

    table_width = 202
    print("\n" + "=" * table_width, flush=True)
    print("FOCUSED CAD CARDIAC MRI PATIENT-LEVEL RESEARCH PANEL", flush=True)
    print("=" * table_width, flush=True)
    print(
        f"Active experiments={len(summary_rows)}; primary="
        f"{PRIMARY_CANDIDATE_EXPERIMENT_ID}; historical reference="
        f"{V4_LOCKED_REFERENCE_EXPERIMENT_ID}.",
        flush=True,
    )

    header = (
        f"{'No.':<4} {'Experiment':<74} {'Role':<36} "
        f"{'AUC [95% CI]':<25} {'Delta AUC vs A17':<31} "
        f"{'Sens.':>7} {'Spec.':>7} {'F1':>7}"
    )

    for title, experiment_ids in group_specs:
        rows = [
            row
            for row in summary_rows
            if row["experiment_id"] in experiment_ids
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: (-float(row["auc"]), row["experiment_id"]))
        print(f"\n{title}", flush=True)
        print("-" * table_width, flush=True)
        print(header, flush=True)
        print("-" * table_width, flush=True)
        for index, row in enumerate(rows, start=1):
            experiment_id = row["experiment_id"]
            if experiment_id == PRIMARY_CANDIDATE_EXPERIMENT_ID:
                display_id = f"{experiment_id} [PRIMARY]"
            elif experiment_id in FUTURE_CANDIDATE_EXPERIMENT_IDS:
                display_id = f"{experiment_id} [FUTURE]"
            else:
                display_id = experiment_id
            auc_text = (
                f"{float(row['auc']):.4f} "
                f"[{float(row['auc_ci_lower']):.4f}, "
                f"{float(row['auc_ci_upper']):.4f}]"
            )
            if row["delta_auc_vs_baseline"] is None:
                delta_text = "primary / unavailable"
            else:
                delta_text = (
                    f"{float(row['delta_auc_vs_baseline']):+.4f} "
                    f"[{float(row['delta_auc_ci_lower']):+.4f}, "
                    f"{float(row['delta_auc_ci_upper']):+.4f}]"
                )
            print(
                f"{index:<4} {display_id:<74} {row['role']:<36} "
                f"{auc_text:<25} {delta_text:<31} "
                f"{float(row['sensitivity']):>7.3f} "
                f"{float(row['specificity']):>7.3f} "
                f"{float(row['f1']):>7.3f}",
                flush=True,
            )
        print("-" * table_width, flush=True)

    print(
        "All rows reuse one patient-level outer-fold manifest. Classifier C and "
        "operating thresholds are learned only from inner out-of-fold training "
        "predictions. The single-split table is descriptive; the repeated-CV "
        "ranking and paired split-seed deltas are the preferred internal "
        "stability evidence.",
        flush=True,
    )

    negative_controls = [
        row
        for row in summary_rows
        if row["role"] == "negative_control"
        and float(row["auc"]) >= SHORTCUT_WARNING_AUC
    ]
    if negative_controls:
        print("\nRESIDUAL SHORTCUT / CONFOUNDING WARNINGS", flush=True)
        for row in sorted(
            negative_controls, key=lambda item: -float(item["auc"])
        ):
            print(
                f"[WARNING] {row['experiment_id']} achieved AUC="
                f"{float(row['auc']):.4f} >= {SHORTCUT_WARNING_AUC:.2f}. "
                "The released cohort remains predictable from a control "
                "representation; a positive A17-control gap does not eliminate "
                "that residual confounding.",
                flush=True,
            )
    else:
        print(
            "\n[CONTROL CHECK] No enabled negative control crossed the "
            f"predeclared AUC warning threshold of {SHORTCUT_WARNING_AUC:.2f}.",
            flush=True,
        )

    segmentation_controls = [
        row
        for row in summary_rows
        if row["role"] == "segmentation_representation_control"
    ]
    if segmentation_controls:
        print("\nEXACT SUPPORT / REPRESENTATION FINDINGS", flush=True)
        for row in sorted(
            segmentation_controls, key=lambda item: -float(item["auc"])
        ):
            print(
                f"[REPRESENTATION CONTROL] {row['experiment_id']}: AUC="
                f"{float(row['auc']):.4f}. Interpret it together with its "
                "predeclared paired A17 comparison; it can reflect genuine "
                "anatomy as well as segmenter, sequence, protocol or export "
                "structure.",
                flush=True,
            )

    if unclassified_ids:
        print(
            "\n[CONFIGURATION WARNING] Active experiments were not assigned to "
            f"a focused scientific group: {sorted(unclassified_ids)}.",
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
            "Detailed tracebacks are preserved in "
            "comparison/failed_experiments.csv and each experiment's "
            "failure.json.",
            flush=True,
        )

    print("=" * table_width, flush=True)

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
    model_selection_summary,
    permutation_summary,
    focused_scorecard,
    focused_row_contract_summary,
    series_annotation_summary,
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
        "focused_suite_version": FOCUSED_SUITE_VERSION,
        "focused_suite_profile": FOCUSED_SUITE_PROFILE,
        "monai_qc_missing_branch_policy": (
            MONAI_QC_MISSING_BRANCH_POLICY
        ),
        "active_experiment_ids": list(EXPERIMENTS_TO_RUN),
        "essential_experiment_ids": list(ESSENTIAL_EXPERIMENT_IDS),
        "future_candidate_experiment_ids": list(
            FUTURE_CANDIDATE_EXPERIMENT_IDS
        ),
        "future_control_experiment_ids": list(FUTURE_CONTROL_EXPERIMENT_IDS),
        "future_matched_control_ids_by_candidate": {
            str(candidate_id): list(control_ids)
            for candidate_id, control_ids in (
                FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.items()
            )
        },
        "focused_registry_experiment_count": int(len(EXPERIMENT_REGISTRY)),
        "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "primary_candidate_experiment_id": PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "valid_only_candidate_experiment_id": (
            VALID_ONLY_CANDIDATE_EXPERIMENT_ID
        ),
        "v4_locked_reference_experiment_id": (
            V4_LOCKED_REFERENCE_EXPERIMENT_ID
        ),
        "development_baseline_experiment_id": (
            DEVELOPMENT_BASELINE_EXPERIMENT_ID
        ),
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
        "active_image_feature_modes": list(
            required_efficientnet_feature_modes(get_enabled_experiments())
        ),
        "active_image_feature_mode_count": int(
            len(required_efficientnet_feature_modes(get_enabled_experiments()))
        ),
        "feature_modes_per_encoder_call": FEATURE_MODES_PER_ENCODER_CALL,
        "monai_soft_histogram_bins": MONAI_SOFT_HISTOGRAM_BINS,
        "monai_soft_block_shuffle_grid": MONAI_SOFT_BLOCK_SHUFFLE_GRID,
        "monai_soft_block_shuffle_version": MONAI_SOFT_BLOCK_SHUFFLE_VERSION,
        "monai_canonical_mask_content_fraction": (
            MONAI_CANONICAL_MASK_CONTENT_FRACTION
        ),
        "repeated_stability_comparisons": [
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
            ) in REPEATED_STABILITY_COMPARISONS
        ],
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
        "deterministic_execution": {
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "torch_deterministic_algorithms_requested": True,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": bool(
                getattr(torch.backends.cuda.matmul, "allow_tf32", False)
            ),
            "cudnn_allow_tf32": bool(
                getattr(torch.backends.cudnn, "allow_tf32", False)
            ),
        },
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
        "focused_representation_controls": {
            "exact_support_only_experiment_id": (
                "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA"
            ),
            "support_intensity_shuffled_experiment_id": (
                "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA"
            ),
            "exact_complement_experiment_id": (
                "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA"
            ),
            "support_intensity_permutation_version": (
                FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION
            ),
            "interpretation": (
                "C31 isolates exact support geometry; C32 preserves support "
                "and every visible intensity while destroying spatial order; "
                "C33 retains the independently normalized exact complement."
            ),
        },
        "repeated_nested_cv_stability": stability_summary,
        "nested_candidate_selection": model_selection_summary,
        "patient_label_permutation_test": permutation_summary,
        "focused_research_scorecard": focused_scorecard,
        "focused_row_contract": focused_row_contract_summary,
        "annotated_series_use_sequence_view_balanced_pooling": bool(
            ANNOTATED_SERIES_USE_SEQUENCE_VIEW_BALANCED_POOLING
        ),
        "annotated_series_subset_analysis": series_annotation_summary,
        "claim_boundary": (
            "Internal patient-level association and specified shortcut testing "
            "only; no CAD-specific clinical or external-generalization claim "
            "without sequence/view review and an independent cohort."
        ),
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
        "focused_suite_version": FOCUSED_SUITE_VERSION,
        "focused_suite_profile": FOCUSED_SUITE_PROFILE,
        "monai_qc_missing_branch_policy": (
            MONAI_QC_MISSING_BRANCH_POLICY
        ),
        "monai_qc_branch_requirements": required_monai_qc_branches(
            experiments
        ),
        "essential_experiment_ids": list(ESSENTIAL_EXPERIMENT_IDS),
        "future_candidate_experiment_ids": list(
            FUTURE_CANDIDATE_EXPERIMENT_IDS
        ),
        "future_control_experiment_ids": list(FUTURE_CONTROL_EXPERIMENT_IDS),
        "future_matched_control_ids_by_candidate": {
            str(candidate_id): list(control_ids)
            for candidate_id, control_ids in (
                FUTURE_MATCHED_CONTROL_IDS_BY_CANDIDATE.items()
            )
        },
        "focused_registry_experiment_count": int(len(EXPERIMENT_REGISTRY)),
        "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "primary_candidate_experiment_id": PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "valid_only_candidate_experiment_id": (
            VALID_ONLY_CANDIDATE_EXPERIMENT_ID
        ),
        "v4_locked_reference_experiment_id": (
            V4_LOCKED_REFERENCE_EXPERIMENT_ID
        ),
        "development_baseline_experiment_id": (
            DEVELOPMENT_BASELINE_EXPERIMENT_ID
        ),
        "experiments": [asdict(experiment) for experiment in experiments],
        "active_image_feature_modes": list(
            required_efficientnet_feature_modes(experiments)
        ),
        "active_image_feature_mode_count": int(
            len(required_efficientnet_feature_modes(experiments))
        ),
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
        "repeated_stability_comparisons": [
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
            ) in REPEATED_STABILITY_COMPARISONS
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
        "deterministic_execution": {
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "torch_deterministic_algorithms_requested": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
        },
        "feature_cache_schema": FEATURE_CACHE_SCHEMA_VERSION,
        "feature_modes_per_encoder_call": FEATURE_MODES_PER_ENCODER_CALL,
        "slice_quality_min_weight": SLICE_QUALITY_MIN_WEIGHT,
        "border_width_fraction": BORDER_WIDTH_FRACTION,
        "standardized_border_width_fractions": list(
            STANDARDIZED_BORDER_WIDTH_FRACTIONS
        ),
        "standardized_corner_fraction": STANDARDIZED_CORNER_WIDTH_FRACTION,
        "center_crop_fallback_fraction": CENTER_CROP_FALLBACK_FRACTION,
        "fixed_center_crop_fractions": list(FIXED_CENTER_CROP_FRACTIONS),
        "standardized_content_long_side": STANDARDIZED_CONTENT_LONG_SIDE,
        "fixed_chunk_size": FIXED_CHUNK_SIZE,
        "focused_fixed_heart_fov_fraction": FOCUSED_FIXED_HEART_FOV_FRACTION,
        "focused_hard_support_extra_dilation_kernel": (
            FOCUSED_HARD_SUPPORT_EXTRA_DILATION_KERNEL
        ),
        "focused_whole_heart_exclusion_center_fraction": (
            FOCUSED_WHOLE_HEART_EXCLUSION_CENTER_FRACTION
        ),
        "focused_whole_heart_exclusion_bbox_context_fraction": (
            FOCUSED_WHOLE_HEART_EXCLUSION_BBOX_CONTEXT_FRACTION
        ),
        "focused_fixed_periphery_exclusion_fraction": (
            FOCUSED_FIXED_PERIPHERY_EXCLUSION_FRACTION
        ),
        "focused_region_norm_lower_percentile": (
            FOCUSED_REGION_NORM_LOWER_PERCENTILE
        ),
        "focused_region_norm_upper_percentile": (
            FOCUSED_REGION_NORM_UPPER_PERCENTILE
        ),
        "focused_region_norm_min_pixels": FOCUSED_REGION_NORM_MIN_PIXELS,
        "focused_region_norm_histogram_bins": (
            FOCUSED_REGION_NORM_HISTOGRAM_BINS
        ),
        "focused_region_norm_min_dynamic_range": (
            FOCUSED_REGION_NORM_MIN_DYNAMIC_RANGE
        ),
        "focused_support_intensity_permutation_version": (
            FOCUSED_SUPPORT_INTENSITY_PERMUTATION_VERSION
        ),
        "monai_soft_histogram_bins": MONAI_SOFT_HISTOGRAM_BINS,
        "monai_soft_block_shuffle_grid": MONAI_SOFT_BLOCK_SHUFFLE_GRID,
        "monai_soft_block_shuffle_version": (
            MONAI_SOFT_BLOCK_SHUFFLE_VERSION
        ),
        "monai_canonical_mask_content_fraction": (
            MONAI_CANONICAL_MASK_CONTENT_FRACTION
        ),
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
            "experiment_ids": list(STABILITY_EXPERIMENT_IDS),
            "repeats": REPEATED_NESTED_CV_REPEATS,
            "random_state": REPEATED_NESTED_CV_RANDOM_STATE,
        },
        "patient_label_permutation_test": {
            "enabled": RUN_PATIENT_LABEL_PERMUTATION_TEST,
            "experiment_ids": list(PERMUTATION_EXPERIMENT_IDS),
            "replicates": LABEL_PERMUTATION_REPLICATES,
            "random_state": LABEL_PERMUTATION_RANDOM_STATE,
        },
        "nested_candidate_selection": {
            "enabled": RUN_NESTED_MODEL_SELECTION_AUDIT,
            "repeated_stability_enabled": (
                RUN_REPEATED_MODEL_SELECTION_STABILITY
            ),
            "candidate_experiment_ids_in_priority_order": list(
                MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
            ),
            "auc_tolerance": MODEL_SELECTION_AUC_TOLERANCE,
        },
        "selection_adjusted_patient_label_permutation": {
            "enabled": RUN_SELECTION_ADJUSTED_PERMUTATION_TEST,
            "replicates": SELECTION_ADJUSTED_PERMUTATION_REPLICATES,
            "random_state": SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE,
            "candidate_experiment_ids_in_priority_order": list(
                MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
            ),
        },
        "scorecard_evidence_rules": {
            "repeated_support_fraction": (
                FOCUSED_REPEATED_SUPPORT_FRACTION
            ),
            "robustness_auc_tolerance": (
                FOCUSED_ROBUSTNESS_AUC_TOLERANCE
            ),
            "shortcut_warning_auc": SHORTCUT_WARNING_AUC,
            "formal_hypothesis_test": False,
        },
        "annotated_series_subset_analysis": {
            "enabled": RUN_ANNOTATED_SERIES_SUBSET_ANALYSIS,
            "annotation_input_path": SERIES_ANNOTATION_INPUT_PATH,
            "selection_name": ANNOTATED_SERIES_SELECTION_NAME,
            "require_contains_heart": (
                ANNOTATED_SERIES_REQUIRE_CONTAINS_HEART
            ),
            "exclude_localizers": ANNOTATED_SERIES_EXCLUDE_LOCALIZERS,
            "exclude_derived_exports": (
                ANNOTATED_SERIES_EXCLUDE_DERIVED_EXPORTS
            ),
            "allowed_sequence_types": list(
                ANNOTATED_SERIES_ALLOWED_SEQUENCE_TYPES
            ),
            "allowed_view_types": list(
                ANNOTATED_SERIES_ALLOWED_VIEW_TYPES
            ),
            "allowed_confidence_values": list(
                ANNOTATED_SERIES_ALLOWED_CONFIDENCE_VALUES
            ),
            "sequence_view_balanced_pooling": bool(
                ANNOTATED_SERIES_USE_SEQUENCE_VIEW_BALANCED_POOLING
            ),
            "experiment_ids": list(ANNOTATED_SERIES_EXPERIMENT_IDS),
        },
        "external_validation_requested": RUN_EXTERNAL_VALIDATION,
        "external_dataset_path": EXTERNAL_DATASET_PATH,
    }
    output_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True),
        encoding="utf-8",
    )




# ---------------------------------------------------------------------------
# V7.2 EXTENSION OF THE PREDECLARED SUITE CONFIGURATION
# ---------------------------------------------------------------------------
# The original writer and every original field are preserved. This wrapper adds
# the series-harmonization contract to suite_configuration.json before image
# discovery starts, so the filtering rule is recorded independently of results.

_original_write_suite_configuration_without_series_harmonization = (
    write_suite_configuration
)


def write_suite_configuration(*args, **kwargs):
    result = _original_write_suite_configuration_without_series_harmonization(
        *args, **kwargs
    )
    output_path = args[0] if args else kwargs.get("output_path")
    if output_path is None:
        raise RuntimeError(
            "Could not identify suite_configuration.json output path in the "
            "V7.2 series-harmonization wrapper."
        )
    output_path = Path(output_path)
    if not output_path.is_file():
        raise RuntimeError(
            "The original configuration writer did not create the expected "
            f"file: {output_path}."
        )
    configuration = json.loads(output_path.read_text(encoding="utf-8"))
    configuration["automatic_cross_class_series_harmonization"] = {
        "enabled": AUTOMATIC_CROSS_CLASS_SERIES_HARMONIZATION,
        "descriptor_version": SERIES_HARMONIZATION_DESCRIPTOR_VERSION,
        "policy_version": SERIES_HARMONIZATION_POLICY_VERSION,
        "scope": SERIES_HARMONIZATION_SCOPE,
        "sample_frames_per_series": (
            SERIES_HARMONIZATION_SAMPLE_FRAMES_PER_SERIES
        ),
        "descriptor_image_size": (
            SERIES_HARMONIZATION_DESCRIPTOR_IMAGE_SIZE
        ),
        "histogram_bins": (
            SERIES_HARMONIZATION_INTENSITY_HISTOGRAM_BINS
        ),
        "dct_side": SERIES_HARMONIZATION_DCT_SIDE,
        "cluster_counts": list(SERIES_HARMONIZATION_CLUSTER_COUNTS),
        "kmeans_n_init": SERIES_HARMONIZATION_KMEANS_N_INIT,
        "random_state": SERIES_HARMONIZATION_RANDOM_STATE,
        "minimum_normal_patients_per_cluster": (
            SERIES_HARMONIZATION_MIN_NORMAL_PATIENTS_PER_CLUSTER
        ),
        "minimum_sick_patients_per_cluster": (
            SERIES_HARMONIZATION_MIN_SICK_PATIENTS_PER_CLUSTER
        ),
        "minimum_normal_series_per_cluster": (
            SERIES_HARMONIZATION_MIN_NORMAL_SERIES_PER_CLUSTER
        ),
        "minimum_sick_series_per_cluster": (
            SERIES_HARMONIZATION_MIN_SICK_SERIES_PER_CLUSTER
        ),
        "minimum_shared_resolutions": (
            SERIES_HARMONIZATION_MIN_SHARED_RESOLUTIONS
        ),
        "cluster_outlier_mad_multiplier": (
            SERIES_HARMONIZATION_CLUSTER_OUTLIER_MAD_MULTIPLIER
        ),
        "cluster_outlier_min_quantile": (
            SERIES_HARMONIZATION_CLUSTER_OUTLIER_MIN_QUANTILE
        ),
        "coverage_rescue_enabled": (
            SERIES_HARMONIZATION_ALLOW_PATIENT_COVERAGE_RESCUE
        ),
        "maximum_coverage_rescues": (
            SERIES_HARMONIZATION_MAX_COVERAGE_RESCUES
        ),
        "statistical_warning": (
            "The descriptors and clusters are label-free, but global shared "
            "support consults both class folders. Interpret this as a cohort-"
            "harmonization sensitivity analysis, not external validation."
        ),
    }
    output_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result



# =============================================================================
# V7.2-SB1 STRICT CROSS-CLASS SERIES OVERLAP REFINEMENT
# =============================================================================
#
# This block is intentionally additive. It runs AFTER V7.2's original
# multi-resolution series harmonization and BEFORE the feature bank is built.
# The existing image preprocessing, MONAI/EfficientNet representations,
# patient-level pooling, CV, controls, reports and detailed comments remain
# unchanged.
#
# WHY A SECOND, STRICTER FILTER?
# -----------------------------
# V7.2 retained 2,673 of 2,859 folder-defined series proxies and materially
# reduced the series-count, geometry, provenance and peripheral controls, but
# several controls remained predictive. Presence in a shared KMeans cluster is
# therefore not sufficient. V7.2-SB1 additionally requires each retained series
# to have close opposite-class support and then selects exactly the same number
# of series for every Directory_* patient.
#
# LABEL-USE / VALIDITY WARNING
# ----------------------------
# Per-series descriptors, robust scaling, PCA and KMeans are label-blind. Class
# labels are nevertheless used globally to decide whether clusters are shared,
# find opposite-class neighbours, and audit/balance class distributions. This is
# a PREDECLARED GLOBAL SENSITIVITY ANALYSIS, not independent validation and not
# a leakage-free estimate for unseen patients. A definitive analysis must fit
# the harmonizer inside each outer/inner training partition and apply the frozen
# rule to validation patients without using their labels.
#
# The filter never deletes a Directory_* patient silently. It first searches for
# the largest exact per-patient target that passes every strict guard. If no
# target passes, SB1.1 can continue with the best attainable exact-count
# candidate while writing the failed guards explicitly. This fallback prevents
# an avoidable Kaggle abort, but it must be reported as BEST_ATTAINABLE rather
# than as evidence that every strict balance guard was satisfied.

from sklearn.cluster import KMeans as _SB1KMeans
from sklearn.decomposition import PCA as _SB1PCA
from sklearn.neighbors import NearestNeighbors as _SB1NearestNeighbors
from sklearn.preprocessing import RobustScaler as _SB1RobustScaler

STRICT_SERIES_BALANCE_VERSION = "v7.2-sb1.1-safe-fallback-v2"
STRICT_SERIES_DESCRIPTOR_SAMPLE_COUNT = 7
STRICT_SERIES_CLUSTER_COUNTS = (47, 59, 71, 83, 97)
STRICT_SERIES_REQUIRED_SHARED_VOTES = 4
STRICT_SERIES_MIN_PATIENTS_PER_CLASS_PER_CLUSTER = 3
STRICT_SERIES_MIN_SERIES_PER_CLASS_PER_CLUSTER = 5
STRICT_SERIES_MIN_CLASS_SUPPORT_RATIO = 0.60
STRICT_SERIES_MAX_CLASS_CENTROID_GAP = 0.70
STRICT_SERIES_MAX_DISTANCE_MEDIAN_RATIO = 1.50
STRICT_SERIES_CLUSTER_DISTANCE_QUANTILE = 0.70
STRICT_SERIES_LENGTH_LOWER_QUANTILE = 0.15
STRICT_SERIES_LENGTH_UPPER_QUANTILE = 0.85
STRICT_SERIES_PCA_COMPONENTS = 24
STRICT_SERIES_OPPOSITE_NEIGHBOUR_QUANTILE = 0.70
STRICT_SERIES_MIN_PER_PATIENT = 12
STRICT_SERIES_MAX_PER_PATIENT = 32
STRICT_SERIES_TARGET_QUANTILE = 0.25
STRICT_SERIES_MAX_RESCUE_FRACTION = 0.10
STRICT_SERIES_MAX_RESCUE_PATIENTS = 6
STRICT_SERIES_MIN_RETAINED_FRACTION_OF_V72 = 0.20
STRICT_SERIES_MAX_RETAINED_FRACTION_OF_V72 = 0.75
STRICT_SERIES_PATIENT_SMD_MEDIAN_LIMIT = 0.35
STRICT_SERIES_PATIENT_SMD_Q90_LIMIT = 0.90
STRICT_SERIES_PATIENT_SMD_MAX_LIMIT = 1.75
STRICT_SERIES_RANDOM_STATE = RANDOM_SEED + 73_000

# SB1.1 keeps all original numerical guards but avoids terminating the complete
# run when their joint intersection is empty. The selected fallback is the
# feasible exact-count target with the smallest normalized guard violation; no
# AUC, model score, or downstream result enters this choice.
STRICT_SERIES_ALLOW_BEST_ATTAINABLE_FALLBACK = True

# A high-dimensional SMD maximum is unstable when a descriptor coordinate is
# nearly constant across all 30 patients. Globally constant dimensions carry no
# balancing information and are excluded; the denominator for varying
# dimensions receives a small total-variation floor. A genuinely class-separated
# coordinate remains strongly penalized rather than being hidden.
STRICT_SERIES_SMD_MIN_TOTAL_STD = 1e-5
STRICT_SERIES_SMD_POOLED_STD_FLOOR_FRACTION = 0.10

# Only the scientifically necessary models and falsification controls are run.
# Definitions and explanatory comments for the omitted experiments remain in
# the source for provenance; EXPERIMENTS_TO_RUN controls execution only.
STRICT_BALANCED_ESSENTIAL_EXPERIMENT_IDS = (
    "A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA",
    "A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA",
    "A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA",
    "A16_HEART_CENTERED_FIXED_FOV_REGION_NORM_HIER_LR_PCA",
    "A9_STANDARDIZED_FULL_HIER_LR_PCA",
    "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
    "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
    "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
    "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA",
    "C17_N_SERIES_ONLY_LR",
    "C19_NATIVE_GEOMETRY_ONLY_LR",
    "C20_FILE_SIZE_ONLY_LR",
)


def _sb1_evenly_spaced_indices(length, maximum_count):
    """Return deterministic unique indices spanning an ordered series."""

    length = int(length)
    maximum_count = int(maximum_count)
    if length <= 0 or maximum_count <= 0:
        raise ValueError("Series sampling requires positive lengths/counts.")
    if length <= maximum_count:
        return np.arange(length, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, length - 1, maximum_count)).astype(np.int64)
    )


def _sb1_image_descriptor(image_path):
    """Create one label-blind image/export descriptor from a grayscale JPEG."""

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.size == 0:
        raise RuntimeError(
            f"Strict series harmonization could not decode: {path}"
        )

    height, width = image.shape
    file_size = float(path.stat().st_size)
    native_pixels = float(max(1, height * width))
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    values = small.astype(np.float32) / 255.0

    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    histogram, _ = np.histogram(values, bins=16, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    histogram /= max(1.0, float(histogram.sum()))
    nonzero_histogram = histogram[histogram > 0.0]
    entropy = float(
        -np.sum(nonzero_histogram * np.log2(nonzero_histogram))
    )

    gradient_x = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    center = values[16:48, 16:48]
    border_mask = np.ones((64, 64), dtype=bool)
    border_mask[8:56, 8:56] = False
    border = values[border_mask]
    quadrants = (
        values[:32, :32],
        values[:32, 32:],
        values[32:, :32],
        values[32:, 32:],
    )

    standardized = (values - float(values.mean())) / (
        float(values.std()) + 1e-6
    )
    dct = cv2.dct(standardized.astype(np.float32))
    low_frequency_dct = (dct[:4, :4].reshape(-1)[1:] / 64.0).astype(
        np.float64
    )

    descriptor = np.concatenate(
        [
            np.asarray(
                [
                    np.log1p(height),
                    np.log1p(width),
                    np.log(max(width / max(height, 1), 1e-6)),
                    np.log1p(native_pixels),
                    np.log1p(file_size),
                    np.log1p(file_size / native_pixels),
                    float(values.mean()),
                    float(values.std()),
                    *[float(value) for value in quantiles],
                    entropy,
                    float(gradient_magnitude.mean()),
                    float(gradient_magnitude.std()),
                    float(center.mean()),
                    float(border.mean()),
                    *[float(region.mean()) for region in quadrants],
                ],
                dtype=np.float64,
            ),
            histogram,
            low_frequency_dct,
        ]
    )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError(f"Non-finite strict descriptor for {path}.")
    return descriptor


def _sb1_series_descriptor(series_rows):
    """Aggregate sampled image descriptors to one robust series descriptor."""

    ordered_rows = sorted(series_rows, key=lambda row: str(row[0]))
    sample_indices = _sb1_evenly_spaced_indices(
        len(ordered_rows),
        STRICT_SERIES_DESCRIPTOR_SAMPLE_COUNT,
    )
    image_descriptors = np.stack(
        [
            _sb1_image_descriptor(ordered_rows[int(index)][0])
            for index in sample_indices
        ],
        axis=0,
    )
    median = np.median(image_descriptors, axis=0)
    iqr = np.quantile(image_descriptors, 0.75, axis=0) - np.quantile(
        image_descriptors, 0.25, axis=0
    )
    descriptor = np.concatenate(
        [
            median,
            iqr,
            np.asarray(
                [
                    np.log1p(len(ordered_rows)),
                    float(len(sample_indices))
                    / float(len(ordered_rows)),
                ],
                dtype=np.float64,
            ),
        ]
    )
    return descriptor


def _sb1_group_samples_by_series(samples):
    """Group rows and verify that every series has one patient and one label."""

    grouped = defaultdict(list)
    for row in samples:
        if len(row) < 4:
            raise ValueError(
                "Expected sample rows (image_path, label, patient_id, series_id)."
            )
        grouped[str(row[3])].append(row)

    records = []
    for series_id in sorted(grouped):
        rows = grouped[series_id]
        labels = {int(row[1]) for row in rows}
        patients = {str(row[2]) for row in rows}
        if len(labels) != 1 or len(patients) != 1:
            raise RuntimeError(
                f"Series {series_id!r} has inconsistent labels/patients."
            )
        records.append(
            {
                "series_id": series_id,
                "patient_id": next(iter(patients)),
                "label": next(iter(labels)),
                "rows": rows,
                "n_images": int(len(rows)),
            }
        )
    return records


def _sb1_effective_cluster_count(requested, n_series):
    """Keep synthetic/small-cohort tests feasible without changing real k."""

    maximum_supported = max(
        2,
        int(n_series)
        // max(2, 2 * STRICT_SERIES_MIN_SERIES_PER_CLASS_PER_CLUSTER),
    )
    return int(max(2, min(int(requested), maximum_supported, n_series - 1)))


def _sb1_patient_level_balance(descriptors, labels, patients, selected_mask):
    """Measure class imbalance at the actual patient-level evaluation unit.

    SB1 used an absolute ``1e-6`` denominator floor for every descriptor. With
    108 correlated coordinates and only 30 patients, a globally almost-constant
    coordinate could therefore create an enormous maximum SMD from numerical
    noise and make every target fail. SB1.1 excludes only globally constant
    coordinates and regularizes the denominator by a small fraction of total
    patient-level variation. Genuine low-variance class separation remains
    visible and can still fail the guard.
    """

    selected_indices = np.flatnonzero(selected_mask)
    selected_patients = sorted(set(patients[selected_indices].tolist()))
    patient_rows = []
    patient_labels = []
    for patient_id in selected_patients:
        indices = selected_indices[patients[selected_indices] == patient_id]
        label_values = np.unique(labels[indices])
        if len(label_values) != 1:
            raise RuntimeError(
                f"Patient {patient_id} has inconsistent strict-filter labels."
            )
        patient_rows.append(np.mean(descriptors[indices], axis=0))
        patient_labels.append(int(label_values[0]))

    patient_rows = np.asarray(patient_rows, dtype=np.float64)
    patient_labels = np.asarray(patient_labels, dtype=np.int64)
    class_zero = patient_rows[patient_labels == 0]
    class_one = patient_rows[patient_labels == 1]
    if len(class_zero) < 2 or len(class_one) < 2:
        raise RuntimeError(
            "Strict balance requires at least two retained patients per class."
        )

    pooled_std = np.sqrt(
        0.5
        * (
            np.var(class_zero, axis=0, ddof=1)
            + np.var(class_one, axis=0, ddof=1)
        )
    )
    total_std = np.std(patient_rows, axis=0, ddof=1)
    informative = total_std > STRICT_SERIES_SMD_MIN_TOTAL_STD
    denominator = np.maximum(
        pooled_std,
        STRICT_SERIES_SMD_POOLED_STD_FLOOR_FRACTION * total_std,
    )
    denominator = np.maximum(denominator, STRICT_SERIES_SMD_MIN_TOTAL_STD)
    mean_gap = np.abs(
        np.mean(class_one, axis=0) - np.mean(class_zero, axis=0)
    )
    standardized_difference = np.zeros(patient_rows.shape[1], dtype=np.float64)
    standardized_difference[informative] = (
        mean_gap[informative] / denominator[informative]
    )
    if not np.all(np.isfinite(standardized_difference)):
        raise RuntimeError("Strict patient-level SMDs contain non-finite values.")

    maximum_index = int(np.argmax(standardized_difference))
    return {
        "median_absolute_patient_smd": float(
            np.median(standardized_difference)
        ),
        "q90_absolute_patient_smd": float(
            np.quantile(standardized_difference, 0.90)
        ),
        "maximum_absolute_patient_smd": float(
            standardized_difference[maximum_index]
        ),
        "maximum_absolute_patient_smd_descriptor_index": maximum_index,
        "informative_descriptor_dimensions": int(np.sum(informative)),
        "total_descriptor_dimensions": int(patient_rows.shape[1]),
        "absolute_patient_smd": standardized_difference,
    }


def _sb1_guard_diagnostics(
    balance,
    rescue_fraction,
    rescue_patients,
    retained_fraction,
):
    """Return strict pass/fail flags and a deterministic violation score."""

    checks = {
        "rescue_fraction": (
            float(rescue_fraction) <= STRICT_SERIES_MAX_RESCUE_FRACTION
        ),
        "rescue_patients": (
            int(rescue_patients) <= STRICT_SERIES_MAX_RESCUE_PATIENTS
        ),
        "median_absolute_patient_smd": (
            balance["median_absolute_patient_smd"]
            <= STRICT_SERIES_PATIENT_SMD_MEDIAN_LIMIT
        ),
        "q90_absolute_patient_smd": (
            balance["q90_absolute_patient_smd"]
            <= STRICT_SERIES_PATIENT_SMD_Q90_LIMIT
        ),
        "maximum_absolute_patient_smd": (
            balance["maximum_absolute_patient_smd"]
            <= STRICT_SERIES_PATIENT_SMD_MAX_LIMIT
        ),
        "minimum_retained_fraction": (
            retained_fraction >= STRICT_SERIES_MIN_RETAINED_FRACTION_OF_V72
        ),
        "maximum_retained_fraction": (
            retained_fraction <= STRICT_SERIES_MAX_RETAINED_FRACTION_OF_V72
        ),
    }
    ratios = np.asarray(
        [
            float(rescue_fraction)
            / max(STRICT_SERIES_MAX_RESCUE_FRACTION, 1e-12),
            float(rescue_patients)
            / max(STRICT_SERIES_MAX_RESCUE_PATIENTS, 1e-12),
            balance["median_absolute_patient_smd"]
            / max(STRICT_SERIES_PATIENT_SMD_MEDIAN_LIMIT, 1e-12),
            balance["q90_absolute_patient_smd"]
            / max(STRICT_SERIES_PATIENT_SMD_Q90_LIMIT, 1e-12),
            balance["maximum_absolute_patient_smd"]
            / max(STRICT_SERIES_PATIENT_SMD_MAX_LIMIT, 1e-12),
            (
                STRICT_SERIES_MIN_RETAINED_FRACTION_OF_V72
                / max(retained_fraction, 1e-12)
                if retained_fraction
                < STRICT_SERIES_MIN_RETAINED_FRACTION_OF_V72
                else 1.0
            ),
            (
                retained_fraction
                / max(STRICT_SERIES_MAX_RETAINED_FRACTION_OF_V72, 1e-12)
                if retained_fraction
                > STRICT_SERIES_MAX_RETAINED_FRACTION_OF_V72
                else 1.0
            ),
        ],
        dtype=np.float64,
    )
    excess = np.maximum(ratios - 1.0, 0.0)
    return {
        "strict_guard_passed": bool(all(checks.values())),
        "checks": checks,
        "violated_guards": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "normalized_guard_violation_score": float(
            100.0 * np.sum(excess**2) + np.sum(ratios)
        ),
    }


def _sb1_select_exact_count_per_patient(
    records,
    strict_mask,
    fallback_mask,
    ranking_score,
    target_per_patient,
):
    """Select exactly one common target count for every Directory_* patient."""

    patients = np.asarray([row["patient_id"] for row in records])
    selected = np.zeros(len(records), dtype=bool)
    rescue = np.zeros(len(records), dtype=bool)

    for patient_id in sorted(set(patients.tolist())):
        patient_indices = np.flatnonzero(patients == patient_id)
        strict_indices = patient_indices[strict_mask[patient_indices]]
        strict_indices = strict_indices[
            np.argsort(ranking_score[strict_indices], kind="mergesort")
        ]
        chosen = list(strict_indices[:target_per_patient])

        if len(chosen) < target_per_patient:
            chosen_set = set(chosen)
            fallback_indices = np.asarray(
                [
                    index
                    for index in patient_indices
                    if fallback_mask[index] and index not in chosen_set
                ],
                dtype=np.int64,
            )
            fallback_indices = fallback_indices[
                np.argsort(ranking_score[fallback_indices], kind="mergesort")
            ]
            needed = target_per_patient - len(chosen)
            rescued = fallback_indices[:needed].tolist()
            chosen.extend(rescued)
            rescue[rescued] = True

        if len(chosen) != target_per_patient:
            raise RuntimeError(
                f"Patient {patient_id} has only {len(chosen)} eligible/rescue "
                f"series, below target={target_per_patient}."
            )
        selected[np.asarray(chosen, dtype=np.int64)] = True

    return selected, rescue


def apply_strict_balanced_series_refinement(samples, output_dir):
    """Apply strict shared-support and exact per-patient series balancing.

    This function starts from the series already retained by V7.2. It does not
    alter any selected series internally: every JPEG belonging to a retained
    folder-defined series proxy remains available to all image representations.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _sb1_group_samples_by_series(samples)
    if not records:
        raise RuntimeError("Strict series refinement received no series.")

    print(
        "[STRICT SERIES BALANCE] Building descriptors for "
        f"{len(records)} V7.2-retained series proxies.",
        flush=True,
    )
    descriptors = []
    progress_step = max(1, len(records) // 20)
    for index, record in enumerate(records, start=1):
        descriptors.append(_sb1_series_descriptor(record["rows"]))
        if index == 1 or index % progress_step == 0 or index == len(records):
            print(
                f"[STRICT SERIES BALANCE] descriptors={index}/{len(records)} "
                f"({100.0 * index / len(records):.1f}%)",
                flush=True,
            )

    descriptors = np.asarray(descriptors, dtype=np.float64)
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    patients = np.asarray([row["patient_id"] for row in records]).astype(str)
    series_ids = np.asarray([row["series_id"] for row in records]).astype(str)
    lengths = np.asarray([row["n_images"] for row in records], dtype=np.int64)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("Strict balancing requires both Normal and Sick series.")

    scaler = _SB1RobustScaler(quantile_range=(25.0, 75.0))
    scaled = scaler.fit_transform(descriptors)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=8.0, neginf=-8.0)
    scaled = np.clip(scaled, -8.0, 8.0)

    n_components = min(
        STRICT_SERIES_PCA_COMPONENTS,
        scaled.shape[1],
        scaled.shape[0] - 1,
    )
    pca = _SB1PCA(
        n_components=n_components,
        whiten=True,
        svd_solver="full",
    )
    representation = pca.fit_transform(scaled)
    representation = np.nan_to_num(
        representation,
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )

    opposite_distance = np.full(len(records), np.inf, dtype=np.float64)
    for source_label, opposite_label in ((0, 1), (1, 0)):
        source_indices = np.flatnonzero(labels == source_label)
        opposite_indices = np.flatnonzero(labels == opposite_label)
        neighbours = _SB1NearestNeighbors(
            n_neighbors=1,
            algorithm="auto",
            metric="euclidean",
        )
        neighbours.fit(representation[opposite_indices])
        distances, _ = neighbours.kneighbors(representation[source_indices])
        opposite_distance[source_indices] = distances[:, 0]

    class_distance_limits = {
        label: float(
            np.quantile(
                opposite_distance[labels == label],
                STRICT_SERIES_OPPOSITE_NEIGHBOUR_QUANTILE,
            )
        )
        for label in (0, 1)
    }
    common_opposite_distance_limit = min(class_distance_limits.values())

    shared_votes = np.zeros(len(records), dtype=np.int64)
    inlier_votes = np.zeros(len(records), dtype=np.int64)
    normalized_cluster_distance_sum = np.zeros(len(records), dtype=np.float64)
    effective_cluster_counts = []
    cluster_manifest_rows = []
    primary_assignments = None

    log_lengths = np.log1p(lengths.astype(np.float64))
    for resolution_index, requested_k in enumerate(
        STRICT_SERIES_CLUSTER_COUNTS
    ):
        k = _sb1_effective_cluster_count(requested_k, len(records))
        effective_cluster_counts.append(k)
        kmeans = _SB1KMeans(
            n_clusters=k,
            random_state=STRICT_SERIES_RANDOM_STATE + resolution_index,
            n_init=20,
            max_iter=500,
        )
        assignments = kmeans.fit_predict(representation)
        distances = np.linalg.norm(
            representation - kmeans.cluster_centers_[assignments],
            axis=1,
        )
        if requested_k == STRICT_SERIES_CLUSTER_COUNTS[
            len(STRICT_SERIES_CLUSTER_COUNTS) // 2
        ]:
            primary_assignments = assignments.copy()

        for cluster_id in range(k):
            cluster_indices = np.flatnonzero(assignments == cluster_id)
            class_indices = {
                label: cluster_indices[labels[cluster_indices] == label]
                for label in (0, 1)
            }
            n_series = {label: int(len(class_indices[label])) for label in (0, 1)}
            n_patients = {
                label: int(len(set(patients[class_indices[label]].tolist())))
                for label in (0, 1)
            }
            series_ratio = (
                min(n_series.values()) / max(n_series.values())
                if max(n_series.values()) > 0
                else 0.0
            )
            patient_ratio = (
                min(n_patients.values()) / max(n_patients.values())
                if max(n_patients.values()) > 0
                else 0.0
            )

            shared = all(
                n_patients[label]
                >= STRICT_SERIES_MIN_PATIENTS_PER_CLASS_PER_CLUSTER
                and n_series[label]
                >= STRICT_SERIES_MIN_SERIES_PER_CLASS_PER_CLUSTER
                for label in (0, 1)
            )
            shared = shared and (
                series_ratio >= STRICT_SERIES_MIN_CLASS_SUPPORT_RATIO
                and patient_ratio >= STRICT_SERIES_MIN_CLASS_SUPPORT_RATIO
            )

            if all(n_series[label] > 0 for label in (0, 1)):
                class_centroid_gap = float(
                    np.mean(
                        np.abs(
                            np.mean(scaled[class_indices[1]], axis=0)
                            - np.mean(scaled[class_indices[0]], axis=0)
                        )
                    )
                )
                distance_medians = {
                    label: float(np.median(distances[class_indices[label]]))
                    for label in (0, 1)
                }
                distance_median_ratio = max(distance_medians.values()) / max(
                    1e-8, min(distance_medians.values())
                )
                lower = max(
                    float(
                        np.quantile(
                            log_lengths[class_indices[label]],
                            STRICT_SERIES_LENGTH_LOWER_QUANTILE,
                        )
                    )
                    for label in (0, 1)
                )
                upper = min(
                    float(
                        np.quantile(
                            log_lengths[class_indices[label]],
                            STRICT_SERIES_LENGTH_UPPER_QUANTILE,
                        )
                    )
                    for label in (0, 1)
                )
                shared = shared and (
                    class_centroid_gap
                    <= STRICT_SERIES_MAX_CLASS_CENTROID_GAP
                    and distance_median_ratio
                    <= STRICT_SERIES_MAX_DISTANCE_MEDIAN_RATIO
                    and lower <= upper
                )
            else:
                class_centroid_gap = float("inf")
                distance_median_ratio = float("inf")
                lower, upper = 1.0, 0.0
                shared = False

            if shared:
                distance_limit = min(
                    float(
                        np.quantile(
                            distances[class_indices[label]],
                            STRICT_SERIES_CLUSTER_DISTANCE_QUANTILE,
                        )
                    )
                    for label in (0, 1)
                )
                eligible = cluster_indices[
                    (distances[cluster_indices] <= distance_limit)
                    & (log_lengths[cluster_indices] >= lower)
                    & (log_lengths[cluster_indices] <= upper)
                ]
                shared_votes[eligible] += 1
                inlier_votes[eligible] += 1
                normalized_cluster_distance_sum[cluster_indices] += (
                    distances[cluster_indices] / max(distance_limit, 1e-8)
                )
            else:
                distance_limit = float("nan")
                normalized_cluster_distance_sum[cluster_indices] += 4.0

            cluster_manifest_rows.append(
                {
                    "requested_k": int(requested_k),
                    "effective_k": int(k),
                    "cluster_id": int(cluster_id),
                    "normal_series": n_series[0],
                    "sick_series": n_series[1],
                    "normal_patients": n_patients[0],
                    "sick_patients": n_patients[1],
                    "series_support_ratio": float(series_ratio),
                    "patient_support_ratio": float(patient_ratio),
                    "mean_absolute_scaled_descriptor_gap": (
                        None
                        if not np.isfinite(class_centroid_gap)
                        else float(class_centroid_gap)
                    ),
                    "class_distance_median_ratio": (
                        None
                        if not np.isfinite(distance_median_ratio)
                        else float(distance_median_ratio)
                    ),
                    "distance_limit": (
                        None
                        if not np.isfinite(distance_limit)
                        else float(distance_limit)
                    ),
                    "shared_strict_cluster": bool(shared),
                }
            )

    if primary_assignments is None:
        raise RuntimeError("Strict harmonization did not establish primary clusters.")

    strict_mask = (
        shared_votes >= STRICT_SERIES_REQUIRED_SHARED_VOTES
    ) & (opposite_distance <= common_opposite_distance_limit)
    # A broader, still label-blind-in-descriptor fallback can rescue a small
    # number of series needed to give every patient an identical series count.
    fallback_mask = (
        shared_votes >= max(2, STRICT_SERIES_REQUIRED_SHARED_VOTES - 1)
    ) & (
        opposite_distance
        <= max(
            float(np.quantile(opposite_distance[labels == 0], 0.85)),
            float(np.quantile(opposite_distance[labels == 1], 0.85)),
        )
    )

    mean_cluster_distance = normalized_cluster_distance_sum / float(
        len(STRICT_SERIES_CLUSTER_COUNTS)
    )
    opposite_scale = max(common_opposite_distance_limit, 1e-8)
    ranking_score = (
        opposite_distance / opposite_scale
        + mean_cluster_distance
        + 0.10 * np.abs(log_lengths - np.median(log_lengths))
    )

    strict_counts = np.asarray(
        [
            int(np.sum(strict_mask[patients == patient_id]))
            for patient_id in sorted(set(patients.tolist()))
        ],
        dtype=np.int64,
    )
    proposed_target = int(
        np.floor(
            np.quantile(strict_counts, STRICT_SERIES_TARGET_QUANTILE)
        )
    )
    proposed_target = max(
        STRICT_SERIES_MIN_PER_PATIENT,
        min(STRICT_SERIES_MAX_PER_PATIENT, proposed_target),
    )

    chosen = None
    chosen_rescue = None
    chosen_balance = None
    chosen_target = None
    chosen_guard = None
    selection_status = None
    selection_policy = None
    candidate_diagnostics = []
    best_feasible_candidate = None

    # Retention is part of the target decision, not a second independent failure
    # after a target has already been accepted. This also exposes the latent SB1
    # incompatibility between a low exact target and the 20% retention floor.
    n_patients = int(len(set(patients.tolist())))
    minimum_target_from_retention = int(
        np.ceil(
            STRICT_SERIES_MIN_RETAINED_FRACTION_OF_V72
            * len(records)
            / max(1, n_patients)
        )
    )
    search_minimum = max(
        STRICT_SERIES_MIN_PER_PATIENT,
        minimum_target_from_retention,
    )
    maximum_target_from_retention = int(
        np.floor(
            STRICT_SERIES_MAX_RETAINED_FRACTION_OF_V72
            * len(records)
            / max(1, n_patients)
        )
    )
    fallback_counts = np.asarray(
        [
            int(np.sum(fallback_mask[patients == patient_id]))
            for patient_id in sorted(set(patients.tolist()))
        ],
        dtype=np.int64,
    )
    search_maximum = min(
        STRICT_SERIES_MAX_PER_PATIENT,
        maximum_target_from_retention,
        int(np.min(fallback_counts)),
    )
    # The original proposed target came only from the strict-count quartile. If
    # it lies below the retention-derived minimum, evaluate the minimum safe
    # target rather than producing an empty Python range and jumping directly to
    # the emergency pool.
    first_target = min(
        search_maximum,
        max(proposed_target, search_minimum),
    )

    for target in range(
        first_target,
        search_minimum - 1,
        -1,
    ):
        try:
            selected, rescue = _sb1_select_exact_count_per_patient(
                records,
                strict_mask,
                fallback_mask,
                ranking_score,
                target,
            )
        except RuntimeError as error:
            candidate_diagnostics.append(
                {"target": int(target), "status": "INFEASIBLE", "reason": str(error)}
            )
            continue

        rescue_fraction = float(np.sum(rescue)) / float(np.sum(selected))
        rescue_patients = int(len(set(patients[rescue].tolist())))
        retained_fraction = float(np.sum(selected)) / float(len(records))
        balance = _sb1_patient_level_balance(
            scaled,
            labels,
            patients,
            selected,
        )
        guard = _sb1_guard_diagnostics(
            balance,
            rescue_fraction,
            rescue_patients,
            retained_fraction,
        )
        candidate_diagnostics.append(
            {
                "target": int(target),
                "status": (
                    "ACCEPTABLE"
                    if guard["strict_guard_passed"]
                    else "BALANCE_GUARD_FAILED"
                ),
                "rescue_fraction": rescue_fraction,
                "rescue_patients": rescue_patients,
                "retained_fraction_of_v72": retained_fraction,
                "median_absolute_patient_smd": balance[
                    "median_absolute_patient_smd"
                ],
                "q90_absolute_patient_smd": balance[
                    "q90_absolute_patient_smd"
                ],
                "maximum_absolute_patient_smd": balance[
                    "maximum_absolute_patient_smd"
                ],
                "maximum_absolute_patient_smd_descriptor_index": balance[
                    "maximum_absolute_patient_smd_descriptor_index"
                ],
                "informative_descriptor_dimensions": balance[
                    "informative_descriptor_dimensions"
                ],
                "violated_guards": guard["violated_guards"],
                "normalized_guard_violation_score": guard[
                    "normalized_guard_violation_score"
                ],
            }
        )

        candidate = {
            "policy": "configured_fallback",
            "selected": selected.copy(),
            "rescue": rescue.copy(),
            "balance": balance,
            "target": int(target),
            "guard": guard,
            "retained_fraction": retained_fraction,
        }
        if (
            best_feasible_candidate is None
            or (
                guard["normalized_guard_violation_score"],
                len(guard["violated_guards"]),
                -int(target),
            )
            < (
                best_feasible_candidate["guard"][
                    "normalized_guard_violation_score"
                ],
                len(best_feasible_candidate["guard"]["violated_guards"]),
                -int(best_feasible_candidate["target"]),
            )
        ):
            best_feasible_candidate = candidate

        if guard["strict_guard_passed"]:
            chosen = selected
            chosen_rescue = rescue
            chosen_balance = balance
            chosen_target = target
            chosen_guard = guard
            selection_status = "OK_STRICT_GUARDS_PASSED"
            selection_policy = "configured_fallback"
            break

    # If the original strict/fallback pool cannot supply any target in the safe
    # retention range, construct one exact-count emergency candidate from the
    # already V7.2-harmonized series. It remains fully audited and is never
    # labelled a strict pass.
    if best_feasible_candidate is None:
        total_counts = np.asarray(
            [
                int(np.sum(patients == patient_id))
                for patient_id in sorted(set(patients.tolist()))
            ],
            dtype=np.int64,
        )
        emergency_maximum = min(
            STRICT_SERIES_MAX_PER_PATIENT,
            maximum_target_from_retention,
            int(np.min(total_counts)),
        )
        emergency_target = min(
            emergency_maximum,
            max(search_minimum, proposed_target),
        )
        if emergency_target < 1:
            raise RuntimeError(
                "No Directory_* patient has a series available for strict "
                "balance fallback."
            )
        all_series_mask = np.ones(len(records), dtype=bool)
        selected, rescue = _sb1_select_exact_count_per_patient(
            records,
            strict_mask,
            all_series_mask,
            ranking_score,
            emergency_target,
        )
        rescue_fraction = float(np.sum(rescue)) / float(np.sum(selected))
        rescue_patients = int(len(set(patients[rescue].tolist())))
        retained_fraction = float(np.sum(selected)) / float(len(records))
        balance = _sb1_patient_level_balance(
            scaled,
            labels,
            patients,
            selected,
        )
        guard = _sb1_guard_diagnostics(
            balance,
            rescue_fraction,
            rescue_patients,
            retained_fraction,
        )
        best_feasible_candidate = {
            "policy": "all_v72_series_emergency",
            "selected": selected,
            "rescue": rescue,
            "balance": balance,
            "target": int(emergency_target),
            "guard": guard,
            "retained_fraction": retained_fraction,
        }
        candidate_diagnostics.append(
            {
                "target": int(emergency_target),
                "status": "EMERGENCY_ALL_V72_SERIES_CANDIDATE",
                "rescue_fraction": rescue_fraction,
                "rescue_patients": rescue_patients,
                "retained_fraction_of_v72": retained_fraction,
                "median_absolute_patient_smd": balance[
                    "median_absolute_patient_smd"
                ],
                "q90_absolute_patient_smd": balance[
                    "q90_absolute_patient_smd"
                ],
                "maximum_absolute_patient_smd": balance[
                    "maximum_absolute_patient_smd"
                ],
                "violated_guards": guard["violated_guards"],
                "normalized_guard_violation_score": guard[
                    "normalized_guard_violation_score"
                ],
            }
        )

    if chosen is None:
        diagnostic_path = output_dir / (
            "strict_series_balance_failed_targets.json"
        )
        diagnostic_path.write_text(
            json.dumps(candidate_diagnostics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not STRICT_SERIES_ALLOW_BEST_ATTAINABLE_FALLBACK:
            raise RuntimeError(
                "No exact per-patient target satisfied every strict guard and "
                "best-attainable fallback is disabled. See "
                + str(diagnostic_path)
            )

        chosen = best_feasible_candidate["selected"]
        chosen_rescue = best_feasible_candidate["rescue"]
        chosen_balance = best_feasible_candidate["balance"]
        chosen_target = best_feasible_candidate["target"]
        chosen_guard = best_feasible_candidate["guard"]
        selection_status = "OK_BEST_ATTAINABLE_BALANCE"
        selection_policy = best_feasible_candidate["policy"]
        print(
            "[STRICT SERIES BALANCE][BEST-ATTAINABLE WARNING] No exact "
            "per-patient target passed every predeclared guard. Continuing "
            "with the feasible target having the smallest normalized guard "
            "violation. Do not describe this run as strictly balanced.",
            flush=True,
        )
        print(
            "[STRICT SERIES BALANCE][BEST-ATTAINABLE WARNING] "
            f"policy={selection_policy}, target={chosen_target}, violated_guards="
            f"{chosen_guard['violated_guards']}, normalized_violation_score="
            f"{chosen_guard['normalized_guard_violation_score']:.4f}.",
            flush=True,
        )

    (output_dir / "strict_series_balance_target_diagnostics.json").write_text(
        json.dumps(candidate_diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    retained_series = set(series_ids[chosen].tolist())
    filtered_samples = [
        row for row in samples if str(row[3]) in retained_series
    ]
    retained_fraction = float(np.sum(chosen)) / float(len(records))
    retention_guard_passed = bool(
        STRICT_SERIES_MIN_RETAINED_FRACTION_OF_V72
        <= retained_fraction
        <= STRICT_SERIES_MAX_RETAINED_FRACTION_OF_V72
    )
    if not retention_guard_passed and selection_status == (
        "OK_STRICT_GUARDS_PASSED"
    ):
        raise RuntimeError(
            "Internal error: a strict target violated the retained-fraction "
            f"guard ({retained_fraction:.3f})."
        )
    if not retention_guard_passed:
        print(
            "[STRICT SERIES BALANCE][BEST-ATTAINABLE WARNING] Retained "
            "fraction lies outside the strict configured interval: "
            f"{retained_fraction:.3f}.",
            flush=True,
        )

    original_patient_set = set(patients.tolist())
    filtered_patient_set = {str(row[2]) for row in filtered_samples}
    if filtered_patient_set != original_patient_set:
        missing = sorted(original_patient_set - filtered_patient_set)
        raise RuntimeError(
            f"Strict series balancing removed complete patients: {missing}."
        )

    per_patient_rows = []
    for patient_id in sorted(original_patient_set):
        patient_indices = np.flatnonzero(patients == patient_id)
        selected_indices = patient_indices[chosen[patient_indices]]
        label_values = np.unique(labels[patient_indices])
        per_patient_rows.append(
            {
                "patient_id": patient_id,
                "label": int(label_values[0]),
                "series_before_strict_refinement": int(len(patient_indices)),
                "strict_eligible_series": int(
                    np.sum(strict_mask[patient_indices])
                ),
                "series_retained": int(len(selected_indices)),
                "rescue_series_retained": int(
                    np.sum(chosen_rescue[selected_indices])
                ),
                "images_retained": int(np.sum(lengths[selected_indices])),
            }
        )

    manifest_rows = []
    for index, record in enumerate(records):
        manifest_rows.append(
            {
                "series_id": record["series_id"],
                "patient_id": record["patient_id"],
                "label": int(record["label"]),
                "n_images": int(record["n_images"]),
                "primary_cluster": int(primary_assignments[index]),
                "shared_inlier_votes": int(shared_votes[index]),
                "opposite_class_neighbour_distance": float(
                    opposite_distance[index]
                ),
                "strict_eligible": bool(strict_mask[index]),
                "fallback_eligible": bool(fallback_mask[index]),
                "ranking_score": float(ranking_score[index]),
                "retained": bool(chosen[index]),
                "balance_rescue": bool(chosen_rescue[index]),
                "decision_reason": (
                    "strict_shared_opposite_class_overlap"
                    if chosen[index] and not chosen_rescue[index]
                    else "limited_balance_rescue"
                    if chosen[index]
                    else "removed_atypical_or_redundant"
                ),
            }
        )

    selection_digest = hashlib.sha256()
    for series_id in sorted(retained_series):
        selection_digest.update((series_id + "\n").encode("utf-8"))

    absolute_smd = chosen_balance.pop("absolute_patient_smd")
    strict_guard_passed = bool(
        selection_status == "OK_STRICT_GUARDS_PASSED"
        and chosen_guard["strict_guard_passed"]
    )
    summary = {
        "status": selection_status,
        "version": STRICT_SERIES_BALANCE_VERSION,
        "strict_guard_passed": strict_guard_passed,
        "selection_policy": selection_policy,
        "selection_guard_checks": chosen_guard["checks"],
        "violated_guards": list(chosen_guard["violated_guards"]),
        "normalized_guard_violation_score": float(
            chosen_guard["normalized_guard_violation_score"]
        ),
        "statistical_scope": (
            "Global label-informed cross-class common-support sensitivity "
            "analysis; not external validation and not fold-local. "
            + (
                "Every strict guard passed."
                if strict_guard_passed
                else "The best-attainable fallback was used; strict balance "
                "was not achieved and must not be claimed."
            )
        ),
        "series_before_strict_refinement": int(len(records)),
        "series_after_strict_refinement": int(np.sum(chosen)),
        "series_removed_by_strict_refinement": int(
            len(records) - np.sum(chosen)
        ),
        "series_retained_fraction_of_v72": retained_fraction,
        "images_before_strict_refinement": int(len(samples)),
        "images_after_strict_refinement": int(len(filtered_samples)),
        "images_removed_by_strict_refinement": int(
            len(samples) - len(filtered_samples)
        ),
        "patients_retained": int(len(filtered_patient_set)),
        "normal_patients_retained": int(
            len(set(patients[(labels == 0) & chosen].tolist()))
        ),
        "sick_patients_retained": int(
            len(set(patients[(labels == 1) & chosen].tolist()))
        ),
        "exact_series_target_per_patient": int(chosen_target),
        "normal_series_per_patient": int(chosen_target),
        "sick_series_per_patient": int(chosen_target),
        "rescue_series": int(np.sum(chosen_rescue)),
        "rescue_fraction": float(np.sum(chosen_rescue) / np.sum(chosen)),
        "rescue_patients": int(
            len(set(patients[chosen_rescue].tolist()))
        ),
        "requested_cluster_counts": list(STRICT_SERIES_CLUSTER_COUNTS),
        "effective_cluster_counts": effective_cluster_counts,
        "required_shared_votes": int(STRICT_SERIES_REQUIRED_SHARED_VOTES),
        "opposite_class_distance_limits_by_label": class_distance_limits,
        "common_opposite_class_distance_limit": float(
            common_opposite_distance_limit
        ),
        "patient_level_descriptor_balance": chosen_balance,
        "selection_signature_sha256": selection_digest.hexdigest(),
        "candidate_target_diagnostics": candidate_diagnostics,
        "minimum_target_from_retention_guard": int(
            minimum_target_from_retention
        ),
        "maximum_target_from_retention_guard": int(
            maximum_target_from_retention
        ),
        "configured_fallback_minimum_series_for_any_patient": int(
            np.min(fallback_counts)
        ),
        "essential_experiment_ids": list(
            STRICT_BALANCED_ESSENTIAL_EXPERIMENT_IDS
        ),
    }

    config = {
        name: value
        for name, value in globals().items()
        if name.startswith("STRICT_SERIES_")
        and isinstance(value, (str, int, float, bool, tuple))
    }
    config["statistical_warning"] = summary["statistical_scope"]
    config["selection_status"] = selection_status
    config["selection_policy"] = selection_policy
    config["strict_guard_passed"] = strict_guard_passed
    (output_dir / "strict_series_balance_configuration.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "strict_series_balance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    def _write_csv(path, rows):
        if not rows:
            Path(path).write_text("\n", encoding="utf-8")
            return
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(output_dir / "strict_series_balance_manifest.csv", manifest_rows)
    _write_csv(
        output_dir / "strict_series_balance_cluster_support.csv",
        cluster_manifest_rows,
    )
    _write_csv(
        output_dir / "strict_series_balance_patient_counts.csv",
        per_patient_rows,
    )
    smd_rows = [
        {
            "descriptor_index": int(index),
            "absolute_patient_level_smd": float(value),
        }
        for index, value in enumerate(absolute_smd)
    ]
    _write_csv(
        output_dir / "strict_series_balance_patient_descriptor_smd.csv",
        smd_rows,
    )

    print("[STRICT SERIES BALANCE] Selection summary", flush=True)
    print(
        f"  Status: {selection_status}; policy={selection_policy}; "
        f"strict_guard_passed={strict_guard_passed}",
        flush=True,
    )
    print(
        f"  Series: {len(records)} -> {int(np.sum(chosen))} "
        f"(removed {len(records) - int(np.sum(chosen))})",
        flush=True,
    )
    print(
        f"  Images: {len(samples)} -> {len(filtered_samples)} "
        f"({100.0 * len(filtered_samples) / len(samples):.1f}% of V7.2 rows retained)",
        flush=True,
    )
    print(
        f"  Patients: {len(filtered_patient_set)}; exact series/patient="
        f"{chosen_target}; rescue series={int(np.sum(chosen_rescue))}",
        flush=True,
    )
    print(
        "  Patient-level descriptor SMD: median="
        f"{chosen_balance['median_absolute_patient_smd']:.3f}, q90="
        f"{chosen_balance['q90_absolute_patient_smd']:.3f}, max="
        f"{chosen_balance['maximum_absolute_patient_smd']:.3f}",
        flush=True,
    )
    print(
        "[STRICT SERIES BALANCE][STATISTICAL WARNING] The retained set was "
        "selected using both class folders globally. Treat this as a common-"
        "support sensitivity analysis, not external or fold-local validation. "
        "When strict_guard_passed=False, report it specifically as the best "
        "attainable exact-count selection rather than strict balance.",
        flush=True,
    )
    return filtered_samples, summary


def main():
    """Run every enabled experiment and compare the patient-level results."""

    pipeline_started_at = time.perf_counter()
    stage_durations = {}
    experiments = get_enabled_experiments()

    print("\n" + "#" * 100, flush=True)
    print("CAD CARDIAC MRI — FOCUSED STRICT-BALANCED SERIES-HARMONIZED RESEARCH SUITE V7.2-SB1.1", flush=True)
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
        f"[SUITE] Focused profile: {FOCUSED_SUITE_PROFILE}; "
        f"image feature modes="
        f"{len(required_efficientnet_feature_modes(experiments))}.",
        flush=True,
    )
    print(
        "[SUITE] Patient definition is fixed: patient_id = Directory_*.",
        flush=True,
    )

    # ======================================================================
    # MAIN STAGE 1 -- VALIDATE THE FOCUSED RESEARCH CONFIGURATION
    # ======================================================================
    # Validation occurs before dataset scanning, model downloads, GPU
    # allocation, cache creation, or experiment fitting. In addition to the
    # image-pipeline checks, this suite validates that every registry entry has
    # an essential or explicitly future role, that future candidates declare a
    # matched control, and that CV, thresholds, duplicate audits, row contracts,
    # deterministic settings and requested feature modes are coherent. Failing
    # here prevents a malformed configuration from producing plausible outputs.
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
        "Focused registry, future-control contract, CV, audit, model and "
        "A17/C31/C32/C33 transform self-tests passed validation.",
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
        "model_selection",
        "sequence_view_analysis",
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
        "Discover patients and select cross-class comparable series",
        "Filesystem scan plus a lightweight sampled-JPEG descriptor pass over folder-defined series proxies.",
    )
    all_discovered_samples = load_samples(DATASET_PATH)
    samples, series_harmonization_summary = (
        harmonize_cross_class_series(
            all_discovered_samples,
            OUTPUT_DIR / "audits",
        )
    )

    # V7.2-SB1 keeps V7.2's original shared-family filter, then applies
    # a stricter opposite-class-overlap and exact per-patient count rule.
    # No downstream image/model/CV implementation is changed.
    samples, strict_series_balance_summary = (
        apply_strict_balanced_series_refinement(
            samples,
            OUTPUT_DIR / "audits",
        )
    )
    stage_durations["03 Dataset discovery"] = _print_stage_complete(
        3,
        "Discover patients and select cross-class comparable series",
        stage_started,
        f"Discovered {len(samples)} image rows.",
    )

    # ======================================================================
    # MAIN STAGE 4 -- LOAD OR EXTRACT ONE SHARED MULTI-VIEW FEATURE BANK
    # ======================================================================
    # This is the expensive image-processing stage. On a cache miss, each JPEG
    # is decoded once. The active panel requires only nine image modes: A9, A12,
    # A16, A17, C31, C32, C33, C28 and C29. One standardized MONAI pass and one
    # raw/robust aligned image contract generate all of them. EfficientNet writes
    # frozen 1280-D embeddings to memory-mapped arrays, grouping a few modes per
    # encoder call to control GPU memory. The four tabular nuisance controls do
    # not add neural inference.
    #
    # Downstream row filters, deduplication, slice dropout, pooling, PCA and
    # Logistic Regression reuse these arrays. The fingerprint includes every
    # setting capable of changing the frozen representation, so a future image
    # candidate triggers a new cache rather than silently reusing incompatible
    # features.
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
    write_series_annotation_template(
        OUTPUT_DIR / "audits" / "series_annotation_template.csv",
        samples,
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
    # area, confidence and fallback behavior. The focused registry computes only
    # the standardized MONAI branch; the historical original branch therefore
    # carries intentional all-NaN sentinels in the shared cache and is written as
    # SKIPPED_NOT_COMPUTED rather than imputed. A future original-MONAI-QC
    # experiment automatically requests that branch and forces a compatible
    # cache rebuild. All available tables are saved descriptively and evaluated
    # through the same outer folds. A high control AUC indicates
    # that Normal/Sick may be predictable from acquisition/export/protocol
    # provenance rather than exclusively from CAD-related anatomy.
    stage_started = _print_stage_start(
        7,
        "Build and save provenance, standardization and MONAI-QC controls",
        "Patient-level aggregation and descriptive audit files.",
    )
    provenance_set = aggregate_patient_provenance_features(bank)
    provenance_component_sets = build_provenance_component_control_sets(
        provenance_set
    )
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
    for control_name, control_set in provenance_component_sets.items():
        write_patient_tabular_features(
            OUTPUT_DIR / "audits" / f"patient_{control_name}.csv",
            *control_set,
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
    for control_name, control_set in provenance_component_sets.items():
        write_tabular_class_summary(
            OUTPUT_DIR / "audits" / f"{control_name}_class_summary.csv",
            control_set[0],
            control_set[1],
            control_set[3],
        )
    write_tabular_class_summary(
        OUTPUT_DIR / "audits" / "standardization_class_summary.csv",
        standardization_set[0],
        standardization_set[1],
        standardization_set[3],
    )
    if monai_qc_set is not None:
        write_tabular_class_summary(
            OUTPUT_DIR / "audits" / "monai_qc_class_summary.csv",
            monai_qc_set[0],
            monai_qc_set[1],
            monai_qc_set[3],
        )
    if standardized_monai_qc_set is not None:
        write_tabular_class_summary(
            OUTPUT_DIR / "audits" / "standardized_monai_qc_class_summary.csv",
            standardized_monai_qc_set[0],
            standardized_monai_qc_set[1],
            standardized_monai_qc_set[3],
        )

    tabular_feature_sets = {
        "provenance_only": provenance_set,
        "standardization_qc_only": standardization_set,
        **provenance_component_sets,
    }
    if monai_qc_set is not None:
        tabular_feature_sets["monai_qc_only"] = monai_qc_set
    if standardized_monai_qc_set is not None:
        tabular_feature_sets[
            "standardized_monai_qc_only"
        ] = standardized_monai_qc_set
    stage_durations["07 QC/provenance controls"] = _print_stage_complete(
        7,
        "Build and save provenance, standardization and MONAI-QC controls",
        stage_started,
        "Patient-level control matrices are ready for provenance and "
        "standardization; original MONAI QC is explicitly skipped when its "
        "historical branch was not computed, while standardized MONAI QC is "
        "retained for the focused cardiac representations.",
    )

    # ======================================================================
    # MAIN STAGE 8 -- RUN THE PREDECLARED EXPERIMENT REGISTRY
    # ======================================================================
    # Each experiment receives the same frozen outer-fold assignments. Any
    # quantitatively selected classifier C or decision threshold is learned only
    # from inner out-of-fold predictions belonging to the current outer-training
    # cohort. The outer-validation fold remains untouched until that fold's
    # preprocessing and classifier are locked. Prepared patient representations
    # are cached in memory, so the compact experiment panel changes one declared
    # scientific factor without repeating neural inference.
    #
    # One experiment failure is written to its own failure.json and does not
    # erase successful results unless FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS=True.
    stage_started = _print_stage_start(
        8,
        "Run every enabled experiment on the shared folds",
        "Sixteen focused patient-level fits; frozen image embeddings are reused.",
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
        preparation_key = experiment_preparation_cache_key(experiment)

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
    # locked A17 candidate by patient ID and computes paired bootstrap distributions of
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
    focused_row_contract_summary = write_focused_row_contract_audit(
        successful_results=successful_results,
        output_path=OUTPUT_DIR / "audits" / "focused_row_contract.json",
    )
    stage_durations["09 Master comparisons"] = _print_stage_complete(
        9,
        "Calculate paired comparisons and write master result files",
        stage_started,
        f"Summary rows={len(summary_rows)}; baseline-paired rows={len(paired_rows)}; "
        f"matched-ablation rows={len(primary_ablation_rows)}; "
        f"row_contract={focused_row_contract_summary.get('status', 'UNKNOWN')}.",
    )

    # ======================================================================
    # MAIN STAGE 10 -- REPEAT SELECTED NESTED-CV EXPERIMENTS ACROSS SPLITS
    # ======================================================================
    # A single five-fold assignment is statistically fragile when the effective
    # sample size is only the number of Directory_* folders. The prior run also
    # showed that shortcut controls could be nearly as predictive as the primary
    # model. This stage therefore repeats the full nested patient-level fitting
    # path not only for the baseline, but also for the main simple-localization
    # candidate, strict ROI candidate, and key padding/border/exterior controls.
    # No split is selected according to AUC and no neural inference is repeated.
    stage_started = _print_stage_start(
        10,
        "Run repeated nested-CV and candidate-selection analyses",
        "Patient-level reruns for the compact model/control panel and the "
        "predeclared candidate-selection rule; frozen features are reused.",
    )
    stability_summaries = {}
    stability_root = OUTPUT_DIR / "stability"
    stability_root.mkdir(parents=True, exist_ok=True)
    successful_by_id = {
        result["config"].experiment_id: result
        for result in successful_results
    }
    experiments_by_id = {
        experiment.experiment_id: experiment
        for experiment in experiments
    }
    model_selection_root = OUTPUT_DIR / "model_selection"
    model_selection_root.mkdir(parents=True, exist_ok=True)
    model_selection_prepared_by_id = {}
    model_selection_summary = {
        "status": "SKIPPED_DISABLED",
        "candidate_experiment_ids_in_priority_order": list(
            MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
        ),
    }

    if RUN_REPEATED_NESTED_CV_STABILITY:
        for stability_experiment_id in STABILITY_EXPERIMENT_IDS:
            experiment = experiments_by_id[stability_experiment_id]
            result = successful_by_id.get(stability_experiment_id)
            preparation_key = experiment_preparation_cache_key(experiment)

            if result is None:
                summary = {
                    "status": "SKIPPED_EXPERIMENT_FAILED",
                    "experiment_id": stability_experiment_id,
                }
                experiment_dir = stability_root / stability_experiment_id
                experiment_dir.mkdir(parents=True, exist_ok=True)
                (experiment_dir / "repeated_nested_cv_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(
                    f"[STABILITY] SKIPPED {stability_experiment_id} because "
                    "its primary experiment did not complete successfully.",
                    flush=True,
                )
            else:
                summary = run_repeated_nested_cv_stability(
                    experiment=experiment,
                    prepared=prepared_cache[preparation_key],
                    base_fold_manifest_rows=fold_manifest_rows,
                    output_dir=stability_root / stability_experiment_id,
                )
            stability_summaries[stability_experiment_id] = summary

        stability_paired_summary = compare_repeated_nested_cv_pairs(
            stability_root,
            REPEATED_STABILITY_COMPARISONS,
        )
        stability_summary = {
            "status": "OK",
            "repeats_per_experiment": int(REPEATED_NESTED_CV_REPEATS),
            "experiment_ids": list(STABILITY_EXPERIMENT_IDS),
            "experiment_summaries": stability_summaries,
            "paired_comparisons": stability_paired_summary,
            "interpretation": (
                "All configured models and controls are repeated over the same "
                "predeclared outer split seeds. Direct paired deltas are aligned "
                "by seed and no best split is selected."
            ),
        }
    else:
        stability_paired_summary = {
            "status": "SKIPPED_DISABLED",
            "comparisons": [],
        }
        stability_summary = {
            "status": "SKIPPED_DISABLED",
            "experiment_ids": list(STABILITY_EXPERIMENT_IDS),
            "paired_comparisons": stability_paired_summary,
        }
        print("[STABILITY] SKIPPED by configuration.", flush=True)

    # The focused suite also evaluates the complete candidate-selection rule.
    # Representation identity and classifier C are selected only inside each
    # outer-training cohort; no outer-validation score participates in the
    # choice between A17, A20, A12 or a declared future candidate.
    if RUN_NESTED_MODEL_SELECTION_AUDIT:
        missing_selection_candidates = [
            experiment_id
            for experiment_id in MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
            if experiment_id not in successful_by_id
        ]
        if missing_selection_candidates:
            model_selection_summary = {
                "status": "SKIPPED_CANDIDATE_EXPERIMENT_FAILED",
                "candidate_experiment_ids_in_priority_order": list(
                    MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
                ),
                "missing_experiments": missing_selection_candidates,
            }
            print(
                "[MODEL SELECTION] SKIPPED because candidate experiments "
                f"failed: {missing_selection_candidates}.",
                flush=True,
            )
        else:
            try:
                for experiment_id in MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS:
                    experiment = experiments_by_id[experiment_id]
                    preparation_key = experiment_preparation_cache_key(experiment)
                    model_selection_prepared_by_id[experiment_id] = (
                        prepared_cache[preparation_key]
                    )
                model_selection_summary = run_nested_candidate_selection_audit(
                    candidate_experiment_ids=(
                        MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
                    ),
                    experiments_by_id=experiments_by_id,
                    prepared_by_id=model_selection_prepared_by_id,
                    fold_manifest_rows=fold_manifest_rows,
                    output_dir=model_selection_root,
                )
                if RUN_REPEATED_MODEL_SELECTION_STABILITY:
                    repeated_selection_summary = (
                        run_repeated_nested_candidate_selection_stability(
                            candidate_experiment_ids=(
                                MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
                            ),
                            experiments_by_id=experiments_by_id,
                            prepared_by_id=model_selection_prepared_by_id,
                            base_fold_manifest_rows=fold_manifest_rows,
                            output_dir=model_selection_root,
                        )
                    )
                else:
                    repeated_selection_summary = {
                        "status": "SKIPPED_DISABLED"
                    }
                model_selection_summary[
                    "repeated_candidate_selection_stability"
                ] = repeated_selection_summary
            except Exception as error:
                model_selection_summary = {
                    "status": "FAILED",
                    "candidate_experiment_ids_in_priority_order": list(
                        MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
                    ),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                print(
                    "[MODEL SELECTION] FAILED: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                traceback.print_exc(file=sys.stdout)
                if FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS:
                    raise
    else:
        print("[MODEL SELECTION] SKIPPED by configuration.", flush=True)

    (model_selection_root / "candidate_selection_summary.json").write_text(
        json.dumps(model_selection_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    stability_summary["nested_candidate_selection"] = model_selection_summary

    (stability_root / "repeated_nested_cv_summary.json").write_text(
        json.dumps(stability_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    stability_ranking_rows = write_stability_ranking(
        summary_rows=summary_rows,
        stability_summaries=stability_summaries,
        output_path=stability_root / "focused_repeated_cv_ranking.csv",
    )
    successful_stability_runs = sum(
        summary.get("status") == "OK"
        for summary in stability_summaries.values()
    )
    stage_durations["10 Repeated CV and model selection"] = _print_stage_complete(
        10,
        "Run repeated nested-CV and candidate-selection analyses",
        stage_started,
        f"Completed={successful_stability_runs}/{len(STABILITY_EXPERIMENT_IDS)} "
        f"stability experiments; candidate_selection="
        f"{model_selection_summary.get('status', 'UNKNOWN')}; "
        f"ranking_rows={len(stability_ranking_rows)}.",
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
        "Run ordinary and selection-adjusted patient-label permutations",
        "Many nested patient-level fits; frozen neural-network features are reused.",
    )
    permutation_summaries = {}
    selection_adjusted_permutation_summary = {
        "status": "SKIPPED_DISABLED"
    }

    if RUN_PATIENT_LABEL_PERMUTATION_TEST:
        for permutation_experiment_id in PERMUTATION_EXPERIMENT_IDS:
            permutation_result = successful_by_id.get(
                permutation_experiment_id
            )
            permutation_experiment = experiments_by_id[
                permutation_experiment_id
            ]
            experiment_output_dir = (
                OUTPUT_DIR / "permutation" / permutation_experiment_id
            )
            experiment_output_dir.mkdir(parents=True, exist_ok=True)

            if permutation_result is None:
                summary = {
                    "status": "SKIPPED_EXPERIMENT_FAILED",
                    "experiment_id": permutation_experiment_id,
                }
                (
                    experiment_output_dir
                    / "patient_label_permutation_summary.json"
                ).write_text(
                    json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(
                    f"[PERMUTATION] SKIPPED {permutation_experiment_id} "
                    "because its primary experiment failed.",
                    flush=True,
                )
            else:
                preparation_key = experiment_preparation_cache_key(
                    permutation_experiment
                )
                observed_auc = float(
                    permutation_result["summary"]["metrics"]["auc"]
                )
                summary = run_patient_label_permutation_test(
                    experiment=permutation_experiment,
                    prepared=prepared_cache[preparation_key],
                    base_fold_manifest_rows=fold_manifest_rows,
                    observed_auc=observed_auc,
                    output_dir=experiment_output_dir,
                )
            permutation_summaries[permutation_experiment_id] = summary

        permutation_summary = {
            "status": "OK",
            "experiment_ids": list(PERMUTATION_EXPERIMENT_IDS),
            "experiment_summaries": permutation_summaries,
            "interpretation": (
                "Each configured patient-embedding model is tested by "
                "permuting Directory_* labels and rerunning the complete nested "
                "fitting path. This tests association/implementation sanity, "
                "not anatomical validity or external generalization."
            ),
        }
    else:
        permutation_summary = {
            "status": "SKIPPED_DISABLED",
            "experiment_ids": list(PERMUTATION_EXPERIMENT_IDS),
            "experiment_summaries": {},
        }
        print("[PERMUTATION] SKIPPED by configuration.", flush=True)

    if RUN_SELECTION_ADJUSTED_PERMUTATION_TEST:
        if model_selection_summary.get("status") != "OK":
            selection_adjusted_permutation_summary = {
                "status": "SKIPPED_MODEL_SELECTION_UNAVAILABLE",
                "model_selection_status": model_selection_summary.get(
                    "status", "UNKNOWN"
                ),
            }
            print(
                "[SELECTION-ADJUSTED PERMUTATION] SKIPPED because the "
                "observed candidate-selection audit is unavailable.",
                flush=True,
            )
        else:
            try:
                selection_adjusted_permutation_summary = (
                    run_selection_adjusted_patient_label_permutation_test(
                        candidate_experiment_ids=(
                            MODEL_SELECTION_CANDIDATE_EXPERIMENT_IDS
                        ),
                        experiments_by_id=experiments_by_id,
                        prepared_by_id=model_selection_prepared_by_id,
                        base_fold_manifest_rows=fold_manifest_rows,
                        observed_auc=float(
                            model_selection_summary[
                                "observed_selection_aware_auc"
                            ]
                        ),
                        output_dir=(
                            model_selection_root
                            / "selection_adjusted_permutation"
                        ),
                    )
                )
            except Exception as error:
                selection_adjusted_permutation_summary = {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                print(
                    "[SELECTION-ADJUSTED PERMUTATION] FAILED: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                traceback.print_exc(file=sys.stdout)
                if FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS:
                    raise
    else:
        print(
            "[SELECTION-ADJUSTED PERMUTATION] SKIPPED by configuration.",
            flush=True,
        )

    permutation_summary[
        "selection_adjusted_candidate_family"
    ] = selection_adjusted_permutation_summary
    model_selection_summary[
        "selection_adjusted_patient_label_permutation"
    ] = selection_adjusted_permutation_summary
    (model_selection_root / "candidate_selection_summary.json").write_text(
        json.dumps(model_selection_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (
        OUTPUT_DIR / "permutation" / "patient_label_permutation_summary.json"
    ).write_text(
        json.dumps(permutation_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    stage_durations["11 Ordinary and adjusted permutation"] = _print_stage_complete(
        11,
        "Run ordinary and selection-adjusted patient-label permutations",
        stage_started,
        f"ordinary={permutation_summary.get('status', 'UNKNOWN')}; "
        f"selection_adjusted="
        f"{selection_adjusted_permutation_summary.get('status', 'UNKNOWN')}.",
    )

    # ======================================================================
    # MAIN STAGE 12 -- OPTIONAL BLINDED SEQUENCE/VIEW SUBSET ANALYSIS
    # ======================================================================
    # The released JPEG folders do not contain sufficient DICOM metadata to
    # infer sequence/view identity safely. This stage therefore remains disabled
    # until a reviewer completes a copy of series_annotation_template.csv. When
    # enabled, it applies the predeclared annotation filters before pooling,
    # creates a fresh patient-level fold manifest for the retained cohort, and
    # reruns only the selected image experiments. Empty annotation fields are
    # never guessed and Normal/Sick labels remain outside the blinded template.
    stage_started = _print_stage_start(
        12,
        "Run optional annotated sequence/view subset analysis",
        "Immediate when disabled; otherwise several patient-level nested-CV fits.",
    )
    series_annotation_summary = run_annotated_series_subset_analysis(
        bank=bank,
        experiments=experiments,
        tabular_feature_sets=tabular_feature_sets,
        base_fold_manifest_rows=fold_manifest_rows,
    )
    stage_durations["12 Annotated sequence/view subset"] = (
        _print_stage_complete(
            12,
            "Run optional annotated sequence/view subset analysis",
            stage_started,
            series_annotation_summary.get("status", "UNKNOWN"),
        )
    )

    # ======================================================================
    # MAIN STAGE 13 -- RECORD, BUT DO NOT FABRICATE, EXTERNAL VALIDATION
    # ======================================================================
    # External validation cannot be made valid merely by pointing the script at
    # an arbitrary cardiac dataset. A verified adapter must first establish a
    # comparable CAD endpoint, one independent patient unit, image/sequence
    # compatibility, and a locked preprocessing contract. Until such an adapter
    # exists, the suite writes an explicit SKIPPED/BLOCKED status instead of
    # silently adapting parameters after inspecting external labels.
    stage_started = _print_stage_start(
        13,
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
    stage_durations["13 External validation status"] = _print_stage_complete(
        13,
        "Record external-validation status",
        stage_started,
        external_status["status"],
    )

    # ======================================================================
    # MAIN STAGE 14 -- PRINT THE FINAL TABLE AND SAVE COMPLETE PROVENANCE
    # ======================================================================
    # The final console table ranks successful experiments but also prints
    # explicit shortcut warnings for negative controls above the configured AUC
    # threshold. The metadata JSON records software versions, model hashes,
    # feature-bank identity, patient counts, audit summaries, successful and
    # failed experiments, and total runtime. All ordinary print() output and
    # tracebacks are simultaneously preserved in console_output.log.
    stage_started = _print_stage_start(
        14,
        "Print final comparison and save suite metadata",
        "Console table, warnings, metadata JSON and timing summary.",
    )
    print_final_comparison(summary_rows, failed_results)
    print_primary_ablation_comparisons(primary_ablation_rows)
    print_focused_candidate_selection_summary(
        model_selection_summary=model_selection_summary,
        selection_adjusted_permutation_summary=(
            selection_adjusted_permutation_summary
        ),
    )
    focused_scorecard = write_focused_research_scorecard(
        summary_rows=summary_rows,
        stability_summary=stability_summary,
        primary_ablation_rows=primary_ablation_rows,
        model_selection_summary=model_selection_summary,
        permutation_summary=permutation_summary,
        series_annotation_summary=series_annotation_summary,
        external_status=external_status,
        output_dir=OUTPUT_DIR / "comparison",
    )
    augment_focused_final_report(
        comparison_dir=OUTPUT_DIR / "comparison",
        focused_scorecard=focused_scorecard,
        stability_summary=stability_summary,
        model_selection_summary=model_selection_summary,
        permutation_summary=permutation_summary,
        series_annotation_summary=series_annotation_summary,
        external_status=external_status,
        focused_row_contract_summary=focused_row_contract_summary,
    )

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
        model_selection_summary=model_selection_summary,
        permutation_summary=permutation_summary,
        focused_scorecard=focused_scorecard,
        focused_row_contract_summary=focused_row_contract_summary,
        series_annotation_summary=series_annotation_summary,
        successful_results=successful_results,
        failed_results=failed_results,
        total_runtime=total_runtime,
    )
    (OUTPUT_DIR / "suite_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    stage_durations["14 Final reporting"] = _print_stage_complete(
        14,
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
# 2. Automatic sequence/view inference remains prohibited. V7.1 can run an
#    optional balanced subset analysis only after a reviewer completes the blinded CSV;
#    SR_* and series* names alone are never treated as validated sequence labels.
# 3. True external validation requires an independent cohort adapter with a
#    comparable CAD endpoint, patient unit and locked preprocessing contract.
#
# These are recorded as scientific next steps rather than silently approximated.
