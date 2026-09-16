# Inițializare unică + codul complet al pipeline-ului
#
# The class is evaluated before the remaining pipeline settings so environment
# overrides are applied consistently in scripts and notebooks.

import os


class SETTINGS_BOOTSTRAP:
    """Environment defaults that must be set before the pipeline is imported."""

    CAD_VALIDATION_PROFILE = "fast"
    CAD_RUNTIME_DEVICE = "cpu"
    CAD_SUITE_DEVICE_TAG = "cuda"
    CAD_FEATURE_CACHE_DEVICE_TAG = "cuda"
    CAD_ATTENTION_STORE_AUTOMATIC_MASKS_IN_TRANSIENT = "0"
    CAD_ATTENTION_STORE_PREDICTED_MASKS_IN_TRANSIENT = "0"
    CAD_ATTENTION_SAVE_ALL_PREDICTED_MASKS = "1"

    REVIEW_SCOPE = "diverse"
    REVIEW_LIMIT = 1200
    REVIEW_SEED = 42


os.environ.setdefault(
    "CAD_VALIDATION_PROFILE", SETTINGS_BOOTSTRAP.CAD_VALIDATION_PROFILE
)
os.environ["CAD_RUNTIME_DEVICE"] = SETTINGS_BOOTSTRAP.CAD_RUNTIME_DEVICE
os.environ["CAD_SUITE_DEVICE_TAG"] = SETTINGS_BOOTSTRAP.CAD_SUITE_DEVICE_TAG
os.environ["CAD_FEATURE_CACHE_DEVICE_TAG"] = (
    SETTINGS_BOOTSTRAP.CAD_FEATURE_CACHE_DEVICE_TAG
)
os.environ["CAD_ATTENTION_STORE_AUTOMATIC_MASKS_IN_TRANSIENT"] = (
    SETTINGS_BOOTSTRAP.CAD_ATTENTION_STORE_AUTOMATIC_MASKS_IN_TRANSIENT
)
os.environ["CAD_ATTENTION_STORE_PREDICTED_MASKS_IN_TRANSIENT"] = (
    SETTINGS_BOOTSTRAP.CAD_ATTENTION_STORE_PREDICTED_MASKS_IN_TRANSIENT
)
os.environ["CAD_ATTENTION_SAVE_ALL_PREDICTED_MASKS"] = (
    SETTINGS_BOOTSTRAP.CAD_ATTENTION_SAVE_ALL_PREDICTED_MASKS
)

# ---------------------------------------------------------------------------
# CODUL COMPLET AL PIPELINE-ULUI
# ---------------------------------------------------------------------------

#%% ============================================================
"""CAD cardiac-MRI patient-level research pipeline.

DATA CONTRACT
-------------
* ``Directory_*`` is the patient unit and never crosses train/validation folds.
* Immediate child folders are series proxies, not verified DICOM UIDs.
* ``Normal``/``Sick`` supplies one patient label; slices are not independent
  labelled observations.

CURRENT PIPELINE
----------------
1. Discover images while preserving patient and series-proxy grouping.
2. Apply label-blind geometric/intensity standardization.
3. Localize ventricular anatomy with the pinned MONAI TorchScript model.
4. Encode the final candidate and matched controls with frozen EfficientNet-B0.
5. Pool slice embeddings to series proxies and then to one patient vector.
6. Fit fold-local scaler, PCA and Logistic Regression inside nested patient CV.
7. Report patient-level OOF metrics, repeated-CV stability and label permutation.
8. Optionally train cross-fitted Attention U-Net models from MONAI pseudo-masks
   plus every saved manual correction and evaluate AU1-AU5.

The active MONAI panel contains one reference, one primary candidate, one
valid-only ablation and four shortcut/anatomy controls. GPU stages perform
neural inference/training; HTML review and cached statistical evaluation run on
CPU. Results are exploratory and not externally validated clinical outputs.
"""
# ---------------------------------------------------------------------------
# VS CODE OUTLINE MAP
# ---------------------------------------------------------------------------
# Important settings are summarized by RuntimeSettings, ValidationSettings,
# MonaiSettings, FeatureBankSettings, AuditSettings, AttentionSettings and
# ReviewSettings. Functional operations are grouped in manager classes near the
# end of this file. Module-level aliases preserve the existing API and cached
# workflow behavior.
# ---------------------------------------------------------------------------


# =============================
# IMPORTS
# =============================
import csv

import gc

import shutil

import sys

import traceback

from collections import OrderedDict, defaultdict

from dataclasses import asdict, dataclass

from itertools import combinations


import hashlib

import math

import json

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import platform

import time

from pathlib import Path

import cv2

import numpy as np

from tqdm import tqdm

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision

import torchvision.transforms as transforms

from torchvision import models

# MONAI is imported lazily only when the verified TorchScript artifact cannot
# be used and reconstruction from model.pt is required.

import sklearn

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


from IPython.display import display, HTML, Javascript


# =============================
# SETTINGS TYPES
# =============================

# ---------------------------------------------------------------------------
# MULTI-EXPERIMENT SUITE CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable description of one auditable experiment in the suite."""

    experiment_id: str
    description: str
    feature_mode: str
    strategy: str = "patient_embedding"
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
    #   "standardized_monai_valid" -> retain only slices for which the
    #       standardized MONAI output passes the predeclared plausibility gate.
    # Matched inside/outside experiments use the same filter so gate failure
    # cannot expose full images in one branch or create zero images in the other.
    role: str = "ablation"
    enabled: bool = True


# region VS CODE OUTLINE — ALL SETTINGS

def _validation_env_positive_int(name, default):
    """Read one strictly positive integer used by repeated analyses."""

    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, received {raw!r}.") from error
    if value < 1:
        raise ValueError(f"{name} must be at least 1, received {value}.")
    return value

def _validation_env_bool(name, default):
    """Read a conventional Boolean environment flag."""

    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name} must be a Boolean token (0/1, true/false), received {raw!r}."
    )

def _env_int(name, default, minimum=1):
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value}.")
    return value

def _env_float(name, default, minimum=None, maximum=None):
    value = float(os.environ.get(name, default))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}; received {value}.")
    return value

def _env_bool(name, default=False):
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a Boolean value; received {value!r}.")

class SETTINGS_RUNTIME:
    """Runtime, reproducibility and progress controls."""

    PIPELINE_SCHEMA_ID = "cad-cardiac-mri-current"

    # Primary MONAI/EfficientNet image batch size.
    # Lower this first after a CUDA OOM; it affects speed, not model semantics.
    BATCH_SIZE = 8

    RUNTIME_DEVICE_CHOICES = ("auto", "cpu", "cuda")

    # Default device policy. Use explicit CPU/GPU staged actions in Kaggle.
    RUNTIME_DEVICE_POLICY = os.environ.get(
        "CAD_RUNTIME_DEVICE", "auto"
    ).strip().lower()

    # Keep this tag identical in GPU-generation and CPU-evaluation sessions
    # so both sessions resolve the same output directory.
    SUITE_DEVICE_TAG = os.environ.get(
        "CAD_SUITE_DEVICE_TAG", "auto"
    ).strip().lower()

    # Master seed for patient folds, bootstrap and repeated analyses.
    RANDOM_SEED = 42

    USE_CUDA_AMP = False

    DATALOADER_NUM_WORKERS = 0

    ENABLE_DETAILED_PROGRESS_PRINTS = True

    PROGRESS_PRINT_EVERY_N_BATCHES = 25

class SETTINGS_EXPERIMENTS:
    """Active MONAI experiments and predeclared paired comparisons."""

    EXPERIMENTS_TO_RUN = None

    REFERENCE_EXPERIMENT_ID = (
        "A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA"
    )

    PRIMARY_CANDIDATE_EXPERIMENT_ID = (
        "A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA"
    )

    VALID_ONLY_ABLATION_EXPERIMENT_ID = (
        "A20_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA"
    )

    FIXED_PERIPHERY_CONTROL_EXPERIMENT_ID = (
        "C29_FIXED_PERIPHERY_REGION_NORM_HIER_LR_PCA"
    )

    MASK_ONLY_CONTROL_EXPERIMENT_ID = (
        "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA"
    )

    SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID = (
        "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA"
    )

    COMPLEMENT_CONTROL_EXPERIMENT_ID = (
        "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA"
    )

    BASELINE_EXPERIMENT_ID = REFERENCE_EXPERIMENT_ID

    EXPERIMENT_REGISTRY = (
        ExperimentConfig(
            experiment_id=REFERENCE_EXPERIMENT_ID,
            description=(
                "Reference model using a zero-background MONAI ROI with a fixed "
                "central fallback, hierarchical patient pooling, PCA and Logistic "
                "Regression."
            ),
            feature_mode="standardized_roi_zero_bg_center_fallback",
            strategy="patient_embedding",
            role="reference",
        ),
        ExperimentConfig(
            experiment_id=PRIMARY_CANDIDATE_EXPERIMENT_ID,
            description=(
                "Primary model using a binary dilated cardiac support, robust "
                "intensity scaling inside the visible region and fixed-centre "
                "fallback."
            ),
            feature_mode="standardized_hard_support_region_norm",
            strategy="patient_embedding",
            role="primary_candidate",
        ),
        ExperimentConfig(
            experiment_id=VALID_ONLY_ABLATION_EXPERIMENT_ID,
            description=(
                "Primary model restricted to slices whose standardized MONAI mask "
                "passes the plausibility gate."
            ),
            feature_mode="standardized_hard_support_region_norm",
            strategy="patient_embedding",
            slice_filter="standardized_monai_valid",
            role="ablation",
        ),
        ExperimentConfig(
            experiment_id=FIXED_PERIPHERY_CONTROL_EXPERIMENT_ID,
            description=(
                "MONAI-independent peripheral control retaining only pixels outside "
                "a fixed central square."
            ),
            feature_mode="standardized_fixed_periphery_region_norm",
            strategy="patient_embedding",
            role="negative_control",
        ),
        ExperimentConfig(
            experiment_id=MASK_ONLY_CONTROL_EXPERIMENT_ID,
            description=(
                "Exact binary support used by the primary model with all MRI "
                "intensities removed."
            ),
            feature_mode="standardized_a17_exact_support_mask_only",
            strategy="patient_embedding",
            role="segmentation_control",
        ),
        ExperimentConfig(
            experiment_id=SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
            description=(
                "Exact support and intensity histogram retained while spatial "
                "anatomy is deterministically shuffled."
            ),
            feature_mode="standardized_a17_support_intensity_affine_shuffled",
            strategy="patient_embedding",
            role="anatomy_destruction_control",
        ),
        ExperimentConfig(
            experiment_id=COMPLEMENT_CONTROL_EXPERIMENT_ID,
            description=(
                "Independently normalized non-padding complement of the exact "
                "primary-model support."
            ),
            feature_mode="standardized_a17_exact_support_complement_region_norm",
            strategy="patient_embedding",
            role="negative_control",
        ),
    )

    PRIMARY_ABLATION_COMPARISONS = (
        (
            "REFERENCE_VS_PRIMARY",
            REFERENCE_EXPERIMENT_ID,
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "Hard-support candidate versus strict-ROI reference.",
        ),
        (
            "PRIMARY_ALL_VS_VALID_ONLY",
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            VALID_ONLY_ABLATION_EXPERIMENT_ID,
            "Effect of removing fallback slices.",
        ),
        (
            "PERIPHERY_VS_PRIMARY",
            FIXED_PERIPHERY_CONTROL_EXPERIMENT_ID,
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "Cardiac candidate versus a MONAI-independent periphery.",
        ),
        (
            "MASK_ONLY_VS_PRIMARY",
            MASK_ONLY_CONTROL_EXPERIMENT_ID,
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "Support plus MRI intensity versus support geometry alone.",
        ),
        (
            "SHUFFLED_INTENSITY_VS_PRIMARY",
            SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "Intact spatial anatomy versus the same histogram after shuffling.",
        ),
        (
            "COMPLEMENT_VS_PRIMARY",
            COMPLEMENT_CONTROL_EXPERIMENT_ID,
            PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "Primary support versus its exact complement.",
        ),
    )

class SETTINGS_VALIDATION:
    """Patient-level CV, bootstrap, stability and permutation settings."""

    N_SPLITS = 5

    CV_RANDOM_STATE = SETTINGS_RUNTIME.RANDOM_SEED

    INNER_CV_SPLITS = 3

    INNER_CV_RANDOM_STATE = SETTINGS_RUNTIME.RANDOM_SEED + 1000

    CLASSIFIER_C_GRID = (0.01, 0.1, 1.0, 10.0)

    LOGISTIC_MAX_ITER = 4000

    PATIENT_PCA_EXPLAINED_VARIANCE = 0.95

    THRESHOLD_SELECTION_METHOD = "youden"

    TARGET_SENSITIVITY = 0.90

    BOOTSTRAP_REPLICATES = 2000

    PAIRED_BOOTSTRAP_REPLICATES = 2000

    BOOTSTRAP_CONFIDENCE = 0.95

    C_SELECTION_AUC_TOLERANCE = 0.01

    # smoke = code checks only; fast = iterative work; full = final reporting.
    VALIDATION_RUNTIME_PROFILE = os.environ.get(
        "CAD_VALIDATION_PROFILE", "fast"
    ).strip().lower()

    VALIDATION_RUNTIME_PROFILES = {
        "smoke": {
            "stability_repeats": 3,
            "permutation_replicates": 25,
            "selection_adjusted_permutations": 25,
        },
        "fast": {
            "stability_repeats": 10,
            "permutation_replicates": 200,
            "selection_adjusted_permutations": 100,
        },
        "full": {
            "stability_repeats": 50,
            "permutation_replicates": 1000,
            "selection_adjusted_permutations": 1000,
        },
    }

    _VALIDATION_PROFILE_DEFAULTS = VALIDATION_RUNTIME_PROFILES[
        VALIDATION_RUNTIME_PROFILE
    ]

    RUN_REPEATED_NESTED_CV_STABILITY = _validation_env_bool(
        "CAD_RUN_STABILITY", True
    )

    REPEATED_NESTED_CV_REPEATS = _validation_env_positive_int(
        "CAD_STABILITY_REPEATS",
        _VALIDATION_PROFILE_DEFAULTS["stability_repeats"],
    )

    REPEATED_NESTED_CV_RANDOM_STATE = SETTINGS_RUNTIME.RANDOM_SEED + 20_000

    STABILITY_EXPERIMENT_IDS = tuple(
        experiment.experiment_id for experiment in SETTINGS_EXPERIMENTS.EXPERIMENT_REGISTRY
    )

    REPEATED_STABILITY_COMPARISONS = (
        ("REFERENCE_VS_PRIMARY", SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID, SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
         "Strict-ROI reference versus primary hard-support model."),
        ("PRIMARY_ALL_VS_VALID_ONLY", SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
         SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID, "All slices versus gate-valid-only slices."),
        ("MASK_ONLY_VS_PRIMARY", SETTINGS_EXPERIMENTS.MASK_ONLY_CONTROL_EXPERIMENT_ID,
         SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID, "Support geometry alone versus support plus MRI."),
        ("SHUFFLED_VS_PRIMARY", SETTINGS_EXPERIMENTS.SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
         SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID, "Shuffled intensities versus intact anatomy."),
        ("COMPLEMENT_VS_PRIMARY", SETTINGS_EXPERIMENTS.COMPLEMENT_CONTROL_EXPERIMENT_ID,
         SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID, "Exact complement versus cardiac support."),
        ("PERIPHERY_VS_PRIMARY", SETTINGS_EXPERIMENTS.FIXED_PERIPHERY_CONTROL_EXPERIMENT_ID,
         SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID, "Fixed periphery versus cardiac support."),
    )

    RUN_PATIENT_LABEL_PERMUTATION_TEST = _validation_env_bool(
        "CAD_RUN_PERMUTATION", True
    )

    LABEL_PERMUTATION_REPLICATES = _validation_env_positive_int(
        "CAD_PERMUTATION_REPLICATES",
        _VALIDATION_PROFILE_DEFAULTS["permutation_replicates"],
    )

    LABEL_PERMUTATION_RANDOM_STATE = SETTINGS_RUNTIME.RANDOM_SEED + 40_000

    PERMUTATION_EXPERIMENT_IDS = (
        SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
    )

    RUN_MODEL_FAMILY_NESTED_SELECTION = True

    MODEL_FAMILY_NESTED_SELECTION_IDS = (
        SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
        SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,
        SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
    )

    MODEL_FAMILY_SELECTION_AUC_TOLERANCE = 0.01

    RUN_SELECTION_ADJUSTED_PERMUTATION_TEST = _validation_env_bool(
        "CAD_RUN_SELECTION_ADJUSTED_PERMUTATION", True
    )

    SELECTION_ADJUSTED_PERMUTATION_REPLICATES = (
        _validation_env_positive_int(
            "CAD_SELECTION_ADJUSTED_PERMUTATIONS",
            _VALIDATION_PROFILE_DEFAULTS[
                "selection_adjusted_permutations"
            ],
        )
    )

    SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE = SETTINGS_RUNTIME.RANDOM_SEED + 60_000

    SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS = (
        SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
        SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
        SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,
    )

class SETTINGS_FEATURE_BANK:
    """Frozen EfficientNet encoder and persistent cache policy."""

    USE_FEATURE_CACHE = True

    # Keep False normally. True forces an expensive full neural cache rebuild.
    FORCE_REBUILD_FEATURE_CACHE = False

    # CPU-only evaluation sets this True to prevent accidental neural inference.
    REQUIRE_EXISTING_FEATURE_CACHE = False

    FEATURE_CACHE_DEVICE_TAG = os.environ.get(
        "CAD_FEATURE_CACHE_DEVICE_TAG", "auto"
    ).strip().lower()

    FEATURE_CACHE_SCHEMA = "cad-standardized-panel-current"

    EFFICIENTNET_FEATURE_DIM = 1280

    # Increase for throughput only when VRAM allows; lower after an OOM.
    FEATURE_MODES_PER_ENCODER_CALL = 4

    EFFICIENTNET_WEIGHTS_NAME = "IMAGENET1K_V1"

    EFFICIENTNET_MEAN = (0.485, 0.456, 0.406)

    EFFICIENTNET_STD = (0.229, 0.224, 0.225)

class SETTINGS_PREPROCESSING:
    """Image standardization, region scaling and exact-support controls."""

    IMG_SIZE = 224

    CENTER_CROP_FALLBACK_FRACTION = 0.60

    STANDARDIZED_CONTENT_LONG_SIDE = 240

    HARD_SUPPORT_FALLBACK_FRACTION = 0.65

    # Must remain odd. Changing it invalidates hard-support feature caches.
    HARD_SUPPORT_DILATION_KERNEL = 15

    FIXED_PERIPHERY_EXCLUSION_FRACTION = 0.75

    # Robust scaling is estimated only from pixels retained by each representation.
    REGION_NORM_LOWER_PERCENTILE = 1.0

    REGION_NORM_UPPER_PERCENTILE = 99.0

    REGION_NORM_MIN_PIXELS = 64

    REGION_NORM_HISTOGRAM_BINS = 256

    REGION_NORM_MIN_DYNAMIC_RANGE = 8.0 / 255.0

    SUPPORT_INTENSITY_SHUFFLE_SCHEMA = "sha256-affine-permutation-v1"

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

class SETTINGS_AUDIT:
    """Duplicate-audit and shortcut-warning safeguards."""

    AUDIT_EXACT_DECODED_PIXEL_DUPLICATES = True

    AUDIT_PERCEPTUAL_NEAR_DUPLICATES = True

    PHASH_HAMMING_THRESHOLD = 3

    GROUP_SPLITS_BY_EXACT_DUPLICATES = True

    GROUP_SPLITS_BY_PHASH_CANDIDATES = False

    FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES = False

    FAIL_ON_CROSS_LABEL_EXACT_DUPLICATES = False

    FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS = False

    SHORTCUT_WARNING_AUC = 0.65

    MONAI_GATE_RATE_DIFFERENCE_WARNING = 0.20

class SETTINGS_REPORTING:
    """Pipeline-stage and patient-level metadata field definitions."""

    PIPELINE_STAGE_COUNT = 12

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

class SETTINGS_MONAI:
    """Pinned MONAI model artifact, ROI and plausibility-gate settings."""

    # Pinned by the pretrained MONAI bundle; do not change independently.
    MONAI_INPUT_SIZE = 256

    MONAI_BUNDLE_NAME = "ventricular_short_axis_3label"

    MONAI_BUNDLE_VERSION = "0.3.5"

    MONAI_HF_REPO_ID = "MONAI/ventricular_short_axis_3label"

    MONAI_HF_REVISION = "eefc17c8e002cc8a567bbfce8f02d7d3116408f4"

    AUTO_DOWNLOAD_MONAI_BUNDLE = True

    VERIFY_MONAI_ARTIFACT_SHA256 = True

    MONAI_OFFICIAL_TORCHSCRIPT_SHA256 = (
        "27d5532401fa6c1883872fa21635adbb7615981e7f385d0c58dd75b355e340b3"
    )

    MONAI_MODEL_SHA256 = (
        "464ca796028831f6c9e2b1cdaebe9af002fc1d7f494f7a89a63f2079e38837a1"
    )

    # Must remain odd. This controls contextual expansion around ventricular masks.
    MONAI_ROI_DILATION_KERNEL = 31

    MONAI_BACKGROUND_WEIGHT = 0.15

    # QC gate thresholds are fixed before CAD evaluation and must not be tuned
    # against AUROC or patient labels.
    MONAI_MIN_HEART_AREA_RATIO = 0.003

    MONAI_MAX_HEART_AREA_RATIO = 0.50

    MONAI_MIN_PEAK_HEART_PROBABILITY = 0.50

    FORCE_REBUILD_MONAI_TORCHSCRIPT = False

    # Bundle location can be overridden by MONAI_BUNDLE_DIR. On Kaggle it is
    # stored under /kaggle/working so the pinned artifact survives cell reruns.
    if Path("/kaggle/working").exists():
        _DEFAULT_BUNDLE_PARENT = Path("/kaggle/working/monai_bundles")
    else:
        try:
            _DEFAULT_BUNDLE_PARENT = Path(__file__).resolve().parent / "monai_bundles"
        except NameError:
            _DEFAULT_BUNDLE_PARENT = Path.cwd() / "monai_bundles"

    MONAI_BUNDLE_DIR = Path(
        os.environ.get("MONAI_BUNDLE_DIR", str(_DEFAULT_BUNDLE_PARENT))
    )

    # Local TorchScript is only the reconstruction fallback. The verified
    # official models/model.ts remains the preferred execution path.
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

class SETTINGS_ATTENTION:
    """Attention U-Net architecture, training, inference, storage and evaluation."""

    # Spatial size used by Attention U-Net training, inference and HTML masks.
    # Changing it invalidates checkpoints, predicted masks and Attention caches.
    ATTENTION_INPUT_SIZE = 256

    # Cross-fitting folds. Each patient is inferred by a model that excluded that
    # patient from segmentation training and early stopping.
    ATTENTION_SEGMENTATION_FOLDS = _env_int(
        "CAD_ATTENTION_UNET_FOLDS", 5, minimum=2
    )

    # Model width. Changing it makes existing Attention checkpoints incompatible.
    ATTENTION_BASE_CHANNELS = _env_int(
        "CAD_ATTENTION_UNET_BASE_CHANNELS", 24, minimum=4
    )

    ATTENTION_EPOCHS = _env_int("CAD_ATTENTION_UNET_EPOCHS", 12, minimum=1)

    ATTENTION_EARLY_STOPPING_PATIENCE = _env_int(
        "CAD_ATTENTION_UNET_EARLY_STOPPING_PATIENCE", 4, minimum=1
    )

    # Training batch size; lower this first after an Attention CUDA OOM.
    ATTENTION_BATCH_SIZE = _env_int(
        "CAD_ATTENTION_UNET_BATCH_SIZE", 12, minimum=1
    )

    # Inference batch size for full-cohort prediction and review-mask generation.
    ATTENTION_INFERENCE_BATCH_SIZE = _env_int(
        "CAD_ATTENTION_UNET_INFERENCE_BATCH_SIZE", SETTINGS_RUNTIME.BATCH_SIZE, minimum=1
    )

    ATTENTION_LEARNING_RATE = _env_float(
        "CAD_ATTENTION_UNET_LEARNING_RATE", 1e-3, minimum=1e-8
    )

    ATTENTION_WEIGHT_DECAY = _env_float(
        "CAD_ATTENTION_UNET_WEIGHT_DECAY", 1e-4, minimum=0.0
    )

    ATTENTION_BCE_WEIGHT = _env_float(
        "CAD_ATTENTION_UNET_BCE_WEIGHT", 0.5, minimum=0.0, maximum=1.0
    )

    ATTENTION_DICE_WEIGHT = 1.0 - ATTENTION_BCE_WEIGHT

    # Binarization threshold for Attention masks. Changing it invalidates every
    # mask-derived AU1-AU5 representation and feature cache.
    ATTENTION_MASK_THRESHOLD = _env_float(
        "CAD_ATTENTION_UNET_MASK_THRESHOLD", 0.50, minimum=0.0, maximum=1.0
    )

    # Attention QC thresholds are safeguards, not AUROC tuning parameters.
    ATTENTION_MIN_HEART_AREA_RATIO = _env_float(
        "CAD_ATTENTION_UNET_MIN_HEART_AREA_RATIO", 0.003, minimum=0.0, maximum=1.0
    )

    ATTENTION_MAX_HEART_AREA_RATIO = _env_float(
        "CAD_ATTENTION_UNET_MAX_HEART_AREA_RATIO", 0.65, minimum=0.0, maximum=1.0
    )

    ATTENTION_MIN_PEAK_PROBABILITY = _env_float(
        "CAD_ATTENTION_UNET_MIN_PEAK_PROBABILITY", 0.50, minimum=0.0, maximum=1.0
    )

    ATTENTION_SUPPORT_DILATION_KERNEL = _env_int(
        "CAD_ATTENTION_UNET_SUPPORT_DILATION_KERNEL", 15, minimum=1
    )

    ATTENTION_PSEUDO_MASK_DILATION_KERNEL = _env_int(
        "CAD_ATTENTION_UNET_PSEUDO_MASK_DILATION_KERNEL", 9, minimum=1
    )

    # Caps only pseudo-labelled training rows. It does not limit full inference.
    ATTENTION_MAX_TRAIN_SLICES_PER_SERIES = _env_int(
        "CAD_ATTENTION_UNET_MAX_TRAIN_SLICES_PER_SERIES", 5, minimum=1
    )

    # Every saved manual mask is included even when outside this pseudo-label cap.
    ATTENTION_MAX_TRAIN_SLICES_PER_PATIENT = _env_int(
        "CAD_ATTENTION_UNET_MAX_TRAIN_SLICES_PER_PATIENT", 40, minimum=1
    )

    ATTENTION_VALIDATION_PATIENT_FRACTION = _env_float(
        "CAD_ATTENTION_UNET_VALIDATION_PATIENT_FRACTION",
        0.20,
        minimum=0.05,
        maximum=0.50,
    )

    ATTENTION_TRAIN_WITH_AMP = _env_bool(
        "CAD_ATTENTION_UNET_TRAIN_WITH_AMP", True
    )

    ATTENTION_SAVE_ALL_PREDICTED_MASKS = _env_bool(
        "CAD_ATTENTION_SAVE_ALL_PREDICTED_MASKS", True
    )

    _ATTENTION_RUNNING_ON_KAGGLE = Path("/kaggle/working").exists()

    ATTENTION_CACHE_TRAINING_IMAGES = _env_bool(
        "CAD_ATTENTION_CACHE_TRAINING_IMAGES",
        not _ATTENTION_RUNNING_ON_KAGGLE,
    )

    # Transient masks under /kaggle/temp disappear after restart. Keep False when
    # a later CPU review session must reuse the generated masks.
    ATTENTION_STORE_AUTOMATIC_MASKS_IN_TRANSIENT = _env_bool(
        "CAD_ATTENTION_STORE_AUTOMATIC_MASKS_IN_TRANSIENT",
        _ATTENTION_RUNNING_ON_KAGGLE,
    )

    # Keep False when predicted Attention masks must survive a Kaggle restart.
    ATTENTION_STORE_PREDICTED_MASKS_IN_TRANSIENT = _env_bool(
        "CAD_ATTENTION_STORE_PREDICTED_MASKS_IN_TRANSIENT",
        _ATTENTION_RUNNING_ON_KAGGLE,
    )

    ATTENTION_RAM_IMAGE_CACHE_MB = _env_int(
        "CAD_ATTENTION_RAM_IMAGE_CACHE_MB",
        384 if _ATTENTION_RUNNING_ON_KAGGLE else 256,
        minimum=0,
    )

    ATTENTION_OPTIONAL_DISK_CACHE_MIN_FREE_MB = _env_int(
        "CAD_ATTENTION_OPTIONAL_DISK_CACHE_MIN_FREE_MB",
        512,
        minimum=0,
    )

    ATTENTION_PERSISTENT_FREE_SPACE_WARNING_MB = _env_int(
        "CAD_ATTENTION_PERSISTENT_FREE_SPACE_WARNING_MB",
        512,
        minimum=0,
    )

    ATTENTION_PNG_COMPRESSION = _env_int(
        "CAD_ATTENTION_PNG_COMPRESSION",
        9,
        minimum=0,
    )

    ATTENTION_MANIFEST_CHECKPOINT_EVERY_BATCHES = _env_int(
        "CAD_ATTENTION_MANIFEST_CHECKPOINT_EVERY_BATCHES",
        10,
        minimum=1,
    )

    ATTENTION_RUN_REPEATED_CV_STABILITY = _env_bool(
        "CAD_ATTENTION_UNET_RUN_STABILITY",
        SETTINGS_VALIDATION.RUN_REPEATED_NESTED_CV_STABILITY,
    )

    ATTENTION_RUN_PERMUTATION_TEST = _env_bool(
        "CAD_ATTENTION_UNET_RUN_PERMUTATION",
        SETTINGS_VALIDATION.RUN_PATIENT_LABEL_PERMUTATION_TEST,
    )

    ATTENTION_REPEATED_CV_REPEATS = _env_int(
        "CAD_ATTENTION_UNET_REPEATED_CV_REPEATS",
        SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS,
        minimum=1,
    )

    ATTENTION_PERMUTATION_REPLICATES = _env_int(
        "CAD_ATTENTION_UNET_PERMUTATIONS",
        SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES,
        minimum=1,
    )

    ATTENTION_RANDOM_SEED = _env_int(
        "CAD_ATTENTION_UNET_RANDOM_SEED", SETTINGS_RUNTIME.RANDOM_SEED + 70_000, minimum=0
    )

    ATTENTION_EXTERNAL_WEIGHTS = os.environ.get(
        "CAD_ATTENTION_UNET_WEIGHTS", ""
    ).strip()

    # Change this schema whenever AU1-AU5 representation semantics change.
    ATTENTION_FEATURE_CACHE_SCHEMA = (
        "2026-09-11-attention-unet-crossfit-patient-disk-safe-v2"
    )

    ATTENTION_FEATURE_MODES = (
        "AU1_ATTENTION_HARD_SUPPORT_REGION_NORM",
        "AU2_ATTENTION_HARD_SUPPORT_REGION_NORM_VALID_ONLY",
        "AU3_ATTENTION_EXACT_SUPPORT_MASK_ONLY",
        "AU4_ATTENTION_SUPPORT_SHUFFLED_INTENSITY",
        "AU5_ATTENTION_EXACT_SUPPORT_COMPLEMENT_REGION_NORM",
    )

    ATTENTION_MANIFEST_FIELDS = (
        "manifest_index",
        "image_token",
        "image_path",
        "patient_id",
        "series_id",
        "segmentation_fold",
        "cached_image_path",
        "automatic_mask_path",
        "manual_mask_path",
        "predicted_attention_mask_path",
        "monai_valid",
        "monai_area_ratio",
        "monai_peak_probability",
        "manual_mask_exists",
    )

    ATTENTION_EXPERIMENTS = (
        ExperimentConfig(
            experiment_id="AU1_ATTENTION_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA",
            description=(
                "Attention U-Net hard support with region-only MRI normalization, "
                "fixed-centre fallback, hierarchical patient pooling, PCA and LR."
            ),
            feature_mode=ATTENTION_FEATURE_MODES[0],
            role="attention_candidate",
        ),
        ExperimentConfig(
            experiment_id="AU2_ATTENTION_HARD_SUPPORT_REGION_NORM_VALID_ONLY_HIER_LR_PCA",
            description=(
                "AU1 restricted to Attention U-Net gate-valid slices. If a patient "
                "has zero valid slices, that patient row transparently reuses AU1 "
                "and is listed in feature-bank metadata rather than being deleted."
            ),
            feature_mode=ATTENTION_FEATURE_MODES[1],
            role="attention_ablation",
        ),
        ExperimentConfig(
            experiment_id="AU3_ATTENTION_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
            description="Exact Attention U-Net support geometry with MRI intensity removed.",
            feature_mode=ATTENTION_FEATURE_MODES[2],
            role="segmentation_representation_control",
        ),
        ExperimentConfig(
            experiment_id="AU4_ATTENTION_SUPPORT_SHUFFLED_INTENSITY_HIER_LR_PCA",
            description=(
                "Exact Attention support and visible intensity multiset with the "
                "spatial intensity assignment deterministically destroyed."
            ),
            feature_mode=ATTENTION_FEATURE_MODES[3],
            role="anatomy_destruction_control",
        ),
        ExperimentConfig(
            experiment_id="AU5_ATTENTION_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
            description=(
                "Independently normalized exact non-padding complement of the "
                "Attention U-Net support."
            ),
            feature_mode=ATTENTION_FEATURE_MODES[4],
            role="negative_control",
        ),
    )

class SETTINGS_REVIEW:
    """HTML review queue, workflow actions and CSV field contracts."""

    # diverse gives a patient/series-balanced queue; all exposes every mask.
    REVIEW_SCOPE = "diverse"

    # Use 0 with scope=all for no global queue limit.
    REVIEW_LIMIT = 1200

    REVIEW_SEED = 42

    ATTENTION_REVIEW_SCHEMA = "cad-mask-review-current"

    ATTENTION_REVIEW_SCOPES = (
        "all",
        "diverse",
        "unreviewed",
        "invalid",
        "disagreement",
        "manual",
        "reviewed",
    )

    ATTENTION_ACTION_CHOICES = (
        "build-monai-feature-cache",
        "generate-all-monai-masks",
        "monai-cpu-from-cache",
        "edit-existing-masks",
        "retrain-regenerate-attention",
        "build-attention-feature-cache",
        "evaluate-attention-cpu",
    )

    CPU_CACHE_ONLY_ACTIONS = {
        "monai-cpu-from-cache",
        "evaluate-attention-cpu",
        "edit-existing-masks",
    }

    ATTENTION_REVIEW_MANIFEST_FIELDS = (
        "review_index",
        "image_token",
        "image_path",
        "patient_id",
        "series_id",
        "segmentation_fold",
        "cached_image_path",
        "automatic_mask_path",
        "manual_mask_path",
        "predicted_attention_mask_path",
        "monai_valid",
        "monai_area_ratio",
        "monai_peak_probability",
        "attention_valid",
        "attention_area_ratio",
        "attention_peak_probability",
        "dice_attention_vs_monai",
        "manual_mask_exists",
        "attention_checkpoint_fingerprint",
    )

    ATTENTION_REVIEW_HISTORY_FIELDS = (
        "timestamp_utc",
        "review_round",
        "review_source",
        "action",
        "image_token",
        "patient_id",
        "series_id",
        "queue_position",
        "queue_size",
        "mask_area_ratio",
        "automatic_mask_path",
        "manual_mask_path",
    )


def _resolve_dataset_path():
    kaggle_path = Path("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset")
    if kaggle_path.exists():
        return str(kaggle_path)
    return r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset"


def _resolve_output_root():
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cad_patient_pipeline_outputs")
    try:
        return Path(__file__).resolve().parent / "cad_patient_pipeline_outputs"
    except NameError:
        return Path.cwd() / "cad_patient_pipeline_outputs"


def _build_suite_identity():
    selected = [
        asdict(experiment)
        for experiment in SETTINGS_EXPERIMENTS.EXPERIMENT_REGISTRY
        if experiment.enabled
        and (
            SETTINGS_EXPERIMENTS.EXPERIMENTS_TO_RUN is None
            or experiment.experiment_id
            in set(SETTINGS_EXPERIMENTS.EXPERIMENTS_TO_RUN)
        )
    ]
    return {
        "pipeline_schema": SETTINGS_RUNTIME.PIPELINE_SCHEMA_ID,
        "experiments": selected,
        "reference_experiment_id": SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
        "primary_candidate_experiment_id": SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
        "valid_only_ablation_experiment_id": SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,
        "control_experiment_ids": (
            SETTINGS_EXPERIMENTS.FIXED_PERIPHERY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.MASK_ONLY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.COMPLEMENT_CONTROL_EXPERIMENT_ID,
        ),
        "primary_ablation_comparisons": SETTINGS_EXPERIMENTS.PRIMARY_ABLATION_COMPARISONS,
        "repeated_stability_comparisons": SETTINGS_VALIDATION.REPEATED_STABILITY_COMPARISONS,
        "random_seed": SETTINGS_RUNTIME.RANDOM_SEED,
        "cv_random_state": SETTINGS_VALIDATION.CV_RANDOM_STATE,
        "inner_cv_random_state": SETTINGS_VALIDATION.INNER_CV_RANDOM_STATE,
        "n_splits": SETTINGS_VALIDATION.N_SPLITS,
        "inner_splits": SETTINGS_VALIDATION.INNER_CV_SPLITS,
        "c_grid": SETTINGS_VALIDATION.CLASSIFIER_C_GRID,
        "threshold_method": SETTINGS_VALIDATION.THRESHOLD_SELECTION_METHOD,
        "bootstrap_replicates": SETTINGS_VALIDATION.BOOTSTRAP_REPLICATES,
        "paired_bootstrap_replicates": SETTINGS_VALIDATION.PAIRED_BOOTSTRAP_REPLICATES,
        "pca_variance": SETTINGS_VALIDATION.PATIENT_PCA_EXPLAINED_VARIANCE,
        "feature_cache_schema": SETTINGS_FEATURE_BANK.FEATURE_CACHE_SCHEMA,
        "device_tag": (
            DEVICE
            if SETTINGS_RUNTIME.SUITE_DEVICE_TAG == "auto"
            else SETTINGS_RUNTIME.SUITE_DEVICE_TAG
        ),
        "img_size": SETTINGS_PREPROCESSING.IMG_SIZE,
        "monai_input_size": SETTINGS_MONAI.MONAI_INPUT_SIZE,
        "monai_bundle": SETTINGS_MONAI.MONAI_BUNDLE_NAME,
        "monai_bundle_version": SETTINGS_MONAI.MONAI_BUNDLE_VERSION,
        "monai_hf_revision": SETTINGS_MONAI.MONAI_HF_REVISION,
        "hard_support_dilation_kernel": SETTINGS_PREPROCESSING.HARD_SUPPORT_DILATION_KERNEL,
        "fixed_periphery_exclusion_fraction": SETTINGS_PREPROCESSING.FIXED_PERIPHERY_EXCLUSION_FRACTION,
        "region_norm_lower_percentile": SETTINGS_PREPROCESSING.REGION_NORM_LOWER_PERCENTILE,
        "region_norm_upper_percentile": SETTINGS_PREPROCESSING.REGION_NORM_UPPER_PERCENTILE,
        "region_norm_min_pixels": SETTINGS_PREPROCESSING.REGION_NORM_MIN_PIXELS,
        "region_norm_histogram_bins": SETTINGS_PREPROCESSING.REGION_NORM_HISTOGRAM_BINS,
        "region_norm_min_dynamic_range": SETTINGS_PREPROCESSING.REGION_NORM_MIN_DYNAMIC_RANGE,
        "support_intensity_shuffle_schema": SETTINGS_PREPROCESSING.SUPPORT_INTENSITY_SHUFFLE_SCHEMA,
        "standardization_feature_names": SETTINGS_PREPROCESSING.STANDARDIZATION_FEATURE_NAMES,
        "validation_runtime_profile": SETTINGS_VALIDATION.VALIDATION_RUNTIME_PROFILE,
        "stability_experiment_ids": SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS,
        "permutation_experiment_ids": SETTINGS_VALIDATION.PERMUTATION_EXPERIMENT_IDS,
        "model_family_selection_ids": SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS,
        "selection_adjusted_candidate_ids": SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS,
    }


class SETTINGS_PATHS:
    """Dataset, model, cache and output locations visible in VS Code Outline."""

    DATASET_PATH = _resolve_dataset_path()
    OUTPUT_ROOT = _resolve_output_root()
    SUITE_CONFIGURATION_TAG = hashlib.sha256(
        json.dumps(_build_suite_identity(), sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    SUITE_NAME = f"multi_experiment_suite__{SUITE_CONFIGURATION_TAG}"
    OUTPUT_DIR = OUTPUT_ROOT / SUITE_NAME
    FEATURE_CACHE_ROOT = OUTPUT_ROOT / "feature_bank_cache"
    CONSOLE_LOG_PATH = OUTPUT_DIR / "console_output.log"



class SETTINGS:
    """One navigation root for every settings class."""

    BOOTSTRAP = SETTINGS_BOOTSTRAP
    RUNTIME = SETTINGS_RUNTIME
    EXPERIMENTS = SETTINGS_EXPERIMENTS
    VALIDATION = SETTINGS_VALIDATION
    PREPROCESSING = SETTINGS_PREPROCESSING
    FEATURE_BANK = SETTINGS_FEATURE_BANK
    AUDIT = SETTINGS_AUDIT
    MONAI = SETTINGS_MONAI
    ATTENTION = SETTINGS_ATTENTION
    REVIEW = SETTINGS_REVIEW
    REPORTING = SETTINGS_REPORTING
    PATHS = SETTINGS_PATHS


# endregion

# Runtime state is deliberately separate from immutable settings.

DEVICE = (
    "cuda"
    if SETTINGS_RUNTIME.RUNTIME_DEVICE_POLICY != "cpu"
    and torch.cuda.is_available()
    else "cpu"
)

MONAI_RUNTIME_SOURCE = "not_loaded"

MONAI_RUNTIME_ARTIFACT_PATH = None



# Deterministic initialization uses the class-based seed.

np.random.seed(SETTINGS_RUNTIME.RANDOM_SEED)

torch.manual_seed(SETTINGS_RUNTIME.RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SETTINGS_RUNTIME.RANDOM_SEED)

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = False

if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = False

try:
    torch.set_float32_matmul_precision("highest")
except (AttributeError, RuntimeError):
    pass

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)



# Import-time consistency checks for the most error-prone settings.

if SETTINGS_ATTENTION.ATTENTION_PNG_COMPRESSION > 9:
    raise ValueError("CAD_ATTENTION_PNG_COMPRESSION must be between 0 and 9.")

if SETTINGS_ATTENTION.ATTENTION_SUPPORT_DILATION_KERNEL % 2 == 0:
    raise ValueError("CAD_ATTENTION_UNET_SUPPORT_DILATION_KERNEL must be odd.")

if SETTINGS_ATTENTION.ATTENTION_PSEUDO_MASK_DILATION_KERNEL % 2 == 0:
    raise ValueError("CAD_ATTENTION_UNET_PSEUDO_MASK_DILATION_KERNEL must be odd.")

if (
    SETTINGS_ATTENTION.ATTENTION_MIN_HEART_AREA_RATIO
    >= SETTINGS_ATTENTION.ATTENTION_MAX_HEART_AREA_RATIO
):
    raise ValueError("Attention U-Net area-ratio limits are inconsistent.")

# =============================
# EXECUTION STATUS + TIMING HELPERS
# =============================

SETTINGS_REPORTING.PIPELINE_STAGE_COUNT = 12
# The number matches the suite orchestration stages inside ``main``.













# =============================
# CONFIGURATION VALIDATION
# =============================

# This validation function is intentionally called before dataset scanning,
# model loading, downloading, GPU allocation, or feature extraction. A malformed
# setting should fail immediately, before the pipeline spends time or creates
# cache/output files that could later be mistaken for a valid experiment.




# =============================
# IMAGE PREPROCESSING HELPERS
# =============================



















# =============================
# PIPELINE STEP 1
# MRI SLICE DATASET + PROVENANCE FEATURES
# =============================



# The complete list above is written to the cohort manifest for descriptive
# audit. The provenance-only classifier intentionally uses a narrower subset
# that is dominated by export geometry, file/container properties, padding, and
# border style. Global intensity, center intensity, entropy, edge density, and
# sharpness are excluded from that classifier because they can contain genuine
# anatomical or pathology-related signal and would make the term "provenance
# only" too strong.






class MRIDataset(Dataset):
    """Decode one JPEG and build only the standardized tensors now required.

    The raw uint8 image is decoded once. Hashes and provenance features are
    calculated before resizing. Three aligned image views are then returned:

    * ``standardized_image``: robust-scaled 3x224x224 classifier input;
    * ``standardized_monai``: min-max 1x256x256 segmentation input;
    * ``standardized_raw``: uint8/255 3x224x224 canvas used for region-specific
      intensity normalization after the final support is selected.

    A one-channel padding mask preserves the non-padding content support. Labels
    travel only as metadata and never influence preprocessing or neural inference.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, patient_id, series_id = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"OpenCV could not read MRI image: {image_path}")

        digest = hashlib.sha256()
        digest.update(np.asarray(image.shape, dtype=np.int32).tobytes())
        digest.update(image.tobytes(order="C"))
        decoded_pixel_hash = digest.hexdigest()
        perceptual_hash = compute_dct_perceptual_hash(image)
        provenance_features = compute_image_provenance_features(image, image_path)

        (
            classifier_canvas,
            monai_canvas,
            raw_canvas,
            padding_canvas,
            standardization_features,
        ) = build_label_blind_standardized_image(image)

        classifier_gray = cv2.resize(
            classifier_canvas, (SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        raw_gray = cv2.resize(
            raw_canvas, (SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        padding_gray = cv2.resize(
            padding_canvas, (SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE), interpolation=cv2.INTER_NEAREST
        )

        classifier_hwc = np.stack([classifier_gray] * 3, axis=-1)
        raw_hwc = np.stack([raw_gray] * 3, axis=-1)
        if self.transform is not None:
            standardized_image = self.transform(classifier_hwc)
            standardized_raw = self.transform(raw_hwc)
        else:
            standardized_image = torch.from_numpy(classifier_hwc).permute(2, 0, 1)
            standardized_raw = torch.from_numpy(raw_hwc).permute(2, 0, 1)

        return (
            standardized_image,
            torch.from_numpy(monai_canvas).unsqueeze(0),
            standardized_raw,
            torch.from_numpy(padding_gray).unsqueeze(0),
            int(label),
            str(patient_id),
            str(series_id),
            int(index),
            decoded_pixel_hash,
            perceptual_hash,
            torch.from_numpy(provenance_features),
            torch.from_numpy(standardization_features),
        )


# =============================
# LOAD DATA
# =============================



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



















# =============================
# PIPELINE STEP 3
# MONAI PROBABILITY MAP + SOFT ROI
# =============================







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
            f"{SETTINGS_FEATURE_BANK.EFFICIENTNET_WEIGHTS_NAME}.",
            flush=True,
        )
        print(
            "[MODEL][EfficientNet] The first run may download the ImageNet "
            "checkpoint; later runs should use the torchvision cache.",
            flush=True,
        )

        weights = models.EfficientNet_B0_Weights[
            SETTINGS_FEATURE_BANK.EFFICIENTNET_WEIGHTS_NAME
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



































# =============================
# PIPELINE STEP 7
# PATIENT-LEVEL HIERARCHICAL POOLING
# =============================






# =============================
# PIPELINE STEP 8
# NESTED PATIENT-LEVEL CROSS-VALIDATION
# =============================























# =============================
# PIPELINE STEP 9
# PATIENT-LEVEL METRICS AND CONFIDENCE INTERVALS
# =============================









# =============================
# PIPELINE STEP 10
# REPEATED NESTED CV + PATIENT-LABEL PERMUTATION
# =============================





















# =============================
# PIPELINE STEP 11
# PAIRED COMPARISONS AND MASTER REPORTS
# =============================
















# =============================
# PIPELINE STEP 11
# SUITE ORCHESTRATION, LOGGING AND FINALIZATION
# =============================










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




# ============================================================================
# ATTENTION U-NET, MANUAL REVIEW AND DISK-SAFE KAGGLE CACHE
# ============================================================================
#
# Attention U-Net uses the same patient contract but a separate workspace and comparison directory, so segmentation training cannot overwrite MONAI reference outputs.
# Standardized 256x256 training inputs are now cached in RAM by default on
# Kaggle; optional disk copies are transient and never required. This prevents
# a full feature-bank directory from causing a fatal libpng write error.
#
# Available command-line actions:
#
#   --attention-action both
#       Run the complete MONAI suite, then generate/reuse pseudo masks, train
#       cross-fitted Attention U-Nets and evaluate AU1-AU5.
#
#   --attention-action monai-only
#       Run only the unchanged MONAI suite.
#
#   --attention-action generate-masks
#       Build a deterministic, label-blind review/training manifest and generate
#       automatic MONAI pseudo masks for the selected images.
#
#   --attention-action edit-masks
#       Open an interactive Matplotlib editor. Saved manual masks have priority
#       over pseudo masks during the next Attention U-Net training run.
#
#   --attention-action train-attention
#       Generate missing pseudo masks and train/reuse five cross-fitted Attention
#       U-Net checkpoints. The CAD Normal/Sick label is never supplied to the
#       segmentation loss or checkpoint selection.
#
#   --attention-action attention-only
#       Generate/reuse masks, train/reuse cross-fitted checkpoints, infer masks
#       for all images, extract AU1-AU5 patient embeddings and run classification
#       comparisons. The suite is not rerun.
#
# METHODOLOGICAL LIMITATION:
# Unless sufficient manual/expert masks or an independently trained checkpoint
# are supplied, Attention U-Net is distilled from MONAI pseudo masks. It is then
# a comparison of segmentation backend, regularization and representation—not
# independent expert-ground-truth validation.
# ============================================================================

import argparse
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# ATTENTION U-NET SETTINGS
# ---------------------------------------------------------------------------
# See SETTINGS_ATTENTION in the central VS Code Outline settings region.

@dataclass(frozen=True)
class AttentionWorkspace:
    """Filesystem contract for masks, checkpoints and comparison outputs."""

    root: Path
    transient_root: Path
    automatic_masks: Path
    manual_masks: Path
    predicted_masks: Path
    mask_overlays: Path
    cached_images: Path
    checkpoints: Path
    manifest_csv: Path
    training_summary_json: Path
    console_log: Path
    comparison_output: Path




# ---------------------------------------------------------------------------
# LABEL-BLIND TRAINING/REVIEW MANIFEST
# ---------------------------------------------------------------------------

















# In-process cache for standardized 256x256 uint8 segmentation inputs. The full
# 4,715-image training subset occupies roughly 295 MiB before dictionary
# overhead, so the Kaggle default of 384 MiB can normally retain the complete
# subset and avoid repeated preprocessing during cross-fitted training.
_ATTENTION_IMAGE_RAM_CACHE = OrderedDict()
_ATTENTION_IMAGE_RAM_CACHE_BYTES = 0
_ATTENTION_OPTIONAL_DISK_CACHE_DISABLED_REASON = None
_ATTENTION_STORAGE_WARNING_KEYS = set()






















# ---------------------------------------------------------------------------
# AUTOMATIC MONAI PSEUDO-MASK GENERATION
# ---------------------------------------------------------------------------

class _AttentionManifestImageDataset(Dataset):
    """Dataset used only to generate automatic pseudo masks."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_attention_segmentation_image(row)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, int(index)








# ---------------------------------------------------------------------------
# ATTENTION U-NET ARCHITECTURE
# ---------------------------------------------------------------------------




class AttentionConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and SiLU activations."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate applied to one U-Net skip connection."""

    def __init__(self, gating_channels, skip_channels, inter_channels):
        super().__init__()
        groups = _group_count(inter_channels)
        self.gating_projection = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(groups, inter_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(groups, inter_channels),
        )
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, gating, skip):
        gating = F.interpolate(
            gating,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        weights = self.psi(
            self.gating_projection(gating) + self.skip_projection(skip)
        )
        return skip * weights


class AttentionUpBlock(nn.Module):
    """Upsample, gate the skip tensor, concatenate and refine."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up_projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gate = AttentionGate(
            gating_channels=out_channels,
            skip_channels=skip_channels,
            inter_channels=max(1, out_channels // 2),
        )
        self.refine = AttentionConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_projection(x)
        gated_skip = self.gate(x, skip)
        return self.refine(torch.cat([x, gated_skip], dim=1))


class AttentionUNet(nn.Module):
    """Binary four-level Attention U-Net for 256x256 cardiac support masks."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=None):
        super().__init__()
        base = int(base_channels or SETTINGS_ATTENTION.ATTENTION_BASE_CHANNELS)
        self.encoder1 = AttentionConvBlock(in_channels, base)
        self.encoder2 = AttentionConvBlock(base, base * 2)
        self.encoder3 = AttentionConvBlock(base * 2, base * 4)
        self.encoder4 = AttentionConvBlock(base * 4, base * 8)
        self.bottleneck = AttentionConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.decoder4 = AttentionUpBlock(base * 16, base * 8, base * 8)
        self.decoder3 = AttentionUpBlock(base * 8, base * 4, base * 4)
        self.decoder2 = AttentionUpBlock(base * 4, base * 2, base * 2)
        self.decoder1 = AttentionUpBlock(base * 2, base, base)
        self.output = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(bottleneck, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return self.output(d1)






# ---------------------------------------------------------------------------
# CROSS-FITTED ATTENTION U-NET TRAINING
# ---------------------------------------------------------------------------




class AttentionMaskTrainingDataset(Dataset):
    """RAM/disk/source images and manual/pseudo masks for one partition."""

    def __init__(self, rows, augment=False, seed=0):
        self.rows = list(rows)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_attention_segmentation_image(row)
        mask_path, source = _resolved_training_mask(row)
        if mask_path is None:
            raise RuntimeError(
                f"No manual or valid pseudo mask for {row['image_token']}."
            )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(
                f"Could not load training pair for {row['image_token']}: "
                f"mask={mask_path}"
            )
        image = cv2.resize(
            image,
            (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        mask = cv2.resize(
            mask,
            (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        ) > 127

        if self.augment:
            rng = np.random.default_rng(
                self.seed + self.epoch * 1_000_003 + int(index)
            )
            if rng.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            if rng.random() < 0.10:
                image = np.flipud(image)
                mask = np.flipud(mask)
            if rng.random() < 0.35:
                k = int(rng.integers(0, 4))
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
            contrast = float(rng.uniform(0.90, 1.10))
            brightness = float(rng.uniform(-0.05, 0.05))
            image = np.clip(image * contrast + brightness, 0.0, 1.0)

        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask.astype(np.float32))
        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
            row["image_token"],
            source,
        )












# ---------------------------------------------------------------------------
# FULL-COHORT ATTENTION INFERENCE AND PATIENT FEATURE POOLING
# ---------------------------------------------------------------------------

class AttentionInferenceDataset(Dataset):
    """Aligned Attention U-Net and EfficientNet inputs for full-cohort inference."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, patient_id, series_id = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not decode MRI image: {image_path}")
        decoded_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        monai_canvas, raw_224, content_224 = _load_attention_canvases(image_path)
        raw_3 = np.stack([raw_224] * 3, axis=0).astype(np.float32)
        return (
            torch.from_numpy(monai_canvas).unsqueeze(0),
            torch.from_numpy(raw_3),
            torch.from_numpy(content_224).unsqueeze(0),
            int(label),
            str(patient_id),
            str(series_id),
            int(index),
            decoded_hash,
            attention_image_token(image_path, patient_id, series_id),
        )








class _StreamingHierarchicalAttentionPool:
    """Stream slice embeddings into series means and equal patient-series means."""

    def __init__(self, modes):
        self.modes = tuple(modes)
        self.series_sums = {mode: {} for mode in self.modes}
        self.series_counts = {mode: defaultdict(int) for mode in self.modes}
        self.series_to_patient = {}
        self.patient_to_label = {}
        self.retained_slice_counts = {mode: 0 for mode in self.modes}
        self.patient_level_fallbacks = {mode: [] for mode in self.modes}

    def add(self, mode, embeddings, patient_ids, series_ids, labels, keep=None):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if keep is None:
            keep = np.ones(len(embeddings), dtype=bool)
        keep = np.asarray(keep, dtype=bool)
        for index in np.flatnonzero(keep):
            patient_id = str(patient_ids[index])
            series_id = str(series_ids[index])
            label = int(labels[index])
            previous = self.patient_to_label.get(patient_id)
            if previous is not None and previous != label:
                raise RuntimeError(f"Inconsistent label for {patient_id}.")
            self.patient_to_label[patient_id] = label
            self.series_to_patient[series_id] = patient_id
            if series_id not in self.series_sums[mode]:
                self.series_sums[mode][series_id] = np.zeros(
                    embeddings.shape[1], dtype=np.float64
                )
            self.series_sums[mode][series_id] += embeddings[index]
            self.series_counts[mode][series_id] += 1
            self.retained_slice_counts[mode] += 1

    def finalize(self, mode):
        patient_series = defaultdict(list)
        for series_id in sorted(self.series_sums[mode]):
            count = self.series_counts[mode][series_id]
            if count <= 0:
                continue
            series_mean = self.series_sums[mode][series_id] / float(count)
            patient_series[self.series_to_patient[series_id]].append(series_mean)
        missing = sorted(set(self.patient_to_label) - set(patient_series))
        if missing and mode == SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[1]:
            # A very poor or deliberately conservative segmenter can mark every
            # slice of one patient invalid. Patient-level CV requires one row per
            # Directory_*. For those rare patients only, AU2 transparently falls
            # back to that patient's AU1 pooled representation rather than
            # deleting the patient or aborting the entire suite. The affected IDs
            # are written to feature-bank metadata and must be reported.
            fallback_mode = SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[0]
            fallback_series = defaultdict(list)
            for series_id in sorted(self.series_sums[fallback_mode]):
                patient_id = self.series_to_patient[series_id]
                if patient_id not in missing:
                    continue
                count = self.series_counts[fallback_mode][series_id]
                if count > 0:
                    fallback_series[patient_id].append(
                        self.series_sums[fallback_mode][series_id] / float(count)
                    )
            for patient_id in missing:
                if not fallback_series.get(patient_id):
                    raise RuntimeError(
                        f"{mode}: no valid-only or AU1 fallback representation "
                        f"for patient {patient_id}."
                    )
                patient_series[patient_id] = fallback_series[patient_id]
            self.patient_level_fallbacks[mode] = missing
            print(
                f"[ATTENTION][AU2] No gate-valid slices for {len(missing)} "
                "patients; their AU2 patient rows transparently reuse AU1. "
                f"Patients={missing}",
                flush=True,
            )
        elif missing:
            raise RuntimeError(
                f"{mode}: no retained series for patients {missing}."
            )
        patients = np.asarray(sorted(patient_series))
        X = np.stack(
            [
                np.mean(np.stack(patient_series[patient], axis=0), axis=0)
                for patient in patients
            ],
            axis=0,
        ).astype(np.float32)
        y = np.asarray(
            [self.patient_to_label[patient] for patient in patients],
            dtype=np.int64,
        )
        return X, y, patients










# ---------------------------------------------------------------------------
# PATIENT-LEVEL AU1-AU5 EVALUATION
# ---------------------------------------------------------------------------













# ---------------------------------------------------------------------------
# ATTENTION EXTENSION SELF-TEST AND ORCHESTRATION
# ---------------------------------------------------------------------------




# ============================================================================
# FULL-COHORT MONAI / ATTENTION U-NET MASK REVIEW
# ============================================================================
#
# This section adds a full-cohort review workflow:
#
#   * every one of the 63,425 image rows can appear in the HTML editor;
#   * MONAI masks can be generated for all rows or only for a selected queue;
#   * cross-fitted Attention U-Net masks can be generated for all rows or only
#     for a selected queue;
#   * review queues can be all, diverse, unreviewed, invalid, disagreement,
#     manually corrected, or previously reviewed;
#   * manual masks saved anywhere in the full cohort are automatically added to
#     the next Attention U-Net training manifest, even when they lie outside the
#     ordinary 5-per-series / 40-per-patient pseudo-label subset;
#   * each manual-review action is logged by source and review round;
#   * after manual corrections, ``retrain-regenerate-attention`` retrains stale
#     folds and regenerates Attention U-Net masks, after which another review
#     round can be opened.
#
# MONAI and Attention predictions are regenerable and remain in the transient
# workspace by default. Manual masks and review history remain persistent under
# /kaggle/working/cad_attention_unet_workspace.

from collections import defaultdict as _review_defaultdict
from datetime import datetime as _review_datetime, timezone as _review_timezone
















































# ---------------------------------------------------------------------------
# TRAINING MANIFEST PATCH: retain compact pseudo subset + every manual mask.
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# HTML EDITOR PATCH: strict source, review log and rapid-review buttons.
# ---------------------------------------------------------------------------

class BaseAttentionMaskEditor:
    """
    Kaggle-safe manual mask editor.

    This implementation deliberately does NOT use ipympl/jupyter-matplotlib.
    It uses standard ipywidgets (which Kaggle/JupyterLab supports) plus an
    HTML5 canvas rendered in an Output widget. The canvas handles mouse
    drawing/erasing in the browser and synchronizes the mask to a standard
    Textarea widget, so no custom Jupyter widget model is required.

    Public controls:
      Previous / Next / Save manual / Reset to auto / Reset to image (no mask) / Clear
      brush slider, left-draw/right-erase, D/E keyboard modes, S/R/C/N/P.

    Button semantics:
      - Reset to auto: restore automatic mask and orange comparison overlay.
      - Reset to image: show the raw MRI with no mask overlay at all.
      - Clear: clear only the editable mask while keeping auto mask in orange.
    """

    def __init__(
        self,
        rows,
        workspace,
        start_index=0,
        brush_radius=8,
        base_source="attention",
    ):
        if not rows:
            raise ValueError("The mask editor requires at least one manifest row.")
        if base_source not in {"attention", "monai"}:
            raise ValueError("base_source must be 'attention' or 'monai'.")

        import base64
        import uuid
        import ipywidgets as widgets
        from IPython.display import display, HTML, clear_output

        self.rows = list(rows)
        self.workspace = workspace
        self.index = int(np.clip(start_index, 0, len(rows) - 1))
        self.brush_radius = int(max(1, brush_radius))
        self.base_source = base_source
        self.mode = "draw"
        self.image = None
        # ``auto_mask`` is the immutable automatic mask loaded from MONAI or
        # Attention U-Net. ``base_mask`` is only the orange comparison overlay
        # currently shown by the browser canvas and can temporarily be hidden.
        self.auto_mask = None
        self.base_mask = None
        self.mask = None

        self._base64 = base64
        self._uuid = uuid.uuid4().hex[:12]
        self._clear_output = clear_output

        self.output = widgets.Output()
        self.mask_sync = widgets.Textarea(
            value="",
            placeholder=f"CAD_MASK_SYNC_{self._uuid}",
            layout=widgets.Layout(width="1px", height="1px", display="none"),
        )

        self.previous_button = widgets.Button(description="Previous")
        self.next_button = widgets.Button(description="Next")
        self.save_button = widgets.Button(description="Save manual")
        self.reset_button = widgets.Button(description="Reset to auto")
        self.clear_drawn_button = widgets.Button(description="Reset to image (no mask)")
        self.clear_button = widgets.Button(description="Clear")
        self.brush_slider = widgets.IntSlider(
            description="Brush",
            value=self.brush_radius,
            min=1,
            max=30,
            step=1,
            continuous_update=True,
        )
        self.status = widgets.HTML()

        self.previous_button.on_click(self._previous_click)
        self.next_button.on_click(self._next_click)
        self.save_button.on_click(self._save_click)
        self.reset_button.on_click(self._reset_click)
        self.clear_drawn_button.on_click(self._clear_drawn_click)
        self.clear_button.on_click(self._clear_click)
        self.brush_slider.observe(self._brush_changed, names="value")
        self.mask_sync.observe(self._mask_sync_changed, names="value")

        self.controls = widgets.VBox([
            widgets.HBox([
                self.previous_button,
                self.next_button,
                self.save_button,
                self.reset_button,
                self.clear_drawn_button,
                self.clear_button,
            ]),
            widgets.HBox([self.brush_slider, self.status]),
            self.mask_sync,
            self.output,
        ])

        self.load_current()

    def _brush_changed(self, change):
        self.brush_radius = int(change["new"])
        if hasattr(self, "output") and hasattr(self, "mask") and self.mask is not None:
            self._render()

    def _automatic_mask_path(self, row):
        attention_path = Path(row["predicted_attention_mask_path"])
        monai_path = Path(row["automatic_mask_path"])
        if self.base_source == "attention" and attention_path.is_file():
            return attention_path
        return monai_path

    def load_current(self):
        row = self.rows[self.index]
        image = load_attention_segmentation_image(row)
        automatic_path = self._automatic_mask_path(row)
        if not automatic_path.is_file():
            raise FileNotFoundError(
                f"Automatic mask unavailable. Run generate-masks first: {automatic_path}"
            )

        automatic = cv2.imread(str(automatic_path), cv2.IMREAD_GRAYSCALE)
        if automatic is None:
            raise RuntimeError(f"Could not load automatic mask: {automatic_path}")
        automatic = cv2.resize(
            automatic,
            (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        manual_path = Path(row["manual_mask_path"])
        if manual_path.is_file():
            manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
            if manual is None:
                raise RuntimeError(f"Could not load manual mask: {manual_path}")
            manual = cv2.resize(
                manual,
                (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            current = manual > 127
        else:
            current = automatic > 127

        self.image = image.astype(np.float32) / 255.0
        self.auto_mask = (automatic > 127).astype(np.uint8)
        self.base_mask = self.auto_mask.copy()
        self.mask = current.astype(np.uint8)
        self._render()

    def _png_data_uri(self, rgb_float):
        arr = np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode editor image.")
        return "data:image/png;base64," + self._base64.b64encode(buf.tobytes()).decode("ascii")

    def _mask_data_uri(self, mask):
        """Encode a binary mask as a transparent PNG for the HTML canvas.

        A plain grayscale PNG is opaque even where its value is zero. The old
        canvas code recolored images through ``source-in``, which uses alpha,
        not grayscale intensity. Consequently, a zero-valued mask still had an
        opaque alpha channel over the entire 256x256 image and Reset/Clear
        appeared to create a full-frame mask.

        This encoder makes background pixels transparent black and foreground
        pixels opaque white. Therefore the orange/magenta overlays are limited
        to true mask pixels, and an all-zero mask produces no overlay at all.
        """

        binary = (np.asarray(mask) > 0).astype(np.uint8)
        bgra = np.zeros((*binary.shape, 4), dtype=np.uint8)
        foreground = binary * 255
        bgra[..., 0] = foreground
        bgra[..., 1] = foreground
        bgra[..., 2] = foreground
        bgra[..., 3] = foreground
        ok, buf = cv2.imencode(".png", bgra)
        if not ok:
            raise RuntimeError("Could not encode transparent editor mask.")
        return (
            "data:image/png;base64,"
            + self._base64.b64encode(buf.tobytes()).decode("ascii")
        )

    def _render(self):
        """Render a Kaggle-safe interactive HTML5 canvas.

        IMPORTANT: JavaScript placed inside HTML output is not reliably executed
        by Kaggle/JupyterLab. Therefore the canvas markup is emitted as HTML and
        the event handlers are injected separately with IPython.display.Javascript.
        This avoids both jupyter-matplotlib/ipympl and inert <script> tags.
        """
        row = self.rows[self.index]
        image_uri = self._png_data_uri(np.stack([self.image] * 3, axis=-1))
        base_uri = self._mask_data_uri(self.base_mask)
        mask_uri = self._mask_data_uri(self.mask)
        sync_placeholder = f"CAD_MASK_SYNC_{self._uuid}"
        canvas_id = f"cad_canvas_{self._uuid}"

        html = f"""
        <div id="cad_editor_wrap_{self._uuid}" style="font-family:Arial,sans-serif;max-width:900px">
          <div style="margin-bottom:6px;font-size:14px">
            <b>{self.index + 1}/{len(self.rows)}</b>
            &nbsp;|&nbsp; {row['patient_id']}
            &nbsp;|&nbsp; {row['series_id']}
            &nbsp;|&nbsp; base={self.base_source}
          </div>
          <canvas id="{canvas_id}" width="{SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE*3}"
                  height="{SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE*3}"
                  style="width:768px;height:768px;max-width:100%;border:1px solid #999;
                         cursor:crosshair;image-rendering:auto;touch-action:none;
                         user-select:none;-webkit-user-select:none;"></canvas>
          <div style="font-size:12px;margin-top:5px">
            <b>Left drag = draw</b> &nbsp;|&nbsp; <b>Right drag = erase</b> &nbsp;|&nbsp;
            D/E = mode &nbsp;|&nbsp; S = save &nbsp;|&nbsp; R = reset &nbsp;|&nbsp;
            C = clear &nbsp;|&nbsp; N/P = navigate
          </div>
        </div>
        """

        js = f"""
        (() => {{
          const canvas = document.getElementById({json.dumps(canvas_id)});
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          const W = {SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE};
          const H = {SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE};
          const SCALE = 3;
          const initialRadius = {int(self.brush_radius)};
          const syncPlaceholder = {json.dumps(sync_placeholder)};
          const image = new Image();
          const base = new Image();
          const maskCanvas = document.createElement('canvas');
          maskCanvas.width = W; maskCanvas.height = H;
          const maskCtx = maskCanvas.getContext('2d', {{willReadFrequently:true}});
          const initialMask = new Image();
          const imageUri = {json.dumps(image_uri)};
          const baseUri = {json.dumps(base_uri)};
          const maskUri = {json.dumps(mask_uri)};

          let drawing = false;
          let erase = false;
          let mode = {json.dumps(self.mode)};
          let lastPoint = null;

          function findHidden() {{
            return Array.from(document.querySelectorAll('textarea'))
              .find(x => x.placeholder === syncPlaceholder);
          }}

          function initMask() {{
            maskCtx.clearRect(0, 0, W, H);
            maskCtx.drawImage(initialMask, 0, 0, W, H);
            drawScene();
          }}

          function drawScene() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

            const tmp = document.createElement('canvas');
            tmp.width = canvas.width; tmp.height = canvas.height;
            const tc = tmp.getContext('2d');
            tc.globalAlpha = 0.35;
            tc.drawImage(base, 0, 0, canvas.width, canvas.height);
            tc.globalCompositeOperation = 'source-in';
            tc.fillStyle = 'rgb(255,165,0)';
            tc.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(tmp, 0, 0);

            const tmp2 = document.createElement('canvas');
            tmp2.width = canvas.width; tmp2.height = canvas.height;
            const t2 = tmp2.getContext('2d');
            t2.globalAlpha = 0.50;
            t2.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
            t2.globalCompositeOperation = 'source-in';
            t2.fillStyle = 'rgb(255,0,200)';
            t2.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(tmp2, 0, 0);
          }}

          function syncToPython() {{
            const hidden = findHidden();
            if (!hidden) return;
            hidden.value = maskCanvas.toDataURL('image/png').split(',')[1];
            hidden.dispatchEvent(new Event('input', {{bubbles:true}}));
            hidden.dispatchEvent(new Event('change', {{bubbles:true}}));
          }}

          function point(e) {{
            const r = canvas.getBoundingClientRect();
            return {{
              x: Math.max(0, Math.min(W - 1, ((e.clientX - r.left) / r.width) * W)),
              y: Math.max(0, Math.min(H - 1, ((e.clientY - r.top) / r.height) * H))
            }};
          }}

          function paintAt(p, doErase) {{
            const radius = initialRadius;
            maskCtx.save();
            maskCtx.globalCompositeOperation = doErase ? 'destination-out' : 'source-over';
            maskCtx.fillStyle = 'white';
            maskCtx.strokeStyle = 'white';
            maskCtx.lineWidth = radius * 2;
            maskCtx.lineCap = 'round';
            maskCtx.lineJoin = 'round';
            if (lastPoint) {{
              maskCtx.beginPath();
              maskCtx.moveTo(lastPoint.x, lastPoint.y);
              maskCtx.lineTo(p.x, p.y);
              maskCtx.stroke();
            }} else {{
              maskCtx.beginPath();
              maskCtx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
              maskCtx.fill();
            }}
            maskCtx.restore();
            lastPoint = p;
            drawScene();
          }}


          // Explicit mouse/pointer handling.  Kaggle/JupyterLab can treat
          // ordinary left-button drags specially unless the canvas claims the
          // pointer immediately.  We therefore handle BOTH Pointer Events and
          // Mouse Events and use the actual button state.
          function beginDraw(e) {{
            if (e.button !== undefined && e.button !== 0 && e.button !== 2) return;
            e.preventDefault();
            e.stopPropagation();
            drawing = true;
            erase = (e.button === 2) || (mode === 'erase');
            lastPoint = null;
            try {{ canvas.setPointerCapture?.(e.pointerId); }} catch (_) {{}}
            paintAt(point(e), erase);
          }}

          function moveDraw(e) {{
            if (!drawing) return;
            // With Pointer Events, buttons is a bit mask: 1=left, 2=right.
            if (e.buttons !== undefined && e.buttons === 0) {{
              finishDraw(e);
              return;
            }}
            e.preventDefault();
            e.stopPropagation();
            paintAt(point(e), erase);
          }}

          function finishDraw(e) {{
            if (!drawing) return;
            e.preventDefault?.();
            e.stopPropagation?.();
            drawing = false;
            lastPoint = null;
            try {{ canvas.releasePointerCapture?.(e.pointerId); }} catch (_) {{}}
            syncToPython();
          }}

          canvas.addEventListener('contextmenu', e => {{
            e.preventDefault();
            e.stopPropagation();
          }}, true);

          // Pointer Events: primary path for modern Chrome/Kaggle.
          canvas.addEventListener('pointerdown', beginDraw, true);
          canvas.addEventListener('pointermove', moveDraw, true);
          canvas.addEventListener('pointerup', finishDraw, true);
          canvas.addEventListener('pointercancel', finishDraw, true);

          // Mouse fallback: this also makes left-button drawing work if the
          // browser/Jupyter environment does not deliver pointer events.
          canvas.addEventListener('mousedown', e => {{
            if (e.button === 0 || e.button === 2) beginDraw(e);
          }}, true);
          canvas.addEventListener('mousemove', moveDraw, true);
          canvas.addEventListener('mouseup', finishDraw, true);

          canvas.addEventListener('mouseleave', e => {{
            if (drawing && e.buttons === 0) finishDraw(e);
          }}, true);

          // Prevent browser text selection / drag behavior over the canvas.
          canvas.addEventListener('dragstart', e => e.preventDefault(), true);
          canvas.style.userSelect = 'none';
          canvas.style.webkitUserSelect = 'none';

          window.addEventListener('keydown', e => {{
            const k = (e.key || '').toLowerCase();
            if (['d','e','s','r','c','n','p'].includes(k)) e.preventDefault();
            if (k === 'd') mode = 'draw';
            else if (k === 'e') mode = 'erase';
            else if (k === 's') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Save manual')?.click();
            else if (k === 'r') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Reset to auto')?.click();
            else if (k === 'c') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Clear')?.click();
            else if (k === 'n') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Next')?.click();
            else if (k === 'p') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Previous')?.click();
          }});

          function loadDataImage(target, source, label) {{
            return new Promise((resolve, reject) => {{
              target.onload = resolve;
              target.onerror = () => reject(
                new Error(`Could not load ${{label}} data URI`)
              );
              target.src = source;
            }});
          }}

          Promise.all([
            loadDataImage(image, imageUri, 'MRI'),
            loadDataImage(base, baseUri, 'automatic mask'),
            loadDataImage(initialMask, maskUri, 'editable mask')
          ]).then(initMask).catch(error => {{
            console.error('[ATTENTION][EDITOR] Canvas initialization failed:', error);
          }});
        }})();
        """

        with self.output:
            self._clear_output(wait=True)
            display(HTML(html))
            display(Javascript(js))

        self.status.value = (
            f"<span style='margin-left:12px'>"
            f"<b>Mode:</b> {self.mode} &nbsp; "
            f"<b>Index:</b> {self.index + 1}/{len(self.rows)}</span>"
        )

    def _sync_mask_to_frontend(self):
        """Synchronize the current Python mask with the hidden HTML widget.

        The same transparent-PNG contract used by the visible canvas is used
        here. This prevents a zero mask from acquiring an opaque full-frame
        background during a Reset/Clear round trip.
        """

        if self.mask is None:
            return
        value = self._mask_data_uri(self.mask).split(",", 1)[1]
        if self.mask_sync.value != value:
            self.mask_sync.value = value

    def _mask_sync_changed(self, change):
        value = change.get("new", "")
        if not value:
            return
        try:
            raw = self._base64.b64decode(value)
            decoded = cv2.imdecode(
                np.frombuffer(raw, np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if decoded is None:
                return

            # Browser canvases export transparent pixels. Use the alpha channel
            # explicitly when available, so transparent RGB values can never be
            # interpreted as foreground over the whole image.
            if decoded.ndim == 3 and decoded.shape[2] == 4:
                alpha = decoded[..., 3]
                gray = cv2.cvtColor(decoded[..., :3], cv2.COLOR_BGR2GRAY)
                binary = ((alpha > 8) & (gray > 127)).astype(np.uint8)
            elif decoded.ndim == 3:
                gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                binary = (gray > 127).astype(np.uint8)
            else:
                binary = (decoded > 127).astype(np.uint8)

            binary = cv2.resize(
                binary,
                (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            self.mask = (binary > 0).astype(np.uint8)
        except Exception as exc:
            print(f"[ATTENTION][EDITOR][WARNING] Mask sync failed: {exc}", flush=True)

    def _previous_click(self, _button):
        self.previous()

    def _next_click(self, _button):
        self.next()

    def _save_click(self, _button):
        self.save()

    def _button_error(self, action, error):
        """Expose callback failures instead of letting ipywidgets hide them."""

        message = f"{action} failed: {type(error).__name__}: {error}"
        self.status.value = (
            "<span style='margin-left:12px;color:#b00020'><b>"
            + message
            + "</b></span>"
        )
        print(f"[ATTENTION][EDITOR][ERROR] {message}", flush=True)

    def _reset_click(self, _button):
        try:
            self.reset()
        except Exception as error:
            self._button_error("Reset to auto", error)

    def _clear_drawn_click(self, _button):
        try:
            self.reset_to_image()
        except Exception as error:
            self._button_error("Reset to image", error)

    def _clear_click(self, _button):
        try:
            self.clear()
        except Exception as error:
            self._button_error("Clear", error)

    def _paint(self, event, erase=False):
        # Retained for API compatibility; browser canvas handles painting.
        return

    def _on_press(self, event):
        return

    def _on_release(self, event):
        return

    def _on_motion(self, event):
        return

    def _on_key(self, event):
        key = str(getattr(event, "key", "") or "").lower()
        if key == "d":
            self.mode = "draw"
        elif key == "e":
            self.mode = "erase"
        elif key == "s":
            self.save()
        elif key == "r":
            self.reset()
        elif key == "c":
            self.clear()
        elif key in {"n", "right"}:
            self.next()
        elif key in {"p", "left"}:
            self.previous()
        self.status.value = (
            f"<span style='margin-left:12px'><b>Mode:</b> {self.mode}</span>"
        )

    def save(self):
        row = self.rows[self.index]
        path = Path(row["manual_mask_path"])
        _atomic_cv2_write(
            path,
            self.mask.astype(np.uint8) * 255,
            required=True,
            purpose="manual Attention U-Net mask",
        )
        overlay_path = self.workspace.mask_overlays / f"{row['image_token']}.png"
        base = np.stack([self.image] * 3, axis=-1)
        overlay = base.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], self.mask * 0.90)
        overlay[..., 1] *= (1.0 - 0.45 * self.mask)
        overlay[..., 2] *= (1.0 - 0.45 * self.mask)
        _atomic_cv2_write(
            overlay_path,
            cv2.cvtColor(
                np.clip(np.round(overlay * 255.0), 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
            required=False,
            purpose="optional manual-mask review overlay",
        )
        row["manual_mask_exists"] = 1
        write_attention_manifest(self.rows, self.workspace.manifest_csv)
        print(f"[ATTENTION][EDITOR] Saved manual mask: {path}", flush=True)

    def reset(self):
        """Restore the immutable automatic mask and its orange reference."""

        if self.auto_mask is None:
            raise RuntimeError("Automatic mask is not loaded.")
        self.base_mask = self.auto_mask.copy()
        self.mask = self.auto_mask.copy()
        self._sync_mask_to_frontend()
        self._render()
        self.status.value = (
            "<span style='margin-left:12px'><b>Reset:</b> automatic mask "
            "restored.</span>"
        )

    def reset_to_image(self):
        """Show only the raw MRI, hiding automatic and editable overlays."""

        if self.auto_mask is None:
            raise RuntimeError("Automatic mask is not loaded.")
        self.base_mask = np.zeros_like(self.auto_mask, dtype=np.uint8)
        self.mask = np.zeros_like(self.auto_mask, dtype=np.uint8)
        self._sync_mask_to_frontend()
        self._render()
        self.status.value = (
            "<span style='margin-left:12px'><b>Raw MRI:</b> all mask "
            "overlays hidden. This is not saved until Save manual is pressed."
            "</span>"
        )

    def clear(self):
        """Clear the editable mask while retaining auto mask as orange guide."""

        if self.auto_mask is None:
            raise RuntimeError("Automatic mask is not loaded.")
        self.base_mask = self.auto_mask.copy()
        self.mask = np.zeros_like(self.auto_mask, dtype=np.uint8)
        self._sync_mask_to_frontend()
        self._render()
        self.status.value = (
            "<span style='margin-left:12px'><b>Editable mask cleared.</b> "
            "The automatic mask remains visible in orange as a guide.</span>"
        )

    def next(self):
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.load_current()

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def show(self):
        from IPython.display import display
        display(self.controls)
        return self


class AttentionMaskEditor(BaseAttentionMaskEditor):
    """Full-cohort iterative editor with persistent review tracking."""

    def __init__(
        self,
        rows,
        workspace,
        start_index=0,
        brush_radius=8,
        base_source="attention",
        review_round=1,
        review_scope="all",
    ):
        self.review_round = int(review_round)
        self.review_scope = str(review_scope)
        super().__init__(
            rows,
            workspace,
            start_index=start_index,
            brush_radius=brush_radius,
            base_source=base_source,
        )
        import ipywidgets as widgets

        self.save_next_button = widgets.Button(
            description="Save & Next", button_style="success"
        )
        self.accept_button = widgets.Button(
            description="Accept auto & Next", button_style="info"
        )
        self.skip_button = widgets.Button(description="Skip & Next")
        self.delete_manual_button = widgets.Button(
            description="Delete manual", button_style="warning"
        )
        self.save_next_button.on_click(self._save_next_click)
        self.accept_button.on_click(self._accept_click)
        self.skip_button.on_click(self._skip_click)
        self.delete_manual_button.on_click(self._delete_manual_click)

        self.controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.previous_button,
                        self.next_button,
                        self.save_button,
                        self.save_next_button,
                        self.accept_button,
                        self.skip_button,
                    ]
                ),
                widgets.HBox(
                    [
                        self.reset_button,
                        self.clear_drawn_button,
                        self.clear_button,
                        self.delete_manual_button,
                    ]
                ),
                widgets.HBox([self.brush_slider, self.status]),
                self.mask_sync,
                self.output,
            ]
        )
        self._update_review_status()

    def _automatic_mask_path(self, row):
        field = (
            "predicted_attention_mask_path"
            if self.base_source == "attention"
            else "automatic_mask_path"
        )
        path = Path(row[field])
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.base_source} mask unavailable for {row['image_token']}: "
                f"{path}. Generate review masks for this source first."
            )
        return path

    def load_current(self):
        row = self.rows[self.index]
        image = load_attention_segmentation_image(row)
        automatic_path = self._automatic_mask_path(row)
        automatic = cv2.imread(str(automatic_path), cv2.IMREAD_GRAYSCALE)
        if automatic is None:
            raise RuntimeError(f"Could not load automatic mask: {automatic_path}")
        automatic = cv2.resize(
            automatic,
            (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        manual_path = Path(row["manual_mask_path"])
        if manual_path.is_file():
            manual = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
            if manual is None:
                raise RuntimeError(f"Could not load manual mask: {manual_path}")
            manual = cv2.resize(
                manual,
                (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            current = manual > 127
        else:
            current = automatic > 127
        self.image = image.astype(np.float32) / 255.0
        self.auto_mask = (automatic > 127).astype(np.uint8)
        self.base_mask = self.auto_mask.copy()
        self.mask = current.astype(np.uint8)
        self._render()
        if hasattr(self, "status"):
            self._update_review_status()

    def _update_review_status(self, prefix=""):
        row = self.rows[self.index]
        manual = Path(row["manual_mask_path"]).is_file()
        if self.base_source == "attention":
            valid = row.get("attention_valid", "")
            peak = row.get("attention_peak_probability", "")
            agreement = row.get("dice_attention_vs_monai", "")
            qc = f"valid={valid}, peak={peak}, Dice-vs-MONAI={agreement}"
        else:
            valid = row.get("monai_valid", "")
            peak = row.get("monai_peak_probability", "")
            qc = f"valid={valid}, peak={peak}"
        self.status.value = (
            f"<span style='margin-left:12px'><b>{prefix}</b> "
            f"source={self.base_source}; scope={self.review_scope}; "
            f"round={self.review_round}; index={self.index + 1}/{len(self.rows)}; "
            f"manual={'yes' if manual else 'no'}; {qc}</span>"
        )

    def _log(self, action, mask=None):
        _review_append_review_event(
            self.workspace,
            self.rows[self.index],
            self.base_source,
            self.review_round,
            action,
            mask=mask,
            queue_position=self.index,
            queue_size=len(self.rows),
        )

    def save(self):
        row = self.rows[self.index]
        path = Path(row["manual_mask_path"])
        _atomic_cv2_write(
            path,
            self.mask.astype(np.uint8) * 255,
            required=True,
            purpose="manual Attention U-Net mask",
        )
        overlay_path = Path(self.workspace.mask_overlays) / (
            f"{row['image_token']}.png"
        )
        base = np.stack([self.image] * 3, axis=-1)
        overlay = base.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], self.mask * 0.90)
        overlay[..., 1] *= 1.0 - 0.45 * self.mask
        overlay[..., 2] *= 1.0 - 0.45 * self.mask
        _atomic_cv2_write(
            overlay_path,
            cv2.cvtColor(
                np.clip(np.round(overlay * 255.0), 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
            required=False,
            purpose="optional manual-mask review overlay",
        )
        row["manual_mask_exists"] = 1
        self._log("save_manual", self.mask)
        self._update_review_status("Saved manual.")
        print(f"[ATTENTION][EDITOR] Saved manual mask: {path}", flush=True)

    def _save_next_click(self, _button):
        try:
            self.save()
            self.next()
        except Exception as error:
            self._button_error("Save & Next", error)

    def _accept_click(self, _button):
        try:
            # Acceptance records that the automatic mask was inspected. It does
            # not copy an automatic prediction into manual ground truth.
            self._log("accept_auto", self.auto_mask)
            self._update_review_status("Accepted automatic mask.")
            self.next()
        except Exception as error:
            self._button_error("Accept auto", error)

    def _skip_click(self, _button):
        try:
            self._log("skip", None)
            self._update_review_status("Skipped.")
            self.next()
        except Exception as error:
            self._button_error("Skip", error)

    def _delete_manual_click(self, _button):
        try:
            row = self.rows[self.index]
            path = Path(row["manual_mask_path"])
            path.unlink(missing_ok=True)
            row["manual_mask_exists"] = 0
            self._log("delete_manual", None)
            self.mask = self.auto_mask.copy()
            self.base_mask = self.auto_mask.copy()
            self._sync_mask_to_frontend()
            self._render()
            self._update_review_status("Deleted manual; automatic restored.")
        except Exception as error:
            self._button_error("Delete manual", error)

    def next(self):
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.load_current()
        else:
            self._update_review_status("End of review queue.")

    def previous(self):
        if self.index > 0:
            self.index -= 1
            self.load_current()




# ---------------------------------------------------------------------------
# Extended CLI / execution workflow.
# ---------------------------------------------------------------------------













# region VS CODE OUTLINE — FUNCTIONAL SERVICES
class RuntimeManager:
    """Runtime device selection, progress logging, timing and console capture."""

    @staticmethod
    def _resolve_requested_runtime_device(requested):
        """Resolve auto/cpu/cuda against the live accelerator state."""

        requested = str(requested or "auto").strip().lower()
        if requested not in SETTINGS_RUNTIME.RUNTIME_DEVICE_CHOICES:
            raise ValueError(
                "Runtime device must be one of auto/cpu/cuda; received "
                f"{requested!r}."
            )
        if requested == "auto":
            return requested, ("cuda" if torch.cuda.is_available() else "cpu")
        return requested, requested

    @staticmethod
    def set_runtime_device(
        requested=None,
        context="runtime",
        strict_cuda=False,
    ):
        """Select CPU/GPU explicitly and release stale CUDA allocations safely.

        Parameters
        ----------
        requested:
            ``"auto"``, ``"cpu"`` or ``"cuda"``. When omitted, the current
            ``CAD_RUNTIME_DEVICE`` environment value is used.
        context:
            Human-readable text included in logs.
        strict_cuda:
            If True, an explicit CUDA request fails immediately when Kaggle has no
            live accelerator. If False, it falls back to CPU with a visible warning.
        """

        global DEVICE

        if requested is None:
            requested = os.environ.get(
                "CAD_RUNTIME_DEVICE", SETTINGS_RUNTIME.RUNTIME_DEVICE_POLICY
            )
        policy, resolved = _resolve_requested_runtime_device(requested)

        if resolved == "cuda" and not torch.cuda.is_available():
            message = (
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Enable a Kaggle GPU accelerator and restart the session before "
                f"running {context}."
            )
            if strict_cuda:
                raise RuntimeError(message)
            print(f"[RUNTIME][DEVICE][WARNING] {message} Falling back to CPU.", flush=True)
            resolved = "cpu"

        previous = str(DEVICE)
        if previous == "cuda" and resolved != "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        DEVICE = resolved
        SETTINGS_RUNTIME.RUNTIME_DEVICE_POLICY = policy
        os.environ["CAD_RUNTIME_DEVICE"] = policy

        if previous != resolved or context:
            print(
                f"[RUNTIME][DEVICE] context={context}; requested={policy}; "
                f"resolved={resolved}; cuda_available={torch.cuda.is_available()}.",
                flush=True,
            )
        return torch.device(resolved)

    @staticmethod
    def refresh_runtime_device(context="runtime"):
        """Refresh the current policy against the live Kaggle runtime."""

        return set_runtime_device(
            requested=os.environ.get(
                "CAD_RUNTIME_DEVICE", SETTINGS_RUNTIME.RUNTIME_DEVICE_POLICY
            ),
            context=context,
            strict_cuda=False,
        )

    @staticmethod
    def release_gpu_resources(context="runtime"):
        """Release Python/CUDA caches after one isolated GPU stage.

        This frees VRAM for later code in the same kernel. It does not detach the
        accelerator from a Kaggle session; saving GPU quota still requires changing
        the notebook Accelerator to None and restarting before CPU-only stages.
        """

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        print(
            f"[RUNTIME][DEVICE] Released GPU caches after {context}.",
            flush=True,
        )

    @staticmethod
    def set_feature_cache_device_tag(tag="auto"):
        """Select which extraction device identity is used in cache fingerprints."""

        tag = str(tag).strip().lower()
        if tag not in SETTINGS_RUNTIME.RUNTIME_DEVICE_CHOICES:
            raise ValueError(
                "Feature-cache device tag must be auto/cpu/cuda; received "
                f"{tag!r}."
            )
        SETTINGS_FEATURE_BANK.FEATURE_CACHE_DEVICE_TAG = tag
        os.environ["CAD_FEATURE_CACHE_DEVICE_TAG"] = tag
        return tag

    @staticmethod
    def resolved_feature_cache_device_tag():
        return DEVICE if SETTINGS_FEATURE_BANK.FEATURE_CACHE_DEVICE_TAG == "auto" else SETTINGS_FEATURE_BANK.FEATURE_CACHE_DEVICE_TAG

    @staticmethod
    def _synchronize_timing_device():
        """Synchronize CUDA before reading a timer when GPU work may be pending."""

        # CUDA kernels are normally asynchronous relative to Python. Without an
        # explicit synchronization, a timer can stop before the GPU has completed
        # the operation being measured. CPU execution requires no synchronization.
        # The availability guard also protects a notebook whose DEVICE variable was
        # created before a Kaggle accelerator/session restart.
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
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

    @staticmethod
    def _print_stage_start(stage_number, title, expected_workload):
        """Print a visible stage header and return a high-resolution start time."""

        print("\n" + "=" * 78, flush=True)
        print(
            f"[PIPELINE {stage_number:02d}/{SETTINGS_REPORTING.PIPELINE_STAGE_COUNT:02d}] "
            f"START: {title}",
            flush=True,
        )
        print(
            f"[PIPELINE {stage_number:02d}/{SETTINGS_REPORTING.PIPELINE_STAGE_COUNT:02d}] "
            f"Expected workload: {expected_workload}",
            flush=True,
        )
        print("=" * 78, flush=True)

        _synchronize_timing_device()
        return time.perf_counter()

    @staticmethod
    def _print_stage_complete(stage_number, title, started_at, details=None):
        """Print measured stage duration and return it in seconds."""

        _synchronize_timing_device()
        elapsed = time.perf_counter() - started_at

        print(
            f"[PIPELINE {stage_number:02d}/{SETTINGS_REPORTING.PIPELINE_STAGE_COUNT:02d}] "
            f"COMPLETED: {title} in {_format_elapsed_time(elapsed)}",
            flush=True,
        )

        if details:
            print(
                f"[PIPELINE {stage_number:02d}/{SETTINGS_REPORTING.PIPELINE_STAGE_COUNT:02d}] "
                f"{details}",
                flush=True,
            )

        return float(elapsed)

    @staticmethod
    def _print_detail(message):
        """Print a flushed substage message when detailed logging is enabled."""

        if SETTINGS_RUNTIME.ENABLE_DETAILED_PROGRESS_PRINTS:
            print(f"[DETAIL] {message}", flush=True)

    @staticmethod
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

    @staticmethod
    def run_with_console_logging():
        """Run main() while preserving every print() and traceback in one log."""

        SETTINGS_PATHS.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with open(SETTINGS_PATHS.CONSOLE_LOG_PATH, "w", encoding="utf-8", buffering=1) as log_file:
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


class ConfigurationManager:
    """Experiment selection, configuration validation and reproducibility metadata."""

    @staticmethod
    def get_enabled_experiments():
        """Return the ordered experiment list selected by hard-coded configuration."""

        selected_ids = None if SETTINGS_EXPERIMENTS.EXPERIMENTS_TO_RUN is None else set(SETTINGS_EXPERIMENTS.EXPERIMENTS_TO_RUN)
        experiments = [
            experiment
            for experiment in SETTINGS_EXPERIMENTS.EXPERIMENT_REGISTRY
            if experiment.enabled
            and (selected_ids is None or experiment.experiment_id in selected_ids)
        ]

        if selected_ids is not None:
            known_ids = {experiment.experiment_id for experiment in SETTINGS_EXPERIMENTS.EXPERIMENT_REGISTRY}
            unknown = sorted(selected_ids - known_ids)
            if unknown:
                raise ValueError(
                    "EXPERIMENTS_TO_RUN contains unknown experiment IDs: "
                    f"{unknown}."
                )

        return experiments

    @staticmethod
    def validate_configuration():
        """Fail before data loading when the final protocol is inconsistent."""

        experiments = get_enabled_experiments()
        ids = [experiment.experiment_id for experiment in experiments]
        expected_ids = {
            SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.FIXED_PERIPHERY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.MASK_ONLY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.COMPLEMENT_CONTROL_EXPERIMENT_ID,
        }
        if set(ids) != expected_ids or len(ids) != len(expected_ids):
            raise ValueError(
                "The active registry must contain exactly the seven final "
                f"experiments; received {ids}."
            )
        if SETTINGS_EXPERIMENTS.BASELINE_EXPERIMENT_ID != SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID:
            raise ValueError("BASELINE_EXPERIMENT_ID must equal the reference model.")

        allowed_modes = {
            "standardized_roi_zero_bg_center_fallback",
            "standardized_hard_support_region_norm",
            "standardized_fixed_periphery_region_norm",
            "standardized_a17_exact_support_mask_only",
            "standardized_a17_support_intensity_affine_shuffled",
            "standardized_a17_exact_support_complement_region_norm",
        }
        for experiment in experiments:
            if experiment.feature_mode not in allowed_modes:
                raise ValueError(
                    f"{experiment.experiment_id}: unsupported feature mode "
                    f"{experiment.feature_mode!r}."
                )
            if experiment.strategy != "patient_embedding":
                raise ValueError("Only patient_embedding experiments are supported.")
            if experiment.pooling_strategy != "hierarchical":
                raise ValueError("Only hierarchical pooling is supported.")
            if experiment.weighting_mode != "equal":
                raise ValueError("Only equal slice weighting is supported.")
            if experiment.classifier_type != "logistic_regression":
                raise ValueError("Only Logistic Regression is supported.")
            if experiment.fusion_method is not None:
                raise ValueError("Probability fusion is not part of the final pipeline.")
            if not experiment.use_pca or not experiment.tune_c:
                raise ValueError("Every final experiment must use fold-local PCA and C selection.")
            if experiment.slice_filter not in {"all", "standardized_monai_valid"}:
                raise ValueError(f"Invalid slice filter: {experiment.slice_filter!r}.")
            if experiment.slice_dropout_rate != 0.0:
                raise ValueError("Slice-dropout experiments are not part of the final pipeline.")
            if experiment.deduplicate_exact_within_patient:
                raise ValueError("Within-patient deduplication is audit-only in the final pipeline.")

        if SETTINGS_VALIDATION.N_SPLITS < 2 or SETTINGS_VALIDATION.INNER_CV_SPLITS < 2:
            raise ValueError("Outer and inner CV need at least two folds.")
        if not SETTINGS_VALIDATION.CLASSIFIER_C_GRID or any(float(value) <= 0 for value in SETTINGS_VALIDATION.CLASSIFIER_C_GRID):
            raise ValueError("CLASSIFIER_C_GRID must contain positive values.")
        if not 0.0 < SETTINGS_VALIDATION.PATIENT_PCA_EXPLAINED_VARIANCE <= 1.0:
            raise ValueError("PATIENT_PCA_EXPLAINED_VARIANCE must lie in (0,1].")
        if SETTINGS_VALIDATION.BOOTSTRAP_REPLICATES < 1 or SETTINGS_VALIDATION.PAIRED_BOOTSTRAP_REPLICATES < 1:
            raise ValueError("Bootstrap replicate counts must be positive.")
        if SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS < 1:
            raise ValueError("Repeated-CV count must be positive.")
        if SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES < 1:
            raise ValueError("Permutation replicate count must be positive.")
        if SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES < 1:
            raise ValueError("Selection-adjusted permutation count must be positive.")

        enabled = set(ids)
        requested = (
            set(SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS)
            | set(SETTINGS_VALIDATION.PERMUTATION_EXPERIMENT_IDS)
            | set(SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS)
            | set(SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS)
        )
        missing = sorted(requested - enabled)
        if missing:
            raise ValueError(f"Validation analysis references disabled experiments: {missing}")

        if SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS[0] != SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID:
            raise ValueError("The primary candidate must be first in model-family ties.")
        if tuple(SETTINGS_VALIDATION.PERMUTATION_EXPERIMENT_IDS) != (SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,):
            raise ValueError("The ordinary label-permutation test must target the primary candidate.")

        catalog = {experiment.experiment_id: experiment for experiment in experiments}
        primary = catalog[SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID]
        valid_only = catalog[SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID]
        exact_controls = (
            catalog[SETTINGS_EXPERIMENTS.MASK_ONLY_CONTROL_EXPERIMENT_ID],
            catalog[SETTINGS_EXPERIMENTS.SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID],
            catalog[SETTINGS_EXPERIMENTS.COMPLEMENT_CONTROL_EXPERIMENT_ID],
        )
        common_fields = (
            "strategy", "pooling_strategy", "weighting_mode", "classifier_type",
            "use_pca", "tune_c", "fixed_c", "slice_dropout_rate",
            "deduplicate_exact_within_patient",
        )
        for control in exact_controls:
            for field in common_fields:
                if getattr(control, field) != getattr(primary, field):
                    raise ValueError(
                        f"{control.experiment_id}: {field} must match the primary model."
                    )
            if control.slice_filter != "all":
                raise ValueError("Exact-support controls must use the same all-slice rows.")
        if valid_only.feature_mode != primary.feature_mode:
            raise ValueError("The valid-only ablation must reuse the primary feature mode.")
        if valid_only.slice_filter != "standardized_monai_valid":
            raise ValueError("The valid-only ablation must use the MONAI-valid filter.")

        if SETTINGS_PREPROCESSING.HARD_SUPPORT_DILATION_KERNEL <= 0 or SETTINGS_PREPROCESSING.HARD_SUPPORT_DILATION_KERNEL % 2 == 0:
            raise ValueError("HARD_SUPPORT_DILATION_KERNEL must be a positive odd integer.")
        if not 0.0 < SETTINGS_PREPROCESSING.FIXED_PERIPHERY_EXCLUSION_FRACTION < 1.0:
            raise ValueError("FIXED_PERIPHERY_EXCLUSION_FRACTION must lie in (0,1).")
        if not 0.0 <= SETTINGS_PREPROCESSING.REGION_NORM_LOWER_PERCENTILE < SETTINGS_PREPROCESSING.REGION_NORM_UPPER_PERCENTILE <= 100.0:
            raise ValueError("Invalid region-normalization percentile interval.")
        if SETTINGS_PREPROCESSING.REGION_NORM_MIN_PIXELS < 1 or SETTINGS_PREPROCESSING.REGION_NORM_HISTOGRAM_BINS < 2:
            raise ValueError("Region-normalization minimums are invalid.")
        if not 0.0 < SETTINGS_PREPROCESSING.REGION_NORM_MIN_DYNAMIC_RANGE <= 1.0:
            raise ValueError("REGION_NORM_MIN_DYNAMIC_RANGE must lie in (0,1].")
        if not str(SETTINGS_PREPROCESSING.SUPPORT_INTENSITY_SHUFFLE_SCHEMA).strip():
            raise ValueError("SUPPORT_INTENSITY_SHUFFLE_SCHEMA cannot be empty.")
        if SETTINGS_RUNTIME.USE_CUDA_AMP:
            raise ValueError("Deterministic final runs require USE_CUDA_AMP=False.")

        run_exact_support_transform_contract_self_test()

    @staticmethod
    def collect_suite_metadata(
        samples,
        fingerprint,
        cache_status,
        exact_summary,
        phash_summary,
        standardized_monai_gate_comparison,
        stability_summary,
        permutation_summary,
        candidate_family_selection_summary,
        exact_support_row_contract_summary,
        successful_results,
        failed_results,
        total_runtime,
    ):
        """Collect compact reproducibility, cohort, audit and completion metadata."""

        patient_ids, patient_labels = build_patient_label_table(
            [sample[1] for sample in samples], [sample[2] for sample in samples]
        )
        bundle_root = locate_monai_bundle_root()
        model_path = bundle_root / "models" / "model.ts"
        return {
            "pipeline_schema": SETTINGS_RUNTIME.PIPELINE_SCHEMA_ID,
            "suite_name": SETTINGS_PATHS.SUITE_NAME,
            "suite_configuration_tag": SETTINGS_PATHS.SUITE_CONFIGURATION_TAG,
            "dataset_path": str(SETTINGS_PATHS.DATASET_PATH),
            "n_images": len(samples),
            "n_patients": len(patient_ids),
            "normal_patients": int(np.sum(patient_labels == 0)),
            "sick_patients": int(np.sum(patient_labels == 1)),
            "patient_definition": "Directory_*",
            "feature_bank_fingerprint": fingerprint,
            "feature_cache_status": cache_status,
            "reference_experiment_id": SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
            "primary_candidate_experiment_id": SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "enabled_experiment_ids": [e.experiment_id for e in get_enabled_experiments()],
            "validation_runtime_profile": SETTINGS_VALIDATION.VALIDATION_RUNTIME_PROFILE,
            "device": DEVICE,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "sklearn_version": sklearn.__version__,
            "opencv_version": cv2.__version__,
            "monai_bundle": SETTINGS_MONAI.MONAI_BUNDLE_NAME,
            "monai_bundle_version": SETTINGS_MONAI.MONAI_BUNDLE_VERSION,
            "monai_runtime_source": MONAI_RUNTIME_SOURCE,
            "monai_model_ts_sha256": sha256_file(model_path) if model_path.is_file() else None,
            "exact_duplicate_audit": exact_summary,
            "perceptual_duplicate_audit": phash_summary,
            "monai_gate_comparison": standardized_monai_gate_comparison,
            "stability_summary": stability_summary,
            "permutation_summary": permutation_summary,
            "model_family_nested_selection": candidate_family_selection_summary,
            "exact_support_row_contract": exact_support_row_contract_summary,
            "successful_experiments": [r["config"].experiment_id for r in successful_results],
            "failed_experiments": [r.get("experiment_id") for r in failed_results],
            "total_runtime_seconds": float(total_runtime),
            "clinical_status": "exploratory_internal_validation_only",
        }

    @staticmethod
    def write_suite_configuration(output_path, experiments):
        """Save the complete current protocol before evaluation begins."""

        configuration = {
            "pipeline_schema": SETTINGS_RUNTIME.PIPELINE_SCHEMA_ID,
            "suite_name": SETTINGS_PATHS.SUITE_NAME,
            "suite_configuration_tag": SETTINGS_PATHS.SUITE_CONFIGURATION_TAG,
            "reference_experiment_id": SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
            "primary_candidate_experiment_id": SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "valid_only_ablation_experiment_id": SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,
            "control_experiment_ids": [
                SETTINGS_EXPERIMENTS.FIXED_PERIPHERY_CONTROL_EXPERIMENT_ID,
                SETTINGS_EXPERIMENTS.MASK_ONLY_CONTROL_EXPERIMENT_ID,
                SETTINGS_EXPERIMENTS.SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
                SETTINGS_EXPERIMENTS.COMPLEMENT_CONTROL_EXPERIMENT_ID,
            ],
            "experiments": [asdict(experiment) for experiment in experiments],
            "primary_ablation_comparisons": SETTINGS_EXPERIMENTS.PRIMARY_ABLATION_COMPARISONS,
            "outer_cv": {"splits": SETTINGS_VALIDATION.N_SPLITS, "random_state": SETTINGS_VALIDATION.CV_RANDOM_STATE},
            "inner_cv": {
                "splits": SETTINGS_VALIDATION.INNER_CV_SPLITS,
                "random_state": SETTINGS_VALIDATION.INNER_CV_RANDOM_STATE,
                "c_grid": SETTINGS_VALIDATION.CLASSIFIER_C_GRID,
                "c_selection_auc_tolerance": SETTINGS_VALIDATION.C_SELECTION_AUC_TOLERANCE,
                "threshold_method": SETTINGS_VALIDATION.THRESHOLD_SELECTION_METHOD,
            },
            "bootstrap": {
                "replicates": SETTINGS_VALIDATION.BOOTSTRAP_REPLICATES,
                "paired_replicates": SETTINGS_VALIDATION.PAIRED_BOOTSTRAP_REPLICATES,
                "confidence": SETTINGS_VALIDATION.BOOTSTRAP_CONFIDENCE,
            },
            "stability": {
                "enabled": SETTINGS_VALIDATION.RUN_REPEATED_NESTED_CV_STABILITY,
                "repeats": SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS,
                "experiment_ids": list(SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS),
                "comparisons": SETTINGS_VALIDATION.REPEATED_STABILITY_COMPARISONS,
            },
            "permutation": {
                "enabled": SETTINGS_VALIDATION.RUN_PATIENT_LABEL_PERMUTATION_TEST,
                "replicates": SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES,
                "experiment_ids": list(SETTINGS_VALIDATION.PERMUTATION_EXPERIMENT_IDS),
                "selection_adjusted_enabled": SETTINGS_VALIDATION.RUN_SELECTION_ADJUSTED_PERMUTATION_TEST,
                "selection_adjusted_replicates": SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES,
                "selection_adjusted_candidate_ids": list(SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS),
            },
            "model_family_nested_selection": {
                "enabled": SETTINGS_VALIDATION.RUN_MODEL_FAMILY_NESTED_SELECTION,
                "experiment_ids": list(SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS),
                "auc_tolerance": SETTINGS_VALIDATION.MODEL_FAMILY_SELECTION_AUC_TOLERANCE,
            },
            "preprocessing": {
                "img_size": SETTINGS_PREPROCESSING.IMG_SIZE,
                "monai_input_size": SETTINGS_MONAI.MONAI_INPUT_SIZE,
                "standardized_content_long_side": SETTINGS_PREPROCESSING.STANDARDIZED_CONTENT_LONG_SIDE,
                "center_crop_fallback_fraction": SETTINGS_PREPROCESSING.CENTER_CROP_FALLBACK_FRACTION,
                "hard_support_dilation_kernel": SETTINGS_PREPROCESSING.HARD_SUPPORT_DILATION_KERNEL,
                "fixed_periphery_exclusion_fraction": SETTINGS_PREPROCESSING.FIXED_PERIPHERY_EXCLUSION_FRACTION,
                "region_norm_percentiles": [SETTINGS_PREPROCESSING.REGION_NORM_LOWER_PERCENTILE, SETTINGS_PREPROCESSING.REGION_NORM_UPPER_PERCENTILE],
                "region_norm_min_pixels": SETTINGS_PREPROCESSING.REGION_NORM_MIN_PIXELS,
                "region_norm_min_dynamic_range": SETTINGS_PREPROCESSING.REGION_NORM_MIN_DYNAMIC_RANGE,
                "support_intensity_shuffle_schema": SETTINGS_PREPROCESSING.SUPPORT_INTENSITY_SHUFFLE_SCHEMA,
            },
            "models": {
                "monai_bundle": SETTINGS_MONAI.MONAI_BUNDLE_NAME,
                "monai_bundle_version": SETTINGS_MONAI.MONAI_BUNDLE_VERSION,
                "monai_hf_revision": SETTINGS_MONAI.MONAI_HF_REVISION,
                "efficientnet_weights": SETTINGS_FEATURE_BANK.EFFICIENTNET_WEIGHTS_NAME,
            },
            "runtime": {
                "validation_profile": SETTINGS_VALIDATION.VALIDATION_RUNTIME_PROFILE,
                "device_policy": SETTINGS_RUNTIME.RUNTIME_DEVICE_POLICY,
                "suite_device_tag": SETTINGS_RUNTIME.SUITE_DEVICE_TAG,
                "feature_cache_device_tag": SETTINGS_FEATURE_BANK.FEATURE_CACHE_DEVICE_TAG,
                "deterministic_algorithms": True,
                "cuda_amp": SETTINGS_RUNTIME.USE_CUDA_AMP,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(configuration, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def validate_attention_extension():
        """Run fast architecture and exact-support contract checks before long jobs."""

        model = AttentionUNet(base_channels=8).cpu().eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 64, 64))
        if output.shape != (2, 1, 64, 64) or not torch.isfinite(output).all():
            raise RuntimeError("Attention U-Net architecture self-test failed.")

        hard = torch.zeros(2, 1, SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE)
        hard[0, 0, 80:130, 85:135] = 1.0
        valid = torch.tensor([True, False])
        content = torch.ones(2, 1, SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE)
        support = create_attention_exact_support_mask(hard, valid, content)
        if support.shape != hard.shape or not torch.any(support[1] > 0.5):
            raise RuntimeError("Attention support/fallback self-test failed.")
        raw = torch.linspace(0.0, 1.0, SETTINGS_PREPROCESSING.IMG_SIZE * SETTINGS_PREPROCESSING.IMG_SIZE).reshape(1, 1, SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE)
        raw = raw.repeat(2, 3, 1, 1)
        au1 = _robust_scale_visible_regions(raw, support)
        hashes = [hashlib.sha256(f"attention-test-{i}".encode()).hexdigest() for i in range(2)]
        shuffled = create_attention_support_shuffled_images(au1, support, hashes)
        complement = _robust_scale_visible_regions(raw, 1.0 - support)
        for index in range(2):
            mask = support[index, 0].reshape(-1) > 0.5
            original_values = np.sort(au1[index, 0].reshape(-1)[mask].numpy())
            shuffled_values = np.sort(shuffled[index, 0].reshape(-1)[mask].numpy())
            if not np.array_equal(original_values, shuffled_values):
                raise RuntimeError("Attention shuffled-histogram self-test failed.")
        if torch.any(au1 * (1.0 - support) != 0) or torch.any(complement * support != 0):
            raise RuntimeError("Attention inside/complement self-test failed.")
        print(
            "[ATTENTION] Architecture and exact-support self-tests passed.",
            flush=True,
        )


class ImagePreprocessor:
    """Label-blind image standardization, intensity scaling and image fingerprints."""

    @staticmethod
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

    @staticmethod
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
            float(values.mean()) <= SETTINGS_PREPROCESSING.STANDARDIZATION_DARK_LINE_MAX_MEAN
            and float(values.std()) <= SETTINGS_PREPROCESSING.STANDARDIZATION_DARK_LINE_MAX_STD
            and float(
                np.mean(values <= SETTINGS_PREPROCESSING.STANDARDIZATION_DARK_PIXEL_MAX_VALUE)
            ) >= SETTINGS_PREPROCESSING.STANDARDIZATION_DARK_PIXEL_MIN_FRACTION
        )

    @staticmethod
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
            int(np.ceil(height * SETTINGS_PREPROCESSING.STANDARDIZATION_MIN_RETAINED_FRACTION)),
        )
        minimum_width = max(
            8,
            int(np.ceil(width * SETTINGS_PREPROCESSING.STANDARDIZATION_MIN_RETAINED_FRACTION)),
        )
        maximum_vertical_crop = int(
            np.floor(height * SETTINGS_PREPROCESSING.STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE)
        )
        maximum_horizontal_crop = int(
            np.floor(width * SETTINGS_PREPROCESSING.STANDARDIZATION_MAX_CROP_FRACTION_PER_SIDE)
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
        if top_crop < SETTINGS_PREPROCESSING.STANDARDIZATION_MIN_PADDING_RUN:
            top_crop = 0
        if bottom_crop < SETTINGS_PREPROCESSING.STANDARDIZATION_MIN_PADDING_RUN:
            bottom_crop = 0
        if left_crop < SETTINGS_PREPROCESSING.STANDARDIZATION_MIN_PADDING_RUN:
            left_crop = 0
        if right_crop < SETTINGS_PREPROCESSING.STANDARDIZATION_MIN_PADDING_RUN:
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

    @staticmethod
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
            np.percentile(image_float, SETTINGS_PREPROCESSING.STANDARDIZATION_LOWER_PERCENTILE)
        )
        upper = float(
            np.percentile(image_float, SETTINGS_PREPROCESSING.STANDARDIZATION_UPPER_PERCENTILE)
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

    @staticmethod
    def _fixed_content_canvas_geometry(height, width):
        """Return deterministic fixed-content geometry for a 256x256 canvas.

        The longest retained image side is resized to exactly
        ``STANDARDIZED_CONTENT_LONG_SIDE`` pixels. Unlike the MONAI
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
        if not 1 <= SETTINGS_PREPROCESSING.STANDARDIZED_CONTENT_LONG_SIDE <= SETTINGS_MONAI.MONAI_INPUT_SIZE:
            raise ValueError(
                "STANDARDIZED_CONTENT_LONG_SIDE must lie between 1 and "
                "MONAI_INPUT_SIZE."
            )

        scale = float(SETTINGS_PREPROCESSING.STANDARDIZED_CONTENT_LONG_SIDE / max(height, width))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))

        # Rounding can move the nominal longest side by one pixel. Force it back to
        # the predeclared target without changing the aspect-ratio calculation for
        # the shorter side.
        if height >= width:
            resized_height = int(SETTINGS_PREPROCESSING.STANDARDIZED_CONTENT_LONG_SIDE)
        else:
            resized_width = int(SETTINGS_PREPROCESSING.STANDARDIZED_CONTENT_LONG_SIDE)

        resized_height = min(resized_height, SETTINGS_MONAI.MONAI_INPUT_SIZE)
        resized_width = min(resized_width, SETTINGS_MONAI.MONAI_INPUT_SIZE)
        top = (SETTINGS_MONAI.MONAI_INPUT_SIZE - resized_height) // 2
        left = (SETTINGS_MONAI.MONAI_INPUT_SIZE - resized_width) // 2
        return resized_height, resized_width, top, left, scale

    @staticmethod
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
            (SETTINGS_MONAI.MONAI_INPUT_SIZE, SETTINGS_MONAI.MONAI_INPUT_SIZE),
            dtype=np.float32,
        )
        canvas[
            top:top + resized_height,
            left:left + resized_width,
        ] = resized
        return canvas, resized_height, resized_width

    @staticmethod
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
            (SETTINGS_MONAI.MONAI_INPUT_SIZE, SETTINGS_MONAI.MONAI_INPUT_SIZE),
            dtype=np.float32,
        )
        canvas[
            top:top + resized_height,
            left:left + resized_width,
        ] = 0.0
        return canvas, resized_height, resized_width

    @staticmethod
    def build_label_blind_standardized_image(image):
        """Create aligned MONAI/classifier canvases and standardization audits.

        Processing order:

            native uint8 JPEG
                -> conservative dark-uniform edge detection
                -> crop only the detected edge runs
                -> min-max scaling for the pretrained MONAI segmenter
                -> robust fixed-percentile scaling for EfficientNet views
                -> unscaled [0,1] uint8-intensity view for region normalization
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
                256x256 retained uint8 intensity divided by 255, with no per-image
                min-max or global percentile scaling. computes percentiles only
                inside the final visible region after MONAI localization.
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
        content_height_fraction = float(resized_height / SETTINGS_MONAI.MONAI_INPUT_SIZE)
        content_width_fraction = float(resized_width / SETTINGS_MONAI.MONAI_INPUT_SIZE)
        pipeline_padding_fraction = float(
            1.0
            - (resized_height * resized_width)
            / float(SETTINGS_MONAI.MONAI_INPUT_SIZE * SETTINGS_MONAI.MONAI_INPUT_SIZE)
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

        if len(features) != len(SETTINGS_PREPROCESSING.STANDARDIZATION_FEATURE_NAMES):
            raise RuntimeError(
                "Standardization feature-name and value counts differ."
            )
        if not np.all(np.isfinite(features)):
            raise RuntimeError("Standardization features contain non-finite values.")

        return robust_canvas, monai_canvas, raw_canvas, padding_canvas, features

    @staticmethod
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

    @staticmethod
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

        if len(features) != len(SETTINGS_REPORTING.PROVENANCE_FEATURE_NAMES):
            raise RuntimeError("Provenance feature-name and value counts differ.")
        if not np.all(np.isfinite(features)):
            raise RuntimeError(
                f"Non-finite provenance feature generated for {image_path}."
            )

        return features


class DatasetManager:
    """Dataset discovery and patient/series-proxy row construction."""

    @staticmethod
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


class MonaiSegmenter:
    """Pinned MONAI bundle preparation, verification and ventricular inference."""

    @staticmethod
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

    @staticmethod
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
        direct_root = SETTINGS_MONAI.MONAI_BUNDLE_DIR / SETTINGS_MONAI.MONAI_BUNDLE_NAME

        if any(
            (direct_root / relative_path).is_file()
            for relative_path in (
                Path("models/model.ts"),
                Path("models/model.pt"),
            )
        ):
            return direct_root

        # If the canonical location is absent, search deterministic alternate
        # layouts. Candidate paths still must contain the requested bundle slug.
        if SETTINGS_MONAI.MONAI_BUNDLE_DIR.exists():
            candidate_artifacts = []

            # Prefer the official TorchScript artifact when several old layouts
            # coexist, then fall back to the state-dict checkpoint.
            for filename in ("model.ts", "model.pt"):
                candidate_artifacts.extend(
                    sorted(SETTINGS_MONAI.MONAI_BUNDLE_DIR.rglob(filename))
                )

            for artifact_path in candidate_artifacts:
                if artifact_path.parent.name != "models":
                    continue

                candidate_root = artifact_path.parent.parent

                if SETTINGS_MONAI.MONAI_BUNDLE_NAME in candidate_root.parts:
                    return candidate_root

        return direct_root

    @staticmethod
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
        if version != SETTINGS_MONAI.MONAI_BUNDLE_VERSION:
            raise RuntimeError(
                "MONAI bundle version mismatch: "
                f"expected {SETTINGS_MONAI.MONAI_BUNDLE_VERSION}, metadata reports {version!r}."
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

    @staticmethod
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

    @staticmethod
    def verify_monai_artifact_sha256(path, expected_sha256, artifact_label):
        """Verify a pinned MONAI artifact before it influences extracted features."""

        if not SETTINGS_MONAI.VERIFY_MONAI_ARTIFACT_SHA256:
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

    @staticmethod
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

        if not SETTINGS_MONAI.AUTO_DOWNLOAD_MONAI_BUNDLE:
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
            f"repository revision {SETTINGS_MONAI.MONAI_HF_REVISION[:8]}: {', '.join(missing)}",
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
                repo_id=SETTINGS_MONAI.MONAI_HF_REPO_ID,
                revision=SETTINGS_MONAI.MONAI_HF_REVISION,
                local_dir=str(bundle_root),
                allow_patterns=list(required_relative_paths),
            )
        except Exception as exc:
            raise RuntimeError(
                "Automatic pinned MONAI download failed. Verify internet access "
                f"and repository availability for {SETTINGS_MONAI.MONAI_HF_REPO_ID} at revision "
                f"{SETTINGS_MONAI.MONAI_HF_REVISION}. For offline execution, copy the requested "
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

    @staticmethod
    def load_and_validate_torchscript_segmenter(path, source_description):
        """Load TorchScript and validate its fixed inference contract immediately.

        The live runtime device is re-evaluated here. If CUDA is available but the
        current TorchScript/PyTorch combination cannot execute on it, one explicit
        CPU retry is attempted before the code falls back to MONAI reconstruction.
        This prevents a stale notebook ``DEVICE='cuda'`` value from causing an
        unnecessary ``ModuleNotFoundError: monai`` on a CPU-only Kaggle session.
        """

        global MONAI_RUNTIME_SOURCE, MONAI_RUNTIME_ARTIFACT_PATH, DEVICE

        load_started_at = time.perf_counter()
        primary_device = refresh_runtime_device(
            f"loading {source_description}"
        )
        attempt_devices = [primary_device]
        if primary_device.type == "cuda":
            # CPU is a compatibility fallback only. It is not selected when CUDA
            # succeeds, and it avoids importing MONAI merely because one CUDA load
            # or zero-input execution path is incompatible with the current build.
            attempt_devices.append(torch.device("cpu"))

        print(
            f"[MODEL][MONAI] Loading {source_description} from: {path}",
            flush=True,
        )
        print(
            f"[MODEL][MONAI] Live target device: {primary_device}. A zero-input "
            "inference sanity check will run immediately after loading.",
            flush=True,
        )

        failures = []
        for attempt_index, runtime_device in enumerate(attempt_devices, start=1):
            try:
                # ``map_location`` must be a device resolved from the CURRENT
                # runtime, not the stale global string captured by an earlier
                # notebook session. Calling ``to`` afterwards also relocates any
                # parameters/buffers not covered by storage remapping.
                network = torch.jit.load(
                    str(path),
                    map_location=runtime_device,
                )
                network = network.to(runtime_device)
                network.eval()

                try:
                    network.requires_grad_(False)
                except (AttributeError, RuntimeError):
                    # Some TorchScript module types do not expose this mutator.
                    # Inference is still protected by torch.inference_mode below.
                    pass

                example_input = torch.zeros(
                    1,
                    1,
                    SETTINGS_MONAI.MONAI_INPUT_SIZE,
                    SETTINGS_MONAI.MONAI_INPUT_SIZE,
                    device=runtime_device,
                    dtype=torch.float32,
                )

                _print_detail(
                    "Running MONAI zero-input shape/finite-value validation on "
                    f"{runtime_device}."
                )
                with torch.inference_mode():
                    example_output = network(example_input)

                if not isinstance(example_output, torch.Tensor):
                    raise RuntimeError(
                        f"{source_description} returned {type(example_output)} "
                        "instead of a tensor."
                    )

                expected_shape = (
                    1,
                    4,
                    SETTINGS_MONAI.MONAI_INPUT_SIZE,
                    SETTINGS_MONAI.MONAI_INPUT_SIZE,
                )
                if tuple(example_output.shape) != expected_shape:
                    raise RuntimeError(
                        f"{source_description} output shape mismatch: expected "
                        f"{expected_shape}, got {tuple(example_output.shape)}."
                    )
                if not torch.isfinite(example_output).all():
                    raise RuntimeError(
                        f"{source_description} produced non-finite values on a "
                        "zero-input sanity check."
                    )

                # Downstream code consistently uses the global DEVICE string. Keep
                # it aligned with the device on which validation actually succeeded.
                DEVICE = runtime_device.type
                MONAI_RUNTIME_SOURCE = str(source_description)
                MONAI_RUNTIME_ARTIFACT_PATH = str(Path(path).resolve())

                if runtime_device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - load_started_at
                print(
                    f"[MODEL][MONAI] Loaded and validated {source_description} on "
                    f"{runtime_device} in {_format_elapsed_time(elapsed)}; "
                    f"output_shape={tuple(example_output.shape)}.",
                    flush=True,
                )
                if runtime_device.type == "cpu" and primary_device.type == "cuda":
                    print(
                        "[MODEL][MONAI][WARNING] CUDA execution failed, so this "
                        "segmenter will continue on CPU. Mask generation will be "
                        "slower, but MONAI Python is not required.",
                        flush=True,
                    )
                return network

            except Exception as exc:
                failures.append((str(runtime_device), exc))
                try:
                    del network
                except UnboundLocalError:
                    pass
                if runtime_device.type == "cuda" and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                if attempt_index < len(attempt_devices):
                    print(
                        "[MODEL][MONAI][WARNING] TorchScript execution failed on "
                        f"{runtime_device}; retrying the same verified artifact on "
                        f"{attempt_devices[attempt_index]}. Reason: {exc}",
                        flush=True,
                    )

        details = "; ".join(
            f"{device}: {type(error).__name__}: {error}"
            for device, error in failures
        )
        final_error = failures[-1][1]
        raise RuntimeError(
            f"{source_description} failed on every safe runtime device ({details})."
        ) from final_error

    @staticmethod
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

    @staticmethod
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

        refresh_runtime_device("building MONAI segmenter")

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

        if not SETTINGS_MONAI.FORCE_REBUILD_MONAI_TORCHSCRIPT:
            bundle_root = ensure_monai_bundle(
                (
                    "models/model.ts",
                    "configs/metadata.json",
                )
            )
            official_torchscript_path = bundle_root / "models" / "model.ts"

            verify_monai_artifact_sha256(
                official_torchscript_path,
                SETTINGS_MONAI.MONAI_OFFICIAL_TORCHSCRIPT_SHA256,
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
            SETTINGS_MONAI.MONAI_TORCHSCRIPT_PATH.is_file()
            and not SETTINGS_MONAI.FORCE_REBUILD_MONAI_TORCHSCRIPT
        ):
            try:
                network = load_and_validate_torchscript_segmenter(
                    SETTINGS_MONAI.MONAI_TORCHSCRIPT_PATH,
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
            SETTINGS_MONAI.MONAI_MODEL_SHA256,
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
            runtime_hint = (
                " Kaggle currently exposes no CUDA device; the verified official "
                "TorchScript artifact was already retried with CPU map_location."
                if not torch.cuda.is_available()
                else ""
            )
            raise ImportError(
                "The verified official/local TorchScript paths failed on all safe "
                "runtime devices, so exceptional model.pt reconstruction now "
                "requires MONAI. Install it in a separate Kaggle setup cell with: "
                "%pip install -q monai==1.6.0, restart the session, and rerun the "
                f"definition cell.{runtime_hint}{reason}"
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
            SETTINGS_MONAI.MONAI_INPUT_SIZE,
            SETTINGS_MONAI.MONAI_INPUT_SIZE,
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
            SETTINGS_MONAI.MONAI_INPUT_SIZE,
            SETTINGS_MONAI.MONAI_INPUT_SIZE,
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

        SETTINGS_MONAI.MONAI_TORCHSCRIPT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        traced_network.save(str(SETTINGS_MONAI.MONAI_TORCHSCRIPT_PATH))

        print(
            "Saved validated fallback MONAI TorchScript cache: "
            f"{SETTINGS_MONAI.MONAI_TORCHSCRIPT_PATH}"
        )

        network = load_and_validate_torchscript_segmenter(
            SETTINGS_MONAI.MONAI_TORCHSCRIPT_PATH,
            "newly reconstructed MONAI TorchScript cache",
        )
        print(
            "[MODEL][MONAI] Exceptional fallback preparation completed in "
            f"{_format_elapsed_time(time.perf_counter() - build_started_at)}.",
            flush=True,
        )
        return network

    @staticmethod
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
            (area_ratio >= SETTINGS_MONAI.MONAI_MIN_HEART_AREA_RATIO)
            & (area_ratio <= SETTINGS_MONAI.MONAI_MAX_HEART_AREA_RATIO)
            & (peak_probability >= SETTINGS_MONAI.MONAI_MIN_PEAK_HEART_PROBABILITY)
        )

        # Max pooling expands the ROI around the predicted ventricles and
        # myocardium. This prevents a narrow segmentation from cutting away nearby
        # diagnostically useful cardiac tissue.
        # Step 7: dilate both soft and hard masks with stride-one max pooling.
        # The operation adds a contextual margin around the predicted ventricles.
        roi_probability_256 = F.max_pool2d(
            heart_probability,
            kernel_size=SETTINGS_MONAI.MONAI_ROI_DILATION_KERNEL,
            stride=1,
            padding=SETTINGS_MONAI.MONAI_ROI_DILATION_KERNEL // 2,
        )

        hard_mask_256 = F.max_pool2d(
            hard_mask_256,
            kernel_size=SETTINGS_MONAI.MONAI_ROI_DILATION_KERNEL,
            stride=1,
            padding=SETTINGS_MONAI.MONAI_ROI_DILATION_KERNEL // 2,
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

    @staticmethod
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
            background_weight = SETTINGS_MONAI.MONAI_BACKGROUND_WEIGHT
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

    @staticmethod
    def normalize_for_efficientnet(images):
        """
        Apply the official ImageNet normalization for EfficientNet-B0.

        Input images must already be float tensors in [0,1]. ROI extraction is
        intentionally completed before this step.
        """

        # Construct per-channel constants on the same device and with the same
        # dtype as the input, avoiding implicit CPU/GPU copies or dtype promotion.
        mean = torch.tensor(
            SETTINGS_FEATURE_BANK.EFFICIENTNET_MEAN,
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            SETTINGS_FEATURE_BANK.EFFICIENTNET_STD,
            device=images.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)

        return (images - mean) / std


class FeatureBankManager:
    """Image-view construction, EfficientNet encoding and feature-cache lifecycle."""

    @staticmethod
    def required_efficientnet_feature_modes(experiments):
        """Return the deterministic EfficientNet views required by the final panel."""

        canonical_order = (
            "standardized_roi_zero_bg_center_fallback",
            "standardized_hard_support_region_norm",
            "standardized_fixed_periphery_region_norm",
            "standardized_a17_exact_support_mask_only",
            "standardized_a17_support_intensity_affine_shuffled",
            "standardized_a17_exact_support_complement_region_norm",
        )
        requested = {
            experiment.feature_mode
            for experiment in experiments
            if experiment.strategy == "patient_embedding"
        }
        unknown = requested.difference(canonical_order)
        if unknown:
            raise ValueError(f"Unsupported EfficientNet feature modes: {sorted(unknown)}")
        return tuple(mode for mode in canonical_order if mode in requested)

    @staticmethod
    def feature_bank_fingerprint(samples, dataset_path):
        """Hash dataset file metadata and every setting that changes frozen views."""

        started = time.perf_counter()
        hasher = hashlib.sha256()
        configuration = {
            "schema": SETTINGS_FEATURE_BANK.FEATURE_CACHE_SCHEMA,
            "dataset_path": str(Path(dataset_path).resolve()),
            "required_modes": list(required_efficientnet_feature_modes(get_enabled_experiments())),
            "img_size": SETTINGS_PREPROCESSING.IMG_SIZE,
            "monai_input_size": SETTINGS_MONAI.MONAI_INPUT_SIZE,
            "standardized_content_long_side": SETTINGS_PREPROCESSING.STANDARDIZED_CONTENT_LONG_SIDE,
            "center_crop_fallback_fraction": SETTINGS_PREPROCESSING.CENTER_CROP_FALLBACK_FRACTION,
            "hard_support_dilation_kernel": SETTINGS_PREPROCESSING.HARD_SUPPORT_DILATION_KERNEL,
            "fixed_periphery_exclusion_fraction": SETTINGS_PREPROCESSING.FIXED_PERIPHERY_EXCLUSION_FRACTION,
            "region_norm": [
                SETTINGS_PREPROCESSING.REGION_NORM_LOWER_PERCENTILE,
                SETTINGS_PREPROCESSING.REGION_NORM_UPPER_PERCENTILE,
                SETTINGS_PREPROCESSING.REGION_NORM_MIN_PIXELS,
                SETTINGS_PREPROCESSING.REGION_NORM_HISTOGRAM_BINS,
                SETTINGS_PREPROCESSING.REGION_NORM_MIN_DYNAMIC_RANGE,
            ],
            "shuffle_schema": SETTINGS_PREPROCESSING.SUPPORT_INTENSITY_SHUFFLE_SCHEMA,
            "monai_bundle": [SETTINGS_MONAI.MONAI_BUNDLE_NAME, SETTINGS_MONAI.MONAI_BUNDLE_VERSION, SETTINGS_MONAI.MONAI_HF_REVISION],
            "efficientnet_weights": SETTINGS_FEATURE_BANK.EFFICIENTNET_WEIGHTS_NAME,
            "device_tag": resolved_feature_cache_device_tag(),
            "use_cuda_amp": SETTINGS_RUNTIME.USE_CUDA_AMP,
        }
        hasher.update(json.dumps(configuration, sort_keys=True).encode("utf-8"))

        dataset_root = Path(dataset_path).resolve()
        total = len(samples)
        checkpoints = set(np.linspace(0, max(total - 1, 0), min(11, max(total, 1)), dtype=int))
        for index, sample in enumerate(samples):
            image_path, label, patient_id, series_id = sample
            path = Path(image_path)
            stat = path.stat()
            try:
                relative = str(path.resolve().relative_to(dataset_root))
            except ValueError:
                relative = str(path.resolve())
            row = (relative, int(label), str(patient_id), str(series_id), int(stat.st_size), int(stat.st_mtime_ns))
            hasher.update(repr(row).encode("utf-8"))
            if index in checkpoints:
                percent = 100.0 * (index + 1) / max(total, 1)
                print(f"[FEATURE BANK FINGERPRINT] {index + 1}/{total} files ({percent:.0f}%)", flush=True)

        digest = hasher.hexdigest()
        print(
            f"[FEATURE BANK] Fingerprint completed in "
            f"{_format_elapsed_time(time.perf_counter() - started)}: {digest[:16]}...",
            flush=True,
        )
        return digest

    @staticmethod
    def _feature_bank_shared_paths(cache_dir):
        """Return the aligned non-embedding arrays stored beside every feature view."""

        names = (
            "labels",
            "patient_ids",
            "series_ids",
            "sample_indices",
            "decoded_pixel_hashes",
            "perceptual_hashes",
            "provenance_features",
            "standardization_features",
            "standardized_monai_valid",
            "standardized_area_ratios",
            "standardized_peak_probabilities",
            "standardized_mean_foreground_probabilities",
        )
        return {name: cache_dir / f"{name}.npy" for name in names}

    @staticmethod
    def _feature_mode_path(cache_dir, mode):
        return cache_dir / f"features__{mode}.npy"

    @staticmethod
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
            if features.shape != (n_slices, SETTINGS_FEATURE_BANK.EFFICIENTNET_FEATURE_DIM):
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

    @staticmethod
    def create_fixed_fraction_center_crop_images(images, crop_fraction):
        """Return one fixed, MONAI-independent center crop resized to input size."""

        crop_fraction = float(crop_fraction)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must lie in (0,1].")
        height, width = images.shape[-2:]
        side = max(2, int(round(min(height, width) * crop_fraction)))
        top = (height - side) // 2
        left = (width - side) // 2
        crop = images[:, :, top:top + side, left:left + side]
        return F.interpolate(
            crop,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def apply_zero_background_roi_with_fixed_center_fallback(
        images,
        roi_probability,
        valid_mask,
        fallback_fraction=SETTINGS_PREPROCESSING.CENTER_CROP_FALLBACK_FRACTION,
    ):
        """Use strict zero-background ROI or a fixed center crop when invalid.

        A full-image fallback would reintroduce export and border information into
        an otherwise strict ROI representation. This helper instead uses a fixed,
        MONAI-independent center crop for invalid masks. It
        retains every slice and makes fallback content deterministic and label-blind.
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

    @staticmethod
    def _fixed_square_bounds(height, width, center_y, center_x, fraction):
        """Return a fixed-size in-bounds square around a floating-point centre."""

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

    @staticmethod
    def _robust_scale_visible_regions(images, visible_masks):
        """Batched masked robust scaling for inside/outside representations.

        ``images`` contains the raw standardized view: retained native JPEG
        intensity divided by 255 and placed in the fixed-content canvas, without a
        global per-image percentile transformation. ``visible_masks`` defines the
        pixels that will remain in each final representation.

        The lower and upper percentiles are estimated independently for every image
        from a fixed-bin masked histogram. This is intentionally batched: an exact
        ``torch.quantile`` call inside a Python loop would sort tens of thousands of
        pixels and synchronize the GPU once per image, making a 63,425-slice
        feature bank unnecessarily slow. With 256 bins, the estimate matches the
        natural precision of the original 8-bit JPEG source while remaining robust
        to the interpolation used for fixed-canvas resizing.

        Pixels outside the visible mask are set to zero after scaling. Consequently,
        cardiac pixels cannot define an outside-region transform and extracardiac
        pixels cannot define an inside-region transform.
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
            raise ValueError(
                "Visible-region masks must align with the image batch and spatial shape."
            )

        bins = int(SETTINGS_PREPROCESSING.REGION_NORM_HISTOGRAM_BINS)
        mask = visible_masks > 0.5
        mask_flat = mask[:, 0].reshape(images.shape[0], -1)
        counts = mask_flat.sum(dim=1).to(torch.long)

        # All three image channels are identical grayscale copies. Histogram only
        # channel 0, then broadcast the resulting limits across all channels.
        values = images[:, 0].float().clamp(0.0, 1.0)
        bin_indices = torch.round(values * float(bins - 1)).to(torch.long)
        bin_indices = bin_indices.clamp_(0, bins - 1).reshape(images.shape[0], -1)

        histogram = torch.zeros(
            images.shape[0],
            bins,
            device=images.device,
            dtype=torch.float32,
        )
        histogram.scatter_add_(
            dim=1,
            index=bin_indices,
            src=mask_flat.to(torch.float32),
        )
        cumulative = torch.cumsum(histogram, dim=1)

        safe_counts = counts.clamp_min(1)
        lower_rank = (
            torch.floor(
                (SETTINGS_PREPROCESSING.REGION_NORM_LOWER_PERCENTILE / 100.0)
                * (safe_counts - 1).to(torch.float32)
            ).to(torch.long)
            + 1
        )
        upper_rank = (
            torch.floor(
                (SETTINGS_PREPROCESSING.REGION_NORM_UPPER_PERCENTILE / 100.0)
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
            (counts >= int(SETTINGS_PREPROCESSING.REGION_NORM_MIN_PIXELS))
            & torch.isfinite(lower)
            & torch.isfinite(upper)
            & (upper > lower)
        )

        lower = lower.view(-1, 1, 1, 1)
        upper = upper.view(-1, 1, 1, 1)
        scaled = (
            (images.float() - lower)
            / (upper - lower).clamp_min(
                float(SETTINGS_PREPROCESSING.REGION_NORM_MIN_DYNAMIC_RANGE)
            )
        ).clamp(0.0, 1.0)
        scaled = scaled * mask.to(scaled.dtype)
        scaled = scaled * valid_rows.view(-1, 1, 1, 1).to(scaled.dtype)
        return scaled.to(images.dtype)

    @staticmethod
    def create_a17_exact_support_mask(
        hard_mask,
        valid_mask,
        content_mask,
    ):
        """Return the exact binary visibility support used by A17 and its controls.

        This helper is the single source of truth for A17, A20, C31, C32 and C33.
        A gate-valid slice receives the already dilated MONAI hard mask followed by
        the fixed additional dilation. A gate-invalid slice receives the fixed
        central square used by A17. In both cases pipeline padding is removed by the
        aligned content mask. No label, patient identifier, series identifier, fold,
        or model score is used.
        """

        if hard_mask.ndim != 4 or hard_mask.shape[1] != 1:
            raise ValueError("hard_mask must have shape [B,1,H,W].")
        if content_mask.ndim != 4 or content_mask.shape[1] != 1:
            raise ValueError("content_mask must have shape [B,1,H,W].")
        if hard_mask.shape[0] != content_mask.shape[0]:
            raise ValueError("hard_mask and content_mask batch sizes must match.")
        if hard_mask.shape[-2:] != content_mask.shape[-2:]:
            raise ValueError("hard_mask and content_mask spatial sizes must match.")
        if len(valid_mask) != hard_mask.shape[0]:
            raise ValueError("valid_mask length must match the image batch size.")

        kernel = int(SETTINGS_PREPROCESSING.HARD_SUPPORT_DILATION_KERNEL)
        binary_support = (hard_mask > 0.5).to(content_mask.dtype)
        binary_support = F.max_pool2d(
            binary_support,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )

        batch_size, _, height, width = hard_mask.shape
        fallback = _fixed_central_square_mask(
            batch_size,
            height,
            width,
            SETTINGS_PREPROCESSING.HARD_SUPPORT_FALLBACK_FRACTION,
            hard_mask.device,
            content_mask.dtype,
        )
        valid = valid_mask.to(device=hard_mask.device, dtype=torch.bool).view(
            -1, 1, 1, 1
        )
        nonempty = torch.any(binary_support > 0.5, dim=(1, 2, 3)).view(
            -1, 1, 1, 1
        )
        support = torch.where(valid & nonempty, binary_support, fallback)
        support = (support > 0.5).to(content_mask.dtype)
        support = support * (content_mask > 0.5).to(content_mask.dtype)
        return support.clamp(0.0, 1.0)

    @staticmethod
    def create_hard_support_region_normalized_images(
        raw_images,
        hard_mask,
        valid_mask,
        content_mask,
    ):
        """Create A17 from the exact binary support without soft confidence."""

        visible = create_a17_exact_support_mask(
            hard_mask,
            valid_mask,
            content_mask,
        )
        return _robust_scale_visible_regions(raw_images, visible)

    @staticmethod
    def create_a17_exact_support_mask_only_images(
        hard_mask,
        valid_mask,
        content_mask,
    ):
        """Create C31: the exact A17 support with every MRI intensity removed."""

        support = create_a17_exact_support_mask(
            hard_mask,
            valid_mask,
            content_mask,
        )
        return support.repeat(1, 3, 1, 1)

    @staticmethod
    def _affine_permutation_parameters(decoded_pixel_hash, n_values):
        """Return deterministic coprime ``a`` and offset ``b`` for C32."""

        n_values = int(n_values)
        if n_values <= 1:
            return 1, 0

        digest = hashlib.sha256(
            (
                str(SETTINGS_PREPROCESSING.SUPPORT_INTENSITY_SHUFFLE_SCHEMA)
                + ":"
                + str(decoded_pixel_hash)
            ).encode("utf-8")
        ).digest()
        candidate = int.from_bytes(digest[:8], byteorder="big", signed=False)
        offset = int.from_bytes(digest[8:16], byteorder="big", signed=False)

        multiplier = candidate % n_values
        if multiplier == 0:
            multiplier = 1

        # Search cyclically for a coprime multiplier. When n>2, avoid a=1 so the
        # mapping is not merely a cyclic shift that preserves most local adjacency.
        # For n=2 the only coprime multiplier is one, so a non-zero offset is used.
        selected = None
        for _ in range(n_values):
            if math.gcd(multiplier, n_values) == 1 and (
                multiplier != 1 or n_values <= 2
            ):
                selected = multiplier
                break
            multiplier = (multiplier + 1) % n_values
            if multiplier == 0:
                multiplier = 1
        if selected is None:
            selected = 1

        offset = int(offset % n_values)
        if selected == 1 and offset == 0:
            offset = 1
        return int(selected), offset

    @staticmethod
    def create_a17_support_intensity_affine_shuffled_images(
        raw_images,
        hard_mask,
        valid_mask,
        content_mask,
        decoded_pixel_hashes,
    ):
        """Create C32 by shuffling A17 intensities inside the exact same support.

        The A17 region-normalized grayscale values are permuted bijectively among
        the visible support pixels. C32 therefore preserves support geometry and the
        complete within-support histogram while destroying the original spatial
        assignment of intensity values. The permutation is label-blind and derived
        only from the exact decoded-pixel hash.
        """

        if len(decoded_pixel_hashes) != raw_images.shape[0]:
            raise ValueError(
                "decoded_pixel_hashes length must match the image batch size."
            )

        support = create_a17_exact_support_mask(
            hard_mask,
            valid_mask,
            content_mask,
        )
        scaled = _robust_scale_visible_regions(raw_images, support)
        output_gray = torch.zeros_like(scaled[:, 0:1])

        for index, decoded_hash in enumerate(decoded_pixel_hashes):
            flat_mask = support[index, 0].reshape(-1) > 0.5
            n_visible = int(torch.sum(flat_mask).item())
            if n_visible == 0:
                continue

            source_values = scaled[index, 0].reshape(-1)[flat_mask]
            multiplier, offset = _affine_permutation_parameters(
                decoded_hash,
                n_visible,
            )
            positions = torch.arange(
                n_visible,
                device=source_values.device,
                dtype=torch.long,
            )
            permutation = (multiplier * positions + offset) % n_visible
            shuffled = source_values[permutation]
            output_flat = output_gray[index, 0].reshape(-1)
            output_flat[flat_mask] = shuffled

        return output_gray.repeat(1, 3, 1, 1)

    @staticmethod
    def create_a17_exact_support_complement_region_normalized_images(
        raw_images,
        hard_mask,
        valid_mask,
        content_mask,
    ):
        """Create C33 from the exact non-padding complement of the A17 support."""

        support = create_a17_exact_support_mask(
            hard_mask,
            valid_mask,
            content_mask,
        )
        visible = (
            (content_mask > 0.5).to(raw_images.dtype)
            - (support > 0.5).to(raw_images.dtype)
        ).clamp(0.0, 1.0)
        return _robust_scale_visible_regions(raw_images, visible)

    @staticmethod
    def run_exact_support_transform_contract_self_test():
        """Fail fast if the exact A17/C31/C32/C33 pixel contract is violated.

        The test uses small CPU tensors and no labels, files, neural networks or
        random number generators. It verifies exact support identity, histogram
        preservation, support/complement separation, deterministic shuffling and
        invariance to pixels excluded from each independently normalized branch.
        """

        height = width = 32
        base = torch.arange(
            2 * height * width,
            dtype=torch.float32,
        ).reshape(2, 1, height, width)
        base = (base % 251.0) / 250.0
        raw = base.repeat(1, 3, 1, 1)

        hard = torch.zeros(2, 1, height, width, dtype=torch.float32)
        hard[0, 0, 11:20, 12:21] = 1.0
        hard[1, 0, 7:13, 8:14] = 1.0
        valid = torch.tensor([True, False], dtype=torch.bool)
        content = torch.ones(2, 1, height, width, dtype=torch.float32)
        content[:, :, :2, :] = 0.0
        content[:, :, -2:, :] = 0.0
        content[:, :, :, :2] = 0.0
        content[:, :, :, -2:] = 0.0
        decoded_hashes = ("a" * 64, "b" * 64)

        support = create_a17_exact_support_mask(hard, valid, content)
        a17 = create_hard_support_region_normalized_images(
            raw,
            hard,
            valid,
            content,
        )
        c31 = create_a17_exact_support_mask_only_images(hard, valid, content)
        c32 = create_a17_support_intensity_affine_shuffled_images(
            raw,
            hard,
            valid,
            content,
            decoded_hashes,
        )
        c33 = create_a17_exact_support_complement_region_normalized_images(
            raw,
            hard,
            valid,
            content,
        )

        if not torch.equal(c31[:, 0:1], support):
            raise RuntimeError("self-test failed: C31 support differs from A17.")
        if torch.any(a17 * (1.0 - support) != 0.0):
            raise RuntimeError("self-test failed: A17 is non-zero outside support.")
        if torch.any(c32 * (1.0 - support) != 0.0):
            raise RuntimeError("self-test failed: C32 is non-zero outside support.")
        if torch.any(c33 * support != 0.0):
            raise RuntimeError("self-test failed: C33 overlaps A17 support.")

        for index in range(raw.shape[0]):
            mask = support[index, 0] > 0.5
            a17_values = torch.sort(a17[index, 0][mask]).values
            c32_values = torch.sort(c32[index, 0][mask]).values
            if not torch.equal(a17_values, c32_values):
                raise RuntimeError(
                    "self-test failed: C32 does not preserve the exact A17 "
                    "within-support intensity multiset."
                )
            if int(mask.sum().item()) > 1 and torch.equal(
                a17[index, 0][mask],
                c32[index, 0][mask],
            ):
                raise RuntimeError(
                    "self-test failed: C32 spatial assignment was unchanged."
                )

        c32_repeat = create_a17_support_intensity_affine_shuffled_images(
            raw,
            hard,
            valid,
            content,
            decoded_hashes,
        )
        if not torch.equal(c32, c32_repeat):
            raise RuntimeError("self-test failed: C32 is not deterministic.")

        changed_outside = raw.clone()
        changed_outside[(1.0 - support).repeat(1, 3, 1, 1) > 0.5] = 0.987
        if not torch.equal(
            a17,
            create_hard_support_region_normalized_images(
                changed_outside,
                hard,
                valid,
                content,
            ),
        ):
            raise RuntimeError(
                "self-test failed: excluded complement pixels affect A17."
            )

        changed_inside = raw.clone()
        changed_inside[support.repeat(1, 3, 1, 1) > 0.5] = 0.123
        if not torch.equal(
            c33,
            create_a17_exact_support_complement_region_normalized_images(
                changed_inside,
                hard,
                valid,
                content,
            ),
        ):
            raise RuntimeError(
                "self-test failed: excluded A17 pixels affect C33."
            )

        complement = (
            (content > 0.5).to(support.dtype) - support
        ).clamp(0.0, 1.0)
        if not torch.equal(
            (support + complement) > 0.5,
            content > 0.5,
        ):
            raise RuntimeError(
                "self-test failed: support and complement do not partition content."
            )

        # Check that the affine mapping is a genuine bijection for representative
        # small and realistic support sizes rather than an accidental identity map.
        for n_values in (2, 3, 17, 64, 997, 10_000):
            multiplier, offset = _affine_permutation_parameters(
                "self-test",
                n_values,
            )
            mapped = {
                (multiplier * index + offset) % n_values
                for index in range(n_values)
            }
            if len(mapped) != n_values:
                raise RuntimeError("self-test failed: C32 map is not bijective.")
            if all(
                (multiplier * index + offset) % n_values == index
                for index in range(n_values)
            ):
                raise RuntimeError("self-test failed: C32 map is the identity.")

    @staticmethod
    def _fixed_central_square_mask(batch_size, height, width, fraction, device, dtype):
        """Create one MONAI-independent central square mask for an entire batch."""

        top, bottom, left, right = _fixed_square_bounds(
            height,
            width,
            (height - 1) / 2.0,
            (width - 1) / 2.0,
            fraction,
        )
        mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
        mask[:, :, top:bottom, left:right] = 1.0
        return mask

    @staticmethod
    def create_fixed_periphery_region_normalized_images(raw_images, content_mask):
        """Retain only a fixed MONAI-independent periphery with local scaling."""

        batch_size, _, height, width = raw_images.shape
        exclusion = _fixed_central_square_mask(
            batch_size,
            height,
            width,
            SETTINGS_PREPROCESSING.FIXED_PERIPHERY_EXCLUSION_FRACTION,
            raw_images.device,
            raw_images.dtype,
        )
        visible = (1.0 - exclusion) * content_mask
        return _robust_scale_visible_regions(raw_images, visible)

    @staticmethod
    def extract_feature_bank(
        dataset,
        monai_segmenter,
        feature_extractor,
        required_modes,
        cache_dir,
        fingerprint,
    ):
        """Build the aligned standardized MONAI/EfficientNet feature bank.

        Every JPEG is decoded once. MONAI runs once on the standardized 256x256
        canvas, and the six active candidate/control views are encoded by frozen
        EfficientNet-B0. Large embedding arrays are written incrementally as NumPy
        memory maps; metadata.json is the final cache-completion marker.
        """

        allowed_modes = set(required_efficientnet_feature_modes(SETTINGS_EXPERIMENTS.EXPERIMENT_REGISTRY))
        unknown = set(required_modes) - allowed_modes
        if not required_modes or unknown:
            raise ValueError(f"Invalid feature modes: {sorted(unknown)}")
        if monai_segmenter is None:
            raise RuntimeError("The active feature panel requires the MONAI segmenter.")

        if cache_dir.exists():
            print(f"[FEATURE BANK] Removing incomplete/rebuilt cache: {cache_dir}", flush=True)
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
                shape=(n_slices, SETTINGS_FEATURE_BANK.EFFICIENTNET_FEATURE_DIM),
            )
            for mode in required_modes
        }
        labels_array = np.empty(n_slices, dtype=np.int64)
        sample_indices_array = np.empty(n_slices, dtype=np.int64)
        provenance_array = np.empty(
            (n_slices, len(SETTINGS_REPORTING.PROVENANCE_FEATURE_NAMES)), dtype=np.float32
        )
        standardization_array = np.empty(
            (n_slices, len(SETTINGS_PREPROCESSING.STANDARDIZATION_FEATURE_NAMES)), dtype=np.float32
        )
        monai_valid_array = np.zeros(n_slices, dtype=bool)
        area_ratio_array = np.full(n_slices, np.nan, dtype=np.float32)
        peak_probability_array = np.full(n_slices, np.nan, dtype=np.float32)
        foreground_probability_array = np.full(n_slices, np.nan, dtype=np.float32)
        patient_ids_values = [None] * n_slices
        series_ids_values = [None] * n_slices
        decoded_hash_values = [None] * n_slices
        perceptual_hash_values = [None] * n_slices

        loader = DataLoader(
            dataset,
            batch_size=SETTINGS_RUNTIME.BATCH_SIZE,
            shuffle=False,
            num_workers=SETTINGS_RUNTIME.DATALOADER_NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
            persistent_workers=(SETTINGS_RUNTIME.DATALOADER_NUM_WORKERS > 0),
        )
        total_batches = len(loader)
        progress_interval = max(1, SETTINGS_RUNTIME.PROGRESS_PRINT_EVERY_N_BATCHES, total_batches // 20)
        autocast_enabled = SETTINGS_RUNTIME.USE_CUDA_AMP and DEVICE == "cuda"
        started = time.perf_counter()
        processed = 0

        print(
            f"[FEATURE BANK] slices={n_slices}, batches={total_batches}, "
            f"modes={list(required_modes)}, device={DEVICE}",
            flush=True,
        )

        with torch.inference_mode():
            for batch_index, batch in enumerate(
                tqdm(loader, desc="Feature bank", unit="batch", dynamic_ncols=True, file=sys.stdout),
                start=1,
            ):
                batch_started = time.perf_counter()
                (
                    standardized_images,
                    standardized_monai_images,
                    standardized_raw_images,
                    padding_images,
                    labels,
                    patient_ids,
                    series_ids,
                    sample_indices,
                    decoded_pixel_hashes,
                    perceptual_hashes,
                    provenance_features,
                    standardization_features,
                ) = batch

                indices = sample_indices.detach().cpu().numpy().astype(np.int64)
                standardized_images = standardized_images.to(DEVICE, non_blocking=True)
                standardized_monai_images = standardized_monai_images.to(DEVICE, non_blocking=True)
                standardized_raw_images = standardized_raw_images.to(DEVICE, non_blocking=True)
                padding_images = padding_images.to(DEVICE, non_blocking=True)

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=autocast_enabled
                ):
                    (
                        roi_probability,
                        hard_mask,
                        valid_mask,
                        area_ratio,
                        peak_probability,
                        mean_foreground_probability,
                    ) = predict_monai_heart_masks(
                        standardized_monai_images,
                        classifier_size=standardized_images.shape[-2:],
                        monai_segmenter=monai_segmenter,
                    )

                content_mask = (1.0 - padding_images).clamp(0.0, 1.0)
                variants = {}
                if "standardized_roi_zero_bg_center_fallback" in required_modes:
                    variants["standardized_roi_zero_bg_center_fallback"] = (
                        apply_zero_background_roi_with_fixed_center_fallback(
                            standardized_images,
                            roi_probability,
                            valid_mask,
                            fallback_fraction=SETTINGS_PREPROCESSING.CENTER_CROP_FALLBACK_FRACTION,
                        )
                    )
                if "standardized_hard_support_region_norm" in required_modes:
                    variants["standardized_hard_support_region_norm"] = (
                        create_hard_support_region_normalized_images(
                            standardized_raw_images, hard_mask, valid_mask, content_mask
                        )
                    )
                if "standardized_fixed_periphery_region_norm" in required_modes:
                    variants["standardized_fixed_periphery_region_norm"] = (
                        create_fixed_periphery_region_normalized_images(
                            standardized_raw_images, content_mask
                        )
                    )
                if "standardized_a17_exact_support_mask_only" in required_modes:
                    variants["standardized_a17_exact_support_mask_only"] = (
                        create_a17_exact_support_mask_only_images(
                            hard_mask, valid_mask, content_mask
                        )
                    )
                if "standardized_a17_support_intensity_affine_shuffled" in required_modes:
                    variants["standardized_a17_support_intensity_affine_shuffled"] = (
                        create_a17_support_intensity_affine_shuffled_images(
                            standardized_raw_images,
                            hard_mask,
                            valid_mask,
                            content_mask,
                            decoded_pixel_hashes,
                        )
                    )
                if "standardized_a17_exact_support_complement_region_norm" in required_modes:
                    variants["standardized_a17_exact_support_complement_region_norm"] = (
                        create_a17_exact_support_complement_region_normalized_images(
                            standardized_raw_images, hard_mask, valid_mask, content_mask
                        )
                    )

                mode_names = list(required_modes)
                for offset in range(0, len(mode_names), SETTINGS_FEATURE_BANK.FEATURE_MODES_PER_ENCODER_CALL):
                    chunk_modes = mode_names[offset:offset + SETTINGS_FEATURE_BANK.FEATURE_MODES_PER_ENCODER_CALL]
                    concatenated = torch.cat([variants[mode] for mode in chunk_modes], dim=0)
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=autocast_enabled
                    ):
                        encoded = feature_extractor(
                            normalize_for_efficientnet(concatenated)
                        ).float()
                    expected_rows = len(indices) * len(chunk_modes)
                    if encoded.shape != (expected_rows, SETTINGS_FEATURE_BANK.EFFICIENTNET_FEATURE_DIM):
                        raise RuntimeError(
                            f"Unexpected EfficientNet shape for {chunk_modes}: {tuple(encoded.shape)}"
                        )
                    for mode, values in zip(chunk_modes, torch.split(encoded, len(indices), dim=0)):
                        feature_maps[mode][indices, :] = values.detach().cpu().numpy()
                    del concatenated, encoded

                labels_array[indices] = labels.detach().cpu().numpy()
                sample_indices_array[indices] = indices
                provenance_array[indices, :] = provenance_features.detach().cpu().numpy()
                standardization_array[indices, :] = standardization_features.detach().cpu().numpy()
                monai_valid_array[indices] = valid_mask.detach().cpu().numpy()
                area_ratio_array[indices] = area_ratio.detach().cpu().numpy()
                peak_probability_array[indices] = peak_probability.detach().cpu().numpy()
                foreground_probability_array[indices] = (
                    mean_foreground_probability.detach().cpu().numpy()
                )

                for local, global_index in enumerate(indices.tolist()):
                    patient_ids_values[global_index] = str(patient_ids[local])
                    series_ids_values[global_index] = str(series_ids[local])
                    decoded_hash_values[global_index] = str(decoded_pixel_hashes[local])
                    perceptual_hash_values[global_index] = str(perceptual_hashes[local])

                processed += len(indices)
                if batch_index == 1 or batch_index % progress_interval == 0 or batch_index == total_batches:
                    _synchronize_timing_device()
                    elapsed = time.perf_counter() - started
                    rate = processed / max(elapsed, 1e-8)
                    eta = (n_slices - processed) / max(rate, 1e-8)
                    print(
                        f"[FEATURE BANK] {processed}/{n_slices} "
                        f"({100.0 * processed / n_slices:.1f}%) | "
                        f"rate={rate:.2f} slices/s | ETA={_format_elapsed_time(eta)} | "
                        f"batch={_format_elapsed_time(time.perf_counter() - batch_started)}",
                        flush=True,
                    )

        if any(value is None for value in patient_ids_values + decoded_hash_values):
            raise RuntimeError("Feature-bank metadata rows are incomplete.")
        if not np.array_equal(sample_indices_array, np.arange(n_slices)):
            raise RuntimeError("Feature-bank sample indices are incomplete or reordered.")

        for values in feature_maps.values():
            values.flush()
        del feature_maps

        shared_arrays = {
            "labels": labels_array,
            "patient_ids": np.asarray(patient_ids_values),
            "series_ids": np.asarray(series_ids_values),
            "sample_indices": sample_indices_array,
            "decoded_pixel_hashes": np.asarray(decoded_hash_values),
            "perceptual_hashes": np.asarray(perceptual_hash_values),
            "provenance_features": provenance_array,
            "standardization_features": standardization_array,
            "standardized_monai_valid": monai_valid_array,
            "standardized_area_ratios": area_ratio_array,
            "standardized_peak_probabilities": peak_probability_array,
            "standardized_mean_foreground_probabilities": foreground_probability_array,
        }
        for name, array in shared_arrays.items():
            np.save(_feature_bank_shared_paths(cache_dir)[name], np.asarray(array), allow_pickle=False)

        metadata = {
            "fingerprint": fingerprint,
            "schema": SETTINGS_FEATURE_BANK.FEATURE_CACHE_SCHEMA,
            "completed_modes": list(required_modes),
            "n_slices": int(n_slices),
            "feature_dimension": SETTINGS_FEATURE_BANK.EFFICIENTNET_FEATURE_DIM,
            "provenance_feature_names": list(SETTINGS_REPORTING.PROVENANCE_FEATURE_NAMES),
            "standardization_feature_names": list(SETTINGS_PREPROCESSING.STANDARDIZATION_FEATURE_NAMES),
            "monai_runtime_source": MONAI_RUNTIME_SOURCE,
            "monai_runtime_artifact_path": MONAI_RUNTIME_ARTIFACT_PATH,
        }
        (cache_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        _synchronize_timing_device()
        print(
            "[FEATURE BANK] Extraction and cache write completed in "
            f"{_format_elapsed_time(time.perf_counter() - started)}.",
            flush=True,
        )
        loaded = load_feature_bank(cache_dir, fingerprint, required_modes)
        if loaded is None:
            raise RuntimeError("Freshly written feature bank could not be reloaded.")
        return loaded

    @staticmethod
    def load_or_extract_feature_bank(samples, required_modes, fingerprint, cache_dir):
        """Load a matching bank or perform one fresh multi-view extraction."""

        if SETTINGS_FEATURE_BANK.USE_FEATURE_CACHE and not SETTINGS_FEATURE_BANK.FORCE_REBUILD_FEATURE_CACHE:
            bank = load_feature_bank(cache_dir, fingerprint, required_modes)
            if bank is not None:
                return bank, "HIT"

        if SETTINGS_FEATURE_BANK.REQUIRE_EXISTING_FEATURE_CACHE:
            raise RuntimeError(
                "A matching frozen feature bank was not found, and this action is "
                "cache-only. Run build-monai-feature-cache in a GPU session first. "
                f"Expected cache directory: {cache_dir}"
            )

        print(
            "[FEATURE BANK] Cache miss or forced rebuild; neural-network inference "
            "will run now.",
            flush=True,
        )
        dataset = MRIDataset(samples, transform)

        # Keep this set synchronized with every representation that consumes a
        # MONAI probability map, hard mask, gate decision, or bounding box. This is
        # Every active image representation depends on standardized MONAI output.
        monai_dependent_modes = {
            'standardized_roi_zero_bg_center_fallback',
            'standardized_hard_support_region_norm',
            'standardized_fixed_periphery_region_norm',
            'standardized_a17_exact_support_mask_only',
            'standardized_a17_support_intensity_affine_shuffled',
            'standardized_a17_exact_support_complement_region_norm',
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
            if SETTINGS_FEATURE_BANK.USE_FEATURE_CACHE
            else SETTINGS_PATHS.OUTPUT_DIR / "_runtime_feature_bank"
        )
        bank = extract_feature_bank(
            dataset=dataset,
            monai_segmenter=monai_segmenter,
            feature_extractor=feature_extractor,
            required_modes=required_modes,
            cache_dir=runtime_cache_dir,
            fingerprint=fingerprint,
        )

        del feature_extractor
        del monai_segmenter
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return bank, "MISS"


class DuplicateAuditManager:
    """Exact/perceptual duplicate audit and duplicate-aware patient fold manifests."""

    @staticmethod
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

        if cross_patient_groups and SETTINGS_AUDIT.FAIL_ON_CROSS_PATIENT_EXACT_DUPLICATES:
            raise RuntimeError(
                f"Cross-patient exact duplicates found; inspect {output_path}."
            )
        if cross_label_groups and SETTINGS_AUDIT.FAIL_ON_CROSS_LABEL_EXACT_DUPLICATES:
            raise RuntimeError(
                f"Cross-label exact duplicates found; inspect {output_path}."
            )

        return summary, patient_edges

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                SETTINGS_AUDIT.PHASH_HAMMING_THRESHOLD,
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
                f"cross-patient pairs with Hamming distance <= {SETTINGS_AUDIT.PHASH_HAMMING_THRESHOLD}"
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

    @staticmethod
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

    @staticmethod
    def build_duplicate_component_map(patient_ids, exact_edges, phash_edges):
        """Create deterministic patient components used as split groups."""

        patient_ids = [str(value) for value in patient_ids]
        union_find = UnionFind(patient_ids)

        if SETTINGS_AUDIT.GROUP_SPLITS_BY_EXACT_DUPLICATES:
            for first, second in exact_edges:
                union_find.union(first, second)

        if SETTINGS_AUDIT.GROUP_SPLITS_BY_PHASH_CANDIDATES:
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

    @staticmethod
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

    @staticmethod
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
            SETTINGS_VALIDATION.N_SPLITS,
            SETTINGS_VALIDATION.CV_RANDOM_STATE,
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

        for fold_index in range(1, SETTINGS_VALIDATION.N_SPLITS + 1):
            fold_rows = [row for row in rows if row["outer_fold"] == fold_index]
            print(
                f"[FOLD MANIFEST] fold={fold_index}: patients={len(fold_rows)}, "
                f"Normal={sum(row['true_label'] == 0 for row in fold_rows)}, "
                f"Sick={sum(row['true_label'] == 1 for row in fold_rows)}, "
                f"components={len(set(row['duplicate_component_id'] for row in fold_rows))}",
                flush=True,
            )

        return rows

    @staticmethod
    def write_patient_fold_manifest(output_path, rows):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def write_cohort_manifest(output_path, samples, bank, fold_manifest_rows):
        """Write one auditable row per image with patient folds and QC metadata."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fold_by_patient = {row["patient_id"]: row["outer_fold"] for row in fold_manifest_rows}
        component_by_patient = {
            row["patient_id"]: row["duplicate_component_id"] for row in fold_manifest_rows
        }
        fieldnames = [
            "sample_index", "image_path", "true_label", "patient_id", "series_id",
            "outer_fold", "duplicate_component_id", "decoded_pixel_sha256",
            "perceptual_hash", "monai_gate_valid", "monai_area_ratio",
            "monai_peak_probability", "monai_mean_foreground_probability",
            *SETTINGS_REPORTING.PROVENANCE_FEATURE_NAMES, *SETTINGS_PREPROCESSING.STANDARDIZATION_FEATURE_NAMES,
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
                    "monai_gate_valid": int(bool(bank["standardized_monai_valid"][index])),
                    "monai_area_ratio": float(bank["standardized_area_ratios"][index]),
                    "monai_peak_probability": float(bank["standardized_peak_probabilities"][index]),
                    "monai_mean_foreground_probability": float(
                        bank["standardized_mean_foreground_probabilities"][index]
                    ),
                }
                row.update({
                    name: float(value)
                    for name, value in zip(SETTINGS_REPORTING.PROVENANCE_FEATURE_NAMES, bank["provenance_features"][index])
                })
                row.update({
                    name: float(value)
                    for name, value in zip(SETTINGS_PREPROCESSING.STANDARDIZATION_FEATURE_NAMES, bank["standardization_features"][index])
                })
                writer.writerow(row)

    @staticmethod
    def _patient_indices(patient_ids):
        mapping = defaultdict(list)
        for index, patient_id in enumerate(patient_ids):
            mapping[str(patient_id)].append(index)
        return mapping


class PatientDataManager:
    """Patient-level tabular controls, embedding pooling and experiment preparation."""

    @staticmethod
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
                SETTINGS_REPORTING.PROVENANCE_FEATURE_NAMES.index(feature_name)
                for feature_name in SETTINGS_REPORTING.PROVENANCE_CLASSIFIER_FEATURE_NAMES
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
        for feature_name in SETTINGS_REPORTING.PROVENANCE_CLASSIFIER_FEATURE_NAMES:
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

    @staticmethod
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
        for feature_name in SETTINGS_PREPROCESSING.STANDARDIZATION_FEATURE_NAMES:
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

    @staticmethod
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

    @staticmethod
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

        alpha = 1.0 - SETTINGS_VALIDATION.BOOTSTRAP_CONFIDENCE
        return (
            float(np.quantile(differences, alpha / 2.0)),
            float(np.quantile(differences, 1.0 - alpha / 2.0)),
        )

    @staticmethod
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
            SETTINGS_VALIDATION.BOOTSTRAP_REPLICATES,
            SETTINGS_RUNTIME.RANDOM_SEED + 1701,
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
                SETTINGS_AUDIT.MONAI_GATE_RATE_DIFFERENCE_WARNING
            ),
            "warning_triggered": bool(
                abs(difference) >= SETTINGS_AUDIT.MONAI_GATE_RATE_DIFFERENCE_WARNING
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

    @staticmethod
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

    @staticmethod
    def aggregate_patient_embeddings(
        features,
        labels,
        patient_ids,
        series_ids,
        slice_weights=None,
        pooling_strategy="hierarchical",
        sample_order_keys=None,
        series_to_pooling_cell=None,
    ):
        """Pool slice embeddings to series means and then one mean per patient."""

        if pooling_strategy != "hierarchical":
            raise ValueError("The final pipeline supports hierarchical pooling only.")
        if series_to_pooling_cell is not None:
            raise ValueError("Sequence/view-balanced pooling is not part of this pipeline.")

        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        patient_ids = np.asarray(patient_ids).astype(str)
        series_ids = np.asarray(series_ids).astype(str)
        if slice_weights is None:
            slice_weights = np.ones(len(labels), dtype=np.float64)
        slice_weights = np.asarray(slice_weights, dtype=np.float64)
        if not (len(features) == len(labels) == len(patient_ids) == len(series_ids) == len(slice_weights)):
            raise ValueError("Pooling arrays must have equal lengths.")

        patient_to_label = {}
        grouped = defaultdict(lambda: defaultdict(list))
        for index, (label, patient_id, series_id) in enumerate(zip(labels, patient_ids, series_ids)):
            previous = patient_to_label.get(patient_id)
            if previous is not None and previous != int(label):
                raise RuntimeError(f"Patient {patient_id} has inconsistent labels.")
            patient_to_label[patient_id] = int(label)
            grouped[patient_id][series_id].append(index)

        pooled, pooled_labels, ordered_patients = [], [], sorted(grouped)
        for patient_id in ordered_patients:
            series_vectors = []
            for series_id in sorted(grouped[patient_id]):
                indices = np.asarray(grouped[patient_id][series_id], dtype=np.int64)
                weights = slice_weights[indices]
                if not np.any(weights > 0):
                    raise RuntimeError("A series proxy has no positive slice weight.")
                series_vectors.append(np.average(features[indices], axis=0, weights=weights))
            pooled.append(np.mean(np.stack(series_vectors, axis=0), axis=0).astype(np.float32))
            pooled_labels.append(patient_to_label[patient_id])

        X = np.stack(pooled, axis=0)
        if not np.all(np.isfinite(X)):
            raise RuntimeError("Pooled patient embeddings contain non-finite values.")
        return X, np.asarray(pooled_labels, dtype=np.int64), np.asarray(ordered_patients)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def prepare_experiment_data(
        experiment,
        bank,
        row_mask=None,
        series_to_pooling_cell=None,
    ):
        """Prepare one leakage-free patient embedding table for an experiment."""

        if row_mask is not None or series_to_pooling_cell is not None:
            raise ValueError("Annotated sequence/view subsets are not part of the final pipeline.")
        if experiment.strategy != "patient_embedding":
            raise ValueError("The final pipeline supports patient embeddings only.")
        if experiment.pooling_strategy != "hierarchical":
            raise ValueError("The final pipeline supports hierarchical pooling only.")
        if experiment.weighting_mode != "equal":
            raise ValueError("The final pipeline uses equal slice weights only.")
        if experiment.classifier_type != "logistic_regression":
            raise ValueError("The final pipeline uses Logistic Regression only.")

        features = np.asarray(bank["features"][experiment.feature_mode], dtype=np.float32)
        labels = np.asarray(bank["labels"], dtype=np.int64)
        patient_ids = np.asarray(bank["patient_ids"]).astype(str)
        series_ids = np.asarray(bank["series_ids"]).astype(str)
        n_source = int(len(labels))

        mask = np.ones(n_source, dtype=bool)
        if experiment.slice_filter == "standardized_monai_valid":
            mask &= np.asarray(bank["standardized_monai_valid"], dtype=bool)
        elif experiment.slice_filter != "all":
            raise ValueError(f"Unsupported slice filter: {experiment.slice_filter!r}.")

        expected_patients = set(patient_ids.tolist())
        retained_patients = set(patient_ids[mask].tolist())
        if retained_patients != expected_patients:
            raise RuntimeError(
                f"{experiment.experiment_id}: filtering removed all rows for "
                f"{sorted(expected_patients - retained_patients)}."
            )

        features = features[mask]
        labels = labels[mask]
        patient_ids = patient_ids[mask]
        series_ids = series_ids[mask]
        X, y, patients = aggregate_patient_embeddings(
            features,
            labels,
            patient_ids,
            series_ids,
            slice_weights=np.ones(len(labels), dtype=np.float64),
        )
        return {
            "unit": "patient",
            "X": X,
            "y": y,
            "patient_ids": patients,
            "feature_names": tuple(f"embedding_{i:04d}" for i in range(X.shape[1])),
            "slice_dropout_rate": 0.0,
            "slice_filter": str(experiment.slice_filter),
            "deduplicate_exact_within_patient": False,
            "n_exact_duplicate_rows_removed": 0,
            "n_source_slices": n_source,
            "n_after_slice_filter": int(mask.sum()),
            "n_retained_slices": int(mask.sum()),
            "annotated_series_subset_applied": False,
            "sequence_view_balanced_pooling": False,
        }


class ModelingManager:
    """Fold-local classifiers, nested CV, stability, permutation and paired tests."""

    @staticmethod
    def build_patient_classifier(experiment, c_value):
        """Create the fixed scaler/PCA/Logistic-Regression patient model."""

        if experiment.classifier_type != "logistic_regression":
            raise ValueError("Only Logistic Regression is supported.")
        steps = [("scaler", StandardScaler())]
        if experiment.use_pca:
            steps.append(("pca", PCA(n_components=SETTINGS_VALIDATION.PATIENT_PCA_EXPLAINED_VARIANCE, svd_solver="full")))
        steps.append(("classifier", LogisticRegression(
            C=float(c_value), max_iter=SETTINGS_VALIDATION.LOGISTIC_MAX_ITER, solver="liblinear",
            random_state=SETTINGS_RUNTIME.RANDOM_SEED,
        )))
        return Pipeline(steps=steps)

    @staticmethod
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

    @staticmethod
    def patient_model_raw_scores(model, experiment, X_valid):
        """Return held-out Logistic-Regression probabilities."""

        return np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)

    @staticmethod
    def fit_predict_raw_for_patient_sets(
        experiment,
        prepared,
        train_patients,
        valid_patients,
        c_value,
    ):
        """Fit one fold-local patient model and score held-out patients."""

        train_patients = set(map(str, train_patients))
        valid_patients = set(map(str, valid_patients))
        overlap = train_patients.intersection(valid_patients)
        if overlap:
            raise RuntimeError(f"Train/validation patient overlap: {sorted(overlap)}")
        patient_ids = np.asarray(prepared["patient_ids"]).astype(str)
        train_mask = np.isin(patient_ids, list(train_patients))
        valid_mask = np.isin(patient_ids, list(valid_patients))
        if not np.any(train_mask) or not np.any(valid_mask):
            raise RuntimeError("A patient-level fold side is empty.")
        model = build_patient_classifier(experiment, c_value)
        fit_patient_classifier(model, prepared["X"][train_mask], prepared["y"][train_mask])
        scores = patient_model_raw_scores(model, experiment, prepared["X"][valid_mask])
        return scores, np.asarray(prepared["y"][valid_mask], dtype=np.int64), patient_ids[valid_mask]

    @staticmethod
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
            SETTINGS_VALIDATION.INNER_CV_SPLITS,
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
                    SETTINGS_VALIDATION.INNER_CV_RANDOM_STATE + 100 * outer_fold_index,
                )
                return folds, n_splits
            except RuntimeError as error:
                last_error = error

        raise RuntimeError(
            "Could not create at least two valid inner folds for training-only "
            f"selection. Last error: {last_error}"
        )

    @staticmethod
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

    @staticmethod
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

        if SETTINGS_VALIDATION.THRESHOLD_SELECTION_METHOD == "youden":
            statistic = true_positive_rate - false_positive_rate
            best_value = float(np.max(statistic))
            candidates = np.flatnonzero(
                np.isclose(statistic, best_value, rtol=0.0, atol=1e-12)
            )
            # A larger threshold is the conservative deterministic tie-break.
            return float(np.max(thresholds[candidates]))

        eligible = np.flatnonzero(true_positive_rate >= SETTINGS_VALIDATION.TARGET_SENSITIVITY)
        if len(eligible) == 0:
            best_index = int(np.argmax(true_positive_rate))
            return float(thresholds[best_index])

        # Among thresholds meeting the sensitivity target, minimize FPR; use the
        # largest threshold as a deterministic secondary tie-break.
        eligible_fpr = false_positive_rate[eligible]
        minimum_fpr = float(np.min(eligible_fpr))
        best = eligible[np.isclose(eligible_fpr, minimum_fpr, atol=1e-12)]
        return float(np.max(thresholds[best]))

    @staticmethod
    def experiment_candidate_c_values(experiment):
        if experiment.tune_c:
            return tuple(sorted(set(map(float, SETTINGS_VALIDATION.CLASSIFIER_C_GRID))))
        return (float(experiment.fixed_c),)

    @staticmethod
    def select_c_and_training_threshold(
        experiment,
        prepared,
        outer_train_patient_ids,
        outer_train_labels,
        patient_to_group,
        outer_fold_index,
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
            if result["inner_auc"] >= best_inner_auc - SETTINGS_VALIDATION.C_SELECTION_AUC_TOLERANCE
        ]
        selected = sorted(
            eligible_results,
            key=lambda result: result["c_value"],
        )[0]

        calibrator = None
        inner_probabilities = np.clip(selected["raw_scores"], 0.0, 1.0)

        threshold = select_decision_threshold(
            selected["labels"],
            inner_probabilities,
        )
        return {
            "selected_c": selected["c_value"],
            "inner_auc": selected["inner_auc"],
            "best_candidate_inner_auc": float(best_inner_auc),
            "c_selection_auc_tolerance": float(SETTINGS_VALIDATION.C_SELECTION_AUC_TOLERANCE),
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

    @staticmethod
    def _safe_divide(numerator, denominator):
        if denominator == 0:
            return float("nan")
        return float(numerator / denominator)

    @staticmethod
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

    @staticmethod
    def bootstrap_patient_metric_intervals(
        labels,
        probabilities,
        predictions,
        n_bootstrap=SETTINGS_VALIDATION.BOOTSTRAP_REPLICATES,
        confidence=SETTINGS_VALIDATION.BOOTSTRAP_CONFIDENCE,
        random_state=SETTINGS_RUNTIME.RANDOM_SEED,
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

    @staticmethod
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
            f"fusion={experiment.fusion_method}, "
            f"slice_dropout={experiment.slice_dropout_rate:.0%}, "
            f"slice_filter={experiment.slice_filter}, "
            f"exact_within_patient_dedup="
            f"{experiment.deduplicate_exact_within_patient}",
            flush=True,
        )
        print("=" * 100, flush=True)

        for outer_fold in range(1, SETTINGS_VALIDATION.N_SPLITS + 1):
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
                f"{outer_fold}/{SETTINGS_VALIDATION.N_SPLITS}",
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
                )
            )

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
            random_state=SETTINGS_RUNTIME.RANDOM_SEED
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
                "n_after_slice_filter": prepared.get("n_after_slice_filter"),
                "n_retained_slices": prepared.get("n_retained_slices"),
                "slice_filter": prepared.get("slice_filter", "all"),
                "sequence_view_balanced_pooling": bool(
                    prepared.get("sequence_view_balanced_pooling", False)
                ),
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
                "sequence_view_balanced_pooling": bool(
                    prepared.get("sequence_view_balanced_pooling", False)
                ),
                "annotated_series_subset_applied": bool(
                    prepared.get("annotated_series_subset_applied", False)
                ),
            },
        }

    @staticmethod
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
            SETTINGS_VALIDATION.N_SPLITS,
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

    @staticmethod
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

        for outer_fold in range(1, SETTINGS_VALIDATION.N_SPLITS + 1):
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
                verbose=verbose,
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

    @staticmethod
    def audit_exact_support_prepared_row_contract(prepared_by_id, output_path):
        """Verify the patient and source-row contract for A17/A20/C31-C33.

        Pixel-level equality is guaranteed by the shared construction helper and is
        covered by the deterministic transformation tests. This runtime audit checks
        the downstream data contract after row filtering and pooling: exact controls
        must use all A17 source rows and the same patient/label order, while A20 may
        remove only gate-invalid rows without removing a complete patient.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exact_ids = (
            SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.MASK_ONLY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
            SETTINGS_EXPERIMENTS.COMPLEMENT_CONTROL_EXPERIMENT_ID,
        )
        required_ids = exact_ids + (SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,)
        missing = [
            experiment_id
            for experiment_id in required_ids
            if experiment_id not in prepared_by_id
        ]
        if missing:
            summary = {
                "status": "SKIPPED_MISSING_PREPARED_EXPERIMENT",
                "missing_experiments": missing,
            }
            output_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return summary

        candidate = prepared_by_id[SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID]
        candidate_patients = np.asarray(candidate["patient_ids"]).astype(str)
        candidate_labels = np.asarray(candidate["y"], dtype=np.int64)
        candidate_source = int(candidate["n_source_slices"])
        candidate_retained = int(candidate["n_retained_slices"])
        if candidate.get("unit") != "patient":
            raise RuntimeError("A17 prepared data must contain one row per patient.")

        exact_rows = {}
        for experiment_id in exact_ids:
            prepared = prepared_by_id[experiment_id]
            patients = np.asarray(prepared["patient_ids"]).astype(str)
            labels = np.asarray(prepared["y"], dtype=np.int64)
            source = int(prepared["n_source_slices"])
            retained = int(prepared["n_retained_slices"])
            if prepared.get("unit") != "patient":
                raise RuntimeError(f"{experiment_id} is not patient-level.")
            if not np.array_equal(patients, candidate_patients):
                raise RuntimeError(
                    f"{experiment_id} patient order differs from A17."
                )
            if not np.array_equal(labels, candidate_labels):
                raise RuntimeError(f"{experiment_id} labels differ from A17.")
            if source != candidate_source or retained != candidate_retained:
                raise RuntimeError(
                    f"{experiment_id} row counts differ from A17: "
                    f"source={source}, retained={retained}, "
                    f"A17_source={candidate_source}, "
                    f"A17_retained={candidate_retained}."
                )
            exact_rows[experiment_id] = {
                "n_source_slices": source,
                "n_retained_slices": retained,
                "n_patients": int(len(patients)),
            }

        valid_only = prepared_by_id[SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID]
        valid_patients = np.asarray(valid_only["patient_ids"]).astype(str)
        valid_labels = np.asarray(valid_only["y"], dtype=np.int64)
        valid_source = int(valid_only["n_source_slices"])
        valid_retained = int(valid_only["n_retained_slices"])
        if not np.array_equal(valid_patients, candidate_patients):
            raise RuntimeError("A20 removed at least one complete patient.")
        if not np.array_equal(valid_labels, candidate_labels):
            raise RuntimeError("A20 labels differ from A17.")
        if valid_source != candidate_source:
            raise RuntimeError("A20 source-row count differs from A17 before filtering.")
        if not 0 < valid_retained <= candidate_retained:
            raise RuntimeError("A20 retained-slice count is invalid.")

        summary = {
            "status": "OK",
            "exact_all_slice_experiment_ids": list(exact_ids),
            "exact_all_slice_rows": exact_rows,
            "valid_only_experiment_id": SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,
            "valid_only_n_source_slices": valid_source,
            "valid_only_n_retained_slices": valid_retained,
            "valid_only_retained_fraction": float(
                valid_retained / candidate_retained
            ),
            "n_patients": int(len(candidate_patients)),
            "construction_contract": (
                "A17/C31/C32/C33 use create_a17_exact_support_mask as their "
                "single support definition. C31 removes intensity; C32 permutes "
                "the A17 intensity multiset inside that support; C33 uses its "
                "non-padding complement."
            ),
        }
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            "[EXACT SUPPORT ROW CONTRACT] A17/C31/C32/C33 share "
            f"{candidate_retained} source rows and {len(candidate_patients)} "
            f"patients; A20 retains {valid_retained} rows.",
            flush=True,
        )
        return summary

    @staticmethod
    def run_model_family_nested_selection(
        experiments_by_id,
        prepared_by_id,
        fold_manifest_rows,
        output_dir,
    ):
        """Evaluate a training-only choice among the predeclared candidates.

        For every outer fold, each candidate independently selects its classifier C
        using only that outer-training cohort. The first candidate whose inner AUC
        lies within the predeclared tolerance of the best candidate is chosen using
        ``MODEL_FAMILY_NESTED_SELECTION_IDS`` as the deterministic tie-break
        order. Only that chosen model is then fitted on the complete outer-training
        cohort and evaluated on the untouched outer fold.

        This estimates the complete model-selection procedure rather than reporting
        the best full-cohort OOF result after comparing several representations.
        It remains an internal analysis on the same 30 patients and does not replace
        external validation.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_ids = tuple(SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS)

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

        for candidate_id in candidate_ids:
            if candidate_id not in experiments_by_id:
                raise KeyError(f"Missing candidate experiment: {candidate_id}.")
            if candidate_id not in prepared_by_id:
                raise KeyError(f"Missing prepared data for candidate: {candidate_id}.")
            prepared = prepared_by_id[candidate_id]
            if prepared.get("unit") != "patient":
                raise ValueError(
                    f"{candidate_id} is not a one-row-per-patient representation."
                )
            prepared_patient_ids = np.asarray(
                prepared["patient_ids"]
            ).astype(str)
            prepared_labels = np.asarray(prepared["y"], dtype=np.int64)
            candidate_patients = set(prepared_patient_ids.tolist())
            if candidate_patients != set(all_patients.tolist()):
                missing = sorted(set(all_patients.tolist()) - candidate_patients)
                extra = sorted(candidate_patients - set(all_patients.tolist()))
                raise RuntimeError(
                    f"{candidate_id} patient set differs from the fold manifest; "
                    f"missing={missing}, extra={extra}."
                )
            for patient_id, label in zip(prepared_patient_ids, prepared_labels):
                if int(label) != patient_to_label[str(patient_id)]:
                    raise RuntimeError(
                        f"{candidate_id} label differs from the fold manifest for "
                        f"{patient_id}."
                    )

        score_by_patient = {}
        fold_by_patient = {}
        selected_model_by_patient = {}
        selected_c_by_patient = {}
        fold_candidate_rows = []
        fold_selection_rows = []

        print(
            "[EXACT SUPPORT MODEL SELECTION] Running outer-fold candidate-family selection: "
            + ", ".join(candidate_ids),
            flush=True,
        )

        for outer_fold in range(1, SETTINGS_VALIDATION.N_SPLITS + 1):
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

            candidate_selections = []
            for order_index, candidate_id in enumerate(candidate_ids):
                experiment = experiments_by_id[candidate_id]
                selection = select_c_and_training_threshold(
                    experiment=experiment,
                    prepared=prepared_by_id[candidate_id],
                    outer_train_patient_ids=train_patients,
                    outer_train_labels=train_labels,
                    patient_to_group=patient_to_group,
                    outer_fold_index=outer_fold,
                    verbose=False,
                )
                candidate_selections.append(
                    {
                        "order_index": int(order_index),
                        "experiment_id": candidate_id,
                        "experiment": experiment,
                        "selection": selection,
                        "inner_auc": float(selection["inner_auc"]),
                    }
                )

            best_inner_auc = max(row["inner_auc"] for row in candidate_selections)
            eligible = [
                row
                for row in candidate_selections
                if row["inner_auc"]
                >= best_inner_auc - SETTINGS_VALIDATION.MODEL_FAMILY_SELECTION_AUC_TOLERANCE
            ]
            chosen = min(eligible, key=lambda row: row["order_index"])
            eligible_ids = {
                row["experiment_id"] for row in eligible
            }
            chosen_id = chosen["experiment_id"]
            chosen_experiment = chosen["experiment"]
            chosen_selection = chosen["selection"]

            for row in candidate_selections:
                fold_candidate_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "candidate_order": int(row["order_index"] + 1),
                        "experiment_id": row["experiment_id"],
                        "inner_pooled_auc": float(row["inner_auc"]),
                        "selected_c": float(row["selection"]["selected_c"]),
                        "best_inner_auc_in_family": float(best_inner_auc),
                        "within_selection_tolerance": bool(
                            row["experiment_id"] in eligible_ids
                        ),
                        "chosen_for_outer_fold": bool(
                            row["experiment_id"] == chosen_id
                        ),
                    }
                )

            raw_scores, valid_labels, evaluated_patients = (
                fit_predict_raw_for_patient_sets(
                    experiment=chosen_experiment,
                    prepared=prepared_by_id[chosen_id],
                    train_patients=train_patients,
                    valid_patients=valid_patients,
                    c_value=chosen_selection["selected_c"],
                )
            )
            probabilities = np.clip(raw_scores, 0.0, 1.0)

            fold_auc = float(roc_auc_score(valid_labels, probabilities))
            fold_selection_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "chosen_experiment_id": chosen_id,
                    "chosen_inner_auc": float(chosen["inner_auc"]),
                    "best_inner_auc_in_family": float(best_inner_auc),
                    "selected_c": float(chosen_selection["selected_c"]),
                    "held_out_fold_auc": fold_auc,
                    "n_train_patients": int(len(train_patients)),
                    "n_valid_patients": int(len(valid_patients)),
                }
            )

            for patient_id, label, probability in zip(
                evaluated_patients,
                valid_labels,
                probabilities,
            ):
                patient_id = str(patient_id)
                if patient_id in score_by_patient:
                    raise RuntimeError(
                        f"Candidate-family selection scored {patient_id} twice."
                    )
                if int(label) != patient_to_label[patient_id]:
                    raise RuntimeError(
                        f"Candidate-family selection label mismatch for {patient_id}."
                    )
                score_by_patient[patient_id] = float(probability)
                fold_by_patient[patient_id] = int(outer_fold)
                selected_model_by_patient[patient_id] = chosen_id
                selected_c_by_patient[patient_id] = float(
                    chosen_selection["selected_c"]
                )

            print(
                f"[EXACT SUPPORT MODEL SELECTION] outer_fold={outer_fold}: "
                f"chosen={chosen_id}, inner_auc={chosen['inner_auc']:.4f}, "
                f"held_out_auc={fold_auc:.4f}",
                flush=True,
            )

        if set(score_by_patient) != set(all_patients.tolist()):
            missing = sorted(set(all_patients.tolist()) - set(score_by_patient))
            raise RuntimeError(
                f"Candidate-family selection is missing OOF patients: {missing}."
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
        auc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        # AUC/AUPRC intervals do not depend on a decision threshold. The temporary
        # 0.5 labels below are used only because the shared bootstrap helper also
        # computes threshold-dependent metrics; only its discrimination intervals
        # are retained for this candidate-family summary.
        discrimination_intervals = bootstrap_patient_metric_intervals(
            labels,
            scores,
            (scores >= 0.5).astype(np.int64),
            random_state=SETTINGS_RUNTIME.RANDOM_SEED + 71_001,
        )

        patient_rows = [
            {
                "patient_id": str(patient_id),
                "true_label": int(patient_to_label[str(patient_id)]),
                "outer_fold": int(fold_by_patient[str(patient_id)]),
                "oof_score": float(score_by_patient[str(patient_id)]),
                "selected_experiment_id": selected_model_by_patient[str(patient_id)],
                "selected_c": float(selected_c_by_patient[str(patient_id)]),
            }
            for patient_id in ordered_patients
        ]
        selected_counts = {
            candidate_id: int(
                sum(
                    row["chosen_experiment_id"] == candidate_id
                    for row in fold_selection_rows
                )
            )
            for candidate_id in candidate_ids
        }
        summary = {
            "status": "OK",
            "candidate_experiment_ids": list(candidate_ids),
            "tie_break_order": list(candidate_ids),
            "inner_auc_tolerance": float(
                SETTINGS_VALIDATION.MODEL_FAMILY_SELECTION_AUC_TOLERANCE
            ),
            "outer_oof_auc": auc,
            "outer_oof_auc_ci": discrimination_intervals["auc"],
            "outer_oof_auprc": auprc,
            "outer_oof_auprc_ci": discrimination_intervals["auprc"],
            "selected_fold_counts": selected_counts,
            "n_patients": int(len(ordered_patients)),
            "interpretation": (
                "Model identity and classifier C are selected using outer-training "
                "data only. This is an internal nested estimate of the complete "
                "selection procedure, not external validation."
            ),
        }

        with open(
            output_dir / "fold_candidate_inner_selection.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(fold_candidate_rows[0]))
            writer.writeheader()
            writer.writerows(fold_candidate_rows)
        with open(
            output_dir / "fold_selected_model.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(fold_selection_rows[0]))
            writer.writeheader()
            writer.writerows(fold_selection_rows)
        with open(
            output_dir / "patient_oof_predictions.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(patient_rows[0]))
            writer.writeheader()
            writer.writerows(patient_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print(
            f"[EXACT SUPPORT MODEL SELECTION] pooled OOF AUC={auc:.4f}, "
            f"AUPRC={auprc:.4f}; selections={selected_counts}",
            flush=True,
        )
        return summary

    @staticmethod
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
            f"[STABILITY] Running {SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS} repeated nested "
            f"patient-level splits for {experiment.experiment_id}.",
            flush=True,
        )

        for repeat_index in range(SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS):
            seed = SETTINGS_VALIDATION.REPEATED_NESTED_CV_RANDOM_STATE + repeat_index
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
                f"{SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS}, seed={seed}, "
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
            "repeats": int(SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS),
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

    @staticmethod
    def write_repeated_stability_ranking(stability_summaries, output_path):
        """Write the preferred split-stability ranking by median repeated-CV AUC.

        The ordinary experiment table is retained because it contains complete OOF
        metrics for one frozen fold manifest. For model ranking in this 30-patient
        cohort, however, gives priority to the median across all predeclared
        repeated nested-CV splits. This helper makes that ordering explicit rather
        than allowing a single favorable split to define the final ranking.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config_by_id = {
            experiment.experiment_id: experiment
            for experiment in SETTINGS_EXPERIMENTS.EXPERIMENT_REGISTRY
        }
        rows = []
        for experiment_id in SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS:
            summary = stability_summaries.get(
                experiment_id,
                {"status": "SKIPPED_MISSING_SUMMARY"},
            )
            experiment = config_by_id[experiment_id]
            row = {
                "experiment_id": experiment_id,
                "status": summary.get("status", "UNKNOWN"),
                "role": experiment.role,
                "description": experiment.description,
                "is_primary_candidate": int(
                    experiment_id == SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID
                ),
                "repeats": summary.get("repeats", ""),
                "auc_median": summary.get("auc_median", ""),
                "auc_q25": summary.get("auc_q25", ""),
                "auc_q75": summary.get("auc_q75", ""),
                "auc_mean": summary.get("auc_mean", ""),
                "auc_std": summary.get("auc_std", ""),
                "auc_minimum": summary.get("auc_minimum", ""),
                "auc_maximum": summary.get("auc_maximum", ""),
            }
            rows.append(row)

        rows.sort(
            key=lambda row: (
                row["status"] != "OK",
                -float(row["auc_median"])
                if row["status"] == "OK"
                else 0.0,
                row["experiment_id"],
            )
        )
        fieldnames = (
            "experiment_id",
            "status",
            "role",
            "description",
            "is_primary_candidate",
            "repeats",
            "auc_median",
            "auc_q25",
            "auc_q75",
            "auc_mean",
            "auc_std",
            "auc_minimum",
            "auc_maximum",
        )
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    @staticmethod
    def print_repeated_stability_ranking(rows):
        """Print the final repeated-CV ranking before the single-manifest table."""

        successful = [row for row in rows if row.get("status") == "OK"]
        if not successful:
            print(
                "\n[STABILITY RANKING] No completed repeated-CV rows are available.",
                flush=True,
            )
            return

        print("\n" + "=" * 154, flush=True)
        print(
            "PRIMARY SPLIT-STABILITY RANKING — ORDERED BY MEDIAN REPEATED NESTED-CV AUC",
            flush=True,
        )
        print("=" * 154, flush=True)
        print(
            f"{'Rank':<6} {'Experiment':<76} {'Role':<35} "
            f"{'Median AUC':>12} {'IQR':>20}",
            flush=True,
        )
        print("-" * 154, flush=True)
        for rank, row in enumerate(successful, start=1):
            marker = " [PRIMARY]" if row["is_primary_candidate"] else ""
            experiment_text = (row["experiment_id"] + marker)[:76]
            iqr_text = f"[{float(row['auc_q25']):.4f}, {float(row['auc_q75']):.4f}]"
            print(
                f"{rank:<6} {experiment_text:<76} {row['role']:<35} "
                f"{float(row['auc_median']):>12.4f} {iqr_text:>20}",
                flush=True,
            )
        print("-" * 154, flush=True)
        print(
            "This is the preferred internal ranking. Repeats reuse the same 30 "
            "patients and are descriptive split-sensitivity analyses, not "
            "independent samples or external validation.",
            flush=True,
        )
        print("=" * 154, flush=True)

    @staticmethod
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

    @staticmethod
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
        rng = np.random.default_rng(SETTINGS_VALIDATION.LABEL_PERMUTATION_RANDOM_STATE)
        rows = []
        progress_interval = max(1, SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES // 10)

        print(
            f"[PERMUTATION] Running {SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES} patient-label "
            f"permutations for {experiment.experiment_id}.",
            flush=True,
        )

        for permutation_index in range(SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES):
            permuted_labels = rng.permutation(original_labels)
            permuted_prepared = dict(prepared)
            permuted_prepared["patient_ids"] = patient_ids
            permuted_prepared["X"] = original_X
            permuted_prepared["y"] = permuted_labels

            # Keep algorithmic CV randomness fixed across permutations. The labels
            # change, so stratified membership can still change, but the random
            # splitting rule itself is conditioned on the same seed as the observed
            # analysis. This yields a cleaner Monte-Carlo randomization test than
            # adding a second source of split randomness to every null replicate.
            fold_seed = SETTINGS_VALIDATION.CV_RANDOM_STATE
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
                or permutation_index + 1 == SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES
            ):
                print(
                    f"[PERMUTATION] {permutation_index + 1}/"
                    f"{SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES} complete; latest AUC={auc:.4f}",
                    flush=True,
                )

        null_aucs = np.asarray(
            [row["permuted_auc"] for row in rows], dtype=np.float64
        )
        empirical_p_value = float(
            (1 + np.sum(null_aucs >= float(observed_auc)))
            / (SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES + 1)
        )
        summary = {
            "status": "OK",
            "experiment_id": experiment.experiment_id,
            "observed_auc": float(observed_auc),
            "permutation_replicates": int(SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES),
            "null_auc_mean": float(np.mean(null_aucs)),
            "null_auc_median": float(np.median(null_aucs)),
            "null_auc_std": float(np.std(null_aucs)),
            "null_auc_q025": float(np.quantile(null_aucs, 0.025)),
            "null_auc_q975": float(np.quantile(null_aucs, 0.975)),
            "empirical_one_sided_p_value": empirical_p_value,
            "permutation_unit": "Directory_* patient labels",
            "outer_cv_random_state": int(SETTINGS_VALIDATION.CV_RANDOM_STATE),
            "outer_cv_randomness_varied_across_permutations": False,
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

    @staticmethod
    def run_selection_adjusted_candidate_family_permutation_test(
        experiments_by_id,
        prepared_by_id,
        base_fold_manifest_rows,
        observed_auc_by_id,
        output_dir,
    ):
        """Run a max-AUROC permutation test across the explored candidates.

        Every permutation uses one common patient-label vector and one common outer
        fold manifest. The complete nested fitting path is rerun separately for each
        predeclared candidate, after which the maximum candidate AUROC is stored.
        The observed maximum is compared with this maximum-null distribution. This
        controls the permutation result for selecting the best representation from
        the stated family, unlike a post-hoc single-model permutation test.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_ids = tuple(SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS)

        first_id = candidate_ids[0]
        first_prepared = prepared_by_id[first_id]
        if first_prepared.get("unit") != "patient":
            raise ValueError("Selection-adjusted permutation requires patient rows.")
        first_order = np.argsort(
            np.asarray(first_prepared["patient_ids"]).astype(str)
        )
        patient_ids = np.asarray(first_prepared["patient_ids"]).astype(str)[
            first_order
        ]
        original_labels = np.asarray(first_prepared["y"], dtype=np.int64)[
            first_order
        ]

        aligned_X = {}
        for candidate_id in candidate_ids:
            if candidate_id not in experiments_by_id:
                raise KeyError(f"Unknown candidate {candidate_id}.")
            prepared = prepared_by_id[candidate_id]
            if prepared.get("unit") != "patient":
                raise ValueError(
                    f"{candidate_id} is not a patient-level representation."
                )
            order = np.argsort(np.asarray(prepared["patient_ids"]).astype(str))
            candidate_patients = np.asarray(prepared["patient_ids"]).astype(str)[
                order
            ]
            candidate_labels = np.asarray(prepared["y"], dtype=np.int64)[order]
            if not np.array_equal(candidate_patients, patient_ids):
                raise RuntimeError(
                    f"{candidate_id} has a different patient set/order."
                )
            if not np.array_equal(candidate_labels, original_labels):
                raise RuntimeError(
                    f"{candidate_id} has labels inconsistent with the candidate family."
                )
            aligned_X[candidate_id] = np.asarray(prepared["X"])[order]

        patient_to_group = {
            str(row["patient_id"]): str(row["duplicate_component_id"])
            for row in base_fold_manifest_rows
        }
        missing_groups = sorted(set(patient_ids.tolist()) - set(patient_to_group))
        if missing_groups:
            raise RuntimeError(
                f"Missing duplicate components for patients: {missing_groups}."
            )

        observed_values = {
            candidate_id: float(observed_auc_by_id[candidate_id])
            for candidate_id in candidate_ids
        }
        observed_max_auc = max(observed_values.values())
        observed_winner = next(
            candidate_id
            for candidate_id in candidate_ids
            if observed_values[candidate_id] == observed_max_auc
        )

        rng = np.random.default_rng(
            SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE
        )
        maximum_rows = []
        candidate_rows = []
        winner_counts = {candidate_id: 0 for candidate_id in candidate_ids}
        progress_interval = max(
            1,
            SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES // 10,
        )

        print(
            "[PERMUTATION][MAX] Running "
            f"{SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES} max-statistic "
            f"permutations across {len(candidate_ids)} candidates.",
            flush=True,
        )

        for permutation_index in range(
            SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES
        ):
            permuted_labels = rng.permutation(original_labels)
            # The observed candidate-family maximum and every permuted maximum use
            # the same outer-CV random-state policy. Only the patient labels are
            # randomized; candidate representations, hyperparameter procedure and
            # algorithmic split seed remain fixed.
            fold_seed = SETTINGS_VALIDATION.CV_RANDOM_STATE
            fold_rows = _build_fold_manifest_rows_for_seed(
                patient_ids,
                permuted_labels,
                patient_to_group,
                fold_seed,
            )

            aucs = {}
            for candidate_id in candidate_ids:
                prepared = dict(prepared_by_id[candidate_id])
                prepared["patient_ids"] = patient_ids
                prepared["X"] = aligned_X[candidate_id]
                prepared["y"] = permuted_labels
                _, labels, scores, _, _ = run_nested_cv_auc_only(
                    experiments_by_id[candidate_id],
                    prepared,
                    fold_rows,
                    verbose=False,
                )
                auc = float(roc_auc_score(labels, scores))
                aucs[candidate_id] = auc
                candidate_rows.append(
                    {
                        "permutation_index": int(permutation_index + 1),
                        "outer_cv_random_state": int(fold_seed),
                        "experiment_id": candidate_id,
                        "permuted_auc": auc,
                    }
                )

            maximum_auc = max(aucs.values())
            winning_id = next(
                candidate_id
                for candidate_id in candidate_ids
                if aucs[candidate_id] == maximum_auc
            )
            winner_counts[winning_id] += 1
            maximum_rows.append(
                {
                    "permutation_index": int(permutation_index + 1),
                    "outer_cv_random_state": int(fold_seed),
                    "maximum_permuted_auc": float(maximum_auc),
                    "winning_experiment_id": winning_id,
                }
            )

            if (
                permutation_index == 0
                or (permutation_index + 1) % progress_interval == 0
                or permutation_index + 1
                == SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES
            ):
                print(
                    f"[PERMUTATION][MAX] {permutation_index + 1}/"
                    f"{SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES} complete; "
                    f"latest max AUC={maximum_auc:.4f} ({winning_id})",
                    flush=True,
                )

        null_maxima = np.asarray(
            [row["maximum_permuted_auc"] for row in maximum_rows],
            dtype=np.float64,
        )
        empirical_p_value = float(
            (1 + np.sum(null_maxima >= observed_max_auc))
            / (SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES + 1)
        )
        individual_candidate_summaries = {}
        for candidate_id in candidate_ids:
            candidate_null = np.asarray(
                [
                    row["permuted_auc"]
                    for row in candidate_rows
                    if row["experiment_id"] == candidate_id
                ],
                dtype=np.float64,
            )
            observed_candidate_auc = observed_values[candidate_id]
            individual_candidate_summaries[candidate_id] = {
                "observed_auc": float(observed_candidate_auc),
                "null_auc_mean": float(np.mean(candidate_null)),
                "null_auc_median": float(np.median(candidate_null)),
                "null_auc_q025": float(np.quantile(candidate_null, 0.025)),
                "null_auc_q975": float(np.quantile(candidate_null, 0.975)),
                "unadjusted_empirical_one_sided_p_value": float(
                    (1 + np.sum(candidate_null >= observed_candidate_auc))
                    / (SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES + 1)
                ),
            }

        summary = {
            "status": "OK",
            "candidate_experiment_ids": list(candidate_ids),
            "observed_auc_by_experiment": observed_values,
            "observed_maximum_auc": float(observed_max_auc),
            "observed_winning_experiment_id": observed_winner,
            "permutation_replicates": int(
                SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES
            ),
            "null_maximum_auc_mean": float(np.mean(null_maxima)),
            "null_maximum_auc_median": float(np.median(null_maxima)),
            "null_maximum_auc_std": float(np.std(null_maxima)),
            "null_maximum_auc_q025": float(np.quantile(null_maxima, 0.025)),
            "null_maximum_auc_q975": float(np.quantile(null_maxima, 0.975)),
            "empirical_familywise_one_sided_p_value": empirical_p_value,
            "outer_cv_random_state": int(SETTINGS_VALIDATION.CV_RANDOM_STATE),
            "outer_cv_randomness_varied_across_permutations": False,
            "null_winner_counts": winner_counts,
            "individual_candidate_null_summaries": (
                individual_candidate_summaries
            ),
            "statistic": "maximum nested-CV AUROC across candidate representations",
            "interpretation": (
                "The maximum-null statistic adjusts the association sanity test "
                "for choosing the best result from the declared candidate family. "
                "It does not establish CAD-specific anatomy or external validity."
            ),
        }

        with open(
            output_dir / "selection_adjusted_permutation_candidate_auc.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(candidate_rows[0]))
            writer.writeheader()
            writer.writerows(candidate_rows)
        with open(
            output_dir / "selection_adjusted_permutation_maximum_auc.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(maximum_rows[0]))
            writer.writeheader()
            writer.writerows(maximum_rows)
        (output_dir / "selection_adjusted_permutation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print(
            f"[PERMUTATION][MAX] observed max AUC={observed_max_auc:.4f} "
            f"({observed_winner}); null median={summary['null_maximum_auc_median']:.4f}; "
            f"family-wise p={empirical_p_value:.6f}",
            flush=True,
        )
        return summary

    @staticmethod
    def paired_auc_difference_interval(
        baseline_labels,
        baseline_scores,
        comparison_scores,
        n_bootstrap=SETTINGS_VALIDATION.PAIRED_BOOTSTRAP_REPLICATES,
        confidence=SETTINGS_VALIDATION.BOOTSTRAP_CONFIDENCE,
        random_state=SETTINGS_RUNTIME.RANDOM_SEED,
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

    @staticmethod
    def compare_experiments_to_baseline(results):
        """Return one paired AUC comparison row for every successful experiment."""

        successful = {
            result["config"].experiment_id: result
            for result in results
            if result.get("status") == "OK"
        }
        if SETTINGS_EXPERIMENTS.BASELINE_EXPERIMENT_ID not in successful:
            print(
                "[COMPARISON] Baseline failed or is missing; paired comparisons "
                "cannot be calculated.",
                flush=True,
            )
            return []

        baseline = successful[SETTINGS_EXPERIMENTS.BASELINE_EXPERIMENT_ID]
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
                random_state=SETTINGS_RUNTIME.RANDOM_SEED
                + int(hashlib.sha256(experiment_id.encode()).hexdigest()[:8], 16),
            )
            row = {
                "baseline_experiment_id": SETTINGS_EXPERIMENTS.BASELINE_EXPERIMENT_ID,
                "comparison_experiment_id": experiment_id,
                **comparison,
            }
            rows.append(row)

        return sorted(rows, key=lambda row: row["comparison_experiment_id"])

    @staticmethod
    def compare_predeclared_ablation_pairs(results):
        """Calculate the scientifically matched paired comparisons declared above.

        Each row uses the reference declared for that scientific question and the
        same patient resamples for both scores.
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
        ) in SETTINGS_EXPERIMENTS.PRIMARY_ABLATION_COMPARISONS:
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
                random_state=SETTINGS_RUNTIME.RANDOM_SEED
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def write_master_outputs(
        results,
        failed_results,
        paired_rows,
        primary_ablation_rows,
        candidate_family_selection_summary,
        exact_support_row_contract_summary,
        comparison_dir,
    ):
        """Write compact cross-experiment tables and one machine-readable report."""

        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_lookup = {row["comparison_experiment_id"]: row for row in paired_rows}
        summary_rows = [
            result_summary_row(result, comparison_lookup)
            for result in results if result.get("status") == "OK"
        ]
        summary_rows.sort(key=lambda row: (-row["auc"], row["experiment_id"]))

        summary_path = comparison_dir / "experiment_summary.csv"
        if summary_rows:
            with open(summary_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
                writer.writeheader(); writer.writerows(summary_rows)
        else:
            summary_path.write_text("experiment_id,status\n", encoding="utf-8")

        failure_path = comparison_dir / "failed_experiments.csv"
        failure_fields = ["experiment_id", "error_type", "error_message", "traceback"]
        with open(failure_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=failure_fields)
            writer.writeheader()
            for row in failed_results:
                writer.writerow({field: row.get(field, "") for field in failure_fields})

        prediction_path = comparison_dir / "patient_predictions_all_experiments.csv"
        with open(prediction_path, "w", newline="", encoding="utf-8") as file:
            fields = ["experiment_id", "patient_id", "true_label", "outer_fold", "oof_score", "predicted_label"]
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
            for result in results:
                if result.get("status") != "OK":
                    continue
                for patient_id, label, fold, score, prediction in zip(
                    result["patient_ids"], result["labels"], result["folds"],
                    result["probabilities"], result["predictions"],
                ):
                    writer.writerow({
                        "experiment_id": result["config"].experiment_id,
                        "patient_id": str(patient_id),
                        "true_label": int(label),
                        "outer_fold": int(fold),
                        "oof_score": float(score),
                        "predicted_label": int(prediction),
                    })

        for filename, rows in (
            ("paired_auc_comparisons.csv", paired_rows),
            ("paired_primary_ablation_comparisons.csv", primary_ablation_rows),
        ):
            path = comparison_dir / filename
            if rows:
                with open(path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            else:
                path.write_text("status\n", encoding="utf-8")

        final_report = {
            "pipeline_schema": SETTINGS_RUNTIME.PIPELINE_SCHEMA_ID,
            "reference_experiment_id": SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
            "primary_candidate_experiment_id": SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
            "successful_experiments": len(summary_rows),
            "failed_experiments": len(failed_results),
            "experiment_summary": summary_rows,
            "paired_auc_comparisons": paired_rows,
            "predeclared_comparisons": primary_ablation_rows,
            "model_family_nested_selection": candidate_family_selection_summary,
            "exact_support_row_contract": exact_support_row_contract_summary,
            "interpretation": (
                "Scores are internal patient-level OOF model outputs. Shortcut "
                "controls and the absence of independent external validation must "
                "be considered before biological or clinical interpretation."
            ),
        }
        (comparison_dir / "final_report.json").write_text(
            json.dumps(final_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary_rows

    @staticmethod
    def print_final_comparison(summary_rows, failed_results=None):
        """Print one compact table plus explicit shortcut-control warnings."""

        failed_results = failed_results or []
        print("\nFINAL PATIENT-LEVEL OOF COMPARISON", flush=True)
        print("-" * 118, flush=True)
        print(f"{'Experiment':<61} {'Role':<24} {'AUC [95% CI]':<24}", flush=True)
        print("-" * 118, flush=True)
        for row in summary_rows:
            auc_text = f"{row['auc']:.4f} [{row['auc_ci_lower']:.4f}, {row['auc_ci_upper']:.4f}]"
            print(f"{row['experiment_id']:<61} {row['role']:<24} {auc_text:<24}", flush=True)
        print("-" * 118, flush=True)

        for row in summary_rows:
            if row["role"] in {"negative_control", "segmentation_control", "anatomy_destruction_control"} and row["auc"] >= SETTINGS_AUDIT.SHORTCUT_WARNING_AUC:
                print(
                    f"[SHORTCUT WARNING] {row['experiment_id']} AUC={row['auc']:.4f} "
                    "exceeds the warning threshold; candidate performance is not "
                    "specific to cardiac pathology in this cohort.",
                    flush=True,
                )
        if failed_results:
            print(f"[WARNING] {len(failed_results)} experiments failed; inspect failed_experiments.csv.", flush=True)


class ReportingManager:
    """CSV/JSON reporting, console summaries and run metadata."""

    @staticmethod
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


class PipelineRunner:
    """MONAI suite orchestration, cached execution and console logging."""

    @staticmethod
    def main():
        """Run every enabled experiment and compare the patient-level results."""

        refresh_runtime_device("main pipeline")
        pipeline_started_at = time.perf_counter()
        stage_durations = {}
        experiments = get_enabled_experiments()

        print("\n" + "#" * 100, flush=True)
        print("CAD CARDIAC MRI — PATIENT-LEVEL RESEARCH PIPELINE", flush=True)
        print("#" * 100, flush=True)
        print(f"[SUITE] Dataset: {SETTINGS_PATHS.DATASET_PATH}", flush=True)
        print(f"[SUITE] Output: {SETTINGS_PATHS.OUTPUT_DIR}", flush=True)
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
        print(
            "[SUITE] Validation profile="
            f"{SETTINGS_VALIDATION.VALIDATION_RUNTIME_PROFILE}; stability="
            f"{SETTINGS_VALIDATION.RUN_REPEATED_NESTED_CV_STABILITY} × "
            f"{SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS} repeats × "
            f"{len(SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS)} experiments; ordinary permutation="
            f"{SETTINGS_VALIDATION.RUN_PATIENT_LABEL_PERMUTATION_TEST} × "
            f"{SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES}; selection-adjusted permutation="
            f"{SETTINGS_VALIDATION.RUN_SELECTION_ADJUSTED_PERMUTATION_TEST} × "
            f"{SETTINGS_VALIDATION.SELECTION_ADJUSTED_PERMUTATION_REPLICATES}.",
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
            "All registry, CV, audit and model settings passed validation; the "
            "exact A17/C31/C32/C33 transform contract self-test also passed.",
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
        SETTINGS_PATHS.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
            directory = SETTINGS_PATHS.OUTPUT_DIR / subdirectory_name
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        write_suite_configuration(
            SETTINGS_PATHS.OUTPUT_DIR / "suite_configuration.json",
            experiments,
        )
        stage_durations["02 Output structure"] = _print_stage_complete(
            2,
            "Create output structure and save predeclared configuration",
            stage_started,
            f"Configuration: {SETTINGS_PATHS.OUTPUT_DIR / 'suite_configuration.json'}",
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
        samples = load_samples(SETTINGS_PATHS.DATASET_PATH)
        stage_durations["03 Dataset discovery"] = _print_stage_complete(
            3,
            "Discover Directory_* patients and image rows",
            stage_started,
            f"Discovered {len(samples)} image rows.",
        )

        # ======================================================================
        # MAIN STAGE 4 -- LOAD OR EXTRACT ONE SHARED MULTI-VIEW FEATURE BANK
        # ======================================================================
        # This is the expensive image-processing stage. Each JPEG is decoded once,
        # standardized once, segmented by MONAI, and converted into the six image
        # representations required by the active candidate/control panel. Frozen
        # EfficientNet embeddings are cached and reused by every CPU evaluation.
        # The fingerprint includes dataset metadata and all feature-affecting settings.
        stage_started = _print_stage_start(
            4,
            "Load or build the shared multi-view feature bank",
            "Potentially the longest stage on a cache miss; all required image "
            "variants are encoded and cached.",
        )
        required_modes = required_efficientnet_feature_modes(experiments)
        fingerprint = feature_bank_fingerprint(samples, SETTINGS_PATHS.DATASET_PATH)
        cache_dir = SETTINGS_PATHS.FEATURE_CACHE_ROOT / fingerprint[:16]
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
        if SETTINGS_AUDIT.AUDIT_EXACT_DECODED_PIXEL_DUPLICATES:
            exact_summary, exact_edges = audit_exact_decoded_pixel_duplicates(
                SETTINGS_PATHS.OUTPUT_DIR / "audits" / "exact_decoded_pixel_duplicate_groups.csv",
                samples,
                bank["decoded_pixel_hashes"],
            )
        else:
            exact_summary = {"enabled": False}
            exact_edges = set()

        if SETTINGS_AUDIT.AUDIT_PERCEPTUAL_NEAR_DUPLICATES:
            phash_summary, phash_edges = audit_perceptual_near_duplicate_candidates(
                SETTINGS_PATHS.OUTPUT_DIR
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
            SETTINGS_PATHS.OUTPUT_DIR / "manifests" / "patient_fold_manifest.csv",
            fold_manifest_rows,
        )
        write_cohort_manifest(
            SETTINGS_PATHS.OUTPUT_DIR / "manifests" / "cohort_manifest.csv",
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
        # These descriptive tables audit whether geometry, file size, padding,
        # intensity scaling or MONAI gate behavior differ between classes. They are
        # not classifier experiments and never influence model selection.
        stage_started = _print_stage_start(
            7,
            "Build and save provenance, standardization and MONAI-QC audits",
            "Patient-level aggregation and descriptive audit files.",
        )
        provenance_set = aggregate_patient_provenance_features(bank)
        standardization_set = aggregate_patient_standardization_features(bank)
        (
            standardized_monai_gate_comparison,
            standardized_monai_qc_set,
        ) = write_standardized_monai_qc_outputs(
            SETTINGS_PATHS.OUTPUT_DIR / "audits",
            bank,
        )
        write_patient_tabular_features(
            SETTINGS_PATHS.OUTPUT_DIR / "audits" / "patient_provenance_features.csv",
            *provenance_set,
        )
        write_patient_tabular_features(
            SETTINGS_PATHS.OUTPUT_DIR / "audits" / "patient_standardization_features.csv",
            *standardization_set,
        )
        write_tabular_class_summary(
            SETTINGS_PATHS.OUTPUT_DIR / "audits" / "provenance_class_summary.csv",
            provenance_set[0],
            provenance_set[1],
            provenance_set[3],
        )
        write_tabular_class_summary(
            SETTINGS_PATHS.OUTPUT_DIR / "audits" / "standardization_class_summary.csv",
            standardization_set[0],
            standardization_set[1],
            standardization_set[3],
        )
        write_tabular_class_summary(
            SETTINGS_PATHS.OUTPUT_DIR / "audits" / "standardized_monai_qc_class_summary.csv",
            standardized_monai_qc_set[0],
            standardized_monai_qc_set[1],
            standardized_monai_qc_set[3],
        )
        stage_durations["07 QC/provenance controls"] = _print_stage_complete(
            7,
            "Build and save provenance, standardization and MONAI-QC audits",
            stage_started,
            "Descriptive provenance, standardization and MONAI-QC audits are ready.",
        )

        # ======================================================================
        # MAIN STAGE 8 -- RUN THE PREDECLARED EXPERIMENT REGISTRY
        # ======================================================================
        # Every experiment reuses the same patient folds. Classifier C and the
        # decision threshold are selected only from inner OOF predictions in the
        # current outer-training cohort; the outer fold remains untouched.
        # Prepared patient representations are cached in memory.
        #
        # One experiment failure is written to its own failure.json and does not
        # erase successful results unless FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS=True.
        stage_started = _print_stage_start(
            8,
            "Run every enabled experiment on the shared folds",
            "Fold-local PCA and Logistic Regression fits on patient embeddings.",
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
            experiment_output = SETTINGS_PATHS.OUTPUT_DIR / "experiments" / experiment.experiment_id
            preparation_key = experiment_preparation_cache_key(experiment)

            try:
                if preparation_key not in prepared_cache:
                    preparation_started = time.perf_counter()
                    prepared_cache[preparation_key] = prepare_experiment_data(
                        experiment,
                        bank,
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

                if SETTINGS_AUDIT.FAIL_SUITE_IF_ANY_EXPERIMENT_FAILS:
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

        successful_by_id = {
            result["config"].experiment_id: result
            for result in successful_results
        }
        experiments_by_id = {
            experiment.experiment_id: experiment
            for experiment in experiments
        }
        exact_support_contract_prepared = {
            experiment_id: prepared_cache[
                experiment_preparation_cache_key(
                    experiments_by_id[experiment_id]
                )
            ]
            for experiment_id in (
                SETTINGS_EXPERIMENTS.REFERENCE_EXPERIMENT_ID,
                SETTINGS_EXPERIMENTS.VALID_ONLY_ABLATION_EXPERIMENT_ID,
                SETTINGS_EXPERIMENTS.MASK_ONLY_CONTROL_EXPERIMENT_ID,
                SETTINGS_EXPERIMENTS.SHUFFLED_INTENSITY_CONTROL_EXPERIMENT_ID,
                SETTINGS_EXPERIMENTS.COMPLEMENT_CONTROL_EXPERIMENT_ID,
            )
            if experiment_id in successful_by_id
        }
        exact_support_row_contract_summary = audit_exact_support_prepared_row_contract(
            exact_support_contract_prepared,
            SETTINGS_PATHS.OUTPUT_DIR / "audits" / "exact_support_row_contract.json",
        )
        candidate_family_output_dir = (
            SETTINGS_PATHS.OUTPUT_DIR / "comparison" / "model_family_nested_selection"
        )
        candidate_family_output_dir.mkdir(parents=True, exist_ok=True)
        if SETTINGS_VALIDATION.RUN_MODEL_FAMILY_NESTED_SELECTION:
            missing_candidate_ids = [
                experiment_id
                for experiment_id in SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS
                if experiment_id not in successful_by_id
            ]
            if missing_candidate_ids:
                candidate_family_selection_summary = {
                    "status": "SKIPPED_MISSING_OR_FAILED_EXPERIMENT",
                    "candidate_experiment_ids": list(
                        SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS
                    ),
                    "missing_experiments": missing_candidate_ids,
                }
                print(
                    "[EXACT SUPPORT MODEL SELECTION] SKIPPED because candidates are missing "
                    f"or failed: {missing_candidate_ids}.",
                    flush=True,
                )
            else:
                try:
                    candidate_prepared_by_id = {
                        experiment_id: prepared_cache[
                            experiment_preparation_cache_key(
                                experiments_by_id[experiment_id]
                            )
                        ]
                        for experiment_id in (
                            SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS
                        )
                    }
                    candidate_family_selection_summary = (
                        run_model_family_nested_selection(
                            experiments_by_id=experiments_by_id,
                            prepared_by_id=candidate_prepared_by_id,
                            fold_manifest_rows=fold_manifest_rows,
                            output_dir=candidate_family_output_dir,
                        )
                    )
                except Exception as error:
                    candidate_family_selection_summary = {
                        "status": "FAILED",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    print(
                        "[EXACT SUPPORT MODEL SELECTION] FAILED: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stdout)
        else:
            candidate_family_selection_summary = {
                "status": "SKIPPED_DISABLED",
                "candidate_experiment_ids": list(
                    SETTINGS_VALIDATION.MODEL_FAMILY_NESTED_SELECTION_IDS
                ),
            }
            print("[EXACT SUPPORT MODEL SELECTION] SKIPPED by configuration.", flush=True)

        (
            candidate_family_output_dir / "summary.json"
        ).write_text(
            json.dumps(
                candidate_family_selection_summary,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        summary_rows = write_master_outputs(
            successful_results,
            failed_results,
            paired_rows,
            primary_ablation_rows,
            candidate_family_selection_summary,
            exact_support_row_contract_summary,
            SETTINGS_PATHS.OUTPUT_DIR / "comparison",
        )
        stage_durations["09 Master comparisons"] = _print_stage_complete(
            9,
            "Calculate paired comparisons and write master result files",
            stage_started,
            f"Summary rows={len(summary_rows)}; baseline-paired rows={len(paired_rows)}; "
            f"matched-ablation rows={len(primary_ablation_rows)}; "
            f"candidate-family selection="
            f"{candidate_family_selection_summary.get('status', 'UNKNOWN')}.",
        )

        # ======================================================================
        # MAIN STAGE 10 -- REPEAT SELECTED NESTED-CV EXPERIMENTS ACROSS SPLITS
        # ======================================================================
        # A single five-fold assignment is statistically fragile when the effective
        # sample size is only the number of Directory_* folders. The run also
        # showed that shortcut controls could be nearly as predictive as the primary
        # model. This stage therefore repeats the full nested patient-level fitting
        # path not only for the baseline, but also for the main simple-localization
        # candidate, strict ROI candidate, and key padding/border/exterior controls.
        # No split is selected according to AUC and no neural inference is repeated.
        stage_started = _print_stage_start(
            10,
            "Run repeated nested-CV split-stability analyses",
            "Several patient-level reruns for selected models and controls; frozen features are reused.",
        )
        stability_summaries = {}
        stability_root = SETTINGS_PATHS.OUTPUT_DIR / "stability"
        stability_root.mkdir(parents=True, exist_ok=True)

        if SETTINGS_VALIDATION.RUN_REPEATED_NESTED_CV_STABILITY:
            successful_by_id = {
                result["config"].experiment_id: result
                for result in successful_results
            }
            experiments_by_id = {
                experiment.experiment_id: experiment
                for experiment in experiments
            }

            for stability_experiment_id in SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS:
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
                SETTINGS_VALIDATION.REPEATED_STABILITY_COMPARISONS,
            )
            stability_ranking_rows = write_repeated_stability_ranking(
                stability_summaries,
                stability_root / "repeated_nested_cv_ranking.csv",
            )
            stability_summary = {
                "status": "OK",
                "repeats_per_experiment": int(SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS),
                "experiment_ids": list(SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS),
                "experiment_summaries": stability_summaries,
                "paired_comparisons": stability_paired_summary,
                "ranking_order": [
                    row["experiment_id"]
                    for row in stability_ranking_rows
                    if row.get("status") == "OK"
                ],
                "ranking_csv": str(
                    stability_root / "repeated_nested_cv_ranking.csv"
                ),
                "interpretation": (
                    "All configured models and controls are repeated over the same "
                    "predeclared outer split seeds. Direct paired deltas are aligned "
                    "by seed and no best split is selected."
                ),
            }
        else:
            stability_ranking_rows = write_repeated_stability_ranking(
                {},
                stability_root / "repeated_nested_cv_ranking.csv",
            )
            stability_paired_summary = {
                "status": "SKIPPED_DISABLED",
                "comparisons": [],
            }
            stability_summary = {
                "status": "SKIPPED_DISABLED",
                "experiment_ids": list(SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS),
                "paired_comparisons": stability_paired_summary,
            }
            print("[STABILITY] SKIPPED by configuration.", flush=True)

        (stability_root / "repeated_nested_cv_summary.json").write_text(
            json.dumps(stability_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        successful_stability_runs = sum(
            summary.get("status") == "OK"
            for summary in stability_summaries.values()
        )
        stage_durations["10 Repeated nested CV"] = _print_stage_complete(
            10,
            "Run repeated nested-CV split-stability analyses",
            stage_started,
            f"Completed={successful_stability_runs}/{len(SETTINGS_VALIDATION.STABILITY_EXPERIMENT_IDS)} configured experiments.",
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
        permutation_summaries = {}
        successful_by_id = {
            result["config"].experiment_id: result
            for result in successful_results
        }
        experiments_by_id = {
            experiment.experiment_id: experiment
            for experiment in experiments
        }

        if SETTINGS_VALIDATION.RUN_PATIENT_LABEL_PERMUTATION_TEST:
            for permutation_experiment_id in SETTINGS_VALIDATION.PERMUTATION_EXPERIMENT_IDS:
                permutation_result = successful_by_id.get(
                    permutation_experiment_id
                )
                permutation_experiment = experiments_by_id[
                    permutation_experiment_id
                ]
                experiment_output_dir = (
                    SETTINGS_PATHS.OUTPUT_DIR / "permutation" / permutation_experiment_id
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
                "experiment_ids": list(SETTINGS_VALIDATION.PERMUTATION_EXPERIMENT_IDS),
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
                "experiment_ids": list(SETTINGS_VALIDATION.PERMUTATION_EXPERIMENT_IDS),
                "experiment_summaries": {},
            }
            print("[PERMUTATION] SKIPPED by configuration.", flush=True)

        selection_adjusted_output_dir = (
            SETTINGS_PATHS.OUTPUT_DIR / "permutation" / "selection_adjusted_candidate_family"
        )
        selection_adjusted_output_dir.mkdir(parents=True, exist_ok=True)
        if SETTINGS_VALIDATION.RUN_SELECTION_ADJUSTED_PERMUTATION_TEST:
            missing_selection_candidates = [
                experiment_id
                for experiment_id in SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS
                if experiment_id not in successful_by_id
                or experiment_id not in experiments_by_id
            ]
            if missing_selection_candidates:
                selection_adjusted_summary = {
                    "status": "SKIPPED_MISSING_OR_FAILED_EXPERIMENT",
                    "candidate_experiment_ids": list(
                        SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS
                    ),
                    "missing_experiments": missing_selection_candidates,
                }
                print(
                    "[PERMUTATION][MAX] SKIPPED because candidates are missing "
                    f"or failed: {missing_selection_candidates}.",
                    flush=True,
                )
            else:
                try:
                    selection_prepared_by_id = {
                        experiment_id: prepared_cache[
                            experiment_preparation_cache_key(
                                experiments_by_id[experiment_id]
                            )
                        ]
                        for experiment_id in (
                            SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS
                        )
                    }
                    observed_auc_by_id = {
                        experiment_id: float(
                            successful_by_id[experiment_id]["summary"]
                            ["metrics"]["auc"]
                        )
                        for experiment_id in (
                            SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS
                        )
                    }
                    selection_adjusted_summary = (
                        run_selection_adjusted_candidate_family_permutation_test(
                            experiments_by_id=experiments_by_id,
                            prepared_by_id=selection_prepared_by_id,
                            base_fold_manifest_rows=fold_manifest_rows,
                            observed_auc_by_id=observed_auc_by_id,
                            output_dir=selection_adjusted_output_dir,
                        )
                    )
                except Exception as error:
                    selection_adjusted_summary = {
                        "status": "FAILED",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    print(
                        "[PERMUTATION][MAX] FAILED: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stdout)
        else:
            selection_adjusted_summary = {
                "status": "SKIPPED_DISABLED",
                "candidate_experiment_ids": list(
                    SETTINGS_VALIDATION.SELECTION_ADJUSTED_CANDIDATE_EXPERIMENT_IDS
                ),
            }
            print("[PERMUTATION][MAX] SKIPPED by configuration.", flush=True)

        (
            selection_adjusted_output_dir
            / "selection_adjusted_permutation_summary.json"
        ).write_text(
            json.dumps(selection_adjusted_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        permutation_summary["selection_adjusted_candidate_family"] = (
            selection_adjusted_summary
        )

        (
            SETTINGS_PATHS.OUTPUT_DIR / "permutation" / "patient_label_permutation_summary.json"
        ).write_text(
            json.dumps(permutation_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        stage_durations["11 Label permutation"] = _print_stage_complete(
            11,
            "Run patient-label permutation sanity test",
            stage_started,
            permutation_summary.get("status", "OK"),
        )
        # ======================================================================
        # MAIN STAGE 12 -- PRINT THE FINAL TABLE AND SAVE COMPLETE PROVENANCE
        # ======================================================================
        # The final console table summarizes successful experiments and prints
        # shortcut warnings for controls above the configured AUC threshold. The
        # metadata JSON records software versions, model hashes,
        # feature-bank identity, patient counts, audit summaries, successful and
        # failed experiments, and total runtime. All ordinary print() output and
        # tracebacks are simultaneously preserved in console_output.log.
        stage_started = _print_stage_start(
            12,
            "Print final comparison and save suite metadata",
            "Console table, warnings, metadata JSON and timing summary.",
        )
        print_repeated_stability_ranking(stability_ranking_rows)
        print_final_comparison(summary_rows, failed_results)
        print_primary_ablation_comparisons(primary_ablation_rows)

        total_runtime = time.perf_counter() - pipeline_started_at
        metadata = collect_suite_metadata(
            samples=samples,
            fingerprint=fingerprint,
            cache_status=cache_status,
            exact_summary=exact_summary,
            phash_summary=phash_summary,
            standardized_monai_gate_comparison=standardized_monai_gate_comparison,
            stability_summary=stability_summary,
            permutation_summary=permutation_summary,
            candidate_family_selection_summary=(
                candidate_family_selection_summary
            ),
            exact_support_row_contract_summary=exact_support_row_contract_summary,
            successful_results=successful_results,
            failed_results=failed_results,
            total_runtime=total_runtime,
        )
        (SETTINGS_PATHS.OUTPUT_DIR / "suite_run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        stage_durations["12 Final reporting"] = _print_stage_complete(
            12,
            "Print final comparison and save suite metadata",
            stage_started,
            f"Master summary: {SETTINGS_PATHS.OUTPUT_DIR / 'comparison' / 'experiment_summary.csv'}",
        )

        total_runtime = time.perf_counter() - pipeline_started_at
        _print_timing_summary(stage_durations, total_runtime)
        print(f"\n[SUITE] Outputs saved under: {SETTINGS_PATHS.OUTPUT_DIR}", flush=True)
        print(f"[SUITE] Console log: {SETTINGS_PATHS.CONSOLE_LOG_PATH}", flush=True)
        print(
            f"[SUITE] Completed with {len(successful_results)} successful and "
            f"{len(failed_results)} failed experiments in "
            f"{_format_elapsed_time(total_runtime)}.",
            flush=True,
        )
        print("Done!", flush=True)

    @staticmethod
    def build_monai_feature_cache_only(dataset_path=None):
        """GPU stage: create/reuse the shared frozen feature bank only.

        No patient classifier, stability analysis or permutation test is run here.
        Those CPU-heavy stages can be executed later through
        ``monai-cpu-from-cache`` after the Kaggle accelerator is disabled.
        """

        refresh_runtime_device("build MONAI/EfficientNet feature cache")
        validate_configuration()
        experiments = get_enabled_experiments()
        dataset_path = Path(dataset_path or SETTINGS_PATHS.DATASET_PATH)
        samples = load_samples(dataset_path)
        required_modes = required_efficientnet_feature_modes(experiments)
        fingerprint = feature_bank_fingerprint(samples, dataset_path)
        cache_dir = SETTINGS_PATHS.FEATURE_CACHE_ROOT / fingerprint[:16]
        bank, cache_status = load_or_extract_feature_bank(
            samples,
            required_modes,
            fingerprint,
            cache_dir,
        )
        summary = {
            "status": "MONAI_FEATURE_CACHE_READY",
            "cache_status": cache_status,
            "cache_dir": str(bank["cache_dir"]),
            "fingerprint": fingerprint,
            "device": DEVICE,
            "feature_cache_device_tag": resolved_feature_cache_device_tag(),
            "n_images": len(samples),
            "modes": list(required_modes),
        }
        print(
            "[STAGED][GPU] MONAI/EfficientNet feature bank ready: "
            f"{summary['cache_dir']}",
            flush=True,
        )
        return summary

    @staticmethod
    def existing_attention_checkpoint_map(workspace):
        """Return all cross-fitted checkpoints without retraining anything."""

        checkpoint_map = {}
        missing = []
        for fold in range(SETTINGS_ATTENTION.ATTENTION_SEGMENTATION_FOLDS):
            path = _checkpoint_path(workspace, fold)
            if path.is_file():
                checkpoint_map[fold] = path
            else:
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "Missing Attention U-Net checkpoints. Run train-attention or "
                "retrain-regenerate-attention in a GPU session first:\n  "
                + "\n  ".join(missing)
            )
        return checkpoint_map

    @staticmethod
    def load_existing_attention_feature_bank(samples, workspace, checkpoint_map):
        """CPU stage: load an already generated AU1-AU5 patient feature bank."""

        output_dir = workspace.comparison_output
        npz_path = output_dir / "attention_unet_patient_feature_bank.npz"
        metadata_path = output_dir / "attention_unet_feature_bank_metadata.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Attention patient feature bank is missing. Run "
                "build-attention-feature-cache in a GPU session first."
            )
        checkpoint_fingerprint = _attention_feature_fingerprint(
            samples, checkpoint_map
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != checkpoint_fingerprint:
            raise RuntimeError(
                "The existing Attention feature bank does not match the current "
                "checkpoints/dataset. Rebuild it in a GPU session with "
                "build-attention-feature-cache."
            )
        loaded = np.load(npz_path, allow_pickle=False)
        print(
            f"[STAGED][CPU] Loading Attention patient feature bank: {npz_path}",
            flush=True,
        )
        return {
            "fingerprint": checkpoint_fingerprint,
            "patient_ids": loaded["patient_ids"].astype(str),
            "labels": loaded["labels"].astype(np.int64),
            "features": {
                mode: loaded[f"X__{mode}"].astype(np.float32)
                for mode in SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES
            },
            "metadata": metadata,
        }

    @staticmethod
    def _review_mask_exists(row, source):
        """Return True when an existing source/manual mask can be opened on CPU."""

        manual = Path(str(row.get("manual_mask_path", "")))
        if manual.is_file():
            return True
        field = (
            "automatic_mask_path"
            if source == "monai"
            else "predicted_attention_mask_path"
        )
        path = Path(str(row.get(field, "")))
        return path.is_file() and path.stat().st_size > 0


class AttentionDataManager:
    """Attention workspace, manifests, storage policy, image caching and pseudo-masks."""

    @staticmethod
    def build_attention_workspace(root=None):
        """Resolve persistent and transient Attention U-Net storage.

        Scientific outputs remain under ``root`` so they can be preserved by a
        Kaggle notebook. Regenerable standardized training-image cache files
        live under ``transient_root``. Kaggle's ``/kaggle/temp`` is preferred for
        these files because filling ``/kaggle/working`` can prevent every later
        result, checkpoint and mask from being saved.
        """

        if root is None:
            default_root = (
                Path("/kaggle/working/cad_attention_unet_workspace")
                if Path("/kaggle/working").exists()
                else SETTINGS_PATHS.OUTPUT_ROOT / "cad_attention_unet_workspace"
            )
            root = Path(
                os.environ.get("CAD_ATTENTION_UNET_WORK_ROOT", str(default_root))
            )
        else:
            root = Path(root)

        if SETTINGS_ATTENTION._ATTENTION_RUNNING_ON_KAGGLE:
            default_transient_root = Path(
                "/kaggle/temp/cad_attention_unet_transient"
            )
        else:
            default_transient_root = root / "_transient"
        transient_root = Path(
            os.environ.get(
                "CAD_ATTENTION_UNET_TRANSIENT_ROOT",
                str(default_transient_root),
            )
        )

        # A custom or unavailable /kaggle/temp path must not prevent the persistent
        # workspace itself from being created. The fallback is still safe because
        # optional disk image caching is disabled by default on Kaggle.
        try:
            transient_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            fallback = (
                Path("/tmp/cad_attention_unet_transient")
                if SETTINGS_ATTENTION._ATTENTION_RUNNING_ON_KAGGLE
                else root / "_transient"
            )
            print(
                "[ATTENTION][STORAGE][WARNING] Could not create transient root "
                f"{transient_root}: {type(error).__name__}: {error}. "
                f"Falling back to {fallback}.",
                flush=True,
            )
            transient_root = fallback
            transient_root.mkdir(parents=True, exist_ok=True)

        # Standardized training images are regenerable and live only in the
        # configured transient cache directory.
        new_cached_images = transient_root / "training_images_256"

        automatic_masks = (
            transient_root / "automatic_masks"
            if SETTINGS_ATTENTION.ATTENTION_STORE_AUTOMATIC_MASKS_IN_TRANSIENT
            else root / "automatic_masks"
        )
        predicted_masks = (
            transient_root / "predicted_attention_masks"
            if SETTINGS_ATTENTION.ATTENTION_STORE_PREDICTED_MASKS_IN_TRANSIENT
            else root / "predicted_attention_masks"
        )

        workspace = AttentionWorkspace(
            root=root,
            transient_root=transient_root,
            automatic_masks=automatic_masks,
            manual_masks=root / "manual_masks",
            predicted_masks=predicted_masks,
            mask_overlays=root / "mask_overlays",
            cached_images=new_cached_images,
            checkpoints=root / "checkpoints",
            manifest_csv=root / "attention_unet_mask_manifest.csv",
            training_summary_json=root / "attention_unet_training_summary.json",
            console_log=root / "attention_unet_console.log",
            comparison_output=SETTINGS_PATHS.OUTPUT_DIR / "attention_unet_comparison",
        )
        for directory in (
            workspace.root,
            workspace.transient_root,
            workspace.automatic_masks,
            workspace.manual_masks,
            workspace.predicted_masks,
            workspace.mask_overlays,
            workspace.cached_images,
            workspace.checkpoints,
            workspace.comparison_output,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return workspace

    @staticmethod
    def _directory_scoped_relative_token(image_path):
        """Return a path token beginning at Directory_* and excluding class name."""

        parts = Path(image_path).parts
        for index, part in enumerate(parts):
            if str(part).startswith("Directory_"):
                return "/".join(map(str, parts[index:]))
        raise ValueError(f"No Directory_* component in path: {image_path}")

    @staticmethod
    def attention_image_token(image_path, patient_id, series_id):
        """Create a stable filename token without exposing Normal/Sick labels."""

        relative = _directory_scoped_relative_token(image_path)
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        safe_series = (
            str(series_id)
            .replace("/", "__")
            .replace("\\", "__")
            .replace(" ", "_")
        )
        return f"{patient_id}__{safe_series}__{digest}"

    @staticmethod
    def _evenly_spaced_subset(rows, maximum):
        """Select at most ``maximum`` rows across an already deterministic order."""

        rows = list(rows)
        if len(rows) <= maximum:
            return rows
        indices = np.linspace(0, len(rows) - 1, num=maximum)
        indices = np.unique(np.round(indices).astype(np.int64))
        if len(indices) < maximum:
            missing = maximum - len(indices)
            available = [i for i in range(len(rows)) if i not in set(indices.tolist())]
            indices = np.sort(np.concatenate([indices, np.asarray(available[:missing])]))
        return [rows[int(index)] for index in indices[:maximum]]

    @staticmethod
    def build_attention_patient_folds(samples):
        """Assign segmentation folds using only patient IDs, never CAD labels."""

        patient_ids = sorted({str(sample[2]) for sample in samples})
        ordered = sorted(
            patient_ids,
            key=lambda patient_id: hashlib.sha256(
                f"{SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED}|segmentation-fold|{patient_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        return {
            patient_id: int(index % SETTINGS_ATTENTION.ATTENTION_SEGMENTATION_FOLDS)
            for index, patient_id in enumerate(ordered)
        }

    @staticmethod
    def write_attention_manifest(rows, path):
        """Write the stable mask manifest atomically."""

        path = Path(path)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temporary, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=SETTINGS_ATTENTION.ATTENTION_MANIFEST_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            field: row.get(field, "")
                            for field in SETTINGS_ATTENTION.ATTENTION_MANIFEST_FIELDS
                        }
                    )
            os.replace(temporary, path)
        except Exception as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f"Could not write Attention mask manifest atomically: {path}. "
                f"{type(error).__name__}: {error}. {_storage_description(path)}"
            ) from error

    @staticmethod
    def read_attention_manifest(workspace):
        """Read a previously generated manifest and verify required columns."""

        if not workspace.manifest_csv.is_file():
            raise FileNotFoundError(
                f"Attention mask manifest not found: {workspace.manifest_csv}"
            )
        with open(workspace.manifest_csv, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            missing = sorted(set(SETTINGS_ATTENTION.ATTENTION_MANIFEST_FIELDS) - set(reader.fieldnames or []))
            if missing:
                raise RuntimeError(f"Attention manifest is missing columns: {missing}")
            return list(reader)

    @staticmethod
    def _load_attention_canvases(image_path):
        """Load one JPEG and return aligned 256/224 inputs without labels."""

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not decode MRI image: {image_path}")
        (
            _robust_canvas,
            monai_canvas,
            raw_canvas,
            padding_canvas,
            _features,
        ) = build_label_blind_standardized_image(image)
        raw_224 = cv2.resize(
            raw_canvas.astype(np.float32),
            (SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        padding_224 = cv2.resize(
            padding_canvas.astype(np.float32),
            (SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        content_224 = np.clip(1.0 - padding_224, 0.0, 1.0)
        return (
            monai_canvas.astype(np.float32),
            raw_224.astype(np.float32),
            content_224.astype(np.float32),
        )

    @staticmethod
    def _format_storage_bytes(value):
        """Return a compact binary-size string for storage diagnostics."""

        value = float(value)
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit = units[0]
        for candidate in units:
            unit = candidate
            if abs(value) < 1024.0 or candidate == units[-1]:
                break
            value /= 1024.0
        return f"{value:.2f} {unit}"

    @staticmethod
    def _existing_parent(path):
        """Return the nearest existing parent used for disk-usage inspection."""

        path = Path(path)
        candidate = path if path.is_dir() else path.parent
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    @staticmethod
    def _storage_description(path):
        """Describe total, used and free bytes for the filesystem containing path."""

        path = Path(path)
        try:
            usage = shutil.disk_usage(_existing_parent(path))
            return (
                f"filesystem={_existing_parent(path)}, "
                f"free={_format_storage_bytes(usage.free)}, "
                f"used={_format_storage_bytes(usage.used)}, "
                f"total={_format_storage_bytes(usage.total)}"
            )
        except OSError as error:
            return f"storage query failed: {type(error).__name__}: {error}"

    @staticmethod
    def _attention_warn_once(key, message):
        """Print one warning per process for repeated optional-cache failures."""

        key = str(key)
        if key in _ATTENTION_STORAGE_WARNING_KEYS:
            return
        _ATTENTION_STORAGE_WARNING_KEYS.add(key)
        print(message, flush=True)

    @staticmethod
    def report_attention_storage(workspace):
        """Print persistent/transient storage state before expensive mask work."""

        print(
            "[ATTENTION][STORAGE] Persistent workspace: "
            f"{workspace.root} | {_storage_description(workspace.root)}",
            flush=True,
        )
        print(
            "[ATTENTION][STORAGE] Transient cache root: "
            f"{workspace.transient_root} | "
            f"{_storage_description(workspace.transient_root)}",
            flush=True,
        )
        print(
            "[ATTENTION][STORAGE] standardized-image disk cache="
            f"{SETTINGS_ATTENTION.ATTENTION_CACHE_TRAINING_IMAGES}; RAM cache limit="
            f"{SETTINGS_ATTENTION.ATTENTION_RAM_IMAGE_CACHE_MB} MiB.",
            flush=True,
        )
        try:
            persistent_free = shutil.disk_usage(
                _existing_parent(workspace.root)
            ).free
            warning_threshold = (
                int(SETTINGS_ATTENTION.ATTENTION_PERSISTENT_FREE_SPACE_WARNING_MB) * 1024 * 1024
            )
            if persistent_free < warning_threshold:
                print(
                    "[ATTENTION][STORAGE][WARNING] Persistent free space is below "
                    f"{SETTINGS_ATTENTION.ATTENTION_PERSISTENT_FREE_SPACE_WARNING_MB} MiB. Automatic "
                    "masks and image cache are transient, but manual masks, five "
                    "checkpoints, logs and comparison outputs still require "
                    "/kaggle/working space. Remove obsolete feature-bank caches or "
                    "save them as a Kaggle Dataset before training.",
                    flush=True,
                )
        except OSError:
            pass

    @staticmethod
    def _atomic_cv2_write(
        path,
        image,
        *,
        required,
        purpose,
        png_compression=SETTINGS_ATTENTION.ATTENTION_PNG_COMPRESSION,
        optional_reserve_mb=0,
    ):
        """Encode with OpenCV, then write and atomically replace the final file.

        ``cv2.imwrite`` hides useful operating-system errors and often reports only
        ``libpng error: Write Error`` when a Kaggle quota is exhausted. Encoding in
        memory and writing through Python exposes ENOSPC/EDQUOT/EACCES, permits a
        same-directory temporary file, and avoids leaving a corrupt final PNG.

        Required scientific artifacts raise a diagnostic RuntimeError. Optional
        acceleration-cache writes return False so the caller can continue from the
        already decoded NumPy image.
        """

        path = Path(path)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            suffix = path.suffix.lower() or ".png"
            parameters = []
            if suffix == ".png":
                parameters = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
            elif suffix in {".jpg", ".jpeg"}:
                parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]

            encoded_ok, encoded = cv2.imencode(suffix, np.asarray(image), parameters)
            if not encoded_ok or encoded is None:
                raise RuntimeError(
                    f"OpenCV could not encode {purpose} as {suffix}: {path}"
                )
            payload = encoded.tobytes()

            if not required and optional_reserve_mb > 0:
                free = shutil.disk_usage(_existing_parent(path)).free
                reserve = int(optional_reserve_mb) * 1024 * 1024
                if free < len(payload) + reserve:
                    raise OSError(
                        28,
                        "optional cache skipped to preserve required-output space; "
                        f"free={_format_storage_bytes(free)}, "
                        f"reserve={_format_storage_bytes(reserve)}",
                    )

            temporary = path.with_name(
                f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            with open(temporary, "wb") as file:
                written = file.write(payload)
                if written != len(payload):
                    raise OSError(
                        28,
                        f"short write: wrote {written}/{len(payload)} bytes",
                    )
            os.replace(temporary, path)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"Atomic write produced an empty file: {path}")
            return True

        except Exception as error:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            details = (
                f"Could not save {purpose}: {path}. "
                f"{type(error).__name__}: {error}. "
                f"{_storage_description(path)}"
            )
            if required:
                raise RuntimeError(
                    details
                    + " Free space in /kaggle/working or set "
                    + "CAD_ATTENTION_UNET_WORK_ROOT to a writable location."
                ) from error
            _attention_warn_once(
                "optional-image-cache-write-disabled",
                "[ATTENTION][CACHE][WARNING] "
                + details
                + " Optional standardized-image disk caching has been disabled for "
                + "the rest of this process; preprocessing will continue from RAM/"
                + "source JPEGs.",
            )
            return False

    @staticmethod
    def _attention_ram_cache_get(key):
        """Return and refresh one LRU entry, or None when absent/disabled."""

        if SETTINGS_ATTENTION.ATTENTION_RAM_IMAGE_CACHE_MB <= 0:
            return None
        key = str(key)
        image = _ATTENTION_IMAGE_RAM_CACHE.pop(key, None)
        if image is None:
            return None
        _ATTENTION_IMAGE_RAM_CACHE[key] = image
        return image

    @staticmethod
    def _attention_ram_cache_put(key, image):
        """Insert one uint8 image and evict oldest entries above the byte limit."""

        global _ATTENTION_IMAGE_RAM_CACHE_BYTES
        limit = int(SETTINGS_ATTENTION.ATTENTION_RAM_IMAGE_CACHE_MB) * 1024 * 1024
        if limit <= 0:
            return
        key = str(key)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.nbytes > limit:
            return
        previous = _ATTENTION_IMAGE_RAM_CACHE.pop(key, None)
        if previous is not None:
            _ATTENTION_IMAGE_RAM_CACHE_BYTES -= int(previous.nbytes)
        image.setflags(write=False)
        _ATTENTION_IMAGE_RAM_CACHE[key] = image
        _ATTENTION_IMAGE_RAM_CACHE_BYTES += int(image.nbytes)
        while (
            _ATTENTION_IMAGE_RAM_CACHE
            and _ATTENTION_IMAGE_RAM_CACHE_BYTES > limit
        ):
            _old_key, old_image = _ATTENTION_IMAGE_RAM_CACHE.popitem(last=False)
            _ATTENTION_IMAGE_RAM_CACHE_BYTES -= int(old_image.nbytes)

    @staticmethod
    def clear_attention_image_ram_cache():
        """Release standardized-image RAM cache before full-dataset inference."""

        global _ATTENTION_IMAGE_RAM_CACHE_BYTES
        count = len(_ATTENTION_IMAGE_RAM_CACHE)
        released = int(_ATTENTION_IMAGE_RAM_CACHE_BYTES)
        _ATTENTION_IMAGE_RAM_CACHE.clear()
        _ATTENTION_IMAGE_RAM_CACHE_BYTES = 0
        if count:
            print(
                f"[ATTENTION][CACHE] Released {count} RAM-cached images "
                f"({_format_storage_bytes(released)}).",
                flush=True,
            )

    @staticmethod
    def load_attention_segmentation_image(row):
        """Return the standardized 256x256 uint8 image without requiring disk cache.

        Read order:

          1. in-process RAM LRU;
          2. valid optional PNG cache, when enabled;
          3. source JPEG plus label-blind standardization.

        Failure to write the optional PNG cache never aborts the pipeline. Corrupt or
        partial old cache files are removed and regenerated from the source image.
        """

        global _ATTENTION_OPTIONAL_DISK_CACHE_DISABLED_REASON

        token = str(row.get("image_token", row.get("image_path", "")))
        cached = _attention_ram_cache_get(token)
        if cached is not None:
            return cached

        path = Path(row["cached_image_path"])
        if SETTINGS_ATTENTION.ATTENTION_CACHE_TRAINING_IMAGES and path.is_file():
            disk_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if (
                disk_image is not None
                and disk_image.shape == (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE)
            ):
                disk_image = np.ascontiguousarray(disk_image, dtype=np.uint8)
                _attention_ram_cache_put(token, disk_image)
                return disk_image
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            _attention_warn_once(
                "corrupt-standardized-image-cache",
                "[ATTENTION][CACHE][WARNING] A corrupt/incomplete standardized "
                "training-image cache file was found and ignored. It will be "
                "regenerated from the source JPEG.",
            )

        monai_canvas, _raw_224, _content_224 = _load_attention_canvases(
            row["image_path"]
        )
        encoded = np.clip(
            np.round(monai_canvas * 255.0), 0, 255
        ).astype(np.uint8)
        _attention_ram_cache_put(token, encoded)

        if (
            SETTINGS_ATTENTION.ATTENTION_CACHE_TRAINING_IMAGES
            and _ATTENTION_OPTIONAL_DISK_CACHE_DISABLED_REASON is None
        ):
            written = _atomic_cv2_write(
                path,
                encoded,
                required=False,
                purpose="optional cached standardized training image",
                optional_reserve_mb=SETTINGS_ATTENTION.ATTENTION_OPTIONAL_DISK_CACHE_MIN_FREE_MB,
            )
            if not written:
                _ATTENTION_OPTIONAL_DISK_CACHE_DISABLED_REASON = (
                    "first optional cache write failed or reserve threshold was reached"
                )
        return encoded

    @staticmethod
    def _largest_connected_component(mask):
        """Keep the largest 8-connected foreground component of a binary mask."""

        mask = (np.asarray(mask) > 0).astype(np.uint8)
        if not np.any(mask):
            return mask
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if count <= 1:
            return mask
        foreground_areas = stats[1:, cv2.CC_STAT_AREA]
        selected_label = int(1 + np.argmax(foreground_areas))
        return (labels == selected_label).astype(np.uint8)

    @staticmethod
    def _attention_mask_file_is_readable(path):
        """Return True only for a non-empty decodable grayscale mask file."""

        path = Path(path)
        if not path.is_file():
            return False
        try:
            if path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return bool(mask is not None and mask.size > 0)

    @staticmethod
    def generate_attention_pseudo_masks(samples, workspace, monai_segmenter=None):
        """Generate label-blind MONAI pseudo masks for the training/review subset."""

        rows = select_attention_training_rows(samples, workspace)
        report_attention_storage(workspace)
        missing_indices = [
            index
            for index, row in enumerate(rows)
            if not _attention_mask_file_is_readable(row["automatic_mask_path"])
            or row.get("monai_valid", "") == ""
        ]
        if not missing_indices:
            print(
                f"[ATTENTION][MASKS] Reusing {len(rows)} existing pseudo masks.",
                flush=True,
            )
            return rows

        if monai_segmenter is None:
            monai_segmenter = build_monai_segmenter()
        monai_segmenter = monai_segmenter.to(DEVICE).eval()

        subset_rows = [rows[index] for index in missing_indices]
        loader = DataLoader(
            _AttentionManifestImageDataset(subset_rows),
            batch_size=SETTINGS_ATTENTION.ATTENTION_INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(DEVICE == "cuda"),
        )
        print(
            f"[ATTENTION][MASKS] Generating {len(subset_rows)} MONAI pseudo masks "
            f"from a label-blind subset of {len(rows)} images.",
            flush=True,
        )

        completed_batches = 0
        try:
            with torch.inference_mode():
                for batch_number, (images, local_indices) in enumerate(
                    tqdm(loader, desc="MONAI pseudo masks"),
                    start=1,
                ):
                    images = images.to(DEVICE, non_blocking=True)
                    logits = monai_segmenter(images)
                    if logits.ndim != 4 or logits.shape[1] != 4:
                        raise RuntimeError(
                            "Unexpected MONAI output while generating pseudo masks: "
                            f"{tuple(logits.shape)}"
                        )
                    probabilities = torch.softmax(logits.float(), dim=1)
                    heart_probability = probabilities[:, 1:].sum(dim=1, keepdim=True)
                    class_map = torch.argmax(probabilities, dim=1, keepdim=True)
                    hard = (class_map > 0).float()
                    area = hard.mean(dim=(1, 2, 3))
                    peak = heart_probability.amax(dim=(1, 2, 3))
                    valid = (
                        (area >= SETTINGS_MONAI.MONAI_MIN_HEART_AREA_RATIO)
                        & (area <= SETTINGS_MONAI.MONAI_MAX_HEART_AREA_RATIO)
                        & (peak >= SETTINGS_MONAI.MONAI_MIN_PEAK_HEART_PROBABILITY)
                    )
                    hard = F.max_pool2d(
                        hard,
                        kernel_size=SETTINGS_ATTENTION.ATTENTION_PSEUDO_MASK_DILATION_KERNEL,
                        stride=1,
                        padding=SETTINGS_ATTENTION.ATTENTION_PSEUDO_MASK_DILATION_KERNEL // 2,
                    )

                    for batch_position, subset_index_tensor in enumerate(local_indices):
                        subset_index = int(subset_index_tensor)
                        original_index = missing_indices[subset_index]
                        row = rows[original_index]
                        mask = _largest_connected_component(
                            hard[batch_position, 0].detach().cpu().numpy() > 0.5
                        )
                        output_path = Path(row["automatic_mask_path"])
                        _atomic_cv2_write(
                            output_path,
                            mask.astype(np.uint8) * 255,
                            required=True,
                            purpose="MONAI pseudo mask",
                        )
                        row["monai_valid"] = int(bool(valid[batch_position].item()))
                        row["monai_area_ratio"] = float(area[batch_position].item())
                        row["monai_peak_probability"] = float(
                            peak[batch_position].item()
                        )
                        row["manual_mask_exists"] = int(
                            Path(row["manual_mask_path"]).is_file()
                        )

                    completed_batches = int(batch_number)
                    if (
                        batch_number
                        % SETTINGS_ATTENTION.ATTENTION_MANIFEST_CHECKPOINT_EVERY_BATCHES
                        == 0
                    ):
                        write_attention_manifest(rows, workspace.manifest_csv)
                        print(
                            "[ATTENTION][MASKS] Manifest checkpoint saved after "
                            f"{batch_number}/{len(loader)} batches.",
                            flush=True,
                        )
        except Exception:
            # Preserve all successfully written mask metadata before propagating the
            # failure. A subsequent launch then resumes from the remaining rows.
            try:
                write_attention_manifest(rows, workspace.manifest_csv)
                print(
                    "[ATTENTION][MASKS] Progress manifest saved after failure at "
                    f"batch {completed_batches}/{len(loader)}.",
                    flush=True,
                )
            except Exception as manifest_error:
                print(
                    "[ATTENTION][MASKS][WARNING] Could not checkpoint the manifest "
                    f"after failure: {type(manifest_error).__name__}: "
                    f"{manifest_error}",
                    flush=True,
                )
            raise

        write_attention_manifest(rows, workspace.manifest_csv)
        valid_count = sum(int(str(row["monai_valid"])) for row in rows)
        print(
            f"[ATTENTION][MASKS] Pseudo-mask generation completed: "
            f"valid={valid_count}/{len(rows)}; manifest={workspace.manifest_csv}",
            flush=True,
        )
        return rows


class AttentionTrainingManager:
    """Attention U-Net training, checkpoints and segmentation inference helpers."""

    @staticmethod
    def _group_count(channels):
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    @staticmethod
    def soft_dice_coefficient_from_logits(logits, targets, epsilon=1e-6):
        probabilities = torch.sigmoid(logits.float())
        targets = targets.float()
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()

    @staticmethod
    def attention_segmentation_loss(logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
        dice_loss = 1.0 - soft_dice_coefficient_from_logits(logits, targets)
        return SETTINGS_ATTENTION.ATTENTION_BCE_WEIGHT * bce + SETTINGS_ATTENTION.ATTENTION_DICE_WEIGHT * dice_loss

    @staticmethod
    def _resolved_training_mask(row):
        """Return manual mask when present; otherwise a valid pseudo mask."""

        manual = Path(row["manual_mask_path"])
        if manual.is_file():
            return manual, "manual"
        automatic = Path(row["automatic_mask_path"])
        valid_text = str(row.get("monai_valid", "")).strip()
        if automatic.is_file() and valid_text in {"1", "True", "true"}:
            return automatic, "monai_pseudo"
        return None, None

    @staticmethod
    def _attention_training_fingerprint(rows, target_fold):
        """Fingerprint all masks and settings that can affect one checkpoint."""

        hasher = hashlib.sha256()
        settings = {
            "schema": SETTINGS_ATTENTION.ATTENTION_FEATURE_CACHE_SCHEMA,
            "target_fold": int(target_fold),
            "base_channels": SETTINGS_ATTENTION.ATTENTION_BASE_CHANNELS,
            "epochs": SETTINGS_ATTENTION.ATTENTION_EPOCHS,
            "patience": SETTINGS_ATTENTION.ATTENTION_EARLY_STOPPING_PATIENCE,
            "learning_rate": SETTINGS_ATTENTION.ATTENTION_LEARNING_RATE,
            "weight_decay": SETTINGS_ATTENTION.ATTENTION_WEIGHT_DECAY,
            "bce_weight": SETTINGS_ATTENTION.ATTENTION_BCE_WEIGHT,
            "dice_weight": SETTINGS_ATTENTION.ATTENTION_DICE_WEIGHT,
            "threshold": SETTINGS_ATTENTION.ATTENTION_MASK_THRESHOLD,
            "random_seed": SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED,
        }
        hasher.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
        for row in sorted(rows, key=lambda x: x["image_token"]):
            mask_path, source = _resolved_training_mask(row)
            if mask_path is None:
                continue
            hasher.update(row["image_token"].encode("utf-8"))
            hasher.update(str(source).encode("utf-8"))
            hasher.update(hashlib.sha256(Path(mask_path).read_bytes()).digest())
        return hasher.hexdigest()

    @staticmethod
    def _checkpoint_path(workspace, fold):
        return workspace.checkpoints / f"attention_unet_fold_{int(fold)}.pt"

    @staticmethod
    def _load_attention_state_dict(path, expected_fingerprint=None):
        checkpoint = torch.load(str(path), map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if expected_fingerprint is not None and checkpoint.get("fingerprint") != expected_fingerprint:
                raise RuntimeError("Checkpoint fingerprint does not match current masks/settings.")
            return checkpoint["state_dict"], checkpoint
        if isinstance(checkpoint, dict):
            return checkpoint, {"external_state_dict_only": True}
        raise RuntimeError(f"Unsupported Attention U-Net checkpoint object: {type(checkpoint)}")

    @staticmethod
    def _select_internal_validation_patients(patient_ids, target_fold):
        ordered = sorted(
            set(map(str, patient_ids)),
            key=lambda patient_id: hashlib.sha256(
                f"{SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED}|validation|{target_fold}|{patient_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        count = max(1, int(round(len(ordered) * SETTINGS_ATTENTION.ATTENTION_VALIDATION_PATIENT_FRACTION)))
        count = min(count, max(1, len(ordered) - 1))
        return set(ordered[:count])

    @staticmethod
    def train_attention_unet_crossfit(rows, workspace):
        """Train or reuse one patient-excluded Attention U-Net per target fold."""

        if SETTINGS_ATTENTION.ATTENTION_EXTERNAL_WEIGHTS:
            external_path = Path(SETTINGS_ATTENTION.ATTENTION_EXTERNAL_WEIGHTS)
            if not external_path.is_file():
                raise FileNotFoundError(
                    f"External Attention U-Net checkpoint not found: {external_path}"
                )
            state_dict, metadata = _load_attention_state_dict(external_path)
            model = AttentionUNet()
            model.load_state_dict(state_dict, strict=True)
            summary = {
                "status": "EXTERNAL_WEIGHTS",
                "external_weights": str(external_path),
                "external_metadata": metadata,
                "important_limitation": (
                    "The code verifies loadability, not whether the external weights "
                    "were trained independently of this CAD cohort."
                ),
            }
            workspace.training_summary_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            return {fold: external_path for fold in range(SETTINGS_ATTENTION.ATTENTION_SEGMENTATION_FOLDS)}

        eligible_rows = []
        source_counts = defaultdict(int)
        for row in rows:
            mask_path, source = _resolved_training_mask(row)
            if mask_path is not None:
                eligible_rows.append(row)
                source_counts[source] += 1
        if not eligible_rows:
            raise RuntimeError(
                "No valid manual/pseudo masks are available for Attention U-Net training."
            )

        checkpoint_map = {}
        fold_summaries = []
        for target_fold in range(SETTINGS_ATTENTION.ATTENTION_SEGMENTATION_FOLDS):
            candidate_rows = [
                row
                for row in eligible_rows
                if int(row["segmentation_fold"]) != target_fold
            ]
            candidate_patients = sorted({row["patient_id"] for row in candidate_rows})
            validation_patients = _select_internal_validation_patients(
                candidate_patients, target_fold
            )
            train_rows = [
                row for row in candidate_rows if row["patient_id"] not in validation_patients
            ]
            validation_rows = [
                row for row in candidate_rows if row["patient_id"] in validation_patients
            ]
            if not train_rows or not validation_rows:
                raise RuntimeError(
                    f"Fold {target_fold}: insufficient train/validation mask rows."
                )

            fingerprint = _attention_training_fingerprint(candidate_rows, target_fold)
            checkpoint_path = _checkpoint_path(workspace, target_fold)
            if checkpoint_path.is_file():
                try:
                    _state, metadata = _load_attention_state_dict(
                        checkpoint_path, expected_fingerprint=fingerprint
                    )
                    print(
                        f"[ATTENTION][TRAIN] Reusing fold {target_fold} checkpoint: "
                        f"{checkpoint_path}",
                        flush=True,
                    )
                    checkpoint_map[target_fold] = checkpoint_path
                    fold_summaries.append(metadata.get("summary", metadata))
                    continue
                except Exception as error:
                    print(
                        f"[ATTENTION][TRAIN] Rebuilding stale fold {target_fold} "
                        f"checkpoint: {error}",
                        flush=True,
                    )

            torch.manual_seed(SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED + target_fold)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED + target_fold)
            model = AttentionUNet().to(DEVICE)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=SETTINGS_ATTENTION.ATTENTION_LEARNING_RATE,
                weight_decay=SETTINGS_ATTENTION.ATTENTION_WEIGHT_DECAY,
            )
            amp_enabled = bool(SETTINGS_ATTENTION.ATTENTION_TRAIN_WITH_AMP and DEVICE == "cuda")
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            train_dataset = AttentionMaskTrainingDataset(
                train_rows,
                augment=True,
                seed=SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED + target_fold * 100,
            )
            validation_dataset = AttentionMaskTrainingDataset(
                validation_rows,
                augment=False,
                seed=SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED + target_fold * 100 + 1,
            )
            generator = torch.Generator()
            generator.manual_seed(SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED + target_fold)
            train_loader = DataLoader(
                train_dataset,
                batch_size=SETTINGS_ATTENTION.ATTENTION_BATCH_SIZE,
                shuffle=True,
                generator=generator,
                num_workers=0,
                pin_memory=(DEVICE == "cuda"),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=SETTINGS_ATTENTION.ATTENTION_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(DEVICE == "cuda"),
            )

            best_dice = -np.inf
            best_epoch = -1
            epochs_without_improvement = 0
            history = []
            best_state = None
            print(
                f"[ATTENTION][TRAIN] Fold {target_fold}: train_patients="
                f"{len(set(row['patient_id'] for row in train_rows))}, "
                f"validation_patients={len(validation_patients)}, "
                f"train_images={len(train_rows)}, validation_images={len(validation_rows)}.",
                flush=True,
            )

            for epoch in range(SETTINGS_ATTENTION.ATTENTION_EPOCHS):
                train_dataset.set_epoch(epoch)
                model.train()
                train_losses = []
                for images, masks, _tokens, _sources in train_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type="cuda" if DEVICE == "cuda" else "cpu",
                        enabled=amp_enabled,
                    ):
                        logits = model(images)
                        loss = attention_segmentation_loss(logits, masks)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    train_losses.append(float(loss.detach().cpu().item()))

                model.eval()
                validation_dice_values = []
                validation_losses = []
                with torch.inference_mode():
                    for images, masks, _tokens, _sources in validation_loader:
                        images = images.to(DEVICE, non_blocking=True)
                        masks = masks.to(DEVICE, non_blocking=True)
                        logits = model(images)
                        validation_losses.append(
                            float(attention_segmentation_loss(logits, masks).cpu().item())
                        )
                        validation_dice_values.append(
                            float(
                                soft_dice_coefficient_from_logits(logits, masks)
                                .cpu()
                                .item()
                            )
                        )
                mean_train_loss = float(np.mean(train_losses))
                mean_validation_loss = float(np.mean(validation_losses))
                mean_validation_dice = float(np.mean(validation_dice_values))
                history.append(
                    {
                        "epoch": int(epoch + 1),
                        "train_loss": mean_train_loss,
                        "validation_loss": mean_validation_loss,
                        "validation_soft_dice": mean_validation_dice,
                    }
                )
                print(
                    f"[ATTENTION][TRAIN] fold={target_fold}, epoch={epoch + 1}/"
                    f"{SETTINGS_ATTENTION.ATTENTION_EPOCHS}, train_loss={mean_train_loss:.5f}, "
                    f"val_loss={mean_validation_loss:.5f}, "
                    f"val_soft_dice={mean_validation_dice:.4f}",
                    flush=True,
                )
                if mean_validation_dice > best_dice + 1e-5:
                    best_dice = mean_validation_dice
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= SETTINGS_ATTENTION.ATTENTION_EARLY_STOPPING_PATIENCE:
                        print(
                            f"[ATTENTION][TRAIN] fold={target_fold} early stopping "
                            f"after epoch {epoch + 1}.",
                            flush=True,
                        )
                        break

            if best_state is None:
                raise RuntimeError(f"Fold {target_fold}: no valid checkpoint was produced.")
            fold_summary = {
                "target_segmentation_fold": int(target_fold),
                "train_patients": sorted({row["patient_id"] for row in train_rows}),
                "validation_patients": sorted(validation_patients),
                "excluded_target_patients": sorted(
                    {
                        row["patient_id"]
                        for row in eligible_rows
                        if int(row["segmentation_fold"]) == target_fold
                    }
                ),
                "train_images": int(len(train_rows)),
                "validation_images": int(len(validation_rows)),
                "best_epoch": int(best_epoch),
                "best_validation_soft_dice": float(best_dice),
                "history": history,
                "fingerprint": fingerprint,
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": best_state,
                    "fingerprint": fingerprint,
                    "model_config": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "base_channels": SETTINGS_ATTENTION.ATTENTION_BASE_CHANNELS,
                    },
                    "summary": fold_summary,
                },
                str(checkpoint_path),
            )
            checkpoint_map[target_fold] = checkpoint_path
            fold_summaries.append(fold_summary)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        summary = {
            "status": "OK",
            "segmentation_folds": int(SETTINGS_ATTENTION.ATTENTION_SEGMENTATION_FOLDS),
            "eligible_training_images": int(len(eligible_rows)),
            "mask_source_counts": dict(source_counts),
            "folds": fold_summaries,
            "methodological_note": (
                "Target-fold patients are excluded from both optimization and early "
                "stopping for the model that segments them. Internal validation uses "
                "only patients from the remaining folds. CAD labels are not supplied."
            ),
        }
        workspace.training_summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return checkpoint_map

    @staticmethod
    def _attention_binary_masks(probability_256):
        """Threshold and retain the largest connected component per image."""

        output = []
        for index in range(probability_256.shape[0]):
            mask = probability_256[index, 0].detach().cpu().numpy()
            component = _largest_connected_component(mask >= SETTINGS_ATTENTION.ATTENTION_MASK_THRESHOLD)
            output.append(torch.from_numpy(component.astype(np.float32)))
        return torch.stack(output, dim=0).unsqueeze(1).to(probability_256.device)

    @staticmethod
    def create_attention_exact_support_mask(hard_mask_224, valid_mask, content_mask):
        """Create one authoritative support for AU1/AU3/AU4/AU5."""

        dilated = F.max_pool2d(
            (hard_mask_224 > 0.5).float(),
            kernel_size=SETTINGS_ATTENTION.ATTENTION_SUPPORT_DILATION_KERNEL,
            stride=1,
            padding=SETTINGS_ATTENTION.ATTENTION_SUPPORT_DILATION_KERNEL // 2,
        )
        support = torch.zeros_like(dilated)
        height, width = support.shape[-2:]
        for index in range(support.shape[0]):
            if bool(valid_mask[index].item()) and bool(torch.any(dilated[index] > 0.5)):
                support[index] = dilated[index]
            else:
                top, bottom, left, right = _fixed_square_bounds(
                    height,
                    width,
                    (height - 1) / 2.0,
                    (width - 1) / 2.0,
                    SETTINGS_PREPROCESSING.HARD_SUPPORT_FALLBACK_FRACTION,
                )
                support[index, 0, top:bottom, left:right] = 1.0
        return (support * (content_mask > 0.5).float()).clamp(0.0, 1.0)

    @staticmethod
    def create_attention_support_shuffled_images(au1_images, support, decoded_hashes):
        """Preserve support and visible intensity multiset, destroy spatial assignment."""

        output = torch.zeros_like(au1_images)
        for index, decoded_hash in enumerate(decoded_hashes):
            flat_mask = support[index, 0].reshape(-1) > 0.5
            n_visible = int(flat_mask.sum().item())
            if n_visible == 0:
                continue
            values = au1_images[index, 0].reshape(-1)[flat_mask]
            multiplier, offset = _affine_permutation_parameters(decoded_hash, n_visible)
            positions = torch.arange(n_visible, device=values.device, dtype=torch.long)
            shuffled = values[(multiplier * positions + offset) % n_visible]
            for channel in range(3):
                output[index, channel].reshape(-1)[flat_mask] = shuffled
        return output


class AttentionEvaluationManager:
    """Attention patient feature bank, AU1-AU5 evaluation and MONAI comparison."""

    @staticmethod
    def _attention_feature_fingerprint(samples, checkpoint_map):
        hasher = hashlib.sha256()
        hasher.update(SETTINGS_ATTENTION.ATTENTION_FEATURE_CACHE_SCHEMA.encode("utf-8"))
        settings = {
            "mask_threshold": SETTINGS_ATTENTION.ATTENTION_MASK_THRESHOLD,
            "min_area": SETTINGS_ATTENTION.ATTENTION_MIN_HEART_AREA_RATIO,
            "max_area": SETTINGS_ATTENTION.ATTENTION_MAX_HEART_AREA_RATIO,
            "min_peak": SETTINGS_ATTENTION.ATTENTION_MIN_PEAK_PROBABILITY,
            "support_dilation": SETTINGS_ATTENTION.ATTENTION_SUPPORT_DILATION_KERNEL,
            "efficientnet_weights": SETTINGS_FEATURE_BANK.EFFICIENTNET_WEIGHTS_NAME,
            "feature_dim": SETTINGS_FEATURE_BANK.EFFICIENTNET_FEATURE_DIM,
        }
        hasher.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
        for fold in sorted(checkpoint_map):
            path = Path(checkpoint_map[fold])
            hasher.update(str(fold).encode("utf-8"))
            hasher.update(hashlib.sha256(path.read_bytes()).digest())
        for image_path, _label, patient_id, series_id in samples:
            stat = Path(image_path).stat()
            hasher.update(_directory_scoped_relative_token(image_path).encode("utf-8"))
            hasher.update(str(patient_id).encode("utf-8"))
            hasher.update(str(series_id).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def _dice_binary(first, second):
        first = np.asarray(first, dtype=bool)
        second = np.asarray(second, dtype=bool)
        denominator = int(first.sum() + second.sum())
        if denominator == 0:
            return 1.0
        return float(2.0 * np.logical_and(first, second).sum() / denominator)

    @staticmethod
    def _load_attention_model(checkpoint_path):
        state_dict, metadata = _load_attention_state_dict(checkpoint_path)
        model_config = metadata.get("model_config", {}) if isinstance(metadata, dict) else {}
        model = AttentionUNet(
            base_channels=int(model_config.get("base_channels", SETTINGS_ATTENTION.ATTENTION_BASE_CHANNELS))
        )
        model.load_state_dict(state_dict, strict=True)
        return model.to(DEVICE).eval(), metadata

    @staticmethod
    def extract_attention_patient_feature_bank(samples, workspace, checkpoint_map):
        """Infer cross-fit masks, encode AU1-AU5 and pool to patient vectors."""

        output_dir = workspace.comparison_output
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / "attention_unet_patient_feature_bank.npz"
        metadata_path = output_dir / "attention_unet_feature_bank_metadata.json"
        fingerprint = _attention_feature_fingerprint(samples, checkpoint_map)
        if npz_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                loaded = np.load(npz_path, allow_pickle=False)
                print(
                    f"[ATTENTION][FEATURES] Reusing patient feature bank: {npz_path}",
                    flush=True,
                )
                return {
                    "fingerprint": fingerprint,
                    "patient_ids": loaded["patient_ids"].astype(str),
                    "labels": loaded["labels"].astype(np.int64),
                    "features": {
                        mode: loaded[f"X__{mode}"].astype(np.float32)
                        for mode in SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES
                    },
                    "metadata": metadata,
                }

        patient_to_fold = build_attention_patient_folds(samples)
        review_rows = read_attention_manifest(workspace)
        review_by_token = {row["image_token"]: row for row in review_rows}
        review_tokens = set(review_by_token)
        pool = _StreamingHierarchicalAttentionPool(SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES)
        slice_qc_rows = []
        predicted_mask_writes_enabled = True
        encoder = FeatureExtractor().to(DEVICE).eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

        for target_fold in range(SETTINGS_ATTENTION.ATTENTION_SEGMENTATION_FOLDS):
            fold_samples = [
                sample
                for sample in samples
                if patient_to_fold[str(sample[2])] == target_fold
            ]
            if not fold_samples:
                continue
            model, checkpoint_metadata = _load_attention_model(
                checkpoint_map[target_fold]
            )
            loader = DataLoader(
                AttentionInferenceDataset(fold_samples),
                batch_size=SETTINGS_ATTENTION.ATTENTION_INFERENCE_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(DEVICE == "cuda"),
            )
            print(
                f"[ATTENTION][INFERENCE] fold={target_fold}, "
                f"patients={len(set(sample[2] for sample in fold_samples))}, "
                f"images={len(fold_samples)}.",
                flush=True,
            )
            with torch.inference_mode():
                for batch in tqdm(loader, desc=f"Attention fold {target_fold}"):
                    (
                        attention_inputs,
                        raw_images,
                        content_masks,
                        labels,
                        patient_ids,
                        series_ids,
                        sample_indices,
                        decoded_hashes,
                        image_tokens,
                    ) = batch
                    attention_inputs = attention_inputs.to(DEVICE, non_blocking=True)
                    raw_images = raw_images.to(DEVICE, non_blocking=True)
                    content_masks = content_masks.to(DEVICE, non_blocking=True)
                    logits = model(attention_inputs)
                    probability_256 = torch.sigmoid(logits.float())
                    hard_256 = _attention_binary_masks(probability_256)
                    area_ratio = hard_256.mean(dim=(1, 2, 3))
                    peak_probability = probability_256.amax(dim=(1, 2, 3))
                    valid_mask = (
                        (area_ratio >= SETTINGS_ATTENTION.ATTENTION_MIN_HEART_AREA_RATIO)
                        & (area_ratio <= SETTINGS_ATTENTION.ATTENTION_MAX_HEART_AREA_RATIO)
                        & (peak_probability >= SETTINGS_ATTENTION.ATTENTION_MIN_PEAK_PROBABILITY)
                    )
                    hard_224 = F.interpolate(
                        hard_256,
                        size=(SETTINGS_PREPROCESSING.IMG_SIZE, SETTINGS_PREPROCESSING.IMG_SIZE),
                        mode="nearest",
                    )
                    support = create_attention_exact_support_mask(
                        hard_224, valid_mask, content_masks
                    )
                    au1 = _robust_scale_visible_regions(raw_images, support)
                    au3 = support.repeat(1, 3, 1, 1)
                    au4 = create_attention_support_shuffled_images(
                        au1, support, list(decoded_hashes)
                    )
                    complement = (
                        (content_masks > 0.5) & (support <= 0.5)
                    ).float()
                    au5 = _robust_scale_visible_regions(raw_images, complement)
                    variants = {
                        SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[0]: au1,
                        SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[1]: au1,
                        SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[2]: au3,
                        SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[3]: au4,
                        SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[4]: au5,
                    }

                    embeddings_by_mode = {}
                    mode_list = list(variants)
                    for start in range(0, len(mode_list), SETTINGS_FEATURE_BANK.FEATURE_MODES_PER_ENCODER_CALL):
                        chunk_modes = mode_list[start:start + SETTINGS_FEATURE_BANK.FEATURE_MODES_PER_ENCODER_CALL]
                        normalized = torch.cat(
                            [normalize_for_efficientnet(variants[mode]) for mode in chunk_modes],
                            dim=0,
                        )
                        encoded = encoder(normalized).float().cpu().numpy()
                        batch_size = len(labels)
                        for chunk_index, mode in enumerate(chunk_modes):
                            embeddings_by_mode[mode] = encoded[
                                chunk_index * batch_size:(chunk_index + 1) * batch_size
                            ]

                    labels_np = labels.numpy().astype(np.int64)
                    valid_np = valid_mask.cpu().numpy().astype(bool)
                    for mode in SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES:
                        keep = valid_np if mode == SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES[1] else None
                        pool.add(
                            mode,
                            embeddings_by_mode[mode],
                            patient_ids,
                            series_ids,
                            labels_np,
                            keep=keep,
                        )

                    hard_cpu = hard_256[:, 0].cpu().numpy() > 0.5
                    support_fraction = support.mean(dim=(1, 2, 3)).cpu().numpy()
                    for index in range(len(labels_np)):
                        token = str(image_tokens[index])
                        review_row = review_by_token.get(token)
                        save_prediction = SETTINGS_ATTENTION.ATTENTION_SAVE_ALL_PREDICTED_MASKS or token in review_tokens
                        predicted_path = workspace.predicted_masks / f"{token}.png"
                        prediction_written = False
                        if save_prediction and predicted_mask_writes_enabled:
                            prediction_written = _atomic_cv2_write(
                                predicted_path,
                                hard_cpu[index].astype(np.uint8) * 255,
                                required=False,
                                purpose="optional Attention U-Net predicted mask",
                            )
                            if not prediction_written:
                                predicted_mask_writes_enabled = False
                                print(
                                    "[ATTENTION][PREDICTIONS][WARNING] Further "
                                    "optional predicted-mask PNG writes are disabled "
                                    "for this run. Feature extraction and AU1-AU5 "
                                    "evaluation will continue.",
                                    flush=True,
                                )
                        dice_monai = None
                        dice_manual = None
                        if review_row is not None:
                            automatic_path = Path(review_row["automatic_mask_path"])
                            manual_path = Path(review_row["manual_mask_path"])
                            if automatic_path.is_file():
                                target = cv2.imread(str(automatic_path), cv2.IMREAD_GRAYSCALE)
                                if target is not None:
                                    target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_NEAREST) > 127
                                    dice_monai = _dice_binary(hard_cpu[index], target)
                            if manual_path.is_file():
                                target = cv2.imread(str(manual_path), cv2.IMREAD_GRAYSCALE)
                                if target is not None:
                                    target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_NEAREST) > 127
                                    dice_manual = _dice_binary(hard_cpu[index], target)
                        slice_qc_rows.append(
                            {
                                "sample_index": int(sample_indices[index]),
                                "patient_id": str(patient_ids[index]),
                                "series_id": str(series_ids[index]),
                                "segmentation_fold": int(target_fold),
                                "attention_valid": int(valid_np[index]),
                                "attention_area_ratio": float(area_ratio[index].cpu().item()),
                                "attention_peak_probability": float(
                                    peak_probability[index].cpu().item()
                                ),
                                "final_support_fraction": float(support_fraction[index]),
                                "dice_vs_monai_pseudo_if_available": dice_monai,
                                "dice_vs_manual_if_available": dice_manual,
                                "predicted_mask_path": (
                                    str(predicted_path) if prediction_written else ""
                                ),
                                "image_token": token,
                                "checkpoint": str(checkpoint_map[target_fold]),
                            }
                        )
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        patient_ids = None
        labels = None
        features = {}
        for mode in SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES:
            X_mode, y_mode, patients_mode = pool.finalize(mode)
            if patient_ids is None:
                patient_ids = patients_mode
                labels = y_mode
            elif not np.array_equal(patient_ids, patients_mode) or not np.array_equal(labels, y_mode):
                raise RuntimeError(f"Patient alignment differs for {mode}.")
            features[mode] = X_mode

        np.savez_compressed(
            npz_path,
            patient_ids=np.asarray(patient_ids).astype(str),
            labels=np.asarray(labels, dtype=np.int64),
            **{f"X__{mode}": features[mode] for mode in SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES},
        )
        with open(
            output_dir / "attention_unet_slice_qc.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(slice_qc_rows[0]))
            writer.writeheader()
            writer.writerows(slice_qc_rows)

        patient_qc_rows = []
        qc_by_patient = defaultdict(list)
        for row in slice_qc_rows:
            qc_by_patient[row["patient_id"]].append(row)
        for patient_id in sorted(qc_by_patient):
            rows = qc_by_patient[patient_id]
            manual_dice = [
                float(row["dice_vs_manual_if_available"])
                for row in rows
                if row["dice_vs_manual_if_available"] not in (None, "")
            ]
            monai_dice = [
                float(row["dice_vs_monai_pseudo_if_available"])
                for row in rows
                if row["dice_vs_monai_pseudo_if_available"] not in (None, "")
            ]
            patient_qc_rows.append(
                {
                    "patient_id": patient_id,
                    "n_images": len(rows),
                    "valid_mask_rate": float(np.mean([row["attention_valid"] for row in rows])),
                    "mean_area_ratio": float(np.mean([row["attention_area_ratio"] for row in rows])),
                    "mean_support_fraction": float(np.mean([row["final_support_fraction"] for row in rows])),
                    "mean_dice_vs_monai_pseudo_if_available": (
                        float(np.mean(monai_dice)) if monai_dice else ""
                    ),
                    "mean_dice_vs_manual_if_available": (
                        float(np.mean(manual_dice)) if manual_dice else ""
                    ),
                }
            )
        with open(
            output_dir / "attention_unet_patient_qc.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(patient_qc_rows[0]))
            writer.writeheader()
            writer.writerows(patient_qc_rows)

        metadata = {
            "status": "OK",
            "fingerprint": fingerprint,
            "schema": SETTINGS_ATTENTION.ATTENTION_FEATURE_CACHE_SCHEMA,
            "n_patients": int(len(patient_ids)),
            "n_images": int(len(samples)),
            "feature_modes": list(SETTINGS_ATTENTION.ATTENTION_FEATURE_MODES),
            "storage_policy": {
                "persistent_workspace": str(workspace.root),
                "transient_root": str(workspace.transient_root),
                "automatic_masks": str(workspace.automatic_masks),
                "predicted_masks": str(workspace.predicted_masks),
                "cached_training_images": str(workspace.cached_images),
                "training_image_disk_cache_enabled": bool(
                    SETTINGS_ATTENTION.ATTENTION_CACHE_TRAINING_IMAGES
                ),
                "training_image_ram_cache_mb": int(
                    SETTINGS_ATTENTION.ATTENTION_RAM_IMAGE_CACHE_MB
                ),
            },
            "retained_slice_counts": dict(pool.retained_slice_counts),
            "patient_level_fallbacks": dict(pool.patient_level_fallbacks),
            "checkpoints": {str(key): str(value) for key, value in checkpoint_map.items()},
            "methodological_note": (
                "Each patient is segmented by the checkpoint associated with that "
                "patient's label-blind segmentation fold. With locally trained "
                "checkpoints, that patient's masks were excluded from optimization "
                "and early stopping."
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {
            "fingerprint": fingerprint,
            "patient_ids": patient_ids,
            "labels": labels,
            "features": features,
            "metadata": metadata,
        }

    @staticmethod
    def _load_or_create_classification_fold_manifest(samples, workspace):
        """Reuse patient folds when present; otherwise create the same contract."""

        candidate_paths = [SETTINGS_PATHS.OUTPUT_DIR / "manifests" / "patient_fold_manifest.csv"]
        candidate_paths.extend(
            sorted(
                SETTINGS_PATHS.OUTPUT_ROOT.glob(
                    "multi_experiment_suite__*/manifests/patient_fold_manifest.csv"
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        )
        for path in candidate_paths:
            if not path.is_file():
                continue
            with open(path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            if rows and {
                "patient_id",
                "true_label",
                "outer_fold",
                "duplicate_component_id",
            }.issubset(rows[0]):
                print(
                    f"[ATTENTION][CV] Reusing fold manifest: {path}",
                    flush=True,
                )
                return rows, path

        patient_to_label = {}
        for _path, label, patient_id, _series_id in samples:
            previous = patient_to_label.get(str(patient_id))
            if previous is not None and previous != int(label):
                raise RuntimeError(f"Inconsistent label for {patient_id}.")
            patient_to_label[str(patient_id)] = int(label)
        patients = np.asarray(sorted(patient_to_label))
        labels = np.asarray([patient_to_label[p] for p in patients], dtype=np.int64)
        groups = patients.copy()
        folds = assign_stratified_patient_folds(
            patients, labels, groups, SETTINGS_VALIDATION.N_SPLITS, SETTINGS_VALIDATION.CV_RANDOM_STATE
        )
        rows = [
            {
                "patient_id": str(patient_id),
                "true_label": int(label),
                "outer_fold": int(fold),
                "duplicate_component_id": str(patient_id),
            }
            for patient_id, label, fold in zip(patients, labels, folds)
        ]
        path = workspace.comparison_output / "attention_patient_fold_manifest.csv"
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return rows, path

    @staticmethod
    @contextmanager
    def _temporary_attention_analysis_settings():
        """Temporarily use Attention-specific repeated/permutation replicate counts."""

        original = (
            SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS,
            SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES,
            SETTINGS_VALIDATION.LABEL_PERMUTATION_RANDOM_STATE,
        )
        SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS = SETTINGS_ATTENTION.ATTENTION_REPEATED_CV_REPEATS
        SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES = SETTINGS_ATTENTION.ATTENTION_PERMUTATION_REPLICATES
        SETTINGS_VALIDATION.LABEL_PERMUTATION_RANDOM_STATE = SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED + 40_000
        try:
            yield
        finally:
            (
                SETTINGS_VALIDATION.REPEATED_NESTED_CV_REPEATS,
                SETTINGS_VALIDATION.LABEL_PERMUTATION_REPLICATES,
                SETTINGS_VALIDATION.LABEL_PERMUTATION_RANDOM_STATE,
            ) = original

    @staticmethod
    def _read_oof_prediction_csv(path):
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        rows.sort(key=lambda row: row["patient_id"])
        return (
            np.asarray([row["patient_id"] for row in rows]),
            np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64),
            np.asarray([float(row["oof_score"]) for row in rows], dtype=np.float64),
        )

    @staticmethod
    def _find_monai_candidate_predictions():
        direct = (
            SETTINGS_PATHS.OUTPUT_DIR
            / "experiments"
            / SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID
            / "patient_oof_predictions.csv"
        )
        if direct.is_file():
            return direct
        candidates = sorted(
            SETTINGS_PATHS.OUTPUT_ROOT.glob(
                "multi_experiment_suite__*/experiments/"
                f"{SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID}/patient_oof_predictions.csv"
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def evaluate_attention_feature_bank(samples, workspace, bank):
        """Run AU1-AU5 nested CV plus profile-controlled stability/permutation and MONAI comparison."""

        print(
            "[ATTENTION][VALIDATION] profile="
            f"{SETTINGS_VALIDATION.VALIDATION_RUNTIME_PROFILE}; stability="
            f"{SETTINGS_ATTENTION.ATTENTION_RUN_REPEATED_CV_STABILITY} × "
            f"{SETTINGS_ATTENTION.ATTENTION_REPEATED_CV_REPEATS} repeats for each AU model; "
            f"permutation={SETTINGS_ATTENTION.ATTENTION_RUN_PERMUTATION_TEST} × "
            f"{SETTINGS_ATTENTION.ATTENTION_PERMUTATION_REPLICATES} replicates.",
            flush=True,
        )

        fold_rows, fold_path = _load_or_create_classification_fold_manifest(
            samples, workspace
        )
        evaluation_root = workspace.comparison_output / "evaluation"
        evaluation_root.mkdir(parents=True, exist_ok=True)
        results = {}
        stability = {}
        prepared_by_id = {}

        with _temporary_attention_analysis_settings():
            for experiment in SETTINGS_ATTENTION.ATTENTION_EXPERIMENTS:
                X = bank["features"][experiment.feature_mode]
                prepared = {
                    "unit": "patient",
                    "X": np.asarray(X, dtype=np.float32),
                    "y": np.asarray(bank["labels"], dtype=np.int64),
                    "patient_ids": np.asarray(bank["patient_ids"]).astype(str),
                    "feature_names": tuple(
                        f"embedding_{index:04d}" for index in range(X.shape[1])
                    ),
                    "slice_filter": (
                        "attention_valid" if experiment.experiment_id.startswith("AU2_") else "all"
                    ),
                    "n_source_slices": int(len(samples)),
                    "n_retained_slices": int(
                        bank["metadata"]["retained_slice_counts"][experiment.feature_mode]
                    ),
                }
                prepared_by_id[experiment.experiment_id] = prepared
                result = run_one_experiment(
                    experiment,
                    prepared,
                    fold_rows,
                    evaluation_root / experiment.experiment_id,
                )
                results[experiment.experiment_id] = result
                if SETTINGS_ATTENTION.ATTENTION_RUN_REPEATED_CV_STABILITY:
                    stability[experiment.experiment_id] = (
                        run_repeated_nested_cv_stability(
                            experiment,
                            prepared,
                            fold_rows,
                            evaluation_root
                            / experiment.experiment_id
                            / "stability",
                        )
                    )
                else:
                    stability[experiment.experiment_id] = {
                        "status": "SKIPPED_DISABLED",
                        "experiment_id": experiment.experiment_id,
                        "configured_repeats": int(
                            SETTINGS_ATTENTION.ATTENTION_REPEATED_CV_REPEATS
                        ),
                    }

            au1_id = SETTINGS_ATTENTION.ATTENTION_EXPERIMENTS[0].experiment_id
            au1_result = results[au1_id]
            if SETTINGS_ATTENTION.ATTENTION_RUN_PERMUTATION_TEST:
                permutation = run_patient_label_permutation_test(
                    SETTINGS_ATTENTION.ATTENTION_EXPERIMENTS[0],
                    prepared_by_id[au1_id],
                    fold_rows,
                    float(au1_result["summary"]["metrics"]["auc"]),
                    evaluation_root / au1_id / "permutation",
                )
            else:
                permutation = {
                    "status": "SKIPPED_DISABLED",
                    "experiment_id": au1_id,
                    "configured_replicates": int(
                        SETTINGS_ATTENTION.ATTENTION_PERMUTATION_REPLICATES
                    ),
                }

        au1 = results[SETTINGS_ATTENTION.ATTENTION_EXPERIMENTS[0].experiment_id]
        pairwise = {}
        for experiment in SETTINGS_ATTENTION.ATTENTION_EXPERIMENTS[1:]:
            comparison = results[experiment.experiment_id]
            au1_order = np.argsort(au1["patient_ids"].astype(str))
            comparison_order = np.argsort(comparison["patient_ids"].astype(str))
            if not np.array_equal(
                au1["patient_ids"][au1_order].astype(str),
                comparison["patient_ids"][comparison_order].astype(str),
            ):
                raise RuntimeError(
                    f"Patient mismatch in AU1 versus {experiment.experiment_id}."
                )
            pairwise[experiment.experiment_id] = paired_auc_difference_interval(
                au1["labels"][au1_order],
                comparison["probabilities"][comparison_order],
                au1["probabilities"][au1_order],
                random_state=SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED
                + int(hashlib.sha256(experiment.experiment_id.encode()).hexdigest()[:8], 16),
            )

        monai_comparison = {"status": "UNAVAILABLE"}
        monai_path = _find_monai_candidate_predictions()
        if monai_path is not None:
            monai_patients, monai_labels, monai_scores = _read_oof_prediction_csv(monai_path)
            order = np.argsort(au1["patient_ids"].astype(str))
            au1_patients = au1["patient_ids"][order].astype(str)
            au1_labels = au1["labels"][order]
            au1_scores = au1["probabilities"][order]
            if np.array_equal(monai_patients, au1_patients) and np.array_equal(monai_labels, au1_labels):
                comparison = paired_auc_difference_interval(
                    au1_labels,
                    monai_scores,
                    au1_scores,
                    random_state=SETTINGS_ATTENTION.ATTENTION_RANDOM_SEED + 91_000,
                )
                monai_comparison = {
                    "status": "OK",
                    "monai_experiment_id": SETTINGS_EXPERIMENTS.PRIMARY_CANDIDATE_EXPERIMENT_ID,
                    "monai_prediction_csv": str(monai_path),
                    "attention_experiment_id": SETTINGS_ATTENTION.ATTENTION_EXPERIMENTS[0].experiment_id,
                    "monai_auc": float(roc_auc_score(monai_labels, monai_scores)),
                    "attention_auc": float(roc_auc_score(au1_labels, au1_scores)),
                    "attention_minus_monai": comparison,
                }
            else:
                monai_comparison = {
                    "status": "PATIENT_OR_LABEL_MISMATCH",
                    "monai_prediction_csv": str(monai_path),
                }

        summary = {
            "status": "OK",
            "pipeline_component_schema": "attention-current",
            "storage_policy": {
                "persistent_workspace": str(workspace.root),
                "transient_root": str(workspace.transient_root),
                "automatic_masks": str(workspace.automatic_masks),
                "predicted_masks": str(workspace.predicted_masks),
                "training_image_disk_cache_enabled": bool(
                    SETTINGS_ATTENTION.ATTENTION_CACHE_TRAINING_IMAGES
                ),
                "training_image_ram_cache_mb": int(
                    SETTINGS_ATTENTION.ATTENTION_RAM_IMAGE_CACHE_MB
                ),
            },
            "fold_manifest": str(fold_path),
            "feature_bank_fingerprint": bank["fingerprint"],
            "experiments": {
                experiment_id: result["summary"]
                for experiment_id, result in results.items()
            },
            "validation_runtime": {
                "profile": SETTINGS_VALIDATION.VALIDATION_RUNTIME_PROFILE,
                "run_repeated_cv": bool(
                    SETTINGS_ATTENTION.ATTENTION_RUN_REPEATED_CV_STABILITY
                ),
                "repeated_cv_repeats": int(
                    SETTINGS_ATTENTION.ATTENTION_REPEATED_CV_REPEATS
                ),
                "run_permutation": bool(SETTINGS_ATTENTION.ATTENTION_RUN_PERMUTATION_TEST),
                "permutation_replicates": int(
                    SETTINGS_ATTENTION.ATTENTION_PERMUTATION_REPLICATES
                ),
            },
            "repeated_nested_cv": stability,
            "au1_vs_controls": pairwise,
            "au1_patient_label_permutation": permutation,
            "au1_vs_monai_primary": monai_comparison,
            "methodological_limitation": (
                "If pseudo masks dominate supervision, Attention U-Net remains a "
                "cross-fitted distillation of MONAI rather than an independent expert "
                "segmentation ground truth."
            ),
        }
        (workspace.comparison_output / "attention_unet_comparison_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary


class MaskReviewManager:
    """Full-cohort mask manifests, review queues, HTML editor and review history."""

    @staticmethod
    def attention_full_review_manifest_path(workspace):
        return Path(workspace.root) / "attention_unet_full_review_manifest.csv"

    @staticmethod
    def attention_review_history_path(workspace):
        return Path(workspace.root) / "attention_unet_review_history.csv"

    @staticmethod
    def attention_prediction_generation_path(workspace):
        return Path(workspace.root) / "attention_unet_prediction_generation.json"

    @staticmethod
    def _review_atomic_csv(rows, path, fieldnames):
        """Write a CSV atomically without assuming the training-manifest schema."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with open(temporary, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(fieldnames))
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})
            os.replace(temporary, path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _review_read_csv_by_token(path):
        path = Path(path)
        if not path.is_file():
            return {}
        with open(path, newline="", encoding="utf-8") as file:
            return {
                str(row.get("image_token", "")): dict(row)
                for row in csv.DictReader(file)
                if str(row.get("image_token", "")).strip()
            }

    @staticmethod
    def _review_to_float(value, default=float("nan")):
        try:
            text = str(value).strip()
            if not text:
                return float(default)
            return float(text)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _review_to_int(value, default=-1):
        try:
            text = str(value).strip()
            if not text:
                return int(default)
            return int(float(text))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def build_attention_full_review_manifest(samples, workspace, refresh=False):
        """Build one label-blind review row for every discovered JPEG image.

        This manifest is separate from ``attention_unet_mask_manifest.csv``. The
        latter remains the compact segmentation-training manifest, while this file
        can contain all 63,425 images and is used only for inspection/correction.
        Existing MONAI/Attention QC values and manual-mask state are preserved.
        """

        path = attention_full_review_manifest_path(workspace)
        existing = {} if refresh else _review_read_csv_by_token(path)

        # Merge QC already known from the compact training manifest.
        compact = _review_read_csv_by_token(workspace.manifest_csv)
        for token, row in compact.items():
            existing.setdefault(token, {}).update(
                {
                    key: row.get(key, "")
                    for key in (
                        "monai_valid",
                        "monai_area_ratio",
                        "monai_peak_probability",
                    )
                }
            )

        # Merge QC from the most recent full-cohort Attention feature extraction.
        slice_qc_path = (
            Path(workspace.comparison_output) / "attention_unet_slice_qc.csv"
        )
        if slice_qc_path.is_file():
            with open(slice_qc_path, newline="", encoding="utf-8") as file:
                for qc in csv.DictReader(file):
                    token = str(qc.get("image_token", "")).strip()
                    if not token:
                        continue
                    existing.setdefault(token, {}).update(
                        {
                            "attention_valid": qc.get("attention_valid", ""),
                            "attention_area_ratio": qc.get(
                                "attention_area_ratio", ""
                            ),
                            "attention_peak_probability": qc.get(
                                "attention_peak_probability", ""
                            ),
                            "dice_attention_vs_monai": qc.get(
                                "dice_vs_monai_pseudo_if_available", ""
                            ),
                        }
                    )

        patient_to_fold = build_attention_patient_folds(samples)
        rows = []
        for image_path, _label, patient_id, series_id in samples:
            token = attention_image_token(image_path, patient_id, series_id)
            old = existing.get(token, {})
            manual_path = Path(workspace.manual_masks) / f"{token}.png"
            rows.append(
                {
                    "image_token": token,
                    "image_path": str(image_path),
                    "patient_id": str(patient_id),
                    "series_id": str(series_id),
                    "segmentation_fold": int(patient_to_fold[str(patient_id)]),
                    "cached_image_path": str(
                        Path(workspace.cached_images) / f"{token}.png"
                    ),
                    "automatic_mask_path": str(
                        Path(workspace.automatic_masks) / f"{token}.png"
                    ),
                    "manual_mask_path": str(manual_path),
                    "predicted_attention_mask_path": str(
                        Path(workspace.predicted_masks) / f"{token}.png"
                    ),
                    "monai_valid": old.get("monai_valid", ""),
                    "monai_area_ratio": old.get("monai_area_ratio", ""),
                    "monai_peak_probability": old.get(
                        "monai_peak_probability", ""
                    ),
                    "attention_valid": old.get("attention_valid", ""),
                    "attention_area_ratio": old.get(
                        "attention_area_ratio", ""
                    ),
                    "attention_peak_probability": old.get(
                        "attention_peak_probability", ""
                    ),
                    "dice_attention_vs_monai": old.get(
                        "dice_attention_vs_monai", ""
                    ),
                    "manual_mask_exists": int(manual_path.is_file()),
                    "attention_checkpoint_fingerprint": old.get(
                        "attention_checkpoint_fingerprint", ""
                    ),
                }
            )

        rows.sort(
            key=lambda row: (
                row["patient_id"],
                row["series_id"],
                row["image_token"],
            )
        )
        for index, row in enumerate(rows):
            row["review_index"] = int(index)
        _review_atomic_csv(rows, path, SETTINGS_REVIEW.ATTENTION_REVIEW_MANIFEST_FIELDS)
        print(
            f"[ATTENTION][REVIEW] Full manifest: {len(rows)} images -> {path}",
            flush=True,
        )
        return rows

    @staticmethod
    def read_attention_full_review_manifest(workspace):
        path = attention_full_review_manifest_path(workspace)
        if not path.is_file():
            raise FileNotFoundError(
                f"Full Attention review manifest not found: {path}"
            )
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        missing = sorted(
            set(SETTINGS_REVIEW.ATTENTION_REVIEW_MANIFEST_FIELDS)
            - set(rows[0].keys() if rows else ())
        )
        if missing:
            raise RuntimeError(f"Full review manifest missing columns: {missing}")
        return rows

    @staticmethod
    def _review_append_review_event(
        workspace,
        row,
        source,
        review_round,
        action,
        mask=None,
        queue_position=-1,
        queue_size=-1,
    ):
        """Append one persistent review action without rewriting 63k CSV rows."""

        path = attention_review_history_path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.is_file() or path.stat().st_size == 0
        area = ""
        if mask is not None:
            area = float((np.asarray(mask) > 0).mean())
        event = {
            "timestamp_utc": _review_datetime.now(_review_timezone.utc).isoformat(),
            "review_round": int(review_round),
            "review_source": str(source),
            "action": str(action),
            "image_token": row.get("image_token", ""),
            "patient_id": row.get("patient_id", ""),
            "series_id": row.get("series_id", ""),
            "queue_position": int(queue_position),
            "queue_size": int(queue_size),
            "mask_area_ratio": area,
            "automatic_mask_path": (
                row.get("predicted_attention_mask_path", "")
                if source == "attention"
                else row.get("automatic_mask_path", "")
            ),
            "manual_mask_path": row.get("manual_mask_path", ""),
        }
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(SETTINGS_REVIEW.ATTENTION_REVIEW_HISTORY_FIELDS)
            )
            if write_header:
                writer.writeheader()
            writer.writerow(event)

    @staticmethod
    def _review_reviewed_tokens(workspace, source=None, review_round=None):
        path = attention_review_history_path(workspace)
        if not path.is_file():
            return set()
        tokens = set()
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if source is not None and row.get("review_source") != source:
                    continue
                if review_round is not None and _review_to_int(
                    row.get("review_round"), -1
                ) != int(review_round):
                    continue
                if row.get("action") in {
                    "save_manual",
                    "accept_auto",
                    "skip",
                    "delete_manual",
                }:
                    tokens.add(row.get("image_token", ""))
        return tokens

    @staticmethod
    def _review_round_robin_diverse(rows, limit, source, seed):
        """Select a deterministic patient/series-balanced high-value queue."""

        rows = list(rows)
        if limit <= 0 or limit >= len(rows):
            return rows

        def priority(row):
            token = row["image_token"]
            hashed = hashlib.sha256(f"{seed}|{token}".encode()).hexdigest()
            if source == "attention":
                valid = _review_to_int(row.get("attention_valid"), -1)
                dice = _review_to_float(row.get("dice_attention_vs_monai"), 1.0)
                peak = _review_to_float(
                    row.get("attention_peak_probability"), 1.0
                )
                area = _review_to_float(row.get("attention_area_ratio"), 0.15)
            else:
                valid = _review_to_int(row.get("monai_valid"), -1)
                dice = 1.0
                peak = _review_to_float(row.get("monai_peak_probability"), 1.0)
                area = _review_to_float(row.get("monai_area_ratio"), 0.15)
            # invalid / low agreement / low confidence / extreme-area rows first.
            return (
                1 if bool(row.get("_reviewed_current_round", False)) else 0,
                0 if valid == 0 else 1,
                dice,
                peak,
                -abs(area - 0.15),
                hashed,
            )

        by_patient_series = _review_defaultdict(lambda: _review_defaultdict(list))
        for row in rows:
            by_patient_series[row["patient_id"]][row["series_id"]].append(row)
        patient_queues = {}
        for patient_id, series_map in by_patient_series.items():
            ordered_series = sorted(series_map)
            for series_id in ordered_series:
                series_map[series_id].sort(key=priority)
            queue = []
            position = 0
            while True:
                added = False
                for series_id in ordered_series:
                    if position < len(series_map[series_id]):
                        queue.append(series_map[series_id][position])
                        added = True
                if not added:
                    break
                position += 1
            patient_queues[patient_id] = queue

        patient_ids = sorted(patient_queues)
        selected = []
        position = 0
        while len(selected) < limit:
            added = False
            for patient_id in patient_ids:
                queue = patient_queues[patient_id]
                if position < len(queue):
                    selected.append(queue[position])
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            position += 1
        return selected

    @staticmethod
    def select_attention_review_rows(
        rows,
        workspace,
        source="monai",
        scope="diverse",
        limit=1200,
        seed=42,
        review_round=1,
    ):
        """Build a review queue from the full cohort without class labels."""

        source = str(source).lower()
        scope = str(scope).lower()
        if source not in {"monai", "attention"}:
            raise ValueError("review source must be 'monai' or 'attention'.")
        if scope not in SETTINGS_REVIEW.ATTENTION_REVIEW_SCOPES:
            raise ValueError(
                f"review scope must be one of {SETTINGS_REVIEW.ATTENTION_REVIEW_SCOPES}."
            )

        rows = [dict(row) for row in rows]
        reviewed = _review_reviewed_tokens(workspace, source=source)
        current_round_reviewed = _review_reviewed_tokens(
            workspace, source=source, review_round=review_round
        )

        if scope == "unreviewed":
            rows = [row for row in rows if row["image_token"] not in reviewed]
        elif scope == "reviewed":
            rows = [row for row in rows if row["image_token"] in reviewed]
        elif scope == "manual":
            rows = [
                row
                for row in rows
                if Path(row["manual_mask_path"]).is_file()
            ]
        elif scope == "invalid":
            field = "attention_valid" if source == "attention" else "monai_valid"
            rows = [row for row in rows if _review_to_int(row.get(field), -1) == 0]
        elif scope == "disagreement":
            if source != "attention":
                raise ValueError("disagreement scope is available for Attention only.")
            rows = [
                row
                for row in rows
                if np.isfinite(
                    _review_to_float(row.get("dice_attention_vs_monai"))
                )
            ]
            rows.sort(
                key=lambda row: (
                    _review_to_float(row.get("dice_attention_vs_monai"), 1.0),
                    row["patient_id"],
                    row["series_id"],
                    row["image_token"],
                )
            )

        if scope == "diverse":
            # Prefer rows not yet reviewed in the current round, then balance across
            # patients and series while prioritizing difficult segmentations.
            for row in rows:
                row["_reviewed_current_round"] = (
                    row["image_token"] in current_round_reviewed
                )
            rows = _review_round_robin_diverse(
                rows, int(limit), source, int(seed)
            )
        elif int(limit) > 0:
            rows = rows[: int(limit)]

        for index, row in enumerate(rows):
            row["queue_index"] = int(index)
        print(
            f"[ATTENTION][REVIEW] source={source}, scope={scope}, "
            f"round={review_round}, queue={len(rows)}, limit={limit}.",
            flush=True,
        )
        return rows

    @staticmethod
    def _review_manifest_image_loader(rows):
        return DataLoader(
            _AttentionManifestImageDataset(rows),
            batch_size=SETTINGS_ATTENTION.ATTENTION_INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(DEVICE == "cuda"),
        )

    @staticmethod
    def generate_monai_review_masks(
        samples,
        workspace,
        rows=None,
        force=False,
    ):
        """Generate/reuse MONAI masks for any full-cohort review rows."""

        full_rows = build_attention_full_review_manifest(samples, workspace)
        full_by_token = {row["image_token"]: row for row in full_rows}
        target_rows = (
            full_rows
            if rows is None
            else [full_by_token[row["image_token"]] for row in rows]
        )
        missing = [
            row
            for row in target_rows
            if force
            or not _attention_mask_file_is_readable(row["automatic_mask_path"])
            or str(row.get("monai_valid", "")).strip() == ""
        ]
        if not missing:
            print(
                f"[ATTENTION][MONAI REVIEW] Reusing {len(target_rows)} masks.",
                flush=True,
            )
            return target_rows

        report_attention_storage(workspace)
        model = build_monai_segmenter().to(DEVICE).eval()
        loader = _review_manifest_image_loader(missing)
        print(
            f"[ATTENTION][MONAI REVIEW] Generating {len(missing)} masks "
            f"for a {len(target_rows)}-row review queue.",
            flush=True,
        )
        completed = 0
        with torch.inference_mode():
            for batch_number, (images, local_indices) in enumerate(
                tqdm(loader, desc="MONAI full-review masks"), start=1
            ):
                images = images.to(DEVICE, non_blocking=True)
                logits = model(images)
                probabilities = torch.softmax(logits.float(), dim=1)
                heart_probability = probabilities[:, 1:].sum(dim=1, keepdim=True)
                class_map = torch.argmax(probabilities, dim=1, keepdim=True)
                hard = (class_map > 0).float()
                area = hard.mean(dim=(1, 2, 3))
                peak = heart_probability.amax(dim=(1, 2, 3))
                valid = (
                    (area >= SETTINGS_MONAI.MONAI_MIN_HEART_AREA_RATIO)
                    & (area <= SETTINGS_MONAI.MONAI_MAX_HEART_AREA_RATIO)
                    & (peak >= SETTINGS_MONAI.MONAI_MIN_PEAK_HEART_PROBABILITY)
                )
                hard = F.max_pool2d(
                    hard,
                    kernel_size=SETTINGS_ATTENTION.ATTENTION_PSEUDO_MASK_DILATION_KERNEL,
                    stride=1,
                    padding=SETTINGS_ATTENTION.ATTENTION_PSEUDO_MASK_DILATION_KERNEL // 2,
                )
                for batch_position, local_index_tensor in enumerate(local_indices):
                    row = missing[int(local_index_tensor)]
                    mask = _largest_connected_component(
                        hard[batch_position, 0].detach().cpu().numpy() > 0.5
                    )
                    _atomic_cv2_write(
                        Path(row["automatic_mask_path"]),
                        mask.astype(np.uint8) * 255,
                        required=True,
                        purpose="full-review MONAI mask",
                    )
                    row["monai_valid"] = int(valid[batch_position].item())
                    row["monai_area_ratio"] = float(area[batch_position].item())
                    row["monai_peak_probability"] = float(
                        peak[batch_position].item()
                    )
                    completed += 1
                if batch_number % 50 == 0:
                    _review_atomic_csv(
                        full_rows,
                        attention_full_review_manifest_path(workspace),
                        SETTINGS_REVIEW.ATTENTION_REVIEW_MANIFEST_FIELDS,
                    )
        _review_atomic_csv(
            full_rows,
            attention_full_review_manifest_path(workspace),
            SETTINGS_REVIEW.ATTENTION_REVIEW_MANIFEST_FIELDS,
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[ATTENTION][MONAI REVIEW] Completed {completed} masks.",
            flush=True,
        )
        refreshed = {row["image_token"]: row for row in full_rows}
        return [refreshed[row["image_token"]] for row in target_rows]

    @staticmethod
    def _review_checkpoint_fingerprint(checkpoint_map):
        hasher = hashlib.sha256()
        for fold in sorted(checkpoint_map):
            path = Path(checkpoint_map[fold])
            hasher.update(str(fold).encode("utf-8"))
            hasher.update(hashlib.sha256(path.read_bytes()).digest())
        return hasher.hexdigest()

    @staticmethod
    def generate_attention_review_masks(
        samples,
        workspace,
        checkpoint_map,
        rows=None,
        force=False,
    ):
        """Generate/reuse cross-fitted Attention masks for review, without EfficientNet."""

        full_rows = build_attention_full_review_manifest(samples, workspace)
        full_by_token = {row["image_token"]: row for row in full_rows}
        target_rows = (
            full_rows
            if rows is None
            else [full_by_token[row["image_token"]] for row in rows]
        )
        fingerprint = _review_checkpoint_fingerprint(checkpoint_map)
        metadata_path = attention_prediction_generation_path(workspace)
        previous_fingerprint = ""
        if metadata_path.is_file():
            try:
                previous_fingerprint = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                ).get("checkpoint_fingerprint", "")
            except Exception:
                previous_fingerprint = ""
        if previous_fingerprint != fingerprint:
            force = True

        missing = [
            row
            for row in target_rows
            if force
            or not _attention_mask_file_is_readable(
                row["predicted_attention_mask_path"]
            )
            or row.get("attention_checkpoint_fingerprint", "") != fingerprint
        ]
        if not missing:
            print(
                f"[ATTENTION][PREDICTION REVIEW] Reusing {len(target_rows)} "
                f"predictions for checkpoint {fingerprint[:12]}.",
                flush=True,
            )
            return target_rows

        by_fold = _review_defaultdict(list)
        for row in missing:
            by_fold[int(row["segmentation_fold"])].append(row)
        print(
            f"[ATTENTION][PREDICTION REVIEW] Generating {len(missing)} masks "
            f"for checkpoint {fingerprint[:12]}.",
            flush=True,
        )
        generated = 0
        for target_fold in sorted(by_fold):
            fold_rows = by_fold[target_fold]
            fold_samples = [
                (
                    row["image_path"],
                    0,
                    row["patient_id"],
                    row["series_id"],
                )
                for row in fold_rows
            ]
            model, _metadata = _load_attention_model(checkpoint_map[target_fold])
            loader = DataLoader(
                AttentionInferenceDataset(fold_samples),
                batch_size=SETTINGS_ATTENTION.ATTENTION_INFERENCE_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(DEVICE == "cuda"),
            )
            with torch.inference_mode():
                for batch_number, batch in enumerate(
                    tqdm(loader, desc=f"Attention review fold {target_fold}"),
                    start=1,
                ):
                    attention_inputs = batch[0].to(DEVICE, non_blocking=True)
                    image_tokens = list(batch[8])
                    probability = torch.sigmoid(model(attention_inputs).float())
                    hard = _attention_binary_masks(probability)
                    area = hard.mean(dim=(1, 2, 3))
                    peak = probability.amax(dim=(1, 2, 3))
                    valid = (
                        (area >= SETTINGS_ATTENTION.ATTENTION_MIN_HEART_AREA_RATIO)
                        & (area <= SETTINGS_ATTENTION.ATTENTION_MAX_HEART_AREA_RATIO)
                        & (peak >= SETTINGS_ATTENTION.ATTENTION_MIN_PEAK_PROBABILITY)
                    )
                    hard_np = hard[:, 0].cpu().numpy() > 0.5
                    for position, token in enumerate(image_tokens):
                        row = full_by_token[str(token)]
                        _atomic_cv2_write(
                            Path(row["predicted_attention_mask_path"]),
                            hard_np[position].astype(np.uint8) * 255,
                            required=True,
                            purpose="full-review Attention U-Net mask",
                        )
                        row["attention_valid"] = int(valid[position].item())
                        row["attention_area_ratio"] = float(area[position].item())
                        row["attention_peak_probability"] = float(
                            peak[position].item()
                        )
                        row["attention_checkpoint_fingerprint"] = fingerprint
                        monai_path = Path(row["automatic_mask_path"])
                        if monai_path.is_file():
                            monai = cv2.imread(
                                str(monai_path), cv2.IMREAD_GRAYSCALE
                            )
                            if monai is not None:
                                monai = cv2.resize(
                                    monai,
                                    (SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE, SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE),
                                    interpolation=cv2.INTER_NEAREST,
                                ) > 127
                                row["dice_attention_vs_monai"] = _dice_binary(
                                    hard_np[position], monai
                                )
                        generated += 1
                    if batch_number % 50 == 0:
                        _review_atomic_csv(
                            full_rows,
                            attention_full_review_manifest_path(workspace),
                            SETTINGS_REVIEW.ATTENTION_REVIEW_MANIFEST_FIELDS,
                        )
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        _review_atomic_csv(
            full_rows,
            attention_full_review_manifest_path(workspace),
            SETTINGS_REVIEW.ATTENTION_REVIEW_MANIFEST_FIELDS,
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "status": "OK",
                    "version": SETTINGS_REVIEW.ATTENTION_REVIEW_SCHEMA,
                    "checkpoint_fingerprint": fingerprint,
                    "generated_masks": int(generated),
                    "target_rows": int(len(target_rows)),
                    "timestamp_utc": _review_datetime.now(
                        _review_timezone.utc
                    ).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(
            f"[ATTENTION][PREDICTION REVIEW] Completed {generated} masks.",
            flush=True,
        )
        refreshed = {row["image_token"]: row for row in full_rows}
        return [refreshed[row["image_token"]] for row in target_rows]

    @staticmethod
    def select_attention_training_rows(samples, workspace):
        """Build compact pseudo supervision and include every manual correction.

        The ordinary per-series/per-patient limits continue to bound automatic
        pseudo supervision. Manual masks saved anywhere in the full review cohort
        are never discarded merely because their image was outside that subset.
        """

        existing = _review_read_csv_by_token(workspace.manifest_csv)
        review_existing = _review_read_csv_by_token(
            attention_full_review_manifest_path(workspace)
        )
        all_rows = {}
        by_series = _review_defaultdict(list)
        for image_path, _label, patient_id, series_id in samples:
            token = attention_image_token(image_path, patient_id, series_id)
            row = {
                "image_token": token,
                "image_path": str(image_path),
                "patient_id": str(patient_id),
                "series_id": str(series_id),
            }
            all_rows[token] = row
            by_series[str(series_id)].append(row)

        series_selected = []
        for series_id in sorted(by_series):
            ordered = sorted(
                by_series[series_id], key=lambda row: row["image_token"]
            )
            series_selected.extend(
                _evenly_spaced_subset(
                    ordered, SETTINGS_ATTENTION.ATTENTION_MAX_TRAIN_SLICES_PER_SERIES
                )
            )
        by_patient = _review_defaultdict(list)
        for row in series_selected:
            by_patient[row["patient_id"]].append(row)
        selected = {}
        for patient_id in sorted(by_patient):
            ordered = sorted(
                by_patient[patient_id], key=lambda row: row["image_token"]
            )
            for row in _evenly_spaced_subset(
                ordered, SETTINGS_ATTENTION.ATTENTION_MAX_TRAIN_SLICES_PER_PATIENT
            ):
                selected[row["image_token"]] = row
        base_count = len(selected)

        manual_added = 0
        for manual_path in Path(workspace.manual_masks).glob("*.png"):
            token = manual_path.stem
            if token in all_rows and token not in selected:
                selected[token] = all_rows[token]
                manual_added += 1

        patient_to_fold = build_attention_patient_folds(samples)
        manifest_rows = []
        for index, row in enumerate(
            sorted(selected.values(), key=lambda x: x["image_token"])
        ):
            token = row["image_token"]
            old = dict(review_existing.get(token, {}))
            old.update(existing.get(token, {}))
            automatic_path = Path(workspace.automatic_masks) / f"{token}.png"
            manual_path = Path(workspace.manual_masks) / f"{token}.png"
            predicted_path = Path(workspace.predicted_masks) / f"{token}.png"
            cached_path = Path(workspace.cached_images) / f"{token}.png"
            manifest_rows.append(
                {
                    "manifest_index": int(index),
                    "image_token": token,
                    "image_path": row["image_path"],
                    "patient_id": row["patient_id"],
                    "series_id": row["series_id"],
                    "segmentation_fold": int(
                        patient_to_fold[row["patient_id"]]
                    ),
                    "cached_image_path": str(cached_path),
                    "automatic_mask_path": str(automatic_path),
                    "manual_mask_path": str(manual_path),
                    "predicted_attention_mask_path": str(predicted_path),
                    "monai_valid": old.get("monai_valid", ""),
                    "monai_area_ratio": old.get("monai_area_ratio", ""),
                    "monai_peak_probability": old.get(
                        "monai_peak_probability", ""
                    ),
                    "manual_mask_exists": int(manual_path.is_file()),
                }
            )
        write_attention_manifest(manifest_rows, workspace.manifest_csv)
        print(
            f"[ATTENTION][TRAIN MANIFEST] automatic subset={base_count}; "
            f"manual rows added outside subset={manual_added}; "
            f"total={len(manifest_rows)}.",
            flush=True,
        )
        return manifest_rows

    @staticmethod
    def open_attention_mask_editor(
        workspace,
        start_index=0,
        brush_radius=8,
        base_source="attention",
        rows=None,
        review_round=1,
        review_scope="all",
    ):
        if rows is None:
            rows = read_attention_full_review_manifest(workspace)
        editor = AttentionMaskEditor(
            rows,
            workspace,
            start_index=start_index,
            brush_radius=brush_radius,
            base_source=base_source,
            review_round=review_round,
            review_scope=review_scope,
        )
        global _ACTIVE_ATTENTION_MASK_EDITOR
        _ACTIVE_ATTENTION_MASK_EDITOR = editor
        return editor.show()


class PipelineApplication:
    """Command-line parsing and staged CPU/GPU application entrypoints."""

    @staticmethod
    def _parse_attention_arguments(argv=None):
        """Parse one explicit staged action; combined execution is intentionally disabled."""

        parser = argparse.ArgumentParser(
            description=(
                "CAD MRI patient-level pipeline with staged GPU generation, "
                "CPU review and cached statistical evaluation."
            )
        )
        parser.add_argument(
            "--attention-action",
            choices=SETTINGS_REVIEW.ATTENTION_ACTION_CHOICES,
            required=True,
        )
        parser.add_argument("--attention-work-root", default=None)
        parser.add_argument("--attention-editor-index", type=int, default=0)
        parser.add_argument("--attention-brush-radius", type=int, default=8)
        parser.add_argument(
            "--attention-editor-base",
            choices=("attention", "monai"),
            default="attention",
        )
        parser.add_argument(
            "--attention-editor-scope",
            choices=SETTINGS_REVIEW.ATTENTION_REVIEW_SCOPES,
            default="diverse",
            help="all/diverse/unreviewed/invalid/disagreement/manual/reviewed",
        )
        parser.add_argument(
            "--attention-editor-limit",
            type=int,
            default=1200,
            help="0 means every available image; otherwise queue size cap.",
        )
        parser.add_argument("--attention-editor-seed", type=int, default=42)
        parser.add_argument("--attention-review-round", type=int, default=1)
        parser.add_argument(
            "--attention-dataset-path",
            default=str(SETTINGS_PATHS.DATASET_PATH),
        )
        parser.add_argument(
            "--runtime-device",
            choices=SETTINGS_RUNTIME.RUNTIME_DEVICE_CHOICES,
            default=os.environ.get("CAD_RUNTIME_DEVICE", "auto"),
            help=(
                "Use cuda for neural inference/training and cpu for review, CV, "
                "stability and permutation stages."
            ),
        )
        parser.add_argument(
            "--feature-cache-device-tag",
            choices=SETTINGS_RUNTIME.RUNTIME_DEVICE_CHOICES,
            default=os.environ.get("CAD_FEATURE_CACHE_DEVICE_TAG", "auto"),
            help="Device identity of the frozen feature cache.",
        )
        parser.add_argument(
            "--keep-gpu-memory",
            action="store_true",
            help="Do not clear CUDA caches after the selected action finishes.",
        )
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            print(
                f"[PIPELINE][CLI] Ignoring unrecognized notebook arguments: {unknown}",
                flush=True,
            )
        if args.attention_editor_limit < 0:
            raise ValueError("--attention-editor-limit must be >= 0.")
        if args.attention_review_round < 1:
            raise ValueError("--attention-review-round must be >= 1.")
        return args

    @staticmethod
    def run_attention_pipeline(samples, workspace, action):
        """Execute one neural or cached Attention/segmentation stage."""

        validate_attention_extension()

        if action == "evaluate-attention-cpu":
            checkpoint_map = existing_attention_checkpoint_map(workspace)
            bank = load_existing_attention_feature_bank(
                samples, workspace, checkpoint_map
            )
            return evaluate_attention_feature_bank(samples, workspace, bank)

        if action == "generate-all-monai-masks":
            rows = generate_monai_review_masks(
                samples, workspace, rows=None, force=False
            )
            return {
                "status": "ALL_MONAI_REVIEW_MASKS_READY",
                "n_rows": len(rows),
                "manifest": str(attention_full_review_manifest_path(workspace)),
            }

        if action not in {
            "retrain-regenerate-attention",
            "build-attention-feature-cache",
        }:
            raise ValueError(f"Unsupported Attention stage: {action!r}.")

        training_rows = generate_attention_pseudo_masks(samples, workspace)
        checkpoint_map = train_attention_unet_crossfit(training_rows, workspace)

        if action == "retrain-regenerate-attention":
            rows = generate_attention_review_masks(
                samples,
                workspace,
                checkpoint_map,
                rows=None,
                force=True,
            )
            clear_attention_image_ram_cache()
            return {
                "status": "ALL_ATTENTION_REVIEW_MASKS_READY",
                "n_rows": len(rows),
                "manifest": str(attention_full_review_manifest_path(workspace)),
            }

        clear_attention_image_ram_cache()
        bank = extract_attention_patient_feature_bank(
            samples, workspace, checkpoint_map
        )
        return {
            "status": "ATTENTION_FEATURE_CACHE_READY",
            "fingerprint": bank["fingerprint"],
            "n_patients": int(len(bank["patient_ids"])),
            "output": str(workspace.comparison_output),
        }

    @staticmethod
    def run_cad_pipeline(argv=None):
        """Dispatch one explicit CPU/GPU pipeline stage."""

        args = _parse_attention_arguments(argv)
        action = args.attention_action

        requested_device = args.runtime_device
        if action in SETTINGS_REVIEW.CPU_CACHE_ONLY_ACTIONS and requested_device == "auto":
            requested_device = "cpu"
        set_runtime_device(
            requested=requested_device,
            context=f"pipeline action {action}",
            strict_cuda=(requested_device == "cuda"),
        )
        set_feature_cache_device_tag(args.feature_cache_device_tag)

        if action == "build-monai-feature-cache":
            try:
                return build_monai_feature_cache_only(args.attention_dataset_path)
            finally:
                if not args.keep_gpu_memory:
                    release_gpu_resources(action)

        if action == "monai-cpu-from-cache":
            previous_cache_requirement = SETTINGS_FEATURE_BANK.REQUIRE_EXISTING_FEATURE_CACHE
            SETTINGS_FEATURE_BANK.REQUIRE_EXISTING_FEATURE_CACHE = True
            try:
                run_with_console_logging()
                return {
                    "status": "MONAI_CPU_EVALUATION_COMPLETED",
                    "device": DEVICE,
                    "feature_cache_device_tag": resolved_feature_cache_device_tag(),
                }
            finally:
                SETTINGS_FEATURE_BANK.REQUIRE_EXISTING_FEATURE_CACHE = previous_cache_requirement
                if not args.keep_gpu_memory:
                    release_gpu_resources(action)

        workspace = build_attention_workspace(args.attention_work_root)
        samples = load_samples(Path(args.attention_dataset_path))

        if action == "edit-existing-masks":
            validate_attention_extension()
            rows = build_attention_full_review_manifest(samples, workspace)
            queue = select_attention_review_rows(
                rows,
                workspace,
                source=args.attention_editor_base,
                scope=args.attention_editor_scope,
                limit=args.attention_editor_limit,
                seed=args.attention_editor_seed,
                review_round=args.attention_review_round,
            )
            queue = [
                row for row in queue
                if _review_mask_exists(row, args.attention_editor_base)
            ]
            if not queue:
                raise RuntimeError(
                    "No existing masks match this review queue. Run the corresponding "
                    "GPU generation stage first."
                )
            print(
                f"[ATTENTION][EDITOR][CPU] Opening {args.attention_editor_base} "
                f"row {args.attention_editor_index} of {len(queue)}; "
                f"scope={args.attention_editor_scope}; "
                f"round={args.attention_review_round}. Class labels are hidden.",
                flush=True,
            )
            return open_attention_mask_editor(
                workspace,
                start_index=args.attention_editor_index,
                brush_radius=args.attention_brush_radius,
                base_source=args.attention_editor_base,
                rows=queue,
                review_round=args.attention_review_round,
                review_scope=args.attention_editor_scope,
            )

        workspace.console_log.parent.mkdir(parents=True, exist_ok=True)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with open(
            workspace.console_log,
            "a",
            encoding="utf-8",
            buffering=1,
        ) as log_file:
            sys.stdout = TeeStream(original_stdout, log_file)
            sys.stderr = TeeStream(original_stderr, log_file)
            try:
                print("\n" + "#" * 100, flush=True)
                print("CAD CARDIAC MRI — STAGED PIPELINE", flush=True)
                print("#" * 100, flush=True)
                print(f"[PIPELINE] action={action}", flush=True)
                print(f"[PIPELINE] runtime_device={DEVICE}", flush=True)
                print(
                    "[PIPELINE] feature_cache_device_tag="
                    f"{resolved_feature_cache_device_tag()}",
                    flush=True,
                )
                print(f"[PIPELINE] workspace={workspace.root}", flush=True)
                summary = run_attention_pipeline(samples, workspace, action)
                print(
                    "[PIPELINE] Completed with status="
                    f"{summary.get('status', 'OK')}",
                    flush=True,
                )
                return summary
            except Exception:
                print("\n[PIPELINE] FATAL ERROR", flush=True)
                traceback.print_exc(file=sys.stdout)
                raise
            finally:
                if not args.keep_gpu_memory:
                    release_gpu_resources(action)
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdout = original_stdout
                sys.stderr = original_stderr


# endregion

# region BACKWARD-COMPATIBLE MODULE API
# Existing internal calls and notebooks continue to use these names.
_resolve_requested_runtime_device = RuntimeManager._resolve_requested_runtime_device
set_runtime_device = RuntimeManager.set_runtime_device
refresh_runtime_device = RuntimeManager.refresh_runtime_device
release_gpu_resources = RuntimeManager.release_gpu_resources
set_feature_cache_device_tag = RuntimeManager.set_feature_cache_device_tag
resolved_feature_cache_device_tag = RuntimeManager.resolved_feature_cache_device_tag
_synchronize_timing_device = RuntimeManager._synchronize_timing_device
_format_elapsed_time = RuntimeManager._format_elapsed_time
_print_stage_start = RuntimeManager._print_stage_start
_print_stage_complete = RuntimeManager._print_stage_complete
_print_detail = RuntimeManager._print_detail
_print_timing_summary = RuntimeManager._print_timing_summary
run_with_console_logging = RuntimeManager.run_with_console_logging

get_enabled_experiments = ConfigurationManager.get_enabled_experiments
validate_configuration = ConfigurationManager.validate_configuration
collect_suite_metadata = ConfigurationManager.collect_suite_metadata
write_suite_configuration = ConfigurationManager.write_suite_configuration
validate_attention_extension = ConfigurationManager.validate_attention_extension

scale_intensity_0_1 = ImagePreprocessor.scale_intensity_0_1
_is_dark_uniform_edge_line = ImagePreprocessor._is_dark_uniform_edge_line
detect_label_blind_dark_padding_bounds = ImagePreprocessor.detect_label_blind_dark_padding_bounds
robust_scale_intensity_0_1 = ImagePreprocessor.robust_scale_intensity_0_1
_fixed_content_canvas_geometry = ImagePreprocessor._fixed_content_canvas_geometry
resize_to_fixed_content_canvas = ImagePreprocessor.resize_to_fixed_content_canvas
build_fixed_content_padding_canvas = ImagePreprocessor.build_fixed_content_padding_canvas
build_label_blind_standardized_image = ImagePreprocessor.build_label_blind_standardized_image
compute_dct_perceptual_hash = ImagePreprocessor.compute_dct_perceptual_hash
compute_image_provenance_features = ImagePreprocessor.compute_image_provenance_features

load_samples = DatasetManager.load_samples

sha256_file = MonaiSegmenter.sha256_file
locate_monai_bundle_root = MonaiSegmenter.locate_monai_bundle_root
validate_monai_bundle_metadata = MonaiSegmenter.validate_monai_bundle_metadata
validate_monai_train_config = MonaiSegmenter.validate_monai_train_config
verify_monai_artifact_sha256 = MonaiSegmenter.verify_monai_artifact_sha256
ensure_monai_bundle = MonaiSegmenter.ensure_monai_bundle
load_and_validate_torchscript_segmenter = MonaiSegmenter.load_and_validate_torchscript_segmenter
load_checkpoint_state_dict = MonaiSegmenter.load_checkpoint_state_dict
build_monai_segmenter = MonaiSegmenter.build_monai_segmenter
predict_monai_heart_masks = MonaiSegmenter.predict_monai_heart_masks
apply_confidence_gated_soft_roi = MonaiSegmenter.apply_confidence_gated_soft_roi
normalize_for_efficientnet = MonaiSegmenter.normalize_for_efficientnet

required_efficientnet_feature_modes = FeatureBankManager.required_efficientnet_feature_modes
feature_bank_fingerprint = FeatureBankManager.feature_bank_fingerprint
_feature_bank_shared_paths = FeatureBankManager._feature_bank_shared_paths
_feature_mode_path = FeatureBankManager._feature_mode_path
load_feature_bank = FeatureBankManager.load_feature_bank
create_fixed_fraction_center_crop_images = FeatureBankManager.create_fixed_fraction_center_crop_images
apply_zero_background_roi_with_fixed_center_fallback = FeatureBankManager.apply_zero_background_roi_with_fixed_center_fallback
_fixed_square_bounds = FeatureBankManager._fixed_square_bounds
_robust_scale_visible_regions = FeatureBankManager._robust_scale_visible_regions
create_a17_exact_support_mask = FeatureBankManager.create_a17_exact_support_mask
create_hard_support_region_normalized_images = FeatureBankManager.create_hard_support_region_normalized_images
create_a17_exact_support_mask_only_images = FeatureBankManager.create_a17_exact_support_mask_only_images
_affine_permutation_parameters = FeatureBankManager._affine_permutation_parameters
create_a17_support_intensity_affine_shuffled_images = FeatureBankManager.create_a17_support_intensity_affine_shuffled_images
create_a17_exact_support_complement_region_normalized_images = FeatureBankManager.create_a17_exact_support_complement_region_normalized_images
run_exact_support_transform_contract_self_test = FeatureBankManager.run_exact_support_transform_contract_self_test
_fixed_central_square_mask = FeatureBankManager._fixed_central_square_mask
create_fixed_periphery_region_normalized_images = FeatureBankManager.create_fixed_periphery_region_normalized_images
extract_feature_bank = FeatureBankManager.extract_feature_bank
load_or_extract_feature_bank = FeatureBankManager.load_or_extract_feature_bank

audit_exact_decoded_pixel_duplicates = DuplicateAuditManager.audit_exact_decoded_pixel_duplicates
_prepare_phash_review_tile = DuplicateAuditManager._prepare_phash_review_tile
write_phash_review_panels = DuplicateAuditManager.write_phash_review_panels
audit_perceptual_near_duplicate_candidates = DuplicateAuditManager.audit_perceptual_near_duplicate_candidates
build_patient_label_table = DuplicateAuditManager.build_patient_label_table
build_duplicate_component_map = DuplicateAuditManager.build_duplicate_component_map
assign_stratified_patient_folds = DuplicateAuditManager.assign_stratified_patient_folds
build_patient_fold_manifest = DuplicateAuditManager.build_patient_fold_manifest
write_patient_fold_manifest = DuplicateAuditManager.write_patient_fold_manifest
write_cohort_manifest = DuplicateAuditManager.write_cohort_manifest
_patient_indices = DuplicateAuditManager._patient_indices

aggregate_patient_provenance_features = PatientDataManager.aggregate_patient_provenance_features
aggregate_patient_standardization_features = PatientDataManager.aggregate_patient_standardization_features
aggregate_patient_standardized_monai_qc_features = PatientDataManager.aggregate_patient_standardized_monai_qc_features
bootstrap_difference_in_means = PatientDataManager.bootstrap_difference_in_means
write_standardized_monai_qc_outputs = PatientDataManager.write_standardized_monai_qc_outputs
write_patient_tabular_features = PatientDataManager.write_patient_tabular_features
aggregate_patient_embeddings = PatientDataManager.aggregate_patient_embeddings
compute_balanced_patient_weights = PatientDataManager.compute_balanced_patient_weights
experiment_preparation_cache_key = PatientDataManager.experiment_preparation_cache_key
prepare_experiment_data = PatientDataManager.prepare_experiment_data

build_patient_classifier = ModelingManager.build_patient_classifier
fit_patient_classifier = ModelingManager.fit_patient_classifier
patient_model_raw_scores = ModelingManager.patient_model_raw_scores
fit_predict_raw_for_patient_sets = ModelingManager.fit_predict_raw_for_patient_sets
create_inner_fold_assignment = ModelingManager.create_inner_fold_assignment
collect_inner_oof_raw_scores = ModelingManager.collect_inner_oof_raw_scores
select_decision_threshold = ModelingManager.select_decision_threshold
experiment_candidate_c_values = ModelingManager.experiment_candidate_c_values
select_c_and_training_threshold = ModelingManager.select_c_and_training_threshold
_safe_divide = ModelingManager._safe_divide
compute_binary_patient_metrics = ModelingManager.compute_binary_patient_metrics
bootstrap_patient_metric_intervals = ModelingManager.bootstrap_patient_metric_intervals
run_one_experiment = ModelingManager.run_one_experiment
_build_fold_manifest_rows_for_seed = ModelingManager._build_fold_manifest_rows_for_seed
run_nested_cv_auc_only = ModelingManager.run_nested_cv_auc_only
audit_exact_support_prepared_row_contract = ModelingManager.audit_exact_support_prepared_row_contract
run_model_family_nested_selection = ModelingManager.run_model_family_nested_selection
run_repeated_nested_cv_stability = ModelingManager.run_repeated_nested_cv_stability
write_repeated_stability_ranking = ModelingManager.write_repeated_stability_ranking
print_repeated_stability_ranking = ModelingManager.print_repeated_stability_ranking
compare_repeated_nested_cv_pairs = ModelingManager.compare_repeated_nested_cv_pairs
run_patient_label_permutation_test = ModelingManager.run_patient_label_permutation_test
run_selection_adjusted_candidate_family_permutation_test = ModelingManager.run_selection_adjusted_candidate_family_permutation_test
paired_auc_difference_interval = ModelingManager.paired_auc_difference_interval
compare_experiments_to_baseline = ModelingManager.compare_experiments_to_baseline
compare_predeclared_ablation_pairs = ModelingManager.compare_predeclared_ablation_pairs
print_primary_ablation_comparisons = ModelingManager.print_primary_ablation_comparisons
result_summary_row = ModelingManager.result_summary_row
write_master_outputs = ModelingManager.write_master_outputs
print_final_comparison = ModelingManager.print_final_comparison

write_tabular_class_summary = ReportingManager.write_tabular_class_summary

main = PipelineRunner.main
build_monai_feature_cache_only = PipelineRunner.build_monai_feature_cache_only
existing_attention_checkpoint_map = PipelineRunner.existing_attention_checkpoint_map
load_existing_attention_feature_bank = PipelineRunner.load_existing_attention_feature_bank
_review_mask_exists = PipelineRunner._review_mask_exists

build_attention_workspace = AttentionDataManager.build_attention_workspace
_directory_scoped_relative_token = AttentionDataManager._directory_scoped_relative_token
attention_image_token = AttentionDataManager.attention_image_token
_evenly_spaced_subset = AttentionDataManager._evenly_spaced_subset
build_attention_patient_folds = AttentionDataManager.build_attention_patient_folds
write_attention_manifest = AttentionDataManager.write_attention_manifest
read_attention_manifest = AttentionDataManager.read_attention_manifest
_load_attention_canvases = AttentionDataManager._load_attention_canvases
_format_storage_bytes = AttentionDataManager._format_storage_bytes
_existing_parent = AttentionDataManager._existing_parent
_storage_description = AttentionDataManager._storage_description
_attention_warn_once = AttentionDataManager._attention_warn_once
report_attention_storage = AttentionDataManager.report_attention_storage
_atomic_cv2_write = AttentionDataManager._atomic_cv2_write
_attention_ram_cache_get = AttentionDataManager._attention_ram_cache_get
_attention_ram_cache_put = AttentionDataManager._attention_ram_cache_put
clear_attention_image_ram_cache = AttentionDataManager.clear_attention_image_ram_cache
load_attention_segmentation_image = AttentionDataManager.load_attention_segmentation_image
_largest_connected_component = AttentionDataManager._largest_connected_component
_attention_mask_file_is_readable = AttentionDataManager._attention_mask_file_is_readable
generate_attention_pseudo_masks = AttentionDataManager.generate_attention_pseudo_masks

_group_count = AttentionTrainingManager._group_count
soft_dice_coefficient_from_logits = AttentionTrainingManager.soft_dice_coefficient_from_logits
attention_segmentation_loss = AttentionTrainingManager.attention_segmentation_loss
_resolved_training_mask = AttentionTrainingManager._resolved_training_mask
_attention_training_fingerprint = AttentionTrainingManager._attention_training_fingerprint
_checkpoint_path = AttentionTrainingManager._checkpoint_path
_load_attention_state_dict = AttentionTrainingManager._load_attention_state_dict
_select_internal_validation_patients = AttentionTrainingManager._select_internal_validation_patients
train_attention_unet_crossfit = AttentionTrainingManager.train_attention_unet_crossfit
_attention_binary_masks = AttentionTrainingManager._attention_binary_masks
create_attention_exact_support_mask = AttentionTrainingManager.create_attention_exact_support_mask
create_attention_support_shuffled_images = AttentionTrainingManager.create_attention_support_shuffled_images

_attention_feature_fingerprint = AttentionEvaluationManager._attention_feature_fingerprint
_dice_binary = AttentionEvaluationManager._dice_binary
_load_attention_model = AttentionEvaluationManager._load_attention_model
extract_attention_patient_feature_bank = AttentionEvaluationManager.extract_attention_patient_feature_bank
_load_or_create_classification_fold_manifest = AttentionEvaluationManager._load_or_create_classification_fold_manifest
_temporary_attention_analysis_settings = AttentionEvaluationManager._temporary_attention_analysis_settings
_read_oof_prediction_csv = AttentionEvaluationManager._read_oof_prediction_csv
_find_monai_candidate_predictions = AttentionEvaluationManager._find_monai_candidate_predictions
evaluate_attention_feature_bank = AttentionEvaluationManager.evaluate_attention_feature_bank

attention_full_review_manifest_path = MaskReviewManager.attention_full_review_manifest_path
attention_review_history_path = MaskReviewManager.attention_review_history_path
attention_prediction_generation_path = MaskReviewManager.attention_prediction_generation_path
_review_atomic_csv = MaskReviewManager._review_atomic_csv
_review_read_csv_by_token = MaskReviewManager._review_read_csv_by_token
_review_to_float = MaskReviewManager._review_to_float
_review_to_int = MaskReviewManager._review_to_int
build_attention_full_review_manifest = MaskReviewManager.build_attention_full_review_manifest
read_attention_full_review_manifest = MaskReviewManager.read_attention_full_review_manifest
_review_append_review_event = MaskReviewManager._review_append_review_event
_review_reviewed_tokens = MaskReviewManager._review_reviewed_tokens
_review_round_robin_diverse = MaskReviewManager._review_round_robin_diverse
select_attention_review_rows = MaskReviewManager.select_attention_review_rows
_review_manifest_image_loader = MaskReviewManager._review_manifest_image_loader
generate_monai_review_masks = MaskReviewManager.generate_monai_review_masks
_review_checkpoint_fingerprint = MaskReviewManager._review_checkpoint_fingerprint
generate_attention_review_masks = MaskReviewManager.generate_attention_review_masks
select_attention_training_rows = MaskReviewManager.select_attention_training_rows
open_attention_mask_editor = MaskReviewManager.open_attention_mask_editor

_parse_attention_arguments = PipelineApplication._parse_attention_arguments
run_attention_pipeline = PipelineApplication.run_attention_pipeline
run_cad_pipeline = PipelineApplication.run_cad_pipeline

# endregion

__all__ = [
    "SETTINGS",
    "SETTINGS_BOOTSTRAP",
    "SETTINGS_RUNTIME",
    "SETTINGS_EXPERIMENTS",
    "SETTINGS_VALIDATION",
    "SETTINGS_PREPROCESSING",
    "SETTINGS_FEATURE_BANK",
    "SETTINGS_AUDIT",
    "SETTINGS_MONAI",
    "SETTINGS_ATTENTION",
    "SETTINGS_REVIEW",
    "SETTINGS_REPORTING",
    "SETTINGS_PATHS",
    "RuntimeManager",
    "ConfigurationManager",
    "ImagePreprocessor",
    "DatasetManager",
    "MonaiSegmenter",
    "FeatureBankManager",
    "DuplicateAuditManager",
    "PatientDataManager",
    "ModelingManager",
    "ReportingManager",
    "PipelineRunner",
    "AttentionDataManager",
    "AttentionTrainingManager",
    "AttentionEvaluationManager",
    "MaskReviewManager",
    "PipelineApplication",
    "ExperimentConfig",
    "MRIDataset",
    "FeatureExtractor",
    "AttentionWorkspace",
    "AttentionUNet",
    "AttentionMaskEditor",
]

print(
    "[PIPELINE] Staged CPU/GPU subsystem loaded.",
    flush=True,
)
print(
    "[PIPELINE] GPU actions: build-monai-feature-cache, "
    "generate-all-monai-masks, retrain-regenerate-attention, "
    "build-attention-feature-cache. CPU actions: monai-cpu-from-cache, "
    "evaluate-attention-cpu, edit-existing-masks.",
    flush=True,
)



# region OPTIONAL COMPATIBILITY ALIASES
# New code should use SETTINGS_<DOMAIN>.<NAME>. These aliases are read-only
# snapshots retained for older notebooks and can be removed after migration.
ATTENTION_INPUT_SIZE = SETTINGS_ATTENTION.ATTENTION_INPUT_SIZE
ATTENTION_SEGMENTATION_FOLDS = SETTINGS_ATTENTION.ATTENTION_SEGMENTATION_FOLDS
ATTENTION_BASE_CHANNELS = SETTINGS_ATTENTION.ATTENTION_BASE_CHANNELS
MONAI_INPUT_SIZE = SETTINGS_MONAI.MONAI_INPUT_SIZE
IMG_SIZE = SETTINGS_PREPROCESSING.IMG_SIZE
BATCH_SIZE = SETTINGS_RUNTIME.BATCH_SIZE
# endregion

# ---------------------------------------------------------------------------
# STANDALONE ENTRYPOINT
# ---------------------------------------------------------------------------
# The script requires one explicit staged action.
if __name__ == "__main__":
    PipelineApplication.run_cad_pipeline()
