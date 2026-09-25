"""Patient-level cardiac MRI CAD research pipeline.

This version keeps the complete scientific workflow while presenting it through
a smaller, ready experiment set. The non-negotiable guarantees are:

1. ``Directory_*`` is the patient identifier and never crosses validation folds.
2. Attention U-Net is trained by patient-level cross-fitting with explicit
   ``HEART_PRESENT``, ``NO_HEART_VISIBLE``, and ``UNUSABLE`` targets.
3. Every segmentation mask used downstream is out-of-fold for its patient.
4. CAD classification and all reported metrics are computed at patient level.
5. ``AU*`` names always denote an automatic Attention ROI, while ``C*`` names
   always denote its complement/control region. ``B*`` denotes a full-image
   baseline and ``M*`` a manual ROI reference.
6. The existing ``/kaggle/working/cad_attention_unet_workspace`` is reused.
   Checkpoints, masks, audits, review labels, matching files, and legacy feature
   bank keys remain compatible; cached neural stages are not rerun needlessly.
7. Cross-class matching preserves a strict mutual-neighbour core and expands it
   with unique, capacity-controlled Sick/Normal pairs from the same acquisition families.
8. ``B2_ATTENTION_ELIGIBLE_FULL_IMAGE``, ``AU1_ATTENTION_ROI``, and
   ``C1_ATTENTION_COMPLEMENT`` are built from exactly the same source slices.
   ``B0_FULL_IMAGE`` is retained only as the broader all-slice reference.

The three Kaggle entry points at the bottom separate CPU audit, GPU segmentation,
and final CPU evaluation so that expensive accelerators are used only where they
provide a real benefit.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import HTML, Javascript, clear_output, display
from sklearn.cluster import MiniBatchKMeans
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
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm.auto import tqdm

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PIPELINE_VERSION = "2026-09-25-clean-v3-same-slice-full-image-control"


def _as_float(value: Any, default: float = 0.0) -> float:


    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _as_int(value: Any, default: int = 0) -> int:
    return int(round(_as_float(value, float(default))))


def _default_dataset_path() -> Path:


    env_value = os.environ.get("CAD_DATASET_PATH", "").strip()
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            Path("/kaggle/input/cad-cardiac-mri-dataset"),
            Path("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"),
            Path(r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset"),
        ]
    )

    def is_dataset_root(path: Path) -> bool:
        return path.is_dir() and (path / "Normal").is_dir() and (path / "Sick").is_dir()

    for candidate in candidates:
        if is_dataset_root(candidate):
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for current_root, directory_names, _ in os.walk(kaggle_input):
            current = Path(current_root)
            try:
                depth = len(current.relative_to(kaggle_input).parts)
            except ValueError:
                depth = 99
            if {"Normal", "Sick"}.issubset(set(directory_names)):
                return current
            if depth >= 3:
                directory_names[:] = []

    return candidates[0] if candidates else Path.cwd() / "CAD Cardiac MRI Dataset"


def _default_workspace_path() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cad_attention_unet_workspace")
    return Path.cwd() / "cad_attention_unet_workspace"


# -----------------------------------------------------------------------------
# Configuration and persistent workspace contract
# -----------------------------------------------------------------------------
class PathSettings:
    """Centralize dataset and workspace locations.

    By default Kaggle artifacts stay in
    ``/kaggle/working/cad_attention_unet_workspace`` so every stage can reuse
    files created by previous CPU or GPU sessions.
    """

    DATASET_PATH = _default_dataset_path()
    WORKSPACE_ROOT = Path(
        os.environ.get("CAD_WORKSPACE_ROOT", str(_default_workspace_path()))
    )


class RuntimeSettings:


    ATTENTION_DEVICE = os.environ.get("CAD_ATTENTION_DEVICE", "cuda").strip().lower()
    FEATURE_DEVICE = os.environ.get("CAD_FEATURE_DEVICE", "cpu").strip().lower()

    RANDOM_SEED = 42
    NUM_WORKERS = 0  
    USE_AMP_ON_CUDA = True
    USE_CHANNELS_LAST_ON_CUDA = True
    CUDNN_BENCHMARK_ON_CUDA = True
    DETERMINISTIC_ALGORITHMS = False
    CPU_THREADS = max(1, min(8, os.cpu_count() or 1))
    PNG_COMPRESSION = 9
    IMAGE_RAM_CACHE_ITEMS = 2048


class ImageSettings:


    SEGMENTATION_SIZE = 256
    CLASSIFICATION_SIZE = 224
    STANDARDIZED_CONTENT_LONG_SIDE = 240


    DARK_LINE_MAX_MEAN = 12.0
    DARK_LINE_MAX_STD = 4.0
    DARK_PIXEL_MAX_VALUE = 20
    DARK_PIXEL_MIN_FRACTION = 0.98
    MAX_CROP_FRACTION_PER_SIDE = 0.20
    MIN_RETAINED_FRACTION = 0.60
    MIN_PADDING_RUN = 2

    ROBUST_LOWER_PERCENTILE = 1.0
    ROBUST_UPPER_PERCENTILE = 99.0


    REGION_LOWER_PERCENTILE = 1.0
    REGION_UPPER_PERCENTILE = 99.0
    REGION_MIN_PIXELS = 64
    REGION_HISTOGRAM_BINS = 256
    REGION_MIN_DYNAMIC_RANGE = 8.0 / 255.0

    QUALITY_THUMBNAIL_SIZE = 128
    QUALITY_MIN_DYNAMIC_RANGE = 12.0
    QUALITY_MIN_LAPLACIAN_VARIANCE = 4.0
    QUALITY_MAX_NOISE_RATIO = 0.30
    QUALITY_PATIENT_BLUR_QUANTILE = 0.03
    QUALITY_PATIENT_NOISE_QUANTILE = 0.97


class SegmentationSettings:


    FOLDS = 5
    BASE_CHANNELS = 24
    INPUT_CHANNELS = 3  
    USE_2_5D = True

    EPOCHS = 36
    EARLY_STOPPING_PATIENCE = 8
    LEARNING_RATE = 8e-4
    WEIGHT_DECAY = 1e-4
    LR_REDUCE_PATIENCE = 3
    LR_REDUCE_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-6


    FOCAL_WEIGHT = 0.40
    TVERSKY_WEIGHT = 0.60
    FOCAL_GAMMA = 2.0
    TVERSKY_ALPHA_FP = 0.65
    TVERSKY_BETA_FN = 0.35
    PRESENCE_LOSS_WEIGHT = 0.30


    MANUAL_DRAWN_WEIGHT = 1.00
    AUTO_CONFIRMED_WEIGHT = 0.72
    NO_HEART_WEIGHT = 0.90
    NEGATIVE_TARGET_FRACTION = 0.30

    BATCH_SIZE_CUDA = 10
    BATCH_SIZE_CPU = 3
    INFERENCE_BATCH_SIZE_CUDA = 10
    INFERENCE_BATCH_SIZE_CPU = 3


    MIN_MANUAL_MASKS = 800
    MANUAL_MIN_AREA_RATIO = 0.0005
    MANUAL_MAX_AREA_RATIO = 0.98
    VALIDATION_PATIENT_FRACTION = 0.20


    AUGMENT_HORIZONTAL_FLIP = False
    AUGMENT_ROTATION_DEGREES = 7.0
    AUGMENT_TRANSLATION_FRACTION = 0.06
    AUGMENT_SCALE_MIN = 0.92
    AUGMENT_SCALE_MAX = 1.08
    AUGMENT_GAMMA_MIN = 0.85
    AUGMENT_GAMMA_MAX = 1.15
    AUGMENT_NOISE_STD_MAX = 0.025
    AUGMENT_BLUR_PROBABILITY = 0.15

    CALIBRATION_THRESHOLDS = (
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    )
    PRESENCE_CALIBRATION_THRESHOLDS = (
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    )
    DEFAULT_THRESHOLD = 0.50
    DEFAULT_PRESENCE_THRESHOLD = 0.50
    PRESENCE_SEGMENTATION_OVERRIDE_PEAK = 0.80
    PREDICTION_MIN_AREA_RATIO = 0.003
    PREDICTION_MAX_AREA_RATIO = 0.65
    PREDICTION_MIN_PEAK_PROBABILITY = 0.50
    REPAIR_THRESHOLD_OFFSETS = (-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20)
    REPAIR_MAX_BOUNDARY_TOUCH = 0.35
    REPAIR_MORPHOLOGY_KERNEL = 5
    PRIOR_MAX_ROBUST_Z = 5.5


    TTA_CONTRAST_FACTOR = 1.10
    UNCERTAINTY_DISAGREEMENT_SCALE = 0.08

    SUPPORT_DILATION_KERNEL = 15
    SAVE_ALL_PREDICTED_MASKS = True


class ClassificationSettings:
    """Frozen feature extraction and patient-level classification settings.

    The registry contains seven primary experiments plus four mask-validation
    controls. ``B0`` remains the broad all-slice reference, while ``B2``, ``AU1``,
    and ``C1`` form the direct same-slice comparison required to isolate the
    effect of the automatic cardiac ROI from image-selection effects.
    """

    FEATURE_BATCH_SIZE_CUDA = 12
    FEATURE_BATCH_SIZE_CPU = 4

    FEATURE_FORWARD_BATCH_SIZE_CUDA = 32
    FEATURE_FORWARD_BATCH_SIZE_CPU = 4

    USE_IMAGENET_WEIGHTS = True
    PCA_EXPLAINED_VARIANCE = 0.95
    OUTER_FOLDS = 5
    INNER_FOLDS = 3
    C_GRID = (0.01, 0.1, 1.0, 10.0)
    BOOTSTRAP_REPEATS = 2000

    # B0 answers the broad contextual question on every dataset slice. The direct
    # automatic-mask experiment is the B2/AU1/C1 triad: all three modes use the
    # exact same images, removing source-image selection as a confounder while
    # retaining each mode's intended full/ROI/complement preprocessing.
    # B1/AU2/C2 repeat that logic on the Sick/Normal matched cohort.
    PRIMARY_MODES = (
        "B0_FULL_IMAGE",
        "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
        "AU1_ATTENTION_ROI",
        "C1_ATTENTION_COMPLEMENT",
        "B1_MATCHED_FULL_IMAGE",
        "AU2_MATCHED_ATTENTION_ROI",
        "C2_MATCHED_ATTENTION_COMPLEMENT",
    )
    VALIDATION_MODES = (
        "M1_MANUAL_ROI",
        "C3_MANUAL_COMPLEMENT",
        "AU3_ATTENTION_ROI_MANUAL_SUBSET",
        "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET",
    )
    MODES = PRIMARY_MODES + VALIDATION_MODES

    # Existing .npz files in /kaggle/working use the historical keys below. The
    # new B2 control has no historical equivalent, so it receives a new key while
    # every pre-existing mode remains readable without recomputation.
    LEGACY_MODE_ALIASES = {
        "B0_FULL_IMAGE": "FULL_IMAGE",
        "B2_ATTENTION_ELIGIBLE_FULL_IMAGE": "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
        "AU1_ATTENTION_ROI": "AU1_ATTENTION_ROI",
        "C1_ATTENTION_COMPLEMENT": "AU5_ATTENTION_COMPLEMENT",
        "B1_MATCHED_FULL_IMAGE": "CROSS_CLASS_MATCHED_FULL_IMAGE",
        "AU2_MATCHED_ATTENTION_ROI": "CROSS_CLASS_MATCHED_AU1_ATTENTION_ROI",
        "C2_MATCHED_ATTENTION_COMPLEMENT": "CROSS_CLASS_MATCHED_AU5_ATTENTION_COMPLEMENT",
        "M1_MANUAL_ROI": "AU6_MANUAL_ROI",
        "C3_MANUAL_COMPLEMENT": "AU7_MANUAL_COMPLEMENT",
        "AU3_ATTENTION_ROI_MANUAL_SUBSET": "AU8_ATTENTION_MATCHED_MANUAL_ROI",
        "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET": "AU9_ATTENTION_MATCHED_MANUAL_COMPLEMENT",
    }
    LEGACY_STORAGE_MODES = (
        "FULL_IMAGE",
        "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
        "AU1_ATTENTION_ROI",
        "AU5_ATTENTION_COMPLEMENT",
        "CROSS_CLASS_MATCHED_FULL_IMAGE",
        "CROSS_CLASS_MATCHED_AU1_ATTENTION_ROI",
        "CROSS_CLASS_MATCHED_AU5_ATTENTION_COMPLEMENT",
        "AU6_MANUAL_ROI",
        "AU7_MANUAL_COMPLEMENT",
        "AU8_ATTENTION_MATCHED_MANUAL_ROI",
        "AU9_ATTENTION_MATCHED_MANUAL_COMPLEMENT",
    )

    ATTENTION_ELIGIBLE_MODES = frozenset(
        {
            "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
            "AU1_ATTENTION_ROI",
            "C1_ATTENTION_COMPLEMENT",
        }
    )
    MATCHED_MODES = frozenset(
        {
            "B1_MATCHED_FULL_IMAGE",
            "AU2_MATCHED_ATTENTION_ROI",
            "C2_MATCHED_ATTENTION_COMPLEMENT",
        }
    )
    MANUAL_SUBSET_MODES = frozenset(VALIDATION_MODES)

    MODE_DESCRIPTIONS = {
        "B0_FULL_IMAGE": "Full-image contextual baseline on every dataset slice",
        "B2_ATTENTION_ELIGIBLE_FULL_IMAGE": "Full image on exactly the AU1/C1 attention-eligible slices",
        "AU1_ATTENTION_ROI": "Automatic Attention U-Net heart ROI on the same attention-eligible slices",
        "C1_ATTENTION_COMPLEMENT": "Pixels outside the automatic heart ROI on the same attention-eligible slices",
        "B1_MATCHED_FULL_IMAGE": "Full-image baseline on the expanded balanced Sick/Normal cohort",
        "AU2_MATCHED_ATTENTION_ROI": "Automatic heart ROI on the expanded balanced matched cohort",
        "C2_MATCHED_ATTENTION_COMPLEMENT": "Automatic-ROI complement on the expanded balanced matched cohort",
        "M1_MANUAL_ROI": "Manual heart ROI on the annotated subset",
        "C3_MANUAL_COMPLEMENT": "Complement of the manual ROI",
        "AU3_ATTENTION_ROI_MANUAL_SUBSET": "Automatic ROI on the same manually annotated images",
        "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET": "Automatic complement on the same manually annotated images",
    }
    MODE_COHORTS = {
        "B0_FULL_IMAGE": "all_dataset_slices",
        "B2_ATTENTION_ELIGIBLE_FULL_IMAGE": "attention_eligible_same_slices",
        "AU1_ATTENTION_ROI": "attention_eligible_same_slices",
        "C1_ATTENTION_COMPLEMENT": "attention_eligible_same_slices",
        "B1_MATCHED_FULL_IMAGE": "cross_class_matched_same_slices",
        "AU2_MATCHED_ATTENTION_ROI": "cross_class_matched_same_slices",
        "C2_MATCHED_ATTENTION_COMPLEMENT": "cross_class_matched_same_slices",
        "M1_MANUAL_ROI": "manual_positive_same_slices",
        "C3_MANUAL_COMPLEMENT": "manual_positive_same_slices",
        "AU3_ATTENTION_ROI_MANUAL_SUBSET": "manual_positive_same_slices",
        "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET": "manual_positive_same_slices",
    }


class MatchingSettings:
    """Settings for strict-core plus expanded Sick/Normal matching.

    The old mutual-top-5 cohort is retained as an auditable core. The cohort used
    by B1/AU2/C2 then adds unique one-to-one pairs that are close in the same
    acquisition family. Expansion is driven mainly by more candidate neighbours
    and larger diversity caps, rather than by removing anatomical constraints.
    """

    ENABLED = True
    MIN_FAMILIES = 8
    MAX_FAMILIES = 32
    TARGET_IMAGES_PER_FAMILY = 1800
    MIN_IMAGES_PER_CLASS_PER_FAMILY = 12
    MIN_PATIENTS_PER_CLASS_PER_FAMILY = 2

    # Strict core: reproduces the previous mutual-neighbour logic.
    MUTUAL_NEIGHBORS = 5
    CALIPER_MAD_MULTIPLIER = 2.5
    CALIPER_QUANTILE = 0.90
    MAX_SEQUENCE_POSITION_DIFFERENCE = 0.25
    MAX_AREA_RATIO_DIFFERENCE = 0.20
    MAX_MATCHES_PER_PATIENT_PER_FAMILY = 20
    MAX_MATCHES_PER_SEQUENCE_GROUP = 5

    # Expanded cohort. Every image is still used at most once and both classes
    # remain exactly balanced. With the current dataset the default target is
    # about 30% of eligible slices, instead of the previous ~13%.
    TARGET_MATCHED_IMAGE_FRACTION = 0.30
    EXTENDED_NEIGHBORS = 12
    EXTENDED_CALIPER_MAD_MULTIPLIER = 3.5
    EXTENDED_CALIPER_QUANTILE = 0.97
    EXTENDED_MAX_CALIPER_MULTIPLIER = 1.18
    EXTENDED_MAX_SEQUENCE_POSITION_DIFFERENCE = 0.25
    EXTENDED_MAX_AREA_RATIO_DIFFERENCE = 0.20
    EXTENDED_MAX_MATCHES_PER_PATIENT_PER_FAMILY = 60
    EXTENDED_MAX_MATCHES_PER_SEQUENCE_GROUP = 15
    EXTENDED_MAX_MATCHES_PER_SEQUENCE_PAIR = 12
    EXTENDED_MAX_MATCHES_PER_PATIENT_PAIR = 60
    PATIENT_TOTAL_CAP_MULTIPLIER = 1.70
    EXTENDED_BALANCE_QUOTA_STEPS = 12
    PRIORITIZE_CORE_SEQUENCE_PAIRS = True

    PHASH_BLOCK_WEIGHT = 0.45
    GEOMETRY_BLOCK_WEIGHT = 0.25
    SEQUENCE_BLOCK_WEIGHT = 0.15
    QUALITY_BLOCK_WEIGHT = 0.15


class ReviewSettings:


    NEW_IMAGES_PER_PATIENT = 10
    PHASH_MAX_DISTANCE = 6
    DEFAULT_LIMIT = 300
    BRUSH_RADIUS = 8
    UNCERTAINTY_MIN_SCORE = 0.18
    PRESENCE_MARGIN = 0.15


class Settings:


    Paths = PathSettings
    Runtime = RuntimeSettings
    Image = ImageSettings
    Segmentation = SegmentationSettings
    Classification = ClassificationSettings
    Matching = MatchingSettings
    Review = ReviewSettings


# -----------------------------------------------------------------------------
# Runtime, data records, and atomic persistence
# -----------------------------------------------------------------------------
class RuntimeManager:
    """Own device initialization at stage boundaries, never at import time.

    This is what permits a single persisted workspace to move safely between
    Kaggle CPU and GPU sessions without reserving CUDA memory during audits,
    manual review, matching, or evaluation.
    """


    @staticmethod
    def resolve_device(requested: str | None = None) -> torch.device:
        requested = (requested or "cpu").strip().lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested for this stage but is unavailable. "
                    "Use device='cpu' or enable a GPU accelerator."
                )
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def seed_everything(
        seed: int | None = None,
        include_cuda: bool = False,
    ) -> None:


        seed = int(Settings.Runtime.RANDOM_SEED if seed is None else seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(Settings.Runtime.CPU_THREADS)

        try:
            torch.use_deterministic_algorithms(
                bool(Settings.Runtime.DETERMINISTIC_ALGORITHMS), warn_only=True
            )
        except Exception:
            pass

        if include_cuda:
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = bool(
                    Settings.Runtime.DETERMINISTIC_ALGORITHMS
                )
                torch.backends.cudnn.benchmark = bool(
                    Settings.Runtime.CUDNN_BENCHMARK_ON_CUDA
                    and not Settings.Runtime.DETERMINISTIC_ALGORITHMS
                )

    @staticmethod
    @contextlib.contextmanager
    def device_scope(requested: str | None, stage_name: str):


        device = RuntimeManager.resolve_device(requested)
        RuntimeManager.seed_everything(include_cuda=device.type == "cuda")
        if device.type == "cuda":
            RuntimeManager.release(device)
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        print(f"[DEVICE] {stage_name}: {device.type}")
        try:
            yield device
        finally:
            if device.type == "cuda":
                try:
                    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(f"[DEVICE] {stage_name}: peak GPU={peak_gb:.2f} GB")
                except Exception:
                    pass
            RuntimeManager.release(device)
            print(f"[DEVICE] {stage_name}: resources released")

    @staticmethod
    def prepare_model(model: nn.Module, device: torch.device) -> nn.Module:
        model = model.to(device)
        if device.type == "cuda" and Settings.Runtime.USE_CHANNELS_LAST_ON_CUDA:
            model = model.to(memory_format=torch.channels_last)
        return model

    @staticmethod
    def move_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        tensor = tensor.to(device, non_blocking=device.type == "cuda")
        if (
            device.type == "cuda"
            and Settings.Runtime.USE_CHANNELS_LAST_ON_CUDA
            and tensor.ndim == 4
        ):
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    @staticmethod
    def train_batch_size(device: torch.device) -> int:
        return (
            Settings.Segmentation.BATCH_SIZE_CUDA
            if device.type == "cuda"
            else Settings.Segmentation.BATCH_SIZE_CPU
        )

    @staticmethod
    def inference_batch_size(device: torch.device) -> int:
        return (
            Settings.Segmentation.INFERENCE_BATCH_SIZE_CUDA
            if device.type == "cuda"
            else Settings.Segmentation.INFERENCE_BATCH_SIZE_CPU
        )

    @staticmethod
    def feature_batch_size(device: torch.device) -> int:
        return (
            Settings.Classification.FEATURE_BATCH_SIZE_CUDA
            if device.type == "cuda"
            else Settings.Classification.FEATURE_BATCH_SIZE_CPU
        )

    @staticmethod
    def feature_forward_batch_size(device: torch.device) -> int:
        return (
            Settings.Classification.FEATURE_FORWARD_BATCH_SIZE_CUDA
            if device.type == "cuda"
            else Settings.Classification.FEATURE_FORWARD_BATCH_SIZE_CPU
        )

    @staticmethod
    def autocast(device: torch.device):


        enabled = bool(Settings.Runtime.USE_AMP_ON_CUDA and device.type == "cuda")
        if not enabled:
            return contextlib.nullcontext()
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=True)

    @staticmethod
    def grad_scaler(device: torch.device):
        enabled = bool(Settings.Runtime.USE_AMP_ON_CUDA and device.type == "cuda")
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    @staticmethod
    def release(device: torch.device) -> None:


        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()

    @staticmethod
    def format_seconds(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"


@dataclass(frozen=True)
class Sample:


    image_path: str
    label: int
    patient_id: str
    series_id: str
    image_token: str
    segmentation_fold: int


@dataclass(frozen=True)
class Workspace:


    root: Path
    manual_masks: Path
    predicted_masks: Path
    mask_overlays: Path
    checkpoints: Path
    outputs: Path
    dataset_manifest: Path
    quality_audit: Path
    manual_audit: Path
    manual_annotations: Path
    prediction_audit: Path
    prediction_parts_dir: Path
    invalid_predictions: Path
    review_history: Path
    training_summary: Path
    prediction_summary: Path
    segmentation_metrics: Path
    segmentation_metrics_summary: Path
    cross_class_matching_manifest: Path
    cross_class_matching_summary: Path
    feature_bank: Path
    feature_metadata: Path
    results_csv: Path
    predictions_dir: Path


class FileManager:


    @staticmethod
    def create_workspace(root: Path | str | None = None) -> Workspace:
        root = Path(root or Settings.Paths.WORKSPACE_ROOT)
        workspace = Workspace(
            root=root,
            manual_masks=root / "manual_masks",
            predicted_masks=root / "predicted_attention_masks",
            mask_overlays=root / "mask_overlays",
            checkpoints=root / "checkpoints",
            outputs=root / "simple_pipeline_outputs",
            dataset_manifest=root / "simple_dataset_manifest.csv",
            quality_audit=root / "simple_image_quality_audit.csv",
            manual_audit=root / "simple_manual_mask_audit.csv",
            manual_annotations=root / "manual_annotation_labels.csv",
            prediction_audit=root / "simple_attention_prediction_audit.csv",
            prediction_parts_dir=root / "simple_pipeline_outputs" / "attention_prediction_parts",
            invalid_predictions=root / "attention_invalid_after_retrain.csv",
            review_history=root / "simple_review_history.csv",
            training_summary=root / "simple_training_summary.json",
            prediction_summary=root / "simple_prediction_summary.json",
            segmentation_metrics=root / "simple_pipeline_outputs" / "attention_oof_segmentation_metrics.csv",
            segmentation_metrics_summary=root / "simple_pipeline_outputs" / "attention_oof_segmentation_summary.json",
            cross_class_matching_manifest=root / "simple_pipeline_outputs" / "cross_class_matching_manifest.csv",
            cross_class_matching_summary=root / "simple_pipeline_outputs" / "cross_class_matching_summary.json",
            feature_bank=root / "simple_pipeline_outputs" / "patient_feature_bank.npz",
            feature_metadata=root / "simple_pipeline_outputs" / "patient_feature_bank.json",
            results_csv=root / "simple_pipeline_outputs" / "evaluation_summary.csv",
            predictions_dir=root / "simple_pipeline_outputs" / "oof_predictions",
        )
        for directory in (
            workspace.root,
            workspace.manual_masks,
            workspace.predicted_masks,
            workspace.mask_overlays,
            workspace.checkpoints,
            workspace.outputs,
            workspace.prediction_parts_dir,
            workspace.predictions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return workspace

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def read_json(path: Path, default: Any = None) -> Any:
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        os.replace(temporary, path)

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def write_png(path: Path, image: np.ndarray) -> None:


        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(image)
        ok, encoded = cv2.imencode(
            ".png",
            array,
            [cv2.IMWRITE_PNG_COMPRESSION, int(Settings.Runtime.PNG_COMPRESSION)],
        )
        if not ok:
            raise RuntimeError(f"OpenCV could not encode PNG: {path}")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


# -----------------------------------------------------------------------------
# Dataset discovery and backward-compatible workspace import
# -----------------------------------------------------------------------------
class WorkspaceCompatibilityManager:
    """Reuse artifacts produced by earlier full or simplified notebooks.

    Conversion is metadata-only: existing PNG masks and image tokens are not
    rewritten, and no model is executed merely to import an older workspace.
    """


    LEGACY_FULL_REVIEW_MANIFEST = "attention_unet_full_review_manifest.csv"
    LEGACY_QUALITY_AUDIT = "attention_image_quality_audit.csv"
    LEGACY_REVIEW_HISTORY = "attention_unet_review_history.csv"
    IMPORT_SUMMARY = "workspace_compatibility_import.json"

    @staticmethod
    def _first_nonempty(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
        for key in keys:
            value = row.get(key, "")
            if value is not None and str(value).strip() != "":
                return value
        return default

    @staticmethod
    def _prediction_path(
        base_row: dict[str, Any],
        source_row: dict[str, Any],
        workspace: Workspace,
    ) -> Path:


        token = str(base_row["image_token"])
        expected = workspace.predicted_masks / f"{token}.png"
        source_value = str(source_row.get("predicted_attention_mask_path", "")).strip()
        source_path = Path(source_value) if source_value else None
        if expected.is_file():
            return expected
        if source_path is not None and source_path.is_file():
            return source_path
        return expected

    @classmethod
    def _normalize_prediction_rows(
        cls,
        source_rows: Sequence[dict[str, Any]],
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
    ) -> list[dict[str, Any]]:


        source_by_token = {
            str(row.get("image_token", "")).strip(): dict(row)
            for row in source_rows
            if str(row.get("image_token", "")).strip()
        }
        normalized: list[dict[str, Any]] = []
        for base in dataset_rows:
            token = str(base["image_token"])
            source = source_by_token.get(token)
            if source is None:
                continue
            mask_path = cls._prediction_path(base, source, workspace)
            if not mask_path.is_file():
                continue

            final_valid = cls._first_nonempty(
                source, "attention_valid_final", "attention_valid", default=""
            )
            initial_valid = cls._first_nonempty(
                source,
                "attention_valid_initial",
                "attention_valid",
                default=final_valid,
            )
            normalized.append(
                {
                    "image_token": token,
                    "image_path": str(base["image_path"]),
                    "patient_id": str(base["patient_id"]),
                    "series_id": str(base["series_id"]),


                    "segmentation_fold": int(base["segmentation_fold"]),
                    "predicted_attention_mask_path": str(mask_path),
                    "attention_valid_initial": initial_valid,
                    "attention_valid_final": final_valid,
                    "attention_invalid_reason_initial": cls._first_nonempty(
                        source, "attention_invalid_reason_initial", default=""
                    ),
                    "attention_invalid_reason_final": cls._first_nonempty(
                        source, "attention_invalid_reason_final", default=""
                    ),
                    "attention_repair_method": cls._first_nonempty(
                        source, "attention_repair_method", default="legacy_import"
                    ),
                    "attention_threshold_used": cls._first_nonempty(
                        source, "attention_threshold_used", default=""
                    ),
                    "attention_area_ratio": cls._first_nonempty(
                        source, "attention_area_ratio", default=""
                    ),
                    "attention_peak_probability": cls._first_nonempty(
                        source, "attention_peak_probability", default=""
                    ),
                    "attention_boundary_touch_fraction": cls._first_nonempty(
                        source, "attention_boundary_touch_fraction", default=""
                    ),
                    "attention_heart_present": cls._first_nonempty(
                        source, "attention_heart_present", default=""
                    ),
                    "attention_presence_probability": cls._first_nonempty(
                        source, "attention_presence_probability", default=""
                    ),
                    "attention_presence_threshold": cls._first_nonempty(
                        source, "attention_presence_threshold", default=""
                    ),
                    "attention_tta_disagreement": cls._first_nonempty(
                        source, "attention_tta_disagreement", default=""
                    ),
                    "attention_mean_entropy": cls._first_nonempty(
                        source, "attention_mean_entropy", default=""
                    ),
                    "attention_uncertainty_score": cls._first_nonempty(
                        source, "attention_uncertainty_score", default=""
                    ),
                    "attention_sequence_inconsistency": cls._first_nonempty(
                        source, "attention_sequence_inconsistency", default=""
                    ),
                    "attention_centroid_x": cls._first_nonempty(
                        source, "attention_centroid_x", default=""
                    ),
                    "attention_centroid_y": cls._first_nonempty(
                        source, "attention_centroid_y", default=""
                    ),
                    "attention_prior_deviation": cls._first_nonempty(
                        source, "attention_prior_deviation", default=""
                    ),
                    "checkpoint_fingerprint": cls._first_nonempty(
                        source,
                        "checkpoint_fingerprint",
                        "attention_checkpoint_fingerprint",
                        default="",
                    ),
                }
            )
        return normalized

    @classmethod
    def _derive_prediction_rows_from_masks(
        cls,
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
    ) -> list[dict[str, Any]]:


        invalid_tokens = {
            str(row.get("image_token", "")).strip()
            for row in FileManager.read_csv(workspace.invalid_predictions)
            if str(row.get("image_token", "")).strip()
        }
        rows: list[dict[str, Any]] = []
        for base in dataset_rows:
            token = str(base["image_token"])
            mask_path = workspace.predicted_masks / f"{token}.png"
            if not mask_path.is_file():
                continue
            is_known_invalid = token in invalid_tokens
            rows.append(
                {
                    "image_token": token,
                    "image_path": str(base["image_path"]),
                    "patient_id": str(base["patient_id"]),
                    "series_id": str(base["series_id"]),
                    "segmentation_fold": int(base["segmentation_fold"]),
                    "predicted_attention_mask_path": str(mask_path),
                    "attention_valid_initial": 0 if is_known_invalid else "",
                    "attention_valid_final": 0 if is_known_invalid else "",
                    "attention_invalid_reason_initial": (
                        "listed_in_attention_invalid_after_retrain"
                        if is_known_invalid
                        else ""
                    ),
                    "attention_invalid_reason_final": (
                        "listed_in_attention_invalid_after_retrain"
                        if is_known_invalid
                        else ""
                    ),
                    "attention_repair_method": "legacy_mask_without_manifest",
                    "attention_threshold_used": "",
                    "attention_area_ratio": "",
                    "attention_peak_probability": "",
                    "attention_boundary_touch_fraction": "",
                    "attention_heart_present": "",
                    "attention_presence_probability": "",
                    "attention_presence_threshold": "",
                    "attention_tta_disagreement": "",
                    "attention_mean_entropy": "",
                    "attention_uncertainty_score": "",
                    "attention_sequence_inconsistency": "",
                    "attention_centroid_x": "",
                    "attention_centroid_y": "",
                    "attention_prior_deviation": "",
                    "checkpoint_fingerprint": "",
                }
            )
        return rows

    @classmethod
    def ensure_prediction_audit(
        cls,
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        require_complete: bool = False,
    ) -> list[dict[str, Any]]:


        current_raw = FileManager.read_csv(workspace.prediction_audit)
        current = cls._normalize_prediction_rows(current_raw, dataset_rows, workspace)

        legacy_path = workspace.root / cls.LEGACY_FULL_REVIEW_MANIFEST
        legacy_raw = FileManager.read_csv(legacy_path)
        legacy = cls._normalize_prediction_rows(legacy_raw, dataset_rows, workspace)

        derived = cls._derive_prediction_rows_from_masks(dataset_rows, workspace)

        candidates = [
            (len(current), 3, "simple", current),
            (len(legacy), 2, "full_manifest", legacy),
            (len(derived), 1, "mask_directory", derived),
        ]
        _count, _priority, source, selected = max(candidates, key=lambda item: (item[0], item[1]))

        if not selected:
            raise FileNotFoundError(
                "No reusable Attention metadata or masks were found. The pipeline searched "
                f"{workspace.prediction_audit}, {legacy_path}, and "
                f"{workspace.predicted_masks}. Run the GPU training/"
                "prediction stage only if these artifacts have not already been generated."
            )

        total = len(dataset_rows)
        if require_complete and len(selected) != total:
            raise RuntimeError(
                "The feature bank requires one Attention mask for every "
                f"image, but only {len(selected)}/{total} were found. Selected source: "
                f"{source}. Review may continue on the available subset; "
                "evaluation requires complete Attention inference."
            )

        should_write = (
            not workspace.prediction_audit.is_file()
            or source != "simple"
            or len(current) != len(selected)
        )
        if should_write:
            FileManager.write_csv(
                workspace.prediction_audit,
                selected,
                SegmentationManager.PREDICTION_FIELDS,
            )
            FileManager.write_json(
                workspace.outputs / cls.IMPORT_SUMMARY,
                {
                    "schema": "simple-full-workspace-compat-v1",
                    "source": source,
                    "dataset_images": total,
                    "imported_prediction_rows": len(selected),
                    "simple_prediction_audit": str(workspace.prediction_audit),
                    "legacy_full_review_manifest": str(legacy_path),
                    "prediction_masks_directory": str(workspace.predicted_masks),
                    "note": (
                        "Metadata were converted for review and feature extraction; "
                        "simple_prediction_summary.json was intentionally left unchanged, "
                        "so the simplified model inference cache remains independent."
                    ),
                },
            )
            print(
                "[COMPATIBILITY] Prediction audit built from "
                f"{source}: {len(selected)}/{total} images -> "
                f"{workspace.prediction_audit}"
            )
        else:
            print(
                "[COMPATIBILITY] Existing simplified audit reused: "
                f"{len(selected)}/{total} images."
            )
        return selected

    @classmethod
    def ensure_quality_audit(
        cls,
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
    ) -> list[dict[str, Any]]:


        current = FileManager.read_csv(workspace.quality_audit)
        current_tokens = {
            str(row.get("image_token", "")).strip()
            for row in current
            if str(row.get("image_token", "")).strip()
        }
        if len(current_tokens) == len(dataset_rows):
            return current

        legacy_path = workspace.root / cls.LEGACY_QUALITY_AUDIT
        legacy = FileManager.read_csv(legacy_path)
        legacy_by_token = {
            str(row.get("image_token", "")).strip(): row
            for row in legacy
            if str(row.get("image_token", "")).strip()
        }
        if not legacy_by_token:
            return current

        normalized: list[dict[str, Any]] = []
        for base in dataset_rows:
            token = str(base["image_token"])
            source = legacy_by_token.get(token)
            if source is None:
                continue
            image_path = Path(base["image_path"])
            stat = image_path.stat() if image_path.is_file() else None
            normalized.append(
                {
                    "image_token": token,
                    "image_path": str(image_path),
                    "patient_id": str(base["patient_id"]),
                    "series_id": str(base["series_id"]),
                    "image_size_bytes": int(stat.st_size) if stat else -1,
                    "image_mtime_ns": int(stat.st_mtime_ns) if stat else -1,
                    "perceptual_hash": cls._first_nonempty(
                        source, "perceptual_hash", default=""
                    ),
                    "sharpness": cls._first_nonempty(
                        source, "sharpness", "image_quality_sharpness", default=""
                    ),
                    "noise_ratio": cls._first_nonempty(
                        source, "noise_ratio", "image_quality_noise_ratio", default=""
                    ),
                    "dynamic_range": cls._first_nonempty(
                        source, "dynamic_range", "image_quality_dynamic_range", default=""
                    ),
                    "patient_blur_threshold": cls._first_nonempty(
                        source, "patient_blur_threshold", default=""
                    ),
                    "patient_noise_threshold": cls._first_nonempty(
                        source, "patient_noise_threshold", default=""
                    ),
                    "quality_valid": cls._first_nonempty(
                        source, "quality_valid", "image_quality_valid", default=0
                    ),
                    "quality_reason": cls._first_nonempty(
                        source, "quality_reason", "image_quality_reason", default=""
                    ),
                }
            )

        if len(normalized) > len(current_tokens):
            FileManager.write_csv(
                workspace.quality_audit,
                normalized,
                QualityManager.FIELDS,
            )
            print(
                "[COMPATIBILITY] Full-workspace quality audit imported: "
                f"{len(normalized)}/{len(dataset_rows)} images -> "
                f"{workspace.quality_audit}"
            )
            return normalized
        return current

    @classmethod
    def legacy_status(cls, workspace: Workspace) -> dict[str, Any]:
        return {
            "legacy_full_review_manifest": (
                workspace.root / cls.LEGACY_FULL_REVIEW_MANIFEST
            ).is_file(),
            "legacy_quality_audit": (
                workspace.root / cls.LEGACY_QUALITY_AUDIT
            ).is_file(),
            "legacy_prediction_generation": (
                workspace.root / "attention_unet_prediction_generation.json"
            ).is_file(),
        }


class DatasetManager:
    """Discover images while treating the patient as the statistical unit.

    Immediate child directories are only series proxies. Fold assignment is
    deterministic and patient-level, preventing slice leakage across folds.
    """


    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    @staticmethod
    def _relative_token_path(image_path: Path) -> str:
        parts = image_path.parts
        for index, part in enumerate(parts):
            if str(part).startswith("Directory_"):
                return "/".join(map(str, parts[index:]))
        raise ValueError(f"Path does not contain Directory_*: {image_path}")

    @staticmethod
    def image_token(image_path: Path, patient_id: str, series_id: str) -> str:


        relative = DatasetManager._relative_token_path(image_path)
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        safe_series = str(series_id).replace("/", "__").replace("\\", "__").replace(" ", "_")
        return f"{patient_id}__{safe_series}__{digest}"

    # Fold assignment is derived only from patient identity and class label.
    # No image-level random split is permitted because thousands of correlated
    # slices from one patient would otherwise leak into both train and validation.
    @staticmethod
    def patient_folds(
        patients: dict[str, int] | Iterable[str],
    ) -> dict[str, int]:


        if isinstance(patients, dict):
            groups: dict[int, list[str]] = defaultdict(list)
            for patient_id, label in patients.items():
                groups[int(label)].append(str(patient_id))
        else:
            groups = {0: list(map(str, patients))}

        result: dict[str, int] = {}
        for label, patient_ids in sorted(groups.items()):
            ordered = sorted(
                set(patient_ids),
                key=lambda patient_id: hashlib.sha256(
                    f"{Settings.Runtime.RANDOM_SEED}|segmentation-fold|{label}|{patient_id}".encode("utf-8")
                ).hexdigest(),
            )
            for index, patient_id in enumerate(ordered):
                result[patient_id] = index % int(Settings.Segmentation.FOLDS)
        return result

    @staticmethod
    def discover(dataset_path: Path | str, workspace: Workspace | None = None) -> list[Sample]:


        dataset_path = Path(dataset_path)
        started = time.perf_counter()
        raw_rows: list[tuple[Path, int, str, str]] = []
        patient_to_label: dict[str, int] = {}

        for class_name, label in (("Normal", 0), ("Sick", 1)):
            class_path = dataset_path / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(f"Missing directory: {class_path}")

            for patient_path in sorted(class_path.iterdir()):
                if not patient_path.is_dir() or not patient_path.name.lower().startswith("directory_"):
                    continue
                patient_id = patient_path.name
                old_label = patient_to_label.get(patient_id)
                if old_label is not None and old_label != label:
                    raise RuntimeError(f"{patient_id} appears in both classes.")
                patient_to_label[patient_id] = label


                for image_path in sorted(patient_path.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in DatasetManager.IMAGE_EXTENSIONS:
                        raw_rows.append((image_path, label, patient_id, f"{patient_id}/__ROOT__"))


                for series_path in sorted(path for path in patient_path.iterdir() if path.is_dir()):
                    series_id = f"{patient_id}/{series_path.name}"
                    for image_path in sorted(series_path.rglob("*")):
                        if image_path.is_file() and image_path.suffix.lower() in DatasetManager.IMAGE_EXTENSIONS:
                            raw_rows.append((image_path, label, patient_id, series_id))

        if not raw_rows:
            raise RuntimeError(f"No images were found in {dataset_path}")

        folds = DatasetManager.patient_folds(patient_to_label)
        samples = [
            Sample(
                image_path=str(path),
                label=int(label),
                patient_id=patient_id,
                series_id=series_id,
                image_token=DatasetManager.image_token(path, patient_id, series_id),
                segmentation_fold=int(folds[patient_id]),
            )
            for path, label, patient_id, series_id in raw_rows
        ]

        if len({sample.image_token for sample in samples}) != len(samples):
            raise RuntimeError("Duplicate image tokens were generated.")

        rows = [
            {
                "manifest_index": index,
                "image_token": sample.image_token,
                "image_path": sample.image_path,
                "label": sample.label,
                "patient_id": sample.patient_id,
                "series_id": sample.series_id,
                "segmentation_fold": sample.segmentation_fold,
                "manual_mask_path": str((workspace.manual_masks if workspace else Path("manual_masks")) / f"{sample.image_token}.png"),
                "predicted_attention_mask_path": str((workspace.predicted_masks if workspace else Path("predicted_attention_masks")) / f"{sample.image_token}.png"),
            }
            for index, sample in enumerate(samples)
        ]
        if workspace is not None:
            FileManager.write_csv(workspace.dataset_manifest, rows, rows[0].keys())

        print("[DATA] Patients:", len(patient_to_label))
        print("[DATA] Normal:", sum(label == 0 for label in patient_to_label.values()))
        print("[DATA] Sick:", sum(label == 1 for label in patient_to_label.values()))
        print("[DATA] Images:", len(samples))
        print("[DATA] Series proxies:", len({sample.series_id for sample in samples}))
        print("[DATA] Scan time:", RuntimeManager.format_seconds(time.perf_counter() - started))
        return samples

    @staticmethod
    def _natural_path_key(path: str) -> tuple[Any, ...]:


        return tuple(
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))
        )

    @staticmethod
    def rows(samples: Sequence[Sample], workspace: Workspace) -> list[dict[str, Any]]:
        rows = [
            {
                "manifest_index": index,
                "image_token": sample.image_token,
                "image_path": sample.image_path,
                "label": sample.label,
                "patient_id": sample.patient_id,
                "series_id": sample.series_id,
                "segmentation_fold": sample.segmentation_fold,
                "manual_mask_path": str(workspace.manual_masks / f"{sample.image_token}.png"),
                "predicted_attention_mask_path": str(workspace.predicted_masks / f"{sample.image_token}.png"),
            }
            for index, sample in enumerate(samples)
        ]


        by_sequence_group: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):


            parent = str(Path(row["image_path"]).parent)
            sequence_group = f"{row['series_id']}::{parent}"
            row["sequence_group_id"] = sequence_group
            by_sequence_group[sequence_group].append(index)
        for indices in by_sequence_group.values():
            ordered = sorted(
                indices,
                key=lambda index: DatasetManager._natural_path_key(rows[index]["image_path"]),
            )
            for position, row_index in enumerate(ordered):
                previous_index = ordered[max(0, position - 1)]
                next_index = ordered[min(len(ordered) - 1, position + 1)]
                rows[row_index]["sequence_index"] = int(position)
                rows[row_index]["sequence_length"] = int(len(ordered))
                rows[row_index]["previous_image_path"] = rows[previous_index]["image_path"]
                rows[row_index]["next_image_path"] = rows[next_index]["image_path"]

        if rows:
            FileManager.write_csv(workspace.dataset_manifest, rows, rows[0].keys())
        return rows


# -----------------------------------------------------------------------------
# Label-independent preprocessing, image quality, and manual targets
# -----------------------------------------------------------------------------
class ImageProcessor:
    """Apply label-independent preprocessing consistently to every split.

    The 256×256 geometry is intentionally unchanged so historical manual masks
    remain aligned and reusable in the current Kaggle workspace.
    """


    @staticmethod
    def read_gray(path: Path | str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {path}")
        return image

    @staticmethod
    def scale_0_1(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        minimum = float(image.min())
        maximum = float(image.max())
        if maximum <= minimum:
            return np.zeros_like(image, dtype=np.float32)
        return (image - minimum) / (maximum - minimum)

    @staticmethod
    def _is_dark_uniform_line(line: np.ndarray) -> bool:
        values = np.asarray(line, dtype=np.float32).reshape(-1)
        return bool(
            values.size
            and float(values.mean()) <= Settings.Image.DARK_LINE_MAX_MEAN
            and float(values.std()) <= Settings.Image.DARK_LINE_MAX_STD
            and float(np.mean(values <= Settings.Image.DARK_PIXEL_MAX_VALUE))
            >= Settings.Image.DARK_PIXEL_MIN_FRACTION
        )

    @staticmethod
    def detect_padding_bounds(image: np.ndarray) -> tuple[int, int, int, int]:
        if image.ndim != 2:
            raise ValueError("The image must be a 2D grayscale array.")
        height, width = image.shape
        min_height = max(8, int(np.ceil(height * Settings.Image.MIN_RETAINED_FRACTION)))
        min_width = max(8, int(np.ceil(width * Settings.Image.MIN_RETAINED_FRACTION)))
        max_vertical = int(np.floor(height * Settings.Image.MAX_CROP_FRACTION_PER_SIDE))
        max_horizontal = int(np.floor(width * Settings.Image.MAX_CROP_FRACTION_PER_SIDE))

        top = 0
        while top < max_vertical and height - top - 1 >= min_height and ImageProcessor._is_dark_uniform_line(image[top, :]):
            top += 1
        bottom_crop = 0
        while bottom_crop < max_vertical and height - top - bottom_crop - 1 >= min_height and ImageProcessor._is_dark_uniform_line(image[height - 1 - bottom_crop, :]):
            bottom_crop += 1
        left = 0
        while left < max_horizontal and width - left - 1 >= min_width and ImageProcessor._is_dark_uniform_line(image[:, left]):
            left += 1
        right_crop = 0
        while right_crop < max_horizontal and width - left - right_crop - 1 >= min_width and ImageProcessor._is_dark_uniform_line(image[:, width - 1 - right_crop]):
            right_crop += 1

        if top < Settings.Image.MIN_PADDING_RUN:
            top = 0
        if bottom_crop < Settings.Image.MIN_PADDING_RUN:
            bottom_crop = 0
        if left < Settings.Image.MIN_PADDING_RUN:
            left = 0
        if right_crop < Settings.Image.MIN_PADDING_RUN:
            right_crop = 0

        bottom = height - bottom_crop
        right = width - right_crop
        if bottom - top < min_height or right - left < min_width:
            return 0, height, 0, width
        return int(top), int(bottom), int(left), int(right)

    @staticmethod
    def robust_scale(image: np.ndarray) -> tuple[np.ndarray, float, float]:
        image = np.asarray(image, dtype=np.float32)
        lower, upper = np.percentile(
            image,
            [Settings.Image.ROBUST_LOWER_PERCENTILE, Settings.Image.ROBUST_UPPER_PERCENTILE],
        )
        lower, upper = float(lower), float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower, upper = float(image.min()), float(image.max())
        if upper <= lower:
            return np.zeros_like(image, dtype=np.float32), lower, upper
        scaled = np.clip(image, lower, upper)
        return ((scaled - lower) / (upper - lower)).astype(np.float32), lower, upper

    @staticmethod
    def _canvas_geometry(height: int, width: int) -> tuple[int, int, int, int, float]:
        size = Settings.Image.SEGMENTATION_SIZE
        long_side = Settings.Image.STANDARDIZED_CONTENT_LONG_SIDE
        scale = float(long_side / max(height, width))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        if height >= width:
            resized_height = long_side
        else:
            resized_width = long_side
        resized_height = min(resized_height, size)
        resized_width = min(resized_width, size)
        top = (size - resized_height) // 2
        left = (size - resized_width) // 2
        return resized_height, resized_width, top, left, scale

    @staticmethod
    def _to_canvas(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        height, width = image.shape
        resized_height, resized_width, top, left, scale = ImageProcessor._canvas_geometry(height, width)
        if (resized_height, resized_width) != (height, width):
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
        else:
            resized = image
        canvas = np.zeros(
            (Settings.Image.SEGMENTATION_SIZE, Settings.Image.SEGMENTATION_SIZE),
            dtype=np.float32,
        )
        canvas[top : top + resized_height, left : left + resized_width] = np.clip(resized, 0.0, 1.0)
        return canvas

    @staticmethod
    def _padding_canvas(height: int, width: int) -> np.ndarray:
        resized_height, resized_width, top, left, _ = ImageProcessor._canvas_geometry(height, width)
        canvas = np.ones(
            (Settings.Image.SEGMENTATION_SIZE, Settings.Image.SEGMENTATION_SIZE),
            dtype=np.float32,
        )
        canvas[top : top + resized_height, left : left + resized_width] = 0.0
        return canvas

    @staticmethod
    def standardize(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        top, bottom, left, right = ImageProcessor.detect_padding_bounds(image)
        cropped = image[top:bottom, left:right]
        if cropped.size == 0:
            raise RuntimeError("Automatic cropping produced an empty image.")
        robust, _, _ = ImageProcessor.robust_scale(cropped)
        minmax = ImageProcessor.scale_0_1(cropped)
        raw = cropped.astype(np.float32) / 255.0
        robust_canvas = ImageProcessor._to_canvas(robust)
        minmax_canvas = ImageProcessor._to_canvas(minmax)
        raw_canvas = ImageProcessor._to_canvas(raw)
        padding = ImageProcessor._padding_canvas(*cropped.shape)
        content = np.clip(1.0 - padding, 0.0, 1.0)
        return robust_canvas, minmax_canvas, raw_canvas, content

    @staticmethod
    def standardized_uint8(path: Path | str) -> np.ndarray:


        image = ImageProcessor.read_gray(path)
        robust, _, _, _ = ImageProcessor.standardize(image)
        return np.clip(np.round(robust * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def classifier_views(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = ImageProcessor.read_gray(path)
        robust, _, raw, content = ImageProcessor.standardize(image)
        size = Settings.Image.CLASSIFICATION_SIZE
        robust_224 = cv2.resize(robust, (size, size), interpolation=cv2.INTER_AREA)
        raw_224 = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
        content_224 = cv2.resize(content, (size, size), interpolation=cv2.INTER_NEAREST)
        return (
            robust_224.astype(np.float32),
            raw_224.astype(np.float32),
            (content_224 > 0.5).astype(np.float32),
        )

    @staticmethod
    def perceptual_hash(image: np.ndarray) -> str:
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        values = cv2.dct(resized.astype(np.float32))[:8, :8].reshape(-1)
        median = float(np.median(values[1:]))
        bits = values > median
        number = 0
        for bit in bits:
            number = (number << 1) | int(bool(bit))
        return f"{number:016x}"


class ImageCache:


    _cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @classmethod
    def get(cls, path: str) -> np.ndarray:
        if path in cls._cache:
            value = cls._cache.pop(path)
            cls._cache[path] = value
            return value.copy()
        value = ImageProcessor.standardized_uint8(path)
        cls._cache[path] = value
        while len(cls._cache) > Settings.Runtime.IMAGE_RAM_CACHE_ITEMS:
            cls._cache.popitem(last=False)
        return value.copy()

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


class QualityManager:


    FIELDS = (
        "image_token",
        "image_path",
        "patient_id",
        "series_id",
        "image_size_bytes",
        "image_mtime_ns",
        "perceptual_hash",
        "sharpness",
        "noise_ratio",
        "dynamic_range",
        "patient_blur_threshold",
        "patient_noise_threshold",
        "quality_valid",
        "quality_reason",
    )

    @staticmethod
    def _metrics(row: dict[str, Any]) -> dict[str, Any]:
        image = ImageProcessor.read_gray(row["image_path"])
        robust, _, raw, content = ImageProcessor.standardize(image)
        coordinates = np.argwhere(content > 0.5)
        if not len(coordinates):
            return {
                "perceptual_hash": "",
                "sharpness": np.nan,
                "noise_ratio": np.nan,
                "dynamic_range": np.nan,
            }
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0) + 1
        size = Settings.Image.QUALITY_THUMBNAIL_SIZE
        robust_thumb = cv2.resize(
            robust[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA
        )
        raw_thumb = cv2.resize(
            raw[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA
        )
        robust_u8 = np.clip(np.round(robust_thumb * 255.0), 0, 255).astype(np.uint8)
        raw_255 = np.clip(raw_thumb * 255.0, 0.0, 255.0).astype(np.float32)
        sharpness = float(cv2.Laplacian(robust_u8, cv2.CV_32F).var())
        residual = raw_255 - cv2.GaussianBlur(raw_255, (3, 3), 0)
        residual_median = float(np.median(residual))
        noise_sigma = float(np.median(np.abs(residual - residual_median)) / 0.6744897501960817)
        lower, upper = np.percentile(raw_255, [5.0, 95.0])
        dynamic_range = float(max(upper - lower, 0.0))
        return {
            "perceptual_hash": ImageProcessor.perceptual_hash(robust_u8),
            "sharpness": sharpness,
            "noise_ratio": float(noise_sigma / max(dynamic_range, 1.0)),
            "dynamic_range": dynamic_range,
        }

    @staticmethod
    def build(rows: Sequence[dict[str, Any]], workspace: Workspace, refresh: bool = False) -> dict[str, dict[str, Any]]:
        existing = {
            row.get("image_token", ""): row
            for row in FileManager.read_csv(workspace.quality_audit)
        }
        records: list[dict[str, Any]] = []
        started = time.perf_counter()
        for index, row in enumerate(rows, start=1):
            path = Path(row["image_path"])
            file_stat = path.stat() if path.is_file() else None
            size_bytes = file_stat.st_size if file_stat is not None else -1
            mtime_ns = file_stat.st_mtime_ns if file_stat is not None else -1
            old = existing.get(row["image_token"], {})
            reusable = (
                not refresh
                and old
                and old.get("image_path") == str(path)
                and _as_int(old.get("image_size_bytes"), -2) == int(size_bytes)
                and _as_int(old.get("image_mtime_ns"), -2) == int(mtime_ns)
                and old.get("perceptual_hash")
            )
            if reusable:
                record = dict(old)
            else:
                metrics = QualityManager._metrics(row)
                record = {
                    "image_token": row["image_token"],
                    "image_path": str(path),
                    "patient_id": row["patient_id"],
                    "series_id": row["series_id"],
                    "image_size_bytes": int(size_bytes),
                    "image_mtime_ns": int(mtime_ns),
                    **metrics,
                }
            records.append(record)
            if index == 1 or index % 5000 == 0 or index == len(rows):
                print(
                    f"[QUALITY] {index}/{len(rows)} | "
                    f"{RuntimeManager.format_seconds(time.perf_counter() - started)}"
                )

        by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_patient[str(record["patient_id"])].append(record)

        for patient_rows in by_patient.values():
            sharp = np.asarray([_as_float(row.get("sharpness"), np.nan) for row in patient_rows])
            noise = np.asarray([_as_float(row.get("noise_ratio"), np.nan) for row in patient_rows])
            finite_sharp = sharp[np.isfinite(sharp)]
            finite_noise = noise[np.isfinite(noise)]
            blur_threshold = float(Settings.Image.QUALITY_MIN_LAPLACIAN_VARIANCE)
            noise_threshold = float(Settings.Image.QUALITY_MAX_NOISE_RATIO)
            if len(finite_sharp) >= 5:
                blur_threshold = max(
                    blur_threshold,
                    float(np.quantile(finite_sharp, Settings.Image.QUALITY_PATIENT_BLUR_QUANTILE)),
                )
            if len(finite_noise) >= 5:
                noise_threshold = min(
                    noise_threshold,
                    float(np.quantile(finite_noise, Settings.Image.QUALITY_PATIENT_NOISE_QUANTILE)),
                )
            for record in patient_rows:
                sharpness = float(record.get("sharpness", np.nan))
                noise_ratio = float(record.get("noise_ratio", np.nan))
                dynamic_range = float(record.get("dynamic_range", np.nan))
                reasons = []
                if not np.isfinite(dynamic_range) or dynamic_range < Settings.Image.QUALITY_MIN_DYNAMIC_RANGE:
                    reasons.append("low_dynamic_range")
                if not np.isfinite(sharpness) or sharpness < blur_threshold:
                    reasons.append("blurred")
                if not np.isfinite(noise_ratio) or noise_ratio > noise_threshold:
                    reasons.append("noisy")
                record["patient_blur_threshold"] = blur_threshold
                record["patient_noise_threshold"] = noise_threshold
                record["quality_valid"] = int(not reasons)
                record["quality_reason"] = ";".join(reasons)

        FileManager.write_csv(workspace.quality_audit, records, QualityManager.FIELDS)
        valid = sum(int(row["quality_valid"]) for row in records)
        print(f"[QUALITY] valid={valid}/{len(records)} | {workspace.quality_audit}")
        return {str(row["image_token"]): row for row in records}


class MaskManager:
    """Maintain explicit, auditable segmentation targets.

    A non-empty heart mask, a valid no-heart frame, and an unusable frame have
    different meanings. They are never inferred from an accidentally empty PNG.
    """


    HEART_PRESENT = "HEART_PRESENT"
    NO_HEART_VISIBLE = "NO_HEART_VISIBLE"
    UNUSABLE = "UNUSABLE"
    VALID_TARGET_TYPES = {HEART_PRESENT, NO_HEART_VISIBLE, UNUSABLE}

    ANNOTATION_FIELDS = (
        "timestamp_utc",
        "image_token",
        "patient_id",
        "series_id",
        "target_type",
        "source",
        "sample_weight",
        "note",
    )

    MANUAL_AUDIT_FIELDS = (
        "image_token",
        "patient_id",
        "series_id",
        "manual_mask_path",
        "status",
        "target_type",
        "heart_present",
        "annotation_source",
        "sample_weight",
        "area_ratio",
        "quality_valid",
        "quality_reason",
        "reason",
    )

    @staticmethod
    def read_binary(path: Path | str, size: int | None = None) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask cannot be read: {path}")
        size = int(size or Settings.Image.SEGMENTATION_SIZE)
        if mask.shape != (size, size):
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8)

    @staticmethod
    def manual_qc(path: Path | str) -> dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            return {"exists": 0, "usable": 0, "area_ratio": np.nan, "reason": "missing"}
        try:
            mask = MaskManager.read_binary(path)
        except Exception:
            return {"exists": 1, "usable": 0, "area_ratio": np.nan, "reason": "unreadable"}
        area = float(mask.mean())
        if area < Settings.Segmentation.MANUAL_MIN_AREA_RATIO:
            reason = "empty_or_nearly_empty"
        elif area > Settings.Segmentation.MANUAL_MAX_AREA_RATIO:
            reason = "nearly_full"
        else:
            reason = ""
        return {"exists": 1, "usable": int(not reason), "area_ratio": area, "reason": reason}

    @staticmethod
    def manual_audit_refresh_reason(workspace: Workspace) -> str:


        path = workspace.manual_audit
        if not path.is_file():
            return "missing_manual_audit"
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
        except Exception as error:
            return f"unreadable_manual_audit:{type(error).__name__}"

        required = {
            "image_token",
            "manual_mask_path",
            "status",
            "target_type",
            "heart_present",
            "annotation_source",
            "sample_weight",
        }
        missing = sorted(required - fields)
        if missing:
            return "legacy_schema_missing_columns:" + ",".join(missing)

        audit_mtime = path.stat().st_mtime_ns
        if (
            workspace.manual_annotations.is_file()
            and workspace.manual_annotations.stat().st_mtime_ns > audit_mtime
        ):
            return "manual_annotations_newer_than_audit"

        try:
            newest_mask_mtime = max(
                (mask_path.stat().st_mtime_ns for mask_path in workspace.manual_masks.glob("*.png")),
                default=0,
            )
        except OSError:
            newest_mask_mtime = audit_mtime + 1
        if newest_mask_mtime > audit_mtime:
            return "manual_mask_files_newer_than_audit"
        return ""

    @staticmethod
    def annotation_map(workspace: Workspace) -> dict[str, dict[str, Any]]:


        result: dict[str, dict[str, Any]] = {}
        for row in FileManager.read_csv(workspace.manual_annotations):
            token = str(row.get("image_token", "")).strip()
            if token:
                result[token] = dict(row)
        return result

    @staticmethod
    def _default_weight(target_type: str, source: str) -> float:
        target_type = str(target_type).upper()
        source = str(source).lower()
        if target_type == MaskManager.NO_HEART_VISIBLE:
            return float(Settings.Segmentation.NO_HEART_WEIGHT)
        if "auto" in source:
            return float(Settings.Segmentation.AUTO_CONFIRMED_WEIGHT)
        return float(Settings.Segmentation.MANUAL_DRAWN_WEIGHT)

    @staticmethod
    def set_annotation(
        workspace: Workspace,
        row: dict[str, Any],
        target_type: str,
        source: str,
        sample_weight: float | None = None,
        note: str = "",
    ) -> None:
        target_type = str(target_type).strip().upper()
        if target_type not in MaskManager.VALID_TARGET_TYPES:
            raise ValueError(f"Unknown annotation type: {target_type}")
        records = MaskManager.annotation_map(workspace)
        weight = (
            MaskManager._default_weight(target_type, source)
            if sample_weight is None
            else float(sample_weight)
        )
        records[str(row["image_token"])] = {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "image_token": str(row["image_token"]),
            "patient_id": str(row.get("patient_id", "")),
            "series_id": str(row.get("series_id", "")),
            "target_type": target_type,
            "source": str(source),
            "sample_weight": float(weight),
            "note": str(note),
        }
        FileManager.write_csv(
            workspace.manual_annotations,
            [records[token] for token in sorted(records)],
            MaskManager.ANNOTATION_FIELDS,
        )

    @staticmethod
    def remove_annotation(workspace: Workspace, image_token: str) -> None:
        records = MaskManager.annotation_map(workspace)
        records.pop(str(image_token), None)
        FileManager.write_csv(
            workspace.manual_annotations,
            [records[token] for token in sorted(records)],
            MaskManager.ANNOTATION_FIELDS,
        )

    @staticmethod
    def register_existing_manual_masks_as_heart_present(
        rows: Sequence[dict[str, Any]],
        workspace: Workspace,
    ) -> dict[str, Any]:


        by_token = {
            str(row.get("image_token", "")).strip(): dict(row)
            for row in rows
            if str(row.get("image_token", "")).strip()
        }
        annotations = MaskManager.annotation_map(workspace)
        manual_files = sorted(workspace.manual_masks.glob("*.png"))
        timestamp = pd.Timestamp.utcnow().isoformat()

        summary: dict[str, Any] = {
            "timestamp_utc": timestamp,
            "manual_png_files": len(manual_files),
            "registered_heart_present": 0,
            "already_explicitly_labeled": 0,
            "skipped_empty_or_invalid": 0,
            "skipped_token_absent_from_dataset": 0,
            "reason_counts": {},
            "annotations_path": str(workspace.manual_annotations),
        }
        reason_counts: dict[str, int] = defaultdict(int)

        for mask_path in manual_files:
            token = str(mask_path.stem)
            row = by_token.get(token)
            if row is None:
                summary["skipped_token_absent_from_dataset"] += 1
                reason_counts["token_absent_from_dataset"] += 1
                continue

            existing = annotations.get(token, {})
            existing_target = str(existing.get("target_type", "")).strip().upper()
            if existing_target in MaskManager.VALID_TARGET_TYPES:
                summary["already_explicitly_labeled"] += 1
                reason_counts[f"already_{existing_target.lower()}"] += 1
                continue

            mask_qc = MaskManager.manual_qc(mask_path)
            if not bool(mask_qc.get("usable", 0)):
                summary["skipped_empty_or_invalid"] += 1
                reason = str(mask_qc.get("reason", "") or "invalid_manual_mask")
                reason_counts[reason] += 1
                continue

            annotations[token] = {
                "timestamp_utc": timestamp,
                "image_token": token,
                "patient_id": str(row.get("patient_id", "")),
                "series_id": str(row.get("series_id", "")),
                "target_type": MaskManager.HEART_PRESENT,
                "source": "existing_manual_on_review",
                "sample_weight": float(Settings.Segmentation.MANUAL_DRAWN_WEIGHT),
                "note": (
                    "Existing non-empty manual PNG registered automatically "
                    "when review was opened; the mask file itself was not rewritten."
                ),
            }
            summary["registered_heart_present"] += 1
            reason_counts["registered_heart_present"] += 1


        if summary["registered_heart_present"] > 0 or not workspace.manual_annotations.is_file():
            FileManager.write_csv(
                workspace.manual_annotations,
                [annotations[token] for token in sorted(annotations)],
                MaskManager.ANNOTATION_FIELDS,
            )

        summary["reason_counts"] = dict(sorted(reason_counts.items()))
        FileManager.write_json(
            workspace.outputs / "review_manual_target_registration.json",
            summary,
        )
        print(
            "[REVIEW][EXISTING MASKS] "
            f"registered HEART_PRESENT={summary['registered_heart_present']}, "
            f"already labeled={summary['already_explicitly_labeled']}, "
            f"empty/invalid={summary['skipped_empty_or_invalid']}, "
            f"token missing={summary['skipped_token_absent_from_dataset']}"
        )
        return summary

    @staticmethod
    def audit_manual_masks(
        rows: Sequence[dict[str, Any]],
        quality: dict[str, dict[str, Any]],
        workspace: Workspace,
        minimum_masks: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:


        minimum_masks = int(minimum_masks or Settings.Segmentation.MIN_MANUAL_MASKS)
        by_token = {str(row["image_token"]): dict(row) for row in rows}
        manual_files = {path.stem: path for path in workspace.manual_masks.glob("*.png")}
        annotations = MaskManager.annotation_map(workspace)
        candidate_tokens = sorted(set(manual_files) | set(annotations))

        accepted: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        patient_counts: dict[str, int] = defaultdict(int)
        positive_patient_counts: dict[str, int] = defaultdict(int)
        fold_positive_counts: dict[int, int] = defaultdict(int)
        fold_negative_counts: dict[int, int] = defaultdict(int)
        positive_count = 0
        negative_count = 0
        unusable_count = 0

        for token in candidate_tokens:
            row = by_token.get(token)
            path = manual_files.get(token, workspace.manual_masks / f"{token}.png")
            annotation = annotations.get(token, {})
            if row is None:
                audit.append(
                    {
                        "image_token": token,
                        "manual_mask_path": str(path),
                        "status": "REJECTED",
                        "target_type": annotation.get("target_type", ""),
                        "reason": "token_absent_from_dataset",
                    }
                )
                continue

            mask_qc = MaskManager.manual_qc(path)
            target_type = str(annotation.get("target_type", "")).strip().upper()
            source = str(annotation.get("source", "")).strip()
            if not target_type:
                if mask_qc["usable"]:
                    target_type = MaskManager.HEART_PRESENT
                    source = "legacy_nonempty_manual"
                else:
                    target_type = "UNLABELED_EMPTY"
                    source = "legacy_unlabeled"

            quality_row = quality.get(token, {})
            quality_valid = _as_int(quality_row.get("quality_valid"), 0) == 1
            reasons: list[str] = []
            heart_present = ""

            if target_type == MaskManager.HEART_PRESENT:
                heart_present = 1
                if not mask_qc["usable"]:
                    reasons.append(mask_qc["reason"] or "missing_positive_mask")
            elif target_type == MaskManager.NO_HEART_VISIBLE:
                heart_present = 0
                if not path.is_file():
                    FileManager.write_png(
                        path,
                        np.zeros(
                            (Settings.Image.SEGMENTATION_SIZE, Settings.Image.SEGMENTATION_SIZE),
                            dtype=np.uint8,
                        ),
                    )
                    mask_qc = MaskManager.manual_qc(path)
                if not np.isfinite(_as_float(mask_qc.get("area_ratio"), np.nan)):
                    reasons.append("unreadable_negative_mask")
                elif _as_float(mask_qc.get("area_ratio"), 1.0) >= Settings.Segmentation.MANUAL_MIN_AREA_RATIO:
                    reasons.append("no_heart_annotation_has_nonempty_mask")
            elif target_type == MaskManager.UNUSABLE:
                unusable_count += 1
                reasons.append("explicitly_unusable")
            else:
                reasons.append("empty_mask_without_explicit_no_heart_label")

            if target_type != MaskManager.UNUSABLE and not quality_valid:
                reasons.append(str(quality_row.get("quality_reason", "quality_invalid")))

            if target_type == MaskManager.UNUSABLE:
                status = "EXCLUDED"
            else:
                status = "ACCEPTED" if not reasons else "REJECTED"

            sample_weight = _as_float(
                annotation.get("sample_weight"),
                MaskManager._default_weight(target_type, source),
            )
            audit.append(
                {
                    "image_token": token,
                    "patient_id": row["patient_id"],
                    "series_id": row["series_id"],
                    "manual_mask_path": str(path),
                    "status": status,
                    "target_type": target_type,
                    "heart_present": heart_present,
                    "annotation_source": source,
                    "sample_weight": sample_weight,
                    "area_ratio": mask_qc.get("area_ratio", np.nan),
                    "quality_valid": quality_row.get("quality_valid", 0),
                    "quality_reason": quality_row.get("quality_reason", ""),
                    "reason": ";".join(filter(None, reasons)),
                }
            )

            if status != "ACCEPTED":
                continue
            row.update(
                {
                    "manual_mask_path": str(path),
                    "manual_mask_area_ratio": float(_as_float(mask_qc.get("area_ratio"), 0.0)),
                    "quality_valid": 1,
                    "perceptual_hash": quality_row.get("perceptual_hash", ""),
                    "segmentation_target_type": target_type,
                    "heart_present": int(heart_present),
                    "sample_weight": float(sample_weight),
                    "annotation_source": source,
                }
            )
            accepted.append(row)
            patient_counts[str(row["patient_id"])] += 1
            fold = int(row["segmentation_fold"])
            if int(heart_present) == 1:
                positive_count += 1
                positive_patient_counts[str(row["patient_id"])] += 1
                fold_positive_counts[fold] += 1
            else:
                negative_count += 1
                fold_negative_counts[fold] += 1

        FileManager.write_csv(workspace.manual_audit, audit, MaskManager.MANUAL_AUDIT_FIELDS)
        rejected = sum(row.get("status") == "REJECTED" for row in audit)
        excluded = sum(row.get("status") == "EXCLUDED" for row in audit)
        unlabeled_empty = sum(
            "without_explicit_no_heart" in str(row.get("reason", "")) for row in audit
        )
        blur_or_noise = sum(
            row.get("status") == "REJECTED" and bool(row.get("quality_reason"))
            for row in audit
        )
        summary = {
            "manual_png_files": len(manual_files),
            "explicit_annotations": len(annotations),
            "accepted_training_targets": len(accepted),
            "accepted_heart_present": positive_count,
            "accepted_no_heart_visible": negative_count,
            "explicit_unusable": unusable_count,
            "rejected_targets": rejected,
            "excluded_targets": excluded,
            "rejected_unlabeled_empty_masks": unlabeled_empty,
            "rejected_blur_or_noise": blur_or_noise,
            "patients_with_training_targets": len(patient_counts),
            "patients_with_positive_masks": len(positive_patient_counts),
            "targets_per_patient": dict(sorted(patient_counts.items())),
            "positive_masks_per_fold": {
                str(fold): int(fold_positive_counts.get(fold, 0))
                for fold in range(Settings.Segmentation.FOLDS)
            },
            "negative_masks_per_fold": {
                str(fold): int(fold_negative_counts.get(fold, 0))
                for fold in range(Settings.Segmentation.FOLDS)
            },
            "audit": str(workspace.manual_audit),
            "annotations": str(workspace.manual_annotations),
        }
        print(
            "[MANUAL MASKS] "
            f"heart_present={positive_count}, no_heart={negative_count}, "
            f"rejected={rejected}, unusable={unusable_count}, "
            f"patients={len(patient_counts)}"
        )

        if positive_count < minimum_masks:
            raise RuntimeError(
                f"Only {positive_count} valid HEART_PRESENT masks remain, below the minimum of "
                f"{minimum_masks}. See {workspace.manual_audit}"
            )
        missing_folds = [
            fold
            for fold in range(Settings.Segmentation.FOLDS)
            if fold_positive_counts.get(fold, 0) == 0
        ]
        if missing_folds:
            raise RuntimeError(
                f"No valid HEART_PRESENT masks exist in folds {missing_folds}."
            )
        if len(positive_patient_counts) < Settings.Segmentation.FOLDS + 1:
            raise RuntimeError("Too few patients have positive masks for cross-fitting.")
        return accepted, summary


# -----------------------------------------------------------------------------
# 2.5D Attention U-Net and leakage-safe out-of-fold inference
# -----------------------------------------------------------------------------
class AttentionConvBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = next(group for group in (8, 4, 2, 1) if out_channels % group == 0)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):


    def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        groups = next(group for group in (8, 4, 2, 1) if inter_channels % group == 0)
        self.gating = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(groups, inter_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(groups, inter_channels),
        )
        self.weight = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gating = F.interpolate(gating, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.weight(self.gating(gating) + self.skip(skip))


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gate = AttentionGate(out_channels, skip_channels, max(1, out_channels // 2))
        self.refine = AttentionConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up(x)
        return self.refine(torch.cat([x, self.gate(x, skip)], dim=1))


class AttentionUNet(nn.Module):
    """2.5D Attention U-Net with segmentation and heart-presence heads.

    Neighboring frames provide local sequence context, while the auxiliary head
    suppresses false positive masks on localizers or frames without visible heart.
    """


    def __init__(
        self,
        base_channels: int | None = None,
        input_channels: int | None = None,
    ):
        super().__init__()
        base = int(base_channels or Settings.Segmentation.BASE_CHANNELS)
        input_channels = int(input_channels or Settings.Segmentation.INPUT_CHANNELS)
        self.input_channels = input_channels
        self.encoder1 = AttentionConvBlock(input_channels, base)
        self.encoder2 = AttentionConvBlock(base, base * 2)
        self.encoder3 = AttentionConvBlock(base * 2, base * 4)
        self.encoder4 = AttentionConvBlock(base * 4, base * 8)
        self.bottleneck = AttentionConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.decoder4 = AttentionUpBlock(base * 16, base * 8, base * 8)
        self.decoder3 = AttentionUpBlock(base * 8, base * 4, base * 4)
        self.decoder2 = AttentionUpBlock(base * 4, base * 2, base * 2)
        self.decoder1 = AttentionUpBlock(base * 2, base, base)
        self.output = nn.Conv2d(base, 1, 1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, base * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(base * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        presence_logits = self.presence_head(bottleneck).squeeze(1)
        d4 = self.decoder4(bottleneck, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return self.output(d1), presence_logits


class SegmentationDataset(Dataset):


    def __init__(self, rows: Sequence[dict[str, Any]], augment: bool, seed: int):
        self.rows = list(rows)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _image_stack(row: dict[str, Any]) -> np.ndarray:
        current = str(row["image_path"])
        if Settings.Segmentation.USE_2_5D:
            paths = (
                str(row.get("previous_image_path") or current),
                current,
                str(row.get("next_image_path") or current),
            )
        else:
            paths = (current,)
        return np.stack(
            [ImageCache.get(path).astype(np.float32) / 255.0 for path in paths],
            axis=0,
        )

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = self._image_stack(row)
        mask = MaskManager.read_binary(row["manual_mask_path"]).astype(np.float32)
        heart_present = float(_as_int(row.get("heart_present"), int(mask.any())))
        sample_weight = float(_as_float(row.get("sample_weight"), 1.0))

        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            if Settings.Segmentation.AUGMENT_HORIZONTAL_FLIP and rng.random() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.fliplr(mask).copy()

            size = Settings.Image.SEGMENTATION_SIZE
            angle = float(
                rng.uniform(
                    -Settings.Segmentation.AUGMENT_ROTATION_DEGREES,
                    Settings.Segmentation.AUGMENT_ROTATION_DEGREES,
                )
            )
            scale = float(
                rng.uniform(
                    Settings.Segmentation.AUGMENT_SCALE_MIN,
                    Settings.Segmentation.AUGMENT_SCALE_MAX,
                )
            )
            shift = Settings.Segmentation.AUGMENT_TRANSLATION_FRACTION * size
            tx, ty = map(float, rng.uniform(-shift, shift, size=2))
            matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
            matrix[:, 2] += (tx, ty)
            image = np.stack(
                [
                    cv2.warpAffine(
                        channel,
                        matrix,
                        (size, size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    for channel in image
                ],
                axis=0,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (size, size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            gamma = float(
                rng.uniform(
                    Settings.Segmentation.AUGMENT_GAMMA_MIN,
                    Settings.Segmentation.AUGMENT_GAMMA_MAX,
                )
            )
            contrast = float(rng.uniform(0.90, 1.10))
            brightness = float(rng.uniform(-0.04, 0.04))
            image = np.power(np.clip(image, 0.0, 1.0), gamma)
            image = np.clip(image * contrast + brightness, 0.0, 1.0)
            if rng.random() < Settings.Segmentation.AUGMENT_BLUR_PROBABILITY:
                image = np.stack(
                    [cv2.GaussianBlur(channel, (3, 3), 0) for channel in image],
                    axis=0,
                )
            noise_std = float(rng.uniform(0.0, Settings.Segmentation.AUGMENT_NOISE_STD_MAX))
            if noise_std > 0:
                image = np.clip(
                    image + rng.normal(0.0, noise_std, size=image.shape).astype(np.float32),
                    0.0,
                    1.0,
                )

        return (
            torch.from_numpy(np.ascontiguousarray(image)).float(),
            torch.from_numpy(np.ascontiguousarray(mask > 0.5)).unsqueeze(0).float(),
            torch.tensor(heart_present, dtype=torch.float32),
            torch.tensor(sample_weight, dtype=torch.float32),
            str(row["patient_id"]),
        )


class SegmentationInferenceDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = SegmentationDataset._image_stack(self.rows[index])
        return torch.from_numpy(np.ascontiguousarray(image)).float(), int(index)


class SegmentationManager:
    """Train, calibrate, and run patient-level out-of-fold segmentation.

    Each patient's predicted mask comes from a model that excluded that patient
    from training and threshold calibration. This is the central leakage barrier.
    """


    PREDICTION_FIELDS = (
        "image_token",
        "image_path",
        "patient_id",
        "series_id",
        "segmentation_fold",
        "predicted_attention_mask_path",
        "attention_valid_initial",
        "attention_valid_final",
        "attention_invalid_reason_initial",
        "attention_invalid_reason_final",
        "attention_repair_method",
        "attention_threshold_used",
        "attention_area_ratio",
        "attention_peak_probability",
        "attention_boundary_touch_fraction",
        "attention_heart_present",
        "attention_presence_probability",
        "attention_presence_threshold",
        "attention_tta_disagreement",
        "attention_mean_entropy",
        "attention_uncertainty_score",
        "attention_sequence_inconsistency",
        "attention_centroid_x",
        "attention_centroid_y",
        "attention_prior_deviation",
        "checkpoint_fingerprint",
    )

    @staticmethod
    def _checkpoint_path(workspace: Workspace, fold: int) -> Path:
        return workspace.checkpoints / f"attention_unet_fold_{int(fold)}.pt"

    @staticmethod
    def _torch_load(path: Path) -> Any:
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    @staticmethod
    def _dice_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        intersection = (probability * targets).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (2.0 * intersection + 1e-6) / (denominator + 1e-6)

    @staticmethod
    def _loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        presence_logits: torch.Tensor,
        presence_targets: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:


        probability = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = probability * targets + (1.0 - probability) * (1.0 - targets)
        focal = ((1.0 - pt).pow(Settings.Segmentation.FOCAL_GAMMA) * bce).mean(
            dim=(1, 2, 3)
        )

        tp = (probability * targets).sum(dim=(1, 2, 3))
        fp = (probability * (1.0 - targets)).sum(dim=(1, 2, 3))
        fn = ((1.0 - probability) * targets).sum(dim=(1, 2, 3))
        tversky = (tp + 1e-6) / (
            tp
            + Settings.Segmentation.TVERSKY_ALPHA_FP * fp
            + Settings.Segmentation.TVERSKY_BETA_FN * fn
            + 1e-6
        )
        segmentation_per_sample = (
            Settings.Segmentation.FOCAL_WEIGHT * focal
            + Settings.Segmentation.TVERSKY_WEIGHT * (1.0 - tversky)
        )

        sample_weights = sample_weights.float().clamp_min(1e-3)
        segmentation_loss = (
            segmentation_per_sample * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-6)
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits.float(), presence_targets.float(), reduction="none"
        )
        presence_loss = (presence_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        return segmentation_loss + Settings.Segmentation.PRESENCE_LOSS_WEIGHT * presence_loss

    @staticmethod
    def _manual_fingerprint(rows: Sequence[dict[str, Any]], fold: int) -> str:
        payload = {
            "schema": "simple-attention-2p5d-presence-v3",
            "fold": int(fold),
            "size": Settings.Image.SEGMENTATION_SIZE,
            "input_channels": Settings.Segmentation.INPUT_CHANNELS,
            "use_2_5d": Settings.Segmentation.USE_2_5D,
            "base_channels": Settings.Segmentation.BASE_CHANNELS,
            "epochs": Settings.Segmentation.EPOCHS,
            "learning_rate": Settings.Segmentation.LEARNING_RATE,
            "weight_decay": Settings.Segmentation.WEIGHT_DECAY,
            "loss": {
                "focal": Settings.Segmentation.FOCAL_WEIGHT,
                "tversky": Settings.Segmentation.TVERSKY_WEIGHT,
                "presence": Settings.Segmentation.PRESENCE_LOSS_WEIGHT,
                "alpha_fp": Settings.Segmentation.TVERSKY_ALPHA_FP,
                "beta_fn": Settings.Segmentation.TVERSKY_BETA_FN,
            },
            "seed": Settings.Runtime.RANDOM_SEED,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        for row in sorted(rows, key=lambda item: item["image_token"]):
            mask_path = Path(row["manual_mask_path"])
            image_path = Path(row["image_path"])
            image_stat = image_path.stat()
            digest.update(str(row["image_token"]).encode("utf-8"))
            digest.update(str(row.get("segmentation_target_type", "")).encode("utf-8"))
            digest.update(str(row.get("annotation_source", "")).encode("utf-8"))
            digest.update(str(row.get("sample_weight", 1.0)).encode("ascii"))
            digest.update(str(image_stat.st_size).encode("ascii"))
            digest.update(str(image_stat.st_mtime_ns).encode("ascii"))
            digest.update(FileManager.sha256_file(mask_path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _validation_patients(patient_ids: Iterable[str], target_fold: int) -> set[str]:
        ordered = sorted(
            set(map(str, patient_ids)),
            key=lambda patient_id: hashlib.sha256(
                f"{Settings.Runtime.RANDOM_SEED}|validation|{target_fold}|{patient_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        count = max(
            1,
            int(
                round(
                    len(ordered)
                    * Settings.Segmentation.VALIDATION_PATIENT_FRACTION
                )
            ),
        )
        count = min(count, max(1, len(ordered) - 1))
        return set(ordered[:count])

    @staticmethod
    def _ensure_presence_coverage(
        rows: Sequence[dict[str, Any]],
        validation_patients: set[str],
        target_fold: int,
    ) -> set[str]:


        patients = sorted({str(row["patient_id"]) for row in rows})
        has_label: dict[int, set[str]] = {0: set(), 1: set()}
        for row in rows:
            has_label[_as_int(row.get("heart_present"), 1)].add(
                str(row["patient_id"])
            )
        validation = set(validation_patients)

        def ordered(values: set[str], salt: str) -> list[str]:
            return sorted(
                values,
                key=lambda patient: hashlib.sha256(
                    f"{Settings.Runtime.RANDOM_SEED}|{target_fold}|{salt}|{patient}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )

        for label in (0, 1):
            labelled = has_label[label]
            if len(labelled) < 2:
                continue
            if not (validation & labelled):
                incoming = ordered(labelled - validation, f"incoming-{label}")[0]
                removable = [
                    patient
                    for patient in validation
                    if patient not in labelled
                    and all(
                        len((validation - {patient}) & has_label[other]) > 0
                        for other in (0, 1)
                        if len(has_label[other]) >= 2
                        and (validation & has_label[other])
                    )
                ]
                if removable:
                    validation.remove(ordered(set(removable), f"remove-{label}")[0])
                validation.add(incoming)
            if not (labelled - validation):

                outgoing = ordered(validation & labelled, f"outgoing-{label}")[-1]
                validation.remove(outgoing)
                replacement_candidates = set(patients) - validation - {outgoing}
                if replacement_candidates:
                    validation.add(
                        ordered(replacement_candidates, f"replacement-{label}")[0]
                    )
        if not validation or len(validation) >= len(patients):
            return set(validation_patients)
        return validation

    @staticmethod
    def _training_sampler(
        rows: Sequence[dict[str, Any]],
        seed: int,
    ) -> WeightedRandomSampler:


        rows = list(rows)
        series_by_patient: dict[str, set[str]] = defaultdict(set)
        clusters_by_series: dict[tuple[str, str], set[str]] = defaultdict(set)
        cluster_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for row in rows:
            patient = str(row["patient_id"])
            series = str(row.get("sequence_group_id") or row["series_id"])
            cluster = str(row.get("perceptual_hash") or row["image_token"])
            series_by_patient[patient].add(series)
            clusters_by_series[(patient, series)].add(cluster)
            cluster_counts[(patient, series, cluster)] += 1

        base_weights: list[float] = []
        labels: list[int] = []
        for row in rows:
            patient = str(row["patient_id"])
            series = str(row.get("sequence_group_id") or row["series_id"])
            cluster = str(row.get("perceptual_hash") or row["image_token"])
            weight = 1.0
            weight /= max(1, len(series_by_patient[patient]))
            weight /= max(1, len(clusters_by_series[(patient, series)]))
            weight /= max(1, cluster_counts[(patient, series, cluster)])
            base_weights.append(weight)
            labels.append(_as_int(row.get("heart_present"), 1))

        base = np.asarray(base_weights, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.int64)
        positive = labels_array == 1
        negative = labels_array == 0
        if positive.any() and negative.any():
            target_negative = float(Settings.Segmentation.NEGATIVE_TARGET_FRACTION)
            target_positive = 1.0 - target_negative
            base[positive] *= target_positive / max(base[positive].sum(), 1e-12)
            base[negative] *= target_negative / max(base[negative].sum(), 1e-12)
        else:
            base /= max(base.sum(), 1e-12)

        generator = torch.Generator().manual_seed(int(seed))
        return WeightedRandomSampler(
            weights=torch.as_tensor(base, dtype=torch.double),
            num_samples=len(rows),
            replacement=True,
            generator=generator,
        )

    @staticmethod
    def _mask_features(mask: np.ndarray) -> dict[str, float]:
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        area, boundary = SegmentationManager._mask_geometry(mask)
        if not mask.any():
            return {
                "area": area,
                "boundary": boundary,
                "centroid_x": np.nan,
                "centroid_y": np.nan,
                "aspect_ratio": np.nan,
            }
        ys, xs = np.nonzero(mask)
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        size = float(mask.shape[0])
        return {
            "area": area,
            "boundary": boundary,
            "centroid_x": float(xs.mean() / max(1.0, mask.shape[1] - 1)),
            "centroid_y": float(ys.mean() / max(1.0, mask.shape[0] - 1)),
            "aspect_ratio": float(width / max(height, 1.0)),
        }

    @staticmethod
    def _geometry_prior(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        features = []
        for row in rows:
            if _as_int(row.get("heart_present"), 1) != 1:
                continue
            try:
                features.append(
                    SegmentationManager._mask_features(
                        MaskManager.read_binary(row["manual_mask_path"])
                    )
                )
            except Exception:
                continue
        keys = ("area", "centroid_x", "centroid_y", "aspect_ratio")
        if not features:
            return {"count": 0, "median": {}, "scale": {}}
        median: dict[str, float] = {}
        scale: dict[str, float] = {}
        for key in keys:
            values = np.asarray([item[key] for item in features], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            median[key] = center
            scale[key] = max(1e-3, 1.4826 * mad)
        return {"count": len(features), "median": median, "scale": scale}

    @staticmethod
    def _prior_deviation(mask: np.ndarray, prior: dict[str, Any] | None) -> float:
        if not prior or _as_int(prior.get("count"), 0) < 5 or not np.asarray(mask).any():
            return 0.0
        features = SegmentationManager._mask_features(mask)
        deviations = []
        for key, center in dict(prior.get("median", {})).items():
            value = features.get(key, np.nan)
            scale = _as_float(dict(prior.get("scale", {})).get(key), 0.0)
            if np.isfinite(value) and scale > 0:
                deviations.append(abs(float(value) - float(center)) / scale)
        return float(max(deviations)) if deviations else 0.0

    @staticmethod
    def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
        labels = np.asarray(labels, dtype=np.int64)
        predictions = np.asarray(predictions, dtype=np.int64)
        values = []
        for label in (0, 1):
            selector = labels == label
            if selector.any():
                values.append(float(np.mean(predictions[selector] == label)))
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _calibrate_threshold(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        geometry_prior: dict[str, Any],
    ) -> dict[str, Any]:
        probabilities: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        presence_probabilities: list[float] = []
        presence_targets: list[int] = []
        patients: list[str] = []
        model.eval()
        with torch.inference_mode():
            for images, masks, target_presence, _weights, patient_ids in loader:
                images = RuntimeManager.move_tensor(images, device)
                with RuntimeManager.autocast(device):
                    logits, presence_logits = model(images)
                    probability = torch.sigmoid(logits)
                    presence_probability = torch.sigmoid(presence_logits)
                probabilities.extend(probability.float().cpu().numpy()[:, 0])
                targets.extend(masks.numpy()[:, 0])
                presence_probabilities.extend(
                    presence_probability.float().cpu().numpy().tolist()
                )
                presence_targets.extend(target_presence.numpy().astype(int).tolist())
                patients.extend(map(str, patient_ids))

        labels_array = np.asarray(presence_targets, dtype=np.int64)
        presence_array = np.asarray(presence_probabilities, dtype=np.float64)
        best_presence = {
            "threshold": Settings.Segmentation.DEFAULT_PRESENCE_THRESHOLD,
            "balanced_accuracy": 0.0,
        }
        if len(np.unique(labels_array)) >= 2:
            for threshold in Settings.Segmentation.PRESENCE_CALIBRATION_THRESHOLDS:
                prediction = (presence_array >= threshold).astype(np.int64)
                balanced = SegmentationManager._balanced_accuracy(
                    labels_array, prediction
                )
                candidate = {
                    "threshold": float(threshold),
                    "balanced_accuracy": balanced,
                }
                if (
                    candidate["balanced_accuracy"],
                    -abs(candidate["threshold"] - 0.5),
                ) > (
                    best_presence["balanced_accuracy"],
                    -abs(best_presence["threshold"] - 0.5),
                ):
                    best_presence = candidate
        elif len(labels_array):
            best_presence["balanced_accuracy"] = 1.0

        best = None
        for threshold in Settings.Segmentation.CALIBRATION_THRESHOLDS:
            by_patient: dict[str, list[float]] = defaultdict(list)
            invalid = 0
            for probability, target, target_presence, patient_id in zip(
                probabilities, targets, presence_targets, patients
            ):
                prediction = SegmentationManager._candidate_mask(
                    probability, threshold
                )
                if int(target_presence) == 1:
                    intersection = float(
                        np.logical_and(prediction, target > 0.5).sum()
                    )
                    denominator = float(
                        prediction.sum() + (target > 0.5).sum()
                    )
                    quality = (2.0 * intersection + 1e-6) / (
                        denominator + 1e-6
                    )
                    valid, _ = SegmentationManager._validate_mask(
                        prediction, probability, geometry_prior
                    )
                    invalid += int(not valid)
                else:

                    quality = 1.0 - min(1.0, float(prediction.mean()) / 0.10)
                by_patient[patient_id].append(float(quality))
            patient_scores = [
                float(np.mean(values)) for values in by_patient.values()
            ]
            mean_score = float(np.mean(patient_scores)) if patient_scores else 0.0
            invalid_rate = float(invalid / max(1, sum(presence_targets)))
            score = mean_score - 0.20 * invalid_rate
            candidate = {
                "threshold": float(threshold),
                "patient_balanced_segmentation_score": mean_score,
                "invalid_rate_positive": invalid_rate,
                "score": score,
            }
            if best is None or (
                candidate["score"], -abs(threshold - 0.5)
            ) > (best["score"], -abs(best["threshold"] - 0.5)):
                best = candidate

        result = best or {
            "threshold": Settings.Segmentation.DEFAULT_THRESHOLD,
            "patient_balanced_segmentation_score": 0.0,
            "invalid_rate_positive": 1.0,
            "score": -1.0,
        }
        result["presence_threshold"] = float(best_presence["threshold"])
        result["presence_balanced_accuracy"] = float(
            best_presence["balanced_accuracy"]
        )
        return result

    # For target fold k, every patient assigned to k is excluded from both model
    # fitting and threshold calibration. The saved checkpoint therefore produces
    # genuinely out-of-fold masks for those patients.
    @staticmethod
    def train_crossfit(
        accepted_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        device: torch.device,
        force: bool = False,
    ) -> dict[int, Path]:


        RuntimeManager.seed_everything(include_cuda=device.type == "cuda")
        checkpoint_map: dict[int, Path] = {}
        fold_summaries = []
        all_patients = sorted({str(row["patient_id"]) for row in accepted_rows})

        for fold in range(Settings.Segmentation.FOLDS):
            fold_started = time.perf_counter()
            checkpoint_path = SegmentationManager._checkpoint_path(workspace, fold)
            non_test_rows = [
                row
                for row in accepted_rows
                if int(row["segmentation_fold"]) != fold
            ]
            validation_patients = SegmentationManager._validation_patients(
                (row["patient_id"] for row in non_test_rows), fold
            )
            validation_patients = SegmentationManager._ensure_presence_coverage(
                non_test_rows, validation_patients, fold
            )
            train_rows = [
                row
                for row in non_test_rows
                if row["patient_id"] not in validation_patients
            ]
            validation_rows = [
                row
                for row in non_test_rows
                if row["patient_id"] in validation_patients
            ]
            fingerprint = SegmentationManager._manual_fingerprint(
                non_test_rows, fold
            )

            if checkpoint_path.is_file() and not force:
                checkpoint = SegmentationManager._torch_load(checkpoint_path)
                if checkpoint.get("fingerprint") == fingerprint:
                    print(
                        f"[ATTENTION] Fold {fold}: compatible 2.5D checkpoint reused."
                    )
                    checkpoint_map[fold] = checkpoint_path
                    fold_summaries.append(
                        {
                            "fold": fold,
                            "status": "reused",
                            "checkpoint": str(checkpoint_path),
                            "fingerprint": fingerprint,
                            "calibration": checkpoint.get("calibration", {}),
                        }
                    )
                    continue

            if not train_rows or not validation_rows:
                raise RuntimeError(f"Fold {fold}: train or validation split is empty.")

            train_positive = sum(
                _as_int(row.get("heart_present"), 1) == 1 for row in train_rows
            )
            train_negative = len(train_rows) - train_positive
            validation_positive = sum(
                _as_int(row.get("heart_present"), 1) == 1
                for row in validation_rows
            )
            validation_negative = len(validation_rows) - validation_positive
            print(
                f"[ATTENTION] Fold {fold}: train={len(train_rows)} "
                f"(+{train_positive}/-{train_negative}), validation={len(validation_rows)} "
                f"(+{validation_positive}/-{validation_negative}), device={device.type}"
            )

            train_dataset = SegmentationDataset(
                train_rows,
                augment=True,
                seed=Settings.Runtime.RANDOM_SEED + fold * 1000,
            )
            validation_dataset = SegmentationDataset(
                validation_rows,
                augment=False,
                seed=Settings.Runtime.RANDOM_SEED,
            )
            train_sampler = SegmentationManager._training_sampler(
                train_rows, Settings.Runtime.RANDOM_SEED + fold
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=RuntimeManager.train_batch_size(device),
                sampler=train_sampler,
                shuffle=False,
                num_workers=Settings.Runtime.NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=RuntimeManager.inference_batch_size(device),
                shuffle=False,
                num_workers=Settings.Runtime.NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )

            model = RuntimeManager.prepare_model(AttentionUNet(), device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Settings.Segmentation.LEARNING_RATE,
                weight_decay=Settings.Segmentation.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=Settings.Segmentation.LR_REDUCE_FACTOR,
                patience=Settings.Segmentation.LR_REDUCE_PATIENCE,
                min_lr=Settings.Segmentation.MIN_LEARNING_RATE,
            )
            scaler = RuntimeManager.grad_scaler(device)
            best_state = None
            best_metric = -1.0
            best_epoch = 0
            patience = 0
            history = []

            for epoch in range(1, Settings.Segmentation.EPOCHS + 1):
                train_dataset.set_epoch(epoch)
                model.train()
                train_losses = []
                for (
                    images,
                    masks,
                    presence_targets,
                    sample_weights,
                    _patient_ids,
                ) in train_loader:
                    images = RuntimeManager.move_tensor(images, device)
                    masks = RuntimeManager.move_tensor(masks, device)
                    presence_targets = RuntimeManager.move_tensor(
                        presence_targets, device
                    )
                    sample_weights = RuntimeManager.move_tensor(
                        sample_weights, device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with RuntimeManager.autocast(device):
                        logits, presence_logits = model(images)
                        loss = SegmentationManager._loss(
                            logits,
                            masks,
                            presence_logits,
                            presence_targets,
                            sample_weights,
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_losses.append(float(loss.detach().cpu()))

                model.eval()
                validation_losses = []
                by_patient_scores: dict[str, list[float]] = defaultdict(list)
                presence_labels: list[int] = []
                presence_predictions: list[int] = []
                with torch.inference_mode():
                    for (
                        images,
                        masks,
                        presence_targets,
                        sample_weights,
                        patient_ids,
                    ) in validation_loader:
                        images = RuntimeManager.move_tensor(images, device)
                        masks = RuntimeManager.move_tensor(masks, device)
                        presence_targets_device = RuntimeManager.move_tensor(
                            presence_targets, device
                        )
                        sample_weights = RuntimeManager.move_tensor(
                            sample_weights, device
                        )
                        with RuntimeManager.autocast(device):
                            logits, presence_logits = model(images)
                            loss = SegmentationManager._loss(
                                logits,
                                masks,
                                presence_logits,
                                presence_targets_device,
                                sample_weights,
                            )
                        validation_losses.append(float(loss.cpu()))
                        binary = (torch.sigmoid(logits) >= 0.5).float()
                        intersection = (binary * masks).sum(dim=(1, 2, 3))
                        denominator = binary.sum(dim=(1, 2, 3)) + masks.sum(
                            dim=(1, 2, 3)
                        )
                        dice = (2 * intersection + 1e-6) / (
                            denominator + 1e-6
                        )
                        negative_quality = 1.0 - binary.mean(dim=(1, 2, 3))
                        target_presence_cpu = presence_targets.numpy().astype(int)
                        sample_scores = torch.where(
                            presence_targets_device > 0.5,
                            dice,
                            negative_quality,
                        ).float().cpu().numpy()
                        for patient_id, score in zip(patient_ids, sample_scores):
                            by_patient_scores[str(patient_id)].append(float(score))
                        presence_labels.extend(target_presence_cpu.tolist())
                        presence_predictions.extend(
                            (
                                torch.sigmoid(presence_logits) >= 0.5
                            ).long().cpu().numpy().tolist()
                        )

                mean_train = float(np.mean(train_losses))
                mean_val = float(np.mean(validation_losses))
                patient_balanced = float(
                    np.mean(
                        [np.mean(values) for values in by_patient_scores.values()]
                    )
                )
                presence_balanced = SegmentationManager._balanced_accuracy(
                    np.asarray(presence_labels),
                    np.asarray(presence_predictions),
                )
                selection_metric = 0.80 * patient_balanced + 0.20 * presence_balanced
                scheduler.step(selection_metric)
                current_lr = float(optimizer.param_groups[0]["lr"])
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": mean_train,
                        "validation_loss": mean_val,
                        "patient_balanced_segmentation_score_0_5": patient_balanced,
                        "presence_balanced_accuracy_0_5": presence_balanced,
                        "selection_metric": selection_metric,
                        "learning_rate": current_lr,
                    }
                )
                print(
                    f"  epoch={epoch:02d} train_loss={mean_train:.4f} "
                    f"val_loss={mean_val:.4f} seg={patient_balanced:.4f} "
                    f"presence={presence_balanced:.4f} metric={selection_metric:.4f} "
                    f"lr={current_lr:.2e}"
                )
                if selection_metric > best_metric + 1e-5:
                    best_metric = selection_metric
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= Settings.Segmentation.EARLY_STOPPING_PATIENCE:
                        print(f"  early stopping after epoch {epoch}.")
                        break

            if best_state is None:
                raise RuntimeError(f"Fold {fold}: no model checkpoint was selected.")
            model.load_state_dict(best_state)
            geometry_prior = SegmentationManager._geometry_prior(train_rows)
            calibration = SegmentationManager._calibrate_threshold(
                model, validation_loader, device, geometry_prior
            )
            checkpoint = {
                "schema": "simple-attention-2p5d-presence-v3",
                "state_dict": best_state,
                "fingerprint": fingerprint,
                "fold": fold,
                "base_channels": Settings.Segmentation.BASE_CHANNELS,
                "input_channels": Settings.Segmentation.INPUT_CHANNELS,
                "best_epoch": best_epoch,
                "best_selection_metric": best_metric,
                "calibration": calibration,
                "geometry_prior": geometry_prior,
                "train_positive_targets": train_positive,
                "train_negative_targets": train_negative,
                "train_patients": sorted(
                    {str(row["patient_id"]) for row in train_rows}
                ),
                "validation_patients": sorted(validation_patients),
                "excluded_test_patients": sorted(
                    patient
                    for patient in all_patients
                    if any(
                        str(row["patient_id"]) == patient
                        and int(row["segmentation_fold"]) == fold
                        for row in accepted_rows
                    )
                ),
                "history": history,
            }
            temporary = checkpoint_path.with_name(
                f".{checkpoint_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            torch.save(checkpoint, temporary)
            os.replace(temporary, checkpoint_path)
            checkpoint_map[fold] = checkpoint_path
            fold_summaries.append(
                {
                    "fold": fold,
                    "status": "trained",
                    "checkpoint": str(checkpoint_path),
                    "fingerprint": fingerprint,
                    "best_epoch": best_epoch,
                    "best_selection_metric": best_metric,
                    "calibration": calibration,
                    "geometry_prior_count": geometry_prior.get("count", 0),
                    "elapsed": RuntimeManager.format_seconds(
                        time.perf_counter() - fold_started
                    ),
                }
            )
            del model, optimizer, scheduler, scaler, train_loader, validation_loader
            del train_dataset, validation_dataset, checkpoint, best_state, train_sampler
            RuntimeManager.release(device)

        FileManager.write_json(
            workspace.training_summary,
            {
                "training_mode": "manual_positive_and_explicit_negative_crossfit_2p5d",
                "device": device.type,
                "accepted_training_targets": len(accepted_rows),
                "heart_present_targets": sum(
                    _as_int(row.get("heart_present"), 1) == 1
                    for row in accepted_rows
                ),
                "no_heart_targets": sum(
                    _as_int(row.get("heart_present"), 1) == 0
                    for row in accepted_rows
                ),
                "folds": fold_summaries,
            },
        )
        ImageCache.clear()
        return checkpoint_map

    @staticmethod
    def load_checkpoint_map(workspace: Workspace) -> dict[int, Path]:
        result = {}
        for fold in range(Settings.Segmentation.FOLDS):
            path = SegmentationManager._checkpoint_path(workspace, fold)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint for fold {fold}: {path}. Run train_attention()."
                )
            result[fold] = path
        return result

    @staticmethod
    def compatible_checkpoint_map(
        accepted_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
    ) -> dict[int, Path] | None:
        result: dict[int, Path] = {}
        for fold in range(Settings.Segmentation.FOLDS):
            checkpoint_path = SegmentationManager._checkpoint_path(workspace, fold)
            if not checkpoint_path.is_file():
                return None
            non_test_rows = [
                row
                for row in accepted_rows
                if int(row["segmentation_fold"]) != fold
            ]
            expected = SegmentationManager._manual_fingerprint(non_test_rows, fold)
            try:
                checkpoint = SegmentationManager._torch_load(checkpoint_path)
            except Exception:
                return None
            if checkpoint.get("fingerprint") != expected:
                return None
            if checkpoint.get("schema") != "simple-attention-2p5d-presence-v3":
                return None
            result[fold] = checkpoint_path
        return result

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return binary
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if count <= 1:
            return binary
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == label).astype(np.uint8)

    @staticmethod
    def _mask_geometry(mask: np.ndarray) -> tuple[float, float]:
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        area = float(mask.mean())
        if not mask.any():
            return area, 0.0
        border = np.zeros_like(mask, dtype=bool)
        border[:2, :] = True
        border[-2:, :] = True
        border[:, :2] = True
        border[:, -2:] = True
        boundary_touch = float(
            np.logical_and(mask > 0, border).sum() / max(1, mask.sum())
        )
        return area, boundary_touch

    @staticmethod
    def _candidate_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
        mask = (probability >= float(threshold)).astype(np.uint8)
        kernel_size = int(Settings.Segmentation.REPAIR_MORPHOLOGY_KERNEL)
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return SegmentationManager._largest_component(mask)

    @staticmethod
    def _validate_mask(
        mask: np.ndarray,
        probability: np.ndarray,
        geometry_prior: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        area, boundary = SegmentationManager._mask_geometry(mask)
        peak = float(np.max(probability))
        reasons = []
        if area < Settings.Segmentation.PREDICTION_MIN_AREA_RATIO:
            reasons.append("area_too_small")
        if area > Settings.Segmentation.PREDICTION_MAX_AREA_RATIO:
            reasons.append("area_too_large")
        if peak < Settings.Segmentation.PREDICTION_MIN_PEAK_PROBABILITY:
            reasons.append("low_peak_probability")
        if boundary > Settings.Segmentation.REPAIR_MAX_BOUNDARY_TOUCH:
            reasons.append("touches_boundary")
        prior_deviation = SegmentationManager._prior_deviation(mask, geometry_prior)
        if prior_deviation > Settings.Segmentation.PRIOR_MAX_ROBUST_Z:
            reasons.append("geometry_outlier")
        return not reasons, ";".join(reasons)

    @staticmethod
    def _repair_probability(
        probability: np.ndarray,
        threshold: float,
        geometry_prior: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, bool, str, float, str, float]:


        candidates = []
        thresholds = [threshold] + [
            float(np.clip(threshold + offset, 0.05, 0.95))
            for offset in Settings.Segmentation.REPAIR_THRESHOLD_OFFSETS
        ]
        for candidate_threshold in sorted(set(thresholds)):
            mask = SegmentationManager._candidate_mask(
                probability, candidate_threshold
            )
            valid, reason = SegmentationManager._validate_mask(
                mask, probability, geometry_prior
            )
            _area, boundary = SegmentationManager._mask_geometry(mask)
            prior_deviation = SegmentationManager._prior_deviation(
                mask, geometry_prior
            )
            mean_inside = (
                float(probability[mask > 0].mean()) if mask.any() else 0.0
            )
            score = (
                mean_inside
                - 0.50 * boundary
                - 0.06 * min(prior_deviation, 10.0)
            )
            candidates.append(
                (
                    valid,
                    score,
                    mask,
                    candidate_threshold,
                    reason,
                    prior_deviation,
                )
            )
        valid_candidates = [candidate for candidate in candidates if candidate[0]]
        if valid_candidates:
            _, _, mask, used_threshold, _, prior_deviation = max(
                valid_candidates, key=lambda item: item[1]
            )
            return (
                mask,
                True,
                "adaptive_threshold_with_fold_prior",
                float(used_threshold),
                "",
                float(prior_deviation),
            )
        _, _, mask, used_threshold, reason, prior_deviation = max(
            candidates, key=lambda item: item[1]
        )
        return (
            mask,
            False,
            "unresolved",
            float(used_threshold),
            reason,
            float(prior_deviation),
        )

    @staticmethod
    def _prediction_fingerprint(
        rows: Sequence[dict[str, Any]], checkpoint_map: dict[int, Path]
    ) -> str:
        digest = hashlib.sha256()
        settings_payload = {
            "schema": "simple-predictions-2p5d-presence-v5",
            "thresholds": Settings.Segmentation.CALIBRATION_THRESHOLDS,
            "presence_thresholds": Settings.Segmentation.PRESENCE_CALIBRATION_THRESHOLDS,
            "area": [
                Settings.Segmentation.PREDICTION_MIN_AREA_RATIO,
                Settings.Segmentation.PREDICTION_MAX_AREA_RATIO,
            ],
            "repair_offsets": Settings.Segmentation.REPAIR_THRESHOLD_OFFSETS,
            "tta_contrast": Settings.Segmentation.TTA_CONTRAST_FACTOR,
            "presence_override_peak": Settings.Segmentation.PRESENCE_SEGMENTATION_OVERRIDE_PEAK,
            "prior_z": Settings.Segmentation.PRIOR_MAX_ROBUST_Z,
        }
        digest.update(json.dumps(settings_payload, sort_keys=True).encode("utf-8"))
        digest.update(str(len(rows)).encode("ascii"))
        for row in rows:
            image_path = Path(row["image_path"])
            image_stat = image_path.stat()
            digest.update(str(row["image_token"]).encode("utf-8"))
            digest.update(str(row["segmentation_fold"]).encode("ascii"))
            digest.update(str(row.get("previous_image_path", "")).encode("utf-8"))
            digest.update(str(row.get("next_image_path", "")).encode("utf-8"))
            digest.update(str(image_stat.st_size).encode("ascii"))
            digest.update(str(image_stat.st_mtime_ns).encode("ascii"))
        for fold, path in sorted(checkpoint_map.items()):
            digest.update(str(fold).encode("ascii"))
            digest.update(FileManager.sha256_file(path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def load_cached_predictions(
        rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        checkpoint_map: dict[int, Path],
    ) -> list[dict[str, Any]] | None:
        fingerprint = SegmentationManager._prediction_fingerprint(
            rows, checkpoint_map
        )
        summary = FileManager.read_json(workspace.prediction_summary, {}) or {}
        audit = FileManager.read_csv(workspace.prediction_audit)
        masks_complete = bool(audit) and all(
            Path(row.get("predicted_attention_mask_path", "")).is_file()
            for row in audit
        )
        required_fields = {
            "attention_presence_probability",
            "attention_uncertainty_score",
            "attention_sequence_inconsistency",
            "attention_centroid_x",
            "attention_centroid_y",
            "attention_heart_present",
        }
        schema_complete = bool(audit) and required_fields.issubset(audit[0].keys())
        if (
            summary.get("fingerprint") == fingerprint
            and len(audit) == len(rows)
            and masks_complete
            and schema_complete
        ):
            return audit
        return None

    @staticmethod
    def _prediction_part_paths(
        workspace: Workspace, fold: int
    ) -> tuple[Path, Path]:
        csv_path = workspace.prediction_parts_dir / f"fold_{int(fold)}.csv"
        json_path = workspace.prediction_parts_dir / f"fold_{int(fold)}.json"
        return csv_path, json_path

    @staticmethod
    def _prediction_part_fingerprint(
        fold: int,
        fold_rows: Sequence[dict[str, Any]],
        checkpoint_path: Path,
    ) -> str:
        digest = hashlib.sha256()
        payload = {
            "schema": "simple-prediction-part-2p5d-presence-v5",
            "fold": int(fold),
            "thresholds": Settings.Segmentation.CALIBRATION_THRESHOLDS,
            "presence_thresholds": Settings.Segmentation.PRESENCE_CALIBRATION_THRESHOLDS,
            "repair_offsets": Settings.Segmentation.REPAIR_THRESHOLD_OFFSETS,
            "morphology": Settings.Segmentation.REPAIR_MORPHOLOGY_KERNEL,
            "tta_contrast": Settings.Segmentation.TTA_CONTRAST_FACTOR,
            "presence_override_peak": Settings.Segmentation.PRESENCE_SEGMENTATION_OVERRIDE_PEAK,
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        digest.update(FileManager.sha256_file(checkpoint_path).encode("ascii"))
        for row in fold_rows:
            image_path = Path(row["image_path"])
            image_stat = image_path.stat()
            digest.update(str(row["image_token"]).encode("utf-8"))
            digest.update(str(image_stat.st_size).encode("ascii"))
            digest.update(str(image_stat.st_mtime_ns).encode("ascii"))
            digest.update(str(row.get("previous_image_path", "")).encode("utf-8"))
            digest.update(str(row.get("next_image_path", "")).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _uncertainty_metrics(
        probability: np.ndarray,
        alternate_probability: np.ndarray,
        presence_probability: float,
        presence_threshold: float,
        prior_deviation: float,
        final_valid: bool,
    ) -> tuple[float, float, float]:
        probability = np.clip(probability.astype(np.float64), 1e-6, 1.0 - 1e-6)
        entropy = -(
            probability * np.log(probability)
            + (1.0 - probability) * np.log(1.0 - probability)
        ) / math.log(2.0)
        entropy_flat = entropy.reshape(-1)
        top_count = max(1, int(round(0.10 * entropy_flat.size)))
        mean_entropy = float(
            np.partition(entropy_flat, entropy_flat.size - top_count)[-top_count:].mean()
        )
        disagreement = float(
            np.mean(np.abs(probability - alternate_probability))
        )
        disagreement_normalized = min(
            1.0,
            disagreement
            / max(Settings.Segmentation.UNCERTAINTY_DISAGREEMENT_SCALE, 1e-6),
        )
        presence_ambiguity = max(
            0.0,
            1.0
            - abs(float(presence_probability) - float(presence_threshold)) / 0.50,
        )
        prior_normalized = min(
            1.0,
            float(prior_deviation)
            / max(Settings.Segmentation.PRIOR_MAX_ROBUST_Z, 1e-6),
        )
        uncertainty = (
            0.35 * mean_entropy
            + 0.35 * disagreement_normalized
            + 0.20 * presence_ambiguity
            + 0.10 * prior_normalized
            + (0.15 if not final_valid else 0.0)
        )
        return disagreement, mean_entropy, float(min(1.0, uncertainty))

    @staticmethod
    def _apply_sequence_consistency(
        prediction_rows: Sequence[dict[str, Any]],
        dataset_rows: Sequence[dict[str, Any]],
    ) -> None:


        base_by_token = {
            str(row["image_token"]): row for row in dataset_rows
        }
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in prediction_rows:
            token = str(prediction.get("image_token", ""))
            base = base_by_token.get(token, {})
            group = str(
                base.get("sequence_group_id")
                or f"{prediction.get('series_id', '')}::{Path(str(prediction.get('image_path', ''))).parent}"
            )
            prediction["_sequence_index"] = _as_int(
                base.get("sequence_index"), 0
            )
            groups[group].append(prediction)

        def pair_inconsistency(first: dict[str, Any], second: dict[str, Any]) -> float:
            first_present = _as_int(first.get("attention_heart_present"), 1)
            second_present = _as_int(second.get("attention_heart_present"), 1)
            if first_present != second_present:
                return 1.0
            if first_present == 0:
                return 0.0
            first_area = _as_float(first.get("attention_area_ratio"), 0.0)
            second_area = _as_float(second.get("attention_area_ratio"), 0.0)
            area_change = min(
                1.0,
                abs(first_area - second_area)
                / max(0.01, 0.5 * (first_area + second_area)),
            )
            first_x = _as_float(first.get("attention_centroid_x"), np.nan)
            first_y = _as_float(first.get("attention_centroid_y"), np.nan)
            second_x = _as_float(second.get("attention_centroid_x"), np.nan)
            second_y = _as_float(second.get("attention_centroid_y"), np.nan)
            if all(np.isfinite(value) for value in (first_x, first_y, second_x, second_y)):
                centroid_change = min(
                    1.0,
                    math.hypot(first_x - second_x, first_y - second_y) / 0.25,
                )
            else:
                centroid_change = 1.0
            return float(0.55 * area_change + 0.45 * centroid_change)

        for group_rows in groups.values():
            group_rows.sort(key=lambda row: _as_int(row.get("_sequence_index"), 0))
            for index, row in enumerate(group_rows):
                comparisons = []
                if index > 0:
                    comparisons.append(pair_inconsistency(row, group_rows[index - 1]))
                if index + 1 < len(group_rows):
                    comparisons.append(pair_inconsistency(row, group_rows[index + 1]))
                inconsistency = float(np.mean(comparisons)) if comparisons else 0.0
                previous_inconsistency = _as_float(
                    row.get("attention_sequence_inconsistency"), 0.0
                )
                base_uncertainty = max(
                    0.0,
                    _as_float(row.get("attention_uncertainty_score"), 0.0)
                    - 0.20 * previous_inconsistency,
                )
                row["attention_sequence_inconsistency"] = inconsistency
                row["attention_uncertainty_score"] = min(
                    1.0, base_uncertainty + 0.20 * inconsistency
                )
                row.pop("_sequence_index", None)

    @staticmethod
    def evaluate_oof_segmentation(
        workspace: Workspace,
        prediction_rows: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:


        predictions = {
            str(row.get("image_token", "")): row for row in prediction_rows
        }
        audit_rows = [
            row
            for row in FileManager.read_csv(workspace.manual_audit)
            if row.get("status") == "ACCEPTED"
        ]
        metrics: list[dict[str, Any]] = []
        for target_row in audit_rows:
            token = str(target_row.get("image_token", ""))
            prediction_row = predictions.get(token)
            if prediction_row is None:
                continue
            predicted_path = Path(
                prediction_row.get("predicted_attention_mask_path", "")
            )
            target_path = Path(target_row.get("manual_mask_path", ""))
            if not predicted_path.is_file() or not target_path.is_file():
                continue
            predicted = MaskManager.read_binary(predicted_path)
            target = MaskManager.read_binary(target_path)
            predicted_bool = predicted > 0
            target_bool = target > 0
            tp = float(np.logical_and(predicted_bool, target_bool).sum())
            fp = float(np.logical_and(predicted_bool, ~target_bool).sum())
            fn = float(np.logical_and(~predicted_bool, target_bool).sum())
            union = tp + fp + fn
            denominator = 2.0 * tp + fp + fn
            dice = 1.0 if denominator == 0 else 2.0 * tp / denominator
            iou = 1.0 if union == 0 else tp / union
            precision = 1.0 if tp + fp == 0 and not target_bool.any() else tp / max(1.0, tp + fp)
            recall = 1.0 if tp + fn == 0 else tp / max(1.0, tp + fn)
            target_features = SegmentationManager._mask_features(target)
            predicted_features = SegmentationManager._mask_features(predicted)
            if target_bool.any() and predicted_bool.any():
                centroid_distance = float(
                    math.hypot(
                        predicted_features["centroid_x"] - target_features["centroid_x"],
                        predicted_features["centroid_y"] - target_features["centroid_y"],
                    )
                )
            else:
                centroid_distance = np.nan
            metrics.append(
                {
                    "image_token": token,
                    "patient_id": target_row.get("patient_id", ""),
                    "series_id": target_row.get("series_id", ""),
                    "target_type": target_row.get("target_type", ""),
                    "heart_present": _as_int(target_row.get("heart_present"), 1),
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "target_area_ratio": float(target_bool.mean()),
                    "predicted_area_ratio": float(predicted_bool.mean()),
                    "absolute_area_error": abs(
                        float(predicted_bool.mean()) - float(target_bool.mean())
                    ),
                    "false_positive_area_ratio": float(fp / predicted_bool.size),
                    "centroid_distance_normalized": centroid_distance,
                    "presence_probability": _as_float(
                        prediction_row.get("attention_presence_probability"), np.nan
                    ),
                    "predicted_heart_present": _as_int(
                        prediction_row.get("attention_heart_present"), 1
                    ),
                    "uncertainty_score": _as_float(
                        prediction_row.get("attention_uncertainty_score"), np.nan
                    ),
                    "sequence_inconsistency": _as_float(
                        prediction_row.get("attention_sequence_inconsistency"), np.nan
                    ),
                }
            )

        fields = (
            "image_token",
            "patient_id",
            "series_id",
            "target_type",
            "heart_present",
            "dice",
            "iou",
            "precision",
            "recall",
            "target_area_ratio",
            "predicted_area_ratio",
            "absolute_area_error",
            "false_positive_area_ratio",
            "centroid_distance_normalized",
            "presence_probability",
            "predicted_heart_present",
            "uncertainty_score",
            "sequence_inconsistency",
        )
        FileManager.write_csv(workspace.segmentation_metrics, metrics, fields)

        positive = [row for row in metrics if _as_int(row["heart_present"], 1) == 1]
        negative = [row for row in metrics if _as_int(row["heart_present"], 1) == 0]

        def patient_balanced(rows: Sequence[dict[str, Any]], key: str) -> float:
            by_patient: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                value = _as_float(row.get(key), np.nan)
                if np.isfinite(value):
                    by_patient[str(row.get("patient_id", ""))].append(value)
            values = [np.mean(items) for items in by_patient.values() if items]
            return float(np.mean(values)) if values else np.nan

        summary = {
            "evaluated_targets": len(metrics),
            "positive_targets": len(positive),
            "negative_targets": len(negative),
            "patient_balanced_dice_all": patient_balanced(metrics, "dice"),
            "patient_balanced_dice_positive": patient_balanced(positive, "dice"),
            "patient_balanced_iou_positive": patient_balanced(positive, "iou"),
            "patient_balanced_precision_positive": patient_balanced(
                positive, "precision"
            ),
            "patient_balanced_recall_positive": patient_balanced(
                positive, "recall"
            ),
            "no_heart_empty_prediction_rate": (
                float(
                    np.mean(
                        [
                            row["predicted_area_ratio"]
                            < Settings.Segmentation.PREDICTION_MIN_AREA_RATIO
                            for row in negative
                        ]
                    )
                )
                if negative
                else np.nan
            ),
            "no_heart_mean_false_positive_area_ratio": (
                float(np.mean([row["false_positive_area_ratio"] for row in negative]))
                if negative
                else np.nan
            ),
            "metrics_csv": str(workspace.segmentation_metrics),
        }
        FileManager.write_json(workspace.segmentation_metrics_summary, summary)
        if metrics:
            print(
                "[ATTENTION OOF] "
                f"targets={len(metrics)}, positive_dice_patient="
                f"{summary['patient_balanced_dice_positive']:.4f}, "
                f"negative_empty_rate={summary['no_heart_empty_prediction_rate']}"
            )
        return summary

    # Inference is resumable per fold. A fold result is reused only when its
    # checkpoint and all image/context fingerprints match; otherwise that fold alone
    # is recomputed. This makes post-review retraining incremental rather than global.
    @staticmethod
    def predict_all(
        rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        device: torch.device,
        checkpoint_map: dict[int, Path] | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:


        checkpoint_map = checkpoint_map or SegmentationManager.load_checkpoint_map(
            workspace
        )
        prediction_fingerprint = SegmentationManager._prediction_fingerprint(
            rows, checkpoint_map
        )
        if not force:
            cached = SegmentationManager.load_cached_predictions(
                rows, workspace, checkpoint_map
            )
            if cached is not None:
                print(
                    "[ATTENTION] Compatible 2.5D predictions are already cached; "
                    "the GPU is not used."
                )
                SegmentationManager.evaluate_oof_segmentation(workspace, cached)
                return cached

        results: list[dict[str, Any] | None] = [None] * len(rows)
        started = time.perf_counter()
        reused_folds: list[int] = []
        computed_folds: list[int] = []

        for fold in range(Settings.Segmentation.FOLDS):
            fold_indices = [
                index
                for index, row in enumerate(rows)
                if int(row["segmentation_fold"]) == fold
            ]
            fold_rows = [rows[index] for index in fold_indices]
            if not fold_rows:
                continue

            part_csv, part_json = SegmentationManager._prediction_part_paths(
                workspace, fold
            )
            part_fingerprint = SegmentationManager._prediction_part_fingerprint(
                fold, fold_rows, checkpoint_map[fold]
            )
            part_metadata = FileManager.read_json(part_json, {}) or {}
            cached_part = FileManager.read_csv(part_csv)
            cached_by_token = {
                str(row.get("image_token", "")): row for row in cached_part
            }
            required = {
                "attention_presence_probability",
                "attention_uncertainty_score",
                "attention_sequence_inconsistency",
                "attention_centroid_x",
                "attention_centroid_y",
                "attention_heart_present",
            }
            part_complete = (
                len(cached_part) == len(fold_rows)
                and len(cached_by_token) == len(fold_rows)
                and (not cached_part or required.issubset(cached_part[0].keys()))
                and all(
                    token in cached_by_token
                    and Path(
                        cached_by_token[token].get(
                            "predicted_attention_mask_path", ""
                        )
                    ).is_file()
                    for token in (
                        str(row["image_token"]) for row in fold_rows
                    )
                )
            )
            if (
                not force
                and part_metadata.get("fingerprint") == part_fingerprint
                and part_complete
            ):
                for global_index, row in zip(fold_indices, fold_rows):
                    results[global_index] = cached_by_token[
                        str(row["image_token"])
                    ]
                reused_folds.append(fold)
                print(
                    f"[ATTENTION] Fold {fold}: rezultat per-fold 2.5D reused."
                )
                continue

            checkpoint = SegmentationManager._torch_load(checkpoint_map[fold])
            model = AttentionUNet(
                base_channels=int(
                    checkpoint.get(
                        "base_channels", Settings.Segmentation.BASE_CHANNELS
                    )
                ),
                input_channels=int(
                    checkpoint.get(
                        "input_channels", Settings.Segmentation.INPUT_CHANNELS
                    )
                ),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model = RuntimeManager.prepare_model(model, device).eval()
            calibration = checkpoint.get("calibration", {}) or {}
            threshold = float(
                calibration.get(
                    "threshold", Settings.Segmentation.DEFAULT_THRESHOLD
                )
            )
            presence_threshold = float(
                calibration.get(
                    "presence_threshold",
                    Settings.Segmentation.DEFAULT_PRESENCE_THRESHOLD,
                )
            )
            geometry_prior = checkpoint.get("geometry_prior", {}) or {}
            checkpoint_fingerprint = str(checkpoint.get("fingerprint", ""))
            loader = DataLoader(
                SegmentationInferenceDataset(fold_rows),
                batch_size=RuntimeManager.inference_batch_size(device),
                shuffle=False,
                num_workers=Settings.Runtime.NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )
            print(
                f"[ATTENTION] 2.5D prediction for fold {fold}: {len(fold_rows)} "
                f"images, device={device.type}"
            )
            fold_results: list[dict[str, Any]] = []

            with torch.inference_mode():
                for images, local_indices in tqdm(
                    loader, desc=f"Attention 2.5D fold {fold}"
                ):
                    images = RuntimeManager.move_tensor(images, device)
                    with RuntimeManager.autocast(device):
                        logits, presence_logits = model(images)
                    original_probability = torch.sigmoid(logits)[:, 0]
                    original_presence = torch.sigmoid(presence_logits)

                    factor = float(Settings.Segmentation.TTA_CONTRAST_FACTOR)
                    tta_images = torch.clamp(
                        (images - 0.5) * factor + 0.5, 0.0, 1.0
                    )
                    with RuntimeManager.autocast(device):
                        tta_logits, tta_presence_logits = model(tta_images)
                    tta_probability = torch.sigmoid(tta_logits)[:, 0]
                    tta_presence = torch.sigmoid(tta_presence_logits)

                    averaged_probability = (
                        original_probability + tta_probability
                    ) / 2.0
                    averaged_presence = (
                        original_presence + tta_presence
                    ) / 2.0
                    probabilities = averaged_probability.float().cpu().numpy()
                    alternate_probabilities = tta_probability.float().cpu().numpy()
                    presence_probabilities = (
                        averaged_presence.float().cpu().numpy()
                    )

                    for batch_position, local_index_tensor in enumerate(
                        local_indices
                    ):
                        local_index = int(local_index_tensor)
                        global_index = fold_indices[local_index]
                        row = rows[global_index]
                        probability = probabilities[batch_position]
                        alternate_probability = alternate_probabilities[
                            batch_position
                        ]
                        presence_probability = float(
                            presence_probabilities[batch_position]
                        )
                        heart_present = bool(
                            presence_probability >= presence_threshold
                            or float(probability.max())
                            >= Settings.Segmentation.PRESENCE_SEGMENTATION_OVERRIDE_PEAK
                        )
                        initial_mask = SegmentationManager._candidate_mask(
                            probability, threshold
                        )
                        initial_valid, initial_reason = (
                            SegmentationManager._validate_mask(
                                initial_mask, probability, geometry_prior
                            )
                        )

                        if not heart_present:
                            final_mask = np.zeros_like(initial_mask, dtype=np.uint8)
                            final_valid = True
                            repair_method = "presence_head_empty"
                            used_threshold = threshold
                            final_reason = ""
                            prior_deviation = 0.0
                        elif initial_valid:
                            final_mask = initial_mask
                            final_valid = True
                            repair_method = "none"
                            used_threshold = threshold
                            final_reason = ""
                            prior_deviation = SegmentationManager._prior_deviation(
                                final_mask, geometry_prior
                            )
                        else:
                            (
                                final_mask,
                                final_valid,
                                repair_method,
                                used_threshold,
                                final_reason,
                                prior_deviation,
                            ) = SegmentationManager._repair_probability(
                                probability, threshold, geometry_prior
                            )
                            repair_method = "photometric_tta+" + repair_method

                        geometry = SegmentationManager._mask_features(final_mask)
                        area = float(geometry["area"])
                        boundary = float(geometry["boundary"])
                        centroid_x = geometry["centroid_x"]
                        centroid_y = geometry["centroid_y"]
                        disagreement, entropy, uncertainty = (
                            SegmentationManager._uncertainty_metrics(
                                probability,
                                alternate_probability,
                                presence_probability,
                                presence_threshold,
                                prior_deviation,
                                final_valid,
                            )
                        )
                        mask_path = Path(row["predicted_attention_mask_path"])
                        if (
                            Settings.Segmentation.SAVE_ALL_PREDICTED_MASKS
                            or not final_valid
                        ):
                            FileManager.write_png(
                                mask_path,
                                final_mask.astype(np.uint8) * 255,
                            )

                        result_row = {
                            **row,
                            "attention_valid_initial": int(initial_valid),
                            "attention_valid_final": int(final_valid),
                            "attention_invalid_reason_initial": initial_reason,
                            "attention_invalid_reason_final": final_reason,
                            "attention_repair_method": repair_method,
                            "attention_threshold_used": float(used_threshold),
                            "attention_area_ratio": area,
                            "attention_peak_probability": float(
                                probability.max()
                            ),
                            "attention_boundary_touch_fraction": boundary,
                            "attention_heart_present": int(heart_present),
                            "attention_presence_probability": presence_probability,
                            "attention_presence_threshold": presence_threshold,
                            "attention_tta_disagreement": disagreement,
                            "attention_mean_entropy": entropy,
                            "attention_uncertainty_score": uncertainty,
                            "attention_sequence_inconsistency": 0.0,
                            "attention_centroid_x": centroid_x,
                            "attention_centroid_y": centroid_y,
                            "attention_prior_deviation": float(prior_deviation),
                            "checkpoint_fingerprint": checkpoint_fingerprint,
                        }
                        results[global_index] = result_row
                        fold_results.append(result_row)

            SegmentationManager._apply_sequence_consistency(
                fold_results, fold_rows
            )
            FileManager.write_csv(
                part_csv, fold_results, SegmentationManager.PREDICTION_FIELDS
            )
            FileManager.write_json(
                part_json,
                {
                    "fingerprint": part_fingerprint,
                    "fold": fold,
                    "images": len(fold_results),
                    "checkpoint": str(checkpoint_map[fold]),
                    "device_used": device.type,
                    "completed": True,
                },
            )
            computed_folds.append(fold)
            del model, loader, checkpoint, fold_results
            RuntimeManager.release(device)

        final_results = [result for result in results if result is not None]
        if len(final_results) != len(rows):
            raise RuntimeError(
                "Prediction did not produce exactly one row per image."
            )
        SegmentationManager._apply_sequence_consistency(final_results, rows)
        FileManager.write_csv(
            workspace.prediction_audit,
            final_results,
            SegmentationManager.PREDICTION_FIELDS,
        )
        invalid_rows = [
            row
            for row in final_results
            if _as_int(row.get("attention_valid_final"), 0) != 1
        ]
        FileManager.write_csv(
            workspace.invalid_predictions,
            invalid_rows,
            SegmentationManager.PREDICTION_FIELDS,
        )
        segmentation_summary = SegmentationManager.evaluate_oof_segmentation(
            workspace, final_results
        )
        summary = {
            "fingerprint": prediction_fingerprint,
            "schema": "simple-predictions-2p5d-presence-v5",
            "device_used_for_computed_folds": device.type,
            "reused_folds": reused_folds,
            "computed_folds": computed_folds,
            "images": len(final_results),
            "valid_initial": sum(
                _as_int(row.get("attention_valid_initial"), 0)
                for row in final_results
            ),
            "valid_final": sum(
                _as_int(row.get("attention_valid_final"), 0)
                for row in final_results
            ),
            "predicted_heart_present": sum(
                _as_int(row.get("attention_heart_present"), 0)
                for row in final_results
            ),
            "predicted_no_heart": sum(
                _as_int(row.get("attention_heart_present"), 0) == 0
                for row in final_results
            ),
            "uncertain_above_review_threshold": sum(
                _as_float(row.get("attention_uncertainty_score"), 0.0)
                >= Settings.Review.UNCERTAINTY_MIN_SCORE
                for row in final_results
            ),
            "invalid_final": len(invalid_rows),
            "prediction_audit": str(workspace.prediction_audit),
            "invalid_queue": str(workspace.invalid_predictions),
            "segmentation_oof": segmentation_summary,
            "elapsed": RuntimeManager.format_seconds(
                time.perf_counter() - started
            ),
        }
        FileManager.write_json(workspace.prediction_summary, summary)
        print(
            "[ATTENTION] "
            f"valid_final={summary['valid_final']}/{summary['images']}, "
            f"heart_present={summary['predicted_heart_present']}, "
            f"invalid={summary['invalid_final']} | {workspace.prediction_audit}"
        )
        ImageCache.clear()
        return final_results


# -----------------------------------------------------------------------------
# Manual review and annotation editor
# -----------------------------------------------------------------------------
class HammingBKTree:


    def __init__(self):
        self.root: tuple[int, dict[int, Any]] | None = None

    @staticmethod
    def distance(first: int, second: int) -> int:
        return int(first ^ second).bit_count()

    def add(self, value: int) -> None:
        value = int(value)
        if self.root is None:
            self.root = (value, {})
            return
        node_value, children = self.root
        while True:
            distance = self.distance(value, node_value)
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node_value, children = child

    def has_near(self, value: int, maximum_distance: int) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node_value, children = stack.pop()
            distance = self.distance(value, node_value)
            if distance <= maximum_distance:
                return True
            low = distance - maximum_distance
            high = distance + maximum_distance
            stack.extend(child for edge, child in children.items() if low <= edge <= high)
        return False


class ReviewManager:
    """Build balanced review queues containing only unresolved labels.

    Selection is distributed across patients and series and can prioritize invalid,
    uncertain, empty, novel, manual, or all currently UNLABELED images.
    """


    HISTORY_FIELDS = (
        "timestamp_utc",
        "review_round",
        "action",
        "image_token",
        "patient_id",
        "series_id",
        "queue_position",
        "queue_size",
        "mask_area_ratio",
        "manual_mask_path",
    )

    @staticmethod
    def _merge_rows(
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
    ) -> list[dict[str, Any]]:
        quality = {
            row["image_token"]: row
            for row in FileManager.read_csv(workspace.quality_audit)
        }
        predictions = {
            row["image_token"]: row
            for row in FileManager.read_csv(workspace.prediction_audit)
        }
        annotations = MaskManager.annotation_map(workspace)
        merged = []
        for original in dataset_rows:
            row = dict(original)
            row.update(quality.get(row["image_token"], {}))
            row.update(predictions.get(row["image_token"], {}))
            annotation = annotations.get(str(row["image_token"]), {})
            annotation_type = ReviewManager._normalize_target_type(
                annotation.get("target_type", "")
            )
            row["manual_annotation_type"] = annotation_type
            row["review_target_type"] = annotation_type or "UNLABELED"
            row["manual_annotation_source"] = annotation.get("source", "")
            mask_qc = MaskManager.manual_qc(row["manual_mask_path"])
            row.update(
                {
                    "manual_mask_exists": mask_qc["exists"],
                    "manual_mask_usable": mask_qc["usable"],
                    "manual_mask_area_ratio": mask_qc["area_ratio"],
                    "manual_mask_reason": mask_qc["reason"],
                }
            )
            merged.append(row)
        return merged

    @staticmethod
    def _normalize_target_type(value: Any) -> str:


        target = str(value or "").strip().upper()
        if target in {"", "UNLABELED", "NONE", "NAN"}:
            return ""
        return target

    @staticmethod
    def _row_target_type(row: dict[str, Any]) -> str:
        return ReviewManager._normalize_target_type(
            row.get("manual_annotation_type", row.get("target_type", ""))
        )

    @staticmethod
    def _is_unlabeled(row: dict[str, Any]) -> bool:
        return ReviewManager._row_target_type(row) == ""

    @staticmethod
    def _reviewed_tokens(workspace: Workspace, review_round: int) -> set[str]:
        return {
            row.get("image_token", "")
            for row in FileManager.read_csv(workspace.review_history)
            if _as_int(row.get("review_round"), 0) == int(review_round)
        }

    @staticmethod
    def _round_robin(
        rows: Sequence[dict[str, Any]],
        limit: int,
        per_patient: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        def priority(row: dict[str, Any]):
            explicit_priority = row.get("review_priority", "")
            if str(explicit_priority).strip() != "":
                primary = _as_float(explicit_priority, 0.0)
            else:
                uncertainty = _as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
                valid = _as_int(row.get("attention_valid_final"), 0)
                primary = -uncertainty if uncertainty > 0 else float(valid)
            peak = _as_float(row.get("attention_peak_probability"), 0.0)
            tie = hashlib.sha256(
                f"{seed}|{row['image_token']}".encode("utf-8")
            ).hexdigest()
            return (primary, peak, tie)

        by_patient_series: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            sequence_group = str(
                row.get("sequence_group_id") or row["series_id"]
            )
            by_patient_series[str(row["patient_id"])][sequence_group].append(row)

        patient_queues: dict[str, list[dict[str, Any]]] = {}
        for patient_id, series_map in by_patient_series.items():
            series_ids = sorted(series_map)
            for series_id in series_ids:
                series_map[series_id].sort(key=priority)
            queue: list[dict[str, Any]] = []
            position = 0
            while len(queue) < per_patient:
                added = False
                for series_id in series_ids:
                    if position < len(series_map[series_id]):
                        queue.append(series_map[series_id][position])
                        added = True
                        if len(queue) >= per_patient:
                            break
                if not added:
                    break
                position += 1
            patient_queues[patient_id] = queue

        selected = []
        position = 0
        while len(selected) < limit:
            added = False
            for patient_id in sorted(patient_queues):
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

    # The UNLABELED filter is always applied before the review scope. This prevents
    # a prior HEART_PRESENT, NO_HEART_VISIBLE, or UNUSABLE decision from silently
    # reappearing in a later annotation round.
    @staticmethod
    def select(
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        scope: str = "invalid",
        limit: int | None = None,
        seed: int = 42,
        review_round: int = 1,
    ) -> list[dict[str, Any]]:


        scope = str(scope).lower()
        allowed = {"invalid", "uncertain", "empty", "novel", "manual", "all"}
        if scope not in allowed:
            raise ValueError(
                "scope must be invalid/uncertain/empty/novel/manual/all."
            )
        limit = int(Settings.Review.DEFAULT_LIMIT if limit is None else limit)
        all_rows = ReviewManager._merge_rows(dataset_rows, workspace)
        target_counts: dict[str, int] = defaultdict(int)
        for row in all_rows:
            target = ReviewManager._row_target_type(row) or "UNLABELED"
            target_counts[target] += 1
        rows = [row for row in all_rows if ReviewManager._is_unlabeled(row)]
        labeled_excluded = len(all_rows) - len(rows)
        print(
            "[REVIEW][TARGET FILTER] "
            f"UNLABELED={len(rows)}, already-labeled excluded={labeled_excluded}, "
            f"distribution={dict(sorted(target_counts.items()))}"
        )
        reviewed = ReviewManager._reviewed_tokens(workspace, review_round)

        if scope == "invalid":
            candidates = [
                row
                for row in rows
                if _as_int(row.get("attention_valid_final"), 1) == 0
                and Path(row["predicted_attention_mask_path"]).is_file()
                and row["image_token"] not in reviewed
            ]
            for row in candidates:
                row["review_priority"] = _as_float(
                    row.get("attention_peak_probability"), 0.0
                )
            candidates = ReviewManager._round_robin(
                candidates,
                limit=max(1, limit),
                per_patient=max(
                    Settings.Review.NEW_IMAGES_PER_PATIENT,
                    int(math.ceil(limit / max(1, len({r['patient_id'] for r in candidates}))))
                    if candidates
                    else Settings.Review.NEW_IMAGES_PER_PATIENT,
                ),
                seed=seed,
            )
        elif scope == "uncertain":
            candidates = []
            for row in rows:
                if row["image_token"] in reviewed:
                    continue
                if _as_int(row.get("quality_valid"), 0) != 1:
                    continue
                if not Path(row["predicted_attention_mask_path"]).is_file():
                    continue
                uncertainty = _as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
                presence = _as_float(
                    row.get("attention_presence_probability"), np.nan
                )
                presence_threshold = _as_float(
                    row.get("attention_presence_threshold"),
                    Settings.Segmentation.DEFAULT_PRESENCE_THRESHOLD,
                )
                near_presence_boundary = bool(
                    np.isfinite(presence)
                    and abs(presence - presence_threshold)
                    <= Settings.Review.PRESENCE_MARGIN
                )
                if (
                    uncertainty < Settings.Review.UNCERTAINTY_MIN_SCORE
                    and not near_presence_boundary
                    and _as_int(row.get("attention_valid_final"), 1) == 1
                ):
                    continue

                row["review_priority"] = -uncertainty
                candidates.append(row)
            candidates = ReviewManager._round_robin(
                candidates,
                limit=max(1, limit),
                per_patient=max(
                    Settings.Review.NEW_IMAGES_PER_PATIENT,
                    int(math.ceil(limit / max(1, len({r['patient_id'] for r in candidates}))))
                    if candidates
                    else Settings.Review.NEW_IMAGES_PER_PATIENT,
                ),
                seed=seed,
            )
        elif scope == "empty":
            candidates = [
                row
                for row in rows
                if Path(row["manual_mask_path"]).is_file()
                and not bool(row.get("manual_annotation_type"))
                and str(row.get("manual_mask_reason", ""))
                == "empty_or_nearly_empty"
                and Path(row["predicted_attention_mask_path"]).is_file()
                and row["image_token"] not in reviewed
            ]
            for row in candidates:
                row["review_priority"] = -_as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
            candidates = ReviewManager._round_robin(
                candidates,
                limit=max(1, limit),
                per_patient=max(
                    Settings.Review.NEW_IMAGES_PER_PATIENT,
                    int(math.ceil(limit / max(1, len({r['patient_id'] for r in candidates}))))
                    if candidates
                    else Settings.Review.NEW_IMAGES_PER_PATIENT,
                ),
                seed=seed,
            )
        elif scope == "manual":


            candidates = [
                row
                for row in rows
                if Path(row["manual_mask_path"]).is_file()
                and Path(row["predicted_attention_mask_path"]).is_file()
                and row["image_token"] not in reviewed
            ]
            candidates.sort(
                key=lambda row: (
                    row["patient_id"],
                    row["series_id"],
                    row["image_token"],
                )
            )
        elif scope == "all":
            candidates = [
                row
                for row in rows
                if Path(row["predicted_attention_mask_path"]).is_file()
                and row["image_token"] not in reviewed
            ]
            candidates.sort(
                key=lambda row: (
                    row["patient_id"],
                    row["series_id"],
                    row["image_token"],
                )
            )
        else:


            manual_hashes = []
            for row in all_rows:
                is_annotated = bool(row.get("manual_annotation_type")) or Path(
                    row["manual_mask_path"]
                ).is_file()
                if is_annotated and row.get("perceptual_hash"):
                    manual_hashes.append(int(str(row["perceptual_hash"]), 16))
            if not manual_hashes:
                raise RuntimeError(
                    "The novel scope requires at least one labeled image."
                )
            tree = HammingBKTree()
            for value in sorted(set(manual_hashes)):
                tree.add(value)

            candidates = []
            excluded = defaultdict(int)
            for row in rows:
                if bool(row.get("manual_annotation_type")) or Path(
                    row["manual_mask_path"]
                ).is_file():
                    excluded["already_annotated"] += 1
                    continue
                if row["image_token"] in reviewed:
                    excluded["already_reviewed_this_round"] += 1
                    continue
                if _as_int(row.get("quality_valid"), 0) != 1:
                    excluded["quality_invalid"] += 1
                    continue
                if not Path(row["predicted_attention_mask_path"]).is_file():
                    excluded["missing_attention_mask"] += 1
                    continue
                phash = str(row.get("perceptual_hash", ""))
                if not phash:
                    excluded["missing_phash"] += 1
                    continue
                if tree.has_near(
                    int(phash, 16), Settings.Review.PHASH_MAX_DISTANCE
                ):
                    excluded["similar_to_annotated"] += 1
                    continue
                row["review_priority"] = -_as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
                candidates.append(row)
            candidates = ReviewManager._round_robin(
                candidates,
                limit=max(1, limit),
                per_patient=Settings.Review.NEW_IMAGES_PER_PATIENT,
                seed=seed,
            )
            print(f"[REVIEW novel] excluded={dict(excluded)}")

        if limit > 0 and scope in {"manual", "all"}:
            candidates = candidates[:limit]
        if not candidates:
            raise RuntimeError(
                "No target=UNLABELED images are eligible for "
                f"scope={scope!r}. HEART_PRESENT, "
                "NO_HEART_VISIBLE, and UNUSABLE images are excluded automatically."
            )
        queue_path = (
            workspace.outputs / f"review_queue_{scope}_round_{review_round}.csv"
        )
        fields = sorted({key for row in candidates for key in row.keys()})
        FileManager.write_csv(queue_path, candidates, fields)
        print(
            f"[REVIEW] scope={scope}, images={len(candidates)}, queue={queue_path}"
        )
        return candidates

    @staticmethod
    def log(
        workspace: Workspace,
        row: dict[str, Any],
        review_round: int,
        action: str,
        mask: np.ndarray | None,
        position: int,
        queue_size: int,
    ) -> None:
        history = FileManager.read_csv(workspace.review_history)
        history.append(
            {
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                "review_round": int(review_round),
                "action": action,
                "image_token": row["image_token"],
                "patient_id": row["patient_id"],
                "series_id": row["series_id"],
                "queue_position": int(position),
                "queue_size": int(queue_size),
                "mask_area_ratio": ""
                if mask is None
                else float((mask > 0).mean()),
                "manual_mask_path": row["manual_mask_path"],
            }
        )
        FileManager.write_csv(
            workspace.review_history, history, ReviewManager.HISTORY_FIELDS
        )


class MaskEditor:
    """Kaggle-safe HTML5 editor for correcting segmentation targets.

    Keyboard listeners are installed once and isolated from notebook command mode.
    Labeled images remain navigable during the current session but are excluded
    when a later review queue is constructed.
    """


    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        start_index: int = 0,
        brush_radius: int = 8,
        review_round: int = 1,
        review_scope: str = "invalid",
    ):
        import ipywidgets as widgets

        annotations = MaskManager.annotation_map(workspace)
        unlabeled_rows: list[dict[str, Any]] = []
        for source_row in rows:
            row = dict(source_row)
            annotation = annotations.get(str(row.get("image_token", "")), {})
            target = ReviewManager._normalize_target_type(
                annotation.get("target_type", row.get("manual_annotation_type", ""))
            )
            if target:
                continue
            row["manual_annotation_type"] = ""
            row["review_target_type"] = "UNLABELED"
            unlabeled_rows.append(row)
        if not unlabeled_rows:
            raise ValueError(
                "The editor has no remaining target=UNLABELED images. "
                "Previously decided targets are not reopened."
            )

        self.rows = unlabeled_rows
        self.workspace = workspace
        self.index = int(np.clip(start_index, 0, len(self.rows) - 1))
        self.brush_radius = max(1, int(brush_radius))
        self.review_round = int(review_round)
        self.review_scope = str(review_scope)
        self.image: np.ndarray | None = None
        self.auto_mask: np.ndarray | None = None
        self.base_mask: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.widget_id = uuid.uuid4().hex[:12]
        self._shortcut_busy = False
        self._last_shortcut_payload = ""
        self._last_shortcut_at = 0.0

        self.output = widgets.Output()
        self.mask_sync = widgets.Textarea(
            value="",
            placeholder=f"CAD_MASK_SYNC_{self.widget_id}",
            layout=widgets.Layout(width="1px", height="1px", display="none"),
        )
        self.shortcut_sync = widgets.Textarea(
            value="",
            placeholder=f"CAD_SHORTCUT_SYNC_{self.widget_id}",
            layout=widgets.Layout(width="1px", height="1px", display="none"),
        )

        def compact_button(
            description: str,
            width: str,
            tooltip: str,
            button_style: str = "",
        ):
            return widgets.Button(
                description=description,
                tooltip=tooltip,
                button_style=button_style,
                layout=widgets.Layout(width=width, height="29px"),
                style={"font_weight": "500"},
            )


        self.previous_button = compact_button("← Prev [P]", "68px", "Previous — P")
        self.next_button = compact_button("Next [N] →", "72px", "Next — N")
        self.save_next_button = compact_button(
            "Save + Next [S]", "102px", "Save HEART_PRESENT mask and go next — S", "success"
        )
        self.raw_button = compact_button(
            "Image [I]", "74px", "Hide overlays and start from the image — I"
        )
        self.reset_button = compact_button(
            "Auto reset [O]", "94px", "Restore the automatic mask — O"
        )
        self.accept_button = compact_button(
            "Auto OK [A]", "84px", "Save confirmed auto mask for training — A", "info"
        )
        self.no_heart_button = compact_button(
            "No heart [H]", "92px", "Valid image, but no heart is visible — H", "info"
        )
        self.unusable_button = compact_button(
            "Unusable [U]", "92px", "Blur/noise/localizer: exclude from training — U", "warning"
        )
        self.skip_button = compact_button("Skip [K]", "66px", "Skip and go next — K")
        self.clear_button = compact_button("Clear [C]", "68px", "Clear editable mask — C")
        self.delete_button = compact_button(
            "Delete [D]", "76px", "Delete manual target and label — D", "danger"
        )
        self.brush_slider = widgets.IntSlider(
            description="Brush",
            value=self.brush_radius,
            min=1,
            max=30,
            step=1,
            continuous_update=False,
            layout=widgets.Layout(width="260px"),
        )
        self.status = widgets.HTML()

        self.previous_button.on_click(lambda _: self._safe("Previous", self.previous))
        self.next_button.on_click(lambda _: self._safe("Next", self.next))
        self.save_next_button.on_click(lambda _: self._safe("Save & Next", self.save_next))
        self.accept_button.on_click(lambda _: self._safe("Auto OK", self.accept_auto))
        self.no_heart_button.on_click(lambda _: self._safe("No heart", self.mark_no_heart))
        self.unusable_button.on_click(lambda _: self._safe("Unusable", self.mark_unusable))
        self.skip_button.on_click(lambda _: self._safe("Skip", self.skip))
        self.reset_button.on_click(lambda _: self._safe("Reset", self.reset_to_auto))
        self.raw_button.on_click(lambda _: self._safe("Reset to image", self.reset_to_image))
        self.clear_button.on_click(lambda _: self._safe("Clear", self.clear_editable))
        self.delete_button.on_click(lambda _: self._safe("Delete manual", self.delete_manual))
        self.brush_slider.observe(self._brush_changed, names="value")
        self.mask_sync.observe(self._mask_sync_changed, names="value")
        self.shortcut_sync.observe(self._shortcut_sync_changed, names="value")

        self.controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.previous_button,
                        self.next_button,
                        self.save_next_button,
                        self.raw_button,
                        self.reset_button,
                        self.accept_button,
                        self.no_heart_button,
                        self.unusable_button,
                        self.skip_button,
                        self.clear_button,
                        self.delete_button,
                    ],
                    layout=widgets.Layout(
                        flex_flow="row nowrap",
                        align_items="center",
                        overflow="auto hidden",
                        width="100%",
                    ),
                ),
                widgets.HBox([self.brush_slider, self.status]),
                self.mask_sync,
                self.shortcut_sync,
                self.output,
            ]
        )
        self.load_current()

    def _safe(self, action: str, function) -> None:
        try:
            function()
        except Exception as error:
            message = f"{action}: {type(error).__name__}: {error}"
            self.status.value = f"<span style='color:#b00020'><b>{message}</b></span>"
            print("[EDITOR ERROR]", message)

    def _set_controls_disabled(self, disabled: bool) -> None:
        for control in (
            self.previous_button,
            self.next_button,
            self.save_next_button,
            self.raw_button,
            self.reset_button,
            self.accept_button,
            self.no_heart_button,
            self.unusable_button,
            self.skip_button,
            self.clear_button,
            self.delete_button,
            self.brush_slider,
        ):
            control.disabled = bool(disabled)

    def _advance_after_target(self, message: str) -> None:


        if not self.rows:
            return
        completed_index = int(self.index)
        completed_token = str(self.rows[completed_index].get("image_token", ""))
        if completed_index < len(self.rows) - 1:
            self.index = completed_index + 1
            self.load_current()
            self.update_status(
                f"{message} Image {completed_token} remains in the session queue; "
                "use Prev [P] to return to it."
            )
            return


        self.render()
        self.update_status(
            f"{message} End of queue; the image remains accessible, and "
            "Prev [P] returns to earlier images."
        )

    def _brush_changed(self, change: dict[str, Any]) -> None:
        self.brush_radius = int(change["new"])
        if self.image is not None:
            self.render()

    @staticmethod
    def _image_uri(rgb: np.ndarray) -> str:
        array = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("The editor image could not be encoded.")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _mask_uri(mask: np.ndarray) -> str:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        bgra = np.zeros((*binary.shape, 4), dtype=np.uint8)
        foreground = binary * 255
        bgra[..., :3] = foreground[..., None]
        bgra[..., 3] = foreground
        ok, encoded = cv2.imencode(".png", bgra)
        if not ok:
            raise RuntimeError("The editor mask could not be encoded.")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    def load_current(self) -> None:
        row = self.rows[self.index]
        auto_path = Path(row["predicted_attention_mask_path"])
        if not auto_path.is_file():
            raise FileNotFoundError(f"Missing Attention mask: {auto_path}")
        image = ImageProcessor.standardized_uint8(row["image_path"])
        auto_mask = MaskManager.read_binary(auto_path)
        manual_path = Path(row["manual_mask_path"])
        current = MaskManager.read_binary(manual_path) if manual_path.is_file() else auto_mask.copy()
        self.image = image.astype(np.float32) / 255.0
        self.auto_mask = auto_mask.astype(np.uint8)
        self.base_mask = self.auto_mask.copy()
        self.mask = current.astype(np.uint8)
        self.render()
        self.update_status()

    def render(self) -> None:
        row = self.rows[self.index]
        target_type = (
            ReviewManager._normalize_target_type(
                row.get("manual_annotation_type", row.get("target_type", ""))
            )
            or "UNLABELED"
        )
        image_uri = self._image_uri(np.stack([self.image] * 3, axis=-1))
        base_uri = self._mask_uri(self.base_mask)
        mask_uri = self._mask_uri(self.mask)
        canvas_id = f"cad_canvas_{self.widget_id}"
        keyboard_sink_id = f"cad_keyboard_sink_{self.widget_id}"
        placeholder = f"CAD_MASK_SYNC_{self.widget_id}"
        size = Settings.Image.SEGMENTATION_SIZE
        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:900px">
          <div style="margin-bottom:6px;font-size:14px">
            <b>{self.index + 1}/{len(self.rows)}</b> | {row['patient_id']} | {row['series_id']} |
            scope={self.review_scope} | target={target_type}
          </div>
          <input id="{keyboard_sink_id}" type="text" value="" tabindex="-1"
                 autocomplete="off" autocapitalize="off" spellcheck="false"
                 aria-label="CAD mask editor keyboard focus"
                 style="position:fixed;left:-10000px;top:0;width:1px;height:1px;
                        opacity:0;pointer-events:none;border:0;padding:0;margin:0" />
          <canvas id="{canvas_id}" width="{size * 3}" height="{size * 3}"
                  oncontextmenu="return false;"
                  style="width:768px;height:768px;max-width:100%;border:1px solid #999;
                         cursor:crosshair;touch-action:none;user-select:none;
                         -webkit-user-select:none;-webkit-touch-callout:none"></canvas>
          <div style="font-size:12px;margin-top:5px;line-height:1.4">
            Left drag = draw | Right drag = erase (Kaggle menu disabled) | S = Save & Next | P/N = Prev/Next |
            I = image | O = auto reset | A = Auto OK | H = no heart | U = unusable |
            K = Skip | C = Clear | D = Delete | [ / ] = Brush −/+
          </div>
        </div>
        """
        js = f"""
        (() => {{
          const canvas = document.getElementById({json.dumps(canvas_id)});
          const keyboardSink = document.getElementById({json.dumps(keyboard_sink_id)});
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          const W = {size}, H = {size};
          const radius = {int(self.brush_radius)};
          const placeholder = {json.dumps(placeholder)};
          const image = new Image(), base = new Image(), initialMask = new Image();
          const maskCanvas = document.createElement('canvas');
          maskCanvas.width = W; maskCanvas.height = H;
          const maskCtx = maskCanvas.getContext('2d', {{willReadFrequently:true}});
          const pointerControllerKey = '__cadMaskEditorPointerController';
          const oldPointerController = window[pointerControllerKey];
          if (oldPointerController && typeof oldPointerController.dispose === 'function') {{
            try {{ oldPointerController.dispose(); }} catch (_) {{}}
          }}

          let drawing = false, erase = false, last = null, rightDragActive = false;

          function focusKeyboardSink() {{
            if (!keyboardSink) return;
            try {{
              keyboardSink.focus({{preventScroll: true}});
            }} catch (_) {{
              try {{ keyboardSink.focus(); }} catch (_) {{}}
            }}
            keyboardSink.value = '';
          }}

          function eventTargetsCanvas(event) {{
            if (event.target === canvas) return true;
            try {{
              const path = typeof event.composedPath === 'function'
                ? event.composedPath()
                : [];
              return path.includes(canvas);
            }} catch (_) {{
              return false;
            }}
          }}
          function blockEvent(event) {{
            if (!event) return false;
            if (typeof event.preventDefault === 'function') event.preventDefault();
            if (typeof event.stopPropagation === 'function') event.stopPropagation();
            if (typeof event.stopImmediatePropagation === 'function') {{
              event.stopImmediatePropagation();
            }}
            return false;
          }}
          function suppressContextMenu(event) {{
            // The window/capture listener runs before Kaggle's context menu.
            // Page behavior outside the canvas is unchanged.
            if (!rightDragActive && !eventTargetsCanvas(event)) return;
            return blockEvent(event);
          }}
          function releasePointerState() {{
            drawing = false;
            erase = false;
            rightDragActive = false;
            last = null;
          }}

          function hidden() {{
            return Array.from(document.querySelectorAll('textarea'))
              .find(x => x.placeholder === placeholder);
          }}
          function overlay(source, color, alpha) {{
            const tmp = document.createElement('canvas');
            tmp.width = canvas.width; tmp.height = canvas.height;
            const c = tmp.getContext('2d');
            c.globalAlpha = alpha;
            c.drawImage(source, 0, 0, canvas.width, canvas.height);
            c.globalCompositeOperation = 'source-in';
            c.fillStyle = color;
            c.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(tmp, 0, 0);
          }}
          function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            overlay(base, 'rgb(255,165,0)', 0.35);
            overlay(maskCanvas, 'rgb(255,0,200)', 0.50);
          }}
          function sync() {{
            const target = hidden();
            if (!target) return;
            target.value = maskCanvas.toDataURL('image/png').split(',')[1];
            target.dispatchEvent(new Event('input', {{bubbles:true}}));
            target.dispatchEvent(new Event('change', {{bubbles:true}}));
          }}
          function point(e) {{
            const r = canvas.getBoundingClientRect();
            return {{
              x: Math.max(0, Math.min(W - 1, (e.clientX - r.left) / r.width * W)),
              y: Math.max(0, Math.min(H - 1, (e.clientY - r.top) / r.height * H))
            }};
          }}
          function paint(p) {{
            maskCtx.save();
            maskCtx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
            maskCtx.fillStyle = 'white'; maskCtx.strokeStyle = 'white';
            maskCtx.lineWidth = radius * 2; maskCtx.lineCap = 'round'; maskCtx.lineJoin = 'round';
            if (last) {{
              maskCtx.beginPath(); maskCtx.moveTo(last.x, last.y); maskCtx.lineTo(p.x, p.y); maskCtx.stroke();
            }} else {{
              maskCtx.beginPath(); maskCtx.arc(p.x, p.y, radius, 0, 2 * Math.PI); maskCtx.fill();
            }}
            maskCtx.restore(); last = p; draw();
          }}
          function begin(e) {{
            if (![0,2].includes(e.button)) return;
            focusKeyboardSink();
            const startsErase = e.button === 2;
            if (startsErase) blockEvent(e);
            else e.preventDefault();
            drawing = true;
            erase = startsErase;
            rightDragActive = startsErase;
            last = null;
            try {{ canvas.setPointerCapture?.(e.pointerId); }} catch (_) {{}}
            paint(point(e));
          }}
          function move(e) {{
            if (!drawing) return;
            const requiredButton = erase ? 2 : 1;
            if ((Number(e.buttons || 0) & requiredButton) === 0) return finish(e);
            if (erase) blockEvent(e);
            else e.preventDefault();
            paint(point(e));
          }}
          function finish(e) {{
            if (!drawing) {{
              if (e && e.button === 2) rightDragActive = false;
              return;
            }}
            const wasErase = erase;
            if (wasErase || (e && e.button === 2)) blockEvent(e);
            else e?.preventDefault?.();
            try {{ canvas.releasePointerCapture?.(e.pointerId); }} catch (_) {{}}
            drawing = false;
            erase = false;
            rightDragActive = false;
            last = null;
            sync();
            focusKeyboardSink();
          }}

          const activeOptions = {{capture: true, passive: false}};
          // Local protection plus an early window listener blocks both the native
          // context menu and the menu installed by Kaggle/Jupyter.
          canvas.oncontextmenu = suppressContextMenu;
          canvas.addEventListener('contextmenu', suppressContextMenu, activeOptions);
          window.addEventListener('contextmenu', suppressContextMenu, activeOptions);
          canvas.addEventListener('auxclick', event => {{
            if (event.button === 2) blockEvent(event);
          }}, activeOptions);
          canvas.addEventListener('dragstart', blockEvent, activeOptions);
          canvas.addEventListener('selectstart', blockEvent, activeOptions);

          // Pointer Events and Mouse Events are never installed together, avoiding
          // duplicate handling of a single press in Chrome/Kaggle.
          if (window.PointerEvent) {{
            canvas.addEventListener('pointerdown', begin, activeOptions);
            canvas.addEventListener('pointermove', move, activeOptions);
            canvas.addEventListener('pointerup', finish, activeOptions);
            canvas.addEventListener('pointercancel', finish, activeOptions);
          }} else {{
            canvas.addEventListener('mousedown', begin, activeOptions);
            canvas.addEventListener('mousemove', move, activeOptions);
            canvas.addEventListener('mouseup', finish, activeOptions);
            canvas.addEventListener('mouseleave', event => {{
              if (drawing && Number(event.buttons || 0) === 0) finish(event);
            }}, activeOptions);
          }}

          window.addEventListener('blur', releasePointerState, true);
          document.addEventListener('visibilitychange', releasePointerState, true);
          window[pointerControllerKey] = {{
            editorId: {json.dumps(self.widget_id)},
            dispose: () => {{
              window.removeEventListener('contextmenu', suppressContextMenu, true);
              window.removeEventListener('blur', releasePointerState, true);
              document.removeEventListener('visibilitychange', releasePointerState, true);
              releasePointerState();
            }}
          }};
          function load(target, source) {{
            return new Promise((resolve, reject) => {{ target.onload = resolve; target.onerror = reject; target.src = source; }});
          }}
          // Focusing an invisible input places the notebook in typing mode before the
          // first shortcut and prevents Kaggle command mode from interpreting
          // O/P/N/etc. as notebook-cell operations.
          focusKeyboardSink();
          requestAnimationFrame(focusKeyboardSink);
          Promise.all([
            load(image, {json.dumps(image_uri)}),
            load(base, {json.dumps(base_uri)}),
            load(initialMask, {json.dumps(mask_uri)})
          ]).then(() => {{
            maskCtx.clearRect(0,0,W,H);
            maskCtx.drawImage(initialMask,0,0,W,H);
            draw();
            focusKeyboardSink();
            requestAnimationFrame(focusKeyboardSink);
          }});
        }})();
        """
        with self.output:
            clear_output(wait=True)
            display(HTML(html))
            display(Javascript(js))

    def _shortcut_script(self) -> str:


        placeholder = f"CAD_SHORTCUT_SYNC_{self.widget_id}"
        keyboard_sink_id = f"cad_keyboard_sink_{self.widget_id}"
        controller_key = "__cadMaskEditorKeyboardController"
        return f"""
        (() => {{
          const controllerKey = {json.dumps(controller_key)};
          const editorId = {json.dumps(self.widget_id)};
          const placeholder = {json.dumps(placeholder)};
          const keyboardSinkId = {json.dumps(keyboard_sink_id)};

          const oldController = window[controllerKey];
          if (oldController && typeof oldController.dispose === 'function') {{
            try {{ oldController.dispose(); }} catch (_) {{}}
          }}

          const held = new Set();
          const lastFire = new Map();
          let serial = 0;
          const minimumGapMs = 220;
          const keyOptions = {{capture: true, passive: false}};

          function commandTarget() {{
            return Array.from(document.querySelectorAll('textarea'))
              .find(node => node.placeholder === placeholder) || null;
          }}

          function keyboardSink() {{
            return document.getElementById(keyboardSinkId) || null;
          }}

          function focusKeyboardSink() {{
            const sink = keyboardSink();
            if (!sink) return;
            try {{
              sink.focus({{preventScroll: true}});
            }} catch (_) {{
              try {{ sink.focus(); }} catch (_) {{}}
            }}
            sink.value = '';
          }}

          function isEditorSink(target) {{
            return Boolean(target && String(target.id || '') === keyboardSinkId);
          }}

          function isTypingTarget(target) {{
            if (!target || isEditorSink(target)) return false;
            const tag = String(target.tagName || '').toLowerCase();
            return tag === 'input' || tag === 'textarea' || tag === 'select' ||
                   target.isContentEditable === true;
          }}

          function normalizeCode(event) {{
            const code = String(event.code || '');
            if (code) return code;
            const key = String(event.key || '').toLowerCase();
            const fallback = {{
              p: 'KeyP', n: 'KeyN', s: 'KeyS', i: 'KeyI', o: 'KeyO',
              // R no longer resets the mask; it is retained only to block the legacy
              // Kaggle shortcut when pressed out of habit.
              r: 'KeyR',
              a: 'KeyA', h: 'KeyH', u: 'KeyU', k: 'KeyK', c: 'KeyC', d: 'KeyD',
              '[': 'BracketLeft', ']': 'BracketRight'
            }};
            return fallback[key] || '';
          }}

          const actions = {{
            KeyP: 'previous',
            KeyN: 'next',
            KeyS: 'save_next',
            KeyI: 'reset_image',
            KeyO: 'reset_auto',
            KeyA: 'accept_auto',
            KeyH: 'no_heart',
            KeyU: 'unusable',
            KeyK: 'skip',
            KeyC: 'clear',
            KeyD: 'delete',
            BracketLeft: 'brush_down',
            BracketRight: 'brush_up',
          }};
          const legacyBlockedCodes = new Set(['KeyR']);

          function blockEvent(event) {{
            if (!event) return;
            if (typeof event.preventDefault === 'function') event.preventDefault();
            if (typeof event.stopPropagation === 'function') event.stopPropagation();
            if (typeof event.stopImmediatePropagation === 'function') {{
              event.stopImmediatePropagation();
            }}
          }}

          function shortcutInfo(event) {{
            const code = normalizeCode(event);
            const action = actions[code] || '';
            const legacyBlocked = legacyBlockedCodes.has(code);
            if ((!action && !legacyBlocked) || event.ctrlKey || event.metaKey ||
                event.altKey || isTypingTarget(event.target)) return null;
            return {{code, action, legacyBlocked}};
          }}

          function onKeyDown(event) {{
            const info = shortcutInfo(event);
            if (!info) return;

            // The event is stopped before any message reaches Python. Therefore O
            // (Kaggle's output shortcut) remains exclusive to the editor, and
            // legacy R cannot modify or create notebook cells.
            blockEvent(event);
            focusKeyboardSink();
            if (info.legacyBlocked) {{
              held.delete(info.code);
              return;
            }}
            if (event.repeat || held.has(info.code)) return;

            const now = performance.now();
            const previous = lastFire.get(info.code) || -Infinity;
            if (now - previous < minimumGapMs) return;

            const target = commandTarget();
            if (!target) return;
            held.add(info.code);
            lastFire.set(info.code, now);
            serial += 1;
            target.value = `${{info.action}}|${{Date.now()}}|${{serial}}`;
            target.dispatchEvent(new Event('input', {{bubbles: true}}));
            target.dispatchEvent(new Event('change', {{bubbles: true}}));
          }}

          function onKeyPress(event) {{
            const info = shortcutInfo(event);
            if (info) blockEvent(event);
          }}

          function onKeyUp(event) {{
            const info = shortcutInfo(event);
            if (!info) return;
            blockEvent(event);
            held.delete(info.code);
            focusKeyboardSink();
          }}

          function releaseAll() {{ held.clear(); }}

          // Window/capture fires before document/capture. Together with the focused
          // input, it isolates editor shortcuts from notebook command mode and
          // prevents accidental cell creation or modification.
          window.addEventListener('keydown', onKeyDown, keyOptions);
          window.addEventListener('keypress', onKeyPress, keyOptions);
          window.addEventListener('keyup', onKeyUp, keyOptions);
          window.addEventListener('blur', releaseAll, true);
          document.addEventListener('visibilitychange', releaseAll, true);

          window[controllerKey] = {{
            editorId,
            dispose: () => {{
              window.removeEventListener('keydown', onKeyDown, true);
              window.removeEventListener('keypress', onKeyPress, true);
              window.removeEventListener('keyup', onKeyUp, true);
              window.removeEventListener('blur', releaseAll, true);
              document.removeEventListener('visibilitychange', releaseAll, true);
              const sink = keyboardSink();
              if (sink && document.activeElement === sink) {{
                try {{ sink.blur(); }} catch (_) {{}}
              }}
              held.clear();
            }}
          }};

          focusKeyboardSink();
          requestAnimationFrame(focusKeyboardSink);
          setTimeout(focusKeyboardSink, 0);
        }})();
        """

    def _shortcut_sync_changed(self, change: dict[str, Any]) -> None:


        payload = str(change.get("new", "") or "").strip()
        if not self.rows:
            return
        if not payload or payload == self._last_shortcut_payload:
            return
        self._last_shortcut_payload = payload
        action = payload.split("|", 1)[0].strip().lower()


        now = time.monotonic()
        if self._shortcut_busy or now - self._last_shortcut_at < 0.12:
            return

        actions = {
            "previous": ("Previous", self.previous),
            "next": ("Next", self.next),
            "save_next": ("Save & Next", self.save_next),
            "reset_image": ("Reset to image", self.reset_to_image),
            "reset_auto": ("Reset to auto", self.reset_to_auto),
            "accept_auto": ("Auto OK", self.accept_auto),
            "no_heart": ("No heart", self.mark_no_heart),
            "unusable": ("Unusable", self.mark_unusable),
            "skip": ("Skip", self.skip),
            "clear": ("Clear", self.clear_editable),
            "delete": ("Delete manual", self.delete_manual),
            "brush_down": (
                "Brush -",
                lambda: setattr(
                    self.brush_slider,
                    "value",
                    max(self.brush_slider.min, self.brush_slider.value - self.brush_slider.step),
                ),
            ),
            "brush_up": (
                "Brush +",
                lambda: setattr(
                    self.brush_slider,
                    "value",
                    min(self.brush_slider.max, self.brush_slider.value + self.brush_slider.step),
                ),
            ),
        }
        selected = actions.get(action)
        if selected is None:
            return

        self._shortcut_busy = True
        self._last_shortcut_at = now
        try:
            label, function = selected
            self._safe(label, function)
        finally:
            self._shortcut_busy = False

    def _mask_sync_changed(self, change: dict[str, Any]) -> None:
        value = change.get("new", "")
        if not value:
            return
        try:
            decoded = cv2.imdecode(
                np.frombuffer(base64.b64decode(value), np.uint8), cv2.IMREAD_UNCHANGED
            )
            if decoded is None:
                return
            if decoded.ndim == 3 and decoded.shape[2] == 4:
                alpha = decoded[..., 3]
                gray = cv2.cvtColor(decoded[..., :3], cv2.COLOR_BGR2GRAY)
                binary = ((alpha > 8) & (gray > 127)).astype(np.uint8)
            elif decoded.ndim == 3:
                binary = (cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8)
            else:
                binary = (decoded > 127).astype(np.uint8)
            self.mask = cv2.resize(
                binary,
                (Settings.Image.SEGMENTATION_SIZE, Settings.Image.SEGMENTATION_SIZE),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        except Exception as error:
            print("[EDITOR WARNING] Mask synchronization failed:", error)

    def _sync_python_mask(self) -> None:
        value = self._mask_uri(self.mask).split(",", 1)[1]
        if self.mask_sync.value != value:
            self.mask_sync.value = value

    def update_status(self, prefix: str = "") -> None:
        if not self.rows:
            self.status.value = (
                "<span style='margin-left:12px'><b>Review complete.</b> "
                "No target=UNLABELED images remain.</span>"
            )
            return
        row = self.rows[self.index]
        manual = MaskManager.manual_qc(row["manual_mask_path"])
        annotation = MaskManager.annotation_map(self.workspace).get(
            str(row["image_token"]), {}
        )
        target_type = (
            ReviewManager._normalize_target_type(annotation.get("target_type", ""))
            or "UNLABELED"
        )
        self.status.value = (
            "<span style='margin-left:12px'>"
            f"<b>{prefix}</b> index={self.index + 1}/{len(self.rows)}; "
            f"target={target_type}; manual={'yes' if manual['exists'] else 'no'}; "
            f"usable={manual['usable']}; attention_valid={row.get('attention_valid_final', '')}; "
            f"presence={_as_float(row.get('attention_presence_probability'), np.nan):.3f}; "
            f"uncertainty={_as_float(row.get('attention_uncertainty_score'), np.nan):.3f}; "
            f"reason={row.get('attention_invalid_reason_final', '')}</span>"
        )

    def _write_target(
        self,
        mask: np.ndarray,
        target_type: str,
        source: str,
        action: str,
        sample_weight: float | None = None,
    ) -> None:
        row = self.rows[self.index]
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        path = Path(row["manual_mask_path"])
        FileManager.write_png(path, binary * 255)
        MaskManager.set_annotation(
            self.workspace,
            row,
            target_type=target_type,
            source=source,
            sample_weight=sample_weight,
        )
        row["manual_annotation_type"] = target_type
        row["manual_annotation_source"] = source

        overlay = np.stack([self.image] * 3, axis=-1)
        overlay[..., 0] = np.maximum(overlay[..., 0], binary * 0.90)
        overlay[..., 1] *= 1.0 - 0.45 * binary
        overlay[..., 2] *= 1.0 - 0.45 * binary
        FileManager.write_png(
            self.workspace.mask_overlays / f"{row['image_token']}.png",
            cv2.cvtColor(
                np.clip(np.round(overlay * 255.0), 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
        )
        ReviewManager.log(
            self.workspace,
            row,
            self.review_round,
            action,
            binary,
            self.index,
            len(self.rows),
        )
        self.mask = binary.copy()

    def save(self) -> None:
        if float((self.mask > 0).mean()) < Settings.Segmentation.MANUAL_MIN_AREA_RATIO:
            raise ValueError(
                "The mask is empty. Use No heart [H] for a valid image "
                "without a visible heart, or Unusable [U] for blur/noise/localizer frames."
            )
        self._write_target(
            self.mask,
            MaskManager.HEART_PRESENT,
            source="human_drawn_or_corrected",
            action="save_manual_heart_present",
            sample_weight=Settings.Segmentation.MANUAL_DRAWN_WEIGHT,
        )
        self.update_status("Saved HEART_PRESENT; weight=1.00.")
        print("[EDITOR]", self.rows[self.index]["manual_mask_path"], "| HEART_PRESENT")

    def save_next(self) -> None:
        self.save()
        self._advance_after_target("Saved HEART_PRESENT.")

    def accept_auto(self) -> None:
        if float((self.auto_mask > 0).mean()) < Settings.Segmentation.MANUAL_MIN_AREA_RATIO:
            raise ValueError(
                "The automatic mask is empty; use No heart [H], not Auto OK."
            )
        self._write_target(
            self.auto_mask,
            MaskManager.HEART_PRESENT,
            source="human_confirmed_auto",
            action="accept_auto_and_save",
            sample_weight=Settings.Segmentation.AUTO_CONFIRMED_WEIGHT,
        )
        self._advance_after_target(
            f"Auto saved HEART_PRESENT; weight={Settings.Segmentation.AUTO_CONFIRMED_WEIGHT:.2f}."
        )

    def mark_no_heart(self) -> None:
        empty = np.zeros_like(self.auto_mask, dtype=np.uint8)
        self._write_target(
            empty,
            MaskManager.NO_HEART_VISIBLE,
            source="human_no_heart_visible",
            action="mark_no_heart_visible",
            sample_weight=Settings.Segmentation.NO_HEART_WEIGHT,
        )
        self._advance_after_target("Saved NO_HEART_VISIBLE negative target.")

    def mark_unusable(self) -> None:
        row = self.rows[self.index]
        Path(row["manual_mask_path"]).unlink(missing_ok=True)
        MaskManager.set_annotation(
            self.workspace,
            row,
            target_type=MaskManager.UNUSABLE,
            source="human_unusable",
            sample_weight=0.0,
        )
        row["manual_annotation_type"] = MaskManager.UNUSABLE
        ReviewManager.log(
            self.workspace,
            row,
            self.review_round,
            "mark_unusable",
            None,
            self.index,
            len(self.rows),
        )
        self._advance_after_target("Marked UNUSABLE; excluded from training.")

    def skip(self) -> None:
        ReviewManager.log(
            self.workspace,
            self.rows[self.index],
            self.review_round,
            "skip",
            None,
            self.index,
            len(self.rows),
        )
        self.next()

    def reset_to_auto(self) -> None:
        self.base_mask = self.auto_mask.copy()
        self.mask = self.auto_mask.copy()
        self._sync_python_mask()
        self.render()
        self.update_status("Automatic mask restored.")

    def reset_to_image(self) -> None:
        self.base_mask = np.zeros_like(self.auto_mask)
        self.mask = np.zeros_like(self.auto_mask)
        self._sync_python_mask()
        self.render()
        self.update_status("All overlays hidden; not saved yet.")

    def clear_editable(self) -> None:
        self.base_mask = self.auto_mask.copy()
        self.mask = np.zeros_like(self.auto_mask)
        self._sync_python_mask()
        self.render()
        self.update_status("Editable mask cleared; auto remains as guide.")

    def delete_manual(self) -> None:
        row = self.rows[self.index]
        Path(row["manual_mask_path"]).unlink(missing_ok=True)
        MaskManager.remove_annotation(self.workspace, str(row["image_token"]))
        row["manual_annotation_type"] = ""
        row["manual_annotation_source"] = ""
        ReviewManager.log(
            self.workspace,
            row,
            self.review_round,
            "delete_manual_and_annotation",
            None,
            self.index,
            len(self.rows),
        )
        self.mask = self.auto_mask.copy()
        self.base_mask = self.auto_mask.copy()
        self._sync_python_mask()
        self.render()
        self.update_status("Manual target and annotation deleted.")

    def next(self) -> None:
        if not self.rows:
            return
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.load_current()
        else:
            self.update_status("End of queue.")

    def previous(self) -> None:
        if not self.rows:
            return
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def show(self):
        display(self.controls)
        display(Javascript(self._shortcut_script()))
        return self


# -----------------------------------------------------------------------------
# Frozen feature extraction, confounder matching, and patient aggregation
# -----------------------------------------------------------------------------
class FeatureDataset(Dataset):


    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        robust, raw, content = ImageProcessor.classifier_views(row["image_path"])
        attention_path = Path(row["predicted_attention_mask_path"])
        if attention_path.is_file():
            attention = MaskManager.read_binary(
                attention_path, size=Settings.Image.CLASSIFICATION_SIZE
            ).astype(np.float32)
        else:
            attention = np.zeros(
                (Settings.Image.CLASSIFICATION_SIZE, Settings.Image.CLASSIFICATION_SIZE),
                dtype=np.float32,
            )
        manual_path = Path(row["manual_mask_path"])
        if manual_path.is_file():
            manual = MaskManager.read_binary(
                manual_path, size=Settings.Image.CLASSIFICATION_SIZE
            ).astype(np.float32)
        else:
            manual = np.zeros_like(attention)

        robust_tensor = torch.from_numpy(np.stack([robust] * 3)).float()
        raw_tensor = torch.from_numpy(np.stack([raw] * 3)).float()
        return (
            robust_tensor,
            raw_tensor,
            torch.from_numpy(content).unsqueeze(0).float(),
            torch.from_numpy(attention).unsqueeze(0).float(),
            torch.from_numpy(manual).unsqueeze(0).float(),
            int(index),
        )


class FrozenEfficientNet(nn.Module):


    def __init__(self):
        super().__init__()
        weights = (
            EfficientNet_B0_Weights.IMAGENET1K_V1
            if Settings.Classification.USE_IMAGENET_WEIGHTS
            else None
        )
        try:
            model = efficientnet_b0(weights=weights)
        except Exception as error:
            raise RuntimeError(
                "EfficientNet-B0 weights could not be loaded. "
                "Enable Internet in Kaggle or place the weights in the Torch cache. "
                f"Original error: {type(error).__name__}: {error}"
            ) from error
        model.classifier = nn.Identity()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model.eval()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images.float() - self.mean) / self.std
        return self.model(images)


class StreamingPatientPool:


    def __init__(self, modes: Sequence[str]):
        self.modes = tuple(modes)
        self.series_data: dict[str, dict[tuple[str, str], list[Any]]] = {
            mode: {} for mode in self.modes
        }
        self.source_slices: dict[str, int] = defaultdict(int)

    def add(
        self,
        mode: str,
        embeddings: np.ndarray,
        rows: Sequence[dict[str, Any]],
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        for embedding, row in zip(embeddings, rows):
            key = (str(row["patient_id"]), str(row["series_id"]))
            entry = self.series_data[mode].get(key)
            if entry is None:
                entry = [np.zeros_like(embedding, dtype=np.float32), 0, int(row["label"])]
                self.series_data[mode][key] = entry
            if int(entry[2]) != int(row["label"]):
                raise RuntimeError(f"Inconsistent labels for patient {row['patient_id']}.")
            entry[0] += embedding
            entry[1] += 1
            self.source_slices[mode] += 1

    def finalize(self, mode: str) -> dict[str, Any]:
        patient_series: dict[str, list[np.ndarray]] = defaultdict(list)
        patient_labels: dict[str, int] = {}
        for (patient_id, _series_id), (embedding_sum, count, label) in self.series_data[mode].items():
            patient_series[patient_id].append(embedding_sum / max(1, count))
            patient_labels[patient_id] = int(label)
        patients = sorted(patient_series)
        if not patients:
            raise RuntimeError(
                f"Mode {mode} contains no patients. For M1/C3/AU3/C4, "
                "inspect [FEATURE BANK][MANUAL] and the manual-audit schema; "
                "legacy audits are rebuilt automatically."
            )
        X = np.stack(
            [np.mean(np.stack(patient_series[patient]), axis=0) for patient in patients]
        ).astype(np.float32)
        y = np.asarray([patient_labels[patient] for patient in patients], dtype=np.int64)
        return {
            "X": X,
            "y": y,
            "patient_ids": np.asarray(patients),
            "source_slices": int(self.source_slices[mode]),
            "series_proxies": int(len(self.series_data[mode])),
        }


class CrossClassMatchingManager:
    """Construct a balanced Sick/Normal sensitivity cohort in two stages.

    Stage 1 is the previous strict mutual-neighbour cohort. Stage 2 keeps every
    strict pair and adds unique one-to-one pairs from the same acquisition family.
    Extended candidates must still satisfy sequence-position and mask-area gates,
    a family-specific distance caliper, per-patient caps, and per-sequence caps.
    """

    MANIFEST_FIELDS = (
        "image_token",
        "image_path",
        "patient_id",
        "series_id",
        "sequence_group_id",
        "label",
        "class_name",
        "eligible_for_matching",
        "acquisition_family",
        "shared_family",
        "selected_for_core_matched_cohort",
        "selected_for_matched_cohort",
        "match_stage",
        "pair_id",
        "matched_partner_token",
        "matched_partner_patient",
        "matched_partner_series",
        "match_distance",
        "mutual_rank_from_sick",
        "mutual_rank_from_normal",
        "reciprocal_extended_neighbor",
        "sequence_pair_seeded_by_core",
        "family_core_caliper",
        "family_extended_caliper",
        "family_caliper",
        "sequence_position",
        "attention_area_ratio",
        "exclusion_reason",
    )

    @staticmethod
    def _settings_payload() -> dict[str, Any]:
        return {
            "schema": "cross-class-matching-v2-core-plus-extended",
            "min_families": Settings.Matching.MIN_FAMILIES,
            "max_families": Settings.Matching.MAX_FAMILIES,
            "target_images_per_family": Settings.Matching.TARGET_IMAGES_PER_FAMILY,
            "minimum_images_per_class_per_family": Settings.Matching.MIN_IMAGES_PER_CLASS_PER_FAMILY,
            "minimum_patients_per_class_per_family": Settings.Matching.MIN_PATIENTS_PER_CLASS_PER_FAMILY,
            "core_mutual_neighbors": Settings.Matching.MUTUAL_NEIGHBORS,
            "core_caliper_mad_multiplier": Settings.Matching.CALIPER_MAD_MULTIPLIER,
            "core_caliper_quantile": Settings.Matching.CALIPER_QUANTILE,
            "maximum_sequence_position_difference": Settings.Matching.MAX_SEQUENCE_POSITION_DIFFERENCE,
            "maximum_area_ratio_difference": Settings.Matching.MAX_AREA_RATIO_DIFFERENCE,
            "core_maximum_matches_per_patient_per_family": Settings.Matching.MAX_MATCHES_PER_PATIENT_PER_FAMILY,
            "core_maximum_matches_per_sequence_group": Settings.Matching.MAX_MATCHES_PER_SEQUENCE_GROUP,
            "target_matched_image_fraction": Settings.Matching.TARGET_MATCHED_IMAGE_FRACTION,
            "extended_neighbors": Settings.Matching.EXTENDED_NEIGHBORS,
            "extended_caliper_mad_multiplier": Settings.Matching.EXTENDED_CALIPER_MAD_MULTIPLIER,
            "extended_caliper_quantile": Settings.Matching.EXTENDED_CALIPER_QUANTILE,
            "extended_max_caliper_multiplier": Settings.Matching.EXTENDED_MAX_CALIPER_MULTIPLIER,
            "extended_maximum_sequence_position_difference": Settings.Matching.EXTENDED_MAX_SEQUENCE_POSITION_DIFFERENCE,
            "extended_maximum_area_ratio_difference": Settings.Matching.EXTENDED_MAX_AREA_RATIO_DIFFERENCE,
            "extended_maximum_matches_per_patient_per_family": Settings.Matching.EXTENDED_MAX_MATCHES_PER_PATIENT_PER_FAMILY,
            "extended_maximum_matches_per_sequence_group": Settings.Matching.EXTENDED_MAX_MATCHES_PER_SEQUENCE_GROUP,
            "extended_maximum_matches_per_sequence_pair": Settings.Matching.EXTENDED_MAX_MATCHES_PER_SEQUENCE_PAIR,
            "extended_maximum_matches_per_patient_pair": Settings.Matching.EXTENDED_MAX_MATCHES_PER_PATIENT_PAIR,
            "patient_total_cap_multiplier": Settings.Matching.PATIENT_TOTAL_CAP_MULTIPLIER,
            "prioritize_core_sequence_pairs": Settings.Matching.PRIORITIZE_CORE_SEQUENCE_PAIRS,
            "block_weights": {
                "phash": Settings.Matching.PHASH_BLOCK_WEIGHT,
                "geometry": Settings.Matching.GEOMETRY_BLOCK_WEIGHT,
                "sequence": Settings.Matching.SEQUENCE_BLOCK_WEIGHT,
                "quality": Settings.Matching.QUALITY_BLOCK_WEIGHT,
            },
            "random_seed": Settings.Runtime.RANDOM_SEED,
        }

    @staticmethod
    def _fingerprint(
        dataset_rows: Sequence[dict[str, Any]], workspace: Workspace
    ) -> str:
        digest = hashlib.sha256(
            json.dumps(
                CrossClassMatchingManager._settings_payload(), sort_keys=True
            ).encode("utf-8")
        )
        for row in dataset_rows:
            digest.update(str(row.get("image_token", "")).encode("utf-8"))
            digest.update(str(row.get("label", "")).encode("ascii"))
            digest.update(str(row.get("patient_id", "")).encode("utf-8"))
            digest.update(str(row.get("sequence_group_id", "")).encode("utf-8"))
            digest.update(str(row.get("sequence_index", "")).encode("ascii"))
            digest.update(str(row.get("sequence_length", "")).encode("ascii"))
        for path in (workspace.quality_audit, workspace.prediction_audit):
            if path.is_file():
                digest.update(path.name.encode("utf-8"))
                digest.update(FileManager.sha256_file(path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _sequence_position(row: dict[str, Any]) -> float:
        length = max(1, _as_int(row.get("sequence_length"), 1))
        index = int(np.clip(_as_int(row.get("sequence_index"), 0), 0, length - 1))
        if length <= 1:
            return 0.5
        return float(index / (length - 1))

    @staticmethod
    def _phash_bits(value: str) -> np.ndarray:
        number = int(str(value), 16)
        return np.asarray(
            [(number >> shift) & 1 for shift in range(63, -1, -1)],
            dtype=np.float32,
        )

    @staticmethod
    def _robust_standardize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).copy()
        if values.ndim == 1:
            values = values[:, None]
        for column in range(values.shape[1]):
            current = values[:, column]
            finite = np.isfinite(current)
            center = float(np.median(current[finite])) if finite.any() else 0.0
            current[~finite] = center
            mad = float(np.median(np.abs(current - center)))
            scale = max(1e-3, 1.4826 * mad)
            values[:, column] = np.clip((current - center) / scale, -5.0, 5.0)
        return values

    @staticmethod
    def _descriptor(rows: Sequence[dict[str, Any]]) -> np.ndarray:
        phash = np.stack(
            [CrossClassMatchingManager._phash_bits(row["perceptual_hash"]) for row in rows]
        )
        phash = phash * 2.0 - 1.0
        geometry = CrossClassMatchingManager._robust_standardize(
            np.asarray(
                [
                    [
                        _as_float(row.get("attention_area_ratio"), np.nan),
                        _as_float(row.get("attention_centroid_x"), np.nan),
                        _as_float(row.get("attention_centroid_y"), np.nan),
                        _as_float(row.get("attention_boundary_touch_fraction"), np.nan),
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            )
        )
        sequence = CrossClassMatchingManager._robust_standardize(
            np.asarray(
                [[CrossClassMatchingManager._sequence_position(row)] for row in rows],
                dtype=np.float32,
            )
        )
        quality = CrossClassMatchingManager._robust_standardize(
            np.asarray(
                [
                    [
                        math.log1p(max(0.0, _as_float(row.get("sharpness"), 0.0))),
                        _as_float(row.get("noise_ratio"), np.nan),
                        _as_float(row.get("dynamic_range"), np.nan),
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            )
        )

        def block_scale(weight: float, dimensions: int) -> float:
            return math.sqrt(max(float(weight), 0.0) / max(1, int(dimensions)))

        descriptor = np.concatenate(
            [
                phash * block_scale(Settings.Matching.PHASH_BLOCK_WEIGHT, phash.shape[1]),
                geometry
                * block_scale(Settings.Matching.GEOMETRY_BLOCK_WEIGHT, geometry.shape[1]),
                sequence
                * block_scale(Settings.Matching.SEQUENCE_BLOCK_WEIGHT, sequence.shape[1]),
                quality
                * block_scale(Settings.Matching.QUALITY_BLOCK_WEIGHT, quality.shape[1]),
            ],
            axis=1,
        )
        return np.ascontiguousarray(descriptor, dtype=np.float32)

    @staticmethod
    def _merge_rows(
        dataset_rows: Sequence[dict[str, Any]], workspace: Workspace
    ) -> list[dict[str, Any]]:
        quality = {
            str(row.get("image_token", "")): row
            for row in FileManager.read_csv(workspace.quality_audit)
        }
        predictions = {
            str(row.get("image_token", "")): row
            for row in FileManager.read_csv(workspace.prediction_audit)
        }
        merged: list[dict[str, Any]] = []
        for original in dataset_rows:
            row = dict(original)
            token = str(row["image_token"])
            row.update(predictions.get(token, {}))
            row.update(quality.get(token, {}))
            centroid_x = _as_float(row.get("attention_centroid_x"), np.nan)
            centroid_y = _as_float(row.get("attention_centroid_y"), np.nan)
            if not (np.isfinite(centroid_x) and np.isfinite(centroid_y)):
                mask_path = Path(str(row.get("predicted_attention_mask_path", "")))
                if mask_path.is_file():
                    try:
                        features = SegmentationManager._mask_features(
                            MaskManager.read_binary(mask_path)
                        )
                        row["attention_area_ratio"] = features["area"]
                        row["attention_boundary_touch_fraction"] = features["boundary"]
                        row["attention_centroid_x"] = features["centroid_x"]
                        row["attention_centroid_y"] = features["centroid_y"]
                    except Exception:
                        pass
            merged.append(row)
        return merged

    @staticmethod
    def _eligibility_reason(row: dict[str, Any]) -> str:
        reasons: list[str] = []
        if _as_int(row.get("quality_valid"), 0) != 1:
            reasons.append("quality_invalid")
        if _as_int(row.get("attention_valid_final"), 0) != 1:
            reasons.append("attention_invalid")
        if _as_int(row.get("attention_heart_present"), 1) != 1:
            reasons.append("heart_not_visible")
        if (
            _as_float(row.get("attention_area_ratio"), 0.0)
            < Settings.Segmentation.PREDICTION_MIN_AREA_RATIO
        ):
            reasons.append("attention_area_too_small")
        if not str(row.get("perceptual_hash", "")).strip():
            reasons.append("missing_phash")
        if not Path(str(row.get("predicted_attention_mask_path", ""))).is_file():
            reasons.append("missing_attention_mask")
        return ";".join(reasons)

    @staticmethod
    def _family_count(number_of_images: int) -> int:
        estimate = int(
            math.ceil(
                number_of_images
                / max(1, Settings.Matching.TARGET_IMAGES_PER_FAMILY)
            )
        )
        count = int(
            np.clip(
                estimate,
                Settings.Matching.MIN_FAMILIES,
                Settings.Matching.MAX_FAMILIES,
            )
        )
        return max(2, min(count, number_of_images))

    @staticmethod
    def _family_is_shared(rows: Sequence[dict[str, Any]]) -> bool:
        for label in (0, 1):
            class_rows = [row for row in rows if _as_int(row.get("label"), -1) == label]
            if len(class_rows) < Settings.Matching.MIN_IMAGES_PER_CLASS_PER_FAMILY:
                return False
            if (
                len({str(row.get("patient_id", "")) for row in class_rows})
                < Settings.Matching.MIN_PATIENTS_PER_CLASS_PER_FAMILY
            ):
                return False
        return True

    @staticmethod
    def _robust_caliper(
        distances: Sequence[float],
        mad_multiplier: float,
        quantile: float,
    ) -> float:
        values = np.asarray(list(distances), dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return np.nan
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = max(1e-9, 1.4826 * mad)
        mad_caliper = median + float(mad_multiplier) * robust_scale
        quantile_caliper = float(np.quantile(values, float(quantile)))
        return float(max(median, min(mad_caliper, quantile_caliper)))

    @staticmethod
    def _edge_passes_anatomical_gates(
        sick_row: dict[str, Any],
        normal_row: dict[str, Any],
        extended: bool,
    ) -> bool:
        sequence_limit = (
            Settings.Matching.EXTENDED_MAX_SEQUENCE_POSITION_DIFFERENCE
            if extended
            else Settings.Matching.MAX_SEQUENCE_POSITION_DIFFERENCE
        )
        area_limit = (
            Settings.Matching.EXTENDED_MAX_AREA_RATIO_DIFFERENCE
            if extended
            else Settings.Matching.MAX_AREA_RATIO_DIFFERENCE
        )
        if (
            abs(
                CrossClassMatchingManager._sequence_position(sick_row)
                - CrossClassMatchingManager._sequence_position(normal_row)
            )
            > sequence_limit
        ):
            return False
        sick_area = _as_float(sick_row.get("attention_area_ratio"), np.nan)
        normal_area = _as_float(normal_row.get("attention_area_ratio"), np.nan)
        if (
            np.isfinite(sick_area)
            and np.isfinite(normal_area)
            and abs(sick_area - normal_area) > area_limit
        ):
            return False
        return True

    @staticmethod
    def _pair_keys(
        sick_row: dict[str, Any], normal_row: dict[str, Any], family: int
    ) -> tuple[
        tuple[str, int],
        tuple[str, int],
        tuple[str, int],
        tuple[str, int],
        tuple[str, str],
        tuple[str, str],
    ]:
        sick_patient = str(sick_row["patient_id"])
        normal_patient = str(normal_row["patient_id"])
        sick_sequence = str(
            sick_row.get("sequence_group_id") or sick_row.get("series_id", "")
        )
        normal_sequence = str(
            normal_row.get("sequence_group_id") or normal_row.get("series_id", "")
        )
        return (
            (sick_patient, family),
            (normal_patient, family),
            (sick_sequence, family),
            (normal_sequence, family),
            (sick_patient, normal_patient),
            (sick_sequence, normal_sequence),
        )

    @staticmethod
    def build(
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        fingerprint = CrossClassMatchingManager._fingerprint(dataset_rows, workspace)
        old_summary = FileManager.read_json(workspace.cross_class_matching_summary, {}) or {}
        old_manifest = FileManager.read_csv(workspace.cross_class_matching_manifest)
        if (
            not force
            and old_summary.get("fingerprint") == fingerprint
            and len(old_manifest) == len(dataset_rows)
            and any(
                _as_int(row.get("selected_for_matched_cohort"), 0) == 1
                for row in old_manifest
            )
        ):
            print(
                "[MATCHING] Compatible expanded Sick↔Normal manifest reused: "
                f"{old_summary.get('matched_pairs', '?')} total pairs "
                f"({old_summary.get('core_matched_pairs', '?')} strict core)."
            )
            return old_manifest

        if not Settings.Matching.ENABLED:
            raise RuntimeError("Cross-class matching is disabled in Settings.Matching.")

        merged = CrossClassMatchingManager._merge_rows(dataset_rows, workspace)
        manifest_by_token: dict[str, dict[str, Any]] = {}
        eligible_rows: list[dict[str, Any]] = []
        for row in merged:
            token = str(row["image_token"])
            reason = CrossClassMatchingManager._eligibility_reason(row)
            record = {
                "image_token": token,
                "image_path": str(row.get("image_path", "")),
                "patient_id": str(row.get("patient_id", "")),
                "series_id": str(row.get("series_id", "")),
                "sequence_group_id": str(
                    row.get("sequence_group_id") or row.get("series_id", "")
                ),
                "label": _as_int(row.get("label"), -1),
                "class_name": "Sick" if _as_int(row.get("label"), -1) == 1 else "Normal",
                "eligible_for_matching": int(not reason),
                "acquisition_family": "",
                "shared_family": 0,
                "selected_for_core_matched_cohort": 0,
                "selected_for_matched_cohort": 0,
                "match_stage": "",
                "pair_id": "",
                "matched_partner_token": "",
                "matched_partner_patient": "",
                "matched_partner_series": "",
                "match_distance": "",
                "mutual_rank_from_sick": "",
                "mutual_rank_from_normal": "",
                "reciprocal_extended_neighbor": "",
                "sequence_pair_seeded_by_core": "",
                "family_core_caliper": "",
                "family_extended_caliper": "",
                "family_caliper": "",
                "sequence_position": CrossClassMatchingManager._sequence_position(row),
                "attention_area_ratio": _as_float(
                    row.get("attention_area_ratio"), np.nan
                ),
                "exclusion_reason": reason,
            }
            manifest_by_token[token] = record
            if not reason:
                eligible_rows.append(row)

        labels = np.asarray(
            [_as_int(row.get("label"), -1) for row in eligible_rows], dtype=np.int64
        )
        if len(eligible_rows) < 4 or set(labels.tolist()) != {0, 1}:
            raise RuntimeError(
                "Matching requires eligible images from both classes; "
                f"remaining Normal={int(np.sum(labels == 0))}, Sick={int(np.sum(labels == 1))}."
            )

        descriptor = CrossClassMatchingManager._descriptor(eligible_rows)
        family_count = CrossClassMatchingManager._family_count(len(eligible_rows))
        batch_size = min(4096, max(256, len(eligible_rows) // 10))
        clusterer = MiniBatchKMeans(
            n_clusters=family_count,
            random_state=Settings.Runtime.RANDOM_SEED,
            batch_size=batch_size,
            n_init=5,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        family_labels = clusterer.fit_predict(descriptor).astype(np.int64)

        family_indices: dict[int, list[int]] = defaultdict(list)
        for index, family in enumerate(family_labels.tolist()):
            family_indices[int(family)].append(index)
            token = str(eligible_rows[index]["image_token"])
            manifest_by_token[token]["acquisition_family"] = int(family)

        shared_families: set[int] = set()
        for family, indices in family_indices.items():
            family_rows = [eligible_rows[index] for index in indices]
            if CrossClassMatchingManager._family_is_shared(family_rows):
                shared_families.add(int(family))
                for index in indices:
                    manifest_by_token[str(eligible_rows[index]["image_token"])][
                        "shared_family"
                    ] = 1
            else:
                for index in indices:
                    manifest_by_token[str(eligible_rows[index]["image_token"])][
                        "exclusion_reason"
                    ] = "family_not_shared_between_classes"

        if not shared_families:
            raise RuntimeError(
                "No acquisition family contains enough images and patients from both classes."
            )

        normal_count = int(np.sum(labels == 0))
        sick_count = int(np.sum(labels == 1))
        maximum_possible_pairs = min(normal_count, sick_count)
        requested_target_pairs = int(
            round(
                len(eligible_rows)
                * float(Settings.Matching.TARGET_MATCHED_IMAGE_FRACTION)
                / 2.0
            )
        )
        requested_target_pairs = max(1, min(requested_target_pairs, maximum_possible_pairs))

        patients_by_label = {
            label: sorted(
                {
                    str(row["patient_id"])
                    for row in eligible_rows
                    if _as_int(row.get("label"), -1) == label
                }
            )
            for label in (0, 1)
        }
        patient_total_caps = {
            label: max(
                Settings.Matching.EXTENDED_MAX_MATCHES_PER_PATIENT_PER_FAMILY,
                int(
                    math.ceil(
                        requested_target_pairs
                        / max(1, len(patients_by_label[label]))
                        * Settings.Matching.PATIENT_TOTAL_CAP_MULTIPLIER
                    )
                ),
            )
            for label in (0, 1)
        }

        used_tokens: set[str] = set()
        patient_family_counts: dict[tuple[str, int], int] = defaultdict(int)
        sequence_counts: dict[tuple[str, int], int] = defaultdict(int)
        patient_total_counts: dict[str, int] = defaultdict(int)
        patient_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        sequence_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        core_sequence_pairs: set[tuple[str, str]] = set()
        selected_pair_distances: list[float] = []
        core_pair_distances: list[float] = []
        extended_pair_distances: list[float] = []
        family_summaries: list[dict[str, Any]] = []
        pair_number = 0
        core_pair_count = 0
        extended_pair_count = 0

        family_data: dict[int, dict[str, Any]] = {}
        token_candidate_stage: dict[str, str] = {}

        # Build candidate graphs first. The extended graph is the union of the
        # top-K neighbours from both directions; the strict graph is its mutual
        # top-5 subset.
        for family in sorted(shared_families):
            indices = family_indices[family]
            normal_positions = [
                index for index in indices if _as_int(eligible_rows[index].get("label"), -1) == 0
            ]
            sick_positions = [
                index for index in indices if _as_int(eligible_rows[index].get("label"), -1) == 1
            ]
            normal_descriptor = descriptor[normal_positions]
            sick_descriptor = descriptor[sick_positions]
            extended_sick_to_normal = min(
                int(Settings.Matching.EXTENDED_NEIGHBORS), len(normal_positions)
            )
            extended_normal_to_sick = min(
                int(Settings.Matching.EXTENDED_NEIGHBORS), len(sick_positions)
            )

            normal_model = NearestNeighbors(
                n_neighbors=extended_sick_to_normal,
                metric="euclidean",
                algorithm="auto",
                n_jobs=-1,
            ).fit(normal_descriptor)
            sick_to_normal_distance, sick_to_normal_index = normal_model.kneighbors(
                sick_descriptor, return_distance=True
            )
            sick_model = NearestNeighbors(
                n_neighbors=extended_normal_to_sick,
                metric="euclidean",
                algorithm="auto",
                n_jobs=-1,
            ).fit(sick_descriptor)
            normal_to_sick_distance, normal_to_sick_index = sick_model.kneighbors(
                normal_descriptor, return_distance=True
            )

            sick_rank_maps = [
                {int(normal_local): int(rank + 1) for rank, normal_local in enumerate(neighbours)}
                for neighbours in sick_to_normal_index
            ]
            normal_rank_maps = [
                {int(sick_local): int(rank + 1) for rank, sick_local in enumerate(neighbours)}
                for neighbours in normal_to_sick_index
            ]
            edge_keys: set[tuple[int, int]] = set()
            for sick_local, neighbours in enumerate(sick_to_normal_index):
                edge_keys.update((int(sick_local), int(value)) for value in neighbours)
            for normal_local, neighbours in enumerate(normal_to_sick_index):
                edge_keys.update((int(value), int(normal_local)) for value in neighbours)

            all_edges: list[dict[str, Any]] = []
            core_edges: list[dict[str, Any]] = []
            for sick_local, normal_local in edge_keys:
                sick_global = sick_positions[sick_local]
                normal_global = normal_positions[normal_local]
                sick_row = eligible_rows[sick_global]
                normal_row = eligible_rows[normal_global]
                if not CrossClassMatchingManager._edge_passes_anatomical_gates(
                    sick_row, normal_row, extended=True
                ):
                    continue
                rank_from_sick = sick_rank_maps[sick_local].get(normal_local)
                rank_from_normal = normal_rank_maps[normal_local].get(sick_local)
                distance = float(
                    np.linalg.norm(
                        descriptor[sick_global] - descriptor[normal_global]
                    )
                )
                reciprocal = rank_from_sick is not None and rank_from_normal is not None
                is_core = bool(
                    reciprocal
                    and rank_from_sick <= Settings.Matching.MUTUAL_NEIGHBORS
                    and rank_from_normal <= Settings.Matching.MUTUAL_NEIGHBORS
                    and CrossClassMatchingManager._edge_passes_anatomical_gates(
                        sick_row, normal_row, extended=False
                    )
                )
                sick_token = str(sick_row["image_token"])
                normal_token = str(normal_row["image_token"])
                token_candidate_stage.setdefault(sick_token, "extended_candidate")
                token_candidate_stage.setdefault(normal_token, "extended_candidate")
                if is_core:
                    token_candidate_stage[sick_token] = "core_candidate"
                    token_candidate_stage[normal_token] = "core_candidate"
                edge = {
                    "family": int(family),
                    "distance": distance,
                    "sick_global": int(sick_global),
                    "normal_global": int(normal_global),
                    "rank_from_sick": rank_from_sick,
                    "rank_from_normal": rank_from_normal,
                    "reciprocal": int(reciprocal),
                    "is_core": int(is_core),
                    "tie": hashlib.sha256(
                        f"{family}|{sick_token}|{normal_token}".encode("utf-8")
                    ).hexdigest(),
                }
                all_edges.append(edge)
                if is_core:
                    core_edges.append(edge)

            core_caliper = CrossClassMatchingManager._robust_caliper(
                [edge["distance"] for edge in core_edges],
                Settings.Matching.CALIPER_MAD_MULTIPLIER,
                Settings.Matching.CALIPER_QUANTILE,
            )
            extended_caliper = CrossClassMatchingManager._robust_caliper(
                [edge["distance"] for edge in all_edges],
                Settings.Matching.EXTENDED_CALIPER_MAD_MULTIPLIER,
                Settings.Matching.EXTENDED_CALIPER_QUANTILE,
            )
            if np.isfinite(core_caliper):
                maximum_extended = (
                    float(core_caliper)
                    * float(Settings.Matching.EXTENDED_MAX_CALIPER_MULTIPLIER)
                )
                if np.isfinite(extended_caliper):
                    extended_caliper = max(
                        float(core_caliper),
                        min(float(extended_caliper), maximum_extended),
                    )
                else:
                    extended_caliper = maximum_extended

            family_data[family] = {
                "normal_positions": normal_positions,
                "sick_positions": sick_positions,
                "core_edges": core_edges,
                "all_edges": all_edges,
                "core_caliper": core_caliper,
                "extended_caliper": extended_caliper,
                "core_selected": 0,
                "extended_selected": 0,
            }
            for index in indices:
                record = manifest_by_token[str(eligible_rows[index]["image_token"])]
                if np.isfinite(core_caliper):
                    record["family_core_caliper"] = float(core_caliper)
                if np.isfinite(extended_caliper):
                    record["family_extended_caliper"] = float(extended_caliper)

        def can_select(
            edge: dict[str, Any],
            stage: str,
            patient_quota: dict[int, int] | None = None,
        ) -> bool:
            family = int(edge["family"])
            sick_row = eligible_rows[int(edge["sick_global"])]
            normal_row = eligible_rows[int(edge["normal_global"])]
            sick_token = str(sick_row["image_token"])
            normal_token = str(normal_row["image_token"])
            if sick_token in used_tokens or normal_token in used_tokens:
                return False
            (
                sick_patient_key,
                normal_patient_key,
                sick_sequence_key,
                normal_sequence_key,
                patient_pair_key,
                sequence_pair_key,
            ) = CrossClassMatchingManager._pair_keys(sick_row, normal_row, family)
            if stage == "core_mutual":
                patient_family_limit = Settings.Matching.MAX_MATCHES_PER_PATIENT_PER_FAMILY
                sequence_limit = Settings.Matching.MAX_MATCHES_PER_SEQUENCE_GROUP
            else:
                patient_family_limit = Settings.Matching.EXTENDED_MAX_MATCHES_PER_PATIENT_PER_FAMILY
                sequence_limit = Settings.Matching.EXTENDED_MAX_MATCHES_PER_SEQUENCE_GROUP
            if (
                patient_family_counts[sick_patient_key] >= patient_family_limit
                or patient_family_counts[normal_patient_key] >= patient_family_limit
                or sequence_counts[sick_sequence_key] >= sequence_limit
                or sequence_counts[normal_sequence_key] >= sequence_limit
            ):
                return False
            if stage != "core_mutual":
                if (
                    patient_pair_counts[patient_pair_key]
                    >= Settings.Matching.EXTENDED_MAX_MATCHES_PER_PATIENT_PAIR
                    or sequence_pair_counts[sequence_pair_key]
                    >= Settings.Matching.EXTENDED_MAX_MATCHES_PER_SEQUENCE_PAIR
                ):
                    return False
                for row in (sick_row, normal_row):
                    label = _as_int(row.get("label"), -1)
                    patient = str(row["patient_id"])
                    cap = int(patient_total_caps[label])
                    if patient_total_counts[patient] >= cap:
                        return False
                    if patient_quota is not None and patient_total_counts[patient] >= int(
                        patient_quota[label]
                    ):
                        return False
            return True

        def select_edge(edge: dict[str, Any], stage: str) -> None:
            nonlocal pair_number, core_pair_count, extended_pair_count
            family = int(edge["family"])
            sick_row = eligible_rows[int(edge["sick_global"])]
            normal_row = eligible_rows[int(edge["normal_global"])]
            sick_token = str(sick_row["image_token"])
            normal_token = str(normal_row["image_token"])
            (
                sick_patient_key,
                normal_patient_key,
                sick_sequence_key,
                normal_sequence_key,
                patient_pair_key,
                sequence_pair_key,
            ) = CrossClassMatchingManager._pair_keys(sick_row, normal_row, family)
            seeded = int(sequence_pair_key in core_sequence_pairs)
            pair_number += 1
            prefix = "CORE" if stage == "core_mutual" else "EXT"
            pair_id = f"CCM_{prefix}_F{family:02d}_P{pair_number:06d}"
            applied_caliper = (
                family_data[family]["core_caliper"]
                if stage == "core_mutual"
                else family_data[family]["extended_caliper"]
            )
            for source_row, partner_row in (
                (sick_row, normal_row),
                (normal_row, sick_row),
            ):
                source_token = str(source_row["image_token"])
                record = manifest_by_token[source_token]
                record.update(
                    {
                        "selected_for_core_matched_cohort": int(stage == "core_mutual"),
                        "selected_for_matched_cohort": 1,
                        "match_stage": stage,
                        "pair_id": pair_id,
                        "matched_partner_token": str(partner_row["image_token"]),
                        "matched_partner_patient": str(partner_row["patient_id"]),
                        "matched_partner_series": str(partner_row["series_id"]),
                        "match_distance": float(edge["distance"]),
                        "mutual_rank_from_sick": (
                            "" if edge.get("rank_from_sick") is None else int(edge["rank_from_sick"])
                        ),
                        "mutual_rank_from_normal": (
                            "" if edge.get("rank_from_normal") is None else int(edge["rank_from_normal"])
                        ),
                        "reciprocal_extended_neighbor": int(edge.get("reciprocal", 0)),
                        "sequence_pair_seeded_by_core": seeded,
                        "family_caliper": float(applied_caliper),
                        "exclusion_reason": "",
                    }
                )
            used_tokens.update((sick_token, normal_token))
            patient_family_counts[sick_patient_key] += 1
            patient_family_counts[normal_patient_key] += 1
            sequence_counts[sick_sequence_key] += 1
            sequence_counts[normal_sequence_key] += 1
            patient_total_counts[str(sick_row["patient_id"])] += 1
            patient_total_counts[str(normal_row["patient_id"])] += 1
            patient_pair_counts[patient_pair_key] += 1
            sequence_pair_counts[sequence_pair_key] += 1
            selected_pair_distances.append(float(edge["distance"]))
            if stage == "core_mutual":
                core_sequence_pairs.add(sequence_pair_key)
                core_pair_distances.append(float(edge["distance"]))
                core_pair_count += 1
                family_data[family]["core_selected"] += 1
            else:
                extended_pair_distances.append(float(edge["distance"]))
                extended_pair_count += 1
                family_data[family]["extended_selected"] += 1

        # Stage 1: preserve the previous strict cohort.
        for family in sorted(shared_families):
            caliper = family_data[family]["core_caliper"]
            if not np.isfinite(caliper):
                continue
            for edge in sorted(
                family_data[family]["core_edges"],
                key=lambda item: (item["distance"], item["tie"]),
            ):
                if float(edge["distance"]) > float(caliper):
                    continue
                if can_select(edge, "core_mutual"):
                    select_edge(edge, "core_mutual")

        # Allocate the remaining target proportionally to the unused common
        # support of each family. This prevents a few large families from taking
        # the whole extension.
        remaining_target = max(0, requested_target_pairs - core_pair_count)
        remaining_capacity = {
            family: max(
                0,
                min(
                    len(family_data[family]["normal_positions"]),
                    len(family_data[family]["sick_positions"]),
                )
                - int(family_data[family]["core_selected"]),
            )
            for family in sorted(shared_families)
        }
        total_remaining_capacity = sum(remaining_capacity.values())
        extension_targets = {family: 0 for family in shared_families}
        if remaining_target > 0 and total_remaining_capacity > 0:
            raw_targets = {
                family: remaining_target
                * remaining_capacity[family]
                / total_remaining_capacity
                for family in shared_families
            }
            extension_targets = {
                family: min(
                    remaining_capacity[family], int(math.floor(raw_targets[family]))
                )
                for family in shared_families
            }
            unassigned = remaining_target - sum(extension_targets.values())
            for family in sorted(
                shared_families,
                key=lambda value: (
                    raw_targets[value] - math.floor(raw_targets[value]),
                    remaining_capacity[value],
                ),
                reverse=True,
            ):
                if unassigned <= 0:
                    break
                if extension_targets[family] < remaining_capacity[family]:
                    extension_targets[family] += 1
                    unassigned -= 1

        # Stage 2: add close one-sided or reciprocal top-K neighbours. Core-seeded
        # sequence pairs are considered first, then reciprocal candidates, then
        # distance/rank. Quota passes stop a single patient from monopolising a family.
        for family in sorted(shared_families):
            family_target = int(extension_targets.get(family, 0))
            if family_target <= 0:
                continue
            extended_caliper = family_data[family]["extended_caliper"]
            if not np.isfinite(extended_caliper):
                continue
            candidates = []
            for edge in family_data[family]["all_edges"]:
                if float(edge["distance"]) > float(extended_caliper):
                    continue
                sick_row = eligible_rows[int(edge["sick_global"])]
                normal_row = eligible_rows[int(edge["normal_global"])]
                sequence_pair = CrossClassMatchingManager._pair_keys(
                    sick_row, normal_row, family
                )[-1]
                seeded = int(sequence_pair in core_sequence_pairs)
                rank_sick = edge.get("rank_from_sick")
                rank_normal = edge.get("rank_from_normal")
                rank_sum = int(rank_sick or Settings.Matching.EXTENDED_NEIGHBORS + 1) + int(
                    rank_normal or Settings.Matching.EXTENDED_NEIGHBORS + 1
                )
                candidates.append(
                    {
                        **edge,
                        "seeded": seeded,
                        "rank_sum": rank_sum,
                        "normalized_distance": float(edge["distance"])
                        / max(float(extended_caliper), 1e-9),
                    }
                )
            candidates.sort(
                key=lambda item: (
                    0
                    if (
                        Settings.Matching.PRIORITIZE_CORE_SEQUENCE_PAIRS
                        and item["seeded"]
                    )
                    else 1,
                    0 if item.get("reciprocal", 0) else 1,
                    item["normalized_distance"],
                    item["rank_sum"],
                    item["tie"],
                )
            )

            selected_here = 0
            max_cap = max(patient_total_caps.values())
            starting_quota = max(
                1,
                min(
                    [patient_total_counts.get(patient, 0) for patient in patient_total_counts]
                    or [1]
                ),
            )
            quota_values = np.unique(
                np.ceil(
                    np.linspace(
                        starting_quota,
                        max_cap,
                        max(2, int(Settings.Matching.EXTENDED_BALANCE_QUOTA_STEPS)),
                    )
                ).astype(int)
            )
            for quota in quota_values:
                if selected_here >= family_target or pair_number >= requested_target_pairs:
                    break
                patient_quota = {0: int(quota), 1: int(quota)}
                for edge in candidates:
                    if selected_here >= family_target or pair_number >= requested_target_pairs:
                        break
                    if can_select(edge, "extended_neighbor", patient_quota=patient_quota):
                        select_edge(edge, "extended_neighbor")
                        selected_here += 1

        # Spillover pass: if a family could not fill its proportional allocation,
        # other families with unused high-quality edges may contribute the remainder.
        # All one-to-one and capacity constraints remain active.
        if pair_number < requested_target_pairs:
            spillover_added = 0
            for family in sorted(shared_families):
                if pair_number >= requested_target_pairs:
                    break
                extended_caliper = family_data[family]["extended_caliper"]
                if not np.isfinite(extended_caliper):
                    continue
                spillover_candidates = []
                for edge in family_data[family]["all_edges"]:
                    if float(edge["distance"]) > float(extended_caliper):
                        continue
                    sick_row = eligible_rows[int(edge["sick_global"])]
                    normal_row = eligible_rows[int(edge["normal_global"])]
                    sequence_pair = CrossClassMatchingManager._pair_keys(
                        sick_row, normal_row, family
                    )[-1]
                    rank_sick = edge.get("rank_from_sick")
                    rank_normal = edge.get("rank_from_normal")
                    spillover_candidates.append(
                        {
                            **edge,
                            "seeded": int(sequence_pair in core_sequence_pairs),
                            "rank_sum": int(
                                rank_sick or Settings.Matching.EXTENDED_NEIGHBORS + 1
                            )
                            + int(
                                rank_normal or Settings.Matching.EXTENDED_NEIGHBORS + 1
                            ),
                            "normalized_distance": float(edge["distance"])
                            / max(float(extended_caliper), 1e-9),
                        }
                    )
                spillover_candidates.sort(
                    key=lambda item: (
                        0
                        if (
                            Settings.Matching.PRIORITIZE_CORE_SEQUENCE_PAIRS
                            and item["seeded"]
                        )
                        else 1,
                        0 if item.get("reciprocal", 0) else 1,
                        item["normalized_distance"],
                        item["rank_sum"],
                        item["tie"],
                    )
                )
                for edge in spillover_candidates:
                    if pair_number >= requested_target_pairs:
                        break
                    if can_select(edge, "extended_neighbor", patient_quota=None):
                        select_edge(edge, "extended_neighbor")
                        spillover_added += 1
            if spillover_added:
                print(
                    "[MATCHING] Spillover redistributed "
                    f"{spillover_added} extended pairs across families."
                )

        # Explain every non-selected image in the manifest.
        for family in sorted(shared_families):
            data = family_data[family]
            candidate_tokens = set()
            below_extended_tokens = set()
            for edge in data["all_edges"]:
                sick_token = str(eligible_rows[int(edge["sick_global"])]["image_token"])
                normal_token = str(eligible_rows[int(edge["normal_global"])]["image_token"])
                candidate_tokens.update((sick_token, normal_token))
                if np.isfinite(data["extended_caliper"]) and float(edge["distance"]) <= float(
                    data["extended_caliper"]
                ):
                    below_extended_tokens.update((sick_token, normal_token))
            for index in family_indices[family]:
                token = str(eligible_rows[index]["image_token"])
                record = manifest_by_token[token]
                if _as_int(record.get("selected_for_matched_cohort"), 0) == 1:
                    continue
                if token not in candidate_tokens:
                    record["exclusion_reason"] = "no_cross_class_neighbor_in_extended_top_k"
                elif token not in below_extended_tokens:
                    record["exclusion_reason"] = "above_extended_family_caliper"
                elif pair_number >= requested_target_pairs:
                    record["exclusion_reason"] = "expanded_target_reached"
                else:
                    record["exclusion_reason"] = "one_to_one_or_capacity_limit"

            family_summaries.append(
                {
                    "family": family,
                    "normal_images": len(data["normal_positions"]),
                    "sick_images": len(data["sick_positions"]),
                    "normal_patients": len(
                        {
                            str(eligible_rows[index]["patient_id"])
                            for index in data["normal_positions"]
                        }
                    ),
                    "sick_patients": len(
                        {
                            str(eligible_rows[index]["patient_id"])
                            for index in data["sick_positions"]
                        }
                    ),
                    "core_candidate_edges": len(data["core_edges"]),
                    "extended_candidate_edges": len(data["all_edges"]),
                    "core_caliper": (
                        float(data["core_caliper"])
                        if np.isfinite(data["core_caliper"])
                        else None
                    ),
                    "extended_caliper": (
                        float(data["extended_caliper"])
                        if np.isfinite(data["extended_caliper"])
                        else None
                    ),
                    "core_selected_pairs": int(data["core_selected"]),
                    "extended_added_pairs": int(data["extended_selected"]),
                    "selected_pairs": int(data["core_selected"] + data["extended_selected"]),
                    "extended_target_pairs": int(extension_targets.get(family, 0)),
                }
            )

        manifest = [
            manifest_by_token[str(row["image_token"])] for row in dataset_rows
        ]
        selected = [
            row
            for row in manifest
            if _as_int(row.get("selected_for_matched_cohort"), 0) == 1
        ]
        core_selected = [
            row
            for row in manifest
            if _as_int(row.get("selected_for_core_matched_cohort"), 0) == 1
        ]
        selected_normal = [row for row in selected if _as_int(row.get("label"), -1) == 0]
        selected_sick = [row for row in selected if _as_int(row.get("label"), -1) == 1]
        normal_patients = sorted({str(row["patient_id"]) for row in selected_normal})
        sick_patients = sorted({str(row["patient_id"]) for row in selected_sick})
        if len(selected_normal) != len(selected_sick):
            raise RuntimeError("Internal matching produced unequal class sizes.")
        if len(normal_patients) < 2 or len(sick_patients) < 2:
            raise RuntimeError(
                "The matched cohort has too few patients for evaluation: "
                f"Normal={len(normal_patients)}, Sick={len(sick_patients)}."
            )

        per_patient: dict[str, int] = defaultdict(int)
        for row in selected:
            per_patient[str(row["patient_id"])] += 1
        selected_fraction_eligible = len(selected) / max(1, len(eligible_rows))
        selected_fraction_dataset = len(selected) / max(1, len(dataset_rows))
        summary = {
            "fingerprint": fingerprint,
            **CrossClassMatchingManager._settings_payload(),
            "dataset_images": len(dataset_rows),
            "eligible_images": len(eligible_rows),
            "acquisition_families": family_count,
            "shared_families": len(shared_families),
            "requested_target_pairs": requested_target_pairs,
            "requested_target_images": requested_target_pairs * 2,
            "core_matched_pairs": len(core_selected) // 2,
            "core_selected_images": len(core_selected),
            "extended_added_pairs": extended_pair_count,
            "extended_added_images": extended_pair_count * 2,
            "matched_pairs": len(selected_normal),
            "selected_images": len(selected),
            "selected_fraction_of_eligible": float(selected_fraction_eligible),
            "selected_fraction_of_dataset": float(selected_fraction_dataset),
            "target_achieved": bool(len(selected_normal) >= requested_target_pairs),
            "selected_normal_images": len(selected_normal),
            "selected_sick_images": len(selected_sick),
            "selected_normal_patients": len(normal_patients),
            "selected_sick_patients": len(sick_patients),
            "normal_patients": normal_patients,
            "sick_patients": sick_patients,
            "patient_total_caps_by_class": {
                "Normal": int(patient_total_caps[0]),
                "Sick": int(patient_total_caps[1]),
            },
            "selected_images_per_patient": dict(sorted(per_patient.items())),
            "distance_median": (
                float(np.median(selected_pair_distances))
                if selected_pair_distances
                else None
            ),
            "distance_p90": (
                float(np.quantile(selected_pair_distances, 0.90))
                if selected_pair_distances
                else None
            ),
            "core_distance_median": (
                float(np.median(core_pair_distances)) if core_pair_distances else None
            ),
            "core_distance_p90": (
                float(np.quantile(core_pair_distances, 0.90))
                if core_pair_distances
                else None
            ),
            "extended_distance_median": (
                float(np.median(extended_pair_distances))
                if extended_pair_distances
                else None
            ),
            "extended_distance_p90": (
                float(np.quantile(extended_pair_distances, 0.90))
                if extended_pair_distances
                else None
            ),
            "families": family_summaries,
            "manifest": str(workspace.cross_class_matching_manifest),
            "interpretation": (
                "B1/AU2/C2 use the expanded balanced cohort. The previous strict "
                "mutual-neighbour cohort is retained in selected_for_core_matched_cohort "
                "for audit and sensitivity checks."
            ),
        }
        FileManager.write_csv(
            workspace.cross_class_matching_manifest,
            manifest,
            CrossClassMatchingManager.MANIFEST_FIELDS,
        )
        FileManager.write_json(workspace.cross_class_matching_summary, summary)
        if not summary["target_achieved"]:
            print(
                "[MATCHING][WARNING] The requested expanded target could not be fully "
                "reached without violating one-to-one, anatomical, patient, or sequence caps."
            )
        print(
            "[MATCHING] Sick↔Normal core+extended: "
            f"eligible={len(eligible_rows)}, core_pairs={len(core_selected) // 2}, "
            f"total_pairs={len(selected_normal)}, images={len(selected)} "
            f"({100.0 * selected_fraction_dataset:.1f}% dataset), "
            f"patients Normal={len(normal_patients)}, Sick={len(sick_patients)} | "
            f"{workspace.cross_class_matching_manifest}"
        )
        return manifest

    @staticmethod
    def selected_tokens(
        manifest: Sequence[dict[str, Any]],
        core_only: bool = False,
    ) -> set[str]:
        field = (
            "selected_for_core_matched_cohort"
            if core_only
            else "selected_for_matched_cohort"
        )
        return {
            str(row.get("image_token", ""))
            for row in manifest
            if _as_int(row.get(field), 0) == 1
        }


class FeatureManager:
    """Create patient-level EfficientNet feature banks for all registered modes.

    ROI and complement construction stays on CPU. Only final three-channel images
    enter the frozen network, and historical .npz keys are retained for reuse.
    """


    @staticmethod
    def _region_normalize(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:


        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        visible = masks > 0.5
        batch_size = images.shape[0]
        bins = int(Settings.Image.REGION_HISTOGRAM_BINS)
        mask_flat = visible[:, 0].reshape(batch_size, -1)
        counts = mask_flat.sum(dim=1).long()
        values = images[:, 0].float().clamp(0.0, 1.0)
        indices = torch.round(values * float(bins - 1)).long().clamp_(0, bins - 1)
        indices = indices.reshape(batch_size, -1)
        histogram = torch.zeros(batch_size, bins, device=images.device, dtype=torch.float32)
        histogram.scatter_add_(1, indices, mask_flat.float())
        cumulative = torch.cumsum(histogram, dim=1)
        safe_counts = counts.clamp_min(1)
        lower_rank = (
            torch.floor(
                Settings.Image.REGION_LOWER_PERCENTILE / 100.0
                * (safe_counts - 1).float()
            ).long()
            + 1
        )
        upper_rank = (
            torch.floor(
                Settings.Image.REGION_UPPER_PERCENTILE / 100.0
                * (safe_counts - 1).float()
            ).long()
            + 1
        )
        lower_bin = (cumulative >= lower_rank[:, None].float()).long().argmax(dim=1)
        upper_bin = (cumulative >= upper_rank[:, None].float()).long().argmax(dim=1)
        lower = lower_bin.float().view(-1, 1, 1, 1) / float(bins - 1)
        upper = upper_bin.float().view(-1, 1, 1, 1) / float(bins - 1)
        valid = (
            (counts >= Settings.Image.REGION_MIN_PIXELS)
            & (upper[:, 0, 0, 0] > lower[:, 0, 0, 0])
        )
        scaled = (images.float() - lower) / (upper - lower).clamp_min(
            Settings.Image.REGION_MIN_DYNAMIC_RANGE
        )
        scaled = scaled.clamp(0.0, 1.0) * visible.float()
        return scaled * valid.view(-1, 1, 1, 1).float()

    @staticmethod
    def _support(mask: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        kernel = int(Settings.Segmentation.SUPPORT_DILATION_KERNEL)
        support = F.max_pool2d(
            (mask > 0.5).float(), kernel_size=kernel, stride=1, padding=kernel // 2
        )
        return (support > 0.5).float() * (content > 0.5).float()

    @staticmethod
    def _accepted_manual_tokens(
        workspace: Workspace,
        verbose: bool = False,
    ) -> set[str]:


        audit_rows = FileManager.read_csv(workspace.manual_audit)
        accepted: set[str] = set()
        explicit_positive = 0
        legacy_inferred = 0
        skipped_nonpositive = 0
        skipped_unusable = 0

        for row in audit_rows:
            if str(row.get("status", "")).strip().upper() != "ACCEPTED":
                continue
            token = str(row.get("image_token", "")).strip()
            if not token:
                continue
            target_type = str(row.get("target_type", "")).strip().upper()
            heart_present_raw = str(row.get("heart_present", "")).strip()

            if target_type in {MaskManager.NO_HEART_VISIBLE, MaskManager.UNUSABLE, "UNLABELED_EMPTY"}:
                skipped_nonpositive += 1
                continue

            path_value = str(row.get("manual_mask_path", "")).strip()
            path = Path(path_value) if path_value else workspace.manual_masks / f"{token}.png"

            if target_type == MaskManager.HEART_PRESENT or heart_present_raw == "1":
                qc = MaskManager.manual_qc(path)
                if qc.get("usable"):
                    accepted.add(token)
                    explicit_positive += 1
                else:
                    skipped_unusable += 1
                continue


            if not target_type and not heart_present_raw:
                qc = MaskManager.manual_qc(path)
                if qc.get("usable"):
                    accepted.add(token)
                    legacy_inferred += 1
                else:
                    skipped_unusable += 1
            else:
                skipped_nonpositive += 1

        if verbose:
            print(
                "[FEATURE BANK][MANUAL] "
                f"audit_rows={len(audit_rows)}, accepted_heart={len(accepted)}, "
                f"explicit={explicit_positive}, legacy_inferred={legacy_inferred}, "
                f"skipped_nonpositive={skipped_nonpositive}, "
                f"skipped_unusable={skipped_unusable}"
            )
        return accepted

    @staticmethod
    def _assert_same_cohort(
        bank: dict[str, dict[str, Any]],
        modes: Sequence[str],
        cohort_name: str,
    ) -> None:
        """Fail loudly if modes advertised as same-slice controls diverge.

        A shared patient list alone is not enough. The source-slice count and the
        number of contributing series proxies must also match, otherwise an AUC
        difference could partly reflect image selection rather than spatial masking.
        """

        modes = tuple(modes)
        reference_mode = modes[0]
        reference = bank[reference_mode]
        reference_patients = np.asarray(reference["patient_ids"]).astype(str)
        reference_labels = np.asarray(reference["y"], dtype=np.int64)
        reference_slices = int(reference["source_slices"])
        reference_series = int(reference["series_proxies"])

        for mode in modes[1:]:
            current = bank[mode]
            checks = {
                "patient_ids": np.array_equal(
                    reference_patients,
                    np.asarray(current["patient_ids"]).astype(str),
                ),
                "labels": np.array_equal(
                    reference_labels,
                    np.asarray(current["y"], dtype=np.int64),
                ),
                "source_slices": reference_slices == int(current["source_slices"]),
                "series_proxies": reference_series == int(current["series_proxies"]),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(
                    f"Same-slice cohort {cohort_name!r} is inconsistent between "
                    f"{reference_mode} and {mode}: {failed}. Rebuild the affected "
                    "feature-bank modes before interpreting their AUC difference."
                )

        print(
            f"[FEATURE BANK][COHORT] {cohort_name}: modes={list(modes)}, "
            f"patients={len(reference_patients)}, slices={reference_slices}, "
            f"series={reference_series}"
        )

    @staticmethod
    def _validate_cohort_alignment(bank: dict[str, dict[str, Any]]) -> None:
        FeatureManager._assert_same_cohort(
            bank,
            (
                "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
                "AU1_ATTENTION_ROI",
                "C1_ATTENTION_COMPLEMENT",
            ),
            "attention_eligible_same_slices",
        )
        FeatureManager._assert_same_cohort(
            bank,
            (
                "B1_MATCHED_FULL_IMAGE",
                "AU2_MATCHED_ATTENTION_ROI",
                "C2_MATCHED_ATTENTION_COMPLEMENT",
            ),
            "cross_class_matched_same_slices",
        )
        FeatureManager._assert_same_cohort(
            bank,
            (
                "M1_MANUAL_ROI",
                "C3_MANUAL_COMPLEMENT",
                "AU3_ATTENTION_ROI_MANUAL_SUBSET",
                "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET",
            ),
            "manual_positive_same_slices",
        )

    @staticmethod
    def _feature_fingerprint(
        rows: Sequence[dict[str, Any]], workspace: Workspace
    ) -> str:
        prediction_summary = FileManager.read_json(workspace.prediction_summary, {}) or {}
        digest = hashlib.sha256()
        payload = {
            "schema": "simple-patient-feature-bank-cross-class-v6-same-slice-baseline",
            "modes": Settings.Classification.LEGACY_STORAGE_MODES,
            "prediction_fingerprint": prediction_summary.get("fingerprint", ""),
            "support_dilation": Settings.Segmentation.SUPPORT_DILATION_KERNEL,
            "imagenet_weights": Settings.Classification.USE_IMAGENET_WEIGHTS,
            "classification_size": Settings.Image.CLASSIFICATION_SIZE,
            "quality_thresholds": {
                "dynamic_range": Settings.Image.QUALITY_MIN_DYNAMIC_RANGE,
                "laplacian_variance": Settings.Image.QUALITY_MIN_LAPLACIAN_VARIANCE,
                "noise_ratio": Settings.Image.QUALITY_MAX_NOISE_RATIO,
                "patient_blur_quantile": Settings.Image.QUALITY_PATIENT_BLUR_QUANTILE,
                "patient_noise_quantile": Settings.Image.QUALITY_PATIENT_NOISE_QUANTILE,
            },
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        digest.update(str(len(rows)).encode("ascii"))
        for row in rows:
            digest.update(str(row["image_token"]).encode("utf-8"))
        for audit_path in (
            workspace.quality_audit,
            workspace.manual_audit,
            workspace.manual_annotations,
            workspace.prediction_audit,
            workspace.cross_class_matching_manifest,
        ):
            if audit_path.is_file():
                digest.update(audit_path.name.encode("utf-8"))
                digest.update(FileManager.sha256_file(audit_path).encode("ascii"))
        accepted_tokens = FeatureManager._accepted_manual_tokens(
            workspace, verbose=False
        )
        for token in sorted(accepted_tokens):
            path = workspace.manual_masks / f"{token}.png"
            if path.is_file():
                digest.update(token.encode("utf-8"))
                digest.update(FileManager.sha256_file(path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _load_modes(
        workspace: Workspace,
        modes: Sequence[str],
        allow_missing: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Load selected modes while accepting the pre-B2 storage schema.

        ``allow_missing=True`` is used only for the controlled cache migration that
        adds B2. A partially written mode still raises an error; only a completely
        absent mode may be skipped.
        """

        if not workspace.feature_bank.is_file():
            raise FileNotFoundError(f"Missing feature bank: {workspace.feature_bank}")

        bank: dict[str, dict[str, Any]] = {}
        with np.load(workspace.feature_bank, allow_pickle=False) as archive:
            available = set(archive.files)
            for mode in modes:
                legacy_mode = Settings.Classification.LEGACY_MODE_ALIASES[mode]
                candidates = tuple(dict.fromkeys((mode, legacy_mode)))
                storage_mode = None
                partial_candidates: list[str] = []
                for candidate in candidates:
                    required = {
                        f"{candidate}__X",
                        f"{candidate}__y",
                        f"{candidate}__patient_ids",
                        f"{candidate}__source_slices",
                        f"{candidate}__series_proxies",
                    }
                    present = required & available
                    if required.issubset(available):
                        storage_mode = candidate
                        break
                    if present:
                        partial_candidates.append(
                            f"{candidate}: missing={sorted(required - available)}"
                        )

                if storage_mode is None:
                    if partial_candidates:
                        raise KeyError(
                            f"Feature bank contains an incomplete entry for {mode}: "
                            + "; ".join(partial_candidates)
                        )
                    if allow_missing:
                        continue
                    raise KeyError(
                        f"Feature bank is missing all storage keys for {mode}. "
                        f"Tried {list(candidates)}."
                    )

                bank[mode] = {
                    "X": archive[f"{storage_mode}__X"],
                    "y": archive[f"{storage_mode}__y"],
                    "patient_ids": archive[f"{storage_mode}__patient_ids"].astype(str),
                    "source_slices": int(
                        archive[f"{storage_mode}__source_slices"][0]
                    ),
                    "series_proxies": int(
                        archive[f"{storage_mode}__series_proxies"][0]
                    ),
                }
        return bank

    @staticmethod
    def load(workspace: Workspace) -> dict[str, dict[str, Any]]:
        return FeatureManager._load_modes(
            workspace,
            Settings.Classification.MODES,
            allow_missing=False,
        )

    @staticmethod
    def _reusable_bank_for_incremental_rebuild(
        workspace: Workspace,
    ) -> dict[str, dict[str, Any]] | None:
        """Reuse old modes for two explicitly safe incremental updates.

        Safe update 1 adds the new B2 same-slice full-image control while leaving
        every existing representation unchanged. Safe update 2 rebuilds only the
        three matched modes after the matching manifest changes. Any newer quality,
        prediction, manual-audit, annotation, or manual-mask input disables reuse.
        """

        if not workspace.feature_bank.is_file() or not workspace.feature_metadata.is_file():
            return None
        try:
            bank_mtime = workspace.feature_bank.stat().st_mtime_ns
        except OSError:
            return None

        static_inputs = (
            workspace.quality_audit,
            workspace.manual_audit,
            workspace.manual_annotations,
            workspace.prediction_audit,
        )
        for path in static_inputs:
            if path.is_file() and path.stat().st_mtime_ns > bank_mtime:
                return None
        try:
            newest_manual = max(
                (path.stat().st_mtime_ns for path in workspace.manual_masks.glob("*.png")),
                default=0,
            )
        except OSError:
            return None
        if newest_manual > bank_mtime:
            return None

        try:
            previous = FeatureManager._load_modes(
                workspace,
                Settings.Classification.MODES,
                allow_missing=True,
            )
        except Exception:
            return None

        b2_mode = "B2_ATTENTION_ELIGIBLE_FULL_IMAGE"
        required_pre_b2_modes = set(Settings.Classification.MODES) - {b2_mode}
        if not required_pre_b2_modes.issubset(previous):
            return None

        missing_b2 = b2_mode not in previous
        matching_changed = bool(
            workspace.cross_class_matching_manifest.is_file()
            and workspace.cross_class_matching_manifest.stat().st_mtime_ns > bank_mtime
        )
        if not missing_b2 and not matching_changed:
            # A different fingerprint cause may represent changed preprocessing or
            # settings; do not silently reuse in that case.
            return None

        reusable = dict(previous)
        if matching_changed:
            for mode in Settings.Classification.MATCHED_MODES:
                reusable.pop(mode, None)

        missing_modes = [
            mode for mode in Settings.Classification.MODES if mode not in reusable
        ]
        allowed_missing = {b2_mode}
        if matching_changed:
            allowed_missing.update(Settings.Classification.MATCHED_MODES)
        if not set(missing_modes).issubset(allowed_missing):
            return None

        print(
            "[FEATURE BANK] Safe incremental reuse: "
            f"reused={sorted(reusable)}, recomputing={missing_modes}."
        )
        return reusable

    @staticmethod
    def load_compatible_cache(
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        fingerprint: str | None = None,
    ) -> dict[str, dict[str, Any]] | None:


        fingerprint = fingerprint or FeatureManager._feature_fingerprint(
            dataset_rows, workspace
        )
        metadata = FileManager.read_json(workspace.feature_metadata, {}) or {}
        if (
            not workspace.feature_bank.is_file()
            or metadata.get("fingerprint") != fingerprint
        ):
            return None
        try:
            return FeatureManager.load(workspace)
        except Exception as error:
            print(
                "[FEATURE BANK] Cache cannot be read and will be rebuilt:",
                f"{type(error).__name__}: {error}",
            )
            return None

    # Build every experiment from a frozen EfficientNet pass. B2, AU1, and C1
    # are forced to share the exact same Attention-eligible rows. During a full
    # rebuild, B0 embeddings are reused for the B2 and B1 subsets, so the added
    # methodological control does not create duplicate full-image forward passes.
    @staticmethod
    def build(
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        device: torch.device,
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:

        fingerprint = FeatureManager._feature_fingerprint(dataset_rows, workspace)
        reusable_bank: dict[str, dict[str, Any]] | None = None
        if not force:
            bank = FeatureManager.load_compatible_cache(
                dataset_rows, workspace, fingerprint=fingerprint
            )
            if bank is not None:
                FeatureManager._validate_cohort_alignment(bank)
                print(
                    "[FEATURE BANK] Compatible cache reused; "
                    "the extractor is not loaded on the GPU."
                )
                return bank
            reusable_bank = FeatureManager._reusable_bank_for_incremental_rebuild(
                workspace
            )

        quality = {
            row["image_token"]: row
            for row in FileManager.read_csv(workspace.quality_audit)
        }
        predictions = {
            row["image_token"]: row
            for row in FileManager.read_csv(workspace.prediction_audit)
        }
        accepted_manual = FeatureManager._accepted_manual_tokens(
            workspace, verbose=True
        )
        if not accepted_manual:
            raise RuntimeError(
                "No usable manual HEART_PRESENT mask is available for "
                "M1/C3/AU3/C4. The audit was checked or migrated, but no "
                "accepted non-empty PNG remained. Inspect simple_manual_mask_audit.csv "
                "and manual_masks/. Baseline and automatic-Attention modes are not the cause of this error."
            )
        matching_manifest = FileManager.read_csv(
            workspace.cross_class_matching_manifest
        )
        matched_tokens = CrossClassMatchingManager.selected_tokens(
            matching_manifest
        )
        if not matched_tokens:
            raise RuntimeError(
                "The cross-class matching manifest is missing or contains no pairs. "
                "Run pipeline.build_cross_class_matching()."
            )
        if len(predictions) != len(dataset_rows):
            raise RuntimeError("The prediction audit does not cover the full dataset.")

        rows: list[dict[str, Any]] = []
        for original in dataset_rows:
            row = dict(original)
            row.update(predictions.get(row["image_token"], {}))
            row.update(quality.get(row["image_token"], {}))
            row["keep_attention"] = int(
                _as_int(row.get("attention_valid_final"), 0) == 1
                and _as_int(row.get("attention_heart_present"), 1) == 1
                and _as_int(row.get("quality_valid"), 0) == 1
                and _as_float(row.get("attention_area_ratio"), 0.0)
                >= Settings.Segmentation.PREDICTION_MIN_AREA_RATIO
            )
            row["keep_manual_matched"] = int(
                row["image_token"] in accepted_manual
            )
            row["keep_cross_class_matched"] = int(
                row["image_token"] in matched_tokens and row["keep_attention"] == 1
            )
            rows.append(row)

        reusable_bank = dict(reusable_bank or {})
        modes_to_compute = tuple(
            mode
            for mode in Settings.Classification.MODES
            if mode not in reusable_bank
        )
        if not modes_to_compute:
            raise RuntimeError(
                "The feature-bank fingerprint changed, but no safe mode was selected "
                "for recomputation. Use force=True for a full rebuild."
            )
        compute_set = set(modes_to_compute)

        requires_all_rows = "B0_FULL_IMAGE" in compute_set
        requires_attention_rows = bool(
            compute_set & Settings.Classification.ATTENTION_ELIGIBLE_MODES
        )
        requires_matched_rows = bool(
            compute_set & Settings.Classification.MATCHED_MODES
        )
        requires_manual_rows = bool(
            compute_set & Settings.Classification.MANUAL_SUBSET_MODES
        )

        if not requires_all_rows:
            rows = [
                row
                for row in rows
                if (
                    (requires_attention_rows and row["keep_attention"] == 1)
                    or (requires_matched_rows and row["keep_cross_class_matched"] == 1)
                    or (requires_manual_rows and row["keep_manual_matched"] == 1)
                )
            ]
        if not rows:
            raise RuntimeError(
                f"No images are eligible for feature-bank modes {list(modes_to_compute)}."
            )

        rebuild_kind = "incremental" if reusable_bank else "full"
        print(
            f"[FEATURE BANK] {rebuild_kind.capitalize()} rebuild: "
            f"input_images={len(rows)}, recomputing={list(modes_to_compute)}, "
            f"reused={sorted(reusable_bank)}"
        )

        extractor = RuntimeManager.prepare_model(FrozenEfficientNet(), device).eval()
        loader = DataLoader(
            FeatureDataset(rows),
            batch_size=RuntimeManager.feature_batch_size(device),
            shuffle=False,
            num_workers=Settings.Runtime.NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        pool = StreamingPatientPool(modes_to_compute)
        started = time.perf_counter()

        # Aliases receive the very same embeddings from a strict subset of the
        # source rows. This avoids duplicate forward passes and guarantees that the
        # same-slice controls differ only by their image representation.
        alias_rules: dict[str, tuple[tuple[str, str], ...]] = {
            "B0_FULL_IMAGE": (
                ("B2_ATTENTION_ELIGIBLE_FULL_IMAGE", "keep_attention"),
                ("B1_MATCHED_FULL_IMAGE", "keep_cross_class_matched"),
            ),
            "B2_ATTENTION_ELIGIBLE_FULL_IMAGE": (
                ("B1_MATCHED_FULL_IMAGE", "keep_cross_class_matched"),
            ),
            "AU1_ATTENTION_ROI": (
                ("AU2_MATCHED_ATTENTION_ROI", "keep_cross_class_matched"),
            ),
            "C1_ATTENTION_COMPLEMENT": (
                ("C2_MATCHED_ATTENTION_COMPLEMENT", "keep_cross_class_matched"),
            ),
        }

        def encode_groups(
            groups: list[tuple[str, torch.Tensor, list[dict[str, Any]]]],
        ) -> None:
            valid_groups = [
                (mode, images, selected_rows)
                for mode, images, selected_rows in groups
                if mode in compute_set
                and len(selected_rows) > 0
                and images.shape[0] > 0
            ]
            if not valid_groups:
                return

            combined = torch.cat(
                [images.contiguous() for _, images, _ in valid_groups], dim=0
            )
            forward_batch = RuntimeManager.feature_forward_batch_size(device)
            embedding_chunks: list[torch.Tensor] = []
            for begin in range(0, combined.shape[0], forward_batch):
                images_device = RuntimeManager.move_tensor(
                    combined[begin : begin + forward_batch], device
                )
                with RuntimeManager.autocast(device):
                    output = extractor(images_device)
                embedding_chunks.append(output.float().cpu())
                del images_device, output

            embeddings = torch.cat(embedding_chunks, dim=0).numpy().astype(np.float32)
            cursor = 0
            for mode, images, selected_rows in valid_groups:
                count = int(images.shape[0])
                current_embeddings = embeddings[cursor : cursor + count]
                pool.add(mode, current_embeddings, selected_rows)

                for alias_mode, selector_field in alias_rules.get(mode, ()):
                    if alias_mode not in compute_set:
                        continue
                    alias_positions = [
                        position
                        for position, row in enumerate(selected_rows)
                        if _as_int(row.get(selector_field), 0) == 1
                    ]
                    if alias_positions:
                        index_array = np.asarray(alias_positions, dtype=np.int64)
                        pool.add(
                            alias_mode,
                            current_embeddings[index_array],
                            [selected_rows[position] for position in alias_positions],
                        )
                cursor += count
            del combined, embeddings, embedding_chunks

        with torch.inference_mode():
            for robust, raw, content, attention, manual, indices in tqdm(
                loader, desc=f"EfficientNet feature bank ({device.type})"
            ):
                batch_rows = [rows[int(index)] for index in indices]
                attention_positions = [
                    position
                    for position, row in enumerate(batch_rows)
                    if row["keep_attention"] == 1
                ]
                matched_positions = [
                    position
                    for position, row in enumerate(batch_rows)
                    if row["keep_cross_class_matched"] == 1
                ]
                manual_positions = [
                    position
                    for position, row in enumerate(batch_rows)
                    if row["keep_manual_matched"] == 1
                ]

                groups: list[
                    tuple[str, torch.Tensor, list[dict[str, Any]]]
                ] = []

                # One full-image pass supplies B0 and, through exact row subsets,
                # B2 and B1. During an incremental migration, B2 alone is computed
                # on the Attention-eligible rows while all previous modes are reused.
                if "B0_FULL_IMAGE" in compute_set:
                    groups.append(("B0_FULL_IMAGE", robust, batch_rows))
                elif (
                    "B2_ATTENTION_ELIGIBLE_FULL_IMAGE" in compute_set
                    and attention_positions
                ):
                    positions = torch.as_tensor(attention_positions, dtype=torch.long)
                    groups.append(
                        (
                            "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
                            robust.index_select(0, positions),
                            [batch_rows[position] for position in attention_positions],
                        )
                    )
                elif "B1_MATCHED_FULL_IMAGE" in compute_set and matched_positions:
                    positions = torch.as_tensor(matched_positions, dtype=torch.long)
                    groups.append(
                        (
                            "B1_MATCHED_FULL_IMAGE",
                            robust.index_select(0, positions),
                            [batch_rows[position] for position in matched_positions],
                        )
                    )

                needs_automatic_regions = bool(
                    compute_set
                    & {
                        "AU1_ATTENTION_ROI",
                        "C1_ATTENTION_COMPLEMENT",
                        "AU2_MATCHED_ATTENTION_ROI",
                        "C2_MATCHED_ATTENTION_COMPLEMENT",
                    }
                )
                if needs_automatic_regions:
                    use_attention_superset = bool(
                        compute_set
                        & {"AU1_ATTENTION_ROI", "C1_ATTENTION_COMPLEMENT"}
                    )
                    automatic_positions = (
                        attention_positions if use_attention_superset else matched_positions
                    )
                    if automatic_positions:
                        positions = torch.as_tensor(
                            automatic_positions, dtype=torch.long
                        )
                        selected_raw = raw.index_select(0, positions)
                        selected_content = content.index_select(0, positions)
                        selected_mask = attention.index_select(0, positions)
                        support = FeatureManager._support(
                            selected_mask, selected_content
                        )
                        selected_rows = [
                            batch_rows[position] for position in automatic_positions
                        ]

                        if "AU1_ATTENTION_ROI" in compute_set:
                            roi_mode = "AU1_ATTENTION_ROI"
                        elif "AU2_MATCHED_ATTENTION_ROI" in compute_set:
                            roi_mode = "AU2_MATCHED_ATTENTION_ROI"
                        else:
                            roi_mode = ""
                        if roi_mode:
                            groups.append(
                                (
                                    roi_mode,
                                    FeatureManager._region_normalize(
                                        selected_raw, support
                                    ),
                                    selected_rows,
                                )
                            )

                        if "C1_ATTENTION_COMPLEMENT" in compute_set:
                            complement_mode = "C1_ATTENTION_COMPLEMENT"
                        elif "C2_MATCHED_ATTENTION_COMPLEMENT" in compute_set:
                            complement_mode = "C2_MATCHED_ATTENTION_COMPLEMENT"
                        else:
                            complement_mode = ""
                        if complement_mode:
                            groups.append(
                                (
                                    complement_mode,
                                    FeatureManager._region_normalize(
                                        selected_raw,
                                        (selected_content > 0.5).float()
                                        * (1.0 - support),
                                    ),
                                    selected_rows,
                                )
                            )

                if (
                    compute_set & Settings.Classification.MANUAL_SUBSET_MODES
                    and manual_positions
                ):
                    positions = torch.as_tensor(manual_positions, dtype=torch.long)
                    selected_raw = raw.index_select(0, positions)
                    selected_content = content.index_select(0, positions)
                    manual_mask = manual.index_select(0, positions)
                    attention_mask = attention.index_select(0, positions)
                    manual_support = FeatureManager._support(
                        manual_mask, selected_content
                    )
                    attention_support = FeatureManager._support(
                        attention_mask, selected_content
                    )
                    selected_rows = [
                        batch_rows[position] for position in manual_positions
                    ]
                    visible_content = (selected_content > 0.5).float()
                    manual_groups = (
                        (
                            "M1_MANUAL_ROI",
                            FeatureManager._region_normalize(
                                selected_raw, manual_support
                            ),
                        ),
                        (
                            "C3_MANUAL_COMPLEMENT",
                            FeatureManager._region_normalize(
                                selected_raw,
                                visible_content * (1.0 - manual_support),
                            ),
                        ),
                        (
                            "AU3_ATTENTION_ROI_MANUAL_SUBSET",
                            FeatureManager._region_normalize(
                                selected_raw, attention_support
                            ),
                        ),
                        (
                            "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET",
                            FeatureManager._region_normalize(
                                selected_raw,
                                visible_content * (1.0 - attention_support),
                            ),
                        ),
                    )
                    groups.extend(
                        (mode, images, selected_rows)
                        for mode, images in manual_groups
                        if mode in compute_set
                    )

                encode_groups(groups)
                del groups

        computed_bank = {
            mode: pool.finalize(mode)
            for mode in modes_to_compute
        }
        bank = dict(reusable_bank)
        bank.update(computed_bank)
        missing_modes = [
            mode for mode in Settings.Classification.MODES if mode not in bank
        ]
        if missing_modes:
            raise RuntimeError(
                f"Feature-bank rebuild is missing modes: {missing_modes}"
            )

        FeatureManager._validate_cohort_alignment(bank)

        arrays: dict[str, np.ndarray] = {}
        metadata_modes: dict[str, dict[str, Any]] = {}
        for mode in Settings.Classification.MODES:
            values = bank[mode]
            storage_mode = Settings.Classification.LEGACY_MODE_ALIASES[mode]
            arrays[f"{storage_mode}__X"] = values["X"]
            arrays[f"{storage_mode}__y"] = values["y"]
            arrays[f"{storage_mode}__patient_ids"] = values["patient_ids"].astype("U")
            arrays[f"{storage_mode}__source_slices"] = np.asarray(
                [values["source_slices"]], dtype=np.int64
            )
            arrays[f"{storage_mode}__series_proxies"] = np.asarray(
                [values["series_proxies"]], dtype=np.int64
            )
            metadata_modes[mode] = {
                "description": Settings.Classification.MODE_DESCRIPTIONS[mode],
                "cohort": Settings.Classification.MODE_COHORTS[mode],
                "storage_key": storage_mode,
                "patients": int(len(values["patient_ids"])),
                "source_slices": int(values["source_slices"]),
                "series_proxies": int(values["series_proxies"]),
            }
        workspace.feature_bank.parent.mkdir(parents=True, exist_ok=True)
        temporary = workspace.feature_bank.with_name(
            f".{workspace.feature_bank.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
        )
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, workspace.feature_bank)
        metadata = {
            "fingerprint": fingerprint,
            "schema": "simple-patient-feature-bank-cross-class-v6-same-slice-baseline",
            "preprocessing_device": "cpu",
            "extractor_device": device.type,
            "modes": metadata_modes,
            "reused_modes": sorted(reusable_bank),
            # Retained for readers expecting the previous metadata field.
            "reused_static_modes": sorted(
                mode
                for mode in reusable_bank
                if mode not in Settings.Classification.MATCHED_MODES
            ),
            "recomputed_modes": sorted(computed_bank),
            "incremental_rebuild": bool(reusable_bank),
            "matched_only_rebuild": (
                set(computed_bank) == set(Settings.Classification.MATCHED_MODES)
            ),
            "same_slice_controls": {
                "attention_eligible": [
                    "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
                    "AU1_ATTENTION_ROI",
                    "C1_ATTENTION_COMPLEMENT",
                ],
                "cross_class_matched": [
                    "B1_MATCHED_FULL_IMAGE",
                    "AU2_MATCHED_ATTENTION_ROI",
                    "C2_MATCHED_ATTENTION_COMPLEMENT",
                ],
            },
            "elapsed": RuntimeManager.format_seconds(
                time.perf_counter() - started
            ),
            "feature_bank": str(workspace.feature_bank),
            "cross_class_matching_manifest": str(
                workspace.cross_class_matching_manifest
            ),
            "cross_class_matching_summary": str(
                workspace.cross_class_matching_summary
            ),
        }
        FileManager.write_json(workspace.feature_metadata, metadata)
        print("[FEATURE BANK] Saved:", workspace.feature_bank)
        del extractor, loader
        RuntimeManager.release(device)
        return bank


# -----------------------------------------------------------------------------
# Nested patient-level evaluation and paired sensitivity comparisons
# -----------------------------------------------------------------------------
class EvaluationManager:
    """Run nested patient-level cross-validation with train-only transforms.

    Scaling, PCA, regularization selection, and decision-threshold selection are
    fitted inside training data. Final scores are strictly out of fold.
    """


    @staticmethod
    def _pipeline(c_value: float) -> SklearnPipeline:
        return SklearnPipeline(
            [
                ("scale", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=Settings.Classification.PCA_EXPLAINED_VARIANCE,
                        svd_solver="full",
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(c_value),
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=Settings.Runtime.RANDOM_SEED,
                    ),
                ),
            ]
        )

    @staticmethod
    def _youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        finite = np.isfinite(thresholds)
        if not np.any(finite):
            return 0.5
        index = np.argmax((tpr - fpr)[finite])
        return float(thresholds[finite][index])

    @staticmethod
    def _select_c_and_threshold(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
        class_counts = np.bincount(y, minlength=2)
        n_splits = min(Settings.Classification.INNER_FOLDS, int(class_counts.min()))
        if n_splits < 2:
            return 1.0, 0.5
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best_c, best_auc, best_scores = 1.0, -np.inf, None
        for c_value in Settings.Classification.C_GRID:
            scores = np.full(len(y), np.nan, dtype=np.float64)
            for train_index, valid_index in splitter.split(X, y):
                model = EvaluationManager._pipeline(c_value)
                model.fit(X[train_index], y[train_index])
                scores[valid_index] = model.predict_proba(X[valid_index])[:, 1]
            auc = roc_auc_score(y, scores)
            if (auc, -abs(math.log10(c_value))) > (best_auc, -abs(math.log10(best_c))):
                best_c, best_auc, best_scores = float(c_value), float(auc), scores.copy()
        threshold = EvaluationManager._youden_threshold(y, best_scores)
        return best_c, threshold

    @staticmethod
    def _fold_map(reference: dict[str, Any]) -> dict[str, int]:
        patients = np.asarray(reference["patient_ids"]).astype(str)
        labels = np.asarray(reference["y"], dtype=np.int64)
        class_counts = np.bincount(labels, minlength=2)
        n_splits = min(Settings.Classification.OUTER_FOLDS, int(class_counts.min()))
        if n_splits < 2:
            raise RuntimeError("At least two patients are required in each class.")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=Settings.Runtime.RANDOM_SEED,
        )
        mapping = {}
        for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
            for index in valid_index:
                mapping[patients[index]] = fold
        return mapping

    @staticmethod
    def _auc_ci(labels: np.ndarray, scores: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(int(repeats)):
            indices = rng.integers(0, len(labels), size=len(labels))
            if len(np.unique(labels[indices])) < 2:
                continue
            values.append(roc_auc_score(labels[indices], scores[indices]))
        if not values:
            return np.nan, np.nan
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    @staticmethod
    def _metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        ci_low, ci_high = EvaluationManager._auc_ci(
            labels,
            scores,
            Settings.Classification.BOOTSTRAP_REPEATS,
            Settings.Runtime.RANDOM_SEED + 9000,
        )
        return {
            "auc": float(roc_auc_score(labels, scores)),
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            "average_precision": float(average_precision_score(labels, scores)),
            "brier": float(brier_score_loss(labels, scores)),
            "accuracy": float(np.mean(predictions == labels)),
            "sensitivity": float(tp / max(1, tp + fn)),
            "specificity": float(tn / max(1, tn + fp)),
        }

    @staticmethod
    def _paired_auc_difference(
        first: pd.DataFrame,
        second: pd.DataFrame,
        repeats: int,
        seed: int,
    ) -> dict[str, Any]:
        merged = first.merge(second, on=["patient_id", "true_label"], suffixes=("_first", "_second"))
        labels = merged["true_label"].to_numpy(dtype=np.int64)
        first_scores = merged["score_first"].to_numpy(dtype=np.float64)
        second_scores = merged["score_second"].to_numpy(dtype=np.float64)
        if len(merged) < 4 or len(np.unique(labels)) < 2:
            return {
                "n_patients": len(merged),
                "auc_difference_first_minus_second": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        observed = float(roc_auc_score(labels, first_scores) - roc_auc_score(labels, second_scores))
        rng = np.random.default_rng(seed)
        differences = []
        for _ in range(int(repeats)):
            indices = rng.integers(0, len(labels), size=len(labels))
            if len(np.unique(labels[indices])) < 2:
                continue
            differences.append(
                roc_auc_score(labels[indices], first_scores[indices])
                - roc_auc_score(labels[indices], second_scores[indices])
            )
        return {
            "n_patients": len(merged),
            "auc_difference_first_minus_second": observed,
            "ci_low": float(np.quantile(differences, 0.025)) if differences else np.nan,
            "ci_high": float(np.quantile(differences, 0.975)) if differences else np.nan,
        }

    # All hyperparameters and probability thresholds are selected inside the outer
    # training fold. The held-out patients are used once, only for OOF scoring.
    @staticmethod
    def evaluate(bank: dict[str, dict[str, Any]], workspace: Workspace) -> pd.DataFrame:
        fold_map = EvaluationManager._fold_map(bank["B0_FULL_IMAGE"])
        summary_rows = []
        prediction_tables: dict[str, pd.DataFrame] = {}

        for mode in Settings.Classification.MODES:
            values = bank[mode]
            X = np.asarray(values["X"], dtype=np.float32)
            y = np.asarray(values["y"], dtype=np.int64)
            patients = np.asarray(values["patient_ids"]).astype(str)
            if len(np.unique(y)) < 2:
                raise RuntimeError(f"{mode}: one class is missing.")
            unknown = sorted(set(patients) - set(fold_map))
            if unknown:
                raise RuntimeError(f"{mode}: patients missing from the fold map: {unknown}")
            patient_folds = np.asarray([fold_map[patient] for patient in patients], dtype=np.int64)
            scores = np.full(len(y), np.nan, dtype=np.float64)
            predictions = np.full(len(y), -1, dtype=np.int64)
            selected_cs = np.full(len(y), np.nan, dtype=np.float64)
            thresholds = np.full(len(y), np.nan, dtype=np.float64)

            for fold in sorted(np.unique(patient_folds)):
                train_index = np.flatnonzero(patient_folds != fold)
                valid_index = np.flatnonzero(patient_folds == fold)
                if len(np.unique(y[train_index])) < 2:
                    raise RuntimeError(
                        f"{mode}: fold {fold} does not contain both classes in training. "
                        "The matched cohort is too sparse; inspect the matching summary."
                    )
                best_c, threshold = EvaluationManager._select_c_and_threshold(
                    X[train_index], y[train_index], Settings.Runtime.RANDOM_SEED + int(fold)
                )
                model = EvaluationManager._pipeline(best_c)
                model.fit(X[train_index], y[train_index])
                fold_scores = model.predict_proba(X[valid_index])[:, 1]
                scores[valid_index] = fold_scores
                predictions[valid_index] = (fold_scores >= threshold).astype(np.int64)
                selected_cs[valid_index] = best_c
                thresholds[valid_index] = threshold

            if not np.all(np.isfinite(scores)) or np.any(predictions < 0):
                raise RuntimeError(f"{mode}: incomplete OOF predictions.")
            metrics = EvaluationManager._metrics(y, scores, predictions)
            table = pd.DataFrame(
                {
                    "patient_id": patients,
                    "true_label": y,
                    "outer_fold": patient_folds,
                    "score": scores,
                    "predicted_label": predictions,
                    "selected_c": selected_cs,
                    "training_threshold": thresholds,
                }
            ).sort_values("patient_id")
            table.to_csv(workspace.predictions_dir / f"{mode}.csv", index=False)
            prediction_tables[mode] = table
            summary_rows.append(
                {
                    "mode": mode,
                    "experiment_group": (
                        "primary"
                        if mode in Settings.Classification.PRIMARY_MODES
                        else "mask_validation"
                    ),
                    "description": Settings.Classification.MODE_DESCRIPTIONS[mode],
                    "cohort": Settings.Classification.MODE_COHORTS[mode],
                    "patients": len(patients),
                    "source_slices": values["source_slices"],
                    "series_proxies": values["series_proxies"],
                    **metrics,
                }
            )
            print(
                f"[EVALUATION] {mode}: AUC={metrics['auc']:.3f} "
                f"AP={metrics['average_precision']:.3f} Brier={metrics['brier']:.3f}"
            )

        summary = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
        summary.to_csv(workspace.results_csv, index=False)

        comparisons = []
        for first_mode, second_mode, question in (
            (
                "AU1_ATTENTION_ROI",
                "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
                "same_attention_eligible_slices_roi_vs_full_image",
            ),
            (
                "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
                "C1_ATTENTION_COMPLEMENT",
                "same_attention_eligible_slices_full_image_vs_complement",
            ),
            (
                "AU1_ATTENTION_ROI",
                "C1_ATTENTION_COMPLEMENT",
                "same_attention_eligible_slices_roi_vs_complement",
            ),
            (
                "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
                "B0_FULL_IMAGE",
                "attention_eligible_selection_effect_on_full_image",
            ),
            (
                "AU2_MATCHED_ATTENTION_ROI",
                "B1_MATCHED_FULL_IMAGE",
                "cross_class_matched_roi_vs_full_image",
            ),
            (
                "AU2_MATCHED_ATTENTION_ROI",
                "C2_MATCHED_ATTENTION_COMPLEMENT",
                "cross_class_matched_roi_vs_complement",
            ),
            (
                "B1_MATCHED_FULL_IMAGE",
                "C2_MATCHED_ATTENTION_COMPLEMENT",
                "cross_class_matched_full_image_vs_complement",
            ),
            (
                "B0_FULL_IMAGE",
                "B1_MATCHED_FULL_IMAGE",
                "all_slices_vs_cross_class_matched_full_image",
            ),
            (
                "AU1_ATTENTION_ROI",
                "AU2_MATCHED_ATTENTION_ROI",
                "attention_eligible_vs_cross_class_matched_roi",
            ),
            (
                "C1_ATTENTION_COMPLEMENT",
                "C2_MATCHED_ATTENTION_COMPLEMENT",
                "attention_eligible_vs_cross_class_matched_complement",
            ),
            (
                "M1_MANUAL_ROI",
                "AU3_ATTENTION_ROI_MANUAL_SUBSET",
                "manual_vs_attention_roi_same_images",
            ),
            (
                "C3_MANUAL_COMPLEMENT",
                "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET",
                "manual_vs_attention_complement_same_images",
            ),
        ):
            comparison = EvaluationManager._paired_auc_difference(
                prediction_tables[first_mode],
                prediction_tables[second_mode],
                Settings.Classification.BOOTSTRAP_REPEATS,
                Settings.Runtime.RANDOM_SEED + len(comparisons) * 100,
            )
            comparisons.append(
                {
                    "comparison": question,
                    "first_mode": first_mode,
                    "second_mode": second_mode,
                    **comparison,
                }
            )
        pd.DataFrame(comparisons).to_csv(
            workspace.outputs / "paired_auc_comparisons.csv", index=False
        )
        print("[EVALUATION] Results:", workspace.results_csv)
        return summary


# -----------------------------------------------------------------------------
# Public pipeline interface and Kaggle execution stages
# -----------------------------------------------------------------------------
class CADPipeline:
    """Small public interface over the persisted CPU/GPU research stages.

    The object stores paths and configuration, not a live device. Every expensive
    stage checks its fingerprinted cache before CUDA or EfficientNet is initialized.
    """


    def __init__(
        self,
        dataset_path: Path | str | None = None,
        workspace_root: Path | str | None = None,
        attention_device: str | None = None,
        feature_device: str | None = None,
    ):
        RuntimeManager.seed_everything(include_cuda=False)
        self.dataset_path = Path(dataset_path or Settings.Paths.DATASET_PATH)
        self.workspace = FileManager.create_workspace(workspace_root)
        self.attention_device = (
            attention_device or Settings.Runtime.ATTENTION_DEVICE
        ).strip().lower()
        self.feature_device = (
            feature_device or Settings.Runtime.FEATURE_DEVICE
        ).strip().lower()
        for name, value in (
            ("attention_device", self.attention_device),
            ("feature_device", self.feature_device),
        ):
            if value not in {"auto", "cpu", "cuda"}:
                raise ValueError(
                    f"{name} must be 'auto', 'cpu', or 'cuda'."
                )

        self.samples: list[Sample] | None = None
        self.dataset_rows: list[dict[str, Any]] | None = None
        print("[PIPELINE] orchestration and non-neural stages: cpu")
        print(f"[PIPELINE] Attention U-Net configured for: {self.attention_device}")
        print(f"[PIPELINE] EfficientNet configured for: {self.feature_device}")
        print(f"[PIPELINE] dataset={self.dataset_path}")
        print(f"[PIPELINE] workspace={self.workspace.root}")


    def prepare(self) -> list[dict[str, Any]]:


        self.samples = DatasetManager.discover(self.dataset_path, self.workspace)
        self.dataset_rows = DatasetManager.rows(self.samples, self.workspace)
        return self.dataset_rows

    def _rows(self) -> list[dict[str, Any]]:
        if self.dataset_rows is None:
            return self.prepare()
        return self.dataset_rows

    def audit_quality(self, refresh: bool = False) -> dict[str, dict[str, Any]]:


        return QualityManager.build(self._rows(), self.workspace, refresh=refresh)

    def audit_manual_masks(
        self,
        refresh_quality: bool = False,
        minimum_masks: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:


        quality = self.audit_quality(refresh=refresh_quality)
        return MaskManager.audit_manual_masks(
            self._rows(), quality, self.workspace, minimum_masks=minimum_masks
        )


    def train_attention(
        self,
        force: bool = False,
        minimum_masks: int | None = None,
        refresh_quality: bool = False,
        device: str | None = None,
    ) -> dict[int, Path]:


        accepted, manual_summary = self.audit_manual_masks(
            refresh_quality=refresh_quality,
            minimum_masks=minimum_masks,
        )
        if not force:
            cached_checkpoints = SegmentationManager.compatible_checkpoint_map(
                accepted, self.workspace
            )
            if cached_checkpoints is not None:
                print(
                    "[ATTENTION] All checkpoints are compatible; "
                    "CUDA is not initialized."
                )
                summary = FileManager.read_json(
                    self.workspace.training_summary, {}
                ) or {}
                summary["manual_mask_audit"] = manual_summary
                summary["last_call"] = "reused_before_device_initialization"
                FileManager.write_json(self.workspace.training_summary, summary)
                return cached_checkpoints

        requested = device or self.attention_device
        with RuntimeManager.device_scope(
            requested, "Attention U-Net training"
        ) as compute_device:
            checkpoints = SegmentationManager.train_crossfit(
                accepted, self.workspace, compute_device, force=force
            )

        summary = FileManager.read_json(self.workspace.training_summary, {}) or {}
        summary["manual_mask_audit"] = manual_summary
        FileManager.write_json(self.workspace.training_summary, summary)
        return checkpoints

    def generate_attention_masks(
        self,
        force: bool = False,
        device: str | None = None,
    ) -> list[dict[str, Any]]:


        rows = self._rows()
        checkpoints = SegmentationManager.load_checkpoint_map(self.workspace)
        if not force:
            cached = SegmentationManager.load_cached_predictions(
                rows, self.workspace, checkpoints
            )
            if cached is not None:
                print(
                    "[ATTENTION] Final predictions are compatible; "
                    "CUDA is not initialized."
                )
                SegmentationManager.evaluate_oof_segmentation(
                    self.workspace, cached
                )
                return cached

        requested = device or self.attention_device
        with RuntimeManager.device_scope(
            requested, "Attention U-Net prediction"
        ) as compute_device:
            return SegmentationManager.predict_all(
                rows,
                self.workspace,
                compute_device,
                checkpoint_map=checkpoints,
                force=force,
            )


    def open_editor(
        self,
        scope: str = "invalid",
        limit: int | None = None,
        start_index: int = 0,
        brush_radius: int | None = None,
        review_round: int = 1,
        seed: int = 42,
    ) -> MaskEditor:


        rows = self._rows()


        imported_quality = WorkspaceCompatibilityManager.ensure_quality_audit(
            rows, self.workspace
        )
        if scope in {"novel", "uncertain"} and len(imported_quality) != len(rows):
            self.audit_quality(refresh=False)
        WorkspaceCompatibilityManager.ensure_prediction_audit(
            rows,
            self.workspace,
            require_complete=False,
        )
        queue = ReviewManager.select(
            rows,
            self.workspace,
            scope=scope,
            limit=limit,
            seed=seed,
            review_round=review_round,
        )
        editor = MaskEditor(
            queue,
            self.workspace,
            start_index=start_index,
            brush_radius=brush_radius or Settings.Review.BRUSH_RADIUS,
            review_round=review_round,
            review_scope=scope,
        )
        return editor.show()


    def build_cross_class_matching(
        self,
        force: bool = False,
    ) -> list[dict[str, Any]]:


        rows = self._rows()
        imported_quality = WorkspaceCompatibilityManager.ensure_quality_audit(
            rows, self.workspace
        )
        if len(imported_quality) != len(rows):
            self.audit_quality(refresh=False)
        WorkspaceCompatibilityManager.ensure_prediction_audit(
            rows,
            self.workspace,
            require_complete=True,
        )
        return CrossClassMatchingManager.build(
            rows,
            self.workspace,
            force=force,
        )


    def build_feature_bank(
        self,
        force: bool = False,
        device: str | None = None,
    ) -> dict[str, dict[str, Any]]:


        rows = self._rows()


        WorkspaceCompatibilityManager.ensure_prediction_audit(
            rows,
            self.workspace,
            require_complete=True,
        )
        manual_audit_refresh_reason = MaskManager.manual_audit_refresh_reason(
            self.workspace
        )
        if manual_audit_refresh_reason:
            print(
                "[MANUAL MASKS] Incompatible or stale audit; rebuilding "
                f"automatically ({manual_audit_refresh_reason})."
            )
            self.audit_manual_masks(refresh_quality=False)


        self.build_cross_class_matching(force=force)
        if not force:
            cached = FeatureManager.load_compatible_cache(rows, self.workspace)
            if cached is not None:
                print(
                    "[FEATURE BANK] Compatible cache found; "
                    "CUDA is not initialized."
                )
                return cached

        requested = device or self.feature_device
        with RuntimeManager.device_scope(
            requested, "EfficientNet feature extraction"
        ) as compute_device:
            return FeatureManager.build(
                rows, self.workspace, compute_device, force=force
            )


    def evaluate(self) -> pd.DataFrame:


        bank = FeatureManager.load(self.workspace)
        return EvaluationManager.evaluate(bank, self.workspace)

    def backup(self, name: str = "cad_attention_workspace_backup") -> Path:


        destination = self.workspace.root.parent / name
        archive = Path(shutil.make_archive(str(destination), "zip", self.workspace.root))
        print("[BACKUP]", archive)
        return archive

    def status(self) -> dict[str, Any]:


        matching_summary = FileManager.read_json(
            self.workspace.cross_class_matching_summary, {}
        ) or {}
        status = {
            "pipeline_version": PIPELINE_VERSION,
            "controller_device": "cpu",
            "attention_device_configured": self.attention_device,
            "feature_device_configured": self.feature_device,
            "dataset": str(self.dataset_path),
            "workspace": str(self.workspace.root),
            "manual_masks": len(list(self.workspace.manual_masks.glob("*.png"))),
            "checkpoints": len(
                list(self.workspace.checkpoints.glob("attention_unet_fold_*.pt"))
            ),
            "prediction_parts": len(
                list(self.workspace.prediction_parts_dir.glob("fold_*.csv"))
            ),
            "predicted_masks": len(
                list(self.workspace.predicted_masks.glob("*.png"))
            ),
            "quality_audit": self.workspace.quality_audit.is_file(),
            "manual_audit": self.workspace.manual_audit.is_file(),
            "manual_annotations": self.workspace.manual_annotations.is_file(),
            "prediction_audit": self.workspace.prediction_audit.is_file(),
            "segmentation_oof_metrics": self.workspace.segmentation_metrics.is_file(),
            "cross_class_matching_manifest": self.workspace.cross_class_matching_manifest.is_file(),
            "cross_class_matching_summary": self.workspace.cross_class_matching_summary.is_file(),
            "cross_class_core_pairs": _as_int(
                matching_summary.get("core_matched_pairs"), 0
            ),
            "cross_class_total_pairs": _as_int(
                matching_summary.get("matched_pairs"), 0
            ),
            "cross_class_selected_images": _as_int(
                matching_summary.get("selected_images"), 0
            ),
            "cross_class_selected_fraction_dataset": _as_float(
                matching_summary.get("selected_fraction_of_dataset"), 0.0
            ),
            "feature_bank": self.workspace.feature_bank.is_file(),
            "results": self.workspace.results_csv.is_file(),
            **WorkspaceCompatibilityManager.legacy_status(self.workspace),
        }
        print(json.dumps(status, indent=2))
        return status


RuntimeManager.seed_everything(include_cuda=False)
print(f"[PIPELINE] Version: {PIPELINE_VERSION}")
print("[PIPELINE] Classes loaded without initializing CUDA.")
print("[PIPELINE] 2.5D context, explicit negatives, and core+extended Sick↔Normal matching are active.")
print("[PIPELINE] Legacy manual-audit migration for M1/C3/AU3/C4 is active.")
print("[PIPELINE] The editor locally blocks Kaggle's context menu during right-click erasing.")
print("[PIPELINE] Review automatically registers existing non-empty manual masks as HEART_PRESENT.")
print("[PIPELINE] The initial review queue contains only target=UNLABELED; labeled images remain navigable during the session.")
print("[PIPELINE] CPU is the default; device='cuda' is passed only to the requested stage.")


def _execution_banner(title: str) -> None:


    line = "=" * 88
    print(f"\n{line}\n{title}\n{line}")


def create_pipeline(
    dataset_path: Path | str | None = None,
    workspace_root: Path | str | None = None,
    attention_device: str = "cuda",
    feature_device: str = "cpu",
) -> CADPipeline:


    return CADPipeline(
        dataset_path=dataset_path,
        workspace_root=workspace_root,
        attention_device=attention_device,
        feature_device=feature_device,
    )


def run_complete_pipeline(
    dataset_path: Path | str | None = None,
    workspace_root: Path | str | None = None,
    attention_device: str = "cuda",
    feature_device: str = "cpu",
    force_attention_training: bool = False,
    force_attention_prediction: bool = False,
    force_feature_bank: bool = False,
    refresh_quality: bool = False,
    minimum_masks: int | None = None,
    create_backup: bool = False,
    backup_name: str = "cad_attention_workspace_backup",
) -> dict[str, Any]:


    _execution_banner("COMPLETE RUN — CPU initialization and manifest")
    pipeline = create_pipeline(
        dataset_path=dataset_path,
        workspace_root=workspace_root,
        attention_device=attention_device,
        feature_device=feature_device,
    )
    dataset_rows = pipeline.prepare()

    _execution_banner(
        f"COMPLETE RUN — Attention U-Net on {str(attention_device).upper()} "
        "(the audit runs on CPU first)"
    )
    attention_checkpoints = pipeline.train_attention(
        force=force_attention_training,
        minimum_masks=minimum_masks,
        refresh_quality=refresh_quality,
        device=attention_device,
    )
    attention_predictions = pipeline.generate_attention_masks(
        force=force_attention_prediction,
        device=attention_device,
    )

    _execution_banner(
        f"COMPLETE RUN — feature bank on {str(feature_device).upper()} and CPU evaluation"
    )
    feature_bank = pipeline.build_feature_bank(
        force=force_feature_bank,
        device=feature_device,
    )
    results = pipeline.evaluate()
    status = pipeline.status()

    training_summary = FileManager.read_json(
        pipeline.workspace.training_summary, {}
    ) or {}
    manual_summary = training_summary.get("manual_mask_audit", {})

    backup_path = None
    if create_backup:
        backup_path = pipeline.backup(backup_name)

    return {
        "pipeline": pipeline,
        "dataset_rows": dataset_rows,
        "manual_summary": manual_summary,
        "attention_checkpoints": attention_checkpoints,
        "attention_predictions": attention_predictions,
        "feature_bank": feature_bank,
        "matching_summary": FileManager.read_json(
            pipeline.workspace.cross_class_matching_summary, {}
        ) or {},
        "results": results,
        "status": status,
        "backup_path": backup_path,
    }


def run_review_block(
    pipeline: CADPipeline,
    scope: str = "invalid",
    limit: int = 300,
    start_index: int = 0,
    review_round: int = 1,
    seed: int = 42,
) -> MaskEditor | None:


    scope = str(scope).strip().lower()
    _execution_banner(f"MANUAL REVIEW ON CPU — scope={scope}")


    dataset_rows = pipeline._rows()
    MaskManager.register_existing_manual_masks_as_heart_present(
        dataset_rows,
        pipeline.workspace,
    )

    if scope == "invalid":
        prediction_rows = WorkspaceCompatibilityManager.ensure_prediction_audit(
            dataset_rows,
            pipeline.workspace,
            require_complete=False,
        )
        annotations = MaskManager.annotation_map(pipeline.workspace)
        labeled_tokens = {
            str(token)
            for token, annotation in annotations.items()
            if ReviewManager._normalize_target_type(
                annotation.get("target_type", "")
            )
        }
        invalid_rows = [
            row
            for row in prediction_rows
            if _as_int(row.get("attention_valid_final"), 1) == 0
            and Path(row.get("predicted_attention_mask_path", "")).is_file()
            and str(row.get("image_token", "")) not in labeled_tokens
        ]
        print("Invalid predictions with target=UNLABELED:", len(invalid_rows))
        if not invalid_rows:
            print(
                "There are no invalid UNLABELED predictions to correct; "
                "already labeled images are not reopened."
            )
            return None

    try:
        return pipeline.open_editor(
            scope=scope,
            limit=limit,
            start_index=start_index,
            review_round=review_round,
            seed=seed,
        )
    except (RuntimeError, ValueError) as error:
        if "UNLABELED" in str(error):
            print(str(error))
            return None
        raise


def run_after_review_pipeline(
    pipeline: CADPipeline,
    attention_device: str | None = None,
    feature_device: str | None = None,
    force_all: bool = False,
    refresh_quality: bool = False,
    minimum_masks: int | None = None,
    create_backup: bool = False,
    backup_name: str = "cad_attention_workspace_after_review",
) -> dict[str, Any]:


    attention_device = attention_device or pipeline.attention_device
    feature_device = feature_device or pipeline.feature_device

    _execution_banner(
        f"AFTER REVIEW — incremental Attention retraining on {str(attention_device).upper()}"
    )
    attention_checkpoints = pipeline.train_attention(
        force=force_all,
        minimum_masks=minimum_masks,
        refresh_quality=refresh_quality,
        device=attention_device,
    )
    attention_predictions = pipeline.generate_attention_masks(
        force=force_all,
        device=attention_device,
    )

    _execution_banner(
        f"AFTER REVIEW — feature bank on {str(feature_device).upper()} and CPU evaluation"
    )
    feature_bank = pipeline.build_feature_bank(
        force=force_all,
        device=feature_device,
    )
    results = pipeline.evaluate()
    status = pipeline.status()

    training_summary = FileManager.read_json(
        pipeline.workspace.training_summary, {}
    ) or {}
    manual_summary = training_summary.get("manual_mask_audit", {})

    backup_path = None
    if create_backup:
        backup_path = pipeline.backup(backup_name)

    return {
        "pipeline": pipeline,
        "manual_summary": manual_summary,
        "attention_checkpoints": attention_checkpoints,
        "attention_predictions": attention_predictions,
        "feature_bank": feature_bank,
        "matching_summary": FileManager.read_json(
            pipeline.workspace.cross_class_matching_summary, {}
        ) or {},
        "results": results,
        "status": status,
        "backup_path": backup_path,
    }


def run_prepare(pipeline: CADPipeline) -> list[dict[str, Any]]:
    dataset_rows = pipeline.prepare()
    pipeline.status()
    return dataset_rows


def run_audit(
    pipeline: CADPipeline,
    refresh: bool = False,
    minimum_masks: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    quality_by_token = pipeline.audit_quality(refresh=refresh)
    accepted_manual_masks, manual_summary = pipeline.audit_manual_masks(
        refresh_quality=False,
        minimum_masks=minimum_masks,
    )
    print(json.dumps(manual_summary, indent=2, ensure_ascii=False, default=str))
    return quality_by_token, accepted_manual_masks, manual_summary


def run_attention_training(
    pipeline: CADPipeline,
    force: bool = False,
    minimum_masks: int | None = None,
    refresh_quality: bool = False,
    device: str | None = None,
) -> dict[int, Path]:
    return pipeline.train_attention(
        force=force,
        minimum_masks=minimum_masks,
        refresh_quality=refresh_quality,
        device=device,
    )


def run_attention_prediction(
    pipeline: CADPipeline,
    force: bool = False,
    device: str | None = None,
) -> list[dict[str, Any]]:
    predictions = pipeline.generate_attention_masks(force=force, device=device)
    print("Attention predictions:", len(predictions))
    return predictions


def run_cross_class_matching(
    pipeline: CADPipeline,
    force: bool = False,
) -> list[dict[str, Any]]:
    manifest = pipeline.build_cross_class_matching(force=force)
    summary = FileManager.read_json(
        pipeline.workspace.cross_class_matching_summary, {}
    ) or {}
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return manifest


def run_feature_bank(
    pipeline: CADPipeline,
    force: bool = False,
    device: str | None = None,
) -> dict[str, dict[str, Any]]:
    feature_bank = pipeline.build_feature_bank(force=force, device=device)
    summary = {
        mode: {
            "patients": len(values["patient_ids"]),
            "source_slices": values["source_slices"],
            "series_proxies": values["series_proxies"],
        }
        for mode, values in feature_bank.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return feature_bank


def run_evaluation(pipeline: CADPipeline) -> pd.DataFrame:
    results = pipeline.evaluate()
    print(results.to_string(index=False))
    return results


def run_kaggle_cpu_stage(
    dataset_path: Path | str | None = None,
    workspace_root: Path | str | None = None,
    refresh_quality: bool = False,
    minimum_masks: int | None = None,
) -> dict[str, Any]:


    _execution_banner("KAGGLE CPU — MANIFEST + AUDIT")
    pipeline = create_pipeline(
        dataset_path=dataset_path,
        workspace_root=workspace_root,
        attention_device="cpu",
        feature_device="cpu",
    )
    dataset_rows = pipeline.prepare()
    accepted_manual_masks, manual_summary = pipeline.audit_manual_masks(
        refresh_quality=refresh_quality,
        minimum_masks=minimum_masks,
    )
    quality_by_token = {
        row["image_token"]: row
        for row in FileManager.read_csv(pipeline.workspace.quality_audit)
    }
    status = pipeline.status()
    print(json.dumps(manual_summary, indent=2, ensure_ascii=False, default=str))
    return {
        "pipeline": pipeline,
        "dataset_rows": dataset_rows,
        "quality_by_token": quality_by_token,
        "accepted_manual_masks": accepted_manual_masks,
        "manual_summary": manual_summary,
        "status": status,
    }


def run_kaggle_gpu_stage(
    dataset_path: Path | str | None = None,
    workspace_root: Path | str | None = None,
    after_review: bool = False,
    force_training: bool = False,
    force_prediction: bool = False,
    refresh_quality: bool = False,
    minimum_masks: int | None = None,
) -> dict[str, Any]:


    if not torch.cuda.is_available():
        raise RuntimeError(
            "A GPU is unavailable. In Kaggle choose "
            "Settings -> Accelerator -> GPU, restart the session, and rerun "
            "the definitions cell before the GPU stage."
        )

    phase = "AFTER REVIEW" if after_review else "INITIAL"
    _execution_banner(f"KAGGLE GPU — ATTENTION U-NET ({phase})")
    print("[GPU]", torch.cuda.get_device_name(0))
    pipeline = create_pipeline(
        dataset_path=dataset_path,
        workspace_root=workspace_root,
        attention_device="cuda",
        feature_device="cpu",
    )

    pipeline.prepare()
    checkpoints = pipeline.train_attention(
        force=force_training,
        minimum_masks=minimum_masks,
        refresh_quality=refresh_quality,
        device="cuda",
    )
    predictions = pipeline.generate_attention_masks(
        force=force_prediction,
        device="cuda",
    )
    status = pipeline.status()
    RuntimeManager.release(torch.device("cuda"))
    print(
        "[GPU] Stage complete. To stop the accelerator, choose Kaggle "
        "Settings -> Accelerator -> None; the session will restart."
    )
    return {
        "pipeline": pipeline,
        "checkpoints": checkpoints,
        "predictions": predictions,
        "status": status,
        "after_review": bool(after_review),
    }


def run_kaggle_cpu_final_stage(
    dataset_path: Path | str | None = None,
    workspace_root: Path | str | None = None,
    action: str = "evaluate",
    force_feature_bank: bool = False,
    create_backup: bool = False,
    backup_name: str = "cad_attention_workspace_final",
    review_scope: str = "invalid",
    review_limit: int = 300,
    review_start_index: int = 0,
    review_round: int = 1,
    review_seed: int = 42,
) -> dict[str, Any]:


    action = str(action).strip().lower()
    if action not in {"matching", "evaluate", "review", "status"}:
        raise ValueError(
            "action must be 'matching', 'evaluate', 'review', or 'status'."
        )

    _execution_banner(f"KAGGLE CPU FINAL — {action.upper()}")
    pipeline = create_pipeline(
        dataset_path=dataset_path,
        workspace_root=workspace_root,
        attention_device="cpu",
        feature_device="cpu",
    )
    pipeline.prepare()

    if action == "status":
        return {"pipeline": pipeline, "status": pipeline.status()}

    if action == "matching":
        manifest = pipeline.build_cross_class_matching(
            force=force_feature_bank
        )
        return {
            "pipeline": pipeline,
            "matching_manifest": manifest,
            "matching_summary": FileManager.read_json(
                pipeline.workspace.cross_class_matching_summary, {}
            ) or {},
            "status": pipeline.status(),
        }

    if action == "review":
        editor = run_review_block(
            pipeline,
            scope=review_scope,
            limit=review_limit,
            start_index=review_start_index,
            review_round=review_round,
            seed=review_seed,
        )
        return {
            "pipeline": pipeline,
            "editor": editor,
            "status": pipeline.status(),
        }

    feature_bank = pipeline.build_feature_bank(
        force=force_feature_bank,
        device="cpu",
    )
    results = pipeline.evaluate()
    status = pipeline.status()
    backup_path = None
    if create_backup:
        backup_path = pipeline.backup(backup_name)
    return {
        "pipeline": pipeline,
        "feature_bank": feature_bank,
        "matching_summary": FileManager.read_json(
            pipeline.workspace.cross_class_matching_summary, {}
        ) or {},
        "results": results,
        "status": status,
        "backup_path": backup_path,
    }


def main() -> None:


    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Cardiac MRI CAD pipeline with grouped CPU/GPU stages. "
            "Review must be opened in a Jupyter or Kaggle frontend."
        )
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="status",
        choices=[
            "full",
            "review",
            "after-review",
            "prepare",
            "audit",
            "train",
            "predict",
            "matching",
            "features",
            "evaluate",
            "status",
            "backup",
            "kaggle-cpu",
            "kaggle-gpu",
            "kaggle-cpu-final",
        ],
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument(
        "--attention-device", default="cuda", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--feature-device", default="cpu", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-quality", action="store_true")
    parser.add_argument("--minimum-masks", type=int, default=None)
    parser.add_argument(
        "--review-scope",
        default="invalid",
        choices=["invalid", "uncertain", "empty", "novel", "manual", "all"],
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--review-round", type=int, default=1)
    parser.add_argument("--create-backup", action="store_true")
    parser.add_argument("--backup-name", default="cad_attention_workspace_backup")
    parser.add_argument("--after-review", action="store_true")
    parser.add_argument(
        "--final-action",
        default="evaluate",
        choices=["matching", "evaluate", "review", "status"],
    )
    args = parser.parse_args()

    if args.stage == "kaggle-cpu":
        run_kaggle_cpu_stage(
            dataset_path=args.dataset_path,
            workspace_root=args.workspace_root,
            refresh_quality=args.refresh_quality,
            minimum_masks=args.minimum_masks,
        )
        return

    if args.stage == "kaggle-gpu":
        run_kaggle_gpu_stage(
            dataset_path=args.dataset_path,
            workspace_root=args.workspace_root,
            after_review=args.after_review,
            force_training=args.force,
            force_prediction=args.force,
            refresh_quality=args.refresh_quality,
            minimum_masks=args.minimum_masks,
        )
        return

    if args.stage == "kaggle-cpu-final":
        output = run_kaggle_cpu_final_stage(
            dataset_path=args.dataset_path,
            workspace_root=args.workspace_root,
            action=args.final_action,
            force_feature_bank=args.force,
            create_backup=args.create_backup,
            backup_name=args.backup_name,
            review_scope=args.review_scope,
            review_limit=args.limit,
            review_start_index=args.start_index,
            review_round=args.review_round,
        )
        if args.final_action == "evaluate":
            print(output["results"].to_string(index=False))
        elif args.final_action == "matching":
            print(json.dumps(output["matching_summary"], indent=2, ensure_ascii=False))
        return

    if args.stage == "full":
        output = run_complete_pipeline(
            dataset_path=args.dataset_path,
            workspace_root=args.workspace_root,
            attention_device=args.attention_device,
            feature_device=args.feature_device,
            force_attention_training=args.force,
            force_attention_prediction=args.force,
            force_feature_bank=args.force,
            refresh_quality=args.refresh_quality,
            minimum_masks=args.minimum_masks,
            create_backup=args.create_backup,
            backup_name=args.backup_name,
        )
        print(output["results"].to_string(index=False))
        return

    pipeline = create_pipeline(
        dataset_path=args.dataset_path,
        workspace_root=args.workspace_root,
        attention_device=args.attention_device,
        feature_device=args.feature_device,
    )

    if args.stage == "review":
        run_review_block(
            pipeline,
            scope=args.review_scope,
            limit=args.limit,
            start_index=args.start_index,
            review_round=args.review_round,
        )
    elif args.stage == "after-review":
        output = run_after_review_pipeline(
            pipeline,
            attention_device=args.attention_device,
            feature_device=args.feature_device,
            force_all=args.force,
            refresh_quality=args.refresh_quality,
            minimum_masks=args.minimum_masks,
            create_backup=args.create_backup,
            backup_name=args.backup_name,
        )
        print(output["results"].to_string(index=False))
    elif args.stage == "prepare":
        run_prepare(pipeline)
    elif args.stage == "audit":
        run_audit(
            pipeline,
            refresh=args.refresh_quality,
            minimum_masks=args.minimum_masks,
        )
    elif args.stage == "train":
        run_attention_training(
            pipeline,
            force=args.force,
            minimum_masks=args.minimum_masks,
            refresh_quality=args.refresh_quality,
            device=args.device,
        )
    elif args.stage == "predict":
        run_attention_prediction(pipeline, force=args.force, device=args.device)
    elif args.stage == "matching":
        run_cross_class_matching(pipeline, force=args.force)
    elif args.stage == "features":
        run_feature_bank(pipeline, force=args.force, device=args.device)
    elif args.stage == "evaluate":
        run_evaluation(pipeline)
    elif args.stage == "backup":
        pipeline.status()
        pipeline.backup(args.backup_name)
    else:
        pipeline.status()


def _running_in_notebook() -> bool:
    try:
        shell = get_ipython()  
    except Exception:
        return False
    return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"


if __name__ == "__main__" and not _running_in_notebook():
    main()
