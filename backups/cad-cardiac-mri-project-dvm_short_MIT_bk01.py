#%% ============================================================================
# CAD CARDIAC MRI — ADMISSIONS SHOWCASE (SINGLE-FILE, CLASS-BASED, KAGGLE READY)
# =============================================================================
#
# PURPOSE
# -------
# This file is a deliberately simplified, presentation-oriented version of the
# full CAD cardiac-MRI research suite. It preserves the methodological choices
# that make the result defensible while removing historical branches and dozens
# of secondary ablations that obscure the central scientific argument.
#
# The public question is:
#
#     Can a patient-level model separate the released Normal/Sick labels while
#     relying more on cardiac-region information than on simple image geometry,
#     central cropping, or signal outside the ventricular proposal?
#
# The six predeclared showcase experiments are:
#
#     1. strict_roi          — locked zero-background MONAI ROI candidate
#     2. full_image          — standardized full-image comparator
#     3. fixed_center_crop   — simple MONAI-independent 60% center crop
#     4. outside_roi         — pixels outside an enlarged ventricular box
#     5. soft_mask_only      — canonicalized MONAI probability map, no MRI pixels
#     6. native_geometry_only— height/width/aspect-ratio control
#
# IMPORTANT DATA CONTRACT
# -----------------------
# The computational patient unit is fixed to:
#
#     patient_id = Directory_*
#
# The top-level folder supplies the released label:
#
#     Normal/Directory_* -> 0
#     Sick/Directory_*   -> 1
#
# SR_* and series* folders are retained only as folder-defined series proxies.
# They are not treated as patients and must not be described as validated DICOM
# SeriesInstanceUID values because the public release contains JPEG files rather
# than original DICOM metadata.
#
# SCIENTIFIC LIMITATION
# ---------------------
# This is exploratory research code, not a clinical diagnostic system. The
# effective labeled sample size is the number of Directory_* folders, not the
# number of JPEG slices. The pretrained MONAI model is short-axis-specific and
# is used as a gated ROI proposal, not as a validated segmentation ground truth.
#
# KAGGLE USAGE
# ------------
# Upload this file to the notebook and run:
#
#     %run /kaggle/working/cad_mri_admissions_showcase_kaggle.py
#
# Or import and call it explicitly:
#
#     from cad_mri_admissions_showcase_kaggle import (
#         AdmissionsShowcaseConfig,
#         CADMRIAdmissionsShowcase,
#     )
#     config = AdmissionsShowcaseConfig(profile="showcase")
#     result = CADMRIAdmissionsShowcase(config).run()
#
# Set the environment variable CAD_MRI_PROFILE=full before execution to use
# 50 repeated nested-CV runs, 1,000 label permutations, and 2,000 bootstraps.
# The default "showcase" profile uses 10 repeats, 200 permutations, and 1,000
# bootstraps so the notebook remains practical for a reproducibility demo.
#
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm


#%% ============================================================================
# IMMUTABLE DATA OBJECTS
# =============================================================================


@dataclass(frozen=True)
class MRISample:
    """One released JPEG plus immutable patient/series metadata."""

    path: Path
    label: int
    patient_id: str
    series_id: str
    sample_index: int
    file_size_bytes: int


@dataclass(frozen=True)
class ExperimentSpec:
    """Readable public name plus the exact representation evaluated."""

    name: str
    display_name: str
    feature_mode: str
    role: str
    scientific_question: str
    historical_id: str
    use_pca: bool = True


@dataclass
class FeatureBank:
    """Memory-mapped frozen features and aligned per-slice metadata."""

    cache_dir: Path
    features: dict[str, np.ndarray]
    labels: np.ndarray
    patient_ids: np.ndarray
    series_ids: np.ndarray
    sample_indices: np.ndarray
    image_paths: np.ndarray
    decoded_pixel_hashes: np.ndarray
    perceptual_hashes: np.ndarray
    native_geometry: np.ndarray
    standardization_qc: np.ndarray
    monai_valid: np.ndarray
    monai_area_ratio: np.ndarray
    monai_peak_probability: np.ndarray


@dataclass(frozen=True)
class PatientTable:
    """Exactly one feature vector and one label per Directory_* patient."""

    patient_ids: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]


@dataclass
class ExperimentResult:
    """One complete nested patient-level cross-validation result."""

    spec: ExperimentSpec
    patient_ids: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    outer_folds: np.ndarray
    selected_c: np.ndarray
    thresholds: np.ndarray
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    fold_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class SegmentationBatch:
    """Aligned MONAI outputs used by the view factory."""

    soft_probability: torch.Tensor
    hard_mask: torch.Tensor
    valid_mask: torch.Tensor
    area_ratio: torch.Tensor
    peak_probability: torch.Tensor
    mean_foreground_probability: torch.Tensor


#%% ============================================================================
# CONFIGURATION
# =============================================================================


def _default_dataset_path() -> Path:
    kaggle = Path(
        "/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"
    )
    if kaggle.exists():
        return kaggle
    return Path(r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset")


def _default_output_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cad_mri_admissions_showcase")
    return Path.cwd() / "cad_mri_admissions_showcase"


def _default_bundle_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/monai_bundles")
    return Path.cwd() / "monai_bundles"


@dataclass(frozen=True)
class AdmissionsShowcaseConfig:
    """All user-editable settings are centralized in one validated object.

    The default profile is intentionally short enough for a Kaggle demonstration.
    Change only ``profile`` to ``"full"`` for the report-grade stability and
    permutation settings used by the complete research workflow.
    """

    profile: str = field(
        default_factory=lambda: os.environ.get("CAD_MRI_PROFILE", "showcase")
    )
    dataset_path: Path = field(default_factory=_default_dataset_path)
    output_root: Path = field(default_factory=_default_output_root)
    monai_bundle_parent: Path = field(default_factory=_default_bundle_root)

    # Frozen image pipeline.
    image_size: int = 224
    monai_input_size: int = 256
    standardized_content_long_side: int = 240
    batch_size: int = 8
    dataloader_workers: int = 0
    use_cuda_amp: bool = True
    encoder_modes_per_call: int = 3

    # Patient-level evaluation.
    random_seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    classifier_c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    c_selection_auc_tolerance: float = 0.01
    logistic_max_iter: int = 4000
    pca_explained_variance: float = 0.95
    threshold_method: str = "youden"

    # Profile-dependent uncertainty settings. ``None`` activates the preset.
    bootstrap_replicates: int | None = None
    repeated_cv_repeats: int | None = None
    permutation_replicates: int | None = None

    # Label-blind standardization.
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    dark_line_max_mean: float = 12.0
    dark_line_max_std: float = 4.0
    dark_pixel_max_value: int = 20
    dark_pixel_min_fraction: float = 0.98
    max_crop_fraction_per_side: float = 0.20
    minimum_retained_fraction: float = 0.60
    minimum_padding_run: int = 2

    # ROI and controls.
    monai_dilation_kernel: int = 31
    monai_min_area_ratio: float = 0.003
    monai_max_area_ratio: float = 0.50
    monai_min_peak_probability: float = 0.50
    fixed_center_fraction: float = 0.60
    outside_bbox_context_fraction: float = 0.30
    canonical_soft_mask_content_fraction: float = 0.75

    # Pinned model provenance.
    monai_bundle_name: str = "ventricular_short_axis_3label"
    monai_bundle_version: str = "0.3.5"
    monai_hf_repo_id: str = "MONAI/ventricular_short_axis_3label"
    monai_hf_revision: str = "eefc17c8e002cc8a567bbfce8f02d7d3116408f4"
    monai_model_ts_sha256: str = (
        "27d5532401fa6c1883872fa21635adbb7615981e7f385d0c58dd75b355e340b3"
    )
    auto_download_monai: bool = True
    efficientnet_weights_name: str = "IMAGENET1K_V1"

    # Cache/audit/reporting.
    use_feature_cache: bool = True
    force_rebuild_feature_cache: bool = False
    feature_cache_schema: str = "2026-09-02-admissions-showcase-v1"
    run_perceptual_duplicate_audit: bool = True
    phash_hamming_threshold: int = 4
    save_debug_panels: bool = True
    debug_panel_count: int = 6
    detailed_console: bool = False

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def effective_bootstrap_replicates(self) -> int:
        if self.bootstrap_replicates is not None:
            return int(self.bootstrap_replicates)
        return 2000 if self.profile == "full" else 1000

    @property
    def effective_repeated_cv_repeats(self) -> int:
        if self.repeated_cv_repeats is not None:
            return int(self.repeated_cv_repeats)
        return 50 if self.profile == "full" else 10

    @property
    def effective_permutation_replicates(self) -> int:
        if self.permutation_replicates is not None:
            return int(self.permutation_replicates)
        return 1000 if self.profile == "full" else 200

    @property
    def feature_modes(self) -> tuple[str, ...]:
        return (
            "strict_roi",
            "full_image",
            "fixed_center_crop",
            "outside_roi",
            "soft_mask_only",
        )

    @property
    def experiments(self) -> tuple[ExperimentSpec, ...]:
        return (
            ExperimentSpec(
                name="strict_roi",
                display_name="Strict cardiac ROI",
                feature_mode="strict_roi",
                role="primary_candidate",
                scientific_question=(
                    "Can a strict gated cardiac ROI provide a stable patient-level "
                    "signal after label-blind standardization?"
                ),
                historical_id=(
                    "A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA"
                ),
            ),
            ExperimentSpec(
                name="full_image",
                display_name="Standardized full image",
                feature_mode="full_image",
                role="localization_comparator",
                scientific_question=(
                    "Does cardiac localization outperform the complete standardized "
                    "field of view?"
                ),
                historical_id="A9_STANDARDIZED_FULL_HIER_LR_PCA",
            ),
            ExperimentSpec(
                name="fixed_center_crop",
                display_name="Fixed 60% center crop",
                feature_mode="fixed_center_crop",
                role="simple_comparator",
                scientific_question=(
                    "Does MONAI add value beyond a simple central crop that is "
                    "independent of labels and segmentation?"
                ),
                historical_id="C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA",
            ),
            ExperimentSpec(
                name="outside_roi",
                display_name="Outside ventricular box",
                feature_mode="outside_roi",
                role="negative_control",
                scientific_question=(
                    "How much label signal remains outside an enlarged MONAI "
                    "ventricular bounding box?"
                ),
                historical_id=(
                    "C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA"
                ),
            ),
            ExperimentSpec(
                name="soft_mask_only",
                display_name="Canonical soft mask only",
                feature_mode="soft_mask_only",
                role="segmentation_representation_control",
                scientific_question=(
                    "Do MRI intensities add information beyond canonicalized MONAI "
                    "soft-mask morphology and confidence?"
                ),
                historical_id=(
                    "C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA"
                ),
            ),
            ExperimentSpec(
                name="native_geometry_only",
                display_name="Native geometry only",
                feature_mode="native_geometry_only",
                role="provenance_control",
                scientific_question=(
                    "Can native height, width, and aspect ratio alone predict the "
                    "released label?"
                ),
                historical_id="C19_NATIVE_GEOMETRY_ONLY_LR",
                use_pca=False,
            ),
        )

    def validate(self) -> None:
        """Fail before scanning or model loading when settings are inconsistent."""

        if self.profile not in {"showcase", "full"}:
            raise ValueError("profile must be 'showcase' or 'full'.")
        if self.image_size != 224:
            raise ValueError("The reviewed EfficientNet contract requires 224x224.")
        if self.monai_input_size != 256:
            raise ValueError("The pinned MONAI bundle requires 256x256 input.")
        if not 1 <= self.standardized_content_long_side <= self.monai_input_size:
            raise ValueError("standardized_content_long_side is outside the canvas.")
        if self.batch_size <= 0 or self.dataloader_workers < 0:
            raise ValueError("Invalid DataLoader configuration.")
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("Nested CV requires at least two outer and inner folds.")
        if any(value <= 0 for value in self.classifier_c_grid):
            raise ValueError("Every C candidate must be strictly positive.")
        if not 0.0 < self.pca_explained_variance < 1.0:
            raise ValueError("PCA explained variance must lie in (0,1).")
        if self.threshold_method != "youden":
            raise ValueError("The showcase intentionally supports only Youden tuning.")
        if self.monai_dilation_kernel <= 0 or self.monai_dilation_kernel % 2 == 0:
            raise ValueError("MONAI dilation kernel must be a positive odd integer.")
        if not (
            0.0 <= self.monai_min_area_ratio
            < self.monai_max_area_ratio
            <= 1.0
        ):
            raise ValueError("Invalid MONAI area-ratio gate.")
        if not 0.0 <= self.monai_min_peak_probability <= 1.0:
            raise ValueError("Invalid MONAI peak-probability gate.")
        if not 0.0 < self.fixed_center_fraction <= 1.0:
            raise ValueError("fixed_center_fraction must lie in (0,1].")
        if self.outside_bbox_context_fraction < 0.0:
            raise ValueError("outside_bbox_context_fraction cannot be negative.")
        if not 0.0 < self.canonical_soft_mask_content_fraction <= 1.0:
            raise ValueError("canonical soft-mask fraction must lie in (0,1].")
        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 100.0:
            raise ValueError("Invalid robust intensity percentiles.")
        if not 0.0 < self.minimum_retained_fraction <= 1.0:
            raise ValueError("minimum_retained_fraction must lie in (0,1].")
        if not 0.0 <= self.max_crop_fraction_per_side < 0.5:
            raise ValueError("max_crop_fraction_per_side must lie in [0,0.5).")
        if self.minimum_padding_run < 1:
            raise ValueError("minimum_padding_run must be at least one.")
        if self.encoder_modes_per_call < 1:
            raise ValueError("encoder_modes_per_call must be positive.")
        if self.efficientnet_weights_name not in (
            models.EfficientNet_B0_Weights.__members__
        ):
            raise ValueError("Unknown EfficientNet-B0 weight enum.")
        digest = self.monai_model_ts_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError("MONAI model.ts SHA-256 must be a hexadecimal digest.")
        for value, label in (
            (self.effective_bootstrap_replicates, "bootstrap_replicates"),
            (self.effective_repeated_cv_repeats, "repeated_cv_repeats"),
            (self.effective_permutation_replicates, "permutation_replicates"),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive.")


#%% ============================================================================
# LOGGING AND SMALL UTILITIES
# =============================================================================


class ShowcaseLogger:
    """INFO goes to the notebook; DEBUG is retained in a full run log."""

    def __init__(self, output_dir: Path, detailed_console: bool) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger_name = f"cad_mri_showcase_{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(logging.DEBUG if detailed_console else logging.INFO)
        stream.setFormatter(formatter)
        self.logger.addHandler(stream)

        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)

    def debug(self, message: str, *args: Any) -> None:
        self.logger.debug(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self.logger.warning(message, *args)


class Timer:
    """Context manager that logs a concise measured stage duration."""

    def __init__(self, logger: ShowcaseLogger, title: str) -> None:
        self.logger = logger
        self.title = title
        self.started = 0.0

    def __enter__(self) -> "Timer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.logger.info("START — %s", self.title)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        if exc is None:
            self.logger.info("DONE  — %s (%s)", self.title, format_duration(elapsed))
        else:
            self.logger.warning("FAILED — %s after %s", self.title, format_duration(elapsed))


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {sec:.1f} s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)} h {int(minutes)} min {sec:.1f} s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


#%% ============================================================================
# DATA DISCOVERY
# =============================================================================


class DatasetScanner:
    """Discover Directory_* patients and folder-defined series proxies."""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(self, config: AdmissionsShowcaseConfig, logger: ShowcaseLogger):
        self.config = config
        self.logger = logger

    def discover(self) -> list[MRISample]:
        root = self.config.dataset_path
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {root}")

        samples: list[MRISample] = []
        patient_to_class: dict[str, str] = {}

        for class_name, label in (("Normal", 0), ("Sick", 1)):
            class_root = root / class_name
            if not class_root.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_root}")
            before = len(samples)
            before_patients = len(patient_to_class)

            for patient_root in sorted(class_root.iterdir()):
                if not patient_root.is_dir() or not patient_root.name.lower().startswith(
                    "directory_"
                ):
                    continue
                patient_id = patient_root.name
                previous = patient_to_class.get(patient_id)
                if previous is not None and previous != class_name:
                    raise RuntimeError(
                        f"{patient_id} appears under both {previous} and {class_name}."
                    )
                patient_to_class[patient_id] = class_name

                root_images = sorted(
                    path
                    for path in patient_root.iterdir()
                    if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
                )
                root_series_id = f"{patient_id}/__ROOT__"
                for path in root_images:
                    samples.append(
                        MRISample(
                            path=path,
                            label=label,
                            patient_id=patient_id,
                            series_id=root_series_id,
                            sample_index=len(samples),
                            file_size_bytes=path.stat().st_size,
                        )
                    )

                for series_root in sorted(path for path in patient_root.iterdir() if path.is_dir()):
                    series_id = f"{patient_id}/{series_root.name}"
                    for path in sorted(series_root.rglob("*")):
                        if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES:
                            samples.append(
                                MRISample(
                                    path=path,
                                    label=label,
                                    patient_id=patient_id,
                                    series_id=series_id,
                                    sample_index=len(samples),
                                    file_size_bytes=path.stat().st_size,
                                )
                            )

            self.logger.info(
                "%s: %d patients, %d images",
                class_name,
                len(patient_to_class) - before_patients,
                len(samples) - before,
            )

        if not samples:
            raise RuntimeError("No MRI JPEG/PNG files were discovered.")

        patient_labels: dict[str, int] = {}
        for sample in samples:
            previous = patient_labels.get(sample.patient_id)
            if previous is not None and previous != sample.label:
                raise RuntimeError(f"Conflicting labels for {sample.patient_id}.")
            patient_labels[sample.patient_id] = sample.label

        counts = np.bincount(np.asarray(list(patient_labels.values())), minlength=2)
        self.logger.info(
            "Cohort: %d Directory_* patients (%d Normal, %d Sick), %d images, %d series proxies",
            len(patient_labels),
            int(counts[0]),
            int(counts[1]),
            len(samples),
            len({sample.series_id for sample in samples}),
        )
        if int(counts.min()) < self.config.outer_splits:
            raise RuntimeError(
                "Each class must contain at least outer_splits patients. "
                f"Found class counts {counts.tolist()}."
            )
        return samples


#%% ============================================================================
# LABEL-BLIND IMAGE STANDARDIZATION
# =============================================================================


class ImageStandardizer:
    """Create aligned MONAI and EfficientNet views without reading labels."""

    QC_NAMES = (
        "crop_applied",
        "top_crop_fraction",
        "bottom_crop_fraction",
        "left_crop_fraction",
        "right_crop_fraction",
        "retained_height_fraction",
        "retained_width_fraction",
        "robust_lower_uint8_fraction",
        "robust_upper_uint8_fraction",
        "pipeline_padding_fraction",
    )

    def __init__(self, config: AdmissionsShowcaseConfig):
        self.config = config

    @staticmethod
    def _minmax(image: np.ndarray) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum <= minimum:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - minimum) / (maximum - minimum)).astype(np.float32)

    def _robust_scale(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        values = np.asarray(image, dtype=np.float32)
        lower = float(np.percentile(values, self.config.lower_percentile))
        upper = float(np.percentile(values, self.config.upper_percentile))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower = float(values.min())
            upper = float(values.max())
        if upper <= lower:
            return np.zeros_like(values, dtype=np.float32), lower, upper
        scaled = np.clip(values, lower, upper)
        scaled = (scaled - lower) / max(upper - lower, 1e-8)
        return scaled.astype(np.float32), lower, upper

    def _dark_uniform_line(self, line: np.ndarray) -> bool:
        values = np.asarray(line, dtype=np.float32).reshape(-1)
        return bool(
            values.size > 0
            and float(values.mean()) <= self.config.dark_line_max_mean
            and float(values.std()) <= self.config.dark_line_max_std
            and float(np.mean(values <= self.config.dark_pixel_max_value))
            >= self.config.dark_pixel_min_fraction
        )

    def detect_padding_bounds(self, image: np.ndarray) -> tuple[int, int, int, int]:
        if image.ndim != 2:
            raise ValueError(f"Expected a grayscale image, got {image.shape}.")
        height, width = image.shape
        min_height = max(8, int(math.ceil(height * self.config.minimum_retained_fraction)))
        min_width = max(8, int(math.ceil(width * self.config.minimum_retained_fraction)))
        max_vertical = int(math.floor(height * self.config.max_crop_fraction_per_side))
        max_horizontal = int(math.floor(width * self.config.max_crop_fraction_per_side))

        top = 0
        while (
            top < max_vertical
            and height - (top + 1) >= min_height
            and self._dark_uniform_line(image[top, :])
        ):
            top += 1

        bottom_count = 0
        while (
            bottom_count < max_vertical
            and height - top - (bottom_count + 1) >= min_height
            and self._dark_uniform_line(image[height - 1 - bottom_count, :])
        ):
            bottom_count += 1

        left = 0
        while (
            left < max_horizontal
            and width - (left + 1) >= min_width
            and self._dark_uniform_line(image[:, left])
        ):
            left += 1

        right_count = 0
        while (
            right_count < max_horizontal
            and width - left - (right_count + 1) >= min_width
            and self._dark_uniform_line(image[:, width - 1 - right_count])
        ):
            right_count += 1

        if top < self.config.minimum_padding_run:
            top = 0
        if bottom_count < self.config.minimum_padding_run:
            bottom_count = 0
        if left < self.config.minimum_padding_run:
            left = 0
        if right_count < self.config.minimum_padding_run:
            right_count = 0

        bottom = height - bottom_count
        right = width - right_count
        if bottom - top < min_height or right - left < min_width:
            return 0, height, 0, width
        return int(top), int(bottom), int(left), int(right)

    def _fixed_canvas(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        values = np.asarray(image, dtype=np.float32)
        height, width = values.shape
        scale = self.config.standardized_content_long_side / max(height, width)
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        if height >= width:
            new_height = self.config.standardized_content_long_side
        else:
            new_width = self.config.standardized_content_long_side
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(values, (new_width, new_height), interpolation=interpolation)
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
        canvas = np.zeros(
            (self.config.monai_input_size, self.config.monai_input_size),
            dtype=np.float32,
        )
        top = (self.config.monai_input_size - new_height) // 2
        left = (self.config.monai_input_size - new_width) // 2
        canvas[top : top + new_height, left : left + new_width] = resized
        return canvas, new_height, new_width

    def prepare(self, raw_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return MONAI [1,256,256], classifier [3,224,224], and QC vector."""

        if raw_uint8.ndim != 2 or raw_uint8.size == 0:
            raise ValueError("A non-empty grayscale image is required.")
        height, width = raw_uint8.shape
        top, bottom, left, right = self.detect_padding_bounds(raw_uint8)
        cropped = raw_uint8[top:bottom, left:right]
        if cropped.size == 0:
            raise RuntimeError("Standardization produced an empty crop.")

        monai_scaled = self._minmax(cropped)
        robust_scaled, lower, upper = self._robust_scale(cropped)
        monai_canvas, monai_h, monai_w = self._fixed_canvas(monai_scaled)
        robust_canvas, robust_h, robust_w = self._fixed_canvas(robust_scaled)
        if (monai_h, monai_w) != (robust_h, robust_w):
            raise RuntimeError("MONAI and classifier canvases are misaligned.")

        classifier_gray = cv2.resize(
            robust_canvas,
            (self.config.image_size, self.config.image_size),
            interpolation=cv2.INTER_AREA,
        )
        classifier_rgb = np.stack([classifier_gray] * 3, axis=0).astype(np.float32)
        monai_chw = monai_canvas[None, :, :].astype(np.float32)

        retained_h = (bottom - top) / height
        retained_w = (right - left) / width
        padding_fraction = 1.0 - (robust_h * robust_w) / float(
            self.config.monai_input_size * self.config.monai_input_size
        )
        qc = np.asarray(
            [
                float(any(v > 0 for v in (top, height - bottom, left, width - right))),
                top / height,
                (height - bottom) / height,
                left / width,
                (width - right) / width,
                retained_h,
                retained_w,
                lower / 255.0,
                upper / 255.0,
                padding_fraction,
            ],
            dtype=np.float32,
        )
        return monai_chw, classifier_rgb, qc


class PerceptualHash:
    """Small DCT pHash used only to screen for near-duplicate candidates."""

    @staticmethod
    def compute_hex(image: np.ndarray) -> str:
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.asarray(resized, dtype=np.float32))
        low = dct[:8, :8].reshape(-1)
        median = float(np.median(low[1:]))
        bits = low > median
        value = 0
        for bit in bits.tolist():
            value = (value << 1) | int(bool(bit))
        return f"{value:016x}"


class MRISliceDataset(Dataset):
    """Decode one JPEG and return two aligned network inputs plus audit data."""

    def __init__(self, samples: Sequence[MRISample], standardizer: ImageStandardizer):
        self.samples = list(samples)
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        raw = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"OpenCV could not decode: {sample.path}")

        pixel_digest = hashlib.sha256()
        pixel_digest.update(np.asarray(raw.shape, dtype=np.int32).tobytes())
        pixel_digest.update(raw.tobytes(order="C"))

        monai_image, classifier_image, standardization_qc = self.standardizer.prepare(raw)
        height, width = raw.shape
        native_geometry = np.asarray(
            [float(height), float(width), float(width / max(height, 1))],
            dtype=np.float32,
        )

        return {
            "monai_image": torch.from_numpy(monai_image),
            "classifier_image": torch.from_numpy(classifier_image),
            "label": int(sample.label),
            "patient_id": sample.patient_id,
            "series_id": sample.series_id,
            "sample_index": int(sample.sample_index),
            "image_path": str(sample.path),
            "decoded_pixel_hash": pixel_digest.hexdigest(),
            "perceptual_hash": PerceptualHash.compute_hex(raw),
            "native_geometry": torch.from_numpy(native_geometry),
            "standardization_qc": torch.from_numpy(standardization_qc),
        }


#%% ============================================================================
# PRETRAINED MODELS
# =============================================================================


class MonaiVentricularSegmenter:
    """Load the verified official MONAI TorchScript artifact, then infer masks.

    The showcase deliberately omits the exceptional model.pt reconstruction
    branch from the full research suite. The normal, reviewed path is the pinned
    official ``models/model.ts`` file. A clear error is preferable to hiding an
    environment-specific reconstruction inside the presentation code.
    """

    def __init__(self, config: AdmissionsShowcaseConfig, logger: ShowcaseLogger):
        self.config = config
        self.logger = logger
        self.model: torch.jit.ScriptModule | None = None
        self.artifact_path: Path | None = None

    @property
    def bundle_root(self) -> Path:
        return self.config.monai_bundle_parent / self.config.monai_bundle_name

    def _ensure_artifact(self) -> Path:
        root = self.bundle_root
        model_path = root / "models" / "model.ts"
        metadata_path = root / "configs" / "metadata.json"
        missing = [path for path in (model_path, metadata_path) if not path.is_file()]
        if missing:
            if not self.config.auto_download_monai:
                raise FileNotFoundError(
                    "Pinned MONAI files are missing and auto-download is disabled: "
                    + ", ".join(map(str, missing))
                )
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "Install huggingface_hub to download the pinned MONAI artifact."
                ) from exc
            root.mkdir(parents=True, exist_ok=True)
            self.logger.info("Downloading the pinned MONAI TorchScript bundle files...")
            snapshot_download(
                repo_id=self.config.monai_hf_repo_id,
                revision=self.config.monai_hf_revision,
                local_dir=str(root),
                allow_patterns=["models/model.ts", "configs/metadata.json"],
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("version", "")) != self.config.monai_bundle_version:
            raise RuntimeError("MONAI bundle metadata version does not match the pin.")
        actual = sha256_file(model_path)
        if actual.lower() != self.config.monai_model_ts_sha256.lower():
            raise RuntimeError(
                "MONAI model.ts SHA-256 mismatch. Delete the cached artifact and rerun."
            )
        return model_path

    def load(self) -> None:
        model_path = self._ensure_artifact()
        self.logger.info("Loading verified MONAI ventricular segmenter...")
        model = torch.jit.load(str(model_path), map_location=self.config.device)
        model.eval()
        try:
            model.requires_grad_(False)
        except (AttributeError, RuntimeError):
            pass
        test_input = torch.zeros(
            1,
            1,
            self.config.monai_input_size,
            self.config.monai_input_size,
            device=self.config.device,
        )
        with torch.inference_mode():
            output = model(test_input)
        expected = (1, 4, self.config.monai_input_size, self.config.monai_input_size)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
            raise RuntimeError(
                f"Unexpected MONAI output contract: {getattr(output, 'shape', None)}"
            )
        if not torch.isfinite(output).all():
            raise RuntimeError("MONAI sanity inference produced non-finite values.")
        self.model = model
        self.artifact_path = model_path

    @torch.inference_mode()
    def predict(self, monai_images: torch.Tensor) -> SegmentationBatch:
        if self.model is None:
            raise RuntimeError("MONAI segmenter has not been loaded.")
        logits = self.model(monai_images)
        if logits.ndim != 4 or logits.shape[1] != 4:
            raise RuntimeError(f"Unexpected MONAI logits shape: {tuple(logits.shape)}")
        probabilities = torch.softmax(logits.float(), dim=1)
        heart = probabilities[:, 1:, :, :].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        class_map = torch.argmax(probabilities, dim=1, keepdim=True)
        hard = (class_map > 0).float()

        area = hard.mean(dim=(1, 2, 3))
        peak = heart.amax(dim=(1, 2, 3))
        foreground_count = hard.sum(dim=(1, 2, 3))
        mean_fg = (heart * hard).sum(dim=(1, 2, 3)) / foreground_count.clamp_min(1.0)
        valid = (
            (area >= self.config.monai_min_area_ratio)
            & (area <= self.config.monai_max_area_ratio)
            & (peak >= self.config.monai_min_peak_probability)
        )

        kernel = self.config.monai_dilation_kernel
        soft_dilated = F.max_pool2d(
            heart,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        hard_dilated = F.max_pool2d(
            hard,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        target = (self.config.image_size, self.config.image_size)
        soft_aligned = F.interpolate(
            soft_dilated,
            size=target,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        hard_aligned = F.interpolate(hard_dilated, size=target, mode="nearest")
        return SegmentationBatch(
            soft_probability=soft_aligned,
            hard_mask=hard_aligned,
            valid_mask=valid,
            area_ratio=area,
            peak_probability=peak,
            mean_foreground_probability=mean_fg,
        )


class FrozenEfficientNetEncoder(nn.Module):
    """ImageNet EfficientNet-B0 with its 1,000-class head removed."""

    FEATURE_DIM = 1280
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, config: AdmissionsShowcaseConfig):
        super().__init__()
        weights = models.EfficientNet_B0_Weights[config.efficientnet_weights_name]
        network = models.efficientnet_b0(weights=weights)
        network.classifier = nn.Identity()
        network.eval()
        network.requires_grad_(False)
        self.network = network.to(config.device)
        self.config = config

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.MEAN, device=images.device, dtype=images.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(self.STD, device=images.device, dtype=images.dtype).view(
            1, 3, 1, 1
        )
        return (images - mean) / std

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = self.normalize(images)
        amp_enabled = self.config.use_cuda_amp and self.config.device == "cuda"
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = self.network(images)
        else:
            features = self.network(images)
        return features.float()


#%% ============================================================================
# SHOWCASE IMAGE VIEWS
# =============================================================================


class ShowcaseViewFactory:
    """Create only the five image representations used in the public story."""

    def __init__(self, config: AdmissionsShowcaseConfig):
        self.config = config

    @staticmethod
    def _bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
        coordinates = torch.nonzero(mask > 0.5, as_tuple=False)
        if coordinates.numel() == 0:
            return None
        top = int(coordinates[:, 0].min().item())
        bottom = int(coordinates[:, 0].max().item()) + 1
        left = int(coordinates[:, 1].min().item())
        right = int(coordinates[:, 1].max().item()) + 1
        return top, bottom, left, right

    def fixed_center_crop(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        crop_h = max(2, int(round(height * self.config.fixed_center_fraction)))
        crop_w = max(2, int(round(width * self.config.fixed_center_fraction)))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        crop = images[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(
            crop,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

    def strict_roi(
        self,
        images: torch.Tensor,
        segmentation: SegmentationBatch,
    ) -> torch.Tensor:
        strict = images * segmentation.soft_probability.repeat(1, 3, 1, 1)
        fallback = self.fixed_center_crop(images)
        selector = segmentation.valid_mask.view(-1, 1, 1, 1)
        return torch.where(selector, strict, fallback)

    def outside_roi(
        self,
        images: torch.Tensor,
        segmentation: SegmentationBatch,
    ) -> torch.Tensor:
        output = torch.zeros_like(images)
        height, width = images.shape[-2:]
        for index in range(images.shape[0]):
            if not bool(segmentation.valid_mask[index].item()):
                continue
            box = self._bbox(segmentation.hard_mask[index, 0])
            if box is None:
                continue
            top, bottom, left, right = box
            margin = int(
                round(
                    max(bottom - top, right - left)
                    * self.config.outside_bbox_context_fraction
                )
            )
            top = max(0, top - margin)
            bottom = min(height, bottom + margin)
            left = max(0, left - margin)
            right = min(width, right + margin)
            output[index] = images[index]
            output[index, :, top:bottom, left:right] = 0.0
        return output

    def canonical_soft_mask(self, segmentation: SegmentationBatch) -> torch.Tensor:
        soft = segmentation.soft_probability
        hard = segmentation.hard_mask
        output = torch.zeros_like(soft)
        batch_size, _, height, width = soft.shape
        target_side = max(
            2,
            int(round(min(height, width) * self.config.canonical_soft_mask_content_fraction)),
        )
        target_top = (height - target_side) // 2
        target_left = (width - target_side) // 2

        for index in range(batch_size):
            if not bool(segmentation.valid_mask[index].item()):
                continue
            box = self._bbox(hard[index, 0])
            if box is None:
                continue
            top, bottom, left, right = box
            crop = soft[index : index + 1, :, top:bottom, left:right]
            if crop.numel() == 0:
                continue
            crop_h, crop_w = int(crop.shape[-2]), int(crop.shape[-1])
            square_side = max(crop_h, crop_w)
            square = torch.zeros(
                1,
                1,
                square_side,
                square_side,
                device=soft.device,
                dtype=soft.dtype,
            )
            square_top = (square_side - crop_h) // 2
            square_left = (square_side - crop_w) // 2
            square[
                :,
                :,
                square_top : square_top + crop_h,
                square_left : square_left + crop_w,
            ] = crop
            resized = F.interpolate(
                square,
                size=(target_side, target_side),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            output[
                index : index + 1,
                :,
                target_top : target_top + target_side,
                target_left : target_left + target_side,
            ] = resized
        return output.repeat(1, 3, 1, 1)

    def build(
        self,
        full_images: torch.Tensor,
        segmentation: SegmentationBatch,
    ) -> dict[str, torch.Tensor]:
        return {
            "strict_roi": self.strict_roi(full_images, segmentation),
            "full_image": full_images,
            "fixed_center_crop": self.fixed_center_crop(full_images),
            "outside_roi": self.outside_roi(full_images, segmentation),
            "soft_mask_only": self.canonical_soft_mask(segmentation),
        }


#%% ============================================================================
# FEATURE BANK AND CACHE
# =============================================================================


class FeatureBankBuilder:
    """Extract each frozen image representation once and reuse it everywhere."""

    def __init__(
        self,
        config: AdmissionsShowcaseConfig,
        logger: ShowcaseLogger,
        run_dir: Path,
    ) -> None:
        self.config = config
        self.logger = logger
        self.run_dir = run_dir
        self.standardizer = ImageStandardizer(config)
        self.segmenter = MonaiVentricularSegmenter(config, logger)
        self.view_factory = ShowcaseViewFactory(config)

    def _fingerprint(self, samples: Sequence[MRISample]) -> str:
        digest = hashlib.sha256()
        extraction_settings = {
            "schema": self.config.feature_cache_schema,
            "image_size": self.config.image_size,
            "monai_input_size": self.config.monai_input_size,
            "standardized_content_long_side": self.config.standardized_content_long_side,
            "lower_percentile": self.config.lower_percentile,
            "upper_percentile": self.config.upper_percentile,
            "dark_line_max_mean": self.config.dark_line_max_mean,
            "dark_line_max_std": self.config.dark_line_max_std,
            "dark_pixel_max_value": self.config.dark_pixel_max_value,
            "dark_pixel_min_fraction": self.config.dark_pixel_min_fraction,
            "max_crop_fraction_per_side": self.config.max_crop_fraction_per_side,
            "minimum_retained_fraction": self.config.minimum_retained_fraction,
            "minimum_padding_run": self.config.minimum_padding_run,
            "monai_revision": self.config.monai_hf_revision,
            "monai_sha256": self.config.monai_model_ts_sha256,
            "monai_dilation_kernel": self.config.monai_dilation_kernel,
            "monai_gate": (
                self.config.monai_min_area_ratio,
                self.config.monai_max_area_ratio,
                self.config.monai_min_peak_probability,
            ),
            "center_fraction": self.config.fixed_center_fraction,
            "outside_bbox_context": self.config.outside_bbox_context_fraction,
            "canonical_soft_fraction": self.config.canonical_soft_mask_content_fraction,
            "efficientnet_weights": self.config.efficientnet_weights_name,
            "efficientnet_mean": FrozenEfficientNetEncoder.MEAN,
            "efficientnet_std": FrozenEfficientNetEncoder.STD,
            "use_cuda_amp": self.config.use_cuda_amp,
            "batch_size": self.config.batch_size,
            "encoder_modes_per_call": self.config.encoder_modes_per_call,
            "feature_modes": self.config.feature_modes,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        }
        digest.update(json.dumps(extraction_settings, sort_keys=True).encode("utf-8"))
        root = self.config.dataset_path.resolve()
        self.logger.info("Fingerprinting %d dataset files for cache validation...", len(samples))
        progress_step = max(1, len(samples) // 4)
        for position, sample in enumerate(samples, start=1):
            stat = sample.path.stat()
            try:
                relative = sample.path.resolve().relative_to(root).as_posix()
            except ValueError:
                relative = sample.path.resolve().as_posix()
            digest.update(
                (
                    f"{relative}|{stat.st_size}|{stat.st_mtime_ns}|{sample.label}|"
                    f"{sample.patient_id}|{sample.series_id}\n"
                ).encode("utf-8")
            )
            if position % progress_step == 0 or position == len(samples):
                self.logger.info(
                    "Cache fingerprint: %d/%d files (%.0f%%)",
                    position,
                    len(samples),
                    100.0 * position / len(samples),
                )
        return digest.hexdigest()

    @staticmethod
    def _required_cache_files(cache_dir: Path, modes: Sequence[str]) -> list[Path]:
        files = [
            cache_dir / "complete.json",
            cache_dir / "labels.npy",
            cache_dir / "patient_ids.npy",
            cache_dir / "series_ids.npy",
            cache_dir / "sample_indices.npy",
            cache_dir / "image_paths.npy",
            cache_dir / "decoded_pixel_hashes.npy",
            cache_dir / "perceptual_hashes.npy",
            cache_dir / "native_geometry.npy",
            cache_dir / "standardization_qc.npy",
            cache_dir / "monai_valid.npy",
            cache_dir / "monai_area_ratio.npy",
            cache_dir / "monai_peak_probability.npy",
        ]
        files.extend(cache_dir / f"features__{mode}.npy" for mode in modes)
        return files

    def _load(self, cache_dir: Path) -> FeatureBank:
        features = {
            mode: np.load(cache_dir / f"features__{mode}.npy", mmap_mode="r")
            for mode in self.config.feature_modes
        }
        return FeatureBank(
            cache_dir=cache_dir,
            features=features,
            labels=np.load(cache_dir / "labels.npy", allow_pickle=False),
            patient_ids=np.load(cache_dir / "patient_ids.npy", allow_pickle=False),
            series_ids=np.load(cache_dir / "series_ids.npy", allow_pickle=False),
            sample_indices=np.load(cache_dir / "sample_indices.npy", allow_pickle=False),
            image_paths=np.load(cache_dir / "image_paths.npy", allow_pickle=False),
            decoded_pixel_hashes=np.load(
                cache_dir / "decoded_pixel_hashes.npy", allow_pickle=False
            ),
            perceptual_hashes=np.load(
                cache_dir / "perceptual_hashes.npy", allow_pickle=False
            ),
            native_geometry=np.load(cache_dir / "native_geometry.npy", mmap_mode="r"),
            standardization_qc=np.load(
                cache_dir / "standardization_qc.npy", mmap_mode="r"
            ),
            monai_valid=np.load(cache_dir / "monai_valid.npy", allow_pickle=False),
            monai_area_ratio=np.load(
                cache_dir / "monai_area_ratio.npy", allow_pickle=False
            ),
            monai_peak_probability=np.load(
                cache_dir / "monai_peak_probability.npy", allow_pickle=False
            ),
        )

    def _copy_cached_debug_panels(self, cache_dir: Path) -> None:
        source = cache_dir / "debug_panels"
        destination = self.run_dir / "figures" / "pipeline_examples"
        if not source.is_dir():
            return
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.glob("*.png"):
            shutil.copy2(path, destination / path.name)

    def load_or_build(self, samples: Sequence[MRISample]) -> tuple[FeatureBank, bool]:
        fingerprint = self._fingerprint(samples)
        cache_root = self.config.output_root / "feature_cache"
        cache_dir = cache_root / fingerprint[:20]
        required = self._required_cache_files(cache_dir, self.config.feature_modes)
        if (
            self.config.use_feature_cache
            and not self.config.force_rebuild_feature_cache
            and all(path.is_file() for path in required)
        ):
            self.logger.info("Feature-bank cache hit: %s", cache_dir)
            self._copy_cached_debug_panels(cache_dir)
            return self._load(cache_dir), True

        self.logger.info("Feature-bank cache miss; frozen inference will run once.")
        temp_dir = cache_root / f".{cache_dir.name}.building"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._extract(samples, temp_dir, fingerprint)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        temp_dir.rename(cache_dir)
        self._copy_cached_debug_panels(cache_dir)
        return self._load(cache_dir), False

    def _extract(
        self,
        samples: Sequence[MRISample],
        cache_dir: Path,
        fingerprint: str,
    ) -> None:
        self.segmenter.load()
        encoder = FrozenEfficientNetEncoder(self.config)
        dataset = MRISliceDataset(samples, self.standardizer)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_workers,
            pin_memory=self.config.device == "cuda",
        )

        n = len(samples)
        feature_maps = {
            mode: np.lib.format.open_memmap(
                cache_dir / f"features__{mode}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, FrozenEfficientNetEncoder.FEATURE_DIM),
            )
            for mode in self.config.feature_modes
        }

        labels = np.empty(n, dtype=np.int64)
        native_geometry = np.empty((n, 3), dtype=np.float32)
        standardization_qc = np.empty(
            (n, len(ImageStandardizer.QC_NAMES)), dtype=np.float32
        )
        monai_valid = np.empty(n, dtype=np.bool_)
        monai_area = np.empty(n, dtype=np.float32)
        monai_peak = np.empty(n, dtype=np.float32)
        patient_ids: list[str] = [""] * n
        series_ids: list[str] = [""] * n
        image_paths: list[str] = [""] * n
        pixel_hashes: list[str] = [""] * n
        phashes: list[str] = [""] * n
        sample_indices = np.arange(n, dtype=np.int64)

        debug_indices: set[int] = set()
        if self.config.save_debug_panels and self.config.debug_panel_count > 0:
            debug_indices = set(
                np.linspace(
                    0,
                    n - 1,
                    num=min(self.config.debug_panel_count, n),
                    dtype=int,
                ).tolist()
            )
        debug_dir = cache_dir / "debug_panels"
        debug_dir.mkdir(parents=True, exist_ok=True)

        offset = 0
        progress = tqdm(loader, desc="Frozen feature bank", unit="batch")
        for batch in progress:
            batch_indices = batch["sample_index"].cpu().numpy().astype(np.int64)
            expected = np.arange(offset, offset + len(batch_indices), dtype=np.int64)
            if not np.array_equal(batch_indices, expected):
                raise RuntimeError("DataLoader order no longer matches sample indices.")

            images = batch["classifier_image"].to(
                self.config.device, non_blocking=True, dtype=torch.float32
            )
            monai_images = batch["monai_image"].to(
                self.config.device, non_blocking=True, dtype=torch.float32
            )
            segmentation = self.segmenter.predict(monai_images)
            views = self.view_factory.build(images, segmentation)

            mode_names = list(self.config.feature_modes)
            for start in range(0, len(mode_names), self.config.encoder_modes_per_call):
                chunk_names = mode_names[start : start + self.config.encoder_modes_per_call]
                joined = torch.cat([views[name] for name in chunk_names], dim=0)
                encoded = encoder.encode(joined).cpu().numpy().astype(np.float32)
                batch_size = images.shape[0]
                for local_mode_index, mode_name in enumerate(chunk_names):
                    begin = local_mode_index * batch_size
                    end = begin + batch_size
                    feature_maps[mode_name][offset : offset + batch_size] = encoded[
                        begin:end
                    ]

            batch_size = images.shape[0]
            sl = slice(offset, offset + batch_size)
            labels[sl] = batch["label"].cpu().numpy().astype(np.int64)
            native_geometry[sl] = batch["native_geometry"].cpu().numpy().astype(
                np.float32
            )
            standardization_qc[sl] = batch["standardization_qc"].cpu().numpy().astype(
                np.float32
            )
            monai_valid[sl] = segmentation.valid_mask.cpu().numpy().astype(np.bool_)
            monai_area[sl] = segmentation.area_ratio.cpu().numpy().astype(np.float32)
            monai_peak[sl] = segmentation.peak_probability.cpu().numpy().astype(np.float32)

            for local_index in range(batch_size):
                global_index = offset + local_index
                patient_ids[global_index] = str(batch["patient_id"][local_index])
                series_ids[global_index] = str(batch["series_id"][local_index])
                image_paths[global_index] = str(batch["image_path"][local_index])
                pixel_hashes[global_index] = str(
                    batch["decoded_pixel_hash"][local_index]
                )
                phashes[global_index] = str(batch["perceptual_hash"][local_index])
                if global_index in debug_indices:
                    self._save_debug_panel(
                        output_dir=debug_dir,
                        global_index=global_index,
                        patient_id=patient_ids[global_index],
                        label=int(labels[global_index]),
                        full_image=images[local_index],
                        soft_mask=segmentation.soft_probability[local_index],
                        strict_roi=views["strict_roi"][local_index],
                        outside_roi=views["outside_roi"][local_index],
                    )
            offset += batch_size

        if offset != n:
            raise RuntimeError(f"Feature extraction wrote {offset} of {n} rows.")
        for memmap in feature_maps.values():
            memmap.flush()

        np.save(cache_dir / "labels.npy", labels, allow_pickle=False)
        np.save(cache_dir / "patient_ids.npy", np.asarray(patient_ids, dtype=str), allow_pickle=False)
        np.save(cache_dir / "series_ids.npy", np.asarray(series_ids, dtype=str), allow_pickle=False)
        np.save(cache_dir / "sample_indices.npy", sample_indices, allow_pickle=False)
        np.save(cache_dir / "image_paths.npy", np.asarray(image_paths, dtype=str), allow_pickle=False)
        np.save(
            cache_dir / "decoded_pixel_hashes.npy",
            np.asarray(pixel_hashes, dtype=str),
            allow_pickle=False,
        )
        np.save(
            cache_dir / "perceptual_hashes.npy",
            np.asarray(phashes, dtype=str),
            allow_pickle=False,
        )
        np.save(cache_dir / "native_geometry.npy", native_geometry, allow_pickle=False)
        np.save(
            cache_dir / "standardization_qc.npy", standardization_qc, allow_pickle=False
        )
        np.save(cache_dir / "monai_valid.npy", monai_valid, allow_pickle=False)
        np.save(cache_dir / "monai_area_ratio.npy", monai_area, allow_pickle=False)
        np.save(cache_dir / "monai_peak_probability.npy", monai_peak, allow_pickle=False)
        write_json(
            cache_dir / "complete.json",
            {
                "fingerprint": fingerprint,
                "n_slices": n,
                "feature_modes": list(self.config.feature_modes),
                "feature_dim": FrozenEfficientNetEncoder.FEATURE_DIM,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    @staticmethod
    def _save_debug_panel(
        output_dir: Path,
        global_index: int,
        patient_id: str,
        label: int,
        full_image: torch.Tensor,
        soft_mask: torch.Tensor,
        strict_roi: torch.Tensor,
        outside_roi: torch.Tensor,
    ) -> None:
        arrays = [
            full_image.detach().cpu().permute(1, 2, 0).numpy(),
            soft_mask.detach().cpu()[0].numpy(),
            strict_roi.detach().cpu().permute(1, 2, 0).numpy(),
            outside_roi.detach().cpu().permute(1, 2, 0).numpy(),
        ]
        titles = ["Standardized image", "MONAI P(heart)", "Strict ROI", "Outside box"]
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for axis, array, title in zip(axes, arrays, titles):
            axis.imshow(array, cmap="gray" if array.ndim == 2 else None, vmin=0.0, vmax=1.0)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{patient_id} | released label={label} | sample={global_index}")
        figure.tight_layout()
        figure.savefig(output_dir / f"sample_{global_index:06d}.png", dpi=150)
        plt.close(figure)


#%% ============================================================================
# DUPLICATE AUDIT AND PATIENT GROUPS
# =============================================================================


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        value = str(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first != root_second:
            self.parent[max(root_first, root_second)] = min(root_first, root_second)


class HammingBKTree:
    """BK-tree for complete radius search over 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self.root: tuple[int, dict[int, Any]] | None = None

    @staticmethod
    def distance(first: int, second: int) -> int:
        return int((first ^ second).bit_count())

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            node_value, children = node
            distance = self.distance(value, node_value)
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child

    def search(self, query: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        output: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node_value, children = stack.pop()
            distance = self.distance(query, node_value)
            if distance <= radius:
                output.append((node_value, distance))
            lower = distance - radius
            upper = distance + radius
            for edge, child in children.items():
                if lower <= edge <= upper:
                    stack.append(child)
        return output


class DuplicateAuditor:
    """Audit exact decoded pixels and optional pHash candidates across patients."""

    def __init__(self, config: AdmissionsShowcaseConfig, logger: ShowcaseLogger):
        self.config = config
        self.logger = logger

    def audit(
        self,
        bank: FeatureBank,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        exact_rows, exact_summary, patient_edges = self._exact(bank)
        write_csv(output_dir / "exact_duplicate_groups.csv", exact_rows)

        patient_ids = sorted(set(map(str, bank.patient_ids.tolist())))
        union = UnionFind(patient_ids)
        for first, second in patient_edges:
            union.union(first, second)
        patient_to_group = {patient: union.find(patient) for patient in patient_ids}

        phash_summary: dict[str, Any] = {"status": "SKIPPED"}
        if self.config.run_perceptual_duplicate_audit:
            phash_rows, phash_summary = self._perceptual(bank)
            write_csv(output_dir / "perceptual_duplicate_candidates.csv", phash_rows)

        summary = {"exact": exact_summary, "perceptual": phash_summary}
        write_json(output_dir / "duplicate_audit_summary.json", summary)
        self.logger.info(
            "Duplicate audit: %d exact groups, %d exact cross-patient groups, %d pHash patient-pair candidates",
            exact_summary["duplicate_groups"],
            exact_summary["cross_patient_groups"],
            int(phash_summary.get("patient_pair_candidates", 0)),
        )
        return patient_to_group, summary

    @staticmethod
    def _exact(
        bank: FeatureBank,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], set[tuple[str, str]]]:
        groups: dict[str, list[int]] = {}
        for index, digest in enumerate(bank.decoded_pixel_hashes.tolist()):
            groups.setdefault(str(digest), []).append(index)

        rows: list[dict[str, Any]] = []
        patient_edges: set[tuple[str, str]] = set()
        duplicate_groups = 0
        cross_patient_groups = 0
        cross_label_groups = 0
        for digest, indices in groups.items():
            if len(indices) < 2:
                continue
            duplicate_groups += 1
            patients = sorted({str(bank.patient_ids[index]) for index in indices})
            labels = sorted({int(bank.labels[index]) for index in indices})
            if len(patients) > 1:
                cross_patient_groups += 1
                for first_index, first in enumerate(patients):
                    for second in patients[first_index + 1 :]:
                        patient_edges.add(tuple(sorted((first, second))))
            if len(labels) > 1:
                cross_label_groups += 1
            rows.append(
                {
                    "decoded_pixel_sha256": digest,
                    "n_images": len(indices),
                    "n_patients": len(patients),
                    "patient_ids": ";".join(patients),
                    "labels": ";".join(map(str, labels)),
                    "example_paths": ";".join(
                        str(bank.image_paths[index]) for index in indices[:4]
                    ),
                }
            )
        summary = {
            "duplicate_groups": duplicate_groups,
            "cross_patient_groups": cross_patient_groups,
            "cross_label_groups": cross_label_groups,
            "patient_edges": len(patient_edges),
        }
        return rows, summary, patient_edges

    def _perceptual(self, bank: FeatureBank) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # Only one representative image per (pHash, patient) is required to
        # identify patient-pair candidates. This avoids quadratic work when a
        # common blank/localizer pattern occurs many times inside one patient.
        hash_to_patient_representative: dict[int, dict[str, int]] = {}
        for index, value in enumerate(bank.perceptual_hashes.tolist()):
            integer_hash = int(str(value), 16)
            patient = str(bank.patient_ids[index])
            hash_to_patient_representative.setdefault(integer_hash, {}).setdefault(
                patient, index
            )

        unique_hashes = sorted(hash_to_patient_representative)
        tree = HammingBKTree()
        rows_by_patient_pair: dict[tuple[str, str], dict[str, Any]] = {}

        def record_candidate(first_index: int, second_index: int, distance: int) -> None:
            first_patient = str(bank.patient_ids[first_index])
            second_patient = str(bank.patient_ids[second_index])
            if first_patient == second_patient:
                return

            # Keep paths aligned with the lexicographically sorted patient IDs.
            if first_patient <= second_patient:
                patient_1, index_1 = first_patient, first_index
                patient_2, index_2 = second_patient, second_index
            else:
                patient_1, index_1 = second_patient, second_index
                patient_2, index_2 = first_patient, first_index
            pair = (patient_1, patient_2)
            current = rows_by_patient_pair.get(pair)
            candidate = {
                "patient_1": patient_1,
                "patient_2": patient_2,
                "minimum_hamming_distance": int(distance),
                "cross_label": int(
                    int(bank.labels[index_1]) != int(bank.labels[index_2])
                ),
                "example_path_1": str(bank.image_paths[index_1]),
                "example_path_2": str(bank.image_paths[index_2]),
                "manual_review_status": "unreviewed",
                "manual_review_notes": "",
            }
            if current is None or distance < current["minimum_hamming_distance"]:
                rows_by_patient_pair[pair] = candidate

        for value in unique_hashes:
            same_hash_representatives = list(
                hash_to_patient_representative[value].values()
            )
            for first_position, first_index in enumerate(same_hash_representatives):
                for second_index in same_hash_representatives[first_position + 1 :]:
                    record_candidate(first_index, second_index, 0)

            for other, distance in tree.search(value, self.config.phash_hamming_threshold):
                for first_index in same_hash_representatives:
                    for second_index in hash_to_patient_representative[other].values():
                        record_candidate(first_index, second_index, distance)
            tree.add(value)

        rows = sorted(
            rows_by_patient_pair.values(),
            key=lambda row: (
                row["minimum_hamming_distance"],
                row["patient_1"],
                row["patient_2"],
            ),
        )
        summary = {
            "status": "COMPLETE_BK_TREE_SCREENING",
            "unique_hashes": len(unique_hashes),
            "hamming_threshold": self.config.phash_hamming_threshold,
            "patient_pair_candidates": len(rows),
            "cross_label_candidates": int(sum(row["cross_label"] for row in rows)),
            "interpretation": (
                "Screening candidates only; manual or stronger similarity review is required."
            ),
        }
        return rows, summary


#%% ============================================================================
# PATIENT REPRESENTATIONS
# =============================================================================


class PatientRepresentationBuilder:
    """Pool slices to series proxies, then series proxies to Directory_* patients."""

    NATIVE_GEOMETRY_BASE_NAMES = ("height", "width", "aspect_ratio")
    SUMMARY_NAMES = ("mean", "median", "std")

    def __init__(self, bank: FeatureBank):
        self.bank = bank
        self._image_tables: dict[str, PatientTable] = {}
        self._geometry_table: PatientTable | None = None

    @staticmethod
    def _patient_label_map(labels: np.ndarray, patient_ids: np.ndarray) -> dict[str, int]:
        output: dict[str, int] = {}
        for label, patient in zip(labels.tolist(), patient_ids.tolist()):
            patient = str(patient)
            previous = output.get(patient)
            if previous is not None and previous != int(label):
                raise RuntimeError(f"Patient {patient} has conflicting labels.")
            output[patient] = int(label)
        return output

    def image_table(self, mode: str) -> PatientTable:
        cached = self._image_tables.get(mode)
        if cached is not None:
            return cached
        features = np.asarray(self.bank.features[mode])
        patient_ids = np.asarray(self.bank.patient_ids, dtype=str)
        series_ids = np.asarray(self.bank.series_ids, dtype=str)
        labels = np.asarray(self.bank.labels, dtype=np.int64)
        patient_to_label = self._patient_label_map(labels, patient_ids)
        ordered_patients = np.asarray(sorted(patient_to_label), dtype=str)

        pooled: list[np.ndarray] = []
        for patient in ordered_patients:
            patient_mask = patient_ids == patient
            patient_series = np.unique(series_ids[patient_mask])
            series_vectors = []
            for series in patient_series:
                mask = patient_mask & (series_ids == series)
                series_vectors.append(np.mean(features[mask], axis=0, dtype=np.float64))
            pooled.append(
                np.mean(np.stack(series_vectors, axis=0), axis=0, dtype=np.float64).astype(
                    np.float32
                )
            )
        table = PatientTable(
            patient_ids=ordered_patients,
            labels=np.asarray([patient_to_label[p] for p in ordered_patients], dtype=np.int64),
            features=np.stack(pooled, axis=0),
            feature_names=tuple(f"embedding_{index:04d}" for index in range(features.shape[1])),
        )
        self._image_tables[mode] = table
        return table

    def native_geometry_table(self) -> PatientTable:
        if self._geometry_table is not None:
            return self._geometry_table
        patient_ids = np.asarray(self.bank.patient_ids, dtype=str)
        labels = np.asarray(self.bank.labels, dtype=np.int64)
        geometry = np.asarray(self.bank.native_geometry, dtype=np.float64)
        patient_to_label = self._patient_label_map(labels, patient_ids)
        ordered_patients = np.asarray(sorted(patient_to_label), dtype=str)

        rows: list[np.ndarray] = []
        names: list[str] = []
        for base_name in self.NATIVE_GEOMETRY_BASE_NAMES:
            for summary_name in self.SUMMARY_NAMES:
                names.append(f"{base_name}_{summary_name}")

        for patient in ordered_patients:
            values = geometry[patient_ids == patient]
            patient_row: list[float] = []
            for column in range(values.shape[1]):
                column_values = values[:, column]
                patient_row.extend(
                    [
                        float(np.mean(column_values)),
                        float(np.median(column_values)),
                        float(np.std(column_values)),
                    ]
                )
            rows.append(np.asarray(patient_row, dtype=np.float32))

        self._geometry_table = PatientTable(
            patient_ids=ordered_patients,
            labels=np.asarray([patient_to_label[p] for p in ordered_patients], dtype=np.int64),
            features=np.stack(rows, axis=0),
            feature_names=tuple(names),
        )
        return self._geometry_table

    def for_experiment(self, spec: ExperimentSpec) -> PatientTable:
        if spec.feature_mode == "native_geometry_only":
            return self.native_geometry_table()
        return self.image_table(spec.feature_mode)


#%% ============================================================================
# PATIENT FOLDS AND NESTED EVALUATION
# =============================================================================


class PatientFoldManager:
    """Create stratified folds while keeping exact-duplicate components together."""

    def __init__(self, config: AdmissionsShowcaseConfig):
        self.config = config

    def assign(
        self,
        patient_ids: np.ndarray,
        labels: np.ndarray,
        patient_to_group: Mapping[str, str],
        n_splits: int,
        random_state: int,
    ) -> np.ndarray:
        patient_ids = np.asarray(patient_ids, dtype=str)
        labels = np.asarray(labels, dtype=np.int64)
        groups = np.asarray([patient_to_group[str(patient)] for patient in patient_ids])

        def validate_assignment(folds: np.ndarray) -> bool:
            if np.any(folds == 0):
                return False
            for fold in range(1, n_splits + 1):
                validation_labels = labels[folds == fold]
                training_labels = labels[folds != fold]
                if len(np.unique(validation_labels)) != 2:
                    return False
                if len(np.unique(training_labels)) != 2:
                    return False
            return True

        if len(np.unique(groups)) == len(patient_ids):
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            folds = np.zeros(len(patient_ids), dtype=np.int64)
            for fold, (_, valid) in enumerate(splitter.split(patient_ids, labels), start=1):
                folds[valid] = fold
            if not validate_assignment(folds):
                raise RuntimeError("Stratified patient folds do not contain both classes.")
            return folds

        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except ImportError as exc:
            raise RuntimeError(
                "Cross-patient exact duplicates require StratifiedGroupKFold."
            ) from exc

        # Group constraints can make one random split invalid even when another
        # split is feasible. Try a bounded sequence of deterministic seeds and
        # accept only a manifest with both classes in every train/validation fold.
        for attempt in range(100):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state + attempt,
            )
            folds = np.zeros(len(patient_ids), dtype=np.int64)
            for fold, (_, valid) in enumerate(
                splitter.split(patient_ids, labels, groups=groups), start=1
            ):
                folds[valid] = fold
            if validate_assignment(folds):
                return folds
        raise RuntimeError(
            "Could not create duplicate-aware stratified folds containing both "
            "classes after 100 deterministic attempts."
        )


class NestedPatientEvaluator:
    """Nested CV with training-only C selection and training-only thresholds."""

    def __init__(
        self,
        config: AdmissionsShowcaseConfig,
        logger: ShowcaseLogger,
        fold_manager: PatientFoldManager,
        patient_to_group: Mapping[str, str],
    ) -> None:
        self.config = config
        self.logger = logger
        self.fold_manager = fold_manager
        self.patient_to_group = dict(patient_to_group)

    @staticmethod
    def _balanced_weights(labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        if int(counts.min()) <= 0:
            raise RuntimeError("A training partition must contain both classes.")
        return np.asarray(
            [len(labels) / (2.0 * counts[int(label)]) for label in labels],
            dtype=np.float64,
        )

    def _build_model(self, spec: ExperimentSpec, c_value: float) -> Pipeline:
        steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
        if spec.use_pca:
            steps.append(
                (
                    "pca",
                    PCA(
                        n_components=self.config.pca_explained_variance,
                        svd_solver="full",
                    ),
                )
            )
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    solver="liblinear",
                    max_iter=self.config.logistic_max_iter,
                    random_state=self.config.random_seed,
                ),
            )
        )
        return Pipeline(steps)

    def _fit(self, model: Pipeline, features: np.ndarray, labels: np.ndarray) -> Pipeline:
        weights = self._balanced_weights(labels)
        model.fit(features, labels, classifier__sample_weight=weights)
        return model

    @staticmethod
    def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            labels, probabilities
        )
        finite = np.isfinite(thresholds)
        if not np.any(finite):
            return 0.5
        false_positive_rate = false_positive_rate[finite]
        true_positive_rate = true_positive_rate[finite]
        thresholds = thresholds[finite]
        youden = true_positive_rate - false_positive_rate
        best = float(np.max(youden))
        candidates = np.flatnonzero(np.isclose(youden, best, rtol=0.0, atol=1e-12))
        return float(np.max(thresholds[candidates]))

    def _inner_oof(
        self,
        spec: ExperimentSpec,
        table: PatientTable,
        train_indices: np.ndarray,
        c_value: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        train_patients = table.patient_ids[train_indices]
        train_labels = table.labels[train_indices]
        maximum_splits = min(
            self.config.inner_splits,
            int(np.bincount(train_labels, minlength=2).min()),
        )
        if maximum_splits < 2:
            raise RuntimeError("Not enough training patients for inner CV.")
        inner_folds = self.fold_manager.assign(
            train_patients,
            train_labels,
            self.patient_to_group,
            n_splits=maximum_splits,
            random_state=seed,
        )
        scores = np.full(len(train_indices), np.nan, dtype=np.float64)
        for fold in range(1, maximum_splits + 1):
            inner_train_local = np.flatnonzero(inner_folds != fold)
            inner_valid_local = np.flatnonzero(inner_folds == fold)
            model = self._build_model(spec, c_value)
            self._fit(
                model,
                table.features[train_indices[inner_train_local]],
                table.labels[train_indices[inner_train_local]],
            )
            scores[inner_valid_local] = model.predict_proba(
                table.features[train_indices[inner_valid_local]]
            )[:, 1]
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Inner OOF scoring left non-finite values.")
        return train_labels, scores

    def _select_c_and_threshold(
        self,
        spec: ExperimentSpec,
        table: PatientTable,
        train_indices: np.ndarray,
        seed: int,
    ) -> tuple[float, float, float, list[dict[str, float]]]:
        candidates: list[dict[str, Any]] = []
        for c_value in sorted(set(map(float, self.config.classifier_c_grid))):
            labels, scores = self._inner_oof(
                spec,
                table,
                train_indices,
                c_value,
                seed,
            )
            auc = float(roc_auc_score(labels, scores))
            candidates.append(
                {"c": c_value, "auc": auc, "labels": labels, "scores": scores}
            )
            self.logger.debug(
                "%s inner CV: C=%g, AUC=%.4f", spec.name, c_value, auc
            )
        best_auc = max(float(candidate["auc"]) for candidate in candidates)
        eligible = [
            candidate
            for candidate in candidates
            if float(candidate["auc"])
            >= best_auc - self.config.c_selection_auc_tolerance
        ]
        selected = sorted(eligible, key=lambda candidate: float(candidate["c"]))[0]
        threshold = self._select_threshold(selected["labels"], selected["scores"])
        compact = [
            {"c": float(candidate["c"]), "inner_auc": float(candidate["auc"])}
            for candidate in candidates
        ]
        return (
            float(selected["c"]),
            float(threshold),
            float(selected["auc"]),
            compact,
        )

    @staticmethod
    def _metrics(
        labels: np.ndarray,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        labels = np.asarray(labels, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.int64)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        def divide(num: float, den: float) -> float:
            return float(num / den) if den else float("nan")

        sensitivity = divide(tp, tp + fn)
        specificity = divide(tn, tn + fp)
        ppv = divide(tp, tp + fp)
        npv = divide(tn, tn + fn)
        accuracy = divide(tp + tn, len(labels))
        return {
            "auc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "accuracy": accuracy,
            "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }

    def _bootstrap_intervals(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        seed: int,
    ) -> dict[str, tuple[float, float]]:
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels, dtype=np.int64)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        metric_names = (
            "auc",
            "auprc",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "accuracy",
            "balanced_accuracy",
        )
        values = {name: [] for name in metric_names}
        for _ in range(self.config.effective_bootstrap_replicates):
            sampled = np.concatenate(
                [
                    rng.choice(negative, size=len(negative), replace=True),
                    rng.choice(positive, size=len(positive), replace=True),
                ]
            )
            metrics = self._metrics(
                labels[sampled], probabilities[sampled], predictions[sampled]
            )
            for name in metric_names:
                if np.isfinite(metrics[name]):
                    values[name].append(metrics[name])
        return {
            name: (
                float(np.quantile(values[name], 0.025)),
                float(np.quantile(values[name], 0.975)),
            )
            for name in metric_names
            if values[name]
        }

    def evaluate(
        self,
        spec: ExperimentSpec,
        table: PatientTable,
        outer_folds: np.ndarray,
        seed: int,
        compute_intervals: bool = True,
        announce: bool = True,
    ) -> ExperimentResult:
        if announce:
            self.logger.info("Evaluating %s...", spec.display_name)
        n = len(table.patient_ids)
        probabilities = np.full(n, np.nan, dtype=np.float64)
        predictions = np.full(n, -1, dtype=np.int64)
        selected_c = np.full(n, np.nan, dtype=np.float64)
        thresholds = np.full(n, np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []

        for fold in sorted(np.unique(outer_folds).tolist()):
            train_indices = np.flatnonzero(outer_folds != fold)
            valid_indices = np.flatnonzero(outer_folds == fold)
            c_value, threshold, inner_auc, candidates = self._select_c_and_threshold(
                spec,
                table,
                train_indices,
                seed + 1000 * int(fold),
            )
            model = self._build_model(spec, c_value)
            self._fit(model, table.features[train_indices], table.labels[train_indices])
            fold_probabilities = model.predict_proba(table.features[valid_indices])[:, 1]
            fold_predictions = (fold_probabilities >= threshold).astype(np.int64)
            probabilities[valid_indices] = fold_probabilities
            predictions[valid_indices] = fold_predictions
            selected_c[valid_indices] = c_value
            thresholds[valid_indices] = threshold
            fold_auc = float(
                roc_auc_score(table.labels[valid_indices], fold_probabilities)
            )
            fold_rows.append(
                {
                    "outer_fold": int(fold),
                    "train_patients": len(train_indices),
                    "valid_patients": len(valid_indices),
                    "selected_c": c_value,
                    "training_only_threshold": threshold,
                    "selected_inner_auc": inner_auc,
                    "held_out_auc": fold_auc,
                    "candidate_c_results_json": json.dumps(candidates),
                }
            )

        if not np.all(np.isfinite(probabilities)) or np.any(predictions < 0):
            raise RuntimeError(f"Incomplete OOF result for {spec.name}.")
        metrics = self._metrics(table.labels, probabilities, predictions)
        intervals = (
            self._bootstrap_intervals(
                table.labels,
                probabilities,
                predictions,
                seed=seed + 90_000,
            )
            if compute_intervals
            else {}
        )
        if announce:
            self.logger.info(
                "%s: AUROC %.4f, sensitivity %.3f, specificity %.3f",
                spec.display_name,
                metrics["auc"],
                metrics["sensitivity"],
                metrics["specificity"],
            )
        return ExperimentResult(
            spec=spec,
            patient_ids=table.patient_ids.copy(),
            labels=table.labels.copy(),
            probabilities=probabilities,
            predictions=predictions,
            outer_folds=np.asarray(outer_folds, dtype=np.int64),
            selected_c=selected_c,
            thresholds=thresholds,
            metrics=metrics,
            intervals=intervals,
            fold_rows=fold_rows,
        )


#%% ============================================================================
# STABILITY, PERMUTATION, AND PAIRED COMPARISONS
# =============================================================================


class RobustnessAnalyzer:
    """Repeated split stability and patient-label permutation for the locked model."""

    STABILITY_EXPERIMENTS = (
        "strict_roi",
        "fixed_center_crop",
        "outside_roi",
        "soft_mask_only",
    )

    def __init__(
        self,
        config: AdmissionsShowcaseConfig,
        logger: ShowcaseLogger,
        fold_manager: PatientFoldManager,
        evaluator: NestedPatientEvaluator,
        patient_to_group: Mapping[str, str],
    ) -> None:
        self.config = config
        self.logger = logger
        self.fold_manager = fold_manager
        self.evaluator = evaluator
        self.patient_to_group = dict(patient_to_group)

    def repeated_cv(
        self,
        specs: Mapping[str, ExperimentSpec],
        tables: Mapping[str, PatientTable],
        output_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        repeats = self.config.effective_repeated_cv_repeats
        self.logger.info(
            "Repeated nested CV: %d seeds × %d key experiments",
            repeats,
            len(self.STABILITY_EXPERIMENTS),
        )
        rows: list[dict[str, Any]] = []
        for repeat in range(repeats):
            seed = self.config.random_seed + 20_000 + repeat
            reference_table = tables["strict_roi"]
            folds = self.fold_manager.assign(
                reference_table.patient_ids,
                reference_table.labels,
                self.patient_to_group,
                self.config.outer_splits,
                seed,
            )
            for name in self.STABILITY_EXPERIMENTS:
                result = self.evaluator.evaluate(
                    specs[name],
                    tables[name],
                    folds,
                    seed=seed,
                    compute_intervals=False,
                    announce=False,
                )
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "seed": seed,
                        "experiment": name,
                        "auc": result.metrics["auc"],
                        "auprc": result.metrics["auprc"],
                        "sensitivity": result.metrics["sensitivity"],
                        "specificity": result.metrics["specificity"],
                    }
                )
            if (repeat + 1) % max(1, repeats // 5) == 0 or repeat + 1 == repeats:
                self.logger.info(
                    "Repeated CV progress: %d/%d seeds", repeat + 1, repeats
                )

        summaries: list[dict[str, Any]] = []
        for name in self.STABILITY_EXPERIMENTS:
            values = np.asarray(
                [row["auc"] for row in rows if row["experiment"] == name],
                dtype=np.float64,
            )
            summaries.append(
                {
                    "experiment": name,
                    "n_repeats": len(values),
                    "mean_auc": float(np.mean(values)),
                    "median_auc": float(np.median(values)),
                    "std_auc": float(np.std(values)),
                    "q1_auc": float(np.quantile(values, 0.25)),
                    "q3_auc": float(np.quantile(values, 0.75)),
                    "min_auc": float(np.min(values)),
                    "max_auc": float(np.max(values)),
                }
            )

        # Same-seed paired deltas against the locked strict ROI.
        paired_rows: list[dict[str, Any]] = []
        strict_by_repeat = {
            int(row["repeat"]): float(row["auc"])
            for row in rows
            if row["experiment"] == "strict_roi"
        }
        for name in self.STABILITY_EXPERIMENTS:
            if name == "strict_roi":
                continue
            deltas = np.asarray(
                [
                    strict_by_repeat[int(row["repeat"])] - float(row["auc"])
                    for row in rows
                    if row["experiment"] == name
                ],
                dtype=np.float64,
            )
            paired_rows.append(
                {
                    "comparison": f"strict_roi_minus_{name}",
                    "median_delta_auc": float(np.median(deltas)),
                    "mean_delta_auc": float(np.mean(deltas)),
                    "q1_delta_auc": float(np.quantile(deltas, 0.25)),
                    "q3_delta_auc": float(np.quantile(deltas, 0.75)),
                    "p2_5_delta_auc": float(np.quantile(deltas, 0.025)),
                    "p97_5_delta_auc": float(np.quantile(deltas, 0.975)),
                    "fraction_strict_roi_better": float(np.mean(deltas > 0)),
                    "fraction_tied": float(np.mean(np.isclose(deltas, 0.0))),
                }
            )

        write_csv(output_dir / "repeated_nested_cv_runs.csv", rows)
        write_csv(output_dir / "repeated_nested_cv_summary.csv", summaries)
        write_csv(output_dir / "paired_repeated_cv_deltas.csv", paired_rows)
        for row in summaries:
            self.logger.info(
                "Stability %-20s median AUROC %.4f [IQR %.4f–%.4f]",
                row["experiment"],
                row["median_auc"],
                row["q1_auc"],
                row["q3_auc"],
            )
        return rows, paired_rows

    def permutation_test(
        self,
        strict_spec: ExperimentSpec,
        strict_table: PatientTable,
        observed_auc: float,
        output_dir: Path,
    ) -> dict[str, Any]:
        replicates = self.config.effective_permutation_replicates
        self.logger.info("Patient-label permutation test: %d replicates", replicates)
        rng = np.random.default_rng(self.config.random_seed + 40_000)
        null_aucs: list[float] = []
        for replicate in range(replicates):
            permuted_labels = rng.permutation(strict_table.labels)
            permuted_table = PatientTable(
                patient_ids=strict_table.patient_ids,
                labels=np.asarray(permuted_labels, dtype=np.int64),
                features=strict_table.features,
                feature_names=strict_table.feature_names,
            )
            seed = self.config.random_seed + 50_000 + replicate
            folds = self.fold_manager.assign(
                permuted_table.patient_ids,
                permuted_table.labels,
                self.patient_to_group,
                self.config.outer_splits,
                seed,
            )
            result = self.evaluator.evaluate(
                strict_spec,
                permuted_table,
                folds,
                seed=seed,
                compute_intervals=False,
                announce=False,
            )
            null_aucs.append(float(result.metrics["auc"]))
            if (replicate + 1) % max(1, replicates // 10) == 0:
                self.logger.info(
                    "Permutation progress: %d/%d", replicate + 1, replicates
                )

        null_array = np.asarray(null_aucs, dtype=np.float64)
        empirical_p = float((1 + np.sum(null_array >= observed_auc)) / (replicates + 1))
        rows = [
            {"replicate": index + 1, "null_auc": value}
            for index, value in enumerate(null_aucs)
        ]
        summary = {
            "experiment": strict_spec.name,
            "observed_auc": float(observed_auc),
            "replicates": replicates,
            "null_mean": float(np.mean(null_array)),
            "null_median": float(np.median(null_array)),
            "null_std": float(np.std(null_array)),
            "null_p2_5": float(np.quantile(null_array, 0.025)),
            "null_p97_5": float(np.quantile(null_array, 0.975)),
            "empirical_one_sided_p": empirical_p,
            "interpretation": (
                "The test establishes association with released labels, not CAD-specific anatomy."
            ),
        }
        write_csv(output_dir / "patient_label_permutation_auc.csv", rows)
        write_json(output_dir / "patient_label_permutation_summary.json", summary)
        self.logger.info(
            "Permutation result: observed AUROC %.4f, null median %.4f, p=%.6f",
            observed_auc,
            summary["null_median"],
            empirical_p,
        )
        return summary


class PairedAUCComparator:
    """Patient-level paired bootstrap differences against strict ROI."""

    def __init__(self, config: AdmissionsShowcaseConfig):
        self.config = config

    def compare(self, results: Mapping[str, ExperimentResult]) -> list[dict[str, Any]]:
        reference = results["strict_roi"]
        positive = np.flatnonzero(reference.labels == 1)
        negative = np.flatnonzero(reference.labels == 0)
        rows: list[dict[str, Any]] = []
        for name, result in results.items():
            if name == "strict_roi":
                continue
            if not np.array_equal(reference.patient_ids, result.patient_ids):
                raise RuntimeError("Paired comparison patient orders differ.")
            rng = np.random.default_rng(self.config.random_seed + 70_000 + len(rows))
            deltas: list[float] = []
            for _ in range(self.config.effective_bootstrap_replicates):
                sampled = np.concatenate(
                    [
                        rng.choice(negative, size=len(negative), replace=True),
                        rng.choice(positive, size=len(positive), replace=True),
                    ]
                )
                ref_auc = roc_auc_score(
                    reference.labels[sampled], reference.probabilities[sampled]
                )
                cmp_auc = roc_auc_score(
                    result.labels[sampled], result.probabilities[sampled]
                )
                deltas.append(float(ref_auc - cmp_auc))
            rows.append(
                {
                    "reference": "strict_roi",
                    "comparison": name,
                    "strict_roi_auc": reference.metrics["auc"],
                    "comparison_auc": result.metrics["auc"],
                    "delta_auc_strict_minus_comparison": (
                        reference.metrics["auc"] - result.metrics["auc"]
                    ),
                    "delta_ci_low": float(np.quantile(deltas, 0.025)),
                    "delta_ci_high": float(np.quantile(deltas, 0.975)),
                }
            )
        return rows


#%% ============================================================================
# REPORTING
# =============================================================================


class AdmissionsReporter:
    """Write compact machine-readable outputs and four presentation figures."""

    def __init__(
        self,
        config: AdmissionsShowcaseConfig,
        logger: ShowcaseLogger,
        run_dir: Path,
    ) -> None:
        self.config = config
        self.logger = logger
        self.run_dir = run_dir

    def save_cohort_manifest(self, samples: Sequence[MRISample]) -> None:
        rows = [
            {
                "sample_index": sample.sample_index,
                "patient_id": sample.patient_id,
                "series_proxy": sample.series_id,
                "released_label": sample.label,
                "image_path": str(sample.path),
                "file_size_bytes": sample.file_size_bytes,
            }
            for sample in samples
        ]
        write_csv(self.run_dir / "cohort_manifest.csv", rows)

    def save_fold_manifest(
        self,
        patient_ids: np.ndarray,
        labels: np.ndarray,
        folds: np.ndarray,
        patient_to_group: Mapping[str, str],
    ) -> None:
        rows = [
            {
                "patient_id": str(patient),
                "released_label": int(label),
                "outer_fold": int(fold),
                "exact_duplicate_component": patient_to_group[str(patient)],
            }
            for patient, label, fold in zip(patient_ids, labels, folds)
        ]
        write_csv(self.run_dir / "patient_fold_manifest.csv", rows)

    def save_results(
        self,
        results: Mapping[str, ExperimentResult],
        paired_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        summary_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        for name, result in results.items():
            auc_interval = result.intervals.get("auc", (float("nan"), float("nan")))
            summary_rows.append(
                {
                    "experiment": name,
                    "display_name": result.spec.display_name,
                    "role": result.spec.role,
                    "historical_v4_id": result.spec.historical_id,
                    "auc": result.metrics["auc"],
                    "auc_ci_low": auc_interval[0],
                    "auc_ci_high": auc_interval[1],
                    "auprc": result.metrics["auprc"],
                    "sensitivity": result.metrics["sensitivity"],
                    "specificity": result.metrics["specificity"],
                    "ppv": result.metrics["ppv"],
                    "npv": result.metrics["npv"],
                    "f1": result.metrics["f1"],
                    "accuracy": result.metrics["accuracy"],
                    "balanced_accuracy": result.metrics["balanced_accuracy"],
                }
            )
            for index, patient in enumerate(result.patient_ids):
                prediction_rows.append(
                    {
                        "experiment": name,
                        "patient_id": str(patient),
                        "released_label": int(result.labels[index]),
                        "outer_fold": int(result.outer_folds[index]),
                        "oof_probability": float(result.probabilities[index]),
                        "training_only_threshold": float(result.thresholds[index]),
                        "prediction": int(result.predictions[index]),
                        "selected_c": float(result.selected_c[index]),
                    }
                )
            write_csv(
                self.run_dir / "results" / name / "fold_metrics.csv",
                result.fold_rows,
            )
        write_csv(self.run_dir / "results" / "experiment_summary.csv", summary_rows)
        write_csv(
            self.run_dir / "results" / "patient_oof_predictions.csv",
            prediction_rows,
        )
        write_csv(
            self.run_dir / "results" / "paired_auc_vs_strict_roi.csv",
            list(paired_rows),
        )

    def save_figures(
        self,
        results: Mapping[str, ExperimentResult],
        stability_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        figure_dir = self.run_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        ordered = [spec.name for spec in self.config.experiments]

        # 1. Main AUROC comparison with patient-bootstrap intervals.
        labels = [results[name].spec.display_name for name in ordered]
        aucs = np.asarray([results[name].metrics["auc"] for name in ordered])
        lows = np.asarray([results[name].intervals["auc"][0] for name in ordered])
        highs = np.asarray([results[name].intervals["auc"][1] for name in ordered])
        figure, axis = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(ordered))
        axis.bar(positions, aucs)
        axis.errorbar(
            positions,
            aucs,
            yerr=np.vstack([aucs - lows, highs - aucs]),
            fmt="none",
            capsize=4,
        )
        axis.axhline(0.5, linestyle="--", linewidth=1)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Patient-level AUROC")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_title("Six predeclared showcase experiments")
        figure.tight_layout()
        figure.savefig(figure_dir / "main_auc_comparison.png", dpi=180)
        plt.close(figure)

        # 2. Pooled OOF ROC curves.
        figure, axis = plt.subplots(figsize=(7, 6))
        for name in ordered:
            result = results[name]
            fpr, tpr, _ = roc_curve(result.labels, result.probabilities)
            axis.plot(fpr, tpr, label=f"{result.spec.display_name} ({result.metrics['auc']:.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.set_title("Pooled out-of-fold ROC curves")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(figure_dir / "oof_roc_curves.png", dpi=180)
        plt.close(figure)

        # 3. Same-seed repeated-CV stability.
        stability_names = RobustnessAnalyzer.STABILITY_EXPERIMENTS
        stability_values = [
            [float(row["auc"]) for row in stability_rows if row["experiment"] == name]
            for name in stability_names
        ]
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.boxplot(stability_values, labels=[results[name].spec.display_name for name in stability_names])
        axis.axhline(0.5, linestyle="--", linewidth=1)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("AUROC")
        axis.set_title("Repeated nested-CV split stability")
        axis.tick_params(axis="x", labelrotation=20)
        figure.tight_layout()
        figure.savefig(figure_dir / "repeated_cv_stability.png", dpi=180)
        plt.close(figure)

        # 4. Confusion matrix for the locked strict ROI using fold-specific,
        # training-only thresholds.
        strict = results["strict_roi"]
        matrix = confusion_matrix(strict.labels, strict.predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_xticks([0, 1], labels=["Predicted Normal", "Predicted Sick"])
        axis.set_yticks([0, 1], labels=["True Normal", "True Sick"])
        axis.set_title("Strict ROI nested-CV confusion matrix")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(figure_dir / "strict_roi_confusion_matrix.png", dpi=180)
        plt.close(figure)

        # A small text figure is not needed; the permutation result is written
        # into the generated Markdown summary below.

    def write_markdown_summary(
        self,
        results: Mapping[str, ExperimentResult],
        stability_rows: Sequence[Mapping[str, Any]],
        paired_stability: Sequence[Mapping[str, Any]],
        paired_auc_rows: Sequence[Mapping[str, Any]],
        permutation_summary: Mapping[str, Any],
        duplicate_summary: Mapping[str, Any],
    ) -> None:
        strict = results["strict_roi"]
        strict_stability = np.asarray(
            [
                float(row["auc"])
                for row in stability_rows
                if row["experiment"] == "strict_roi"
            ],
            dtype=np.float64,
        )
        lines = [
            "# CAD Cardiac MRI — Admissions Showcase Result",
            "",
            "## Research question",
            "",
            "Can a patient-level model separate the released Normal/Sick labels while relying more on cardiac-region information than on simple cropping or export geometry?",
            "",
            "## Locked primary result",
            "",
            f"- Patient-level AUROC: **{strict.metrics['auc']:.4f}** "
            f"(95% bootstrap CI {strict.intervals['auc'][0]:.4f}–{strict.intervals['auc'][1]:.4f})",
            f"- Sensitivity: **{strict.metrics['sensitivity']:.4f}**",
            f"- Specificity: **{strict.metrics['specificity']:.4f}**",
            f"- Repeated nested-CV median AUROC: **{np.median(strict_stability):.4f}** "
            f"(IQR {np.quantile(strict_stability, 0.25):.4f}–{np.quantile(strict_stability, 0.75):.4f})",
            f"- Patient-label permutation p-value: **{float(permutation_summary['empirical_one_sided_p']):.6f}**",
            "",
            "## Six-experiment comparison",
            "",
            "| Experiment | Role | AUROC | Sensitivity | Specificity |",
            "|---|---|---:|---:|---:|",
        ]
        for spec in self.config.experiments:
            result = results[spec.name]
            lines.append(
                f"| {spec.display_name} | {spec.role} | {result.metrics['auc']:.4f} | "
                f"{result.metrics['sensitivity']:.4f} | {result.metrics['specificity']:.4f} |"
            )
        lines.extend(
            [
                "",
                "## Methodological safeguards retained",
                "",
                "- `Directory_*` is the patient unit; no patient's slices cross folds.",
                "- C and the decision threshold are selected only from inner out-of-fold training predictions.",
                "- MONAI and EfficientNet remain frozen; fold-local StandardScaler, PCA, and Logistic Regression are the fitted components.",
                "- Exact decoded-pixel duplicates are audited across patients; pHash results are screening candidates requiring review.",
                "- Repeated nested CV reports split sensitivity; the label-permutation test checks whether the observed association exceeds chance.",
                "",
                "## Important limitation",
                "",
                "The released cohort contains a small number of computational `Directory_*` patient units and exhibits protocol/export signal. This analysis establishes association with the released labels, not clinical validation or proof of CAD-specific causality. External validation and sequence/view matching remain necessary.",
                "",
                "## Duplicate audit",
                "",
                f"- Exact cross-patient duplicate groups: **{duplicate_summary['exact']['cross_patient_groups']}**",
                f"- pHash patient-pair candidates: **{duplicate_summary['perceptual'].get('patient_pair_candidates', 0)}**",
                f"- pHash cross-label candidates: **{duplicate_summary['perceptual'].get('cross_label_candidates', 0)}**",
                "",
                "## Main-split paired bootstrap comparisons",
                "",
                "Positive values favor strict ROI.",
                "",
                "| Comparison | ΔAUROC | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for row in paired_auc_rows:
            lines.append(
                f"| strict_roi vs {row['comparison']} | "
                f"{float(row['delta_auc_strict_minus_comparison']):+.4f} | "
                f"[{float(row['delta_ci_low']):+.4f}, {float(row['delta_ci_high']):+.4f}] |"
            )
        lines.extend(
            [
                "",
                "## Repeated-CV paired deltas",
                "",
                "Positive values favor strict ROI.",
                "",
                "| Comparison | Median ΔAUROC | Strict ROI better |",
                "|---|---:|---:|",
            ]
        )
        for row in paired_stability:
            lines.append(
                f"| {row['comparison']} | {float(row['median_delta_auc']):+.4f} | "
                f"{float(row['fraction_strict_roi_better']):.1%} |"
            )
        lines.extend(
            [
                "",
                "## Files",
                "",
                "- `results/experiment_summary.csv`",
                "- `results/patient_oof_predictions.csv`",
                "- `stability/repeated_nested_cv_summary.csv`",
                "- `permutation/patient_label_permutation_summary.json`",
                "- `figures/main_auc_comparison.png`",
                "- `figures/oof_roc_curves.png`",
                "- `figures/repeated_cv_stability.png`",
                "- `figures/strict_roi_confusion_matrix.png`",
            ]
        )
        (self.run_dir / "ADMISSIONS_SUMMARY.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def print_final_table(self, results: Mapping[str, ExperimentResult]) -> None:
        print("\n" + "=" * 100)
        print("CAD CARDIAC MRI — ADMISSIONS SHOWCASE")
        print("=" * 100)
        print(
            f"{'Experiment':<30} {'Role':<37} {'AUROC':>8} {'Sens.':>8} {'Spec.':>8}"
        )
        print("-" * 100)
        for spec in self.config.experiments:
            result = results[spec.name]
            print(
                f"{spec.display_name:<30} {spec.role:<37} "
                f"{result.metrics['auc']:>8.4f} "
                f"{result.metrics['sensitivity']:>8.3f} "
                f"{result.metrics['specificity']:>8.3f}"
            )
        print("-" * 100)
        print("All rows use the same patient-fold manifest. C and thresholds are training-only.")
        print("=" * 100)


#%% ============================================================================
# ORCHESTRATOR
# =============================================================================


class CADMRIAdmissionsShowcase:
    """One readable class coordinates the complete single-file Kaggle workflow."""

    def __init__(self, config: AdmissionsShowcaseConfig | None = None) -> None:
        self.config = config or AdmissionsShowcaseConfig()
        self.config.validate()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        config_digest = hashlib.sha256(
            json.dumps(
                {
                    "profile": self.config.profile,
                    "seed": self.config.random_seed,
                    "experiments": [spec.name for spec in self.config.experiments],
                    "cache_schema": self.config.feature_cache_schema,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        self.run_dir = self.config.output_root / f"showcase__{timestamp}__{config_digest}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ShowcaseLogger(self.run_dir, self.config.detailed_console)

    def _seed_everything(self) -> None:
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def _save_configuration(self) -> None:
        config_dict = asdict(self.config)
        for key in ("dataset_path", "output_root", "monai_bundle_parent"):
            config_dict[key] = str(config_dict[key])
        config_dict["effective_bootstrap_replicates"] = (
            self.config.effective_bootstrap_replicates
        )
        config_dict["effective_repeated_cv_repeats"] = (
            self.config.effective_repeated_cv_repeats
        )
        config_dict["effective_permutation_replicates"] = (
            self.config.effective_permutation_replicates
        )
        config_dict["experiments"] = [asdict(spec) for spec in self.config.experiments]
        write_json(self.run_dir / "configuration.json", config_dict)

    def run(self) -> dict[str, Any]:
        total_started = time.perf_counter()
        self._seed_everything()
        self._save_configuration()
        self.logger.info("Dataset: %s", self.config.dataset_path)
        self.logger.info("Output:  %s", self.run_dir)
        self.logger.info("Device:  %s", self.config.device)
        if self.config.device == "cuda":
            self.logger.info("GPU:     %s", torch.cuda.get_device_name(0))
        self.logger.info(
            "Profile: %s (%d repeated CV runs, %d permutations)",
            self.config.profile,
            self.config.effective_repeated_cv_repeats,
            self.config.effective_permutation_replicates,
        )

        reporter = AdmissionsReporter(self.config, self.logger, self.run_dir)

        with Timer(self.logger, "Discover Directory_* cohort"):
            samples = DatasetScanner(self.config, self.logger).discover()
            reporter.save_cohort_manifest(samples)

        with Timer(self.logger, "Load or build five-view frozen feature bank"):
            bank, cache_hit = FeatureBankBuilder(
                self.config, self.logger, self.run_dir
            ).load_or_build(samples)

        with Timer(self.logger, "Audit exact and perceptual duplicates"):
            patient_to_group, duplicate_summary = DuplicateAuditor(
                self.config, self.logger
            ).audit(bank, self.run_dir / "audits")

        with Timer(self.logger, "Build one vector per Directory_* patient"):
            representation_builder = PatientRepresentationBuilder(bank)
            specs = {spec.name: spec for spec in self.config.experiments}
            tables = {
                name: representation_builder.for_experiment(spec)
                for name, spec in specs.items()
            }
            reference = tables["strict_roi"]
            for name, table in tables.items():
                if not np.array_equal(reference.patient_ids, table.patient_ids):
                    raise RuntimeError(f"Patient order mismatch in {name}.")
                if not np.array_equal(reference.labels, table.labels):
                    raise RuntimeError(f"Patient labels mismatch in {name}.")

        fold_manager = PatientFoldManager(self.config)
        outer_folds = fold_manager.assign(
            reference.patient_ids,
            reference.labels,
            patient_to_group,
            self.config.outer_splits,
            self.config.random_seed,
        )
        reporter.save_fold_manifest(
            reference.patient_ids,
            reference.labels,
            outer_folds,
            patient_to_group,
        )

        evaluator = NestedPatientEvaluator(
            self.config,
            self.logger,
            fold_manager,
            patient_to_group,
        )
        results: dict[str, ExperimentResult] = {}
        with Timer(self.logger, "Run six predeclared nested-CV experiments"):
            for spec in self.config.experiments:
                results[spec.name] = evaluator.evaluate(
                    spec,
                    tables[spec.name],
                    outer_folds,
                    seed=self.config.random_seed,
                )

        paired_auc_rows = PairedAUCComparator(self.config).compare(results)
        reporter.save_results(results, paired_auc_rows)

        robustness = RobustnessAnalyzer(
            self.config,
            self.logger,
            fold_manager,
            evaluator,
            patient_to_group,
        )
        with Timer(self.logger, "Run repeated nested-CV split stability"):
            stability_rows, paired_stability = robustness.repeated_cv(
                specs,
                tables,
                self.run_dir / "stability",
            )

        with Timer(self.logger, "Run patient-label permutation sanity test"):
            permutation_summary = robustness.permutation_test(
                specs["strict_roi"],
                tables["strict_roi"],
                observed_auc=results["strict_roi"].metrics["auc"],
                output_dir=self.run_dir / "permutation",
            )

        with Timer(self.logger, "Create presentation figures and summary"):
            reporter.save_figures(results, stability_rows)
            reporter.write_markdown_summary(
                results,
                stability_rows,
                paired_stability,
                paired_auc_rows,
                permutation_summary,
                duplicate_summary,
            )
            reporter.print_final_table(results)

        total_runtime = time.perf_counter() - total_started
        metadata = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "profile": self.config.profile,
            "dataset_path": str(self.config.dataset_path),
            "output_dir": str(self.run_dir),
            "cache_hit": bool(cache_hit),
            "n_images": len(samples),
            "n_patients": len(reference.patient_ids),
            "patient_definition": "Directory_*",
            "series_definition": "folder-defined proxy; not validated DICOM UID",
            "device": self.config.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
            "total_runtime_seconds": total_runtime,
            "primary_result": results["strict_roi"].metrics,
            "permutation": permutation_summary,
            "duplicate_audit": duplicate_summary,
            "external_validation_status": "NOT_CONFIGURED",
        }
        write_json(self.run_dir / "run_metadata.json", metadata)
        self.logger.info("Completed in %s", format_duration(total_runtime))
        self.logger.info("All outputs: %s", self.run_dir)
        return {
            "output_dir": self.run_dir,
            "results": results,
            "stability_rows": stability_rows,
            "permutation_summary": permutation_summary,
            "duplicate_summary": duplicate_summary,
            "metadata": metadata,
        }


#%% ============================================================================
# NOTEBOOK-FRIENDLY ENTRY POINT
# =============================================================================


def run_showcase(profile: str | None = None) -> dict[str, Any]:
    """Run from a Kaggle cell or from ``%run`` without command-line arguments."""

    config = AdmissionsShowcaseConfig(profile=profile or os.environ.get("CAD_MRI_PROFILE", "showcase"))
    return CADMRIAdmissionsShowcase(config).run()


if __name__ == "__main__":
    SHOWCASE_RESULT = run_showcase()
