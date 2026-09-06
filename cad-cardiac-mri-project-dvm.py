#!/usr/bin/env python3
"""
CAD CARDIAC MRI — UNIV ADMISSIONS SHOWCASE PIPELINE
==================================================

Research question
-----------------
Can a patient-level AI system detect the Normal/Sick label in a cardiac-MRI
cohort primarily from cardiac image information, rather than from shortcuts
such as image borders, mask geometry, file-export style, or anatomy outside the
selected cardiac region?

Why this file exists
--------------------
The previous research suite was intentionally exhaustive: it contained 58
experiments and many diagnostics. That suite was useful for scientific auditing,
but it was too large to explain clearly in an admissions portfolio.

This file keeps only the scientifically essential path:

    JPEG slices
        -> label-blind geometric standardization
        -> frozen MONAI ventricular localization
        -> exact binary cardiac support
        -> frozen EfficientNet-B0 embeddings
        -> slice -> series proxy -> patient pooling
        -> nested patient-level cross-validation
        -> cardiac model + matched confounding controls

It also implements three small, original mathematical methods that are easy to
explain and inspect:

1. Fold-local nuisance projection
   Remove the linear component of the cardiac embedding that can be predicted
   from mask-only, outside-region, and export/QC features.

2. Confounder-penalized logistic regression
   Optimize logistic loss while directly penalizing covariance between model
   logits and nuisance variables.

3. Simulated-annealing score correction
   Search for a small correction of the cardiac logit using mask-only and
   outside-region scores, optimizing an inner-training objective that rewards
   AUC and penalizes nuisance correlation.

A fourth method, a Bayesian-bootstrap logistic ensemble, estimates uncertainty
by repeatedly reweighting the outer-training patients with Dirichlet samples.

Connection to CS50's Introduction to AI with Python
---------------------------------------------------
- Week 2, Uncertainty: probability, sampling, Bayesian reasoning.
- Week 3, Optimization: local search and simulated annealing.
- Week 4, Learning: supervised learning, regression, SVMs, overfitting,
  regularization, and k-means clustering.
- Week 5, Neural Networks: gradient descent, image convolution, and CNNs.

Important scientific liunivation
-------------------------------
This code cannot prove that confounding has been eliminated. The released
cohort contains only 30 patient folders, and the latest audit found that support
geometry and extracardiac regions remain predictive. This pipeline therefore
reports confounding diagnostics instead of claiming clinical validity.

The unit of analysis is always one Directory_* patient. Slices or series proxies
from the same patient never cross a train/validation boundary.

Typical Kaggle command
----------------------
python cad_mri_univ_showcase_pipeline.py \
    --dataset /kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset \
    --output /kaggle/working/cad_mri_univ_showcase

The first run creates a reusable feature bank. Later runs reuse it unless
--rebuild-cache is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class Config:
    """All settings that can change the mathematical experiment.

    Keeping these values in one immutable object makes the experiment easier to
    explain and makes the cache fingerprint reproducible.
    """

    dataset_root: Path
    output_root: Path
    rebuild_cache: bool = False
    series_annotation_csv: Path | None = None
    # Optional blinded review table. When supplied, only rows explicitly marked
    # contains_heart=True, is_localizer=False, is_derived_export=False are pooled.
    # Completed sequence_type and view_type values define equal-weight cells.

    # Image and feature extraction.
    monai_size: int = 256
    efficientnet_size: int = 224
    standardized_content_long_side: int = 240
    batch_size: int = 8
    dataloader_workers: int = 0
    embedding_dim: int = 1280
    cache_dtype: str = "float32"

    # MONAI localization and A17 support.
    monai_dilation_kernel: int = 31
    a17_extra_dilation_kernel: int = 15
    fallback_square_fraction: float = 0.65
    min_heart_area_ratio: float = 0.003
    max_heart_area_ratio: float = 0.50
    min_peak_heart_probability: float = 0.50

    # Region-only robust normalization.
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    min_region_pixels: int = 64
    min_dynamic_range: float = 8.0 / 255.0

    # Label-blind dark-border removal.
    dark_line_max_mean: float = 12.0
    dark_line_max_std: float = 4.0
    dark_pixel_max_value: int = 20
    dark_pixel_min_fraction: float = 0.98
    max_crop_fraction_per_side: float = 0.20
    min_retained_fraction: float = 0.60
    min_padding_run: int = 2

    # Patient-level validation.
    random_seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    repeated_cv_runs: int = 10
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    pca_variance: float = 0.95
    c_tolerance: float = 0.01
    bootstrap_replicates: int = 2000

    # Custom algorithm 1: nuisance projection.
    nuisance_components: int = 8
    projection_ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0)

    # Custom algorithm 2: confounder-penalized logistic regression.
    custom_l2_grid: tuple[float, ...] = (0.01, 0.1)
    custom_gamma_grid: tuple[float, ...] = (0.0, 0.1, 1.0)
    custom_selection_penalty: float = 0.05
    gradient_steps: int = 1200
    gradient_learning_rate: float = 0.03

    # Custom algorithm 3: simulated annealing.
    annealing_steps: int = 600
    annealing_start_temperature: float = 0.25
    annealing_cooling: float = 0.992
    annealing_step_scale: float = 0.20
    annealing_nuisance_penalty: float = 0.08
    annealing_coefficient_penalty: float = 0.01

    # Custom algorithm 4: Bayesian bootstrap.
    bayesian_bootstrap_draws: int = 100

    # Pinned external models.
    monai_repo: str = "MONAI/ventricular_short_axis_3label"
    monai_revision: str = "eefc17c8e002cc8a567bbfce8f02d7d3116408f4"
    monai_model_filename: str = "models/model.ts"
    monai_model_sha256: str = (
        "27d5532401fa6c1883872fa21635adbb7615981e7f385d0c58dd75b355e340b3"
    )
    efficientnet_weights_name: str = "IMAGENET1K_V1"

    # Version string changes whenever feature pixels or embeddings change.
    feature_cache_schema: str = "univ-showcase-a17-four-view-v1"


IMAGE_FEATURE_MODES = (
    "heart",       # A17 region-normalized cardiac intensity.
    "mask",        # C31 exact binary support only.
    "shuffled",    # C32 same support/histogram, destroyed spatial arrangement.
    "outside",     # C33 exact support complement.
)

SLICE_METADATA_NAMES = (
    "native_height",
    "native_width",
    "native_aspect_ratio",
    "file_bytes_per_native_pixel",
    "retained_height_fraction",
    "retained_width_fraction",
    "content_fraction_of_canvas",
    "support_fraction_of_content",
    "monai_gate_valid",
)

PATIENT_METADATA_NAMES = tuple(
    [f"mean_{name}" for name in SLICE_METADATA_NAMES]
    + [f"std_{name}" for name in SLICE_METADATA_NAMES]
    + ["log1p_slice_count", "log1p_series_count"]
)


@dataclass(frozen=True)
class SliceRecord:
    """One JPEG slice and its patient-safe identifiers."""

    path: Path
    label: int
    patient_id: str
    series_id: str


@dataclass(frozen=True)
class SeriesSelection:
    """Optional manually reviewed series subset and pooling-cell map."""

    selected_series: frozenset[str]
    series_to_cell: dict[str, str]
    source_csv: Path


@dataclass
class FeatureBank:
    """Memory-mapped slice embeddings plus aligned metadata."""

    records: list[SliceRecord]
    features: dict[str, np.ndarray]
    slice_metadata: np.ndarray
    decoded_hashes: list[str]
    cache_dir: Path


@dataclass
class PatientTable:
    """One row per Directory_* patient."""

    patient_ids: np.ndarray
    labels: np.ndarray
    heart: np.ndarray
    mask: np.ndarray
    shuffled: np.ndarray
    outside: np.ndarray
    metadata: np.ndarray

    @property
    def nuisance(self) -> np.ndarray:
        """Nuisance matrix used only inside training folds.

        It includes exact mask geometry, the outside-region embedding, and
        low-dimensional export/QC summaries. It deliberately excludes shuffled
        heart intensity because that representation can contain genuine cardiac
        intensity distribution as well as nuisance information.
        """

        return np.concatenate([self.mask, self.outside, self.metadata], axis=1)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    kind: Literal[
        "standard",
        "projected",
        "penalized",
        "annealed",
        "bayesian",
    ]
    source: Literal[
        "heart",
        "mask",
        "shuffled",
        "outside",
        "metadata",
    ]
    description: str
    role: str


EXPERIMENTS = (
    ExperimentSpec(
        name="HEART_BASELINE_A17",
        kind="standard",
        source="heart",
        role="primary_candidate",
        description=(
            "Region-normalized MRI intensity inside the exact A17 cardiac support."
        ),
    ),
    ExperimentSpec(
        name="CONTROL_MASK_ONLY_C31",
        kind="standard",
        source="mask",
        role="segmentation_control",
        description="Exact A17 binary support with MRI intensities removed.",
    ),
    ExperimentSpec(
        name="CONTROL_SHUFFLED_C32",
        kind="standard",
        source="shuffled",
        role="texture_control",
        description=(
            "Exact A17 support and intensity histogram, but spatial positions shuffled."
        ),
    ),
    ExperimentSpec(
        name="CONTROL_OUTSIDE_C33",
        kind="standard",
        source="outside",
        role="negative_control",
        description="Independently normalized exact complement of A17's support.",
    ),
    ExperimentSpec(
        name="CONTROL_METADATA_ONLY",
        kind="standard",
        source="metadata",
        role="negative_control",
        description=(
            "Only image geometry, export size, crop, support, gate and count summaries."
        ),
    ),
    ExperimentSpec(
        name="CUSTOM_NUISANCE_PROJECTED_HEART",
        kind="projected",
        source="heart",
        role="custom_deconfounding",
        description=(
            "Fold-local ridge projection removes the nuisance-predictable linear subspace."
        ),
    ),
    ExperimentSpec(
        name="CUSTOM_CONFOUNDER_PENALIZED_LOGISTIC",
        kind="penalized",
        source="heart",
        role="custom_deconfounding",
        description=(
            "Full-batch gradient descent minimizes logistic loss plus nuisance covariance."
        ),
    ),
    ExperimentSpec(
        name="CUSTOM_SIMULATED_ANNEALING_SCORE_DEBIASER",
        kind="annealed",
        source="heart",
        role="custom_deconfounding",
        description=(
            "Simulated annealing learns a small inner-training correction using mask and outside scores."
        ),
    ),
    ExperimentSpec(
        name="CUSTOM_BAYESIAN_BOOTSTRAP_ENSEMBLE",
        kind="bayesian",
        source="heart",
        role="uncertainty_model",
        description=(
            "Dirichlet-reweighted logistic ensemble averages plausible training-population fits."
        ),
    ),
)


# =============================================================================
# 2. REPRODUCIBILITY, LOGGING, AND SMALL UTILITIES
# =============================================================================


class Tee:
    """Write console messages to the notebook and to a text log."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {seconds:.1f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {seconds:.1f} s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def safe_abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def nuisance_correlation(scores: np.ndarray, nuisance: np.ndarray) -> float:
    """Mean absolute score correlation with non-constant nuisance columns."""

    nuisance = np.asarray(nuisance, dtype=np.float64)
    values = [
        safe_abs_correlation(scores, nuisance[:, column])
        for column in range(nuisance.shape[1])
        if np.std(nuisance[:, column]) > 1e-12
    ]
    return float(np.mean(values)) if values else 0.0


# =============================================================================
# 3. DATASET DISCOVERY — THE PATIENT IS ALWAYS Directory_*
# =============================================================================


def discover_records(dataset_root: Path) -> list[SliceRecord]:
    """Discover JPEGs deterministically without decoding pixels.

    The class label comes only from the top-level Normal/Sick folder. Every
    Directory_* folder is one patient. The immediate parent folder of each image
    becomes a patient-scoped series proxy; it is not claimed to be a DICOM UID.
    """

    records: list[SliceRecord] = []
    class_folders = (("Normal", 0), ("Sick", 1))

    for class_name, label in class_folders:
        class_root = dataset_root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_root}")

        patient_dirs = sorted(
            path for path in class_root.iterdir()
            if path.is_dir() and path.name.startswith("Directory_")
        )
        for patient_dir in patient_dirs:
            image_paths = sorted(
                path
                for path in patient_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            )
            if not image_paths:
                continue
            for image_path in image_paths:
                relative_parent = image_path.parent.relative_to(patient_dir)
                series_token = (
                    "_root" if str(relative_parent) == "." else relative_parent.as_posix()
                )
                records.append(
                    SliceRecord(
                        path=image_path,
                        label=label,
                        patient_id=patient_dir.name,
                        series_id=f"{patient_dir.name}/{series_token}",
                    )
                )

    if not records:
        raise RuntimeError(f"No JPEG images were found under {dataset_root}")

    patient_to_label: dict[str, int] = {}
    for record in records:
        previous = patient_to_label.setdefault(record.patient_id, record.label)
        if previous != record.label:
            raise RuntimeError(f"Patient {record.patient_id} has inconsistent labels.")

    counts = defaultdict(int)
    for label in patient_to_label.values():
        counts[label] += 1
    print(
        f"[DATA] patients={len(patient_to_label)} "
        f"(Normal={counts[0]}, Sick={counts[1]}), "
        f"slices={len(records)}, series_proxies={len({r.series_id for r in records})}",
        flush=True,
    )
    return records




def write_series_annotation_template(
    path: Path,
    records: Sequence[SliceRecord],
) -> None:
    """Create a blinded manual-review table without exposing Normal/Sick labels."""

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        grouped[(record.patient_id, record.series_id)].append(str(record.path))
    rows = []
    for patient_id, series_id in sorted(grouped):
        images = sorted(grouped[(patient_id, series_id)])
        rows.append(
            {
                "patient_id": patient_id,
                "series_proxy_id": series_id,
                "n_images": len(images),
                "first_image_path": images[0],
                "middle_image_path": images[len(images) // 2],
                "last_image_path": images[-1],
                "contains_heart": "",
                "is_localizer": "",
                "is_derived_export": "",
                "sequence_type": "",
                "view_type": "",
                "reviewer": "",
                "notes": "",
            }
        )
    write_csv(path, rows)


def _parse_annotation_bool(value: str, field: str, series_id: str) -> bool:
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    raise ValueError(
        f"Series {series_id!r}: {field} must be true/false, got {value!r}."
    )


def load_series_selection(
    csv_path: Path | None,
    records: Sequence[SliceRecord],
) -> SeriesSelection | None:
    """Load optional blinded sequence/view review without guessing folder meaning.

    Required columns:
        series_proxy_id (or series_id), contains_heart, is_localizer,
        is_derived_export, sequence_type, view_type

    Every retained series must have explicit sequence and view labels. These are
    used only to balance pooling cells; they are never inferred from SR_* names.
    """

    if csv_path is None:
        return None
    if not csv_path.is_file():
        raise FileNotFoundError(f"Series annotation CSV not found: {csv_path}")

    known_series = {record.series_id for record in records}
    selected: set[str] = set()
    cells: dict[str, str] = {}
    required = {
        "contains_heart",
        "is_localizer",
        "is_derived_export",
        "sequence_type",
        "view_type",
    }
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        series_column = (
            "series_proxy_id"
            if "series_proxy_id" in fields
            else "series_id"
            if "series_id" in fields
            else None
        )
        if missing or series_column is None:
            missing_display = sorted(
                missing | ({"series_proxy_id or series_id"} if series_column is None else set())
            )
            raise ValueError(
                f"Series annotation CSV is missing columns: {missing_display}"
            )
        for row in reader:
            series_id = str(row[series_column]).strip()
            if not series_id or series_id not in known_series:
                continue
            contains_heart = _parse_annotation_bool(
                row["contains_heart"], "contains_heart", series_id
            )
            is_localizer = _parse_annotation_bool(
                row["is_localizer"], "is_localizer", series_id
            )
            is_derived = _parse_annotation_bool(
                row["is_derived_export"], "is_derived_export", series_id
            )
            if not contains_heart or is_localizer or is_derived:
                continue
            sequence = str(row["sequence_type"]).strip().lower()
            view = str(row["view_type"]).strip().lower()
            if not sequence or not view:
                raise ValueError(
                    f"Selected series {series_id!r} needs sequence_type and view_type."
                )
            selected.add(series_id)
            cells[series_id] = f"{sequence}::{view}"

    if not selected:
        raise ValueError("Series annotations retained no eligible series.")
    all_patients = {record.patient_id for record in records}
    retained_patients = {
        record.patient_id for record in records if record.series_id in selected
    }
    missing_patients = sorted(all_patients - retained_patients)
    if missing_patients:
        raise ValueError(
            "Series selection removed every series for patients: "
            f"{missing_patients}"
        )
    print(
        f"[SERIES REVIEW] retained {len(selected)}/{len(known_series)} series "
        f"in {len(set(cells.values()))} explicit sequence/view cells.",
        flush=True,
    )
    return SeriesSelection(frozenset(selected), cells, csv_path)


# =============================================================================
# 4. LABEL-BLIND IMAGE STANDARDIZATION
# =============================================================================


def _is_dark_uniform_line(line: np.ndarray, cfg: Config) -> bool:
    line = np.asarray(line, dtype=np.float32)
    dark_fraction = float(np.mean(line <= cfg.dark_pixel_max_value))
    return (
        float(np.mean(line)) <= cfg.dark_line_max_mean
        and float(np.std(line)) <= cfg.dark_line_max_std
        and dark_fraction >= cfg.dark_pixel_min_fraction
    )


def detect_dark_padding_bounds(image: np.ndarray, cfg: Config) -> tuple[int, int, int, int]:
    """Remove only consecutive dark, nearly uniform edge lines.

    No class label, patient identifier, or model prediction is used. Conservative
    safety liunivs prevent the crop from deleting a large part of the field of view.
    """

    height, width = image.shape
    max_vertical = int(round(height * cfg.max_crop_fraction_per_side))
    max_horizontal = int(round(width * cfg.max_crop_fraction_per_side))

    top = 0
    while top < max_vertical and _is_dark_uniform_line(image[top, :], cfg):
        top += 1
    bottom = height
    while height - bottom < max_vertical and _is_dark_uniform_line(image[bottom - 1, :], cfg):
        bottom -= 1
    left = 0
    while left < max_horizontal and _is_dark_uniform_line(image[:, left], cfg):
        left += 1
    right = width
    while width - right < max_horizontal and _is_dark_uniform_line(image[:, right - 1], cfg):
        right -= 1

    if top < cfg.min_padding_run:
        top = 0
    if height - bottom < cfg.min_padding_run:
        bottom = height
    if left < cfg.min_padding_run:
        left = 0
    if width - right < cfg.min_padding_run:
        right = width

    retained_height = bottom - top
    retained_width = right - left
    if (
        retained_height < height * cfg.min_retained_fraction
        or retained_width < width * cfg.min_retained_fraction
    ):
        return 0, height, 0, width
    return top, bottom, left, right


def fixed_canvas_geometry(height: int, width: int, cfg: Config) -> tuple[int, int, int, int]:
    scale = cfg.standardized_content_long_side / float(max(height, width))
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    top = (cfg.monai_size - resized_height) // 2
    left = (cfg.monai_size - resized_width) // 2
    return resized_height, resized_width, top, left


def resize_into_canvas(
    image: np.ndarray,
    geometry: tuple[int, int, int, int],
    cfg: Config,
    interpolation: int,
) -> np.ndarray:
    resized_height, resized_width, top, left = geometry
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.zeros((cfg.monai_size, cfg.monai_size), dtype=np.float32)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_grayscale_image(
    path: Path,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Decode one JPEG and create aligned raw, MONAI, and content canvases."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unreadable JPEG: {path}")

    native_height, native_width = image.shape
    top, bottom, left, right = detect_dark_padding_bounds(image, cfg)
    cropped = image[top:bottom, left:right]
    geometry = fixed_canvas_geometry(*cropped.shape, cfg)

    raw_crop = cropped.astype(np.float32) / 255.0
    raw_canvas = resize_into_canvas(raw_crop, geometry, cfg, cv2.INTER_AREA)

    minimum = float(np.min(cropped))
    maximum = float(np.max(cropped))
    if maximum > minimum:
        monai_crop = (cropped.astype(np.float32) - minimum) / (maximum - minimum)
    else:
        monai_crop = np.zeros_like(cropped, dtype=np.float32)
    monai_canvas = resize_into_canvas(monai_crop, geometry, cfg, cv2.INTER_AREA)

    resized_height, resized_width, canvas_top, canvas_left = geometry
    content_canvas = np.zeros((cfg.monai_size, cfg.monai_size), dtype=np.float32)
    content_canvas[
        canvas_top : canvas_top + resized_height,
        canvas_left : canvas_left + resized_width,
    ] = 1.0

    file_size = path.stat().st_size
    metadata = np.asarray(
        [
            float(native_height),
            float(native_width),
            float(native_width / max(native_height, 1)),
            float(file_size / max(native_height * native_width, 1)),
            float(cropped.shape[0] / native_height),
            float(cropped.shape[1] / native_width),
            float(np.mean(content_canvas)),
        ],
        dtype=np.float32,
    )

    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    decoded_hash = digest.hexdigest()
    return raw_canvas, monai_canvas, content_canvas, metadata, decoded_hash


class CardiacSliceDataset(Dataset):
    """Decode images lazily so a DataLoader can batch model inference."""

    def __init__(self, records: Sequence[SliceRecord], cfg: Config):
        self.records = list(records)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        record = self.records[index]
        raw, monai, content, metadata, decoded_hash = preprocess_grayscale_image(
            record.path, self.cfg
        )
        return (
            torch.from_numpy(raw).unsqueeze(0),
            torch.from_numpy(monai).unsqueeze(0),
            torch.from_numpy(content).unsqueeze(0),
            torch.from_numpy(metadata),
            decoded_hash,
            index,
        )


# =============================================================================
# 5. FROZEN MONAI AND EFFICIENTNET MODELS
# =============================================================================


def load_monai_segmenter(cfg: Config, device: torch.device) -> torch.jit.ScriptModule:
    """Download one pinned official TorchScript artifact and verify its digest."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Install huggingface_hub before running the pipeline: "
            "pip install huggingface_hub"
        ) from error

    bundle_dir = cfg.output_root / "model_cache" / "monai"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(
        hf_hub_download(
            repo_id=cfg.monai_repo,
            filename=cfg.monai_model_filename,
            revision=cfg.monai_revision,
            local_dir=str(bundle_dir),
        )
    )
    observed = sha256_file(model_path)
    if observed != cfg.monai_model_sha256:
        raise RuntimeError(
            "MONAI model checksum mismatch. "
            f"Expected {cfg.monai_model_sha256}, observed {observed}."
        )

    model = torch.jit.load(str(model_path), map_location=device).eval()
    with torch.inference_mode():
        test = model(torch.zeros(1, 1, cfg.monai_size, cfg.monai_size, device=device))
    if test.ndim != 4 or test.shape[1] != 4 or not torch.isfinite(test).all():
        raise RuntimeError(f"Unexpected MONAI output contract: {tuple(test.shape)}")
    return model


class EfficientNetEncoder(nn.Module):
    """Frozen ImageNet EfficientNet-B0 with its classifier removed."""

    def __init__(self, cfg: Config):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        network = models.efficientnet_b0(weights=weights)
        network.classifier = nn.Identity()
        self.network = network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        self.cfg = cfg

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        return self.network(images)


def predict_exact_a17_support(
    monai_images: torch.Tensor,
    content_mask: torch.Tensor,
    segmenter: torch.jit.ScriptModule,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the exact binary support used by the simplified cardiac model.

    Valid MONAI outputs use the dilated hard ventricular mask. Invalid outputs
    use one fixed central square. Both paths are intersected with content_mask,
    making pipeline-added padding invisible.
    """

    logits = segmenter(monai_images)
    probabilities = torch.softmax(logits.float(), dim=1)
    heart_probability = probabilities[:, 1:].sum(dim=1, keepdim=True)
    class_map = torch.argmax(probabilities, dim=1, keepdim=True)
    hard = (class_map > 0).float()

    area_ratio = hard.mean(dim=(1, 2, 3))
    peak_probability = heart_probability.amax(dim=(1, 2, 3))
    valid = (
        (area_ratio >= cfg.min_heart_area_ratio)
        & (area_ratio <= cfg.max_heart_area_ratio)
        & (peak_probability >= cfg.min_peak_heart_probability)
    )

    hard = F.max_pool2d(
        hard,
        kernel_size=cfg.monai_dilation_kernel,
        stride=1,
        padding=cfg.monai_dilation_kernel // 2,
    )
    hard = F.max_pool2d(
        hard,
        kernel_size=cfg.a17_extra_dilation_kernel,
        stride=1,
        padding=cfg.a17_extra_dilation_kernel // 2,
    )

    side = int(round(cfg.monai_size * cfg.fallback_square_fraction))
    side = max(2, min(side, cfg.monai_size))
    top = (cfg.monai_size - side) // 2
    left = (cfg.monai_size - side) // 2
    fallback = torch.zeros_like(hard)
    fallback[:, :, top : top + side, left : left + side] = 1.0

    support = torch.where(valid.view(-1, 1, 1, 1), hard, fallback)
    support = support * (content_mask > 0.5).to(support.dtype)
    return support.clamp(0.0, 1.0), valid, area_ratio, peak_probability


# =============================================================================
# 6. FOUR MATCHED IMAGE REPRESENTATIONS
# =============================================================================


def robust_region_scale(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Scale only visible pixels; hidden pixels cannot affect the transform."""

    visible = mask > 0.5
    values = image[visible]
    if values.size < cfg.min_region_pixels:
        return np.zeros_like(image, dtype=np.float32)
    lower, upper = np.percentile(
        values,
        [cfg.lower_percentile, cfg.upper_percentile],
    )
    dynamic = max(float(upper - lower), cfg.min_dynamic_range)
    scaled = np.clip((image - float(lower)) / dynamic, 0.0, 1.0)
    return (scaled * visible.astype(np.float32)).astype(np.float32)


def affine_shuffle_visible_values(
    image: np.ndarray,
    support: np.ndarray,
    decoded_hash: str,
) -> np.ndarray:
    """Deterministically permute every visible value exactly once.

    The affine map k -> (a*k+b) mod n is bijective whenever gcd(a,n)=1.
    This preserves the complete intensity multiset while destroying the original
    spatial arrangement.
    """

    positions = np.flatnonzero(support.reshape(-1) > 0.5)
    output = np.zeros_like(image, dtype=np.float32)
    n_values = int(positions.size)
    if n_values <= 1:
        return output

    token = decoded_hash.lower()
    candidate = 2 + int(token[:16], 16) % max(1, n_values - 2)
    candidate %= n_values
    if candidate == 0:
        candidate = 1
    while math.gcd(candidate, n_values) != 1:
        candidate = 1 if candidate + 1 >= n_values else candidate + 1
    offset = int(token[16:32], 16) % n_values
    if candidate == 1 and offset == 0:
        offset = 1

    destination_rank = np.arange(n_values, dtype=np.int64)
    source_rank = (candidate * destination_rank + offset) % n_values
    source_values = image.reshape(-1)[positions]
    output.reshape(-1)[positions] = source_values[source_rank]
    return output


def build_matched_views(
    raw_image: np.ndarray,
    content_mask: np.ndarray,
    support: np.ndarray,
    decoded_hash: str,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Build the cardiac candidate and three exact controls from one support."""

    support = ((support > 0.5) & (content_mask > 0.5)).astype(np.float32)
    complement = ((content_mask > 0.5) & (support <= 0.5)).astype(np.float32)

    heart = robust_region_scale(raw_image, support, cfg)
    outside = robust_region_scale(raw_image, complement, cfg)
    shuffled = affine_shuffle_visible_values(heart, support, decoded_hash)

    return {
        "heart": heart,
        "mask": support,
        "shuffled": shuffled,
        "outside": outside,
    }


def views_to_tensor(
    views: Sequence[np.ndarray],
    cfg: Config,
) -> torch.Tensor:
    """Resize full canvases uniformly and repeat grayscale into RGB channels."""

    resized = [
        cv2.resize(
            view,
            (cfg.efficientnet_size, cfg.efficientnet_size),
            interpolation=cv2.INTER_AREA,
        )
        for view in views
    ]
    array = np.stack(resized, axis=0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(1).repeat(1, 3, 1, 1)


# =============================================================================
# 7. REUSABLE FEATURE BANK
# =============================================================================


def cache_fingerprint(records: Sequence[SliceRecord], cfg: Config) -> str:
    """Hash file identities and all feature-affecting configuration values."""

    digest = hashlib.sha256()
    settings = {
        "schema": cfg.feature_cache_schema,
        "monai_revision": cfg.monai_revision,
        "monai_sha256": cfg.monai_model_sha256,
        "efficientnet_weights": cfg.efficientnet_weights_name,
        "monai_size": cfg.monai_size,
        "efficientnet_size": cfg.efficientnet_size,
        "content_long_side": cfg.standardized_content_long_side,
        "monai_dilation": cfg.monai_dilation_kernel,
        "extra_dilation": cfg.a17_extra_dilation_kernel,
        "fallback_fraction": cfg.fallback_square_fraction,
        "gate": (
            cfg.min_heart_area_ratio,
            cfg.max_heart_area_ratio,
            cfg.min_peak_heart_probability,
        ),
        "region_scaling": (
            cfg.lower_percentile,
            cfg.upper_percentile,
            cfg.min_region_pixels,
            cfg.min_dynamic_range,
        ),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))

    root = cfg.dataset_root.resolve()
    for record in records:
        stat = record.path.stat()
        relative = record.path.resolve().relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()[:20]


def _bank_paths(cache_dir: Path) -> dict[str, Path]:
    paths = {
        mode: cache_dir / f"features_{mode}.npy" for mode in IMAGE_FEATURE_MODES
    }
    paths.update(
        {
            "slice_metadata": cache_dir / "slice_metadata.npy",
            "decoded_hashes": cache_dir / "decoded_hashes.txt",
            "records": cache_dir / "records.csv",
            "metadata": cache_dir / "metadata.json",
        }
    )
    return paths


def load_feature_bank(
    records: list[SliceRecord],
    cache_dir: Path,
) -> FeatureBank | None:
    paths = _bank_paths(cache_dir)
    if not all(path.is_file() for path in paths.values()):
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not metadata.get("complete") or metadata.get("n_slices") != len(records):
        return None

    features = {
        mode: np.load(paths[mode], mmap_mode="r") for mode in IMAGE_FEATURE_MODES
    }
    slice_metadata = np.load(paths["slice_metadata"], mmap_mode="r")
    decoded_hashes = paths["decoded_hashes"].read_text(encoding="utf-8").splitlines()
    return FeatureBank(records, features, slice_metadata, decoded_hashes, cache_dir)


def audit_exact_cross_patient_duplicates(
    records: Sequence[SliceRecord],
    hashes: Sequence[str],
) -> dict[str, Any]:
    hash_to_patients: dict[str, set[str]] = defaultdict(set)
    hash_to_labels: dict[str, set[int]] = defaultdict(set)
    for record, digest in zip(records, hashes):
        hash_to_patients[digest].add(record.patient_id)
        hash_to_labels[digest].add(record.label)
    cross_patient = [key for key, values in hash_to_patients.items() if len(values) > 1]
    cross_label = [key for key, values in hash_to_labels.items() if len(values) > 1]
    return {
        "cross_patient_hash_groups": len(cross_patient),
        "cross_label_hash_groups": len(cross_label),
        "safe_for_patient_only_splitting": len(cross_patient) == 0,
    }


def extract_feature_bank(
    records: list[SliceRecord],
    cfg: Config,
    cache_dir: Path,
) -> FeatureBank:
    """Run MONAI and EfficientNet once, storing four aligned representations."""

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _bank_paths(cache_dir)
    paths["metadata"].write_text(
        json.dumps({"complete": False, "n_slices": len(records)}, indent=2),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FEATURES] device={device}, slices={len(records)}", flush=True)
    segmenter = load_monai_segmenter(cfg, device)
    encoder = EfficientNetEncoder(cfg).to(device).eval()

    dtype = np.float16 if cfg.cache_dtype == "float16" else np.float32
    feature_arrays = {
        mode: np.lib.format.open_memmap(
            paths[mode],
            mode="w+",
            dtype=dtype,
            shape=(len(records), cfg.embedding_dim),
        )
        for mode in IMAGE_FEATURE_MODES
    }
    slice_metadata = np.lib.format.open_memmap(
        paths["slice_metadata"],
        mode="w+",
        dtype=np.float32,
        shape=(len(records), len(SLICE_METADATA_NAMES)),
    )
    decoded_hashes = [""] * len(records)

    dataset = CardiacSliceDataset(records, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_workers,
        pin_memory=device.type == "cuda",
    )

    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Feature bank", unit="batch"):
            raw_cpu, monai_cpu, content_cpu, base_meta, hashes, indices = batch
            monai = monai_cpu.to(device, non_blocking=True)
            content = content_cpu.to(device, non_blocking=True)
            support, valid, _, _ = predict_exact_a17_support(
                monai, content, segmenter, cfg
            )

            raw_np = raw_cpu[:, 0].numpy()
            content_np = content_cpu[:, 0].numpy()
            support_np = support[:, 0].cpu().numpy()
            valid_np = valid.cpu().numpy().astype(np.float32)
            index_np = np.asarray(indices, dtype=np.int64)

            mode_views: dict[str, list[np.ndarray]] = {
                mode: [] for mode in IMAGE_FEATURE_MODES
            }
            for local_index, global_index in enumerate(index_np):
                view_dict = build_matched_views(
                    raw_np[local_index],
                    content_np[local_index],
                    support_np[local_index],
                    str(hashes[local_index]),
                    cfg,
                )
                for mode in IMAGE_FEATURE_MODES:
                    mode_views[mode].append(view_dict[mode])

                support_count = float(np.sum(support_np[local_index] > 0.5))
                content_count = float(np.sum(content_np[local_index] > 0.5))
                slice_metadata[global_index] = np.concatenate(
                    [
                        base_meta[local_index].numpy().astype(np.float32),
                        np.asarray(
                            [
                                support_count / max(content_count, 1.0),
                                valid_np[local_index],
                            ],
                            dtype=np.float32,
                        ),
                    ]
                )
                decoded_hashes[global_index] = str(hashes[local_index])

            # The simplified suite has only four image modes, so all four are
            # concatenated into one EfficientNet call. This is clearer and much
            # cheaper than the 32-view research suite while remaining safe for
            # EfficientNet-B0 at batch_size=8 on common Kaggle GPUs.
            tensors = [
                views_to_tensor(mode_views[mode], cfg)
                for mode in IMAGE_FEATURE_MODES
            ]
            concatenated = torch.cat(tensors, dim=0).to(
                device, non_blocking=True
            )
            embeddings = encoder(concatenated).float().cpu().numpy()
            current_batch_size = len(index_np)
            for offset, mode in enumerate(IMAGE_FEATURE_MODES):
                feature_arrays[mode][index_np] = embeddings[
                    offset * current_batch_size : (offset + 1) * current_batch_size
                ].astype(dtype)

    for array in feature_arrays.values():
        array.flush()
    slice_metadata.flush()

    paths["decoded_hashes"].write_text(
        "\n".join(decoded_hashes) + "\n", encoding="utf-8"
    )
    with paths["records"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("index", "path", "label", "patient_id", "series_id"),
        )
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "index": index,
                    "path": str(record.path),
                    "label": record.label,
                    "patient_id": record.patient_id,
                    "series_id": record.series_id,
                }
            )

    duplicate_audit = audit_exact_cross_patient_duplicates(records, decoded_hashes)
    metadata = {
        "complete": True,
        "n_slices": len(records),
        "feature_modes": list(IMAGE_FEATURE_MODES),
        "embedding_dim": cfg.embedding_dim,
        "slice_metadata_names": list(SLICE_METADATA_NAMES),
        "duplicate_audit": duplicate_audit,
        "runtime_seconds": time.perf_counter() - started,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[FEATURES] completed in {elapsed(metadata['runtime_seconds'])}; "
        f"exact cross-patient groups={duplicate_audit['cross_patient_hash_groups']}",
        flush=True,
    )
    if not duplicate_audit["safe_for_patient_only_splitting"]:
        raise RuntimeError(
            "Exact decoded pixels occur across patients. Group those patients "
            "before interpreting cross-validation."
        )
    loaded = load_feature_bank(records, cache_dir)
    if loaded is None:
        raise RuntimeError("Feature bank could not be reloaded after extraction.")
    return loaded


def load_or_build_feature_bank(records: list[SliceRecord], cfg: Config) -> FeatureBank:
    fingerprint = cache_fingerprint(records, cfg)
    cache_dir = cfg.output_root / "feature_cache" / fingerprint
    if not cfg.rebuild_cache:
        existing = load_feature_bank(records, cache_dir)
        if existing is not None:
            print(f"[FEATURES] cache hit: {cache_dir}", flush=True)
            return existing
    print(f"[FEATURES] cache miss: {cache_dir}", flush=True)
    return extract_feature_bank(records, cfg, cache_dir)


# =============================================================================
# 8. HIERARCHICAL SLICE -> SERIES -> PATIENT POOLING
# =============================================================================


def hierarchical_patient_pool(
    features: np.ndarray,
    records: Sequence[SliceRecord],
    row_mask: np.ndarray | None = None,
    series_to_cell: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool slices -> series -> optional sequence/view cells -> patient.

    Without annotations, every series proxy receives equal patient-level weight.
    With annotations, series are first averaged inside each explicit
    (sequence_type, view_type) cell and cells then receive equal weight.
    """

    if row_mask is None:
        row_mask = np.ones(len(records), dtype=bool)
    row_mask = np.asarray(row_mask, dtype=bool)

    patient_to_series: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    patient_to_label: dict[str, int] = {}
    for index, record in enumerate(records):
        if not row_mask[index]:
            continue
        patient_to_series[record.patient_id][record.series_id].append(index)
        patient_to_label[record.patient_id] = record.label

    all_patients = sorted({record.patient_id for record in records})
    missing = [patient for patient in all_patients if patient not in patient_to_series]
    if missing:
        raise RuntimeError(f"Filtering removed every slice for patients: {missing}")

    pooled, labels = [], []
    for patient_id in all_patients:
        series_vectors: dict[str, np.ndarray] = {}
        for series_id in sorted(patient_to_series[patient_id]):
            indices = patient_to_series[patient_id][series_id]
            series_vectors[series_id] = np.asarray(
                features[indices], dtype=np.float32
            ).mean(axis=0)

        if series_to_cell is None:
            patient_vector = np.stack(list(series_vectors.values()), axis=0).mean(axis=0)
        else:
            cell_to_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
            for series_id, vector in series_vectors.items():
                if series_id not in series_to_cell:
                    raise RuntimeError(
                        f"Selected series {series_id!r} has no sequence/view cell."
                    )
                cell_to_vectors[series_to_cell[series_id]].append(vector)
            cell_vectors = [
                np.stack(cell_to_vectors[cell], axis=0).mean(axis=0)
                for cell in sorted(cell_to_vectors)
            ]
            patient_vector = np.stack(cell_vectors, axis=0).mean(axis=0)

        pooled.append(patient_vector)
        labels.append(patient_to_label[patient_id])
    return (
        np.stack(pooled).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(all_patients),
    )


def aggregate_patient_metadata(
    slice_metadata: np.ndarray,
    records: Sequence[SliceRecord],
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if row_mask is None:
        row_mask = np.ones(len(records), dtype=bool)
    row_mask = np.asarray(row_mask, dtype=bool)
    patient_to_indices: dict[str, list[int]] = defaultdict(list)
    patient_to_series: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        if not row_mask[index]:
            continue
        patient_to_indices[record.patient_id].append(index)
        patient_to_series[record.patient_id].add(record.series_id)

    patient_ids = np.asarray(sorted(patient_to_indices))
    rows = []
    for patient_id in patient_ids:
        values = np.asarray(slice_metadata[patient_to_indices[patient_id]], dtype=np.float32)
        rows.append(
            np.concatenate(
                [
                    values.mean(axis=0),
                    values.std(axis=0),
                    [
                        np.log1p(len(patient_to_indices[patient_id])),
                        np.log1p(len(patient_to_series[patient_id])),
                    ],
                ]
            )
        )
    return np.stack(rows).astype(np.float32), patient_ids


def build_patient_table(
    bank: FeatureBank,
    selection: SeriesSelection | None = None,
) -> PatientTable:
    pooled: dict[str, np.ndarray] = {}
    selected_rows = np.asarray(
        [
            selection is None or record.series_id in selection.selected_series
            for record in bank.records
        ],
        dtype=bool,
    )
    series_to_cell = None if selection is None else selection.series_to_cell
    labels: np.ndarray | None = None
    patient_ids: np.ndarray | None = None

    for mode in IMAGE_FEATURE_MODES:
        matrix, mode_labels, mode_patients = hierarchical_patient_pool(
            bank.features[mode],
            bank.records,
            row_mask=selected_rows,
            series_to_cell=series_to_cell,
        )
        pooled[mode] = matrix
        if labels is None:
            labels, patient_ids = mode_labels, mode_patients
        elif not np.array_equal(labels, mode_labels) or not np.array_equal(
            patient_ids, mode_patients
        ):
            raise RuntimeError("Patient pooling is not aligned across feature modes.")

    metadata, metadata_patients = aggregate_patient_metadata(
        bank.slice_metadata, bank.records, row_mask=selected_rows
    )
    if not np.array_equal(patient_ids, metadata_patients):
        raise RuntimeError("Patient rows are not aligned with metadata pooling.")

    return PatientTable(
        patient_ids=patient_ids,
        labels=labels,
        heart=pooled["heart"],
        mask=pooled["mask"],
        shuffled=pooled["shuffled"],
        outside=pooled["outside"],
        metadata=metadata,
    )


# =============================================================================
# 9. STANDARD LINEAR MODEL AND CUSTOM MATHEMATICAL ALGORITHMS
# =============================================================================


def build_standard_pipeline(c_value: float, cfg: Config, seed: int) -> Pipeline:
    """Fold-local scaler -> PCA -> regularized logistic regression."""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=cfg.pca_variance, svd_solver="full")),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


class NuisanceProjectedModel:
    """Fold-local linear residualization followed by logistic regression.

    Mathematics
    -----------
    Let H be standardized cardiac embeddings and Z low-dimensional nuisance
    components. Fit a ridge map B on training patients only:

        B = (Z^T Z + alpha I)^(-1) Z^T H

    Then remove the predictable component:

        H_clean = H - Z B

    Validation patients are transformed with the same training-fitted scalers,
    PCA basis, and B. No validation label participates in projection.
    """

    def __init__(self, alpha: float, c_value: float, cfg: Config, seed: int):
        self.alpha = float(alpha)
        self.c_value = float(c_value)
        self.cfg = cfg
        self.seed = seed

    def fit(self, heart: np.ndarray, nuisance: np.ndarray, y: np.ndarray) -> "NuisanceProjectedModel":
        self.heart_scaler = StandardScaler().fit(heart)
        heart_scaled = self.heart_scaler.transform(heart)

        self.nuisance_scaler = StandardScaler().fit(nuisance)
        nuisance_scaled = self.nuisance_scaler.transform(nuisance)
        n_components = max(
            1,
            min(
                self.cfg.nuisance_components,
                nuisance_scaled.shape[1],
                nuisance_scaled.shape[0] - 2,
            ),
        )
        self.nuisance_pca = PCA(n_components=n_components, svd_solver="full").fit(
            nuisance_scaled
        )
        z = self.nuisance_pca.transform(nuisance_scaled)
        identity = np.eye(z.shape[1], dtype=np.float64)
        self.coefficients = np.linalg.solve(
            z.T @ z + self.alpha * identity,
            z.T @ heart_scaled,
        )
        cleaned = heart_scaled - z @ self.coefficients
        self.classifier = build_standard_pipeline(
            self.c_value, self.cfg, self.seed
        ).fit(cleaned, y)
        return self

    def transform(self, heart: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
        heart_scaled = self.heart_scaler.transform(heart)
        z = self.nuisance_pca.transform(self.nuisance_scaler.transform(nuisance))
        return heart_scaled - z @ self.coefficients

    def predict_proba(self, heart: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
        cleaned = self.transform(heart, nuisance)
        return self.classifier.predict_proba(cleaned)[:, 1]


class ConfounderPenalizedLogisticRegression:
    """Custom full-batch Adam optimizer with a nuisance-covariance penalty.

    Objective
    ---------

        logistic_loss(y, Xw+b)
        + (lambda / 2) ||w||^2
        + (gamma / 2) || Cov(Z, Xw+b) ||^2

    X contains cardiac features. Z contains nuisance components. The final term
    discourages logits that vary linearly with acquisition/export proxies.
    This can reduce *measured linear confounding*; it cannot guarantee causal
    independence or clinical validity.
    """

    def __init__(
        self,
        l2: float,
        gamma: float,
        steps: int,
        learning_rate: float,
        seed: int,
    ):
        self.l2 = float(l2)
        self.gamma = float(gamma)
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

    def fit(self, x: np.ndarray, z: np.ndarray, y: np.ndarray) -> "ConfounderPenalizedLogisticRegression":
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z_centered = z - z.mean(axis=0, keepdims=True)

        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.01, size=x.shape[1])
        prevalence = np.clip(y.mean(), 1e-3, 1.0 - 1e-3)
        b = float(np.log(prevalence / (1.0 - prevalence)))

        positive_weight = len(y) / (2.0 * max(np.sum(y == 1), 1))
        negative_weight = len(y) / (2.0 * max(np.sum(y == 0), 1))
        sample_weight = np.where(y == 1, positive_weight, negative_weight)
        sample_weight /= sample_weight.sum()

        mw = np.zeros_like(w)
        vw = np.zeros_like(w)
        mb = 0.0
        vb = 0.0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        best_loss = float("inf")
        best = (w.copy(), b)
        patience = 0

        for step in range(1, self.steps + 1):
            logits = x @ w + b
            probabilities = safe_sigmoid(logits)
            residual = sample_weight * (probabilities - y)

            centered_logits = logits - logits.mean()
            covariance = z_centered.T @ centered_logits / len(y)
            nuisance_logit_gradient = (
                self.gamma * (z_centered @ covariance) / len(y)
            )

            grad_w = x.T @ (residual + nuisance_logit_gradient) + self.l2 * w
            grad_b = float(np.sum(residual))

            mw = beta1 * mw + (1.0 - beta1) * grad_w
            vw = beta2 * vw + (1.0 - beta2) * np.square(grad_w)
            mb = beta1 * mb + (1.0 - beta1) * grad_b
            vb = beta2 * vb + (1.0 - beta2) * grad_b * grad_b
            mw_hat = mw / (1.0 - beta1**step)
            vw_hat = vw / (1.0 - beta2**step)
            mb_hat = mb / (1.0 - beta1**step)
            vb_hat = vb / (1.0 - beta2**step)
            w -= self.learning_rate * mw_hat / (np.sqrt(vw_hat) + epsilon)
            b -= self.learning_rate * mb_hat / (math.sqrt(vb_hat) + epsilon)

            if step % 20 == 0 or step == self.steps:
                # Recompute the objective *after* the Adam update. Using the
                # pre-update probability here would make early stopping lag one
                # optimization step and would be mathematically inconsistent.
                updated_logits = x @ w + b
                updated_probabilities = safe_sigmoid(updated_logits)
                updated_covariance = (
                    z_centered.T @ (updated_logits - updated_logits.mean()) / len(y)
                )
                clipped = np.clip(
                    updated_probabilities, 1e-9, 1.0 - 1e-9
                )
                logistic = -np.sum(
                    sample_weight
                    * (y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
                )
                loss = (
                    logistic
                    + 0.5 * self.l2 * float(w @ w)
                    + 0.5 * self.gamma
                    * float(updated_covariance @ updated_covariance)
                )
                if loss < best_loss - 1e-8:
                    best_loss = loss
                    best = (w.copy(), b)
                    patience = 0
                else:
                    patience += 1
                if patience >= 20:
                    break

        self.w, self.b = best
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return safe_sigmoid(np.asarray(x, dtype=np.float64) @ self.w + self.b)


class PenalizedHeartModel:
    """Fold-local preprocessing wrapper for the custom optimizer."""

    def __init__(self, l2: float, gamma: float, cfg: Config, seed: int):
        self.l2 = l2
        self.gamma = gamma
        self.cfg = cfg
        self.seed = seed

    def fit(self, heart: np.ndarray, nuisance: np.ndarray, y: np.ndarray) -> "PenalizedHeartModel":
        self.heart_scaler = StandardScaler().fit(heart)
        heart_scaled = self.heart_scaler.transform(heart)
        heart_components = max(1, min(10, heart_scaled.shape[0] - 2, heart_scaled.shape[1]))
        self.heart_pca = PCA(n_components=heart_components, svd_solver="full").fit(
            heart_scaled
        )
        x = self.heart_pca.transform(heart_scaled)

        self.nuisance_scaler = StandardScaler().fit(nuisance)
        nuisance_scaled = self.nuisance_scaler.transform(nuisance)
        nuisance_components = max(
            1,
            min(
                self.cfg.nuisance_components,
                nuisance_scaled.shape[0] - 2,
                nuisance_scaled.shape[1],
            ),
        )
        self.nuisance_pca = PCA(
            n_components=nuisance_components, svd_solver="full"
        ).fit(nuisance_scaled)
        z = self.nuisance_pca.transform(nuisance_scaled)

        self.model = ConfounderPenalizedLogisticRegression(
            l2=self.l2,
            gamma=self.gamma,
            steps=self.cfg.gradient_steps,
            learning_rate=self.cfg.gradient_learning_rate,
            seed=self.seed,
        ).fit(x, z, y)
        return self

    def transformed(self, heart: np.ndarray, nuisance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.heart_pca.transform(self.heart_scaler.transform(heart))
        z = self.nuisance_pca.transform(self.nuisance_scaler.transform(nuisance))
        return x, z

    def predict_proba(self, heart: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
        x, _ = self.transformed(heart, nuisance)
        return self.model.predict_proba(x)


class SimulatedAnnealingDebiaser:
    """CS50-inspired local search over two score-correction coefficients."""

    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.seed = seed

    def _objective(
        self,
        y: np.ndarray,
        heart_z: np.ndarray,
        mask_z: np.ndarray,
        outside_z: np.ndarray,
        alpha: float,
        beta: float,
    ) -> float:
        corrected = heart_z - alpha * mask_z - beta * outside_z
        auc = roc_auc_score(y, corrected)
        nuisance = 0.5 * (
            safe_abs_correlation(corrected, mask_z)
            + safe_abs_correlation(corrected, outside_z)
        )
        coefficient_cost = alpha * alpha + beta * beta
        return float(
            auc
            - self.cfg.annealing_nuisance_penalty * nuisance
            - self.cfg.annealing_coefficient_penalty * coefficient_cost
        )

    def fit(
        self,
        y: np.ndarray,
        heart_scores: np.ndarray,
        mask_scores: np.ndarray,
        outside_scores: np.ndarray,
    ) -> "SimulatedAnnealingDebiaser":
        logits = [safe_logit(values) for values in (heart_scores, mask_scores, outside_scores)]
        self.means = np.asarray([values.mean() for values in logits])
        self.stds = np.asarray([max(values.std(), 1e-6) for values in logits])
        heart_z, mask_z, outside_z = [
            (values - mean) / std
            for values, mean, std in zip(logits, self.means, self.stds)
        ]

        rng = np.random.default_rng(self.seed)
        alpha = beta = 0.0
        current = self._objective(y, heart_z, mask_z, outside_z, alpha, beta)
        best = (current, alpha, beta)
        temperature = self.cfg.annealing_start_temperature

        for _ in range(self.cfg.annealing_steps):
            candidate_alpha = float(
                np.clip(
                    alpha + rng.normal(0.0, self.cfg.annealing_step_scale * temperature),
                    -2.0,
                    2.0,
                )
            )
            candidate_beta = float(
                np.clip(
                    beta + rng.normal(0.0, self.cfg.annealing_step_scale * temperature),
                    -2.0,
                    2.0,
                )
            )
            candidate = self._objective(
                y,
                heart_z,
                mask_z,
                outside_z,
                candidate_alpha,
                candidate_beta,
            )
            delta = candidate - current
            if delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-8)):
                alpha, beta, current = candidate_alpha, candidate_beta, candidate
            if current > best[0]:
                best = (current, alpha, beta)
            temperature *= self.cfg.annealing_cooling

        self.objective_, self.alpha_, self.beta_ = best
        return self

    def transform(
        self,
        heart_scores: np.ndarray,
        mask_scores: np.ndarray,
        outside_scores: np.ndarray,
    ) -> np.ndarray:
        logits = [safe_logit(values) for values in (heart_scores, mask_scores, outside_scores)]
        z = [
            (values - mean) / std
            for values, mean, std in zip(logits, self.means, self.stds)
        ]
        corrected = z[0] - self.alpha_ * z[1] - self.beta_ * z[2]
        return safe_sigmoid(corrected)


# =============================================================================
# 10. NESTED PATIENT-LEVEL EVALUATION
# =============================================================================


def source_matrix(table: PatientTable, source: str) -> np.ndarray:
    return np.asarray(getattr(table, source), dtype=np.float32)


def select_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, scores)
    objective = true_positive_rate - false_positive_rate
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    best = np.where(finite)[0][np.argmax(objective[finite])]
    return float(thresholds[best])


def standard_param_grid(cfg: Config) -> list[dict[str, float]]:
    return [{"c": c} for c in cfg.c_grid]


def projected_param_grid(cfg: Config) -> list[dict[str, float]]:
    return [
        {"c": c, "alpha": alpha}
        for alpha in cfg.projection_ridge_grid
        for c in cfg.c_grid
    ]


def penalized_param_grid(cfg: Config) -> list[dict[str, float]]:
    return [
        {"l2": l2, "gamma": gamma}
        for l2 in cfg.custom_l2_grid
        for gamma in cfg.custom_gamma_grid
    ]


def fit_predict_basic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    c_value: float,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, Any]:
    model = build_standard_pipeline(c_value, cfg, seed).fit(x_train, y_train)
    return model.predict_proba(x_valid)[:, 1], model


def fit_predict_experiment(
    spec: ExperimentSpec,
    table: PatientTable,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    parameters: dict[str, float],
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, Any]:
    y_train = table.labels[train_indices]
    x = source_matrix(table, spec.source)

    if spec.kind == "standard":
        return fit_predict_basic(
            x[train_indices],
            y_train,
            x[valid_indices],
            parameters["c"],
            cfg,
            seed,
        )

    if spec.kind == "projected":
        model = NuisanceProjectedModel(
            alpha=parameters["alpha"],
            c_value=parameters["c"],
            cfg=cfg,
            seed=seed,
        ).fit(table.heart[train_indices], table.nuisance[train_indices], y_train)
        return (
            model.predict_proba(
                table.heart[valid_indices], table.nuisance[valid_indices]
            ),
            model,
        )

    if spec.kind == "penalized":
        model = PenalizedHeartModel(
            l2=parameters["l2"],
            gamma=parameters["gamma"],
            cfg=cfg,
            seed=seed,
        ).fit(table.heart[train_indices], table.nuisance[train_indices], y_train)
        return (
            model.predict_proba(
                table.heart[valid_indices], table.nuisance[valid_indices]
            ),
            model,
        )

    raise ValueError(f"fit_predict_experiment does not handle {spec.kind!r}")


def parameter_grid(spec: ExperimentSpec, cfg: Config) -> list[dict[str, float]]:
    if spec.kind == "standard" or spec.kind == "bayesian":
        return standard_param_grid(cfg)
    if spec.kind == "projected":
        return projected_param_grid(cfg)
    if spec.kind == "penalized":
        return penalized_param_grid(cfg)
    raise ValueError(f"No direct grid for {spec.kind!r}")


def inner_oof_scores(
    spec: ExperimentSpec,
    table: PatientTable,
    outer_train_indices: np.ndarray,
    parameters: dict[str, float],
    cfg: Config,
    seed: int,
) -> np.ndarray:
    y = table.labels[outer_train_indices]
    splitter = StratifiedKFold(
        n_splits=cfg.inner_folds,
        shuffle=True,
        random_state=seed,
    )
    scores = np.zeros(len(outer_train_indices), dtype=np.float64)
    for inner_fold, (local_train, local_valid) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        train_indices = outer_train_indices[local_train]
        valid_indices = outer_train_indices[local_valid]
        fold_scores, _ = fit_predict_experiment(
            spec,
            table,
            train_indices,
            valid_indices,
            parameters,
            cfg,
            seed + inner_fold,
        )
        scores[local_valid] = fold_scores
    return scores


def compact_nuisance_coordinates(
    nuisance: np.ndarray,
    max_components: int,
) -> np.ndarray:
    """Create a small nuisance coordinate system for training-only diagnostics.

    The transformation is fitted on the current outer-training cohort. It uses
    no validation patients and no labels. Reducing thousands of nuisance
    dimensions to a few principal components avoids averaging correlations over
    many noisy embedding coordinates.
    """

    scaled = StandardScaler().fit_transform(nuisance)
    components = max(
        1,
        min(max_components, scaled.shape[0] - 2, scaled.shape[1]),
    )
    return PCA(n_components=components, svd_solver="full").fit_transform(scaled)


def select_parameters(
    spec: ExperimentSpec,
    table: PatientTable,
    outer_train_indices: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, float]:
    candidates = []
    y = table.labels[outer_train_indices]
    nuisance_coordinates = compact_nuisance_coordinates(
        table.nuisance[outer_train_indices], cfg.nuisance_components
    )
    for parameters in parameter_grid(spec, cfg):
        scores = inner_oof_scores(
            spec, table, outer_train_indices, parameters, cfg, seed
        )
        auc = float(roc_auc_score(y, scores))
        correlation = nuisance_correlation(
            safe_logit(scores), nuisance_coordinates
        )
        if spec.kind == "penalized":
            objective = auc - cfg.custom_selection_penalty * correlation
        else:
            objective = auc
        candidates.append((objective, auc, correlation, parameters, scores))

    best_objective = max(row[0] for row in candidates)
    eligible = [
        row for row in candidates if row[0] >= best_objective - cfg.c_tolerance
    ]
    # Deterministic tie rules:
    #   - ordinary models prefer smaller C;
    #   - custom deconfounding models first prefer lower measured nuisance
    #     correlation, then simpler/stronger-regularized settings.
    if spec.kind == "penalized":
        eligible.sort(
            key=lambda row: (
                row[2],
                -row[3].get("l2", 0.0),
                -row[3].get("gamma", 0.0),
            )
        )
    elif spec.kind == "projected":
        eligible.sort(
            key=lambda row: (
                row[2],
                row[3].get("c", 1.0),
                row[3].get("alpha", 0.0),
            )
        )
    else:
        eligible.sort(key=lambda row: row[3].get("c", 1.0))
    selected = eligible[0]
    return selected[3], selected[4], selected[2]


def train_basic_with_selected_c(
    source: str,
    table: PatientTable,
    outer_train_indices: np.ndarray,
    valid_indices: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    temporary = ExperimentSpec(
        name=f"TEMP_{source}",
        kind="standard",
        source=source,  # type: ignore[arg-type]
        description="Temporary nested score model.",
        role="internal",
    )
    parameters, inner_scores, _ = select_parameters(
        temporary, table, outer_train_indices, cfg, seed
    )
    valid_scores, _ = fit_predict_experiment(
        temporary,
        table,
        outer_train_indices,
        valid_indices,
        parameters,
        cfg,
        seed + 99,
    )
    return valid_scores, inner_scores, parameters["c"]


def evaluate_annealed_fold(
    table: PatientTable,
    outer_train_indices: np.ndarray,
    valid_indices: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit three inner-validated score models, then anneal only on training OOF."""

    # Use the same inner splitter seed for all three score streams. Their OOF
    # values are therefore matched patient by patient under the same partitions.
    valid_heart, inner_heart, heart_c = train_basic_with_selected_c(
        "heart", table, outer_train_indices, valid_indices, cfg, seed
    )
    valid_mask, inner_mask, mask_c = train_basic_with_selected_c(
        "mask", table, outer_train_indices, valid_indices, cfg, seed
    )
    valid_outside, inner_outside, outside_c = train_basic_with_selected_c(
        "outside", table, outer_train_indices, valid_indices, cfg, seed
    )

    debiaser = SimulatedAnnealingDebiaser(cfg, seed + 40).fit(
        table.labels[outer_train_indices],
        inner_heart,
        inner_mask,
        inner_outside,
    )
    valid_scores = debiaser.transform(valid_heart, valid_mask, valid_outside)
    inner_scores = debiaser.transform(inner_heart, inner_mask, inner_outside)
    parameters = {
        "heart_c": heart_c,
        "mask_c": mask_c,
        "outside_c": outside_c,
        "alpha": debiaser.alpha_,
        "beta": debiaser.beta_,
        "annealing_objective": debiaser.objective_,
    }
    return valid_scores, inner_scores, parameters


def evaluate_bayesian_fold(
    table: PatientTable,
    outer_train_indices: np.ndarray,
    valid_indices: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Bayesian bootstrap around a training-only selected cardiac classifier."""

    base_spec = next(spec for spec in EXPERIMENTS if spec.name == "HEART_BASELINE_A17")
    parameters, inner_scores, _ = select_parameters(
        base_spec, table, outer_train_indices, cfg, seed
    )
    c_value = parameters["c"]

    # Fit the feature transform once on outer training. Dirichlet draws then
    # represent uncertainty in the finite training-patient distribution.
    scaler = StandardScaler().fit(table.heart[outer_train_indices])
    x_train_scaled = scaler.transform(table.heart[outer_train_indices])
    x_valid_scaled = scaler.transform(table.heart[valid_indices])
    pca = PCA(n_components=cfg.pca_variance, svd_solver="full").fit(x_train_scaled)
    x_train = pca.transform(x_train_scaled)
    x_valid = pca.transform(x_valid_scaled)
    y_train = table.labels[outer_train_indices]

    rng = np.random.default_rng(seed + 1000)
    draw_predictions = []
    for draw in range(cfg.bayesian_bootstrap_draws):
        weights = rng.dirichlet(np.ones(len(y_train))) * len(y_train)
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed + draw,
        ).fit(x_train, y_train, sample_weight=weights)
        draw_predictions.append(model.predict_proba(x_valid)[:, 1])
    matrix = np.stack(draw_predictions, axis=0)
    return (
        matrix.mean(axis=0),
        inner_scores,
        matrix.std(axis=0),
        {"c": c_value, "draws": float(cfg.bayesian_bootstrap_draws)},
    )


def fold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_accuracy": 0.5
        * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
    }


def run_one_repeated_cv(
    spec: ExperimentSpec,
    table: PatientTable,
    cfg: Config,
    repeat_index: int,
) -> dict[str, Any]:
    seed = cfg.random_seed + 10_000 * repeat_index
    splitter = StratifiedKFold(
        n_splits=cfg.outer_folds,
        shuffle=True,
        random_state=seed,
    )
    scores = np.zeros(len(table.labels), dtype=np.float64)
    thresholds = np.zeros(len(table.labels), dtype=np.float64)
    uncertainties = np.zeros(len(table.labels), dtype=np.float64)
    fold_ids = np.zeros(len(table.labels), dtype=np.int64)
    selected_rows = []

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(np.zeros(len(table.labels)), table.labels), start=1
    ):
        fold_seed = seed + fold * 100

        if spec.kind in {"standard", "projected", "penalized"}:
            parameters, inner_scores, inner_corr = select_parameters(
                spec, table, train_indices, cfg, fold_seed
            )
            valid_scores, _ = fit_predict_experiment(
                spec,
                table,
                train_indices,
                valid_indices,
                parameters,
                cfg,
                fold_seed + 50,
            )
            uncertainty = np.zeros(len(valid_indices), dtype=np.float64)
        elif spec.kind == "annealed":
            valid_scores, inner_scores, parameters = evaluate_annealed_fold(
                table, train_indices, valid_indices, cfg, fold_seed
            )
            inner_corr = nuisance_correlation(
                safe_logit(inner_scores), table.metadata[train_indices]
            )
            uncertainty = np.zeros(len(valid_indices), dtype=np.float64)
        elif spec.kind == "bayesian":
            valid_scores, inner_scores, uncertainty, parameters = evaluate_bayesian_fold(
                table, train_indices, valid_indices, cfg, fold_seed
            )
            inner_corr = nuisance_correlation(
                safe_logit(inner_scores), table.metadata[train_indices]
            )
        else:
            raise ValueError(spec.kind)

        threshold = select_threshold(table.labels[train_indices], inner_scores)
        scores[valid_indices] = valid_scores
        thresholds[valid_indices] = threshold
        uncertainties[valid_indices] = uncertainty
        fold_ids[valid_indices] = fold
        selected_rows.append(
            {
                "fold": fold,
                "parameters": parameters,
                "training_only_threshold": threshold,
                "inner_nuisance_correlation": inner_corr,
            }
        )

    predictions = (scores >= thresholds).astype(np.int64)
    metrics = fold_metrics(table.labels, scores, 0.5)
    # Threshold-dependent metrics must use each patient's fold-specific threshold.
    tn, fp, fn, tp = confusion_matrix(
        table.labels, predictions, labels=[0, 1]
    ).ravel()
    metrics.update(
        {
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "balanced_accuracy": 0.5
            * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
            "metadata_score_correlation": nuisance_correlation(
                safe_logit(scores), table.metadata
            ),
        }
    )
    return {
        "repeat": repeat_index,
        "scores": scores,
        "thresholds": thresholds,
        "predictions": predictions,
        "uncertainties": uncertainties,
        "fold_ids": fold_ids,
        "metrics": metrics,
        "fold_selections": selected_rows,
    }


def bootstrap_auc_interval(
    y: np.ndarray,
    scores: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(cfg.bootstrap_replicates):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        values.append(roc_auc_score(y[indices], scores[indices]))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_experiments(
    table: PatientTable,
    cfg: Config,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summaries = []
    prediction_rows = []
    run_history: dict[str, dict[str, np.ndarray]] = {}

    for spec in EXPERIMENTS:
        print(f"\n[EXPERIMENT] {spec.name}: {spec.description}", flush=True)
        runs = [
            run_one_repeated_cv(spec, table, cfg, repeat)
            for repeat in range(cfg.repeated_cv_runs)
        ]
        aucs = np.asarray([run["metrics"]["auc"] for run in runs])
        correlations = np.asarray(
            [run["metrics"]["metadata_score_correlation"] for run in runs]
        )
        run_history[spec.name] = {
            "auc": aucs.copy(),
            "metadata_correlation": correlations.copy(),
        }
        primary = runs[0]
        ci_low, ci_high = bootstrap_auc_interval(
            table.labels,
            primary["scores"],
            cfg,
            cfg.random_seed + 70_000 + len(summaries),
        )
        summary = {
            "experiment": spec.name,
            "role": spec.role,
            "description": spec.description,
            "primary_auc": float(primary["metrics"]["auc"]),
            "primary_auc_ci_low": ci_low,
            "primary_auc_ci_high": ci_high,
            "primary_auprc": float(primary["metrics"]["auprc"]),
            "primary_sensitivity": float(primary["metrics"]["sensitivity"]),
            "primary_specificity": float(primary["metrics"]["specificity"]),
            "repeated_auc_median": float(np.median(aucs)),
            "repeated_auc_q25": float(np.quantile(aucs, 0.25)),
            "repeated_auc_q75": float(np.quantile(aucs, 0.75)),
            "repeated_auc_min": float(np.min(aucs)),
            "repeated_auc_max": float(np.max(aucs)),
            "repeated_metadata_correlation_median": float(np.median(correlations)),
        }
        summaries.append(summary)
        print(
            f"[RESULT] AUC={summary['primary_auc']:.4f}; repeated median="
            f"{summary['repeated_auc_median']:.4f} "
            f"[{summary['repeated_auc_q25']:.4f}, {summary['repeated_auc_q75']:.4f}]; "
            f"metadata-correlation={summary['repeated_metadata_correlation_median']:.3f}",
            flush=True,
        )

        for index, patient_id in enumerate(table.patient_ids):
            prediction_rows.append(
                {
                    "experiment": spec.name,
                    "patient_id": patient_id,
                    "true_label": int(table.labels[index]),
                    "fold": int(primary["fold_ids"][index]),
                    "score": float(primary["scores"][index]),
                    "training_only_threshold": float(primary["thresholds"][index]),
                    "predicted_label": int(primary["predictions"][index]),
                    "bayesian_score_std": float(primary["uncertainties"][index]),
                }
            )

    # Same-repeat comparisons are more informative than subtracting two
    # independently summarized medians. A positive delta always favors the
    # second experiment named below.
    comparison_specs = (
        ("HEART_OVER_MASK_GEOMETRY", "CONTROL_MASK_ONLY_C31", "HEART_BASELINE_A17"),
        ("HEART_OVER_SHUFFLED_HISTOGRAM", "CONTROL_SHUFFLED_C32", "HEART_BASELINE_A17"),
        ("HEART_OVER_EXACT_COMPLEMENT", "CONTROL_OUTSIDE_C33", "HEART_BASELINE_A17"),
        ("HEART_OVER_METADATA_ONLY", "CONTROL_METADATA_ONLY", "HEART_BASELINE_A17"),
        ("PROJECTED_MINUS_BASELINE", "HEART_BASELINE_A17", "CUSTOM_NUISANCE_PROJECTED_HEART"),
        ("PENALIZED_MINUS_BASELINE", "HEART_BASELINE_A17", "CUSTOM_CONFOUNDER_PENALIZED_LOGISTIC"),
        ("ANNEALED_MINUS_BASELINE", "HEART_BASELINE_A17", "CUSTOM_SIMULATED_ANNEALING_SCORE_DEBIASER"),
        ("BAYESIAN_MINUS_BASELINE", "HEART_BASELINE_A17", "CUSTOM_BAYESIAN_BOOTSTRAP_ENSEMBLE"),
    )
    paired_rows = []
    for name, reference, comparison in comparison_specs:
        auc_delta = run_history[comparison]["auc"] - run_history[reference]["auc"]
        corr_delta = (
            run_history[comparison]["metadata_correlation"]
            - run_history[reference]["metadata_correlation"]
        )
        paired_rows.append(
            {
                "comparison": name,
                "reference_experiment": reference,
                "comparison_experiment": comparison,
                "auc_delta_median": float(np.median(auc_delta)),
                "auc_delta_q25": float(np.quantile(auc_delta, 0.25)),
                "auc_delta_q75": float(np.quantile(auc_delta, 0.75)),
                "fraction_auc_delta_positive": float(np.mean(auc_delta > 0.0)),
                "metadata_correlation_delta_median": float(np.median(corr_delta)),
                "interpretation": (
                    "Positive AUC delta favors comparison_experiment; negative "
                    "metadata-correlation delta means lower measured metadata dependence."
                ),
            }
        )

    summaries.sort(key=lambda row: row["repeated_auc_median"], reverse=True)
    return summaries, prediction_rows, paired_rows


# =============================================================================
# 11. UNSUPERVISED STYLE-CLUSTER AUDIT (CS50 WEEK 4: K-MEANS)
# =============================================================================


def run_style_cluster_audit(table: PatientTable, cfg: Config) -> list[dict[str, Any]]:
    """Cluster only export/QC summaries, then inspect label concentration.

    Labels are not used to fit k-means. High class purity after clustering is a
    warning that acquisition/export style may be entangled with the outcome.
    The cluster IDs are diagnostics, not validated scanner or sequence labels.
    """

    scaled = StandardScaler().fit_transform(table.metadata)
    cluster_count = min(3, len(table.patient_ids))
    clusters = KMeans(
        n_clusters=cluster_count,
        n_init=50,
        random_state=cfg.random_seed,
    ).fit_predict(scaled)

    rows = []
    for cluster in range(cluster_count):
        indices = np.where(clusters == cluster)[0]
        normal = int(np.sum(table.labels[indices] == 0))
        sick = int(np.sum(table.labels[indices] == 1))
        purity = max(normal, sick) / max(len(indices), 1)
        rows.append(
            {
                "cluster": cluster,
                "patients": len(indices),
                "normal": normal,
                "sick": sick,
                "class_purity": purity,
                "warning": purity >= 0.80,
                "patient_ids": ";".join(table.patient_ids[indices]),
            }
        )
    return rows


# =============================================================================
# 12. REPORTING
# =============================================================================


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_patient_table(path: Path, table: PatientTable) -> None:
    rows = []
    for index, patient_id in enumerate(table.patient_ids):
        row = {"patient_id": patient_id, "label": int(table.labels[index])}
        row.update(
            {
                name: float(table.metadata[index, column])
                for column, name in enumerate(PATIENT_METADATA_NAMES)
            }
        )
        rows.append(row)
    write_csv(path, rows)


def build_research_report(
    summaries: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    style_rows: list[dict[str, Any]],
    cfg: Config,
    bank: FeatureBank,
) -> dict[str, Any]:
    lookup = {row["experiment"]: row for row in summaries}
    heart = lookup["HEART_BASELINE_A17"]
    controls = [
        lookup["CONTROL_MASK_ONLY_C31"],
        lookup["CONTROL_SHUFFLED_C32"],
        lookup["CONTROL_OUTSIDE_C33"],
        lookup["CONTROL_METADATA_ONLY"],
    ]
    custom = [
        lookup["CUSTOM_NUISANCE_PROJECTED_HEART"],
        lookup["CUSTOM_CONFOUNDER_PENALIZED_LOGISTIC"],
        lookup["CUSTOM_SIMULATED_ANNEALING_SCORE_DEBIASER"],
        lookup["CUSTOM_BAYESIAN_BOOTSTRAP_ENSEMBLE"],
    ]
    return {
        "project": "CAD Cardiac MRI — UNIV Admissions Showcase",
        "scientific_question": (
            "Can cardiac-region information predict the cohort label more strongly "
            "than exact support geometry, intensity histogram, outside anatomy, and "
            "export/QC metadata?"
        ),
        "dataset": {
            "root": str(cfg.dataset_root),
            "patients": len({record.patient_id for record in bank.records}),
            "slices": len(bank.records),
            "series_proxies": len({record.series_id for record in bank.records}),
        },
        "primary_candidate": heart,
        "matched_controls": controls,
        "custom_algorithms": custom,
        "paired_repeated_cv_comparisons": paired_rows,
        "style_cluster_audit": style_rows,
        "success_criteria": {
            "high_cardiac_auc": "Repeated median AUROC remains high.",
            "mask_gap": "Cardiac AUROC clearly exceeds exact mask-only AUROC.",
            "texture_gap": "Cardiac AUROC clearly exceeds shuffled-intensity AUROC.",
            "outside_gap": "Cardiac AUROC clearly exceeds complement AUROC.",
            "reduced_nuisance": (
                "A custom model retains AUROC while lowering score correlation "
                "with export/QC metadata."
            ),
        },
        "liunivations": [
            "Only 30 released patient folders are available.",
            "Normal/Sick may be entangled with sequence, protocol, scanner, or export style.",
            "MONAI is short-axis-specific while the released series are heterogeneous.",
            "Internal cross-validation is not external clinical validation.",
            "Reducing measured linear dependence does not prove causal deconfounding.",
        ],
        "configuration": asdict(cfg),
    }


def print_summary(summaries: Sequence[dict[str, Any]]) -> None:
    print("\n" + "=" * 110)
    print("FINAL REPEATED-CV SUMMARY")
    print("=" * 110)
    print(
        f"{'Experiment':49} {'AUC':>7} {'Repeated median [IQR]':>28} {'Meta corr':>10}"
    )
    for row in summaries:
        print(
            f"{row['experiment'][:49]:49} "
            f"{row['primary_auc']:7.4f} "
            f"{row['repeated_auc_median']:7.4f} "
            f"[{row['repeated_auc_q25']:.4f}, {row['repeated_auc_q75']:.4f}] "
            f"{row['repeated_metadata_correlation_median']:10.3f}"
        )


# =============================================================================
# 13. MAIN — SIX READABLE RESEARCH STAGES
# =============================================================================


def validate_config(cfg: Config) -> None:
    if cfg.outer_folds < 2 or cfg.inner_folds < 2:
        raise ValueError("Outer and inner fold counts must be at least 2.")
    if cfg.repeated_cv_runs < 1:
        raise ValueError("repeated_cv_runs must be positive.")
    if cfg.monai_dilation_kernel % 2 == 0 or cfg.a17_extra_dilation_kernel % 2 == 0:
        raise ValueError("Dilation kernels must be odd.")
    if not 0.0 < cfg.fallback_square_fraction <= 1.0:
        raise ValueError("fallback_square_fraction must lie in (0,1].")
    if cfg.cache_dtype not in {"float16", "float32"}:
        raise ValueError("cache_dtype must be float16 or float32.")


def run(cfg: Config) -> None:
    validate_config(cfg)
    seed_everything(cfg.random_seed)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.output_root / "results"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_handle = (run_dir / "console_output.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)

    started = time.perf_counter()
    try:
        print("\nCAD CARDIAC MRI — UNIV SHOWCASE PIPELINE")
        print(f"Dataset: {cfg.dataset_root}")
        print(f"Output:  {run_dir}")
        print(f"Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("Primary scientific unit: one Directory_* patient")

        print("\n[STAGE 1/6] Discover patient-safe image records")
        records = discover_records(cfg.dataset_root)
        write_series_annotation_template(
            run_dir / "series_annotation_template.csv", records
        )

        print("\n[STAGE 2/6] Load or build the four-view frozen feature bank")
        bank = load_or_build_feature_bank(records, cfg)

        print("\n[STAGE 3/6] Pool slices -> series proxies -> patients")
        selection = load_series_selection(cfg.series_annotation_csv, records)
        table = build_patient_table(bank, selection=selection)
        write_patient_table(run_dir / "patient_metadata.csv", table)
        print(f"[PATIENT TABLE] shape: {table.heart.shape}; patients={len(table.labels)}")

        print("\n[STAGE 4/6] Run unsupervised acquisition-style audit")
        style_rows = run_style_cluster_audit(table, cfg)
        write_csv(run_dir / "style_cluster_audit.csv", style_rows)
        for row in style_rows:
            print(
                f"[STYLE CLUSTER {row['cluster']}] patients={row['patients']}, "
                f"Normal={row['normal']}, Sick={row['sick']}, "
                f"purity={row['class_purity']:.2f}"
            )

        print("\n[STAGE 5/6] Run nested, repeated patient-level experiments")
        summaries, prediction_rows, paired_rows = run_experiments(table, cfg)
        write_csv(run_dir / "experiment_summary.csv", summaries)
        write_csv(run_dir / "patient_oof_predictions.csv", prediction_rows)
        write_csv(run_dir / "paired_repeated_cv_comparisons.csv", paired_rows)

        print("\n[STAGE 6/6] Write the machine-readable research report")
        report = build_research_report(
            summaries, paired_rows, style_rows, cfg, bank
        )
        (run_dir / "research_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        (run_dir / "configuration.json").write_text(
            json.dumps(asdict(cfg), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print_summary(summaries)
        print(f"\nCompleted in {elapsed(time.perf_counter() - started)}")
        print(
            "Interpretation rule: a high cardiac AUC is not enough. The cardiac "
            "model should also exceed mask, shuffled, outside, and metadata controls."
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def parse_args() -> argparse.Namespace:
    default_dataset = Path(
        "/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"
    )
    default_output = Path("/kaggle/working/cad_mri_univ_showcase")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--series-annotations",
        type=Path,
        default=None,
        help=(
            "Optional completed blinded CSV with contains_heart, localizer, "
            "derived-export, sequence_type and view_type columns."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Repeated outer-CV runs; use 50 for the final scientific report.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Reduce if GPU memory is insufficient.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        dataset_root=args.dataset,
        output_root=args.output,
        rebuild_cache=args.rebuild_cache,
        series_annotation_csv=args.series_annotations,
        repeated_cv_runs=args.repeats,
        batch_size=args.batch_size,
    )
    run(cfg)


if __name__ == "__main__":
    main()
