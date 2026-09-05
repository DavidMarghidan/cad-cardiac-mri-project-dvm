from __future__ import annotations

"""
CAD CARDIAC MRI — CLEAR PATIENT-LEVEL RESEARCH PIPELINE
=======================================================

Purpose
-------
This file is a compact research version of the larger multi-experiment suite.
It is designed to be readable in an admissions portfolio and to make the
scientific question explicit:

    Does heart-localized MRI signal predict the released Normal/Sick labels
    beyond information contained in mask geometry, image periphery, export
    metadata, and other dataset-specific shortcuts?

The pipeline deliberately contains only:

    1. one transparent A17-style cardiac reference;
    2. four project-specific mathematical methods;
    3. five negative/diagnostic controls;
    4. patient-level nested cross-validation;
    5. repeated split-stability and paired bootstrap comparisons.

The custom mathematical components are:

    RSC  — Robust Series Consensus pooling;
    PRC  — Paired Regional Contrast (heart minus outside representation);
    FLCO — Fold-Local Confounder Orthogonalization;
    APPS — AUC-Preserving Pareto Selection of hyperparameters.

A fifth optional component, Protocol-Cell Balanced Pooling, becomes active only
when a human-reviewed series annotation CSV is supplied. The code never guesses
MRI sequence/view from SR_* or series* folder names.

Important limitations
---------------------
* The dataset contains only 30 validated computational patient units under the
  mapping patient_id = Directory_*.
* The top-level Normal/Sick folder supplies the label.
* The JPEG release lacks original DICOM acquisition metadata.
* MONAI is a short-axis ventricular segmenter and is used here as a localization
  tool, not as proof of anatomically correct segmentation on every image.
* Results are exploratory and are not suitable for clinical use.
* Removing confounding can reduce internal AUROC. Such a decrease can be a more
  scientifically honest result than preserving a shortcut-driven score.

Expected runtime
----------------
The expensive stage is frozen feature extraction over all JPEG slices. Only four
image views are encoded, rather than the 32 views used by the audit suite, so the
feature bank is substantially simpler and faster. Patient-level experiments are
small and reuse the same cached features.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import csv
import hashlib
import json
import logging
import math
import os

# Set before the first CUDA operation. PyTorch reads this environment variable
# when deterministic cuBLAS kernels are initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older sklearn fallback
    StratifiedGroupKFold = None


# =============================================================================
# 2. CONFIGURATION
# =============================================================================


def _default_dataset_path() -> Path:
    kaggle_path = Path(
        "/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"
    )
    if kaggle_path.exists():
        return kaggle_path
    return Path(
        os.environ.get(
            "CAD_MRI_DATASET",
            r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset",
        )
    )


def _default_output_path() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cad_mri_mit_clear_pipeline")
    return Path(os.environ.get("CAD_MIT_OUTPUT", "cad_mri_mit_clear_pipeline"))


@dataclass(frozen=True)
class Config:
    """All scientific choices are visible in one compact configuration."""

    dataset_path: Path = _default_dataset_path()
    output_dir: Path = _default_output_path()
    seed: int = 42

    # Image geometry.
    classifier_size: int = 224
    monai_size: int = 256
    fixed_content_long_side: int = 240
    batch_size: int = 8
    num_workers: int = 0
    encoder_modes_per_call: int = 4

    # Label-blind native-border removal.
    dark_line_max_mean: float = 12.0
    dark_line_max_std: float = 4.0
    dark_pixel_max_value: int = 20
    dark_pixel_min_fraction: float = 0.98
    max_crop_fraction_per_side: float = 0.20
    min_retained_fraction: float = 0.60
    min_padding_run: int = 2

    # MONAI localization gate.
    monai_dilation_kernel: int = 31
    a17_extra_dilation_kernel: int = 15
    min_heart_area_ratio: float = 0.003
    max_heart_area_ratio: float = 0.50
    min_peak_heart_probability: float = 0.50
    invalid_mask_center_fraction: float = 0.65

    # Region-only intensity normalization.
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    histogram_bins: int = 256
    min_visible_pixels: int = 64
    min_dynamic_range: float = 8.0 / 255.0

    # Patient-level evaluation.
    outer_folds: int = 5
    inner_folds: int = 3
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    pca_variance: float = 0.95
    auc_tolerance: float = 0.01
    stability_repeats: int = 20
    bootstrap_replicates: int = 2000

    # Custom Robust Series Consensus.
    rsc_max_iterations: int = 60
    rsc_tolerance: float = 1e-5
    rsc_distance_epsilon: float = 1e-6
    rsc_cauchy_scale: float = 2.5

    # Fold-Local Confounder Orthogonalization.
    nuisance_variance: float = 0.95
    max_nuisance_components: int = 5

    # Optional manually reviewed protocol balancing.
    series_annotation_csv: str | None = os.environ.get(
        "CAD_SERIES_ANNOTATION_CSV"
    )
    allowed_protocol_cells: tuple[str, ...] = tuple(
        value.strip().lower()
        for value in os.environ.get("CAD_ALLOWED_PROTOCOL_CELLS", "").split(",")
        if value.strip()
    )
    # Example environment value: "cine::short_axis,lge::short_axis". A manually
    # predeclared common cell set is stronger than selecting cells after seeing
    # final model performance. Empty means all reviewed usable cells are retained.
    require_contains_heart: bool = True
    exclude_localizers: bool = True
    exclude_derived_exports: bool = True

    # Frozen model identifiers.
    monai_repo_id: str = "MONAI/ventricular_short_axis_3label"
    monai_revision: str = "eefc17c8e002cc8a567bbfce8f02d7d3116408f4"
    monai_torchscript_sha256: str = (
        "27d5532401fa6c1883872fa21635adbb7615981e7f385d0c58dd75b355e340b3"
    )
    efficientnet_weights: str = "IMAGENET1K_V1"

    # Cache version changes whenever feature-producing logic changes.
    feature_cache_version: str = "mit-clear-v1-four-views-rsc-flco-2026-09-05"

    # Label-blind duplicate handling before patient aggregation.
    deduplicate_exact_within_patient: bool = True

    # Reproducibility. AMP is disabled in the default research run.
    use_amp: bool = False


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Four frozen image views only. This is the key simplification relative to the
# 58-experiment audit suite.
IMAGE_MODES = ("heart", "mask", "shuffled", "outside")
EMBEDDING_DIM = 1280

# The custom combined model is predeclared before evaluation. The A17-style
# reference remains separately reported.
PRIMARY_EXPERIMENT = "M4_RSC_FLCO"
REFERENCE_EXPERIMENT = "A17_REFERENCE"


@dataclass(frozen=True)
class Experiment:
    """One concise patient-level experiment."""

    experiment_id: str
    description: str
    feature_source: str
    pooling: str = "mean"
    deconfound: bool = False
    use_pca: bool = True
    role: str = "candidate"


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "A17_REFERENCE",
        "A17-style heart support, ordinary equal series mean.",
        "heart",
        pooling="mean",
        role="reference",
    ),
    Experiment(
        "A20_VALID_ONLY",
        "A17 pixels after excluding MONAI-gate-invalid slices.",
        "heart_valid",
        pooling="mean",
        role="ablation",
    ),
    Experiment(
        "M1_ROBUST_SERIES_CONSENSUS",
        "Custom robust patient pooling that downweights outlying series.",
        "heart",
        pooling="rsc",
        role="custom_candidate",
    ),
    Experiment(
        "M2_PAIRED_REGIONAL_CONTRAST",
        "L2-normalized cardiac embedding minus matched outside embedding.",
        "contrast",
        pooling="mean",
        role="custom_candidate",
    ),
    Experiment(
        "M3_FLCO",
        "Fold-local ridge orthogonalization against mask/outside/metadata nuisance.",
        "heart",
        pooling="mean",
        deconfound=True,
        role="custom_candidate",
    ),
    Experiment(
        "M4_RSC_FLCO",
        "Robust Series Consensus followed by Fold-Local Confounder Orthogonalization.",
        "heart",
        pooling="rsc",
        deconfound=True,
        role="prospective_primary",
    ),
    Experiment(
        "C1_EXACT_SUPPORT_ONLY",
        "Exact A17 support geometry without MRI intensity.",
        "mask",
        pooling="mean",
        role="control",
    ),
    Experiment(
        "C2_SHUFFLED_HEART_INTENSITY",
        "Exact support and intensity multiset, but spatial anatomy destroyed.",
        "shuffled",
        pooling="mean",
        role="control",
    ),
    Experiment(
        "C3_EXACT_SUPPORT_COMPLEMENT",
        "Region-normalized pixels outside the exact A17 support.",
        "outside",
        pooling="mean",
        role="control",
    ),
    Experiment(
        "C4_EXPORT_METADATA_ONLY",
        "Counts, geometry, padding, file size, gate rate and support size only.",
        "metadata",
        pooling="not_applicable",
        use_pca=False,
        role="control",
    ),
    Experiment(
        "C5_COMBINED_NUISANCE_ONLY",
        "Mask, outside-region and interpretable metadata combined without heart intensity.",
        "nuisance",
        pooling="mean",
        use_pca=True,
        role="control",
    ),
)

# Metadata features are intentionally simple and interpretable. They are not fed
# to the cardiac classifier; they are used as a negative control and as part of
# the nuisance matrix for FLCO.
SLICE_METADATA_NAMES = (
    "native_height",
    "native_width",
    "native_aspect_ratio",
    "bytes_per_native_pixel",
    "detected_native_padding_fraction",
    "fixed_canvas_padding_fraction",
    "monai_mask_valid",
    "a17_support_fraction",
)

PATIENT_METADATA_NAMES = (
    "log1p_slice_count",
    "log1p_series_count",
    "mean_native_aspect_ratio",
    "std_native_aspect_ratio",
    "mean_bytes_per_native_pixel",
    "std_bytes_per_native_pixel",
    "mean_detected_native_padding_fraction",
    "mean_fixed_canvas_padding_fraction",
    "monai_valid_rate",
    "mean_a17_support_fraction",
    "std_a17_support_fraction",
    "reviewed_protocol_cell_count",
    "reviewed_protocol_cell_entropy",
    "reviewed_protocol_max_fraction",
)


# =============================================================================
# 3. REPRODUCIBILITY AND LOGGING
# =============================================================================


def set_reproducibility(seed: int) -> None:
    """Request deterministic execution where PyTorch supports it."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older PyTorch
        torch.use_deterministic_algorithms(True)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cad_mri_mit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


# =============================================================================
# 4. DATA DISCOVERY AND OPTIONAL HUMAN-REVIEWED SERIES ANNOTATIONS
# =============================================================================


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    patient_id: str
    series_id: str


def discover_records(root: Path, logger: logging.Logger) -> list[ImageRecord]:
    """Discover JPEG rows while defining patients only by Directory_* folders."""

    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    records: list[ImageRecord] = []
    label_map = {"Normal": 0, "Sick": 1}
    patient_to_label: dict[str, int] = {}

    for class_name, label in label_map.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for patient_dir in sorted(class_dir.glob("Directory_*")):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            previous = patient_to_label.get(patient_id)
            if previous is not None and previous != label:
                raise RuntimeError(f"Patient {patient_id} appears in both classes.")
            patient_to_label[patient_id] = label

            image_paths = sorted(
                path
                for path in patient_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for image_path in image_paths:
                relative_parent = image_path.parent.relative_to(patient_dir)
                series_suffix = relative_parent.as_posix()
                if series_suffix == ".":
                    series_suffix = "root"
                series_id = f"{patient_id}/{series_suffix}"
                records.append(
                    ImageRecord(
                        str(image_path), label, patient_id, series_id
                    )
                )

    if not records:
        raise RuntimeError("No MRI images were discovered.")

    labels_by_patient = {
        patient_id: label for patient_id, label in patient_to_label.items()
    }
    normal = sum(value == 0 for value in labels_by_patient.values())
    sick = sum(value == 1 for value in labels_by_patient.values())
    series_count = len({record.series_id for record in records})
    logger.info(
        "Dataset: %d patients (%d Normal, %d Sick), %d images, %d series proxies.",
        len(labels_by_patient),
        normal,
        sick,
        len(records),
        series_count,
    )
    return records


def write_series_annotation_template(
    records: Sequence[ImageRecord], output_path: Path
) -> None:
    """Create a label-blind template for manual sequence/view review."""

    series_to_count: dict[str, int] = defaultdict(int)
    series_to_patient: dict[str, str] = {}
    for record in records:
        series_to_count[record.series_id] += 1
        series_to_patient[record.series_id] = record.patient_id

    rows = []
    for series_id in sorted(series_to_count):
        rows.append(
            {
                "series_id": series_id,
                "patient_id": series_to_patient[series_id],
                "image_count": series_to_count[series_id],
                "contains_heart": "",
                "is_localizer": "",
                "is_derived": "",
                "sequence_type": "",
                "view_type": "",
                "review_confidence": "",
                "review_notes": "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_bool(value: str) -> bool | None:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return None


def load_series_annotations(
    csv_path: str | None,
    logger: logging.Logger,
) -> dict[str, str]:
    """Return series_id -> protocol cell for reviewed usable series.

    The protocol cell is ``sequence_type::view_type``. Empty or incomplete rows
    are excluded rather than guessed. The mapping is label-blind.
    """

    if not csv_path:
        logger.warning(
            "No reviewed series annotation CSV supplied. Protocol-cell balancing "
            "is disabled; sequence/view confounding remains an open limitation."
        )
        return {}

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Series annotation CSV not found: {path}")

    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "series_id",
            "contains_heart",
            "is_localizer",
            "is_derived",
            "sequence_type",
            "view_type",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")

        for row in reader:
            contains_heart = _parse_bool(row["contains_heart"])
            is_localizer = _parse_bool(row["is_localizer"])
            is_derived = _parse_bool(row["is_derived"])
            sequence = row["sequence_type"].strip().lower()
            view = row["view_type"].strip().lower()
            if None in {contains_heart, is_localizer, is_derived}:
                continue
            if CFG.require_contains_heart and not contains_heart:
                continue
            if CFG.exclude_localizers and is_localizer:
                continue
            if CFG.exclude_derived_exports and is_derived:
                continue
            if not sequence or not view:
                continue
            cell = f"{sequence}::{view}"
            if CFG.allowed_protocol_cells and cell not in set(
                CFG.allowed_protocol_cells
            ):
                continue
            mapping[row["series_id"].strip()] = cell

    if not mapping:
        raise ValueError("The annotation CSV retained no usable reviewed series.")
    logger.info(
        "Protocol-aware mode enabled: %d reviewed series, %d sequence/view cells.",
        len(mapping),
        len(set(mapping.values())),
    )
    return mapping


# =============================================================================
# 5. LABEL-BLIND IMAGE STANDARDIZATION
# =============================================================================


def _is_dark_uniform_line(line: np.ndarray) -> bool:
    values = np.asarray(line, dtype=np.float32)
    return bool(
        values.mean() <= CFG.dark_line_max_mean
        and values.std() <= CFG.dark_line_max_std
        and np.mean(values <= CFG.dark_pixel_max_value)
        >= CFG.dark_pixel_min_fraction
    )


def detect_dark_padding(image: np.ndarray) -> tuple[int, int, int, int]:
    """Detect only consecutive dark, nearly uniform native edge runs."""

    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image, got {image.shape}.")
    height, width = image.shape
    min_h = max(8, int(math.ceil(height * CFG.min_retained_fraction)))
    min_w = max(8, int(math.ceil(width * CFG.min_retained_fraction)))
    max_v = int(math.floor(height * CFG.max_crop_fraction_per_side))
    max_h = int(math.floor(width * CFG.max_crop_fraction_per_side))

    top = 0
    while top < max_v and height - top - 1 >= min_h and _is_dark_uniform_line(image[top]):
        top += 1
    bottom_crop = 0
    while (
        bottom_crop < max_v
        and height - top - bottom_crop - 1 >= min_h
        and _is_dark_uniform_line(image[height - 1 - bottom_crop])
    ):
        bottom_crop += 1
    left = 0
    while left < max_h and width - left - 1 >= min_w and _is_dark_uniform_line(image[:, left]):
        left += 1
    right_crop = 0
    while (
        right_crop < max_h
        and width - left - right_crop - 1 >= min_w
        and _is_dark_uniform_line(image[:, width - 1 - right_crop])
    ):
        right_crop += 1

    if top < CFG.min_padding_run:
        top = 0
    if bottom_crop < CFG.min_padding_run:
        bottom_crop = 0
    if left < CFG.min_padding_run:
        left = 0
    if right_crop < CFG.min_padding_run:
        right_crop = 0

    bottom = height - bottom_crop
    right = width - right_crop
    if bottom - top < min_h:
        top, bottom = 0, height
    if right - left < min_w:
        left, right = 0, width
    return top, bottom, left, right


def _resize_fixed_content(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize the longest retained side to 240 and center in a 256 canvas."""

    height, width = image.shape
    scale = CFG.fixed_content_long_side / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    if height >= width:
        new_h = CFG.fixed_content_long_side
    else:
        new_w = CFG.fixed_content_long_side
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32)

    canvas = np.zeros((CFG.monai_size, CFG.monai_size), dtype=np.float32)
    content = np.zeros_like(canvas)
    top = (CFG.monai_size - new_h) // 2
    left = (CFG.monai_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    content[top : top + new_h, left : left + new_w] = 1.0
    return canvas, content


def standardize_image(
    image: np.ndarray,
    file_size_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build raw, MONAI and content canvases plus interpretable metadata."""

    height, width = image.shape
    top, bottom, left, right = detect_dark_padding(image)
    cropped = image[top:bottom, left:right]
    if cropped.size == 0:
        raise RuntimeError("Border removal produced an empty image.")

    raw = cropped.astype(np.float32) / 255.0
    minimum = float(raw.min())
    maximum = float(raw.max())
    if maximum > minimum:
        monai = (raw - minimum) / (maximum - minimum)
    else:
        monai = np.zeros_like(raw)

    raw_canvas, content_canvas = _resize_fixed_content(raw)
    monai_canvas, monai_content = _resize_fixed_content(monai)
    if not np.array_equal(content_canvas, monai_content):
        raise RuntimeError("Raw and MONAI canvas geometries differ.")

    retained_fraction = ((bottom - top) * (right - left)) / float(height * width)
    metadata = np.asarray(
        [
            float(height),
            float(width),
            float(width / max(height, 1)),
            float(file_size_bytes / max(height * width, 1)),
            float(1.0 - retained_fraction),
            float(1.0 - content_canvas.mean()),
            0.0,  # replaced after MONAI inference
            0.0,  # replaced after exact support construction
        ],
        dtype=np.float32,
    )
    return raw_canvas, monai_canvas, content_canvas, metadata


class CardiacDataset(Dataset):
    """Decode each JPEG once and return aligned, label-blind tensors."""

    def __init__(self, records: Sequence[ImageRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        record = self.records[index]
        image = cv2.imread(record.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Unreadable image: {record.path}")
        decoded_hash = hashlib.sha256(image.tobytes()).hexdigest()
        raw, monai, content, metadata = standardize_image(
            image, os.path.getsize(record.path)
        )
        return (
            torch.from_numpy(raw).unsqueeze(0),
            torch.from_numpy(monai).unsqueeze(0),
            torch.from_numpy(content).unsqueeze(0),
            torch.from_numpy(metadata),
            torch.tensor(record.label, dtype=torch.long),
            record.patient_id,
            record.series_id,
            record.path,
            decoded_hash,
            torch.tensor(index, dtype=torch.long),
        )


# =============================================================================
# 6. FROZEN MODELS
# =============================================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_monai_segmenter(cache_dir: Path, logger: logging.Logger) -> torch.jit.ScriptModule:
    """Download and verify the pinned official MONAI TorchScript model."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Install huggingface_hub: pip install huggingface_hub"
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=CFG.monai_repo_id,
            filename="models/model.ts",
            revision=CFG.monai_revision,
            local_dir=str(cache_dir),
        )
    )
    actual = sha256_file(path)
    if actual != CFG.monai_torchscript_sha256:
        raise RuntimeError(
            "MONAI checksum mismatch. "
            f"Expected {CFG.monai_torchscript_sha256}, received {actual}."
        )
    model = torch.jit.load(str(path), map_location=DEVICE).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, CFG.monai_size, CFG.monai_size, device=DEVICE))
    if tuple(output.shape) != (1, 4, CFG.monai_size, CFG.monai_size):
        raise RuntimeError(f"Unexpected MONAI output shape: {tuple(output.shape)}")
    logger.info("Loaded verified MONAI segmenter from %s.", path)
    return model


class EfficientNetEncoder(nn.Module):
    """ImageNet EfficientNet-B0 with the classification head removed."""

    def __init__(self) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        network = models.efficientnet_b0(weights=weights)
        network.classifier = nn.Identity()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def load_encoder(logger: logging.Logger) -> EfficientNetEncoder:
    encoder = EfficientNetEncoder().to(DEVICE).eval()
    encoder.requires_grad_(False)
    logger.info("Loaded frozen EfficientNet-B0 encoder on %s.", DEVICE)
    return encoder


# =============================================================================
# 7. EXACT A17 SUPPORT AND FOUR IMAGE VIEWS
# =============================================================================


@torch.inference_mode()
def predict_heart_mask(
    monai_images: torch.Tensor,
    segmenter: torch.jit.ScriptModule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dilated hard masks at 224×224 and a per-slice validity flag."""

    logits = segmenter(monai_images)
    probabilities = torch.softmax(logits.float(), dim=1)
    heart_probability = probabilities[:, 1:].sum(dim=1, keepdim=True)
    class_map = probabilities.argmax(dim=1, keepdim=True)
    hard_256 = (class_map > 0).float()
    area = hard_256.mean(dim=(1, 2, 3))
    peak = heart_probability.amax(dim=(1, 2, 3))
    valid = (
        (area >= CFG.min_heart_area_ratio)
        & (area <= CFG.max_heart_area_ratio)
        & (peak >= CFG.min_peak_heart_probability)
    )
    hard_256 = F.max_pool2d(
        hard_256,
        kernel_size=CFG.monai_dilation_kernel,
        stride=1,
        padding=CFG.monai_dilation_kernel // 2,
    )
    hard_224 = F.interpolate(
        hard_256,
        size=(CFG.classifier_size, CFG.classifier_size),
        mode="nearest",
    )
    return hard_224, valid


def build_exact_a17_support(
    hard_mask: torch.Tensor,
    valid: torch.Tensor,
    content_mask: torch.Tensor,
) -> torch.Tensor:
    """Build one exact support reused by heart, mask, shuffled and outside views."""

    support = F.max_pool2d(
        (hard_mask > 0.5).float(),
        kernel_size=CFG.a17_extra_dilation_kernel,
        stride=1,
        padding=CFG.a17_extra_dilation_kernel // 2,
    )
    batch, _, height, width = support.shape
    side = int(round(min(height, width) * CFG.invalid_mask_center_fraction))
    side = max(2, min(side, height, width))
    top = (height - side) // 2
    left = (width - side) // 2
    fallback = torch.zeros_like(support)
    fallback[:, :, top : top + side, left : left + side] = 1.0
    valid_selector = valid.view(batch, 1, 1, 1)
    support = torch.where(valid_selector, support, fallback)
    return (support * (content_mask > 0.5).float()).clamp(0.0, 1.0)


def robust_scale_visible_regions(
    images: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Scale each image using only pixels that remain visible in its final view.

    A fixed 256-bin masked histogram approximates the 1st and 99th percentiles.
    Because source JPEG intensities are 8-bit, this retains the natural intensity
    resolution while avoiding a slow per-image sort.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W].")
    if masks.shape != images[:, :1].shape:
        raise ValueError("masks must have shape [B,1,H,W].")

    batch = images.shape[0]
    mask = masks > 0.5
    values = images[:, 0].float().clamp(0.0, 1.0)
    flat_mask = mask[:, 0].reshape(batch, -1)
    counts = flat_mask.sum(dim=1).long()

    bins = CFG.histogram_bins
    bin_index = torch.round(values * (bins - 1)).long().clamp(0, bins - 1)
    bin_index = bin_index.reshape(batch, -1)
    histogram = torch.zeros(batch, bins, device=images.device)
    histogram.scatter_add_(1, bin_index, flat_mask.float())
    cumulative = histogram.cumsum(dim=1)

    safe_count = counts.clamp_min(1)
    lower_rank = (
        torch.floor((CFG.lower_percentile / 100.0) * (safe_count - 1).float()).long()
        + 1
    )
    upper_rank = (
        torch.floor((CFG.upper_percentile / 100.0) * (safe_count - 1).float()).long()
        + 1
    )
    lower_bin = (cumulative >= lower_rank[:, None]).long().argmax(dim=1)
    upper_bin = (cumulative >= upper_rank[:, None]).long().argmax(dim=1)
    lower = lower_bin.float() / (bins - 1)
    upper = upper_bin.float() / (bins - 1)
    valid_range = (
        (counts >= CFG.min_visible_pixels)
        & torch.isfinite(lower)
        & torch.isfinite(upper)
        & (upper > lower)
    )

    lower = lower[:, None, None, None]
    upper = upper[:, None, None, None]
    denominator = (upper - lower).clamp_min(CFG.min_dynamic_range)
    scaled = ((images - lower) / denominator).clamp(0.0, 1.0)
    scaled = scaled * mask.float()
    scaled = scaled * valid_range[:, None, None, None].float()
    return scaled


def _coprime_affine_parameters(pixel_hash: str, n: int) -> tuple[int, int]:
    """Construct a deterministic full permutation of n visible positions."""

    if n <= 1:
        return 1, 0
    a = 2 + int(pixel_hash[:16], 16) % max(1, n - 2)
    a %= n
    if a == 0:
        a = 1
    while math.gcd(a, n) != 1:
        a += 1
        if a >= n:
            a = 1
    b = int(pixel_hash[16:32], 16) % n
    if a == 1 and b == 0:
        b = 1
    return a, b


def shuffle_inside_support(
    heart_images: torch.Tensor,
    support: torch.Tensor,
    pixel_hashes: Sequence[str],
) -> torch.Tensor:
    """Preserve support and intensity multiset while destroying spatial anatomy."""

    output = torch.zeros_like(heart_images)
    for index in range(heart_images.shape[0]):
        positions = torch.nonzero(
            support[index, 0].reshape(-1) > 0.5,
            as_tuple=False,
        ).reshape(-1)
        n = int(positions.numel())
        if n == 0:
            continue
        values = heart_images[index, 0].reshape(-1)[positions]
        a, b = _coprime_affine_parameters(str(pixel_hashes[index]), n)
        destination = torch.arange(n, device=heart_images.device)
        source = (destination * a + b) % n
        shuffled = values[source]
        output[index].reshape(3, -1)[:, positions] = shuffled[None, :]
    return output


def build_four_views(
    raw_canvas: torch.Tensor,
    content_canvas: torch.Tensor,
    hard_mask: torch.Tensor,
    valid: torch.Tensor,
    pixel_hashes: Sequence[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Create the four matched image views from one exact support tensor."""

    raw_224 = F.interpolate(
        raw_canvas,
        size=(CFG.classifier_size, CFG.classifier_size),
        mode="bilinear",
        align_corners=False,
    ).repeat(1, 3, 1, 1)
    content_224 = F.interpolate(
        content_canvas,
        size=(CFG.classifier_size, CFG.classifier_size),
        mode="nearest",
    )
    support = build_exact_a17_support(hard_mask, valid, content_224)
    heart = robust_scale_visible_regions(raw_224, support)
    outside_mask = ((content_224 > 0.5) & (support <= 0.5)).float()
    outside = robust_scale_visible_regions(raw_224, outside_mask)
    mask_only = support.repeat(1, 3, 1, 1)
    shuffled = shuffle_inside_support(heart, support, pixel_hashes)
    return {
        "heart": heart,
        "mask": mask_only,
        "shuffled": shuffled,
        "outside": outside,
    }, support


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def normalize_for_encoder(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images - mean) / std


# =============================================================================
# 8. FOUR-VIEW FEATURE BANK AND EXACT-DUPLICATE AUDIT
# =============================================================================


def feature_fingerprint(records: Sequence[ImageRecord]) -> str:
    """Hash file identity and all feature-producing settings."""

    digest = hashlib.sha256()
    settings = {
        "cache_version": CFG.feature_cache_version,
        "records": len(records),
        "monai_revision": CFG.monai_revision,
        "monai_sha": CFG.monai_torchscript_sha256,
        "classifier_size": CFG.classifier_size,
        "content_size": CFG.fixed_content_long_side,
        "dilation": CFG.monai_dilation_kernel,
        "extra_dilation": CFG.a17_extra_dilation_kernel,
        "gate": [
            CFG.min_heart_area_ratio,
            CFG.max_heart_area_ratio,
            CFG.min_peak_heart_probability,
        ],
        "normalization": [
            CFG.lower_percentile,
            CFG.upper_percentile,
            CFG.histogram_bins,
            CFG.min_dynamic_range,
        ],
        "amp": CFG.use_amp,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for record in records:
        stat = os.stat(record.path)
        digest.update(record.path.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def _allocate_feature_files(cache_dir: Path, n_rows: int) -> dict[str, np.memmap]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        mode: np.lib.format.open_memmap(
            cache_dir / f"{mode}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_rows, EMBEDDING_DIM),
        )
        for mode in IMAGE_MODES
    }


@torch.inference_mode()
def extract_feature_bank(
    records: Sequence[ImageRecord],
    cache_dir: Path,
    segmenter: torch.jit.ScriptModule,
    encoder: EfficientNetEncoder,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Encode four views and save one aligned row per decoded image."""

    n_rows = len(records)
    feature_files = _allocate_feature_files(cache_dir, n_rows)
    labels = np.empty(n_rows, dtype=np.int8)
    valid = np.empty(n_rows, dtype=bool)
    metadata = np.empty((n_rows, len(SLICE_METADATA_NAMES)), dtype=np.float32)
    patient_ids: list[str] = [""] * n_rows
    series_ids: list[str] = [""] * n_rows
    paths: list[str] = [""] * n_rows
    hashes: list[str] = [""] * n_rows

    dataset = CardiacDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=DEVICE.type == "cuda",
    )

    offset = 0
    for batch in tqdm(loader, desc="Four-view feature bank", unit="batch"):
        (
            raw,
            monai,
            content,
            batch_metadata,
            batch_labels,
            batch_patients,
            batch_series,
            batch_paths,
            batch_hashes,
            batch_indices,
        ) = batch
        raw = raw.to(DEVICE, non_blocking=True)
        monai = monai.to(DEVICE, non_blocking=True)
        content = content.to(DEVICE, non_blocking=True)
        hard, batch_valid = predict_heart_mask(monai, segmenter)
        views, support = build_four_views(
            raw, content, hard, batch_valid, list(batch_hashes)
        )

        # Fill MONAI validity and exact-support fraction in the metadata matrix.
        batch_metadata = batch_metadata.numpy().astype(np.float32)
        batch_metadata[:, 6] = batch_valid.cpu().numpy().astype(np.float32)
        batch_metadata[:, 7] = (
            support.mean(dim=(1, 2, 3)).cpu().numpy().astype(np.float32)
        )

        batch_size = len(batch_patients)
        end = offset + batch_size
        labels[offset:end] = batch_labels.numpy().astype(np.int8)
        valid[offset:end] = batch_valid.cpu().numpy()
        metadata[offset:end] = batch_metadata
        patient_ids[offset:end] = list(batch_patients)
        series_ids[offset:end] = list(batch_series)
        paths[offset:end] = list(batch_paths)
        hashes[offset:end] = list(batch_hashes)

        # Encode all four views. They can be concatenated because every view uses
        # identical 224×224 geometry and ImageNet normalization.
        for start in range(0, len(IMAGE_MODES), CFG.encoder_modes_per_call):
            mode_chunk = IMAGE_MODES[start : start + CFG.encoder_modes_per_call]
            combined = torch.cat(
                [normalize_for_encoder(views[mode]) for mode in mode_chunk], dim=0
            )
            if CFG.use_amp and DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    encoded = encoder(combined)
            else:
                encoded = encoder(combined)
            encoded = encoded.float().cpu().numpy()
            for chunk_index, mode in enumerate(mode_chunk):
                begin = chunk_index * batch_size
                finish = begin + batch_size
                feature_files[mode][offset:end] = encoded[begin:finish]

        offset = end

    if offset != n_rows:
        raise RuntimeError(f"Feature bank wrote {offset}/{n_rows} rows.")
    for array in feature_files.values():
        array.flush()

    np.savez_compressed(
        cache_dir / "rows.npz",
        labels=labels,
        valid=valid,
        metadata=metadata,
        patient_ids=np.asarray(patient_ids),
        series_ids=np.asarray(series_ids),
        paths=np.asarray(paths),
        hashes=np.asarray(hashes),
    )
    (cache_dir / "cache.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_rows": n_rows,
                "modes": list(IMAGE_MODES),
                "embedding_dim": EMBEDDING_DIM,
                "metadata_names": list(SLICE_METADATA_NAMES),
                "config": asdict(CFG),
            },
            indent=2,
            default=str,
            sort_keys=True,
        )
    )
    logger.info("Feature bank completed: %d rows × %d image modes.", n_rows, 4)
    return load_feature_bank(cache_dir)


def load_feature_bank(cache_dir: Path) -> dict[str, Any]:
    metadata_path = cache_dir / "cache.json"
    rows_path = cache_dir / "rows.npz"
    if not metadata_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError("Incomplete feature cache.")
    info = json.loads(metadata_path.read_text())
    if info.get("status") != "complete":
        raise RuntimeError("Feature cache is not marked complete.")
    n_rows = int(info["n_rows"])
    bank: dict[str, Any] = {"features": {}}
    for mode in IMAGE_MODES:
        path = cache_dir / f"{mode}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cache view: {path}")
        array = np.load(path, mmap_mode="r")
        if array.shape != (n_rows, EMBEDDING_DIM):
            raise RuntimeError(f"Unexpected {mode} cache shape: {array.shape}")
        bank["features"][mode] = array
    with np.load(rows_path, allow_pickle=False) as rows:
        for key in rows.files:
            bank[key] = rows[key].copy()
    return bank


def get_or_build_feature_bank(
    records: Sequence[ImageRecord],
    logger: logging.Logger,
) -> tuple[dict[str, Any], Path]:
    fingerprint = feature_fingerprint(records)
    cache_dir = CFG.output_dir / "cache" / fingerprint
    if (cache_dir / "cache.json").is_file():
        logger.info("Loading feature cache: %s", cache_dir)
        return load_feature_bank(cache_dir), cache_dir

    logger.info("Cache miss. Building four-view feature bank on %s.", DEVICE)
    segmenter = load_monai_segmenter(CFG.output_dir / "models", logger)
    encoder = load_encoder(logger)
    return extract_feature_bank(records, cache_dir, segmenter, encoder, logger), cache_dir


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def audit_exact_duplicates(
    bank: dict[str, Any], output_dir: Path, logger: logging.Logger
) -> dict[str, str]:
    """Group patients sharing exact decoded pixels so folds cannot split them."""

    hashes = np.asarray(bank["hashes"]).astype(str)
    patients = np.asarray(bank["patient_ids"]).astype(str)
    labels = np.asarray(bank["labels"], dtype=int)
    unique_patients = sorted(set(patients.tolist()))
    union = UnionFind(unique_patients)
    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(hashes):
        hash_to_indices[value].append(index)

    rows = []
    cross_patient = 0
    for value, indices in hash_to_indices.items():
        involved = sorted(set(patients[indices].tolist()))
        if len(involved) <= 1:
            continue
        involved_labels = sorted(set(labels[indices].tolist()))
        if len(involved_labels) > 1:
            raise RuntimeError(
                "An exact decoded image appears across opposite labels: "
                f"hash={value}, patients={involved}."
            )
        cross_patient += 1
        for patient in involved[1:]:
            union.union(involved[0], patient)
        rows.append(
            {
                "decoded_sha256": value,
                "patients": "|".join(involved),
                "label": involved_labels[0],
                "image_rows": len(indices),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "exact_cross_patient_duplicates.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["decoded_sha256", "patients", "label", "image_rows"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Exact cross-patient duplicate groups: %d.", cross_patient)
    return {patient: union.find(patient) for patient in unique_patients}


# =============================================================================
# 9. CUSTOM MATHEMATICAL ALGORITHMS
# =============================================================================


def l2_normalize_rows(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, epsilon)


def geometric_median(vectors: np.ndarray) -> np.ndarray:
    """Weiszfeld iterations for the multivariate geometric median."""

    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) == 1:
        return vectors[0].copy()
    estimate = np.mean(vectors, axis=0)
    for _ in range(CFG.rsc_max_iterations):
        distance = np.linalg.norm(vectors - estimate, axis=1)
        if np.any(distance < CFG.rsc_distance_epsilon):
            estimate = vectors[np.argmin(distance)].copy()
            break
        weight = 1.0 / np.maximum(distance, CFG.rsc_distance_epsilon)
        updated = np.average(vectors, axis=0, weights=weight)
        if np.linalg.norm(updated - estimate) <= CFG.rsc_tolerance:
            estimate = updated
            break
        estimate = updated
    return estimate


def robust_series_consensus(series_vectors: np.ndarray) -> np.ndarray:
    """Project-specific robust pooling over a patient's series embeddings.

    Mathematical idea
    -----------------
    1. Normalize each series direction to prevent raw magnitude dominating.
    2. Find the geometric median g using Weiszfeld's algorithm.
    3. Measure angular outlyingness d_j = ||s_j - g||.
    4. Assign Cauchy weights w_j = 1 / (1 + (d_j / scale)^2).
    5. Average the original, unnormalized series vectors with those weights.

    A few unusual localizers, derived exports, or protocol-specific series can no
    longer dominate the patient vector merely by being far from the consensus.
    """

    vectors = np.asarray(series_vectors, dtype=np.float64)
    if len(vectors) == 1:
        return vectors[0].astype(np.float32)
    unit = l2_normalize_rows(vectors)
    center = geometric_median(unit)
    distance = np.linalg.norm(unit - center, axis=1)
    median_distance = float(np.median(distance))
    mad = float(np.median(np.abs(distance - median_distance)))
    scale = max(
        median_distance + CFG.rsc_cauchy_scale * mad,
        CFG.rsc_distance_epsilon,
    )
    weights = 1.0 / (1.0 + np.square(distance / scale))
    pooled = np.average(vectors, axis=0, weights=weights)
    return pooled.astype(np.float32)


def mean_pool(series_vectors: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(series_vectors), axis=0).astype(np.float32)


def protocol_balanced_pool(
    series_vectors: np.ndarray,
    series_ids: Sequence[str],
    series_to_cell: dict[str, str],
    robust: bool,
) -> np.ndarray:
    """Equalize reviewed sequence/view cells before patient pooling.

    Series are first averaged inside each human-reviewed protocol cell. Every
    resulting cell receives one vote, regardless of how many exported folders it
    contains. This directly targets protocol composition confounding.
    """

    if not series_to_cell:
        return robust_series_consensus(series_vectors) if robust else mean_pool(series_vectors)
    cell_to_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for vector, series_id in zip(series_vectors, series_ids):
        cell = series_to_cell.get(str(series_id))
        if cell is not None:
            cell_to_vectors[cell].append(vector)
    if not cell_to_vectors:
        raise RuntimeError("A patient retained no human-reviewed protocol cells.")
    cell_vectors = np.stack(
        [np.mean(cell_to_vectors[cell], axis=0) for cell in sorted(cell_to_vectors)]
    )
    return robust_series_consensus(cell_vectors) if robust else mean_pool(cell_vectors)


class FoldLocalConfounderOrthogonalizer:
    """Ridge-residualize cardiac embeddings against nuisance only in training.

    Given cardiac features X and nuisance features Z, the training fold solves

        B = (Z^T Z + lambda I)^(-1) Z^T X
        X_clean = X - Z B

    after training-only centering, scaling, and low-rank SVD of Z. Validation
    patients use the same learned means, scales, nuisance basis and B matrix.
    """

    def __init__(self, ridge_lambda: float):
        self.ridge_lambda = float(ridge_lambda)

    def fit(self, x: np.ndarray, z: np.ndarray) -> "FoldLocalConfounderOrthogonalizer":
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        self.x_mean_ = x.mean(axis=0)
        self.z_mean_ = z.mean(axis=0)
        self.z_scale_ = z.std(axis=0)
        self.z_scale_[self.z_scale_ < 1e-8] = 1.0
        z_standard = (z - self.z_mean_) / self.z_scale_

        _, singular_values, right_vectors = np.linalg.svd(
            z_standard, full_matrices=False
        )
        variance = np.square(singular_values)
        if variance.sum() <= 0:
            rank = 0
        else:
            cumulative = np.cumsum(variance) / variance.sum()
            rank = int(np.searchsorted(cumulative, CFG.nuisance_variance) + 1)
        rank = min(rank, CFG.max_nuisance_components, max(0, len(x) - 2))
        self.nuisance_basis_ = right_vectors[:rank].T if rank > 0 else np.empty((z.shape[1], 0))

        scores = z_standard @ self.nuisance_basis_
        self.nuisance_median_ = (
            np.median(scores, axis=0) if rank > 0 else np.empty(0)
        )
        x_centered = x - self.x_mean_
        if rank == 0:
            self.coefficient_ = np.empty((0, x.shape[1]))
        else:
            gram = scores.T @ scores
            penalty = self.ridge_lambda * np.eye(rank)
            self.coefficient_ = np.linalg.solve(
                gram + penalty, scores.T @ x_centered
            )
        return self

    def nuisance_scores(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        standardized = (z - self.z_mean_) / self.z_scale_
        return standardized @ self.nuisance_basis_

    def transform(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        scores = self.nuisance_scores(z)
        predicted_nuisance = scores @ self.coefficient_
        return (x - self.x_mean_ - predicted_nuisance).astype(np.float32)

    def transform_counterfactual_median(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.coefficient_.shape[0] == 0:
            predicted = np.zeros_like(x)
        else:
            predicted = np.repeat(
                (self.nuisance_median_ @ self.coefficient_)[None, :],
                len(x),
                axis=0,
            )
        return (x - self.x_mean_ - predicted).astype(np.float32)


def paired_regional_contrast(heart: np.ndarray, outside: np.ndarray) -> np.ndarray:
    """Subtract matched outside representation from the cardiac representation.

    Both branches use the same frozen encoder. L2 normalization makes subtraction
    compare direction rather than raw activation scale:

        x_contrast = normalize(x_heart) - normalize(x_outside)

    Shared export/style directions can cancel, while region-specific directions
    are retained.
    """

    contrast = l2_normalize_rows(heart) - l2_normalize_rows(outside)
    return l2_normalize_rows(contrast).astype(np.float32)


# =============================================================================
# 10. PATIENT REPRESENTATIONS
# =============================================================================


def build_patient_representations(
    bank: dict[str, Any],
    series_to_cell: dict[str, str],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Aggregate slices to series, then series/protocol cells to patients."""

    patient_ids = np.asarray(bank["patient_ids"]).astype(str)
    series_ids = np.asarray(bank["series_ids"]).astype(str)
    labels = np.asarray(bank["labels"], dtype=int)
    valid = np.asarray(bank["valid"], dtype=bool)
    metadata = np.asarray(bank["metadata"], dtype=np.float32)
    hashes = np.asarray(bank["hashes"]).astype(str)

    selected_rows = np.ones(len(labels), dtype=bool)
    if CFG.deduplicate_exact_within_patient:
        keep = np.zeros(len(labels), dtype=bool)
        seen: set[tuple[str, str]] = set()
        for index in range(len(labels)):
            key = (patient_ids[index], hashes[index])
            if key in seen:
                continue
            seen.add(key)
            keep[index] = True
        selected_rows &= keep
        logger.info(
            "Exact within-patient deduplication retained %d/%d image rows.",
            int(keep.sum()),
            len(keep),
        )

    if series_to_cell:
        selected_rows = np.asarray(
            [series_id in series_to_cell for series_id in series_ids], dtype=bool
        )
        if not np.any(selected_rows):
            raise RuntimeError("No feature rows match the reviewed annotations.")

    patients = np.asarray(sorted(set(patient_ids[selected_rows].tolist())))
    patient_label: dict[str, int] = {}
    for patient in patients:
        values = np.unique(labels[(patient_ids == patient) & selected_rows])
        if len(values) != 1:
            raise RuntimeError(f"Inconsistent labels for {patient}.")
        patient_label[patient] = int(values[0])

    # patient -> mode -> series -> row indices
    patient_series_indices: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for patient in patients:
        patient_mask = (patient_ids == patient) & selected_rows
        for series in sorted(set(series_ids[patient_mask].tolist())):
            patient_series_indices[patient][series] = np.where(
                patient_mask & (series_ids == series)
            )[0]

    mean_features: dict[str, list[np.ndarray]] = {
        mode: [] for mode in IMAGE_MODES
    }
    robust_features: dict[str, list[np.ndarray]] = {
        mode: [] for mode in IMAGE_MODES
    }
    heart_valid_features: list[np.ndarray] = []
    patient_metadata: list[np.ndarray] = []

    for patient in patients:
        all_patient_rows = np.where((patient_ids == patient) & selected_rows)[0]
        series_names = sorted(patient_series_indices[patient])

        for mode in IMAGE_MODES:
            series_vectors = np.stack(
                [
                    np.mean(
                        np.asarray(bank["features"][mode][
                            patient_series_indices[patient][series]
                        ]),
                        axis=0,
                    )
                    for series in series_names
                ]
            )
            mean_features[mode].append(
                protocol_balanced_pool(
                    series_vectors, series_names, series_to_cell, robust=False
                )
            )
            robust_features[mode].append(
                protocol_balanced_pool(
                    series_vectors, series_names, series_to_cell, robust=True
                )
            )

        valid_series_vectors: list[np.ndarray] = []
        valid_series_names: list[str] = []
        for series in series_names:
            rows = patient_series_indices[patient][series]
            rows = rows[valid[rows]]
            if len(rows) == 0:
                continue
            valid_series_vectors.append(
                np.mean(np.asarray(bank["features"]["heart"][rows]), axis=0)
            )
            valid_series_names.append(series)
        if not valid_series_vectors:
            raise RuntimeError(f"No valid MONAI slices remain for {patient}.")
        heart_valid_features.append(
            protocol_balanced_pool(
                np.stack(valid_series_vectors),
                valid_series_names,
                series_to_cell,
                robust=False,
            )
        )

        meta = metadata[all_patient_rows]
        if series_to_cell:
            cell_counts: dict[str, int] = defaultdict(int)
            for series in series_names:
                cell = series_to_cell.get(series)
                if cell is not None:
                    cell_counts[cell] += 1
            count_values = np.asarray(list(cell_counts.values()), dtype=np.float64)
            if count_values.size:
                probabilities = count_values / count_values.sum()
                protocol_cell_count = float(len(count_values))
                protocol_entropy = float(
                    -np.sum(probabilities * np.log(probabilities + 1e-12))
                )
                protocol_max_fraction = float(probabilities.max())
            else:
                protocol_cell_count = protocol_entropy = protocol_max_fraction = 0.0
        else:
            protocol_cell_count = protocol_entropy = protocol_max_fraction = 0.0

        patient_metadata.append(
            np.asarray(
                [
                    math.log1p(len(all_patient_rows)),
                    math.log1p(len(series_names)),
                    meta[:, 2].mean(),
                    meta[:, 2].std(),
                    meta[:, 3].mean(),
                    meta[:, 3].std(),
                    meta[:, 4].mean(),
                    meta[:, 5].mean(),
                    meta[:, 6].mean(),
                    meta[:, 7].mean(),
                    meta[:, 7].std(),
                    protocol_cell_count,
                    protocol_entropy,
                    protocol_max_fraction,
                ],
                dtype=np.float32,
            )
        )

    mean_arrays = {mode: np.stack(values) for mode, values in mean_features.items()}
    robust_arrays = {mode: np.stack(values) for mode, values in robust_features.items()}
    metadata_array = np.stack(patient_metadata)
    labels_array = np.asarray([patient_label[p] for p in patients], dtype=int)

    # Nuisance representation deliberately excludes the shuffled-heart diagnostic.
    # It combines support geometry, outside-region embeddings and interpretable
    # export/protocol metadata.
    nuisance_mean = np.concatenate(
        [
            l2_normalize_rows(mean_arrays["mask"]),
            l2_normalize_rows(mean_arrays["outside"]),
            metadata_array,
        ],
        axis=1,
    ).astype(np.float32)
    nuisance_robust = np.concatenate(
        [
            l2_normalize_rows(robust_arrays["mask"]),
            l2_normalize_rows(robust_arrays["outside"]),
            metadata_array,
        ],
        axis=1,
    ).astype(np.float32)

    logger.info(
        "Built patient tables: %d patients, mean embedding=%s, nuisance=%s.",
        len(patients),
        mean_arrays["heart"].shape,
        nuisance_mean.shape,
    )
    return {
        "patient_ids": patients,
        "labels": labels_array,
        "mean": mean_arrays,
        "robust": robust_arrays,
        "heart_valid": np.stack(heart_valid_features),
        "metadata": metadata_array,
        "nuisance_mean": nuisance_mean,
        "nuisance_robust": nuisance_robust,
        "protocol_balanced": bool(series_to_cell),
    }


def experiment_matrix(
    experiment: Experiment,
    tables: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return candidate X and optional nuisance Z for one experiment."""

    pool = tables["robust"] if experiment.pooling == "rsc" else tables["mean"]
    if experiment.feature_source == "heart":
        x = pool["heart"]
    elif experiment.feature_source == "heart_valid":
        x = tables["heart_valid"]
    elif experiment.feature_source == "contrast":
        x = paired_regional_contrast(pool["heart"], pool["outside"])
    elif experiment.feature_source in {"mask", "shuffled", "outside"}:
        x = pool[experiment.feature_source]
    elif experiment.feature_source == "metadata":
        x = tables["metadata"]
    elif experiment.feature_source == "nuisance":
        x = tables["nuisance_mean"]
    else:
        raise ValueError(f"Unknown feature source: {experiment.feature_source}")

    if experiment.deconfound:
        z = (
            tables["nuisance_robust"]
            if experiment.pooling == "rsc"
            else tables["nuisance_mean"]
        )
    else:
        z = None
    return np.asarray(x, dtype=np.float32), z


# =============================================================================
# 11. PATIENT-LEVEL NESTED CROSS-VALIDATION
# =============================================================================


def make_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratify patients and keep exact-duplicate components together."""

    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups).astype(str)
    if len(set(groups.tolist())) == len(groups):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(labels)), labels))
    if StratifiedGroupKFold is None:
        raise RuntimeError(
            "Cross-patient exact duplicates require sklearn StratifiedGroupKFold."
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def build_classifier(c_value: float, use_pca: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=CFG.pca_variance, svd_solver="full")))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=5000,
                solver="liblinear",
                random_state=CFG.seed,
            ),
        )
    )
    return Pipeline(steps)


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    objective = tpr[finite] - fpr[finite]
    candidates = thresholds[finite]
    return float(candidates[int(np.argmax(objective))])


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def fit_transform_candidate(
    x_train: np.ndarray,
    z_train: np.ndarray | None,
    x_valid: np.ndarray,
    z_valid: np.ndarray | None,
    ridge_lambda: float | None,
) -> tuple[np.ndarray, np.ndarray, FoldLocalConfounderOrthogonalizer | None]:
    if z_train is None:
        return x_train, x_valid, None
    if ridge_lambda is None:
        raise ValueError("A deconfounded experiment requires ridge_lambda.")
    transform = FoldLocalConfounderOrthogonalizer(ridge_lambda).fit(x_train, z_train)
    return (
        transform.transform(x_train, z_train),
        transform.transform(x_valid, z_valid),
        transform,
    )


def low_rank_nuisance_scores(z: np.ndarray) -> np.ndarray:
    """Return a small label-blind nuisance coordinate system for diagnostics."""

    z = np.asarray(z, dtype=np.float64)
    mean = z.mean(axis=0)
    scale = z.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (z - mean) / scale
    u, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    variance = np.square(singular_values)
    if variance.sum() <= 0:
        return np.zeros((len(z), 0), dtype=np.float64)
    cumulative = np.cumsum(variance) / variance.sum()
    rank = int(np.searchsorted(cumulative, CFG.nuisance_variance) + 1)
    rank = min(rank, CFG.max_nuisance_components, max(0, len(z) - 2))
    if rank == 0:
        return np.zeros((len(z), 0), dtype=np.float64)
    # U*S are the principal-component scores without constructing a large
    # feature-space basis again.
    return u[:, :rank] * singular_values[:rank]


def conditional_nuisance_dependence(
    scores: np.ndarray,
    nuisance_scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Measure score–nuisance association after removing class means.

    A raw correlation can be high simply because both score and nuisance differ
    between Normal and Sick. This diagnostic first centers both quantities within
    each class, then returns the largest absolute correlation with a low-rank
    nuisance component. Lower values indicate less within-class shortcut
    dependence. It is a project-specific diagnostic, not a standard clinical
    metric.
    """

    scores = np.asarray(scores, dtype=np.float64).copy()
    nuisance_scores = np.asarray(nuisance_scores, dtype=np.float64).copy()
    labels = np.asarray(labels, dtype=int)
    if nuisance_scores.shape[1] == 0:
        return 0.0
    for label in np.unique(labels):
        mask = labels == label
        scores[mask] -= scores[mask].mean()
        nuisance_scores[mask] -= nuisance_scores[mask].mean(axis=0)
    score_std = scores.std()
    if score_std < 1e-12:
        return 0.0
    correlations = []
    for column in range(nuisance_scores.shape[1]):
        component = nuisance_scores[:, column]
        if component.std() < 1e-12:
            continue
        correlations.append(abs(float(np.corrcoef(scores, component)[0, 1])))
    return max(correlations, default=0.0)


def inner_select_hyperparameters(
    experiment: Experiment,
    x: np.ndarray,
    z: np.ndarray | None,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """APPS: preserve near-best AUC, then minimize nuisance sensitivity.

    For every (C, lambda) candidate, inner OOF predictions are generated. APPS
    first retains configurations within ``auc_tolerance`` of the best AUC. Among
    these near-optimal candidates, it chooses the lowest class-conditional
    dependence between OOF scores and low-rank nuisance coordinates. Remaining
    ties prefer stronger classifier regularization. Counterfactual score change
    is saved as a separate interpretability diagnostic, not the selection target.
    """

    ridge_values: tuple[float | None, ...] = (
        CFG.ridge_grid if experiment.deconfound else (None,)
    )
    inner_splits = make_splits(labels, groups, CFG.inner_folds, seed)
    nuisance_summary = (
        low_rank_nuisance_scores(z)
        if z is not None
        else np.zeros((len(labels), 0), dtype=np.float64)
    )
    rows: list[dict[str, float]] = []

    for ridge_lambda in ridge_values:
        for c_value in CFG.c_grid:
            oof = np.full(len(labels), np.nan, dtype=np.float64)
            counterfactual_differences: list[float] = []
            for train_index, valid_index in inner_splits:
                x_train, x_valid, transform = fit_transform_candidate(
                    x[train_index],
                    None if z is None else z[train_index],
                    x[valid_index],
                    None if z is None else z[valid_index],
                    ridge_lambda,
                )
                model = build_classifier(c_value, experiment.use_pca)
                model.fit(x_train, labels[train_index])
                original = model.predict_proba(x_valid)[:, 1]
                oof[valid_index] = original
                if transform is not None:
                    counterfactual_x = transform.transform_counterfactual_median(
                        x[valid_index]
                    )
                    counterfactual = model.predict_proba(counterfactual_x)[:, 1]
                    counterfactual_differences.extend(
                        np.abs(original - counterfactual).tolist()
                    )

            auc = float(roc_auc_score(labels, oof))
            sensitivity = (
                float(np.mean(counterfactual_differences))
                if counterfactual_differences
                else 0.0
            )
            dependence = conditional_nuisance_dependence(
                oof, nuisance_summary, labels
            )
            rows.append(
                {
                    "c": float(c_value),
                    "ridge": float(ridge_lambda) if ridge_lambda is not None else -1.0,
                    "auc": auc,
                    "conditional_nuisance_dependence": dependence,
                    "counterfactual_score_change": sensitivity,
                    "threshold": choose_threshold(labels, oof),
                }
            )

    best_auc = max(row["auc"] for row in rows)
    eligible = [
        row for row in rows if row["auc"] >= best_auc - CFG.auc_tolerance
    ]
    eligible.sort(
        key=lambda row: (
            row["conditional_nuisance_dependence"],
            row["c"],
            row["ridge"],
        )
    )
    selected = dict(eligible[0])
    selected["best_inner_auc"] = best_auc
    return selected


def evaluate_once(
    experiment: Experiment,
    tables: dict[str, Any],
    patient_groups: dict[str, str],
    seed: int,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one experiment with strict patient-level nested CV."""

    x, z = experiment_matrix(experiment, tables)
    labels = np.asarray(tables["labels"], dtype=int)
    patients = np.asarray(tables["patient_ids"]).astype(str)
    groups = np.asarray([patient_groups[p] for p in patients])
    outer_splits = make_splits(labels, groups, CFG.outer_folds, seed)

    oof_scores = np.full(len(labels), np.nan)
    oof_predictions = np.full(len(labels), -1, dtype=int)
    oof_thresholds = np.full(len(labels), np.nan)
    fold_rows: list[dict[str, Any]] = []
    counterfactual_change = np.full(len(labels), np.nan)

    for fold_index, (train_index, valid_index) in enumerate(outer_splits, start=1):
        selected = inner_select_hyperparameters(
            experiment,
            x[train_index],
            None if z is None else z[train_index],
            labels[train_index],
            groups[train_index],
            seed + 1000 + fold_index,
        )
        ridge_lambda = None if selected["ridge"] < 0 else selected["ridge"]
        x_train, x_valid, transform = fit_transform_candidate(
            x[train_index],
            None if z is None else z[train_index],
            x[valid_index],
            None if z is None else z[valid_index],
            ridge_lambda,
        )
        model = build_classifier(selected["c"], experiment.use_pca)
        model.fit(x_train, labels[train_index])
        score = model.predict_proba(x_valid)[:, 1]
        threshold = float(selected["threshold"])
        prediction = (score >= threshold).astype(int)
        oof_scores[valid_index] = score
        oof_predictions[valid_index] = prediction
        oof_thresholds[valid_index] = threshold

        if transform is not None:
            counterfactual = model.predict_proba(
                transform.transform_counterfactual_median(x[valid_index])
            )[:, 1]
            counterfactual_change[valid_index] = np.abs(score - counterfactual)

        fold_rows.append(
            {
                "fold": fold_index,
                "selected_c": selected["c"],
                "selected_ridge": ridge_lambda,
                "selected_inner_auc": selected["auc"],
                "best_inner_auc": selected["best_inner_auc"],
                "inner_conditional_nuisance_dependence": selected[
                    "conditional_nuisance_dependence"
                ],
                "inner_counterfactual_score_change": selected[
                    "counterfactual_score_change"
                ],
                "threshold": threshold,
                "outer_auc": float(roc_auc_score(labels[valid_index], score)),
            }
        )

    if np.any(~np.isfinite(oof_scores)) or np.any(oof_predictions < 0):
        raise RuntimeError(f"Incomplete OOF predictions for {experiment.experiment_id}.")

    metrics = binary_metrics(labels, oof_scores, oof_predictions)
    oof_nuisance_dependence = (
        conditional_nuisance_dependence(
            oof_scores, low_rank_nuisance_scores(z), labels
        )
        if z is not None
        else None
    )
    result = {
        "experiment": asdict(experiment),
        "seed": int(seed),
        "metrics": metrics,
        "mean_counterfactual_score_change": (
            float(np.nanmean(counterfactual_change))
            if np.any(np.isfinite(counterfactual_change))
            else None
        ),
        "oof_conditional_nuisance_dependence": oof_nuisance_dependence,
        "patients": patients,
        "labels": labels,
        "scores": oof_scores,
        "predictions": oof_predictions,
        "thresholds": oof_thresholds,
        "fold_rows": fold_rows,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "patient_predictions.csv", "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "patient_id",
                "true_label",
                "score",
                "threshold",
                "predicted_label",
                "counterfactual_score_change",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, patient in enumerate(patients):
                writer.writerow(
                    {
                        "patient_id": patient,
                        "true_label": int(labels[index]),
                        "score": float(oof_scores[index]),
                        "threshold": float(oof_thresholds[index]),
                        "predicted_label": int(oof_predictions[index]),
                        "counterfactual_score_change": (
                            ""
                            if not np.isfinite(counterfactual_change[index])
                            else float(counterfactual_change[index])
                        ),
                    }
                )
        with open(save_dir / "fold_selection.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
            writer.writeheader()
            writer.writerows(fold_rows)
        (save_dir / "summary.json").write_text(
            json.dumps(
                {
                    "experiment": asdict(experiment),
                    "seed": seed,
                    "metrics": metrics,
                    "mean_counterfactual_score_change": result[
                        "mean_counterfactual_score_change"
                    ],
                    "oof_conditional_nuisance_dependence": result[
                        "oof_conditional_nuisance_dependence"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return result


# =============================================================================
# 12. UNCERTAINTY, STABILITY AND CONFOUNDING REPORTS
# =============================================================================


def bootstrap_auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(CFG.bootstrap_replicates):
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(float(roc_auc_score(sampled_labels, scores[index])))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_bootstrap_auc_difference(
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    control_scores: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(CFG.bootstrap_replicates):
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        differences.append(
            float(
                roc_auc_score(sampled_labels, candidate_scores[index])
                - roc_auc_score(sampled_labels, control_scores[index])
            )
        )
    array = np.asarray(differences)
    return {
        "median_delta_auc": float(np.median(array)),
        "ci_lower": float(np.quantile(array, 0.025)),
        "ci_upper": float(np.quantile(array, 0.975)),
        "probability_delta_positive": float(np.mean(array > 0)),
    }


def run_stability(
    experiment: Experiment,
    tables: dict[str, Any],
    groups: dict[str, str],
    logger: logging.Logger,
) -> dict[str, Any]:
    aucs: list[float] = []
    for repeat in range(CFG.stability_repeats):
        seed = CFG.seed + 20_000 + repeat
        result = evaluate_once(experiment, tables, groups, seed)
        aucs.append(result["metrics"]["auc"])
    array = np.asarray(aucs)
    summary = {
        "repeats": CFG.stability_repeats,
        "auc_mean": float(array.mean()),
        "auc_median": float(np.median(array)),
        "auc_q25": float(np.quantile(array, 0.25)),
        "auc_q75": float(np.quantile(array, 0.75)),
        "auc_min": float(array.min()),
        "auc_max": float(array.max()),
        "all_aucs": aucs,
    }
    logger.info(
        "%s stability: median AUC %.4f, IQR [%.4f, %.4f].",
        experiment.experiment_id,
        summary["auc_median"],
        summary["auc_q25"],
        summary["auc_q75"],
    )
    return summary


def write_research_report(
    primary_results: dict[str, dict[str, Any]],
    stability: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize predictive performance and confounding burden.

    The single-manifest OOF scores support paired patient bootstrap intervals.
    The headline ranking and shortcut gaps use same-seed repeated-CV medians,
    because one favorable split is too fragile for a 30-patient cohort.
    """

    labels = next(iter(primary_results.values()))["labels"]
    candidate = primary_results[PRIMARY_EXPERIMENT]
    candidate_auc = candidate["metrics"]["auc"]
    candidate_median_auc = stability[PRIMARY_EXPERIMENT]["auc_median"]
    control_ids = [
        "C1_EXACT_SUPPORT_ONLY",
        "C2_SHUFFLED_HEART_INTENSITY",
        "C3_EXACT_SUPPORT_COMPLEMENT",
        "C4_EXPORT_METADATA_ONLY",
        "C5_COMBINED_NUISANCE_ONLY",
    ]
    control_single_aucs = {
        experiment_id: primary_results[experiment_id]["metrics"]["auc"]
        for experiment_id in control_ids
    }
    control_median_aucs = {
        experiment_id: stability[experiment_id]["auc_median"]
        for experiment_id in control_ids
    }
    strongest_control = max(control_median_aucs, key=control_median_aucs.get)
    strict_nuisance_ids = [
        "C3_EXACT_SUPPORT_COMPLEMENT",
        "C4_EXPORT_METADATA_ONLY",
    ]
    strongest_strict = max(
        strict_nuisance_ids,
        key=lambda experiment_id: control_median_aucs[experiment_id],
    )

    paired_bootstrap = {}
    repeated_paired = {}
    candidate_repeated = np.asarray(
        stability[PRIMARY_EXPERIMENT]["all_aucs"], dtype=np.float64
    )
    for control_index, control_id in enumerate(control_ids):
        paired_bootstrap[control_id] = paired_bootstrap_auc_difference(
            labels,
            candidate["scores"],
            primary_results[control_id]["scores"],
            CFG.seed + 9000 + control_index,
        )
        control_repeated = np.asarray(
            stability[control_id]["all_aucs"], dtype=np.float64
        )
        delta = candidate_repeated - control_repeated
        repeated_paired[control_id] = {
            "median_delta_auc": float(np.median(delta)),
            "delta_q25": float(np.quantile(delta, 0.25)),
            "delta_q75": float(np.quantile(delta, 0.75)),
            "candidate_better_fraction": float(np.mean(delta > 0)),
        }

    # Project-specific descriptive indices. They are not standard clinical metrics.
    representation_gap = candidate_median_auc - max(
        control_median_aucs["C1_EXACT_SUPPORT_ONLY"],
        control_median_aucs["C2_SHUFFLED_HEART_INTENSITY"],
    )
    strict_confounding_gap = (
        candidate_median_auc - control_median_aucs[strongest_strict]
    )
    shortcut_fraction = max(
        0.0, control_median_aucs[strongest_strict] - 0.5
    ) / max(candidate_median_auc - 0.5, 1e-8)

    report = {
        "research_question": (
            "Does heart-localized MRI signal predict the released label beyond "
            "mask geometry, shuffled cardiac intensity, outside-heart pixels and "
            "export metadata?"
        ),
        "primary_experiment": PRIMARY_EXPERIMENT,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "patient_count": int(len(labels)),
        "protocol_balanced": bool(
            next(iter(primary_results.values())).get("protocol_balanced", False)
        ),
        "single_manifest": {
            "primary_auc": float(candidate_auc),
            "primary_auc_bootstrap_ci": bootstrap_auc_interval(
                labels, candidate["scores"], CFG.seed + 8000
            ),
            "reference_auc": float(
                primary_results[REFERENCE_EXPERIMENT]["metrics"]["auc"]
            ),
            "control_aucs": control_single_aucs,
            "paired_primary_minus_control_bootstrap": paired_bootstrap,
        },
        "repeated_nested_cv": {
            "primary_median_auc": float(candidate_median_auc),
            "reference_median_auc": float(
                stability[REFERENCE_EXPERIMENT]["auc_median"]
            ),
            "control_median_aucs": control_median_aucs,
            "strongest_control": strongest_control,
            "combined_nuisance_control_auc": control_median_aucs[
                "C5_COMBINED_NUISANCE_ONLY"
            ],
            "paired_primary_minus_control": repeated_paired,
        },
        "custom_descriptive_indices_from_repeated_medians": {
            "representation_gap": float(representation_gap),
            "strict_confounding_gap": float(strict_confounding_gap),
            "strict_shortcut_fraction": float(shortcut_fraction),
            "definitions": {
                "representation_gap": (
                    "primary median AUC minus max(mask-only median AUC, "
                    "shuffled-intensity median AUC)"
                ),
                "strict_confounding_gap": (
                    "primary median AUC minus max(outside median AUC, "
                    "metadata-only median AUC)"
                ),
                "strict_shortcut_fraction": (
                    "(max strict-control median AUC - 0.5) / "
                    "(primary median AUC - 0.5); descriptive only, lower is better"
                ),
            },
        },
        "primary_results": {
            key: {
                "metrics": value["metrics"],
                "mean_counterfactual_score_change": value[
                    "mean_counterfactual_score_change"
                ],
                "oof_conditional_nuisance_dependence": value[
                    "oof_conditional_nuisance_dependence"
                ],
            }
            for key, value in primary_results.items()
        },
        "stability": stability,
        "interpretation": (
            "A high candidate AUC is insufficient if mask/outside/metadata controls "
            "remain high. The most persuasive result combines a stable cardiac AUC, "
            "positive same-seed paired gaps, low within-class nuisance dependence, "
            "reviewed protocol balancing and external validation."
        ),
    }
    (output_dir / "research_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


# =============================================================================
# 13. VALIDATION AND MAIN
# =============================================================================


def validate_configuration() -> None:
    ids = [experiment.experiment_id for experiment in EXPERIMENTS]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique.")
    if PRIMARY_EXPERIMENT not in ids or REFERENCE_EXPERIMENT not in ids:
        raise ValueError("Primary/reference experiment is missing.")
    if CFG.monai_dilation_kernel % 2 == 0 or CFG.a17_extra_dilation_kernel % 2 == 0:
        raise ValueError("Dilation kernels must be odd.")
    if not 0 < CFG.fixed_content_long_side <= CFG.monai_size:
        raise ValueError("Invalid fixed-content geometry.")
    if CFG.outer_folds < 2 or CFG.inner_folds < 2:
        raise ValueError("Nested CV requires at least two folds.")
    if not CFG.c_grid or not CFG.ridge_grid:
        raise ValueError("Hyperparameter grids cannot be empty.")
    if CFG.use_amp:
        # AMP can be enabled intentionally, but the default admissions/research
        # version is deterministic float32.
        logging.getLogger("cad_mri_mit").warning(
            "AMP is enabled; exact numerical repeatability may decrease."
        )


def save_configuration(output_dir: Path) -> None:
    payload = {
        "config": asdict(CFG),
        "device": str(DEVICE),
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "primary_experiment": PRIMARY_EXPERIMENT,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "mathematical_components": {
            "RSC": "Robust Series Consensus",
            "PRC": "Paired Regional Contrast",
            "FLCO": "Fold-Local Confounder Orthogonalization",
            "APPS": "AUC-Preserving Pareto Selection",
        },
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    validate_configuration()
    set_reproducibility(CFG.seed)
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(CFG.output_dir)
    save_configuration(CFG.output_dir)

    logger.info("CAD MRI MIT clear pipeline started on %s.", DEVICE)
    logger.info("Dataset path: %s", CFG.dataset_path)
    logger.info("Experiments: %s", [e.experiment_id for e in EXPERIMENTS])
    started = time.perf_counter()

    records = discover_records(CFG.dataset_path, logger)
    write_series_annotation_template(
        records, CFG.output_dir / "series_annotation_template.csv"
    )
    series_to_cell = load_series_annotations(CFG.series_annotation_csv, logger)
    records_for_run = (
        [record for record in records if record.series_id in series_to_cell]
        if series_to_cell
        else records
    )
    if series_to_cell:
        logger.info(
            "Human-reviewed filtering retained %d/%d images before neural inference.",
            len(records_for_run),
            len(records),
        )
    bank, cache_dir = get_or_build_feature_bank(records_for_run, logger)
    patient_groups = audit_exact_duplicates(
        bank, CFG.output_dir / "audits", logger
    )
    tables = build_patient_representations(bank, series_to_cell, logger)

    main_results: dict[str, dict[str, Any]] = {}
    stability: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        logger.info("Evaluating %s: %s", experiment.experiment_id, experiment.description)
        result = evaluate_once(
            experiment,
            tables,
            patient_groups,
            CFG.seed,
            CFG.output_dir / "experiments" / experiment.experiment_id,
        )
        result["protocol_balanced"] = bool(series_to_cell)
        main_results[experiment.experiment_id] = result
        logger.info(
            "%s: AUC=%.4f, AUPRC=%.4f, sensitivity=%.3f, specificity=%.3f.",
            experiment.experiment_id,
            result["metrics"]["auc"],
            result["metrics"]["auprc"],
            result["metrics"]["sensitivity"],
            result["metrics"]["specificity"],
        )

    # Stability is most informative for the reference, custom methods and controls.
    for experiment in EXPERIMENTS:
        stability[experiment.experiment_id] = run_stability(
            experiment, tables, patient_groups, logger
        )

    report = write_research_report(main_results, stability, CFG.output_dir)

    summary_rows = []
    for experiment in EXPERIMENTS:
        result = main_results[experiment.experiment_id]
        stable = stability[experiment.experiment_id]
        summary_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "role": experiment.role,
                "auc": result["metrics"]["auc"],
                "auprc": result["metrics"]["auprc"],
                "sensitivity": result["metrics"]["sensitivity"],
                "specificity": result["metrics"]["specificity"],
                "f1": result["metrics"]["f1"],
                "median_repeated_auc": stable["auc_median"],
                "repeated_auc_q25": stable["auc_q25"],
                "repeated_auc_q75": stable["auc_q75"],
                "mean_counterfactual_score_change": result[
                    "mean_counterfactual_score_change"
                ],
                "oof_conditional_nuisance_dependence": result[
                    "oof_conditional_nuisance_dependence"
                ],
            }
        )
    summary_rows.sort(key=lambda row: -row["median_repeated_auc"])
    with open(
        CFG.output_dir / "experiment_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Final ranking by median repeated AUC:")
    for rank, row in enumerate(summary_rows, start=1):
        logger.info(
            "%2d. %-35s median=%.4f IQR=[%.4f, %.4f]",
            rank,
            row["experiment_id"],
            row["median_repeated_auc"],
            row["repeated_auc_q25"],
            row["repeated_auc_q75"],
        )

    logger.info(
        "Primary=%s median repeated AUC=%.4f; representation gap=%+.4f; "
        "strict confounding gap=%+.4f.",
        PRIMARY_EXPERIMENT,
        report["repeated_nested_cv"]["primary_median_auc"],
        report["custom_descriptive_indices_from_repeated_medians"][
            "representation_gap"
        ],
        report["custom_descriptive_indices_from_repeated_medians"][
            "strict_confounding_gap"
        ],
    )
    logger.info("Feature cache: %s", cache_dir)
    logger.info("Completed in %.1f minutes.", (time.perf_counter() - started) / 60.0)


if __name__ == "__main__":
    main()
