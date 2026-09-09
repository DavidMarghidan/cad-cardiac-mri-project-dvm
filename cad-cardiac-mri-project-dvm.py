#%% ============================================================================
# CAD CARDIAC MRI — FOLD-LOCAL SHARED-SERIES HARMONIZATION V7.2
# ============================================================================
#
# PURPOSE
# -------
# This script is a focused extension of the V7.1 patient-level research suite.
# It keeps only folder-defined series proxies whose acquisition/export profile is
# represented in BOTH Normal and Sick training patients, and removes series that
# are atypical relative to those shared families.
#
# It deliberately reuses the V7.1 frozen feature bank. MONAI and EfficientNet do
# not run again. The expensive neural-network stage therefore remains identical
# to the validated V7.1 execution, while the new scientific question is isolated:
#
#     Does the A17 result remain strong after restricting every fold to
#     comparable, non-atypical series families shared by Normal and Sick?
#
# CRITICAL LEAKAGE RULE
# ---------------------
# “Shared by Normal and Sick” is label-aware. It would be invalid to inspect all
# 30 labels once, choose a global subset of series, and then claim an unbiased
# cross-validated AUC on that same subset. The held-out patients would have
# influenced cohort construction.
#
# This implementation therefore fits the series selector separately inside each
# training partition:
#
#     outer-training series only
#         -> robust nuisance-feature scaling
#         -> label-blind clustering
#         -> identify clusters supported by BOTH training classes
#         -> estimate cluster-specific typicality radii
#
# The resulting frozen selector is then applied to outer-validation series using
# only their label-free descriptors. Validation labels are never supplied to the
# selector. The same nesting is repeated inside each inner split before choosing
# Logistic-Regression C or the operating threshold.
#
# WHAT “SIMILAR SERIES” MEANS HERE
# --------------------------------
# Similarity is intentionally calculated from protocol/export/QC proxies rather
# than from A17 image embeddings. Using the candidate embedding itself to decide
# which series to retain could suppress true disease variation or select series
# that already separate the labels.
#
# The default descriptor contains:
#
#   - log number of slices in the series proxy;
#   - native height, width and aspect ratio;
#   - file-size and bytes-per-pixel proxies;
#   - dark/white-border and padding proportions;
#   - label-blind crop and fixed-canvas geometry;
#   - standardized MONAI plausible-mask rate.
#
# Raw center intensity, entropy, sharpness, A17 embeddings, CAD scores and class
# labels are NOT descriptor coordinates. Training labels are used only after
# clustering to determine whether a cluster has adequate support in both classes.
#
# TARGET POPULATION AND COVERAGE
# ------------------------------
# A patient can be outside the common support if none of their series is retained.
# This script does not silently reinsert the “least atypical” series, because that
# would violate the requested filter. It instead reports common-support coverage
# and evaluates the conditional target population formed by patients with at least
# one retained series. Coverage is reported overall and by class after scoring.
#
# MAIN OUTPUTS
# ------------
#   configuration.json
#   source_cache.json
#   global_audit/series_descriptors.csv
#   global_audit/shared_cluster_summary.csv
#   global_audit/series_selection.csv
#   global_audit/global_selection_summary.json
#   primary_cv/<experiment_id>/patient_oof_predictions.csv
#   primary_cv/<experiment_id>/outer_fold_metrics.csv
#   primary_cv/<experiment_id>/summary.json
#   primary_cv/fold_*/series_selection.csv
#   primary_cv/fold_*/shared_cluster_summary.csv
#   comparison/experiment_summary.csv
#   comparison/paired_auc_differences.json
#   repeated_cv/repeated_runs.csv
#   repeated_cv/summary.json
#   permutation/summary.json                      (when enabled)
#   final_report.json
#   console_output.log
#
# RUNNING IN KAGGLE
# -----------------
# 1. Run V7.1 once so its feature-bank cache exists.
# 2. Upload this file and execute:
#
#       %run /kaggle/working/cad_mri_series_overlap_harmonized_v7_2.py
#
# The script searches compatible caches under /kaggle/working and /kaggle/input.
# An explicit path can be supplied with:
#
#       os.environ["CAD_V71_FEATURE_CACHE_DIR"] = "/path/to/cache/fingerprint"
#
# This is an exploratory internal-validation extension. It is not external
# validation and does not establish clinical diagnostic performance.
# ============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
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


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "7.2"
SCRIPT_NAME = "CAD MRI fold-local shared-series harmonization"
RANDOM_SEED = 42

# The current V7.1 feature bank contains these nine image representations. The
# extension reuses all of them so every image candidate/control can be evaluated
# on exactly the same fold-local selected series rows.
FEATURE_MODE_BY_EXPERIMENT = {
    "A17_ALL_SERIES_REFERENCE": "standardized_hard_support_region_norm",
    "A17_SHARED_TYPICAL_SERIES": "standardized_hard_support_region_norm",
    "A12_SHARED_TYPICAL_SERIES": (
        "standardized_roi_zero_bg_center_fallback"
    ),
    "A16_SHARED_TYPICAL_SERIES": (
        "standardized_heart_centered_fixed_fov_region_norm"
    ),
    "A9_SHARED_TYPICAL_SERIES": "standardized_full_image",
    "C31_SHARED_TYPICAL_SUPPORT_ONLY": (
        "standardized_a17_exact_support_mask_only"
    ),
    "C32_SHARED_TYPICAL_SHUFFLED_INTENSITY": (
        "standardized_a17_support_intensity_affine_shuffled"
    ),
    "C33_SHARED_TYPICAL_EXACT_COMPLEMENT": (
        "standardized_a17_exact_support_complement_region_norm"
    ),
    "C28_SHARED_TYPICAL_CONSERVATIVE_OUTSIDE": (
        "standardized_outside_whole_heart_region_norm"
    ),
    "C29_SHARED_TYPICAL_FIXED_PERIPHERY": (
        "standardized_fixed_periphery_region_norm"
    ),
}

# Experiments in this tuple receive a fold-local shared-series selector. The
# all-series A17 reference is intentionally the only unfiltered image experiment.
FILTERED_EXPERIMENT_IDS = tuple(
    experiment_id
    for experiment_id in FEATURE_MODE_BY_EXPERIMENT
    if experiment_id != "A17_ALL_SERIES_REFERENCE"
)

# This control does not use an image embedding. It asks whether the selector's
# retained-cluster composition, retention fraction and typicality distances alone
# can classify Normal versus Sick. A high AUC means protocol composition remains
# confounded even after excluding unsupported and atypical series.
SERIES_STRUCTURE_CONTROL_ID = "C34_SHARED_SERIES_STRUCTURE_ONLY"
ALL_EXPERIMENT_IDS = tuple(FEATURE_MODE_BY_EXPERIMENT) + (
    SERIES_STRUCTURE_CONTROL_ID,
)
PRIMARY_EXPERIMENT_ID = "A17_SHARED_TYPICAL_SERIES"

# -----------------------------------------------------------------------------
# Series descriptor and common-support selector
# -----------------------------------------------------------------------------

SERIES_CLUSTER_COUNT = 12
SERIES_CLUSTER_BATCH_SIZE = 256
SERIES_CLUSTER_N_INIT = 10
SERIES_CLUSTER_MAX_ITER = 300

MIN_SLICES_PER_SERIES = 2
MIN_STANDARDIZED_MONAI_VALID_RATE = 0.10
# These two label-blind rules remove extremely fragmented proxies and series for
# which the short-axis ventricular segmenter almost never finds plausible heart
# anatomy. They are fixed before evaluation and are never tuned against AUC.

MIN_SERIES_PER_CLASS_IN_CLUSTER = 8
MIN_PATIENTS_PER_CLASS_IN_CLUSTER = 2
MIN_CLASS_SERIES_BALANCE_RATIO = 0.20
MIN_CLASS_PATIENT_BALANCE_RATIO = 0.40
MAX_SINGLE_PATIENT_SERIES_FRACTION_IN_SHARED_CLUSTER = 0.50
# A shared cluster must contain more than a few duplicated folders and must be
# represented by at least two distinct patients from each training class. It is
# also required to have a non-negligible minority/majority balance in both series
# count and distinct-patient count. Thus a cluster with, for example, hundreds of
# Normal exports but only a handful of Sick exports is not called “shared” merely
# because both labels occur. Finally, the cluster is rejected when one patient's
# fragmented export supplies more than half of all series. These rules prevent
# SR_* over-fragmentation from defining a nominal “series family” by itself.

CLUSTER_TYPICALITY_RADIUS_QUANTILE = 0.95
MAX_CLASS_CENTROID_GAP_RATIO = 1.00
# The cluster radius is the 95th percentile of training-series distances to the
# cluster centroid. A cluster is retained only when the distance between its
# Normal and Sick patient-centroid means is no larger than one pooled radius.
# This removes clusters whose nominal overlap hides strongly displaced classes.

MIN_TRAIN_PATIENT_COVERAGE_PER_CLASS = 0.70
# If fewer than 70% of training patients in either class retain any series, the
# selector is considered too restrictive and the fold fails rather than silently
# relaxing thresholds after viewing validation performance.

CONSENSUS_TYPICAL_MIN_OUTER_FOLD_FRACTION = 0.80
# This threshold is used only for a post-evaluation audit file. A series proxy is
# described as "consistently typical/shared" when at least 80% of the five outer
# selectors retain it. The consensus list is NEVER fed back into CV; doing so
# would let held-out folds influence the evaluated cohort.

# Descriptor names are selected explicitly from metadata. Features associated
# with central anatomy, entropy, edge density or sharpness are excluded.
PROVENANCE_DESCRIPTOR_NAMES = (
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
STANDARDIZATION_DESCRIPTOR_NAMES = (
    "crop_top_fraction",
    "crop_bottom_fraction",
    "crop_left_fraction",
    "crop_right_fraction",
    "retained_height_fraction",
    "retained_width_fraction",
    "detected_padding_fraction",
    "fixed_content_height_fraction_of_canvas",
    "fixed_content_width_fraction_of_canvas",
    "fixed_pipeline_padding_fraction",
)

# -----------------------------------------------------------------------------
# Patient classifier and evaluation
# -----------------------------------------------------------------------------

N_SPLITS = 5
INNER_SPLITS = 3
C_GRID = (0.01, 0.1, 1.0, 10.0)
C_SELECTION_AUC_TOLERANCE = 0.01
PCA_EXPLAINED_VARIANCE = 0.95
LOGISTIC_MAX_ITER = 4000
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95

RUN_REPEATED_CV = True
REPEATED_CV_REPEATS = 50
REPEATED_CV_SEED_START = RANDOM_SEED + 20_000
REPEATED_CV_EXPERIMENT_IDS = (
    "A17_ALL_SERIES_REFERENCE",
    "A17_SHARED_TYPICAL_SERIES",
    "A12_SHARED_TYPICAL_SERIES",
    "A16_SHARED_TYPICAL_SERIES",
    "A9_SHARED_TYPICAL_SERIES",
    "C31_SHARED_TYPICAL_SUPPORT_ONLY",
    "C32_SHARED_TYPICAL_SHUFFLED_INTENSITY",
    "C33_SHARED_TYPICAL_EXACT_COMPLEMENT",
    "C28_SHARED_TYPICAL_CONSERVATIVE_OUTSIDE",
    "C29_SHARED_TYPICAL_FIXED_PERIPHERY",
    SERIES_STRUCTURE_CONTROL_ID,
)

RUN_PERMUTATION_TEST = False
PERMUTATION_REPLICATES = 200
PERMUTATION_RANDOM_STATE = RANDOM_SEED + 60_000
# Disabled by default because the fully nested selector must be refitted after
# every label permutation. Enable for a final confirmatory run after inspecting
# common-support coverage. With 200 replicates, the minimum empirical p-value is
# 1/(200+1) ≈ 0.00498.

# Cache discovery and output paths.
EXPLICIT_FEATURE_CACHE_DIR = os.environ.get("CAD_V71_FEATURE_CACHE_DIR", "")
EXPLICIT_V71_RUN_DIR = os.environ.get("CAD_V71_RUN_DIR", "")
OUTPUT_ROOT = Path(
    os.environ.get(
        "CAD_SERIES_OVERLAP_OUTPUT_ROOT",
        "/kaggle/working/cad_patient_pipeline_outputs"
        if Path("/kaggle/working").exists()
        else str(Path.cwd() / "cad_patient_pipeline_outputs"),
    )
)

REQUIRED_FEATURE_MODES = tuple(sorted(set(FEATURE_MODE_BY_EXPERIMENT.values())))
FEATURE_DIMENSION = 1280


# =============================================================================
# SMALL DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    feature_mode: str | None
    use_series_filter: bool
    use_pca: bool = True
    role: str = "candidate"


EXPERIMENT_SPECS = tuple(
    ExperimentSpec(
        experiment_id=experiment_id,
        feature_mode=feature_mode,
        use_series_filter=(experiment_id in FILTERED_EXPERIMENT_IDS),
        use_pca=True,
        role=(
            "reference"
            if experiment_id == "A17_ALL_SERIES_REFERENCE"
            else "candidate"
            if not experiment_id.startswith("C")
            else "control"
        ),
    )
    for experiment_id, feature_mode in FEATURE_MODE_BY_EXPERIMENT.items()
) + (
    ExperimentSpec(
        experiment_id=SERIES_STRUCTURE_CONTROL_ID,
        feature_mode=None,
        use_series_filter=True,
        use_pca=False,
        role="negative_control",
    ),
)


@dataclass
class SeriesTable:
    series_ids: np.ndarray
    patient_ids: np.ndarray
    labels: np.ndarray
    slice_counts: np.ndarray
    standardized_monai_valid_rates: np.ndarray
    descriptor_names: tuple[str, ...]
    descriptors: np.ndarray
    slice_indices: list[np.ndarray]
    series_index_by_id: dict[str, int]
    patient_label_by_id: dict[str, int]


@dataclass
class SelectorModel:
    descriptor_median: np.ndarray
    descriptor_scale: np.ndarray
    kmeans: MiniBatchKMeans
    accepted_clusters: np.ndarray
    cluster_radii: np.ndarray
    cluster_reasons: tuple[str, ...]
    cluster_summary_rows: list[dict]
    training_patient_ids: tuple[str, ...]
    random_state: int


@dataclass
class SelectionResult:
    series_indices: np.ndarray
    cluster_ids: np.ndarray
    distances: np.ndarray
    keep_mask: np.ndarray
    reasons: np.ndarray
    patient_ids: np.ndarray
    labels: np.ndarray


# =============================================================================
# LOGGING AND FILE HELPERS
# =============================================================================

class TeeStream:
    """Duplicate text output to the live stream and a persistent log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None):
    """Write a deterministic UTF-8 CSV, including an empty header when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = tuple(rows[0].keys()) if rows else ("status",)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def elapsed_text(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {seconds:.1f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {seconds:.1f} s"


# =============================================================================
# CACHE DISCOVERY AND LOADING
# =============================================================================

def _candidate_cache_roots() -> list[Path]:
    roots = []
    if EXPLICIT_FEATURE_CACHE_DIR:
        roots.append(Path(EXPLICIT_FEATURE_CACHE_DIR))
    roots.extend(
        [
            Path("/kaggle/working/cad_patient_pipeline_outputs/feature_bank_cache"),
            Path("/kaggle/input"),
            Path.cwd(),
        ]
    )
    unique = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if str(resolved) not in seen:
            seen.add(str(resolved))
            unique.append(root)
    return unique


def _is_compatible_cache(cache_dir: Path) -> tuple[bool, dict | None, str]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        return False, None, "metadata.json missing"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as error:
        return False, None, f"metadata unreadable: {error}"

    completed_modes = set(metadata.get("completed_modes", []))
    missing_modes = sorted(set(REQUIRED_FEATURE_MODES) - completed_modes)
    if missing_modes:
        return False, metadata, f"missing modes: {missing_modes}"

    required_arrays = (
        "labels.npy",
        "patient_ids.npy",
        "series_ids.npy",
        "sample_indices.npy",
        "provenance_features.npy",
        "standardization_features.npy",
        "standardized_monai_valid.npy",
        "standardized_area_ratios.npy",
        "standardized_peak_probabilities.npy",
        "standardized_mean_foreground_probabilities.npy",
    )
    missing = [name for name in required_arrays if not (cache_dir / name).is_file()]
    missing.extend(
        f"features__{mode}.npy"
        for mode in REQUIRED_FEATURE_MODES
        if not (cache_dir / f"features__{mode}.npy").is_file()
    )
    if missing:
        return False, metadata, f"missing files: {missing[:8]}"

    if int(metadata.get("feature_dimension", FEATURE_DIMENSION)) != FEATURE_DIMENSION:
        return False, metadata, "unexpected feature dimension"
    return True, metadata, "compatible"


def discover_feature_cache() -> tuple[Path, dict]:
    """Find the newest complete V7.1-compatible cache without guessing a slug."""

    candidates = []
    for root in _candidate_cache_roots():
        if not root.exists():
            continue
        if (root / "metadata.json").is_file():
            dirs = [root]
        else:
            # Kaggle input datasets can add one or more wrapper directories.
            dirs = [path.parent for path in root.rglob("metadata.json")]
        for cache_dir in dirs:
            compatible, metadata, reason = _is_compatible_cache(cache_dir)
            if compatible:
                try:
                    modified = (cache_dir / "metadata.json").stat().st_mtime_ns
                except OSError:
                    modified = 0
                candidates.append((modified, cache_dir, metadata))
            elif EXPLICIT_FEATURE_CACHE_DIR and cache_dir.resolve() == Path(
                EXPLICIT_FEATURE_CACHE_DIR
            ).resolve():
                raise RuntimeError(
                    f"Explicit feature cache is incompatible: {cache_dir}: {reason}"
                )

    if not candidates:
        raise FileNotFoundError(
            "No compatible V7.1 feature bank was found. Run the focused V7.1 "
            "pipeline first, attach its output as a Kaggle input, or set "
            "CAD_V71_FEATURE_CACHE_DIR to the exact fingerprint directory."
        )
    candidates.sort(key=lambda row: (row[0], str(row[1])), reverse=True)
    _, cache_dir, metadata = candidates[0]
    print(f"[CACHE] Selected compatible feature bank: {cache_dir}", flush=True)
    return cache_dir, metadata


def load_feature_bank(cache_dir: Path, metadata: dict) -> dict:
    """Memory-map only arrays required by the harmonization extension."""

    names = (
        "labels",
        "patient_ids",
        "series_ids",
        "sample_indices",
        "provenance_features",
        "standardization_features",
        "standardized_monai_valid",
        "standardized_area_ratios",
        "standardized_peak_probabilities",
        "standardized_mean_foreground_probabilities",
    )
    bank = {
        name: np.load(cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in names
    }
    bank["features"] = {
        mode: np.load(
            cache_dir / f"features__{mode}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for mode in REQUIRED_FEATURE_MODES
    }
    bank["metadata"] = metadata
    bank["cache_dir"] = cache_dir

    n_slices = len(bank["labels"])
    for name in names:
        if len(bank[name]) != n_slices:
            raise RuntimeError(
                f"Cache row mismatch: {name} has {len(bank[name])}, expected {n_slices}."
            )
    for mode, values in bank["features"].items():
        if values.shape != (n_slices, FEATURE_DIMENSION):
            raise RuntimeError(
                f"Feature mode {mode!r} has shape {values.shape}, expected "
                f"({n_slices}, {FEATURE_DIMENSION})."
            )
    print(
        f"[CACHE] Loaded {n_slices} slice rows and {len(REQUIRED_FEATURE_MODES)} "
        "frozen image representations.",
        flush=True,
    )
    return bank


def discover_v71_run_dir() -> Path | None:
    """Find a V7.1 run directory only to reuse its primary patient folds."""

    if EXPLICIT_V71_RUN_DIR:
        path = Path(EXPLICIT_V71_RUN_DIR)
        if not path.is_dir():
            raise FileNotFoundError(f"CAD_V71_RUN_DIR does not exist: {path}")
        return path

    roots = [Path("/kaggle/working/cad_patient_pipeline_outputs"), Path("/kaggle/input")]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for manifest in root.rglob("patient_fold_manifest.csv"):
            if "focused_research_suite" not in str(manifest.parent.parent):
                continue
            candidates.append((manifest.stat().st_mtime_ns, manifest.parent.parent))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# =============================================================================
# SERIES TABLE CONSTRUCTION
# =============================================================================

def _index_names(metadata: dict, key: str, fallback: Sequence[str]) -> dict[str, int]:
    names = tuple(metadata.get(key, fallback))
    return {str(name): index for index, name in enumerate(names)}


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def build_series_table(bank: dict) -> SeriesTable:
    """Aggregate immutable label-free slice metadata to one row per series.

    The folder path remains only a patient-scoped proxy, not a DICOM
    SeriesInstanceUID. Series descriptors are computed before any CV split, but
    they contain no class labels or fitted parameters. Robust scaling, clustering,
    class-support decisions and outlier radii are fitted inside each fold.
    """

    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"]).astype(str)
    series_ids = np.asarray(bank["series_ids"]).astype(str)
    provenance = np.asarray(bank["provenance_features"], dtype=np.float32)
    standardization = np.asarray(bank["standardization_features"], dtype=np.float32)
    valid = np.asarray(bank["standardized_monai_valid"], dtype=bool)

    metadata = bank["metadata"]
    provenance_index = _index_names(
        metadata,
        "provenance_feature_names",
        PROVENANCE_DESCRIPTOR_NAMES,
    )
    standardization_index = _index_names(
        metadata,
        "standardization_feature_names",
        STANDARDIZATION_DESCRIPTOR_NAMES,
    )

    missing_provenance = [
        name for name in PROVENANCE_DESCRIPTOR_NAMES if name not in provenance_index
    ]
    missing_standardization = [
        name
        for name in STANDARDIZATION_DESCRIPTOR_NAMES
        if name not in standardization_index
    ]
    if missing_provenance or missing_standardization:
        raise RuntimeError(
            "Feature-bank metadata does not expose the required label-free series "
            f"descriptor coordinates. Missing provenance={missing_provenance}; "
            f"missing standardization={missing_standardization}."
        )

    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, series_id in enumerate(series_ids):
        grouped[str(series_id)].append(int(row_index))

    ordered_series_ids = np.asarray(sorted(grouped))
    series_patient_ids = []
    series_labels = []
    slice_counts = []
    valid_rates = []
    descriptors = []
    slice_indices: list[np.ndarray] = []

    descriptor_names = (
        "log1p_n_slices",
        "standardized_monai_valid_rate",
        "log1p_median_native_height",
        "log1p_median_native_width",
        "median_aspect_ratio_width_over_height",
        "log1p_median_file_size_bytes",
        "log1p_median_bytes_per_native_pixel",
        "median_near_black_fraction",
        "median_near_white_fraction",
        "median_border_mean_intensity_0_1",
        "median_border_std_intensity_0_1",
        "median_border_near_black_fraction",
        *(
            f"median_{name}"
            for name in STANDARDIZATION_DESCRIPTOR_NAMES
        ),
    )

    for series_id in ordered_series_ids:
        indices = np.asarray(grouped[str(series_id)], dtype=np.int64)
        unique_patients = np.unique(patient_ids[indices])
        unique_labels = np.unique(labels[indices])
        if len(unique_patients) != 1 or len(unique_labels) != 1:
            raise RuntimeError(
                f"Series proxy {series_id!r} crosses patients or labels: "
                f"patients={unique_patients.tolist()}, labels={unique_labels.tolist()}."
            )

        p = provenance[indices]
        s = standardization[indices]
        p_get = lambda name: p[:, provenance_index[name]]
        s_get = lambda name: s[:, standardization_index[name]]

        row = [
            math.log1p(len(indices)),
            float(np.mean(valid[indices])),
            math.log1p(max(_median(p_get("native_height")), 0.0)),
            math.log1p(max(_median(p_get("native_width")), 0.0)),
            _median(p_get("aspect_ratio_width_over_height")),
            math.log1p(max(_median(p_get("file_size_bytes")), 0.0)),
            math.log1p(max(_median(p_get("bytes_per_native_pixel")), 0.0)),
            _median(p_get("near_black_fraction")),
            _median(p_get("near_white_fraction")),
            _median(p_get("border_mean_intensity")) / 255.0,
            _median(p_get("border_std_intensity")) / 255.0,
            _median(p_get("border_near_black_fraction")),
        ]
        row.extend(_median(s_get(name)) for name in STANDARDIZATION_DESCRIPTOR_NAMES)

        series_patient_ids.append(str(unique_patients[0]))
        series_labels.append(int(unique_labels[0]))
        slice_counts.append(int(len(indices)))
        valid_rates.append(float(np.mean(valid[indices])))
        descriptors.append(row)
        slice_indices.append(indices)

    descriptor_matrix = np.asarray(descriptors, dtype=np.float32)
    if descriptor_matrix.shape != (len(ordered_series_ids), len(descriptor_names)):
        raise RuntimeError(
            f"Series descriptor shape mismatch: {descriptor_matrix.shape}."
        )
    if not np.all(np.isfinite(descriptor_matrix)):
        bad = np.argwhere(~np.isfinite(descriptor_matrix))[:20].tolist()
        raise RuntimeError(
            f"Series descriptors contain non-finite values at {bad}."
        )

    patient_label_by_id = {}
    for patient_id, label in zip(patient_ids, labels):
        previous = patient_label_by_id.get(str(patient_id))
        if previous is not None and previous != int(label):
            raise RuntimeError(f"Patient {patient_id} has inconsistent labels.")
        patient_label_by_id[str(patient_id)] = int(label)

    table = SeriesTable(
        series_ids=ordered_series_ids,
        patient_ids=np.asarray(series_patient_ids),
        labels=np.asarray(series_labels, dtype=np.int64),
        slice_counts=np.asarray(slice_counts, dtype=np.int64),
        standardized_monai_valid_rates=np.asarray(valid_rates, dtype=np.float32),
        descriptor_names=tuple(descriptor_names),
        descriptors=descriptor_matrix,
        slice_indices=slice_indices,
        series_index_by_id={
            str(series_id): index for index, series_id in enumerate(ordered_series_ids)
        },
        patient_label_by_id=patient_label_by_id,
    )
    print(
        f"[SERIES] Built {len(table.series_ids)} series-proxy descriptors for "
        f"{len(table.patient_label_by_id)} patients.",
        flush=True,
    )
    return table


def write_series_descriptors(path: Path, table: SeriesTable):
    rows = []
    for index, series_id in enumerate(table.series_ids):
        row = {
            "series_id": str(series_id),
            "patient_id": str(table.patient_ids[index]),
            "label_for_audit_only": int(table.labels[index]),
            "class_name_for_audit_only": (
                "Sick" if int(table.labels[index]) == 1 else "Normal"
            ),
            "n_slices": int(table.slice_counts[index]),
        }
        row.update(
            {
                name: float(value)
                for name, value in zip(
                    table.descriptor_names,
                    table.descriptors[index],
                )
            }
        )
        rows.append(row)
    write_csv(path, rows)


# =============================================================================
# SERIES-LEVEL EMBEDDING CACHE
# =============================================================================

def _series_cache_fingerprint(bank: dict, table: SeriesTable) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "source_feature_fingerprint": bank["metadata"].get("fingerprint"),
        "series_ids_sha256": hashlib.sha256(
            "\n".join(map(str, table.series_ids)).encode("utf-8")
        ).hexdigest(),
        "required_modes": REQUIRED_FEATURE_MODES,
        "aggregation": "equal_slice_mean_inside_series_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_or_load_series_embeddings(
    bank: dict,
    table: SeriesTable,
    output_root: Path,
) -> tuple[dict[str, np.ndarray], Path]:
    """Compute one frozen embedding per series and mode, then cache it.

    Filtering happens after this label-free aggregation. Reusing series means is
    mathematically identical to the V7.1 first hierarchy level for equal slice
    weights and makes nested fold-local selection inexpensive.
    """

    fingerprint = _series_cache_fingerprint(bank, table)
    cache_dir = output_root / "series_embedding_cache" / fingerprint[:16]
    metadata_path = cache_dir / "metadata.json"
    expected_paths = {
        mode: cache_dir / f"series_features__{mode}.npy"
        for mode in REQUIRED_FEATURE_MODES
    }

    if metadata_path.is_file() and all(path.is_file() for path in expected_paths.values()):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        if metadata.get("fingerprint") == fingerprint:
            loaded = {
                mode: np.load(path, mmap_mode="r", allow_pickle=False)
                for mode, path in expected_paths.items()
            }
            if all(
                values.shape == (len(table.series_ids), FEATURE_DIMENSION)
                for values in loaded.values()
            ):
                print(f"[SERIES CACHE] Reused {cache_dir}", flush=True)
                return loaded, cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    series_features = {}
    started = time.perf_counter()
    for mode_index, mode in enumerate(REQUIRED_FEATURE_MODES, start=1):
        print(
            f"[SERIES CACHE] Aggregating mode {mode_index}/{len(REQUIRED_FEATURE_MODES)}: {mode}",
            flush=True,
        )
        slice_features = bank["features"][mode]
        output = np.empty(
            (len(table.series_ids), FEATURE_DIMENSION),
            dtype=np.float32,
        )
        for series_index, indices in enumerate(table.slice_indices):
            output[series_index] = np.asarray(
                np.mean(slice_features[indices], axis=0, dtype=np.float64),
                dtype=np.float32,
            )
        np.save(expected_paths[mode], output, allow_pickle=False)
        series_features[mode] = np.load(
            expected_paths[mode], mmap_mode="r", allow_pickle=False
        )

    metadata = {
        "fingerprint": fingerprint,
        "source_feature_fingerprint": bank["metadata"].get("fingerprint"),
        "n_series": int(len(table.series_ids)),
        "feature_dimension": FEATURE_DIMENSION,
        "modes": list(REQUIRED_FEATURE_MODES),
        "aggregation": "equal_slice_mean_inside_series_v1",
        "runtime_seconds": float(time.perf_counter() - started),
    }
    json_write(metadata_path, metadata)
    print(
        f"[SERIES CACHE] Built in {elapsed_text(time.perf_counter() - started)}: {cache_dir}",
        flush=True,
    )
    return series_features, cache_dir


# =============================================================================
# FOLD-LOCAL COMMON-SUPPORT SELECTOR
# =============================================================================

def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    scale = q75 - q25
    mad = 1.4826 * np.median(np.abs(values - median), axis=0)
    scale = np.where(scale > 1e-8, scale, mad)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return median.astype(np.float64), scale.astype(np.float64)


def _hard_eligible_mask(table: SeriesTable, indices: np.ndarray) -> np.ndarray:
    return (
        (table.slice_counts[indices] >= MIN_SLICES_PER_SERIES)
        & (
            table.standardized_monai_valid_rates[indices]
            >= MIN_STANDARDIZED_MONAI_VALID_RATE
        )
        & np.all(np.isfinite(table.descriptors[indices]), axis=1)
    )


def fit_series_selector(
    table: SeriesTable,
    training_patient_ids: Sequence[str],
    random_state: int,
) -> SelectorModel:
    """Fit shared cluster support and typicality on training patients only."""

    training_patient_ids = tuple(sorted(map(str, training_patient_ids)))
    train_mask = np.isin(table.patient_ids, np.asarray(training_patient_ids))
    train_indices = np.flatnonzero(train_mask)
    eligible = _hard_eligible_mask(table, train_indices)
    eligible_indices = train_indices[eligible]

    if len(eligible_indices) < 2 * SERIES_CLUSTER_COUNT:
        raise RuntimeError(
            "Too few eligible training series for shared-series clustering: "
            f"{len(eligible_indices)}."
        )
    eligible_labels = table.labels[eligible_indices]
    if set(np.unique(eligible_labels).tolist()) != {0, 1}:
        raise RuntimeError("Eligible training series must contain both classes.")

    median, scale = _robust_location_scale(table.descriptors[eligible_indices])
    z = (table.descriptors[eligible_indices] - median) / scale

    n_clusters = min(
        SERIES_CLUSTER_COUNT,
        max(2, len(eligible_indices) // 20),
    )
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(random_state),
        n_init=SERIES_CLUSTER_N_INIT,
        max_iter=SERIES_CLUSTER_MAX_ITER,
        batch_size=min(SERIES_CLUSTER_BATCH_SIZE, len(eligible_indices)),
        reassignment_ratio=0.0,
    )

    # Give every training patient equal total clustering weight. Without this
    # correction, one patient split into hundreds of SR_* folders could move a
    # centroid much more than another patient represented by a few long series.
    eligible_patients = table.patient_ids[eligible_indices]
    patient_series_counts = {
        str(patient_id): int(np.sum(eligible_patients == patient_id))
        for patient_id in np.unique(eligible_patients)
    }
    kmeans_weights = np.asarray(
        [1.0 / patient_series_counts[str(patient_id)] for patient_id in eligible_patients],
        dtype=np.float64,
    )
    kmeans_weights *= len(kmeans_weights) / max(kmeans_weights.sum(), 1e-12)
    try:
        kmeans.fit(z, sample_weight=kmeans_weights)
    except TypeError:
        # Compatibility fallback for older scikit-learn releases. The complete
        # run metadata records the library version, and the audit JSON reports
        # whether patient-balanced KMeans weights were accepted.
        kmeans.fit(z)
        patient_weighting_supported = False
    else:
        patient_weighting_supported = True
    cluster_ids = np.asarray(kmeans.labels_, dtype=np.int64)
    distances = np.linalg.norm(z - kmeans.cluster_centers_[cluster_ids], axis=1)

    accepted = np.zeros(n_clusters, dtype=bool)
    radii = np.full(n_clusters, np.nan, dtype=np.float64)
    reasons: list[str] = []
    summaries: list[dict] = []

    for cluster_id in range(n_clusters):
        local = np.flatnonzero(cluster_ids == cluster_id)
        local_indices = eligible_indices[local]
        local_labels = table.labels[local_indices]
        local_patients = table.patient_ids[local_indices]
        class_series = {
            label: int(np.sum(local_labels == label)) for label in (0, 1)
        }
        class_patients = {
            label: int(len(np.unique(local_patients[local_labels == label])))
            for label in (0, 1)
        }

        # Estimate the typicality radius with equal influence from each patient:
        # first summarize the upper distance tail within each patient, then take
        # the same quantile across patient summaries.
        patient_distance_limits = []
        patient_centroids = []
        patient_centroid_labels = []
        patient_series_counts_in_cluster = []
        for patient_id in np.unique(local_patients):
            patient_local = local[local_patients == patient_id]
            patient_distance_limits.append(
                float(
                    np.quantile(
                        distances[patient_local],
                        CLUSTER_TYPICALITY_RADIUS_QUANTILE,
                    )
                )
            )
            patient_centroids.append(np.mean(z[patient_local], axis=0))
            patient_centroid_labels.append(
                int(np.unique(local_labels[local_patients == patient_id])[0])
            )
            patient_series_counts_in_cluster.append(int(len(patient_local)))
        radius = float(
            np.quantile(
                patient_distance_limits,
                CLUSTER_TYPICALITY_RADIUS_QUANTILE,
            )
        )
        radius = max(radius, 1e-6)
        radii[cluster_id] = radius
        patient_centroids = np.asarray(patient_centroids, dtype=np.float64)
        patient_centroid_labels = np.asarray(patient_centroid_labels, dtype=np.int64)

        if all(np.any(patient_centroid_labels == label) for label in (0, 1)):
            normal_center = np.mean(
                patient_centroids[patient_centroid_labels == 0], axis=0
            )
            sick_center = np.mean(
                patient_centroids[patient_centroid_labels == 1], axis=0
            )
            centroid_gap = float(np.linalg.norm(normal_center - sick_center))
            gap_ratio = centroid_gap / radius
        else:
            centroid_gap = float("inf")
            gap_ratio = float("inf")

        support_ok = all(
            class_series[label] >= MIN_SERIES_PER_CLASS_IN_CLUSTER
            and class_patients[label] >= MIN_PATIENTS_PER_CLASS_IN_CLUSTER
            for label in (0, 1)
        )
        series_balance_ratio = float(
            min(class_series.values()) / max(max(class_series.values()), 1)
        )
        patient_balance_ratio = float(
            min(class_patients.values()) / max(max(class_patients.values()), 1)
        )
        class_balance_ok = bool(
            series_balance_ratio >= MIN_CLASS_SERIES_BALANCE_RATIO
            and patient_balance_ratio >= MIN_CLASS_PATIENT_BALANCE_RATIO
        )
        max_patient_fraction = float(
            max(patient_series_counts_in_cluster) / max(len(local), 1)
        )
        patient_dominance_ok = bool(
            max_patient_fraction
            <= MAX_SINGLE_PATIENT_SERIES_FRACTION_IN_SHARED_CLUSTER
        )
        gap_ok = bool(gap_ratio <= MAX_CLASS_CENTROID_GAP_RATIO)
        accepted[cluster_id] = bool(
            support_ok and class_balance_ok and patient_dominance_ok and gap_ok
        )
        if not support_ok:
            reason = "insufficient_support_in_one_or_both_classes"
        elif not class_balance_ok:
            reason = "class_support_too_imbalanced"
        elif not patient_dominance_ok:
            reason = "single_patient_dominates_cluster"
        elif not gap_ok:
            reason = "class_centroids_too_far_apart"
        else:
            reason = "accepted_shared_cluster"
        reasons.append(reason)

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "accepted": bool(accepted[cluster_id]),
                "reason": reason,
                "normal_series": class_series[0],
                "sick_series": class_series[1],
                "normal_patients": class_patients[0],
                "sick_patients": class_patients[1],
                "minority_to_majority_series_ratio": series_balance_ratio,
                "minority_to_majority_patient_ratio": patient_balance_ratio,
                "typicality_radius": radius,
                "class_centroid_distance": centroid_gap,
                "class_centroid_gap_over_radius": gap_ratio,
                "maximum_single_patient_series_fraction": max_patient_fraction,
                "patient_balanced_kmeans_weighting_supported": bool(
                    patient_weighting_supported
                ),
            }
        )

    if not np.any(accepted):
        raise RuntimeError(
            "No cluster satisfied the predeclared Normal/Sick common-support "
            "criteria. Do not choose thresholds from validation AUC; inspect the "
            "saved/global descriptor audit and revise the predeclared selector."
        )

    model = SelectorModel(
        descriptor_median=median,
        descriptor_scale=scale,
        kmeans=kmeans,
        accepted_clusters=accepted,
        cluster_radii=radii,
        cluster_reasons=tuple(reasons),
        cluster_summary_rows=summaries,
        training_patient_ids=training_patient_ids,
        random_state=int(random_state),
    )

    training_result = apply_series_selector(table, train_indices, model)
    coverage = patient_coverage_summary(training_result)
    for label in (0, 1):
        class_name = "Sick" if label == 1 else "Normal"
        rate = coverage[class_name]["patient_coverage_rate"]
        if rate < MIN_TRAIN_PATIENT_COVERAGE_PER_CLASS:
            raise RuntimeError(
                f"Fold-local shared-series selector covers only {rate:.1%} of "
                f"{class_name} training patients; minimum is "
                f"{MIN_TRAIN_PATIENT_COVERAGE_PER_CLASS:.1%}."
            )
    return model


def apply_series_selector(
    table: SeriesTable,
    series_indices: Sequence[int],
    model: SelectorModel,
) -> SelectionResult:
    """Apply a frozen training-only selector without using query labels."""

    series_indices = np.asarray(series_indices, dtype=np.int64)
    z = (
        table.descriptors[series_indices] - model.descriptor_median
    ) / model.descriptor_scale
    finite = np.all(np.isfinite(z), axis=1)
    hard_eligible = _hard_eligible_mask(table, series_indices) & finite

    cluster_ids = np.full(len(series_indices), -1, dtype=np.int64)
    distances = np.full(len(series_indices), np.inf, dtype=np.float64)
    if np.any(hard_eligible):
        local_z = z[hard_eligible]
        local_clusters = model.kmeans.predict(local_z)
        local_distances = np.linalg.norm(
            local_z - model.kmeans.cluster_centers_[local_clusters],
            axis=1,
        )
        cluster_ids[hard_eligible] = local_clusters
        distances[hard_eligible] = local_distances

    keep = np.zeros(len(series_indices), dtype=bool)
    reasons = np.empty(len(series_indices), dtype=object)
    for local_index, series_index in enumerate(series_indices):
        if table.slice_counts[series_index] < MIN_SLICES_PER_SERIES:
            reasons[local_index] = "too_few_slices"
            continue
        if (
            table.standardized_monai_valid_rates[series_index]
            < MIN_STANDARDIZED_MONAI_VALID_RATE
        ):
            reasons[local_index] = "low_standardized_monai_valid_rate"
            continue
        cluster_id = int(cluster_ids[local_index])
        if cluster_id < 0:
            reasons[local_index] = "nonfinite_descriptor"
            continue
        if not model.accepted_clusters[cluster_id]:
            reasons[local_index] = model.cluster_reasons[cluster_id]
            continue
        if distances[local_index] > model.cluster_radii[cluster_id]:
            reasons[local_index] = "distance_outlier_within_shared_cluster"
            continue
        keep[local_index] = True
        reasons[local_index] = "kept_shared_typical"

    return SelectionResult(
        series_indices=series_indices,
        cluster_ids=cluster_ids,
        distances=distances,
        keep_mask=keep,
        reasons=np.asarray(reasons, dtype=str),
        patient_ids=table.patient_ids[series_indices],
        labels=table.labels[series_indices],
    )


def patient_coverage_summary(result: SelectionResult) -> dict:
    summary = {}
    for label, class_name in ((0, "Normal"), (1, "Sick")):
        class_patients = np.unique(result.patient_ids[result.labels == label])
        kept_patients = np.unique(
            result.patient_ids[(result.labels == label) & result.keep_mask]
        )
        class_series = int(np.sum(result.labels == label))
        kept_series = int(np.sum((result.labels == label) & result.keep_mask))
        summary[class_name] = {
            "patients_total": int(len(class_patients)),
            "patients_with_retained_series": int(len(kept_patients)),
            "patient_coverage_rate": (
                float(len(kept_patients) / len(class_patients))
                if len(class_patients)
                else float("nan")
            ),
            "series_total": class_series,
            "series_retained": kept_series,
            "series_retention_rate": (
                float(kept_series / class_series) if class_series else float("nan")
            ),
        }
    return summary


def selection_rows(table: SeriesTable, result: SelectionResult, partition: str) -> list[dict]:
    rows = []
    for position, series_index in enumerate(result.series_indices):
        rows.append(
            {
                "partition": partition,
                "series_id": str(table.series_ids[series_index]),
                "patient_id": str(table.patient_ids[series_index]),
                "label_for_posthoc_audit_only": int(table.labels[series_index]),
                "class_name_for_posthoc_audit_only": (
                    "Sick" if int(table.labels[series_index]) == 1 else "Normal"
                ),
                "n_slices": int(table.slice_counts[series_index]),
                "standardized_monai_valid_rate": float(
                    table.standardized_monai_valid_rates[series_index]
                ),
                "assigned_cluster": int(result.cluster_ids[position]),
                "distance_to_training_cluster_centroid": float(
                    result.distances[position]
                ),
                "retained": bool(result.keep_mask[position]),
                "reason": str(result.reasons[position]),
            }
        )
    return rows


def selection_reason_summary_rows(
    table: SeriesTable,
    result: SelectionResult,
    partition: str,
) -> list[dict]:
    """Count retained/rejected series reasons for transparent QC reporting."""

    rows = []
    for label, class_name in ((0, "Normal"), (1, "Sick")):
        class_mask = result.labels == label
        class_reasons = result.reasons[class_mask]
        for reason in sorted(set(map(str, class_reasons.tolist()))):
            count = int(np.sum(class_reasons == reason))
            rows.append(
                {
                    "partition": partition,
                    "class_name_for_posthoc_audit_only": class_name,
                    "reason": reason,
                    "series_count": count,
                    "fraction_within_partition_class": float(
                        count / max(int(np.sum(class_mask)), 1)
                    ),
                }
            )
    return rows


def write_consensus_series_selection_audit(
    path: Path,
    all_fold_rows: Sequence[dict],
):
    """Summarize how often each series is retained across outer selectors.

    This is an interpretability artifact only. It must not be used to rerun the
    same CV as a globally filtered cohort, because the consensus combines
    decisions from multiple outer folds.
    """

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in all_fold_rows:
        grouped[str(row["series_id"])].append(dict(row))

    output_rows = []
    for series_id in sorted(grouped):
        rows = grouped[series_id]
        retained = np.asarray(
            [str(row["retained"]).lower() in {"true", "1"} for row in rows],
            dtype=bool,
        )
        rejected_reasons = [
            str(row["reason"])
            for row, was_retained in zip(rows, retained)
            if not was_retained
        ]
        reason_counts = {
            reason: rejected_reasons.count(reason)
            for reason in sorted(set(rejected_reasons))
        }
        dominant_reason = (
            max(reason_counts, key=lambda reason: (reason_counts[reason], reason))
            if reason_counts
            else "none_retained_in_all_folds"
        )
        retained_fraction = float(np.mean(retained))
        first = rows[0]
        output_rows.append(
            {
                "series_id": series_id,
                "patient_id": str(first["patient_id"]),
                "label_for_posthoc_audit_only": int(
                    first["label_for_posthoc_audit_only"]
                ),
                "class_name_for_posthoc_audit_only": str(
                    first["class_name_for_posthoc_audit_only"]
                ),
                "outer_selectors_evaluated": int(len(rows)),
                "retained_count": int(np.sum(retained)),
                "retained_fraction": retained_fraction,
                "consistently_typical_shared_audit_only": bool(
                    retained_fraction
                    >= CONSENSUS_TYPICAL_MIN_OUTER_FOLD_FRACTION
                ),
                "dominant_rejection_reason": dominant_reason,
                "rejection_reason_counts_json": json.dumps(
                    reason_counts,
                    sort_keys=True,
                ),
                "warning": (
                    "Post-evaluation audit only; never reuse this consensus to "
                    "claim unbiased CV on the same patients."
                ),
            }
        )
    write_csv(path, output_rows)


# =============================================================================
# PATIENT POOLING AND SERIES-STRUCTURE CONTROL
# =============================================================================

def _selected_series_by_patient(
    table: SeriesTable,
    result: SelectionResult | None,
    patient_ids: Sequence[str],
) -> dict[str, list[int]]:
    patient_ids = set(map(str, patient_ids))
    mapping: dict[str, list[int]] = defaultdict(list)
    if result is None:
        candidate_indices = np.flatnonzero(np.isin(table.patient_ids, list(patient_ids)))
    else:
        candidate_indices = result.series_indices[result.keep_mask]
    for series_index in candidate_indices:
        patient_id = str(table.patient_ids[series_index])
        if patient_id in patient_ids:
            mapping[patient_id].append(int(series_index))
    return mapping


def pool_image_patient_rows(
    table: SeriesTable,
    series_embeddings: np.ndarray,
    patient_ids: Sequence[str],
    selection: SelectionResult | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Equal-mean retained series embeddings to one row per covered patient."""

    mapping = _selected_series_by_patient(table, selection, patient_ids)
    rows = []
    labels = []
    covered = []
    uncovered = []
    for patient_id in sorted(map(str, patient_ids)):
        indices = mapping.get(patient_id, [])
        if not indices:
            uncovered.append(patient_id)
            continue
        row = np.mean(series_embeddings[indices], axis=0, dtype=np.float64)
        rows.append(np.asarray(row, dtype=np.float32))
        labels.append(int(table.patient_label_by_id[patient_id]))
        covered.append(patient_id)
    if not rows:
        return (
            np.empty((0, FEATURE_DIMENSION), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=str),
            uncovered,
        )
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(covered),
        uncovered,
    )


def pool_structure_control_rows(
    table: SeriesTable,
    patient_ids: Sequence[str],
    selection: SelectionResult,
    selector_model: SelectorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build a patient vector from selection/cluster composition only.

    No MRI embedding is used. Features comprise normalized retained-cluster
    frequencies, total/retained series counts, retained fraction, mean/max
    typicality distance and mean MONAI-valid rate among retained series.
    """

    n_clusters = selector_model.kmeans.n_clusters
    local_position = {
        int(series_index): position
        for position, series_index in enumerate(selection.series_indices)
    }
    all_by_patient: dict[str, list[int]] = defaultdict(list)
    kept_by_patient: dict[str, list[int]] = defaultdict(list)
    for series_index in selection.series_indices:
        patient_id = str(table.patient_ids[series_index])
        all_by_patient[patient_id].append(int(series_index))
    for series_index in selection.series_indices[selection.keep_mask]:
        patient_id = str(table.patient_ids[series_index])
        kept_by_patient[patient_id].append(int(series_index))

    rows = []
    labels = []
    covered = []
    uncovered = []
    for patient_id in sorted(map(str, patient_ids)):
        all_indices = all_by_patient.get(patient_id, [])
        kept_indices = kept_by_patient.get(patient_id, [])
        if not kept_indices:
            uncovered.append(patient_id)
            continue
        cluster_hist = np.zeros(n_clusters, dtype=np.float64)
        distances = []
        valid_rates = []
        for series_index in kept_indices:
            position = local_position[series_index]
            cluster_id = int(selection.cluster_ids[position])
            cluster_hist[cluster_id] += 1.0
            distances.append(float(selection.distances[position]))
            valid_rates.append(
                float(table.standardized_monai_valid_rates[series_index])
            )
        cluster_hist /= max(cluster_hist.sum(), 1.0)
        total_count = len(all_indices)
        retained_count = len(kept_indices)
        extras = np.asarray(
            [
                math.log1p(total_count),
                math.log1p(retained_count),
                retained_count / max(total_count, 1),
                float(np.mean(distances)),
                float(np.max(distances)),
                float(np.mean(valid_rates)),
            ],
            dtype=np.float64,
        )
        rows.append(np.concatenate([cluster_hist, extras]).astype(np.float32))
        labels.append(int(table.patient_label_by_id[patient_id]))
        covered.append(patient_id)

    dimension = n_clusters + 6
    if not rows:
        return (
            np.empty((0, dimension), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=str),
            uncovered,
        )
    return (
        np.stack(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(covered),
        uncovered,
    )


# =============================================================================
# CLASSIFIER AND METRICS
# =============================================================================

def balanced_patient_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    if set(counts) != {0, 1}:
        raise RuntimeError("Training rows must contain both classes.")
    weights = np.asarray(
        [len(labels) / (2.0 * counts[int(label)]) for label in labels],
        dtype=np.float64,
    )
    return weights


def build_classifier(c_value: float, use_pca: bool) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=PCA_EXPLAINED_VARIANCE,
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
                max_iter=LOGISTIC_MAX_ITER,
                random_state=RANDOM_SEED,
            ),
        )
    )
    return Pipeline(steps)


def fit_classifier(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    model.fit(
        X,
        y,
        scaler__sample_weight=np.ones(len(y), dtype=np.float64),
        classifier__sample_weight=balanced_patient_weights(y),
    )
    return model


def choose_training_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    objective = tpr[finite] - fpr[finite]
    candidates = np.flatnonzero(objective == np.max(objective))
    # Largest threshold among ties is the conservative deterministic choice.
    return float(np.max(thresholds[finite][candidates]))


def binary_metrics(labels, scores, predictions) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def bootstrap_auc_interval(labels, scores, random_state: int) -> list[float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    normal = np.flatnonzero(labels == 0)
    sick = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [
                rng.choice(normal, size=len(normal), replace=True),
                rng.choice(sick, size=len(sick), replace=True),
            ]
        )
        values.append(float(roc_auc_score(labels[sampled], scores[sampled])))
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    return [
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    ]


# =============================================================================
# SPLIT PREPARATION AND NESTED EVALUATION
# =============================================================================

def patient_arrays(table: SeriesTable) -> tuple[np.ndarray, np.ndarray]:
    patient_ids = np.asarray(sorted(table.patient_label_by_id))
    labels = np.asarray(
        [table.patient_label_by_id[str(patient_id)] for patient_id in patient_ids],
        dtype=np.int64,
    )
    return patient_ids, labels


def primary_fold_assignments(table: SeriesTable) -> dict[str, int]:
    """Reuse the V7.1 fold manifest when available; otherwise reconstruct it."""

    run_dir = discover_v71_run_dir()
    if run_dir is not None:
        manifest = run_dir / "manifests" / "patient_fold_manifest.csv"
        if manifest.is_file():
            mapping = {}
            with manifest.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    mapping[str(row["patient_id"])] = int(row["outer_fold"])
            expected, _ = patient_arrays(table)
            if set(mapping) == set(map(str, expected)):
                print(f"[FOLDS] Reused V7.1 manifest: {manifest}", flush=True)
                return mapping
            print(
                "[FOLDS] V7.1 manifest patient set does not match the cache; "
                "reconstructing deterministic folds.",
                flush=True,
            )

    patient_ids, labels = patient_arrays(table)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    mapping = {}
    for fold, (_, valid_indices) in enumerate(
        splitter.split(patient_ids, labels), start=1
    ):
        for index in valid_indices:
            mapping[str(patient_ids[index])] = int(fold)
    print("[FOLDS] Built deterministic patient folds with random_state=42.", flush=True)
    return mapping


class SelectorCache:
    """Cache selectors by exact training-patient set within one script run."""

    def __init__(self, table: SeriesTable):
        self.table = table
        self._models: dict[tuple, SelectorModel] = {}

    def get(self, training_patient_ids: Sequence[str], random_state: int) -> SelectorModel:
        key = (tuple(sorted(map(str, training_patient_ids))), int(random_state))
        if key not in self._models:
            self._models[key] = fit_series_selector(
                self.table,
                key[0],
                random_state=key[1],
            )
        return self._models[key]


def series_indices_for_patients(table: SeriesTable, patient_ids: Sequence[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(table.patient_ids, np.asarray(list(patient_ids))))


def prepare_rows_for_split(
    spec: ExperimentSpec,
    table: SeriesTable,
    series_features: dict[str, np.ndarray],
    train_patients: Sequence[str],
    valid_patients: Sequence[str],
    selector_cache: SelectorCache,
    selector_random_state: int,
) -> dict:
    """Fit/apply selector and create patient matrices for one train/valid split."""

    selector_model = None
    train_selection = None
    valid_selection = None
    if spec.use_series_filter:
        selector_model = selector_cache.get(train_patients, selector_random_state)
        train_selection = apply_series_selector(
            table,
            series_indices_for_patients(table, train_patients),
            selector_model,
        )
        valid_selection = apply_series_selector(
            table,
            series_indices_for_patients(table, valid_patients),
            selector_model,
        )

    if spec.experiment_id == SERIES_STRUCTURE_CONTROL_ID:
        X_train, y_train, p_train, missing_train = pool_structure_control_rows(
            table,
            train_patients,
            train_selection,
            selector_model,
        )
        X_valid, y_valid, p_valid, missing_valid = pool_structure_control_rows(
            table,
            valid_patients,
            valid_selection,
            selector_model,
        )
    else:
        mode_features = series_features[spec.feature_mode]
        X_train, y_train, p_train, missing_train = pool_image_patient_rows(
            table,
            mode_features,
            train_patients,
            train_selection,
        )
        X_valid, y_valid, p_valid, missing_valid = pool_image_patient_rows(
            table,
            mode_features,
            valid_patients,
            valid_selection,
        )

    if len(np.unique(y_train)) != 2:
        raise RuntimeError(
            f"{spec.experiment_id}: series filtering removed a training class."
        )
    if len(y_valid) == 0 or len(np.unique(y_valid)) < 2:
        # AUC is pooled over all OOF patients, but C selection would be unstable
        # if an entire inner/outer validation fold loses all rows or one class.
        raise RuntimeError(
            f"{spec.experiment_id}: selected validation rows are empty or contain "
            f"one class only; missing patients={missing_valid}. The selector is "
            "too strict for unbiased nested evaluation."
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "p_train": p_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "p_valid": p_valid,
        "missing_train": missing_train,
        "missing_valid": missing_valid,
        "selector_model": selector_model,
        "train_selection": train_selection,
        "valid_selection": valid_selection,
    }


def select_c_and_threshold(
    spec: ExperimentSpec,
    table: SeriesTable,
    series_features: dict[str, np.ndarray],
    outer_train_patients: np.ndarray,
    outer_train_labels: np.ndarray,
    selector_cache: SelectorCache,
    outer_fold: int,
    base_seed: int,
) -> tuple[float, float, dict]:
    """Nested C and threshold selection with selector refit in every inner fold."""

    inner = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=int(base_seed + 1000 + outer_fold),
    )
    inner_splits = list(inner.split(outer_train_patients, outer_train_labels))

    # Series matching and patient pooling do not depend on classifier C. Build
    # every inner train/validation representation once, then reuse it for all C
    # values. This reduces selector fits and high-dimensional pooling work by a
    # factor of len(C_GRID) without changing any statistical decision.
    prepared_inner_splits = []
    for inner_fold, (train_indices, valid_indices) in enumerate(
        inner_splits, start=1
    ):
        train_patients = outer_train_patients[train_indices]
        valid_patients = outer_train_patients[valid_indices]
        prepared_inner_splits.append(
            prepare_rows_for_split(
                spec,
                table,
                series_features,
                train_patients,
                valid_patients,
                selector_cache,
                selector_random_state=(
                    base_seed + outer_fold * 100 + inner_fold
                ),
            )
        )

    c_results = {}
    for c_value in C_GRID:
        scores = []
        labels = []
        patients = []
        missing = []
        for prepared in prepared_inner_splits:
            model = fit_classifier(
                build_classifier(c_value, spec.use_pca),
                prepared["X_train"],
                prepared["y_train"],
            )
            fold_scores = model.predict_proba(prepared["X_valid"])[:, 1]
            scores.extend(map(float, fold_scores))
            labels.extend(map(int, prepared["y_valid"]))
            patients.extend(map(str, prepared["p_valid"]))
            missing.extend(map(str, prepared["missing_valid"]))

        y = np.asarray(labels, dtype=np.int64)
        s = np.asarray(scores, dtype=np.float64)
        if len(np.unique(y)) != 2:
            raise RuntimeError(
                f"{spec.experiment_id}: inner OOF retained one class only."
            )
        c_results[float(c_value)] = {
            "auc": float(roc_auc_score(y, s)),
            "labels": y,
            "scores": s,
            "patients": np.asarray(patients),
            "missing_patients": sorted(set(missing)),
        }

    best_auc = max(row["auc"] for row in c_results.values())
    eligible = [
        c_value
        for c_value in C_GRID
        if c_results[float(c_value)]["auc"]
        >= best_auc - C_SELECTION_AUC_TOLERANCE
    ]
    selected_c = float(min(eligible))
    selected = c_results[selected_c]
    threshold = choose_training_threshold(selected["labels"], selected["scores"])
    summary = {
        "selected_c": selected_c,
        "selected_inner_auc": float(selected["auc"]),
        "best_inner_auc": float(best_auc),
        "threshold": float(threshold),
        "inner_auc_by_c": {
            str(c): float(c_results[float(c)]["auc"]) for c in C_GRID
        },
        "inner_missing_patients": selected["missing_patients"],
    }
    return selected_c, threshold, summary


def evaluate_experiment(
    spec: ExperimentSpec,
    table: SeriesTable,
    series_features: dict[str, np.ndarray],
    fold_mapping: dict[str, int],
    output_dir: Path | None,
    base_seed: int,
    write_fold_audits: bool,
    selector_cache: SelectorCache | None = None,
) -> dict:
    """Run one fully nested patient-level experiment on fixed outer folds."""

    patient_ids, patient_labels = patient_arrays(table)
    if selector_cache is None:
        selector_cache = SelectorCache(table)
    score_by_patient = {}
    prediction_by_patient = {}
    fold_by_patient = {}
    threshold_by_patient = {}
    fold_rows = []
    selection_audit_rows = []
    cluster_audit_rows = []

    for outer_fold in range(1, N_SPLITS + 1):
        valid_patients = np.asarray(
            [p for p in patient_ids if fold_mapping[str(p)] == outer_fold]
        )
        train_patients = np.asarray(
            [p for p in patient_ids if fold_mapping[str(p)] != outer_fold]
        )
        train_labels = np.asarray(
            [table.patient_label_by_id[str(p)] for p in train_patients],
            dtype=np.int64,
        )

        selected_c, threshold, inner_summary = select_c_and_threshold(
            spec,
            table,
            series_features,
            train_patients,
            train_labels,
            selector_cache,
            outer_fold=outer_fold,
            base_seed=base_seed,
        )
        prepared = prepare_rows_for_split(
            spec,
            table,
            series_features,
            train_patients,
            valid_patients,
            selector_cache,
            selector_random_state=base_seed + outer_fold * 100,
        )
        model = fit_classifier(
            build_classifier(selected_c, spec.use_pca),
            prepared["X_train"],
            prepared["y_train"],
        )
        scores = model.predict_proba(prepared["X_valid"])[:, 1]
        predictions = (scores >= threshold).astype(np.int64)

        for patient_id, label, score, prediction in zip(
            prepared["p_valid"],
            prepared["y_valid"],
            scores,
            predictions,
        ):
            patient_id = str(patient_id)
            if patient_id in score_by_patient:
                raise RuntimeError(f"Patient {patient_id} received two OOF scores.")
            score_by_patient[patient_id] = float(score)
            prediction_by_patient[patient_id] = int(prediction)
            fold_by_patient[patient_id] = int(outer_fold)
            threshold_by_patient[patient_id] = float(threshold)

        if len(np.unique(prepared["y_valid"])) == 2:
            fold_auc = float(
                roc_auc_score(prepared["y_valid"], scores)
            )
        else:
            fold_auc = float("nan")
        fold_rows.append(
            {
                "outer_fold": int(outer_fold),
                "train_patients_before_filter": int(len(train_patients)),
                "train_patients_after_filter": int(len(prepared["p_train"])),
                "validation_patients_before_filter": int(len(valid_patients)),
                "validation_patients_after_filter": int(len(prepared["p_valid"])),
                "missing_train_patients": "|".join(prepared["missing_train"]),
                "missing_validation_patients": "|".join(prepared["missing_valid"]),
                "selected_c": float(selected_c),
                "training_only_threshold": float(threshold),
                "selected_inner_auc": float(inner_summary["selected_inner_auc"]),
                "best_inner_auc": float(inner_summary["best_inner_auc"]),
                "held_out_auc": fold_auc,
            }
        )

        if write_fold_audits and spec.use_series_filter:
            # ``output_dir`` is primary_cv/<experiment_id>; shared fold-level
            # selector audits belong beside experiment directories under
            # primary_cv/fold_<k>, not at the run root.
            fold_dir = output_dir.parent / f"fold_{outer_fold}"
            if not (fold_dir / "series_selection.csv").is_file():
                rows = selection_rows(
                    table,
                    prepared["train_selection"],
                    "outer_train",
                )
                rows.extend(
                    selection_rows(
                        table,
                        prepared["valid_selection"],
                        "outer_validation",
                    )
                )
                for row in rows:
                    row["outer_fold"] = int(outer_fold)
                selection_audit_rows.extend(rows)
                write_csv(fold_dir / "series_selection.csv", rows)
                reason_rows = selection_reason_summary_rows(
                    table,
                    prepared["train_selection"],
                    "outer_train",
                )
                reason_rows.extend(
                    selection_reason_summary_rows(
                        table,
                        prepared["valid_selection"],
                        "outer_validation",
                    )
                )
                for row in reason_rows:
                    row["outer_fold"] = int(outer_fold)
                write_csv(
                    fold_dir / "selection_reason_summary.csv",
                    reason_rows,
                )
                write_csv(
                    fold_dir / "shared_cluster_summary.csv",
                    prepared["selector_model"].cluster_summary_rows,
                )
                json_write(
                    fold_dir / "coverage_summary.json",
                    {
                        "outer_train": patient_coverage_summary(
                            prepared["train_selection"]
                        ),
                        "outer_validation": patient_coverage_summary(
                            prepared["valid_selection"]
                        ),
                    },
                )

    evaluated_patients = np.asarray(sorted(score_by_patient))
    labels = np.asarray(
        [table.patient_label_by_id[p] for p in evaluated_patients],
        dtype=np.int64,
    )
    scores = np.asarray([score_by_patient[p] for p in evaluated_patients])
    predictions = np.asarray(
        [prediction_by_patient[p] for p in evaluated_patients], dtype=np.int64
    )
    if len(np.unique(labels)) != 2:
        raise RuntimeError(
            f"{spec.experiment_id}: common-support OOF cohort contains one class."
        )
    metrics = binary_metrics(labels, scores, predictions)
    metrics["auc_ci_95"] = bootstrap_auc_interval(
        labels,
        scores,
        random_state=base_seed + 90_000,
    )

    all_patients, all_labels = patient_arrays(table)
    evaluated_set = set(evaluated_patients.tolist())
    coverage_by_class = {}
    for label, name in ((0, "Normal"), (1, "Sick")):
        class_patients = all_patients[all_labels == label]
        covered = [p for p in class_patients if str(p) in evaluated_set]
        coverage_by_class[name] = {
            "total": int(len(class_patients)),
            "evaluated": int(len(covered)),
            "coverage_rate": float(len(covered) / len(class_patients)),
        }

    result = {
        "experiment_id": spec.experiment_id,
        "feature_mode": spec.feature_mode,
        "use_series_filter": bool(spec.use_series_filter),
        "role": spec.role,
        "n_patients_total": int(len(all_patients)),
        "n_patients_evaluated": int(len(evaluated_patients)),
        "coverage_by_class": coverage_by_class,
        "metrics": metrics,
        "fold_rows": fold_rows,
        "patient_ids": evaluated_patients,
        "labels": labels,
        "scores": scores,
        "predictions": predictions,
        "folds": np.asarray([fold_by_patient[p] for p in evaluated_patients]),
        "thresholds": np.asarray(
            [threshold_by_patient[p] for p in evaluated_patients]
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        patient_rows = []
        for patient_id, label, fold, score, prediction, threshold in zip(
            result["patient_ids"],
            result["labels"],
            result["folds"],
            result["scores"],
            result["predictions"],
            result["thresholds"],
        ):
            patient_rows.append(
                {
                    "patient_id": str(patient_id),
                    "true_label": int(label),
                    "outer_fold": int(fold),
                    "oof_score": float(score),
                    "training_only_threshold": float(threshold),
                    "predicted_label": int(prediction),
                }
            )
        write_csv(output_dir / "patient_oof_predictions.csv", patient_rows)
        write_csv(output_dir / "outer_fold_metrics.csv", fold_rows)
        if write_fold_audits and selection_audit_rows:
            write_csv(
                output_dir / "series_selection_all_outer_folds.csv",
                selection_audit_rows,
            )
            write_consensus_series_selection_audit(
                output_dir / "series_consensus_selection_audit_only.csv",
                selection_audit_rows,
            )
        serializable = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "patient_ids",
                "labels",
                "scores",
                "predictions",
                "folds",
                "thresholds",
            }
        }
        json_write(output_dir / "summary.json", serializable)

    return result


def fold_mapping_for_seed(table: SeriesTable, seed: int) -> dict[str, int]:
    patient_ids, labels = patient_arrays(table)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(seed))
    mapping = {}
    for fold, (_, valid) in enumerate(splitter.split(patient_ids, labels), start=1):
        for index in valid:
            mapping[str(patient_ids[index])] = int(fold)
    return mapping


# =============================================================================
# GLOBAL AUDIT — NOT USED FOR REPORTED CROSS-VALIDATED SCORES
# =============================================================================

def run_global_audit(table: SeriesTable, output_dir: Path) -> dict:
    """Fit one all-patient selector only to visualize retained series families.

    Because this audit uses all labels to identify common clusters, its selection
    is NOT supplied to any reported CV model. Its purpose is to create one stable
    review file showing which series appear typical/shared in the released cohort.
    """

    patient_ids, _ = patient_arrays(table)
    model = fit_series_selector(table, patient_ids, RANDOM_SEED + 77_000)
    all_indices = np.arange(len(table.series_ids), dtype=np.int64)
    result = apply_series_selector(table, all_indices, model)
    write_csv(output_dir / "shared_cluster_summary.csv", model.cluster_summary_rows)
    write_csv(
        output_dir / "series_selection.csv",
        selection_rows(table, result, "global_audit_only"),
    )
    summary = {
        "status": "AUDIT_ONLY_NOT_USED_FOR_CV",
        "coverage": patient_coverage_summary(result),
        "n_series_total": int(len(result.series_indices)),
        "n_series_retained": int(np.sum(result.keep_mask)),
        "series_retention_rate": float(np.mean(result.keep_mask)),
        "accepted_cluster_count": int(np.sum(model.accepted_clusters)),
        "total_cluster_count": int(len(model.accepted_clusters)),
        "warning": (
            "This all-patient selection is descriptive only. Reported AUC values "
            "use selectors refitted inside each inner/outer training partition."
        ),
    }
    json_write(output_dir / "global_selection_summary.json", summary)
    return summary


# =============================================================================
# COMPARISON, REPEATED CV AND OPTIONAL PERMUTATION
# =============================================================================

def paired_auc_difference(reference: dict, changed: dict) -> dict:
    common = sorted(
        set(map(str, reference["patient_ids"]))
        & set(map(str, changed["patient_ids"]))
    )
    ref_lookup = {
        str(p): (int(y), float(s))
        for p, y, s in zip(
            reference["patient_ids"], reference["labels"], reference["scores"]
        )
    }
    changed_lookup = {
        str(p): (int(y), float(s))
        for p, y, s in zip(
            changed["patient_ids"], changed["labels"], changed["scores"]
        )
    }
    labels = np.asarray([ref_lookup[p][0] for p in common], dtype=np.int64)
    if any(changed_lookup[p][0] != ref_lookup[p][0] for p in common):
        raise RuntimeError("Paired comparison label mismatch.")
    ref_scores = np.asarray([ref_lookup[p][1] for p in common])
    changed_scores = np.asarray([changed_lookup[p][1] for p in common])
    if len(np.unique(labels)) != 2:
        return {"status": "UNAVAILABLE_ONE_CLASS", "n_common_patients": len(common)}
    return {
        "status": "OK",
        "n_common_patients": int(len(common)),
        "reference_auc": float(roc_auc_score(labels, ref_scores)),
        "changed_auc": float(roc_auc_score(labels, changed_scores)),
        "changed_minus_reference_auc": float(
            roc_auc_score(labels, changed_scores)
            - roc_auc_score(labels, ref_scores)
        ),
    }


def run_repeated_cv(
    specs: dict[str, ExperimentSpec],
    table: SeriesTable,
    series_features: dict[str, np.ndarray],
    output_dir: Path,
) -> dict:
    rows = []
    paired_rows = []
    for repeat_index in range(REPEATED_CV_REPEATS):
        seed = REPEATED_CV_SEED_START + repeat_index
        folds = fold_mapping_for_seed(table, seed)
        repeat_results = {}
        repeat_selector_cache = SelectorCache(table)
        print(
            f"[REPEATED CV] repeat {repeat_index + 1}/{REPEATED_CV_REPEATS}, seed={seed}",
            flush=True,
        )
        for experiment_id in REPEATED_CV_EXPERIMENT_IDS:
            result = evaluate_experiment(
                specs[experiment_id],
                table,
                series_features,
                folds,
                output_dir=None,
                base_seed=seed,
                write_fold_audits=False,
                selector_cache=repeat_selector_cache,
            )
            repeat_results[experiment_id] = result
            rows.append(
                {
                    "repeat_index": int(repeat_index + 1),
                    "outer_cv_seed": int(seed),
                    "experiment_id": experiment_id,
                    "auc": float(result["metrics"]["auc"]),
                    "auprc": float(result["metrics"]["auprc"]),
                    "n_patients_evaluated": int(result["n_patients_evaluated"]),
                    "normal_coverage_rate": float(
                        result["coverage_by_class"]["Normal"]["coverage_rate"]
                    ),
                    "sick_coverage_rate": float(
                        result["coverage_by_class"]["Sick"]["coverage_rate"]
                    ),
                }
            )

        # Recalculate each delta on the patient intersection for this exact
        # repeat. This is essential when comparing the all-series reference,
        # which covers every patient, with a common-support model that may
        # legitimately exclude patients lacking any matched series.
        primary_result = repeat_results[PRIMARY_EXPERIMENT_ID]
        for experiment_id, comparator_result in repeat_results.items():
            if experiment_id == PRIMARY_EXPERIMENT_ID:
                continue
            paired = paired_auc_difference(comparator_result, primary_result)
            paired_rows.append(
                {
                    "repeat_index": int(repeat_index + 1),
                    "outer_cv_seed": int(seed),
                    "primary_experiment_id": PRIMARY_EXPERIMENT_ID,
                    "comparator_experiment_id": experiment_id,
                    "status": paired.get("status"),
                    "n_common_patients": paired.get("n_common_patients"),
                    "primary_minus_comparator_auc": paired.get(
                        "changed_minus_reference_auc"
                    ),
                }
            )

    write_csv(output_dir / "repeated_runs.csv", rows)
    write_csv(output_dir / "repeated_paired_deltas.csv", paired_rows)
    summary = {"status": "OK", "repeats": REPEATED_CV_REPEATS, "experiments": {}}
    for experiment_id in REPEATED_CV_EXPERIMENT_IDS:
        values = np.asarray(
            [row["auc"] for row in rows if row["experiment_id"] == experiment_id]
        )
        coverage = np.asarray(
            [
                row["n_patients_evaluated"]
                for row in rows
                if row["experiment_id"] == experiment_id
            ]
        )
        summary["experiments"][experiment_id] = {
            "auc_mean": float(np.mean(values)),
            "auc_median": float(np.median(values)),
            "auc_q25": float(np.quantile(values, 0.25)),
            "auc_q75": float(np.quantile(values, 0.75)),
            "auc_minimum": float(np.min(values)),
            "auc_maximum": float(np.max(values)),
            "evaluated_patients_median": float(np.median(coverage)),
            "evaluated_patients_minimum": int(np.min(coverage)),
        }

    # Same-seed paired deltas against the matched A17 candidate, always
    # calculated on the common patient rows for that repeat.
    paired = {}
    for experiment_id in REPEATED_CV_EXPERIMENT_IDS:
        if experiment_id == PRIMARY_EXPERIMENT_ID:
            continue
        comparison_rows = [
            row
            for row in paired_rows
            if row["comparator_experiment_id"] == experiment_id
            and row["status"] == "OK"
        ]
        deltas = np.asarray(
            [row["primary_minus_comparator_auc"] for row in comparison_rows],
            dtype=np.float64,
        )
        common_counts = np.asarray(
            [row["n_common_patients"] for row in comparison_rows],
            dtype=np.int64,
        )
        paired[f"{PRIMARY_EXPERIMENT_ID}_MINUS_{experiment_id}"] = {
            "successful_repeats": int(len(deltas)),
            "median_common_patients": (
                float(np.median(common_counts)) if len(common_counts) else None
            ),
            "minimum_common_patients": (
                int(np.min(common_counts)) if len(common_counts) else None
            ),
            "median_delta_auc": (float(np.median(deltas)) if len(deltas) else None),
            "q25_delta_auc": (
                float(np.quantile(deltas, 0.25)) if len(deltas) else None
            ),
            "q75_delta_auc": (
                float(np.quantile(deltas, 0.75)) if len(deltas) else None
            ),
            "fraction_primary_better": (
                float(np.mean(deltas > 0)) if len(deltas) else None
            ),
            "fraction_primary_not_worse": (
                float(np.mean(deltas >= 0)) if len(deltas) else None
            ),
        }
    summary["paired_deltas"] = paired
    json_write(output_dir / "summary.json", summary)
    return summary


def run_permutation_test(
    spec: ExperimentSpec,
    table: SeriesTable,
    series_features: dict[str, np.ndarray],
    observed_auc: float,
    output_dir: Path,
) -> dict:
    """Repeat the entire nested selector under patient-label permutations."""

    rng = np.random.default_rng(PERMUTATION_RANDOM_STATE)
    patient_ids, original_labels = patient_arrays(table)
    rows = []
    original_series_labels = table.labels.copy()
    original_patient_label_map = dict(table.patient_label_by_id)

    try:
        for permutation_index in range(PERMUTATION_REPLICATES):
            permuted = rng.permutation(original_labels)
            mapping = {
                str(patient_id): int(label)
                for patient_id, label in zip(patient_ids, permuted)
            }
            table.patient_label_by_id = mapping
            table.labels = np.asarray(
                [mapping[str(patient_id)] for patient_id in table.patient_ids],
                dtype=np.int64,
            )
            folds = fold_mapping_for_seed(table, RANDOM_SEED)
            result = evaluate_experiment(
                spec,
                table,
                series_features,
                folds,
                output_dir=None,
                base_seed=RANDOM_SEED,
                write_fold_audits=False,
            )
            rows.append(
                {
                    "permutation_index": int(permutation_index + 1),
                    "auc": float(result["metrics"]["auc"]),
                    "n_patients_evaluated": int(result["n_patients_evaluated"]),
                }
            )
            if (permutation_index + 1) % max(1, PERMUTATION_REPLICATES // 10) == 0:
                print(
                    f"[PERMUTATION] {permutation_index + 1}/{PERMUTATION_REPLICATES}",
                    flush=True,
                )
    finally:
        table.labels = original_series_labels
        table.patient_label_by_id = original_patient_label_map

    null = np.asarray([row["auc"] for row in rows])
    p_value = float((1 + np.sum(null >= observed_auc)) / (len(null) + 1))
    summary = {
        "status": "OK",
        "experiment_id": spec.experiment_id,
        "observed_auc": float(observed_auc),
        "replicates": int(PERMUTATION_REPLICATES),
        "null_mean": float(np.mean(null)),
        "null_median": float(np.median(null)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "empirical_one_sided_p_value": p_value,
        "selector_refitted_after_every_label_permutation": True,
    }
    write_csv(output_dir / "permutation_runs.csv", rows)
    json_write(output_dir / "summary.json", summary)
    return summary


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration():
    ids = [spec.experiment_id for spec in EXPERIMENT_SPECS]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique.")
    if set(ids) != set(ALL_EXPERIMENT_IDS):
        raise ValueError("Experiment registry and ALL_EXPERIMENT_IDS disagree.")
    if PRIMARY_EXPERIMENT_ID not in ids:
        raise ValueError("Primary experiment is missing.")
    if not 2 <= SERIES_CLUSTER_COUNT <= 100:
        raise ValueError("SERIES_CLUSTER_COUNT must lie in [2,100].")
    if not 0.0 < CLUSTER_TYPICALITY_RADIUS_QUANTILE < 1.0:
        raise ValueError("Invalid cluster radius quantile.")
    if MIN_PATIENTS_PER_CLASS_IN_CLUSTER < 1:
        raise ValueError("Cluster patient support must be positive.")
    if MIN_SERIES_PER_CLASS_IN_CLUSTER < 1:
        raise ValueError("Cluster series support must be positive.")
    if not 0.0 < MIN_CLASS_SERIES_BALANCE_RATIO <= 1.0:
        raise ValueError("Invalid class series-balance ratio.")
    if not 0.0 < MIN_CLASS_PATIENT_BALANCE_RATIO <= 1.0:
        raise ValueError("Invalid class patient-balance ratio.")
    if not 0.0 < MAX_SINGLE_PATIENT_SERIES_FRACTION_IN_SHARED_CLUSTER <= 1.0:
        raise ValueError("Invalid single-patient cluster dominance limit.")
    if not 0.0 < MIN_TRAIN_PATIENT_COVERAGE_PER_CLASS <= 1.0:
        raise ValueError("Invalid training patient coverage floor.")
    if not 0.0 < CONSENSUS_TYPICAL_MIN_OUTER_FOLD_FRACTION <= 1.0:
        raise ValueError("Invalid audit-only consensus selection fraction.")
    if set(REPEATED_CV_EXPERIMENT_IDS) - set(ids):
        raise ValueError("Repeated-CV experiment list contains unknown IDs.")
    if len(C_GRID) == 0 or any(float(c) <= 0 for c in C_GRID):
        raise ValueError("C_GRID must contain positive values.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    started = time.perf_counter()
    validate_configuration()
    cache_dir, source_metadata = discover_feature_cache()
    bank = load_feature_bank(cache_dir, source_metadata)
    table = build_series_table(bank)

    configuration = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "random_seed": RANDOM_SEED,
        "feature_mode_by_experiment": FEATURE_MODE_BY_EXPERIMENT,
        "filtered_experiment_ids": FILTERED_EXPERIMENT_IDS,
        "series_structure_control_id": SERIES_STRUCTURE_CONTROL_ID,
        "primary_experiment_id": PRIMARY_EXPERIMENT_ID,
        "series_cluster_count": SERIES_CLUSTER_COUNT,
        "minimum_slices_per_series": MIN_SLICES_PER_SERIES,
        "minimum_standardized_monai_valid_rate": (
            MIN_STANDARDIZED_MONAI_VALID_RATE
        ),
        "minimum_series_per_class_in_cluster": (
            MIN_SERIES_PER_CLASS_IN_CLUSTER
        ),
        "minimum_patients_per_class_in_cluster": (
            MIN_PATIENTS_PER_CLASS_IN_CLUSTER
        ),
        "minimum_class_series_balance_ratio": (
            MIN_CLASS_SERIES_BALANCE_RATIO
        ),
        "minimum_class_patient_balance_ratio": (
            MIN_CLASS_PATIENT_BALANCE_RATIO
        ),
        "maximum_single_patient_series_fraction_in_shared_cluster": (
            MAX_SINGLE_PATIENT_SERIES_FRACTION_IN_SHARED_CLUSTER
        ),
        "cluster_typicality_radius_quantile": (
            CLUSTER_TYPICALITY_RADIUS_QUANTILE
        ),
        "maximum_class_centroid_gap_ratio": MAX_CLASS_CENTROID_GAP_RATIO,
        "minimum_train_patient_coverage_per_class": (
            MIN_TRAIN_PATIENT_COVERAGE_PER_CLASS
        ),
        "consensus_typical_min_outer_fold_fraction_audit_only": (
            CONSENSUS_TYPICAL_MIN_OUTER_FOLD_FRACTION
        ),
        "provenance_descriptor_names": PROVENANCE_DESCRIPTOR_NAMES,
        "standardization_descriptor_names": STANDARDIZATION_DESCRIPTOR_NAMES,
        "n_splits": N_SPLITS,
        "inner_splits": INNER_SPLITS,
        "c_grid": C_GRID,
        "c_selection_auc_tolerance": C_SELECTION_AUC_TOLERANCE,
        "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "repeated_cv_enabled": RUN_REPEATED_CV,
        "repeated_cv_repeats": REPEATED_CV_REPEATS,
        "repeated_cv_experiment_ids": REPEATED_CV_EXPERIMENT_IDS,
        "permutation_enabled": RUN_PERMUTATION_TEST,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "methodological_warning": (
            "Global all-patient selection is audit-only. Every reported CV score "
            "uses selectors fitted within inner/outer training partitions."
        ),
    }
    tag = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    output_dir = OUTPUT_ROOT / f"series_overlap_harmonization__{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_write(output_dir / "configuration.json", configuration)
    json_write(
        output_dir / "source_cache.json",
        {
            "cache_dir": str(cache_dir),
            "source_metadata": source_metadata,
        },
    )

    write_series_descriptors(
        output_dir / "global_audit" / "series_descriptors.csv",
        table,
    )
    try:
        global_audit = run_global_audit(table, output_dir / "global_audit")
    except Exception as error:
        # The all-patient selector is descriptive and must never control whether
        # the leakage-safe fold-local evaluation can run. Preserve its failure
        # transparently, then continue with selectors fitted inside CV.
        global_audit = {
            "status": "FAILED_AUDIT_ONLY",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        json_write(
            output_dir / "global_audit" / "global_selection_summary.json",
            global_audit,
        )
        print(
            "[GLOBAL AUDIT] Failed descriptively but fold-local evaluation will "
            f"continue: {type(error).__name__}: {error}",
            flush=True,
        )
    series_features, series_cache_dir = build_or_load_series_embeddings(
        bank,
        table,
        OUTPUT_ROOT,
    )

    fold_mapping = primary_fold_assignments(table)
    specs = {spec.experiment_id: spec for spec in EXPERIMENT_SPECS}
    results = {}
    primary_selector_cache = SelectorCache(table)
    for index, spec in enumerate(EXPERIMENT_SPECS, start=1):
        print(
            f"\n[PRIMARY CV] {index}/{len(EXPERIMENT_SPECS)}: {spec.experiment_id}",
            flush=True,
        )
        result = evaluate_experiment(
            spec,
            table,
            series_features,
            fold_mapping,
            output_dir=output_dir / "primary_cv" / spec.experiment_id,
            base_seed=RANDOM_SEED,
            write_fold_audits=(spec.experiment_id == PRIMARY_EXPERIMENT_ID),
            selector_cache=primary_selector_cache,
        )
        results[spec.experiment_id] = result
        print(
            f"[PRIMARY CV] {spec.experiment_id}: AUC="
            f"{result['metrics']['auc']:.4f}; patients="
            f"{result['n_patients_evaluated']}/{result['n_patients_total']}",
            flush=True,
        )

    summary_rows = []
    for spec in EXPERIMENT_SPECS:
        result = results[spec.experiment_id]
        summary_rows.append(
            {
                "experiment_id": spec.experiment_id,
                "role": spec.role,
                "feature_mode": spec.feature_mode or "series_structure_only",
                "series_filter": bool(spec.use_series_filter),
                "n_patients_evaluated": result["n_patients_evaluated"],
                "normal_coverage_rate": result["coverage_by_class"]["Normal"][
                    "coverage_rate"
                ],
                "sick_coverage_rate": result["coverage_by_class"]["Sick"][
                    "coverage_rate"
                ],
                "auc": result["metrics"]["auc"],
                "auc_ci_lower": result["metrics"]["auc_ci_95"][0],
                "auc_ci_upper": result["metrics"]["auc_ci_95"][1],
                "auprc": result["metrics"]["auprc"],
                "sensitivity": result["metrics"]["sensitivity"],
                "specificity": result["metrics"]["specificity"],
                "f1": result["metrics"]["f1"],
            }
        )
    summary_rows.sort(key=lambda row: (-float(row["auc"]), row["experiment_id"]))
    write_csv(output_dir / "comparison" / "experiment_summary.csv", summary_rows)

    comparisons = {
        "filtered_A17_vs_all_series_A17": paired_auc_difference(
            results["A17_ALL_SERIES_REFERENCE"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_A12": paired_auc_difference(
            results["A12_SHARED_TYPICAL_SERIES"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_shape_suppressed_A16": paired_auc_difference(
            results["A16_SHARED_TYPICAL_SERIES"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_filtered_full_image_A9": paired_auc_difference(
            results["A9_SHARED_TYPICAL_SERIES"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_support_only_C31": paired_auc_difference(
            results["C31_SHARED_TYPICAL_SUPPORT_ONLY"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_shuffled_C32": paired_auc_difference(
            results["C32_SHARED_TYPICAL_SHUFFLED_INTENSITY"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_complement_C33": paired_auc_difference(
            results["C33_SHARED_TYPICAL_EXACT_COMPLEMENT"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_conservative_outside_C28": paired_auc_difference(
            results["C28_SHARED_TYPICAL_CONSERVATIVE_OUTSIDE"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_fixed_periphery_C29": paired_auc_difference(
            results["C29_SHARED_TYPICAL_FIXED_PERIPHERY"],
            results[PRIMARY_EXPERIMENT_ID],
        ),
        "filtered_A17_vs_series_structure_only": paired_auc_difference(
            results[SERIES_STRUCTURE_CONTROL_ID],
            results[PRIMARY_EXPERIMENT_ID],
        ),
    }
    json_write(
        output_dir / "comparison" / "paired_auc_differences.json",
        comparisons,
    )

    if RUN_REPEATED_CV:
        repeated_summary = run_repeated_cv(
            specs,
            table,
            series_features,
            output_dir / "repeated_cv",
        )
    else:
        repeated_summary = {"status": "SKIPPED_DISABLED"}
        json_write(output_dir / "repeated_cv" / "summary.json", repeated_summary)

    if RUN_PERMUTATION_TEST:
        permutation_summary = run_permutation_test(
            specs[PRIMARY_EXPERIMENT_ID],
            table,
            series_features,
            observed_auc=results[PRIMARY_EXPERIMENT_ID]["metrics"]["auc"],
            output_dir=output_dir / "permutation",
        )
    else:
        permutation_summary = {"status": "SKIPPED_DISABLED"}
        json_write(output_dir / "permutation" / "summary.json", permutation_summary)

    final_report = {
        "status": "OK",
        "script_version": SCRIPT_VERSION,
        "source_feature_cache": str(cache_dir),
        "series_embedding_cache": str(series_cache_dir),
        "global_audit": global_audit,
        "primary_experiment_id": PRIMARY_EXPERIMENT_ID,
        "primary_result": {
            key: value
            for key, value in results[PRIMARY_EXPERIMENT_ID].items()
            if key
            not in {
                "patient_ids",
                "labels",
                "scores",
                "predictions",
                "folds",
                "thresholds",
                "fold_rows",
            }
        },
        "experiment_summary": summary_rows,
        "paired_comparisons": comparisons,
        "repeated_cv": repeated_summary,
        "permutation": permutation_summary,
        "interpretation": {
            "valid_claim": (
                "Performance is conditional on series profiles that lie in the "
                "fold-local Normal/Sick common support."
            ),
            "invalid_claim": (
                "The filter does not prove sequence identity or clinical CAD "
                "specificity, and global audit selection is not an evaluation set."
            ),
            "coverage_requirement": (
                "AUC must always be reported together with overall and class-"
                "specific patient coverage."
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    json_write(output_dir / "final_report.json", final_report)

    print("\n" + "=" * 88, flush=True)
    print("FINAL FOLD-LOCAL SHARED-SERIES RESULT", flush=True)
    print("=" * 88, flush=True)
    primary = results[PRIMARY_EXPERIMENT_ID]
    print(
        f"{PRIMARY_EXPERIMENT_ID}: AUC={primary['metrics']['auc']:.4f} "
        f"CI=[{primary['metrics']['auc_ci_95'][0]:.4f}, "
        f"{primary['metrics']['auc_ci_95'][1]:.4f}]",
        flush=True,
    )
    print(
        "Patient coverage: Normal="
        f"{primary['coverage_by_class']['Normal']['coverage_rate']:.1%}, Sick="
        f"{primary['coverage_by_class']['Sick']['coverage_rate']:.1%}.",
        flush=True,
    )
    print(
        "All-series A17 reference AUC="
        f"{results['A17_ALL_SERIES_REFERENCE']['metrics']['auc']:.4f}.",
        flush=True,
    )
    print(
        "Series-structure-only control AUC="
        f"{results[SERIES_STRUCTURE_CONTROL_ID]['metrics']['auc']:.4f}.",
        flush=True,
    )
    print(f"Outputs: {output_dir}", flush=True)
    print(f"Runtime: {elapsed_text(time.perf_counter() - started)}", flush=True)
    print("=" * 88, flush=True)


def run_with_console_logging():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / "series_overlap_harmonization_console_output.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_handle:
        sys.stdout = TeeStream(original_stdout, log_handle)
        sys.stderr = TeeStream(original_stderr, log_handle)
        try:
            main()
        except Exception:
            print("\n[SERIES OVERLAP SUITE] FATAL ERROR", flush=True)
            traceback.print_exc()
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    run_with_console_logging()
