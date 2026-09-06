#%% ============================================================================
# CAD CARDIAC MRI — MATHEMATICAL DECONFOUNDING EXTENSION V7
# ================================================================================
#
# PURPOSE
# -------
# This single-file extension consumes the frozen per-slice feature bank produced
# by the V6 cardiac-MRI pipeline. It does NOT repeat JPEG decoding, MONAI
# segmentation, or EfficientNet inference. Instead, it evaluates several small-
# sample mathematical learning algorithms at the validated Directory_* patient
# level:
#
#   A17-R   Reproduced V6 A17 baseline (hierarchical pooling + LR/PCA)
#   A21     Fold-Local Nuisance Orthogonalization (FLNO) + Logistic Regression
#   A22     FLNO + confounder-matched pairwise AUC ranking (CMP-AUC)
#   A23     Robust Median-of-Means pooling (RMoM) + FLNO + Logistic Regression
#   A24     RMoM + FLNO + CMP-AUC
#
# Matched controls are evaluated with the same hierarchical or RMoM pooling:
#
#   C31     Exact A17 binary support without MRI intensity
#   C32     A17 support with the exact visible intensity multiset spatially shuffled
#   C33     Independently normalized exact complement of the A17 support
#
# WHY A POST-CACHE EXTENSION?
# ---------------------------
# All proposed V7 methods operate on frozen embeddings and patient-level nuisance
# metadata. Reusing the V6 feature bank avoids repeating approximately 63,425
# MONAI and EfficientNet inferences every time a mathematical objective changes.
# It also makes the new evaluation contract easier to audit: the image encoder
# and all pixels are frozen before any V7 label-dependent operation begins.
#
# CS50 AI CONNECTIONS
# -------------------
# The implementation deliberately mirrors ideas taught in CS50's Introduction to
# Artificial Intelligence with Python:
#
#   * Optimization / objective functions:
#       inner-CV hyperparameters are selected with a deconfounded utility rather
#       than AUROC alone.
#   * Regression / learning:
#       FLNO estimates and subtracts the embedding component predictable from
#       non-clinical nuisance variables using ridge regression.
#   * k-nearest-neighbour intuition:
#       CMP-AUC gives the greatest ranking weight to positive/negative patients
#       with similar nuisance profiles.
#   * Maximum-margin / ranking:
#       CMP-AUC learns a linear ordering from positive-minus-negative patient
#       differences, which is directly related to AUROC.
#   * Robust optimization:
#       RMoM uses block means and a geometric median so a small number of unusual
#       or duplicated slice groups cannot dominate one series representation.
#
# IMPORTANT LIMITATION
# --------------------
# This is exploratory internal validation on only 30 released Directory_*
# patients. No internal algorithm can prove external clinical generalization or
# fully identify CAD separately from protocol when label and protocol overlap are
# limited. Sequence/view annotation and an independent cohort remain necessary.
#
# EXPECTED V6 CACHE CONTRACT
# --------------------------
# The script automatically searches for a V6 feature-bank directory containing:
#
#   features__standardized_hard_support_region_norm.npy
#   features__standardized_a17_exact_support_mask_only.npy
#   features__standardized_a17_support_intensity_affine_shuffled.npy
#   features__standardized_a17_exact_support_complement_region_norm.npy
#
# plus the shared V6 arrays (labels, patient_ids, series_ids, sample_indices,
# provenance_features, standardization_features, standardized MONAI QC arrays).
#
# Cache discovery is session-portable:
#   1. same-session caches under /kaggle/working are detected automatically;
#   2. previous V6 notebook outputs attached to a fresh Kaggle session are
#      searched recursively under /kaggle/input, regardless of dataset slug;
#   3. CAD_V6_FEATURE_CACHE_DIR may point either to the exact fingerprint folder
#      OR to any ancestor containing feature_bank_cache;
#   4. CAD_V6_SEARCH_ROOTS can supply additional os.pathsep-separated ancestors.
#
# The script also tries to reuse the V6 patient_fold_manifest.csv. Set
# CAD_V6_SUITE_DIR explicitly when several V6 suite folders exist.
#
# ================================================================================

from __future__ import annotations

import csv
import hashlib
import json
import os

# The V7 workload contains many tiny fold-local matrix fits. Restricting native
# BLAS/OpenMP pools to one thread avoids oversubscription and rare thread-pool
# stalls in notebook runtimes, while improving reproducibility for 30-row jobs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import platform
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
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
from sklearn.preprocessing import StandardScaler


# ================================================================================
# CONFIGURATION
# ================================================================================

RANDOM_SEED = 42
N_SPLITS = 5
INNER_SPLITS = 3
INNER_RANDOM_STATE = RANDOM_SEED + 1_000
C_SELECTION_AUC_TOLERANCE = 0.01
CLASSIFIER_C_GRID = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_MAX_ITER = 8_000
PATIENT_PCA_EXPLAINED_VARIANCE = 0.95
PATIENT_PCA_MAX_COMPONENTS = 10

# FLNO grid. The grid is intentionally compact because only 24 patients are
# available in each outer-training cohort.
FLNO_ALPHA_GRID = (0.50, 0.75, 1.00)
FLNO_RIDGE_GRID = (0.10, 1.00)
NUISANCE_PCA_EXPLAINED_VARIANCE = 0.95
NUISANCE_PCA_MAX_COMPONENTS = 5

# CMP-AUC pair weighting. tau is derived from the median cross-class nuisance
# distance inside the current training fold and multiplied by one of these fixed
# values. The minimum pair weight prevents all but one pair from disappearing.
PAIRWISE_C_GRID = (0.01, 0.1, 1.0)
PAIRWISE_TAU_MULTIPLIER_GRID = (0.75, 1.25)
PAIRWISE_MIN_WEIGHT = 0.05

# Deconfounded Validation Utility (DVU):
#   inner AUROC - lambda * maximum within-class absolute Spearman correlation
#                            between model score and up to five label-blind
#                            nuisance PCA coordinates.
# Conditioning the correlation diagnostic on class reduces the extent to which a
# real label association is penalized merely because the label itself correlates
# with a nuisance variable.
DVU_NUISANCE_CORRELATION_PENALTY = 0.10
DVU_SELECTION_TOLERANCE = 0.0025

# Robust Median-of-Means pooling.
RMOM_BLOCKS_PER_SERIES = 5
GEOMETRIC_MEDIAN_MAX_ITERATIONS = 100
GEOMETRIC_MEDIAN_TOLERANCE = 1e-6
GEOMETRIC_MEDIAN_EPSILON = 1e-8

# Evaluation workload. These stages use only 30 patient rows and frozen features.
BOOTSTRAP_REPLICATES = int(os.environ.get("CAD_V7_BOOTSTRAP_REPLICATES", "2000"))
PAIRED_BOOTSTRAP_REPLICATES = int(
    os.environ.get("CAD_V7_PAIRED_BOOTSTRAP_REPLICATES", "2000")
)
REPEATED_NESTED_CV_REPEATS = int(
    os.environ.get("CAD_V7_REPEATED_CV_REPEATS", "50")
)
REPEATED_NESTED_CV_RANDOM_STATE = RANDOM_SEED + 20_000
RUN_SELECTION_ADJUSTED_PERMUTATION = (
    os.environ.get("CAD_V7_RUN_SELECTION_ADJUSTED_PERMUTATION", "1")
    not in {"0", "false", "False"}
)
SELECTION_ADJUSTED_PERMUTATION_REPLICATES = int(
    os.environ.get("CAD_V7_SELECTION_ADJUSTED_PERMUTATIONS", "200")
)
SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE = RANDOM_SEED + 60_000
VERBOSE_INNER_FOR_CANDIDATES = (
    os.environ.get("CAD_V7_VERBOSE_INNER", "1")
    not in {"0", "false", "False"}
)
# Detailed parameter rows are printed for the five candidate models only. The
# 15 algorithm-matched controls still report completion metrics but do not flood
# the Kaggle log with thousands of nearly identical inner-grid lines.

# A small protocol-cluster stress test is descriptive only. The clusters are
# fitted without labels inside each training fold; it is not a replacement for
# real scanner/sequence metadata.
RUN_LEAVE_ONE_NUISANCE_CLUSTER_OUT_AUDIT = (
    os.environ.get("CAD_V7_RUN_NUISANCE_CLUSTER_AUDIT", "1")
    not in {"0", "false", "False"}
)
NUISANCE_CLUSTER_COUNT = 2

V6_A17_FEATURE_MODE = "standardized_hard_support_region_norm"
V6_C31_FEATURE_MODE = "standardized_a17_exact_support_mask_only"
V6_C32_FEATURE_MODE = "standardized_a17_support_intensity_affine_shuffled"
V6_C33_FEATURE_MODE = (
    "standardized_a17_exact_support_complement_region_norm"
)
REQUIRED_V6_FEATURE_MODES = (
    V6_A17_FEATURE_MODE,
    V6_C31_FEATURE_MODE,
    V6_C32_FEATURE_MODE,
    V6_C33_FEATURE_MODE,
)

# Conservative metadata-only nuisance inputs. Central intensity, entropy,
# sharpness and raw image texture are intentionally excluded.
PROVENANCE_NUISANCE_FEATURE_NAMES = (
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
STANDARDIZATION_NUISANCE_FEATURE_NAMES = (
    "crop_applied",
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

# The conservative primary nuisance matrix excludes MONAI area/confidence because
# those variables may contain real anatomy. A separate sensitivity matrix adds
# standardized MONAI-QC summaries and is reported, but it does not define A21-A24.
INCLUDE_MONAI_QC_IN_PRIMARY_FLNO = False

# Automatic location defaults.
if os.path.exists("/kaggle/working"):
    DEFAULT_V6_OUTPUT_ROOT = Path("/kaggle/working/cad_patient_pipeline_outputs")
    DEFAULT_V7_OUTPUT_ROOT = Path(
        "/kaggle/working/cad_v7_mathematical_deconfounding"
    )
else:
    DEFAULT_V6_OUTPUT_ROOT = Path.cwd() / "cad_patient_pipeline_outputs"
    DEFAULT_V7_OUTPUT_ROOT = Path.cwd() / "cad_v7_mathematical_deconfounding"

V6_OUTPUT_ROOT = Path(
    os.environ.get("CAD_V6_OUTPUT_ROOT", str(DEFAULT_V6_OUTPUT_ROOT))
)
V6_FEATURE_CACHE_DIR_OVERRIDE = os.environ.get("CAD_V6_FEATURE_CACHE_DIR")
V6_SEARCH_ROOTS_OVERRIDE = os.environ.get("CAD_V6_SEARCH_ROOTS")
# Optional os.pathsep-separated search roots. This is useful when a previous V6
# notebook output has been attached to a fresh Kaggle session under /kaggle/input.
# Unlike CAD_V6_FEATURE_CACHE_DIR, each entry may point to any ancestor directory;
# V7 searches recursively beneath it for a compatible cache contract.
V6_SUITE_DIR_OVERRIDE = os.environ.get("CAD_V6_SUITE_DIR")
OUTPUT_ROOT = Path(
    os.environ.get("CAD_V7_OUTPUT_ROOT", str(DEFAULT_V7_OUTPUT_ROOT))
)


@dataclass(frozen=True)
class ModelSpec:
    """One predeclared V7 model or matched control."""

    experiment_id: str
    description: str
    feature_mode: str
    pooling: str
    learner: str
    role: str
    use_flno: bool = False
    use_pairwise_auc: bool = False
    use_dvu_for_selection: bool = False


MODEL_SPECS = (
    ModelSpec(
        experiment_id="A17R_V6_REPRODUCED_HIER_LR_PCA",
        description=(
            "V6 A17 frozen slice embeddings with hierarchical series-to-patient "
            "pooling and the reviewed scaler/PCA/Logistic-Regression path."
        ),
        feature_mode=V6_A17_FEATURE_MODE,
        pooling="hierarchical",
        learner="logistic_regression",
        role="locked_reference",
    ),
    ModelSpec(
        experiment_id="A21_FLNO_META_HIER_LR",
        description=(
            "A17 hierarchical patient embeddings after fold-local nuisance "
            "orthogonalization, followed by Logistic Regression."
        ),
        feature_mode=V6_A17_FEATURE_MODE,
        pooling="hierarchical",
        learner="flno_logistic_regression",
        role="v7_candidate",
        use_flno=True,
        use_dvu_for_selection=True,
    ),
    ModelSpec(
        experiment_id="A22_FLNO_META_MATCHED_PAIRWISE_AUC",
        description=(
            "A17 hierarchical embeddings after FLNO, optimized with a "
            "confounder-matched pairwise logistic ranking objective."
        ),
        feature_mode=V6_A17_FEATURE_MODE,
        pooling="hierarchical",
        learner="flno_pairwise_auc",
        role="v7_prospective_candidate",
        use_flno=True,
        use_pairwise_auc=True,
        use_dvu_for_selection=True,
    ),
    ModelSpec(
        experiment_id="A23_RMOM_FLNO_META_LR",
        description=(
            "A17 slice embeddings pooled by robust series-level median-of-means, "
            "then FLNO and Logistic Regression."
        ),
        feature_mode=V6_A17_FEATURE_MODE,
        pooling="robust_median_of_means",
        learner="flno_logistic_regression",
        role="v7_candidate",
        use_flno=True,
        use_dvu_for_selection=True,
    ),
    ModelSpec(
        experiment_id="A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
        description=(
            "RMoM A17 patient embeddings followed by FLNO and nuisance-matched "
            "pairwise AUC optimization."
        ),
        feature_mode=V6_A17_FEATURE_MODE,
        pooling="robust_median_of_means",
        learner="flno_pairwise_auc",
        role="v7_candidate",
        use_flno=True,
        use_pairwise_auc=True,
        use_dvu_for_selection=True,
    ),
    ModelSpec(
        experiment_id="C31R_EXACT_SUPPORT_ONLY_HIER_LR_PCA",
        description="V6 C31 exact-support-only control with hierarchical pooling.",
        feature_mode=V6_C31_FEATURE_MODE,
        pooling="hierarchical",
        learner="logistic_regression",
        role="matched_control",
    ),
    ModelSpec(
        experiment_id="C32R_SHUFFLED_INTENSITY_HIER_LR_PCA",
        description=(
            "V6 C32 exact-support and exact-histogram spatial-shuffle control "
            "with hierarchical pooling."
        ),
        feature_mode=V6_C32_FEATURE_MODE,
        pooling="hierarchical",
        learner="logistic_regression",
        role="matched_control",
    ),
    ModelSpec(
        experiment_id="C33R_EXACT_COMPLEMENT_HIER_LR_PCA",
        description="V6 C33 exact-complement control with hierarchical pooling.",
        feature_mode=V6_C33_FEATURE_MODE,
        pooling="hierarchical",
        learner="logistic_regression",
        role="matched_control",
    ),
    ModelSpec(
        experiment_id="C31_RMOM_EXACT_SUPPORT_ONLY_LR_PCA",
        description="C31 support-only control with exactly the RMoM pooling of A23/A24.",
        feature_mode=V6_C31_FEATURE_MODE,
        pooling="robust_median_of_means",
        learner="logistic_regression",
        role="matched_control",
    ),
    ModelSpec(
        experiment_id="C32_RMOM_SHUFFLED_INTENSITY_LR_PCA",
        description="C32 shuffled-intensity control with exactly the RMoM pooling of A23/A24.",
        feature_mode=V6_C32_FEATURE_MODE,
        pooling="robust_median_of_means",
        learner="logistic_regression",
        role="matched_control",
    ),
    ModelSpec(
        experiment_id="C33_RMOM_EXACT_COMPLEMENT_LR_PCA",
        description="C33 exact-complement control with exactly the RMoM pooling of A23/A24.",
        feature_mode=V6_C33_FEATURE_MODE,
        pooling="robust_median_of_means",
        learner="logistic_regression",
        role="matched_control",
    ),
    ModelSpec(
        experiment_id="C34_META_NUISANCE_ONLY_LR",
        description=(
            "Patient classifier that receives only the conservative V7 nuisance "
            "matrix. It quantifies residual label predictability from protocol/export metadata."
        ),
        feature_mode="v7_nuisance_only",
        pooling="patient_tabular",
        learner="logistic_regression",
        role="negative_control",
    ),
)


# Each V7 candidate is paired with controls that use the identical pooling and
# learner. This prevents an apparent anatomical gap from being caused by a
# different classifier or a different nuisance-removal path.
V7_ALGORITHM_MATCHED_CONTROL_SPECS = (
    ModelSpec(
        "C31_A21_FLNO_META_HIER_LR",
        "C31 exact support with A21's hierarchical FLNO + LR path.",
        V6_C31_FEATURE_MODE, "hierarchical", "flno_logistic_regression",
        "matched_control", True, False, True,
    ),
    ModelSpec(
        "C32_A21_FLNO_META_HIER_LR",
        "C32 shuffled intensity with A21's hierarchical FLNO + LR path.",
        V6_C32_FEATURE_MODE, "hierarchical", "flno_logistic_regression",
        "matched_control", True, False, True,
    ),
    ModelSpec(
        "C33_A21_FLNO_META_HIER_LR",
        "C33 exact complement with A21's hierarchical FLNO + LR path.",
        V6_C33_FEATURE_MODE, "hierarchical", "flno_logistic_regression",
        "matched_control", True, False, True,
    ),
    ModelSpec(
        "C31_A22_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C31 exact support with A22's hierarchical FLNO + CMP-AUC path.",
        V6_C31_FEATURE_MODE, "hierarchical", "flno_pairwise_auc",
        "matched_control", True, True, True,
    ),
    ModelSpec(
        "C32_A22_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C32 shuffled intensity with A22's hierarchical FLNO + CMP-AUC path.",
        V6_C32_FEATURE_MODE, "hierarchical", "flno_pairwise_auc",
        "matched_control", True, True, True,
    ),
    ModelSpec(
        "C33_A22_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C33 exact complement with A22's hierarchical FLNO + CMP-AUC path.",
        V6_C33_FEATURE_MODE, "hierarchical", "flno_pairwise_auc",
        "matched_control", True, True, True,
    ),
    ModelSpec(
        "C31_A23_RMOM_FLNO_META_LR",
        "C31 exact support with A23's RMoM + FLNO + LR path.",
        V6_C31_FEATURE_MODE, "robust_median_of_means",
        "flno_logistic_regression", "matched_control", True, False, True,
    ),
    ModelSpec(
        "C32_A23_RMOM_FLNO_META_LR",
        "C32 shuffled intensity with A23's RMoM + FLNO + LR path.",
        V6_C32_FEATURE_MODE, "robust_median_of_means",
        "flno_logistic_regression", "matched_control", True, False, True,
    ),
    ModelSpec(
        "C33_A23_RMOM_FLNO_META_LR",
        "C33 exact complement with A23's RMoM + FLNO + LR path.",
        V6_C33_FEATURE_MODE, "robust_median_of_means",
        "flno_logistic_regression", "matched_control", True, False, True,
    ),
    ModelSpec(
        "C31_A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C31 exact support with A24's RMoM + FLNO + CMP-AUC path.",
        V6_C31_FEATURE_MODE, "robust_median_of_means",
        "flno_pairwise_auc", "matched_control", True, True, True,
    ),
    ModelSpec(
        "C32_A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C32 shuffled intensity with A24's RMoM + FLNO + CMP-AUC path.",
        V6_C32_FEATURE_MODE, "robust_median_of_means",
        "flno_pairwise_auc", "matched_control", True, True, True,
    ),
    ModelSpec(
        "C33_A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C33 exact complement with A24's RMoM + FLNO + CMP-AUC path.",
        V6_C33_FEATURE_MODE, "robust_median_of_means",
        "flno_pairwise_auc", "matched_control", True, True, True,
    ),
)

MODEL_SPECS = MODEL_SPECS + V7_ALGORITHM_MATCHED_CONTROL_SPECS

V7_CANDIDATE_IDS = (
    "A17R_V6_REPRODUCED_HIER_LR_PCA",
    "A21_FLNO_META_HIER_LR",
    "A22_FLNO_META_MATCHED_PAIRWISE_AUC",
    "A23_RMOM_FLNO_META_LR",
    "A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
)
V7_PROSPECTIVE_CHALLENGER_ID = "A22_FLNO_META_MATCHED_PAIRWISE_AUC"
V7_CANDIDATE_SELECTION_PRIORITY = (
    V7_PROSPECTIVE_CHALLENGER_ID,
    "A21_FLNO_META_HIER_LR",
    "A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
    "A23_RMOM_FLNO_META_LR",
    "A17R_V6_REPRODUCED_HIER_LR_PCA",
)
# Model identity is selected only from outer-training inner-OOF DVU. Candidates
# within DVU_SELECTION_TOLERANCE are resolved by this immutable priority order,
# which gives the prospectively declared A22 challenger first priority.

# Predeclared candidate-to-control mapping for the final anatomical gap.
MATCHED_CONTROL_IDS_BY_CANDIDATE = {
    "A17R_V6_REPRODUCED_HIER_LR_PCA": (
        "C31R_EXACT_SUPPORT_ONLY_HIER_LR_PCA",
        "C32R_SHUFFLED_INTENSITY_HIER_LR_PCA",
        "C33R_EXACT_COMPLEMENT_HIER_LR_PCA",
    ),
    "A21_FLNO_META_HIER_LR": (
        "C31_A21_FLNO_META_HIER_LR",
        "C32_A21_FLNO_META_HIER_LR",
        "C33_A21_FLNO_META_HIER_LR",
    ),
    "A22_FLNO_META_MATCHED_PAIRWISE_AUC": (
        "C31_A22_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C32_A22_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C33_A22_FLNO_META_MATCHED_PAIRWISE_AUC",
    ),
    "A23_RMOM_FLNO_META_LR": (
        "C31_A23_RMOM_FLNO_META_LR",
        "C32_A23_RMOM_FLNO_META_LR",
        "C33_A23_RMOM_FLNO_META_LR",
    ),
    "A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC": (
        "C31_A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C32_A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
        "C33_A24_RMOM_FLNO_META_MATCHED_PAIRWISE_AUC",
    ),
}



# ================================================================================
# SMALL UTILITIES
# ================================================================================


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {seconds:.1f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {seconds:.1f} s"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compute_balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Balanced weights require both classes.")
    return len(labels) / (2.0 * counts[labels])


def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def select_youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    statistic = tpr - fpr
    best = np.max(statistic)
    candidates = np.flatnonzero(np.isclose(statistic, best, atol=1e-12))
    return float(np.max(thresholds[candidates]))


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    def divide(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(np.mean(labels == predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def stratified_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    replicates: int,
    random_state: int,
) -> dict[str, list[float | None]]:
    labels = np.asarray(labels, dtype=np.int64)
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(replicates)):
        sample = np.concatenate(
            [
                rng.choice(class0, len(class0), replace=True),
                rng.choice(class1, len(class1), replace=True),
            ]
        )
        metrics = binary_metrics(
            labels[sample], probabilities[sample], predictions[sample]
        )
        for key, value in metrics.items():
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                values[key].append(float(value))
    intervals: dict[str, list[float | None]] = {}
    for key in (
        "auc",
        "auprc",
        "brier_score",
        "accuracy",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "ppv",
        "npv",
        "f1",
    ):
        if not values[key]:
            intervals[key] = [None, None]
        else:
            array = np.asarray(values[key], dtype=np.float64)
            intervals[key] = [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ]
    return intervals


def average_rank(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic handling of ties, without SciPy."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def safe_abs_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    xr = average_rank(x)
    yr = average_rank(y)
    correlation = np.corrcoef(xr, yr)[0, 1]
    return float(abs(correlation)) if np.isfinite(correlation) else 0.0


def maximum_within_class_nuisance_correlation(
    probabilities: np.ndarray,
    nuisance_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Maximum score/nuisance Spearman correlation calculated within class."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    nuisance_matrix = np.asarray(nuisance_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    class_values: dict[str, float] = {}
    for label in (0, 1):
        mask = labels == label
        correlations = [
            safe_abs_spearman(probabilities[mask], nuisance_matrix[mask, index])
            for index in range(nuisance_matrix.shape[1])
        ]
        class_values[str(label)] = float(max(correlations, default=0.0))
    return float(max(class_values.values(), default=0.0)), class_values


def nuisance_diagnostic_coordinates(Z: np.ndarray) -> np.ndarray:
    """Compress nuisance metadata to a small label-blind diagnostic subspace.

    Taking the maximum over more than one hundred raw correlations would be
    inflated by redundancy and small-sample multiple comparisons. DVU therefore
    evaluates score dependence against at most five nuisance PCA coordinates.
    During hyperparameter selection this projector is fitted only on the current
    outer-training cohort.
    """

    return TruncatedPCA(
        NUISANCE_PCA_EXPLAINED_VARIANCE,
        NUISANCE_PCA_MAX_COMPONENTS,
    ).fit(Z).transform(Z)


# ================================================================================
# V6 CACHE DISCOVERY AND LOADING
# ================================================================================


def cache_has_required_contract(cache_dir: Path) -> tuple[bool, dict[str, Any] | None]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        return False, None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    completed_modes = set(metadata.get("completed_modes", []))
    if not set(REQUIRED_V6_FEATURE_MODES).issubset(completed_modes):
        return False, metadata
    required_shared = (
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
    if not all((cache_dir / name).is_file() for name in required_shared):
        return False, metadata
    if not all(
        (cache_dir / f"features__{mode}.npy").is_file()
        for mode in REQUIRED_V6_FEATURE_MODES
    ):
        return False, metadata
    return True, metadata


def _iter_candidate_v6_cache_dirs(root: Path, recursive: bool) -> list[Path]:
    """Return plausible cache directories below ``root`` without duplicates.

    A Kaggle notebook output attached as an input is mounted under an arbitrary
    dataset slug, for example::

        /kaggle/input/<slug>/cad_patient_pipeline_outputs/
            feature_bank_cache/<fingerprint>/metadata.json

    The original V7 implementation searched only ``/kaggle/working`` and
    therefore could not reuse a cache from a previous Kaggle session. This
    helper accepts either the exact fingerprint directory, the feature_bank_cache
    directory, a suite/output directory, or any ancestor mounted under
    ``/kaggle/input``.
    """

    root = Path(root)
    if root.is_file():
        if root.name == "metadata.json":
            return [root.parent]
        return []
    if not root.is_dir():
        return []

    found: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path):
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            found.append(candidate)

    # Exact cache directory.
    if (root / "metadata.json").is_file():
        add(root)

    # Common direct layouts.
    for pattern in (
        "*/metadata.json",
        "feature_bank_cache/*/metadata.json",
        "cad_patient_pipeline_outputs/feature_bank_cache/*/metadata.json",
    ):
        try:
            for metadata_path in root.glob(pattern):
                add(metadata_path.parent)
        except OSError:
            pass

    if recursive:
        # Restrict the recursive walk to metadata files and validate each parent
        # with the exact V6 contract; unrelated metadata.json files are harmless.
        try:
            for metadata_path in root.rglob("metadata.json"):
                # Fast path filter avoids reading most unrelated JSON files.
                parent_parts = set(metadata_path.parent.parts)
                if (
                    "feature_bank_cache" not in parent_parts
                    and not (metadata_path.parent / "labels.npy").is_file()
                ):
                    continue
                add(metadata_path.parent)
        except OSError:
            pass

    return found


def _compatible_v6_caches_under(
    root: Path,
    recursive: bool,
    priority: int,
) -> list[tuple[int, float, Path, dict[str, Any]]]:
    rows: list[tuple[int, float, Path, dict[str, Any]]] = []
    for cache_dir in _iter_candidate_v6_cache_dirs(root, recursive=recursive):
        valid, metadata = cache_has_required_contract(cache_dir)
        if not valid or metadata is None:
            continue
        try:
            mtime = float((cache_dir / "metadata.json").stat().st_mtime)
        except OSError:
            mtime = 0.0
        rows.append((int(priority), mtime, cache_dir, metadata))
    return rows


def discover_v6_feature_cache() -> tuple[Path, dict[str, Any]]:
    """Locate a compatible V6 cache in working storage or attached Kaggle inputs.

    Search order is intentionally broad but contract-strict. The cache is not
    accepted merely because its directory is called ``feature_bank_cache``; it
    must contain every V7-required V6 feature mode and every row-aligned shared
    array. This lets a fresh Kaggle session reuse a previous notebook output
    attached under ``/kaggle/input`` without requiring the user to know the
    fingerprint directory in advance.
    """

    searched_roots: list[str] = []
    candidates: list[tuple[int, float, Path, dict[str, Any]]] = []

    # Explicit override is flexible: exact cache directory OR any ancestor.
    if V6_FEATURE_CACHE_DIR_OVERRIDE:
        override = Path(V6_FEATURE_CACHE_DIR_OVERRIDE)
        searched_roots.append(str(override))
        candidates.extend(
            _compatible_v6_caches_under(
                override,
                recursive=True,
                priority=1000,
            )
        )
        if not candidates:
            raise FileNotFoundError(
                "CAD_V6_FEATURE_CACHE_DIR was provided, but no compatible V6 "
                "A17/C31/C32/C33 cache was found at or below: "
                f"{override}. The value may point to the exact fingerprint "
                "folder OR to an ancestor such as feature_bank_cache/."
            )

    # Optional additional roots supplied by the user. These are searched before
    # automatic locations but after an explicit cache-dir override.
    if V6_SEARCH_ROOTS_OVERRIDE:
        for raw_root in V6_SEARCH_ROOTS_OVERRIDE.split(os.pathsep):
            raw_root = raw_root.strip()
            if not raw_root:
                continue
            root = Path(raw_root)
            searched_roots.append(str(root))
            candidates.extend(
                _compatible_v6_caches_under(
                    root,
                    recursive=True,
                    priority=900,
                )
            )

    automatic_roots: list[tuple[Path, bool, int]] = []
    automatic_roots.append((V6_OUTPUT_ROOT, False, 800))

    if os.path.exists("/kaggle/working"):
        # Handles a V6 run earlier in the same live Kaggle session, even if the
        # output root was changed from the default.
        automatic_roots.append((Path("/kaggle/working"), True, 700))

    if os.path.exists("/kaggle/input"):
        # Critical fix: prior Kaggle notebook/dataset outputs live here in a new
        # session. Search recursively because the dataset slug is arbitrary.
        automatic_roots.append((Path("/kaggle/input"), True, 600))

    # Local/notebook fallbacks.
    automatic_roots.append((Path.cwd(), True, 500))
    try:
        automatic_roots.append((Path(__file__).resolve().parent, True, 500))
    except NameError:
        pass

    seen_roots: set[str] = set()
    for root, recursive, priority in automatic_roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        searched_roots.append(str(root))
        candidates.extend(
            _compatible_v6_caches_under(
                root,
                recursive=recursive,
                priority=priority,
            )
        )

    if not candidates:
        roots_text = "\n  - ".join(searched_roots) if searched_roots else "(none)"
        raise FileNotFoundError(
            "No compatible V6 feature cache was found. V7 needs the frozen V6 "
            "A17/C31/C32/C33 embeddings; those arrays cannot be reconstructed "
            "from the V7 mathematics alone.\n\nSearched roots:\n  - "
            f"{roots_text}\n\nIn a fresh Kaggle session, attach the OUTPUT of "
            "the completed V6 notebook as a Kaggle input. This fixed V7 version "
            "will discover feature_bank_cache/<fingerprint> recursively under "
            "/kaggle/input automatically. Alternatively set "
            "CAD_V6_FEATURE_CACHE_DIR to either the exact fingerprint directory "
            "or any ancestor containing feature_bank_cache."
        )

    # De-duplicate the same physical cache discovered through overlapping roots.
    unique: dict[str, tuple[int, float, Path, dict[str, Any]]] = {}
    for row in candidates:
        priority, mtime, cache_dir, metadata = row
        try:
            key = str(cache_dir.resolve())
        except OSError:
            key = str(cache_dir)
        previous = unique.get(key)
        if previous is None or (priority, mtime) > (previous[0], previous[1]):
            unique[key] = row

    candidates = list(unique.values())
    # Prefer explicit/search-root matches, then same-session working caches, then
    # attached read-only inputs. Within the same priority use newest metadata.
    candidates.sort(
        key=lambda row: (row[0], row[1], str(row[2])),
        reverse=True,
    )
    _, _, cache_dir, metadata = candidates[0]
    return cache_dir, metadata


def discover_v6_suite_dir(patient_ids: Sequence[str]) -> Path | None:
    if V6_SUITE_DIR_OVERRIDE:
        suite_dir = Path(V6_SUITE_DIR_OVERRIDE)
        manifest = suite_dir / "manifests" / "patient_fold_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"CAD_V6_SUITE_DIR has no patient fold manifest: {suite_dir}"
            )
        return suite_dir

    expected = set(map(str, patient_ids))
    candidates: list[tuple[float, Path]] = []
    if V6_OUTPUT_ROOT.is_dir():
        for manifest in V6_OUTPUT_ROOT.glob(
            "multi_experiment_suite__*/manifests/patient_fold_manifest.csv"
        ):
            suite_dir = manifest.parents[1]
            try:
                with manifest.open("r", encoding="utf-8", newline="") as file:
                    rows = list(csv.DictReader(file))
                found = {str(row["patient_id"]) for row in rows}
            except (OSError, KeyError, csv.Error):
                continue
            if found != expected:
                continue

            # Prefer a suite whose saved configuration explicitly contains the
            # V6 A17/C31/C32/C33 family. Older suites have the same 30 patients
            # and therefore cannot be identified from the manifest alone.
            configuration_path = suite_dir / "suite_configuration.json"
            v6_contract_confirmed = False
            if configuration_path.is_file():
                try:
                    configuration_text = configuration_path.read_text(
                        encoding="utf-8"
                    )
                    v6_contract_confirmed = all(
                        token in configuration_text
                        for token in (
                            "A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA",
                            "C31_A17_EXACT_SUPPORT_MASK_ONLY_HIER_LR_PCA",
                            "C32_A17_SUPPORT_INTENSITY_AFFINE_SHUFFLED_HIER_LR_PCA",
                            "C33_A17_EXACT_SUPPORT_COMPLEMENT_REGION_NORM_HIER_LR_PCA",
                        )
                    )
                except OSError:
                    pass
            priority_time = manifest.stat().st_mtime + (1e12 if v6_contract_confirmed else 0.0)
            candidates.append((priority_time, suite_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], str(row[1])), reverse=True)
    return candidates[0][1]


def load_array(cache_dir: Path, name: str) -> np.ndarray:
    return np.load(cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def load_v6_bank(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    bank = {
        "cache_dir": cache_dir,
        "metadata": metadata,
        "labels": load_array(cache_dir, "labels"),
        "patient_ids": load_array(cache_dir, "patient_ids").astype(str),
        "series_ids": load_array(cache_dir, "series_ids").astype(str),
        "sample_indices": load_array(cache_dir, "sample_indices"),
        "provenance_features": load_array(cache_dir, "provenance_features"),
        "standardization_features": load_array(cache_dir, "standardization_features"),
        "standardized_monai_valid": load_array(cache_dir, "standardized_monai_valid"),
        "standardized_area_ratios": load_array(cache_dir, "standardized_area_ratios"),
        "standardized_peak_probabilities": load_array(
            cache_dir, "standardized_peak_probabilities"
        ),
        "standardized_mean_foreground_probabilities": load_array(
            cache_dir, "standardized_mean_foreground_probabilities"
        ),
        "features": {
            mode: np.load(
                cache_dir / f"features__{mode}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for mode in REQUIRED_V6_FEATURE_MODES
        },
    }
    n_rows = len(bank["labels"])
    aligned_names = (
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
    for name in aligned_names:
        if len(bank[name]) != n_rows:
            raise RuntimeError(f"Cache array {name!r} is not row-aligned.")
    for mode, features in bank["features"].items():
        if features.shape[0] != n_rows:
            raise RuntimeError(f"Feature mode {mode!r} is not row-aligned.")
        if not np.all(np.isfinite(np.asarray(features[: min(n_rows, 100)]))):
            raise RuntimeError(f"Feature mode {mode!r} contains non-finite values.")
    return bank


def load_or_create_outer_manifest(
    suite_dir: Path | None,
    patient_ids: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    if suite_dir is not None:
        manifest_path = suite_dir / "manifests" / "patient_fold_manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "patient_id": str(row["patient_id"]),
                    "true_label": int(row["true_label"]),
                    "outer_fold": int(row["outer_fold"]),
                    "duplicate_component_id": str(
                        row.get("duplicate_component_id", row["patient_id"])
                    ),
                }
            )
        if {row["patient_id"] for row in normalized} != set(patient_ids.tolist()):
            raise RuntimeError("V6 fold manifest patient set does not match cache.")
        return normalized

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    folds = np.zeros(len(patient_ids), dtype=np.int64)
    for fold_index, (_, valid_indices) in enumerate(
        splitter.split(patient_ids, labels), start=1
    ):
        folds[valid_indices] = fold_index
    return [
        {
            "patient_id": str(patient_id),
            "true_label": int(label),
            "outer_fold": int(fold),
            "duplicate_component_id": str(patient_id),
        }
        for patient_id, label, fold in zip(patient_ids, labels, folds)
    ]


# ================================================================================
# PATIENT POOLING
# ================================================================================


def geometric_median(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Geometric median requires a non-empty 2D array.")
    if len(points) == 1:
        return points[0].astype(np.float32)
    estimate = np.mean(points, axis=0)
    for _ in range(GEOMETRIC_MEDIAN_MAX_ITERATIONS):
        distances = np.linalg.norm(points - estimate, axis=1)
        close = np.flatnonzero(distances <= GEOMETRIC_MEDIAN_EPSILON)
        if len(close):
            return points[int(close[0])].astype(np.float32)
        inverse = 1.0 / np.maximum(distances, GEOMETRIC_MEDIAN_EPSILON)
        updated = np.sum(points * inverse[:, None], axis=0) / np.sum(inverse)
        if np.linalg.norm(updated - estimate) <= GEOMETRIC_MEDIAN_TOLERANCE:
            estimate = updated
            break
        estimate = updated
    return estimate.astype(np.float32)


def series_rmom_embedding(
    features: np.ndarray,
    row_indices: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    stable_order = np.argsort(sample_indices[row_indices], kind="mergesort")
    ordered = row_indices[stable_order]
    block_count = min(RMOM_BLOCKS_PER_SERIES, len(ordered))
    blocks = [block for block in np.array_split(ordered, block_count) if len(block)]
    block_means = np.stack(
        [np.mean(np.asarray(features[block]), axis=0) for block in blocks], axis=0
    )
    return geometric_median(block_means)


def pool_feature_mode(
    features: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    series_ids: np.ndarray,
    sample_indices: np.ndarray,
    pooling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids).astype(str)
    series_ids = np.asarray(series_ids).astype(str)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if pooling not in {"hierarchical", "robust_median_of_means"}:
        raise ValueError(f"Unsupported image pooling strategy: {pooling!r}.")

    patient_to_label: dict[str, int] = {}
    patient_to_series: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (label, patient_id, series_id) in enumerate(
        zip(labels, patient_ids, series_ids)
    ):
        previous = patient_to_label.get(patient_id)
        if previous is not None and previous != int(label):
            raise RuntimeError(f"Patient {patient_id} has inconsistent labels.")
        patient_to_label[patient_id] = int(label)
        patient_to_series[patient_id][series_id].append(index)

    ordered_patients = np.asarray(sorted(patient_to_label))
    pooled: list[np.ndarray] = []
    pooled_labels: list[int] = []
    for patient_id in ordered_patients:
        series_embeddings = []
        for series_id in sorted(patient_to_series[patient_id]):
            indices = np.asarray(patient_to_series[patient_id][series_id], dtype=np.int64)
            if pooling == "hierarchical":
                embedding = np.mean(np.asarray(features[indices]), axis=0)
            else:
                embedding = series_rmom_embedding(features, indices, sample_indices)
            series_embeddings.append(np.asarray(embedding, dtype=np.float32))
        patient_embedding = np.mean(np.stack(series_embeddings, axis=0), axis=0)
        pooled.append(patient_embedding.astype(np.float32))
        pooled_labels.append(patient_to_label[patient_id])

    X = np.stack(pooled, axis=0)
    y = np.asarray(pooled_labels, dtype=np.int64)
    if not np.all(np.isfinite(X)):
        raise RuntimeError(f"Pooling strategy {pooling!r} produced non-finite values.")
    return X, y, ordered_patients


# ================================================================================
# NUISANCE MATRIX
# ================================================================================


def robust_patient_summaries(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.concatenate(
        [
            np.mean(values, axis=0),
            np.std(values, axis=0),
            np.quantile(values, 0.10, axis=0),
            np.quantile(values, 0.50, axis=0),
            np.quantile(values, 0.90, axis=0),
        ]
    )


def build_patient_nuisance_matrix(
    bank: dict[str, Any],
    ordered_patients: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    metadata = bank["metadata"]
    provenance_names = tuple(metadata.get("provenance_feature_names", []))
    standardization_names = tuple(
        metadata.get("standardization_feature_names", [])
    )
    if not provenance_names or not standardization_names:
        raise RuntimeError(
            "V6 metadata must contain provenance and standardization feature names."
        )

    def indices_for(names: Sequence[str], available: Sequence[str]) -> list[int]:
        missing = [name for name in names if name not in available]
        if missing:
            raise RuntimeError(f"V6 metadata lacks nuisance features: {missing}.")
        return [available.index(name) for name in names]

    provenance_indices = indices_for(
        PROVENANCE_NUISANCE_FEATURE_NAMES, provenance_names
    )
    standardization_indices = indices_for(
        STANDARDIZATION_NUISANCE_FEATURE_NAMES, standardization_names
    )
    slice_patient_ids = np.asarray(bank["patient_ids"]).astype(str)
    slice_series_ids = np.asarray(bank["series_ids"]).astype(str)
    slice_labels = np.asarray(bank["labels"], dtype=np.int64)

    summary_suffixes = ("mean", "std", "q10", "median", "q90")
    names: list[str] = []
    for source, selected_names in (
        ("prov", PROVENANCE_NUISANCE_FEATURE_NAMES),
        ("std", STANDARDIZATION_NUISANCE_FEATURE_NAMES),
    ):
        for suffix in summary_suffixes:
            names.extend(f"{source}__{name}__{suffix}" for name in selected_names)
    names.extend(
        [
            "count__n_slices",
            "count__n_series",
            "series_length__mean",
            "series_length__std",
            "series_length__median",
            "series_length__maximum",
            "series_length__minimum",
        ]
    )
    if INCLUDE_MONAI_QC_IN_PRIMARY_FLNO:
        names.extend(
            [
                "monai_qc__valid_rate",
                "monai_qc__area_mean",
                "monai_qc__area_std",
                "monai_qc__peak_mean",
                "monai_qc__peak_std",
                "monai_qc__foreground_mean",
                "monai_qc__foreground_std",
            ]
        )

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for patient_id in ordered_patients:
        indices = np.flatnonzero(slice_patient_ids == str(patient_id))
        patient_labels = np.unique(slice_labels[indices])
        if len(patient_labels) != 1:
            raise RuntimeError(f"Patient {patient_id} has inconsistent labels.")
        provenance = np.asarray(bank["provenance_features"][indices])[
            :, provenance_indices
        ]
        standardization = np.asarray(bank["standardization_features"][indices])[
            :, standardization_indices
        ]
        values = [
            robust_patient_summaries(provenance),
            robust_patient_summaries(standardization),
        ]
        patient_series = slice_series_ids[indices]
        _, series_counts = np.unique(patient_series, return_counts=True)
        values.append(
            np.asarray(
                [
                    len(indices),
                    len(series_counts),
                    np.mean(series_counts),
                    np.std(series_counts),
                    np.median(series_counts),
                    np.max(series_counts),
                    np.min(series_counts),
                ],
                dtype=np.float64,
            )
        )
        if INCLUDE_MONAI_QC_IN_PRIMARY_FLNO:
            valid = np.asarray(bank["standardized_monai_valid"][indices], dtype=float)
            area = np.asarray(bank["standardized_area_ratios"][indices], dtype=float)
            peak = np.asarray(
                bank["standardized_peak_probabilities"][indices], dtype=float
            )
            foreground = np.asarray(
                bank["standardized_mean_foreground_probabilities"][indices],
                dtype=float,
            )
            values.append(
                np.asarray(
                    [
                        np.mean(valid),
                        np.mean(area),
                        np.std(area),
                        np.mean(peak),
                        np.std(peak),
                        np.mean(foreground),
                        np.std(foreground),
                    ],
                    dtype=np.float64,
                )
            )
        rows.append(np.concatenate(values))
        labels.append(int(patient_labels[0]))

    Z = np.stack(rows, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    # Keep the full finite metadata matrix here. Constant-column removal and all
    # PCA decisions are fitted separately inside each training fold.
    if not np.all(np.isfinite(Z)):
        raise RuntimeError("The V7 nuisance matrix contains non-finite values.")
    return Z.astype(np.float64), tuple(names), y


# ================================================================================
# FOLD-LOCAL TRANSFORMERS AND LEARNERS
# ================================================================================


class ExactVarianceProjector:
    """Reproduce V6 StandardScaler + PCA(n_components=0.95) exactly.

    V6 retains every embedding coordinate, including any constant coordinate;
    StandardScaler maps constants to zero and PCA then ignores them naturally.
    Avoiding a separate variance filter keeps this reference path numerically as
    close as possible to the saved V6 A17 implementation.
    """

    def __init__(self, variance_target: float):
        self.variance_target = float(variance_target)
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "ExactVarianceProjector":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("ExactVarianceProjector requires a 2D training matrix.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Training features contain non-finite values.")
        self.scaler = StandardScaler().fit(X)
        standardized = self.scaler.transform(X)
        self.pca = PCA(
            n_components=self.variance_target,
            svd_solver="full",
        ).fit(standardized)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise RuntimeError("ExactVarianceProjector is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError("Transformed features contain non-finite values.")
        return self.pca.transform(self.scaler.transform(X))


class TruncatedPCA:
    """Fold-local StandardScaler + variance-target PCA with a component cap."""

    def __init__(self, variance_target: float, maximum_components: int):
        self.variance_target = float(variance_target)
        self.maximum_components = int(maximum_components)
        self.keep_columns_: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.n_components_: int | None = None

    def fit(self, X: np.ndarray) -> "TruncatedPCA":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("TruncatedPCA requires a 2D training matrix.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Training features contain non-finite values.")
        # Constant-column removal is fitted only on the current training fold.
        # Validation rows therefore cannot determine which nuisance variables are
        # retained by FLNO.
        keep = np.std(X, axis=0) > 1e-12
        if not np.any(keep):
            raise ValueError("Training features contain no variable columns.")
        self.keep_columns_ = keep
        retained = X[:, keep]
        self.scaler = StandardScaler().fit(retained)
        standardized = self.scaler.transform(retained)
        maximum = min(
            self.maximum_components,
            standardized.shape[1],
            max(1, standardized.shape[0] - 1),
        )
        full = PCA(n_components=maximum, svd_solver="full")
        full.fit(standardized)
        cumulative = np.cumsum(full.explained_variance_ratio_)
        selected = int(np.searchsorted(cumulative, self.variance_target) + 1)
        selected = max(1, min(selected, maximum))
        self.pca = full
        self.n_components_ = selected
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if (
            self.keep_columns_ is None
            or self.scaler is None
            or self.pca is None
            or self.n_components_ is None
        ):
            raise RuntimeError("TruncatedPCA is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        retained = X[:, self.keep_columns_]
        if not np.all(np.isfinite(retained)):
            raise ValueError("Transformed features contain non-finite values.")
        standardized = self.scaler.transform(retained)
        return self.pca.transform(standardized)[:, : self.n_components_]


class FLNOTransformer:
    """Fold-local nuisance orthogonalization of a patient embedding.

    Let U be a low-dimensional signal representation and V a low-dimensional
    nuisance representation. Ridge regression estimates B in V B ≈ U using only
    the current training fold. The cleaned representation is U - alpha V B.
    """

    def __init__(self, alpha: float, ridge: float):
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.signal_projector = TruncatedPCA(
            PATIENT_PCA_EXPLAINED_VARIANCE, PATIENT_PCA_MAX_COMPONENTS
        )
        self.nuisance_projector = TruncatedPCA(
            NUISANCE_PCA_EXPLAINED_VARIANCE, NUISANCE_PCA_MAX_COMPONENTS
        )
        self.coefficients_: np.ndarray | None = None
        self.cleaned_scaler_: StandardScaler | None = None

    def fit(self, X: np.ndarray, Z: np.ndarray) -> "FLNOTransformer":
        U = self.signal_projector.fit(X).transform(X)
        V = self.nuisance_projector.fit(Z).transform(Z)
        gram = V.T @ V
        penalty = self.ridge * np.eye(gram.shape[0], dtype=np.float64)
        self.coefficients_ = np.linalg.solve(gram + penalty, V.T @ U)
        cleaned = U - self.alpha * (V @ self.coefficients_)
        self.cleaned_scaler_ = StandardScaler().fit(cleaned)
        return self

    def transform(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None or self.cleaned_scaler_ is None:
            raise RuntimeError("FLNOTransformer is not fitted.")
        U = self.signal_projector.transform(X)
        V = self.nuisance_projector.transform(Z)
        cleaned = U - self.alpha * (V @ self.coefficients_)
        return self.cleaned_scaler_.transform(cleaned)

    def nuisance_coordinates(self, Z: np.ndarray) -> np.ndarray:
        return self.nuisance_projector.transform(Z)


class StandardPatientModel:
    def __init__(self, c_value: float):
        self.c_value = float(c_value)
        self.projector = ExactVarianceProjector(
            PATIENT_PCA_EXPLAINED_VARIANCE
        )
        self.classifier = LogisticRegression(
            C=self.c_value,
            solver="liblinear",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=RANDOM_SEED,
        )

    def fit(self, X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> "StandardPatientModel":
        transformed = self.projector.fit(X).transform(X)
        self.classifier.fit(
            transformed,
            y,
            sample_weight=compute_balanced_sample_weights(y),
        )
        return self

    def raw_scores(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        transformed = self.projector.transform(X)
        return self.classifier.predict_proba(transformed)[:, 1]

    @property
    def requires_calibration(self) -> bool:
        return False


class FLNOLogisticModel:
    def __init__(self, alpha: float, ridge: float, c_value: float):
        self.transformer = FLNOTransformer(alpha=alpha, ridge=ridge)
        self.classifier = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=RANDOM_SEED,
        )

    def fit(self, X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> "FLNOLogisticModel":
        transformed = self.transformer.fit(X, Z).transform(X, Z)
        self.classifier.fit(
            transformed,
            y,
            sample_weight=compute_balanced_sample_weights(y),
        )
        return self

    def raw_scores(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        transformed = self.transformer.transform(X, Z)
        return self.classifier.predict_proba(transformed)[:, 1]

    @property
    def requires_calibration(self) -> bool:
        return False


class FLNOPairwiseAUCModel:
    def __init__(
        self,
        alpha: float,
        ridge: float,
        c_value: float,
        tau_multiplier: float,
    ):
        self.transformer = FLNOTransformer(alpha=alpha, ridge=ridge)
        self.c_value = float(c_value)
        self.tau_multiplier = float(tau_multiplier)
        self.ranker = LogisticRegression(
            C=self.c_value,
            fit_intercept=False,
            solver="liblinear",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=RANDOM_SEED,
        )
        self.tau_: float | None = None
        self.n_pairs_: int | None = None

    def fit(self, X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> "FLNOPairwiseAUCModel":
        U = self.transformer.fit(X, Z).transform(X, Z)
        V = self.transformer.nuisance_coordinates(Z)
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError("Pairwise AUC fitting requires both classes.")

        differences: list[np.ndarray] = []
        distances: list[float] = []
        for pos in positive:
            for neg in negative:
                differences.append(U[pos] - U[neg])
                distances.append(float(np.linalg.norm(V[pos] - V[neg])))
        D = np.stack(differences, axis=0)
        distances_array = np.asarray(distances, dtype=np.float64)
        positive_distances = distances_array[distances_array > 1e-12]
        base_tau = (
            float(np.median(positive_distances))
            if len(positive_distances)
            else 1.0
        )
        self.tau_ = max(1e-8, base_tau * self.tau_multiplier)
        pair_weights = np.exp(
            -np.square(distances_array) / (2.0 * self.tau_ * self.tau_)
        )
        pair_weights = np.maximum(pair_weights, PAIRWISE_MIN_WEIGHT)
        pair_weights /= np.mean(pair_weights)

        # Adding both d with label 1 and -d with label 0 gives the symmetric
        # pairwise logistic ranking objective without an intercept.
        X_pairs = np.concatenate([D, -D], axis=0)
        y_pairs = np.concatenate(
            [np.ones(len(D), dtype=np.int64), np.zeros(len(D), dtype=np.int64)]
        )
        weights = np.concatenate([pair_weights, pair_weights])
        self.ranker.fit(X_pairs, y_pairs, sample_weight=weights)
        self.n_pairs_ = int(len(D))
        return self

    def raw_scores(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        U = self.transformer.transform(X, Z)
        return np.asarray(self.ranker.decision_function(U), dtype=np.float64)

    @property
    def requires_calibration(self) -> bool:
        return True


class SigmoidCalibrator:
    def __init__(self):
        self.model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=RANDOM_SEED,
        )

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        self.model.fit(np.asarray(raw_scores).reshape(-1, 1), labels)
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(raw_scores).reshape(-1, 1))[:, 1]


def create_model(spec: ModelSpec, parameters: dict[str, float]) -> Any:
    if spec.learner == "logistic_regression":
        return StandardPatientModel(c_value=parameters["c"])
    if spec.learner == "flno_logistic_regression":
        return FLNOLogisticModel(
            alpha=parameters["alpha"],
            ridge=parameters["ridge"],
            c_value=parameters["c"],
        )
    if spec.learner == "flno_pairwise_auc":
        return FLNOPairwiseAUCModel(
            alpha=parameters["alpha"],
            ridge=parameters["ridge"],
            c_value=parameters["c"],
            tau_multiplier=parameters["tau_multiplier"],
        )
    raise ValueError(f"Unsupported learner: {spec.learner!r}.")


def candidate_parameter_grid(spec: ModelSpec) -> list[dict[str, float]]:
    if spec.learner == "logistic_regression":
        return [{"c": float(c)} for c in CLASSIFIER_C_GRID]
    if spec.learner == "flno_logistic_regression":
        return [
            {"alpha": float(alpha), "ridge": float(ridge), "c": float(c)}
            for alpha in FLNO_ALPHA_GRID
            for ridge in FLNO_RIDGE_GRID
            for c in CLASSIFIER_C_GRID
        ]
    if spec.learner == "flno_pairwise_auc":
        return [
            {
                "alpha": float(alpha),
                "ridge": float(ridge),
                "c": float(c),
                "tau_multiplier": float(tau),
            }
            for alpha in FLNO_ALPHA_GRID
            for ridge in FLNO_RIDGE_GRID
            for c in PAIRWISE_C_GRID
            for tau in PAIRWISE_TAU_MULTIPLIER_GRID
        ]
    raise ValueError(f"Unsupported learner: {spec.learner!r}.")


# ================================================================================
# NESTED CV
# ================================================================================


def create_inner_folds(labels: np.ndarray, outer_fold_index: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    maximum = min(INNER_SPLITS, int(np.bincount(labels, minlength=2).min()))
    if maximum < 2:
        raise RuntimeError("Cannot create at least two stratified inner folds.")
    splitter = StratifiedKFold(
        n_splits=maximum,
        shuffle=True,
        random_state=INNER_RANDOM_STATE + 100 * int(outer_fold_index),
    )
    folds = np.zeros(len(labels), dtype=np.int64)
    dummy = np.zeros((len(labels), 1), dtype=np.float64)
    for fold_index, (_, valid) in enumerate(splitter.split(dummy, labels), start=1):
        folds[valid] = fold_index
    return folds


def inner_oof_for_parameters(
    spec: ModelSpec,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, float],
    outer_fold_index: int,
) -> dict[str, Any]:
    folds = create_inner_folds(y, outer_fold_index)
    raw_scores = np.full(len(y), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        model = create_model(spec, parameters)
        model.fit(X[train], Z[train], y[train])
        raw_scores[valid] = model.raw_scores(X[valid], Z[valid])
    if not np.all(np.isfinite(raw_scores)):
        raise RuntimeError("Inner OOF generation produced missing/non-finite scores.")

    probe_model = create_model(spec, parameters)
    # Only the model family is needed to determine calibration behavior; no data
    # from the full outer-training set are fitted here.
    if probe_model.requires_calibration:
        calibrator = SigmoidCalibrator().fit(raw_scores, y)
        probabilities = calibrator.predict(raw_scores)
    else:
        calibrator = None
        probabilities = np.clip(raw_scores, 0.0, 1.0)

    auc = float(roc_auc_score(y, probabilities))
    diagnostic_Z = nuisance_diagnostic_coordinates(Z)
    nuisance_corr, class_corr = maximum_within_class_nuisance_correlation(
        probabilities, diagnostic_Z, y
    )
    dvu = float(auc - DVU_NUISANCE_CORRELATION_PENALTY * nuisance_corr)
    return {
        "parameters": dict(parameters),
        "raw_scores": raw_scores,
        "probabilities": probabilities,
        "labels": y.copy(),
        "auc": auc,
        "within_class_max_abs_nuisance_correlation": nuisance_corr,
        "within_class_correlations": class_corr,
        "dvu": dvu,
        "calibrator": calibrator,
        "inner_splits": int(len(np.unique(folds))),
    }



def _parameter_key(parameters: dict[str, float]) -> tuple[tuple[str, float], ...]:
    """Canonical, hashable representation of one numerical parameter row."""

    return tuple(sorted((str(key), float(value)) for key, value in parameters.items()))


def _fit_pairwise_ranker_from_transformed_data(
    transformed_signal: np.ndarray,
    transformed_nuisance: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    tau_multiplier: float,
) -> LogisticRegression:
    """Fit the CMP-AUC linear ranker after FLNO has already been fitted.

    Separating this step lets the inner-CV grid reuse one expensive fold-local
    signal/nuisance PCA and ridge residualization across all C and tau choices.
    """

    transformed_signal = np.asarray(transformed_signal, dtype=np.float64)
    transformed_nuisance = np.asarray(transformed_nuisance, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Pairwise AUC fitting requires both classes.")

    differences = np.stack(
        [
            transformed_signal[pos] - transformed_signal[neg]
            for pos in positive
            for neg in negative
        ],
        axis=0,
    )
    distances = np.asarray(
        [
            np.linalg.norm(
                transformed_nuisance[pos] - transformed_nuisance[neg]
            )
            for pos in positive
            for neg in negative
        ],
        dtype=np.float64,
    )
    positive_distances = distances[distances > 1e-12]
    base_tau = (
        float(np.median(positive_distances))
        if len(positive_distances)
        else 1.0
    )
    tau = max(1e-8, base_tau * float(tau_multiplier))
    pair_weights = np.exp(-np.square(distances) / (2.0 * tau * tau))
    pair_weights = np.maximum(pair_weights, PAIRWISE_MIN_WEIGHT)
    pair_weights /= np.mean(pair_weights)

    pair_features = np.concatenate([differences, -differences], axis=0)
    pair_labels = np.concatenate(
        [
            np.ones(len(differences), dtype=np.int64),
            np.zeros(len(differences), dtype=np.int64),
        ]
    )
    pair_sample_weights = np.concatenate([pair_weights, pair_weights])
    ranker = LogisticRegression(
        C=float(c_value),
        fit_intercept=False,
        solver="liblinear",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_SEED,
    )
    ranker.fit(
        pair_features,
        pair_labels,
        sample_weight=pair_sample_weights,
    )
    return ranker


def evaluate_inner_parameter_grid(
    spec: ModelSpec,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    outer_fold_index: int,
) -> list[dict[str, Any]]:
    """Generate every inner-OOF parameter result with shared transformations.

    The unoptimized formulation fitted the same PCA and FLNO residualizer once
    for every C/tau combination. This equivalent implementation fits them once
    per inner fold and (alpha, ridge) pair, then reuses transformed arrays. That
    substantially reduces V7 runtime without sharing information across folds.
    """

    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    parameters = candidate_parameter_grid(spec)
    folds = create_inner_folds(y, outer_fold_index)
    raw_by_key = {
        _parameter_key(row): np.full(len(y), np.nan, dtype=np.float64)
        for row in parameters
    }

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold

        if spec.learner == "logistic_regression":
            projector = ExactVarianceProjector(
                PATIENT_PCA_EXPLAINED_VARIANCE
            ).fit(X[train])
            train_signal = projector.transform(X[train])
            valid_signal = projector.transform(X[valid])
            sample_weights = compute_balanced_sample_weights(y[train])
            for row in parameters:
                classifier = LogisticRegression(
                    C=float(row["c"]),
                    solver="liblinear",
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=RANDOM_SEED,
                )
                classifier.fit(
                    train_signal,
                    y[train],
                    sample_weight=sample_weights,
                )
                raw_by_key[_parameter_key(row)][valid] = (
                    classifier.predict_proba(valid_signal)[:, 1]
                )
            continue

        alpha_ridge_pairs = sorted(
            {
                (float(row["alpha"]), float(row["ridge"]))
                for row in parameters
            }
        )
        for alpha, ridge in alpha_ridge_pairs:
            transformer = FLNOTransformer(alpha=alpha, ridge=ridge).fit(
                X[train], Z[train]
            )
            train_signal = transformer.transform(X[train], Z[train])
            valid_signal = transformer.transform(X[valid], Z[valid])

            matching_rows = [
                row
                for row in parameters
                if float(row["alpha"]) == alpha
                and float(row["ridge"]) == ridge
            ]
            if spec.learner == "flno_logistic_regression":
                sample_weights = compute_balanced_sample_weights(y[train])
                for row in matching_rows:
                    classifier = LogisticRegression(
                        C=float(row["c"]),
                        solver="liblinear",
                        max_iter=LOGISTIC_MAX_ITER,
                        random_state=RANDOM_SEED,
                    )
                    classifier.fit(
                        train_signal,
                        y[train],
                        sample_weight=sample_weights,
                    )
                    raw_by_key[_parameter_key(row)][valid] = (
                        classifier.predict_proba(valid_signal)[:, 1]
                    )
            elif spec.learner == "flno_pairwise_auc":
                train_nuisance = transformer.nuisance_coordinates(Z[train])
                for row in matching_rows:
                    ranker = _fit_pairwise_ranker_from_transformed_data(
                        train_signal,
                        train_nuisance,
                        y[train],
                        c_value=float(row["c"]),
                        tau_multiplier=float(row["tau_multiplier"]),
                    )
                    raw_by_key[_parameter_key(row)][valid] = np.asarray(
                        ranker.decision_function(valid_signal),
                        dtype=np.float64,
                    )
            else:
                raise ValueError(f"Unsupported learner: {spec.learner!r}.")

    diagnostic_Z = nuisance_diagnostic_coordinates(Z)
    results: list[dict[str, Any]] = []
    for row in parameters:
        raw_scores = raw_by_key[_parameter_key(row)]
        if not np.all(np.isfinite(raw_scores)):
            raise RuntimeError(
                "Optimized inner OOF generation produced missing/non-finite scores."
            )
        if spec.use_pairwise_auc:
            calibrator = SigmoidCalibrator().fit(raw_scores, y)
            probabilities = calibrator.predict(raw_scores)
        else:
            calibrator = None
            probabilities = np.clip(raw_scores, 0.0, 1.0)
        auc = float(roc_auc_score(y, probabilities))
        nuisance_corr, class_corr = maximum_within_class_nuisance_correlation(
            probabilities,
            diagnostic_Z,
            y,
        )
        dvu = float(
            auc - DVU_NUISANCE_CORRELATION_PENALTY * nuisance_corr
        )
        results.append(
            {
                "parameters": dict(row),
                "raw_scores": raw_scores,
                "probabilities": probabilities,
                "labels": y.copy(),
                "auc": auc,
                "within_class_max_abs_nuisance_correlation": nuisance_corr,
                "within_class_correlations": class_corr,
                "dvu": dvu,
                "calibrator": calibrator,
                "inner_splits": int(len(np.unique(folds))),
            }
        )
    return results


def select_parameters(
    spec: ModelSpec,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    outer_fold_index: int,
    verbose: bool,
) -> dict[str, Any]:
    candidates = evaluate_inner_parameter_grid(
        spec,
        X,
        Z,
        y,
        outer_fold_index,
    )
    if verbose:
        for result in candidates:
            print(
                f"[INNER][{spec.experiment_id}][outer={outer_fold_index}] "
                f"params={result['parameters']}, AUC={result['auc']:.4f}, "
                f"nuisance_corr={result['within_class_max_abs_nuisance_correlation']:.4f}, "
                f"DVU={result['dvu']:.4f}",
                flush=True,
            )

    if spec.use_dvu_for_selection:
        best_value = max(row["dvu"] for row in candidates)
        eligible = [
            row for row in candidates if row["dvu"] >= best_value - DVU_SELECTION_TOLERANCE
        ]
        # Within a nearly indistinguishable DVU band, prefer stronger nuisance
        # removal, then stronger ridge regularization, then smaller classifier C,
        # then the narrower matched-pair kernel.
        selected = sorted(
            eligible,
            key=lambda row: (
                -row["parameters"].get("alpha", 0.0),
                -row["parameters"].get("ridge", 0.0),
                row["parameters"].get("c", 0.0),
                row["parameters"].get("tau_multiplier", 0.0),
            ),
        )[0]
        selection_metric = "DVU"
    else:
        best_value = max(row["auc"] for row in candidates)
        eligible = [
            row
            for row in candidates
            if row["auc"] >= best_value - C_SELECTION_AUC_TOLERANCE
        ]
        selected = sorted(eligible, key=lambda row: row["parameters"]["c"])[0]
        selection_metric = "AUC"

    threshold = select_youden_threshold(
        selected["labels"], selected["probabilities"]
    )
    return {
        **selected,
        "threshold": threshold,
        "selection_metric": selection_metric,
        "best_selection_value": float(best_value),
        "candidate_table": [
            {
                "parameters": row["parameters"],
                "auc": row["auc"],
                "within_class_max_abs_nuisance_correlation": row[
                    "within_class_max_abs_nuisance_correlation"
                ],
                "dvu": row["dvu"],
            }
            for row in candidates
        ],
    }


def run_nested_cv_once(
    spec: ModelSpec,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    fold_by_patient: dict[str, int],
    verbose: bool,
) -> dict[str, Any]:
    patient_ids = np.asarray(patient_ids).astype(str)
    folds = np.asarray([fold_by_patient[pid] for pid in patient_ids], dtype=np.int64)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    predictions = np.full(len(y), -1, dtype=np.int64)
    thresholds = np.full(len(y), np.nan, dtype=np.float64)
    selected_rows: list[dict[str, Any]] = []

    for outer_fold in sorted(np.unique(folds)):
        train = folds != outer_fold
        valid = folds == outer_fold
        selected = select_parameters(
            spec,
            X[train],
            Z[train],
            y[train],
            outer_fold_index=int(outer_fold),
            verbose=verbose,
        )
        model = create_model(spec, selected["parameters"])
        model.fit(X[train], Z[train], y[train])
        raw_valid = model.raw_scores(X[valid], Z[valid])
        if model.requires_calibration:
            calibrator = selected["calibrator"]
            if calibrator is None:
                raise RuntimeError("Pairwise model is missing inner-OOF calibration.")
            probability_valid = calibrator.predict(raw_valid)
        else:
            probability_valid = np.clip(raw_valid, 0.0, 1.0)
        prediction_valid = (probability_valid >= selected["threshold"]).astype(np.int64)
        scores[valid] = probability_valid
        predictions[valid] = prediction_valid
        thresholds[valid] = float(selected["threshold"])
        selected_rows.append(
            {
                "outer_fold": int(outer_fold),
                "train_patients": int(np.sum(train)),
                "valid_patients": int(np.sum(valid)),
                "selected_parameters_json": json.dumps(
                    selected["parameters"], sort_keys=True
                ),
                "selection_metric": selected["selection_metric"],
                "selected_inner_auc": float(selected["auc"]),
                "selected_inner_dvu": float(selected["dvu"]),
                "selected_inner_nuisance_correlation": float(
                    selected["within_class_max_abs_nuisance_correlation"]
                ),
                "best_selection_value": float(selected["best_selection_value"]),
                "training_only_threshold": float(selected["threshold"]),
                "held_out_auc": (
                    float(roc_auc_score(y[valid], probability_valid))
                    if len(np.unique(y[valid])) == 2
                    else float("nan")
                ),
            }
        )
        if verbose:
            print(
                f"[OUTER][{spec.experiment_id}] fold={outer_fold}, "
                f"params={selected['parameters']}, inner_AUC={selected['auc']:.4f}, "
                f"inner_DVU={selected['dvu']:.4f}, "
                f"held_out_AUC={selected_rows[-1]['held_out_auc']:.4f}",
                flush=True,
            )

    if not np.all(np.isfinite(scores)) or np.any(predictions < 0):
        raise RuntimeError(f"{spec.experiment_id}: incomplete outer OOF predictions.")
    metrics = binary_metrics(y, scores, predictions)
    diagnostic_Z = nuisance_diagnostic_coordinates(Z)
    nuisance_corr, nuisance_corr_by_class = maximum_within_class_nuisance_correlation(
        scores, diagnostic_Z, y
    )
    return {
        "spec": spec,
        "patient_ids": patient_ids,
        "labels": y.copy(),
        "folds": folds,
        "probabilities": scores,
        "predictions": predictions,
        "thresholds": thresholds,
        "metrics": metrics,
        "outer_fold_rows": selected_rows,
        "within_class_max_abs_nuisance_correlation": nuisance_corr,
        "within_class_nuisance_correlation_by_class": nuisance_corr_by_class,
        "deconfounded_utility": float(
            metrics["auc"] - DVU_NUISANCE_CORRELATION_PENALTY * nuisance_corr
        ),
    }


# ================================================================================
# REPRESENTATION PREPARATION
# ================================================================================


def prepare_patient_representations(bank: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(bank["labels"], dtype=np.int64)
    patient_ids = np.asarray(bank["patient_ids"]).astype(str)
    series_ids = np.asarray(bank["series_ids"]).astype(str)
    sample_indices = np.asarray(bank["sample_indices"], dtype=np.int64)

    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pooling in ("hierarchical", "robust_median_of_means"):
        for mode in REQUIRED_V6_FEATURE_MODES:
            key = f"{mode}::{pooling}"
            prepared[key] = pool_feature_mode(
                bank["features"][mode],
                labels,
                patient_ids,
                series_ids,
                sample_indices,
                pooling,
            )

    reference_key = f"{V6_A17_FEATURE_MODE}::hierarchical"
    _, y_reference, ordered_patients = prepared[reference_key]
    Z, nuisance_names, y_nuisance = build_patient_nuisance_matrix(
        bank, ordered_patients
    )
    if not np.array_equal(y_reference, y_nuisance):
        raise RuntimeError("Nuisance matrix labels do not align with A17 patients.")
    for key, (_, y, patients) in prepared.items():
        if not np.array_equal(patients, ordered_patients):
            raise RuntimeError(f"Prepared patient order differs for {key}.")
        if not np.array_equal(y, y_reference):
            raise RuntimeError(f"Prepared labels differ for {key}.")

    return {
        "image_representations": prepared,
        "nuisance_matrix": Z,
        "nuisance_feature_names": nuisance_names,
        "labels": y_reference,
        "patient_ids": ordered_patients,
    }


def representation_for_spec(
    spec: ModelSpec,
    prepared: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z = prepared["nuisance_matrix"]
    y = prepared["labels"]
    patients = prepared["patient_ids"]
    if spec.feature_mode == "v7_nuisance_only":
        # Z is supplied both as signal and nuisance only to keep the generic
        # nested-CV function contract. StandardPatientModel ignores its Z input.
        return Z.copy(), Z, y, patients
    key = f"{spec.feature_mode}::{spec.pooling}"
    X, y_mode, patient_mode = prepared["image_representations"][key]
    if not np.array_equal(y_mode, y) or not np.array_equal(patient_mode, patients):
        raise RuntimeError(f"Representation alignment failed for {spec.experiment_id}.")
    return X, Z, y, patients


# ================================================================================
# FOLD MANIFESTS, REPEATS, COMPARISONS AND STRESS TESTS
# ================================================================================


def fold_map_from_manifest(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {str(row["patient_id"]): int(row["outer_fold"]) for row in rows}


def create_repeated_fold_map(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> dict[str, int]:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=int(random_state),
    )
    folds = np.zeros(len(patient_ids), dtype=np.int64)
    dummy = np.zeros((len(patient_ids), 1), dtype=np.float64)
    for fold, (_, valid) in enumerate(splitter.split(dummy, labels), start=1):
        folds[valid] = fold
    return {str(patient): int(fold) for patient, fold in zip(patient_ids, folds)}


def paired_auc_bootstrap(
    labels: np.ndarray,
    reference_scores: np.ndarray,
    comparison_scores: np.ndarray,
    replicates: int,
    random_state: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    observed = float(
        roc_auc_score(labels, comparison_scores)
        - roc_auc_score(labels, reference_scores)
    )
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(int(replicates)):
        sample = np.concatenate(
            [
                rng.choice(class0, len(class0), replace=True),
                rng.choice(class1, len(class1), replace=True),
            ]
        )
        values.append(
            roc_auc_score(labels[sample], comparison_scores[sample])
            - roc_auc_score(labels[sample], reference_scores[sample])
        )
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "delta_auc": observed,
        "ci_lower": float(np.quantile(values_array, 0.025)),
        "ci_upper": float(np.quantile(values_array, 0.975)),
        "probability_delta_above_zero": float(np.mean(values_array > 0.0)),
    }



def deterministic_small_kmeans(
    coordinates: np.ndarray,
    n_clusters: int,
    maximum_iterations: int = 100,
) -> np.ndarray:
    """Pure-NumPy deterministic k-means for the tiny nuisance stress audit.

    Farthest-first initialization is label blind and removes reliance on native
    OpenMP thread pools. The routine is intentionally limited to the 30-patient
    diagnostic setting; it is not a general large-scale clustering library.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or len(coordinates) < int(n_clusters):
        raise ValueError("Small k-means requires enough 2D coordinate rows.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Small k-means coordinates contain non-finite values.")
    n_clusters = int(n_clusters)
    center = np.mean(coordinates, axis=0)
    first = int(np.argmax(np.linalg.norm(coordinates - center, axis=1)))
    chosen = [first]
    while len(chosen) < n_clusters:
        squared = np.stack(
            [
                np.sum(np.square(coordinates - coordinates[index]), axis=1)
                for index in chosen
            ],
            axis=1,
        )
        nearest = np.min(squared, axis=1)
        nearest[np.asarray(chosen, dtype=np.int64)] = -np.inf
        chosen.append(int(np.argmax(nearest)))
    centroids = coordinates[np.asarray(chosen)].copy()
    labels = np.full(len(coordinates), -1, dtype=np.int64)

    for _ in range(int(maximum_iterations)):
        distances = np.stack(
            [
                np.sum(np.square(coordinates - centroid), axis=1)
                for centroid in centroids
            ],
            axis=1,
        )
        updated_labels = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(updated_labels, labels):
            break
        labels = updated_labels
        updated_centroids = []
        for cluster in range(n_clusters):
            members = coordinates[labels == cluster]
            if len(members):
                updated_centroids.append(np.mean(members, axis=0))
            else:
                # Re-seed an empty cluster with the point currently farthest
                # from its assigned centroid; no random or label input is used.
                assigned_distance = distances[
                    np.arange(len(coordinates)), labels
                ]
                updated_centroids.append(
                    coordinates[int(np.argmax(assigned_distance))]
                )
        centroids = np.stack(updated_centroids, axis=0)
    return labels


def leave_one_nuisance_cluster_out_audit(
    spec: ModelSpec,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
) -> dict[str, Any]:
    """Descriptive train-one-cluster/test-the-other stress test.

    Clusters are fitted on all nuisance rows without labels because the audit is
    descriptive and no hyperparameter is selected from its result. It should not
    be confused with an unbiased external-validation estimate.
    """

    scaler = StandardScaler().fit(Z)
    standardized = scaler.transform(Z)
    components = min(3, standardized.shape[1], len(Z) - 1)
    coordinates = PCA(
        n_components=components, svd_solver="full", random_state=RANDOM_SEED
    ).fit_transform(standardized)
    clusters = deterministic_small_kmeans(
        coordinates,
        n_clusters=NUISANCE_CLUSTER_COUNT,
    )
    cluster_rows = []
    test_scores = np.full(len(y), np.nan, dtype=np.float64)
    for held_out in sorted(np.unique(clusters)):
        train = clusters != held_out
        valid = clusters == held_out
        class_counts_train = np.bincount(y[train], minlength=2)
        class_counts_valid = np.bincount(y[valid], minlength=2)
        minimum_training_count = 2 if spec.use_pairwise_auc else 1
        identifiable = bool(
            np.all(class_counts_train >= minimum_training_count)
            and np.all(class_counts_valid > 0)
        )
        row = {
            "held_out_cluster": int(held_out),
            "train_patients": int(np.sum(train)),
            "valid_patients": int(np.sum(valid)),
            "train_normal": int(class_counts_train[0]),
            "train_sick": int(class_counts_train[1]),
            "valid_normal": int(class_counts_valid[0]),
            "valid_sick": int(class_counts_valid[1]),
            "identifiable": identifiable,
        }
        if not identifiable:
            row["auc"] = None
            row["status"] = "NON_IDENTIFIABLE_PROTOCOL_LABEL_OVERLAP"
            cluster_rows.append(row)
            continue
        # Use one predeclared central FLNO setting for this stress test; do not
        # tune against the held-out cluster.
        if spec.use_pairwise_auc:
            model = FLNOPairwiseAUCModel(0.75, 1.0, 0.1, 1.0)
        elif spec.use_flno:
            model = FLNOLogisticModel(0.75, 1.0, 0.1)
        else:
            model = StandardPatientModel(0.1)
        model.fit(X[train], Z[train], y[train])
        raw = model.raw_scores(X[valid], Z[valid])
        if model.requires_calibration:
            # Fit calibration only on training scores generated by 3-fold OOF.
            folds = create_inner_folds(y[train], outer_fold_index=90 + int(held_out))
            train_raw = np.full(np.sum(train), np.nan, dtype=np.float64)
            X_train, Z_train, y_train = X[train], Z[train], y[train]
            for fold in np.unique(folds):
                fit_mask = folds != fold
                val_mask = folds == fold
                fold_model = FLNOPairwiseAUCModel(0.75, 1.0, 0.1, 1.0)
                fold_model.fit(X_train[fit_mask], Z_train[fit_mask], y_train[fit_mask])
                train_raw[val_mask] = fold_model.raw_scores(
                    X_train[val_mask], Z_train[val_mask]
                )
            probabilities = SigmoidCalibrator().fit(train_raw, y_train).predict(raw)
        else:
            probabilities = np.clip(raw, 0.0, 1.0)
        test_scores[valid] = probabilities
        row["auc"] = float(roc_auc_score(y[valid], probabilities))
        row["status"] = "OK"
        cluster_rows.append(row)
    finite = np.isfinite(test_scores)
    pooled_auc = (
        float(roc_auc_score(y[finite], test_scores[finite]))
        if np.all(finite) and len(np.unique(y[finite])) == 2
        else None
    )
    return {
        "experiment_id": spec.experiment_id,
        "status": "OK",
        "cluster_rows": cluster_rows,
        "pooled_leave_cluster_out_auc": pooled_auc,
        "interpretation": (
            "Clusters are nuisance-derived proxies, not validated scanner or "
            "sequence labels. A non-identifiable cluster/class split is itself "
            "evidence that protocol and label cannot be separated internally."
        ),
        "patient_clusters": [
            {
                "patient_id": str(patient),
                "true_label": int(label),
                "nuisance_cluster": int(cluster),
            }
            for patient, label, cluster in zip(patient_ids, y, clusters)
        ],
    }



def select_candidate_family_from_nested_results(
    candidate_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Select model identity by inner DVU and reuse its valid outer prediction.

    Each candidate result was independently generated by full nested CV on the
    same outer folds. For a given outer fold, candidate identity is chosen using
    only ``selected_inner_dvu`` from that fold's outer-training cohort. Reusing
    the already computed held-out prediction is algebraically equivalent to
    refitting that selected candidate again, but avoids duplicate computation.
    """

    missing = sorted(set(V7_CANDIDATE_IDS) - set(candidate_results))
    if missing:
        raise KeyError(f"Candidate-family selection is missing results: {missing}.")
    reference = candidate_results[V7_CANDIDATE_IDS[0]]
    patients = np.asarray(reference["patient_ids"]).astype(str)
    labels = np.asarray(reference["labels"], dtype=np.int64)
    folds = np.asarray(reference["folds"], dtype=np.int64)
    for experiment_id in V7_CANDIDATE_IDS[1:]:
        row = candidate_results[experiment_id]
        if not np.array_equal(np.asarray(row["patient_ids"]).astype(str), patients):
            raise RuntimeError(
                f"{experiment_id}: patient order differs in candidate selection."
            )
        if not np.array_equal(np.asarray(row["labels"], dtype=np.int64), labels):
            raise RuntimeError(
                f"{experiment_id}: labels differ in candidate selection."
            )
        if not np.array_equal(np.asarray(row["folds"], dtype=np.int64), folds):
            raise RuntimeError(
                f"{experiment_id}: outer folds differ in candidate selection."
            )

    score = np.full(len(labels), np.nan, dtype=np.float64)
    prediction = np.full(len(labels), -1, dtype=np.int64)
    threshold = np.full(len(labels), np.nan, dtype=np.float64)
    selected_model = np.empty(len(labels), dtype=object)
    fold_rows: list[dict[str, Any]] = []

    for fold in sorted(np.unique(folds)):
        inner_dvu_by_id = {}
        for experiment_id in V7_CANDIDATE_IDS:
            outer_rows = {
                int(row["outer_fold"]): row
                for row in candidate_results[experiment_id]["outer_fold_rows"]
            }
            if int(fold) not in outer_rows:
                raise RuntimeError(
                    f"{experiment_id}: missing outer-fold row {fold}."
                )
            inner_dvu_by_id[experiment_id] = float(
                outer_rows[int(fold)]["selected_inner_dvu"]
            )
        best_inner_dvu = max(inner_dvu_by_id.values())
        eligible = {
            experiment_id
            for experiment_id, value in inner_dvu_by_id.items()
            if value >= best_inner_dvu - DVU_SELECTION_TOLERANCE
        }
        chosen = next(
            experiment_id
            for experiment_id in V7_CANDIDATE_SELECTION_PRIORITY
            if experiment_id in eligible
        )
        valid = folds == fold
        chosen_result = candidate_results[chosen]
        score[valid] = np.asarray(chosen_result["probabilities"])[valid]
        prediction[valid] = np.asarray(chosen_result["predictions"])[valid]
        threshold[valid] = np.asarray(chosen_result["thresholds"])[valid]
        selected_model[valid] = chosen
        fold_rows.append(
            {
                "outer_fold": int(fold),
                "selected_experiment_id": chosen,
                "selected_inner_dvu": float(inner_dvu_by_id[chosen]),
                "best_candidate_inner_dvu": float(best_inner_dvu),
                "eligible_candidate_ids_json": json.dumps(
                    [
                        experiment_id
                        for experiment_id in V7_CANDIDATE_SELECTION_PRIORITY
                        if experiment_id in eligible
                    ]
                ),
                "candidate_inner_dvu_json": json.dumps(
                    inner_dvu_by_id, sort_keys=True
                ),
                "held_out_patients": int(np.sum(valid)),
                "held_out_auc": (
                    float(roc_auc_score(labels[valid], score[valid]))
                    if len(np.unique(labels[valid])) == 2
                    else None
                ),
            }
        )

    if not np.all(np.isfinite(score)) or np.any(prediction < 0):
        raise RuntimeError("Candidate-family nested selection is incomplete.")
    metrics = binary_metrics(labels, score, prediction)
    selection_counts = {
        experiment_id: int(
            sum(
                row["selected_experiment_id"] == experiment_id
                for row in fold_rows
            )
        )
        for experiment_id in V7_CANDIDATE_SELECTION_PRIORITY
    }
    return {
        "patient_ids": patients,
        "labels": labels,
        "folds": folds,
        "probabilities": score,
        "predictions": prediction,
        "thresholds": threshold,
        "selected_experiment_ids": selected_model.astype(str),
        "metrics": metrics,
        "selection_counts_across_outer_folds": selection_counts,
        "outer_fold_rows": fold_rows,
        "selection_rule": (
            "Choose the first predeclared candidate within DVU_SELECTION_TOLERANCE "
            "of the greatest outer-training inner-OOF DVU. Outer-test scores do "
            "not participate in model identity selection."
        ),
    }


# ================================================================================
# SELECTION-ADJUSTED PERMUTATION
# ================================================================================


def run_selection_adjusted_permutation(
    candidate_specs: Sequence[ModelSpec],
    representations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    patient_ids: np.ndarray,
    original_labels: np.ndarray,
    observed_best_auc: float,
    observed_selection_auc: float,
    output_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(SELECTION_ADJUSTED_PERMUTATION_RANDOM_STATE)
    rows = []
    progress_interval = max(1, SELECTION_ADJUSTED_PERMUTATION_REPLICATES // 10)
    for permutation_index in range(SELECTION_ADJUSTED_PERMUTATION_REPLICATES):
        permuted_labels = rng.permutation(original_labels)
        fold_map = create_repeated_fold_map(
            patient_ids,
            permuted_labels,
            random_state=RANDOM_SEED,
        )
        best_auc = -np.inf
        best_id = None
        candidate_aucs: dict[str, float] = {}
        candidate_results: dict[str, dict[str, Any]] = {}
        for spec in candidate_specs:
            X, Z, _, patients = representations[spec.experiment_id]
            result = run_nested_cv_once(
                spec,
                X,
                Z,
                permuted_labels,
                patients,
                fold_map,
                verbose=False,
            )
            candidate_results[spec.experiment_id] = result
            auc = float(result["metrics"]["auc"])
            candidate_aucs[spec.experiment_id] = auc
            if auc > best_auc:
                best_auc = auc
                best_id = spec.experiment_id
        selected_procedure = select_candidate_family_from_nested_results(
            candidate_results
        )
        rows.append(
            {
                "permutation_index": permutation_index + 1,
                "maximum_auc": float(best_auc),
                "maximum_auc_experiment_id": str(best_id),
                "nested_selection_auc": float(
                    selected_procedure["metrics"]["auc"]
                ),
                "nested_selection_counts_json": json.dumps(
                    selected_procedure[
                        "selection_counts_across_outer_folds"
                    ],
                    sort_keys=True,
                ),
                "candidate_aucs_json": json.dumps(candidate_aucs, sort_keys=True),
            }
        )
        if (
            permutation_index == 0
            or (permutation_index + 1) % progress_interval == 0
            or permutation_index + 1 == SELECTION_ADJUSTED_PERMUTATION_REPLICATES
        ):
            print(
                f"[V7 PERMUTATION] {permutation_index + 1}/"
                f"{SELECTION_ADJUSTED_PERMUTATION_REPLICATES}; "
                f"latest maximum AUC={best_auc:.4f} ({best_id}); "
                f"nested-selection AUC="
                f"{selected_procedure['metrics']['auc']:.4f}",
                flush=True,
            )
    maximum_null = np.asarray(
        [row["maximum_auc"] for row in rows], dtype=np.float64
    )
    selection_null = np.asarray(
        [row["nested_selection_auc"] for row in rows], dtype=np.float64
    )
    familywise_p_value = float(
        (1 + np.sum(maximum_null >= observed_best_auc))
        / (SELECTION_ADJUSTED_PERMUTATION_REPLICATES + 1)
    )
    selection_p_value = float(
        (1 + np.sum(selection_null >= observed_selection_auc))
        / (SELECTION_ADJUSTED_PERMUTATION_REPLICATES + 1)
    )
    summary = {
        "status": "OK",
        "candidate_experiment_ids": [spec.experiment_id for spec in candidate_specs],
        "candidate_selection_priority": list(V7_CANDIDATE_SELECTION_PRIORITY),
        "observed_best_auc": float(observed_best_auc),
        "observed_nested_selection_auc": float(observed_selection_auc),
        "replicates": int(SELECTION_ADJUSTED_PERMUTATION_REPLICATES),
        "maximum_auc_null_mean": float(np.mean(maximum_null)),
        "maximum_auc_null_median": float(np.median(maximum_null)),
        "maximum_auc_null_q025": float(np.quantile(maximum_null, 0.025)),
        "maximum_auc_null_q975": float(np.quantile(maximum_null, 0.975)),
        "empirical_familywise_max_auc_p_value": familywise_p_value,
        "nested_selection_null_mean": float(np.mean(selection_null)),
        "nested_selection_null_median": float(np.median(selection_null)),
        "nested_selection_null_q025": float(np.quantile(selection_null, 0.025)),
        "nested_selection_null_q975": float(np.quantile(selection_null, 0.975)),
        "empirical_nested_selection_p_value": selection_p_value,
        # Backward-readable aliases for simple report consumers.
        "null_mean": float(np.mean(maximum_null)),
        "null_median": float(np.median(maximum_null)),
        "null_q025": float(np.quantile(maximum_null, 0.025)),
        "null_q975": float(np.quantile(maximum_null, 0.975)),
        "empirical_familywise_p_value": familywise_p_value,
        "interpretation": (
            "The max-AUC null corrects for choosing the best pooled OOF AUC among "
            "the five listed candidates. The nested-selection null separately "
            "repeats the predeclared inner-DVU model-identity rule in every "
            "permutation. Neither test corrects for unlisted historical ideas."
        ),
    }
    write_csv(output_dir / "selection_adjusted_permutation_runs.csv", rows)
    write_json(output_dir / "selection_adjusted_permutation_summary.json", summary)
    return summary


def compare_reproduced_a17_with_v6(
    suite_dir: Path | None,
    reproduced_result: dict[str, Any],
) -> dict[str, Any]:
    """Compare V7's A17 reproduction with the saved V6 OOF result."""

    if suite_dir is None:
        return {"status": "UNAVAILABLE_NO_V6_SUITE_DIR"}
    prediction_path = (
        suite_dir
        / "experiments"
        / "A17_HARD_SUPPORT_REGION_NORM_HIER_LR_PCA"
        / "patient_oof_predictions.csv"
    )
    if not prediction_path.is_file():
        return {
            "status": "UNAVAILABLE_NO_V6_A17_PREDICTIONS",
            "path": str(prediction_path),
        }
    with prediction_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    score_field = next(
        (field for field in ("oof_score", "probability", "score") if rows and field in rows[0]),
        None,
    )
    if score_field is None:
        return {
            "status": "UNAVAILABLE_UNKNOWN_SCORE_COLUMN",
            "path": str(prediction_path),
        }
    v6_scores = {str(row["patient_id"]): float(row[score_field]) for row in rows}
    v7_scores = {
        str(patient): float(score)
        for patient, score in zip(
            reproduced_result["patient_ids"],
            reproduced_result["probabilities"],
        )
    }
    if set(v6_scores) != set(v7_scores):
        return {
            "status": "FAILED_PATIENT_SET_MISMATCH",
            "v6_only": sorted(set(v6_scores) - set(v7_scores)),
            "v7_only": sorted(set(v7_scores) - set(v6_scores)),
        }
    ordered = sorted(v6_scores)
    differences = np.asarray(
        [v7_scores[patient] - v6_scores[patient] for patient in ordered],
        dtype=np.float64,
    )
    labels = np.asarray(reproduced_result["labels"], dtype=np.int64)
    patient_to_label = {
        str(patient): int(label)
        for patient, label in zip(
            reproduced_result["patient_ids"],
            labels,
        )
    }
    ordered_labels = np.asarray(
        [patient_to_label[patient] for patient in ordered], dtype=np.int64
    )
    v6_array = np.asarray([v6_scores[patient] for patient in ordered])
    v7_array = np.asarray([v7_scores[patient] for patient in ordered])
    maximum = float(np.max(np.abs(differences)))
    return {
        "status": "MATCH" if maximum <= 1e-6 else "NUMERIC_DIFFERENCE",
        "v6_prediction_path": str(prediction_path),
        "maximum_absolute_score_difference": maximum,
        "mean_absolute_score_difference": float(np.mean(np.abs(differences))),
        "v6_auc": float(roc_auc_score(ordered_labels, v6_array)),
        "v7_reproduced_auc": float(roc_auc_score(ordered_labels, v7_array)),
        "auc_difference": float(
            roc_auc_score(ordered_labels, v7_array)
            - roc_auc_score(ordered_labels, v6_array)
        ),
        "interpretation": (
            "MATCH confirms that the post-cache extension reproduces the V6 A17 "
            "patient scores before applying new mathematics. A small numeric "
            "difference can arise from a library-version change; a large one "
            "requires investigation before interpreting V7 challengers."
        ),
    }


# ================================================================================
# MAIN
# ================================================================================


def validate_configuration() -> None:
    ids = [spec.experiment_id for spec in MODEL_SPECS]
    if len(ids) != len(set(ids)):
        raise ValueError("V7 experiment IDs must be unique.")
    by_id = {spec.experiment_id: spec for spec in MODEL_SPECS}
    if V7_PROSPECTIVE_CHALLENGER_ID not in ids:
        raise ValueError("The V7 prospective challenger is not registered.")
    if not set(V7_CANDIDATE_IDS).issubset(ids):
        raise ValueError("Every candidate ID must be registered.")
    if set(V7_CANDIDATE_SELECTION_PRIORITY) != set(V7_CANDIDATE_IDS):
        raise ValueError(
            "V7_CANDIDATE_SELECTION_PRIORITY must contain every candidate once."
        )
    if len(V7_CANDIDATE_SELECTION_PRIORITY) != len(
        set(V7_CANDIDATE_SELECTION_PRIORITY)
    ):
        raise ValueError("Candidate-selection priority IDs must be unique.")
    if set(MATCHED_CONTROL_IDS_BY_CANDIDATE) != set(V7_CANDIDATE_IDS):
        raise ValueError("Every V7 candidate requires a matched-control tuple.")

    allowed_pooling = {
        "hierarchical",
        "robust_median_of_means",
        "patient_tabular",
    }
    allowed_learners = {
        "logistic_regression",
        "flno_logistic_regression",
        "flno_pairwise_auc",
    }
    allowed_modes = set(REQUIRED_V6_FEATURE_MODES) | {"v7_nuisance_only"}
    for spec in MODEL_SPECS:
        if spec.pooling not in allowed_pooling:
            raise ValueError(
                f"{spec.experiment_id}: unsupported pooling {spec.pooling!r}."
            )
        if spec.learner not in allowed_learners:
            raise ValueError(
                f"{spec.experiment_id}: unsupported learner {spec.learner!r}."
            )
        if spec.feature_mode not in allowed_modes:
            raise ValueError(
                f"{spec.experiment_id}: unsupported feature mode "
                f"{spec.feature_mode!r}."
            )
        if spec.use_pairwise_auc != (spec.learner == "flno_pairwise_auc"):
            raise ValueError(
                f"{spec.experiment_id}: pairwise flag and learner disagree."
            )
        if spec.use_flno != spec.learner.startswith("flno_"):
            raise ValueError(
                f"{spec.experiment_id}: FLNO flag and learner disagree."
            )

    # Every anatomical gap must compare representations through the identical
    # pooling and learner. Otherwise the delta would conflate anatomy with an
    # algorithm change.
    for candidate_id, control_ids in MATCHED_CONTROL_IDS_BY_CANDIDATE.items():
        if len(control_ids) != 3 or len(set(control_ids)) != 3:
            raise ValueError(
                f"{candidate_id}: exactly three unique C31/C32/C33 controls are required."
            )
        candidate = by_id[candidate_id]
        for control_id in control_ids:
            if control_id not in by_id:
                raise ValueError(
                    f"{candidate_id}: unknown matched control {control_id!r}."
                )
            control = by_id[control_id]
            contract = (
                control.pooling == candidate.pooling
                and control.learner == candidate.learner
                and control.use_flno == candidate.use_flno
                and control.use_pairwise_auc == candidate.use_pairwise_auc
                and control.use_dvu_for_selection
                == candidate.use_dvu_for_selection
            )
            if not contract:
                raise ValueError(
                    f"{candidate_id} and {control_id} do not share the exact "
                    "pooling/learner/selection contract."
                )

    if not 0 <= DVU_NUISANCE_CORRELATION_PENALTY:
        raise ValueError("DVU nuisance penalty cannot be negative.")
    if DVU_SELECTION_TOLERANCE < 0:
        raise ValueError("DVU_SELECTION_TOLERANCE cannot be negative.")
    if RMOM_BLOCKS_PER_SERIES < 1:
        raise ValueError("RMOM_BLOCKS_PER_SERIES must be positive.")
    if N_SPLITS < 2 or INNER_SPLITS < 2:
        raise ValueError("Outer and inner CV require at least two folds.")
    if NUISANCE_CLUSTER_COUNT < 2:
        raise ValueError("NUISANCE_CLUSTER_COUNT must be at least two.")
    if any(alpha < 0 or alpha > 1 for alpha in FLNO_ALPHA_GRID):
        raise ValueError("FLNO alpha values must lie in [0,1].")
    if any(value <= 0 for value in FLNO_RIDGE_GRID):
        raise ValueError("FLNO ridge values must be positive.")
    if any(value <= 0 for value in CLASSIFIER_C_GRID + PAIRWISE_C_GRID):
        raise ValueError("Classifier C values must be positive.")
    if any(value <= 0 for value in PAIRWISE_TAU_MULTIPLIER_GRID):
        raise ValueError("Pairwise tau multipliers must be positive.")
    if not 0 < PAIRWISE_MIN_WEIGHT <= 1:
        raise ValueError("PAIRWISE_MIN_WEIGHT must lie in (0,1].")
    for name, value in (
        ("BOOTSTRAP_REPLICATES", BOOTSTRAP_REPLICATES),
        ("PAIRED_BOOTSTRAP_REPLICATES", PAIRED_BOOTSTRAP_REPLICATES),
        ("REPEATED_NESTED_CV_REPEATS", REPEATED_NESTED_CV_REPEATS),
        (
            "SELECTION_ADJUSTED_PERMUTATION_REPLICATES",
            SELECTION_ADJUSTED_PERMUTATION_REPLICATES,
        ),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")


def main() -> None:
    started = time.perf_counter()
    np.random.seed(RANDOM_SEED)
    validate_configuration()

    cache_dir, cache_metadata = discover_v6_feature_cache()
    cache_fingerprint = str(cache_metadata.get("fingerprint", cache_dir.name))
    configuration_payload = {
        "version": "V7 mathematical deconfounding extension",
        "v6_cache_dir": str(cache_dir),
        "v6_cache_fingerprint": cache_fingerprint,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "v7_candidate_ids": list(V7_CANDIDATE_IDS),
        "v7_prospective_challenger_id": V7_PROSPECTIVE_CHALLENGER_ID,
        "v7_candidate_selection_priority": list(
            V7_CANDIDATE_SELECTION_PRIORITY
        ),
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "inner_splits": INNER_SPLITS,
        "classifier_c_grid": CLASSIFIER_C_GRID,
        "flno_alpha_grid": FLNO_ALPHA_GRID,
        "flno_ridge_grid": FLNO_RIDGE_GRID,
        "pairwise_c_grid": PAIRWISE_C_GRID,
        "pairwise_tau_multiplier_grid": PAIRWISE_TAU_MULTIPLIER_GRID,
        "dvu_nuisance_correlation_penalty": DVU_NUISANCE_CORRELATION_PENALTY,
        "rmom_blocks_per_series": RMOM_BLOCKS_PER_SERIES,
        "repeated_nested_cv_repeats": REPEATED_NESTED_CV_REPEATS,
        "selection_adjusted_permutation_replicates": (
            SELECTION_ADJUSTED_PERMUTATION_REPLICATES
        ),
        "verbose_inner_for_candidates": VERBOSE_INNER_FOR_CANDIDATES,
        "run_leave_one_nuisance_cluster_out_audit": (
            RUN_LEAVE_ONE_NUISANCE_CLUSTER_OUT_AUDIT
        ),
        "nuisance_cluster_count": NUISANCE_CLUSTER_COUNT,
        "provenance_nuisance_features": PROVENANCE_NUISANCE_FEATURE_NAMES,
        "standardization_nuisance_features": (
            STANDARDIZATION_NUISANCE_FEATURE_NAMES
        ),
        "include_monai_qc_in_primary_flno": INCLUDE_MONAI_QC_IN_PRIMARY_FLNO,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    suite_tag = hashlib.sha256(
        json.dumps(json_ready(configuration_payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    output_dir = OUTPUT_ROOT / f"v7_math_deconfounding__{suite_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "v7_configuration.json", configuration_payload)

    print("#" * 96, flush=True)
    print("CAD CARDIAC MRI — MATHEMATICAL DECONFOUNDING EXTENSION V7", flush=True)
    print("#" * 96, flush=True)
    print(f"[V7] V6 cache: {cache_dir}", flush=True)
    print(f"[V7] Output: {output_dir}", flush=True)

    bank = load_v6_bank(cache_dir, cache_metadata)
    print(
        f"[V7] Loaded {len(bank['labels'])} frozen slice rows and "
        f"{len(REQUIRED_V6_FEATURE_MODES)} required V6 feature modes.",
        flush=True,
    )
    prepared = prepare_patient_representations(bank)
    patient_ids = prepared["patient_ids"]
    labels = prepared["labels"]
    nuisance = prepared["nuisance_matrix"]
    nuisance_names = prepared["nuisance_feature_names"]
    print(
        f"[V7] Patient rows={len(patient_ids)}; Normal={int(np.sum(labels == 0))}; "
        f"Sick={int(np.sum(labels == 1))}; nuisance_features={nuisance.shape[1]}.",
        flush=True,
    )

    nuisance_rows = []
    for patient_id, label, values in zip(patient_ids, labels, nuisance):
        row = {"patient_id": str(patient_id), "true_label": int(label)}
        row.update({name: float(value) for name, value in zip(nuisance_names, values)})
        nuisance_rows.append(row)
    write_csv(output_dir / "audits" / "patient_nuisance_matrix.csv", nuisance_rows)

    suite_dir = discover_v6_suite_dir(patient_ids)
    outer_manifest = load_or_create_outer_manifest(suite_dir, patient_ids, labels)
    write_csv(output_dir / "manifests" / "patient_fold_manifest.csv", outer_manifest)
    fold_map = fold_map_from_manifest(outer_manifest)
    print(
        f"[V7] Outer folds: {'reused from ' + str(suite_dir) if suite_dir else 'rebuilt deterministically'}.",
        flush=True,
    )

    representations: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {
        spec.experiment_id: representation_for_spec(spec, prepared)
        for spec in MODEL_SPECS
    }

    # ------------------------------- Main nested CV -------------------------------
    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(MODEL_SPECS, start=1):
        print(
            f"\n[V7] Experiment {index}/{len(MODEL_SPECS)}: {spec.experiment_id}",
            flush=True,
        )
        experiment_started = time.perf_counter()
        X, Z, y, patients = representations[spec.experiment_id]
        result = run_nested_cv_once(
            spec,
            X,
            Z,
            y,
            patients,
            fold_map,
            verbose=(
                VERBOSE_INNER_FOR_CANDIDATES
                and spec.experiment_id in V7_CANDIDATE_IDS
            ),
        )
        intervals = stratified_bootstrap_intervals(
            y,
            result["probabilities"],
            result["predictions"],
            BOOTSTRAP_REPLICATES,
            RANDOM_SEED + 500 + index,
        )
        result["confidence_intervals"] = intervals
        results[spec.experiment_id] = result
        experiment_dir = output_dir / "experiments" / spec.experiment_id
        prediction_rows = [
            {
                "patient_id": str(patient),
                "true_label": int(label),
                "outer_fold": int(fold),
                "oof_score": float(score),
                "training_only_threshold": float(threshold),
                "predicted_label": int(prediction),
            }
            for patient, label, fold, score, threshold, prediction in zip(
                result["patient_ids"],
                result["labels"],
                result["folds"],
                result["probabilities"],
                result["thresholds"],
                result["predictions"],
            )
        ]
        write_csv(experiment_dir / "patient_oof_predictions.csv", prediction_rows)
        write_csv(experiment_dir / "outer_fold_selection.csv", result["outer_fold_rows"])
        summary = {
            "experiment": asdict(spec),
            "metrics": result["metrics"],
            "confidence_intervals": intervals,
            "within_class_max_abs_nuisance_correlation": result[
                "within_class_max_abs_nuisance_correlation"
            ],
            "within_class_nuisance_correlation_by_class": result[
                "within_class_nuisance_correlation_by_class"
            ],
            "deconfounded_utility": result["deconfounded_utility"],
            "runtime_seconds": float(time.perf_counter() - experiment_started),
        }
        write_json(experiment_dir / "summary.json", summary)
        row = {
            "experiment_id": spec.experiment_id,
            "role": spec.role,
            "pooling": spec.pooling,
            "learner": spec.learner,
            **result["metrics"],
            "auc_ci_lower": intervals["auc"][0],
            "auc_ci_upper": intervals["auc"][1],
            "within_class_max_abs_nuisance_correlation": result[
                "within_class_max_abs_nuisance_correlation"
            ],
            "deconfounded_utility": result["deconfounded_utility"],
            "runtime_seconds": summary["runtime_seconds"],
        }
        summary_rows.append(row)
        print(
            f"[V7] COMPLETED {spec.experiment_id}: AUC={row['auc']:.4f}, "
            f"AUPRC={row['auprc']:.4f}, nuisance_corr="
            f"{row['within_class_max_abs_nuisance_correlation']:.4f}, "
            f"DVU={row['deconfounded_utility']:.4f}, "
            f"runtime={format_elapsed(row['runtime_seconds'])}",
            flush=True,
        )

    reproduction_audit = compare_reproduced_a17_with_v6(
        suite_dir,
        results["A17R_V6_REPRODUCED_HIER_LR_PCA"],
    )
    write_json(output_dir / "audits" / "v6_a17_reproduction.json", reproduction_audit)
    print(
        "[V7] V6 A17 reproduction audit: "
        f"{reproduction_audit.get('status', 'UNKNOWN')}; "
        f"max_score_difference="
        f"{reproduction_audit.get('maximum_absolute_score_difference', 'n/a')}.",
        flush=True,
    )

    # Candidate identity is now selected only from each outer-training cohort's
    # inner-OOF DVU. This estimates the complete V7 choice rule rather than
    # retrospectively reporting whichever pooled OOF candidate looks best.
    nested_candidate_selection = select_candidate_family_from_nested_results(
        {experiment_id: results[experiment_id] for experiment_id in V7_CANDIDATE_IDS}
    )
    nested_selection_intervals = stratified_bootstrap_intervals(
        nested_candidate_selection["labels"],
        nested_candidate_selection["probabilities"],
        nested_candidate_selection["predictions"],
        BOOTSTRAP_REPLICATES,
        RANDOM_SEED + 8_000,
    )
    nested_selection_corr, nested_selection_class_corr = (
        maximum_within_class_nuisance_correlation(
            nested_candidate_selection["probabilities"],
            nuisance_diagnostic_coordinates(nuisance),
            labels,
        )
    )
    nested_candidate_selection["confidence_intervals"] = (
        nested_selection_intervals
    )
    nested_candidate_selection[
        "within_class_max_abs_nuisance_correlation"
    ] = nested_selection_corr
    nested_candidate_selection[
        "within_class_nuisance_correlation_by_class"
    ] = nested_selection_class_corr
    nested_candidate_selection["deconfounded_utility"] = float(
        nested_candidate_selection["metrics"]["auc"]
        - DVU_NUISANCE_CORRELATION_PENALTY * nested_selection_corr
    )
    nested_selection_dir = output_dir / "candidate_family_selection"
    write_csv(
        nested_selection_dir / "outer_fold_selection.csv",
        nested_candidate_selection["outer_fold_rows"],
    )
    write_csv(
        nested_selection_dir / "patient_oof_predictions.csv",
        [
            {
                "patient_id": str(patient),
                "true_label": int(label),
                "outer_fold": int(fold),
                "oof_score": float(score),
                "training_only_threshold": float(threshold),
                "predicted_label": int(prediction),
                "selected_experiment_id": str(selected_id),
            }
            for patient, label, fold, score, threshold, prediction, selected_id
            in zip(
                nested_candidate_selection["patient_ids"],
                nested_candidate_selection["labels"],
                nested_candidate_selection["folds"],
                nested_candidate_selection["probabilities"],
                nested_candidate_selection["thresholds"],
                nested_candidate_selection["predictions"],
                nested_candidate_selection["selected_experiment_ids"],
            )
        ],
    )
    nested_selection_summary = {
        key: value
        for key, value in nested_candidate_selection.items()
        if key
        not in {
            "patient_ids",
            "labels",
            "folds",
            "probabilities",
            "predictions",
            "thresholds",
            "selected_experiment_ids",
        }
    }
    write_json(
        nested_selection_dir / "summary.json",
        nested_selection_summary,
    )
    print(
        "[V7 CANDIDATE SELECTION] OOF AUC="
        f"{nested_candidate_selection['metrics']['auc']:.4f}; DVU="
        f"{nested_candidate_selection['deconfounded_utility']:.4f}; "
        f"fold selections="
        f"{nested_candidate_selection['selection_counts_across_outer_folds']}",
        flush=True,
    )

    # --------------------------- Paired main comparisons --------------------------
    paired_rows: list[dict[str, Any]] = []
    reference = results["A17R_V6_REPRODUCED_HIER_LR_PCA"]
    for index, candidate_id in enumerate(V7_CANDIDATE_IDS[1:], start=1):
        candidate = results[candidate_id]
        paired = paired_auc_bootstrap(
            labels,
            reference["probabilities"],
            candidate["probabilities"],
            PAIRED_BOOTSTRAP_REPLICATES,
            RANDOM_SEED + 10_000 + index,
        )
        paired_rows.append(
            {
                "comparison": f"A17R_vs_{candidate_id}",
                "reference_experiment_id": "A17R_V6_REPRODUCED_HIER_LR_PCA",
                "comparison_experiment_id": candidate_id,
                **paired,
            }
        )

    matched_bootstrap_seed_offset = 100
    for candidate_id in V7_CANDIDATE_IDS:
        for control_id in MATCHED_CONTROL_IDS_BY_CANDIDATE[candidate_id]:
            paired = paired_auc_bootstrap(
                labels,
                results[control_id]["probabilities"],
                results[candidate_id]["probabilities"],
                PAIRED_BOOTSTRAP_REPLICATES,
                RANDOM_SEED + 10_000 + matched_bootstrap_seed_offset,
            )
            paired_rows.append(
                {
                    "comparison": f"{control_id}_vs_{candidate_id}",
                    "reference_experiment_id": control_id,
                    "comparison_experiment_id": candidate_id,
                    **paired,
                }
            )
            matched_bootstrap_seed_offset += 1
    for index, candidate_id in enumerate(V7_CANDIDATE_IDS, start=500):
        paired = paired_auc_bootstrap(
            labels,
            results["C34_META_NUISANCE_ONLY_LR"]["probabilities"],
            results[candidate_id]["probabilities"],
            PAIRED_BOOTSTRAP_REPLICATES,
            RANDOM_SEED + 10_000 + index,
        )
        paired_rows.append(
            {
                "comparison": f"C34_META_NUISANCE_ONLY_LR_vs_{candidate_id}",
                "reference_experiment_id": "C34_META_NUISANCE_ONLY_LR",
                "comparison_experiment_id": candidate_id,
                **paired,
            }
        )
    write_csv(output_dir / "comparison" / "paired_auc_comparisons.csv", paired_rows)

    # Anatomical gap and final candidate table.
    summary_by_id = {row["experiment_id"]: row for row in summary_rows}
    candidate_rows = []
    for candidate_id in V7_CANDIDATE_IDS:
        candidate = summary_by_id[candidate_id]
        controls = [summary_by_id[cid] for cid in MATCHED_CONTROL_IDS_BY_CANDIDATE[candidate_id]]
        strongest_control = max(controls, key=lambda row: row["auc"])
        candidate_rows.append(
            {
                **candidate,
                "strongest_matched_control_id": strongest_control["experiment_id"],
                "strongest_matched_control_auc": float(strongest_control["auc"]),
                "candidate_minus_strongest_control_auc": float(
                    candidate["auc"] - strongest_control["auc"]
                ),
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            -row["deconfounded_utility"],
            -row["candidate_minus_strongest_control_auc"],
            -row["auc"],
        )
    )
    write_csv(output_dir / "comparison" / "candidate_summary.csv", candidate_rows)
    write_csv(
        output_dir / "comparison" / "all_experiment_summary.csv",
        sorted(summary_rows, key=lambda row: (-row["auc"], row["experiment_id"])),
    )

    # ----------------------------- Repeated nested CV -----------------------------
    repeated_rows: list[dict[str, Any]] = []
    stability_ids = tuple(
        dict.fromkeys(
            [
                *V7_CANDIDATE_IDS,
                *(
                    control_id
                    for candidate_id in V7_CANDIDATE_IDS
                    for control_id in MATCHED_CONTROL_IDS_BY_CANDIDATE[
                        candidate_id
                    ]
                ),
                "C34_META_NUISANCE_ONLY_LR",
            ]
        )
    )
    print("\n[V7 STABILITY] Starting repeated nested CV.", flush=True)
    for repeat_index in range(REPEATED_NESTED_CV_REPEATS):
        seed = REPEATED_NESTED_CV_RANDOM_STATE + repeat_index
        repeat_fold_map = create_repeated_fold_map(patient_ids, labels, seed)
        for experiment_id in stability_ids:
            spec = next(spec for spec in MODEL_SPECS if spec.experiment_id == experiment_id)
            X, Z, y, patients = representations[experiment_id]
            repeat_result = run_nested_cv_once(
                spec, X, Z, y, patients, repeat_fold_map, verbose=False
            )
            repeated_rows.append(
                {
                    "repeat_index": repeat_index + 1,
                    "random_state": seed,
                    "experiment_id": experiment_id,
                    "auc": float(repeat_result["metrics"]["auc"]),
                    "auprc": float(repeat_result["metrics"]["auprc"]),
                    "within_class_max_abs_nuisance_correlation": float(
                        repeat_result["within_class_max_abs_nuisance_correlation"]
                    ),
                    "deconfounded_utility": float(
                        repeat_result["deconfounded_utility"]
                    ),
                }
            )
        if (
            repeat_index == 0
            or (repeat_index + 1) % 5 == 0
            or repeat_index + 1 == REPEATED_NESTED_CV_REPEATS
        ):
            print(
                f"[V7 STABILITY] repeat={repeat_index + 1}/"
                f"{REPEATED_NESTED_CV_REPEATS} completed.",
                flush=True,
            )
    write_csv(output_dir / "stability" / "repeated_nested_cv_runs.csv", repeated_rows)

    stability_summary_rows = []
    for experiment_id in stability_ids:
        rows = [row for row in repeated_rows if row["experiment_id"] == experiment_id]
        aucs = np.asarray([row["auc"] for row in rows], dtype=np.float64)
        dvus = np.asarray([row["deconfounded_utility"] for row in rows], dtype=np.float64)
        corrs = np.asarray(
            [row["within_class_max_abs_nuisance_correlation"] for row in rows],
            dtype=np.float64,
        )
        stability_summary_rows.append(
            {
                "experiment_id": experiment_id,
                "auc_mean": float(np.mean(aucs)),
                "auc_median": float(np.median(aucs)),
                "auc_std": float(np.std(aucs)),
                "auc_q25": float(np.quantile(aucs, 0.25)),
                "auc_q75": float(np.quantile(aucs, 0.75)),
                "auc_min": float(np.min(aucs)),
                "auc_max": float(np.max(aucs)),
                "dvu_median": float(np.median(dvus)),
                "nuisance_correlation_median": float(np.median(corrs)),
            }
        )
    stability_summary_rows.sort(
        key=lambda row: (-row["dvu_median"], -row["auc_median"])
    )
    write_csv(
        output_dir / "stability" / "repeated_nested_cv_summary.csv",
        stability_summary_rows,
    )

    # Candidate-only repeated-CV report with controls evaluated through the
    # exact same learner and pooling path. This is the preferred ranking table:
    # it combines stability, nuisance association, and the anatomical gap rather
    # than promoting a model from one favorable pooled OOF manifest.
    stability_summary_by_id = {
        row["experiment_id"]: row for row in stability_summary_rows
    }
    nuisance_only_stability = stability_summary_by_id[
        "C34_META_NUISANCE_ONLY_LR"
    ]
    candidate_stability_rows = []
    for candidate_id in V7_CANDIDATE_IDS:
        candidate = stability_summary_by_id[candidate_id]
        controls = [
            stability_summary_by_id[control_id]
            for control_id in MATCHED_CONTROL_IDS_BY_CANDIDATE[candidate_id]
        ]
        strongest_control = max(controls, key=lambda row: row["auc_median"])
        candidate_stability_rows.append(
            {
                **candidate,
                "strongest_matched_control_id": strongest_control[
                    "experiment_id"
                ],
                "strongest_matched_control_auc_median": float(
                    strongest_control["auc_median"]
                ),
                "candidate_minus_strongest_control_auc_median": float(
                    candidate["auc_median"] - strongest_control["auc_median"]
                ),
                "nuisance_only_control_auc_median": float(
                    nuisance_only_stability["auc_median"]
                ),
                "candidate_minus_nuisance_only_auc_median": float(
                    candidate["auc_median"]
                    - nuisance_only_stability["auc_median"]
                ),
            }
        )
    candidate_stability_rows.sort(
        key=lambda row: (
            -row["dvu_median"],
            -row["candidate_minus_strongest_control_auc_median"],
            -row["auc_median"],
            V7_CANDIDATE_SELECTION_PRIORITY.index(row["experiment_id"]),
        )
    )
    write_csv(
        output_dir / "stability" / "candidate_stability_summary.csv",
        candidate_stability_rows,
    )

    repeated_by_id_seed = {
        (row["experiment_id"], row["random_state"]): row for row in repeated_rows
    }
    repeated_paired_rows = []
    for candidate_id in V7_CANDIDATE_IDS:
        comparator_ids = (
            ("A17R_V6_REPRODUCED_HIER_LR_PCA",)
            if candidate_id != "A17R_V6_REPRODUCED_HIER_LR_PCA"
            else ()
        ) + MATCHED_CONTROL_IDS_BY_CANDIDATE[candidate_id] + (
            "C34_META_NUISANCE_ONLY_LR",
        )
        for reference_id in comparator_ids:
            deltas = []
            for repeat_index in range(REPEATED_NESTED_CV_REPEATS):
                seed = REPEATED_NESTED_CV_RANDOM_STATE + repeat_index
                deltas.append(
                    repeated_by_id_seed[(candidate_id, seed)]["auc"]
                    - repeated_by_id_seed[(reference_id, seed)]["auc"]
                )
            deltas_array = np.asarray(deltas, dtype=np.float64)
            repeated_paired_rows.append(
                {
                    "reference_experiment_id": reference_id,
                    "comparison_experiment_id": candidate_id,
                    "median_delta_auc": float(np.median(deltas_array)),
                    "delta_auc_q25": float(np.quantile(deltas_array, 0.25)),
                    "delta_auc_q75": float(np.quantile(deltas_array, 0.75)),
                    "fraction_comparison_better": float(np.mean(deltas_array > 0)),
                    "fraction_comparison_not_worse": float(np.mean(deltas_array >= 0)),
                    "minimum_delta_auc": float(np.min(deltas_array)),
                    "maximum_delta_auc": float(np.max(deltas_array)),
                }
            )
    write_csv(
        output_dir / "stability" / "same_seed_paired_deltas.csv",
        repeated_paired_rows,
    )

    # ------------------------ Nuisance-cluster stress audit -----------------------
    cluster_audits = {}
    if RUN_LEAVE_ONE_NUISANCE_CLUSTER_OUT_AUDIT:
        for candidate_id in V7_CANDIDATE_IDS:
            spec = next(spec for spec in MODEL_SPECS if spec.experiment_id == candidate_id)
            X, Z, y, patients = representations[candidate_id]
            audit = leave_one_nuisance_cluster_out_audit(spec, X, Z, y, patients)
            cluster_audits[candidate_id] = audit
            write_json(
                output_dir / "audits" / "nuisance_clusters" / f"{candidate_id}.json",
                audit,
            )
            write_csv(
                output_dir
                / "audits"
                / "nuisance_clusters"
                / f"{candidate_id}__patient_clusters.csv",
                audit["patient_clusters"],
            )

    # --------------------- Selection-adjusted permutation test --------------------
    candidate_specs = [
        next(spec for spec in MODEL_SPECS if spec.experiment_id == experiment_id)
        for experiment_id in V7_CANDIDATE_IDS
    ]
    observed_best_auc = max(summary_by_id[eid]["auc"] for eid in V7_CANDIDATE_IDS)
    if RUN_SELECTION_ADJUSTED_PERMUTATION:
        permutation_summary = run_selection_adjusted_permutation(
            candidate_specs,
            representations,
            patient_ids,
            labels,
            observed_best_auc,
            float(nested_candidate_selection["metrics"]["auc"]),
            output_dir / "permutation",
        )
    else:
        permutation_summary = {"status": "SKIPPED_DISABLED"}
        write_json(
            output_dir / "permutation" / "selection_adjusted_permutation_summary.json",
            permutation_summary,
        )

    # -------------------------------- Final report --------------------------------
    best_candidate = candidate_rows[0]
    report = {
        "status": "OK",
        "v6_cache_dir": str(cache_dir),
        "v6_suite_dir": None if suite_dir is None else str(suite_dir),
        "n_slices": int(len(bank["labels"])),
        "n_patients": int(len(patient_ids)),
        "normal_patients": int(np.sum(labels == 0)),
        "sick_patients": int(np.sum(labels == 1)),
        "nuisance_feature_count": int(nuisance.shape[1]),
        "prospective_v7_challenger": V7_PROSPECTIVE_CHALLENGER_ID,
        "v6_a17_reproduction_audit": reproduction_audit,
        "best_main_split_candidate_by_dvu": best_candidate,
        "nested_candidate_family_selection": nested_selection_summary,
        "candidate_summary": candidate_rows,
        "stability_summary": stability_summary_rows,
        "candidate_stability_summary": candidate_stability_rows,
        "best_repeated_cv_candidate_by_dvu": candidate_stability_rows[0],
        "same_seed_paired_deltas": repeated_paired_rows,
        "selection_adjusted_permutation": permutation_summary,
        "nuisance_cluster_audits": cluster_audits,
        "interpretation_rules": {
            "flno": (
                "Every nuisance scaler, PCA and ridge projection is fitted only "
                "inside the current training fold."
            ),
            "cmp_auc": (
                "Cross-class pairs receive larger loss weight when their training-"
                "fold nuisance profiles are similar."
            ),
            "dvu": (
                "DVU is an inner-selection criterion, not a clinical metric. It "
                "penalizes within-class score/nuisance association."
            ),
            "rmom": (
                "RMoM is label-blind and robust to unusual slice blocks, but it "
                "does not infer true acquisition order or sequence identity."
            ),
            "clinical": (
                "All results remain internal exploratory associations until "
                "sequence/view matching and independent external validation are completed."
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "final_report.json", report)

    print("\n" + "=" * 120, flush=True)
    print("V7 CANDIDATE SUMMARY — ORDERED BY MAIN-SPLIT DECONFOUNDED UTILITY", flush=True)
    print("=" * 120, flush=True)
    for rank, row in enumerate(candidate_rows, start=1):
        print(
            f"{rank:>2}. {row['experiment_id']:<48} "
            f"AUC={row['auc']:.4f}  DVU={row['deconfounded_utility']:.4f}  "
            f"nuisance_corr={row['within_class_max_abs_nuisance_correlation']:.4f}  "
            f"gap={row['candidate_minus_strongest_control_auc']:+.4f} "
            f"vs {row['strongest_matched_control_id']}",
            flush=True,
        )
    print("\nV7 REPEATED-CV SUMMARY — ORDERED BY MEDIAN DVU", flush=True)
    for rank, row in enumerate(candidate_stability_rows, start=1):
        print(
            f"{rank:>2}. {row['experiment_id']:<48} "
            f"median_AUC={row['auc_median']:.4f} "
            f"IQR=[{row['auc_q25']:.4f}, {row['auc_q75']:.4f}] "
            f"median_DVU={row['dvu_median']:.4f} "
            f"median_gap={row['candidate_minus_strongest_control_auc_median']:+.4f} "
            f"vs {row['strongest_matched_control_id']}",
            flush=True,
        )
    if permutation_summary.get("status") == "OK":
        print(
            "\n[V7 PERMUTATION] observed best AUC="
            f"{permutation_summary['observed_best_auc']:.4f}; max-null median="
            f"{permutation_summary['maximum_auc_null_median']:.4f}; family-wise p="
            f"{permutation_summary['empirical_familywise_max_auc_p_value']:.6f}; "
            f"nested-selection p="
            f"{permutation_summary['empirical_nested_selection_p_value']:.6f}",
            flush=True,
        )
    print(f"\n[V7] Outputs saved under: {output_dir}", flush=True)
    print(f"[V7] Total runtime: {format_elapsed(time.perf_counter() - started)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"[V7] FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
