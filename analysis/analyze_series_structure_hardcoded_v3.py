#!/usr/bin/env python3
"""
analyze_series_structure_hardcoded_v3.py

Structural analysis for a cardiac-MRI image dataset organized in folders such as:

    <root>/Sick/Directory_24/SR_3/IM00001.jpg
    <root>/Normal/Directory_10/series0032-Body/...
    <root>/Normal/Directory_10/1-short/series0032-Body/...
    <root>/Normal/Directory_10/42/1-short/series0032-Body/...

This version:
1. supports both SR_* and Series... folders;
2. labels one-image folders explicitly as SINGLE_IMAGE;
3. collapses exact physical copies into one LOGICAL series, but only when they have:
      - the same class (Sick/Normal/Healthy),
      - the same Directory_*,
      - the same normalized series name,
      - the exact same ordered image hashes;
4. detects stronger internal change-points and suppresses very short false segments;
5. searches for internal lag periodicity to flag likely SPATIOTEMPORAL series;
6. writes both physical-series and logical-series CSVs;
7. optionally generates montages for logical canonical series.

IMPORTANT
---------
This is a heuristic STRUCTURAL classifier from exported image pixels.
It is NOT a validated MR sequence classifier. It cannot reliably assign
LGE / T2 / SSFP / perfusion ground-truth without DICOM metadata or manual
validation.

Dependencies:
    pip install numpy opencv-python

Run:
    python analyze_series_structure_hardcoded_v3.py
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# HARDCODED CONFIGURATION
# =============================================================================

# Change this if necessary.
DATASET_ROOT = Path(r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset')

# None -> <DATASET_ROOT>\_series_analysis_v3
OUTPUT_DIR = None

# Generate contact sheets for canonical/logical series.
CREATE_MONTAGES = True

# Maximum number of tiles in one montage.
MONTAGE_MAX_TILES = 80

# =============================================================================
# IMAGE / HEURISTIC CONFIGURATION
# =============================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

PREPROCESS_SIZE = 192

# Series-structure similarity thresholds.
TEMPORAL_MEDIAN_SIM = 0.82
TEMPORAL_P10_SIM = 0.68
SPATIAL_MEDIAN_SIM = 0.58

# Exact-duplicate-heavy threshold inside a series.
DUPLICATE_HEAVY_FRACTION = 0.60

# Change-point detection.
CHANGE_Z_MAD = 2.75
MIN_CHANGE_DROP = 0.12
CHANGE_ABS_MAX_SIM = 0.74
MAX_MULTIPLANAR_CHANGEPOINTS = 5

# Periodicity search.
MIN_PERIOD = 2
MAX_PERIOD = 80
MIN_PERIOD_REPEATS = 3
PERIODICITY_MIN_SIM = 0.78
PERIODICITY_GAIN_OVER_ADJ = 0.08
PERIODICITY_GAIN_OVER_NEIGHBOR_LAGS = 0.04

# Classification helpers.
MIN_SPATIOTEMPORAL_IMAGES = 12


# =============================================================================
# BASIC HELPERS
# =============================================================================

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ],
        key=natural_key,
    )


def is_series_folder_name(name: str) -> bool:
    name_upper = name.strip().upper()
    return name_upper.startswith("SR_") or name_upper.startswith("SERIES")


def find_series(root: Path) -> list[Path]:
    out = []

    for current, dirs, files in os.walk(root):
        p = Path(current)

        if is_series_folder_name(p.name):
            if any(Path(f).suffix.lower() in IMAGE_EXTS for f in files):
                out.append(p)

    return sorted(out, key=lambda x: str(x).lower())


def parse_group_info(series_path: Path, root: Path) -> tuple[str, str, str]:
    try:
        rel = series_path.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = series_path.parts

    label = next(
        (p for p in parts if p.lower() in {"sick", "normal", "healthy"}),
        "",
    )

    directory = next(
        (p for p in parts if p.lower().startswith("directory_")),
        "",
    )

    return label, directory, series_path.name


def normalize_series_name(name: str) -> str:
    """
    Normalize only enough to match physical copies of the same logical series.

    Examples:
        Series0032-Body -> series0032-body
        series0032-Body -> series0032-body
        SR_3            -> sr_3
    """
    s = name.strip().lower()
    s = re.sub(r"\s+", "", s)
    return s


# =============================================================================
# HASHING / EXACT DUPLICATES
# =============================================================================

_FILE_HASH_CACHE: dict[str, str] = {}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    key = str(path.resolve())
    if key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[key]

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)

    digest = h.hexdigest()
    _FILE_HASH_CACHE[key] = digest
    return digest


def series_ordered_hash_signature(paths: list[Path]) -> str:
    """
    Exact signature of the ordered image sequence.
    Equal only if the same number of images appear in the same order
    with byte-identical file contents.
    """
    h = hashlib.sha256()

    for p in paths:
        digest = sha256_file(p)
        h.update(digest.encode("ascii"))
        h.update(b"\0")

    return h.hexdigest()


def duplicate_fraction_within_series(paths: list[Path]) -> tuple[int, int, float]:
    groups = defaultdict(list)

    for p in paths:
        groups[sha256_file(p)].append(p)

    dup_groups = [g for g in groups.values() if len(g) > 1]
    duplicate_images_after_first = sum(len(g) - 1 for g in dup_groups)
    frac = duplicate_images_after_first / max(1, len(paths))

    return duplicate_images_after_first, len(dup_groups), frac


# =============================================================================
# IMAGE SIMILARITY
# =============================================================================

def read_gray(path: Path, size: int = PREPROCESS_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.resize(
        img,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    lo, hi = np.percentile(img, [1, 99])

    if hi > lo:
        img = np.clip(
            (img.astype(np.float32) - lo) * 255.0 / (hi - lo),
            0,
            255,
        )
    else:
        img = img.astype(np.float32)

    return img.astype(np.uint8)


def equalized(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img)


def pearson_ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32).ravel()
    bb = b.astype(np.float32).ravel()

    aa -= aa.mean()
    bb -= bb.mean()

    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))

    if den < 1e-8:
        return 1.0 if np.allclose(a, b) else 0.0

    return float(np.dot(aa, bb) / den)


def align_translation(
    ref: np.ndarray,
    mov: np.ndarray,
    max_shift: float = 12.0,
) -> np.ndarray:
    a = ref.astype(np.float32)
    b = mov.astype(np.float32)

    try:
        (dx, dy), response = cv2.phaseCorrelate(a, b)
    except cv2.error:
        return mov

    if not np.isfinite(dx) or not np.isfinite(dy):
        return mov

    dx = float(np.clip(dx, -max_shift, max_shift))
    dy = float(np.clip(dy, -max_shift, max_shift))

    M = np.float32([
        [1, 0, -dx],
        [0, 1, -dy],
    ])

    return cv2.warpAffine(
        mov,
        M,
        (mov.shape[1], mov.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Pixel-structure similarity in [0, 1].

    Uses:
      - histogram-equalized image correlation;
      - edge correlation;
      - small global translation compensation.
    """
    a = equalized(img1)
    b = equalized(img2)

    b = align_translation(a, b)

    corr = pearson_ncc(a, b)
    corr01 = (corr + 1.0) / 2.0

    ea = cv2.Canny(a, 40, 120)
    eb = cv2.Canny(b, 40, 120)

    edge_corr = pearson_ncc(ea, eb)
    edge01 = (edge_corr + 1.0) / 2.0

    score = 0.70 * corr01 + 0.30 * edge01

    return float(np.clip(score, 0.0, 1.0))


def center_intensity(img: np.ndarray) -> float:
    h, w = img.shape

    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4

    return float(img[y0:y1, x0:x1].mean())


# =============================================================================
# CHANGE-POINT DETECTION
# =============================================================================

def min_segment_len_for_n(n: int) -> int:
    """
    Dynamic minimum segment length.

    For n=57 -> 6, so a spurious 4-frame segment such as the old
    20;24;40 result is suppressed.
    """
    if n < 12:
        return 2

    return max(4, min(10, int(round(n * 0.10))))


def raw_change_candidates(sims: np.ndarray) -> list[int]:
    """
    sims[i] compares image i with image i+1.
    Returns zero-based index of the NEW segment start.
    """
    if len(sims) < 3:
        return []

    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med)))

    drop = max(MIN_CHANGE_DROP, CHANGE_Z_MAD * mad)
    threshold = min(CHANGE_ABS_MAX_SIM, med - drop)

    return [
        i + 1
        for i, s in enumerate(sims)
        if s < threshold
    ]


def suppress_close_change_points(
    sims: np.ndarray,
    candidates: list[int],
    n_images: int,
) -> list[int]:
    """
    Non-maximum suppression with a dynamic minimum segment length.

    Keeps the deepest break in a neighborhood and avoids tiny segments.
    """
    if not candidates:
        return []

    min_seg = min_segment_len_for_n(n_images)

    # Ignore breaks too close to edges.
    candidates = [
        cp for cp in candidates
        if cp >= min_seg and (n_images - cp) >= min_seg
    ]

    if not candidates:
        return []

    # Strongest break first = lowest adjacent similarity.
    ranked = sorted(
        candidates,
        key=lambda cp: sims[cp - 1],
    )

    selected = []

    for cp in ranked:
        if all(abs(cp - other) >= min_seg for other in selected):
            selected.append(cp)

    selected.sort()

    # One more pass: ensure all resulting segments meet min length.
    changed = True

    while changed and selected:
        changed = False
        bounds = [0] + selected + [n_images]
        lengths = [
            b - a
            for a, b in zip(bounds[:-1], bounds[1:])
        ]

        bad_index = next(
            (i for i, L in enumerate(lengths) if L < min_seg),
            None,
        )

        if bad_index is None:
            break

        # Remove one of the two neighboring CPs; keep the stronger break.
        if bad_index == 0:
            selected.pop(0)
        elif bad_index == len(lengths) - 1:
            selected.pop(-1)
        else:
            left_cp_idx = bad_index - 1
            right_cp_idx = bad_index

            left_cp = selected[left_cp_idx]
            right_cp = selected[right_cp_idx]

            left_strength = sims[left_cp - 1]
            right_strength = sims[right_cp - 1]

            # Larger similarity = weaker break -> remove it.
            if left_strength >= right_strength:
                selected.pop(left_cp_idx)
            else:
                selected.pop(right_cp_idx)

        changed = True

    return selected


def robust_change_points(sims: np.ndarray, n_images: int) -> list[int]:
    candidates = raw_change_candidates(sims)

    return suppress_close_change_points(
        sims=sims,
        candidates=candidates,
        n_images=n_images,
    )


def segment_lengths(n: int, cps: list[int]) -> list[int]:
    bounds = [0] + cps + [n]

    return [
        b - a
        for a, b in zip(bounds[:-1], bounds[1:])
    ]


# =============================================================================
# PERIODICITY / SPATIOTEMPORAL HEURISTIC
# =============================================================================

@dataclass
class PeriodicityResult:
    best_period: int
    best_similarity: float
    adjacent_similarity: float
    neighbor_lag_similarity: float
    periodicity_gain_over_adjacent: float
    periodicity_gain_over_neighbor_lags: float
    repeats: float
    is_strong: bool


def lag_similarity(
    imgs: list[np.ndarray],
    lag: int,
    max_pairs: int = 80,
) -> float:
    """
    Median similarity between frame i and frame i+lag.
    Samples pairs evenly for speed on long series.
    """
    n = len(imgs)
    count = n - lag

    if count <= 0:
        return float("nan")

    if count <= max_pairs:
        starts = np.arange(count)
    else:
        starts = np.linspace(
            0,
            count - 1,
            max_pairs,
        ).round().astype(int)

    vals = [
        structural_similarity(imgs[i], imgs[i + lag])
        for i in starts
    ]

    return float(np.median(vals))


def detect_periodicity(
    imgs: list[np.ndarray],
    adjacent_median: float,
) -> PeriodicityResult:
    n = len(imgs)

    empty = PeriodicityResult(
        best_period=0,
        best_similarity=0.0,
        adjacent_similarity=adjacent_median,
        neighbor_lag_similarity=0.0,
        periodicity_gain_over_adjacent=0.0,
        periodicity_gain_over_neighbor_lags=0.0,
        repeats=0.0,
        is_strong=False,
    )

    if n < MIN_SPATIOTEMPORAL_IMAGES:
        return empty

    max_period = min(
        MAX_PERIOD,
        n // MIN_PERIOD_REPEATS,
    )

    if max_period < MIN_PERIOD:
        return empty

    lag_scores: dict[int, float] = {}

    for lag in range(MIN_PERIOD, max_period + 1):
        lag_scores[lag] = lag_similarity(imgs, lag)

    if not lag_scores:
        return empty

    best_period = max(
        lag_scores,
        key=lambda k: lag_scores[k],
    )

    best_sim = lag_scores[best_period]

    neighbor_scores = [
        lag_scores[k]
        for k in (
            best_period - 2,
            best_period - 1,
            best_period + 1,
            best_period + 2,
        )
        if k in lag_scores
    ]

    neighbor_med = (
        float(np.median(neighbor_scores))
        if neighbor_scores
        else 0.0
    )

    gain_adj = best_sim - adjacent_median
    gain_neighbor = best_sim - neighbor_med
    repeats = n / best_period

    strong = (
        repeats >= MIN_PERIOD_REPEATS
        and best_sim >= PERIODICITY_MIN_SIM
        and (
            gain_adj >= PERIODICITY_GAIN_OVER_ADJ
            or gain_neighbor >= PERIODICITY_GAIN_OVER_NEIGHBOR_LAGS
        )
    )

    return PeriodicityResult(
        best_period=int(best_period),
        best_similarity=float(best_sim),
        adjacent_similarity=float(adjacent_median),
        neighbor_lag_similarity=float(neighbor_med),
        periodicity_gain_over_adjacent=float(gain_adj),
        periodicity_gain_over_neighbor_lags=float(gain_neighbor),
        repeats=float(repeats),
        is_strong=bool(strong),
    )


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_series(
    n: int,
    sims: np.ndarray,
    cps: list[int],
    dup_fraction: float,
    center_curve: np.ndarray,
    periodicity: PeriodicityResult,
) -> tuple[str, str]:
    if n == 1:
        return "SINGLE_IMAGE", "exactly_one_image"

    if n == 2:
        return "HETEROGENEOUS_OR_UNKNOWN", "only_two_images"

    if dup_fraction >= DUPLICATE_HEAVY_FRACTION:
        return (
            "STATIC_OR_DUPLICATE_HEAVY",
            f"exact_duplicate_fraction={dup_fraction:.3f}",
        )

    med = float(np.median(sims)) if len(sims) else 0.0
    p10 = float(np.percentile(sims, 10)) if len(sims) else 0.0

    lengths = segment_lengths(n, cps)
    min_seg = min_segment_len_for_n(n)

    valid_segments = (
        bool(lengths)
        and all(L >= min_seg for L in lengths)
    )

    # Strong internal periodicity is a useful clue that one physical folder
    # may contain >1 dimension (e.g. slice × phase/time).
    if periodicity.is_strong:
        return (
            "SPATIOTEMPORAL_LIKELY",
            (
                f"period={periodicity.best_period}; "
                f"period_sim={periodicity.best_similarity:.3f}; "
                f"gain_adj={periodicity.periodicity_gain_over_adjacent:.3f}; "
                f"gain_neighbor={periodicity.periodicity_gain_over_neighbor_lags:.3f}; "
                f"repeats={periodicity.repeats:.2f}"
            ),
        )

    # A small number of strong discontinuities separating coherent blocks
    # suggests multiplanar scout/localizer or a mixed spatial series.
    if (
        1 <= len(cps) <= MAX_MULTIPLANAR_CHANGEPOINTS
        and valid_segments
    ):
        cp_sims = np.array(
            [sims[c - 1] for c in cps],
            dtype=float,
        )

        non_cp_mask = np.ones(
            len(sims),
            dtype=bool,
        )

        for c in cps:
            non_cp_mask[c - 1] = False

        within = sims[non_cp_mask]

        within_med = (
            float(np.median(within))
            if len(within)
            else med
        )

        cp_med = (
            float(np.median(cp_sims))
            if len(cp_sims)
            else med
        )

        if within_med - cp_med >= 0.15:
            return (
                "MULTIPLANAR_OR_MIXED_SPATIAL",
                (
                    f"{len(cps)}_strong_change_points; "
                    f"within_med={within_med:.3f}; "
                    f"cp_med={cp_med:.3f}"
                ),
            )

    # Temporal-looking: geometry remains very similar throughout.
    if (
        med >= TEMPORAL_MEDIAN_SIM
        and p10 >= TEMPORAL_P10_SIM
        and len(cps) == 0
    ):
        mean = (
            float(np.mean(center_curve))
            if len(center_curve)
            else 0.0
        )

        rel_range = (
            float(np.ptp(center_curve)) / max(mean, 1e-6)
            if len(center_curve)
            else 0.0
        )

        return (
            "TEMPORAL_LIKELY",
            (
                f"median_adj_sim={med:.3f}; "
                f"p10={p10:.3f}; "
                f"center_rel_range={rel_range:.3f}"
            ),
        )

    # Spatial stack: neighboring anatomy is related but evolves progressively.
    if med >= SPATIAL_MEDIAN_SIM and len(cps) <= 1:
        return (
            "SPATIAL_STACK_LIKELY",
            (
                f"median_adj_sim={med:.3f}; "
                f"p10={p10:.3f}; "
                f"change_points={len(cps)}"
            ),
        )

    return (
        "HETEROGENEOUS_OR_UNKNOWN",
        (
            f"median_adj_sim={med:.3f}; "
            f"p10={p10:.3f}; "
            f"change_points={len(cps)}"
        ),
    )


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class SeriesResult:
    path: str
    label: str
    directory: str
    series_name: str

    class_label: str
    reason: str

    n_images: int

    exact_dup_images_after_first: int
    exact_dup_groups: int
    exact_dup_fraction: float

    median_adj_similarity: float
    p10_adj_similarity: float
    p90_adj_similarity: float
    min_adj_similarity: float
    max_adj_similarity: float

    change_points_1based: str
    segment_lengths: str

    center_intensity_rel_range: float

    best_period: int
    best_period_similarity: float
    best_period_repeats: float
    periodicity_gain_over_adjacent: float
    periodicity_gain_over_neighbor_lags: float
    strong_periodicity: bool

    ordered_series_hash: str


# =============================================================================
# SERIES ANALYSIS
# =============================================================================

def analyze_one_series(
    series_path: Path,
    root: Path,
) -> tuple[
    SeriesResult,
    list[float],
    list[int],
    list[Path],
]:
    paths = list_images(series_path)

    imgs = []
    good_paths = []

    for p in paths:
        try:
            imgs.append(read_gray(p))
            good_paths.append(p)
        except Exception as e:
            print(f"[WARN] {e}")

    paths = good_paths
    n = len(paths)

    if n == 0:
        raise ValueError(
            f"No readable images in {series_path}"
        )

    label, directory, series_name = parse_group_info(
        series_path,
        root,
    )

    ordered_hash = series_ordered_hash_signature(paths)

    (
        dup_after_first,
        dup_groups,
        dup_frac,
    ) = duplicate_fraction_within_series(paths)

    sims = [
        structural_similarity(imgs[i], imgs[i + 1])
        for i in range(n - 1)
    ]

    sims_arr = np.array(sims, dtype=float)

    cps = robust_change_points(
        sims=sims_arr,
        n_images=n,
    )

    ccurve = np.array(
        [center_intensity(im) for im in imgs],
        dtype=float,
    )

    med = (
        float(np.median(sims_arr))
        if len(sims_arr)
        else 0.0
    )

    periodicity = detect_periodicity(
        imgs=imgs,
        adjacent_median=med,
    )

    class_label, reason = classify_series(
        n=n,
        sims=sims_arr,
        cps=cps,
        dup_fraction=dup_frac,
        center_curve=ccurve,
        periodicity=periodicity,
    )

    p10 = (
        float(np.percentile(sims_arr, 10))
        if len(sims_arr)
        else 0.0
    )

    p90 = (
        float(np.percentile(sims_arr, 90))
        if len(sims_arr)
        else 0.0
    )

    mn = (
        float(np.min(sims_arr))
        if len(sims_arr)
        else 0.0
    )

    mx = (
        float(np.max(sims_arr))
        if len(sims_arr)
        else 0.0
    )

    mean_center = float(np.mean(ccurve))

    rel_range = (
        float(np.ptp(ccurve)) / max(mean_center, 1e-6)
        if len(ccurve)
        else 0.0
    )

    # cp zero-based = NEW segment start.
    # Human output is 1-based image number.
    cps_1based = [c + 1 for c in cps]

    result = SeriesResult(
        path=str(series_path),
        label=label,
        directory=directory,
        series_name=series_name,

        class_label=class_label,
        reason=reason,

        n_images=n,

        exact_dup_images_after_first=dup_after_first,
        exact_dup_groups=dup_groups,
        exact_dup_fraction=dup_frac,

        median_adj_similarity=med,
        p10_adj_similarity=p10,
        p90_adj_similarity=p90,
        min_adj_similarity=mn,
        max_adj_similarity=mx,

        change_points_1based=";".join(
            map(str, cps_1based)
        ),

        segment_lengths=";".join(
            map(str, segment_lengths(n, cps))
        ),

        center_intensity_rel_range=rel_range,

        best_period=periodicity.best_period,
        best_period_similarity=periodicity.best_similarity,
        best_period_repeats=periodicity.repeats,
        periodicity_gain_over_adjacent=(
            periodicity.periodicity_gain_over_adjacent
        ),
        periodicity_gain_over_neighbor_lags=(
            periodicity.periodicity_gain_over_neighbor_lags
        ),
        strong_periodicity=periodicity.is_strong,

        ordered_series_hash=ordered_hash,
    )

    return result, sims, cps, paths


# =============================================================================
# LOGICAL SERIES COLLAPSING
# =============================================================================

def logical_group_key(
    result: SeriesResult,
) -> tuple[str, str, str, str]:
    """
    Conservative logical deduplication.

    We collapse only if ALL of these match:
      class + Directory_* + normalized series basename + exact ordered content.

    This avoids accidentally merging identical-looking or byte-identical
    acquisitions belonging to different Directory_* groups.
    """
    return (
        result.label.lower(),
        result.directory.lower(),
        normalize_series_name(result.series_name),
        result.ordered_series_hash,
    )


def canonical_path(
    paths: list[str],
    root: Path,
) -> str:
    """
    Prefer the shallowest path under the dataset root, then lexical order.
    """
    def key(s: str):
        p = Path(s)

        try:
            rel = p.relative_to(root)
            depth = len(rel.parts)
        except ValueError:
            depth = len(p.parts)

        return (depth, len(s), s.lower())

    return min(paths, key=key)


def collapse_logical_series(
    results: list[SeriesResult],
    root: Path,
) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)

    for r in results:
        groups[logical_group_key(r)].append(r)

    logical_rows = []
    duplicate_rows = []

    logical_id = 0

    for key, group in sorted(
        groups.items(),
        key=lambda kv: canonical_path(
            [x.path for x in kv[1]],
            root,
        ).lower(),
    ):
        logical_id += 1

        paths = [r.path for r in group]
        canonical = canonical_path(paths, root)

        representative = next(
            r for r in group
            if r.path == canonical
        )

        row = asdict(representative)

        row.update({
            "logical_series_id": logical_id,
            "canonical_path": canonical,
            "physical_copy_count": len(group),
            "all_physical_paths": " | ".join(
                sorted(paths)
            ),
            "is_collapsed_duplicate": len(group) > 1,
        })

        logical_rows.append(row)

        if len(group) > 1:
            duplicate_rows.append({
                "logical_series_id": logical_id,
                "label": representative.label,
                "directory": representative.directory,
                "normalized_series_name": (
                    normalize_series_name(
                        representative.series_name
                    )
                ),
                "ordered_series_hash": (
                    representative.ordered_series_hash
                ),
                "physical_copy_count": len(group),
                "canonical_path": canonical,
                "all_physical_paths": " | ".join(
                    sorted(paths)
                ),
            })

    return logical_rows, duplicate_rows


# =============================================================================
# MONTAGES
# =============================================================================

def make_montage(
    paths: list[Path],
    out_path: Path,
    max_tiles: int = MONTAGE_MAX_TILES,
    tile_w: int = 160,
):
    if not paths:
        return

    if len(paths) > max_tiles:
        idx = np.linspace(
            0,
            len(paths) - 1,
            max_tiles,
        ).round().astype(int)

        shown = [paths[i] for i in idx]
    else:
        shown = paths

    tiles = []

    for p in shown:
        img = cv2.imread(
            str(p),
            cv2.IMREAD_GRAYSCALE,
        )

        if img is None:
            continue

        h, w = img.shape
        scale = tile_w / max(1, w)
        nh = max(
            1,
            int(round(h * scale)),
        )

        tile = cv2.resize(
            img,
            (tile_w, nh),
            interpolation=cv2.INTER_AREA,
        )

        tile = cv2.cvtColor(
            tile,
            cv2.COLOR_GRAY2BGR,
        )

        label = p.stem

        cv2.rectangle(
            tile,
            (0, 0),
            (min(tile_w, 105), 18),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            tile,
            label,
            (3, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        tiles.append(tile)

    if not tiles:
        return

    max_h = max(
        t.shape[0]
        for t in tiles
    )

    norm = []

    for t in tiles:
        if t.shape[0] < max_h:
            pad = np.zeros(
                (
                    max_h - t.shape[0],
                    t.shape[1],
                    3,
                ),
                dtype=np.uint8,
            )

            t = np.vstack([t, pad])

        norm.append(t)

    cols = min(
        8,
        max(
            1,
            math.ceil(
                math.sqrt(
                    len(norm) * 1.6
                )
            ),
        ),
    )

    rows = math.ceil(
        len(norm) / cols
    )

    blank = np.zeros_like(norm[0])

    while len(norm) < rows * cols:
        norm.append(blank.copy())

    row_imgs = []

    for r in range(rows):
        row_imgs.append(
            np.hstack(
                norm[
                    r * cols:
                    (r + 1) * cols
                ]
            )
        )

    montage = np.vstack(row_imgs)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(out_path),
        montage,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )


# =============================================================================
# GLOBAL EXACT DUPLICATES
# =============================================================================

def write_global_duplicates(
    root: Path,
    series_dirs: list[Path],
    out_txt: Path,
):
    groups = defaultdict(list)
    total = 0

    for sr in series_dirs:
        for p in list_images(sr):
            total += 1

            try:
                groups[sha256_file(p)].append(p)
            except OSError as e:
                print(
                    f"[WARN] hash failed: {p}: {e}"
                )

    dup_groups = [
        g for g in groups.values()
        if len(g) > 1
    ]

    dup_groups.sort(
        key=lambda g: (
            -len(g),
            str(g[0]).lower(),
        )
    )

    with out_txt.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"Total images scanned: {total}\n"
        )

        f.write(
            f"Exact duplicate groups: "
            f"{len(dup_groups)}\n"
        )

        f.write(
            "Images involved in duplicate groups: "
            f"{sum(len(g) for g in dup_groups)}\n\n"
        )

        for i, g in enumerate(
            dup_groups,
            1,
        ):
            f.write(
                f"DUPLICATE GROUP {i} "
                f"(n={len(g)})\n"
            )

            for p in g:
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p

                f.write(
                    f"  {rel}\n"
                )

            f.write("\n")

    return (
        len(dup_groups),
        sum(len(g) for g in dup_groups),
    )


# =============================================================================
# CSV WRITING
# =============================================================================

def write_dict_rows(
    path: Path,
    rows: list[dict],
):
    if not rows:
        path.write_text(
            "",
            encoding="utf-8-sig",
        )
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        w.writeheader()
        w.writerows(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = DATASET_ROOT.expanduser().resolve()

    if not root.exists():
        raise SystemExit(
            f"Dataset root does not exist:\n"
            f"  {root}\n\n"
            "Edit DATASET_ROOT near the top "
            "of the script."
        )

    if OUTPUT_DIR is None:
        out_dir = (
            root / "_series_analysis_v3"
        )
    else:
        out_dir = (
            Path(OUTPUT_DIR)
            .expanduser()
            .resolve()
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== CONFIGURATION ===")
    print(f"Dataset root : {root}")
    print(f"Output dir   : {out_dir}")
    print(f"Montages     : {CREATE_MONTAGES}")
    print()

    series_dirs = find_series(root)

    print(
        "Found SR_* / Series... folders: "
        f"{len(series_dirs)}"
    )

    if not series_dirs:
        raise SystemExit(
            "No SR_* or Series... folders "
            "with images were found."
        )

    # -------------------------------------------------------------------------
    # Global exact duplicate report
    # -------------------------------------------------------------------------
    dup_txt = (
        out_dir
        / "global_exact_duplicate_groups.txt"
    )

    (
        n_dup_groups,
        n_dup_images,
    ) = write_global_duplicates(
        root,
        series_dirs,
        dup_txt,
    )

    print(
        "Global exact duplicate groups: "
        f"{n_dup_groups}"
    )

    print(
        "Images involved in duplicate groups: "
        f"{n_dup_images}"
    )

    # -------------------------------------------------------------------------
    # Analyze every PHYSICAL series folder
    # -------------------------------------------------------------------------
    results: list[SeriesResult] = []
    cp_rows = []

    for idx, sr in enumerate(
        series_dirs,
        1,
    ):
        try:
            (
                result,
                sims,
                cps,
                paths,
            ) = analyze_one_series(
                sr,
                root,
            )

            results.append(result)

            for cp in cps:
                cp_rows.append({
                    "series_path": str(sr),
                    "label": result.label,
                    "directory": result.directory,
                    "series": result.series_name,

                    # cp=19 means break between image #19 and #20.
                    "break_between_image_numbers": (
                        f"{cp}-{cp + 1}"
                    ),

                    "new_segment_starts_at_image_number": (
                        cp + 1
                    ),

                    "similarity_across_break": (
                        sims[cp - 1]
                        if cp - 1 < len(sims)
                        else ""
                    ),
                })

            print(
                f"[{idx:4d}/{len(series_dirs)}] "
                f"{result.class_label:30s} "
                f"n={result.n_images:4d} "
                f"med={result.median_adj_similarity:.3f} "
                f"period={result.best_period:3d} "
                f"cp={result.change_points_1based or '-':12s} "
                f"{sr.relative_to(root)}"
            )

        except Exception as e:
            print(
                f"[ERROR] {sr}: {e}"
            )

    # -------------------------------------------------------------------------
    # Collapse exact redundant folder copies to LOGICAL series
    # -------------------------------------------------------------------------
    (
        logical_rows,
        duplicate_series_rows,
    ) = collapse_logical_series(
        results,
        root,
    )

    # -------------------------------------------------------------------------
    # Generate montages only once per LOGICAL series
    # -------------------------------------------------------------------------
    if CREATE_MONTAGES:
        montage_dir = (
            out_dir / "montages_logical"
        )

        for i, row in enumerate(
            logical_rows,
            1,
        ):
            canonical = Path(
                row["canonical_path"]
            )

            paths = list_images(canonical)

            try:
                rel = canonical.relative_to(root)
                safe = "__".join(rel.parts)
            except ValueError:
                safe = canonical.name

            make_montage(
                paths,
                montage_dir / f"{safe}.jpg",
            )

            if i % 100 == 0:
                print(
                    f"[montages] {i}/"
                    f"{len(logical_rows)}"
                )

    # -------------------------------------------------------------------------
    # Write CSV files
    # -------------------------------------------------------------------------
    physical_csv = (
        out_dir
        / "series_analysis_physical.csv"
    )

    write_dict_rows(
        physical_csv,
        [asdict(r) for r in results],
    )

    logical_csv = (
        out_dir
        / "series_analysis_logical.csv"
    )

    write_dict_rows(
        logical_csv,
        logical_rows,
    )

    cp_csv = (
        out_dir
        / "series_change_points.csv"
    )

    write_dict_rows(
        cp_csv,
        cp_rows,
    )

    duplicate_series_csv = (
        out_dir
        / "series_collapsed_duplicates.csv"
    )

    write_dict_rows(
        duplicate_series_csv,
        duplicate_series_rows,
    )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    physical_counts = defaultdict(int)

    for r in results:
        physical_counts[
            r.class_label
        ] += 1

    logical_counts = defaultdict(int)

    for row in logical_rows:
        logical_counts[
            row["class_label"]
        ] += 1

    print("\n=== PHYSICAL SERIES SUMMARY ===")

    for k in sorted(physical_counts):
        print(
            f"{k:30s}: "
            f"{physical_counts[k]}"
        )

    print("\n=== LOGICAL SERIES SUMMARY ===")

    for k in sorted(logical_counts):
        print(
            f"{k:30s}: "
            f"{logical_counts[k]}"
        )

    physical_n = len(results)
    logical_n = len(logical_rows)

    print("\n=== DEDUPLICATION SUMMARY ===")
    print(
        f"Physical series folders : "
        f"{physical_n}"
    )
    print(
        f"Logical series          : "
        f"{logical_n}"
    )
    print(
        f"Collapsed folder copies : "
        f"{physical_n - logical_n}"
    )
    print(
        f"Duplicate logical groups: "
        f"{len(duplicate_series_rows)}"
    )

    print("\nOutputs:")
    print(f"  {physical_csv}")
    print(f"  {logical_csv}")
    print(f"  {cp_csv}")
    print(f"  {duplicate_series_csv}")
    print(f"  {dup_txt}")

    if CREATE_MONTAGES:
        print(
            f"  {out_dir / 'montages_logical'}"
        )

    print(
        "\nNOTE: Labels are heuristic structural "
        "labels derived from exported image pixels. "
        "Manually validate a representative subset "
        "before using them as model ground truth."
    )


if __name__ == "__main__":
    main()
