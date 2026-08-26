#!/usr/bin/env python3
"""
analyze_series_structure.py

Heuristic structural analysis for a JPEG/PNG cardiac-MRI dataset organized as:
    <root>/Sick/Directory_XX/SR_YY/IM00001.jpg
    <root>/Normal/Directory_XX/SR_YY/IM00001.jpg

Goal:
- find all SR_* and Series... folders recursively
- detect exact duplicates globally
- estimate whether each SR_* is:
    TEMPORAL_LIKELY
    SPATIAL_STACK_LIKELY
    MULTIPLANAR_OR_MIXED_SPATIAL
    STATIC_OR_DUPLICATE_HEAVY
    HETEROGENEOUS_OR_UNKNOWN
- find likely abrupt change-points inside a series
- export CSV summaries and optional contact sheets

Important:
This classifies SERIES STRUCTURE, not MR pulse sequence.
From JPEG pixels alone, LGE vs T2 vs SSFP vs perfusion cannot be identified
reliably enough for automatic ground truth without DICOM metadata or a
separately validated sequence classifier.

Dependencies:
    pip install numpy opencv-python

Configuration:
Edit the hardcoded variables near the top of the file:
    DATASET_ROOT
    OUTPUT_DIR
    CREATE_MONTAGES

Then run:
    python analyze_series_structure_hardcoded.py

Outputs:
    <output>/series_analysis.csv
    <output>/series_change_points.csv
    <output>/global_exact_duplicate_groups.txt
    <output>/montages/*.jpg          (with --montages)
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Heuristic thresholds. These are deliberately exposed near the top so they
# can be calibrated after manually checking a few dozen series.
TEMPORAL_MEDIAN_SIM = 0.82
TEMPORAL_P10_SIM = 0.68
SPATIAL_MEDIAN_SIM = 0.58

MIN_CHANGE_DROP = 0.12
CHANGE_ABS_MAX_SIM = 0.72
MIN_SEGMENT_LEN = 4
MAX_MULTIPLANAR_CHANGEPOINTS = 5

PREPROCESS_SIZE = 192


# =============================================================================
# HARDCODED CONFIGURATION
# =============================================================================

# Dataset root. Change this path if your dataset is elsewhere.
DATASET_ROOT = Path(r'C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset')

# Output directory. Leave as None to use:
#     <DATASET_ROOT>\_series_analysis
OUTPUT_DIR = None

# True  -> generate contact-sheet montages for every SR_* folder.
# False -> skip montage generation.
CREATE_MONTAGES = True



def natural_key(path: Path):
    """Natural sort: IM2 before IM10."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def is_series_folder_name(name: str) -> bool:
    """
    Accept both naming conventions found in the dataset, case-insensitively:

        SR_1
        SR_123
        Series1
        Series_1
        Series 1
        series0001
        series-anything

    In other words:
      - names beginning with "SR_"
      - names beginning with "SERIES"
    """
    name_upper = name.strip().upper()
    return name_upper.startswith("SR_") or name_upper.startswith("SERIES")


def find_series(root: Path) -> list[Path]:
    """
    Find all image-containing series folders recursively.

    Supported folder names:
        SR_*
        Series...
    """
    out = []

    for current, dirs, files in os.walk(root):
        p = Path(current)

        if is_series_folder_name(p.name):
            if any(Path(f).suffix.lower() in IMAGE_EXTS for f in files):
                out.append(p)

    return sorted(out, key=lambda x: str(x).lower())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_gray(path: Path, size: int = PREPROCESS_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    # Resize to a fixed square. This is for structural comparison only.
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Suppress black borders / background dominance by clipping to body-ish ROI.
    # We still keep most of the field of view.
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo:
        img = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
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


def align_translation(ref: np.ndarray, mov: np.ndarray, max_shift: float = 12.0) -> np.ndarray:
    """Correct small global x/y shifts before comparing two frames."""
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

    # phaseCorrelate returns the shift from src1 to src2;
    # shift moving image back by the negative translation.
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(
        mov,
        M,
        (mov.shape[1], mov.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


def structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Robust-ish structure similarity in [0, 1]:
    - histogram-equalized intensity correlation
    - edge correlation
    after small translational alignment.

    High score -> likely same geometry / neighboring anatomy.
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
    """Mean of the central 50% FOV, useful as a rough temporal intensity metric."""
    h, w = img.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    return float(img[y0:y1, x0:x1].mean())


def robust_change_points(sims: np.ndarray) -> list[int]:
    """
    sims[i] compares image i with i+1.
    Return image index where the NEW segment begins, 0-based.
    Example cp=19 means break between image 18 and 19.
    """
    if len(sims) < 3:
        return []

    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med)))

    # Need both a relative drop and an absolute low-enough similarity.
    drop = max(MIN_CHANGE_DROP, 2.5 * mad)
    threshold = min(CHANGE_ABS_MAX_SIM, med - drop)

    cps = [i + 1 for i, s in enumerate(sims) if s < threshold]

    # Merge adjacent detections, retaining the deepest local break.
    if not cps:
        return []

    merged = []
    group = [cps[0]]
    for cp in cps[1:]:
        if cp - group[-1] <= 2:
            group.append(cp)
        else:
            best = min(group, key=lambda c: sims[c - 1])
            merged.append(best)
            group = [cp]
    best = min(group, key=lambda c: sims[c - 1])
    merged.append(best)

    return merged


def segment_lengths(n: int, cps: list[int]) -> list[int]:
    bounds = [0] + cps + [n]
    return [b - a for a, b in zip(bounds[:-1], bounds[1:])]


def duplicate_fraction_within_series(paths: list[Path]) -> tuple[int, int, float]:
    """
    Returns:
        duplicate_images_after_first,
        number_of_duplicate_groups,
        duplicate_fraction
    """
    groups = defaultdict(list)
    for p in paths:
        groups[sha256_file(p)].append(p)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    duplicate_images_after_first = sum(len(g) - 1 for g in dup_groups)
    frac = duplicate_images_after_first / max(1, len(paths))
    return duplicate_images_after_first, len(dup_groups), frac


def classify_series(
    n: int,
    sims: np.ndarray,
    cps: list[int],
    dup_fraction: float,
    center_curve: np.ndarray,
) -> tuple[str, str]:
    """
    Classify structural organization only.
    Returns (class_name, reason).
    """
    if n <= 2:
        return "HETEROGENEOUS_OR_UNKNOWN", "too_few_images"

    if dup_fraction >= 0.60:
        return "STATIC_OR_DUPLICATE_HEAVY", f"exact_duplicate_fraction={dup_fraction:.3f}"

    med = float(np.median(sims)) if len(sims) else 0.0
    p10 = float(np.percentile(sims, 10)) if len(sims) else 0.0

    lengths = segment_lengths(n, cps)
    valid_segments = all(L >= MIN_SEGMENT_LEN for L in lengths) if lengths else False

    # A few strong discontinuities separating coherent blocks strongly suggests
    # multiplanar scout/localizer or a mixed series.
    if (
        1 <= len(cps) <= MAX_MULTIPLANAR_CHANGEPOINTS
        and valid_segments
    ):
        cp_sims = np.array([sims[c - 1] for c in cps], dtype=float)
        non_cp_mask = np.ones(len(sims), dtype=bool)
        for c in cps:
            non_cp_mask[c - 1] = False
        within = sims[non_cp_mask]
        within_med = float(np.median(within)) if len(within) else med
        cp_med = float(np.median(cp_sims)) if len(cp_sims) else med

        if within_med - cp_med >= 0.15:
            return (
                "MULTIPLANAR_OR_MIXED_SPATIAL",
                f"{len(cps)}_strong_change_points; within_med={within_med:.3f}; cp_med={cp_med:.3f}",
            )

    # Likely temporal: the global anatomy remains highly similar over time.
    if med >= TEMPORAL_MEDIAN_SIM and p10 >= TEMPORAL_P10_SIM and len(cps) == 0:
        # Center intensity variation is only a hint; it can distinguish "dynamic-looking"
        # temporal series from nearly static repeats, but not perfusion vs cine reliably.
        mean = float(np.mean(center_curve)) if len(center_curve) else 0.0
        rel_range = (
            float(np.ptp(center_curve)) / max(mean, 1e-6)
            if len(center_curve)
            else 0.0
        )
        return (
            "TEMPORAL_LIKELY",
            f"median_adj_sim={med:.3f}; p10={p10:.3f}; center_rel_range={rel_range:.3f}",
        )

    # Spatial stack: adjacent slices are related, but anatomy changes progressively.
    if med >= SPATIAL_MEDIAN_SIM and len(cps) <= 1:
        return (
            "SPATIAL_STACK_LIKELY",
            f"median_adj_sim={med:.3f}; p10={p10:.3f}; change_points={len(cps)}",
        )

    return (
        "HETEROGENEOUS_OR_UNKNOWN",
        f"median_adj_sim={med:.3f}; p10={p10:.3f}; change_points={len(cps)}",
    )


@dataclass
class SeriesResult:
    path: str
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

    # Can be either SR_* or Series...
    series_name = series_path.name

    return label, directory, series_name


def analyze_one_series(series_path: Path) -> tuple[SeriesResult, list[float], list[int], list[Path]]:
    paths = list_images(series_path)
    n = len(paths)

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
        raise ValueError(f"No readable images in {series_path}")

    dup_after_first, dup_groups, dup_frac = duplicate_fraction_within_series(paths)

    sims = []
    for i in range(n - 1):
        sims.append(structural_similarity(imgs[i], imgs[i + 1]))
    sims_arr = np.array(sims, dtype=float)

    cps = robust_change_points(sims_arr)
    ccurve = np.array([center_intensity(im) for im in imgs], dtype=float)

    class_label, reason = classify_series(n, sims_arr, cps, dup_frac, ccurve)

    med = float(np.median(sims_arr)) if len(sims_arr) else 0.0
    p10 = float(np.percentile(sims_arr, 10)) if len(sims_arr) else 0.0
    p90 = float(np.percentile(sims_arr, 90)) if len(sims_arr) else 0.0
    mn = float(np.min(sims_arr)) if len(sims_arr) else 0.0
    mx = float(np.max(sims_arr)) if len(sims_arr) else 0.0
    mean_center = float(np.mean(ccurve))
    rel_range = float(np.ptp(ccurve)) / max(mean_center, 1e-6)

    # 1-based display: if cp=19 zero-based, new segment starts at image #20.
    cps_1based = [c + 1 for c in cps]

    result = SeriesResult(
        path=str(series_path),
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
        change_points_1based=";".join(map(str, cps_1based)),
        segment_lengths=";".join(map(str, segment_lengths(n, cps))),
        center_intensity_rel_range=rel_range,
    )
    return result, sims, cps, paths


def make_montage(paths: list[Path], out_path: Path, max_tiles: int = 60, tile_w: int = 160):
    if not paths:
        return

    # Sample evenly if the series is very long.
    if len(paths) > max_tiles:
        idx = np.linspace(0, len(paths) - 1, max_tiles).round().astype(int)
        shown = [paths[i] for i in idx]
    else:
        shown = paths

    tiles = []
    for p in shown:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        scale = tile_w / max(1, w)
        nh = max(1, int(round(h * scale)))
        tile = cv2.resize(img, (tile_w, nh), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        label = p.stem
        cv2.rectangle(tile, (0, 0), (min(tile_w, 90), 18), (0, 0, 0), -1)
        cv2.putText(tile, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    if not tiles:
        return

    max_h = max(t.shape[0] for t in tiles)
    norm = []
    for t in tiles:
        if t.shape[0] < max_h:
            pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
            t = np.vstack([t, pad])
        norm.append(t)

    cols = min(8, max(1, math.ceil(math.sqrt(len(norm) * 1.6))))
    rows = math.ceil(len(norm) / cols)

    blank = np.zeros_like(norm[0])
    while len(norm) < rows * cols:
        norm.append(blank.copy())

    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.hstack(norm[r * cols:(r + 1) * cols]))
    montage = np.vstack(row_imgs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_global_duplicates(root: Path, series_dirs: list[Path], out_txt: Path):
    groups = defaultdict(list)
    total = 0

    for sr in series_dirs:
        for p in list_images(sr):
            total += 1
            try:
                groups[sha256_file(p)].append(p)
            except OSError as e:
                print(f"[WARN] hash failed: {p}: {e}")

    dup_groups = [g for g in groups.values() if len(g) > 1]
    dup_groups.sort(key=lambda g: (-len(g), str(g[0]).lower()))

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"Total images scanned: {total}\n")
        f.write(f"Exact duplicate groups: {len(dup_groups)}\n")
        f.write(f"Images involved in duplicate groups: {sum(len(g) for g in dup_groups)}\n\n")

        for i, g in enumerate(dup_groups, 1):
            f.write(f"DUPLICATE GROUP {i}  (n={len(g)})\n")
            for p in g:
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                f.write(f"  {rel}\n")
            f.write("\n")

    return len(dup_groups), sum(len(g) for g in dup_groups)


def main():
    # -------------------------------------------------------------------------
    # No command-line parameters.
    # Everything is configured by the hardcoded variables at the top.
    # -------------------------------------------------------------------------
    root = DATASET_ROOT.expanduser().resolve()

    if not root.exists():
        raise SystemExit(
            f"Dataset root does not exist:\n  {root}\n\n"
            "Edit DATASET_ROOT near the top of the script."
        )

    if OUTPUT_DIR is None:
        out_dir = root / "_series_analysis"
    else:
        out_dir = Path(OUTPUT_DIR).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CONFIGURATION ===")
    print(f"Dataset root : {root}")
    print(f"Output dir   : {out_dir}")
    print(f"Montages     : {CREATE_MONTAGES}")
    print()

    series_dirs = find_series(root)
    print(f"Found SR_* / Series... folders: {len(series_dirs)}")

    if not series_dirs:
        raise SystemExit("No SR_* or Series... folders with images were found.")

    # Global exact duplicates.
    dup_txt = out_dir / "global_exact_duplicate_groups.txt"
    n_dup_groups, n_dup_images = write_global_duplicates(root, series_dirs, dup_txt)
    print(f"Global exact duplicate groups: {n_dup_groups}")
    print(f"Images involved in duplicate groups: {n_dup_images}")

    results: list[SeriesResult] = []
    cp_rows = []

    for idx, sr in enumerate(series_dirs, 1):
        try:
            result, sims, cps, paths = analyze_one_series(sr)
            results.append(result)

            label, directory, sr_name = parse_group_info(sr, root)

            for cp in cps:
                # cp is zero-based index of NEW segment.
                cp_rows.append({
                    "series_path": str(sr),
                    "label": label,
                    "directory": directory,
                    "series": sr_name,
                    "break_between_image_numbers": f"{cp}-{cp+1}",
                    "new_segment_starts_at_image_number": cp + 1,
                    "similarity_across_break": sims[cp - 1] if cp - 1 < len(sims) else "",
                })

            if CREATE_MONTAGES:
                safe = "__".join(sr.relative_to(root).parts)
                make_montage(paths, out_dir / "montages" / f"{safe}.jpg")

            print(
                f"[{idx:4d}/{len(series_dirs)}] "
                f"{result.class_label:30s} "
                f"n={result.n_images:4d} "
                f"med={result.median_adj_similarity:.3f} "
                f"cp={result.change_points_1based or '-':10s} "
                f"{sr.relative_to(root)}"
            )

        except Exception as e:
            print(f"[ERROR] {sr}: {e}")

    # Main CSV.
    analysis_csv = out_dir / "series_analysis.csv"
    fields = [
        "path",
        "class_label",
        "reason",
        "n_images",
        "exact_dup_images_after_first",
        "exact_dup_groups",
        "exact_dup_fraction",
        "median_adj_similarity",
        "p10_adj_similarity",
        "p90_adj_similarity",
        "min_adj_similarity",
        "max_adj_similarity",
        "change_points_1based",
        "segment_lengths",
        "center_intensity_rel_range",
    ]

    with analysis_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)

    cp_csv = out_dir / "series_change_points.csv"
    cp_fields = [
        "series_path",
        "label",
        "directory",
        "series",
        "break_between_image_numbers",
        "new_segment_starts_at_image_number",
        "similarity_across_break",
    ]

    with cp_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cp_fields)
        w.writeheader()
        w.writerows(cp_rows)

    # Class summary.
    counts = defaultdict(int)
    for r in results:
        counts[r.class_label] += 1

    print("\n=== STRUCTURE SUMMARY ===")
    for k in sorted(counts):
        print(f"{k:30s}: {counts[k]}")

    print("\nOutputs:")
    print(f"  {analysis_csv}")
    print(f"  {cp_csv}")
    print(f"  {dup_txt}")

    if CREATE_MONTAGES:
        print(f"  {out_dir / 'montages'}")

    print(
        "\nNOTE: These are heuristic structural labels. "
        "Calibrate thresholds against a manually reviewed subset before using them as ground truth."
    )


if __name__ == "__main__":
    main()
