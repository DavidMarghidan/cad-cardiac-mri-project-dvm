"""One-step, in-memory cardiac MRI CAD research pipeline.

Only manual PNG masks and explicit labels are written by this pipeline. All quality
measurements, best model states, OOF masks, matching records, features, and results
stay in RAM. A kernel restart discards them; each full run recomputes them.
The HTML editor is displayed live and remains a separate, optional user action.

PIPELINE
1. DataStage       -> dataset, preprocessing, quality, manual targets
2. AttentionStage  -> five patient-level cross-fit Attention U-Nets + OOF masks
3. ReviewStage     -> optional HTML review of unresolved masks
4. MatchingStage   -> balanced Sick/Normal acquisition matching
5. FeatureStage    -> frozen EfficientNet features for all 11 experiments
6. EvaluationStage -> nested patient-level classification + paired AUC tests
7. Pipeline        -> the two commands a user normally calls

Scientific safeguards remain unchanged: patient-level folds, out-of-fold masks,
same-slice controls, matched cohorts, and patient-level evaluation.
"""

import base64
import contextlib
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import time
import uuid
from collections import OrderedDict, defaultdict, namedtuple
from pathlib import Path
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

# The workspace describes manual data only; intermediate stages use RAM.
Workspace = namedtuple("Workspace", "root manual_masks manual_annotations")

# Manual target labels are semantic values, not experiment settings.
HEART_PRESENT = "HEART_PRESENT"
NO_HEART_VISIBLE = "NO_HEART_VISIBLE"
UNUSABLE = "UNUSABLE"
VALID_TARGET_TYPES = {HEART_PRESENT, NO_HEART_VISIBLE, UNUSABLE}
ANNOTATION_FIELDS = (
    "timestamp_utc", "image_token", "patient_id", "series_id",
    "target_type", "source", "sample_weight", "note",
)

# =============================================================================
# STAGE 1 — DATASET, PREPROCESSING, QUALITY, AND MANUAL TARGETS
# =============================================================================
class DataStage:
    """Everything needed before neural training.

    No audit/history file is reused. The dataset and quality measurements are rebuilt
    every run. Only manual PNG masks and manual labels are persistent.
    """

    image_cache = OrderedDict()  # in-memory only; discarded when the run ends

    # Convert CSV/string values safely before numeric comparisons.
    @staticmethod
    def _as_float(value, default=0.0):
        try:
            result=float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if np.isfinite(result) else float(default)
    # Convert CSV/string values safely to an integer.
    @staticmethod
    def _as_int(value, default=0):
        return int(round(DataStage._as_float(value, float(default))))
    # Find the CAD dataset in Kaggle, an environment override, or the local fallback.
    @staticmethod
    def _default_dataset_path():
        env_value=os.environ.get("CAD_DATASET_PATH", "").strip()
        candidates=[]
        if env_value:
            candidates.append(Path(env_value))
        candidates.extend(
            [
                Path("/kaggle/input/cad-cardiac-mri-dataset"),
                Path("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"),
                Path(r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset"),
            ]
        )

        def is_dataset_root(path):
            return path.is_dir() and (path / "Normal").is_dir() and (path / "Sick").is_dir()

        for candidate in candidates:
            if is_dataset_root(candidate):
                return candidate

        kaggle_input=Path("/kaggle/input")
        if kaggle_input.is_dir():
            for current_root, directory_names, _ in os.walk(kaggle_input):
                current=Path(current_root)
                try:
                    depth=len(current.relative_to(kaggle_input).parts)
                except ValueError:
                    depth=99
                if {"Normal", "Sick"}.issubset(set(directory_names)):
                    return current
                if depth >= 3:
                    directory_names[:]=[]

        return candidates[0] if candidates else Path.cwd() / "CAD Cardiac MRI Dataset"
    # Locate the existing manual-mask workspace without changing its folder layout.
    @staticmethod
    def _default_workspace_path():
        if Path("/kaggle/working").exists():
            return Path("/kaggle/working/cad_attention_unet_workspace")
        return Path.cwd() / "cad_attention_unet_workspace"
    # Seed Python, NumPy and PyTorch so patient folds and training are reproducible.
    @staticmethod
    def seed_everything(seed=None, include_cuda=False):
        seed=int(42 if seed is None else seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        if include_cuda:
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms were disabled in the researched runs; cuDNN
            # may therefore choose its fastest kernel for the current GPU shapes.
            torch.backends.cudnn.deterministic=False
            torch.backends.cudnn.benchmark=True
    # Initialize only the CPU/GPU resources needed by the current research stage.
    @staticmethod
    def start_device_stage(requested, stage_name):
        # A stage chooses CPU or CUDA only when it starts. Importing this file never
        # allocates GPU memory, which is important when moving between Kaggle sessions.
        requested=(requested or "cpu").strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Enable a Kaggle GPU.")
        if requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")
        if requested == "auto":
            requested="cuda" if torch.cuda.is_available() else "cpu"
        device=torch.device(requested)
        DataStage.seed_everything(include_cuda=device.type == "cuda")
        if device.type == "cuda":
            DataStage.release_device(device)
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        print(f"[DEVICE] {stage_name}: {device.type}")
        return device
    # Report peak GPU memory when relevant, then release stage resources.
    @staticmethod
    def finish_device_stage(device, stage_name):
        # Print peak GPU memory when CUDA supplied this stage.
        if device.type == "cuda":
            try:
                peak_gb=torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[DEVICE] {stage_name}: peak GPU={peak_gb:.2f} GB")
            except Exception:
                pass
        # Release Python and CUDA memory regardless of the stage result.
        DataStage.release_device(device)
        print(f"[DEVICE] {stage_name}: resources released")
    # Move a neural model to the selected device and efficient CUDA memory layout.
    @staticmethod
    def prepare_model(model, device):
        model=model.to(device)
        if device.type == "cuda":
            model=model.to(memory_format=torch.channels_last)
        return model
    # Move a tensor to the active stage device without changing its values.
    @staticmethod
    def move_tensor(tensor, device):
        tensor=tensor.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda" and tensor.ndim == 4:
            tensor=tensor.contiguous(memory_format=torch.channels_last)
        return tensor
    # Use mixed precision only on CUDA; CPU execution remains ordinary float32.
    @staticmethod
    def autocast(device):
        enabled=device.type == "cuda"
        if not enabled:
            return contextlib.nullcontext()
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=True)
    # Create the AMP gradient scaler used by GPU Attention U-Net training.
    @staticmethod
    def grad_scaler(device):
        enabled=device.type == "cuda"
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)
    # Free Python/CUDA caches between folds and between Kaggle stages.
    @staticmethod
    def release_device(device):
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()
    # Format stage timing for concise console output.
    @staticmethod
    def format_seconds(seconds):
        seconds=max(0, int(round(seconds)))
        hours, seconds=divmod(seconds, 3600)
        minutes, seconds=divmod(seconds, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"
    # Keep the existing manual-mask folder and label filename; create no run/output tree.
    @staticmethod
    def create_workspace(manual_root=None):
        root = Path(manual_root or DataStage._default_workspace_path())
        manual_masks = root / "manual_masks"
        manual_masks.mkdir(parents=True, exist_ok=True)
        return Workspace(root, manual_masks, root / "manual_annotation_labels.csv")
    # Save manual PNG targets only; automatic masks never use this function.
    @staticmethod
    def write_png(path, image):
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), np.asarray(image), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise RuntimeError(f"OpenCV could not write PNG: {path}")
    # Create a stable image identifier from its patient-relative path.
    @staticmethod
    def image_token(image_path, patient_id, series_id):
        # Hash the path starting at Directory_* so tokens remain stable if the dataset
        # is mounted under a different Kaggle/local root.
        parts=image_path.parts
        patient_index=next(
            (i for i, part in enumerate(parts) if str(part).startswith("Directory_")),
            None,
        )
        if patient_index is None:
            raise ValueError(f"Path does not contain Directory_*: {image_path}")
        relative="/".join(map(str, parts[patient_index:]))
        digest=hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        safe_series=str(series_id).replace("/", "__").replace("\\", "__").replace(" ", "_")
        return f"{patient_id}__{safe_series}__{digest}"
    # Fold assignment is derived only from patient identity and class label.
    # No image-level random split is permitted because thousands of correlated
    # slices from one patient would otherwise leak into both train and validation.
    @staticmethod
    def patient_folds(
        patients,
    ):
        if isinstance(patients, dict):
            groups=defaultdict(list)
            for patient_id, label in patients.items():
                groups[int(label)].append(str(patient_id))
        else:
            groups={0: list(map(str, patients))}

        result={}
        for label, patient_ids in sorted(groups.items()):
            ordered=sorted(
                set(patient_ids),
                key=lambda patient_id: hashlib.sha256(
                    f"{42}|segmentation-fold|{label}|{patient_id}".encode("utf-8")
                ).hexdigest(),
            )
            for index, patient_id in enumerate(ordered):
                result[patient_id]=index % int(5)
        return result
    # Scan the dataset and add patient folds plus previous/current/next image context.
    @staticmethod
    def discover_dataset(dataset_path, workspace):
        dataset_path=Path(dataset_path)
        started=time.perf_counter()
        raw_rows=[]
        patient_to_label={}

        for class_name, label in (("Normal", 0), ("Sick", 1)):
            class_path=dataset_path / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(f"Missing directory: {class_path}")

            for patient_path in sorted(class_path.iterdir()):
                if not patient_path.is_dir() or not patient_path.name.lower().startswith("directory_"):
                    continue
                patient_id=patient_path.name
                old_label=patient_to_label.get(patient_id)
                if old_label is not None and old_label != label:
                    raise RuntimeError(f"{patient_id} appears in both classes.")
                patient_to_label[patient_id]=label

                for image_path in sorted(patient_path.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        raw_rows.append((image_path, label, patient_id, f"{patient_id}/__ROOT__"))

                for series_path in sorted(path for path in patient_path.iterdir() if path.is_dir()):
                    series_id=f"{patient_id}/{series_path.name}"
                    for image_path in sorted(series_path.rglob("*")):
                        if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                            raw_rows.append((image_path, label, patient_id, series_id))

        if not raw_rows:
            raise RuntimeError(f"No images were found in {dataset_path}")

        folds = DataStage.patient_folds(patient_to_label)
        rows = [
            {
                "manifest_index": index,
                "image_path": str(path),
                "label": int(label),
                "patient_id": patient_id,
                "series_id": series_id,
                "image_token": DataStage.image_token(path, patient_id, series_id),
                "segmentation_fold": int(folds[patient_id]),
            }
            for index, (path, label, patient_id, series_id) in enumerate(raw_rows)
        ]
        if len({row["image_token"] for row in rows}) != len(rows):
            raise RuntimeError("Duplicate image tokens were generated.")
        for row in rows:
            row["manual_mask_path"] = str(workspace.manual_masks / f"{row['image_token']}.png")

        # Neighbours stay inside the same sequence directory and patient.
        by_sequence_group=defaultdict(list)
        for index, row in enumerate(rows):

            parent=str(Path(row["image_path"]).parent)
            sequence_group=f"{row['series_id']}::{parent}"
            row["sequence_group_id"]=sequence_group
            by_sequence_group[sequence_group].append(index)
        for indices in by_sequence_group.values():
            ordered=sorted(
                indices,
                key=lambda index: DataStage.natural_path_key(rows[index]["image_path"]),
            )
            for position, row_index in enumerate(ordered):
                previous_index=ordered[max(0, position - 1)]
                next_index=ordered[min(len(ordered) - 1, position + 1)]
                rows[row_index]["sequence_index"]=int(position)
                rows[row_index]["sequence_length"]=int(len(ordered))
                rows[row_index]["previous_image_path"]=rows[previous_index]["image_path"]
                rows[row_index]["next_image_path"]=rows[next_index]["image_path"]

        print("[DATA] Patients:", len(patient_to_label))
        print("[DATA] Normal:", sum(label == 0 for label in patient_to_label.values()))
        print("[DATA] Sick:", sum(label == 1 for label in patient_to_label.values()))
        print("[DATA] Images:", len(rows))
        print("[DATA] Series proxies:", len({row["series_id"] for row in rows}))
        print("[DATA] Scan time:", DataStage.format_seconds(time.perf_counter() - started))
        return rows
    # Sort slice filenames numerically (img2 before img10) when building sequence context.
    @staticmethod
    def natural_path_key(path):
        return tuple(
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))
        )
    # Read gray.
    @staticmethod
    def read_gray(path):
        image=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"OpenCV cannot read image: {path}")
        return image
    # Detect near-black, low-variance border lines that are safe to treat as scanner padding.
    @staticmethod
    def is_dark_uniform_line(line):
        values=np.asarray(line, dtype=np.float32).reshape(-1)
        return bool(
            values.size
            and float(values.mean()) <= 12.0
            and float(values.std()) <= 4.0
            and float(np.mean(values <= 20))
            >= 0.98
        )
    # Find conservative crop bounds without using CAD labels or model outputs.
    @staticmethod
    def detect_padding_bounds(image):
        if image.ndim != 2:
            raise ValueError("The image must be a 2D grayscale array.")
        height, width=image.shape
        # Padding removal is deliberately conservative: at least 60% of each
        # original dimension must remain, and at most 20% is removed from one side.
        min_height=max(8, int(np.ceil(height * 0.60)))
        min_width=max(8, int(np.ceil(width * 0.60)))
        max_vertical=int(np.floor(height * 0.20))
        max_horizontal=int(np.floor(width * 0.20))

        top=0
        while top < max_vertical and height - top - 1 >= min_height and DataStage.is_dark_uniform_line(image[top, :]):
            top +=1
        bottom_crop=0
        while bottom_crop < max_vertical and height - top - bottom_crop - 1 >= min_height and DataStage.is_dark_uniform_line(image[height - 1 - bottom_crop, :]):
            bottom_crop +=1
        left=0
        while left < max_horizontal and width - left - 1 >= min_width and DataStage.is_dark_uniform_line(image[:, left]):
            left +=1
        right_crop=0
        while right_crop < max_horizontal and width - left - right_crop - 1 >= min_width and DataStage.is_dark_uniform_line(image[:, width - 1 - right_crop]):
            right_crop +=1

        if top < 2:
            top=0
        if bottom_crop < 2:
            bottom_crop=0
        if left < 2:
            left=0
        if right_crop < 2:
            right_crop=0

        bottom=height - bottom_crop
        right=width - right_crop
        if bottom - top < min_height or right - left < min_width:
            return 0, height, 0, width
        return int(top), int(bottom), int(left), int(right)
    # Scale intensities with 1st/99th percentiles so isolated extremes do not dominate.
    @staticmethod
    def robust_scale(image):
        image=np.asarray(image, dtype=np.float32)
        lower, upper=np.percentile(
            image,
            [1.0, 99.0],
        )
        lower, upper=float(lower), float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower, upper=float(image.min()), float(image.max())
        if upper <= lower:
            return np.zeros_like(image, dtype=np.float32), lower, upper
        scaled=np.clip(image, lower, upper)
        return ((scaled - lower) / (upper - lower)).astype(np.float32), lower, upper
    # Compute the resize and centering geometry used by both images and masks.
    @staticmethod
    def canvas_geometry(height, width):
        size=256
        long_side=240
        scale=float(long_side / max(height, width))
        resized_height=max(1, int(round(height * scale)))
        resized_width=max(1, int(round(width * scale)))
        if height >= width:
            resized_height=long_side
        else:
            resized_width=long_side
        resized_height=min(resized_height, size)
        resized_width=min(resized_width, size)
        top=(size - resized_height) // 2
        left=(size - resized_width) // 2
        return resized_height, resized_width, top, left, scale
    # Resize one image while preserving aspect ratio and center it on the fixed canvas.
    @staticmethod
    def to_canvas(image):
        image=np.asarray(image, dtype=np.float32)
        height, width=image.shape
        resized_height, resized_width, top, left, scale=DataStage.canvas_geometry(height, width)
        if (resized_height, resized_width) != (height, width):
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized=cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
        else:
            resized=image
        canvas=np.zeros(
            (256, 256),
            dtype=np.float32,
        )
        canvas[top : top + resized_height, left : left + resized_width]=np.clip(resized, 0.0, 1.0)
        return canvas
    # Return only the robust/raw/content views actually consumed by the pipeline.
    @staticmethod
    def standardize_image(image):
        top, bottom, left, right=DataStage.detect_padding_bounds(image)
        cropped=image[top:bottom, left:right]
        if cropped.size == 0:
            raise RuntimeError("Automatic cropping produced an empty image.")
        robust, _, _=DataStage.robust_scale(cropped)
        cropped_float=cropped.astype(np.float32)
        raw=cropped_float / 255.0
        robust_canvas=DataStage.to_canvas(robust)
        raw_canvas=DataStage.to_canvas(raw)
        resized_height, resized_width, top_pad, left_pad, _=DataStage.canvas_geometry(*cropped.shape)
        content=np.zeros((256, 256), dtype=np.float32)
        content[top_pad:top_pad + resized_height, left_pad:left_pad + resized_width]=1.0
        return robust_canvas, raw_canvas, content
    # Return the canonical 256×256 robust-scaled image used by Attention U-Net.
    @staticmethod
    def standardized_uint8(path):
        image=DataStage.read_gray(path)
        robust, _, _=DataStage.standardize_image(image)
        return np.clip(np.round(robust * 255.0), 0, 255).astype(np.uint8)
    # Create the robust/raw/content views later combined with ROI or complement masks.
    @staticmethod
    def classifier_views(path):
        image=DataStage.read_gray(path)
        robust, raw, content=DataStage.standardize_image(image)
        size=224
        robust_224=cv2.resize(robust, (size, size), interpolation=cv2.INTER_AREA)
        raw_224=cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
        content_224=cv2.resize(content, (size, size), interpolation=cv2.INTER_NEAREST)
        return (
            robust_224.astype(np.float32),
            raw_224.astype(np.float32),
            (content_224 > 0.5).astype(np.float32),
        )
    # Build a compact appearance hash used for duplicate-aware review and matching.
    @staticmethod
    def perceptual_hash(image):
        resized=cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        values=cv2.dct(resized.astype(np.float32))[:8, :8].reshape(-1)
        median=float(np.median(values[1:]))
        bits=values > median
        number=0
        for bit in bits:
            number=(number << 1) | int(bool(bit))
        return f"{number:016x}"
    # Reuse recently standardized slices to avoid repeating OpenCV preprocessing.
    @staticmethod
    def get_cached_image(path):
        if path in DataStage.image_cache:
            value=DataStage.image_cache.pop(path)
            DataStage.image_cache[path]=value
            return value.copy()
        value=DataStage.standardized_uint8(path)
        DataStage.image_cache[path]=value
        while len(DataStage.image_cache) > 2048:
            DataStage.image_cache.popitem(last=False)
        return value.copy()
    # Release standardized-image RAM between expensive stages.
    @staticmethod
    def clear_image_cache():
        DataStage.image_cache.clear()
    # Measure sharpness, noise, dynamic range, and perceptual hash for one slice.
    @staticmethod
    def image_quality_metrics(row):
        image=DataStage.read_gray(row["image_path"])
        robust, raw, content=DataStage.standardize_image(image)
        coordinates=np.argwhere(content > 0.5)
        if not len(coordinates):
            return {
                "perceptual_hash": "",
                "sharpness": np.nan,
                "noise_ratio": np.nan,
                "dynamic_range": np.nan,
            }
        top, left=coordinates.min(axis=0)
        bottom, right=coordinates.max(axis=0) + 1
        size=128
        robust_thumb=cv2.resize(
            robust[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA
        )
        raw_thumb=cv2.resize(
            raw[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA
        )
        robust_u8=np.clip(np.round(robust_thumb * 255.0), 0, 255).astype(np.uint8)
        raw_255=np.clip(raw_thumb * 255.0, 0.0, 255.0).astype(np.float32)
        sharpness=float(cv2.Laplacian(robust_u8, cv2.CV_32F).var())
        residual=raw_255 - cv2.GaussianBlur(raw_255, (3, 3), 0)
        residual_median=float(np.median(residual))
        noise_sigma=float(np.median(np.abs(residual - residual_median)) / 0.6744897501960817)
        lower, upper=np.percentile(raw_255, [5.0, 95.0])
        dynamic_range=float(max(upper - lower, 0.0))
        return {
            "perceptual_hash": DataStage.perceptual_hash(robust_u8),
            "sharpness": sharpness,
            "noise_ratio": float(noise_sigma / max(dynamic_range, 1.0)),
            "dynamic_range": dynamic_range,
        }
    # Recompute image quality in RAM; thresholds and measurements are unchanged.
    @staticmethod
    def build_quality_audit(rows):
        records = []
        started = time.perf_counter()
        for index, row in enumerate(rows, start=1):
            records.append({**row, **DataStage.image_quality_metrics(row)})
            if index == 1 or index % 5000 == 0 or index == len(rows):
                print(f"[QUALITY] {index}/{len(rows)} | "
                      f"{DataStage.format_seconds(time.perf_counter() - started)}")

        by_patient=defaultdict(list)
        for record in records:
            by_patient[str(record["patient_id"])].append(record)

        for patient_rows in by_patient.values():
            sharp=np.asarray([DataStage._as_float(row.get("sharpness"), np.nan) for row in patient_rows])
            noise=np.asarray([DataStage._as_float(row.get("noise_ratio"), np.nan) for row in patient_rows])
            finite_sharp=sharp[np.isfinite(sharp)]
            finite_noise=noise[np.isfinite(noise)]
            blur_threshold=float(4.0)
            noise_threshold=float(0.30)
            if len(finite_sharp) >= 5:
                blur_threshold=max(
                    blur_threshold,
                    float(np.quantile(finite_sharp, 0.03)),
                )
            if len(finite_noise) >= 5:
                noise_threshold=min(
                    noise_threshold,
                    float(np.quantile(finite_noise, 0.97)),
                )
            for record in patient_rows:
                sharpness=float(record.get("sharpness", np.nan))
                noise_ratio=float(record.get("noise_ratio", np.nan))
                dynamic_range=float(record.get("dynamic_range", np.nan))
                reasons=[]
                if not np.isfinite(dynamic_range) or dynamic_range < 12.0:
                    reasons.append("low_dynamic_range")
                if not np.isfinite(sharpness) or sharpness < blur_threshold:
                    reasons.append("blurred")
                if not np.isfinite(noise_ratio) or noise_ratio > noise_threshold:
                    reasons.append("noisy")
                record["patient_blur_threshold"]=blur_threshold
                record["patient_noise_threshold"]=noise_threshold
                record["quality_valid"]=int(not reasons)
                record["quality_reason"]=";".join(reasons)

        valid = sum(int(row["quality_valid"]) for row in records)
        print(f"[QUALITY] valid={valid}/{len(records)}; measurements retained in RAM")
        return records
    # Read binary mask.
    @staticmethod
    def read_binary_mask(path, size=None):
        mask=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask cannot be read: {path}")
        size=int(size or 256)
        if mask.shape != (size, size):
            mask=cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8)
    # Check whether a saved manual mask is readable and has a plausible non-trivial area.
    @staticmethod
    def manual_mask_qc(path):
        path=Path(path)
        if not path.is_file():
            return {"exists": 0, "usable": 0, "area_ratio": np.nan, "reason": "missing"}
        try:
            mask=DataStage.read_binary_mask(path)
        except Exception:
            return {"exists": 1, "usable": 0, "area_ratio": np.nan, "reason": "unreadable"}
        area=float(mask.mean())
        if area < 0.0005:
            reason="empty_or_nearly_empty"
        elif area > 0.98:
            reason="nearly_full"
        else:
            reason=""
        return {"exists": 1, "usable": int(not reason), "area_ratio": area, "reason": reason}
    # Read only the persistent human decisions. Empty/unavailable labels mean no decisions.
    @staticmethod
    def load_annotations(workspace):
        if not workspace.manual_annotations.is_file():
            return {}
        with workspace.manual_annotations.open(newline="", encoding="utf-8") as handle:
            return {
                str(row["image_token"]).strip(): dict(row)
                for row in csv.DictReader(handle)
                if str(row.get("image_token", "")).strip()
            }

    # Replace the label CSV only after the complete new table has been written.
    @staticmethod
    def save_annotations(workspace, annotations):
        path = workspace.manual_annotations
        pending = path.with_suffix(".csv.tmp")
        try:
            with pending.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(annotations[token] for token in sorted(annotations))
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    # Use lower weight for confirmed-auto masks while preserving explicit negatives.
    @staticmethod
    def default_target_weight(target_type, source):
        target_type=str(target_type).upper()
        source=str(source).lower()
        if target_type == NO_HEART_VISIBLE:
            return float(0.90)
        if "auto" in source:
            return float(0.72)
        return float(1.0)
    # Persist manual decisions only; no audit/history output file is needed.
    @staticmethod
    def set_annotation(
        workspace,
        row,
        target_type,
        source,
        sample_weight=None,
        note="",
    ):
        target_type=str(target_type).strip().upper()
        if target_type not in VALID_TARGET_TYPES:
            raise ValueError(f"Unknown annotation type: {target_type}")
        records=DataStage.load_annotations(workspace)
        weight=(
            DataStage.default_target_weight(target_type, source)
            if sample_weight is None
            else float(sample_weight)
        )
        records[str(row["image_token"])]={
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "image_token": str(row["image_token"]),
            "patient_id": str(row.get("patient_id", "")),
            "series_id": str(row.get("series_id", "")),
            "target_type": target_type,
            "source": str(source),
            "sample_weight": float(weight),
            "note": str(note),
        }
        DataStage.save_annotations(workspace, records)
    # Persist manual decisions only; no audit/history output file is needed.
    @staticmethod
    def remove_annotation(workspace, image_token):
        records=DataStage.load_annotations(workspace)
        records.pop(str(image_token), None)
        DataStage.save_annotations(workspace, records)
    # Persist manual decisions only; no audit/history output file is needed.
    @staticmethod
    def register_existing_manual_masks(
        rows,
        workspace,
    ):
        by_token={
            str(row.get("image_token", "")).strip(): dict(row)
            for row in rows
            if str(row.get("image_token", "")).strip()
        }
        annotations=DataStage.load_annotations(workspace)
        manual_files=sorted(workspace.manual_masks.glob("*.png"))
        timestamp=pd.Timestamp.utcnow().isoformat()

        summary={
            "timestamp_utc": timestamp,
            "manual_png_files": len(manual_files),
            "registered_heart_present": 0,
            "already_explicitly_labeled": 0,
            "skipped_empty_or_invalid": 0,
            "skipped_token_absent_from_dataset": 0,
            "reason_counts": {},
            "annotations_path": str(workspace.manual_annotations),
        }
        reason_counts=defaultdict(int)

        for mask_path in manual_files:
            token=str(mask_path.stem)
            row=by_token.get(token)
            if row is None:
                summary["skipped_token_absent_from_dataset"] +=1
                reason_counts["token_absent_from_dataset"] +=1
                continue

            existing=annotations.get(token, {})
            existing_target=str(existing.get("target_type", "")).strip().upper()
            if existing_target in VALID_TARGET_TYPES:
                summary["already_explicitly_labeled"] +=1
                reason_counts[f"already_{existing_target.lower()}"] +=1
                continue

            mask_qc=DataStage.manual_mask_qc(mask_path)
            if not bool(mask_qc.get("usable", 0)):
                summary["skipped_empty_or_invalid"] +=1
                reason=str(mask_qc.get("reason", "") or "invalid_manual_mask")
                reason_counts[reason] +=1
                continue

            annotations[token]={
                "timestamp_utc": timestamp,
                "image_token": token,
                "patient_id": str(row.get("patient_id", "")),
                "series_id": str(row.get("series_id", "")),
                "target_type": HEART_PRESENT,
                "source": "existing_manual_on_review",
                "sample_weight": float(1.0),
                "note": (
                    "Existing non-empty manual PNG registered automatically "
                    "when review was opened; the mask file itself was not rewritten."
                ),
            }
            summary["registered_heart_present"] +=1
            reason_counts["registered_heart_present"] +=1

        if summary["registered_heart_present"] > 0 or not workspace.manual_annotations.is_file():
            DataStage.save_annotations(workspace, annotations)

        summary["reason_counts"]=dict(sorted(reason_counts.items()))
        print(
            "[REVIEW][EXISTING MASKS] "
            f"registered HEART_PRESENT={summary['registered_heart_present']}, "
            f"already labeled={summary['already_explicitly_labeled']}, "
            f"empty/invalid={summary['skipped_empty_or_invalid']}, "
            f"token missing={summary['skipped_token_absent_from_dataset']}"
        )
        return summary
    # Validate current manual targets and return the audit in memory, never as a CSV.
    @staticmethod
    def audit_manual_masks(
        rows,
        workspace,
        minimum_masks=None,
    ):
        # The research pipeline was trained only after at least 800 positive manual masks
        # survived quality/mask QC. The caller may override this only for diagnostics.
        minimum_masks=800 if minimum_masks is None else int(minimum_masks)
        by_token={str(row["image_token"]): dict(row) for row in rows}
        manual_files={path.stem: path for path in workspace.manual_masks.glob("*.png")}
        annotations=DataStage.load_annotations(workspace)
        candidate_tokens=sorted(set(manual_files) | set(annotations))

        accepted=[]
        audit=[]
        patient_counts=defaultdict(int)
        positive_patient_counts=defaultdict(int)
        fold_positive_counts=defaultdict(int)
        fold_negative_counts=defaultdict(int)
        positive_count=0
        negative_count=0
        unusable_count=0

        for token in candidate_tokens:
            row=by_token.get(token)
            path=manual_files.get(token, workspace.manual_masks / f"{token}.png")
            annotation=annotations.get(token, {})
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

            mask_qc=DataStage.manual_mask_qc(path)
            target_type=str(annotation.get("target_type", "")).strip().upper()
            source=str(annotation.get("source", "")).strip()
            if not target_type:
                if mask_qc["usable"]:
                    target_type=HEART_PRESENT
                    source="legacy_nonempty_manual"
                else:
                    target_type="UNLABELED_EMPTY"
                    source="legacy_unlabeled"

            quality_row=row
            quality_valid=DataStage._as_int(quality_row.get("quality_valid"), 0) == 1
            reasons=[]
            heart_present=""

            if target_type == HEART_PRESENT:
                heart_present=1
                if not mask_qc["usable"]:
                    reasons.append(mask_qc["reason"] or "missing_positive_mask")
            elif target_type == NO_HEART_VISIBLE:
                heart_present=0
                if not path.is_file():
                    DataStage.write_png(
                        path,
                        np.zeros(
                            (256, 256),
                            dtype=np.uint8,
                        ),
                    )
                    mask_qc=DataStage.manual_mask_qc(path)
                if not np.isfinite(DataStage._as_float(mask_qc.get("area_ratio"), np.nan)):
                    reasons.append("unreadable_negative_mask")
                elif DataStage._as_float(mask_qc.get("area_ratio"), 1.0) >= 0.0005:
                    reasons.append("no_heart_annotation_has_nonempty_mask")
            elif target_type == UNUSABLE:
                unusable_count +=1
                reasons.append("explicitly_unusable")
            else:
                reasons.append("empty_mask_without_explicit_no_heart_label")

            if target_type != UNUSABLE and not quality_valid:
                reasons.append(str(quality_row.get("quality_reason", "quality_invalid")))

            if target_type == UNUSABLE:
                status="EXCLUDED"
            else:
                status="ACCEPTED" if not reasons else "REJECTED"

            sample_weight=DataStage._as_float(
                annotation.get("sample_weight"),
                DataStage.default_target_weight(target_type, source),
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
                    "manual_mask_area_ratio": float(DataStage._as_float(mask_qc.get("area_ratio"), 0.0)),
                    "quality_valid": 1,
                    "perceptual_hash": quality_row.get("perceptual_hash", ""),
                    "segmentation_target_type": target_type,
                    "heart_present": int(heart_present),
                    "sample_weight": float(sample_weight),
                    "annotation_source": source,
                }
            )
            accepted.append(row)
            patient_counts[str(row["patient_id"])] +=1
            fold=int(row["segmentation_fold"])
            if int(heart_present) == 1:
                positive_count +=1
                positive_patient_counts[str(row["patient_id"])] +=1
                fold_positive_counts[fold] +=1
            else:
                negative_count +=1
                fold_negative_counts[fold] +=1

        rejected=sum(row.get("status") == "REJECTED" for row in audit)
        excluded=sum(row.get("status") == "EXCLUDED" for row in audit)
        unlabeled_empty=sum(
            "without_explicit_no_heart" in str(row.get("reason", "")) for row in audit
        )
        blur_or_noise=sum(
            row.get("status") == "REJECTED" and bool(row.get("quality_reason"))
            for row in audit
        )
        summary={
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
                for fold in range(5)
            },
            "negative_masks_per_fold": {
                str(fold): int(fold_negative_counts.get(fold, 0))
                for fold in range(5)
            },
            "annotations": str(workspace.manual_annotations),
        }
        print(
            "[MANUAL MASKS] "
            f"heart_present={positive_count}, no_heart={negative_count}, "
            f"rejected={rejected}, unusable={unusable_count}, "
            f"patients={len(patient_counts)}"
        )

        reason_counts = defaultdict(int)
        for record in audit:
            if record.get("status") != "ACCEPTED":
                reason_counts[record.get("reason", "unknown")] += 1
        summary["rejection_reasons"] = dict(sorted(reason_counts.items()))
        if reason_counts:
            print("[MANUAL MASKS] rejected/excluded reasons:", summary["rejection_reasons"])

        if positive_count < minimum_masks:
            raise RuntimeError(
                f"Only {positive_count} valid HEART_PRESENT masks remain, below the minimum of "
                f"{minimum_masks}. Inspect the manual-target summary above and the HTML editor."
            )
        missing_folds=[
            fold
            for fold in range(5)
            if fold_positive_counts.get(fold, 0) == 0
        ]
        if missing_folds:
            raise RuntimeError(
                f"No valid HEART_PRESENT masks exist in folds {missing_folds}."
            )
        if len(positive_patient_counts) < 5 + 1:
            raise RuntimeError("Too few patients have positive masks for cross-fitting.")
        return accepted, audit, summary

# =============================================================================
# STAGE 2 — ATTENTION U-NET: MODEL + OOF SEGMENTATION
# =============================================================================
# These small classes are required by PyTorch. They are implementation details of
# AttentionStage, not separate pipeline-management layers.
class AttentionConvBlock(nn.Module):
    """Apply two normalized convolution layers inside the Attention U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups=next(group for group in (8, 4, 2, 1) if out_channels % group == 0)
        self.block=nn.Sequential(
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
    """Weight an encoder skip connection using decoder context."""

    def __init__(self, gating_channels, skip_channels, inter_channels):
        super().__init__()
        groups=next(group for group in (8, 4, 2, 1) if inter_channels % group == 0)
        self.gating=nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(groups, inter_channels),
        )
        self.skip=nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(groups, inter_channels),
        )
        self.weight=nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, gating, skip):
        gating=F.interpolate(gating, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.weight(self.gating(gating) + self.skip(skip))
class AttentionUpBlock(nn.Module):
    """Upsample decoder features, gate the skip connection, and refine the result."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up=nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gate=AttentionGate(out_channels, skip_channels, max(1, out_channels // 2))
        self.refine=AttentionConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x=F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x=self.up(x)
        return self.refine(torch.cat([x, self.gate(x, skip)], dim=1))
class AttentionUNet(nn.Module):
    """2.5D Attention U-Net with segmentation and heart-presence heads.

    Neighboring frames provide local sequence context, while the auxiliary head
    suppresses false positive masks on localizers or frames without visible heart.
    """

    def __init__(
        self,
        base_channels=None,
        input_channels=None,):
        super().__init__()
        base=int(base_channels or 24)  # 24 channels in the first U-Net block
        input_channels=int(input_channels or 3)  # previous/current/next = 2.5D input
        self.input_channels=input_channels
        self.encoder1=AttentionConvBlock(input_channels, base)
        self.encoder2=AttentionConvBlock(base, base * 2)
        self.encoder3=AttentionConvBlock(base * 2, base * 4)
        self.encoder4=AttentionConvBlock(base * 4, base * 8)
        self.bottleneck=AttentionConvBlock(base * 8, base * 16)
        self.pool=nn.MaxPool2d(2)
        self.decoder4=AttentionUpBlock(base * 16, base * 8, base * 8)
        self.decoder3=AttentionUpBlock(base * 8, base * 4, base * 4)
        self.decoder2=AttentionUpBlock(base * 4, base * 2, base * 2)
        self.decoder1=AttentionUpBlock(base * 2, base, base)
        self.output=nn.Conv2d(base, 1, 1)
        self.presence_head=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, base * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(base * 4, 1),
        )

    def forward(self, x):
        e1=self.encoder1(x)
        e2=self.encoder2(self.pool(e1))
        e3=self.encoder3(self.pool(e2))
        e4=self.encoder4(self.pool(e3))
        bottleneck=self.bottleneck(self.pool(e4))
        presence_logits=self.presence_head(bottleneck).squeeze(1)
        d4=self.decoder4(bottleneck, e4)
        d3=self.decoder3(d4, e3)
        d2=self.decoder2(d3, e2)
        d1=self.decoder1(d2, e1)
        return self.output(d1), presence_logits
class SegmentationDataset(Dataset):
    """Load manual segmentation targets and optional training augmentation."""

    def __init__(self, rows, augment, seed):
        self.rows=list(rows)
        self.augment=bool(augment)
        self.seed=int(seed)
        self.epoch=0

    def set_epoch(self, epoch):
        self.epoch=int(epoch)

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _image_stack(row):
        current=str(row["image_path"])
        # 2.5D input: previous, current, and next slices are always used.
        paths=(
            str(row.get("previous_image_path") or current),
            current,
            str(row.get("next_image_path") or current),
        )
        return np.stack(
            [DataStage.get_cached_image(path).astype(np.float32) / 255.0 for path in paths],
            axis=0,
        )

    def __getitem__(self, index):
        row=self.rows[index]
        image=self._image_stack(row)
        mask=DataStage.read_binary_mask(row["manual_mask_path"]).astype(np.float32)
        heart_present=float(DataStage._as_int(row.get("heart_present"), int(mask.any())))
        sample_weight=float(DataStage._as_float(row.get("sample_weight"), 1.0))

        if self.augment:
            rng=np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            size=256
            angle=float(
                rng.uniform(
                    -7.0,
                    7.0,
                )
            )
            scale=float(
                rng.uniform(
                    0.92,
                    1.08,
                )
            )
            shift=0.06 * size
            tx, ty=map(float, rng.uniform(-shift, shift, size=2))
            matrix=cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
            matrix[:, 2] +=(tx, ty)
            image=np.stack(
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
            mask=cv2.warpAffine(
                mask,
                matrix,
                (size, size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            gamma=float(
                rng.uniform(
                    0.85,
                    1.15,
                )
            )
            contrast=float(rng.uniform(0.90, 1.10))
            brightness=float(rng.uniform(-0.04, 0.04))
            image=np.power(np.clip(image, 0.0, 1.0), gamma)
            image=np.clip(image * contrast + brightness, 0.0, 1.0)
            if rng.random() < 0.15:
                image=np.stack(
                    [cv2.GaussianBlur(channel, (3, 3), 0) for channel in image],
                    axis=0,
                )
            noise_std=float(rng.uniform(0.0, 0.025))
            if noise_std > 0:
                image=np.clip(
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
    """Load image stacks for deterministic segmentation inference."""

    def __init__(self, rows):
        self.rows=list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image=SegmentationDataset._image_stack(self.rows[index])
        return torch.from_numpy(np.ascontiguousarray(image)).float(), int(index)

class AttentionStage:
    """Train five patient-level folds from scratch and predict every image OOF."""

    # Combine focal, Tversky, and heart-presence losses using per-target review weights.
    @staticmethod
    def segmentation_loss(
        logits,
        targets,
        presence_logits,
        presence_targets,
        sample_weights,
    ):
        probability=torch.sigmoid(logits)
        bce=F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt=probability * targets + (1.0 - probability) * (1.0 - targets)
        focal=((1.0 - pt).pow(2.0) * bce).mean(
            dim=(1, 2, 3)
        )

        tp=(probability * targets).sum(dim=(1, 2, 3))
        fp=(probability * (1.0 - targets)).sum(dim=(1, 2, 3))
        fn=((1.0 - probability) * targets).sum(dim=(1, 2, 3))
        tversky=(tp + 1e-6) / (
            tp
            + 0.65 * fp  # penalize false-positive mask area
            + 0.35 * fn  # penalize false-negative mask area
            + 1e-6
        )
        segmentation_per_sample=(
            0.40 * focal  # focal-loss contribution
            + 0.60 * (1.0 - tversky)  # Tversky-loss contribution
        )

        sample_weights=sample_weights.float().clamp_min(1e-3)
        segmentation_loss=(
            segmentation_per_sample * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-6)
        presence_loss=F.binary_cross_entropy_with_logits(
            presence_logits.float(), presence_targets.float(), reduction="none"
        )
        presence_loss=(presence_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        return segmentation_loss + 0.30 * presence_loss  # auxiliary heart-presence loss
    # Select validation patients.
    @staticmethod
    def select_validation_patients(patient_ids, target_fold):
        ordered=sorted(
            set(map(str, patient_ids)),
            key=lambda patient_id: hashlib.sha256(
                f"{42}|validation|{target_fold}|{patient_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        count=max(
            1,
            int(
                round(
                    len(ordered)
                    * 0.20
                )
            ),
        )
        count=min(count, max(1, len(ordered) - 1))
        return set(ordered[:count])
    # Keep positive and negative presence examples represented in fold validation.
    @staticmethod
    def ensure_presence_coverage(
        rows,
        validation_patients,
        target_fold,
    ):
        patients=sorted({str(row["patient_id"]) for row in rows})
        has_label={0: set(), 1: set()}
        for row in rows:
            has_label[DataStage._as_int(row.get("heart_present"), 1)].add(
                str(row["patient_id"])
            )
        validation=set(validation_patients)

        def ordered(values, salt):
            return sorted(
                values,
                key=lambda patient: hashlib.sha256(
                    f"{42}|{target_fold}|{salt}|{patient}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )

        for label in (0, 1):
            labelled=has_label[label]
            if len(labelled) < 2:
                continue
            if not (validation & labelled):
                incoming=ordered(labelled - validation, f"incoming-{label}")[0]
                removable=[
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

                outgoing=ordered(validation & labelled, f"outgoing-{label}")[-1]
                validation.remove(outgoing)
                replacement_candidates=set(patients) - validation - {outgoing}
                if replacement_candidates:
                    validation.add(
                        ordered(replacement_candidates, f"replacement-{label}")[0]
                    )
        if not validation or len(validation) >= len(patients):
            return set(validation_patients)
        return validation
    # Balance patients, series, duplicate clusters, and explicit no-heart targets in training.
    @staticmethod
    def training_sampler(
        rows,
        seed,
    ):
        rows=list(rows)
        series_by_patient=defaultdict(set)
        clusters_by_series=defaultdict(set)
        cluster_counts=defaultdict(int)
        for row in rows:
            patient=str(row["patient_id"])
            series=str(row.get("sequence_group_id") or row["series_id"])
            cluster=str(row.get("perceptual_hash") or row["image_token"])
            series_by_patient[patient].add(series)
            clusters_by_series[(patient, series)].add(cluster)
            cluster_counts[(patient, series, cluster)] +=1

        base_weights=[]
        labels=[]
        for row in rows:
            patient=str(row["patient_id"])
            series=str(row.get("sequence_group_id") or row["series_id"])
            cluster=str(row.get("perceptual_hash") or row["image_token"])
            weight=1.0
            weight /=max(1, len(series_by_patient[patient]))
            weight /=max(1, len(clusters_by_series[(patient, series)]))
            weight /=max(1, cluster_counts[(patient, series, cluster)])
            base_weights.append(weight)
            labels.append(DataStage._as_int(row.get("heart_present"), 1))

        base=np.asarray(base_weights, dtype=np.float64)
        labels_array=np.asarray(labels, dtype=np.int64)
        positive=labels_array == 1
        negative=labels_array == 0
        if positive.any() and negative.any():
            target_negative=float(0.30)
            target_positive=1.0 - target_negative
            base[positive] *=target_positive / max(base[positive].sum(), 1e-12)
            base[negative] *=target_negative / max(base[negative].sum(), 1e-12)
        else:
            base /=max(base.sum(), 1e-12)

        generator=torch.Generator().manual_seed(int(seed))
        return WeightedRandomSampler(
            weights=torch.as_tensor(base, dtype=torch.double),
            num_samples=len(rows),
            replacement=True,
            generator=generator,
        )
    # Summarize mask area, boundary contact, centroid, and aspect ratio for QC.
    @staticmethod
    def mask_features(mask):
        mask=(np.asarray(mask) > 0).astype(np.uint8)
        area, boundary=AttentionStage.mask_geometry(mask)
        if not mask.any():
            return {
                "area": area,
                "boundary": boundary,
                "centroid_x": np.nan,
                "centroid_y": np.nan,
                "aspect_ratio": np.nan,
            }
        ys, xs=np.nonzero(mask)
        width=float(xs.max() - xs.min() + 1)
        height=float(ys.max() - ys.min() + 1)
        return {
            "area": area,
            "boundary": boundary,
            "centroid_x": float(xs.mean() / max(1.0, mask.shape[1] - 1)),
            "centroid_y": float(ys.mean() / max(1.0, mask.shape[0] - 1)),
            "aspect_ratio": float(width / max(height, 1.0)),
        }
    # Build geometry prior.
    @staticmethod
    def build_geometry_prior(rows):
        features=[]
        for row in rows:
            if DataStage._as_int(row.get("heart_present"), 1) != 1:
                continue
            try:
                features.append(
                    AttentionStage.mask_features(
                        DataStage.read_binary_mask(row["manual_mask_path"])
                    )
                )
            except Exception:
                continue
        keys=("area", "centroid_x", "centroid_y", "aspect_ratio")
        if not features:
            return {"count": 0, "median": {}, "scale": {}}
        median={}
        scale={}
        for key in keys:
            values=np.asarray([item[key] for item in features], dtype=np.float64)
            values=values[np.isfinite(values)]
            if not len(values):
                continue
            center=float(np.median(values))
            mad=float(np.median(np.abs(values - center)))
            median[key]=center
            scale[key]=max(1e-3, 1.4826 * mad)
        return {"count": len(features), "median": median, "scale": scale}
    # Calculate prior deviation.
    @staticmethod
    def calculate_prior_deviation(mask, prior):
        if not prior or DataStage._as_int(prior.get("count"), 0) < 5 or not np.asarray(mask).any():
            return 0.0
        features=AttentionStage.mask_features(mask)
        deviations=[]
        for key, center in dict(prior.get("median", {})).items():
            value=features.get(key, np.nan)
            scale=DataStage._as_float(dict(prior.get("scale", {})).get(key), 0.0)
            if np.isfinite(value) and scale > 0:
                deviations.append(abs(float(value) - float(center)) / scale)
        return float(max(deviations)) if deviations else 0.0
    # Average recall across available heart-present / no-heart classes.
    @staticmethod
    def balanced_accuracy(labels, predictions):
        labels=np.asarray(labels, dtype=np.int64)
        predictions=np.asarray(predictions, dtype=np.int64)
        values=[]
        for label in (0, 1):
            selector=labels == label
            if selector.any():
                values.append(float(np.mean(predictions[selector] == label)))
        return float(np.mean(values)) if values else 0.0
    # Choose mask and presence thresholds only on non-test validation patients.
    @staticmethod
    def calibrate_threshold(
        model,
        loader,
        device,
        geometry_prior,
    ):
        probabilities=[]
        targets=[]
        presence_probabilities=[]
        presence_targets=[]
        patients=[]
        model.eval()
        with torch.inference_mode():
            for images, masks, target_presence, _weights, patient_ids in loader:
                images=DataStage.move_tensor(images, device)
                with DataStage.autocast(device):
                    logits, presence_logits=model(images)
                    probability=torch.sigmoid(logits)
                    presence_probability=torch.sigmoid(presence_logits)
                probabilities.extend(probability.float().cpu().numpy()[:, 0])
                targets.extend(masks.numpy()[:, 0])
                presence_probabilities.extend(
                    presence_probability.float().cpu().numpy().tolist()
                )
                presence_targets.extend(target_presence.numpy().astype(int).tolist())
                patients.extend(map(str, patient_ids))

        labels_array=np.asarray(presence_targets, dtype=np.int64)
        presence_array=np.asarray(presence_probabilities, dtype=np.float64)
        best_presence={
            "threshold": 0.50,
            "balanced_accuracy": 0.0,
        }
        if len(np.unique(labels_array)) >= 2:
            for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
                prediction=(presence_array >= threshold).astype(np.int64)
                balanced=AttentionStage.balanced_accuracy(
                    labels_array, prediction
                )
                candidate={
                    "threshold": float(threshold),
                    "balanced_accuracy": balanced,
                }
                if (
                    candidate["balanced_accuracy"],
                    -abs(candidate["threshold"] - 0.5),
                ) > (
                    best_presence["balanced_accuracy"],
                    -abs(best_presence["threshold"] - 0.5),):
                    best_presence=candidate
        elif len(labels_array):
            best_presence["balanced_accuracy"]=1.0

        best=None
        for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            by_patient=defaultdict(list)
            invalid=0
            for probability, target, target_presence, patient_id in zip(
                probabilities, targets, presence_targets, patients):
                prediction=AttentionStage.candidate_mask(
                    probability, threshold
                )
                if int(target_presence) == 1:
                    intersection=float(
                        np.logical_and(prediction, target > 0.5).sum()
                    )
                    denominator=float(
                        prediction.sum() + (target > 0.5).sum()
                    )
                    quality=(2.0 * intersection + 1e-6) / (
                        denominator + 1e-6
                    )
                    valid, _=AttentionStage.validate_mask(
                        prediction, probability, geometry_prior
                    )
                    invalid +=int(not valid)
                else:

                    quality=1.0 - min(1.0, float(prediction.mean()) / 0.10)
                by_patient[patient_id].append(float(quality))
            patient_scores=[
                float(np.mean(values)) for values in by_patient.values()
            ]
            mean_score=float(np.mean(patient_scores)) if patient_scores else 0.0
            invalid_rate=float(invalid / max(1, sum(presence_targets)))
            score=mean_score - 0.20 * invalid_rate
            candidate={
                "threshold": float(threshold),
                "patient_balanced_segmentation_score": mean_score,
                "invalid_rate_positive": invalid_rate,
                "score": score,
            }
            if best is None or (
                candidate["score"], -abs(threshold - 0.5)
            ) > (best["score"], -abs(best["threshold"] - 0.5)):
                best=candidate

        result=best or {
            "threshold": 0.50,
            "patient_balanced_segmentation_score": 0.0,
            "invalid_rate_positive": 1.0,
            "score": -1.0,
        }
        result["presence_threshold"]=float(best_presence["threshold"])
        result["presence_balanced_accuracy"]=float(
            best_presence["balanced_accuracy"]
        )
        return result
    # Exclude each test fold from fitting and calibration; retain best CPU weights in RAM.
    @staticmethod
    def train_attention_crossfit(accepted_rows, device):
        DataStage.seed_everything(include_cuda=device.type == "cuda")
        fold_states={}
        all_patients=sorted({str(row["patient_id"]) for row in accepted_rows})

        for fold in range(5):  # five patient-level folds
            fold_started=time.perf_counter()
            non_test_rows=[
                row
                for row in accepted_rows
                if int(row["segmentation_fold"]) != fold
            ]
            validation_patients=AttentionStage.select_validation_patients(
                (row["patient_id"] for row in non_test_rows), fold
            )
            validation_patients=AttentionStage.ensure_presence_coverage(
                non_test_rows, validation_patients, fold
            )
            train_rows=[
                row
                for row in non_test_rows
                if row["patient_id"] not in validation_patients
            ]
            validation_rows=[
                row
                for row in non_test_rows
                if row["patient_id"] in validation_patients
            ]
            if not train_rows or not validation_rows:
                raise RuntimeError(f"Fold {fold}: train or validation split is empty.")

            train_positive=sum(
                DataStage._as_int(row.get("heart_present"), 1) == 1 for row in train_rows
            )
            train_negative=len(train_rows) - train_positive
            validation_positive=sum(
                DataStage._as_int(row.get("heart_present"), 1) == 1
                for row in validation_rows
            )
            validation_negative=len(validation_rows) - validation_positive
            print(
                f"[ATTENTION] Fold {fold}: train={len(train_rows)} "
                f"(+{train_positive}/-{train_negative}), validation={len(validation_rows)} "
                f"(+{validation_positive}/-{validation_negative}), device={device.type}"
            )

            train_dataset=SegmentationDataset(
                train_rows,
                augment=True,
                seed=42 + fold * 1000,
            )
            validation_dataset=SegmentationDataset(
                validation_rows,
                augment=False,
                seed=42,
            )
            train_sampler=AttentionStage.training_sampler(
                train_rows, 42 + fold
            )
            train_loader=DataLoader(
                train_dataset,
                batch_size=10 if device.type == "cuda" else 3,
                sampler=train_sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            validation_loader=DataLoader(
                validation_dataset,
                batch_size=10 if device.type == "cuda" else 3,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )

            model=DataStage.prepare_model(AttentionUNet(), device)
            optimizer=torch.optim.AdamW(
                model.parameters(),
                lr=8e-4,  # initial AdamW learning rate used in the research run
                weight_decay=1e-4,  # small L2 regularization
            )
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )
            scaler=DataStage.grad_scaler(device)
            best_state=None
            best_metric=-1.0
            best_epoch=0
            patience=0
            history=[]

            for epoch in range(1, 36 + 1):  # train at most 36 epochs per fold
                train_dataset.set_epoch(epoch)
                model.train()
                train_losses=[]
                for (
                    images,
                    masks,
                    presence_targets,
                    sample_weights,
                    _patient_ids,
                ) in train_loader:
                    images=DataStage.move_tensor(images, device)
                    masks=DataStage.move_tensor(masks, device)
                    presence_targets=DataStage.move_tensor(
                        presence_targets, device
                    )
                    sample_weights=DataStage.move_tensor(
                        sample_weights, device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with DataStage.autocast(device):
                        logits, presence_logits=model(images)
                        loss=AttentionStage.segmentation_loss(
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
                validation_losses=[]
                by_patient_scores=defaultdict(list)
                presence_labels=[]
                presence_predictions=[]
                with torch.inference_mode():
                    for (
                        images,
                        masks,
                        presence_targets,
                        sample_weights,
                        patient_ids,
                    ) in validation_loader:
                        images=DataStage.move_tensor(images, device)
                        masks=DataStage.move_tensor(masks, device)
                        presence_targets_device=DataStage.move_tensor(
                            presence_targets, device
                        )
                        sample_weights=DataStage.move_tensor(
                            sample_weights, device
                        )
                        with DataStage.autocast(device):
                            logits, presence_logits=model(images)
                            loss=AttentionStage.segmentation_loss(
                                logits,
                                masks,
                                presence_logits,
                                presence_targets_device,
                                sample_weights,
                            )
                        validation_losses.append(float(loss.cpu()))
                        binary=(torch.sigmoid(logits) >= 0.5).float()
                        intersection=(binary * masks).sum(dim=(1, 2, 3))
                        denominator=binary.sum(dim=(1, 2, 3)) + masks.sum(
                            dim=(1, 2, 3)
                        )
                        dice=(2 * intersection + 1e-6) / (
                            denominator + 1e-6
                        )
                        negative_quality=1.0 - binary.mean(dim=(1, 2, 3))
                        target_presence_cpu=presence_targets.numpy().astype(int)
                        sample_scores=torch.where(
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

                mean_train=float(np.mean(train_losses))
                mean_val=float(np.mean(validation_losses))
                patient_balanced=float(
                    np.mean(
                        [np.mean(values) for values in by_patient_scores.values()]
                    )
                )
                presence_balanced=AttentionStage.balanced_accuracy(
                    np.asarray(presence_labels),
                    np.asarray(presence_predictions),
                )
                selection_metric=0.80 * patient_balanced + 0.20 * presence_balanced
                scheduler.step(selection_metric)
                current_lr=float(optimizer.param_groups[0]["lr"])
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
                    best_metric=selection_metric
                    best_epoch=epoch
                    best_state={
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    patience=0
                else:
                    patience +=1
                    if patience >= 8:
                        print(f"  early stopping after epoch {epoch}.")
                        break

            if best_state is None:
                raise RuntimeError(f"Fold {fold}: no best model state was selected.")
            model.load_state_dict(best_state)
            geometry_prior=AttentionStage.build_geometry_prior(train_rows)
            calibration=AttentionStage.calibrate_threshold(
                model, validation_loader, device, geometry_prior
            )
            fold_state={
                "schema": "simple-attention-2p5d-presence-v3",
                "state_dict": best_state,
                    "fold": fold,
                "base_channels": 24,
                "input_channels": 3,
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
            # CPU tensors are retained only until this fold has produced its OOF masks.
            fold_states[fold]=fold_state
            del model, optimizer, scheduler, scaler, train_loader, validation_loader
            del train_dataset, validation_dataset, fold_state, best_state, train_sampler
            DataStage.release_device(device)
            print(f"[ATTENTION] fold {fold} trained in "
                  f"{DataStage.format_seconds(time.perf_counter() - fold_started)}")

        DataStage.clear_image_cache()
        return fold_states
    # Keep the dominant connected mask component and discard small detached islands.
    @staticmethod
    def largest_component(mask):
        binary=(np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return binary
        count, labels, stats, _=cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if count <= 1:
            return binary
        label=1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == label).astype(np.uint8)
    # Measure mask area and the fraction touching the image boundary.
    @staticmethod
    def mask_geometry(mask):
        mask=(np.asarray(mask) > 0).astype(np.uint8)
        area=float(mask.mean())
        if not mask.any():
            return area, 0.0
        border=np.zeros_like(mask, dtype=bool)
        border[:2, :]=True
        border[-2:, :]=True
        border[:, :2]=True
        border[:, -2:]=True
        boundary_touch=float(
            np.logical_and(mask > 0, border).sum() / max(1, mask.sum())
        )
        return area, boundary_touch
    # Threshold, close small gaps, and keep the largest connected Attention region.
    @staticmethod
    def candidate_mask(probability, threshold):
        mask=(probability >= float(threshold)).astype(np.uint8)
        kernel_size=5  # close small holes/gaps in thresholded masks
        if kernel_size > 1:
            kernel=np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return AttentionStage.largest_component(mask)
    # Validate mask.
    @staticmethod
    def validate_mask(
        mask,
        probability,
        geometry_prior=None,
    ):
        area, boundary=AttentionStage.mask_geometry(mask)
        peak=float(np.max(probability))
        reasons=[]
        if area < 0.003:
            reasons.append("area_too_small")
        if area > 0.65:  # reject implausibly large heart masks
            reasons.append("area_too_large")
        if peak < 0.50:
            reasons.append("low_peak_probability")
        if boundary > 0.35:
            reasons.append("touches_boundary")
        prior_deviation=AttentionStage.calculate_prior_deviation(mask, geometry_prior)
        if prior_deviation > 5.5:
            reasons.append("geometry_outlier")
        return not reasons, ";".join(reasons)
    # Try nearby thresholds and geometry priors when the first automatic mask is implausible.
    @staticmethod
    def repair_probability(
        probability,
        threshold,
        geometry_prior=None,
    ):
        candidates=[]
        thresholds=[threshold] + [
            float(np.clip(threshold + offset, 0.05, 0.95))
            for offset in (-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20)
        ]
        for candidate_threshold in sorted(set(thresholds)):
            mask=AttentionStage.candidate_mask(
                probability, candidate_threshold
            )
            valid, reason=AttentionStage.validate_mask(
                mask, probability, geometry_prior
            )
            _area, boundary=AttentionStage.mask_geometry(mask)
            prior_deviation=AttentionStage.calculate_prior_deviation(
                mask, geometry_prior
            )
            mean_inside=(
                float(probability[mask > 0].mean()) if mask.any() else 0.0
            )
            score=(
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
        valid_candidates=[candidate for candidate in candidates if candidate[0]]
        if valid_candidates:
            _, _, mask, used_threshold, _, prior_deviation=max(
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
        _, _, mask, used_threshold, reason, prior_deviation=max(
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
    # Combine entropy, TTA disagreement, presence ambiguity, and geometry deviation.
    @staticmethod
    def uncertainty_metrics(
        probability,
        alternate_probability,
        presence_probability,
        presence_threshold,
        prior_deviation,
        final_valid,
    ):
        probability=np.clip(probability.astype(np.float64), 1e-6, 1.0 - 1e-6)
        entropy=-(
            probability * np.log(probability)
            + (1.0 - probability) * np.log(1.0 - probability)
        ) / math.log(2.0)
        entropy_flat=entropy.reshape(-1)
        top_count=max(1, int(round(0.10 * entropy_flat.size)))
        mean_entropy=float(
            np.partition(entropy_flat, entropy_flat.size - top_count)[-top_count:].mean()
        )
        disagreement=float(
            np.mean(np.abs(probability - alternate_probability))
        )
        disagreement_normalized=min(
            1.0,
            disagreement
            / max(0.08, 1e-6),
        )
        presence_ambiguity=max(
            0.0,
            1.0
            - abs(float(presence_probability) - float(presence_threshold)) / 0.50,
        )
        prior_normalized=min(
            1.0,
            float(prior_deviation)
            / max(5.5, 1e-6),
        )
        uncertainty=(
            0.35 * mean_entropy
            + 0.35 * disagreement_normalized
            + 0.20 * presence_ambiguity
            + 0.10 * prior_normalized
            + (0.15 if not final_valid else 0.0)
        )
        return disagreement, mean_entropy, float(min(1.0, uncertainty))
    # Apply sequence consistency.
    @staticmethod
    def apply_sequence_consistency(
        prediction_rows,
        dataset_rows,
    ):
        base_by_token={
            str(row["image_token"]): row for row in dataset_rows
        }
        groups=defaultdict(list)
        for prediction in prediction_rows:
            token=str(prediction.get("image_token", ""))
            base=base_by_token.get(token, {})
            group=str(
                base.get("sequence_group_id")
                or f"{prediction.get('series_id', '')}::{Path(str(prediction.get('image_path', ''))).parent}"
            )
            prediction["_sequence_index"]=DataStage._as_int(
                base.get("sequence_index"), 0
            )
            groups[group].append(prediction)

        def pair_inconsistency(first, second):
            first_present=DataStage._as_int(first.get("attention_heart_present"), 1)
            second_present=DataStage._as_int(second.get("attention_heart_present"), 1)
            if first_present != second_present:
                return 1.0
            if first_present == 0:
                return 0.0
            first_area=DataStage._as_float(first.get("attention_area_ratio"), 0.0)
            second_area=DataStage._as_float(second.get("attention_area_ratio"), 0.0)
            area_change=min(
                1.0,
                abs(first_area - second_area)
                / max(0.01, 0.5 * (first_area + second_area)),
            )
            first_x=DataStage._as_float(first.get("attention_centroid_x"), np.nan)
            first_y=DataStage._as_float(first.get("attention_centroid_y"), np.nan)
            second_x=DataStage._as_float(second.get("attention_centroid_x"), np.nan)
            second_y=DataStage._as_float(second.get("attention_centroid_y"), np.nan)
            if all(np.isfinite(value) for value in (first_x, first_y, second_x, second_y)):
                centroid_change=min(
                    1.0,
                    math.hypot(first_x - second_x, first_y - second_y) / 0.25,
                )
            else:
                centroid_change=1.0
            return float(0.55 * area_change + 0.45 * centroid_change)

        for group_rows in groups.values():
            group_rows.sort(key=lambda row: DataStage._as_int(row.get("_sequence_index"), 0))
            for index, row in enumerate(group_rows):
                comparisons=[]
                if index > 0:
                    comparisons.append(pair_inconsistency(row, group_rows[index - 1]))
                if index + 1 < len(group_rows):
                    comparisons.append(pair_inconsistency(row, group_rows[index + 1]))
                inconsistency=float(np.mean(comparisons)) if comparisons else 0.0
                previous_inconsistency=DataStage._as_float(
                    row.get("attention_sequence_inconsistency"), 0.0
                )
                base_uncertainty=max(
                    0.0,
                    DataStage._as_float(row.get("attention_uncertainty_score"), 0.0)
                    - 0.20 * previous_inconsistency,
                )
                row["attention_sequence_inconsistency"]=inconsistency
                row["attention_uncertainty_score"]=min(
                    1.0, base_uncertainty + 0.20 * inconsistency
                )
                row.pop("_sequence_index", None)
    # Compare in-memory OOF predictions with accepted persistent manual targets.
    @staticmethod
    def evaluate_oof_segmentation(
        manual_audit,
        prediction_rows,
    ):
        predictions={
            str(row.get("image_token", "")): row for row in prediction_rows
        }
        audit_rows=[
            row
            for row in manual_audit
            if row.get("status") == "ACCEPTED"
        ]
        metrics=[]
        for target_row in audit_rows:
            token=str(target_row.get("image_token", ""))
            prediction_row=predictions.get(token)
            if prediction_row is None:
                continue
            target_path=Path(target_row.get("manual_mask_path", ""))
            if not target_path.is_file():
                continue
            predicted=AttentionStage.read_prediction_mask(prediction_row)
            target=DataStage.read_binary_mask(target_path)
            predicted_bool=predicted > 0
            target_bool=target > 0
            tp=float(np.logical_and(predicted_bool, target_bool).sum())
            fp=float(np.logical_and(predicted_bool, ~target_bool).sum())
            fn=float(np.logical_and(~predicted_bool, target_bool).sum())
            union=tp + fp + fn
            denominator=2.0 * tp + fp + fn
            dice=1.0 if denominator == 0 else 2.0 * tp / denominator
            iou=1.0 if union == 0 else tp / union
            precision=1.0 if tp + fp == 0 and not target_bool.any() else tp / max(1.0, tp + fp)
            recall=1.0 if tp + fn == 0 else tp / max(1.0, tp + fn)
            target_features=AttentionStage.mask_features(target)
            predicted_features=AttentionStage.mask_features(predicted)
            if target_bool.any() and predicted_bool.any():
                centroid_distance=float(
                    math.hypot(
                        predicted_features["centroid_x"] - target_features["centroid_x"],
                        predicted_features["centroid_y"] - target_features["centroid_y"],
                    )
                )
            else:
                centroid_distance=np.nan
            metrics.append(
                {
                    "image_token": token,
                    "patient_id": target_row.get("patient_id", ""),
                    "series_id": target_row.get("series_id", ""),
                    "target_type": target_row.get("target_type", ""),
                    "heart_present": DataStage._as_int(target_row.get("heart_present"), 1),
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
                    "presence_probability": DataStage._as_float(
                        prediction_row.get("attention_presence_probability"), np.nan
                    ),
                    "predicted_heart_present": DataStage._as_int(
                        prediction_row.get("attention_heart_present"), 1
                    ),
                    "uncertainty_score": DataStage._as_float(
                        prediction_row.get("attention_uncertainty_score"), np.nan
                    ),
                    "sequence_inconsistency": DataStage._as_float(
                        prediction_row.get("attention_sequence_inconsistency"), np.nan
                    ),
                }
            )

        positive=[row for row in metrics if DataStage._as_int(row["heart_present"], 1) == 1]
        negative=[row for row in metrics if DataStage._as_int(row["heart_present"], 1) == 0]

        def patient_balanced(rows, key):
            by_patient=defaultdict(list)
            for row in rows:
                value=DataStage._as_float(row.get(key), np.nan)
                if np.isfinite(value):
                    by_patient[str(row.get("patient_id", ""))].append(value)
            values=[np.mean(items) for items in by_patient.values() if items]
            return float(np.mean(values)) if values else np.nan

        summary={
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
                            < 0.003
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
        }
        if metrics:
            print(
                "[ATTENTION OOF] "
                f"targets={len(metrics)}, positive_dice_patient="
                f"{summary['patient_balanced_dice_positive']:.4f}, "
                f"negative_empty_rate={summary['no_heart_empty_prediction_rate']}"
            )
        return summary, pd.DataFrame(metrics)
    # Predict every image from freshly trained OOF states, without saving automatic masks.
    @staticmethod
    def predict_attention_masks(rows, device, fold_states):
        # Automatic masks are never reused. Every run predicts the complete dataset
        # from the five freshly trained out-of-fold models.
        results=[None] * len(rows)
        started=time.perf_counter()

        for fold in range(5):  # five patient-level folds
            fold_indices=[
                index
                for index, row in enumerate(rows)
                if int(row["segmentation_fold"]) == fold
            ]
            fold_rows=[rows[index] for index in fold_indices]
            if not fold_rows:
                continue

            fold_state=fold_states.pop(fold)
            held_out={str(row["patient_id"]) for row in fold_rows}
            seen=set(fold_state["train_patients"]) | set(fold_state["validation_patients"])
            if held_out & seen:
                raise RuntimeError(f"Fold {fold}: OOF prediction would leak held-out patients.")
            model=AttentionUNet(
                base_channels=int(
                    fold_state.get(
                        "base_channels", 24
                    )
                ),
                input_channels=int(
                    fold_state.get(
                        "input_channels", 3
                    )
                ),
            )
            model.load_state_dict(fold_state["state_dict"])
            model=DataStage.prepare_model(model, device).eval()
            calibration=fold_state.get("calibration", {}) or {}
            threshold=float(
                calibration.get(
                    "threshold", 0.50
                )
            )
            presence_threshold=float(
                calibration.get(
                    "presence_threshold",
                    0.50,
                )
            )
            geometry_prior=fold_state.get("geometry_prior", {}) or {}
            loader=DataLoader(
                SegmentationInferenceDataset(fold_rows),
                batch_size=10 if device.type == "cuda" else 3,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            print(
                f"[ATTENTION] 2.5D prediction for fold {fold}: {len(fold_rows)} "
                f"images, device={device.type}"
            )
            fold_results=[]

            with torch.inference_mode():
                for images, local_indices in tqdm(
                    loader, desc=f"Attention 2.5D fold {fold}"):
                    images=DataStage.move_tensor(images, device)
                    with DataStage.autocast(device):
                        logits, presence_logits=model(images)
                    original_probability=torch.sigmoid(logits)[:, 0]
                    original_presence=torch.sigmoid(presence_logits)

                    factor=1.10  # second photometric view for uncertainty estimation
                    tta_images=torch.clamp(
                        (images - 0.5) * factor + 0.5, 0.0, 1.0
                    )
                    with DataStage.autocast(device):
                        tta_logits, tta_presence_logits=model(tta_images)
                    tta_probability=torch.sigmoid(tta_logits)[:, 0]
                    tta_presence=torch.sigmoid(tta_presence_logits)

                    averaged_probability=(
                        original_probability + tta_probability
                    ) / 2.0
                    averaged_presence=(
                        original_presence + tta_presence
                    ) / 2.0
                    probabilities=averaged_probability.float().cpu().numpy()
                    alternate_probabilities=tta_probability.float().cpu().numpy()
                    presence_probabilities=(
                        averaged_presence.float().cpu().numpy()
                    )

                    for batch_position, local_index_tensor in enumerate(
                        local_indices):
                        local_index=int(local_index_tensor)
                        global_index=fold_indices[local_index]
                        row=rows[global_index]
                        probability=probabilities[batch_position]
                        alternate_probability=alternate_probabilities[
                            batch_position
                        ]
                        presence_probability=float(
                            presence_probabilities[batch_position]
                        )
                        heart_present=bool(
                            presence_probability >= presence_threshold
                            or float(probability.max())
                            >= 0.80
                        )
                        initial_mask=AttentionStage.candidate_mask(
                            probability, threshold
                        )
                        initial_valid, initial_reason=(
                            AttentionStage.validate_mask(
                                initial_mask, probability, geometry_prior
                            )
                        )

                        if not heart_present:
                            final_mask=np.zeros_like(initial_mask, dtype=np.uint8)
                            final_valid=True
                            repair_method="presence_head_empty"
                            used_threshold=threshold
                            final_reason=""
                            prior_deviation=0.0
                        elif initial_valid:
                            final_mask=initial_mask
                            final_valid=True
                            repair_method="none"
                            used_threshold=threshold
                            final_reason=""
                            prior_deviation=AttentionStage.calculate_prior_deviation(
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
                            )=AttentionStage.repair_probability(
                                probability, threshold, geometry_prior
                            )
                            repair_method="photometric_tta+" + repair_method

                        geometry=AttentionStage.mask_features(final_mask)
                        area=float(geometry["area"])
                        boundary=float(geometry["boundary"])
                        centroid_x=geometry["centroid_x"]
                        centroid_y=geometry["centroid_y"]
                        disagreement, entropy, uncertainty=(
                            AttentionStage.uncertainty_metrics(
                                probability,
                                alternate_probability,
                                presence_probability,
                                presence_threshold,
                                prior_deviation,
                                final_valid,
                            )
                        )
                        # One bit per pixel, lossless: 8192 bytes for a 256 x 256 mask.
                        packed_mask=np.packbits(final_mask, axis=None)

                        result_row={
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
                            "attention_mask_bits": packed_mask,
                        }
                        results[global_index]=result_row
                        fold_results.append(result_row)

            AttentionStage.apply_sequence_consistency(
                fold_results, fold_rows
            )
            del model, loader, fold_state, fold_results
            DataStage.release_device(device)

        final_results=[result for result in results if result is not None]
        if len(final_results) != len(rows):
            raise RuntimeError(
                "Prediction did not produce exactly one row per image."
            )
        AttentionStage.apply_sequence_consistency(final_results, rows)
        invalid_rows=[
            row
            for row in final_results
            if DataStage._as_int(row.get("attention_valid_final"), 0) != 1
        ]
        summary={
            "device": device.type,
            "images": len(final_results),
            "valid_initial": sum(
                DataStage._as_int(row.get("attention_valid_initial"), 0)
                for row in final_results
            ),
            "valid_final": sum(
                DataStage._as_int(row.get("attention_valid_final"), 0)
                for row in final_results
            ),
            "predicted_heart_present": sum(
                DataStage._as_int(row.get("attention_heart_present"), 0)
                for row in final_results
            ),
            "predicted_no_heart": sum(
                DataStage._as_int(row.get("attention_heart_present"), 0) == 0
                for row in final_results
            ),
            "uncertain_above_review_threshold": sum(
                DataStage._as_float(row.get("attention_uncertainty_score"), 0.0)
                >= 0.18
                for row in final_results
            ),
            "invalid_final": len(invalid_rows),
            "elapsed": DataStage.format_seconds(
                time.perf_counter() - started
            ),
        }
        print(
            "[ATTENTION] "
            f"valid_final={summary['valid_final']}/{summary['images']}, "
            f"heart_present={summary['predicted_heart_present']}, "
            f"invalid={summary['invalid_final']} | automatic masks retained in RAM"
        )
        DataStage.clear_image_cache()
        return final_results, summary

    # Decode a losslessly packed automatic mask; resize exactly as for the former PNG.
    @staticmethod
    def read_prediction_mask(row, size=256):
        packed = row.get("attention_mask_bits")
        if packed is None:
            raise RuntimeError(f"No current-session OOF mask for {row.get('image_token', '?')}.")
        packed = np.asarray(packed, dtype=np.uint8)
        if packed.size != 8192:
            raise RuntimeError("An automatic mask must contain exactly 256 x 256 packed bits.")
        mask = np.unpackbits(packed).reshape(256, 256)
        if int(size) != 256:
            mask = cv2.resize(mask, (int(size), int(size)), interpolation=cv2.INTER_NEAREST)
        return mask

# =============================================================================
# STAGE 3 — OPTIONAL HTML REVIEW
# =============================================================================
class HammingBKTree:
    """Search perceptual hashes by Hamming distance without scanning every hash."""

    def __init__(self):
        self.root=None

    def add(self, value):
        value=int(value)
        if self.root is None:
            self.root=(value, {})
            return
        node_value, children=self.root
        while True:
            distance=int(value ^ node_value).bit_count()
            child=children.get(distance)
            if child is None:
                children[distance]=(value, {})
                return
            node_value, children=child

    def has_near(self, value, maximum_distance):
        if self.root is None:
            return False
        stack=[self.root]
        while stack:
            node_value, children=stack.pop()
            distance=int(value ^ node_value).bit_count()
            if distance <= maximum_distance:
                return True
            low=distance - maximum_distance
            high=distance + maximum_distance
            stack.extend(child for edge, child in children.items() if low <= edge <= high)
        return False

class ReviewStage:
    """Select unresolved images for manual review in the current session."""

    # Add the latest manual decisions to the current-session image records.
    @staticmethod
    def merge_review_rows(
        dataset_rows,
        workspace,
    ):
        annotations=DataStage.load_annotations(workspace)
        merged=[]
        for original in dataset_rows:
            row=dict(original)
            annotation=annotations.get(str(row["image_token"]), {})
            annotation_type=ReviewStage.normalize_target_type(
                annotation.get("target_type", "")
            )
            row["manual_annotation_type"]=annotation_type
            row["review_target_type"]=annotation_type or "UNLABELED"
            row["manual_annotation_source"]=annotation.get("source", "")
            mask_qc=DataStage.manual_mask_qc(row["manual_mask_path"])
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
    # Normalize target type.
    @staticmethod
    def normalize_target_type(value):
        target=str(value or "").strip().upper()
        if target in {"", "UNLABELED", "NONE", "NAN"}:
            return ""
        return target
    # Read the normalized explicit review label attached to one image.
    @staticmethod
    def row_target_type(row):
        return ReviewStage.normalize_target_type(
            row.get("manual_annotation_type", row.get("target_type", ""))
        )
    # Distribute review candidates across patients and series instead of oversampling one scan.
    @staticmethod
    def round_robin_review(
        rows,
        limit,
        per_patient,
        seed,
    ):
        def priority(row):
            explicit_priority=row.get("review_priority", "")
            if str(explicit_priority).strip() != "":
                primary=DataStage._as_float(explicit_priority, 0.0)
            else:
                uncertainty=DataStage._as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
                valid=DataStage._as_int(row.get("attention_valid_final"), 0)
                primary=-uncertainty if uncertainty > 0 else float(valid)
            peak=DataStage._as_float(row.get("attention_peak_probability"), 0.0)
            tie=hashlib.sha256(
                f"{seed}|{row['image_token']}".encode("utf-8")
            ).hexdigest()
            return (primary, peak, tie)

        by_patient_series=defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            sequence_group=str(
                row.get("sequence_group_id") or row["series_id"]
            )
            by_patient_series[str(row["patient_id"])][sequence_group].append(row)

        patient_queues={}
        for patient_id, series_map in by_patient_series.items():
            series_ids=sorted(series_map)
            for series_id in series_ids:
                series_map[series_id].sort(key=priority)
            queue=[]
            position=0
            while len(queue) < per_patient:
                added=False
                for series_id in series_ids:
                    if position < len(series_map[series_id]):
                        queue.append(series_map[series_id][position])
                        added=True
                        if len(queue) >= per_patient:
                            break
                if not added:
                    break
                position +=1
            patient_queues[patient_id]=queue

        selected=[]
        position=0
        while len(selected) < limit:
            added=False
            for patient_id in sorted(patient_queues):
                queue=patient_queues[patient_id]
                if position < len(queue):
                    selected.append(queue[position])
                    added=True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            position +=1
        return selected
    # Exclude old decisions, but keep labeled images navigable inside the open editor.
    @staticmethod
    def select_review_rows(
        dataset_rows,
        workspace,
        scope="invalid",
        limit=None,
        seed=42,
    ):
        scope=str(scope).lower()
        allowed={"invalid", "uncertain", "empty", "novel", "manual", "all"}
        if scope not in allowed:
            raise ValueError(
                "scope must be invalid/uncertain/empty/novel/manual/all."
            )
        limit=int(300 if limit is None else limit)
        all_rows=ReviewStage.merge_review_rows(dataset_rows, workspace)
        target_counts=defaultdict(int)
        for row in all_rows:
            target=ReviewStage.row_target_type(row) or "UNLABELED"
            target_counts[target] +=1
        rows=[row for row in all_rows if ReviewStage.row_target_type(row) == ""]
        labeled_excluded=len(all_rows) - len(rows)
        print(
            "[REVIEW][TARGET FILTER] "
            f"UNLABELED={len(rows)}, already-labeled excluded={labeled_excluded}, "
            f"distribution={dict(sorted(target_counts.items()))}"
        )

        if scope == "invalid":
            candidates=[
                row
                for row in rows
                if DataStage._as_int(row.get("attention_valid_final"), 1) == 0
                and row.get("attention_mask_bits") is not None
            ]
            for row in candidates:
                row["review_priority"]=DataStage._as_float(
                    row.get("attention_peak_probability"), 0.0
                )
            candidates=ReviewStage.round_robin_review(
                candidates,
                limit=max(1, limit),
                per_patient=max(
                    10,
                    int(math.ceil(limit / max(1, len({r['patient_id'] for r in candidates}))))
                    if candidates
                    else 10,
                ),
                seed=seed,
            )
        elif scope == "uncertain":
            candidates=[]
            for row in rows:
                if DataStage._as_int(row.get("quality_valid"), 0) != 1:
                    continue
                if row.get("attention_mask_bits") is None:
                    continue
                uncertainty=DataStage._as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
                presence=DataStage._as_float(
                    row.get("attention_presence_probability"), np.nan
                )
                presence_threshold=DataStage._as_float(
                    row.get("attention_presence_threshold"),
                    0.50,
                )
                near_presence_boundary=bool(
                    np.isfinite(presence)
                    and abs(presence - presence_threshold)
                    <= 0.15
                )
                if (
                    uncertainty < 0.18
                    and not near_presence_boundary
                    and DataStage._as_int(row.get("attention_valid_final"), 1) == 1):
                    continue

                row["review_priority"]=-uncertainty
                candidates.append(row)
            candidates=ReviewStage.round_robin_review(
                candidates,
                limit=max(1, limit),
                per_patient=max(
                    10,
                    int(math.ceil(limit / max(1, len({r['patient_id'] for r in candidates}))))
                    if candidates
                    else 10,
                ),
                seed=seed,
            )
        elif scope == "empty":
            candidates=[
                row
                for row in rows
                if Path(row["manual_mask_path"]).is_file()
                and not bool(row.get("manual_annotation_type"))
                and str(row.get("manual_mask_reason", ""))
                == "empty_or_nearly_empty"
            ]
            for row in candidates:
                row["review_priority"]=-DataStage._as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
            candidates=ReviewStage.round_robin_review(
                candidates,
                limit=max(1, limit),
                per_patient=max(
                    10,
                    int(math.ceil(limit / max(1, len({r['patient_id'] for r in candidates}))))
                    if candidates
                    else 10,
                ),
                seed=seed,
            )
        elif scope == "manual":

            candidates=[
                row
                for row in rows
                if Path(row["manual_mask_path"]).is_file()
            ]
            candidates.sort(
                key=lambda row: (
                    row["patient_id"],
                    row["series_id"],
                    row["image_token"],
                )
            )
        elif scope == "all":
            candidates=list(rows)
            candidates.sort(
                key=lambda row: (
                    row["patient_id"],
                    row["series_id"],
                    row["image_token"],
                )
            )
        else:

            manual_hashes=[]
            for row in all_rows:
                is_annotated=bool(row.get("manual_annotation_type")) or Path(
                    row["manual_mask_path"]
                ).is_file()
                if is_annotated and row.get("perceptual_hash"):
                    manual_hashes.append(int(str(row["perceptual_hash"]), 16))
            if not manual_hashes:
                raise RuntimeError(
                    "The novel scope requires at least one labeled image."
                )
            tree=HammingBKTree()
            for value in sorted(set(manual_hashes)):
                tree.add(value)

            candidates=[]
            excluded=defaultdict(int)
            for row in rows:
                if bool(row.get("manual_annotation_type")) or Path(
                    row["manual_mask_path"]
                ).is_file():
                    excluded["already_annotated"] +=1
                    continue
                if DataStage._as_int(row.get("quality_valid"), 0) != 1:
                    excluded["quality_invalid"] +=1
                    continue
                phash=str(row.get("perceptual_hash", ""))
                if not phash:
                    excluded["missing_phash"] +=1
                    continue
                if tree.has_near(
                    int(phash, 16), 6):
                    excluded["similar_to_annotated"] +=1
                    continue
                row["review_priority"]=-DataStage._as_float(
                    row.get("attention_uncertainty_score"), 0.0
                )
                candidates.append(row)
            candidates=ReviewStage.round_robin_review(
                candidates,
                limit=max(1, limit),
                per_patient=10,
                seed=seed,
            )
            print(f"[REVIEW novel] excluded={dict(excluded)}")

        if limit > 0 and scope in {"manual", "all"}:
            candidates=candidates[:limit]
        if not candidates:
            raise RuntimeError(
                "No target=UNLABELED images are eligible for "
                f"scope={scope!r}. HEART_PRESENT, "
                "NO_HEART_VISIBLE, and UNUSABLE images are excluded automatically."
            )
        print(f"[REVIEW] scope={scope}, images={len(candidates)}")
        return candidates

class MaskEditor:
    """Kaggle-safe HTML5 editor for correcting segmentation targets.

    Keyboard listeners are installed once and isolated from notebook command mode.
    Labeled images remain navigable during the current session but are excluded
    when a later review queue is constructed.
    """

    def __init__(
        self,
        rows,
        workspace,
        start_index=0,
        brush_radius=8,
        review_scope="invalid",):
        try:
            widgets=importlib.import_module("ipywidgets")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "The HTML review editor requires ipywidgets in the active Python/Jupyter "
                "environment. Install it with: python -m pip install ipywidgets"
            ) from error

        annotations=DataStage.load_annotations(workspace)
        unlabeled_rows=[]
        for source_row in rows:
            row=dict(source_row)
            annotation=annotations.get(str(row.get("image_token", "")), {})
            target=ReviewStage.normalize_target_type(
                annotation.get("target_type", row.get("manual_annotation_type", ""))
            )
            if target:
                continue
            row["manual_annotation_type"]=""
            row["review_target_type"]="UNLABELED"
            unlabeled_rows.append(row)
        if not unlabeled_rows:
            raise ValueError(
                "The editor has no remaining target=UNLABELED images. "
                "Previously decided targets are not reopened."
            )

        self.rows=unlabeled_rows
        self.workspace=workspace
        self.index=int(np.clip(start_index, 0, len(self.rows) - 1))
        self.brush_radius=max(1, int(brush_radius))
        self.review_scope=str(review_scope)
        self.image=None
        self.auto_mask=None
        self.base_mask=None
        self.mask=None
        self.widget_id=uuid.uuid4().hex[:12]
        self._shortcut_busy=False
        self._last_shortcut_payload=""
        self._last_shortcut_at=0.0

        self.output=widgets.Output()
        self.mask_sync=widgets.Textarea(
            value="",
            placeholder=f"CAD_MASK_SYNC_{self.widget_id}",
            layout=widgets.Layout(width="1px", height="1px", display="none"),
        )
        self.shortcut_sync=widgets.Textarea(
            value="",
            placeholder=f"CAD_SHORTCUT_SYNC_{self.widget_id}",
            layout=widgets.Layout(width="1px", height="1px", display="none"),
        )

        def compact_button(
            description,
            width,
            tooltip,
            button_style="",):
            return widgets.Button(
                description=description,
                tooltip=tooltip,
                button_style=button_style,
                layout=widgets.Layout(width=width, height="29px"),
                style={"font_weight": "500"},
            )

        self.previous_button=compact_button("← Prev [P]", "68px", "Previous — P")
        self.next_button=compact_button("Next [N] →", "72px", "Next — N")
        self.save_next_button=compact_button(
            "Save + Next [S]", "102px", "Save HEART_PRESENT mask and go next — S", "success"
        )
        self.raw_button=compact_button(
            "Image [I]", "74px", "Hide overlays and start from the image — I"
        )
        self.reset_button=compact_button(
            "Auto reset [O]", "94px", "Restore the automatic mask — O"
        )
        self.accept_button=compact_button(
            "Auto OK [A]", "84px", "Save confirmed auto mask for training — A", "info"
        )
        self.no_heart_button=compact_button(
            "No heart [H]", "92px", "Valid image, but no heart is visible — H", "info"
        )
        self.unusable_button=compact_button(
            "Unusable [U]", "92px", "Blur/noise/localizer: exclude from training — U", "warning"
        )
        self.skip_button=compact_button("Skip [K]", "66px", "Skip and go next — K")
        self.clear_button=compact_button("Clear [C]", "68px", "Clear editable mask — C")
        self.delete_button=compact_button(
            "Delete [D]", "76px", "Delete manual target and label — D", "danger"
        )
        self.brush_slider=widgets.IntSlider(
            description="Brush",
            value=self.brush_radius,
            min=1,
            max=30,
            step=1,
            continuous_update=False,
            layout=widgets.Layout(width="260px"),
        )
        self.status=widgets.HTML()

        self.previous_button.on_click(lambda _: self._safe("Previous", self.previous))
        self.next_button.on_click(lambda _: self._safe("Next", self.next))
        self.save_next_button.on_click(lambda _: self._safe("Save & Next", self.save_next))
        self.accept_button.on_click(lambda _: self._safe("Auto OK", self.accept_auto))
        self.no_heart_button.on_click(lambda _: self._safe("No heart", self.mark_no_heart))
        self.unusable_button.on_click(lambda _: self._safe("Unusable", self.mark_unusable))
        self.skip_button.on_click(lambda _: self._safe("Skip", self.next))
        self.reset_button.on_click(lambda _: self._safe("Reset", self.reset_to_auto))
        self.raw_button.on_click(lambda _: self._safe("Reset to image", self.reset_to_image))
        self.clear_button.on_click(lambda _: self._safe("Clear", self.clear_editable))
        self.delete_button.on_click(lambda _: self._safe("Delete manual", self.delete_manual))
        self.brush_slider.observe(self._brush_changed, names="value")
        self.mask_sync.observe(self._mask_sync_changed, names="value")
        self.shortcut_sync.observe(self._shortcut_sync_changed, names="value")

        self.controls=widgets.VBox(
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

    def _safe(self, action, function):
        try:
            function()
        except Exception as error:
            message=f"{action}: {type(error).__name__}: {error}"
            self.status.value=f"<span style='color:#b00020'><b>{message}</b></span>"
            print("[EDITOR ERROR]", message)

    def _advance_after_target(self, message):
        if not self.rows:
            return
        completed_index=int(self.index)
        completed_token=str(self.rows[completed_index].get("image_token", ""))
        if completed_index < len(self.rows) - 1:
            self.index=completed_index + 1
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

    def _brush_changed(self, change):
        self.brush_radius=int(change["new"])
        if self.image is not None:
            self.render()

    @staticmethod
    def _image_uri(rgb):
        array=np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        ok, encoded=cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("The editor image could not be encoded.")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _mask_uri(mask):
        binary=(np.asarray(mask) > 0).astype(np.uint8)
        bgra=np.zeros((*binary.shape, 4), dtype=np.uint8)
        foreground=binary * 255
        bgra[..., :3]=foreground[..., None]
        bgra[..., 3]=foreground
        ok, encoded=cv2.imencode(".png", bgra)
        if not ok:
            raise RuntimeError("The editor mask could not be encoded.")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    def load_current(self):
        row=self.rows[self.index]
        image=DataStage.standardized_uint8(row["image_path"])
        # Manual drawing also works before OOF models have been trained in this session.
        auto_mask=(
            AttentionStage.read_prediction_mask(row)
            if row.get("attention_mask_bits") is not None
            else np.zeros((256, 256), dtype=np.uint8)
        )
        manual_path=Path(row["manual_mask_path"])
        current=DataStage.read_binary_mask(manual_path) if manual_path.is_file() else auto_mask.copy()
        self.image=image.astype(np.float32) / 255.0
        self.auto_mask=auto_mask.astype(np.uint8)
        self.base_mask=self.auto_mask.copy()
        self.mask=current.astype(np.uint8)
        self.render()
        self.update_status()

    def render(self):
        row=self.rows[self.index]
        target_type=(
            ReviewStage.normalize_target_type(
                row.get("manual_annotation_type", row.get("target_type", ""))
            )
            or "UNLABELED"
        )
        image_uri=self._image_uri(np.stack([self.image] * 3, axis=-1))
        base_uri=self._mask_uri(self.base_mask)
        mask_uri=self._mask_uri(self.mask)
        canvas_id=f"cad_canvas_{self.widget_id}"
        keyboard_sink_id=f"cad_keyboard_sink_{self.widget_id}"
        placeholder=f"CAD_MASK_SYNC_{self.widget_id}"
        size=256
        html=f"""
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
            Left drag=draw | Right drag=erase (Kaggle menu disabled) | S=Save & Next | P/N=Prev/Next |
            I=image | O=auto reset | A=Auto OK | H=no heart | U=unusable |
            K=Skip | C=Clear | D=Delete | [ / ]=Brush −/+
          </div>
        </div>
        """
        js=f"""
        (()=> {{
          const canvas=document.getElementById({json.dumps(canvas_id)});
          const keyboardSink=document.getElementById({json.dumps(keyboard_sink_id)});
          if (!canvas) return;
          const ctx=canvas.getContext('2d');
          const W={size}, H={size};
          const radius={int(self.brush_radius)};
          const placeholder={json.dumps(placeholder)};
          const image=new Image(), base=new Image(), initialMask=new Image();
          const maskCanvas=document.createElement('canvas');
          maskCanvas.width=W; maskCanvas.height=H;
          const maskCtx=maskCanvas.getContext('2d', {{willReadFrequently:true}});
          const pointerControllerKey='__cadMaskEditorPointerController';
          const oldPointerController=window[pointerControllerKey];
          if (oldPointerController && typeof oldPointerController.dispose === 'function') {{
            try {{ oldPointerController.dispose(); }} catch (_) {{}}
          }}

          let drawing=false, erase=false, last=null, rightDragActive=false;

          function focusKeyboardSink() {{
            if (!keyboardSink) return;
            try {{
              keyboardSink.focus({{preventScroll: true}});
            }} catch (_) {{
              try {{ keyboardSink.focus(); }} catch (_) {{}}
            }}
            keyboardSink.value='';
          }}

          function eventTargetsCanvas(event) {{
            if (event.target === canvas) return true;
            try {{
              const path=typeof event.composedPath === 'function'
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
            drawing=false;
            erase=false;
            rightDragActive=false;
            last=null;
          }}

          function hidden() {{
            return Array.from(document.querySelectorAll('textarea'))
              .find(x=> x.placeholder === placeholder);
          }}
          function overlay(source, color, alpha) {{
            const tmp=document.createElement('canvas');
            tmp.width=canvas.width; tmp.height=canvas.height;
            const c=tmp.getContext('2d');
            c.globalAlpha=alpha;
            c.drawImage(source, 0, 0, canvas.width, canvas.height);
            c.globalCompositeOperation='source-in';
            c.fillStyle=color;
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
            const target=hidden();
            if (!target) return;
            target.value=maskCanvas.toDataURL('image/png').split(',')[1];
            target.dispatchEvent(new Event('input', {{bubbles:true}}));
            target.dispatchEvent(new Event('change', {{bubbles:true}}));
          }}
          function point(e) {{
            const r=canvas.getBoundingClientRect();
            return {{
              x: Math.max(0, Math.min(W - 1, (e.clientX - r.left) / r.width * W)),
              y: Math.max(0, Math.min(H - 1, (e.clientY - r.top) / r.height * H))
            }};
          }}
          function paint(p) {{
            maskCtx.save();
            maskCtx.globalCompositeOperation=erase ? 'destination-out' : 'source-over';
            maskCtx.fillStyle='white'; maskCtx.strokeStyle='white';
            maskCtx.lineWidth=radius * 2; maskCtx.lineCap='round'; maskCtx.lineJoin='round';
            if (last) {{
              maskCtx.beginPath(); maskCtx.moveTo(last.x, last.y); maskCtx.lineTo(p.x, p.y); maskCtx.stroke();
            }} else {{
              maskCtx.beginPath(); maskCtx.arc(p.x, p.y, radius, 0, 2 * Math.PI); maskCtx.fill();
            }}
            maskCtx.restore(); last=p; draw();
          }}
          function begin(e) {{
            if (![0,2].includes(e.button)) return;
            focusKeyboardSink();
            const startsErase=e.button === 2;
            if (startsErase) blockEvent(e);
            else e.preventDefault();
            drawing=true;
            erase=startsErase;
            rightDragActive=startsErase;
            last=null;
            try {{ canvas.setPointerCapture?.(e.pointerId); }} catch (_) {{}}
            paint(point(e));
          }}
          function move(e) {{
            if (!drawing) return;
            const requiredButton=erase ? 2 : 1;
            if ((Number(e.buttons || 0) & requiredButton) === 0) return finish(e);
            if (erase) blockEvent(e);
            else e.preventDefault();
            paint(point(e));
          }}
          function finish(e) {{
            if (!drawing) {{
              if (e && e.button === 2) rightDragActive=false;
              return;
            }}
            const wasErase=erase;
            if (wasErase || (e && e.button === 2)) blockEvent(e);
            else e?.preventDefault?.();
            try {{ canvas.releasePointerCapture?.(e.pointerId); }} catch (_) {{}}
            drawing=false;
            erase=false;
            rightDragActive=false;
            last=null;
            sync();
            focusKeyboardSink();
          }}

          const activeOptions={{capture: true, passive: false}};
          // Local protection plus an early window listener blocks both the native
          // context menu and the menu installed by Kaggle/Jupyter.
          canvas.oncontextmenu=suppressContextMenu;
          canvas.addEventListener('contextmenu', suppressContextMenu, activeOptions);
          window.addEventListener('contextmenu', suppressContextMenu, activeOptions);
          canvas.addEventListener('auxclick', event=> {{
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
            canvas.addEventListener('mouseleave', event=> {{
              if (drawing && Number(event.buttons || 0) === 0) finish(event);
            }}, activeOptions);
          }}

          window.addEventListener('blur', releasePointerState, true);
          document.addEventListener('visibilitychange', releasePointerState, true);
          window[pointerControllerKey]={{
            editorId: {json.dumps(self.widget_id)},
            dispose: ()=> {{
              window.removeEventListener('contextmenu', suppressContextMenu, true);
              window.removeEventListener('blur', releasePointerState, true);
              document.removeEventListener('visibilitychange', releasePointerState, true);
              releasePointerState();
            }}
          }};
          function load(target, source) {{
            return new Promise((resolve, reject)=> {{ target.onload=resolve; target.onerror=reject; target.src=source; }});
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
          ]).then(()=> {{
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

    def _shortcut_script(self):
        placeholder=f"CAD_SHORTCUT_SYNC_{self.widget_id}"
        keyboard_sink_id=f"cad_keyboard_sink_{self.widget_id}"
        controller_key="__cadMaskEditorKeyboardController"
        return f"""
        (()=> {{
          const controllerKey={json.dumps(controller_key)};
          const editorId={json.dumps(self.widget_id)};
          const placeholder={json.dumps(placeholder)};
          const keyboardSinkId={json.dumps(keyboard_sink_id)};

          const oldController=window[controllerKey];
          if (oldController && typeof oldController.dispose === 'function') {{
            try {{ oldController.dispose(); }} catch (_) {{}}
          }}

          const held=new Set();
          const lastFire=new Map();
          let serial=0;
          const minimumGapMs=220;
          const keyOptions={{capture: true, passive: false}};

          function commandTarget() {{
            return Array.from(document.querySelectorAll('textarea'))
              .find(node=> node.placeholder === placeholder) || null;
          }}

          function keyboardSink() {{
            return document.getElementById(keyboardSinkId) || null;
          }}

          function focusKeyboardSink() {{
            const sink=keyboardSink();
            if (!sink) return;
            try {{
              sink.focus({{preventScroll: true}});
            }} catch (_) {{
              try {{ sink.focus(); }} catch (_) {{}}
            }}
            sink.value='';
          }}

          function isEditorSink(target) {{
            return Boolean(target && String(target.id || '') === keyboardSinkId);
          }}

          function isTypingTarget(target) {{
            if (!target || isEditorSink(target)) return false;
            const tag=String(target.tagName || '').toLowerCase();
            return tag === 'input' || tag === 'textarea' || tag === 'select' ||
                   target.isContentEditable === true;
          }}

          function normalizeCode(event) {{
            const code=String(event.code || '');
            if (code) return code;
            const key=String(event.key || '').toLowerCase();
            const fallback={{
              p: 'KeyP', n: 'KeyN', s: 'KeyS', i: 'KeyI', o: 'KeyO',
              // R no longer resets the mask; it is retained only to block the legacy
              // Kaggle shortcut when pressed out of habit.
              r: 'KeyR',
              a: 'KeyA', h: 'KeyH', u: 'KeyU', k: 'KeyK', c: 'KeyC', d: 'KeyD',
              '[': 'BracketLeft', ']': 'BracketRight'
            }};
            return fallback[key] || '';
          }}

          const actions={{
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
          const legacyBlockedCodes=new Set(['KeyR']);

          function blockEvent(event) {{
            if (!event) return;
            if (typeof event.preventDefault === 'function') event.preventDefault();
            if (typeof event.stopPropagation === 'function') event.stopPropagation();
            if (typeof event.stopImmediatePropagation === 'function') {{
              event.stopImmediatePropagation();
            }}
          }}

          function shortcutInfo(event) {{
            const code=normalizeCode(event);
            const action=actions[code] || '';
            const legacyBlocked=legacyBlockedCodes.has(code);
            if ((!action && !legacyBlocked) || event.ctrlKey || event.metaKey ||
                event.altKey || isTypingTarget(event.target)) return null;
            return {{code, action, legacyBlocked}};
          }}

          function onKeyDown(event) {{
            const info=shortcutInfo(event);
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

            const now=performance.now();
            const previous=lastFire.get(info.code) || -Infinity;
            if (now - previous < minimumGapMs) return;

            const target=commandTarget();
            if (!target) return;
            held.add(info.code);
            lastFire.set(info.code, now);
            serial +=1;
            target.value=`${{info.action}}|${{Date.now()}}|${{serial}}`;
            target.dispatchEvent(new Event('input', {{bubbles: true}}));
            target.dispatchEvent(new Event('change', {{bubbles: true}}));
          }}

          function onKeyPress(event) {{
            const info=shortcutInfo(event);
            if (info) blockEvent(event);
          }}

          function onKeyUp(event) {{
            const info=shortcutInfo(event);
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

          window[controllerKey]={{
            editorId,
            dispose: ()=> {{
              window.removeEventListener('keydown', onKeyDown, true);
              window.removeEventListener('keypress', onKeyPress, true);
              window.removeEventListener('keyup', onKeyUp, true);
              window.removeEventListener('blur', releaseAll, true);
              document.removeEventListener('visibilitychange', releaseAll, true);
              const sink=keyboardSink();
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

    def _shortcut_sync_changed(self, change):
        payload=str(change.get("new", "") or "").strip()
        if not self.rows:
            return
        if not payload or payload == self._last_shortcut_payload:
            return
        self._last_shortcut_payload=payload
        action=payload.split("|", 1)[0].strip().lower()

        now=time.monotonic()
        if self._shortcut_busy or now - self._last_shortcut_at < 0.12:
            return

        actions={
            "previous": ("Previous", self.previous),
            "next": ("Next", self.next),
            "save_next": ("Save & Next", self.save_next),
            "reset_image": ("Reset to image", self.reset_to_image),
            "reset_auto": ("Reset to auto", self.reset_to_auto),
            "accept_auto": ("Auto OK", self.accept_auto),
            "no_heart": ("No heart", self.mark_no_heart),
            "unusable": ("Unusable", self.mark_unusable),
            "skip": ("Skip", self.next),
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
        selected=actions.get(action)
        if selected is None:
            return

        self._shortcut_busy=True
        self._last_shortcut_at=now
        try:
            label, function=selected
            self._safe(label, function)
        finally:
            self._shortcut_busy=False

    def _mask_sync_changed(self, change):
        value=change.get("new", "")
        if not value:
            return
        try:
            decoded=cv2.imdecode(
                np.frombuffer(base64.b64decode(value), np.uint8), cv2.IMREAD_UNCHANGED
            )
            if decoded is None:
                return
            if decoded.ndim == 3 and decoded.shape[2] == 4:
                alpha=decoded[..., 3]
                gray=cv2.cvtColor(decoded[..., :3], cv2.COLOR_BGR2GRAY)
                binary=((alpha > 8) & (gray > 127)).astype(np.uint8)
            elif decoded.ndim == 3:
                binary=(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8)
            else:
                binary=(decoded > 127).astype(np.uint8)
            self.mask=cv2.resize(
                binary,
                (256, 256),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        except Exception as error:
            print("[EDITOR WARNING] Mask synchronization failed:", error)

    def _sync_python_mask(self):
        value=self._mask_uri(self.mask).split(",", 1)[1]
        if self.mask_sync.value != value:
            self.mask_sync.value=value

    def update_status(self, prefix=""):
        if not self.rows:
            self.status.value=(
                "<span style='margin-left:12px'><b>Review complete.</b> "
                "No target=UNLABELED images remain.</span>"
            )
            return
        row=self.rows[self.index]
        manual=DataStage.manual_mask_qc(row["manual_mask_path"])
        annotation=DataStage.load_annotations(self.workspace).get(
            str(row["image_token"]), {}
        )
        target_type=(
            ReviewStage.normalize_target_type(annotation.get("target_type", ""))
            or "UNLABELED"
        )
        self.status.value=(
            "<span style='margin-left:12px'>"
            f"<b>{prefix}</b> index={self.index + 1}/{len(self.rows)}; "
            f"target={target_type}; manual={'yes' if manual['exists'] else 'no'}; "
            f"usable={manual['usable']}; attention_valid={row.get('attention_valid_final', '')}; "
            f"presence={DataStage._as_float(row.get('attention_presence_probability'), np.nan):.3f}; "
            f"uncertainty={DataStage._as_float(row.get('attention_uncertainty_score'), np.nan):.3f}; "
            f"reason={row.get('attention_invalid_reason_final', '')}</span>"
        )

    def _write_target(
        self,
        mask,
        target_type,
        source,
        sample_weight=None,
    ):
        row=self.rows[self.index]
        binary=(np.asarray(mask) > 0).astype(np.uint8)
        path=Path(row["manual_mask_path"])
        DataStage.write_png(path, binary * 255)
        DataStage.set_annotation(
            self.workspace,
            row,
            target_type=target_type,
            source=source,
            sample_weight=sample_weight,
        )
        row["manual_annotation_type"]=target_type
        row["manual_annotation_source"]=source

        # The editor displays overlays live; no overlay/history file is stored.
        self.mask=binary.copy()

    def save(self):
        if float((self.mask > 0).mean()) < 0.0005:
            raise ValueError(
                "The mask is empty. Use No heart [H] for a valid image "
                "without a visible heart, or Unusable [U] for blur/noise/localizer frames."
            )
        self._write_target(
            self.mask,
            HEART_PRESENT,
            source="human_drawn_or_corrected",
            sample_weight=1.0,
        )
        self.update_status("Saved HEART_PRESENT; weight=1.00.")
        print("[EDITOR]", self.rows[self.index]["manual_mask_path"], "| HEART_PRESENT")

    def save_next(self):
        self.save()
        self._advance_after_target("Saved HEART_PRESENT.")

    def accept_auto(self):
        if float((self.auto_mask > 0).mean()) < 0.0005:
            raise ValueError(
                "The automatic mask is empty; use No heart [H], not Auto OK."
            )
        self._write_target(
            self.auto_mask,
            HEART_PRESENT,
            source="human_confirmed_auto",
            sample_weight=0.72,
        )
        self._advance_after_target(
            f"Auto saved HEART_PRESENT; weight={0.72:.2f}."
        )

    def mark_no_heart(self):
        empty=np.zeros_like(self.auto_mask, dtype=np.uint8)
        self._write_target(
            empty,
            NO_HEART_VISIBLE,
            source="human_no_heart_visible",
            sample_weight=0.90,
        )
        self._advance_after_target("Saved NO_HEART_VISIBLE negative target.")

    def mark_unusable(self):
        row=self.rows[self.index]
        Path(row["manual_mask_path"]).unlink(missing_ok=True)
        DataStage.set_annotation(
            self.workspace,
            row,
            target_type=UNUSABLE,
            source="human_unusable",
            sample_weight=0.0,
        )
        row["manual_annotation_type"]=UNUSABLE
        self._advance_after_target("Marked UNUSABLE; excluded from training.")

    def reset_to_auto(self):
        self.base_mask=self.auto_mask.copy()
        self.mask=self.auto_mask.copy()
        self._sync_python_mask()
        self.render()
        self.update_status("Automatic mask restored.")

    def reset_to_image(self):
        self.base_mask=np.zeros_like(self.auto_mask)
        self.mask=np.zeros_like(self.auto_mask)
        self._sync_python_mask()
        self.render()
        self.update_status("All overlays hidden; not saved yet.")

    def clear_editable(self):
        self.base_mask=self.auto_mask.copy()
        self.mask=np.zeros_like(self.auto_mask)
        self._sync_python_mask()
        self.render()
        self.update_status("Editable mask cleared; auto remains as guide.")

    def delete_manual(self):
        row=self.rows[self.index]
        Path(row["manual_mask_path"]).unlink(missing_ok=True)
        DataStage.remove_annotation(self.workspace, str(row["image_token"]))
        row["manual_annotation_type"]=""
        row["manual_annotation_source"]=""
        self.mask=self.auto_mask.copy()
        self.base_mask=self.auto_mask.copy()
        self._sync_python_mask()
        self.render()
        self.update_status("Manual target and annotation deleted.")

    def next(self):
        if not self.rows:
            return
        if self.index < len(self.rows) - 1:
            self.index +=1
            self.load_current()
        else:
            self.update_status("End of queue.")

    def previous(self):
        if not self.rows:
            return
        if self.index > 0:
            self.index -=1
            self.load_current()

    def show(self):
        display(self.controls)
        display(Javascript(self._shortcut_script()))
        return self

# =============================================================================
# STAGE 4 — SICK/NORMAL MATCHING
# =============================================================================
class MatchingStage:
    """Rebuild strict-core and expanded balanced matching from current-run data."""

    # Convert slice position to a normalized within-series coordinate for matching.
    @staticmethod
    def sequence_position(row):
        length=max(1, DataStage._as_int(row.get("sequence_length"), 1))
        index=int(np.clip(DataStage._as_int(row.get("sequence_index"), 0), 0, length - 1))
        if length <= 1:
            return 0.5
        return float(index / (length - 1))
    # Convert the perceptual hash string to bits used by the matching descriptor.
    @staticmethod
    def phash_bits(value):
        number=int(str(value), 16)
        return np.asarray(
            [(number >> shift) & 1 for shift in range(63, -1, -1)],
            dtype=np.float32,
        )
    # Median/MAD standardize matching variables so one descriptor block cannot dominate.
    @staticmethod
    def robust_standardize(values):
        values=np.asarray(values, dtype=np.float32).copy()
        if values.ndim == 1:
            values=values[:, None]
        for column in range(values.shape[1]):
            current=values[:, column]
            finite=np.isfinite(current)
            center=float(np.median(current[finite])) if finite.any() else 0.0
            current[~finite]=center
            mad=float(np.median(np.abs(current - center)))
            scale=max(1e-3, 1.4826 * mad)
            values[:, column]=np.clip((current - center) / scale, -5.0, 5.0)
        return values
    # Build acquisition/anatomy/quality descriptors without using the CAD label as a feature.
    @staticmethod
    def matching_descriptor(rows):
        phash=np.stack(
            [MatchingStage.phash_bits(row["perceptual_hash"]) for row in rows]
        )
        phash=phash * 2.0 - 1.0
        geometry=MatchingStage.robust_standardize(
            np.asarray(
                [
                    [
                        DataStage._as_float(row.get("attention_area_ratio"), np.nan),
                        DataStage._as_float(row.get("attention_centroid_x"), np.nan),
                        DataStage._as_float(row.get("attention_centroid_y"), np.nan),
                        DataStage._as_float(row.get("attention_boundary_touch_fraction"), np.nan),
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            )
        )
        sequence=MatchingStage.robust_standardize(
            np.asarray(
                [[MatchingStage.sequence_position(row)] for row in rows],
                dtype=np.float32,
            )
        )
        quality=MatchingStage.robust_standardize(
            np.asarray(
                [
                    [
                        math.log1p(max(0.0, DataStage._as_float(row.get("sharpness"), 0.0))),
                        DataStage._as_float(row.get("noise_ratio"), np.nan),
                        DataStage._as_float(row.get("dynamic_range"), np.nan),
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            )
        )

        def block_scale(weight, dimensions):
            return math.sqrt(max(float(weight), 0.0) / max(1, int(dimensions)))

        descriptor=np.concatenate(
            [
                phash * block_scale(0.45, phash.shape[1]),
                geometry
                * block_scale(0.25, geometry.shape[1]),
                sequence
                * block_scale(0.15, sequence.shape[1]),
                quality
                * block_scale(0.15, quality.shape[1]),
            ],
            axis=1,
        )
        return np.ascontiguousarray(descriptor, dtype=np.float32)
    # Require the same quality/geometry gates and a current in-memory automatic mask.
    @staticmethod
    def matching_eligibility_reason(row):
        reasons=[]
        if DataStage._as_int(row.get("quality_valid"), 0) != 1:
            reasons.append("quality_invalid")
        if DataStage._as_int(row.get("attention_valid_final"), 0) != 1:
            reasons.append("attention_invalid")
        if DataStage._as_int(row.get("attention_heart_present"), 1) != 1:
            reasons.append("heart_not_visible")
        if (
            DataStage._as_float(row.get("attention_area_ratio"), 0.0)
            < 0.003):
            reasons.append("attention_area_too_small")
        if not str(row.get("perceptual_hash", "")).strip():
            reasons.append("missing_phash")
        if row.get("attention_mask_bits") is None:
            reasons.append("missing_attention_mask")
        return ";".join(reasons)
    # Choose acquisition-family granularity from cohort size while respecting class coverage.
    @staticmethod
    def choose_family_count(number_of_images):
        estimate=int(
            math.ceil(
                number_of_images
                / max(1, 1800)
            )
        )
        count=int(
            np.clip(
                estimate,
                8,
                32,
            )
        )
        return max(2, min(count, number_of_images))
    # Require each acquisition family to contain enough Sick and Normal patients/slices.
    @staticmethod
    def family_is_shared(rows):
        for label in (0, 1):
            class_rows=[row for row in rows if DataStage._as_int(row.get("label"), -1) == label]
            if len(class_rows) < 12:
                return False
            if (
                len({str(row.get("patient_id", "")) for row in class_rows})
                < 2):
                return False
        return True
    # Derive a distance cutoff from robust neighbour-distance statistics.
    @staticmethod
    def robust_caliper(
        distances,
        mad_multiplier,
        quantile,
    ):
        values=np.asarray(list(distances), dtype=np.float64)
        values=values[np.isfinite(values)]
        if not len(values):
            return np.nan
        median=float(np.median(values))
        mad=float(np.median(np.abs(values - median)))
        robust_scale=max(1e-9, 1.4826 * mad)
        mad_caliper=median + float(mad_multiplier) * robust_scale
        quantile_caliper=float(np.quantile(values, float(quantile)))
        return float(max(median, min(mad_caliper, quantile_caliper)))
    # Both core and extended matching use the same anatomical limits.
    @staticmethod
    def passes_anatomical_gates(
        sick_row,
        normal_row,
    ):
        sequence_limit=0.25
        area_limit=0.20
        if (
            abs(
                MatchingStage.sequence_position(sick_row)
                - MatchingStage.sequence_position(normal_row)
            )
            > sequence_limit):
            return False
        sick_area=DataStage._as_float(sick_row.get("attention_area_ratio"), np.nan)
        normal_area=DataStage._as_float(normal_row.get("attention_area_ratio"), np.nan)
        if (
            np.isfinite(sick_area)
            and np.isfinite(normal_area)
            and abs(sick_area - normal_area) > area_limit):
            return False
        return True
    # Return stable patient/series identifiers used by matching capacity limits.
    @staticmethod
    def pair_keys(
        sick_row, normal_row, family
    ):
        sick_patient=str(sick_row["patient_id"])
        normal_patient=str(normal_row["patient_id"])
        sick_sequence=str(
            sick_row.get("sequence_group_id") or sick_row.get("series_id", "")
        )
        normal_sequence=str(
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
    # Rebuild the strict core and expanded balanced cohort from current RAM records.
    @staticmethod
    def build_cross_class_matching(dataset_rows):
        # Matching is intentionally rebuilt every run. It uses only the current OOF
        # masks, quality measurements, labels, and dataset geometry.
        manifest_by_token={}
        eligible_rows=[]
        for row in dataset_rows:
            token=str(row["image_token"])
            reason=MatchingStage.matching_eligibility_reason(row)
            record={
                "image_token": token,
                "image_path": str(row.get("image_path", "")),
                "patient_id": str(row.get("patient_id", "")),
                "series_id": str(row.get("series_id", "")),
                "sequence_group_id": str(
                    row.get("sequence_group_id") or row.get("series_id", "")
                ),
                "label": DataStage._as_int(row.get("label"), -1),
                "class_name": "Sick" if DataStage._as_int(row.get("label"), -1) == 1 else "Normal",
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
                "sequence_position": MatchingStage.sequence_position(row),
                "attention_area_ratio": DataStage._as_float(
                    row.get("attention_area_ratio"), np.nan
                ),
                "exclusion_reason": reason,
            }
            manifest_by_token[token]=record
            if not reason:
                eligible_rows.append(row)

        labels=np.asarray(
            [DataStage._as_int(row.get("label"), -1) for row in eligible_rows], dtype=np.int64
        )
        if len(eligible_rows) < 4 or set(labels.tolist()) != {0, 1}:
            raise RuntimeError(
                "Matching requires eligible images from both classes; "
                f"remaining Normal={int(np.sum(labels == 0))}, Sick={int(np.sum(labels == 1))}."
            )

        descriptor=MatchingStage.matching_descriptor(eligible_rows)
        family_count=MatchingStage.choose_family_count(len(eligible_rows))
        batch_size=min(4096, max(256, len(eligible_rows) // 10))
        clusterer=MiniBatchKMeans(
            n_clusters=family_count,
            random_state=42,
            batch_size=batch_size,
            n_init=5,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        family_labels=clusterer.fit_predict(descriptor).astype(np.int64)

        family_indices=defaultdict(list)
        for index, family in enumerate(family_labels.tolist()):
            family_indices[int(family)].append(index)
            token=str(eligible_rows[index]["image_token"])
            manifest_by_token[token]["acquisition_family"]=int(family)

        shared_families=set()
        for family, indices in family_indices.items():
            family_rows=[eligible_rows[index] for index in indices]
            if MatchingStage.family_is_shared(family_rows):
                shared_families.add(int(family))
                for index in indices:
                    manifest_by_token[str(eligible_rows[index]["image_token"])][
                        "shared_family"
                    ]=1
            else:
                for index in indices:
                    manifest_by_token[str(eligible_rows[index]["image_token"])][
                        "exclusion_reason"
                    ]="family_not_shared_between_classes"

        if not shared_families:
            raise RuntimeError(
                "No acquisition family contains enough images and patients from both classes."
            )

        normal_count=int(np.sum(labels == 0))
        sick_count=int(np.sum(labels == 1))
        maximum_possible_pairs=min(normal_count, sick_count)
        requested_target_pairs=int(
            round(
                len(eligible_rows)
                * float(0.50)
                / 2.0
            )
        )
        requested_target_pairs=max(1, min(requested_target_pairs, maximum_possible_pairs))

        patients_by_label={
            label: sorted(
                {
                    str(row["patient_id"])
                    for row in eligible_rows
                    if DataStage._as_int(row.get("label"), -1) == label
                }
            )
            for label in (0, 1)
        }
        patient_total_caps={
            label: max(
                60,
                int(
                    math.ceil(
                        requested_target_pairs
                        / max(1, len(patients_by_label[label]))
                        * 1.70
                    )
                ),
            )
            for label in (0, 1)
        }

        used_tokens=set()
        patient_family_counts=defaultdict(int)
        sequence_counts=defaultdict(int)
        patient_total_counts=defaultdict(int)
        patient_pair_counts=defaultdict(int)
        sequence_pair_counts=defaultdict(int)
        core_sequence_pairs=set()
        selected_pair_distances=[]
        core_pair_distances=[]
        extended_pair_distances=[]
        family_summaries=[]
        pair_number=0
        core_pair_count=0
        extended_pair_count=0

        family_data={}
        token_candidate_stage={}

        # Build candidate graphs first. The extended graph is the union of the
        # top-K neighbours from both directions; the strict graph is its mutual
        # top-5 subset.
        for family in sorted(shared_families):
            indices=family_indices[family]
            normal_positions=[
                index for index in indices if DataStage._as_int(eligible_rows[index].get("label"), -1) == 0
            ]
            sick_positions=[
                index for index in indices if DataStage._as_int(eligible_rows[index].get("label"), -1) == 1
            ]
            normal_descriptor=descriptor[normal_positions]
            sick_descriptor=descriptor[sick_positions]
            extended_sick_to_normal=min(
                int(12), len(normal_positions)
            )
            extended_normal_to_sick=min(
                int(12), len(sick_positions)
            )

            normal_model=NearestNeighbors(
                n_neighbors=extended_sick_to_normal,
                metric="euclidean",
                algorithm="auto",
                n_jobs=-1,
            ).fit(normal_descriptor)
            sick_to_normal_index=normal_model.kneighbors(
                sick_descriptor, return_distance=False
            )
            sick_model=NearestNeighbors(
                n_neighbors=extended_normal_to_sick,
                metric="euclidean",
                algorithm="auto",
                n_jobs=-1,
            ).fit(sick_descriptor)
            normal_to_sick_index=sick_model.kneighbors(
                normal_descriptor, return_distance=False
            )

            sick_rank_maps=[
                {int(normal_local): int(rank + 1) for rank, normal_local in enumerate(neighbours)}
                for neighbours in sick_to_normal_index
            ]
            normal_rank_maps=[
                {int(sick_local): int(rank + 1) for rank, sick_local in enumerate(neighbours)}
                for neighbours in normal_to_sick_index
            ]
            edge_keys=set()
            for sick_local, neighbours in enumerate(sick_to_normal_index):
                edge_keys.update((int(sick_local), int(value)) for value in neighbours)
            for normal_local, neighbours in enumerate(normal_to_sick_index):
                edge_keys.update((int(value), int(normal_local)) for value in neighbours)

            all_edges=[]
            core_edges=[]
            for sick_local, normal_local in edge_keys:
                sick_global=sick_positions[sick_local]
                normal_global=normal_positions[normal_local]
                sick_row=eligible_rows[sick_global]
                normal_row=eligible_rows[normal_global]
                if not MatchingStage.passes_anatomical_gates(
                    sick_row, normal_row):
                    continue
                rank_from_sick=sick_rank_maps[sick_local].get(normal_local)
                rank_from_normal=normal_rank_maps[normal_local].get(sick_local)
                distance=float(
                    np.linalg.norm(
                        descriptor[sick_global] - descriptor[normal_global]
                    )
                )
                reciprocal=rank_from_sick is not None and rank_from_normal is not None
                is_core=bool(
                    reciprocal
                    and rank_from_sick <= 5
                    and rank_from_normal <= 5
                )
                sick_token=str(sick_row["image_token"])
                normal_token=str(normal_row["image_token"])
                token_candidate_stage.setdefault(sick_token, "extended_candidate")
                token_candidate_stage.setdefault(normal_token, "extended_candidate")
                if is_core:
                    token_candidate_stage[sick_token]="core_candidate"
                    token_candidate_stage[normal_token]="core_candidate"
                edge={
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

            core_caliper=MatchingStage.robust_caliper(
                [edge["distance"] for edge in core_edges],
                2.5,
                0.90,
            )
            extended_caliper=MatchingStage.robust_caliper(
                [edge["distance"] for edge in all_edges],
                3.5,
                0.97,
            )
            if np.isfinite(core_caliper):
                maximum_extended=(
                    float(core_caliper)
                    * float(1.18)
                )
                if np.isfinite(extended_caliper):
                    extended_caliper=max(
                        float(core_caliper),
                        min(float(extended_caliper), maximum_extended),
                    )
                else:
                    extended_caliper=maximum_extended

            family_data[family]={
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
                record=manifest_by_token[str(eligible_rows[index]["image_token"])]
                if np.isfinite(core_caliper):
                    record["family_core_caliper"]=float(core_caliper)
                if np.isfinite(extended_caliper):
                    record["family_extended_caliper"]=float(extended_caliper)

        def can_select(
            edge,
            stage,
            patient_quota=None,
        ):
            family=int(edge["family"])
            sick_row=eligible_rows[int(edge["sick_global"])]
            normal_row=eligible_rows[int(edge["normal_global"])]
            sick_token=str(sick_row["image_token"])
            normal_token=str(normal_row["image_token"])
            if sick_token in used_tokens or normal_token in used_tokens:
                return False
            (
                sick_patient_key,
                normal_patient_key,
                sick_sequence_key,
                normal_sequence_key,
                patient_pair_key,
                sequence_pair_key,
            )=MatchingStage.pair_keys(sick_row, normal_row, family)
            if stage == "core_mutual":
                patient_family_limit=20
                sequence_limit=5
            else:
                patient_family_limit=60
                sequence_limit=15
            if (
                patient_family_counts[sick_patient_key] >= patient_family_limit
                or patient_family_counts[normal_patient_key] >= patient_family_limit
                or sequence_counts[sick_sequence_key] >= sequence_limit
                or sequence_counts[normal_sequence_key] >= sequence_limit):
                return False
            if stage != "core_mutual":
                if (
                    patient_pair_counts[patient_pair_key]
                    >= 60
                    or sequence_pair_counts[sequence_pair_key]
                    >= 12):
                    return False
                for row in (sick_row, normal_row):
                    label=DataStage._as_int(row.get("label"), -1)
                    patient=str(row["patient_id"])
                    cap=int(patient_total_caps[label])
                    if patient_total_counts[patient] >= cap:
                        return False
                    if patient_quota is not None and patient_total_counts[patient] >= int(
                        patient_quota[label]):
                        return False
            return True

        def select_edge(edge, stage):
            nonlocal pair_number, core_pair_count, extended_pair_count
            family=int(edge["family"])
            sick_row=eligible_rows[int(edge["sick_global"])]
            normal_row=eligible_rows[int(edge["normal_global"])]
            sick_token=str(sick_row["image_token"])
            normal_token=str(normal_row["image_token"])
            (
                sick_patient_key,
                normal_patient_key,
                sick_sequence_key,
                normal_sequence_key,
                patient_pair_key,
                sequence_pair_key,
            )=MatchingStage.pair_keys(sick_row, normal_row, family)
            seeded=int(sequence_pair_key in core_sequence_pairs)
            pair_number +=1
            prefix="CORE" if stage == "core_mutual" else "EXT"
            pair_id=f"CCM_{prefix}_F{family:02d}_P{pair_number:06d}"
            applied_caliper=(
                family_data[family]["core_caliper"]
                if stage == "core_mutual"
                else family_data[family]["extended_caliper"]
            )
            for source_row, partner_row in (
                (sick_row, normal_row),
                (normal_row, sick_row),):
                source_token=str(source_row["image_token"])
                record=manifest_by_token[source_token]
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
            patient_family_counts[sick_patient_key] +=1
            patient_family_counts[normal_patient_key] +=1
            sequence_counts[sick_sequence_key] +=1
            sequence_counts[normal_sequence_key] +=1
            patient_total_counts[str(sick_row["patient_id"])] +=1
            patient_total_counts[str(normal_row["patient_id"])] +=1
            patient_pair_counts[patient_pair_key] +=1
            sequence_pair_counts[sequence_pair_key] +=1
            selected_pair_distances.append(float(edge["distance"]))
            if stage == "core_mutual":
                core_sequence_pairs.add(sequence_pair_key)
                core_pair_distances.append(float(edge["distance"]))
                core_pair_count +=1
                family_data[family]["core_selected"] +=1
            else:
                extended_pair_distances.append(float(edge["distance"]))
                extended_pair_count +=1
                family_data[family]["extended_selected"] +=1

        # Stage 1: preserve the previous strict cohort.
        for family in sorted(shared_families):
            caliper=family_data[family]["core_caliper"]
            if not np.isfinite(caliper):
                continue
            for edge in sorted(
                family_data[family]["core_edges"],
                key=lambda item: (item["distance"], item["tie"]),):
                if float(edge["distance"]) > float(caliper):
                    continue
                if can_select(edge, "core_mutual"):
                    select_edge(edge, "core_mutual")

        # Allocate the remaining target proportionally to the unused common
        # support of each family. This prevents a few large families from taking
        # the whole extension.
        remaining_target=max(0, requested_target_pairs - core_pair_count)
        remaining_capacity={
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
        total_remaining_capacity=sum(remaining_capacity.values())
        extension_targets={family: 0 for family in shared_families}
        if remaining_target > 0 and total_remaining_capacity > 0:
            raw_targets={
                family: remaining_target
                * remaining_capacity[family]
                / total_remaining_capacity
                for family in shared_families
            }
            extension_targets={
                family: min(
                    remaining_capacity[family], int(math.floor(raw_targets[family]))
                )
                for family in shared_families
            }
            unassigned=remaining_target - sum(extension_targets.values())
            for family in sorted(
                shared_families,
                key=lambda value: (
                    raw_targets[value] - math.floor(raw_targets[value]),
                    remaining_capacity[value],
                ),
                reverse=True,):
                if unassigned <= 0:
                    break
                if extension_targets[family] < remaining_capacity[family]:
                    extension_targets[family] +=1
                    unassigned -=1

        # Stage 2: add close one-sided or reciprocal top-K neighbours. Core-seeded
        # sequence pairs are considered first, then reciprocal candidates, then
        # distance/rank. Quota passes stop a single patient from monopolising a family.
        for family in sorted(shared_families):
            family_target=int(extension_targets.get(family, 0))
            if family_target <= 0:
                continue
            extended_caliper=family_data[family]["extended_caliper"]
            if not np.isfinite(extended_caliper):
                continue
            candidates=[]
            for edge in family_data[family]["all_edges"]:
                if float(edge["distance"]) > float(extended_caliper):
                    continue
                sick_row=eligible_rows[int(edge["sick_global"])]
                normal_row=eligible_rows[int(edge["normal_global"])]
                sequence_pair=MatchingStage.pair_keys(
                    sick_row, normal_row, family
                )[-1]
                seeded=int(sequence_pair in core_sequence_pairs)
                rank_sick=edge.get("rank_from_sick")
                rank_normal=edge.get("rank_from_normal")
                rank_sum=int(rank_sick or 12 + 1) + int(
                    rank_normal or 12 + 1
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
                        True
                        and item["seeded"]
                    )
                    else 1,
                    0 if item.get("reciprocal", 0) else 1,
                    item["normalized_distance"],
                    item["rank_sum"],
                    item["tie"],
                )
            )

            selected_here=0
            max_cap=max(patient_total_caps.values())
            starting_quota=max(
                1,
                min(
                    [patient_total_counts.get(patient, 0) for patient in patient_total_counts]
                    or [1]
                ),
            )
            quota_values=np.unique(
                np.ceil(
                    np.linspace(
                        starting_quota,
                        max_cap,
                        max(2, int(12)),
                    )
                ).astype(int)
            )
            for quota in quota_values:
                if selected_here >= family_target or pair_number >= requested_target_pairs:
                    break
                patient_quota={0: int(quota), 1: int(quota)}
                for edge in candidates:
                    if selected_here >= family_target or pair_number >= requested_target_pairs:
                        break
                    if can_select(edge, "extended_neighbor", patient_quota=patient_quota):
                        select_edge(edge, "extended_neighbor")
                        selected_here +=1

        # Spillover pass: if a family could not fill its proportional allocation,
        # other families with unused high-quality edges may contribute the remainder.
        # All one-to-one and capacity constraints remain active.
        if pair_number < requested_target_pairs:
            spillover_added=0
            for family in sorted(shared_families):
                if pair_number >= requested_target_pairs:
                    break
                extended_caliper=family_data[family]["extended_caliper"]
                if not np.isfinite(extended_caliper):
                    continue
                spillover_candidates=[]
                for edge in family_data[family]["all_edges"]:
                    if float(edge["distance"]) > float(extended_caliper):
                        continue
                    sick_row=eligible_rows[int(edge["sick_global"])]
                    normal_row=eligible_rows[int(edge["normal_global"])]
                    sequence_pair=MatchingStage.pair_keys(
                        sick_row, normal_row, family
                    )[-1]
                    rank_sick=edge.get("rank_from_sick")
                    rank_normal=edge.get("rank_from_normal")
                    spillover_candidates.append(
                        {
                            **edge,
                            "seeded": int(sequence_pair in core_sequence_pairs),
                            "rank_sum": int(
                                rank_sick or 12 + 1
                            )
                            + int(
                                rank_normal or 12 + 1
                            ),
                            "normalized_distance": float(edge["distance"])
                            / max(float(extended_caliper), 1e-9),
                        }
                    )
                spillover_candidates.sort(
                    key=lambda item: (
                        0
                        if (
                            True
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
                        spillover_added +=1
            if spillover_added:
                print(
                    "[MATCHING] Spillover redistributed "
                    f"{spillover_added} extended pairs across families."
                )

        # Explain every non-selected image in the manifest.
        for family in sorted(shared_families):
            data=family_data[family]
            candidate_tokens=set()
            below_extended_tokens=set()
            for edge in data["all_edges"]:
                sick_token=str(eligible_rows[int(edge["sick_global"])]["image_token"])
                normal_token=str(eligible_rows[int(edge["normal_global"])]["image_token"])
                candidate_tokens.update((sick_token, normal_token))
                if np.isfinite(data["extended_caliper"]) and float(edge["distance"]) <= float(
                    data["extended_caliper"]):
                    below_extended_tokens.update((sick_token, normal_token))
            for index in family_indices[family]:
                token=str(eligible_rows[index]["image_token"])
                record=manifest_by_token[token]
                if DataStage._as_int(record.get("selected_for_matched_cohort"), 0) == 1:
                    continue
                if token not in candidate_tokens:
                    record["exclusion_reason"]="no_cross_class_neighbor_in_extended_top_k"
                elif token not in below_extended_tokens:
                    record["exclusion_reason"]="above_extended_family_caliper"
                elif pair_number >= requested_target_pairs:
                    record["exclusion_reason"]="expanded_target_reached"
                else:
                    record["exclusion_reason"]="one_to_one_or_capacity_limit"

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

        manifest=[
            manifest_by_token[str(row["image_token"])] for row in dataset_rows
        ]
        selected=[
            row
            for row in manifest
            if DataStage._as_int(row.get("selected_for_matched_cohort"), 0) == 1
        ]
        core_selected=[
            row
            for row in manifest
            if DataStage._as_int(row.get("selected_for_core_matched_cohort"), 0) == 1
        ]
        selected_normal=[row for row in selected if DataStage._as_int(row.get("label"), -1) == 0]
        selected_sick=[row for row in selected if DataStage._as_int(row.get("label"), -1) == 1]
        normal_patients=sorted({str(row["patient_id"]) for row in selected_normal})
        sick_patients=sorted({str(row["patient_id"]) for row in selected_sick})
        if len(selected_normal) != len(selected_sick):
            raise RuntimeError("Internal matching produced unequal class sizes.")
        if len(normal_patients) < 2 or len(sick_patients) < 2:
            raise RuntimeError(
                "The matched cohort has too few patients for evaluation: "
                f"Normal={len(normal_patients)}, Sick={len(sick_patients)}."
            )

        per_patient=defaultdict(int)
        for row in selected:
            per_patient[str(row["patient_id"])] +=1
        selected_fraction_eligible=len(selected) / max(1, len(eligible_rows))
        selected_fraction_dataset=len(selected) / max(1, len(dataset_rows))
        summary={
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
            "interpretation": (
                "B1/AU2/C2 use the expanded balanced cohort. The previous strict "
                "mutual-neighbour cohort is retained in selected_for_core_matched_cohort "
                "for audit and sensitivity checks."
            ),
        }
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
            f"patients Normal={len(normal_patients)}, Sick={len(sick_patients)}"
        )
        return manifest, summary

# =============================================================================
# STAGE 5 — FROZEN EFFICIENTNET FEATURES
# =============================================================================
# PyTorch Dataset/model/pool classes hold actual runtime state, so classes are natural here.
class FeatureDataset(Dataset):
    """Load full-image, automatic-mask, and manual-mask tensors for feature extraction."""

    def __init__(self, rows):
        self.rows=list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row=self.rows[index]
        robust, raw, content=DataStage.classifier_views(row["image_path"])
        attention=AttentionStage.read_prediction_mask(row, size=224).astype(np.float32)
        manual_path=Path(row["manual_mask_path"])
        if manual_path.is_file():
            manual=DataStage.read_binary_mask(
                manual_path, size=224
            ).astype(np.float32)
        else:
            manual=np.zeros_like(attention)

        robust_tensor=torch.from_numpy(np.stack([robust] * 3)).float()
        raw_tensor=torch.from_numpy(np.stack([raw] * 3)).float()
        return (
            robust_tensor,
            raw_tensor,
            torch.from_numpy(content).unsqueeze(0).float(),
            torch.from_numpy(attention).unsqueeze(0).float(),
            torch.from_numpy(manual).unsqueeze(0).float(),
            int(index),
        )
class FrozenEfficientNet(nn.Module):
    """Expose pretrained EfficientNet-B0 as a frozen embedding extractor."""

    def __init__(self):
        super().__init__()
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        try:
            model=efficientnet_b0(weights=weights)
        except Exception as error:
            raise RuntimeError(
                "EfficientNet-B0 weights could not be loaded. "
                "Enable Internet in Kaggle or place the weights in the Torch cache. "
                f"Original error: {type(error).__name__}: {error}"
            ) from error
        model.classifier=nn.Identity()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model=model.eval()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images):
        images=(images.float() - self.mean) / self.std
        return self.model(images)
class StreamingPatientPool:
    """Aggregate slice embeddings first by series and then by patient."""

    def __init__(self, modes):
        self.series_data={
            mode: {} for mode in modes
        }
        self.source_slices=defaultdict(int)

    def add(
        self,
        mode,
        embeddings,
        rows,
    ):
        embeddings=np.asarray(embeddings, dtype=np.float32)
        for embedding, row in zip(embeddings, rows):
            key=(str(row["patient_id"]), str(row["series_id"]))
            entry=self.series_data[mode].get(key)
            if entry is None:
                entry=[np.zeros_like(embedding, dtype=np.float32), 0, int(row["label"])]
                self.series_data[mode][key]=entry
            if int(entry[2]) != int(row["label"]):
                raise RuntimeError(f"Inconsistent labels for patient {row['patient_id']}.")
            entry[0] +=embedding
            entry[1] +=1
            self.source_slices[mode] +=1

    def finalize(self, mode):
        patient_series=defaultdict(list)
        patient_labels={}
        for (patient_id, _series_id), (embedding_sum, count, label) in self.series_data[mode].items():
            patient_series[patient_id].append(embedding_sum / max(1, count))
            patient_labels[patient_id]=int(label)
        patients=sorted(patient_series)
        if not patients:
            raise RuntimeError(
                f"Mode {mode} contains no patients. For M1/C3/AU3/C4, "
                "inspect [FEATURES][MANUAL] and the current manual-target audit."
            )
        X=np.stack(
            [np.mean(np.stack(patient_series[patient]), axis=0) for patient in patients]
        ).astype(np.float32)
        y=np.asarray([patient_labels[patient] for patient in patients], dtype=np.int64)
        return {
            "X": X,
            "y": y,
            "patient_ids": np.asarray(patients),
            "source_slices": int(self.source_slices[mode]),
            "series_proxies": int(len(self.series_data[mode])),
        }

class FeatureStage:
    """Build all eleven full-image/ROI/complement patient feature sets from scratch."""

    # These names define the research experiments; they are not tunable settings.
    MODES = (
        "B0_FULL_IMAGE",
        "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
        "AU1_ATTENTION_ROI",
        "C1_ATTENTION_COMPLEMENT",
        "B1_MATCHED_FULL_IMAGE",
        "AU2_MATCHED_ATTENTION_ROI",
        "C2_MATCHED_ATTENTION_COMPLEMENT",
        "M1_MANUAL_ROI",
        "C3_MANUAL_COMPLEMENT",
        "AU3_ATTENTION_ROI_MANUAL_SUBSET",
        "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET",
    )
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

    @staticmethod
    def region_normalize(images, masks):
        if masks.ndim == 3:
            masks=masks.unsqueeze(1)
        visible=masks > 0.5
        batch_size=images.shape[0]
        bins=int(256)
        mask_flat=visible[:, 0].reshape(batch_size, -1)
        counts=mask_flat.sum(dim=1).long()
        values=images[:, 0].float().clamp(0.0, 1.0)
        indices=torch.round(values * float(bins - 1)).long().clamp_(0, bins - 1)
        indices=indices.reshape(batch_size, -1)
        histogram=torch.zeros(batch_size, bins, device=images.device, dtype=torch.float32)
        histogram.scatter_add_(1, indices, mask_flat.float())
        cumulative=torch.cumsum(histogram, dim=1)
        safe_counts=counts.clamp_min(1)
        lower_rank=(
            torch.floor(
                1.0 / 100.0
                * (safe_counts - 1).float()
            ).long()
            + 1
        )
        upper_rank=(
            torch.floor(
                99.0 / 100.0
                * (safe_counts - 1).float()
            ).long()
            + 1
        )
        lower_bin=(cumulative >= lower_rank[:, None].float()).long().argmax(dim=1)
        upper_bin=(cumulative >= upper_rank[:, None].float()).long().argmax(dim=1)
        lower=lower_bin.float().view(-1, 1, 1, 1) / float(bins - 1)
        upper=upper_bin.float().view(-1, 1, 1, 1) / float(bins - 1)
        valid=(
            (counts >= 64)
            & (upper[:, 0, 0, 0] > lower[:, 0, 0, 0])
        )
        scaled=(images.float() - lower) / (upper - lower).clamp_min(
            8.0 / 255.0
        )
        scaled=scaled.clamp(0.0, 1.0) * visible.float()
        return scaled * valid.view(-1, 1, 1, 1).float()
    # Dilate the heart mask slightly before ROI/complement feature extraction.
    @staticmethod
    def support_mask(mask, content):
        kernel=15  # expand ROI slightly before extracting ROI/complement features
        support=F.max_pool2d(
            (mask > 0.5).float(), kernel_size=kernel, stride=1, padding=kernel // 2
        )
        return (support > 0.5).float() * (content > 0.5).float()
    @staticmethod
    def assert_same_cohort(
        bank,
        modes,
        cohort_name,
    ):

        modes=tuple(modes)
        reference_mode=modes[0]
        reference=bank[reference_mode]
        reference_patients=np.asarray(reference["patient_ids"]).astype(str)
        reference_labels=np.asarray(reference["y"], dtype=np.int64)
        reference_slices=int(reference["source_slices"])
        reference_series=int(reference["series_proxies"])

        for mode in modes[1:]:
            current=bank[mode]
            checks={
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
            failed=[name for name, passed in checks.items() if not passed]
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
    def validate_cohort_alignment(bank):
        FeatureStage.assert_same_cohort(
            bank,
            (
                "B2_ATTENTION_ELIGIBLE_FULL_IMAGE",
                "AU1_ATTENTION_ROI",
                "C1_ATTENTION_COMPLEMENT",
            ),
            "attention_eligible_same_slices",
        )
        FeatureStage.assert_same_cohort(
            bank,
            (
                "B1_MATCHED_FULL_IMAGE",
                "AU2_MATCHED_ATTENTION_ROI",
                "C2_MATCHED_ATTENTION_COMPLEMENT",
            ),
            "cross_class_matched_same_slices",
        )
        FeatureStage.assert_same_cohort(
            bank,
            (
                "M1_MANUAL_ROI",
                "C3_MANUAL_COMPLEMENT",
                "AU3_ATTENTION_ROI_MANUAL_SUBSET",
                "C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET",
            ),
            "manual_positive_same_slices",
        )
    # Compute all eleven modes every run; share identical embeddings across exact subsets.
    @staticmethod
    def build_feature_bank(dataset_rows, accepted_rows, matching_manifest, device):
        accepted_manual = {
            str(row["image_token"])
            for row in accepted_rows
            if row.get("segmentation_target_type") == HEART_PRESENT
            and DataStage.manual_mask_qc(row["manual_mask_path"])["usable"]
        }
        print(f"[FEATURES][MANUAL] accepted positive masks={len(accepted_manual)}")
        if not accepted_manual:
            raise RuntimeError("No usable accepted HEART_PRESENT masks remain for M1/C3/AU3/C4.")
        matched_tokens = {
            str(row["image_token"])
            for row in matching_manifest
            if DataStage._as_int(row.get("selected_for_matched_cohort"), 0) == 1
        }
        if not matched_tokens:
            raise RuntimeError("The current matching stage produced no usable matched pairs.")
        rows = []
        for original in dataset_rows:
            row = dict(original)
            if row.get("attention_mask_bits") is None:
                raise RuntimeError(f"Missing current-session OOF prediction for {row['image_token']}.")
            row["keep_attention"] = int(
                DataStage._as_int(row.get("attention_valid_final"), 0) == 1
                and DataStage._as_int(row.get("attention_heart_present"), 1) == 1
                and DataStage._as_int(row.get("quality_valid"), 0) == 1
                and DataStage._as_float(row.get("attention_area_ratio"), 0.0) >= 0.003
            )
            row["keep_manual_matched"] = int(row["image_token"] in accepted_manual)
            row["keep_cross_class_matched"] = int(
                row["image_token"] in matched_tokens and row["keep_attention"] == 1
            )
            rows.append(row)
        if not rows:
            raise RuntimeError("No images are eligible for feature extraction.")
        print(f"[FEATURES] rebuilding all 11 experiments from {len(rows)} source images")
        extractor = DataStage.prepare_model(FrozenEfficientNet(), device).eval()
        loader = DataLoader(
            FeatureDataset(rows),
            batch_size=12 if device.type == "cuda" else 4,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        pool = StreamingPatientPool(FeatureStage.MODES)
        started = time.perf_counter()

        # Reuse embeddings only inside this run, never from a previous run or file.
        alias_rules = {
            "B0_FULL_IMAGE": (
                ("B2_ATTENTION_ELIGIBLE_FULL_IMAGE", "keep_attention"),
                ("B1_MATCHED_FULL_IMAGE", "keep_cross_class_matched"),
            ),
            "AU1_ATTENTION_ROI": (("AU2_MATCHED_ATTENTION_ROI", "keep_cross_class_matched"),),
            "C1_ATTENTION_COMPLEMENT": (("C2_MATCHED_ATTENTION_COMPLEMENT", "keep_cross_class_matched"),),
        }

        def encode_groups(
            groups,
        ):
            valid_groups=[
                (mode, images, selected_rows)
                for mode, images, selected_rows in groups
                if len(selected_rows) > 0
                and images.shape[0] > 0
            ]
            if not valid_groups:
                return

            combined=torch.cat(
                [images.contiguous() for _, images, _ in valid_groups], dim=0
            )
            forward_batch=32 if device.type == "cuda" else 4
            embedding_chunks=[]
            for begin in range(0, combined.shape[0], forward_batch):
                images_device=DataStage.move_tensor(
                    combined[begin : begin + forward_batch], device
                )
                with DataStage.autocast(device):
                    output=extractor(images_device)
                embedding_chunks.append(output.float().cpu())
                del images_device, output

            embeddings=torch.cat(embedding_chunks, dim=0).numpy().astype(np.float32)
            cursor=0
            for mode, images, selected_rows in valid_groups:
                count=int(images.shape[0])
                current_embeddings=embeddings[cursor : cursor + count]
                pool.add(mode, current_embeddings, selected_rows)

                for alias_mode, selector_field in alias_rules.get(mode, ()):
                    alias_positions=[
                        position
                        for position, row in enumerate(selected_rows)
                        if DataStage._as_int(row.get(selector_field), 0) == 1
                    ]
                    if alias_positions:
                        index_array=np.asarray(alias_positions, dtype=np.int64)
                        pool.add(
                            alias_mode,
                            current_embeddings[index_array],
                            [selected_rows[position] for position in alias_positions],
                        )
                cursor +=count
            del combined, embeddings, embedding_chunks

        with torch.inference_mode():
            for robust, raw, content, attention, manual, indices in tqdm(
                loader, desc=f"EfficientNet feature bank ({device.type})"
            ):
                batch_rows = [rows[int(index)] for index in indices]
                attention_positions = [i for i, row in enumerate(batch_rows) if row["keep_attention"]]
                manual_positions = [i for i, row in enumerate(batch_rows) if row["keep_manual_matched"]]
                groups = [("B0_FULL_IMAGE", robust, batch_rows)]

                # Automatic ROI/complement plus aliases for the matched cohort.
                if attention_positions:
                    positions = torch.as_tensor(attention_positions, dtype=torch.long)
                    selected_raw = raw.index_select(0, positions)
                    selected_content = content.index_select(0, positions)
                    selected_mask = attention.index_select(0, positions)
                    support = FeatureStage.support_mask(selected_mask, selected_content)
                    selected_rows = [batch_rows[i] for i in attention_positions]
                    groups.extend([
                        ("AU1_ATTENTION_ROI", FeatureStage.region_normalize(selected_raw, support), selected_rows),
                        ("C1_ATTENTION_COMPLEMENT", FeatureStage.region_normalize(
                            selected_raw, (selected_content > 0.5).float() * (1.0 - support)
                        ), selected_rows),
                    ])

                # All four manual-reference experiments use exactly these same slices.
                if manual_positions:
                    positions = torch.as_tensor(manual_positions, dtype=torch.long)
                    selected_raw = raw.index_select(0, positions)
                    selected_content = content.index_select(0, positions)
                    manual_support = FeatureStage.support_mask(manual.index_select(0, positions), selected_content)
                    attention_support = FeatureStage.support_mask(attention.index_select(0, positions), selected_content)
                    selected_rows = [batch_rows[i] for i in manual_positions]
                    visible_content = (selected_content > 0.5).float()
                    regions = (
                        ("M1_MANUAL_ROI", manual_support),
                        ("C3_MANUAL_COMPLEMENT", visible_content * (1.0 - manual_support)),
                        ("AU3_ATTENTION_ROI_MANUAL_SUBSET", attention_support),
                        ("C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET", visible_content * (1.0 - attention_support)),
                    )
                    groups.extend(
                        (mode, FeatureStage.region_normalize(selected_raw, support), selected_rows)
                        for mode, support in regions
                    )
                encode_groups(groups)
                del groups

        bank = {mode: pool.finalize(mode) for mode in FeatureStage.MODES}
        FeatureStage.validate_cohort_alignment(bank)
        print("[FEATURES] all experiment representations built in RAM | "
              + DataStage.format_seconds(time.perf_counter() - started))
        del extractor, loader, pool
        DataStage.release_device(device)
        return bank

# =============================================================================
# STAGE 6 — PATIENT-LEVEL EVALUATION
# =============================================================================
class EvaluationStage:
    """Nested CV, confidence intervals, and paired AUC comparisons."""

    # Build classifier pipeline.
    @staticmethod
    def build_classifier_pipeline(c_value):
        return SklearnPipeline(
            [
                ("scale", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=0.95,
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
                        random_state=42,
                    ),
                ),
            ]
        )
    # Choose the probability threshold maximizing sensitivity + specificity - 1.
    @staticmethod
    def youden_threshold(labels, scores):
        fpr, tpr, thresholds=roc_curve(labels, scores)
        finite=np.isfinite(thresholds)
        if not np.any(finite):
            return 0.5
        index=np.argmax((tpr - fpr)[finite])
        return float(thresholds[finite][index])
    # Select c and threshold.
    @staticmethod
    def select_c_and_threshold(X, y, seed):
        class_counts=np.bincount(y, minlength=2)
        n_splits=min(3, int(class_counts.min()))
        if n_splits < 2:
            return 1.0, 0.5
        splitter=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best_c, best_auc, best_scores=1.0, -np.inf, None
        for c_value in (0.01, 0.1, 1.0, 10.0):
            scores=np.full(len(y), np.nan, dtype=np.float64)
            for train_index, valid_index in splitter.split(X, y):
                model=EvaluationStage.build_classifier_pipeline(c_value)
                model.fit(X[train_index], y[train_index])
                scores[valid_index]=model.predict_proba(X[valid_index])[:, 1]
            auc=roc_auc_score(y, scores)
            if (auc, -abs(math.log10(c_value))) > (best_auc, -abs(math.log10(best_c))):
                best_c, best_auc, best_scores=float(c_value), float(auc), scores.copy()
        threshold=EvaluationStage.youden_threshold(y, best_scores)
        return best_c, threshold
    # Create deterministic stratified outer/inner folds over patients, never slices.
    @staticmethod
    def patient_fold_map(reference):
        patients=np.asarray(reference["patient_ids"]).astype(str)
        labels=np.asarray(reference["y"], dtype=np.int64)
        class_counts=np.bincount(labels, minlength=2)
        n_splits=min(5, int(class_counts.min()))
        if n_splits < 2:
            raise RuntimeError("At least two patients are required in each class.")
        splitter=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        mapping={}
        for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
            for index in valid_index:
                mapping[patients[index]]=fold
        return mapping
    # Bootstrap patient-level AUC to report a sampling confidence interval.
    @staticmethod
    def auc_confidence_interval(labels, scores, repeats, seed):
        rng=np.random.default_rng(seed)
        values=[]
        for _ in range(int(repeats)):
            indices=rng.integers(0, len(labels), size=len(labels))
            if len(np.unique(labels[indices])) < 2:
                continue
            values.append(roc_auc_score(labels[indices], scores[indices]))
        if not values:
            return np.nan, np.nan
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    # Calculate classification metrics.
    @staticmethod
    def classification_metrics(labels, scores, predictions):
        tn, fp, fn, tp=confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        ci_low, ci_high=EvaluationStage.auc_confidence_interval(
            labels,
            scores,
            2000,
            42 + 9000,
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
    # Calculate paired auc difference.
    @staticmethod
    def paired_auc_difference(
        first,
        second,
        repeats,
        seed,
    ):
        merged=first.merge(second, on=["patient_id", "true_label"], suffixes=("_first", "_second"))
        labels=merged["true_label"].to_numpy(dtype=np.int64)
        first_scores=merged["score_first"].to_numpy(dtype=np.float64)
        second_scores=merged["score_second"].to_numpy(dtype=np.float64)
        if len(merged) < 4 or len(np.unique(labels)) < 2:
            return {
                "n_patients": len(merged),
                "auc_difference_first_minus_second": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        observed=float(roc_auc_score(labels, first_scores) - roc_auc_score(labels, second_scores))
        rng=np.random.default_rng(seed)
        differences=[]
        for _ in range(int(repeats)):
            indices=rng.integers(0, len(labels), size=len(labels))
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
    # Select classifier settings inside outer training folds and return all results in RAM.
    @staticmethod
    def evaluate_experiments(bank):
        fold_map=EvaluationStage.patient_fold_map(bank["B0_FULL_IMAGE"])
        summary_rows=[]
        prediction_tables={}

        for mode in FeatureStage.MODES:
            values=bank[mode]
            X=np.asarray(values["X"], dtype=np.float32)
            y=np.asarray(values["y"], dtype=np.int64)
            patients=np.asarray(values["patient_ids"]).astype(str)
            if len(np.unique(y)) < 2:
                raise RuntimeError(f"{mode}: one class is missing.")
            unknown=sorted(set(patients) - set(fold_map))
            if unknown:
                raise RuntimeError(f"{mode}: patients missing from the fold map: {unknown}")
            patient_folds=np.asarray([fold_map[patient] for patient in patients], dtype=np.int64)
            scores=np.full(len(y), np.nan, dtype=np.float64)
            predictions=np.full(len(y), -1, dtype=np.int64)
            selected_cs=np.full(len(y), np.nan, dtype=np.float64)
            thresholds=np.full(len(y), np.nan, dtype=np.float64)

            for fold in sorted(np.unique(patient_folds)):
                train_index=np.flatnonzero(patient_folds != fold)
                valid_index=np.flatnonzero(patient_folds == fold)
                if len(np.unique(y[train_index])) < 2:
                    raise RuntimeError(
                        f"{mode}: fold {fold} does not contain both classes in training. "
                        "The matched cohort is too sparse; inspect the matching summary."
                    )
                best_c, threshold=EvaluationStage.select_c_and_threshold(
                    X[train_index], y[train_index], 42 + int(fold)
                )
                model=EvaluationStage.build_classifier_pipeline(best_c)
                model.fit(X[train_index], y[train_index])
                fold_scores=model.predict_proba(X[valid_index])[:, 1]
                scores[valid_index]=fold_scores
                predictions[valid_index]=(fold_scores >= threshold).astype(np.int64)
                selected_cs[valid_index]=best_c
                thresholds[valid_index]=threshold

            if not np.all(np.isfinite(scores)) or np.any(predictions < 0):
                raise RuntimeError(f"{mode}: incomplete OOF predictions.")
            metrics=EvaluationStage.classification_metrics(y, scores, predictions)
            table=pd.DataFrame(
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
            prediction_tables[mode]=table
            summary_rows.append(
                {
                    "mode": mode,
                    "experiment_group": (
                        "primary"
                        if mode in FeatureStage.MODES[:7]
                        else "mask_validation"
                    ),
                    "description": FeatureStage.MODE_DESCRIPTIONS[mode],
                    "cohort": FeatureStage.MODE_COHORTS[mode],
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

        summary=pd.DataFrame(summary_rows).sort_values("auc", ascending=False)

        comparisons=[]
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
            ),):
            comparison=EvaluationStage.paired_auc_difference(
                prediction_tables[first_mode],
                prediction_tables[second_mode],
                2000,
                42 + len(comparisons) * 100,
            )
            comparisons.append(
                {
                    "comparison": question,
                    "first_mode": first_mode,
                    "second_mode": second_mode,
                    **comparison,
                }
            )
        return summary, pd.DataFrame(comparisons), prediction_tables

# =============================================================================
# STAGE 7 — SIMPLE USER API
# =============================================================================
class Pipeline:
    """Only two commands are needed: Pipeline.run() and Pipeline.review()."""

    last_run = None  # current-session state only; never written as history

    @staticmethod
    def run(dataset_path=None, manual_root=None, minimum_masks=None):
        """Run all research stages in one call; write only manual targets when necessary."""
        line = "=" * 88
        print(f"\n{line}\nCARDIAC MRI CAD — ONE-STEP IN-MEMORY RUN\n{line}")
        print("[ORDER] Dataset -> quality/manual targets -> five Attention U-Nets -> OOF masks")
        print("[ORDER] -> Sick/Normal matching -> 11 EfficientNet representations -> patient-level evaluation")
        print("[REVIEW] Optional HTML editing is separate; new labels apply to the next full run.")
        # A failed new run must never masquerade as the results of an older run.
        Pipeline.last_run = None
        DataStage.clear_image_cache()
        DataStage.seed_everything(include_cuda=False)
        dataset_path = Path(dataset_path or DataStage._default_dataset_path())
        workspace = DataStage.create_workspace(manual_root)
        rows = DataStage.discover_dataset(dataset_path, workspace)
        rows = DataStage.build_quality_audit(rows)
        state = {"rows": rows, "workspace": workspace, "status": "manual_targets"}
        Pipeline.last_run = state
        print(f"[PERSISTENT] {workspace.manual_masks}")
        print(f"[PERSISTENT] {workspace.manual_annotations}")
        print("[FILES] No output directory, checkpoint, automatic-mask PNG, CSV/JSON audit, or feature archive.")

        # STEP 1 — MANUAL TARGETS: preserve PNGs, explicit negative labels and weights.
        accepted, manual_audit, manual_summary = DataStage.audit_manual_masks(
            rows, workspace, minimum_masks=minimum_masks
        )
        state.update(manual_audit=manual_audit, manual_summary=manual_summary)
        print("\n[MANUAL TARGETS]")
        display(pd.DataFrame([manual_summary]))

        # STEP 2 — TRAIN: best weights stay on CPU in RAM until their OOF prediction.
        state["status"] = "attention_training"
        device = DataStage.start_device_stage("cuda", "Attention U-Net training")
        try:
            fold_states = AttentionStage.train_attention_crossfit(accepted, device)
        finally:
            DataStage.clear_image_cache()
            DataStage.finish_device_stage(device, "Attention U-Net training")
        state["training"] = {
            fold: {key: value for key, value in fold_state.items() if key != "state_dict"}
            for fold, fold_state in fold_states.items()
        }

        # STEP 3 — OOF PREDICTION: masks are losslessly packed in RAM, not saved.
        state["status"] = "attention_prediction"
        device = DataStage.start_device_stage("cuda", "Attention U-Net OOF prediction")
        try:
            rows, prediction_summary = AttentionStage.predict_attention_masks(rows, device, fold_states)
        finally:
            fold_states.clear()
            DataStage.clear_image_cache()
            DataStage.finish_device_stage(device, "Attention U-Net OOF prediction")
        state.update(rows=rows, predictions=rows, prediction_summary=prediction_summary)
        segmentation_summary, segmentation_metrics = AttentionStage.evaluate_oof_segmentation(manual_audit, rows)
        state.update(segmentation_summary=segmentation_summary, segmentation_metrics=segmentation_metrics)
        print("\n[ATTENTION OOF SUMMARY]")
        display(pd.DataFrame([prediction_summary]))
        display(pd.DataFrame([segmentation_summary]))

        # STEP 4 — MATCHING: retain both the strict core and expanded cohort in RAM.
        # Review remains available even if matching or a later stage fails.
        state["status"] = "matching"
        matching, matching_summary = MatchingStage.build_cross_class_matching(rows)
        state.update(matching=matching, matching_summary=matching_summary)
        print("\n[MATCHING]")
        display(pd.DataFrame([matching_summary]))

        # STEP 5 — FEATURES: frozen ImageNet EfficientNet-B0; all eleven modes.
        state["status"] = "feature_extraction"
        device = DataStage.start_device_stage("cuda", "EfficientNet feature extraction")
        try:
            bank = FeatureStage.build_feature_bank(rows, accepted, matching, device)
        finally:
            DataStage.finish_device_stage(device, "EfficientNet feature extraction")
        state["feature_bank"] = bank

        # STEP 6 — EVALUATION: nested patient-level CV and paired bootstrap AUC tests.
        state["status"] = "evaluation"
        results, comparisons, patient_predictions = EvaluationStage.evaluate_experiments(bank)
        state.update(results=results, comparisons=comparisons, patient_predictions=patient_predictions, status="complete")
        print("\n[EVALUATION SUMMARY]")
        display(results)
        print("\n[PAIRED AUC COMPARISONS]")
        display(comparisons)
        print("[DONE] Results are displayed above and available through result / Pipeline.last_run.")
        return state
    @staticmethod
    def review(scope="invalid", limit=300, start_index=0, seed=42, dataset_path=None, manual_root=None):
        """Open HTML review from current RAM data, or prepare a CPU-only manual-label session."""
        # Explicit paths intentionally start a review of that dataset/workspace.
        if Pipeline.last_run is None or dataset_path is not None or manual_root is not None:
            workspace = DataStage.create_workspace(manual_root)
            dataset_path = Path(dataset_path or DataStage._default_dataset_path())
            rows = DataStage.discover_dataset(dataset_path, workspace)
            rows = DataStage.build_quality_audit(rows)
            Pipeline.last_run = {"rows": rows, "workspace": workspace, "status": "manual_review"}
        rows = Pipeline.last_run["rows"]
        workspace = Pipeline.last_run["workspace"]
        DataStage.register_existing_manual_masks(rows, workspace)
        if scope in {"invalid", "uncertain"} and not any(row.get("attention_mask_bits") is not None for row in rows):
            print("[REVIEW] This scope needs current-session OOF predictions. "
                  "Use scope='all' to draw initial manual masks, or run Pipeline.run() first.")
            return None
        try:
            queue = ReviewStage.select_review_rows(rows, workspace, scope=scope, limit=limit, seed=seed)
        except RuntimeError as error:
            print(error)
            return None
        editor = MaskEditor(queue, workspace, start_index=start_index, brush_radius=8, review_scope=scope)
        return editor.show()

# Importing definitions does not train models or create output directories.
DataStage.seed_everything(include_cuda=False)
print("[PIPELINE] Run all stages: result = Pipeline.run()")
print("[PIPELINE] Optional editing: editor = Pipeline.review(scope='invalid')")
print("[PIPELINE] Before the first training run: Pipeline.review(scope='all')")
print("[PIPELINE] Persistent data: manual_masks/ + manual_annotation_labels.csv only")
print("[PIPELINE] After manual changes, rerun Pipeline.run(); no generated state is resumed.")
