# %% [1] IMPORTURI ȘI SETĂRI
"""Pipeline simplificat pentru clasificarea CAD din imagini cardiac MRI.

Ideea centrală:
1. Directory_* este pacientul și nu traversează niciodată foldurile.
2. Attention U-Net este antrenat numai din măști manuale valide.
3. Fiecare pacient este prezis de un model care nu a văzut acel pacient.
4. EfficientNet este folosit doar ca extractor frozen de caracteristici.
5. Clasificarea și toate metricile sunt calculate la nivel de pacient.

Codul este împărțit în clase mici. Setările care se modifică frecvent sunt
centralizate mai jos; restul claselor implementează câte o singură etapă.
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
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm.auto import tqdm

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convertește sigur valori citite din CSV, inclusiv șiruri goale."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _as_int(value: Any, default: int = 0) -> int:
    return int(round(_as_float(value, float(default))))


def _default_dataset_path() -> Path:
    """Alege automat calea Kaggle; în afara Kaggle folosește CAD_DATASET_PATH."""

    env_value = os.environ.get("CAD_DATASET_PATH", "").strip()
    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            Path("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"),
            Path(r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else Path.cwd() / "CAD Cardiac MRI Dataset"


def _default_workspace_path() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cad_attention_unet_workspace")
    return Path.cwd() / "cad_attention_unet_workspace"


class PathSettings:
    """Toate locațiile importante. În mod normal se modifică doar DATASET_PATH."""

    DATASET_PATH = _default_dataset_path()
    WORKSPACE_ROOT = Path(
        os.environ.get("CAD_WORKSPACE_ROOT", str(_default_workspace_path()))
    )


class RuntimeSettings:
    """Setări generale de rulare și reproducibilitate."""

    DEVICE = os.environ.get("CAD_DEVICE", "auto").strip().lower()  # auto/cpu/cuda
    RANDOM_SEED = 42
    NUM_WORKERS = 0  # 0 este cel mai robust în notebook/Kaggle
    USE_AMP_ON_CUDA = True
    CPU_THREADS = max(1, min(8, os.cpu_count() or 1))
    PNG_COMPRESSION = 9
    IMAGE_RAM_CACHE_ITEMS = 2048


class ImageSettings:
    """Geometria comună pentru segmentare și clasificare."""

    SEGMENTATION_SIZE = 256
    CLASSIFICATION_SIZE = 224
    STANDARDIZED_CONTENT_LONG_SIDE = 240

    # Detectarea conservatoare a benzilor negre de la marginea exportului JPEG.
    DARK_LINE_MAX_MEAN = 12.0
    DARK_LINE_MAX_STD = 4.0
    DARK_PIXEL_MAX_VALUE = 20
    DARK_PIXEL_MIN_FRACTION = 0.98
    MAX_CROP_FRACTION_PER_SIDE = 0.20
    MIN_RETAINED_FRACTION = 0.60
    MIN_PADDING_RUN = 2

    ROBUST_LOWER_PERCENTILE = 1.0
    ROBUST_UPPER_PERCENTILE = 99.0

    # Normalizarea este calculată numai în regiunea care rămâne vizibilă.
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
    """Setările Attention U-Net și regulile fixe de validare a măștilor."""

    FOLDS = 5
    BASE_CHANNELS = 24
    EPOCHS = 12
    EARLY_STOPPING_PATIENCE = 4
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BCE_WEIGHT = 0.50

    BATCH_SIZE_CUDA = 12
    BATCH_SIZE_CPU = 4
    INFERENCE_BATCH_SIZE_CUDA = 12
    INFERENCE_BATCH_SIZE_CPU = 4

    MIN_MANUAL_MASKS = 800
    MANUAL_MIN_AREA_RATIO = 0.0005
    MANUAL_MAX_AREA_RATIO = 0.98
    VALIDATION_PATIENT_FRACTION = 0.20

    CALIBRATION_THRESHOLDS = (
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
    )
    DEFAULT_THRESHOLD = 0.50
    PREDICTION_MIN_AREA_RATIO = 0.003
    PREDICTION_MAX_AREA_RATIO = 0.65
    PREDICTION_MIN_PEAK_PROBABILITY = 0.50
    REPAIR_THRESHOLD_OFFSETS = (-0.15, -0.10, -0.05, 0.05, 0.10, 0.15)
    REPAIR_MAX_BOUNDARY_TOUCH = 0.35
    REPAIR_MORPHOLOGY_KERNEL = 5

    SUPPORT_DILATION_KERNEL = 15
    SAVE_ALL_PREDICTED_MASKS = True


class ClassificationSettings:
    """Setările extractorului frozen și ale clasificării la nivel de pacient."""

    FEATURE_BATCH_SIZE_CUDA = 16
    FEATURE_BATCH_SIZE_CPU = 4
    USE_IMAGENET_WEIGHTS = True
    PCA_EXPLAINED_VARIANCE = 0.95
    OUTER_FOLDS = 5
    INNER_FOLDS = 3
    C_GRID = (0.01, 0.1, 1.0, 10.0)
    BOOTSTRAP_REPEATS = 2000

    # Au rămas doar experimentele care răspund direct întrebării proiectului.
    MODES = (
        "FULL_IMAGE",
        "AU1_ATTENTION_ROI",
        "AU5_ATTENTION_COMPLEMENT",
        "AU6_MANUAL_ROI",
        "AU7_MANUAL_COMPLEMENT",
        "AU8_ATTENTION_MATCHED_MANUAL_ROI",
        "AU9_ATTENTION_MATCHED_MANUAL_COMPLEMENT",
    )


class ReviewSettings:
    """Selectarea imaginilor pentru corectare manuală."""

    NEW_IMAGES_PER_PATIENT = 10
    PHASH_MAX_DISTANCE = 6
    DEFAULT_LIMIT = 300
    BRUSH_RADIUS = 8


class Settings:
    """Un singur punct de acces pentru toate clasele de setări."""

    Paths = PathSettings
    Runtime = RuntimeSettings
    Image = ImageSettings
    Segmentation = SegmentationSettings
    Classification = ClassificationSettings
    Review = ReviewSettings


class RuntimeManager:
    """Alege CPU/GPU, fixează seed-urile și eliberează memoria GPU."""

    @staticmethod
    def resolve_device(requested: str | None = None) -> torch.device:
        requested = (requested or Settings.Runtime.DEVICE).strip().lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device trebuie să fie 'auto', 'cpu' sau 'cuda'.")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA a fost cerut, dar nu este disponibil.")
            return torch.device("cuda")
        if requested == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def seed_everything(seed: int | None = None) -> None:
        seed = int(Settings.Runtime.RANDOM_SEED if seed is None else seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(Settings.Runtime.CPU_THREADS)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

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
    def autocast(device: torch.device):
        """Folosește mixed precision numai pe GPU; pe CPU nu schimbă nimic."""

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
        if device.type == "cuda":
            torch.cuda.synchronize()
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
    """O imagine și identitatea pacientului/seriei din care provine."""

    image_path: str
    label: int
    patient_id: str
    series_id: str
    image_token: str
    segmentation_fold: int


@dataclass(frozen=True)
class Workspace:
    """Fișiere persistente produse de etapele pipeline-ului."""

    root: Path
    manual_masks: Path
    predicted_masks: Path
    mask_overlays: Path
    checkpoints: Path
    outputs: Path
    dataset_manifest: Path
    quality_audit: Path
    manual_audit: Path
    prediction_audit: Path
    invalid_predictions: Path
    review_history: Path
    training_summary: Path
    prediction_summary: Path
    feature_bank: Path
    feature_metadata: Path
    results_csv: Path
    predictions_dir: Path


class FileManager:
    """Scrieri atomice: un fișier incomplet nu înlocuiește rezultatul bun anterior."""

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
            prediction_audit=root / "simple_attention_prediction_audit.csv",
            invalid_predictions=root / "attention_invalid_after_retrain.csv",
            review_history=root / "simple_review_history.csv",
            training_summary=root / "simple_training_summary.json",
            prediction_summary=root / "simple_prediction_summary.json",
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
        """Evită fișiere PNG parțiale și eroarea repetată `libpng Write Error`."""

        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(image)
        ok, encoded = cv2.imencode(
            ".png",
            array,
            [cv2.IMWRITE_PNG_COMPRESSION, int(Settings.Runtime.PNG_COMPRESSION)],
        )
        if not ok:
            raise RuntimeError(f"OpenCV nu a putut encoda PNG-ul: {path}")
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


class DatasetManager:
    """Descoperă imaginile și păstrează pacientul ca unitate statistică."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    @staticmethod
    def _relative_token_path(image_path: Path) -> str:
        parts = image_path.parts
        for index, part in enumerate(parts):
            if str(part).startswith("Directory_"):
                return "/".join(map(str, parts[index:]))
        raise ValueError(f"Calea nu conține Directory_*: {image_path}")

    @staticmethod
    def image_token(image_path: Path, patient_id: str, series_id: str) -> str:
        """Păstrează exact schema veche, astfel încât măștile manuale existente rămân valide."""

        relative = DatasetManager._relative_token_path(image_path)
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        safe_series = str(series_id).replace("/", "__").replace("\\", "__").replace(" ", "_")
        return f"{patient_id}__{safe_series}__{digest}"

    @staticmethod
    def patient_folds(
        patients: dict[str, int] | Iterable[str],
    ) -> dict[str, int]:
        """Împarte pacienții determinist; când există etichete, păstrează echilibrul claselor."""

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
        """Scanează folderul fără să decodeze imaginile.

        `Directory_*` este pacientul. Primul subfolder din pacient este doar un
        proxy de serie; foldurile sunt create după pacient, nu după imagine.
        """

        dataset_path = Path(dataset_path)
        started = time.perf_counter()
        raw_rows: list[tuple[Path, int, str, str]] = []
        patient_to_label: dict[str, int] = {}

        for class_name, label in (("Normal", 0), ("Sick", 1)):
            class_path = dataset_path / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(f"Lipsește folderul: {class_path}")

            for patient_path in sorted(class_path.iterdir()):
                if not patient_path.is_dir() or not patient_path.name.lower().startswith("directory_"):
                    continue
                patient_id = patient_path.name
                old_label = patient_to_label.get(patient_id)
                if old_label is not None and old_label != label:
                    raise RuntimeError(f"{patient_id} apare în ambele clase.")
                patient_to_label[patient_id] = label

                # Imagini direct în Directory_*.
                for image_path in sorted(patient_path.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in DatasetManager.IMAGE_EXTENSIONS:
                        raw_rows.append((image_path, label, patient_id, f"{patient_id}/__ROOT__"))

                # Fiecare copil imediat este un proxy de serie; subfolderele lui nu îl despart.
                for series_path in sorted(path for path in patient_path.iterdir() if path.is_dir()):
                    series_id = f"{patient_id}/{series_path.name}"
                    for image_path in sorted(series_path.rglob("*")):
                        if image_path.is_file() and image_path.suffix.lower() in DatasetManager.IMAGE_EXTENSIONS:
                            raw_rows.append((image_path, label, patient_id, series_id))

        if not raw_rows:
            raise RuntimeError(f"Nu au fost găsite imagini în {dataset_path}")

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
            raise RuntimeError("Au fost generate token-uri duplicate pentru imagini.")

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

        print("[DATE] Pacienți:", len(patient_to_label))
        print("[DATE] Normal:", sum(label == 0 for label in patient_to_label.values()))
        print("[DATE] Sick:", sum(label == 1 for label in patient_to_label.values()))
        print("[DATE] Imagini:", len(samples))
        print("[DATE] Proxy-uri de serie:", len({sample.series_id for sample in samples}))
        print("[DATE] Scanare:", RuntimeManager.format_seconds(time.perf_counter() - started))
        return samples

    @staticmethod
    def rows(samples: Sequence[Sample], workspace: Workspace) -> list[dict[str, Any]]:
        return [
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


# %% [2] PREPROCESAREA IMAGINILOR, CALITATE ȘI MĂȘTI MANUALE
class ImageProcessor:
    """Transformări fără etichetă, identice pentru train, validare și test.

    Geometria de 256x256 este păstrată compatibilă cu măștile manuale create în
    versiunile anterioare ale notebook-ului.
    """

    @staticmethod
    def read_gray(path: Path | str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"OpenCV nu poate citi imaginea: {path}")
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
            raise ValueError("Imaginea trebuie să fie grayscale 2D.")
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
        """Returnează patru imagini aliniate: robustă, min-max, raw și content mask."""

        top, bottom, left, right = ImageProcessor.detect_padding_bounds(image)
        cropped = image[top:bottom, left:right]
        if cropped.size == 0:
            raise RuntimeError("Crop-ul automat a produs o imagine goală.")
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
        _, minmax, _, _ = ImageProcessor.standardize(image)
        return np.clip(np.round(minmax * 255.0), 0, 255).astype(np.uint8)

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
    """Cache RAM mic pentru imaginile 256x256 folosite repetat în antrenare."""

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
    """Calculează blur, zgomot și pHash fără a folosi eticheta CAD."""

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
                    f"[CALITATE] {index}/{len(rows)} | "
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
        print(f"[CALITATE] valide={valid}/{len(records)} | {workspace.quality_audit}")
        return {str(row["image_token"]): row for row in records}


class MaskManager:
    """Validează măștile manuale și construiește cohorta de antrenare."""

    MANUAL_AUDIT_FIELDS = (
        "image_token",
        "patient_id",
        "series_id",
        "manual_mask_path",
        "status",
        "area_ratio",
        "quality_valid",
        "quality_reason",
        "reason",
    )

    @staticmethod
    def read_binary(path: Path | str, size: int | None = None) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Masca nu poate fi citită: {path}")
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
    def audit_manual_masks(
        rows: Sequence[dict[str, Any]],
        quality: dict[str, dict[str, Any]],
        workspace: Workspace,
        minimum_masks: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Păstrează numai măști non-goale pe imagini suficient de clare.

        Măștile goale NU sunt folosite ca ținte de antrenare. Fișierul lor rămâne
        însă în `manual_masks`, deci imaginea și vecinii ei apropiați pot fi
        excluși ulterior din coada `novel`.
        """

        minimum_masks = int(minimum_masks or Settings.Segmentation.MIN_MANUAL_MASKS)
        by_token = {row["image_token"]: dict(row) for row in rows}
        manual_files = sorted(workspace.manual_masks.glob("*.png"))
        accepted: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        patient_counts: dict[str, int] = defaultdict(int)
        fold_counts: dict[int, int] = defaultdict(int)

        for path in manual_files:
            row = by_token.get(path.stem)
            if row is None:
                audit.append(
                    {
                        "image_token": path.stem,
                        "manual_mask_path": str(path),
                        "status": "REJECTED",
                        "reason": "token_absent_from_dataset",
                    }
                )
                continue
            mask_qc = MaskManager.manual_qc(path)
            quality_row = quality.get(path.stem, {})
            reasons = []
            if not mask_qc["usable"]:
                reasons.append(mask_qc["reason"])
            if _as_int(quality_row.get("quality_valid"), 0) != 1:
                reasons.append(quality_row.get("quality_reason", "quality_invalid"))
            status = "ACCEPTED" if not reasons else "REJECTED"
            audit.append(
                {
                    "image_token": row["image_token"],
                    "patient_id": row["patient_id"],
                    "series_id": row["series_id"],
                    "manual_mask_path": str(path),
                    "status": status,
                    "area_ratio": mask_qc["area_ratio"],
                    "quality_valid": quality_row.get("quality_valid", 0),
                    "quality_reason": quality_row.get("quality_reason", ""),
                    "reason": ";".join(filter(None, reasons)),
                }
            )
            if status == "ACCEPTED":
                row.update(
                    {
                        "manual_mask_path": str(path),
                        "manual_mask_area_ratio": float(mask_qc["area_ratio"]),
                        "quality_valid": 1,
                        "perceptual_hash": quality_row.get("perceptual_hash", ""),
                    }
                )
                accepted.append(row)
                patient_counts[row["patient_id"]] += 1
                fold_counts[int(row["segmentation_fold"])] += 1

        FileManager.write_csv(workspace.manual_audit, audit, MaskManager.MANUAL_AUDIT_FIELDS)
        rejected = sum(row["status"] == "REJECTED" for row in audit)
        empty = sum("empty" in str(row.get("reason", "")) for row in audit)
        blur_or_noise = sum(
            row["status"] == "REJECTED" and bool(row.get("quality_reason")) for row in audit
        )
        summary = {
            "manual_png_files": len(manual_files),
            "accepted_manual_masks": len(accepted),
            "rejected_manual_masks": rejected,
            "rejected_empty_manual_masks": empty,
            "rejected_blur_or_noise": blur_or_noise,
            "patients_with_accepted_masks": len(patient_counts),
            "masks_per_patient": dict(sorted(patient_counts.items())),
            "masks_per_fold": {
                str(fold): int(fold_counts.get(fold, 0))
                for fold in range(Settings.Segmentation.FOLDS)
            },
            "audit": str(workspace.manual_audit),
        }
        print(
            "[MĂȘTI MANUALE] "
            f"accepted={len(accepted)}, rejected={rejected}, empty={empty}, "
            f"blur_or_noise={blur_or_noise}, patients={len(patient_counts)}"
        )

        if len(accepted) < minimum_masks:
            raise RuntimeError(
                f"Au rămas {len(accepted)} măști valide, sub minimul {minimum_masks}. "
                f"Vezi {workspace.manual_audit}"
            )
        missing_folds = [
            fold for fold in range(Settings.Segmentation.FOLDS) if fold_counts.get(fold, 0) == 0
        ]
        if missing_folds:
            raise RuntimeError(f"Nu există măști manuale valide în foldurile {missing_folds}.")
        if len(patient_counts) < Settings.Segmentation.FOLDS + 1:
            raise RuntimeError("Prea puțini pacienți au măști valide pentru cross-fitting.")
        return accepted, summary


# %% [3] ATTENTION U-NET: ANTRENARE CROSS-FIT ȘI PREDICȚIA MĂȘTILOR
class AttentionConvBlock(nn.Module):
    """Două convoluții. GroupNorm funcționează stabil și la batch-uri mici pe CPU."""

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
    """Învață cât din conexiunea U-Net de tip skip trebuie păstrat."""

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
    """Attention U-Net binar pentru regiunea cardiacă."""

    def __init__(self, base_channels: int | None = None):
        super().__init__()
        base = int(base_channels or Settings.Segmentation.BASE_CHANNELS)
        self.encoder1 = AttentionConvBlock(1, base)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class SegmentationDataset(Dataset):
    """Imagini 256x256 și măști manuale; augmentările păstrează alinierea."""

    def __init__(self, rows: Sequence[dict[str, Any]], augment: bool, seed: int):
        self.rows = list(rows)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = ImageCache.get(row["image_path"]).astype(np.float32) / 255.0
        mask = MaskManager.read_binary(row["manual_mask_path"]).astype(np.float32)

        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            if rng.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            angle = float(rng.uniform(-7.0, 7.0))
            matrix = cv2.getRotationMatrix2D(
                (Settings.Image.SEGMENTATION_SIZE / 2, Settings.Image.SEGMENTATION_SIZE / 2),
                angle,
                1.0,
            )
            image = cv2.warpAffine(
                image,
                matrix,
                (Settings.Image.SEGMENTATION_SIZE, Settings.Image.SEGMENTATION_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (Settings.Image.SEGMENTATION_SIZE, Settings.Image.SEGMENTATION_SIZE),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            contrast = float(rng.uniform(0.90, 1.10))
            brightness = float(rng.uniform(-0.05, 0.05))
            image = np.clip(image * contrast + brightness, 0.0, 1.0)

        return (
            torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float(),
            torch.from_numpy(np.ascontiguousarray(mask > 0.5)).unsqueeze(0).float(),
            str(row["patient_id"]),
        )


class SegmentationInferenceDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = ImageCache.get(self.rows[index]["image_path"]).astype(np.float32) / 255.0
        return torch.from_numpy(image).unsqueeze(0).float(), int(index)


class SegmentationManager:
    """Antrenează câte un model per fold și generează măști fără leakage."""

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
        return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()

    @staticmethod
    def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice_loss = 1.0 - SegmentationManager._dice_from_logits(logits, targets)
        weight = float(Settings.Segmentation.BCE_WEIGHT)
        return weight * bce + (1.0 - weight) * dice_loss

    @staticmethod
    def _manual_fingerprint(rows: Sequence[dict[str, Any]], fold: int) -> str:
        payload = {
            "schema": "simple-attention-manual-only-v1",
            "fold": int(fold),
            "size": Settings.Image.SEGMENTATION_SIZE,
            "base_channels": Settings.Segmentation.BASE_CHANNELS,
            "epochs": Settings.Segmentation.EPOCHS,
            "learning_rate": Settings.Segmentation.LEARNING_RATE,
            "weight_decay": Settings.Segmentation.WEIGHT_DECAY,
            "seed": Settings.Runtime.RANDOM_SEED,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        for row in sorted(rows, key=lambda item: item["image_token"]):
            mask_path = Path(row["manual_mask_path"])
            image_path = Path(row["image_path"])
            image_stat = image_path.stat()
            digest.update(row["image_token"].encode("utf-8"))
            digest.update(str(image_stat.st_size).encode("ascii"))
            digest.update(str(image_stat.st_mtime_ns).encode("ascii"))
            digest.update(FileManager.sha256_file(mask_path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _validation_patients(patient_ids: Iterable[str], target_fold: int) -> set[str]:
        ordered = sorted(
            set(map(str, patient_ids)),
            key=lambda patient_id: hashlib.sha256(
                f"{Settings.Runtime.RANDOM_SEED}|validation|{target_fold}|{patient_id}".encode("utf-8")
            ).hexdigest(),
        )
        count = max(1, int(round(len(ordered) * Settings.Segmentation.VALIDATION_PATIENT_FRACTION)))
        count = min(count, max(1, len(ordered) - 1))
        return set(ordered[:count])

    @staticmethod
    def _calibrate_threshold(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, Any]:
        probabilities: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        patients: list[str] = []
        model.eval()
        with torch.inference_mode():
            for images, masks, patient_ids in loader:
                images = images.to(device, non_blocking=True)
                probability = torch.sigmoid(model(images)).cpu().numpy()[:, 0]
                probabilities.extend(probability)
                targets.extend(masks.numpy()[:, 0])
                patients.extend(map(str, patient_ids))

        best = None
        for threshold in Settings.Segmentation.CALIBRATION_THRESHOLDS:
            by_patient: dict[str, list[float]] = defaultdict(list)
            invalid = 0
            for probability, target, patient_id in zip(probabilities, targets, patients):
                prediction = (probability >= threshold).astype(np.uint8)
                prediction = SegmentationManager._largest_component(prediction)
                intersection = float(np.logical_and(prediction, target > 0.5).sum())
                denominator = float(prediction.sum() + (target > 0.5).sum())
                dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
                by_patient[patient_id].append(dice)
                valid, _ = SegmentationManager._validate_mask(prediction, probability)
                invalid += int(not valid)
            patient_dice = [float(np.mean(values)) for values in by_patient.values()]
            mean_dice = float(np.mean(patient_dice)) if patient_dice else 0.0
            invalid_rate = float(invalid / max(1, len(probabilities)))
            score = mean_dice - 0.25 * invalid_rate
            candidate = {
                "threshold": float(threshold),
                "patient_balanced_dice": mean_dice,
                "invalid_rate": invalid_rate,
                "score": score,
            }
            if best is None or (candidate["score"], -abs(threshold - 0.5)) > (
                best["score"],
                -abs(best["threshold"] - 0.5),
            ):
                best = candidate
        return best or {
            "threshold": Settings.Segmentation.DEFAULT_THRESHOLD,
            "patient_balanced_dice": 0.0,
            "invalid_rate": 1.0,
            "score": -1.0,
        }

    @staticmethod
    def train_crossfit(
        accepted_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        device: torch.device,
        force: bool = False,
    ) -> dict[int, Path]:
        """Pentru foldul k, pacienții din foldul k nu apar în train sau calibrare."""

        RuntimeManager.seed_everything()
        checkpoint_map: dict[int, Path] = {}
        fold_summaries = []
        all_patients = sorted({row["patient_id"] for row in accepted_rows})

        for fold in range(Settings.Segmentation.FOLDS):
            fold_started = time.perf_counter()
            checkpoint_path = SegmentationManager._checkpoint_path(workspace, fold)
            non_test_rows = [
                row for row in accepted_rows if int(row["segmentation_fold"]) != fold
            ]
            validation_patients = SegmentationManager._validation_patients(
                (row["patient_id"] for row in non_test_rows), fold
            )
            train_rows = [
                row for row in non_test_rows if row["patient_id"] not in validation_patients
            ]
            validation_rows = [
                row for row in non_test_rows if row["patient_id"] in validation_patients
            ]
            fingerprint = SegmentationManager._manual_fingerprint(non_test_rows, fold)

            if checkpoint_path.is_file() and not force:
                checkpoint = SegmentationManager._torch_load(checkpoint_path)
                if checkpoint.get("fingerprint") == fingerprint:
                    print(f"[ATTENTION] Fold {fold}: checkpoint compatibil reutilizat.")
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
                raise RuntimeError(f"Fold {fold}: train/validation este gol.")

            print(
                f"[ATTENTION] Fold {fold}: train={len(train_rows)}, "
                f"validation={len(validation_rows)}, device={device.type}"
            )
            train_dataset = SegmentationDataset(
                train_rows, augment=True, seed=Settings.Runtime.RANDOM_SEED + fold * 1000
            )
            validation_dataset = SegmentationDataset(
                validation_rows, augment=False, seed=Settings.Runtime.RANDOM_SEED
            )
            generator = torch.Generator().manual_seed(Settings.Runtime.RANDOM_SEED + fold)
            train_loader = DataLoader(
                train_dataset,
                batch_size=RuntimeManager.train_batch_size(device),
                shuffle=True,
                generator=generator,
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

            model = AttentionUNet().to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Settings.Segmentation.LEARNING_RATE,
                weight_decay=Settings.Segmentation.WEIGHT_DECAY,
            )
            scaler = RuntimeManager.grad_scaler(device)
            best_state = None
            best_dice = -1.0
            best_epoch = 0
            patience = 0
            history = []

            for epoch in range(1, Settings.Segmentation.EPOCHS + 1):
                train_dataset.set_epoch(epoch)
                model.train()
                train_losses = []
                for images, masks, _ in train_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with RuntimeManager.autocast(device):
                        logits = model(images)
                        loss = SegmentationManager._loss(logits, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_losses.append(float(loss.detach().cpu()))

                model.eval()
                validation_losses = []
                validation_dice = []
                with torch.inference_mode():
                    for images, masks, _ in validation_loader:
                        images = images.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        with RuntimeManager.autocast(device):
                            logits = model(images)
                            loss = SegmentationManager._loss(logits, masks)
                        validation_losses.append(float(loss.cpu()))
                        binary = (torch.sigmoid(logits) >= 0.5).float()
                        intersection = (binary * masks).sum(dim=(1, 2, 3))
                        denominator = binary.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                        validation_dice.extend(
                            ((2 * intersection + 1e-6) / (denominator + 1e-6)).cpu().tolist()
                        )

                mean_train = float(np.mean(train_losses))
                mean_val = float(np.mean(validation_losses))
                mean_dice = float(np.mean(validation_dice))
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": mean_train,
                        "validation_loss": mean_val,
                        "validation_dice_0_5": mean_dice,
                    }
                )
                print(
                    f"  epoch={epoch:02d} train_loss={mean_train:.4f} "
                    f"val_loss={mean_val:.4f} val_dice={mean_dice:.4f}"
                )
                if mean_dice > best_dice + 1e-5:
                    best_dice = mean_dice
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= Settings.Segmentation.EARLY_STOPPING_PATIENCE:
                        print(f"  early stopping după epoch {epoch}.")
                        break

            if best_state is None:
                raise RuntimeError(f"Fold {fold}: nu s-a salvat niciun model.")
            model.load_state_dict(best_state)
            calibration = SegmentationManager._calibrate_threshold(model, validation_loader, device)
            checkpoint = {
                "schema": "simple-attention-manual-only-v1",
                "state_dict": best_state,
                "fingerprint": fingerprint,
                "fold": fold,
                "base_channels": Settings.Segmentation.BASE_CHANNELS,
                "best_epoch": best_epoch,
                "best_validation_dice_0_5": best_dice,
                "calibration": calibration,
                "train_patients": sorted({row["patient_id"] for row in train_rows}),
                "validation_patients": sorted(validation_patients),
                "excluded_test_patients": sorted(
                    patient for patient in all_patients
                    if any(
                        row["patient_id"] == patient and int(row["segmentation_fold"]) == fold
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
                    "best_validation_dice_0_5": best_dice,
                    "calibration": calibration,
                    "elapsed": RuntimeManager.format_seconds(time.perf_counter() - fold_started),
                }
            )
            RuntimeManager.release(device)

        FileManager.write_json(
            workspace.training_summary,
            {
                "training_mode": "manual_only_crossfit",
                "device": device.type,
                "accepted_manual_masks": len(accepted_rows),
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
                    f"Lipsește checkpointul foldului {fold}: {path}. Rulează train_attention()."
                )
            result[fold] = path
        return result

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return binary
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return binary
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == label).astype(np.uint8)

    @staticmethod
    def _mask_geometry(mask: np.ndarray) -> tuple[float, float]:
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        area = float(mask.mean())
        if not mask.any():
            return area, 1.0
        border = np.zeros_like(mask, dtype=bool)
        border[:2, :] = True
        border[-2:, :] = True
        border[:, :2] = True
        border[:, -2:] = True
        boundary_touch = float(np.logical_and(mask > 0, border).sum() / max(1, mask.sum()))
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
    def _validate_mask(mask: np.ndarray, probability: np.ndarray) -> tuple[bool, str]:
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
        return not reasons, ";".join(reasons)

    @staticmethod
    def _repair_probability(
        probability: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, bool, str, float, str]:
        """Încearcă praguri vecine; nu introduce niciodată un pătrat central fallback."""

        candidates = []
        thresholds = [threshold] + [
            float(np.clip(threshold + offset, 0.05, 0.95))
            for offset in Settings.Segmentation.REPAIR_THRESHOLD_OFFSETS
        ]
        for candidate_threshold in sorted(set(thresholds)):
            mask = SegmentationManager._candidate_mask(probability, candidate_threshold)
            valid, reason = SegmentationManager._validate_mask(mask, probability)
            area, boundary = SegmentationManager._mask_geometry(mask)
            if mask.any():
                mean_inside = float(probability[mask > 0].mean())
            else:
                mean_inside = 0.0
            score = mean_inside - 0.50 * boundary - 0.20 * abs(area - 0.15)
            candidates.append((valid, score, mask, candidate_threshold, reason))
        valid_candidates = [candidate for candidate in candidates if candidate[0]]
        if valid_candidates:
            _, _, mask, used_threshold, _ = max(valid_candidates, key=lambda item: item[1])
            return mask, True, "adaptive_threshold", float(used_threshold), ""
        _, _, mask, used_threshold, reason = max(candidates, key=lambda item: item[1])
        return mask, False, "unresolved", float(used_threshold), reason

    @staticmethod
    def _prediction_fingerprint(rows: Sequence[dict[str, Any]], checkpoint_map: dict[int, Path]) -> str:
        digest = hashlib.sha256()
        settings_payload = {
            "schema": "simple-predictions-v1",
            "thresholds": Settings.Segmentation.CALIBRATION_THRESHOLDS,
            "area": [
                Settings.Segmentation.PREDICTION_MIN_AREA_RATIO,
                Settings.Segmentation.PREDICTION_MAX_AREA_RATIO,
            ],
            "repair_offsets": Settings.Segmentation.REPAIR_THRESHOLD_OFFSETS,
        }
        digest.update(json.dumps(settings_payload, sort_keys=True).encode("utf-8"))
        digest.update(str(len(rows)).encode("ascii"))
        for row in rows:
            image_path = Path(row["image_path"])
            image_stat = image_path.stat()
            digest.update(str(row["image_token"]).encode("utf-8"))
            digest.update(str(row["segmentation_fold"]).encode("ascii"))
            digest.update(str(image_stat.st_size).encode("ascii"))
            digest.update(str(image_stat.st_mtime_ns).encode("ascii"))
        for fold, path in sorted(checkpoint_map.items()):
            digest.update(str(fold).encode("ascii"))
            digest.update(FileManager.sha256_file(path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def predict_all(
        rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        device: torch.device,
        checkpoint_map: dict[int, Path] | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Generează masca fiecărei imagini cu modelul foldului pacientului."""

        checkpoint_map = checkpoint_map or SegmentationManager.load_checkpoint_map(workspace)
        prediction_fingerprint = SegmentationManager._prediction_fingerprint(rows, checkpoint_map)
        old_summary = FileManager.read_json(workspace.prediction_summary, {}) or {}
        old_audit = FileManager.read_csv(workspace.prediction_audit)
        cached_masks_complete = bool(old_audit) and all(
            Path(row.get("predicted_attention_mask_path", "")).is_file()
            for row in old_audit
        )
        if (
            not force
            and old_summary.get("fingerprint") == prediction_fingerprint
            and len(old_audit) == len(rows)
            and cached_masks_complete
        ):
            print("[ATTENTION] Predicțiile compatibile sunt deja în cache.")
            return old_audit
        if old_audit and not cached_masks_complete:
            print("[ATTENTION] Cache incomplet: lipsesc măști PNG; predicțiile vor fi refăcute.")

        results: list[dict[str, Any] | None] = [None] * len(rows)
        started = time.perf_counter()
        for fold in range(Settings.Segmentation.FOLDS):
            fold_indices = [
                index for index, row in enumerate(rows)
                if int(row["segmentation_fold"]) == fold
            ]
            fold_rows = [rows[index] for index in fold_indices]
            checkpoint = SegmentationManager._torch_load(checkpoint_map[fold])
            model = AttentionUNet(base_channels=int(checkpoint.get("base_channels", Settings.Segmentation.BASE_CHANNELS)))
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            threshold = float(
                checkpoint.get("calibration", {}).get(
                    "threshold", Settings.Segmentation.DEFAULT_THRESHOLD
                )
            )
            checkpoint_fingerprint = str(checkpoint.get("fingerprint", ""))
            loader = DataLoader(
                SegmentationInferenceDataset(fold_rows),
                batch_size=RuntimeManager.inference_batch_size(device),
                shuffle=False,
                num_workers=Settings.Runtime.NUM_WORKERS,
                pin_memory=device.type == "cuda",
            )
            print(f"[ATTENTION] Predicție fold {fold}: {len(fold_rows)} imagini, device={device.type}")

            with torch.inference_mode():
                for images, local_indices in tqdm(loader, desc=f"Attention fold {fold}"):
                    images = images.to(device, non_blocking=True)
                    probability_tensor = torch.sigmoid(model(images))[:, 0]
                    probabilities = probability_tensor.cpu().numpy()

                    initial_masks = [
                        SegmentationManager._candidate_mask(probability, threshold)
                        for probability in probabilities
                    ]
                    initial_validity = [
                        SegmentationManager._validate_mask(mask, probability)
                        for mask, probability in zip(initial_masks, probabilities)
                    ]

                    invalid_positions = [
                        position for position, (valid, _) in enumerate(initial_validity)
                        if not valid
                    ]
                    if invalid_positions:
                        invalid_tensor = images[invalid_positions]
                        flipped_probability = torch.sigmoid(
                            model(torch.flip(invalid_tensor, dims=[3]))
                        )[:, 0]
                        flipped_probability = torch.flip(flipped_probability, dims=[2]).cpu().numpy()
                        for destination, tta_probability in zip(invalid_positions, flipped_probability):
                            probabilities[destination] = (
                                probabilities[destination] + tta_probability
                            ) / 2.0

                    for batch_position, local_index_tensor in enumerate(local_indices):
                        local_index = int(local_index_tensor)
                        global_index = fold_indices[local_index]
                        row = rows[global_index]
                        probability = probabilities[batch_position]
                        initial_mask = initial_masks[batch_position]
                        initial_valid, initial_reason = initial_validity[batch_position]
                        if initial_valid:
                            final_mask = initial_mask
                            final_valid = True
                            repair_method = "none"
                            used_threshold = threshold
                            final_reason = ""
                        else:
                            final_mask, final_valid, repair_method, used_threshold, final_reason = (
                                SegmentationManager._repair_probability(probability, threshold)
                            )
                            if invalid_positions:
                                repair_method = "tta+" + repair_method

                        area, boundary = SegmentationManager._mask_geometry(final_mask)
                        mask_path = Path(row["predicted_attention_mask_path"])
                        if Settings.Segmentation.SAVE_ALL_PREDICTED_MASKS or not final_valid:
                            FileManager.write_png(mask_path, final_mask.astype(np.uint8) * 255)

                        results[global_index] = {
                            **row,
                            "attention_valid_initial": int(initial_valid),
                            "attention_valid_final": int(final_valid),
                            "attention_invalid_reason_initial": initial_reason,
                            "attention_invalid_reason_final": final_reason,
                            "attention_repair_method": repair_method,
                            "attention_threshold_used": float(used_threshold),
                            "attention_area_ratio": area,
                            "attention_peak_probability": float(probability.max()),
                            "attention_boundary_touch_fraction": boundary,
                            "checkpoint_fingerprint": checkpoint_fingerprint,
                        }
            del model
            RuntimeManager.release(device)

        final_results = [result for result in results if result is not None]
        if len(final_results) != len(rows):
            raise RuntimeError("Predicția nu a produs câte un rând pentru fiecare imagine.")
        FileManager.write_csv(
            workspace.prediction_audit, final_results, SegmentationManager.PREDICTION_FIELDS
        )
        invalid_rows = [
            row for row in final_results if _as_int(row.get("attention_valid_final"), 0) != 1
        ]
        FileManager.write_csv(
            workspace.invalid_predictions, invalid_rows, SegmentationManager.PREDICTION_FIELDS
        )
        summary = {
            "fingerprint": prediction_fingerprint,
            "device": device.type,
            "images": len(final_results),
            "valid_initial": sum(_as_int(row.get("attention_valid_initial"), 0) for row in final_results),
            "valid_final": sum(_as_int(row.get("attention_valid_final"), 0) for row in final_results),
            "invalid_final": len(invalid_rows),
            "prediction_audit": str(workspace.prediction_audit),
            "invalid_queue": str(workspace.invalid_predictions),
            "elapsed": RuntimeManager.format_seconds(time.perf_counter() - started),
        }
        FileManager.write_json(workspace.prediction_summary, summary)
        print(
            "[ATTENTION] "
            f"valid_final={summary['valid_final']}/{summary['images']}, "
            f"invalid={summary['invalid_final']} | {workspace.prediction_audit}"
        )
        ImageCache.clear()
        return final_results


# %% [4] SELECTAREA ȘI EDITAREA MANUALĂ A MĂȘTILOR
class HammingBKTree:
    """Structură mică pentru a găsi rapid imagini cu pHash foarte apropiat."""

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
    """Construiește cozi compacte: invalide, noi/diverse, manuale sau toate."""

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
            row["image_token"]: row for row in FileManager.read_csv(workspace.quality_audit)
        }
        predictions = {
            row["image_token"]: row for row in FileManager.read_csv(workspace.prediction_audit)
        }
        merged = []
        for original in dataset_rows:
            row = dict(original)
            row.update(quality.get(row["image_token"], {}))
            row.update(predictions.get(row["image_token"], {}))
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
            valid = _as_int(row.get("attention_valid_final"), 0)
            peak = _as_float(row.get("attention_peak_probability"), 0.0)
            area = _as_float(row.get("attention_area_ratio"), 0.0)
            tie = hashlib.sha256(f"{seed}|{row['image_token']}".encode("utf-8")).hexdigest()
            return (valid, peak, abs(area - 0.15), tie)

        by_patient_series: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            by_patient_series[row["patient_id"]][row["series_id"]].append(row)

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

    @staticmethod
    def select(
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        scope: str = "invalid",
        limit: int | None = None,
        seed: int = 42,
        review_round: int = 1,
    ) -> list[dict[str, Any]]:
        """Scope-uri păstrate intenționat puține și clare.

        - invalid: numai predicțiile rămase invalide după reparare;
        - novel: imagini clare, fără mască manuală și diferite de cele deja marcate;
        - manual: imagini care au deja un fișier în manual_masks;
        - all: toate imaginile cu mască Attention existentă.
        """

        scope = str(scope).lower()
        if scope not in {"invalid", "novel", "manual", "all"}:
            raise ValueError("scope trebuie să fie invalid/novel/manual/all.")
        limit = int(Settings.Review.DEFAULT_LIMIT if limit is None else limit)
        rows = ReviewManager._merge_rows(dataset_rows, workspace)
        reviewed = ReviewManager._reviewed_tokens(workspace, review_round)

        if scope == "invalid":
            candidates = [
                row for row in rows
                if _as_int(row.get("attention_valid_final"), 1) == 0
                and Path(row["predicted_attention_mask_path"]).is_file()
            ]
            candidates.sort(
                key=lambda row: (
                    row["patient_id"],
                    row["series_id"],
                    _as_float(row.get("attention_peak_probability"), 0.0),
                )
            )
        elif scope == "manual":
            candidates = [row for row in rows if Path(row["manual_mask_path"]).is_file()]
            candidates.sort(key=lambda row: (row["patient_id"], row["series_id"], row["image_token"]))
        elif scope == "all":
            candidates = [
                row for row in rows if Path(row["predicted_attention_mask_path"]).is_file()
            ]
            candidates.sort(key=lambda row: (row["patient_id"], row["series_id"], row["image_token"]))
        else:
            # Orice fișier manual, inclusiv o mască intenționat goală, devine
            # referință de noutate și blochează imagini aproape identice.
            manual_hashes = []
            for row in rows:
                if Path(row["manual_mask_path"]).is_file() and row.get("perceptual_hash"):
                    manual_hashes.append(int(str(row["perceptual_hash"]), 16))
            if not manual_hashes:
                raise RuntimeError("Scope-ul novel necesită cel puțin o mască manuală existentă.")
            tree = HammingBKTree()
            for value in sorted(set(manual_hashes)):
                tree.add(value)

            candidates = []
            excluded = defaultdict(int)
            for row in rows:
                if Path(row["manual_mask_path"]).is_file():
                    excluded["already_manual"] += 1
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
                if tree.has_near(int(phash, 16), Settings.Review.PHASH_MAX_DISTANCE):
                    excluded["similar_to_manual"] += 1
                    continue
                candidates.append(row)
            candidates = ReviewManager._round_robin(
                candidates,
                limit=max(1, limit),
                per_patient=Settings.Review.NEW_IMAGES_PER_PATIENT,
                seed=seed,
            )
            print(f"[REVIEW novel] excluse={dict(excluded)}")

        if limit > 0 and scope != "novel":
            candidates = candidates[:limit]
        if not candidates:
            raise RuntimeError(f"Nu există imagini eligibile pentru scope={scope!r}.")
        queue_path = workspace.outputs / f"review_queue_{scope}_round_{review_round}.csv"
        fields = sorted({key for row in candidates for key in row.keys()})
        FileManager.write_csv(queue_path, candidates, fields)
        print(f"[REVIEW] scope={scope}, imagini={len(candidates)}, coadă={queue_path}")
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
                "mask_area_ratio": "" if mask is None else float((mask > 0).mean()),
                "manual_mask_path": row["manual_mask_path"],
            }
        )
        FileManager.write_csv(workspace.review_history, history, ReviewManager.HISTORY_FIELDS)


class MaskEditor:
    """Editor HTML5 compatibil Kaggle/JupyterLab, fără jupyter-matplotlib.

    Portocaliu = predicția automată. Magenta = masca editabilă. Click stânga
    desenează, click dreapta șterge. `Save & Next` scrie masca manuală atomic.
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
        if not rows:
            raise ValueError("Editorul necesită cel puțin un rând.")
        import ipywidgets as widgets

        self.rows = list(rows)
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

        self.output = widgets.Output()
        self.mask_sync = widgets.Textarea(
            value="",
            placeholder=f"CAD_MASK_SYNC_{self.widget_id}",
            layout=widgets.Layout(width="1px", height="1px", display="none"),
        )
        self.previous_button = widgets.Button(description="Previous")
        self.next_button = widgets.Button(description="Next")
        self.save_next_button = widgets.Button(description="Save & Next", button_style="success")
        self.accept_button = widgets.Button(description="Auto OK (not training)", button_style="info")
        self.skip_button = widgets.Button(description="Skip & Next")
        self.reset_button = widgets.Button(description="Reset to auto")
        self.raw_button = widgets.Button(description="Reset to image")
        self.clear_button = widgets.Button(description="Clear")
        self.delete_button = widgets.Button(description="Delete manual", button_style="warning")
        self.brush_slider = widgets.IntSlider(
            description="Brush", value=self.brush_radius, min=1, max=30, step=1
        )
        self.status = widgets.HTML()

        self.previous_button.on_click(lambda _: self._safe("Previous", self.previous))
        self.next_button.on_click(lambda _: self._safe("Next", self.next))
        self.save_next_button.on_click(lambda _: self._safe("Save & Next", self.save_next))
        self.accept_button.on_click(lambda _: self._safe("Auto OK", self.accept_auto))
        self.skip_button.on_click(lambda _: self._safe("Skip", self.skip))
        self.reset_button.on_click(lambda _: self._safe("Reset", self.reset_to_auto))
        self.raw_button.on_click(lambda _: self._safe("Reset to image", self.reset_to_image))
        self.clear_button.on_click(lambda _: self._safe("Clear", self.clear_editable))
        self.delete_button.on_click(lambda _: self._safe("Delete manual", self.delete_manual))
        self.brush_slider.observe(self._brush_changed, names="value")
        self.mask_sync.observe(self._mask_sync_changed, names="value")

        self.controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.previous_button,
                        self.next_button,
                        self.save_next_button,
                        self.reset_button,
                        self.raw_button,
                    ]
                ),
                widgets.HBox(
                    [self.accept_button, self.skip_button, self.clear_button, self.delete_button]
                ),
                widgets.HBox([self.brush_slider, self.status]),
                self.mask_sync,
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

    def _brush_changed(self, change: dict[str, Any]) -> None:
        self.brush_radius = int(change["new"])
        if self.image is not None:
            self.render()

    @staticmethod
    def _image_uri(rgb: np.ndarray) -> str:
        array = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Imaginea editorului nu poate fi encodată.")
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
            raise RuntimeError("Masca editorului nu poate fi encodată.")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    def load_current(self) -> None:
        row = self.rows[self.index]
        auto_path = Path(row["predicted_attention_mask_path"])
        if not auto_path.is_file():
            raise FileNotFoundError(f"Lipsește masca Attention: {auto_path}")
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
        image_uri = self._image_uri(np.stack([self.image] * 3, axis=-1))
        base_uri = self._mask_uri(self.base_mask)
        mask_uri = self._mask_uri(self.mask)
        canvas_id = f"cad_canvas_{self.widget_id}"
        placeholder = f"CAD_MASK_SYNC_{self.widget_id}"
        size = Settings.Image.SEGMENTATION_SIZE
        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:900px">
          <div style="margin-bottom:6px;font-size:14px">
            <b>{self.index + 1}/{len(self.rows)}</b> | {row['patient_id']} | {row['series_id']} |
            scope={self.review_scope}
          </div>
          <canvas id="{canvas_id}" width="{size * 3}" height="{size * 3}"
                  style="width:768px;height:768px;max-width:100%;border:1px solid #999;
                         cursor:crosshair;touch-action:none;user-select:none"></canvas>
          <div style="font-size:12px;margin-top:5px">
            Left drag = draw | Right drag = erase | S = Save & Next | N/P = navigation
          </div>
        </div>
        """
        js = f"""
        (() => {{
          const canvas = document.getElementById({json.dumps(canvas_id)});
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          const W = {size}, H = {size};
          const radius = {int(self.brush_radius)};
          const placeholder = {json.dumps(placeholder)};
          const image = new Image(), base = new Image(), initialMask = new Image();
          const maskCanvas = document.createElement('canvas');
          maskCanvas.width = W; maskCanvas.height = H;
          const maskCtx = maskCanvas.getContext('2d', {{willReadFrequently:true}});
          let drawing = false, erase = false, last = null;

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
            e.preventDefault(); drawing = true; erase = e.button === 2; last = null;
            try {{ canvas.setPointerCapture?.(e.pointerId); }} catch (_) {{}}
            paint(point(e));
          }}
          function move(e) {{
            if (!drawing) return;
            if (e.buttons === 0) return finish(e);
            e.preventDefault(); paint(point(e));
          }}
          function finish(e) {{
            if (!drawing) return;
            e.preventDefault?.(); drawing = false; last = null; sync();
          }}
          canvas.addEventListener('contextmenu', e => e.preventDefault(), true);
          canvas.addEventListener('pointerdown', begin, true);
          canvas.addEventListener('pointermove', move, true);
          canvas.addEventListener('pointerup', finish, true);
          canvas.addEventListener('pointercancel', finish, true);
          canvas.addEventListener('mousedown', begin, true);
          canvas.addEventListener('mousemove', move, true);
          canvas.addEventListener('mouseup', finish, true);
          window.addEventListener('keydown', e => {{
            const k = (e.key || '').toLowerCase();
            if (k === 's') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Save & Next')?.click();
            if (k === 'n') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Next')?.click();
            if (k === 'p') Array.from(document.querySelectorAll('button')).find(x => x.innerText === 'Previous')?.click();
          }});
          function load(target, source) {{
            return new Promise((resolve, reject) => {{ target.onload = resolve; target.onerror = reject; target.src = source; }});
          }}
          Promise.all([
            load(image, {json.dumps(image_uri)}),
            load(base, {json.dumps(base_uri)}),
            load(initialMask, {json.dumps(mask_uri)})
          ]).then(() => {{ maskCtx.clearRect(0,0,W,H); maskCtx.drawImage(initialMask,0,0,W,H); draw(); }});
        }})();
        """
        with self.output:
            clear_output(wait=True)
            display(HTML(html))
            display(Javascript(js))

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
            print("[EDITOR WARNING] Sincronizarea măștii a eșuat:", error)

    def _sync_python_mask(self) -> None:
        value = self._mask_uri(self.mask).split(",", 1)[1]
        if self.mask_sync.value != value:
            self.mask_sync.value = value

    def update_status(self, prefix: str = "") -> None:
        row = self.rows[self.index]
        manual = MaskManager.manual_qc(row["manual_mask_path"])
        self.status.value = (
            "<span style='margin-left:12px'>"
            f"<b>{prefix}</b> index={self.index + 1}/{len(self.rows)}; "
            f"manual={'yes' if manual['exists'] else 'no'}; usable={manual['usable']}; "
            f"attention_valid={row.get('attention_valid_final', '')}; "
            f"reason={row.get('attention_invalid_reason_final', '')}; "
            f"quality={row.get('quality_valid', '')}</span>"
        )

    def save(self) -> None:
        row = self.rows[self.index]
        path = Path(row["manual_mask_path"])
        FileManager.write_png(path, self.mask.astype(np.uint8) * 255)
        overlay = np.stack([self.image] * 3, axis=-1)
        overlay[..., 0] = np.maximum(overlay[..., 0], self.mask * 0.90)
        overlay[..., 1] *= 1.0 - 0.45 * self.mask
        overlay[..., 2] *= 1.0 - 0.45 * self.mask
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
            "save_manual",
            self.mask,
            self.index,
            len(self.rows),
        )
        qc = MaskManager.manual_qc(path)
        prefix = "Saved; valid for training." if qc["usable"] else f"Saved; excluded: {qc['reason']}."
        self.update_status(prefix)
        print("[EDITOR]", path, "|", prefix)

    def save_next(self) -> None:
        self.save()
        self.next()

    def accept_auto(self) -> None:
        ReviewManager.log(
            self.workspace,
            self.rows[self.index],
            self.review_round,
            "accept_auto",
            self.auto_mask,
            self.index,
            len(self.rows),
        )
        self.next()

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
        ReviewManager.log(
            self.workspace,
            row,
            self.review_round,
            "delete_manual",
            None,
            self.index,
            len(self.rows),
        )
        self.mask = self.auto_mask.copy()
        self.base_mask = self.auto_mask.copy()
        self._sync_python_mask()
        self.render()
        self.update_status("Manual mask deleted.")

    def next(self) -> None:
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.load_current()
        else:
            self.update_status("End of queue.")

    def previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def show(self):
        display(self.controls)
        return self


# %% [5] FEATURE BANK ȘI CLASIFICAREA LA NIVEL DE PACIENT
class FeatureDataset(Dataset):
    """Pregătește o singură dată imaginile și măștile necesare tuturor modurilor."""

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
    """EfficientNet-B0 fără capul de clasificare; greutățile rămân înghețate."""

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
                "Greutățile EfficientNet-B0 nu au putut fi încărcate. "
                "Activează Internet în Kaggle sau pune greutățile în cache-ul Torch. "
                f"Eroare originală: {type(error).__name__}: {error}"
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
    """Media pe imagini în fiecare serie, apoi media seriilor în fiecare pacient."""

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
                raise RuntimeError(f"Etichete inconsistente pentru pacientul {row['patient_id']}.")
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
            raise RuntimeError(f"Modul {mode} nu conține niciun pacient.")
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


class FeatureManager:
    """Extrage doar cele șapte reprezentări necesare și salvează vectori per pacient."""

    @staticmethod
    def _region_normalize(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Scalează percentil doar în interiorul măștii, apoi face zero în exterior."""

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
    def _feature_fingerprint(
        rows: Sequence[dict[str, Any]], workspace: Workspace
    ) -> str:
        prediction_summary = FileManager.read_json(workspace.prediction_summary, {}) or {}
        digest = hashlib.sha256()
        payload = {
            "schema": "simple-patient-feature-bank-v1",
            "modes": Settings.Classification.MODES,
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
            workspace.prediction_audit,
        ):
            if audit_path.is_file():
                digest.update(audit_path.name.encode("utf-8"))
                digest.update(FileManager.sha256_file(audit_path).encode("ascii"))
        accepted_tokens = {
            row["image_token"]
            for row in FileManager.read_csv(workspace.manual_audit)
            if row.get("status") == "ACCEPTED"
        }
        for token in sorted(accepted_tokens):
            path = workspace.manual_masks / f"{token}.png"
            if path.is_file():
                digest.update(token.encode("utf-8"))
                digest.update(FileManager.sha256_file(path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def load(workspace: Workspace) -> dict[str, dict[str, Any]]:
        if not workspace.feature_bank.is_file():
            raise FileNotFoundError(f"Lipsește feature bank-ul: {workspace.feature_bank}")
        archive = np.load(workspace.feature_bank, allow_pickle=False)
        bank = {}
        for mode in Settings.Classification.MODES:
            bank[mode] = {
                "X": archive[f"{mode}__X"],
                "y": archive[f"{mode}__y"],
                "patient_ids": archive[f"{mode}__patient_ids"].astype(str),
                "source_slices": int(archive[f"{mode}__source_slices"][0]),
                "series_proxies": int(archive[f"{mode}__series_proxies"][0]),
            }
        return bank

    @staticmethod
    def build(
        dataset_rows: Sequence[dict[str, Any]],
        workspace: Workspace,
        device: torch.device,
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:
        fingerprint = FeatureManager._feature_fingerprint(dataset_rows, workspace)
        old_metadata = FileManager.read_json(workspace.feature_metadata, {}) or {}
        if (
            not force
            and workspace.feature_bank.is_file()
            and old_metadata.get("fingerprint") == fingerprint
        ):
            try:
                bank = FeatureManager.load(workspace)
            except Exception as error:
                print(
                    "[FEATURE BANK] Cache-ul nu poate fi citit și va fi refăcut:",
                    f"{type(error).__name__}: {error}",
                )
            else:
                print("[FEATURE BANK] Cache compatibil reutilizat.")
                return bank

        quality = {
            row["image_token"]: row for row in FileManager.read_csv(workspace.quality_audit)
        }
        predictions = {
            row["image_token"]: row for row in FileManager.read_csv(workspace.prediction_audit)
        }
        accepted_manual = {
            row["image_token"]
            for row in FileManager.read_csv(workspace.manual_audit)
            if row.get("status") == "ACCEPTED"
        }
        if len(predictions) != len(dataset_rows):
            raise RuntimeError("Prediction audit nu acoperă întregul dataset.")

        rows = []
        for original in dataset_rows:
            row = dict(original)
            row.update(predictions.get(row["image_token"], {}))
            row.update(quality.get(row["image_token"], {}))
            row["keep_attention"] = int(
                _as_int(row.get("attention_valid_final"), 0) == 1
                and _as_int(row.get("quality_valid"), 0) == 1
            )
            row["keep_manual_matched"] = int(row["image_token"] in accepted_manual)
            rows.append(row)

        extractor = FrozenEfficientNet().to(device).eval()
        loader = DataLoader(
            FeatureDataset(rows),
            batch_size=RuntimeManager.feature_batch_size(device),
            shuffle=False,
            num_workers=Settings.Runtime.NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        pool = StreamingPatientPool(Settings.Classification.MODES)
        started = time.perf_counter()

        def encode(mode: str, images: torch.Tensor, selected_rows: list[dict[str, Any]]) -> None:
            if not len(selected_rows):
                return
            embeddings = extractor(images).detach().cpu().numpy().astype(np.float32)
            pool.add(mode, embeddings, selected_rows)

        with torch.inference_mode():
            for robust, raw, content, attention, manual, indices in tqdm(
                loader, desc="EfficientNet feature bank"
            ):
                batch_rows = [rows[int(index)] for index in indices]
                robust = robust.to(device, non_blocking=True)
                raw = raw.to(device, non_blocking=True)
                content = content.to(device, non_blocking=True)
                attention = attention.to(device, non_blocking=True)
                manual = manual.to(device, non_blocking=True)

                encode("FULL_IMAGE", robust, batch_rows)

                attention_positions = [
                    position for position, row in enumerate(batch_rows) if row["keep_attention"]
                ]
                if attention_positions:
                    positions = torch.as_tensor(attention_positions, device=device)
                    selected_raw = raw.index_select(0, positions)
                    selected_content = content.index_select(0, positions)
                    selected_mask = attention.index_select(0, positions)
                    support = FeatureManager._support(selected_mask, selected_content)
                    roi = FeatureManager._region_normalize(selected_raw, support)
                    complement = FeatureManager._region_normalize(
                        selected_raw, (selected_content > 0.5).float() * (1.0 - support)
                    )
                    selected_rows = [batch_rows[position] for position in attention_positions]
                    encode("AU1_ATTENTION_ROI", roi, selected_rows)
                    encode("AU5_ATTENTION_COMPLEMENT", complement, selected_rows)

                manual_positions = [
                    position for position, row in enumerate(batch_rows)
                    if row["keep_manual_matched"]
                ]
                if manual_positions:
                    positions = torch.as_tensor(manual_positions, device=device)
                    selected_raw = raw.index_select(0, positions)
                    selected_content = content.index_select(0, positions)
                    manual_mask = manual.index_select(0, positions)
                    attention_mask = attention.index_select(0, positions)
                    manual_support = FeatureManager._support(manual_mask, selected_content)
                    attention_support = FeatureManager._support(attention_mask, selected_content)
                    selected_rows = [batch_rows[position] for position in manual_positions]
                    encode(
                        "AU6_MANUAL_ROI",
                        FeatureManager._region_normalize(selected_raw, manual_support),
                        selected_rows,
                    )
                    encode(
                        "AU7_MANUAL_COMPLEMENT",
                        FeatureManager._region_normalize(
                            selected_raw,
                            (selected_content > 0.5).float() * (1.0 - manual_support),
                        ),
                        selected_rows,
                    )
                    encode(
                        "AU8_ATTENTION_MATCHED_MANUAL_ROI",
                        FeatureManager._region_normalize(selected_raw, attention_support),
                        selected_rows,
                    )
                    encode(
                        "AU9_ATTENTION_MATCHED_MANUAL_COMPLEMENT",
                        FeatureManager._region_normalize(
                            selected_raw,
                            (selected_content > 0.5).float() * (1.0 - attention_support),
                        ),
                        selected_rows,
                    )

        bank = {mode: pool.finalize(mode) for mode in Settings.Classification.MODES}
        arrays = {}
        metadata_modes = {}
        for mode, values in bank.items():
            arrays[f"{mode}__X"] = values["X"]
            arrays[f"{mode}__y"] = values["y"]
            arrays[f"{mode}__patient_ids"] = values["patient_ids"].astype("U")
            arrays[f"{mode}__source_slices"] = np.asarray([values["source_slices"]], dtype=np.int64)
            arrays[f"{mode}__series_proxies"] = np.asarray([values["series_proxies"]], dtype=np.int64)
            metadata_modes[mode] = {
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
            "device": device.type,
            "modes": metadata_modes,
            "elapsed": RuntimeManager.format_seconds(time.perf_counter() - started),
            "feature_bank": str(workspace.feature_bank),
        }
        FileManager.write_json(workspace.feature_metadata, metadata)
        print("[FEATURE BANK] Salvat:", workspace.feature_bank)
        del extractor
        RuntimeManager.release(device)
        return bank


class EvaluationManager:
    """Nested CV simplificat: toate transformările sunt învățate numai pe train."""

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
            raise RuntimeError("Sunt necesari cel puțin doi pacienți în fiecare clasă.")
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

    @staticmethod
    def evaluate(bank: dict[str, dict[str, Any]], workspace: Workspace) -> pd.DataFrame:
        fold_map = EvaluationManager._fold_map(bank["FULL_IMAGE"])
        summary_rows = []
        prediction_tables: dict[str, pd.DataFrame] = {}

        for mode in Settings.Classification.MODES:
            values = bank[mode]
            X = np.asarray(values["X"], dtype=np.float32)
            y = np.asarray(values["y"], dtype=np.int64)
            patients = np.asarray(values["patient_ids"]).astype(str)
            if len(np.unique(y)) < 2:
                raise RuntimeError(f"{mode}: lipsește una dintre clase.")
            unknown = sorted(set(patients) - set(fold_map))
            if unknown:
                raise RuntimeError(f"{mode}: pacienți necunoscuți în fold map: {unknown}")
            patient_folds = np.asarray([fold_map[patient] for patient in patients], dtype=np.int64)
            scores = np.full(len(y), np.nan, dtype=np.float64)
            predictions = np.full(len(y), -1, dtype=np.int64)
            selected_cs = np.full(len(y), np.nan, dtype=np.float64)
            thresholds = np.full(len(y), np.nan, dtype=np.float64)

            for fold in sorted(np.unique(patient_folds)):
                train_index = np.flatnonzero(patient_folds != fold)
                valid_index = np.flatnonzero(patient_folds == fold)
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
                raise RuntimeError(f"{mode}: predicții OOF incomplete.")
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
                    "patients": len(patients),
                    "source_slices": values["source_slices"],
                    "series_proxies": values["series_proxies"],
                    **metrics,
                }
            )
            print(
                f"[EVALUARE] {mode}: AUC={metrics['auc']:.3f} "
                f"AP={metrics['average_precision']:.3f} Brier={metrics['brier']:.3f}"
            )

        summary = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
        summary.to_csv(workspace.results_csv, index=False)

        comparisons = []
        for first_mode, second_mode, question in (
            ("AU1_ATTENTION_ROI", "AU5_ATTENTION_COMPLEMENT", "heart_roi_vs_outside"),
            ("AU6_MANUAL_ROI", "AU8_ATTENTION_MATCHED_MANUAL_ROI", "manual_vs_attention_same_images"),
            ("AU7_MANUAL_COMPLEMENT", "AU9_ATTENTION_MATCHED_MANUAL_COMPLEMENT", "manual_vs_attention_complement_same_images"),
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
        print("[EVALUARE] Rezultate:", workspace.results_csv)
        return summary


# %% [6] FAȚADA PIPELINE-ULUI
class CADPipeline:
    """API-ul simplu folosit în celulele de execuție ale notebook-ului."""

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        workspace_root: Path | str | None = None,
        device: str | None = None,
    ):
        RuntimeManager.seed_everything()
        self.dataset_path = Path(dataset_path or Settings.Paths.DATASET_PATH)
        self.workspace = FileManager.create_workspace(workspace_root)
        self.device = RuntimeManager.resolve_device(device)
        self.samples: list[Sample] | None = None
        self.dataset_rows: list[dict[str, Any]] | None = None
        print(f"[PIPELINE] device={self.device.type}")
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
    ) -> dict[int, Path]:
        accepted, manual_summary = self.audit_manual_masks(
            refresh_quality=refresh_quality,
            minimum_masks=minimum_masks,
        )
        checkpoints = SegmentationManager.train_crossfit(
            accepted, self.workspace, self.device, force=force
        )
        summary = FileManager.read_json(self.workspace.training_summary, {}) or {}
        summary["manual_mask_audit"] = manual_summary
        FileManager.write_json(self.workspace.training_summary, summary)
        return checkpoints

    def generate_attention_masks(self, force: bool = False) -> list[dict[str, Any]]:
        return SegmentationManager.predict_all(
            self._rows(),
            self.workspace,
            self.device,
            checkpoint_map=SegmentationManager.load_checkpoint_map(self.workspace),
            force=force,
        )

    def train_and_generate(
        self,
        force_training: bool = False,
        force_predictions: bool = False,
        minimum_masks: int | None = None,
    ) -> dict[str, Any]:
        checkpoints = self.train_attention(
            force=force_training, minimum_masks=minimum_masks
        )
        predictions = SegmentationManager.predict_all(
            self._rows(),
            self.workspace,
            self.device,
            checkpoint_map=checkpoints,
            force=force_predictions,
        )
        return {
            "checkpoints": {fold: str(path) for fold, path in checkpoints.items()},
            "predictions": len(predictions),
            "training_summary": str(self.workspace.training_summary),
            "prediction_summary": str(self.workspace.prediction_summary),
        }

    def open_editor(
        self,
        scope: str = "invalid",
        limit: int | None = None,
        start_index: int = 0,
        brush_radius: int | None = None,
        review_round: int = 1,
        seed: int = 42,
    ) -> MaskEditor:
        if not self.workspace.quality_audit.is_file():
            self.audit_quality(refresh=False)
        if not self.workspace.prediction_audit.is_file():
            raise FileNotFoundError("Rulează generate_attention_masks() înainte de editor.")
        queue = ReviewManager.select(
            self._rows(),
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

    def build_feature_bank(self, force: bool = False) -> dict[str, dict[str, Any]]:
        if not self.workspace.prediction_audit.is_file():
            raise FileNotFoundError("Rulează generate_attention_masks() înainte de feature bank.")
        if not self.workspace.manual_audit.is_file():
            self.audit_manual_masks()
        return FeatureManager.build(
            self._rows(), self.workspace, self.device, force=force
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
        status = {
            "device": self.device.type,
            "dataset": str(self.dataset_path),
            "workspace": str(self.workspace.root),
            "manual_masks": len(list(self.workspace.manual_masks.glob("*.png"))),
            "checkpoints": len(list(self.workspace.checkpoints.glob("attention_unet_fold_*.pt"))),
            "predicted_masks": len(list(self.workspace.predicted_masks.glob("*.png"))),
            "quality_audit": self.workspace.quality_audit.is_file(),
            "manual_audit": self.workspace.manual_audit.is_file(),
            "prediction_audit": self.workspace.prediction_audit.is_file(),
            "feature_bank": self.workspace.feature_bank.is_file(),
            "results": self.workspace.results_csv.is_file(),
        }
        print(json.dumps(status, indent=2))
        return status


RuntimeManager.seed_everything()
print("[PIPELINE] Clasele pipeline-ului simplificat au fost încărcate.")
print("[PIPELINE] Setează device='cpu', 'cuda' sau 'auto' la CADPipeline(...).")
