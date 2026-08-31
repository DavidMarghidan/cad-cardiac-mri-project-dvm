# CAD Cardiac MRI Multi-Experiment Suite — Deconfounding Revision

## Purpose

This revision preserves the detailed original patient-level pipeline and its original sixteen experiments, while adding the most important corrections suggested by the first real Kaggle run. That run produced a strong original MONAI-ROI result but also high AUC values for border-only, outside-mask, and provenance-only controls. The new revision therefore prioritizes a stricter scientific question:

> Does patient-level performance remain after reducing export, padding, border, and crop shortcuts, and does a cardiac-localized representation remain stronger than non-anatomical controls?

The validated computational patient definition remains unchanged:

```text
patient_id = Directory_*
```

`SR_*` and `series*` folders remain patient-scoped series proxies, not patients and not validated DICOM `SeriesInstanceUID` values.

## What was preserved

The revision keeps the original detailed comments and the established core methods:

- pinned MONAI `ventricular_short_axis_3label` bundle and hash verification;
- lazy MONAI import only for the exceptional `model.pt` fallback;
- frozen ImageNet EfficientNet-B0 feature encoder;
- patient-level and duplicate-aware train/validation separation;
- hierarchical slice → series proxy → patient embedding pooling;
- legacy slice-level classifier only as an ablation;
- nested patient-level selection of classifier `C` and the decision threshold;
- fold-local PCA, scaling, Logistic Regression, SVM calibration, and paired bootstrap comparisons;
- exact decoded-pixel duplicate audit, machine-readable outputs, console logging, and feature caching.

## New primary baseline

The primary baseline is now:

```text
B1_STANDARDIZED_ROI_HIER_LR_PCA
```

Its label-blind preprocessing is:

```text
native grayscale JPEG
  → detect only consecutive edge rows/columns that are dark and nearly uniform
  → enforce conservative crop limits
  → remove only the accepted native padding
  → robust 1st/99th-percentile intensity scaling
  → aspect-ratio-preserving 256×256 canvas
  → MONAI segmentation and plausibility gate
  → confidence-gated soft ROI / standardized full-image fallback
  → frozen EfficientNet-B0
  → hierarchical patient embedding
  → fold-local PCA + Logistic Regression
```

The original baseline is retained as:

```text
B0_ROI_HIER_LR_PCA
```

with role `historical_baseline`, so the effect of label-blind standardization is measured rather than hidden.

## New deconfounding experiments

The original sixteen experiments are retained. Twelve new experiments are added:

| ID | Purpose |
|---|---|
| `B1_STANDARDIZED_ROI_HIER_LR_PCA` | New standardized primary baseline |
| `A9_STANDARDIZED_FULL_HIER_LR_PCA` | Standardized full-image ablation |
| `C5_STANDARDIZED_BORDER05_HIER_LR_PCA` | Outer 5% standardized border only |
| `C6_STANDARDIZED_BORDER10_HIER_LR_PCA` | Outer 10% standardized border only |
| `C7_DETECTED_PADDING_MASK_HIER_LR_PCA` | Binary native/canvas padding geometry only |
| `C8_STANDARDIZED_CORNERS_HIER_LR_PCA` | Four standardized corners only |
| `C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA` | MONAI-area-matched central-crop localization control |
| `A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA` | Standardized ROI with zero retained background |
| `A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA` | Aspect-preserving MONAI bounding-box crop |
| `C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA` | Signal outside a larger MONAI box |
| `C11_STANDARDIZATION_QC_ONLY_LR` | Crop/padding/robust-range QC only |
| `C12_STANDARDIZED_MONAI_QC_ONLY_LR` | MONAI gate/QC after standardization only |

The center-crop control is marked as a localization control rather than a negative control, because it can contain genuine cardiac anatomy.

## Additional methodological improvements

### Conservative `C` selection

The suite no longer automatically selects the largest `C` after a negligible inner-AUC improvement. It selects the smallest `C` whose inner AUC is within:

```python
C_SELECTION_AUC_TOLERANCE = 0.01
```

of the best candidate. The best candidate AUC, selected candidate AUC, tolerance, and complete candidate table are saved in each fold output.

### Complete pHash candidate search

The earlier pHash screening could skip large buckets and truncate after a pair cap. The revised audit uses a complete BK-tree radius search over unique 64-bit pHash values. It reports progress, does not truncate, and aggregates evidence by patient pair without materializing every image pair.

A pHash match remains a screening candidate and must not be described as a confirmed duplicate without manual or stronger similarity review.

### Repeated nested patient-level CV

The new primary baseline is rerun across multiple deterministic outer split seeds:

```python
REPEATED_NESTED_CV_REPEATS = 10
```

The suite reports the complete AUC distribution and patient-level score variability. It never selects the most favorable split.

### Patient-label permutation test

The complete patient-level nested fitting path is repeated after permuting labels at the `Directory_*` level:

```python
LABEL_PERMUTATION_REPLICATES = 200
```

This produces an empirical null AUC distribution and one-sided empirical p-value. It is a leakage/signal sanity check, not a substitute for external validation.

### Aspect-preserving ROI crops

MONAI bounding-box and center-crop views are square-padded before resizing. Rectangular crops are not stretched directly to 224×224, avoiding an additional geometry artifact.

### Efficient feature-bank execution

The feature bank now contains fourteen image representations. To reduce Python and CUDA launch overhead without holding all views simultaneously, a conservative number of modes are concatenated per EfficientNet call:

```python
FEATURE_MODES_PER_ENCODER_CALL = 4
```

The value can be reduced if GPU memory is insufficient.

## Cache behavior

The extraction semantics changed, so the feature-cache schema is now:

```text
2026-08-31-deconfounding-v3
```

The first run of this revision must build a new cache. It does not silently reuse the old four-view feature bank. Later runs with identical files, software, and feature-affecting settings reuse the new cache.

The fourteen float32 embedding matrices require substantially more disk space than the original four-view suite. They are stored as memory-mapped `.npy` arrays rather than being held together in RAM.

## Main new outputs

In addition to all previous outputs, the revision writes:

```text
audits/patient_standardization_features.csv
audits/standardization_class_summary.csv
audits/standardized_monai_qc_by_patient.csv
audits/standardized_monai_gate_class_comparison.json
audits/standardized_monai_qc_class_summary.csv

stability/repeated_nested_cv_runs.csv
stability/repeated_nested_cv_oof_predictions.csv
stability/patient_score_stability.csv
stability/repeated_nested_cv_summary.json

permutation/patient_label_permutation_auc.csv
permutation/patient_label_permutation_summary.json
```

The final ranking uses `B1_STANDARDIZED_ROI_HIER_LR_PCA` as the paired-comparison baseline. The original B0 model remains visible as a historical reference.

## Running the suite

```bash
python cad_mri_multi_experiment_suite_final_improved.py
```

For a preliminary run, set `EXPERIMENTS_TO_RUN` to a tuple of experiment IDs. Keep the new baseline enabled whenever paired comparisons, repeated CV, or permutation testing are requested:

```python
EXPERIMENTS_TO_RUN = (
    "B0_ROI_HIER_LR_PCA",
    "B1_STANDARDIZED_ROI_HIER_LR_PCA",
    "A9_STANDARDIZED_FULL_HIER_LR_PCA",
    "C5_STANDARDIZED_BORDER05_HIER_LR_PCA",
    "C6_STANDARDIZED_BORDER10_HIER_LR_PCA",
    "C7_DETECTED_PADDING_MASK_HIER_LR_PCA",
    "C8_STANDARDIZED_CORNERS_HIER_LR_PCA",
    "C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA",
    "A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA",
    "A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA",
    "C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA",
    "C11_STANDARDIZATION_QC_ONLY_LR",
    "C12_STANDARDIZED_MONAI_QC_ONLY_LR",
)
```

## Interpretation target

A scientifically stronger outcome is not merely a higher primary AUC. The desired pattern is:

```text
standardized ROI remains useful
standardized narrow-border/corner/padding controls approach chance
standardization/provenance-only controls weaken
standardized ROI exceeds area-matched center crop and outside-box controls
repeated-CV AUC is reasonably stable
permuted-label AUC is centered near 0.5
```

If a control remains highly predictive, that is a substantive dataset finding and should be reported rather than hidden.

## Validation performed before delivery

The delivered file was checked through:

- Python bytecode compilation;
- import and complete configuration validation;
- serialization of the full 28-experiment protocol;
- unit checks for conservative dark-padding detection and crop safety;
- shape checks for aspect-preserving ROI crop helpers;
- a synthetic multi-view feature-bank run covering all fourteen modes;
- a synthetic repeated nested-CV run;
- a synthetic patient-label permutation run;
- a representative synthetic full-suite orchestration covering patient embeddings, tabular controls, legacy fusion/cache reuse, Linear SVM, quality weighting, PCA-off, standardized controls, slice dropout, paired comparison files, stability, permutation, metadata, and console logging.

The real CAD dataset and real MONAI/EfficientNet checkpoints were not rerun in this delivery environment. Real medical performance, GPU memory, disk use, and execution duration therefore remain to be measured on Kaggle.

## Remaining non-automated requirements

The file does not fabricate evidence that cannot be recovered from the JPEG release:

- sequence/view-specific analysis still requires reliable blinded annotation;
- MONAI anatomical validity still requires manual clinical review on a sampled subset;
- pHash candidates still require confirmation;
- a cardiac-MRI-pretrained encoder requires a compatible sequence/frame contract;
- true external validation requires an independent hospital cohort with a comparable CAD endpoint and a locked adapter.
