# CAD Cardiac MRI Multi-Experiment Suite — V3 Locked-Candidate Validation Revision

## Purpose

This revision preserves the detailed patient-level pipeline, the historical experiment suite, and the prior deconfounding work. It adds the specific validation experiments required by the latest Kaggle run, in which the strict standardized candidate

```text
A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA
```

was the strongest and most stable internal model, while several questions remained unresolved:

- whether A12 depends on the released `SR_*` / `series*` folder boundaries;
- whether repeated exact pixel exports inside one patient inflate the pooled representation;
- whether MONAI mask morphology, confidence, position, or bounding-box geometry alone predicts the class;
- whether A12 remains better than its principal controls on the **same repeated-CV split seeds**;
- whether the A12 signal survives a direct patient-label permutation test;
- which pHash candidates and series/view identities still require blinded manual review.

The validated computational patient definition remains unchanged:

```text
patient_id = Directory_*
```

`SR_*` and `series*` folders remain patient-scoped operational series proxies. They are not patients and are not claimed to be validated DICOM `SeriesInstanceUID` values.

---

## What was preserved

The revision keeps the existing pipeline architecture and detailed explanatory comments:

- pinned MONAI `ventricular_short_axis_3label` bundle, immutable revision, and SHA-256 verification;
- direct loading of the official TorchScript `model.ts`, with lazy MONAI import only for the exceptional reconstruction fallback;
- frozen ImageNet EfficientNet-B0 encoder;
- label-blind original and standardized preprocessing branches;
- `Directory_*` patient-level and duplicate-aware train/validation separation;
- hierarchical slice → series proxy → patient embedding pooling;
- legacy slice-level classifier only as a declared weak-supervision ablation;
- nested patient-level selection of classifier `C` and the operating threshold;
- fold-local scaler, PCA, Logistic Regression, Linear SVM calibration, and threshold selection;
- exact decoded-pixel duplicate audit and complete BK-tree pHash candidate search;
- original, standardized, border, padding, provenance, MONAI-QC, outside-region, and center-crop controls;
- fixed-content `240 in 256` geometry and separate min-max MONAI / robust-percentile EfficientNet inputs;
- reusable multi-view feature caching, machine-readable outputs, and mirrored console logging;
- historical B0 and standardized-development B1 results for transparent comparison.

No clinical sequence or view is inferred from folder names. No external validation result is fabricated when an independent cohort is unavailable.

---

## Locked primary candidate and development baseline

### Locked primary candidate

```text
A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA
```

A12 uses:

```text
label-blind native dark-padding removal
→ longest retained side fixed to 240 pixels in a 256×256 canvas
→ min-max image for MONAI
→ robust 1st/99th-percentile image for EfficientNet
→ valid MONAI mask: zero-background soft ROI
→ invalid MONAI mask: fixed 60% center crop
→ hierarchical embedding pooling
→ fold-local scaling + PCA
→ Logistic Regression
```

A12 is now the reference used by the final baseline-relative comparison tables.

### Standardized development baseline

```text
B1_STANDARDIZED_ROI_HIER_LR_PCA
```

B1 remains enabled and unchanged as the earlier standardized development baseline. It is not deleted or silently relabeled as the final model.

### Historical reference

```text
B0_ROI_HIER_LR_PCA
```

B0 remains the original-canvas historical reference.

---

## New V3 experiments

The default registry now contains **43 experiments**. Five experiments are new in V3.

| ID | Purpose |
|---|---|
| `A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA` | Uses exactly the A12 pixels but replaces folder-defined hierarchical pooling with deterministic, folder-independent fixed-size chunks. |
| `A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA` | Uses A12 after collapsing exact decoded-pixel repetitions inside each `Directory_*` patient. |
| `C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA` | EfficientNet sees only the standardized dilated soft MONAI probability map; MRI intensities are absent. |
| `C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA` | EfficientNet sees only the standardized dilated hard MONAI mask; soft confidence and MRI intensities are absent. |
| `C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA` | EfficientNet sees only a filled rectangle representing valid MONAI bounding-box location and extent. |

These experiments are matched controls. They do not change the patient definition, labels, folds, EfficientNet checkpoint, classifier family, or threshold-selection policy unless the experiment description explicitly says so.

---

## A14 — A12 with folder-independent pooling

A14 asks whether A12 benefits from the released folder organization rather than from its image representation.

For every patient:

```text
all retained slice embeddings
→ deterministic SHA-256-derived order independent of series_id and label
→ groups of at most FIXED_CHUNK_SIZE slices
→ mean embedding inside each chunk
→ equal mean across chunks
→ one patient embedding
```

Default:

```python
FIXED_CHUNK_SIZE = 20
```

A12 and A14 use identical frozen slice embeddings. Only the pooling rule changes.

The key comparison is:

```text
A14 fixed chunks → A12 hierarchical pooling
```

A positive repeated-CV delta means hierarchical A12 performed better on the same seed.

---

## A15 — exact within-patient deduplication

The exact duplicate audit previously showed many repeated pixel matrices within patients but no confirmed exact cross-patient duplicate groups. Such repetitions do not create train/validation leakage, but they can still make an identical exported frame contribute multiple times during pooling.

A15 applies a deterministic label-blind mask before weighting and pooling:

```text
(patient_id, decoded_pixel_sha256)
→ retain one canonical row
→ remove additional exact copies from that patient
```

The canonical row is selected by:

1. lexicographically smallest patient-scoped `series_id`;
2. smallest deterministic `sample_index`.

The process:

- never merges patients;
- never changes `patient_id = Directory_*`;
- never uses the class label, model score, fold, or OOF result;
- records the number of removed exact repetitions in experiment outputs.

The key comparison is:

```text
A12 original retained rows → A15 exact within-patient deduplication
```

A positive delta for A15 means deduplication improved the model; a negative delta means repeated exports were not the source of A12’s advantage or their removal discarded useful representation balance.

---

## C21–C23 — MONAI morphology-only controls

A high A12 score could theoretically arise from the MONAI mask itself—for example, systematic class differences in mask shape, position, size, or confidence—rather than from MRI intensities inside the ROI.

V3 therefore encodes three image controls through the same frozen EfficientNet and patient-level evaluation pipeline.

### C21: soft probability-map only

```text
valid mask → dilated P(heart), repeated to 3 channels
invalid mask → all-zero image
```

This preserves soft confidence, shape, and position but removes MRI intensity.

### C22: hard mask only

```text
valid mask → dilated binary hard mask, repeated to 3 channels
invalid mask → all-zero image
```

This removes MRI intensity and soft confidence while preserving binary morphology and position.

### C23: bounding-box geometry only

```text
valid mask → filled binary rectangle around the dilated hard mask
invalid or empty mask → all-zero image
```

This discards within-box morphology and tests only box position and extent.

The all-zero invalid-mask policy is intentional. A full-image fallback would invalidate a mask-only negative control by exposing the complete MRI.

---

## Direct same-seed repeated-CV comparisons

The prior suite reported repeated nested-CV distributions separately. V3 additionally calculates **paired deltas on exactly the same outer split seeds**.

Default stability settings:

```python
REPEATED_NESTED_CV_REPEATS = 50
```

The following experiments are rerun on the same 50 seeds:

```text
B1 standardized development baseline
A9 standardized full image
A10 zero-background ROI with full-image fallback
A11 MONAI bounding-box crop
C14 fixed center crop 60%
A12 locked primary candidate
A14 A12 fixed-chunk pooling
A15 A12 exact within-patient deduplication
C10 outside enlarged MONAI ventricular bounding box
C5 standardized border 5%
C6 standardized border 10%
C7 fixed-canvas padding-mask control
C21 soft MONAI mask only
C22 hard MONAI mask only
C23 MONAI bounding-box geometry only
```

Predeclared paired comparisons include:

```text
A12 versus fixed center 60%
A12 versus standardized full image
A12 versus outside enlarged MONAI ventricular box
A12 versus zero-background ROI with full-image fallback
A12 versus MONAI bounding-box crop
A12 hierarchical versus A14 fixed chunks
A12 versus A15 exact deduplication
A12 versus soft-mask-only
A12 versus hard-mask-only
A12 versus bounding-box-mask-only
```

For each common seed:

```text
delta AUC = comparison AUC − reference AUC
```

The suite saves:

```text
stability/repeated_nested_cv_paired_deltas.csv
stability/repeated_nested_cv_paired_comparisons.csv
stability/repeated_nested_cv_paired_comparisons.json
```

The summary reports:

- number of common seeds;
- mean and median delta AUROC;
- standard deviation;
- 2.5%, 25%, 75%, and 97.5% empirical quantiles;
- minimum and maximum delta;
- fraction of seeds favoring the comparison, reference, or producing a tie.

Repeated seeds are treated as a stability analysis, not as independent clinical subjects.

---

## Direct patient-label permutation for A12

The permutation target is now the locked A12 candidate rather than B1.

Default:

```python
LABEL_PERMUTATION_REPLICATES = 1000
PERMUTATION_EXPERIMENT_IDS = (
    "A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA",
)
```

For every replicate:

1. labels are permuted at the `Directory_*` patient level;
2. duplicate-component grouping is retained;
3. outer folds are rebuilt deterministically;
4. nested selection of `C` and the decision threshold is repeated;
5. one OOF AUROC is recorded.

Outputs:

```text
permutation/A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA/
    patient_label_permutation_auc.csv
    patient_label_permutation_summary.json

permutation/patient_label_permutation_summary.json
```

The permutation test is an association and implementation sanity check. It cannot distinguish genuine CAD-related anatomy from a protocol/export property that is genuinely correlated with the original labels.

---

## Complete pHash review support

The complete BK-tree pHash audit remains unchanged, but V3 now prepares manual-review material.

For every candidate patient pair, the suite creates a blinded side-by-side panel under:

```text
audits/phash_review_panels/
```

The panel title hides class labels and cross-label status. The candidate CSV retains those values for later merging and adds empty review fields:

```text
manual_review_status
manual_review_notes
review_panel_path
```

Recommended review labels:

```text
confirmed_near_duplicate
same_generic_localizer_or_template
similar_but_independent
false_positive
uncertain
```

A pHash candidate is not automatically treated as a confirmed duplicate. `GROUP_SPLITS_BY_PHASH_CANDIDATES` remains `False` unless review supports a stricter policy.

---

## Blinded sequence/view annotation template

The pipeline still refuses to infer clinical sequence identity from `SR_*` or `series*` folder names.

V3 writes:

```text
audits/series_annotation_template.csv
audits/series_annotation_label_key.csv
```

The blinded template includes one row per patient-scoped series proxy and representative first, middle, and last image paths. Empty fields are provided for:

```text
sequence_type
view_type
contains_heart
is_localizer
is_derived_image
monai_mask_validity
annotation_confidence
reviewer
notes
```

Class labels are written to the separate label-key file so sequence/view annotation can be performed without showing `Normal` / `Sick` status.

This step creates the infrastructure for later sequence-specific analysis. It does not pretend that automatic annotation has already occurred.

---

## Feature-bank changes

The suite now creates up to **21 EfficientNet feature modes**.

For approximately 63,425 slices:

```text
one 63,425 × 1,280 float32 matrix ≈ 310 MiB
21 matrices ≈ 6.35 GiB
```

Additional shared arrays, manifests, outputs, and filesystem overhead require extra space. The matrices are NumPy memory maps, so they are not all loaded into RAM simultaneously.

New cache schema:

```text
2026-09-01-primary-candidate-validation-v5
```

The first V3 run must rebuild the feature bank because mask-only embeddings are new. Later runs with identical feature-affecting settings reuse the cache.

`FEATURE_MODES_PER_ENCODER_CALL = 4` remains the default T4-oriented compromise. Reduce it to `2` or `1` if GPU memory is insufficient.

A MONAI-loading correction was also made: preliminary runs that select only A12 or only C21–C23 now correctly load the segmenter even when B1 is disabled.

---

## Main output package

```text
console_output.log
suite_configuration.json
suite_run_metadata.json
external_validation_status.json

manifests/
    cohort_manifest.csv
    patient_fold_manifest.csv

audits/
    exact_decoded_pixel_duplicate_groups.csv
    perceptual_near_duplicate_patient_pairs.csv
    phash_review_panels/*.png
    series_annotation_template.csv
    series_annotation_label_key.csv
    monai_qc_by_patient.csv
    standardized_monai_qc_by_patient.csv
    monai_gate_class_comparison.json
    standardized_monai_gate_class_comparison.json
    patient_provenance_features.csv
    patient_standardization_features.csv
    patient_n_slices_only.csv
    patient_n_series_only.csv
    patient_series_length_only.csv
    patient_native_geometry_only.csv
    patient_file_size_only.csv

experiments/<experiment_id>/
    experiment_config.json
    patient_oof_predictions.csv
    fold_metrics.csv
    summary.json
    failure.json                  # only when the experiment fails

comparison/
    experiment_summary.csv
    patient_predictions_all_experiments.csv
    paired_auc_comparisons.csv
    paired_primary_ablation_comparisons.csv
    failed_experiments.csv
    final_report.json

stability/<experiment_id>/
    repeated_nested_cv_runs.csv
    repeated_nested_cv_oof_predictions.csv
    patient_score_stability.csv
    repeated_nested_cv_summary.json

stability/
    repeated_nested_cv_summary.json
    repeated_nested_cv_paired_deltas.csv
    repeated_nested_cv_paired_comparisons.csv
    repeated_nested_cv_paired_comparisons.json

permutation/<experiment_id>/
    patient_label_permutation_auc.csv
    patient_label_permutation_summary.json

permutation/
    patient_label_permutation_summary.json
```

All ordinary `print()` messages, progress output, warnings, and tracebacks are mirrored to `console_output.log`.

---

## Running the suite

```bash
python cad_mri_multi_experiment_suite_final_improved_v3.py
```

The default runs all 43 enabled experiments, 15 repeated-CV stability analyses, 10 paired repeated-CV comparisons, and 1,000 A12 label permutations.

For a focused publication-candidate run, `EXPERIMENTS_TO_RUN` can be restricted while retaining A12 and the required controls, for example:

```python
EXPERIMENTS_TO_RUN = (
    "A9_STANDARDIZED_FULL_HIER_LR_PCA",
    "A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA",
    "A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA",
    "C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA",
    "A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA",
    "A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA",
    "A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA",
    "C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA",
    "C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA",
    "C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA",
    "C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA",
)
```

When restricting the experiment list, corresponding stability/comparison IDs must remain internally consistent. `validate_configuration()` fails early when a requested repeated-CV or permutation experiment is not enabled.

---

## How to interpret the new results

### A12 versus A14

- Similar performance suggests that A12 does not depend strongly on released folder-proxy boundaries.
- Lower A14 performance suggests that series-proxy structure carries useful information, but that information may be clinical sequence structure or export provenance; sequence annotation is needed.

### A12 versus A15

- Similar performance suggests that repeated exact exports do not drive A12.
- A large A15 decrease means duplicate repetitions materially affect the pooled patient representation and must be disclosed.
- A15 improvement suggests that repeated exports were adding noise or overweighting.

### A12 versus C21–C23

- Mask-only controls near chance while A12 remains high support the claim that MRI intensities contribute beyond MONAI morphology.
- A high C21 score suggests soft confidence/shape may encode class or protocol.
- A high C22 score suggests hard-mask morphology or position may encode class.
- A high C23 score suggests bounding-box geometry alone may encode class or acquisition style.

### Paired repeated-CV deltas

The most important result is not merely the median AUC of two experiments. It is whether the same-seed delta consistently favors A12 across the repeated splits.

### Permutation

A low empirical p-value supports a non-random association with released labels, but not anatomical specificity or external generalization.

---

## Remaining limitations

Even after V3, the following statements are not justified without further evidence:

- that A12 detects CAD-specific pathology rather than a correlated sequence/protocol distribution;
- that `Directory_*` performance generalizes to independent clinical patients;
- that MONAI masks are anatomically correct on all heterogeneous views;
- that pHash candidates are harmless before manual review;
- that series proxies correspond to true DICOM acquisition series;
- that A12 probabilities are clinically calibrated;
- that a fixed operating threshold transfers to another hospital.

The decisive next step remains a **locked external zero-shot validation** on an independent cohort with a comparable CAD endpoint. Local recalibration, when needed, must be reported only after the zero-shot result and evaluated on a separate local test subset.

---

## Validation performed before delivery

The delivered file was checked through:

- Python compilation;
- complete import;
- `validate_configuration()` across all 43 experiments;
- validation of all 21 image feature modes;
- mask-only helper shape and invalid-mask tests;
- deterministic exact within-patient deduplication tests;
- A14 fixed-chunk preparation tests;
- blinded pHash review-panel generation;
- blinded series annotation and separate label-key generation;
- repeated nested-CV paired-comparison tests;
- direct A12 permutation-path tests;
- a reduced integrated `main()` smoke test covering manifests, the new experiments, repeated stability, same-seed paired comparisons, A12 permutation, final reports, and required output files.

The full real CAD dataset was not rerun during delivery. Actual Kaggle runtime, disk usage, medical metrics, manual review outcomes, and external validation remain results of the user’s execution environment.
