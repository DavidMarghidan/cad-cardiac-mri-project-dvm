# Comparison and Decisions for the Final Version

## Files Compared

1. `Pasted markdown(20260831-061030).md` — the attached version, containing very detailed documentation and comments, together with an initial experiment matrix.
2. `cad_mri_multi_experiment_suite.py` — the previously generated suite, with an experiment-registry structure, nested cross-validation, confounding controls, and centralized reports.
3. `cad_mri_multi_experiment_suite_final.py` — the final version produced by combining the two.

## What Was Preserved from the Attached File

The terminology, organization, and detailed explanations were preserved for:

- `patient_id = Directory_*`;
- the interpretation of `SR_*` / `series*` folders as operational series proxies rather than validated DICOM UIDs;
- the separation of MONAI preprocessing from EfficientNet preprocessing;
- zero-padding to 256×256 and alignment with the 224×224 EfficientNet input;
- loading the MONAI TorchScript model with SHA-256 verification, with MONAI imported only for the fallback path;
- the MONAI confidence gate and full-image fallback;
- the frozen EfficientNet-B0 encoder and removal of the ImageNet classification head;
- hierarchical pooling from slice → series proxy → patient;
- the explicit limitation of the legacy strategy that repeats patient labels at slice level;
- the detailed explanations in `main()`, the `print()` messages, and stage-duration measurements;
- warnings regarding the small cohort, the MONAI domain limitation, and the absence of external clinical calibration.

## Main Methodological Corrections

### 1. The Fixed 0.50 Threshold Was Removed from the Primary Evaluation

The attached file calculated sensitivity, specificity, PPV, NPV, and F1 at a fixed threshold of 0.50. The final version selects the threshold exclusively from the inner-OOF predictions generated on the outer-training data. The outer-validation data are never used to choose the threshold.

### 2. `C` Is Selected Through Nested Patient-Level Cross-Validation

For Logistic Regression and Linear SVM, the values in `CLASSIFIER_C_GRID` are compared inside the inner cross-validation loop. Selection is performed separately within each outer fold, without access to that fold’s held-out validation patients.

### 3. The SVM No Longer Uses the Arbitrary Transformation `sigmoid(decision_function)`

The attached file directly transformed SVM margins using the sigmoid function. The final version learns a sigmoid calibrator from inner-OOF scores belonging only to the training patients, then applies that calibrator to the outer-validation scores. The calibrator does not use class balancing, because doing so would artificially impose a 50/50 prevalence.

### 4. Folds Are Constructed After the Duplicate Audit

The exact-duplicate audit is no longer merely informative. Patients connected by exact duplicates are assigned to the same duplicate component and retained in the same fold. `patient_id` always remains `Directory_*`.

### 5. A pHash-Based Near-Duplicate Audit Was Added

pHash pairs are saved as potential near-duplicates. By default, they do not alter the folds because they must first be reviewed before being treated as confirmed duplicates.

### 6. Full-Image and MONAI-ROI Variants Use the Same Dataset Pass

A single feature-bank pass generates and stores embeddings for:

- MONAI ROI with confidence gating;
- the full image;
- border-only input;
- outside-MONAI-mask input.

MONAI is executed only once per batch, and the same images, indices, and metadata are used across all variants.

### 7. Explicit Shortcut and Confounding Controls Were Added

The final version includes:

- a border-only classifier;
- an outside-MONAI-mask classifier;
- an export/provenance-only classifier;
- a MONAI-QC-only classifier.

The provenance classifier uses only features that are predominantly structural or export-related: image dimensions, aspect ratio, numbers of images and series, file size, bytes per pixel, padding, and border statistics. Central texture features, entropy, edge density, and sharpness remain in the descriptive manifest but are excluded from this classifier because they may also contain genuine anatomical signal.

### 8. The Fusion Ablation Is Now Clean at Both Levels

In the attached file, `mean_probability` was applied at the slice → series level, while the series → patient level still used log-odds. The final version applies the declared fusion rule consistently at both levels, so A3 versus A4 is a genuine comparison between mean-probability fusion and log-odds fusion.

### 9. The Legacy Strategy Has a Matched Comparator

The experiment `A8_ROI_HIER_LR_NO_PCA_FIXEDC` was added. It uses:

- the same MONAI-ROI embeddings;
- equal weights;
- Logistic Regression;
- no PCA;
- fixed `C=1.0`.

Therefore, A8 versus A3 more cleanly isolates the difference between patient-embedding training and repeated-label slice-classifier training, without simultaneously changing PCA or the method used to select `C`.

### 10. Slice-Dropout Robustness Was Preserved and Corrected

Variants with 10%, 25%, and 50% slice dropout were included. Slice selection is:

- deterministic;
- label-blind;
- performed separately within each series proxy;
- independent of array order and Python hash randomization;
- constrained to retain at least one slice in every series.

### 11. Comparisons Are Paired

In addition to ranking all experiments against the baseline, the final version produces a separate file containing the predeclared matched comparisons. `Delta-AUROC` is calculated through bootstrap resampling of the same patients simultaneously for both models.

### 12. Results and Errors Are Fully Saved

All `print()` messages, progress information, and tracebacks are copied to `console_output.log`. An optional experiment may fail without deleting the results of the remaining experiments; the error is saved in `failure.json` and `comparison/failed_experiments.csv`.

## Experiments Executed by Default

1. `B0_ROI_HIER_LR_PCA`
2. `A1_FULL_HIER_LR_PCA`
3. `C1_BORDER_ONLY_HIER_LR_PCA`
4. `C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA`
5. `C3_EXPORT_PROVENANCE_ONLY_LR`
6. `C4_MONAI_QC_ONLY_LR`
7. `A2_ROI_FLAT_LR_PCA`
8. `A3_ROI_LEGACY_LOGODDS_LR`
9. `A4_ROI_LEGACY_MEANPROB_LR`
10. `A5_ROI_HIER_LINEAR_SVM_PCA`
11. `A6_ROI_HIER_LR_QUALITY_PCA`
12. `A7_ROI_HIER_LR_NO_PCA`
13. `A8_ROI_HIER_LR_NO_PCA_FIXEDC`
14. `R1_ROI_HIER_LR_PCA_DROP10`
15. `R2_ROI_HIER_LR_PCA_DROP25`
16. `R3_ROI_HIER_LR_PCA_DROP50`

## Main Output Files

- `console_output.log`
- `suite_configuration.json`
- `suite_run_metadata.json`
- `manifests/cohort_manifest.csv`
- `manifests/patient_fold_manifest.csv`
- `audits/exact_decoded_pixel_duplicate_groups.csv`
- `audits/perceptual_near_duplicate_patient_pairs.csv`
- `audits/monai_qc_by_patient.csv`
- `audits/monai_gate_class_comparison.json`
- `audits/patient_provenance_features.csv`
- `experiments/<experiment_id>/patient_oof_predictions.csv`
- `experiments/<experiment_id>/fold_metrics.csv`
- `experiments/<experiment_id>/summary.json`
- `comparison/experiment_summary.csv`
- `comparison/patient_predictions_all_experiments.csv`
- `comparison/paired_auc_comparisons.csv`
- `comparison/paired_primary_ablation_comparisons.csv`
- `comparison/failed_experiments.csv`
- `comparison/final_report.json`

## Validation Performed on the Final Version

- compilation with `python -m py_compile`;
- complete import and execution of `validate_configuration()`;
- execution of all 16 experiments on synthetic embeddings;
- verification of nested cross-validation, Logistic Regression, Linear SVM, and training-only calibration;
- verification that slice-level probabilities are reused between the two fusion methods;
- verification of duplicate-aware folds;
- verification of deterministic slice dropout and retention of every series;
- complete execution of `main()` on a synthetic JPEG dataset using mocked MONAI and EfficientNet models, including the feature bank, audits, all experiments, final output files, and logged console output.

The pipeline was not executed end-to-end in this environment on the real CAD dataset, and the real MONAI and EfficientNet checkpoints were not downloaded or executed here. The first real run should be monitored through `console_output.log`, and medical results should be interpreted only after reviewing the duplicate, provenance, and MONAI-gate audits.

## Execution

```bash
python cad_mri_multi_experiment_suite_final.py
```

For a smaller preliminary run, modify `EXPERIMENTS_TO_RUN`, while keeping `B0_ROI_HIER_LR_PCA` in the list.
