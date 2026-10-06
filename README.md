# Cardiac MRI CAD: leakage-controlled, resumable research pipeline

**Author and implementation:** David Vlad Marghidan  
**Scientific supervision:** Prof. Dr. Smaranda Belciug, INFusion Artificial Intelligence Research Laboratory, University of Craiova  
**Reference release:** 5 October 2026


## Documentation map

| Document | Best for | Link |
|---|---|---|
| One-page abstract poster | quick review or research-supplement upload | [`docs/research-supplement/`](docs/research-supplement/) |
| Six-page research overview | admissions officers and interdisciplinary readers | [`docs/project-overview/`](docs/project-overview/) |
| Contribution statement | authorship, supervision, and AI-use boundaries | [`docs/research-supplement/`](docs/research-supplement/) |
| Current technical pipeline summary | technical faculty and reproducibility review | [`docs/technical/`](docs/technical/) |
| Full preprint v1.2 | complete methods, results, declarations, and references | [`docs/manuscript/`](docs/manuscript/) |
| Documentation index | complete reading map | [`docs/README.md`](docs/README.md) |

The new overview and technical summary are synchronized to the complete 5 October 2026 execution. The earlier 3 October extended visual guide is not used as the release-specific source of current numerical results.

## Scope

This repository contains a research pipeline for a secondary analysis of the public **CAD Cardiac MRI Dataset**. The software asks a methodological question: **is label-related information more concentrated in an automatically localized cardiac region than in the complete image or in the pixels outside that region?**

The project is a proof of concept. It predicts the dataset's supplied `Normal` / `Sick` labels; it does not independently reconstruct a clinical diagnosis, identify coronary plaque, or provide patient-care advice. It is **not a medical device**, has not undergone external or clinical validation, and must not be used for medical decisions.

The result-associated reference execution described by the accompanying preprint used 63,425 JPEG images from 30 top-level patient folders (16 `Normal`, 14 `Sick`), 2,859 folder-defined series proxies, and 5,097 accepted segmentation targets. The independent evaluation unit is the **patient**, not the image.

## Why the pipeline is patient-level

Thousands of neighbouring MRI frames from one person are correlated. An image-level random split could place slices from the same person in both training and validation, allowing a model to exploit patient, scanner, sequence, framing, or export characteristics. This implementation therefore keeps every image from one `Directory_*` patient folder inside one fold throughout segmentation and classification.

The final classifier receives one representation per patient. Many images improve the description of that patient; they do not become thousands of independent votes.

## Three restart-safe Kaggle stages

| Stage | Accelerator | Main operations | Persistent products |
|---|---|---|---|
| **Initial CPU** | CPU / no accelerator | dataset discovery, manifest, preprocessing, image-quality audit, manual-target audit | CSV/JSON audits and stable metadata |
| **GPU** | CUDA | five cross-fitted 2.5D Attention U-Nets, threshold calibration, out-of-fold inference, mask QC | one checkpoint and one prediction part per fold, automatic mask PNGs |
| **Final CPU** | CPU / no accelerator | matching, eleven EfficientNet representations, series/patient pooling, nested evaluation | matching manifest, NPZ feature bank, result tables, patient-level predictions |

The default shared workspace is:

```text
/kaggle/working/cad_attention_unet_workspace
```

A browser or kernel interruption does not require a complete restart. Atomic file replacement and semantic fingerprints determine whether each artifact is complete and compatible with its current inputs.

## Technical workflow

### 1. Dataset discovery and stable manifest

The expected release structure is:

```text
CAD Cardiac MRI Dataset/
├── Normal/
│   └── Directory_*/...
└── Sick/
    └── Directory_*/...
```

Each `Directory_*` folder is treated as one patient. Nested folders are treated as **series proxies** because the JPEG release does not include verified DICOM series identifiers. PNG, JPG, and JPEG files are accepted.

For every image, `DataStage` records the class label, patient ID, series proxy, sequence position, previous/current/next paths, and a stable `image_token` derived from the path beginning at `Directory_*`. The token remains stable when Kaggle mounts the same dataset under a different root.

### 2. Preprocessing and image-quality audit

Images are read in grayscale. The preprocessing path:

1. conservatively removes near-black, low-variance scanner padding;
2. retains at least 60% of each original dimension and removes at most 20% from one side;
3. clips intensities to the 1st and 99th percentiles and scales them to `[0, 1]`;
4. preserves aspect ratio, makes the long side 240 pixels, and centres the image on a `256 x 256` canvas;
5. creates `224 x 224` classifier views and a content mask that separates visible image content from added canvas padding.

The quality audit measures sharpness, estimated noise relative to dynamic range, dynamic range, and a perceptual hash. Patient-adaptive tails are combined with fixed safeguards. These metrics are technical quality indicators, not clinical judgements.

### 3. Manual segmentation targets

The HTML review workflow supports three semantic target types:

| Target | Meaning | Training role |
|---|---|---|
| `HEART_PRESENT` | the general heart region is visible and a non-trivial mask exists | positive mask and presence target |
| `NO_HEART_VISIBLE` | the image is usable but the correct mask is empty | explicit negative target |
| `UNUSABLE` | blur, noise, localizer, or another unsuitable frame | excluded from segmentation training |

A non-empty legacy mask is not confused with an explicit no-heart decision. The normal integrity gates require at least 800 accepted positive masks, positive coverage in all five segmentation folds, and positive masks from at least six patients. These gates check consistency; they do not prove anatomical correctness.

### 4. Five-fold 2.5D Attention U-Net

The segmentation model receives three channels: the previous, current, and next image in the same folder-defined sequence. This **2.5D** input supplies local context without claiming a reconstructed 3D volume.

The network is an Attention U-Net with encoder widths `24 -> 48 -> 96 -> 192 -> 384`, gated skip connections, a `256 x 256` segmentation output, and an auxiliary heart-presence head. The `Normal` / `Sick` classification label is not supplied to this model.

Training uses:

- focal loss and Tversky overlap loss, with a stronger penalty on false-positive mask area;
- an auxiliary binary cross-entropy loss for heart presence;
- patient-, series-, and duplicate-aware weighted sampling;
- small geometric and intensity augmentations;
- AdamW, learning-rate reduction on a plateau, early stopping, and at most 36 epochs per fold;
- CUDA automatic mixed precision when the GPU stage is selected.

For fold `k`, every patient assigned to fold `k` is excluded from fitting and calibration. The resulting model predicts only that held-out fold. Joining the five held-out parts creates an **out-of-fold (OOF)** mask set for the complete dataset.

### 5. Mask calibration, plausibility checks, and uncertainty

Mask and presence thresholds are calibrated only on non-test validation patients. A probability map is thresholded, morphologically closed, and reduced to its largest connected component. Plausibility checks consider mask area, peak probability, image-boundary contact, and deviation from geometry learned from non-test masks.

If an initial mask is implausible, nearby thresholds are tested under the same fold-specific prior. Uncertainty combines entropy, test-time contrast disagreement, presence ambiguity, geometry deviation, and sequence inconsistency. Automatic masks and their metadata remain traceable to the checkpoint that created them.

### 6. Full-image, ROI, and complement controls

The central shortcut-learning test compares representations created from aligned source images:

- **full image**: all visible image content;
- **ROI**: the automatically or manually defined cardiac support;
- **complement**: visible content outside that support.

The primary `B2 / AU1 / C1` comparison uses exactly the same attention-eligible slices. A secondary Sick-Normal matching stage creates a more comparable cohort using appearance, mask geometry, sequence position, and image-quality descriptors. Because class labels are used to form cross-class pairs, matched modes are sensitivity analyses rather than independent prospective estimates.

### 7. Eleven frozen EfficientNet-B0 representations

`FeatureStage` builds the following fixed research modes:

| Mode | Cohort and representation |
|---|---|
| `B0_FULL_IMAGE` | full-image baseline on all released slices |
| `B2_ATTENTION_ELIGIBLE_FULL_IMAGE` | full image on exactly the AU1/C1 attention-eligible slices |
| `AU1_ATTENTION_ROI` | automatic cardiac ROI on the attention-eligible slices |
| `C1_ATTENTION_COMPLEMENT` | visible-content complement of AU1 on those same slices |
| `B1_MATCHED_FULL_IMAGE` | full image on the cross-class matched cohort |
| `AU2_MATCHED_ATTENTION_ROI` | automatic cardiac ROI on the matched cohort |
| `C2_MATCHED_ATTENTION_COMPLEMENT` | automatic-ROI complement on the matched cohort |
| `M1_MANUAL_ROI` | accepted manual ROI on the manual-positive subset |
| `C3_MANUAL_COMPLEMENT` | complement of the accepted manual ROI |
| `AU3_ATTENTION_ROI_MANUAL_SUBSET` | automatic ROI on the same images used by M1 |
| `C4_ATTENTION_COMPLEMENT_MANUAL_SUBSET` | automatic complement on the same images used by C3 |

A frozen `EfficientNet-B0` with `IMAGENET1K_V1` weights converts each view into a 1,280-dimensional embedding. No EfficientNet parameter is updated with CAD labels. The result-associated protocol runs EfficientNet on CPU in float32 with batch size 4 and forward-batch size 4.

### 8. Hierarchical pooling

Slice embeddings are averaged in two steps:

```text
slice embeddings -> mean within each series proxy -> mean across series -> one patient vector
```

This prevents a patient with a very long series from dominating solely through image count.

### 9. Nested patient-level evaluation

All final predictions are out of fold at patient level. For each outer split, model selection is performed only on the outer training patients:

```text
StandardScaler -> PCA retaining 95% variance -> class-balanced logistic regression
```

Inner cross-validation selects `C` from `{0.01, 0.1, 1, 10}` and chooses a probability threshold using the Youden index. Reported outputs include ROC AUC, average precision, Brier score, accuracy, sensitivity, specificity, and a 95% patient-bootstrap AUC interval. Aligned representations are compared with paired patient-level bootstrap differences using 2,000 resamples.

### 10. Persistence and cache validation

The pipeline does not trust a filename merely because it exists. Reuse requires compatible schemas, complete expected tokens, required companion files, and semantic fingerprints of the relevant settings and upstream inputs.

Examples:

- the quality cache uses stable dataset identity rather than volatile Kaggle mount times;
- manual-state fingerprints include semantic label, annotation source, sample weight, quality state, and mask bytes;
- checkpoints bind model settings, patient splits, target state, and training metadata;
- OOF prediction parts bind checkpoint hashes, expected fold images, prediction settings, and complete mask files;
- matching, feature, and evaluation caches validate their upstream lineage and cohort structure.

PNG, CSV, JSON, PyTorch, and NPZ artifacts are written through temporary files and `os.replace`. Identical PNG, CSV, and JSON content is left untouched when possible, preventing harmless rewrites from invalidating downstream work.

## Public API and Kaggle execution

Run the notebook's definitions cell after every kernel restart. Importing the source defines classes and seeds CPU-side randomness; it does not train a model.

### Initial CPU stage

```python
cpu_state = Pipeline.run_kaggle_cpu_stage(
    dataset_path=None,       # automatic discovery, or provide the dataset root
    workspace_root=None,     # default Kaggle workspace
    refresh_quality=False,
    minimum_masks=None,      # keeps the researched minimum of 800 positives
)
```

### GPU stage

```python
gpu_state = Pipeline.run_kaggle_gpu_stage(
    dataset_path=None,
    workspace_root=None,
    after_review=True,
    force_training=False,
    force_prediction=False,
    refresh_quality=False,
    minimum_masks=None,
)
```

Keep both force flags `False` for normal resume behaviour. When manual targets change, semantic fingerprints decide which folds and prediction parts must be rebuilt.

### Final CPU stage

```python
final_state = Pipeline.run_kaggle_cpu_final_stage(
    dataset_path=None,
    workspace_root=None,
    action="evaluate",      # matching / features / evaluate / review / status
    force_matching=False,
    force_feature_bank=False,
    force_evaluation=False,
    refresh_quality=False,
    minimum_masks=None,
)

results = final_state.get("results")
comparisons = final_state.get("comparisons")
```

### Review, validation, status, and backup

```python
editor = Pipeline.review(scope="uncertain", limit=300)
cache_report = Pipeline.persistence_status()
status = Pipeline.status()
backup_path = Pipeline.backup(name="cad_attention_workspace_backup")
```

Before deleting anything from `/kaggle/working`, preview the explicit cleanup contract:

```python
Pipeline.clean_kaggle_working(dry_run=True)
```

## Main persistent artifacts

| Artifact family | Examples |
|---|---|
| Human review state | `manual_masks/`, `manual_annotation_labels.csv`, review history and overlays |
| Segmentation | `checkpoints/attention_unet_fold_*.pt`, per-fold CSV/JSON prediction parts, `predicted_attention_masks/*.png` |
| Audits | dataset manifest, image-quality audit, manual-target audit, segmentation metrics |
| Final CPU analysis | matching manifest, `patient_feature_bank.npz`, feature metadata, evaluation summary, paired comparisons, per-mode patient predictions |
| Recovery state | `pipeline_stage_state.json` and stage-specific fingerprint metadata |

The cleanup utility is dry-run by default and never touches `/kaggle/input`. Back up human-created masks and labels before any deliberate workspace reset.

## Interpretation and limitations

A high internal score does not establish a clinically meaningful mechanism. The complement modes are deliberate negative controls: predictive performance outside the heart indicates residual shortcut signal or confounding. The most important limitations are the 30-patient sample, one public JPEG release, unavailable DICOM metadata and clinical covariates, single-researcher masks without independent radiologist adjudication, outcome-informed matching in the secondary analysis, and absent external validation.

The repository preserves source and restart-safe logic, but it does not freeze a complete Kaggle container. CUDA kernels, drivers, library versions, and nondeterministic GPU operations may prevent bit-identical replay across environments even when the scientific protocol and semantic cache checks agree.

## Repository layout

```text
cad-cardiac-mri-project-dvm.py       # complete pipeline implementation
cad-cardiac-mri-project-dvm.ipynb    # Kaggle execution cells plus this documentation
README.md                             # technical repository guide
LICENSE                               # MIT License for original source code
LICENSES/CC-BY-4.0.txt              # CC BY 4.0 notice for original prose/diagrams
DATA_AND_THIRD_PARTY_NOTICE.md         # dataset, weights, dependencies, and clinical-use boundaries
.gitignore                            # excludes workspaces, model artifacts, and dataset folders
```

## Licensing and redistribution boundaries

| Material | Terms |
|---|---|
| Original Python source and original code cells | MIT License |
| Original README/notebook prose, tables, and explanatory diagrams | Creative Commons Attribution 4.0 International (`CC BY 4.0`) |
| Combined notebook | code cells are MIT-licensed; original narrative material is CC BY 4.0 |
| CAD Cardiac MRI Dataset and source images | not covered by this repository's licenses; obtain separately and follow the dataset provider's terms |
| Manual masks, checkpoints, automatic masks, feature banks, and data-derived outputs | not granted for redistribution by the source-code license; review dataset and institutional terms before sharing |
| Third-party Python packages and pretrained weights | remain under their own licenses and terms |

The project-specific academic-use permission described in the associated manuscript is not a sublicense for repository users. No source patient image is included in this publication package.

See `LICENSE`, `LICENSES/CC-BY-4.0.txt`, and `DATA_AND_THIRD_PARTY_NOTICE.md` for the complete repository notice.

## Authorship, supervision, and AI assistance

David Vlad Marghidan performed the dataset audit, implementation, experiments, manual review, analysis, and documentation under the scientific supervision of Prof. Dr. Smaranda Belciug. Danial Sharifrazi created and distributed the public dataset and clarified its patient-folder structure to the project author.

OpenAI ChatGPT was used as an assistive tool for code-review and refactoring discussions, debugging support, documentation, language editing, and consistency checking. It did not execute the Kaggle training run or generate, estimate, or impute the scientific metrics. Final scientific and editorial responsibility remains with David Vlad Marghidan under human supervision.

## Suggested citation

> Marghidan, David Vlad. *Patient-level coronary artery disease classification from cardiac MRI using out-of-fold heart segmentation and same-slice region controls*. Preprint manuscript, version 1.2, 5 October 2026.

Dataset article:

> Khozeimeh F, Sharifrazi D, Izadi NH, et al. RF-CNN-F: random forest with convolutional neural network features for coronary artery disease diagnosis based on cardiac magnetic resonance. *Scientific Reports*. 2022;12:11178. DOI: 10.1038/s41598-022-15374-5.
