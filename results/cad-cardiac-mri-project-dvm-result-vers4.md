```
####################################################################################################
CAD CARDIAC MRI — MULTI-EXPERIMENT PATIENT-LEVEL SUITE
####################################################################################################
[SUITE] Dataset: /kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset
[SUITE] Output: /kaggle/working/cad_patient_pipeline_outputs/multi_experiment_suite__3997634de7
[SUITE] Device: cuda
[SUITE] CUDA device: Tesla P100-PCIE-16GB
[SUITE] Enabled experiments (47): B0_ROI_HIER_LR_PCA, A1_FULL_HIER_LR_PCA, C1_BORDER_ONLY_HIER_LR_PCA, C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA, C3_EXPORT_PROVENANCE_ONLY_LR, C4_MONAI_QC_ONLY_LR, A2_ROI_FLAT_LR_PCA, A3_ROI_LEGACY_LOGODDS_LR, A4_ROI_LEGACY_MEANPROB_LR, A5_ROI_HIER_LINEAR_SVM_PCA, A6_ROI_HIER_LR_QUALITY_PCA, A7_ROI_HIER_LR_NO_PCA, A8_ROI_HIER_LR_NO_PCA_FIXEDC, B1_STANDARDIZED_ROI_HIER_LR_PCA, A9_STANDARDIZED_FULL_HIER_LR_PCA, C5_STANDARDIZED_BORDER05_HIER_LR_PCA, C6_STANDARDIZED_BORDER10_HIER_LR_PCA, C7_DETECTED_PADDING_MASK_HIER_LR_PCA, C8_STANDARDIZED_CORNERS_HIER_LR_PCA, C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA, A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA, A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA, C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA, C11_STANDARDIZATION_QC_ONLY_LR, C12_STANDARDIZED_MONAI_QC_ONLY_LR, C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA, C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA, C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA, A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA, A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA, A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA, A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA, C16_N_SLICES_ONLY_LR, C17_N_SERIES_ONLY_LR, C18_SERIES_LENGTH_ONLY_LR, C19_NATIVE_GEOMETRY_ONLY_LR, C20_FILE_SIZE_ONLY_LR, C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA, C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA, C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA, C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA, C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA, C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA, C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA, R1_ROI_HIER_LR_PCA_DROP10, R2_ROI_HIER_LR_PCA_DROP25, R3_ROI_HIER_LR_PCA_DROP50
[SUITE] Patient definition is fixed: patient_id = Directory_*.

==============================================================================
[PIPELINE 01/14] START: Validate suite configuration
[PIPELINE 01/14] Expected workload: Fast; no image scanning or model loading.
==============================================================================
[PIPELINE 01/14] COMPLETED: Validate suite configuration in 0 ms
[PIPELINE 01/14] All registry, CV, audit and model settings passed validation.

==============================================================================
[PIPELINE 02/14] START: Create output structure and save predeclared configuration
[PIPELINE 02/14] Expected workload: Fast filesystem and JSON operations.
==============================================================================
[PIPELINE 02/14] COMPLETED: Create output structure and save predeclared configuration in 3 ms
[PIPELINE 02/14] Configuration: /kaggle/working/cad_patient_pipeline_outputs/multi_experiment_suite__3997634de7/suite_configuration.json

==============================================================================
[PIPELINE 03/14] START: Discover Directory_* patients and image rows
[PIPELINE 03/14] Expected workload: Depends on filesystem and folder count; no pixel decoding yet.
==============================================================================
[DATA] Starting dataset discovery under: /kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset
[DATA] This stage scans folders and filenames only; JPEG pixels are decoded later during feature extraction.
[DATA] Scanning class folder: Normal
[DATA] Finished Normal: patients_added=16, images_added=37564, elapsed=44.21 s
[DATA] Scanning class folder: Sick
[DATA] Finished Sick: patients_added=14, images_added=25861, elapsed=19.95 s
Dataset discovery summary
  Patients: 30
  Normal patients: 16
  Sick patients: 14
  Images: 63425
  Series proxies: 2859
[DATA] Dataset discovery completed in 1 min 4.2 s
[PIPELINE 03/14] COMPLETED: Discover Directory_* patients and image rows in 1 min 4.2 s
[PIPELINE 03/14] Discovered 63425 image rows.

==============================================================================
[PIPELINE 04/14] START: Load or build the shared multi-view feature bank
[PIPELINE 04/14] Expected workload: Potentially the longest stage on a cache miss; all required image variants are encoded and cached.
==============================================================================
[FEATURE BANK FINGERPRINT] 1/63425 files (0%)
[FEATURE BANK FINGERPRINT] 6342/63425 files (10%)
[FEATURE BANK FINGERPRINT] 12684/63425 files (20%)
[FEATURE BANK FINGERPRINT] 19026/63425 files (30%)
[FEATURE BANK FINGERPRINT] 25368/63425 files (40%)
[FEATURE BANK FINGERPRINT] 31710/63425 files (50%)
[FEATURE BANK FINGERPRINT] 38052/63425 files (60%)
[FEATURE BANK FINGERPRINT] 44394/63425 files (70%)
[FEATURE BANK FINGERPRINT] 50736/63425 files (80%)
[FEATURE BANK FINGERPRINT] 57078/63425 files (90%)
[FEATURE BANK FINGERPRINT] 63420/63425 files (100%)
[FEATURE BANK FINGERPRINT] 63425/63425 files (100%)
[FEATURE BANK] Fingerprint completed in 43.84 s: 78729f5d81b9a8ef...
[FEATURE BANK] Cache miss or forced rebuild; neural-network inference will run now.
[MODEL][MONAI] Preparing the pretrained ventricular segmenter.
[MODEL][MONAI] Preferred path: verified official model.ts without importing the MONAI Python package.
[DETAIL] Checking whether the pinned MONAI bundle files are already present.
[DETAIL] MONAI bundle root resolved to: /kaggle/working/monai_bundles/ventricular_short_axis_3label
[MODEL][MONAI] Downloading missing bundle files from the pinned repository revision eefc17c8: models/model.ts, configs/metadata.json
[MODEL][MONAI] This can be slow on the first run and depends on the network connection. Later runs should reuse the local files.

```

Download complete: 

 6.63M/? [00:03<00:00, 5.37MB/s]

Fetching 2 files: 100%

 2/2 [00:01<00:00,  1.38it/s]

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

```

```
[MODEL][MONAI] Download call completed in 1.55 s.
[MODEL][MONAI] Pinned files are ready and cached at /kaggle/working/monai_bundles/ventricular_short_axis_3label. Total preparation time: 2.56 s
[MODEL][MONAI] Loading official pinned MONAI TorchScript segmenter from: /kaggle/working/monai_bundles/ventricular_short_axis_3label/models/model.ts
[MODEL][MONAI] Target device: cuda. A zero-input inference sanity check will run immediately after loading.
[DETAIL] Running MONAI zero-input shape/finite-value validation.
[MODEL][MONAI] Loaded and validated official pinned MONAI TorchScript segmenter in 1.72 s; output_shape=(1, 4, 256, 256).
[MODEL][MONAI] Segmenter preparation completed through the preferred official path in 4.30 s.
[MODEL][EfficientNet] Loading EfficientNet-B0 weights IMAGENET1K_V1.
[MODEL][EfficientNet] The first run may download the ImageNet checkpoint; later runs should use the torchvision cache.
Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth

```

```
100%|##########| 20.5M/20.5M [00:00<00:00, 81.8MB/s]

```

```
[MODEL][EfficientNet] Backbone initialized and ImageNet classifier removed in 471 ms.
[FEATURE BANK] Starting multi-view frozen extraction.
[FEATURE BANK] slices=63425, batches=7929, batch_size=8, modes=['monai_roi', 'full_image', 'border_only', 'outside_heart', 'standardized_monai_roi', 'standardized_full_image', 'standardized_border_05', 'standardized_border_10', 'detected_padding_mask', 'standardized_corners', 'standardized_center_crop', 'standardized_fixed_center_50', 'standardized_fixed_center_60', 'standardized_fixed_center_70', 'standardized_roi_zero_background', 'standardized_roi_zero_bg_center_fallback', 'standardized_roi_bbox', 'standardized_outside_large_bbox', 'standardized_soft_monai_mask_only', 'standardized_hard_monai_mask_only', 'standardized_monai_bbox_mask_only', 'standardized_soft_monai_histogram_only', 'standardized_soft_monai_block_shuffled', 'standardized_canonical_hard_monai_mask_only', 'standardized_canonical_soft_monai_mask_only'], MONAI_required=True, device=cuda
Feature bank:   0%|          | 0/7929 [00:00<?, ?batch/s][FEATURE BANK] 8/63425 slices (0.0%) | batch=1/7929 | rate=3.15 slices/s | ETA=5 h 35 min 33.7 s | latest_batch=2.24 s
Feature bank:   5%|4         | 395/7929 [02:27<52:40,  2.38batch/s]  [FEATURE BANK] 3168/63425 slices (5.0%) | batch=396/7929 | rate=21.40 slices/s | ETA=46 min 55.7 s | latest_batch=196 ms
Feature bank:  10%|9         | 791/7929 [04:58<50:21,  2.36batch/s][FEATURE BANK] 6336/63425 slices (10.0%) | batch=792/7929 | rate=21.21 slices/s | ETA=44 min 52.0 s | latest_batch=197 ms
Feature bank:  15%|#4        | 1187/7929 [07:31<43:43,  2.57batch/s][FEATURE BANK] 9504/63425 slices (15.0%) | batch=1188/7929 | rate=21.02 slices/s | ETA=42 min 45.7 s | latest_batch=194 ms
Feature bank:  20%|#9        | 1583/7929 [10:06<42:05,  2.51batch/s][FEATURE BANK] 12672/63425 slices (20.0%) | batch=1584/7929 | rate=20.87 slices/s | ETA=40 min 31.5 s | latest_batch=201 ms
Feature bank:  25%|##4       | 1979/7929 [12:44<31:55,  3.11batch/s][FEATURE BANK] 15840/63425 slices (25.0%) | batch=1980/7929 | rate=20.70 slices/s | ETA=38 min 18.5 s | latest_batch=195 ms
Feature bank:  30%|##9       | 2375/7929 [15:15<35:47,  2.59batch/s][FEATURE BANK] 19008/63425 slices (30.0%) | batch=2376/7929 | rate=20.76 slices/s | ETA=35 min 39.5 s | latest_batch=194 ms
Feature bank:  35%|###4      | 2771/7929 [17:46<38:05,  2.26batch/s][FEATURE BANK] 22176/63425 slices (35.0%) | batch=2772/7929 | rate=20.78 slices/s | ETA=33 min 4.6 s | latest_batch=196 ms
Feature bank:  40%|###9      | 3167/7929 [20:16<31:25,  2.53batch/s][FEATURE BANK] 25344/63425 slices (40.0%) | batch=3168/7929 | rate=20.83 slices/s | ETA=30 min 28.3 s | latest_batch=197 ms
Feature bank:  45%|####4     | 3563/7929 [22:44<30:14,  2.41batch/s][FEATURE BANK] 28512/63425 slices (45.0%) | batch=3564/7929 | rate=20.89 slices/s | ETA=27 min 51.6 s | latest_batch=198 ms
Feature bank:  50%|####9     | 3959/7929 [25:10<25:30,  2.59batch/s][FEATURE BANK] 31680/63425 slices (49.9%) | batch=3960/7929 | rate=20.97 slices/s | ETA=25 min 13.5 s | latest_batch=195 ms
Feature bank:  55%|#####4    | 4355/7929 [27:47<23:54,  2.49batch/s][FEATURE BANK] 34848/63425 slices (54.9%) | batch=4356/7929 | rate=20.89 slices/s | ETA=22 min 47.8 s | latest_batch=190 ms
Feature bank:  60%|#####9    | 4751/7929 [30:09<17:36,  3.01batch/s][FEATURE BANK] 38016/63425 slices (59.9%) | batch=4752/7929 | rate=21.01 slices/s | ETA=20 min 9.6 s | latest_batch=199 ms
Feature bank:  65%|######4   | 5147/7929 [32:27<14:12,  3.26batch/s][FEATURE BANK] 41184/63425 slices (64.9%) | batch=5148/7929 | rate=21.14 slices/s | ETA=17 min 31.9 s | latest_batch=188 ms
Feature bank:  70%|######9   | 5543/7929 [34:36<13:28,  2.95batch/s][FEATURE BANK] 44352/63425 slices (69.9%) | batch=5544/7929 | rate=21.36 slices/s | ETA=14 min 53.1 s | latest_batch=195 ms
Feature bank:  75%|#######4  | 5939/7929 [37:05<10:33,  3.14batch/s][FEATURE BANK] 47520/63425 slices (74.9%) | batch=5940/7929 | rate=21.35 slices/s | ETA=12 min 24.9 s | latest_batch=189 ms
Feature bank:  80%|#######9  | 6335/7929 [39:34<08:37,  3.08batch/s][FEATURE BANK] 50688/63425 slices (79.9%) | batch=6336/7929 | rate=21.34 slices/s | ETA=9 min 56.8 s | latest_batch=197 ms
Feature bank:  85%|########4 | 6731/7929 [41:47<06:49,  2.93batch/s][FEATURE BANK] 53856/63425 slices (84.9%) | batch=6732/7929 | rate=21.47 slices/s | ETA=7 min 25.6 s | latest_batch=193 ms
Feature bank:  90%|########9 | 7127/7929 [44:07<05:20,  2.50batch/s][FEATURE BANK] 57024/63425 slices (89.9%) | batch=7128/7929 | rate=21.53 slices/s | ETA=4 min 57.3 s | latest_batch=197 ms
Feature bank:  95%|#########4| 7523/7929 [46:23<02:14,  3.02batch/s][FEATURE BANK] 60192/63425 slices (94.9%) | batch=7524/7929 | rate=21.62 slices/s | ETA=2 min 29.5 s | latest_batch=196 ms
Feature bank: 100%|#########9| 7919/7929 [48:29<00:03,  3.07batch/s][FEATURE BANK] 63360/63425 slices (99.9%) | batch=7920/7929 | rate=21.77 slices/s | ETA=2.99 s | latest_batch=196 ms
Feature bank: 100%|#########9| 7928/7929 [48:32<00:00,  3.03batch/s][FEATURE BANK] 63425/63425 slices (100.0%) | batch=7929/7929 | rate=21.77 slices/s | ETA=0 ms | latest_batch=646 ms
Feature bank: 100%|##########| 7929/7929 [48:33<00:00,  2.72batch/s]
[FEATURE BANK] Extraction and cache write completed in 48 min 33.7 s.
[FEATURE BANK] Loading cache: /kaggle/working/cad_patient_pipeline_outputs/feature_bank_cache/78729f5d81b9a8ef
[FEATURE BANK] Cache hit loaded in 14 ms; slices=63425, modes=['monai_roi', 'full_image', 'border_only', 'outside_heart', 'standardized_monai_roi', 'standardized_full_image', 'standardized_border_05', 'standardized_border_10', 'detected_padding_mask', 'standardized_corners', 'standardized_center_crop', 'standardized_fixed_center_50', 'standardized_fixed_center_60', 'standardized_fixed_center_70', 'standardized_roi_zero_background', 'standardized_roi_zero_bg_center_fallback', 'standardized_roi_bbox', 'standardized_outside_large_bbox', 'standardized_soft_monai_mask_only', 'standardized_hard_monai_mask_only', 'standardized_monai_bbox_mask_only', 'standardized_soft_monai_histogram_only', 'standardized_soft_monai_block_shuffled', 'standardized_canonical_hard_monai_mask_only', 'standardized_canonical_soft_monai_mask_only'].
[PIPELINE 04/14] COMPLETED: Load or build the shared multi-view feature bank in 49 min 22.4 s
[PIPELINE 04/14] Cache=MISS; modes=['monai_roi', 'full_image', 'border_only', 'outside_heart', 'standardized_monai_roi', 'standardized_full_image', 'standardized_border_05', 'standardized_border_10', 'detected_padding_mask', 'standardized_corners', 'standardized_center_crop', 'standardized_fixed_center_50', 'standardized_fixed_center_60', 'standardized_fixed_center_70', 'standardized_roi_zero_background', 'standardized_roi_zero_bg_center_fallback', 'standardized_roi_bbox', 'standardized_outside_large_bbox', 'standardized_soft_monai_mask_only', 'standardized_hard_monai_mask_only', 'standardized_monai_bbox_mask_only', 'standardized_soft_monai_histogram_only', 'standardized_soft_monai_block_shuffled', 'standardized_canonical_hard_monai_mask_only', 'standardized_canonical_soft_monai_mask_only']; path=/kaggle/working/cad_patient_pipeline_outputs/feature_bank_cache/78729f5d81b9a8ef

==============================================================================
[PIPELINE 05/14] START: Audit exact and perceptual cross-patient duplicates
[PIPELINE 05/14] Expected workload: Hash grouping and pHash candidate search; no neural-network inference.
==============================================================================
[EXACT DUPLICATES] groups=1094, cross_patient_groups=0, cross_label_groups=0, patient_edges=0
[PERCEPTUAL DUPLICATES] Starting complete BK-tree radius search over 30225 unique pHash values.
[PERCEPTUAL DUPLICATES] 1/30225 hashes (0%) | elapsed=1 ms | ETA=22.65 s
[PERCEPTUAL DUPLICATES] 3022/30225 hashes (10%) | elapsed=388 ms | ETA=3.49 s
[PERCEPTUAL DUPLICATES] 6044/30225 hashes (20%) | elapsed=839 ms | ETA=3.36 s
[PERCEPTUAL DUPLICATES] 9066/30225 hashes (30%) | elapsed=1.34 s | ETA=3.12 s
[PERCEPTUAL DUPLICATES] 12088/30225 hashes (40%) | elapsed=1.81 s | ETA=2.72 s
[PERCEPTUAL DUPLICATES] 15110/30225 hashes (50%) | elapsed=2.32 s | ETA=2.32 s
[PERCEPTUAL DUPLICATES] 18132/30225 hashes (60%) | elapsed=2.76 s | ETA=1.84 s
[PERCEPTUAL DUPLICATES] 21154/30225 hashes (70%) | elapsed=3.20 s | ETA=1.37 s
[PERCEPTUAL DUPLICATES] 24176/30225 hashes (80%) | elapsed=3.68 s | ETA=921 ms
[PERCEPTUAL DUPLICATES] 27198/30225 hashes (90%) | elapsed=4.14 s | ETA=461 ms
[PERCEPTUAL DUPLICATES] 30220/30225 hashes (100%) | elapsed=4.55 s | ETA=1 ms
[PERCEPTUAL DUPLICATES] 30225/30225 hashes (100%) | elapsed=4.55 s | ETA=0 ms
[PERCEPTUAL DUPLICATES] method=complete_BK_tree, unique_hashes=30225, patient_pair_candidates=24, cross_label=10, skipped_large_buckets=0, truncated=False
[WARNING] Perceptual-hash matches are screening candidates, not confirmed duplicates. Review the saved example pairs manually.
[PIPELINE 05/14] COMPLETED: Audit exact and perceptual cross-patient duplicates in 5.94 s
[PIPELINE 05/14] Exact patient edges=0; perceptual candidate edges=24.

==============================================================================
[PIPELINE 06/14] START: Create one duplicate-aware outer-fold and cohort manifest
[PIPELINE 06/14] Expected workload: Fast patient-level grouping plus one large CSV write.
==============================================================================
[FOLD MANIFEST] fold=1: patients=6, Normal=4, Sick=2, components=6
[FOLD MANIFEST] fold=2: patients=6, Normal=3, Sick=3, components=6
[FOLD MANIFEST] fold=3: patients=6, Normal=3, Sick=3, components=6
[FOLD MANIFEST] fold=4: patients=6, Normal=3, Sick=3, components=6
[FOLD MANIFEST] fold=5: patients=6, Normal=3, Sick=3, components=6
[SERIES REVIEW] Wrote 2859 blinded folder-proxy rows: /kaggle/working/cad_patient_pipeline_outputs/multi_experiment_suite__3997634de7/audits/series_annotation_template.csv
[SERIES REVIEW] Label key kept separately: /kaggle/working/cad_patient_pipeline_outputs/multi_experiment_suite__3997634de7/audits/series_annotation_label_key.csv
[PIPELINE 06/14] COMPLETED: Create one duplicate-aware outer-fold and cohort manifest in 4.41 s
[PIPELINE 06/14] Every experiment will reuse these exact outer folds.

==============================================================================
[PIPELINE 07/14] START: Build and save provenance, standardization and MONAI-QC controls
[PIPELINE 07/14] Expected workload: Patient-level aggregation and descriptive audit files.
==============================================================================
[MONAI QC] Patient-level mean plausible-mask rate: Normal=0.937, Sick=0.930, difference=-0.007, 95% CI=[-0.031, +0.014]
[STANDARDIZED MONAI QC] Patient-level mean plausible-mask rate: Normal=0.964, Sick=0.974, difference=+0.009, 95% CI=[-0.003, +0.020]
[PIPELINE 07/14] COMPLETED: Build and save provenance, standardization and MONAI-QC controls in 426 ms
[PIPELINE 07/14] Patient-level control matrices are ready for broad and decomposed provenance, original/standardized MONAI QC, and standardization geometry.

==============================================================================
[PIPELINE 08/14] START: Run every enabled experiment on the shared folds
[PIPELINE 08/14] Expected workload: Several fold-local fits. Patient-embedding experiments are fast; legacy slice classifiers are substantially heavier.
==============================================================================

[SUITE] Launching experiment 1/47: B0_ROI_HIER_LR_PCA
[SUITE] Prepared data representation ('monai_roi', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 471 ms.

====================================================================================================
[EXPERIMENT] START: B0_ROI_HIER_LR_PCA
[EXPERIMENT] Historical original-canvas reference: confidence-gated MONAI ROI, equal slice weights, hierarchical embedding pooling, PCA and Logistic Regression.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT B0_ROI_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][B0_ROI_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8958
[INNER CV][B0_ROI_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9028
[INNER CV][B0_ROI_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8958
[INNER CV][B0_ROI_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9028
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8958, best_candidate_inner_AUC=0.9028, training_only_threshold=0.632320, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=213 ms

[EXPERIMENT B0_ROI_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B0_ROI_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][B0_ROI_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][B0_ROI_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9580
[INNER CV][B0_ROI_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9790
[OUTER 2] selected_C=10, selected_inner_AUC=0.9790, best_candidate_inner_AUC=0.9790, training_only_threshold=0.074422, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=108 ms

[EXPERIMENT B0_ROI_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B0_ROI_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8042
[INNER CV][B0_ROI_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][B0_ROI_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8252
[INNER CV][B0_ROI_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8322
[OUTER 3] selected_C=1, selected_inner_AUC=0.8252, best_candidate_inner_AUC=0.8322, training_only_threshold=0.656002, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=108 ms

[EXPERIMENT B0_ROI_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B0_ROI_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][B0_ROI_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][B0_ROI_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8601
[INNER CV][B0_ROI_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8671
[OUTER 4] selected_C=1, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8671, training_only_threshold=0.547159, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=126 ms

[EXPERIMENT B0_ROI_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B0_ROI_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8741
[INNER CV][B0_ROI_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8951
[INNER CV][B0_ROI_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.9161
[INNER CV][B0_ROI_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.9161
[OUTER 5] selected_C=1, selected_inner_AUC=0.9161, best_candidate_inner_AUC=0.9161, training_only_threshold=0.630527, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=109 ms
[EXPERIMENT] COMPLETED: B0_ROI_HIER_LR_PCA | AUC=0.9241 [0.8036, 1.0000] | AUPRC=0.9309 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.76 s

[SUITE] Launching experiment 2/47: A1_FULL_HIER_LR_PCA
[SUITE] Prepared data representation ('full_image', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 473 ms.

====================================================================================================
[EXPERIMENT] START: A1_FULL_HIER_LR_PCA
[EXPERIMENT] Full-image ablation with all downstream settings unchanged.
[EXPERIMENT] feature_mode=full_image, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A1_FULL_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A1_FULL_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8194
[INNER CV][A1_FULL_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8333
[INNER CV][A1_FULL_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8194
[INNER CV][A1_FULL_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8264
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.8333, best_candidate_inner_AUC=0.8333, training_only_threshold=0.705708, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=107 ms

[EXPERIMENT A1_FULL_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A1_FULL_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9441
[INNER CV][A1_FULL_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][A1_FULL_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9510
[INNER CV][A1_FULL_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9580
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9580, training_only_threshold=0.432436, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=111 ms

[EXPERIMENT A1_FULL_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A1_FULL_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7762
[INNER CV][A1_FULL_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7413
[INNER CV][A1_FULL_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7343
[INNER CV][A1_FULL_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7273
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7762, best_candidate_inner_AUC=0.7762, training_only_threshold=0.531255, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT A1_FULL_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A1_FULL_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8462
[INNER CV][A1_FULL_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][A1_FULL_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8462
[INNER CV][A1_FULL_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8462
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8462, best_candidate_inner_AUC=0.8462, training_only_threshold=0.547304, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=111 ms

[EXPERIMENT A1_FULL_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A1_FULL_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8881
[INNER CV][A1_FULL_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8951
[INNER CV][A1_FULL_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8951
[INNER CV][A1_FULL_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8951
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8881, best_candidate_inner_AUC=0.8951, training_only_threshold=0.431776, inner_splits=3
[OUTER 5] held-out AUC=0.7778; runtime=125 ms
[EXPERIMENT] COMPLETED: A1_FULL_HIER_LR_PCA | AUC=0.8973 [0.7589, 0.9866] | AUPRC=0.8666 | Sensitivity=0.7143 | Specificity=0.9375 | F1=0.8000 | runtime=5.70 s

[SUITE] Launching experiment 3/47: C1_BORDER_ONLY_HIER_LR_PCA
[SUITE] Prepared data representation ('border_only', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 512 ms.

====================================================================================================
[EXPERIMENT] START: C1_BORDER_ONLY_HIER_LR_PCA
[EXPERIMENT] Negative control retaining only the outer image border; a high AUC would indicate export/style shortcut risk.
[EXPERIMENT] feature_mode=border_only, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C1_BORDER_ONLY_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.6111
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.6319
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.6250
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.6389
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.6319, best_candidate_inner_AUC=0.6389, training_only_threshold=0.423506, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT C1_BORDER_ONLY_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8252
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.8601
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.8671
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.8671
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8671, training_only_threshold=0.764597, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=114 ms

[EXPERIMENT C1_BORDER_ONLY_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7413
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7622
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7832
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7832
[OUTER 3] selected_C=1, selected_inner_AUC=0.7832, best_candidate_inner_AUC=0.7832, training_only_threshold=0.885483, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT C1_BORDER_ONLY_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8322
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8182
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8252
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8322, best_candidate_inner_AUC=0.8322, training_only_threshold=0.589937, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=112 ms

[EXPERIMENT C1_BORDER_ONLY_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.6713
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.6923
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.6923
[INNER CV][C1_BORDER_ONLY_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.6993
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.6923, best_candidate_inner_AUC=0.6993, training_only_threshold=0.361000, inner_splits=3
[OUTER 5] held-out AUC=0.6667; runtime=117 ms
[EXPERIMENT] COMPLETED: C1_BORDER_ONLY_HIER_LR_PCA | AUC=0.8750 [0.7188, 0.9732] | AUPRC=0.8971 | Sensitivity=0.6429 | Specificity=0.7500 | F1=0.6667 | runtime=5.77 s

[SUITE] Launching experiment 4/47: C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA
[SUITE] Prepared data representation ('outside_heart', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 485 ms.

====================================================================================================
[EXPERIMENT] START: C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA
[EXPERIMENT] Negative control retaining signal outside the dilated MONAI soft mask. The mask is applied unconditionally and remains a proxy, not validated anatomy.
[EXPERIMENT] feature_mode=outside_heart, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7778
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.7917
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.7917
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.7917
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.7917, best_candidate_inner_AUC=0.7917, training_only_threshold=0.274620, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=114 ms

[EXPERIMENT C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8881
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9301
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9371
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9441
[OUTER 2] selected_C=1, selected_inner_AUC=0.9371, best_candidate_inner_AUC=0.9441, training_only_threshold=0.439043, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=110 ms

[EXPERIMENT C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8392
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8531
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8531
[OUTER 3] selected_C=0.1, selected_inner_AUC=0.8462, best_candidate_inner_AUC=0.8531, training_only_threshold=0.245438, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=112 ms

[EXPERIMENT C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9021
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9301
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9231
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9231
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.9301, best_candidate_inner_AUC=0.9301, training_only_threshold=0.788495, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=108 ms

[EXPERIMENT C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8601
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8881
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8951
[INNER CV][C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8951
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.8881, best_candidate_inner_AUC=0.8951, training_only_threshold=0.781822, inner_splits=3
[OUTER 5] held-out AUC=0.7778; runtime=108 ms
[EXPERIMENT] COMPLETED: C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA | AUC=0.8839 [0.7411, 0.9911] | AUPRC=0.8976 | Sensitivity=0.7143 | Specificity=0.9375 | F1=0.8000 | runtime=5.69 s

[SUITE] Launching experiment 5/47: C3_EXPORT_PROVENANCE_ONLY_LR
[SUITE] Prepared data representation ('provenance_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C3_EXPORT_PROVENANCE_ONLY_LR
[EXPERIMENT] Patient classifier using only the conservative export/provenance subset: native geometry, file-size-per-pixel, padding and border statistics. Central intensity, entropy and sharpness are excluded.
[EXPERIMENT] feature_mode=provenance_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C3_EXPORT_PROVENANCE_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.7431
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.6667
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.6528
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.6458
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7431, best_candidate_inner_AUC=0.7431, training_only_threshold=0.578834, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=59 ms

[EXPERIMENT C3_EXPORT_PROVENANCE_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.8392
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.8671
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.8531
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.8462
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.8671, best_candidate_inner_AUC=0.8671, training_only_threshold=0.532593, inner_splits=3
[OUTER 2] held-out AUC=0.6667; runtime=55 ms

[EXPERIMENT C3_EXPORT_PROVENANCE_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.7552
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.7483
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.7133
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.6993
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7552, best_candidate_inner_AUC=0.7552, training_only_threshold=0.532105, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=54 ms

[EXPERIMENT C3_EXPORT_PROVENANCE_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.8392
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.7972
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.8252
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.7762
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8392, best_candidate_inner_AUC=0.8392, training_only_threshold=0.465787, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=57 ms

[EXPERIMENT C3_EXPORT_PROVENANCE_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.8112
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.7692
[INNER CV][C3_EXPORT_PROVENANCE_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.7273
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8531, best_candidate_inner_AUC=0.8531, training_only_threshold=0.523220, inner_splits=3
[OUTER 5] held-out AUC=0.5556; runtime=55 ms
[EXPERIMENT] COMPLETED: C3_EXPORT_PROVENANCE_ONLY_LR | AUC=0.7902 [0.5938, 0.9554] | AUPRC=0.8452 | Sensitivity=0.6429 | Specificity=0.8750 | F1=0.7200 | runtime=5.44 s

[SUITE] Launching experiment 6/47: C4_MONAI_QC_ONLY_LR
[SUITE] Prepared data representation ('monai_qc_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C4_MONAI_QC_ONLY_LR
[EXPERIMENT] Patient classifier using only MONAI gate/QC statistics. A high AUC would suggest protocol or sequence confounding.
[EXPERIMENT] feature_mode=monai_qc_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C4_MONAI_QC_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.3125
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.3403
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.4028
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.4306
[OUTER 1] selected_C=10, selected_inner_AUC=0.4306, best_candidate_inner_AUC=0.4306, training_only_threshold=0.999429, inner_splits=3
[OUTER 1] held-out AUC=0.8750; runtime=50 ms

[EXPERIMENT C4_MONAI_QC_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.7133
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.6993
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.6573
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.6154
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.7133, best_candidate_inner_AUC=0.7133, training_only_threshold=0.509963, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=51 ms

[EXPERIMENT C4_MONAI_QC_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.3776
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.3986
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.4476
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.4406
[OUTER 3] selected_C=1, selected_inner_AUC=0.4476, best_candidate_inner_AUC=0.4476, training_only_threshold=0.171349, inner_splits=3
[OUTER 3] held-out AUC=0.7778; runtime=51 ms

[EXPERIMENT C4_MONAI_QC_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.5175
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.5035
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.5594
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.6014
[OUTER 4] selected_C=10, selected_inner_AUC=0.6014, best_candidate_inner_AUC=0.6014, training_only_threshold=0.403005, inner_splits=3
[OUTER 4] held-out AUC=0.4444; runtime=54 ms

[EXPERIMENT C4_MONAI_QC_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.8392
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.8322
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.8112
[INNER CV][C4_MONAI_QC_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.8042
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8392, best_candidate_inner_AUC=0.8392, training_only_threshold=0.500617, inner_splits=3
[OUTER 5] held-out AUC=0.1111; runtime=51 ms
[EXPERIMENT] COMPLETED: C4_MONAI_QC_ONLY_LR | AUC=0.6741 [0.4686, 0.8527] | AUPRC=0.6339 | Sensitivity=0.4286 | Specificity=0.5625 | F1=0.4444 | runtime=5.34 s

[SUITE] Launching experiment 7/47: A2_ROI_FLAT_LR_PCA
[SUITE] Prepared data representation ('monai_roi', 'patient_embedding', 'flat', 'equal', 0.0, False) in 361 ms.

====================================================================================================
[EXPERIMENT] START: A2_ROI_FLAT_LR_PCA
[EXPERIMENT] Flat slice-to-patient embedding mean; long series can dominate because the series hierarchy is intentionally removed.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=flat, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A2_ROI_FLAT_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7986
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.7917
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.7986
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8056
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7986, best_candidate_inner_AUC=0.8056, training_only_threshold=0.702288, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=112 ms

[EXPERIMENT A2_ROI_FLAT_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8322
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.8322
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.8252
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.8462
[OUTER 2] selected_C=10, selected_inner_AUC=0.8462, best_candidate_inner_AUC=0.8462, training_only_threshold=0.547520, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=112 ms

[EXPERIMENT A2_ROI_FLAT_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7273
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7063
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.6923
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.6923
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7273, best_candidate_inner_AUC=0.7273, training_only_threshold=0.585853, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=115 ms

[EXPERIMENT A2_ROI_FLAT_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.7972
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.7972
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.7832
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.7762
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7972, best_candidate_inner_AUC=0.7972, training_only_threshold=0.557877, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=114 ms

[EXPERIMENT A2_ROI_FLAT_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8252
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8112
[INNER CV][A2_ROI_FLAT_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8112
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8252, best_candidate_inner_AUC=0.8252, training_only_threshold=0.577996, inner_splits=3
[OUTER 5] held-out AUC=0.7778; runtime=112 ms
[EXPERIMENT] COMPLETED: A2_ROI_FLAT_LR_PCA | AUC=0.8125 [0.6250, 0.9420] | AUPRC=0.8438 | Sensitivity=0.6429 | Specificity=0.8750 | F1=0.7200 | runtime=5.66 s

[SUITE] Launching experiment 8/47: A3_ROI_LEGACY_LOGODDS_LR
[SUITE] Prepared data representation ('monai_roi', 'slice_probability_fusion', 'probability_fusion', 'equal', 0.0, False) in 4 ms.

====================================================================================================
[EXPERIMENT] START: A3_ROI_LEGACY_LOGODDS_LR
[EXPERIMENT] Legacy weakly supervised slice classifier with hierarchical log-odds fusion; retained only as an ablation.
[EXPERIMENT] feature_mode=monai_roi, strategy=slice_probability_fusion, pooling=probability_fusion, weighting=equal, classifier=logistic_regression, PCA=False, fusion=log_odds, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A3_ROI_LEGACY_LOGODDS_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A3_ROI_LEGACY_LOGODDS_LR][outer=1] C=1, pooled_inner_AUC=0.8194
[OUTER 1] selected_C=1, selected_inner_AUC=0.8194, best_candidate_inner_AUC=0.8194, training_only_threshold=0.496822, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=1 min 7.7 s

[EXPERIMENT A3_ROI_LEGACY_LOGODDS_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A3_ROI_LEGACY_LOGODDS_LR][outer=2] C=1, pooled_inner_AUC=0.8741
[OUTER 2] selected_C=1, selected_inner_AUC=0.8741, best_candidate_inner_AUC=0.8741, training_only_threshold=0.444866, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=1 min 7.3 s

[EXPERIMENT A3_ROI_LEGACY_LOGODDS_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A3_ROI_LEGACY_LOGODDS_LR][outer=3] C=1, pooled_inner_AUC=0.7972
[OUTER 3] selected_C=1, selected_inner_AUC=0.7972, best_candidate_inner_AUC=0.7972, training_only_threshold=0.525120, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=1 min 5.5 s

[EXPERIMENT A3_ROI_LEGACY_LOGODDS_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A3_ROI_LEGACY_LOGODDS_LR][outer=4] C=1, pooled_inner_AUC=0.8811
[OUTER 4] selected_C=1, selected_inner_AUC=0.8811, best_candidate_inner_AUC=0.8811, training_only_threshold=0.473544, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=1 min 8.1 s

[EXPERIMENT A3_ROI_LEGACY_LOGODDS_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A3_ROI_LEGACY_LOGODDS_LR][outer=5] C=1, pooled_inner_AUC=0.8252
[OUTER 5] selected_C=1, selected_inner_AUC=0.8252, best_candidate_inner_AUC=0.8252, training_only_threshold=0.466441, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=59.35 s
[EXPERIMENT] COMPLETED: A3_ROI_LEGACY_LOGODDS_LR | AUC=0.8839 [0.7232, 0.9911] | AUPRC=0.9131 | Sensitivity=0.7143 | Specificity=0.9375 | F1=0.8000 | runtime=5 min 33.3 s

[SUITE] Launching experiment 9/47: A4_ROI_LEGACY_MEANPROB_LR

====================================================================================================
[EXPERIMENT] START: A4_ROI_LEGACY_MEANPROB_LR
[EXPERIMENT] Same legacy slice classifier as A3, but mean-probability fusion replaces log-odds fusion.
[EXPERIMENT] feature_mode=monai_roi, strategy=slice_probability_fusion, pooling=probability_fusion, weighting=equal, classifier=logistic_regression, PCA=False, fusion=mean_probability, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A4_ROI_LEGACY_MEANPROB_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[INNER CV][A4_ROI_LEGACY_MEANPROB_LR][outer=1] C=1, pooled_inner_AUC=0.8194
[OUTER 1] selected_C=1, selected_inner_AUC=0.8194, best_candidate_inner_AUC=0.8194, training_only_threshold=0.497236, inner_splits=3
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[OUTER 1] held-out AUC=1.0000; runtime=294 ms

[EXPERIMENT A4_ROI_LEGACY_MEANPROB_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[INNER CV][A4_ROI_LEGACY_MEANPROB_LR][outer=2] C=1, pooled_inner_AUC=0.8741
[OUTER 2] selected_C=1, selected_inner_AUC=0.8741, best_candidate_inner_AUC=0.8741, training_only_threshold=0.463528, inner_splits=3
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[OUTER 2] held-out AUC=0.8889; runtime=293 ms

[EXPERIMENT A4_ROI_LEGACY_MEANPROB_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[INNER CV][A4_ROI_LEGACY_MEANPROB_LR][outer=3] C=1, pooled_inner_AUC=0.8042
[OUTER 3] selected_C=1, selected_inner_AUC=0.8042, best_candidate_inner_AUC=0.8042, training_only_threshold=0.528321, inner_splits=3
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[OUTER 3] held-out AUC=1.0000; runtime=298 ms

[EXPERIMENT A4_ROI_LEGACY_MEANPROB_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[INNER CV][A4_ROI_LEGACY_MEANPROB_LR][outer=4] C=1, pooled_inner_AUC=0.8741
[OUTER 4] selected_C=1, selected_inner_AUC=0.8741, best_candidate_inner_AUC=0.8741, training_only_threshold=0.479734, inner_splits=3
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[OUTER 4] held-out AUC=0.7778; runtime=299 ms

[EXPERIMENT A4_ROI_LEGACY_MEANPROB_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[INNER CV][A4_ROI_LEGACY_MEANPROB_LR][outer=5] C=1, pooled_inner_AUC=0.8042
[OUTER 5] selected_C=1, selected_inner_AUC=0.8042, best_candidate_inner_AUC=0.8042, training_only_threshold=0.569833, inner_splits=3
[LEGACY CACHE] Reusing fold-local slice probabilities; only mean_probability fusion is recomputed.
[OUTER 5] held-out AUC=1.0000; runtime=296 ms
[EXPERIMENT] COMPLETED: A4_ROI_LEGACY_MEANPROB_LR | AUC=0.8795 [0.7232, 0.9911] | AUPRC=0.9109 | Sensitivity=0.7143 | Specificity=0.9375 | F1=0.8000 | runtime=6.75 s

[SUITE] Launching experiment 10/47: A5_ROI_HIER_LINEAR_SVM_PCA

====================================================================================================
[EXPERIMENT] START: A5_ROI_HIER_LINEAR_SVM_PCA
[EXPERIMENT] Linear SVM ablation. Fold-local sigmoid calibration is learned only from inner out-of-fold training scores.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=linear_svm, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A5_ROI_HIER_LINEAR_SVM_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8889
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8819
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=1] C=1, pooled_inner_AUC=0.8889
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=1] C=10, pooled_inner_AUC=0.8889
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8889, best_candidate_inner_AUC=0.8889, training_only_threshold=0.609082, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=168 ms

[EXPERIMENT A5_ROI_HIER_LINEAR_SVM_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9860
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9860
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=2] C=1, pooled_inner_AUC=0.9860
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=2] C=10, pooled_inner_AUC=0.9790
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.9860, best_candidate_inner_AUC=0.9860, training_only_threshold=0.360887, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=133 ms

[EXPERIMENT A5_ROI_HIER_LINEAR_SVM_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8252
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8322
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=3] C=1, pooled_inner_AUC=0.8322
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=3] C=10, pooled_inner_AUC=0.8042
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.8252, best_candidate_inner_AUC=0.8322, training_only_threshold=0.540882, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=126 ms

[EXPERIMENT A5_ROI_HIER_LINEAR_SVM_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8671
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8671
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=4] C=1, pooled_inner_AUC=0.8671
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=4] C=10, pooled_inner_AUC=0.8601
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8671, best_candidate_inner_AUC=0.8671, training_only_threshold=0.448254, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=130 ms

[EXPERIMENT A5_ROI_HIER_LINEAR_SVM_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=5] C=0.01, pooled_inner_AUC=0.9161
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=5] C=0.1, pooled_inner_AUC=0.9231
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=5] C=1, pooled_inner_AUC=0.9231
[INNER CV][A5_ROI_HIER_LINEAR_SVM_PCA][outer=5] C=10, pooled_inner_AUC=0.9161
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.9161, best_candidate_inner_AUC=0.9231, training_only_threshold=0.505172, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=128 ms
[EXPERIMENT] COMPLETED: A5_ROI_HIER_LINEAR_SVM_PCA | AUC=0.9420 [0.8302, 1.0000] | AUPRC=0.9490 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.92 s

[SUITE] Launching experiment 11/47: A6_ROI_HIER_LR_QUALITY_PCA
[SUITE] Prepared data representation ('monai_roi', 'patient_embedding', 'hierarchical', 'quality', 0.0, False) in 4.08 s.

====================================================================================================
[EXPERIMENT] START: A6_ROI_HIER_LR_QUALITY_PCA
[EXPERIMENT] Optional series-local intensity-standard-deviation weighting heuristic; every slice remains included.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=quality, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A6_ROI_HIER_LR_QUALITY_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9236
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9097
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=1] C=1, pooled_inner_AUC=0.9167
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=1] C=10, pooled_inner_AUC=0.9167
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.9236, best_candidate_inner_AUC=0.9236, training_only_threshold=0.433002, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT A6_ROI_HIER_LR_QUALITY_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9650
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=2] C=1, pooled_inner_AUC=0.9860
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=2] C=10, pooled_inner_AUC=0.9860
[OUTER 2] selected_C=1, selected_inner_AUC=0.9860, best_candidate_inner_AUC=0.9860, training_only_threshold=0.129575, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=127 ms

[EXPERIMENT A6_ROI_HIER_LR_QUALITY_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7972
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8042
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=3] C=1, pooled_inner_AUC=0.8042
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=3] C=10, pooled_inner_AUC=0.8042
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7972, best_candidate_inner_AUC=0.8042, training_only_threshold=0.519321, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=117 ms

[EXPERIMENT A6_ROI_HIER_LR_QUALITY_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=4] C=1, pooled_inner_AUC=0.8392
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=4] C=10, pooled_inner_AUC=0.8531
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8531, best_candidate_inner_AUC=0.8531, training_only_threshold=0.457943, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=110 ms

[EXPERIMENT A6_ROI_HIER_LR_QUALITY_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8811
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8881
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=5] C=1, pooled_inner_AUC=0.9021
[INNER CV][A6_ROI_HIER_LR_QUALITY_PCA][outer=5] C=10, pooled_inner_AUC=0.9091
[OUTER 5] selected_C=1, selected_inner_AUC=0.9021, best_candidate_inner_AUC=0.9091, training_only_threshold=0.649760, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=115 ms
[EXPERIMENT] COMPLETED: A6_ROI_HIER_LR_QUALITY_PCA | AUC=0.9375 [0.8393, 1.0000] | AUPRC=0.9259 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.74 s

[SUITE] Launching experiment 12/47: A7_ROI_HIER_LR_NO_PCA

====================================================================================================
[EXPERIMENT] START: A7_ROI_HIER_LR_NO_PCA
[EXPERIMENT] PCA-off ablation with all other baseline settings unchanged.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A7_ROI_HIER_LR_NO_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8819
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8819
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=1] C=1, pooled_inner_AUC=0.8889
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=1] C=10, pooled_inner_AUC=0.8889
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8819, best_candidate_inner_AUC=0.8889, training_only_threshold=0.595882, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=85 ms

[EXPERIMENT A7_ROI_HIER_LR_NO_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9580
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=2] C=1, pooled_inner_AUC=0.9720
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=2] C=10, pooled_inner_AUC=0.9720
[OUTER 2] selected_C=1, selected_inner_AUC=0.9720, best_candidate_inner_AUC=0.9720, training_only_threshold=0.276854, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=90 ms

[EXPERIMENT A7_ROI_HIER_LR_NO_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7902
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=3] C=1, pooled_inner_AUC=0.8322
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=3] C=10, pooled_inner_AUC=0.8252
[OUTER 3] selected_C=1, selected_inner_AUC=0.8322, best_candidate_inner_AUC=0.8322, training_only_threshold=0.174164, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=92 ms

[EXPERIMENT A7_ROI_HIER_LR_NO_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8531
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=4] C=1, pooled_inner_AUC=0.8671
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=4] C=10, pooled_inner_AUC=0.8811
[OUTER 4] selected_C=10, selected_inner_AUC=0.8811, best_candidate_inner_AUC=0.8811, training_only_threshold=0.478691, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=89 ms

[EXPERIMENT A7_ROI_HIER_LR_NO_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8741
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8951
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=5] C=1, pooled_inner_AUC=0.8951
[INNER CV][A7_ROI_HIER_LR_NO_PCA][outer=5] C=10, pooled_inner_AUC=0.8951
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.8951, best_candidate_inner_AUC=0.8951, training_only_threshold=0.589469, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=86 ms
[EXPERIMENT] COMPLETED: A7_ROI_HIER_LR_NO_PCA | AUC=0.9107 [0.7722, 1.0000] | AUPRC=0.9263 | Sensitivity=0.8571 | Specificity=0.9375 | F1=0.8889 | runtime=5.72 s

[SUITE] Launching experiment 13/47: A8_ROI_HIER_LR_NO_PCA_FIXEDC

====================================================================================================
[EXPERIMENT] START: A8_ROI_HIER_LR_NO_PCA_FIXEDC
[EXPERIMENT] Matched patient-embedding reference for the legacy slice-classifier ablation: no PCA and fixed C=1.0, so A8 versus A3 isolates the training/pooling strategy rather than changing PCA or C selection.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A8_ROI_HIER_LR_NO_PCA_FIXEDC] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A8_ROI_HIER_LR_NO_PCA_FIXEDC][outer=1] C=1, pooled_inner_AUC=0.8889
[OUTER 1] selected_C=1, selected_inner_AUC=0.8889, best_candidate_inner_AUC=0.8889, training_only_threshold=0.745567, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=34 ms

[EXPERIMENT A8_ROI_HIER_LR_NO_PCA_FIXEDC] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A8_ROI_HIER_LR_NO_PCA_FIXEDC][outer=2] C=1, pooled_inner_AUC=0.9720
[OUTER 2] selected_C=1, selected_inner_AUC=0.9720, best_candidate_inner_AUC=0.9720, training_only_threshold=0.276854, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=33 ms

[EXPERIMENT A8_ROI_HIER_LR_NO_PCA_FIXEDC] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A8_ROI_HIER_LR_NO_PCA_FIXEDC][outer=3] C=1, pooled_inner_AUC=0.8322
[OUTER 3] selected_C=1, selected_inner_AUC=0.8322, best_candidate_inner_AUC=0.8322, training_only_threshold=0.174164, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=33 ms

[EXPERIMENT A8_ROI_HIER_LR_NO_PCA_FIXEDC] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A8_ROI_HIER_LR_NO_PCA_FIXEDC][outer=4] C=1, pooled_inner_AUC=0.8671
[OUTER 4] selected_C=1, selected_inner_AUC=0.8671, best_candidate_inner_AUC=0.8671, training_only_threshold=0.498421, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=34 ms

[EXPERIMENT A8_ROI_HIER_LR_NO_PCA_FIXEDC] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A8_ROI_HIER_LR_NO_PCA_FIXEDC][outer=5] C=1, pooled_inner_AUC=0.8951
[OUTER 5] selected_C=1, selected_inner_AUC=0.8951, best_candidate_inner_AUC=0.8951, training_only_threshold=0.670899, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=34 ms
[EXPERIMENT] COMPLETED: A8_ROI_HIER_LR_NO_PCA_FIXEDC | AUC=0.9375 [0.8348, 1.0000] | AUPRC=0.9446 | Sensitivity=0.8571 | Specificity=0.9375 | F1=0.8889 | runtime=5.36 s

[SUITE] Launching experiment 14/47: B1_STANDARDIZED_ROI_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_monai_roi', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 562 ms.

====================================================================================================
[EXPERIMENT] START: B1_STANDARDIZED_ROI_HIER_LR_PCA
[EXPERIMENT] Primary deconfounded baseline: label-blind dark-padding removal, fixed-content 240-in-256 geometry, MONAI min-max input separated from robust classifier scaling, confidence-gated soft ROI, hierarchical embedding pooling, PCA and Logistic Regression.
[EXPERIMENT] feature_mode=standardized_monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT B1_STANDARDIZED_ROI_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8750
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8889
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8958
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8958
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.8889, best_candidate_inner_AUC=0.8958, training_only_threshold=0.291917, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT B1_STANDARDIZED_ROI_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9371
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9720
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9720
[OUTER 2] selected_C=1, selected_inner_AUC=0.9720, best_candidate_inner_AUC=0.9720, training_only_threshold=0.341551, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=110 ms

[EXPERIMENT B1_STANDARDIZED_ROI_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7552
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7762
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7762
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7902
[OUTER 3] selected_C=10, selected_inner_AUC=0.7902, best_candidate_inner_AUC=0.7902, training_only_threshold=0.862127, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT B1_STANDARDIZED_ROI_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8601
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8462
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8462
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8601, training_only_threshold=0.457156, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=109 ms

[EXPERIMENT B1_STANDARDIZED_ROI_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8531
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8531
[INNER CV][B1_STANDARDIZED_ROI_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8531
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8531, best_candidate_inner_AUC=0.8531, training_only_threshold=0.545501, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=110 ms
[EXPERIMENT] COMPLETED: B1_STANDARDIZED_ROI_HIER_LR_PCA | AUC=0.8259 [0.6561, 0.9510] | AUPRC=0.8305 | Sensitivity=0.6429 | Specificity=0.8125 | F1=0.6923 | runtime=5.75 s

[SUITE] Launching experiment 15/47: A9_STANDARDIZED_FULL_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_full_image', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 533 ms.

====================================================================================================
[EXPERIMENT] START: A9_STANDARDIZED_FULL_HIER_LR_PCA
[EXPERIMENT] Label-blind standardized full-image ablation with all downstream settings matched to the deconfounded baseline.
[EXPERIMENT] feature_mode=standardized_full_image, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A9_STANDARDIZED_FULL_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7847
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.7847
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.7639
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.7708
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7847, best_candidate_inner_AUC=0.7847, training_only_threshold=0.623461, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=110 ms

[EXPERIMENT A9_STANDARDIZED_FULL_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9580
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9650
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9650
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9650
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.9580, best_candidate_inner_AUC=0.9650, training_only_threshold=0.319718, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=115 ms

[EXPERIMENT A9_STANDARDIZED_FULL_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7483
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7203
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7203
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7203
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7483, best_candidate_inner_AUC=0.7483, training_only_threshold=0.468223, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=122 ms

[EXPERIMENT A9_STANDARDIZED_FULL_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8601
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8531
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8671
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8601
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8671, training_only_threshold=0.369717, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=115 ms

[EXPERIMENT A9_STANDARDIZED_FULL_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8462
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8601
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8601
[INNER CV][A9_STANDARDIZED_FULL_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8671
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8671, training_only_threshold=0.550463, inner_splits=3
[OUTER 5] held-out AUC=0.8889; runtime=110 ms
[EXPERIMENT] COMPLETED: A9_STANDARDIZED_FULL_HIER_LR_PCA | AUC=0.8795 [0.7410, 0.9821] | AUPRC=0.9025 | Sensitivity=0.7143 | Specificity=0.9375 | F1=0.8000 | runtime=5.73 s

[SUITE] Launching experiment 16/47: C5_STANDARDIZED_BORDER05_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_border_05', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 539 ms.

====================================================================================================
[EXPERIMENT] START: C5_STANDARDIZED_BORDER05_HIER_LR_PCA
[EXPERIMENT] Negative control retaining only the outer 5% of the standardized image. High AUC indicates residual padding/export shortcut risk.
[EXPERIMENT] feature_mode=standardized_border_05, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C5_STANDARDIZED_BORDER05_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.6875
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.6806
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.6806
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.6736
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.6875, best_candidate_inner_AUC=0.6875, training_only_threshold=0.476998, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT C5_STANDARDIZED_BORDER05_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.6364
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.6224
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.6154
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.6084
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.6364, best_candidate_inner_AUC=0.6364, training_only_threshold=0.539942, inner_splits=3
[OUTER 2] held-out AUC=0.5556; runtime=104 ms

[EXPERIMENT C5_STANDARDIZED_BORDER05_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7692
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7762
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7762
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7692
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7692, best_candidate_inner_AUC=0.7762, training_only_threshold=0.516294, inner_splits=3
[OUTER 3] held-out AUC=0.6667; runtime=106 ms

[EXPERIMENT C5_STANDARDIZED_BORDER05_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.7762
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.7343
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.7273
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.7203
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7762, best_candidate_inner_AUC=0.7762, training_only_threshold=0.358044, inner_splits=3
[OUTER 4] held-out AUC=0.3333; runtime=103 ms

[EXPERIMENT C5_STANDARDIZED_BORDER05_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.4266
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.4056
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.3986
[INNER CV][C5_STANDARDIZED_BORDER05_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.3916
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.4266, best_candidate_inner_AUC=0.4266, training_only_threshold=0.211599, inner_splits=3
[OUTER 5] held-out AUC=0.4444; runtime=105 ms
[EXPERIMENT] COMPLETED: C5_STANDARDIZED_BORDER05_HIER_LR_PCA | AUC=0.5938 [0.3839, 0.7991] | AUPRC=0.6121 | Sensitivity=0.7143 | Specificity=0.5625 | F1=0.6452 | runtime=5.60 s

[SUITE] Launching experiment 17/47: C6_STANDARDIZED_BORDER10_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_border_10', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 538 ms.

====================================================================================================
[EXPERIMENT] START: C6_STANDARDIZED_BORDER10_HIER_LR_PCA
[EXPERIMENT] Negative control retaining only the outer 10% of the standardized image. This is stricter than the original 15% border control.
[EXPERIMENT] feature_mode=standardized_border_10, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C6_STANDARDIZED_BORDER10_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.6389
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.5833
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.5903
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.5903
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.6389, best_candidate_inner_AUC=0.6389, training_only_threshold=0.588381, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=107 ms

[EXPERIMENT C6_STANDARDIZED_BORDER10_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.7133
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.7273
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.7133
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.7203
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.7273, best_candidate_inner_AUC=0.7273, training_only_threshold=0.838080, inner_splits=3
[OUTER 2] held-out AUC=0.6667; runtime=107 ms

[EXPERIMENT C6_STANDARDIZED_BORDER10_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7622
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7413
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7343
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7343
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7622, best_candidate_inner_AUC=0.7622, training_only_threshold=0.644892, inner_splits=3
[OUTER 3] held-out AUC=0.6667; runtime=136 ms

[EXPERIMENT C6_STANDARDIZED_BORDER10_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8042
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8252
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8252
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8252
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.8252, best_candidate_inner_AUC=0.8252, training_only_threshold=0.825011, inner_splits=3
[OUTER 4] held-out AUC=0.4444; runtime=113 ms

[EXPERIMENT C6_STANDARDIZED_BORDER10_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.6434
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.6503
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.6573
[INNER CV][C6_STANDARDIZED_BORDER10_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.6434
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.6503, best_candidate_inner_AUC=0.6573, training_only_threshold=0.731285, inner_splits=3
[OUTER 5] held-out AUC=0.6667; runtime=160 ms
[EXPERIMENT] COMPLETED: C6_STANDARDIZED_BORDER10_HIER_LR_PCA | AUC=0.6920 [0.4732, 0.8884] | AUPRC=0.7751 | Sensitivity=0.5000 | Specificity=0.9375 | F1=0.6364 | runtime=5.70 s

[SUITE] Launching experiment 18/47: C7_DETECTED_PADDING_MASK_HIER_LR_PCA
[SUITE] Prepared data representation ('detected_padding_mask', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 530 ms.

====================================================================================================
[EXPERIMENT] START: C7_DETECTED_PADDING_MASK_HIER_LR_PCA
[EXPERIMENT] Negative control using only the fixed-canvas padding geometry after label-blind native dark-padding removal and content-size normalization.
[EXPERIMENT] feature_mode=detected_padding_mask, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C7_DETECTED_PADDING_MASK_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7431
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.6944
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.6389
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.6389
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7431, best_candidate_inner_AUC=0.7431, training_only_threshold=0.740551, inner_splits=3
[OUTER 1] held-out AUC=0.7500; runtime=106 ms

[EXPERIMENT C7_DETECTED_PADDING_MASK_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.6294
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.6643
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.6643
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.6364
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.6643, best_candidate_inner_AUC=0.6643, training_only_threshold=0.786931, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=106 ms

[EXPERIMENT C7_DETECTED_PADDING_MASK_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.6713
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.6573
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.6503
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.6294
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.6713, best_candidate_inner_AUC=0.6713, training_only_threshold=0.707094, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=104 ms

[EXPERIMENT C7_DETECTED_PADDING_MASK_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.7692
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8042
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8042
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.7972
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.8042, best_candidate_inner_AUC=0.8042, training_only_threshold=0.508461, inner_splits=3
[OUTER 4] held-out AUC=0.5556; runtime=108 ms

[EXPERIMENT C7_DETECTED_PADDING_MASK_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.5874
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.5804
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.6084
[INNER CV][C7_DETECTED_PADDING_MASK_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.6084
[OUTER 5] selected_C=1, selected_inner_AUC=0.6084, best_candidate_inner_AUC=0.6084, training_only_threshold=0.540730, inner_splits=3
[OUTER 5] held-out AUC=0.7778; runtime=109 ms
[EXPERIMENT] COMPLETED: C7_DETECTED_PADDING_MASK_HIER_LR_PCA | AUC=0.7098 [0.5000, 0.8884] | AUPRC=0.7358 | Sensitivity=0.4286 | Specificity=0.8750 | F1=0.5455 | runtime=5.63 s

[SUITE] Launching experiment 19/47: C8_STANDARDIZED_CORNERS_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_corners', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 531 ms.

====================================================================================================
[EXPERIMENT] START: C8_STANDARDIZED_CORNERS_HIER_LR_PCA
[EXPERIMENT] Negative control retaining only four standardized-image corners; it targets scanner overlays, crop geometry, and export templates.
[EXPERIMENT] feature_mode=standardized_corners, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C8_STANDARDIZED_CORNERS_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.5625
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.5764
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.5903
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.5625
[OUTER 1] selected_C=1, selected_inner_AUC=0.5903, best_candidate_inner_AUC=0.5903, training_only_threshold=0.095100, inner_splits=3
[OUTER 1] held-out AUC=0.8750; runtime=110 ms

[EXPERIMENT C8_STANDARDIZED_CORNERS_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.6713
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.6713
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.6573
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.7063
[OUTER 2] selected_C=10, selected_inner_AUC=0.7063, best_candidate_inner_AUC=0.7063, training_only_threshold=0.873537, inner_splits=3
[OUTER 2] held-out AUC=0.4444; runtime=110 ms

[EXPERIMENT C8_STANDARDIZED_CORNERS_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.5524
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.5385
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.4685
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.4615
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.5524, best_candidate_inner_AUC=0.5524, training_only_threshold=0.405769, inner_splits=3
[OUTER 3] held-out AUC=0.7778; runtime=114 ms

[EXPERIMENT C8_STANDARDIZED_CORNERS_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.6014
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.4545
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.3916
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.3636
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.6014, best_candidate_inner_AUC=0.6014, training_only_threshold=0.668591, inner_splits=3
[OUTER 4] held-out AUC=0.4444; runtime=111 ms

[EXPERIMENT C8_STANDARDIZED_CORNERS_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.5105
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.5175
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.5245
[INNER CV][C8_STANDARDIZED_CORNERS_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.5245
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.5175, best_candidate_inner_AUC=0.5245, training_only_threshold=0.178560, inner_splits=3
[OUTER 5] held-out AUC=0.4444; runtime=109 ms
[EXPERIMENT] COMPLETED: C8_STANDARDIZED_CORNERS_HIER_LR_PCA | AUC=0.5714 [0.3661, 0.7769] | AUPRC=0.5886 | Sensitivity=0.5000 | Specificity=0.5625 | F1=0.5000 | runtime=5.67 s

[SUITE] Launching experiment 20/47: C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_center_crop', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 519 ms.

====================================================================================================
[EXPERIMENT] START: C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA
[EXPERIMENT] Area-matched central-crop control. It tests whether MONAI adds anatomical localization beyond simply concentrating on the center.
[EXPERIMENT] feature_mode=standardized_center_crop, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9028
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8889
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8958
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9097
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.9028, best_candidate_inner_AUC=0.9097, training_only_threshold=0.528317, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=105 ms

[EXPERIMENT C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9650
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9650
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9580
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.9650, best_candidate_inner_AUC=0.9650, training_only_threshold=0.501750, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=105 ms

[EXPERIMENT C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7413
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7483
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7343
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7343
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7413, best_candidate_inner_AUC=0.7483, training_only_threshold=0.689584, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=106 ms

[EXPERIMENT C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9580
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9580
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9580, training_only_threshold=0.706901, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=108 ms

[EXPERIMENT C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.9091
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.9091
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.9091
[INNER CV][C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8951
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.9091, best_candidate_inner_AUC=0.9091, training_only_threshold=0.438268, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=104 ms
[EXPERIMENT] COMPLETED: C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA | AUC=0.9241 [0.7857, 1.0000] | AUPRC=0.9426 | Sensitivity=0.9286 | Specificity=0.9375 | F1=0.9286 | runtime=5.66 s

[SUITE] Launching experiment 21/47: A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_roi_zero_background', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 542 ms.

====================================================================================================
[EXPERIMENT] START: A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA
[EXPERIMENT] Strict standardized soft ROI with zero background for valid MONAI masks and full-image fallback for invalid masks.
[EXPERIMENT] feature_mode=standardized_roi_zero_background, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9375
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9444
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.9306
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9167
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.9375, best_candidate_inner_AUC=0.9444, training_only_threshold=0.556726, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=106 ms

[EXPERIMENT A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9860
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9930
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=1.0000
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=1.0000
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.9930, best_candidate_inner_AUC=1.0000, training_only_threshold=0.449859, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=105 ms

[EXPERIMENT A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8322
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8322
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8322
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8252
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.8322, best_candidate_inner_AUC=0.8322, training_only_threshold=0.484458, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=103 ms

[EXPERIMENT A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9441
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9580
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9580
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.9580, best_candidate_inner_AUC=0.9580, training_only_threshold=0.533446, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=109 ms

[EXPERIMENT A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.9580
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.9650
[INNER CV][A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.9650
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.9580, best_candidate_inner_AUC=0.9650, training_only_threshold=0.398769, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=112 ms
[EXPERIMENT] COMPLETED: A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA | AUC=0.9286 [0.8214, 1.0000] | AUPRC=0.9468 | Sensitivity=0.8571 | Specificity=0.8750 | F1=0.8571 | runtime=5.63 s

[SUITE] Launching experiment 22/47: A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_roi_bbox', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 556 ms.

====================================================================================================
[EXPERIMENT] START: A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA
[EXPERIMENT] Strict standardized crop around the dilated MONAI hard-mask bounding box with a fixed label-blind context margin.
[EXPERIMENT] feature_mode=standardized_roi_bbox, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9028
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9167
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.9167
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9167
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.9167, best_candidate_inner_AUC=0.9167, training_only_threshold=0.375076, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=108 ms

[EXPERIMENT A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9580
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9650
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9650
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.9580, best_candidate_inner_AUC=0.9650, training_only_threshold=0.309551, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=112 ms

[EXPERIMENT A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7273
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.6923
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.6923
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.6853
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7273, best_candidate_inner_AUC=0.7273, training_only_threshold=0.631978, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=108 ms

[EXPERIMENT A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9580
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9580
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9580, training_only_threshold=0.602308, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=107 ms

[EXPERIMENT A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.9091
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.9091
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.9161
[INNER CV][A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.9231
[OUTER 5] selected_C=1, selected_inner_AUC=0.9161, best_candidate_inner_AUC=0.9231, training_only_threshold=0.737761, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=108 ms
[EXPERIMENT] COMPLETED: A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA | AUC=0.9375 [0.8170, 1.0000] | AUPRC=0.9520 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.71 s

[SUITE] Launching experiment 23/47: C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_outside_large_bbox', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 534 ms.

====================================================================================================
[EXPERIMENT] START: C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA
[EXPERIMENT] Strict control retaining only pixels outside an enlarged MONAI ventricular bounding box; invalid masks yield a zero image rather than a full-image fallback. This is not a validated outside-heart mask.
[EXPERIMENT] feature_mode=standardized_outside_large_bbox, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7083
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.7083
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.7014
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.7083
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7083, best_candidate_inner_AUC=0.7083, training_only_threshold=0.586766, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=110 ms

[EXPERIMENT C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8601
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.8671
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.8671
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.8811
[OUTER 2] selected_C=10, selected_inner_AUC=0.8811, best_candidate_inner_AUC=0.8811, training_only_threshold=0.626105, inner_splits=3
[OUTER 2] held-out AUC=0.6667; runtime=109 ms

[EXPERIMENT C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8182
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8601
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8601
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8671
[OUTER 3] selected_C=0.1, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8671, training_only_threshold=0.605083, inner_splits=3
[OUTER 3] held-out AUC=0.7778; runtime=110 ms

[EXPERIMENT C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8322
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8392
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8322
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8322
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8322, best_candidate_inner_AUC=0.8392, training_only_threshold=0.601069, inner_splits=3
[OUTER 4] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.7413
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.7692
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.7902
[INNER CV][C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.7972
[OUTER 5] selected_C=1, selected_inner_AUC=0.7902, best_candidate_inner_AUC=0.7972, training_only_threshold=0.167407, inner_splits=3
[OUTER 5] held-out AUC=0.8889; runtime=109 ms
[EXPERIMENT] COMPLETED: C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA | AUC=0.8259 [0.6384, 0.9732] | AUPRC=0.8613 | Sensitivity=0.7857 | Specificity=0.8750 | F1=0.8148 | runtime=5.76 s

[SUITE] Launching experiment 24/47: C11_STANDARDIZATION_QC_ONLY_LR
[SUITE] Prepared data representation ('standardization_qc_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C11_STANDARDIZATION_QC_ONLY_LR
[EXPERIMENT] Patient classifier using only label-blind crop fractions and robust intensity-scaling limits generated by standardization.
[EXPERIMENT] feature_mode=standardization_qc_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C11_STANDARDIZATION_QC_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.5139
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.3264
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.2153
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.2431
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.5139, best_candidate_inner_AUC=0.5139, training_only_threshold=0.526558, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=54 ms

[EXPERIMENT C11_STANDARDIZATION_QC_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.7483
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.7483
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.7413
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.7343
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.7483, best_candidate_inner_AUC=0.7483, training_only_threshold=0.500484, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=52 ms

[EXPERIMENT C11_STANDARDIZATION_QC_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.7762
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.7343
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.7413
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.7273
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7762, best_candidate_inner_AUC=0.7762, training_only_threshold=0.548137, inner_splits=3
[OUTER 3] held-out AUC=0.6667; runtime=53 ms

[EXPERIMENT C11_STANDARDIZATION_QC_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.7203
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.6923
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.6573
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.6573
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7203, best_candidate_inner_AUC=0.7203, training_only_threshold=0.592518, inner_splits=3
[OUTER 4] held-out AUC=0.5556; runtime=56 ms

[EXPERIMENT C11_STANDARDIZATION_QC_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.6434
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.6434
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.5594
[INNER CV][C11_STANDARDIZATION_QC_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.5524
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.6434, best_candidate_inner_AUC=0.6434, training_only_threshold=0.475625, inner_splits=3
[OUTER 5] held-out AUC=0.7778; runtime=56 ms
[EXPERIMENT] COMPLETED: C11_STANDARDIZATION_QC_ONLY_LR | AUC=0.7277 [0.5267, 0.8885] | AUPRC=0.7621 | Sensitivity=0.5000 | Specificity=0.6875 | F1=0.5385 | runtime=5.51 s

[SUITE] Launching experiment 25/47: C12_STANDARDIZED_MONAI_QC_ONLY_LR
[SUITE] Prepared data representation ('standardized_monai_qc_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C12_STANDARDIZED_MONAI_QC_ONLY_LR
[EXPERIMENT] Patient classifier using only MONAI gate/QC statistics produced after label-blind standardization.
[EXPERIMENT] feature_mode=standardized_monai_qc_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C12_STANDARDIZED_MONAI_QC_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.6181
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.6181
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.5556
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.5139
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.6181, best_candidate_inner_AUC=0.6181, training_only_threshold=0.510009, inner_splits=3
[OUTER 1] held-out AUC=0.8750; runtime=51 ms

[EXPERIMENT C12_STANDARDIZED_MONAI_QC_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.6154
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.5874
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.6014
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.6364
[OUTER 2] selected_C=10, selected_inner_AUC=0.6364, best_candidate_inner_AUC=0.6364, training_only_threshold=0.796648, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=51 ms

[EXPERIMENT C12_STANDARDIZED_MONAI_QC_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.7273
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.7622
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.7972
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.7902
[OUTER 3] selected_C=1, selected_inner_AUC=0.7972, best_candidate_inner_AUC=0.7972, training_only_threshold=0.477089, inner_splits=3
[OUTER 3] held-out AUC=0.2222; runtime=50 ms

[EXPERIMENT C12_STANDARDIZED_MONAI_QC_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.5315
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.5734
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.6154
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.6713
[OUTER 4] selected_C=10, selected_inner_AUC=0.6713, best_candidate_inner_AUC=0.6713, training_only_threshold=0.568583, inner_splits=3
[OUTER 4] held-out AUC=0.6667; runtime=52 ms

[EXPERIMENT C12_STANDARDIZED_MONAI_QC_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.6783
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.6923
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.6853
[INNER CV][C12_STANDARDIZED_MONAI_QC_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.7832
[OUTER 5] selected_C=10, selected_inner_AUC=0.7832, best_candidate_inner_AUC=0.7832, training_only_threshold=0.620611, inner_splits=3
[OUTER 5] held-out AUC=0.6667; runtime=51 ms
[EXPERIMENT] COMPLETED: C12_STANDARDIZED_MONAI_QC_ONLY_LR | AUC=0.6250 [0.4018, 0.8080] | AUPRC=0.6848 | Sensitivity=0.5000 | Specificity=0.7500 | F1=0.5600 | runtime=5.46 s

[SUITE] Launching experiment 26/47: C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_fixed_center_50', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 519 ms.

====================================================================================================
[EXPERIMENT] START: C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA
[EXPERIMENT] Fixed 50% central crop after label-blind geometry standardization. Crop size is independent of MONAI masks, labels and model scores.
[EXPERIMENT] feature_mode=standardized_fixed_center_50, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8472
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8333
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8194
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8056
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8472, best_candidate_inner_AUC=0.8472, training_only_threshold=0.487441, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9650
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9580
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9510
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.9650, best_candidate_inner_AUC=0.9650, training_only_threshold=0.630811, inner_splits=3
[OUTER 2] held-out AUC=0.6667; runtime=126 ms

[EXPERIMENT C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.6993
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.6853
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.6573
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.6364
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.6993, best_candidate_inner_AUC=0.6993, training_only_threshold=0.479172, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=114 ms

[EXPERIMENT C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8392
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8392
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8531, best_candidate_inner_AUC=0.8531, training_only_threshold=0.627312, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=105 ms

[EXPERIMENT C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8531
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8531
[INNER CV][C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8601
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8531, best_candidate_inner_AUC=0.8601, training_only_threshold=0.320576, inner_splits=3
[OUTER 5] held-out AUC=0.8889; runtime=105 ms
[EXPERIMENT] COMPLETED: C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA | AUC=0.8929 [0.7634, 0.9821] | AUPRC=0.8749 | Sensitivity=0.6429 | Specificity=0.8125 | F1=0.6923 | runtime=5.67 s

[SUITE] Launching experiment 27/47: C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_fixed_center_60', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 559 ms.

====================================================================================================
[EXPERIMENT] START: C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA
[EXPERIMENT] Fixed 60% central crop after label-blind geometry standardization. This is the predeclared simple-localization candidate for stability analysis.
[EXPERIMENT] feature_mode=standardized_fixed_center_60, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8194
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8056
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.7917
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8056
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8194, best_candidate_inner_AUC=0.8194, training_only_threshold=0.573088, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9441
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9650
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9650
[OUTER 2] selected_C=1, selected_inner_AUC=0.9650, best_candidate_inner_AUC=0.9650, training_only_threshold=0.200680, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=106 ms

[EXPERIMENT C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.6993
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.6923
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.6783
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.6853
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.6993, best_candidate_inner_AUC=0.6993, training_only_threshold=0.714612, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=106 ms

[EXPERIMENT C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8671
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8671
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8671
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8671
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8671, best_candidate_inner_AUC=0.8671, training_only_threshold=0.535140, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=111 ms

[EXPERIMENT C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8881
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.9021
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.9161
[INNER CV][C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.9161
[OUTER 5] selected_C=1, selected_inner_AUC=0.9161, best_candidate_inner_AUC=0.9161, training_only_threshold=0.269985, inner_splits=3
[OUTER 5] held-out AUC=0.7778; runtime=108 ms
[EXPERIMENT] COMPLETED: C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA | AUC=0.8795 [0.7232, 0.9955] | AUPRC=0.8493 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.83 s

[SUITE] Launching experiment 28/47: C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_fixed_center_70', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 543 ms.

====================================================================================================
[EXPERIMENT] START: C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA
[EXPERIMENT] Fixed 70% central crop after label-blind geometry standardization. It tests whether retaining more central context changes ranking.
[EXPERIMENT] feature_mode=standardized_fixed_center_70, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7778
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.7708
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.7847
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.7986
[OUTER 1] selected_C=10, selected_inner_AUC=0.7986, best_candidate_inner_AUC=0.7986, training_only_threshold=0.224333, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=117 ms

[EXPERIMENT C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9371
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9650
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9650
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9790
[OUTER 2] selected_C=10, selected_inner_AUC=0.9790, best_candidate_inner_AUC=0.9790, training_only_threshold=0.058243, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=109 ms

[EXPERIMENT C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7343
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7063
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7203
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7273
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7343, best_candidate_inner_AUC=0.7343, training_only_threshold=0.541001, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=107 ms

[EXPERIMENT C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8462
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8462
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8462
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8462, best_candidate_inner_AUC=0.8462, training_only_threshold=0.455647, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=110 ms

[EXPERIMENT C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8601
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8741
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8881
[INNER CV][C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8951
[OUTER 5] selected_C=1, selected_inner_AUC=0.8881, best_candidate_inner_AUC=0.8951, training_only_threshold=0.118372, inner_splits=3
[OUTER 5] held-out AUC=0.6667; runtime=109 ms
[EXPERIMENT] COMPLETED: C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA | AUC=0.8438 [0.6786, 0.9643] | AUPRC=0.8185 | Sensitivity=0.7857 | Specificity=0.6875 | F1=0.7333 | runtime=5.84 s

[SUITE] Launching experiment 29/47: A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_roi_zero_bg_center_fallback', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 543 ms.

====================================================================================================
[EXPERIMENT] START: A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA
[EXPERIMENT] Zero-background standardized MONAI ROI for valid masks, with a fixed 60% center crop rather than the full image when the gate fails.
[EXPERIMENT] feature_mode=standardized_roi_zero_bg_center_fallback, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9306
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9167
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.9167
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9167
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.9306, best_candidate_inner_AUC=0.9306, training_only_threshold=0.691342, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=115 ms

[EXPERIMENT A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9930
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9930
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=1.0000
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=1.0000
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.9930, best_candidate_inner_AUC=1.0000, training_only_threshold=0.400703, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=113 ms

[EXPERIMENT A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8252
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8252
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8252
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8392
[OUTER 3] selected_C=10, selected_inner_AUC=0.8392, best_candidate_inner_AUC=0.8392, training_only_threshold=0.228468, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9441
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9580
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9580
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9580, training_only_threshold=0.563544, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=108 ms

[EXPERIMENT A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.9580
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.9650
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.9720
[INNER CV][A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.9720
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.9650, best_candidate_inner_AUC=0.9720, training_only_threshold=0.390117, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=107 ms
[EXPERIMENT] COMPLETED: A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA | AUC=0.9643 [0.8929, 1.0000] | AUPRC=0.9662 | Sensitivity=0.8571 | Specificity=0.9375 | F1=0.8889 | runtime=5.84 s

[SUITE] Launching experiment 30/47: A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA
[SUITE] Prepared data representation ('standardized_monai_roi', 'patient_embedding', 'fixed_chunk', 'equal', 0.0, False) in 557 ms.

====================================================================================================
[EXPERIMENT] START: A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA
[EXPERIMENT] Folder-independent fixed-size chunk pooling over standardized ROI slice order. It tests whether SR_*/series* export partitioning itself creates the apparent advantage of hierarchical folder pooling.
[EXPERIMENT] feature_mode=standardized_monai_roi, strategy=patient_embedding, pooling=fixed_chunk, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8403
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8611
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8542
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8403
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.8611, best_candidate_inner_AUC=0.8611, training_only_threshold=0.565667, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8042
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.8182
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.8042
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.8182, best_candidate_inner_AUC=0.8182, training_only_threshold=0.659763, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=107 ms

[EXPERIMENT A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.7413
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7063
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.6643
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.6573
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.7413, best_candidate_inner_AUC=0.7413, training_only_threshold=0.544070, inner_splits=3
[OUTER 3] held-out AUC=0.8889; runtime=108 ms

[EXPERIMENT A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8182
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8182
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8252
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.8182, best_candidate_inner_AUC=0.8252, training_only_threshold=0.741247, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=121 ms

[EXPERIMENT A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.7622
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.7343
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.7343
[INNER CV][A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.7343
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.7622, best_candidate_inner_AUC=0.7622, training_only_threshold=0.425673, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=121 ms
[EXPERIMENT] COMPLETED: A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA | AUC=0.8393 [0.6786, 0.9643] | AUPRC=0.8654 | Sensitivity=0.6429 | Specificity=0.8125 | F1=0.6923 | runtime=5.78 s

[SUITE] Launching experiment 31/47: A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA
[SUITE] Prepared data representation ('standardized_roi_zero_bg_center_fallback', 'patient_embedding', 'fixed_chunk', 'equal', 0.0, False) in 554 ms.

====================================================================================================
[EXPERIMENT] START: A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA
[EXPERIMENT] Primary-candidate pixels with folder-independent fixed-size chunk pooling. This isolates the effect of SR_*/series* proxy boundaries while keeping the strict ROI/fallback representation unchanged.
[EXPERIMENT] feature_mode=standardized_roi_zero_bg_center_fallback, strategy=patient_embedding, pooling=fixed_chunk, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9444
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9653
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.9653
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9722
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.9653, best_candidate_inner_AUC=0.9722, training_only_threshold=0.453595, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=108 ms

[EXPERIMENT A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9161
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9231
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9301
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9441
[OUTER 2] selected_C=10, selected_inner_AUC=0.9441, best_candidate_inner_AUC=0.9441, training_only_threshold=0.096732, inner_splits=3
[OUTER 2] held-out AUC=1.0000; runtime=106 ms

[EXPERIMENT A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8741
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8531
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8462
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8462
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.8741, best_candidate_inner_AUC=0.8741, training_only_threshold=0.507163, inner_splits=3
[OUTER 3] held-out AUC=0.8889; runtime=102 ms

[EXPERIMENT A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9510
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9580
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9580, training_only_threshold=0.359319, inner_splits=3
[OUTER 4] held-out AUC=1.0000; runtime=107 ms

[EXPERIMENT A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.7762
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8112
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8112
[INNER CV][A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8112
[OUTER 5] selected_C=0.1, selected_inner_AUC=0.8112, best_candidate_inner_AUC=0.8112, training_only_threshold=0.210854, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=108 ms
[EXPERIMENT] COMPLETED: A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA | AUC=0.9732 [0.9107, 1.0000] | AUPRC=0.9713 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.75 s

[SUITE] Launching experiment 32/47: A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA
[EXACT DEDUP] A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA: retained 62160/63425 slices; removed=1265 repeated exact exports.
[SUITE] Prepared data representation ('standardized_roi_zero_bg_center_fallback', 'patient_embedding', 'hierarchical', 'equal', 0.0, True) in 689 ms.

====================================================================================================
[EXPERIMENT] START: A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA
[EXPERIMENT] Primary-candidate pixels after deterministic exact decoded-pixel deduplication inside each Directory_* patient. Hierarchical pooling is otherwise unchanged, so repeated identical exports cannot receive multiple nominal contributions.
[EXPERIMENT] feature_mode=standardized_roi_zero_bg_center_fallback, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=True
====================================================================================================

[EXPERIMENT A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9167
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9167
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.9167
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9097
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.9167, best_candidate_inner_AUC=0.9167, training_only_threshold=0.500455, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=110 ms

[EXPERIMENT A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9790
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9930
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9930
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9930
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.9930, best_candidate_inner_AUC=0.9930, training_only_threshold=0.324565, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=108 ms

[EXPERIMENT A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8252
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8252
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8252
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8182
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.8252, best_candidate_inner_AUC=0.8252, training_only_threshold=0.433379, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9441
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9580
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9580
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9580
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.9580, best_candidate_inner_AUC=0.9580, training_only_threshold=0.530052, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=111 ms

[EXPERIMENT A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.9301
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.9301
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.9371
[INNER CV][A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.9441
[OUTER 5] selected_C=1, selected_inner_AUC=0.9371, best_candidate_inner_AUC=0.9441, training_only_threshold=0.329079, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=111 ms
[EXPERIMENT] COMPLETED: A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA | AUC=0.9464 [0.8527, 1.0000] | AUPRC=0.9560 | Sensitivity=0.8571 | Specificity=0.9375 | F1=0.8889 | runtime=5.76 s

[SUITE] Launching experiment 33/47: C16_N_SLICES_ONLY_LR
[SUITE] Prepared data representation ('n_slices_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C16_N_SLICES_ONLY_LR
[EXPERIMENT] Negative control using only the number of exported slices per Directory_* patient.
[EXPERIMENT] feature_mode=n_slices_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C16_N_SLICES_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C16_N_SLICES_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.6389
[INNER CV][C16_N_SLICES_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.6389
[INNER CV][C16_N_SLICES_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.6319
[INNER CV][C16_N_SLICES_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.6181
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.6389, best_candidate_inner_AUC=0.6389, training_only_threshold=0.500742, inner_splits=3
[OUTER 1] held-out AUC=0.5000; runtime=56 ms

[EXPERIMENT C16_N_SLICES_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C16_N_SLICES_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.6713
[INNER CV][C16_N_SLICES_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.6713
[INNER CV][C16_N_SLICES_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.6643
[INNER CV][C16_N_SLICES_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.6713
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.6713, best_candidate_inner_AUC=0.6713, training_only_threshold=0.501612, inner_splits=3
[OUTER 2] held-out AUC=0.4444; runtime=55 ms

[EXPERIMENT C16_N_SLICES_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C16_N_SLICES_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.4266
[INNER CV][C16_N_SLICES_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.4266
[INNER CV][C16_N_SLICES_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.4266
[INNER CV][C16_N_SLICES_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.4266
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.4266, best_candidate_inner_AUC=0.4266, training_only_threshold=0.506539, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=53 ms

[EXPERIMENT C16_N_SLICES_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C16_N_SLICES_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.5245
[INNER CV][C16_N_SLICES_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.5245
[INNER CV][C16_N_SLICES_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.5245
[INNER CV][C16_N_SLICES_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.5245
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.5245, best_candidate_inner_AUC=0.5245, training_only_threshold=0.508118, inner_splits=3
[OUTER 4] held-out AUC=0.6667; runtime=58 ms

[EXPERIMENT C16_N_SLICES_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C16_N_SLICES_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.7622
[INNER CV][C16_N_SLICES_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.7622
[INNER CV][C16_N_SLICES_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.7622
[INNER CV][C16_N_SLICES_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.7622
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.7622, best_candidate_inner_AUC=0.7622, training_only_threshold=0.502001, inner_splits=3
[OUTER 5] held-out AUC=0.2222; runtime=53 ms
[EXPERIMENT] COMPLETED: C16_N_SLICES_ONLY_LR | AUC=0.6027 [0.3883, 0.8125] | AUPRC=0.6418 | Sensitivity=0.3571 | Specificity=0.6875 | F1=0.4167 | runtime=5.55 s

[SUITE] Launching experiment 34/47: C17_N_SERIES_ONLY_LR
[SUITE] Prepared data representation ('n_series_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C17_N_SERIES_ONLY_LR
[EXPERIMENT] Negative control using only the number of folder-defined series proxies per patient.
[EXPERIMENT] feature_mode=n_series_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C17_N_SERIES_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C17_N_SERIES_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.6736
[INNER CV][C17_N_SERIES_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.6736
[INNER CV][C17_N_SERIES_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.6806
[INNER CV][C17_N_SERIES_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.6806
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.6736, best_candidate_inner_AUC=0.6806, training_only_threshold=0.499832, inner_splits=3
[OUTER 1] held-out AUC=0.8750; runtime=52 ms

[EXPERIMENT C17_N_SERIES_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C17_N_SERIES_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.7727
[INNER CV][C17_N_SERIES_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.7727
[INNER CV][C17_N_SERIES_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.7727
[INNER CV][C17_N_SERIES_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.7517
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.7727, best_candidate_inner_AUC=0.7727, training_only_threshold=0.495067, inner_splits=3
[OUTER 2] held-out AUC=0.3333; runtime=51 ms

[EXPERIMENT C17_N_SERIES_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C17_N_SERIES_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.6049
[INNER CV][C17_N_SERIES_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.6049
[INNER CV][C17_N_SERIES_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.6049
[INNER CV][C17_N_SERIES_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.5909
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.6049, best_candidate_inner_AUC=0.6049, training_only_threshold=0.499641, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=50 ms

[EXPERIMENT C17_N_SERIES_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C17_N_SERIES_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.6294
[INNER CV][C17_N_SERIES_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.6224
[INNER CV][C17_N_SERIES_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.6154
[INNER CV][C17_N_SERIES_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.6224
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.6294, best_candidate_inner_AUC=0.6294, training_only_threshold=0.501399, inner_splits=3
[OUTER 4] held-out AUC=0.9444; runtime=51 ms

[EXPERIMENT C17_N_SERIES_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C17_N_SERIES_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.7902
[INNER CV][C17_N_SERIES_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.7902
[INNER CV][C17_N_SERIES_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.7762
[INNER CV][C17_N_SERIES_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.7762
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.7902, best_candidate_inner_AUC=0.7902, training_only_threshold=0.497284, inner_splits=3
[OUTER 5] held-out AUC=0.1111; runtime=57 ms
[EXPERIMENT] COMPLETED: C17_N_SERIES_ONLY_LR | AUC=0.6987 [0.5022, 0.8728] | AUPRC=0.7121 | Sensitivity=0.6429 | Specificity=0.6875 | F1=0.6429 | runtime=5.50 s

[SUITE] Launching experiment 35/47: C18_SERIES_LENGTH_ONLY_LR
[SUITE] Prepared data representation ('series_length_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C18_SERIES_LENGTH_ONLY_LR
[EXPERIMENT] Negative control using only mean and maximum folder-proxy lengths.
[EXPERIMENT] feature_mode=series_length_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C18_SERIES_LENGTH_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.3403
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.3333
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.3889
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.4444
[OUTER 1] selected_C=10, selected_inner_AUC=0.4444, best_candidate_inner_AUC=0.4444, training_only_threshold=0.554468, inner_splits=3
[OUTER 1] held-out AUC=0.7500; runtime=52 ms

[EXPERIMENT C18_SERIES_LENGTH_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.4965
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.5035
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.5315
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.5385
[OUTER 2] selected_C=1, selected_inner_AUC=0.5315, best_candidate_inner_AUC=0.5385, training_only_threshold=0.559997, inner_splits=3
[OUTER 2] held-out AUC=0.6667; runtime=53 ms

[EXPERIMENT C18_SERIES_LENGTH_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.3427
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.3566
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.4755
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.4965
[OUTER 3] selected_C=10, selected_inner_AUC=0.4965, best_candidate_inner_AUC=0.4965, training_only_threshold=0.674019, inner_splits=3
[OUTER 3] held-out AUC=0.7778; runtime=49 ms

[EXPERIMENT C18_SERIES_LENGTH_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.1678
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.1818
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.1399
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.1189
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.1818, best_candidate_inner_AUC=0.1818, training_only_threshold=0.420223, inner_splits=3
[OUTER 4] held-out AUC=1.0000; runtime=52 ms

[EXPERIMENT C18_SERIES_LENGTH_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.6084
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.6573
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.7063
[INNER CV][C18_SERIES_LENGTH_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.6993
[OUTER 5] selected_C=1, selected_inner_AUC=0.7063, best_candidate_inner_AUC=0.7063, training_only_threshold=0.491841, inner_splits=3
[OUTER 5] held-out AUC=0.1111; runtime=50 ms
[EXPERIMENT] COMPLETED: C18_SERIES_LENGTH_ONLY_LR | AUC=0.5625 [0.3438, 0.7634] | AUPRC=0.5942 | Sensitivity=0.5714 | Specificity=0.5625 | F1=0.5517 | runtime=5.48 s

[SUITE] Launching experiment 36/47: C19_NATIVE_GEOMETRY_ONLY_LR
[SUITE] Prepared data representation ('native_geometry_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C19_NATIVE_GEOMETRY_ONLY_LR
[EXPERIMENT] Negative control using only patient-aggregated native height, width and aspect-ratio features; image counts and file size are excluded.
[EXPERIMENT] feature_mode=native_geometry_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C19_NATIVE_GEOMETRY_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.7014
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.6806
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.6458
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.5278
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7014, best_candidate_inner_AUC=0.7014, training_only_threshold=0.539858, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=51 ms

[EXPERIMENT C19_NATIVE_GEOMETRY_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.8182
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.8042
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.7343
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.7063
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.8182, best_candidate_inner_AUC=0.8182, training_only_threshold=0.536137, inner_splits=3
[OUTER 2] held-out AUC=0.6667; runtime=51 ms

[EXPERIMENT C19_NATIVE_GEOMETRY_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.6993
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.6224
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.5105
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.4545
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.6993, best_candidate_inner_AUC=0.6993, training_only_threshold=0.513338, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=50 ms

[EXPERIMENT C19_NATIVE_GEOMETRY_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.7343
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.6993
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.6224
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.5385
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7343, best_candidate_inner_AUC=0.7343, training_only_threshold=0.524193, inner_splits=3
[OUTER 4] held-out AUC=0.6667; runtime=53 ms

[EXPERIMENT C19_NATIVE_GEOMETRY_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.8392
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.8112
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.7413
[INNER CV][C19_NATIVE_GEOMETRY_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.6434
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8392, best_candidate_inner_AUC=0.8392, training_only_threshold=0.527420, inner_splits=3
[OUTER 5] held-out AUC=0.5556; runtime=54 ms
[EXPERIMENT] COMPLETED: C19_NATIVE_GEOMETRY_ONLY_LR | AUC=0.7902 [0.5893, 0.9465] | AUPRC=0.8577 | Sensitivity=0.6429 | Specificity=0.9375 | F1=0.7500 | runtime=5.37 s

[SUITE] Launching experiment 37/47: C20_FILE_SIZE_ONLY_LR
[SUITE] Prepared data representation ('file_size_only', 'patient_tabular', 'patient_tabular', 'not_applicable', 0.0, False) in 0 ms.

====================================================================================================
[EXPERIMENT] START: C20_FILE_SIZE_ONLY_LR
[EXPERIMENT] Negative control using only patient-aggregated file-size and bytes-per-native-pixel statistics; geometry and image counts are excluded.
[EXPERIMENT] feature_mode=file_size_only, strategy=patient_tabular, pooling=patient_tabular, weighting=not_applicable, classifier=logistic_regression, PCA=False, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C20_FILE_SIZE_ONLY_LR] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=1] C=0.01, pooled_inner_AUC=0.7778
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=1] C=0.1, pooled_inner_AUC=0.7639
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=1] C=1, pooled_inner_AUC=0.7292
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=1] C=10, pooled_inner_AUC=0.7431
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7778, best_candidate_inner_AUC=0.7778, training_only_threshold=0.513464, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=55 ms

[EXPERIMENT C20_FILE_SIZE_ONLY_LR] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=2] C=0.01, pooled_inner_AUC=0.8392
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=2] C=0.1, pooled_inner_AUC=0.8322
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=2] C=1, pooled_inner_AUC=0.8462
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=2] C=10, pooled_inner_AUC=0.8671
[OUTER 2] selected_C=10, selected_inner_AUC=0.8671, best_candidate_inner_AUC=0.8671, training_only_threshold=0.741681, inner_splits=3
[OUTER 2] held-out AUC=0.5556; runtime=48 ms

[EXPERIMENT C20_FILE_SIZE_ONLY_LR] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=3] C=0.01, pooled_inner_AUC=0.7832
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=3] C=0.1, pooled_inner_AUC=0.7902
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=3] C=1, pooled_inner_AUC=0.8112
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=3] C=10, pooled_inner_AUC=0.8112
[OUTER 3] selected_C=1, selected_inner_AUC=0.8112, best_candidate_inner_AUC=0.8112, training_only_threshold=0.508693, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=77 ms

[EXPERIMENT C20_FILE_SIZE_ONLY_LR] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=4] C=0.01, pooled_inner_AUC=0.7552
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=4] C=0.1, pooled_inner_AUC=0.7483
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=4] C=1, pooled_inner_AUC=0.7203
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=4] C=10, pooled_inner_AUC=0.7622
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7552, best_candidate_inner_AUC=0.7622, training_only_threshold=0.501975, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=53 ms

[EXPERIMENT C20_FILE_SIZE_ONLY_LR] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=5] C=0.01, pooled_inner_AUC=0.8392
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=5] C=0.1, pooled_inner_AUC=0.8322
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=5] C=1, pooled_inner_AUC=0.7133
[INNER CV][C20_FILE_SIZE_ONLY_LR][outer=5] C=10, pooled_inner_AUC=0.6853
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8392, best_candidate_inner_AUC=0.8392, training_only_threshold=0.527768, inner_splits=3
[OUTER 5] held-out AUC=0.5556; runtime=51 ms
[EXPERIMENT] COMPLETED: C20_FILE_SIZE_ONLY_LR | AUC=0.7679 [0.5759, 0.9331] | AUPRC=0.7055 | Sensitivity=0.7143 | Specificity=0.6875 | F1=0.6897 | runtime=5.48 s

[SUITE] Launching experiment 38/47: C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_soft_monai_mask_only', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 531 ms.

====================================================================================================
[EXPERIMENT] START: C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA
[EXPERIMENT] Segmentation-derived control encoding only the standardized dilated soft MONAI probability map for gate-valid slices; invalid slices are all zero. It tests whether mask shape/confidence alone predicts the class.
[EXPERIMENT] feature_mode=standardized_soft_monai_mask_only, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8472
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.7986
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.7639
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.7361
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8472, best_candidate_inner_AUC=0.8472, training_only_threshold=0.605212, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=116 ms

[EXPERIMENT C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8671
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.8811
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.8671
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.8671
[OUTER 2] selected_C=0.1, selected_inner_AUC=0.8811, best_candidate_inner_AUC=0.8811, training_only_threshold=0.882848, inner_splits=3
[OUTER 2] held-out AUC=1.0000; runtime=113 ms

[EXPERIMENT C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.9510
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.9510
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9510, training_only_threshold=0.699209, inner_splits=3
[OUTER 3] held-out AUC=0.7778; runtime=111 ms

[EXPERIMENT C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8881
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8951
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9161
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9091
[OUTER 4] selected_C=1, selected_inner_AUC=0.9161, best_candidate_inner_AUC=0.9161, training_only_threshold=0.821672, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=108 ms

[EXPERIMENT C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.7273
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.7133
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.7273
[INNER CV][C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.7343
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.7273, best_candidate_inner_AUC=0.7343, training_only_threshold=0.790724, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=108 ms
[EXPERIMENT] COMPLETED: C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA | AUC=0.8973 [0.7589, 0.9911] | AUPRC=0.9114 | Sensitivity=0.6429 | Specificity=0.9375 | F1=0.7500 | runtime=5.71 s

[SUITE] Launching experiment 39/47: C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_hard_monai_mask_only', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 532 ms.

====================================================================================================
[EXPERIMENT] START: C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA
[EXPERIMENT] Segmentation-derived control encoding only the standardized dilated hard MONAI mask for gate-valid slices; invalid slices are all zero. It removes MRI intensities while preserving mask morphology and position.
[EXPERIMENT] feature_mode=standardized_hard_monai_mask_only, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8750
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8542
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8542
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8542
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8750, best_candidate_inner_AUC=0.8750, training_only_threshold=0.609332, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=112 ms

[EXPERIMENT C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8671
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.8951
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9021
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9091
[OUTER 2] selected_C=1, selected_inner_AUC=0.9021, best_candidate_inner_AUC=0.9091, training_only_threshold=0.905688, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=107 ms

[EXPERIMENT C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.9301
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.9371
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.9510
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.9580
[OUTER 3] selected_C=1, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9580, training_only_threshold=0.836337, inner_splits=3
[OUTER 3] held-out AUC=0.5556; runtime=105 ms

[EXPERIMENT C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8182
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8531
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8462
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8462
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.8531, best_candidate_inner_AUC=0.8531, training_only_threshold=0.600259, inner_splits=3
[OUTER 4] held-out AUC=1.0000; runtime=105 ms

[EXPERIMENT C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.6503
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.6014
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.5944
[INNER CV][C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.6014
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.6503, best_candidate_inner_AUC=0.6503, training_only_threshold=0.413255, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=109 ms
[EXPERIMENT] COMPLETED: C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA | AUC=0.7857 [0.5893, 0.9554] | AUPRC=0.7772 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.66 s

[SUITE] Launching experiment 40/47: C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_monai_bbox_mask_only', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 534 ms.

====================================================================================================
[EXPERIMENT] START: C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA
[EXPERIMENT] Segmentation-derived control encoding only a binary rectangle around each valid standardized MONAI hard-mask bounding box; invalid slices are zero. It tests whether ROI location and extent alone encode the label.
[EXPERIMENT] feature_mode=standardized_monai_bbox_mask_only, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7083
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.7083
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.6875
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.6736
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7083, best_candidate_inner_AUC=0.7083, training_only_threshold=0.646772, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.6503
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.6783
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.7413
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.7552
[OUTER 2] selected_C=10, selected_inner_AUC=0.7552, best_candidate_inner_AUC=0.7552, training_only_threshold=0.644713, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=107 ms

[EXPERIMENT C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8322
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.7483
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.7622
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.7448
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.8322, best_candidate_inner_AUC=0.8322, training_only_threshold=0.565465, inner_splits=3
[OUTER 3] held-out AUC=0.7778; runtime=106 ms

[EXPERIMENT C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.7063
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.6783
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.6503
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.6783
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7063, best_candidate_inner_AUC=0.7063, training_only_threshold=0.511406, inner_splits=3
[OUTER 4] held-out AUC=0.5556; runtime=110 ms

[EXPERIMENT C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.7413
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.7972
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8112
[INNER CV][C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8112
[OUTER 5] selected_C=1, selected_inner_AUC=0.8112, best_candidate_inner_AUC=0.8112, training_only_threshold=0.054472, inner_splits=3
[OUTER 5] held-out AUC=0.5556; runtime=107 ms
[EXPERIMENT] COMPLETED: C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA | AUC=0.7098 [0.5000, 0.8884] | AUPRC=0.7239 | Sensitivity=0.6429 | Specificity=0.8125 | F1=0.6923 | runtime=5.72 s

[SUITE] Launching experiment 41/47: C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_soft_monai_histogram_only', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 531 ms.

====================================================================================================
[EXPERIMENT] START: C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA
[EXPERIMENT] Segmentation-derived distribution control encoding only a fixed-bin cumulative histogram of the standardized soft MONAI probability map. Original pixel position and mask morphology are absent.
[EXPERIMENT] feature_mode=standardized_soft_monai_histogram_only, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.8194
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8194
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8264
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8333
[OUTER 1] selected_C=1, selected_inner_AUC=0.8264, best_candidate_inner_AUC=0.8333, training_only_threshold=0.732385, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=110 ms

[EXPERIMENT C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8112
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.7972
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.7692
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.7622
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.8112, best_candidate_inner_AUC=0.8112, training_only_threshold=0.635219, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=109 ms

[EXPERIMENT C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.9371
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.9441
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.9441
[OUTER 3] selected_C=0.1, selected_inner_AUC=0.9510, best_candidate_inner_AUC=0.9510, training_only_threshold=0.528472, inner_splits=3
[OUTER 3] held-out AUC=0.5556; runtime=110 ms

[EXPERIMENT C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.7832
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.7762
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.7133
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.7133
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7832, best_candidate_inner_AUC=0.7832, training_only_threshold=0.508879, inner_splits=3
[OUTER 4] held-out AUC=0.6667; runtime=104 ms

[EXPERIMENT C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.7063
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.6993
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.6713
[INNER CV][C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.5804
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.7063, best_candidate_inner_AUC=0.7063, training_only_threshold=0.445127, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=108 ms
[EXPERIMENT] COMPLETED: C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA | AUC=0.7679 [0.5714, 0.9286] | AUPRC=0.7900 | Sensitivity=0.6429 | Specificity=0.8125 | F1=0.6923 | runtime=5.78 s

[SUITE] Launching experiment 42/47: C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_soft_monai_block_shuffled', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 515 ms.

====================================================================================================
[EXPERIMENT] START: C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA
[EXPERIMENT] Segmentation-derived control preserving each soft MONAI map's probability values and local within-block texture while applying a deterministic per-image global block permutation that destroys the original global shape and location.
[EXPERIMENT] feature_mode=standardized_soft_monai_block_shuffled, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9028
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.8958
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.8958
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.8958
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.9028, best_candidate_inner_AUC=0.9028, training_only_threshold=0.539025, inner_splits=3
[OUTER 1] held-out AUC=0.8750; runtime=111 ms

[EXPERIMENT C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.8392
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.8252
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.8322
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.8531, best_candidate_inner_AUC=0.8531, training_only_threshold=0.599161, inner_splits=3
[OUTER 2] held-out AUC=1.0000; runtime=106 ms

[EXPERIMENT C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.9371
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.9301
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.9231
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.9231
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.9371, best_candidate_inner_AUC=0.9371, training_only_threshold=0.547802, inner_splits=3
[OUTER 3] held-out AUC=0.7778; runtime=106 ms

[EXPERIMENT C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.9021
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.9091
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.9161
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.9091
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.9091, best_candidate_inner_AUC=0.9161, training_only_threshold=0.475297, inner_splits=3
[OUTER 4] held-out AUC=0.7778; runtime=108 ms

[EXPERIMENT C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.8811
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.8741
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.8741
[INNER CV][C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.8741
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.8811, best_candidate_inner_AUC=0.8811, training_only_threshold=0.612388, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=112 ms
[EXPERIMENT] COMPLETED: C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA | AUC=0.8795 [0.7232, 1.0000] | AUPRC=0.8855 | Sensitivity=0.8571 | Specificity=0.8750 | F1=0.8571 | runtime=5.74 s

[SUITE] Launching experiment 43/47: C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_canonical_hard_monai_mask_only', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 576 ms.

====================================================================================================
[EXPERIMENT] START: C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA
[EXPERIMENT] Segmentation-derived control that crops each valid hard MONAI mask to its own bounding box, preserves relative shape, then centers it at one fixed scale. Absolute mask position and size are removed.
[EXPERIMENT] feature_mode=standardized_canonical_hard_monai_mask_only, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.7292
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.6736
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.6389
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.6389
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.7292, best_candidate_inner_AUC=0.7292, training_only_threshold=0.448857, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=114 ms

[EXPERIMENT C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.6713
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.6643
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.6434
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.5874
[OUTER 2] selected_C=0.01, selected_inner_AUC=0.6713, best_candidate_inner_AUC=0.6713, training_only_threshold=0.746093, inner_splits=3
[OUTER 2] held-out AUC=0.7778; runtime=112 ms

[EXPERIMENT C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8811
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8601
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8462
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8392
[OUTER 3] selected_C=0.01, selected_inner_AUC=0.8811, best_candidate_inner_AUC=0.8811, training_only_threshold=0.650776, inner_splits=3
[OUTER 3] held-out AUC=0.6667; runtime=107 ms

[EXPERIMENT C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.7273
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.6923
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.6783
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.6573
[OUTER 4] selected_C=0.01, selected_inner_AUC=0.7273, best_candidate_inner_AUC=0.7273, training_only_threshold=0.613979, inner_splits=3
[OUTER 4] held-out AUC=0.6667; runtime=108 ms

[EXPERIMENT C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.5734
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.4895
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.4685
[INNER CV][C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.4755
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.5734, best_candidate_inner_AUC=0.5734, training_only_threshold=0.648146, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=106 ms
[EXPERIMENT] COMPLETED: C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA | AUC=0.7545 [0.5625, 0.9198] | AUPRC=0.7833 | Sensitivity=0.5714 | Specificity=0.6875 | F1=0.5926 | runtime=5.73 s

[SUITE] Launching experiment 44/47: C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA
[SUITE] Prepared data representation ('standardized_canonical_soft_monai_mask_only', 'patient_embedding', 'hierarchical', 'equal', 0.0, False) in 521 ms.

====================================================================================================
[EXPERIMENT] START: C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA
[EXPERIMENT] Segmentation-derived control that canonicalizes the valid soft MONAI probability map to a fixed centered scale using the hard-mask bounding box. Relative morphology and confidence remain, while absolute position and extent are removed.
[EXPERIMENT] feature_mode=standardized_canonical_soft_monai_mask_only, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=0%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.01, pooled_inner_AUC=0.9236
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=0.1, pooled_inner_AUC=0.9375
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=1, pooled_inner_AUC=0.9375
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=1] C=10, pooled_inner_AUC=0.9306
[OUTER 1] selected_C=0.1, selected_inner_AUC=0.9375, best_candidate_inner_AUC=0.9375, training_only_threshold=0.252153, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.01, pooled_inner_AUC=0.9021
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=0.1, pooled_inner_AUC=0.9231
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=1, pooled_inner_AUC=0.9441
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=2] C=10, pooled_inner_AUC=0.9441
[OUTER 2] selected_C=1, selected_inner_AUC=0.9441, best_candidate_inner_AUC=0.9441, training_only_threshold=0.150550, inner_splits=3
[OUTER 2] held-out AUC=1.0000; runtime=104 ms

[EXPERIMENT C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.01, pooled_inner_AUC=0.8601
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=0.1, pooled_inner_AUC=0.8881
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=1, pooled_inner_AUC=0.8951
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=3] C=10, pooled_inner_AUC=0.8951
[OUTER 3] selected_C=0.1, selected_inner_AUC=0.8881, best_candidate_inner_AUC=0.8951, training_only_threshold=0.782301, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=109 ms

[EXPERIMENT C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.01, pooled_inner_AUC=0.8462
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=0.1, pooled_inner_AUC=0.8671
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=1, pooled_inner_AUC=0.8601
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=4] C=10, pooled_inner_AUC=0.8601
[OUTER 4] selected_C=0.1, selected_inner_AUC=0.8671, best_candidate_inner_AUC=0.8671, training_only_threshold=0.874757, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=106 ms

[EXPERIMENT C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.01, pooled_inner_AUC=0.7902
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=0.1, pooled_inner_AUC=0.7552
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=1, pooled_inner_AUC=0.7413
[INNER CV][C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA][outer=5] C=10, pooled_inner_AUC=0.7483
[OUTER 5] selected_C=0.01, selected_inner_AUC=0.7902, best_candidate_inner_AUC=0.7902, training_only_threshold=0.545122, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=108 ms
[EXPERIMENT] COMPLETED: C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA | AUC=0.8929 [0.7633, 0.9821] | AUPRC=0.9028 | Sensitivity=0.7857 | Specificity=0.8125 | F1=0.7857 | runtime=5.66 s

[SUITE] Launching experiment 45/47: R1_ROI_HIER_LR_PCA_DROP10
[ROBUSTNESS] R1_ROI_HIER_LR_PCA_DROP10: retained 57675/63425 slices after 10% within-series dropout.
[SUITE] Prepared data representation ('monai_roi', 'patient_embedding', 'hierarchical', 'equal', 0.1, False) in 3.72 s.

====================================================================================================
[EXPERIMENT] START: R1_ROI_HIER_LR_PCA_DROP10
[EXPERIMENT] Exploratory robustness control after deterministic 10% slice dropout within each series proxy; at least one slice per proxy is retained.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=10%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT R1_ROI_HIER_LR_PCA_DROP10] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=1] C=0.01, pooled_inner_AUC=0.8958
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=1] C=0.1, pooled_inner_AUC=0.8958
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=1] C=1, pooled_inner_AUC=0.8958
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=1] C=10, pooled_inner_AUC=0.9028
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8958, best_candidate_inner_AUC=0.9028, training_only_threshold=0.632095, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=113 ms

[EXPERIMENT R1_ROI_HIER_LR_PCA_DROP10] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=2] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=2] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=2] C=1, pooled_inner_AUC=0.9580
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=2] C=10, pooled_inner_AUC=0.9790
[OUTER 2] selected_C=10, selected_inner_AUC=0.9790, best_candidate_inner_AUC=0.9790, training_only_threshold=0.078971, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=110 ms

[EXPERIMENT R1_ROI_HIER_LR_PCA_DROP10] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=3] C=0.01, pooled_inner_AUC=0.8042
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=3] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=3] C=1, pooled_inner_AUC=0.8112
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=3] C=10, pooled_inner_AUC=0.8182
[OUTER 3] selected_C=0.1, selected_inner_AUC=0.8182, best_candidate_inner_AUC=0.8182, training_only_threshold=0.595775, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=107 ms

[EXPERIMENT R1_ROI_HIER_LR_PCA_DROP10] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=4] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=4] C=1, pooled_inner_AUC=0.8601
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=4] C=10, pooled_inner_AUC=0.8671
[OUTER 4] selected_C=1, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8671, training_only_threshold=0.553626, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=108 ms

[EXPERIMENT R1_ROI_HIER_LR_PCA_DROP10] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=5] C=0.01, pooled_inner_AUC=0.8741
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=5] C=0.1, pooled_inner_AUC=0.8951
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=5] C=1, pooled_inner_AUC=0.9161
[INNER CV][R1_ROI_HIER_LR_PCA_DROP10][outer=5] C=10, pooled_inner_AUC=0.9161
[OUTER 5] selected_C=1, selected_inner_AUC=0.9161, best_candidate_inner_AUC=0.9161, training_only_threshold=0.632946, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=112 ms
[EXPERIMENT] COMPLETED: R1_ROI_HIER_LR_PCA_DROP10 | AUC=0.9196 [0.7946, 1.0000] | AUPRC=0.9324 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.66 s

[SUITE] Launching experiment 46/47: R2_ROI_HIER_LR_PCA_DROP25
[ROBUSTNESS] R2_ROI_HIER_LR_PCA_DROP25: retained 48283/63425 slices after 25% within-series dropout.
[SUITE] Prepared data representation ('monai_roi', 'patient_embedding', 'hierarchical', 'equal', 0.25, False) in 3.79 s.

====================================================================================================
[EXPERIMENT] START: R2_ROI_HIER_LR_PCA_DROP25
[EXPERIMENT] Exploratory robustness control after deterministic 25% slice dropout within each series proxy; at least one slice per proxy is retained.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=25%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT R2_ROI_HIER_LR_PCA_DROP25] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=1] C=0.01, pooled_inner_AUC=0.8958
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=1] C=0.1, pooled_inner_AUC=0.8958
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=1] C=1, pooled_inner_AUC=0.8958
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=1] C=10, pooled_inner_AUC=0.8958
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8958, best_candidate_inner_AUC=0.8958, training_only_threshold=0.628926, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT R2_ROI_HIER_LR_PCA_DROP25] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=2] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=2] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=2] C=1, pooled_inner_AUC=0.9580
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=2] C=10, pooled_inner_AUC=0.9860
[OUTER 2] selected_C=10, selected_inner_AUC=0.9860, best_candidate_inner_AUC=0.9860, training_only_threshold=0.089454, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=117 ms

[EXPERIMENT R2_ROI_HIER_LR_PCA_DROP25] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=3] C=0.01, pooled_inner_AUC=0.8112
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=3] C=0.1, pooled_inner_AUC=0.8182
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=3] C=1, pooled_inner_AUC=0.8252
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=3] C=10, pooled_inner_AUC=0.8322
[OUTER 3] selected_C=1, selected_inner_AUC=0.8252, best_candidate_inner_AUC=0.8322, training_only_threshold=0.660131, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=106 ms

[EXPERIMENT R2_ROI_HIER_LR_PCA_DROP25] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=4] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=4] C=0.1, pooled_inner_AUC=0.8462
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=4] C=1, pooled_inner_AUC=0.8601
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=4] C=10, pooled_inner_AUC=0.8671
[OUTER 4] selected_C=1, selected_inner_AUC=0.8601, best_candidate_inner_AUC=0.8671, training_only_threshold=0.527886, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=108 ms

[EXPERIMENT R2_ROI_HIER_LR_PCA_DROP25] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=5] C=0.01, pooled_inner_AUC=0.8601
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=5] C=0.1, pooled_inner_AUC=0.9021
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=5] C=1, pooled_inner_AUC=0.9161
[INNER CV][R2_ROI_HIER_LR_PCA_DROP25][outer=5] C=10, pooled_inner_AUC=0.9161
[OUTER 5] selected_C=1, selected_inner_AUC=0.9161, best_candidate_inner_AUC=0.9161, training_only_threshold=0.635838, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=105 ms
[EXPERIMENT] COMPLETED: R2_ROI_HIER_LR_PCA_DROP25 | AUC=0.9241 [0.8125, 1.0000] | AUPRC=0.9309 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.71 s

[SUITE] Launching experiment 47/47: R3_ROI_HIER_LR_PCA_DROP50
[ROBUSTNESS] R3_ROI_HIER_LR_PCA_DROP50: retained 32414/63425 slices after 50% within-series dropout.
[SUITE] Prepared data representation ('monai_roi', 'patient_embedding', 'hierarchical', 'equal', 0.5, False) in 3.64 s.

====================================================================================================
[EXPERIMENT] START: R3_ROI_HIER_LR_PCA_DROP50
[EXPERIMENT] Exploratory robustness control after deterministic 50% slice dropout within each series proxy; at least one slice per proxy is retained.
[EXPERIMENT] feature_mode=monai_roi, strategy=patient_embedding, pooling=hierarchical, weighting=equal, classifier=logistic_regression, PCA=True, fusion=None, slice_dropout=50%, exact_within_patient_dedup=False
====================================================================================================

[EXPERIMENT R3_ROI_HIER_LR_PCA_DROP50] OUTER FOLD 1/5
  train_patients=24 (Normal=12, Sick=12)
  valid_patients=6 (Normal=4, Sick=2)
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=1] C=0.01, pooled_inner_AUC=0.8958
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=1] C=0.1, pooled_inner_AUC=0.8819
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=1] C=1, pooled_inner_AUC=0.8819
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=1] C=10, pooled_inner_AUC=0.8681
[OUTER 1] selected_C=0.01, selected_inner_AUC=0.8958, best_candidate_inner_AUC=0.8958, training_only_threshold=0.453732, inner_splits=3
[OUTER 1] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT R3_ROI_HIER_LR_PCA_DROP50] OUTER FOLD 2/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=2] C=0.01, pooled_inner_AUC=0.9510
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=2] C=0.1, pooled_inner_AUC=0.9510
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=2] C=1, pooled_inner_AUC=0.9510
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=2] C=10, pooled_inner_AUC=0.9720
[OUTER 2] selected_C=10, selected_inner_AUC=0.9720, best_candidate_inner_AUC=0.9720, training_only_threshold=0.221234, inner_splits=3
[OUTER 2] held-out AUC=0.8889; runtime=111 ms

[EXPERIMENT R3_ROI_HIER_LR_PCA_DROP50] OUTER FOLD 3/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=3] C=0.01, pooled_inner_AUC=0.8182
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=3] C=0.1, pooled_inner_AUC=0.8322
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=3] C=1, pooled_inner_AUC=0.8322
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=3] C=10, pooled_inner_AUC=0.8392
[OUTER 3] selected_C=0.1, selected_inner_AUC=0.8322, best_candidate_inner_AUC=0.8392, training_only_threshold=0.647246, inner_splits=3
[OUTER 3] held-out AUC=1.0000; runtime=111 ms

[EXPERIMENT R3_ROI_HIER_LR_PCA_DROP50] OUTER FOLD 4/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=4] C=0.01, pooled_inner_AUC=0.8531
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=4] C=0.1, pooled_inner_AUC=0.8601
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=4] C=1, pooled_inner_AUC=0.8671
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=4] C=10, pooled_inner_AUC=0.8741
[OUTER 4] selected_C=1, selected_inner_AUC=0.8671, best_candidate_inner_AUC=0.8741, training_only_threshold=0.446549, inner_splits=3
[OUTER 4] held-out AUC=0.8889; runtime=115 ms

[EXPERIMENT R3_ROI_HIER_LR_PCA_DROP50] OUTER FOLD 5/5
  train_patients=24 (Normal=13, Sick=11)
  valid_patients=6 (Normal=3, Sick=3)
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=5] C=0.01, pooled_inner_AUC=0.8811
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=5] C=0.1, pooled_inner_AUC=0.8951
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=5] C=1, pooled_inner_AUC=0.9021
[INNER CV][R3_ROI_HIER_LR_PCA_DROP50][outer=5] C=10, pooled_inner_AUC=0.9091
[OUTER 5] selected_C=1, selected_inner_AUC=0.9021, best_candidate_inner_AUC=0.9091, training_only_threshold=0.570100, inner_splits=3
[OUTER 5] held-out AUC=1.0000; runtime=117 ms
[EXPERIMENT] COMPLETED: R3_ROI_HIER_LR_PCA_DROP50 | AUC=0.9152 [0.7812, 1.0000] | AUPRC=0.9277 | Sensitivity=0.7857 | Specificity=0.9375 | F1=0.8462 | runtime=5.75 s
[PIPELINE 08/14] COMPLETED: Run every enabled experiment on the shared folds in 10 min 25.7 s
[PIPELINE 08/14] Successful=47, failed=0.

==============================================================================
[PIPELINE 09/14] START: Calculate paired comparisons and write master result files
[PIPELINE 09/14] Expected workload: Patient-level bootstrap comparisons and CSV/JSON consolidation.
==============================================================================
[PIPELINE 09/14] COMPLETED: Calculate paired comparisons and write master result files in 6 min 24.3 s
[PIPELINE 09/14] Summary rows=47; baseline-paired rows=47; matched-ablation rows=43.

==============================================================================
[PIPELINE 10/14] START: Run repeated nested-CV split-stability analyses
[PIPELINE 10/14] Expected workload: Several patient-level reruns for selected models and controls; frozen features are reused.
==============================================================================
[STABILITY] Running 50 repeated nested patient-level splits for B1_STANDARDIZED_ROI_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.8616, AUPRC=0.8601
[STABILITY] repeat=2/50, seed=20043, AUC=0.8795, AUPRC=0.8734
[STABILITY] repeat=3/50, seed=20044, AUC=0.8750, AUPRC=0.8816
[STABILITY] repeat=4/50, seed=20045, AUC=0.9554, AUPRC=0.9564
[STABILITY] repeat=5/50, seed=20046, AUC=0.8527, AUPRC=0.8440
[STABILITY] repeat=6/50, seed=20047, AUC=0.8438, AUPRC=0.8791
[STABILITY] repeat=7/50, seed=20048, AUC=0.8884, AUPRC=0.8827
[STABILITY] repeat=8/50, seed=20049, AUC=0.8616, AUPRC=0.8670
[STABILITY] repeat=9/50, seed=20050, AUC=0.9286, AUPRC=0.9314
[STABILITY] repeat=10/50, seed=20051, AUC=0.8438, AUPRC=0.8577
[STABILITY] repeat=11/50, seed=20052, AUC=0.8393, AUPRC=0.8241
[STABILITY] repeat=12/50, seed=20053, AUC=0.8438, AUPRC=0.8550
[STABILITY] repeat=13/50, seed=20054, AUC=0.8839, AUPRC=0.9061
[STABILITY] repeat=14/50, seed=20055, AUC=0.8571, AUPRC=0.8856
[STABILITY] repeat=15/50, seed=20056, AUC=0.8616, AUPRC=0.8655
[STABILITY] repeat=16/50, seed=20057, AUC=0.8839, AUPRC=0.9049
[STABILITY] repeat=17/50, seed=20058, AUC=0.8839, AUPRC=0.9021
[STABILITY] repeat=18/50, seed=20059, AUC=0.8348, AUPRC=0.8485
[STABILITY] repeat=19/50, seed=20060, AUC=0.7857, AUPRC=0.8261
[STABILITY] repeat=20/50, seed=20061, AUC=0.9018, AUPRC=0.9009
[STABILITY] repeat=21/50, seed=20062, AUC=0.9062, AUPRC=0.9055
[STABILITY] repeat=22/50, seed=20063, AUC=0.8259, AUPRC=0.8715
[STABILITY] repeat=23/50, seed=20064, AUC=0.8705, AUPRC=0.8412
[STABILITY] repeat=24/50, seed=20065, AUC=0.8348, AUPRC=0.8462
[STABILITY] repeat=25/50, seed=20066, AUC=0.9062, AUPRC=0.9282
[STABILITY] repeat=26/50, seed=20067, AUC=0.8393, AUPRC=0.8304
[STABILITY] repeat=27/50, seed=20068, AUC=0.8661, AUPRC=0.8593
[STABILITY] repeat=28/50, seed=20069, AUC=0.8705, AUPRC=0.8461
[STABILITY] repeat=29/50, seed=20070, AUC=0.8125, AUPRC=0.8153
[STABILITY] repeat=30/50, seed=20071, AUC=0.9018, AUPRC=0.9094
[STABILITY] repeat=31/50, seed=20072, AUC=0.8080, AUPRC=0.8390
[STABILITY] repeat=32/50, seed=20073, AUC=0.7812, AUPRC=0.7882
[STABILITY] repeat=33/50, seed=20074, AUC=0.8616, AUPRC=0.8593
[STABILITY] repeat=34/50, seed=20075, AUC=0.8750, AUPRC=0.8485
[STABILITY] repeat=35/50, seed=20076, AUC=0.7991, AUPRC=0.8526
[STABILITY] repeat=36/50, seed=20077, AUC=0.8661, AUPRC=0.8916
[STABILITY] repeat=37/50, seed=20078, AUC=0.8661, AUPRC=0.8744
[STABILITY] repeat=38/50, seed=20079, AUC=0.9330, AUPRC=0.9368
[STABILITY] repeat=39/50, seed=20080, AUC=0.8750, AUPRC=0.8669
[STABILITY] repeat=40/50, seed=20081, AUC=0.8750, AUPRC=0.8743
[STABILITY] repeat=41/50, seed=20082, AUC=0.8839, AUPRC=0.8951
[STABILITY] repeat=42/50, seed=20083, AUC=0.9330, AUPRC=0.9341
[STABILITY] repeat=43/50, seed=20084, AUC=0.8929, AUPRC=0.9031
[STABILITY] repeat=44/50, seed=20085, AUC=0.8616, AUPRC=0.8616
[STABILITY] repeat=45/50, seed=20086, AUC=0.8571, AUPRC=0.8480
[STABILITY] repeat=46/50, seed=20087, AUC=0.8750, AUPRC=0.8683
[STABILITY] repeat=47/50, seed=20088, AUC=0.9152, AUPRC=0.9138
[STABILITY] repeat=48/50, seed=20089, AUC=0.7857, AUPRC=0.8279
[STABILITY] repeat=49/50, seed=20090, AUC=0.8973, AUPRC=0.9013
[STABILITY] repeat=50/50, seed=20091, AUC=0.8571, AUPRC=0.8810
[STABILITY] AUC median=0.8661, IQR=[0.8438, 0.8839], range=[0.7812, 0.9554]
[STABILITY] Running 50 repeated nested patient-level splits for A9_STANDARDIZED_FULL_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.8795, AUPRC=0.8557
[STABILITY] repeat=2/50, seed=20043, AUC=0.7991, AUPRC=0.7852
[STABILITY] repeat=3/50, seed=20044, AUC=0.8304, AUPRC=0.8677
[STABILITY] repeat=4/50, seed=20045, AUC=0.8527, AUPRC=0.8849
[STABILITY] repeat=5/50, seed=20046, AUC=0.7991, AUPRC=0.7787
[STABILITY] repeat=6/50, seed=20047, AUC=0.8438, AUPRC=0.8724
[STABILITY] repeat=7/50, seed=20048, AUC=0.8839, AUPRC=0.8884
[STABILITY] repeat=8/50, seed=20049, AUC=0.8080, AUPRC=0.7765
[STABILITY] repeat=9/50, seed=20050, AUC=0.8929, AUPRC=0.8822
[STABILITY] repeat=10/50, seed=20051, AUC=0.8304, AUPRC=0.8652
[STABILITY] repeat=11/50, seed=20052, AUC=0.8080, AUPRC=0.8039
[STABILITY] repeat=12/50, seed=20053, AUC=0.8080, AUPRC=0.8037
[STABILITY] repeat=13/50, seed=20054, AUC=0.8348, AUPRC=0.8672
[STABILITY] repeat=14/50, seed=20055, AUC=0.9018, AUPRC=0.9108
[STABILITY] repeat=15/50, seed=20056, AUC=0.8973, AUPRC=0.8891
[STABILITY] repeat=16/50, seed=20057, AUC=0.8125, AUPRC=0.8302
[STABILITY] repeat=17/50, seed=20058, AUC=0.7991, AUPRC=0.8642
[STABILITY] repeat=18/50, seed=20059, AUC=0.8571, AUPRC=0.8541
[STABILITY] repeat=19/50, seed=20060, AUC=0.7902, AUPRC=0.7916
[STABILITY] repeat=20/50, seed=20061, AUC=0.8482, AUPRC=0.8850
[STABILITY] repeat=21/50, seed=20062, AUC=0.8170, AUPRC=0.8652
[STABILITY] repeat=22/50, seed=20063, AUC=0.7857, AUPRC=0.8506
[STABILITY] repeat=23/50, seed=20064, AUC=0.7991, AUPRC=0.8230
[STABILITY] repeat=24/50, seed=20065, AUC=0.8393, AUPRC=0.8613
[STABILITY] repeat=25/50, seed=20066, AUC=0.8527, AUPRC=0.8396
[STABILITY] repeat=26/50, seed=20067, AUC=0.8348, AUPRC=0.8300
[STABILITY] repeat=27/50, seed=20068, AUC=0.7902, AUPRC=0.8310
[STABILITY] repeat=28/50, seed=20069, AUC=0.8661, AUPRC=0.8319
[STABILITY] repeat=29/50, seed=20070, AUC=0.7366, AUPRC=0.8009
[STABILITY] repeat=30/50, seed=20071, AUC=0.8795, AUPRC=0.8396
[STABILITY] repeat=31/50, seed=20072, AUC=0.8036, AUPRC=0.8510
[STABILITY] repeat=32/50, seed=20073, AUC=0.7277, AUPRC=0.7119
[STABILITY] repeat=33/50, seed=20074, AUC=0.7500, AUPRC=0.7992
[STABILITY] repeat=34/50, seed=20075, AUC=0.8571, AUPRC=0.8147
[STABILITY] repeat=35/50, seed=20076, AUC=0.8036, AUPRC=0.8157
[STABILITY] repeat=36/50, seed=20077, AUC=0.8527, AUPRC=0.8697
[STABILITY] repeat=37/50, seed=20078, AUC=0.8304, AUPRC=0.8502
[STABILITY] repeat=38/50, seed=20079, AUC=0.8527, AUPRC=0.8913
[STABILITY] repeat=39/50, seed=20080, AUC=0.7812, AUPRC=0.7920
[STABILITY] repeat=40/50, seed=20081, AUC=0.8839, AUPRC=0.8944
[STABILITY] repeat=41/50, seed=20082, AUC=0.8214, AUPRC=0.8240
[STABILITY] repeat=42/50, seed=20083, AUC=0.8080, AUPRC=0.8165
[STABILITY] repeat=43/50, seed=20084, AUC=0.7812, AUPRC=0.8419
[STABILITY] repeat=44/50, seed=20085, AUC=0.8616, AUPRC=0.8482
[STABILITY] repeat=45/50, seed=20086, AUC=0.8304, AUPRC=0.8200
[STABILITY] repeat=46/50, seed=20087, AUC=0.8125, AUPRC=0.8192
[STABILITY] repeat=47/50, seed=20088, AUC=0.8482, AUPRC=0.8750
[STABILITY] repeat=48/50, seed=20089, AUC=0.7812, AUPRC=0.7624
[STABILITY] repeat=49/50, seed=20090, AUC=0.8571, AUPRC=0.8766
[STABILITY] repeat=50/50, seed=20091, AUC=0.8482, AUPRC=0.8654
[STABILITY] AUC median=0.8304, IQR=[0.8002, 0.8527], range=[0.7277, 0.9018]
[STABILITY] Running 50 repeated nested patient-level splits for A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9509, AUPRC=0.9587
[STABILITY] repeat=2/50, seed=20043, AUC=0.9643, AUPRC=0.9662
[STABILITY] repeat=3/50, seed=20044, AUC=0.9509, AUPRC=0.9559
[STABILITY] repeat=4/50, seed=20045, AUC=0.9554, AUPRC=0.9618
[STABILITY] repeat=5/50, seed=20046, AUC=0.9554, AUPRC=0.9614
[STABILITY] repeat=6/50, seed=20047, AUC=0.9554, AUPRC=0.9589
[STABILITY] repeat=7/50, seed=20048, AUC=0.9643, AUPRC=0.9599
[STABILITY] repeat=8/50, seed=20049, AUC=0.9375, AUPRC=0.9367
[STABILITY] repeat=9/50, seed=20050, AUC=0.9598, AUPRC=0.9615
[STABILITY] repeat=10/50, seed=20051, AUC=0.9062, AUPRC=0.9354
[STABILITY] repeat=11/50, seed=20052, AUC=0.9375, AUPRC=0.9399
[STABILITY] repeat=12/50, seed=20053, AUC=0.9375, AUPRC=0.9411
[STABILITY] repeat=13/50, seed=20054, AUC=0.9688, AUPRC=0.9706
[STABILITY] repeat=14/50, seed=20055, AUC=0.9420, AUPRC=0.9432
[STABILITY] repeat=15/50, seed=20056, AUC=0.9643, AUPRC=0.9638
[STABILITY] repeat=16/50, seed=20057, AUC=0.9420, AUPRC=0.9485
[STABILITY] repeat=17/50, seed=20058, AUC=0.9598, AUPRC=0.9652
[STABILITY] repeat=18/50, seed=20059, AUC=0.9732, AUPRC=0.9746
[STABILITY] repeat=19/50, seed=20060, AUC=0.9464, AUPRC=0.9560
[STABILITY] repeat=20/50, seed=20061, AUC=0.9643, AUPRC=0.9690
[STABILITY] repeat=21/50, seed=20062, AUC=0.9375, AUPRC=0.9458
[STABILITY] repeat=22/50, seed=20063, AUC=0.9375, AUPRC=0.9552
[STABILITY] repeat=23/50, seed=20064, AUC=0.9196, AUPRC=0.9445
[STABILITY] repeat=24/50, seed=20065, AUC=0.9420, AUPRC=0.9505
[STABILITY] repeat=25/50, seed=20066, AUC=0.9598, AUPRC=0.9644
[STABILITY] repeat=26/50, seed=20067, AUC=0.9152, AUPRC=0.9071
[STABILITY] repeat=27/50, seed=20068, AUC=0.9420, AUPRC=0.9475
[STABILITY] repeat=28/50, seed=20069, AUC=0.9554, AUPRC=0.9571
[STABILITY] repeat=29/50, seed=20070, AUC=0.8750, AUPRC=0.8896
[STABILITY] repeat=30/50, seed=20071, AUC=0.9241, AUPRC=0.9047
[STABILITY] repeat=31/50, seed=20072, AUC=0.9375, AUPRC=0.9385
[STABILITY] repeat=32/50, seed=20073, AUC=0.8839, AUPRC=0.8956
[STABILITY] repeat=33/50, seed=20074, AUC=0.9821, AUPRC=0.9823
[STABILITY] repeat=34/50, seed=20075, AUC=0.9732, AUPRC=0.9761
[STABILITY] repeat=35/50, seed=20076, AUC=0.9375, AUPRC=0.9444
[STABILITY] repeat=36/50, seed=20077, AUC=0.9688, AUPRC=0.9717
[STABILITY] repeat=37/50, seed=20078, AUC=0.9330, AUPRC=0.9402
[STABILITY] repeat=38/50, seed=20079, AUC=0.9643, AUPRC=0.9690
[STABILITY] repeat=39/50, seed=20080, AUC=0.9554, AUPRC=0.9576
[STABILITY] repeat=40/50, seed=20081, AUC=0.9375, AUPRC=0.9435
[STABILITY] repeat=41/50, seed=20082, AUC=0.9732, AUPRC=0.9740
[STABILITY] repeat=42/50, seed=20083, AUC=0.9688, AUPRC=0.9707
[STABILITY] repeat=43/50, seed=20084, AUC=0.9509, AUPRC=0.9575
[STABILITY] repeat=44/50, seed=20085, AUC=0.9375, AUPRC=0.9497
[STABILITY] repeat=45/50, seed=20086, AUC=0.9241, AUPRC=0.9310
[STABILITY] repeat=46/50, seed=20087, AUC=0.9375, AUPRC=0.9432
[STABILITY] repeat=47/50, seed=20088, AUC=0.9420, AUPRC=0.9440
[STABILITY] repeat=48/50, seed=20089, AUC=0.9286, AUPRC=0.9313
[STABILITY] repeat=49/50, seed=20090, AUC=0.9777, AUPRC=0.9779
[STABILITY] repeat=50/50, seed=20091, AUC=0.9643, AUPRC=0.9673
[STABILITY] AUC median=0.9487, IQR=[0.9375, 0.9643], range=[0.8750, 0.9821]
[STABILITY] Running 50 repeated nested patient-level splits for A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9241, AUPRC=0.9206
[STABILITY] repeat=2/50, seed=20043, AUC=0.8884, AUPRC=0.8741
[STABILITY] repeat=3/50, seed=20044, AUC=0.8884, AUPRC=0.8719
[STABILITY] repeat=4/50, seed=20045, AUC=0.9509, AUPRC=0.9606
[STABILITY] repeat=5/50, seed=20046, AUC=0.9062, AUPRC=0.9030
[STABILITY] repeat=6/50, seed=20047, AUC=0.9107, AUPRC=0.9392
[STABILITY] repeat=7/50, seed=20048, AUC=0.8929, AUPRC=0.8753
[STABILITY] repeat=8/50, seed=20049, AUC=0.9464, AUPRC=0.9570
[STABILITY] repeat=9/50, seed=20050, AUC=0.9286, AUPRC=0.9334
[STABILITY] repeat=10/50, seed=20051, AUC=0.8527, AUPRC=0.8447
[STABILITY] repeat=11/50, seed=20052, AUC=0.8973, AUPRC=0.9004
[STABILITY] repeat=12/50, seed=20053, AUC=0.8795, AUPRC=0.9035
[STABILITY] repeat=13/50, seed=20054, AUC=0.9375, AUPRC=0.9534
[STABILITY] repeat=14/50, seed=20055, AUC=0.9286, AUPRC=0.9468
[STABILITY] repeat=15/50, seed=20056, AUC=0.9464, AUPRC=0.9596
[STABILITY] repeat=16/50, seed=20057, AUC=0.9107, AUPRC=0.9321
[STABILITY] repeat=17/50, seed=20058, AUC=0.9286, AUPRC=0.9440
[STABILITY] repeat=18/50, seed=20059, AUC=0.9152, AUPRC=0.9065
[STABILITY] repeat=19/50, seed=20060, AUC=0.8839, AUPRC=0.8862
[STABILITY] repeat=20/50, seed=20061, AUC=0.9420, AUPRC=0.9532
[STABILITY] repeat=21/50, seed=20062, AUC=0.9241, AUPRC=0.9369
[STABILITY] repeat=22/50, seed=20063, AUC=0.9152, AUPRC=0.9334
[STABILITY] repeat=23/50, seed=20064, AUC=0.9062, AUPRC=0.9045
[STABILITY] repeat=24/50, seed=20065, AUC=0.8929, AUPRC=0.9100
[STABILITY] repeat=25/50, seed=20066, AUC=0.9330, AUPRC=0.9455
[STABILITY] repeat=26/50, seed=20067, AUC=0.8839, AUPRC=0.9067
[STABILITY] repeat=27/50, seed=20068, AUC=0.8973, AUPRC=0.9088
[STABILITY] repeat=28/50, seed=20069, AUC=0.9018, AUPRC=0.8722
[STABILITY] repeat=29/50, seed=20070, AUC=0.8304, AUPRC=0.8499
[STABILITY] repeat=30/50, seed=20071, AUC=0.9018, AUPRC=0.8625
[STABILITY] repeat=31/50, seed=20072, AUC=0.8527, AUPRC=0.8682
[STABILITY] repeat=32/50, seed=20073, AUC=0.8482, AUPRC=0.8831
[STABILITY] repeat=33/50, seed=20074, AUC=0.9554, AUPRC=0.9612
[STABILITY] repeat=34/50, seed=20075, AUC=0.9107, AUPRC=0.8994
[STABILITY] repeat=35/50, seed=20076, AUC=0.8661, AUPRC=0.8405
[STABILITY] repeat=36/50, seed=20077, AUC=0.9062, AUPRC=0.9282
[STABILITY] repeat=37/50, seed=20078, AUC=0.8839, AUPRC=0.9088
[STABILITY] repeat=38/50, seed=20079, AUC=0.9464, AUPRC=0.9327
[STABILITY] repeat=39/50, seed=20080, AUC=0.8795, AUPRC=0.8630
[STABILITY] repeat=40/50, seed=20081, AUC=0.8750, AUPRC=0.8945
[STABILITY] repeat=41/50, seed=20082, AUC=0.9464, AUPRC=0.9555
[STABILITY] repeat=42/50, seed=20083, AUC=0.9554, AUPRC=0.9669
[STABILITY] repeat=43/50, seed=20084, AUC=0.8929, AUPRC=0.9054
[STABILITY] repeat=44/50, seed=20085, AUC=0.8750, AUPRC=0.8483
[STABILITY] repeat=45/50, seed=20086, AUC=0.8884, AUPRC=0.8868
[STABILITY] repeat=46/50, seed=20087, AUC=0.8929, AUPRC=0.8939
[STABILITY] repeat=47/50, seed=20088, AUC=0.9196, AUPRC=0.9223
[STABILITY] repeat=48/50, seed=20089, AUC=0.9018, AUPRC=0.9143
[STABILITY] repeat=49/50, seed=20090, AUC=0.9554, AUPRC=0.9634
[STABILITY] repeat=50/50, seed=20091, AUC=0.8929, AUPRC=0.9114
[STABILITY] AUC median=0.9040, IQR=[0.8884, 0.9286], range=[0.8304, 0.9554]
[STABILITY] Running 50 repeated nested patient-level splits for C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.8616, AUPRC=0.8359
[STABILITY] repeat=2/50, seed=20043, AUC=0.8527, AUPRC=0.8564
[STABILITY] repeat=3/50, seed=20044, AUC=0.8259, AUPRC=0.8649
[STABILITY] repeat=4/50, seed=20045, AUC=0.9509, AUPRC=0.9448
[STABILITY] repeat=5/50, seed=20046, AUC=0.8304, AUPRC=0.8298
[STABILITY] repeat=6/50, seed=20047, AUC=0.8884, AUPRC=0.9099
[STABILITY] repeat=7/50, seed=20048, AUC=0.9018, AUPRC=0.9025
[STABILITY] repeat=8/50, seed=20049, AUC=0.7991, AUPRC=0.8174
[STABILITY] repeat=9/50, seed=20050, AUC=0.9107, AUPRC=0.8999
[STABILITY] repeat=10/50, seed=20051, AUC=0.8393, AUPRC=0.8551
[STABILITY] repeat=11/50, seed=20052, AUC=0.8750, AUPRC=0.8916
[STABILITY] repeat=12/50, seed=20053, AUC=0.7946, AUPRC=0.7955
[STABILITY] repeat=13/50, seed=20054, AUC=0.8795, AUPRC=0.8790
[STABILITY] repeat=14/50, seed=20055, AUC=0.8482, AUPRC=0.8575
[STABILITY] repeat=15/50, seed=20056, AUC=0.9152, AUPRC=0.8953
[STABILITY] repeat=16/50, seed=20057, AUC=0.8214, AUPRC=0.7921
[STABILITY] repeat=17/50, seed=20058, AUC=0.8438, AUPRC=0.8455
[STABILITY] repeat=18/50, seed=20059, AUC=0.7857, AUPRC=0.7937
[STABILITY] repeat=19/50, seed=20060, AUC=0.7812, AUPRC=0.8085
[STABILITY] repeat=20/50, seed=20061, AUC=0.8571, AUPRC=0.8550
[STABILITY] repeat=21/50, seed=20062, AUC=0.8482, AUPRC=0.8675
[STABILITY] repeat=22/50, seed=20063, AUC=0.8393, AUPRC=0.8764
[STABILITY] repeat=23/50, seed=20064, AUC=0.8795, AUPRC=0.8502
[STABILITY] repeat=24/50, seed=20065, AUC=0.8304, AUPRC=0.8096
[STABILITY] repeat=25/50, seed=20066, AUC=0.9062, AUPRC=0.9056
[STABILITY] repeat=26/50, seed=20067, AUC=0.8036, AUPRC=0.7598
[STABILITY] repeat=27/50, seed=20068, AUC=0.8438, AUPRC=0.8476
[STABILITY] repeat=28/50, seed=20069, AUC=0.8661, AUPRC=0.8479
[STABILITY] repeat=29/50, seed=20070, AUC=0.7768, AUPRC=0.8001
[STABILITY] repeat=30/50, seed=20071, AUC=0.8304, AUPRC=0.8142
[STABILITY] repeat=31/50, seed=20072, AUC=0.8438, AUPRC=0.8717
[STABILITY] repeat=32/50, seed=20073, AUC=0.7812, AUPRC=0.7535
[STABILITY] repeat=33/50, seed=20074, AUC=0.8795, AUPRC=0.8377
[STABILITY] repeat=34/50, seed=20075, AUC=0.8884, AUPRC=0.8809
[STABILITY] repeat=35/50, seed=20076, AUC=0.8705, AUPRC=0.8700
[STABILITY] repeat=36/50, seed=20077, AUC=0.8616, AUPRC=0.8633
[STABILITY] repeat=37/50, seed=20078, AUC=0.8170, AUPRC=0.8302
[STABILITY] repeat=38/50, seed=20079, AUC=0.8616, AUPRC=0.8977
[STABILITY] repeat=39/50, seed=20080, AUC=0.8705, AUPRC=0.8622
[STABILITY] repeat=40/50, seed=20081, AUC=0.8393, AUPRC=0.8416
[STABILITY] repeat=41/50, seed=20082, AUC=0.8393, AUPRC=0.8367
[STABILITY] repeat=42/50, seed=20083, AUC=0.8839, AUPRC=0.8804
[STABILITY] repeat=43/50, seed=20084, AUC=0.9018, AUPRC=0.8834
[STABILITY] repeat=44/50, seed=20085, AUC=0.8527, AUPRC=0.8214
[STABILITY] repeat=45/50, seed=20086, AUC=0.8929, AUPRC=0.8774
[STABILITY] repeat=46/50, seed=20087, AUC=0.8438, AUPRC=0.8677
[STABILITY] repeat=47/50, seed=20088, AUC=0.8438, AUPRC=0.8375
[STABILITY] repeat=48/50, seed=20089, AUC=0.7902, AUPRC=0.8189
[STABILITY] repeat=49/50, seed=20090, AUC=0.8661, AUPRC=0.8666
[STABILITY] repeat=50/50, seed=20091, AUC=0.8750, AUPRC=0.8964
[STABILITY] AUC median=0.8504, IQR=[0.8304, 0.8783], range=[0.7768, 0.9509]
[STABILITY] Running 50 repeated nested patient-level splits for A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9643, AUPRC=0.9673
[STABILITY] repeat=2/50, seed=20043, AUC=0.9464, AUPRC=0.9503
[STABILITY] repeat=3/50, seed=20044, AUC=0.9509, AUPRC=0.9587
[STABILITY] repeat=4/50, seed=20045, AUC=0.9643, AUPRC=0.9673
[STABILITY] repeat=5/50, seed=20046, AUC=0.9777, AUPRC=0.9790
[STABILITY] repeat=6/50, seed=20047, AUC=0.9509, AUPRC=0.9550
[STABILITY] repeat=7/50, seed=20048, AUC=0.9598, AUPRC=0.9584
[STABILITY] repeat=8/50, seed=20049, AUC=0.9554, AUPRC=0.9546
[STABILITY] repeat=9/50, seed=20050, AUC=0.9598, AUPRC=0.9615
[STABILITY] repeat=10/50, seed=20051, AUC=0.9152, AUPRC=0.9346
[STABILITY] repeat=11/50, seed=20052, AUC=0.9330, AUPRC=0.9375
[STABILITY] repeat=12/50, seed=20053, AUC=0.9420, AUPRC=0.9471
[STABILITY] repeat=13/50, seed=20054, AUC=0.9777, AUPRC=0.9790
[STABILITY] repeat=14/50, seed=20055, AUC=0.9464, AUPRC=0.9490
[STABILITY] repeat=15/50, seed=20056, AUC=0.9643, AUPRC=0.9638
[STABILITY] repeat=16/50, seed=20057, AUC=0.9420, AUPRC=0.9485
[STABILITY] repeat=17/50, seed=20058, AUC=0.9554, AUPRC=0.9618
[STABILITY] repeat=18/50, seed=20059, AUC=0.9732, AUPRC=0.9761
[STABILITY] repeat=19/50, seed=20060, AUC=0.9643, AUPRC=0.9678
[STABILITY] repeat=20/50, seed=20061, AUC=0.9643, AUPRC=0.9690
[STABILITY] repeat=21/50, seed=20062, AUC=0.9464, AUPRC=0.9532
[STABILITY] repeat=22/50, seed=20063, AUC=0.9286, AUPRC=0.9487
[STABILITY] repeat=23/50, seed=20064, AUC=0.9196, AUPRC=0.9458
[STABILITY] repeat=24/50, seed=20065, AUC=0.9554, AUPRC=0.9597
[STABILITY] repeat=25/50, seed=20066, AUC=0.9732, AUPRC=0.9746
[STABILITY] repeat=26/50, seed=20067, AUC=0.9152, AUPRC=0.9091
[STABILITY] repeat=27/50, seed=20068, AUC=0.9554, AUPRC=0.9576
[STABILITY] repeat=28/50, seed=20069, AUC=0.9554, AUPRC=0.9571
[STABILITY] repeat=29/50, seed=20070, AUC=0.9018, AUPRC=0.9043
[STABILITY] repeat=30/50, seed=20071, AUC=0.9375, AUPRC=0.9258
[STABILITY] repeat=31/50, seed=20072, AUC=0.9286, AUPRC=0.9349
[STABILITY] repeat=32/50, seed=20073, AUC=0.8973, AUPRC=0.9076
[STABILITY] repeat=33/50, seed=20074, AUC=0.9821, AUPRC=0.9823
[STABILITY] repeat=34/50, seed=20075, AUC=0.9732, AUPRC=0.9761
[STABILITY] repeat=35/50, seed=20076, AUC=0.9464, AUPRC=0.9516
[STABILITY] repeat=36/50, seed=20077, AUC=0.9866, AUPRC=0.9860
[STABILITY] repeat=37/50, seed=20078, AUC=0.9152, AUPRC=0.9302
[STABILITY] repeat=38/50, seed=20079, AUC=0.9643, AUPRC=0.9711
[STABILITY] repeat=39/50, seed=20080, AUC=0.9509, AUPRC=0.9535
[STABILITY] repeat=40/50, seed=20081, AUC=0.9420, AUPRC=0.9465
[STABILITY] repeat=41/50, seed=20082, AUC=0.9777, AUPRC=0.9779
[STABILITY] repeat=42/50, seed=20083, AUC=0.9777, AUPRC=0.9790
[STABILITY] repeat=43/50, seed=20084, AUC=0.9509, AUPRC=0.9575
[STABILITY] repeat=44/50, seed=20085, AUC=0.9464, AUPRC=0.9560
[STABILITY] repeat=45/50, seed=20086, AUC=0.9420, AUPRC=0.9440
[STABILITY] repeat=46/50, seed=20087, AUC=0.9330, AUPRC=0.9429
[STABILITY] repeat=47/50, seed=20088, AUC=0.9420, AUPRC=0.9440
[STABILITY] repeat=48/50, seed=20089, AUC=0.9509, AUPRC=0.9606
[STABILITY] repeat=49/50, seed=20090, AUC=0.9732, AUPRC=0.9740
[STABILITY] repeat=50/50, seed=20091, AUC=0.9509, AUPRC=0.9594
[STABILITY] AUC median=0.9509, IQR=[0.9420, 0.9643], range=[0.8973, 0.9866]
[STABILITY] Running 50 repeated nested patient-level splits for A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9464, AUPRC=0.9471
[STABILITY] repeat=2/50, seed=20043, AUC=0.9286, AUPRC=0.9339
[STABILITY] repeat=3/50, seed=20044, AUC=0.9241, AUPRC=0.9282
[STABILITY] repeat=4/50, seed=20045, AUC=0.9420, AUPRC=0.9532
[STABILITY] repeat=5/50, seed=20046, AUC=0.9330, AUPRC=0.9347
[STABILITY] repeat=6/50, seed=20047, AUC=0.9509, AUPRC=0.9505
[STABILITY] repeat=7/50, seed=20048, AUC=0.9330, AUPRC=0.9411
[STABILITY] repeat=8/50, seed=20049, AUC=0.9688, AUPRC=0.9676
[STABILITY] repeat=9/50, seed=20050, AUC=0.9062, AUPRC=0.9151
[STABILITY] repeat=10/50, seed=20051, AUC=0.9062, AUPRC=0.9260
[STABILITY] repeat=11/50, seed=20052, AUC=0.8661, AUPRC=0.9008
[STABILITY] repeat=12/50, seed=20053, AUC=0.9241, AUPRC=0.9305
[STABILITY] repeat=13/50, seed=20054, AUC=0.8705, AUPRC=0.9218
[STABILITY] repeat=14/50, seed=20055, AUC=0.9598, AUPRC=0.9620
[STABILITY] repeat=15/50, seed=20056, AUC=0.9330, AUPRC=0.9505
[STABILITY] repeat=16/50, seed=20057, AUC=0.9420, AUPRC=0.9517
[STABILITY] repeat=17/50, seed=20058, AUC=0.8527, AUPRC=0.8881
[STABILITY] repeat=18/50, seed=20059, AUC=0.8884, AUPRC=0.9091
[STABILITY] repeat=19/50, seed=20060, AUC=0.9107, AUPRC=0.9256
[STABILITY] repeat=20/50, seed=20061, AUC=0.8839, AUPRC=0.9029
[STABILITY] repeat=21/50, seed=20062, AUC=0.9286, AUPRC=0.9385
[STABILITY] repeat=22/50, seed=20063, AUC=0.8929, AUPRC=0.8818
[STABILITY] repeat=23/50, seed=20064, AUC=0.8661, AUPRC=0.9088
[STABILITY] repeat=24/50, seed=20065, AUC=0.8973, AUPRC=0.9075
[STABILITY] repeat=25/50, seed=20066, AUC=0.8929, AUPRC=0.9186
[STABILITY] repeat=26/50, seed=20067, AUC=0.8839, AUPRC=0.8910
[STABILITY] repeat=27/50, seed=20068, AUC=0.9375, AUPRC=0.9427
[STABILITY] repeat=28/50, seed=20069, AUC=0.9330, AUPRC=0.9401
[STABILITY] repeat=29/50, seed=20070, AUC=0.8571, AUPRC=0.8905
[STABILITY] repeat=30/50, seed=20071, AUC=0.8661, AUPRC=0.8880
[STABILITY] repeat=31/50, seed=20072, AUC=0.9196, AUPRC=0.9336
[STABILITY] repeat=32/50, seed=20073, AUC=0.8482, AUPRC=0.8857
[STABILITY] repeat=33/50, seed=20074, AUC=0.9330, AUPRC=0.9342
[STABILITY] repeat=34/50, seed=20075, AUC=0.8527, AUPRC=0.8816
[STABILITY] repeat=35/50, seed=20076, AUC=0.9330, AUPRC=0.9421
[STABILITY] repeat=36/50, seed=20077, AUC=0.8884, AUPRC=0.8880
[STABILITY] repeat=37/50, seed=20078, AUC=0.8795, AUPRC=0.8977
[STABILITY] repeat=38/50, seed=20079, AUC=0.8973, AUPRC=0.9242
[STABILITY] repeat=39/50, seed=20080, AUC=0.9241, AUPRC=0.9340
[STABILITY] repeat=40/50, seed=20081, AUC=0.9107, AUPRC=0.9119
[STABILITY] repeat=41/50, seed=20082, AUC=0.9554, AUPRC=0.9576
[STABILITY] repeat=42/50, seed=20083, AUC=0.9420, AUPRC=0.9425
[STABILITY] repeat=43/50, seed=20084, AUC=0.9330, AUPRC=0.9342
[STABILITY] repeat=44/50, seed=20085, AUC=0.9152, AUPRC=0.9011
[STABILITY] repeat=45/50, seed=20086, AUC=0.9152, AUPRC=0.9344
[STABILITY] repeat=46/50, seed=20087, AUC=0.8973, AUPRC=0.9040
[STABILITY] repeat=47/50, seed=20088, AUC=0.9643, AUPRC=0.9680
[STABILITY] repeat=48/50, seed=20089, AUC=0.8750, AUPRC=0.8633
[STABILITY] repeat=49/50, seed=20090, AUC=0.8705, AUPRC=0.8847
[STABILITY] repeat=50/50, seed=20091, AUC=0.8973, AUPRC=0.9121
[STABILITY] AUC median=0.9129, IQR=[0.8850, 0.9330], range=[0.8482, 0.9688]
[STABILITY] Running 50 repeated nested patient-level splits for A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9688, AUPRC=0.9707
[STABILITY] repeat=2/50, seed=20043, AUC=0.9464, AUPRC=0.9490
[STABILITY] repeat=3/50, seed=20044, AUC=0.9554, AUPRC=0.9614
[STABILITY] repeat=4/50, seed=20045, AUC=0.9643, AUPRC=0.9678
[STABILITY] repeat=5/50, seed=20046, AUC=0.9821, AUPRC=0.9815
[STABILITY] repeat=6/50, seed=20047, AUC=0.9375, AUPRC=0.9458
[STABILITY] repeat=7/50, seed=20048, AUC=0.9464, AUPRC=0.9503
[STABILITY] repeat=8/50, seed=20049, AUC=0.9330, AUPRC=0.9226
[STABILITY] repeat=9/50, seed=20050, AUC=0.9554, AUPRC=0.9576
[STABILITY] repeat=10/50, seed=20051, AUC=0.9241, AUPRC=0.9243
[STABILITY] repeat=11/50, seed=20052, AUC=0.9286, AUPRC=0.9375
[STABILITY] repeat=12/50, seed=20053, AUC=0.9286, AUPRC=0.9358
[STABILITY] repeat=13/50, seed=20054, AUC=0.9598, AUPRC=0.9656
[STABILITY] repeat=14/50, seed=20055, AUC=0.9420, AUPRC=0.9483
[STABILITY] repeat=15/50, seed=20056, AUC=0.9598, AUPRC=0.9618
[STABILITY] repeat=16/50, seed=20057, AUC=0.9286, AUPRC=0.9394
[STABILITY] repeat=17/50, seed=20058, AUC=0.9330, AUPRC=0.9467
[STABILITY] repeat=18/50, seed=20059, AUC=0.9732, AUPRC=0.9735
[STABILITY] repeat=19/50, seed=20060, AUC=0.9464, AUPRC=0.9572
[STABILITY] repeat=20/50, seed=20061, AUC=0.9420, AUPRC=0.9536
[STABILITY] repeat=21/50, seed=20062, AUC=0.9018, AUPRC=0.9284
[STABILITY] repeat=22/50, seed=20063, AUC=0.9241, AUPRC=0.9448
[STABILITY] repeat=23/50, seed=20064, AUC=0.9062, AUPRC=0.9351
[STABILITY] repeat=24/50, seed=20065, AUC=0.9509, AUPRC=0.9587
[STABILITY] repeat=25/50, seed=20066, AUC=0.9598, AUPRC=0.9652
[STABILITY] repeat=26/50, seed=20067, AUC=0.8929, AUPRC=0.8929
[STABILITY] repeat=27/50, seed=20068, AUC=0.9420, AUPRC=0.9475
[STABILITY] repeat=28/50, seed=20069, AUC=0.9554, AUPRC=0.9571
[STABILITY] repeat=29/50, seed=20070, AUC=0.8795, AUPRC=0.8689
[STABILITY] repeat=30/50, seed=20071, AUC=0.9598, AUPRC=0.9605
[STABILITY] repeat=31/50, seed=20072, AUC=0.9018, AUPRC=0.9199
[STABILITY] repeat=32/50, seed=20073, AUC=0.8795, AUPRC=0.8849
[STABILITY] repeat=33/50, seed=20074, AUC=0.9643, AUPRC=0.9673
[STABILITY] repeat=34/50, seed=20075, AUC=0.9643, AUPRC=0.9711
[STABILITY] repeat=35/50, seed=20076, AUC=0.9464, AUPRC=0.9499
[STABILITY] repeat=36/50, seed=20077, AUC=0.9598, AUPRC=0.9652
[STABILITY] repeat=37/50, seed=20078, AUC=0.8973, AUPRC=0.9081
[STABILITY] repeat=38/50, seed=20079, AUC=0.9777, AUPRC=0.9779
[STABILITY] repeat=39/50, seed=20080, AUC=0.9420, AUPRC=0.9411
[STABILITY] repeat=40/50, seed=20081, AUC=0.9509, AUPRC=0.9530
[STABILITY] repeat=41/50, seed=20082, AUC=0.9777, AUPRC=0.9779
[STABILITY] repeat=42/50, seed=20083, AUC=0.9821, AUPRC=0.9823
[STABILITY] repeat=43/50, seed=20084, AUC=0.9286, AUPRC=0.9361
[STABILITY] repeat=44/50, seed=20085, AUC=0.9598, AUPRC=0.9636
[STABILITY] repeat=45/50, seed=20086, AUC=0.8973, AUPRC=0.9164
[STABILITY] repeat=46/50, seed=20087, AUC=0.9509, AUPRC=0.9537
[STABILITY] repeat=47/50, seed=20088, AUC=0.9598, AUPRC=0.9593
[STABILITY] repeat=48/50, seed=20089, AUC=0.9375, AUPRC=0.9485
[STABILITY] repeat=49/50, seed=20090, AUC=0.9509, AUPRC=0.9563
[STABILITY] repeat=50/50, seed=20091, AUC=0.9509, AUPRC=0.9594
[STABILITY] AUC median=0.9464, IQR=[0.9286, 0.9598], range=[0.8795, 0.9821]
[STABILITY] Running 50 repeated nested patient-level splits for C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.8705, AUPRC=0.8908
[STABILITY] repeat=2/50, seed=20043, AUC=0.8393, AUPRC=0.8858
[STABILITY] repeat=3/50, seed=20044, AUC=0.7634, AUPRC=0.7503
[STABILITY] repeat=4/50, seed=20045, AUC=0.8482, AUPRC=0.8783
[STABILITY] repeat=5/50, seed=20046, AUC=0.8080, AUPRC=0.8595
[STABILITY] repeat=6/50, seed=20047, AUC=0.8438, AUPRC=0.8779
[STABILITY] repeat=7/50, seed=20048, AUC=0.8438, AUPRC=0.8658
[STABILITY] repeat=8/50, seed=20049, AUC=0.8616, AUPRC=0.9089
[STABILITY] repeat=9/50, seed=20050, AUC=0.8393, AUPRC=0.8984
[STABILITY] repeat=10/50, seed=20051, AUC=0.8125, AUPRC=0.8464
[STABILITY] repeat=11/50, seed=20052, AUC=0.7768, AUPRC=0.8278
[STABILITY] repeat=12/50, seed=20053, AUC=0.8214, AUPRC=0.8544
[STABILITY] repeat=13/50, seed=20054, AUC=0.8348, AUPRC=0.8609
[STABILITY] repeat=14/50, seed=20055, AUC=0.8929, AUPRC=0.9228
[STABILITY] repeat=15/50, seed=20056, AUC=0.8661, AUPRC=0.8650
[STABILITY] repeat=16/50, seed=20057, AUC=0.7812, AUPRC=0.8624
[STABILITY] repeat=17/50, seed=20058, AUC=0.8304, AUPRC=0.8738
[STABILITY] repeat=18/50, seed=20059, AUC=0.8571, AUPRC=0.8803
[STABILITY] repeat=19/50, seed=20060, AUC=0.7634, AUPRC=0.8029
[STABILITY] repeat=20/50, seed=20061, AUC=0.9152, AUPRC=0.9302
[STABILITY] repeat=21/50, seed=20062, AUC=0.8214, AUPRC=0.8679
[STABILITY] repeat=22/50, seed=20063, AUC=0.8259, AUPRC=0.8756
[STABILITY] repeat=23/50, seed=20064, AUC=0.8348, AUPRC=0.8825
[STABILITY] repeat=24/50, seed=20065, AUC=0.8571, AUPRC=0.8726
[STABILITY] repeat=25/50, seed=20066, AUC=0.7902, AUPRC=0.7957
[STABILITY] repeat=26/50, seed=20067, AUC=0.7545, AUPRC=0.7929
[STABILITY] repeat=27/50, seed=20068, AUC=0.7857, AUPRC=0.8065
[STABILITY] repeat=28/50, seed=20069, AUC=0.8304, AUPRC=0.8583
[STABILITY] repeat=29/50, seed=20070, AUC=0.6920, AUPRC=0.7228
[STABILITY] repeat=30/50, seed=20071, AUC=0.7768, AUPRC=0.8409
[STABILITY] repeat=31/50, seed=20072, AUC=0.7500, AUPRC=0.8072
[STABILITY] repeat=32/50, seed=20073, AUC=0.8304, AUPRC=0.8866
[STABILITY] repeat=33/50, seed=20074, AUC=0.8170, AUPRC=0.8518
[STABILITY] repeat=34/50, seed=20075, AUC=0.7991, AUPRC=0.8075
[STABILITY] repeat=35/50, seed=20076, AUC=0.7902, AUPRC=0.8210
[STABILITY] repeat=36/50, seed=20077, AUC=0.8259, AUPRC=0.8619
[STABILITY] repeat=37/50, seed=20078, AUC=0.8125, AUPRC=0.8680
[STABILITY] repeat=38/50, seed=20079, AUC=0.7946, AUPRC=0.8366
[STABILITY] repeat=39/50, seed=20080, AUC=0.8125, AUPRC=0.8747
[STABILITY] repeat=40/50, seed=20081, AUC=0.7902, AUPRC=0.8376
[STABILITY] repeat=41/50, seed=20082, AUC=0.7723, AUPRC=0.8351
[STABILITY] repeat=42/50, seed=20083, AUC=0.8348, AUPRC=0.8705
[STABILITY] repeat=43/50, seed=20084, AUC=0.8125, AUPRC=0.8322
[STABILITY] repeat=44/50, seed=20085, AUC=0.8125, AUPRC=0.8120
[STABILITY] repeat=45/50, seed=20086, AUC=0.8170, AUPRC=0.8244
[STABILITY] repeat=46/50, seed=20087, AUC=0.8438, AUPRC=0.8767
[STABILITY] repeat=47/50, seed=20088, AUC=0.7812, AUPRC=0.8376
[STABILITY] repeat=48/50, seed=20089, AUC=0.8036, AUPRC=0.8603
[STABILITY] repeat=49/50, seed=20090, AUC=0.8348, AUPRC=0.8985
[STABILITY] repeat=50/50, seed=20091, AUC=0.7902, AUPRC=0.8160
[STABILITY] AUC median=0.8170, IQR=[0.7902, 0.8382], range=[0.6920, 0.9152]
[STABILITY] Running 50 repeated nested patient-level splits for C5_STANDARDIZED_BORDER05_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.5670, AUPRC=0.5727
[STABILITY] repeat=2/50, seed=20043, AUC=0.6250, AUPRC=0.6452
[STABILITY] repeat=3/50, seed=20044, AUC=0.6295, AUPRC=0.6215
[STABILITY] repeat=4/50, seed=20045, AUC=0.6652, AUPRC=0.6478
[STABILITY] repeat=5/50, seed=20046, AUC=0.5045, AUPRC=0.4894
[STABILITY] repeat=6/50, seed=20047, AUC=0.6696, AUPRC=0.6774
[STABILITY] repeat=7/50, seed=20048, AUC=0.7679, AUPRC=0.7848
[STABILITY] repeat=8/50, seed=20049, AUC=0.6027, AUPRC=0.5895
[STABILITY] repeat=9/50, seed=20050, AUC=0.7232, AUPRC=0.6949
[STABILITY] repeat=10/50, seed=20051, AUC=0.6562, AUPRC=0.6848
[STABILITY] repeat=11/50, seed=20052, AUC=0.5714, AUPRC=0.5228
[STABILITY] repeat=12/50, seed=20053, AUC=0.4732, AUPRC=0.4987
[STABILITY] repeat=13/50, seed=20054, AUC=0.6295, AUPRC=0.5960
[STABILITY] repeat=14/50, seed=20055, AUC=0.6116, AUPRC=0.6330
[STABILITY] repeat=15/50, seed=20056, AUC=0.6384, AUPRC=0.5994
[STABILITY] repeat=16/50, seed=20057, AUC=0.5982, AUPRC=0.6068
[STABILITY] repeat=17/50, seed=20058, AUC=0.5848, AUPRC=0.5965
[STABILITY] repeat=18/50, seed=20059, AUC=0.5223, AUPRC=0.5354
[STABILITY] repeat=19/50, seed=20060, AUC=0.4420, AUPRC=0.5220
[STABILITY] repeat=20/50, seed=20061, AUC=0.5223, AUPRC=0.5369
[STABILITY] repeat=21/50, seed=20062, AUC=0.4688, AUPRC=0.4658
[STABILITY] repeat=22/50, seed=20063, AUC=0.5625, AUPRC=0.5653
[STABILITY] repeat=23/50, seed=20064, AUC=0.5670, AUPRC=0.5312
[STABILITY] repeat=24/50, seed=20065, AUC=0.6652, AUPRC=0.6621
[STABILITY] repeat=25/50, seed=20066, AUC=0.6473, AUPRC=0.6052
[STABILITY] repeat=26/50, seed=20067, AUC=0.6741, AUPRC=0.6140
[STABILITY] repeat=27/50, seed=20068, AUC=0.5179, AUPRC=0.4921
[STABILITY] repeat=28/50, seed=20069, AUC=0.6473, AUPRC=0.5746
[STABILITY] repeat=29/50, seed=20070, AUC=0.6741, AUPRC=0.6880
[STABILITY] repeat=30/50, seed=20071, AUC=0.6920, AUPRC=0.6781
[STABILITY] repeat=31/50, seed=20072, AUC=0.6741, AUPRC=0.6547
[STABILITY] repeat=32/50, seed=20073, AUC=0.6562, AUPRC=0.6730
[STABILITY] repeat=33/50, seed=20074, AUC=0.5938, AUPRC=0.6355
[STABILITY] repeat=34/50, seed=20075, AUC=0.6384, AUPRC=0.6466
[STABILITY] repeat=35/50, seed=20076, AUC=0.5893, AUPRC=0.5987
[STABILITY] repeat=36/50, seed=20077, AUC=0.4509, AUPRC=0.4960
[STABILITY] repeat=37/50, seed=20078, AUC=0.6250, AUPRC=0.5926
[STABILITY] repeat=38/50, seed=20079, AUC=0.7143, AUPRC=0.6951
[STABILITY] repeat=39/50, seed=20080, AUC=0.6920, AUPRC=0.6961
[STABILITY] repeat=40/50, seed=20081, AUC=0.6116, AUPRC=0.5834
[STABILITY] repeat=41/50, seed=20082, AUC=0.5446, AUPRC=0.5681
[STABILITY] repeat=42/50, seed=20083, AUC=0.6652, AUPRC=0.7064
[STABILITY] repeat=43/50, seed=20084, AUC=0.5938, AUPRC=0.5940
[STABILITY] repeat=44/50, seed=20085, AUC=0.6875, AUPRC=0.6722
[STABILITY] repeat=45/50, seed=20086, AUC=0.5804, AUPRC=0.5244
[STABILITY] repeat=46/50, seed=20087, AUC=0.6518, AUPRC=0.6653
[STABILITY] repeat=47/50, seed=20088, AUC=0.5134, AUPRC=0.5141
[STABILITY] repeat=48/50, seed=20089, AUC=0.4688, AUPRC=0.5029
[STABILITY] repeat=49/50, seed=20090, AUC=0.6741, AUPRC=0.6556
[STABILITY] repeat=50/50, seed=20091, AUC=0.5893, AUPRC=0.6304
[STABILITY] AUC median=0.6183, IQR=[0.5670, 0.6652], range=[0.4420, 0.7679]
[STABILITY] Running 50 repeated nested patient-level splits for C6_STANDARDIZED_BORDER10_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.6518, AUPRC=0.6432
[STABILITY] repeat=2/50, seed=20043, AUC=0.6295, AUPRC=0.6776
[STABILITY] repeat=3/50, seed=20044, AUC=0.7277, AUPRC=0.8224
[STABILITY] repeat=4/50, seed=20045, AUC=0.7768, AUPRC=0.8333
[STABILITY] repeat=5/50, seed=20046, AUC=0.6071, AUPRC=0.6316
[STABILITY] repeat=6/50, seed=20047, AUC=0.6830, AUPRC=0.7622
[STABILITY] repeat=7/50, seed=20048, AUC=0.7143, AUPRC=0.7939
[STABILITY] repeat=8/50, seed=20049, AUC=0.6964, AUPRC=0.7528
[STABILITY] repeat=9/50, seed=20050, AUC=0.7902, AUPRC=0.8501
[STABILITY] repeat=10/50, seed=20051, AUC=0.7812, AUPRC=0.8531
[STABILITY] repeat=11/50, seed=20052, AUC=0.7054, AUPRC=0.7834
[STABILITY] repeat=12/50, seed=20053, AUC=0.6875, AUPRC=0.7575
[STABILITY] repeat=13/50, seed=20054, AUC=0.6964, AUPRC=0.7128
[STABILITY] repeat=14/50, seed=20055, AUC=0.7009, AUPRC=0.7316
[STABILITY] repeat=15/50, seed=20056, AUC=0.7232, AUPRC=0.8210
[STABILITY] repeat=16/50, seed=20057, AUC=0.6384, AUPRC=0.6918
[STABILITY] repeat=17/50, seed=20058, AUC=0.7143, AUPRC=0.7161
[STABILITY] repeat=18/50, seed=20059, AUC=0.7009, AUPRC=0.7174
[STABILITY] repeat=19/50, seed=20060, AUC=0.6652, AUPRC=0.6599
[STABILITY] repeat=20/50, seed=20061, AUC=0.7054, AUPRC=0.7592
[STABILITY] repeat=21/50, seed=20062, AUC=0.6652, AUPRC=0.6798
[STABILITY] repeat=22/50, seed=20063, AUC=0.6652, AUPRC=0.7365
[STABILITY] repeat=23/50, seed=20064, AUC=0.7500, AUPRC=0.8256
[STABILITY] repeat=24/50, seed=20065, AUC=0.7009, AUPRC=0.7457
[STABILITY] repeat=25/50, seed=20066, AUC=0.7321, AUPRC=0.8059
[STABILITY] repeat=26/50, seed=20067, AUC=0.6339, AUPRC=0.6089
[STABILITY] repeat=27/50, seed=20068, AUC=0.6518, AUPRC=0.7364
[STABILITY] repeat=28/50, seed=20069, AUC=0.6830, AUPRC=0.7376
[STABILITY] repeat=29/50, seed=20070, AUC=0.6786, AUPRC=0.7443
[STABILITY] repeat=30/50, seed=20071, AUC=0.7589, AUPRC=0.8295
[STABILITY] repeat=31/50, seed=20072, AUC=0.7232, AUPRC=0.7749
[STABILITY] repeat=32/50, seed=20073, AUC=0.7411, AUPRC=0.8204
[STABILITY] repeat=33/50, seed=20074, AUC=0.7009, AUPRC=0.7169
[STABILITY] repeat=34/50, seed=20075, AUC=0.7500, AUPRC=0.8138
[STABILITY] repeat=35/50, seed=20076, AUC=0.7054, AUPRC=0.7830
[STABILITY] repeat=36/50, seed=20077, AUC=0.7188, AUPRC=0.8033
[STABILITY] repeat=37/50, seed=20078, AUC=0.6741, AUPRC=0.7263
[STABILITY] repeat=38/50, seed=20079, AUC=0.6071, AUPRC=0.6267
[STABILITY] repeat=39/50, seed=20080, AUC=0.6786, AUPRC=0.7256
[STABILITY] repeat=40/50, seed=20081, AUC=0.6473, AUPRC=0.7127
[STABILITY] repeat=41/50, seed=20082, AUC=0.7411, AUPRC=0.8120
[STABILITY] repeat=42/50, seed=20083, AUC=0.7188, AUPRC=0.7798
[STABILITY] repeat=43/50, seed=20084, AUC=0.7321, AUPRC=0.7756
[STABILITY] repeat=44/50, seed=20085, AUC=0.7589, AUPRC=0.8329
[STABILITY] repeat=45/50, seed=20086, AUC=0.7500, AUPRC=0.8176
[STABILITY] repeat=46/50, seed=20087, AUC=0.6562, AUPRC=0.6728
[STABILITY] repeat=47/50, seed=20088, AUC=0.6696, AUPRC=0.7337
[STABILITY] repeat=48/50, seed=20089, AUC=0.6473, AUPRC=0.7248
[STABILITY] repeat=49/50, seed=20090, AUC=0.7009, AUPRC=0.7303
[STABILITY] repeat=50/50, seed=20091, AUC=0.7411, AUPRC=0.8032
[STABILITY] AUC median=0.7009, IQR=[0.6663, 0.7310], range=[0.6071, 0.7902]
[STABILITY] Running 50 repeated nested patient-level splits for C7_DETECTED_PADDING_MASK_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.5759, AUPRC=0.6542
[STABILITY] repeat=2/50, seed=20043, AUC=0.7188, AUPRC=0.7469
[STABILITY] repeat=3/50, seed=20044, AUC=0.7232, AUPRC=0.7826
[STABILITY] repeat=4/50, seed=20045, AUC=0.6161, AUPRC=0.6391
[STABILITY] repeat=5/50, seed=20046, AUC=0.6339, AUPRC=0.6704
[STABILITY] repeat=6/50, seed=20047, AUC=0.7054, AUPRC=0.7556
[STABILITY] repeat=7/50, seed=20048, AUC=0.7679, AUPRC=0.7948
[STABILITY] repeat=8/50, seed=20049, AUC=0.6384, AUPRC=0.6632
[STABILITY] repeat=9/50, seed=20050, AUC=0.7277, AUPRC=0.7709
[STABILITY] repeat=10/50, seed=20051, AUC=0.7455, AUPRC=0.7627
[STABILITY] repeat=11/50, seed=20052, AUC=0.7321, AUPRC=0.7350
[STABILITY] repeat=12/50, seed=20053, AUC=0.5134, AUPRC=0.5944
[STABILITY] repeat=13/50, seed=20054, AUC=0.6518, AUPRC=0.7323
[STABILITY] repeat=14/50, seed=20055, AUC=0.5893, AUPRC=0.6265
[STABILITY] repeat=15/50, seed=20056, AUC=0.6920, AUPRC=0.7551
[STABILITY] repeat=16/50, seed=20057, AUC=0.6786, AUPRC=0.7382
[STABILITY] repeat=17/50, seed=20058, AUC=0.6741, AUPRC=0.7355
[STABILITY] repeat=18/50, seed=20059, AUC=0.6741, AUPRC=0.7661
[STABILITY] repeat=19/50, seed=20060, AUC=0.5580, AUPRC=0.6143
[STABILITY] repeat=20/50, seed=20061, AUC=0.6473, AUPRC=0.7382
[STABILITY] repeat=21/50, seed=20062, AUC=0.6518, AUPRC=0.6562
[STABILITY] repeat=22/50, seed=20063, AUC=0.7500, AUPRC=0.7684
[STABILITY] repeat=23/50, seed=20064, AUC=0.6607, AUPRC=0.6805
[STABILITY] repeat=24/50, seed=20065, AUC=0.6384, AUPRC=0.6695
[STABILITY] repeat=25/50, seed=20066, AUC=0.6116, AUPRC=0.6995
[STABILITY] repeat=26/50, seed=20067, AUC=0.5982, AUPRC=0.6575
[STABILITY] repeat=27/50, seed=20068, AUC=0.6696, AUPRC=0.6639
[STABILITY] repeat=28/50, seed=20069, AUC=0.7188, AUPRC=0.7351
[STABILITY] repeat=29/50, seed=20070, AUC=0.7054, AUPRC=0.7408
[STABILITY] repeat=30/50, seed=20071, AUC=0.7009, AUPRC=0.7219
[STABILITY] repeat=31/50, seed=20072, AUC=0.7009, AUPRC=0.7427
[STABILITY] repeat=32/50, seed=20073, AUC=0.7143, AUPRC=0.7622
[STABILITY] repeat=33/50, seed=20074, AUC=0.7143, AUPRC=0.7593
[STABILITY] repeat=34/50, seed=20075, AUC=0.7366, AUPRC=0.7471
[STABILITY] repeat=35/50, seed=20076, AUC=0.6652, AUPRC=0.7215
[STABILITY] repeat=36/50, seed=20077, AUC=0.7098, AUPRC=0.7956
[STABILITY] repeat=37/50, seed=20078, AUC=0.7009, AUPRC=0.7231
[STABILITY] repeat=38/50, seed=20079, AUC=0.7545, AUPRC=0.7612
[STABILITY] repeat=39/50, seed=20080, AUC=0.5982, AUPRC=0.6526
[STABILITY] repeat=40/50, seed=20081, AUC=0.6473, AUPRC=0.6923
[STABILITY] repeat=41/50, seed=20082, AUC=0.7411, AUPRC=0.7911
[STABILITY] repeat=42/50, seed=20083, AUC=0.6429, AUPRC=0.6944
[STABILITY] repeat=43/50, seed=20084, AUC=0.7143, AUPRC=0.7336
[STABILITY] repeat=44/50, seed=20085, AUC=0.7500, AUPRC=0.7741
[STABILITY] repeat=45/50, seed=20086, AUC=0.7500, AUPRC=0.8045
[STABILITY] repeat=46/50, seed=20087, AUC=0.6250, AUPRC=0.6572
[STABILITY] repeat=47/50, seed=20088, AUC=0.8036, AUPRC=0.8576
[STABILITY] repeat=48/50, seed=20089, AUC=0.5536, AUPRC=0.5823
[STABILITY] repeat=49/50, seed=20090, AUC=0.7277, AUPRC=0.7519
[STABILITY] repeat=50/50, seed=20091, AUC=0.7277, AUPRC=0.7321
[STABILITY] AUC median=0.6964, IQR=[0.6395, 0.7266], range=[0.5134, 0.8036]
[STABILITY] Running 50 repeated nested patient-level splits for C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9554, AUPRC=0.9634
[STABILITY] repeat=2/50, seed=20043, AUC=0.8750, AUPRC=0.8715
[STABILITY] repeat=3/50, seed=20044, AUC=0.8304, AUPRC=0.8681
[STABILITY] repeat=4/50, seed=20045, AUC=0.7812, AUPRC=0.8183
[STABILITY] repeat=5/50, seed=20046, AUC=0.8438, AUPRC=0.8697
[STABILITY] repeat=6/50, seed=20047, AUC=0.8571, AUPRC=0.8339
[STABILITY] repeat=7/50, seed=20048, AUC=0.7991, AUPRC=0.7945
[STABILITY] repeat=8/50, seed=20049, AUC=0.9152, AUPRC=0.9172
[STABILITY] repeat=9/50, seed=20050, AUC=0.8348, AUPRC=0.8332
[STABILITY] repeat=10/50, seed=20051, AUC=0.8973, AUPRC=0.8909
[STABILITY] repeat=11/50, seed=20052, AUC=0.7812, AUPRC=0.8507
[STABILITY] repeat=12/50, seed=20053, AUC=0.8304, AUPRC=0.8099
[STABILITY] repeat=13/50, seed=20054, AUC=0.7946, AUPRC=0.8265
[STABILITY] repeat=14/50, seed=20055, AUC=0.8438, AUPRC=0.8800
[STABILITY] repeat=15/50, seed=20056, AUC=0.8080, AUPRC=0.8160
[STABILITY] repeat=16/50, seed=20057, AUC=0.8929, AUPRC=0.9053
[STABILITY] repeat=17/50, seed=20058, AUC=0.8929, AUPRC=0.9030
[STABILITY] repeat=18/50, seed=20059, AUC=0.8750, AUPRC=0.9083
[STABILITY] repeat=19/50, seed=20060, AUC=0.8884, AUPRC=0.8940
[STABILITY] repeat=20/50, seed=20061, AUC=0.8705, AUPRC=0.9061
[STABILITY] repeat=21/50, seed=20062, AUC=0.7723, AUPRC=0.7849
[STABILITY] repeat=22/50, seed=20063, AUC=0.8214, AUPRC=0.8592
[STABILITY] repeat=23/50, seed=20064, AUC=0.7946, AUPRC=0.7743
[STABILITY] repeat=24/50, seed=20065, AUC=0.8839, AUPRC=0.9082
[STABILITY] repeat=25/50, seed=20066, AUC=0.8616, AUPRC=0.8905
[STABILITY] repeat=26/50, seed=20067, AUC=0.8393, AUPRC=0.7819
[STABILITY] repeat=27/50, seed=20068, AUC=0.9286, AUPRC=0.9291
[STABILITY] repeat=28/50, seed=20069, AUC=0.9152, AUPRC=0.9293
[STABILITY] repeat=29/50, seed=20070, AUC=0.8036, AUPRC=0.8026
[STABILITY] repeat=30/50, seed=20071, AUC=0.8259, AUPRC=0.8385
[STABILITY] repeat=31/50, seed=20072, AUC=0.8884, AUPRC=0.9086
[STABILITY] repeat=32/50, seed=20073, AUC=0.8393, AUPRC=0.8612
[STABILITY] repeat=33/50, seed=20074, AUC=0.8214, AUPRC=0.8661
[STABILITY] repeat=34/50, seed=20075, AUC=0.8929, AUPRC=0.9155
[STABILITY] repeat=35/50, seed=20076, AUC=0.8170, AUPRC=0.8313
[STABILITY] repeat=36/50, seed=20077, AUC=0.8661, AUPRC=0.8993
[STABILITY] repeat=37/50, seed=20078, AUC=0.8482, AUPRC=0.8807
[STABILITY] repeat=38/50, seed=20079, AUC=0.8348, AUPRC=0.8359
[STABILITY] repeat=39/50, seed=20080, AUC=0.8973, AUPRC=0.9151
[STABILITY] repeat=40/50, seed=20081, AUC=0.8393, AUPRC=0.8343
[STABILITY] repeat=41/50, seed=20082, AUC=0.8750, AUPRC=0.8702
[STABILITY] repeat=42/50, seed=20083, AUC=0.8438, AUPRC=0.8547
[STABILITY] repeat=43/50, seed=20084, AUC=0.9196, AUPRC=0.9256
[STABILITY] repeat=44/50, seed=20085, AUC=0.8705, AUPRC=0.9018
[STABILITY] repeat=45/50, seed=20086, AUC=0.8125, AUPRC=0.8294
[STABILITY] repeat=46/50, seed=20087, AUC=0.8259, AUPRC=0.8188
[STABILITY] repeat=47/50, seed=20088, AUC=0.8080, AUPRC=0.8791
[STABILITY] repeat=48/50, seed=20089, AUC=0.8438, AUPRC=0.8774
[STABILITY] repeat=49/50, seed=20090, AUC=0.8527, AUPRC=0.8929
[STABILITY] repeat=50/50, seed=20091, AUC=0.9152, AUPRC=0.9263
[STABILITY] AUC median=0.8438, IQR=[0.8225, 0.8873], range=[0.7723, 0.9554]
[STABILITY] Running 50 repeated nested patient-level splits for C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9018, AUPRC=0.9222
[STABILITY] repeat=2/50, seed=20043, AUC=0.8482, AUPRC=0.8990
[STABILITY] repeat=3/50, seed=20044, AUC=0.7991, AUPRC=0.8617
[STABILITY] repeat=4/50, seed=20045, AUC=0.8080, AUPRC=0.8626
[STABILITY] repeat=5/50, seed=20046, AUC=0.8214, AUPRC=0.8607
[STABILITY] repeat=6/50, seed=20047, AUC=0.7321, AUPRC=0.7185
[STABILITY] repeat=7/50, seed=20048, AUC=0.7679, AUPRC=0.7763
[STABILITY] repeat=8/50, seed=20049, AUC=0.8705, AUPRC=0.8713
[STABILITY] repeat=9/50, seed=20050, AUC=0.8705, AUPRC=0.9002
[STABILITY] repeat=10/50, seed=20051, AUC=0.8705, AUPRC=0.9010
[STABILITY] repeat=11/50, seed=20052, AUC=0.7723, AUPRC=0.8357
[STABILITY] repeat=12/50, seed=20053, AUC=0.7812, AUPRC=0.7861
[STABILITY] repeat=13/50, seed=20054, AUC=0.8170, AUPRC=0.8442
[STABILITY] repeat=14/50, seed=20055, AUC=0.8125, AUPRC=0.8435
[STABILITY] repeat=15/50, seed=20056, AUC=0.7857, AUPRC=0.8326
[STABILITY] repeat=16/50, seed=20057, AUC=0.8527, AUPRC=0.8965
[STABILITY] repeat=17/50, seed=20058, AUC=0.9018, AUPRC=0.9304
[STABILITY] repeat=18/50, seed=20059, AUC=0.8661, AUPRC=0.8723
[STABILITY] repeat=19/50, seed=20060, AUC=0.8705, AUPRC=0.8790
[STABILITY] repeat=20/50, seed=20061, AUC=0.8661, AUPRC=0.8622
[STABILITY] repeat=21/50, seed=20062, AUC=0.8438, AUPRC=0.9006
[STABILITY] repeat=22/50, seed=20063, AUC=0.7768, AUPRC=0.7798
[STABILITY] repeat=23/50, seed=20064, AUC=0.7946, AUPRC=0.8614
[STABILITY] repeat=24/50, seed=20065, AUC=0.8795, AUPRC=0.9130
[STABILITY] repeat=25/50, seed=20066, AUC=0.8170, AUPRC=0.8373
[STABILITY] repeat=26/50, seed=20067, AUC=0.8125, AUPRC=0.8520
[STABILITY] repeat=27/50, seed=20068, AUC=0.8571, AUPRC=0.8694
[STABILITY] repeat=28/50, seed=20069, AUC=0.8884, AUPRC=0.9085
[STABILITY] repeat=29/50, seed=20070, AUC=0.8482, AUPRC=0.8809
[STABILITY] repeat=30/50, seed=20071, AUC=0.7679, AUPRC=0.7932
[STABILITY] repeat=31/50, seed=20072, AUC=0.8616, AUPRC=0.9089
[STABILITY] repeat=32/50, seed=20073, AUC=0.8839, AUPRC=0.9108
[STABILITY] repeat=33/50, seed=20074, AUC=0.7366, AUPRC=0.7825
[STABILITY] repeat=34/50, seed=20075, AUC=0.8929, AUPRC=0.9163
[STABILITY] repeat=35/50, seed=20076, AUC=0.7768, AUPRC=0.8042
[STABILITY] repeat=36/50, seed=20077, AUC=0.8393, AUPRC=0.8540
[STABILITY] repeat=37/50, seed=20078, AUC=0.8170, AUPRC=0.8684
[STABILITY] repeat=38/50, seed=20079, AUC=0.8929, AUPRC=0.9092
[STABILITY] repeat=39/50, seed=20080, AUC=0.7857, AUPRC=0.8340
[STABILITY] repeat=40/50, seed=20081, AUC=0.8705, AUPRC=0.8887
[STABILITY] repeat=41/50, seed=20082, AUC=0.8750, AUPRC=0.9058
[STABILITY] repeat=42/50, seed=20083, AUC=0.8571, AUPRC=0.8912
[STABILITY] repeat=43/50, seed=20084, AUC=0.8661, AUPRC=0.8873
[STABILITY] repeat=44/50, seed=20085, AUC=0.8304, AUPRC=0.8584
[STABILITY] repeat=45/50, seed=20086, AUC=0.8304, AUPRC=0.8876
[STABILITY] repeat=46/50, seed=20087, AUC=0.8348, AUPRC=0.8892
[STABILITY] repeat=47/50, seed=20088, AUC=0.8080, AUPRC=0.8583
[STABILITY] repeat=48/50, seed=20089, AUC=0.8750, AUPRC=0.9068
[STABILITY] repeat=49/50, seed=20090, AUC=0.8438, AUPRC=0.8852
[STABILITY] repeat=50/50, seed=20091, AUC=0.8839, AUPRC=0.8748
[STABILITY] AUC median=0.8438, IQR=[0.8080, 0.8705], range=[0.7321, 0.9018]
[STABILITY] Running 50 repeated nested patient-level splits for C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.7545, AUPRC=0.7824
[STABILITY] repeat=2/50, seed=20043, AUC=0.6562, AUPRC=0.6243
[STABILITY] repeat=3/50, seed=20044, AUC=0.7232, AUPRC=0.7414
[STABILITY] repeat=4/50, seed=20045, AUC=0.6585, AUPRC=0.6389
[STABILITY] repeat=5/50, seed=20046, AUC=0.7054, AUPRC=0.7598
[STABILITY] repeat=6/50, seed=20047, AUC=0.6875, AUPRC=0.7479
[STABILITY] repeat=7/50, seed=20048, AUC=0.6786, AUPRC=0.7073
[STABILITY] repeat=8/50, seed=20049, AUC=0.6027, AUPRC=0.6602
[STABILITY] repeat=9/50, seed=20050, AUC=0.7545, AUPRC=0.7672
[STABILITY] repeat=10/50, seed=20051, AUC=0.6875, AUPRC=0.5976
[STABILITY] repeat=11/50, seed=20052, AUC=0.6518, AUPRC=0.6957
[STABILITY] repeat=12/50, seed=20053, AUC=0.6786, AUPRC=0.6943
[STABILITY] repeat=13/50, seed=20054, AUC=0.7723, AUPRC=0.8152
[STABILITY] repeat=14/50, seed=20055, AUC=0.7009, AUPRC=0.7604
[STABILITY] repeat=15/50, seed=20056, AUC=0.6473, AUPRC=0.7317
[STABILITY] repeat=16/50, seed=20057, AUC=0.6696, AUPRC=0.6951
[STABILITY] repeat=17/50, seed=20058, AUC=0.7143, AUPRC=0.7219
[STABILITY] repeat=18/50, seed=20059, AUC=0.7143, AUPRC=0.7341
[STABILITY] repeat=19/50, seed=20060, AUC=0.6116, AUPRC=0.7074
[STABILITY] repeat=20/50, seed=20061, AUC=0.7277, AUPRC=0.7604
[STABILITY] repeat=21/50, seed=20062, AUC=0.6116, AUPRC=0.6425
[STABILITY] repeat=22/50, seed=20063, AUC=0.6116, AUPRC=0.6623
[STABILITY] repeat=23/50, seed=20064, AUC=0.7143, AUPRC=0.7351
[STABILITY] repeat=24/50, seed=20065, AUC=0.6429, AUPRC=0.6715
[STABILITY] repeat=25/50, seed=20066, AUC=0.6607, AUPRC=0.6922
[STABILITY] repeat=26/50, seed=20067, AUC=0.6116, AUPRC=0.6715
[STABILITY] repeat=27/50, seed=20068, AUC=0.7857, AUPRC=0.8314
[STABILITY] repeat=28/50, seed=20069, AUC=0.6250, AUPRC=0.6759
[STABILITY] repeat=29/50, seed=20070, AUC=0.6607, AUPRC=0.6521
[STABILITY] repeat=30/50, seed=20071, AUC=0.7455, AUPRC=0.7595
[STABILITY] repeat=31/50, seed=20072, AUC=0.7723, AUPRC=0.7539
[STABILITY] repeat=32/50, seed=20073, AUC=0.5938, AUPRC=0.6086
[STABILITY] repeat=33/50, seed=20074, AUC=0.7321, AUPRC=0.7050
[STABILITY] repeat=34/50, seed=20075, AUC=0.6964, AUPRC=0.7500
[STABILITY] repeat=35/50, seed=20076, AUC=0.6830, AUPRC=0.7018
[STABILITY] repeat=36/50, seed=20077, AUC=0.6607, AUPRC=0.6707
[STABILITY] repeat=37/50, seed=20078, AUC=0.6964, AUPRC=0.7094
[STABILITY] repeat=38/50, seed=20079, AUC=0.6875, AUPRC=0.7246
[STABILITY] repeat=39/50, seed=20080, AUC=0.6920, AUPRC=0.7546
[STABILITY] repeat=40/50, seed=20081, AUC=0.5625, AUPRC=0.5582
[STABILITY] repeat=41/50, seed=20082, AUC=0.7902, AUPRC=0.8141
[STABILITY] repeat=42/50, seed=20083, AUC=0.7098, AUPRC=0.7723
[STABILITY] repeat=43/50, seed=20084, AUC=0.6964, AUPRC=0.7340
[STABILITY] repeat=44/50, seed=20085, AUC=0.6920, AUPRC=0.7088
[STABILITY] repeat=45/50, seed=20086, AUC=0.6652, AUPRC=0.7368
[STABILITY] repeat=46/50, seed=20087, AUC=0.6562, AUPRC=0.6220
[STABILITY] repeat=47/50, seed=20088, AUC=0.7768, AUPRC=0.7392
[STABILITY] repeat=48/50, seed=20089, AUC=0.7009, AUPRC=0.7431
[STABILITY] repeat=49/50, seed=20090, AUC=0.6518, AUPRC=0.6403
[STABILITY] repeat=50/50, seed=20091, AUC=0.7098, AUPRC=0.6673
[STABILITY] AUC median=0.6875, IQR=[0.6562, 0.7143], range=[0.5625, 0.7902]
[STABILITY] Running 50 repeated nested patient-level splits for C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.8080, AUPRC=0.8193
[STABILITY] repeat=2/50, seed=20043, AUC=0.8214, AUPRC=0.8392
[STABILITY] repeat=3/50, seed=20044, AUC=0.8080, AUPRC=0.8291
[STABILITY] repeat=4/50, seed=20045, AUC=0.8125, AUPRC=0.8103
[STABILITY] repeat=5/50, seed=20046, AUC=0.7946, AUPRC=0.8289
[STABILITY] repeat=6/50, seed=20047, AUC=0.7188, AUPRC=0.7469
[STABILITY] repeat=7/50, seed=20048, AUC=0.7946, AUPRC=0.7992
[STABILITY] repeat=8/50, seed=20049, AUC=0.8348, AUPRC=0.8582
[STABILITY] repeat=9/50, seed=20050, AUC=0.7500, AUPRC=0.7598
[STABILITY] repeat=10/50, seed=20051, AUC=0.8036, AUPRC=0.7960
[STABILITY] repeat=11/50, seed=20052, AUC=0.8080, AUPRC=0.8347
[STABILITY] repeat=12/50, seed=20053, AUC=0.7723, AUPRC=0.7932
[STABILITY] repeat=13/50, seed=20054, AUC=0.7902, AUPRC=0.8160
[STABILITY] repeat=14/50, seed=20055, AUC=0.7857, AUPRC=0.8003
[STABILITY] repeat=15/50, seed=20056, AUC=0.7634, AUPRC=0.8048
[STABILITY] repeat=16/50, seed=20057, AUC=0.7991, AUPRC=0.7795
[STABILITY] repeat=17/50, seed=20058, AUC=0.7902, AUPRC=0.8309
[STABILITY] repeat=18/50, seed=20059, AUC=0.8170, AUPRC=0.8258
[STABILITY] repeat=19/50, seed=20060, AUC=0.7991, AUPRC=0.8140
[STABILITY] repeat=20/50, seed=20061, AUC=0.7723, AUPRC=0.7747
[STABILITY] repeat=21/50, seed=20062, AUC=0.7902, AUPRC=0.7839
[STABILITY] repeat=22/50, seed=20063, AUC=0.7500, AUPRC=0.7666
[STABILITY] repeat=23/50, seed=20064, AUC=0.7188, AUPRC=0.7098
[STABILITY] repeat=24/50, seed=20065, AUC=0.8304, AUPRC=0.8478
[STABILITY] repeat=25/50, seed=20066, AUC=0.7411, AUPRC=0.7539
[STABILITY] repeat=26/50, seed=20067, AUC=0.8214, AUPRC=0.8446
[STABILITY] repeat=27/50, seed=20068, AUC=0.7723, AUPRC=0.7669
[STABILITY] repeat=28/50, seed=20069, AUC=0.8036, AUPRC=0.8228
[STABILITY] repeat=29/50, seed=20070, AUC=0.8393, AUPRC=0.8422
[STABILITY] repeat=30/50, seed=20071, AUC=0.7589, AUPRC=0.7840
[STABILITY] repeat=31/50, seed=20072, AUC=0.8304, AUPRC=0.8522
[STABILITY] repeat=32/50, seed=20073, AUC=0.7812, AUPRC=0.7985
[STABILITY] repeat=33/50, seed=20074, AUC=0.8214, AUPRC=0.8446
[STABILITY] repeat=34/50, seed=20075, AUC=0.8080, AUPRC=0.8236
[STABILITY] repeat=35/50, seed=20076, AUC=0.7991, AUPRC=0.8154
[STABILITY] repeat=36/50, seed=20077, AUC=0.7679, AUPRC=0.7719
[STABILITY] repeat=37/50, seed=20078, AUC=0.7857, AUPRC=0.8157
[STABILITY] repeat=38/50, seed=20079, AUC=0.8125, AUPRC=0.8208
[STABILITY] repeat=39/50, seed=20080, AUC=0.7857, AUPRC=0.7493
[STABILITY] repeat=40/50, seed=20081, AUC=0.8036, AUPRC=0.8177
[STABILITY] repeat=41/50, seed=20082, AUC=0.8259, AUPRC=0.7896
[STABILITY] repeat=42/50, seed=20083, AUC=0.7812, AUPRC=0.7558
[STABILITY] repeat=43/50, seed=20084, AUC=0.8304, AUPRC=0.8326
[STABILITY] repeat=44/50, seed=20085, AUC=0.7098, AUPRC=0.7571
[STABILITY] repeat=45/50, seed=20086, AUC=0.7946, AUPRC=0.7685
[STABILITY] repeat=46/50, seed=20087, AUC=0.7857, AUPRC=0.7700
[STABILITY] repeat=47/50, seed=20088, AUC=0.7723, AUPRC=0.8050
[STABILITY] repeat=48/50, seed=20089, AUC=0.7411, AUPRC=0.7480
[STABILITY] repeat=49/50, seed=20090, AUC=0.7768, AUPRC=0.7468
[STABILITY] repeat=50/50, seed=20091, AUC=0.8571, AUPRC=0.8707
[STABILITY] AUC median=0.7946, IQR=[0.7723, 0.8114], range=[0.7098, 0.8571]
[STABILITY] Running 50 repeated nested patient-level splits for C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9062, AUPRC=0.9084
[STABILITY] repeat=2/50, seed=20043, AUC=0.8438, AUPRC=0.8083
[STABILITY] repeat=3/50, seed=20044, AUC=0.9018, AUPRC=0.9070
[STABILITY] repeat=4/50, seed=20045, AUC=0.9018, AUPRC=0.9164
[STABILITY] repeat=5/50, seed=20046, AUC=0.9420, AUPRC=0.9656
[STABILITY] repeat=6/50, seed=20047, AUC=0.8571, AUPRC=0.8474
[STABILITY] repeat=7/50, seed=20048, AUC=0.8884, AUPRC=0.8764
[STABILITY] repeat=8/50, seed=20049, AUC=0.9152, AUPRC=0.9454
[STABILITY] repeat=9/50, seed=20050, AUC=0.8661, AUPRC=0.8854
[STABILITY] repeat=10/50, seed=20051, AUC=0.8661, AUPRC=0.8564
[STABILITY] repeat=11/50, seed=20052, AUC=0.9018, AUPRC=0.9294
[STABILITY] repeat=12/50, seed=20053, AUC=0.9062, AUPRC=0.9245
[STABILITY] repeat=13/50, seed=20054, AUC=0.8750, AUPRC=0.8717
[STABILITY] repeat=14/50, seed=20055, AUC=0.8750, AUPRC=0.9063
[STABILITY] repeat=15/50, seed=20056, AUC=0.8839, AUPRC=0.9084
[STABILITY] repeat=16/50, seed=20057, AUC=0.9152, AUPRC=0.9275
[STABILITY] repeat=17/50, seed=20058, AUC=0.8616, AUPRC=0.8551
[STABILITY] repeat=18/50, seed=20059, AUC=0.9152, AUPRC=0.9245
[STABILITY] repeat=19/50, seed=20060, AUC=0.8750, AUPRC=0.8920
[STABILITY] repeat=20/50, seed=20061, AUC=0.8884, AUPRC=0.8783
[STABILITY] repeat=21/50, seed=20062, AUC=0.8571, AUPRC=0.8712
[STABILITY] repeat=22/50, seed=20063, AUC=0.8973, AUPRC=0.9158
[STABILITY] repeat=23/50, seed=20064, AUC=0.8884, AUPRC=0.9007
[STABILITY] repeat=24/50, seed=20065, AUC=0.8616, AUPRC=0.8551
[STABILITY] repeat=25/50, seed=20066, AUC=0.8571, AUPRC=0.8290
[STABILITY] repeat=26/50, seed=20067, AUC=0.8705, AUPRC=0.8288
[STABILITY] repeat=27/50, seed=20068, AUC=0.8750, AUPRC=0.8607
[STABILITY] repeat=28/50, seed=20069, AUC=0.8973, AUPRC=0.9100
[STABILITY] repeat=29/50, seed=20070, AUC=0.9152, AUPRC=0.9400
[STABILITY] repeat=30/50, seed=20071, AUC=0.8705, AUPRC=0.8631
[STABILITY] repeat=31/50, seed=20072, AUC=0.9286, AUPRC=0.9473
[STABILITY] repeat=32/50, seed=20073, AUC=0.8839, AUPRC=0.8980
[STABILITY] repeat=33/50, seed=20074, AUC=0.8839, AUPRC=0.8995
[STABILITY] repeat=34/50, seed=20075, AUC=0.9375, AUPRC=0.9493
[STABILITY] repeat=35/50, seed=20076, AUC=0.9196, AUPRC=0.9338
[STABILITY] repeat=36/50, seed=20077, AUC=0.9062, AUPRC=0.9335
[STABILITY] repeat=37/50, seed=20078, AUC=0.8750, AUPRC=0.8717
[STABILITY] repeat=38/50, seed=20079, AUC=0.9286, AUPRC=0.9384
[STABILITY] repeat=39/50, seed=20080, AUC=0.8750, AUPRC=0.8804
[STABILITY] repeat=40/50, seed=20081, AUC=0.9018, AUPRC=0.9217
[STABILITY] repeat=41/50, seed=20082, AUC=0.8705, AUPRC=0.8554
[STABILITY] repeat=42/50, seed=20083, AUC=0.8973, AUPRC=0.9220
[STABILITY] repeat=43/50, seed=20084, AUC=0.9018, AUPRC=0.9191
[STABILITY] repeat=44/50, seed=20085, AUC=0.8750, AUPRC=0.8753
[STABILITY] repeat=45/50, seed=20086, AUC=0.8884, AUPRC=0.9121
[STABILITY] repeat=46/50, seed=20087, AUC=0.8705, AUPRC=0.8250
[STABILITY] repeat=47/50, seed=20088, AUC=0.8750, AUPRC=0.8977
[STABILITY] repeat=48/50, seed=20089, AUC=0.8839, AUPRC=0.8680
[STABILITY] repeat=49/50, seed=20090, AUC=0.9152, AUPRC=0.9400
[STABILITY] repeat=50/50, seed=20091, AUC=0.9107, AUPRC=0.9356
[STABILITY] AUC median=0.8884, IQR=[0.8750, 0.9062], range=[0.8438, 0.9420]
[STABILITY] Running 50 repeated nested patient-level splits for C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.8125, AUPRC=0.8342
[STABILITY] repeat=2/50, seed=20043, AUC=0.7054, AUPRC=0.6955
[STABILITY] repeat=3/50, seed=20044, AUC=0.7545, AUPRC=0.8026
[STABILITY] repeat=4/50, seed=20045, AUC=0.7277, AUPRC=0.7400
[STABILITY] repeat=5/50, seed=20046, AUC=0.6786, AUPRC=0.6896
[STABILITY] repeat=6/50, seed=20047, AUC=0.7455, AUPRC=0.7473
[STABILITY] repeat=7/50, seed=20048, AUC=0.7455, AUPRC=0.7736
[STABILITY] repeat=8/50, seed=20049, AUC=0.6920, AUPRC=0.7024
[STABILITY] repeat=9/50, seed=20050, AUC=0.7812, AUPRC=0.8186
[STABILITY] repeat=10/50, seed=20051, AUC=0.7589, AUPRC=0.7403
[STABILITY] repeat=11/50, seed=20052, AUC=0.6964, AUPRC=0.7189
[STABILITY] repeat=12/50, seed=20053, AUC=0.7679, AUPRC=0.7906
[STABILITY] repeat=13/50, seed=20054, AUC=0.6786, AUPRC=0.6941
[STABILITY] repeat=14/50, seed=20055, AUC=0.7455, AUPRC=0.7402
[STABILITY] repeat=15/50, seed=20056, AUC=0.7366, AUPRC=0.7685
[STABILITY] repeat=16/50, seed=20057, AUC=0.6696, AUPRC=0.7690
[STABILITY] repeat=17/50, seed=20058, AUC=0.7812, AUPRC=0.7880
[STABILITY] repeat=18/50, seed=20059, AUC=0.8259, AUPRC=0.8522
[STABILITY] repeat=19/50, seed=20060, AUC=0.7857, AUPRC=0.8070
[STABILITY] repeat=20/50, seed=20061, AUC=0.7321, AUPRC=0.7891
[STABILITY] repeat=21/50, seed=20062, AUC=0.7634, AUPRC=0.7839
[STABILITY] repeat=22/50, seed=20063, AUC=0.7500, AUPRC=0.7484
[STABILITY] repeat=23/50, seed=20064, AUC=0.7589, AUPRC=0.7716
[STABILITY] repeat=24/50, seed=20065, AUC=0.7857, AUPRC=0.8373
[STABILITY] repeat=25/50, seed=20066, AUC=0.7723, AUPRC=0.7909
[STABILITY] repeat=26/50, seed=20067, AUC=0.7455, AUPRC=0.7909
[STABILITY] repeat=27/50, seed=20068, AUC=0.7232, AUPRC=0.7598
[STABILITY] repeat=28/50, seed=20069, AUC=0.7723, AUPRC=0.7926
[STABILITY] repeat=29/50, seed=20070, AUC=0.7679, AUPRC=0.8148
[STABILITY] repeat=30/50, seed=20071, AUC=0.5804, AUPRC=0.6766
[STABILITY] repeat=31/50, seed=20072, AUC=0.7768, AUPRC=0.8253
[STABILITY] repeat=32/50, seed=20073, AUC=0.7232, AUPRC=0.7198
[STABILITY] repeat=33/50, seed=20074, AUC=0.7723, AUPRC=0.7974
[STABILITY] repeat=34/50, seed=20075, AUC=0.7277, AUPRC=0.7642
[STABILITY] repeat=35/50, seed=20076, AUC=0.6741, AUPRC=0.7330
[STABILITY] repeat=36/50, seed=20077, AUC=0.6830, AUPRC=0.6782
[STABILITY] repeat=37/50, seed=20078, AUC=0.7634, AUPRC=0.7673
[STABILITY] repeat=38/50, seed=20079, AUC=0.7857, AUPRC=0.8191
[STABILITY] repeat=39/50, seed=20080, AUC=0.7500, AUPRC=0.7936
[STABILITY] repeat=40/50, seed=20081, AUC=0.7098, AUPRC=0.7169
[STABILITY] repeat=41/50, seed=20082, AUC=0.8036, AUPRC=0.8406
[STABILITY] repeat=42/50, seed=20083, AUC=0.8259, AUPRC=0.8751
[STABILITY] repeat=43/50, seed=20084, AUC=0.7232, AUPRC=0.7238
[STABILITY] repeat=44/50, seed=20085, AUC=0.7098, AUPRC=0.7242
[STABILITY] repeat=45/50, seed=20086, AUC=0.7723, AUPRC=0.7784
[STABILITY] repeat=46/50, seed=20087, AUC=0.7723, AUPRC=0.7771
[STABILITY] repeat=47/50, seed=20088, AUC=0.6652, AUPRC=0.7044
[STABILITY] repeat=48/50, seed=20089, AUC=0.7054, AUPRC=0.6622
[STABILITY] repeat=49/50, seed=20090, AUC=0.7098, AUPRC=0.7667
[STABILITY] repeat=50/50, seed=20091, AUC=0.7455, AUPRC=0.7640
[STABILITY] AUC median=0.7455, IQR=[0.7098, 0.7723], range=[0.5804, 0.8259]
[STABILITY] Running 50 repeated nested patient-level splits for C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA.
[STABILITY] repeat=1/50, seed=20042, AUC=0.9107, AUPRC=0.9076
[STABILITY] repeat=2/50, seed=20043, AUC=0.8884, AUPRC=0.9067
[STABILITY] repeat=3/50, seed=20044, AUC=0.8884, AUPRC=0.9039
[STABILITY] repeat=4/50, seed=20045, AUC=0.8795, AUPRC=0.9189
[STABILITY] repeat=5/50, seed=20046, AUC=0.8884, AUPRC=0.9196
[STABILITY] repeat=6/50, seed=20047, AUC=0.9152, AUPRC=0.9244
[STABILITY] repeat=7/50, seed=20048, AUC=0.8616, AUPRC=0.8542
[STABILITY] repeat=8/50, seed=20049, AUC=0.9107, AUPRC=0.9202
[STABILITY] repeat=9/50, seed=20050, AUC=0.8884, AUPRC=0.8878
[STABILITY] repeat=10/50, seed=20051, AUC=0.9330, AUPRC=0.9299
[STABILITY] repeat=11/50, seed=20052, AUC=0.8973, AUPRC=0.8972
[STABILITY] repeat=12/50, seed=20053, AUC=0.8348, AUPRC=0.8848
[STABILITY] repeat=13/50, seed=20054, AUC=0.8616, AUPRC=0.8950
[STABILITY] repeat=14/50, seed=20055, AUC=0.9107, AUPRC=0.9245
[STABILITY] repeat=15/50, seed=20056, AUC=0.8795, AUPRC=0.8890
[STABILITY] repeat=16/50, seed=20057, AUC=0.9062, AUPRC=0.9230
[STABILITY] repeat=17/50, seed=20058, AUC=0.9286, AUPRC=0.9219
[STABILITY] repeat=18/50, seed=20059, AUC=0.9509, AUPRC=0.9473
[STABILITY] repeat=19/50, seed=20060, AUC=0.8661, AUPRC=0.8947
[STABILITY] repeat=20/50, seed=20061, AUC=0.8929, AUPRC=0.9104
[STABILITY] repeat=21/50, seed=20062, AUC=0.9196, AUPRC=0.9199
[STABILITY] repeat=22/50, seed=20063, AUC=0.8616, AUPRC=0.8762
[STABILITY] repeat=23/50, seed=20064, AUC=0.8929, AUPRC=0.8925
[STABILITY] repeat=24/50, seed=20065, AUC=0.9018, AUPRC=0.9169
[STABILITY] repeat=25/50, seed=20066, AUC=0.8973, AUPRC=0.9167
[STABILITY] repeat=26/50, seed=20067, AUC=0.8973, AUPRC=0.9024
[STABILITY] repeat=27/50, seed=20068, AUC=0.8884, AUPRC=0.8832
[STABILITY] repeat=28/50, seed=20069, AUC=0.9241, AUPRC=0.9242
[STABILITY] repeat=29/50, seed=20070, AUC=0.9107, AUPRC=0.9082
[STABILITY] repeat=30/50, seed=20071, AUC=0.8616, AUPRC=0.8662
[STABILITY] repeat=31/50, seed=20072, AUC=0.9018, AUPRC=0.9230
[STABILITY] repeat=32/50, seed=20073, AUC=0.8348, AUPRC=0.8774
[STABILITY] repeat=33/50, seed=20074, AUC=0.8705, AUPRC=0.8955
[STABILITY] repeat=34/50, seed=20075, AUC=0.9018, AUPRC=0.9233
[STABILITY] repeat=35/50, seed=20076, AUC=0.8795, AUPRC=0.9057
[STABILITY] repeat=36/50, seed=20077, AUC=0.8973, AUPRC=0.9037
[STABILITY] repeat=37/50, seed=20078, AUC=0.8973, AUPRC=0.8984
[STABILITY] repeat=38/50, seed=20079, AUC=0.8616, AUPRC=0.8789
[STABILITY] repeat=39/50, seed=20080, AUC=0.9018, AUPRC=0.8997
[STABILITY] repeat=40/50, seed=20081, AUC=0.8973, AUPRC=0.8907
[STABILITY] repeat=41/50, seed=20082, AUC=0.8839, AUPRC=0.8931
[STABILITY] repeat=42/50, seed=20083, AUC=0.9107, AUPRC=0.9285
[STABILITY] repeat=43/50, seed=20084, AUC=0.8839, AUPRC=0.8987
[STABILITY] repeat=44/50, seed=20085, AUC=0.8884, AUPRC=0.8924
[STABILITY] repeat=45/50, seed=20086, AUC=0.9062, AUPRC=0.9267
[STABILITY] repeat=46/50, seed=20087, AUC=0.9107, AUPRC=0.9186
[STABILITY] repeat=47/50, seed=20088, AUC=0.8750, AUPRC=0.8622
[STABILITY] repeat=48/50, seed=20089, AUC=0.8795, AUPRC=0.9064
[STABILITY] repeat=49/50, seed=20090, AUC=0.9330, AUPRC=0.9404
[STABILITY] repeat=50/50, seed=20091, AUC=0.9152, AUPRC=0.9253
[STABILITY] AUC median=0.8973, IQR=[0.8795, 0.9107], range=[0.8348, 0.9509]

[STABILITY] PAIRED REPEATED-CV COMPARISONS
[STABILITY][PAIRED] A12_VS_FIXED_CENTER_60: median_delta=+0.0982, IQR=[+0.0770, +0.1150], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_STANDARDIZED_FULL_IMAGE: median_delta=+0.1205, IQR=[+0.0960, +0.1473], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_OUTSIDE_LARGE_BBOX: median_delta=+0.1384, IQR=[+0.1071, +0.1607], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_ZERO_BG_FULL_FALLBACK: median_delta=+0.0000, IQR=[-0.0000, +0.0123], comparison_better=50.0%
[STABILITY][PAIRED] A12_VS_MONAI_BBOX_CROP: median_delta=+0.0402, IQR=[+0.0268, +0.0625], comparison_better=100.0%
[STABILITY][PAIRED] A12_HIERARCHICAL_VS_FIXED_CHUNK: median_delta=+0.0335, IQR=[+0.0179, +0.0569], comparison_better=90.0%
[STABILITY][PAIRED] A12_VS_EXACT_DEDUP: median_delta=-0.0089, IQR=[-0.0179, +0.0000], comparison_better=22.0%
[STABILITY][PAIRED] A12_VS_SOFT_MASK_ONLY: median_delta=+0.1027, IQR=[+0.0681, +0.1283], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_HARD_MASK_ONLY: median_delta=+0.1027, IQR=[+0.0815, +0.1551], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_BBOX_MASK_ONLY: median_delta=+0.2634, IQR=[+0.2377, +0.3036], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_SOFT_HISTOGRAM_ONLY: median_delta=+0.1585, IQR=[+0.1395, +0.1830], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_BLOCK_SHUFFLED_SOFT_MAP: median_delta=+0.0603, IQR=[+0.0402, +0.0804], comparison_better=96.0%
[STABILITY][PAIRED] A12_VS_CANONICAL_HARD_MASK: median_delta=+0.2009, IQR=[+0.1741, +0.2366], comparison_better=100.0%
[STABILITY][PAIRED] A12_VS_CANONICAL_SOFT_MASK: median_delta=+0.0603, IQR=[+0.0357, +0.0748], comparison_better=96.0%
[STABILITY][PAIRED] SOFT_SPATIAL_MAP_VS_HISTOGRAM_ONLY: median_delta=+0.0580, IQR=[+0.0357, +0.0926], comparison_better=90.0%
[STABILITY][PAIRED] SOFT_SPATIAL_MAP_VS_BLOCK_SHUFFLED: median_delta=-0.0402, IQR=[-0.0759, +0.0033], comparison_better=26.0%
[STABILITY][PAIRED] POSITIONED_HARD_MASK_VS_CANONICAL_HARD: median_delta=+0.0893, IQR=[+0.0547, +0.1384], comparison_better=96.0%
[STABILITY][PAIRED] POSITIONED_SOFT_MASK_VS_CANONICAL_SOFT: median_delta=-0.0379, IQR=[-0.0670, -0.0134], comparison_better=12.0%
[STABILITY][PAIRED] CANONICAL_SOFT_MASK_VS_HISTOGRAM_ONLY: median_delta=+0.1027, IQR=[+0.0714, +0.1283], comparison_better=100.0%
[PIPELINE 10/14] COMPLETED: Run repeated nested-CV split-stability analyses in 8 min 6.6 s
[PIPELINE 10/14] Completed=19/19 configured experiments.

==============================================================================
[PIPELINE 11/14] START: Run patient-label permutation sanity test
[PIPELINE 11/14] Expected workload: Many lightweight patient-level fits; no neural-network inference.
==============================================================================
[PERMUTATION] Running 1000 patient-label permutations for A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA.
[PERMUTATION] 1/1000 complete; latest AUC=0.4643
[PERMUTATION] 100/1000 complete; latest AUC=0.3839
[PERMUTATION] 200/1000 complete; latest AUC=0.2991
[PERMUTATION] 300/1000 complete; latest AUC=0.4911
[PERMUTATION] 400/1000 complete; latest AUC=0.7946
[PERMUTATION] 500/1000 complete; latest AUC=0.6205
[PERMUTATION] 600/1000 complete; latest AUC=0.4107
[PERMUTATION] 700/1000 complete; latest AUC=0.4420
[PERMUTATION] 800/1000 complete; latest AUC=0.3571
[PERMUTATION] 900/1000 complete; latest AUC=0.3661
[PERMUTATION] 1000/1000 complete; latest AUC=0.4688
[PERMUTATION] observed AUC=0.9643, null median=0.5045, empirical p=0.000999
[PIPELINE 11/14] COMPLETED: Run patient-label permutation sanity test in 8 min 21.1 s
[PIPELINE 11/14] OK

==============================================================================
[PIPELINE 12/14] START: Run optional annotated sequence/view subset analysis
[PIPELINE 12/14] Expected workload: Immediate when disabled; otherwise several patient-level nested-CV fits.
==============================================================================
[SERIES SUBSET] SKIPPED: no completed annotation CSV was enabled.
[PIPELINE 12/14] COMPLETED: Run optional annotated sequence/view subset analysis in 1 ms
[PIPELINE 12/14] SKIPPED_DISABLED

==============================================================================
[PIPELINE 13/14] START: Record external-validation status
[PIPELINE 13/14] Expected workload: Immediate unless a verified independent-data adapter is later added.
==============================================================================
[EXTERNAL VALIDATION] SKIPPED: no independent dataset configured.
[PIPELINE 13/14] COMPLETED: Record external-validation status in 1 ms
[PIPELINE 13/14] SKIPPED_NOT_CONFIGURED

==============================================================================
[PIPELINE 14/14] START: Print final comparison and save suite metadata
[PIPELINE 14/14] Expected workload: Console table, warnings, metadata JSON and timing summary.
==============================================================================

====================================================================================================================================================================================================
FINAL MULTI-EXPERIMENT PATIENT-LEVEL COMPARISON
====================================================================================================================================================================================================

CURRENT STANDARDIZED MODELS / LOCALIZATION CANDIDATES
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
No.  Experiment                                                               Role                                 AUC [95% CI]              Delta AUC vs baseline             Sens.   Spec.      F1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA          ablation                             0.9732 [0.9107, 1.0000]   +0.0089 [-0.0536, +0.0804]        0.786   0.938   0.846
2    A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA                 baseline                             0.9643 [0.8929, 1.0000]   +0.0000 [+0.0000, +0.0000]        0.857   0.938   0.889
3    A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA           ablation                             0.9464 [0.8527, 1.0000]   -0.0179 [-0.0804, +0.0223]        0.857   0.938   0.889
4    A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA                                    ablation                             0.9375 [0.8170, 1.0000]   -0.0268 [-0.1027, +0.0179]        0.786   0.938   0.846
5    A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA                                 ablation                             0.9286 [0.8214, 1.0000]   -0.0357 [-0.1116, +0.0134]        0.857   0.875   0.857
6    C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA                                  localization_control                 0.9241 [0.7857, 1.0000]   -0.0402 [-0.1339, +0.0134]        0.929   0.938   0.929
7    C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA                              localization_control                 0.8929 [0.7634, 0.9821]   -0.0714 [-0.1607, +0.0000]        0.643   0.812   0.692
8    A9_STANDARDIZED_FULL_HIER_LR_PCA                                         ablation                             0.8795 [0.7410, 0.9821]   -0.0848 [-0.1964, -0.0089]        0.714   0.938   0.800
9    C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA                              localization_control                 0.8795 [0.7232, 0.9955]   -0.0848 [-0.2144, +0.0134]        0.786   0.938   0.846
10   C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA                              localization_control                 0.8438 [0.6786, 0.9643]   -0.1205 [-0.2456, -0.0134]        0.786   0.688   0.733
11   A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA                                  ablation                             0.8393 [0.6786, 0.9643]   -0.1250 [-0.2768, -0.0134]        0.643   0.812   0.692
12   B1_STANDARDIZED_ROI_HIER_LR_PCA                                          development_baseline                 0.8259 [0.6561, 0.9510]   -0.1384 [-0.2857, -0.0357]        0.643   0.812   0.692
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

NEGATIVE, EXPORT AND STRUCTURAL CONTROLS
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
No.  Experiment                                                               Role                                 AUC [95% CI]              Delta AUC vs baseline             Sens.   Spec.      F1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA                                        negative_control                     0.8839 [0.7411, 0.9911]   -0.0804 [-0.1875, +0.0000]        0.714   0.938   0.800
2    C1_BORDER_ONLY_HIER_LR_PCA                                               negative_control                     0.8750 [0.7188, 0.9732]   -0.0893 [-0.2143, -0.0089]        0.643   0.750   0.667
3    C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA                          negative_control                     0.8259 [0.6384, 0.9732]   -0.1384 [-0.3170, +0.0000]        0.786   0.875   0.815
4    C19_NATIVE_GEOMETRY_ONLY_LR                                              negative_control                     0.7902 [0.5893, 0.9465]   -0.1741 [-0.3839, +0.0000]        0.643   0.938   0.750
5    C3_EXPORT_PROVENANCE_ONLY_LR                                             negative_control                     0.7902 [0.5938, 0.9554]   -0.1741 [-0.3661, -0.0089]        0.643   0.875   0.720
6    C20_FILE_SIZE_ONLY_LR                                                    negative_control                     0.7679 [0.5759, 0.9331]   -0.1964 [-0.3929, -0.0268]        0.714   0.688   0.690
7    C11_STANDARDIZATION_QC_ONLY_LR                                           negative_control                     0.7277 [0.5267, 0.8885]   -0.2366 [-0.4196, -0.0848]        0.500   0.688   0.538
8    C7_DETECTED_PADDING_MASK_HIER_LR_PCA                                     negative_control                     0.7098 [0.5000, 0.8884]   -0.2545 [-0.4509, -0.0848]        0.429   0.875   0.545
9    C17_N_SERIES_ONLY_LR                                                     negative_control                     0.6987 [0.5022, 0.8728]   -0.2656 [-0.4621, -0.0848]        0.643   0.688   0.643
10   C6_STANDARDIZED_BORDER10_HIER_LR_PCA                                     negative_control                     0.6920 [0.4732, 0.8884]   -0.2723 [-0.4643, -0.1026]        0.500   0.938   0.636
11   C4_MONAI_QC_ONLY_LR                                                      negative_control                     0.6741 [0.4686, 0.8527]   -0.2902 [-0.4821, -0.1250]        0.429   0.562   0.444
12   C12_STANDARDIZED_MONAI_QC_ONLY_LR                                        negative_control                     0.6250 [0.4018, 0.8080]   -0.3393 [-0.5446, -0.1652]        0.500   0.750   0.560
13   C16_N_SLICES_ONLY_LR                                                     negative_control                     0.6027 [0.3883, 0.8125]   -0.3616 [-0.5849, -0.1384]        0.357   0.688   0.417
14   C5_STANDARDIZED_BORDER05_HIER_LR_PCA                                     negative_control                     0.5938 [0.3839, 0.7991]   -0.3705 [-0.5759, -0.1740]        0.714   0.562   0.645
15   C8_STANDARDIZED_CORNERS_HIER_LR_PCA                                      negative_control                     0.5714 [0.3661, 0.7769]   -0.3929 [-0.5893, -0.1962]        0.500   0.562   0.500
16   C18_SERIES_LENGTH_ONLY_LR                                                negative_control                     0.5625 [0.3438, 0.7634]   -0.4018 [-0.6116, -0.2008]        0.571   0.562   0.552
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MONAI-DERIVED MORPHOLOGY / CONFIDENCE CONTROLS
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
No.  Experiment                                                               Role                                 AUC [95% CI]              Delta AUC vs baseline             Sens.   Spec.      F1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA                        segmentation_representation_control  0.8973 [0.7589, 0.9911]   -0.0670 [-0.2054, +0.0357]        0.643   0.938   0.750
2    C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA              segmentation_representation_control  0.8929 [0.7633, 0.9821]   -0.0714 [-0.1830, +0.0000]        0.786   0.812   0.786
3    C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA                   segmentation_representation_control  0.8795 [0.7232, 1.0000]   -0.0848 [-0.2456, +0.0446]        0.857   0.875   0.857
4    C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA                        segmentation_representation_control  0.7857 [0.5893, 0.9554]   -0.1786 [-0.3705, -0.0179]        0.786   0.938   0.846
5    C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA                   segmentation_representation_control  0.7679 [0.5714, 0.9286]   -0.1964 [-0.3974, -0.0223]        0.643   0.812   0.692
6    C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA              segmentation_representation_control  0.7545 [0.5625, 0.9198]   -0.2098 [-0.3973, -0.0491]        0.571   0.688   0.593
7    C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA                        segmentation_representation_control  0.7098 [0.5000, 0.8884]   -0.2545 [-0.4732, -0.0714]        0.643   0.812   0.692
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

HISTORICAL ORIGINAL-CANVAS MODELS AND METHOD ABLATIONS
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
No.  Experiment                                                               Role                                 AUC [95% CI]              Delta AUC vs baseline             Sens.   Spec.      F1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    A5_ROI_HIER_LINEAR_SVM_PCA                                               ablation                             0.9420 [0.8302, 1.0000]   -0.0223 [-0.0804, +0.0135]        0.786   0.938   0.846
2    A6_ROI_HIER_LR_QUALITY_PCA                                               ablation                             0.9375 [0.8393, 1.0000]   -0.0268 [-0.0938, +0.0179]        0.786   0.938   0.846
3    A8_ROI_HIER_LR_NO_PCA_FIXEDC                                             matched_reference                    0.9375 [0.8348, 1.0000]   -0.0268 [-0.0848, +0.0089]        0.857   0.938   0.889
4    B0_ROI_HIER_LR_PCA                                                       historical_baseline                  0.9241 [0.8036, 1.0000]   -0.0402 [-0.1116, +0.0000]        0.786   0.938   0.846
5    A7_ROI_HIER_LR_NO_PCA                                                    ablation                             0.9107 [0.7722, 1.0000]   -0.0536 [-0.1429, +0.0000]        0.857   0.938   0.889
6    A1_FULL_HIER_LR_PCA                                                      ablation                             0.8973 [0.7589, 0.9866]   -0.0670 [-0.1741, +0.0134]        0.714   0.938   0.800
7    A3_ROI_LEGACY_LOGODDS_LR                                                 ablation                             0.8839 [0.7232, 0.9911]   -0.0804 [-0.1875, -0.0045]        0.714   0.938   0.800
8    A4_ROI_LEGACY_MEANPROB_LR                                                ablation                             0.8795 [0.7232, 0.9911]   -0.0848 [-0.1920, +0.0000]        0.714   0.938   0.800
9    A2_ROI_FLAT_LR_PCA                                                       ablation                             0.8125 [0.6250, 0.9420]   -0.1518 [-0.3214, -0.0402]        0.643   0.875   0.720
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ROBUSTNESS EXPERIMENTS
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
No.  Experiment                                                               Role                                 AUC [95% CI]              Delta AUC vs baseline             Sens.   Spec.      F1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    R2_ROI_HIER_LR_PCA_DROP25                                                robustness                           0.9241 [0.8125, 1.0000]   -0.0402 [-0.1071, +0.0045]        0.786   0.938   0.846
2    R1_ROI_HIER_LR_PCA_DROP10                                                robustness                           0.9196 [0.7946, 1.0000]   -0.0446 [-0.1205, +0.0089]        0.786   0.938   0.846
3    R3_ROI_HIER_LR_PCA_DROP50                                                robustness                           0.9152 [0.7812, 1.0000]   -0.0491 [-0.1339, +0.0000]        0.786   0.938   0.846
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
All rows use the same outer patient-fold manifest. Thresholds and quantitatively selected C values were learned only from inner OOF training predictions. Tables are separated so controls are not misrepresented as candidate clinical models.

SHORTCUT / CONFOUNDING WARNINGS
[WARNING] C2_OUTSIDE_MONAI_MASK_HIER_LR_PCA achieved AUC=0.8839 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C1_BORDER_ONLY_HIER_LR_PCA achieved AUC=0.8750 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA achieved AUC=0.8259 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C19_NATIVE_GEOMETRY_ONLY_LR achieved AUC=0.7902 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C3_EXPORT_PROVENANCE_ONLY_LR achieved AUC=0.7902 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C20_FILE_SIZE_ONLY_LR achieved AUC=0.7679 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C11_STANDARDIZATION_QC_ONLY_LR achieved AUC=0.7277 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C7_DETECTED_PADDING_MASK_HIER_LR_PCA achieved AUC=0.7098 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C17_N_SERIES_ONLY_LR achieved AUC=0.6987 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C6_STANDARDIZED_BORDER10_HIER_LR_PCA achieved AUC=0.6920 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.
[WARNING] C4_MONAI_QC_ONLY_LR achieved AUC=0.6741 >= 0.65. The cohort remains predictable from a negative-control representation; anatomical validity is therefore not established.

MONAI-DERIVED REPRESENTATION FINDINGS
[SEGMENTATION CONTROL] C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA achieved AUC=0.8973. This means label information remains in MONAI-derived probability, morphology, position, scale, or confidence structure. It is not, by itself, proof of a non-anatomical shortcut because the representation is derived from the MRI image.
[SEGMENTATION CONTROL] C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA achieved AUC=0.8929. This means label information remains in MONAI-derived probability, morphology, position, scale, or confidence structure. It is not, by itself, proof of a non-anatomical shortcut because the representation is derived from the MRI image.
[SEGMENTATION CONTROL] C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA achieved AUC=0.8795. This means label information remains in MONAI-derived probability, morphology, position, scale, or confidence structure. It is not, by itself, proof of a non-anatomical shortcut because the representation is derived from the MRI image.
[SEGMENTATION CONTROL] C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA achieved AUC=0.7857. This means label information remains in MONAI-derived probability, morphology, position, scale, or confidence structure. It is not, by itself, proof of a non-anatomical shortcut because the representation is derived from the MRI image.
[SEGMENTATION CONTROL] C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA achieved AUC=0.7679. This means label information remains in MONAI-derived probability, morphology, position, scale, or confidence structure. It is not, by itself, proof of a non-anatomical shortcut because the representation is derived from the MRI image.
[SEGMENTATION CONTROL] C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA achieved AUC=0.7545. This means label information remains in MONAI-derived probability, morphology, position, scale, or confidence structure. It is not, by itself, proof of a non-anatomical shortcut because the representation is derived from the MRI image.
[SEGMENTATION CONTROL] C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA achieved AUC=0.7098. This means label information remains in MONAI-derived probability, morphology, position, scale, or confidence structure. It is not, by itself, proof of a non-anatomical shortcut because the representation is derived from the MRI image.
====================================================================================================================================================================================================

====================================================================================================================================================================================
PREDECLARED MATCHED ABLATION COMPARISONS
====================================================================================================================================================================================
Comparison                                        Reference -> Changed                                                         Delta AUC [95% CI]            Status                
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ORIGINAL_ROI_VS_LABEL_BLIND_STANDARDIZED_ROI      B0_ROI_HIER_LR_PCA -> B1_STANDARDIZED_ROI_HIER_LR_PCA                        -0.0982 [-0.2366, +0.0089]    OK                    
STANDARDIZED_ROI_VS_STANDARDIZED_FULL_IMAGE       B1_STANDARDIZED_ROI_HIER_LR_PCA -> A9_STANDARDIZED_FULL_HIER_LR_PCA          +0.0536 [-0.0938, +0.2098]    OK                    
STANDARDIZED_ROI_VS_AREA_MATCHED_CENTER_CROP      B1_STANDARDIZED_ROI_HIER_LR_PCA -> C9_STANDARDIZED_CENTER_CROP_HIER_LR_PCA   +0.0982 [-0.0357, +0.2501]    OK                    
STANDARDIZED_ROI_VS_FIXED_CENTER_50               B1_STANDARDIZED_ROI_HIER_LR_PCA -> C13_STANDARDIZED_FIXED_CENTER50_HIER_LR_PCA +0.0670 [-0.0402, +0.2098]    OK                    
STANDARDIZED_ROI_VS_FIXED_CENTER_60               B1_STANDARDIZED_ROI_HIER_LR_PCA -> C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA +0.0536 [-0.0982, +0.2187]    OK                    
STANDARDIZED_ROI_VS_FIXED_CENTER_70               B1_STANDARDIZED_ROI_HIER_LR_PCA -> C15_STANDARDIZED_FIXED_CENTER70_HIER_LR_PCA +0.0179 [-0.1071, +0.1607]    OK                    
STANDARDIZED_SOFT_ROI_VS_ZERO_BACKGROUND_CENTER_FALLBACK B1_STANDARDIZED_ROI_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.1384 [+0.0312, +0.2857]    OK                    
PRIMARY_CANDIDATE_VS_FIXED_CENTER_60              C14_STANDARDIZED_FIXED_CENTER60_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.0848 [-0.0134, +0.2188]    OK                    
PRIMARY_CANDIDATE_VS_STANDARDIZED_FULL_IMAGE      A9_STANDARDIZED_FULL_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.0848 [+0.0089, +0.2009]    OK                    
PRIMARY_CANDIDATE_VS_OUTSIDE_LARGE_BOUNDING_BOX   C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.1384 [+0.0045, +0.3170]    OK                    
PRIMARY_CANDIDATE_VS_ZERO_BACKGROUND_FULL_FALLBACK A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.0357 [-0.0134, +0.1116]    OK                    
PRIMARY_CANDIDATE_VS_MONAI_BOUNDING_BOX_CROP      A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.0268 [-0.0179, +0.1028]    OK                    
PRIMARY_CANDIDATE_HIERARCHICAL_VS_FIXED_CHUNK_POOLING A14_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_FIXED_CHUNK_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA -0.0089 [-0.0714, +0.0536]    OK                    
PRIMARY_CANDIDATE_VS_EXACT_WITHIN_PATIENT_DEDUPLICATION A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA -> A15_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_DEDUP_HIER_LR_PCA -0.0179 [-0.0759, +0.0223]    OK                    
PRIMARY_CANDIDATE_VS_SOFT_MONAI_MASK_ONLY         C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.0670 [-0.0402, +0.1920]    OK                    
PRIMARY_CANDIDATE_VS_HARD_MONAI_MASK_ONLY         C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.1786 [+0.0179, +0.3706]    OK                    
PRIMARY_CANDIDATE_VS_MONAI_BBOX_MASK_ONLY         C23_STANDARDIZED_MONAI_BBOX_MASK_ONLY_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.2545 [+0.0670, +0.4643]    OK                    
PRIMARY_CANDIDATE_VS_SOFT_MONAI_HISTOGRAM_ONLY    C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.1964 [+0.0312, +0.3884]    OK                    
PRIMARY_CANDIDATE_VS_BLOCK_SHUFFLED_SOFT_MONAI_MAP C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.0848 [-0.0446, +0.2411]    OK                    
PRIMARY_CANDIDATE_VS_CANONICAL_HARD_MONAI_MASK    C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.2098 [+0.0625, +0.4062]    OK                    
PRIMARY_CANDIDATE_VS_CANONICAL_SOFT_MONAI_MASK    C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA -> A12_STANDARDIZED_ROI_ZERO_BG_CENTER_FALLBACK_HIER_LR_PCA +0.0714 [+0.0000, +0.1786]    OK                    
SOFT_MONAI_SPATIAL_MAP_VS_HISTOGRAM_ONLY          C24_STANDARDIZED_SOFT_MONAI_HISTOGRAM_ONLY_HIER_LR_PCA -> C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA +0.1295 [+0.0357, +0.2589]    OK                    
SOFT_MONAI_SPATIAL_MAP_VS_BLOCK_SHUFFLED          C25_STANDARDIZED_SOFT_MONAI_BLOCK_SHUFFLED_HIER_LR_PCA -> C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA +0.0179 [-0.0938, +0.1161]    OK                    
POSITIONED_HARD_MASK_VS_CANONICAL_HARD_SHAPE      C26_STANDARDIZED_CANONICAL_HARD_MONAI_MASK_ONLY_HIER_LR_PCA -> C22_STANDARDIZED_HARD_MONAI_MASK_ONLY_HIER_LR_PCA +0.0313 [-0.1116, +0.1741]    OK                    
POSITIONED_SOFT_MASK_VS_CANONICAL_SOFT_SHAPE      C27_STANDARDIZED_CANONICAL_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA -> C21_STANDARDIZED_SOFT_MONAI_MASK_ONLY_HIER_LR_PCA +0.0045 [-0.0804, +0.0937]    OK                    
STANDARDIZED_HIERARCHICAL_VS_FIXED_CHUNK_POOLING  B1_STANDARDIZED_ROI_HIER_LR_PCA -> A13_STANDARDIZED_ROI_FIXED_CHUNK_LR_PCA   +0.0134 [-0.1473, +0.1741]    OK                    
STANDARDIZED_SOFT_ROI_VS_ZERO_BACKGROUND          B1_STANDARDIZED_ROI_HIER_LR_PCA -> A10_STANDARDIZED_ROI_ZERO_BG_HIER_LR_PCA  +0.1027 [-0.0089, +0.2366]    OK                    
STANDARDIZED_SOFT_ROI_VS_BOUNDING_BOX_CROP        B1_STANDARDIZED_ROI_HIER_LR_PCA -> A11_STANDARDIZED_ROI_BBOX_HIER_LR_PCA     +0.1116 [-0.0179, +0.2634]    OK                    
STANDARDIZED_ROI_VS_OUTSIDE_LARGE_BOUNDING_BOX    B1_STANDARDIZED_ROI_HIER_LR_PCA -> C10_STANDARDIZED_OUTSIDE_LARGE_BBOX_HIER_LR_PCA +0.0000 [-0.1384, +0.1429]    OK                    
STANDARDIZED_ROI_VS_STANDARDIZED_BORDER_05        B1_STANDARDIZED_ROI_HIER_LR_PCA -> C5_STANDARDIZED_BORDER05_HIER_LR_PCA      -0.2321 [-0.4420, -0.0179]    OK                    
STANDARDIZED_ROI_VS_STANDARDIZED_BORDER_10        B1_STANDARDIZED_ROI_HIER_LR_PCA -> C6_STANDARDIZED_BORDER10_HIER_LR_PCA      -0.1339 [-0.3214, +0.0313]    OK                    
STANDARDIZED_ROI_VS_DETECTED_PADDING_MASK         B1_STANDARDIZED_ROI_HIER_LR_PCA -> C7_DETECTED_PADDING_MASK_HIER_LR_PCA      -0.1161 [-0.3260, +0.1027]    OK                    
STANDARDIZED_ROI_VS_STANDARDIZED_CORNERS          B1_STANDARDIZED_ROI_HIER_LR_PCA -> C8_STANDARDIZED_CORNERS_HIER_LR_PCA       -0.2545 [-0.4554, -0.0758]    OK                    
ROI_VS_FULL_IMAGE                                 B0_ROI_HIER_LR_PCA -> A1_FULL_HIER_LR_PCA                                    -0.0268 [-0.1205, +0.0670]    OK                    
HIERARCHICAL_VS_FLAT_POOLING                      B0_ROI_HIER_LR_PCA -> A2_ROI_FLAT_LR_PCA                                     -0.1116 [-0.2768, +0.0312]    OK                    
PATIENT_EMBEDDING_VS_LEGACY_SLICE_CLASSIFIER      A8_ROI_HIER_LR_NO_PCA_FIXEDC -> A3_ROI_LEGACY_LOGODDS_LR                     -0.0536 [-0.1339, +0.0089]    OK                    
LOGODDS_VS_MEAN_PROBABILITY_FUSION                A3_ROI_LEGACY_LOGODDS_LR -> A4_ROI_LEGACY_MEANPROB_LR                        -0.0045 [-0.0357, +0.0223]    OK                    
LOGISTIC_REGRESSION_VS_LINEAR_SVM                 B0_ROI_HIER_LR_PCA -> A5_ROI_HIER_LINEAR_SVM_PCA                             +0.0179 [-0.0089, +0.0625]    OK                    
EQUAL_VS_QUALITY_SLICE_WEIGHTS                    B0_ROI_HIER_LR_PCA -> A6_ROI_HIER_LR_QUALITY_PCA                             +0.0134 [-0.0357, +0.0848]    OK                    
PCA_ON_VS_PCA_OFF                                 B0_ROI_HIER_LR_PCA -> A7_ROI_HIER_LR_NO_PCA                                  -0.0134 [-0.0580, +0.0134]    OK                    
ROBUSTNESS_AFTER_10_PERCENT_SLICE_DROPOUT         B0_ROI_HIER_LR_PCA -> R1_ROI_HIER_LR_PCA_DROP10                              -0.0045 [-0.0357, +0.0179]    OK                    
ROBUSTNESS_AFTER_25_PERCENT_SLICE_DROPOUT         B0_ROI_HIER_LR_PCA -> R2_ROI_HIER_LR_PCA_DROP25                              +0.0000 [-0.0000, +0.0000]    OK                    
ROBUSTNESS_AFTER_50_PERCENT_SLICE_DROPOUT         B0_ROI_HIER_LR_PCA -> R3_ROI_HIER_LR_PCA_DROP50                              -0.0089 [-0.0402, +0.0000]    OK                    
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Delta AUC is changed configuration minus its declared reference. A confidence interval containing zero does not establish a reliable difference.
====================================================================================================================================================================================
[PIPELINE 14/14] COMPLETED: Print final comparison and save suite metadata in 151 ms
[PIPELINE 14/14] Master summary: /kaggle/working/cad_patient_pipeline_outputs/multi_experiment_suite__3997634de7/comparison/experiment_summary.csv

------------------------------------------------------------------------------
EXECUTION TIMING SUMMARY
------------------------------------------------------------------------------
  01 Configuration validation                                          0 ms
  02 Output structure                                                  3 ms
  03 Dataset discovery                                          1 min 4.2 s
  04 Shared feature bank                                      49 min 22.4 s
  05 Duplicate audits                                                5.94 s
  06 Manifests                                                       4.41 s
  07 QC/provenance controls                                          426 ms
  08 Experiment execution                                     10 min 25.7 s
  09 Master comparisons                                        6 min 24.3 s
  10 Repeated nested CV                                         8 min 6.6 s
  11 Label permutation                                         8 min 21.1 s
  12 Annotated sequence/view subset                                    1 ms
  13 External validation status                                        1 ms
  14 Final reporting                                                 151 ms
------------------------------------------------------------------------------
  TOTAL PIPELINE RUNTIME                                  1 h 23 min 55.5 s
------------------------------------------------------------------------------

[SUITE] Outputs saved under: /kaggle/working/cad_patient_pipeline_outputs/multi_experiment_suite__3997634de7
[SUITE] Console log: /kaggle/working/cad_patient_pipeline_outputs/multi_experiment_suite__3997634de7/console_output.log
[SUITE] Completed with 47 successful and 0 failed experiments in 1 h 23 min 55.5 s.
Done!
```