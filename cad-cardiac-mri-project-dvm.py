#!/usr/bin/env python3
r"""CAD Cardiac MRI — University Admissions Research Portfolio Pipeline
========================================================================

DOCUMENT PURPOSE AND AUDIENCE
-----------------------------
This single Python file is intentionally self-contained.  It contains the
research motivation, dataset contract, mathematical derivations, implementation,
Kaggle instructions, output definitions, evaluation rules, limitations, and a
suggested university-application narrative.  A reviewer should be able to
understand the project without opening separate README or methodology files.

The intended audience is broad and institution-neutral:

* university admissions readers evaluating intellectual initiative;
* research mentors reviewing experimental design;
* technical readers checking leakage control and mathematical correctness;
* future collaborators who need a reproducible execution entry point.

The recommended generic filename is:

    cad_mri_university_showcase.py

A compatibility copy may use an older filename, but the scientific content and
terminology in this module are deliberately university-generic.

ONE-SENTENCE PROJECT DESCRIPTION
--------------------------------
I built a patient-level cardiac MRI classifier, discovered that a high internal
score was partly reproducible from non-cardiac and segmentation-derived
shortcuts, and redesigned the project around exact matched controls and custom
mathematical methods for reducing dependence on nuisance information.

THE CENTRAL RESEARCH QUESTION
-----------------------------
The project asks a harder question than "How high is the AUROC?":

    Can cardiac-region MRI information classify the released Normal/Sick labels
    more strongly than exact matched shortcut controls, and can compact custom
    mathematics preserve discrimination while reducing dependence on those
    controls?

This is important because a medical-imaging model can be accurate for the wrong
reason.  It may exploit scanner style, JPEG export geometry, sequence mixture,
mask shape, padding, or anatomy outside the intended region instead of learning
the disease-related information that motivated the study.

DATASET AND PATIENT CONTRACT
----------------------------
Dataset:

    CAD Cardiac MRI Dataset
    https://www.kaggle.com/datasets/danialsharifrazi/cad-cardiac-mri-dataset

Validated computational patient unit for this release:

    patient_id = Directory_*

Class mapping:

    Normal/Directory_* -> label 0
    Sick/Directory_*   -> label 1

Every descendant image of one Directory_* folder belongs to the same patient.
The immediate image parent is treated only as a folder-defined series proxy.
It is not described as a verified DICOM SeriesInstanceUID because the public
release contains JPEG exports rather than original DICOM metadata.

The release contains approximately:

    30 patient units
    16 Normal patients
    14 Sick patients
    63,425 image files
    2,859 folder-defined series proxies

The effective labeled sample size is 30 patients, not 63,425 images.  All images
and series proxies from one patient remain in the same train/validation fold.
No slice is treated as an independent labeled patient.

HISTORICAL RESULT THAT MOTIVATED THIS SIMPLIFICATION
----------------------------------------------------
The larger audit suite evaluated 58 experiments and 32 image representations.
In the supplied V6 Kaggle audit, the repeated patient-level median AUROCs that
most directly motivate this simplified file were approximately:

    A17 cardiac hard-support intensity            0.9732
    C31 exact A17 support only                     0.8661
    C32 same support/histogram, shuffled spatially 0.7991
    C33 exact complement of the A17 support        0.8170

These values are historical motivation only.  They are not hard-coded into this
pipeline and do not count as results of this simplified implementation.  This
file must recompute its own metrics.

The intellectual turning point was that the candidate model was highly
predictive, but carefully chosen controls were also predictive.  The project
therefore changed from "optimize one model" to "test what information the model
uses and whether that dependence can be reduced."

WHY THIS VERSION IS SMALLER
---------------------------
The full audit is valuable as an archival stress-test, but it is not the clearest
first view of the project.  This file reduces the scientific core to seven
experiments with one interpretable purpose each:

1. BASELINE_A17
   Region-normalized MRI intensities inside one exact binary cardiac support.

2. CONTROL_SUPPORT_ONLY
   The exact same final support mask, with MRI intensities removed.

3. CONTROL_SHUFFLED_INTENSITY
   The same support and exact visible intensity multiset as the baseline, but
   with spatial locations deterministically permuted.

4. CONTROL_OUTSIDE_SUPPORT
   The exact retained-content complement of the baseline support, independently
   normalized.

5. CUSTOM_NUISANCE_PROJECTION
   A fold-local ridge projection that removes cardiac-feature components that
   can be reconstructed from nuisance variables.

6. CUSTOM_MINIMAX_CONFOUNDER_GUARD
   A logistic classifier optimized from scratch with a penalty on the strongest
   linear covariance between its score and any nuisance direction.

7. CUSTOM_BAYESIAN_SERIES_FUSION
   A patient-trained classifier whose validation-series evidence is combined in
   prior-relative log-odds space with saturating reliability weights.

The following historical branches are removed from the default execution path:

* the 58-entry experiment registry;
* many border widths, corner crops, center crops, fixed chunks, dropout levels,
  SVM variants, no-PCA variants, and legacy slice classifiers;
* 32 independently cached slice-level feature views;
* multiple generations of candidate aliases and broad report orchestration.

They are not declared useless.  They are moved out of the explanatory core
because they answer historical debugging questions rather than the final
scientific question.

END-TO-END PIPELINE
-------------------

    JPEG cardiac MRI files
            |
            v
    Directory_* patient grouping
            |
            v
    Conservative label-blind dark-border removal
            |
            v
    Fixed 240-in-256 geometry with preserved aspect ratio
            |
            v
    Pinned pretrained MONAI ventricular segmentation
            |
            v
    One exact final binary support per slice
            |-----------------------|-----------------------|
            v                       v                       v
    cardiac intensity        support-only control    exact complement
            |
            v
    deterministic within-support shuffled-intensity control
            |
            v
    Frozen ImageNet EfficientNet-B0 embeddings
            |
            v
    mean embedding inside each folder-defined series proxy
            |
            v
    equal mean across all series proxies of one patient
            |
            v
    exactly one feature vector and one label per patient
            |
            v
    nested patient-level cross-validation
            |
            v
    AUROC, AUPRC, threshold metrics, repeated-split stability,
    exact-control comparisons, and nuisance-score predictability

DATA AND SHAPE CONTRACT
-----------------------
A. SliceRecord

    path       : path to one image file
    label      : 0 or 1, carried as metadata only during frozen inference
    patient_id : Directory_* identifier
    series_id  : patient-scoped folder proxy

B. SliceDataset item

    raw       : [1, 256, 256] retained uint8 intensity divided by 255
    monai     : [1, 256, 256] min-max view for the MONAI segmenter
    content   : [1, 256, 256] mask of retained content versus canvas padding
    metadata  : compact acquisition/export proxy vector
    label     : scalar patient label, never passed to MONAI or EfficientNet
    patient_id, series_id, decoded-pixel hash, perceptual hash

C. MONAI output

    logits             : [B, 4, 256, 256]
    heart probability  : sum of LV blood-pool, LV myocardium, and RV channels
    valid gate          : one Boolean per slice
    exact support       : [B, 1, 256, 256]

D. Four exact matched image views

    cardiac             : MRI intensity visible only inside exact support
    support_only        : exact support without MRI intensity
    shuffled_intensity  : exact support + exact intensity multiset, shuffled
    outside_support     : independently scaled exact support complement

E. EfficientNet output

    one frozen 1,280-dimensional embedding per slice and view

F. Series cache

    one mean embedding per series proxy and view, plus metadata and slice count

G. Patient representation

    equal mean across the patient's series-proxy means

FROZEN VERSUS FITTED COMPONENTS
-------------------------------
Frozen and label-independent:

* dark-border detection and fixed-canvas geometry;
* pinned MONAI ventricular segmenter;
* exact support construction;
* matched control construction;
* ImageNet EfficientNet-B0 encoder;
* decoded-pixel and perceptual hashing;
* slice-to-series and series-to-patient mean rules;
* fixed random projection used only to compact nuisance features.

Fitted separately inside training partitions:

* StandardScaler objects;
* PCA for cardiac representations;
* PCA for nuisance representations;
* ridge coefficients used by nuisance residualization;
* Logistic Regression coefficients;
* the custom confounder-guard coefficients;
* hyperparameters selected by inner cross-validation;
* the operating threshold selected from inner out-of-fold scores.

Validation-patient labels never fit preprocessing, nuisance removal,
classification, hyperparameter selection, or threshold selection.

COMPUTATIONAL SIMPLIFICATION
----------------------------
The full audit conceptually stores:

    63,425 slices x 32 views x 1,280 features

This file stores approximately:

    2,859 series proxies x 4 views x 1,280 features

For mean-based hierarchical pooling, online accumulation is algebraically exact:

    series_mean = (1 / n_s) * sum_i embedding(slice_i)

No label is used in this compression.  The reduction primarily saves disk space
and makes the data flow easier to explain.

CUSTOM ALGORITHM A — FOLD-LOCAL NUISANCE PROJECTION
---------------------------------------------------
Inputs:

    X : patient cardiac embedding matrix
    Z : nuisance matrix built from support-only features, outside-support
        features, acquisition/export metadata, series count, and slice count
    y : patient labels

Inside each training fold:

1. Standardize and compress Z using training patients only.
2. Center X using the training mean.
3. Solve the ridge problem

       B* = argmin_B ||X_c - Z_c B||_F^2 + alpha ||B||_F^2

4. Use the closed-form solution

       B* = (Z_c^T Z_c + alpha I)^(-1) Z_c^T X_c

5. Remove the predictable nuisance component

       X_clean = X_c - Z_c B*

6. Fit the ordinary compact PCA + Logistic Regression model to X_clean.

Why it may help:
A cardiac embedding dimension may respond to both myocardial texture and JPEG
export geometry.  If the export-related part is predictable from Z, the ridge
projection removes it while retaining unexplained cardiac variation.

Main risk:
A nuisance variable can contain real biological signal or can be causally linked
to clinical workflow.  Projection may remove legitimate information.  The
method is therefore a sensitivity analysis, not a causal guarantee.

CUSTOM ALGORITHM B — WORST-CASE LINEAR CONFOUNDER GUARD
-------------------------------------------------------
Let

    s = Xw + b

be the patient score and let Z contain centered nuisance axes.  The model solves

    min_(w,b) logistic_loss(y, s)
              + lambda * max_(||a||_2 <= 1) Cov(s, Za)^2
              + (gamma / 2) ||w||_2^2.

The inner adversarial maximum has the closed form

    max_(||a||_2 <= 1) Cov(s, Za)^2
      = || Z^T (s - mean(s)) / n ||_2^2.

Therefore no second adversarial neural network is needed.  The code optimizes
one differentiable objective with explicit gradients and backtracking line
search.

Conceptual interpretation:

* the predictor minimizes classification error;
* an implicit adversary finds the strongest linear nuisance direction;
* the covariance penalty makes scores expensive when that adversary can align
  them with acquisition or support-derived features.

Why it may help:
Unlike hard residualization, the model may retain a feature direction when it is
important for the label, while paying a penalty only when the final decision
score remains strongly aligned with nuisance axes.

Main risk:
Only linear score dependence is penalized.  Nonlinear confounding can remain.
The 30-patient sample is too small to justify a high-capacity adversarial neural
network as the principal method.

CUSTOM ALGORITHM C — CONSERVATIVE BAYESIAN-STYLE SERIES FUSION
--------------------------------------------------------------
A simple patient mean can dilute a focal signal found in a few informative
series.  A literal product of hundreds of series probabilities is also invalid
because those series are correlated and would create extreme overconfidence.

The classifier is trained on one mean embedding per training patient.  The same
decision function then scores each validation series.  For series s:

    reliability r_s = n_s / (n_s + tau)

where n_s is the slice count.  Patient evidence is combined as

    logit(p_patient) = logit(prior)
                       + sum_s r_s [logit(p_s) - logit(prior)] / sum_s r_s.

The prior is estimated from the outer-training cohort only.

Important consequence:
Duplicating identical series evidence does not multiply certainty without bound,
because evidence is averaged rather than naively multiplied.

Main risk:
A classifier trained on patient means may produce noisy series-level scores.
This method is experimental, not the default clinical interpretation.

OPTIONAL FUTURE ALGORITHM D — SIMULATED-ANNEALING POOLING SEARCH
---------------------------------------------------------------
This method is documented but intentionally not enabled in the seven-method
panel.  A small pooling state could combine mean, top-quartile, and maximum
series evidence:

    w = (w_mean, w_top25, w_max),  w_i >= 0,  sum_i w_i = 1.

A fold-local objective could be

    J(w) = AUC_inner(w) - beta * max(0, nuisance_R2(w)).

A neighboring state transfers a small amount of weight between two coordinates.
Simulated annealing may accept a temporarily worse state with probability

    exp((J_new - J_old) / T).

This must be searched only inside inner cross-validation.  Searching many
pooling rules on all 30 labels would recreate the multiple-comparison problem
the simplified pipeline is designed to avoid.

OPTIONAL FUTURE ALGORITHM E — CONFOUNDER-MATCHED k-NEAREST NEIGHBORS
-------------------------------------------------------------------
A diagnostic k-nearest-neighbor adaptation could compare a validation patient
only with training patients that are close in nuisance space, then vote using
cardiac-space distance.  One possible combined distance is

    d(i,j) = ||x_i - x_j||_2^2 + eta ||z_i - z_j||_2^2,

with a hard nuisance-distance eligibility threshold.

With only 30 patients, this should remain a diagnostic of whether discrimination
survives acquisition-style matching, not the principal model.

RELATION TO CS50'S INTRODUCTION TO ARTIFICIAL INTELLIGENCE WITH PYTHON
---------------------------------------------------------------------
The code does not copy course assignments.  It adapts central ideas to a new
medical-imaging research problem:

* Search and data structures:
  a BK-tree performs complete Hamming-radius search for perceptual-hash
  duplicate candidates.

* Minimax:
  the confounder guard treats the strongest linear nuisance direction as an
  adversary to the disease predictor.

* Uncertainty:
  Bayesian-style fusion combines prior-relative series evidence without assuming
  hundreds of correlated series are independent.

* Optimization:
  explicit objective functions, hyperparameter search, gradient descent, and
  backtracking line search are implemented directly.

* Learning:
  all supervised fitting occurs at patient level with strict train/test
  separation.

* Neural networks:
  MONAI and EfficientNet are frozen representation functions because only 30
  labeled patients are available.

Official course notes used as conceptual references:

    https://cs50.harvard.edu/ai/notes/0/  search and minimax
    https://cs50.harvard.edu/ai/notes/2/  uncertainty and Bayesian networks
    https://cs50.harvard.edu/ai/notes/3/  optimization
    https://cs50.harvard.edu/ai/notes/4/  learning
    https://cs50.harvard.edu/ai/notes/5/  neural networks and gradient descent

DUPLICATE AUDIT
---------------
Exact decoded-pixel SHA-256 hashes are computed during feature extraction.
Confirmed cross-patient exact duplicates can stop the run before evaluation.

A 64-bit DCT perceptual hash and a BK-tree provide a complete radius search for
near-duplicate screening candidates.  A close perceptual hash is not proof that
two images are duplicates.  The output must be manually reviewed before using
perceptual candidates to alter fold grouping.

EVALUATION DESIGN
-----------------
The outer loop estimates patient-level generalization within the released
cohort.  The inner loop selects model hyperparameters and the operating
threshold.  Every patient receives exactly one outer out-of-fold score per run.

Repeated cross-validation changes the patient split seed and is used to examine
ranking stability.  Repeated runs reuse the same 30 patients; they are
sensitivity analyses, not independent validation cohorts.

The summary is ranked by repeated-CV median AUROC rather than by one favorable
split.

NUISANCE-SCORE R-SQUARED
------------------------
For each model, the pipeline asks how well nuisance variables can reconstruct
the model's out-of-fold score.  A fold-local Ridge model predicts the score from
nuisance features only:

    nuisance_score_R2 = 1 - SSE / SST.

Interpretation:

* high positive R2: model score is substantially predictable from nuisance;
* near zero: little linear out-of-fold reconstructability;
* negative: nuisance prediction generalizes worse than predicting the mean.

This diagnostic measures modeled linear dependence; it is not proof of causal
independence.

HOW TO JUDGE SUCCESS
--------------------
A custom method is promising only if both conditions are considered:

1. repeated-CV AUROC is preserved or improved relative to BASELINE_A17;
2. nuisance_score_R2 decreases.

A high AUROC with unchanged or increased nuisance predictability is not evidence
that confounding has been solved.

The most important comparisons are:

    BASELINE_A17 - CONTROL_SUPPORT_ONLY
    BASELINE_A17 - CONTROL_SHUFFLED_INTENSITY
    BASELINE_A17 - CONTROL_OUTSIDE_SUPPORT
    CUSTOM_NUISANCE_PROJECTION - BASELINE_A17
    CUSTOM_MINIMAX_CONFOUNDER_GUARD - BASELINE_A17
    CUSTOM_BAYESIAN_SERIES_FUSION - BASELINE_A17

A method that slightly lowers AUROC while sharply reducing nuisance dependence
may be scientifically more valuable than a method that maximizes AUROC alone.

PORTABLE DATASET AND OUTPUT PATH RESOLUTION
-------------------------------------------
The same source file runs in Kaggle, Colab/Jupyter, Windows, Linux, or macOS.
The entry point intentionally does NOT parse ``sys.argv``.  Notebook kernels
usually inject internal arguments such as ``-f <kernel.json>``; ignoring command-
line arguments prevents those kernel parameters from reaching a command-line
parser and raising ``SystemExit: 2``.

Runtime paths are resolved before the expensive neural-network stage by the
following fixed priority:

Dataset root:

    1. optional source-code constant ``MAIN_DATASET_PATH_OVERRIDE``;
    2. ``CAD_MRI_DATASET_PATH`` environment variable;
    3. ``CAD_DATASET_PATH`` compatibility environment variable;
    4. exact Kaggle paths used by this project;
    5. a bounded search below ``/kaggle/input`` for a directory containing both
       ``Normal/`` and ``Sick/``;
    6. the validated Windows path used by the full V6 suite;
    7. common dataset folder names beside the script, in the current working
       directory, or in their ``datasets/`` subfolders.

Output directory:

    1. optional source-code constant ``MAIN_OUTPUT_DIR_OVERRIDE``;
    2. ``CAD_MRI_OUTPUT_DIR`` environment variable;
    3. ``CAD_OUTPUT_DIR`` compatibility environment variable;
    4. ``/kaggle/working/cad_mri_university_showcase`` on Kaggle;
    5. ``cad_mri_university_showcase_outputs`` beside the script locally, or in
       the current working directory when the script directory is not writable.

An explicitly supplied source-code or environment path is strict: if it does not
lead to a directory containing both ``Normal/`` and ``Sick/``, the script stops
with a detailed error instead of silently using another dataset.

KAGGLE — NORMAL RUN WITHOUT MANUAL ARGUMENTS
-----------------------------------------------
Attach the public dataset:

    danialsharifrazi/cad-cardiac-mri-dataset

Install the lightweight dependency if it is not already present:

    !pip -q install huggingface_hub

Then execute the source file directly, without command-line arguments:

    %run /kaggle/input/YOUR-CODE-DATASET/cad_mri_university_showcase.py

or:

    !python /kaggle/input/YOUR-CODE-DATASET/cad_mri_university_showcase.py

The script recognizes both common Kaggle mount forms:

    /kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset
    /kaggle/input/cad-cardiac-mri-dataset

It also performs a small bounded search below ``/kaggle/input`` so a renamed
Kaggle dataset slug can still be found from the required ``Normal/`` and
``Sick/`` directory contract.

LOCAL — AUTOMATIC OR EXPLICIT SOURCE-CODE OVERRIDE
--------------------------------------------------
The validated V6 local default remains:

    C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset

The script also checks common relative locations such as:

    ./CAD Cardiac MRI Dataset
    ./cad-cardiac-mri-dataset
    ./datasets/CAD Cardiac MRI Dataset
    ./datasets/cad-cardiac-mri-dataset

For another location, edit the small MAIN EXECUTION SETTINGS block near the
configuration section:

    MAIN_DATASET_PATH_OVERRIDE = r"D:\Datasets\CAD Cardiac MRI Dataset"
    MAIN_OUTPUT_DIR_OVERRIDE = r"D:\Results\cad_mri_university_showcase"
    MAIN_BATCH_SIZE = 8
    MAIN_REPEATED_CV_RUNS = 20

Then run without arguments:

    python cad_mri_university_showcase.py

ENVIRONMENT-VARIABLE OVERRIDES
------------------------------
Environment variables are useful in IDEs, notebooks, containers, and scheduled
runs:

Windows PowerShell:

    $env:CAD_MRI_DATASET_PATH = "D:\Datasets\CAD Cardiac MRI Dataset"
    $env:CAD_MRI_OUTPUT_DIR = "D:\Results\cad_mri_university_showcase"
    python cad_mri_university_showcase.py

Linux/macOS:

    export CAD_MRI_DATASET_PATH="$HOME/datasets/cad-cardiac-mri-dataset"
    export CAD_MRI_OUTPUT_DIR="$HOME/results/cad_mri_university_showcase"
    python cad_mri_university_showcase.py

PATH-ONLY DIAGNOSTIC
--------------------
Before starting a long run, set this source-code constant:

    MAIN_CHECK_PATHS_ONLY = True

and execute the file normally.  The script validates the paths, prints them,
and exits before loading MONAI or decoding any image.  It prints:

    [PATHS] Runtime environment: Kaggle or local
    [PATHS] Dataset source: ...
    [PATHS] Dataset root: ...
    [PATHS] Output source: ...
    [PATHS] Output directory: ...

Explicit paths remain supported on Kaggle through the same source-code
overrides or environment variables.  The default uses full-precision neural
inference.  Set ``MAIN_USE_AMP = True`` only for a faster exploratory extraction.
The first run downloads the pinned MONAI TorchScript artifact and EfficientNet-B0
weights.  Later runs reuse model caches and the compact series feature bank stored
below the selected output directory.

EXPECTED FIRST-RUN WORK
-----------------------
The first run will:

* discover the Directory_* patients;
* download and verify the pinned MONAI artifact if absent;
* load EfficientNet-B0 ImageNet weights;
* decode every image;
* build four exact matched representations;
* store only series-level means;
* perform exact and perceptual duplicate screening;
* run the seven nested-CV experiments;
* run repeated patient-level split sensitivity;
* write machine-readable OOF, parameter, summary, and report files.

MAIN OUTPUTS
------------

    configuration.json
    run_metadata.json
    duplicate_audit.json
    series_feature_bank_<fingerprint>.npz
    series_feature_bank_<fingerprint>.json
    experiment_summary.csv
    repeated_cv_results.csv
    oof_<experiment>.csv
    selected_params_<experiment>.json
    final_report.json

experiment_summary.csv is ranked by repeated-CV median AUROC.

SUGGESTED UNIVERSITY-APPLICATION PROJECT NARRATIVE
--------------------------------------------------
One-sentence description:

    I built a patient-level cardiac MRI classifier, discovered that its high
    performance was partly reproducible from non-cardiac and segmentation-derived
    shortcuts, and redesigned the project around exact matched controls and
    custom mathematical deconfounding.

Intellectual turning point:
The first model appeared successful because its patient-level AUROC was high.
Instead of stopping, I asked whether the model could classify patients when the
heart was removed, when only the support mask remained, or when intensities were
spatially shuffled.  Several controls remained predictive.  That changed the
project from model optimization into an investigation of shortcut learning and
scientific validity.

Personal implementation highlights:

* patient-level grouping and nested cross-validation;
* pinned MONAI segmentation;
* exact matched image controls;
* compact series-level feature caching;
* decoded-pixel hashing and BK-tree near-duplicate screening;
* fold-local nuisance residualization using ridge linear algebra;
* a minimax-inspired logistic objective optimized from scratch;
* Bayesian-style series evidence fusion;
* repeated split analysis and nuisance-predictability diagnostics.

Strongest mathematical contribution:
Rather than training a large adversarial network on 30 patients, derive the
strongest linear nuisance direction in closed form and penalize its covariance
with the classifier score.  This keeps the method interpretable and appropriate
for the sample size.

Honest result statement:
The project can demonstrate robust internal association with the released
Normal/Sick labels and test whether spatial cardiac information contributes
beyond exact controls.  It does not, by itself, establish clinical CAD diagnosis.
Protocol/export confounding, sequence/view matching, and external validation
remain decisive.

Why the project matters:
In high-stakes AI, accuracy is insufficient when a model can be right for the
wrong reason.  The contribution is the experimental framework that makes the
source of predictive information testable.

RECOMMENDED PRESENTATION ORDER
------------------------------
Because this file embeds the complete explanation, a repository can be presented
in a compact order:

1. cad_mri_university_showcase.py
2. a results folder from a complete Kaggle run
3. an archival link to the full 58-experiment audit
4. optional figures or a concise research report

The strongest narrative is not simply "I obtained AUROC 0.97."  It is:

    I challenged an apparently successful medical AI result, designed controls
    that exposed shortcut signals, and developed interpretable mathematical
    methods to test and reduce those dependencies.

SCIENTIFIC LIMITATIONS
----------------------
* The effective labeled sample size is 30 patients.
* Repeated splits reuse the same patients and are not external cohorts.
* The MONAI model is short-axis-specific, while the JPEG release is heterogeneous.
* Folder-defined series are proxies, not verified DICOM series identifiers.
* The Normal/Sick endpoint does not identify which pixels represent coronary
  pathology.
* Support geometry can contain true anatomy and protocol information at the same
  time.
* Nuisance projection and covariance penalties remove only modeled linear
  dependence.
* A high negative-control AUROC means unresolved dataset structure remains.
* Model outputs are research scores, not calibrated clinical probabilities.
* Manually verified sequence/view matching remains important.
* Independent external validation is required before a clinical claim.

REPRODUCIBILITY AND ARTIFACT INTEGRITY
--------------------------------------
* The MONAI repository revision is pinned.
* The MONAI TorchScript SHA-256 is verified before inference.
* EfficientNet uses an explicit ImageNet weight enum.
* Configuration and software versions are written to JSON.
* Feature-affecting settings and the file manifest determine the cache fingerprint.
* Exact decoded-pixel hashes are computed during extraction.
* Random seeds are explicit, while the code does not overclaim bitwise identity
  across every hardware and library combination.

LOCAL VALIDATION EXPECTATIONS
-----------------------------
Before distribution, the source should pass:

* Python compilation;
* module import;
* configuration validation;
* command-line --help;
* exact support/complement separation tests;
* shuffled-intensity multiset preservation;
* deterministic affine-permutation tests;
* nuisance-residualization sanity tests;
* confounder-guard loss-decrease tests;
* bounded Bayesian-fusion tests;
* all seven nested-CV paths on synthetic patient data;
* report-writing tests.

Passing those checks validates program structure and mathematical contracts.  It
does not replace a complete run on the public dataset with the real MONAI and
EfficientNet models.

CURRENT ARTIFACT VALIDATION
---------------------------
The university-generic, self-contained source delivered with this project was
checked after documentation integration:

* Python compilation: PASS
* AST parsing: PASS
* module import: PASS
* notebook-safe main that ignores kernel ``sys.argv``: PASS
* configuration validation: PASS
* all classes and functions have explanatory docstrings: PASS
* module-level research guide exceeds 600 lines: PASS
* exact support / complement separation: PASS
* shuffled-intensity multiset preservation: PASS
* deterministic affine permutation: PASS
* nuisance residualization sanity check: PASS
* confounder-guard accepted-loss monotonicity: PASS
* Bayesian fusion boundedness and duplicate-evidence invariance: PASS
* all seven nested-CV method paths on synthetic 30-patient data: PASS
* complete orchestration and report writing with a synthetic feature bank: PASS

These checks validate implementation contracts only.  The revised source has not
been rerun here over all 63,425 images with the real pinned MONAI and EfficientNet
models, so no new real-data AUROC is claimed by this documentation revision.
"""

from __future__ import annotations
# Postpone evaluation of type annotations.  This keeps annotations readable and
# avoids importing or resolving every referenced type at function-definition time.

import csv
# Writes patient-level OOF predictions, repeated-CV results, and ranked summaries.

import hashlib
# Creates checkpoint digests, decoded-pixel hashes, and feature-cache identities.

import json
# Serializes configuration, software provenance, duplicate audits, and reports.

import math
# Provides numerically explicit operations such as gcd, square roots, and logic
# used by the affine permutation and custom optimization procedures.

import os
# Handles environment-aware path logic and portable folder traversal.

import platform
# Records the Python/runtime platform in run_metadata.json for reproducibility.

import random
# Seeds Python's pseudo-random generator alongside NumPy and PyTorch.

import time
# Measures model loading, extraction, evaluation, and total execution time.

from collections import defaultdict
# Accumulates per-series feature sums and duplicate-hash patient memberships.

from dataclasses import asdict, dataclass
# Defines explicit data contracts and converts configuration to machine-readable JSON.

from pathlib import Path
# Provides safe, portable path construction for datasets, caches, and outputs.

from typing import Any, Literal, Sequence
# Makes tensor/data contracts and allowed argument values visible to reviewers.

import cv2
# Decodes grayscale images, resizes canvases, and computes the DCT pHash.

import numpy as np
# Implements array algebra, custom optimization, metrics preparation, and pooling.

import sklearn
# The package object is imported so its exact version can be written to metadata.

import torch
# Runs the frozen MONAI and EfficientNet models and batched image transformations.

import torch.nn as nn
# Supplies the Module abstraction used by the frozen EfficientNet wrapper.

import torch.nn.functional as F
# Supplies softmax, interpolation, and max-pooling used for segmentation support.

import torchvision
# The package object is imported to record its exact installed version.

from sklearn.decomposition import PCA
# Performs fold-local compact representations for cardiac and nuisance features.

from sklearn.linear_model import LogisticRegression, Ridge
# Logistic Regression is the small-cohort classifier; Ridge supports nuisance tests.

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
# All reported discrimination and threshold metrics are calculated at patient level.

from sklearn.model_selection import StratifiedKFold
# Creates outer and inner patient-level folds while preserving class proportions.

from sklearn.preprocessing import StandardScaler
# Fits feature scaling only on the current training partition.

from torch.utils.data import DataLoader, Dataset
# Decodes many images in deterministic batches without redefining the patient unit.

from torchvision import models
# Provides the explicitly selected ImageNet EfficientNet-B0 architecture and weights.

from tqdm import tqdm

# Displays progress during the expensive all-slice frozen feature extraction stage.


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


# -----------------------------------------------------------------------------
# PORTABLE RUNTIME PATHS
# -----------------------------------------------------------------------------
#
# The full multi-experiment V6 suite used one simple environment split:
#
#     Kaggle canonical dataset path, when present
#         otherwise
#     a validated Windows development path.
#
# This compact portfolio keeps those two paths and extends them conservatively:
# source-code and environment overrides have priority, Kaggle input is searched
# only to a bounded depth, and local relative folders are checked beside both the
# script and the current working directory.  Detection is based on the actual
# data contract -- a root must contain both ``Normal/`` and ``Sick/`` -- rather
# than on a folder name alone.

DATASET_ENVIRONMENT_VARIABLES = (
    "CAD_MRI_DATASET_PATH",
    "CAD_DATASET_PATH",
)
# The first variable is the documented name.  The second is retained as a short
# compatibility alias for existing notebooks and local automation.

OUTPUT_ENVIRONMENT_VARIABLES = (
    "CAD_MRI_OUTPUT_DIR",
    "CAD_OUTPUT_DIR",
)

KAGGLE_DATASET_CANDIDATES = (
    Path("/kaggle/input/datasets/danialsharifrazi/cad-cardiac-mri-dataset"),
    Path("/kaggle/input/cad-cardiac-mri-dataset"),
)
# The first path matches the validated V6 configuration.  The second is Kaggle's
# common direct slug mount.  A bounded search below /kaggle/input is attempted
# afterward, so a harmless slug rename does not require source-code editing.

VALIDATED_LOCAL_DATASET_CANDIDATES = (
    Path(r"C:\F\_Develop\AI\Datasets\CAD Cardiac MRI Dataset"),
)
# This is the local Windows path used by the complete V6 suite.  It is treated as
# a candidate rather than an unconditional default, so Linux/macOS runs do not
# receive a misleading Windows path when the folder is absent.

COMMON_DATASET_DIRECTORY_NAMES = (
    "CAD Cardiac MRI Dataset",
    "cad-cardiac-mri-dataset",
)
# These names are searched only in a small set of predictable local locations.
# The code never scans an entire drive or home directory.

DEFAULT_KAGGLE_OUTPUT_DIRECTORY = Path(
    "/kaggle/working/cad_mri_university_showcase"
)
DEFAULT_LOCAL_OUTPUT_DIRECTORY_NAME = "cad_mri_university_showcase_outputs"

# -----------------------------------------------------------------------------
# MAIN EXECUTION SETTINGS — EDIT THESE INSTEAD OF PASSING COMMAND-LINE ARGUMENTS
# -----------------------------------------------------------------------------
# Jupyter, Colab, VS Code notebooks, and IPython kernels add their own arguments
# to ``sys.argv``.  A common example is:
#
#     -f /root/.local/share/jupyter/runtime/kernel-....json
#
# The previous command-line parser interpreted that kernel argument as a user
# error and stopped with ``SystemExit: 2``.  The entry point below never
# reads ``sys.argv``.  Runtime options are deliberately visible here as ordinary
# constants, which makes the same file safe in notebooks, Kaggle, and local Python.
#
# Leave path overrides as ``None`` for automatic Kaggle/local detection.  Set a
# string only when the dataset or output folder lives somewhere unusual.
MAIN_DATASET_PATH_OVERRIDE: str | None = None
MAIN_OUTPUT_DIR_OVERRIDE: str | None = None

MAIN_BATCH_SIZE = 8
MAIN_REPEATED_CV_RUNS = 20
MAIN_NUM_WORKERS = 0
MAIN_FORCE_REBUILD_CACHE = False
MAIN_USE_AMP = False
MAIN_CHECK_PATHS_ONLY = False


# Set MAIN_CHECK_PATHS_ONLY=True for a fast path diagnostic that loads no model.
# Notebook/Kaggle-cell safety: ``main()`` and run metadata never require
# ``__file__``.  When source code is executed directly from a cell, the script
# path/hash are recorded as unavailable instead of raising NameError.


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    """Resolved input/output paths plus their provenance.

    Keeping the source strings is useful for reproducibility.  Two runs can use
    the same physical dataset while one was resolved automatically and the other
    was pinned through a source-code override.  The resolved absolute paths and
    their sources are saved in ``configuration.json`` through the main ``Config``
    dataclass.
    """

    dataset_path: Path
    output_dir: Path
    dataset_source: str
    output_source: str
    runtime_environment: str
    checked_dataset_paths: tuple[str, ...]


def _script_directory() -> Path:
    """Return the source-file directory, with a notebook-safe fallback.

    ``__file__`` exists when the pipeline is executed as a normal Python script
    or imported from a file.  Some notebook execution methods omit it, in which
    case the current working directory is the only reliable portable anchor.
    """

    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _runtime_source_identity() -> dict[str, str | None]:
    """Return notebook-safe source provenance without requiring ``__file__``.

    WHY THIS HELPER EXISTS
    ----------------------
    A normal ``python script.py`` execution defines ``__file__``.  Kaggle,
    Colab and Jupyter also allow the entire source to be pasted/executed as a
    notebook cell; in that execution mode ``__file__`` is not defined at all.
    Reading ``Path(__file__)`` directly would therefore raise ``NameError``
    *after* expensive setup, even though the scientific pipeline itself is
    otherwise valid.

    The source-file hash is reproducibility metadata, not a model input.  When
    a real source file exists we save its absolute path and SHA-256.  When code
    is running from an in-memory notebook cell, we record that fact explicitly
    and use ``None`` for the unavailable file path/hash.  No synthetic path or
    misleading checksum is invented.

    Returns
    -------
    dict
        ``script_path`` and ``script_sha256`` when a regular file is available,
        plus ``script_identity_source`` describing how provenance was resolved.
    """

    try:
        candidate = Path(__file__).resolve(strict=False)
    except NameError:
        return {
            "script_path": None,
            "script_sha256": None,
            "script_identity_source": "notebook_cell_without___file__",
        }

    if candidate.is_file():
        return {
            "script_path": str(candidate),
            "script_sha256": sha256_file(candidate),
            "script_identity_source": "__file__",
        }

    # Some interactive launchers may technically define ``__file__`` to a
    # pseudo-name that is not a real file.  Treat it as notebook-style source
    # rather than failing or hashing a nonexistent path.
    return {
        "script_path": str(candidate),
        "script_sha256": None,
        "script_identity_source": "__file___not_a_regular_file",
    }


def _is_kaggle_runtime() -> bool:
    """Identify Kaggle from its writable working directory contract."""

    return Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()


def _normalize_runtime_path(value: str | os.PathLike[str]) -> Path:
    """Expand environment/user markers and return a non-strict absolute path.

    ``strict=False`` allows output paths to be resolved before they exist.  The
    dataset resolver subsequently verifies existence and the Normal/Sick folder
    contract explicitly.
    """

    raw = str(value).strip()
    if not raw:
        raise ValueError("A runtime path cannot be empty.")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    return Path(expanded).resolve(strict=False)


def _is_valid_dataset_root(path: Path) -> bool:
    """Return True only for a directory containing exact Normal/ and Sick/."""

    return (
            path.is_dir()
            and (path / "Normal").is_dir()
            and (path / "Sick").is_dir()
    )


def _bounded_directory_walk(
        base: Path,
        *,
        maximum_depth: int,
        maximum_directories: int = 500,
):
    """Yield directories breadth-first without performing an unbounded scan.

    The helper is used only around an explicitly supplied parent or below
    ``/kaggle/input``.  It stops after a fixed number of directories and never
    traverses symbolic-link directories, preventing accidental scans of a whole
    filesystem or cyclic links.
    """

    if maximum_depth < 0:
        raise ValueError("maximum_depth cannot be negative.")
    queue: list[tuple[Path, int]] = [(base, 0)]
    visited: set[str] = set()
    emitted = 0

    while queue and emitted < maximum_directories:
        current, depth = queue.pop(0)
        key = str(current)
        if key in visited:
            continue
        visited.add(key)
        if not current.is_dir():
            continue
        emitted += 1
        yield current

        if depth >= maximum_depth:
            continue
        try:
            children = sorted(
                (
                    child
                    for child in current.iterdir()
                    if child.is_dir()
                       and not child.is_symlink()
                       and not child.name.startswith(".")
                ),
                key=lambda child: child.name.casefold(),
            )
        except (OSError, PermissionError):
            continue
        queue.extend((child, depth + 1) for child in children)


def _find_dataset_root_from_candidate(
        candidate: Path,
        *,
        maximum_depth: int,
) -> Path | None:
    """Resolve a dataset root from a direct root or a nearby parent directory."""

    for possible_root in _bounded_directory_walk(
            candidate,
            maximum_depth=maximum_depth,
    ):
        if _is_valid_dataset_root(possible_root):
            return possible_root.resolve()
    return None


def _local_dataset_candidates() -> list[tuple[str, Path]]:
    """Return deterministic local candidates near cwd and the source file."""

    script_dir = _script_directory()
    cwd = Path.cwd().resolve()
    bases: list[tuple[str, Path]] = [
        ("current working directory", cwd),
        ("script directory", script_dir),
        ("current working directory/datasets", cwd / "datasets"),
        ("script directory/datasets", script_dir / "datasets"),
        ("script parent", script_dir.parent),
        ("script parent/datasets", script_dir.parent / "datasets"),
    ]

    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for source, base in bases:
        direct_key = str(base)
        if direct_key not in seen:
            seen.add(direct_key)
            candidates.append((source, base))
        for folder_name in COMMON_DATASET_DIRECTORY_NAMES:
            path = base / folder_name
            key = str(path)
            if key not in seen:
                seen.add(key)
                candidates.append((f"{source}/{folder_name}", path))
    return candidates


def resolve_dataset_path(explicit_path: str | None = None) -> tuple[Path, str, tuple[str, ...]]:
    """Resolve the CAD MRI dataset root using a strict documented priority.

    Explicit source-code/environment values are authoritative: a typo raises a detailed
    error instead of falling through to a different dataset.  Automatic search
    uses exact known paths first and then bounded nearby scans.  The returned
    root is guaranteed to contain ``Normal/`` and ``Sick/``.
    """

    checked: list[str] = []

    def inspect(
            raw_path: str | os.PathLike[str],
            source: str,
            *,
            maximum_depth: int,
            strict: bool,
    ) -> tuple[Path, str] | None:
        """Check one prioritized candidate and enforce strict overrides."""

        candidate = _normalize_runtime_path(raw_path)
        checked.append(f"{source}: {candidate}")
        found = _find_dataset_root_from_candidate(
            candidate,
            maximum_depth=maximum_depth,
        )
        if found is not None:
            return found, source
        if strict:
            raise FileNotFoundError(
                "The explicitly selected dataset path is invalid.\n"
                f"Source: {source}\n"
                f"Path:   {candidate}\n"
                "Expected a directory containing both Normal/ and Sick/, or a "
                "nearby parent containing that dataset root."
            )
        return None

    if explicit_path is not None:
        result = inspect(
            explicit_path,
            "source-code MAIN_DATASET_PATH_OVERRIDE",
            maximum_depth=3,
            strict=True,
        )
        assert result is not None
        return result[0], result[1], tuple(checked)

    for variable_name in DATASET_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable_name)
        if value and value.strip():
            result = inspect(
                value,
                f"environment variable {variable_name}",
                maximum_depth=3,
                strict=True,
            )
            assert result is not None
            return result[0], result[1], tuple(checked)

    for candidate in KAGGLE_DATASET_CANDIDATES:
        result = inspect(
            candidate,
            f"known Kaggle path {candidate}",
            maximum_depth=1,
            strict=False,
        )
        if result is not None:
            return result[0], result[1], tuple(checked)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        result = inspect(
            kaggle_input,
            "bounded Kaggle /kaggle/input search",
            maximum_depth=4,
            strict=False,
        )
        if result is not None:
            return result[0], result[1], tuple(checked)

    for candidate in VALIDATED_LOCAL_DATASET_CANDIDATES:
        result = inspect(
            candidate,
            f"validated V6 local path {candidate}",
            maximum_depth=1,
            strict=False,
        )
        if result is not None:
            return result[0], result[1], tuple(checked)

    for source, candidate in _local_dataset_candidates():
        result = inspect(
            candidate,
            f"automatic local candidate: {source}",
            maximum_depth=2,
            strict=False,
        )
        if result is not None:
            return result[0], result[1], tuple(checked)

    attempted = "\n".join(f"  - {entry}" for entry in checked)
    raise FileNotFoundError(
        "Could not locate the CAD Cardiac MRI dataset automatically.\n"
        "A valid root must contain both Normal/ and Sick/.\n\n"
        "Set one of the following:\n"
        "  MAIN_DATASET_PATH_OVERRIDE = '/path/to/dataset'\n"
        "  CAD_MRI_DATASET_PATH=/path/to/dataset\n\n"
        "Checked locations:\n"
        f"{attempted}"
    )


def resolve_output_directory(explicit_path: str | None = None) -> tuple[Path, str]:
    """Resolve a writable output destination without touching the dataset.

    Creation is deferred to ``run_research_portfolio``.  Keeping resolution and
    creation separate lets ``MAIN_CHECK_PATHS_ONLY`` inspect the decision without
    writing any file.
    """

    if explicit_path is not None:
        return (
            _normalize_runtime_path(explicit_path),
            "source-code MAIN_OUTPUT_DIR_OVERRIDE",
        )

    for variable_name in OUTPUT_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable_name)
        if value and value.strip():
            return (
                _normalize_runtime_path(value),
                f"environment variable {variable_name}",
            )

    if _is_kaggle_runtime():
        return (
            DEFAULT_KAGGLE_OUTPUT_DIRECTORY.resolve(strict=False),
            "automatic Kaggle /kaggle/working output",
        )

    script_dir = _script_directory()
    local_base = (
        script_dir
        if script_dir.is_dir() and os.access(script_dir, os.W_OK)
        else Path.cwd().resolve()
    )
    return (
        (local_base / DEFAULT_LOCAL_OUTPUT_DIRECTORY_NAME).resolve(strict=False),
        "automatic local output beside script"
        if local_base == script_dir
        else "automatic local output in current working directory",
    )


def resolve_runtime_paths(
        dataset_argument: str | None,
        output_argument: str | None,
) -> ResolvedRuntimePaths:
    """Resolve, validate, and describe all paths needed by one run."""

    dataset_path, dataset_source, checked = resolve_dataset_path(dataset_argument)
    output_dir, output_source = resolve_output_directory(output_argument)
    runtime_environment = "Kaggle" if _is_kaggle_runtime() else "local"
    return ResolvedRuntimePaths(
        dataset_path=dataset_path,
        output_dir=output_dir,
        dataset_source=dataset_source,
        output_source=output_source,
        runtime_environment=runtime_environment,
        checked_dataset_paths=checked,
    )


def print_resolved_runtime_paths(paths: ResolvedRuntimePaths) -> None:
    """Print one clear path-resolution summary before any expensive work."""

    print("\n" + "=" * 78, flush=True)
    print("PORTABLE RUNTIME PATHS", flush=True)
    print("=" * 78, flush=True)
    print(
        f"[PATHS] Runtime environment: {paths.runtime_environment}",
        flush=True,
    )
    print(f"[PATHS] Dataset source: {paths.dataset_source}", flush=True)
    print(f"[PATHS] Dataset root: {paths.dataset_path}", flush=True)
    print(f"[PATHS] Output source: {paths.output_source}", flush=True)
    print(f"[PATHS] Output directory: {paths.output_dir}", flush=True)
    print(
        "[PATHS] Dataset contract: Normal/ and Sick/ are present.",
        flush=True,
    )
    print("=" * 78, flush=True)


@dataclass(frozen=True)
class Config:
    """Immutable configuration for one complete research run.

    The fields are grouped by scientific role: reproducibility, geometry,
    segmentation, normalization, frozen checkpoints, nested evaluation, custom
    algorithms, caching, and duplicate auditing.  A configuration value may be
    chosen before a run or selected inside inner cross-validation, but it must
    never be adjusted after examining an outer-validation result.

    Serializing this dataclass makes it possible to reproduce exactly which
    assumptions created a feature bank and which assumptions affected only the
    downstream patient-level models.
    """

    dataset_path: str
    # Root containing the two class folders ``Normal`` and ``Sick``.

    output_dir: str
    # Run-specific destination for configuration, cache, OOF predictions, and reports.

    dataset_path_source: str = "provided programmatically"
    # Records whether the path came from a source-code override, an environment
    # variable, a known
    # Kaggle mount, the validated V6 Windows path, or automatic local discovery.

    output_dir_source: str = "provided programmatically"
    # Saved beside the resolved output path for reproducible runtime provenance.

    runtime_environment: str = "programmatic"
    # Usually ``Kaggle`` or ``local`` when the notebook-safe entry point resolves it.

    # -------------------------------------------------------------------------
    # REPRODUCIBILITY AND HARDWARE
    # -------------------------------------------------------------------------
    # These values affect fold generation, batching, and numerical precision.
    # ``use_amp`` is opt-in because full precision is easier to reproduce and to
    # compare against the validated V6 audit.
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    use_amp: bool = False

    # -------------------------------------------------------------------------
    # GEOMETRY AND SEGMENTATION
    # -------------------------------------------------------------------------
    # MONAI receives a 256x256 canvas.  EfficientNet later receives a uniformly
    # resized 224x224 version of the complete aligned canvas.  The hard threshold,
    # plausibility gate, and dilation widths are fixed before model evaluation.
    monai_input_size: int = 256
    encoder_input_size: int = 224
    standardized_content_long_side: int = 240
    monai_hard_threshold: float = 0.50
    monai_min_area_ratio: float = 0.003
    monai_max_area_ratio: float = 0.50
    monai_min_peak_probability: float = 0.50
    monai_dilation_kernel: int = 31
    a17_extra_dilation_kernel: int = 15
    fallback_square_fraction: float = 0.65

    # -------------------------------------------------------------------------
    # REGION-ONLY ROBUST NORMALIZATION
    # -------------------------------------------------------------------------
    # Percentiles are estimated only from pixels visible in the final candidate
    # or control.  This prevents excluded heart pixels from setting the scale of
    # an outside-support control and vice versa.
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    histogram_bins: int = 256
    minimum_visible_pixels: int = 64
    minimum_dynamic_range: float = 8.0 / 255.0

    # -------------------------------------------------------------------------
    # CONSERVATIVE LABEL-BLIND DARK-BORDER REMOVAL
    # -------------------------------------------------------------------------
    # A crop is accepted only for consecutive edge lines that are both dark and
    # nearly uniform, and only if a large central fraction remains.  Crop geometry
    # is retained as nuisance metadata rather than hidden from the audit.
    dark_line_max_mean: float = 12.0
    dark_line_max_std: float = 4.0
    dark_pixel_max_value: int = 20
    dark_pixel_min_fraction: float = 0.98
    max_crop_fraction_per_side: float = 0.20
    min_retained_fraction: float = 0.60
    min_padding_run: int = 2

    # -------------------------------------------------------------------------
    # FROZEN MODEL IDENTITIES
    # -------------------------------------------------------------------------
    # The MONAI repository revision and SHA-256 are pinned.  EfficientNet uses an
    # explicit weight enum rather than ``DEFAULT`` so a future torchvision release
    # cannot silently change the encoder checkpoint.
    monai_repo_id: str = "MONAI/ventricular_short_axis_3label"
    monai_revision: str = "eefc17c8e002cc8a567bbfce8f02d7d3116408f4"
    monai_filename: str = "models/model.ts"
    monai_sha256: str = (
        "27d5532401fa6c1883872fa21635adbb7615981e7f385d0c58dd75b355e340b3"
    )
    efficientnet_weights: str = "IMAGENET1K_V1"
    embedding_dim: int = 1280

    # -------------------------------------------------------------------------
    # NESTED PATIENT-LEVEL EVALUATION
    # -------------------------------------------------------------------------
    # Hyperparameters and thresholds are selected in inner folds.  Outer folds
    # receive one score per Directory_* patient.  Repeated runs change only the
    # split seed and are sensitivity analyses, not independent cohorts.
    outer_folds: int = 5
    inner_folds: int = 3
    repeated_cv_runs: int = 20
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    pca_variance: float = 0.95
    max_pca_components: int = 10
    auc_tolerance: float = 0.01
    bootstrap_replicates: int = 2000

    # -------------------------------------------------------------------------
    # CUSTOM METHOD 1: FOLD-LOCAL NUISANCE PROJECTION
    # -------------------------------------------------------------------------
    # Ridge ``alpha`` controls how aggressively cardiac dimensions predictable
    # from the nuisance subspace are removed.  The nuisance projection itself is
    # label-independent; scaling/PCA and ridge coefficients remain fold-local.
    residual_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    max_nuisance_components: int = 5
    nuisance_random_projection_dim: int = 16
    nuisance_random_projection_seed: int = 91_731

    # -------------------------------------------------------------------------
    # CUSTOM METHOD 2: WORST-CASE LINEAR CONFOUNDER GUARD
    # -------------------------------------------------------------------------
    # ``guard_lambda_grid`` controls the adversarial covariance penalty and
    # ``guard_l2_grid`` controls ordinary weight shrinkage.  Optimization uses
    # explicit gradients with backtracking line search.
    guard_lambda_grid: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
    guard_l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0)
    guard_learning_rate: float = 0.20
    guard_max_iterations: int = 1500
    guard_tolerance: float = 1e-7

    # -------------------------------------------------------------------------
    # CUSTOM METHOD 3: CONSERVATIVE BAYESIAN-STYLE SERIES FUSION
    # -------------------------------------------------------------------------
    # ``tau`` sets how rapidly reliability n/(n+tau) saturates with series length.
    # It is selected only inside inner CV.
    bayes_tau_grid: tuple[float, ...] = (5.0, 20.0, 50.0)

    # -------------------------------------------------------------------------
    # CACHE AND DUPLICATE AUDITS
    # -------------------------------------------------------------------------
    # The cache stores one mean embedding per series proxy.  Exact cross-patient
    # pixel duplicates can stop evaluation; pHash matches remain review candidates.
    force_rebuild_cache: bool = False
    fail_on_cross_patient_exact_duplicate: bool = True
    run_perceptual_hash_audit: bool = True
    phash_hamming_radius: int = 3


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# File extensions accepted during discovery.  The public release is JPEG-based,
# but allowing common raster formats makes local testing easier without changing
# the patient/series contract.

FEATURE_MODES = (
    "cardiac",
    "support_only",
    "shuffled_intensity",
    "outside_support",
)
# The four views are deliberately small and mutually interpretable.  They are
# encoded by the same frozen network and aggregated by the same series/patient
# rules, so comparisons focus on visible information rather than model capacity.

METADATA_NAMES = (
    "native_height",
    "native_width",
    "native_aspect_ratio",
    "file_bytes_per_pixel",
    "crop_fraction",
    "content_fraction",
    "monai_valid",
    "support_fraction",
)
# Metadata columns are averaged within series and patients, then appended to the
# nuisance matrix.  They never enter the ordinary baseline classifier directly.

EXPERIMENTS = (
    "BASELINE_A17",
    "CONTROL_SUPPORT_ONLY",
    "CONTROL_SHUFFLED_INTENSITY",
    "CONTROL_OUTSIDE_SUPPORT",
    "CUSTOM_NUISANCE_PROJECTION",
    "CUSTOM_MINIMAX_CONFOUNDER_GUARD",
    "CUSTOM_BAYESIAN_SERIES_FUSION",
)


# Experiment order is fixed before evaluation.  It also determines presentation
# order before the final table is re-ranked by repeated-CV median AUROC.


@dataclass(frozen=True)
class SliceRecord:
    """Immutable index entry for one decoded-image candidate.

    ``patient_id`` is always the validated ``Directory_*`` unit.  ``series_id``
    is only a patient-scoped folder proxy.  The class label is carried as
    metadata for later patient-level evaluation and is never passed to MONAI,
    EfficientNet, border detection, hashing, or control construction.
    """

    path: str
    label: int
    patient_id: str
    series_id: str


@dataclass
class SeriesBank:
    """Compact frozen representation with one row per series proxy.

    ``features[mode]`` has shape ``[n_series, 1280]`` and stores the exact mean
    of all frozen slice embeddings in that series.  ``metadata`` stores the
    corresponding mean export/QC variables, while ``n_slices`` preserves the
    amount of evidence available to Bayesian-style fusion.  Array rows are kept
    in a single deterministic ``series_ids`` order.
    """

    series_ids: np.ndarray
    patient_ids: np.ndarray
    labels: np.ndarray
    n_slices: np.ndarray
    features: dict[str, np.ndarray]
    metadata: np.ndarray


@dataclass
class PatientBank:
    """One-row-per-patient representation used for supervised learning.

    Each image feature mode is averaged equally across a patient's series
    proxies, so a long exported series cannot dominate merely by containing more
    JPEG frames.  ``nuisance`` combines compressed support-only, outside-support,
    and acquisition/export variables.  ``patient_to_series_rows`` allows the
    Bayesian-style method to return to series-level evidence without redefining
    a series as an independent labeled patient.
    """

    patient_ids: np.ndarray
    labels: np.ndarray
    features: dict[str, np.ndarray]
    nuisance: np.ndarray
    metadata: np.ndarray
    patient_to_series_rows: dict[str, np.ndarray]


def validate_config(config: Config) -> None:
    """Validate every predeclared numerical and structural contract early.

    This function runs before dataset scanning, network downloads, or GPU work.
    It checks odd dilation kernels, legal probability/percentile intervals,
    positive hyperparameter grids, valid fold counts, and checkpoint-digest
    formatting.  Failing early prevents an expensive partial cache from being
    mistaken for a scientifically valid run.
    """

    dataset_root = _normalize_runtime_path(config.dataset_path)
    output_root = _normalize_runtime_path(config.output_dir)
    if not _is_valid_dataset_root(dataset_root):
        raise FileNotFoundError(
            "config.dataset_path must contain both Normal/ and Sick/: "
            f"{dataset_root}"
        )
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(
            f"config.output_dir exists but is not a directory: {output_root}"
        )
    try:
        output_root.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "The output directory must not be inside the dataset root. "
            "Otherwise generated files could contaminate future discovery or "
            f"dataset fingerprints: {output_root}"
        )
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    for name, value in (
            ("monai_dilation_kernel", config.monai_dilation_kernel),
            ("a17_extra_dilation_kernel", config.a17_extra_dilation_kernel),
    ):
        if value <= 0 or value % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer.")
    if not 0.0 < config.fallback_square_fraction <= 1.0:
        raise ValueError("fallback_square_fraction must lie in (0, 1].")
    if not 0.0 <= config.lower_percentile < config.upper_percentile <= 100.0:
        raise ValueError("Invalid robust-normalization percentile interval.")
    if config.histogram_bins < 2 or config.minimum_visible_pixels < 1:
        raise ValueError("Histogram bins and visible-pixel minimum must be positive.")
    if config.minimum_dynamic_range <= 0.0:
        raise ValueError("minimum_dynamic_range must be positive.")
    if config.outer_folds < 2 or config.inner_folds < 2:
        raise ValueError("Outer and inner CV need at least two folds.")
    if config.repeated_cv_runs < 1:
        raise ValueError("repeated_cv_runs must be at least one.")
    if not config.c_grid or any(value <= 0 for value in config.c_grid):
        raise ValueError("Every Logistic Regression C must be positive.")
    if not config.residual_alpha_grid or any(
            value <= 0 for value in config.residual_alpha_grid
    ):
        raise ValueError("Every residualization alpha must be positive.")
    if not config.guard_lambda_grid or any(
            value < 0 for value in config.guard_lambda_grid
    ):
        raise ValueError("Confounder penalties cannot be negative.")
    if not config.guard_l2_grid or any(value <= 0 for value in config.guard_l2_grid):
        raise ValueError("Every custom-model L2 value must be positive.")
    if not config.bayes_tau_grid or any(value <= 0 for value in config.bayes_tau_grid):
        raise ValueError("Every Bayesian reliability tau must be positive.")
    if config.nuisance_random_projection_dim < 1:
        raise ValueError("nuisance_random_projection_dim must be positive.")
    if len(config.monai_sha256) != 64:
        raise ValueError("monai_sha256 must contain 64 hexadecimal characters.")


# =============================================================================
# 2. GENERAL UTILITIES
# =============================================================================


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    The seed controls fold shuffling, bootstrap resampling, fixed random nuisance
    projections, and deterministic initialization of custom optimization.  The
    function deliberately does not promise bit-identical results across all GPU,
    CUDA, driver, and library combinations; software versions are recorded so
    residual numerical variation can be audited.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    """Write a stable, human-readable JSON artifact.

    Parent folders are created automatically.  Keys are sorted and indentation
    is fixed so configuration and result changes are easy to inspect with a
    version-control diff.  Only JSON-serializable values should reach this helper.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Evaluate the logistic function without exponential overflow.

    Positive and negative inputs use algebraically equivalent branches.  This
    avoids evaluating ``exp(-x)`` for very negative ``x`` or ``exp(x)`` for very
    positive ``x`` and is used by the custom optimizer and Bayesian fusion.
    """

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to log-odds after fixed numerical clipping.

    Clipping to ``[eps, 1-eps]`` prevents infinite evidence when a linear model
    produces a score numerically equal to zero or one.  The clipping constant is
    numerical protection, not a learned calibration parameter.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file using bounded-memory streaming.

    The helper verifies the pinned MONAI artifact and records source integrity.
    Reading in one-megabyte chunks avoids loading a model checkpoint entirely
    into RAM.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_fingerprint(config: Config, records: Sequence[SliceRecord]) -> str:
    """Create a deterministic cache identity from feature-affecting settings.

    The fingerprint includes preprocessing, segmentation, encoder, batch, and
    hardware-relevant feature settings, together with each relative file path,
    size, and modification time.  Classifier grids and repeated-CV settings are
    excluded because they do not change frozen embeddings.  Exact decoded-pixel
    hashes are still calculated during extraction for the scientific duplicate
    audit.

    This is a practical cache invalidation contract, not a cryptographic hash of
    every dataset byte before extraction.
    """

    root = Path(config.dataset_path).resolve()
    feature_settings = {
        key: value
        for key, value in asdict(config).items()
        if key
           not in {
               "output_dir",
               "dataset_path_source",
               "output_dir_source",
               "runtime_environment",
               "force_rebuild_cache",
               "outer_folds",
               "inner_folds",
               "repeated_cv_runs",
               "c_grid",
               "pca_variance",
               "max_pca_components",
               "auc_tolerance",
               "bootstrap_replicates",
               "residual_alpha_grid",
               "max_nuisance_components",
               "guard_lambda_grid",
               "guard_l2_grid",
               "guard_learning_rate",
               "guard_max_iterations",
               "guard_tolerance",
               "bayes_tau_grid",
           }
    }
    digest = hashlib.sha256(
        json.dumps(feature_settings, sort_keys=True).encode("utf-8")
    )
    for record in records:
        path = Path(record.path)
        stat = path.stat()
        relative = str(path.resolve().relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()[:16]


# =============================================================================
# 3. DATASET DISCOVERY
# =============================================================================


def discover_records(dataset_path: str) -> list[SliceRecord]:
    """Build the complete image manifest without decoding pixels.

    Hard invariants:
    * ``Directory_*`` is the patient unit;
    * every descendant image remains assigned to that patient;
    * the immediate parent path becomes a patient-scoped series proxy;
    * one patient may not occur under both labels;
    * both classes must be present.

    The deterministic final sort fixes the row order used by caching and audit
    operations.
    """

    root = Path(dataset_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[SliceRecord] = []
    patient_to_label: dict[str, int] = {}
    for class_name, label in (("Normal", 0), ("Sick", 1)):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        patient_dirs = sorted(
            path
            for path in class_dir.glob("Directory_*")
            if path.is_dir()
        )
        if not patient_dirs:
            raise RuntimeError(f"No Directory_* folders found under {class_dir}")

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            previous = patient_to_label.get(patient_id)
            if previous is not None and previous != label:
                raise RuntimeError(
                    f"Patient {patient_id} appears under both class labels."
                )
            patient_to_label[patient_id] = label

            image_paths = sorted(
                path
                for path in patient_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_paths:
                raise RuntimeError(f"Patient {patient_id} has no image files.")

            for image_path in image_paths:
                relative_parent = image_path.parent.relative_to(patient_dir)
                series_suffix = str(relative_parent).replace(os.sep, "/")
                if series_suffix == ".":
                    series_suffix = "ROOT"
                series_id = f"{patient_id}/{series_suffix}"
                records.append(
                    SliceRecord(
                        path=str(image_path),
                        label=label,
                        patient_id=patient_id,
                        series_id=series_id,
                    )
                )

    records.sort(key=lambda row: (row.patient_id, row.series_id, row.path))
    labels = [patient_to_label[patient] for patient in sorted(patient_to_label)]
    if len(set(labels)) != 2:
        raise RuntimeError("Both Normal and Sick patients are required.")

    series_count = len({row.series_id for row in records})
    print(
        "[DATA] "
        f"patients={len(patient_to_label)}, "
        f"Normal={sum(v == 0 for v in patient_to_label.values())}, "
        f"Sick={sum(v == 1 for v in patient_to_label.values())}, "
        f"images={len(records)}, series_proxies={series_count}",
        flush=True,
    )
    return records


# =============================================================================
# 4. LABEL-BLIND IMAGE STANDARDIZATION
# =============================================================================


def _line_is_dark_and_uniform(line: np.ndarray, config: Config) -> bool:
    """Test whether one edge row or column resembles removable padding.

    A line must simultaneously have low mean intensity, low standard deviation,
    and a high fraction of pixels below a dark-value threshold.  The rule is
    fixed and label-blind; it never reads patient class, MONAI output, or model
    score.
    """
    line = line.astype(np.float32)
    return bool(
        float(line.mean()) <= config.dark_line_max_mean
        and float(line.std()) <= config.dark_line_max_std
        and float(np.mean(line <= config.dark_pixel_max_value))
        >= config.dark_pixel_min_fraction
    )


def _count_dark_edge_lines(
        image: np.ndarray,
        axis: Literal[0, 1],
        from_end: bool,
        config: Config,
) -> int:
    """Count consecutive padding-like lines from one selected image edge.

    Counting stops at the first non-padding-like line and is capped by a fixed
    fraction of the native dimension.  A run shorter than ``min_padding_run`` is
    ignored, reducing the risk that isolated dark anatomy triggers a crop.
    """

    length = image.shape[axis]
    maximum = int(math.floor(length * config.max_crop_fraction_per_side))
    count = 0
    indices = range(length - 1, -1, -1) if from_end else range(length)
    for index in indices:
        line = image[index, :] if axis == 0 else image[:, index]
        if not _line_is_dark_and_uniform(line, config):
            break
        count += 1
        if count >= maximum:
            break
    return count if count >= config.min_padding_run else 0


def remove_conservative_dark_border(
        image: np.ndarray, config: Config
) -> tuple[np.ndarray, float]:
    """Remove only safe consecutive dark and nearly uniform edge lines.

    Candidate crop widths are detected independently on all four sides.  A
    second safety check rejects the crop if too little native height or width
    would remain.  The returned fraction is recorded as nuisance/QC metadata so
    the project can test whether preprocessing geometry itself predicts class.
    """

    height, width = image.shape
    top = _count_dark_edge_lines(image, axis=0, from_end=False, config=config)
    bottom = _count_dark_edge_lines(image, axis=0, from_end=True, config=config)
    left = _count_dark_edge_lines(image, axis=1, from_end=False, config=config)
    right = _count_dark_edge_lines(image, axis=1, from_end=True, config=config)

    retained_height = height - top - bottom
    retained_width = width - left - right
    if (
            retained_height < config.min_retained_fraction * height
            or retained_width < config.min_retained_fraction * width
    ):
        top = bottom = left = right = 0
        retained_height, retained_width = height, width

    cropped = image[top: height - bottom, left: width - right]
    crop_fraction = 1.0 - (cropped.size / float(image.size))
    return cropped, float(crop_fraction)


def resize_to_fixed_canvas(
        image: np.ndarray,
        content_long_side: int,
        canvas_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize retained content with preserved aspect ratio into one square canvas.

    The longest retained side becomes ``content_long_side`` and the result is
    centered in a ``canvas_size`` square.  A second binary canvas marks true
    resized content versus zero padding.  Raw and MONAI intensity views call
    this same geometric rule and must produce identical content masks.
    """

    height, width = image.shape
    scale = content_long_side / float(max(height, width))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    content = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    top = (canvas_size - new_height) // 2
    left = (canvas_size - new_width) // 2
    canvas[top: top + new_height, left: left + new_width] = resized
    content[top: top + new_height, left: left + new_width] = 1.0
    return canvas, content


def prepare_slice(
        path: str, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Decode one image and construct all label-independent slice inputs.

    Processing order:
    1. decode grayscale pixels;
    2. remove only conservative dark border runs;
    3. create raw uint8/255 and MONAI min-max intensity views;
    4. place both on identical fixed canvases;
    5. compute export/QC metadata;
    6. compute exact decoded-pixel and perceptual hashes.

    The returned label-free tensors are later batched by ``SliceDataset``.  An
    unreadable file raises an error instead of being silently omitted.
    """

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode: {path}")
    native_height, native_width = image.shape
    cropped, crop_fraction = remove_conservative_dark_border(image, config)

    raw_01 = cropped.astype(np.float32) / 255.0
    minimum = float(cropped.min())
    maximum = float(cropped.max())
    if maximum > minimum:
        monai_01 = (cropped.astype(np.float32) - minimum) / (maximum - minimum)
    else:
        monai_01 = np.zeros_like(cropped, dtype=np.float32)

    raw_canvas, content_mask = resize_to_fixed_canvas(
        raw_01,
        config.standardized_content_long_side,
        config.monai_input_size,
    )
    monai_canvas, second_content_mask = resize_to_fixed_canvas(
        monai_01,
        config.standardized_content_long_side,
        config.monai_input_size,
    )
    if not np.array_equal(content_mask, second_content_mask):
        raise RuntimeError("Raw and MONAI canvas geometries are not aligned.")

    file_bytes = Path(path).stat().st_size
    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    metadata = np.asarray(
        [
            float(native_height),
            float(native_width),
            float(native_width / max(1, native_height)),
            float(file_bytes / max(1, image.size)),
            float(crop_fraction),
            float(content_mask.mean()),
        ],
        dtype=np.float32,
    )
    return (
        raw_canvas,
        monai_canvas,
        content_mask,
        metadata,
        pixel_hash,
        perceptual_hash_64(image),
    )


def perceptual_hash_64(image: np.ndarray) -> int:
    """Compute a 64-bit DCT perceptual hash for screening near duplicates.

    The image is reduced to a low-frequency 8x8 DCT block and thresholded around
    its non-DC median.  Hamming proximity is only a candidate signal; visually
    similar anatomy, blank frames, or repeated acquisition patterns can collide.
    Exact duplication decisions therefore require decoded-pixel hashes or manual
    review.
    """

    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))[:8, :8]
    values = dct.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return int(result)


class SliceDataset(Dataset):
    """PyTorch dataset that decodes and standardizes one slice per request.

    The dataset returns aligned tensors and immutable identifiers.  It does not
    perform augmentation and does not expose labels to frozen image models.  A
    deterministic DataLoader order allows series sums, hashes, and cache rows to
    be reconstructed exactly.
    """

    def __init__(self, records: Sequence[SliceRecord], config: Config):
        """Store a deterministic record list and immutable run configuration.

        Records are copied into a list so later iteration is unaffected by an
        external generator or mutation of the caller's sequence.
        """
        self.records = list(records)
        self.config = config

    def __len__(self) -> int:
        """Return the number of image rows in the discovered manifest."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode one record and return aligned tensors plus audit metadata.

        The dictionary format lets the DataLoader collate numeric tensors while
        preserving patient IDs, series IDs, exact hashes, and pHash strings.  The
        patient label remains metadata and is not used to construct any image view.
        """
        record = self.records[index]
        raw, monai, content, metadata, pixel_hash, phash = prepare_slice(
            record.path, self.config
        )
        return {
            "raw": torch.from_numpy(raw).unsqueeze(0),
            "monai": torch.from_numpy(monai).unsqueeze(0),
            "content": torch.from_numpy(content).unsqueeze(0),
            "metadata": torch.from_numpy(metadata),
            "label": torch.tensor(record.label, dtype=torch.long),
            "patient_id": record.patient_id,
            "series_id": record.series_id,
            "pixel_hash": pixel_hash,
            "phash": f"{phash:016x}",
        }


# =============================================================================
# 5. FROZEN MONAI AND EFFICIENTNET MODELS
# =============================================================================


def load_monai_segmenter(config: Config, device: torch.device) -> torch.jit.ScriptModule:
    """Download, verify, load, and sanity-check the pinned MONAI model.

    The Hugging Face repository is pinned to an immutable revision.  SHA-256 is
    checked before ``torch.jit.load``.  A zero-input inference verifies that the
    runtime output can be normalized to the expected four-channel shape.  This
    preferred TorchScript path avoids importing or rebuilding the full MONAI
    network during ordinary runs.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install huggingface_hub before running feature extraction."
        ) from error

    model_path = Path(
        hf_hub_download(
            repo_id=config.monai_repo_id,
            filename=config.monai_filename,
            revision=config.monai_revision,
        )
    )
    actual_digest = sha256_file(model_path)
    if actual_digest.lower() != config.monai_sha256.lower():
        raise RuntimeError(
            "MONAI TorchScript SHA-256 mismatch. "
            f"Expected {config.monai_sha256}, received {actual_digest}."
        )

    model = torch.jit.load(str(model_path), map_location=device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, config.monai_input_size, config.monai_input_size, device=device))
    output = normalize_monai_output(output)
    if output.shape != (1, 4, config.monai_input_size, config.monai_input_size):
        raise RuntimeError(f"Unexpected MONAI output shape: {tuple(output.shape)}")
    return model


def normalize_monai_output(output: Any) -> torch.Tensor:
    """Extract a tensor from the supported TorchScript output containers.

    Different export/runtime combinations may return a tensor directly, a
    one-element sequence, or a dictionary containing ``pred``, ``logits``, or
    ``output``.  Recursive normalization keeps the downstream segmentation code
    independent of that shallow container choice while rejecting unknown types.
    """

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return normalize_monai_output(output[0])
    if isinstance(output, dict):
        for key in ("pred", "logits", "output"):
            if key in output:
                return normalize_monai_output(output[key])
    raise TypeError(f"Unsupported MONAI output type: {type(output).__name__}")


class EfficientNetEncoder(nn.Module):
    """Frozen ImageNet EfficientNet-B0 used only as a feature function.

    The 1,000-class ImageNet classifier is replaced by ``Identity``, exposing a
    1,280-dimensional descriptor.  The network is not fine-tuned on the 30
    patient labels, which reduces the risk of severe high-capacity overfitting.
    """

    def __init__(self) -> None:
        """Instantiate the explicit IMAGENET1K_V1 checkpoint and remove its head."""
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        network = models.efficientnet_b0(weights=weights)
        network.classifier = nn.Identity()
        self.network = network.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one frozen 1,280-dimensional descriptor per normalized image."""
        return self.network(images)


def efficientnet_normalize(images: torch.Tensor, output_size: int) -> torch.Tensor:
    """Convert aligned grayscale canvases to EfficientNet input tensors.

    One channel is repeated to RGB, the complete square canvas is resized to the
    encoder resolution, and the ImageNet mean/std associated with the pinned
    checkpoint are applied.  Resizing the whole canvas preserves alignment with
    the support mask rather than applying an undocumented center crop.
    """

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(
        images,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# =============================================================================
# 6. EXACT A17 SUPPORT AND MATCHED CONTROLS
# =============================================================================


def fixed_central_square_mask(
        batch: int,
        height: int,
        width: int,
        fraction: float,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    """Create the deterministic fallback support used when MONAI is implausible.

    The square size is a fixed fraction of the canvas and is independent of
    label, fold, model score, or predicted mask area.  It prevents an invalid
    segmentation from exposing the entire image while retaining a reproducible
    central field of view.
    """
    side = max(2, min(height, width, int(round(min(height, width) * fraction))))
    top = (height - side) // 2
    left = (width - side) // 2
    mask = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
    mask[:, :, top: top + side, left: left + side] = 1.0
    return mask


def infer_a17_support(
        monai_model: nn.Module,
        monai_images: torch.Tensor,
        content_mask: torch.Tensor,
        config: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer the exact binary support shared by candidate and matched controls.

    The function sums the three non-background MONAI channels, applies fixed
    area/peak plausibility gates, performs the two declared dilations, and uses a
    fixed central fallback on invalid slices.  The final support is intersected
    with the content mask so pipeline-added canvas padding is never visible.

    Returning ``support``, ``valid``, and ``heart_probability`` separately makes
    the scientific contract auditable without recomputing segmentation.
    """

    logits = normalize_monai_output(monai_model(monai_images))
    probabilities = torch.softmax(logits.float(), dim=1)
    heart_probability = probabilities[:, 1:4].sum(dim=1, keepdim=True)

    initial_hard = heart_probability >= config.monai_hard_threshold
    area_ratio = initial_hard.float().mean(dim=(1, 2, 3))
    peak = heart_probability.amax(dim=(1, 2, 3))
    valid = (
            (area_ratio >= config.monai_min_area_ratio)
            & (area_ratio <= config.monai_max_area_ratio)
            & (peak >= config.monai_min_peak_probability)
    )

    dilated = F.max_pool2d(
        initial_hard.float(),
        kernel_size=config.monai_dilation_kernel,
        stride=1,
        padding=config.monai_dilation_kernel // 2,
    )
    dilated = F.max_pool2d(
        dilated,
        kernel_size=config.a17_extra_dilation_kernel,
        stride=1,
        padding=config.a17_extra_dilation_kernel // 2,
    )

    fallback = fixed_central_square_mask(
        batch=monai_images.shape[0],
        height=monai_images.shape[-2],
        width=monai_images.shape[-1],
        fraction=config.fallback_square_fraction,
        device=monai_images.device,
        dtype=monai_images.dtype,
    )
    support = torch.where(valid[:, None, None, None], dilated, fallback)
    support = (support > 0.5).to(monai_images.dtype)
    support = support * (content_mask > 0.5).to(monai_images.dtype)
    return support, valid, heart_probability


def robust_scale_visible_region(
        images: torch.Tensor,
        masks: torch.Tensor,
        config: Config,
) -> torch.Tensor:
    """Robustly normalize each image using only its final visible pixels.

    A batched fixed-bin histogram estimates the declared lower and upper
    percentiles.  The transform excludes hidden pixels and canvas padding, then
    zeroes everything outside the mask.  Consequently, cardiac pixels cannot
    determine the scale of the outside-support control and outside pixels cannot
    determine the scale of the cardiac candidate.

    The minimum visible-pixel and dynamic-range safeguards prevent unstable
    amplification of nearly empty or nearly constant regions.
    """

    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must have shape [B,1,H,W]")
    if masks.shape != images.shape:
        raise ValueError("masks must align exactly with images")

    batch = images.shape[0]
    bins = config.histogram_bins
    visible = masks > 0.5
    flat_mask = visible[:, 0].reshape(batch, -1)
    counts = flat_mask.sum(dim=1)
    values = images[:, 0].float().clamp(0.0, 1.0)
    indices = torch.round(values * (bins - 1)).long().reshape(batch, -1)

    histogram = torch.zeros(batch, bins, device=images.device, dtype=torch.float32)
    histogram.scatter_add_(1, indices, flat_mask.float())
    cumulative = histogram.cumsum(dim=1)

    safe_counts = counts.clamp_min(1)
    lower_rank = (
            torch.floor((config.lower_percentile / 100.0) * (safe_counts - 1).float()).long()
            + 1
    )
    upper_rank = (
            torch.floor((config.upper_percentile / 100.0) * (safe_counts - 1).float()).long()
            + 1
    )
    lower_bin = (cumulative >= lower_rank[:, None]).long().argmax(dim=1)
    upper_bin = (cumulative >= upper_rank[:, None]).long().argmax(dim=1)
    lower = lower_bin.float() / float(bins - 1)
    upper = upper_bin.float() / float(bins - 1)

    valid = (
            (counts >= config.minimum_visible_pixels)
            & torch.isfinite(lower)
            & torch.isfinite(upper)
            & (upper > lower)
    )
    denominator = (upper - lower).clamp_min(config.minimum_dynamic_range)
    scaled = (images.float() - lower[:, None, None, None]) / denominator[:, None, None, None]
    scaled = scaled.clamp(0.0, 1.0) * visible.float()
    scaled = scaled * valid[:, None, None, None].float()
    return scaled.to(images.dtype)


def coprime_affine_parameters(pixel_hash: str, n: int) -> tuple[int, int]:
    """Derive a deterministic bijection over visible pixel ranks.

    Source rank is ``(a*k+b) mod n``.  Requiring ``gcd(a,n)=1`` guarantees a full
    permutation, so every visible intensity is used exactly once.  Parameters
    depend only on the decoded-pixel hash, never on label, patient, fold, or
    classifier output.
    """

    if n <= 1:
        return 1, 0
    candidate = 2 + (int(pixel_hash[:16], 16) % max(1, n - 2))
    candidate %= n
    candidate = max(candidate, 1)
    start = candidate
    while math.gcd(candidate, n) != 1:
        candidate = 1 if candidate + 1 >= n else candidate + 1
        if candidate == start:
            raise RuntimeError("Could not find a coprime affine multiplier.")
    offset = int(pixel_hash[16:32], 16) % n
    if candidate == 1 and offset == 0:
        offset = 1
    return candidate, offset


def shuffle_inside_support(
        cardiac_images: torch.Tensor,
        support: torch.Tensor,
        pixel_hashes: Sequence[str],
) -> torch.Tensor:
    """Destroy spatial arrangement while preserving support and intensity multiset.

    For every slice, the normalized cardiac values are permuted bijectively
    among the exact support coordinates.  The support boundary, number of
    visible pixels, histogram, mean, variance, quantiles, and extrema remain
    unchanged; only correspondence between intensity and anatomical location is
    destroyed.  Pixels outside the support remain zero.
    """

    output = torch.zeros_like(cardiac_images)
    for index, pixel_hash in enumerate(pixel_hashes):
        positions = torch.nonzero(support[index, 0].reshape(-1) > 0.5).reshape(-1)
        n_values = int(positions.numel())
        if n_values == 0:
            continue
        source = cardiac_images[index, 0].reshape(-1)[positions]
        multiplier, offset = coprime_affine_parameters(pixel_hash, n_values)
        destination = torch.arange(n_values, device=images_device(cardiac_images))
        source_rank = (destination * multiplier + offset) % n_values
        shuffled = source[source_rank]
        for channel in range(3):
            output[index, channel].reshape(-1)[positions] = shuffled
    return output


def images_device(images: torch.Tensor) -> torch.device:
    """Return the tensor device for readable device-local index creation."""

    return images.device


def make_four_views(
        raw_images: torch.Tensor,
        support: torch.Tensor,
        content_mask: torch.Tensor,
        pixel_hashes: Sequence[str],
        config: Config,
) -> dict[str, torch.Tensor]:
    """Construct the candidate and three controls from one exact support.

    ``cardiac`` and ``outside_support`` are independently region-normalized.
    ``support_only`` removes MRI intensity while preserving exact geometry.
    ``shuffled_intensity`` preserves exact geometry and intensity distribution
    while destroying spatial arrangement.  Reusing the same support prevents
    candidate/control drift.
    """

    cardiac_gray = robust_scale_visible_region(raw_images, support, config)
    outside_mask = ((content_mask > 0.5) & (support <= 0.5)).float()
    outside_gray = robust_scale_visible_region(raw_images, outside_mask, config)

    cardiac_rgb = cardiac_gray.repeat(1, 3, 1, 1)
    support_rgb = support.repeat(1, 3, 1, 1)
    shuffled_rgb = shuffle_inside_support(cardiac_rgb, support, pixel_hashes)
    outside_rgb = outside_gray.repeat(1, 3, 1, 1)
    return {
        "cardiac": cardiac_rgb,
        "support_only": support_rgb,
        "shuffled_intensity": shuffled_rgb,
        "outside_support": outside_rgb,
    }


# =============================================================================
# 7. COMPACT SERIES-LEVEL FEATURE BANK
# =============================================================================


def extract_series_bank(config: Config, records: Sequence[SliceRecord]) -> SeriesBank:
    """Run frozen segmentation/encoding once and accumulate exact series means.

    On a cache miss this is the expensive stage.  It:
    * creates a deterministic DataLoader;
    * loads verified MONAI and frozen EfficientNet models;
    * builds the four matched views per batch;
    * encodes all views in one concatenated call;
    * accumulates float64 sums by series and stores float32 means;
    * accumulates metadata and slice counts;
    * computes exact/pHash duplicate audits;
    * writes a compact NPZ cache and JSON manifest.

    Labels are carried only to verify consistent patient/series membership and
    are never supplied to either neural network.
    """

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = configuration_fingerprint(config, records)
    cache_path = output_dir / f"series_feature_bank_{fingerprint}.npz"
    manifest_path = output_dir / f"series_feature_bank_{fingerprint}.json"

    if cache_path.is_file() and manifest_path.is_file() and not config.force_rebuild_cache:
        print(f"[CACHE] Loading {cache_path}", flush=True)
        return load_series_bank(cache_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] device={device}", flush=True)
    monai_model = load_monai_segmenter(config, device)
    encoder = EfficientNetEncoder().to(device).eval()

    dataset = SliceDataset(records, config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feature_sums: dict[str, dict[str, np.ndarray]] = {
        mode: {} for mode in FEATURE_MODES
    }
    metadata_sums: dict[str, np.ndarray] = {}
    series_counts: dict[str, int] = defaultdict(int)
    series_patient: dict[str, str] = {}
    series_label: dict[str, int] = {}

    hash_to_patients: dict[str, set[str]] = defaultdict(set)
    phash_to_patients: dict[int, set[str]] = defaultdict(set)

    autocast_enabled = bool(config.use_amp and device.type == "cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting four views", unit="batch"):
            raw = batch["raw"].to(device, non_blocking=True)
            monai = batch["monai"].to(device, non_blocking=True)
            content = batch["content"].to(device, non_blocking=True)
            pixel_hashes = list(batch["pixel_hash"])

            support, valid, _ = infer_a17_support(
                monai_model, monai, content, config
            )
            views = make_four_views(raw, support, content, pixel_hashes, config)
            normalized = [
                efficientnet_normalize(views[mode], config.encoder_input_size)
                for mode in FEATURE_MODES
            ]
            stacked = torch.cat(normalized, dim=0)
            with torch.autocast(
                    device_type=device.type,
                    enabled=autocast_enabled,
                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ):
                embeddings = encoder(stacked).float()
            split_embeddings = embeddings.split(raw.shape[0], dim=0)

            support_fraction = support.mean(dim=(1, 2, 3)).cpu().numpy()
            valid_np = valid.float().cpu().numpy()
            base_metadata = batch["metadata"].cpu().numpy().astype(np.float64)
            complete_metadata = np.column_stack(
                [base_metadata, valid_np, support_fraction]
            )

            labels = batch["label"].cpu().numpy().astype(np.int64)
            patient_ids = list(batch["patient_id"])
            series_ids = list(batch["series_id"])
            phashes = [int(value, 16) for value in batch["phash"]]

            for row_index, (patient_id, series_id, label) in enumerate(
                    zip(patient_ids, series_ids, labels)
            ):
                patient_id = str(patient_id)
                series_id = str(series_id)
                label = int(label)
                previous_patient = series_patient.get(series_id)
                previous_label = series_label.get(series_id)
                if previous_patient is not None and previous_patient != patient_id:
                    raise RuntimeError(f"Series ID collision: {series_id}")
                if previous_label is not None and previous_label != label:
                    raise RuntimeError(f"Series label conflict: {series_id}")
                series_patient[series_id] = patient_id
                series_label[series_id] = label
                series_counts[series_id] += 1

                if series_id not in metadata_sums:
                    metadata_sums[series_id] = np.zeros(
                        len(METADATA_NAMES), dtype=np.float64
                    )
                metadata_sums[series_id] += complete_metadata[row_index]

                for mode, tensor in zip(FEATURE_MODES, split_embeddings):
                    vector = tensor[row_index].detach().cpu().numpy().astype(np.float64)
                    if series_id not in feature_sums[mode]:
                        feature_sums[mode][series_id] = np.zeros(
                            config.embedding_dim, dtype=np.float64
                        )
                    feature_sums[mode][series_id] += vector

                hash_to_patients[pixel_hashes[row_index]].add(patient_id)
                phash_to_patients[int(phashes[row_index])].add(patient_id)

    series_ids = np.asarray(sorted(series_counts))
    patient_ids = np.asarray([series_patient[sid] for sid in series_ids])
    labels = np.asarray([series_label[sid] for sid in series_ids], dtype=np.int64)
    counts = np.asarray([series_counts[sid] for sid in series_ids], dtype=np.int64)
    metadata = np.stack(
        [metadata_sums[sid] / series_counts[sid] for sid in series_ids]
    ).astype(np.float32)
    features = {
        mode: np.stack(
            [feature_sums[mode][sid] / series_counts[sid] for sid in series_ids]
        ).astype(np.float32)
        for mode in FEATURE_MODES
    }

    cross_patient_exact = {
        digest: sorted(patients)
        for digest, patients in hash_to_patients.items()
        if len(patients) > 1
    }

    audit = {
        "decoded_slices": len(records),
        "series_rows": int(len(series_ids)),
        "exact_hash_groups": int(len(hash_to_patients)),
        "cross_patient_exact_groups": cross_patient_exact,
        "perceptual_hash_unique_values": int(len(phash_to_patients)),
    }
    if config.run_perceptual_hash_audit:
        patient_label_map = {
            patient: int(label)
            for patient, label in zip(patient_ids.tolist(), labels.tolist())
        }
        candidates = phash_patient_pairs(
            phash_to_patients,
            config.phash_hamming_radius,
            patient_label_map,
        )
        audit["perceptual_patient_pair_candidates"] = candidates
        audit["perceptual_candidate_pair_count"] = len(candidates)
        audit["perceptual_cross_label_pair_count"] = sum(
            int(row["cross_label"]) for row in candidates
        )
    write_json(output_dir / "duplicate_audit.json", audit)
    if cross_patient_exact and config.fail_on_cross_patient_exact_duplicate:
        raise RuntimeError(
            "Confirmed decoded-pixel duplicates cross patients. "
            "Inspect duplicate_audit.json before evaluation."
        )

    np.savez(
        cache_path,
        series_ids=series_ids,
        patient_ids=patient_ids,
        labels=labels,
        n_slices=counts,
        metadata=metadata,
        **{f"features_{mode}": features[mode] for mode in FEATURE_MODES},
    )
    write_json(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "config": asdict(config),
            "feature_modes": list(FEATURE_MODES),
            "metadata_names": list(METADATA_NAMES),
            "elapsed_seconds": time.perf_counter() - started,
            "cache_path": str(cache_path),
        },
    )
    print(
        f"[CACHE] Wrote {len(series_ids)} series rows in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return SeriesBank(series_ids, patient_ids, labels, counts, features, metadata)


def load_series_bank(path: Path) -> SeriesBank:
    """Reconstruct a ``SeriesBank`` from the compact NPZ cache.

    The loader restores fixed dtypes and all four declared modes.  The JSON
    manifest and configuration fingerprint are checked by the caller before this
    function is used.
    """
    data = np.load(path, allow_pickle=False)
    return SeriesBank(
        series_ids=data["series_ids"],
        patient_ids=data["patient_ids"],
        labels=data["labels"].astype(np.int64),
        n_slices=data["n_slices"].astype(np.int64),
        features={mode: data[f"features_{mode}"].astype(np.float32) for mode in FEATURE_MODES},
        metadata=data["metadata"].astype(np.float32),
    )


class BKTree:
    """Minimal Burkhard-Keller tree for exact Hamming-radius search.

    Hamming distance over 64-bit pHashes is a metric, so BK-tree triangle-inequality
    bounds can prune most comparisons without truncating the search.  The tree
    returns screening candidates only; it does not declare clinical images to be
    duplicates.
    """

    def __init__(self) -> None:
        """Initialize an empty metric tree."""
        self.root: tuple[int, dict[int, Any]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        """Return exact 64-bit Hamming distance using XOR bit count."""
        return int((left ^ right).bit_count())

    def add(self, value: int) -> None:
        """Insert one unique hash into the metric tree without rebalancing."""
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            current, children = node
            distance = self.distance(value, current)
            if distance == 0:
                return
            if distance not in children:
                children[distance] = (value, {})
                return
            node = children[distance]

    def query(self, value: int, radius: int) -> list[int]:
        """Return every stored hash within the inclusive radius.

        Child edges outside ``distance ± radius`` cannot contain valid matches by
        the triangle inequality and are safely pruned.
        """
        if self.root is None:
            return []
        matches: list[int] = []
        stack = [self.root]
        while stack:
            current, children = stack.pop()
            distance = self.distance(value, current)
            if distance <= radius:
                matches.append(current)
            lower, upper = distance - radius, distance + radius
            stack.extend(
                child
                for edge, child in children.items()
                if lower <= edge <= upper
            )
        return matches


def phash_patient_pairs(
        phash_to_patients: dict[int, set[str]],
        radius: int,
        patient_to_label: dict[str, int],
) -> list[dict[str, Any]]:
    """Convert near-hash matches into deduplicated patient-pair candidates.

    For each patient pair, the minimum observed pHash distance is retained and a
    cross-label flag is recorded.  The result is an audit queue for manual review,
    not evidence that two patients or images are identical.
    """

    tree = BKTree()
    values = sorted(phash_to_patients)
    for value in values:
        tree.add(value)
    pairs: dict[tuple[str, str], int] = {}
    for value in values:
        for neighbor in tree.query(value, radius):
            if neighbor < value:
                continue
            distance = BKTree.distance(value, neighbor)
            for left in phash_to_patients[value]:
                for right in phash_to_patients[neighbor]:
                    if left == right:
                        continue
                    pair = tuple(sorted((left, right)))
                    pairs[pair] = min(distance, pairs.get(pair, 65))
    return [
        {
            "patient_a": left,
            "patient_b": right,
            "minimum_hamming_distance": distance,
            "cross_label": bool(
                patient_to_label[left] != patient_to_label[right]
            ),
        }
        for (left, right), distance in sorted(pairs.items())
    ]


# =============================================================================
# 8. SERIES -> PATIENT POOLING AND NUISANCE MATRIX
# =============================================================================


def build_patient_bank(
        series: SeriesBank, nuisance_projection_dim: int, nuisance_projection_seed: int
) -> PatientBank:
    """Pool series equally to patients and build the nuisance design matrix.

    Each patient feature is the unweighted mean of its series-proxy means.  The
    nuisance matrix contains fixed random projections of exact support-only and
    outside-support embeddings plus simple acquisition/export metadata and series
    statistics.  The random projection is label-independent and fixed before
    evaluation; fold-local scaling and PCA are still fitted later.

    This design makes the confounding hypothesis explicit while keeping the
    nuisance dimension appropriate for only 24 outer-training patients.
    """

    patient_ids = np.asarray(sorted(set(series.patient_ids.tolist())))
    patient_to_rows = {
        patient_id: np.flatnonzero(series.patient_ids == patient_id)
        for patient_id in patient_ids
    }
    labels = []
    patient_features = {mode: [] for mode in FEATURE_MODES}
    patient_metadata = []

    for patient_id in patient_ids:
        rows = patient_to_rows[patient_id]
        patient_labels = np.unique(series.labels[rows])
        if len(patient_labels) != 1:
            raise RuntimeError(f"Inconsistent labels for {patient_id}")
        labels.append(int(patient_labels[0]))
        for mode in FEATURE_MODES:
            patient_features[mode].append(series.features[mode][rows].mean(axis=0))
        metadata_mean = series.metadata[rows].mean(axis=0)
        metadata_augmented = np.concatenate(
            [
                metadata_mean,
                np.asarray(
                    [
                        float(len(rows)),
                        float(series.n_slices[rows].sum()),
                        float(series.n_slices[rows].mean()),
                        float(series.n_slices[rows].max()),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        patient_metadata.append(metadata_augmented)

    feature_arrays = {
        mode: np.asarray(values, dtype=np.float32)
        for mode, values in patient_features.items()
    }
    metadata_array = np.asarray(patient_metadata, dtype=np.float32)
    # A fixed Johnson-Lindenstrauss-style random projection keeps the nuisance
    # matrix compact without learning from labels or from the cohort.  Fold-local
    # scaling/PCA is still fitted later.  This is far faster than repeatedly
    # decomposing 2,560 control dimensions inside every inner fold.
    projection_dim = max(4, min(64, int(nuisance_projection_dim)))
    rng = np.random.default_rng(nuisance_projection_seed)
    scale = 1.0 / math.sqrt(projection_dim)
    support_projection = rng.normal(
        0.0, scale, size=(feature_arrays["support_only"].shape[1], projection_dim)
    ).astype(np.float32)
    outside_projection = rng.normal(
        0.0, scale, size=(feature_arrays["outside_support"].shape[1], projection_dim)
    ).astype(np.float32)
    nuisance = np.concatenate(
        [
            feature_arrays["support_only"] @ support_projection,
            feature_arrays["outside_support"] @ outside_projection,
            metadata_array,
        ],
        axis=1,
    )
    return PatientBank(
        patient_ids=patient_ids,
        labels=np.asarray(labels, dtype=np.int64),
        features=feature_arrays,
        nuisance=nuisance.astype(np.float32),
        metadata=metadata_array,
        patient_to_series_rows=patient_to_rows,
    )


# =============================================================================
# 9. FOLD-LOCAL LINEAR ALGEBRA
# =============================================================================


class CompactPCA:
    """Fold-local standardization followed by a deliberately small PCA.

    The retained dimension is the smaller of the declared variance requirement,
    maximum component cap, feature count, and training-sample constraint.  Both
    scaler and PCA are fitted only on the current training partition.
    """

    def __init__(self, variance: float, maximum_components: int):
        """Store the variance target and maximum component cap."""
        self.variance = variance
        self.maximum_components = maximum_components
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "CompactPCA":
        """Fit StandardScaler, probe explained variance, and refit the chosen PCA."""
        Xs = self.scaler.fit_transform(X)
        maximum = max(1, min(self.maximum_components, Xs.shape[0] - 2, Xs.shape[1]))
        probe = PCA(n_components=maximum, svd_solver="full", random_state=0).fit(Xs)
        cumulative = np.cumsum(probe.explained_variance_ratio_)
        required = int(np.searchsorted(cumulative, self.variance) + 1)
        components = max(1, min(maximum, required))
        self.pca = PCA(n_components=components, svd_solver="full", random_state=0).fit(Xs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the already-fitted scaler and PCA without refitting."""
        if self.pca is None:
            raise RuntimeError("CompactPCA must be fit before transform.")
        return self.pca.transform(self.scaler.transform(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on the supplied training matrix and return its compact coordinates."""
        return self.fit(X).transform(X)


class NuisanceEncoder:
    """Fold-local compression of nuisance controls into a few linear axes.

    Keeping the nuisance subspace small stabilizes residualization and the
    minimax penalty when an outer fold contains only about 24 training patients.
    """

    def __init__(self, maximum_components: int):
        """Store the maximum nuisance-axis count and create an unfitted scaler."""
        self.maximum_components = maximum_components
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, Z: np.ndarray) -> "NuisanceEncoder":
        """Fit nuisance scaling and full-SVD PCA on training patients only."""
        Zs = self.scaler.fit_transform(Z)
        components = max(1, min(self.maximum_components, Zs.shape[0] - 2, Zs.shape[1]))
        self.pca = PCA(n_components=components, svd_solver="full", random_state=0).fit(Zs)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Project nuisance variables through the fitted training-only transform."""
        if self.pca is None:
            raise RuntimeError("NuisanceEncoder must be fit before transform.")
        return self.pca.transform(self.scaler.transform(Z))


class NuisanceResidualizer:
    """Remove cardiac variation that is linearly reconstructable from nuisance.

    For centered cardiac features ``Xc`` and compressed nuisance features ``Zc``,
    solve

        B* = argmin_B ||Xc - Zc B||_F^2 + alpha ||B||_F^2

    and return ``Xc - Zc B*``.  Ridge regularization prevents the unstable exact
    projection that would arise from a tiny training cohort and many controls.
    Every mean, scaler, PCA axis, and coefficient is learned inside the current
    training fold.  The operation is a sensitivity analysis, not a proof of
    causal deconfounding.
    """

    def __init__(self, alpha: float, max_nuisance_components: int):
        """Create an unfitted ridge residualizer and fold-local nuisance encoder."""
        self.alpha = float(alpha)
        self.encoder = NuisanceEncoder(max_nuisance_components)
        self.x_mean: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None

    def fit(self, X: np.ndarray, Z: np.ndarray) -> "NuisanceResidualizer":
        """Fit nuisance axes and solve the regularized normal equations."""
        self.x_mean = np.asarray(X, dtype=np.float64).mean(axis=0)
        Zc = self.encoder.fit(Z).transform(Z).astype(np.float64)
        Xc = np.asarray(X, dtype=np.float64) - self.x_mean
        gram = Zc.T @ Zc + self.alpha * np.eye(Zc.shape[1])
        self.coefficients = np.linalg.solve(gram, Zc.T @ Xc)
        return self

    def transform(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Subtract the training-estimated nuisance component from new cardiac rows."""
        if self.x_mean is None or self.coefficients is None:
            raise RuntimeError("NuisanceResidualizer is not fitted.")
        Zc = self.encoder.transform(Z).astype(np.float64)
        return (
                np.asarray(X, dtype=np.float64)
                - self.x_mean
                - Zc @ self.coefficients
        ).astype(np.float32)


# =============================================================================
# 10. CUSTOM MINIMAX-INSPIRED CONFOUNDER-GUARDED LOGISTIC REGRESSION
# =============================================================================


class WorstCaseLinearConfounderLogistic:
    """Logistic regression with a closed-form worst linear nuisance adversary.

    With score ``s=Xw+b`` and centered nuisance axes ``Z``, the strongest squared
    covariance over all unit nuisance combinations equals

        || Z^T (s-mean(s)) / n ||_2^2.

    The optimized objective is balanced logistic loss plus this covariance
    penalty and L2 regularization.  Explicit gradients and backtracking line
    search keep the implementation interpretable and suitable for a very small
    patient cohort.  Only modeled linear score dependence is controlled.
    """

    def __init__(
            self,
            confounder_lambda: float,
            l2: float,
            learning_rate: float,
            max_iterations: int,
            tolerance: float,
    ) -> None:
        """Store optimization constants and initialize an unfitted linear model."""
        self.confounder_lambda = float(confounder_lambda)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.training_trace: list[float] = []

    def _loss_and_gradient(
            self,
            X: np.ndarray,
            Z: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray,
            bias: float,
    ) -> tuple[float, np.ndarray, float]:
        """Evaluate the full objective and its analytical gradients.

        Class-balanced weights prevent the 16/14 class split from changing the
        objective simply through frequency.  The nuisance term uses the closed-form
        maximizing covariance vector, so no separately trained adversarial network
        is required.
        """
        n = len(y)
        scores = X @ weights + bias
        probabilities = stable_sigmoid(scores)

        # Balanced observation weights keep the two patient classes comparable.
        positives = max(1, int(np.sum(y == 1)))
        negatives = max(1, int(np.sum(y == 0)))
        sample_weight = np.where(y == 1, n / (2.0 * positives), n / (2.0 * negatives))
        normalizer = float(sample_weight.sum())
        logistic_terms = np.logaddexp(0.0, scores) - y * scores
        logistic_loss = float(np.dot(sample_weight, logistic_terms) / normalizer)
        grad_scores = sample_weight * (probabilities - y) / normalizer

        # Z is centered by the fold-local nuisance encoder.  The expression below
        # is the maximizing linear nuisance direction in closed form.
        centered_scores = scores - scores.mean()
        covariance_vector = Z.T @ centered_scores / float(n)
        confounder_penalty = self.confounder_lambda * float(
            covariance_vector @ covariance_vector
        )
        grad_scores += (
                2.0
                * self.confounder_lambda
                / float(n)
                * (Z @ covariance_vector)
        )

        regularization = 0.5 * self.l2 * float(weights @ weights)
        loss = logistic_loss + confounder_penalty + regularization
        grad_w = X.T @ grad_scores + self.l2 * weights
        grad_b = float(np.sum(sample_weight * (probabilities - y) / normalizer))
        return loss, grad_w, grad_b

    def fit(self, X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> "WorstCaseLinearConfounderLogistic":
        """Optimize weights with batch gradient descent and Armijo backtracking.

        The model starts at zero, stops on small gradient norm or loss improvement,
        and records every accepted objective value.  Candidate step sizes are reduced
        until they satisfy a sufficient-decrease condition, avoiding a fixed step
        that may diverge for one fold.
        """
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = np.zeros(X.shape[1], dtype=np.float64)
        bias = 0.0
        learning_rate = self.learning_rate

        loss, grad_w, grad_b = self._loss_and_gradient(X, Z, y, weights, bias)
        self.training_trace = [loss]
        for _ in range(self.max_iterations):
            gradient_norm = math.sqrt(float(grad_w @ grad_w) + grad_b * grad_b)
            if gradient_norm < self.tolerance:
                break

            accepted = False
            trial_rate = learning_rate
            for _ in range(25):
                candidate_w = weights - trial_rate * grad_w
                candidate_b = bias - trial_rate * grad_b
                candidate_loss, candidate_grad_w, candidate_grad_b = self._loss_and_gradient(
                    X, Z, y, candidate_w, candidate_b
                )
                if candidate_loss <= loss - 1e-4 * trial_rate * gradient_norm ** 2:
                    weights, bias = candidate_w, candidate_b
                    loss, grad_w, grad_b = (
                        candidate_loss,
                        candidate_grad_w,
                        candidate_grad_b,
                    )
                    learning_rate = min(self.learning_rate, trial_rate * 1.05)
                    accepted = True
                    break
                trial_rate *= 0.5
            if not accepted:
                break
            self.training_trace.append(loss)
            if len(self.training_trace) > 2 and abs(
                    self.training_trace[-2] - self.training_trace[-1]
            ) < self.tolerance:
                break

        self.weights = weights
        self.bias = float(bias)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return two-column logistic scores after verifying the model is fitted."""
        if self.weights is None:
            raise RuntimeError("The model must be fitted before prediction.")
        probability = stable_sigmoid(np.asarray(X) @ self.weights + self.bias)
        return np.column_stack([1.0 - probability, probability])


# =============================================================================
# 11. BAYESIAN-STYLE SERIES EVIDENCE FUSION
# =============================================================================


def fuse_series_evidence(
        probabilities: np.ndarray,
        n_slices: np.ndarray,
        prior: float,
        tau: float,
) -> float:
    """Combine correlated series scores conservatively in prior-relative log-odds.

    Reliability ``n/(n+tau)`` increases with slice count but saturates.  Evidence
    is averaged rather than multiplied, so exporting the same information many
    times cannot make certainty grow without bound.  This is Bayesian-inspired
    accounting, not a complete causal Bayesian network.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    reliability = np.asarray(n_slices, dtype=np.float64) / (
            np.asarray(n_slices, dtype=np.float64) + float(tau)
    )
    evidence = logit(probabilities) - float(logit(prior))
    total_weight = max(float(reliability.sum()), 1e-12)
    patient_logit = float(logit(prior)) + float(np.dot(reliability, evidence) / total_weight)
    return float(stable_sigmoid(np.asarray([patient_logit]))[0])


# =============================================================================
# 12. MODEL FIT/PREDICT ROUTINES
# =============================================================================


def fit_standard_logistic(
        X: np.ndarray,
        y: np.ndarray,
        C: float,
        config: Config,
) -> tuple[CompactPCA, LogisticRegression]:
    """Fit the standard compact patient-level linear baseline.

    Standardization and PCA are training-only.  Logistic Regression uses balanced
    class weights and explicit regularization ``C`` selected by inner CV.  The
    returned tuple contains every object required to transform validation rows
    without refitting.
    """
    preprocessor = CompactPCA(config.pca_variance, config.max_pca_components)
    transformed = preprocessor.fit_transform(X)
    model = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        solver="liblinear",
        max_iter=4000,
        random_state=config.seed,
    ).fit(transformed, y)
    return preprocessor, model


def predict_standard_logistic(
        fitted: tuple[CompactPCA, LogisticRegression], X: np.ndarray
) -> np.ndarray:
    """Apply a fitted compact PCA/logistic pipeline and return positive-class scores."""
    preprocessor, model = fitted
    return model.predict_proba(preprocessor.transform(X))[:, 1]


def fit_residualized_logistic(
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        alpha: float,
        C: float,
        config: Config,
) -> tuple[NuisanceResidualizer, CompactPCA, LogisticRegression]:
    """Fit nuisance residualization followed by the standard logistic pipeline.

    Both the residualizer and classifier are fitted on the same training
    partition.  Validation nuisance variables are used only to apply the already
    learned subtraction, never to estimate coefficients.
    """
    residualizer = NuisanceResidualizer(alpha, config.max_nuisance_components).fit(X, Z)
    cleaned = residualizer.transform(X, Z)
    preprocessor, model = fit_standard_logistic(cleaned, y, C, config)
    return residualizer, preprocessor, model


def predict_residualized_logistic(
        fitted: tuple[NuisanceResidualizer, CompactPCA, LogisticRegression],
        X: np.ndarray,
        Z: np.ndarray,
) -> np.ndarray:
    """Apply the training-fitted residualizer and logistic classifier to new rows."""
    residualizer, preprocessor, model = fitted
    cleaned = residualizer.transform(X, Z)
    return model.predict_proba(preprocessor.transform(cleaned))[:, 1]


def fit_guarded_logistic(
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        confounder_lambda: float,
        l2: float,
        config: Config,
) -> tuple[CompactPCA, NuisanceEncoder, WorstCaseLinearConfounderLogistic]:
    """Fit fold-local cardiac/nuisance encoders and the custom guarded objective.

    Cardiac and nuisance PCA transforms are learned only from training patients.
    The custom model then optimizes disease discrimination while penalizing score
    covariance with the compressed nuisance axes.
    """
    x_encoder = CompactPCA(config.pca_variance, config.max_pca_components).fit(X)
    z_encoder = NuisanceEncoder(config.max_nuisance_components).fit(Z)
    Xc = x_encoder.transform(X)
    Zc = z_encoder.transform(Z)
    model = WorstCaseLinearConfounderLogistic(
        confounder_lambda=confounder_lambda,
        l2=l2,
        learning_rate=config.guard_learning_rate,
        max_iterations=config.guard_max_iterations,
        tolerance=config.guard_tolerance,
    ).fit(Xc, Zc, y)
    return x_encoder, z_encoder, model


def predict_guarded_logistic(
        fitted: tuple[CompactPCA, NuisanceEncoder, WorstCaseLinearConfounderLogistic],
        X: np.ndarray,
        Z: np.ndarray,
) -> np.ndarray:
    """Predict from cardiac features using a fitted confounder-guarded model.

    Nuisance variables are transformed only to verify the fold-local data
    contract; the deployed score itself depends on the cardiac representation
    and learned weights, not on acquisition metadata supplied at prediction.
    """
    x_encoder, z_encoder, model = fitted
    # Nuisance variables regularize training; prediction itself uses only the
    # cardiac representation, so no acquisition metadata is required at deploy time.
    _ = z_encoder.transform(Z)  # Validate the fold-local nuisance contract.
    return model.predict_proba(x_encoder.transform(X))[:, 1]


def fit_bayesian_base_classifier(
        patient_bank: PatientBank,
        train_indices: np.ndarray,
        C: float,
        config: Config,
) -> tuple[CompactPCA, LogisticRegression, float]:
    """Fit one patient-level base classifier for later series evidence fusion.

    Training uses one cardiac mean per patient, avoiding repeated labels on
    series rows.  The training-class prevalence becomes the fold-local prior.
    At validation time the same decision function is reused on each series and
    evidence is fused conservatively.
    """

    fitted = fit_standard_logistic(
        patient_bank.features["cardiac"][train_indices],
        patient_bank.labels[train_indices],
        C,
        config,
    )
    prior = float(patient_bank.labels[train_indices].mean())
    return fitted[0], fitted[1], prior


def predict_bayesian_patients(
        fitted: tuple[CompactPCA, LogisticRegression, float],
        series: SeriesBank,
        patient_ids: Sequence[str],
        tau: float,
) -> np.ndarray:
    """Score every validation series and fuse evidence to one score per patient.

    Series rows are selected only by patient ID.  Slice counts determine
    saturating reliability weights, and ``tau`` is chosen in inner CV.
    """
    preprocessor, model, prior = fitted
    probabilities = []
    for patient_id in patient_ids:
        rows = np.flatnonzero(series.patient_ids == patient_id)
        series_probability = model.predict_proba(
            preprocessor.transform(series.features["cardiac"][rows])
        )[:, 1]
        probabilities.append(
            fuse_series_evidence(
                series_probability,
                series.n_slices[rows],
                prior=prior,
                tau=tau,
            )
        )
    return np.asarray(probabilities, dtype=np.float64)


# =============================================================================
# 13. NESTED PATIENT-LEVEL CROSS-VALIDATION
# =============================================================================


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose a conservative Youden threshold from inner out-of-fold scores.

    The utility is sensitivity minus false-positive rate.  If several finite
    thresholds tie, the highest threshold is selected.  Outer-validation scores
    never influence the operating point.
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    utility = true_positive_rate - false_positive_rate
    candidates = np.flatnonzero(finite & (utility == np.max(utility[finite])))
    # Among ties choose the highest threshold, which is more conservative.
    return float(np.max(thresholds[candidates]))


def candidate_grid(experiment: str, config: Config) -> list[dict[str, float]]:
    """Return the predeclared hyperparameter combinations for one experiment.

    Standard controls tune only ``C``.  Residualization tunes ridge ``alpha`` and
    ``C``; the guard tunes nuisance penalty and L2; Bayesian fusion tunes ``C``
    and reliability ``tau``.  No grid is expanded after observing outer results.
    """
    if experiment in {
        "BASELINE_A17",
        "CONTROL_SUPPORT_ONLY",
        "CONTROL_SHUFFLED_INTENSITY",
        "CONTROL_OUTSIDE_SUPPORT",
    }:
        return [{"C": C} for C in config.c_grid]
    if experiment == "CUSTOM_NUISANCE_PROJECTION":
        return [
            {"alpha": alpha, "C": C}
            for alpha in config.residual_alpha_grid
            for C in config.c_grid
        ]
    if experiment == "CUSTOM_MINIMAX_CONFOUNDER_GUARD":
        return [
            {"lambda": value, "l2": l2}
            for value in config.guard_lambda_grid
            for l2 in config.guard_l2_grid
        ]
    if experiment == "CUSTOM_BAYESIAN_SERIES_FUSION":
        return [
            {"C": C, "tau": tau}
            for C in config.c_grid
            for tau in config.bayes_tau_grid
        ]
    raise KeyError(experiment)


def method_feature_mode(experiment: str) -> str:
    """Map each experiment to its frozen patient feature representation."""
    return {
        "BASELINE_A17": "cardiac",
        "CONTROL_SUPPORT_ONLY": "support_only",
        "CONTROL_SHUFFLED_INTENSITY": "shuffled_intensity",
        "CONTROL_OUTSIDE_SUPPORT": "outside_support",
        "CUSTOM_NUISANCE_PROJECTION": "cardiac",
        "CUSTOM_MINIMAX_CONFOUNDER_GUARD": "cardiac",
    }.get(experiment, "cardiac")


def fit_predict_method(
        experiment: str,
        params: dict[str, float],
        patient_bank: PatientBank,
        series_bank: SeriesBank,
        train_indices: np.ndarray,
        valid_indices: np.ndarray,
        config: Config,
) -> np.ndarray:
    """Fit one declared method on training patients and score validation patients.

    This dispatcher centralizes the leakage boundary.  It selects the correct
    frozen feature mode, passes nuisance variables only to deconfounding methods,
    and invokes series-level evidence only for the Bayesian-style experiment.
    It always returns one score per validation patient.
    """
    train_patients = patient_bank.patient_ids[train_indices]
    valid_patients = patient_bank.patient_ids[valid_indices]
    y_train = patient_bank.labels[train_indices]

    if experiment == "CUSTOM_BAYESIAN_SERIES_FUSION":
        fitted = fit_bayesian_base_classifier(
            patient_bank,
            train_indices,
            params["C"],
            config,
        )
        return predict_bayesian_patients(
            fitted,
            series_bank,
            valid_patients,
            params["tau"],
        )

    mode = method_feature_mode(experiment)
    X_train = patient_bank.features[mode][train_indices]
    X_valid = patient_bank.features[mode][valid_indices]

    if experiment in {
        "BASELINE_A17",
        "CONTROL_SUPPORT_ONLY",
        "CONTROL_SHUFFLED_INTENSITY",
        "CONTROL_OUTSIDE_SUPPORT",
    }:
        fitted = fit_standard_logistic(X_train, y_train, params["C"], config)
        return predict_standard_logistic(fitted, X_valid)

    Z_train = patient_bank.nuisance[train_indices]
    Z_valid = patient_bank.nuisance[valid_indices]
    if experiment == "CUSTOM_NUISANCE_PROJECTION":
        fitted = fit_residualized_logistic(
            X_train,
            Z_train,
            y_train,
            params["alpha"],
            params["C"],
            config,
        )
        return predict_residualized_logistic(fitted, X_valid, Z_valid)

    if experiment == "CUSTOM_MINIMAX_CONFOUNDER_GUARD":
        fitted = fit_guarded_logistic(
            X_train,
            Z_train,
            y_train,
            params["lambda"],
            params["l2"],
            config,
        )
        return predict_guarded_logistic(fitted, X_valid, Z_valid)

    raise KeyError(experiment)


def parameter_preference(experiment: str, params: dict[str, float]) -> tuple:
    """Apply deterministic scientific tie-breaking within the AUC tolerance.

    The rule favors simpler/stronger-regularized settings or, for the guard,
    stronger nuisance control when inner AUROC differences are negligible.  This
    avoids repeatedly selecting an extreme weakly regularized setting because of
    one tiny inner-CV fluctuation.
    """

    if experiment == "CUSTOM_NUISANCE_PROJECTION":
        # Smaller alpha removes nuisance more strongly; smaller C regularizes more.
        return (params["alpha"], params["C"])
    if experiment == "CUSTOM_MINIMAX_CONFOUNDER_GUARD":
        # Prefer stronger nuisance guard, then stronger L2, if AUC is effectively tied.
        return (-params["lambda"], -params["l2"])
    if experiment == "CUSTOM_BAYESIAN_SERIES_FUSION":
        # Prefer stronger classifier regularization and weaker slice-count
        # dependence when inner AUC is effectively tied.
        return (params["C"], -params["tau"])
    return (params["C"],)


def select_hyperparameters(
        experiment: str,
        patient_bank: PatientBank,
        series_bank: SeriesBank,
        outer_train_indices: np.ndarray,
        config: Config,
        seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Select parameters using only inner out-of-fold patient scores.

    Every candidate is evaluated on identical stratified inner folds within the
    current outer-training cohort.  Candidates within ``auc_tolerance`` of the
    best inner AUROC are resolved by ``parameter_preference``.  The selected
    inner OOF scores are also returned for threshold choice.
    """

    y_outer = patient_bank.labels[outer_train_indices]
    splitter = StratifiedKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=seed,
    )
    candidates = []
    for params in candidate_grid(experiment, config):
        scores = np.full(len(outer_train_indices), np.nan, dtype=np.float64)
        for inner_train_local, inner_valid_local in splitter.split(
                np.zeros(len(y_outer)), y_outer
        ):
            inner_train = outer_train_indices[inner_train_local]
            inner_valid = outer_train_indices[inner_valid_local]
            scores[inner_valid_local] = fit_predict_method(
                experiment,
                params,
                patient_bank,
                series_bank,
                inner_train,
                inner_valid,
                config,
            )
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(f"Non-finite inner scores for {experiment}: {params}")
        candidates.append(
            {
                "params": params,
                "auc": float(roc_auc_score(y_outer, scores)),
                "scores": scores,
            }
        )

    best_auc = max(row["auc"] for row in candidates)
    eligible = [
        row for row in candidates if row["auc"] >= best_auc - config.auc_tolerance
    ]
    selected = min(
        eligible,
        key=lambda row: parameter_preference(experiment, row["params"]),
    )
    return selected["params"], y_outer, selected["scores"]


def nested_cv_once(
        experiment: str,
        patient_bank: PatientBank,
        series_bank: SeriesBank,
        config: Config,
        seed: int,
) -> dict[str, Any]:
    """Run one complete nested patient-level cross-validation pass.

    For each outer fold, hyperparameters and threshold are selected inside the
    outer-training patients, the selected method is refit on all outer training
    rows, and each held-out patient receives exactly one score.  The output stores
    fold IDs, scores, thresholds, predictions, selected parameters, and pooled
    patient-level metrics.
    """

    labels = patient_bank.labels
    splitter = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=seed,
    )
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    predictions = np.full(len(labels), -1, dtype=np.int64)
    thresholds = np.full(len(labels), np.nan, dtype=np.float64)
    folds = np.full(len(labels), -1, dtype=np.int64)
    selected_params: list[dict[str, Any]] = []

    for fold_index, (train_indices, valid_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        params, inner_labels, inner_scores = select_hyperparameters(
            experiment,
            patient_bank,
            series_bank,
            train_indices,
            config,
            seed=seed + 1000 + fold_index,
        )
        threshold = choose_threshold(inner_labels, inner_scores)
        fold_scores = fit_predict_method(
            experiment,
            params,
            patient_bank,
            series_bank,
            train_indices,
            valid_indices,
            config,
        )
        scores[valid_indices] = fold_scores
        thresholds[valid_indices] = threshold
        predictions[valid_indices] = (fold_scores >= threshold).astype(np.int64)
        folds[valid_indices] = fold_index
        selected_params.append(
            {
                "outer_fold": fold_index,
                "params": params,
                "inner_auc": float(roc_auc_score(inner_labels, inner_scores)),
                "threshold": threshold,
            }
        )

    if not np.all(np.isfinite(scores)) or np.any(predictions < 0):
        raise RuntimeError(f"Incomplete OOF predictions for {experiment}")
    metrics = compute_metrics(labels, scores, predictions)
    return {
        "experiment": experiment,
        "seed": seed,
        "patient_ids": patient_bank.patient_ids.copy(),
        "labels": labels.copy(),
        "scores": scores,
        "predictions": predictions,
        "thresholds": thresholds,
        "folds": folds,
        "selected_params": selected_params,
        "metrics": metrics,
    }


def compute_metrics(
        labels: np.ndarray,
        scores: np.ndarray,
        predictions: np.ndarray,
) -> dict[str, float]:
    """Compute patient-level discrimination and threshold metrics.

    AUROC and AUPRC use continuous out-of-fold scores.  Sensitivity, specificity,
    and F1 use the fold-specific thresholds chosen from training data.  Metrics
    describe the released cohort and are not clinical operating estimates.
    """
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def bootstrap_auc_interval(
        labels: np.ndarray,
        scores: np.ndarray,
        replicates: int,
        seed: int,
) -> tuple[float, float]:
    """Estimate a percentile AUROC interval by resampling patients.

    Each bootstrap replicate samples patient rows with replacement.  Replicates
    containing only one class are skipped because AUROC is undefined.  The
    interval quantifies finite-cohort sampling uncertainty but does not account
    for external domain shift.
    """
    rng = np.random.default_rng(seed)
    values = []
    indices = np.arange(len(labels))
    for _ in range(replicates):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sample])) < 2:
            continue
        values.append(roc_auc_score(labels[sample], scores[sample]))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def cross_validated_nuisance_r2(
        scores: np.ndarray,
        nuisance: np.ndarray,
        folds: np.ndarray,
) -> float:
    """Measure out-of-fold linear predictability of a model score from nuisance.

    Within each existing outer fold, nuisance scaling, PCA, and Ridge regression
    are fitted on all other patients and predict the held-out scores.  The final
    R-squared compares these predictions with the observed OOF score variance.
    A lower value is desirable for deconfounding, but near-zero linear R-squared
    cannot rule out nonlinear dependence.
    """

    predicted = np.full(len(scores), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        scaler = StandardScaler().fit(nuisance[train])
        Z_train = scaler.transform(nuisance[train])
        Z_valid = scaler.transform(nuisance[valid])
        maximum = max(1, min(5, Z_train.shape[0] - 2, Z_train.shape[1]))
        pca = PCA(n_components=maximum, svd_solver="full").fit(Z_train)
        model = Ridge(alpha=1.0).fit(pca.transform(Z_train), scores[train])
        predicted[valid] = model.predict(pca.transform(Z_valid))
    denominator = float(np.sum((scores - scores.mean()) ** 2))
    return float(1.0 - np.sum((scores - predicted) ** 2) / max(denominator, 1e-12))


# =============================================================================
# 14. RUN, SUMMARIZE, AND SAVE
# =============================================================================


def save_oof_result(result: dict[str, Any], output_dir: Path) -> None:
    """Save one row per patient and the selected fold parameters for an experiment.

    The CSV supports direct inspection of labels, fold assignments, continuous
    scores, training-only thresholds, and predictions.  A companion JSON file
    preserves the chosen hyperparameters and inner AUROC for every outer fold.
    """
    path = output_dir / f"oof_{result['experiment']}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "patient_id",
                "label",
                "fold",
                "score",
                "threshold",
                "prediction",
            ]
        )
        for row in zip(
                result["patient_ids"],
                result["labels"],
                result["folds"],
                result["scores"],
                result["thresholds"],
                result["predictions"],
        ):
            writer.writerow(row)
    write_json(
        output_dir / f"selected_params_{result['experiment']}.json",
        result["selected_params"],
    )


def run_research_portfolio(config: Config) -> dict[str, Any]:
    """Execute the complete self-contained research portfolio pipeline.

    The orchestration order is deliberate:
    1. validate configuration and seed generators;
    2. save configuration/software provenance;
    3. discover the patient/image manifest;
    4. load or build the compact frozen series bank;
    5. pool to patient features and nuisance variables;
    6. run all seven nested-CV experiments;
    7. repeat each experiment over predeclared split seeds;
    8. compute paired deltas and nuisance-score R-squared;
    9. write ranked summaries and an interpretation-guarded final report.

    The function returns the same final report that is written to disk.
    """
    # ---------------------------------------------------------------------
    # STAGE 1 — VALIDATE THE DECLARED EXPERIMENT BEFORE EXPENSIVE WORK
    # ---------------------------------------------------------------------
    # A malformed percentile, kernel, fold count, or hyperparameter grid should
    # fail before model downloads and before a partial cache can be produced.
    validate_config(config)
    seed_everything(config.seed)

    # ---------------------------------------------------------------------
    # STAGE 2 — CREATE THE OUTPUT ROOT AND FREEZE RUN PROVENANCE
    # ---------------------------------------------------------------------
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "configuration.json", asdict(config))
    source_identity = _runtime_source_identity()
    write_json(
        output_dir / "run_metadata.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "runtime_environment": config.runtime_environment,
            "dataset_path": str(Path(config.dataset_path).resolve()),
            "dataset_path_source": config.dataset_path_source,
            "output_directory": str(Path(config.output_dir).resolve(strict=False)),
            "output_directory_source": config.output_dir_source,
            **source_identity,
        },
    )

    # ---------------------------------------------------------------------
    # STAGE 3 — DISCOVER THE COMPLETE PATIENT/IMAGE MANIFEST
    # ---------------------------------------------------------------------
    # Discovery reads folder and filename structure only.  Pixels are decoded in
    # the feature-bank stage, after patient and series membership is fixed.
    records = discover_records(config.dataset_path)

    # ---------------------------------------------------------------------
    # STAGE 4 — LOAD OR BUILD THE FROZEN SERIES-LEVEL FEATURE BANK
    # ---------------------------------------------------------------------
    # This is the only stage that requires MONAI/EfficientNet inference.  On a
    # cache hit, all downstream mathematical experiments reuse the same frozen
    # series rows and can execute without decoding images again.
    series_bank = extract_series_bank(config, records)

    # ---------------------------------------------------------------------
    # STAGE 5 — POOL SERIES TO ONE ROW PER PATIENT AND BUILD NUISANCE AXES
    # ---------------------------------------------------------------------
    patient_bank = build_patient_bank(
        series_bank,
        config.nuisance_random_projection_dim,
        config.nuisance_random_projection_seed,
    )
    class_counts = np.bincount(patient_bank.labels, minlength=2)
    if np.min(class_counts) < config.outer_folds:
        raise ValueError(
            "Each class must contain at least outer_folds patients; "
            f"counts={class_counts.tolist()}."
        )
    if len(patient_bank.patient_ids) != 30:
        print(
            f"[WARNING] Expected the validated release to contain 30 patients; "
            f"discovered {len(patient_bank.patient_ids)}.",
            flush=True,
        )

    # ---------------------------------------------------------------------
    # STAGE 6 — RUN THE LOCKED SEVEN-EXPERIMENT PANEL
    # ---------------------------------------------------------------------
    # Every experiment receives the same patients and follows the same nested-CV
    # boundary.  The baseline and controls differ only in frozen information;
    # custom methods differ in their fold-local mathematical treatment.
    main_results: dict[str, dict[str, Any]] = {}
    repeated_rows = []
    for experiment in EXPERIMENTS:
        print(f"\n[EVALUATION] {experiment}", flush=True)
        main = nested_cv_once(
            experiment,
            patient_bank,
            series_bank,
            config,
            seed=config.seed,
        )
        main["metrics"]["nuisance_score_r2"] = cross_validated_nuisance_r2(
            main["scores"], patient_bank.nuisance, main["folds"]
        )
        main["metrics"]["auc_ci_low"], main["metrics"]["auc_ci_high"] = (
            bootstrap_auc_interval(
                main["labels"],
                main["scores"],
                config.bootstrap_replicates,
                seed=config.seed + 50_000,
            )
        )
        main_results[experiment] = main
        save_oof_result(main, output_dir)

        # Repeated splits are generated after the main manifest.  They are not
        # used to choose a favorable run; all declared repetitions are saved.
        for repeat in range(config.repeated_cv_runs):
            repeated = nested_cv_once(
                experiment,
                patient_bank,
                series_bank,
                config,
                seed=config.seed + 10_000 + repeat,
            )
            repeated_rows.append(
                {
                    "experiment": experiment,
                    "repeat": repeat + 1,
                    "seed": repeated["seed"],
                    **repeated["metrics"],
                }
            )
        print(
            f"[RESULT] AUC={main['metrics']['auc']:.4f}; "
            f"AUPRC={main['metrics']['auprc']:.4f}; "
            f"nuisance_score_R2={main['metrics']['nuisance_score_r2']:.4f}",
            flush=True,
        )

    # ---------------------------------------------------------------------
    # STAGE 7 — SAVE EVERY REPEATED RUN BEFORE COMPUTING SUMMARIES
    # ---------------------------------------------------------------------
    repeated_path = output_dir / "repeated_cv_results.csv"
    with repeated_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(repeated_rows[0]))
        writer.writeheader()
        writer.writerows(repeated_rows)

    # ---------------------------------------------------------------------
    # STAGE 8 — RANK BY REPEATED-CV MEDIAN, NOT BY ONE FAVORABLE SPLIT
    # ---------------------------------------------------------------------
    summary_rows = []
    for experiment in EXPERIMENTS:
        auc_values = np.asarray(
            [row["auc"] for row in repeated_rows if row["experiment"] == experiment]
        )
        metrics = main_results[experiment]["metrics"]
        summary_rows.append(
            {
                "experiment": experiment,
                **metrics,
                "repeated_auc_median": float(np.median(auc_values)),
                "repeated_auc_q25": float(np.quantile(auc_values, 0.25)),
                "repeated_auc_q75": float(np.quantile(auc_values, 0.75)),
                "repeated_auc_min": float(np.min(auc_values)),
                "repeated_auc_max": float(np.max(auc_values)),
            }
        )
    summary_rows.sort(key=lambda row: -row["repeated_auc_median"])

    with (output_dir / "experiment_summary.csv").open(
            "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---------------------------------------------------------------------
    # STAGE 9 — COMPUTE SAME-SEED PAIRED DELTAS AGAINST BASELINE_A17
    # ---------------------------------------------------------------------
    # Pairing by seed removes variation caused solely by using different patient
    # splits when comparing two methods.
    lookup = {row["experiment"]: row for row in summary_rows}
    baseline_auc = lookup["BASELINE_A17"]["repeated_auc_median"]
    auc_by_experiment_and_seed = {
        experiment: {
            int(row["seed"]): float(row["auc"])
            for row in repeated_rows
            if row["experiment"] == experiment
        }
        for experiment in EXPERIMENTS
    }
    paired_repeated_deltas = {}
    baseline_by_seed = auc_by_experiment_and_seed["BASELINE_A17"]
    for experiment in EXPERIMENTS:
        if experiment == "BASELINE_A17":
            continue
        comparison_by_seed = auc_by_experiment_and_seed[experiment]
        common_seeds = sorted(set(baseline_by_seed) & set(comparison_by_seed))
        deltas = np.asarray(
            [comparison_by_seed[seed] - baseline_by_seed[seed] for seed in common_seeds],
            dtype=np.float64,
        )
        paired_repeated_deltas[experiment] = {
            "definition": "comparison AUROC minus BASELINE_A17 AUROC",
            "runs": int(len(deltas)),
            "median": float(np.median(deltas)),
            "q25": float(np.quantile(deltas, 0.25)),
            "q75": float(np.quantile(deltas, 0.75)),
            "minimum": float(np.min(deltas)),
            "maximum": float(np.max(deltas)),
            "fraction_above_zero": float(np.mean(deltas > 0.0)),
        }
    # ---------------------------------------------------------------------
    # STAGE 10 — WRITE AN INTERPRETATION-GUARDED FINAL REPORT
    # ---------------------------------------------------------------------
    # The report includes explicit rules that prevent a reader from treating a
    # high internal AUROC as proof of clinical CAD detection or causal validity.
    report = {
        "research_question": (
            "Can cardiac-region MRI information outperform exact matched shortcut "
            "controls, and can fold-local mathematical deconfounding preserve or "
            "improve patient-level discrimination?"
        ),
        "patient_count": int(len(patient_bank.patient_ids)),
        "series_count": int(len(series_bank.series_ids)),
        "slice_count": int(len(records)),
        "results_ranked_by_repeated_auc_median": summary_rows,
        "baseline_minus_controls": {
            control: float(baseline_auc - lookup[control]["repeated_auc_median"])
            for control in (
                "CONTROL_SUPPORT_ONLY",
                "CONTROL_SHUFFLED_INTENSITY",
                "CONTROL_OUTSIDE_SUPPORT",
            )
        },
        "custom_minus_baseline": {
            method: float(lookup[method]["repeated_auc_median"] - baseline_auc)
            for method in (
                "CUSTOM_NUISANCE_PROJECTION",
                "CUSTOM_MINIMAX_CONFOUNDER_GUARD",
                "CUSTOM_BAYESIAN_SERIES_FUSION",
            )
        },
        "paired_repeated_auc_deltas_vs_baseline": paired_repeated_deltas,
        "deconfounding_success_rule": (
            "A custom method is promising only if AUROC is preserved or improved "
            "and nuisance_score_R2 decreases. AUROC alone is not sufficient."
        ),
        "interpretation_guardrails": [
            "The effective labeled sample size is the number of patients, not images.",
            "Repeated split results are sensitivity analyses, not independent cohorts.",
            "A high control AUC indicates unresolved dataset structure or confounding.",
            "External validation and sequence/view matching are required for clinical claims.",
        ],
    }
    write_json(output_dir / "final_report.json", report)

    print("\nFINAL RANKING BY REPEATED-CV MEDIAN AUC", flush=True)
    for rank, row in enumerate(summary_rows, start=1):
        print(
            f"{rank:>2}. {row['experiment']:<36} "
            f"median={row['repeated_auc_median']:.4f} "
            f"IQR=[{row['repeated_auc_q25']:.4f}, {row['repeated_auc_q75']:.4f}] "
            f"nuisance_R2={row['nuisance_score_r2']:.4f}",
            flush=True,
        )
    return report


def run_showcase(config: Config) -> dict[str, Any]:
    """Backward-compatible generic alias for ``run_research_portfolio``.

    It exists only so earlier notebooks continue to run.  All scientific logic
    remains in one implementation.
    """

    return run_research_portfolio(config)


# =============================================================================
# 15. NOTEBOOK-SAFE ENTRY POINT
# =============================================================================


def build_main_config() -> tuple[Config, ResolvedRuntimePaths]:
    """Resolve paths and construct the immutable configuration used by ``main``.

    This function intentionally accepts no command-line namespace and never reads
    ``sys.argv``.  Consequently, internal Jupyter/Colab arguments such as
    ``-f <kernel.json>`` cannot alter the pipeline or trigger a parser failure.  Edit the ``MAIN_*`` constants above, or use the documented
    environment variables, when a non-default runtime setting is required.
    """

    paths = resolve_runtime_paths(
        MAIN_DATASET_PATH_OVERRIDE,
        MAIN_OUTPUT_DIR_OVERRIDE,
    )
    config = Config(
        dataset_path=str(paths.dataset_path),
        output_dir=str(paths.output_dir),
        dataset_path_source=paths.dataset_source,
        output_dir_source=paths.output_source,
        runtime_environment=paths.runtime_environment,
        batch_size=int(MAIN_BATCH_SIZE),
        num_workers=int(MAIN_NUM_WORKERS),
        repeated_cv_runs=int(MAIN_REPEATED_CV_RUNS),
        force_rebuild_cache=bool(MAIN_FORCE_REBUILD_CACHE),
        use_amp=bool(MAIN_USE_AMP),
    )
    return config, paths


def main() -> None:
    """Run the complete pipeline without consuming any command-line arguments.

    The function is safe when this file is:

    * launched with ``python cad_mri_university_showcase.py``;
    * executed with IPython ``%run``;
    * pasted or executed inside Colab/Jupyter;
    * imported and called as ``module.main()``.

    Notebook kernels may leave arbitrary values in ``sys.argv``.  They are
    deliberately ignored.  Runtime paths still support automatic Kaggle/local
    discovery, source-code overrides, and environment-variable overrides.
    """

    config, paths = build_main_config()
    print_resolved_runtime_paths(paths)

    if MAIN_CHECK_PATHS_ONLY:
        print(
            "[PATHS] Validation completed. No model was loaded and no output "
            "directory was created.",
            flush=True,
        )
        return

    run_research_portfolio(config)


if __name__ == "__main__":
    main()
