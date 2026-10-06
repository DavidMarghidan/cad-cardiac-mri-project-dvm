# Data, third-party material, and research-use notice

This file defines what the repository licenses do **not** grant.

## 1. CAD Cardiac MRI Dataset

The MIT source-code license and CC BY 4.0 documentation notice do not grant rights to the CAD Cardiac MRI Dataset, its JPEG images, metadata, labels, or any other source material. Obtain the dataset from its official distribution source and comply with the dataset provider's current terms and the associated publication.

The associated preprint records project-specific correspondence in which the dataset creator granted David Vlad Marghidan academic-research use with citation and clarified that each top-level `Directory_*` folder represents one patient in the analysed Kaggle subset. That communication must not be interpreted as a sublicense or a general redistribution permission for third parties.

No source patient image is included in this publication package.

## 2. Manual and generated artifacts

The repository package does not include the project's manual masks, annotation-label CSV, trained checkpoints, automatic mask PNGs, feature bank, or complete persistent Kaggle workspace. The source-code license does not automatically grant permission to redistribute those items.

Generated outputs may encode or be derived from the source dataset. Before publishing masks, embeddings, checkpoints, patient-level predictions, examples, or other artifacts, review:

- the dataset provider's terms;
- institutional and supervisory requirements;
- privacy and re-identification risk;
- whether the artifact can reveal source-image content; and
- the intended scientific citation and provenance record.

## 3. Third-party software and pretrained weights

The pipeline imports Python, PyTorch, torchvision, OpenCV, NumPy, pandas, scikit-learn, IPython, and tqdm components. Those projects remain under their own licenses. They are dependencies, not relicensed copies of this repository's original source.

`EfficientNet_B0_Weights.IMAGENET1K_V1` is obtained through torchvision. The pretrained weights and the ImageNet source material remain subject to their own terms. The MIT License for this repository does not relicense them.

## 4. Research and clinical boundary

This software is supplied for research and education. It is a secondary analysis of a small public JPEG release and has not been externally or clinically validated. It is not a medical device and must not be used to diagnose, treat, triage, or make decisions about a patient.

No warranty is provided regarding scientific validity, fitness for a particular purpose, clinical safety, regulatory compliance, or reproducibility on different hardware and software environments. The warranty and liability terms of the MIT License apply to the original source code.

## 5. Citation and provenance

When using the code or documentation, retain the repository license notices and cite the relevant project manuscript and dataset article. When reporting results, identify the exact source version, notebook version, target state, workspace lineage, and whether caches were reused or rebuilt.
