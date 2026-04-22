# autoPET PSMA+FDG nnU-Net Segmentation Pipeline

This repository implements a whole-body PET/CT lesion segmentation pipeline using the autoPET dataset (PSMA-FDG-PET-CT) and the nnU-Net framework. It was developed as a modular research project for a medical imaging course.

## Repository Overview

- **`src/`**: Modular pipeline scripts (`step01` to `step08`) handling data inventory, preprocessing, training, inference, and evaluation.
- **`configs/`**: Centralized YAML configurations to decouple logic from data and parameters:
  - `paths.yaml`: Manages all directory structures (raw data, outputs, and nnU-Net internal paths). **Benefit**: Simplifies migration to different workstations.
  - `subset.yaml`: Controls training/validation split sizes and filters (e.g., specific diagnoses). **Benefit**: Ensures reproducible sampling for experiments.
  - `training.yaml`: Defines nnU-Net dataset IDs, network architectures (e.g., 3D fullres), and training duration. **Benefit**: Allows easy model switching without code changes.
  - `evaluation.yaml`: Sets performance metrics (Dice, Sensitivity) and lesion-level filtering thresholds. **Benefit**: Standardizes evaluation criteria across runs.
- **`tests/`**: Unit tests for core pipeline logic.
- **`output/`**:
  - `figures/`: Generated visualization panels and distribution plots.
  - `metrics/`: CSV files containing quantitative evaluation results.
  - *Note: NIfTI predictions, raw data, and preprocessed intermediates are ignored to maintain repository efficiency.*

## Setup & Dependencies

### External Frameworks
This project relies on the following external repositories which are **not** tracked in this repository:
1. **nnU-Net**: The core segmentation framework. 
   - Repository: [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
   - Setup: Clone and install according to their documentation.
2. **autoPET-3 Submission**: Submission-ready wrappers for the challenge.
   - Repository: [autoPET/autopet-3-submission](https://github.com/treichler/autopet-3-submission)

### Environment Variables
nnU-Net requires specific environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) to be set. These should point to directories within the `output/` folder or elsewhere as per your local setup.

## Pipeline Workflow

This project follows a structured 8-step execution pipeline:

1.  **Step 01 — Inventory**: Scans the raw data directory to identify all available PSMA and FDG PET/CT cases.
2.  **Step 02 — Manifest**: Joins inventory data with metadata and official splits to create a master project manifest.
3.  **Step 03 — Subset**: Samples a balanced subset (e.g., 80 train, 20 validation) for faster iteration and testing.
4.  **Step 04 — nnU-Net Setup**: Organizes files into the required nnU-Net folder structure and runs preprocessing.
5.  **Step 05 — Training**: Executes the training process (currently configured for a 3D fullres U-Net).
6.  **Step 06 — Inference**: Generates segmentation masks for the validation/test cases using the trained model.
7.  **Step 07 — Evaluation**: Computes Dice scores, Sensitivity, and False Positive components per case.
8.  **Step 08 — Visualisation**: Generates comparison panels (Overlay plots) and distribution charts for analysis.

## Results
Final evaluation metrics and visualizations are stored in `output/metrics/` and `output/figures/` respectively.
