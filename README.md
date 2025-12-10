# Detecting Alzheimer's Disease from MRI using Convolutional Neural Networks

**Course:** CSCA 5642 - Introduction to Deep Learning  
**Institution:** University of Colorado Boulder  
**Author:** Philipp Adrian Pohlmann  
**Date:** December 2025

## Overview

This project explores how convolutional neural networks (CNNs) can classify brain MRI slices into diagnostic categories related to Alzheimer's disease. The analysis compares three distinct CNN architectures to understand the relationship between model complexity, dataset size, and generalization performance in medical imaging.

### Problem Statement

Alzheimer's disease affects millions globally, and early detection through brain MRI analysis is critical for intervention. This project develops and evaluates deep learning models to:
- Classify brain MRI slices into three categories: Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Normal Control (NC)
- Compare architectures from simple CNNs to transfer learning approaches
- Understand what model complexity is appropriate for small medical imaging datasets

### Results

- **Best Model:** Simple Baseline CNN (Val Acc: 95.7%, Test Acc: 85.4%)
- Deep CNN suffered from severe overfitting (Val Acc: 51.1%)
- Transfer learning underperformed due to domain mismatch (Val Acc: 46.8%)
- Insight:** Model complexity must match dataset size—simple architectures outperform complex ones on small datasets

## Dataset

- **Source:** Structural brain MRI scans in DICOM format
- **Size:** 474 total samples with class imbalance
- **Classes:**
  - AD: Alzheimer's Disease (197 samples)
  - MCI: Mild Cognitive Impairment (204 samples)
  - NC: Normal Control / healthy (73 samples)
- **Class Imbalance:** NC underrepresented (common in clinical datasets)
- **Split:** 329 training / 67 validation / 48 test
- **Preprocessing:**
  - Normalized to [0, 1] range
  - Resized to 160×160×3 pixels
  - Data augmentation: random flips, rotations, zoom

## Model Architectures

### 1. Baseline CNN (Winner) - 5.0M parameters
- 2 convolutional blocks with MaxPooling
- Dropout (0.5) for regularization
- Dense layers for classification
- **Design Philosophy:** Low capacity to prevent overfitting on small dataset

### 2. Deep CNN - 6.5M parameters
- 4 convolutional blocks with BatchNormalization
- More aggressive dropout
- Increased model capacity
- **Result:** Overfitted despite regularization (51% val accuracy)

### 3. Transfer Learning (MobileNetV2)
- Pretrained ImageNet weights (frozen → fine-tuned)
- Custom classification head
- Two-phase training approach
- **Result:** Poor transfer from natural to medical images (47% val accuracy)

## Results Summary

| Model | Parameters | Val Accuracy | Test Accuracy | Macro F1 (Test) |
|-------|-----------|-------------|---------------|-----------------|
| **Baseline CNN** | **5.0M** | **0.957** | **0.854** | **0.855** |
| Deep CNN | 6.5M | 0.511 | - | - |
| Transfer Learning (MobileNetV2) | - | 0.468 | - | - |

## EDA

Comprehensive EDA was performed to understand dataset characteristics:
- No missing data or corrupted DICOM files
- Class imbalance identified (NC: 73 vs AD/MCI: ~200 each)
- Sample MRI visualizations show subtle anatomical differences
- Preprocessing strategy justified based on data characteristics
- F1 score used to handle class imbalance fairly

## Stack used

- **Python 3.13**
- **TensorFlow / Keras** - Deep learning framework
- **NumPy / Pandas** - Data manipulation
- **Matplotlib / Seaborn** - Visualization
- **pydicom** - DICOM file handling
- **scikit-learn** - Metrics and evaluation

## Requirements

```bash
pip install tensorflow numpy pandas matplotlib seaborn pydicom pylibjpeg scikit-learn jupyter
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### Running the Analysis

1. Clone this repository:
```bash
git clone https://github.com/yourusername/alzheimers-mri-detection.git
cd alzheimers-mri-detection
```

2. Place your DICOM data in the `data/` directory with subdirectories `AD/`, `MCI/`, and `NC/`

3. Open and run the Jupyter notebook:
```bash
jupyter notebook alzheimers_disease.ipynb
```

4. The notebook will:
   - Load and preprocess DICOM files
   - Perform exploratory data analysis
   - Train three CNN architectures
   - Evaluate on test set
   - Generate visualizations

## Findings

### 1. Model Complexity Must Match Dataset Size
With only 474 samples, the baseline CNN (5M parameters) outperformed the deep CNN (6.5M parameters) by a massive margin. Additional model capacity led to overfitting rather than improved performance—even with batch normalization and dropout.

### 2. Domain-Specific Training Beats Transfer Learning
ImageNet features (edges, textures, object shapes) do not transfer effectively to medical imaging. Task-specific architectures trained from scratch (95.7% val accuracy) dramatically outperformed pretrained models (46.8% val accuracy).

### 3. CNNs Learn Spatial Patterns, Not Just Intensity
The model successfully captured anatomical spatial features rather than relying on simple brightness differences, achieving 85.4% test accuracy despite limited training data.

## Future Work

1. **3D CNNs:** Process volumetric data instead of 2D slices to capture spatial continuity
2. **Larger Dataset:** Collect more samples to enable deeper architectures
3. **Aggressive Augmentation:** Elastic deformations, intensity shifts to simulate MRI variability
4. **Extended Metrics:** ROC curves, per-class precision-recall, calibration analysis
5. **Domain-Specific Transfer Learning:** Pretrain on other medical imaging datasets

## References

- Deep Learning Book (Goodfellow, Bengio, Courville)
- Course materials: CSCA 5642 Introduction to Deep Learning
- Keras Documentation

## Assignment Deliverables

- **Deliverable 1:** Comprehensive Jupyter notebook with problem description, EDA, model analysis, and conclusions
- **Deliverable 2:** Video presentation (5-10 minutes) covering problem, methods, and results
