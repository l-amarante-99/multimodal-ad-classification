# Multimodal Alzheimer's Disease Classification

This repository contains the source code for my bachelor thesis on **Alzheimer's disease classification** using multimodal deep learning.The project integrates **MRI data** and **clinical tabular data** from the ADNI dataset, comparing unimodal baselines with a late-fusion multimodal model.

## Features
- **Multimodal Deep Learning**: Late-fusion architecture combining 3D CNN for MRI data and MLP for clinical data
- **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation
- **Experiment Tracking**: Weights & Biases (wandb) integration for comprehensive logging
- **Hyperparameter Optimization**: Automated hyperparameter sweeps
- **Data Augmentation**: 3D augmentations for MRI data (flip, affine, noise)
- **GPU Optimization**: PyTorch Lightning with gradient accumulation and efficient data loading

## Project Structure

```
├── Imaging/                   # MRI data processing and modeling
│   ├── img_model.py           # 3D ResNet18 architecture (MONAI)
│   ├── img_data.py            # MRI data loading and preprocessing
│   ├── img_train.py           # Training script for MRI-only baseline
│   └── config.py              # Imaging-specific configuration
├── Clinical/                  # Clinical tabular data processing and modeling
│   ├── clinical_model.py      # Multi-layer perceptron for clinical data
│   ├── clinical_data.py       # Clinical data preprocessing
│   ├── clinical_train.py      # Training script for clinical-only baseline
│   └── config.py              # Clinical-specific configuration
├── Pre-processing/            # Data preparation and analysis
│   ├── create_clinical_dataset.ipynb
│   ├── create_meta_csv.ipynb
│   ├── create_multimodal_csv.ipynb
│   ├── demographics.ipynb
│   └── find_nii_for_subject_and_date.py
├── multimodal_model.py         # Late-fusion multimodal architecture
├── multimodal_data.py          # Multimodal data module
├── multimodal_train.py         # Main training script with cross-validation
├── multimodal_sweep.py         # Hyperparameter sweep automation
└── config.py                   # Main configuration file
```

## Dataset

- **ADNI (Alzheimer's Disease Neuroimaging Initiative)**
- MRI scans: T1-weighted structural images, preprocessed to (96, 96, 96) voxels
- Clinical data: Demographics, cognitive assessments, biomarkers (14 features)
- Binary classification: Cognitive Normal (CN) vs Alzheimer's Disease (AD)

## Models

### Unimodal Baselines
- **MRI-only**: 3D ResNet18 using MONAI framework
- **Clinical-only**: Multi-layer perceptron with batch normalization and dropout

### Multimodal Architecture
- **Late-fusion model**: Combines features from both modalities
- Feature extraction: Clinical MLP (32-dim) + 3D ResNet (512-dim)
- Fusion: LayerNorm + fully connected layers with dropout
- Output: Single binary classification head

## Key Technologies

- **PyTorch Lightning**: Training framework with automatic logging and checkpointing
- **MONAI**: Medical imaging framework for 3D data processing
- **Weights & Biases**: Experiment tracking and hyperparameter optimization
- **scikit-learn**: Cross-validation and data splitting
- **Environment variable configuration**: Flexible hyperparameter management

## Training Features

- **5-fold stratified cross-validation** for robust evaluation
- **Early stopping** with patience-based validation monitoring
- **Model checkpointing** with best validation AUC saving
- **Weighted sampling** for class imbalance handling
- **Gradient accumulation** for effective large batch training
- **Comprehensive metrics**: Accuracy, F1-score, AUC-ROC

## Usage

### Training the Multimodal Model
```bash
python multimodal_train.py
```

### Hyperparameter Sweep
```bash
python multimodal_sweep.py
```

### Training Individual Baselines
```bash
# MRI-only baseline
python Imaging/img_train.py

# Clinical-only baseline  
python Clinical/clinical_train.py
```

## Configuration

The project uses environment variables for flexible configuration:
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `BATCH_SIZE`: Batch size (default: 4)
- `DROPOUT_PROB`: Dropout probability (default: 0.3)
- `FUSION_HIDDEN_DIM`: Fusion layer dimension (default: 128)
- `ACCUMULATE_GRAD_BATCHES`: Gradient accumulation steps (default: 32)

See `config.py` for all available parameters.

## Requirements

- PyTorch & PyTorch Lightning
- MONAI
- Weights & Biases
- scikit-learn
- NumPy, pandas
- CUDA-compatible GPU recommended

## License

MIT License - see LICENSE file for details.


