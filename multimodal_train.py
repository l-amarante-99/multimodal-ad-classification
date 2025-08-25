import os
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from multimodal_model import LateFusionModel
from multimodal_data import MultimodalDataModule
import config

pl.seed_everything(config.CV_RANDOM_STATE, workers=True)

def create_dataloader(dataset, batch_size, num_workers, shuffle=False, sampler=None, drop_last=False):
    """Helper function to create DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if num_workers > 0 else False,
        drop_last=drop_last
    )

def setup_training_components(fold):
    """Create logger and callbacks for a training fold"""
    run_group = os.environ.get("WANDB_RUN_GROUP", f"cross-validation-{config.RUN_NUMBER}")
    name_prefix = os.environ.get("WANDB_NAME_PREFIX", f"lr={config.LEARNING_RATE}-bs={config.BATCH_SIZE}")
    
    fold_logger = WandbLogger(
        project=config.WANDB_PROJECT,
        name=f"{name_prefix}-fold-{fold+1}",
        group=run_group,
        tags=["cross_validation", f"fold_{fold+1}", "multimodal"],
        save_dir=config.SAVE_DIR,
    )
    
    early_stopping = EarlyStopping(
        monitor='val_auc', patience=config.PATIENCE, verbose=True, mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc', mode='max', save_top_k=1,
        filename=f'multimodal-fold-{fold+1}-best-{{val_auc:.3f}}',
        verbose=True, dirpath=config.SAVE_DIR,
    )
    
    trainer = pl.Trainer(
        logger=fold_logger,
        max_epochs=config.EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        accumulate_grad_batches=config.ACCUMULATE_GRAD_BATCHES,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        gradient_clip_val=1.0,
        callbacks=[early_stopping, checkpoint_callback],
        default_root_dir=config.SAVE_DIR,
    )
    
    return trainer, fold_logger, checkpoint_callback

def print_fold_summary(fold, train_dataset, val_dataset, label_counts, val_meta):
    """Print fold information"""
    val_counts = Counter(val_meta["Diagnosis_Code"].values)
    print(f"\nFold {fold+1}/{config.N_SPLITS}: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"  Train labels: {dict(label_counts)} | Val labels: {dict(val_counts)}")

def print_cv_results(all_fold_metrics):
    """Print cross-validation results summary"""
    print("CROSS-VALIDATION RESULTS")
    
    for key in all_fold_metrics[0].keys():
        values = [fold[key] for fold in all_fold_metrics]
        mean, std = np.mean(values), np.std(values)
        print(f"{key:12s}: {mean:.4f} ± {std:.4f}")

def print_fold_val_results(fold, val_results):
    """Print validation metrics for a single fold"""
    results = val_results[0]
    metrics_str = " | ".join([f"{k.replace('val_', '')}: {v:.3f}" for k, v in results.items()])
    print(f"  Fold {fold+1} results: {metrics_str}")

# Validate configuration
try:
    config.validate_config()
except FileNotFoundError as e:
    print(f"Configuration validation failed:\n{e}")
    exit(1)

# Create a data module for cross-validation (use full dataset)
dm_temp = MultimodalDataModule(
    merged_csv_path=config.MERGED_CSV,
    data_dir=config.DATA_DIR,
    cache_path=config.CACHE_PATH,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    val_split=0.0,  # No validation split during CV
    test_split=0.0,  # No test split -> use the full dataset for CV
    target_shape=config.TARGET_SHAPE,
)

dm_temp.setup()

cv_metadata = dm_temp.df_matched
clinical_features = dm_temp.clinical_features
input_dim = len(clinical_features)

labels = np.array(cv_metadata["Diagnosis_Code"].values)

skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.CV_RANDOM_STATE)

all_fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(cv_metadata, labels)):
    train_meta = cv_metadata.iloc[train_idx].reset_index(drop=True)
    val_meta = cv_metadata.iloc[val_idx].reset_index(drop=True)
    train_dataset, val_dataset = dm_temp.create_cv_datasets(
        train_meta, val_meta, clinical_features, config.TARGET_SHAPE
    )
    
    train_labels = train_meta["Diagnosis_Code"].values
    label_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / max(count, 1) for cls, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print_fold_summary(fold, train_dataset, val_dataset, label_counts, val_meta)

    train_loader = create_dataloader(
        train_dataset, config.BATCH_SIZE, config.NUM_WORKERS,
        shuffle=False, sampler=train_sampler, drop_last=True
    )
    val_loader = create_dataloader(
        val_dataset, config.BATCH_SIZE, config.NUM_WORKERS,
        shuffle=False, drop_last=False
    )
    
    model = LateFusionModel(
        input_dim=input_dim,
        clinical_feature_dim=config.CLINICAL_FEATURE_DIM,
        mri_feature_dim=config.MRI_FEATURE_DIM,
        fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        dropout_prob=config.DROPOUT_PROB
    )
    
    trainer, fold_logger, checkpoint_callback = setup_training_components(fold)
    trainer.fit(model, train_loader, val_loader)
    
    val_results = trainer.validate(dataloaders=val_loader, ckpt_path=checkpoint_callback.best_model_path, verbose=False)
    print_fold_val_results(fold, val_results)
    
    all_fold_metrics.append(val_results[0])
    fold_logger.experiment.finish()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # clean-up GPU memory

if not all_fold_metrics:
    print("No fold metrics collected!")
    exit(1)

print_cv_results(all_fold_metrics)

try:
    run_group = os.environ.get("WANDB_RUN_GROUP", f"cv-summary-{config.RUN_NUMBER}")
    name_prefix = os.environ.get("WANDB_NAME_PREFIX", f"bs={config.BATCH_SIZE}-epochs={config.EPOCHS}-lr={config.LEARNING_RATE}-acc={config.ACCUMULATE_GRAD_BATCHES}")
    
    summary_logger = WandbLogger(
        project=config.WANDB_PROJECT,
        name=f"multimodal-cv-summary-{name_prefix}",
        group=run_group,
        tags=["summary", "cross_validation", "multimodal"],
        save_dir=config.SAVE_DIR,
    )
    
    for key in all_fold_metrics[0].keys():
        values = [fold[key] for fold in all_fold_metrics]
        mean, std = float(np.mean(values)), float(np.std(values))
        summary_logger.log_metrics({f"cv_{key}_mean": mean, f"cv_{key}_std": std})
        
        for fold_idx, value in enumerate(values):
            summary_logger.log_metrics({f"fold_{fold_idx+1}_{key}": float(value)})
    
    summary_logger.experiment.finish()
    print("CV summary logged to W&B")
    
except Exception as e:
    print(f"W&B logging failed: {e}")
