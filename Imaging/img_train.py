import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from img_model import ResNet3D
from img_data import MRIDataModule, MRIDataset

from config import (
    META_CSV, DATA_DIR, CACHE_PATH, BATCH_SIZE, NUM_WORKERS, 
    N_SPLITS, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, WANDB_PROJECT, LOG_EVERY_N_STEPS, 
    ACCUMULATE_GRAD_BATCHES, CV_RANDOM_STATE, PIN_MEMORY, 
    PERSISTENT_WORKERS, TARGET_SHAPE, validate_config
)

pl.seed_everything(CV_RANDOM_STATE, workers=True)

def create_dataloader(dataset, batch_size, num_workers, shuffle=False, sampler=None, drop_last=False):
    """Helper function to create DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=drop_last
    )

def setup_training_components(fold):
    """Create logger and callbacks for a training fold"""
    wandb_config = {
        "fold": fold + 1,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "epochs": EPOCHS,
        "num_workers": NUM_WORKERS,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
        "n_splits": N_SPLITS,
        "cv_random_state": CV_RANDOM_STATE,
    }
    
    fold_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=f"fold-{fold+1}-lr={LEARNING_RATE}-bs={BATCH_SIZE}-wd={WEIGHT_DECAY}-acc={ACCUMULATE_GRAD_BATCHES}",
        group="cross-validation",
        tags=["cross_validation", f"fold_{fold+1}"],
        config=wandb_config,
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        mode='max',
        save_top_k=1,
        filename=f'fold-{fold+1}-best-{{val_auc:.3f}}',
        verbose=True
    )
    
    trainer = pl.Trainer(
        logger=fold_logger,
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        enable_checkpointing=True,
        enable_progress_bar=True,
        precision="16-mixed",
        deterministic=False,
        callbacks=[early_stopping, checkpoint_callback],
    )
    
    return trainer, checkpoint_callback

def print_fold_val_results(fold, val_results):
    """Print validation metrics for a single fold"""
    print(f"Fold {fold+1} - Validation metrics:")
    for key, value in val_results[0].items():
        print(f"  {key}: {value:.4f}")

try:
    validate_config()
except FileNotFoundError as e:
    print(f"Configuration validation failed:\n{e}")
    exit(1)

dm_temp = MRIDataModule(
    meta_csv=META_CSV,
    data_dir=DATA_DIR,
    cache_path=CACHE_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    val_split=0.0,  
    test_split=0.0,  
    target_shape=TARGET_SHAPE,
)

dm_temp.setup()

cv_metadata = dm_temp.df_matched

labels = np.array(cv_metadata["Diagnosis_Code"].values)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

all_fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(labels, labels)):
    train_meta = cv_metadata.iloc[train_idx].reset_index(drop=True)
    val_meta = cv_metadata.iloc[val_idx].reset_index(drop=True)

    train_dataset = MRIDataset(
        meta_df=train_meta,
        data_dir=DATA_DIR,
        cache_path=CACHE_PATH,
        target_shape=TARGET_SHAPE,
        training=True 
    )
    
    val_dataset = MRIDataset(
        meta_df=val_meta,
        data_dir=DATA_DIR,
        cache_path=CACHE_PATH,
        target_shape=TARGET_SHAPE,
        training=False
    )

    train_label_counts = Counter(train_meta["Diagnosis_Code"])
    class_weights = {cls: 1.0 / max(count, 1) for cls, count in train_label_counts.items()}
    sample_weights = [class_weights[label] for label in train_meta["Diagnosis_Code"]]
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = create_dataloader(train_dataset, BATCH_SIZE, NUM_WORKERS, sampler=train_sampler, drop_last=True)
    val_loader = create_dataloader(val_dataset, BATCH_SIZE, NUM_WORKERS)

    model = ResNet3D(in_channels=1, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trainer, checkpoint_callback = setup_training_components(fold)
    
    try:
        trainer.fit(model, train_loader, val_loader)
        
        best_model = ResNet3D.load_from_checkpoint(checkpoint_callback.best_model_path)
        val_results = trainer.test(best_model, dataloaders=val_loader, verbose=False)
        all_fold_metrics.append(val_results[0])
        
        print_fold_val_results(fold, val_results)
        
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()
        
        # Clean up to free memory
        del model, best_model, checkpoint_callback, train_loader, val_loader, train_dataset, val_dataset, train_sampler
            
    except Exception as e:
        print(f"Error in fold {fold+1}: {str(e)}")
        try:
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.finish()
                print(f"Cleaned up W&B run for failed fold {fold+1}")
        except Exception:
            pass  # If cleanup fails, just continue
        continue

    # Final cleanup of trainer after successful completion
    finally:
        if 'trainer' in locals():
            del trainer

if not all_fold_metrics:
    print("No successful folds completed!")
    exit(1)

print("CROSS-VALIDATION RESULTS SUMMARY")
print("\nValidation Results (CV folds):")

metrics_keys = all_fold_metrics[0].keys()

for key in metrics_keys:
    values = [fold[key] for fold in all_fold_metrics]
    mean = np.mean(values)
    std = np.std(values)
    print(f"{key}: {mean:.4f} ± {std:.4f}")

print("\nLogging cross-validation summary to W&B...")
try:
    if wandb.run is not None:
        wandb.finish()
    
    summary_wandb_config = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "epochs": EPOCHS,
        "num_workers": NUM_WORKERS,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
        "n_splits": N_SPLITS,
        "cv_random_state": CV_RANDOM_STATE,
        "total_samples": len(cv_metadata),
        "experiment_type": "cv_summary",
    }
    
    # Initialize a separate W&B run for the summary
    wandb.init(
        project=WANDB_PROJECT,
        name=f"summary-{N_SPLITS}-folds-batches={BATCH_SIZE}-lr={LEARNING_RATE}-wd={WEIGHT_DECAY}-patience={PATIENCE}",
        group="cross-validation",
        tags=["cross_validation", "summary", "final_results"],
        config=summary_wandb_config
    )

    cv_stats = {}
    for metric in metrics_keys:
        values = [fold[metric] for fold in all_fold_metrics]
        mean_val = float(np.mean(values))  
        std_val = float(np.std(values))  
        cv_stats[f"cv_{metric}_mean"] = mean_val
        cv_stats[f"cv_{metric}_std"] = std_val
        print(f"  {metric}: mean={mean_val:.4f}, std={std_val:.4f}")
        
    wandb.log(cv_stats)
   
    columns = ["fold"] + list(metrics_keys)
    fold_data = []
    for i, fold_metrics in enumerate(all_fold_metrics):
        row = [i + 1]
        for metric in metrics_keys:
            row.append(fold_metrics[metric])
        fold_data.append(row)
    
    fold_table = wandb.Table(data=fold_data, columns=columns)
    wandb.log({"cv_fold_results": fold_table})
    
    wandb.finish()
    
except Exception as e:
    print(f"\nError: Failed to log CV summary to W&B: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()