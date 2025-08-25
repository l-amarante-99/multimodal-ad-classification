import numpy as np
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from clinical_data import ClinicalDataModule
from clinical_model import ClinicalMLP
from config import config

def setup_training_environment():
    """Setup training environment and optimizations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    seed_everything(config.random_state, workers=True)

def train_fold(fold: int, group_name: str):
    """Train a single fold and return validation metrics."""
    print(f"\n--- Starting Fold {fold + 1}/{config.n_splits} ---")

    dm = ClinicalDataModule(
        data_path=config.data_path, 
        target_col=config.target_col,
        batch_size=config.batch_size, 
        impute_strategy=config.impute_strategy,
        val_size=config.val_size, 
        random_state=config.random_state,
        k_folds=config.n_splits, 
        fold_idx=fold,
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory,
    )
    dm.setup(stage='fit')

    print(f"Training data shape: {dm.X_train.shape if dm.X_train is not None else 'None'}")
    print(f"Validation data shape: {dm.X_val.shape if dm.X_val is not None else 'None'}")
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty!")

    pos_weight = (len(dm.y_train) - dm.y_train.sum()) / dm.y_train.sum() if dm.y_train.sum() > 0 else 1.0
    print(f"Positive class weight: {pos_weight:.4f}")

    assert config.hidden_dims is not None and config.dropout_rates is not None, "Architecture not initialized"
    model = ClinicalMLP(
        input_dim=dm.X_train.shape[1], 
        hidden_dims=config.hidden_dims, 
        dropout_rates=config.dropout_rates,
        lr=config.learning_rate,
        class_weights=[1.0, pos_weight], 
        weight_decay=config.weight_decay,
    )
    
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename=f'best-fold-{fold}'),
        EarlyStopping(monitor="val_loss", patience=config.patience, mode="min", verbose=True)
    ]

    wandb_config = {
        "fold": fold,
        "batch_size": config.batch_size, "lr": config.learning_rate,
        "val_size": config.val_size, "n_splits": config.n_splits,
        "epochs": config.max_epochs, "patience": config.patience,
        "seed": config.random_state, "weight_decay": config.weight_decay,
        "hidden_dims": config.hidden_dims, "dropout_rates": config.dropout_rates,
        "pos_weight": pos_weight,
    }
    
    wandb_logger = WandbLogger(
        project=config.project_name,
        group=group_name,
        name=f"fold-{fold+1}-batch-size={config.batch_size}-lr={config.learning_rate}-patience={config.patience}",
        config=wandb_config,
    )

    trainer = Trainer(
        max_epochs=config.max_epochs, callbacks=callbacks, deterministic=True,
        accelerator="auto", devices=1, log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        gradient_clip_val=1.0, enable_model_summary=False,
        logger=wandb_logger,
    )
    
    print(f"Starting training with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    trainer.fit(model, train_loader, val_loader)

    best_model = ClinicalMLP.load_from_checkpoint(callbacks[0].best_model_path)
    val_results = trainer.validate(best_model, dataloaders=dm.val_dataloader())[0]

    wandb_logger.experiment.finish()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {k.replace('val_', ''): v for k, v in val_results.items() if k.startswith('val_')}

def log_cv_summary(metrics: dict, group_name: str):
    """Log cross-validation summary to W&B."""
    wandb_config = {
        "batch_size": config.batch_size, "lr": config.learning_rate,
        "val_size": config.val_size, "n_splits": config.n_splits,
        "epochs": config.max_epochs, "patience": config.patience,
        "seed": config.random_state, "weight_decay": config.weight_decay,
        "hidden_dims": config.hidden_dims, "dropout_rates": config.dropout_rates,
        "experiment_type": "cv_summary",
    }
    
    summary_logger = WandbLogger(
        project=config.project_name, group=group_name,
        name="cv-summary", config=wandb_config,
    )
    
    # Calculate CV statistics
    cv_stats = {
        f"cv_{metric}_{stat}": func(metrics[metric])
        for metric in ['acc', 'f1', 'auc']
        for stat, func in [('mean', np.mean), ('std', np.std)]
    }

    summary_logger.experiment.summary.update(cv_stats)
    summary_logger.log_metrics(cv_stats)

    print("\n--- Cross-Validation Results ---")
    for metric in ['acc', 'f1', 'auc']:
        print(f"CV {metric.upper()}: {cv_stats[f'cv_{metric}_mean']:.4f} ± {cv_stats[f'cv_{metric}_std']:.4f}")
    
    summary_logger.experiment.finish()

def main():
    """Main training function."""
    try:
        setup_training_environment()
        group_name = f"cv-experiment-{int(__import__('time').time())}"
        metrics = {"acc": [], "f1": [], "auc": []}

        # Cross-validation training
        for fold in range(config.n_splits):
            fold_metrics = train_fold(fold, group_name)
            for metric in metrics:
                metrics[metric].append(fold_metrics[metric])
            
            print(f"Fold {fold + 1}: " + ", ".join([f"val_{k}={v:.4f}" for k, v in fold_metrics.items()]))

        log_cv_summary(metrics, group_name)

    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
