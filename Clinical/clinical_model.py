import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
from typing import Optional, List

class ClinicalMLP(pl.LightningModule):
    """MLP for binary classification on clinical data."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rates: List[float],
                 lr: float = 1e-4, class_weights: Optional[List[float]] = None, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()

        # Build model architecture from config
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        all_dropout_rates = [0] + dropout_rates + [0] # No dropout for input and output layers
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No BatchNorm/ReLU/Dropout for output layer
                layers.extend([
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(all_dropout_rates[i + 1])
                ])
        
        self.model = nn.Sequential(*layers)

        if class_weights:
            pos_weight = torch.tensor(class_weights[1], dtype=torch.float32)
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.register_buffer('pos_weight', None)

        self._init_metrics()

    def _init_metrics(self):
        """Initialize all metrics for train and validation."""
        metrics = ['acc', 'f1', 'auc']
        metric_classes = [BinaryAccuracy, BinaryF1Score, BinaryAUROC]
        
        for stage in ['train', 'val']:
            for metric_name, metric_class in zip(metrics, metric_classes):
                setattr(self, f'{stage}_{metric_name}', metric_class())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x).squeeze(-1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss."""
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight
        )

    def training_step(self, batch, batch_idx):
        """Training step with explicit metric logging."""
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        y_long = y.long()

        self.train_acc.update(preds, y_long)
        self.train_f1.update(preds, y_long)
        self.train_auc.update(probs, y_long)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with explicit metric logging."""
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        y_long = y.long()

        self.val_acc.update(preds, y_long)
        self.val_f1.update(preds, y_long)
        self.val_auc.update(probs, y_long)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_training_epoch_end(self):
        """Log training metrics at epoch end."""
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        train_auc = self.train_auc.compute()
        
        self.log('train_acc', train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1', train_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_auc', train_auc, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auc.reset()

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        val_auc = self.val_auc.compute()
        
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1', val_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_auc', val_auc, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
