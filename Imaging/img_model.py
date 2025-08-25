import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import resnet
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

class ResNet3D(pl.LightningModule):
    def __init__(self, in_channels=1, learning_rate=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 3D ResNet model from MONAI
        self.model = resnet.resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=2,
        )

        self.train_accuracy = BinaryAccuracy()
        self.train_f1_score = BinaryF1Score()
        self.train_auc_score = BinaryAUROC()
        
        self.val_accuracy = BinaryAccuracy()
        self.val_f1_score = BinaryF1Score()
        self.val_auc_score = BinaryAUROC()
        
        self.test_accuracy = BinaryAccuracy()
        self.test_f1_score = BinaryF1Score()
        self.test_auc_score = BinaryAUROC()

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, phase="train"):
        """Shared logic for train/val/test steps"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1 if phase == "train" else 0.0)

        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)[:, 1]

        if phase == "train":
            self.train_accuracy.update(preds, y)
            self.train_f1_score.update(preds, y)
            self.train_auc_score.update(probs, y)
        elif phase == "val":
            self.val_accuracy.update(preds, y)
            self.val_f1_score.update(preds, y)
            self.val_auc_score.update(probs, y)
        elif phase == "test":
            self.test_accuracy.update(preds, y)
            self.test_f1_score.update(preds, y)
            self.test_auc_score.update(probs, y)

        self.log(f"{phase}_loss", loss, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def _log_epoch_metrics(self, phase):
        """Log metrics at the end of each epoch"""
        if phase == "train":
            acc = self.train_accuracy.compute()
            f1 = self.train_f1_score.compute()
            auc = self.train_auc_score.compute()
            
            self.train_accuracy.reset()
            self.train_f1_score.reset()
            self.train_auc_score.reset()
        elif phase == "val":
            acc = self.val_accuracy.compute()
            f1 = self.val_f1_score.compute()
            auc = self.val_auc_score.compute()
    
            self.val_accuracy.reset()
            self.val_f1_score.reset()
            self.val_auc_score.reset()
        elif phase == "test":
            acc = self.test_accuracy.compute()
            f1 = self.test_f1_score.compute()
            auc = self.test_auc_score.compute()
 
            self.test_accuracy.reset()
            self.test_f1_score.reset()
            self.test_auc_score.reset()
        else:
            raise ValueError(f"Unknown phase: {phase}")

        self.log(f"{phase}_acc", acc, prog_bar=True)
        self.log(f"{phase}_f1", f1, prog_bar=True)
        self.log(f"{phase}_auc", auc, prog_bar=True)

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
