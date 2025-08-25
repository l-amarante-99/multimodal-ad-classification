import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryAUROC
from Clinical.clinical_model import ClinicalMLP
from Imaging.img_model import ResNet3D

class LateFusionModel(pl.LightningModule):
    def __init__(self, input_dim=50, clinical_feature_dim=32, mri_feature_dim=512, 
                 fusion_hidden_dim=128, learning_rate=1e-4, weight_decay=1e-4, dropout_prob=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.clinical_model = ClinicalMLP(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128, 64, clinical_feature_dim],
            dropout_rates=[0.2, 0.2, 0.1, 0.1, 0.1],
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.mri_model = ResNet3D(in_channels=1, learning_rate=learning_rate, weight_decay=weight_decay)

        clinical_layers = list(self.clinical_model.model.children())[:-1]  # Remove final Linear(32, 1)
        self.clinical_feature_extractor = nn.Sequential(*clinical_layers)

        mri_backbone = self.mri_model.model
        
        # Replace the final classification layer with Identity to get features
        mri_backbone.fc = nn.Identity()
        self.mri_feature_extractor = nn.Sequential(
            mri_backbone,
            nn.Flatten(1)
        )

        self.clinical_norm = nn.LayerNorm(clinical_feature_dim)
        self.mri_norm = nn.LayerNorm(mri_feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(clinical_feature_dim + mri_feature_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_hidden_dim, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self._init_metrics()

    def _init_metrics(self):
        """Initialize all metrics for train, validation, and test."""
        self.train_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.train_auc = BinaryAUROC()
        
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_auc = BinaryAUROC()
        
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_auc = BinaryAUROC()

    def forward(self, clinical_x, mri_x):
        c_feat = self.clinical_feature_extractor(clinical_x)

        m_feat = self.mri_feature_extractor(mri_x)

        c_feat = self.clinical_norm(c_feat)
        m_feat = self.mri_norm(m_feat)

        combined = torch.cat([c_feat, m_feat], dim=1)
        return self.classifier(combined).squeeze(-1)

    def training_step(self, batch, batch_idx):
        (clinical_x, mri_x), y = batch
        logits = self(clinical_x, mri_x)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)
        self.train_auc.update(probs, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (clinical_x, mri_x), y = batch
        logits = self(clinical_x, mri_x)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_auc.update(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        (clinical_x, mri_x), y = batch
        logits = self(clinical_x, mri_x)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_auc.update(probs, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _log_epoch_metrics(self, phase):
        """Log metrics at the end of each epoch"""
        if phase == "train":
            acc = self.train_acc.compute()
            f1 = self.train_f1.compute()
            auc = self.train_auc.compute()
            
            self.train_acc.reset()
            self.train_f1.reset()
            self.train_auc.reset()
        elif phase == "val":
            acc = self.val_acc.compute()
            f1 = self.val_f1.compute()
            auc = self.val_auc.compute()

            self.val_acc.reset()
            self.val_f1.reset()
            self.val_auc.reset()
        elif phase == "test":
            acc = self.test_acc.compute()
            f1 = self.test_f1.compute()
            auc = self.test_auc.compute()

            self.test_acc.reset()
            self.test_f1.reset()
            self.test_auc.reset()
        else:
            raise ValueError(f"Unknown phase: {phase}")

        self.log(f"{phase}_acc", acc, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_f1", f1, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_auc", auc, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.get('learning_rate', 1e-4),
            weight_decay=self.hparams.get('weight_decay', 1e-4)
        )
        return optimizer
