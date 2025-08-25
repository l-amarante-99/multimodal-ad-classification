import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Optional

def _build_preprocessor(impute_strategy="mean"):
    return Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler", StandardScaler())
    ])

class ClinicalDataModule(pl.LightningDataModule):
    """
    LightningDataModule supporting optional k-fold cross-validation.
    """
    def __init__(
        self,
        data_path: str,
        target_col: str,
        batch_size: int = 32,
        impute_strategy: str = "mean",
        val_size: float = 0.2,
        random_state: int = 42,
        k_folds: Optional[int] = None,
        fold_idx: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.target_col = target_col
        self.batch_size = batch_size
        self.impute_strategy = impute_strategy
        self.val_size = val_size
        self.random_state = random_state
        self.k_folds = k_folds
        self.fold_idx = fold_idx
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.preprocessor = _build_preprocessor(impute_strategy)

    def _load_and_validate_data(self):
        """Load data and perform basic validation."""
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file is empty: {self.data_path}")
        
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data. "
                           f"Available columns: {list(df.columns)}")
        
        df = df.drop(columns=["PTID", "RID"], errors="ignore")
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].values
        
        # Basic validation
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, found {len(unique_classes)} classes: {unique_classes}")
        if len(df) < 10:
            raise ValueError(f"Insufficient data: only {len(df)} samples found")
        
        print(f"Class distribution: {dict(zip(unique_classes, counts))}")
        return X, y

    def setup(self, stage=None):
        """Setup data splits."""
        _ = stage 
        X, y = self._load_and_validate_data()

        if self.k_folds and self.fold_idx is not None:
            if not (0 <= self.fold_idx < self.k_folds):
                raise ValueError(f"fold_idx must be in [0, {self.k_folds}), got {self.fold_idx}")
            
            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            train_idx, val_idx = list(skf.split(X, y))[self.fold_idx]
            X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
            self.y_train, self.y_val = y[train_idx], y[val_idx]
        else:
            X_train_raw, X_val_raw, self.y_train, self.y_val = train_test_split(
                X, y, test_size=self.val_size, stratify=y, random_state=self.random_state
            )
      
        self.preprocessor = _build_preprocessor(self.impute_strategy)
        self.X_train = self.preprocessor.fit_transform(X_train_raw)
        self.X_val = self.preprocessor.transform(X_val_raw)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, 
                         num_workers=self.num_workers, pin_memory=self.pin_memory)

    def train_dataloader(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available.")
        return self._make_loader(self.X_train, self.y_train, shuffle=True)

    def val_dataloader(self):
        if self.X_val is None or self.y_val is None:
            raise ValueError("Validation data not available.")
        return self._make_loader(self.X_val, self.y_val, shuffle=False)