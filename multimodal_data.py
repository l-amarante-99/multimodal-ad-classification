import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from collections import Counter
from monai.transforms import Resize, Compose, RandFlip, RandGaussianNoise, RandAffine
import nibabel as nib
import config

class MultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, clinical_features: list, features_scaled: np.ndarray, target_shape=(96, 96, 96), training=False):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.target_shape = target_shape
        self.features_scaled = features_scaled

        self.resize = Resize(spatial_size=target_shape)

        self.augment = Compose([
            RandFlip(spatial_axis=[0], prob=0.5),
            RandAffine(
                rotate_range=(0.1, 0.05, 0.05),
                translate_range=(4, 4, 4),
                scale_range=(0.05, 0.05, 0.05),
                padding_mode="border",
                prob=0.4
            ),
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.1)
        ]) if training else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nii_path = row["NII_PATH"]
        
        try:
            # Load NIfTI and canonicalize orientation
            nii_img = nib.load(nii_path)
            nii_canonical = nib.as_closest_canonical(nii_img)
            vol = nii_canonical.get_fdata().astype(np.float32)
            
            # z-score normalization on non-background voxels only
            nonzero_mask = vol != 0
            if np.any(nonzero_mask):
                nonzero_voxels = vol[nonzero_mask]
                mean_val = np.mean(nonzero_voxels)
                std_val = np.std(nonzero_voxels)
                vol = (vol - mean_val) / (std_val + 1e-8)
            else:
                vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-8)
            
            vol = vol.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading NII file {nii_path}: {str(e)}")
        
        vol_tensor = torch.from_numpy(vol)[None, ...]  # shape: (1, D, H, W)
        vol_tensor = self.resize(vol_tensor)

        if self.augment:
            vol_tensor = self.augment(vol_tensor)

        clinical = torch.tensor(self.features_scaled[idx], dtype=torch.float32)
        label = int(row['Diagnosis_Code'])

        return (clinical, vol_tensor), label


class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, merged_csv_path, data_dir, cache_path, batch_size=4, num_workers=4, val_split=0.2, test_split=0.1, target_shape=(96, 96, 96)):
        super().__init__()
        self.merged_csv_path = merged_csv_path
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.target_shape = target_shape
        self.imputer = None
        self.scaler = None

    def _preprocess_clinical_features(self, train_df, val_df=None, test_df=None):
        """Fit preprocessing on train data and apply to all splits"""
        # Fit imputer and scaler on training data only
        train_features = train_df[self.clinical_features].values.astype(np.float32)
        
        self.imputer = SimpleImputer(strategy="mean")
        train_features_imputed = self.imputer.fit_transform(train_features)
        
        self.scaler = StandardScaler()
        train_features_scaled = self.scaler.fit_transform(train_features_imputed)

        processed_features = {"train": train_features_scaled}
        
        if val_df is not None:
            val_features = val_df[self.clinical_features].values.astype(np.float32)
            val_features_imputed = self.imputer.transform(val_features)
            val_features_scaled = self.scaler.transform(val_features_imputed)
            processed_features["val"] = np.asarray(val_features_scaled)
            
        if test_df is not None:
            test_features = test_df[self.clinical_features].values.astype(np.float32)
            test_features_imputed = self.imputer.transform(test_features)
            test_features_scaled = self.scaler.transform(test_features_imputed)
            processed_features["test"] = np.asarray(test_features_scaled)
            
        return processed_features

    def preprocess_cv_fold(self, train_df, val_df):
        """Preprocess clinical features for a single CV fold"""
        return self._preprocess_clinical_features(train_df, val_df)
    
    def create_cv_datasets(self, train_df, val_df, clinical_features, target_shape):
        """Create train and validation datasets for CV with proper preprocessing"""
        processed_features = self.preprocess_cv_fold(train_df, val_df)
        
        train_dataset = MultimodalDataset(
            train_df, clinical_features, processed_features["train"], target_shape, training=True
        )
        val_dataset = MultimodalDataset(
            val_df, clinical_features, processed_features["val"], target_shape, training=False
        )
        
        return train_dataset, val_dataset

    def setup(self, stage=None):
        df = pd.read_csv(self.merged_csv_path)
        df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
        df = df.sort_values('EXAMDATE').groupby('PTID').tail(1)

        try:
            with open(self.cache_path, "rb") as f:
                nii_cache = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"NIfTI cache file not found: {self.cache_path}\n"
            )
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(
                f"NIfTI cache file is corrupted or invalid: {self.cache_path}\n"
                f"Error: {e}\n"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error loading NIfTI cache from {self.cache_path}: {e}\n"
            )

        matched_rows = []
        for _, row in df.iterrows():
            ptid_path = os.path.join(self.data_dir, row["PTID"])
            exam_date = row["EXAMDATE"].strftime("%Y-%m-%d")
            cache_key = (ptid_path, exam_date)
            nii_path = nii_cache.get(cache_key)
            if nii_path:
                row = row.copy()
                row["EXAMDATE"] = exam_date
                row["NII_PATH"] = nii_path
                matched_rows.append(row)

        df = pd.DataFrame(matched_rows)
        df = df.dropna(subset=["Diagnosis_Code"])  # Ensure no NaNs in labels

        clinical_features = [col for col in df.columns if col not in ['PTID', 'EXAMDATE', 'Diagnosis_Code', 'NII_PATH']]
        self.clinical_features = clinical_features
        self.df_matched = df
        
        if self.val_split == 0.0 and self.test_split == 0.0: # Cross-validation mode
            self.train_ds = None
            self.val_ds = None
            self.test_ds = None
            return

        train_df, testval_df = train_test_split(df, test_size=self.val_split + self.test_split, stratify=df['Diagnosis_Code'], random_state=42)
        val_df, test_df = train_test_split(testval_df, test_size=self.test_split / (self.val_split + self.test_split), stratify=testval_df['Diagnosis_Code'], random_state=42)

        processed_features = self._preprocess_clinical_features(train_df, val_df, test_df)
        
        self.train_ds = MultimodalDataset(train_df, clinical_features, processed_features["train"], self.target_shape, training=True)
        self.val_ds = MultimodalDataset(val_df, clinical_features, processed_features["val"], self.target_shape, training=False)
        self.test_ds = MultimodalDataset(test_df, clinical_features, processed_features["test"], self.target_shape, training=False)

        train_labels = train_df["Diagnosis_Code"].values
        label_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / max(count, 1) for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        self.train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def train_dataloader(self):
        if self.train_ds is None:
            raise ValueError("train_ds is None.")
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            sampler=self.train_sampler, 
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=config.PERSISTENT_WORKERS if self.num_workers > 0 else False,
            drop_last=True
        )

    def val_dataloader(self):
        if self.val_ds is None:
            raise ValueError("val_ds is None.")
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=config.PERSISTENT_WORKERS if self.num_workers > 0 else False,
            drop_last=False
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise ValueError("test_ds is None.")
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=config.PERSISTENT_WORKERS if self.num_workers > 0 else False,
            drop_last=False
        )
