import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, WeightedRandomSampler
from monai.transforms import Resize, Compose, RandFlip, RandGaussianNoise, RandAffine
import pytorch_lightning as pl
from collections import Counter
from sklearn.model_selection import train_test_split
from config import TARGET_SHAPE

class MRIDataset(Dataset):
    """
    Loads full 3D MRI volumes using a .nii cache, applies z-score normalization and resizing.
    Applies light augmentation if `training=True`.
    """
    def __init__(self, meta_df: pd.DataFrame, data_dir: str, cache_path: str,
                 target_shape=TARGET_SHAPE, training: bool = False):
        self.meta = meta_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.cache_path = cache_path 
        self.target_shape = target_shape
        self.training = training
        self.resize = Resize(spatial_size=target_shape)

        self.augment = None
        if training:
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
            ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
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

        vol_tensor = torch.from_numpy(vol)[None, ...]  # (1, D, H, W)
        vol_tensor = self.resize(vol_tensor)

        if self.augment:
            vol_tensor = self.augment(vol_tensor)

        label = torch.tensor(int(row["Diagnosis_Code"]), dtype=torch.long)
        return vol_tensor, label

class MRIDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading AD vs CN MRI data from full 3D volumes.
    """
    def __init__(
        self,
        meta_csv: str,
        data_dir: str,
        cache_path: str,
        batch_size: int = 2,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        target_shape=TARGET_SHAPE,
    ):
        super().__init__()
        self.meta_csv = meta_csv
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.target_shape = target_shape

    def setup(self, stage=None):
        full_df = pd.read_csv(
            self.meta_csv,
            usecols=["RID", "PTID", "EXAMDATE", "Diagnosis_Code"],
            dtype={"RID": str, "Diagnosis_Code": int}
        )
        full_df = full_df[full_df["Diagnosis_Code"].isin([0, 2])].copy()
        full_df["Diagnosis_Code"] = full_df["Diagnosis_Code"].map({0: 0, 2: 1})
        full_df["EXAMDATE"] = pd.to_datetime(full_df["EXAMDATE"])

        latest_labels = full_df.sort_values("EXAMDATE").groupby("PTID").tail(1)

        try:
            with open(self.cache_path, "rb") as f:
                nii_cache = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading cache file {self.cache_path}: {str(e)}")

        matched_rows = []
        for _, row in latest_labels.iterrows():
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
        self.df_matched = df 

        if self.val_split == 0.0 and self.test_split == 0.0:
            # Cross-validation mode
            self.train_ds = None
            self.val_ds = None
            self.test_ds = None
            self.train_sampler = None
            
            print("Full dataset mode (for cross-validation):")
            print(f"  Total samples: {len(df)}")
            label_counts = Counter(df["Diagnosis_Code"])
            print("Label distribution:", label_counts)
            return

        if self.val_split == 0.0 and self.test_split > 0.0:
            train_df, test_df = train_test_split(
                df, test_size=self.test_split,
                stratify=df["Diagnosis_Code"], random_state=42
            )
            
            self.train_ds = MRIDataset(train_df, self.data_dir, self.cache_path, target_shape=self.target_shape, training=True)
            self.val_ds = None
            self.test_ds = MRIDataset(test_df, self.data_dir, self.cache_path, target_shape=self.target_shape, training=False)
            self.train_sampler = None
            
            print("Cross-validation mode with held-out test set:")
            print(f"  Train (for CV): {len(train_df)} | Held-out Test: {len(test_df)}")
            train_labels = Counter(train_df["Diagnosis_Code"])
            test_labels = Counter(test_df["Diagnosis_Code"])
            print("Train labels:", train_labels)
            print("Test labels:", test_labels)
            return

        # Normal train/val/test split
        train_df, testval_df = train_test_split(
            df, test_size=self.val_split + self.test_split,
            stratify=df["Diagnosis_Code"], random_state=42
        )
        val_df, test_df = train_test_split(
            testval_df,
            test_size=self.test_split / (self.val_split + self.test_split),
            stratify=testval_df["Diagnosis_Code"], random_state=42
        )

        self.train_ds = MRIDataset(train_df, self.data_dir, self.cache_path, target_shape=self.target_shape, training=True)
        self.val_ds   = MRIDataset(val_df,   self.data_dir, self.cache_path, target_shape=self.target_shape, training=False)
        self.test_ds  = MRIDataset(test_df,  self.data_dir, self.cache_path, target_shape=self.target_shape, training=False)

        label_counts = Counter(train_df["Diagnosis_Code"])
        class_weights = {cls: 1.0 / max(count, 1) for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in train_df["Diagnosis_Code"]]
        self.train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        print("Final split sizes:")
        print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        print("Train labels:", label_counts)
        print("Val labels:  ", Counter(val_df["Diagnosis_Code"]))
        print("Test labels: ", Counter(test_df["Diagnosis_Code"]))
