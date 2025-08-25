# Clinical MLP Configuration
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    data_path: str = "/home/lude14/bachelorarbeit/MRI_CNN/clinical-maddi-3.csv"
    target_col: str = "label"
    
    # Model hyperparameters
    batch_size: int = 32  
    learning_rate: float = 1e-4  
    weight_decay: float = 1e-5
    
    # Model architecture
    hidden_dims: Optional[List[int]] = None 
    dropout_rates: Optional[List[float]] = None  
    
    # Training parameters
    max_epochs: int = 100
    patience: int = 25  
    
    # Cross-validation
    n_splits: int = 5
    val_size: float = 0.2
    
    # Data preprocessing
    impute_strategy: str = "mean"
    
    # Other
    random_state: int = 42
    num_workers: int = 4  
    pin_memory: bool = True 

    # Weights & Biases
    project_name: str = "clinical-mlp-cross-validation"
    
    def __post_init__(self):
        # Initialize model architecture if not set
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
        if self.dropout_rates is None:
            self.dropout_rates = [0.05, 0.05, 0.05]  
  
        env_data_path = os.environ.get('CLINICAL_DATA_PATH')
        if env_data_path:
            self.data_path = env_data_path
            
        # Create data directory if it doesn't exist
        if self.data_path:
            data_dir = os.path.dirname(self.data_path)
            if data_dir:
                os.makedirs(data_dir, exist_ok=True)

        validations = [
            (self.batch_size <= 0, f"batch_size must be positive, got {self.batch_size}"),
            (self.learning_rate <= 0, f"learning_rate must be positive, got {self.learning_rate}"),
            (not (0 < self.val_size < 1), f"val_size must be between 0 and 1, got {self.val_size}"),
            (len(self.hidden_dims) != len(self.dropout_rates), 
             f"hidden_dims and dropout_rates must have same length, got {len(self.hidden_dims)} vs {len(self.dropout_rates)}"),
            (self.n_splits < 2, f"n_splits must be >= 2, got {self.n_splits}")
        ]
        
        for condition, message in validations:
            if condition:
                raise ValueError(message)

config = Config()
