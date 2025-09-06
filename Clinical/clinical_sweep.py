"""
Clinical MLP Hyperparameter Sweep Script

This script performs hyperparameter sweeps for the clinical MLP model,
testing different combinations of learning rates, batch sizes, and architectures.

"""

import os
import subprocess
import itertools
import datetime
import sys
from pathlib import Path

# Grid to try 
LEARNING_RATES = [1e-3, 1e-4]
BATCH_SIZES = [16, 32, 64]
ARCHITECTURES = [[128, 64, 32], [256, 128, 64, 32], [64, 32]]

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "clinical-mlp-cross-validation-sweep")
RUN_GROUP = os.environ.get("WANDB_SWEEP_GROUP") or f"sweep-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
BASE_ENV = os.environ.copy()
BASE_ENV["WANDB_PROJECT"] = WANDB_PROJECT

def run_one(lr, batch_size, architecture):
    """Run one training configuration with specified hyperparameters."""
    env = BASE_ENV.copy()

    env["LEARNING_RATE"] = str(lr)
    env["BATCH_SIZE"] = str(batch_size)
    env["HIDDEN_DIMS"] = ",".join(map(str, architecture))  # Convert list to comma-separated string

    env["WANDB_RUN_GROUP"] = RUN_GROUP
    arch_str = "-".join(map(str, architecture))
    env["WANDB_NAME_PREFIX"] = f"lr={lr}-bs={batch_size}-arch={arch_str}"

    env["RUN_NUMBER"] = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-lr{lr}-bs{batch_size}-{arch_str}"

    print(f"\n Running clinical_train.py with LR={lr} | BATCH_SIZE={batch_size} | ARCH={architecture}")

    result = subprocess.run(
        ["python", "clinical_train.py"], 
        env=env
    )
    
    if result.returncode != 0:
        print(f"Run failed for lr={lr}, batch_size={batch_size}, arch={architecture} (exit code {result.returncode})")
        return False
    return True


if __name__ == "__main__":
    total_runs = len(LEARNING_RATES) * len(BATCH_SIZES) * len(ARCHITECTURES)
    successful_runs = 0
    failed_runs = 0
    
    print(f"Starting clinical MLP hyperparameter sweep with {total_runs} total combinations")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Sweep group: {RUN_GROUP}")
    
    for i, (lr, batch_size, architecture) in enumerate(itertools.product(LEARNING_RATES, BATCH_SIZES, ARCHITECTURES), 1):
        print(f"\n[{i}/{total_runs}] Starting run {i}")
        print(f"Configuration: lr={lr}, batch_size={batch_size}, architecture={architecture}")
        
        success = run_one(lr, batch_size, architecture)
        if success:
            successful_runs += 1
            print(f"Run {i} completed successfully")
        else:
            failed_runs += 1
            print(f"Run {i} failed")
    
    print(f"\nSweep complete. Group: {RUN_GROUP}")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    print(f"Failed runs: {failed_runs}/{total_runs}")
    
    if failed_runs > 0:
        print(f"\nWarning: {failed_runs} runs failed.")
    else:
        print("\nSweep completed successfully!")
    
    if failed_runs > 0:
        sys.exit(1)
