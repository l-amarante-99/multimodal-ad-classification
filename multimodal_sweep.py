import os
import subprocess
import itertools
import uuid

LEARNING_RATES = [1e-4, 5e-5]
BATCH_SIZE = 8  # Fixed batch size! (Set after experimentation)
ACCUMULATE_GRAD_BATCHES = [2, 4]
DROPOUT_PROBS = [0.2, 0.3]
FUSION_HIDDEN_DIMS = [64, 128]

SWEEP_ID = str(uuid.uuid4())[:8]
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "multimodal-alzheimer-cross-validation")
RUN_GROUP = os.environ.get("WANDB_SWEEP_GROUP") or f"quick-sweep-{SWEEP_ID}"
BASE_ENV = os.environ.copy()
BASE_ENV["WANDB_PROJECT"] = WANDB_PROJECT

def run_one(lr, accum, dropout, fusion_dim):
    """Run one hyperparameter combination"""
    env = BASE_ENV.copy()
    env["LEARNING_RATE"] = str(lr)
    env["BATCH_SIZE"] = str(BATCH_SIZE) 
    env["ACCUMULATE_GRAD_BATCHES"] = str(accum)
    env["DROPOUT_PROB"] = str(dropout)
    env["FUSION_HIDDEN_DIM"] = str(fusion_dim)
    env["RUN_NUMBER"] = SWEEP_ID 
    env["WANDB_RUN_GROUP"] = RUN_GROUP  
    env["WANDB_NAME_PREFIX"] = f"lr={lr}-bs={BATCH_SIZE}-acc={accum}-dropout={dropout}-fusion={fusion_dim}"

    print(f"\n Running multimodal_train.py with LR={lr} | BS={BATCH_SIZE} | Accum={accum} | Dropout={dropout} | Fusion={fusion_dim}")
    
    result = subprocess.run(["python", "multimodal_train.py"], env=env)
    if result.returncode != 0:
        print(f"Run failed for lr={lr}, bs={BATCH_SIZE}, accum={accum}, dropout={dropout}, fusion_dim={fusion_dim} (exit code {result.returncode})")
        return False
    return True

if __name__ == "__main__":
    total_runs = len(LEARNING_RATES) * len(ACCUMULATE_GRAD_BATCHES) * len(DROPOUT_PROBS) * len(FUSION_HIDDEN_DIMS)
    successful_runs = 0
    failed_runs = 0
    
    print(f"Starting QUICK multimodal hyperparameter sweep with {total_runs} total combinations")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Batch size: {BATCH_SIZE} (fixed)")
    print(f"Accumulate grad batches: {ACCUMULATE_GRAD_BATCHES}")
    print(f"Dropout probabilities: {DROPOUT_PROBS}")
    print(f"Fusion hidden dimensions: {FUSION_HIDDEN_DIMS}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Sweep group: {RUN_GROUP}")
    print(f"Sweep ID: {SWEEP_ID}")
    
    for i, (lr, accum, dropout, fusion_dim) in enumerate(itertools.product(
        LEARNING_RATES, ACCUMULATE_GRAD_BATCHES, DROPOUT_PROBS, FUSION_HIDDEN_DIMS), 1):
        
        print(f"\n[{i}/{total_runs}] Starting run {i}")
        success = run_one(lr, accum, dropout, fusion_dim)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1

    print(f"\nQuick sweep complete. Group: {RUN_GROUP}")
    print(f"Sweep ID: {SWEEP_ID}")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    print(f"Failed runs: {failed_runs}/{total_runs}")
