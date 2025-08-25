import os
import subprocess
import itertools
import datetime

LEARNING_RATES = [1e-3, 1e-4]
ACCUM_STEPS    = [2, 4, 8, 16]

WANDB_PROJECT   = os.environ.get("WANDB_PROJECT", "mri-resnet-3D-Cross-Validation") 
RUN_GROUP       = os.environ.get("WANDB_SWEEP_GROUP") or f"sweep-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
BASE_ENV = os.environ.copy()
BASE_ENV["WANDB_PROJECT"] = WANDB_PROJECT

def run_one(lr, accum):
    env = BASE_ENV.copy()
    env["LEARNING_RATE"] = str(lr)
    env["ACCUMULATE_GRAD_BATCHES"] = str(accum)
    env["WANDB_RUN_GROUP"] = RUN_GROUP
    env["WANDB_NAME_PREFIX"] = f"lr={lr}-acc={accum}"

    print(f"\n Running img_train.py with LR={lr} | ACCUMULATE_GRAD_BATCHES={accum} ")
    result = subprocess.run(["python", "img_train.py"], env=env)
    if result.returncode != 0:
        print(f"Run failed for lr={lr}, accum={accum} (exit code {result.returncode})")
        return False
    return True

if __name__ == "__main__":
    total_runs = len(LEARNING_RATES) * len(ACCUM_STEPS)
    successful_runs = 0
    failed_runs = 0
    
    print(f"Starting hyperparameter sweep with {total_runs} total combinations")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Accumulation steps: {ACCUM_STEPS}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Sweep group: {RUN_GROUP}")
    
    for i, (lr, accum) in enumerate(itertools.product(LEARNING_RATES, ACCUM_STEPS), 1):
        print(f"\n[{i}/{total_runs}] Starting run {i}")
        success = run_one(lr, accum)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1

    print(f"\nSweep complete. Group: {RUN_GROUP}")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    print(f"Failed runs: {failed_runs}/{total_runs}")
