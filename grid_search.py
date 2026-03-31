import subprocess
import yaml
import itertools
from copy import deepcopy

# Define hyperparameter grid
lrs = [0.001, 0.0005]
temperatures = [0.1]
init_log_sds = [-7.0, -5.0]

# Load base config
with open("deep_uncertainty/training/vi_train_config.yaml", "r") as f:
    base_config = yaml.safe_load(f)

for lr, temp, log_sds in itertools.product(lrs, temperatures, init_log_sds):
    config = deepcopy(base_config)
    config["training"]["num_epochs"] = 35  # For quick testing, set to 1 epoch. Adjust as needed.

    
    # Set hyperparameters
    config["training"]["build_params"]["optimizer_lr"] = lr
    config["training"]["build_params"]["temperature"] = temp
    config["training"]["build_params"]["init_log_sds"] = log_sds
    config["training"]["optimizer"]["kwargs"]["lr"] = lr
    
    # Unique experiment name and checkpoint dir per run
    run_name = f"vi-lr{lr}-temp{temp}-logsd{log_sds}"
    config["experiment_name"] = run_name
    
    # Save config
    config_path = f"deep_uncertainty/training/grid_configs/{run_name}.yaml"
    import os
    os.makedirs("deep_uncertainty/training/grid_configs", exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Running: {run_name}")
    subprocess.run([
        "python", "-m", "deep_uncertainty.training.train_model",
        "--config", config_path
    ], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    
    chkp_name = run_name.replace("-", "_")
    ckpt_path = f"checkpoints/{chkp_name}/version_0/last.ckpt"

    print(f"Evaluating: {run_name}")
    subprocess.run([
        "python", "-m", "deep_uncertainty.evaluation.eval_model",
        "--config-path", config_path,
        "--chkp-path", ckpt_path,
        "--log-dir", f"eval_logs/{run_name}/"
    ])

print("Grid search complete!")
