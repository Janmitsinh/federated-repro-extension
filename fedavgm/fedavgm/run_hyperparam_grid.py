
import subprocess
import glob
import os
import sys
import pickle
import matplotlib.pyplot as plt
import time
import itertools

def run_experiment(rounds, dataset, learning_rate, C, E, label):
    print(f"Running {label} (LR={learning_rate}, C={C}, E={E})...")
    cmd = [
        sys.executable, "-u", "main_sequential.py", 
        f"num_rounds={rounds}",
        f"dataset={dataset}",
        f"client.lr={learning_rate}",
        f"server.reporting_fraction={C}",
        f"client.local_epochs={E}",
        f"strategy=fedavgm" 
    ]
    
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running {label}:")
        print(e.stderr[-1000:] if e.stderr else "No stderr")
        return None
        
    return find_latest_pkl()

def find_latest_pkl():
    files = glob.glob(os.path.join("outputs", "**", "*.pkl"), recursive=True)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def plot_results(results_map, output_file, title_suffix):
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="viridis")
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    plt.figure(figsize=(12, 8))
    
    for label, filepath in results_map.items():
        if not filepath or not os.path.exists(filepath):
            continue
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        history = data["history"]
        if "accuracy" in history.metrics_centralized:
            rounds, acc = zip(*history.metrics_centralized["accuracy"])
            if has_seaborn:
                sns.lineplot(x=rounds, y=acc, label=label, marker='o', linewidth=2.0)
            else:
                plt.plot(rounds, acc, label=label, marker='o', linewidth=2.0)
            
    plt.xlabel("Rounds", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12, fontweight='bold')
    plt.title(f"Hyperparameter Sensitivity: {title_suffix}", fontsize=14, fontweight='bold', pad=15)
    plt.legend(title="Config", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    rounds = 5
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])

    # 1. Sensitivity to Learning Rate
    lrs = [0.01, 0.001] 
    results_lr = {}
    print(f"\n>>> Analyzing Learning Rate Sensitivity...")
    for lr in lrs:
        label = f"LR={lr}"
        pkl = run_experiment(rounds, "cifar10", lr, 0.05, 1, label)
        results_lr[label] = pkl
        time.sleep(1)
        
    os.makedirs("hyperparam_outputs", exist_ok=True)
    plot_results(results_lr, "hyperparam_outputs/sensitivity_lr.png", "Learning Rate")

    # 2. Sensitivity to Local Epochs
    epochs = [1, 5] 
    results_E = {}
    print(f"\n>>> Analyzing Local Epochs Sensitivity...")
    for E in epochs:
        label = f"E={E}"
        # detailed comparison:
        pkl = run_experiment(rounds, "cifar10", 0.01, 0.05, E, label)
        results_E[label] = pkl
        time.sleep(1)
    
    plot_results(results_E, "hyperparam_outputs/sensitivity_epochs.png", "Local Epochs (E)")

    # 3. Sensitivity to Reporting Fraction
    # Note: reporting fraction affects how many clients are sampled.
    fractions = [0.1, 0.2]
    results_C = {}
    print(f"\n>>> Analyzing Reporting Fraction Sensitivity...")
    for C in fractions:
        label = f"C={C}"
        pkl = run_experiment(rounds, "cifar10", 0.01, C, 1, label)
        results_C[label] = pkl
        time.sleep(1)
        
    plot_results(results_C, "hyperparam_outputs/sensitivity_fraction.png", "Reporting Fraction (C)")
    
    print("\nHyperparameter analysis checked.")
