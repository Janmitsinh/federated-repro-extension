
import subprocess
import glob
import os
import sys
import pickle
import matplotlib.pyplot as plt
import time
import math

def run_experiment_alpha(rounds, alpha):
    """Run FedAvgM with specific alpha"""
    print(f"Running FedAvgM (alpha={alpha})...")
    cmd = [
        sys.executable, "-u", "main_sequential.py", 
        f"num_rounds={rounds}",
        f"dataset=cifar10", 
        f"noniid.concentration={alpha}",
        f"strategy=fedavgm"
    ]
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running alpha={alpha}:")
        print(e.stderr[-1000:] if e.stderr else "No stderr")
        return None
    return find_latest_pkl()

def find_latest_pkl():
    files = glob.glob(os.path.join("outputs", "**", "*.pkl"), recursive=True)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def plot_alpha_sweep(results_map, output_file):
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="coolwarm")
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    plt.figure(figsize=(12, 8))
    
    # Sort keys by alpha value
    sorted_alphas = sorted(results_map.keys())
    
    for alpha in sorted_alphas:
        filepath = results_map[alpha]
        if not filepath or not os.path.exists(filepath):
            continue
            
        label = f"α={alpha}"
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
    plt.title(f"Impact of Non-IID Heterogeneity (Alpha Sweep)", fontsize=14, fontweight='bold', pad=15)
    plt.legend(title="Concentration (α)", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    rounds = 5
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])
        
    alphas = [0.05, 0.1, 0.5, 1, 10, 100]
    results = {}
    
    print(f">>> Starting Alpha Sweep (CIFAR-10) for {rounds} rounds...")
    print(f"Alphas to test: {alphas}")
    
    for alpha in alphas:
        pkl = run_experiment_alpha(rounds, alpha)
        if pkl:
            results[alpha] = pkl
            print(f"Finished alpha={alpha}")
        else:
            print(f"Failed alpha={alpha}")
        time.sleep(1)
        
    if results:
        os.makedirs("cifar10_repro_outputs", exist_ok=True)
        plot_alpha_sweep(results, "cifar10_repro_outputs/alpha_sweep_comparison.png")
