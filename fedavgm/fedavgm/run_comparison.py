
import subprocess
import glob
import os
import sys
import pickle
import matplotlib.pyplot as plt
import time

def run_experiment(strategy, rounds, dataset="imagenette"):
    print(f"Running strategy={strategy} for rounds={rounds}...")
    cmd = [
        sys.executable, "-u", "main_sequential.py", 
        f"strategy={strategy}", 
        f"num_rounds={rounds}",
        f"dataset={dataset}"
    ]
    
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        print(f"Output for {strategy}:")
        print(result.stdout[-500:]) # Print last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"Error running {strategy}:")
        print("STDOUT:", e.stdout[-2000:] if e.stdout else "None") # Print more context
        print("STDERR:", e.stderr[-2000:] if e.stderr else "None")
        return None

    return find_latest_pkl()

def find_latest_pkl():
    files = glob.glob(os.path.join("outputs", "**", "*.pkl"), recursive=True)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def plot_results(files, output_file):
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="rocket")
        has_seaborn = True
    except ImportError:
        print("Seaborn not found, falling back to matplotlib")
        has_seaborn = False
    
    plt.figure(figsize=(12, 8))
    
    # Custom styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Roboto', 'Arial', 'sans-serif']
    
    for label, filepath in files.items():
        if not filepath or not os.path.exists(filepath):
            print(f"File not found for {label}: {filepath}")
            continue
            
        print(f"Loading {label} from {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        history = data["history"]
        if "accuracy" in history.metrics_centralized:
            rounds, acc = zip(*history.metrics_centralized["accuracy"])
            # Plot
            if has_seaborn:
                sns.lineplot(x=rounds, y=acc, label=label, marker='o', linewidth=2.5, markersize=8)
            else:
                plt.plot(rounds, acc, label=label, marker='o', linewidth=2.5, markersize=8)
        else:
            print(f"No accuracy metrics found for {label}")
        
    plt.xlabel("Rounds", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.title("Benchmarking: Custom FedAvgM vs FedAvgM (Imagenette)", fontsize=18, fontweight='bold', pad=20)
    
    plt.legend(title="Strategy", fontsize=12, title_fontsize=12, loc='lower right', frameon=True, shadow=True)
    
    # Add a subtle grid and remove top/right spines
    if has_seaborn:
        try:
             sns.despine(trim=True)
        except:
             pass
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    # Default to 5 rounds for demo, or take from arg
    rounds = 5
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])
        
    print(f"Starting comparison for {rounds} rounds...")
    
    # 1. Run CustomFedAvgM
    pkl_custom = run_experiment("custom-fedavgm", rounds)
    time.sleep(1) # Ensure timestamp diff
    
    # 2. Run FedAvgM
    pkl_standard = run_experiment("fedavgm", rounds)
    
    # 3. Plot
    if pkl_custom and pkl_standard:
        output_dir = "imagenette_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"comparison_rounds_{rounds}.png")
        
        plot_results({
            "Custom FedAvgM": pkl_custom,
            "Standard FedAvgM": pkl_standard
        }, output_file=output_filename)
        print(f"Done. Check {output_filename}")
    else:
        print("Failed to complete runs.")
