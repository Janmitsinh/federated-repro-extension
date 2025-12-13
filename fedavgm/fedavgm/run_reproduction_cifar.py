
import subprocess
import glob
import os
import sys
import pickle
import matplotlib.pyplot as plt
import time

def run_experiment(strategy, rounds, concentration, dataset="cifar10"):
    print(f"Running strategy={strategy} rounds={rounds} alpha={concentration} dataset={dataset}...")
    cmd = [
        sys.executable, "-u", "main_sequential.py", 
        f"strategy={strategy}", 
        f"num_rounds={rounds}",
        f"dataset={dataset}",
        f"noniid.concentration={concentration}"
    ]
    
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        # We don't want to fail hard if one run crashes, but we should log it
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        print(f"Output for {strategy} (alpha={concentration}):")
        print(result.stdout[-500:]) 
    except subprocess.CalledProcessError as e:
        print(f"Error running {strategy} (alpha={concentration}):")
        print("STDOUT:", e.stdout[-2000:] if e.stdout else "None")
        print("STDERR:", e.stderr[-2000:] if e.stderr else "None")
        return None

    return find_latest_pkl()

def find_latest_pkl():
    files = glob.glob(os.path.join("outputs", "**", "*.pkl"), recursive=True)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def plot_reproduction_results(results_map, output_file, alpha):
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="rocket")
        has_seaborn = True
    except ImportError:
        print("Seaborn not found, falling back to matplotlib")
        has_seaborn = False
    
    plt.figure(figsize=(10, 6))
    
    for label, filepath in results_map.items():
        if not filepath or not os.path.exists(filepath):
            continue
        print(f"Loading {label} from {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        history = data["history"]
        if "accuracy" in history.metrics_centralized:
            rounds, acc = zip(*history.metrics_centralized["accuracy"])
            if has_seaborn:
                sns.lineplot(x=rounds, y=acc, label=label, marker='o', linewidth=2.0)
            else:
                plt.plot(rounds, acc, label=label, marker='o', linewidth=2.0)
        else:
            print(f"No accuracy metrics found for {label}")
            
    plt.xlabel("Rounds", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12, fontweight='bold')
    plt.title(f"CIFAR-10 Reproduction (alpha={alpha})", fontsize=14, fontweight='bold', pad=15)
    plt.legend(title="Strategy", loc='lower right')
    
    if has_seaborn:
        try:
             sns.despine(trim=True)
        except:
             pass
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

    
def run_custom_cmd(rounds, alpha, strategy, momentum, label):
    print(f"Running {label}...")
    cmd = [
        sys.executable, "-u", "main_sequential.py", 
        f"strategy={strategy}", 
        f"num_rounds={rounds}",
        f"dataset=cifar10", 
        f"noniid.concentration={alpha}",
        f"server.momentum={momentum}"
    ]
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running {label}:")
        print(e.stderr[-1000:])
        return None
    return find_latest_pkl()

if __name__ == "__main__":
    rounds = 5
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])
    
    # Check for Alpha=0.1 (High Heterogeneity)
    alpha = 0.1
    
    print(f">>> REPRODUCTION: CIFAR-10, Alpha={alpha}, Rounds={rounds}")
    
    # 1. FedAvg (FedAvgM with momentum=0.0)
    pkl_fedavg = run_custom_cmd(rounds, alpha, "fedavgm", 0.0, "FedAvg (Mom=0)")
    time.sleep(2)
    
    # 2. FedAvgM (Momentum=0.9)
    pkl_fedavgm = run_custom_cmd(rounds, alpha, "fedavgm", 0.9, "FedAvgM (Mom=0.9)")
    
    if pkl_fedavg and pkl_fedavgm:
        output_dir = "cifar10_repro_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"repro_cifar10_alpha_{alpha}_rounds_{rounds}.png")
        
        plot_reproduction_results({
            "FedAvg (M=0)": pkl_fedavg,
            "FedAvgM (M=0.9)": pkl_fedavgm
        }, output_file=output_filename, alpha=alpha)
        print(f"Done. Check {output_filename}")
