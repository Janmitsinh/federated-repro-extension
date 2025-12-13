
import subprocess
import glob
import os
import sys
import pickle
import matplotlib.pyplot as plt
import time

def run_experiment_clients(rounds, num_clients):
    """Run FedAvgM with specific number of clients"""
    print(f"Running FedAvgM (Clients={num_clients})...")
    
    cmd = [
        sys.executable, "-u", "main_sequential.py", 
        f"num_rounds={rounds}",
        f"dataset=cifar10", 
        f"num_clients={num_clients}",
        f"strategy=fedavgm"
    ]
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running Clients={num_clients}:")
        print(e.stderr[-1000:] if e.stderr else "No stderr")
        return None
    return find_latest_pkl()

def find_latest_pkl():
    files = glob.glob(os.path.join("outputs", "**", "*.pkl"), recursive=True)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def plot_client_sweep(results_map, output_file):
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="magma")
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    plt.figure(figsize=(12, 8))
    
    sorted_clients = sorted(results_map.keys())
    
    for n in sorted_clients:
        filepath = results_map[n]
        if not filepath or not os.path.exists(filepath):
            continue
            
        label = f"N={n}"
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
    plt.title(f"Scalability: Impact of Client Count (N)", fontsize=14, fontweight='bold', pad=15)
    plt.legend(title="Num Clients", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    rounds = 5
    if len(sys.argv) > 1:
        rounds = int(sys.argv[1])
        
    client_counts = [10, 20, 50] # Example counts
    results = {}
    
    print(f">>> Starting Client Sweep (CIFAR-10) for {rounds} rounds...")
    
    for n in client_counts:
        pkl = run_experiment_clients(rounds, n)
        if pkl:
            results[n] = pkl
            print(f"Finished N={n}")
        else:
            print(f"Failed N={n}")
        time.sleep(1)
        
    if results:
        os.makedirs("cifar10_repro_outputs", exist_ok=True)
        plot_client_sweep(results, "cifar10_repro_outputs/client_sweep_comparison.png")
