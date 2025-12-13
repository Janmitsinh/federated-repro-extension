
import glob
import os
import pickle

def analyze_latest_results():
    files = glob.glob(os.path.join("outputs", "**", "*.pkl"), recursive=True)
    if not files:
        print("No pickles found.")
        return

    # Filter for our specific recent runs
    fedadam_runs = [f for f in files if "FedAdam" in f]
    fedavgm_runs = [f for f in files if "FedAvgM" in f]
    
    if fedadam_runs:
        latest_fedadam = max(fedadam_runs, key=os.path.getmtime)
        with open(latest_fedadam, "rb") as f:
            data = pickle.load(f)
            accs = [mp[1] for mp in data["history"].metrics_centralized["accuracy"]]
            print(f"FedAdam: Start={accs[0]:.4f}, End={accs[-1]:.4f}, Max={max(accs):.4f} (File: {os.path.basename(latest_fedadam)})")
    else:
        print("No FedAdam run found.")

    if fedavgm_runs:
        latest_fedavgm = max(fedavgm_runs, key=os.path.getmtime)
        with open(latest_fedavgm, "rb") as f:
            data = pickle.load(f)
            accs = [mp[1] for mp in data["history"].metrics_centralized["accuracy"]]
            print(f"FedAvgM: Start={accs[0]:.4f}, End={accs[-1]:.4f}, Max={max(accs):.4f} (File: {os.path.basename(latest_fedavgm)})")
    else:
        print("No FedAvgM run found.")

if __name__ == "__main__":
    try:
        analyze_latest_results()
    except Exception as e:
        print(f"Error analyzing: {e}")
