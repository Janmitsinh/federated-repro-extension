from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(r"C:\Users\Nishith Soni\Desktop\Topics in CS\federated-repro-extension\fedavgm\fedavgm\outputs\imagenette_runs")

concentrations = [10.0, 1.0, 0.1, 0.01]

def find_result_file(prefix: str, alpha: float) -> Path:
    """prefix: 'results_FedAvg' or 'results_FedAvgM'."""
    alpha_str = f"_alpha={alpha}"
    for p in RESULTS_DIR.glob("results_*.pkl"):
        name = p.name
        if name.startswith(prefix) and alpha_str in name:
            return p
    raise FileNotFoundError(f"No file for {prefix} alpha={alpha} in {RESULTS_DIR}")

def load_final_accuracy(path: Path) -> float:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    history = obj["history"]
    _, acc = history.metrics_centralized["accuracy"][-1]
    return acc

fedavg_accs = []
fedavgm_accs = []

for alpha in concentrations:
    f_fedavg  = find_result_file("results_FedAvg",  alpha)
    f_fedavgm = find_result_file("results_FedAvgM", alpha)

    fedavg_accs.append(load_final_accuracy(f_fedavg))
    fedavgm_accs.append(load_final_accuracy(f_fedavgm))

concs = np.array(concentrations)

plt.figure(figsize=(8, 4))
plt.plot(concs, fedavg_accs, marker="^", label="FedAvg")
plt.plot(concs, fedavgm_accs, marker="o", label="FedAvgM")
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel("Concentration")
plt.ylabel("Test Accuracy")
plt.title("Imagenette\nLocal Epoch E = 1 | Reporting Fraction C = 0.05 | num_rounds = 5")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.5)

static_dir = Path(__file__).resolve().parent / "_static"
static_dir.mkdir(exist_ok=True)
out_path = static_dir / "fedavgm_vs_fedavg_rounds-5_imagenette.jpg"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved plot to {out_path}")