# Federated Learning Reproducibility & Extension Project  
**COMP 691 – Reproducing “Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification” (Hsu et al., 2019)**

This repository contains a full reproducibility and extended experimentation pipeline for the 2019 paper by Hsu, Qi, and Brown, which investigates the effects of non-IID client data in Federated Learning (FL).  
Our project faithfully reproduces the FedAvg and FedAvgM baselines on CIFAR-10 and extends the study to a more realistic dataset (Imagenette or MedMNIST) while introducing adaptive FL optimizers (FedAdam, FedYogi).

---

## Overview

Federated Learning (FL) is highly sensitive to non-identical (non-IID) client distributions.  
The original paper proposes:
- A Dirichlet-based method for synthesizing client heterogeneity.
- Baseline FedAvg performance across α ∈ {0.05, 0.1, 0.5, 1, 10, 100}.
- FedAvgM (server momentum) as a stability improvement.

This project:
1. **Reproduces** the paper’s CIFAR-10 experiments.  
2. **Analyzes** discrepancies, hyperparameter sensitivity, and convergence behavior.  
3. **Extends** the work to:  
   - A more realistic dataset (Imagenette or a medical dataset such as MedMNIST).  
   - Alternate optimizers: FedAdam & FedYogi.  
   - Additional non-IID schemes and scaling to more clients (50–500).  

---

## Project Goals

### Reproduction
- Implement FedAvg and FedAvgM in PyTorch.
- Generate non-IID Dirichlet client partitions.
- Reproduce:
  - Accuracy trends vs α.
  - Sensitivity to η, E, C.
  - Stability effects of server momentum β.

### Extension
- Add Imagenette or MedMNIST as a more realistic dataset.
- Implement FedAdam and FedYogi server optimizers.
- Evaluate scaling to larger numbers of clients.
- Explore additional non-IID modes: label skew, quantity skew.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/federated-repro-extension.git
cd federated-repro-extension
```

### 2. Install the requirements 
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

