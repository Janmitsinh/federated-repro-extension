#!/usr/bin/env python3
"""
Federated learning reproducibility script (CIFAR-10 baseline)

Features:
 - FedAvg and FedAvgM (server momentum)
 - Dirichlet alpha partitions for non-IID client data
 - Grid/sweep support for alpha, lr, C (fraction), E (local epochs)
 - Robust aggregation (handles float vs integer buffers)
 - Per-round CSV logging + best-model checkpoint
 - Defaults tuned for CIFAR-10 reproduction experiments

Usage:
  python federated_cifar.py --help
"""
import argparse
import copy
import csv
import os
import random
import time
from collections import defaultdict


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# -------------------------
# Simple CNN used for CIFAR-10 baseline (fixed fc input dim)
# -------------------------
class SimpleCNN(nn.Module):
    """
    Simple CNN with dynamic computation of the flattened feature size.
    Use: SimpleCNN(num_classes=10, in_channels=3, input_size=(32,32))
    - input_size is (H, W) of individual images (default 32x32 for CIFAR-10).
    The constructor runs a single dummy forward to infer the linear layer input dim.
    """
    def __init__(self, num_classes=10, in_channels=3, input_size=(32, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool3 = nn.MaxPool2d(3, stride=2)

        # compute flattened feature size by doing a forward pass with a dummy tensor
        self._feature_dim = self._get_feature_dim(input_size, in_channels)

        # fully-connected layers
        self.fc1 = nn.Linear(self._feature_dim, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def _get_feature_dim(self, input_size, in_channels):
        """
        Run a dummy forward pass (no gradient) to determine the flattened feature dimension
        after the conv / pooling pipeline.
        """
        h, w = input_size
        with torch.no_grad():
            x = torch.zeros(1, in_channels, h, w)
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            feat_dim = x.view(1, -1).shape[1]
        return feat_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# -------------------------
# Dirichlet partitioning
# -------------------------
def create_dirichlet_partitions(dataset_targets, num_clients=100, alpha=0.5, min_size=10, seed=0):
    """
    Create a dict mapping client_id -> list of sample indices using Dirichlet distribution.
    """
    random.seed(seed)
    np.random.seed(seed)
    num_classes = int(max(dataset_targets) + 1)
    class_indices = [[] for _ in range(num_classes)]
    for idx, y in enumerate(dataset_targets):
        class_indices[y].append(idx)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    client_class_prop = np.random.dirichlet([alpha] * num_classes, size=num_clients)

    for cls in range(num_classes):
        cls_idxs = class_indices[cls]
        if len(cls_idxs) == 0:
            continue
        proportions = client_class_prop[:, cls]
        counts = (proportions / proportions.sum() * len(cls_idxs)).astype(int)
        remainder = len(cls_idxs) - counts.sum()
        if remainder > 0:
            for i in np.argsort(-proportions)[:remainder]:
                counts[i] += 1
        ptr = 0
        for client_id, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[client_id].extend(cls_idxs[ptr:ptr + cnt])
                ptr += cnt
        if ptr < len(cls_idxs):
            left = cls_idxs[ptr:]
            i = 0
            for idx in left:
                client_indices[i % num_clients].append(idx)
                i += 1

    # enforce minimum size by stealing from largest
    for i in range(num_clients):
        if len(client_indices[i]) < min_size:
            needed = min_size - len(client_indices[i])
            donors = sorted(range(num_clients), key=lambda k: len(client_indices[k]), reverse=True)
            for d in donors:
                if d == i:
                    continue
                take = min(needed, len(client_indices[d]) - min_size)
                if take <= 0:
                    continue
                moved = client_indices[d][-take:]
                client_indices[d] = client_indices[d][:-take]
                client_indices[i].extend(moved)
                needed -= take
                if needed <= 0:
                    break
    return {i: client_indices[i] for i in range(num_clients)}

# -------------------------
# Client local update
# -------------------------
def client_update(model, train_dataset, indices, device, epochs, batch_size, lr, wd, num_workers=2):
    model.train()
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict()), len(indices)

# -------------------------
# Aggregation helpers (robust to dtype)
# -------------------------
def weighted_average_states(global_state, client_states, client_sizes):
    """
    Returns avg_state where float entries are averaged (float32 accumulator).
    Integer entries are copied from global_state unchanged.
    """
    total = float(sum(client_sizes))
    avg_state = {}
    for k, v in global_state.items():
        if v.dtype.is_floating_point:
            avg_state[k] = torch.zeros_like(v, dtype=torch.float32)
        else:
            avg_state[k] = v.clone()
    for st, sz in zip(client_states, client_sizes):
        w = float(sz) / total
        for k in avg_state.keys():
            if avg_state[k].dtype.is_floating_point:
                avg_state[k] += st[k].to(torch.float32) * w
    return avg_state

def apply_delta_to_global(global_state, delta, server_velocity=None, beta=0.9, use_momentum=False, device='cpu'):
    """
    Apply delta (floats on device) to global_state and return new global_state dict.
    Keeps integer entries unchanged.
    """
    new_global = {}
    for k, old in global_state.items():
        if old.dtype.is_floating_point:
            old_f = old.to(device).to(torch.float32)
            if use_momentum:
                server_velocity[k] = server_velocity[k].to(device).to(torch.float32) * beta + delta[k].to(device)
                updated = old_f + server_velocity[k]
                new_global[k] = updated.cpu()
            else:
                updated = old_f + delta[k].to(device)
                new_global[k] = updated.cpu()
        else:
            new_global[k] = old
    return new_global

# -------------------------
# Evaluation
# -------------------------
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total

# -------------------------
# Experiment runner
# -------------------------
def run_experiment(cfg):
    # unpack
    device = cfg['device']
    rounds = cfg['rounds']
    clients = cfg['clients']
    C = cfg['C']
    E = cfg['E']
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    wd = cfg['wd']
    alpha = cfg['alpha']
    seed = cfg['seed']
    eval_every = cfg['eval_every']
    fedavgm = cfg['fedavgm']
    beta = cfg['beta']
    out_dir = cfg['out_dir']
    num_workers = cfg.get('num_workers', 2)

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CIFAR-10 loaders
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_dataset = datasets.CIFAR10(root=os.path.join('./data','cifar10'), train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=os.path.join('./data','cifar10'), train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers)

    # model and targets
    global_model = SimpleCNN(num_classes=10).to(device)
    targets = [int(y) for y in train_dataset.targets]

    # global_state on CPU
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    server_velocity = None
    if fedavgm:
        server_velocity = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()}

    partitions = create_dirichlet_partitions(targets, num_clients=clients, alpha=alpha, seed=seed)
    partitions = {k: v for k, v in partitions.items() if len(v) > 0}
    clients_list = list(partitions.keys())
    clients_per_round = max(1, int(C * clients))

    print(f'Experiment: rounds={rounds} clients={clients} C={C} clients_per_round={clients_per_round} alpha={alpha} lr={lr} E={E}')

    csv_path = os.path.join(out_dir, f'results_alpha={alpha}_lr={lr}_C={C}_E={E}_seed={seed}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['round','test_acc','test_loss','time_elapsed'])

    # initial eval
    global_model.load_state_dict(global_state)
    t0 = time.time()
    acc0, loss0 = evaluate(global_model.to(device), test_loader, device)
    csv_writer.writerow([0, acc0, loss0, 0.0])
    print(f'Round 0 | Test Acc: {acc0*100:.2f}% | Loss: {loss0:.4f}')

    best_acc = acc0
    for r in range(1, rounds+1):
        selected = random.sample(clients_list, clients_per_round)
        client_states = []
        client_sizes = []
        for cid in selected:
            local_model = SimpleCNN(num_classes=10).to(device)
            local_model.load_state_dict(global_state)
            st, n = client_update(local_model, train_dataset, partitions[cid], device, epochs=E, batch_size=batch_size, lr=lr, wd=wd, num_workers=num_workers)
            for k in st.keys():
                st[k] = st[k].cpu()
            client_states.append(st)
            client_sizes.append(n)

        avg_state = weighted_average_states(global_state, client_states, client_sizes)
        # build delta on device
        delta = {}
        for k in global_state.keys():
            if avg_state[k].dtype.is_floating_point:
                delta[k] = (avg_state[k].to(device) - global_state[k].to(device).to(torch.float32))
            else:
                delta[k] = torch.zeros_like(global_state[k], dtype=torch.float32).to(device)

        global_state = apply_delta_to_global(global_state, delta, server_velocity=server_velocity, beta=beta, use_momentum=fedavgm, device=device)
        global_model.load_state_dict(global_state)

        if r % eval_every == 0 or r == 1:
            acc, loss = evaluate(global_model.to(device), test_loader, device)
            elapsed = time.time() - t0
            csv_writer.writerow([r, acc, loss, elapsed])
            print(f'Round {r:4d} | Test Acc: {acc*100:.2f}% | Loss: {loss:.4f} | Elapsed: {elapsed:.1f}s')
            if acc > best_acc:
                best_acc = acc
                torch.save(global_model.state_dict(), os.path.join(out_dir, f'best_model_alpha={alpha}_lr={lr}_C={C}_E={E}.pt'))

    csv_file.close()
    return {'csv': csv_path, 'best_acc': best_acc}

# -------------------------
# CLI and grid orchestration
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=200)
    p.add_argument('--clients', type=int, default=100)
    p.add_argument('--C', type=float, default=0.1)
    p.add_argument('--E', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=0.004)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--fedavgm', action='store_true')
    p.add_argument('--beta', type=float, default=0.9)
    p.add_argument('--out_dir', type=str, default='./results')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--alpha_list', nargs='*', type=float, help='alpha sweep (overrides --alpha)')
    p.add_argument('--lr_list', nargs='*', type=float, help='lr sweep (overrides --lr)')
    p.add_argument('--C_list', nargs='*', type=float, help='C sweep (overrides --C)')
    p.add_argument('--E_list', nargs='*', type=int, help='E sweep (overrides --E)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device
    alpha_list = args.alpha_list if args.alpha_list else [args.alpha]
    lr_list = args.lr_list if args.lr_list else [args.lr]
    C_list = args.C_list if args.C_list else [args.C]
    E_list = args.E_list if args.E_list else [args.E]

    total = len(alpha_list) * len(lr_list) * len(C_list) * len(E_list)
    runs = 0
    summary = []
    for alpha in alpha_list:
        for lr in lr_list:
            for C in C_list:
                for E in E_list:
                    runs += 1
                    cfg = {
                        'device': device,
                        'rounds': args.rounds,
                        'clients': args.clients,
                        'C': C,
                        'E': E,
                        'batch_size': args.batch_size,
                        'lr': lr,
                        'wd': args.wd,
                        'alpha': alpha,
                        'seed': args.seed,
                        'eval_every': args.eval_every,
                        'fedavgm': args.fedavgm,
                        'beta': args.beta,
                        'out_dir': args.out_dir,
                        'num_workers': args.num_workers
                    }
                    print(f'\nRun {runs}/{total}  alpha={alpha} lr={lr} C={C} E={E} device={device}')
                    res = run_experiment(cfg)
                    summary.append({'alpha': alpha, 'lr': lr, 'C': C, 'E': E, 'csv': res['csv'], 'best_acc': res['best_acc']})

    # write summary CSV
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, f'summary_seed={args.seed}.csv')
    with open(summary_path, 'w', newline='') as sf:
        w = csv.writer(sf)
        w.writerow(['alpha','lr','C','E','best_acc','csv'])
        for row in summary:
            w.writerow([row['alpha'], row['lr'], row['C'], row['E'], row['best_acc'], row['csv']])
    print('All runs complete. Summary saved to', summary_path)

if __name__ == '__main__':
    main()
