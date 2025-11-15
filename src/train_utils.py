# src/train_utils.py
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def client_update(model, train_dataset, indices, device, epochs, batch_size, lr, wd, num_workers=2):
    """
    Run local SGD on a client (returns state_dict and number of samples).
    """
    model.train()
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
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

def weighted_average_states(global_state, client_states, client_sizes):
    """
    Weighted average for float params; copy integer buffers unchanged.
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
