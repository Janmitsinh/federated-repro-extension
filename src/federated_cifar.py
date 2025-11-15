# src/federated_cifar.py
#!/usr/bin/env python3
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from models import SimpleCNN
from partition import create_dirichlet_partitions
from train_utils import client_update, weighted_average_states, apply_delta_to_global, evaluate
from datasets import get_cifar10

def run_experiment(cfg):
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

    train_dataset, test_loader = get_cifar10(data_root=cfg.get('data_root','./data'),
                                            batch_size=batch_size, num_workers=num_workers)

    global_model = SimpleCNN(num_classes=10).to(device)
    targets = [int(y) for y in train_dataset.targets]

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
    p.add_argument('--data_root', type=str, default='./data')
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
                        'num_workers': args.num_workers,
                        'data_root': args.data_root
                    }
                    print(f'\nRun {runs}/{total}  alpha={alpha} lr={lr} C={C} E={E} device={device}')
                    res = run_experiment(cfg)
                    summary.append({'alpha': alpha, 'lr': lr, 'C': C, 'E': E, 'csv': res['csv'], 'best_acc': res['best_acc']})

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, f'summary_seed={args.seed}.csv')
    with open(summary_path, 'w', newline='') as sf:
        import csv as _csv
        w = _csv.writer(sf)
        w.writerow(['alpha','lr','C','E','best_acc','csv'])
        for row in summary:
            w.writerow([row['alpha'], row['lr'], row['C'], row['E'], row['best_acc'], row['csv']])
    print('All runs complete. Summary saved to', summary_path)

if __name__ == '__main__':
    main()
