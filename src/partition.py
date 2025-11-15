# src/partition.py
import random
import numpy as np

def create_dirichlet_partitions(dataset_targets, num_clients=100, alpha=0.5, min_size=10, seed=0):
    """
    Create dict: client_id -> list of sample indices using Dirichlet distribution.
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
