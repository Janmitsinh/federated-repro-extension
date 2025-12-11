"""Dataset utilities for federated learning."""

import os
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow import keras

from fedavgm.common import create_lda_partitions

def imagenette(num_classes, input_shape, data_dir=None, max_samples: int | None = None):
    """Prepare Imagenette dataset (loads images using Keras utilities).

    If max_samples is provided, dataset is truncated to that many examples
    (useful for low-memory development runs).
    """
    print(f">>> [Dataset] Loading Imagenette from {data_dir} with input shape {input_shape}")

    # Use image_dataset_from_directory to load and resize images
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    # Default data_dir if None: use hydra expanded runtime cwd at call site
    if data_dir is None:
        raise ValueError("data_dir must be provided for imagenette loader")

    ds_train = image_dataset_from_directory(
        str(Path(data_dir) / "train"),
        labels="inferred",
        label_mode="int",
        image_size=(input_shape[0], input_shape[1]),
        batch_size=128,
        shuffle=False,
    )
    ds_val = image_dataset_from_directory(
        str(Path(data_dir) / "val"),
        labels="inferred",
        label_mode="int",
        image_size=(input_shape[0], input_shape[1]),
        batch_size=128,
        shuffle=False,
    )

    # Convert to numpy arrays if max_samples is specified (truncated)
    def dataset_to_numpy(ds, max_samples=None):
        xs = []
        ys = []
        seen = 0
        for batch_x, batch_y in ds:
            batch_x = batch_x.numpy().astype("float32") / 255.0
            batch_y = batch_y.numpy().astype("int64")
            xs.append(batch_x)
            ys.append(batch_y)
            seen += batch_x.shape[0]
            if max_samples is not None and seen >= max_samples:
                break
        if len(xs) == 0:
            return np.empty((0, *input_shape), dtype=np.float32), np.empty((0,), dtype=np.int64)
        x_arr = np.concatenate(xs, axis=0)
        y_arr = np.concatenate(ys, axis=0).reshape((-1,))
        if max_samples is not None and x_arr.shape[0] > max_samples:
            x_arr = x_arr[:max_samples]
            y_arr = y_arr[:max_samples]
        return x_arr, y_arr

    x_train, y_train = dataset_to_numpy(ds_train, max_samples=max_samples)
    x_test, y_test = dataset_to_numpy(ds_val, max_samples=None)

    input_shape = tuple(input_shape)
    num_classes = num_classes

    print(f">>> Imagenette loaded: train={len(x_train)} val={len(x_test)}")
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def cifar10(num_classes, input_shape):
    """Prepare the CIFAR-10.

    This method considers CIFAR-10 for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading CIFAR-10. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def fmnist(num_classes, input_shape):
    """Prepare the FMNIST.

    This method considers FMNIST for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading FMNIST. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def create_shards_partitions(x, y, num_clients, shards_per_client=2):
    

    idx = np.argsort(y, axis=0)
    x_sorted = x[idx]
    y_sorted = y[idx]


    total_samples = x.shape[0]
    total_shards = num_clients * shards_per_client
    shard_size = total_samples // total_shards

    if shard_size * total_shards != total_samples:
        print(f"Warning: Dropping {total_samples % total_shards} samples to ensure equal shards.")
        cutoff = shard_size * total_shards
        x_sorted = x_sorted[:cutoff]
        y_sorted = y_sorted[:cutoff]


    shards_x = np.split(x_sorted, total_shards)
    shards_y = np.split(y_sorted, total_shards)
    

    shard_idxs = np.random.permutation(total_shards)
    
    partitions = []
    for i in range(num_clients):
        my_shards_indices = shard_idxs[i * shards_per_client : (i + 1) * shards_per_client]
        
        x_local = np.concatenate([shards_x[j] for j in my_shards_indices], axis=0)
        y_local = np.concatenate([shards_y[j] for j in my_shards_indices], axis=0)
        
        partitions.append((x_local, y_local))
        
    return partitions


def partition(x_train, y_train, num_clients, concentration):
    """Create non-iid partitions.

    The partitions uses a LDA distribution based on concentration.
    """
    # --- LOGIC FOR SHARDS ---
    if concentration < 0:
        print(f">>> [Dataset] Using SHARD-BASED partition (FedAvg original).")
        return create_shards_partitions(x_train, y_train, num_clients, shards_per_client=2)
    # ----------------------------
    print(
        f">>> [Dataset] {num_clients} clients, non-iid concentration {concentration}..."
    )
    dataset = [x_train, y_train]
    partitions, _ = create_lda_partitions(
    dataset,
    num_partitions=num_clients,
    # concentration=concentration * num_classes,
    concentration=concentration,
    accept_imbalanced=True,   # allow partitions when sample count not divisible
    seed=1234,
    )
    return partitions
