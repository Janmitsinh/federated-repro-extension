"""Dataset utilities for federated learning."""

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

from fedavgm.common import create_lda_partitions

def imagenette(num_classes, input_shape, data_dir):
    """Load Imagenette (directory layout) and return numpy arrays in the same
    format as cifar10/fmnist loaders used by main.py.

    Expects `data_dir` with subfolders 'train' and 'val' containing class subfolders.
    """
    print(f">>> [Dataset] Loading Imagenette from {data_dir} with input shape {input_shape}")

    img_height, img_width = int(input_shape[0]), int(input_shape[1])
    batch_size = 32

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Imagenette train/val directories not found at {train_dir} or {val_dir}."
        )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
    )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

    # Convert to numpy arrays (Imagenette small subset — should fit memory on standard dev machine)
    x_train_batches = []
    y_train_batches = []
    for x_batch, y_batch in train_ds:
        x_train_batches.append(x_batch.numpy())
        y_train_batches.append(y_batch.numpy())
    x_train = np.concatenate(x_train_batches, axis=0)
    y_train = np.concatenate(y_train_batches, axis=0)

    x_test_batches = []
    y_test_batches = []
    for x_batch, y_batch in val_ds:
        x_test_batches.append(x_batch.numpy())
        y_test_batches.append(y_batch.numpy())
    x_test = np.concatenate(x_test_batches, axis=0)
    y_test = np.concatenate(y_test_batches, axis=0)

    # Ensure dtype
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Return same shape format as other loaders
    return x_train, y_train, x_test, y_test, (img_height, img_width, 3), num_classes

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


def partition(x_train, y_train, num_clients, concentration):
    """Create non-iid partitions.

    The partitions uses a LDA distribution based on concentration.
    """
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
