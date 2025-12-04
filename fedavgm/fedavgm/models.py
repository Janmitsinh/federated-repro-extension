"""CNN model architecture."""

from flwr.common import ndarrays_to_parameters
from keras.optimizers import SGD
from keras.regularizers import l2
from tensorflow import keras
from tensorflow.nn import local_response_normalization  # pylint: disable=import-error
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2


def mobilenetv2(input_shape, num_classes, learning_rate):
    """MobileNetV2 transfer learning head suitable for Imagenette.

    Returns a compiled Keras model. Uses categorical_crossentropy to be compatible
    with existing client code that converts labels via to_categorical.
    """
    input_shape = tuple(input_shape)

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze by default for faster experiments

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def cnn(input_shape, num_classes, learning_rate):
    """CNN Model from (McMahan et. al., 2017).

    Communication-efficient learning of deep networks from decentralized data
    """
    input_shape = tuple(input_shape)

    weight_decay = 0.004
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(
                384, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(
                192, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def tf_example(input_shape, num_classes, learning_rate):
    """CNN Model from TensorFlow v1.x example.

    This is the model referenced on the FedAvg paper.

    Reference:
    https://web.archive.org/web/20170807002954/https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    """
    input_shape = tuple(input_shape)

    weight_decay = 0.004
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.Lambda(
                local_response_normalization,
                arguments={
                    "depth_radius": 4,
                    "bias": 1.0,
                    "alpha": 0.001 / 9.0,
                    "beta": 0.75,
                },
            ),
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
            ),
            keras.layers.Lambda(
                local_response_normalization,
                arguments={
                    "depth_radius": 4,
                    "bias": 1.0,
                    "alpha": 0.001 / 9.0,
                    "beta": 0.75,
                },
            ),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(
                384, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(
                192, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def model_to_parameters(model):
    """Retrieve model weigths and convert to ndarrays."""
    return ndarrays_to_parameters(model.get_weights())
