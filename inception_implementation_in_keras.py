"""📚 Import Required Libraries"""

import math

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    GlobalAveragePooling2D,
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense,
    concatenate,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10

"""🗂️ Prepare the CIFAR-10 Dataset"""

num_classes = 10


def load_cifar10_data(img_rows, img_cols):
    """Load CIFAR-10 dataset, resize, normalize, and one-hot encode labels."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Resize training & test images to match GoogLeNet input (224x224)
    x_train = np.array([cv2.resize(img, dsize=(img_rows, img_cols)) for img in x_train])
    x_test = np.array([cv2.resize(img, dsize=(img_rows, img_cols)) for img in x_test])

    # One-hot Encode Labels
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Normalize the Data (scale to [0, 1])
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train, x_test, y_test)


# Load CIFAR-10 resized to 224x224 for GoogLeNet
x_train, y_train, x_test, y_test = load_cifar10_data(224, 224)

"""🔧 Define the Inception Module"""


def inception_module(
    x,
    filter_1x1,
    filter_3x3_reduce,
    filter_3x3,
    filter_5x5_reduce,
    filter_5x5,
    filter_pool_proj,
    kernel_init=tf.keras.initializers.glorot_uniform(),
    bias_init=tf.keras.initializers.constant(value=0.2),
    name=None,
):
    """
    Build an Inception module with:
    - 1x1 Convolution
    - 1x1 -> 3x3 Convolutions
    - 1x1 -> 5x5 Convolutions
    - 3x3 MaxPooling -> 1x1 Convolution
    """
    # 1x1 Convolution branch
    conv_1x1 = Conv2D(
        filter_1x1,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(x)

    # 1x1 -> 3x3 Convolution branch
    conv_3x3 = Conv2D(
        filter_3x3_reduce,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(x)
    conv_3x3 = Conv2D(
        filter_3x3,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(conv_3x3)

    # 1x1 -> 5x5 Convolution branch
    conv_5x5 = Conv2D(
        filter_5x5_reduce,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(x)
    conv_5x5 = Conv2D(
        filter_5x5,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(conv_5x5)

    # MaxPooling -> 1x1 Convolution branch
    pool_proj = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    pool_proj = Conv2D(
        filter_pool_proj,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(pool_proj)

    # Concatenate outputs along the depth dimension
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output


"""🏗️ Build the GoogLeNet (Inception v1) Architecture"""

# Define kernel and bias initializers
kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.constant(value=0.2)

# Input Layer
input_layer = Input(shape=(224, 224, 3))

# Initial Convolution + Pooling layers
x = Conv2D(
    64,
    kernel_size=(7, 7),
    strides=(2, 2),
    padding="same",
    activation="relu",
    name="conv_1_7x7\2",
    kernel_initializer=kernel_init,
    bias_initializer=bias_init,
)(input_layer)
x = MaxPool2D(
    pool_size=(3, 3), strides=(2, 2), padding="same", name="max_pool_1_3x3\2"
)(x)

x = Conv2D(
    192,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation="relu",
    name="conv_2b_3x3\1",
)(x)
x = MaxPool2D(
    pool_size=(3, 3), strides=(2, 2), padding="same", name="max_pool_2_3x3\2"
)(x)

# Inception modules (3a, 3b, then pooling)
x = inception_module(
    x,
    filter_1x1=64,
    filter_3x3_reduce=96,
    filter_3x3=128,
    filter_5x5_reduce=16,
    filter_5x5=32,
    filter_pool_proj=32,
    name="inception_3a",
)
x = inception_module(
    x,
    filter_1x1=128,
    filter_3x3_reduce=128,
    filter_3x3=192,
    filter_5x5_reduce=32,
    filter_5x5=96,
    filter_pool_proj=64,
    name="inception_3b",
)
x = MaxPool2D(
    pool_size=(3, 3), strides=(2, 2), padding="same", name="max_pool_3_3x3\2"
)(x)

# Inception 4a with auxiliary classifier 1
x = inception_module(
    x,
    filter_1x1=192,
    filter_3x3_reduce=96,
    filter_3x3=208,
    filter_5x5_reduce=16,
    filter_5x5=48,
    filter_pool_proj=64,
    name="inception_4a",
)
classifier_1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
classifier_1 = Conv2D(128, kernel_size=(1, 1), padding="same", activation="relu")(
    classifier_1
)
classifier_1 = Flatten()(classifier_1)
classifier_1 = Dense(1024, activation="relu")(classifier_1)
classifier_1 = Dropout(0.7)(classifier_1)
classifier_1 = Dense(10, activation="softmax", name="auxilliary_output_1")(classifier_1)

# Inception 4b–4d with auxiliary classifier 2
x = inception_module(
    x,
    filter_1x1=160,
    filter_3x3_reduce=112,
    filter_3x3=224,
    filter_5x5_reduce=24,
    filter_5x5=64,
    filter_pool_proj=64,
    name="inception_4b",
)
x = inception_module(
    x,
    filter_1x1=128,
    filter_3x3_reduce=128,
    filter_3x3=256,
    filter_5x5_reduce=24,
    filter_5x5=64,
    filter_pool_proj=64,
    name="inception_4c",
)
x = inception_module(
    x,
    filter_1x1=112,
    filter_3x3_reduce=144,
    filter_3x3=288,
    filter_5x5_reduce=32,
    filter_5x5=64,
    filter_pool_proj=64,
    name="inception_4d",
)
classifier_2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
classifier_2 = Conv2D(128, kernel_size=(1, 1), padding="same", activation="relu")(
    classifier_2
)
classifier_2 = Flatten()(classifier_2)
classifier_2 = Dense(1024, activation="relu")(classifier_2)
classifier_2 = Dropout(0.7)(classifier_2)
classifier_2 = Dense(10, activation="softmax", name="auxilliary_output_2")(classifier_2)

# Inception 4e + Pooling
x = inception_module(
    x,
    filter_1x1=256,
    filter_3x3_reduce=160,
    filter_3x3=320,
    filter_5x5_reduce=32,
    filter_5x5=128,
    filter_pool_proj=128,
    name="inception_4e",
)
x = MaxPool2D(
    pool_size=(3, 3), strides=(2, 2), padding="same", name="max_pool_4_3x3\2"
)(x)

# Final Inception blocks (5a, 5b) + Average Pooling
x = inception_module(
    x,
    filter_1x1=256,
    filter_3x3_reduce=160,
    filter_3x3=320,
    filter_5x5_reduce=32,
    filter_5x5=128,
    filter_pool_proj=128,
    name="inception_5a",
)
x = inception_module(
    x,
    filter_1x1=384,
    filter_3x3_reduce=192,
    filter_3x3=384,
    filter_5x5_reduce=48,
    filter_5x5=128,
    filter_pool_proj=128,
    name="inception_5b",
)
x = MaxPool2D(
    pool_size=(7, 7), strides=(1, 1), padding="valid", name="avg_pool_5_3x3\1"
)(x)

# Dropout + Fully Connected Layers
x = Dropout(0.4)(x)
x = Dense(1000, activation="relu", name="linear")(x)
x = Dense(1000, activation="softmax", name="output")(x)

"""📌 GoogLeNet Model (Without Auxiliary Classifiers)"""

model = Model(input_layer, [x], name="googlenet")
model.summary()

"""📌 GoogLeNet Model (With Auxiliary Classifiers)"""

model_with_classifiers = Model(
    input_layer, [x, classifier_1, classifier_2], name="googlenet_complete_architecture"
)
model_with_classifiers.summary()

"""⚡ Compile and Train the Model with Learning Rate Scheduler"""

epochs = 25
initial_lrate = 0.01


# Learning Rate Decay Function
def decay(epoch, steps=100):
    """Exponential decay of learning rate every few epochs."""
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# Optimizer
sgd = SGD(learning_rate=initial_lrate, momentum=0.9, nesterov=False)

# Learning Rate Scheduler Callback
lr_sc = LearningRateScheduler(decay, verbose=1)

# Compile the model with 3 outputs (main + 2 auxiliary classifiers)
model_with_classifiers.compile(
    optimizer=sgd,
    loss=[
        "categorical_crossentropy",
        "categorical_crossentropy",
        "categorical_crossentropy",
    ],
    loss_weights=[1, 0.3, 0.3],  # Auxiliary classifiers weighted lower
    metrics=["accuracy"],
)

# Train the complete GoogLeNet model
history = model_with_classifiers.fit(
    x_train,
    [y_train, y_train, y_train],
    validation_data=(x_test, [y_test, y_test, y_test]),
    epochs=epochs,
    batch_size=256,
    callbacks=[lr_sc],
)
