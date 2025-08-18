"""📚 Import Required Libraries"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Activation,
    MaxPool2D,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.regularizers import l2

"""⚙️ Build AlexNet Architecture"""

# Weight decay factor for L2 regularization
weight_decay = 5e-4

# Initialize a Sequential model with name "Alexnet"
model = Sequential(name="Alexnet")

# ---------------------------
# Layer 1: Convolution + ReLU + MaxPooling + BatchNorm
# ---------------------------
# Input shape: 227x227 RGB image (3 channels)
model.add(Input(shape=(227, 227, 3)))
model.add(
    Conv2D(
        96,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding="valid",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
model.add(BatchNormalization())

# ---------------------------
# Layer 2: Convolution + ReLU + MaxPooling + BatchNorm
# ---------------------------
model.add(
    Conv2D(
        256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
model.add(BatchNormalization())

# ---------------------------
# Layer 3: Convolution + ReLU + BatchNorm
# ---------------------------
model.add(
    Conv2D(
        384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(Activation("relu"))
model.add(BatchNormalization())

# ---------------------------
# Layer 4: Convolution + ReLU + BatchNorm
# ---------------------------
model.add(
    Conv2D(
        384,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(Activation("relu"))
model.add(BatchNormalization())

# ---------------------------
# Layer 5: Convolution + ReLU + BatchNorm + MaxPooling
# ---------------------------
model.add(
    Conv2D(
        256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

# Flatten feature maps before entering fully connected layers
model.add(Flatten())

# ---------------------------
# Layer 6: Fully Connected (Dense) + ReLU + Dropout
# ---------------------------
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))

# ---------------------------
# Layer 7: Fully Connected (Dense) + ReLU + Dropout
# ---------------------------
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))

# ---------------------------
# Layer 8: Output Layer (Dense) with Softmax
# ---------------------------
# Produces class probabilities across 1000 classes (ImageNet)
model.add(Dense(1000, activation="softmax"))

# Display the full model summary
model.summary()
