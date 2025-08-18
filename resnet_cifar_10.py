"""📚 Import Libraries"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

"""⚙️ Training Parameters"""

batch_size = 32
epochs = 200
data_augmentation = True
num_classes = 10

# Subtracting pixel mean helps improve accuracy
subtract_pixel_mean = True

# ResNet depth/variant configuration
n = 3
version = 1  # ResNet v1 or v2

# Depth rule: ResNet v1 -> 6n+2, ResNet v2 -> 9n+2
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

model_type = f"resnet{depth}v{version}"

"""📥 Load CIFAR-10 Dataset"""

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

"""⚖️ Normalize Data"""

input_shape = x_train.shape[1:]

# Scale pixel values to [0,1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Optionally subtract pixel mean
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("y_train shape:", y_train.shape)

"""🔢 One-Hot Encode Labels"""

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

"""🔧 ResNet Building Block"""


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    """
    2D Convolution-Batch Normalization-Activation stack builder.

    Args:
        inputs (tensor): input tensor
        num_filters (int): number of filters
        kernel_size (int): convolution kernel size
        strides (int): convolution strides
        activation (str): activation function
        batch_normalization (bool): whether to include batch norm
        conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)

    Returns:
        tensor: output tensor
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
    )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


"""🏗️ ResNet V1 (6n+2)"""


def resnet_v1(input_shape, depth, num_classes=10):
    """
    ResNet V1 model builder (6n+2 layers).
    """
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (e.g 20, 32, 44)")

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Stacks of residual blocks
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # First layer of each stack (except first)
                num_filters *= 2
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                # Projection shortcut to match dimensions
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = Add()([x, y])
            x = Activation("relu")(x)

    # Classification head
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        x
    )

    model = Model(inputs=inputs, outputs=outputs)
    return model


"""🏗️ ResNet V2 (9n+2)"""


def resnet_v2(input_shape, depth, num_classes=10):
    """
    ResNet V2 model builder (9n+2 layers).
    """
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (e.g 56, 110)")

    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # Stacks of residual blocks
    for stack in range(3):
        for res_block in range(num_res_blocks):
            activation = "relu"
            batch_normalization = True
            strides = 1

            if stack == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2  # downsample

            # Bottleneck block
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(
                inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False
            )
            if res_block == 0:
                # Projection shortcut
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = Add()([x, y])
        num_filters_in = num_filters_out

    # Classification head (BN-ReLU before pooling)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        x
    )

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Choose ResNet version
if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.summary()
print(model_type)

"""⚙️ Compile the Model"""


def lr_schedule(epoch):
    """
    Learning rate schedule:
    - Start with 1e-3
    - Decay at epochs 80, 120, 160, 180
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr


model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=lr_schedule(0)),
    metrics=["accuracy"],
)

"""🚀 Train the Model"""

checkpoint = ModelCheckpoint(
    filepath=f"model.cifar10_{model_type}.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation
if not data_augmentation:
    print("Not using data augmentation")
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks,
    )
else:
    print("Using real-time data augmentation")
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
    )
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        verbose=1,
        steps_per_epoch=x_train.shape[0] // batch_size,
        callbacks=callbacks,
    )

"""💾 Load Best Saved Model"""

model.load_weights(f"model.cifar10_{model_type}.keras")

"""📊 Evaluate the Model"""

scores = model.evaluate(x_test, y_test, verbose=1)
print(f"Test loss: {scores[0]}")
print(f"Test accuracy: {scores[1]}")

"""📈 Plot Training Curves"""

plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="test")
plt.legend()
plt.show()

"""🔮 Visualize Model Predictions"""

# Generate predictions
y_hat = model.predict(x_test)

cifar10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[idx])
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title(
        f"{cifar10_labels[pred_idx]} ({cifar10_labels[true_idx]})",
        color=("green" if pred_idx == true_idx else "red"),
    )
