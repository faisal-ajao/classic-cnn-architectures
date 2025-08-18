"""📚 Import Libraries"""

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    AveragePooling2D,
    MaxPooling2D,
    BatchNormalization,
    Add,
    Flatten,
    Dense,
)
from tensorflow.keras.models import Model

"""🧱 Bottleneck Residual Block (Building Block of ResNet50)"""


def bottleneck_residual_block(x, f, filters, stage, block, reduce=False, s=2):
    """
    Implements a single bottleneck residual block.

    Parameters:
        x       : input tensor
        f       : filter size for the middle CONV layer
        filters : list of integers [F1, F2, F3]
        stage   : integer, current stage label (for naming layers)
        block   : string, current block label (for naming layers)
        reduce  : boolean, whether to reduce dimensions with stride
        s       : stride for convolution when reduce=True

    Returns:
        Tensor after applying bottleneck residual block.
    """

    # Weight initializer
    kernel_init = glorot_uniform(seed=0)

    # Naming convention for layers
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve filter dimensions
    F1, F2, F3 = filters

    # Save input tensor for the shortcut path
    x_shortcut = x

    # ============ Main Path ============
    if reduce:
        # First component of main path (with stride)
        x = Conv2D(
            filters=F1,
            kernel_size=(1, 1),
            strides=(s, s),
            padding="valid",
            name=conv_name_base + "2a",
            kernel_initializer=kernel_init,
        )(x)
        x = BatchNormalization(axis=3, name=bn_name_base + "2a")(x)
        x = Activation("relu")(x)

        # Adjust shortcut path dimensions
        x_shortcut = Conv2D(
            filters=F3,
            kernel_size=(1, 1),
            strides=(s, s),
            padding="valid",
            name=conv_name_base + "1",
            kernel_initializer=kernel_init,
        )(x_shortcut)
        x_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(x_shortcut)

    else:
        # First component of main path (no stride)
        x = Conv2D(
            filters=F1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name=conv_name_base + "2a",
            kernel_initializer=kernel_init,
        )(x)
        x = BatchNormalization(axis=3, name=bn_name_base + "2a")(x)
        x = Activation("relu")(x)

    # Second component of main path
    x = Conv2D(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=kernel_init,
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    # Third component of main path
    x = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2c",
        kernel_initializer=kernel_init,
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2c")(x)

    # Add shortcut to main path and pass through ReLU
    x = Add()([x, x_shortcut])
    x = Activation("relu")(x)

    return x


"""🏗️ Build the ResNet50 Architecture"""


def Resnet50(input_shape, classes):
    """
    Implements the ResNet50 architecture.

    Parameters:
        input_shape : tuple, shape of input images (e.g., (32,32,3))
        classes     : int, number of output classes

    Returns:
        model : Keras Model instance
    """

    # Input tensor
    x_input = Input(input_shape)

    # Stage 1
    x = Conv2D(
        64,
        kernel_size=(7, 7),
        strides=(2, 2),
        name="conv1",
        kernel_initializer=glorot_uniform(seed=0),
    )(x_input)
    x = BatchNormalization(axis=3, name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Stage 2 (3 blocks)
    x = bottleneck_residual_block(
        x, f=3, filters=[64, 64, 256], stage=2, block="a", reduce=True, s=1
    )
    x = bottleneck_residual_block(x, f=3, filters=[64, 64, 256], stage=2, block="b")
    x = bottleneck_residual_block(x, f=3, filters=[64, 64, 256], stage=2, block="c")

    # Stage 3 (4 blocks)
    x = bottleneck_residual_block(
        x, f=3, filters=[128, 128, 512], stage=3, block="a", reduce=True, s=2
    )
    x = bottleneck_residual_block(x, f=3, filters=[128, 128, 512], stage=3, block="b")
    x = bottleneck_residual_block(x, f=3, filters=[128, 128, 512], stage=3, block="c")
    x = bottleneck_residual_block(x, f=3, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4 (6 blocks)
    x = bottleneck_residual_block(
        x, f=3, filters=[256, 256, 1024], stage=4, block="a", reduce=True, s=2
    )
    x = bottleneck_residual_block(x, f=3, filters=[256, 256, 1024], stage=4, block="b")
    x = bottleneck_residual_block(x, f=3, filters=[256, 256, 1024], stage=4, block="c")
    x = bottleneck_residual_block(x, f=3, filters=[256, 256, 1024], stage=4, block="d")
    x = bottleneck_residual_block(x, f=3, filters=[256, 256, 1024], stage=4, block="e")
    x = bottleneck_residual_block(x, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5 (3 blocks)
    x = bottleneck_residual_block(
        x, f=3, filters=[512, 512, 2048], stage=5, block="a", reduce=True, s=2
    )
    x = bottleneck_residual_block(x, f=3, filters=[512, 512, 2048], stage=5, block="b")
    x = bottleneck_residual_block(x, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # Average Pooling
    x = AveragePooling2D(pool_size=(1, 1), name="avg_pool")(x)

    # Output layer
    x = Flatten()(x)
    x = Dense(
        classes,
        activation="softmax",
        name="fc" + str(classes),
        kernel_initializer=glorot_uniform(seed=0),
    )(x)

    # Create model
    model = Model(inputs=x_input, outputs=x, name="ResNet50")
    return model


"""🧪 Instantiate Model and Print Summary"""

model = Resnet50(input_shape=(32, 32, 3), classes=10)
model.summary()
