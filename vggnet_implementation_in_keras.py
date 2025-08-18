"""📚 Import Libraries"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPool2D, Dropout

"""🏗️ VGG-16 (Congfiguration D)"""

vgg_16 = Sequential()

# Input layer
vgg_16.add(Input(shape=(224, 224, 3)))

# Block 1: Two 3x3 conv layers (64 filters), followed by max pooling
vgg_16.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 2: Two 3x3 conv layers (128 filters), followed by max pooling
vgg_16.add(Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 3: Three 3x3 conv layers (256 filters), followed by max pooling
vgg_16.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 4: Three 3x3 conv layers (512 filters), followed by max pooling
vgg_16.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 5: Three 3x3 conv layers (512 filters), followed by max pooling
vgg_16.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Fully connected layers (classifier)
vgg_16.add(Flatten())
vgg_16.add(Dense(4096, activation="relu"))
vgg_16.add(Dropout(0.5))
vgg_16.add(Dense(4096, activation="relu"))
vgg_16.add(Dropout(0.5))

# Output layer: 1000-way classification (ImageNet)
vgg_16.add(Dense(1000, activation="softmax"))

# Model summary
vgg_16.summary()

"""🚀 VGG-19 (Congfiguration E)"""

vgg_19 = Sequential()

# Input layer
vgg_19.add(Input(shape=(224, 224, 3)))

# Block 1: Two 3x3 conv layers (64 filters), followed by max pooling
vgg_19.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 2: Two 3x3 conv layers (128 filters), followed by max pooling
vgg_19.add(Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 3: Four 3x3 conv layers (256 filters), followed by max pooling
vgg_19.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 4: Four 3x3 conv layers (512 filters), followed by max pooling
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Block 5: Four 3x3 conv layers (512 filters), followed by max pooling
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(Conv2D(512, (3, 3), strides=(1, 1), activation="relu", padding="same"))
vgg_19.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Fully connected layers (classifier)
vgg_19.add(Flatten())
vgg_19.add(Dense(4096, activation="relu"))
vgg_19.add(Dropout(0.5))
vgg_19.add(Dense(4096, activation="relu"))
vgg_19.add(Dropout(0.5))

# Output layer: 1000-way classification (ImageNet)
vgg_19.add(Dense(1000, activation="softmax"))

# Model summary
vgg_19.summary()
