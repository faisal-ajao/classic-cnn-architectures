"""📥 Load and Explore the MNIST Dataset"""

# Import the MNIST dataset from Keras
from tensorflow.keras.datasets import mnist

# Load training and test data from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Dataset loaded successfully.")
print(f"The MNIST database has a training set of {len(x_train)} examples.")
print(f"The MNIST database has a test set of {len(x_test)} examples.")

"""👀 Visualize Sample Images"""

import matplotlib.pyplot as plt
import numpy as np

# Plot the first six training images with their labels
fig = plt.figure(figsize=(20, 20))
for i in range(6):
    ax = fig.add_subplot(1, 6, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(str(y_train[i]))

"""🔎 Inspect Pixel Values of a Single Image"""


def visualize_input(img, ax):
    """
    Helper function to display an image with its pixel values.
    Darker pixels are annotated with white text for visibility.
    """
    ax.imshow(img, cmap="gray")
    thresh = img.max() / 2.5
    width, height = img.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(
                text=str(round(img[x][y], 2)),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if img[x][y] < thresh else "black",
            )


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
visualize_input(x_train[0], ax)

"""⚖️ Normalize the Dataset"""

# Normalize data: subtract mean and divide by standard deviation
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

"""🔢 One-Hot Encode Labels"""

from tensorflow.keras.utils import to_categorical

# Get number of classes (10 digits: 0–9)
num_classes = len(np.unique(y_train))

print("Integer-valued labels:")
print(y_train[:10])

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("One-hot labels:")
print(y_train[:10])

"""📐 Reshape Data for CNN Input"""

# Reshape images to include channel dimension (28x28x1)
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print("input_shape:", input_shape)
print("x_train shape:", x_train.shape)

"""🏗️ Define the LeNet-5 Model Architecture"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense

# Initialize the model
model = Sequential()

# Input layer
model.add(Input(shape=(28, 28, 1)))

# C1 Convolutional Layer (6 filters, 5x5 kernel, tanh activation)
model.add(
    Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation="tanh", padding="same")
)

# S2 Average Pooling Layer
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid"))

# C3 Convolutional Layer (16 filters)
model.add(
    Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation="tanh", padding="valid")
)

# S4 Average Pooling Layer
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid"))

# C5 Convolutional Layer (120 filters)
model.add(
    Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation="tanh", padding="valid")
)

# Flatten before Fully Connected layers
model.add(Flatten())

# FC6 Fully Connected Layer (84 units)
model.add(Dense(84, activation="tanh"))

# FC7 Output Layer (10 units, Softmax for classification)
model.add(Dense(10, activation="softmax"))

# Display model summary
model.summary()

"""⚙️ Compile the Model"""

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

"""🚀 Train the Model"""

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def lr_schedule(epoch):
    """
    Custom learning rate scheduler:
    - Higher LR in early epochs
    - Gradual decay in later epochs
    """
    if epoch <= 2:
        lr = 5e-4
    elif epoch > 2 and epoch <= 5:
        lr = 2e-4
    elif epoch > 5 and epoch <= 9:
        lr = 5e-5
    else:
        lr = 1e-5
    return lr


# Define callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)
checkpointer = ModelCheckpoint(
    filepath="model.lenet5_mnist.keras",
    verbose=1,
    save_best_only=True,
)

# Train model
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[checkpointer, lr_scheduler],
    verbose=2,
    shuffle=True,
)

"""💾 Load the Best (Saved) Model"""

model.load_weights("model.lenet5_mnist.keras")

"""📊 Evaluate the Model"""

score = model.evaluate(x_test, y_test, verbose=0)
accuracy = score[1] * 100
print("\n", "Test accuracy: %.4f%%" % accuracy)

"""📈 Plot Training Accuracy"""

f, ax = plt.subplots()
ax.plot([None] + history.history["accuracy"], "o-")
ax.plot([None] + history.history["val_accuracy"], "x-")

ax.legend(["Train acc", "Validation acc"])
ax.set_title("Training/Validation Acc per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Acc")
plt.show()

"""📉 Plot Training Loss"""

f, ax = plt.subplots()
ax.plot([None] + history.history["loss"], "o-")
ax.plot([None] + history.history["val_loss"], "x-")

ax.legend(["Train loss", "Val loss"])
ax.set_title("Training/Validation Loss per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.show()

"""🔮 Visualize Model Predictions"""

# Generate predictions on the test set
y_hat = model.predict(x_test)

# Plot a random sample of test images with predictions
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(x_test[idx], cmap="gray")
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title(
        f"{pred_idx} ({true_idx})", color=("green" if pred_idx == true_idx else "red")
    )
