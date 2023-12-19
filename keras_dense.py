import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve

from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.models.Sequential([
    keras.layers.Input((28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile("adam", loss="categorical_crossentropy", metrics="accuracy")
model.summary()

if 1:
    model.fit(x_train, keras.utils.to_categorical(y_train), epochs=10)

    data = model.layers[1].weights[0].numpy().reshape(28, 28, 64)

    plt.figure(0, (8, 8))
    plt.axes([0, 0, 1, 1])
    for i in range(64):
        x, y = np.divmod(i, 8)
        plt.imshow(data[:, :, i], extent=[28*x, 28+28*x, 28+28*y, 28*y], cmap="gray")
    plt.xlim(0, 28*8)
    plt.ylim(28*8, 0)
    plt.savefig("test3.png")



model2 = keras.models.Sequential([
    keras.layers.Input((28, 28, 1)),
    keras.layers.Conv2D(64, kernel_size=5, activation="relu", padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(128, kernel_size=5, activation="relu", padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax"),
])

model2.compile("adam", loss="categorical_crossentropy", metrics="accuracy")
model2.summary()

model2.fit(x_train[:, :, :, None], keras.utils.to_categorical(y_train), epochs=10)

data = model2.layers[0].weights[0].numpy()[:, :, 0, :]

plt.figure(0, (8, 8))
plt.axes([0, 0, 1, 1])
for i in range(64):
    x, y = np.divmod(i, 8)
    plt.imshow(data[:, :, i], extent=[6*x, 5+6*x, 5+6*y, 6*y], cmap="gray")
plt.xlim(0, 6*8)
plt.ylim(6*8, 0)
plt.savefig("test2.png")