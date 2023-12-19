import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import convolve

ax = plt.gca()
def addBox(x, y, w, h, color, text1, text2, text3):
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, color=color,
                               alpha=0.5, zorder=1000, clip_on=False))
    plt.text(x + 0.02, y + h / 2, text1, rotation=90,
             va="center", ha="center", size=12)
    plt.text(x + 0.05, y + h / 2, text2, rotation=90,
             va="center", ha="center")
    plt.text(x + 0.08, y + h / 2, text3, rotation=90,
             va="center", ha="center", size=12)



im = imageio.imread("chicago_skyline_shrunk_v2.bmp")
kernel = [[
    1, -1,
    1, -1,
]]

im2 = convolve(im, kernel)
plt.imsave("result.png", im2, cmap="gray")
plt.imsave("source.png", im, cmap="gray")


from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Input((5,1)),
#    keras.layers.Dense(5),
    keras.layers.Conv1D(1, 3, padding="same"),
])

model.summary()