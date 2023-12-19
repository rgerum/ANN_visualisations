import numpy as np
from includes import draw
import matplotlib.pyplot as plt


if 0:
    layer1 = draw.addLayer(5, pos=0, text=lambda x: f"$x_{x+1}$", color="C0", orientation="vertical")
    draw.addText(0, -2.8, "Input\nLayer")
    layer2 = draw.addLayer(3, pos=2, text=lambda x: f"$h_{x+1}$", color="C2", orientation="vertical")
    draw.addText(2, -2.8, "Hidden\nLayer")
    layer3 = draw.addLayer(2, pos=4, text=lambda x: f"$y_{x+1}$", color="C3", orientation="vertical")
    draw.addText(4, -2.8, "Output\nLayer")

    draw.connectLayersDense(layer1, layer2)
    draw.connectLayersDense(layer2, layer3)

    plt.savefig("test_out1.png", dpi=80)
elif 1:
    #plt.subplot(221)

    layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", "C0")
    layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", lambda x: "C2" if x == 2 else "w")

    draw.connectLayersDense(layer1, layer2)

    draw.crop_figure()
    plt.savefig("conn1.png", dpi=160)
    plt.clf()

    #plt.subplot(222)
    layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if 0 < x <= 3 else "w")
    layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", lambda x: "C2" if x == 2 else "w")

    draw.connectLayersConv(layer1, layer2, 3)

    draw.crop_figure()
    plt.savefig("conn2.png", dpi=160)
    plt.clf()

    #plt.subplot(223)
    layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if x == 2 else "w")
    layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", "C2")

    draw.connectLayersDense(layer1, layer2)

    draw.crop_figure()
    plt.savefig("conn3.png", dpi=160)
    plt.clf()

    layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if x == 2 else "w")
    layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", lambda x: "C2" if 0 < x <= 3 else "w")

    draw.connectLayersConv(layer1, layer2, 3)

    draw.crop_figure()
    plt.savefig("conn4.png", dpi=160)
    plt.clf()

    layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if -1 < x <= 4 else "w")
    layer2 = draw.addLayer(5, 1.5, lambda x: f"$h_{x + 1}$", lambda x: "C2" if 0 < x <= 3 else "w")
    layer3 = draw.addLayer(5, 3.0, lambda x: f"$g_{x + 1}$", lambda x: "C4" if x == 2 else "w")

    draw.connectLayersConv(layer1, layer2, 3)
    draw.connectLayersConv(layer2, layer3, 3)

    draw.crop_figure()
    plt.savefig("conn5.png", dpi=160)
    plt.show()
    plt.clf()
elif 1:
    layer1 = addLayer(5, 0, lambda x: f"$x_{x + 1}$", "w")
    layer2 = addLayer(5, 1.5, lambda x: f"$h_{x + 1}$", "w")

    connectLayersDense(layer1, layer2, lambda x, y: "k" if x == 2 and y == 2 else "gray", lambda x, y: 2 if x == 2 and y == 2 else 1)

    crop_figure()
    #plt.show()
    plt.savefig("conn1.png", dpi=160)
    plt.clf()

    layer1 = addLayer(5, 0, lambda x: f"$x_{x + 1}$", "w")
    layer2 = addLayer(5, 1.5, lambda x: f"$h_{x + 1}$", "w")

    connectLayersConv(layer1, layer2, 3, lambda x, y: "k" if x == y else "gray", lambda x, y: 2 if x == y else 1)

    crop_figure()
    plt.savefig("conn2.png", dpi=160)
    plt.show()
else:
    from tensorflow import keras
    model = keras.models.Sequential([
        keras.layers.Input((32, 32, 1)),
        keras.layers.Conv2D(64, 3),
        keras.layers.MaxPool2D(2),
        keras.layers.Conv2D(32, 3),
    #    keras.layers.MaxPool2D(2),
    #    keras.layers.Flatten(),
    #    keras.layers.Dense(10),
    ])
    plt.cla()
    colors = {
        "Conv2D": "C4",
        "MaxPooling2D": "C5",
        "Flatten": "C1",
        "Dense": "C2",
    }
    old_layer = plotBox(0, -1, "Input", width=2, height=0.7, text2=list(model.input.shape[1:]))
    for i, layer in enumerate(model.layers):
        print(layer.output.shape)
        l1 = plotBox(0, i, layer.name, color=colors[type(layer).__name__], width=2, height=0.7, text2=list(layer.output.shape[1:]))
        if old_layer is not None:
            plotArrow(old_layer, l1)
        old_layer = l1



    if 0:
        for ax in plt.gcf().axes:
            # Get dimensions of y-axis in pixels
            y1, y2 = plt.gca().get_window_extent().get_points()[:, 1]

            ymin = plt.gca().get_position().y0
            ymax = plt.gca().get_position().y1
            ymin, ymax = ax.get_ylim()
            # Get unit scale
            print(y2, y1, ymax, ymin)
            yscale = (y2 - y1) / (ymax - ymin)
            for text in ax.texts:
                if scale := getattr(text, "data_fontsize", None):
                    # We want 2 of these as fontsize
                    fontsize = scale * yscale
                    text.set_fontsize(fontsize)
draw.crop_figure()
plt.savefig("test_out1.png", dpi=80)
plt.savefig("test_out2.png", dpi=160)
plt.show()
