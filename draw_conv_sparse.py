from includes.draw import Image

draw = Image()
layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", "C0")
layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", lambda x: "C2" if x == 2 else "w")

draw.connectLayersDense(layer1, layer2)

draw.save("draw_conv_sparse_1.png", dpi=160)

draw = Image()
layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if 0 < x <= 3 else "w")
layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", lambda x: "C2" if x == 2 else "w")

draw.connectLayersConv(layer1, layer2, 3)

draw.save("draw_conv_sparse_2.png", dpi=160)

draw = Image()
layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if x == 2 else "w")
layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", "C2")

draw.connectLayersDense(layer1, layer2)

draw.save("draw_conv_sparse_3.png", dpi=160)

draw = Image()
layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if x == 2 else "w")
layer2 = draw.addLayer(5, 2, lambda x: f"$h_{x + 1}$", lambda x: "C2" if 0 < x <= 3 else "w")

draw.connectLayersConv(layer1, layer2, 3)

draw.save("draw_conv_sparse_4.png", dpi=160)

draw = Image()
layer1 = draw.addLayer(5, 0, lambda x: f"$x_{x + 1}$", lambda x: "C0" if -1 < x <= 4 else "w")
layer2 = draw.addLayer(5, 1.5, lambda x: f"$h_{x + 1}$", lambda x: "C2" if 0 < x <= 3 else "w")
layer3 = draw.addLayer(5, 3.0, lambda x: f"$g_{x + 1}$", lambda x: "C4" if x == 2 else "w")

draw.connectLayersConv(layer1, layer2, 3)
draw.connectLayersConv(layer2, layer3, 3)

draw.save("draw_conv_sparse_5.png", dpi=160)
