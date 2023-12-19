from includes.draw import Image

draw = Image()
layer1 = draw.addLayer(5, pos=0, text=lambda x: f"$x_{x+1}$", color="C0", orientation="vertical")
draw.addText(0, -2.8, "Input\nLayer")
layer2 = draw.addLayer(3, pos=2, text=lambda x: f"$h_{x+1}$", color="C2", orientation="vertical")
draw.addText(2, -2.8, "Hidden\nLayer")
layer3 = draw.addLayer(2, pos=4, text=lambda x: f"$y_{x+1}$", color="C3", orientation="vertical")
draw.addText(4, -2.8, "Output\nLayer")

draw.connectLayersDense(layer1, layer2)
draw.connectLayersDense(layer2, layer3)

draw.save("draw_mlp.png", dpi=80)
draw.show()
