from includes.draw import Image

draw = Image()
layer1 = draw.addLayer(5, pos=0, color="C0", orientation="vertical")
#draw.addText(0, -2.8, "Input\nLayer")
layer2 = draw.addLayer(3, pos=2, color="C2", orientation="vertical")
#draw.addText(2, -2.8, "Hidden\nLayer")
layer3 = draw.addLayer(2, pos=4, color="C3", orientation="vertical")
#draw.addText(4, -2.8, "Output\nLayer")

draw.connectLayersDense(layer1, layer2)
draw.connectLayersDense(layer2, layer3)

draw.save("draw_mlp_2.png", dpi=80)
draw.save("draw_mlp_2.svg", dpi=80)
draw.show()
