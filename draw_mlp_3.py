import pylustrator
pylustrator.start()
from includes.draw import Image
import numpy as np
from matplotlib import colors
#weight = np
np.random.seed(124)
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})

def error(i):
    f = np.random.rand()
    color = np.array(colors.to_rgb("#8054cc"))
    return color * f

draw = Image()
layer1 = draw.addLayer(4, pos=0, text=lambda x: f"$x_{x+1}$", color=error, orientation="vertical")
draw.addText(0, -2.2, "Input\nLayer")
layer2 = draw.addLayer(3, pos=2, text=lambda x: f"$h_{x+1}$", color=error, orientation="vertical")
draw.addText(2, -2.2, "Hidden\nLayer")
layer3 = draw.addLayer(2, pos=4, text=lambda x: f"$y_{x+1}$", color=error, orientation="vertical")
draw.addText(4, -2.2, "Output\nLayer")

draw.connectLayersDense(layer2, layer1)
draw.connectLayersDense(layer3, layer2)

draw.save("draw_mlp.pdf", dpi=80)
draw.save("draw_mlp.png", dpi=80)
draw.show()
