from includes.draw import Image

draw = Image()
"""
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=x_train.shape[1:]),
    keras.layers.Lambda(lambda x: x/255),

    augmentation,

    Conv2DNew(p.conv1(32), 5, 2, activation='relu', kernel_initializer='he_uniform'),
    RegLayer(p.reg1(1.), p.reg1value(1.)),
    #DimensionReg(p.reg1(0.), p.reg1value(1.)),

    Conv2DNew(p.conv2(64), 5, 2, activation='relu', kernel_initializer='he_uniform'),
    #DimensionRegGammaWeights(p.reg2(0.), p.reg2value(0.)),

    Conv2DNew(p.conv3(128), 3, 1, activation='relu', kernel_initializer='he_uniform'),
    #DimensionRegGammaWeights(p.reg3(0.), p.reg3value(0.)),

    keras.layers.Flatten(),
    keras.layers.Dense(units=p.dense1(1024), activation='relu'),
    #DimensionReg(p.reg4(0.), p.reg4value(0.)),
    keras.layers.Dense(units=num_classes, activation='softmax'),
])
"""
layers = {
    "1_Input": ["C0", "Input", "(32, 32, 3)"],
    "2_Conv2D": ["C1", "FreeConv2D 64", "(15, 15, 64)"],
    "2_Flatten": ["C2", "Flatten", "(14400)"],
    "4_Dense": ["C3", "Dense 10", "(10)"],
}
layers = {
    "1_Input": ["C0", "Input", "(32, 32, 3)"],
    "2_Flatten": ["C2", "Flatten", "(3072)"],
    "3_Dense": ["C3", "Dense 1024", "(1024)"],
    "4_Dense": ["C3", "Dense 10", "(10)"],
}
x = 0
y = 0
layer_old = None
for name, layer in layers.items():
    color, layer, shape = layer
    layer1 = draw.plotBox(x, y, layer, color=color, height=1, width=2, text2=shape)
    if layer_old is not None:
        draw.plotArrow(layer_old, layer1, color="k")
    layer_old = layer1
    y += 1.5

draw.save("draw_box_net.png", dpi=80)
draw.save("draw_box_net.svg", dpi=80)
draw.show()
