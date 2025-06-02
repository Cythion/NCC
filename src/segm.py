from cellpose import models
from cellpose.io import imread, imsave

model = models.CellposeModel(gpu=True)

img = imread("test.tif")

masks, flows, styles, diams = model.eval(img)

imsave("output.tif", masks)
