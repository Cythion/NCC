from cellpose import models
from skimage.io import imread, imsave
from skimage import restoration
from skimage.util import invert
from os import listdir

model = models.CellposeModel(
        gpu=True,
        model_type='cyto3',
        diam_mean=10,
        )


path = "/home/cyril/Documents/Programs/NCC/samples/bright/"
save = "/home/cyril/Documents/Programs/NCC/output/"
samples = listdir(path)

for index in samples:
    img = imread(path + index)
    print(f'Processing sample: {index}')
    
    save_path = save + index[:-4] + "_mask.tif"
    masks, flows, styles  = model.eval(img, flow_threshold=0.4, cellprob_threshold=0.0)
    imsave(save_path, masks)
