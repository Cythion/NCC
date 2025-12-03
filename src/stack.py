from skimage import io, restoration, measure
import numpy as np
import math
from os import listdir

path = "/home/cyril/Documents/Programs/NCC/samples/snrk1/stk/"
output_path = "/home/cyril/Documents/Programs/NCC/output/snrk1/"
samples = listdir(path)
num_samples = len(samples)
current_sample = 1


def zProjection(z_stack):
    # Take the size of the object and determine the middle third of the z axis
    zSize, xSize, ySize = z_stack.shape
    zThirds = math.floor(zSize / 3)
    stack_slice = z_stack[zThirds:zSize - zThirds, :, :]
    # The middle part is the then used to cacluclate the z projection
    avg_projection = np.mean(stack_slice, axis=0)
    return avg_projection


# /////////////////
# // Start main  //
# /////////////////

for index in samples:
    print(f'Processing sample {current_sample}/{num_samples}: {index}')
    z_stack = io.imread(path + index)
    output_sample = output_path + index[:-4] + "_prj.tif"
    avg_projection = zProjection(z_stack)
    denoise = restoration.denoise_nl_means(avg_projection)
    io.imsave(output_sample, denoise)
    current_sample += 1
