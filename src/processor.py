from skimage import io, restoration, exposure
import numpy as np
import math
from os import listdir

path = "/home/cyril/Documents/Programs/NCC/samples/"
output_path = "/home/cyril/Documents/Programs/NCC/output/"
samples = listdir(path)



def zProjection(z_stack):
    # Take the size of the object and determine the middle third of the z axis
    zSize, xSize, ySize = z_stack.shape
    zThirds = math.floor(zSize / 3)
    stack_slice = z_stack[zThirds:zSize - zThirds, :, :]
    # The middle part is the then used to cacluclate the z projection
    avg_projection = np.mean(stack_slice, axis=0)
    return avg_projection

def subBackground(z_projection):
    background = restoration.rolling_ball(z_projection, radius=50, nansafe=True)
    clean_image = z_projection - background
    return clean_image


for index in samples:
    print(f'Processing sample: {index}')
    output_sample = output_path + index[:-4] + ".tif"
    z_stack = io.imread(path + index)
    avg_projection = zProjection(z_stack)
    denoised = restoration.denoise_nl_means(avg_projection)
    cleaned_img = subBackground(denoised)
    io.imsave(output_sample, cleaned_img)


# z_stack = io.imread(path + "WT-RAP-1_w1confGFP.stk")
#
# avg_projection = zProjection(z_stack)
# subSample = subBackground(avg_projection)
#
# io.imsave(path + "cleaned_output.tif", subSample)
