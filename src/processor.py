from skimage import io, restoration, measure
import numpy as np
import math
from os import listdir

path = "/home/cyril/Documents/Programs/NCC/samples/stk/"
output_path = "/home/cyril/Documents/Programs/NCC/output/"
samples = listdir(path)
SEG_IMG_PATH = "/home/cyril/Documents/Programs/NCC/output/non_bright_mask.tif"
GFP_IMG_PATH = "/home/cyril/Documents/Programs/NCC/output/non_GFP_prj.tif"
MCHR_IMG_PATH = "/home/cyril/Documents/Programs/NCC/output/non_mCherry_prj.tif"
OUTPUT_TXT = output_path + "output.txt"

NUCLEAR_CUTOFF = 400
NUCLEAR_THRESHOLD = 0.6



def zProjection(z_stack):
    # Take the size of the object and determine the middle third of the z axis
    zSize, xSize, ySize = z_stack.shape
    zThirds = math.floor(zSize / 3)
    stack_slice = z_stack[zThirds:zSize - zThirds, :, :]
    # The middle part is the then used to cacluclate the z projection
    avg_projection = np.mean(stack_slice, axis=0)
    return avg_projection

def cherryPickMiddle(z_stack):
    zSize, xSize, ySize = z_stack.shape
    middle = math.floor(zSize / 2)
    stack_slice = z_stack[middle, :, :]
    return stack_slice

def subBackground(z_projection):
    background = restoration.rolling_ball(z_projection, radius=50, nansafe=True)
    clean_image = z_projection - background
    return clean_image

def isolateCell(img, cell_props):
    min_row, min_col, max_row, max_col = cell_props.bbox
    slice = img[min_row:max_row, min_col:max_col]
    return slice

def createMask(slice):
    slice_height, slice_width = slice.shape

    # Determin the value in the middle of the picture which is the label
    # given by cellpose
    label_id = slice[math.floor(slice_height/2), math.floor(slice_width/2)]
    bool_mask = np.zeros((slice_height, slice_width), dtype=bool)
    int_mask = np.zeros((slice_height, slice_width), dtype=np.int8)

    # We check each pixel of the segmented image of cellpose for 
    # the label_id and if it is the id we change the value in the zero 
    # array mask. This removes adjacent cells.
    w = 0
    h = 0
    while h < slice_height:
        while w < slice_width:
            if slice[h,w] == label_id:
                bool_mask[h,w] = True
                int_mask[h,w] = 127
            w += 1
        w = 0
        h += 1
    return bool_mask, int_mask




def coeffMeasure(segmentation_img, gfp_img, mcherry_img):
    props = measure.regionprops(segmentation_img)

    i = 0
    for region_img in props:
        mask = createMask(region_img, segmentation_img)
        name = output_path + f'cell_{i}.tif'
        io.imsave(name, mask)
        i += 1


    


# for index in samples:
#     print(f'Processing sample: {index}')
#     z_stack = io.imread(path + index)
#     output_sample = output_path + index[:-4] + "_middle.tif"
#     io.imsave(output_sample, cherryPickMiddle(z_stack))
#     output_sample = output_path + index[:-4] + "_prj.tif"
#     avg_projection = zProjection(z_stack)
#     denoise = restoration.denoise_nl_means(avg_projection)
#     io.imsave(output_sample, denoise)

seg_img = io.imread(SEG_IMG_PATH)
gfp_img = io.imread(GFP_IMG_PATH)
mcherry_img = io.imread(MCHR_IMG_PATH)

props = measure.regionprops(seg_img)

i = 0
for cell_props in props:
    gfp_slice = isolateCell(gfp_img, cell_props)
    mcherry_slice = isolateCell(mcherry_img, cell_props)
    slice = isolateCell(seg_img, cell_props)
    bool_mask, int_mask = createMask(slice)
    props = measure.regionprops(int_mask, mcherry_slice)

    for index in props:
        if index.intensity_max > NUCLEAR_CUTOFF:
            tot += 1
            coeff, pvalue = measure.pearson_corr_coeff(gfp_slice, mcherry_slice, mask=bool_mask)
            with open(OUTPUT_TXT, "a") as txt:
                txt.write(str(coeff)+"\n")

            if coeff > NUCLEAR_THRESHOLD:
                print(f'Nuclear signal detected inside cell {i} with value {coeff}')
                name = output_path + f'cell_{i}_gfp.tif'
                io.imsave(name, gfp_slice)
                name = output_path + f'cell_{i}_mcherry.tif'
                io.imsave(name, mcherry_slice)
                nuc += 1

    i += 1

    


# while i < 10:
#     min_row, min_col, max_row, max_col = props[i].bbox
#     slice = img[min_row:max_row,min_col:max_col]
#     mr, mc = slice.shape
#     label_id = slice[math.floor(mr/2), math.floor(mc/2)]
#     print(label_id)
#     mask = np.zeros((max_row-min_row, max_col-min_col), dtype=np.int8)
#     w = 0
#     h = 0
#     while w < max_row-min_row:
#         while h < max_col-min_col:
#             if slice[w,h] == label_id:
#                 mask[w,h] = 127
#             h = h + 1
#         h = 0
#         w = w + 1
#     name = output_path + f'cell_{i}.tif'
#     io.imsave(name, mask)
#     name = output_path + f'cell_{i}_source.tif'
#     io.imsave(name, slice)
#     i = i + 1
