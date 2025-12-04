from skimage import io, restoration, measure
import numpy as np
import math
import re # Regex lib
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir

IMPORT_PATH = "/home/cyril/Documents/Programs/NCC/output/snrk1/"
EXPORT_PATH = "/home/cyril/Documents/Programs/NCC/summaries/"

CONDITIONS = ['Exp', '10min', '20min', '30min', '45min']
MUTANTS = ['SNF1', 'SnRK1']

FILTER_PATTERN = ".*mask.tif"
SUB_PATTERN = "_mask.tif"

NUCLEAR_CUTOFF = 40
NUCLEAR_THRESHOLD = 0.75

def filterList(pattern, sample_list):
    filtered_list = []
    for sample in sample_list:
        x = re.search(pattern, sample)
        if x:
            filtered_list.append(sample)
        else:
            continue
    return filtered_list

def getSampleList(filter_pattern, sub_pattern, sample_list):
    samples = []
    filtered_list = filterList(filter_pattern, sample_list)

    for sample in filtered_list:
        samples.append(re.sub(sub_pattern, "", sample))

    return samples

def loadSampleSet(sample_name, path):
    seg_img = io.imread(IMPORT_PATH + sample_name + "_mask.tif")
    gfp_img = io.imread(IMPORT_PATH + sample_name + "_GFP_prj.tif")
    mch_img = io.imread(IMPORT_PATH + sample_name + "_mCherry_prj.tif")
    return seg_img, gfp_img, mch_img
            

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

def getNucProps(name, path, data):
    # Load all images needed for the measurement
    seg_img, gfp_img, mch_img = loadSampleSet(name, path)

    # Get the bounding box of each cell by using the region props from the mask
    props = measure.regionprops(seg_img)

    for cell in props:
        seg_slice = isolateCell(seg_img, cell)
        gfp_slice = isolateCell(gfp_img, cell)
        mch_slice = isolateCell(mch_img, cell)

        # The regionprops function needs a intiger mask and the pearson func
        # needs a boolean mask I just make mask of both types to satisfy these two
        bool_mask, int_mask = createMask(seg_slice)

        # Not all nuclear signals are good so I cutoff the weak ones by looking
        # at a certain threshold of intensity
        mch_props = measure.regionprops(int_mask, mch_slice)

        for mch_sig in mch_props:
            if mch_sig.intensity_max > NUCLEAR_CUTOFF:
                coeff, pvalue = measure.pearson_corr_coeff(gfp_slice, mch_slice, mask=bool_mask)
                if coeff >= 0:
                    data.append(coeff)

def countNucCells(data):
    num_total = 0
    num_nuclear = 0
    for entries in data:
        if entries > NUCLEAR_THRESHOLD:
            num_nuclear += 1
        num_total += 1

    return num_nuclear, num_total



# ########
# # Main #
# ########

sample_names_raw = listdir(IMPORT_PATH)
sample_names = getSampleList(FILTER_PATTERN, SUB_PATTERN, sample_names_raw)

data_frame = pd.DataFrame()
for mut in MUTANTS:
    total = 0
    nuclear = 0
    for cond in CONDITIONS:
        data = []
        print(f'Processing {mut} samples in {cond} condition:')

        for names in sample_names:
            if re.search(mut, names) and re.search(cond, names):
                getNucProps(names, IMPORT_PATH, data)
        num_nuclear, num_total = countNucCells(data)
        print(f'Nuclear: {num_nuclear}\nTotal: {num_total}\nPercentage Nuclear: {num_nuclear/num_total*100}')

        temp_dataframe = pd.DataFrame()
        temp_dataframe [(f'{cond}',f'{mut}')] = data
        data_frame = pd.concat([data_frame, temp_dataframe], ignore_index=True)

data_frame.to_csv(EXPORT_PATH + "snrk1Summary.csv")
