import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
import time
import cv2
import copy
from scipy.io import savemat

# This script extracts patches, augment nodules and make the files ready for deep learning algorithm. 
# The script contians four section, each of which activated by a specific value of the variable "funtion"
# These sections are: 1) extract patches, 2) augment nodules, 2) crop the samples, 3) resize the samples to uniform voxel space, 4) visualizing the patches.
# The result patches will be saved at '../patches/'. This directory can be changed at the 
# end of crop volume section. Other import parameter of the script are defined as follow:

dir_prefix = '../scans'  # directory where Luna scans are located.
# patch_sizeB = [121, 121, 25]  # patch size to extract nodule volumes (selected such that nodule is included inside ROI after augmentation).
# This size is used only to extract patches from annotation.csv file (positive samples).
# If patch_size = (x,x,25), the value for patch_sizeB should be (x*sqrt(2),x*sqrt(2),25) to make sure the positive samples
# will fit inside the ROI after augmentation.
patch_size = [49, 49, 17]
patch_sizeB = [49, 49, 17]
dest_res = [0.625, 0.625, 2]  # uniform resolution along each axis
crop_size = [32, 32, 10]  # the size of final patches (in voxel space)
function = 0  # it determines which action should be done.  0: extract all patches from raw data by reading candidate csv file.
output_dir = "../patches/"
annotation_dir = "../annotations/"
# 1: Augment the positive samples
# 2: crop the samples (all samples including negatives and positives, and augmented samples)
# 3: resize the crop sized to have 48x48x16 dimension in voxel space.
# 4: visualize the patches in a directory. This directory can be updated in the very bottom of the script. This function is just for inspecting the extracted patches

def load_itk_image(filename):
    # input: address of .mhd and .raw files
    # output: CT scan stored in numpy array and physical origing and resolution spacing
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    # a routine for reading csv files. return csv file as a 2-D list
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def worldToVoxelCoord(worldCoord, origin, spacing):
    # convert world coordinate to voxel coordinates
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord.astype(int)


def normalizePlanes(npzarray):
    # Clip HUs to [-1000,400], and normalize it to [0,1]
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


def uniform_res(volume, orign_res, dest_res):
    # input: 3-D scan as the numpy array, origin_res: resolution of the scan, dest_res: final resolution we want to reach.
    xy_res = orign_res[2]
    z_res = orign_res[0]
    resize_factor = [xy_res / dest_res[0], xy_res / dest_res[0],
                     z_res / dest_res[2]]  # based on the origin_res and dest_res we compute the resize factor
    niformed_volume = ndimage.interpolation.zoom(volume, zoom=tuple(resize_factor),
                                                 order=3)  # resize the image to have the desired resolution
    return niformed_volume, resize_factor  # resize factor is returned back as well because it will be used to update the coordinate of nodule location.

def extract_nodules(load_dir_prefix, save_dir_prefix, patch_sizeH):
    # The function extracts the nodules by reading the annotation csv files
    # in this function both the ct scans and nodule coordinates are revised to match
    # the destination resolution. The patches are extracted from CT scans (after they went through re-sampling)
    # and are saved into save_dir.
    
    # annotation_file = load_dir_prefix + '/annotations.csv'
    annotation_file = output_dir + 'annotations.csv'
    annotation_list = readCSV(annotation_file)
    seriesuidS = [i[0] for i in annotation_list]

    for i in range(9):
        print(i)
        fold = load_dir_prefix + '/subset' + str(i)
        for name in glob.glob(fold + '/*.mhd'):
            seriesuid = name[
                        -68:-4]  # seriesuid is the same as the file name (excluding the extension of the file name)
            print(seriesuid)
            ct, numpyOrigin, numpySpacing = load_itk_image(name)
            ct = ct.astype(np.float32)
            ct_new = np.copy(ct)
            tmp = np.swapaxes(ct_new, 0, 2)
            ct_new = np.swapaxes(tmp, 0, 1)  # axis swapped in order to have the axial view of the scan
            ### no normalization
            # ct_new, resize_factor = uniform_res(ct_new, numpySpacing, dest_res)  # resolution of the scan uniformed.
            ct_new = normalizePlanes(ct_new)  # convert the HUs to the range of (0,1)
            # ct_new = np.pad(ct_new, ((patch_sizeH[0], patch_sizeH[0]), (patch_sizeH[1], patch_sizeH[1]), (patch_sizeH[2], patch_sizeH[2])),
            #                 'constant', constant_values=0)  # padding the scan with half the size the batch size because the nodule coordinate might be in the border of scan.
                         
            matches = [y for y, x in enumerate(seriesuidS) if
                       x == seriesuid]  # find the rows of the csv file that correspond to the same patient (seriesuid)
            print(matches)
            for match in matches:
                print(match)
                worldCoord = np.asarray([float(annotation_list[match][3]), float(annotation_list[match][2]), float(annotation_list[match][1])])  # read the coordinate of the nodule in csv fil
                voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin,
                                               numpySpacing)  # convert world coordinate to voxel coordinate
                voxelCoord_new = np.copy(voxelCoord)
                voxelCoord_new[0], voxelCoord_new[2] = voxelCoord_new[2], voxelCoord_new[0]
                voxelCoord_new[0], voxelCoord_new[1] = voxelCoord_new[1], voxelCoord_new[0]  # the x, y, and z coordinates are swapped the same way that ct scan coordinates swapped earlier.

                # voxelCoord_new = np.round(voxelCoord_new * np.asarray(resize_factor)).astype(int)  # Since the CT scan resized we have to update the nodule coordinate
                # voxelCoord_new = voxelCoord_new + patch_sizeH  # because of the padding to the CT scan earlier, need to add the same padding size to the coordinate.
                patch_new = ct_new[voxelCoord_new[0] - patch_sizeH[0]:voxelCoord_new[0] + patch_sizeH[0] + 1,
                            voxelCoord_new[1] - patch_sizeH[1]:voxelCoord_new[1] + patch_sizeH[1] + 1,
                            voxelCoord_new[2] - patch_sizeH[2]:voxelCoord_new[2] + patch_sizeH[
                                2] + 1]  # extract the patch from the center of the nodule coordinate
                
                save_dir = save_dir_prefix + '/' + str(patch_sizeB[0]) + 'x' + str(patch_sizeB[1]) + 'x' + str(
                    patch_sizeB[2]) + '/Nodule/Fold' + str(i) + '/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_dir + str(match + 1),
                        patch_new)  # The name of the patch - that is saved to the hard disk - is the row number in annotation csv file.
    

def extract_candidates(load_dir_prefix, save_dir_prefix, patch_sizeH):
    # The function is the same as the extract nodules function, but instead of nodules it extracts the candidates.
    # for further detail about each each line, go to the corresponding line in extract_nodules function
    # This function may be merged with extract_nodules function in the future.
    # candidate_file = load_dir_prefix + '/candidates_V2.csv'
    
    # candidate_file = "/gpfs_projects/moktari.mostofa/Fall_2023/LUNA2016_challenge/LUNA16_Patch_Generation_Aug23/candidates_V2.csv"
    candidate_file = annotation_dir + 'candidates_V2_kernel_excluded.csv'
    candidate_list = readCSV(candidate_file)
    lesion_type = [i[4] for i in candidate_list]
    seriesuidS = [i[0] for i in candidate_list]

    for i in range(9,10):
        print(i)
        fold = load_dir_prefix + '/subset' + str(i)
        for name in glob.glob(fold + '/*.mhd'):
            seriesuid = name[-68:-4]
            ct, numpyOrigin, numpySpacing = load_itk_image(name)
            ct = ct.astype(np.float32)
            ct_new = np.copy(ct)
            tmp = np.swapaxes(ct_new, 0, 2)
            ct_new = np.swapaxes(tmp, 0, 1)
            # ct_new, resize_factor = uniform_res(ct_new, numpySpacing,dest_res)
            ct_new = normalizePlanes(ct_new)
            ct_new = np.pad(ct_new, ((patch_sizeH[0], patch_sizeH[0]), (patch_sizeH[1], patch_sizeH[1]), (patch_sizeH[2], patch_sizeH[2])),
                            'constant', constant_values=0)

            matches = [y for y, x in enumerate(seriesuidS) if x == seriesuid]

            for match in matches:
                worldCoord = np.asarray(
                    [float(candidate_list[match][3]), float(candidate_list[match][2]), float(candidate_list[match][1])])
                voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)

                voxelCoord_new = np.copy(voxelCoord)
                voxelCoord_new[0], voxelCoord_new[2] = voxelCoord_new[2], voxelCoord_new[0]
                voxelCoord_new[0], voxelCoord_new[1] = voxelCoord_new[1], voxelCoord_new[0]

                # voxelCoord_new = np.round(voxelCoord_new*np.asarray(resize_factor)).astype(int)
                voxelCoord_new = voxelCoord_new + patch_sizeH
                patch_new = ct_new[voxelCoord_new[0] - patch_sizeH[0]:voxelCoord_new[0] + patch_sizeH[0] + 1,
                            voxelCoord_new[1] - patch_sizeH[1]:voxelCoord_new[1] + patch_sizeH[1] + 1,
                            voxelCoord_new[2] - patch_sizeH[2]:voxelCoord_new[2] + patch_sizeH[2] + 1]
            
                print(patch_new.shape, 'patch_new_shape')
                save_dir = save_dir_prefix + '/' + str(patch_size[0]) + 'x' + str(patch_size[1]) + 'x' + str(patch_size[2]) + '/Non-Nodule/Fold' + str(i) + '/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if(lesion_type[match] == '0'):  # if non-nodule
                    np.save(save_dir + str(match+1),patch_new)
                
                save_dir = save_dir_prefix + '/' + str(patch_size[0]) + 'x' + str(patch_size[1]) + 'x' + str(patch_size[2]) + '/Nodule_CandList/Fold' + str(i) + '/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if(lesion_type[match] == '1'):  # if nodule
                    np.save(save_dir + str(match+1),patch_new)



if (function == 0):
    load_dir_prefix = dir_prefix
    save_dir_prefix = output_dir + 'test2/'
    patch_sizeH = [i // 2 for i in patch_sizeB]
    extract_nodules(load_dir_prefix, save_dir_prefix,patch_sizeH)
    patch_sizeH = [i // 2 for i in patch_size]
    extract_candidates(load_dir_prefix, save_dir_prefix, patch_sizeH)
