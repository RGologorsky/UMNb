# set root directory of data
DATA = f"/home/gologors/data/test/1.2.840.113711.98223.701.3360.497997740.26.2116281012.1240820"

# Template: https://github.com/InsightSoftwareConsortium/ITKElastix/blob/master/examples/ITK_Example01_SimpleRegistration.ipynb

# utilities
import os, sys, time, json
from pathlib import Path

# graphing
from matplotlib import gridspec, colors

import matplotlib.pyplot as plt
import seaborn as sns

# data
import numpy as np
import pandas as pd

# nii, registration
import itk
import SimpleITK as sitk

# interactive
from ipywidgets import interact, interactive, IntSlider, ToggleButtons, fixed

# helpers

# add root to filepath (=os.path.join)
def getfp(fn, root=DATA): return f"{root}/{fn}"

# sitk obj and np array have different index conventions
# deep copy
def sitk2np(obj): return np.swapaxes(sitk.GetArrayFromImage(obj), 0, 2)
def np2sitk(arr): return sitk.GetImageFromArray(np.swapaxes(arr, 0, 2))

# round all floats in a tuple to 3 decimal places
def round_tuple(t, d=2): return tuple(round(x,d) for x in t)

def orient_test(image):
    orient = sitk.DICOMOrientImageFilter()
    orient.DebugOn()
    print(round_tuple(image.GetDirection(), d=2))
    print(orient.GetOrientationFromDirectionCosines(image.GetDirection()))

# print sitk info
def print_sitk_info(image):   
    orient = sitk.DICOMOrientImageFilter()

    print("Size: ", image.GetSize())
    print("Origin: ", image.GetOrigin())
    print("Spacing: ", image.GetSpacing())
    print("Direction: ", round_tuple(image.GetDirection(), d=2))
    print("Orientation: ", orient.GetOrientationFromDirectionCosines(image.GetDirection()))
    print(f"Pixel type: {image.GetPixelIDValue()} = {image.GetPixelIDTypeAsString()}")

# read sequence names from csv
series_info = pd.read_csv(getfp("Series_2.csv"), header=None)
print(series_info.loc[:,5])

#################################
# CODE
###############################

START = time.time()

# Sequences to align

# We align the following: Cor T1, Cor T2, +Cor T1 (i.e. with contrast).

seqs = ["COR T1", "COR T2", "+COR T1"]

def get_arr(seq):
    # get filepath to sequence
    seq_idx = series_info.loc[:,5] == seq
    fn      = series_info.loc[seq_idx,1].values[0]
    
    # get files in dir
    all_files = sorted(os.listdir(getfp(fn)))
    niis  = [x for x in all_files if x.endswith(".nii.gz")]
    jsons = [x for x in all_files if x.endswith(".json")] 
    dcms  = [x for x in all_files if x.endswith(".dcm")]
    
    # check if more than one
    if len(niis) != 1:
        print(f"Nii isn't unique! {seq}")
        
    # open .nii files
    im_obj = sitk.ReadImage(getfp(f"{fn}/{niis[0]}"))
    
    # reorient to LAS
    im_obj = sitk.DICOMOrient(im_obj, "LAS")
        
    im_arr = sitk2np(im_obj)
    
    return im_arr

ims = [get_arr(seq) for seq in seqs]

# Alignment

# Template:
# - https://github.com/InsightSoftwareConsortium/ITKElastix/blob/master/examples/ITK_Example01_SimpleRegistration.ipynb

# SITK -> ITK - safest way is thru numpy
# https://discourse.itk.org/t/in-python-how-to-convert-between-simpleitk-and-itk-images/1922/2


def get_fn(seq):
    # get filepath to sequence
    seq_idx = series_info.loc[:,5] == seq
    fn      = series_info.loc[seq_idx,1].values[0]
    
    return fn

def seq2itk(seq):
    
    # get filepath to sequence
    seq_idx = series_info.loc[:,5] == seq
    fn      = series_info.loc[seq_idx,1].values[0]
    
    # get files in dir
    all_files = sorted(os.listdir(getfp(fn)))
    niis  = [x for x in all_files if x.endswith(".nii.gz")]
    jsons = [x for x in all_files if x.endswith(".json")] 
    dcms  = [x for x in all_files if x.endswith(".dcm")]
    
    # check if more than one
    if len(niis) != 1:
        print(f"Nii isn't unique! {seq}")
        
    # open .nii files
    sitk_image = sitk.ReadImage(getfp(f"{fn}/{niis[0]}"))
    
    # reorient to LAS
    sitk_image = sitk.DICOMOrient(sitk_image, "LAS")
        
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(sitk_image), is_vector = sitk_image.GetNumberOfComponentsPerPixel()>1)
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())   
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()), [sitk_image.GetDimension()]*2)))
    
    return itk_image

# Back to a simpleitk image from the itk image
# https://discourse.itk.org/t/in-python-how-to-convert-between-simpleitk-and-itk-images/1922/2
def itk2sitk(itk_image):
    new_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image), isVector=itk_image.GetNumberOfComponentsPerPixel()>1)
    new_sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    new_sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    new_sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten()) 
    return new_sitk_image

def align(fixed_image, moving_image):
    # parameter map
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid_parameter_map)

    # Call registration function
    result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object=parameter_object,
    log_to_console=False)
    
    print(result_transform_parameters)
    print(result_image.shape, "result image shape")

    return result_image

print("Seq2ITK")

start = time.time()

fixed_seq   = "COR T1"
fixed_image = seq2itk(fixed_seq)

elapsed = time.time() - start
print(f"Elapsed: {elapsed:0.2f} s.")
    
    
for moving_seq in ["COR T2", "+COR T1"]:
    # time it
    print("Aligning", moving_seq)
    start = time.time()

    moving_image = seq2itk(moving_seq)
    result_image = align(fixed_image, moving_image)

    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:0.2f} s.")
    
    # Save image with itk
    output_fn = getfp(f"elastix/{moving_seq}_aligned_to_{fixed_seq}.nii")
    Path(getfp(f"elastix")).mkdir(parents=True, exist_ok=True)
    
    itk.imwrite(result_image, output_fn)


elapsed = time.time() - START
print(f"Elapsed since START: {elapsed:0.2f} s.")
print("Done.")