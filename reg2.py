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

def seq2fp(seq):
    """ Get .nii filename associated w/ seq """
    
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
        
    return getfp(f"{fn}/{niis[0]}")

def fp2sitk(fp):
    """ Orient to LAS & Crop to Foreground """
    im_obj = sitk.ReadImage(fp)
    return threshold_based_crop(sitk.DICOMOrient(im_obj, "LAS"))
    
# Crop https://github.com/SimpleITK/ISBI2018_TUTORIAL/blob/master/python/03_data_augmentation.ipynb
def threshold_based_crop(image):
    '''
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.  
    '''
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])

# get standard reference domain

# src: https://github.com/SimpleITK/ISBI2018_TUTORIAL/blob/master/python/03_data_augmentation.ipynb
def get_reference_frame(objs, new_spacing):
    img_data = [(o.GetSize(), o.GetSpacing()) for o in objs]

    dimension = 3 # 3D MRs
    pixel_id = 8 # 2 = 16-bit signed integer

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    for img_sz, img_spc in img_data:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx else mx \
                                      for sz, spc, mx in zip(img_sz, img_spc, reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension     
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    # Non-Isotropic pixels
    reference_spacing = new_spacing
    reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    # Set reference image attributes
    reference_image = sitk.Image(reference_size, pixel_id)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    return reference_image, (reference_size, pixel_id, reference_origin, reference_spacing, reference_direction, reference_center)

def get_reference_image(reference_frame):
    reference_size, pixel_id, reference_origin, reference_spacing, reference_direction, reference_center = reference_frame
    reference_image = sitk.Image(reference_size, pixel_id)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    return reference_image, reference_center

def resample2reference(img, reference_image, reference_center, \
                       interpolator = sitk.sitkLinear, default_intensity_value = 0.0, dimension=3):
    
    # Define translation transform mapping origins from reference_image to the current img
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_image.GetOrigin())
    
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    
    centered_transform = sitk.CompositeTransform([transform, centering_transform])
    
#     centered_transform = sitk.Transform(transform)
#     centered_transform.AddTransform(centering_transform)
    
    return sitk.Resample(img, reference_image, centered_transform, interpolator, default_intensity_value, img.GetPixelID())



seqs = ["COR T1", "COR T2", "+COR T1"]
fps  = {seq:seq2fp(seq) for seq in seqs}
objs = {seq:fp2sitk(fp) for seq,fp in fps.items()}

for seq,obj in objs.items():
    print("*"*50)
    print(seq)
    print_sitk_info(obj)
    print("*"*50)

# resample to same domain
new_spacing = (0.41015625, 3.0, 0.41015625)
reference_image, reference_frame = get_reference_frame(objs.values(), new_spacing)

# unpack reference frame
reference_size, pixel_id, reference_origin, reference_spacing, reference_direction, reference_center = reference_frame

# print
print("Reference Image")
print_sitk_info(reference_image)

# resample to reference
resampled_objs = {seq:resample2reference(o, reference_image, reference_center) for seq,o in objs.items()}

for seq,obj in resampled_objs.items():
    print("*"*50)
    print(seq)
    print_sitk_info(obj)
    print("*"*50)

# ########################################
# ALIGN
# ########################################

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

def sitk2itk(sitk_image):
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
fixed_image = sitk2itk(resampled_objs[fixed_seq])

elapsed = time.time() - start
print(f"Elapsed: {elapsed:0.2f} s.")
    
    
for moving_seq in ["COR T2", "+COR T1"]:
    # time it
    print("Aligning", moving_seq)
    start = time.time()

    moving_image = sitk2itk(resampled_objs[moving_seq])
    result_image = align(fixed_image, moving_image)

    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:0.2f} s.")
    
    # Save image with itk
    output_fn = getfp(f"elastix/resampled_{moving_seq}_aligned_to_{fixed_seq}.nii")
    Path(getfp(f"elastix")).mkdir(parents=True, exist_ok=True)
    
    itk.imwrite(result_image, output_fn)


elapsed = time.time() - START
print(f"Elapsed since START: {elapsed:0.2f} s.")
print("Done.")
