__author__ = 'jason'

import os
import numpy as np
from skimage.io import imread
from skimage import morphology
from skimage.filter import threshold_otsu, rank
from skimage import measure
import matplotlib.pyplot as plt


def get_max_region(label_list, imagethres):
    regions = measure.regionprops(label_list)
    region_max = None
    for region in regions:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[label_list == region.label])*1.0/region.area < 0.50:
            continue
        if region_max is None:
            region_max = region
        if region_max.filled_area < region.filled_area:
            region_max = region
    return region_max


def get_minor_major_ratio(region_prop):
    # guard against cases where the segmentation fails by providing zeros
    ratio = region_prop.minor_axis_length*1.0 / region_prop.major_axis_length \
        if region_prop and region_prop.major_axis_length else 0.0
    return ratio


def gray_to_bw(image, method='mean', locality_radius=15):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    # todo: consider the use of Otsu's method
    im_thresh = None
    if method == 'mean':
        im_thresh = np.where(image > np.mean(image), 0., 1.0)
    elif method == 'global_otsu':
        threshold_global_otsu = threshold_otsu(image)
        im_thresh = image >= threshold_global_otsu
    elif method == 'local_otsu':
        locality_radius = 15
        strel = morphology.disk(locality_radius)
        local_otsu = rank.otsu(image, strel)
        im_thresh = image >= local_otsu
    else:
        print 'Invalid threshold method'

    return im_thresh


def bw_labels(im_bw):
    #Dilate the image
    imdilated = morphology.dilation(im_bw, np.ones((4, 4)))

    # Create the label list
    # label dilated image so that close regions have the same label
    label_list = measure.label(imdilated)
    label_list = (im_bw*label_list).astype(int)

    return label_list


def feat_potpourri(region_prop):
    props = ['area', 'convex_area', 'eccentricity', 'extent', 'filled_area',
             'inertia_tensor', 'inertia_tensor_eigvals', 'orientation', 'perimeter', 'solidity']
    if not region_prop:
        return np.zeros(14)
    feats_many = np.concatenate(
        [np.array(eval('region_prop.' + prop)).flatten() for prop in props])
    return feats_many


if __name__ == '__main__':
    curdir, _ = os.path.split(__file__)
    f_dir = os.path.join(curdir, '../data/')
    # f_name = '55511.jpg'
    f_name = '116800.jpg'
    f_path = os.path.join(f_dir, f_name)
    im = imread(f_path)
    im_bw = gray_to_bw(im)
    labels = bw_labels(im_bw)
    region_max = get_max_region(labels, im_bw)
    mmr = get_minor_major_ratio(region_max)
    print mmr
    feats_many = feat_potpourri(region_max)
    print feats_many

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im, cmap='gray', interpolation='none')
    ax2.imshow(region_max.image, cmap='gray', interpolation='none')
    ax1.set_title(f_path)
