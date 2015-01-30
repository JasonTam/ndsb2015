__author__ = 'jason'

import os
import numpy as np
from skimage.io import imread
import feature.improc as improc
from skimage.transform import resize


def im_features(im_path):
    im = imread(im_path, as_grey=True)
    im_bw = improc.gray_to_bw(im)
    labels = improc.bw_labels(im_bw)
    region_max = improc.get_max_region(labels, im_bw)
    axisratio = improc.get_minor_major_ratio(region_max)
    feats_pot = improc.feat_potpourri(region_max)

    # maxPixel = 25
    # imageSize = maxPixel * maxPixel
    # im = resize(im, (maxPixel, maxPixel))

    # Store the rescaled image pixels and the axis ratio
    # im_flat = im.flatten()
    # x = np.concatenate((im_flat, [axisratio]))
    x = np.concatenate((feats_pot, [axisratio]))

    return x


if __name__ == '__main__':
    curdir, _ = os.path.split(__file__)
    f_dir = os.path.join(curdir, '../data/')
    # f_name = '55511.jpg'
    f_name = '116800.jpg'
    f_path = os.path.join(f_dir, f_name)

    features = im_features(f_path)