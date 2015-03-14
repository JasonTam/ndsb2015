import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import time
import skimage
import skimage.transform
from skimage import measure
from skimage import morphology
#from skimage.filters import threshold_otsu
import os
import cPickle

OUT_SHAPE = (64, 64)
imshow = lambda im: plt.imshow(im, cmap='gray', interpolation='none')

try:
    feats_d = cPickle.load(open('/media/raid_arr/data/ndsb/region_feats.p', 'rb'))
except IOError:
    feats_d = None


def get_largest_region(im, show_plots=False):
    # find the largest nonzero region
    imthr = np.array(im)
    thresh_fn = lambda x: x > np.mean(x)
#     thresh_fn = lambda x: x > threshold_otsu(x)*.9
    imthr = np.where(thresh_fn(im),1.,0.)
    imdilated = morphology.dilation(imthr, np.ones((4,4)))
    labels = measure.label(imdilated)
    labels = imthr * labels
    labels = labels.astype(int)
    
    if show_plots:
        f = plt.figure(figsize=(12,3))
        sub1 = plt.subplot(1,4,1)
        imshow(im)
        sub2 = plt.subplot(1,4,2)
        imshow(imthr)
        sub3 = plt.subplot(1, 4, 3)
        imshow(imdilated)
        sub4 = plt.subplot(1, 4, 4)
        plt.imshow(labels)
        sub1.set_title("Original Image")
        sub2.set_title("Thresholded Image")
        sub3.set_title("Dilated Image")
        sub4.set_title("Labeled Image")
    
    regions = measure.regionprops(labels)
    regionmaxprop = None
    for regionprop in regions:
        # check to see if the region is at least 50% nonzero
        if sum(imthr[labels == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop, labels


""" This is all just modified code from Sanders"""
def get_normal_perturb_tform(zoom_musig=(1, 0.02),
                         rotation_musig=(0, 45),
                         shear_musig=(0, 1),
                         translation_musig=(0, 1), 
                         do_flip=False,
                         verbose=False):
    """
    Since our images going to be normalized already,
    we don't want to deviate too much
    perturb given params picked from normal distribution
        - provide (mu, sigma)
    (rotation params should be given in degrees)
    # todo: may want to consider drawing from dists with finite support
    """    
    shift_x = np.random.normal(*translation_musig)
    shift_y = np.random.normal(*translation_musig)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = np.random.normal(*rotation_musig)

    # random shear [0, 5]
    shear = np.random.normal(*shear_musig)

    # # flip
    if do_flip and (np.random.randint(2) > 0): # flip half of the time
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.
    if np.random.randint(2) > 0:
        rotation += 180    # also, do a random 180
        

    # random zoom [0.9, 1.1]
    zoom_x = np.random.normal(*zoom_musig)
    zoom_y = np.random.normal(*zoom_musig)
    
    tform = skimage.transform.AffineTransform(scale=(1/zoom_x, 1/zoom_y),
                                              rotation=np.deg2rad(rotation),
                                              shear=np.deg2rad(shear),
                                              translation=translation)
    if verbose:
		print 'Perturb Params:'
		print 'Scale:', (1/zoom_x, 1/zoom_y)
		print 'Rotation:', rotation
		print 'Shear:', shear
		print 'Translation:', translation
    return tform
  
    
def get_normalizing_tform(rotation, zoom):
    tform = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), 
                                              rotation=np.deg2rad(rotation), 
                                              shear=0, 
                                              translation=0)
    return tform
    
    
def fast_warp(img, tf, output_shape=OUT_SHAPE, mode='constant'):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    m = tf.params
    img_wf = skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode)
    
    return img_wf



def get_rp_features(rp):
    if rp:
        ext = rp.extent
        hu = rp.moments_hu
        sol = rp.solidity
        bbox = rp.bbox
        theta = rp.orientation
        return ext, hu, sol, bbox, theta
    else:
        return None



def get_features(im_file, out_shape=OUT_SHAPE, norm_orientation=True, perturb=True,
                 verbose=False, show_plots=False, **kwargs):
    
    im = 255 - cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)   # subject is white, bckg is black
    regionmax, labels = get_largest_region(im, show_plots=show_plots)
    
    h, w = sorted(im.shape)
    f_size = os.stat(im_file).st_size
    if regionmax:

        #ext = regionmax.extent
        #hu = regionmax.moments_hu
        #sol = regionmax.solidity
        #bbox = regionmax.bbox
        #theta = regionmax.orientation
        if feats_d:
            im_name = os.path.basename(im_file)
            if im_name in feats_d.keys():
                ext, hu, sol, bbox, theta = feats_d[im_name]
            else:
                ext, hu, sol, bbox, theta = get_rp_features(regionmax)
        else:
            ext, hu, sol, bbox, theta = get_rp_features(regionmax)


        if show_plots:
            # Can only show this plot if the featues are calculated on the spot rather than loaded
            bw = np.where(labels == regionmax.label,True,False)
            plt.figure()
            imshow(bw)

        center_in = np.array(bbox).reshape((2, 2)).mean(axis=0)[::-1]
        center_out = np.array(out_shape) / 2. - 0.5

        tf_cent = skimage.transform.SimilarityTransform(translation=-center_out)
        tf_uncent = skimage.transform.SimilarityTransform(translation=center_in)

        if norm_orientation:
            max_side = np.diff(np.array(bbox).reshape((2, 2)), axis=0).max()
            alpha = out_shape[0] / float(max_side)

            if verbose:
                print 'Rotate:', np.rad2deg(theta)
                print 'Zoom:', alpha

            tform_norm = get_normalizing_tform(-np.rad2deg(theta), alpha)
        else:
            tform_norm = skimage.transform.SimilarityTransform()
            
        if perturb:
            tform_perturb = get_normal_perturb_tform(verbose=verbose, **kwargs)
        else:
            tform_perturb = skimage.transform.SimilarityTransform()
            
        tform = tf_cent + tform_norm + tform_perturb + tf_uncent

        im_w = fast_warp(img=im, tf=tform)

        if show_plots:
            plt.figure()
            imshow(im_w)
        
    else:
        # Could not find a valid region
        # Just scale the image to the appropriate size
        # and return empty features
        im_w = skimage.transform.resize(im, out_shape, mode='constant')
        ext, hu, sol = 0, np.zeros(7), 0

    return (im_w, (f_size, h, w, ext, hu, sol))

