import numpy as np
import skimage

PIXELS = 64
# def transform(Xb, yb):
def transform(im_list):
#     Xb, yb = super(DataAugmentationBatchIterator, self).transform(Xb, yb)
    augmentation_params = {
        #'zoom_range': (0.9, 1.1),
        'zoom_range': (0.8, 1.2),
        'rotation_range': (0, 360),
        'shear_range': (0, 30),
        #'translation_range': (-4, 4),
        'translation_range': (-10, 10),
    }

    IMAGE_WIDTH = PIXELS
    IMAGE_HEIGHT = PIXELS

    def fast_warp(img, tf, output_shape=(PIXELS,PIXELS), mode='nearest'):
        """
        This wrapper function is about five times faster than skimage.transform.warp, for our use case.
        """
        #m = tf._matrix
        m = tf.params
        img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')
        #for k in xrange(1):
        #    img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, output_shape=output_shape, mode=mode)
        img_wf = skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode)
        return img_wf

    def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True):
        shift_x = np.random.uniform(*translation_range)
        shift_y = np.random.uniform(*translation_range)
        translation = (shift_x, shift_y)

        # random rotation [0, 360]
        rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

        # random shear [0, 20]
        shear = np.random.uniform(*shear_range)

        # random zoom [0.9, 1.1]
        # zoom = np.random.uniform(*zoom_range)
        log_zoom_range = [np.log(z) for z in zoom_range]
        zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

        ## flip
        if do_flip and (np.random.randint(2) > 0): # flip half of the time
            shear += 180
            rotation += 180
            # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
            # So after that we rotate it another 180 degrees to get just the flip.            

        '''
        print "translation = ", translation
        print "rotation = ", rotation
        print "shear = ",shear
        print "zoom = ",zoom
        print ""
        '''

        return build_augmentation_transform(zoom, rotation, shear, translation)


    center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
    tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

    def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
        tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), 
                                                  rotation=np.deg2rad(rotation), 
                                                  shear=np.deg2rad(shear), 
                                                  translation=translation)
        tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
        return tform

#     tform_augment = random_perturbation_transform(**augmentation_params)
    tform_identity = skimage.transform.AffineTransform()
    tform_ds = skimage.transform.AffineTransform()

#     for i in range(Xb.shape[0]):
#         new = fast_warp(Xb[i][0], tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32')
#         Xb[i,:] = new

    im_out_list = []
    for im in im_list:
        tform_augment = random_perturbation_transform(**augmentation_params) # different xform per image
        im_out_list.append(fast_warp(im, tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32'))

    return im_out_list
