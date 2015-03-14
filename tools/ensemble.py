# import tools.le
import os
import numpy as np
import pandas as pd
from time import time

PRED_FILE = '../sandbox/submissions/SUBMISSION_PL_deep_56000_SVC_fc2.csv'
PRED_FILE2 = '../sandbox/submissions/SUBMISSION_PL_deep_41000_augagg.csv'

f_paths = [PRED_FILE, PRED_FILE2]


def check_head_valid(f_paths):
    heads = set()
    for f_path in f_paths:
        with open(f_path) as f:
            heads.add(hash(f.readline()))
    return len(heads) == 1


def weighted_mean_preds(f_paths, weights=None):
    tic = time()
    if weights is None:
        weights = np.ones(len(f_paths))
    elif len(weights) != len(f_paths):
        return 'Length of weights doesnt math # of files'
    else:
        weights = np.array(weights)
    weights = weights/weights.sum()

    # Check that all the columns are in the same order
    if not check_head_valid(f_paths):
        return 'Files do not have the same headers and ' \
               'I dont want to sort them because something else ' \
               'is probably wrong (the label encoding)'

    probs = np.array([pd.io.parsers.read_csv(
        os.path.abspath(f_path), header=0).sort(
        columns='image').as_matrix()[:, 1:].astype(float)
                      for f_path in f_paths])

    prob_avg = np.average(probs, weights=weights, axis=0)

    with open(iter(f_paths).next()) as f:
        header = np.array(f.readline().split(','))


    im_names = pd.io.parsers.read_csv(
        os.path.abspath(iter(f_paths).next()), header=0).sort(
        columns='image')['image'].as_matrix()
    preds_arr = np.c_[im_names, prob_avg]
    save_arr = np.r_[header[None, :], preds_arr]


    curdir, _ = os.path.split(__file__)
    savedir = '../submissions/'
    f_name = '_'.join(['ensemble'] +
                      [os.path.splitext(os.path.basename(f_path))[0]
                       for f_path in f_paths]) + '.gz'
    save_path = os.path.join(curdir, savedir, f_name)
    np.savetxt(save_path, save_arr, fmt='%s', delimiter=',')

    toc = time() - tic
    print 'Ensemble saved in: %s' % save_path
    print 'Done in %s sec' % str(toc)
    return save_path
