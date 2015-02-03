import tools.le
import os
import numpy as np


def make_submission(test_files, prediction, f_name='submission.csv'):
    curdir, _ = os.path.split(__file__)

    im_names = np.array([os.path.basename(f) for f in test_files])
    names = tools.le.le.inverse_transform(range(prediction.shape[1]))
    header = np.r_[['image'], names]
    preds_arr = np.c_[im_names, prediction]
    save_arr = np.r_[header[None, :], preds_arr]


    savedir = '../submissions/'
    f_path = os.path.join(curdir, savedir, f_name)
    np.savetxt(f_path, save_arr, fmt='%s', delimiter=',')
    return f_path
