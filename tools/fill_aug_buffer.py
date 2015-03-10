import os
import subprocess as sub
import sys
import time
import numpy as np
import my_io

BUFF_SIZE = 20
BUFFER_PATH = '/media/raid_arr/tmp/aug_buffer/'
TRAIN_FOLD_TXT = '/media/raid_arr/data/ndsb/folds/train0.txt'
CHECK_INTERVAL = 5


def add_to_buffer(im_paths, buffer_dir, verbose=False):
    db_name = str(int(time.time()))
    db_out_path = os.path.join(buffer_dir, db_name)
    np.random.shuffle(train_fold_paths)
    my_io.multi_extract(train_fold_paths, db_out_path, backend='leveldb',
                    perturb=True, verbose=True)
    if verbose:
        print 'Created db:', db_out_path

def act_buffer(im_paths, buffer_dir=BUFFER_PATH, buff_size=BUFF_SIZE, check_int=CHECK_INTERVAL, verbose=False):
    while True:
        if len(os.walk(BUFFER_PATH).next()[1]) < buff_size:
            add_to_buffer(im_paths, buffer_dir, verbose=verbose)
        else:
            if verbose:
                print 'Waiting', check_int
            time.sleep(check_int)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train_fold_paths = np.loadtxt(TRAIN_FOLD_TXT, delimiter='\t', dtype=str)[:, 0]
    act_buffer(train_fold_paths, 
               buffer_dir=BUFFER_PATH, 
               buff_size=BUFF_SIZE, 
               check_int=CHECK_INTERVAL)

