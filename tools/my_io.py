import numpy as np
import sys
from PIL import Image
import augment as aug
import matplotlib.pyplot as plt
from time import time
import lmdb
import plyvel
from random import shuffle
from caffe.proto import caffe_pb2

datum = caffe_pb2.Datum()

def bs_to_l(bs):
    """
    Grabs the label from the serialized binary
    """
    datum.ParseFromString(bs)
    return datum.label

def bs_to_im(bs, dtype=np.float32):
    """
    Converts serialized binary str from db into np.array image
    """
    datum.ParseFromString(bs)
    image_dims = (datum.height, datum.width)
    im = np.array(Image.frombytes('L', image_dims, datum.data))[:, :, None].astype(dtype)
    return im


def load_lmdb(db_path):
    """
    Grabs all entries in the database
    Returns a list of entries
    Each entry contained in a tuple
        (image_path, image as np.array, label)
    """
    db = lmdb.open(db_path, readonly=True)
    with db.begin() as txn:
        cursor = txn.cursor()
        data = [(k.split('_', 1)[1], bs_to_im(v), bs_to_l(v)) for k, v in cursor]
    return data

def load_lmdb_chunk(db_path, start_key='', n=None):
    """
    :param start: key to start at (excluding that key)
    :param n: number of entries to retrieve after the start key
    """
    if n < 0:
        n = sys.maxint
    db = lmdb.open(db_path, readonly=True)
    with db.begin() as txn:
        cursor = txn.cursor()
        cursor.set_range(start_key)
        data = [(k.split('_', 1)[1], bs_to_im(v), bs_to_l(v)) for ii, (k, v) in enumerate(cursor) if ii < n]
        if n != sys.maxint:
            cursor.set_range(start_key)
            for ii in range(n):
                cursor.next()
                if not cursor:
                    continue
            next_key = cursor.key()
        else:
            next_key = ''
    return data, next_key


def lmdb_to_lvl():
    pass


ORIG_LMDB_PATH = '/dev/shm/train0_lmdb'
AUG_LMDB_PATH = '/dev/shm/train0_aug_lmdb'

def create_aug_lmdb(orig_db=ORIG_LMDB_PATH, aug_db=AUG_LMDB_PATH, verbose=True):
    tic = time()
    db = lmdb.open(orig_db)
    new_db = lmdb.open(aug_db, map_size=1e12, metasync=False)

    with db.begin() as txn:
        with new_db.begin(write=True) as txn2:
            cursor1 = txn.cursor()
            for ii, (k, v) in enumerate(cursor1):
                datum.ParseFromString(v)  # not needed if the global `datum` is modified in fn elsewhere
                im = bs_to_im(v)
                new_im = aug.transform([im.squeeze()])[0]
                datum.data = new_im.astype('uint8').tobytes()
        #         datum.label = datum.label
                new_bs = datum.SerializeToString()
                txn2.replace(k, new_bs)
                if ii % 5000 == 0:
                    if verbose:
                        print 'Processed:', ii
    toc = time() - tic
    if verbose:
        print 'Conversion of images from %s to %s took %s sec' % (orig_db, aug_db, toc)  
    
    new_db.sync()


#ORIG_LVL_PATH = '/dev/shm/train0_lvl'
#AUG_LVL_PATH = '/dev/shm/train0_aug_lvl'
AUG_LVL_PATH = '/media/raid_arr/tmp/train0_aug_lvl'

def create_aug_lvl(orig_db=ORIG_LMDB_PATH, aug_db=AUG_LVL_PATH, shuffle_keys=True, verbose=True):
    tic = time()
    db = lmdb.open(orig_db)
    new_db = plyvel.DB(aug_db, create_if_missing=True)

    with db.begin() as txn:
        cursor1 = txn.cursor()
        keys_all = [k for k, _ in cursor1]
        if shuffle_keys:
            shuffle(keys_all)
        for ii, k in enumerate(keys_all):
        #for ii, (k, v) in enumerate(cursor1):
            v = txn.get(k)
            datum.ParseFromString(v)  # not needed if the global `datum` is modified in fn elsewhere
            im = bs_to_im(v)
            new_im = aug.transform([im.squeeze()])[0]
            datum.data = new_im.astype('uint8').tobytes()
    #         datum.label = datum.label
            new_bs = datum.SerializeToString()
            new_db.put(k, new_bs)
            if ii % 5000 == 0:
                if verbose:
                    print 'Processed:', ii
    toc = time() - tic
    if verbose:
        print 'Conversion of images from %s to %s took %s sec' % (orig_db, aug_db, toc)

    new_db.close()
    sys.stdout.flush()
