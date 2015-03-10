import numpy as np
import sys
import os
from PIL import Image
import augment as aug
import matplotlib.pyplot as plt
from time import time
import lmdb
import plyvel
import multiprocessing
import preproc as pp
from le import le
from le import scaler
import taxonomy as tax
from random import shuffle
from multiprocessing import Pool
from functools import partial
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


#ORIG_LMDB_PATH = '/dev/shm/train0_lmdb'
ORIG_LMDB_PATH = '/media/raid_arr/data/ndsb/folds/train0_lmdb'
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
    
    
OUT_SHAPE = (64, 64)


def make_db_entry(im_file, out_shape=OUT_SHAPE, perturb=True):
    y_str = os.path.basename(os.path.dirname(im_file))
    y = le.transform(y_str)
    
    im_w, (f_size, h, w, ext, hu, sol) = pp.get_features(im_file, 
                                                      out_shape=out_shape,
                                                      norm_orientation=True,
                                                      perturb=perturb)
    extra_feats = np.r_[f_size, h, w, ext, hu, sol]
    
    # Fill in Datum
    datum.channels = 1
    datum.height = out_shape[0]
    datum.width = out_shape[1]
    datum.data = im_w.astype('uint8').tobytes()
    datum.label = y

    #float_data = 6;

    #encoded = 7 [default = false];
    datum.label0 = tax.encode_parent(y_str, 0)
    datum.label1 = tax.encode_parent(y_str, 1)
    datum.label2 = tax.encode_parent(y_str, 2)
    datum.label3 = tax.encode_parent(y_str, 3)
    datum.label4 = tax.encode_parent(y_str, 4)
    datum.label5 = tax.encode_parent(y_str, 5)
    datum.label6 = tax.encode_parent(y_str, 6)

    # Normalized Features (Normalize on second pass)
#     datum.orig_space = f_size
#     datum.orig_height = h
#     datum.orig_width = w

#     datum.extent = ext
#     datum.hu1 = hu[0]
#     datum.hu2 = hu[1]
#     datum.hu3 = hu[2]
#     datum.hu4 = hu[3]
#     datum.hu5 = hu[4]
#     datum.hu6 = hu[5]
#     datum.hu7 = hu[6]
#     datum.solidity = sol
    
    scaled_feats = scaler.transform(extra_feats)
    datum.orig_space = scaled_feats[0]
    datum.orig_height = scaled_feats[1]
    datum.orig_width = scaled_feats[2]
    datum.extent = scaled_feats[3]
    datum.hu1 = scaled_feats[4]
    datum.hu2 = scaled_feats[5]
    datum.hu3 = scaled_feats[6]
    datum.hu4 = scaled_feats[7]
    datum.hu5 = scaled_feats[8]
    datum.hu6 = scaled_feats[9]
    datum.hu7 = scaled_feats[10]
    datum.solidity = scaled_feats[11]
    
    k = os.path.basename(im_file)
    v = datum.SerializeToString()
    return k, v


def single_extract(im_files, db_path, backend='lmdb', perturb=True, out_shape=OUT_SHAPE, verbose=False):
    if backend == 'leveldb':
        db = plyvel.DB(db_path, create_if_missing=True)
        wb = db.write_batch()
    elif backend == 'lmdb':
        db = lmdb.open(db_path, map_size=1e12)
        txn = db.begin(write=True)
    tic = time()
    for im_file in im_files:
        k, v = make_db_entry(im_file, out_shape=out_shape, perturb=perturb)
        if backend == 'leveldb':
            wb.put(k, v)
        elif backend == 'lmdb':
            txn.put(k, v)
    if backend == 'leveldb':
        wb.write()
    elif backend == 'lmdb':
        txn.commit()
    db.close()

    toc = time() - tic
    if verbose:
        print 'Exrtaction to db done in', toc



def get_kv(im_file):    
    return make_db_entry(im_file, out_shape=OUT_SHAPE, perturb=True)


def multi_extract(im_files, db_path, backend='lmdb', perturb=True, out_shape=OUT_SHAPE, verbose=False):
  
    tic = time()
    pool = Pool(processes=7)   # process per core
    all_kv = pool.map(get_kv, im_files)  # process data_inputs iterable with pool
    pool.close()
    if backend == 'leveldb':
        db = plyvel.DB(db_path, create_if_missing=True)
        wb = db.write_batch()
    elif backend == 'lmdb':
        db = lmdb.open(db_path, map_size=1e12)
        txn = db.begin(write=True)
        
    for k, v in all_kv:
        if backend == 'leveldb':
            wb.put(k, v)
        elif backend == 'lmdb':
            txn.put(k, v)
    if backend == 'leveldb':
        wb.write()
    elif backend == 'lmdb':
        txn.commit()
        
    db.close()
    toc = time() - tic
    if verbose:
        print 'Multiproc extraction:', toc


"""
class ExtractionTask(object):
    def __init__(self, im_file, dbwriter):
        self.im_file = im_file
        self.dbwriter = dbwriter

    def __call__(self):
        k, v = make_db_entry(im_file, out_shape=out_shape, perturb=perturb)
        dbwriter.put(k, v)
        #if backend == 'leveldb':
        #    wb.put(k, v)
        #elif backend == 'lmdb':
        #    txn.put(k, v)

    def __str__(self):
        return self.im_file

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
#                 print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
#             print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return




def multi_extract(im_files, db_path, backend='leveldb', perturb=True, out_shape=OUT_SHAPE, verbose=False):
    if backend == 'leveldb':
        db = plyvel.DB(db_path, create_if_missing=True)
        wb = db.write_batch()
    elif backend == 'lmdb':
        db = lmdb.open(db_path, map_size=1e12)
        txn = db.begin(write=True)

    tic = time()

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for ii in xrange(num_consumers)]
    for w in consumers:
        w.start()

    # Enqueue jobs
    # todo: consider using `tblib` to make sure we can debug traceback
    num_jobs = len(im_files)
    for im_file in im_files:
        if backend == 'leveldb':
            tasks.put(ExtractionTask(im_file, wb))
        elif backend == 'lmdb':
            tasks.put(ExtractionTask(im_file, txn))
        #tasks.put(ExtractionTask(im_file, txn))

    # Add a poison pill for each consumer
    for ii in xrange(num_consumers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # Combine Results
    #     while num_jobs:
    #         ret = results.get()
    #         num_jobs -= 1  
    if backend == 'leveldb':
        wb.write()
    elif backend == 'lmdb':
        txn.commit()
    db.close()

    toc = time() - tic
    if verbose:
        print 'Exrtaction to db done in', toc
        
        
"""
