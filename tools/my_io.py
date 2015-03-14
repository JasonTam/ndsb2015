import numpy as np
import sys
import os
from PIL import Image
import augment as aug
#import matplotlib.pyplot as plt
from time import time
import lmdb
import plyvel
import multiprocessing
import preproc as pp
from le import le
from le import scaler
import taxonomy as tax
from collections import OrderedDict
from random import shuffle
from multiprocessing import Pool
from functools import partial
from caffe.proto import caffe_pb2

datum = caffe_pb2.Datum()

###########################################################
# TODO: I should really move this somewhere else - oh well
specialists_d = OrderedDict([
    ('chaetognath', [
        'chaetognath_non_sagitta', 
        'chaetognath_other',
        'chaetognath_sagitta']),
    ('copepod', [
        'copepod_calanoid',
        'copepod_calanoid_eggs',
        'copepod_calanoid_eucalanus',
        'copepod_calanoid_flatheads',
        'copepod_calanoid_frillyAntennae',
        'copepod_calanoid_large',
        'copepod_calanoid_large_side_antennatucked',
        'copepod_calanoid_octomoms',
        'copepod_calanoid_small_longantennae',
        'copepod_cyclopoid_copilia',
        'copepod_cyclopoid_oithona',
        'copepod_cyclopoid_oithona_eggs',
        'copepod_other']),
    ('tunicate_doliolid', [
        'tunicate_doliolid', 
        'tunicate_doliolid_nurse']),
])
sp_member_d = {}
for p, c in specialists_d.items():
    for m in c:
        sp_member_d[m] = p
###########################################################
        

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
        #~ data = [(k.split('_', 1)[1], bs_to_im(v), bs_to_l(v)) for k, v in cursor]
        data = [(k, bs_to_im(v), bs_to_l(v)) for k, v in cursor]
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
        #data = [(k.split('_', 1)[1], bs_to_im(v), bs_to_l(v)) for ii, (k, v) in enumerate(cursor) if ii < n]
        data = [(k, bs_to_im(v), bs_to_l(v)) for ii, (k, v) in enumerate(cursor) if ii < n]
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


def make_db_entry(im_file, out_shape=OUT_SHAPE, perturb=True, mode='train'):
    
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
    
    if mode == 'train':
        # Setup up ground truth labels if training data
        y_str = os.path.basename(os.path.dirname(im_file))
        y = le.transform(y_str)
        datum.label = y

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


# This is some BS
# Train perturb 
def get_kv_peturb(im_file):
    return make_db_entry(im_file, out_shape=OUT_SHAPE, perturb=True, mode='train')

# Train no perturb 
def get_kv_nopeturb(im_file):
    return make_db_entry(im_file, out_shape=OUT_SHAPE, perturb=False, mode='train')

# Test (perturb)
def get_kv_test(im_file):
    return make_db_entry(im_file, out_shape=OUT_SHAPE, perturb=False, mode='test')

def multi_extract(im_files, db_path, backend='lmdb', perturb=True, out_shape=OUT_SHAPE, 
                  transfer_feats=True,
                  transfer_lbls=True,
                  create_specialists=True,
                  mode='train', verbose=False):
    tic = time()
    pool = Pool(processes=7)   # process per core
    if mode == 'test':
        all_kv = pool.map(get_kv_test, im_files)
    elif perturb:   # training and peturb
        all_kv = pool.map(get_kv_peturb, im_files)  # process data_inputs iterable with pool
    else:     # training and no peturb
        all_kv = pool.map(get_kv_nopeturb, im_files)  # process data_inputs iterable with pool
        
        
        
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
        
    if mode == 'train' and create_specialists:
        if verbose:
            print 'Creating specialist db'
        make_specialist_db(db_path, backend=backend, verbose=verbose)
              
    if transfer_feats:
        if verbose:
            print 'Transfering feats to another db'
        feats_db = db_path + '_feats'
        transfer_feats_db(db_path, feats_db, 
                          backend=backend, verbose=verbose)
    if transfer_lbls:
        if verbose:
            print 'Transfering parent labels to another db'
        feats_db = db_path + '_lbls'
        transfer_parentlbls_db(db_path, feats_db, 
                          backend=backend, verbose=verbose)



def transfer_feats_db(core_db, feats_db, backend='lmdb', verbose=False):
    """
    Transfer features to a separate db
    """
    if backend == 'leveldb':
        c = plyvel.DB(core_db)
        db_feats = plyvel.DB(feats_db, create_if_missing=True)
        txn_feats = db_feats.write_batch()
    elif backend == 'lmdb':   
        db = lmdb.open(core_db)
        db_feats = lmdb.open(feats_db, map_size=1e12)
        txn = db.begin()
        c = txn.cursor()
        txn_feats = db_feats.begin(write=True)

    std_scale = 2.
    tic = time()
    for k, v in c:
        datum = caffe_pb2.Datum()
        datum.ParseFromString(v)
        extra_feats = np.array([
            datum.orig_space,
            datum.orig_height,
            datum.orig_width,
            datum.extent,
            datum.hu1,
            datum.hu2,
            datum.hu3,
            datum.hu4,
            datum.hu5,
            datum.hu6,
            datum.hu7,
            datum.solidity,
        ])[None, None, :]
        datum.channels, datum.height, datum.width = extra_feats.shape
        scale_map = ((extra_feats + std_scale) * 128./std_scale).clip(0, 255).astype('uint8') 
        datum.data = scale_map.tobytes()
    #     datum.float_data.extend(extra_feats.flat)
        v_feats = datum.SerializeToString()
        
        txn_feats.put(k, v_feats)

    if backend == 'leveldb':
        txn_feats.write()
        c.close()
        db_feats.close()
    elif backend == 'lmdb':
        txn_feats.commit()
        db.close()
        db_feats.close()
    
    if verbose:
        print 'Feat transfer done:', time() - tic
        
        
def transfer_parentlbls_db(core_db, feats_db, backend='lmdb', verbose=False):
    """
    Transfer parent labels to a separate db
    """
    if backend == 'leveldb':
        c = plyvel.DB(core_db)
        db_feats = plyvel.DB(feats_db, create_if_missing=True)
        txn_feats = db_feats.write_batch()
    elif backend == 'lmdb':   
        db = lmdb.open(core_db)
        db_feats = lmdb.open(feats_db, map_size=1e12)
        txn = db.begin()
        c = txn.cursor()
        txn_feats = db_feats.begin(write=True)

    tic = time()
    for k, v in c:
        datum = caffe_pb2.Datum()
        datum.ParseFromString(v)
        extra_lbls = np.array([
            datum.label0,
            datum.label1,
            datum.label2,
            datum.label3,
            datum.label4,
            datum.label5,
            datum.label6,
        ])[:, None, None]
        datum.channels, datum.height, datum.width = extra_lbls.shape
        datum.data = extra_lbls.astype('uint8').tobytes()
        v_lbls = datum.SerializeToString()
        
        txn_feats.put(k, v_lbls)

    if backend == 'leveldb':
        txn_feats.write()
        c.close()
        db_feats.close()
    elif backend == 'lmdb':
        txn_feats.commit()
        db.close()
        db_feats.close()             
        
    if verbose:
        print 'Parent labels transfer done:', time() - tic
        
        
def make_specialist_db(core_db, backend='lmdb', verbose=False):
    """
    Make specialist db
    """
    if backend == 'leveldb':
        c = plyvel.DB(core_db)
    elif backend == 'lmdb':   
        db = lmdb.open(core_db)
        txn_core = db.begin()
        c = txn_core.cursor()

    txn_sp_d = {}   # dictionary of db transactions for each specialist
    db_sp_d = {}
    for sp_name in specialists_d.keys():        
        sp_db = core_db + '_' + sp_name
        if backend == 'leveldb':
            db_sp = plyvel.DB(sp_db, create_if_missing=True)
            txn_sp = db_sp.write_batch()
        elif backend == 'lmdb':   
            db_sp = lmdb.open(sp_db, map_size=1e12)
            txn_sp = db_sp.begin(write=True)
        db_sp_d[sp_name] = db_sp
        txn_sp_d[sp_name] = txn_sp
            
    # Move entries in core to appropriate specialist db
    tic = time()
    for k, v in c:
		datum = caffe_pb2.Datum()
		datum.ParseFromString(v)
		l = datum.label
		l_str = le.inverse_transform(l)
		if l_str in sp_member_d.keys():
			parent = sp_member_d[l_str]
			# Ghetto string encode
			datum.label = specialists_d[parent].index(l_str)
			print parent, l_str, datum.label
			v_sp = datum.SerializeToString()
			txn_sp_d[parent].put(k, v_sp)

    # Write/commit and close
    if backend == 'leveldb':
        for txn in txn_sp_d.values():
            txn.write()
        for db_sp in db_sp_d.values():
            db_sp.close()
        c.close()
        
    elif backend == 'lmdb':
        for txn in txn_sp_d.values():
            txn.commit()
        for db_sp in db_sp_d.values():
            db_sp.close()
        db.close()
        db_sp.close()
    
    if verbose:
        print 'Specialist creation done:', time() - tic
        
        
