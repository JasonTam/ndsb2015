import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import lmdb
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
    db = lmdb.open(db_path, readonly=True)
    with db.begin() as txn:
        cursor = txn.cursor()
        cursor.set_range(start_key)
        data = [(k.split('_', 1)[1], bs_to_im(v), bs_to_l(v)) for ii, (k, v) in enumerate(cursor) if ii < n]
       
        cursor.set_range(start_key)
        for ii in range(n):
            cursor.next()
            if not cursor:
                continue
        next_key = cursor.key()
    return data, next_key







