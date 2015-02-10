import numpy as np
import time
import os
import pickle

# Make sure that caffe is on the python path:
caffe_root = '/afs/ee.cooper.edu/user/t/a/tam8/documents/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# DEFN
curdir, _ = os.path.split(__file__)
MODEL_FILE = os.path.join(curdir, '../deploy_lb.prototxt')
PRETRAINED = os.path.join(curdir, '../models/lb65_iter_100000.caffemodel')
MEAN_FILE = os.path.join(curdir, '../data/64x64/ndsb_mean_test.npy')
TEST_FILE = os.path.join(curdir, '../data/test_final.txt')
N = 20000    # batch size for our pythno script to feed to caffe to avoid memory issues

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(MEAN_FILE),
                       raw_scale=255,
                       image_dims=(57, 57),
                       gpu=True)
caffe.set_mode_gpu()
caffe.set_phase_test()


# Loading test paths
test_doc = np.genfromtxt(TEST_FILE,dtype='str')
test_files = test_doc

# Partition test_files for memory sake
def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    #for i in xrange(0, len(l), n):
    #    yield l[i:i+n]
    return [l[i:i+n] for i in xrange(0, len(l), n)]

test_files_chunks = chunks(test_files, N) 
prediction_list = []
for ii, test_chunk in enumerate(test_files_chunks):
    print 'Chunk:', ii
    # Load Images into memory
    print 'Loading images to memory'
    start = time.time()
    #images = [caffe.io.load_image(im_path, color=False) for im_path in test_files]
    images = [caffe.io.load_image(im_path, color=False) for im_path in test_chunk]
    print "Load images done in %.2f s." % (time.time() - start)

    # DO THE MASS PREDICTION
    print 'Starting Mass Prediction'
    start = time.time()
    prediction = net.predict(images)
    print "Prediction done in %.2f s." % (time.time() - start)

    prediction_list.append(prediction)
    images = None

pickle.dump(prediction_list, open('./pred.p', 'wb'))
predictions = np.concatenate(prediction_list)

# Make the submission
print 'Formatting submission'
import tools.submission as sub
sub.make_submission(test_files, predictions, f_name='SUBMISSION_again.csv')
print 'Done'

